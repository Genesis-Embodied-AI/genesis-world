import os
import xml.etree.ElementTree as ET
from bisect import bisect_right
from dataclasses import replace
from itertools import chain
from pathlib import Path

# Note the importing mujoco with env var `MUJOCO_GL=EGL` forcibly defines `PYOPENGL_PLATFORM=egl`
import mujoco

import numpy as np
import trimesh
from trimesh.visual.texture import TextureVisuals
from PIL import Image

import genesis as gs
from genesis.ext import urdfpy
from genesis.utils.description import EqualityDescription, GeomDescription, JointDescription, LinkDescription

from . import geom as gu
from . import urdf as uu
from .collision import solve_contype_conaffinity
from .misc import get_assets_dir, redirect_libc_stderr


MIN_TIMECONST = np.finfo(np.double).eps


def get_model_name(file_path):
    """
    Extract the model name from an MJCF file, if specified.

    The name is extracted from the optional ``<mujoco model="...">`` attribute.

    Reference: https://mujoco.readthedocs.io/en/stable/XMLreference.html#mujoco

    Parameters
    ----------
    file_path : str or Path
        Path to the MJCF file.

    Returns
    -------
    str or None
        The model name, or None if not specified.

    Raises
    ------
    ET.ParseError
        If the file cannot be parsed as XML.
    FileNotFoundError
        If the file does not exist.
    OSError
        If there is an error reading the file.
    """
    try:
        # Inline XML content parses directly; a file path does not and falls back to reading from disk.
        root = ET.fromstring(file_path)
    except ET.ParseError:
        root = ET.parse(os.path.join(get_assets_dir(), file_path)).getroot()
    if root.tag == "mujoco":
        return root.attrib.get("model")
    return None


def build_model(
    xml,
    discard_visual,
    default_armature=None,
    merge_fixed_links=False,
    exclude_ground_plane=False,
    links_to_keep=(),
):
    if isinstance(xml, (str, Path, urdfpy.URDF)):
        if isinstance(xml, urdfpy.URDF):
            is_urdf_file = True
            asset_path = get_assets_dir()
            root = xml._unparse(asset_path)
            mjcf = ET.SubElement(root, "mujoco")
        else:
            # Make sure that it is pointing to a valid XML content (either file path or string)
            path = os.path.join(get_assets_dir(), xml)
            is_valid_path = False
            try:
                if os.path.exists(path):
                    xml = ET.parse(path)
                    is_valid_path = True
                else:
                    xml = ET.fromstring(xml)
            except ET.ParseError:
                gs.raise_exception_from(f"'{xml}' is not a valid XML file path or string.")

            # Best guess for the search path
            asset_path = os.path.dirname(path) if is_valid_path else os.getcwd()

            # Detect whether it is a URDF file or a Mujoco MJCF file. `ET.parse` yields an ElementTree, while
            # `ET.fromstring` (inline XML content) yields the root Element directly.
            root = xml.getroot() if isinstance(xml, ET.ElementTree) else xml
            is_urdf_file = root.tag == "robot"
            mjcf = ET.SubElement(root, "mujoco") if is_urdf_file else root

        # Parse all included sub-models recursively
        root_parent_stack = [(mjcf, Path(""))]
        while root_parent_stack:
            xml_root, parent_path = root_parent_stack.pop()
            for elem in tuple(xml_root.findall("include")):
                include_path = parent_path / elem.attrib["file"]
                include_root = ET.parse(Path(asset_path) / include_path).getroot()
                # Mesh file paths in the included file are relative to that file's directory, whereas meshdir
                # points at the top-level model, so rewrite them to stay valid once inlined. Only <asset> meshes
                # reference a file; a <mesh> under <default> is a default class setting attributes (e.g.
                # maxhullvert) and a vertex mesh under <asset> is procedural, both carrying no file to rewrite.
                for include_elem in include_root.findall(".//asset/mesh[@file]"):
                    include_elem.attrib["file"] = str(include_path.parent / include_elem.attrib["file"])
                for child in include_root:
                    mjcf.append(child)
                mjcf.remove(elem)
                root_parent_stack.append((include_root, include_path))

        # Drop ground planes authored directly under the worldbody so a model that embeds its own floor can be
        # loaded into a scene that already provides a ground. Removing the source geoms before compilation leaves
        # planes authored under child bodies untouched, even when the compiler fuses them into the worldbody.
        if not is_urdf_file and exclude_ground_plane:
            for worldbody in mjcf.findall("worldbody"):
                for geom in tuple(worldbody.findall("geom")):
                    if geom.attrib.get("type") == "plane":
                        worldbody.remove(geom)

        # Make sure compiler options are defined
        compiler = mjcf.find("compiler")
        if compiler is None:
            compiler = ET.SubElement(mjcf, "compiler")

        # Set absolute asset search directory
        for name in ("assetdir", "meshdir", "texturedir"):
            compiler.attrib[name] = str(Path(asset_path) / compiler.attrib.get(name, ""))

        # Set default constraint solver time constant and motor armature.
        # Note that these default options are ignored when parsing URDF files.
        default = mjcf.find("default")
        if default is None:
            default = ET.SubElement(mjcf, "default")
        for group_name, params_name in (
            ("geom", ("solref",)),
            ("joint", ("solreflimit", "solreffriction")),
            ("equality", ("solref",)),
        ):
            group = default.find(group_name)
            if group is None:
                group = ET.SubElement(default, group_name)
            for param_name in params_name:
                # 0.0 cannot be used because it is considered as an error, so that it will fallback to the original
                # default value...
                group.attrib.setdefault(param_name, str(MIN_TIMECONST))
        if default_armature is not None:
            # The default rotor armature only fills in joints whose armature is authored neither on the element nor
            # anywhere in their default class chain, so the values authored in the model file are always preserved.
            # First scan the nested default classes: a class authors armature if itself or any ancestor class sets it.
            has_armature_by_class = {}
            default_stack = [(elem, False) for elem in mjcf.findall("default")]
            while default_stack:
                default_elem, has_armature = default_stack.pop()
                joint_elem = default_elem.find("joint")
                has_armature |= joint_elem is not None and "armature" in joint_elem.attrib
                has_armature_by_class[default_elem.attrib.get("class", "main")] = has_armature
                default_stack.extend((child, has_armature) for child in default_elem.findall("default"))
            # Then walk the kinematic tree while tracking the childclass in effect to resolve each joint's class.
            # Bodies may be nested under grouping meta-elements (frame, replicate) at any depth, and composite
            # elements hold joint configuration subelements that take armature like regular joints.
            for worldbody in mjcf.findall("worldbody"):
                body_stack = [
                    (elem, "main")
                    for tag in ("body", "frame", "replicate", "composite")
                    for elem in worldbody.findall(tag)
                ]
                while body_stack:
                    body_elem, childclass = body_stack.pop()
                    childclass = body_elem.attrib.get("childclass", childclass)
                    for joint_elem in body_elem.findall("joint"):
                        if joint_elem.attrib.get("type") == "free":
                            continue
                        joint_class = joint_elem.attrib.get("class", childclass)
                        if not has_armature_by_class.get(joint_class, False):
                            joint_elem.attrib.setdefault("armature", str(default_armature))
                    body_stack.extend(
                        (elem, childclass)
                        for tag in ("body", "frame", "replicate", "composite")
                        for elem in body_elem.findall(tag)
                    )

        # Must pre-process URDF to overwrite default Mujoco compile flags
        if is_urdf_file:
            robot = urdfpy.URDF._from_xml(root, root, asset_path)

            # Merge fixed links if requested
            if merge_fixed_links:
                robot = uu.merge_fixed_links(robot, links_to_keep)
                root = robot._to_xml(None, asset_path)
                root.append(mjcf)

            # Enforce some compiler options
            compiler.attrib |= dict(
                fusestatic="false",
                strippath="false",
                inertiafromgeom="false",
                balanceinertia="false",
                discardvisual="true" if discard_visual else "false",
                autolimits="true",
            )

            # Bound mass and inertia if necessary
            if not all(link.inertial is not None for link in robot.links):
                compiler.attrib |= dict(
                    boundmass=str(MIN_TIMECONST),
                    boundinertia=str(MIN_TIMECONST),
                )

            # Resolve relative mesh paths
            for elem in root.findall(".//mesh"):
                mesh_path = elem.get("filename")
                if mesh_path.startswith("package://"):
                    mesh_path = mesh_path[10:]
                # Beware symlinks must NOT be resolved, otherwise it may break the file extension, which is used by
                # Mujoco MJCF parser to determine how to load mesh files.
                elem.set("filename", str(Path(asset_path) / mesh_path))

        with open(os.devnull, "w") as stderr, redirect_libc_stderr(stderr):
            # Parse updated URDF file as a string
            data = ET.tostring(root, encoding="utf8")
            mj = mujoco.MjModel.from_xml_string(data)

            # Special treatment for URDF
            if is_urdf_file:
                # Discard placeholder inertias that were used to avoid parsing failure
                for link in robot.links:
                    if link.inertial is None:
                        body = mj.body(link.name)
                        body.inertia[:] = 0.0
                        body.mass[:] = 0.0
                        body.invweight0[:] = 0.0

                # Set default constraint solver time constant
                mj.jnt_solref[:, 0] = MIN_TIMECONST
                mj.geom_solref[:, 0] = MIN_TIMECONST
                mj.eq_solref[:, 0] = MIN_TIMECONST
    elif isinstance(xml, mujoco.MjModel):
        mj = xml
    else:
        gs.raise_exception(f"'{xml}' is not a valid MJCF or URDF file.")

    return mj


def parse_xml(morph, surface, rigid_options=None):
    # Always merge fixed links unless explicitly asked not to do so
    merge_fixed_links, links_to_keep = False, ()
    if isinstance(morph, (gs.morphs.URDF, gs.morphs.Drone)):
        merge_fixed_links = morph.merge_fixed_links
        links_to_keep = morph.links_to_keep

    # Build model from XML (either URDF or MJCF)
    exclude_ground_plane = isinstance(morph, gs.morphs.MJCF) and morph.exclude_ground_plane
    mj = build_model(
        morph.file,
        not morph.visualization,
        morph.default_armature,
        merge_fixed_links,
        exclude_ground_plane,
        links_to_keep,
    )

    # We have another more informative warning later so we suppress this one
    # gs.logger.warning(f"(MJCF) Approximating tendon by joint actuator for `{j_desc.name}`")
    # if mj.ntendon:
    #     gs.logger.warning("(MJCF) Tendon not supported")

    # Parse all geometries grouped by parent joint (or world)
    links_g_descs = parse_geoms(mj, morph.scale, surface, morph.file)

    # Parse all bodies (links and joints)
    l_descs, links_j_descs = parse_links(mj, morph.scale)

    # Re-order kinematic tree info
    l_descs, links_j_descs, links_g_descs, _ = uu.order_links_depth_first(l_descs, links_j_descs, links_g_descs)

    # Parsing all equality constraints
    eq_descs = parse_equalities(mj, morph.scale)

    # Friction features the model declares but the rigid options leave disabled parse to inert values; warn so their
    # absence in the simulation is no surprise. rigid_options is None on the secondary URDF parse, whose compiled
    # model only carries MuJoCo defaults.
    if rigid_options is not None:
        if mj.opt.cone == mujoco.mjtCone.mjCONE_ELLIPTIC and rigid_options.friction_cone != gs.friction_cone.elliptic:
            gs.logger.warning(
                "(MJCF) The model declares the elliptic friction cone; set 'friction_cone' to "
                "'gs.friction_cone.elliptic' to honor it."
            )
        geoms_max_condim = mj.geom_condim.max() if mj.ngeom else 3
        if geoms_max_condim >= 4 and not rigid_options.enable_torsional_friction:
            gs.logger.warning(
                "(MJCF) The model declares torsional friction (geom condim >= 4); enable "
                "'enable_torsional_friction' to honor the parsed coefficients."
            )
        if geoms_max_condim >= 6 and not rigid_options.enable_rolling_friction:
            gs.logger.warning(
                "(MJCF) The model declares rolling friction (geom condim >= 6); enable "
                "'enable_rolling_friction' to honor the parsed coefficients."
            )

    return l_descs, links_j_descs, links_g_descs, eq_descs


def parse_link(mj, i_l, scale):
    # mj.body
    name_start = mj.name_bodyadr[i_l]
    link_name, *_ = mj.names[name_start:].decode("utf-8").split("\x00")

    l_desc = LinkDescription(
        name=link_name,
        parent_idx=-1 if mj.body_parentid[i_l] == i_l else int(mj.body_parentid[i_l]),
        pos=mj.body_pos[i_l],
        quat=mj.body_quat[i_l],
        root_idx=int(mj.body_rootid[i_l]),
        inertial_pos=mj.body_ipos[i_l],
        inertial_quat=mj.body_iquat[i_l],
        inertia=np.diag(mj.body_inertia[i_l]),
        mass=float(mj.body_mass[i_l]),
        invweight=mj.body_invweight0[i_l],
    )

    jnt_adr = mj.body_jntadr[i_l]
    jnt_num = mj.body_jntnum[i_l]

    j_descs = []
    for i_j in range(jnt_adr, jnt_adr + max(jnt_num, 1)):
        # Parsing joint type
        mj_type = mj.jnt_type[i_j] if i_j != -1 else None
        if mj_type is None:
            gs_type = gs.JOINT_TYPE.FIXED
            n_qs, n_dofs = 0, 0
        elif mj_type == mujoco.mjtJoint.mjJNT_FREE:
            gs_type = gs.JOINT_TYPE.FREE
            n_qs, n_dofs = 7, 6
        elif mj_type == mujoco.mjtJoint.mjJNT_HINGE:
            gs_type = gs.JOINT_TYPE.REVOLUTE
            n_qs, n_dofs = 1, 1
        elif mj_type == mujoco.mjtJoint.mjJNT_SLIDE:
            gs_type = gs.JOINT_TYPE.PRISMATIC
            n_qs, n_dofs = 1, 1
        elif mj_type == mujoco.mjtJoint.mjJNT_BALL:
            gs_type = gs.JOINT_TYPE.SPHERICAL
            n_qs, n_dofs = 4, 3
        else:
            gs.raise_exception(f"Unsupported MJCF joint type: {mj_type}")

        # Parsing joint parameters that are type-agnostic
        mj_dof_offset = mj.jnt_dofadr[i_j] if i_j != -1 else 0
        mj_qpos_offset = mj.jnt_qposadr[i_j] if i_j != -1 else 0
        if i_j == -1:
            joint_name = l_desc.name
            joint_pos = np.array([0.0, 0.0, 0.0])
        else:
            name_start = mj.name_jntadr[i_j]
            joint_name, *_ = mj.names[name_start:].decode("utf-8").split("\x00")
            if not joint_name:
                joint_name = l_desc.name
            joint_pos = mj.jnt_pos[i_j]
        joint_quat = np.array([1.0, 0.0, 0.0, 0.0])
        init_qpos = np.array(mj.qpos0[mj_qpos_offset : (mj_qpos_offset + n_qs)])
        dofs_damping = mj.dof_damping[mj_dof_offset : (mj_dof_offset + n_dofs)]
        dofs_invweight = mj.dof_invweight0[mj_dof_offset : (mj_dof_offset + n_dofs)]
        dofs_armature = mj.dof_armature[mj_dof_offset : (mj_dof_offset + n_dofs)]
        dofs_frictionloss = mj.dof_frictionloss[mj_dof_offset : (mj_dof_offset + n_dofs)]
        if mj.njnt > 0:
            mj_jnt_offset = i_j if i_j != -1 else 0
            sol_params = np.concatenate((mj.jnt_solref[mj_jnt_offset], mj.jnt_solimp[mj_jnt_offset]))
        else:
            sol_params = gu.default_solver_params()  # Placeholder. It will not be used anyway.

        # Parsing joint parameters that are type-specific
        mj_stiffness = mj.jnt_stiffness[i_j] if i_j != -1 else 0.0
        mj_is_limited = mj.jnt_limited[i_j] == 1 if i_j != -1 else False
        if gs_type == gs.JOINT_TYPE.FIXED:
            dofs_motion_ang = np.zeros((0, 3))
            dofs_motion_vel = np.zeros((0, 3))
            dofs_limit = np.zeros((0, 2))
            dofs_stiffness = np.zeros((0))
        elif gs_type == gs.JOINT_TYPE.FREE:
            if mj_stiffness > 0.0:
                gs.raise_exception("(MJCF) Joint stiffness not supported for free joints")

            dofs_motion_ang = np.eye(6, 3, -3)
            dofs_motion_vel = np.eye(6, 3)
            dofs_limit = np.tile([-np.inf, np.inf], (6, 1))
            dofs_stiffness = np.zeros(6)
            init_qpos[:3] *= scale
        elif gs_type == gs.JOINT_TYPE.SPHERICAL:
            if mj_is_limited:
                gs.logger.warning("(MJCF) Joint limit ignored for ball joints")

            dofs_motion_ang = np.eye(3)
            dofs_motion_vel = np.zeros((3, 3))
            dofs_limit = np.tile([-np.inf, np.inf], (3, 1))
            dofs_stiffness = np.full((3,), mj_stiffness)
        else:
            mj_axis = mj.jnt_axis[i_j]
            mj_limit = mj.jnt_range[i_j] if mj_is_limited else np.array([-np.inf, np.inf])

            if gs_type == gs.JOINT_TYPE.REVOLUTE:
                dofs_motion_ang = np.array([mj_axis])
                dofs_motion_vel = np.zeros((1, 3))
                dofs_limit = np.array([mj_limit])
                dofs_stiffness = np.array([mj_stiffness])
            else:  # gs_type == gs.JOINT_TYPE.PRISMATIC:
                dofs_motion_ang = np.zeros((1, 3))
                dofs_motion_vel = np.array([mj_axis])
                dofs_limit = np.array([mj_limit]) * scale
                dofs_stiffness = np.array([mj_stiffness])
                init_qpos *= scale

        j_desc = JointDescription(
            name=joint_name,
            type=gs_type,
            pos=joint_pos,
            quat=joint_quat,
            init_qpos=init_qpos,
            sol_params=sol_params,
            dofs_motion_ang=dofs_motion_ang,
            dofs_motion_vel=dofs_motion_vel,
            dofs_limit=dofs_limit,
            dofs_invweight=dofs_invweight,
            dofs_frictionloss=dofs_frictionloss,
            dofs_stiffness=dofs_stiffness,
            dofs_damping=dofs_damping,
            dofs_armature=dofs_armature,
        )

        # Parsing actuator parameters.
        # MuJoCo general actuator model (gaintype=FIXED, biastype=AFFINE):
        #   force = gainprm[0] * ctrl + biasprm[0] + biasprm[1] * pos + biasprm[2] * vel
        # See: https://mujoco.readthedocs.io/en/stable/XMLreference.html#actuator-general
        i_a = -1
        try:
            actuator_mask_j = (mj.actuator_trnid[:, 0] == i_j) & (mj.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT)
            if actuator_mask_j.any():
                (i_a,) = np.nonzero(actuator_mask_j)[0]
            else:  # No actuator directly attached to the joint via mechanical transmission
                # Special case where all tendon are attached to joint. Very common in practice.
                if (mj.wrap_type == mujoco.mjtWrap.mjWRAP_JOINT).all():
                    if i_j in mj.wrap_objid:
                        (m,) = np.nonzero(mj.wrap_objid == i_j)[0]
                        i_t = bisect_right(np.cumsum(mj.tendon_num), m)
                        actuator_mask_t = (mj.actuator_trnid[:, 0] == i_t) & (
                            mj.actuator_trntype == mujoco.mjtTrn.mjTRN_TENDON
                        )
                        (i_a,) = np.nonzero(actuator_mask_t)[0]
                        gs.logger.warning(f"(MJCF) Approximating tendon by joint actuator for `{j_desc.name}`")
        except ValueError:
            gs.logger.warning(f"(MJCF) Failed to parse actuator for joint `{j_desc.name}`.")

        if i_a >= 0:
            if mj.actuator_dyntype[i_a] != mujoco.mjtDyn.mjDYN_NONE:
                gs.logger.warning("(MJCF) Actuator internal dynamics not supported")
            gaintype = mujoco.mjtGain(mj.actuator_gaintype[i_a])
            if gaintype != mujoco.mjtGain.mjGAIN_FIXED:
                gs.logger.warning(f"(MJCF) Actuator control gain of type '{gaintype}' not supported")
            biastype = mujoco.mjtBias(mj.actuator_biastype[i_a])
            if biastype not in (mujoco.mjtBias.mjBIAS_NONE, mujoco.mjtBias.mjBIAS_AFFINE):
                gs.logger.warning(f"(MJCF) Actuator control bias of type '{biastype}' not supported")
            if n_dofs > 1 and not (mj.actuator_gear[i_a, :n_dofs] == 1.0).all():
                gs.logger.warning("(MJCF) Actuator transmission gear is only supported of 1DoF joints")

            gear = mj.actuator_gear[i_a, 0]

            if biastype == mujoco.mjtBias.mjBIAS_NONE:
                # Direct-drive (motor): force = gainprm[0] * ctrl
                # FORCE mode handles this directly; store gainprm for model consistency.
                j_desc.dofs_act_gain = np.full(
                    (n_dofs,), float(gear * mj.actuator_gainprm[i_a, 0] * scale**3), dtype=gs.np_float
                )
            else:
                # General actuator: force = gainprm[0] * ctrl + biasprm[0] + biasprm[1] * pos + biasprm[2] * vel
                gainprm = mj.actuator_gainprm[i_a]
                biasprm = mj.actuator_biasprm[i_a]
                j_desc.dofs_act_gain = np.full((n_dofs,), float(gear * gainprm[0] * scale**3), dtype=gs.np_float)
                j_desc.dofs_act_bias = np.tile(gear * biasprm[:3] * scale**3, (n_dofs, 1)).astype(gs.np_float)

            if mj.actuator_forcelimited[i_a]:
                j_desc.dofs_force_range = np.tile(mj.actuator_forcerange[i_a], (n_dofs, 1))
            if mj.actuator_ctrllimited[i_a] and biastype == mujoco.mjtBias.mjBIAS_NONE:
                j_desc.dofs_force_range = np.minimum(
                    j_desc.dofs_force_range, np.tile(gear * mj.actuator_ctrlrange[i_a], (n_dofs, 1))
                )
        elif gs_type not in (gs.JOINT_TYPE.FIXED, gs.JOINT_TYPE.FREE):
            gs.logger.debug(f"(MJCF) No actuator found for joint `{j_desc.name}`")

        j_descs.append(j_desc)

    # Applying scale if necessary.
    # Note that the mass matrix of a poly-articulated robot does not scale trivially as it is a copnfiguration-depends
    # mixing of s ** 3 factor for masses and s ** 5 factor for inertia tensors. As a result, it is much simpler to
    # consider invweight indefined, which will trigger recomputation at build time.
    if abs(1.0 - scale) > gs.EPS:
        l_desc.pos *= scale
        l_desc.inertial_pos *= scale
        l_desc.mass *= scale**3
        l_desc.inertia *= scale**5
        l_desc.invweight[:] = -1.0
        for j_desc in j_descs:
            j_desc.pos *= scale
            j_desc.dofs_invweight[:] = -1.0

    return l_desc, j_descs


def parse_links(mj, scale):
    l_descs = []
    j_descs = []

    for i_l in range(mj.nbody):
        l_desc, j_desc = parse_link(mj, i_l, scale)

        l_descs.append(l_desc)
        j_descs.append(j_desc)

    return l_descs, j_descs


def parse_geom(mj, i_g, scale, surface, xml_path):
    mj_geom = mj.geom(i_g)

    geom_size = mj_geom.size
    is_col = mj_geom.contype or mj_geom.conaffinity
    metadata = {}

    # Store geom name in metadata
    name_start = mj.name_geomadr[i_g]
    metadata["name"] = mj.names[name_start : mj.names.find(b"\x00", name_start)].decode("utf-8")

    mj_mat_id = int(mj_geom.matid[0])
    is_2d_texture = False
    has_explicit_texcoords = False
    if mj_mat_id >= 0:
        mj_mat = mj.mat(mj_mat_id)
        tex_id_RGB = mj_mat.texid[mujoco.mjtTextureRole.mjTEXROLE_RGB]
        tex_id_RGBA = mj_mat.texid[mujoco.mjtTextureRole.mjTEXROLE_RGBA]
        tex_id = tex_id_RGB if tex_id_RGB >= 0 else tex_id_RGBA
        if tex_id >= 0:
            mj_tex = mj.tex(tex_id)
            is_2d_texture = mj_tex.type[0] == mujoco.mjtTexture.mjTEXTURE_2D
            H, W, C = mj_tex.height[0], mj_tex.width[0], mj_tex.nchannel[0]
            mj_mat_img = mj.tex_data[mj_tex.adr[0] : (mj_tex.adr[0] + H * W * C)].reshape(H, W, C)
            mj_mat_img = Image.fromarray(mj_mat_img)
        else:
            mj_mat_img = None
        mj_rgba = np.asarray(mj_mat.rgba, dtype=np.float32)
        mj_specular = np.full(3, mj_mat.specular[0], dtype=np.float32)
        mj_glossiness = mj_mat.shininess[0] * 128.0
        tmesh_mat = trimesh.visual.material.SimpleMaterial(
            image=mj_mat_img,
            diffuse=mj_rgba,
            specular=mj_specular,
            glossiness=mj_glossiness,
        )
    else:
        mj_rgba = np.asarray(mj_geom.rgba, dtype=np.float32)
        mj_mat = None
        tmesh_mat = trimesh.visual.material.SimpleMaterial(diffuse=mj_rgba)

    if mj_geom.type == mujoco.mjtGeom.mjGEOM_PLANE:
        length, width, _ = geom_size
        length = length or 1e3
        width = width or 1e3

        uv = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
        mesh_params = dict(
            vertices=np.array(
                [[-length, width, 0.0], [length, width, 0.0], [-length, -width, 0.0], [length, -width, 0.0]],
                dtype=np.float32,
            ),
            faces=np.array([[0, 2, 3], [0, 3, 1]], dtype=np.int64),
            face_normals=np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float32),
        )
        geom_data = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        gs_type = gs.GEOM_TYPE.PLANE

    elif mj_geom.type == mujoco.mjtGeom.mjGEOM_SPHERE:
        radius = geom_size[0]
        if is_col:
            tmesh = trimesh.creation.icosphere(radius=radius, subdivisions=2)
        else:
            tmesh = trimesh.creation.icosphere(radius=radius)
        mesh_params = dict(vertices=tmesh.vertices, faces=tmesh.faces)
        uv = None
        gs_type = gs.GEOM_TYPE.SPHERE
        geom_data = np.array([radius * scale])

    elif mj_geom.type == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        if is_col:
            tmesh = trimesh.creation.icosphere(radius=1.0, subdivisions=2)
        else:
            tmesh = trimesh.creation.icosphere(radius=1.0)
        tmesh.apply_transform(np.diag([*geom_size, 1]))
        mesh_params = dict(vertices=tmesh.vertices, faces=tmesh.faces)
        uv = None
        gs_type = gs.GEOM_TYPE.ELLIPSOID
        geom_data = geom_size * scale

    elif mj_geom.type == mujoco.mjtGeom.mjGEOM_CAPSULE:
        radius = geom_size[0]
        height = geom_size[1] * 2
        if is_col:
            tmesh = trimesh.creation.capsule(radius=radius, height=height, count=(8, 12))
        else:
            tmesh = trimesh.creation.capsule(radius=radius, height=height)
        mesh_params = dict(vertices=tmesh.vertices, faces=tmesh.faces)
        uv = None
        gs_type = gs.GEOM_TYPE.CAPSULE
        geom_data = np.array([radius * scale, height * scale])

    elif mj_geom.type == mujoco.mjtGeom.mjGEOM_CYLINDER:
        radius = geom_size[0]
        height = geom_size[1] * 2
        tmesh = trimesh.creation.cylinder(radius=radius, height=height)
        mesh_params = dict(vertices=tmesh.vertices, faces=tmesh.faces)
        uv = None
        gs_type = gs.GEOM_TYPE.CYLINDER
        geom_data = np.array([radius * scale, height * scale])

    elif mj_geom.type == mujoco.mjtGeom.mjGEOM_BOX:
        tmesh = trimesh.creation.box(extents=geom_size * 2)
        mesh_params = dict(vertices=tmesh.vertices, faces=tmesh.faces)
        uv = tmesh.vertices[:, :2].copy()
        uv -= uv.min(axis=0)
        uv /= uv.max(axis=0)
        gs_type = gs.GEOM_TYPE.BOX
        geom_data = 2 * geom_size * scale

    elif mj_geom.type == mujoco.mjtGeom.mjGEOM_MESH:
        mj_mesh = mj.mesh(mj_geom.dataid[0])

        vert_start = mj_mesh.vertadr[0]
        vert_end = vert_start + mj_mesh.vertnum[0]

        norm_start = int(mj.mesh_normaladr[mj_mesh.id])
        norm_end = norm_start + int(mj.mesh_normalnum[mj_mesh.id])

        face_start = mj_mesh.faceadr[0]
        face_end = face_start + mj_mesh.facenum[0]

        vertices = mj.mesh_vert[vert_start:vert_end]
        normals = mj.mesh_normal[norm_start:norm_end]
        faces = mj.mesh_face[face_start:face_end]
        norm_faces = mj.mesh_facenormal[face_start:face_end]

        tex_vert_start = int(mj.mesh_texcoordadr[mj_mesh.id])
        tex_vert_end = tex_vert_start + int(mj.mesh_texcoordnum[mj_mesh.id])

        # MuJoCo stores vertices, normals and texcoords in independently-addressed blocks, with each face
        # carrying separate vertex/normal/texcoord index triplets. Split shared vertices so that every unique
        # (vertex, normal[, texcoord]) combination becomes a single trimesh vertex.
        index_faces = [faces.ravel(), norm_faces.ravel()]
        if tex_vert_start != -1:  # -1 means no texcoord
            tex_faces = mj.mesh_facetexcoord[face_start:face_end]
            # This slice is a view of MuJoCo-owned data; transformations create new arrays
            uv = mj.mesh_texcoord[tex_vert_start:tex_vert_end]
            has_explicit_texcoords = True
            index_faces.append(tex_faces.ravel())
        else:
            uv = None

        pairs = np.stack(index_faces, axis=1)  # (face_num * 3, 2 or 3)
        uniq, inv = np.unique(pairs, axis=0, return_inverse=True)

        vertices = vertices[uniq[:, 0]]
        normals = normals[uniq[:, 1]]
        if uv is not None:
            uv = uv[uniq[:, 2]]
            uv[:, 1] = 1.0 - uv[:, 1]
        faces = inv.reshape(-1, 3).astype(np.int64)

        mesh_params = dict(vertices=vertices, faces=faces, vertex_normals=normals)
        gs_type = gs.GEOM_TYPE.MESH
        geom_data = None

        mesh_path_start = mj.mesh_pathadr[mj_mesh.id]
        metadata["mesh_path"], *_ = mj.paths[mesh_path_start:].decode("utf-8").split("\x00")
    else:
        gs.logger.warning(f"Unsupported MJCF geom type '{mj_geom.type}'.")
        return None

    if mj_mat is not None:
        if is_2d_texture and not has_explicit_texcoords:
            render_size = geom_size[:2].copy()
            if mj_geom.type[0] in (
                mujoco.mjtGeom.mjGEOM_SPHERE,
                mujoco.mjtGeom.mjGEOM_CAPSULE,
                mujoco.mjtGeom.mjGEOM_CYLINDER,
            ):
                render_size[1] = render_size[0]
            is_size_finite = render_size > 0

            object_xy = mesh_params["vertices"][:, :2].copy()
            if mj_geom.type[0] != mujoco.mjtGeom.mjGEOM_MESH:
                # Normalize finite primitive axes; true meshes and infinite plane axes remain spatial
                np.divide(object_xy, render_size, out=object_xy, where=is_size_finite)

            repeat = mj_mat.texrepeat.copy()
            if mj_geom.dataid[0] >= 0:
                # MuJoCo divides by geom size for every retained mesh dataid, including mesh-fitted primitives
                np.divide(repeat, render_size, out=repeat, where=is_size_finite)
            if mj_mat.texuniform[0]:
                # Spatial repetition includes the scale applied by the Genesis morph
                repeat *= np.where(is_size_finite, render_size, 1.0) * scale

            # MuJoCo's object-linear two-dimensional texture mapping
            #   s =  0.5 * repeat_x * x - 0.5
            #   t = -0.5 * repeat_y * y - 0.5
            uv = np.empty_like(object_xy)
            uv[:, 0] = 0.5 * repeat[0] * object_xy[:, 0] - 0.5
            uv[:, 1] = -0.5 * repeat[1] * object_xy[:, 1] - 0.5

            # Trimesh measures the vertical texture coordinate from the top edge
            uv[:, 1] = 1.0 - uv[:, 1]
        elif uv is not None and not is_2d_texture:
            uv = uv * mj_mat.texrepeat
    tmesh = trimesh.Trimesh(
        **mesh_params,
        visual=TextureVisuals(uv=uv, material=tmesh_mat),
        process=False,
    )
    mesh = gs.Mesh.from_trimesh(
        tmesh, scale=scale, surface=gs.surfaces.Collision() if is_col else surface, metadata=metadata
    )

    return GeomDescription(
        contype=mj_geom.contype[0],
        conaffinity=mj_geom.conaffinity[0],
        type=gs_type,
        data=geom_data,
        pos=mj_geom.pos * scale,
        quat=mj_geom.quat,
        mesh=mesh if is_col else None,
        vmesh=None if is_col else mesh,
        friction=mj_geom.friction[0],
        # MuJoCo only applies torsional friction from condim 4 and rolling friction from condim 6 onward, and the
        # friction vector carries its defaults on every geom regardless, so the coefficients of a lower-condim geom
        # must parse as inert or the geom would resist spin or rolling that MuJoCo leaves free.
        friction_torsional=mj_geom.friction[1] if mj_geom.condim[0] >= 4 else 0.0,
        friction_rolling=mj_geom.friction[2] if mj_geom.condim[0] >= 6 else 0.0,
        group=mj_geom.group[0],
        sol_params=np.concatenate((mj_geom.solref, mj_geom.solimp)),
    )


def parse_geoms(mj, scale, surface, xml_path):
    links_g_descs = [[] for _ in range(mj.nbody)]

    # Loop over all geometries sequentially
    is_any_col = False
    for i_g in range(mj.ngeom):
        if mj.geom_bodyid[i_g] < 0:
            continue

        # try parsing a given geometry
        g_desc = parse_geom(mj, i_g, scale, surface, xml_path)
        if g_desc is None:
            continue

        # Ignore world when looking for collision geometries
        if mj.geom_bodyid[i_g] == 0:
            is_any_col |= g_desc.contype or g_desc.conaffinity

        # assign geoms to link
        link_idx = mj.geom_bodyid[i_g]
        links_g_descs[link_idx].append(g_desc)

    # Update contype and conaffinity to take into account any additional list of explicitly excluded collision pairs
    if mj.nexclude:
        # Extract the list of collision geometries
        cg_descs = []
        for g_desc in chain.from_iterable(links_g_descs):
            if g_desc.contype or g_desc.conaffinity:
                cg_descs.append(g_desc)

        # Compute the original of all the excluded collision pairs
        invalid_set = set()
        for i, g_desc_1 in enumerate(cg_descs):
            for j, g_desc_2 in enumerate(cg_descs):
                if i >= j:
                    continue
                if g_desc_1.contype & g_desc_2.conaffinity:
                    continue
                if g_desc_2.contype & g_desc_1.conaffinity:
                    continue
                invalid_set.add(frozenset((i, j)))

        # Append all the explicitly excluded collision pairs
        for exclude_signature in mj.exclude_signature:
            body_1 = (exclude_signature >> 16) & 0xFFFF
            body_2 = exclude_signature & 0xFFFF

            geoms_1, geoms_2 = [], []
            for body_idx, geoms_idx in ((body_1, geoms_1), (body_2, geoms_2)):
                for g_desc in links_g_descs[body_idx]:
                    for geom_idx, cg_desc in enumerate(cg_descs):
                        if g_desc is cg_desc:
                            geoms_idx.append(geom_idx)
                            break

            for geom_1 in geoms_1:
                for geom_2 in geoms_2:
                    invalid_set.add(frozenset((geom_1, geom_2)))

        # Compute updated contype and conaffinity from the complete list of invalid collision pairs
        masks = solve_contype_conaffinity(len(cg_descs), invalid_set)
        if masks is None:
            gs.logger.warning(
                "Compatible collision geometries cannot be described using bitmasks 'contype' and 'conaffinity'. "
                "Using default values..."
            )
            for g_desc in cg_descs:
                g_desc.contype, g_desc.conaffinity = 1, 1
        else:
            for g_desc, (contype, conaffinity) in zip(cg_descs, masks):
                g_desc.contype, g_desc.conaffinity = contype, conaffinity

    # Inform the user that collision geometries are not displayed by default
    if is_any_col and surface.vis_mode != "collision":
        gs.logger.info(
            "Collision meshes are not visualized by default. To visualize them, please use `vis_mode='collision'` "
            "when calling `scene.add_entity`."
        )

    # Parse geometry group if available
    for link_g_descs in links_g_descs:
        for g_desc in tuple(link_g_descs):
            # Skip visual geometries
            if not (g_desc.contype or g_desc.conaffinity):
                continue

            # Duplicate collision geometries as visual in accordance with Mujoco logics:
            # If groups are defined, only create visual for geoms in visual groups (0, 1 or 2).
            if g_desc.group in (0, 1, 2):
                vmesh = gs.Mesh.from_trimesh(
                    mesh=g_desc.mesh.trimesh,
                    surface=surface,
                    metadata=g_desc.mesh.metadata,
                )
                link_g_descs.append(replace(g_desc, contype=0, conaffinity=0, mesh=None, vmesh=vmesh))

    return links_g_descs


def parse_equalities(mj, scale):
    eq_descs = []
    for i_e in range(mj.neq):
        mj_equality = mj.equality(i_e)

        eq_data = mj.eq_data[i_e]
        sol_params = np.concatenate((mj.eq_solref[i_e], mj.eq_solimp[i_e]))

        objs_idx = [mj.eq_obj1id[i_e], mj.eq_obj2id[i_e]]
        if mj.eq_objtype[i_e] == mujoco.mjtObj.mjOBJ_SITE:
            # Must convert site into relative link position because Genesis does not implement site abstraction
            name_objadr = mj.name_bodyadr
            sites_pos, sites_quat = [], []
            for i, site_idx in enumerate(objs_idx):
                objs_idx[i] = mj.site_bodyid[site_idx]
                sites_pos.append(mj.site_pos[site_idx])
                sites_quat.append(mj.site_quat[site_idx])
            if mj.eq_type[i_e] == mujoco.mjtEq.mjEQ_WELD:
                eq_data[:3], eq_data[3:6] = (sites_pos[1], sites_pos[0])
                eq_data[6:10] = gu.transform_quat_by_quat(sites_quat[0], gu.inv_quat(sites_quat[1]))
            else:
                eq_data[:3], eq_data[3:6] = (sites_pos[0], sites_pos[1])
        elif mj.eq_type[i_e] == mujoco.mjtEq.mjEQ_JOINT:
            name_objadr = mj.name_jntadr
        elif mj.eq_objtype[i_e] == mujoco.mjtObj.mjOBJ_BODY:
            name_objadr = mj.name_bodyadr
        else:
            gs.raise_exception(f"Unsupported MJCF equality object type: {mj.eq_objtype[i_e]}")

        if mj.eq_type[i_e] == mujoco.mjtEq.mjEQ_CONNECT:
            eq_type = gs.EQUALITY_TYPE.CONNECT
            eq_data[:6] *= scale
        elif mj.eq_type[i_e] == mujoco.mjtEq.mjEQ_WELD:
            eq_type = gs.EQUALITY_TYPE.WELD
            eq_data[:6] *= scale
        elif mj.eq_type[i_e] == mujoco.mjtEq.mjEQ_JOINT:
            eq_type = gs.EQUALITY_TYPE.JOINT
            follower_scale = scale if mj.jnt_type[objs_idx[0]] == mujoco.mjtJoint.mjJNT_SLIDE else 1.0
            driver_scale = (
                scale if objs_idx[1] >= 0 and mj.jnt_type[objs_idx[1]] == mujoco.mjtJoint.mjJNT_SLIDE else 1.0
            )

            # eq_data[0:5] stores a0..a4 in q_follower - q_follower0 = sum(a_k * (q_driver - q_driver0)^k).
            # A model scale transforms a_k to s_f * a_k / s_d**k, where s_f and s_d are scale for prismatic
            # coordinates and 1.0 for revolute coordinates.
            eq_data[:5] *= follower_scale / driver_scale ** np.arange(5)
        else:
            gs.raise_exception(f"Unsupported MJCF equality type: {mj.eq_type[i_e]}")

        objs_name = []
        for obj_idx in objs_idx:
            if obj_idx < 0:
                obj_name = None
            else:
                name_start = name_objadr[obj_idx]
                obj_name, *_ = mj.names[name_start:].decode("utf-8").split("\x00")
            objs_name.append(obj_name)
        eq_descs.append(
            EqualityDescription(
                name=mj_equality.name,
                type=eq_type,
                objs_name=tuple(objs_name),
                data=eq_data,
                sol_params=sol_params,
            )
        )

    return eq_descs

import os
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh

import xacro

import genesis as gs
import genesis.utils.gltf as gltf_utils
from genesis.constants import GLTF_FORMATS, XACRO_FORMAT
from genesis.ext import urdfpy

from . import geom as gu
from . import mesh as mu
from .misc import get_assets_dir


def get_robot_name(file_path):
    """
    Extract the robot name from a URDF file.

    The name is extracted from the ``<robot name="...">`` attribute, which is
    required by the URDF specification.

    Reference: https://wiki.ros.org/urdf/XML/robot

    Parameters
    ----------
    file_path : str or Path
        Path to the URDF file, or inline URDF XML content.

    Returns
    -------
    str
        The robot name.

    Raises
    ------
    ValueError
        If the robot name attribute is missing or empty.
    """
    try:
        # Inline XML content parses directly; a file path does not and falls back to reading from disk.
        root = ET.fromstring(file_path)
    except ET.ParseError:
        root = ET.parse(os.path.join(get_assets_dir(), file_path)).getroot()
    if root.tag == "robot":
        name = root.attrib.get("name")
        if name:
            return name
        raise ValueError(f"URDF file '{file_path}' is missing required 'name' attribute on <robot> element.")
    raise ValueError(f"Invalid URDF file '{file_path}'. Missing <robot> root element.")


def load_xacro(path, mappings):
    """Load a XACRO file into a ``urdfpy.URDF`` object with absolute mesh paths.

    Expands all xacro macros, properties, includes, and conditionals in-memory using the ``xacro`` package, then
    parses the resulting URDF XML into a ``urdfpy.URDF`` object. All relative mesh filenames are resolved to absolute
    paths so that downstream consumers do not need access to the original source directory.

    Note
    ----
    ``$(find package_name)`` substitutions require ROS's ``ament_index_python`` package.

    Parameters
    ----------
    path : str
        Absolute path to the ``.xacro`` or ``.urdf.xacro`` file.
    mappings : dict
        Key-value pairs to override ``xacro:arg`` declarations.

    Returns
    -------
    urdfpy.URDF
        The parsed URDF with absolute mesh paths.
    """
    doc = xacro.process_file(path, mappings=dict(mappings))
    node = ET.fromstring(doc.toxml())
    source_dir = os.path.dirname(path)
    robot = urdfpy.URDF._from_xml(node, node, source_dir)
    for link in robot.links:
        for geom_prop in (*link.collisions, *link.visuals):
            if isinstance(geom_prop.geometry.geometry, urdfpy.Mesh):
                geom_prop.geometry.geometry.filename = urdfpy.utils.get_filename(
                    source_dir, geom_prop.geometry.geometry.filename
                )
    return robot


def order_links_depth_first(l_infos, j_infos, links_g_infos=None):
    # Re-order links depth-first so that each subtree occupies a contiguous range and parents precede their children.
    n_links = len(l_infos)
    dict_child = {k: [] for k in range(n_links)}
    for lc in range(n_links):
        if "parent_idx" not in l_infos[lc]:
            l_infos[lc]["parent_idx"] = -1
        lp = l_infos[lc]["parent_idx"]
        if lp != -1:
            dict_child[lp].append(lc)

    # Depth-first pre-order: a parent always precedes its children (required downstream), and each subtree occupies a
    # contiguous index range, so a free body and all of its descendants get contiguous links and DOFs. Siblings keep
    # their original relative order, and the result matches MuJoCo's body ordering. Contiguity lets the per-tree
    # mass-matrix factorization stay block-local instead of spanning the whole (possibly multi-body) entity.
    ordered_links_idx = []
    stack_topology = [lc for lc in reversed(range(n_links)) if l_infos[lc]["parent_idx"] == -1]
    while stack_topology:
        link = stack_topology.pop()
        ordered_links_idx.append(link)
        stack_topology.extend(reversed(dict_child[link]))
    ordered_links_idx = tuple(ordered_links_idx)

    if not ordered_links_idx:
        # avoid case with worldbody without any body (geom directly assigned to worldbody)
        return [], [], [], []

    for l_info in l_infos:
        if l_info["parent_idx"] >= 0:  # non-base link
            l_info["parent_idx"] = ordered_links_idx.index(l_info["parent_idx"])
        if "root_idx" in l_info:
            l_info["root_idx"] = ordered_links_idx.index(l_info["root_idx"])

    new_l_infos = [l_infos[i] for i in ordered_links_idx]
    new_j_infos = [j_infos[i] for i in ordered_links_idx]

    if links_g_infos is not None:
        links_g_infos = [links_g_infos[i] for i in ordered_links_idx]

    return new_l_infos, new_j_infos, links_g_infos, ordered_links_idx


def parse_urdf(morph, surface):
    if morph.is_format(XACRO_FORMAT):
        # Expanded by the parser reading it, so the morph keeps the file provided by the user (see 'parse_xml' in mjcf.py)
        parent_dir = os.getcwd()
        robot = load_xacro(morph.file, morph.xacro_args)
    elif isinstance(morph.file, (str, Path)):
        # Inline XML content parses directly; a file path does not and falls back to reading from disk.
        try:
            node = ET.fromstring(morph.file)
            parent_dir = os.getcwd()
            robot = urdfpy.URDF._from_xml(node, node, parent_dir)
        except (ET.ParseError, TypeError):
            path = os.path.join(get_assets_dir(), morph.file)
            parent_dir = os.path.dirname(path)
            robot = urdfpy.URDF.load(path)
    else:
        parent_dir = os.getcwd()
        # The caller's model is parsed on a copy: the scaling and fixed-link merging below write into it, and the MuJoCo
        # pass parses the same object again (see parse_xml in mjcf.py).
        robot = morph.file.copy()

    # Merge links connected by fixed joints
    if morph.merge_fixed_links:
        robot = merge_fixed_links(robot, morph.links_to_keep)

    link_name_to_idx = dict()
    for idx, link in enumerate(robot.links):
        link_name_to_idx[link.name] = idx

    # Note that each link corresponds to one joint
    n_links = len(robot.links)
    assert n_links == len(robot.joints) + 1
    l_infos = [dict() for _ in range(n_links)]
    links_j_infos = [[] for _ in range(n_links)]
    links_g_infos = [[] for _ in range(n_links)]

    for link, l_info, link_g_infos in zip(robot.links, l_infos, links_g_infos):
        l_info["name"] = link.name

        # No parent by default. It will be overwritten latter on if appropriate.
        l_info["parent_idx"] = -1

        # Neutral pose by default. It will be overwritten latter on if necessary.
        l_info["pos"] = gu.zero_pos()
        l_info["quat"] = gu.identity_quat()

        # we compute urdf's invweight later
        l_info["invweight"] = np.full((2,), fill_value=-1.0)

        l_info["inertial_pos"] = None
        l_info["inertial_quat"] = gu.identity_quat()
        l_info["inertial_i"] = None
        l_info["inertial_mass"] = None
        if link.inertial is not None:
            if link.inertial.origin is not None:
                l_info["inertial_pos"] = link.inertial.origin[:3, 3]
                l_info["inertial_quat"] = gu.R_to_quat(link.inertial.origin[:3, :3])
            if link.inertial.inertia is not None:
                l_info["inertial_i"] = link.inertial.inertia
            if link.inertial.mass is not None:
                l_info["inertial_mass"] = link.inertial.mass

        for geom_prop in (*link.collisions, *link.visuals):
            geometry = geom_prop.geometry.geometry
            geom_is_col = not isinstance(geom_prop, urdfpy.Visual)

            geom_meshes = []
            if isinstance(geometry, urdfpy.Mesh):
                geom_type = gs.GEOM_TYPE.MESH
                geom_data = None

                # One asset may contain multiple meshes (.obj, .glb, ...)
                mesh_path = urdfpy.utils.get_filename(parent_dir, geometry.filename)
                tmeshes = geometry.meshes
                metadatas = [{"mesh_path": mesh_path} for _ in tmeshes]
                if mesh_path.lower().endswith(GLTF_FORMATS):
                    meshes = gltf_utils.parse_mesh_glb(
                        mesh_path, group_by_material=False, scale=None, is_mesh_zup=True, surface=surface
                    )
                    tmeshes, metadatas = zip(*[(mesh.trimesh, mesh.metadata) for mesh in meshes])

                # Compute the absolute scale of the geometry
                scale = float(morph.scale)
                if geometry.scale is not None:
                    scale *= geometry.scale

                is_mesh_zup = morph.file_meshes_are_zup
            else:
                if isinstance(geometry, urdfpy.Box):
                    tmesh = trimesh.creation.box(extents=geometry.size)
                    geom_type = gs.GEOM_TYPE.BOX
                    geom_data = np.array(geometry.size)
                elif isinstance(geometry, urdfpy.Capsule):
                    tmesh = trimesh.creation.capsule(radius=geometry.radius, height=geometry.length)
                    geom_type = gs.GEOM_TYPE.CAPSULE
                    geom_data = np.array([geometry.radius, geometry.length])
                elif isinstance(geometry, urdfpy.Cylinder):
                    tmesh = trimesh.creation.cylinder(radius=geometry.radius, height=geometry.length)
                    geom_type = gs.GEOM_TYPE.CYLINDER
                    geom_data = np.array([geometry.radius, geometry.length])
                elif isinstance(geometry, urdfpy.Sphere):
                    if geom_is_col:
                        tmesh = trimesh.creation.icosphere(radius=geometry.radius, subdivisions=2)
                    else:
                        tmesh = trimesh.creation.icosphere(radius=geometry.radius)
                    geom_type = gs.GEOM_TYPE.SPHERE
                    geom_data = np.array([geometry.radius])
                else:
                    gs.raise_exception(f"Unsupported primitive geometry: {geometry}")

                tmeshes = [tmesh]

                scale = morph.scale
                metadatas = [{}]
                is_mesh_zup = True

            # Each mesh is one RigidGeom in genesis
            for tmesh, metadata in zip(tmeshes, metadatas, strict=True):
                # Overwrite surface color by original color specified in URDF file only if necessary
                is_urdf_material = False
                # trimesh gives a mesh holding texture coordinates and no material its placeholder material, which
                # states nothing about the appearance the asset authored
                has_asset_material = tmesh.visual.defined and not (
                    isinstance(tmesh.visual, trimesh.visual.texture.TextureVisuals)
                    and hash(tmesh.visual.material) == hash(trimesh.visual.material.empty_material())
                )
                if geom_is_col:
                    geom_surface = gs.surfaces.Collision()
                elif (
                    surface.texture is None
                    and geom_prop.material is not None
                    and (geom_prop.material.texture is not None or geom_prop.material.color is not None)
                    and (morph.prioritize_urdf_material or not has_asset_material)
                ):
                    is_urdf_material = True
                    if geom_prop.material.texture is not None:
                        # The image stands in for the color, as it does for the material of a mesh asset (see
                        # from_trimesh in engine/mesh.py)
                        geom_surface = gs.surfaces.Default(
                            diffuse_texture=gs.textures.ImageTexture(
                                image_array=mu.PIL_to_array(geom_prop.material.texture.image)
                            )
                        )
                    else:
                        geom_surface = gs.surfaces.Default(color=geom_prop.material.color)
                else:
                    geom_surface = surface

                mesh = gs.Mesh.from_trimesh(
                    tmesh,
                    scale=scale,
                    surface=geom_surface,
                    is_mesh_zup=is_mesh_zup,
                    metadata=metadata,
                )

                # Material color defined in URDF are not considered as visual overwrite
                if is_urdf_material:
                    mesh.metadata["is_visual_overwritten"] = False

                geom_meshes.append(mesh)

            for mesh in geom_meshes:
                g_info = {
                    "mesh" if geom_is_col else "vmesh": mesh,
                    "type": geom_type,
                    "data": geom_data,
                    "pos": geom_prop.origin[:3, 3].copy(),
                    "quat": gu.R_to_quat(geom_prop.origin[:3, :3]),
                    "contype": 1 if geom_is_col else 0,
                    "conaffinity": 1 if geom_is_col else 0,
                    "friction": gu.default_friction(),
                    "sol_params": gu.default_solver_params(),
                }
                link_g_infos.append(g_info)

    #########################  non-base joints and links #########################
    for joint in robot.joints:
        idx = link_name_to_idx[joint.child]
        l_info = l_infos[idx]
        j_info = dict()
        links_j_infos[idx].append(j_info)

        j_info["name"] = joint.name
        j_info["pos"] = gu.zero_pos()
        j_info["quat"] = gu.identity_quat()

        l_info["parent_idx"] = link_name_to_idx[joint.parent]
        l_info["pos"] = joint.origin[:3, 3]
        l_info["quat"] = gu.R_to_quat(joint.origin[:3, :3])

        if joint.joint_type == "fixed":
            j_info["dofs_motion_ang"] = np.zeros((0, 3))
            j_info["dofs_motion_vel"] = np.zeros((0, 3))
            j_info["dofs_limit"] = np.zeros((0, 2))
            j_info["dofs_stiffness"] = np.zeros((0))

            j_info["type"] = gs.JOINT_TYPE.FIXED
            j_info["n_qs"] = 0
            j_info["n_dofs"] = 0
            j_info["init_qpos"] = np.zeros(0)

        elif joint.joint_type == "revolute":
            j_info["dofs_motion_ang"] = np.array([joint.axis])
            j_info["dofs_motion_vel"] = np.zeros((1, 3))
            j_info["dofs_limit"] = np.array(
                [
                    [
                        joint.limit.lower if joint.limit.lower is not None else -np.inf,
                        joint.limit.upper if joint.limit.upper is not None else np.inf,
                    ]
                ]
            )
            j_info["dofs_stiffness"] = np.array([0.0])

            j_info["type"] = gs.JOINT_TYPE.REVOLUTE
            j_info["n_qs"] = 1
            j_info["n_dofs"] = 1
            j_info["init_qpos"] = np.zeros(1)
        elif joint.joint_type == "continuous":
            j_info["dofs_motion_ang"] = np.array([joint.axis])
            j_info["dofs_motion_vel"] = np.zeros((1, 3))
            j_info["dofs_limit"] = np.array([[-np.inf, np.inf]])
            j_info["dofs_stiffness"] = np.array([0.0])

            j_info["type"] = gs.JOINT_TYPE.REVOLUTE
            j_info["n_qs"] = 1
            j_info["n_dofs"] = 1
            j_info["init_qpos"] = np.zeros(1)
        elif joint.joint_type == "prismatic":
            j_info["dofs_motion_ang"] = np.zeros((1, 3))
            j_info["dofs_motion_vel"] = np.array([joint.axis])
            j_info["dofs_limit"] = np.array(
                [
                    [
                        morph.scale * joint.limit.lower if joint.limit.lower is not None else -np.inf,
                        morph.scale * joint.limit.upper if joint.limit.upper is not None else np.inf,
                    ]
                ]
            )
            j_info["dofs_stiffness"] = np.array([0.0])

            j_info["type"] = gs.JOINT_TYPE.PRISMATIC
            j_info["n_qs"] = 1
            j_info["n_dofs"] = 1
            j_info["init_qpos"] = np.zeros(1)
        elif joint.joint_type == "floating":
            j_info["dofs_motion_ang"] = np.eye(6, 3, -3)
            j_info["dofs_motion_vel"] = np.eye(6, 3)
            j_info["dofs_limit"] = np.tile([-np.inf, np.inf], (6, 1))
            j_info["dofs_stiffness"] = np.zeros(6)

            j_info["type"] = gs.JOINT_TYPE.FREE
            j_info["n_qs"] = 7
            j_info["n_dofs"] = 6
            j_info["init_qpos"] = np.concatenate([gu.zero_pos(), gu.identity_quat()])
        else:
            gs.raise_exception(f"Unsupported URDF joint type: {joint.joint_type}")

        j_info["sol_params"] = gu.default_solver_params()
        j_info["dofs_invweight"] = np.full((j_info["n_dofs"],), fill_value=-1.0)

        joint_friction, joint_damping = 0.0, 0.0
        if joint.dynamics is not None:
            if joint.dynamics.friction is not None:
                joint_friction = joint.dynamics.friction
            if joint.dynamics.damping is not None:
                joint_damping = joint.dynamics.damping
        j_info["dofs_frictionloss"] = np.full(j_info["n_dofs"], joint_friction)
        j_info["dofs_damping"] = np.full(j_info["n_dofs"], joint_damping)
        j_info["dofs_armature"] = np.zeros(j_info["n_dofs"])

        kp = gu.default_dofs_kp(j_info["n_dofs"])
        kv = gu.default_dofs_kv(j_info["n_dofs"])
        if joint.safety_controller is not None:
            if joint.safety_controller.k_position is not None:
                kp = np.tile(joint.safety_controller.k_position, j_info["n_dofs"])
            if joint.safety_controller.k_velocity is not None:
                kv = np.tile(joint.safety_controller.k_velocity, j_info["n_dofs"])
        j_info["dofs_act_gain"] = kp
        j_info["dofs_act_bias"] = np.column_stack([np.zeros_like(kp), -kp, -kv])

        j_info["dofs_force_range"] = np.tile([-np.inf, np.inf], (j_info["n_dofs"], 1))
        if joint.limit is not None and joint.limit.effort is not None:
            j_info["dofs_force_range"] = np.tile([-joint.limit.effort, joint.limit.effort], (j_info["n_dofs"], 1))

    # Apply scaling factor
    for l_info, link_j_infos, link_g_infos in zip(l_infos, links_j_infos, links_g_infos):
        l_info["pos"] *= morph.scale
        if l_info["inertial_pos"] is not None:
            l_info["inertial_pos"] *= morph.scale
        if l_info["inertial_mass"] is not None:
            l_info["inertial_mass"] *= morph.scale**3
        if l_info["inertial_i"] is not None:
            l_info["inertial_i"] *= morph.scale**5
        for j_info in link_j_infos:
            j_info["pos"] *= morph.scale
        for g_info in link_g_infos:
            g_info["pos"] *= morph.scale

    # Re-order kinematic tree info
    l_infos, links_j_infos, links_g_infos, _ = order_links_depth_first(l_infos, links_j_infos, links_g_infos)

    eqs_info = parse_equalities(robot, morph)

    return l_infos, links_j_infos, links_g_infos, eqs_info


def parse_equalities(robot, morph):
    eqs_info = []

    for joint in robot.joints:
        if joint.mimic:
            gs.logger.debug(
                f"Joint '{joint.name}' mimics '{joint.mimic.joint}' with multiplier {joint.mimic.multiplier} and offset {joint.mimic.offset}"
            )

            mimic_joint = robot.joint_map[joint.mimic.joint]
            follower_scale = morph.scale if joint.joint_type == "prismatic" else 1.0
            driver_scale = morph.scale if mimic_joint.joint_type == "prismatic" else 1.0

            eq_info = dict()
            eq_info["type"] = gs.EQUALITY_TYPE.JOINT
            eq_info["name"] = f"mimic_{joint.name}_to_{joint.mimic.joint}"
            eq_info["objs_name"] = (joint.name, joint.mimic.joint)
            eq_info["sol_params"] = gu.default_solver_params()

            # eq_data[0:5] stores a0..a4 in q_follower - q_follower0 = sum(a_k * (q_driver - q_driver0)^k).
            # A model scale transforms a_k to s_f * a_k / s_d**k, where s_f and s_d are morph.scale for
            # prismatic coordinates and 1.0 for revolute coordinates.
            eq_info["data"] = np.zeros([11])
            eq_info["data"][0] = follower_scale * joint.mimic.offset
            eq_info["data"][1] = follower_scale / driver_scale * joint.mimic.multiplier

            eqs_info.append(eq_info)

    return eqs_info


def merge_fixed_links(robot, links_to_keep):
    # Compute breadth-first ordered sequence of joints from root to leaves
    child_map = {}
    for joint in robot.joints:
        child_map.setdefault(joint.parent, []).append(joint)
    joints_ordered = []
    queue = [robot.base_link.name]
    visited = set()
    while queue:
        link_name = queue.pop(0)
        if link_name in visited:
            continue
        visited.add(link_name)
        for joint in child_map.get(link_name, []):
            joints_ordered.append(joint)
            queue.append(joint.child)

    # Prune fixed joints in descending order
    links = robot.links.copy()
    joints = robot.joints.copy()
    for joint in joints_ordered[::-1]:
        if joint.joint_type == "fixed" and joint.child not in links_to_keep:
            parent_link = next(iter(link for link in links if link.name == joint.parent))
            child_link = next(iter(link for link in links if link.name == joint.child))

            update_subtree(links, joints, child_link.name, joint.origin)
            merge_inertia(parent_link, child_link)
            parent_link.visuals.extend(child_link.visuals)
            parent_link.collisions.extend(child_link.collisions)
            joints.remove(joint)
            links.remove(child_link)

            for joint in joints:
                if joint.parent == child_link.name:
                    joint.parent = parent_link.name

    return urdfpy.URDF(robot.name, links=links, joints=joints, materials=robot.materials)


def translate_inertia(I, m, dist):
    """Translate inertia tensor I by dist for a body with mass m."""
    dist = np.array(dist)
    dist_squared = np.dot(dist, dist)
    identity_matrix = np.eye(3)
    translation_matrix = m * (dist_squared * identity_matrix - np.outer(dist, dist))
    return I + translation_matrix


def rotate_inertia(I, R):
    """Rotate inertia tensor I by rotation matrix R."""
    return R @ I @ R.T


def principal_axes_rot(I, tol=1e-3):
    """Return rotation matrix R whose columns are the (unsorted) principal axes of inertia I."""
    I = 0.5 * (I + I.T)
    eigvals, eigvecs = np.linalg.eigh(I)

    if 1.0 - np.mean(np.abs(eigvecs).max(axis=1)) < tol:
        return np.eye(3)

    # Find degenerate groups
    groups, i = [], 0
    while i < 3:
        j = i + 1
        while j < 3 and eigvals[j] - eigvals[i] < tol * eigvals[-1]:
            j += 1
        groups.append(list(range(i, j)))
        i = j

    # Assign unique eigenvectors to their closest coordinate axis first
    out_cols = [None, None, None]
    used_axes = set()
    for g in groups:
        if len(g) == 1:
            col = g[0]
            for ax in np.argsort(np.abs(eigvecs[:, col]))[::-1]:
                if ax not in used_axes:
                    out_cols[ax] = col
                    used_axes.add(ax)
                    break

    # Fill remaining axes with degenerate group columns
    remaining_axes = [ax for ax in range(3) if ax not in used_axes]
    remaining_cols = [c for g in groups if len(g) > 1 for c in g]
    for ax, col in zip(remaining_axes, remaining_cols):
        out_cols[ax] = col

    R = eigvecs[:, out_cols]

    # Procrustes on the degenerate axes (which may be non-contiguous after reordering)
    if remaining_axes:
        U = R[:, remaining_axes]
        T = np.eye(3)[:, remaining_axes]
        Usvd, _, Vt = np.linalg.svd(U.T @ T)
        Q = Usvd @ Vt
        if np.linalg.det(Q) < 0:
            Usvd[:, -1] *= -1
            Q = Usvd @ Vt
        R[:, remaining_axes] = U @ Q

    # Canonicalize sign: largest-magnitude element of each column should be positive
    R *= np.sign(R[np.argmax(np.abs(R), axis=0), np.arange(3)])

    # Make sure the matrix is a pure rotation
    if np.linalg.det(R) < 0:
        R[:, 0] = -R[:, 0]
    return R


def compose_inertial_properties(mass1, com1, inertia1, mass2, com2, inertia2):
    """
    Compose inertial properties of two bodies.

    Args:
        mass1: Mass of first body
        com1: Center of mass of first body (3,) array
        inertia1: Inertia tensor of first body (3,3) array
        mass2: Mass of second body
        com2: Center of mass of second body (3,) array
        inertia2: Inertia tensor of second body (3,3) array

    Returns:
        combined_mass: Combined mass
        combined_com: Combined center of mass (3,) array
        combined_inertia: Combined inertia tensor (3,3) array
    """
    combined_mass = mass1 + mass2
    if combined_mass <= 0.0:
        gs.raise_exception("Combined mass is zero")
    combined_com = (mass1 * com1 + mass2 * com2) / combined_mass
    inertia1_new = translate_inertia(inertia1, mass1, combined_com - com1)
    inertia2_new = translate_inertia(inertia2, mass2, combined_com - com2)
    combined_inertia = inertia1_new + inertia2_new
    return combined_mass, combined_com, combined_inertia


def merge_inertia(link1, link2):
    """Combine two links with fixed joint."""
    if link2.inertial is None:
        return
    elif link2.inertial.origin is None:
        link2.inertial.origin = np.eye(4, dtype=np.float64)

    if link1.inertial is None:
        link1.inertial = link2.inertial
        return
    elif link1.inertial.origin is None:
        link1.inertial.origin = np.eye(4, dtype=np.float64)

    m1 = link1.inertial.mass
    m2 = link2.inertial.mass

    com1, R1 = link1.inertial.origin[:3, 3], link1.inertial.origin[:3, :3]
    com2, R2 = link2.inertial.origin[:3, 3], link2.inertial.origin[:3, :3]

    combined_mass = m1 + m2
    if combined_mass > 0.0:
        combined_com = (m1 * com1 + m2 * com2) / combined_mass
    else:
        combined_com = com1

    # Rotate and translate inertia tensors to the new center of mass
    inertia1_rotated = rotate_inertia(link1.inertial.inertia, R1)
    inertia2_rotated = rotate_inertia(link2.inertial.inertia, R2)

    inertia1_new = translate_inertia(inertia1_rotated, m1, combined_com - com1)
    inertia2_new = translate_inertia(inertia2_rotated, m2, combined_com - com2)

    # Combine the inertia tensors
    combined_inertia = inertia1_new + inertia2_new

    # Set the properties of the combined link
    link1.inertial.mass = combined_mass
    link1.inertial.origin[:3, 3] = combined_com
    link1.inertial.origin[:3, :3] = np.eye(3)  # Reset rotation to identity since it's now aligned
    link1.inertial.inertia = combined_inertia


def transform_inertial(inertial, transform):
    if inertial is None:
        return None

    new_origin = transform @ inertial

    return urdfpy.Inertial(origin=new_origin, mass=inertial.mass, inertia=inertial.inertia)


def update_subtree(links, joints, root_name, transform):
    current_name = root_name
    current_idx = next((idx for idx, link in enumerate(links) if link.name == current_name), None)
    if current_idx is None:
        return
    current_link = links[current_idx]

    # Apply the transformation to the current link
    if current_link.inertial is not None:
        if current_link.inertial.origin is None:
            current_link.inertial.origin = transform
        else:
            current_link.inertial.origin = transform @ current_link.inertial.origin

    for geom in (*current_link.visuals, *current_link.collisions):
        geom.origin = transform @ geom.origin

    for joint in joints:
        if joint.parent == current_name:
            joint.origin = transform @ joint.origin

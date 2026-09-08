"""Describe a rigid entity: everything its simulation uses. That description suffices to create one.

Resolving what a parse states fills these in, and an entity builds itself from them alone: every field holds
the value it is built with, so a consumer reads a field and applies no fallback of its own. The exception is the
inverse weight of a link and of a joint: only the solver can state it, and its refresh completes the sentinel '-1.0'
at build. A link the world carries is the one case stated here, as zeros.

A built link, joint, geom or equality constraint keeps its description as 'desc'. The matching 'get_*' accessor of
that object returns the value the simulation currently uses.
"""

import math
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field, fields
from itertools import chain
from typing import Any, Sequence, TypeVar

import numpy as np
import trimesh
from typing_extensions import Self

import genesis as gs
from genesis.constants import EQUALITY_TYPE, GEOM_TYPE, JOINT_TYPE
from genesis.engine.materials.base import Material
from genesis.engine.mesh import InertialProperties
from genesis.options.morphs import Morph
from genesis.options.options import Options
from genesis.options.surfaces import Surface
from genesis.utils import geom as gu
from genesis.utils import mesh as mu
from genesis.utils import mjcf as mju
from genesis.utils import terrain as tu
from genesis.utils import urdf as uu
from genesis.utils.misc import get_assets_dir

from ..base_entity import EntityDescription
from .inertial import (
    AABB_EPS,
    INERTIA_EIGVAL_ATOL,
    INERTIA_EIGVAL_RTOL,
    INERTIA_RATIO_MAX,
    MASS_EPS,
    RHO_MUJOCO,
    RHO_OBJECT,
    RHO_ROBOT,
    GeomInertialInfo,
    LinkInertial,
    LinkInertialInfo,
    compose_inertial_from_g_infos,
    compose_inertial_properties,
    finalize_inertial,
)

# Rounding a pose parsed from a file may carry, within which it reads as the identity
POSE_EPS = 1e-9


@dataclass(kw_only=True)
class BaseRigidGeomDescription:
    """Pose of one geom in its link frame. Collision and visual geom descriptions both hold these fields."""

    pos: np.ndarray = field(default_factory=gu.zero_pos)
    quat: np.ndarray = field(default_factory=gu.identity_quat)


@dataclass(kw_only=True)
class RigidVisGeomDescription(BaseRigidGeomDescription):
    """Describe one geometry a link is drawn with: where it sits on the link, and the mesh drawn there."""

    vmesh: "gs.Mesh"


@dataclass(kw_only=True)
class RigidGeomDescription(BaseRigidGeomDescription):
    """Describe one geometry a link collides with: where it sits on the link, its shape, and how it is simulated.

    Every field is resolved: the geometry has been split and convexified as the morph asked, and the coefficients are
    the ones the geom is built with, the material's where it states them and the asset's otherwise. What an asset
    states before any of that is resolved travels as parsed info, which is where the post-processing options live.

    'density' is the density of this geometry alone, which the link inertial is resolved against. It is None where
    neither the asset nor the material states one, leaving the geom to take the density of its entity.
    """

    type: GEOM_TYPE
    data: np.ndarray | None
    mesh: "gs.Mesh"
    contype: int
    conaffinity: int
    density: float | None
    friction: float
    friction_torsional: float
    friction_rolling: float
    sol_params: np.ndarray


@dataclass
class RigidJointDescription:
    """Describe one joint of a link, with its per-degree-of-freedom quantities.

    The type gives the number of generalized coordinates and of degrees of freedom the joint holds, and construction
    sizes every per-degree-of-freedom quantity the asset leaves unset from that number, the motion axes of a free
    joint and of a fixed one included.
    """

    name: str
    type: JOINT_TYPE
    n_qs: int
    n_dofs: int
    pos: np.ndarray = field(default_factory=gu.zero_pos)
    quat: np.ndarray = field(default_factory=gu.identity_quat)
    init_qpos: np.ndarray | None = None
    sol_params: np.ndarray | None = None
    dofs_motion_ang: np.ndarray | None = None
    dofs_motion_vel: np.ndarray | None = None
    dofs_limit: np.ndarray | None = None
    dofs_invweight: np.ndarray | None = None
    dofs_frictionloss: np.ndarray | None = None
    dofs_stiffness: np.ndarray | None = None
    dofs_damping: np.ndarray | None = None
    dofs_armature: np.ndarray | None = None
    dofs_act_gain: np.ndarray | None = None
    dofs_act_bias: np.ndarray | None = None
    dofs_force_range: np.ndarray | None = None

    def __post_init__(self):
        if self.init_qpos is None:
            self.init_qpos = np.zeros(self.n_dofs)
        if self.sol_params is None:
            self.sol_params = gu.default_solver_params()
        # Resolve the motion axes of a free joint and of a fixed one, the two types whose axes follow from the
        # number of degrees of freedom. A parser fills in the axes of every other type, here or afterwards
        if self.n_dofs in (0, 6):
            if self.dofs_motion_ang is None:
                self.dofs_motion_ang = np.eye(6, 3, -3) if self.n_dofs == 6 else np.zeros((0, 3))
            if self.dofs_motion_vel is None:
                self.dofs_motion_vel = np.eye(6, 3) if self.n_dofs == 6 else np.zeros((0, 3))
        if self.dofs_limit is None:
            self.dofs_limit = np.tile([[-np.inf, np.inf]], [self.n_dofs, 1])
        if self.dofs_force_range is None:
            self.dofs_force_range = np.tile([[-np.inf, np.inf]], [self.n_dofs, 1])
        if self.dofs_act_bias is None:
            self.dofs_act_bias = np.zeros((self.n_dofs, 3))
        if self.dofs_invweight is None:
            self.dofs_invweight = np.zeros(self.n_dofs)
        if self.dofs_frictionloss is None:
            self.dofs_frictionloss = np.zeros(self.n_dofs)
        if self.dofs_stiffness is None:
            self.dofs_stiffness = np.zeros(self.n_dofs)
        if self.dofs_damping is None:
            self.dofs_damping = np.zeros(self.n_dofs)
        if self.dofs_armature is None:
            self.dofs_armature = np.zeros(self.n_dofs)
        if self.dofs_act_gain is None:
            self.dofs_act_gain = np.zeros(self.n_dofs)


@dataclass(kw_only=True)
class KinematicLinkDescription:
    """Describe one link of an entity: the frame it stands at, the joints that move it, and the geoms it is drawn as.

    'is_aligned' says that the build moved the link frame onto the anchor its geoms share.
    """

    name: str
    parent_idx: int
    root_idx: int | None = None
    pos: np.ndarray = field(default_factory=gu.zero_pos)
    quat: np.ndarray = field(default_factory=gu.identity_quat)
    # Offset applied to the link frame: the morph pose offset composed with the root anchoring transform, identity
    # on child links. The relative pose getters strip it, so they report the pose the user gave.
    offset_pos: np.ndarray = field(default_factory=gu.zero_pos)
    offset_quat: np.ndarray = field(default_factory=gu.identity_quat)
    is_robot: bool = False
    is_aligned: bool = False
    joints: list["RigidJointDescription"] = field(default_factory=list)
    vgeoms: list[RigidVisGeomDescription] = field(default_factory=list)


@dataclass(kw_only=True)
class RigidLinkDescription(KinematicLinkDescription):
    """Describe one simulated link: the frame and geoms of a kinematic link, plus its dynamics.

    Every inertial quantity of a moving link holds the simulated value: the authored one where the asset states it,
    the geometry estimate otherwise. A link the world carries holds what the asset states and None where it states
    nothing, until an attach sets it moving or the build resolves it. The inverse weight holds the asset value, zeros
    for a link the world carries, or the sentinel '-1.0', which the solver's refresh completes at build.
    """

    inertial_pos: np.ndarray | None
    inertial_quat: np.ndarray | None
    inertia: np.ndarray | None
    mass: float | None
    invweight: np.ndarray
    geoms: list[RigidGeomDescription] = field(default_factory=list)


@dataclass
class RigidEqualityDescription:
    """Describe one constraint tying two links or two joints of an entity together, named as the asset names them."""

    name: str
    type: EQUALITY_TYPE
    objs_name: tuple[str, str]
    data: np.ndarray
    sol_params: np.ndarray


@dataclass(kw_only=True)
class KinematicVariantLinkDescription:
    """What one variant of a heterogeneous entity gives one link to be drawn as."""

    vgeoms: list[RigidVisGeomDescription] = field(default_factory=list)


@dataclass(kw_only=True)
class RigidVariantLinkDescription(KinematicVariantLinkDescription):
    """What one variant gives one simulated link: the geoms it adds, and the inertial resolved for them.

    A variant stands for the same asset loaded on its own, so its inertial resolves from its own file and its own
    geometry.
    """

    mass: float
    inertial_pos: np.ndarray
    inertial_quat: np.ndarray
    inertia: np.ndarray
    geoms: list[RigidGeomDescription] = field(default_factory=list)


@dataclass
class KinematicVariantDescription:
    """Describe one variant of a heterogeneous entity: where it stands, and what it gives each link.

    A heterogeneous entity holds one set of links and dispatches a variant per environment, so a variant is described
    beside those links rather than as an entity of its own.

    'default_armature' is the default rotor inertia of its morph, None for a morph without one. Every joint a motor
    could drive gets one, a revolute or prismatic joint moving a link on its own: a motor's rotor is real inertia, and
    the constraint solve needs the armature for its stability anyway, so whether a joint is truly actuated, which the
    asset file often leaves unspecified, is of no consequence. At build, every such degree of freedom whose armature
    the asset leaves at zero receives that fraction of the joint-space sub-tree inertia it drives at the neutral
    configuration, so the regularisation scales with the robot. Each variant asks its own, per environment.
    """

    init_qpos: np.ndarray
    offset_pos: np.ndarray
    offset_quat: np.ndarray
    links: list[KinematicVariantLinkDescription] = field(default_factory=list)
    default_armature: float | None = None


@dataclass
class KinematicAttachmentDescription:
    """Where an entity hangs: the entity its base link was attached to, and the link of it that carries it.

    Both are named as the scene names them, since an index shifts as entities are added. The mount pose lives in the
    base link's own pose, so 'attach' without one restores it.
    """

    entity_name: str
    link_name: str


DescriptionT = TypeVar("DescriptionT")


def description_from_info(cls: type[DescriptionT], info: dict) -> DescriptionT:
    """Build one description of class 'cls' from what a parse states, taking the fields it declares and no others.

    A key the class declares no field for is dropped, and a field the parse states nothing for keeps its declared
    default.
    """
    return cls(**{f.name: info[f.name] for f in fields(cls) if f.name in info})


def is_link_fixed(links: Sequence[KinematicLinkDescription], idx: int) -> bool:
    """Whether the link at 'idx' reaches the world through fixed joints alone, so the world carries it.

    Reads upwards from that link, so every parent must already be described, which the parse order guarantees.
    """
    while True:
        l_desc = links[idx]
        if any(j_desc.type is not gs.JOINT_TYPE.FIXED for j_desc in l_desc.joints):
            return False
        if l_desc.parent_idx == -1:
            return True
        idx = l_desc.parent_idx


@dataclass
class Resolution:
    """Transient state of one resolution, dropped once the description is complete.

    'links_inertial_info' holds one entry per link, then one per variant of that link, in resolution order. The align
    anchoring consumes it (see 'LinkInertialInfo').
    """

    options: Options
    is_mujoco_compatible: bool
    links_inertial_info: list[list[LinkInertialInfo]] = field(default_factory=list)


@dataclass
class KinematicEntityDescription(EntityDescription):
    """Describe one entity as its resolution left it: every link it holds, and every constraint tying them together.

    Each description here stands as the build resolved it rather than as the asset authored it: collision meshes
    split and convexified, link frames aligned on the anchor their geoms share, and the friction the material states
    written into the geoms it overrides. A geom carries its mesh, so an entity is created from this alone, with no
    asset to read and no geometry to process again.

    The morphs, material and surface it was given stand here as well, since a link states none of them: the material
    carries the physics the geoms are simulated under, the surface how they are drawn, and a morph what a kind of
    entity needs beyond its links - the propeller links of a drone, the grid of a terrain. 'morphs' lists every
    morph the entity dispatches, the primary first, and a homogeneous entity lists exactly one.
    """

    morphs: list[Morph]
    surface: Surface
    visualize_contact: bool = False
    name: str | None = None
    links: list[KinematicLinkDescription] = field(default_factory=list)
    variants: list[KinematicVariantDescription] = field(default_factory=list)
    attachment: KinematicAttachmentDescription | None = None

    @classmethod
    def resolve(
        cls,
        morphs: Sequence[Morph],
        material: Material,
        surface: Surface,
        options: Options,
        enable_mujoco_compatibility: bool,
        visualize_contact: bool = False,
        name: str | None = None,
    ) -> Self:
        """Resolve what an entity is created with into the description it is built from.

        This is where every asset a morph names is read and post-processed, so an entity created from the returned
        description opens no file. 'morphs' lists the primary morph first, then one morph per heterogeneous variant.
        'options' are those of the solver simulating the entity, which the parsers check the assets against.
        """
        morphs = list(morphs)
        if len(morphs) > 1:
            if not all(
                isinstance(morph, (gs.morphs.Primitive, gs.morphs.Mesh, gs.morphs.URDF, gs.morphs.MJCF))
                for morph in morphs
            ):
                gs.raise_exception("Heterogeneous morphs only support Primitive, Mesh, URDF and MJCF types.")
            if len(set(isinstance(morph, (gs.morphs.URDF, gs.morphs.MJCF)) for morph in morphs)) > 1:
                gs.raise_exception(
                    "Heterogeneous morphs must be consistent: either all articulated robots (ie URDF, MJCF) or all "
                    "basic objects (ie Primitive, Mesh)."
                )

        if isinstance(material, gs.materials.Rigid):
            # small sdf res is sufficient for primitives regardless of size
            if isinstance(morphs[0], gs.morphs.Primitive):
                material.sdf_max_res = 32

        # Set material-dependent default options
        for morph in morphs:
            if isinstance(morph, gs.morphs.FileMorph):
                # Rigid entities will convexify geom by default
                if morph.convexify is None:
                    morph.convexify = isinstance(material, gs.materials.Rigid)
                # Decimation simplifies away the very surface detail that a non-convex collision mesh is kept for, so
                # it defaults off when convexify is off and on otherwise. Only applies to meshes that skip
                # watertightening (already-watertight inputs); watertighten does its own feature-preserving QEM.
                if morph.decimate is None:
                    morph.decimate = morph.convexify
                # Genesis fills in a default rotor armature for joints whose armature is not specified in the model
                # file, while MuJoCo's own default may differ. Under MuJoCo compatibility, the default is dropped
                # unless set manually, deferring to MuJoCo. USD keeps the Genesis default since MuJoCo is not
                # involved in its parsing.
                if (
                    isinstance(morph, (gs.morphs.MJCF, gs.morphs.URDF, gs.morphs.Drone))
                    and "default_armature" not in morph.model_fields_set
                    and enable_mujoco_compatibility
                ):
                    morph.default_armature = None

        desc = cls(morphs=morphs, material=material, surface=surface, visualize_contact=visualize_contact, name=name)
        desc._load_morph(morphs[0], Resolution(options, enable_mujoco_compatibility))
        return desc

    def _load_morph(self, morph: Morph, resolution: Resolution):
        """Load a single morph into the description."""
        if isinstance(morph, gs.morphs.Mesh):
            self._load_mesh(morph, resolution)
        elif isinstance(morph, (gs.morphs.MJCF, gs.morphs.URDF, gs.morphs.Drone, gs.morphs.USD)):
            self._load_scene(morph, resolution)
        elif isinstance(morph, gs.morphs.Primitive):
            self._load_primitive(morph, resolution)
        else:
            gs.raise_exception(f"Unsupported morph: {morph}.")

        # Load heterogeneous variants (if any)
        self._load_heterogeneous_morphs(resolution)

        # The per-link inertia (and per heterogeneous variant) is now resolved, so anchor each aligned free root at
        # its fixed subtree center of mass and principal axes. Must run before the solver reads the link poses and
        # inertia. Defined here on the base class so kinematic and rigid entities anchor identically: a kinematic
        # entity is commonly used to visualize the target reference motion a rigid entity tracks, so the two coexist
        # and the same qpos must map to the same world pose for both.
        self._align_free_roots(resolution)

    def _load_heterogeneous_morphs(self, resolution: Resolution):
        """Load heterogeneous morphs (additional geometry variants for parallel environments).

        Each variant is loaded as additional geoms/vgeoms attached to links.
        Each variant is described beside the links (see 'KinematicVariantDescription').
        """
        if not len(self.morphs) > 1:
            return

        # The per-variant offset and inertial alignment are tracked for a single root only; a multi-root entity (one
        # MJCF/URDF with several free root bodies) would cross-contaminate the roots' offsets and init poses.
        if sum(l_desc.parent_idx == -1 for l_desc in self.links) > 1:
            gs.raise_exception("Heterogeneous morphs are not supported on multi-root entities.")

        # Track per-variant init_qpos and base-link offset for per-environment dispatch (primary first). Each variant
        # accumulates its own below so an asymmetric variant is aligned exactly like its homogeneous equivalent.
        init_qpos_chunks = [j_desc.init_qpos for l_desc in self.links for j_desc in l_desc.joints]
        self.variants.append(
            KinematicVariantDescription(
                init_qpos=np.concatenate(init_qpos_chunks) if init_qpos_chunks else np.array([], dtype=gs.np_float),
                offset_pos=self.links[0].offset_pos,
                offset_quat=self.links[0].offset_quat,
                # Only a scene or robot description file yields a robot link, and those morphs state a default armature.
                default_armature=self.morphs[0].default_armature if any(l.is_robot for l in self.links) else None,
            )
        )

        n_links = len(self.links)

        # Load additional heterogeneous variants
        for morph in self.morphs[1:]:
            if isinstance(morph, (gs.morphs.URDF, gs.morphs.MJCF)):
                # Parse variant scene file
                v_l_infos, v_links_j_infos, v_links_g_infos, _ = self._parse_scene(morph, resolution)

                # Validate that the variant has the same joint structure as the primary
                if len(v_l_infos) != n_links:
                    gs.raise_exception(
                        f"Heterogeneous variant has {len(v_l_infos)} links, "
                        f"but primary has {n_links}. All variants must have the same link count."
                    )
                for i_l, (l_desc, v_j_infos) in enumerate(zip(self.links, v_links_j_infos)):
                    primary_joints = l_desc.joints
                    if len(v_j_infos) != len(primary_joints):
                        gs.raise_exception(
                            f"Heterogeneous variant link {i_l} has {len(v_j_infos)} joints, "
                            f"but primary has {len(primary_joints)}."
                        )
                    for p_joint, v_j_info in zip(primary_joints, v_j_infos):
                        if p_joint.name != v_j_info["name"]:
                            gs.raise_exception(
                                f"Joint name mismatch at link {i_l}: primary has '{p_joint.name}', "
                                f"variant has '{v_j_info['name']}'. All variants must have the same joint names."
                            )
                        if p_joint.type != v_j_info["type"]:
                            gs.raise_exception(
                                f"Joint type mismatch for '{p_joint.name}': primary has {p_joint.type}, "
                                f"variant has {v_j_info['type']}."
                            )
                        if p_joint.n_dofs != v_j_info["n_dofs"]:
                            gs.raise_exception(
                                f"DoF count mismatch for joint '{p_joint.name}': primary has {p_joint.n_dofs}, "
                                f"variant has {v_j_info['n_dofs']}."
                            )

                # Post-process each link's geoms. The COM/principal-axis anchoring of the floating base is deferred
                # to '_align_free_roots' (where the resolved per-variant composite inertia is known); here
                # only the morph pose offset is composed into the variant's init_qpos and offset.
                offset_pos = np.array(morph.offset_pos, dtype=gs.np_float)
                offset_quat = np.array(morph.offset_quat, dtype=gs.np_float)
                cg_vg_infos = []
                for v_l_info, v_j_infos, v_g_infos in zip(v_l_infos, v_links_j_infos, v_links_g_infos):
                    is_robot = v_l_info["is_robot"]
                    cg_infos, vg_infos = self._resolve_geoms(morph, v_g_infos, is_robot)
                    cg_vg_infos.append((cg_infos, vg_infos))

                # Extract variant's init_qpos from parsed joint infos, composing the morph offset into the free joint so
                # relative getters report the variant's authored frame.
                variant_init_qpos_parts = []
                for v_l_info, v_j_infos in zip(v_l_infos, v_links_j_infos):
                    is_root = v_l_info["parent_idx"] == -1
                    for j_info in v_j_infos:
                        qpos = j_info["init_qpos"]
                        if is_root and j_info["type"] == gs.JOINT_TYPE.FREE:
                            init_pos, init_quat = gu.transform_pos_quat_by_trans_quat(
                                np.array(morph.offset_pos, dtype=gs.np_float),
                                np.array(morph.offset_quat, dtype=gs.np_float),
                                qpos[:3],
                                qpos[3:7],
                            )
                            qpos = np.concatenate([init_pos, init_quat])
                        variant_init_qpos_parts.append(qpos)

                # Resolve each link's inertial for this variant and stash it for the align anchor.
                is_inertia_recomputed = morph.recompute_inertia
                variant_links = []
                for i_link, (v_l_info, (cg_infos, vg_infos)) in enumerate(zip(v_l_infos, cg_vg_infos)):
                    inertial_info = self._resolve_inertial(
                        None if is_inertia_recomputed else v_l_info.get("inertial_mass"),
                        None if is_inertia_recomputed else v_l_info.get("inertial_pos"),
                        None if is_inertia_recomputed else v_l_info.get("inertial_quat"),
                        None if is_inertia_recomputed else v_l_info.get("inertial_i"),
                        cg_infos,
                        vg_infos,
                        bool(v_l_info["is_robot"]),
                        resolution,
                    )
                    resolution.links_inertial_info[i_link].append(inertial_info)
                    variant_links.append(
                        self._describe_variant_link(i_link, v_l_info, cg_infos, vg_infos, morph, inertial_info)
                    )
                self.variants.append(
                    KinematicVariantDescription(
                        init_qpos=np.concatenate(variant_init_qpos_parts) if variant_init_qpos_parts else np.array([]),
                        offset_pos=offset_pos,
                        offset_quat=offset_quat,
                        links=variant_links,
                        default_armature=morph.default_armature,
                    )
                )

            elif isinstance(morph, (gs.morphs.Mesh, gs.morphs.Primitive)):
                if isinstance(morph, gs.morphs.Mesh):
                    g_infos = self._load_mesh(morph, resolution, load_geom_only_for_heterogeneous=True)
                else:
                    g_infos = self._load_primitive(morph, resolution, load_geom_only_for_heterogeneous=True)
                if morph.fixed != self.morphs[0].fixed:
                    gs.raise_exception("Mixing fixed and non-fixed morphs in heterogeneous entities is not supported.")
                cg_infos, vg_infos = self._resolve_geoms(morph, g_infos, is_robot=False)

                # The COM/principal-axis anchoring is deferred to '_align_free_roots', which runs once every variant
                # is described. Compose only the morph pose offset here.
                offset_pos = np.array(morph.offset_pos, dtype=gs.np_float)
                offset_quat = np.array(morph.offset_quat, dtype=gs.np_float)

                # Mesh/Primitive variants have no explicit inertial; the anchor inertia comes from their geometry.
                inertial_info = self._resolve_inertial(None, None, None, None, cg_infos, vg_infos, False, resolution)
                resolution.links_inertial_info[0].append(inertial_info)
                variant_link = self._describe_variant_link(0, None, cg_infos, vg_infos, morph, inertial_info)

                if morph.fixed:
                    init_qpos = np.array((), dtype=gs.np_float)
                else:
                    init_pos, init_quat = gu.transform_pos_quat_by_trans_quat(
                        np.array(morph.offset_pos, dtype=gs.np_float),
                        np.array(morph.offset_quat, dtype=gs.np_float),
                        np.array(morph.pos, dtype=gs.np_float),
                        np.array(morph.quat, dtype=gs.np_float),
                    )
                    init_qpos = np.concatenate([init_pos, init_quat])
                self.variants.append(
                    KinematicVariantDescription(
                        init_qpos=init_qpos,
                        offset_pos=offset_pos,
                        offset_quat=offset_quat,
                        links=[variant_link],
                    )
                )
            else:
                gs.raise_exception(
                    f"Heterogeneous morphs only support URDF, MJCF, Primitive, and Mesh, got: {type(morph).__name__}."
                )

    def _describe_variant_link(self, i_link, v_l_info, cg_infos, vg_infos, morph, inertial_info):
        """Describe the geoms one variant gives one link. A kinematic entity only ever draws them."""
        return KinematicVariantLinkDescription(
            vgeoms=[description_from_info(RigidVisGeomDescription, vg_info) for vg_info in vg_infos]
        )

    def _load_primitive(self, morph, resolution: Resolution, load_geom_only_for_heterogeneous=False):
        if morph.fixed:
            joint_type = gs.JOINT_TYPE.FIXED
            n_qs = 0
            n_dofs = 0
            init_qpos = np.array([])
        else:
            joint_type = gs.JOINT_TYPE.FREE
            n_qs = 7
            n_dofs = 6
            init_qpos = np.concatenate([morph.pos, morph.quat])

        metadata: dict[str, Any] = {"texture_path": None}

        if isinstance(morph, gs.options.morphs.Box):
            extents = np.array(morph.size)
            tmesh = mu.create_box(extents=extents)
            cmesh = tmesh
            geom_data = extents
            geom_type = gs.GEOM_TYPE.BOX
            link_name_prefix = "box"
        elif isinstance(morph, gs.options.morphs.Sphere):
            tmesh = mu.create_sphere(radius=morph.radius)
            cmesh = tmesh
            geom_data = np.array([morph.radius])
            geom_type = gs.GEOM_TYPE.SPHERE
            link_name_prefix = "sphere"
        elif isinstance(morph, gs.options.morphs.Cylinder):
            tmesh = mu.create_cylinder(radius=morph.radius, height=morph.height)
            cmesh = tmesh
            geom_data = np.array([morph.radius, morph.height])
            geom_type = gs.GEOM_TYPE.CYLINDER
            link_name_prefix = "cylinder"
        elif isinstance(morph, gs.options.morphs.Plane):
            metadata["texture_path"] = mu.DEFAULT_PLANE_TEXTURE_PATH
            tmesh, cmesh = mu.create_plane(
                normal=morph.normal,
                plane_size=morph.plane_size,
                tile_size=morph.tile_size,
                color_or_texture=metadata["texture_path"],
            )
            geom_data = np.array(morph.normal)
            geom_type = gs.GEOM_TYPE.PLANE
            link_name_prefix = "plane"
        else:
            gs.raise_exception("Unsupported primitive shape")

        # contains one visual geom (vgeom) and one collision geom (geom)
        g_infos = []
        if morph.visualization:
            g_infos.append(
                dict(
                    contype=0, conaffinity=0, vmesh=gs.Mesh.from_trimesh(tmesh, surface=self.surface, metadata=metadata)
                )
            )
        if (morph.contype or morph.conaffinity) and morph.collision:
            g_infos.append(
                dict(
                    contype=morph.contype,
                    conaffinity=morph.conaffinity,
                    mesh=gs.Mesh.from_trimesh(cmesh, surface=gs.surfaces.Collision()),
                    type=geom_type,
                    data=geom_data,
                    sol_params=gu.default_solver_params(),
                )
            )

        # For heterogeneous simulation, only return geometry info without creating link/joint
        if load_geom_only_for_heterogeneous:
            return g_infos

        self._resolve_link(
            l_info=dict(
                name=f"{link_name_prefix}_baselink",
                parent_idx=-1,
                root_idx=None,
                pos=np.array(morph.pos),
                quat=np.array(morph.quat),
                is_robot=False,
                # The center of mass and everything derived from it come from the geometry the link is given
                inertial_pos=None,
                inertial_quat=gu.identity_quat(),
                inertial_i=None,
                inertial_mass=None,
                invweight=None,
            ),
            j_infos=[
                dict(
                    name=f"{link_name_prefix}_baselink_joint",
                    n_qs=n_qs,
                    n_dofs=n_dofs,
                    type=joint_type,
                    init_qpos=init_qpos,
                )
            ],
            g_infos=g_infos,
            morph=morph,
            resolution=resolution,
        )
        return g_infos

    def _load_mesh(self, morph, resolution: Resolution, load_geom_only_for_heterogeneous=False):
        # Load meshes
        meshes = gs.Mesh.from_morph_surface(morph, self.surface)

        link_pos, link_quat = map(np.array, (morph.pos, morph.quat))

        if morph.fixed:
            joint_type = gs.JOINT_TYPE.FIXED
            n_qs = 0
            n_dofs = 0
            init_qpos = np.array([])
        else:
            joint_type = gs.JOINT_TYPE.FREE
            n_qs = 7
            n_dofs = 6
            init_qpos = np.concatenate([link_pos, link_quat])

        g_infos = []
        if morph.visualization:
            for mesh in meshes:
                g_infos.append(dict(contype=0, conaffinity=0, vmesh=mesh, pos=gu.zero_pos(), quat=gu.identity_quat()))
        if morph.collision:
            if morph.merge_submeshes_for_collision and len(meshes) > 1:
                # Merge every submesh into a single collision geom if requested.
                collision_groups = [list(meshes)]
            else:
                # A source mesh node split into several visual materials is one physical body, so its submeshes are
                # merged into a single collision geom rather than split per material. Pieces meant to collide
                # separately must be authored as separate nodes. Meshes with no source node are each their own body.
                collision_groups = []
                groups_by_node = {}
                for mesh in meshes:
                    node_index = mesh.metadata.get("node_index")
                    if node_index is None:
                        collision_groups.append([mesh])
                        continue
                    group = groups_by_node.get(node_index)
                    if group is None:
                        group = groups_by_node[node_index] = []
                        collision_groups.append(group)
                    group.append(mesh)

            for group in collision_groups:
                if len(group) == 1:
                    mesh = group[0]
                else:
                    tmesh = trimesh.util.concatenate([submesh.trimesh for submesh in group])
                    mesh = gs.Mesh.from_trimesh(mesh=tmesh, surface=gs.surfaces.Collision())
                g_infos.append(
                    dict(
                        contype=morph.contype,
                        conaffinity=morph.conaffinity,
                        mesh=mesh,
                        type=gs.GEOM_TYPE.MESH,
                        sol_params=gu.default_solver_params(),
                        pos=gu.zero_pos(),
                        quat=gu.identity_quat(),
                    )
                )

        # For heterogeneous simulation, only return geometry info without creating link/joint
        if load_geom_only_for_heterogeneous:
            return g_infos

        link_name = os.path.basename(morph.file).replace(".", "_")

        self._resolve_link(
            l_info=dict(
                name=f"{link_name}_baselink",
                parent_idx=-1,
                root_idx=None,
                pos=link_pos,
                quat=link_quat,
                is_robot=False,
                inertial_pos=None,
                inertial_quat=None,
                inertial_i=None,
                inertial_mass=None,
                invweight=None,
            ),
            j_infos=[
                dict(
                    name=f"{link_name}_baselink_joint",
                    n_qs=n_qs,
                    n_dofs=n_dofs,
                    type=joint_type,
                    init_qpos=init_qpos,
                )
            ],
            g_infos=g_infos,
            morph=morph,
            resolution=resolution,
        )
        return g_infos

    def _parse_scene(self, morph, resolution: Resolution):
        # Whether the parsed inverse weight has been invalidated, either because the inertia it derives from was
        # replaced, or because it was never trustworthy in the first place. It is applied once, after every reason
        # to invalidate has been collected.
        is_inertia_invalid = False

        # Mujoco's unified MJCF+URDF parser is not good enough for now to be used for loading both MJCF and URDF files.
        # First, it would happen when loading visual meshes having supported format (i.e. Collada files '.dae').
        # Second, it does not take into account URDF 'mimic' joint constraints. However, it does a better job at
        # initialized undetermined physics parameters.
        if isinstance(morph, gs.morphs.MJCF):
            # Mujoco's unified MJCF+URDF parser systematically for MJCF files
            l_infos, links_j_infos, links_g_infos, eqs_info = mju.parse_xml(morph, self.surface, resolution.options)
        elif isinstance(morph, (gs.morphs.URDF, gs.morphs.Drone)):
            # Custom "legacy" URDF parser for loading geometries (visual and collision) and equality constraints.
            # This is necessary because Mujoco cannot parse visual geometries (meshes) reliably for URDF.
            l_infos, links_j_infos, links_g_infos, eqs_info = uu.parse_urdf(morph, self.surface)

            # Mujoco's unified MJCF+URDF parser for only link, joints, and collision geometries properties
            morph_ = morph.model_copy(update=dict(visualization=False))
            try:
                # Mujoco's unified MJCF+URDF parser for URDF files.
                # Note that Mujoco URDF parser completely ignores equality constraints.
                l_infos_mj, links_j_infos_mj, links_g_infos_mj, _ = mju.parse_xml(morph_, self.surface)

                # Unset link inertial properties that are actually undefined to force recomputation by genesis
                if not resolution.is_mujoco_compatible:
                    for l_info_gs in l_infos:
                        for l_info_mj in l_infos_mj:
                            if l_info_gs["name"] == l_info_mj["name"]:
                                for key, value in l_info_gs.items():
                                    if value is None:
                                        l_info_mj[key] = None
                                        is_inertia_invalid = True
                                break
                l_infos = l_infos_mj

                # Mujoco is not parsing actuators properties
                for j_info_gs in chain.from_iterable(links_j_infos):
                    for j_info_mj in chain.from_iterable(links_j_infos_mj):
                        if j_info_mj["name"] == j_info_gs["name"]:
                            for name in ("dofs_force_range", "dofs_armature", "dofs_act_gain", "dofs_act_bias"):
                                j_info_mj[name] = j_info_gs[name]
                            break
                links_j_infos = links_j_infos_mj

                # Take into account 'world' body if it was added automatically for our legacy URDF parser
                if len(links_g_infos_mj) == len(links_g_infos) + 1:
                    assert not links_g_infos_mj[0]
                    links_g_infos.insert(0, [])
                assert len(links_g_infos_mj) == len(links_g_infos)

                # Update collision geometries, ignoring fake" visual geometries returned by Mujoco, (which is using
                # collision as visual to avoid loading mesh files), and keeping the true visual geometries provided
                # by our custom legacy URDF parser.
                # Note that the Kinematic tree ordering is stable between Mujoco and Genesis (Hopefully!).
                for link_g_infos, link_g_infos_mj in zip(links_g_infos, links_g_infos_mj):
                    # Remove collision geometries from our legacy URDF parser
                    for i_g, g_info in tuple(enumerate(link_g_infos))[::-1]:
                        is_col = g_info["contype"] or g_info["conaffinity"]
                        if is_col:
                            del link_g_infos[i_g]

                    # Add visual geometries from Mujoco's unified MJCF+URDF parser
                    for g_info in link_g_infos_mj:
                        is_col = g_info["contype"] or g_info["conaffinity"]
                        if is_col:
                            link_g_infos.append(g_info)
            except (ValueError, AssertionError) as e:
                gs.logger.warning(
                    "Falling back to legacy URDF parser. Default values of physics properties may be off:\n"
                    + str(e).replace("\n", " - ")
                )
        elif isinstance(morph, gs.morphs.USD):
            # Imported here since USD support is optional and its bindings load with the package
            from genesis.utils.usd import parse_usd_rigid_entity

            # Unified parser handles both articulations and rigid bodies
            l_infos, links_j_infos, links_g_infos, eqs_info = parse_usd_rigid_entity(morph, self.surface)

        # Make sure that the inertia matrix of all links is valid
        if not morph.recompute_inertia:
            for l_info in l_infos:
                inertia_i = l_info.get("inertial_i")
                if inertia_i is None:
                    continue

                # Compute eigenvalues of inertia matrix after enforcing symmetry
                inertia_diag, Q = np.linalg.eigh(0.5 * (inertia_i + inertia_i.T))

                # Make sure that all eigenvalues are positive, ignoring the rounding of the authored tensor
                if (inertia_diag < -(np.abs(inertia_diag).max() * INERTIA_EIGVAL_RTOL + INERTIA_EIGVAL_ATOL)).any():
                    gs.raise_exception(
                        f"Inertia matrix of link '{l_info['name']}' not positive definite (eigenvalues: {inertia_diag})."
                    )

                # Make sure that the inertia matrix is physically valid (nothing to do with numerical conditioning)
                if any(
                    inertia_diag[i] + inertia_diag[(i + 1) % 3]
                    < inertia_diag[(i + 2) % 3] * (1.0 - INERTIA_EIGVAL_RTOL) - INERTIA_EIGVAL_ATOL
                    for i in range(3)
                ):
                    gs.raise_exception(
                        f"Inertia matrix of link '{l_info['name']}' does not satisfy A+B>=C for all permutations "
                        f"(eigenvalues: {inertia_diag}). Please fix manually you morph file '{morph.file}' or specify "
                        "`recompute_inertia=True`."
                    )

                # Make sure that the inertia matrix is symmetric with positive eigenvalues
                l_info["inertial_i"] = Q @ np.diag(np.maximum(inertia_diag, 0.0)) @ Q.T

        # Remove any "virtual" root link that was not present in the original file morph.
        # Mujoco unified parser and our legacy parser have different behaviors.
        # * Mujoco unified parser always adds a root 'world' link if it does not exist, and fuse all fixed links from
        #   root to first articulated body.
        # * Our legacy parser adds a root 'world' link if the root joint is not a fixed joint in file morph.
        # Remove this virtual world link if the child has a free joint (the free joint absorbs the full pose into
        # 'init_qpos' regardless of pos/quat), or if the child has an identity transform.
        base_j_info, base_g_info = links_j_infos[0], links_g_infos[0]
        if len(l_infos) > 1 and (sum(j_info["n_dofs"] for j_info in base_j_info) == 0) and not base_g_info:
            child_has_freejoint = any(j_info["type"] == gs.JOINT_TYPE.FREE for j_info in links_j_infos[1])
            child_is_identity = np.allclose(l_infos[1]["pos"], 0.0, rtol=0.0, atol=POSE_EPS) and np.allclose(
                l_infos[1]["quat"], (1.0, 0.0, 0.0, 0.0), rtol=0.0, atol=POSE_EPS
            )
            if child_has_freejoint or child_is_identity:
                del l_infos[0], links_j_infos[0], links_g_infos[0]
                for l_info in l_infos:
                    l_info["parent_idx"] = max(l_info["parent_idx"] - 1, -1)
                    if "root_idx" in l_info:
                        l_info["root_idx"] = max(l_info["root_idx"] - 1, -1)

        # URDF is a robot description file so all links have same root_idx
        if isinstance(morph, gs.morphs.URDF) and not resolution.is_mujoco_compatible:
            for l_info in l_infos:
                l_info["root_idx"] = 0

        # Genesis requires links associated with free joints to be attached to the world directly
        for l_info, link_j_infos in zip(l_infos, links_j_infos):
            if all(j_info["type"] == gs.JOINT_TYPE.FREE for j_info in link_j_infos):
                l_info["parent_idx"] = -1

        # Add free floating joint at root if necessary
        if (
            (isinstance(morph, gs.morphs.Drone) or (isinstance(morph, gs.morphs.URDF) and not morph.fixed))
            and links_j_infos
            and sum(j_info["n_dofs"] for j_info in links_j_infos[0]) == 0
        ):
            # Define free joint
            dofs_damping = np.zeros(6)
            if isinstance(morph, gs.morphs.Drone):
                # FIXME: This pattern not ideal because the inertial mass may be unknown at this point.
                mass_tot = sum(l_info.get("inertial_mass") or 0.0 for l_info in l_infos)
                dofs_damping[3:] = mass_tot * morph.default_base_ang_damping_scale
            links_j_infos[0] = [
                dict(
                    name="root_joint",
                    type=gs.JOINT_TYPE.FREE,
                    n_qs=7,
                    n_dofs=6,
                    init_qpos=np.concatenate([gu.zero_pos(), gu.identity_quat()]),
                    dofs_damping=dofs_damping,
                )
            ]

            # The base link is now moving, which merges every kinematic tree connected to it into a single tree rooted
            # at it. Parser-provided root indices assume a base welded to the world, so recompute them by propagating
            # each parent's root (parents precede children); links with no parent root their own tree.
            for i_l, l_info in enumerate(l_infos):
                if "root_idx" in l_info:
                    parent_idx = l_info["parent_idx"]
                    l_info["root_idx"] = l_infos[parent_idx]["root_idx"] if parent_idx != -1 else i_l

            # Must invalidate invweight for all child links and joints because the root joint was fixed when it was
            # initially computed. Re-initialize it to some strictly negative value to trigger recomputation in solver.
            for i_l in range(len(l_infos)):
                l_infos[i_l]["invweight"] = np.full((2,), fill_value=-1.0)
                for j_info in links_j_infos[i_l]:
                    j_info["dofs_invweight"] = np.full((j_info["n_dofs"],), fill_value=-1.0)

        # Force recomputing inertial information based on geometry if ill-defined for some reason.
        # A moving link needs a well-defined inertia only for its own rigid body; a rigidly-attached (fixed-joint) child
        # folds its mass into the parent's composite-rigid-body inertia. An authored mass counts only when positive, and
        # a link without an authored mass counts when it has geometry.
        has_links_subtree_mass = [
            bool(link_g_infos) if l_info.get("inertial_mass") is None else l_info["inertial_mass"] > 0.0
            for link_g_infos, l_info in zip(links_g_infos, l_infos)
        ]
        has_links_fixed_child_mass = [False] * len(l_infos)
        for i_l in reversed(range(len(l_infos))):
            parent_idx = l_infos[i_l]["parent_idx"]
            if parent_idx >= 0 and all(j_info["type"] == gs.JOINT_TYPE.FIXED for j_info in links_j_infos[i_l]):
                has_links_subtree_mass[parent_idx] |= has_links_subtree_mass[i_l]
                has_links_fixed_child_mass[parent_idx] |= has_links_subtree_mass[i_l]

        for l_info, link_g_infos, link_j_infos, has_link_subtree_mass, has_link_fixed_child_mass in zip(
            l_infos, links_g_infos, links_j_infos, has_links_subtree_mass, has_links_fixed_child_mass
        ):
            is_fixed_link = all(j_info["type"] == gs.JOINT_TYPE.FIXED for j_info in link_j_infos)
            is_fixed_child = is_fixed_link and l_info["parent_idx"] >= 0
            is_mass_degenerate = (l_info.get("inertial_mass") or 0.0) <= 0.0
            inertia = l_info.get("inertial_i")
            is_inertia_degenerate = inertia is None or (np.diag(inertia) <= 0.0).any()
            # The world or the parent carries a fixed link without geometry, so its values stay as authored.
            if is_fixed_link and not link_g_infos:
                continue
            # A fixed child keeps its authored mass, zero included. Its inertia contributes to the parent composite, so a
            # degenerate inertia is recovered from geometry.
            if is_fixed_child and not is_inertia_degenerate:
                continue
            if not (is_mass_degenerate or is_inertia_degenerate):
                continue

            # The own inertia is degenerate, so the parsed inverse weight (derived from it) must be recomputed
            # regardless of how the mass is resolved below.
            is_inertia_invalid = True

            # A geometry-less moving link whose rigidly-attached (fixed-joint) subtree carries the mass keeps its
            # (near-)zero own inertia: the composite-rigid-body inertia is finite, so leave it as parsed.
            if not link_g_infos and has_link_subtree_mass:
                continue
            if not link_g_infos:
                gs.logger.warning(
                    f"Moving link '{l_info['name']}' has no mass, inertia or geometry, and no rigidly-attached "
                    "child provides any. Setting its mass to 'gs.EPS'."
                )
                continue
            if l_info.get("inertial_mass") is not None or inertia is not None:
                gs.logger.debug(
                    f"Invalid or undefined inertia for link '{l_info['name']}'. Force recomputing it based on geometry."
                )
            # A moving link with a massless fixed subtree has a singular composite, so its mass comes from the geometry
            # estimate. A fixed link, or a moving link with mass-bearing fixed children, keeps its zero mass because the
            # composite is finite.
            if is_mass_degenerate and not is_fixed_child and not has_link_fixed_child_mass:
                l_info["inertial_mass"] = None
            # The authored inertia fits the authored mass, so the mass reset also discards the inertia.
            if is_inertia_degenerate or l_info.get("inertial_mass") is None:
                l_info["inertial_i"] = None
        if is_inertia_invalid:
            for l_info, link_j_infos in zip(l_infos, links_j_infos):
                l_info["invweight"] = np.full((2,), fill_value=-1.0)
                for j_info in link_j_infos:
                    j_info["dofs_invweight"] = np.full((j_info["n_dofs"],), fill_value=-1.0)

        # Check if there is something weird with the options
        non_physical_fieldnames = ("dofs_frictionloss", "dofs_damping", "dofs_armature")
        for j_info in (
            j_info for link_j_infos in links_j_infos for j_info in link_j_infos if j_info["type"] == gs.JOINT_TYPE.FREE
        ):
            if any((j_info[name] != 0.0).any() for name in non_physical_fieldnames if name in j_info):
                gs.logger.warning(
                    "Some free joint has non-zero frictionloss, damping or armature parameters. Beware it is "
                    "non-physical."
                )

        # Define a flag that determines whether the link at hand is associated with a robot.
        # Note that 0d array is used rather than native type because this algo requires mutable objects.
        for l_info, link_j_infos in zip(l_infos, links_j_infos):
            if not link_j_infos or all(j_info["type"] == gs.JOINT_TYPE.FIXED for j_info in link_j_infos):
                if l_info["parent_idx"] >= 0:
                    l_info["is_robot"] = l_infos[l_info["parent_idx"]]["is_robot"]
                else:
                    l_info["is_robot"] = np.array(False, dtype=np.bool_)
            elif all(j_info["type"] == gs.JOINT_TYPE.FREE for j_info in link_j_infos):
                l_info["is_robot"] = np.array(False, dtype=np.bool_)
            else:
                l_info["is_robot"] = np.array(True, dtype=np.bool_)
                if l_info["parent_idx"] >= 0:
                    l_infos[l_info["parent_idx"]]["is_robot"][()] = True

        # Apply morph pos and quat if specified
        for l_info, link_j_infos in zip(l_infos, links_j_infos):
            if l_info["parent_idx"] < 0:
                if morph.pos is not None or morph.quat is not None:
                    gs.logger.debug("Applying offset to base link's pose with user provided value in morph.")
                    pos = l_info["pos"]
                    quat = l_info["quat"]
                    if morph.pos is None:
                        pos_offset = np.zeros((3,))
                    else:
                        pos_offset = np.asarray(morph.pos)
                    if morph.quat is None:
                        quat_offset = np.array((1.0, 0.0, 0.0, 0.0))
                    else:
                        quat_offset = np.asarray(morph.quat)
                    l_info["pos"], l_info["quat"] = gu.transform_pos_quat_by_trans_quat(
                        pos, quat, pos_offset, quat_offset
                    )

                for j_info in link_j_infos:
                    if j_info["type"] == gs.JOINT_TYPE.FREE:
                        # in this case, l_info['pos'] and l_info['quat'] are actually not used in solver,
                        # but this initial value will be reflected
                        j_info["init_qpos"] = np.concatenate([l_info["pos"], l_info["quat"]])

        # Exclude joints with 0 dofs to align with Mujoco
        links_j_infos = [[j_info for j_info in link_j_infos if j_info["n_dofs"] > 0] for link_j_infos in links_j_infos]

        return l_infos, links_j_infos, links_g_infos, eqs_info

    def _load_scene(self, morph, resolution: Resolution):
        l_infos, links_j_infos, links_g_infos, _eqs_info = self._parse_scene(morph, resolution)

        # Add (link, joints, geoms) tuples sequentially
        for l_info, link_j_infos, link_g_infos in zip(l_infos, links_j_infos, links_g_infos):
            self._resolve_link(l_info, link_j_infos, link_g_infos, morph, resolution)

    def _align_free_roots(self, resolution: Resolution):
        """Anchor each aligned free root at the center of mass and principal axes of its fixed subtree.

        Composes the inertia resolved for every link and variant, and applies the anchoring per heterogeneous
        variant, each against its own inertia and geoms. This delivers the 'align' option's promise (a COM-centered,
        principal-axis frame, which also conditions the constraint solve) for mesh and fixed-child bodies, not just
        primitives, while keeping a heterogeneous entity bit-identical to its variants' separate entities.
        """
        links = self.links
        variants = self.variants
        for i_root, root in enumerate(links):
            if root.parent_idx != -1 or not root.is_aligned:
                continue

            # Gather the fixed subtree (root + transitive n_dofs == 0 descendants) and each link's pose in the root
            # frame; links are in build order so a parent is always visited before its children. A DOF-bearing
            # descendant makes the root an articulated chain rather than a single rigid body: its joint-space mass is
            # not diagonal and the frames of its moving children are not re-expressed here, so such roots are skipped.
            pose_in_root = {i_root: (gu.zero_pos(), gu.identity_quat())}
            subtree = [i_root]
            is_articulated = False
            for i_l, l_desc in enumerate(links):
                if i_l == i_root or l_desc.parent_idx not in pose_in_root:
                    continue
                if any(j_desc.n_dofs for j_desc in l_desc.joints):
                    is_articulated = True
                    break
                pose_in_root[i_l] = gu.transform_pos_quat_by_trans_quat(
                    l_desc.pos, l_desc.quat, *pose_in_root[l_desc.parent_idx]
                )
                subtree.append(i_l)
            if is_articulated:
                root.is_aligned = False
                continue

            # Each variant anchors on its own resolved inertia and geoms, which kinematic and rigid entities both
            # have. The anchor composes the fixed subtree's inertia and moves the body frame to its center of mass
            # and principal axes. It re-expresses the variant geoms, so the geometry keeps its world pose, and folds
            # into the variant offset and init_qpos.
            for i_v in range(len(resolution.links_inertial_info[i_root])):
                inertial_info = []
                explicit_mass_flags = set()
                for i_l in subtree:
                    props, is_mass_explicit, _ = resolution.links_inertial_info[i_l][i_v]
                    # The stash holds the unit-density estimate, so a mass however small is a body of the composite, and
                    # only a link without geometry or authored mass is left out.
                    if props.mass <= 0.0:
                        continue
                    explicit_mass_flags.add(is_mass_explicit)
                    rot = gu.quat_to_R(props.quat)
                    inertia_in_link = rot @ props.inertia @ rot.T
                    inertial_info.append(
                        GeomInertialInfo(InertialProperties(props.mass, props.com, inertia_in_link), *pose_in_root[i_l])
                    )

                # Inertia alignment is insensitive to density if and only if the density is homogeneous. That must
                # hold among the geometry materials of a link, and among the links the composite anchors on (see
                # 'LinkInertialInfo.is_mass_explicit'). Otherwise the procedure requires the density of every
                # geometry. A kinematic entity only visualizes, so its material carries no default density and an
                # unspecified one cannot be resolved for it. A rigid entity and the kinematic ghost of it must
                # anchor at one frame, so the inhomogeneous case is structurally impossible to support. Require
                # all-or-none and raise otherwise.
                if None in explicit_mass_flags:
                    gs.raise_exception(
                        f"Link '{root.name}': a link of its aligned free body mixes geoms with and without an "
                        "authored density. Author a density on all of its geoms or none of them."
                    )
                if len(explicit_mass_flags) > 1:
                    gs.raise_exception(
                        f"Link '{root.name}': its aligned free body mixes explicit (mass or authored per-geom "
                        "densities) and geometry-estimated link masses. Specify the mass or density of all of its "
                        "links or none of them."
                    )
                # A body without inertia gets no anchor and keeps its frame, like the articulated body above. The solver
                # and the setters then read it as unaligned (see 'func_midpoint_is_aligned' and '_init_tree_fields').
                if not inertial_info:
                    root.is_aligned = False
                    continue
                mass_total, com_root, inertia_root = compose_inertial_properties(inertial_info)
                if mass_total <= 0.0:
                    root.is_aligned = False
                    continue
                principal_quat = gu.R_to_quat(uu.principal_axes_rot(inertia_root))

                # Re-express the variant's geoms across the subtree so the body frame can move to (com_root, principal
                # axes) while the world geometry stays fixed. The static link poses are shared across heterogeneous
                # variants, so the anchoring is undone by a transform conjugated into each link frame, reused below for
                # the inertial frame of a fixed child. A kinematic link has only visual geoms; a rigid link has both.
                alignment_inv_in_link = {}
                for i_l in subtree:
                    link_pos, link_quat = pose_in_root[i_l]
                    pos, quat = gu.inv_transform_pos_quat_by_trans_quat(link_pos, link_quat, com_root, principal_quat)
                    pos, quat = gu.inv_transform_pos_quat_by_trans_quat(pos, quat, link_pos, link_quat)
                    alignment_inv_in_link[i_l] = (pos, quat)
                    link_desc = links[i_l] if i_v == 0 else variants[i_v].links[i_l]
                    geoms = list(link_desc.vgeoms)
                    if isinstance(link_desc, (RigidLinkDescription, RigidVariantLinkDescription)):
                        geoms += link_desc.geoms
                    for geom in geoms:
                        geom.pos, geom.quat = gu.transform_pos_quat_by_trans_quat(geom.pos, geom.quat, pos, quat)

                # Every link keeps its own mass and inertia. The inertial frame moves with the geoms into the anchored
                # frame: getters and wrenches read it as 'link_COM', and the inverse weight derives from it. The solver
                # composes the body from its links each step, so the composite is diagonal there (see
                # '_init_tree_fields').
                if isinstance(root, RigidLinkDescription):
                    for i_l in subtree:
                        desc = links[i_l] if i_v == 0 else variants[i_v].links[i_l]
                        desc.inertial_pos, desc.inertial_quat = gu.transform_pos_quat_by_trans_quat(
                            desc.inertial_pos, desc.inertial_quat, *alignment_inv_in_link[i_l]
                        )

                # Fold the anchoring into the offset (relative getters keep reporting the authored frame) and the root
                # free-joint init_qpos (the body frame moves to old_pose o (com_root, principal), keeping the
                # re-expressed geoms in world).
                if variants:
                    off_pos, off_quat = variants[i_v].offset_pos, variants[i_v].offset_quat
                    variants[i_v].offset_pos = gu.transform_by_trans_quat(com_root, off_pos, off_quat)
                    variants[i_v].offset_quat = gu.transform_quat_by_quat(principal_quat, off_quat)
                    # The solver broadcasts '_links_offset_*' (not '_variant_offset_*') when all variants share an
                    # offset, so keep the base link offset in sync with the primary variant's aligned offset.
                    if i_v == 0:
                        root.offset_pos = variants[0].offset_pos
                        root.offset_quat = variants[0].offset_quat
                    qpos = variants[i_v].init_qpos
                    new_pos = gu.transform_by_trans_quat(com_root, qpos[:3], qpos[3:7])
                    new_quat = gu.transform_quat_by_quat(principal_quat, qpos[3:7])
                    qpos[:3], qpos[3:7] = new_pos, new_quat
                else:
                    root.offset_pos = gu.transform_by_trans_quat(com_root, root.offset_pos, root.offset_quat)
                    root.offset_quat = gu.transform_quat_by_quat(principal_quat, root.offset_quat)
                    for j_desc in root.joints:
                        if j_desc.type == gs.JOINT_TYPE.FREE:
                            init_qpos = j_desc.init_qpos
                            new_pos = gu.transform_by_trans_quat(com_root, init_qpos[:3], init_qpos[3:7])
                            new_quat = gu.transform_quat_by_quat(principal_quat, init_qpos[3:7])
                            j_desc.init_qpos = np.concatenate([new_pos, new_quat])
                    root.pos = gu.transform_by_trans_quat(com_root, root.pos, root.quat)
                    root.quat = gu.transform_quat_by_quat(principal_quat, root.quat)

        # Only the estimates and the anchors need the inertial info. The alignment now lives in the persistent
        # geometry, offset and init_qpos.
        resolution.links_inertial_info.clear()

    def _resolve_link(self, l_info, j_infos, g_infos, morph, resolution: Resolution):
        """Describe one link from what a parse states about it, and add that description to the entity's own.

        Splitting and convexifying the collision geometry, anchoring the link frame, resolving the inertial and
        writing in the coefficients the material overrides are what resolution decides, and the description carries
        all of it. Creating the link is the build's alone, which walks the descriptions.
        """
        if len(j_infos) > 1 and any(j_info["type"] in (gs.JOINT_TYPE.FREE, gs.JOINT_TYPE.FIXED) for j_info in j_infos):
            raise ValueError(
                "Compounding joints of types 'FREE' or 'FIXED' with any other joint on the same body not supported"
            )

        # Split and convexify collision geometry. Must be done before alignment so that convexified geoms are used
        # to compute the inertia frame.
        cg_infos, vg_infos = self._resolve_geoms(morph, g_infos, l_info["is_robot"])

        # Carry the morph pose offset into root links and align them to their collision-geometry COM/principal axes.
        inertial_info = self._align_link(l_info, j_infos, cg_infos, vg_infos, morph, resolution)

        self.links.append(self._describe_link(l_info, j_infos, cg_infos, vg_infos, morph, inertial_info))

    def _describe_link(self, l_info, j_infos, cg_infos, vg_infos, morph, inertial_info):
        """Describe one link as its resolution left it: the frame it stands at, and what it is drawn as.

        A kinematic link is placed by the kinematics, so an authored mass and inertia are dropped. Its collision
        geometry serves the align anchoring and is discarded afterwards.
        """
        return KinematicLinkDescription(
            name=l_info["name"],
            parent_idx=l_info["parent_idx"],
            root_idx=l_info.get("root_idx"),
            pos=l_info["pos"],
            quat=l_info["quat"],
            offset_pos=l_info["offset_pos"],
            offset_quat=l_info["offset_quat"],
            is_robot=bool(l_info["is_robot"]),
            is_aligned=l_info["is_aligned"],
            joints=[description_from_info(RigidJointDescription, j_info) for j_info in j_infos],
            vgeoms=[description_from_info(RigidVisGeomDescription, vg_info) for vg_info in vg_infos],
        )

    def _resolve_geoms(self, morph, g_infos, is_robot):
        """Split what a parse states about the geoms of one link, and post-process its collision geometry.

        Post-processing convexifies the collision meshes, decomposing the ones a single hull cannot approximate within
        the threshold the morph carries. Both kinds of geometry stay as parsed info, so the inertial composition
        treats them alike and '_describe_geoms' resolves the collision coefficients later. A plain load and a
        heterogeneous variant both go through here.
        """
        # An explicitly set material density overrides any asset-authored per-geom density, as material friction does
        # for authored frictions. Dropping the keys up front keeps the align anchor, its all-or-none source check,
        # and the build-time inertial estimate consistently uniform-density.
        if isinstance(self.material, gs.materials.Rigid) and self.material.rho is not None:
            for g_info in g_infos:
                g_info.pop("density", None)

        cg_infos, vg_infos = [], []
        for g_info in g_infos:
            is_col = g_info["contype"] or g_info["conaffinity"]
            if morph.collision and is_col:
                cg_infos.append(g_info)
            if morph.visualization and not is_col:
                vg_infos.append(g_info)

        # Post-process all collision meshes at once.
        # Destroying the original geometries should be avoided if possible as it will change the way objects
        # interact with the world due to only computing one contact point per convex geometry. The idea is to
        # check if each geometry can be convexified independently without resorting on convex decomposition.
        # If so, the original geometries are preserve. If not, then they are all merged as one. Following the
        # same approach as before, the resulting geometry is convexify without resorting on convex decomposition
        # if possible. Mergeing before falling back directly to convex decompositio is important as it gives one
        # last chance to avoid it. Moreover, it tends to reduce the final number of collision geometries. In
        # both cases, this improves runtime performance, numerical stability and compilation time.
        if isinstance(morph, gs.options.morphs.FileMorph):
            # Choose the appropriate convex decomposition error threshold depending on whether the link at hand
            # is associated with a robot.
            # The rational behind it is that performing convex decomposition for robots is mostly useless because
            # the non-physical part that is added to the original geometries to convexify them are generally inside
            # the mechanical structure and not interacting directly with the outer world. On top of that, not only
            # iy increases the memory footprint and compilation time, but also the simulation speed (marginally).
            if is_robot:
                decompose_error_threshold = morph.decompose_robot_error_threshold
            else:
                decompose_error_threshold = morph.decompose_object_error_threshold

            # A collision geom may carry per-geom post-processing overrides (a USD MeshCollisionAPI approximation
            # hint sets "convexify"/"decimate"/"decompose_error_threshold"), with the morph options as defaults for
            # whatever is left unset. Post-processing merges geoms within a call, so each set of effective options
            # gets its own call; geoms without overrides share one, preserving the whole-entity merge behavior.
            # Thresholds match within the tolerance the mesh cache matches options with, so float noise cannot split a
            # group.
            cg_infos_by_options: list[tuple[tuple, list]] = []
            for g_info in cg_infos:
                convexify = g_info.pop("convexify", morph.convexify)
                decimate = g_info.pop("decimate", morph.decimate)
                threshold = g_info.pop("decompose_error_threshold", decompose_error_threshold)
                for options, options_cg_infos in cg_infos_by_options:
                    if options[:2] == (convexify, decimate) and math.isclose(
                        options[2], threshold, rel_tol=mu.MESH_CACHE_MATCH_RTOL
                    ):
                        options_cg_infos.append(g_info)
                        break
                else:
                    cg_infos_by_options.append(((convexify, decimate, threshold), [g_info]))

            cg_infos = []
            for (convexify, decimate, threshold), options_cg_infos in cg_infos_by_options:
                cg_infos += mu.postprocess_collision_geoms(
                    options_cg_infos,
                    decimate,
                    morph.decimate_face_num,
                    morph.decimate_aggressiveness,
                    convexify,
                    threshold,
                    morph.coacd_options,
                    morph.watertighten,
                )

        # Randomize collision mesh colors. This is especially useful to check convex decomposition.
        for g_info in cg_infos:
            mesh = g_info["mesh"]
            mesh.set_color((*np.random.rand(3), 0.7))

        return cg_infos, vg_infos

    def _resolve_inertial(
        self, explicit_mass, explicit_com, explicit_quat, explicit_inertia, cg_infos, vg_infos, is_robot, resolution
    ):
        """Compute a link's load-time inertial data (see 'LinkInertialInfo').

        The align-anchor inertial weighs each collision geom by its authored density, falling back to unit density,
        so it never needs the material density - which a kinematic entity does not have. The geometry hint consumed
        by '_describe_link' uses the resolved material density as fallback instead, and falls back to the visual
        geoms for a link without collision geometry.
        """
        hint = compose_inertial_from_g_infos(cg_infos, rho=1.0)
        props = finalize_inertial(
            explicit_mass, explicit_com, explicit_quat, explicit_inertia, *hint, clamp_min_mass=False
        )
        if explicit_mass is not None and explicit_mass > 0.0:
            is_mass_explicit = True
        else:
            geoms_with_density = sum(g_info.get("density") is not None for g_info in cg_infos)
            if geoms_with_density == 0:
                is_mass_explicit = False
            elif geoms_with_density == len(cg_infos):
                is_mass_explicit = True
            else:
                is_mass_explicit = None

        dynamics_hint = None
        if isinstance(self.material, gs.materials.Rigid):
            rho = self.material.rho
            if rho is None:
                if resolution.is_mujoco_compatible:
                    rho = RHO_MUJOCO
                else:
                    rho = RHO_ROBOT if is_robot else RHO_OBJECT

            # The estimate comes from the collision geometry of the link when it has any, and from its visual
            # geometry otherwise. A link with neither contributes nothing, so only the asset's values remain.
            dynamics_hint = compose_inertial_from_g_infos(cg_infos or vg_infos, rho)
        return LinkInertialInfo(props, is_mass_explicit, dynamics_hint)

    def _align_link(self, l_info, j_infos, cg_infos, vg_infos, morph, resolution: Resolution):
        """Carry the morph pose offset into a root link, record the offset it leaves, and resolve its inertia.

        Only root (floating-base) links carry the morph 'offset_pos'/'offset_quat' and the 'aligned' flag. The
        relative getters strip the resulting body-frame offset to report the user's original pose. Separately, every
        link's finalized inertia (explicit values, else from its collision geometry) is resolved here - the one point
        where the collision geometry is available to both kinematic and rigid entities - and returned, so that
        '_align_free_roots' derives the COM/principal anchor identically for both.
        """

        # Resolve the local inertia of this link (primary variant). recompute_inertia discards explicit values
        # for non-world-fixed links, exactly as the link description does; an aligned free body's subtree is never
        # world-fixed, so it suffices to honor the morph flag here.
        is_inertia_recomputed = isinstance(morph, gs.options.morphs.FileMorph) and morph.recompute_inertia
        inertial_info = self._resolve_inertial(
            None if is_inertia_recomputed else l_info.get("inertial_mass"),
            None if is_inertia_recomputed else l_info.get("inertial_pos"),
            None if is_inertia_recomputed else l_info.get("inertial_quat"),
            None if is_inertia_recomputed else l_info.get("inertial_i"),
            cg_infos,
            vg_infos,
            l_info["is_robot"],
            resolution,
        )
        resolution.links_inertial_info.append([inertial_info])

        # Only a root link carries the morph pose offset, so a child link keeps an identity offset and stays
        # unaligned
        l_info["offset_pos"], l_info["offset_quat"] = gu.zero_pos(), gu.identity_quat()
        l_info["is_aligned"] = False

        if l_info["parent_idx"] != -1:
            return inertial_info

        # Compose the morph pose offset into the root link's world pose. The solver strips the matching offset in
        # relative getters, so the authored frame is unchanged.
        offset_pos = np.array(morph.offset_pos, dtype=gs.np_float)
        offset_quat = np.array(morph.offset_quat, dtype=gs.np_float)
        l_info["pos"], l_info["quat"] = gu.transform_pos_quat_by_trans_quat(
            offset_pos, offset_quat, l_info["pos"], l_info["quat"]
        )

        is_aligned = morph.align if isinstance(morph, gs.options.morphs.FileMorph) else False
        if is_aligned is None:
            # Auto: True for basic rigid objects (root with free joint only, no articulated descendants). A link
            # mixing geoms with and without an authored density (see 'LinkInertialInfo.is_mass_explicit') quietly
            # declines auto-alignment; asking for it with an explicit align=True raises at build instead.
            geoms_with_density = sum(g_info.get("density") is not None for g_info in cg_infos)
            is_aligned = (
                not bool(l_info["is_robot"])
                and all(j_info["type"] == gs.JOINT_TYPE.FREE for j_info in j_infos)
                and geoms_with_density in (0, len(cg_infos))
            )

        # A free body opting into alignment (or any primitive, which is inherently principal-axis and COM-centered) has
        # an exactly-diagonal joint-space mass matrix once anchored. The COM/principal-axis anchoring itself is deferred
        # to '_align_free_roots', where the resolved composite inertia of the fixed subtree is known and
        # can be applied per heterogeneous variant. Here only the morph pose offset is composed (child link poses are
        # defined relative to it); the anchoring transform is folded in later.
        l_info["is_aligned"] = any(j_info["type"] == gs.JOINT_TYPE.FREE for j_info in j_infos) and (
            is_aligned or isinstance(morph, gs.options.morphs.Primitive)
        )

        # Refresh the free joint init_qpos to reflect the composed world pose.
        for j_info in j_infos:
            if j_info["type"] == gs.JOINT_TYPE.FREE:
                j_info["init_qpos"] = np.concatenate([l_info["pos"], l_info["quat"]])

        # Record the body-frame offset.
        l_info["offset_pos"], l_info["offset_quat"] = offset_pos, offset_quat

        return inertial_info


@dataclass
class RigidEntityDescription(KinematicEntityDescription):
    """Describe one rigid entity, which is a kinematic entity whose links are simulated.

    A constraint tying two links or two joints together is described here rather than beside a kinematic entity,
    since the rigid solver is what enforces one.
    """

    links: list[RigidLinkDescription] = field(default_factory=list)
    equalities: list[RigidEqualityDescription] = field(default_factory=list)

    def _describe_variant_link(self, i_link, v_l_info, cg_infos, vg_infos, morph, inertial_info):
        """Describe what one variant gives one simulated link, the inertial it is simulated with included.

        A variant is resolved against its own file and its own geometry, exactly as the link itself is, so it behaves
        like that asset loaded on its own. A Mesh or Primitive variant authors no inertial, and a morph asking for the
        inertia to be recomputed drops what its file stated.
        """
        mass, com, quat, inertia = None, None, None, None
        if v_l_info is not None and not (morph.recompute_inertia and not is_link_fixed(self.links, i_link)):
            mass, com, quat, inertia = (
                v_l_info.get("inertial_mass"),
                v_l_info.get("inertial_pos"),
                v_l_info.get("inertial_quat"),
                v_l_info.get("inertial_i"),
            )
        inertial = finalize_inertial(
            mass, com, quat, inertia, *inertial_info.hint, clamp_min_mass=not is_link_fixed(self.links, i_link)
        )
        return RigidVariantLinkDescription(
            vgeoms=[description_from_info(RigidVisGeomDescription, vg_info) for vg_info in vg_infos],
            mass=inertial.mass,
            inertial_pos=inertial.com,
            inertial_quat=inertial.quat,
            inertia=inertial.inertia,
            geoms=self._describe_geoms(cg_infos),
        )

    def _load_scene(self, morph, resolution: Resolution):
        l_infos, links_j_infos, links_g_infos, eqs_info = self._parse_scene(morph, resolution)

        # Add (link, joints, geoms) tuples sequentially
        for l_info, link_j_infos, link_g_infos in zip(l_infos, links_j_infos, links_g_infos):
            self._resolve_link(l_info, link_j_infos, link_g_infos, morph, resolution)

        # Add equality constraints sequentially
        self.equalities.extend(description_from_info(RigidEqualityDescription, e_info) for e_info in eqs_info)

    def _describe_geoms(self, cg_infos):
        """Resolve what a parse stated about each collision geom into the description the geom is built from.

        A coefficient is the material's where it states one, the asset's where the material states none, and the
        solver's default where neither states any. What only the resolution needs stays behind in the parsed info
        (see 'RigidGeomDescription').
        """
        cg_descs = []
        for g_info in cg_infos:
            coefficients = []
            for material_coefficient, asset_coefficient, default_coefficient in (
                (self.material.friction, g_info.get("friction"), gu.default_friction),
                (self.material.friction_torsional, g_info.get("friction_torsional"), gu.default_friction_torsional),
                (self.material.friction_rolling, g_info.get("friction_rolling"), gu.default_friction_rolling),
            ):
                coefficient = asset_coefficient if material_coefficient is None else material_coefficient
                coefficients.append(default_coefficient() if coefficient is None else coefficient)
            friction, friction_torsional, friction_rolling = coefficients
            cg_descs.append(
                RigidGeomDescription(
                    pos=g_info.get("pos", gu.zero_pos()),
                    quat=g_info.get("quat", gu.identity_quat()),
                    type=g_info["type"],
                    data=g_info.get("data"),
                    mesh=g_info["mesh"],
                    contype=g_info["contype"],
                    conaffinity=g_info["conaffinity"],
                    density=g_info.get("density"),
                    friction=friction,
                    friction_torsional=friction_torsional,
                    friction_rolling=friction_rolling,
                    sol_params=g_info.get("sol_params", gu.default_solver_params()),
                )
            )
        return cg_descs

    def _describe_link(self, l_info, j_infos, cg_infos, vg_infos, morph, inertial_info):
        """Describe one simulated link as its resolution left it, the inertial it is simulated with included.

        What the asset states is kept where it states it, and the estimate from the geometry fills in everywhere
        else, both through one resolution, so the dynamics inertia and the align anchor stay in lockstep.
        'recompute_inertia' drops what the asset stated, on a moving link alone. The inverse weight keeps the value
        the asset carries only while the inertia it derives from is the simulated one, and holds the sentinel
        otherwise for the solver's refresh to complete at build.
        """
        is_fixed = all(j_info["type"] is gs.JOINT_TYPE.FIXED for j_info in j_infos) and (
            l_info["parent_idx"] == -1 or is_link_fixed(self.links, l_info["parent_idx"])
        )
        is_inertia_recomputed = (
            not is_fixed and isinstance(morph, gs.options.morphs.FileMorph) and morph.recompute_inertia
        )
        mass = None if is_inertia_recomputed else l_info.get("inertial_mass")
        com = None if is_inertia_recomputed else l_info.get("inertial_pos")
        quat = None if is_inertia_recomputed else l_info.get("inertial_quat")
        inertia = None if is_inertia_recomputed else l_info.get("inertial_i")
        invweight = l_info.get("invweight")

        # A fixed link takes no geometry estimate: it may be the environment or an object welded for this scene alone,
        # and nothing in the asset tells which. It keeps what the asset states for an attach that sets it moving to
        # resolve (see 'KinematicEntity.attach'), and the build zeroes the rest (see 'RigidEntity._build'). A moving
        # link without geometry keeps the mass 'finalize_inertial' floors at 'gs.EPS'.
        hint = InertialProperties(0.0, np.zeros(3, dtype=gs.np_float), np.zeros((3, 3), dtype=gs.np_float))
        if not is_fixed and inertial_info.hint is not None:
            hint = inertial_info.hint

            # Compute the bounding box of the links using both visual and collision geometries to be conservative
            aabb_min = np.full((3,), float("inf"), dtype=gs.np_float)
            aabb_max = np.full((3,), float("-inf"), dtype=gs.np_float)
            for mesh, geom_pos, geom_quat in chain(
                (
                    (g_info["mesh"], g_info.get("pos", gu.zero_pos()), g_info.get("quat", gu.identity_quat()))
                    for g_info in cg_infos
                ),
                (
                    (g_info["vmesh"], g_info.get("pos", gu.zero_pos()), g_info.get("quat", gu.identity_quat()))
                    for g_info in vg_infos
                ),
            ):
                verts = gu.transform_by_trans_quat(mesh.verts, geom_pos, geom_quat)
                aabb_min = np.minimum(aabb_min, verts.min(axis=0))
                aabb_max = np.maximum(aabb_max, verts.max(axis=0))

            # Make sure that provided spatial inertia is consistent with the estimate from the geometries if not fixed
            if (mass or hint.mass) > MASS_EPS and hint.mass > 0.0:
                # An omitted center of mass resolves to the link frame origin whenever the inertia tensor is
                # authored (see 'finalize_inertial'). It then takes the same consistency check as an authored one.
                checked_com = gu.zero_pos() if com is None and inertia is not None else com
                if checked_com is not None:
                    tol = (aabb_max - aabb_min) * AABB_EPS + AABB_EPS
                    if not ((aabb_min - tol < checked_com) & (checked_com < aabb_max + tol)).all():
                        com_str = ", ".join(f"{axis}={pos:0.3f}" for axis, pos in zip("xyz", checked_com))
                        aabb_str = ", ".join(
                            f"{axis}=({low:0.3f}, {high:0.3f})" for axis, low, high in zip("xyz", aabb_min, aabb_max)
                        )
                        gs.logger.warning(
                            f"Link '{l_info['name']}' has dubious center of mass [{com_str}] compared to the "
                            f"bounding box from geometry [{aabb_str}]."
                        )

                hint_inertia = hint.i
                if mass is not None:
                    if not (hint.mass / INERTIA_RATIO_MAX <= mass <= INERTIA_RATIO_MAX * hint.mass):
                        gs.logger.warning(
                            f"Link '{l_info['name']}' has dubious mass {mass:0.3f} compared to the estimate from "
                            f"geometry {hint.mass:0.3f}."
                        )
                    hint_inertia = hint_inertia * (mass / hint.mass)

                if inertia is not None:
                    inertia_diag, hint_diag = np.diag(inertia), np.diag(hint_inertia)
                    if not (
                        (hint_diag / INERTIA_RATIO_MAX <= inertia_diag)
                        & (inertia_diag <= INERTIA_RATIO_MAX * hint_diag)
                    ).all():
                        inertias_str = [
                            ",".join(f"{axis}={val:0.3e}" for axis, val in zip(("ixx", "iyy", "izz"), data))
                            for data in (inertia_diag, hint_diag)
                        ]
                        gs.logger.warning(
                            f"Link '{l_info['name']}' has dubious inertia [{inertias_str[0]}] compared to the "
                            f"estimate from geometry [{inertias_str[1]}]."
                        )

        if mass is None or inertia is None:
            if not is_fixed and vg_infos and not cg_infos:
                gs.logger.info(
                    f"Mass is not specified and collision geoms can not be found for link '{l_info['name']}'. "
                    f"Using visual geoms to compute inertial properties."
                )
            # The parsed inverse weight matches the inertia the asset declares. The inertia recomputed here breaks
            # that match, so the value is discarded
            invweight = None

        # The final inertial comes from the explicit values and the geometry estimate. The align anchor shares
        # 'finalize_inertial', so the dynamics inertia and that anchor stay in lockstep.
        if is_fixed:
            inertial = LinkInertial(mass, com, quat, inertia)
        else:
            inertial = finalize_inertial(mass, com, quat, inertia, *hint)

        # override invweight if fixed
        if is_fixed:
            invweight = np.zeros((2,), dtype=gs.np_float)
        # Postpone computation of inverse weight if not specified
        elif invweight is None:
            invweight = np.full((2,), fill_value=-1.0, dtype=gs.np_float)

        return RigidLinkDescription(
            name=l_info["name"],
            parent_idx=l_info["parent_idx"],
            root_idx=l_info.get("root_idx"),
            pos=l_info["pos"],
            quat=l_info["quat"],
            offset_pos=l_info["offset_pos"],
            offset_quat=l_info["offset_quat"],
            is_robot=bool(l_info["is_robot"]),
            is_aligned=l_info["is_aligned"],
            joints=[description_from_info(RigidJointDescription, j_info) for j_info in j_infos],
            vgeoms=[description_from_info(RigidVisGeomDescription, vg_info) for vg_info in vg_infos],
            inertial_pos=inertial.com,
            inertial_quat=inertial.quat,
            inertia=inertial.inertia,
            mass=inertial.mass,
            invweight=invweight,
            geoms=self._describe_geoms(cg_infos),
        )


@dataclass
class TerrainEntityDescription(RigidEntityDescription):
    """Describe one terrain, which is a rigid entity beside the height field its surface is made of.

    The field holds the elevation of every grid cell in meters, vertical scale applied, and the scale gives the
    horizontal size of a cell and the vertical factor. The collider reads them at build and the height query reads
    them at runtime. They stand here because no link describes them.
    """

    terrain_hf: np.ndarray | None = None
    terrain_scale: np.ndarray | None = None

    def _load_morph(self, morph, resolution: Resolution):
        """Load the height field a terrain morph describes, with its scales and the meshes standing for it."""
        vmesh, mesh, terrain_hf = tu.parse_terrain(morph, self.surface)
        self.terrain_scale = np.array((morph.horizontal_scale, morph.vertical_scale), dtype=gs.np_float)
        self.terrain_hf = terrain_hf * self.terrain_scale[1]

        g_infos = []
        if morph.visualization:
            g_infos.append(dict(contype=0, conaffinity=0, vmesh=vmesh))
        if morph.collision:
            g_infos.append(
                dict(
                    contype=1,
                    conaffinity=1,
                    mesh=mesh,
                    type=gs.GEOM_TYPE.TERRAIN,
                    sol_params=gu.default_solver_params(),
                    pos=gu.zero_pos(),
                    quat=gu.identity_quat(),
                )
            )

        self._resolve_link(
            l_info=dict(
                name="baselink",
                parent_idx=-1,
                root_idx=None,
                pos=np.array(morph.pos),
                quat=np.array(morph.quat),
                is_robot=False,
                inertial_pos=None,
                inertial_quat=gu.identity_quat(),
                inertial_i=None,
                inertial_mass=None,
                invweight=None,
            ),
            j_infos=[dict(name="joint_baselink", type=gs.JOINT_TYPE.FIXED, n_qs=0, n_dofs=0)],
            g_infos=g_infos,
            morph=morph,
            resolution=resolution,
        )

        # Load heterogeneous variants (if any)
        self._load_heterogeneous_morphs(resolution)


@dataclass
class DroneEntityDescription(RigidEntityDescription):
    """Describe one drone, which is a rigid entity beside the two coefficients its asset states for the propellers.

    The thrust and torque coefficients are read from the asset and appear nowhere else, so a drone created from this
    carries the same flight behaviour without the file it was authored from.
    """

    kf: float | None = None
    km: float | None = None

    def _load_scene(self, morph, resolution: Resolution):
        super()._load_scene(morph, resolution)

        properties = ET.parse(os.path.join(get_assets_dir(), morph.file)).getroot()[0].attrib
        self.kf = float(properties["kf"])
        self.km = float(properties["km"])

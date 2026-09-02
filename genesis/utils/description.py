"""Describe a rigid entity as its asset authored it, before the entity is built.

A parser fills these records in, and the entity builds itself from them. Each record resolves its own defaults on
construction, so a consumer reads a field and applies no fallback of its own.

A built link, joint, geom or equality constraint keeps its record as 'desc', where a quantity holds the value the
build gave it. The matching 'get_*' accessor of that object returns the value the simulation currently uses.
"""

from dataclasses import dataclass, field

import numpy as np

import genesis as gs
from genesis.constants import EQUALITY_TYPE, GEOM_TYPE, JOINT_TYPE
from genesis.utils import geom as gu


# Generalized coordinates and degrees of freedom held by each joint type.
JOINT_TYPE_SIZE = {
    JOINT_TYPE.FIXED: (0, 0),
    JOINT_TYPE.REVOLUTE: (1, 1),
    JOINT_TYPE.PRISMATIC: (1, 1),
    JOINT_TYPE.SPHERICAL: (4, 3),
    JOINT_TYPE.FREE: (7, 6),
}


@dataclass
class GeomDescription:
    """Describe one geometry of a link, used for collision or for visualization.

    A collision geom carries 'mesh', its shape 'type', and the 'data' that type reads. A visual geom carries 'vmesh'.
    An unset field means the asset authored nothing: the density then falls back to the one the material states, and
    the post-processing options to the ones the morph carries. 'prim_path' names the source prim of a geom parsed
    from a USD stage, which the collision filtering of that format resolves its groups against.
    """

    contype: int
    conaffinity: int
    type: GEOM_TYPE | None = None
    data: np.ndarray | None = None
    pos: np.ndarray = field(default_factory=gu.zero_pos)
    quat: np.ndarray = field(default_factory=gu.identity_quat)
    mesh: "gs.Mesh | None" = None
    vmesh: "gs.Mesh | None" = None
    friction: float | None = None
    friction_torsional: float | None = None
    friction_rolling: float | None = None
    group: int | None = None
    sol_params: np.ndarray | None = None
    density: float | None = None
    convexify: bool | None = None
    decimate: bool | None = None
    decompose_error_threshold: float | None = None
    prim_path: str | None = None

    def __post_init__(self):
        # Apply the default coefficients of the solver to a geom whose asset authored no friction, so that the
        # record always carries the value the geom is built with
        if self.friction is None:
            self.friction = gu.default_friction()
        if self.friction_torsional is None:
            self.friction_torsional = gu.default_friction_torsional()
        if self.friction_rolling is None:
            self.friction_rolling = gu.default_friction_rolling()

    @property
    def is_collision(self) -> bool:
        """
        Get whether the geom takes part in collision detection.
        """
        return bool(self.contype or self.conaffinity)


@dataclass
class JointDescription:
    """Describe one joint of a link, with its per-degree-of-freedom quantities.

    The type gives the number of generalized coordinates and of degrees of freedom the joint holds, and construction
    resolves every quantity the asset leaves unset from that number, including the motion axes of a free joint and
    of a fixed one.
    """

    name: str
    type: JOINT_TYPE
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

    @property
    def n_qs(self) -> int:
        """
        Get the number of generalized coordinates the joint holds.
        """
        return JOINT_TYPE_SIZE[self.type][0]

    @property
    def n_dofs(self) -> int:
        """
        Get the number of degrees of freedom the joint holds.
        """
        return JOINT_TYPE_SIZE[self.type][1]

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


@dataclass
class LinkDescription:
    """Describe one link of an entity, beside the joints that move it and the geoms it holds.

    The inertial quantities hold what the asset authored, and the build estimates each one left unset from the
    geometry. 'is_aligned' records that the build moved the link frame onto the anchor its geoms share.
    """

    name: str
    parent_idx: int | None = None
    root_idx: int | None = None
    pos: np.ndarray = field(default_factory=gu.zero_pos)
    quat: np.ndarray = field(default_factory=gu.identity_quat)
    inertial_pos: np.ndarray | None = None
    inertial_quat: np.ndarray | None = None
    inertia: np.ndarray | None = None
    mass: float | None = None
    invweight: np.ndarray | None = None
    is_robot: bool = False
    is_aligned: bool = False


@dataclass
class EqualityDescription:
    """Describe one constraint tying two objects, or one joint to a constant, using their asset names."""

    name: str
    type: EQUALITY_TYPE
    objs_name: tuple[str, str | None]
    data: np.ndarray
    sol_params: np.ndarray

"""
The rigid contact proxies of the links in the contact of the Newton solver (see genesis/engine/solvers/ipc_contact.py
and genesis/engine/ipc/se3.py).

Every link holding collision geoms of an entity that couples with the other solvers ('RigidMaterial(needs_coup)') owns a
proxy: a massless copy of the pose of the inertial frame of the link (its center of mass and its principal axes of
inertia), the only pose the contact sees. The screw a proxy moves along in the line search turns about the origin of
its frame, the center of mass, as the free motion of a rigid body. The vertices of the proxy are those of the collision
geoms of the link, held in the frame of the link and placed through its inertial frame at every substep, which follows
the inertial parameters the setters write, and published into the contact surface after the cloth. The proxy of a fixed
link takes the pose of its link at the start of every substep.

The proxy of a moving link is an independent state of the Newton solve, held equal to the pose of its link by the
constraint of genesis/engine/ipc/se3.py. The unknowns of the rigid solver are x = dt^2 qacc, from which it reaches the
configuration q1(x) = q0 (+) (dt v0 + x) at the end of the substep, (+) adding a revolute or prismatic coordinate and a
free translation, and turning a free or spherical rotation as R0 Exp(phi), phi its rotation vector in the frame of the
link (see 'func_integrate' in abd/forward_dynamics.py). The link Jacobian B of the constraint is the derivative of the
forward kinematics of q1(x), its rotation columns carrying the right Jacobian Jr(phi) of the retraction. Only the
linearized increment of the proxy is eliminated (dg = S B dx + d_p, see 'func_proxy_maps' in genesis/engine/ipc/se3.py),
the proxy moving on its own screw along the line search, its exact mismatch with its link bounded by the FK-defect
limiter and checked at every trial.

The line search globalizes the solve as CGQ's (see 'func_line_search_trial' in newton_solver.py): a merit weighs the
mismatch against the energy once the proxies lie within the solve tolerance, a filter takes the steps toward feasibility
otherwise, and a line search that runs out of trials with infeasible proxies enters the restoration. Over one iteration,
the slack s of the constraint of every moving proxy joins the unknowns in the rows of the proxy (dg = S B dx + d_p + N s,
the linearized constraint ending at s), priced by an augmented Lagrangian dual . s + s^T K s / 2, K a multiple of the
spatial inertia of its link. The hard iterations resume afterwards, the duals taking K c of the mismatch left.
"""

from typing import NamedTuple

import numpy as np

import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu
from genesis.engine.ipc.se3 import (
    func_fk_defect_prefix_cap,
    func_proxy_constraint,
    func_proxy_maps,
    func_so3_left_jacobian,
)
from genesis.engine.numerics.bcoo import func_read_entry
from genesis.engine.numerics.newton_rows import NewtonRows, NewtonSegment, func_rows_matvec
from genesis.engine.solvers.ipc_contact import IPCState, IPCSurfacePrimitives, surface_area_weights
from genesis.engine.solvers.rigid.abd.inverse_kinematics import func_forward_kinematics_scratch
from genesis.engine.solvers.rigid.data import DOFS, JOINTS, LINKS, QS, DynInfo, DynState, RigidInfo
from genesis.utils.data import ENVS, Data, DataKind, Size, field

# The proxies, one per link holding collision geoms, their vertices, and the entries of the filter of the line search,
# one per Newton iteration of a substep at most
PROXIES = Size("proxies")
PROXY_VERTICES = Size("proxy_vertices")
PROXY_FILTER = Size("proxy_filter")


class ContactProxies(NamedTuple):
    """
    The rigid contact proxies of a scene at build (see the module docstring), one row per proxy or per vertex.

    A proxy carries its link, whether its link is fixed, the number of links from its link to the root of its tree and
    its contact body (-1 for a fixed link). A vertex carries its proxy and its position in the frame of the link.
    `surface` holds the contact primitives of the proxies, their vertices in the order of the vertices of the proxies.
    """

    links_idx: np.ndarray
    is_fixed: np.ndarray
    chain_depth: np.ndarray
    body_idx: np.ndarray
    vertices_proxy_idx: np.ndarray
    vertices_local_pos: np.ndarray
    surface: IPCSurfacePrimitives


class IPCProxiesInfo(Data, kind=DataKind.CONSTANT, shape=(PROXIES,)):
    """The constant parameters of the proxies (see ContactProxies)."""

    link_idx: qd.Tensor = field(int)
    is_fixed: qd.Tensor = field(bool)
    chain_depth: qd.Tensor = field(int)
    body_idx: qd.Tensor = field(int)


class IPCProxyVerticesInfo(Data, kind=DataKind.CONSTANT, shape=(PROXY_VERTICES,)):
    """The vertices of the proxies, their proxy and their position in the frame of its link."""

    proxy_idx: qd.Tensor = field(int)
    local_pos: qd.Tensor = field((float, 3))


class IPCProxyDofsInfo(Data, kind=DataKind.CONSTANT, shape=(DOFS,)):
    """
    The dofs of the rigid solver as the proxies see them.

    `body_idx` is the contact body of the entity of a dof (see IPCBodies in genesis/engine/solvers/ipc_contact.py), -1
    for an entity without moving proxies, and `is_measured` whether the update of the dof measures its convergence, its
    link owning no proxy, whose surface displacement measures it otherwise.
    """

    body_idx: qd.Tensor = field(int)
    is_measured: qd.Tensor = field(bool)


@qd.func(requires_top_level=True)
def func_proxies_scale_dofs(
    envs_mask: qd.Tensor,
    alpha: qd.Tensor,
    dofs_dx: qd.Tensor,
    proxy_dofs_info: IPCProxyDofsInfo,
    is_by_body: qd.template(),
):
    """Scale the Newton direction of the dofs of the environments selected by the step of their environment in `alpha`,
    or with `is_by_body` the dofs of the entities with moving proxies by the step of the group of the contact body of
    their entity (see IPCBodies)."""
    for i_d, i_b in qd.ndrange(*dofs_dx.shape):
        if envs_mask[i_b]:
            scale = gs.qd_float(1.0)
            if qd.static(is_by_body):
                i_body = proxy_dofs_info.body_idx[i_d]
                if i_body >= 0:
                    scale = alpha[i_body, i_b]
            else:
                scale = alpha[i_b]
            dofs_dx[i_d, i_b] = scale * dofs_dx[i_d, i_b]


class IPCProxiesState(Data, kind=DataKind.STATE, shape=(PROXIES, ENVS)):
    """
    The pose of every proxy, at the current iterate of the Newton solve and accepted by it at the end of a substep, and
    its pose at the start of the current substep.

    `inertial_pos` and `inertial_quat` are the inertial frame of its link in the frame of the link, and
    `surface_radius` the largest distance of its vertices from the center of mass, both set at the start of every
    substep (see 'func_proxies_publish_contact').
    """

    pos: qd.Tensor = field((float, 3))
    quat: qd.Tensor = field((float, 4))
    pos_start: qd.Tensor = field((float, 3))
    quat_start: qd.Tensor = field((float, 4))
    inertial_pos: qd.Tensor = field((float, 3), kind=DataKind.DERIVED)
    inertial_quat: qd.Tensor = field((float, 4), kind=DataKind.DERIVED)
    surface_radius: qd.Tensor = field(float, kind=DataKind.DERIVED)


class IPCProxies(Data, kind=DataKind.SCRATCH, shape=(PROXIES, ENVS)):
    """
    The constraint of every moving proxy at the current Newton iterate and its Newton direction.

    `constraint`, the rotation blocks `tangent_rotation` and `normal_rotation` of its maps and `particular` follow
    'func_proxy_maps' in genesis/engine/ipc/se3.py. `pos_iter` and `quat_iter` are the pose at the start of the line
    search, `direction_pos` and `direction_rot` the world twist the step moves it by, `path_limit` the mismatch with its
    link the step keeps within (see 'func_proxies_limit_step'), and `target_pos` the center of mass of its link at the
    current iterate, about which its link Jacobian measures the motion of the link.

    `gradient` and `hessian_particular` are the gradient g of the energy in its rows at the iterate and the product
    H d_p there, which the slope of the merit takes (see 'func_proxies_merit_probe' in newton_solver.py), and
    `hessian` the block H_pp of its rows. The restoration (see the module docstring) prices the slack s of its
    constraint by `dual` . s + s^T K s / 2, K the `metric`, `slack` holding the slack of the Newton direction and
    `slack_search` that of the search direction of the conjugate gradient across its product.
    """

    constraint: qd.Tensor = field((float, 6))
    tangent_rotation: qd.Tensor = field((float, (3, 3)))
    normal_rotation: qd.Tensor = field((float, (3, 3)))
    particular: qd.Tensor = field((float, 6))
    pos_iter: qd.Tensor = field((float, 3))
    quat_iter: qd.Tensor = field((float, 4))
    direction_pos: qd.Tensor = field((float, 3))
    direction_rot: qd.Tensor = field((float, 3))
    path_limit: qd.Tensor = field(float)
    target_pos: qd.Tensor = field((float, 3))
    gradient: qd.Tensor = field((float, 6))
    hessian_particular: qd.Tensor = field((float, 6))
    hessian: qd.Tensor = field((float, (6, 6)))
    metric: qd.Tensor = field((float, (6, 6)))
    dual: qd.Tensor = field((float, 6))
    slack: qd.Tensor = field((float, 6))
    slack_search: qd.Tensor = field((float, 6))


class IPCProxiesEnvs(Data, kind=DataKind.SCRATCH, shape=(ENVS,)):
    """
    The scalars of the proxies of every environment: `max_residual` is the largest surface mismatch |c_t| + r |c_w| of
    its proxies at the current iterate, `trial_max_residual` at the trial point of its line search, and `fk_alpha` the
    step the FK-defect limiter allows (see 'func_proxies_limit_step').

    The line search accepts a step by the rules of 'func_line_search_trial' in newton_solver.py: `merit_rho` and
    `merit_slope` are the weight and the slope of the merit E + rho h once `is_merit_active`, `is_merit_probe` asks for
    them and `is_merit_evaluated` records that they were asked for in the iteration. `n_filter` entries of the filter
    (see IPCProxiesFilter) hold, `is_filter_retry` forcing the filter on the next iteration. `is_restoration` prices the
    slack of the constraints in the iteration, `is_restoration_entered` holds over the iteration that enters it,
    `is_hard_probe` over the hard iterations that follow it until they converge, and `is_dual_update` asks the next
    iteration to update the duals. `penalty` is the price of the slack that the energy at the start point holds, and
    `penalty_trial` that of the trial point. `is_displacement_converged` says whether the Newton direction moves
    everything by less than the tolerance, the step converging only once the FK-defect limiter keeps it whole.
    """

    max_residual: qd.Tensor = field(float)
    trial_max_residual: qd.Tensor = field(float)
    fk_alpha: qd.Tensor = field(float)
    merit_rho: qd.Tensor = field(float)
    merit_slope: qd.Tensor = field(float)
    is_merit_active: qd.Tensor = field(bool)
    is_merit_probe: qd.Tensor = field(bool)
    is_merit_evaluated: qd.Tensor = field(bool)
    n_filter: qd.Tensor = field(int)
    is_filter_retry: qd.Tensor = field(bool)
    is_restoration: qd.Tensor = field(bool)
    is_restoration_entered: qd.Tensor = field(bool)
    is_hard_probe: qd.Tensor = field(bool)
    is_dual_update: qd.Tensor = field(bool)
    penalty: qd.Tensor = field(float)
    penalty_trial: qd.Tensor = field(float)
    is_displacement_converged: qd.Tensor = field(bool)


class IPCProxiesFilter(Data, kind=DataKind.SCRATCH, shape=(PROXY_FILTER, ENVS)):
    """The filter of the line search of every environment, pairs of the largest surface mismatch and of the energy of
    the iterates it accepted while restoring the feasibility of its proxies, one per Newton iteration at most."""

    residual: qd.Tensor = field(float)
    energy: qd.Tensor = field(float)


class ProxyKinematics(Data, kind=DataKind.SCRATCH):
    """
    The forward kinematics of the configuration q1(x) the rigid solver reaches from an iterate of the Newton solve (see
    the module docstring), apart from the state of the rigid solver.

    `qpos` is the configuration, `links_pos`, `links_quat`, `joints_xanchor` and `joints_xaxis` its frames (see
    'func_forward_kinematics_scratch' in abd/inverse_kinematics.py), and `dofs_motion_vel` and `dofs_motion_ang` the
    derivative of the motion of the links along every dof there, the velocity of the point at the world origin and the
    angular velocity.
    """

    qpos: qd.Tensor = field(float, shape=(QS, ENVS))
    links_pos: qd.Tensor = field((float, 3), shape=(LINKS, ENVS))
    links_quat: qd.Tensor = field((float, 4), shape=(LINKS, ENVS))
    joints_xanchor: qd.Tensor = field((float, 3), shape=(JOINTS, ENVS))
    joints_xaxis: qd.Tensor = field((float, 3), shape=(JOINTS, ENVS))
    dofs_motion_vel: qd.Tensor = field((float, 3), shape=(DOFS, ENVS))
    dofs_motion_ang: qd.Tensor = field((float, 3), shape=(DOFS, ENVS))


def build_contact_proxies(links_geoms: list, n_bodies_start: int, d_hat: float) -> ContactProxies:
    """
    Build the rigid contact proxies of the links holding collision geoms (see ContactProxies).

    `links_geoms` lists, per link owning a proxy, the link, the rank of the contact body of its entity among the rigid
    entities with moving proxies (-1 for a fixed link), the depth of the link in its tree and its collision geoms. A
    proxy stands for the merged meshes of its geoms in the frame of its link. The vertices of a moving link belong to body
    `n_bodies_start` plus the rank of their entity, those of a fixed link to none. The contact vertices of a proxy are
    rigid, of no thickness and of no resistance (the cloth sets the resistance of their pairs), with the friction
    coefficient of their geom for coupling.
    """
    links_idx, is_fixed, chain_depth, body_idx = [], [], [], []
    vertices_proxy_idx, vertices_local_pos, vertices_body_idx, vertices_friction_mu = [], [], [], []
    tris_all = []
    n_verts = 0
    for i_proxy, (link, i_body, depth, geoms) in enumerate(links_geoms):
        proxy_verts = []
        for geom in geoms:
            verts = gu.transform_by_trans_quat(geom.init_verts, geom.init_pos, geom.init_quat)
            tris_all.append(geom.init_faces + n_verts)
            proxy_verts.append(verts)
            vertices_friction_mu.append(np.full(len(verts), geom.coup_friction, dtype=gs.np_float))
            n_verts += len(verts)
        proxy_verts = np.concatenate(proxy_verts)
        i_body_proxy = -1 if link.is_fixed else n_bodies_start + i_body
        links_idx.append(link.idx)
        is_fixed.append(link.is_fixed)
        chain_depth.append(depth)
        body_idx.append(i_body_proxy)
        vertices_proxy_idx.append(np.full(len(proxy_verts), i_proxy, dtype=gs.np_int))
        vertices_local_pos.append(proxy_verts)
        vertices_body_idx.append(np.full(len(proxy_verts), i_body_proxy, dtype=gs.np_int))
    pos = np.concatenate(vertices_local_pos)
    tris = np.concatenate(tris_all)
    areas, edges_verts_idx, edges_area_weight, edges_rest_length2, vertices_area_weight = surface_area_weights(
        pos, tris
    )
    surface = IPCSurfacePrimitives(
        vertices_thickness=np.zeros(n_verts, dtype=gs.np_float),
        vertices_d_hat=np.full(n_verts, d_hat, dtype=gs.np_float),
        vertices_area_weight=vertices_area_weight,
        vertices_resistance=np.zeros(n_verts, dtype=gs.np_float),
        vertices_friction_mu=np.concatenate(vertices_friction_mu),
        vertices_body_idx=np.concatenate(vertices_body_idx),
        vertices_is_rigid=np.ones(n_verts, dtype=bool),
        edges_verts_idx=edges_verts_idx,
        edges_area_weight=edges_area_weight,
        edges_rest_length2=edges_rest_length2,
        triangles_verts_idx=tris,
        triangles_area_weight=areas,
    )
    return ContactProxies(
        links_idx=np.array(links_idx, dtype=gs.np_int),
        is_fixed=np.array(is_fixed, dtype=bool),
        chain_depth=np.array(chain_depth, dtype=gs.np_int),
        body_idx=np.array(body_idx, dtype=gs.np_int),
        vertices_proxy_idx=np.concatenate(vertices_proxy_idx),
        vertices_local_pos=pos,
        surface=surface,
    )


# ------------------------------------------------------------------------------------
# --------------------------------- publication --------------------------------------
# ------------------------------------------------------------------------------------


@qd.func
def func_proxy_row(i_p: int, i_b: int, n_proxies: int, newton_rows: NewtonRows) -> int:
    """Return the translation row of a proxy in the segment of the proxies, its rotation row following (see NewtonSegment
    in numerics/newton_rows.py), the proxies of an environment following one another, those of a fixed link unused."""
    return newton_rows.segments_start[qd.static(int(NewtonSegment.PROXIES))] + 2 * (i_b * n_proxies + i_p)


@qd.func
def func_proxy_vertex_pos(
    i_pv: int,
    i_b: int,
    pos: qd.types.vector(3),
    quat: qd.types.vector(4),
    proxies_state: IPCProxiesState,
    proxy_vertices_info: IPCProxyVerticesInfo,
):
    """Return the world position of a vertex of a proxy at the pose (`pos`, `quat`) of its proxy, through the inertial
    frame of its link (see IPCProxiesState)."""
    i_p = proxy_vertices_info.proxy_idx[i_pv]
    local_pos = gu.qd_inv_transform_by_trans_quat(
        proxy_vertices_info.local_pos[i_pv], proxies_state.inertial_pos[i_p, i_b], proxies_state.inertial_quat[i_p, i_b]
    )
    return pos + gu.qd_transform_by_quat(local_pos, quat)


@qd.func
def func_proxy_inertial_frame(
    i_p: int, i_b: int, proxies_info: IPCProxiesInfo, dyn_info: DynInfo, rigid_config: qd.template()
):
    """Return the inertial frame of the link of a proxy in the frame of the link, its center of mass and the rotation of
    its principal axes of inertia."""
    i_l = proxies_info.link_idx[i_p]
    I_l = [i_l, i_b] if qd.static(rigid_config.batch_links_info) else i_l
    _, axes = qd.sym_eig(dyn_info.links.inertial_i[I_l], gs.qd_float)
    # The eigenvectors make a basis of either hand, which a rotation takes right-handed
    if axes.determinant() < 0.0:
        for k in qd.static(range(3)):
            axes[k, 2] = -axes[k, 2]
    quat = gu.qd_transform_quat_by_quat(gu.qd_R_to_quat(axes), dyn_info.links.inertial_quat[I_l])
    return dyn_info.links.inertial_pos[I_l], quat


@qd.func(requires_top_level=True)
def func_proxies_publish_contact(
    vertex_start: int,
    ipc_state: IPCState,
    proxies_state: IPCProxiesState,
    dyn_state: DynState,
    proxies_info: IPCProxiesInfo,
    proxy_vertices_info: IPCProxyVerticesInfo,
    dyn_info: DynInfo,
    rigid_config: qd.template(),
):
    """
    Publish the vertices of the proxies at the start of the substep into their range of the contact surface, from the
    contact vertex `vertex_start` on (see IPCVertices in genesis/engine/solvers/ipc_contact.py).

    The inertial frame of every proxy and its surface radius follow the inertial parameters of its link. The proxy of a
    fixed link takes the pose of the inertial frame of its link, and the others start at their accepted pose. A vertex
    stands at the pose of its proxy at the start, its lagged position too, its proxy being its pivot, without rotation.
    The vertices of a fixed link take no contact terms.
    """
    for i_p, i_b in qd.ndrange(*proxies_state.pos.shape):
        inertial_pos, inertial_quat = func_proxy_inertial_frame(i_p, i_b, proxies_info, dyn_info, rigid_config)
        proxies_state.inertial_pos[i_p, i_b] = inertial_pos
        proxies_state.inertial_quat[i_p, i_b] = inertial_quat
        proxies_state.surface_radius[i_p, i_b] = 0.0
        pos = proxies_state.pos[i_p, i_b]
        quat = proxies_state.quat[i_p, i_b]
        if proxies_info.is_fixed[i_p]:
            i_l = proxies_info.link_idx[i_p]
            pos, quat = gu.qd_transform_pos_quat_by_trans_quat(
                inertial_pos, inertial_quat, dyn_state.links.pos[i_l, i_b], dyn_state.links.quat[i_l, i_b]
            )
            proxies_state.pos[i_p, i_b] = pos
            proxies_state.quat[i_p, i_b] = quat
        proxies_state.pos_start[i_p, i_b] = pos
        proxies_state.quat_start[i_p, i_b] = quat
    for i_pv, i_b in qd.ndrange(proxy_vertices_info.proxy_idx.shape[0], proxies_state.pos.shape[1]):
        i_p = proxy_vertices_info.proxy_idx[i_pv]
        pos = proxies_state.pos_start[i_p, i_b]
        x = func_proxy_vertex_pos(
            i_pv, i_b, pos, proxies_state.quat_start[i_p, i_b], proxies_state, proxy_vertices_info
        )
        qd.atomic_max(proxies_state.surface_radius[i_p, i_b], (x - pos).norm())
        i_v = vertex_start + i_pv
        ipc_state.vertices.pos[i_v, i_b] = x
        ipc_state.vertices.pos_lagged[i_v, i_b] = x
        ipc_state.vertices.displacement[i_v, i_b] = qd.Vector.zero(gs.qd_float, 3)
        ipc_state.vertices.is_fixed[i_v, i_b] = proxies_info.is_fixed[i_p]
        ipc_state.vertices.row[i_v, i_b] = 0
        ipc_state.vertices.pivot[i_v, i_b] = pos
        ipc_state.vertices.pivot_displacement[i_v, i_b] = qd.Vector.zero(gs.qd_float, 3)
        ipc_state.vertices.rotation[i_v, i_b] = qd.Vector.zero(gs.qd_float, 3)


@qd.kernel(fastcache=True)
def kernel_proxies_init_pose(
    envs_mask: qd.Tensor,
    proxies_state: IPCProxiesState,
    dyn_state: DynState,
    proxies_info: IPCProxiesInfo,
    dyn_info: DynInfo,
    rigid_config: qd.template(),
):
    """Set the pose of every proxy of the environments selected to the pose of the inertial frame of its link, its
    constraint vanishing (see genesis/engine/ipc/se3.py)."""
    for i_p, i_b in qd.ndrange(*proxies_state.pos.shape):
        if envs_mask[i_b]:
            i_l = proxies_info.link_idx[i_p]
            inertial_pos, inertial_quat = func_proxy_inertial_frame(i_p, i_b, proxies_info, dyn_info, rigid_config)
            proxies_state.pos[i_p, i_b], proxies_state.quat[i_p, i_b] = gu.qd_transform_pos_quat_by_trans_quat(
                inertial_pos, inertial_quat, dyn_state.links.pos[i_l, i_b], dyn_state.links.quat[i_l, i_b]
            )


# ------------------------------------------------------------------------------------
# ---------------------------------- kinematics --------------------------------------
# ------------------------------------------------------------------------------------


@qd.func
def func_rotation_increment(i_d: int, i_b: int, x: qd.Tensor, dyn_state: DynState, rigid_info: RigidInfo):
    """Return the rotation vector phi = dt w0 + x of the free or spherical rotation whose first dof is i_d, in the frame
    of its link (see the module docstring)."""
    dt = rigid_info.substep_dt[None]
    return qd.Vector(
        [dt * dyn_state.dofs.vel[i_d + k, i_b] + x[i_d + k, i_b] for k in qd.static(range(3))], dt=gs.qd_float
    )


@qd.func(requires_top_level=True)
def func_proxy_kinematics(
    x: qd.Tensor,
    kinematics: ProxyKinematics,
    dyn_state: DynState,
    dyn_info: DynInfo,
    rigid_info: RigidInfo,
    rigid_config: qd.template(),
    has_motions: qd.template(),
):
    """
    Compute the configuration q1(x) the rigid solver reaches from the unknowns `x` of the Newton solve and its forward
    kinematics (see ProxyKinematics), with `has_motions` the motion of the links along every dof there.

    The configuration follows the integration of 'func_integrate' in abd/forward_dynamics.py, the links of a proxy being
    constrained, which keeps them off its midpoint rule (see 'func_proxies_mark_constrained').
    """
    dt = rigid_info.substep_dt[None]
    n_envs = kinematics.qpos.shape[1]
    for i_l, i_b in qd.ndrange(kinematics.links_pos.shape[0], n_envs):
        I_l = [i_l, i_b] if qd.static(rigid_config.batch_links_info) else i_l
        q_start = dyn_info.links.q_start[I_l]
        q_end = dyn_info.links.q_end[I_l]
        dof_start = dyn_info.links.dof_start[I_l]
        if dyn_info.links.n_dofs[I_l] > 0:
            i_j = dyn_info.links.joint_start[I_l]
            I_j = [i_j, i_b] if qd.static(rigid_config.batch_joints_info) else i_j
            joint_type = dyn_info.joints.type[I_j]
            if joint_type == gs.JOINT_TYPE.FREE:
                for k in qd.static(range(3)):
                    kinematics.qpos[q_start + k, i_b] = (
                        rigid_info.qpos[q_start + k, i_b]
                        + dt * dyn_state.dofs.vel[dof_start + k, i_b]
                        + x[dof_start + k, i_b]
                    )
            if joint_type == gs.JOINT_TYPE.SPHERICAL or joint_type == gs.JOINT_TYPE.FREE:
                rot_offset = 3 if joint_type == gs.JOINT_TYPE.FREE else 0
                rot0 = qd.Vector(
                    [rigid_info.qpos[q_start + rot_offset + k, i_b] for k in qd.static(range(4))], dt=gs.qd_float
                )
                phi = func_rotation_increment(dof_start + rot_offset, i_b, x, dyn_state, rigid_info)
                rot = gu.qd_transform_quat_by_quat(gu.qd_rotvec_to_quat(phi), rot0)
                for k in qd.static(range(4)):
                    kinematics.qpos[q_start + rot_offset + k, i_b] = rot[k]
            else:
                for j_ in range(q_end - q_start):
                    kinematics.qpos[q_start + j_, i_b] = (
                        rigid_info.qpos[q_start + j_, i_b]
                        + dt * dyn_state.dofs.vel[dof_start + j_, i_b]
                        + x[dof_start + j_, i_b]
                    )
        else:
            for j_ in range(q_end - q_start):
                kinematics.qpos[q_start + j_, i_b] = rigid_info.qpos[q_start + j_, i_b]
    for i_e, i_b in qd.ndrange(dyn_info.entities.link_start.shape[0], n_envs):
        func_forward_kinematics_scratch(
            i_b,
            i_b,
            i_b,
            i_e,
            0,
            0,
            0,
            kinematics.qpos,
            kinematics.links_pos,
            kinematics.links_quat,
            kinematics.joints_xanchor,
            kinematics.joints_xaxis,
            dyn_state,
            dyn_info,
            rigid_info,
            rigid_config,
        )
    if qd.static(has_motions):
        EPS = rigid_info.EPS[None]
        for i_l, i_b in qd.ndrange(kinematics.links_pos.shape[0], n_envs):
            I_l = [i_l, i_b] if qd.static(rigid_config.batch_links_info) else i_l
            for i_j in range(dyn_info.links.joint_start[I_l], dyn_info.links.joint_end[I_l]):
                I_j = [i_j, i_b] if qd.static(rigid_config.batch_joints_info) else i_j
                joint_type = dyn_info.joints.type[I_j]
                i_d = dyn_info.joints.dof_start[I_j]
                anchor = kinematics.joints_xanchor[i_j, i_b]
                axis = kinematics.joints_xaxis[i_j, i_b]
                if joint_type == gs.JOINT_TYPE.REVOLUTE:
                    kinematics.dofs_motion_ang[i_d, i_b] = axis
                    kinematics.dofs_motion_vel[i_d, i_b] = anchor.cross(axis)
                elif joint_type == gs.JOINT_TYPE.PRISMATIC:
                    kinematics.dofs_motion_ang[i_d, i_b] = qd.Vector.zero(gs.qd_float, 3)
                    kinematics.dofs_motion_vel[i_d, i_b] = axis
                elif joint_type == gs.JOINT_TYPE.FREE or joint_type == gs.JOINT_TYPE.SPHERICAL:
                    rot_offset = 0
                    if joint_type == gs.JOINT_TYPE.FREE:
                        rot_offset = 3
                        for k in qd.static(range(3)):
                            kinematics.dofs_motion_ang[i_d + k, i_b] = qd.Vector.zero(gs.qd_float, 3)
                            kinematics.dofs_motion_vel[i_d + k, i_b] = qd.Vector(
                                [1.0 if j == k else 0.0 for j in qd.static(range(3))], dt=gs.qd_float
                            )
                    # The rotation turns as R0 Exp(phi) in the frame of the link, whose derivative in the world is
                    # R1 Jr(phi), Jr(phi) = Jl(-phi)
                    phi = func_rotation_increment(i_d + rot_offset, i_b, x, dyn_state, rigid_info)
                    rotation = gu.qd_quat_to_R(kinematics.links_quat[i_l, i_b], EPS) @ func_so3_left_jacobian(-phi)
                    for k in qd.static(range(3)):
                        ang = qd.Vector([rotation[0, k], rotation[1, k], rotation[2, k]], dt=gs.qd_float)
                        kinematics.dofs_motion_ang[i_d + rot_offset + k, i_b] = ang
                        kinematics.dofs_motion_vel[i_d + rot_offset + k, i_b] = anchor.cross(ang)


@qd.func
def func_link_twist(
    i_l: int,
    i_b: int,
    depth: int,
    point: qd.types.vector(3),
    src: qd.Tensor,
    kinematics: ProxyKinematics,
    dyn_info: DynInfo,
    rigid_config: qd.template(),
):
    """Return the world twist of a link along a vector `src` over the dofs, the displacement of its point `point` and
    its rotation, summed over the dofs of the `depth` links of its chain (see ProxyKinematics)."""
    motion_vel = qd.Vector.zero(gs.qd_float, 3)
    motion_ang = qd.Vector.zero(gs.qd_float, 3)
    j_l = i_l
    for _ in range(depth):
        J_l = [j_l, i_b] if qd.static(rigid_config.batch_links_info) else j_l
        for i_d in range(dyn_info.links.dof_start[J_l], dyn_info.links.dof_end[J_l]):
            motion_vel = motion_vel + kinematics.dofs_motion_vel[i_d, i_b] * src[i_d, i_b]
            motion_ang = motion_ang + kinematics.dofs_motion_ang[i_d, i_b] * src[i_d, i_b]
        j_l = dyn_info.links.parent_idx[J_l]
    return motion_vel + motion_ang.cross(point), motion_ang


@qd.func
def func_link_wrench_to_dofs(
    i_l: int,
    i_b: int,
    depth: int,
    point: qd.types.vector(3),
    force: qd.types.vector(3),
    torque: qd.types.vector(3),
    out: qd.Tensor,
    kinematics: ProxyKinematics,
    dyn_info: DynInfo,
    rigid_config: qd.template(),
):
    """Add to a vector `out` over the dofs the generalized force of a wrench on a link, its force at its point `point`
    and its torque, the transpose of 'func_link_twist'."""
    torque_origin = torque + point.cross(force)
    j_l = i_l
    for _ in range(depth):
        J_l = [j_l, i_b] if qd.static(rigid_config.batch_links_info) else j_l
        for i_d in range(dyn_info.links.dof_start[J_l], dyn_info.links.dof_end[J_l]):
            qd.atomic_add(
                out[i_d, i_b],
                kinematics.dofs_motion_vel[i_d, i_b].dot(force)
                + kinematics.dofs_motion_ang[i_d, i_b].dot(torque_origin),
            )
        j_l = dyn_info.links.parent_idx[J_l]


@qd.func
def func_proxy_expand(
    i_p: int,
    i_b: int,
    src: qd.Tensor,
    proxies: IPCProxies,
    kinematics: ProxyKinematics,
    proxies_info: IPCProxiesInfo,
    dyn_info: DynInfo,
    rigid_config: qd.template(),
):
    """Return the twist S B src of a moving proxy along a vector `src` over the dofs, translation and rotation (see the
    module docstring), the twin of 'func_proxy_restrict'."""
    twist_pos, twist_rot = func_link_twist(
        proxies_info.link_idx[i_p],
        i_b,
        proxies_info.chain_depth[i_p],
        proxies.target_pos[i_p, i_b],
        src,
        kinematics,
        dyn_info,
        rigid_config,
    )
    return twist_pos, proxies.tangent_rotation[i_p, i_b] @ twist_rot


@qd.func
def func_proxy_restrict(
    i_p: int,
    i_b: int,
    force: qd.types.vector(3),
    torque: qd.types.vector(3),
    out: qd.Tensor,
    proxies: IPCProxies,
    kinematics: ProxyKinematics,
    proxies_info: IPCProxiesInfo,
    dyn_info: DynInfo,
    rigid_config: qd.template(),
):
    """Add to a vector `out` over the dofs the restriction B^T S^T of a wrench on a moving proxy, its force and its torque,
    the twin of 'func_proxy_expand'."""
    func_link_wrench_to_dofs(
        proxies_info.link_idx[i_p],
        i_b,
        proxies_info.chain_depth[i_p],
        proxies.target_pos[i_p, i_b],
        force,
        proxies.tangent_rotation[i_p, i_b].transpose() @ torque,
        out,
        kinematics,
        dyn_info,
        rigid_config,
    )


# ------------------------------------------------------------------------------------
# ----------------------------------- Newton solve -----------------------------------
# ------------------------------------------------------------------------------------


@qd.func(requires_top_level=True)
def func_proxies_init_substep(
    x: qd.Tensor,
    proxies_envs: IPCProxiesEnvs,
    proxies: IPCProxies,
    dyn_state: DynState,
    proxies_info: IPCProxiesInfo,
    proxy_dofs_info: IPCProxyDofsInfo,
    dyn_info: DynInfo,
    rigid_info: RigidInfo,
    rigid_config: qd.template(),
    penalty_ratio: float,
):
    """
    Start the dofs of the entities with moving proxies at rest, x = -dt v0, where the configuration they reach is the one
    at the start of the substep (see the module docstring), constrain the links of the proxies, and start the
    globalization of the line search afresh (see IPCProxiesEnvs).

    The proxies start at the poses their links held when the last substep committed them, so that their constraint
    starts within its solve tolerance, where the FK-defect limiter certifies a full step. The constraint of the links
    keeps them off the midpoint rule of the free bodies, whose position follows another map of x. The metric of the
    restoration of a proxy is `penalty_ratio` times the spatial inertia of its link about its center of mass, its dual
    starting at zero.
    """
    dt = rigid_info.substep_dt[None]
    EPS = rigid_info.EPS[None]
    for i_d, i_b in qd.ndrange(*x.shape):
        if proxy_dofs_info.body_idx[i_d] >= 0:
            x[i_d, i_b] = -dt * dyn_state.dofs.vel[i_d, i_b]
    for i_p, i_b in qd.ndrange(proxies_info.link_idx.shape[0], x.shape[1]):
        if not proxies_info.is_fixed[i_p]:
            i_l = proxies_info.link_idx[i_p]
            I_l = [i_l, i_b] if qd.static(rigid_config.batch_links_info) else i_l
            dyn_state.links.is_constrained[i_l, i_b] = True
            rotation = gu.qd_quat_to_R(dyn_state.links.i_quat[i_l, i_b], EPS)
            inertia = rotation @ dyn_info.links.inertial_i[I_l] @ rotation.transpose()
            metric = qd.Matrix.zero(gs.qd_float, 6, 6)
            for k, j in qd.static(qd.ndrange(3, 3)):
                if qd.static(k == j):
                    metric[k, j] = penalty_ratio * dyn_info.links.inertial_mass[I_l]
                metric[3 + k, 3 + j] = penalty_ratio * inertia[k, j]
            proxies.metric[i_p, i_b] = metric
            proxies.dual[i_p, i_b] = qd.Vector.zero(gs.qd_float, 6)
            proxies.slack[i_p, i_b] = qd.Vector.zero(gs.qd_float, 6)
    for i_b in range(x.shape[1]):
        proxies_envs.is_merit_active[i_b] = False
        proxies_envs.is_merit_probe[i_b] = False
        proxies_envs.is_merit_evaluated[i_b] = False
        proxies_envs.n_filter[i_b] = 0
        proxies_envs.is_filter_retry[i_b] = False
        proxies_envs.is_restoration[i_b] = False
        proxies_envs.is_restoration_entered[i_b] = False
        proxies_envs.is_hard_probe[i_b] = False
        proxies_envs.is_dual_update[i_b] = False
        proxies_envs.penalty[i_b] = 0.0
        proxies_envs.penalty_trial[i_b] = 0.0


@qd.func(requires_top_level=True)
def func_proxies_prepare(
    vertex_start: int,
    envs_mask: qd.Tensor,
    x: qd.Tensor,
    energy_start: qd.Tensor,
    newton_rows: NewtonRows,
    ipc_state: IPCState,
    proxies_envs: IPCProxiesEnvs,
    proxies: IPCProxies,
    proxies_state: IPCProxiesState,
    kinematics: ProxyKinematics,
    dyn_state: DynState,
    proxies_info: IPCProxiesInfo,
    proxy_vertices_info: IPCProxyVerticesInfo,
    dyn_info: DynInfo,
    rigid_info: RigidInfo,
    rigid_config: qd.template(),
):
    """
    Prepare the proxies for the Newton iteration at the iterate `x` of the dofs: the forward kinematics of q1(x) with the
    motions of its dofs, the constraint of every moving proxy and its maps (see 'func_proxy_maps' in
    genesis/engine/ipc/se3.py), the largest surface mismatch of every environment, and the rows of the proxies, whose
    gradients the contact adds to, published for their vertices from the contact vertex `vertex_start` on.

    The duals of the environments selected that ask for it take K c (see IPCProxiesEnvs), and their energy at the start
    point in `energy_start` takes the price of the slack c of their constraints over the restoration alone.
    """
    func_proxy_kinematics(x, kinematics, dyn_state, dyn_info, rigid_info, rigid_config, True)
    n_proxies = proxies_info.link_idx.shape[0]
    n_envs = proxies_state.pos.shape[1]
    for i_b in range(n_envs):
        proxies_envs.max_residual[i_b] = 0.0
        if envs_mask[i_b]:
            energy_start[i_b] = energy_start[i_b] - proxies_envs.penalty[i_b]
            proxies_envs.penalty[i_b] = 0.0
    for i_p, i_b in qd.ndrange(n_proxies, n_envs):
        i_r = func_proxy_row(i_p, i_b, n_proxies, newton_rows)
        for k in qd.static(range(2)):
            newton_rows.rows_env[i_r + k] = i_b
            newton_rows.gradient[i_r + k] = qd.Vector.zero(gs.qd_float, 3)
        if not proxies_info.is_fixed[i_p]:
            i_l = proxies_info.link_idx[i_p]
            target_pos, target_quat = gu.qd_transform_pos_quat_by_trans_quat(
                proxies_state.inertial_pos[i_p, i_b],
                proxies_state.inertial_quat[i_p, i_b],
                kinematics.links_pos[i_l, i_b],
                kinematics.links_quat[i_l, i_b],
            )
            constraint = func_proxy_constraint(
                proxies_state.pos[i_p, i_b], proxies_state.quat[i_p, i_b], target_pos, target_quat
            )
            tangent_rotation, normal_rotation, particular = func_proxy_maps(constraint)
            proxies.target_pos[i_p, i_b] = target_pos
            proxies.constraint[i_p, i_b] = constraint
            proxies.tangent_rotation[i_p, i_b] = tangent_rotation
            proxies.normal_rotation[i_p, i_b] = normal_rotation
            proxies.particular[i_p, i_b] = particular
            residual = func_surface_residual(constraint, proxies_state.surface_radius[i_p, i_b])
            qd.atomic_max(proxies_envs.max_residual[i_b], residual)
            if envs_mask[i_b]:
                metric = proxies.metric[i_p, i_b]
                if proxies_envs.is_dual_update[i_b]:
                    proxies.dual[i_p, i_b] = proxies.dual[i_p, i_b] + metric @ constraint
                if proxies_envs.is_restoration[i_b]:
                    penalty = proxies.dual[i_p, i_b].dot(constraint) + 0.5 * constraint.dot(metric @ constraint)
                    qd.atomic_add(proxies_envs.penalty[i_b], penalty)
                    qd.atomic_add(energy_start[i_b], penalty)
    for i_pv, i_b in qd.ndrange(proxy_vertices_info.proxy_idx.shape[0], n_envs):
        i_p = proxy_vertices_info.proxy_idx[i_pv]
        ipc_state.vertices.row[vertex_start + i_pv, i_b] = func_proxy_row(i_p, i_b, n_proxies, newton_rows)
    for i_b in range(n_envs):
        if envs_mask[i_b]:
            proxies_envs.is_dual_update[i_b] = False
            proxies_envs.is_restoration_entered[i_b] = False


@qd.func
def func_surface_residual(constraint: qd.types.vector(6), radius: float) -> float:
    """Return the surface mismatch |c_t| + r |c_w| of a proxy with its link, r its surface radius."""
    return (
        qd.Vector([constraint[0], constraint[1], constraint[2]]).norm()
        + radius * qd.Vector([constraint[3], constraint[4], constraint[5]]).norm()
    )


@qd.func(requires_top_level=True)
def func_proxies_link_stiffness(
    newton_rows: NewtonRows,
    link_stiffness: qd.Tensor,
    proxies: IPCProxies,
    dyn_state: DynState,
    proxies_info: IPCProxiesInfo,
):
    """
    Set the spatial stiffness of the link of every moving proxy, which the factor of the rigid solver adds to its Hessian
    (see RigidContactStiffness in rigid/global_newton.py): T^T S^T H_pp S T, H_pp the 6 x 6 block of the proxy in the
    rows, S its tangent map and T the map of the spatial displacement of the link about the center of mass of its root,
    as the motion subspace of the dofs, to its twist at the center of mass of the link.
    """
    n_proxies = proxies_info.link_idx.shape[0]
    n_envs = proxies.constraint.shape[1]
    for i_p, i_b in qd.ndrange(n_proxies, n_envs):
        i_l = proxies_info.link_idx[i_p]
        link_stiffness[i_l, i_b] = qd.Matrix.zero(gs.qd_float, 6, 6)
    row_start = newton_rows.segments_start[qd.static(int(NewtonSegment.PROXIES))]
    row_end = newton_rows.segments_start[qd.static(int(NewtonSegment.PROXIES)) + 1]
    for i_entry in range(newton_rows.matrix.n_entries[None]):
        i_r, j_r, block = func_read_entry(newton_rows.matrix, i_entry, block_rows=3, block_cols=3)
        # The blocks of a proxy with itself, its translation row even in the segment, the upper triangle alone
        if i_r >= row_start and i_r < row_end and (j_r - row_start) // 2 == (i_r - row_start) // 2:
            i_pb = (i_r - row_start) // 2
            i_p = i_pb % n_proxies
            i_b = i_pb // n_proxies
            if not proxies_info.is_fixed[i_p]:
                i_l = proxies_info.link_idx[i_p]
                a = (i_r - row_start) % 2
                c = (j_r - row_start) % 2
                # Every entry writes its own block, the transposed one of an off-diagonal entry too
                for k, j in qd.static(qd.ndrange(3, 3)):
                    link_stiffness[i_l, i_b][3 * a + k, 3 * c + j] = block[k, j]
                    if a != c:
                        link_stiffness[i_l, i_b][3 * c + j, 3 * a + k] = block[k, j]
    for i_p, i_b in qd.ndrange(n_proxies, n_envs):
        if not proxies_info.is_fixed[i_p]:
            i_l = proxies_info.link_idx[i_p]
            proxies.hessian[i_p, i_b] = link_stiffness[i_l, i_b]
            lever_skew = gu.qd_vec_to_skew(dyn_state.links.i_pos[i_l, i_b])
            tangent_rotation = proxies.tangent_rotation[i_p, i_b]
            lift = qd.Matrix.identity(gs.qd_float, 6)
            for k, j in qd.static(qd.ndrange(3, 3)):
                lift[k, 3 + j] = -lever_skew[k, j]
                lift[3 + k, 3 + j] = tangent_rotation[k, j]
            link_stiffness[i_l, i_b] = lift.transpose() @ link_stiffness[i_l, i_b] @ lift


@qd.func(requires_top_level=True)
def func_proxies_reduce_gradient(
    envs_mask: qd.Tensor,
    p_dot_ap: qd.Tensor,
    dofs_gradient: qd.Tensor,
    newton_rows: NewtonRows,
    proxies_envs: IPCProxiesEnvs,
    proxies: IPCProxies,
    kinematics: ProxyKinematics,
    proxies_info: IPCProxiesInfo,
    dyn_info: DynInfo,
    rigid_config: qd.template(),
):
    """
    Reduce the gradient of the rows and of the dofs of every environment selected to the unknowns of the Newton solve,
    P^T (g + H d_p), d_p the particular step of the proxies, the rows of the proxies holding the slack of their
    constraints over the restoration alone (see the module docstring), no unknown otherwise.

    The product H d_p of the rows goes through the search direction and its product (see 'func_rows_matvec' in
    numerics/newton_rows.py), which the conjugate gradient starts afresh afterwards, and `p_dot_ap` takes the energy form
    of the product along. The wrench of a proxy, its gradient plus its part of H d_p, restricts to `dofs_gradient`, and
    over the restoration to its slack, N^T (g + H d_p) + dual.
    """
    n_proxies = proxies_info.link_idx.shape[0]
    n_envs = proxies.constraint.shape[1]
    n_rows = newton_rows.segments_start[newton_rows.segments_start.shape[0] - 1]
    row_start = newton_rows.segments_start[qd.static(int(NewtonSegment.PROXIES))]
    row_end = newton_rows.segments_start[qd.static(int(NewtonSegment.PROXIES)) + 1]
    for i_r in range(n_rows):
        newton_rows.p[i_r] = qd.Vector.zero(gs.qd_float, 3)
        newton_rows.Ap[i_r] = qd.Vector.zero(gs.qd_float, 3)
    for i_p, i_b in qd.ndrange(n_proxies, n_envs):
        if not proxies_info.is_fixed[i_p]:
            i_r = func_proxy_row(i_p, i_b, n_proxies, newton_rows)
            particular = proxies.particular[i_p, i_b]
            newton_rows.p[i_r] = qd.Vector([particular[0], particular[1], particular[2]])
            newton_rows.p[i_r + 1] = qd.Vector([particular[3], particular[4], particular[5]])
    func_rows_matvec(newton_rows, envs_mask, p_dot_ap, False)
    for i_r in range(n_rows):
        if i_r < row_start or i_r >= row_end:
            i_b = newton_rows.rows_env[i_r]
            if envs_mask[i_b]:
                newton_rows.gradient[i_r] = newton_rows.gradient[i_r] + newton_rows.Ap[i_r]
    for i_p, i_b in qd.ndrange(n_proxies, n_envs):
        i_r = func_proxy_row(i_p, i_b, n_proxies, newton_rows)
        slack_force = qd.Vector.zero(gs.qd_float, 3)
        slack_torque = qd.Vector.zero(gs.qd_float, 3)
        if envs_mask[i_b] and not proxies_info.is_fixed[i_p]:
            gradient_pos = newton_rows.gradient[i_r]
            gradient_rot = newton_rows.gradient[i_r + 1]
            product_pos = newton_rows.Ap[i_r]
            product_rot = newton_rows.Ap[i_r + 1]
            proxies.gradient[i_p, i_b] = qd.Vector(
                [gradient_pos[0], gradient_pos[1], gradient_pos[2], gradient_rot[0], gradient_rot[1], gradient_rot[2]]
            )
            proxies.hessian_particular[i_p, i_b] = qd.Vector(
                [product_pos[0], product_pos[1], product_pos[2], product_rot[0], product_rot[1], product_rot[2]]
            )
            force = gradient_pos + product_pos
            torque = gradient_rot + product_rot
            func_proxy_restrict(
                i_p, i_b, force, torque, dofs_gradient, proxies, kinematics, proxies_info, dyn_info, rigid_config
            )
            if proxies_envs.is_restoration[i_b]:
                dual = proxies.dual[i_p, i_b]
                slack_force = force + qd.Vector([dual[0], dual[1], dual[2]])
                slack_torque = proxies.normal_rotation[i_p, i_b].transpose() @ torque + qd.Vector(
                    [dual[3], dual[4], dual[5]]
                )
        newton_rows.gradient[i_r] = slack_force
        newton_rows.gradient[i_r + 1] = slack_torque
    for i_r in range(n_rows):
        newton_rows.p[i_r] = qd.Vector.zero(gs.qd_float, 3)
        newton_rows.Ap[i_r] = qd.Vector.zero(gs.qd_float, 3)


@qd.func(requires_top_level=True)
def func_proxies_precondition_rows(
    newton_rows: NewtonRows, proxies_envs: IPCProxiesEnvs, proxies: IPCProxies, proxies_info: IPCProxiesInfo
):
    """
    Set the preconditioner of the rows of the proxies: over the restoration, the inverses of the diagonal blocks of the
    Hessian N^T H_pp N + K of the slack of a moving proxy, and zero otherwise, the rows holding no unknown of the
    conjugate gradient, so that their residual and their search direction stay zero.
    """
    n_proxies = proxies_info.link_idx.shape[0]
    for i_p, i_b in qd.ndrange(n_proxies, proxies.constraint.shape[1]):
        i_r = func_proxy_row(i_p, i_b, n_proxies, newton_rows)
        jacobi_pos = qd.Matrix.zero(gs.qd_float, 3, 3)
        jacobi_rot = qd.Matrix.zero(gs.qd_float, 3, 3)
        if proxies_envs.is_restoration[i_b] and not proxies_info.is_fixed[i_p]:
            hessian = proxies.hessian[i_p, i_b]
            metric = proxies.metric[i_p, i_b]
            normal_rotation = proxies.normal_rotation[i_p, i_b]
            block_pos = qd.Matrix.zero(gs.qd_float, 3, 3)
            block_rot = qd.Matrix.zero(gs.qd_float, 3, 3)
            for k, j in qd.static(qd.ndrange(3, 3)):
                block_pos[k, j] = hessian[k, j] + metric[k, j]
                block_rot[k, j] = hessian[3 + k, 3 + j]
            block_rot = normal_rotation.transpose() @ block_rot @ normal_rotation
            for k, j in qd.static(qd.ndrange(3, 3)):
                block_rot[k, j] = block_rot[k, j] + metric[3 + k, 3 + j]
            jacobi_pos = block_pos.inverse()
            jacobi_rot = block_rot.inverse()
        newton_rows.jacobi[i_r] = jacobi_pos
        newton_rows.jacobi[i_r + 1] = jacobi_rot


@qd.func(requires_top_level=True)
def func_proxies_expand(
    envs_mask: qd.Tensor,
    dofs_p: qd.Tensor,
    newton_rows: NewtonRows,
    proxies_envs: IPCProxiesEnvs,
    proxies: IPCProxies,
    kinematics: ProxyKinematics,
    proxies_info: IPCProxiesInfo,
    dyn_info: DynInfo,
    rigid_config: qd.template(),
):
    """
    Write in the rows of every moving proxy of the environments selected the twist S B p of the search direction
    `dofs_p` of the dofs, plus N y over the restoration, y the slack of the search direction in the rows, which the
    product of the rows carries to the other rows (see 'func_proxies_restrict').
    """
    n_proxies = proxies_info.link_idx.shape[0]
    for i_p, i_b in qd.ndrange(n_proxies, proxies.constraint.shape[1]):
        if envs_mask[i_b] and not proxies_info.is_fixed[i_p]:
            i_r = func_proxy_row(i_p, i_b, n_proxies, newton_rows)
            twist_pos, twist_rot = func_proxy_expand(
                i_p, i_b, dofs_p, proxies, kinematics, proxies_info, dyn_info, rigid_config
            )
            if proxies_envs.is_restoration[i_b]:
                slack_pos = newton_rows.p[i_r]
                slack_rot = newton_rows.p[i_r + 1]
                proxies.slack_search[i_p, i_b] = qd.Vector(
                    [slack_pos[0], slack_pos[1], slack_pos[2], slack_rot[0], slack_rot[1], slack_rot[2]]
                )
                twist_pos = twist_pos + slack_pos
                twist_rot = twist_rot + proxies.normal_rotation[i_p, i_b] @ slack_rot
            newton_rows.p[i_r] = twist_pos
            newton_rows.p[i_r + 1] = twist_rot


@qd.func(requires_top_level=True)
def func_proxies_restrict(
    envs_mask: qd.Tensor,
    dofs_ap: qd.Tensor,
    newton_rows: NewtonRows,
    proxies_envs: IPCProxiesEnvs,
    proxies: IPCProxies,
    kinematics: ProxyKinematics,
    proxies_info: IPCProxiesInfo,
    dyn_info: DynInfo,
    rigid_config: qd.template(),
):
    """
    Restrict the product of the rows of every moving proxy of the environments selected to `dofs_ap`, B^T S^T (H P p),
    and over the restoration to the slack in its rows, N^T (H P p) + K y with y back in the search direction, the rows of
    the proxies being cleared otherwise, as they hold no unknown (see 'func_proxies_expand').
    """
    n_proxies = proxies_info.link_idx.shape[0]
    for i_p, i_b in qd.ndrange(n_proxies, proxies.constraint.shape[1]):
        if envs_mask[i_b]:
            i_r = func_proxy_row(i_p, i_b, n_proxies, newton_rows)
            product_pos = qd.Vector.zero(gs.qd_float, 3)
            product_rot = qd.Vector.zero(gs.qd_float, 3)
            slack_pos = qd.Vector.zero(gs.qd_float, 3)
            slack_rot = qd.Vector.zero(gs.qd_float, 3)
            if not proxies_info.is_fixed[i_p]:
                force = newton_rows.Ap[i_r]
                torque = newton_rows.Ap[i_r + 1]
                func_proxy_restrict(
                    i_p, i_b, force, torque, dofs_ap, proxies, kinematics, proxies_info, dyn_info, rigid_config
                )
                if proxies_envs.is_restoration[i_b]:
                    slack = proxies.slack_search[i_p, i_b]
                    product = proxies.metric[i_p, i_b] @ slack
                    slack_pos = qd.Vector([slack[0], slack[1], slack[2]])
                    slack_rot = qd.Vector([slack[3], slack[4], slack[5]])
                    product_pos = force + qd.Vector([product[0], product[1], product[2]])
                    product_rot = proxies.normal_rotation[i_p, i_b].transpose() @ torque + qd.Vector(
                        [product[3], product[4], product[5]]
                    )
            newton_rows.Ap[i_r] = product_pos
            newton_rows.Ap[i_r + 1] = product_rot
            newton_rows.p[i_r] = slack_pos
            newton_rows.p[i_r + 1] = slack_rot


@qd.func(requires_top_level=True)
def func_proxies_init_line_search(
    envs_mask: qd.Tensor,
    max_displacement: qd.Tensor,
    dofs_dx: qd.Tensor,
    newton_rows: NewtonRows,
    proxies_envs: IPCProxiesEnvs,
    proxies: IPCProxies,
    proxies_state: IPCProxiesState,
    kinematics: ProxyKinematics,
    proxies_info: IPCProxiesInfo,
    dyn_info: DynInfo,
    rigid_config: qd.template(),
):
    """
    Set the Newton direction of every moving proxy of the environments selected, S B dx + d_p from the direction `dofs_dx`
    of the dofs (see the module docstring), plus N s over the restoration, s the slack minus the solution of its rows,
    from its pose at the start of the line search.

    The surface displacement |dt| + r |dw| of a proxy adds to the entry of its environment in `max_displacement`, the
    world displacement that measures the convergence of the rigid links.
    """
    n_proxies = proxies_info.link_idx.shape[0]
    for i_p, i_b in qd.ndrange(n_proxies, proxies.constraint.shape[1]):
        if envs_mask[i_b] and not proxies_info.is_fixed[i_p]:
            twist_pos, twist_rot = func_proxy_expand(
                i_p, i_b, dofs_dx, proxies, kinematics, proxies_info, dyn_info, rigid_config
            )
            particular = proxies.particular[i_p, i_b]
            direction_pos = twist_pos + qd.Vector([particular[0], particular[1], particular[2]])
            direction_rot = twist_rot + qd.Vector([particular[3], particular[4], particular[5]])
            slack = qd.Vector.zero(gs.qd_float, 6)
            if proxies_envs.is_restoration[i_b]:
                i_r = func_proxy_row(i_p, i_b, n_proxies, newton_rows)
                slack_pos = -newton_rows.solution[i_r]
                slack_rot = -newton_rows.solution[i_r + 1]
                slack = qd.Vector([slack_pos[0], slack_pos[1], slack_pos[2], slack_rot[0], slack_rot[1], slack_rot[2]])
                direction_pos = direction_pos + slack_pos
                direction_rot = direction_rot + proxies.normal_rotation[i_p, i_b] @ slack_rot
            proxies.slack[i_p, i_b] = slack
            proxies.direction_pos[i_p, i_b] = direction_pos
            proxies.direction_rot[i_p, i_b] = direction_rot
            proxies.pos_iter[i_p, i_b] = proxies_state.pos[i_p, i_b]
            proxies.quat_iter[i_p, i_b] = proxies_state.quat[i_p, i_b]
            qd.atomic_max(
                max_displacement[i_b],
                direction_pos.norm() + proxies_state.surface_radius[i_p, i_b] * direction_rot.norm(),
            )


@qd.func(requires_top_level=True)
def func_proxies_limit_step(
    envs_mask: qd.Tensor,
    dofs_dx: qd.Tensor,
    proxies_envs: IPCProxiesEnvs,
    proxies: IPCProxies,
    proxies_state: IPCProxiesState,
    kinematics: ProxyKinematics,
    proxies_info: IPCProxiesInfo,
    dyn_info: DynInfo,
    rigid_info: RigidInfo,
    rigid_config: qd.template(),
    path_tolerance: float,
):
    """
    Bound the step of every environment selected by the FK-defect limiter: the largest step along which the surface
    mismatch of every moving proxy with its link stays within its path limit, max(`path_tolerance`, h0), h0 its mismatch
    at the start (see 'func_fk_defect_prefix_cap' in genesis/engine/ipc/se3.py).

    The mismatch grows at most as the curvature |dw|^2 r of the screw of the proxy plus that of its link along the
    retraction of the dofs. The latter bounds every chain of dofs by 2 omega s, omega the summed rotation rates and s the
    summed speeds of the point at the surface radius from the joints, as the articulated trajectories of CGQ, a joint of
    which reaches the point through the norms of the frames of the links, from the origin of the link of the proxy past
    its center of mass. A free or spherical rotation, whose axis turns along the retraction R0 Exp(phi + alpha dphi),
    adds |dphi|^2 / 2 times its reach, which bounds the derivative of its angular velocity Jr(phi) dphi along the step.
    """
    for i_b in range(proxies_envs.fk_alpha.shape[0]):
        proxies_envs.fk_alpha[i_b] = 1.0
    for i_p, i_b in qd.ndrange(proxies_info.link_idx.shape[0], proxies.constraint.shape[1]):
        if envs_mask[i_b] and not proxies_info.is_fixed[i_p]:
            radius = proxies_state.surface_radius[i_p, i_b]
            h0 = func_surface_residual(proxies.constraint[i_p, i_b], radius)
            limit = qd.max(path_tolerance, h0)
            proxies.path_limit[i_p, i_b] = limit
            reach = radius + proxies_state.inertial_pos[i_p, i_b].norm()
            speed = gs.qd_float(0.0)
            omega = gs.qd_float(0.0)
            extra = gs.qd_float(0.0)
            j_l = proxies_info.link_idx[i_p]
            for _ in range(proxies_info.chain_depth[i_p]):
                J_l = [j_l, i_b] if qd.static(rigid_config.batch_links_info) else j_l
                travel = gs.qd_float(0.0)
                for i_j in range(dyn_info.links.joint_start[J_l], dyn_info.links.joint_end[J_l]):
                    I_j = [i_j, i_b] if qd.static(rigid_config.batch_joints_info) else i_j
                    joint_type = dyn_info.joints.type[I_j]
                    i_d = dyn_info.joints.dof_start[I_j]
                    reach_joint = reach + dyn_info.joints.pos[I_j].norm()
                    if joint_type == gs.JOINT_TYPE.REVOLUTE:
                        speed = speed + qd.abs(dofs_dx[i_d, i_b]) * reach_joint
                        omega = omega + qd.abs(dofs_dx[i_d, i_b])
                    elif joint_type == gs.JOINT_TYPE.PRISMATIC:
                        q_start = dyn_info.joints.q_start[I_j]
                        q_rel = kinematics.qpos[q_start, i_b] - rigid_info.qpos0[q_start, i_b]
                        travel = travel + qd.max(qd.abs(q_rel), qd.abs(q_rel + dofs_dx[i_d, i_b]))
                        speed = speed + qd.abs(dofs_dx[i_d, i_b])
                    elif joint_type == gs.JOINT_TYPE.FREE or joint_type == gs.JOINT_TYPE.SPHERICAL:
                        rot_offset = 3 if joint_type == gs.JOINT_TYPE.FREE else 0
                        rate = qd.Vector([dofs_dx[i_d + rot_offset + k, i_b] for k in qd.static(range(3))]).norm()
                        speed = speed + rate * reach_joint
                        omega = omega + rate
                        extra = extra + 0.5 * rate * rate * reach_joint
                    reach = reach_joint
                reach = reach + travel + dyn_info.links.pos[J_l].norm()
                j_l = dyn_info.links.parent_idx[J_l]
            rate_proxy = proxies.direction_rot[i_p, i_b].norm()
            curvature = rate_proxy * rate_proxy * radius + 2.0 * omega * speed + extra
            h_slack = func_surface_residual(proxies.slack[i_p, i_b], radius)
            alpha = func_fk_defect_prefix_cap(h0, h_slack, curvature, limit)
            qd.atomic_min(proxies_envs.fk_alpha[i_b], alpha)


@qd.func(requires_top_level=True)
def func_proxies_scale_direction(
    envs_mask: qd.Tensor, alpha: qd.Tensor, proxies: IPCProxies, proxies_info: IPCProxiesInfo, is_by_body: qd.template()
):
    """Scale the Newton direction of every moving proxy of the environments selected by the step of its environment in
    `alpha`, or with `is_by_body` by that of the group of its contact body (see IPCBodies)."""
    for i_p, i_b in qd.ndrange(proxies_info.link_idx.shape[0], proxies.constraint.shape[1]):
        if envs_mask[i_b] and not proxies_info.is_fixed[i_p]:
            scale = gs.qd_float(0.0)
            if qd.static(is_by_body):
                i_body = proxies_info.body_idx[i_p]
                scale = alpha[i_body, i_b]
            else:
                scale = alpha[i_b]
            proxies.direction_pos[i_p, i_b] = scale * proxies.direction_pos[i_p, i_b]
            proxies.direction_rot[i_p, i_b] = scale * proxies.direction_rot[i_p, i_b]


@qd.func(requires_top_level=True)
def func_proxies_publish_sweep(
    vertex_start: int,
    envs_mask: qd.Tensor,
    ipc_state: IPCState,
    proxies: IPCProxies,
    proxies_info: IPCProxiesInfo,
    proxy_vertices_info: IPCProxyVerticesInfo,
):
    """Publish the sweep of the vertices of every moving proxy of the environments selected along its Newton direction,
    the screw of its pose (see IPCVertices in genesis/engine/solvers/ipc_contact.py)."""
    for i_pv, i_b in qd.ndrange(proxy_vertices_info.proxy_idx.shape[0], proxies.constraint.shape[1]):
        i_p = proxy_vertices_info.proxy_idx[i_pv]
        if envs_mask[i_b] and not proxies_info.is_fixed[i_p]:
            i_v = vertex_start + i_pv
            pivot = proxies.pos_iter[i_p, i_b]
            direction_pos = proxies.direction_pos[i_p, i_b]
            direction_rot = proxies.direction_rot[i_p, i_b]
            lever = ipc_state.vertices.pos[i_v, i_b] - pivot
            ipc_state.vertices.pivot[i_v, i_b] = pivot
            ipc_state.vertices.pivot_displacement[i_v, i_b] = direction_pos
            ipc_state.vertices.rotation[i_v, i_b] = direction_rot
            ipc_state.vertices.displacement[i_v, i_b] = (
                direction_pos + gu.qd_transform_by_quat(lever, gu.qd_rotvec_to_quat(direction_rot)) - lever
            )


@qd.func(requires_top_level=True)
def func_proxies_merit_slope(
    envs_mask: qd.Tensor,
    dofs_dx: qd.Tensor,
    newton_rows: NewtonRows,
    proxies_envs: IPCProxiesEnvs,
    proxies: IPCProxies,
    kinematics: ProxyKinematics,
    proxies_info: IPCProxiesInfo,
    dyn_info: DynInfo,
    rigid_config: qd.template(),
):
    """
    Add to the slope of the merit of every environment selected (see IPCProxiesEnvs) the terms of its moving proxies in
    the derivative g . d of the energy along the Newton direction d, from the reduced slope over its unknowns.

    A proxy adds g_p . d_p, d_p its own direction, takes off (g_p + H d_p) . S B dx, the part of its wrench that the
    reduced gradient of the dofs holds, and takes off d_p . (H d_c), the part of H d_p that the reduced gradient of
    the other rows holds, the product of the rows with the direction d_c of the other rows standing in their search
    direction.
    """
    n_proxies = proxies_info.link_idx.shape[0]
    for i_p, i_b in qd.ndrange(n_proxies, proxies.constraint.shape[1]):
        if envs_mask[i_b] and not proxies_info.is_fixed[i_p]:
            i_r = func_proxy_row(i_p, i_b, n_proxies, newton_rows)
            twist_pos, twist_rot = func_proxy_expand(
                i_p, i_b, dofs_dx, proxies, kinematics, proxies_info, dyn_info, rigid_config
            )
            gradient = proxies.gradient[i_p, i_b]
            wrench = gradient + proxies.hessian_particular[i_p, i_b]
            particular = proxies.particular[i_p, i_b]
            direction_pos = proxies.direction_pos[i_p, i_b]
            direction_rot = proxies.direction_rot[i_p, i_b]
            product_pos = newton_rows.Ap[i_r]
            product_rot = newton_rows.Ap[i_r + 1]
            slope = gs.qd_float(0.0)
            for k in qd.static(range(3)):
                slope = (
                    slope
                    + gradient[k] * direction_pos[k]
                    + gradient[3 + k] * direction_rot[k]
                    - wrench[k] * twist_pos[k]
                    - wrench[3 + k] * twist_rot[k]
                    - particular[k] * product_pos[k]
                    - particular[3 + k] * product_rot[k]
                )
            qd.atomic_add(proxies_envs.merit_slope[i_b], slope)


@qd.func(requires_top_level=True)
def func_proxies_init_merit(
    envs_mask: qd.Tensor,
    energy_start: qd.Tensor,
    proxies_envs: IPCProxiesEnvs,
    ipc_state: IPCState,
    proxies_info: IPCProxiesInfo,
    solve_tolerance: float,
    residual_floor: float,
    energy_roundoff: float,
    has_ccd_partition: qd.template(),
):
    """
    Set the merit E + rho h of every environment selected from the slope g . d of its energy in `merit_slope`, h the
    largest surface mismatch of its proxies, as the merit of CGQ.

    The merit holds once h lies in (`residual_floor`, `solve_tolerance`], the step decreases it, and the slope exceeds
    the round-off of the energy, `energy_roundoff` (1 + |E0|). Its weight rho makes its slope g . d - rho h_rate as
    negative as g . d is positive, h_rate the decrease of h over the step, h times the scale of the direction of the
    proxies (the smaller of the FK-defect step and of the steps of their contact bodies).
    """
    for i_b in range(proxies_envs.merit_slope.shape[0]):
        if envs_mask[i_b]:
            scale = proxies_envs.fk_alpha[i_b]
            if qd.static(has_ccd_partition):
                for i_p in range(proxies_info.link_idx.shape[0]):
                    if not proxies_info.is_fixed[i_p]:
                        i_body = proxies_info.body_idx[i_p]
                        scale = qd.min(scale, ipc_state.bodies.alpha[i_body, i_b])
            h0 = proxies_envs.max_residual[i_b]
            h_rate = scale * h0
            slope = proxies_envs.merit_slope[i_b]
            slope_floor = energy_roundoff * (1.0 + qd.abs(energy_start[i_b]))
            rho = gs.qd_float(0.0)
            is_merit_active = (
                h0 > residual_floor and h0 <= solve_tolerance and h_rate > residual_floor and slope > slope_floor
            )
            if is_merit_active:
                rho = (slope + qd.max(slope_floor, qd.abs(slope))) / h_rate
                proxies_envs.merit_slope[i_b] = slope - rho * h_rate
            proxies_envs.merit_rho[i_b] = rho
            proxies_envs.is_merit_active[i_b] = is_merit_active
            proxies_envs.is_merit_probe[i_b] = False


@qd.func(requires_top_level=True)
def func_proxies_step_forward(
    vertex_start: int,
    envs_mask: qd.Tensor,
    alpha: qd.Tensor,
    ipc_state: IPCState,
    proxies: IPCProxies,
    proxies_state: IPCProxiesState,
    proxies_info: IPCProxiesInfo,
    proxy_vertices_info: IPCProxyVerticesInfo,
):
    """Move every moving proxy of the environments selected along its screw to the trial point of the step `alpha` of
    its environment, (t + alpha dt, Exp(alpha dw) R), and publish the positions of its vertices."""
    for i_p, i_b in qd.ndrange(proxies_info.link_idx.shape[0], proxies.constraint.shape[1]):
        if envs_mask[i_b] and not proxies_info.is_fixed[i_p]:
            step = alpha[i_b]
            proxies_state.pos[i_p, i_b] = proxies.pos_iter[i_p, i_b] + step * proxies.direction_pos[i_p, i_b]
            proxies_state.quat[i_p, i_b] = gu.qd_transform_quat_by_quat(
                proxies.quat_iter[i_p, i_b], gu.qd_rotvec_to_quat(step * proxies.direction_rot[i_p, i_b])
            )
    for i_pv, i_b in qd.ndrange(proxy_vertices_info.proxy_idx.shape[0], proxies.constraint.shape[1]):
        i_p = proxy_vertices_info.proxy_idx[i_pv]
        if envs_mask[i_b] and not proxies_info.is_fixed[i_p]:
            i_v = vertex_start + i_pv
            pos = proxies_state.pos[i_p, i_b]
            ipc_state.vertices.pos[i_v, i_b] = func_proxy_vertex_pos(
                i_pv, i_b, pos, proxies_state.quat[i_p, i_b], proxies_state, proxy_vertices_info
            )
            ipc_state.vertices.pivot[i_v, i_b] = pos


@qd.func(requires_top_level=True)
def func_proxies_trial_residual(
    envs_mask: qd.Tensor,
    x: qd.Tensor,
    proxies_envs: IPCProxiesEnvs,
    proxies: IPCProxies,
    proxies_state: IPCProxiesState,
    kinematics: ProxyKinematics,
    dyn_state: DynState,
    proxies_info: IPCProxiesInfo,
    dyn_info: DynInfo,
    rigid_info: RigidInfo,
    rigid_config: qd.template(),
    guard_tolerance: float,
    guard_errno: int,
    errno: qd.Tensor,
):
    """
    Evaluate the exact surface mismatch of every moving proxy of the environments selected with its link at the trial
    point, the dofs at `x`, its largest one in every environment.

    A mismatch beyond the path limit of its proxy, past the relative round-off `guard_tolerance`, breaks the certificate
    of the FK-defect limiter (see 'func_proxies_limit_step'): it sets the flag `guard_errno` in the errno of its
    environment, never a shorter step. Over the restoration, the price of the slack c of the constraints at the trial
    point adds up in the environment (see IPCProxiesEnvs).
    """
    func_proxy_kinematics(x, kinematics, dyn_state, dyn_info, rigid_info, rigid_config, False)
    for i_b in range(proxies_envs.trial_max_residual.shape[0]):
        if envs_mask[i_b]:
            proxies_envs.trial_max_residual[i_b] = 0.0
            proxies_envs.penalty_trial[i_b] = 0.0
    for i_p, i_b in qd.ndrange(proxies_info.link_idx.shape[0], proxies.constraint.shape[1]):
        if envs_mask[i_b] and not proxies_info.is_fixed[i_p]:
            i_l = proxies_info.link_idx[i_p]
            target_pos, target_quat = gu.qd_transform_pos_quat_by_trans_quat(
                proxies_state.inertial_pos[i_p, i_b],
                proxies_state.inertial_quat[i_p, i_b],
                kinematics.links_pos[i_l, i_b],
                kinematics.links_quat[i_l, i_b],
            )
            constraint = func_proxy_constraint(
                proxies_state.pos[i_p, i_b], proxies_state.quat[i_p, i_b], target_pos, target_quat
            )
            residual = func_surface_residual(constraint, proxies_state.surface_radius[i_p, i_b])
            qd.atomic_max(proxies_envs.trial_max_residual[i_b], residual)
            path_limit = proxies.path_limit[i_p, i_b]
            if residual > path_limit + guard_tolerance * (1.0 + path_limit):
                qd.atomic_or(errno[i_b], guard_errno)
            if proxies_envs.is_restoration[i_b]:
                metric = proxies.metric[i_p, i_b]
                penalty = proxies.dual[i_p, i_b].dot(constraint) + 0.5 * constraint.dot(metric @ constraint)
                qd.atomic_add(proxies_envs.penalty_trial[i_b], penalty)

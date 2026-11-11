"""
Incremental potential contact (IPC) of the cloth solved by the Newton solver against the fixed planes of the scene.

Every vertex of the cloth and every plane form a pair. Its barrier acts once the signed distance d of the vertex to the
plane lies between the contact thickness xi of the vertex and xi + d_hat, and its friction acts on the pairs of the
previous substep (see IPCHalfplanePairs). The energy of a pair is dt^2 w d_hat times its barrier plus its friction, w
being the area weight of the vertex, the area of the surface it stands for. A step of the Newton solve stays short of
the time of contact of every pair its sweep brings into the band.
"""

import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu
from genesis.engine.ipc.barrier import (
    func_barrier,
    func_barrier_normal_force,
    func_point_halfplane_barrier_gradient_hessian,
)
from genesis.engine.ipc.ccd import func_point_halfplane_candidate_reach, func_point_halfplane_ccd
from genesis.engine.ipc.friction import (
    func_friction_energy,
    func_friction_gradient_hessian,
    func_point_halfplane_friction_frame,
)
from genesis.engine.solvers.rigid.data import DynInfo, DynState
from genesis.utils.data import ENVS, Data, DataKind, Size, field
from genesis.utils.misc import qd_isfinite

# The vertices of the cloth (see FEMVertices in fem_solver.py) and the fixed planes of the scene
VERTICES = Size("vertices")
HALFPLANES = Size("halfplanes")


class IPCHalfplanes(Data, kind=DataKind.SCRATCH, shape=(HALFPLANES, ENVS)):
    """The fixed planes of the scene over the substep, a point of each one and its unit normal, toward its free side."""

    pos: qd.Tensor = field((float, 3))
    normal: qd.Tensor = field((float, 3))


class IPCHalfplanePairs(Data, kind=DataKind.STATE, shape=(VERTICES, HALFPLANES, ENVS)):
    """
    The pairs of a vertex of the cloth and a plane.

    `is_candidate` says whether the sweep of the vertex along the last Newton direction reaches the band of the plane,
    the pairs the step and the friction of the next substep take. `friction_force` is the normal force of the barrier
    of a pair at the start of the substep, the friction pairs being the candidates of the previous substep that lie in
    the band there, and zero for the other pairs.
    """

    is_candidate: qd.Tensor = field(bool)
    friction_force: qd.Tensor = field(float, kind=DataKind.SCRATCH)


class IPCState(Data, kind=DataKind.STATE):
    """Runtime state of the contact of the cloth with the fixed planes of the scene."""

    halfplanes: IPCHalfplanes
    pairs: IPCHalfplanePairs


class IPCInfo(Data, kind=DataKind.CONSTANT):
    """
    Model parameters of the contact of the cloth (see 'gs.options.IPCOptions').

    `halfplanes_geom_idx` is the rigid geom of every plane, `ccd_eta` the fraction of the gap above the thickness that a
    step keeps at the time of contact of a pair.
    """

    halfplanes_geom_idx: qd.Tensor = field(int, shape=(HALFPLANES,))
    d_hat: qd.Tensor = field(float, shape=())
    friction_eps_v: qd.Tensor = field(float, shape=())
    ccd_eta: qd.Tensor = field(float, shape=())


@qd.func(requires_top_level=True)
def func_halfplanes_init_substep(
    vertices_pos: qd.Tensor,
    vertices_contact_thickness: qd.Tensor,
    vertices_contact_resistance: qd.Tensor,
    ipc_state: IPCState,
    dyn_state: DynState,
    ipc_info: IPCInfo,
    dyn_info: DynInfo,
    penetration_errno: int,
    errno: qd.Tensor,
):
    """
    Place the planes for the substep, and set the normal force of the friction pairs from the positions at its start.

    A vertex starting within the thickness of a plane leaves the barrier undefined, which sets the flag
    `penetration_errno` in the errno of its environment.
    """
    for i_h, i_b in qd.ndrange(*ipc_state.halfplanes.pos.shape):
        i_g = ipc_info.halfplanes_geom_idx[i_h]
        geom_quat = dyn_state.geoms.quat[i_g, i_b]
        normal = gu.qd_transform_by_quat(
            qd.Vector([dyn_info.geoms.data[i_g][0], dyn_info.geoms.data[i_g][1], dyn_info.geoms.data[i_g][2]]),
            geom_quat,
        )
        ipc_state.halfplanes.pos[i_h, i_b] = dyn_state.geoms.pos[i_g, i_b]
        ipc_state.halfplanes.normal[i_h, i_b] = normal.normalized()
    d_hat = ipc_info.d_hat[None]
    for i_v, i_h, i_b in qd.ndrange(*ipc_state.pairs.is_candidate.shape):
        thickness = vertices_contact_thickness[i_v]
        normal = ipc_state.halfplanes.normal[i_h, i_b]
        distance = normal.dot(vertices_pos[i_v, i_b] - ipc_state.halfplanes.pos[i_h, i_b])
        if not (distance > thickness):
            qd.atomic_or(errno[i_b], penetration_errno)
        friction_force = gs.qd_float(0.0)
        if ipc_state.pairs.is_candidate[i_v, i_h, i_b] and distance < thickness + d_hat and distance > thickness:
            friction_force = func_barrier_normal_force(
                distance * distance, d_hat, thickness, vertices_contact_resistance[i_v]
            )
        ipc_state.pairs.friction_force[i_v, i_h, i_b] = friction_force


@qd.func
def func_halfplanes_energy(
    i_v: int,
    i_b: int,
    x: qd.types.vector(3),
    x_lagged: qd.types.vector(3),
    contact_thickness: float,
    contact_area_weight: float,
    friction_mu: float,
    contact_resistance: float,
    dt: float,
    ipc_state: IPCState,
    ipc_info: IPCInfo,
) -> float:
    """Compute the contact energy of a vertex of the cloth at `x` against every plane, `x_lagged` being its position at
    the start of the substep (see the module docstring)."""
    d_hat = ipc_info.d_hat[None]
    energy = gs.qd_float(0.0)
    for i_h in range(ipc_state.halfplanes.pos.shape[0]):
        normal = ipc_state.halfplanes.normal[i_h, i_b]
        distance = normal.dot(x - ipc_state.halfplanes.pos[i_h, i_b])
        if distance > contact_thickness and distance < contact_thickness + d_hat:
            barrier, _, _ = func_barrier(distance * distance, d_hat, contact_thickness, contact_resistance)
            energy += barrier
        friction_force = ipc_state.pairs.friction_force[i_v, i_h, i_b]
        if friction_force > 0.0:
            basis, weights = func_point_halfplane_friction_frame(normal)
            energy += func_friction_energy(
                basis,
                weights,
                qd.Matrix.rows([x - x_lagged]),
                friction_force,
                friction_mu,
                ipc_info.friction_eps_v[None] * dt,
            )
    return dt * dt * contact_area_weight * d_hat * energy


@qd.func
def func_halfplanes_gradient_hessian(
    i_v: int,
    i_b: int,
    x: qd.types.vector(3),
    x_lagged: qd.types.vector(3),
    contact_thickness: float,
    contact_area_weight: float,
    friction_mu: float,
    contact_resistance: float,
    dt: float,
    ipc_state: IPCState,
    ipc_info: IPCInfo,
):
    """Compute the gradient and the positive semi-definite Hessian of the contact energy of a vertex of the cloth (see
    func_halfplanes_energy)."""
    d_hat = ipc_info.d_hat[None]
    gradient = qd.Vector.zero(gs.qd_float, 3)
    hessian = qd.Matrix.zero(gs.qd_float, 3, 3)
    for i_h in range(ipc_state.halfplanes.pos.shape[0]):
        normal = ipc_state.halfplanes.normal[i_h, i_b]
        distance = normal.dot(x - ipc_state.halfplanes.pos[i_h, i_b])
        if distance > contact_thickness and distance < contact_thickness + d_hat:
            gradient_pair, hessian_pair = func_point_halfplane_barrier_gradient_hessian(
                distance, normal, d_hat, contact_thickness, contact_resistance
            )
            gradient += gradient_pair
            hessian += hessian_pair
        friction_force = ipc_state.pairs.friction_force[i_v, i_h, i_b]
        if friction_force > 0.0:
            basis, weights = func_point_halfplane_friction_frame(normal)
            gradient_pair, hessian_pair = func_friction_gradient_hessian(
                basis,
                weights,
                qd.Matrix.rows([x - x_lagged]),
                friction_force,
                friction_mu,
                ipc_info.friction_eps_v[None] * dt,
            )
            gradient += gradient_pair
            hessian += hessian_pair
    scale = dt * dt * contact_area_weight * d_hat
    return scale * gradient, scale * hessian


@qd.func
def func_halfplanes_sweep(
    i_v: int,
    i_b: int,
    x: qd.types.vector(3),
    displacement: qd.types.vector(3),
    contact_thickness: float,
    ipc_state: IPCState,
    ipc_info: IPCInfo,
) -> float:
    """
    Flag the planes the sweep of a vertex of the cloth from `x` along `displacement` brings into their band, and return
    the fraction of the sweep the vertex may travel, 1 when no candidate bounds it.

    The fraction is the conservative time of contact of the vertex with its candidates (see func_point_halfplane_ccd).
    """
    band = ipc_info.d_hat[None] + contact_thickness
    fraction = gs.qd_float(1.0)
    for i_h in range(ipc_state.halfplanes.pos.shape[0]):
        normal = ipc_state.halfplanes.normal[i_h, i_b]
        plane_pos = ipc_state.halfplanes.pos[i_h, i_b]
        is_candidate = func_point_halfplane_candidate_reach(x, x + displacement, plane_pos, normal, 0.0, band)
        ipc_state.pairs.is_candidate[i_v, i_h, i_b] = is_candidate
        if is_candidate:
            toi, is_bounding = func_point_halfplane_ccd(
                x, displacement, plane_pos, normal, ipc_info.ccd_eta[None], contact_thickness
            )
            if is_bounding and qd_isfinite(toi):
                fraction = qd.min(fraction, toi)
    return fraction

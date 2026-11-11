"""
Incremental potential contact (IPC) of the cloth solved by the Newton solver, with itself, with the other cloths and
with the fixed planes of the scene.

A pair is a vertex against a plane (point-halfplane), a vertex against a triangle (point-triangle, PT) or an edge against
an edge (edge-edge, EE), PT and EE pairs reducing to their closest features (see genesis/engine/ipc/distance.py). Its
barrier acts once the distance d of the pair lies between its contact thickness xi and xi + d_hat, and its friction acts
on the pairs in that band at the start of the substep, their frames and normal forces frozen there. The energy of a
pair is dt^2 w times its barrier plus its friction, w being its area weight: the area of the surface the vertex stands
for times d_hat for a point-halfplane pair, and d_hat / 4 times the smaller area weight of its two primitives otherwise.

The PT and EE pairs are the candidates that a broad phase finds before every use: the pairs near enough to enter the
band over the sweep of the vertices along the Newton direction, or at the start point of the substep. A step of the
Newton solve stays short of the time of contact of every candidate its sweep brings into the band, every cloth being a
contact body whose direction shrinks by the time of contact of the group of bodies it may touch (see IPCBodies). With
the adaptive stiffness, the stiffness of a pair grows with the closing of its gap (see IPCVerticesKappa).
"""

import math

import numpy as np

import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu
from genesis.engine.bvh import (
    BoxQuery,
    BVHLeaves,
    BVHQueryResults,
    BVHTreeState,
    build_bvh,
    func_bvh_query_leaves,
    get_bvh_data,
)
from genesis.engine.ipc.barrier import (
    func_barrier,
    func_barrier_gradient_hessian,
    func_barrier_normal_force,
    func_mollified_barrier,
    func_mollified_barrier_gradient_hessian,
    func_pair_barrier_parameters,
    func_point_halfplane_barrier_gradient_hessian,
)
from genesis.engine.ipc.ccd import (
    func_additive_ccd,
    func_candidate_reach,
    func_directional_ccd,
    func_is_swept_within,
    func_point_halfplane_candidate_reach,
    func_point_halfplane_ccd,
)
from genesis.engine.ipc.distance import (
    func_edge_edge_distance_flag,
    func_edge_edge_flagged_distance2,
    func_point_triangle_distance_flag,
    func_point_triangle_flagged_distance2,
)
from genesis.engine.ipc.friction import (
    func_edge_edge_friction_frame,
    func_friction_energy,
    func_friction_gradient_hessian,
    func_point_edge_friction_frame,
    func_point_halfplane_friction_frame,
    func_point_point_friction_frame,
    func_point_triangle_friction_frame,
)
from genesis.engine.ipc.intersection import func_edge_triangle_intersect
from genesis.engine.numerics.bcoo import func_add_by_key, func_write_triplet
from genesis.engine.numerics.newton_rows import NewtonRows, NewtonSegment
from genesis.engine.solvers.fem_solver import FEMInfo, FEMState, func_add_stencil_terms
from genesis.engine.solvers.rigid.data import DynInfo, DynState
from genesis.utils.data import ENVS, Data, DataKind, Size, field, iter_state_arrays
from genesis.utils.misc import qd_destroy_data, qd_isfinite, qd_to_numpy

# The vertices of the cloth (see FEMVertices in fem_solver.py), the fixed planes of the scene, the contact bodies, one
# per cloth entity, and the candidate pairs of the broad phase
VERTICES = Size("vertices")
HALFPLANES = Size("halfplanes")
BODIES = Size("bodies")
PT_PAIRS = Size("pt_pairs")
EE_PAIRS = Size("ee_pairs")

# The arrays of the planes, of the partition of the step into contact bodies and of the adaptive stiffness, which the
# scenes without them leave out
HALFPLANES_LABEL = "halfplanes"
PARTITION = "partition"
KAPPA = "kappa"

# The triplets of the upper triangle of the Hessian of a pair, over its 4 vertices. 'func_contact_assemble' declares the
# same count as a local (see TRIANGLE_PAIRS in fem_solver.py).
PAIR_TRIPLETS = 10


class IPCHalfplanes(Data, kind=DataKind.SCRATCH, shape=(HALFPLANES, ENVS)):
    """
    The fixed planes of the scene over the substep, a point of each one and its unit normal, toward its free side.

    `friction_force` is the normal force of the barrier of every vertex against every plane at the start of the
    substep, at the stiffness of the cloth material, the friction pairs being those in the band there, and zero for the
    other pairs.
    """

    pos: qd.Tensor = field((float, 3))
    normal: qd.Tensor = field((float, 3))
    friction_force: qd.Tensor = field(float, shape=(VERTICES, HALFPLANES, ENVS))


class IPCBodies(Data, kind=DataKind.SCRATCH, shape=(BODIES, ENVS)):
    """
    The contact bodies of every environment, one per cloth entity, over the sweep of a Newton direction.

    `toc` is the smallest time of contact of the pairs of a body, `is_linked` whether a pair joins two bodies, `group` the
    smallest body of the group of bodies linked to it, and `alpha` the step of the group, the smallest time of contact of
    its bodies, which scales the direction of their vertices.
    """

    toc: qd.Tensor = field(float)
    is_linked: qd.Tensor = field(bool, shape=(BODIES, BODIES, ENVS))
    group: qd.Tensor = field(int)
    alpha: qd.Tensor = field(float)


class IPCVerticesKappa(Data, kind=DataKind.STATE, shape=(VERTICES, ENVS)):
    """
    The adaptive stiffness of the contacts of the vertices of the cloth (see 'gs.options.IPCOptions').

    `scale` multiplies the stiffness of the material, a pair taking the largest scale of its vertices, and `calm` counts
    the substeps a vertex has spent without a collapsing gap. `frame_min_gap` and `iter_min_gap` are the smallest gap of
    the pairs of a vertex in the band, over d_hat beyond the thickness, since the last substep and the last Newton
    iteration, 'min_gap_unrecorded' without any.
    """

    scale: qd.Tensor = field(float)
    calm: qd.Tensor = field(int)
    frame_min_gap: qd.Tensor = field(float)
    iter_min_gap: qd.Tensor = field(float)


class IPCState(Data, kind=DataKind.STATE):
    """Runtime state of the contact of the cloth."""

    halfplanes: IPCHalfplanes = field(labels=HALFPLANES_LABEL)
    bodies: IPCBodies = field(labels=PARTITION)
    vertices_kappa: IPCVerticesKappa = field(labels=KAPPA)
    # Whether the adaptive stiffness of a vertex grew at the start of the Newton iteration of an environment
    is_kappa_grown: qd.Tensor = field(bool, shape=(ENVS,), kind=DataKind.SCRATCH)


class IPCInfo(Data, kind=DataKind.CONSTANT):
    """
    Model parameters of the contact of the cloth (see 'gs.options.IPCOptions').

    `halfplanes_geom_idx` is the rigid geom of every plane, `ccd_eta` the fraction of the gap above the thickness that a
    step keeps at the time of contact of a pair, and `ccd_max_iterations` the iterations of its search at most. The
    barrier freezes its curvature below the ratio `hessian_clamp_ratio` of the band (see 'func_barrier_gradient_hessian'
    in genesis/engine/ipc/barrier.py), and two edges whose squared cross norm is below `mollifier_ratio` times the
    product of their squared rest lengths are mollified, the Hessian of their barrier being projected with the
    tolerances `projection_rank_tolerance` and `projection_eigen_tolerance` in `projection_max_sweeps` sweeps at most
    (see 'qd_project_psd' in genesis/utils/geom.py). The distance of two edges is classified with the tolerance
    `ee_parallel_tolerance`, and `ee_type_parallel_tolerance` in the additive time of contact. Two edges in contact whose
    squared cross norm is below `friction_parallel_ratio` times the product of their squared lengths take no friction.
    """

    halfplanes_geom_idx: qd.Tensor = field(int, shape=(HALFPLANES,), labels=HALFPLANES_LABEL)
    friction_eps_v: qd.Tensor = field(float, shape=())
    ccd_eta: qd.Tensor = field(float, shape=())
    ccd_max_iterations: qd.Tensor = field(int, shape=())
    hessian_clamp_ratio: qd.Tensor = field(float, shape=())
    mollifier_ratio: qd.Tensor = field(float, shape=())
    projection_rank_tolerance: qd.Tensor = field(float, shape=())
    projection_eigen_tolerance: qd.Tensor = field(float, shape=())
    projection_max_sweeps: qd.Tensor = field(int, shape=())
    ee_parallel_tolerance: qd.Tensor = field(float, shape=())
    ee_type_parallel_tolerance: qd.Tensor = field(float, shape=())
    friction_parallel_ratio: qd.Tensor = field(float, shape=())
    # The relative tolerance of the test of an edge crossing a triangle (see 'func_edge_triangle_intersect')
    intersection_tolerance: qd.Tensor = field(float, shape=())
    kappa_gap_ratio: qd.Tensor = field(float, shape=())
    kappa_grow: qd.Tensor = field(float, shape=())
    kappa_max_scale: qd.Tensor = field(float, shape=())
    kappa_calm_time: qd.Tensor = field(float, shape=())
    kappa_relax_time: qd.Tensor = field(float, shape=())
    kappa_hysteresis: qd.Tensor = field(float, shape=())
    min_gap_unrecorded: qd.Tensor = field(float, shape=())


class IPCPairs(Data, kind=DataKind.SCRATCH):
    """
    The candidate pairs of the broad phase: point-triangle pairs (environment, vertex, triangle) and edge-edge pairs
    (environment, edge, edge), the first edge of lower index.

    `pt_flag` and `ee_flag` are the distance flags of the pairs at the last assembly (see genesis/engine/ipc/distance.py),
    0 for a pair outside the band. `counts` holds the numbers of point-triangle and of edge-edge pairs the broad phase
    found, the pairs beyond the capacity being dropped.
    """

    pt: qd.Tensor = field((int, 3), shape=(PT_PAIRS,), ndarray=True)
    pt_flag: qd.Tensor = field(int, shape=(PT_PAIRS,), ndarray=True)
    ee: qd.Tensor = field((int, 3), shape=(EE_PAIRS,), ndarray=True)
    ee_flag: qd.Tensor = field(int, shape=(EE_PAIRS,), ndarray=True)
    counts: qd.Tensor = field(int, shape=(2,), ndarray=True)


class IPCFrictionPairs(Data, kind=DataKind.SCRATCH):
    """
    The friction pairs of the substep, the candidate pairs in the band at its start (see IPCPairs).

    `pt_flag` and `ee_flag` are their distance flags there, which set their frames, and `pt_normal_force` and
    `ee_normal_force` the normal forces of their barriers there, at the stiffness of the cloth material.
    """

    pt: qd.Tensor = field((int, 3), shape=(PT_PAIRS,), ndarray=True)
    pt_flag: qd.Tensor = field(int, shape=(PT_PAIRS,), ndarray=True)
    pt_normal_force: qd.Tensor = field(float, shape=(PT_PAIRS,), ndarray=True)
    ee: qd.Tensor = field((int, 3), shape=(EE_PAIRS,), ndarray=True)
    ee_flag: qd.Tensor = field(int, shape=(EE_PAIRS,), ndarray=True)
    ee_normal_force: qd.Tensor = field(float, shape=(EE_PAIRS,), ndarray=True)
    counts: qd.Tensor = field(int, shape=(2,), ndarray=True)


@qd.data_oriented
class PointTriangleQuery(BoxQuery):
    """The box query of the vertices of the cloth against its triangles, dropping a vertex of the triangle."""

    def __init__(self, results: BVHQueryResults, triangles_verts_idx: qd.Tensor) -> None:
        super().__init__(results)
        self.triangles_verts_idx = triangles_verts_idx

    @qd.func
    def is_kept(self, i_t: int, i_q: int, i_k: int, i_a: int) -> bool:
        tri = self.triangles_verts_idx[i_a]
        return i_q != tri[0] and i_q != tri[1] and i_q != tri[2]


@qd.data_oriented
class EdgeEdgeQuery(BoxQuery):
    """The box query of the edges of the cloth against its edges, each pair once with the edge of lower index querying,
    dropping the edges that share a vertex."""

    def __init__(self, results: BVHQueryResults, edges_verts_idx: qd.Tensor) -> None:
        super().__init__(results)
        self.edges_verts_idx = edges_verts_idx

    @qd.func
    def is_kept(self, i_t: int, i_q: int, i_k: int, i_a: int) -> bool:
        is_kept = i_q < i_a
        if is_kept:
            edge_q = self.edges_verts_idx[i_q]
            edge_a = self.edges_verts_idx[i_a]
            for i, j in qd.static(qd.ndrange(2, 2)):
                if edge_q[i] == edge_a[j]:
                    is_kept = False
        return is_kept


@qd.data_oriented
class EdgeTriangleQuery(BoxQuery):
    """The box query of the edges of the cloth against its triangles, keeping the edges that cross a triangle they do not
    share a vertex with (see 'func_edge_triangle_intersect' in genesis/engine/ipc/intersection.py)."""

    def __init__(
        self,
        results: BVHQueryResults,
        edges_verts_idx: qd.Tensor,
        triangles_verts_idx: qd.Tensor,
        positions: qd.Tensor,
        tolerance: qd.Tensor,
    ) -> None:
        super().__init__(results)
        self.edges_verts_idx = edges_verts_idx
        self.triangles_verts_idx = triangles_verts_idx
        self.positions = positions
        self.tolerance = tolerance

    @qd.func
    def is_kept(self, i_t: int, i_q: int, i_k: int, i_a: int) -> bool:
        edge = self.edges_verts_idx[i_q]
        tri = self.triangles_verts_idx[i_a]
        is_kept = True
        for i, j in qd.static(qd.ndrange(2, 3)):
            if edge[i] == tri[j]:
                is_kept = False
        if is_kept:
            is_kept = func_edge_triangle_intersect(
                self.positions[tri[0], i_t],
                self.positions[tri[1], i_t],
                self.positions[tri[2], i_t],
                self.positions[edge[0], i_t],
                self.positions[edge[1], i_t],
                self.tolerance[None],
            )
        return is_kept


@qd.kernel
def kernel_contact_intersections(edges_query: BVHLeaves, triangles_tree: BVHTreeState, visitor: qd.template()):
    """Collect the edges of the cloth that cross one of its triangles, in every environment (see EdgeTriangleQuery)."""
    for _ in range(1):
        visitor.results.count[0] = 0
    for i_b in range(visitor.results.trees_count.shape[0]):
        visitor.results.trees_count[i_b] = 0
    func_bvh_query_leaves(edges_query, triangles_tree, visitor)


# ------------------------------------------------------------------------------------
# -------------------------------------- pairs ---------------------------------------
# ------------------------------------------------------------------------------------


@qd.func
def func_pair_parameters(i_a: int, i_b: int, fem_info: FEMInfo, is_edge_edge: qd.template()):
    """
    Return the stencil of a pair of primitives i_a and i_b, its activation distance, its contact thickness, its area
    weight, its contact resistance and its friction coefficient.

    A point-triangle pair (vertex i_a, triangle i_b) has the stencil (vertex, triangle vertices), and an edge-edge pair
    (edge i_a, edge i_b) the stencil (edge A vertices, edge B vertices). The resistance and the friction coefficient of a
    pair are the means of those of the first vertex of each side.
    """
    verts_idx = qd.Vector.zero(gs.qd_int, 4)
    d_hat, thickness, area_weight = gs.qd_float(0.0), gs.qd_float(0.0), gs.qd_float(0.0)
    if qd.static(is_edge_edge):
        edge_a = fem_info.cloth_edges.verts_idx[i_a]
        edge_b = fem_info.cloth_edges.verts_idx[i_b]
        verts_idx = qd.Vector([edge_a[0], edge_a[1], edge_b[0], edge_b[1]])
        d_hat, thickness, area_weight = func_pair_barrier_parameters(
            fem_info.cloth_vertices.contact_d_hat[edge_a[0]],
            fem_info.cloth_vertices.contact_d_hat[edge_b[0]],
            qd.Vector([fem_info.cloth_vertices.contact_thickness[edge_a[k]] for k in qd.static(range(2))]),
            qd.Vector([fem_info.cloth_vertices.contact_thickness[edge_b[k]] for k in qd.static(range(2))]),
            fem_info.cloth_edges.contact_area_weight[i_a],
            fem_info.cloth_edges.contact_area_weight[i_b],
        )
    else:
        tri = fem_info.cloth_triangles.verts_idx[i_b]
        verts_idx = qd.Vector([i_a, tri[0], tri[1], tri[2]])
        d_hat, thickness, area_weight = func_pair_barrier_parameters(
            fem_info.cloth_vertices.contact_d_hat[i_a],
            fem_info.cloth_vertices.contact_d_hat[tri[0]],
            qd.Vector([fem_info.cloth_vertices.contact_thickness[i_a]]),
            qd.Vector([fem_info.cloth_vertices.contact_thickness[tri[k]] for k in qd.static(range(3))]),
            fem_info.cloth_vertices.contact_area_weight[i_a],
            fem_info.cloth_triangles.contact_area_weight[i_b],
        )
    i_v_a = verts_idx[0]
    i_v_b = verts_idx[2] if qd.static(is_edge_edge) else verts_idx[1]
    resistance = 0.5 * (
        fem_info.cloth_vertices.contact_resistance[i_v_a] + fem_info.cloth_vertices.contact_resistance[i_v_b]
    )
    friction_mu = 0.5 * (fem_info.cloth_vertices.friction_mu[i_v_a] + fem_info.cloth_vertices.friction_mu[i_v_b])
    return verts_idx, d_hat, thickness, area_weight, resistance, friction_mu


@qd.func
def func_stencil_positions(i_b: int, verts_idx, positions: qd.Tensor):
    """Return the positions of the vertices of a stencil in environment i_b, as the rows of a matrix."""
    return qd.Matrix.rows([positions[verts_idx[k], i_b] for k in qd.static(range(4))])


@qd.func
def func_split_stencil(x, is_edge_edge: qd.template()):
    """Split the rows of a stencil into its two sides, one point against three for a point-triangle pair, two against
    two for an edge-edge pair."""
    N_A = qd.static(2 if is_edge_edge else 1)
    return qd.Matrix.rows([x[k, :] for k in qd.static(range(N_A))]), qd.Matrix.rows(
        [x[k, :] for k in qd.static(range(N_A, 4))]
    )


@qd.func
def func_kappa_scale(i_b: int, verts_idx, ipc_state: IPCState, has_adaptive_kappa: qd.template()) -> float:
    """Return the factor of the stiffness of a pair, the largest adaptive scale of its vertices (1 without it)."""
    scale = gs.qd_float(1.0)
    if qd.static(has_adaptive_kappa):
        scale = ipc_state.vertices_kappa.scale[verts_idx[0], i_b]
        for k in qd.static(range(1, verts_idx.n)):
            scale = qd.max(scale, ipc_state.vertices_kappa.scale[verts_idx[k], i_b])
    return scale


@qd.func
def func_report_gap(i_b: int, verts_idx, gap: float, ipc_state: IPCState, is_kappa_tick_newton: qd.template()):
    """Lower the smallest gaps of the vertices of a pair in the band (see IPCVerticesKappa)."""
    for k in qd.static(range(verts_idx.n)):
        qd.atomic_min(ipc_state.vertices_kappa.frame_min_gap[verts_idx[k], i_b], gap)
        if qd.static(is_kappa_tick_newton):
            qd.atomic_min(ipc_state.vertices_kappa.iter_min_gap[verts_idx[k], i_b], gap)


@qd.func
def func_flagged_distance2(x, ipc_info: IPCInfo, is_edge_edge: qd.template()):
    """Classify the closest features of a stencil at positions `x`, and return its distance flag, its squared distance
    and the gradient of the latter (see genesis/engine/ipc/distance.py)."""
    flag = 0
    distance2 = gs.qd_float(0.0)
    distance2_gradient = qd.Vector.zero(gs.qd_float, 12)
    if qd.static(is_edge_edge):
        flag = func_edge_edge_distance_flag(x[0, :], x[1, :], x[2, :], x[3, :], ipc_info.ee_parallel_tolerance[None])
        distance2, distance2_gradient = func_edge_edge_flagged_distance2(flag, x[0, :], x[1, :], x[2, :], x[3, :])
    else:
        flag = func_point_triangle_distance_flag(x[0, :], x[1, :], x[2, :], x[3, :])
        distance2, distance2_gradient = func_point_triangle_flagged_distance2(flag, x[0, :], x[1, :], x[2, :], x[3, :])
    return flag, distance2, distance2_gradient


@qd.func
def func_pair_barrier(
    x,
    mollifier_threshold: float,
    d_hat: float,
    thickness: float,
    kappa: float,
    ipc_info: IPCInfo,
    is_edge_edge: qd.template(),
):
    """
    Compute the distance flag of a stencil at positions `x`, whether it lies in the band, its gap over d_hat beyond the
    thickness, and the energy, the gradient and the positive semi-definite Hessian of its barrier.

    An edge-edge barrier is mollified once the squared cross norm of the edges falls below `mollifier_threshold`.
    """
    flag, distance2, distance2_gradient = func_flagged_distance2(x, ipc_info, is_edge_edge)
    is_active = distance2 > thickness * thickness and distance2 < (d_hat + thickness) * (d_hat + thickness)
    energy = gs.qd_float(0.0)
    gradient = qd.Vector.zero(gs.qd_float, 12)
    hessian = qd.Matrix.zero(gs.qd_float, 12, 12)
    gap = gs.qd_float(0.0)
    if is_active:
        is_mollified = False
        cross_norm2 = gs.qd_float(0.0)
        if qd.static(is_edge_edge):
            cross_norm2 = (x[1, :] - x[0, :]).cross(x[3, :] - x[2, :]).norm_sqr()
            is_mollified = cross_norm2 < mollifier_threshold
        if is_mollified:
            energy = func_mollified_barrier(distance2, cross_norm2, mollifier_threshold, d_hat, thickness, kappa)
            gradient, hessian = func_mollified_barrier_gradient_hessian(
                flag,
                x[0, :],
                x[1, :],
                x[2, :],
                x[3, :],
                distance2,
                distance2_gradient,
                mollifier_threshold,
                d_hat,
                thickness,
                kappa,
                ipc_info.projection_rank_tolerance[None],
                ipc_info.projection_eigen_tolerance[None],
                ipc_info.projection_max_sweeps[None],
            )
        else:
            energy, _, _ = func_barrier(distance2, d_hat, thickness, kappa)
            gradient, hessian = func_barrier_gradient_hessian(
                distance2, distance2_gradient, d_hat, thickness, kappa, ipc_info.hessian_clamp_ratio[None]
            )
        gap = (qd.sqrt(distance2) - thickness) / d_hat
    return flag, is_active, gap, energy, gradient, hessian


@qd.func
def func_flagged_friction_frame(flag: int, x_lagged, is_edge_edge: qd.template()):
    """
    Compute the tangent basis of a stencil at its lagged positions and the weights of its 4 vertices, for the closest
    features of its distance flag, the inactive vertices weighing zero (see genesis/engine/ipc/friction.py).
    """
    x0, x1, x2, x3 = x_lagged[0, :], x_lagged[1, :], x_lagged[2, :], x_lagged[3, :]
    n_active = qd.math.popcnt(flag)
    basis = qd.Matrix.zero(gs.qd_float, 3, 2)
    weights = qd.Vector.zero(gs.qd_float, 4)
    if n_active == 4:
        if qd.static(is_edge_edge):
            basis, weights = func_edge_edge_friction_frame(x0, x1, x2, x3)
        else:
            basis, weights = func_point_triangle_friction_frame(x0, x1, x2, x3)
    else:
        # The point and the vertices of the edge of a point-edge feature, the two vertices of a point-point one
        i_v_point, i_v_first, i_v_last = 0, 1, 1
        if qd.static(is_edge_edge):
            i_va = 0 if flag & 0x1 else 1
            i_vb = 2 if flag & 0x4 else 3
            is_point_on_a = (flag & 0x3) != 0x3
            i_v_point = i_va if is_point_on_a else i_vb
            i_v_first = 2 if is_point_on_a else 0
            i_v_last = i_v_first + 1
            if n_active == 2:
                i_v_point = i_va
                i_v_first = i_vb
        else:
            i_v_first = 1 if flag & 0x2 else (2 if flag & 0x4 else 3)
            i_v_last = 3 if flag & 0x8 else (2 if flag & 0x4 else 1)
        point = qd.select(i_v_point == 0, x0, qd.select(i_v_point == 1, x1, qd.select(i_v_point == 2, x2, x3)))
        first = qd.select(i_v_first == 0, x0, qd.select(i_v_first == 1, x1, qd.select(i_v_first == 2, x2, x3)))
        last = qd.select(i_v_last == 0, x0, qd.select(i_v_last == 1, x1, qd.select(i_v_last == 2, x2, x3)))
        if n_active == 3:
            basis_pe, weights_pe = func_point_edge_friction_frame(point, first, last)
            basis = basis_pe
            for k in qd.static(range(4)):
                weights[k] = qd.select(
                    k == i_v_point,
                    weights_pe[0],
                    qd.select(k == i_v_first, weights_pe[1], qd.select(k == i_v_last, weights_pe[2], 0.0)),
                )
        else:
            basis_pp, weights_pp = func_point_point_friction_frame(point, first)
            basis = basis_pp
            for k in qd.static(range(4)):
                weights[k] = qd.select(k == i_v_point, weights_pp[0], qd.select(k == i_v_first, weights_pp[1], 0.0))
    return basis, weights


@qd.func
def func_pair_terms(
    i_b: int,
    i_a_prim: int,
    i_b_prim: int,
    flag_lagged: int,
    normal_force: float,
    fem_state: FEMState,
    ipc_state: IPCState,
    fem_info: FEMInfo,
    ipc_info: IPCInfo,
    newton_config: qd.template(),
    is_edge_edge: qd.template(),
    is_friction: qd.template(),
):
    """
    Compute the barrier of a candidate pair, or the friction of a friction pair, of primitives i_a_prim and i_b_prim in
    environment i_b at the positions of its vertices: its stencil, its distance flag, whether it lies in the band, its
    gap over d_hat beyond the thickness, and its energy, gradient and positive semi-definite Hessian, scaled by dt^2
    times its area weight.

    The friction of a pair takes its frame from its distance flag `flag_lagged` at the lagged positions of its vertices,
    and its normal force `normal_force` there at the stiffness of the material, both scaled by its adaptive stiffness.
    """
    verts_idx, d_hat, thickness, area_weight, resistance, friction_mu = func_pair_parameters(
        i_a_prim, i_b_prim, fem_info, is_edge_edge
    )
    kappa_scale = func_kappa_scale(i_b, verts_idx, ipc_state, newton_config.has_adaptive_kappa)
    x = func_stencil_positions(i_b, verts_idx, fem_state.vertices.pos_next)
    flag, is_active, gap = 0, False, gs.qd_float(0.0)
    energy = gs.qd_float(0.0)
    gradient = qd.Vector.zero(gs.qd_float, 12)
    hessian = qd.Matrix.zero(gs.qd_float, 12, 12)
    if qd.static(is_friction):
        x_lagged = func_stencil_positions(i_b, verts_idx, fem_state.vertices.pos)
        basis, weights = func_flagged_friction_frame(flag_lagged, x_lagged, is_edge_edge)
        eps_vh = ipc_info.friction_eps_v[None] * fem_info.substep_dt[None]
        force = kappa_scale * normal_force
        energy = func_friction_energy(basis, weights, x - x_lagged, force, friction_mu, eps_vh)
        gradient, hessian = func_friction_gradient_hessian(basis, weights, x - x_lagged, force, friction_mu, eps_vh)
    else:
        mollifier_threshold = gs.qd_float(0.0)
        if qd.static(is_edge_edge):
            # The mollifier threshold from the rest lengths (see 'func_edge_edge_mollifier_threshold' in ipc/barrier.py)
            mollifier_threshold = (
                ipc_info.mollifier_ratio[None]
                * fem_info.cloth_edges.rest_length2[i_a_prim]
                * fem_info.cloth_edges.rest_length2[i_b_prim]
            )
        flag, is_active, gap, energy, gradient, hessian = func_pair_barrier(
            x, mollifier_threshold, d_hat, thickness, resistance * kappa_scale, ipc_info, is_edge_edge
        )
    scale = fem_info.substep_dt[None] * fem_info.substep_dt[None] * area_weight
    return verts_idx, flag, is_active, gap, scale * energy, scale * gradient, scale * hessian


@qd.func
def func_halfplanes_terms(
    i_v: int,
    i_b: int,
    fem_state: FEMState,
    ipc_state: IPCState,
    fem_info: FEMInfo,
    ipc_info: IPCInfo,
    newton_config: qd.template(),
    is_assembly: qd.template(),
):
    """
    Compute the contact energy of a vertex of the cloth against every plane at its position, and its gradient and
    positive semi-definite Hessian (see the module docstring).

    The assembly lowers the smallest gap of the vertex with the adaptive stiffness.
    """
    x = fem_state.vertices.pos_next[i_v, i_b]
    x_lagged = fem_state.vertices.pos[i_v, i_b]
    thickness = fem_info.cloth_vertices.contact_thickness[i_v]
    d_hat = fem_info.cloth_vertices.contact_d_hat[i_v]
    kappa_scale = func_kappa_scale(i_b, qd.Vector([i_v]), ipc_state, newton_config.has_adaptive_kappa)
    kappa = kappa_scale * fem_info.cloth_vertices.contact_resistance[i_v]
    friction_mu = fem_info.cloth_vertices.friction_mu[i_v]
    eps_vh = ipc_info.friction_eps_v[None] * fem_info.substep_dt[None]
    energy = gs.qd_float(0.0)
    gradient = qd.Vector.zero(gs.qd_float, 3)
    hessian = qd.Matrix.zero(gs.qd_float, 3, 3)
    for i_h in range(ipc_state.halfplanes.pos.shape[0]):
        normal = ipc_state.halfplanes.normal[i_h, i_b]
        distance = normal.dot(x - ipc_state.halfplanes.pos[i_h, i_b])
        if distance > thickness and distance < thickness + d_hat:
            barrier, _, _ = func_barrier(distance * distance, d_hat, thickness, kappa)
            energy += barrier
            gradient_pair, hessian_pair = func_point_halfplane_barrier_gradient_hessian(
                distance, normal, d_hat, thickness, kappa
            )
            gradient += gradient_pair
            hessian += hessian_pair
            if qd.static(is_assembly and newton_config.has_adaptive_kappa):
                func_report_gap(
                    i_b, qd.Vector([i_v]), (distance - thickness) / d_hat, ipc_state, newton_config.is_kappa_tick_newton
                )
        friction_force = ipc_state.halfplanes.friction_force[i_v, i_h, i_b]
        if friction_force > 0.0:
            basis, weights = func_point_halfplane_friction_frame(normal)
            displacements = qd.Matrix.rows([x - x_lagged])
            force = kappa_scale * friction_force
            energy += func_friction_energy(basis, weights, displacements, force, friction_mu, eps_vh)
            gradient_pair, hessian_pair = func_friction_gradient_hessian(
                basis, weights, displacements, force, friction_mu, eps_vh
            )
            gradient += gradient_pair
            hessian += hessian_pair
    scale = fem_info.substep_dt[None] * fem_info.substep_dt[None] * fem_info.cloth_vertices.contact_area_weight[i_v]
    scale = scale * d_hat
    return scale * energy, scale * gradient, scale * hessian


@qd.func(requires_top_level=True)
def func_contact_assemble(
    triplet_start: int,
    newton_rows: NewtonRows,
    fem_state: FEMState,
    ipc_pairs: IPCPairs,
    friction_pairs: IPCFrictionPairs,
    ipc_state: IPCState,
    fem_info: FEMInfo,
    ipc_info: IPCInfo,
    newton_config: qd.template(),
):
    """
    Add the gradients of the contacts of the cloth to its rows, and write the blocks of the upper triangle of their
    projected Hessians from the triplet `triplet_start` on: one per vertex against the planes, then PAIR_TRIPLETS per
    candidate point-triangle pair, edge-edge pair, friction point-triangle pair and friction edge-edge pair in turn.

    The barrier of a candidate pair writes its distance flag (see IPCPairs) and lowers the smallest gaps of its
    vertices with the adaptive stiffness. A fixed vertex takes no contact terms.
    """
    # See PAIR_TRIPLETS at module level
    PAIR_TRIPLETS = qd.static(10)
    n_vertices = fem_state.vertices.pos.shape[0]
    n_envs = fem_state.vertices.pos.shape[1]
    row_start = newton_rows.segments_start[qd.static(int(NewtonSegment.CLOTH_VERTICES))]
    i_triplet = triplet_start
    if qd.static(newton_config.has_halfplanes):
        for i_v, i_b in qd.ndrange(n_vertices, n_envs):
            i_r = row_start + i_b * n_vertices + i_v
            hessian = qd.Matrix.zero(gs.qd_float, 3, 3)
            if not fem_state.cloth_vertices.is_fixed[i_v, i_b]:
                _energy, gradient, hessian_contact = func_halfplanes_terms(
                    i_v, i_b, fem_state, ipc_state, fem_info, ipc_info, newton_config, is_assembly=True
                )
                hessian = hessian_contact
                qd.atomic_add(newton_rows.gradient[i_r], gradient)
            func_write_triplet(
                newton_rows.matrix, i_triplet + i_b * n_vertices + i_v, i_r, i_r, hessian, is_symmetric=True
            )
        i_triplet += n_envs * n_vertices
    for i_kind in qd.static(range(2)):
        n_pairs = qd.min(ipc_pairs.counts[i_kind], ipc_pairs.pt.shape[0])
        for i_p in range(n_pairs):
            pair = qd.Vector.zero(gs.qd_int, 3)
            if qd.static(i_kind == 1):
                pair = ipc_pairs.ee[i_p]
            else:
                pair = ipc_pairs.pt[i_p]
            i_b = pair[0]
            verts_idx, flag, is_active, gap, _energy, gradient, hessian = func_pair_terms(
                i_b,
                pair[1],
                pair[2],
                0,
                0.0,
                fem_state,
                ipc_state,
                fem_info,
                ipc_info,
                newton_config,
                is_edge_edge=i_kind == 1,
                is_friction=False,
            )
            if qd.static(i_kind == 1):
                ipc_pairs.ee_flag[i_p] = qd.select(is_active, flag, 0)
            else:
                ipc_pairs.pt_flag[i_p] = qd.select(is_active, flag, 0)
            if qd.static(newton_config.has_adaptive_kappa):
                if is_active:
                    func_report_gap(i_b, verts_idx, gap, ipc_state, newton_config.is_kappa_tick_newton)
            func_add_stencil_terms(
                i_b, i_triplet + PAIR_TRIPLETS * i_p, row_start, verts_idx, gradient, hessian, newton_rows, fem_state
            )
        i_triplet += PAIR_TRIPLETS * n_pairs
    for i_kind in qd.static(range(2)):
        n_pairs = qd.min(friction_pairs.counts[i_kind], friction_pairs.pt.shape[0])
        for i_p in range(n_pairs):
            pair = qd.Vector.zero(gs.qd_int, 3)
            flag, normal_force = 0, gs.qd_float(0.0)
            if qd.static(i_kind == 1):
                pair = friction_pairs.ee[i_p]
                flag, normal_force = friction_pairs.ee_flag[i_p], friction_pairs.ee_normal_force[i_p]
            else:
                pair = friction_pairs.pt[i_p]
                flag, normal_force = friction_pairs.pt_flag[i_p], friction_pairs.pt_normal_force[i_p]
            i_b = pair[0]
            verts_idx, _flag, _is_active, _gap, _energy, gradient, hessian = func_pair_terms(
                i_b,
                pair[1],
                pair[2],
                flag,
                normal_force,
                fem_state,
                ipc_state,
                fem_info,
                ipc_info,
                newton_config,
                is_edge_edge=i_kind == 1,
                is_friction=True,
            )
            func_add_stencil_terms(
                i_b, i_triplet + PAIR_TRIPLETS * i_p, row_start, verts_idx, gradient, hessian, newton_rows, fem_state
            )
        i_triplet += PAIR_TRIPLETS * n_pairs


@qd.func(requires_top_level=True)
def func_contact_energy(
    energies: qd.Tensor,
    envs_mask: qd.Tensor,
    fem_state: FEMState,
    ipc_pairs: IPCPairs,
    friction_pairs: IPCFrictionPairs,
    ipc_state: IPCState,
    fem_info: FEMInfo,
    ipc_info: IPCInfo,
    newton_config: qd.template(),
):
    """Add the energy of the contacts of the cloth at the positions of its vertices into the entry of every environment
    selected: the vertices against the planes, the barriers of the candidate pairs and the friction of the friction
    pairs (see 'func_contact_assemble')."""
    n_vertices = fem_state.vertices.pos.shape[0]
    n_envs = fem_state.vertices.pos.shape[1]
    if qd.static(newton_config.has_halfplanes):
        qd.loop_config(block_dim=256)
        for i_th in range((n_envs * n_vertices + 31) // 32 * 32):
            i_b_key = gs.qd_int(-1)
            energy = gs.qd_float(0.0)
            if i_th < n_envs * n_vertices:
                i_b = i_th // n_vertices
                i_v = i_th % n_vertices
                if envs_mask[i_b] and not fem_state.cloth_vertices.is_fixed[i_v, i_b]:
                    i_b_key = i_b
                    energy, _gradient, _hessian = func_halfplanes_terms(
                        i_v, i_b, fem_state, ipc_state, fem_info, ipc_info, newton_config, is_assembly=False
                    )
            func_add_by_key(i_b_key, energy, energies)
    for i_list in qd.static(range(4)):
        is_edge_edge = qd.static(i_list % 2 == 1)
        is_friction = qd.static(i_list >= 2)
        n_pairs = gs.qd_int(0)
        if qd.static(is_friction):
            n_pairs = qd.min(friction_pairs.counts[i_list % 2], friction_pairs.pt.shape[0])
        else:
            n_pairs = qd.min(ipc_pairs.counts[i_list % 2], ipc_pairs.pt.shape[0])
        qd.loop_config(block_dim=256)
        for i_th in range((n_pairs + 31) // 32 * 32):
            i_b_key = gs.qd_int(-1)
            energy = gs.qd_float(0.0)
            if i_th < n_pairs:
                pair = qd.Vector.zero(gs.qd_int, 3)
                flag, normal_force = 0, gs.qd_float(0.0)
                if qd.static(is_friction and is_edge_edge):
                    pair = friction_pairs.ee[i_th]
                    flag, normal_force = friction_pairs.ee_flag[i_th], friction_pairs.ee_normal_force[i_th]
                elif qd.static(is_friction):
                    pair = friction_pairs.pt[i_th]
                    flag, normal_force = friction_pairs.pt_flag[i_th], friction_pairs.pt_normal_force[i_th]
                elif qd.static(is_edge_edge):
                    pair = ipc_pairs.ee[i_th]
                else:
                    pair = ipc_pairs.pt[i_th]
                i_b = pair[0]
                if envs_mask[i_b]:
                    i_b_key = i_b
                    _verts_idx, _flag, _is_active, _gap, energy, _gradient, _hessian = func_pair_terms(
                        i_b,
                        pair[1],
                        pair[2],
                        flag,
                        normal_force,
                        fem_state,
                        ipc_state,
                        fem_info,
                        ipc_info,
                        newton_config,
                        is_edge_edge=is_edge_edge,
                        is_friction=is_friction,
                    )
            func_add_by_key(i_b_key, energy, energies)


# ------------------------------------------------------------------------------------
# ----------------------------------- substep ----------------------------------------
# ------------------------------------------------------------------------------------


@qd.func(requires_top_level=True)
def func_contact_init_substep(
    fem_state: FEMState,
    ipc_state: IPCState,
    dyn_state: DynState,
    fem_info: FEMInfo,
    ipc_info: IPCInfo,
    dyn_info: DynInfo,
    newton_config: qd.template(),
    penetration_errno: int,
    errno: qd.Tensor,
):
    """
    Place the planes for the substep, set the normal force of the friction pairs of the vertices against them from the
    positions at its start, and update the adaptive stiffness of the vertices from the smallest gap of their pairs over
    the last substep.

    A vertex starting within the thickness of a plane leaves the barrier undefined, which sets the flag
    `penetration_errno` in the errno of its environment. A collapsed gap stiffens the vertex by the grow factor up to the
    largest scale, unless the stiffness grows within the Newton solve, where it only rearms the calm hold. A gap within
    the hysteresis band holds the stiffness and the hold. A calm vertex holds its stiffness over the calm time, then
    relaxes it toward that of its material with the time constant of the relaxation.
    """
    if qd.static(newton_config.has_halfplanes):
        for i_h, i_b in qd.ndrange(*ipc_state.halfplanes.pos.shape):
            i_g = ipc_info.halfplanes_geom_idx[i_h]
            geom_quat = dyn_state.geoms.quat[i_g, i_b]
            normal = gu.qd_transform_by_quat(
                qd.Vector([dyn_info.geoms.data[i_g][0], dyn_info.geoms.data[i_g][1], dyn_info.geoms.data[i_g][2]]),
                geom_quat,
            )
            ipc_state.halfplanes.pos[i_h, i_b] = dyn_state.geoms.pos[i_g, i_b]
            ipc_state.halfplanes.normal[i_h, i_b] = normal.normalized()
        for i_v, i_h, i_b in qd.ndrange(*ipc_state.halfplanes.friction_force.shape):
            thickness = fem_info.cloth_vertices.contact_thickness[i_v]
            d_hat = fem_info.cloth_vertices.contact_d_hat[i_v]
            normal = ipc_state.halfplanes.normal[i_h, i_b]
            distance = normal.dot(fem_state.vertices.pos[i_v, i_b] - ipc_state.halfplanes.pos[i_h, i_b])
            if not (distance > thickness):
                qd.atomic_or(errno[i_b], penetration_errno)
            friction_force = gs.qd_float(0.0)
            if distance < thickness + d_hat and distance > thickness:
                friction_force = func_barrier_normal_force(
                    distance * distance, d_hat, thickness, fem_info.cloth_vertices.contact_resistance[i_v]
                )
            ipc_state.halfplanes.friction_force[i_v, i_h, i_b] = friction_force
    if qd.static(newton_config.has_adaptive_kappa):
        dt = fem_info.substep_dt[None]
        gap_ratio = ipc_info.kappa_gap_ratio[None]
        n_hold = qd.max(qd.cast(qd.ceil(ipc_info.kappa_calm_time[None] / dt), gs.qd_int), 1)
        for i_v, i_b in qd.ndrange(*ipc_state.vertices_kappa.scale.shape):
            gap = ipc_state.vertices_kappa.frame_min_gap[i_v, i_b]
            scale = ipc_state.vertices_kappa.scale[i_v, i_b]
            calm = ipc_state.vertices_kappa.calm[i_v, i_b]
            if gap < gap_ratio:
                calm = 0
                if qd.static(not newton_config.is_kappa_tick_newton):
                    scale = qd.min(scale * ipc_info.kappa_grow[None], ipc_info.kappa_max_scale[None])
            elif not (gap <= ipc_info.kappa_hysteresis[None] * gap_ratio):
                if calm < n_hold:
                    calm += 1
                else:
                    scale = qd.max(scale * qd.exp(-dt / ipc_info.kappa_relax_time[None]), 1.0)
            ipc_state.vertices_kappa.scale[i_v, i_b] = scale
            ipc_state.vertices_kappa.calm[i_v, i_b] = calm
            ipc_state.vertices_kappa.frame_min_gap[i_v, i_b] = ipc_info.min_gap_unrecorded[None]


@qd.kernel(fastcache=True)
def kernel_contact_init_kappa(ipc_state: IPCState, ipc_info: IPCInfo):
    """Set the adaptive stiffness of every vertex to that of its material, without any gap recorded."""
    for i_v, i_b in qd.ndrange(*ipc_state.vertices_kappa.scale.shape):
        ipc_state.vertices_kappa.scale[i_v, i_b] = 1.0
        ipc_state.vertices_kappa.calm[i_v, i_b] = 0
        ipc_state.vertices_kappa.frame_min_gap[i_v, i_b] = ipc_info.min_gap_unrecorded[None]
        ipc_state.vertices_kappa.iter_min_gap[i_v, i_b] = ipc_info.min_gap_unrecorded[None]


@qd.kernel(fastcache=True)
def kernel_contact_kappa_iteration(envs_mask: qd.Tensor, ipc_state: IPCState, ipc_info: IPCInfo):
    """
    Stiffen the vertices of the environments selected whose gap collapsed over the last Newton iteration, at the start
    of the next one, and flag the environments where the stiffness grew, whose energy at the start point changes.
    """
    for i_b in range(ipc_state.is_kappa_grown.shape[0]):
        ipc_state.is_kappa_grown[i_b] = False
    for i_v, i_b in qd.ndrange(*ipc_state.vertices_kappa.scale.shape):
        if envs_mask[i_b]:
            if ipc_state.vertices_kappa.iter_min_gap[i_v, i_b] < ipc_info.kappa_gap_ratio[None]:
                scale = ipc_state.vertices_kappa.scale[i_v, i_b]
                scale_grown = qd.min(scale * ipc_info.kappa_grow[None], ipc_info.kappa_max_scale[None])
                if scale_grown > scale:
                    ipc_state.vertices_kappa.scale[i_v, i_b] = scale_grown
                    ipc_state.is_kappa_grown[i_b] = True
        ipc_state.vertices_kappa.iter_min_gap[i_v, i_b] = ipc_info.min_gap_unrecorded[None]


# ------------------------------------------------------------------------------------
# ---------------------------------- broad phase -------------------------------------
# ------------------------------------------------------------------------------------


@qd.func
def func_sweep_box(verts_idx, i_b: int, fem_state: FEMState, is_swept: qd.template()):
    """Return the box of the positions of the vertices of a primitive in environment i_b at the start and at the end of
    their sweep along the Newton direction, or at their positions."""
    lower = qd.Vector([1e300] * 3, dt=gs.qd_float)
    upper = qd.Vector([-1e300] * 3, dt=gs.qd_float)
    for k in qd.static(range(verts_idx.n)):
        x = fem_state.vertices.pos_next[verts_idx[k], i_b]
        lower = qd.min(lower, x)
        upper = qd.max(upper, x)
        if qd.static(is_swept):
            end = x + fem_state.cloth_vertices.displacement[verts_idx[k], i_b]
            lower = qd.min(lower, end)
            upper = qd.max(upper, end)
    return lower, upper


@qd.kernel(fastcache=True)
def kernel_contact_boxes(
    fem_state: FEMState,
    fem_info: FEMInfo,
    triangles_leaves: BVHLeaves,
    edges_leaves: BVHLeaves,
    points_query: BVHLeaves,
    edges_query: BVHLeaves,
    is_swept: qd.template(),
):
    """
    Set the boxes of the triangles and of the edges of the cloth, and the query boxes of its vertices and of its edges,
    over the sweep of its vertices from their positions along the Newton direction, or at their positions.

    A triangle or an edge is inflated by the largest thickness of its vertices and by its activation distance, a vertex
    by its thickness and its activation distance, and a query edge by its activation distance once more, the gap of the
    two edge boxes.
    """
    n_envs = fem_state.vertices.pos.shape[1]
    for i_t, i_b in qd.ndrange(fem_info.cloth_triangles.verts_idx.shape[0], n_envs):
        tri = fem_info.cloth_triangles.verts_idx[i_t]
        lower, upper = func_sweep_box(tri, i_b, fem_state, is_swept)
        thickness = fem_info.cloth_vertices.contact_thickness[tri[0]]
        for k in qd.static(range(1, 3)):
            thickness = qd.max(thickness, fem_info.cloth_vertices.contact_thickness[tri[k]])
        i_v0 = tri[0]
        margin = thickness + fem_info.cloth_vertices.contact_d_hat[i_v0]
        triangles_leaves.aabbs_min[i_b, i_t] = lower - margin
        triangles_leaves.aabbs_max[i_b, i_t] = upper + margin
    for i_e, i_b in qd.ndrange(fem_info.cloth_edges.verts_idx.shape[0], n_envs):
        edge = fem_info.cloth_edges.verts_idx[i_e]
        lower, upper = func_sweep_box(edge, i_b, fem_state, is_swept)
        thickness = qd.max(
            fem_info.cloth_vertices.contact_thickness[edge[0]], fem_info.cloth_vertices.contact_thickness[edge[1]]
        )
        i_v0 = edge[0]
        d_hat = fem_info.cloth_vertices.contact_d_hat[i_v0]
        margin = thickness + d_hat
        edges_leaves.aabbs_min[i_b, i_e] = lower - margin
        edges_leaves.aabbs_max[i_b, i_e] = upper + margin
        edges_query.aabbs_min[i_b, i_e] = lower - (margin + d_hat)
        edges_query.aabbs_max[i_b, i_e] = upper + (margin + d_hat)
    for i_v, i_b in qd.ndrange(*fem_state.vertices.pos.shape):
        lower, upper = func_sweep_box(qd.Vector([i_v]), i_b, fem_state, is_swept)
        gap = fem_info.cloth_vertices.contact_d_hat[i_v] + fem_info.cloth_vertices.contact_thickness[i_v]
        points_query.aabbs_min[i_b, i_v] = lower - gap
        points_query.aabbs_max[i_b, i_v] = upper + gap


@qd.kernel
def kernel_contact_query(
    points_query: BVHLeaves,
    triangles_tree: BVHTreeState,
    edges_query: BVHLeaves,
    edges_tree: BVHTreeState,
    pt_visitor: qd.template(),
    ee_visitor: qd.template(),
):
    """Collect the triangles that the box of every vertex meets and the edges that the box of every edge meets, in the
    tree of its environment (see PointTriangleQuery and EdgeEdgeQuery)."""
    for _ in range(1):
        pt_visitor.results.count[0] = 0
        ee_visitor.results.count[0] = 0
    for i_b in range(pt_visitor.results.trees_count.shape[0]):
        pt_visitor.results.trees_count[i_b] = 0
        ee_visitor.results.trees_count[i_b] = 0
    func_bvh_query_leaves(points_query, triangles_tree, pt_visitor)
    func_bvh_query_leaves(edges_query, edges_tree, ee_visitor)


@qd.func
def func_set_link(i_b: int, i_body_a: int, i_body_b: int, ipc_state: IPCState):
    """Link two contact bodies of environment i_b, which then take the same step (see IPCBodies)."""
    if i_body_a != i_body_b:
        ipc_state.bodies.is_linked[i_body_a, i_body_b, i_b] = True
        ipc_state.bodies.is_linked[i_body_b, i_body_a, i_b] = True


@qd.func(requires_top_level=True)
def func_filter_hits(
    hits: BVHQueryResults,
    ipc_pairs: IPCPairs,
    fem_state: FEMState,
    ipc_state: IPCState,
    fem_info: FEMInfo,
    ipc_info: IPCInfo,
    newton_config: qd.template(),
    is_edge_edge: qd.template(),
    is_swept: qd.template(),
):
    """Keep the pairs of the hits of a box query that may enter the band over the sweep of their vertices (see
    'kernel_contact_filter')."""
    i_kind = qd.static(1 if is_edge_edge else 0)
    for i_r in range(hits.count[0]):
        i_b = hits.triplets[i_r][0]
        i_a = hits.triplets[i_r][1]
        i_q = hits.triplets[i_r][2]
        verts_idx, d_hat, thickness, _, _, _ = func_pair_parameters(i_q, i_a, fem_info, is_edge_edge)
        x = func_stencil_positions(i_b, verts_idx, fem_state.vertices.pos_next)
        end = x
        if qd.static(is_swept):
            end = x + func_stencil_positions(i_b, verts_idx, fem_state.cloth_vertices.displacement)
        points_a, points_b = func_split_stencil(x, is_edge_edge)
        ends_a, ends_b = func_split_stencil(end, is_edge_edge)
        N_A = qd.static(2 if is_edge_edge else 1)
        verdict = func_candidate_reach(
            points_a,
            points_b,
            ends_a,
            ends_b,
            qd.Vector.zero(gs.qd_float, N_A),
            qd.Vector.zero(gs.qd_float, 4 - N_A),
            qd.Vector.zero(gs.qd_float, N_A),
            qd.Vector.zero(gs.qd_float, 4 - N_A),
            d_hat + thickness,
            ipc_info.ee_parallel_tolerance[None],
            True,
        )
        if verdict == 1:
            i_p = qd.atomic_add(ipc_pairs.counts[i_kind], 1)
            if i_p < ipc_pairs.pt.shape[0]:
                if qd.static(is_edge_edge):
                    ipc_pairs.ee[i_p] = qd.Vector([i_b, i_q, i_a])
                else:
                    ipc_pairs.pt[i_p] = qd.Vector([i_b, i_q, i_a])
        elif verdict == 2:
            if qd.static(newton_config.has_ccd_partition):
                i_v_a = verts_idx[0]
                i_v_b = verts_idx[3]
                func_set_link(
                    i_b,
                    fem_info.cloth_vertices.contact_body_idx[i_v_a],
                    fem_info.cloth_vertices.contact_body_idx[i_v_b],
                    ipc_state,
                )


@qd.kernel(fastcache=True)
def kernel_contact_filter(
    pt_hits: BVHQueryResults,
    ee_hits: BVHQueryResults,
    ipc_pairs: IPCPairs,
    fem_state: FEMState,
    ipc_state: IPCState,
    fem_info: FEMInfo,
    ipc_info: IPCInfo,
    newton_config: qd.template(),
    is_swept: qd.template(),
):
    """
    Keep the pairs of the hits of the box queries that may enter the band over the sweep of their vertices (see
    'func_candidate_reach' in genesis/engine/ipc/ccd.py), the candidates.

    A pair that only stays out of the band if its two sides take the same step links their contact bodies, with the
    partition of the step into contact bodies (see IPCBodies).
    """
    for _ in range(1):
        ipc_pairs.counts[0] = 0
        ipc_pairs.counts[1] = 0
    if qd.static(newton_config.has_ccd_partition):
        for i_body_a, i_body_b, i_b in qd.ndrange(*ipc_state.bodies.is_linked.shape):
            ipc_state.bodies.is_linked[i_body_a, i_body_b, i_b] = False
    func_filter_hits(
        pt_hits,
        ipc_pairs,
        fem_state,
        ipc_state,
        fem_info,
        ipc_info,
        newton_config,
        is_edge_edge=False,
        is_swept=is_swept,
    )
    func_filter_hits(
        ee_hits,
        ipc_pairs,
        fem_state,
        ipc_state,
        fem_info,
        ipc_info,
        newton_config,
        is_edge_edge=True,
        is_swept=is_swept,
    )


@qd.func(requires_top_level=True)
def func_set_friction_pairs(
    ipc_pairs: IPCPairs,
    friction_pairs: IPCFrictionPairs,
    fem_state: FEMState,
    fem_info: FEMInfo,
    ipc_info: IPCInfo,
    penetration_errno: int,
    errno: qd.Tensor,
    is_edge_edge: qd.template(),
):
    """Set the friction pairs of one kind from the candidate pairs (see 'kernel_contact_friction_pairs')."""
    i_kind = qd.static(1 if is_edge_edge else 0)
    for i_p in range(qd.min(ipc_pairs.counts[i_kind], ipc_pairs.pt.shape[0])):
        pair = qd.Vector.zero(gs.qd_int, 3)
        if qd.static(is_edge_edge):
            pair = ipc_pairs.ee[i_p]
        else:
            pair = ipc_pairs.pt[i_p]
        i_b = pair[0]
        verts_idx, d_hat, thickness, _, resistance, _ = func_pair_parameters(pair[1], pair[2], fem_info, is_edge_edge)
        x = func_stencil_positions(i_b, verts_idx, fem_state.vertices.pos)
        flag, distance2, _ = func_flagged_distance2(x, ipc_info, is_edge_edge)
        if not (distance2 > thickness * thickness):
            qd.atomic_or(errno[i_b], penetration_errno)
        is_friction = distance2 > thickness * thickness and distance2 < (d_hat + thickness) * (d_hat + thickness)
        if qd.static(is_edge_edge):
            if qd.math.popcnt(flag) == 4:
                edge_a = x[1, :] - x[0, :]
                edge_b = x[3, :] - x[2, :]
                if edge_a.cross(edge_b).norm_sqr() < (
                    ipc_info.friction_parallel_ratio[None] * edge_a.norm_sqr() * edge_b.norm_sqr()
                ):
                    is_friction = False
        if is_friction:
            normal_force = func_barrier_normal_force(distance2, d_hat, thickness, resistance)
            i_f = qd.atomic_add(friction_pairs.counts[i_kind], 1)
            if qd.static(is_edge_edge):
                friction_pairs.ee[i_f] = pair
                friction_pairs.ee_flag[i_f] = flag
                friction_pairs.ee_normal_force[i_f] = normal_force
            else:
                friction_pairs.pt[i_f] = pair
                friction_pairs.pt_flag[i_f] = flag
                friction_pairs.pt_normal_force[i_f] = normal_force


@qd.kernel(fastcache=True)
def kernel_contact_friction_pairs(
    ipc_pairs: IPCPairs,
    friction_pairs: IPCFrictionPairs,
    fem_state: FEMState,
    fem_info: FEMInfo,
    ipc_info: IPCInfo,
    penetration_errno: int,
    errno: qd.Tensor,
):
    """
    Set the friction pairs of the substep, the candidate pairs in the band at its start, with their distance flags and
    the normal forces of their barriers there (see IPCFrictionPairs).

    Two edges whose cross norm nearly vanishes take no friction. A pair starting within its thickness leaves its barrier
    undefined, which sets the flag `penetration_errno` in the errno of its environment.
    """
    for _ in range(1):
        friction_pairs.counts[0] = 0
        friction_pairs.counts[1] = 0
    func_set_friction_pairs(
        ipc_pairs, friction_pairs, fem_state, fem_info, ipc_info, penetration_errno, errno, is_edge_edge=False
    )
    func_set_friction_pairs(
        ipc_pairs, friction_pairs, fem_state, fem_info, ipc_info, penetration_errno, errno, is_edge_edge=True
    )


# ------------------------------------------------------------------------------------
# --------------------------------------- CCD ----------------------------------------
# ------------------------------------------------------------------------------------


@qd.func
def func_bound_step(
    i_b: int,
    i_body_a: int,
    i_body_b: int,
    toc: float,
    alpha: qd.Tensor,
    ipc_state: IPCState,
    has_ccd_partition: qd.template(),
):
    """Bound the step of environment i_b, or that of the contact bodies of a pair, which it links, by the time of contact
    of the pair (see IPCBodies)."""
    if qd.static(has_ccd_partition):
        qd.atomic_min(ipc_state.bodies.toc[i_body_a, i_b], toc)
        func_set_link(i_b, i_body_a, i_body_b, ipc_state)
    else:
        qd.atomic_min(alpha[i_b], toc)


@qd.func(requires_top_level=True)
def func_pairs_ccd(
    envs_mask: qd.Tensor,
    alpha: qd.Tensor,
    ipc_pairs: IPCPairs,
    fem_state: FEMState,
    ipc_state: IPCState,
    fem_info: FEMInfo,
    ipc_info: IPCInfo,
    newton_config: qd.template(),
    is_edge_edge: qd.template(),
):
    """Bound the steps by the time of contact of the candidate pairs of one kind (see 'kernel_contact_ccd')."""
    i_kind = qd.static(1 if is_edge_edge else 0)
    eta = ipc_info.ccd_eta[None]
    for i_p in range(qd.min(ipc_pairs.counts[i_kind], ipc_pairs.pt.shape[0])):
        pair = qd.Vector.zero(gs.qd_int, 3)
        if qd.static(is_edge_edge):
            pair = ipc_pairs.ee[i_p]
        else:
            pair = ipc_pairs.pt[i_p]
        i_b = pair[0]
        if envs_mask[i_b]:
            verts_idx, d_hat, thickness, _, _, _ = func_pair_parameters(pair[1], pair[2], fem_info, is_edge_edge)
            x = func_stencil_positions(i_b, verts_idx, fem_state.vertices.pos_next)
            displacement = func_stencil_positions(i_b, verts_idx, fem_state.cloth_vertices.displacement)
            points_a, points_b = func_split_stencil(x, is_edge_edge)
            displacements_a, displacements_b = func_split_stencil(displacement, is_edge_edge)
            toc = gs.qd_float(1.0)
            if qd.static(newton_config.is_ccd_directional):
                toc = func_directional_ccd(
                    points_a,
                    points_b,
                    displacements_a,
                    displacements_b,
                    eta,
                    thickness,
                    ipc_info.ccd_max_iterations[None],
                )
            else:
                toc = func_additive_ccd(
                    points_a,
                    points_b,
                    displacements_a,
                    displacements_b,
                    eta,
                    thickness,
                    ipc_info.ccd_max_iterations[None],
                    ipc_info.ee_type_parallel_tolerance[None],
                )
            i_v_a = verts_idx[0]
            i_v_b = verts_idx[3]
            i_body_a = fem_info.cloth_vertices.contact_body_idx[i_v_a]
            i_body_b = fem_info.cloth_vertices.contact_body_idx[i_v_b]
            is_bounding = True
            if qd.static(newton_config.has_ccd_partition):
                is_bounding = func_is_swept_within(
                    points_a, points_b, displacements_a, displacements_b, d_hat + thickness
                )
            if is_bounding:
                func_bound_step(i_b, i_body_a, i_body_b, toc, alpha, ipc_state, newton_config.has_ccd_partition)


@qd.kernel(fastcache=True)
def kernel_contact_ccd(
    envs_mask: qd.Tensor,
    alpha: qd.Tensor,
    ipc_pairs: IPCPairs,
    fem_state: FEMState,
    ipc_state: IPCState,
    fem_info: FEMInfo,
    ipc_info: IPCInfo,
    newton_config: qd.template(),
):
    """
    Bound the step of every environment selected by the conservative time of contact of the candidate pairs and of the
    pairs of its vertices with the planes along the Newton direction, or the steps of its contact bodies with the
    partition of the step into contact bodies (see IPCBodies).

    A candidate pair bounds the body of its first side once the sweeps of its two sides come within its band of each
    other, which links their bodies, and a vertex its own body once it moves toward a plane it reaches before the end of
    the step.
    """
    if qd.static(newton_config.has_ccd_partition):
        for i_body, i_b in qd.ndrange(*ipc_state.bodies.toc.shape):
            ipc_state.bodies.toc[i_body, i_b] = 1.0
    if qd.static(newton_config.has_halfplanes):
        eta = ipc_info.ccd_eta[None]
        for i_v, i_h, i_b in qd.ndrange(*ipc_state.halfplanes.friction_force.shape):
            if envs_mask[i_b]:
                x = fem_state.vertices.pos_next[i_v, i_b]
                displacement = fem_state.cloth_vertices.displacement[i_v, i_b]
                thickness = fem_info.cloth_vertices.contact_thickness[i_v]
                normal = ipc_state.halfplanes.normal[i_h, i_b]
                plane_pos = ipc_state.halfplanes.pos[i_h, i_b]
                band = fem_info.cloth_vertices.contact_d_hat[i_v] + thickness
                if func_point_halfplane_candidate_reach(x, x + displacement, plane_pos, normal, 0.0, band):
                    toi, is_bounding = func_point_halfplane_ccd(x, displacement, plane_pos, normal, eta, thickness)
                    if is_bounding and qd_isfinite(toi) and toi < 1.0:
                        i_body = fem_info.cloth_vertices.contact_body_idx[i_v]
                        func_bound_step(i_b, i_body, i_body, toi, alpha, ipc_state, newton_config.has_ccd_partition)
    func_pairs_ccd(
        envs_mask, alpha, ipc_pairs, fem_state, ipc_state, fem_info, ipc_info, newton_config, is_edge_edge=False
    )
    func_pairs_ccd(
        envs_mask, alpha, ipc_pairs, fem_state, ipc_state, fem_info, ipc_info, newton_config, is_edge_edge=True
    )


@qd.kernel(fastcache=True)
def kernel_contact_partition(envs_mask: qd.Tensor, fem_state: FEMState, ipc_state: IPCState, fem_info: FEMInfo):
    """
    Group the contact bodies of every environment selected by their links, and scale the Newton direction of the
    vertices of every group by the smallest time of contact of its bodies (see IPCBodies).

    The groups take the smallest body linked to them, propagated over as many passes as there are bodies.
    """
    n_bodies = ipc_state.bodies.toc.shape[0]
    for i_b in range(ipc_state.bodies.toc.shape[1]):
        if envs_mask[i_b]:
            for i_body in range(n_bodies):
                ipc_state.bodies.group[i_body, i_b] = i_body
            for _ in range(n_bodies):
                for i_body in range(n_bodies):
                    for j_body in range(n_bodies):
                        if ipc_state.bodies.is_linked[i_body, j_body, i_b]:
                            ipc_state.bodies.group[i_body, i_b] = qd.min(
                                ipc_state.bodies.group[i_body, i_b], ipc_state.bodies.group[j_body, i_b]
                            )
            for i_body in range(n_bodies):
                alpha = gs.qd_float(1.0)
                for j_body in range(n_bodies):
                    if ipc_state.bodies.group[j_body, i_b] == ipc_state.bodies.group[i_body, i_b]:
                        alpha = qd.min(alpha, ipc_state.bodies.toc[j_body, i_b])
                ipc_state.bodies.alpha[i_body, i_b] = alpha
    for i_v, i_b in qd.ndrange(*fem_state.vertices.pos.shape):
        if envs_mask[i_b]:
            i_body = fem_info.cloth_vertices.contact_body_idx[i_v]
            fem_state.cloth_vertices.displacement[i_v, i_b] = (
                ipc_state.bodies.alpha[i_body, i_b] * fem_state.cloth_vertices.displacement[i_v, i_b]
            )


# ------------------------------------------------------------------------------------
# ----------------------------------- host side --------------------------------------
# ------------------------------------------------------------------------------------


class IPCContact:
    """
    The broad phase of the contact of the cloth and the pairs it finds, owned by the Newton solver.

    The triangles and the edges of the cloth of every environment span one tree each (see genesis/engine/bvh.py). A query
    collects the raw hits of the boxes, which a filter turns into the candidate pairs (see 'kernel_contact_filter'). The
    raw hits and the pairs start at 'IPCOptions.init_collision_pair_capacity' per environment, and once a search finds
    more, their capacity grows to 'IPCOptions.capacity_grow_factor' times the count and the search runs again.
    """

    def __init__(self, n_envs: int, n_vertices: int, fem_info: FEMInfo, ipc_options) -> None:
        self._n_envs = n_envs
        self._fem_info = fem_info
        self._grow_factor = ipc_options.capacity_grow_factor
        n_edges = fem_info.cloth_edges.verts_idx.shape[0]
        self._triangles_bvh = get_bvh_data(n_envs, fem_info.cloth_triangles.verts_idx.shape[0])
        self._edges_bvh = get_bvh_data(n_envs, n_edges)
        self._points_query = BVHLeaves(sizes=dict(bvh_trees=n_envs, bvh_leaves=n_vertices))
        self._edges_query = BVHLeaves(sizes=dict(bvh_trees=n_envs, bvh_leaves=n_edges))
        # FIXME: quadrants#1074 - a launch is cached under the identity of its struct arguments, which a new struct could
        # take over from a dead one, so the structs replaced stay referenced, with their arrays freed.
        self._replaced: list = []
        capacity = ipc_options.init_collision_pair_capacity * n_envs
        self._pt_hits: BVHQueryResults | None = None
        self._ee_hits: BVHQueryResults | None = None
        self._pt_visitor: PointTriangleQuery | None = None
        self._ee_visitor: EdgeEdgeQuery | None = None
        self._grow_hits(capacity)
        self.pairs = IPCPairs(sizes={PT_PAIRS: capacity, EE_PAIRS: capacity})
        self.friction_pairs = IPCFrictionPairs(sizes={PT_PAIRS: capacity, EE_PAIRS: capacity})
        # The numbers of point-triangle and edge-edge candidates and friction pairs, read back after each search
        self.n_pairs = (0, 0)
        self.n_friction_pairs = (0, 0)

    def _retire(self, struct: Data) -> None:
        for _, array in iter_state_arrays(struct):
            qd_destroy_data(array)
        self._replaced.append(struct)

    def _grow_hits(self, capacity: int) -> None:
        for hits in (self._pt_hits, self._ee_hits):
            if hits is not None:
                self._retire(hits)
        self._pt_hits = BVHQueryResults(sizes=dict(query_results=capacity, bvh_trees=self._n_envs))
        self._ee_hits = BVHQueryResults(sizes=dict(query_results=capacity, bvh_trees=self._n_envs))
        self._pt_visitor = PointTriangleQuery(self._pt_hits, self._fem_info.cloth_triangles.verts_idx)
        self._ee_visitor = EdgeEdgeQuery(self._ee_hits, self._fem_info.cloth_edges.verts_idx)

    def _grown_capacity(self, count: int, capacity: int) -> int:
        return max(int(math.ceil(count * self._grow_factor)), capacity)

    @property
    def n_triplets(self) -> int:
        """The triplets of the Hessians of the candidate and friction pairs (see 'func_contact_assemble')."""
        return PAIR_TRIPLETS * (sum(self.n_pairs) + sum(self.n_friction_pairs))

    def search(
        self, fem_state: FEMState, ipc_state: IPCState, fem_info: FEMInfo, ipc_info: IPCInfo, newton_config, is_swept
    ) -> None:
        """Find the candidate pairs over the sweep of the vertices along the Newton direction, or at their positions,
        growing the capacities until they hold every hit and every pair."""
        kernel_contact_boxes(
            fem_state,
            fem_info,
            self._triangles_bvh.state.leaves,
            self._edges_bvh.state.leaves,
            self._points_query,
            self._edges_query,
            is_swept,
        )
        for bvh in (self._triangles_bvh, self._edges_bvh):
            build_bvh(bvh.state, bvh.config, eps=gs.EPS)
        is_complete = False
        while not is_complete:
            kernel_contact_query(
                self._points_query,
                self._triangles_bvh.state.tree,
                self._edges_query,
                self._edges_bvh.state.tree,
                self._pt_visitor,
                self._ee_visitor,
            )
            share = self._pt_hits.triplets.shape[0] // self._n_envs
            n_hits = max(int(qd_to_numpy(hits.trees_count).max()) for hits in (self._pt_hits, self._ee_hits))
            is_complete = n_hits <= share
            if not is_complete:
                self._grow_hits(self._grown_capacity(n_hits, share) * self._n_envs)
        is_complete = False
        while not is_complete:
            kernel_contact_filter(
                self._pt_hits,
                self._ee_hits,
                self.pairs,
                fem_state,
                ipc_state,
                fem_info,
                ipc_info,
                newton_config,
                is_swept,
            )
            n_pt, n_ee = (int(count) for count in qd_to_numpy(self.pairs.counts))
            capacity = self.pairs.pt.shape[0]
            is_complete = max(n_pt, n_ee) <= capacity
            if not is_complete:
                self._retire(self.pairs)
                capacity = self._grown_capacity(max(n_pt, n_ee), capacity)
                self.pairs = IPCPairs(sizes={PT_PAIRS: capacity, EE_PAIRS: capacity})
        self.n_pairs = (n_pt, n_ee)

    def find_intersections(self, fem_state: FEMState, fem_info: FEMInfo, ipc_info: IPCInfo) -> np.ndarray:
        """Return the (environment, edge, triangle) triplets of the edges of the cloth that cross one of its triangles at
        the positions of its vertices, in no particular order."""
        kernel_contact_boxes(
            fem_state,
            fem_info,
            self._triangles_bvh.state.leaves,
            self._edges_bvh.state.leaves,
            self._points_query,
            self._edges_query,
            False,
        )
        build_bvh(self._triangles_bvh.state, self._triangles_bvh.config, eps=gs.EPS)
        capacity = self._pt_hits.triplets.shape[0]
        triplets = np.zeros((0, 3), dtype=gs.np_int)
        is_complete = False
        while not is_complete:
            hits = BVHQueryResults(sizes=dict(query_results=capacity, bvh_trees=self._n_envs))
            visitor = EdgeTriangleQuery(
                hits,
                fem_info.cloth_edges.verts_idx,
                fem_info.cloth_triangles.verts_idx,
                fem_state.vertices.pos_next,
                ipc_info.intersection_tolerance,
            )
            kernel_contact_intersections(self._edges_query, self._triangles_bvh.state.tree, visitor)
            share = capacity // self._n_envs
            n_hits = int(qd_to_numpy(hits.trees_count).max())
            is_complete = n_hits <= share
            if is_complete:
                # The hits are (environment, triangle, edge) triplets
                triplets = qd_to_numpy(hits.triplets, copy=True)[: int(qd_to_numpy(hits.count)[0]), (0, 2, 1)]
            else:
                capacity = self._grown_capacity(n_hits, share) * self._n_envs
            self._retire(hits)
        return triplets

    def set_friction_pairs(
        self, fem_state: FEMState, fem_info: FEMInfo, ipc_info: IPCInfo, penetration_errno: int, errno: qd.Tensor
    ) -> None:
        """Set the friction pairs of the substep from the candidate pairs at its start (see IPCFrictionPairs)."""
        capacity = self.pairs.pt.shape[0]
        if self.friction_pairs.pt.shape[0] < capacity:
            self._retire(self.friction_pairs)
            self.friction_pairs = IPCFrictionPairs(sizes={PT_PAIRS: capacity, EE_PAIRS: capacity})
        kernel_contact_friction_pairs(
            self.pairs, self.friction_pairs, fem_state, fem_info, ipc_info, penetration_errno, errno
        )
        self.n_friction_pairs = tuple(int(count) for count in qd_to_numpy(self.friction_pairs.counts))

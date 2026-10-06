import numpy as np
import pytest
import quadrants as qd

import genesis as gs
from genesis.engine.systems.consistent_ipc_contact import (
    ConsistentIPCContactConstitution,
    cipc_count_active_pt_kernel,
)
from genesis.engine.systems.contact import ContactTabular
from genesis.engine.systems.contact_function.cipc_simplex_scatter import (
    cipc_scatter_triplets_upper,
    cipc_scatter_triplets_upper_rank1,
)
from genesis.engine.systems.contact_function.codim_thickness import pair_thickness_pt
from genesis.engine.systems.contact_function.distance_flag import (
    _popcount4,
    pt_distance_flag,
    pt_flagged_distance2,
)
from genesis.engine.systems.contact_function.gipc_barrier import (
    gipc_barrier_energy,
    gipc_barrier_energy_mollified,
    gipc_barrier_first_derivative,
    gipc_barrier_second_derivative,
    gipc_normal_force,
)
from genesis.engine.systems.contact_function.gipc_barrier_gradient_hessian import (
    gipc_barrier_grad_hess_ee,
    gipc_barrier_grad_hess_ee_rank1,
    gipc_barrier_grad_hess_pp,
)
from genesis.engine.systems.contact_function.halfplane_contact import (
    halfplane_barrier_gradient,
    halfplane_barrier_hessian,
    halfplane_signed_distance,
)
from genesis.engine.systems.contact_function.pair_d_hat import pair_d_hat_pt
from genesis.engine.systems.contact_system import ContactSystem
from genesis.engine.systems.finite_element.fem_contact_assemble import (
    distribute_fem_fem_kernel,
    distribute_fem_gradient_kernel,
)
from genesis.engine.systems.global_linear_system import GlobalLinearSystem
from genesis.engine.systems.global_surface_manager import GlobalSurfaceManager
from genesis.engine.systems.global_vertex_manager import GlobalVertexManager
from genesis.utils.misc import qd_to_numpy


@qd.data_oriented
class FEMContactFixture:
    def __init__(self, n_vertices):
        self.global_vert_offset = qd.ndarray(qd.i32, shape=())
        self.n_fem_verts = qd.ndarray(qd.i32, shape=())
        self.dof_offset = qd.ndarray(qd.i32, shape=())
        self.is_fixed = qd.ndarray(qd.i32, shape=(n_vertices,))
        self.global_vert_offset.from_numpy(np.array(0, dtype=np.int32))
        self.n_fem_verts.from_numpy(np.array(n_vertices, dtype=np.int32))
        self.dof_offset.from_numpy(np.array(0, dtype=np.int32))
        self.is_fixed.from_numpy(np.zeros(n_vertices, dtype=np.int32))


@qd.data_oriented
class TripletScatterFixture:
    def __init__(self):
        self.n_contact_triplets = qd.ndarray(qd.i32, shape=())
        self.contact_triplet_rows = qd.ndarray(qd.i32, shape=(10,))
        self.contact_triplet_cols = qd.ndarray(qd.i32, shape=(10,))
        self.contact_triplet_values = qd.ndarray(qd.f64, shape=(10, 3, 3))


@qd.kernel(fastcache=True)
def evaluate_contact_functions(
    parameters: qd.types.ndarray(qd.f64, ndim=1),
    normal: qd.types.ndarray(qd.f64, ndim=1),
    output: qd.types.ndarray(qd.f64, ndim=1),
):
    for _ in range(1):
        D = parameters[0]
        d_hat = parameters[1]
        xi = parameters[2]
        kappa = parameters[3]
        I1 = parameters[4]
        eps_x = parameters[5]
        output[0] = gipc_barrier_energy(D, d_hat, xi, kappa)
        output[1] = gipc_barrier_first_derivative(D, d_hat, xi, kappa)
        output[2] = gipc_barrier_second_derivative(D, d_hat, xi, kappa)
        output[3] = gipc_normal_force(D, d_hat, xi, kappa)
        output[4] = gipc_barrier_energy_mollified(D, d_hat, xi, kappa, I1, eps_x)

        distance = halfplane_signed_distance(
            parameters[6],
            parameters[7],
            parameters[8],
            parameters[9],
            parameters[10],
            parameters[11],
            normal[0],
            normal[1],
            normal[2],
        )
        output[5] = distance
        gradient = qd.Vector.zero(qd.f64, 3)
        hessian = qd.Vector.zero(qd.f64, 9)
        halfplane_barrier_gradient(output[1], distance, normal[0], normal[1], normal[2], gradient)
        halfplane_barrier_hessian(output[2], output[1], distance * distance, normal[0], normal[1], normal[2], hessian)
        for axis in qd.static(range(3)):
            output[6 + axis] = gradient[axis]
        for component in qd.static(range(9)):
            output[9 + component] = hessian[component]


@qd.kernel(fastcache=True)
def evaluate_pp_barrier(
    positions: qd.types.ndarray(qd.f64, ndim=1),
    parameters: qd.types.ndarray(qd.f64, ndim=1),
    output: qd.types.ndarray(qd.f64, ndim=1),
):
    for _ in range(1):
        gradient = qd.Vector.zero(qd.f64, 6)
        hessian = qd.Vector.zero(qd.f64, 36)
        gipc_barrier_grad_hess_pp(
            positions[0],
            positions[1],
            positions[2],
            positions[3],
            positions[4],
            positions[5],
            parameters[0],
            parameters[1],
            parameters[2],
            gradient,
            hessian,
        )
        for component in qd.static(range(6)):
            output[component] = gradient[component]
        for component in qd.static(range(36)):
            output[6 + component] = hessian[component]


@qd.kernel(fastcache=True)
def evaluate_ee_rank1_barrier(
    positions: qd.types.ndarray(qd.f64, ndim=1),
    parameters: qd.types.ndarray(qd.f64, ndim=1),
    output: qd.types.ndarray(qd.f64, ndim=1),
):
    for _ in range(1):
        gradient = qd.Vector.zero(qd.f64, 12)
        hessian = qd.Vector.zero(qd.f64, 144)
        gipc_barrier_grad_hess_ee(
            positions[0],
            positions[1],
            positions[2],
            positions[3],
            positions[4],
            positions[5],
            positions[6],
            positions[7],
            positions[8],
            positions[9],
            positions[10],
            positions[11],
            parameters[0],
            parameters[1],
            parameters[2],
            gradient,
            hessian,
        )
        values = qd.Vector.zero(qd.f64, 12)
        grad_scale = qd.Vector.zero(qd.f64, 1)
        hess_coef = qd.Vector.zero(qd.f64, 1)
        gipc_barrier_grad_hess_ee_rank1(
            positions[0],
            positions[1],
            positions[2],
            positions[3],
            positions[4],
            positions[5],
            positions[6],
            positions[7],
            positions[8],
            positions[9],
            positions[10],
            positions[11],
            parameters[0],
            parameters[1],
            parameters[2],
            values,
            grad_scale,
            hess_coef,
        )
        for component in qd.static(range(12)):
            output[component] = gradient[component]
            output[156 + component] = values[component] * grad_scale[0]
        for row in qd.static(range(12)):
            for column in qd.static(range(12)):
                component = row * 12 + column
                output[12 + component] = hessian[component]
                output[168 + component] = hess_coef[0] * values[row] * values[column]


@qd.kernel(fastcache=True)
def evaluate_rank1_triplet_scatter(
    dense: qd.template(),
    rank1: qd.template(),
    values_input: qd.types.ndarray(qd.f64, ndim=1),
    ids_input: qd.types.ndarray(qd.i32, ndim=1),
    coefficient: qd.f64,
):
    for _ in range(1):
        dense.n_contact_triplets[()] = 0
        rank1.n_contact_triplets[()] = 0
        values = qd.Vector.zero(qd.f64, 12)
        ids = qd.Vector.zero(qd.i32, 4)
        hessian = qd.Vector.zero(qd.f64, 144)
        for component in qd.static(range(12)):
            values[component] = values_input[component]
        for point in qd.static(range(4)):
            ids[point] = ids_input[point]
        for row in qd.static(range(12)):
            for column in qd.static(range(12)):
                hessian[row * 12 + column] = coefficient * values[row] * values[column]
        cipc_scatter_triplets_upper(dense, hessian, qd.i32(0xF), ids)
        cipc_scatter_triplets_upper_rank1(rank1, values, coefficient, ids)


@qd.kernel
def evaluate_consistent_ipc_constitution(
    contact: qd.template(),
    constitution: qd.template(),
    surface: qd.template(),
    vertex: qd.template(),
):
    for _ in range(1):
        contact.n_counted_doublets[()] = 0
        contact.n_counted_triplets[()] = 0
        contact.n_contact_doublets[()] = 0
        contact.n_contact_triplets[()] = 0
        contact.n_active_pairs[()] = 0
        contact.barrier_energy[()] = 0.0
        contact.friction_energy[()] = 0.0
    constitution.count_active(contact, surface, vertex)
    constitution.filter_assemble(contact, surface, vertex)
    constitution.contact_energy(contact, surface, vertex)


@qd.kernel
def evaluate_friction_snapshot(
    contact: qd.template(),
    constitution: qd.template(),
    surface: qd.template(),
    vertex: qd.template(),
):
    constitution.friction_snapshot(contact, surface, vertex)


@qd.kernel
def evaluate_pt_count_active(
    contact: qd.template(),
    surface: qd.template(),
    vertex: qd.template(),
):
    for _ in range(1):
        contact.n_counted_doublets[()] = 0
        contact.n_counted_triplets[()] = 0
    cipc_count_active_pt_kernel(contact, surface, vertex)


@qd.kernel
def evaluate_contact_sort_reduce(contact: qd.template()):
    contact.sort_reduce()


@qd.kernel
def evaluate_contact_distribute(
    contact: qd.template(),
    fem: qd.template(),
    global_linear_system: qd.template(),
):
    global_linear_system.compute_n_triplets(contact)
    global_linear_system.zero_rhs()
    global_linear_system.zero_triplet()
    distribute_fem_gradient_kernel(contact, fem, global_linear_system)
    distribute_fem_fem_kernel(contact, fem, global_linear_system)
    global_linear_system.body_sort_reduce()


@qd.kernel
def evaluate_adaptive_kappa_newton_tick(contact: qd.template()):
    contact.adaptive_kappa_newton_tick()


@qd.kernel
def evaluate_contact_ccd(contact: qd.template()):
    contact.init_ccd()
    contact.ccd()


@qd.kernel
def evaluate_halfplane_query(contact: qd.template()):
    contact.reset_collision_counts()
    contact.halfplane_query()


@qd.kernel(fastcache=True)
def inspect_pt_pair(
    contact: qd.template(),
    surface: qd.template(),
    vertex: qd.template(),
    output: qd.types.ndarray(qd.f64, ndim=1),
):
    for _ in range(1):
        vertex_id = surface.surf_verts[contact.pairs_pt[0, 0]]
        face = contact.pairs_pt[0, 1]
        v1 = surface.surf_triangles[face, 0]
        v2 = surface.surf_triangles[face, 1]
        v3 = surface.surf_triangles[face, 2]
        positions = qd.Matrix.zero(qd.f64, 4, 3)
        ids = qd.Vector([vertex_id, v1, v2, v3])
        for point in qd.static(range(4)):
            for axis in qd.static(range(3)):
                positions[point, axis] = vertex.positions[ids[point], axis]
        flag = pt_distance_flag(
            positions[0, 0],
            positions[0, 1],
            positions[0, 2],
            positions[1, 0],
            positions[1, 1],
            positions[1, 2],
            positions[2, 0],
            positions[2, 1],
            positions[2, 2],
            positions[3, 0],
            positions[3, 1],
            positions[3, 2],
        )
        output[0] = qd.f64(flag)
        output[1] = pt_flagged_distance2(flag, positions)
        output[2] = pair_d_hat_pt(vertex.d_hats, contact.d_hat[()], vertex_id, v1, v2, v3)
        output[3] = pair_thickness_pt(vertex.thicknesses, vertex_id, v1, v2, v3)
        output[4] = qd.f64(contact.n_pairs_pt[()])
        output[5] = qd.f64(_popcount4(flag))


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("backend", [gs.gpu])
def test_barrier_and_halfplane_functions():
    D = 0.006**2
    d_hat = 0.01
    xi = 0.001
    kappa = 1e4
    I1 = 0.25
    eps_x = 0.5
    point = np.array([0.2, -0.4, 0.006], dtype=np.float64)
    plane_point = np.zeros(3, dtype=np.float64)
    normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    parameters = qd.ndarray(qd.f64, shape=(12,))
    normal_buffer = qd.ndarray(qd.f64, shape=(3,))
    output = qd.ndarray(qd.f64, shape=(18,))
    parameters.from_numpy(np.array([D, d_hat, xi, kappa, I1, eps_x, *point, *plane_point], dtype=np.float64))
    normal_buffer.from_numpy(normal)

    evaluate_contact_functions(parameters, normal_buffer, output)
    actual = qd_to_numpy(output)

    d_tilde = D - xi * xi
    dH_tilde = d_hat * d_hat + 2.0 * d_hat * xi
    ratio = d_tilde / dH_tilde
    log_ratio = np.log(ratio)
    A = log_ratio + (ratio - 1.0) / ratio
    energy = kappa * (ratio - 1.0) ** 2 * log_ratio**2
    first = 2.0 * kappa * (ratio - 1.0) * log_ratio * A / dH_tilde
    second = 2.0 * kappa * (A * A + (ratio * ratio - 1.0) * log_ratio / (ratio * ratio)) / (dH_tilde * dH_tilde)
    distance = point[2]
    gradient = first * 2.0 * distance * normal
    coefficient = max(4.0 * distance * distance * second + 2.0 * first, 0.0)
    hessian = coefficient * np.outer(normal, normal)

    np.testing.assert_allclose(actual[0], energy, rtol=1e-12)
    np.testing.assert_allclose(actual[1], first, rtol=1e-12)
    np.testing.assert_allclose(actual[2], second, rtol=1e-12)
    np.testing.assert_allclose(actual[3], -first * 2.0 * np.sqrt(D), rtol=1e-12)
    np.testing.assert_allclose(
        actual[4],
        (-(I1 * I1) / (eps_x * eps_x) + 2.0 * I1 / eps_x) * energy,
        rtol=1e-12,
    )
    np.testing.assert_allclose(actual[5], distance, rtol=1e-12)
    np.testing.assert_allclose(actual[6:9], gradient, rtol=1e-12)
    np.testing.assert_allclose(actual[9:18].reshape(3, 3), hessian, rtol=1e-12)


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("backend", [gs.gpu])
def test_pp_barrier_derivatives():
    x = np.array([0.0, 0.0, 0.0, 0.006, 0.001, -0.0005], dtype=np.float64)
    d_hat = 0.01
    xi = 0.001
    kappa = 1e4
    positions = qd.ndarray(qd.f64, shape=(6,))
    parameters = qd.ndarray(qd.f64, shape=(3,))
    output = qd.ndarray(qd.f64, shape=(42,))
    positions.from_numpy(x)
    parameters.from_numpy(np.array([d_hat, xi, kappa], dtype=np.float64))
    evaluate_pp_barrier(positions, parameters, output)
    actual = qd_to_numpy(output)

    def energy(values):
        delta = values[:3] - values[3:]
        D = float(delta @ delta)
        d_tilde = D - xi * xi
        dH_tilde = d_hat * d_hat + 2.0 * d_hat * xi
        ratio = d_tilde / dH_tilde
        return kappa * (ratio - 1.0) ** 2 * np.log(ratio) ** 2

    epsilon = 1e-6
    gradient = np.zeros(6, dtype=np.float64)
    for row in range(6):
        offset = np.zeros(6, dtype=np.float64)
        offset[row] = epsilon
        gradient[row] = (energy(x + offset) - energy(x - offset)) / (2.0 * epsilon)

    np.testing.assert_allclose(actual[:6], gradient, rtol=2e-4, atol=1e-5)
    delta = x[:3] - x[3:]
    D = float(delta @ delta)
    d_tilde = D - xi * xi
    dH_tilde = d_hat * d_hat + 2.0 * d_hat * xi
    I5 = d_tilde / dH_tilde
    log_I5 = np.log(I5)
    A = log_I5 + (I5 - 1.0) / I5
    lambda0 = 8.0 * kappa * (I5 * A * A + (I5 * I5 - 1.0) * log_I5 / I5 + (I5 - 1.0) * log_I5 * A * 0.5)
    pfpx = np.concatenate((delta, -delta)) / (np.sqrt(D) * np.sqrt(dH_tilde))
    expected_hessian = lambda0 * D / d_tilde * np.outer(pfpx, pfpx)
    np.testing.assert_allclose(actual[6:].reshape(6, 6), expected_hessian, rtol=1e-11, atol=1e-8)


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("backend", [gs.gpu])
def test_ee_rank1_matches_dense_barrier():
    positions = qd.ndarray(qd.f64, shape=(12,))
    parameters = qd.ndarray(qd.f64, shape=(3,))
    output = qd.ndarray(qd.f64, shape=(312,))
    positions.from_numpy(
        np.array(
            [
                -0.5,
                0.0,
                0.0,
                0.5,
                0.0,
                0.0,
                0.0,
                -0.5,
                0.006,
                0.0,
                0.5,
                0.006,
            ],
            dtype=np.float64,
        )
    )
    parameters.from_numpy(np.array([0.01, 0.001, 1.0e4], dtype=np.float64))

    evaluate_ee_rank1_barrier(positions, parameters, output)
    actual = qd_to_numpy(output)
    np.testing.assert_allclose(actual[:12], actual[156:168], rtol=1e-12, atol=1e-11)
    np.testing.assert_allclose(actual[12:156], actual[168:312], rtol=1e-12, atol=1e-8)


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("backend", [gs.gpu])
def test_rank1_triplet_scatter_matches_dense():
    dense = TripletScatterFixture()
    rank1 = TripletScatterFixture()
    values = qd.ndarray(qd.f64, shape=(12,))
    ids = qd.ndarray(qd.i32, shape=(4,))
    values.from_numpy(np.linspace(-0.9, 1.3, 12, dtype=np.float64))
    ids.from_numpy(np.array([9, 2, 7, 1], dtype=np.int32))

    evaluate_rank1_triplet_scatter(dense, rank1, values, ids, 3.75)

    np.testing.assert_array_equal(qd_to_numpy(dense.n_contact_triplets), 10)
    np.testing.assert_array_equal(qd_to_numpy(rank1.n_contact_triplets), 10)
    np.testing.assert_array_equal(qd_to_numpy(rank1.contact_triplet_rows), qd_to_numpy(dense.contact_triplet_rows))
    np.testing.assert_array_equal(qd_to_numpy(rank1.contact_triplet_cols), qd_to_numpy(dense.contact_triplet_cols))
    np.testing.assert_allclose(
        qd_to_numpy(rank1.contact_triplet_values),
        qd_to_numpy(dense.contact_triplet_values),
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("backend", [gs.gpu])
def test_halfplane_query_respects_contact_tabular():
    vertex = GlobalVertexManager()
    vertex.init(1)
    vertex.positions.from_numpy(np.array([[0.0, 0.0, 0.006]], dtype=np.float64))
    vertex.trajectory_end_positions.from_numpy(np.array([[0.0, 0.0, -0.004]], dtype=np.float64))
    vertex.wire_thickness_data(np.array([0.001], dtype=np.float64))
    vertex.wire_d_hat_data(np.array([0.01], dtype=np.float64))

    surface = GlobalSurfaceManager()
    surface.wire_surface_data(
        np.empty((0, 3), dtype=np.int32),
        np.empty((0, 2), dtype=np.int32),
        np.array([0], dtype=np.int32),
    )

    tabular = ContactTabular()
    tabular.default_model(
        friction_rate=0.0,
        resistance=1.0e4,
        enable=False,
        enable_ee=False,
    )
    contact = ContactSystem()
    contact.vertex = vertex
    contact.surface = surface
    contact.wire_params(d_hat=0.01, kappa=1.0e4, init_pair_capacity=1)
    contact.wire_contact_tabular(tabular)
    contact.wire_halfplanes(
        np.array([[0.0, 0.0, 0.0]], dtype=np.float64),
        np.array([[0.0, 0.0, 1.0]], dtype=np.float64),
    )
    contact.set_contact_constitution(ConsistentIPCContactConstitution())
    contact.init(1)

    evaluate_halfplane_query(contact)

    assert int(qd_to_numpy(contact.n_pairs_ph)) == 0
    assert int(qd_to_numpy(contact.intersection_flag)) == 0


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("backend", [gs.gpu])
def test_consistent_ipc_halfplane_assembly():
    vertex = GlobalVertexManager()
    vertex.init(1)
    vertex.positions.from_numpy(np.array([[0.0, 0.0, 0.006]], dtype=np.float64))
    vertex.x_bar.from_numpy(np.array([[0.0, 0.0, 0.006]], dtype=np.float64))
    vertex.body_id.from_numpy(np.array([0], dtype=np.int32))
    vertex.wire_thickness_data(np.array([0.001], dtype=np.float64))
    vertex.wire_d_hat_data(np.array([0.01], dtype=np.float64))
    vertex.wire_is_fixed_data(np.array([0], dtype=np.int32))

    surface = GlobalSurfaceManager()
    surface.wire_surface_data(
        np.empty((0, 3), dtype=np.int32),
        np.empty((0, 2), dtype=np.int32),
        np.array([0], dtype=np.int32),
    )
    surface.wire_vert_dimensions(np.array([2], dtype=np.int32))
    surface.wire_area_weights(
        np.array([0.04], dtype=np.float64),
        np.empty(0, dtype=np.float64),
        np.empty(0, dtype=np.float64),
    )

    contact = ContactSystem()
    contact.vertex = vertex
    contact.surface = surface
    contact.wire_params(d_hat=0.01, kappa=1e4, init_pair_capacity=1)
    contact.set_dt_sq(0.01**2)
    contact.wire_friction_params(mu=0.05, eps_v=1e-2)
    contact.wire_contact_tabular(ContactTabular())
    contact.wire_halfplanes(
        np.array([[0.0, 0.0, 0.0]], dtype=np.float64),
        np.array([[0.0, 0.0, 1.0]], dtype=np.float64),
    )
    contact.set_adaptive_kappa("per-body", "newton", 1)
    constitution = ConsistentIPCContactConstitution()
    contact.set_contact_constitution(constitution)
    contact.init(1)
    contact.pairs_ph.from_numpy(np.array([[0, 0]], dtype=np.int32))
    contact.n_pairs_ph.from_numpy(np.array(1, dtype=np.int32))

    evaluate_friction_snapshot(contact, constitution, surface, vertex)
    vertex.positions.from_numpy(np.array([[0.001, 0.0, 0.006]], dtype=np.float64))
    evaluate_consistent_ipc_constitution(contact, constitution, surface, vertex)
    np.testing.assert_array_equal(qd_to_numpy(contact.n_counted_doublets), 1)
    np.testing.assert_array_equal(qd_to_numpy(contact.n_counted_triplets), 1)
    np.testing.assert_array_equal(qd_to_numpy(contact.n_friction_pairs_ph), 1)
    np.testing.assert_array_equal(qd_to_numpy(contact.n_friction_demand_doublets), 1)
    np.testing.assert_array_equal(qd_to_numpy(contact.n_friction_demand_triplets), 1)
    np.testing.assert_array_equal(qd_to_numpy(contact.n_contact_doublets), 2)
    np.testing.assert_array_equal(qd_to_numpy(contact.n_contact_triplets), 2)
    np.testing.assert_array_equal(qd_to_numpy(contact.n_active_pairs), 1)
    assert float(qd_to_numpy(contact.barrier_energy)) > 0.0
    assert float(qd_to_numpy(contact.friction_energy)) > 0.0
    np.testing.assert_array_equal(qd_to_numpy(contact.contact_doublet_vertices)[0], 0)
    assert qd_to_numpy(contact.contact_doublet_gradients)[0, 2] < 0.0
    contact.iter_body_min_gap.from_numpy(np.array([0.005], dtype=np.float64))
    evaluate_adaptive_kappa_newton_tick(contact)
    np.testing.assert_allclose(qd_to_numpy(contact.body_kappa_scale)[0], 2.0)
    np.testing.assert_array_equal(qd_to_numpy(contact.adaptive_kappa_grew), 1)
    vertex.trajectory_end_positions.from_numpy(np.array([[0.001, 0.0, -0.004]], dtype=np.float64))
    evaluate_contact_ccd(contact)
    np.testing.assert_allclose(qd_to_numpy(contact.ccd_alpha), 0.4, rtol=1e-12)


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("backend", [gs.gpu])
def test_consistent_ipc_point_triangle_assembly():
    positions_np = np.array(
        [
            [0.25, 0.25, 0.006],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    vertex = GlobalVertexManager()
    vertex.init(4)
    vertex.positions.from_numpy(positions_np)
    vertex.x_bar.from_numpy(positions_np)
    vertex.body_id.from_numpy(np.array([0, 1, 1, 1], dtype=np.int32))
    vertex.wire_thickness_data(np.full(4, 0.001, dtype=np.float64))
    vertex.wire_d_hat_data(np.full(4, 0.01, dtype=np.float64))
    vertex.wire_is_fixed_data(np.zeros(4, dtype=np.int32))

    surface = GlobalSurfaceManager()
    surface.wire_surface_data(
        np.array([[1, 2, 3]], dtype=np.int32),
        np.empty((0, 2), dtype=np.int32),
        np.arange(4, dtype=np.int32),
    )
    surface.wire_vert_dimensions(np.full(4, 2, dtype=np.int32))
    surface.wire_area_weights(
        np.full(4, 0.25, dtype=np.float64),
        np.empty(0, dtype=np.float64),
        np.array([0.5], dtype=np.float64),
    )

    contact = ContactSystem()
    contact.vertex = vertex
    contact.surface = surface
    contact.wire_params(d_hat=0.01, kappa=1e4, init_pair_capacity=1)
    contact.set_dt_sq(0.01**2)
    contact.wire_friction_params(mu=0.05, eps_v=1e-2)
    contact.wire_contact_tabular(ContactTabular())
    contact.wire_halfplanes(
        np.empty((0, 3), dtype=np.float64),
        np.empty((0, 3), dtype=np.float64),
    )
    contact.set_adaptive_kappa("off", "frame", 2)
    constitution = ConsistentIPCContactConstitution()
    contact.set_contact_constitution(constitution)
    contact.init(4)
    contact.pairs_pt.from_numpy(np.array([[0, 0]], dtype=np.int32))
    contact.n_pairs_pt.from_numpy(np.array(1, dtype=np.int32))

    evaluate_friction_snapshot(contact, constitution, surface, vertex)
    positions_np[0, 0] += 0.001
    vertex.positions.from_numpy(positions_np)
    pair_diagnostics = qd.ndarray(qd.f64, shape=(6,))
    inspect_pt_pair(contact, surface, vertex, pair_diagnostics)
    np.testing.assert_allclose(
        qd_to_numpy(pair_diagnostics),
        np.array([15.0, 0.006**2, 0.01, 0.002, 1.0, 4.0]),
        rtol=1e-12,
        atol=1e-15,
    )
    evaluate_pt_count_active(contact, surface, vertex)
    np.testing.assert_array_equal(qd_to_numpy(contact.n_counted_doublets), 4)
    np.testing.assert_array_equal(qd_to_numpy(contact.n_counted_triplets), 10)
    evaluate_consistent_ipc_constitution(contact, constitution, surface, vertex)
    np.testing.assert_array_equal(qd_to_numpy(contact.n_counted_doublets), 4)
    np.testing.assert_array_equal(qd_to_numpy(contact.n_counted_triplets), 10)
    np.testing.assert_array_equal(qd_to_numpy(contact.n_friction_pairs_pt), 1)
    np.testing.assert_array_equal(qd_to_numpy(contact.n_friction_demand_doublets), 4)
    np.testing.assert_array_equal(qd_to_numpy(contact.n_friction_demand_triplets), 10)
    np.testing.assert_array_equal(qd_to_numpy(contact.n_contact_doublets), 8)
    np.testing.assert_array_equal(qd_to_numpy(contact.n_contact_triplets), 20)
    np.testing.assert_array_equal(qd_to_numpy(contact.n_active_pairs), 1)
    assert float(qd_to_numpy(contact.barrier_energy)) > 0.0
    assert float(qd_to_numpy(contact.friction_energy)) > 0.0
    gradient = qd_to_numpy(contact.contact_doublet_gradients)[:8]
    np.testing.assert_allclose(gradient.sum(axis=0), 0.0, atol=1e-10)
    evaluate_contact_sort_reduce(contact)
    np.testing.assert_array_equal(qd_to_numpy(contact.n_unique_doublets), 4)
    np.testing.assert_array_equal(qd_to_numpy(contact.n_unique_triplets), 10)
    np.testing.assert_allclose(
        qd_to_numpy(contact.unique_doublet_gradients)[:4].sum(axis=0),
        0.0,
        atol=1e-10,
    )
    fem = FEMContactFixture(4)
    global_linear_system = GlobalLinearSystem()
    global_linear_system.init(4, 0, 10, 0, 1e-4)
    evaluate_contact_distribute(contact, fem, global_linear_system)
    np.testing.assert_array_equal(qd_to_numpy(global_linear_system.n_triplets), 10)
    np.testing.assert_array_equal(qd_to_numpy(global_linear_system.bcoo_nnz), 10)
    np.testing.assert_allclose(
        qd_to_numpy(global_linear_system.b_rhs).reshape(4, 3).sum(axis=0),
        0.0,
        atol=1e-10,
    )


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("backend", [gs.gpu])
def test_consistent_ipc_edge_edge_assembly():
    positions_np = np.array(
        [
            [-0.5, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.0, -0.5, 0.006],
            [0.0, 0.5, 0.006],
        ],
        dtype=np.float64,
    )
    vertex = GlobalVertexManager()
    vertex.init(4)
    vertex.positions.from_numpy(positions_np)
    vertex.x_bar.from_numpy(positions_np)
    vertex.body_id.from_numpy(np.array([0, 0, 1, 1], dtype=np.int32))
    vertex.wire_thickness_data(np.full(4, 0.001, dtype=np.float64))
    vertex.wire_d_hat_data(np.full(4, 0.01, dtype=np.float64))
    vertex.wire_is_fixed_data(np.zeros(4, dtype=np.int32))

    surface = GlobalSurfaceManager()
    surface.wire_surface_data(
        np.empty((0, 3), dtype=np.int32),
        np.array([[0, 1], [2, 3]], dtype=np.int32),
        np.arange(4, dtype=np.int32),
    )
    surface.wire_vert_dimensions(np.full(4, 2, dtype=np.int32))
    surface.wire_area_weights(
        np.full(4, 0.25, dtype=np.float64),
        np.full(2, 0.5, dtype=np.float64),
        np.empty(0, dtype=np.float64),
    )

    contact = ContactSystem()
    contact.vertex = vertex
    contact.surface = surface
    contact.wire_params(d_hat=0.01, kappa=1e4, init_pair_capacity=1)
    contact.set_dt_sq(0.01**2)
    contact.wire_friction_params(mu=0.05, eps_v=1e-2)
    contact.wire_contact_tabular(ContactTabular())
    contact.wire_halfplanes(
        np.empty((0, 3), dtype=np.float64),
        np.empty((0, 3), dtype=np.float64),
    )
    contact.set_adaptive_kappa("off", "frame", 2)
    constitution = ConsistentIPCContactConstitution()
    contact.set_contact_constitution(constitution)
    contact.init(4)
    contact.pairs_ee.from_numpy(np.array([[0, 1]], dtype=np.int32))
    contact.n_pairs_ee.from_numpy(np.array(1, dtype=np.int32))

    evaluate_friction_snapshot(contact, constitution, surface, vertex)
    positions_np[:2, 0] += 0.001
    vertex.positions.from_numpy(positions_np)
    evaluate_consistent_ipc_constitution(contact, constitution, surface, vertex)
    np.testing.assert_array_equal(qd_to_numpy(contact.n_counted_doublets), 4)
    np.testing.assert_array_equal(qd_to_numpy(contact.n_counted_triplets), 10)
    np.testing.assert_array_equal(qd_to_numpy(contact.n_friction_pairs_ee), 1)
    np.testing.assert_array_equal(qd_to_numpy(contact.n_contact_doublets), 8)
    np.testing.assert_array_equal(qd_to_numpy(contact.n_contact_triplets), 20)
    np.testing.assert_array_equal(qd_to_numpy(contact.n_active_pairs), 1)
    assert float(qd_to_numpy(contact.barrier_energy)) > 0.0
    assert float(qd_to_numpy(contact.friction_energy)) > 0.0
    gradient = qd_to_numpy(contact.contact_doublet_gradients)[:8]
    np.testing.assert_allclose(gradient.sum(axis=0), 0.0, atol=1e-10)

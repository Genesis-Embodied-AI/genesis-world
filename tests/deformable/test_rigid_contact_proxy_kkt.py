from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import quadrants as qd

import genesis as gs
from examples.newton_coupling.cloth_grid_asset import cloth_grid_asset
from genesis.engine.systems import ContactTabular, build_scene_engine
from genesis.engine.systems.bcoo_matrix import sort_reduce_bcoo, zero_bcoo_triplets
from genesis.engine.systems.bcoo_operations import sym_bcoo_spmv_naive
from genesis.engine.systems.contact_function.screw_ccd import screw_halfplane_ccd
from genesis.engine.systems.global_linear_system import (
    GlobalLinearSystem,
    derive_extents,
    zero_rhs,
)
from genesis.engine.systems.global_vertex_manager import GlobalVertexManager
from genesis.engine.systems.rigid_contact_assemble import (
    RigidContactAssemble,
    classify as classify_rigid_contact,
    distribute as distribute_rigid_contact,
)
from genesis.engine.systems.rigid_contact_proxy import (
    RigidContactProxyGeometry,
    RigidContactProxySystem,
    _populate_rigid_contact_proxy_data,
    forward_global_vertices as forward_proxy_global_vertices,
    initialize_global_vertices as initialize_proxy_global_vertices,
    initialize_proxy_state,
    prepare_constraint,
    prepare_metric,
    publish_trajectory_end_positions,
)
from genesis.engine.systems.rigid_contact_proxy_kkt import (
    fk_defect_prefix_cap,
    rigid_contact_proxy_constraint,
    rigid_contact_proxy_prepare_maps,
)
from genesis.engine.systems.rigid_joint_forest import (
    RigidJointForestSystem,
    _populate_rigid_joint_forest_data,
    clear_body_wrench,
    compute_endpoint_fk,
    expand_reduced_direction,
    finish_reduced_spmv,
    forest_precond_apply_level,
    forest_precond_apply_tree_shared,
    particular_spmv,
    prepare_particular,
    prepare_physical_direction as prepare_forest_physical_direction,
    project_physical_rhs,
    restrict_body_wrenches,
    restrict_proxy_wrenches,
)
from genesis.engine.systems.rigid_system import RigidSystem
from genesis.utils.misc import qd_to_numpy


def create_global_vertex_data(*args, **kwargs):
    system = GlobalVertexManager()
    system.wire_data(*args, **kwargs)
    system.init()
    return system.data


def create_rigid_system_data(rigid_solver):
    system = RigidSystem()
    system.wire_solver(rigid_solver)
    system.init(0)
    return system.data


def create_rigid_contact_proxy_data(**kwargs):
    data = RigidContactProxySystem.Data()
    _populate_rigid_contact_proxy_data(data, **kwargs)
    return data


def create_rigid_joint_forest_data(rigid_solver, rigid_data, **kwargs):
    kwargs.setdefault("n_proxy_pairs", int(kwargs["proxy_data"].n_pairs.to_numpy()))
    system = RigidJointForestSystem()
    system.fused_enabled = bool(kwargs["fused_enabled"])
    system.genesis_legacy_enabled = bool(kwargs["genesis_legacy_enabled"])
    _populate_rigid_joint_forest_data(system, rigid_solver, rigid_data, **kwargs)
    return system.data


def create_rigid_contact_assemble_data(contact):
    system = RigidContactAssemble()
    system.contact_system = SimpleNamespace(data=contact, genesis_legacy_sort_reduce=False)
    system.init()
    return system


@qd.data_oriented
class ProxyContactFixture:
    def __init__(self):
        self.genesis_legacy_sort_reduce_host = False
        self.n_unique_doublets = qd.ndarray(qd.i32, shape=())
        self.n_unique_triplets = qd.ndarray(qd.i32, shape=())
        self.unique_doublet_vertices = qd.ndarray(qd.i32, shape=(3,))
        self.unique_doublet_gradients = qd.ndarray(qd.f64, shape=(3, 3))
        self.unique_triplet_rows = qd.ndarray(qd.i32, shape=(3,))
        self.unique_triplet_cols = qd.ndarray(qd.i32, shape=(3,))
        self.unique_triplet_values = qd.ndarray(qd.f64, shape=(3, 3, 3))


@qd.data_oriented
class ProxyFEMFixture:
    def __init__(self):
        self.dof_offset = qd.ndarray(qd.i32, shape=())
        self.is_fixed = qd.ndarray(qd.i32, shape=(1,))


@qd.data_oriented
class ProxyForestFixture:
    def __init__(self):
        self.proxy_dof_offset = qd.ndarray(qd.i32, shape=())


def _skew(value):
    x, y, z = value
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def _quat_exp(rotation):
    angle = np.linalg.norm(rotation)
    if angle < 1.0e-12:
        return np.array([1.0, *(0.5 * rotation)], dtype=np.float64)
    half = 0.5 * angle
    return np.array([np.cos(half), *(np.sin(half) * rotation / angle)], dtype=np.float64)


def _quat_mul(left, right):
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return np.array(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        dtype=np.float64,
    )


def _left_jacobian(rotation):
    theta_sq = float(rotation @ rotation)
    hat = _skew(rotation)
    if theta_sq < 1.0e-10:
        return np.eye(3) + 0.5 * hat + (1.0 / 6.0) * hat @ hat
    theta = np.sqrt(theta_sq)
    return (
        np.eye(3) + (1.0 - np.cos(theta)) / theta_sq * hat + (theta - np.sin(theta)) / (theta_sq * theta) * (hat @ hat)
    )


def _right_jacobian_inverse(rotation):
    theta_sq = float(rotation @ rotation)
    hat = _skew(-rotation)
    if theta_sq < 1.0e-10:
        coefficient = 1.0 / 12.0 + theta_sq / 720.0
    else:
        theta = np.sqrt(theta_sq)
        half = 0.5 * theta
        coefficient = (1.0 - half / np.tan(half)) / theta_sq
    return np.eye(3) - 0.5 * hat + coefficient * hat @ hat


@qd.kernel(fastcache=True)
def evaluate_proxy_maps(
    mechanism_position: qd.types.ndarray(qd.f64, ndim=1),
    mechanism_quaternion: qd.types.ndarray(qd.f64, ndim=1),
    proxy_position: qd.types.ndarray(qd.f64, ndim=1),
    proxy_quaternion: qd.types.ndarray(qd.f64, ndim=1),
    output: qd.types.ndarray(qd.f64, ndim=1),
):
    for _ in range(1):
        mechanism_t = qd.Vector.zero(qd.f64, 3)
        mechanism_q = qd.Vector.zero(qd.f64, 4)
        proxy_t = qd.Vector.zero(qd.f64, 3)
        proxy_q = qd.Vector.zero(qd.f64, 4)
        for axis in qd.static(range(3)):
            mechanism_t[axis] = mechanism_position[axis]
            proxy_t[axis] = proxy_position[axis]
        for axis in qd.static(range(4)):
            mechanism_q[axis] = mechanism_quaternion[axis]
            proxy_q[axis] = proxy_quaternion[axis]

        translation = qd.Vector.zero(qd.f64, 3)
        rotation = qd.Vector.zero(qd.f64, 3)
        rigid_contact_proxy_constraint(
            mechanism_t,
            mechanism_q,
            proxy_t,
            proxy_q,
            translation,
            rotation,
        )
        constraint = qd.Vector.zero(qd.f64, 6)
        tangent_map = qd.Matrix.zero(qd.f64, 6, 6)
        normal_map = qd.Matrix.zero(qd.f64, 6, 6)
        particular = qd.Vector.zero(qd.f64, 6)
        rigid_contact_proxy_prepare_maps(
            translation,
            rotation,
            constraint,
            tangent_map,
            normal_map,
            particular,
        )
        for row in qd.static(range(6)):
            output[row] = constraint[row]
            output[6 + row] = particular[row]
            for column in qd.static(range(6)):
                output[12 + row * 6 + column] = tangent_map[row, column]
                output[48 + row * 6 + column] = normal_map[row, column]


@qd.kernel(fastcache=True)
def evaluate_fk_prefix_caps(
    inputs: qd.types.ndarray(qd.f64, ndim=2),
    outputs: qd.types.ndarray(qd.f64, ndim=1),
):
    for index in range(inputs.shape[0]):
        outputs[index] = fk_defect_prefix_cap(
            inputs[index, 0],
            inputs[index, 1],
            inputs[index, 2],
            inputs[index, 3],
        )


@qd.kernel(fastcache=True)
def initialize_and_prepare_proxy(
    proxy: qd.template(),
    rigid: qd.template(),
    forest: qd.template(),
    vertex: qd.template(),
):
    compute_endpoint_fk(forest, rigid, proxy)
    initialize_proxy_state(proxy, rigid, forest)
    prepare_metric(proxy, rigid, forest)
    prepare_constraint(proxy, rigid, forest)
    initialize_proxy_global_vertices(proxy, rigid, forest, vertex)


@qd.kernel(fastcache=True)
def publish_proxy_trajectory(
    proxy: qd.template(),
    rigid: qd.template(),
    forest: qd.template(),
    vertex: qd.template(),
):
    publish_trajectory_end_positions(proxy, rigid, forest, vertex)


@qd.kernel(fastcache=True)
def prepare_proxy_constraint(proxy: qd.template(), rigid: qd.template(), forest: qd.template()):
    prepare_constraint(proxy, rigid, forest)


@qd.kernel(fastcache=True)
def evaluate_forest_virtual_work(
    forest: qd.template(),
    rigid: qd.template(),
    proxy: qd.template(),
    reduced_direction: qd.types.ndarray(qd.f64, ndim=1),
    proxy_wrench: qd.types.ndarray(qd.f64, ndim=1),
    reduced_wrench: qd.types.ndarray(qd.f64, ndim=1),
    genesis_legacy_enabled: qd.template(),
    use_fused_tree_path: qd.template(),
    n_links: qd.template(),
):
    compute_endpoint_fk(forest, rigid, proxy)
    expand_reduced_direction(
        forest,
        rigid,
        proxy,
        reduced_direction,
        genesis_legacy_enabled,
        use_fused_tree_path,
        n_links,
    )
    clear_body_wrench(forest, rigid, proxy)
    for component in range(6):
        forest.body_wrench[proxy.proxy_body[0], component] = proxy_wrench[component]
    restrict_proxy_wrenches(forest, rigid, proxy)
    for dof in range(reduced_wrench.shape[0]):
        reduced_wrench[dof] = 0.0
    restrict_body_wrenches(
        forest,
        rigid,
        proxy,
        reduced_wrench,
        genesis_legacy_enabled,
        use_fused_tree_path,
        n_links,
    )


@qd.kernel(fastcache=True)
def evaluate_forest_transform(
    forest: qd.template(),
    rigid: qd.template(),
    proxy: qd.template(),
    reduced_direction: qd.types.ndarray(qd.f64, ndim=1),
    body_wrench: qd.types.ndarray(qd.f64, ndim=2),
    reduced_wrench: qd.types.ndarray(qd.f64, ndim=1),
    genesis_legacy_enabled: qd.template(),
    use_fused_tree_path: qd.template(),
    n_links: qd.template(),
):
    compute_endpoint_fk(forest, rigid, proxy)
    expand_reduced_direction(
        forest,
        rigid,
        proxy,
        reduced_direction,
        genesis_legacy_enabled,
        use_fused_tree_path,
        n_links,
    )
    for body in range(forest.n_mechanism_bodies[()]):
        for component in qd.static(range(6)):
            forest.body_wrench[body, component] = body_wrench[body, component]
    for dof in range(forest.total_dof[()]):
        reduced_wrench[dof] = 0.0
    restrict_body_wrenches(
        forest,
        rigid,
        proxy,
        reduced_wrench,
        genesis_legacy_enabled,
        use_fused_tree_path,
        n_links,
    )


@qd.kernel(fastcache=True)
def apply_forest_preconditioner_level(
    forest: qd.template(),
    rigid: qd.template(),
    proxy: qd.template(),
    residual: qd.types.ndarray(qd.f64, ndim=1),
    result: qd.types.ndarray(qd.f64, ndim=1),
    n_links: qd.template(),
):
    forest_precond_apply_level(forest, rigid, proxy, residual, result, n_links)


@qd.kernel(fastcache=True)
def apply_forest_preconditioner_tree(
    forest: qd.template(),
    rigid: qd.template(),
    proxy: qd.template(),
    residual: qd.types.ndarray(qd.f64, ndim=1),
    result: qd.types.ndarray(qd.f64, ndim=1),
):
    forest_precond_apply_tree_shared(forest, rigid, proxy, residual, result)


@qd.kernel(fastcache=True)
def apply_reduced_contact_operator(
    forest: qd.template(),
    rigid: qd.template(),
    proxy: qd.template(),
    linear_system_data: qd.template(),
    direction: qd.types.ndarray(qd.f64, ndim=1),
    result: qd.types.ndarray(qd.f64, ndim=1),
    genesis_legacy_enabled: qd.template(),
    use_fused_tree_path: qd.template(),
    n_links: qd.template(),
):
    for dof in range(forest.total_dof[()]):
        result[dof] = 0.0
        forest.physical_Ap[dof] = 0.0
    prepare_forest_physical_direction(
        forest,
        rigid,
        proxy,
        direction,
        genesis_legacy_enabled,
        use_fused_tree_path,
        n_links,
    )
    sym_bcoo_spmv_naive(linear_system_data.matrix, forest.physical_p, forest.physical_Ap)
    finish_reduced_spmv(
        forest,
        rigid,
        proxy,
        direction,
        result,
        genesis_legacy_enabled,
        use_fused_tree_path,
        n_links,
    )


@qd.kernel(fastcache=True)
def project_particular_rhs(
    forest: qd.template(),
    rigid: qd.template(),
    proxy: qd.template(),
    linear_system_data: qd.template(),
    genesis_legacy_enabled: qd.template(),
    use_fused_tree_path: qd.template(),
    n_links: qd.template(),
):
    for dof in range(forest.total_dof[()]):
        linear_system_data.b_rhs[dof] = 0.0
    prepare_particular(forest, rigid, proxy)
    particular_spmv(forest, rigid, proxy, linear_system_data)
    project_physical_rhs(
        forest,
        rigid,
        proxy,
        linear_system_data,
        genesis_legacy_enabled,
        use_fused_tree_path,
        n_links,
    )


@qd.kernel(fastcache=True)
def prepare_physical_direction(
    forest: qd.template(),
    rigid: qd.template(),
    proxy: qd.template(),
    direction: qd.types.ndarray(qd.f64, ndim=1),
    genesis_legacy_enabled: qd.template(),
    use_fused_tree_path: qd.template(),
    n_links: qd.template(),
):
    prepare_forest_physical_direction(
        forest,
        rigid,
        proxy,
        direction,
        genesis_legacy_enabled,
        use_fused_tree_path,
        n_links,
    )


@qd.kernel(fastcache=True)
def assemble_proxy_contact(
    route: qd.template(),
    proxy: qd.template(),
    forest: qd.template(),
    vertex: qd.template(),
    contact: qd.template(),
    fem_data: qd.template(),
    linear_system_data: qd.template(),
):
    classify_rigid_contact(route, proxy, forest, vertex, contact, linear_system_data, 0)
    derive_extents(linear_system_data)
    zero_rhs(linear_system_data)
    zero_bcoo_triplets(linear_system_data.matrix)
    distribute_rigid_contact(route, proxy, forest, vertex, contact, fem_data, fem_data, linear_system_data, 0)
    sort_reduce_bcoo(linear_system_data.matrix)


@qd.kernel(fastcache=True)
def refresh_proxy_residual(
    forest: qd.template(),
    rigid: qd.template(),
    proxy: qd.template(),
):
    compute_endpoint_fk(forest, rigid, proxy)
    prepare_constraint(proxy, rigid, forest)


@qd.kernel(fastcache=True)
def evaluate_screw_halfplane(
    vertex: qd.template(),
    output: qd.types.ndarray(qd.f64, ndim=1),
):
    result = qd.Vector.zero(qd.f64, 1)
    screw_halfplane_ccd(
        vertex,
        0,
        qd.Vector([0.0, -1.0, 0.0]),
        -0.7,
        0.1,
        0.0,
        50_000,
        result,
    )
    for _ in range(1):
        output[0] = result[0]


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("backend", [gs.gpu])
def test_proxy_constraint_maps():
    mechanism_t_host = np.array([0.2, -0.1, 0.3], dtype=np.float64)
    mechanism_q_host = _quat_exp(np.array([0.2, -0.1, 0.15], dtype=np.float64))
    rotation = np.array([-0.12, 0.08, 0.05], dtype=np.float64)
    proxy_t_host = mechanism_t_host + np.array([0.03, 0.02, -0.03], dtype=np.float64)
    proxy_q_host = _quat_mul(_quat_exp(rotation), mechanism_q_host)

    mechanism_t = qd.ndarray(qd.f64, shape=(3,))
    mechanism_q = qd.ndarray(qd.f64, shape=(4,))
    proxy_t = qd.ndarray(qd.f64, shape=(3,))
    proxy_q = qd.ndarray(qd.f64, shape=(4,))
    output = qd.ndarray(qd.f64, shape=(84,))
    mechanism_t.from_numpy(mechanism_t_host)
    mechanism_q.from_numpy(mechanism_q_host)
    proxy_t.from_numpy(proxy_t_host)
    proxy_q.from_numpy(proxy_q_host)

    evaluate_proxy_maps(mechanism_t, mechanism_q, proxy_t, proxy_q, output)
    actual = qd_to_numpy(output)

    translation = proxy_t_host - mechanism_t_host
    left = _left_jacobian(rotation)
    expected_constraint = np.concatenate((translation, rotation))
    expected_particular = np.concatenate((-translation, -(left @ rotation)))
    expected_tangent = np.eye(6)
    expected_tangent[3:, 3:] = left @ _right_jacobian_inverse(rotation)
    expected_normal = np.eye(6)
    expected_normal[3:, 3:] = left
    np.testing.assert_allclose(actual[:6], expected_constraint, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(actual[6:12], expected_particular, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(actual[12:48].reshape(6, 6), expected_tangent, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(actual[48:84].reshape(6, 6), expected_normal, rtol=1.0e-12, atol=1.0e-12)


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("backend", [gs.gpu])
def test_fk_defect_prefix_cap():
    cases = np.array(
        [
            [0.2, 0.1, 0.0, 0.2],
            [0.2, 0.5, 0.0, 0.35],
            [0.2, 0.1, 3.0, 0.2],
            [0.2, 0.5, 3.0, 0.2],
            [0.0, 0.0, 3.0, 0.4],
        ],
        dtype=np.float64,
    )
    inputs = qd.ndarray(qd.f64, shape=cases.shape)
    outputs = qd.ndarray(qd.f64, shape=(len(cases),))
    inputs.from_numpy(cases)

    evaluate_fk_prefix_caps(inputs, outputs)
    caps = qd_to_numpy(outputs)
    for cap, (h0, h_slack, curvature, limit) in zip(caps, cases, strict=True):
        alpha = np.linspace(0.0, cap, 129)
        bound = h0 + (h_slack - h0) * alpha + 0.5 * curvature * alpha * alpha
        assert np.all(bound <= limit + 1.0e-12)


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("backend", [gs.gpu])
def test_screw_halfplane_ccd_certifies_rotational_arc():
    vertex = create_global_vertex_data(1)
    vertex.positions.from_numpy(np.array([[1.0, 0.0, 0.0]], dtype=np.float64))
    vertex.trajectory_end_positions.from_numpy(np.array([[0.0, 1.0, 0.0]], dtype=np.float64))
    vertex.path_rot.from_numpy(np.array([[0.0, 0.0, 0.5 * np.pi]], dtype=np.float64))
    vertex.path_pivot.from_numpy(np.zeros((1, 3), dtype=np.float64))
    vertex.path_pivot_disp.from_numpy(np.zeros((1, 3), dtype=np.float64))
    output = qd.ndarray(qd.f64, shape=(1,))

    evaluate_screw_halfplane(vertex, output)

    time = float(qd_to_numpy(output)[0])
    exact = np.arcsin(0.7) / (0.5 * np.pi)
    assert 0.0 < time <= exact + 1.0e-12


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("backend", [gs.gpu])
@pytest.mark.parametrize(
    "forest_path",
    ("genesis_legacy", "level", "tree"),
)
def test_proxy_system_initializes_on_genesis_inertial_pose(forest_path):
    scene = gs.Scene()
    scene.add_entity(
        morph=gs.morphs.Sphere(pos=(0.1, -0.2, 0.4), radius=0.1),
        vis_mode="collision",
    )
    scene.build()
    geometry = RigidContactProxyGeometry()
    assert geometry.init(scene, 0.01)
    rigid = create_rigid_system_data(scene.rigid_solver)
    rigid_storage_dofs = int(qd_to_numpy(rigid.n_storage_dofs))
    total_dof = rigid_storage_dofs + 6
    proxy = create_rigid_contact_proxy_data(
        n_links_host=scene.rigid_solver.n_links,
        n_instances_host=scene.rigid_solver._B,
        n_rigid_bodies=2,
        mechanism_body=np.array([0], dtype=np.int32),
        proxy_body=np.array([1], dtype=np.int32),
        surface_radius=np.array([0.1], dtype=np.float64),
        geometry=geometry,
        global_vert_offset=0,
        global_body_offset=0,
        merit_gradient_capacity=total_dof,
    )
    forest = create_rigid_joint_forest_data(
        scene.rigid_solver,
        rigid,
        total_dof=total_dof,
        n_rigid_bodies=2,
        proxy_dof_offset=rigid_storage_dofs,
        proxy_data=proxy,
        fused_enabled=forest_path == "tree",
        genesis_legacy_enabled=forest_path == "genesis_legacy",
    )
    vertex = create_global_vertex_data(len(geometry.local_positions))

    initialize_and_prepare_proxy(proxy, rigid, forest, vertex)

    np.testing.assert_allclose(
        qd_to_numpy(proxy.t)[0],
        np.array([0.1, -0.2, 0.4]),
        atol=1.0e-14,
    )
    expected_constraint = np.zeros(6, dtype=np.float64)
    expected_constraint[:3] = qd_to_numpy(proxy.t)[0] - qd_to_numpy(forest.endpoint_t)[0]
    np.testing.assert_allclose(qd_to_numpy(proxy.constraint)[0], expected_constraint, atol=1.0e-14)
    np.testing.assert_allclose(qd_to_numpy(proxy.particular)[0], -expected_constraint, atol=1.0e-14)
    np.testing.assert_allclose(qd_to_numpy(proxy.tangent_map)[0], np.eye(6), atol=1.0e-14)
    np.testing.assert_allclose(qd_to_numpy(proxy.normal_map)[0], np.eye(6), atol=1.0e-14)
    assert np.linalg.eigvalsh(qd_to_numpy(proxy.metric)[0]).min() > 0.0
    assert np.isfinite(qd_to_numpy(vertex.positions)).all()

    direction = np.zeros((1, 6), dtype=np.float64)
    direction[0, 0] = 0.02
    direction[0, 5] = 0.1
    proxy.dq.from_numpy(direction)
    publish_proxy_trajectory(proxy, rigid, forest, vertex)
    assert np.linalg.norm(qd_to_numpy(vertex.trajectory_end_positions) - qd_to_numpy(vertex.positions)) > 0.0
    assert qd_to_numpy(vertex.path_inflation).max() > 0.0
    np.testing.assert_array_equal(qd_to_numpy(vertex.path_kind), 0)

    genesis_legacy_enabled = forest_path == "genesis_legacy"
    use_fused_tree_path = forest_path == "tree"
    reduced_direction = qd.ndarray(qd.f64, shape=(total_dof,))
    proxy_wrench = qd.ndarray(qd.f64, shape=(6,))
    reduced_wrench = qd.ndarray(qd.f64, shape=(total_dof,))
    direction_host = np.linspace(-0.3, 0.4, total_dof, dtype=np.float64)
    wrench_host = np.array([0.7, -0.2, 0.5, -0.1, 0.4, 0.3], dtype=np.float64)
    reduced_direction.from_numpy(direction_host)
    proxy_wrench.from_numpy(wrench_host)
    evaluate_forest_virtual_work(
        forest,
        rigid,
        proxy,
        reduced_direction,
        proxy_wrench,
        reduced_wrench,
        genesis_legacy_enabled,
        use_fused_tree_path,
        scene.rigid_solver.n_links,
    )
    mechanism_twist = qd_to_numpy(forest.body_twist)[0]
    proxy_twist = qd_to_numpy(forest.body_twist)[1]
    np.testing.assert_allclose(
        mechanism_twist,
        direction_host[:6],
        rtol=1.0e-12,
        atol=1.0e-13,
    )
    np.testing.assert_allclose(
        direction_host @ qd_to_numpy(reduced_wrench),
        proxy_twist @ wrench_host,
        rtol=1.0e-11,
        atol=1.0e-12,
    )

    linear_system_system = GlobalLinearSystem()
    linear_system_system.wire_data(
        n_block_rows=4,
        n_elastic_triplets=3,
        max_contact_body_triplets=0,
        dof_block_base=2,
    )
    linear_system_system.init()
    linear_system_data = linear_system_system.data
    linear_system_data.matrix.bcoo_nnz.from_numpy(np.array(3, dtype=np.int32))
    linear_system_data.matrix.bcoo_row.from_numpy(np.array([2, 2, 3], dtype=np.int32))
    linear_system_data.matrix.bcoo_col.from_numpy(np.array([2, 3, 3], dtype=np.int32))
    blocks = np.array(
        [
            [[4.0, 0.2, 0.0], [0.2, 5.0, 0.1], [0.0, 0.1, 6.0]],
            [[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]],
            [[3.0, 0.0, 0.0], [0.0, 4.0, 0.2], [0.0, 0.2, 5.0]],
        ],
        dtype=np.float64,
    )
    linear_system_data.matrix.bcoo_val.from_numpy(
        np.pad(blocks.reshape(-1), (0, linear_system_data.matrix.bcoo_val.shape[0] - blocks.size))
    )
    second = qd.ndarray(qd.f64, shape=(total_dof,))
    first_result = qd.ndarray(qd.f64, shape=(total_dof,))
    second_result = qd.ndarray(qd.f64, shape=(total_dof,))
    second_host = np.linspace(0.5, -0.2, total_dof, dtype=np.float64)
    second.from_numpy(second_host)
    apply_reduced_contact_operator(
        forest,
        rigid,
        proxy,
        linear_system_data,
        reduced_direction,
        first_result,
        genesis_legacy_enabled,
        use_fused_tree_path,
        scene.rigid_solver.n_links,
    )
    apply_reduced_contact_operator(
        forest,
        rigid,
        proxy,
        linear_system_data,
        second,
        second_result,
        genesis_legacy_enabled,
        use_fused_tree_path,
        scene.rigid_solver.n_links,
    )
    first_result_host = qd_to_numpy(first_result)
    second_result_host = qd_to_numpy(second_result)
    np.testing.assert_allclose(
        direction_host @ second_result_host,
        second_host @ first_result_host,
        rtol=1.0e-11,
        atol=1.0e-12,
    )
    assert direction_host @ first_result_host > 0.0

    perturbed = qd_to_numpy(proxy.t).copy()
    perturbed[0] += np.array([0.01, -0.005, 0.002], dtype=np.float64)
    proxy.t.from_numpy(perturbed)
    prepare_proxy_constraint(proxy, rigid, forest)
    assert np.linalg.norm(qd_to_numpy(proxy.particular)[0]) > 0.0
    project_particular_rhs(
        forest,
        rigid,
        proxy,
        linear_system_data,
        genesis_legacy_enabled,
        use_fused_tree_path,
        scene.rigid_solver.n_links,
    )
    reduced_rhs = qd_to_numpy(linear_system_data.b_rhs)
    h_particular = qd_to_numpy(forest.physical_Ap).copy()
    prepare_physical_direction(
        forest,
        rigid,
        proxy,
        reduced_direction,
        genesis_legacy_enabled,
        use_fused_tree_path,
        scene.rigid_solver.n_links,
    )
    physical_direction = qd_to_numpy(forest.physical_p)
    np.testing.assert_allclose(
        direction_host @ reduced_rhs,
        physical_direction @ h_particular,
        rtol=1.0e-11,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        reduced_rhs[rigid_storage_dofs:],
        0.0,
        atol=1.0e-14,
    )


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("backend", [gs.gpu])
def test_proxy_contact_routes_emit_block_counts():
    contact = ProxyContactFixture()
    contact.n_unique_doublets.from_numpy(np.array(3, dtype=np.int32))
    contact.n_unique_triplets.from_numpy(np.array(3, dtype=np.int32))
    contact.unique_doublet_vertices.from_numpy(np.array([0, 1, 2], dtype=np.int32))
    contact.unique_doublet_gradients.from_numpy(
        np.array(
            [
                [0.15, -0.25, 0.35],
                [0.3, -0.2, 0.4],
                [-0.1, 0.5, 0.2],
            ],
            dtype=np.float64,
        )
    )
    contact.unique_triplet_rows.from_numpy(np.array([0, 0, 1], dtype=np.int32))
    contact.unique_triplet_cols.from_numpy(np.array([0, 1, 2], dtype=np.int32))
    contact.unique_triplet_values.from_numpy(
        np.array(
            [
                np.diag([2.0, 3.0, 4.0]),
                [[0.2, 0.0, 0.0], [0.0, 0.3, 0.0], [0.0, 0.0, 0.4]],
                [[1.0, 0.1, 0.0], [0.1, 1.2, 0.2], [0.0, 0.2, 1.4]],
            ],
            dtype=np.float64,
        )
    )

    fem = ProxyFEMFixture()
    fem.dof_offset.from_numpy(np.array(0, dtype=np.int32))
    fem.is_fixed.from_numpy(np.array([0], dtype=np.int32))
    forest = ProxyForestFixture()
    forest.proxy_dof_offset.from_numpy(np.array(3, dtype=np.int32))

    geometry = RigidContactProxyGeometry()
    geometry.local_positions = np.array(
        [[0.5, 0.0, 0.0], [0.0, 0.5, 0.0]],
        dtype=np.float64,
    )
    geometry.vertex_pair = np.array([0, 0], dtype=np.int32)
    proxy = create_rigid_contact_proxy_data(
        n_links_host=1,
        n_instances_host=1,
        n_rigid_bodies=2,
        mechanism_body=np.array([0], dtype=np.int32),
        proxy_body=np.array([1], dtype=np.int32),
        surface_radius=np.array([1.0], dtype=np.float64),
        geometry=geometry,
        global_vert_offset=1,
        global_body_offset=0,
        merit_gradient_capacity=9,
    )
    proxy.t.from_numpy(np.zeros((1, 3), dtype=np.float64))

    vertex_system = GlobalVertexManager()
    vertex_system.wire_data(3)
    vertex_system.init()
    vertex = vertex_system.data
    vertex.positions.from_numpy(
        np.array(
            [[-0.2, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0]],
            dtype=np.float64,
        )
    )

    linear_system_system = GlobalLinearSystem()
    linear_system_system.wire_data(
        n_block_rows=3,
        n_elastic_triplets=0,
        max_contact_body_triplets=9,
        dof_block_base=0,
    )
    linear_system_system.init()
    linear_system_data = linear_system_system.data
    linear_system_data.n_extent_slots.from_numpy(np.array(1, dtype=np.int32))
    route = create_rigid_contact_assemble_data(contact)

    assemble_proxy_contact(route, proxy, forest, vertex, contact, fem, linear_system_data)

    np.testing.assert_array_equal(qd_to_numpy(linear_system_data.matrix.n_triplets), 9)
    assert int(qd_to_numpy(linear_system_data.matrix.bcoo_valid)) == 1
    assert int(qd_to_numpy(linear_system_data.matrix.bcoo_nnz)) <= 9
    rhs = qd_to_numpy(linear_system_data.b_rhs)
    np.testing.assert_allclose(rhs[:3], [0.15, -0.25, 0.35], atol=1.0e-12)
    gradient = np.array([0.2, 0.3, 0.6])
    angular = np.cross(np.array([0.5, 0.0, 0.0]), np.array([0.3, -0.2, 0.4]))
    angular += np.cross(np.array([0.0, 0.5, 0.0]), np.array([-0.1, 0.5, 0.2]))
    np.testing.assert_allclose(rhs[3:6], gradient, atol=1.0e-12)
    np.testing.assert_allclose(rhs[6:9], angular, atol=1.0e-12)


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("backend", [gs.gpu])
def test_standard_pcg_kkt_builder_path():
    scene = gs.Scene(
        engine_options=gs.options.NewtonEngineOptions(),
    )
    scene.add_entity(
        morph=gs.morphs.Sphere(
            pos=(2.0, 0.0, 1.0),
            radius=0.1,
        ),
        vis_mode="collision",
    )
    cloth = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=str(Path(__file__).parents[2] / "examples" / "newton_coupling" / "assets" / "qcloth_grid.obj"),
            pos=(-5.0, 0.0, 0.0),
        ),
        material=gs.materials.FEM.QCloth(E=1.0e4, thickness=1.0e-3),
    )
    scene.build()
    cloth.set_vertex_constraints([0, 4, 20, 24])

    engine = build_scene_engine(scene, contact_config={})
    assert engine.rigid_contact_proxy is not None
    assert engine.rigid_forest is not None
    assert engine.rigid_contact_assemble is not None
    assert engine.rigid_forest.selected_path == "tree"
    engine.step()

    mechanism = int(qd_to_numpy(engine.rigid_contact_proxy.data.mechanism_body)[0])
    link = mechanism % engine.rigid_contact_proxy.n_links
    mechanism_position = (
        scene.rigid_solver.get_links_pos(links_idx=np.array([link], dtype=np.int32)).cpu().numpy().reshape(-1, 3)[0]
    )
    proxy_position = qd_to_numpy(engine.rigid_contact_proxy.data.t)[0]
    residual = float(np.linalg.norm(proxy_position - mechanism_position))
    tolerance = min(
        float(qd_to_numpy(engine.sim_config_system.data.tol)),
        0.01 * float(qd_to_numpy(engine.contact_system.data.d_hat)),
    )
    assert np.isfinite(residual)
    assert residual <= tolerance, (
        residual,
        proxy_position,
        mechanism_position,
        qd_to_numpy(engine.rigid_contact_proxy.data.mechanism_body),
    )


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("backend", [gs.gpu])
def test_kkt_restoration_returns_through_hard_probe():
    scene = gs.Scene(
        engine_options=gs.options.NewtonEngineOptions(),
    )
    scene.add_entity(
        morph=gs.morphs.Sphere(pos=(2.0, 0.0, 1.0), radius=0.1),
        vis_mode="collision",
    )
    cloth = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=str(Path(__file__).parents[2] / "examples" / "newton_coupling" / "assets" / "qcloth_grid.obj"),
            pos=(-5.0, 0.0, 0.0),
        ),
        material=gs.materials.FEM.QCloth(E=1.0e4, thickness=1.0e-3),
    )
    scene.build()
    cloth.set_vertex_constraints([0, 4, 20, 24])
    engine = build_scene_engine(
        scene,
        contact_config={"extras/ls_forensics/test_energy_bias": 1.0},
    )

    proxy = engine.rigid_contact_proxy
    perturbed = qd_to_numpy(proxy.t).copy()
    perturbed[0, 0] += 0.02
    proxy.t.from_numpy(perturbed)
    engine.step()

    assert int(qd_to_numpy(proxy.restoration_entries)) == 1
    assert int(qd_to_numpy(proxy.restoration_newton_epochs)) >= 1
    assert int(qd_to_numpy(proxy.restoration_hard_probes)) >= 1
    assert int(qd_to_numpy(proxy.restoration_active)) == 0
    assert int(qd_to_numpy(proxy.restoration_hard_probe)) == 0
    assert float(qd_to_numpy(proxy.max_surface_residual)) <= float(qd_to_numpy(proxy.solve_tolerance))


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("backend", [gs.gpu])
def test_kkt_newton_exhaustion_does_not_commit_previous_state():
    scene = gs.Scene(
        engine_options=gs.options.NewtonEngineOptions(),
    )
    scene.add_entity(
        morph=gs.morphs.Sphere(pos=(2.0, 0.0, 1.0), radius=0.1),
        vis_mode="collision",
    )
    cloth = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=str(Path(__file__).parents[2] / "examples" / "newton_coupling" / "assets" / "qcloth_grid.obj"),
            pos=(-5.0, 0.0, 0.0),
        ),
        material=gs.materials.FEM.QCloth(E=1.0e4, thickness=1.0e-3),
    )
    scene.build()
    cloth.set_vertex_constraints([0, 4, 20, 24])
    engine = build_scene_engine(scene, contact_config={})
    engine.sim_config_system.data.max_newton_iter.from_numpy(np.array(1, dtype=np.int64))

    proxy = engine.rigid_contact_proxy
    perturbed = qd_to_numpy(proxy.t).copy()
    perturbed[0, 0] += 0.02
    proxy.t.from_numpy(perturbed)
    proxy_t_prev = qd_to_numpy(proxy.t_prev).copy()
    proxy_quat_prev = qd_to_numpy(proxy.quat_prev).copy()
    rigid_velocity = qd_to_numpy(scene.rigid_solver.dyn_state.dofs.vel).copy()

    with pytest.raises(RuntimeError, match="exhausted the Newton budget"):
        engine.step()

    np.testing.assert_array_equal(qd_to_numpy(proxy.t_prev), proxy_t_prev)
    np.testing.assert_array_equal(qd_to_numpy(proxy.quat_prev), proxy_quat_prev)
    np.testing.assert_array_equal(
        qd_to_numpy(scene.rigid_solver.dyn_state.dofs.vel),
        rigid_velocity,
    )


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("backend", [gs.gpu])
def test_rigid_proxy_cloth_contact_step():
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01),
        engine_options=gs.options.NewtonEngineOptions(),
    )
    sphere = scene.add_entity(
        morph=gs.morphs.Sphere(pos=(0.0, 0.0, 0.15), radius=0.1),
        vis_mode="collision",
    )
    cloth = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=str(Path(__file__).parents[2] / "examples" / "newton_coupling" / "assets" / "qcloth_grid.obj"),
        ),
        material=gs.materials.FEM.QCloth(E=1.0e4, thickness=1.0e-3),
    )
    scene.build()
    cloth.set_vertex_constraints(np.arange(25, dtype=np.int32))
    sphere.set_dofs_velocity([0.0, 0.0, -5.0, 0.0, 0.0, 0.0])

    engine = build_scene_engine(scene, contact_config={})
    old_doublet_capacity = engine.contact_system.data.unique_doublet_vertices.shape[0]
    old_triplet_capacity = engine.contact_system.data.unique_triplet_rows.shape[0]
    engine.contact_system.realloc_assembly_buffers(
        old_doublet_capacity + 1,
        old_triplet_capacity + 1,
    )
    engine.rigid_contact_assemble.realloc_assembly_buffers(engine.contact_system.data)
    assert engine.rigid_contact_assemble.data.doublet_scanner.status.shape[0] == max(
        (engine.contact_system.data.unique_doublet_vertices.shape[0] + 3071) // 3072,
        1,
    )
    assert engine.rigid_contact_assemble.data.triplet_scanner.status.shape[0] == max(
        (engine.contact_system.data.unique_triplet_rows.shape[0] + 3071) // 3072,
        1,
    )
    engine.step()

    assert int(qd_to_numpy(engine.contact_system.data.intersection_flag)) == 0
    assert float(qd_to_numpy(engine.contact_system.data.ccd_alpha)) > 0.0
    assert int(qd_to_numpy(engine.rigid_contact_assemble.data.rigid_doublet_total)) > 0
    assert np.linalg.norm(qd_to_numpy(engine.rigid_contact_proxy.data.reaction)) > 0.0
    assert float(qd_to_numpy(engine.rigid_contact_proxy.data.max_surface_residual)) <= float(
        qd_to_numpy(engine.rigid_contact_proxy.data.solve_tolerance)
    )


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("backend", [gs.gpu])
def test_cloth_drapes_on_fixed_proxy_box():
    cube_top = 0.08
    cloth_resolution = 25
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01),
        engine_options=gs.options.NewtonEngineOptions(),
    )
    scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.0, 0.0, 0.04),
            size=(0.08, 0.08, 0.08),
            fixed=True,
        ),
        vis_mode="collision",
    )
    cloth = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=str(cloth_grid_asset(resolution=cloth_resolution)),
            pos=(0.0, 0.0, 0.14),
        ),
        material=gs.materials.FEM.QCloth(
            E=2e4,
            shear_modulus=2e3,
            rho=200.0,
            thickness=1e-3,
            bending_youngs_modulus=3e3,
        ),
    )
    scene.build()
    engine = build_scene_engine(
        scene,
        contact_config={
            "contact/d_hat": 1e-3,
            "contact/intersection_check": 1,
        },
    )

    maximum_proxy_doublets = 0
    for _ in range(120):
        engine.step()
        maximum_proxy_doublets = max(
            maximum_proxy_doublets,
            int(qd_to_numpy(engine.rigid_contact_assemble.data.rigid_doublet_total)),
        )

    center_vertex = cloth_resolution * cloth_resolution // 2
    center_height = float(qd_to_numpy(engine.fem_system.data.x)[center_vertex, 2])
    assert maximum_proxy_doublets > 0
    assert center_height > cube_top


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("backend", [gs.gpu])
def test_newton_engine_cloth_drapes_on_fixed_proxy_box():
    cube_top = 0.08
    cloth_resolution = 25
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01),
        engine_options=gs.options.NewtonEngineOptions(
            contact_d_hat=1e-3,
            contact_friction_mu=0.05,
            contact_resistance=1e4,
        ),
    )
    scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.0, 0.0, 0.04),
            size=(0.08, 0.08, 0.08),
            fixed=True,
        ),
        vis_mode="collision",
    )
    cloth = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=str(cloth_grid_asset(resolution=cloth_resolution)),
            pos=(0.0, 0.0, 0.14),
        ),
        material=gs.materials.FEM.QCloth(
            E=2e4,
            shear_modulus=2e3,
            rho=200.0,
            thickness=1e-3,
            bending_youngs_modulus=3e3,
        ),
    )
    scene.build()

    maximum_proxy_doublets = 0
    for _ in range(120):
        scene.step()
        maximum_proxy_doublets = max(
            maximum_proxy_doublets,
            int(qd_to_numpy(scene.sim.engine.rigid_contact_assemble.data.rigid_doublet_total)),
        )

    center_vertex = cloth_resolution * cloth_resolution // 2
    center_height = float(cloth.get_state().pos.reshape(-1, 3)[center_vertex, 2])
    assert maximum_proxy_doublets > 0
    assert center_height > cube_top


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("backend", [gs.gpu])
def test_franka_forest_paths_match_P_and_PT():
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01),
        coupler_options=gs.options.LegacyCouplerOptions(rigid_fem=False),
    )
    scene.add_entity(
        morph=gs.morphs.MJCF(
            file="xml/franka_emika_panda/panda.xml",
        ),
        vis_mode="collision",
    )
    scene.add_entity(
        morph=gs.morphs.Box(
            pos=(2.0, 0.0, 1.0),
            size=(0.1, 0.1, 0.1),
        ),
        vis_mode="collision",
    )
    scene.build()

    rigid = create_rigid_system_data(scene.rigid_solver)
    total_dof = int(qd_to_numpy(rigid.n_storage_dofs))
    n_bodies = scene.rigid_solver.n_links * scene.rigid_solver._B
    empty_geometry = RigidContactProxyGeometry()
    empty_geometry.local_positions = np.empty((0, 3), dtype=np.float64)
    empty_geometry.vertex_pair = np.empty(0, dtype=np.int32)
    proxy = create_rigid_contact_proxy_data(
        n_links_host=scene.rigid_solver.n_links,
        n_instances_host=scene.rigid_solver._B,
        n_rigid_bodies=n_bodies,
        mechanism_body=np.empty(0, dtype=np.int32),
        proxy_body=np.empty(0, dtype=np.int32),
        surface_radius=np.empty(0, dtype=np.float64),
        geometry=empty_geometry,
        global_vert_offset=0,
        global_body_offset=0,
        merit_gradient_capacity=total_dof,
    )
    direction_host = np.linspace(-0.4, 0.3, total_dof, dtype=np.float64)
    body_wrench_host = np.linspace(
        -0.7,
        0.8,
        n_bodies * 6,
        dtype=np.float64,
    ).reshape(n_bodies, 6)
    direction = qd.ndarray(qd.f64, shape=(total_dof,))
    body_wrench = qd.ndarray(qd.f64, shape=(n_bodies, 6))
    residual = qd.ndarray(qd.f64, shape=(total_dof,))
    direction.from_numpy(direction_host)
    body_wrench.from_numpy(body_wrench_host)
    residual_host = np.linspace(-2e-7, 3e-7, total_dof, dtype=np.float64)
    residual.from_numpy(residual_host)

    outputs = {}
    for path in ("genesis_legacy", "level", "tree"):
        forest = create_rigid_joint_forest_data(
            scene.rigid_solver,
            rigid,
            total_dof=total_dof,
            n_rigid_bodies=n_bodies,
            proxy_dof_offset=total_dof,
            proxy_data=proxy,
            fused_enabled=path == "tree",
            genesis_legacy_enabled=path == "genesis_legacy",
        )
        edge_capacity = forest.edge_d.shape[0]
        forest.edge_basis.from_numpy(
            np.linspace(
                -0.3,
                0.4,
                edge_capacity * 6,
                dtype=np.float64,
            ).reshape(edge_capacity, 6)
        )
        forest.edge_u.from_numpy(
            np.linspace(
                0.05,
                -0.04,
                edge_capacity * 6,
                dtype=np.float64,
            ).reshape(edge_capacity, 6)
        )
        forest.edge_d.from_numpy(np.linspace(1.5, 2.5, edge_capacity, dtype=np.float64))
        forest.root_inverse.from_numpy(
            np.repeat(
                np.eye(6, dtype=np.float64)[None, :, :],
                n_bodies,
                axis=0,
            )
        )
        reduced_wrench = qd.ndarray(qd.f64, shape=(total_dof,))
        preconditioned = qd.ndarray(qd.f64, shape=(total_dof,))
        preconditioned.from_numpy(np.full(total_dof, np.nan, dtype=np.float64))
        evaluate_forest_transform(
            forest,
            rigid,
            proxy,
            direction,
            body_wrench,
            reduced_wrench,
            path == "genesis_legacy",
            path == "tree",
            scene.rigid_solver.n_links,
        )
        if path == "tree":
            apply_forest_preconditioner_tree(
                forest,
                rigid,
                proxy,
                residual,
                preconditioned,
            )
        else:
            apply_forest_preconditioner_level(
                forest,
                rigid,
                proxy,
                residual,
                preconditioned,
                scene.rigid_solver.n_links,
            )
        outputs[path] = (
            qd_to_numpy(forest.body_twist).copy(),
            qd_to_numpy(reduced_wrench).copy(),
            qd_to_numpy(preconditioned).copy(),
        )
        np.testing.assert_allclose(
            direction_host @ outputs[path][1],
            np.sum(outputs[path][0][:n_bodies] * body_wrench_host),
            rtol=1.0e-11,
            atol=1.0e-12,
        )

    for path in ("level", "tree"):
        np.testing.assert_allclose(
            outputs[path][0],
            outputs["genesis_legacy"][0],
            rtol=1.0e-12,
            atol=1.0e-13,
        )
        np.testing.assert_allclose(
            outputs[path][1],
            outputs["genesis_legacy"][1],
            rtol=1.0e-12,
            atol=1.0e-13,
        )
        np.testing.assert_allclose(
            outputs[path][2],
            outputs["genesis_legacy"][2],
            rtol=1.0e-11,
            atol=1.0e-12,
        )


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("backend", [gs.gpu])
def test_franka_cloth_reduced_kkt_step():
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01),
        engine_options=gs.options.NewtonEngineOptions(),
    )
    franka = scene.add_entity(
        morph=gs.morphs.MJCF(
            file="xml/franka_emika_panda/panda.xml",
        ),
        vis_mode="collision",
    )
    cloth = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=str(Path(__file__).parents[2] / "examples" / "newton_coupling" / "assets" / "qcloth_grid.obj"),
            pos=(0.9, 0.0, 0.7),
        ),
        material=gs.materials.FEM.QCloth(
            E=2e4,
            shear_modulus=2e3,
            rho=200.0,
            thickness=1e-3,
            bending_youngs_modulus=1e6,
        ),
    )
    scene.build()
    cloth.set_vertex_constraints([20, 24])
    contact_tabular = ContactTabular()
    contact_tabular.default_model(friction_rate=1.0, resistance=1e4)
    engine = build_scene_engine(
        scene,
        contact_config={
            "contact/d_hat": 1e-3,
            "contact/init_collision_pair_capacity": 20_000,
        },
        contact_tabular=contact_tabular,
    )
    assert engine.rigid_forest.selected_path == "tree"
    assert not engine.rigid.has_collision

    home_qpos = np.array(
        [0.0, 0.0, 0.0, -1.5708, 0.0, 1.5708, -0.7854, 0.04, 0.04],
        dtype=np.float64,
    )
    franka.set_qpos(home_qpos)
    franka.set_dofs_kp(
        np.array([4500.0, 4500.0, 3500.0, 3500.0, 2000.0, 2000.0, 2000.0, 100.0, 100.0]),
    )
    franka.set_dofs_kv(
        np.array([450.0, 450.0, 350.0, 350.0, 200.0, 200.0, 200.0, 10.0, 10.0]),
    )
    franka.control_dofs_position(home_qpos)
    newton_iterations = []
    for _ in range(3):
        engine.step()
        newton_iterations.append(engine.get_newton_iters())

    edge_count = int(qd_to_numpy(engine.rigid_forest.data.n_edges))
    assert edge_count == 9
    assert int(qd_to_numpy(engine.rigid.data.constraint_state.n_constraints)[0]) > 0
    assert max(newton_iterations) <= 4
    assert engine.get_total_pcg_iters() >= engine.get_max_pcg_iters()
    assert int(qd_to_numpy(engine.frame_failed)) == 0
    assert float(qd_to_numpy(engine.rigid_contact_proxy.data.max_surface_residual)) <= float(
        qd_to_numpy(engine.rigid_contact_proxy.data.solve_tolerance)
    )

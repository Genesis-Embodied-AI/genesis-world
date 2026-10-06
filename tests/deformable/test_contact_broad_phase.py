import numpy as np
import pytest
import quadrants as qd

import genesis as gs
from genesis.engine.systems.bvh_math import f64_to_f32_rd, f64_to_f32_ru
from genesis.engine.systems.dual_ee_query import DualEEQueryState
from genesis.engine.systems.global_body_manager import GlobalBodyManager
from genesis.engine.systems.global_surface_manager import GlobalSurfaceManager
from genesis.engine.systems.global_vertex_manager import GlobalVertexManager
from genesis.engine.systems.lbvh import LBVH
from genesis.utils.misc import qd_to_numpy


@qd.data_oriented
class ContactPredicateFixture:
    def __init__(self, n_vertices: int, enabled: bool = True):
        flag = int(enabled)
        self.n_contact_elements = qd.ndarray(qd.i32, shape=())
        self.enable_table = qd.ndarray(qd.i32, shape=(1,))
        self.enable_ee_table = qd.ndarray(qd.i32, shape=(1,))
        self.vert_contact_element_ids = qd.ndarray(
            qd.i32,
            shape=(max(n_vertices, 1),),
        )
        self.n_contact_elements.from_numpy(np.array(1, dtype=np.int32))
        self.enable_table.from_numpy(np.array([flag], dtype=np.int32))
        self.enable_ee_table.from_numpy(np.array([flag], dtype=np.int32))
        self.vert_contact_element_ids.from_numpy(np.zeros(max(n_vertices, 1), dtype=np.int32))


@qd.kernel
def build_and_query_pt(
    bvh: qd.template(),
    surface: qd.template(),
    vertex: qd.template(),
    body: qd.template(),
    contact: qd.template(),
    pairs: qd.types.ndarray(qd.i32, ndim=2),
    n_pairs: qd.types.ndarray(qd.i32, ndim=0),
    overflow: qd.types.ndarray(qd.i32, ndim=0),
):
    bvh.calc_leaf_aabb_tri(surface, vertex)
    bvh.reduce_scene_aabb()
    bvh.calc_morton()
    bvh.sort_morton()
    bvh.extract_indices()
    bvh.copy_leaf_aabb_to_temp()
    bvh.reorder_leaf_aabb()
    bvh.calc_leaf_nodes()
    bvh.calc_internal_nodes()
    bvh.memset_flags()
    bvh.calc_internal_aabb()
    for _ in range(1):
        n_pairs[()] = 0
        overflow[()] = 0
    bvh.query_pt_warp(
        surface,
        vertex,
        body,
        contact,
        pairs,
        n_pairs,
        16,
        0.01,
        overflow,
    )


@qd.kernel
def build_triangle_bvh(
    bvh: qd.template(),
    surface: qd.template(),
    vertex: qd.template(),
):
    bvh.calc_leaf_aabb_tri(surface, vertex)
    bvh.reduce_scene_aabb()
    bvh.calc_morton()
    bvh.sort_morton()
    bvh.extract_indices()
    bvh.copy_leaf_aabb_to_temp()
    bvh.reorder_leaf_aabb()
    bvh.calc_leaf_nodes()
    bvh.calc_internal_nodes()
    bvh.memset_flags()
    bvh.calc_internal_aabb()


@qd.kernel
def query_pt_batched_only(
    bvh: qd.template(),
    surface: qd.template(),
    vertex: qd.template(),
    body: qd.template(),
    contact: qd.template(),
    pairs: qd.types.ndarray(qd.i32, ndim=2),
    n_pairs: qd.types.ndarray(qd.i32, ndim=0),
    max_pairs: qd.i32,
    overflow: qd.types.ndarray(qd.i32, ndim=0),
):
    for _ in range(1):
        n_pairs[()] = 0
        overflow[()] = 0
    bvh.query_pt_batched(
        surface,
        vertex,
        body,
        contact,
        pairs,
        n_pairs,
        max_pairs,
        0.01,
        overflow,
    )


@qd.kernel
def query_pt_warp_only(
    bvh: qd.template(),
    surface: qd.template(),
    vertex: qd.template(),
    body: qd.template(),
    contact: qd.template(),
    pairs: qd.types.ndarray(qd.i32, ndim=2),
    n_pairs: qd.types.ndarray(qd.i32, ndim=0),
    max_pairs: qd.i32,
    overflow: qd.types.ndarray(qd.i32, ndim=0),
):
    for _ in range(1):
        n_pairs[()] = 0
        overflow[()] = 0
    bvh.query_pt_warp(
        surface,
        vertex,
        body,
        contact,
        pairs,
        n_pairs,
        max_pairs,
        0.01,
        overflow,
    )


@qd.kernel
def build_and_query_ee_dual(
    bvh: qd.template(),
    dual: qd.template(),
    surface: qd.template(),
    vertex: qd.template(),
    body: qd.template(),
    contact: qd.template(),
    pairs: qd.types.ndarray(qd.i32, ndim=2),
    n_pairs: qd.types.ndarray(qd.i32, ndim=0),
    overflow: qd.types.ndarray(qd.i32, ndim=0),
):
    bvh.calc_leaf_aabb_edge(surface, vertex)
    bvh.reduce_scene_aabb()
    bvh.calc_morton()
    bvh.sort_morton()
    bvh.extract_indices()
    bvh.copy_leaf_aabb_to_temp()
    bvh.reorder_leaf_aabb()
    bvh.calc_leaf_nodes()
    bvh.calc_internal_nodes()
    bvh.memset_flags()
    bvh.calc_internal_aabb()
    for _ in range(1):
        n_pairs[()] = 0
        overflow[()] = 0
    dual.query(
        bvh,
        surface,
        vertex,
        body,
        contact,
        pairs,
        n_pairs,
        16,
        overflow,
    )


@qd.kernel
def query_ee_warp(
    bvh: qd.template(),
    surface: qd.template(),
    vertex: qd.template(),
    body: qd.template(),
    contact: qd.template(),
    pairs: qd.types.ndarray(qd.i32, ndim=2),
    n_pairs: qd.types.ndarray(qd.i32, ndim=0),
    overflow: qd.types.ndarray(qd.i32, ndim=0),
):
    for _ in range(1):
        n_pairs[()] = 0
        overflow[()] = 0
    bvh.query_ee_warp(
        surface,
        vertex,
        body,
        contact,
        pairs,
        n_pairs,
        64,
        0.0,
        overflow,
    )


@qd.kernel
def query_ee_dual_only(
    bvh: qd.template(),
    dual: qd.template(),
    surface: qd.template(),
    vertex: qd.template(),
    body: qd.template(),
    contact: qd.template(),
    pairs: qd.types.ndarray(qd.i32, ndim=2),
    n_pairs: qd.types.ndarray(qd.i32, ndim=0),
    overflow: qd.types.ndarray(qd.i32, ndim=0),
):
    for _ in range(1):
        overflow[()] = 0
    dual.query(
        bvh,
        surface,
        vertex,
        body,
        contact,
        pairs,
        n_pairs,
        64,
        overflow,
    )


@qd.kernel
def sort_morton_only(bvh: qd.template()):
    bvh.sort_morton()


@qd.kernel
def convert_f64_to_f32_outward(
    values: qd.types.ndarray(qd.f64, ndim=1),
    lower: qd.types.ndarray(qd.f32, ndim=1),
    upper: qd.types.ndarray(qd.f32, ndim=1),
):
    for index in range(values.shape[0]):
        lower[index] = f64_to_f32_rd(values[index])
        upper[index] = f64_to_f32_ru(values[index])


@qd.kernel
def calc_triangle_leaves(
    bvh: qd.template(),
    surface: qd.template(),
    vertex: qd.template(),
):
    bvh.calc_leaf_aabb_tri(surface, vertex)


def make_single_body_pt_scene(
    positions: np.ndarray,
    triangles: np.ndarray,
):
    n_vertices = positions.shape[0]
    vertex = GlobalVertexManager()
    vertex.init(n_vertices)
    vertex.positions.from_numpy(positions)
    vertex.safe_positions.from_numpy(positions)
    vertex.trajectory_end_positions.from_numpy(positions)
    vertex.x_bar.from_numpy(positions)
    vertex.body_id.from_numpy(np.zeros(n_vertices, dtype=np.int32))
    vertex.wire_thickness_data(np.full(n_vertices, 0.001, dtype=np.float64))
    vertex.wire_d_hat_data(np.full(n_vertices, 0.01, dtype=np.float64))
    vertex.wire_is_fixed_data(np.zeros(n_vertices, dtype=np.int32))

    surface = GlobalSurfaceManager()
    surface.wire_surface_data(
        triangles,
        np.empty((0, 2), dtype=np.int32),
        np.arange(n_vertices, dtype=np.int32),
    )
    surface.wire_vert_dimensions(np.full(n_vertices, 2, dtype=np.int32))
    surface.wire_area_weights(
        np.full(n_vertices, 1.0 / n_vertices, dtype=np.float64),
        np.empty(0, dtype=np.float64),
        np.full(
            triangles.shape[0],
            1.0 / triangles.shape[0],
            dtype=np.float64,
        ),
    )

    body = GlobalBodyManager()
    body.init(1)
    body.vertex_offsets.from_numpy(np.array([0, n_vertices], dtype=np.int32))
    body.self_collision.from_numpy(np.ones(1, dtype=np.int32))
    body.wire_body_contact_ignorance(
        np.array([0, 0], dtype=np.int32),
        np.empty(0, dtype=np.int32),
    )
    return surface, vertex, body, ContactPredicateFixture(n_vertices)


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("backend", [gs.gpu])
def test_f64_to_f32_outward_rounding_is_directed():
    f32_values = np.array(
        [
            -np.finfo(np.float32).max,
            -12345.5,
            -1.0,
            -np.finfo(np.float32).tiny,
            -0.0,
            0.0,
            np.finfo(np.float32).tiny,
            1.0,
            12345.5,
            np.finfo(np.float32).max,
        ],
        dtype=np.float32,
    )
    exact = f32_values.astype(np.float64)
    values = np.concatenate(
        (
            exact,
            np.nextafter(exact, -np.inf),
            np.nextafter(exact, np.inf),
            np.array(
                [
                    -1e-300,
                    1e-300,
                    -np.pi,
                    np.pi,
                    -1e20 + 3.0,
                    1e20 - 3.0,
                ],
                dtype=np.float64,
            ),
        )
    )
    source = qd.ndarray(qd.f64, shape=(len(values),))
    lower = qd.ndarray(qd.f32, shape=(len(values),))
    upper = qd.ndarray(qd.f32, shape=(len(values),))
    source.from_numpy(values)
    convert_f64_to_f32_outward(source, lower, upper)

    actual_lower = qd_to_numpy(lower)
    actual_upper = qd_to_numpy(upper)
    nearest = values.astype(np.float32)
    expected_lower = nearest.copy()
    expected_upper = nearest.copy()
    lower_adjust = nearest.astype(np.float64) > values
    upper_adjust = nearest.astype(np.float64) < values
    with np.errstate(over="ignore"):
        expected_lower[lower_adjust] = np.nextafter(
            expected_lower[lower_adjust],
            np.float32(-np.inf),
        )
        expected_upper[upper_adjust] = np.nextafter(
            expected_upper[upper_adjust],
            np.float32(np.inf),
        )

    np.testing.assert_array_equal(actual_lower, expected_lower)
    np.testing.assert_array_equal(actual_upper, expected_upper)
    bad_lower = np.flatnonzero(actual_lower.astype(np.float64) > values)
    bad_upper = np.flatnonzero(actual_upper.astype(np.float64) < values)
    assert not len(bad_lower), [(int(index), values[index], actual_lower[index]) for index in bad_lower]
    assert not len(bad_upper), [(int(index), values[index], actual_upper[index]) for index in bad_upper]
    np.testing.assert_array_equal(actual_lower[: len(exact)], f32_values)
    np.testing.assert_array_equal(actual_upper[: len(exact)], f32_values)


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("backend", [gs.gpu])
def test_dop14f_leaf_bounds_conservatively_contain_fp64_reference():
    base = np.array(
        [
            [0.1, -0.2, 0.3],
            [0.4, 0.5, -0.6],
            [-0.7, 0.8, 0.9],
            [1.0, -1.1, 1.2],
            [-1.3, 1.4, -1.5],
            [1.6, 1.7, -1.8],
        ],
        dtype=np.float32,
    ).astype(np.float64)
    direction = np.where(np.arange(base.size).reshape(base.shape) % 2 == 0, np.inf, -np.inf)
    positions = np.nextafter(base, direction)
    triangles = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
    surface, vertex, _, _ = make_single_body_pt_scene(positions, triangles)
    endpoints = np.nextafter(positions + 3.0e-8, -direction)
    vertex.trajectory_end_positions.from_numpy(endpoints)

    dop14f = LBVH(2, 6, "dop14")
    fp64_reference = LBVH(
        2,
        6,
        "dop14",
        False,
        True,
    )
    calc_triangle_leaves(dop14f, surface, vertex)
    calc_triangle_leaves(fp64_reference, surface, vertex)

    packed = qd_to_numpy(dop14f.aabbs)
    reference = qd_to_numpy(fp64_reference.aabbs)
    assert packed.dtype == np.float32
    assert packed.shape == (3, 16)
    assert packed.strides[0] == 64
    device_view = dop14f.aabbs.to_torch(copy=False)
    assert device_view.data_ptr() % 64 == 0
    assert device_view.stride(0) * device_view.element_size() == 64

    leaves = slice(1, 3)
    assert np.all(packed[leaves, :7].astype(np.float64) <= reference[leaves, :7])
    assert np.all(packed[leaves, 7:14].astype(np.float64) >= reference[leaves, 7:14])


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("backend", [gs.gpu])
def test_lbvh_dynamic_morton_sort_matches_retained_generic_path():
    count = 257
    rng = np.random.default_rng(31)
    padded = ((count + 63) // 64) * 64
    keys = np.full(padded, np.iinfo(np.uint64).max, dtype=np.uint64)
    keys[:count] = rng.integers(
        0,
        np.iinfo(np.uint64).max,
        size=count,
        dtype=np.uint64,
        endpoint=False,
    )
    keys[1:count:17] = keys[:count:17][: len(keys[1:count:17])]
    permutation = np.full(padded, -1, dtype=np.int32)
    permutation[:count] = np.arange(count, dtype=np.int32)

    outputs = []
    for genesis_legacy_sort_reduce in (False, True):
        bvh = LBVH(
            count,
            count,
            "aabb",
            genesis_legacy_sort_reduce,
        )
        bvh.morton.from_numpy(keys)
        bvh.morton_tmp.from_numpy(np.zeros(padded, dtype=np.uint64))
        bvh.srt_perm.from_numpy(permutation)
        bvh.srt_tmp_perm.from_numpy(np.zeros(padded, dtype=np.int32))
        sort_morton_only(bvh)
        qd.sync()
        outputs.append(
            (
                qd_to_numpy(bvh.morton)[:count],
                qd_to_numpy(bvh.srt_perm)[:count],
            )
        )

    order = np.argsort(keys[:count], kind="stable")
    for actual_keys, actual_permutation in outputs:
        np.testing.assert_array_equal(actual_keys, keys[:count][order])
        np.testing.assert_array_equal(
            actual_permutation,
            permutation[:count][order],
        )
    np.testing.assert_array_equal(outputs[0][0], outputs[1][0])
    np.testing.assert_array_equal(outputs[0][1], outputs[1][1])


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("backend", [gs.gpu])
@pytest.mark.parametrize(
    (
        "bound_type",
        "genesis_legacy_fp64_bounds",
        "genesis_legacy_refit",
    ),
    [
        ("aabb", False, False),
        ("dop14", False, False),
        ("dop14", True, True),
    ],
)
def test_lbvh_pt_candidates_match_two_parallel_triangles(
    bound_type,
    genesis_legacy_fp64_bounds,
    genesis_legacy_refit,
):
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.006],
            [1.0, 0.0, 0.006],
            [0.0, 1.0, 0.006],
        ],
        dtype=np.float64,
    )
    vertex = GlobalVertexManager()
    vertex.init(6)
    vertex.positions.from_numpy(positions)
    vertex.safe_positions.from_numpy(positions)
    vertex.trajectory_end_positions.from_numpy(positions)
    vertex.x_bar.from_numpy(positions)
    vertex.body_id.from_numpy(np.array([0, 0, 0, 1, 1, 1], dtype=np.int32))
    vertex.wire_thickness_data(np.full(6, 0.001, dtype=np.float64))
    vertex.wire_d_hat_data(np.full(6, 0.01, dtype=np.float64))
    vertex.wire_is_fixed_data(np.zeros(6, dtype=np.int32))

    surface = GlobalSurfaceManager()
    surface.wire_surface_data(
        np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32),
        np.array([[0, 1], [1, 2], [0, 2], [3, 4], [4, 5], [3, 5]], dtype=np.int32),
        np.arange(6, dtype=np.int32),
    )
    surface.wire_vert_dimensions(np.full(6, 2, dtype=np.int32))
    surface.wire_area_weights(
        np.full(6, 1.0 / 6.0, dtype=np.float64),
        np.full(6, 1.0 / 6.0, dtype=np.float64),
        np.full(2, 0.5, dtype=np.float64),
    )

    body = GlobalBodyManager()
    body.init(2)
    body.vertex_offsets.from_numpy(np.array([0, 3, 6], dtype=np.int32))
    body.self_collision.from_numpy(np.ones(2, dtype=np.int32))
    body.wire_body_contact_ignorance(
        np.array([0, 0, 0], dtype=np.int32),
        np.empty(0, dtype=np.int32),
    )

    bvh = LBVH(
        2,
        6,
        bound_type,
        False,
        genesis_legacy_fp64_bounds,
        genesis_legacy_refit,
    )
    pairs = qd.ndarray(qd.i32, shape=(16, 2))
    n_pairs = qd.ndarray(qd.i32, shape=())
    overflow = qd.ndarray(qd.i32, shape=())
    contact = ContactPredicateFixture(len(positions))
    build_and_query_pt(
        bvh,
        surface,
        vertex,
        body,
        contact,
        pairs,
        n_pairs,
        overflow,
    )

    assert int(qd_to_numpy(overflow)) == 0
    assert int(qd_to_numpy(n_pairs)) == 6
    actual = {tuple(pair) for pair in qd_to_numpy(pairs)[:6]}
    assert actual == {(0, 1), (1, 1), (2, 1), (3, 0), (4, 0), (5, 0)}


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("backend", [gs.gpu])
def test_pt_warp_matches_batched_for_single_leaf():
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.2, 0.2, 0.006],
        ],
        dtype=np.float64,
    )
    triangles = np.array([[0, 1, 2]], dtype=np.int32)
    surface, vertex, body, contact = make_single_body_pt_scene(
        positions,
        triangles,
    )
    bvh = LBVH(1, 4, "dop14")
    build_triangle_bvh(bvh, surface, vertex)

    batched_pairs = qd.ndarray(qd.i32, shape=(4, 2))
    batched_count = qd.ndarray(qd.i32, shape=())
    warp_pairs = qd.ndarray(qd.i32, shape=(4, 2))
    warp_count = qd.ndarray(qd.i32, shape=())
    overflow = qd.ndarray(qd.i32, shape=())

    query_pt_batched_only(
        bvh,
        surface,
        vertex,
        body,
        contact,
        batched_pairs,
        batched_count,
        4,
        overflow,
    )
    query_pt_warp_only(
        bvh,
        surface,
        vertex,
        body,
        contact,
        warp_pairs,
        warp_count,
        4,
        overflow,
    )

    assert int(qd_to_numpy(batched_count)) == 1
    assert int(qd_to_numpy(warp_count)) == 1
    np.testing.assert_array_equal(
        qd_to_numpy(batched_pairs)[0],
        np.array([3, 0], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        qd_to_numpy(warp_pairs)[0],
        np.array([3, 0], dtype=np.int32),
    )

    contact.enable_table.from_numpy(np.zeros(1, dtype=np.int32))
    query_pt_batched_only(
        bvh,
        surface,
        vertex,
        body,
        contact,
        batched_pairs,
        batched_count,
        4,
        overflow,
    )
    query_pt_warp_only(
        bvh,
        surface,
        vertex,
        body,
        contact,
        warp_pairs,
        warp_count,
        4,
        overflow,
    )
    assert int(qd_to_numpy(batched_count)) == 0
    assert int(qd_to_numpy(warp_count)) == 0


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("backend", [gs.gpu])
def test_pt_warp_matches_batched_dense_and_exact_overflow_count():
    n_vertices = 64
    n_triangles = 33
    rng = np.random.default_rng(47)
    positions = rng.uniform(
        -1e-4,
        1e-4,
        size=(n_vertices, 3),
    ).astype(np.float64)
    triangle_index = np.arange(n_triangles, dtype=np.int32)
    triangles = np.column_stack(
        (
            triangle_index,
            (triangle_index + 17) % n_vertices,
            (triangle_index + 37) % n_vertices,
        )
    ).astype(np.int32)
    surface, vertex, body, contact = make_single_body_pt_scene(
        positions,
        triangles,
    )
    bvh = LBVH(n_triangles, n_vertices, "dop14")
    build_triangle_bvh(bvh, surface, vertex)

    expected_count = n_vertices * n_triangles - 3 * n_triangles
    batched_pairs = qd.ndarray(
        qd.i32,
        shape=(expected_count, 2),
    )
    batched_count = qd.ndarray(qd.i32, shape=())
    warp_pairs = qd.ndarray(
        qd.i32,
        shape=(expected_count, 2),
    )
    warp_count = qd.ndarray(qd.i32, shape=())
    overflow = qd.ndarray(qd.i32, shape=())

    query_pt_batched_only(
        bvh,
        surface,
        vertex,
        body,
        contact,
        batched_pairs,
        batched_count,
        expected_count,
        overflow,
    )
    assert int(qd_to_numpy(overflow)) == 0
    query_pt_warp_only(
        bvh,
        surface,
        vertex,
        body,
        contact,
        warp_pairs,
        warp_count,
        expected_count,
        overflow,
    )
    assert int(qd_to_numpy(overflow)) == 0
    assert int(qd_to_numpy(batched_count)) == expected_count
    assert int(qd_to_numpy(warp_count)) == expected_count

    batched = {tuple(pair) for pair in qd_to_numpy(batched_pairs)[:expected_count]}
    warp = {tuple(pair) for pair in qd_to_numpy(warp_pairs)[:expected_count]}
    assert batched == warp
    assert len(warp) == expected_count

    limited_capacity = 17
    query_pt_batched_only(
        bvh,
        surface,
        vertex,
        body,
        contact,
        batched_pairs,
        batched_count,
        limited_capacity,
        overflow,
    )
    assert int(qd_to_numpy(overflow)) == 1
    assert int(qd_to_numpy(batched_count)) == expected_count
    query_pt_warp_only(
        bvh,
        surface,
        vertex,
        body,
        contact,
        warp_pairs,
        warp_count,
        limited_capacity,
        overflow,
    )
    assert int(qd_to_numpy(overflow)) == 1
    assert int(qd_to_numpy(warp_count)) == expected_count


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("backend", [gs.gpu])
def test_dop14_dual_ee_query():
    positions = np.array(
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
    vertex.positions.from_numpy(positions)
    vertex.safe_positions.from_numpy(positions)
    vertex.trajectory_end_positions.from_numpy(positions)
    vertex.x_bar.from_numpy(positions)
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

    body = GlobalBodyManager()
    body.init(2)
    body.vertex_offsets.from_numpy(np.array([0, 2, 4], dtype=np.int32))
    body.self_collision.from_numpy(np.ones(2, dtype=np.int32))
    body.wire_body_contact_ignorance(
        np.array([0, 0, 0], dtype=np.int32),
        np.empty(0, dtype=np.int32),
    )

    bvh = LBVH(2, 2, "dop14")
    dual = DualEEQueryState(2, 0, 24.0, 18)
    pairs = qd.ndarray(qd.i32, shape=(16, 2))
    n_pairs = qd.ndarray(qd.i32, shape=())
    overflow = qd.ndarray(qd.i32, shape=())
    contact = ContactPredicateFixture(len(positions))
    build_and_query_ee_dual(
        bvh,
        dual,
        surface,
        vertex,
        body,
        contact,
        pairs,
        n_pairs,
        overflow,
    )

    assert int(qd_to_numpy(overflow)) == 0
    assert int(qd_to_numpy(n_pairs)) == 1
    np.testing.assert_array_equal(qd_to_numpy(pairs)[0], np.array([0, 1], dtype=np.int32))

    contact.enable_ee_table.from_numpy(np.zeros(1, dtype=np.int32))
    query_ee_dual_only(
        bvh,
        dual,
        surface,
        vertex,
        body,
        contact,
        pairs,
        n_pairs,
        overflow,
    )
    assert int(qd_to_numpy(n_pairs)) == 0
    query_ee_warp(
        bvh,
        surface,
        vertex,
        body,
        contact,
        pairs,
        n_pairs,
        overflow,
    )
    assert int(qd_to_numpy(n_pairs)) == 0


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("backend", [gs.gpu])
def test_dop14_dual_ee_matches_per_edge_query():
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.006],
            [1.0, 0.0, 0.006],
            [0.0, 1.0, 0.006],
        ],
        dtype=np.float64,
    )
    edges = np.array([[0, 1], [1, 2], [0, 2], [3, 4], [4, 5], [3, 5]], dtype=np.int32)
    vertex = GlobalVertexManager()
    vertex.init(6)
    vertex.positions.from_numpy(positions)
    vertex.safe_positions.from_numpy(positions)
    vertex.trajectory_end_positions.from_numpy(positions)
    vertex.x_bar.from_numpy(positions)
    vertex.body_id.from_numpy(np.array([0, 0, 0, 1, 1, 1], dtype=np.int32))
    vertex.wire_thickness_data(np.full(6, 0.001, dtype=np.float64))
    vertex.wire_d_hat_data(np.full(6, 0.01, dtype=np.float64))
    vertex.wire_is_fixed_data(np.zeros(6, dtype=np.int32))
    surface = GlobalSurfaceManager()
    surface.wire_surface_data(
        np.empty((0, 3), dtype=np.int32),
        edges,
        np.arange(6, dtype=np.int32),
    )
    surface.wire_vert_dimensions(np.full(6, 2, dtype=np.int32))
    surface.wire_area_weights(
        np.full(6, 1.0 / 6.0, dtype=np.float64),
        np.full(6, 1.0 / 6.0, dtype=np.float64),
        np.empty(0, dtype=np.float64),
    )
    body = GlobalBodyManager()
    body.init(2)
    body.vertex_offsets.from_numpy(np.array([0, 3, 6], dtype=np.int32))
    body.self_collision.from_numpy(np.ones(2, dtype=np.int32))
    body.wire_body_contact_ignorance(
        np.array([0, 0, 0], dtype=np.int32),
        np.empty(0, dtype=np.int32),
    )

    bvh = LBVH(6, 6, "dop14")
    dual_state = DualEEQueryState(6, 0, 24.0, 18)
    dual_pairs = qd.ndarray(qd.i32, shape=(64, 2))
    dual_count = qd.ndarray(qd.i32, shape=())
    warp_pairs = qd.ndarray(qd.i32, shape=(64, 2))
    warp_count = qd.ndarray(qd.i32, shape=())
    overflow = qd.ndarray(qd.i32, shape=())
    contact = ContactPredicateFixture(len(positions))
    build_and_query_ee_dual(
        bvh,
        dual_state,
        surface,
        vertex,
        body,
        contact,
        dual_pairs,
        dual_count,
        overflow,
    )
    query_ee_warp(
        bvh,
        surface,
        vertex,
        body,
        contact,
        warp_pairs,
        warp_count,
        overflow,
    )

    dual = {tuple(pair) for pair in qd_to_numpy(dual_pairs)[: int(qd_to_numpy(dual_count))]}
    warp = {tuple(pair) for pair in qd_to_numpy(warp_pairs)[: int(qd_to_numpy(warp_count))]}
    assert dual == warp

    dual_state.frontier_capacity.from_numpy(np.array(1, dtype=np.int32))
    build_and_query_ee_dual(
        bvh,
        dual_state,
        surface,
        vertex,
        body,
        contact,
        dual_pairs,
        dual_count,
        overflow,
    )
    assert int(qd_to_numpy(overflow)) == 1
    assert int(qd_to_numpy(dual_state.required_frontier)) > 1
    assert dual_state.handle_overflow()
    query_ee_dual_only(
        bvh,
        dual_state,
        surface,
        vertex,
        body,
        contact,
        dual_pairs,
        dual_count,
        overflow,
    )
    assert int(qd_to_numpy(overflow)) == 0
    replay = {tuple(pair) for pair in qd_to_numpy(dual_pairs)[: int(qd_to_numpy(dual_count))]}
    assert replay == warp


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("backend", [gs.gpu])
def test_dual_ee_morton_reorders_body_metadata_with_leaves():
    positions = np.array(
        [
            [-0.01, 0.0, 0.0],
            [0.01, 0.0, 0.0],
            [0.99, 0.0, 0.0],
            [1.01, 0.0, 0.0],
            [0.0, -0.01, 0.001],
            [0.0, 0.01, 0.001],
            [1.0, -0.01, 0.001],
            [1.0, 0.01, 0.001],
        ],
        dtype=np.float64,
    )
    edges = np.array(
        [[0, 1], [2, 3], [4, 5], [6, 7]],
        dtype=np.int32,
    )
    vertex = GlobalVertexManager()
    vertex.init(8)
    vertex.positions.from_numpy(positions)
    vertex.safe_positions.from_numpy(positions)
    vertex.trajectory_end_positions.from_numpy(positions)
    vertex.x_bar.from_numpy(positions)
    vertex.body_id.from_numpy(np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int32))
    vertex.wire_thickness_data(np.full(8, 0.001, dtype=np.float64))
    vertex.wire_d_hat_data(np.full(8, 0.01, dtype=np.float64))
    vertex.wire_is_fixed_data(np.zeros(8, dtype=np.int32))

    surface = GlobalSurfaceManager()
    surface.wire_surface_data(
        np.empty((0, 3), dtype=np.int32),
        edges,
        np.arange(8, dtype=np.int32),
    )
    surface.wire_vert_dimensions(np.full(8, 2, dtype=np.int32))
    surface.wire_area_weights(
        np.full(8, 1.0 / 8.0, dtype=np.float64),
        np.full(4, 0.25, dtype=np.float64),
        np.empty(0, dtype=np.float64),
    )
    body = GlobalBodyManager()
    body.init(2)
    body.vertex_offsets.from_numpy(np.array([0, 4, 8], dtype=np.int32))
    body.self_collision.from_numpy(np.ones(2, dtype=np.int32))
    body.wire_body_contact_ignorance(
        np.array([0, 0, 0], dtype=np.int32),
        np.empty(0, dtype=np.int32),
    )

    bvh = LBVH(4, 4, "dop14")
    dual_state = DualEEQueryState(4, 0, 24.0, 18)
    dual_pairs = qd.ndarray(qd.i32, shape=(16, 2))
    dual_count = qd.ndarray(qd.i32, shape=())
    warp_pairs = qd.ndarray(qd.i32, shape=(16, 2))
    warp_count = qd.ndarray(qd.i32, shape=())
    overflow = qd.ndarray(qd.i32, shape=())
    contact = ContactPredicateFixture(len(positions))
    build_and_query_ee_dual(
        bvh,
        dual_state,
        surface,
        vertex,
        body,
        contact,
        dual_pairs,
        dual_count,
        overflow,
    )
    query_ee_warp(
        bvh,
        surface,
        vertex,
        body,
        contact,
        warp_pairs,
        warp_count,
        overflow,
    )

    n = int(qd_to_numpy(bvh.n_prims))
    elements = qd_to_numpy(bvh.nodes_element)[n - 1 : 2 * n - 1]
    actual_bodies = qd_to_numpy(bvh.node_body_id)[n - 1 : 2 * n - 1]
    expected_bodies = np.array(
        [qd_to_numpy(vertex.body_id)[edges[element, 0]] for element in elements],
        dtype=np.int32,
    )
    np.testing.assert_array_equal(actual_bodies, expected_bodies)

    dual_pairs_set = {tuple(pair) for pair in qd_to_numpy(dual_pairs)[: int(qd_to_numpy(dual_count))]}
    warp_pairs_set = {tuple(pair) for pair in qd_to_numpy(warp_pairs)[: int(qd_to_numpy(warp_count))]}
    assert dual_pairs_set == warp_pairs_set
    assert dual_pairs_set == {(0, 2), (1, 3)}

import numpy as np
import pytest
import quadrants as qd

import genesis as gs
from genesis.engine.systems.fsr_reduce import (
    fast_segmented_reduce_body,
    fast_segmented_reduce_doublet,
    fast_segmented_reduce_triplet,
)
from genesis.utils.misc import qd_to_numpy


@qd.kernel(graph=True, fastcache=True)
def run_fsr_doublet(
    seg_ids: qd.types.ndarray(),
    permutation: qd.types.ndarray(),
    keys: qd.types.ndarray(),
    values: qd.types.ndarray(),
    output: qd.types.ndarray(),
    n: qd.types.ndarray(),
):
    for index in range(output.shape[0]):
        for component in qd.static(range(3)):
            output[index, component] = 0.0
    fast_segmented_reduce_doublet(
        seg_ids,
        permutation,
        keys,
        values,
        output,
        n,
        n,
        output.shape[0],
    )


@qd.kernel(graph=True, fastcache=True)
def run_fsr_triplet(
    seg_ids: qd.types.ndarray(),
    permutation: qd.types.ndarray(),
    keys: qd.types.ndarray(),
    values: qd.types.ndarray(),
    output: qd.types.ndarray(),
    n: qd.types.ndarray(),
):
    for index in range(output.shape[0]):
        for row in qd.static(range(3)):
            for column in qd.static(range(3)):
                output[index, row, column] = 0.0
    fast_segmented_reduce_triplet(
        seg_ids,
        permutation,
        keys,
        values,
        output,
        n,
        n,
        output.shape[0],
    )


@qd.kernel(graph=True, fastcache=True)
def run_fsr_body(
    seg_ids: qd.types.ndarray(),
    permutation: qd.types.ndarray(),
    keys: qd.types.ndarray(),
    values: qd.types.ndarray(),
    output: qd.types.ndarray(),
    n: qd.types.ndarray(),
):
    for index in range(output.shape[0]):
        output[index] = 0.0
    fast_segmented_reduce_body(
        seg_ids,
        permutation,
        keys,
        values,
        output,
        n,
        n,
        output.shape[0] // 9,
        qd.f64,
        9,
    )


def segment_ids(keys: np.ndarray) -> np.ndarray:
    result = np.zeros(len(keys), dtype=np.int32)
    if len(keys) > 1:
        result[1:] = np.cumsum(keys[1:] != keys[:-1])
    return result


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("backend", [gs.gpu])
def test_fsr_matches_numpy_segmented_sum():
    capacity = 1024
    count = 777
    rng = np.random.default_rng(11)
    live_keys = np.sort(
        rng.integers(0, 97, size=count, dtype=np.uint32),
        kind="stable",
    )
    segments = segment_ids(live_keys)
    n_segments = int(segments[-1]) + 1
    permutation_host = rng.permutation(capacity).astype(np.int32)
    n = qd.ndarray(qd.i32, shape=())
    n.from_numpy(np.array(count, dtype=np.int32))

    keys_u32 = qd.ndarray(qd.u32, shape=(capacity,))
    keys_u64 = qd.ndarray(qd.u64, shape=(capacity,))
    seg_ids = qd.ndarray(qd.i32, shape=(capacity,))
    permutation = qd.ndarray(qd.i32, shape=(capacity,))
    keys32_host = np.full(capacity, np.iinfo(np.uint32).max, dtype=np.uint32)
    keys64_host = np.full(capacity, np.iinfo(np.uint64).max, dtype=np.uint64)
    seg_host = np.zeros(capacity, dtype=np.int32)
    keys32_host[:count] = live_keys
    keys64_host[:count] = live_keys.astype(np.uint64)
    seg_host[:count] = segments
    keys_u32.from_numpy(keys32_host)
    keys_u64.from_numpy(keys64_host)
    seg_ids.from_numpy(seg_host)
    permutation.from_numpy(permutation_host)

    doublet_values_host = rng.normal(size=(capacity, 3))
    doublet_values = qd.ndarray(qd.f64, shape=(capacity, 3))
    doublet_output = qd.ndarray(qd.f64, shape=(capacity, 3))
    doublet_values.from_numpy(doublet_values_host)
    run_fsr_doublet(
        seg_ids,
        permutation,
        keys_u32,
        doublet_values,
        doublet_output,
        n,
    )

    triplet_values_host = rng.normal(size=(capacity, 3, 3))
    triplet_values = qd.ndarray(qd.f64, shape=(capacity, 3, 3))
    triplet_output = qd.ndarray(qd.f64, shape=(capacity, 3, 3))
    triplet_values.from_numpy(triplet_values_host)
    run_fsr_triplet(
        seg_ids,
        permutation,
        keys_u64,
        triplet_values,
        triplet_output,
        n,
    )

    body_values_host = rng.normal(size=capacity * 9)
    body_values = qd.ndarray(qd.f64, shape=(capacity * 9,))
    body_output = qd.ndarray(qd.f64, shape=(capacity * 9,))
    body_values.from_numpy(body_values_host)
    run_fsr_body(
        seg_ids,
        permutation,
        keys_u64,
        body_values,
        body_output,
        n,
    )
    qd.sync()

    expected_doublet = np.zeros((n_segments, 3))
    expected_triplet = np.zeros((n_segments, 3, 3))
    expected_body = np.zeros((n_segments, 9))
    for index in range(count):
        source = permutation_host[index]
        segment = segments[index]
        expected_doublet[segment] += doublet_values_host[source]
        expected_triplet[segment] += triplet_values_host[source]
        expected_body[segment] += body_values_host[source * 9 : source * 9 + 9]

    np.testing.assert_allclose(
        qd_to_numpy(doublet_output)[:n_segments],
        expected_doublet,
        rtol=1.0e-13,
        atol=1.0e-13,
    )
    np.testing.assert_allclose(
        qd_to_numpy(triplet_output)[:n_segments],
        expected_triplet,
        rtol=1.0e-13,
        atol=1.0e-13,
    )
    np.testing.assert_allclose(
        qd_to_numpy(body_output)[: n_segments * 9].reshape(
            n_segments,
            9,
        ),
        expected_body,
        rtol=1.0e-13,
        atol=1.0e-13,
    )


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("backend", [gs.gpu])
def test_fsr_large_simulation_counts():
    capacity = 1_000_000
    keys = qd.ndarray(qd.u64, shape=(capacity,))
    seg_ids = qd.ndarray(qd.i32, shape=(capacity,))
    permutation = qd.ndarray(qd.i32, shape=(capacity,))
    values = qd.ndarray(qd.f64, shape=(capacity * 9,))
    output = qd.ndarray(qd.f64, shape=(capacity * 9,))
    n = qd.ndarray(qd.i32, shape=())

    permutation_host = np.arange(capacity, dtype=np.int32)
    component_scale = np.arange(1, 10, dtype=np.float64)
    values_host = np.tile(component_scale, capacity)
    permutation.from_numpy(permutation_host)
    values.from_numpy(values_host)

    for count in (4_865, 100_000, capacity):
        live_keys = (np.arange(count, dtype=np.uint64) // 7).astype(np.uint64)
        segments = segment_ids(live_keys)
        n_segments = int(segments[-1]) + 1
        keys_host = np.full(
            capacity,
            np.iinfo(np.uint64).max,
            dtype=np.uint64,
        )
        segments_host = np.zeros(capacity, dtype=np.int32)
        keys_host[:count] = live_keys
        segments_host[:count] = segments
        keys.from_numpy(keys_host)
        seg_ids.from_numpy(segments_host)
        n.from_numpy(np.array(count, dtype=np.int32))

        run_fsr_body(
            seg_ids,
            permutation,
            keys,
            values,
            output,
            n,
        )
        qd.sync()

        segment_counts = np.bincount(
            segments,
            minlength=n_segments,
        )
        expected = segment_counts[:, None] * component_scale[None, :]
        np.testing.assert_allclose(
            qd_to_numpy(output)[: n_segments * 9].reshape(
                n_segments,
                9,
            ),
            expected,
            rtol=0.0,
            atol=0.0,
        )

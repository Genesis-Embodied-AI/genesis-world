import numpy as np
import pytest
import quadrants as qd

import genesis as gs
from genesis.engine.systems.dynamic_radix_sort import (
    DynamicRadixSort,
    dynamic_radix_sort,
)
from genesis.utils.misc import qd_to_numpy


@qd.data_oriented
class DynamicRadixGrowthFixture:
    def __init__(self, capacity: int):
        self.reallocate(capacity)

    def reallocate(self, capacity: int) -> None:
        self.sorter = DynamicRadixSort(qd.u64, capacity)
        self.keys_a = qd.ndarray(qd.u64, shape=(capacity,))
        self.keys_b = qd.ndarray(qd.u64, shape=(capacity,))
        self.perm_a = qd.ndarray(qd.i32, shape=(capacity,))
        self.perm_b = qd.ndarray(qd.i32, shape=(capacity,))
        self.n = qd.ndarray(qd.i32, shape=())

    @qd.kernel(graph=True, fastcache=False)
    def run(self):
        dynamic_radix_sort(
            self.sorter,
            self.keys_a,
            self.keys_b,
            self.perm_a,
            self.perm_b,
            self.n[()],
        )


@qd.kernel(graph=True, fastcache=False)
def run_dynamic_radix_sort(
    sorter: qd.template(),
    keys_a: qd.types.ndarray(),
    keys_b: qd.types.ndarray(),
    perm_a: qd.types.ndarray(),
    perm_b: qd.types.ndarray(),
    n: qd.types.ndarray(),
):
    dynamic_radix_sort(
        sorter,
        keys_a,
        keys_b,
        perm_a,
        perm_b,
        n[()],
    )


@qd.kernel(graph=True, fastcache=False)
def run_dynamic_radix_histogram(
    sorter: qd.template(),
    keys: qd.types.ndarray(),
    n: qd.types.ndarray(),
):
    sorter.clear_bins()
    sorter.histogram(keys, n[()])
    sorter.scan_bins()


@qd.kernel(graph=True, fastcache=False)
def run_dynamic_radix_one_pass(
    sorter: qd.template(),
    keys_a: qd.types.ndarray(),
    keys_b: qd.types.ndarray(),
    perm_a: qd.types.ndarray(),
    perm_b: qd.types.ndarray(),
    n: qd.types.ndarray(),
):
    sorter.clear_bins()
    sorter.histogram(keys_a, n[()])
    sorter.scan_bins()
    sorter.clear_pass(0, n[()])
    sorter.onesweep_pass(
        keys_a,
        keys_b,
        perm_a,
        perm_b,
        n[()],
        0,
    )


@pytest.mark.required
@pytest.mark.parametrize("backend", [gs.gpu])
def test_dynamic_radix_histogram():
    capacity = 1024
    sorter = DynamicRadixSort(qd.u32, capacity)
    keys = qd.ndarray(qd.u32, shape=(capacity,))
    n = qd.ndarray(qd.i32, shape=())
    values = np.arange(capacity, dtype=np.uint32) % 64
    keys.from_numpy(values)
    n.from_numpy(np.array(capacity, dtype=np.int32))

    run_dynamic_radix_histogram(sorter, keys, n)
    qd.sync()

    expected = np.bincount(values & 0xFF, minlength=256)
    expected = np.cumsum(expected, dtype=np.int64) - expected
    np.testing.assert_array_equal(
        qd_to_numpy(sorter.bins)[0],
        expected,
    )


@pytest.mark.required
@pytest.mark.parametrize("backend", [gs.gpu])
def test_dynamic_radix_one_pass():
    capacity = 1024
    sorter = DynamicRadixSort(qd.u32, capacity)
    keys_a = qd.ndarray(qd.u32, shape=(capacity,))
    keys_b = qd.ndarray(qd.u32, shape=(capacity,))
    perm_a = qd.ndarray(qd.i32, shape=(capacity,))
    perm_b = qd.ndarray(qd.i32, shape=(capacity,))
    n = qd.ndarray(qd.i32, shape=())
    values = np.arange(capacity, dtype=np.uint32) % 64
    permutation = np.arange(capacity, dtype=np.int32)
    keys_a.from_numpy(values)
    keys_b.from_numpy(np.zeros(capacity, dtype=np.uint32))
    perm_a.from_numpy(permutation)
    perm_b.from_numpy(np.zeros(capacity, dtype=np.int32))
    n.from_numpy(np.array(capacity, dtype=np.int32))

    run_dynamic_radix_one_pass(
        sorter,
        keys_a,
        keys_b,
        perm_a,
        perm_b,
        n,
    )
    qd.sync()

    order = np.argsort(values & 0xFF, kind="stable")
    actual_keys = qd_to_numpy(keys_b)
    np.testing.assert_array_equal(actual_keys, values[order])
    np.testing.assert_array_equal(
        qd_to_numpy(perm_b),
        permutation[order],
    )


@pytest.mark.required
@pytest.mark.parametrize("backend", [gs.gpu])
def test_dynamic_radix_sort_reallocation_updates_dynamic_launch():
    rng = np.random.default_rng(13)
    fixture = DynamicRadixGrowthFixture(4_865)

    for capacity in (4_865, 20_000):
        if capacity != fixture.keys_a.shape[0]:
            fixture.reallocate(capacity)
        keys = rng.integers(
            0,
            np.iinfo(np.uint64).max,
            size=capacity,
            dtype=np.uint64,
        )
        keys[1::17] = keys[::17][: len(keys[1::17])]
        permutation = np.arange(capacity, dtype=np.int32)
        fixture.keys_a.from_numpy(keys)
        fixture.keys_b.from_numpy(np.zeros(capacity, dtype=np.uint64))
        fixture.perm_a.from_numpy(permutation)
        fixture.perm_b.from_numpy(np.zeros(capacity, dtype=np.int32))
        fixture.n.from_numpy(np.array(capacity, dtype=np.int32))

        fixture.run()
        qd.sync()

        order = np.argsort(keys, kind="stable")
        np.testing.assert_array_equal(
            qd_to_numpy(fixture.keys_a),
            keys[order],
        )
        np.testing.assert_array_equal(
            qd_to_numpy(fixture.perm_a),
            permutation[order],
        )


@pytest.mark.required
@pytest.mark.parametrize("backend", [gs.gpu])
@pytest.mark.parametrize(
    ("qd_dtype", "np_dtype"),
    ((qd.u32, np.uint32), (qd.u64, np.uint64)),
)
def test_dynamic_radix_sort_matches_stable_numpy(
    qd_dtype,
    np_dtype,
):
    capacity = 1024
    rng = np.random.default_rng(7)
    sorter = DynamicRadixSort(qd_dtype, capacity)
    keys_a = qd.ndarray(qd_dtype, shape=(capacity,))
    keys_b = qd.ndarray(qd_dtype, shape=(capacity,))
    perm_a = qd.ndarray(qd.i32, shape=(capacity,))
    perm_b = qd.ndarray(qd.i32, shape=(capacity,))
    n = qd.ndarray(qd.i32, shape=())

    for count in (0, 1, 31, 255, 256, 257, 777, capacity):
        keys = rng.integers(
            0,
            64,
            size=capacity,
            dtype=np_dtype,
        )
        if count:
            keys[0] = np.iinfo(np_dtype).max
        permutation = np.arange(capacity, dtype=np.int32)
        keys_a.from_numpy(keys)
        keys_b.from_numpy(np.zeros(capacity, dtype=np_dtype))
        perm_a.from_numpy(permutation)
        perm_b.from_numpy(np.zeros(capacity, dtype=np.int32))
        n.from_numpy(np.array(count, dtype=np.int32))

        run_dynamic_radix_sort(
            sorter,
            keys_a,
            keys_b,
            perm_a,
            perm_b,
            n,
        )
        qd.sync()

        order = np.argsort(keys[:count], kind="stable")
        np.testing.assert_array_equal(
            qd_to_numpy(keys_a)[:count],
            keys[:count][order],
        )
        np.testing.assert_array_equal(
            qd_to_numpy(perm_a)[:count],
            permutation[:count][order],
        )


@pytest.mark.required
@pytest.mark.parametrize("backend", [gs.gpu])
@pytest.mark.parametrize(
    ("qd_dtype", "np_dtype"),
    ((qd.u32, np.uint32), (qd.u64, np.uint64)),
)
def test_dynamic_radix_sort_large_simulation_counts(
    qd_dtype,
    np_dtype,
):
    capacity = 1_000_000
    rng = np.random.default_rng(19)
    sorter = DynamicRadixSort(qd_dtype, capacity)
    keys_a = qd.ndarray(qd_dtype, shape=(capacity,))
    keys_b = qd.ndarray(qd_dtype, shape=(capacity,))
    perm_a = qd.ndarray(qd.i32, shape=(capacity,))
    perm_b = qd.ndarray(qd.i32, shape=(capacity,))
    n = qd.ndarray(qd.i32, shape=())

    if np_dtype == np.uint32:
        keys = rng.integers(
            0,
            1 << 32,
            size=capacity,
            dtype=np.uint32,
        )
    else:
        low = rng.integers(
            0,
            1 << 32,
            size=capacity,
            dtype=np.uint64,
        )
        high = rng.integers(
            0,
            1 << 32,
            size=capacity,
            dtype=np.uint64,
        )
        keys = (high << np.uint64(32)) | low
    keys[1::97] = keys[::97][: len(keys[1::97])]
    keys[0] = np.iinfo(np_dtype).max
    permutation = np.arange(capacity, dtype=np.int32)

    for count in (4_865, 100_000, 500_000, capacity):
        keys_a.from_numpy(keys)
        keys_b.from_numpy(np.zeros(capacity, dtype=np_dtype))
        perm_a.from_numpy(permutation)
        perm_b.from_numpy(np.zeros(capacity, dtype=np.int32))
        n.from_numpy(np.array(count, dtype=np.int32))

        run_dynamic_radix_sort(
            sorter,
            keys_a,
            keys_b,
            perm_a,
            perm_b,
            n,
        )
        qd.sync()

        order = np.argsort(keys[:count], kind="stable")
        np.testing.assert_array_equal(
            qd_to_numpy(keys_a)[:count],
            keys[:count][order],
        )
        np.testing.assert_array_equal(
            qd_to_numpy(perm_a)[:count],
            permutation[:count][order],
        )

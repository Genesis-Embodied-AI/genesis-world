import numpy as np
import quadrants as qd

from genesis.engine.systems.dynamic_exclusive_sum import (
    DynamicExclusiveSum,
    dynamic_exclusive_sum,
)


@qd.data_oriented
class DynamicExclusiveSumFixture:
    def __init__(self, capacity: int):
        self.values = qd.ndarray(qd.i32, shape=(capacity,))
        self.output = qd.ndarray(qd.i32, shape=(capacity,))
        self.n = qd.ndarray(qd.i32, shape=())
        self.scanner = DynamicExclusiveSum(capacity)

    @qd.kernel(fastcache=False)
    def run(self):
        dynamic_exclusive_sum(
            self.scanner,
            self.values,
            self.output,
            self.n[()],
        )


def test_dynamic_exclusive_sum():
    qd.init(
        arch=qd.cuda,
        debug=False,
        log_level=qd.ERROR,
        advanced_optimization=False,
    )
    capacity = 1_000_000
    fixture = DynamicExclusiveSumFixture(capacity)

    rng = np.random.default_rng(20260930 + capacity)
    values = rng.integers(0, 8, size=capacity, dtype=np.int32)
    fixture.values.from_numpy(values)
    sentinel = np.iinfo(np.int32).max
    for n in (
        0,
        1,
        31,
        129,
        1024,
        1025,
        4865,
        99_731,
        capacity,
    ):
        fixture.output.from_numpy(np.full(capacity, sentinel, dtype=np.int32))
        fixture.n.from_numpy(np.array(n, dtype=np.int32))

        fixture.run()
        qd.sync()

        actual = fixture.output.to_numpy()
        expected = np.zeros(n, dtype=np.int32)
        if n > 1:
            expected[1:] = np.cumsum(
                values[: n - 1],
                dtype=np.int32,
            )
        np.testing.assert_array_equal(actual[:n], expected)
        if n < capacity:
            np.testing.assert_array_equal(
                actual[n:],
                np.full(
                    capacity - n,
                    sentinel,
                    dtype=np.int32,
                ),
            )

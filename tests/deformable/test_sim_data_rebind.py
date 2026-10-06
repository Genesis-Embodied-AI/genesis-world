import numpy as np
import pytest
import quadrants as qd

import genesis as gs


@qd.data_oriented
class RebindData:
    values: qd.Ndarray
    live_count: qd.Ndarray


def make_data(values: np.ndarray) -> RebindData:
    values = np.ascontiguousarray(values, dtype=np.int32)
    data = RebindData()
    data.values = qd.ndarray(qd.i32, shape=values.shape)
    data.live_count = qd.ndarray(qd.i32, shape=())
    data.values.from_numpy(values)
    data.live_count.from_numpy(np.array(len(values), dtype=np.int32))
    return data


@qd.kernel(graph=True, checkpoints=True, fastcache=True)
def copy_rebound_data(
    data: qd.template(),
    output: qd.types.ndarray(qd.i32, ndim=1),
    resize_required: qd.types.ndarray(qd.i32, ndim=0),
    never_yield: qd.types.ndarray(qd.i32, ndim=0),
):
    with qd.checkpoint(10, yield_on=resize_required):
        for _ in range(1):
            output[0] = -1
    with qd.checkpoint(20, yield_on=never_yield):
        for i in range(data.live_count[()]):
            output[i] = data.values[i]


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("backend", [gs.gpu])
def test_data_oriented_rebind_updates_pointer_on_launch_and_checkpoint_resume():
    output = qd.ndarray(qd.i32, shape=(8,))
    resize_required = qd.ndarray(qd.i32, shape=())
    never_yield = qd.ndarray(qd.i32, shape=())
    resize_required.from_numpy(np.array(0, dtype=np.int32))
    never_yield.from_numpy(np.array(0, dtype=np.int32))

    first = make_data(np.array([1, 2, 3, 4], dtype=np.int32))
    status = copy_rebound_data(first, output, resize_required, never_yield)
    assert not status.yielded
    np.testing.assert_array_equal(output.to_numpy()[:4], [1, 2, 3, 4])

    same_shape = make_data(np.array([5, 6, 7, 8], dtype=np.int32))
    first.values = same_shape.values
    first.live_count = same_shape.live_count
    status = copy_rebound_data(first, output, resize_required, never_yield)
    assert not status.yielded
    np.testing.assert_array_equal(output.to_numpy()[:4], [5, 6, 7, 8])

    resize_required.from_numpy(np.array(1, dtype=np.int32))
    status = copy_rebound_data(first, output, resize_required, never_yield)
    assert status.yielded
    assert status.checkpoint == 10

    grown = make_data(np.arange(10, 18, dtype=np.int32))
    first.values = grown.values
    first.live_count = grown.live_count
    resize_required.from_numpy(np.array(0, dtype=np.int32))
    status = copy_rebound_data.resume(
        first,
        output,
        resize_required,
        never_yield,
        from_checkpoint=20,
    )
    assert not status.yielded
    np.testing.assert_array_equal(output.to_numpy(), np.arange(10, 18, dtype=np.int32))

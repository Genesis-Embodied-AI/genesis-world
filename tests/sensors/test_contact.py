import numpy as np
import pytest

import genesis as gs
import genesis.utils.geom as gu

from ..utils.assertions import assert_allclose


@pytest.mark.slow  # ~200s
@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
def test_gravity_force(free_box, n_envs, show_viewer, tol):
    GRAVITY = -10.0
    BIAS = (0.1, 0.2, 0.3)
    NOISE = 0.01
    DT = 1e-2
    DELAY_STEPS = 2

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=DT,
            gravity=(0.0, 0.0, GRAVITY),
        ),
        profiling_options=gs.options.ProfilingOptions(
            show_FPS=False,
        ),
        show_viewer=show_viewer,
    )

    floor = scene.add_entity(
        morph=gs.morphs.Plane(),
    )

    # Add duck (with convex decomposition enabled) to offset geom index vs link index
    scene.add_entity(
        morph=gs.morphs.Mesh(
            file="meshes/duck.obj",
            scale=0.04,
            pos=(0.0, 1.0, 0.2),
            euler=(90, 0, 90),
        ),
    )

    box = scene.add_entity(
        morph=gs.morphs.MJCF(
            file=free_box((0.5, 0.5, 0.5), 1),  # volume = 1 m^3, mass = 1.0 kg
            pos=(0.0, 0.0, 0.55),
            align=False,
            default_armature=0.0,
        ),
        surface=gs.surfaces.Default(
            color=(1.0, 0.0, 0.0, 1.0),
        ),
    )
    box_2 = scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.2, 0.2, 0.2),  # volume = 0.008 m^3
            pos=(1.0, 0.0, 0.4),
        ),
        material=gs.materials.Rigid(
            rho=100.0,  # mass = 0.8 kg
        ),
        surface=gs.surfaces.Default(
            color=(0.0, 1.0, 0.0, 1.0),
        ),
    )
    box_3 = scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.2, 0.2, 0.2),  # volume = 0.008 m^3
            pos=(1.0, 0.0, 0.61),
        ),
        material=gs.materials.Rigid(
            rho=25.0,  # mass = 0.2 kg
        ),
        surface=gs.surfaces.Default(
            color=(0.0, 0.0, 1.0, 1.0),
        ),
    )

    bool_sensor_floor = scene.add_sensor(
        gs.sensors.Contact(
            entity_idx=floor.idx,
        )
    )
    bool_sensor_box_2 = scene.add_sensor(
        gs.sensors.Contact(
            entity_idx=box_2.idx,
        )
    )
    force_sensor = scene.add_sensor(
        gs.sensors.ContactForce(
            entity_idx=box.idx,
        )
    )
    force_sensor_box_2 = scene.add_sensor(
        gs.sensors.ContactForce(
            entity_idx=box_2.idx,
        )
    )
    force_sensor_noisy = scene.add_sensor(
        gs.sensors.ContactForce(
            entity_idx=box.idx,
            min_force=0.01,
            max_force=(10.0, 20.0, -GRAVITY / 2),
            noise=NOISE,
            bias=BIAS,
            random_walk=(NOISE * 0.01, NOISE * 0.02, NOISE * 0.03),
            delay=DT * DELAY_STEPS,
            jitter=0.01,
        )
    )
    # Adding extra sensor sharing same dtype to force discontinuous memory layout for ground truth when batched
    scene.add_sensor(
        gs.sensors.IMU(
            entity_idx=box.idx,
        )
    )

    scene.build(n_envs=n_envs)

    # Move CoM to get unbalanced forces on each contact points
    box_com_offset = (0.3, 0.1, 0.0)
    box.set_links_COM(box_com_offset)

    # Rotate the box make sure the force is correctly computed in local frame
    box_2.set_dofs_position((np.pi / 2, np.pi / 4, np.pi / 2), dofs_idx_local=slice(3, None))

    # Add another cube on top of it make sure the forces are correctly aggregated
    box_3.set_dofs_position((-np.pi / 2, -np.pi / 4, -np.pi / 2), dofs_idx_local=slice(3, None))

    # Note that it is necessary to do a first step, because the initial state right after reset is not valid
    for _ in range(DELAY_STEPS + 1):
        scene.step()

    # Make sure that box CoM is valid
    assert_allclose(box.get_links_pos(ref=gs.link_ref_frame.root_COM)[..., :2], box_com_offset[:2], tol=tol)

    assert not bool_sensor_floor.read().any(), "ContactSensor for floor should not detect any contact yet."
    assert not bool_sensor_box_2.read().any(), "ContactSensor for box_2 should not detect any contact yet."
    assert_allclose(force_sensor_noisy.read_ground_truth(), 0.0, tol=gs.EPS)
    assert_allclose(force_sensor.read(), force_sensor_noisy.read_ground_truth(), tol=gs.EPS)
    assert_allclose(force_sensor_noisy.read(), BIAS, tol=NOISE * 3)

    for _ in range(20):
        scene.step()

    assert bool_sensor_floor.read().all(), "ContactSensor for floor should detect contact with the ground"
    assert not bool_sensor_box_2.read().any(), "ContactSensor for box_2 should not detect any contact yet."
    assert_allclose(force_sensor_noisy.read(), force_sensor_noisy.read(), tol=gs.EPS)

    for _ in range(80):
        scene.step()

    assert bool_sensor_box_2.read().all(), "ContactSensor for box_2 should detect contact with the ground"

    # Moving force back in world frame because box is not perfectly flat on the ground due to CoM offset
    with np.testing.assert_raises(AssertionError):
        assert_allclose(box.get_quat(), 0.0, atol=tol)
    # Unsaturated GT physics check uses force_sensor (no max_force). force_sensor_noisy clamps in _post_process,
    # which applies uniformly to read() and read_ground_truth().
    assert_allclose(
        gu.transform_by_quat(force_sensor.read_ground_truth(), box.get_quat()), (0.0, 0.0, -GRAVITY), tol=tol
    )

    # FIXME: Adding CoM offset on box is disturbing contact force computations on box_2 for some reason...
    assert_allclose(force_sensor_box_2.read_ground_truth(), (-0.8 * GRAVITY, 0.0, 0.0), tol=1e-2)

    assert_allclose(force_sensor_noisy.read()[..., :2], BIAS[:2], tol=NOISE * 3)
    assert_allclose(force_sensor_noisy.read()[..., 2], -GRAVITY / 2, tol=gs.EPS)


@pytest.mark.slow  # ~200s
@pytest.mark.required
def test_filter_link_idx(show_viewer, tol):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            gravity=(0.0, 0.0, -10.0),
        ),
        profiling_options=gs.options.ProfilingOptions(
            show_FPS=False,
        ),
        show_viewer=show_viewer,
    )
    floor = scene.add_entity(
        morph=gs.morphs.Plane(),
    )
    box_on_floor = scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.2, 0.2, 0.2),
            pos=(0.0, 0.0, 0.1),
        ),
    )
    box = scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.2, 0.2, 0.2),
            pos=(0.0, 0.5, 0.1),
        ),
    )
    sensor = scene.add_sensor(
        gs.sensors.Contact(
            entity_idx=box_on_floor.idx,
        )
    )
    sensor_filtered = scene.add_sensor(
        gs.sensors.Contact(
            entity_idx=box_on_floor.idx,
            filter_link_idx=(floor.link_start,),
        )
    )
    force_sensor = scene.add_sensor(
        gs.sensors.ContactForce(
            entity_idx=box_on_floor.idx,
        )
    )
    force_sensor_filtered = scene.add_sensor(
        gs.sensors.ContactForce(
            entity_idx=box_on_floor.idx,
            filter_link_idx=(floor.link_start,),
        )
    )
    scene.build(n_envs=2)
    box.set_pos(
        (
            (0.0, 0.5, 0.1),  # box not touching box_on_floor
            (0.0, 0.0, 0.3),  # box on top of box_on_floor
        )
    )
    for _ in range(20):  # make sure the boxes are stably resting
        scene.step()
    data = sensor.read()
    filtered_data = sensor_filtered.read()
    assert data[0], "Contact sensor should detect contact with the floor"
    assert not filtered_data[0], "Contact sensor with filter_link_idx should filter out contact with the floor"
    assert data[1], "Contact sensor should detect contact with the box"
    assert filtered_data[1], "Contact sensor with filter_link_idx should still detect contact with the box"

    force = force_sensor.read()
    force_filtered = force_sensor_filtered.read()
    # env 0: box_on_floor only touches the floor, which supports it (+z).
    assert force[0, 2] > 1.0, "ContactForce should report the upward floor support force"
    # filtering the floor (the only contact) leaves zero force
    assert_allclose(force_filtered[0], 0.0, tol=tol)
    # env 1: box_on_floor touches the floor (below, +z support) and the top box (above, -z push).
    assert force[1, 2] > 1.0, "net contact force on box_on_floor is upward (floor support exceeds top-box weight)"
    assert force_filtered[1, 2] < -1.0, "filtering the floor leaves only the top box's downward push"
    # filtering the floor must change the result
    with np.testing.assert_raises(AssertionError):
        assert_allclose(force[1], force_filtered[1], tol=tol)

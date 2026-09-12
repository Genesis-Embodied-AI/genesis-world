import numpy as np
import pytest
import torch

import genesis as gs
import genesis.utils.geom as gu

from ..utils.assertions import assert_allclose, assert_equal


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
def test_sensor(free_box, show_viewer, tol, n_envs):
    GRAVITY = -10.0
    DT = 1e-2
    BIAS = (0.1, 0.2, 0.3)
    DELAY_STEPS = 2
    MAG_FIELD = (0.3, 0.1, 0.5)  # arbitrary world magnetic field

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=DT,
            substeps=1,
            gravity=(0.0, 0.0, GRAVITY),
        ),
        profiling_options=gs.options.ProfilingOptions(
            show_FPS=False,
        ),
        show_viewer=show_viewer,
    )

    scene.add_entity(gs.morphs.Plane())

    box = scene.add_entity(
        morph=gs.morphs.MJCF(
            file=free_box((0.05, 0.05, 0.05), 600),
            pos=(0.0, 0.0, 0.2),
            align=False,
            default_armature=0.0,
        ),
    )

    imu = scene.add_sensor(
        gs.sensors.IMU(
            entity_idx=box.idx,
            magnetic_field=MAG_FIELD,
        )
    )
    imu_delayed = scene.add_sensor(
        gs.sensors.IMU(
            entity_idx=box.idx,
            delay=DT * DELAY_STEPS,
            magnetic_field=MAG_FIELD,
        )
    )
    imu_noisy = scene.add_sensor(
        gs.sensors.IMU(
            entity_idx=box.idx,
            acc_cross_axis_coupling=0.01,
            gyro_cross_axis_coupling=(0.02, 0.03, 0.04),
            mag_cross_axis_coupling=0.01,
            acc_noise=(0.01, 0.01, 0.01),
            gyro_noise=(0.01, 0.01, 0.01),
            mag_noise=(0.01, 0.01, 0.01),
            acc_random_walk=(0.001, 0.001, 0.001),
            gyro_random_walk=(0.001, 0.001, 0.001),
            mag_random_walk=(0.001, 0.001, 0.001),
            delay=DT,
            magnetic_field=MAG_FIELD,
            jitter=DT * 0.1,
        )
    )

    scene.build(n_envs=n_envs)

    # box is in freefall
    for _ in range(10):
        scene.step()

    # IMU should calculate "classical linear acceleration" using the local frame without accounting for gravity
    # acc_classical_lin_z = - theta_dot ** 2 - cos(theta) * g
    assert_allclose(imu.read().lin_acc, 0.0, tol=tol)
    assert_allclose(imu.read().ang_vel, 0.0, tol=tol)
    assert_allclose(imu.read().mag, MAG_FIELD, tol=tol)
    assert_allclose(imu_noisy.read().lin_acc, 0.0, tol=1e-1)
    assert_allclose(imu_noisy.read().ang_vel, 0.0, tol=1e-1)
    assert_allclose(imu_noisy.read().mag, MAG_FIELD, tol=1e-1)

    # shift COM to induce angular velocity
    box.set_links_COM([0.05, 0.05, 0.05])

    # update noise and bias for accelerometer, gyroscope and magnetometer
    imu_noisy.set_noise((0.01, 0.01, 0.01, 0.02, 0.02, 0.02, 0.05, 0.05, 0.05))
    imu_noisy.set_bias((0.01, 0.01, 0.01, 0.02, 0.02, 0.02, 0.05, 0.05, 0.05))
    imu_noisy.set_jitter(0.001)

    for _ in range(10 - DELAY_STEPS):
        scene.step()

    true_imu_delayed_reading = imu_delayed.read_ground_truth()

    for _ in range(DELAY_STEPS):
        scene.step()

    assert_equal(imu_delayed.read().lin_acc, true_imu_delayed_reading.lin_acc)
    assert_equal(imu_delayed.read().ang_vel, true_imu_delayed_reading.ang_vel)
    assert_equal(imu_delayed.read().mag, true_imu_delayed_reading.mag)

    # check that position offset affects linear acceleration
    imu.set_pos_offset((0.5, 0.0, 0.0))
    lin_acc_no_offset = imu.read().lin_acc
    scene.step()
    lin_acc_with_offset = imu.read().lin_acc
    with np.testing.assert_raises(AssertionError):
        assert_allclose(lin_acc_no_offset, lin_acc_with_offset, atol=0.2)
    imu.set_pos_offset((0.0, 0.0, 0.0))

    # let box collide with ground
    for _ in range(20):
        scene.step()

    assert_equal(imu.read_ground_truth().lin_acc, imu_delayed.read_ground_truth().lin_acc)
    assert_equal(imu.read_ground_truth().ang_vel, imu_delayed.read_ground_truth().ang_vel)
    assert_equal(imu.read_ground_truth().mag, imu_delayed.read_ground_truth().mag)

    with np.testing.assert_raises(AssertionError, msg="Angular velocity should not be zero due to COM shift"):
        assert_allclose(imu.read_ground_truth().ang_vel, 0.0, tol=tol)

    with np.testing.assert_raises(AssertionError, msg="Delayed accl data should not be equal to the ground truth data"):
        assert_equal(imu_delayed.read().lin_acc - imu_delayed.read_ground_truth().lin_acc, 0.0)

    with np.testing.assert_raises(AssertionError, msg="Delayed mag data should not be equal to the ground truth data"):
        assert_equal(imu_delayed.read().mag - imu_delayed.read_ground_truth().mag, 0.0)

    box.set_links_COM([0.0, 0.0, 0.0])
    box.set_quat((0.0, 0.0, 0.0, 1.0))  # pi rotation around z-axis

    # wait for the box to be stationary on ground
    for _ in range(50):
        scene.step()

    assert_allclose(imu.read().lin_acc, (0.0, 0.0, -GRAVITY), tol=5e-6)
    assert_allclose(imu.read().ang_vel, (0.0, 0.0, 0.0), tol=1e-5)
    assert_allclose(imu.read().mag, (-MAG_FIELD[0], -MAG_FIELD[1], MAG_FIELD[2]), tol=tol)

    # rotate IMU 90 deg around x axis means gravity should be along -y axis
    imu.set_quat_offset(gu.euler_to_quat((90.0, 0.0, 0.0)))
    scene.step()
    assert_allclose(imu.read().lin_acc, (0.0, GRAVITY, 0.0), tol=5e-6)
    assert_allclose(imu.read().mag, (-MAG_FIELD[0], -MAG_FIELD[2], -MAG_FIELD[1]), tol=tol)

    imu.set_acc_cross_axis_coupling((0.0, 1.0, 0.0))
    scene.step()
    assert_allclose(imu.read().lin_acc, GRAVITY, tol=5e-6)

    scene.reset()
    box.set_dofs_velocity((1.0, 2.0, 3.0), dofs_idx_local=slice(3, None))
    scene.step()
    assert_allclose(imu.read_ground_truth().ang_vel, (1.0, 3.0, -2.0), tol=0.1)

    imu.set_quat_offset((1.0, 0.0, 0.0, 0.0))
    imu.set_acc_cross_axis_coupling((0.0, 0.0, 0.0))
    scene.reset()

    assert_allclose(imu.read().lin_acc, 0.0, tol=gs.EPS)  # biased, but cache hasn't been updated yet
    assert_allclose(imu_delayed.read().lin_acc, 0.0, tol=gs.EPS)
    assert_allclose(imu_noisy.read().ang_vel, 0.0, tol=gs.EPS)
    assert_allclose(imu_noisy.read().mag, 0.0, tol=gs.EPS)  # biased

    imu.set_bias(BIAS + 2 * (0.0, 0.0, 0.0))
    scene.step()
    assert_allclose(imu.read().lin_acc, BIAS, tol=tol)
    assert_allclose(imu.read().mag, MAG_FIELD, tol=tol)


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
def test_lone_sensor_resolution_and_offset_setter(show_viewer, n_envs):
    # IMU with only `*_resolution` set (no other noise/delay) returns acceleration components quantized to that
    # resolution. Being the only sensor of its class, this scene also covers the per-sensor metadata tables in the
    # configuration where a single entry spans the whole batch.
    GRAVITY = -10.0
    RESOLUTION = 0.5
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            gravity=(0.0, 0.0, GRAVITY),
        ),
        profiling_options=gs.options.ProfilingOptions(
            show_FPS=False,
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(
        morph=gs.morphs.Plane(),
    )
    box = scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(0.0, 0.0, 0.05),
        ),
    )
    imu = scene.add_sensor(
        gs.sensors.IMU(
            entity_idx=box.idx,
            acc_resolution=RESOLUTION,
        ),
    )
    scene.build(n_envs=n_envs)
    # Step until the contact force balances gravity: a free-falling sensor reads zero, which quantizes trivially
    for _ in range(8):
        scene.step()

    measured = imu.read().lin_acc
    remainders = (measured / RESOLUTION) - torch.round(measured / RESOLUTION)
    assert_allclose(remainders, 0.0, tol=gs.EPS)

    # At rest the accelerometer measures proper acceleration, i.e. gravity reversed and expressed in the sensor frame,
    # so rotating the sensor frame by 90 degrees about X moves it from +Z to +Y.
    assert_allclose(imu.read().lin_acc, (0.0, 0.0, -GRAVITY), tol=RESOLUTION)
    imu.set_quat_offset(gu.euler_to_quat((90.0, 0.0, 0.0)))
    scene.step()
    assert_allclose(imu.read().lin_acc, (0.0, -GRAVITY, 0.0), tol=RESOLUTION)

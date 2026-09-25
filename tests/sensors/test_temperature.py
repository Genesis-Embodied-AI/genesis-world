import pytest
import torch

import genesis as gs

from ..utils.assertions import assert_allclose, assert_equal


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
def test_grid_sensor_contact_and_reset(show_viewer, tol, n_envs):
    BOX_SIZE = 0.06
    PLATFORM_SIZE = 0.2
    FAR_POS = (PLATFORM_SIZE * 1.5, PLATFORM_SIZE * 1.5, PLATFORM_SIZE * 1.5)
    GRID_SIZE = (3, 3, 1)
    GRID_CENTER = (GRID_SIZE[0] // 2, GRID_SIZE[1] // 2, GRID_SIZE[2] // 2)
    BASE_TEMP = 22.0
    DIFF_TEMP = 0.5

    scene = gs.Scene(show_viewer=show_viewer)
    scene.add_entity(gs.morphs.Plane())
    platform = scene.add_entity(
        gs.morphs.Box(
            size=(PLATFORM_SIZE, PLATFORM_SIZE, PLATFORM_SIZE),
            pos=(0.0, 0.0, PLATFORM_SIZE / 2),
            fixed=True,
        )
    )
    hot_box = scene.add_entity(
        gs.morphs.Box(
            size=(BOX_SIZE, BOX_SIZE, BOX_SIZE),
            pos=(0.0, 0.0, PLATFORM_SIZE + BOX_SIZE / 2),
        )
    )
    cold_box = scene.add_entity(
        gs.morphs.Box(
            size=(BOX_SIZE, BOX_SIZE, BOX_SIZE),
            pos=FAR_POS,
        ),
    )
    TemperatureProperties = gs.sensors.TemperatureProperties
    sensor = scene.add_sensor(
        gs.sensors.TemperatureGrid(
            ambient_temperature=BASE_TEMP,
            convection_coefficient=0.0,
            simulate_all_link_temperatures=False,
            entity_idx=platform.idx,
            grid_size=GRID_SIZE,
            properties_dict={
                platform.base_link_idx: TemperatureProperties(
                    base_temperature=BASE_TEMP,
                    conductivity=400.0,
                    density=2000.0,
                    specific_heat=1.0,
                    emissivity=0.95,
                ),
                hot_box.base_link_idx: TemperatureProperties(
                    base_temperature=BASE_TEMP + 100.0,
                    conductivity=200.0,
                    density=3000.0,
                    specific_heat=1.0,
                    emissivity=0.1,
                ),
                # default properties; should apply to the cold box
                -1: TemperatureProperties(
                    base_temperature=BASE_TEMP - 100.0,
                    conductivity=150.0,
                    density=8000.0,
                    specific_heat=1.0,
                    emissivity=0.2,
                ),
            },
        )
    )
    scene.build(n_envs=n_envs)

    # After build, all cells at base temperature
    assert_allclose(sensor.read_ground_truth(), BASE_TEMP, tol=tol)

    # Hot box on center
    hot_box.set_pos((0.0, 0.0, PLATFORM_SIZE + BOX_SIZE / 2))
    for _ in range(50):
        scene.step()
    data = sensor.read()
    assert (data > BASE_TEMP + DIFF_TEMP).all(), f"Hot box should have heated the grid by at least {DIFF_TEMP} C"
    assert (data[..., GRID_CENTER[0], GRID_CENTER[1], GRID_CENTER[2]] > data[0, 0, 0]).all(), (
        "Center cell should be hotter than corner"
    )

    # Reset: exactly base temperature everywhere
    scene.reset()
    assert_allclose(sensor.read_ground_truth(), BASE_TEMP, tol=tol)

    # Cold box on center
    hot_box.set_pos(FAR_POS)
    cold_box.set_pos((0.0, 0.0, PLATFORM_SIZE + BOX_SIZE / 2))
    for _ in range(50):
        scene.step()
    data = sensor.read()
    assert (data < BASE_TEMP - DIFF_TEMP).all(), f"Cold box should have cooled the grid by at least {DIFF_TEMP} C"
    assert (data[..., GRID_CENTER[0], GRID_CENTER[1], GRID_CENTER[2]] < data[0, 0, 0]).all(), (
        "Center cell should be colder than corner"
    )

    # Move both away; step until grid returns near base
    hot_box.set_pos(FAR_POS)
    cold_box.set_pos((-FAR_POS[0], -FAR_POS[1], FAR_POS[2]))
    for _ in range(150):
        scene.step()
    data = sensor.read()
    assert_allclose(data, BASE_TEMP, tol=5e-2)


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
def test_grid_simulate_all_link_temps(show_viewer, tol, n_envs):
    BOX_SIZE = 0.06
    BASE_TEMP = 22.0
    HOT_BASE = BASE_TEMP + 80.0
    COLD_BASE = BASE_TEMP - 80.0

    scene = gs.Scene(show_viewer=show_viewer)
    scene.add_entity(gs.morphs.Plane())
    hot_box = scene.add_entity(
        gs.morphs.Box(
            size=(BOX_SIZE, BOX_SIZE, BOX_SIZE),
            pos=(0.0, 0.0, BOX_SIZE),
        )
    )
    cold_box = scene.add_entity(
        gs.morphs.Box(
            size=(BOX_SIZE, BOX_SIZE, BOX_SIZE),
            pos=(0.0, 0.0, BOX_SIZE * 2 + 0.001),
        )
    )
    hot_link_idx = hot_box.base_link_idx
    cold_link_idx = cold_box.base_link_idx
    sensor1 = scene.add_sensor(
        gs.sensors.TemperatureGrid(
            entity_idx=hot_box.idx,
            grid_size=(1, 1, 1),
            ambient_temperature=BASE_TEMP,
            properties_dict={
                hot_link_idx: gs.sensors.TemperatureProperties(
                    base_temperature=HOT_BASE,
                    conductivity=200.0,
                    density=2000.0,
                    specific_heat=1.0,
                    emissivity=0.1,
                ),
                cold_link_idx: gs.sensors.TemperatureProperties(
                    base_temperature=COLD_BASE,
                    conductivity=200.0,
                    density=2000.0,
                    specific_heat=1.0,
                    emissivity=0.1,
                ),
            },
            simulate_all_link_temperatures=True,
        )
    )
    sensor2 = scene.add_sensor(
        gs.sensors.TemperatureGrid(
            entity_idx=cold_box.idx,
            grid_size=(1, 1, 1),
        )
    )
    scene.build(n_envs=n_envs)

    link_temps = sensor1.link_temperatures  # (n_envs, n_links)

    assert_equal(link_temps[..., hot_link_idx], HOT_BASE)
    assert_equal(link_temps[..., cold_link_idx], COLD_BASE)

    cold_box.set_pos((0.0, 0.0, BOX_SIZE / 2))
    for _ in range(100):
        scene.step()

    assert_equal(sensor1.link_temperatures, sensor2.link_temperatures)

    assert (link_temps[..., hot_link_idx] < HOT_BASE - 1.0).all(), "Hot box link should have cooled"
    assert (link_temps[..., cold_link_idx] > COLD_BASE + 1.0).all(), "Cold box link should have heated up"

    assert_allclose(torch.mean(sensor1.read()), link_temps[..., hot_link_idx], tol=2e-2)
    assert_allclose(torch.mean(sensor2.read()), link_temps[..., cold_link_idx], tol=2e-2)

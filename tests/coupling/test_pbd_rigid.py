import math

import pytest
import torch

import genesis as gs


@pytest.mark.required
def test_particles_stay_on_entry_side_of_fixed_box(show_viewer):
    PLATE_ANGLE = 30.0
    PLATE_POS = (0.0, 0.0, 0.5)
    PLATE_THICKNESS = 0.01
    CUBE_SIZE = 0.06
    CUBE_GAP = 0.01
    # Per step, the slow cubes travel half the thickness of the plate and the fast one several times its thickness
    SLOW_SPEED = 0.5
    FAST_SPEED = 8.0

    plate_normal = (math.sin(math.radians(PLATE_ANGLE)), 0.0, math.cos(math.radians(PLATE_ANGLE)))
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.3, 1.5, 1.0),
            camera_lookat=PLATE_POS,
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(
        morph=gs.morphs.Box(
            pos=PLATE_POS,
            euler=(0.0, PLATE_ANGLE, 0.0),
            fixed=True,
            size=(0.5, 0.5, PLATE_THICKNESS),
        ),
    )
    # One cube faces each side of the plate, and a third one faces the plane of the plate beyond its edge
    cubes = []
    for side, y in ((1, -0.12), (-1, 0.12), (1, 0.35)):
        offset = side * (0.5 * PLATE_THICKNESS + CUBE_GAP + 0.5 * CUBE_SIZE)
        cubes.append(
            scene.add_entity(
                morph=gs.morphs.Box(
                    pos=(PLATE_POS[0] + offset * plate_normal[0], y, PLATE_POS[2] + offset * plate_normal[2]),
                    euler=(0.0, PLATE_ANGLE, 0.0),
                    size=(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE),
                ),
                material=gs.materials.PBD.Elastic(),
            )
        )
    scene.build(n_envs=2)
    cube_above, cube_below, cube_beside = cubes

    normal = torch.tensor(plate_normal, dtype=gs.tc_float, device=gs.device)
    cube_above.set_velocity(-SLOW_SPEED * normal)
    cube_below.set_velocity(FAST_SPEED * normal)
    cube_beside.set_velocity(-SLOW_SPEED * normal)

    center = torch.tensor(PLATE_POS, dtype=gs.tc_float, device=gs.device)
    for _ in range(20):
        scene.step()
        assert ((cube_above.get_particles_pos() - center) @ normal >= 0.5 * PLATE_THICKNESS).all()
        assert ((cube_below.get_particles_pos() - center) @ normal <= -0.5 * PLATE_THICKNESS).all()
    assert ((cube_beside.get_particles_pos() - center) @ normal < -0.5 * PLATE_THICKNESS).all()

import math

import pytest
import torch

import genesis as gs

from ..utils.assertions import assert_allclose


@pytest.mark.required
def test_particles_stay_on_entry_side_of_fixed_box(show_viewer):
    DT = 0.01
    PLATE_ANGLE = 30.0
    PLATE_POS = (0.0, 0.0, 0.5)
    PLATE_THICKNESS = 0.01
    PLATE_WIDTH = 0.5
    PARTICLE_SIZE = 0.01
    POSITION_TOL = 1e-6
    CUBE_SIZE = 0.06
    CUBE_GAP = 0.01
    # Per step, the slow cubes travel half the thickness of the plate and the fast one several times its thickness
    SLOW_SPEED = 0.5 * PLATE_THICKNESS / DT
    FAST_SPEED = 8.0 * PLATE_THICKNESS / DT

    plate_normal = (math.sin(math.radians(PLATE_ANGLE)), 0.0, math.cos(math.radians(PLATE_ANGLE)))
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=DT,
            substeps=1,
        ),
        pbd_options=gs.options.PBDOptions(
            particle_size=PARTICLE_SIZE,
        ),
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
            size=(PLATE_WIDTH, PLATE_WIDTH, PLATE_THICKNESS),
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
    # Half a radius on each axis touches the corner, while four fifths on each axis clears it
    radius = 0.5 * PARTICLE_SIZE
    edge = 0.5 * PLATE_WIDTH
    probe_height = 0.5 * PLATE_THICKNESS + CUBE_GAP
    probes = []
    for x, y in ((0.0, edge + 0.5 * radius), (-edge - 0.5 * radius, -edge - 0.5 * radius), (edge + 0.8 * radius,) * 2):
        probes.append(
            scene.add_entity(
                morph=gs.morphs.Box(
                    pos=(
                        PLATE_POS[0] + x * plate_normal[2] + probe_height * plate_normal[0],
                        y,
                        PLATE_POS[2] - x * plate_normal[0] + probe_height * plate_normal[2],
                    ),
                    size=(PARTICLE_SIZE,) * 3,
                ),
                material=gs.materials.PBD.Particle(sampler="regular"),
            )
        )
    scene.build(n_envs=2)
    cube_above, cube_below, cube_beside = cubes

    normal = torch.tensor(plate_normal, dtype=gs.tc_float, device=gs.device)
    cube_above.set_velocity(-SLOW_SPEED * normal)
    cube_below.set_velocity(FAST_SPEED * normal)
    cube_beside.set_velocity(-SLOW_SPEED * normal)

    center = torch.tensor(PLATE_POS, dtype=gs.tc_float, device=gs.device)
    for probe in probes:
        probe.set_velocity(-FAST_SPEED * normal)
    pos_probe_start = torch.stack([probe.get_particles_pos() for probe in probes])
    clearance = 0.5 * PLATE_THICKNESS + radius
    faces = ((cube_above, 1), (cube_below, -1))
    for cube, side in faces:
        assert (side * ((cube.get_particles_pos() - center) @ normal) > clearance + 0.25 * CUBE_GAP).all()

    distances = []
    for i_step in range(20):
        scene.step()
        distances.append(
            torch.stack([(side * ((cube.get_particles_pos() - center) @ normal)).amin(-1) for cube, side in faces])
        )
        if i_step == 0:
            pos_probe = torch.stack([probe.get_particles_pos() for probe in probes])
            # Only contact can deflect a probe along y, perpendicular to both gravity and its incident velocity
            dy_edge, dy_corner, dy_miss = (pos_probe - pos_probe_start)[..., 1]
            assert (dy_edge > radius).all()
            assert (dy_corner < -radius).all()
            assert_allclose(dy_miss, 0.0, atol=POSITION_TOL)
            assert ((pos_probe[2] - center) @ normal < -clearance).all()
    assert_allclose(torch.stack(distances).amin(0), clearance, atol=POSITION_TOL)
    assert ((cube_beside.get_particles_pos() - center) @ normal < -0.5 * PLATE_THICKNESS).all()

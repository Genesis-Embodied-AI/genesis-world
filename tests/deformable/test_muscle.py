import numpy as np
import pytest

import genesis as gs


@pytest.mark.parametrize("n_envs", [0, pytest.param(2, marks=pytest.mark.required)])
@pytest.mark.parametrize("muscle_material", [gs.materials.MPM.Muscle, gs.materials.FEM.Muscle])
def test_muscle(n_envs, muscle_material, show_viewer):
    mpm_options = gs.options.MPMOptions(
        lower_bound=(-1.0, -1.0, -0.2),
        upper_bound=(1.0, 1.0, 1.0),
    )
    plane_material = gs.materials.Rigid(
        coup_friction=5.0,
    )
    # The implicit MPM takes one large step, while the explicit FEM needs small substeps to stay stable
    if muscle_material is gs.materials.MPM.Muscle:
        sim_options = gs.options.SimOptions()
    else:
        sim_options = gs.options.SimOptions(
            dt=5e-4,
            substeps=10,
        )
    scene = gs.Scene(
        sim_options=sim_options,
        mpm_options=mpm_options,
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.0, -0.6, 0.5),
            camera_lookat=(0.0, 0.0, 0.05),
            camera_fov=40,
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(
        morph=gs.morphs.Plane(),
        material=plane_material,
    )
    # The body rests on the plane from the start: the plane holds MPM particles, which lie half a particle size inside
    # the surface of the body, by their centers at its coupling softness
    height = 0.05
    if muscle_material is gs.materials.MPM.Muscle:
        height -= 0.5 * mpm_options.particle_size - plane_material.coup_softness
    # The tetrahedralization is refined, surface included, so that every muscle group spans many tetrahedra. Its
    # well-shaped tetrahedra keep the explicit FEM integration stable.
    box = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.0, 0.0, height),
            size=(0.1, 0.5, 0.1),
            nobisect=False,
            maxvolume=2e-5,
        ),
        material=muscle_material(
            E=5e5,
            nu=0.45,
            rho=10000.0,
            n_groups=4,
        ),
    )
    scene.build(n_envs=n_envs)

    if isinstance(box.material, gs.materials.MPM.Muscle):
        pos = box.get_state().pos[0]
    else:  # isinstance(box.material, gs.materials.FEM.Muscle):
        pos = box.get_state().pos[0, box.get_el2v()].mean(1)

    # Four groups: upper or lower part of the body, fore or hind part
    pos_min, pos_max = pos.amin(0), pos.amax(0)
    is_lower = pos[:, 2] <= pos_min[2] + 0.3 * (pos_max[2] - pos_min[2])
    is_hind = pos[:, 1] >= pos_min[1] + 0.6 * (pos_max[1] - pos_min[1])
    box.set_muscle(muscle_group=2 * is_lower + is_hind, muscle_direction=(0.0, 1.0, 0.0))

    pos_init = box.get_state().pos.clone()
    for i in range(50):
        box.set_actuation([0.0, 0.0, 0.0, 0.5 + np.sin(0.005 * np.pi * i)])
        scene.step()
    pos = box.get_state().pos

    # Contracting the lower hind fibers along the body axis shortens the body, and bends its hind part upward off the
    # plane, which raises its center of mass
    length_init = pos_init[..., 1].amax(-1) - pos_init[..., 1].amin(-1)
    length = pos[..., 1].amax(-1) - pos[..., 1].amin(-1)
    assert (length < length_init - 5e-3).all()
    assert (pos[..., 2].mean(-1) > pos_init[..., 2].mean(-1) + 5e-4).all()
    assert (pos[..., 2] > -gs.EPS).all()

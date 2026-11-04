import numpy as np
import pytest

import genesis as gs


@pytest.mark.parametrize("n_envs", [0, pytest.param(2, marks=pytest.mark.required)])
@pytest.mark.parametrize(
    "options_cls, solver_options_cls",
    [
        (gs.entities.MPM, gs.options.MPMEntitySolverOptions),
        (gs.entities.FEM, gs.options.FEMEntitySolverOptions),
    ],
)
def test_muscle(n_envs, options_cls, solver_options_cls, show_viewer):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=5e-4,
            substeps=10,
        ),
        mpm_options=gs.options.MPMOptions(
            lower_bound=(-1.0, -1.0, -0.2),
            upper_bound=(1.0, 1.0, 1.0),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.0, -0.6, 0.5),
            camera_lookat=(0.0, 0.0, 0.05),
            camera_fov=40,
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(
        morph=gs.morphs.Plane(),
        material=gs.materials.Rigid(
            coup_friction=5.0,
        ),
    )
    # The tetrahedralization is refined, surface included, so that every muscle group spans many tetrahedra. Its
    # well-shaped tetrahedra keep the explicit FEM integration stable at the substep MPM needs.
    box = scene.add_entity(
        options=options_cls(
            morph=gs.morphs.Box(
                pos=(0.0, 0.0, 0.05),
                size=(0.1, 0.5, 0.1),
                nobisect=False,
                maxvolume=2e-5,
            ),
            material=gs.materials.Muscle(
                density=10000.0,
                youngs_modulus=5e5,
                poisson_ratio=0.45,
            ),
            solver_options=solver_options_cls(
                n_groups=4,
            ),
        ),
    )
    scene.build(n_envs=n_envs)

    if isinstance(box.options, gs.entities.MPM):
        pos = box.get_state().pos[0]
    else:
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

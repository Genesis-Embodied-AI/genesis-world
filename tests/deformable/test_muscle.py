import numpy as np
import pytest

import genesis as gs
from genesis.utils.misc import tensor_to_array


# This test cannot be flagged as required because it takes 500s + 250s to run on CPU.
# @pytest.mark.required
@pytest.mark.slow("cpu")  # cpu ~500s
@pytest.mark.parametrize("n_envs", [0, 2])
@pytest.mark.parametrize("muscle_material", [gs.materials.MPM.Muscle, gs.materials.FEM.Muscle])
def test_muscle(n_envs, muscle_material, show_viewer):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=5e-4,
            substeps=10,
            gravity=(0, 0, -9.8),
        ),
        mpm_options=gs.options.MPMOptions(
            lower_bound=(-1.0, -1.0, -0.2),
            upper_bound=(1.0, 1.0, 1.0),
        ),
        vis_options=gs.options.VisOptions(
            show_world_frame=True,
            visualize_mpm_boundary=True,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.5, 0, 0.8),
            camera_lookat=(0.0, 0.0, 0.0),
            camera_fov=40,
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )

    scene.add_entity(
        morph=gs.morphs.Plane(),
        material=gs.materials.Rigid(
            coup_friction=5.0,
        ),
    )
    worm = scene.add_entity(
        morph=gs.morphs.Mesh(
            file="meshes/worm/worm.obj",
            pos=(0.3, 0.3, 0.001),
            scale=0.1,
            euler=(90, 0, 0),
        ),
        material=muscle_material(
            E=5e5,
            nu=0.45,
            rho=10000.0,
            n_groups=4,
        ),
        surface=gs.surfaces.Default(
            diffuse_texture=gs.textures.ImageTexture(
                image_path="meshes/worm/bdy_Base_Color.png",
            ),
        ),
    )
    if n_envs > 0:
        scene.build(n_envs=n_envs)
    else:
        scene.build()

    if isinstance(worm.material, gs.materials.MPM.Muscle):
        pos = worm.get_state().pos[0]
        n_units = worm.n_particles
    else:  # isinstance(worm.material, gs.materials.FEM.Muscle):
        pos = worm.get_state().pos[0, worm.get_el2v()].mean(1)
        n_units = worm.n_elements

    pos = tensor_to_array(pos)
    pos_max, pos_min = pos.max(0), pos.min(0)
    pos_range = pos_max - pos_min
    lu_thresh, fh_thresh = 0.3, 0.6
    muscle_group = np.zeros((n_units,), dtype=int)
    mask_upper = pos[:, 2] > (pos_min[2] + pos_range[2] * lu_thresh)
    mask_fore = pos[:, 1] < (pos_min[1] + pos_range[1] * fh_thresh)
    muscle_group[mask_upper & mask_fore] = 0  # upper fore body
    muscle_group[mask_upper & ~mask_fore] = 1  # upper hind body
    muscle_group[~mask_upper & mask_fore] = 2  # lower fore body
    muscle_group[~mask_upper & ~mask_fore] = 3  # lower hind body

    muscle_direction = np.array([[0.0, 1.0, 0.0]] * n_units, dtype=gs.np_float)

    worm.set_muscle(
        muscle_group=muscle_group,
        muscle_direction=muscle_direction,
    )

    scene.reset()
    for i in range(200):
        actuation = np.array([0.0, 0.0, 0.0, 1.0 * (0.5 + np.sin(0.005 * np.pi * i))])
        if n_envs > 1:
            actuation = np.tile(actuation, (n_envs, 1))
        worm.set_actuation(actuation)
        scene.step()

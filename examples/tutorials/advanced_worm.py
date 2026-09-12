import math
import os
import torch
import genesis as gs


gs.init(backend=gs.cpu, precision="32", logging_level="info")

scene = gs.Scene(
    sim_options=gs.options.SimOptions(
        dt=5e-4,
        gravity=(0, 0, -9.8),
    ),
    rigid_options=gs.options.RigidOptions(
        dt=5e-5,
    ),
    mpm_options=gs.options.MPMOptions(
        dt=5e-5,
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
    material=gs.materials.MPM.Muscle(
        E=5e5,
        nu=0.45,
        rho=10000.0,
        model="neohooken",
        n_groups=4,
    ),
    surface=gs.surfaces.Default(
        diffuse_texture=gs.textures.ImageTexture(
            image_path="meshes/worm/bdy_Base_Color.png",
        ),
    ),
)

scene.build(n_envs=3)


if isinstance(worm.material, gs.materials.MPM.Muscle):
    pos = worm.get_state().pos[0]
    n_units = worm.n_particles
elif isinstance(worm.material, gs.materials.FEM.Muscle):
    pos = worm.get_state().pos[0, worm.get_el2v()].mean(dim=1)
    n_units = worm.n_elements
else:
    raise NotImplementedError

pos_max, pos_min = pos.max(dim=0).values, pos.min(dim=0).values
pos_range = pos_max - pos_min

lu_thr, fh_thr = 0.3, 0.6
muscle_group = torch.zeros((n_units,), dtype=gs.tc_int, device=gs.device)
mask_upper = pos[:, 2] > (pos_min[2] + pos_range[2] * lu_thr)
mask_fore = pos[:, 1] < (pos_min[1] + pos_range[1] * fh_thr)
muscle_group[mask_upper & mask_fore] = 0  # upper fore body
muscle_group[mask_upper & ~mask_fore] = 1  # upper hind body
muscle_group[~mask_upper & mask_fore] = 2  # lower fore body
muscle_group[~mask_upper & ~mask_fore] = 3  # lower hind body

muscle_direction = (0.0, 1.0, 0.0)

worm.set_muscle(
    muscle_group=muscle_group,
    muscle_direction=muscle_direction,
)


horizon = 1000 if "PYTEST_VERSION" not in os.environ else 5
scene.reset()
for i in range(horizon):
    actu = (0.0, 0.0, 0.0, 1.0 * (0.5 + math.sin(0.005 * math.pi * i)))
    worm.set_actuation(actu)
    scene.step()

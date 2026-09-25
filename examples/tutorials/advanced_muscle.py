import os
import numpy as np
import genesis as gs


gs.init(backend=gs.cpu, precision="32", logging_level="info")

dt = 5e-4
substep_dt = 5e-5
scene = gs.Scene(
    sim_options=gs.options.SimOptions(
        dt=dt,
        gravity=(0, 0, 0),
    ),
    rigid_options=gs.options.RigidOptions(
        dt=substep_dt,
    ),
    mpm_options=gs.options.MPMOptions(
        dt=substep_dt,
        lower_bound=(-1.0, -1.0, -0.2),
        upper_bound=(1.0, 1.0, 1.0),
    ),
    fem_options=gs.options.FEMOptions(
        dt=substep_dt,
        damping=45.0,
    ),
    vis_options=gs.options.VisOptions(
        show_world_frame=False,
    ),
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(1.5, 0, 0.8),
        camera_lookat=(0.0, 0.0, 0.0),
        camera_fov=40,
    ),
    show_viewer=True,
)

scene.add_entity(
    morph=gs.morphs.Plane(),
)

E, nu = 3.0e4, 0.45
rho = 1000.0

robot_mpm = scene.add_entity(
    morph=gs.morphs.Sphere(
        pos=(0.5, 0.2, 0.3),
        radius=0.1,
    ),
    material=gs.materials.MPM.Muscle(
        E=E,
        nu=nu,
        rho=rho,
        model="neohooken",
    ),
)

robot_fem = scene.add_entity(
    morph=gs.morphs.Sphere(
        pos=(0.5, -0.2, 0.3),
        radius=0.1,
    ),
    material=gs.materials.FEM.Muscle(
        E=E,
        nu=nu,
        rho=rho,
        model="stable_neohookean",
    ),
)

scene.build(n_envs=0)

horizon = 1000 if "PYTEST_VERSION" not in os.environ else 5
scene.reset()
for i in range(horizon):
    actu = [0.2 * (0.5 + np.sin(0.01 * np.pi * i))]
    robot_mpm.set_actuation(actu)
    robot_fem.set_actuation(actu)
    scene.step()

import torch

import genesis as gs

gs.init(backend=gs.cpu)

scene = gs.Scene(
    sim_options=gs.options.SimOptions(
        dt=0.01,
    ),
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(3.5, -1.0, 2.5),
        camera_lookat=(0.0, 0.0, 0.5),
        camera_fov=40,
    ),
    show_viewer=False,
)

plane = scene.add_entity(
    gs.morphs.Plane(),
)

franka = scene.add_entity(
    gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
)

# create 20 parallel environments
B = 20
scene.build(n_envs=B, env_spacing=(1.0, 1.0))

# control all the robots
franka.control_dofs_position(
    torch.tile(torch.tensor([0, 0, 0, -1.0, 0, 1.0, 0, 0.02, 0.02], device=gs.device), (B, 1)),
)

# 'envs_idx' narrows a command to a subset of the environments, leaving the rest on their previous target
franka.control_dofs_position(
    torch.zeros(3, 9, device=gs.device),
    envs_idx=torch.tensor([1, 5, 7], device=gs.device),
)

for i in range(1000):
    scene.step()

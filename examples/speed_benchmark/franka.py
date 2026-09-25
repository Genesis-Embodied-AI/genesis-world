import torch

import genesis as gs

# The benchmark measures throughput at 30000 parallel environments, which needs a GPU.
gs.init(backend=gs.gpu, performance_mode=True)

scene = gs.Scene(
    sim_options=gs.options.SimOptions(
        dt=0.01,
    ),
    viewer_options=gs.options.ViewerOptions(
        res=(1920, 1080),
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


# create 30000 parallel environments
B = 30000
scene.build(n_envs=B, env_spacing=(1.0, 1.0))

# control all the robots
# with the following control: 43M FPS
# without the following control (arm in collision with the floor): 32M FPS
franka.control_dofs_position(
    torch.tile(torch.tensor([0, 0, 0, -1.0, 0, 1.0, 0, 0.02, 0.02], device=gs.device), (B, 1)),
)

for i in range(1000):
    scene.step()

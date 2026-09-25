import os
import genesis as gs

gs.init(backend=gs.cpu)

scene = gs.Scene(
    vis_options=gs.options.VisOptions(
        rendered_envs_idx=list(range(10, 15)),  # render the 11th to 15th environments
    ),
    viewer_options=gs.options.ViewerOptions(
        res=(1280, 960),
        camera_pos=(3.5, 0.0, 2.5),
        camera_lookat=(0.0, 0.0, 0.5),
        camera_fov=50,
    ),
    show_viewer=True,
)

plane = scene.add_entity(
    gs.morphs.Plane(),
)
franka = scene.add_entity(
    gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
)


scene.build(n_envs=20, env_spacing=(1.0, 1.0))

horizon = 1000 if "PYTEST_VERSION" not in os.environ else 5
for i in range(horizon):
    scene.step()

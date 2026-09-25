import os
import math
import genesis as gs

gs.init(backend=gs.cpu)

scene = gs.Scene(
    vis_options=gs.options.VisOptions(
        show_world_frame=True,
        world_frame_size=1.0,
        show_link_frame=False,
        show_cameras=False,
        plane_reflection=True,
        ambient_light=(0.1, 0.1, 0.1),
    ),
    viewer_options=gs.options.ViewerOptions(
        res=(1280, 960) if "PYTEST_VERSION" not in os.environ else (64, 64),
        camera_pos=(3.5, 0.0, 2.5),
        camera_lookat=(0.0, 0.0, 0.5),
        camera_fov=40,
    ),
    renderer=gs.renderers.Rasterizer(),
    show_viewer=True,
)

plane = scene.add_entity(
    gs.morphs.Plane(),
)
franka = scene.add_entity(
    gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
)

cam = scene.add_camera(
    res=(640, 480),
    pos=(3.5, 0.0, 2.5),
    lookat=(0, 0, 0.5),
    fov=30,
    GUI="PYTEST_VERSION" not in os.environ,
)

scene.build()

rgb, depth, segmentation, normal = cam.render(rgb=True, depth=True, segmentation=True, normal=True)

cam.start_recording(save_to_filename="out/visualization.mp4", fps=60)

horizon = 120 if "PYTEST_VERSION" not in os.environ else 1
for i in range(horizon):
    cam.set_pose(
        pos=(3.0 * math.sin(i / 60), 3.0 * math.cos(i / 60), 2.5),
        lookat=(0, 0, 0.5),
    )
    scene.step()
cam.stop_recording()

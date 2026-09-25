import os
import threading

import genesis as gs


def run_sim(scene):
    for _ in range(200):
        scene.step(refresh_visualizer=False)
        if "PYTEST_VERSION" in os.environ:
            break


def main():
    gs.init(backend=gs.cpu)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
        ),
        rigid_options=gs.options.RigidOptions(
            gravity=(0.0, 0.0, -10.0),
        ),
        viewer_options=gs.options.ViewerOptions(
            run_in_thread=False,
            camera_pos=(3.5, 0.0, 2.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=40,
        ),
        show_viewer=True,
        show_FPS=True,
    )

    plane = scene.add_entity(gs.morphs.Plane())
    r0 = scene.add_entity(
        gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
    )

    scene.build()

    threading.Thread(target=run_sim, args=(scene,)).start()
    if "PYTEST_VERSION" not in os.environ:
        scene.viewer.run()


if __name__ == "__main__":
    main()

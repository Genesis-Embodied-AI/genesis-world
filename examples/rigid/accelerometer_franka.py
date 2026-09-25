import argparse
import genesis as gs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", help="Show visualization GUI")
    args = parser.parse_args()

    gs.init(backend=gs.cpu)
    sim_dt = 0.01
    viewer_options = gs.options.ViewerOptions(
        camera_pos=(0, -3.5, 2.5),
        camera_lookat=(0.0, 0.0, 0.5),
        camera_fov=40,
    )

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=sim_dt,
        ),
        viewer_options=viewer_options,
        show_viewer=args.vis,
    )

    plane = scene.add_entity(
        gs.morphs.Plane(),
    )
    franka = scene.add_entity(
        gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
    )
    scene.build()

    last_links_vel = None
    for _ in range(500):
        scene.step()

        links_acc = franka.get_links_acc()
        links_pos = franka.get_links_pos()
        # 'get_links_acc' reports the classical acceleration at the link origin, so the finite difference it is
        # compared against must differentiate the velocity expressed at that same reference point.
        links_vel = franka.get_links_vel()
        scene.clear_debug_objects()

        for link_idx in range(len(links_acc)):
            scene.draw_debug_arrow(
                pos=links_pos[link_idx].tolist(),
                vec=links_acc[link_idx].tolist(),
            )
            if last_links_vel is not None:
                scene.draw_debug_arrow(
                    pos=links_pos[link_idx].tolist(),
                    vec=((links_vel[link_idx] - last_links_vel[link_idx]) / sim_dt).tolist(),
                    color=(0, 1, 0),
                )

        last_links_vel = links_vel


if __name__ == "__main__":
    main()

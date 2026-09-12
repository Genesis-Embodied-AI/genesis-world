import math
from typing import TYPE_CHECKING

import genesis as gs
from genesis.vis.camera import Camera

from quadcopter_controller import DronePIDController

if TYPE_CHECKING:
    from genesis.engine.entities import DroneEntity


base_rpm = 14468.429183500699
min_rpm = 0.9 * base_rpm
max_rpm = 1.5 * base_rpm


def hover(drone: "DroneEntity"):
    drone.set_propellers_rpm([base_rpm, base_rpm, base_rpm, base_rpm])


def clamp(rpm):
    return max(min_rpm, min(int(rpm), max_rpm))


def fly_to_point(target, controller: "DronePIDController", scene: gs.Scene, cam: Camera):
    drone = controller.drone
    step = 0
    x = target[0] - drone.get_pos()[0]
    y = target[1] - drone.get_pos()[1]
    z = target[2] - drone.get_pos()[2]

    distance = math.sqrt(x**2 + y**2 + z**2)

    while distance > 0.1 and step < 1000:
        [M1, M2, M3, M4] = controller.update(target)
        M1 = clamp(M1)
        M2 = clamp(M2)
        M3 = clamp(M3)
        M4 = clamp(M4)
        drone.set_propellers_rpm([M1, M2, M3, M4])
        drone_pos = drone.get_pos()
        drone_pos = drone_pos.cpu().numpy()
        x = drone_pos[0]
        y = drone_pos[1]
        z = drone_pos[2]
        # Aiming the camera before stepping, since the step is what captures the frame
        cam.set_pose(lookat=(x, y, z))
        scene.step()
        x = target[0] - x
        y = target[1] - y
        z = target[2] - z
        distance = math.sqrt(x**2 + y**2 + z**2)
        step += 1


def main():
    gs.init(backend=gs.cpu)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
        ),
        show_viewer=False,
    )

    plane = scene.add_entity(
        morph=gs.morphs.Plane(),
    )

    drone = scene.add_entity(
        morph=gs.morphs.Drone(
            file="urdf/drones/cf2x.urdf",
            pos=(0, 0, 0.2),
        )
    )

    # parameters are tuned such that the
    # drone can fly, not optimized
    pid_params = [
        [2.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [20.0, 0.0, 20.0],
        [20.0, 0.0, 20.0],
        [25.0, 0.0, 20.0],
        [10.0, 0.0, 1.0],
        [10.0, 0.0, 1.0],
        [2.0, 0.0, 0.2],
    ]

    controller = DronePIDController(drone=drone, dt=0.01, base_rpm=base_rpm, pid_params=pid_params)

    cam = scene.add_camera(
        pos=(1, 1, 1),
        lookat=drone.morph.pos,
        GUI=False,
        res=(640, 480),
        fov=30,
    )

    scene.build()

    cam.start_recording(save_to_filename="out/fly_route.mp4")

    points = [(1, 1, 2), (-1, 2, 1), (0, 0, 0.5)]

    for point in points:
        fly_to_point(point, controller, scene, cam)

    cam.stop_recording()


if __name__ == "__main__":
    main()

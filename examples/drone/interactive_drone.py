import os

import numpy as np

import genesis as gs
from genesis.vis.keybindings import Key, KeyAction, Keybind


class DroneController:
    def __init__(self):
        self.thrust = 14475.8  # Base RPM for constant hover
        self.rotation_delta = 100.0  # Differential RPM for rotation
        self.thrust_delta = 10.0  # Amount to change thrust by when accelerating/decelerating
        self.cur_dir = np.array([0.0, 0.0, 0.0, 0.0])  # rotor directions

    def update_rpms(self):
        """Compute RPMs based on current direction and thrust"""
        clipped_dir = np.clip(self.cur_dir, -1.0, 1.0)
        rpms = self.thrust + clipped_dir * self.rotation_delta
        return np.clip(rpms, 0, 25000)

    def add_direction(self, direction: np.ndarray):
        """Add direction vector (on key press)"""
        self.cur_dir += direction

    def accelerate(self):
        """Increase base thrust"""
        self.thrust = min(self.thrust + self.thrust_delta, 25000)

    def decelerate(self):
        """Decrease base thrust"""
        self.thrust = max(self.thrust - self.thrust_delta, 0)


def main():
    # Initialize Genesis
    gs.init(backend=gs.cpu)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
            gravity=(0, 0, -9.81),
        ),
        vis_options=gs.options.VisOptions(
            show_world_frame=False,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.0, -2.0, 1.0),
            camera_lookat=(0.0, 0.0, 0.3),
            camera_fov=45,
        ),
        show_viewer=True,
        show_FPS=False,
    )

    # Add entities
    scene.add_entity(gs.morphs.Plane())
    drone = scene.add_entity(
        morph=gs.morphs.Drone(
            file="urdf/drones/cf2x.urdf",
            pos=(0.0, 0, 0.5),  # Start a bit higher
        ),
    )

    scene.viewer.follow_entity(drone)

    # Initialize controller
    controller = DroneController()

    scene.build()

    # Register keybindings
    def direction_keybinds(name: str, key: Key, direction: tuple[float, float, float, float]):
        """Helper to create press/release keybinds for a direction"""
        dir_arr = np.array(direction)
        return [
            Keybind(
                name=f"{name}_hold",
                key=key,
                key_action=KeyAction.HOLD,
                callback=controller.add_direction,
                args=(dir_arr,),
            ),
            Keybind(
                name=f"{name}_release",
                key=key,
                key_action=KeyAction.RELEASE,
                callback=controller.add_direction,
                args=(-dir_arr,),
            ),
        ]

    is_running = True

    def stop():
        nonlocal is_running
        is_running = False

    scene.viewer.register_keybinds(
        *direction_keybinds("move_forward", Key.UP, (1.0, 1.0, -1.0, -1.0)),
        *direction_keybinds("move_backward", Key.DOWN, (-1.0, -1.0, 1.0, 1.0)),
        *direction_keybinds("move_left", Key.LEFT, (-1.0, 1.0, -1.0, 1.0)),
        *direction_keybinds("move_right", Key.RIGHT, (1.0, -1.0, 1.0, -1.0)),
        Keybind("accelerate", Key.SPACE, KeyAction.HOLD, callback=controller.accelerate),
        Keybind("decelerate", Key.LSHIFT, KeyAction.HOLD, callback=controller.decelerate),
        Keybind("quit", Key.ESCAPE, KeyAction.RELEASE, callback=stop),
    )

    # Print control instructions
    print("\nDrone Controls:")
    print("Up - Move Forward (North)")
    print("Down - Move Backward (South)")
    print("Left - Move Left (West)")
    print("Right - Move Right (East)")
    print("space - Increase RPM")
    print("shift - Decrease RPM")

    # Run simulation
    try:
        while is_running:
            # Update and apply RPMs based on current direction
            rpms = controller.update_rpms()
            drone.set_propellers_rpm(rpms)

            # Step simulation
            scene.step()

            if "PYTEST_VERSION" in os.environ:
                break
    except KeyboardInterrupt:
        gs.logger.info("Simulation interrupted, exiting.")
    finally:
        gs.logger.info("Simulation finished.")


if __name__ == "__main__":
    main()

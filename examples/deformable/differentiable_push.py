import argparse
import os

import torch

import genesis as gs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", help="Show visualization GUI")
    args = parser.parse_args()

    gs.init(backend=gs.cpu, precision="32", logging_level="warning")

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=2e-3,
            substeps=10,
            requires_grad=True,
        ),
        mpm_options=gs.options.MPMOptions(
            lower_bound=(0.0, -1.0, 0.0),
            upper_bound=(1.0, 1.0, 0.55),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(2.5, -0.15, 2.42),
            camera_lookat=(0.5, 0.5, 0.1),
        ),
        show_viewer=args.vis,
    )

    plane = scene.add_entity(
        morph=gs.morphs.URDF(
            file="urdf/plane/plane.urdf",
            fixed=True,
        )
    )
    stick = scene.add_entity(
        morph=gs.morphs.Mesh(
            file="meshes/stirrer.obj",
            scale=1.0,
            pos=(0.5, 0.5, 0.05),
            euler=(90.0, 0.0, 0.0),
        ),
        # Heavy enough for the explicit coupling impulse of a high friction to stay small next to its momentum
        material=gs.materials.Rigid(
            rho=1e6,
            coup_friction=8.0,
            gravity_compensation=1.0,
        ),
        surface=gs.surfaces.Default(
            color=(0.85, 0.2, 0.15, 1.0),
        ),
    )
    obj1 = scene.add_entity(
        material=gs.materials.MPM.Elastic(
            rho=500,
        ),
        morph=gs.morphs.Box(
            lower=(0.2, 0.1, 0.05),
            upper=(0.4, 0.3, 0.15),
        ),
        surface=gs.surfaces.Default(
            color=(0.9, 0.9, 0.9, 1.0),
        ),
        vis_mode="particle",
    )
    obj2 = scene.add_entity(
        material=gs.materials.MPM.Elastic(
            rho=500,
        ),
        morph=gs.morphs.Mesh(
            file="meshes/duck.obj",
            pos=(0.4, 0.55, 0.056),
            scale=0.07,
            euler=(90.0, 0.0, 90.0),
        ),
        surface=gs.surfaces.Default(
            color=(0.9, 0.8, 0.2, 1.0),
        ),
        vis_mode="particle",
    )

    cam_0 = scene.add_camera(
        pos=(1.5, 0.5, 2.42),
        lookat=(0.5, 0.5, 0.1),
        fov=30,
        GUI=True,
    )
    cam_1 = scene.add_camera(
        pos=(-3.0, 1.5, 2.0),
        lookat=(0.5, 0.5, 0.1),
        fov=30,
        GUI=True,
    )

    scene.build(n_envs=2)

    # Two independent forward/backward passes, each restarting from the same initial state.
    horizon = 150 if "PYTEST_VERSION" not in os.environ else 5
    v_list = [
        torch.tensor([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]], dtype=gs.tc_float, device=gs.device, requires_grad=True)
        for _ in range(horizon)
    ]
    for _ in range(2):
        scene.reset()
        init_pos = torch.tensor(
            [[0.3, 0.19, 0.1], [0.3, 0.19, 0.5]], dtype=gs.tc_float, device=gs.device, requires_grad=True
        )
        stick.set_pos(init_pos)
        pos_obj1_init = torch.tensor([0.3, 0.3, 0.1], dtype=gs.tc_float, device=gs.device, requires_grad=True)
        obj1.set_position(pos_obj1_init)
        v_obj1_init = torch.tensor([0.0, -1.0, 0.0], dtype=gs.tc_float, device=gs.device, requires_grad=True)
        obj1.set_velocity(v_obj1_init)

        # forward pass
        print("forward")
        timer = gs.tools.Timer()
        loss = 0.0
        for i, v_i in enumerate(v_list):
            stick.set_dofs_velocity(v_i, dofs_idx_local=slice(0, 3))

            scene.step()

            # you can use the getters of a solver
            if i == 25:
                # compute loss
                goal = torch.tensor([0.5, 0.8, 0.05], dtype=gs.tc_float, device=gs.device)
                particles_pos = scene.mpm_solver.get_particles_pos()
                particles_active = scene.mpm_solver.get_particles_active()
                loss += torch.pow(particles_pos[particles_active] - goal, 2).sum()

            # you can also use the getters of an entity
            if i == horizon - 1:
                # compute loss
                goal = torch.tensor([0.5, 0.8, 0.05], dtype=gs.tc_float, device=gs.device)
                loss += torch.pow(obj1.get_particles_pos() - goal, 2).sum()

        timer.stamp("forward took: ")
        # backward pass
        print("backward")
        loss.backward()  # this lets gradient flow all the way back to tensor input
        timer.stamp("backward took: ")
        for v_i in v_list:
            print(v_i.grad)
            v_i.grad = None
        init_pos.grad = None
        print(loss.item())


if __name__ == "__main__":
    main()

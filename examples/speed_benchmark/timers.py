import argparse
import os
from contextlib import nullcontext

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import plotext as plt
import torch

import genesis as gs
from genesis.utils.misc import qd_to_torch

torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", type=int, default=2, choices=(0, 1, 2), help="Control noise: 0 none, 1 uniform, 2 per-environment"
    )
    parser.add_argument("--profiling", action="store_true", help="Enable the Genesis profiler")
    args = parser.parse_args()

    # The benchmark measures per-environment timings across 128 environments, which needs a GPU.
    gs.init(backend=gs.gpu, precision="32", performance_mode=True, seed=0, logging_level="warning")

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.02,
            substeps=2,
        ),
        rigid_options=gs.options.RigidOptions(
            enable_self_collision=False,
            max_collision_pairs=30,
            iterations=1,
            ls_iterations=1,
        ),
        show_viewer=False,
        show_FPS=True,
    )
    scene.add_entity(
        gs.morphs.Plane(),
    )
    robot = scene.add_entity(
        gs.morphs.URDF(
            file="urdf/go2/urdf/go2.urdf",
        ),
        vis_mode="collision",
    )
    scene.build(n_envs=128)

    ctrl_pos_0 = torch.tensor(
        [0.0, 0.0, 0.0, 0.0, 0.8, 0.8, 1.0, 1.0, -1.5, -1.5, -1.5, -1.5],
        dtype=gs.tc_float,
        device=gs.device,
    )
    init_qpos = torch.tensor(
        [0.0, 0.0, 0.42, 1.0, 0.0, 0.0, 0.0, *ctrl_pos_0],
        dtype=gs.tc_float,
        device=gs.device,
    )
    robot.set_qpos(init_qpos)
    robot.control_dofs_position(ctrl_pos_0, dofs_idx_local=slice(6, 18))

    timers = qd_to_torch(scene.rigid_solver.constraint_solver.constraint_state.timers)
    stats = torch.zeros((3, *timers.shape), dtype=gs.tc_float, device=gs.device)

    TIMER_LABELS = ("func_solve",)
    with (
        torch.profiler.profile(
            on_trace_ready=torch.profiler.tensorboard_trace_handler("./benchmark"),
            schedule=torch.profiler.schedule(wait=0, warmup=0, active=1),
            record_shapes=False,
            profile_memory=False,
            with_stack=True,
            with_flops=False,
        )
        if args.profiling
        else nullcontext()
    ):
        for step in range(500):
            scene.step()
            noise = (args.mode > 0) * torch.rand(
                (*((scene.n_envs,) if (args.mode > 1) else ()), robot.n_dofs - 6),
                dtype=gs.tc_float,
                device=gs.device,
            )
            robot.control_dofs_position(ctrl_pos_0 + 0.3 * noise, slice(6, 18))
            if not args.profiling:
                if not gs.use_zerocopy:
                    timers = qd_to_torch(scene.rigid_solver.constraint_solver.constraint_state.timers)
                stats[0] = timers
                stats[1] = stats[1] * (step / (step + 1)) + timers / (step + 1)
                stats[2] = stats[2] * (step / (step + 1)) + timers.sort(descending=False).values / (step + 1)
                if (step + 1) % 500 == 0:
                    plt.clf()
                    plt.plot_size(260, 25)
                    plt.subplots(1, 3)
                    for i, mode in enumerate(("Last", "Average ordered", "Average sorted")):
                        for data, label in zip(stats[i], TIMER_LABELS):
                            plt.subplot(1, i + 1).plot(data.cpu().numpy(), label=label)
                        plt.title(f"[mode {args.mode}] {mode} per-env timings")
                    plt.show()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Data-parallel training across GPUs with PyTorch DistributedDataParallel.

Each rank owns a full scene with its own batch of environments, so the effective batch is the per-rank
``--num-envs`` times the number of ranks: adding GPUs lowers gradient noise rather than changing any scene.

Single machine, 2 GPUs:
    torchrun --standalone --nnodes=1 --nproc_per_node=2 examples/rigid/ddp_multi_gpu.py

Launched without torchrun, the script trains as a single rank, on CPU when there is no GPU.
"""

import argparse
import os

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

import genesis as gs


class TinyMLP(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, act_dim),
        )

    def forward(self, x):
        return self.net(x.float())


def run_worker(args: argparse.Namespace) -> None:
    # setup
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)
    os.environ["QD_VISIBLE_DEVICE"] = str(local_rank)
    # FIXME: Forcing rendering device is not working reliably on all machines
    # os.environ["EGL_DEVICE_ID"] = str(local_rank)

    # Each rank simulates its own batch of environments on its GPU, on CPU when there is none.
    gs.init(backend=gs.gpu, seed=local_rank)

    # sim
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3.5, 0.0, 2.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=40,
        ),
        show_viewer=args.vis and local_rank == 0,
        show_FPS=False,
    )
    scene.add_entity(gs.morphs.Plane())
    scene.add_entity(
        gs.morphs.MJCF(
            file="xml/franka_emika_panda/panda.xml",
        ),
        visualize_contact=True,
    )
    scene.build(n_envs=args.num_envs)

    # model, trained on the CUDA GPU of the rank, otherwise on CPU since gradients are only synchronized across CUDA
    # (NCCL) or CPU (Gloo) tensors, whatever device the scene runs on
    if torch.cuda.is_available():
        gpu_id = 0
        torch.cuda.set_device(gpu_id)
        backend, device = "nccl", torch.device("cuda", gpu_id)
    else:
        backend, device = "gloo", torch.device("cpu")
    if "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend=backend, init_method="env://")
    else:
        # Launched without torchrun, the script trains as a single rank whose group needs no network rendezvous
        dist.init_process_group(backend=backend, store=dist.HashStore(), rank=0, world_size=1)

    rigid = scene.sim.rigid_solver
    qpos = rigid.get_qpos().to(device)
    obs_dim = qpos.shape[1]
    act_dim = 1
    model = TinyMLP(obs_dim, act_dim).to(device)
    model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None)
    optim = torch.optim.Adam(model.parameters(), lr=3e-4)

    # train loop
    for step in range(args.steps):
        scene.step()
        qpos = rigid.get_qpos().to(device)

        obs = qpos + torch.randn_like(qpos)

        logits = model(obs)
        target = qpos.sum(dim=1, keepdim=True)
        loss = torch.nn.functional.mse_loss(logits, target)

        optim.zero_grad(set_to_none=True)
        loss.backward()  # DDP handles all-reduce, gradients are averaged
        optim.step()

        if local_rank == 0 and step % 100 == 0:
            print(f"[{step:04d}/{args.steps}] loss = {loss.item():.6f}")

    # cleanup
    dist.barrier()  # sync all ranks before shutting down the process group
    dist.destroy_process_group()
    gs.destroy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", help="Open the viewer on rank 0")
    is_test = "PYTEST_VERSION" in os.environ
    parser.add_argument(
        "-b", "--num-envs", type=int, default=2048 if not is_test else 2, help="Number of parallel environments"
    )
    parser.add_argument("-s", "--steps", type=int, default=1000 if not is_test else 5, help="Number of training steps")
    run_worker(parser.parse_args())


if __name__ == "__main__":
    main()

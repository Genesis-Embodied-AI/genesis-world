import argparse
import multiprocessing
import os
import sys

import torch

import genesis as gs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", help="Show visualization GUI")
    args = parser.parse_args()

    # Selecting the CUDA device to simulate on is the whole point of this example.
    gs.init(backend=gs.gpu, logger_verbose_time=True)

    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3.5, 0.0, 2.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=40,
        ),
        show_viewer=args.vis,
        show_FPS=False,
    )

    plane = scene.add_entity(
        gs.morphs.Plane(),
    )
    franka = scene.add_entity(
        gs.morphs.MJCF(
            file="xml/franka_emika_panda/panda.xml",
        ),
        visualize_contact=True,
    )

    scene.build()
    for i in range(1000 if "PYTEST_VERSION" not in os.environ else 5):
        scene.step()


def run(gpu_id, func):
    # Set environment args
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["QD_VISIBLE_DEVICE"] = str(gpu_id)
    os.environ["EGL_DEVICE_ID"] = str(gpu_id)
    # main script
    func()


if __name__ == "__main__":
    # One simulation per GPU, a single one when there is no GPU
    num_gpus = max(torch.cuda.device_count(), 1)

    processes = []
    for i in range(num_gpus):
        p = multiprocessing.Process(target=run, args=(i, main))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
    if any(p.exitcode != 0 for p in processes):
        sys.exit(1)

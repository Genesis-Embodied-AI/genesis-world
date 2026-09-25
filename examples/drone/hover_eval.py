import argparse
import os
import pickle
from importlib import metadata

import torch

try:
    if int(metadata.version("rsl-rl-lib").split(".")[0]) < 5:
        raise ImportError
except (metadata.PackageNotFoundError, ImportError, ValueError) as e:
    raise ImportError("Please install 'rsl-rl-lib>=5.0.0'.") from e
from rsl_rl.runners import OnPolicyRunner

import genesis as gs

from hover_env import HoverEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e",
        "--exp-name",
        type=str,
        default="drone-hovering",
        help="Experiment name; also the log directory under logs/",
    )
    parser.add_argument("--ckpt", type=int, default=300, help="Checkpoint iteration to load")
    parser.add_argument("-r", "--record", action="store_true", help="Record the rollout to video")
    args = parser.parse_args()

    gs.init(backend=gs.cpu)

    log_dir = f"logs/{args.exp_name}"
    with open(f"logs/{args.exp_name}/cfgs.pkl", "rb") as f:
        env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(f)
    reward_cfg["reward_scales"] = {}

    env_cfg["visualize_target"] = True
    # for video recording
    env_cfg["visualize_camera"] = args.record
    # set the max FPS for visualization
    env_cfg["max_visualize_FPS"] = 60

    env = HoverEnv(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=True,
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    runner.load(os.path.join(log_dir, f"model_{args.ckpt}.pt"))
    policy = runner.get_inference_policy(device=gs.device)

    obs_dict = env.reset()

    max_sim_step = int(env_cfg["episode_length_s"] * env_cfg["max_visualize_FPS"])
    with torch.no_grad():
        if args.record:
            env.cam.start_recording(save_to_filename="out/hover.mp4", fps=env_cfg["max_visualize_FPS"])
            for _ in range(max_sim_step):
                actions = policy(obs_dict)
                obs_dict, rews, dones, infos = env.step(actions)
            env.cam.stop_recording()
        else:
            for _ in range(max_sim_step):
                actions = policy(obs_dict)
                obs_dict, rews, dones, infos = env.step(actions)


if __name__ == "__main__":
    main()

"""
# evaluation
python examples/drone/hover_eval.py --ckpt 300

# Note
If you experience slow performance or encounter other issues
during evaluation, try removing the --record option.
"""

import argparse
import re
import pickle
from importlib import metadata
from pathlib import Path

import torch

try:
    if int(metadata.version("rsl-rl-lib").split(".")[0]) < 5:
        raise ImportError
except (metadata.PackageNotFoundError, ImportError, ValueError) as e:
    raise ImportError("Please install 'rsl-rl-lib>=5.0.0'.") from e
from rsl_rl.runners import OnPolicyRunner

import genesis as gs

from grasp_env import GraspEnv
from behavior_cloning import BehaviorCloning


def load_rl_policy(env, train_cfg, log_dir):
    """Load reinforcement learning policy."""
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)

    checkpoint_files = [f for f in log_dir.iterdir() if re.match(r"model_\d+\.pt", f.name)]
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {log_dir}")

    last_ckpt = max(checkpoint_files, key=lambda f: int(re.search(r"\d+", f.stem).group()))
    runner.load(last_ckpt)
    print(f"Loaded RL checkpoint from {last_ckpt}")

    return runner.get_inference_policy(device=gs.device)


def load_bc_policy(env, bc_cfg, log_dir):
    """Load behavior cloning policy."""
    bc_runner = BehaviorCloning(env, bc_cfg, None, device=gs.device)

    checkpoint_files = [f for f in log_dir.iterdir() if re.match(r"checkpoint_\d+\.pt", f.name)]
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {log_dir}")

    last_ckpt = max(checkpoint_files, key=lambda f: int(re.search(r"\d+", f.stem).group()))
    print(f"Loaded BC checkpoint from {last_ckpt}")
    bc_runner.load(last_ckpt)

    return bc_runner._policy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e", "--exp-name", type=str, default="grasp", help="Experiment name; also the log directory under logs/"
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="rl",
        choices=["rl", "bc"],
        help="Model type: 'rl' for reinforcement learning, 'bc' for behavior cloning",
    )
    parser.add_argument("-r", "--record", action="store_true", help="Record stereo images as video during evaluation")
    parser.add_argument(
        "--video-path",
        type=str,
        default=None,
        help="Video filename for the main camera, written under the run's output directory",
    )
    args = parser.parse_args()

    gs.init(backend=gs.cpu)

    run_name = f"{args.exp_name}_{args.stage}"
    log_dir = Path("logs") / run_name
    video_dir = Path("out") / run_name

    with open(log_dir / "cfgs.pkl", "rb") as f:
        env_cfg, reward_cfg, robot_cfg, rl_train_cfg, bc_train_cfg = pickle.load(f)

    env_cfg["num_envs"] = 10
    env_cfg["box_fixed"] = False
    env_cfg["visualize_camera"] = True

    if args.record:
        env_cfg["record_video"] = {
            "vis_cam": str(video_dir / (args.video_path or "video.mp4")),
            "left_cam": str(video_dir / "left_cam.mp4"),
            "right_cam": str(video_dir / "right_cam.mp4"),
        }

    env = GraspEnv(
        env_cfg=env_cfg,
        reward_cfg=reward_cfg,
        robot_cfg=robot_cfg,
        show_viewer=True,
    )

    # Load the appropriate policy based on model type
    if args.stage == "rl":
        policy = load_rl_policy(env, rl_train_cfg, log_dir)
    else:
        policy = load_bc_policy(env, bc_train_cfg, log_dir)
        policy.eval()

    obs_dict = env.reset()

    max_sim_step = int(env_cfg["episode_length_s"] / env_cfg["ctrl_dt"])

    with torch.no_grad():
        for step in range(max_sim_step):
            if args.stage == "rl":
                actions = policy(obs_dict)
            else:
                rgb_obs = env.get_stereo_rgb_images(normalize=True).float()
                ee_pose = env.robot.ee_pose.float()
                actions = policy(rgb_obs, ee_pose)

            obs_dict, rews, dones, infos = env.step(actions)
        env.grasp_and_lift_demo()
        if args.record:
            env.scene.stop_recording()


if __name__ == "__main__":
    main()

"""
# evaluation
# For reinforcement learning model:
python examples/manipulation/grasp_eval.py --stage=rl

# For behavior cloning model:
python examples/manipulation/grasp_eval.py --stage=bc

# With video recording:
python examples/manipulation/grasp_eval.py --stage=bc --record
"""

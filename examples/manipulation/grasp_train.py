import argparse
import re
import pickle
from importlib import metadata
from pathlib import Path

try:
    if int(metadata.version("rsl-rl-lib").split(".")[0]) < 5:
        raise ImportError
except (metadata.PackageNotFoundError, ImportError) as e:
    raise ImportError("Please install 'rsl-rl-lib>=5.0.0'.") from e

from rsl_rl.runners import OnPolicyRunner
from behavior_cloning import BehaviorCloning

import genesis as gs

from grasp_env import GraspEnv


def get_train_cfg(exp_name):
    # stage 1: privileged reinforcement learning
    rl_cfg_dict = {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.0,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.0003,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "actor": {
            "class_name": "MLPModel",
            "hidden_dims": [256, 256, 128],
            "activation": "relu",
            "distribution_cfg": {
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        },
        "critic": {
            "class_name": "MLPModel",
            "hidden_dims": [256, 256, 128],
            "activation": "relu",
        },
        "obs_groups": {
            "actor": ["policy"],
            "critic": ["policy"],
        },
        "num_steps_per_env": 24,
        "save_interval": 100,
        "run_name": exp_name,
        "logger": "tensorboard",
    }

    # stage 2: vision-based behavior cloning
    bc_cfg_dict = {
        # Basic training parameters
        "num_steps_per_env": 24,
        "learning_rate": 0.001,
        "num_epochs": 5,
        "num_mini_batches": 10,
        "max_grad_norm": 1.0,
        # Network architecture
        "policy": {
            "vision_encoder": {
                "conv_layers": [
                    {
                        "in_channels": 3,  # 3 channel for rgb image
                        "out_channels": 8,
                        "kernel_size": 3,
                        "stride": 1,
                        "padding": 1,
                    },
                    {
                        "in_channels": 8,
                        "out_channels": 16,
                        "kernel_size": 3,
                        "stride": 2,
                        "padding": 1,
                    },
                    {
                        "in_channels": 16,
                        "out_channels": 32,
                        "kernel_size": 3,
                        "stride": 2,
                        "padding": 1,
                    },
                ],
                "pooling": "adaptive_avg",
            },
            "action_head": {
                "state_obs_dim": 7,  # end-effector pose as additional state observation
                "hidden_dims": [128, 128, 64],
            },
            "pose_head": {
                "hidden_dims": [64, 64],
            },
        },
        # Training settings
        "buffer_size": 1000,
        "log_freq": 10,
        "save_freq": 50,
        "eval_freq": 50,
    }

    return rl_cfg_dict, bc_cfg_dict


def get_task_cfgs():
    env_cfg = {
        "num_envs": 10,
        "num_actions": 6,
        "action_scales": [0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
        "episode_length_s": 3.0,
        "ctrl_dt": 0.01,
        "box_size": [0.08, 0.03, 0.06],
        "image_resolution": (64, 64),
        "visualize_camera": False,
    }
    reward_scales = {
        "keypoints": 1.0,
    }
    # panda robot specific
    robot_cfg = {
        "ee_link_name": "hand",
        "gripper_link_names": ["left_finger", "right_finger"],
        "default_arm_dof": [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785],
        "default_gripper_dof": [0.04, 0.04],
        "ik_method": "dls_ik",
    }
    return env_cfg, reward_scales, robot_cfg


def load_teacher_policy(env, rl_train_cfg, exp_name):
    # load teacher policy
    log_dir = Path("logs") / f"{exp_name + '_' + 'rl'}"
    assert log_dir.exists(), f"Log directory {log_dir} does not exist"
    checkpoint_files = [f for f in log_dir.iterdir() if re.match(r"model_\d+\.pt", f.name)]
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {log_dir}")
    last_ckpt = max(checkpoint_files, key=lambda f: int(re.search(r"\d+", f.stem).group()))
    runner = OnPolicyRunner(env, rl_train_cfg, log_dir, device=gs.device)
    runner.load(last_ckpt)
    print(f"Loaded teacher policy from checkpoint {last_ckpt} from {log_dir}")
    teacher_policy = runner.get_inference_policy(device=gs.device)
    return teacher_policy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e", "--exp-name", type=str, default="grasp", help="Experiment name; also the log directory under logs/"
    )
    parser.add_argument("-v", "--vis", action="store_true", help="Show visualization GUI")
    parser.add_argument("-b", "--num-envs", type=int, default=2048, help="Number of parallel environments")
    parser.add_argument("--max-iterations", type=int, default=300, help="Number of learning iterations")
    parser.add_argument(
        "--stage",
        type=str,
        default="rl",
        choices=("rl", "bc"),
        help="Training stage: 'rl' for reinforcement learning, 'bc' for behavior cloning",
    )
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    args = parser.parse_args()

    # Training throughput comes from the 2048 parallel environments, which need a GPU.
    gs.init(backend=gs.gpu, precision="32", logging_level="warning", seed=args.seed, performance_mode=True)

    env_cfg, reward_scales, robot_cfg = get_task_cfgs()
    rl_train_cfg, bc_train_cfg = get_train_cfg(args.exp_name)

    log_dir = Path("logs") / f"{args.exp_name + '_' + args.stage}"
    log_dir.mkdir(parents=True, exist_ok=True)

    # BC only needs a small number of envs, e.g., 10
    env_cfg["num_envs"] = args.num_envs if args.stage == "rl" else 10

    with open(log_dir / "cfgs.pkl", "wb") as f:
        pickle.dump((env_cfg, reward_scales, robot_cfg, rl_train_cfg, bc_train_cfg), f)
    env = GraspEnv(
        env_cfg=env_cfg,
        reward_cfg=reward_scales,
        robot_cfg=robot_cfg,
        show_viewer=args.vis,
    )

    if args.stage == "bc":
        teacher_policy = load_teacher_policy(env, rl_train_cfg, args.exp_name)
        runner = BehaviorCloning(env, bc_train_cfg, teacher_policy, device=gs.device)
        runner.learn(num_learning_iterations=args.max_iterations, log_dir=log_dir)
    else:
        runner = OnPolicyRunner(env, rl_train_cfg, log_dir, device=gs.device)
        runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()

"""
# training

# to train the RL policy
python examples/manipulation/grasp_train.py --stage=rl

# to train the BC policy (requires RL policy to be trained first)
python examples/manipulation/grasp_train.py --stage=bc
"""

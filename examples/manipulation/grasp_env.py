import math
from functools import partial
from typing import Literal

import torch
from tensordict import TensorDict

import genesis as gs
from genesis.options.sensors import BatchRendererCameraOptions, RasterizerCameraOptions
from genesis.vis.camera import Camera
from genesis.utils.geom import (
    xyz_to_quat,
    transform_quat_by_quat,
    transform_by_trans_quat,
)

try:
    import gs_madrona

    _ENABLE_MADRONA = True
except ImportError:
    _ENABLE_MADRONA = False


class GraspEnv:
    def __init__(
        self,
        env_cfg: dict,
        reward_cfg: dict,
        robot_cfg: dict,
        show_viewer: bool = False,
    ) -> None:
        self.num_envs = env_cfg["num_envs"]
        self.num_actions = env_cfg["num_actions"]
        self.cfg = env_cfg
        self.device = gs.device

        self.ctrl_dt = env_cfg["ctrl_dt"]
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.ctrl_dt)

        # configs
        self.env_cfg = env_cfg
        self.reward_scales = reward_cfg
        self.action_scales = torch.tensor(env_cfg["action_scales"], device=self.device)

        self.image_width = env_cfg["image_resolution"][0]
        self.image_height = env_cfg["image_resolution"][1]
        self.rgb_image_shape = (3, self.image_height, self.image_width)

        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=self.ctrl_dt,
            ),
            rigid_options=gs.options.RigidOptions(
                dt=self.ctrl_dt / 2,
                enable_collision=True,
                enable_joint_limit=True,
                constraint_solver=gs.constraint_solver.Newton,
            ),
            vis_options=gs.options.VisOptions(
                split_envs=True,
                rendered_envs_idx=list(range(min(10, self.num_envs))),
            ),
            viewer_options=gs.options.ViewerOptions(
                res=(1280, 960),
                camera_pos=(2.5, -1.0, 2.5),
                camera_lookat=(0.5, -0.3, 0.1),
                camera_fov=55,
            ),
            profiling_options=gs.options.ProfilingOptions(
                show_FPS=False,
            ),
            show_viewer=show_viewer,
        )

        self.scene.add_entity(
            gs.morphs.Plane(),
        )

        self.robot = Manipulator(
            num_envs=self.num_envs,
            scene=self.scene,
            args=robot_cfg,
            device=gs.device,
        )

        self.object = self.scene.add_entity(
            gs.morphs.Box(
                size=env_cfg["box_size"],
                fixed=env_cfg.get("box_fixed", True),
                batch_fixed_verts=True,
            ),
            surface=gs.surfaces.Rough(
                diffuse_texture=gs.textures.ColorTexture(
                    color=(1.0, 0.0, 0.0),
                ),
            ),
        )

        # Debug-only visualization camera, on the scene camera API.
        if self.env_cfg.get("visualize_camera", False):
            self.vis_cam = self.scene.add_camera(
                res=(1280, 960),
                pos=(3.5, 0.0, 2.5),
                lookat=(1.2, 1.0, 0.0),
                fov=52,
                GUI=False,
                debug=True,
            )

        # Stereo camera sensors render lazily: they cost nothing until read().
        if _ENABLE_MADRONA and gs.backend == gs.cuda:
            CameraOptions = BatchRendererCameraOptions
            cam_kwargs = dict(use_rasterizer=True)
        else:
            CameraOptions = RasterizerCameraOptions
            cam_kwargs = {}

        self.left_cam = self.scene.add_sensor(
            CameraOptions(
                res=(self.image_width, self.image_height),
                pos=(1.25, 0.3, 0.3),
                lookat=(0.0, 0.0, 0.0),
                fov=60,
                **cam_kwargs,
            )
        )
        self.right_cam = self.scene.add_sensor(
            CameraOptions(
                res=(self.image_width, self.image_height),
                pos=(1.25, -0.3, 0.3),
                lookat=(0.0, 0.0, 0.0),
                fov=60,
                **cam_kwargs,
            )
        )

        def _read_scene_cam(cam):
            rgb = cam.render(rgb=True)[0]
            if rgb.ndim == 4:
                rgb = rgb[0]
            return rgb[..., :3]

        def _read_sensor_cam(cam):
            return cam.read(envs_idx=0).rgb

        # Debug live preview of sensor cameras
        if self.env_cfg.get("visualize_camera", False):
            self.scene.add_recorder(
                data_func=partial(_read_sensor_cam, self.left_cam),
                rec_options=gs.recorders.MPLImagePlot(title="Left Camera"),
            )
            self.scene.add_recorder(
                data_func=partial(_read_sensor_cam, self.right_cam),
                rec_options=gs.recorders.MPLImagePlot(title="Right Camera"),
            )

        # Video recording must be set up before the scene is built.
        record_video = env_cfg.get("record_video", {})
        for cam_name, filename in record_video.items():
            cam = getattr(self, cam_name)
            reader = _read_scene_cam if isinstance(cam, Camera) else _read_sensor_cam
            self.scene.add_recorder(
                data_func=partial(reader, cam),
                rec_options=gs.recorders.VideoFile(filename=filename),
            )

        self.scene.build(n_envs=env_cfg["num_envs"], env_spacing=(1.0, 1.0))
        # set pd gains (must be called after scene.build)
        self.robot.set_pd_gains()

        # prepare reward functions and multiply reward scales by dt
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.ctrl_dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)

        self.keypoints_offset = self.get_keypoint_offsets(batch_size=self.num_envs, device=self.device, unit_length=0.5)
        self._init_buffers()
        self.reset()

    def _init_buffers(self) -> None:
        self.episode_length_buf = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_int)
        self.reset_buf = torch.ones(self.num_envs, dtype=gs.tc_bool, device=gs.device)
        self.goal_pose = torch.zeros(self.num_envs, 7, device=gs.device, dtype=gs.tc_float)
        self.extras = dict()

    def _reset_idx(self, envs_idx=None) -> None:
        """Reset specified environments.

        Parameters
        ----------
        envs_idx : torch.Tensor or None
            Boolean mask of shape (num_envs,) for selective reset, or None for full reset.
        """
        self.robot.reset(envs_idx)

        # Generate random object state for all envs
        random_x = torch.rand(self.num_envs, device=self.device) * 0.4 + 0.2
        random_y = (torch.rand(self.num_envs, device=self.device) - 0.5) * 0.5
        random_z = torch.full((self.num_envs,), 0.025, device=self.device)
        random_pos = torch.stack([random_x, random_y, random_z], dim=-1)

        q_downward = torch.tensor([0.0, 1.0, 0.0, 0.0], device=self.device).expand(self.num_envs, -1)
        random_yaw = (torch.rand(self.num_envs, device=self.device) * 2 * math.pi - math.pi) * 0.25
        q_yaw = torch.stack(
            [
                torch.cos(random_yaw / 2),
                torch.zeros(self.num_envs, device=self.device),
                torch.zeros(self.num_envs, device=self.device),
                torch.sin(random_yaw / 2),
            ],
            dim=-1,
        )
        goal_yaw = transform_quat_by_quat(q_yaw, q_downward)
        goal_pose = torch.cat([random_pos, goal_yaw], dim=-1)

        # Reset object - set_pos/set_quat with skip_forward, then FK runs once for everything
        if envs_idx is None:
            self.goal_pose.copy_(goal_pose)
            self.object.set_pos(random_pos, skip_forward=True)
            self.object.set_quat(goal_yaw, skip_forward=False)
            self.episode_length_buf.zero_()
            self.reset_buf.fill_(True)
        else:
            torch.where(envs_idx[:, None], goal_pose, self.goal_pose, out=self.goal_pose)
            self.object.set_pos(random_pos, envs_idx=envs_idx, skip_forward=True)
            self.object.set_quat(goal_yaw, envs_idx=envs_idx, skip_forward=False)
            self.episode_length_buf.masked_fill_(envs_idx, 0)
            self.reset_buf.masked_fill_(envs_idx, True)

        # Invalidate camera caches after state change
        self.left_cam._stale = True
        self.right_cam._stale = True

        # Fill extras
        n_envs = envs_idx.sum() if envs_idx is not None else self.num_envs
        self.extras["episode"] = {}
        for key, value in self.episode_sums.items():
            if envs_idx is None:
                mean = value.mean()
            else:
                mean = torch.where(n_envs > 0, value[envs_idx].sum() / n_envs, 0.0)
            self.extras["episode"]["rew_" + key] = mean / self.env_cfg["episode_length_s"]
            if envs_idx is None:
                value.zero_()
            else:
                value.masked_fill_(envs_idx, 0.0)

    def reset(self) -> TensorDict:
        self._reset_idx()
        return self.get_observations()

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        # apply action
        actions = self.rescale_action(actions)
        self.robot.apply_action(actions, open_gripper=True)
        self.scene.step()

        # update time
        self.episode_length_buf += 1

        # check termination (bool mask)
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= self.scene.rigid_solver.get_error_envs_mask()

        # timeout for value bootstrapping (only true timeouts, not NaN errors)
        self.extras["time_outs"] = (self.episode_length_buf > self.max_episode_length).to(dtype=gs.tc_float)

        # compute reward before reset (reflects terminal state)
        reward = torch.zeros(self.num_envs, device=gs.device, dtype=gs.tc_float)
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            reward += rew
            self.episode_sums[name] += rew

        # soft-reset envs that need it
        self._reset_idx(self.reset_buf)

        return self.get_observations(), reward, self.reset_buf, self.extras

    def get_observations(self) -> TensorDict:
        # Current end-effector pose
        finger_pos, finger_quat = (
            self.robot.center_finger_pose[:, :3],
            self.robot.center_finger_pose[:, 3:7],
        )
        obj_pos, obj_quat = self.object.get_pos(), self.object.get_quat()
        obs_components = [
            finger_pos - obj_pos,  # 3D position difference
            finger_quat,  # current orientation (w, x, y, z)
            obj_pos,  # goal position
            obj_quat,  # goal orientation (w, x, y, z)
        ]
        self.obs_buf = torch.cat(obs_components, dim=-1)
        return TensorDict({"policy": self.obs_buf}, batch_size=[self.num_envs])

    def rescale_action(self, action: torch.Tensor) -> torch.Tensor:
        return action * self.action_scales

    def get_stereo_rgb_images(self, normalize: bool = True) -> torch.Tensor:
        rgb_left = self.left_cam.read().rgb  # (B, H, W, 3)
        rgb_right = self.right_cam.read().rgb  # (B, H, W, 3)

        # Convert to (B, 3, H, W) float
        rgb_left = rgb_left.permute(0, 3, 1, 2).float()
        rgb_right = rgb_right.permute(0, 3, 1, 2).float()

        if normalize:
            rgb_left = rgb_left / 255.0
            rgb_right = rgb_right / 255.0

        # Concatenate left and right rgb images along channel dimension: [B, 6, H, W]
        return torch.cat([rgb_left, rgb_right], dim=1)

    # Reward functions.
    def _reward_keypoints(self) -> torch.Tensor:
        keypoints_offset = self.keypoints_offset
        # there is a offset between the finger tip and the finger base frame
        finger_tip_z_offset = torch.tensor(
            [0.0, 0.0, -0.06],
            device=self.device,
            dtype=gs.tc_float,
        ).repeat(self.num_envs, 1)
        finger_pos_keypoints = self._to_world_frame(
            self.robot.center_finger_pose[:, :3] + finger_tip_z_offset,
            self.robot.center_finger_pose[:, 3:7],
            keypoints_offset,
        )
        object_pos_keypoints = self._to_world_frame(self.object.get_pos(), self.object.get_quat(), keypoints_offset)
        dist = torch.norm(finger_pos_keypoints - object_pos_keypoints, p=2, dim=-1).sum(-1)
        return torch.exp(-dist)

    @staticmethod
    def _to_world_frame(
        position: torch.Tensor,  # [B, 3]
        quaternion: torch.Tensor,  # [B, 4]
        keypoints_offset: torch.Tensor,  # [B, K, 3]
    ) -> torch.Tensor:
        return transform_by_trans_quat(keypoints_offset, position.unsqueeze(1), quaternion.unsqueeze(1))

    @staticmethod
    def get_keypoint_offsets(batch_size: int, device: str, unit_length: float = 0.5) -> torch.Tensor:
        """
        Get uniformly-spaced keypoints along a line of unit length, centered at body center.
        """
        keypoint_offsets = (
            torch.tensor(
                [
                    [0, 0, 0],  # origin
                    [-1.0, 0, 0],  # x-negative
                    [1.0, 0, 0],  # x-positive
                    [0, -1.0, 0],  # y-negative
                    [0, 1.0, 0],  # y-positive
                    [0, 0, -1.0],  # z-negative
                    [0, 0, 1.0],  # z-positive
                ],
                device=device,
                dtype=torch.float32,
            )
            * unit_length
        )
        return keypoint_offsets[None].repeat((batch_size, 1, 1))

    def grasp_and_lift_demo(self) -> None:
        total_steps = 500
        goal_pose = self.robot.ee_pose.clone()
        # lift pose (above the object)
        lift_height = 0.3
        lift_pose = goal_pose.clone()
        lift_pose[:, 2] += lift_height
        # final pose (above the table)
        final_pose = goal_pose.clone()
        final_pose[:, 0] = 0.3
        final_pose[:, 1] = 0.0
        final_pose[:, 2] = 0.4
        # reset pose (home pose)
        reset_pose = torch.tensor([0.2, 0.0, 0.4, 0.0, 1.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        for i in range(total_steps):
            if i < total_steps / 4:  # grasping
                self.robot.go_to_goal(goal_pose, open_gripper=False)
            elif i < total_steps / 2:  # lifting
                self.robot.go_to_goal(lift_pose, open_gripper=False)
            elif i < total_steps * 3 / 4:  # final
                self.robot.go_to_goal(final_pose, open_gripper=False)
            else:  # reset
                self.robot.go_to_goal(reset_pose, open_gripper=True)
            self.scene.step()


class Manipulator:
    def __init__(self, num_envs: int, scene: gs.Scene, args: dict, device: str = "cpu"):
        self._device = device
        self._scene = scene
        self._num_envs = num_envs
        self._args = args

        material: gs.materials.Rigid = gs.materials.Rigid()
        morph: gs.morphs.MJCF = gs.morphs.MJCF(
            file="xml/franka_emika_panda/panda.xml",
            pos=(0.0, 0.0, 0.0),
            quat=(1.0, 0.0, 0.0, 0.0),
        )
        self._robot_entity: gs.Entity = scene.add_entity(
            morph=morph,
            material=material,
        )

        self._gripper_open_dof = 0.04
        self._gripper_close_dof = 0.00

        self._ik_method: Literal["gs_ik", "dls_ik"] = args["ik_method"]

        self._init()

    def set_pd_gains(self):
        # set control gains
        # Note: the following values are tuned for achieving best behavior with Franka
        # Typically, each new robot would have a different set of parameters.
        # Sometimes high-quality URDF or XML file would also provide this and will be parsed.
        self._robot_entity.set_dofs_kp(
            torch.tensor([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]),
        )
        self._robot_entity.set_dofs_kv(
            torch.tensor([450, 450, 350, 350, 200, 200, 200, 10, 10]),
        )
        self._robot_entity.set_dofs_force_range(
            torch.tensor([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
            torch.tensor([87, 87, 87, 87, 12, 12, 12, 100, 100]),
        )

    def _init(self):
        self._arm_dof_dim = self._robot_entity.n_dofs - 2  # total number of arm joints
        self._gripper_dim = 2  # number of gripper joints

        self._arm_dof_idx = torch.arange(self._arm_dof_dim, device=self._device)
        self._fingers_dof = torch.arange(
            self._arm_dof_dim,
            self._arm_dof_dim + self._gripper_dim,
            device=self._device,
        )
        self._left_finger_dof = self._fingers_dof[0]
        self._right_finger_dof = self._fingers_dof[1]
        self._ee_link = self._robot_entity.get_link(self._args["ee_link_name"])
        self._left_finger_link = self._robot_entity.get_link(self._args["gripper_link_names"][0])
        self._right_finger_link = self._robot_entity.get_link(self._args["gripper_link_names"][1])
        self._default_joint_angles = self._args["default_arm_dof"]
        if self._args["default_gripper_dof"] is not None:
            self._default_joint_angles += self._args["default_gripper_dof"]
        self._init_qpos = torch.tensor(self._default_joint_angles, dtype=torch.float32, device=self._device)
        # On MPS/Metal, batched linear algebra is extremely slow due to per-element kernel dispatch.
        # Running the DLS solve on CPU is ~300x faster in that case.
        self._dls_solve_on_cpu = self._device == "mps" or str(self._device).startswith("mps")
        dls_lam_device = "cpu" if self._dls_solve_on_cpu else self._device
        self._dls_lambda_matrix = (0.01**2) * torch.eye(6, device=dls_lam_device)

    def reset(self, envs_idx=None, skip_forward=True):
        self._robot_entity.set_qpos(
            self._init_qpos,
            envs_idx=envs_idx,
            zero_velocity=True,
            skip_forward=skip_forward,
        )

    def apply_action(self, action: torch.Tensor, open_gripper: bool) -> None:
        """Apply the action to the robot."""
        if self._ik_method == "gs_ik":
            q_pos = self._gs_ik(action)
        elif self._ik_method == "dls_ik":
            q_pos = self._dls_ik(action)
        else:
            raise ValueError(f"Invalid control mode: {self._ik_method}")
        if open_gripper:
            q_pos[:, self._fingers_dof] = self._gripper_open_dof
        else:
            q_pos[:, self._fingers_dof] = self._gripper_close_dof
        self._robot_entity.control_dofs_position(position=q_pos)

    def _gs_ik(self, action: torch.Tensor) -> torch.Tensor:
        """
        Genesis inverse kinematics
        """
        delta_position = action[:, :3]
        delta_orientation = action[:, 3:6]

        # compute target pose
        target_position = delta_position + self._ee_link.get_pos()
        quat_rel = xyz_to_quat(delta_orientation, rpy=True, degrees=False)
        target_orientation = transform_quat_by_quat(quat_rel, self._ee_link.get_quat())
        q_pos = self._robot_entity.inverse_kinematics(
            link=self._ee_link,
            pos=target_position,
            quat=target_orientation,
            dofs_idx_local=self._arm_dof_idx,
        )
        return q_pos

    def _dls_ik(self, action: torch.Tensor) -> torch.Tensor:
        """
        Damped least squares inverse kinematics.

        Solves (J @ J^T + lambda^2 * I) @ y = dx, then dq = J^T @ y.
        """
        delta_pose = action[:, :6]
        jacobian = self._robot_entity.get_jacobian(link=self._ee_link)
        if self._dls_solve_on_cpu:
            jacobian = jacobian.cpu()
            delta_pose = delta_pose.cpu()
        A = torch.baddbmm(self._dls_lambda_matrix, jacobian, jacobian.mT)
        y = torch.linalg.solve(A, delta_pose)
        delta_joint_pos = (jacobian.mT @ y.unsqueeze(-1)).squeeze(-1)
        if self._dls_solve_on_cpu:
            delta_joint_pos = delta_joint_pos.to(self._device)
        return self._robot_entity.get_qpos() + delta_joint_pos

    def go_to_goal(self, goal_pose: torch.Tensor, open_gripper: bool = True):
        q_pos = self._robot_entity.inverse_kinematics(
            link=self._ee_link,
            pos=goal_pose[:, :3],
            quat=goal_pose[:, 3:7],
            dofs_idx_local=self._arm_dof_idx,
        )
        if open_gripper:
            q_pos[:, self._fingers_dof] = self._gripper_open_dof
        else:
            q_pos[:, self._fingers_dof] = self._gripper_close_dof
        self._robot_entity.control_dofs_position(position=q_pos)

    @property
    def base_pos(self):
        return self._robot_entity.get_pos()

    @property
    def ee_pose(self) -> torch.Tensor:
        """
        The end-effector pose (the hand pose)
        """
        pos, quat = self._ee_link.get_pos(), self._ee_link.get_quat()
        return torch.cat([pos, quat], dim=-1)

    @property
    def left_finger_pose(self) -> torch.Tensor:
        pos, quat = self._left_finger_link.get_pos(), self._left_finger_link.get_quat()
        return torch.cat([pos, quat], dim=-1)

    @property
    def right_finger_pose(self) -> torch.Tensor:
        pos, quat = (
            self._right_finger_link.get_pos(),
            self._right_finger_link.get_quat(),
        )
        return torch.cat([pos, quat], dim=-1)

    @property
    def center_finger_pose(self) -> torch.Tensor:
        """
        The center finger pose is the average of the left and right finger poses.
        """
        left_finger_pose = self.left_finger_pose
        right_finger_pose = self.right_finger_pose
        center_finger_pos = (left_finger_pose[:, :3] + right_finger_pose[:, :3]) / 2
        center_finger_quat = left_finger_pose[:, 3:7]
        return torch.cat([center_finger_pos, center_finger_quat], dim=-1)

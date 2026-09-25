from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import quadrants as qd
import torch

import genesis as gs
import genesis.utils.array_class as array_class
import genesis.utils.geom as gu
from genesis.options.sensors import IMU as IMUOptions
from genesis.options.sensors import CrossCouplingAxisType
from genesis.utils.misc import concat_with_tensor, make_tensor_field, tensor_to_array

from .base_sensor import SimpleSensor, RigidSensorMetadataMixin, RigidSensorMixin, SimpleSensorMetadata

if TYPE_CHECKING:
    from genesis.ext.pyrender.mesh import Mesh
    from genesis.utils.ring_buffer import TensorRingBuffer
    from genesis.vis.rasterizer_context import RasterizerContext

    from .sensor_manager import SensorManager


@qd.kernel(fastcache=True)
def _kernel_update_imu_raw_data(
    links_idx: qd.types.ndarray(),
    offsets_pos: qd.types.ndarray(),
    offsets_quat: qd.types.ndarray(),
    magnetic_field_vector: qd.types.ndarray(),
    raw_data_T: qd.types.ndarray(),
    dyn_state: array_class.DynState,
    rigid_info: array_class.RigidInfo,
):
    """
    Compute the raw (linear acceleration, angular velocity, magnetic field) triplet of every IMU in its own frame.

    Reading the link state directly keeps the whole per-step update to a single launch, which is what the cost of the
    sensor is made of: the arithmetic itself is negligible next to a simulation step.
    """
    for i_s, i_b in qd.ndrange(links_idx.shape[0], raw_data_T.shape[-1]):
        i_l = links_idx[i_s]

        offset_pos = qd.Vector.zero(gs.qd_float, 3)
        mag = qd.Vector.zero(gs.qd_float, 3)
        for j in qd.static(range(3)):
            offset_pos[j] = offsets_pos[i_b, i_s, j]
            mag[j] = magnetic_field_vector[i_b, i_s, j]
        offset_quat = qd.Vector.zero(gs.qd_float, 4)
        for j in qd.static(range(4)):
            offset_quat[j] = offsets_quat[i_b, i_s, j]

        # Spatial velocity and acceleration are expressed at the root of the kinematic tree, hence the transport to the
        # link origin before converting the spatial acceleration to the classical one.
        cpos = dyn_state.links.pos[i_l, i_b] - dyn_state.links.root_COM[i_l, i_b]
        acc_ang = dyn_state.links.cacc_ang[i_l, i_b]
        ang = dyn_state.links.cd_ang[i_l, i_b]
        vel = dyn_state.links.cd_vel[i_l, i_b] + ang.cross(cpos)
        acc = dyn_state.links.cacc_lin[i_l, i_b] + acc_ang.cross(cpos) + ang.cross(vel)

        # Rigid-body transport from the link origin to the measurement point: a_imu = a + alpha x r + w x (w x r). A
        # zero mounting offset contributes nothing, so this stays unconditional.
        quat = dyn_state.links.quat[i_l, i_b]
        offset_pos_world = gu.qd_transform_by_quat(offset_pos, quat)
        acc = acc + acc_ang.cross(offset_pos_world) + ang.cross(ang.cross(offset_pos_world))

        # An accelerometer measures proper acceleration, so gravity is subtracted before rotating into the sensor frame
        sensor_quat = gu.qd_transform_quat_by_quat(quat, offset_quat)
        local_acc = gu.qd_inv_transform_by_quat(acc - rigid_info.gravity[i_b], sensor_quat)
        local_ang = gu.qd_inv_transform_by_quat(ang, sensor_quat)
        local_mag = gu.qd_inv_transform_by_quat(mag, sensor_quat)

        # Raw buffer layout: (n_imus * 9, B), one contiguous (acc, ang, mag) triplet per sensor
        i_cache_start = i_s * 9
        for j in qd.static(range(3)):
            raw_data_T[i_cache_start + j, i_b] = local_acc[j]
            raw_data_T[i_cache_start + 3 + j, i_b] = local_ang[j]
            raw_data_T[i_cache_start + 6 + j, i_b] = local_mag[j]


def _get_cross_axis_coupling_to_alignment_matrix(
    input: CrossCouplingAxisType, out: torch.Tensor | None = None
) -> torch.Tensor:
    """
    Convert the alignment input to a matrix. Modifies in place if provided, else allocate a new matrix.
    """
    if out is None:
        out = torch.eye(3, dtype=gs.tc_float, device=gs.device)

    if isinstance(input, float):
        # set off-diagonal elements to the scalar value
        torch.diagonal(out)[:] = input
        out.fill_diagonal_(1.0)
    elif isinstance(input, torch.Tensor):
        out.copy_(input)
    else:
        np_input = np.array(input)
        if np_input.shape == (3,):
            # set off-diagonal elements to the vector values
            out[1, 0] = np_input[0]
            out[2, 0] = np_input[0]
            out[0, 1] = np_input[1]
            out[2, 1] = np_input[1]
            out[0, 2] = np_input[2]
            out[1, 2] = np_input[2]
        elif np_input.shape == (3, 3):
            out.copy_(torch.tensor(np_input, dtype=gs.tc_float, device=gs.device))
    return out


@dataclass
class IMUSharedMetadata(RigidSensorMetadataMixin, SimpleSensorMetadata):
    """
    Shared metadata between all IMU sensors.
    """

    alignment_rot_matrix: torch.Tensor = make_tensor_field((0, 0, 3, 3))
    magnetic_field_vector: torch.Tensor = make_tensor_field((0, 0, 3))  # added another dimension to match data layout
    acc_indices: torch.Tensor = make_tensor_field((0, 0), dtype_factory=lambda: gs.tc_int)
    gyro_indices: torch.Tensor = make_tensor_field((0, 0), dtype_factory=lambda: gs.tc_int)
    mag_indices: torch.Tensor = make_tensor_field((0, 0), dtype_factory=lambda: gs.tc_int)


class IMUReturnType(NamedTuple):
    lin_acc: torch.Tensor
    ang_vel: torch.Tensor
    mag: torch.Tensor  # added magnetometer to complete 9-axis IMU


class IMUSensor(RigidSensorMixin[IMUSharedMetadata], SimpleSensor[IMUOptions, None, IMUSharedMetadata, IMUReturnType]):
    def __init__(
        self,
        options: IMUOptions,
        idx: int,
        shared_context,
        shared_metadata: IMUSharedMetadata,
        manager: "SensorManager",
    ):
        # FIXME: Resolution should be made private in mixin, so that it cannot be set by the user directly.
        options.resolution = options.acc_resolution + options.gyro_resolution + options.mag_resolution
        options.bias = options.acc_bias + options.gyro_bias + options.mag_bias
        options.random_walk = options.acc_random_walk + options.gyro_random_walk + options.mag_random_walk
        options.noise = options.acc_noise + options.gyro_noise + options.mag_noise

        super().__init__(options, idx, shared_context, shared_metadata, manager)

        self.debug_objects: list["Mesh"] = []
        self.quat_offset: torch.Tensor
        self.pos_offset: torch.Tensor

    @gs.assert_built
    def set_acc_cross_axis_coupling(self, cross_axis_coupling: CrossCouplingAxisType, envs_idx=None):
        envs_idx = self._sanitize_envs_idx(envs_idx)
        rot_matrix = _get_cross_axis_coupling_to_alignment_matrix(cross_axis_coupling)
        self._shared_metadata.alignment_rot_matrix[envs_idx, self._idx * 3, :, :] = rot_matrix

    @gs.assert_built
    def set_gyro_cross_axis_coupling(self, cross_axis_coupling: CrossCouplingAxisType, envs_idx=None):
        envs_idx = self._sanitize_envs_idx(envs_idx)
        rot_matrix = _get_cross_axis_coupling_to_alignment_matrix(cross_axis_coupling)
        self._shared_metadata.alignment_rot_matrix[envs_idx, self._idx * 3 + 1, :, :] = rot_matrix

    @gs.assert_built
    def set_mag_cross_axis_coupling(self, cross_axis_coupling: CrossCouplingAxisType, envs_idx=None):
        envs_idx = self._sanitize_envs_idx(envs_idx)
        rot_matrix = _get_cross_axis_coupling_to_alignment_matrix(cross_axis_coupling)
        self._shared_metadata.alignment_rot_matrix[envs_idx, self._idx * 3 + 2, :, :] = rot_matrix

    # ================================ internal methods ================================

    def build(self):
        """
        Initialize all shared metadata needed to update all IMU sensors.
        """
        super().build()

        self._shared_metadata.alignment_rot_matrix = concat_with_tensor(
            self._shared_metadata.alignment_rot_matrix,
            torch.stack(
                [
                    _get_cross_axis_coupling_to_alignment_matrix(self._options.acc_cross_axis_coupling),
                    _get_cross_axis_coupling_to_alignment_matrix(self._options.gyro_cross_axis_coupling),
                    _get_cross_axis_coupling_to_alignment_matrix(self._options.mag_cross_axis_coupling),
                ]
            ),
            expand=(self._manager._sim._B, 3, 3, 3),  # 3 sub-matrices after adding mag
            dim=1,
        )

        # Initialize global magnetic field vector
        default_field = self._options.magnetic_field if self._options.magnetic_field is not None else (0.0, 0.0, 0.5)
        if not isinstance(default_field, torch.Tensor):
            default_field = torch.tensor(default_field, device=gs.device, dtype=gs.tc_float)

        self._shared_metadata.magnetic_field_vector = concat_with_tensor(
            self._shared_metadata.magnetic_field_vector, default_field, expand=(self._manager._sim._B, 1, 3), dim=1
        )

        if self._options.draw_debug:
            self.quat_offset = self._shared_metadata.offsets_quat[0, self._idx]
            self.pos_offset = self._shared_metadata.offsets_pos[0, self._idx]

    def _get_return_format(self) -> tuple[tuple[int, ...], ...]:
        return ((3,), (3,), (3,))

    @classmethod
    def _get_cache_dtype(cls) -> torch.dtype:
        return gs.tc_float

    @classmethod
    def _update_raw_data(cls, shared_context: None, shared_metadata: IMUSharedMetadata, raw_data_T: torch.Tensor):
        _kernel_update_imu_raw_data(
            shared_metadata.links_idx,
            shared_metadata.offsets_pos,
            shared_metadata.offsets_quat,
            shared_metadata.magnetic_field_vector,
            raw_data_T,
            shared_metadata.solver.dyn_state,
            shared_metadata.solver.rigid_info,
        )

    @classmethod
    def _apply_transform(cls, shared_metadata: IMUSharedMetadata, data: torch.Tensor, timeline, *, is_measured: bool):
        # Apply alignment rotation to the (lin_acc, ang_vel, mag) triplet. View the flat cache as a stack of 3-vectors
        # and rotate them in place with the per-sensor `alignment_rot_matrix`. Branch-symmetric stateless transform:
        # `timeline` and `is_measured` are received for API uniformity but not read.
        data_xyz = data.view(data.shape[0], -1, 3)
        data_xyz.copy_(torch.matmul(shared_metadata.alignment_rot_matrix, data_xyz.unsqueeze(-1)).squeeze(-1))

    def _draw_debug(self, context: "RasterizerContext"):
        """
        Draw debug arrow for the IMU acceleration.

        Only draws for first rendered environment.
        """
        env_idx = context.rendered_envs_idx[0] if self._manager._sim.n_envs > 0 else None

        quat = self._link.get_quat(env_idx, relative=False).reshape((4,))
        pos = self._link.get_pos(env_idx, relative=False).reshape((3,)) + gu.transform_by_quat(self.pos_offset, quat)

        # cannot specify envs_idx for read() when n_envs=0
        data = self.read(env_idx)
        acc_vec = data.lin_acc.reshape((3,)) * self._options.debug_acc_scale
        gyro_vec = data.ang_vel.reshape((3,)) * self._options.debug_gyro_scale
        mag_vec = data.mag.reshape((3,)) * self._options.debug_mag_scale

        # transform from local frame to world frame
        offset_quat = gu.transform_quat_by_quat(self.quat_offset, quat)
        acc_vec = tensor_to_array(gu.transform_by_quat(acc_vec, offset_quat))
        gyro_vec = tensor_to_array(gu.transform_by_quat(gyro_vec, offset_quat))
        mag_vec = tensor_to_array(gu.transform_by_quat(mag_vec, offset_quat))

        for debug_object in self.debug_objects:
            context.clear_debug_object(debug_object)
        self.debug_objects.clear()

        self.debug_objects += filter(
            None,
            (
                context.draw_debug_arrow(pos=pos, vec=acc_vec, radius=0.006, color=self._options.debug_acc_color),
                context.draw_debug_arrow(pos=pos, vec=gyro_vec, radius=0.0055, color=self._options.debug_gyro_color),
                context.draw_debug_arrow(pos=pos, vec=mag_vec, radius=0.005, color=self._options.debug_mag_color),
            ),
        )

from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

import torch

import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu
from genesis.engine.render_context import Arrows, Shape
from genesis.engine.solvers.rigid.data import DynState, RigidInfo
from genesis.engine.state_change import StateChange
from genesis.options.sensors import IMU as IMUOptions, CrossCouplingAxisType
from genesis.utils.misc import concat_with_tensor, make_tensor_field

from .base_sensor import (
    RigidSensorMetadataMixin,
    RigidSensorMixin,
    SimpleSensor,
    SimpleSensorMetadata,
    func_sensor_link_motion,
    func_sensor_link_pose,
)

if TYPE_CHECKING:
    from .sensor_manager import SensorManager


@qd.kernel(fastcache=True)
def _kernel_update_imu_raw_data(
    links_idx: qd.types.ndarray(),
    offsets_pos: qd.types.ndarray(),
    offsets_quat: qd.types.ndarray(),
    magnetic_field_vector: qd.types.ndarray(),
    raw_data_T: qd.types.ndarray(),
    dyn_state: DynState,
    rigid_info: RigidInfo,
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
        pos, quat = func_sensor_link_pose(i_l, i_b, dyn_state.links)
        cd_vel, ang, cacc_lin, acc_ang, root_com = func_sensor_link_motion(i_l, i_b, dyn_state.links)
        cpos = pos - root_com
        vel = cd_vel + ang.cross(cpos)
        acc = cacc_lin + acc_ang.cross(cpos) + ang.cross(vel)

        # Rigid-body transport from the link origin to the measurement point: a_imu = a + alpha x r + w x (w x r). A
        # zero mounting offset contributes nothing, so this stays unconditional.
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

    coupling = torch.as_tensor(input, dtype=gs.tc_float, device=gs.device)
    if coupling.shape in ((), (3,)):
        # set off-diagonal elements to the scalar value, or those of column j to the vector element j
        out.copy_(coupling.expand(3, 3))
        out.fill_diagonal_(1.0)
    elif coupling.shape == (3, 3):
        out.copy_(coupling)
    else:
        gs.raise_exception(
            f"Cross-axis coupling must be a scalar, a 3-vector or a 3x3 matrix. Got shape {tuple(coupling.shape)}."
        )
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

    @gs.assert_built
    def set_acc_cross_axis_coupling(self, cross_axis_coupling: CrossCouplingAxisType, envs_idx=None):
        envs_idx = self._sanitize_envs_idx(envs_idx)
        rot_matrix = _get_cross_axis_coupling_to_alignment_matrix(cross_axis_coupling)
        self._shared_metadata.alignment_rot_matrix[envs_idx, self._idx * 3, :, :] = rot_matrix
        self.changelog.add(StateChange.MODEL, envs_idx=envs_idx)

    @gs.assert_built
    def set_gyro_cross_axis_coupling(self, cross_axis_coupling: CrossCouplingAxisType, envs_idx=None):
        envs_idx = self._sanitize_envs_idx(envs_idx)
        rot_matrix = _get_cross_axis_coupling_to_alignment_matrix(cross_axis_coupling)
        self._shared_metadata.alignment_rot_matrix[envs_idx, self._idx * 3 + 1, :, :] = rot_matrix
        self.changelog.add(StateChange.MODEL, envs_idx=envs_idx)

    @gs.assert_built
    def set_mag_cross_axis_coupling(self, cross_axis_coupling: CrossCouplingAxisType, envs_idx=None):
        envs_idx = self._sanitize_envs_idx(envs_idx)
        rot_matrix = _get_cross_axis_coupling_to_alignment_matrix(cross_axis_coupling)
        self._shared_metadata.alignment_rot_matrix[envs_idx, self._idx * 3 + 2, :, :] = rot_matrix
        self.changelog.add(StateChange.MODEL, envs_idx=envs_idx)

    # ================================ internal methods ================================

    @gs.assert_unbuilt
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
            expand=(self._manager.n_envs_, 3, 3, 3),  # 3 sub-matrices after adding mag
            dim=1,
        )

        # Initialize global magnetic field vector
        default_field = self._options.magnetic_field if self._options.magnetic_field is not None else (0.0, 0.0, 0.5)
        if not isinstance(default_field, torch.Tensor):
            default_field = torch.tensor(default_field, device=gs.device, dtype=gs.tc_float)

        self._shared_metadata.magnetic_field_vector = concat_with_tensor(
            self._shared_metadata.magnetic_field_vector, default_field, expand=(self._manager.n_envs_, 1, 3), dim=1
        )

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

    def _update_markers(self, markers: tuple[Shape, ...], envs_idx: torch.Tensor | None) -> tuple[Arrows, ...]:
        options = self._options
        if not markers:
            # One arrow per reading, each following what the reading depends on: the acceleration the configuration,
            # the velocities and the model, the angular velocity the configuration and the velocities, and the
            # magnetic field the configuration alone.
            solver = self._shared_metadata.solver
            markers = tuple(
                Arrows(
                    torch.zeros((self._manager.n_envs_, 1, 3), dtype=gs.tc_float, device=gs.device),
                    torch.zeros((self._manager.n_envs_, 1, 3), dtype=gs.tc_float, device=gs.device),
                    radius,
                    color,
                    torch.ones((self._manager.n_envs_, 1), dtype=gs.tc_bool, device=gs.device),
                    invalidated_by={solver: changes, self: (StateChange.MODEL,)},
                )
                for radius, color, changes in (
                    (0.006, options.debug_acc_color, (StateChange.GEOMETRY, StateChange.DYNAMICS, StateChange.MODEL)),
                    (0.0055, options.debug_gyro_color, (StateChange.GEOMETRY, StateChange.DYNAMICS)),
                    (0.005, options.debug_mag_color, (StateChange.GEOMETRY,)),
                )
            )
        link_pos = self._link.get_pos(envs_idx, relative=False)
        link_quat = self._link.get_quat(envs_idx, relative=False)
        # The offsets hold a row in a scene without parallel environments too
        sensor_envs_idx = self._sanitize_envs_idx(envs_idx)
        offset_pos = self._shared_metadata.offsets_pos[sensor_envs_idx, self._idx]
        offset_quat = self._shared_metadata.offsets_quat[sensor_envs_idx, self._idx]
        pos = link_pos + gu.transform_by_quat(offset_pos, link_quat)

        data = self.read(envs_idx)
        vecs = torch.stack(
            (
                data.lin_acc * options.debug_acc_scale,
                data.ang_vel * options.debug_gyro_scale,
                data.mag * options.debug_mag_scale,
            ),
            dim=-2,
        )
        # The readings are stated in the frame of the sensor
        sensor_quat = gu.transform_quat_by_quat(offset_quat, link_quat)
        vecs = gu.transform_by_quat(vecs, sensor_quat[..., None, :])
        for i_m, marker in enumerate(markers):
            arrows_pos, arrows_vec = marker.pos, marker.vec
            arrows_pos[envs_idx] = pos[..., None, :]
            arrows_vec[envs_idx] = vecs[..., i_m : i_m + 1, :]
        return markers

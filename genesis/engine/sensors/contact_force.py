from dataclasses import dataclass
from typing import TYPE_CHECKING, Type

import numpy as np
import quadrants as qd
import torch

import genesis as gs
from genesis.options.sensors import Contact as ContactSensorOptions
from genesis.options.sensors import ContactForce as ContactForceSensorOptions
from genesis.utils.geom import inv_transform_by_quat, qd_inv_transform_by_quat, transform_by_quat
from genesis.utils.misc import concat_with_tensor, make_tensor_field, qd_to_torch, tensor_to_array
from genesis.utils.ring_buffer import TensorRingBuffer

from .base_sensor import RigidSensorMetadataMixin, RigidSensorMixin, Sensor, SimpleSensor, SimpleSensorMetadata

if TYPE_CHECKING:
    from genesis.engine.entities.rigid_entity.rigid_link import RigidLink
    from genesis.engine.solvers import RigidSolver
    from genesis.ext.pyrender.mesh import Mesh
    from genesis.vis.rasterizer_context import RasterizerContext

    from .sensor_manager import SensorManager


@qd.func
def _func_link_is_filtered(i_s: int, link: int, filter_links_idx: qd.types.ndarray()):
    """Whether ``link`` is in sensor ``i_s``'s filter row.

    The -1 padding never matches a real (non-negative) link, so every column can be scanned unconditionally.
    """
    is_filtered = False
    for i_f in range(filter_links_idx.shape[-1]):
        if filter_links_idx[i_s, i_f] == link:
            is_filtered = True
            break
    return is_filtered


def _drop_filtered_counterpart_contacts(
    is_a: torch.Tensor,
    is_b: torch.Tensor,
    link_a: torch.Tensor,
    link_b: torch.Tensor,
    shared_metadata: "ContactFilterMetadataMixin",
) -> None:
    """Clear (in place) each per-side contact mask wherever the counterpart link is in the sensor's filter row.

    Only the sensors listed in ``filtered_sensor_idx`` are touched, so the unfiltered majority keeps the untouched
    masks. ``is_a`` / ``is_b`` are ``(B, n_sensors, n_contacts)`` bool masks; ``link_a`` / ``link_b`` are the
    ``(B, n_contacts)`` contact-participant links.
    """
    filtered_sensors = shared_metadata.filtered_sensor_idx
    if filtered_sensors.numel() == 0:
        return
    filter_rows = shared_metadata.filter_links_idx[filtered_sensors][None, :, None, :]
    is_a[:, filtered_sensors, :] &= ~(link_b[:, None, :, None] == filter_rows).any(dim=-1)
    is_b[:, filtered_sensors, :] &= ~(link_a[:, None, :, None] == filter_rows).any(dim=-1)


@qd.kernel
def _kernel_get_contacts_forces(
    sensors_link_idx: qd.types.ndarray(),
    filter_links_idx: qd.types.ndarray(),
    contact_forces: qd.types.ndarray(),
    link_a: qd.types.ndarray(),
    link_b: qd.types.ndarray(),
    links_quat: qd.types.ndarray(),
    output: qd.types.ndarray(),
):
    for i_c, i_s, i_b in qd.ndrange(link_a.shape[-1], sensors_link_idx.shape[-1], output.shape[-1]):
        contact_data_link_a = link_a[i_b, i_c]
        contact_data_link_b = link_b[i_b, i_c]
        if contact_data_link_a == sensors_link_idx[i_s] or contact_data_link_b == sensors_link_idx[i_s]:
            j_s = i_s * 3  # per-sensor output dimension is 3

            quat_a = qd.Vector.zero(gs.qd_float, 4)
            quat_b = qd.Vector.zero(gs.qd_float, 4)
            for j in qd.static(range(4)):
                quat_a[j] = links_quat[i_b, contact_data_link_a, j]
                quat_b[j] = links_quat[i_b, contact_data_link_b, j]

            force_vec = qd.Vector.zero(gs.qd_float, 3)
            for j in qd.static(range(3)):
                force_vec[j] = contact_forces[i_b, i_c, j]

            force_a = qd_inv_transform_by_quat(-force_vec, quat_a)
            force_b = qd_inv_transform_by_quat(force_vec, quat_b)

            # Accumulate the force on whichever side is the sensor link, dropping it when the counterpart is filtered.
            if contact_data_link_a == sensors_link_idx[i_s] and not _func_link_is_filtered(
                i_s, contact_data_link_b, filter_links_idx
            ):
                for j in qd.static(range(3)):
                    output[j_s + j, i_b] += force_a[j]
            if contact_data_link_b == sensors_link_idx[i_s] and not _func_link_is_filtered(
                i_s, contact_data_link_a, filter_links_idx
            ):
                for j in qd.static(range(3)):
                    output[j_s + j, i_b] += force_b[j]


@dataclass
class ContactFilterMetadataMixin:
    """
    Shared state for sensors that scope contacts by counterpart link (see ``ContactFilterOptionsMixin``).

    ``filter_links_idx`` is a ``(n_sensors, max_num_filter_links)`` table; each sensor's row lists its filter links,
    unused slots (and rows for sensors with no filter) are ``-1`` so a kernel can scan every column unconditionally.
    ``filtered_sensor_idx`` lists the rows that declared a filter, letting the aggregation-path sensors (Contact,
    ContactForce) skip the per-contact comparison for the unfiltered majority; the contact-prefilter tactile sensors
    instead apply the filter directly in their build kernels.
    """

    filter_links_idx: torch.Tensor = make_tensor_field((0, 0), dtype_factory=lambda: gs.tc_int)
    filtered_sensor_idx: torch.Tensor = make_tensor_field((0,), dtype_factory=lambda: gs.tc_int)

    def append_filter(self, filter_link_idx) -> None:
        """Append one sensor's filter links as a new table row.

        Grows the column count to fit and back-fills unused slots (and empty filters) with ``-1``. The table keeps at
        least one column so an all-empty table stays a valid (non-zero-dim) kernel argument. Rows with a non-empty
        filter also register in ``filtered_sensor_idx``.
        """
        n_sensors, current_max = self.filter_links_idx.shape
        new_max = max(current_max, len(filter_link_idx), 1)
        table = torch.full((n_sensors + 1, new_max), -1, dtype=gs.tc_int, device=gs.device)
        table[:n_sensors, :current_max] = self.filter_links_idx
        if len(filter_link_idx) > 0:
            table[n_sensors, : len(filter_link_idx)] = torch.tensor(filter_link_idx, dtype=gs.tc_int, device=gs.device)
            self.filtered_sensor_idx = concat_with_tensor(self.filtered_sensor_idx, n_sensors, expand=(1,), dim=0)
        self.filter_links_idx = table


@dataclass
class ContactSensorMetadata(ContactFilterMetadataMixin, SimpleSensorMetadata):
    """
    Metadata for all rigid contact sensors.
    """

    solver: "RigidSolver | None" = None
    expanded_links_idx: torch.Tensor = make_tensor_field((0,), dtype_factory=lambda: gs.tc_int)
    # Per-sensor bool threshold (broadcast over B); _post_process returns `tensor > thresholds`.
    thresholds: torch.Tensor = make_tensor_field((0,))


class ContactSensor(SimpleSensor[ContactSensorOptions, None, ContactSensorMetadata]):
    """
    Sensor that returns bool based on whether associated RigidLink is in contact.
    """

    def __init__(
        self, options: ContactSensorOptions, idx: int, shared_context, shared_metadata, manager: "SensorManager"
    ):
        super().__init__(options, idx, shared_context, shared_metadata, manager)

        self._link: "RigidLink | None" = None
        self.debug_object: "Mesh | None" = None

    def build(self):
        super().build()

        if self._shared_metadata.solver is None:
            self._shared_metadata.solver = self._manager._sim.rigid_solver

        entity = self._shared_metadata.solver.entities[self._options.entity_idx]
        link_idx = self._options.link_idx_local + entity.link_start
        self._link = entity.links[self._options.link_idx_local]

        self._shared_metadata.expanded_links_idx = concat_with_tensor(
            self._shared_metadata.expanded_links_idx, link_idx, expand=(1,), dim=0
        )

        self._shared_metadata.append_filter(self._options.filter_link_idx)

        self._shared_metadata.thresholds = concat_with_tensor(
            self._shared_metadata.thresholds, float(self._options.threshold), expand=(1,)
        )

    def _get_return_format(self) -> tuple[int, ...]:
        return (1,)

    @classmethod
    def _get_cache_dtype(cls) -> torch.dtype:
        return gs.tc_bool

    @classmethod
    def _get_intermediate_dtype(cls) -> torch.dtype:
        # Float kernel output; bool projection happens in `_post_process`. Shape matches `_get_return_format`.
        return gs.tc_float

    @classmethod
    def _update_raw_data(cls, shared_context: None, shared_metadata: ContactSensorMetadata, raw_data_T: torch.Tensor):
        assert shared_metadata.solver is not None
        all_contacts = shared_metadata.solver.collider.get_contacts(as_tensor=True, to_torch=True)
        link_a, link_b = all_contacts["link_a"], all_contacts["link_b"]
        if link_a.shape[-1] == 0:
            raw_data_T.zero_()
            return
        if shared_metadata.solver.n_envs == 0:
            link_a, link_b = link_a[None], link_b[None]

        is_contact_a = link_a[..., None, :] == shared_metadata.expanded_links_idx[..., None]
        is_contact_b = link_b[..., None, :] == shared_metadata.expanded_links_idx[..., None]
        _drop_filtered_counterpart_contacts(is_contact_a, is_contact_b, link_a, link_b, shared_metadata)
        # Float-valued contact count per sensor (intermediate cache is float; bool projection in `_post_process`).
        result = (is_contact_a | is_contact_b).sum(dim=-1).to(dtype=gs.tc_float)
        raw_data_T[:] = result.T

    @classmethod
    def _post_process(
        cls, shared_metadata: ContactSensorMetadata, tensor: torch.Tensor, timeline, *, is_measured: bool
    ) -> torch.Tensor:
        return tensor > shared_metadata.thresholds

    def _draw_debug(self, context: "RasterizerContext"):
        """
        Draw debug sphere when the sensor detects contact.

        Only draws for first rendered environment.
        """
        env_idx = context.rendered_envs_idx[0] if self._manager._sim.n_envs > 0 else None

        pos = self._link.get_pos(env_idx, relative=False).reshape((3,))
        is_contact = self.read(env_idx)

        if self.debug_object is not None:
            context.clear_debug_object(self.debug_object)
            self.debug_object = None

        if is_contact:
            self.debug_object = context.draw_debug_sphere(
                pos=pos, radius=self._options.debug_sphere_radius, color=self._options.debug_color
            )


# ==========================================================================================================


@dataclass
class ContactForceSensorMetadata(ContactFilterMetadataMixin, RigidSensorMetadataMixin, SimpleSensorMetadata):
    """
    Shared metadata for all contact force sensors.
    """

    min_force: torch.Tensor = make_tensor_field((0, 3))
    max_force: torch.Tensor = make_tensor_field((0, 3))


class ContactForceSensor(
    RigidSensorMixin[ContactForceSensorMetadata],
    SimpleSensor[ContactForceSensorOptions, None, ContactForceSensorMetadata],
):
    """
    Sensor that returns the total contact force being applied to the associated RigidLink in its local frame.
    """

    def __init__(
        self, options: ContactForceSensorOptions, idx: int, shared_context, shared_metadata, manager: "SensorManager"
    ):
        super().__init__(options, idx, shared_context, shared_metadata, manager)

        self.debug_object: "Mesh" | None = None

    def build(self):
        super().build()

        if self._shared_metadata.solver is None:
            self._shared_metadata.solver = self._manager._sim.rigid_solver

        self._shared_metadata.min_force = concat_with_tensor(
            self._shared_metadata.min_force, self._options.min_force, expand=(1, 3)
        )
        self._shared_metadata.max_force = concat_with_tensor(
            self._shared_metadata.max_force, self._options.max_force, expand=(1, 3)
        )

        self._shared_metadata.append_filter(self._options.filter_link_idx)

    def _get_return_format(self) -> tuple[int, ...]:
        return (3,)

    @classmethod
    def _get_cache_dtype(cls) -> torch.dtype:
        return gs.tc_float

    @classmethod
    def _get_intermediate_dtype(cls) -> torch.dtype:
        # Required override because `_post_process` is overridden, even though shape and dtype coincide with return. The
        # intermediate buffer must be a distinct buffer (the timeline ring is in intermediate space).
        return cls._get_cache_dtype()

    @classmethod
    def _update_raw_data(
        cls, shared_context: None, shared_metadata: ContactForceSensorMetadata, raw_data_T: torch.Tensor
    ):
        assert shared_metadata.solver is not None

        # Note that forcing GPU sync to operate on `slice(0, max(n_contacts))` is usually faster overall.
        all_contacts = shared_metadata.solver.collider.get_contacts(as_tensor=True, to_torch=True)
        force, link_a, link_b = all_contacts["force"], all_contacts["link_a"], all_contacts["link_b"]
        if shared_metadata.solver.n_envs == 0:
            force, link_a, link_b = force[None], link_a[None], link_b[None]

        # Short-circuit if no contacts
        if link_a.shape[-1] == 0:
            raw_data_T.zero_()
            return

        links_quat = shared_metadata.solver.get_links_quat()
        if shared_metadata.solver.n_envs == 0:
            links_quat = links_quat[None]

        if gs.use_zerocopy:
            # Forces are aggregated BEFORE moving them in local frame for efficiency.
            force_mask_a = link_a[:, None] == shared_metadata.links_idx[None, :, None]
            force_mask_b = link_b[:, None] == shared_metadata.links_idx[None, :, None]
            _drop_filtered_counterpart_contacts(force_mask_a, force_mask_b, link_a, link_b, shared_metadata)
            force_mask = force_mask_b.to(dtype=gs.tc_float) - force_mask_a.to(dtype=gs.tc_float)
            sensors_force = (force_mask[..., None] * force[:, None]).sum(dim=2)
            sensors_quat = links_quat[:, shared_metadata.links_idx]
            n_envs = max(shared_metadata.solver.n_envs, 1)
            result = inv_transform_by_quat(sensors_force, sensors_quat)  # (B, n_sensors, 3)
            raw_data_T[:] = result.permute(1, 2, 0).reshape(-1, n_envs)
        else:
            raw_data_T.zero_()
            _kernel_get_contacts_forces(
                shared_metadata.links_idx,
                shared_metadata.filter_links_idx,
                force.contiguous(),
                link_a.contiguous(),
                link_b.contiguous(),
                links_quat.contiguous(),
                raw_data_T,
            )

    @classmethod
    def _post_process(
        cls, shared_metadata: ContactForceSensorMetadata, tensor: torch.Tensor, timeline, *, is_measured: bool
    ) -> torch.Tensor:
        # Saturate at max_force and zero out values below the min_force dead band. Applied after quantization (which
        # happens upstream in `_apply_hardware_imperfections`); for max_force values that are not multiples of
        # resolution this produces a non-quantized saturation value, accepted as minor drift in that edge case.
        per_sensor = tensor.reshape((tensor.shape[0], -1, 3))
        out = per_sensor.clamp(min=-shared_metadata.max_force, max=shared_metadata.max_force)
        out = out.masked_fill(out.abs() < shared_metadata.min_force, 0.0)
        return out.reshape(tensor.shape)

    def _draw_debug(self, context: "RasterizerContext"):
        """
        Draw debug arrow representing the contact force.

        Only draws for first rendered environment.
        """
        env_idx = context.rendered_envs_idx[0] if self._manager._sim.n_envs > 0 else None

        pos = self._link.get_pos(env_idx, relative=False).reshape((3,))
        quat = self._link.get_quat(env_idx, relative=False).reshape((4,))

        force = self.read(env_idx).reshape((3,))
        vec = tensor_to_array(transform_by_quat(force * self._options.debug_scale, quat))

        if self.debug_object is not None:
            context.clear_debug_object(self.debug_object)
            self.debug_object = None

        self.debug_object = context.draw_debug_arrow(pos=pos, vec=vec, color=self._options.debug_color)

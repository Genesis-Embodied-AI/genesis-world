from typing import TYPE_CHECKING

import torch

import quadrants as qd

import genesis as gs
from genesis.options.sensors import Contact as ContactSensorOptions
from genesis.options.sensors import ContactForce as ContactForceSensorOptions
from genesis.utils.geom import inv_transform_by_quat, qd_inv_transform_by_quat, transform_by_quat
from genesis.utils.misc import tensor_to_array

from .base_sensor import LinkAttachedSensorMixin, RigidSensorArrayMixin, SimpleSensor, SimpleSensorArray

if TYPE_CHECKING:
    from genesis.ext.pyrender.mesh import Mesh
    from genesis.utils.ring_buffer import TensorRingBuffer
    from genesis.vis.rasterizer_context import RasterizerContext


@qd.func
def _func_link_is_filtered(i_s: int, link: int, filter_links_idx: qd.types.ndarray()):
    """Return whether ``link`` is in the filter row of sensor ``i_s``."""
    is_filtered = False
    for i_f in range(filter_links_idx.shape[-1]):
        if filter_links_idx[i_s, i_f] == link:
            is_filtered = True
            break
    return is_filtered


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
    for i_c, i_s, i_b in qd.ndrange(link_a.shape[-1], sensors_link_idx.shape[-1], output.shape[0]):
        contact_data_link_a = link_a[i_b, i_c]
        contact_data_link_b = link_b[i_b, i_c]
        if contact_data_link_a == sensors_link_idx[i_s] or contact_data_link_b == sensors_link_idx[i_s]:
            j_s = i_s * 3

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
                    output[i_b, j_s + j] += force_a[j]
            if contact_data_link_b == sensors_link_idx[i_s] and not _func_link_is_filtered(
                i_s, contact_data_link_a, filter_links_idx
            ):
                for j in qd.static(range(3)):
                    output[i_b, j_s + j] += force_b[j]


class ContactFilterArrayMixin:
    """
    Array of sensors that scope contacts by counterpart link (see ``ContactFilterOptionsMixin``).

    ``filter_links_idx`` is a ``(n_sensors, max_num_filter_links)`` table. The row of each sensor lists its filter
    links, and the unused slots (the whole row for a sensor with no filter) hold ``-1``, so a kernel scans every column
    unconditionally. ``filtered_sensor_idx`` lists the rows that declared a filter, and the aggregation-path sensors
    (Contact, ContactForce) skip the per-contact comparison for the unfiltered majority. The contact-prefilter tactile
    sensors apply the filter in their build kernels instead.
    """

    def build(self) -> None:
        """Lay the filter links of every sensor as one table row each and list the rows that declared a filter."""
        super().build()

        filters = [tuple(sensor.options.filter_link_idx) for sensor in self._sensors]
        n_cols = max(len(links_idx) for links_idx in (*filters, ()))
        self.filter_links_idx = torch.full((len(filters), max(n_cols, 1)), -1, dtype=gs.tc_int, device=gs.device)
        for i_s, links_idx in enumerate(filters):
            self.filter_links_idx[i_s, : len(links_idx)] = torch.tensor(links_idx, dtype=gs.tc_int, device=gs.device)
        filtered_sensor_idx = [i_s for i_s, links_idx in enumerate(filters) if links_idx]
        self.filtered_sensor_idx = torch.tensor(filtered_sensor_idx, dtype=gs.tc_int, device=gs.device)

    def _drop_filtered_counterpart_contacts(
        self, is_a: torch.Tensor, is_b: torch.Tensor, link_a: torch.Tensor, link_b: torch.Tensor
    ) -> None:
        """
        Clear in place each per-side contact mask where the counterpart link is in the filter row of the sensor.

        Only the sensors listed in ``filtered_sensor_idx`` are touched, and the unfiltered majority keeps its masks.
        ``is_a`` and ``is_b`` are ``(B, n_sensors, n_contacts)`` bool masks. ``link_a`` and ``link_b`` are the ``(B,
        n_contacts)`` links of the contact participants.
        """
        filtered_sensors = self.filtered_sensor_idx
        if filtered_sensors.numel() == 0:
            return
        filter_rows = self.filter_links_idx[filtered_sensors][None, :, None, :]
        is_a[:, filtered_sensors, :] &= ~(link_b[:, None, :, None] == filter_rows).any(dim=-1)
        is_b[:, filtered_sensors, :] &= ~(link_a[:, None, :, None] == filter_rows).any(dim=-1)


class ContactSensorArray(ContactFilterArrayMixin, RigidSensorArrayMixin, SimpleSensorArray[ContactSensorOptions]):
    """Array of every contact sensor of the scene."""

    def build(self) -> None:
        super().build()

        # Per-sensor bool threshold (broadcast over B); _post_process returns `tensor > thresholds`
        self.thresholds = torch.tensor(
            [sensor.options.threshold for sensor in self._sensors], dtype=gs.tc_float, device=gs.device
        )
        self._debug_objects: list["Mesh | None"] = [None] * len(self._sensors)

    def _get_return_format(self, options: ContactSensorOptions) -> tuple[int, ...]:
        return (1,)

    def _get_cache_dtype(self) -> torch.dtype:
        return gs.tc_bool

    def _get_intermediate_dtype(self) -> torch.dtype:
        # The kernel counts contacts in float; `_post_process` projects the count to the bool return
        return gs.tc_float

    def _update_raw_data(self, raw_data: torch.Tensor) -> None:
        all_contacts = self.solver.collider.get_contacts(as_tensor=True, to_torch=True)
        link_a, link_b = all_contacts["link_a"], all_contacts["link_b"]
        if link_a.shape[-1] == 0:
            raw_data.zero_()
            return
        if self.solver.n_envs == 0:
            link_a, link_b = link_a[None], link_b[None]

        is_contact_a = link_a[..., None, :] == self.links_idx[..., None]
        is_contact_b = link_b[..., None, :] == self.links_idx[..., None]
        self._drop_filtered_counterpart_contacts(is_contact_a, is_contact_b, link_a, link_b)
        raw_data.copy_((is_contact_a | is_contact_b).sum(dim=-1))

    def _post_process(self, tensor: torch.Tensor, timeline: "TensorRingBuffer", *, is_measured: bool) -> torch.Tensor:
        return tensor > self.thresholds

    def _draw_debug(self, i_s: int, context: "RasterizerContext") -> None:
        """Draw a sphere at sensor ``i_s`` while it detects a contact, in the first rendered environment."""
        options = self._sensors[i_s].options
        env_idx = context.rendered_envs_idx[0] if self._sim.n_envs > 0 else None

        pos = self._links[i_s].get_pos(env_idx, relative=False).reshape((3,))
        is_contact = self.read(i_s, env_idx)

        if self._debug_objects[i_s] is not None:
            context.clear_debug_object(self._debug_objects[i_s])
            self._debug_objects[i_s] = None

        if is_contact:
            self._debug_objects[i_s] = context.draw_debug_sphere(
                pos=pos, radius=options.debug_sphere_radius, color=options.debug_color
            )


class ContactSensor(LinkAttachedSensorMixin, SimpleSensor[ContactSensorOptions, ContactSensorArray]):
    """Sensor reading whether its link is in contact, as a boolean."""


# ==========================================================================================================


class ContactForceSensorArray(
    ContactFilterArrayMixin, RigidSensorArrayMixin, SimpleSensorArray[ContactForceSensorOptions]
):
    """Array of every contact force sensor of the scene."""

    def build(self) -> None:
        super().build()

        self.min_force = torch.tensor(
            [sensor.options.min_force for sensor in self._sensors], dtype=gs.tc_float, device=gs.device
        )
        self.max_force = torch.tensor(
            [sensor.options.max_force for sensor in self._sensors], dtype=gs.tc_float, device=gs.device
        )
        self._debug_objects: list["Mesh | None"] = [None] * len(self._sensors)

    def _get_return_format(self, options: ContactForceSensorOptions) -> tuple[int, ...]:
        return (3,)

    def _get_cache_dtype(self) -> torch.dtype:
        return gs.tc_float

    def _update_raw_data(self, raw_data: torch.Tensor) -> None:
        # Note that forcing GPU sync to operate on `slice(0, max(n_contacts))` is usually faster overall.
        all_contacts = self.solver.collider.get_contacts(as_tensor=True, to_torch=True)
        force, link_a, link_b = all_contacts["force"], all_contacts["link_a"], all_contacts["link_b"]
        if self.solver.n_envs == 0:
            force, link_a, link_b = force[None], link_a[None], link_b[None]

        # Short-circuit if no contacts
        if link_a.shape[-1] == 0:
            raw_data.zero_()
            return

        links_quat = self.solver.get_links_quat()
        if self.solver.n_envs == 0:
            links_quat = links_quat[None]

        if gs.use_zerocopy:
            # Forces are aggregated BEFORE moving them in local frame for efficiency.
            force_mask_a = link_a[:, None] == self.links_idx[None, :, None]
            force_mask_b = link_b[:, None] == self.links_idx[None, :, None]
            self._drop_filtered_counterpart_contacts(force_mask_a, force_mask_b, link_a, link_b)
            force_mask = force_mask_b.to(dtype=gs.tc_float) - force_mask_a.to(dtype=gs.tc_float)
            sensors_force = (force_mask[..., None] * force[:, None]).sum(dim=2)
            sensors_quat = links_quat[:, self.links_idx]
            # (B, n_sensors, 3), laid one sensor after the other in the flat cache
            raw_data.copy_(inv_transform_by_quat(sensors_force, sensors_quat).reshape(raw_data.shape))
        else:
            raw_data.zero_()
            _kernel_get_contacts_forces(
                self.links_idx,
                self.filter_links_idx,
                force.contiguous(),
                link_a.contiguous(),
                link_b.contiguous(),
                links_quat.contiguous(),
                raw_data,
            )

    def _post_process(self, tensor: torch.Tensor, timeline: "TensorRingBuffer", *, is_measured: bool) -> torch.Tensor:
        # Saturate at max_force and zero out values below the min_force dead band. Applied after quantization (which
        # happens upstream in `_apply_hardware_imperfections`); for max_force values that are not multiples of
        # resolution this produces a non-quantized saturation value, accepted as minor drift in that edge case.
        per_sensor = tensor.reshape((tensor.shape[0], -1, 3))
        out = per_sensor.clamp(min=-self.max_force, max=self.max_force)
        out = out.masked_fill(out.abs() < self.min_force, 0.0)
        return out.reshape(tensor.shape)

    def _draw_debug(self, i_s: int, context: "RasterizerContext") -> None:
        """Draw the contact force arrow of sensor ``i_s``, in the first rendered environment."""
        options = self._sensors[i_s].options
        env_idx = context.rendered_envs_idx[0] if self._sim.n_envs > 0 else None

        link = self._links[i_s]
        pos = link.get_pos(env_idx, relative=False).reshape((3,))
        quat = link.get_quat(env_idx, relative=False).reshape((4,))

        force = self.read(i_s, env_idx).reshape((3,))
        vec = tensor_to_array(transform_by_quat(force * options.debug_scale, quat))

        if self._debug_objects[i_s] is not None:
            context.clear_debug_object(self._debug_objects[i_s])
            self._debug_objects[i_s] = None

        self._debug_objects[i_s] = context.draw_debug_arrow(pos=pos, vec=vec, color=options.debug_color)


class ContactForceSensor(LinkAttachedSensorMixin, SimpleSensor[ContactForceSensorOptions, ContactForceSensorArray]):
    """Sensor reading the total contact force applied to its link, in the link frame."""

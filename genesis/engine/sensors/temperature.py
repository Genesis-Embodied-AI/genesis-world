from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING

import numpy as np
import quadrants as qd
import torch
import trimesh

import genesis as gs
import genesis.utils.array_class as array_class
import genesis.utils.geom as gu
from genesis.options.sensors import TemperatureGrid as TemperatureGridOptions
from genesis.options.sensors import TemperatureProperties
from genesis.utils.misc import concat_with_tensor, make_tensor_field, tensor_to_array
from genesis.utils.ring_buffer import TensorRingBuffer

from .base_sensor import SimpleSensor, RigidSensorMetadataMixin, RigidSensorMixin, SimpleSensorMetadata

if TYPE_CHECKING:
    from genesis.engine.entities.rigid_entity.rigid_link import RigidLink
    from genesis.vis.rasterizer_context import RasterizerContext

    from .sensor_manager import SensorManager


STEFAN_BOLTZMANN = 5.670374419e-8  # W / (m²·K⁴)
KELVIN_OFFSET = 273.15
MAX_TEMP = 1000.0  # °C


class _PropIdx(IntEnum):
    BASE_TEMP = 0
    CONDUCTIVITY = 1
    EMISSIVITY = 2
    RHO_CP = 3


class _ScratchIdx(IntEnum):
    OTHER_LINK = 0
    CONTACT_IDX = 1
    DEPTH = 2
    POS_X = 3
    POS_Y = 4
    POS_Z = 5
    NORMAL_X = 6
    NORMAL_Y = 7
    NORMAL_Z = 8
    GROUP_CONTACT_IDX = 9
    GROUP_POS_X = 10
    GROUP_POS_Y = 11
    GROUP_POS_Z = 12
    GROUP_NORMAL_X = 13
    GROUP_NORMAL_Y = 14
    GROUP_NORMAL_Z = 15
    GROUP_DEPTH = 16
    GROUP_POS2_X = 17
    GROUP_POS2_Y = 18


@torch.jit.script
def _compute_K2_rfft3(
    nx: int, ny: int, nz: int, dx: float, dy: float, dz: float, device: torch.device, dtype: torch.dtype, eps: float
) -> torch.Tensor:
    """Squared wave numbers for 3D real FFT: K2[i,j,k] = (2*pi*kx)^2 + (2*pi*ky)^2 + (2*pi*kz)^2 with rfft layout."""
    kx = torch.fft.fftfreq(nx, d=dx, device=device).to(dtype)
    ky = torch.fft.fftfreq(ny, d=dy, device=device).to(dtype)
    kz = torch.fft.rfftfreq(nz, d=dz, device=device).to(dtype)
    K2 = (2 * torch.pi * kx).reshape(-1, 1, 1) ** 2
    K2 = K2 + (2 * torch.pi * ky).reshape(1, -1, 1) ** 2
    K2 = K2 + (2 * torch.pi * kz).reshape(1, 1, -1) ** 2
    K2[0, 0, 0] = max(K2[0, 0, 0], eps)
    # MPS silently ignores the device arg on fftfreq/rfftfreq and creates on CPU, so move explicitly.
    return K2.to(device=device)


@torch.jit.script
def _compute_surface_mask(nx: int, ny: int, nz: int, device: torch.device) -> torch.Tensor:
    """Boolean mask of boundary voxels (at least one face on grid boundary). Shape (nx, ny, nz)."""
    ix, iy, iz = torch.meshgrid(
        torch.arange(nx, device=device), torch.arange(ny, device=device), torch.arange(nz, device=device), indexing="ij"
    )
    return (ix == 0) | (ix == nx - 1) | (iy == 0) | (iy == ny - 1) | (iz == 0) | (iz == nz - 1)


@torch.jit.script
def _apply_diffusion_and_heat_generation(
    cache_sizes: list[int],
    grid_size: torch.Tensor,
    heat_generation: list[torch.Tensor | None],
    voxel_size: torch.Tensor,
    links_idx: torch.Tensor,
    link_to_material_idx: torch.Tensor,
    link_rho_cp: torch.Tensor,
    link_conductivity: torch.Tensor,
    K2_spectral: list[torch.Tensor],
    dt: float,
    eps: float,
    output: torch.Tensor,
) -> None:
    """Batched FFT semi-implicit diffusion with mirror padding (Neumann BC, no wrap-around)."""
    n_batches = output.shape[-1]
    start = 0
    for i_s in range(len(cache_sizes)):
        size = cache_sizes[i_s]
        nx, ny, nz = int(grid_size[i_s][0]), int(grid_size[i_s][1]), int(grid_size[i_s][2])
        mat_idx = link_to_material_idx[links_idx[i_s]]
        rcp = link_rho_cp[mat_idx]
        k = link_conductivity[mat_idx]
        alpha = k / rcp
        T = output[start : start + size].view(nx, ny, nz, n_batches)
        # Mirror-pad to (2*nx, 2*ny, 2*nz) for zero-flux (Neumann) boundaries; avoids FFT wrap-around.
        T_x = torch.cat([T, torch.flip(T, dims=(0,))], dim=0)
        T_xy = torch.cat([T_x, torch.flip(T_x, dims=(1,))], dim=1)
        T_pad = torch.cat([T_xy, torch.flip(T_xy, dims=(2,))], dim=2)
        T_hat = torch.fft.rfftn(T_pad, dim=(0, 1, 2))
        T_hat = T_hat / (1.0 + dt * alpha * K2_spectral[i_s].unsqueeze(-1))
        T_pad = torch.fft.irfftn(T_hat, s=(2 * nx, 2 * ny, 2 * nz), dim=(0, 1, 2))
        T = T_pad[:nx, :ny, :nz]
        output[start : start + size] = T.reshape(-1, n_batches)

        # Add internal heat generation (W/m² -> Q_vol = Q_surface / dz).
        q = heat_generation[i_s]
        if q is not None:
            dz = max(voxel_size[i_s, 2], eps)
            Q_vol = q.reshape(-1) / dz
            delta_T = dt * Q_vol / rcp
            output[start : start + size] += delta_T.unsqueeze(-1).expand(-1, n_batches)
        start += size


@qd.func
def _qd_polygon_area_from_points_3d(i_b: int, n: int, scratch: qd.types.ndarray(), eps: float) -> float:
    """Area of polygon from scratch buffer."""
    area = gs.qd_float(0.0)
    if n >= 3:
        cx = gs.qd_float(0.0)
        cy = gs.qd_float(0.0)
        cz = gs.qd_float(0.0)
        nx = gs.qd_float(0.0)
        ny = gs.qd_float(0.0)
        nz = gs.qd_float(0.0)
        for i in range(n):
            cx = cx + qd.cast(scratch[i_b, i, _ScratchIdx.GROUP_POS_X], gs.qd_float)
            cy = cy + qd.cast(scratch[i_b, i, _ScratchIdx.GROUP_POS_Y], gs.qd_float)
            cz = cz + qd.cast(scratch[i_b, i, _ScratchIdx.GROUP_POS_Z], gs.qd_float)
            nx = nx + qd.cast(scratch[i_b, i, _ScratchIdx.GROUP_NORMAL_X], gs.qd_float)
            ny = ny + qd.cast(scratch[i_b, i, _ScratchIdx.GROUP_NORMAL_Y], gs.qd_float)
            nz = nz + qd.cast(scratch[i_b, i, _ScratchIdx.GROUP_NORMAL_Z], gs.qd_float)
        n_inv = gs.qd_float(1.0) / gs.qd_float(n)
        cx, cy, cz = cx * n_inv, cy * n_inv, cz * n_inv
        nx, ny, nz = nx * n_inv, ny * n_inv, nz * n_inv
        n_norm = qd.sqrt(nx * nx + ny * ny + nz * nz) + eps
        nx, ny, nz = nx / n_norm, ny / n_norm, nz / n_norm
        ax = 0 if qd.abs(nx) < gs.qd_float(0.9) else 1
        ux = gs.qd_float(0.0)
        uy = gs.qd_float(0.0)
        uz = gs.qd_float(0.0)
        if ax == 0:
            ux = gs.qd_float(1.0)
        else:
            uy = gs.qd_float(1.0)
        dot = ux * nx + uy * ny + uz * nz
        ux, uy, uz = ux - dot * nx, uy - dot * ny, uz - dot * nz
        u_norm = qd.sqrt(ux * ux + uy * uy + uz * uz) + eps
        ux, uy, uz = ux / u_norm, uy / u_norm, uz / u_norm
        vx = ny * uz - nz * uy
        vy = nz * ux - nx * uz
        vz = nx * uy - ny * ux
        v_norm = qd.sqrt(vx * vx + vy * vy + vz * vz) + eps
        vx, vy, vz = vx / v_norm, vy / v_norm, vz / v_norm
        for i in range(n):
            rx = scratch[i_b, i, 10] - cx
            ry = scratch[i_b, i, 11] - cy
            rz = scratch[i_b, i, 12] - cz
            scratch[i_b, i, _ScratchIdx.GROUP_POS2_X] = rx * ux + ry * uy + rz * uz
            scratch[i_b, i, _ScratchIdx.GROUP_POS2_Y] = rx * vx + ry * vy + rz * vz
        for i in range(1, n):
            key_x = scratch[i_b, i, _ScratchIdx.GROUP_POS2_X]
            key_y = scratch[i_b, i, _ScratchIdx.GROUP_POS2_Y]
            j = i - 1
            key_angle = qd.atan2(key_y, key_x)
            while (
                j >= 0
                and qd.atan2(scratch[i_b, j, _ScratchIdx.GROUP_POS2_Y], scratch[i_b, j, _ScratchIdx.GROUP_POS2_X])
                > key_angle
            ):
                scratch[i_b, j + 1, _ScratchIdx.GROUP_POS2_X] = scratch[i_b, j, _ScratchIdx.GROUP_POS2_X]
                scratch[i_b, j + 1, _ScratchIdx.GROUP_POS2_Y] = scratch[i_b, j, _ScratchIdx.GROUP_POS2_Y]
                j = j - 1
            scratch[i_b, j + 1, _ScratchIdx.GROUP_POS2_X] = key_x
            scratch[i_b, j + 1, _ScratchIdx.GROUP_POS2_Y] = key_y
        for i in range(n):
            i_next = (i + 1) % n
            area = (
                area
                + scratch[i_b, i, _ScratchIdx.GROUP_POS2_X] * scratch[i_b, i_next, _ScratchIdx.GROUP_POS2_Y]
                - scratch[i_b, i_next, _ScratchIdx.GROUP_POS2_X] * scratch[i_b, i, _ScratchIdx.GROUP_POS2_Y]
            )
        area = qd.abs(area) * gs.qd_float(0.5)

    return area


@qd.kernel
def _kernel_compute_contact_areas(
    contact_area: qd.types.ndarray(),
    scratch: qd.types.ndarray(),
    dyn_state: array_class.DynState,
    collider_state: array_class.ColliderState,
    eps: float,
):
    # contact_area shape (n_c_max, n_batches). scratch (n_batches, n_c_max, len(_ScratchIdx)).
    n_batches = contact_area.shape[1]
    for i_b in range(n_batches):
        n_c = collider_state.n_contacts[i_b]
        for i_c in range(n_c):
            i_col = collider_state.contact_sort_idx[i_c, i_b]
            la = collider_state.contact_data.link_a[i_col, i_b]
            lb = collider_state.contact_data.link_b[i_col, i_b]
            scratch[i_b, i_c, _ScratchIdx.OTHER_LINK] = gs.qd_float(lb)
            scratch[i_b, i_c, _ScratchIdx.CONTACT_IDX] = gs.qd_float(i_c)
            scratch[i_b, i_c, _ScratchIdx.DEPTH] = collider_state.contact_data.penetration[i_col, i_b]
            p_world = collider_state.contact_data.pos[i_col, i_b]
            link_pos = dyn_state.links.pos[la, i_b]
            link_quat = dyn_state.links.quat[la, i_b]
            p_local = gu.qd_inv_transform_by_trans_quat(p_world, link_pos, link_quat)
            scratch[i_b, i_c, _ScratchIdx.POS_X] = p_local.x
            scratch[i_b, i_c, _ScratchIdx.POS_Y] = p_local.y
            scratch[i_b, i_c, _ScratchIdx.POS_Z] = p_local.z
            n_w = collider_state.contact_data.normal[i_col, i_b]
            scratch[i_b, i_c, _ScratchIdx.NORMAL_X] = n_w.x
            scratch[i_b, i_c, _ScratchIdx.NORMAL_Y] = n_w.y
            scratch[i_b, i_c, _ScratchIdx.NORMAL_Z] = n_w.z

        for i_c in range(n_c):
            i_col = collider_state.contact_sort_idx[i_c, i_b]
            la = collider_state.contact_data.link_a[i_col, i_b]
            lb = collider_state.contact_data.link_b[i_col, i_b]
            is_first = True
            for k in range(i_c):
                k_phys = collider_state.contact_sort_idx[k, i_b]
                la_k = collider_state.contact_data.link_a[k_phys, i_b]
                lb_k = collider_state.contact_data.link_b[k_phys, i_b]
                if la_k == la and lb_k == lb:
                    is_first = False
            if not is_first:
                continue

            count = 0
            for j in range(n_c):
                j_phys = collider_state.contact_sort_idx[j, i_b]
                la_j = collider_state.contact_data.link_a[j_phys, i_b]
                lb_j = collider_state.contact_data.link_b[j_phys, i_b]
                if la_j == la and lb_j == lb:
                    scratch[i_b, count, _ScratchIdx.GROUP_CONTACT_IDX] = scratch[i_b, j, _ScratchIdx.CONTACT_IDX]
                    scratch[i_b, count, _ScratchIdx.GROUP_POS_X] = scratch[i_b, j, _ScratchIdx.POS_X]
                    scratch[i_b, count, _ScratchIdx.GROUP_POS_Y] = scratch[i_b, j, _ScratchIdx.POS_Y]
                    scratch[i_b, count, _ScratchIdx.GROUP_POS_Z] = scratch[i_b, j, _ScratchIdx.POS_Z]
                    scratch[i_b, count, _ScratchIdx.GROUP_NORMAL_X] = scratch[i_b, j, _ScratchIdx.NORMAL_X]
                    scratch[i_b, count, _ScratchIdx.GROUP_NORMAL_Y] = scratch[i_b, j, _ScratchIdx.NORMAL_Y]
                    scratch[i_b, count, _ScratchIdx.GROUP_NORMAL_Z] = scratch[i_b, j, _ScratchIdx.NORMAL_Z]
                    scratch[i_b, count, _ScratchIdx.GROUP_DEPTH] = scratch[i_b, j, _ScratchIdx.DEPTH]
                    count = count + 1

            group_area = eps
            if count >= 3:
                group_area = _qd_polygon_area_from_points_3d(i_b, count, scratch, eps)
            else:
                for k in range(count):
                    d = scratch[i_b, k, _ScratchIdx.GROUP_DEPTH]
                    group_area = group_area + d * qd.cast(qd.math.pi, gs.qd_float)

            area_per_contact = group_area / (gs.qd_float(count) + eps)
            for k in range(count):
                contact_idx = gs.qd_int(scratch[i_b, k, _ScratchIdx.GROUP_CONTACT_IDX])
                contact_area[contact_idx, i_b] = area_per_contact


@qd.func
def _qd_k_eff(k_a: float, k_b: float, eps: float) -> float:
    """Effective conductivity for series thermal resistance: 2*k_a*k_b/(k_a+k_b+eps)."""
    return gs.qd_float(2.0) * k_a * k_b / (k_a + k_b + eps)


@qd.kernel
def _kernel_contact_heat(
    links_idx: qd.types.ndarray(),
    sensor_cache_start: qd.types.ndarray(),
    link_to_material_idx: qd.types.ndarray(),
    aabb_min: qd.types.ndarray(),
    grid_size: qd.types.ndarray(),
    voxel_size: qd.types.ndarray(),
    voxel_volume: qd.types.ndarray(),
    depth_weight: qd.types.ndarray(),
    link_temps: qd.types.ndarray(),
    link_volume: qd.types.ndarray(),
    link_base_temperature: qd.types.ndarray(),
    link_conductivity: qd.types.ndarray(),
    link_rho_cp: qd.types.ndarray(),
    contact_area: qd.types.ndarray(),
    output: qd.types.ndarray(),
    dyn_state: array_class.DynState,
    collider_state: array_class.ColliderState,
    dt: float,
    eps: float,
):
    # contact_area shape (n_c_max, n_batches)
    n_batches = output.shape[-1]
    n_sensors = links_idx.shape[0]
    use_link_temps = link_temps.shape[0] > 0

    # Grid update: only for contacts that involve a sensorized link; use contact_area[i_c, i_b]
    for i_s, i_b in qd.ndrange(n_sensors, n_batches):
        sensor_link_idx = links_idx[i_s]
        dw = depth_weight[i_s]
        start = sensor_cache_start[i_s]
        nx = grid_size[i_s, 0]
        ny = grid_size[i_s, 1]
        nz = grid_size[i_s, 2]
        vol = voxel_volume[i_s] + eps
        mat_idx_sensor = link_to_material_idx[sensor_link_idx]
        if mat_idx_sensor < 0:
            continue
        rcp = link_rho_cp[mat_idx_sensor] + eps

        k_sensor = link_conductivity[mat_idx_sensor]
        amin = qd.math.vec3(aabb_min[i_s, 0], aabb_min[i_s, 1], aabb_min[i_s, 2])
        vs = qd.math.vec3(voxel_size[i_s, 0] + eps, voxel_size[i_s, 1] + eps, voxel_size[i_s, 2] + eps)
        n_c = collider_state.n_contacts[i_b]
        for i_c in range(n_c):
            i_col = collider_state.contact_sort_idx[i_c, i_b]
            la = collider_state.contact_data.link_a[i_col, i_b]
            lb = collider_state.contact_data.link_b[i_col, i_b]
            if la != sensor_link_idx and lb != sensor_link_idx:
                continue
            other_link = lb if la == sensor_link_idx else la
            mat_other = link_to_material_idx[other_link]
            if mat_other >= 0:
                T_other = link_base_temperature[mat_other]
                if use_link_temps:
                    T_other = link_temps[i_b, other_link]
                k_other = link_conductivity[mat_other]
                k_eff = _qd_k_eff(k_sensor, k_other, eps)
                p_world = collider_state.contact_data.pos[i_col, i_b]
                link_pos = dyn_state.links.pos[sensor_link_idx, i_b]
                link_quat = dyn_state.links.quat[sensor_link_idx, i_b]
                p_local = gu.qd_inv_transform_by_trans_quat(p_world, link_pos, link_quat)
                u_x = (p_local.x - amin.x) / vs.x
                u_y = (p_local.y - amin.y) / vs.y
                u_z = (p_local.z - amin.z) / vs.z
                ix = min(max(0, int(u_x)), nx - 1)
                iy = min(max(0, int(u_y)), ny - 1)
                iz = min(max(0, int(u_z)), nz - 1)
                cell_idx = ix * (ny * nz) + iy * nz + iz
                T_cell = output[start + cell_idx, i_b]
                area_base = contact_area[i_c, i_b] + eps
                area = qd.max(
                    area_base,
                    qd.cast(qd.math.pi, gs.qd_float) * dw * collider_state.contact_data.penetration[i_col, i_b],
                )
                flux = k_eff * (T_other - T_cell) / (vol / area + eps)
                Q_vol = flux * area / vol
                delta_T = dt * Q_vol / rcp
                output[start + cell_idx, i_b] = T_cell + delta_T

    # Link temps update for all contacts (both links) when use_link_temps
    if use_link_temps:
        for i_b in range(n_batches):
            n_c = collider_state.n_contacts[i_b]
            for i_c in range(n_c):
                i_col = collider_state.contact_sort_idx[i_c, i_b]
                la = collider_state.contact_data.link_a[i_col, i_b]
                lb = collider_state.contact_data.link_b[i_col, i_b]
                mat_la = link_to_material_idx[la]
                mat_lb = link_to_material_idx[lb]
                if mat_la < 0 or mat_lb < 0:
                    continue
                T_la = link_temps[i_b, la]
                T_lb = link_temps[i_b, lb]
                k_la = link_conductivity[mat_la] + eps
                k_lb = link_conductivity[mat_lb] + eps
                k_eff = _qd_k_eff(k_la, k_lb, eps)
                area = contact_area[i_c, i_b] + eps
                vol_la = link_volume[la] + eps
                vol_lb = link_volume[lb] + eps
                length_scale = (vol_la + vol_lb) / (gs.qd_float(2.0) * area)
                flux = k_eff * (T_la - T_lb) / length_scale
                power = flux * area
                rcp_vol_la = link_rho_cp[mat_la] * vol_la + eps
                rcp_vol_lb = link_rho_cp[mat_lb] * vol_lb + eps
                delta_T_la = gs.qd_float(-1.0) * dt * power / rcp_vol_la
                delta_T_lb = dt * power / rcp_vol_lb
                link_temps[i_b, la] = link_temps[i_b, la] + delta_T_la
                link_temps[i_b, lb] = link_temps[i_b, lb] + delta_T_lb


def _radiation_convection_delta_T(
    T: torch.Tensor,
    emissivity: torch.Tensor | float,
    convection_coeff: float,
    ambient_temp: float,
    rho_cp_vol: torch.Tensor | float,
    dt: float,
) -> torch.Tensor:
    """Temperature change (to subtract) from radiation + convection: -dt * (q_rad + q_conv) / (rho_cp * vol)."""
    T_K = T + KELVIN_OFFSET
    T_amb_K = ambient_temp + KELVIN_OFFSET
    q_rad = emissivity * STEFAN_BOLTZMANN * (T_K**4 - T_amb_K**4)
    q_conv = convection_coeff * (T - ambient_temp)
    return dt * (q_rad + q_conv) / (rho_cp_vol + gs.EPS)


def _apply_radiation_convection(
    cache_sizes: list[int],
    sensor_surface_mask: list[torch.Tensor],
    voxel_volume: torch.Tensor,
    links_idx: torch.Tensor,
    link_temps: torch.Tensor,
    link_volume: torch.Tensor,
    link_to_material_idx: torch.Tensor,
    link_emissivity: torch.Tensor,
    link_rho_cp: torch.Tensor,
    ambient_temp: float,
    convection_coeff: float,
    dt: float,
    output: torch.Tensor,
) -> None:
    """Radiation + convection on surface voxels and (when allocated) on link temperatures.
    For link_temps, links with link_to_material_idx == -1 are treated as material index 0 (default properties) for
    emissivity/rho_cp; only links with valid material are updated.
    """
    start = 0
    for i_s in range(len(cache_sizes)):
        size = cache_sizes[i_s]
        mask = sensor_surface_mask[i_s].reshape(-1)
        vol = max(voxel_volume[i_s].item(), gs.EPS)
        mat_idx = link_to_material_idx[links_idx[i_s]]
        emiss = link_emissivity[mat_idx].item()
        rcp = link_rho_cp[mat_idx].item()
        denom = rcp * vol
        T_flat = output[start : start + size]
        delta = _radiation_convection_delta_T(T_flat, emiss, convection_coeff, ambient_temp, denom, dt)
        output[start : start + size] -= delta * mask.unsqueeze(-1)
        start += size

    if link_temps.numel() > 0:
        valid = link_to_material_idx >= 0  # (n_links,)
        mat_idx = link_to_material_idx.clamp(min=0)  # -1 -> 0 (default material) for indexing
        rcp_vol = link_rho_cp[mat_idx] * link_volume  # (n_links,)
        delta = _radiation_convection_delta_T(
            link_temps, link_emissivity[mat_idx], convection_coeff, ambient_temp, rcp_vol.unsqueeze(0), dt
        )
        link_temps.sub_(delta * valid.unsqueeze(0).to(gs.tc_float))


def _apply_T_measured_filter(
    cache_sizes: list[int],
    sensor_time_const: torch.Tensor,
    dt: float,
    T_actual: torch.Tensor,
    T_measured: torch.Tensor,
) -> None:
    """T_measured += (dt/tau)*(T - T_measured); if tau<=0 then T_measured = T. Batched over envs."""
    start = 0
    for i_s in range(len(cache_sizes)):
        size = cache_sizes[i_s]
        tau = sensor_time_const[i_s].item()
        T_slice = T_actual[:, start : start + size]
        T_meas_slice = T_measured[:, start : start + size]
        if tau > 0:
            alpha = dt / tau
            T_measured[:, start : start + size] = T_meas_slice + alpha * (T_slice - T_meas_slice)
        else:
            T_measured[:, start : start + size] = T_slice
        start += size


@dataclass
class TemperatureGridSensorMetadata(RigidSensorMetadataMixin, SimpleSensorMetadata):
    """Shared metadata for all temperature grid sensors."""

    ambient_temperature: float = 21.0
    convection_coeff: float = 1.0
    link_to_material_idx: torch.Tensor = make_tensor_field((0,), dtype_factory=lambda: gs.tc_int)
    link_material_properties: torch.Tensor = make_tensor_field((0, len(_PropIdx)), dtype_factory=lambda: gs.tc_float)
    properties_dict: dict[int, TemperatureProperties] = field(default_factory=dict)
    simulate_all_link_temps: bool = False
    link_temps: torch.Tensor = make_tensor_field((0, 0))
    link_volume: torch.Tensor = make_tensor_field((0,))

    aabb_min: torch.Tensor = make_tensor_field((0, 3))
    aabb_extent: torch.Tensor = make_tensor_field((0, 3))
    grid_size: torch.Tensor = make_tensor_field((0, 3), dtype_factory=lambda: gs.tc_int)
    voxel_size: torch.Tensor = make_tensor_field((0, 3))
    voxel_volume: torch.Tensor = make_tensor_field((0,))
    sensor_time_const: torch.Tensor = make_tensor_field((0,))
    contact_depth_weight: torch.Tensor = make_tensor_field((0,))
    K2_spectral: list[torch.Tensor] = field(default_factory=list)
    sensor_surface_mask: list[torch.Tensor] = field(default_factory=list)
    heat_generation: list[torch.Tensor | None] = field(default_factory=list)
    contact_area_scratch: torch.Tensor = make_tensor_field((0, len(_ScratchIdx)))
    contact_area_buffer: torch.Tensor = make_tensor_field((0, 0))

    sensor_cache_start: torch.Tensor = make_tensor_field((0,), dtype_factory=lambda: gs.tc_int)


class TemperatureGridSensor(
    RigidSensorMixin[TemperatureGridSensorMetadata],
    SimpleSensor[TemperatureGridOptions, None, TemperatureGridSensorMetadata, TemperatureGridSensorMetadata],
):
    def __init__(
        self, options: TemperatureGridOptions, idx: int, shared_context, shared_metadata, manager: "SensorManager"
    ):
        super().__init__(options, idx, shared_context, shared_metadata, manager)

        self._link: "RigidLink | None" = None
        self._debug_objects: list = []
        self._debug_t_min: float = self._options.debug_temperature_range[0]
        self._debug_t_range: float = self._options.debug_temperature_range[1] - self._debug_t_min
        self._debug_cell_local_positions: np.ndarray = np.array([])  # set in build

    def build(self):
        super().build()

        solver = self._shared_metadata.solver

        # Same for all sensors
        if self._options.ambient_temperature is not None:
            self._shared_metadata.ambient_temperature = self._options.ambient_temperature
        if self._options.convection_coefficient is not None:
            self._shared_metadata.convection_coeff = self._options.convection_coefficient

        if self._shared_metadata.link_to_material_idx.shape[0] == 0:
            self._shared_metadata.link_to_material_idx = torch.full(
                (solver.n_links,), -1, dtype=gs.tc_int, device=gs.device
            )
        self._shared_metadata.properties_dict.update(self._options.properties_dict)
        if len(self._shared_metadata.properties_dict) > len(self._shared_metadata.link_material_properties):
            self._shared_metadata.link_material_properties = torch.empty(
                (len(_PropIdx), len(self._shared_metadata.properties_dict)), dtype=gs.tc_float, device=gs.device
            )
            # -1 in link_to_material_idx means invalid, 0 uses the default properties
            self._shared_metadata.link_to_material_idx[:] = 0 if -1 in self._shared_metadata.properties_dict else -1
            # sort properties_dict by link index to ensure default properties are at index 0
            for i, (prop_idx, props) in enumerate(
                sorted(self._shared_metadata.properties_dict.items(), key=lambda x: x[0])
            ):
                self._shared_metadata.link_material_properties[:, i] = torch.tensor(
                    # order should match _PropIdx
                    [props.base_temperature, props.conductivity, props.emissivity, props.density * props.specific_heat],
                    dtype=gs.tc_float,
                    device=gs.device,
                )
                if prop_idx >= 0:
                    self._shared_metadata.link_to_material_idx[prop_idx] = i
        assert self._link.idx in self._shared_metadata.properties_dict or -1 in self._shared_metadata.properties_dict, (
            f"Temperature properties for the attached link index {self._link.idx} should be provided"
            " in properties_dict, or use key -1 for default properties for all links."
        )
        if self._options.simulate_all_link_temperatures:
            self._shared_metadata.simulate_all_link_temps = True
            if len(self._shared_metadata.link_temps) == 0:
                self._shared_metadata.link_temps = torch.empty(
                    (solver._B, solver.n_links), dtype=gs.tc_float, device=gs.device
                )
                self._shared_metadata.link_volume = torch.empty(solver.n_links, dtype=gs.tc_float, device=gs.device)

                link_volume = self._shared_metadata.link_volume
                for entity in solver._entities:
                    for link in entity.links:
                        li = link.idx
                        if link.n_geoms > 0:
                            aabb = link.get_AABB()
                            if aabb.ndim == 3:
                                aabb = aabb[0]
                            vol = (aabb[1] - aabb[0]).prod().clamp(min=gs.EPS)
                            link_volume[li] = vol
                ambient_T = self._shared_metadata.ambient_temperature
                link_base_T = self._shared_metadata.link_material_properties[_PropIdx.BASE_TEMP]
                link_to_mat = self._shared_metadata.link_to_material_idx
                base_T_per_link = torch.where(
                    link_to_mat >= 0,
                    link_base_T[link_to_mat],
                    torch.tensor(ambient_T, dtype=gs.tc_float, device=gs.device),
                )
                n_batches = solver._B
                self._shared_metadata.link_temps.copy_(base_T_per_link.unsqueeze(0).expand(n_batches, -1))

        # Per-sensor properties
        assert self._link is not None
        aabb_world = self._link.get_AABB()
        if aabb_world.ndim == 2:
            aabb_world = aabb_world.unsqueeze(0)  # (1, 2, 3)
        aabb_min_w = aabb_world[0, 0]  # (3,)
        aabb_max_w = aabb_world[0, 1]  # (3,)
        link_pos, link_quat = self._link.get_pos(relative=False), self._link.get_quat(relative=False)
        if link_pos.ndim == 2:
            link_pos, link_quat = link_pos[0], link_quat[0]
        aabb_min_local = gu.inv_transform_by_trans_quat(aabb_min_w, link_pos, link_quat)
        aabb_max_local = gu.inv_transform_by_trans_quat(aabb_max_w, link_pos, link_quat)
        aabb_extent = (aabb_max_local - aabb_min_local).reshape(3)
        self._shared_metadata.aabb_min = concat_with_tensor(
            self._shared_metadata.aabb_min, aabb_min_local, expand=(1, 3), dim=0
        )
        self._shared_metadata.aabb_extent = concat_with_tensor(
            self._shared_metadata.aabb_extent, aabb_extent, expand=(1, 3), dim=0
        )
        grid_size_tensor = torch.tensor(self._options.grid_size, dtype=gs.tc_int, device=gs.device)
        self._shared_metadata.grid_size = concat_with_tensor(
            self._shared_metadata.grid_size, grid_size_tensor, expand=(1, 3), dim=0
        )
        voxel_size = aabb_extent / grid_size_tensor
        self._shared_metadata.voxel_size = concat_with_tensor(
            self._shared_metadata.voxel_size, voxel_size, expand=(1, 3), dim=0
        )
        self._shared_metadata.voxel_volume = concat_with_tensor(
            self._shared_metadata.voxel_volume, voxel_size.prod(), expand=(1,), dim=0
        )
        self._shared_metadata.sensor_time_const = concat_with_tensor(
            self._shared_metadata.sensor_time_const, self._options.sensor_time_constant, expand=(1,), dim=0
        )
        self._shared_metadata.contact_depth_weight = concat_with_tensor(
            self._shared_metadata.contact_depth_weight, self._options.contact_depth_weight, expand=(1,), dim=0
        )

        dx, dy, dz = voxel_size.tolist()
        nx, ny, nz = grid_size_tensor.tolist()

        xs = torch.arange(nx, device=gs.device, dtype=gs.tc_float) + 0.5
        ys = torch.arange(ny, device=gs.device, dtype=gs.tc_float) + 0.5
        zs = torch.arange(nz, device=gs.device, dtype=gs.tc_float) + 0.5
        grid = torch.stack(torch.meshgrid(xs, ys, zs, indexing="ij"), dim=-1).reshape(-1, 3)
        self._debug_cell_local_positions = (aabb_min_local.unsqueeze(0) + grid * voxel_size.unsqueeze(0)).cpu().numpy()

        K2_padded = _compute_K2_rfft3(nx * 2, ny * 2, nz * 2, dx, dy, dz, gs.device, gs.tc_float, gs.EPS)
        self._shared_metadata.K2_spectral.append(K2_padded)

        surface_mask = _compute_surface_mask(nx, ny, nz, gs.device).to(gs.tc_float)
        self._shared_metadata.sensor_surface_mask.append(surface_mask)

        if self._options.heat_generation is not None:
            q = torch.tensor(self._options.heat_generation, dtype=gs.tc_float, device=gs.device)
            if q.shape != (nx, ny, nz):
                raise ValueError(f"heat_generation shape {tuple(q.shape)} does not match grid_size ({nx}, {ny}, {nz})")
            self._shared_metadata.heat_generation.append(q)
        else:
            self._shared_metadata.heat_generation.append(None)

        current_cache_start = sum(self._shared_metadata.cache_sizes[:-1]) if self._shared_metadata.cache_sizes else 0
        self._shared_metadata.sensor_cache_start = concat_with_tensor(
            self._shared_metadata.sensor_cache_start, current_cache_start, expand=(1,), dim=0
        )

        # Contact area buffers
        n_c_max = int(solver.collider.collider_info.max_candidate_contacts[None])
        self._shared_metadata.contact_area_buffer = torch.zeros(
            (n_c_max, solver._B), device=gs.device, dtype=gs.tc_float
        )
        self._shared_metadata.contact_area_scratch = torch.empty(
            (solver._B, n_c_max, len(_ScratchIdx)), device=gs.device, dtype=gs.tc_float
        )

    def _get_return_format(self) -> tuple[int, ...]:
        return (self._options.grid_size,)

    @classmethod
    def _get_cache_dtype(cls) -> torch.dtype:
        return gs.tc_float

    @classmethod
    def reset(cls, shared_metadata: TemperatureGridSensorMetadata, current_ground_truth_data_T: torch.Tensor, envs_idx):
        super().reset(shared_metadata, current_ground_truth_data_T, envs_idx)
        sensors_mat_idx = shared_metadata.link_to_material_idx[shared_metadata.links_idx]
        sensors_base_T = shared_metadata.link_material_properties[_PropIdx.BASE_TEMP][sensors_mat_idx]
        start = 0
        for i_s, size in enumerate(shared_metadata.cache_sizes):
            current_ground_truth_data_T[start : start + size, envs_idx] = sensors_base_T[i_s]
            start += size
        if shared_metadata.link_temps.numel() > 0:
            ambient_T = shared_metadata.ambient_temperature
            link_base_T = shared_metadata.link_material_properties[_PropIdx.BASE_TEMP]
            link_to_mat = shared_metadata.link_to_material_idx
            base_T_per_link = torch.where(
                link_to_mat >= 0,
                link_base_T[link_to_mat],
                torch.tensor(ambient_T, dtype=gs.tc_float, device=shared_metadata.link_temps.device),
            )
            n_envs = envs_idx.shape[0]
            shared_metadata.link_temps[envs_idx, :] = base_T_per_link.unsqueeze(0).expand(n_envs, -1)

    @classmethod
    def _update_raw_data(
        cls, shared_context: None, shared_metadata: TemperatureGridSensorMetadata, raw_data_T: torch.Tensor
    ):
        solver = shared_metadata.solver
        dt = solver._sim.dt
        props = shared_metadata.link_material_properties
        link_conductivity = props[_PropIdx.CONDUCTIVITY]
        link_base_temperature = props[_PropIdx.BASE_TEMP]
        link_emissivity = props[_PropIdx.EMISSIVITY]
        link_rho_cp = props[_PropIdx.RHO_CP]

        # 1) Batched FFT semi-implicit diffusion + 2) Heat generation
        _apply_diffusion_and_heat_generation(
            shared_metadata.cache_sizes,
            shared_metadata.grid_size,
            shared_metadata.heat_generation,
            shared_metadata.voxel_size,
            shared_metadata.links_idx,
            shared_metadata.link_to_material_idx,
            link_rho_cp,
            link_conductivity,
            shared_metadata.K2_spectral,
            dt,
            gs.EPS,
            raw_data_T,
        )
        # 3) Contact heat transfer
        collider_state = solver.collider.collider_state
        shared_metadata.contact_area_buffer.zero_()
        _kernel_compute_contact_areas(
            shared_metadata.contact_area_buffer,
            shared_metadata.contact_area_scratch,
            solver.dyn_state,
            collider_state,
            gs.EPS,
        )
        _kernel_contact_heat(
            shared_metadata.links_idx,
            shared_metadata.sensor_cache_start,
            shared_metadata.link_to_material_idx,
            shared_metadata.aabb_min,
            shared_metadata.grid_size,
            shared_metadata.voxel_size,
            shared_metadata.voxel_volume,
            shared_metadata.contact_depth_weight,
            shared_metadata.link_temps,
            shared_metadata.link_volume,
            link_base_temperature,
            link_conductivity,
            link_rho_cp,
            shared_metadata.contact_area_buffer,
            raw_data_T,
            solver.dyn_state,
            collider_state,
            dt,
            gs.EPS,
        )
        raw_data_T.clamp_(-MAX_TEMP, MAX_TEMP)
        # 4) Radiation and convection
        _apply_radiation_convection(
            shared_metadata.cache_sizes,
            shared_metadata.sensor_surface_mask,
            shared_metadata.voxel_volume,
            shared_metadata.links_idx,
            shared_metadata.link_temps,
            shared_metadata.link_volume,
            shared_metadata.link_to_material_idx,
            link_emissivity,
            link_rho_cp,
            shared_metadata.ambient_temperature,
            shared_metadata.convection_coeff,
            dt,
            raw_data_T,
        )

    @classmethod
    def _apply_transform(
        cls,
        shared_metadata: TemperatureGridSensorMetadata,
        data: torch.Tensor,
        timeline: "TensorRingBuffer",
        *,
        is_measured: bool,
    ):
        # First-order RC filter modelling the sensor element's thermal response time. The thermal mass is a property of
        # the sensor element only, so the filter is measured-only - ground truth exposes the raw simulated temperature
        # unchanged so that `read_ground_truth()` returns the underlying physical phenomenon. `data IS timeline.at(0)`
        # (the measured ring slot 0), pre-populated with the current raw temperature by `_update_current_timestep_data`;
        # the previous filtered value lives in `timeline.at(1)`.
        if not is_measured:
            return
        raw = data.clone()
        previous = timeline.at(1).clone()
        _apply_T_measured_filter(
            shared_metadata.cache_sizes,
            shared_metadata.sensor_time_const,
            shared_metadata.solver._sim.dt,
            raw,
            previous,
        )
        data.copy_(previous)

    def _draw_debug(self, context: "RasterizerContext"):
        """
        Draw a single flat mesh colored by temperature (cool=blue, hot=red).

        Only draws for the first rendered environment.
        """
        env_idx = context.rendered_envs_idx[0] if self._manager._sim.n_envs > 0 else None
        if self._link is None:
            return

        for obj in self._debug_objects:
            if obj is not None:
                context.clear_debug_object(obj)
        self._debug_objects = []

        link_pos = self._link.get_pos(env_idx, relative=False)
        link_quat = self._link.get_quat(env_idx, relative=False)
        link_pos = tensor_to_array(link_pos).reshape(3)
        link_quat = tensor_to_array(link_quat).reshape(4)
        link_T = gu.trans_quat_to_T(link_pos, link_quat)

        voxel_size = tensor_to_array(self._shared_metadata.voxel_size[self._idx]).reshape(3)

        # Per-cell color from temperature (blue=cool, red=hot)
        temps = self.read_ground_truth(env_idx)
        temps = tensor_to_array(temps).reshape(-1)
        t_min, t_range = self._debug_t_min, self._debug_t_range
        if t_range <= 0:
            t_range = 1.0
        norm = np.clip((temps - t_min) / t_range, 0.0, 1.0)
        colors_rgba = np.column_stack((norm, np.zeros_like(norm), 1.0 - norm, np.full_like(norm, 0.5)))

        # Build a single mesh: one quad (2 triangles) per cell on the top face
        n_cells = len(self._debug_cell_local_positions)
        hx, hy, hz = voxel_size[0] / 2, voxel_size[1] / 2, voxel_size[2] / 2
        quad_offsets = np.array([[-hx, -hy, hz], [hx, -hy, hz], [hx, hy, hz], [-hx, hy, hz]])
        vertices = (self._debug_cell_local_positions[:, np.newaxis, :] + quad_offsets[np.newaxis, :, :]).reshape(-1, 3)

        idx = np.arange(n_cells, dtype=np.int64) * 4
        faces = np.empty((n_cells * 2, 3), dtype=np.int64)
        faces[0::2] = np.column_stack([idx, idx + 1, idx + 2])
        faces[1::2] = np.column_stack([idx, idx + 2, idx + 3])

        face_colors_u8 = np.empty((n_cells * 2, 4), dtype=np.uint8)
        face_colors_u8[0::2] = (colors_rgba * 255).astype(np.uint8)
        face_colors_u8[1::2] = face_colors_u8[0::2]

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, face_colors=face_colors_u8)
        self._debug_objects.append(context.draw_debug_mesh(mesh, T=link_T))

    @property
    def link_temperatures(self) -> torch.Tensor:
        return self._shared_metadata.link_temps

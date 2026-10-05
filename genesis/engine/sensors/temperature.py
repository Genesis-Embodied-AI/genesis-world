from enum import IntEnum
from typing import TYPE_CHECKING

import numpy as np
import torch

import trimesh

import quadrants as qd

import genesis as gs
import genesis.utils.array_class as array_class
import genesis.utils.geom as gu
from genesis.options.sensors import TemperatureGrid as TemperatureGridOptions
from genesis.options.sensors import TemperatureProperties
from genesis.typing import EnvsIdxType
from genesis.utils.misc import tensor_to_array

from .base_sensor import LinkAttachedSensorMixin, RigidSensorArrayMixin, SimpleSensor, SimpleSensorArray

if TYPE_CHECKING:
    from genesis.utils.ring_buffer import TensorRingBuffer
    from genesis.vis.rasterizer_context import RasterizerContext


STEFAN_BOLTZMANN = 5.670374419e-8  # W / (m^2 * K^4)
KELVIN_OFFSET = 273.15
MAX_TEMP = 1000.0  # degC


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
    """Boolean mask of shape (nx, ny, nz) of the boundary voxels, those with at least one face on the grid boundary."""
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
    """
    Diffuse the temperature grid of every sensor by a semi-implicit fast Fourier transform (FFT) step with mirror
    padding, a Neumann boundary condition, and add its heat generation.
    """
    n_batches = output.shape[0]
    start = 0
    # FIXME: one pass per sensor; the array's tables allow one pass over every column.
    for i_s in range(len(cache_sizes)):
        size = cache_sizes[i_s]
        nx, ny, nz = int(grid_size[i_s][0]), int(grid_size[i_s][1]), int(grid_size[i_s][2])
        mat_idx = link_to_material_idx[links_idx[i_s]]
        rcp = link_rho_cp[mat_idx]
        k = link_conductivity[mat_idx]
        alpha = k / rcp
        T = output[:, start : start + size].view(n_batches, nx, ny, nz)
        # Mirror-pad to (2*nx, 2*ny, 2*nz) for zero-flux (Neumann) boundaries; avoids FFT wrap-around.
        T_x = torch.cat([T, torch.flip(T, dims=(1,))], dim=1)
        T_xy = torch.cat([T_x, torch.flip(T_x, dims=(2,))], dim=2)
        T_pad = torch.cat([T_xy, torch.flip(T_xy, dims=(3,))], dim=3)
        T_hat = torch.fft.rfftn(T_pad, dim=(1, 2, 3))
        T_hat = T_hat / (1.0 + dt * alpha * K2_spectral[i_s].unsqueeze(0))
        T_pad = torch.fft.irfftn(T_hat, s=(2 * nx, 2 * ny, 2 * nz), dim=(1, 2, 3))
        T = T_pad[:, :nx, :ny, :nz]
        output[:, start : start + size] = T.reshape(n_batches, -1)

        # Add internal heat generation (W/m^2 -> Q_vol = Q_surface / dz).
        q = heat_generation[i_s]
        if q is not None:
            dz = max(voxel_size[i_s, 2], eps)
            Q_vol = q.reshape(-1) / dz
            delta_T = dt * Q_vol / rcp
            output[:, start : start + size] += delta_T.unsqueeze(0)
        start += size


@qd.func
def _qd_polygon_area_from_points_3d(i_b: int, n: int, scratch: qd.types.ndarray(), eps: float) -> float:
    """Return the area of the polygon whose vertices the scratch buffer holds."""
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
    n_batches = contact_area.shape[0]
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
                contact_area[i_b, contact_idx] = area_per_contact


@qd.func
def _qd_k_eff(k_a: float, k_b: float, eps: float) -> float:
    """Return the effective conductivity of two materials in series, ``2 * k_a * k_b / (k_a + k_b + eps)``."""
    return gs.qd_float(2.0) * k_a * k_b / (k_a + k_b + eps)


@qd.kernel
def _kernel_contact_heat(
    links_idx: qd.types.ndarray(),
    sensor_cache_start: qd.types.ndarray(),
    link_to_material_idx: qd.types.ndarray(),
    dt: float,
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
    eps: float,
):
    n_batches = output.shape[0]
    n_sensors = links_idx.shape[0]
    use_link_temps = link_temps.shape[0] > 0

    # Grid update, for the contacts involving a sensorized link
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
                T_cell = output[i_b, start + cell_idx]
                area_base = contact_area[i_b, i_c] + eps
                area = qd.max(
                    area_base,
                    qd.cast(qd.math.pi, gs.qd_float) * dw * collider_state.contact_data.penetration[i_col, i_b],
                )
                flux = k_eff * (T_other - T_cell) / (vol / area + eps)
                Q_vol = flux * area / vol
                delta_T = dt * Q_vol / rcp
                output[i_b, start + cell_idx] = T_cell + delta_T

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
                area = contact_area[i_b, i_c] + eps
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
    """
    Return the temperature drop from radiation and convection over a step, ``dt * (q_rad + q_conv) / (rho_cp * vol)``,
    which the caller subtracts.
    """
    T_K = T + KELVIN_OFFSET
    T_amb_K = ambient_temp + KELVIN_OFFSET
    q_rad = emissivity * STEFAN_BOLTZMANN * (T_K**4 - T_amb_K**4)
    q_conv = convection_coeff * (T - ambient_temp)
    return dt * (q_rad + q_conv) / (rho_cp_vol + gs.EPS)


# FIXME: one pass per sensor with host reads of its scalars; batch over the columns from the tables.
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
    """Apply radiation and convection to the surface voxels and, when allocated, to the link temperatures.

    A link with ``link_to_material_idx == -1`` takes the emissivity and rho_cp of material index 0 (the default
    properties), and only the links with a valid material are updated.
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
        T_flat = output[:, start : start + size]
        delta = _radiation_convection_delta_T(T_flat, emiss, convection_coeff, ambient_temp, denom, dt)
        output[:, start : start + size] -= delta * mask.unsqueeze(0)
        start += size

    if link_temps.numel() > 0:
        valid = link_to_material_idx >= 0  # (n_links,)
        mat_idx = link_to_material_idx.clamp(min=0)  # -1 -> 0 (default material) for indexing
        rcp_vol = link_rho_cp[mat_idx] * link_volume  # (n_links,)
        delta = _radiation_convection_delta_T(
            link_temps, link_emissivity[mat_idx], convection_coeff, ambient_temp, rcp_vol.unsqueeze(0), dt
        )
        link_temps.sub_(delta * valid.unsqueeze(0).to(gs.tc_float))


# FIXME: one pass per sensor with host reads of its scalars; batch over the columns from the tables.
def _apply_T_measured_filter(
    cache_sizes: list[int],
    sensor_time_const: torch.Tensor,
    dt: float,
    T_prev: torch.Tensor,
    T_out: torch.Tensor,
) -> None:
    """Apply the first-order response of the sensor element over the columns of each sensor, batched over envs.

    ``T_out`` holds the raw temperature on entry and ``T_prev + (dt / tau) * (T_out - T_prev)`` on exit. A sensor with
    ``tau <= 0`` keeps its raw temperature.
    """
    start = 0
    for i_s in range(len(cache_sizes)):
        size = cache_sizes[i_s]
        tau = sensor_time_const[i_s].item()
        if tau > 0:
            alpha = dt / tau
            # T_prev + alpha * (T_raw - T_prev), as one in-place blend of the raw slice
            T_out[:, start : start + size].mul_(alpha).add_(T_prev[:, start : start + size], alpha=1.0 - alpha)
        start += size


class TemperatureGridSensorArray(RigidSensorArrayMixin, SimpleSensorArray[TemperatureGridOptions]):
    """Array of every temperature grid sensor of the scene."""

    def build(self) -> None:
        """Assemble the thermal environment and material tables from every sensor, then stack the grid of each sensor.

        The ambient temperature and convection coefficient are shared by the type: a sensor setting one sets it for
        all, the last one in order winning.
        """
        super().build()

        _B = self._sim._B
        solver = self._sim.rigid_solver
        sensors_options = [sensor.options for sensor in self._sensors]
        self.ambient_temperature = 21.0
        self.convection_coeff = 1.0
        self.properties_dict: dict[int, TemperatureProperties] = {}
        for sensor_options in sensors_options:
            if sensor_options.ambient_temperature is not None:
                self.ambient_temperature = sensor_options.ambient_temperature
            if sensor_options.convection_coefficient is not None:
                self.convection_coeff = sensor_options.convection_coefficient
            self.properties_dict.update(sensor_options.properties_dict)
        for link in self._links:
            if link.idx not in self.properties_dict and -1 not in self.properties_dict:
                gs.raise_exception(
                    f"Temperature properties for the attached link index {link.idx} should be provided in "
                    "properties_dict, or use key -1 for default properties for all links."
                )

        # One material column per entry, sorted by link index so the default properties (key -1) come first. A link
        # without properties maps to the default column when there is one, to -1 (invalid) otherwise
        self.link_material_properties = torch.empty(
            (len(_PropIdx), len(self.properties_dict)), dtype=gs.tc_float, device=gs.device
        )
        self.link_to_material_idx = torch.full(
            (solver.n_links,), 0 if -1 in self.properties_dict else -1, dtype=gs.tc_int, device=gs.device
        )
        for i, (prop_idx, props) in enumerate(sorted(self.properties_dict.items(), key=lambda x: x[0])):
            # order should match _PropIdx
            self.link_material_properties[:, i] = torch.tensor(
                [props.base_temperature, props.conductivity, props.emissivity, props.density * props.specific_heat],
                dtype=gs.tc_float,
                device=gs.device,
            )
            if prop_idx >= 0:
                self.link_to_material_idx[prop_idx] = i

        # The kernels read the link temperatures when the table has rows, so it stays empty when no sensor asks for them
        self.simulate_all_link_temps = any(
            sensor_options.simulate_all_link_temperatures for sensor_options in sensors_options
        )
        # The temperature every link starts from, the ambient one for a link without material
        self.link_base_temps = torch.where(
            self.link_to_material_idx >= 0,
            self.link_material_properties[_PropIdx.BASE_TEMP][self.link_to_material_idx],
            torch.tensor(self.ambient_temperature, dtype=gs.tc_float, device=gs.device),
        )
        self.link_temps = torch.empty((0, 0), dtype=gs.tc_float, device=gs.device)
        self.link_volume = torch.empty((0,), dtype=gs.tc_float, device=gs.device)
        if self.simulate_all_link_temps:
            self.link_volume = torch.empty(solver.n_links, dtype=gs.tc_float, device=gs.device)
            for entity in solver.entities:
                for link in entity.links:
                    if link.n_geoms > 0:
                        aabb = link.get_AABB()
                        if aabb.ndim == 3:
                            aabb = aabb[0]
                        self.link_volume[link.idx] = (aabb[1] - aabb[0]).prod().clamp_min(gs.EPS)
            self.link_temps = self.link_base_temps.expand(_B, -1).clone()

        # The grid of each sensor, spanning the AABB of its link in the link frame
        aabbs_min = []
        aabbs_extent = []
        for link in self._links:
            aabb_world = link.get_AABB()
            if aabb_world.ndim == 2:
                aabb_world = aabb_world.unsqueeze(0)
            link_pos, link_quat = link.get_pos(relative=False), link.get_quat(relative=False)
            if link_pos.ndim == 2:
                link_pos, link_quat = link_pos[0], link_quat[0]
            aabb_min = gu.inv_transform_by_trans_quat(aabb_world[0, 0], link_pos, link_quat)
            aabb_max = gu.inv_transform_by_trans_quat(aabb_world[0, 1], link_pos, link_quat)
            aabbs_min.append(aabb_min)
            aabbs_extent.append((aabb_max - aabb_min).reshape(3))
        self.aabb_min = torch.stack(aabbs_min)
        self.aabb_extent = torch.stack(aabbs_extent)
        self.grid_size = torch.tensor(
            [sensor_options.grid_size for sensor_options in sensors_options], dtype=gs.tc_int, device=gs.device
        )
        self.voxel_size = self.aabb_extent / self.grid_size
        self.voxel_volume = self.voxel_size.prod(dim=1)
        self.sensor_time_const = torch.tensor(
            [sensor_options.sensor_time_constant for sensor_options in sensors_options],
            dtype=gs.tc_float,
            device=gs.device,
        )
        self.contact_depth_weight = torch.tensor(
            [sensor_options.contact_depth_weight for sensor_options in sensors_options],
            dtype=gs.tc_float,
            device=gs.device,
        )
        self.K2_spectral = []
        self.sensor_surface_mask = []
        self.heat_generation: list[torch.Tensor | None] = []
        for sensor_options, (nx, ny, nz), (dx, dy, dz) in zip(
            sensors_options, self.grid_size.tolist(), self.voxel_size.tolist()
        ):
            self.K2_spectral.append(
                _compute_K2_rfft3(nx * 2, ny * 2, nz * 2, dx, dy, dz, gs.device, gs.tc_float, gs.EPS)
            )
            self.sensor_surface_mask.append(_compute_surface_mask(nx, ny, nz, gs.device).to(gs.tc_float))
            heat_generation = None
            if sensor_options.heat_generation is not None:
                heat_generation = torch.tensor(sensor_options.heat_generation, dtype=gs.tc_float, device=gs.device)
                if heat_generation.shape != (nx, ny, nz):
                    gs.raise_exception(
                        f"heat_generation shape {tuple(heat_generation.shape)} does not match grid_size "
                        f"({nx}, {ny}, {nz})"
                    )
            self.heat_generation.append(heat_generation)

        # Contact area buffers, one row per environment
        n_c_max = int(solver.collider.collider_info.max_candidate_contacts[None])
        self.contact_area_buffer = torch.zeros((_B, n_c_max), device=gs.device, dtype=gs.tc_float)
        self.contact_area_scratch = torch.empty((_B, n_c_max, len(_ScratchIdx)), device=gs.device, dtype=gs.tc_float)
        self._debug_objects: list[list] = [[] for _ in sensors_options]

    def _get_return_format(self, options: TemperatureGridOptions) -> tuple[int, ...]:
        return (options.grid_size,)

    def _get_cache_dtype(self) -> torch.dtype:
        return gs.tc_float

    def reset(self, envs_idx: EnvsIdxType | None) -> None:
        super().reset(envs_idx)

        # The temperature field integrates from step to step, so a reset seats every cell of the ground-truth cache at
        # the base temperature of its link's material
        sensors_mat_idx = self.link_to_material_idx[self.links_idx]
        sensors_base_T = self.link_material_properties[_PropIdx.BASE_TEMP][sensors_mat_idx]
        for i_s in range(len(self._sensors)):
            self._ground_truth_cache[(*envs_idx, self._cache_slice(i_s))] = sensors_base_T[i_s]
        if self.link_temps.numel() > 0:
            self.link_temps[envs_idx] = self.link_base_temps

    def _update_raw_data(self, raw_data: torch.Tensor) -> None:
        solver = self.solver
        dt = self._sim.dt
        props = self.link_material_properties
        link_conductivity = props[_PropIdx.CONDUCTIVITY]
        link_base_temperature = props[_PropIdx.BASE_TEMP]
        link_emissivity = props[_PropIdx.EMISSIVITY]
        link_rho_cp = props[_PropIdx.RHO_CP]
        # 1) Batched FFT semi-implicit diffusion + 2) Heat generation
        _apply_diffusion_and_heat_generation(
            self.cache_sizes,
            self.grid_size,
            self.heat_generation,
            self.voxel_size,
            self.links_idx,
            self.link_to_material_idx,
            link_rho_cp,
            link_conductivity,
            self.K2_spectral,
            dt,
            gs.EPS,
            raw_data,
        )
        # 3) Contact heat transfer
        collider_state = solver.collider.collider_state
        self.contact_area_buffer.zero_()
        _kernel_compute_contact_areas(
            self.contact_area_buffer,
            self.contact_area_scratch,
            solver.dyn_state,
            collider_state,
            gs.EPS,
        )
        _kernel_contact_heat(
            self.links_idx,
            self.sensors_cache_start,
            self.link_to_material_idx,
            dt,
            self.aabb_min,
            self.grid_size,
            self.voxel_size,
            self.voxel_volume,
            self.contact_depth_weight,
            self.link_temps,
            self.link_volume,
            link_base_temperature,
            link_conductivity,
            link_rho_cp,
            self.contact_area_buffer,
            raw_data,
            solver.dyn_state,
            collider_state,
            gs.EPS,
        )
        raw_data.clamp_(-MAX_TEMP, MAX_TEMP)
        # 4) Radiation and convection
        _apply_radiation_convection(
            self.cache_sizes,
            self.sensor_surface_mask,
            self.voxel_volume,
            self.links_idx,
            self.link_temps,
            self.link_volume,
            self.link_to_material_idx,
            link_emissivity,
            link_rho_cp,
            self.ambient_temperature,
            self.convection_coeff,
            dt,
            raw_data,
        )

    def _apply_transform(self, data: torch.Tensor, timeline: "TensorRingBuffer", *, is_measured: bool) -> None:
        # First-order resistor-capacitor (RC) filter modelling the thermal response time of the sensor element. The
        # thermal mass is a property of the sensor element only, so the filter is measured-only and ground truth exposes
        # the raw simulated temperature. `data` is the measured ring slot 0, holding the current raw temperature written
        # by `_update_current_timestep_data`; the previous filtered value lives in `timeline.at(1)`.
        if not is_measured:
            return
        _apply_T_measured_filter(self.cache_sizes, self.sensor_time_const, self._sim.dt, timeline.at(1), data)

    def _draw_debug(self, i_s: int, context: "RasterizerContext") -> None:
        """Draw a single flat mesh colored by the temperature of sensor ``i_s`` (cool=blue, hot=red), in the first
        rendered environment."""
        options = self._sensors[i_s].options
        env_idx = context.rendered_envs_idx[0] if self._sim.n_envs > 0 else None
        debug_objects = self._debug_objects[i_s]
        for obj in debug_objects:
            context.clear_debug_object(obj)
        debug_objects.clear()

        link = self._links[i_s]
        link_pos = tensor_to_array(link.get_pos(env_idx, relative=False)).reshape(3)
        link_quat = tensor_to_array(link.get_quat(env_idx, relative=False)).reshape(4)
        link_T = gu.trans_quat_to_T(link_pos, link_quat)
        voxel_size = tensor_to_array(self.voxel_size[i_s]).reshape(3)
        # Per-cell color from temperature (blue=cool, red=hot)
        temps = tensor_to_array(self.read(i_s, env_idx, is_ground_truth=True)).reshape(-1)
        t_min = options.debug_temperature_range[0]
        t_range = options.debug_temperature_range[1] - t_min
        if t_range <= 0:
            t_range = 1.0
        norm = np.clip((temps - t_min) / t_range, 0.0, 1.0)
        colors_rgba = np.column_stack((norm, np.zeros_like(norm), 1.0 - norm, np.full_like(norm, 0.5)))
        # Build a single mesh: one quad (2 triangles) per cell on the top face, at the cell centers of this grid
        nx, ny, nz = options.grid_size
        cells = np.stack(np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij"), axis=-1).reshape(
            -1, 3
        )
        aabb_min = tensor_to_array(self.aabb_min[i_s])
        cell_positions = aabb_min + (cells + 0.5) * voxel_size
        n_cells = len(cell_positions)
        hx, hy, hz = voxel_size[0] / 2, voxel_size[1] / 2, voxel_size[2] / 2
        quad_offsets = np.array([[-hx, -hy, hz], [hx, -hy, hz], [hx, hy, hz], [-hx, hy, hz]])
        vertices = (cell_positions[:, np.newaxis, :] + quad_offsets[np.newaxis, :, :]).reshape(-1, 3)
        idx = np.arange(n_cells, dtype=gs.np_int) * 4
        faces = np.empty((n_cells * 2, 3), dtype=gs.np_int)
        faces[0::2] = np.column_stack([idx, idx + 1, idx + 2])
        faces[1::2] = np.column_stack([idx, idx + 2, idx + 3])
        face_colors_u8 = np.empty((n_cells * 2, 4), dtype=np.uint8)
        face_colors_u8[0::2] = (colors_rgba * 255).astype(np.uint8)
        face_colors_u8[1::2] = face_colors_u8[0::2]
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, face_colors=face_colors_u8)
        debug_objects.append(context.draw_debug_mesh(mesh, T=link_T))


class TemperatureGridSensor(LinkAttachedSensorMixin, SimpleSensor[TemperatureGridOptions, TemperatureGridSensorArray]):
    """Temperature grid sensor: a voxel grid of temperatures over the AABB of its link, heated by the contacts and
    exchanging with the ambient by radiation and convection."""

    @property
    def link_temperatures(self) -> torch.Tensor:
        return self._array.link_temps

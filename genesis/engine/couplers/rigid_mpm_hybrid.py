"""The coupling of the rigid and soft parts of the hybrid entities, run after the coupling of the solvers."""

import quadrants as qd

import genesis.utils.geom as gu
from genesis.engine.solvers.mpm_solver import MPMInfo, MPMState
from genesis.engine.solvers.rigid.data import GeomsInfo, LinksState
from genesis.utils.data import eoa


@qd.kernel
def kernel_update_hybrid_soft_part(
    f: int,
    particle_start: int,
    substep_dt: float,
    dt: float,
    rigid_dt: float,
    damping: float,
    soft_dv_coef: float,
    part_soft_info: qd.template(),
    part_soft_init_positions: qd.Tensor,
    mpm_state: MPMState,
    links_state: LinksState,
    mpm_info: MPMInfo,
    geoms_info: GeomsInfo,
):
    """Move the soft particles of a hybrid entity with its rigid links and couple their velocities both ways.

    The particles of the soft part start at `particle_start` among those of the solver, each following the link of its
    muscle group (see 'part_soft_info' of HybridEntity). They are read at frame f + 1, past the grid-to-particle
    transfer of frame f.
    """
    for i_p_, i_b in qd.ndrange(
        part_soft_init_positions.shape[0],
        mpm_state.particles.pos.shape[2],
        axes=qd.static(eoa(mpm_state.particles.pos)),
    ):
        i_p = i_p_ + particle_start
        if mpm_state.particles.is_active[f, i_p, i_b]:
            group_idx = mpm_info.particles.muscle_group[i_p]
            link_idx = part_soft_info.link_idx[group_idx]
            geom_idx = part_soft_info.geom_idx[group_idx]
            trans_local_to_global = part_soft_info.trans_local_to_global[group_idx]
            quat_local_to_global = part_soft_info.quat_local_to_global[group_idx]

            g_pos_0 = geoms_info.pos[geom_idx]
            g_quat_0 = geoms_info.quat[geom_idx]

            # The target position follows the link in minimal coordinates
            x_init_local = gu.qd_inv_transform_by_trans_quat(
                part_soft_init_positions[i_p_], trans_local_to_global, quat_local_to_global
            )
            scaled_pos = gu.qd_transform_by_quat(gu.qd_inv_transform_by_quat(g_pos_0, g_quat_0), g_quat_0)
            tx_pos, tx_quat = gu.qd_transform_pos_quat_by_trans_quat(
                scaled_pos, g_quat_0, links_state.pos[link_idx, i_b], links_state.quat[link_idx, i_b]
            )
            new_x_pos = gu.qd_transform_by_trans_quat(x_init_local, trans=tx_pos, quat=tx_quat)

            # The velocity closing the gap moves the soft part incrementally over the substeps of the soft solver,
            # scaled linearly between the time steps of the soft and rigid solvers
            x_pos = mpm_state.particles.pos[f + 1, i_p, i_b]
            xd_vel = (new_x_pos - x_pos) / substep_dt * (substep_dt / rigid_dt)
            vel_d = (xd_vel - mpm_state.particles.vel[f + 1, i_p, i_b]) * qd.exp(-dt * damping)

            # Soft-to-rigid coupling, the rigid time step integrating the force within the substep of the soft solver
            mass_real = mpm_info.particles.mass[i_p] / mpm_info.particle_volume_scale[None]
            frc_vel = mass_real * (vel_d / rigid_dt)
            frc_ang = (x_pos - links_state.root_COM[link_idx, i_b]).cross(frc_vel)
            qd.atomic_add(links_state.cfrc_coupling_vel[link_idx, i_b], frc_vel)
            qd.atomic_add(links_state.cfrc_coupling_ang[link_idx, i_b], frc_ang)

            # Rigid-to-soft coupling, which may feed an unstable loop back
            mpm_state.particles.vel[f + 1, i_p, i_b] = mpm_state.particles.vel[f + 1, i_p, i_b] + vel_d * soft_dv_coef

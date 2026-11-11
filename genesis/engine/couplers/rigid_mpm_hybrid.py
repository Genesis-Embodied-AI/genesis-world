"""The coupling of the rigid and soft parts of the hybrid entities, run after the coupling of the solvers."""

import numpy as np
import torch

import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu
from genesis.engine.entities import HybridEntity
from genesis.engine.solvers.mpm_solver import MPMInfo, MPMSolver, MPMState, func_add_particle_vel
from genesis.engine.solvers.rigid.abd.misc import func_add_external_wrench
from genesis.engine.solvers.rigid.data import GeomsInfo, LinksState
from genesis.engine.solvers.rigid.rigid_solver import RigidSolver
from genesis.grad.bridge import reversed_with
from genesis.utils.data import Data, DataKind, Size, eoa, field
from genesis.utils.geom import PARTICLES
from genesis.utils.misc import qd_from_torch

# The muscle groups of the soft parts of every hybrid entity
GROUPS = Size("groups")


class RigidMPMHybridInfo(Data, kind=DataKind.CONSTANT):
    """The soft parts of the hybrid entities, whose particles follow the rigid links of their muscle groups."""

    # The rigid link each muscle group follows, the geom it is attached to, and the frame its particles are expressed in
    groups_link_idx: qd.Tensor = field(int, shape=(GROUPS,))
    groups_geom_idx: qd.Tensor = field(int, shape=(GROUPS,))
    groups_trans_local_to_global: qd.Tensor = field((float, 3), shape=(GROUPS,))
    groups_quat_local_to_global: qd.Tensor = field((float, 4), shape=(GROUPS,))
    # The damping of the velocity closing the gap to the target and the share of it the particles take, per muscle group
    groups_damping: qd.Tensor = field(float, shape=(GROUPS,))
    groups_soft_dv_coef: qd.Tensor = field(float, shape=(GROUPS,))
    # The index of each particle among those of the MPM solver, and its position when its hybrid entity was created
    particles_idx: qd.Tensor = field(int, shape=(PARTICLES,))
    particles_init_pos: qd.Tensor = field((float, 3), shape=(PARTICLES,))
    # The first muscle group of the hybrid entity of each particle, which its muscle group counts from
    particles_group_start: qd.Tensor = field(int, shape=(PARTICLES,))


@reversed_with(is_backward=True)
@qd.kernel
def kernel_update_hybrid_soft_part(
    substep_dt: float,
    dt: float,
    rigid_dt: float,
    mpm_state: MPMState,
    links_state: LinksState,
    mpm_info: MPMInfo,
    geoms_info: GeomsInfo,
    rigid_mpm_hybrid_info: RigidMPMHybridInfo,
    mpm_config: qd.template(),
    is_backward: qd.template(),
):
    """
    Move the soft particles of the hybrid entities with their rigid links and couple their velocities both ways.

    Each particle follows the link of its muscle group (see RigidMPMHybridInfo). The particles are read once the substep
    integrated them, past the grid-to-particle transfer. Their forces add exactly to the coupling forces of the links
    with `is_deterministic` (see func_add_external_wrench), except in the variant autodiff reverses (`is_backward`).
    """
    for i_hp, i_b in qd.ndrange(
        rigid_mpm_hybrid_info.particles_idx.shape[0],
        mpm_state.particles.pos.shape[1],
        axes=qd.static(eoa(mpm_state.particles.pos)),
    ):
        i_p = rigid_mpm_hybrid_info.particles_idx[i_hp]
        if mpm_state.particles.is_active[i_p, i_b]:
            i_group_start = rigid_mpm_hybrid_info.particles_group_start[i_hp]
            i_group = i_group_start + mpm_info.particles.muscle_group[i_p]
            i_l = rigid_mpm_hybrid_info.groups_link_idx[i_group]
            i_g = rigid_mpm_hybrid_info.groups_geom_idx[i_group]
            trans_local_to_global = rigid_mpm_hybrid_info.groups_trans_local_to_global[i_group]
            quat_local_to_global = rigid_mpm_hybrid_info.groups_quat_local_to_global[i_group]

            g_pos_0 = geoms_info.pos[i_g]
            g_quat_0 = geoms_info.quat[i_g]

            # The target position follows the link in minimal coordinates
            x_init_local = gu.qd_inv_transform_by_trans_quat(
                rigid_mpm_hybrid_info.particles_init_pos[i_hp], trans_local_to_global, quat_local_to_global
            )
            scaled_pos = gu.qd_transform_by_quat(gu.qd_inv_transform_by_quat(g_pos_0, g_quat_0), g_quat_0)
            tx_pos, tx_quat = gu.qd_transform_pos_quat_by_trans_quat(
                scaled_pos, g_quat_0, links_state.pos[i_l, i_b], links_state.quat[i_l, i_b]
            )
            new_x_pos = gu.qd_transform_by_trans_quat(x_init_local, trans=tx_pos, quat=tx_quat)

            # The velocity closing the gap moves the soft part incrementally over the substeps of the soft solver,
            # scaled linearly between the time steps of the soft and rigid solvers
            x_pos = mpm_state.particles.pos[i_p, i_b]
            xd_vel = (new_x_pos - x_pos) / substep_dt * (substep_dt / rigid_dt)
            vel_d = (xd_vel - mpm_state.particles.vel[i_p, i_b]) * qd.exp(
                -dt * rigid_mpm_hybrid_info.groups_damping[i_group]
            )

            # Soft-to-rigid coupling, the rigid time step integrating the force within the substep of the soft solver
            mass_real = mpm_info.particles.mass[i_p] / mpm_info.particle_volume_scale[None]
            # The force on the link is the reaction of the force the link applies the particle
            func_add_external_wrench(
                i_l,
                i_b,
                x_pos,
                force=-mass_real * (vel_d / rigid_dt),
                torque=qd.Vector.zero(gs.qd_float, 3),
                links_state=links_state,
                is_exact=mpm_config.is_deterministic and not is_backward,
            )

            # Rigid-to-soft coupling, which may feed an unstable loop back
            func_add_particle_vel(
                i_p, i_b, vel=vel_d * rigid_mpm_hybrid_info.groups_soft_dv_coef[i_group], mpm_state=mpm_state
            )


class RigidMPMHybrid:
    """
    Drive the particles of the soft part of the hybrid entities toward the links of their rigid part (see
    kernel_update_hybrid_soft_part).

    The simulator hands it the hybrid entities without the default coupling, and runs the update once the particles
    left the grid.
    """

    def __init__(self, entities: list[HybridEntity], mpm_solver: MPMSolver, rigid_solver: RigidSolver) -> None:
        self._entities = entities
        self.mpm_solver = mpm_solver
        self.rigid_solver = rigid_solver
        self.rigid_mpm_hybrid_info: RigidMPMHybridInfo | None = None

    def build(self) -> None:
        for entity in self._entities:
            entity.build()

        soft_parts = [entity.mpm_hybrid_part for entity in self._entities]
        groups_count = [len(part.groups_link_idx) for part in soft_parts]
        particles_count = [len(part.particles_idx) for part in soft_parts]
        hybrid_info = RigidMPMHybridInfo(sizes=dict(groups=sum(groups_count), particles=sum(particles_count)))
        for array, values in (
            (hybrid_info.groups_link_idx, [part.groups_link_idx for part in soft_parts]),
            (hybrid_info.groups_geom_idx, [part.groups_geom_idx for part in soft_parts]),
            (hybrid_info.groups_trans_local_to_global, [part.groups_trans_local_to_global for part in soft_parts]),
            (hybrid_info.groups_quat_local_to_global, [part.groups_quat_local_to_global for part in soft_parts]),
            (hybrid_info.groups_damping, [part.groups_damping for part in soft_parts]),
            (hybrid_info.groups_soft_dv_coef, [part.groups_soft_dv_coef for part in soft_parts]),
            (hybrid_info.particles_idx, [part.particles_idx for part in soft_parts]),
            (hybrid_info.particles_init_pos, [part.particles_init_pos for part in soft_parts]),
            (hybrid_info.particles_group_start, [np.repeat(np.cumsum([0, *groups_count[:-1]]), particles_count)]),
        ):
            qd_from_torch(array, torch.from_numpy(np.concatenate(values)))
        if gs.use_zerocopy.is_supported and gs.backend == gs.metal:
            torch.mps.synchronize()
        self.rigid_mpm_hybrid_info = hybrid_info

    def update_soft_parts(self) -> None:
        kernel_update_hybrid_soft_part(
            self.mpm_solver.substep_dt,
            self.mpm_solver.dt,
            self.rigid_solver.dt,
            self.mpm_solver.mpm_state,
            self.rigid_solver.dyn_state.links,
            self.mpm_solver.mpm_info,
            self.rigid_solver.dyn_info.geoms,
            self.rigid_mpm_hybrid_info,
            self.mpm_solver.mpm_config,
            is_backward=False,
        )

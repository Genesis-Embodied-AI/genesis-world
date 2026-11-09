from typing import TYPE_CHECKING

import numpy as np
import quadrants as qd

import genesis as gs
import genesis.utils.sdf as sdf

from genesis.options.solvers import LegacyCouplerOptions
from genesis.repr_base import RBC
from genesis.utils.array_class import GeomsInfo, GeomsState, LinksInfo, LinksState, RigidInfo, SDFInfo
from genesis.utils.data import opt_in
from genesis.utils.geom import qd_inv_transform_by_trans_quat, qd_transform_by_trans_quat

if TYPE_CHECKING:
    from genesis.engine.simulator import Simulator

CLAMPED_INV_DT = 50.0


@qd.data_oriented
class LegacyCoupler(RBC):
    """
    This class handles all the coupling between different solvers. LegacyCoupler will be deprecated in the future.
    """

    # ------------------------------------------------------------------------------------
    # --------------------------------- Initialization -----------------------------------
    # ------------------------------------------------------------------------------------

    def __init__(self, simulator: "Simulator", options: "LegacyCouplerOptions") -> None:
        self.sim = simulator
        self.options = options

        self.rigid_solver = self.sim.rigid_solver
        self.mpm_solver = self.sim.mpm_solver
        self.sph_solver = self.sim.sph_solver
        self.pbd_solver = self.sim.pbd_solver
        self.fem_solver = self.sim.fem_solver

        # The particles and the FEM vertices collide with the rigid geoms through their SDF
        opt_in(SDFInfo, cond=lambda: any(self.rigid_couplings))

    @property
    def rigid_couplings(self) -> tuple[bool, bool, bool, bool]:
        """Whether the rigid geoms couple with the MPM, SPH, PBD and FEM solvers, in this order."""
        return tuple(
            [
                self.rigid_solver.is_active and solver.is_active and is_enabled
                for solver, is_enabled in (
                    (self.mpm_solver, self.options.rigid_mpm),
                    (self.sph_solver, self.options.rigid_sph),
                    (self.pbd_solver, self.options.rigid_pbd),
                    (self.fem_solver, self.options.rigid_fem),
                )
            ]
        )

    def build(self) -> None:
        # The couplings with MPM are decided in _resolve_mpm_couplings
        _, self._rigid_sph, self._rigid_pbd, self._rigid_fem = self.rigid_couplings
        self._fem_sph = self.fem_solver.is_active and self.sph_solver.is_active and self.options.fem_sph

        if self._rigid_sph:
            self.sph_rigid_normal = qd.Vector.field(
                3,
                dtype=gs.qd_float,
                shape=(self.sph_solver.n_particles, self.rigid_solver.n_geoms_, self.sph_solver._B),
            )
            self.sph_rigid_normal_reordered = qd.Vector.field(
                3,
                dtype=gs.qd_float,
                shape=(self.sph_solver.n_particles, self.rigid_solver.n_geoms_, self.sph_solver._B),
            )

        if self._rigid_pbd:
            self.pbd_rigid_normal_reordered = qd.Vector.field(
                3, dtype=gs.qd_float, shape=(self.pbd_solver.n_particles, self.pbd_solver._B, self.rigid_solver.n_geoms)
            )

            struct_particle_attach_info = qd.types.struct(link_idx=gs.qd_int, local_pos=gs.qd_vec3)

            self.particle_attach_info = struct_particle_attach_info.field(
                shape=(self.pbd_solver._n_particles, self.pbd_solver._B), layout=qd.Layout.SOA
            )
            self.particle_attach_info.link_idx.fill(-1)
            self.particle_attach_info.local_pos.fill(0.0)

        ## DEBUG
        self._dx = 1 / 1024
        self._stencil_size = int(np.floor(self._dx / self.sph_solver.hash_grid_cell_size) + 2)

        self._resolve_mpm_couplings()
        self._build_mpm_couplings()
        self.reset(envs_idx=self.sim.scene._envs_idx)

    def _resolve_mpm_couplings(self) -> None:
        if self.mpm_solver.is_active and self.rigid_solver.is_active and self.options.rigid_mpm:
            if self.rigid_solver._options.friction_cone == gs.friction_cone.elliptic:
                gs.raise_exception(
                    "Coupling the implicit MPM solver with an elliptic Rigid friction cone is not supported."
                )
            # The joint solve uses Rigid's Newton Hessian factor as its preconditioner and integrates every Rigid dof
            if self.rigid_solver._options.constraint_solver != gs.constraint_solver.Newton:
                gs.raise_exception("Coupling the implicit MPM solver with Rigid requires Rigid's Newton solver.")
            if self.rigid_solver._use_hibernation or self.rigid_solver._options.noslip_iterations > 0:
                gs.raise_exception(
                    "Coupling the implicit MPM solver with Rigid does not support hibernation or noslip iterations."
                )
        # The particles get contact pairs on the coupled Rigid geoms, and Rigid's dofs join the Newton solve with them
        # if it has any: pairs on static geoms (a plane, a terrain) act on the particle displacement alone, as wall
        # pairs do
        self._is_mpm_rigid_coupled = (
            self.mpm_solver.is_active
            and self.rigid_solver.is_active
            and self.options.rigid_mpm
            and any(geom.needs_coup for geom in self.rigid_solver.geoms)
        )
        self._has_mpm_rigid_dofs = self._is_mpm_rigid_coupled and self.rigid_solver.n_dofs > 0
        if self._is_mpm_rigid_coupled and self.sim.options.requires_grad and self.rigid_solver.has_coupled_terrain:
            gs.raise_exception(
                "Differentiable simulation with the implicit MPM solver does not support the contact with a Rigid "
                "terrain yet."
            )
        self._is_mpm_fem_coupled = self.mpm_solver.is_active and self.fem_solver.is_active and self.options.fem_mpm
        if self._is_mpm_fem_coupled and not self.fem_solver._use_implicit_solver:
            gs.raise_exception(
                "Coupling the implicit MPM solver with FEM requires FEM's implicit solver "
                "(FEMOptions(use_implicit_solver=True)), or LegacyCouplerOptions(fem_mpm=False)."
            )
        self._is_mpm_pbd_coupled = self.mpm_solver.is_active and self.pbd_solver.is_active and self.options.mpm_pbd
        self._is_mpm_sph_coupled = self.mpm_solver.is_active and self.sph_solver.is_active and self.options.mpm_sph

    def _build_mpm_couplings(self) -> None:
        """Build the couplings of the implicit MPM solver and register them with their participants in Newton.

        Registering activates the implicit solve of every substep whenever MPM is in the scene. MPM joins first, then
        the solvers it couples with implicitly, each owning the data of its own unknowns. The substep of such a solver
        stops before its own solve, which the joint solve takes over. The MPM particles are also boundary particles of
        the SPH fluid (see mpm_sph_func) and obstacles of the PBD particles, both staggered with the solve."""
        # Imported here to avoid a circular dependency: the couplings with MPM import the solvers, whose entities import
        # the couplers
        from genesis.engine.solvers.newton_contact import NewtonContact

        from .contact_candidates import CouplingPairsHost
        from .mpm_fem_func import FemContactParams, FemTriangleQuery, MpmFemContact
        from .mpm_pbd_func import MpmParticleObstacles, MpmPbdContact, PbdContactParams, PbdParticleQuery
        from .mpm_rigid_func import RigidGeomQuery
        from .mpm_rigid_newton import MpmRigidContact, MpmRigidContactHost
        from .mpm_sph_func import MpmSphBoundary

        if not self.mpm_solver.is_active:
            return
        mpm_host, mpm_participant = self.mpm_solver.build_newton_participant()
        # The participants owning rows of the linear system of the solve, whose vertices are contact vertices in the
        # same order. The contact points of the Rigid links come last, since their count grows with the pairs (see
        # NewtonContact.lay_out_vertices).
        owners = [(mpm_host, mpm_participant)]
        rigid_member = None
        couplings = []
        host_only_devices = []
        pair_kinds = [mpm_participant]
        rigid_contact = None
        if self._is_mpm_rigid_coupled:
            # Without Rigid dofs, the participant holds the Rigid data the pairs read, and the solve leaves Rigid's own
            rigid_host, rigid_participant = self.rigid_solver.build_newton_participant(
                is_member=self._has_mpm_rigid_dofs
            )
            if self._has_mpm_rigid_dofs:
                rigid_member = (rigid_host, rigid_participant)
            else:
                host_only_devices.append(rigid_participant)
            query = RigidGeomQuery(rigid_participant, self.sim._B)
            rigid_contact = MpmRigidContact(mpm_participant, rigid_participant, query, self._has_mpm_rigid_dofs)
            couplings.append((MpmRigidContactHost(rigid_contact), rigid_contact))
            pair_kinds.append(rigid_contact)
        if self._is_mpm_fem_coupled:
            fem_host, fem_participant = self.fem_solver.build_newton_participant()
            owners.append((fem_host, fem_participant))
            fem_contact = FemContactParams(values={"thickness": 0.5 * self.mpm_solver.dx, "stiffness": 10.0})
            query = FemTriangleQuery(fem_participant, mpm_participant.mpm_info, fem_contact, self.sim._B)
            contact = MpmFemContact(mpm_participant, fem_participant, query)
            couplings.append((CouplingPairsHost(contact), contact))
            pair_kinds.append(contact)
        if self._is_mpm_pbd_coupled:
            pbd_contact = PbdContactParams(
                values={"thickness": 0.5 * self.mpm_solver.dx, "stiffness": 10.0, "friction": 0.15}
            )
            pbd_host, pbd_participant = self.pbd_solver.build_newton_participant()
            owners.append((pbd_host, pbd_participant))
            query = PbdParticleQuery(pbd_participant, mpm_participant.mpm_info, pbd_contact, self.sim._B)
            contact = MpmPbdContact(mpm_participant, pbd_participant, query)
            couplings.append((CouplingPairsHost(contact), contact))
            pair_kinds.append(contact)
            self.pbd_solver.set_particle_obstacles(
                MpmParticleObstacles(self.mpm_solver, self.pbd_solver, pbd_contact, self.sim._B)
            )
        if self._is_mpm_sph_coupled:
            self._sph_boundary = MpmSphBoundary(self.mpm_solver, self.sph_solver, self.mpm_solver.mpm_info, self.sim._B)
            self.sph_solver.set_boundary(self._sph_boundary)
        rigid_map = rigid_contact if self._has_mpm_rigid_dofs else None
        contact_system = NewtonContact([device for _, device in owners], rigid_map, pair_kinds)
        self.sim.newton_solver.register(owners, rigid_member, couplings, contact_system, host_only_devices)

    def reset(self, envs_idx=None) -> None:
        if self._rigid_sph:
            if envs_idx is None:
                self.sph_rigid_normal.fill(0)
            else:
                self._kernel_reset_sph(envs_idx)

    @qd.kernel
    def _kernel_reset_sph(self, envs_idx: qd.types.ndarray()):
        for i_p, i_g, i_b_ in qd.ndrange(self.sph_solver.n_particles, self.rigid_solver.n_geoms, envs_idx.shape[0]):
            self.sph_rigid_normal[i_p, i_g, envs_idx[i_b_]] = 0.0

    @qd.func
    def _func_collide_with_rigid(
        self,
        f,
        pos_world,
        vel,
        mass,
        i_b,
        geoms_state: GeomsState,
        geoms_info: GeomsInfo,
        links_state: LinksState,
        rigid_info: RigidInfo,
        sdf_info: SDFInfo,
        collider_config: qd.template(),
    ):
        for i_g in range(self.rigid_solver.n_geoms):
            if geoms_info.needs_coup[i_g]:
                vel = self._func_collide_with_rigid_geom(
                    pos_world,
                    vel,
                    mass,
                    i_g,
                    i_b,
                    geoms_state=geoms_state,
                    geoms_info=geoms_info,
                    links_state=links_state,
                    rigid_info=rigid_info,
                    sdf_info=sdf_info,
                    collider_config=collider_config,
                )
        return vel

    @qd.func
    def _func_collide_with_rigid_geom(
        self,
        pos_world,
        vel,
        mass,
        geom_idx,
        batch_idx,
        geoms_state: GeomsState,
        geoms_info: GeomsInfo,
        links_state: LinksState,
        rigid_info: RigidInfo,
        sdf_info: SDFInfo,
        collider_config: qd.template(),
    ):
        signed_dist = sdf.sdf_func_world(geom_idx, batch_idx, pos_world, geoms_state, geoms_info, sdf_info)

        # bigger coup_softness implies that the coupling influence extends further away from the object.
        influence = qd.min(qd.exp(-signed_dist / max(1e-10, geoms_info.coup_softness[geom_idx])), 1)

        if influence > 0.1:
            normal_rigid = sdf.sdf_func_normal_world(
                geom_idx, batch_idx, pos_world, geoms_state, geoms_info, rigid_info, sdf_info, collider_config
            )
            vel = self._func_collide_in_rigid_geom(
                pos_world, vel, mass, normal_rigid, influence, geom_idx, batch_idx, geoms_info, links_state, rigid_info
            )

        return vel

    @qd.func
    def _func_collide_with_rigid_geom_robust(
        self,
        pos_world,
        vel,
        mass,
        pressure,
        normal_prev,
        geom_idx,
        batch_idx,
        geoms_state: GeomsState,
        geoms_info: GeomsInfo,
        links_state: LinksState,
        links_info: LinksInfo,
        rigid_info: RigidInfo,
        sdf_info: SDFInfo,
        collider_config: qd.template(),
    ):
        """
        Similar to _func_collide_with_rigid_geom, but additionally handles potential side flip due to penetration.
        """
        signed_dist = sdf.sdf_func_world(geom_idx, batch_idx, pos_world, geoms_state, geoms_info, sdf_info)
        normal_rigid = sdf.sdf_func_normal_world(
            geom_idx, batch_idx, pos_world, geoms_state, geoms_info, rigid_info, sdf_info, collider_config
        )

        # bigger coup_softness implies that the coupling influence extends further away from the object.
        influence = qd.min(qd.exp(-signed_dist / max(1e-10, geoms_info.coup_softness[geom_idx])), 1)

        # if normal_rigid.dot(normal_prev) < 0: # side flip due to penetration
        #     influence = 1.0
        #     normal_rigid = normal_prev
        if influence > 0.1:
            vel = self._func_collide_in_rigid_geom(
                pos_world, vel, mass, normal_rigid, influence, geom_idx, batch_idx, geoms_info, links_state, rigid_info
            )

        # Static fluid pressure pushes on the geom even at rest, where the velocity-gated collision response above
        # transfers nothing; this is what makes submerged geoms buoyant. Mirroring the particle pressure across the
        # surface and integrating the symmetric pressure force over the truncated kernel support yields the factor
        # 2 * sigma(signed_dist), with sigma the kernel plane integral (see cubic_kernel_plane_integral in
        # sph_solver.py). Sigma integrates to 1/2 across the support band, so a covering particle layer transmits
        # exactly p per unit area: Archimedes buoyancy with no tuning constant. Fixed links are exempt: they cannot
        # respond to the force, and the reaction is a conservative stiff kick that keeps fluid resting on them
        # ringing forever, pumped by the acoustic pressure fluctuations of the fluid.
        link_idx = geoms_info.link_idx[geom_idx]
        I_l = [link_idx, batch_idx] if qd.static(self.rigid_solver._options.batch_links_info) else link_idx
        if signed_dist < self.sph_solver._support_radius and pressure > 0 and not links_info.is_fixed[I_l]:
            pressure_force = (
                -2.0
                * pressure
                * self.sph_solver._particle_volume
                * self.sph_solver.cubic_kernel_plane_integral(signed_dist)
                * normal_rigid
            )
            self.rigid_solver._func_apply_coupling_force(link_idx, batch_idx, pos_world, pressure_force, links_state)
            vel = vel - pressure_force * (rigid_info.substep_dt[None] / mass)

        # attraction force
        # if 0.001 < signed_dist < 0.01:
        #     vel = vel - normal_rigid * 0.1 * signed_dist

        return vel, normal_rigid

    @qd.func
    def _func_collide_in_rigid_geom(
        self,
        pos_world,
        vel,
        mass,
        normal_rigid,
        influence,
        geom_idx,
        i_b,
        geoms_info: GeomsInfo,
        links_state: LinksState,
        rigid_info: RigidInfo,
    ):
        """
        Resolves collision when a particle is already in collision with a rigid object.
        This function assumes known normal_rigid and influence.
        """
        vel_rigid = self.rigid_solver._func_vel_at_point(
            pos_world=pos_world, link_idx=geoms_info.link_idx[geom_idx], i_b=i_b, links_state=links_state
        )

        # v w.r.t rigid
        rvel = vel - vel_rigid
        rvel_normal_magnitude = rvel.dot(normal_rigid)  # negative if inward

        if rvel_normal_magnitude < 0:  # colliding
            #################### rigid -> particle ####################
            # tangential component
            rvel_tan = rvel - rvel_normal_magnitude * normal_rigid
            rvel_tan_norm = rvel_tan.norm(gs.EPS)

            # tangential component after friction
            rvel_tan = (
                rvel_tan
                / rvel_tan_norm
                * qd.max(0, rvel_tan_norm + rvel_normal_magnitude * geoms_info.coup_friction[geom_idx])
            )

            # normal component after collision
            rvel_normal = -normal_rigid * rvel_normal_magnitude * geoms_info.coup_restitution[geom_idx]

            # normal + tangential component
            rvel_new = rvel_tan + rvel_normal

            # apply influence
            vel_old = vel
            vel = vel_rigid + rvel_new * influence + rvel * (1 - influence)

            #################### particle -> rigid ####################
            # Compute delta momentum and apply to rigid body.
            delta_mv = mass * (vel - vel_old)
            force = -delta_mv / rigid_info.substep_dt[None]
            self.rigid_solver._func_apply_coupling_force(
                geoms_info.link_idx[geom_idx], i_b, pos_world, force, links_state
            )

        return vel

    @qd.kernel
    def fem_surface_force(
        self,
        f: qd.i32,
        geoms_state: GeomsState,
        geoms_info: GeomsInfo,
        links_state: LinksState,
        rigid_info: RigidInfo,
        sdf_info: SDFInfo,
        collider_config: qd.template(),
    ):
        # TODO: all collisions are on vertices instead of surface and edge
        for i_s, i_b in qd.ndrange(self.fem_solver.n_surfaces, self.fem_solver._B):
            dt = self.fem_solver.substep_dt
            iel = self.fem_solver.surface[i_s].tri2el
            mass = self.fem_solver.elements_i[iel].mass_scaled / self.fem_solver.vol_scale

            p1 = self.fem_solver.elements_v[f, self.fem_solver.surface[i_s].tri2v[0], i_b].pos
            p2 = self.fem_solver.elements_v[f, self.fem_solver.surface[i_s].tri2v[1], i_b].pos
            p3 = self.fem_solver.elements_v[f, self.fem_solver.surface[i_s].tri2v[2], i_b].pos
            u = p2 - p1
            v = p3 - p1
            surface_normal = qd.math.cross(u, v)
            surface_normal = surface_normal / surface_normal.norm(gs.EPS)

            # FEM <-> Rigid
            if qd.static(self._rigid_fem):
                # NOTE: collision only on surface vertices
                for j in qd.static(range(3)):
                    iv = self.fem_solver.surface[i_s].tri2v[j]
                    vel_fem_sv = self._func_collide_with_rigid(
                        f,
                        self.fem_solver.elements_v[f, iv, i_b].pos,
                        self.fem_solver.elements_v[f + 1, iv, i_b].vel,
                        mass / 3.0,  # assume element mass uniformly distributed to vertices
                        i_b,
                        geoms_state,
                        geoms_info,
                        links_state,
                        rigid_info,
                        sdf_info,
                        collider_config,
                    )
                    self.fem_solver.elements_v[f + 1, iv, i_b].vel = vel_fem_sv

            # FEM <-> SPH TODO: this doesn't work well
            if qd.static(self._fem_sph):
                for j in qd.static(range(3)):
                    iv = self.fem_solver.surface[i_s].tri2v[j]
                    pos = self.fem_solver.elements_v[f, iv, i_b].pos
                    vel_fem_sv = self.fem_solver.elements_v[f + 1, iv, i_b].vel
                    mass_fem_sv = mass / 4.0

                    dx = self.sph_solver.hash_grid_cell_size  # self._dx
                    stencil_size = 2  # self._stencil_size

                    base = self.sph_solver.sh.pos_to_grid(pos - 0.5 * dx)

                    # ---------- SPH -> FEM ----------
                    sph_vel = qd.Vector([0.0, 0.0, 0.0])
                    colliding_particles = 0
                    for offset in qd.grouped(qd.ndrange(stencil_size, stencil_size, stencil_size)):
                        slot_idx = self.sph_solver.sh.grid_to_slot(base + offset)
                        for k in range(
                            self.sph_solver.sh.slot_start[slot_idx, i_b],
                            self.sph_solver.sh.slot_start[slot_idx, i_b] + self.sph_solver.sh.slot_size[slot_idx, i_b],
                        ):
                            if qd.abs(pos - self.sph_solver.particles_reordered.pos[k, i_b]).max() < dx * 0.5:
                                sph_vel += self.sph_solver.particles_reordered.vel[k, i_b]
                                colliding_particles += 1

                    if colliding_particles > 0:
                        vel_old = vel_fem_sv
                        vel_fem_sv_unprojected = sph_vel / colliding_particles
                        vel_fem_sv = (
                            vel_fem_sv_unprojected.dot(surface_normal) * surface_normal
                        )  # exclude tangential velocity

                        # ---------- FEM -> SPH ----------
                        delta_mv = mass_fem_sv * (vel_fem_sv - vel_old)

                        for offset in qd.grouped(qd.ndrange(stencil_size, stencil_size, stencil_size)):
                            slot_idx = self.sph_solver.sh.grid_to_slot(base + offset)
                            for k in range(
                                self.sph_solver.sh.slot_start[slot_idx, i_b],
                                self.sph_solver.sh.slot_start[slot_idx, i_b]
                                + self.sph_solver.sh.slot_size[slot_idx, i_b],
                            ):
                                if qd.abs(pos - self.sph_solver.particles_reordered.pos[k, i_b]).max() < dx * 0.5:
                                    self.sph_solver.particles_reordered[k, i_b].vel = (
                                        self.sph_solver.particles_reordered[k, i_b].vel
                                        - delta_mv / self.sph_solver.particles_info_reordered[k, i_b].mass
                                    )

                        self.fem_solver.elements_v[f + 1, iv, i_b].vel = vel_fem_sv

    @qd.kernel
    def sph_rigid(
        self,
        f: qd.i32,
        geoms_state: GeomsState,
        geoms_info: GeomsInfo,
        links_state: LinksState,
        links_info: LinksInfo,
        rigid_info: RigidInfo,
        sdf_info: SDFInfo,
        collider_config: qd.template(),
    ):
        for i_p, i_b in qd.ndrange(self.sph_solver._n_particles, self.sph_solver._B):
            if self.sph_solver.particles_ng_reordered[i_p, i_b].active:
                for i_g in range(self.rigid_solver.n_geoms):
                    if geoms_info.needs_coup[i_g]:
                        (
                            self.sph_solver.particles_reordered[i_p, i_b].vel,
                            self.sph_rigid_normal_reordered[i_p, i_g, i_b],
                        ) = self._func_collide_with_rigid_geom_robust(
                            self.sph_solver.particles_reordered[i_p, i_b].pos,
                            self.sph_solver.particles_reordered[i_p, i_b].vel,
                            self.sph_solver.particles_info_reordered[i_p, i_b].mass,
                            self.sph_solver.particles_reordered[i_p, i_b].p,
                            self.sph_rigid_normal_reordered[i_p, i_g, i_b],
                            i_g,
                            i_b,
                            geoms_state,
                            geoms_info,
                            links_state,
                            links_info,
                            rigid_info,
                            sdf_info,
                            collider_config,
                        )

    @qd.kernel
    def kernel_pbd_rigid_collide(
        self,
        geoms_state: GeomsState,
        geoms_info: GeomsInfo,
        links_state: LinksState,
        sdf_info: SDFInfo,
        rigid_info: RigidInfo,
        collider_config: qd.template(),
    ):
        for i_p, i_b in qd.ndrange(self.pbd_solver._n_particles, self.sph_solver._B):
            if self.pbd_solver.particles_ng_reordered[i_p, i_b].active:
                # NOTE: Couldn't figure out a good way to handle collision with non-free particle. Such collision is not phsically plausible anyway.
                for i_g in range(self.rigid_solver.n_geoms):
                    if geoms_info.needs_coup[i_g]:
                        (
                            self.pbd_solver.particles_reordered[i_p, i_b].pos,
                            self.pbd_solver.particles_reordered[i_p, i_b].vel,
                            self.pbd_rigid_normal_reordered[i_p, i_b, i_g],
                        ) = self._func_pbd_collide_with_rigid_geom(
                            i_p,
                            self.pbd_solver.particles_reordered[i_p, i_b].pos,
                            self.pbd_solver.particles_reordered[i_p, i_b].vel,
                            self.pbd_solver.particles_info_reordered[i_p, i_b].mass,
                            self.pbd_rigid_normal_reordered[i_p, i_b, i_g],
                            i_g,
                            i_b,
                            geoms_state,
                            geoms_info,
                            links_state,
                            sdf_info,
                            rigid_info,
                            collider_config,
                        )

    @qd.kernel
    def kernel_attach_pbd_to_rigid_link(
        self, particles_idx: qd.types.ndarray(), envs_idx: qd.types.ndarray(), link_idx: qd.i32, links_state: LinksState
    ) -> None:
        """
        Sets listed particles in listed environments to be animated by the link.

        Current position of the particle, relatively to the link, is stored and preserved.
        """
        pdb = self.pbd_solver

        for i_p_, i_b_ in qd.ndrange(particles_idx.shape[1], envs_idx.shape[0]):
            i_p = particles_idx[i_b_, i_p_]
            i_b = envs_idx[i_b_]
            link_pos = links_state.pos[link_idx, i_b]
            link_quat = links_state.quat[link_idx, i_b]

            # compute local offset from link to the particle
            world_pos = pdb.particles[i_p, i_b].pos
            local_pos = qd_inv_transform_by_trans_quat(world_pos, link_pos, link_quat)

            # set particle to be animated (not free) and store animation info
            pdb.particles[i_p, i_b].free = False
            self.particle_attach_info[i_p, i_b].link_idx = link_idx
            self.particle_attach_info[i_p, i_b].local_pos = local_pos

    @qd.kernel
    def kernel_pbd_rigid_clear_animate_particles_by_link(
        self, particles_idx: qd.types.ndarray(), envs_idx: qd.types.ndarray()
    ) -> None:
        """Detach listed particles from links, and simulate them freely."""
        pdb = self.pbd_solver
        for i_p_, i_b_ in qd.ndrange(particles_idx.shape[1], envs_idx.shape[0]):
            i_p = particles_idx[i_b_, i_p_]
            i_b = envs_idx[i_b_]
            pdb.particles[i_p, i_b].free = True
            self.particle_attach_info[i_p, i_b].link_idx = -1
            self.particle_attach_info[i_p, i_b].local_pos = qd.math.vec3([0.0, 0.0, 0.0])

    @qd.kernel
    def kernel_pbd_rigid_solve_animate_particles_by_link(self, clamped_inv_dt: float, links_state: LinksState):
        """
        Itearates all particles and environments, and sets corrective velocity for all animated particle.

        Computes target position and velocity from the attachment/reference link and local offset position.

        Note, that this step shoudl be done after rigid solver update, and before PDB solver update.
        Currently, this is done after both rigid and PBD solver updates, hence the corrective velocity
        is off by a frame.

        Note, it's adviced to clamp inv_dt to avoid large jerks and instability. 1/0.02 might be a good max value.
        """
        pdb = self.pbd_solver
        for i_p, i_env in qd.ndrange(pdb._n_particles, pdb._B):
            if self.particle_attach_info[i_p, i_env].link_idx >= 0:
                # read link state
                link_idx = self.particle_attach_info[i_p, i_env].link_idx
                link_pos = links_state.pos[link_idx, i_env]
                link_quat = links_state.quat[link_idx, i_env]

                link_lin_vel = links_state.cd_vel[link_idx, i_env]
                link_ang_vel = links_state.cd_ang[link_idx, i_env]
                link_com_in_world = links_state.root_COM[link_idx, i_env] + links_state.i_pos[link_idx, i_env]

                # calculate target pos and vel of the particle
                local_pos = self.particle_attach_info[i_p, i_env].local_pos
                target_world_pos = qd_transform_by_trans_quat(local_pos, link_pos, link_quat)

                world_arm = target_world_pos - link_com_in_world
                target_world_vel = link_lin_vel + link_ang_vel.cross(world_arm)

                # compute and apply corrective velocity
                i_rp = pdb.particles_ng[i_p, i_env].reordered_idx
                particle_pos = pdb.particles_reordered[i_rp, i_env].pos
                pos_correction = target_world_pos - particle_pos
                corrective_vel = pos_correction * clamped_inv_dt
                pdb.particles_reordered[i_rp, i_env].vel = corrective_vel + target_world_vel

    @qd.func
    def _func_pbd_collide_with_rigid_geom(
        self,
        i,
        pos_world,
        vel,
        mass,
        normal_prev,
        geom_idx,
        batch_idx,
        geoms_state: GeomsState,
        geoms_info: GeomsInfo,
        links_state: LinksState,
        sdf_info: SDFInfo,
        rigid_info: RigidInfo,
        collider_config: qd.template(),
    ):
        """
        Resolves collision when a particle is already in collision with a rigid object.
        This function assumes known normal_rigid and influence.
        """
        signed_dist = sdf.sdf_func_world(geom_idx, batch_idx, pos_world, geoms_state, geoms_info, sdf_info)
        contact_normal = sdf.sdf_func_normal_world(
            geom_idx, batch_idx, pos_world, geoms_state, geoms_info, rigid_info, sdf_info, collider_config
        )
        new_pos = pos_world
        new_vel = vel
        if signed_dist < self.pbd_solver.particle_size / 2:  # skip non-penetration particles
            stiffness = 1.0  # value in [0, 1]

            # we don't consider friction for now
            # friction = 0.15
            # vel_rigid = self.rigid_solver._func_vel_at_point(
            #     pos_world=pos_world,
            #     link_idx=geoms_info.link_idx[geom_idx],
            #     i_b=batch_idx,
            #     links_state=links_state,
            # )
            # rvel = vel - vel_rigid
            # rvel_normal_magnitude = rvel.dot(contact_normal)  # negative if inward
            # rvel_tan = rvel - rvel_normal_magnitude * contact_normal
            # rvel_tan_norm = rvel_tan.norm(gs.EPS)

            #################### rigid -> particle ####################

            energy_loss = 0.0  # value in [0, 1]
            new_pos = pos_world + stiffness * contact_normal * (self.pbd_solver.particle_size / 2 - signed_dist)
            prev_pos = self.pbd_solver.particles_reordered[i, batch_idx].ipos
            new_vel = (new_pos - prev_pos) / self.pbd_solver._substep_dt

            #################### particle -> rigid ####################
            delta_mv = mass * (new_vel - vel)
            force = (-delta_mv / self.rigid_solver._substep_dt) * (1 - energy_loss)

            self.rigid_solver._func_apply_coupling_force(
                geoms_info.link_idx[geom_idx], batch_idx, pos_world, force, links_state
            )

        return new_pos, new_vel, contact_normal

    def preprocess(self, f):
        pass

    def couple_sph_rigid(self, f):
        """The collision of the SPH particles with Rigid, at the end of every SPH integration step."""
        if self._rigid_sph:
            self.sph_rigid(
                f,
                self.rigid_solver.dyn_state.geoms,
                self.rigid_solver.dyn_info.geoms,
                self.rigid_solver.dyn_state.links,
                self.rigid_solver.dyn_info.links,
                self.rigid_solver.rigid_info,
                self.rigid_solver.collider.collider_info.sdf,
                self.rigid_solver.collider.collider_config,
            )

    def couple(self, f):
        # SPH <-> Rigid
        self.couple_sph_rigid(f)

        # PBD <-> Rigid
        if self._rigid_pbd:
            self.kernel_pbd_rigid_collide(
                geoms_state=self.rigid_solver.dyn_state.geoms,
                geoms_info=self.rigid_solver.dyn_info.geoms,
                links_state=self.rigid_solver.dyn_state.links,
                sdf_info=self.rigid_solver.collider.collider_info.sdf,
                rigid_info=self.rigid_solver.rigid_info,
                collider_config=self.rigid_solver.collider.collider_config,
            )

            # 1-way: animate particles by links, over the substep interval and no faster than the clamp allows.
            substep_inv_dt = 1.0 / self.pbd_solver.substep_dt
            clamped_inv_dt = min(substep_inv_dt, CLAMPED_INV_DT)
            self.kernel_pbd_rigid_solve_animate_particles_by_link(clamped_inv_dt, self.rigid_solver.dyn_state.links)

        if self.fem_solver.is_active and not self._is_mpm_fem_coupled:
            self.couple_fem(f)

        # The impulses of the SPH particles on the MPM particles enter the next MPM particle-to-grid transfer, which the
        # Newton solve of the substep runs
        if self._is_mpm_sph_coupled:
            self._sph_boundary.apply_impulse(f)

    def couple_fem(self, f):
        """The coupling of FEM's end-of-substep velocities with Rigid and SPH.

        A deferred FEM substep (see FemNewtonHost) runs it after the joint solve that gives the velocities."""
        self.fem_surface_force(
            f,
            self.rigid_solver.dyn_state.geoms,
            self.rigid_solver.dyn_info.geoms,
            self.rigid_solver.dyn_state.links,
            self.rigid_solver.rigid_info,
            self.rigid_solver.collider.collider_info.sdf,
            self.rigid_solver.collider.collider_config,
        )

    @property
    def is_differentiable(self) -> bool:
        """
        Whether the coupler supports a scene with `requires_grad` (see 'Solver.is_differentiable').

        The couplings of the FEM solver with the rigid and SPH solvers carry no gradient: each surface triangle of an
        FEM entity overwrites the velocity of the vertices it shares with its neighbours, which the reverse pass cannot
        undo without a race. The coupling of FEM with MPM, in the joint Newton solve of their implicit solvers, carries
        none either.
        """
        return not (self._rigid_fem or self._fem_sph)

    @property
    def active_solvers(self):
        """All the active solvers managed by the scene's simulator."""
        return self.sim.active_solvers

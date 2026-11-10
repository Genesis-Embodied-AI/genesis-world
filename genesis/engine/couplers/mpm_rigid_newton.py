"""The contact pairs between the material point method (MPM) and Rigid as a pair kind of the NewtonSolver (see
newton_contact.py), on the participants of MPM and Rigid (see RigidSolver.build_newton_participant)."""

from collections.abc import Iterator

import quadrants as qd

import genesis as gs
from genesis.engine.solvers.mpm_func import (
    contact_friction,
    contact_gap,
    is_within_band,
    is_within_reach,
    particle_invweight,
    particle_of_sorted,
)
from genesis.engine.solvers.mpm_solver import MpmNewtonParticipant
from genesis.engine.solvers.newton_contact import DST_GRADIENT, MAX_VERTICES, SRC_ITERATE, VertexRange
from genesis.engine.solvers.newton_linear import MIN_SLOTS
from genesis.engine.solvers.rigid.abd.forward_dynamics import func_vel_at_point
from genesis.engine.solvers.rigid.newton_func import constraint_force, func_link_force_to_dofs, func_link_spatial_disp
from genesis.engine.solvers.rigid.newton_participant import RigidNewtonParticipant
from genesis.utils.data import DataItem, iter_data
from genesis.utils.misc import qd_to_numpy

from . import mpm_rigid_adjoint_func
from .contact_candidates import CouplingPairs, CouplingPairsHost, pair_relative_displacement
from .mpm_rigid_func import (
    MpmRigidPairs,
    RigidGeomQuery,
    contact_arm,
    geom_normal,
    geom_sdf,
    point_motion,
    point_stiffness,
    point_wrench,
    rigid_pair_rows,
    rigid_row_coefficients,
)


@qd.func
def has_force(g: qd.types.vector(3)):
    return g.norm_sqr() > 0.0


@qd.func
def gradient_force(g: qd.types.vector(3), dt: float):
    """Force of a gradient of the incremental potential, which scales forces by dt^2."""
    return g / (dt * dt)


@qd.data_oriented
class MpmRigidContact(CouplingPairs):
    """Pair kind of the contacts between the material point method (MPM) and Rigid, the pair of a particle and its
    nearest coupled geom."""

    pairs_type = MpmRigidPairs

    def __init__(
        self, mpm: MpmNewtonParticipant, rigid: RigidNewtonParticipant, query: RigidGeomQuery, has_rigid_dofs: bool
    ):
        self.rigid = rigid
        self.query = query
        self.has_rigid_dofs = has_rigid_dofs
        # A pair writes on its particle, and with Rigid dofs on its contact point and the block between them
        self.n_doublets = 2 if has_rigid_dofs else 1
        self.n_triplets = 3 if has_rigid_dofs else 1
        # The contact point of every pair slot is a contact vertex
        self.vertex_range = VertexRange(values={"offset": 0, "count": MIN_SLOTS})
        super().__init__(mpm)

    @property
    def data(self) -> Iterator[DataItem]:
        yield from super().data
        yield from iter_data(self, "vertex_range")
        yield from iter_data(self.query, "coupled_geoms")

    def build_query(self, f: int) -> None:
        self.query.build(self.mpm.mpm_info)

    def grow_pairs(self, capacity: int) -> None:
        # The contact points come last in the vertices, so that their growth moves no other vertex map
        if int(qd_to_numpy(self.vertex_range.offset)) + capacity > MAX_VERTICES:
            gs.raise_exception(
                f"The contacts of the implicit solve need more than the {MAX_VERTICES} vertices that a 32-bit index "
                "addresses: reduce the number of environments or of particles."
            )
        super().grow_pairs(capacity)
        self.vertex_range.count.fill(capacity)

    @qd.func
    def candidate_score(self, f: int, i_b: int, i_a: int, i_ps: int):
        """Gap of a sorted particle to the contact range of a coupled geom."""
        i_g = self.query.coupled_geoms.idx[i_a]
        i_p, _, _ = particle_of_sorted(i_ps, self.mpm.grid, self.mpm.mpm_info)
        x_p = self.mpm.particles.pos[f, i_p, i_b]
        phi = geom_sdf(i_g, i_b, x_p, self.rigid.dyn_state.geoms, self.rigid.dyn_info.geoms, self.rigid.sdf_info)
        return contact_gap(phi, self.rigid.dyn_info.geoms.coup_softness[i_g]), 0.0

    @qd.func
    def build_pair(self, f: int, i_b: int, i_a: int, i_ps: int, phi: float):
        """Pair a sorted particle with a coupled geom if it lies within its soft range or short of it by less than
        `contact_band_width`."""
        i_g = self.query.coupled_geoms.idx[i_a]
        i_p, _, _ = particle_of_sorted(i_ps, self.mpm.grid, self.mpm.mpm_info)
        x_p = self.mpm.particles.pos[f, i_p, i_b]
        v_p = self.mpm.particles.vel[f, i_p, i_b]
        dt = self.mpm.mpm_info.substep_dt[()]
        dx = self.mpm.mpm_info.dx[()]
        i_l = self.rigid.dyn_info.geoms.link_idx[i_g]
        v_r = func_vel_at_point(i_l, i_b, x_p, self.rigid.dyn_state.links)
        # The relative speed bounds the normal approach speed, which needs the normal
        if is_within_reach(phi, v_p, v_r, dt, dx):
            normal = geom_normal(
                i_g,
                i_b,
                x_p,
                self.rigid.dyn_state.geoms,
                self.rigid.dyn_info.geoms,
                self.rigid.rigid_info,
                self.rigid.sdf_info,
                self.rigid.config,
            )
            if is_within_band(phi, normal, v_p, v_r, dt, dx):
                I_l = [i_l, i_b] if qd.static(self.rigid.config.rigid_batch_links_info) else i_l
                friction = contact_friction(
                    self.rigid.dyn_info.geoms.friction[i_g],
                    self.rigid.dyn_state.geoms.friction_ratio[i_g, i_b],
                    self.mpm.mpm_info.contact_min_friction[()],
                )
                diag_coef, diag_coef_scale, sol_params_row = rigid_row_coefficients(
                    friction,
                    self.rigid.rigid_info.impratio[None],
                    particle_invweight(f, i_p, i_b, normal, self.mpm.particles, self.mpm.particles_info),
                    self.rigid.dyn_info.links.invweight[I_l][0],
                    self.rigid.dyn_info.geoms.sol_params[i_g],
                    dt,
                )
                stiffness, target = rigid_pair_rows(normal, friction, diag_coef, sol_params_row, phi, v_p, v_r, dt)
                i_pair = self.add_pair(i_ps, i_b, normal, friction, stiffness, target)
                if i_pair >= 0:
                    self.pairs.geom_idx[i_pair] = i_g
                    self.pairs.link_idx[i_pair] = i_l
                    self.pairs.phi[i_pair] = phi

    @qd.func
    def pair_vertices(self, f: int, i_pair: int):
        """The particle of a pair, of weight 1, and with Rigid dofs its contact point, of weight -1, with their relative
        displacement at the current iterate."""
        i_v_mpm = self.mpm.vertex_range.offset[()] + self.pairs.sorted_idx[i_pair]
        u_mpm = self.mpm.func_vertex_value(f, self.pairs.sorted_idx[i_pair])
        if qd.static(self.has_rigid_dofs):
            verts = qd.Vector([i_v_mpm, self.vertex_range.offset[()] + i_pair], dt=gs.qd_int)
            weights = qd.Vector([1.0, -1.0], dt=gs.qd_float)
            values = qd.Matrix.rows([u_mpm, self.func_vertex_value(f, i_pair, SRC_ITERATE)])
            return verts, weights, pair_relative_displacement(weights, values)
        else:
            return qd.Vector([i_v_mpm], dt=gs.qd_int), qd.Vector([1.0], dt=gs.qd_float), u_mpm

    # ------------------------------------------- vertex map -------------------------------------------

    @qd.func
    def contact_point_arm(self, f: int, i_pair: int):
        """Link, environment and arm from the root center of mass (COM) of the link of the contact point of a pair, at
        the position of its particle at the substep start."""
        i_p, _, _ = particle_of_sorted(self.pairs.sorted_idx[i_pair], self.mpm.grid, self.mpm.mpm_info)
        i_l = self.pairs.link_idx[i_pair]
        i_b = self.pairs.i_b[i_pair]
        return i_l, i_b, contact_arm(self.mpm.particles.pos[f, i_p, i_b], self.rigid.dyn_state.links.root_COM[i_l, i_b])

    @qd.func(requires_top_level=True)
    def func_vertex_begin(self, f: int, src: qd.template()):
        if qd.static(src == SRC_ITERATE):
            func_link_spatial_disp(
                src=self.rigid.rigid_data.x,
                rigid_data=self.rigid.rigid_data,
                dofs_state=self.rigid.dyn_state.dofs,
                links_info=self.rigid.dyn_info.links,
                config=self.rigid.config,
            )
        else:
            func_link_spatial_disp(
                src=self.rigid.rigid_data.p,
                rigid_data=self.rigid.rigid_data,
                dofs_state=self.rigid.dyn_state.dofs,
                links_info=self.rigid.dyn_info.links,
                config=self.rigid.config,
            )

    @qd.func(requires_top_level=True)
    def func_vertex_end(self, dst: qd.template()):
        if qd.static(dst == DST_GRADIENT):
            func_link_force_to_dofs(
                scale=1.0,
                out=self.rigid.rigid_data.grad,
                rigid_data=self.rigid.rigid_data,
                dofs_state=self.rigid.dyn_state.dofs,
                links_info=self.rigid.dyn_info.links,
                config=self.rigid.config,
            )
        else:
            func_link_force_to_dofs(
                scale=1.0,
                out=self.rigid.rigid_data.ap,
                rigid_data=self.rigid.rigid_data,
                dofs_state=self.rigid.dyn_state.dofs,
                links_info=self.rigid.dyn_info.links,
                config=self.rigid.config,
            )

    @qd.func
    def func_vertex_env(self, i_v: int):
        return self.pairs.i_b[i_v]

    @qd.func
    def func_vertex_value(self, f: int, i_v: int, src: qd.template()):
        # The link displacements of `src` are those of func_vertex_begin
        i_l, i_b, r = self.contact_point_arm(f, i_v)
        return point_motion(
            self.rigid.rigid_data.links_motion_vel[i_l, i_b], self.rigid.rigid_data.links_motion_ang[i_l, i_b], r
        )

    @qd.func
    def func_vertex_add_force(self, f: int, i_v: int, y: qd.types.vector(3)):
        """Add the spatial force of a vector on a contact point to its link, which func_vertex_end maps to the dofs."""
        i_l, i_b, r = self.contact_point_arm(f, i_v)
        force, torque = point_wrench(y, r)
        qd.atomic_add(self.rigid.rigid_data.links_force_vel[i_l, i_b], force)
        qd.atomic_add(self.rigid.rigid_data.links_force_ang[i_l, i_b], torque)

    @qd.func
    def func_vertex_visit_diag(self, f: int, i_v: int, H: qd.types.matrix(3, 3)):
        # The spatial stiffness K_l = sum P^T H P of every link joins the factor of Rigid (see
        # func_add_island_hessian_batch)
        i_l, i_b, r = self.contact_point_arm(f, i_v)
        qd.atomic_add(self.rigid.rigid_data.links_stiffness[i_l, i_b], point_stiffness(H, r))


@qd.kernel
def kernel_mpm_reaction(f: int, contact: qd.template()):
    """Apply the force of the pairs on Rigid at the converged solve, after func_rigid_write_back."""
    contact.func_vertex_begin(f, SRC_ITERATE)
    qd.loop_config(name="mpm_reaction_pairs")
    for i_pair in range(contact.pairs.n_pairs[()]):
        energy, g, H, verts, weights = contact.pair_terms(f, i_pair)
        if has_force(g):
            contact.func_vertex_add_force(f, i_pair, g)
            contact.rigid.dyn_state.links.is_constrained[contact.pairs.link_idx[i_pair], contact.pairs.i_b[i_pair]] = (
                True
            )
    qd.loop_config(name="mpm_reaction_qfrc")
    for i_d, i_b in qd.ndrange(contact.rigid.rigid_data.x.shape[0], contact.rigid.rigid_data.x.shape[1]):
        contact.rigid.constraint_state.qfrc_constraint[i_d, i_b] = constraint_force(
            contact.rigid.rigid_data.Ma[i_d, i_b], contact.rigid.dyn_state.dofs.qf_smooth[i_d, i_b]
        )


@qd.kernel
def kernel_add_mpm_contact_force(contact: qd.template()):
    """Add the force of the pairs on each link (see kernel_mpm_reaction) to `contact_force`, after Rigid's own contact
    forces were written by the post-solve stage of its substep."""
    dt = contact.mpm.mpm_info.substep_dt[()]
    n_links = contact.rigid.rigid_data.links_force_vel.shape[0]
    for i_l, i_b in qd.ndrange(n_links, contact.rigid.rigid_data.links_force_vel.shape[1]):
        contact_force = contact.rigid.dyn_state.links.contact_force[i_l, i_b]
        contact.rigid.dyn_state.links.contact_force[i_l, i_b] = contact_force + gradient_force(
            contact.rigid.rigid_data.links_force_vel[i_l, i_b], dt
        )


class MpmRigidContactHost(CouplingPairsHost):
    """Host stages of the MPM-Rigid contact pairs around the joint solve: the pairs of the substep, their pullback, and
    with Rigid dofs their force on Rigid at the converged solve and its share of Rigid's contact forces."""

    is_differentiable = True

    def newton_finish(self, f, control):
        if self._device.has_rigid_dofs:
            kernel_mpm_reaction(f, self._device)

    def newton_after(self, f):
        if self._device.has_rigid_dofs:
            kernel_add_mpm_contact_force(self._device)

    def newton_adjoint_finish(self, f):
        """Pull the contact pairs back into the adjoints of the particles and of Rigid's state."""
        mpm_rigid_adjoint_func.kernel_rigid_contact_adjoint(
            f,
            self._device.mpm.particles,
            self._device.mpm.grid,
            self._device.mpm.newton,
            self._device.mpm.linear,
            self._device.mpm.linear_block,
            self._device.pairs,
            self._device.rigid.rigid_data,
            self._device.rigid.dyn_state.geoms,
            self._device.rigid.dyn_state.links,
            self._device.rigid.dyn_state.dofs,
            self._device.mpm.particles_info,
            self._device.rigid.dyn_info.geoms,
            self._device.rigid.dyn_info.links,
            self._device.rigid.rigid_info,
            self._device.rigid.sdf_info,
            self._device.mpm.mpm_info,
            self._device.rigid.config,
        )

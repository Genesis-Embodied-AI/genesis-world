"""The contact pairs between the material point method (MPM) and Rigid as a pair kind of the NewtonSolver (see
newton_contact.py).

This module binds the stage functions of `mpm_rigid_func` to the data of one scene, on the participants of MPM and
Rigid (see RigidSolver.build_newton_participant). The coupling builds the pairs of the particles with the coupled geoms
every substep. With Rigid dofs, it is also the vertex map of their contact points, and on the geoms of a Rigid without
dofs the pairs act on the particle displacement alone.
"""

import quadrants as qd

import genesis as gs
from genesis.engine.solvers.mpm_solver import MpmNewtonParticipant
from genesis.engine.solvers.newton_contact import DST_GRADIENT, MAX_VERTICES, SRC_ITERATE
from genesis.engine.solvers.newton_linear import MIN_SLOTS
from genesis.engine.solvers.rigid.newton_func import func_link_force_to_dofs, func_link_spatial_disp
from genesis.engine.solvers.rigid.newton_participant import RigidNewtonParticipant

from . import mpm_rigid_adjoint_func, mpm_rigid_func
from .contact_candidates import CouplingPairs, CouplingPairsHost


@qd.data_oriented
class MpmRigidContact(CouplingPairs):
    """Pair kind of the contacts between the material point method (MPM) and Rigid, the pair of a particle and its
    nearest coupled geom, and with `has_rigid_dofs` the vertex map of their contact points on the links.

    `rigid` is the participant of Rigid, whose dofs the contact points move with. A Rigid without dofs gives a
    participant that is no member of the solve, and a pair then writes its gradient and Hessian on its particle
    alone."""

    build_pairs = staticmethod(mpm_rigid_func.build_rigid_pairs)

    def __init__(
        self,
        mpm: MpmNewtonParticipant,
        rigid: RigidNewtonParticipant,
        query: mpm_rigid_func.RigidGeomQuery,
        has_rigid_dofs: bool,
    ):
        self.rigid = rigid
        self.query = query
        self.has_rigid_dofs = has_rigid_dofs
        # A pair writes on its particle, and with Rigid dofs on its contact point and the block between them
        self.n_doublets = 2 if has_rigid_dofs else 1
        self.n_triplets = 3 if has_rigid_dofs else 1
        # The contact point of every pair slot is a contact vertex
        self.vertex_offset = qd.ndarray(gs.qd_int, shape=())
        self.vertex_offset.fill(0)
        self.vertex_count = qd.ndarray(gs.qd_int, shape=())
        self.vertex_count.fill(MIN_SLOTS)
        super().__init__(mpm)

    def build_query(self, f: int) -> None:
        self.query.build(self.mpm.mpm_info)

    def grow_pairs(self, capacity: int) -> None:
        # The contact points come last in the vertices, so that their growth moves no other vertex map
        if int(self.vertex_offset.to_numpy()) + capacity > MAX_VERTICES:
            gs.raise_exception(
                f"The contacts of the implicit solve need more than the {MAX_VERTICES} vertices that a 32-bit index "
                "addresses: reduce the number of environments or of particles."
            )
        super().grow_pairs(capacity)
        self.vertex_count.fill(capacity)

    @qd.func(requires_top_level=True)
    def func_query(self, f: int):
        mpm_rigid_func.func_query_geoms(
            f,
            self.mpm.mpm_info.substep_dt[()],
            self.mpm.particles,
            self.mpm.grid,
            self.query,
            self.candidates,
            self.mpm.mpm_info,
        )

    @qd.func(requires_top_level=True)
    def func_score(self, f: int):
        mpm_rigid_func.func_score_geoms(
            f, self.mpm.particles, self.mpm.grid, self.query, self.candidates, self.mpm.mpm_info
        )

    @qd.func(requires_top_level=True)
    def func_build_pairs(self, f: int):
        mpm_rigid_func.func_build_rigid_pairs(
            f,
            self.mpm.mpm_info.substep_dt[()],
            self.mpm.particles,
            self.mpm.grid,
            self.pairs,
            self.rigid.dyn_state.geoms,
            self.rigid.dyn_state.links,
            self.query,
            self.candidates,
            self.mpm.particles_info,
            self.rigid.dyn_info.geoms,
            self.rigid.dyn_info.links,
            self.rigid.rigid_info,
            self.rigid.sdf_info,
            self.mpm.mpm_info,
            self.rigid.config,
        )

    @qd.func(requires_top_level=True)
    def func_pair_terms(self, f: int, i_kind: qd.template(), contact: qd.template()):
        mpm_rigid_func.func_rigid_pair_terms(
            f,
            contact.matrices.doublet_offsets[i_kind],
            contact.matrices.triplet_offsets[i_kind],
            self.mpm,
            self,
            contact,
        )

    @qd.func(requires_top_level=True)
    def func_pair_energy(self, f: int, energies: qd.template()):
        mpm_rigid_func.func_rigid_pair_energy(f, self.mpm, self, energies)

    # ------------------------------------------- vertex map -------------------------------------------

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
        i_l, i_b, r = mpm_rigid_func.contact_point_arm(
            f, i_v, self.mpm.particles, self.mpm.grid, self.pairs, self.rigid.dyn_state.links, self.mpm.mpm_info
        )
        return self.rigid.rigid_data.link_motion_vel[i_l, i_b] + self.rigid.rigid_data.link_motion_ang[i_l, i_b].cross(
            r
        )

    @qd.func
    def func_vertex_add_force(self, f: int, i_v: int, y: qd.types.vector(3)):
        """Add the spatial force (y, r x y) of a vector on a contact point to its link, which func_vertex_end maps to
        the dofs."""
        i_l, i_b, r = mpm_rigid_func.contact_point_arm(
            f, i_v, self.mpm.particles, self.mpm.grid, self.pairs, self.rigid.dyn_state.links, self.mpm.mpm_info
        )
        qd.atomic_add(self.rigid.rigid_data.link_force_vel[i_l, i_b], y)
        qd.atomic_add(self.rigid.rigid_data.link_force_ang[i_l, i_b], r.cross(y))

    @qd.func
    def func_vertex_visit_diag(self, f: int, i_v: int, H: qd.types.matrix(3, 3)):
        # The spatial stiffness K_l = sum P^T H P of every link feeds Rigid's preconditioner (see func_rigid_factor)
        i_l, i_b, r = mpm_rigid_func.contact_point_arm(
            f, i_v, self.mpm.particles, self.mpm.grid, self.pairs, self.rigid.dyn_state.links, self.mpm.mpm_info
        )
        P = mpm_rigid_func.link_point_jacobian(r)
        qd.atomic_add(self.rigid.rigid_data.link_K[i_l, i_b], P.transpose() @ H @ P)


@qd.kernel
def kernel_mpm_reaction(f: int, contact: qd.template()):
    mpm_rigid_func.func_mpm_reaction(
        f,
        contact.mpm,
        contact,
        contact.rigid.rigid_data,
        contact.rigid.dyn_state,
        contact.rigid.constraint_state,
        contact.rigid.config,
    )


@qd.kernel
def kernel_add_mpm_contact_force(contact: qd.template()):
    mpm_rigid_func.func_add_mpm_contact_force(
        contact.rigid.rigid_data, contact.rigid.dyn_state.links, contact.mpm.mpm_info
    )


@qd.kernel
def kernel_rigid_contact_adjoint(f: int, contact: qd.template()):
    mpm_rigid_adjoint_func.func_rigid_contact_adjoint(
        f,
        contact.mpm.particles,
        contact.mpm.grid,
        contact.mpm.newton,
        contact.mpm.linear,
        contact.mpm.linear_block,
        contact.pairs,
        contact.rigid.rigid_data,
        contact.rigid.dyn_state.geoms,
        contact.rigid.dyn_state.links,
        contact.rigid.dyn_state.dofs,
        contact.mpm.particles_info,
        contact.rigid.dyn_info.geoms,
        contact.rigid.dyn_info.links,
        contact.rigid.rigid_info,
        contact.rigid.sdf_info,
        contact.mpm.mpm_info,
        contact.rigid.config,
    )


class MpmRigidContactHost(CouplingPairsHost):
    """Host stages of the MPM-Rigid contact pairs around the joint solve.

    They build the pairs of the substep, and the backward pass pulls them back. With Rigid dofs, they also apply the
    force of the MPM pairs on Rigid at the converged solve, and add its share to Rigid's reported contact forces once
    Rigid wrote its own."""

    is_differentiable = True

    def newton_finish(self, f, control):
        if self._device.has_rigid_dofs:
            kernel_mpm_reaction(f, self._device)

    def newton_after(self, f):
        if self._device.has_rigid_dofs:
            kernel_add_mpm_contact_force(self._device)

    def newton_adjoint_finish(self, f):
        """Pull the contact pairs back into the adjoints of the particles and of Rigid's state."""
        kernel_rigid_contact_adjoint(f, self._device)

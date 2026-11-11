"""
The coupling of the implicit material point method (MPM) solver with the rigid solver in the global Newton solve (see
NewtonSolver): the contacts of the particles with the rigid geoms at the iterate of the solve.

The contact of a particle acts on its displacement relative to the point of the link it touches. Its gradient and its
Hessian in that displacement are staged at the slot of the particle (see MPMRigidContacts), from which the MPM solver
and the rigid solver assemble their own blocks, the coupling between them going through the products of the
preconditioned conjugate gradient.
"""

import quadrants as qd

import genesis as gs
from genesis.engine.numerics.soft_contact import func_pyramid_terms
from genesis.engine.solvers.mpm_solver import (
    MPMInfo,
    MPMSparseGrid,
    MPMState,
    NewtonRows,
    func_particle_displacement,
    func_q1_stencil,
    func_sorted_particle,
    func_stencil_slots,
)
from genesis.engine.solvers.rigid.data import DynInfo, DynState
from genesis.engine.solvers.rigid.global_newton import (
    NewtonRigidDofs,
    NewtonRigidLinks,
    func_clear_link_forces,
    func_link_spatial_disp,
    func_point_motion,
    func_point_stiffness,
)


@qd.func
def func_contact_arm(i_s: int, i_g: int, mpm_state: MPMState, dyn_state: DynState, dyn_info: DynInfo):
    """Return the link of the geom a sorted particle touches, the environment and the arm of the particle from the root
    center of mass of the link."""
    i_p, i_b, cell = func_sorted_particle(i_s, mpm_state.particles.pos.shape[0], mpm_state.sorted_particles)
    i_l = dyn_info.geoms.link_idx[i_g]
    return i_l, i_b, mpm_state.particles.pos[i_p, i_b] - dyn_state.links.root_COM[i_l, i_b]


@qd.func
def func_contact_partner_disp(
    i_s: int, mpm_state: MPMState, rigid_links: NewtonRigidLinks, dyn_state: DynState, dyn_info: DynInfo
):
    """
    Return the displacement of the point of the rigid link a sorted particle touches, from the spatial displacement of
    the link (see 'func_link_spatial_disp' in rigid/global_newton.py), zero for a particle in no contact.
    """
    disp = qd.Vector.zero(gs.qd_float, 3)
    i_g = mpm_state.rigid_contacts.geom_idx[i_s]
    if i_g >= 0:
        i_l, i_b, r = func_contact_arm(i_s, i_g, mpm_state, dyn_state, dyn_info)
        disp = func_point_motion(rigid_links.motion_vel[i_l, i_b], rigid_links.motion_ang[i_l, i_b], r)
    return disp


@qd.func
def func_contact_terms(
    i_s: int,
    mpm_state: MPMState,
    sparse_grid: MPMSparseGrid,
    rigid_links: NewtonRigidLinks,
    dyn_state: DynState,
    mpm_info: MPMInfo,
    dyn_info: DynInfo,
    has_rigid_dofs: qd.template(),
):
    """
    Return the energy, the gradient and the Hessian of the contact of a sorted particle with a rigid geom at the current
    iterate, in the displacement of the particle relative to the point of the link it touches.

    The terms are zero for a particle in no contact. The rows of the contact are those staged at the slot of the
    particle (see MPMRigidContacts). The point of the link moves with the iterate of the dofs of the rigid solver when
    they join the unknowns (`has_rigid_dofs`, see 'func_link_spatial_disp' in rigid/global_newton.py).
    """
    energy = gs.qd_float(0.0)
    gradient = qd.Vector.zero(gs.qd_float, 3)
    hessian = qd.Matrix.zero(gs.qd_float, 3, 3)
    if mpm_state.rigid_contacts.geom_idx[i_s] >= 0:
        _, disp = func_particle_displacement(i_s, sparse_grid.nodes_u, mpm_state, sparse_grid, mpm_info)
        disp_partner = qd.Vector.zero(gs.qd_float, 3)
        if qd.static(has_rigid_dofs):
            disp_partner = func_contact_partner_disp(i_s, mpm_state, rigid_links, dyn_state, dyn_info)
        energy_rows, gradient_rows, hessian_rows = func_pyramid_terms(
            mpm_state.rigid_contacts.normal[i_s],
            mpm_state.rigid_contacts.friction[i_s],
            mpm_state.rigid_contacts.stiffness[i_s],
            mpm_state.rigid_contacts.target[i_s],
            disp - disp_partner,
        )
        energy = energy_rows
        gradient = gradient_rows
        hessian = hessian_rows
    return energy, gradient, hessian


@qd.func(requires_top_level=True)
def func_mark_constrained_links(mpm_state: MPMState, dyn_state: DynState, dyn_info: DynInfo):
    """Mark constrained over the substep every link whose geom holds the rows of the contact of a particle (see
    'LinksState.is_constrained' in rigid/data.py), as the contacts of the rigid solver mark theirs."""
    for i_s in range(mpm_state.sorted_particles.n_valid[None]):
        i_g = mpm_state.rigid_contacts.geom_idx[i_s]
        if i_g >= 0:
            i_l, i_b, _ = func_contact_arm(i_s, i_g, mpm_state, dyn_state, dyn_info)
            dyn_state.links.is_constrained[i_l, i_b] = True


@qd.func(requires_top_level=True)
def func_assemble_mpm_rigid_coupling(
    mpm_state: MPMState,
    sparse_grid: MPMSparseGrid,
    rigid_dofs: NewtonRigidDofs,
    rigid_links: NewtonRigidLinks,
    dyn_state: DynState,
    mpm_info: MPMInfo,
    dyn_info: DynInfo,
    rigid_config: qd.template(),
    has_rigid_dofs: qd.template(),
):
    """
    Stage the gradient and the Hessian of the contact of every sorted particle with a rigid geom at the current iterate.

    The MPM solver and the rigid solver assemble their own blocks from them (see 'func_mpm_assemble' in mpm_solver.py
    and 'func_add_contact_link_terms'). The terms lie in the displacement of the particle relative to the point of the
    link it touches (see 'func_contact_terms'): the particle takes the gradient and the Hessian, the point the opposite
    of the gradient and the Hessian, and the coupling between them the opposite of the Hessian, which the preconditioned
    conjugate gradient applies to its search direction (see 'func_contact_matvec').
    """
    if qd.static(has_rigid_dofs):
        func_link_spatial_disp(rigid_dofs.x, rigid_links, dyn_state.dofs, dyn_info.links, rigid_config)
    for i_s in range(mpm_state.sorted_particles.n_valid[None]):
        if mpm_state.rigid_contacts.geom_idx[i_s] >= 0:
            _, gradient, hessian = func_contact_terms(
                i_s, mpm_state, sparse_grid, rigid_links, dyn_state, mpm_info, dyn_info, has_rigid_dofs
            )
            mpm_state.rigid_contacts.gradient[i_s] = gradient
            mpm_state.rigid_contacts.hessian[i_s] = hessian


@qd.func(requires_top_level=True)
def func_add_contact_link_terms(
    mpm_state: MPMState, rigid_links: NewtonRigidLinks, dyn_state: DynState, dyn_info: DynInfo
):
    """Set the spatial force and the spatial stiffness of the contacts of the particles on every link, from the terms
    staged at the current iterate (see 'func_assemble_mpm_rigid_coupling')."""
    func_clear_link_forces(rigid_links)
    for i_l, i_b in qd.ndrange(rigid_links.stiffness.shape[0], rigid_links.stiffness.shape[1]):
        rigid_links.stiffness[i_l, i_b] = qd.Matrix.zero(gs.qd_float, 6, 6)
    for i_s in range(mpm_state.sorted_particles.n_valid[None]):
        i_g = mpm_state.rigid_contacts.geom_idx[i_s]
        if i_g >= 0:
            i_l, i_b, r = func_contact_arm(i_s, i_g, mpm_state, dyn_state, dyn_info)
            force = -mpm_state.rigid_contacts.gradient[i_s]
            qd.atomic_add(rigid_links.force_vel[i_l, i_b], force)
            qd.atomic_add(rigid_links.force_ang[i_l, i_b], r.cross(force))
            qd.atomic_add(
                rigid_links.stiffness[i_l, i_b], func_point_stiffness(mpm_state.rigid_contacts.hessian[i_s], r)
            )


@qd.func(requires_top_level=True)
def func_contact_matvec(
    newton_rows: NewtonRows,
    envs_mask: qd.Tensor,
    mpm_state: MPMState,
    sparse_grid: MPMSparseGrid,
    rigid_links: NewtonRigidLinks,
    dyn_state: DynState,
    mpm_info: MPMInfo,
    dyn_info: DynInfo,
):
    """
    Add the product of the coupling of the contacts with the search direction of the environments selected.

    The product between the nodes of a particle and the link it touches adds to the product of the rows, and the product
    between the link and itself to the spatial force on the link. The spatial displacement of the links is that of the
    search direction of the dofs (see 'func_link_spatial_disp' in rigid/global_newton.py). The blocks between the nodes
    of a particle are in the matrix of the rows already (see 'MPMRigidContacts.hessian' in mpm_solver.py).
    """
    func_clear_link_forces(rigid_links)
    n_particles = mpm_state.particles.pos.shape[0]
    inv_dx = mpm_info.inv_dx[None]
    for i_s in range(mpm_state.sorted_particles.n_valid[None]):
        i_g = mpm_state.rigid_contacts.geom_idx[i_s]
        if i_g >= 0:
            i_p, i_b, cell = func_sorted_particle(i_s, n_particles, mpm_state.sorted_particles)
            if envs_mask[i_b]:
                x_p = mpm_state.particles.pos[i_p, i_b]
                w, grad_w = func_q1_stencil(x_p * inv_dx - cell.cast(gs.qd_float), inv_dx)
                slots = func_stencil_slots(i_s, cell, mpm_state.sorted_particles, sparse_grid)
                weights = qd.Vector(
                    [w[a] * gs.qd_float(sparse_grid.nodes_is_fixed[slots[a]] == 0) for a in qd.static(range(8))]
                )
                p_particle = qd.Vector.zero(gs.qd_float, 3)
                for a in qd.static(range(8)):
                    p_particle += weights[a] * newton_rows.p[slots[a]]
                i_l = dyn_info.geoms.link_idx[i_g]
                r = x_p - dyn_state.links.root_COM[i_l, i_b]
                p_partner = func_point_motion(rigid_links.motion_vel[i_l, i_b], rigid_links.motion_ang[i_l, i_b], r)
                hessian = mpm_state.rigid_contacts.hessian[i_s]
                product_partner = hessian @ p_partner
                for a in qd.static(range(8)):
                    qd.atomic_add(newton_rows.Ap[slots[a]], -weights[a] * product_partner)
                force = product_partner - hessian @ p_particle
                qd.atomic_add(rigid_links.force_vel[i_l, i_b], force)
                qd.atomic_add(rigid_links.force_ang[i_l, i_b], r.cross(force))

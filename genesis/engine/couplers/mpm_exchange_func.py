"""Momentum exchanges of the implicit MPM solver with SPH and PBD.

Each exchange is a perfectly inelastic impulse between an MPM node and the partner particles in its cell, applied to
the free velocity of the node before the Newton solve, so that the solve enforces the walls, the Rigid contacts and the
elasticity on top of it: the pair moves to its common velocity v_c = (m_a v_a + M v) / (m_a + M), which conserves the
momentum and dissipates energy. The partner side keeps its relative velocities and takes the opposite momentum. The
neighbourhoods are the legacy coupler's: the partner particles in the cell of the node.
"""

import quadrants as qd

import genesis as gs
from genesis.engine.solvers.mpm_data import BLOCK_CELLS
from genesis.engine.solvers.mpm_func import local_offset, unpack_block


@qd.func(requires_top_level=True)
def func_exchange_particles(
    grid: qd.template(),  # MpmGrid
    params: qd.template(),  # MpmParams
    partner: qd.template(),  # SPHSolver or PBDSolver, after its particles were reordered by its spatial hash
    stencil: qd.template(),  # int, hash cells per axis covering one MPM cell
    only_free: qd.template(),  # bool, exchange with the free particles only (PBD)
):
    """Exchange momentum between every free MPM node and the partner particles in its cell, |x_s - x_n|_inf < dx / 2.

    A particle lies in the cell of one node only, so every node writes its own particles."""
    dx = params.dx[()]
    sh = partner.sh
    qd.loop_config(name="mpm_exchange_particles")
    for node in range(grid.n_blocks[()] * BLOCK_CELLS):
        if grid.node_mass[node] > 0.0 and grid.node_fixed[node] == 0:
            env, block = unpack_block(grid.block_coords[node // BLOCK_CELLS])
            x_n = (block * 4 + local_offset(node % BLOCK_CELLS)).cast(gs.qd_float) * dx
            base = sh.pos_to_grid(x_n - 0.5 * dx)
            mass_s = gs.qd_float(0.0)
            mom_s = qd.Vector.zero(gs.qd_float, 3)
            for offset in qd.grouped(qd.ndrange(stencil, stencil, stencil)):
                slot = sh.grid_to_slot(base + offset)
                for i in range(sh.slot_start[slot, env], sh.slot_start[slot, env] + sh.slot_size[slot, env]):
                    is_member = qd.abs(partner.particles_reordered.pos[i, env] - x_n).max() < 0.5 * dx
                    if qd.static(only_free):
                        is_member = is_member and partner.particles_reordered.free[i, env]
                    if is_member:
                        m_s = partner.particles_info_reordered.mass[i, env]
                        mass_s += m_s
                        mom_s += m_s * partner.particles_reordered.vel[i, env]
            if mass_s > 0.0:
                m_a = grid.node_mass[node]
                v_a = qd.Vector([grid.node_v_free[3 * node + d] for d in qd.static(range(3))], dt=gs.qd_float)
                v_c = (m_a * v_a + mom_s) / (m_a + mass_s)
                dv_s = v_c - mom_s / mass_s
                for d in qd.static(range(3)):
                    grid.node_v_free[3 * node + d] = v_c[d]
                for offset in qd.grouped(qd.ndrange(stencil, stencil, stencil)):
                    slot = sh.grid_to_slot(base + offset)
                    for i in range(sh.slot_start[slot, env], sh.slot_start[slot, env] + sh.slot_size[slot, env]):
                        is_member = qd.abs(partner.particles_reordered.pos[i, env] - x_n).max() < 0.5 * dx
                        if qd.static(only_free):
                            is_member = is_member and partner.particles_reordered.free[i, env]
                        if is_member:
                            partner.particles_reordered.vel[i, env] = partner.particles_reordered.vel[i, env] + dv_s

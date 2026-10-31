from enum import IntEnum
from typing import TYPE_CHECKING

import numpy as np
import torch

import quadrants as qd

import genesis as gs
import genesis.utils.array_class as array_class
import genesis.utils.geom as gu
from genesis.engine.boundaries import CubeBoundary
from genesis.engine.couplers.mpm_fem_func import FemTriangleQuery
from genesis.engine.couplers.mpm_fem_newton import (
    FemNewtonHost,
    FemNewtonParticipant,
    MpmFemContact,
    MpmFemContactHost,
)
from genesis.engine.couplers.mpm_pbd_func import (
    MpmParticleObstacles,
    MpmPbdContact,
    MpmPbdContactHost,
    PbdNewtonData,
    PbdNewtonHost,
    PbdNewtonParticipant,
    PbdParticleQuery,
)
from genesis.engine.couplers.mpm_rigid_func import RigidGeomQuery, func_build_rigid_rows
from genesis.engine.couplers.mpm_sph_func import MpmSphBoundary, kernel_apply_boundary_impulse
from genesis.engine.couplers.mpm_rigid_newton import (
    MpmRigidContact,
    MpmRigidContactHost,
    RigidNewtonHost,
    RigidNewtonParticipant,
    kernel_rigid_contact_adjoint,
)
from genesis.engine.entities import MPMEntity
from genesis.engine.materials import MPM
from genesis.engine.state_change import StateChange, mutates
from genesis.engine.numerics.bcoo_matrix import func_sort_reduce_bcoo, sort_reduce_bcoo
from genesis.engine.numerics.dynamic_exclusive_sum import kernel_dynamic_exclusive_sum
from genesis.engine.numerics.dynamic_radix_sort import dynamic_radix_sort
from genesis.engine.numerics.linear_pcg import (
    func_init_linear_pcg_direction,
    func_init_linear_pcg_residual,
    func_linear_pcg_direction_update,
    func_linear_pcg_spmv,
    func_linear_pcg_update_xr,
)
from genesis.engine.numerics.newton_control import (
    func_accept_or_backtrack,
    func_check_convergence,
    func_init_line_search,
    func_reset_newton_control,
    func_reset_newton_measures,
    func_update_convergence,
)
from genesis.engine.states.solvers import MPMSolverState
from genesis.options.solvers import LegacyCouplerOptions, MPMOptions
from genesis.grad.bridge import no_grad
from genesis.utils.misc import (
    DeprecationError,
    broadcast_tensor,
    qd_from_torch,
    qd_isfinite,
    qd_to_torch,
    qd_zero_grad,
)

from . import mpm_adjoint_func, mpm_func
from .mpm_data import (
    BLOCK_CELLS,
    MpmForceFields,
    MpmGrid,
    MpmLinearSystem,
    MpmMaterialTable,
    MpmNewton,
    MpmParams,
    MpmParticles,
    MpmParticlesInfo,
    MpmParticlesNG,
    MpmRigidCoupling,
    MpmStaticConfig,
    get_mpm_material_table,
    get_mpm_particles,
    get_mpm_particles_info,
    get_mpm_particles_ng,
    initial_capacities,
    triplet_capacity,
    CAPACITY_GROW_FACTOR,
    MIN_BLOCKS,
)
from .particle_solver import ParticleSolver

if TYPE_CHECKING:
    from genesis.engine.scene import Scene
    from genesis.engine.simulator import Simulator

# Number of substeps between two reads of whether a Newton solve stopped before converging since the previous read,
# each read waiting for the device
NEWTON_FAILURE_CHECK_PERIOD = 32

# Rows of the dense matrix of a direct adjoint solve at most, 3.2 GB in fp64; the option documents the value
# (see MPMOptions.adjoint_linear_solver)
ADJOINT_DIRECT_MAX_ROWS = 20000

# Run the substep as the single checkpointed graph `kernel_substep`, or as separate launches driven from the host with
# the PCG iterations and the radix sorts as small graphs of their own.
# FIXME: The single graph takes several minutes to compile cold, against a fraction of that for the separate launches,
# whose kernels each compile once and are shared by every call site; the host-driven substep is the default meanwhile.
USE_SUBSTEP_GRAPH = False


def material_table_column(material):
    """The column of a material in the material table, its energy family and its return map (design §6.1)."""
    column = np.zeros(mpm_func.N_MATERIAL_PARAMS, dtype=gs.np_float)
    column[mpm_func.PARAM_MU] = material.mu
    column[mpm_func.PARAM_LAM] = material.lam
    if isinstance(material, MPM.Sand):
        column[mpm_func.PARAM_SAND_ALPHA] = material.alpha
        return column, mpm_func.FAMILY_EXTENDED_HENCKY, mpm_func.PLASTIC_SAND
    if isinstance(material, MPM.Liquid):
        return column, mpm_func.FAMILY_LIQUID, mpm_func.PLASTIC_LIQUID
    if isinstance(material, MPM.ElastoPlastic):
        column[mpm_func.PARAM_YIELD_LOWER] = material.yield_lower
        column[mpm_func.PARAM_YIELD_HIGHER] = material.yield_higher
        column[mpm_func.PARAM_YIELD_STRESS] = material.von_mises_yield_stress
        if isinstance(material, MPM.Snow):
            return column, mpm_func.FAMILY_FIXED_COROTATED, mpm_func.PLASTIC_SNOW
        if material.use_von_mises:
            return column, mpm_func.FAMILY_FIXED_COROTATED, mpm_func.PLASTIC_VON_MISES
        return column, mpm_func.FAMILY_FIXED_COROTATED, mpm_func.PLASTIC_CLAMP
    if isinstance(material, MPM.Muscle):
        column[mpm_func.PARAM_FIBRE] = material.stiffness
    if isinstance(material, MPM.Elastic) and material.model == "neohooken":
        return column, mpm_func.FAMILY_STABLE_NEOHOOKEAN, mpm_func.PLASTIC_NONE
    return column, mpm_func.FAMILY_FIXED_COROTATED, mpm_func.PLASTIC_NONE


@qd.data_oriented
class MpmNewtonParticipant:
    """The grid displacement as the primary participant of the NewtonSolver: it owns the BCOO matrix and the PCG
    workspace `ws` whose scalars every participant reduces into, with the brick preconditioner over its rows."""

    def __init__(self, particles, particles_info, materials, grid, newton, linear, params):
        self.particles = particles
        self.particles_info = particles_info
        self.materials = materials
        self.grid = grid
        self.newton = newton
        self.linear = linear
        self.params = params
        self.ws = linear.pcg

    @qd.func(requires_top_level=True)
    def func_assemble(self, control: qd.template(), f):
        """The Newton system at the current iterate, inside a graph that needs no storage growth (see
        NewtonSolver.kernel_newton_loop)."""
        func_reset_newton_measures(control)
        mpm_func.func_report_extent(self.grid, self.linear)
        mpm_func.func_check_loop_capacity(self.grid, self.linear, self.params)
        mpm_func.func_assemble(
            self.particles,
            self.particles_info,
            self.materials,
            self.grid,
            self.newton,
            self.linear,
            self.params,
            f,
            self.params.substep_dt[()],
        )
        func_sort_reduce_bcoo(self.linear.matrix)

    @qd.func(requires_top_level=True)
    def func_pcg_prepare(self):
        mpm_func.func_reset_brick(self.grid, self.linear)
        mpm_func.func_gather_brick(self.linear)
        mpm_func.func_invert_brick(self.grid, self.linear)

    @qd.func(requires_top_level=True)
    def func_pcg_init_residual(self):
        func_init_linear_pcg_residual(self.ws, self.linear.b_rhs, self.linear.solution)

    @qd.func(requires_top_level=True)
    def func_precondition(self):
        mpm_func.func_apply_brick(self.grid, self.linear)

    @qd.func(requires_top_level=True)
    def func_pcg_init_direction(self):
        func_init_linear_pcg_direction(self.ws)

    @qd.func(requires_top_level=True)
    def func_pcg_spmv(self):
        func_linear_pcg_spmv(self.ws, self.linear.matrix)

    @qd.func(requires_top_level=True)
    def func_pcg_update_xr(self):
        func_linear_pcg_update_xr(self.ws, self.linear.solution)

    @qd.func(requires_top_level=True)
    def func_pcg_direction_update(self):
        func_linear_pcg_direction_update(self.ws)

    @qd.func(requires_top_level=True)
    def func_start_line_search(self, control: qd.template(), f):
        mpm_func.func_negate_du(self.grid, self.newton, self.linear)
        mpm_func.func_record_start_point(self.grid, self.newton)
        mpm_func.func_contribute_newton_max_disp(self.particles, self.grid, self.newton, control, self.params, f)

    @qd.func(requires_top_level=True)
    def func_step_forward(self, control: qd.template()):
        mpm_func.func_step_forward(self.grid, self.newton, control)

    @qd.func(requires_top_level=True)
    def func_energy(self, control: qd.template(), f, is_trial):
        mpm_func.func_energy(
            self.particles,
            self.particles_info,
            self.materials,
            self.grid,
            self.newton,
            control,
            self.params,
            f,
            self.params.substep_dt[()],
            is_trial,
        )


@qd.data_oriented
class MPMSolver(ParticleSolver):
    """Implicit material point method.

    Every substep solves for the grid node displacement that minimizes the incremental potential of the grid inertia
    and the particle elastic energy, on a block-sparse grid rebuilt from the particles, by a Newton method whose linear
    systems a preconditioned conjugate gradient solves. The substep runs as separate launches driven from the host, or
    as one graph (see `USE_SUBSTEP_GRAPH`); this class owns its data and composes the stages of `mpm_func`.
    """

    material_cls = MPM.Base
    has_frames = True
    # The configurations without an adjoint raise at build (see '_check_supported')
    is_differentiable = True

    # ------------------------------------------------------------------------------------
    # --------------------------------- Initialization -----------------------------------
    # ------------------------------------------------------------------------------------

    def __init__(self, scene: "Scene", sim: "Simulator", options: "MPMOptions"):
        super().__init__(scene, sim, options)

        # options
        self._grid_density = options.grid_density
        self._particle_size = options.particle_size
        self._upper_bound = np.array(options.upper_bound)
        self._lower_bound = np.array(options.lower_bound)
        self._enable_CPIC = options.enable_CPIC
        self._constraints_initialized = False
        self._has_warned_adjoint_projection = False

        self._n_vvert_supports = self.scene.options.vis.n_support_neighbors

        # `_particle_volume_scale` is used to avoid potential numerical instability, as the actual `_particle_volume` may be very small.
        # Particle masses are stored scaled by it; the implicit step divides it back out to work in kilograms.
        self._particle_volume_real = float(self._particle_size**3)
        self._particle_volume_scale = 1e3
        self._particle_volume = self._particle_volume_real * self._particle_volume_scale

        # Other derived parameters
        self._dx = float(1.0 / self._grid_density)
        self._inv_dx = float(self._grid_density)
        self._lower_bound_cell = np.round(self._grid_density * self._lower_bound).astype(gs.np_int)
        self._upper_bound_cell = np.round(self._grid_density * self._upper_bound).astype(gs.np_int)
        self._grid_res = self._upper_bound_cell - self._lower_bound_cell + 1  # +1 to include both corner
        self._grid_offset = qd.Vector(self._lower_bound_cell)

        # materials
        self._materials = list()
        self._materials_idx = list()

        # boundary
        self.setup_boundary()

    def setup_boundary(self):
        # safety padding
        self.boundary_padding = 3 * self._dx
        self.boundary = CubeBoundary(
            lower=self._lower_bound + self.boundary_padding, upper=self._upper_bound - self.boundary_padding
        )

    def init_particle_fields(self):
        # Particle state of the frames, its gradient-free part and the static info, as dynamic tensors in the default
        # array mode, so that a scene with another particle count reuses the compiled kernels
        self.particles = get_mpm_particles(self.n_frames, self._n_particles, self._B, self._requires_grad)
        self.particles_ng = get_mpm_particles_ng(self.n_frames, self._n_particles, self._B)
        self.particles_info = get_mpm_particles_info(self._n_particles)

        # The errors of each env, raised at the beginning of the next step (see 'Solver.register_errors')
        self.errno = qd.field(gs.qd_int, shape=(self._B,))
        self.register_errors(
            self.errno,
            {
                array_class.ErrorCode.INVALID_PARTICLE_NAN: (
                    "NaN detected in MPM states. Try reducing the time step size or adjusting simulation parameters."
                ),
                array_class.ErrorCode.INVALID_MPM_SOLVE_NAN: (
                    "The implicit MPM Newton solve produced non-finite values. Try reducing the time step size or "
                    "adjusting simulation parameters."
                ),
                array_class.ErrorCode.OVERFLOW_MPM_GRID_RANGE: (
                    "MPM particles have non-finite positions or left the range the sparse grid can index."
                ),
                array_class.ErrorCode.INVALID_MPM_GRID: (
                    "Internal error of the implicit MPM solver: its sparse grid or matrix storage is inconsistent. "
                    "Rerun with `gs.init(debug=True)` to stop at the faulty kernel."
                ),
            },
        )

        # single frame particle state for rendering
        struct_particle_state_render = qd.types.struct(pos=gs.qd_vec3, vel=gs.qd_vec3, active=gs.qd_bool)
        self.particles_render = struct_particle_state_render.field(
            shape=(self._n_particles, self._B), needs_grad=False, layout=qd.Layout.SOA
        )

    def init_implicit_data(self):
        """Allocate the data of the implicit step at the smallest capacity; the first substep sizes it to the
        particles, and later substeps grow it when the grid or the matrix overflows."""
        n_sort = self._B * self._n_particles
        max_blocks = MIN_BLOCKS
        max_triplets = triplet_capacity(max_blocks, MIN_BLOCKS)
        self._mpm_sized = False
        materials = {material.idx: material for material in self._materials}
        n_materials = len(materials)
        table = np.zeros((mpm_func.N_MATERIAL_PARAMS, n_materials), dtype=gs.np_float)
        family = np.zeros(n_materials, dtype=gs.np_int)
        plastic = np.zeros(n_materials, dtype=gs.np_int)
        for idx, material in materials.items():
            table[:, idx], family[idx], plastic[idx] = material_table_column(material)

        rigid = self._sim.rigid_solver
        # A Rigid solver without any coupled geom gives no contact rows
        if self._has_rigid and not rigid.dyn_info.geoms.needs_coup.to_numpy().any():
            self._has_rigid = False
        self._mpm_config = MpmStaticConfig(
            has_rigid=self._has_rigid,
            rigid_batch_links_info=rigid._options.batch_links_info,
            has_rigid_rows=self._has_rigid and not rigid._disable_constraint,
            has_terrain=self._has_rigid and rigid.collider.collider_config.has_terrain,
        )
        # Rigid geoms give contact rows whenever Rigid is coupled, but its dofs join the Newton solve only if it has
        # any: rows on static geoms (a plane, a terrain) act on the particle displacement alone, as wall rows do
        self._rigid_coupled = self._has_rigid and rigid.n_dofs > 0
        self._mpm_coupling = MpmRigidCoupling(rigid.n_dofs_ if self._rigid_coupled else 1, rigid.n_links, self._B)
        # Rigid stops its substep after assembling its constraint rows; the MPM substep solves them with its own
        # unknowns and finishes Rigid's substep
        rigid._defer_constraint_solve = self._rigid_coupled
        self._mpm_params = MpmParams(
            n_envs=self._B,
            n_particles=self._n_particles,
            substep_dt=self.substep_dt,
            dx=self._dx,
            particle_volume=self._particle_volume_real,
            mass_scale=self._particle_volume_scale,
            n_materials=n_materials,
            wall_sol_params=self._wall_sol_params(),
            wall_lower=self.boundary.lower,
            wall_upper=self.boundary.upper,
        )
        self._mpm_materials = get_mpm_material_table(table.reshape(-1), family, plastic)
        self._mpm_grid = MpmGrid(n_sort, max_blocks, max_blocks, self.errno)
        self._mpm_newton = MpmNewton(n_sort, max_blocks * BLOCK_CELLS)
        self._rigid_query = RigidGeomQuery(rigid, self._mpm_newton, self._B, self._has_rigid)
        self._mpm_linear = MpmLinearSystem(max_blocks * BLOCK_CELLS, max_triplets)
        newton_solver = self._sim.newton_solver
        self._mpm_control = newton_solver.control
        self._mpm_control.disp_tol.from_numpy(
            np.array(self._options.newton_velocity_tol * self.substep_dt, dtype=gs.np_float)
        )
        self._mpm_force_fields = MpmForceFields(self._ffs)

        # The grid displacement is the primary unknown of the substep's Newton solve; Rigid's dofs and the contact rows
        # on them join it when coupled
        self._newton_participant = MpmNewtonParticipant(
            self.particles,
            self.particles_info,
            self._mpm_materials,
            self._mpm_grid,
            self._mpm_newton,
            self._mpm_linear,
            self._mpm_params,
        )
        newton_solver.register_primary(self, self._newton_participant)
        if self._has_rigid:
            if self._requires_grad and self._mpm_config.has_terrain:
                gs.raise_exception(
                    "Differentiable simulation with the implicit MPM solver does not support the contact with a Rigid "
                    "terrain yet."
                )
            rigid_participant = RigidNewtonParticipant(rigid, self._mpm_coupling, self._mpm_config, self._mpm_params)
            # The pullback of the contact rows, which the contact coupling runs when Rigid's dofs join the solve
            self._rigid_contact = MpmRigidContact(self._newton_participant, rigid_participant)
        if self._rigid_coupled:
            newton_solver.register_secondary(RigidNewtonHost(rigid, rigid_participant), rigid_participant)
            newton_solver.register_coupling(MpmRigidContactHost(self._rigid_contact), self._rigid_contact)
        # Implicit FEM's substep joins the solve, its surface triangles giving the particles contact rows
        fem = self._sim.fem_solver
        fem._defer_solve = self._fem_coupled
        if self._fem_coupled:
            fem_participant = FemNewtonParticipant(fem)
            newton_solver.register_secondary(FemNewtonHost(fem, self._coupler), fem_participant)
            query = FemTriangleQuery(fem, self._mpm_newton, self._mpm_params, self._B)
            contact = MpmFemContact(self._newton_participant, fem_participant, query)
            newton_solver.register_coupling(MpmFemContactHost(contact), contact)
        # The MPM particles are boundary particles of the SPH fluid, whose impulses enter the next P2G (see
        # mpm_sph_func)
        if self._sph_coupled:
            sph = self._sim.sph_solver
            self._sph_boundary = MpmSphBoundary(self, sph, self._mpm_params, self._B)
            sph.set_boundary(self._sph_boundary)
        # PBD's particles join the solve with their inertia around PBD's own projection (see mpm_pbd_func)
        if self._pbd_coupled:
            pbd = self._sim.pbd_solver
            pbd_participant = PbdNewtonParticipant(PbdNewtonData(pbd, self._B))
            newton_solver.register_secondary(PbdNewtonHost(pbd_participant.pd, self._mpm_params), pbd_participant)
            query = PbdParticleQuery(pbd, self._mpm_newton, self._mpm_params, self._B)
            contact = MpmPbdContact(self._newton_participant, pbd_participant, query)
            newton_solver.register_coupling(MpmPbdContactHost(contact), contact)
            pbd.set_particle_obstacles(MpmParticleObstacles(self, pbd, self._mpm_params, self._B))

    def _wall_sol_params(self):
        """Rigid's default solref / solimp parameters, with the Rigid constraint timeconst floored at twice the
        substep as Rigid floors its contacts."""
        params = gu.default_solver_params()
        timeconst = self._sim.rigid_solver._options.constraint_timeconst
        params[0] = max(timeconst if timeconst is not None else 0.0, 2.0 * self.substep_dt)
        return tuple(float(p) for p in params)

    def init_vvert_fields(self):
        struct_vvert_info = qd.types.struct(
            support_idxs=qd.types.vector(self._n_vvert_supports, gs.qd_int),
            support_weights=qd.types.vector(self._n_vvert_supports, gs.qd_float),
        )
        self.vverts_info = struct_vvert_info.field(shape=(max(1, self._n_vverts),), layout=qd.Layout.SOA)

        struct_vvert_state_render = qd.types.struct(pos=gs.qd_vec3, active=gs.qd_bool)
        self.vverts_render = struct_vvert_state_render.field(
            shape=(max(1, self._n_vverts), self._B), layout=qd.Layout.SOA
        )

    def init_constraints(self):
        """Lazy initialization of particle constraint fields."""
        # Memory check: ensure index fits in int32
        if self._n_particles * self._B * 3 > np.iinfo(np.int32).max:
            gs.raise_exception(
                f"Particle constraint shape (n_envs={self._B}, n_particles={self._n_particles}, 3) is too large. "
                "Consider reducing n_envs or n_particles."
            )

        self._constraints_initialized = True

        particle_constraint_info = qd.types.struct(
            is_constrained=gs.qd_bool,  # whether particle is constrained
            target_pos=gs.qd_vec3,  # target position for the constraint
            stiffness=gs.qd_float,  # spring stiffness
            link_idx=gs.qd_int,  # index of the rigid link (-1 if not linked)
            link_local_pos=gs.qd_vec3,  # offset from link origin in link's local frame
        )

        # FIXME: AOS, which does not match other Genesis structs. Old, untested code. We prefer not to touch for now.
        self.particle_constraints = particle_constraint_info.field(
            shape=(self._n_particles, self._B), needs_grad=False, layout=qd.Layout.AOS
        )

        self.particle_constraints.is_constrained.fill(False)
        self.particle_constraints.link_idx.fill(-1)

    def reset_grad(self):
        if self._requires_grad and self.is_active:
            qd_zero_grad(self.particles)
            if gs.use_zerocopy.is_supported and gs.backend == gs.metal:
                torch.mps.synchronize()

    def _check_supported(self):
        """Raise for the configurations the implicit solver does not handle yet."""
        if self._enable_CPIC:
            gs.raise_exception("CPIC is not supported by the implicit MPM solver.")
        if self._B > mpm_func.MAX_KEY_ENVS:
            gs.raise_exception(f"The implicit MPM solver supports {mpm_func.MAX_KEY_ENVS} environments at most.")
        cells = np.concatenate([self._lower_bound, self._upper_bound]) * self._grid_density
        if cells.min() < mpm_func.CELL_MIN or cells.max() >= mpm_func.CELL_END:
            gs.raise_exception(
                f"The MPM domain spans the cells [{cells.min():.0f}, {cells.max():.0f}], past the range "
                f"[{mpm_func.CELL_MIN}, {mpm_func.CELL_END}) the sparse grid of the implicit solver indexes."
            )
        for material in self._materials:
            if not isinstance(material, (MPM.Elastic, MPM.ElastoPlastic, MPM.Liquid, MPM.Sand)):
                gs.raise_exception(
                    f"The implicit MPM solver has no energy for the material {type(material).__name__}; it supports "
                    "the materials of genesis.materials.MPM."
                )
        # Only the legacy coupler couples MPM with other solvers, each pair behind its own option
        coupler_options = self._sim.coupler.options
        is_legacy = isinstance(coupler_options, LegacyCouplerOptions)
        self._has_rigid = self._sim.rigid_solver.is_active and is_legacy and coupler_options.rigid_mpm
        if self._has_rigid:
            rigid_options = self._sim.rigid_solver._options
            if rigid_options.friction_cone == gs.friction_cone.elliptic:
                gs.raise_exception(
                    "Coupling the implicit MPM solver with an elliptic Rigid friction cone is not supported."
                )
            # The joint solve uses Rigid's Newton Hessian factor as its preconditioner and integrates every Rigid dof
            if rigid_options.constraint_solver != gs.constraint_solver.Newton:
                gs.raise_exception("Coupling the implicit MPM solver with Rigid requires Rigid's Newton solver.")
            if self._sim.rigid_solver._use_hibernation or rigid_options.noslip_iterations > 0:
                gs.raise_exception(
                    "Coupling the implicit MPM solver with Rigid does not support hibernation or noslip iterations."
                )
        # The couplings with the particle and FEM solvers
        self._sph_coupled = self._sim.sph_solver.is_active and is_legacy and coupler_options.mpm_sph
        self._pbd_coupled = self._sim.pbd_solver.is_active and is_legacy and coupler_options.mpm_pbd
        self._fem_coupled = self._sim.fem_solver.is_active and is_legacy and coupler_options.fem_mpm
        if self._fem_coupled and not self._sim.fem_solver._use_implicit_solver:
            gs.raise_exception(
                "Coupling the implicit MPM solver with FEM requires FEM's implicit solver "
                "(FEMOptions(use_implicit_solver=True)), or LegacyCouplerOptions(fem_mpm=False)."
            )
        if self._requires_grad:
            for material in self._materials:
                if isinstance(material, (MPM.ElastoPlastic, MPM.Liquid, MPM.Sand)):
                    gs.raise_exception(
                        "Differentiable simulation with the implicit MPM solver does not support the plasticity of "
                        f"the material {type(material).__name__} yet."
                    )
            for force_field in self._ffs:
                if not force_field.differentiable:
                    gs.raise_exception(
                        f"Differentiable simulation does not support the force field {type(force_field).__name__}."
                    )

    def build(self):
        super().build()

        # particles and entities
        self._B = self._sim._B
        self._n_particles = self.n_particles
        self._n_vverts = self.n_vverts
        self._n_vfaces = self.n_vfaces

        self._coupler = self.sim._coupler

        if self.is_active:
            self._check_supported()

            self.init_particle_fields()
            self.init_vvert_fields()

            for entity in self._entities:
                entity._add_to_solver()

            self.init_implicit_data()

        self._build_gravity()
        if self.is_active:
            self._init_contact_invweight()

    def _init_contact_invweight(self):
        """The inverse weight of every particle in its contact rows (see func_contact_invweight), from the physical
        Hessian of the configuration the particles are added in, before any step.

        Like the inverse weights Rigid computes from the mass matrix of its initial configuration, it is a model
        constant: a row's stiffness does not depend on the state, so that its gradients involve the second derivatives
        of the energy alone. A particle inactive in that configuration keeps the inverse of its own mass."""
        f = 0
        grid, params = self._mpm_grid, self._mpm_params
        self._size_to_particles(f)
        build_grid(self.particles, self.particles_ng, grid, params, f)
        while int(grid.grid_overflow.to_numpy()):
            self._grow_grid()
            build_grid(self.particles, self.particles_ng, grid, params, f)
        kernel_predict(
            self.particles,
            self.particles_info,
            grid,
            self._mpm_newton,
            self._mpm_force_fields,
            params,
            self._gravity,
            f,
            0.0,
        )
        self._assemble(f)
        kernel_contact_invweight(self.particles, self.particles_info, grid, self._mpm_linear, params, f)

    # ------------------------------------------------------------------------------------
    # -------------------------------------- misc ----------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def is_active(self):
        return self.n_particles > 0

    def add_entity(
        self, idx, material, morph, surface, visualize_contact=False, name: str | None = None, desc=None
    ) -> "MPMEntity":
        self.add_material(material)

        # create entity
        entity = MPMEntity(
            scene=self._scene,
            solver=self,
            material=material,
            morph=morph,
            surface=surface,
            particle_size=self._particle_size,
            idx=idx,
            particle_start=self.n_particles,
            vvert_start=self.n_vverts,
            vface_start=self.n_vfaces,
            name=name,
        )
        self._entities.append(entity)

        return entity

    def add_material(self, material):
        # Register the material if and only if it is not already registered
        for material_i in self._materials:
            if material == material_i:
                material.idx = material_i.idx
                break
        else:
            material.idx = len(self._materials_idx)
            self._materials_idx.append(material.idx)
        self._materials.append(material)

    # ------------------------------------------------------------------------------------
    # ----------------------------------- simulation -------------------------------------
    # ------------------------------------------------------------------------------------

    @no_grad
    @qd.kernel
    def _kernel_flag_non_finite_pos(self, f: int):
        """Flag INVALID_PARTICLE_NAN for every env with a particle whose position in the given frame is not finite."""
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if not qd_isfinite(self.particles.pos[f, i_p, i_b]):
                self.errno[i_b] = self.errno[i_b] | array_class.ErrorCode.INVALID_PARTICLE_NAN

    @no_grad
    @qd.kernel
    def _kernel_flag_newton_non_finite(self, control: qd.template()):
        """Flag INVALID_MPM_SOLVE_NAN for every env, which share the solve, once a Newton solve since the last call
        stopped on a non-finite measure (see NewtonControl), and clear the latch."""
        for _ in range(1):
            if control.failed_non_finite[()] != 0:
                for i_b in range(self._B):
                    self.errno[i_b] = self.errno[i_b] | array_class.ErrorCode.INVALID_MPM_SOLVE_NAN
                control.failed_non_finite[()] = 0

    # ------------------------------------------------------------------------------------
    # ------------------------------------ stepping --------------------------------------
    # ------------------------------------------------------------------------------------

    def substep_pre_coupling(self, f):
        pass

    def _resize_blocks(self, max_blocks):
        """Reallocate the grid, the Newton state and the matrix rows for max_blocks node blocks. The grid carries no
        state across substeps and the growth happens before any consumer reads it, so nothing is copied."""
        n_sort = self._B * self._n_particles
        self._mpm_grid.allocate(n_sort, max_blocks, max_blocks)
        self._mpm_newton.allocate(n_sort, max_blocks * BLOCK_CELLS)
        self._mpm_linear.allocate_rows(max_blocks * BLOCK_CELLS)

    def _size_to_particles(self, f):
        positions = self.particles.pos.to_numpy()[f].transpose(1, 0, 2)
        max_blocks, max_triplets = initial_capacities(positions, self._inv_dx)
        self._resize_blocks(max_blocks)
        self._mpm_linear.matrix.grow(max_triplets)
        self._mpm_sized = True

    def _grow_grid(self):
        grid = self._mpm_grid
        demand = max(int(grid.grid_demand.to_numpy()), int(grid.n_particle_blocks.to_numpy()))
        self._resize_blocks(max(int(np.ceil(demand * CAPACITY_GROW_FACTOR)), grid.max_blocks + 1))
        grid.grid_overflow.from_numpy(np.array(0, dtype=np.int32))

    def _grow_triplets(self):
        matrix = self._mpm_linear.matrix
        demand = int(matrix.triplet_demand.to_numpy())
        matrix.grow(max(int(np.ceil(demand * CAPACITY_GROW_FACTOR)), int(matrix.max_triplets.to_numpy()) + 1))
        matrix.triplet_overflow.from_numpy(np.array(0, dtype=np.int32))

    def substep_post_coupling(self, f):
        """The substep's G2P happened in the Newton solver's substep, which ran before this; under the single substep
        graph, this runs the whole substep instead."""
        if USE_SUBSTEP_GRAPH:
            if self._constraints_initialized:
                gs.raise_exception("The single substep graph of the implicit MPM solver has no particle constraints.")
            if not self._mpm_sized:
                self._size_to_particles(f)
            self._substep_graph(f)

        # The positions a step ends with are checked once, an error being raised at the beginning of the next step
        if (self._sim._cur_substep_global + 1) % self._substeps == 0:
            self._kernel_flag_non_finite_pos(f + 1)
            self._kernel_flag_newton_non_finite(self._mpm_control)
        if self._sim._cur_substep_global % NEWTON_FAILURE_CHECK_PERIOD == 0:
            control = self._mpm_control
            if int(control.failed_iterations.to_numpy()):
                control.failed_iterations.from_numpy(np.array(0, dtype=np.int32))
                gs.logger.warning("The implicit MPM Newton solve stopped before converging.")

    def _assemble(self, f, is_exact=False):
        """Assemble the Newton system, growing the triplet storage until it holds the assembly. The elastic Hessian is
        SPD-projected, or exact with `is_exact` (see func_elastic_assemble)."""
        args = (
            self.particles,
            self.particles_info,
            self._mpm_materials,
            self._mpm_grid,
            self._mpm_newton,
            self._mpm_linear,
            self._mpm_control,
            self._mpm_params,
            f,
            is_exact,
        )
        kernel_assemble(*args)
        while int(self._mpm_linear.matrix.triplet_overflow.to_numpy()):
            self._grow_triplets()
            kernel_assemble(*args)
        sort_reduce_bcoo(self._mpm_linear.matrix)

    # ------------------------------------------------------------------------------------
    # --------------------------- Newton solver participant -----------------------------
    # ------------------------------------------------------------------------------------

    def runs_own_substep(self):
        """Whether this solver runs its whole substep itself (see USE_SUBSTEP_GRAPH), the Newton solver skipping it."""
        return USE_SUBSTEP_GRAPH

    def newton_prepare(self, f):
        """The substep up to the Newton solve: the impulses of the SPH fluid, the grid, P2G with the force fields, the
        physical Hessian at u = 0 and the contact rows, which arm the Newton loop."""
        if not self._mpm_sized:
            self._size_to_particles(f)
        if self._sph_coupled:
            kernel_apply_boundary_impulse(self._sph_boundary, f)
        rigid = self._sim.rigid_solver
        grid, params = self._mpm_grid, self._mpm_params
        build_grid(self.particles, self.particles_ng, grid, params, f)
        while int(grid.grid_overflow.to_numpy()):
            self._grow_grid()
            build_grid(self.particles, self.particles_ng, grid, params, f)
        if self._constraints_initialized:
            kernel_constraint_rows(
                self.particles_info, grid, self._mpm_newton, params, self.particle_constraints, rigid.dyn_state.links
            )
        kernel_predict(
            self.particles,
            self.particles_info,
            grid,
            self._mpm_newton,
            self._mpm_force_fields,
            params,
            self._gravity,
            f,
            self._sim.cur_t,
        )
        # The physical Hessian at u = 0 sizes the triplet storage for the substep's fixed grid topology, which every
        # assembly of the Newton loop shares
        self._assemble(f)
        self._rigid_query.build(params)
        kernel_contact_rows(
            self.particles,
            self.particles_info,
            grid,
            self._mpm_newton,
            self._mpm_control,
            self._mpm_config,
            params,
            rigid.dyn_state.geoms,
            rigid.dyn_info.geoms,
            rigid.dyn_state.links,
            rigid.dyn_info.links,
            rigid.rigid_info,
            rigid.collider.sdf.sdf_info,
            self._rigid_query,
            f,
        )

    def newton_assemble(self, f):
        self._assemble(f)

    def newton_finish(self, f):
        """G2P from the solved grid displacement."""
        kernel_finalize(
            self.particles,
            self.particles_ng,
            self.particles_info,
            self._mpm_materials,
            self._mpm_grid,
            self._mpm_newton,
            self._mpm_params,
            f,
        )

    def newton_after(self, f):
        pass

    def newton_end(self, f):
        pass

    def newton_backward(self, f):
        """Reverse substep f from the state the replay of its forward pass left (see mpm_adjoint_func): the reverse of
        G2P into the adjoint of u*, the adjoint solve at the converged u* on the Hessian and by the solver the options
        select, then the pullback and the reverse of P2G into the adjoint of frame f."""
        grid, newton, linear = self._mpm_grid, self._mpm_newton, self._mpm_linear
        mode = self._options.adjoint_hessian
        is_exact = mode in ("exact", "auto")
        newton_solver = self._sim.newton_solver
        kernel_adjoint_g2p(self.particles, grid, newton, self._mpm_params, f)
        self._assemble(f, is_exact=is_exact)
        kernel_adjoint_rhs(grid, newton, linear)
        newton_solver.stage_adjoint(f)
        if self._options.adjoint_linear_solver == "direct":
            if newton_solver.has_secondaries:
                gs.raise_exception(
                    "The direct adjoint solve of the implicit MPM solver does not support the joint solve with other "
                    "solvers; use MPMOptions(adjoint_linear_solver='pcg')."
                )
            self._solve_adjoint_direct()
        elif not newton_solver.solve_adjoint(f, check_curvature=is_exact):
            # At a converged solution the Hessian of the incremental potential is positive definite
            if mode == "exact":
                gs.raise_exception(
                    "The exact Hessian of an implicit MPM substep is not positive definite at its solution, which the "
                    "forward Newton solve most likely stopped short of: tighten MPMOptions(newton_velocity_tol), or "
                    "use MPMOptions(adjoint_hessian='auto') or 'projected'."
                )
            if not self._has_warned_adjoint_projection:
                gs.logger.warning(
                    "The exact Hessian of an implicit MPM substep is not positive definite at its solution, which the "
                    "forward Newton solve most likely stopped short of: its adjoint solves on the projected Hessian, "
                    "whose gradients are approximate. Tighten MPMOptions(newton_velocity_tol) for exact ones."
                )
                self._has_warned_adjoint_projection = True
            self._assemble(f)
            kernel_adjoint_rhs(grid, newton, linear)
            newton_solver.stage_adjoint(f)
            newton_solver.solve_adjoint(f)
        newton_solver.finish_adjoint(f)
        # Rows on static Rigid geoms act on the particle displacement alone, outside of any coupling
        if self._has_rigid and not self._rigid_coupled:
            kernel_rigid_contact_adjoint(self._rigid_contact, f)
        if self._constraints_initialized:
            kernel_constraint_adjoint(
                self.particles,
                grid,
                newton,
                linear,
                self._mpm_params,
                self.particle_constraints,
                self._sim.rigid_solver.dyn_state.links,
                f,
            )
        kernel_adjoint_pullback(
            self.particles,
            self.particles_ng,
            self.particles_info,
            self._mpm_materials,
            self._mpm_grid,
            self._mpm_newton,
            self._mpm_linear,
            self._mpm_force_fields,
            self._mpm_params,
            self._gravity,
            f,
            self._sim.cur_t,
        )

    def _solve_adjoint_direct(self):
        """Solve the adjoint system the primary staged (see `NewtonSolver.solve_adjoint`) by a dense LU factorization
        of the matrix on the device, the symmetric matrix rebuilt from its stored diagonal and upper blocks."""
        linear = self._mpm_linear
        matrix = linear.matrix
        n_rows = 3 * int(linear.pcg.n_rows.to_numpy())
        if n_rows > ADJOINT_DIRECT_MAX_ROWS:
            gs.raise_exception(
                f"The direct adjoint solve of the implicit MPM solver, meant for gradient tests on small scenes, holds "
                f"{ADJOINT_DIRECT_MAX_ROWS} rows at most, got {n_rows}: shrink the scene or its environments, or use "
                "MPMOptions(adjoint_linear_solver='pcg')."
            )
        nnz = int(matrix.bcoo_nnz.to_numpy())
        rows = qd_to_torch(matrix.bcoo_row, copy=True)[:nnz].long()
        cols = qd_to_torch(matrix.bcoo_col, copy=True)[:nnz].long()
        blocks = qd_to_torch(matrix.bcoo_val, copy=True)[: 9 * nnz].reshape(nnz, 3, 3)
        offsets = torch.arange(3, device=gs.device)
        block_rows = (3 * rows[:, None, None] + offsets[None, :, None]).expand(nnz, 3, 3)
        block_cols = (3 * cols[:, None, None] + offsets[None, None, :]).expand(nnz, 3, 3)
        H = torch.zeros((n_rows, n_rows), dtype=gs.tc_float, device=gs.device)
        H.index_put_((block_rows.reshape(-1), block_cols.reshape(-1)), blocks.reshape(-1), accumulate=True)
        is_off = rows != cols
        H.index_put_(
            (block_cols[is_off].reshape(-1), block_rows[is_off].reshape(-1)),
            blocks[is_off].reshape(-1),
            accumulate=True,
        )
        rhs = qd_to_torch(linear.b_rhs, copy=True)
        solution = qd_to_torch(linear.solution, copy=True)
        solution[:n_rows] = torch.linalg.solve(H, rhs[:n_rows])
        qd_from_torch(linear.solution, solution)

    def _substep_graph(self, f):
        if self._rigid_coupled:
            gs.raise_exception(
                "The implicit MPM substep graph does not support the coupling with Rigid dofs; use USE_NEWTON_GRAPH."
            )
        if self._sph_coupled or self._pbd_coupled or self._fem_coupled:
            gs.raise_exception(
                "The implicit MPM substep graph does not support the coupling with SPH, PBD or FEM; use "
                "USE_NEWTON_GRAPH."
            )
        rigid = self._sim.rigid_solver
        args = (
            self.particles,
            self.particles_ng,
            self.particles_info,
            self._mpm_materials,
            self._mpm_grid,
            self._mpm_newton,
            self._mpm_linear,
            self._mpm_control,
            self._mpm_force_fields,
            self._mpm_config,
            self._mpm_params,
            self._gravity,
            rigid.dyn_state.geoms,
            rigid.dyn_info.geoms,
            rigid.dyn_state.links,
            rigid.dyn_info.links,
            rigid.rigid_info,
            rigid.collider.sdf.sdf_info,
            self._rigid_query,
            f,
            self._sim.cur_t,
        )
        # Rigid holds still over the MPM substep, so its trees are built ahead of the graph
        self._rigid_query.build(self._mpm_params)
        status = kernel_substep(*args)
        while status.yielded:
            if status.checkpoint == MpmCheckpoint.GRID:
                self._grow_grid()
            elif status.checkpoint == MpmCheckpoint.ASSEMBLE:
                self._grow_triplets()
            else:
                gs.raise_exception(f"The implicit MPM substep yielded at an unexpected checkpoint: {status.checkpoint}")
            status = kernel_substep.resume(*args, from_checkpoint=status.checkpoint)

    @qd.kernel
    def copy_frame(self, source: qd.i32, target: qd.i32):
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            self.particles.pos[target, i_p, i_b] = self.particles.pos[source, i_p, i_b]
            self.particles.vel[target, i_p, i_b] = self.particles.vel[source, i_p, i_b]
            self.particles.F[target, i_p, i_b] = self.particles.F[source, i_p, i_b]
            self.particles.C[target, i_p, i_b] = self.particles.C[source, i_p, i_b]
            self.particles.Jp[target, i_p, i_b] = self.particles.Jp[source, i_p, i_b]
            self.particles.actu[target, i_p, i_b] = self.particles.actu[source, i_p, i_b]

            self.particles_ng.active[target, i_p, i_b] = self.particles_ng.active[source, i_p, i_b]

    # ------------------------------------------------------------------------------------
    # --------------------------------------- io -----------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def data(self):
        """Yield the state of the particles, the frame 0 of which a boundary checkpoint holds (see 'Solver.data')."""
        for name, field in (
            ("particles.pos", self.particles.pos),
            ("particles.vel", self.particles.vel),
            ("particles.C", self.particles.C),
            ("particles.F", self.particles.F),
            ("particles.Jp", self.particles.Jp),
            ("particles.actu", self.particles.actu),
            ("particles_ng.active", self.particles_ng.active),
        ):
            yield array_class.ArrayItem(name, array_class.DataKind.STATE, field)

    @mutates()
    def set_state(self, f, state, envs_idx=None):
        if self.is_active:
            self._kernel_set_state(f, state.pos, state.vel, state.C, state.F, state.Jp, state.actu, state.active)
            self.clear_errno(envs_idx)

    @qd.kernel
    def _kernel_set_state(
        self,
        f: qd.i32,
        pos: qd.types.ndarray(),  # shape [B, n_particles, 3]
        vel: qd.types.ndarray(),  # shape [B, n_particles, 3]
        C: qd.types.ndarray(),  # shape [B, n_particles, 3, 3]
        F: qd.types.ndarray(),  # shape [B, n_particles, 3, 3]
        Jp: qd.types.ndarray(),  # shape [B, n_particles]
        actu: qd.types.ndarray(),  # shape [B, n_particles]
        active: qd.types.ndarray(),  # shape [B, n_particles]
    ):
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            for j in qd.static(range(3)):
                self.particles.pos[f, i_p, i_b][j] = pos[i_b, i_p, j]
                self.particles.vel[f, i_p, i_b][j] = vel[i_b, i_p, j]
                for k in qd.static(range(3)):
                    self.particles.C[f, i_p, i_b][j, k] = C[i_b, i_p, j, k]
                    self.particles.F[f, i_p, i_b][j, k] = F[i_b, i_p, j, k]
            self.particles.Jp[f, i_p, i_b] = Jp[i_b, i_p]
            self.particles.actu[f, i_p, i_b] = actu[i_b, i_p]
            self.particles_ng.active[f, i_p, i_b] = active[i_b, i_p]

    def get_state(self, f):
        if not self.is_active:
            return None

        pos, vel, C, F, Jp = self._read_data(
            (self.particles.pos, self.particles.vel, self.particles.C, self.particles.F, self.particles.Jp),
            None,
            None,
            copy=True,
        )
        # The actuation and the activity are inputs of the step, which carry no gradient back
        actu, active = [
            qd_to_torch(field, row_mask=0, keepdim=False).movedim(1, 0).clone(memory_format=torch.contiguous_format)
            for field in (self.particles.actu, self.particles_ng.active)
        ]
        return MPMSolverState(self._scene, pos, vel, C, F, Jp, actu, active)

    def update_render_fields(self):
        self._kernel_update_render_fields(self.sim.cur_substep_local)

    @qd.kernel
    def _kernel_update_render_fields(self, f: qd.i32):
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles_ng.active[f, i_p, i_b]:
                self.particles_render[i_p, i_b].pos = self.particles.pos[f, i_p, i_b]
                self.particles_render[i_p, i_b].vel = self.particles.vel[f, i_p, i_b]
            else:
                self.particles_render[i_p, i_b].pos = gu.qd_nowhere()
            self.particles_render[i_p, i_b].active = self.particles_ng.active[f, i_p, i_b]

        for i_v, i_b in qd.ndrange(self._n_vverts, self._B):
            vvert_pos = qd.Vector.zero(gs.qd_float, 3)
            for j in range(self._n_vvert_supports):
                vvert_pos += (
                    self.particles.pos[f, self.vverts_info.support_idxs[i_v][j], i_b]
                    * self.vverts_info.support_weights[i_v][j]
                )
            self.vverts_render[i_v, i_b].pos = vvert_pos
            self.vverts_render[i_v, i_b].active = self.particles_render[
                self.vverts_info.support_idxs[i_v][0], i_b
            ].active

    @qd.kernel
    def add_particles(
        self,
        f: qd.i32,
        active: qd.i32,
        particle_start: qd.i32,
        n_particles: qd.i32,
        material_idx: qd.i32,
        mat_default_Jp: qd.f32,
        mat_rho: qd.f32,
        pos: qd.types.ndarray(),  # shape [n_particles, 3]
    ):
        for i_p_ in range(n_particles):
            i_p = i_p_ + particle_start

            self.particles_info.material_idx[i_p] = material_idx
            self.particles_info.default_Jp[i_p] = mat_default_Jp
            self.particles_info.mass[i_p] = self._particle_volume * mat_rho
            self.particles_info.free[i_p] = True
            self.particles_info.muscle_group[i_p] = 0
            self.particles_info.muscle_direction[i_p] = qd.Vector([0.0, 0.0, 1.0], dt=gs.qd_float)
            # The inverse mass of the particle alone, for a particle the rest configuration does not hold
            self.particles_info.contact_invweight[i_p] = qd.Matrix.identity(gs.qd_float, 3) / (
                self._particle_volume_real * mat_rho
            )

        for i_p_, i_b in qd.ndrange(n_particles, self._B):
            i_p = i_p_ + particle_start

            self.particles_ng.active[f, i_p, i_b] = qd.cast(active, gs.qd_bool)
            for i in qd.static(range(3)):
                self.particles.pos[f, i_p, i_b][i] = pos[i_p_, i]

            self.particles.vel[f, i_p, i_b] = qd.Vector.zero(gs.qd_float, 3)
            self.particles.F[f, i_p, i_b] = qd.Matrix.identity(gs.qd_float, 3)
            self.particles.C[f, i_p, i_b] = qd.Matrix.zero(gs.qd_float, 3, 3)
            self.particles.Jp[f, i_p, i_b] = mat_default_Jp
            self.particles.actu[f, i_p, i_b] = gs.qd_float(0.0)

    @gs.assert_built
    @mutates()
    def set_particles_actu(self, actus, particles_idx=None, envs_idx=None, *, n_groups=1):
        """
        Set the actuation of some particles, each particle taking the actuation of its muscle group.

        Parameters
        ----------
        actus : array_like, shape ([M,] [N,] [n_groups,])
            Actuation of each muscle group, for each particle.
        particles_idx : None | int | array_like, shape ([M,] N), optional
            Global index of the particles (see 'set_particles_pos'). If None, all particles will be considered.
        envs_idx : None | int | array_like, shape (M,), optional
            The indices of the environments to set. If None, all environments will be considered. Defaults to None.
        n_groups : int, optional
            The number of muscle groups. Defaults to 1.
        """
        actus, particles_idx, envs_idx = self._sanitize_io_variables(
            actus, particles_idx, self.n_particles, "particles_idx", envs_idx, (n_groups,), batched_idx=True
        )
        self._kernel_set_particles_actu(particles_idx, envs_idx, actus, n_groups)

    def get_particles_actu(self, particles_idx=None, envs_idx=None):
        """Get the actuation of some particles (see 'set_particles_actu')."""
        actus, particles_idx, envs_idx = self._sanitize_io_variables(
            None, particles_idx, self.n_particles, "particles_idx", envs_idx, batched_idx=True
        )
        self._kernel_get_particles_actu(particles_idx, envs_idx, actus)
        return actus[0] if self._scene.n_envs == 0 else actus

    @mutates(StateChange.MODEL)
    def set_particles_muscle_group(self, muscle_group, particles_idx=None):
        """Set the muscle group of some particles, shared by every environment."""
        muscle_group, particles_idx, _ = self._sanitize_io_variables(
            muscle_group, particles_idx, self.n_particles, "particles_idx", dtype=gs.tc_int, batched=False
        )
        self._kernel_set_particles_muscle_group(particles_idx, muscle_group)

    def get_particles_muscle_group(self, particles_idx=None):
        """Get the muscle group of some particles (see 'set_particles_muscle_group')."""
        muscle_group, particles_idx, _ = self._sanitize_io_variables(
            None, particles_idx, self.n_particles, "particles_idx", dtype=gs.tc_int, batched=False
        )
        self._kernel_get_particles_muscle_group(particles_idx, muscle_group)
        return muscle_group

    @mutates(StateChange.MODEL)
    def set_particles_muscle_direction(self, muscle_direction, particles_idx=None):
        """Set the unit direction of the muscle fibers of some particles, shared by every environment."""
        muscle_direction, particles_idx, _ = self._sanitize_io_variables(
            muscle_direction, particles_idx, self.n_particles, "particles_idx", element_shape=(3,), batched=False
        )
        self._kernel_set_particles_muscle_direction(particles_idx, muscle_direction)

    @mutates(StateChange.MODEL)
    def set_particles_free(self, free, particles_idx=None):
        """Set whether some particles move freely or stay fixed, shared by every environment."""
        free, particles_idx, _ = self._sanitize_io_variables(
            free, particles_idx, self.n_particles, "particles_idx", dtype=gs.tc_bool, batched=False
        )
        self._kernel_set_particles_free(particles_idx, free)

    def get_particles_free(self, particles_idx=None):
        """Get whether some particles move freely (see 'set_particles_free')."""
        free, particles_idx, _ = self._sanitize_io_variables(
            None, particles_idx, self.n_particles, "particles_idx", dtype=gs.tc_bool, batched=False
        )
        self._kernel_get_particles_free(particles_idx, free)
        return free

    @qd.kernel
    def _kernel_set_particles_pos(
        self, particles_idx: qd.types.ndarray(), envs_idx: qd.types.ndarray(), poss: qd.types.ndarray()
    ):
        for i_p_, i_b_ in qd.ndrange(particles_idx.shape[1], envs_idx.shape[0]):
            i_p = particles_idx[i_b_, i_p_]
            i_b = envs_idx[i_b_]
            # The earlier occurrences of a repeated particle are -1 (see 'drop_repeated_index'). Between two steps, the
            # state sits in frame 0 (see 'Solver.has_frames').
            if i_p >= 0:
                for i in qd.static(range(3)):
                    self.particles.pos[0, i_p, i_b][i] = poss[i_b_, i_p_, i]

                # Reset these attributes whenever overwriting particle positions manually
                self.particles.vel[0, i_p, i_b] = qd.Vector.zero(gs.qd_float, 3)
                self.particles.F[0, i_p, i_b] = qd.Matrix.identity(gs.qd_float, 3)
                self.particles.C[0, i_p, i_b] = qd.Matrix.zero(gs.qd_float, 3, 3)
                self.particles.Jp[0, i_p, i_b] = self.particles_info.default_Jp[i_p]

    @qd.kernel
    def _kernel_get_particles_pos(
        self, particles_idx: qd.types.ndarray(), envs_idx: qd.types.ndarray(), poss: qd.types.ndarray()
    ):
        for i_p_, i_b_ in qd.ndrange(particles_idx.shape[1], envs_idx.shape[0]):
            i_p = particles_idx[i_b_, i_p_]
            i_b = envs_idx[i_b_]
            for i in qd.static(range(3)):
                poss[i_b_, i_p_, i] = self.particles.pos[0, i_p, i_b][i]

    @qd.kernel
    def _kernel_set_particles_vel(
        self, particles_idx: qd.types.ndarray(), envs_idx: qd.types.ndarray(), vels: qd.types.ndarray()
    ):
        for i_p_, i_b_ in qd.ndrange(particles_idx.shape[1], envs_idx.shape[0]):
            i_p = particles_idx[i_b_, i_p_]
            i_b = envs_idx[i_b_]
            # See '_kernel_set_particles_pos'
            if i_p >= 0:
                for i in qd.static(range(3)):
                    self.particles.vel[0, i_p, i_b][i] = vels[i_b_, i_p_, i]

    @qd.kernel
    def _kernel_get_particles_vel(
        self, particles_idx: qd.types.ndarray(), envs_idx: qd.types.ndarray(), vels: qd.types.ndarray()
    ):
        for i_p_, i_b_ in qd.ndrange(particles_idx.shape[1], envs_idx.shape[0]):
            i_p = particles_idx[i_b_, i_p_]
            i_b = envs_idx[i_b_]
            for i in qd.static(range(3)):
                vels[i_b_, i_p_, i] = self.particles.vel[0, i_p, i_b][i]

    @qd.kernel
    def _kernel_set_particles_active(
        self, particles_idx: qd.types.ndarray(), envs_idx: qd.types.ndarray(), actives: qd.types.ndarray()
    ):
        for i_p_, i_b_ in qd.ndrange(particles_idx.shape[1], envs_idx.shape[0]):
            i_p = particles_idx[i_b_, i_p_]
            i_b = envs_idx[i_b_]
            # See '_kernel_set_particles_pos'
            if i_p >= 0:
                self.particles_ng.active[0, i_p, i_b] = qd.cast(actives[i_b_, i_p_], gs.qd_bool)

    @qd.kernel
    def _kernel_get_particles_active(
        self, particles_idx: qd.types.ndarray(), envs_idx: qd.types.ndarray(), actives: qd.types.ndarray()
    ):
        for i_p_, i_b_ in qd.ndrange(particles_idx.shape[1], envs_idx.shape[0]):
            i_p = particles_idx[i_b_, i_p_]
            i_b = envs_idx[i_b_]
            actives[i_b_, i_p_] = self.particles_ng.active[0, i_p, i_b]

    @qd.kernel
    def _kernel_set_particles_actu(
        self,
        particles_idx: qd.types.ndarray(),
        envs_idx: qd.types.ndarray(),
        actus: qd.types.ndarray(),  # shape [B, n_particles, n_groups]
        n_groups: int,
    ):
        for i_p_, i_g, i_b_ in qd.ndrange(particles_idx.shape[1], n_groups, envs_idx.shape[0]):
            i_p = particles_idx[i_b_, i_p_]
            i_b = envs_idx[i_b_]
            # See '_kernel_set_particles_pos'
            if i_p >= 0:
                if self.particles_info.muscle_group[i_p] == i_g:
                    self.particles.actu[0, i_p, i_b] = actus[i_b_, i_p_, i_g]

    @qd.kernel
    def _kernel_get_particles_actu(
        self, particles_idx: qd.types.ndarray(), envs_idx: qd.types.ndarray(), actus: qd.types.ndarray()
    ):
        for i_p_, i_b_ in qd.ndrange(particles_idx.shape[1], envs_idx.shape[0]):
            i_p = particles_idx[i_b_, i_p_]
            i_b = envs_idx[i_b_]
            actus[i_b_, i_p_] = self.particles.actu[0, i_p, i_b]

    @qd.kernel
    def _kernel_set_particles_muscle_group(self, particles_idx: qd.types.ndarray(), muscle_group: qd.types.ndarray()):
        for i_p_ in range(particles_idx.shape[0]):
            i_p = particles_idx[i_p_]
            # See '_kernel_set_particles_pos'
            if i_p >= 0:
                self.particles_info.muscle_group[i_p] = muscle_group[i_p_]

    @qd.kernel
    def _kernel_get_particles_muscle_group(self, particles_idx: qd.types.ndarray(), muscle_group: qd.types.ndarray()):
        for i_p_ in range(particles_idx.shape[0]):
            i_p = particles_idx[i_p_]
            muscle_group[i_p_] = self.particles_info.muscle_group[i_p]

    @qd.kernel
    def _kernel_set_particles_muscle_direction(
        self, particles_idx: qd.types.ndarray(), muscle_direction: qd.types.ndarray()
    ):
        for i_p_ in range(particles_idx.shape[0]):
            i_p = particles_idx[i_p_]
            # See '_kernel_set_particles_pos'
            if i_p >= 0:
                for i in qd.static(range(3)):
                    self.particles_info.muscle_direction[i_p][i] = muscle_direction[i_p_, i]

    @qd.kernel
    def _kernel_set_particles_free(self, particles_idx: qd.types.ndarray(), free: qd.types.ndarray()):
        for i_p_ in range(particles_idx.shape[0]):
            i_p = particles_idx[i_p_]
            # See '_kernel_set_particles_pos'
            if i_p >= 0:
                self.particles_info.free[i_p] = free[i_p_]

    @qd.kernel
    def _kernel_get_particles_free(self, particles_idx: qd.types.ndarray(), free: qd.types.ndarray()):
        for i_p_ in range(particles_idx.shape[0]):
            i_p = particles_idx[i_p_]
            free[i_p_] = self.particles_info.free[i_p]

    @qd.kernel
    def _kernel_get_particles_mass(
        self, particles_idx: qd.types.ndarray(), envs_idx: qd.types.ndarray(), masses: qd.types.ndarray()
    ):
        for i_p_, i_b_ in qd.ndrange(particles_idx.shape[1], envs_idx.shape[0]):
            i_p = particles_idx[i_b_, i_p_]
            masses[i_b_, i_p_] = self.particles_info.mass[i_p] / self._particle_volume_scale

    # ------------------------------------------------------------------------------------
    # -------------------------------- particle constraints ------------------------------
    # ------------------------------------------------------------------------------------

    @gs.assert_built
    @mutates(StateChange.MODEL)
    def set_particle_constraints(self, particles_mask, link_idx, stiffness, *, particles_start=0):
        """
        Attach some particles to a rigid link with critically damped springs.

        Each particle is pulled toward the position it holds now, expressed in the frame of the link. The springs are
        terms of the implicit solve (see mpm_func.func_build_constraint_rows), stable at any stiffness, and act on the
        particles alone: the link feels no reaction.

        Parameters
        ----------
        particles_mask : array_like, shape ([n_envs,] n_particles)
            Boolean mask over consecutive particles of the solver, selecting those to attach.
        link_idx : int
            Global index of the rigid link.
        stiffness : float
            Stiffness of the springs.
        particles_start : int, optional
            Global index of the particle the mask starts at. Defaults to 0.
        """
        if not isinstance(link_idx, int):
            gs.raise_exception("link_idx must be an integer.")
        if not self._constraints_initialized:
            self.init_constraints()
        particles_mask = broadcast_tensor(
            particles_mask, gs.tc_bool, (self._B, -1), ("envs_idx", "particles_idx")
        ).contiguous()
        if particles_start < 0 or particles_start + particles_mask.shape[1] > self.n_particles:
            gs.raise_exception("The particle mask overruns the particles of the solver.")
        rigid_solver = self._sim.rigid_solver
        # The attachment is a configuration of the constraint, which carries no gradient
        with torch.no_grad():
            link_pos = rigid_solver.get_links_pos(links_idx=link_idx)[..., 0, :]
            link_quat = rigid_solver.get_links_quat(links_idx=link_idx)[..., 0, :]
        link_pos = broadcast_tensor(link_pos, gs.tc_float, (self._B, 3)).contiguous()
        link_quat = broadcast_tensor(link_quat, gs.tc_float, (self._B, 4)).contiguous()
        self._kernel_set_particle_constraints(particles_start, link_idx, stiffness, particles_mask, link_pos, link_quat)

    @gs.assert_built
    @mutates(StateChange.MODEL)
    def remove_particle_constraints(self, particles_mask=None, *, particles_start=0):
        """
        Detach some particles from the rigid link they are attached to (see 'set_particle_constraints').

        Parameters
        ----------
        particles_mask : None | array_like, shape ([n_envs,] n_particles), optional
            Boolean mask over consecutive particles of the solver, selecting those to detach. If None, all particles
            from `particles_start` on will be detached. Defaults to None.
        particles_start : int, optional
            Global index of the particle the mask starts at. Defaults to 0.
        """
        if not self._constraints_initialized:
            return
        n_particles = -1
        if particles_mask is None:
            particles_mask = True
            n_particles = self.n_particles - particles_start
        particles_mask = broadcast_tensor(
            particles_mask, gs.tc_bool, (self._B, n_particles), ("envs_idx", "particles_idx")
        ).contiguous()
        if particles_start < 0 or particles_start + particles_mask.shape[1] > self.n_particles:
            gs.raise_exception("The particle mask overruns the particles of the solver.")
        self._kernel_remove_particle_constraints(particles_start, particles_mask)

    @qd.kernel
    def _kernel_set_particle_constraints(
        self,
        particles_start: int,
        link_idx: int,
        stiffness: float,
        particles_mask: qd.types.ndarray(),  # shape [n_envs, n_particles] boolean mask
        link_pos: qd.types.ndarray(),  # shape [n_envs, 3]
        link_quat: qd.types.ndarray(),  # shape [n_envs, 4]
    ):
        for i_p_, i_b in qd.ndrange(particles_mask.shape[1], particles_mask.shape[0]):
            if particles_mask[i_b, i_p_]:
                i_p = i_p_ + particles_start
                # Get current particle position. Between two steps, the state sits in frame 0 (see 'Solver.has_frames')
                pos = self.particles.pos[0, i_p, i_b]

                # Get link transform
                l_pos = qd.Vector([link_pos[i_b, 0], link_pos[i_b, 1], link_pos[i_b, 2]], dt=gs.qd_float)
                l_quat = qd.Vector(
                    [link_quat[i_b, 0], link_quat[i_b, 1], link_quat[i_b, 2], link_quat[i_b, 3]], dt=gs.qd_float
                )

                # Compute offset in link's local frame
                local_pos = gu.qd_inv_transform_by_trans_quat(pos, l_pos, l_quat)

                # Store constraint info
                self.particle_constraints[i_p, i_b].is_constrained = True
                self.particle_constraints[i_p, i_b].target_pos = pos  # initial target is current position
                self.particle_constraints[i_p, i_b].stiffness = stiffness
                self.particle_constraints[i_p, i_b].link_idx = link_idx
                self.particle_constraints[i_p, i_b].link_local_pos = local_pos

    @qd.kernel
    def _kernel_remove_particle_constraints(
        self,
        particles_start: int,
        particles_mask: qd.types.ndarray(),  # shape [n_envs, n_particles] boolean mask
    ):
        for i_p_, i_b in qd.ndrange(particles_mask.shape[1], particles_mask.shape[0]):
            if particles_mask[i_b, i_p_]:
                i_p = i_p_ + particles_start
                self.particle_constraints[i_p, i_b].is_constrained = False
                self.particle_constraints[i_p, i_b].link_idx = -1

    # ------------------------------------------------------------------------------------
    # ----------------------------------- properties -------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def n_particles(self):
        if self.is_built:
            return self._n_particles
        return sum(entity.n_particles for entity in self._entities)

    @property
    def n_vvert_supports(self):
        """
        Number of particles supporting each visual vertex.
        """
        return self._n_vvert_supports

    @property
    def n_vverts(self):
        if self.is_built:
            return self._n_vverts
        return sum(entity.n_vverts for entity in self._entities)

    @property
    def n_vfaces(self):
        if self.is_built:
            return self._n_vfaces
        return sum(entity.n_vfaces for entity in self._entities)

    @property
    def grid_density(self):
        return self._grid_density

    @property
    def particle_size(self):
        return self._particle_size

    @property
    def particle_radius(self):
        return self._particle_size / 2.0

    @property
    def upper_bound(self):
        return self._upper_bound

    @property
    def lower_bound(self):
        return self._lower_bound

    @property
    def leaf_block_size(self):
        raise DeprecationError("This property has been removed.")

    @property
    def use_sparse_grid(self):
        return DeprecationError("This property has been removed.")

    @property
    def dx(self):
        return self._dx

    @property
    def inv_dx(self):
        return self._inv_dx

    @property
    def particle_volume_real(self):
        return self._particle_volume_real

    @property
    def particle_volume(self):
        return self._particle_volume

    @property
    def particle_volume_scale(self):
        return self._particle_volume_scale

    @property
    def is_built(self):
        return self._scene._is_built

    @property
    def lower_bound_cell(self):
        return self._lower_bound_cell

    @property
    def upper_bound_cell(self):
        return self._upper_bound_cell

    @property
    def grid_res(self):
        return self._grid_res

    @property
    def grid_offset(self):
        return self._grid_offset

    @property
    def enable_CPIC(self):
        return self._enable_CPIC

    @property
    def enable_particle_constraints(self):
        return self._enable_particle_constraints


class MpmCheckpoint(IntEnum):
    """Checkpoints of the substep graph. GRID and ASSEMBLE pause on a capacity overflow; the others only gate a
    resume, so that a resumed launch skips every stage before its checkpoint."""

    GRID = 0
    PREDICT = 1
    ASSEMBLE = 2
    SOLVE = 3
    PCG = 4
    SOLVE_POST = 5
    LINE_SEARCH_TRIAL = 6
    NEWTON_POST = 7
    FINALIZE = 8


@qd.kernel(graph=True, checkpoints=True)
def kernel_substep(
    particles: MpmParticles,
    particles_ng: MpmParticlesNG,
    particles_info: MpmParticlesInfo,
    materials: MpmMaterialTable,
    grid: qd.template(),
    newton: qd.template(),
    linear: qd.template(),
    control: qd.template(),
    force_fields: qd.template(),
    config: qd.template(),
    params: qd.template(),
    gravity: qd.Tensor,
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    links_state: array_class.LinksState,
    links_info: array_class.LinksInfo,
    rigid_info: array_class.RigidInfo,
    sdf_info: array_class.SDFInfo,
    rigid_query: qd.template(),
    f: qd.i32,
    t: qd.f32,
):
    """One implicit MPM substep, from frame f to frame f + 1, as one graph.

    Every stage sits under a checkpoint: a launch resumed after a capacity yield must not run any stage before the
    resume point (Quadrants #956 may otherwise run a top-level func after a yielding checkpoint once). Checkpoint
    bodies wrap their calls in a static branch, as checkpoint lowering rejects bare top-level func calls.
    """
    dt = params.substep_dt[()]
    with qd.checkpoint(MpmCheckpoint.GRID, yield_on=grid.grid_overflow):
        if qd.static(True):
            mpm_func.func_build_grid(particles, particles_ng, grid, params, f)
    with qd.checkpoint(MpmCheckpoint.PREDICT, yield_on=control.never_yield):
        if qd.static(True):
            mpm_func.func_p2g(particles, particles_info, grid, params, gravity, f, dt)
            mpm_func.func_apply_force_fields(grid, force_fields, params, t, dt)
            mpm_func.func_fix_nodes(grid)
            mpm_func.func_reset_contact_rows(grid, newton)
            mpm_func.func_newton_reset(grid, newton)
            mpm_func.func_build_wall_rows(particles, particles_info, grid, newton, params, f, dt)
            if qd.static(config.has_rigid):
                func_build_rigid_rows(
                    particles,
                    particles_info,
                    grid,
                    newton,
                    geoms_state,
                    geoms_info,
                    links_state,
                    links_info,
                    rigid_info,
                    sdf_info,
                    config,
                    config,
                    params,
                    rigid_query,
                    f,
                    dt,
                )
            mpm_func.func_newton_start(grid, newton, params, dt)
            func_reset_newton_control(control)
            mpm_func.func_energy(particles, particles_info, materials, grid, newton, control, params, f, dt, False)

    while qd.graph.do_while(control.newton_cond):
        with qd.checkpoint(MpmCheckpoint.ASSEMBLE, yield_on=linear.matrix.triplet_overflow):
            if qd.static(True):
                func_reset_newton_measures(control)
                mpm_func.func_report_extent(grid, linear)
                mpm_func.func_assemble(particles, particles_info, materials, grid, newton, linear, params, f, dt)
        with qd.checkpoint(MpmCheckpoint.SOLVE, yield_on=control.never_yield):
            if qd.static(True):
                func_sort_reduce_bcoo(linear.matrix)
                mpm_func.func_reset_brick(grid, linear)
                mpm_func.func_gather_brick(linear)
                mpm_func.func_invert_brick(grid, linear)
                func_init_linear_pcg_residual(linear.pcg, linear.b_rhs, linear.solution)
                mpm_func.func_apply_brick(grid, linear)
                func_init_linear_pcg_direction(linear.pcg)
        with qd.checkpoint(MpmCheckpoint.PCG, yield_on=control.never_yield):
            if qd.static(True):
                mpm_func.func_solve_brick_pcg(grid, linear)
        with qd.checkpoint(MpmCheckpoint.SOLVE_POST, yield_on=control.never_yield):
            if qd.static(True):
                mpm_func.func_negate_du(grid, newton, linear)
                mpm_func.func_record_start_point(grid, newton)
                mpm_func.func_contribute_newton_max_disp(particles, grid, newton, control, params, f)
                func_check_convergence(control)
                func_init_line_search(control)
        while qd.graph.do_while(control.ls_cond):
            with qd.checkpoint(MpmCheckpoint.LINE_SEARCH_TRIAL, yield_on=control.never_yield):
                if qd.static(True):
                    mpm_func.func_step_forward(grid, newton, control)
                    mpm_func.func_energy(
                        particles, particles_info, materials, grid, newton, control, params, f, dt, True
                    )
                    func_accept_or_backtrack(control)
        with qd.checkpoint(MpmCheckpoint.NEWTON_POST, yield_on=control.never_yield):
            if qd.static(True):
                func_update_convergence(control)

    with qd.checkpoint(MpmCheckpoint.FINALIZE, yield_on=control.never_yield):
        if qd.static(True):
            mpm_func.func_grid_velocity(grid, newton, dt)
            mpm_func.func_g2p_advect(particles, particles_ng, particles_info, materials, grid, params, f, dt)


@qd.kernel
def kernel_grid_keys(
    particles: MpmParticles,
    particles_ng: MpmParticlesNG,
    grid: qd.template(),
    params: qd.template(),
    f: qd.i32,
):
    mpm_func.func_reset_grid(grid)
    mpm_func.func_compose_particle_keys(particles, particles_ng, grid, params, f)


@qd.kernel
def kernel_grid_blocks(
    grid: qd.template(),
    params: qd.template(),
):
    mpm_func.func_append_particle_blocks(grid, params)
    mpm_func.func_gen_neighbor_keys(grid)


@qd.kernel
def kernel_grid_block_flags(grid: qd.template()):
    mpm_func.func_block_flags(grid)


@qd.kernel
def kernel_grid_publish_blocks(grid: qd.template()):
    mpm_func.func_publish_blocks(grid)
    mpm_func.func_publish_grid_demand(grid)
    mpm_func.func_cell_flags(grid)


@qd.kernel
def kernel_grid_cells(grid: qd.template()):
    mpm_func.func_publish_cells(grid)
    mpm_func.func_reset_cell_segs(grid)
    mpm_func.func_fill_cell_segs(grid)


def build_grid(particles, particles_ng, grid, params, f):
    """Host-driven func_build_grid, its sorts and scans shared launches."""
    kernel_grid_keys(particles, particles_ng, grid, params, f)
    dynamic_radix_sort(
        grid.particle_sort, grid.sort_keys, grid.sort_keys_tmp, grid.sort_ids, grid.sort_ids_tmp, grid.n_sort_keys
    )
    kernel_grid_blocks(grid, params)
    dynamic_radix_sort(
        grid.block_sort, grid.nbr_keys, grid.nbr_keys_tmp, grid.nbr_perm, grid.nbr_perm_tmp, grid.n_neighbor_keys
    )
    kernel_grid_block_flags(grid)
    kernel_dynamic_exclusive_sum(grid.block_scan, grid.block_flags, grid.block_ranks, grid.n_neighbor_keys)
    kernel_grid_publish_blocks(grid)
    kernel_dynamic_exclusive_sum(grid.cell_scan, grid.cell_flags, grid.cell_ranks, grid.n_valid)
    kernel_dynamic_exclusive_sum(grid.task_scan, grid.task_flags, grid.task_ranks, grid.n_valid)
    kernel_grid_cells(grid)


@qd.kernel
def kernel_predict(
    particles: MpmParticles,
    particles_info: MpmParticlesInfo,
    grid: qd.template(),
    newton: qd.template(),
    force_fields: qd.template(),
    params: qd.template(),
    gravity: qd.Tensor,
    f: qd.i32,
    t: qd.f32,
):
    dt = params.substep_dt[()]
    mpm_func.func_p2g(particles, particles_info, grid, params, gravity, f, dt)
    mpm_func.func_apply_force_fields(grid, force_fields, params, t, dt)
    mpm_func.func_fix_nodes(grid)
    mpm_func.func_reset_contact_rows(grid, newton)
    mpm_func.func_newton_reset(grid, newton)


@qd.kernel
def kernel_contact_rows(
    particles: MpmParticles,
    particles_info: MpmParticlesInfo,
    grid: qd.template(),
    newton: qd.template(),
    control: qd.template(),
    config: qd.template(),
    params: qd.template(),
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    links_state: array_class.LinksState,
    links_info: array_class.LinksInfo,
    rigid_info: array_class.RigidInfo,
    sdf_info: array_class.SDFInfo,
    rigid_query: qd.template(),
    f: qd.i32,
):
    """The wall and Rigid contact rows of the substep, then arm the Newton loop."""
    dt = params.substep_dt[()]
    mpm_func.func_build_wall_rows(particles, particles_info, grid, newton, params, f, dt)
    if qd.static(config.has_rigid):
        func_build_rigid_rows(
            particles,
            particles_info,
            grid,
            newton,
            geoms_state,
            geoms_info,
            links_state,
            links_info,
            rigid_info,
            sdf_info,
            config,
            config,
            params,
            rigid_query,
            f,
            dt,
        )
    mpm_func.func_newton_start(grid, newton, params, dt)
    func_reset_newton_control(control)


@qd.kernel
def kernel_constraint_rows(
    particles_info: MpmParticlesInfo,
    grid: qd.template(),
    newton: qd.template(),
    params: qd.template(),
    constraints: qd.template(),
    links_state: array_class.LinksState,
):
    mpm_func.func_build_constraint_rows(particles_info, grid, newton, params, constraints, links_state)


@qd.kernel
def kernel_constraint_adjoint(
    particles: MpmParticles,
    grid: qd.template(),
    newton: qd.template(),
    linear: qd.template(),
    params: qd.template(),
    constraints: qd.template(),
    links_state: array_class.LinksState,
    f: qd.i32,
):
    mpm_adjoint_func.func_constraint_adjoint(particles, grid, newton, linear, params, constraints, links_state, f)


@qd.kernel
def kernel_contact_invweight(
    particles: MpmParticles,
    particles_info: MpmParticlesInfo,
    grid: qd.template(),
    linear: qd.template(),
    params: qd.template(),
    f: qd.i32,
):
    mpm_func.func_contact_invweight(particles, particles_info, grid, linear, params, f)


@qd.kernel
def kernel_assemble(
    particles: MpmParticles,
    particles_info: MpmParticlesInfo,
    materials: MpmMaterialTable,
    grid: qd.template(),
    newton: qd.template(),
    linear: qd.template(),
    control: qd.template(),
    params: qd.template(),
    f: qd.i32,
    is_exact: qd.template(),
):
    dt = params.substep_dt[()]
    func_reset_newton_measures(control)
    mpm_func.func_report_extent(grid, linear)
    mpm_func.func_assemble(particles, particles_info, materials, grid, newton, linear, params, f, dt, is_exact)


@qd.kernel
def kernel_finalize(
    particles: MpmParticles,
    particles_ng: MpmParticlesNG,
    particles_info: MpmParticlesInfo,
    materials: MpmMaterialTable,
    grid: qd.template(),
    newton: qd.template(),
    params: qd.template(),
    f: qd.i32,
):
    dt = params.substep_dt[()]
    mpm_func.func_grid_velocity(grid, newton, dt)
    mpm_func.func_g2p_advect(particles, particles_ng, particles_info, materials, grid, params, f, dt)


@qd.kernel
def kernel_adjoint_g2p(
    particles: MpmParticles,
    grid: qd.template(),
    newton: qd.template(),
    params: qd.template(),
    f: qd.i32,
):
    """The adjoint of u* into `newton.du` and the direct terms of the adjoint of frame f (see mpm_adjoint_func)."""
    dt = params.substep_dt[()]
    mpm_adjoint_func.func_adjoint_reset_u_bar(grid, newton)
    mpm_adjoint_func.func_adjoint_g2p(particles, grid, newton, params, f, dt)


@qd.kernel
def kernel_adjoint_rhs(
    grid: qd.template(),
    newton: qd.template(),
    linear: qd.template(),
):
    mpm_adjoint_func.func_adjoint_rhs(grid, newton, linear)


@qd.kernel
def kernel_adjoint_pullback(
    particles: MpmParticles,
    particles_ng: MpmParticlesNG,
    particles_info: MpmParticlesInfo,
    materials: MpmMaterialTable,
    grid: qd.template(),
    newton: qd.template(),
    linear: qd.template(),
    force_fields: qd.template(),
    params: qd.template(),
    gravity: qd.Tensor,
    f: qd.i32,
    t: qd.f32,
):
    """The pullback of the adjoint solution, its share of the wall rows and the reverse of P2G into frame f, then the
    adjoint of frame f + 1 cleared (see mpm_adjoint_func). `t` is the time the forward substep evaluated the force
    fields at."""
    dt = params.substep_dt[()]
    mpm_adjoint_func.func_adjoint_nodes(grid, newton, linear, force_fields, params, t, dt)
    mpm_adjoint_func.func_adjoint_particles(
        particles, particles_info, materials, grid, newton, linear, params, gravity, f, dt
    )
    mpm_adjoint_func.func_wall_contact_adjoint(particles, particles_info, grid, newton, linear, params, f, dt)
    mpm_adjoint_func.func_adjoint_consume(particles, particles_ng, params, f)

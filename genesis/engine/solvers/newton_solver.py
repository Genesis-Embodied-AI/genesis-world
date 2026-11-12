"""
Newton solver of the substeps of the implicit material point method (MPM) solver and of the cloth.

The solver runs the Newton iterations, the preconditioned conjugate gradient and the line search. The terms of the
energy and their solve come from the solvers that own them: the MPM solver for the nodes of its grid (see
'func_mpm_assemble' in mpm_solver.py), the FEM solver for the vertices of the cloth (see 'func_cloth_assemble' in
fem_solver.py), the rigid solver for its dofs (see rigid/global_newton.py), the coupling for the contacts between the
particles and the rigid geoms (see couplers/mpm_rigid_global_newton.py), and the incremental potential contact of the
cloth with itself, with the fixed planes and with the rigid contact proxies of the links (see ipc_contact.py and
rigid/contact_proxy.py), whose data the Newton solver holds.
"""

from collections.abc import Iterator
from enum import IntEnum, auto
from typing import TYPE_CHECKING

import numpy as np

import quadrants as qd

import genesis as gs
from genesis.engine.couplers.mpm_rigid_global_newton import (
    func_add_contact_link_terms,
    func_assemble_mpm_rigid_coupling,
    func_contact_matvec,
    func_contact_terms,
    func_mark_constrained_links,
)
from genesis.engine.numerics.bcoo import assemble_bcoo_matrix, func_add_by_key
from genesis.engine.numerics.newton_rows import (
    NewtonRows,
    NewtonSegment,
    func_rows_matvec,
    func_rows_precond_invert,
    func_set_segment,
    grow_newton_rows,
)
from genesis.engine.solvers.rigid.constraint.data import ConstraintState
from genesis.engine.solvers.rigid.contact_proxy import (
    PROXIES,
    PROXY_VERTICES,
    IPCProxiesInfo,
    IPCProxiesState,
    IPCProxyVerticesInfo,
    build_contact_proxies,
    func_proxies_publish_contact,
    kernel_proxies_init_pose,
)
from genesis.engine.solvers.rigid.data import DynInfo, DynState, RigidInfo
from genesis.engine.solvers.rigid.global_newton import (
    NewtonRigidDofs,
    NewtonRigidLinks,
    RigidContactStiffness,
    func_link_force_to_dofs,
    func_link_spatial_disp,
    func_rigid_assemble_gradient,
    func_rigid_factor,
    func_rigid_finish,
    func_rigid_init_line_search,
    func_rigid_init_substep,
    func_rigid_matvec,
    func_rigid_solve_batch,
    func_rigid_step_forward,
    func_rigid_update_constraint,
)
from genesis.engine.materials import FEM
from genesis.options.solvers import LegacyCouplerOptions, NewtonOptions
from genesis.utils.data import ENVS, Config, Data, DataItem, DataKind, field, iter_data
from genesis.utils.misc import qd_isfinite, qd_to_numpy

from .base_solver import ErrorCode, Solver
from .fem_solver import (
    TRIANGLE_PAIRS,
    HINGE_PAIRS,
    FEMInfo,
    FEMState,
    func_cloth_assemble,
    func_cloth_energy,
    func_cloth_init_line_search,
    func_cloth_publish_contact,
    func_cloth_publish_rows,
    func_cloth_scale_direction,
    func_cloth_step_forward,
)
from .ipc_contact import (
    BODIES,
    CONTACT_EDGES,
    CONTACT_TRIANGLES,
    CONTACT_VERTICES,
    EE_PAIRS,
    HALFPLANE_VERTICES,
    HALFPLANES_LABEL,
    KAPPA,
    PARTITION,
    PT_PAIRS,
    RIGID_VERTICES,
    IPCContact,
    IPCFrictionPairs,
    IPCInfo,
    IPCPairs,
    IPCState,
    IPCSurfacePrimitives,
    func_contact_assemble,
    func_contact_energy,
    func_contact_init_substep,
    kernel_contact_ccd,
    kernel_contact_init_kappa,
    kernel_contact_kappa_iteration,
    kernel_contact_partition,
)
from .mpm_solver import (
    STENCIL_PAIRS,
    MPMInfo,
    MPMSparseGrid,
    MPMState,
    func_mpm_assemble,
    func_mpm_init_line_search,
    func_mpm_kinetic_energy,
    func_mpm_step_forward,
    func_particle_energy,
)

if TYPE_CHECKING:
    from genesis.engine.scene import Scene
    from genesis.engine.simulator import Simulator

# The arrays of the dofs and of the links of the rigid solver in the Newton solve, which a solve without them leaves out,
# and those of the contact of the cloth, which a scene without it leaves out
RIGID = "rigid"
IPC = "ipc"
# The arrays of the rigid contact proxies, which a scene without them leaves out
PROXIES_LABEL = "proxies"


class NewtonErrorCode(ErrorCode):
    """The error flags the Newton solver writes in its errno (see 'Solver.register_errors')."""

    INVALID_SOLVE_NAN = auto()
    INVALID_CONTACT_THICKNESS = auto()


class NewtonFailure(IntEnum):
    """The reason the Newton solve of an environment stopped short of convergence."""

    NONE = 0
    ITERATIONS = 1
    NON_FINITE = 2


class NewtonEnvs(Data, kind=DataKind.SCRATCH, shape=(ENVS,)):
    """
    The progress and the scalars of the Newton solve of every environment.

    `is_active` says whether its Newton iterations go on, `is_converged` whether its last direction moved every
    particle, vertex of the cloth and dof by less than the tolerance, and `failure` why it stopped short of convergence
    (see NewtonFailure). `energy_start` is the energy at the start point of the line search, `energy_trial` at its trial
    point of step `alpha`, and `max_displacement` the largest update of the displacement of a particle, of a vertex or
    of a dof along the direction. The scalars of the preconditioned conjugate gradient (PCG) follow: `rz` is r . z, `rz_tol` the value
    below which it stops, and `pcg_alpha` and `pcg_beta` its steps. `rigid_cost` is the cost of the constraint solve of
    the rigid solver at the current iterate of its dofs (see 'func_rigid_update_constraint' in rigid/global_newton.py).
    """

    is_active: qd.Tensor = field(bool)
    is_converged: qd.Tensor = field(bool)
    failure: qd.Tensor = field(int)
    n_iterations: qd.Tensor = field(int)
    energy_start: qd.Tensor = field(float)
    energy_trial: qd.Tensor = field(float)
    max_displacement: qd.Tensor = field(float)
    alpha: qd.Tensor = field(float)
    is_line_search_active: qd.Tensor = field(bool)
    n_line_search_iterations: qd.Tensor = field(int)
    is_pcg_active: qd.Tensor = field(bool)
    rz: qd.Tensor = field(float)
    rz_new: qd.Tensor = field(float)
    rz_tol: qd.Tensor = field(float)
    p_dot_ap: qd.Tensor = field(float)
    pcg_alpha: qd.Tensor = field(float)
    pcg_beta: qd.Tensor = field(float)
    rigid_cost: qd.Tensor = field(float, labels=RIGID)


class NewtonLoops(Data, kind=DataKind.SCRATCH, shape=()):
    """
    The flags that keep the loops of the Newton solve running as long as an environment iterates on, with the iterations
    left to the preconditioned conjugate gradient (PCG).

    The flags are ndarrays on every backend, since a loop of a graph tests the same ndarray on every call. Between two
    checks (see 'NewtonOptions.failure_check_interval'), `has_failed_iterations` latches a solve that reached its
    iteration cap.
    """

    is_newton_running: qd.Tensor = field(qd.i32, ndarray=True)
    is_line_search_running: qd.Tensor = field(qd.i32, ndarray=True)
    is_pcg_running: qd.Tensor = field(qd.i32, ndarray=True)
    n_pcg_iterations_left: qd.Tensor = field(int, ndarray=True)
    has_failed_iterations: qd.Tensor = field(int, ndarray=True)


class NewtonInfo(Data, kind=DataKind.CONSTANT, shape=()):
    """The constants of the Newton solve (see NewtonOptions), the tolerance on the displacement of a particle, a vertex
    of the cloth or a dof being the tolerance on its velocity over a substep."""

    max_iterations: qd.Tensor = field(int)
    line_search_max_iterations: qd.Tensor = field(int)
    line_search_shrink: qd.Tensor = field(float)
    displacement_tol: qd.Tensor = field(float)
    pcg_max_iterations: qd.Tensor = field(int)
    pcg_tol: qd.Tensor = field(float)


class NewtonConfig(Config):
    """The compile-time configuration of the kernels of the Newton solver."""

    # Whether the dofs of the rigid solver join the unknowns (see 'RigidSolver.is_solved_in_global_newton')
    has_rigid_dofs: bool
    # Whether the rigid solver has constraint rows. Their cost then joins the energy and the factor of the Hessian of
    # its Newton constraint solver preconditions its dofs, its Gauss cost and the factor of its mass matrix otherwise.
    has_rigid_rows: bool
    # Whether the nodes of the implicit MPM solver and the vertices of the cloth join the unknowns (see NewtonSegment),
    # and whether the cloth touches fixed planes (see ipc_contact.py)
    has_mpm: bool
    has_cloth: bool
    has_halfplanes: bool
    # Whether the contact surface holds rigid vertices, those of the rigid contact proxies (see ipc_contact.py)
    has_rigid_vertices: bool
    # The contact of the cloth (see 'gs.options.IPCOptions'): whether its time of contact bounds the closing rate of a
    # pair rather than the motion of its vertices, whether every group of contact bodies takes a step of its own, whether
    # its stiffness adapts, and whether it grows at the start of every Newton iteration rather than of every substep
    is_ccd_directional: bool
    has_ccd_partition: bool
    has_adaptive_kappa: bool
    is_kappa_tick_newton: bool


class NewtonData(Data):
    """The arrays of the Newton solver of a fixed size, allocated once its scene is built."""

    newton_envs: NewtonEnvs
    newton_loops: NewtonLoops
    newton_info: NewtonInfo
    rigid_dofs: NewtonRigidDofs = field(labels=RIGID)
    rigid_links: NewtonRigidLinks = field(labels=RIGID)
    errno: qd.Tensor = field(int, shape=(ENVS,), kind=DataKind.STATE)
    ipc_state: IPCState = field(labels=IPC)
    ipc_info: IPCInfo = field(labels=IPC)
    proxies_state: IPCProxiesState = field(labels=PROXIES_LABEL)
    proxies_info: IPCProxiesInfo = field(labels=PROXIES_LABEL)
    proxy_vertices_info: IPCProxyVerticesInfo = field(labels=PROXIES_LABEL)


class NewtonSolver(Solver):
    """
    Solver of the substeps of the implicit material point method (MPM) solver and of the cloth by Newton iterations on
    their incremental potential.

    The unknowns of a substep are the displacement of the nodes of the sparse grid of the MPM solver over it (see
    MPMSparseGrid), and when the rigid geoms couple with the particles, the accelerations of the dofs of the rigid
    solver, whose constraint solve then joins the energy (see 'RigidSolver.is_solved_in_global_newton'), or the
    positions of the vertices of the cloth at its end (see 'FEMSolver.is_solved_in_global_newton'), with their contacts
    over the contact surface (see ipc_contact.py). Every iteration solves the projected Hessian against the gradient by a
    preconditioned conjugate gradient (PCG) and searches a step along the direction that lowers the energy, every
    environment iterating until its own convergence. The solve runs once every solver has prepared its substep and the
    coupler has run, and before the solvers finish it, since the MPM solver moves its particles by the displacement
    solved, the FEM solver takes the positions solved and the rigid solver integrates the accelerations solved.
    """

    # Its whole state lives in the arrays 'data' yields (see 'CheckpointMixin')
    is_checkpointable = True

    def __init__(self, scene: "Scene", sim: "Simulator", options: NewtonOptions) -> None:
        super().__init__(scene, sim, options)
        self.data_manager: NewtonData | None = None
        self.newton_envs: NewtonEnvs | None = None
        self.newton_loops: NewtonLoops | None = None
        self.newton_info: NewtonInfo | None = None
        self.newton_config: NewtonConfig | None = None
        self.rigid_dofs: NewtonRigidDofs | None = None
        self.rigid_links: NewtonRigidLinks | None = None
        self.ipc_state: IPCState | None = None
        self.ipc_info: IPCInfo | None = None
        # The broad phase and the pairs of the contact of the cloth, the pairs being placeholders without cloth
        self.ipc_contact: IPCContact | None = None
        self._ipc_pairs_unused: IPCPairs | None = None
        self._friction_pairs_unused: IPCFrictionPairs | None = None
        # The stiffness of the contacts that the factor of the rigid solver adds, made once the rigid solver is built
        self._rigid_contact_stiffness: RigidContactStiffness | None = None
        # The rows of the linear system, rebuilt with a larger capacity when the grid of the MPM solver outgrows them,
        # and the rows they replaced, kept referenced with their arrays freed (see 'grow_newton_rows')
        self.newton_rows: NewtonRows | None = None
        self._newton_rows_replaced: list[NewtonRows] = []

    @property
    def is_active(self) -> bool:
        mpm_solver = self._sim.mpm_solver
        is_mpm_implicit = mpm_solver.is_active and mpm_solver._use_implicit_solver
        return is_mpm_implicit or self._sim.fem_solver.is_solved_in_global_newton

    def build(self) -> None:
        super().build()
        if "ipc" in self._scene.options.model_fields_set and not isinstance(
            self._sim.coupler.options, LegacyCouplerOptions
        ):
            gs.raise_exception(
                "IPCOptions configure the contact of the cloth solved by the Newton solver, which requires "
                f"LegacyCouplerOptions, got {type(self._sim.coupler.options).__name__}."
            )
        if not self.is_active:
            return
        if self._requires_grad:
            gs.raise_exception(
                "Differentiable simulation ('requires_grad=True') is not supported by the Newton solver of the implicit "
                "MPM solver and of the cloth: set MPMOptions(use_implicit_solver=False), and remove the "
                "gs.materials.FEM.Cloth entities."
            )
        if not isinstance(self._sim.coupler.options, LegacyCouplerOptions):
            gs.raise_exception(
                "The Newton solver of the implicit MPM solver and of the cloth requires LegacyCouplerOptions, got "
                f"{type(self._sim.coupler.options).__name__}."
            )
        mpm_solver = self._sim.mpm_solver
        fem_solver = self._sim.fem_solver
        rigid_solver = self._sim.rigid_solver
        has_mpm = mpm_solver.is_active and mpm_solver._use_implicit_solver
        has_cloth = fem_solver.is_solved_in_global_newton
        halfplanes_geom_idx = []
        links_geoms = []
        if has_cloth and gs.qd_float == qd.f32:
            gs.logger.warning(
                "The cloth runs in single precision, where its contact with itself and with other cloths is less "
                "reliable: use gs.init(precision='64')."
            )
        if has_cloth:
            if has_mpm:
                gs.raise_exception(
                    "The cloth is not supported together with the implicit MPM solver: set "
                    "MPMOptions(use_implicit_solver=False)."
                )
            if not all([isinstance(entity.material, FEM.Cloth) for entity in fem_solver.entities]):
                gs.raise_exception("The cloth is not supported together with volumetric FEM entities.")
            # The fixed planes bound the cloth as halfplanes, and every link holding other collision geoms of an entity
            # coupling with the cloth owns a rigid contact proxy (see rigid/contact_proxy.py)
            entities_moving = []
            for entity in rigid_solver.entities:
                for link in entity.links:
                    geoms = []
                    for geom in link.geoms:
                        if geom.type == gs.GEOM_TYPE.PLANE:
                            if not geom.is_fixed:
                                gs.raise_exception(
                                    f"The cloth supports fixed planes alone, got a plane on the moving link "
                                    f"'{link.name}' of entity '{entity.name}'."
                                )
                            halfplanes_geom_idx.append(geom.idx)
                        elif entity.material.needs_coup:
                            geoms.append(geom)
                    if geoms:
                        if not link.is_fixed:
                            gs.raise_exception(
                                "The cloth supports fixed rigid links alone for now, got the moving link "
                                f"'{link.name}' of entity '{entity.name}'."
                            )
                        if not link.is_fixed and entity not in entities_moving:
                            entities_moving.append(entity)
                        links_geoms.append((link, entities_moving.index(entity) if not link.is_fixed else -1, geoms))
        rigid_options = self._scene.options.rigid
        has_rigid_dofs = rigid_solver.is_solved_in_global_newton
        if has_rigid_dofs:
            # The Newton solve preconditions the dofs of the rigid solver by the factor of the Hessian of its Newton
            # constraint solver, over the rows of the pyramidal friction cone, and its solution is final for every dof,
            # which a sleeping body would discard and a noslip pass after it would change
            for is_supported, requirement in (
                (rigid_options.friction_cone == gs.friction_cone.pyramidal, "friction_cone=gs.friction_cone.pyramidal"),
                (
                    rigid_options.constraint_solver == gs.constraint_solver.Newton,
                    "constraint_solver=gs.constraint_solver.Newton",
                ),
                (not rigid_options.use_hibernation, "use_hibernation=False"),
                (rigid_options.noslip_iterations == 0, "noslip_iterations=0"),
            ):
                if not is_supported:
                    gs.raise_exception(
                        f"Coupling the rigid solver with the implicit MPM solver requires RigidOptions({requirement}), "
                        "or else LegacyCouplerOptions(rigid_mpm=False)."
                    )
        has_halfplanes = len(halfplanes_geom_idx) > 0
        ipc_options = self._scene.options.ipc
        # The contact surface, every participant publishing its primitives into its own range, the cloth first
        surface, proxies = None, None
        if has_cloth:
            surface = fem_solver.build_cloth_contact_surface()
        if links_geoms:
            proxies = build_contact_proxies(links_geoms, len(fem_solver.entities), ipc_options.d_hat)
            n_cloth_vertices = len(surface.vertices_body_idx)
            surface = IPCSurfacePrimitives(
                *[
                    np.concatenate((cloth_values, proxy_values + n_cloth_vertices if is_indices else proxy_values))
                    for cloth_values, proxy_values, is_indices in zip(
                        surface,
                        proxies.surface,
                        [name in ("edges_verts_idx", "triangles_verts_idx") for name in IPCSurfacePrimitives._fields],
                    )
                ]
            )
        self.newton_config = NewtonConfig(
            has_rigid_dofs=has_rigid_dofs,
            has_rigid_rows=not rigid_options.disable_constraint,
            has_mpm=has_mpm,
            has_cloth=has_cloth,
            has_halfplanes=has_halfplanes,
            has_rigid_vertices=proxies is not None,
            is_ccd_directional=ipc_options.ccd_bound == "directional",
            has_ccd_partition=has_cloth and ipc_options.ccd_partition,
            has_adaptive_kappa=has_cloth and ipc_options.adaptive_kappa_mode == "per-vertex",
            is_kappa_tick_newton=ipc_options.adaptive_kappa_tick == "newton",
        )
        self.data_manager = NewtonData(
            sizes={
                "envs": self._B,
                "dofs": max(rigid_solver.n_dofs, 1),
                "links": max(rigid_solver.n_links, 1),
                "halfplanes": len(halfplanes_geom_idx),
                HALFPLANE_VERTICES: fem_solver.n_vertices,
                CONTACT_VERTICES: len(surface.vertices_body_idx) if has_cloth else 0,
                CONTACT_EDGES: len(surface.edges_verts_idx) if has_cloth else 0,
                CONTACT_TRIANGLES: len(surface.triangles_verts_idx) if has_cloth else 0,
                BODIES: int(surface.vertices_body_idx.max(initial=-1)) + 1 if has_cloth else 0,
                PROXIES: len(proxies.links_idx) if proxies is not None else 0,
                PROXY_VERTICES: len(proxies.vertices_proxy_idx) if proxies is not None else 0,
            },
            values=dict(
                newton_info=dict(
                    max_iterations=self._options.max_iterations,
                    line_search_max_iterations=self._options.line_search_max_iterations,
                    line_search_shrink=self._options.line_search_shrink,
                    displacement_tol=self._options.velocity_tol * self._sim.substep_dt,
                    pcg_max_iterations=self._options.pcg_max_iterations,
                    pcg_tol=self._options.pcg_tol,
                ),
                ipc_info=dict(
                    substep_dt=self._sim.substep_dt,
                    friction_eps_v=ipc_options.friction_eps_v,
                    ccd_eta=0.2,
                    ccd_max_iterations=50000,
                    hessian_clamp_ratio=1e-6,
                    mollifier_ratio=1e-3,
                    projection_rank_tolerance=1e-12,
                    projection_eigen_tolerance=1e-15,
                    projection_max_sweeps=12,
                    ee_parallel_tolerance=1e-6,
                    ee_type_parallel_tolerance=1e-14,
                    friction_parallel_ratio=1e-3,
                    intersection_tolerance=1e-12,
                    kappa_gap_ratio=ipc_options.adaptive_kappa_gap_ratio,
                    kappa_grow=ipc_options.adaptive_kappa_grow,
                    kappa_max_scale=ipc_options.adaptive_kappa_max_scale,
                    kappa_calm_time=ipc_options.adaptive_kappa_calm_time,
                    kappa_relax_time=ipc_options.adaptive_kappa_relax_time,
                    kappa_hysteresis=ipc_options.adaptive_kappa_hysteresis,
                    min_gap_unrecorded=float(np.finfo(gs.np_float).max),
                ),
            ),
            # The labels of the parts of the contact of the cloth select arrays within those of the IPC label
            unused={
                RIGID: not has_rigid_dofs,
                PROXIES_LABEL: proxies is None,
                **(
                    {
                        HALFPLANES_LABEL: not has_halfplanes,
                        PARTITION: not self.newton_config.has_ccd_partition,
                        KAPPA: not self.newton_config.has_adaptive_kappa,
                        RIGID_VERTICES: not self.newton_config.has_rigid_vertices,
                    }
                    if has_cloth
                    else {IPC: True}
                ),
            },
        )
        # The errors of each env, raised at the beginning of the next step (see 'Solver.register_errors')
        self.register_errors(
            self.data_manager.errno,
            {
                NewtonErrorCode.INVALID_SOLVE_NAN: (
                    "The Newton solve of the implicit MPM solver or of the cloth produced values that are not finite. "
                    "Please decrease the simulation timestep or adjust the material parameters."
                ),
                NewtonErrorCode.INVALID_CONTACT_THICKNESS: (
                    "The cloth lies within its contact thickness of a plane or of another part of the cloth, where its "
                    "contact is undefined. Start the cloth farther than its contact thickness from the planes and from "
                    "itself, or lower gs.materials.FEM.Cloth(contact_thickness)."
                ),
            },
        )
        self.newton_envs = self.data_manager.newton_envs
        self.newton_loops = self.data_manager.newton_loops
        self.newton_info = self.data_manager.newton_info
        self.rigid_dofs = self.data_manager.rigid_dofs
        self.rigid_links = self.data_manager.rigid_links
        self.ipc_state = self.data_manager.ipc_state
        self.ipc_info = self.data_manager.ipc_info
        if has_halfplanes:
            self.ipc_info.halfplanes_geom_idx.from_numpy(np.array(halfplanes_geom_idx, dtype=gs.np_int))
        if has_cloth:
            vertices, edges, triangles = self.ipc_info.vertices, self.ipc_info.edges, self.ipc_info.triangles
            surface_values = [
                (vertices.thickness, surface.vertices_thickness),
                (vertices.d_hat, surface.vertices_d_hat),
                (vertices.area_weight, surface.vertices_area_weight),
                (vertices.resistance, surface.vertices_resistance),
                (vertices.friction_mu, surface.vertices_friction_mu),
                (vertices.body_idx, surface.vertices_body_idx),
                (vertices.is_rigid, surface.vertices_is_rigid),
                (edges.verts_idx, surface.edges_verts_idx),
                (edges.area_weight, surface.edges_area_weight),
                (edges.rest_length2, surface.edges_rest_length2),
                (triangles.verts_idx, surface.triangles_verts_idx),
                (triangles.area_weight, surface.triangles_area_weight),
            ]
            if proxies is not None:
                proxies_info, proxy_vertices_info = (
                    self.data_manager.proxies_info,
                    self.data_manager.proxy_vertices_info,
                )
                surface_values += [
                    (proxies_info.link_idx, proxies.links_idx),
                    (proxies_info.is_fixed, proxies.is_fixed),
                    (proxies_info.surface_radius, proxies.surface_radius),
                    (proxy_vertices_info.proxy_idx, proxies.vertices_proxy_idx),
                    (proxy_vertices_info.local_pos, proxies.vertices_local_pos),
                ]
            # The arrays take the dtypes of Genesis, the meshes giving their own
            for array, values in surface_values:
                if values.dtype != bool:
                    values = values.astype(gs.np_float if array.dtype == gs.qd_float else gs.np_int, copy=False)
                array.from_numpy(values)
            if self.newton_config.has_adaptive_kappa:
                kernel_contact_init_kappa(self.ipc_state, self.ipc_info)
        else:
            self._ipc_pairs_unused = IPCPairs(sizes={PT_PAIRS: 1, EE_PAIRS: 1}, unused=True)
            self._friction_pairs_unused = IPCFrictionPairs(sizes={PT_PAIRS: 1, EE_PAIRS: 1}, unused=True)

    @property
    def data(self) -> Iterator[DataItem]:
        if self.data_manager is not None:
            yield from iter_data(self.data_manager, "newton_info")
            yield from iter_data(self.data_manager, "newton_envs")
            yield from iter_data(self.data_manager, "newton_loops")
            if self.newton_config.has_rigid_dofs:
                yield from iter_data(self.data_manager, "rigid_dofs")
                yield from iter_data(self.data_manager, "rigid_links")
            if self.newton_config.has_cloth:
                yield from iter_data(self.data_manager, "ipc_info")
                yield from iter_data(self.data_manager, "ipc_state")
            if self.newton_config.has_rigid_vertices:
                yield from iter_data(self.data_manager, "proxies_info")
                yield from iter_data(self.data_manager, "proxy_vertices_info")
                yield from iter_data(self.data_manager, "proxies_state")
            yield from iter_data(self.data_manager, "errno", DataKind.STATE)
        if self.newton_rows is not None:
            yield from iter_data(self, "newton_rows")

    def substep_pre_coupling(self) -> None:
        pass

    def substep_post_coupling(self) -> None:
        """
        Solve the substep of the implicit material point method (MPM) solver or of the cloth before the solvers finish
        it, with that of the rigid solver when it is solved in the global Newton solve (see
        'RigidSolver.is_solved_in_global_newton').
        """
        if not self.is_active:
            return
        mpm_solver = self._sim.mpm_solver
        fem_solver = self._sim.fem_solver
        rigid_solver = self._sim.rigid_solver
        mpm_state, mpm_info, sparse_grid = mpm_solver.mpm_state, mpm_solver.mpm_info, mpm_solver.sparse_grid
        fem_state, fem_info = fem_solver.fem_state, fem_solver.fem_info
        dyn_state, dyn_info, rigid_info = rigid_solver.dyn_state, rigid_solver.dyn_info, rigid_solver.rigid_info
        constraint_state = rigid_solver.constraint_solver.constraint_state
        errno = self._errno
        if self.newton_config.has_rigid_dofs and self._rigid_contact_stiffness is None:
            self._rigid_contact_stiffness = RigidContactStiffness(self.rigid_links, dyn_state.dofs, dyn_info.links)
        # The rows of every segment (see NewtonSegment): the nodes of the grid hold their kinetic block and the stencil
        # blocks of the tasks of the assembly (see STENCIL_PAIRS in mpm_solver.py), and the vertices of the cloth their
        # diagonal block and the blocks of their triangles and hinges (see TRIANGLE_PAIRS in fem_solver.py)
        n_mpm_rows, n_mpm_triplets, n_cloth_rows, n_cloth_triplets = 0, 0, 0, 0
        if self.newton_config.has_mpm:
            n_mpm_rows = int(qd_to_numpy(sparse_grid.n_blocks)) * 64
            n_mpm_triplets = n_mpm_rows + STENCIL_PAIRS * int(qd_to_numpy(mpm_state.sorted_particles.n_tasks))
        if self.newton_config.has_cloth:
            n_cloth_rows = self._B * fem_solver.n_vertices
            n_hinges = 0
            if fem_solver.fem_config.has_quadratic_bending:
                n_hinges += fem_info.cloth_quadratic_hinges.verts_idx.shape[0]
            if fem_solver.fem_config.has_dihedral_bending:
                n_hinges += fem_info.cloth_dihedral_hinges.verts_idx.shape[0]
            n_cloth_triplets = n_cloth_rows + self._B * (
                TRIANGLE_PAIRS * fem_info.cloth_triangles.verts_idx.shape[0] + HINGE_PAIRS * n_hinges
            )
        args_info = (mpm_info, fem_info, self.ipc_info, dyn_info, rigid_info)
        args_config = (mpm_solver.mpm_config, fem_solver.fem_config, rigid_solver.rigid_config, self.newton_config)
        proxies_state = self.data_manager.proxies_state
        proxies_info, proxy_vertices_info = self.data_manager.proxies_info, self.data_manager.proxy_vertices_info
        if self.newton_config.has_rigid_vertices and self.ipc_contact is None:
            kernel_proxies_init_pose(proxies_state, dyn_state, proxies_info)
        if self.newton_config.has_cloth:
            kernel_publish_contact(
                fem_solver.n_vertices,
                self.ipc_state,
                fem_state,
                proxies_state,
                dyn_state,
                proxies_info,
                proxy_vertices_info,
                self.newton_config,
            )
        if self.newton_config.has_cloth and self.ipc_contact is None:
            self.ipc_contact = IPCContact(
                self._B, self.ipc_info, self.newton_config.has_rigid_vertices, self._scene.options.ipc
            )
            if self._scene.options.ipc.intersection_check:
                intersections = self.ipc_contact.find_intersections(self.ipc_state, self.ipc_info)
                if len(intersections) > 0:
                    i_b, i_e, i_t = intersections[0]
                    edge = qd_to_numpy(self.ipc_info.edges.verts_idx)[i_e]
                    tri = qd_to_numpy(self.ipc_info.triangles.verts_idx)[i_t]
                    entities_v_start = np.array([entity.v_start for entity in fem_solver.entities])
                    edge_entity = fem_solver.entities[np.searchsorted(entities_v_start, edge[0], side="right") - 1]
                    tri_entity = fem_solver.entities[np.searchsorted(entities_v_start, tri[0], side="right") - 1]
                    gs.raise_exception(
                        f"The cloth starts crossed at {len(intersections)} edges: in environment {i_b}, the edge of "
                        f"the vertices {(edge - edge_entity.v_start).tolist()} of entity '{edge_entity.name}' crosses "
                        f"the triangle of the vertices {(tri - tri_entity.v_start).tolist()} of entity "
                        f"'{tri_entity.name}'. The contact keeps crossed cloths crossed: place them apart."
                    )
        contact = self.ipc_contact
        ipc_pairs, friction_pairs = self._ipc_pairs_unused, self._friction_pairs_unused
        if contact is not None:
            # The candidates at the start point of the substep give its first assembly and its friction pairs
            contact.search(self.ipc_state, self.ipc_info, self.newton_config, is_swept=False)
            contact.set_friction_pairs(
                self.ipc_state, self.ipc_info, int(NewtonErrorCode.INVALID_CONTACT_THICKNESS), errno
            )
            ipc_pairs, friction_pairs = contact.pairs, contact.friction_pairs
        args_state = (
            mpm_state,
            sparse_grid,
            fem_state,
            ipc_pairs,
            friction_pairs,
            self.ipc_state,
            self.rigid_dofs,
            self.rigid_links,
            dyn_state,
            constraint_state,
        )
        kernel_init_substep(self.newton_envs, self.newton_loops, *args_state, *args_info, *args_config, errno)
        for _ in range(self._options.max_iterations):
            if self.newton_config.has_adaptive_kappa and self.newton_config.is_kappa_tick_newton:
                kernel_contact_kappa_iteration(self.newton_envs.is_active, self.ipc_state, self.ipc_info)
                kernel_refresh_energy_start(self.newton_envs, *args_state, *args_info, *args_config)
            # The triplets of the contacts of the cloth follow its own (see 'func_contact_assemble' in ipc_contact.py)
            n_contact_triplets = 0
            if contact is not None:
                n_contact_triplets = contact.n_triplets
                if self.newton_config.has_halfplanes:
                    n_contact_triplets += n_cloth_rows
            self.newton_rows = grow_newton_rows(
                n_mpm_rows + n_cloth_rows,
                n_mpm_triplets + n_cloth_triplets + n_contact_triplets,
                self.newton_rows,
                self._newton_rows_replaced,
            )
            kernel_set_segments(
                n_mpm_rows, n_mpm_triplets, n_cloth_rows, n_cloth_triplets + n_contact_triplets, self.newton_rows
            )
            if mpm_solver.mpm_config.has_rigid_contact:
                kernel_assemble_mpm_rigid_coupling(
                    mpm_state,
                    sparse_grid,
                    self.rigid_dofs,
                    self.rigid_links,
                    dyn_state,
                    mpm_info,
                    dyn_info,
                    rigid_solver.rigid_config,
                    self.newton_config,
                )
            kernel_assemble_rows(
                n_cloth_triplets,
                self.newton_rows,
                mpm_state,
                sparse_grid,
                fem_state,
                ipc_pairs,
                friction_pairs,
                self.ipc_state,
                mpm_info,
                fem_info,
                self.ipc_info,
                mpm_solver.mpm_config,
                fem_solver.fem_config,
                self.newton_config,
            )
            assemble_bcoo_matrix(self.newton_rows.matrix, block_scalars=9)
            if self.newton_config.has_rigid_dofs:
                kernel_assemble_rigid(
                    self.newton_envs,
                    *args_state,
                    *args_info,
                    *args_config,
                    self._rigid_contact_stiffness,
                )
            kernel_pcg_start(
                self.newton_rows,
                self.newton_envs,
                self.newton_loops,
                *args_state,
                self.newton_info,
                *args_info,
                *args_config,
            )
            kernel_pcg_loop(
                self.newton_rows, self.newton_envs, self.newton_loops, *args_state, *args_info, *args_config
            )
            kernel_start_line_search(
                self.newton_rows,
                self.newton_envs,
                self.newton_loops,
                *args_state,
                self.newton_info,
                *args_info,
                *args_config,
            )
            if contact is not None:
                # The candidates over the sweep along the direction serve the step bound, the line search and the
                # assembly of the next iteration
                contact.search(self.ipc_state, self.ipc_info, self.newton_config, is_swept=True)
                ipc_pairs = contact.pairs
                args_state = (*args_state[:3], ipc_pairs, *args_state[4:])
                kernel_contact_ccd(
                    self.newton_envs.is_active,
                    self.newton_envs.alpha,
                    ipc_pairs,
                    self.ipc_state,
                    self.ipc_info,
                    self.newton_config,
                )
                if self.newton_config.has_ccd_partition:
                    kernel_contact_partition(self.newton_envs.is_active, self.ipc_state)
                    kernel_scale_directions(self.newton_envs.is_active, fem_state, self.ipc_state, self.ipc_info)
            while True:
                kernel_line_search_trial(
                    self.newton_envs, self.newton_loops, *args_state, self.newton_info, *args_info, *args_config
                )
                if not qd_to_numpy(self.newton_loops.is_line_search_running):
                    break
            kernel_end_iteration(self.newton_envs, self.newton_loops, self.newton_info)
            if not qd_to_numpy(self.newton_loops.is_newton_running):
                break
        kernel_flag_failures(self.newton_envs, self.newton_loops, errno)
        if self.newton_config.has_rigid_dofs:
            kernel_finish_rigid(self.newton_envs, *args_state, *args_info, *args_config, errno)
        if self._sim.cur_substep_global % self._options.failure_check_interval == 0:
            if qd_to_numpy(self.newton_loops.has_failed_iterations):
                kernel_clear_failed_iterations(self.newton_loops)
                gs.logger.warning(
                    "The Newton solve of the implicit MPM solver or of the cloth stopped before converging: raise "
                    "NewtonOptions(max_iterations) or loosen NewtonOptions(velocity_tol)."
                )


# ------------------------------------------------------------------------------------
# ------------------------------------- energy ---------------------------------------
# ------------------------------------------------------------------------------------


@qd.func(requires_top_level=True)
def func_add_energies(
    energies: qd.Tensor,
    envs_mask: qd.Tensor,
    rigid_cost: qd.Tensor,
    mpm_state: MPMState,
    sparse_grid: MPMSparseGrid,
    fem_state: FEMState,
    ipc_pairs: IPCPairs,
    friction_pairs: IPCFrictionPairs,
    ipc_state: IPCState,
    rigid_dofs: NewtonRigidDofs,
    rigid_links: NewtonRigidLinks,
    dyn_state: DynState,
    constraint_state: ConstraintState,
    mpm_info: MPMInfo,
    fem_info: FEMInfo,
    ipc_info: IPCInfo,
    dyn_info: DynInfo,
    rigid_info: RigidInfo,
    mpm_config: qd.template(),
    fem_config: qd.template(),
    rigid_config: qd.template(),
    newton_config: qd.template(),
):
    """
    Add the energy of the Newton solve at the current iterate into the entry of every environment selected.

    `envs_mask` selects the environments, and `energies` holds their entries. The energy is the incremental potential of
    the particles, of their contacts with the rigid geoms and of the nodes, the particles and the nodes of an
    environment following one another, or that of the cloth (see 'func_cloth_energy' in fem_solver.py) and of its
    contacts (see 'func_contact_energy' in ipc_contact.py), and when the
    dofs of the rigid solver join the unknowns, the cost of the rigid solver at their iterate, scaled by dt^4 to the
    units of the incremental potential.
    """
    if qd.static(newton_config.has_rigid_dofs):
        func_link_spatial_disp(rigid_dofs.x, rigid_links, dyn_state.dofs, dyn_info.links, rigid_config)
    if qd.static(newton_config.has_mpm):
        n_valid = mpm_state.sorted_particles.n_valid[None]
        qd.loop_config(block_dim=256)
        for i_s in range((n_valid + 31) // 32 * 32):
            i_b_key = gs.qd_int(-1)
            energy = gs.qd_float(0.0)
            if i_s < n_valid:
                energy_contact = gs.qd_float(0.0)
                if qd.static(mpm_config.has_rigid_contact):
                    energy_contact, gradient_contact, hessian_contact = func_contact_terms(
                        i_s,
                        mpm_state,
                        sparse_grid,
                        rigid_links,
                        dyn_state,
                        mpm_info,
                        dyn_info,
                        newton_config.has_rigid_dofs,
                    )
                i_b, energy_particle = func_particle_energy(i_s, energy_contact, mpm_state, sparse_grid, mpm_info)
                if envs_mask[i_b]:
                    i_b_key = i_b
                    energy = energy_particle
            func_add_by_key(i_b_key, energy, energies)
        func_mpm_kinetic_energy(energies, envs_mask, sparse_grid, mpm_info)
    if qd.static(newton_config.has_cloth):
        func_cloth_energy(energies, envs_mask, fem_state, fem_info, fem_config)
        func_contact_energy(energies, envs_mask, ipc_pairs, friction_pairs, ipc_state, ipc_info, newton_config)
    if qd.static(newton_config.has_rigid_dofs):
        func_rigid_update_constraint(
            rigid_cost, rigid_dofs, dyn_state, constraint_state, rigid_info, rigid_config, newton_config.has_rigid_rows
        )
        dt = rigid_info.substep_dt[None]
        for i_b in range(energies.shape[0]):
            if envs_mask[i_b]:
                energies[i_b] = energies[i_b] + dt * dt * dt * dt * rigid_cost[i_b]


@qd.kernel(fastcache=True)
def kernel_set_segments(
    n_mpm_rows: int, n_mpm_triplets: int, n_cloth_rows: int, n_cloth_triplets: int, newton_rows: NewtonRows
):
    """Lay out the rows of the iteration from the rows and the triplets of every segment (see NewtonSegment)."""
    for _ in range(1):
        func_set_segment(NewtonSegment.MPM_NODES, n_mpm_rows, n_mpm_triplets, newton_rows)
        func_set_segment(NewtonSegment.CLOTH_VERTICES, n_cloth_rows, n_cloth_triplets, newton_rows)


@qd.kernel(fastcache=True)
def kernel_publish_contact(
    proxy_vertex_start: int,
    ipc_state: IPCState,
    fem_state: FEMState,
    proxies_state: IPCProxiesState,
    dyn_state: DynState,
    proxies_info: IPCProxiesInfo,
    proxy_vertices_info: IPCProxyVerticesInfo,
    newton_config: qd.template(),
):
    """Publish the participants of the contact into the contact surface at the start point of the substep: the cloth
    (see 'func_cloth_publish_contact' in fem_solver.py), then the rigid contact proxies from the contact vertex
    `proxy_vertex_start` on (see 'func_proxies_publish_contact' in rigid/contact_proxy.py)."""
    func_cloth_publish_contact(ipc_state, fem_state, newton_config.has_rigid_vertices)
    if qd.static(newton_config.has_rigid_vertices):
        func_proxies_publish_contact(
            proxy_vertex_start, ipc_state, proxies_state, dyn_state, proxies_info, proxy_vertices_info
        )


@qd.kernel(fastcache=True)
def kernel_scale_directions(envs_mask: qd.Tensor, fem_state: FEMState, ipc_state: IPCState, ipc_info: IPCInfo):
    """Scale the Newton direction of the participants of the contact of every environment selected by the steps of their
    groups of contact bodies (see 'kernel_contact_partition' in ipc_contact.py)."""
    func_cloth_scale_direction(envs_mask, fem_state, ipc_state, ipc_info)


@qd.kernel(fastcache=True)
def kernel_init_substep(
    newton_envs: NewtonEnvs,
    newton_loops: NewtonLoops,
    mpm_state: MPMState,
    sparse_grid: MPMSparseGrid,
    fem_state: FEMState,
    ipc_pairs: IPCPairs,
    friction_pairs: IPCFrictionPairs,
    ipc_state: IPCState,
    rigid_dofs: NewtonRigidDofs,
    rigid_links: NewtonRigidLinks,
    dyn_state: DynState,
    constraint_state: ConstraintState,
    mpm_info: MPMInfo,
    fem_info: FEMInfo,
    ipc_info: IPCInfo,
    dyn_info: DynInfo,
    rigid_info: RigidInfo,
    mpm_config: qd.template(),
    fem_config: qd.template(),
    rigid_config: qd.template(),
    newton_config: qd.template(),
    errno: qd.Tensor,
):
    """
    Initialize the Newton solve of the substep in every environment at zero displacement of the nodes, but those whose
    solve failed earlier in the step, and evaluate its energy there.

    The dofs of the rigid solver start as its constraint solve starts (see 'func_rigid_init_substep' in
    rigid/global_newton.py), and the links that the particles touch are constrained over the substep (see
    'func_mark_constrained_links' in couplers/mpm_rigid_global_newton.py). The contact of the cloth takes the pose of
    the planes, its friction and its adaptive stiffness from the start of the substep (see
    'func_contact_init_substep' in ipc_contact.py).
    """
    for _ in range(1):
        newton_loops.is_newton_running[None] = 0
    if qd.static(newton_config.has_cloth):
        func_contact_init_substep(
            ipc_state,
            dyn_state,
            ipc_info,
            dyn_info,
            newton_config,
            penetration_errno=int(NewtonErrorCode.INVALID_CONTACT_THICKNESS),
            errno=errno,
        )
    for i_b in range(newton_envs.is_active.shape[0]):
        is_active = errno[i_b] == 0
        newton_envs.is_active[i_b] = is_active
        newton_envs.is_converged[i_b] = False
        newton_envs.failure[i_b] = NewtonFailure.NONE
        newton_envs.n_iterations[i_b] = 0
        newton_envs.energy_start[i_b] = 0.0
        if is_active:
            qd.atomic_max(newton_loops.is_newton_running[None], 1)
    if qd.static(newton_config.has_rigid_dofs):
        func_rigid_init_substep(
            rigid_dofs, dyn_state, constraint_state, rigid_info, rigid_config, newton_config.has_rigid_rows
        )
        func_mark_constrained_links(mpm_state, dyn_state, dyn_info)
    func_add_energies(
        newton_envs.energy_start,
        newton_envs.is_active,
        newton_envs.rigid_cost,
        mpm_state,
        sparse_grid,
        fem_state,
        ipc_pairs,
        friction_pairs,
        ipc_state,
        rigid_dofs,
        rigid_links,
        dyn_state,
        constraint_state,
        mpm_info,
        fem_info,
        ipc_info,
        dyn_info,
        rigid_info,
        mpm_config,
        fem_config,
        rigid_config,
        newton_config,
    )


@qd.kernel(fastcache=True)
def kernel_refresh_energy_start(
    newton_envs: NewtonEnvs,
    mpm_state: MPMState,
    sparse_grid: MPMSparseGrid,
    fem_state: FEMState,
    ipc_pairs: IPCPairs,
    friction_pairs: IPCFrictionPairs,
    ipc_state: IPCState,
    rigid_dofs: NewtonRigidDofs,
    rigid_links: NewtonRigidLinks,
    dyn_state: DynState,
    constraint_state: ConstraintState,
    mpm_info: MPMInfo,
    fem_info: FEMInfo,
    ipc_info: IPCInfo,
    dyn_info: DynInfo,
    rigid_info: RigidInfo,
    mpm_config: qd.template(),
    fem_config: qd.template(),
    rigid_config: qd.template(),
    newton_config: qd.template(),
):
    """Evaluate again the energy at the start point of the iteration of every environment whose adaptive stiffness grew
    (see 'kernel_contact_kappa_iteration' in ipc_contact.py), so that its line search compares energies of one
    stiffness."""
    for i_b in range(newton_envs.energy_start.shape[0]):
        if ipc_state.is_kappa_grown[i_b]:
            newton_envs.energy_start[i_b] = 0.0
    func_add_energies(
        newton_envs.energy_start,
        ipc_state.is_kappa_grown,
        newton_envs.rigid_cost,
        mpm_state,
        sparse_grid,
        fem_state,
        ipc_pairs,
        friction_pairs,
        ipc_state,
        rigid_dofs,
        rigid_links,
        dyn_state,
        constraint_state,
        mpm_info,
        fem_info,
        ipc_info,
        dyn_info,
        rigid_info,
        mpm_config,
        fem_config,
        rigid_config,
        newton_config,
    )


# ------------------------------------------------------------------------------------
# ------------------------------------ assembly --------------------------------------
# ------------------------------------------------------------------------------------


@qd.kernel(fastcache=True)
def kernel_assemble_mpm_rigid_coupling(
    mpm_state: MPMState,
    sparse_grid: MPMSparseGrid,
    rigid_dofs: NewtonRigidDofs,
    rigid_links: NewtonRigidLinks,
    dyn_state: DynState,
    mpm_info: MPMInfo,
    dyn_info: DynInfo,
    rigid_config: qd.template(),
    newton_config: qd.template(),
):
    """Stage the terms of the contacts of the particles with the rigid geoms at the current iterate (see
    'func_assemble_mpm_rigid_coupling' in couplers/mpm_rigid_global_newton.py)."""
    func_assemble_mpm_rigid_coupling(
        mpm_state,
        sparse_grid,
        rigid_dofs,
        rigid_links,
        dyn_state,
        mpm_info,
        dyn_info,
        rigid_config,
        newton_config.has_rigid_dofs,
    )


@qd.kernel(fastcache=True)
def kernel_assemble_rows(
    n_cloth_triplets: int,
    newton_rows: NewtonRows,
    mpm_state: MPMState,
    sparse_grid: MPMSparseGrid,
    fem_state: FEMState,
    ipc_pairs: IPCPairs,
    friction_pairs: IPCFrictionPairs,
    ipc_state: IPCState,
    mpm_info: MPMInfo,
    fem_info: FEMInfo,
    ipc_info: IPCInfo,
    mpm_config: qd.template(),
    fem_config: qd.template(),
    newton_config: qd.template(),
):
    """
    Assemble the rows of every segment at the current iterate: the nodes of the MPM solver (see 'func_mpm_assemble' in
    mpm_solver.py) and the vertices of the cloth (see 'func_cloth_assemble' in fem_solver.py), the triplets of its
    contacts following its first `n_cloth_triplets` ones (see 'func_contact_assemble' in ipc_contact.py).
    """
    if qd.static(newton_config.has_mpm):
        func_mpm_assemble(newton_rows, mpm_state, sparse_grid, mpm_info, mpm_config)
    if qd.static(newton_config.has_cloth):
        func_cloth_assemble(newton_rows, fem_state, fem_info, fem_config)
        func_cloth_publish_rows(newton_rows, ipc_state, fem_state)
        triplet_start = newton_rows.segments_triplet_start[qd.static(int(NewtonSegment.CLOTH_VERTICES))]
        func_contact_assemble(
            triplet_start + n_cloth_triplets, newton_rows, ipc_pairs, friction_pairs, ipc_state, ipc_info, newton_config
        )


@qd.kernel(fastcache=True)
def kernel_assemble_rigid(
    newton_envs: NewtonEnvs,
    mpm_state: MPMState,
    sparse_grid: MPMSparseGrid,
    fem_state: FEMState,
    ipc_pairs: IPCPairs,
    friction_pairs: IPCFrictionPairs,
    ipc_state: IPCState,
    rigid_dofs: NewtonRigidDofs,
    rigid_links: NewtonRigidLinks,
    dyn_state: DynState,
    constraint_state: ConstraintState,
    mpm_info: MPMInfo,
    fem_info: FEMInfo,
    ipc_info: IPCInfo,
    dyn_info: DynInfo,
    rigid_info: RigidInfo,
    mpm_config: qd.template(),
    fem_config: qd.template(),
    rigid_config: qd.template(),
    newton_config: qd.template(),
    rigid_contact_stiffness: qd.template(),
):
    """
    Assemble the gradient of the energy in the dofs of the rigid solver at the current iterate, and factor their block
    of the Hessian.

    The assembly takes the terms of the contacts of the particles that the coupling staged (see
    'kernel_assemble_mpm_rigid_coupling'), and the factor preconditions the dofs in the linear solve. The gradient holds
    that of the cost of the rigid solver and the forces of the contacts of the particles on the links (see
    'func_add_contact_link_terms' in couplers/mpm_rigid_global_newton.py). The block holds the Hessian of the rigid
    solver and the stiffness of these contacts (see RigidContactStiffness).
    """
    has_rigid_rows = qd.static(newton_config.has_rigid_rows)
    func_rigid_update_constraint(
        newton_envs.rigid_cost, rigid_dofs, dyn_state, constraint_state, rigid_info, rigid_config, has_rigid_rows
    )
    func_rigid_assemble_gradient(rigid_dofs, dyn_state, constraint_state, rigid_info, has_rigid_rows)
    func_add_contact_link_terms(mpm_state, rigid_links, dyn_state, dyn_info)
    func_link_force_to_dofs(rigid_dofs.gradient, rigid_links, dyn_state.dofs, dyn_info.links, rigid_config)
    func_rigid_factor(constraint_state, rigid_info, rigid_config, rigid_contact_stiffness, has_rigid_rows)


# ------------------------------------------------------------------------------------
# -------------------------- preconditioned conjugate gradient -----------------------
# ------------------------------------------------------------------------------------


@qd.kernel(fastcache=True)
def kernel_pcg_start(
    newton_rows: NewtonRows,
    newton_envs: NewtonEnvs,
    newton_loops: NewtonLoops,
    mpm_state: MPMState,
    sparse_grid: MPMSparseGrid,
    fem_state: FEMState,
    ipc_pairs: IPCPairs,
    friction_pairs: IPCFrictionPairs,
    ipc_state: IPCState,
    rigid_dofs: NewtonRigidDofs,
    rigid_links: NewtonRigidLinks,
    dyn_state: DynState,
    constraint_state: ConstraintState,
    newton_info: NewtonInfo,
    mpm_info: MPMInfo,
    fem_info: FEMInfo,
    ipc_info: IPCInfo,
    dyn_info: DynInfo,
    rigid_info: RigidInfo,
    mpm_config: qd.template(),
    fem_config: qd.template(),
    rigid_config: qd.template(),
    newton_config: qd.template(),
):
    """
    Start the preconditioned conjugate gradient (PCG) of the environments still iterating from a zero solution.

    The preconditioner inverts the diagonal blocks of the rows (see 'func_rows_precond_invert' in numerics/newton_rows.py)
    and the block of the dofs by its factor. An environment stops once r . z falls below `pcg_tol` times its value at
    the start, and fails its Newton solve once r . z is not finite, which a value that is not finite in its rows or its
    dofs makes.
    """
    func_rows_precond_invert(newton_rows)
    for i_b in range(newton_envs.is_active.shape[0]):
        newton_envs.is_pcg_active[i_b] = newton_envs.is_active[i_b]
        newton_envs.rz[i_b] = 0.0
        newton_envs.p_dot_ap[i_b] = 0.0
    n_rows = newton_rows.segments_start[newton_rows.segments_start.shape[0] - 1]
    qd.loop_config(block_dim=256)
    for i_r in range((n_rows + 31) // 32 * 32):
        i_b_key = gs.qd_int(-1)
        rz = gs.qd_float(0.0)
        if i_r < n_rows:
            i_b = newton_rows.rows_env[i_r]
            r = newton_rows.gradient[i_r]
            z = newton_rows.jacobi[i_r] @ r
            newton_rows.solution[i_r] = qd.Vector.zero(gs.qd_float, 3)
            newton_rows.r[i_r] = r
            newton_rows.z[i_r] = z
            newton_rows.p[i_r] = z
            newton_rows.Ap[i_r] = qd.Vector.zero(gs.qd_float, 3)
            if newton_envs.is_pcg_active[i_b]:
                i_b_key = i_b
                rz = r.dot(z)
        func_add_by_key(i_b_key, rz, newton_envs.rz)
    if qd.static(newton_config.has_rigid_dofs):
        for i_d, i_b in qd.ndrange(rigid_dofs.x.shape[0], rigid_dofs.x.shape[1]):
            rigid_dofs.r[i_d, i_b] = rigid_dofs.gradient[i_d, i_b]
            rigid_dofs.solution[i_d, i_b] = 0.0
            rigid_dofs.Ap[i_d, i_b] = 0.0
        for i_b in range(rigid_dofs.x.shape[1]):
            func_rigid_solve_batch(
                i_b,
                rigid_dofs.r,
                rigid_dofs.z,
                dyn_state,
                constraint_state,
                rigid_info,
                rigid_config,
                newton_config.has_rigid_rows,
            )
        for i_d, i_b in qd.ndrange(rigid_dofs.x.shape[0], rigid_dofs.x.shape[1]):
            rigid_dofs.p[i_d, i_b] = rigid_dofs.z[i_d, i_b]
            if newton_envs.is_pcg_active[i_b]:
                qd.atomic_add(newton_envs.rz[i_b], rigid_dofs.r[i_d, i_b] * rigid_dofs.z[i_d, i_b])
    for _ in range(1):
        newton_loops.is_pcg_running[None] = 0
        newton_loops.n_pcg_iterations_left[None] = newton_info.pcg_max_iterations[None]
    for i_b in range(newton_envs.is_active.shape[0]):
        newton_envs.rz_tol[i_b] = newton_info.pcg_tol[None] * qd.abs(newton_envs.rz[i_b])
        if newton_envs.is_pcg_active[i_b]:
            if not qd_isfinite(newton_envs.rz[i_b]):
                newton_envs.is_pcg_active[i_b] = False
                newton_envs.failure[i_b] = NewtonFailure.NON_FINITE
            elif newton_envs.rz[i_b] > 0.0:
                qd.atomic_max(newton_loops.is_pcg_running[None], 1)
            else:
                newton_envs.is_pcg_active[i_b] = False


@qd.func(requires_top_level=True)
def func_pcg_iteration(
    newton_rows: NewtonRows,
    newton_envs: NewtonEnvs,
    newton_loops: NewtonLoops,
    mpm_state: MPMState,
    sparse_grid: MPMSparseGrid,
    rigid_dofs: NewtonRigidDofs,
    rigid_links: NewtonRigidLinks,
    dyn_state: DynState,
    constraint_state: ConstraintState,
    mpm_info: MPMInfo,
    dyn_info: DynInfo,
    rigid_info: RigidInfo,
    rigid_config: qd.template(),
    newton_config: qd.template(),
):
    """
    Run one iteration of the preconditioned conjugate gradient (PCG) of the environments still solving.

    The product of the Hessian with the search direction is that of the rows (see 'func_rows_matvec' in
    numerics/newton_rows.py), in the energy form p . Ap of every environment without the dofs. When the dofs of the
    rigid solver join the unknowns, it adds their block (see 'func_rigid_matvec' in rigid/global_newton.py) and the
    coupling of the contacts of the particles with the links (see 'func_contact_matvec' in
    couplers/mpm_rigid_global_newton.py).
    """
    func_rows_matvec(
        newton_rows, newton_envs.is_pcg_active, newton_envs.p_dot_ap, qd.static(not newton_config.has_rigid_dofs)
    )
    n_rows = newton_rows.segments_start[newton_rows.segments_start.shape[0] - 1]
    if qd.static(newton_config.has_rigid_dofs):
        func_rigid_matvec(rigid_dofs, constraint_state, rigid_info, rigid_config, newton_config.has_rigid_rows)
        func_link_spatial_disp(rigid_dofs.p, rigid_links, dyn_state.dofs, dyn_info.links, rigid_config)
        func_contact_matvec(
            newton_rows,
            newton_envs.is_pcg_active,
            mpm_state,
            sparse_grid,
            rigid_links,
            dyn_state,
            mpm_info,
            dyn_info,
        )
        func_link_force_to_dofs(rigid_dofs.Ap, rigid_links, dyn_state.dofs, dyn_info.links, rigid_config)
        # p . Ap over the rows and over the dofs, once Ap holds the whole product
        qd.loop_config(block_dim=256)
        for i_r in range((n_rows + 31) // 32 * 32):
            i_b_key = gs.qd_int(-1)
            energy = gs.qd_float(0.0)
            if i_r < n_rows:
                i_b = newton_rows.rows_env[i_r]
                if newton_envs.is_pcg_active[i_b]:
                    i_b_key = i_b
                    energy = newton_rows.p[i_r].dot(newton_rows.Ap[i_r])
            func_add_by_key(i_b_key, energy, newton_envs.p_dot_ap)
        for i_d, i_b in qd.ndrange(rigid_dofs.p.shape[0], rigid_dofs.p.shape[1]):
            if newton_envs.is_pcg_active[i_b]:
                qd.atomic_add(newton_envs.p_dot_ap[i_b], rigid_dofs.p[i_d, i_b] * rigid_dofs.Ap[i_d, i_b])
    for i_b in range(newton_envs.is_active.shape[0]):
        if newton_envs.is_pcg_active[i_b]:
            p_dot_ap = newton_envs.p_dot_ap[i_b]
            pcg_alpha = gs.qd_float(0.0)
            if not qd_isfinite(p_dot_ap):
                newton_envs.is_pcg_active[i_b] = False
                newton_envs.failure[i_b] = NewtonFailure.NON_FINITE
            elif qd.abs(p_dot_ap) > 0.0:
                pcg_alpha = newton_envs.rz[i_b] / p_dot_ap
            newton_envs.pcg_alpha[i_b] = pcg_alpha
        newton_envs.rz_new[i_b] = 0.0
        newton_envs.p_dot_ap[i_b] = 0.0
    qd.loop_config(block_dim=256)
    for i_r in range((n_rows + 31) // 32 * 32):
        i_b_key = gs.qd_int(-1)
        rz = gs.qd_float(0.0)
        if i_r < n_rows:
            i_b = newton_rows.rows_env[i_r]
            if newton_envs.is_pcg_active[i_b]:
                pcg_alpha = newton_envs.pcg_alpha[i_b]
                r = newton_rows.r[i_r] - pcg_alpha * newton_rows.Ap[i_r]
                z = newton_rows.jacobi[i_r] @ r
                newton_rows.solution[i_r] = newton_rows.solution[i_r] + pcg_alpha * newton_rows.p[i_r]
                newton_rows.r[i_r] = r
                newton_rows.z[i_r] = z
                i_b_key = i_b
                rz = r.dot(z)
        func_add_by_key(i_b_key, rz, newton_envs.rz_new)
    if qd.static(newton_config.has_rigid_dofs):
        for i_d, i_b in qd.ndrange(rigid_dofs.p.shape[0], rigid_dofs.p.shape[1]):
            if newton_envs.is_pcg_active[i_b]:
                pcg_alpha = newton_envs.pcg_alpha[i_b]
                rigid_dofs.solution[i_d, i_b] = rigid_dofs.solution[i_d, i_b] + pcg_alpha * rigid_dofs.p[i_d, i_b]
                rigid_dofs.r[i_d, i_b] = rigid_dofs.r[i_d, i_b] - pcg_alpha * rigid_dofs.Ap[i_d, i_b]
        for i_b in range(rigid_dofs.p.shape[1]):
            if newton_envs.is_pcg_active[i_b]:
                func_rigid_solve_batch(
                    i_b,
                    rigid_dofs.r,
                    rigid_dofs.z,
                    dyn_state,
                    constraint_state,
                    rigid_info,
                    rigid_config,
                    newton_config.has_rigid_rows,
                )
        for i_d, i_b in qd.ndrange(rigid_dofs.p.shape[0], rigid_dofs.p.shape[1]):
            if newton_envs.is_pcg_active[i_b]:
                qd.atomic_add(newton_envs.rz_new[i_b], rigid_dofs.r[i_d, i_b] * rigid_dofs.z[i_d, i_b])
    for _ in range(1):
        newton_loops.n_pcg_iterations_left[None] = newton_loops.n_pcg_iterations_left[None] - 1
        newton_loops.is_pcg_running[None] = 0
    for i_b in range(newton_envs.is_active.shape[0]):
        if newton_envs.is_pcg_active[i_b]:
            rz = newton_envs.rz[i_b]
            rz_new = newton_envs.rz_new[i_b]
            newton_envs.pcg_beta[i_b] = rz_new / rz
            newton_envs.rz[i_b] = rz_new
            if not qd_isfinite(rz_new):
                newton_envs.is_pcg_active[i_b] = False
                newton_envs.failure[i_b] = NewtonFailure.NON_FINITE
            elif qd.abs(rz_new) <= newton_envs.rz_tol[i_b]:
                newton_envs.is_pcg_active[i_b] = False
            elif newton_loops.n_pcg_iterations_left[None] > 0:
                qd.atomic_max(newton_loops.is_pcg_running[None], 1)
    for i_r in range(n_rows):
        newton_rows.Ap[i_r] = qd.Vector.zero(gs.qd_float, 3)
        i_b = newton_rows.rows_env[i_r]
        if newton_envs.is_pcg_active[i_b]:
            newton_rows.p[i_r] = newton_rows.z[i_r] + newton_envs.pcg_beta[i_b] * newton_rows.p[i_r]
    if qd.static(newton_config.has_rigid_dofs):
        for i_d, i_b in qd.ndrange(rigid_dofs.p.shape[0], rigid_dofs.p.shape[1]):
            rigid_dofs.Ap[i_d, i_b] = 0.0
            if newton_envs.is_pcg_active[i_b]:
                rigid_dofs.p[i_d, i_b] = rigid_dofs.z[i_d, i_b] + newton_envs.pcg_beta[i_b] * rigid_dofs.p[i_d, i_b]


@qd.kernel(graph=True, fastcache=True)
def kernel_pcg_loop(
    newton_rows: NewtonRows,
    newton_envs: NewtonEnvs,
    newton_loops: NewtonLoops,
    mpm_state: MPMState,
    sparse_grid: MPMSparseGrid,
    fem_state: FEMState,
    ipc_pairs: IPCPairs,
    friction_pairs: IPCFrictionPairs,
    ipc_state: IPCState,
    rigid_dofs: NewtonRigidDofs,
    rigid_links: NewtonRigidLinks,
    dyn_state: DynState,
    constraint_state: ConstraintState,
    mpm_info: MPMInfo,
    fem_info: FEMInfo,
    ipc_info: IPCInfo,
    dyn_info: DynInfo,
    rigid_info: RigidInfo,
    mpm_config: qd.template(),
    fem_config: qd.template(),
    rigid_config: qd.template(),
    newton_config: qd.template(),
):
    """
    Run the iterations of the preconditioned conjugate gradient (PCG) as one graph, until every environment converges or
    the iterations run out.
    """
    while qd.graph.do_while(newton_loops.is_pcg_running):
        func_pcg_iteration(
            newton_rows,
            newton_envs,
            newton_loops,
            mpm_state,
            sparse_grid,
            rigid_dofs,
            rigid_links,
            dyn_state,
            constraint_state,
            mpm_info,
            dyn_info,
            rigid_info,
            rigid_config,
            newton_config,
        )


# ------------------------------------------------------------------------------------
# ----------------------------------- line search ------------------------------------
# ------------------------------------------------------------------------------------


@qd.kernel(fastcache=True)
def kernel_start_line_search(
    newton_rows: NewtonRows,
    newton_envs: NewtonEnvs,
    newton_loops: NewtonLoops,
    mpm_state: MPMState,
    sparse_grid: MPMSparseGrid,
    fem_state: FEMState,
    ipc_pairs: IPCPairs,
    friction_pairs: IPCFrictionPairs,
    ipc_state: IPCState,
    rigid_dofs: NewtonRigidDofs,
    rigid_links: NewtonRigidLinks,
    dyn_state: DynState,
    constraint_state: ConstraintState,
    newton_info: NewtonInfo,
    mpm_info: MPMInfo,
    fem_info: FEMInfo,
    ipc_info: IPCInfo,
    dyn_info: DynInfo,
    rigid_info: RigidInfo,
    mpm_config: qd.template(),
    fem_config: qd.template(),
    rigid_config: qd.template(),
    newton_config: qd.template(),
):
    """
    Set the Newton direction of every environment still iterating, minus the solution of the preconditioned conjugate
    gradient (PCG), and start its line search from the full step.

    An environment converges once its direction moves every particle, every vertex of the cloth and every dof by less
    than the tolerance, the step being taken still. The first step stays short of the first zero volume of a liquid
    particle (see 'func_mpm_init_line_search' in mpm_solver.py), and the contacts of the cloth bound it afterwards (see
    'kernel_contact_ccd' in ipc_contact.py).
    """
    for i_b in range(newton_envs.is_active.shape[0]):
        newton_envs.max_displacement[i_b] = 0.0
        newton_envs.alpha[i_b] = 1.0
        newton_envs.n_line_search_iterations[i_b] = 0
    if qd.static(newton_config.has_mpm):
        func_mpm_init_line_search(
            newton_rows,
            newton_envs.is_active,
            newton_envs.max_displacement,
            newton_envs.alpha,
            mpm_state,
            sparse_grid,
            mpm_info,
        )
    if qd.static(newton_config.has_cloth):
        func_cloth_init_line_search(
            newton_rows,
            newton_envs.is_active,
            newton_envs.max_displacement,
            fem_state,
            ipc_state,
            newton_config.has_rigid_vertices,
        )
    if qd.static(newton_config.has_rigid_dofs):
        func_rigid_init_line_search(rigid_dofs, newton_envs.is_active, newton_envs.max_displacement)
    for _ in range(1):
        newton_loops.is_line_search_running[None] = 0
    for i_b in range(newton_envs.is_active.shape[0]):
        is_active = newton_envs.is_active[i_b]
        if is_active:
            max_displacement = newton_envs.max_displacement[i_b]
            newton_envs.is_converged[i_b] = max_displacement < newton_info.displacement_tol[None]
            if not (qd_isfinite(max_displacement) and qd_isfinite(newton_envs.energy_start[i_b])):
                newton_envs.failure[i_b] = NewtonFailure.NON_FINITE
            qd.atomic_max(newton_loops.is_line_search_running[None], 1)
        newton_envs.is_line_search_active[i_b] = is_active


@qd.kernel(fastcache=True)
def kernel_line_search_trial(
    newton_envs: NewtonEnvs,
    newton_loops: NewtonLoops,
    mpm_state: MPMState,
    sparse_grid: MPMSparseGrid,
    fem_state: FEMState,
    ipc_pairs: IPCPairs,
    friction_pairs: IPCFrictionPairs,
    ipc_state: IPCState,
    rigid_dofs: NewtonRigidDofs,
    rigid_links: NewtonRigidLinks,
    dyn_state: DynState,
    constraint_state: ConstraintState,
    newton_info: NewtonInfo,
    mpm_info: MPMInfo,
    fem_info: FEMInfo,
    ipc_info: IPCInfo,
    dyn_info: DynInfo,
    rigid_info: RigidInfo,
    mpm_config: qd.template(),
    fem_config: qd.template(),
    rigid_config: qd.template(),
    newton_config: qd.template(),
):
    """
    Move the nodes, the vertices of the cloth and the dofs of every environment still searching to the trial point of
    its step, and accept the step once the energy there is at most the energy at the start point, or shrink it.

    A converged or failed environment takes its step at once, and an environment out of trials keeps its last one.
    """
    if qd.static(newton_config.has_mpm):
        func_mpm_step_forward(sparse_grid, newton_envs.is_line_search_active, newton_envs.alpha)
    if qd.static(newton_config.has_cloth):
        func_cloth_step_forward(
            newton_envs.is_line_search_active, newton_envs.alpha, fem_state, ipc_state, newton_config.has_rigid_vertices
        )
    if qd.static(newton_config.has_rigid_dofs):
        func_rigid_step_forward(rigid_dofs, newton_envs.is_line_search_active, newton_envs.alpha)
    for i_b in range(newton_envs.is_active.shape[0]):
        newton_envs.energy_trial[i_b] = 0.0
    func_add_energies(
        newton_envs.energy_trial,
        newton_envs.is_line_search_active,
        newton_envs.rigid_cost,
        mpm_state,
        sparse_grid,
        fem_state,
        ipc_pairs,
        friction_pairs,
        ipc_state,
        rigid_dofs,
        rigid_links,
        dyn_state,
        constraint_state,
        mpm_info,
        fem_info,
        ipc_info,
        dyn_info,
        rigid_info,
        mpm_config,
        fem_config,
        rigid_config,
        newton_config,
    )
    for _ in range(1):
        newton_loops.is_line_search_running[None] = 0
    for i_b in range(newton_envs.is_active.shape[0]):
        if newton_envs.is_line_search_active[i_b]:
            newton_envs.n_line_search_iterations[i_b] = newton_envs.n_line_search_iterations[i_b] + 1
            is_accepted = newton_envs.is_converged[i_b] or newton_envs.failure[i_b] != NewtonFailure.NONE
            if newton_envs.energy_trial[i_b] <= newton_envs.energy_start[i_b]:
                is_accepted = True
            if newton_envs.n_line_search_iterations[i_b] >= newton_info.line_search_max_iterations[None]:
                is_accepted = True
            if is_accepted:
                newton_envs.is_line_search_active[i_b] = False
                newton_envs.energy_start[i_b] = newton_envs.energy_trial[i_b]
            else:
                newton_envs.alpha[i_b] = newton_info.line_search_shrink[None] * newton_envs.alpha[i_b]
                qd.atomic_max(newton_loops.is_line_search_running[None], 1)


@qd.kernel(fastcache=True)
def kernel_end_iteration(newton_envs: NewtonEnvs, newton_loops: NewtonLoops, newton_info: NewtonInfo):
    """Stop every environment that converged or failed, or reached the iteration cap, which it latches as a failure."""
    for _ in range(1):
        newton_loops.is_newton_running[None] = 0
    for i_b in range(newton_envs.is_active.shape[0]):
        if newton_envs.is_active[i_b]:
            newton_envs.n_iterations[i_b] = newton_envs.n_iterations[i_b] + 1
            if newton_envs.is_converged[i_b] or newton_envs.failure[i_b] != NewtonFailure.NONE:
                newton_envs.is_active[i_b] = False
            elif newton_envs.n_iterations[i_b] >= newton_info.max_iterations[None]:
                newton_envs.is_active[i_b] = False
                newton_envs.failure[i_b] = NewtonFailure.ITERATIONS
                qd.atomic_max(newton_loops.has_failed_iterations[None], 1)
            else:
                qd.atomic_max(newton_loops.is_newton_running[None], 1)


@qd.kernel(fastcache=True)
def kernel_finish_rigid(
    newton_envs: NewtonEnvs,
    mpm_state: MPMState,
    sparse_grid: MPMSparseGrid,
    fem_state: FEMState,
    ipc_pairs: IPCPairs,
    friction_pairs: IPCFrictionPairs,
    ipc_state: IPCState,
    rigid_dofs: NewtonRigidDofs,
    rigid_links: NewtonRigidLinks,
    dyn_state: DynState,
    constraint_state: ConstraintState,
    mpm_info: MPMInfo,
    fem_info: FEMInfo,
    ipc_info: IPCInfo,
    dyn_info: DynInfo,
    rigid_info: RigidInfo,
    mpm_config: qd.template(),
    fem_config: qd.template(),
    rigid_config: qd.template(),
    newton_config: qd.template(),
    errno: qd.Tensor,
):
    """
    Hand the accelerations of the dofs solved to the rigid solver (see 'func_rigid_finish' in rigid/global_newton.py).

    An environment whose Newton solve failed keeps the state of the rigid solver until the error raises (see
    'RigidInfo.is_env_frozen').
    """
    for i_b in range(rigid_info.is_env_frozen.shape[0]):
        rigid_info.is_env_frozen[i_b] = (errno[i_b] & NewtonErrorCode.INVALID_SOLVE_NAN) != 0
    func_rigid_finish(
        newton_envs.rigid_cost,
        rigid_dofs,
        dyn_state,
        constraint_state,
        rigid_info,
        rigid_config,
        newton_config.has_rigid_rows,
    )


@qd.kernel(fastcache=True)
def kernel_flag_failures(newton_envs: NewtonEnvs, newton_loops: NewtonLoops, errno: qd.Tensor):
    """Flag the environments whose Newton solve met a value that is not finite, which keep the state of their particles,
    of their cloth and of the rigid solver until the error raises (see 'kernel_finish_rigid', 'kernel_mpm_g2p' in
    mpm_solver.py and 'kernel_cloth_finish' in fem_solver.py)."""
    for i_b in range(newton_envs.is_active.shape[0]):
        if newton_envs.failure[i_b] == NewtonFailure.NON_FINITE:
            qd.atomic_or(errno[i_b], NewtonErrorCode.INVALID_SOLVE_NAN)


@qd.kernel(fastcache=True)
def kernel_clear_failed_iterations(newton_loops: NewtonLoops):
    for _ in range(1):
        newton_loops.has_failed_iterations[None] = 0

# pylint: disable=no-value-for-parameter

import numpy as np
import torch

import igl

import quadrants as qd

import genesis as gs
import genesis.utils.simt as su
from genesis.engine.entities.fem_entity import FEMEntity
from genesis.engine.materials import FEM
from genesis.engine.state_change import StateChange, mutates
from genesis.engine.states.solvers import FEMSolverState
from genesis.grad.bridge import custom_backward, no_grad
from genesis.utils.array_class import ErrorCode, LinksState
from genesis.utils.data import ArrayItem, DataKind
from genesis.utils.geom import qd_inv_quat, qd_transform_by_quat, qd_transform_quat_by_quat
from genesis.utils.misc import broadcast_tensor, get_gpu_core_count, qd_isfinite, qd_to_numpy, qd_to_torch, qd_zero_grad

from .base_solver import GravityMixin, Solver, TimeBasedMixin

# Lanes of a GPU subgroup tile that gather together the element corners of a vertex (see 'vertices_corners_idx')
LOG2_CORNER_LANES = 2
# Threads of the GPU block that sums the terms of an environment over its vertices and elements in a fixed order
ENV_BLOCK_DIM = 256


@qd.data_oriented
class FEMSolver(GravityMixin, TimeBasedMixin, Solver):
    material_cls = FEM.Base
    is_differentiable = True
    has_frames = True

    # ------------------------------------------------------------------------------------
    # --------------------------------- Initialization -----------------------------------
    # ------------------------------------------------------------------------------------

    def __init__(self, scene, sim, options):
        super().__init__(scene, sim, options)

        # options
        self._damping = options.damping
        self._use_implicit_solver = options.use_implicit_solver
        self._n_newton_iterations = options.n_newton_iterations
        self._newton_dx_threshold = options.newton_dx_threshold
        self._n_pcg_iterations = options.n_pcg_iterations
        self._pcg_threshold = options.pcg_threshold
        self._n_linesearch_iterations = options.n_linesearch_iterations
        self._linesearch_c = options.linesearch_c
        self._linesearch_tau = options.linesearch_tau
        self._damping_alpha = options.damping_alpha
        self._damping_beta = options.damping_beta
        self._enable_vertex_constraints = options.enable_vertex_constraints

        # The implicit solve adds up every term in a fixed order, so it is reproducible bitwise. On CPU, one thread runs
        # each environment, the environments filling the cores. On GPU, the lanes of a subgroup tile gather the terms
        # of a vertex and a block of threads sums those of an environment.
        self._env_block_dim = 1 if gs.backend == gs.cpu else ENV_BLOCK_DIM
        self._is_pcg_solved_per_env_block = False

        # use scaled volume for better numerical stability, similar to p_vol_scale in mpm
        self._vol_scale = float(1e4)

        # materials
        self._mats = list()

        # lazy initialization
        self._constraints_initialized = False

    def init_batch_fields(self):
        self.batch_active = qd.field(dtype=gs.qd_bool, shape=(self._B,), needs_grad=False)
        self.batch_pcg_active = qd.field(dtype=gs.qd_bool, shape=(self._B,), needs_grad=False)
        self.batch_linesearch_active = qd.field(dtype=gs.qd_bool, shape=(self._B,), needs_grad=False)

        # The errors of each env, raised at the beginning of the next step (see 'Solver.register_errors')
        self.errno = qd.field(gs.qd_int, shape=(self._B,))
        self.register_errors(
            self.errno,
            {
                ErrorCode.INVALID_VERTEX_NAN: (
                    "NaN detected in FEM states. Try reducing the time step size or adjusting simulation parameters."
                ),
            },
        )

        pcg_state = qd.types.struct(
            rTr=gs.qd_float,
            rTz=gs.qd_float,
            rTr_new=gs.qd_float,
            rTz_new=gs.qd_float,
            pTAp=gs.qd_float,
            alpha=gs.qd_float,
            beta=gs.qd_float,
            # The squared residual below which the PCG solve of a differentiable scene stops (see 'pcg_threshold_rel')
            rTr_threshold=gs.qd_float,
        )
        self.pcg_state = pcg_state.field(shape=(self._B,), needs_grad=False, layout=qd.Layout.SOA)

        linesearch_state = qd.types.struct(
            prev_energy=gs.qd_float,
            energy=gs.qd_float,
            step_size=gs.qd_float,
            m=gs.qd_float,
        )
        self.linesearch_state = linesearch_state.field(shape=(self._B,), needs_grad=False, layout=qd.Layout.SOA)

    def init_element_fields(self):
        # element state in vertices
        element_state_v = qd.types.struct(
            pos=gs.qd_vec3,  # position
            vel=gs.qd_vec3,  # velocity
        )

        # element state in elements
        element_state_el = qd.types.struct(
            actu=gs.qd_float,  # actuation
        )

        # element info (properties that remain static through time)
        element_info = qd.types.struct(
            el2v=gs.qd_ivec4,  # vertex index of an element
            mu=gs.qd_float,  # lame parameters (1)
            lam=gs.qd_float,  # lame parameters (2)
            mass_scaled=gs.qd_float,  # scaled element mass. The real mass is mass_scaled / self._vol_scale
            mat_idx=gs.qd_int,  # material model index
            B=gs.qd_mat3,  # inverse of the deformation gradient at rest state
            V=gs.qd_float,  # rest volume of the element
            friction_mu=gs.qd_float,  # friction coefficient for contact
            # for muscle
            muscle_group=gs.qd_int,
            muscle_direction=gs.qd_vec3,
        )

        # element state for energy
        element_state_el_energy = qd.types.struct(
            energy=gs.qd_float,  # energy density for the element
            # gradient density for the element, del energy / del F, plus the stress of the Rayleigh stiffness damping
            # under the implicit solver.
            gradient=gs.qd_mat3,
        )

        element_state_v_energy = qd.types.struct(
            inertia=gs.qd_vec3,  # inertia for the vertex
            force=gs.qd_vec3,
        )

        element_v_info = qd.types.struct(
            mass=gs.qd_float,  # mass of the vertex
            mass_inv=gs.qd_float,  # inverse mass of the vertex
            mass_over_dt2=gs.qd_float,  # scaled mass of the vertex over dt^2
            friction_mu=gs.qd_float,  # friction coefficient for contact
        )

        pcg_state_v = qd.types.struct(
            diag3x3=gs.qd_mat3,  # diagonal 3-by-3 block of the hessian
            prec=gs.qd_mat3,  # preconditioner
            x=gs.qd_vec3,  # solution vector
            r=gs.qd_vec3,  # residual vector
            z=gs.qd_vec3,  # preconditioned residual vector
            p=gs.qd_vec3,  # search direction vector
            Ap=gs.qd_vec3,  # matrix-vector product
        )

        linesearch_state_v = qd.types.struct(
            x_prev=gs.qd_vec3,  # solution vector
        )

        # construct field
        self.elements_v = element_state_v.field(
            shape=(self.n_frames, self.n_vertices, self._B), needs_grad=True, layout=qd.Layout.SOA
        )
        self.elements_el = element_state_el.field(
            shape=(self.n_frames, self.n_elements, self._B), needs_grad=True, layout=qd.Layout.SOA
        )
        self.elements_i = element_info.field(
            shape=(self.n_elements),
            needs_grad=False,
            layout=qd.Layout.SOA,
        )

        self.elements_el_energy = element_state_el_energy.field(
            shape=(self._B, self.n_elements),
            needs_grad=False,
            layout=qd.Layout.SOA,
        )

        # The adjoint of the forces holds the solution of the adjoint solve (see '_implicit_solve_grad')
        self.elements_v_energy = element_state_v_energy.field(
            shape=(self._B, self.n_vertices),
            needs_grad=self._requires_grad and self._use_implicit_solver,
            layout=qd.Layout.SOA,
        )

        self.elements_v_info = element_v_info.field(
            shape=(self.n_vertices),
            needs_grad=False,
            layout=qd.Layout.SOA,
        )

        self.pcg_state_v = pcg_state_v.field(
            shape=(self._B, self.n_vertices),
            needs_grad=False,
            layout=qd.Layout.SOA,
        )

        self.linesearch_state_v = linesearch_state_v.field(
            shape=(self._B, self.n_vertices),
            needs_grad=False,
            layout=qd.Layout.SOA,
        )

    def init_surface_fields(self):
        n_surfaces_max = self.n_surfaces

        # surface info (for coupling)
        surface_state = qd.types.struct(
            tri2v=gs.qd_ivec3,  # vertex index of a triangle
            tri2el=gs.qd_int,  # element index of a triangle
        )

        self.surface = surface_state.field(
            shape=(n_surfaces_max),
            needs_grad=False,
            layout=qd.Layout.SOA,
        )

    def init_vvert_fields(self):
        """Allocate the render geometry of every visual geom of every entity, laid out back-to-back.

        Several vverts may stand for a single simulated vertex, so each one carries its own UVs and gathers its
        position through 'vert_idx' (see 'FEMVisGeom'). A contiguous layout lets a renderer consume positions, UVs and
        topology as three flat arrays.
        """
        struct_vvert_info = qd.types.struct(
            vert_idx=gs.qd_int,  # simulated vertex standing for this vvert
        )
        self.vverts_info = struct_vvert_info.field(shape=(max(self._n_vverts, 1),), layout=qd.Layout.SOA)

        # environment-offset vvert positions
        struct_vvert_state_render = qd.types.struct(
            pos=gs.qd_vec3,
        )
        self.vverts_render = struct_vvert_state_render.field(
            shape=(max(self._n_vverts, 1), self._B), layout=qd.Layout.SOA
        )

        # static, shared across all batch envs
        self.vverts_uvs = qd.field(dtype=gs.qd_vec2, shape=(max(self._n_vverts, 1),))

        # static, in the solver's global vvert space
        self.vfaces_indices = qd.field(dtype=gs.qd_ivec3, shape=(max(self._n_vfaces, 1),))

    def _init_surface_info(self):
        self.vertices_on_surface = qd.field(dtype=gs.qd_bool, shape=(self.n_vertices,))
        self.elements_on_surface = qd.field(dtype=gs.qd_bool, shape=(self.n_elements,))
        self.compute_surface_vertices()
        self.compute_surface_elements()
        vertices_on_surface_np = self.vertices_on_surface.to_numpy()
        elements_on_surface_np = self.elements_on_surface.to_numpy()
        (surface_vertices_np,) = vertices_on_surface_np.nonzero()
        self.surface_vertices = qd.field(
            dtype=qd.i32,
            shape=(len(surface_vertices_np),),
            needs_grad=False,
        )
        self.surface_vertices.from_numpy(surface_vertices_np.astype(np.int32, copy=False))
        (surface_elements_np,) = elements_on_surface_np.nonzero()
        self.surface_elements = qd.field(
            dtype=qd.i32,
            shape=(len(surface_elements_np),),
            needs_grad=False,
        )
        self.surface_elements.from_numpy(surface_elements_np.astype(np.int32, copy=False))

        surface_triangles_np = self.surface.tri2v.to_numpy()
        pos_np = self.elements_v.pos.to_numpy()[0, :, 0, :][surface_vertices_np]
        surface_vertices_mapping = np.full(self.n_vertices, -1, dtype=np.int32)
        surface_vertices_mapping[surface_vertices_np] = np.arange(len(surface_vertices_np))
        surface_tris_sv_idx = surface_vertices_mapping[surface_triangles_np]
        mass = igl.massmatrix(pos_np, surface_tris_sv_idx)
        surface_vert_mass_np = mass.diagonal().astype(gs.np_float, copy=False)
        self.surface_vert_mass = qd.field(
            dtype=gs.qd_float,
            shape=(len(surface_vertices_np),),
            needs_grad=False,
        )
        self.surface_vert_mass.from_numpy(surface_vert_mass_np)

        # The incident triangles of surface vertex i_sv are surface_vertices_tri_idx[surface_vertices_tri_start[i_sv]:
        # surface_vertices_tri_start[i_sv + 1]], in increasing triangle index (stable sort of the triangle corners).
        corners_order = np.argsort(surface_tris_sv_idx, axis=None, kind="stable")
        self.surface_vertices_tri_start = qd.field(dtype=gs.qd_int, shape=(len(surface_vertices_np) + 1,))
        self.surface_vertices_tri_start.from_numpy(
            np.searchsorted(surface_tris_sv_idx.take(corners_order), np.arange(len(surface_vertices_np) + 1))
        )
        corners_tri_idx = corners_order // 3
        self.surface_vertices_tri_idx = qd.field(dtype=gs.qd_int, shape=(len(corners_order),))
        self.surface_vertices_tri_idx.from_numpy(corners_tri_idx)
        # The scaled element mass of each incident triangle, gathered once so that the coupling kernel reads it without
        # going through the triangle and its element.
        self.surface_vertices_tri_mass_scaled = qd.field(dtype=gs.qd_float, shape=(len(corners_order),))
        self.surface_vertices_tri_mass_scaled.from_numpy(
            qd_to_numpy(self.elements_i.mass_scaled)[qd_to_numpy(self.surface.tri2el)[corners_tri_idx]]
        )

    @qd.kernel
    def compute_surface_vertices(self):
        for i_v in range(self.n_vertices):
            self.vertices_on_surface[i_v] = False

        for i_s in range(self.n_surfaces):
            tri2v = self.surface[i_s].tri2v
            for i in qd.static(range(3)):
                self.vertices_on_surface[tri2v[i]] = True

    @qd.kernel
    def compute_surface_elements(self):
        for i_e in range(self.n_elements):
            i_v = self.elements_i[i_e].el2v
            self.elements_on_surface[i_e] = (
                self.vertices_on_surface[i_v[0]]
                or self.vertices_on_surface[i_v[1]]
                or self.vertices_on_surface[i_v[2]]
                or self.vertices_on_surface[i_v[3]]
            )

    def init_constraints(self):
        self._constraints_initialized = True

        vertex_constraint_info = qd.types.struct(
            is_constrained=gs.qd_bool,  # boolean flag indicating if vertex is constrained
            target_pos=gs.qd_vec3,  # target position for the constraint
            is_soft_constraint=gs.qd_bool,  # use spring for soft constraints
            stiffness=gs.qd_float,  # spring stiffness
            link_idx=gs.qd_int,  # index of the rigid link (-1 if not linked)
            link_offset_pos=gs.qd_vec3,  # offset position of link
            link_init_quat=gs.qd_vec4,  # offset rotation of link
        )

        # FIXME: AOS, which does not match other Genesis structs. Old, untested code. We prefer not to touch for now.
        self.vertex_constraints = vertex_constraint_info.field(
            shape=(self.n_vertices, self._B), needs_grad=False, layout=qd.Layout.AOS
        )

        self.vertex_constraints.is_constrained.fill(False)
        self.vertex_constraints.link_idx.fill(-1)

    def reset_grad(self):
        if self._requires_grad:
            qd_zero_grad(self.elements_v)
            qd_zero_grad(self.elements_el)

    def build(self):
        super().build()

        self.n_envs = self.sim.n_envs
        self._B = self.sim._B

        # elements and bodies
        self._n_elements_max = self.n_elements
        self._n_vertices_max = self.n_vertices
        self._n_vverts = self.n_vverts
        self._n_vfaces = self.n_vfaces

        if self.n_elements_max > 0:
            self.tet_wrong_order = qd.field(dtype=gs.qd_bool, shape=(), needs_grad=False)

            # batch fields
            self.init_batch_fields()

            self.init_element_fields()
            self.init_surface_fields()
            self.init_vvert_fields()

            for entity in self._entities:
                entity._add_to_solver()

        for mat in self._mats:
            mat.build(self)

        if self.n_elements_max > 0:
            self._init_surface_info()
            if self._env_block_dim > 1:
                # The corners of the elements, vertex by vertex then element by element, each one packed as 4 times its
                # element plus its local index. A vertex gathers the terms of its corners in this order, so the
                # implicit solve and the internal force getter sum them without atomics.
                el2v = qd_to_numpy(self.elements_i.el2v)
                self.vertices_corners_start = qd.field(dtype=gs.qd_int, shape=(self.n_vertices + 1,))
                self.vertices_corners_start.from_numpy(
                    np.searchsorted(np.sort(el2v, axis=None), np.arange(self.n_vertices + 1))
                )
                self.vertices_corners_idx = qd.field(dtype=gs.qd_int, shape=(4 * self.n_elements,))
                self.vertices_corners_idx.from_numpy(np.argsort(el2v, axis=None, kind="stable"))
            if self._use_implicit_solver:
                # One block of threads runs the whole PCG solve of an environment once the blocks of all environments
                # occupy every core of the device: each environment then keeps its data close to its block and steps
                # without kernel boundaries. With fewer environments, the device-side loop spreads each environment
                # over the whole device instead.
                self._is_pcg_solved_per_env_block = (
                    self._env_block_dim > 1 and self._B * ENV_BLOCK_DIM >= get_gpu_core_count()
                )

                # The iteration budgets and thresholds of the implicit solve, read by its kernels
                solve_info = qd.types.struct(
                    n_newton_iterations=gs.qd_int,
                    n_pcg_iterations=gs.qd_int,
                    n_linesearch_iterations=gs.qd_int,
                    pcg_threshold=gs.qd_float,
                    pcg_threshold_rel=gs.qd_float,
                    linesearch_c=gs.qd_float,
                    linesearch_tau=gs.qd_float,
                )
                self.solve_info = solve_info.field(shape=(), needs_grad=False)
                self.solve_info[None].n_newton_iterations = self._n_newton_iterations
                self.solve_info[None].n_pcg_iterations = self._n_pcg_iterations
                self.solve_info[None].n_linesearch_iterations = self._n_linesearch_iterations
                self.solve_info[None].pcg_threshold = self._pcg_threshold
                # A differentiable scene solves to round-off, relative to the first residual of the substep, since its
                # backward pass is the adjoint of the root of the solve, and a gradient scales with its loss whatever
                # the scale. The later Newton iterations of the substep then start at their root.
                self.solve_info[None].pcg_threshold_rel = (1e-14 if gs.np_float == np.float64 else 1e-6) ** 2
                self.solve_info[None].linesearch_c = self._linesearch_c
                self.solve_info[None].linesearch_tau = self._linesearch_tau

                # The iterations left to each loop of the implicit solve, and the flags that keep the loops running on
                # the device as long as an environment iterates on (see '_kernel_batch_solve').
                solve_iterations_left = qd.types.struct(newton=gs.qd_int, pcg=gs.qd_int, linesearch=gs.qd_int)
                self.solve_iterations_left = solve_iterations_left.field(shape=(), needs_grad=False)
                self._newton_counter = qd.ndarray(qd.i32, shape=())
                self._pcg_counter = qd.ndarray(qd.i32, shape=())
                self._linesearch_counter = qd.ndarray(qd.i32, shape=())
            if self.tet_wrong_order[None]:
                raise RuntimeError(
                    "The order of vertices in the tetrahedral elements is not correct. "
                    "Please check the input mesh or the FEM solver implementation."
                )

        if self.n_vertices_max > 0 and self._enable_vertex_constraints and not self._constraints_initialized:
            self.init_constraints()

        # Kernels of this solver take the solver itself, so gravity has to be a field for them.
        self._build_gravity()

    @property
    def is_active(self):
        # Counted from the entities rather than from what build recorded, so the answer stands before it: the rate
        # every solver integrates at is settled from the active ones, and that is settled before any of them builds.
        return self.n_elements > 0

    def add_entity(
        self, idx, material, morph, surface, visualize_contact=False, name: str | None = None, desc=None
    ) -> "FEMEntity":
        # The kernels dispatch on the index of the material among the distinct ones, so each one is registered and built
        # once, whatever the number of entities sharing it.
        for mat in self._mats:
            if material == mat:
                material.idx = mat.idx
                break
        else:
            material.idx = len(self._mats)
            self._mats.append(material)

        # create entity
        entity = FEMEntity(
            scene=self._scene,
            solver=self,
            material=material,
            morph=morph,
            surface=surface,
            idx=idx,
            v_start=self.n_vertices,
            el_start=self.n_elements,
            s_start=self.n_surfaces,
            vvert_start=self.n_vverts,
            vface_start=self.n_vfaces,
            name=name,
        )

        self._entities.append(entity)
        return entity

    # ------------------------------------------------------------------------------------
    # ----------------------------------- simulation -------------------------------------
    # ------------------------------------------------------------------------------------

    @qd.kernel
    def init_pos_and_vel(self, f: qd.i32):
        for i_v, i_b in qd.ndrange(self.n_vertices, self._B):
            self.elements_v[f + 1, i_v, i_b].pos = self.elements_v[f, i_v, i_b].pos
            self.elements_v[f + 1, i_v, i_b].vel = self.elements_v[f, i_v, i_b].vel

    @qd.func
    def _func_element_internal_force(self, f: int, i_e: int, i_b: int):
        # The elastic force of the element on each of its vertices, one per column, from the stress its solver uses. The
        # columns sum to zero, so the forces conserve momentum once each vertex divides its own by its own mass.
        J, F = self._compute_ele_J_F(f, i_e, i_b)
        mu = self.elements_i[i_e].mu
        lam = self.elements_i[i_e].lam
        actu = self.elements_el[f, i_e, i_b].actu
        m_dir = self.elements_i[i_e].muscle_direction
        stress = qd.Matrix.zero(gs.qd_float, 3, 3)
        for mat_idx in qd.static(range(len(self._mats))):
            if self.elements_i[i_e].mat_idx == mat_idx:
                if qd.static(self._use_implicit_solver):
                    _, stress = self._mats[mat_idx].compute_energy_gradient(i_e, i_b, mu, lam, J, F, actu, m_dir)
                else:
                    stress = self._mats[mat_idx].update_stress(mu, lam, J, F, actu, m_dir)
        H = -self.elements_i[i_e].V * stress @ self.elements_i[i_e].B.transpose()
        return qd.Matrix.cols([H[:, 0], H[:, 1], H[:, 2], -H[:, 0] - H[:, 1] - H[:, 2]])

    @qd.kernel
    def compute_vel(self, f: qd.i32):
        for i_e, i_b in qd.ndrange(self.n_elements, self._B):
            force = self._func_element_internal_force(f, i_e, i_b)
            i_vs = self.elements_i[i_e].el2v
            for i in qd.static(range(4)):
                i_v = i_vs[i]
                dv = self.substep_dt * self.elements_v_info[i_v].mass_inv * force[:, i]
                qd.atomic_add(self.elements_v[f + 1, i_v, i_b].vel, dv)

    @qd.kernel
    def apply_uniform_force(self, f: qd.i32):
        for i_v, i_b in qd.ndrange(self.n_vertices, self._B):
            # NOTE: damping should only be applied to velocity from internal force and thus come first here
            #       given the immediate previous function call is compute_internal_vel --> however, shouldn't
            #       be done at dv only and need to wait for all elements updated (cannot be in the compute_internal_vel kernel)
            #       however, this inevitably damp the gravity.
            self.elements_v[f + 1, i_v, i_b].vel *= qd.exp(-self.substep_dt * self.damping)
            # Add gravity (avoiding damping on gravity)
            self.elements_v[f + 1, i_v, i_b].vel += self.substep_dt * self._gravity[i_b]

    @qd.kernel
    def compute_pos(self, f: qd.i32):
        # Every solver and coupler path ends a substep here, so the positions it integrates are checked once. The hard
        # constraints overwrite some of them afterward, and check their targets (see 'apply_hard_constraints').
        for i_v, i_b in qd.ndrange(self.n_vertices, self._B):
            pos = self.substep_dt * self.elements_v[f + 1, i_v, i_b].vel + self.elements_v[f, i_v, i_b].pos
            self.elements_v[f + 1, i_v, i_b].pos = pos
            if not qd_isfinite(pos):
                self.errno[i_b] = self.errno[i_b] | ErrorCode.INVALID_VERTEX_NAN
        # With gradients, the substeps of a step run in successive frames, to which the actuation set in frame 0 carries
        # over
        if qd.static(self._requires_grad):
            for i_e, i_b in qd.ndrange(self.n_elements, self._B):
                self.elements_el[f + 1, i_e, i_b].actu = self.elements_el[f, i_e, i_b].actu

    # The backward pass of the solve reverses the data along with the forces it enters (see '_implicit_solve_grad')
    @no_grad
    @qd.kernel
    def precompute_material_data(self, f: qd.i32):
        for i_b, i_e in qd.ndrange(self._B, self.n_elements):
            J, F = self._compute_ele_J_F(f, i_e, i_b)  # use last time step's pos to compute
            for mat_idx in qd.static(range(len(self._mats))):
                if self.elements_i[i_e].mat_idx == mat_idx:
                    self._mats[mat_idx].pre_compute(i_e, i_b, J, F, self.elements_i[i_e].muscle_direction)

    @qd.kernel
    def init_pos_and_inertia(self, f: qd.i32, links_state: LinksState):
        # Every environment enters the Newton iterations of the substep that follows
        for i_b in range(self._B):
            self.batch_active[i_b] = True
        for i_v, i_b in qd.ndrange(self.n_vertices, self._B):
            dt2 = self.substep_dt**2
            if qd.static(self._enable_vertex_constraints):
                if self.vertex_constraints.is_constrained[i_v, i_b]:
                    target_pos = self._func_get_constraint_target(i_v, i_b, links_state)
                    self.elements_v[f + 1, i_v, i_b].pos = target_pos
                    self.elements_v_energy[i_b, i_v].inertia = target_pos
                else:
                    self.elements_v_energy[i_b, i_v].inertia = (
                        self.elements_v[f, i_v, i_b].pos
                        + self.elements_v[f, i_v, i_b].vel * self.substep_dt
                        + self._gravity[i_b] * dt2
                    )
                    self.elements_v[f + 1, i_v, i_b].pos = self.elements_v[f, i_v, i_b].pos
            else:
                self.elements_v_energy[i_b, i_v].inertia = (
                    self.elements_v[f, i_v, i_b].pos
                    + self.elements_v[f, i_v, i_b].vel * self.substep_dt
                    + self._gravity[i_b] * dt2
                )
                self.elements_v[f + 1, i_v, i_b].pos = self.elements_v[f, i_v, i_b].pos

    @qd.func
    def _compute_ele_J_F(self, f: qd.i32, i_e: qd.i32, i_b: qd.i32):
        """
        Compute the determinant (J) and deformation gradient (F) for an element.
        """
        i_v0, i_v1, i_v2, i_v3 = self.elements_i[i_e].el2v
        pos_v0 = self.elements_v[f, i_v0, i_b].pos
        pos_v1 = self.elements_v[f, i_v1, i_b].pos
        pos_v2 = self.elements_v[f, i_v2, i_b].pos
        pos_v3 = self.elements_v[f, i_v3, i_b].pos
        D = qd.Matrix.cols([pos_v0 - pos_v3, pos_v1 - pos_v3, pos_v2 - pos_v3])

        B = self.elements_i[i_e].B
        F = D @ B
        J = F.determinant()

        return J, F

    @qd.func
    def _func_element_energy_gradient(self, f: int, i_e: int, i_b: int):
        # The energy density of an element at the end of a substep of the implicit solver, and its gradient with respect
        # to the deformation gradient plus the stress of the Rayleigh stiffness damping for the displacement over it
        J, F = self._compute_ele_J_F(f + 1, i_e, i_b)
        mu = self.elements_i[i_e].mu
        lam = self.elements_i[i_e].lam
        actu = self.elements_el[f, i_e, i_b].actu
        m_dir = self.elements_i[i_e].muscle_direction
        energy = gs.qd_float(0.0)
        gradient = qd.Matrix.zero(gs.qd_float, 3, 3)
        for mat_idx in qd.static(range(len(self._mats))):
            if self.elements_i[i_e].mat_idx == mat_idx:
                energy, gradient = self._mats[mat_idx].compute_energy_gradient(i_e, i_b, mu, lam, J, F, actu, m_dir)
        if qd.static(self._damping_beta > gs.EPS):
            damping_beta_over_dt = self._damping_beta / self._substep_dt
            i_vs = self.elements_i[i_e].el2v
            S = self._func_compute_element_mapping_matrix(i_vs, self.elements_i[i_e].B, i_b)
            _, dP = self.compute_elastic_product(i_e, i_b, S, self._func_element_displacement(f, i_e, i_b))
            gradient = gradient + damping_beta_over_dt * dP
        return energy, gradient

    @qd.func
    def _func_element_displacement(self, f: int, i_e: int, i_b: int):
        # The displacement of each vertex of an element over a substep, one per column
        i_vs = self.elements_i[i_e].el2v
        return qd.Matrix.cols(
            [
                self.elements_v[f + 1, i_vs[i_k], i_b].pos - self.elements_v[f, i_vs[i_k], i_b].pos
                for i_k in qd.static(range(4))
            ]
        )

    @qd.func
    def _func_compute_ele_energy_gradient(self, f: int):
        for i_b, i_e in qd.ndrange(self._B, self.n_elements):
            if self.batch_active[i_b]:
                energy, gradient = self._func_element_energy_gradient(f, i_e, i_b)
                self.elements_el_energy[i_b, i_e].energy = energy
                self.elements_el_energy[i_b, i_e].gradient = gradient

    @qd.func
    def _func_vertex_inertia_force(self, f: int, i_v: int, i_b: int):
        # The force of the inertia and of the Rayleigh mass damping on a vertex at the end of a substep of the implicit
        # solver. A constrained vertex sits at its target, which the damping of its jump would otherwise pull it from.
        damping_alpha_dt = self._damping_alpha * self._substep_dt
        pos = self.elements_v[f + 1, i_v, i_b].pos
        force = -self.elements_v_info[i_v].mass_over_dt2 * (
            (pos - self.elements_v_energy[i_b, i_v].inertia)
            + (pos - self.elements_v[f, i_v, i_b].pos) * damping_alpha_dt
        )
        if qd.static(self._enable_vertex_constraints):
            if self.vertex_constraints.is_constrained[i_v, i_b]:
                force = qd.Vector.zero(gs.qd_float, 3)
        return force

    @qd.func
    def compute_elastic_product(self, i_e: int, i_b: int, S: qd.types.matrix(4, 3), x: qd.types.matrix(3, 4)):
        """
        Return the change of the deformation gradient of an element along a displacement of its vertices, and the product
        of the Hessian of its energy density with that change.

        `x` holds the displacement of each vertex of the element in its columns, and `S` maps them to the deformation
        gradient (see _func_compute_element_mapping_matrix). The product of the Hessian of the element with the
        displacement is then the product of the second matrix with the transpose of `S`, and the second order term of
        the energy density along it half the scalar product of the two matrices.
        """
        dF = x @ S
        mu = self.elements_i[i_e].mu
        lam = self.elements_i[i_e].lam
        dP = qd.Matrix.zero(gs.qd_float, 3, 3)
        for mat_idx in qd.static(range(len(self._mats))):
            if self.elements_i[i_e].mat_idx == mat_idx:
                dP = self._mats[mat_idx].hessian_product(i_e, i_b, mu, lam, dF)
        return dF, dP

    @qd.func
    def _func_compute_element_mapping_matrix(self, i_vs, B, i_b):
        """
        Compute the element mapping matrix S for an element.
        """
        S = qd.Matrix.zero(gs.qd_float, 4, 3)
        S[:3, :] = B
        S[3, :] = -B[0, :] - B[1, :] - B[2, :]

        if qd.static(self._enable_vertex_constraints):
            for i in qd.static(range(4)):
                if self.vertex_constraints.is_constrained[i_vs[i], i_b]:
                    S[i, :] = qd.Vector.zero(gs.qd_float, 3)
        return S

    @qd.func
    def _func_compute_ele_energy(self, f: qd.i32):
        """
        Compute the energy for each element in the batch. Should only be used in linesearch.
        """
        for i_b, i_e in qd.ndrange(self._B, self.n_elements):
            if not self.batch_linesearch_active[i_b]:
                continue

            J, F = self._compute_ele_J_F(f + 1, i_e, i_b)
            mu = self.elements_i[i_e].mu
            lam = self.elements_i[i_e].lam
            actu = self.elements_el[f, i_e, i_b].actu
            m_dir = self.elements_i[i_e].muscle_direction
            for mat_idx in qd.static(range(len(self._mats))):
                if self.elements_i[i_e].mat_idx == mat_idx:
                    self.elements_el_energy[i_b, i_e].energy = self._mats[mat_idx].compute_energy(
                        i_e, i_b, mu, lam, J, F, actu, m_dir
                    )

            # add linearized damping energy
            if self._damping_beta > gs.EPS:
                damping_beta_over_dt = self._damping_beta / self._substep_dt
                i_vs = self.elements_i[i_e].el2v
                S = self._func_compute_element_mapping_matrix(i_vs, self.elements_i[i_e].B, i_b)

                x_diff = qd.Matrix.cols(
                    [
                        self.elements_v[f + 1, i_vs[i], i_b].pos - self.elements_v[f, i_vs[i], i_b].pos
                        for i in qd.static(range(4))
                    ]
                )
                dF, dP = self.compute_elastic_product(i_e, i_b, S, x_diff)
                self.elements_el_energy[i_b, i_e].energy = (
                    self.elements_el_energy[i_b, i_e].energy + 0.5 * damping_beta_over_dt * (dF * dP).sum()
                )

    @qd.func
    def _func_block_sum(self, i_t: int, values, sh_sum: qd.template()):
        # The tiles of 32 threads sum first, then every thread adds up the tile sums in order, so the totals are the same
        # bitwise on every run. On CPU, the block is a single thread.
        totals = values
        if qd.static(self._env_block_dim > 1):
            for i in qd.static(range(values.n)):
                value_tile = su.qd_block_sum(values[i])
                if i_t % 32 == 0:
                    sh_sum[i, i_t // 32] = value_tile
            qd.simt.block.sync()
            totals = qd.Vector.zero(gs.qd_float, values.n)
            for i_tile in range(self._env_block_dim // 32):
                totals = totals + qd.Vector([sh_sum[i, i_tile] for i in qd.static(range(values.n))])
            qd.simt.block.sync()
        return totals

    @qd.func
    def _func_accumulate_vertex(
        self,
        f: int,
        i_b: int,
        i_v: int,
        i_lane: int,
        forces: qd.template(),
        batch_active: qd.template(),
        gather_Ap: qd.template(),
        gather_internal_force: qd.template(),
        clip_vertex: qd.template(),
    ):
        # The elastic terms of a vertex, summed in a fixed order: the internal force alone into 'forces' for every
        # environment if asked, else Ap alone or the force and diagonal block of the active environments. On CPU, the
        # thread of an environment walks the corners of all its elements in element order and adds each term to its
        # vertex. On GPU, the lanes of a vertex stride its corners (see 'vertices_corners_idx'), then add up their sums
        # in a fixed order, every lane of a subgroup tile taking part and the lanes past the last vertex doing nothing
        # when 'clip_vertex' is set.
        is_serial = qd.static(self._env_block_dim == 1)
        n_lanes = qd.static(1 if is_serial else 2**LOG2_CORNER_LANES)
        damping_alpha_dt = self._damping_alpha * self._substep_dt
        damping_beta_factor = self._damping_beta / self._substep_dt + 1.0
        I_3 = qd.Matrix.identity(gs.qd_float, 3)
        vec_elastic = qd.Vector.zero(gs.qd_float, 3)
        mat_elastic = qd.Matrix.zero(gs.qd_float, 3, 3)
        is_active = True
        if qd.static(not gather_internal_force):
            is_active = batch_active[i_b]
        if qd.static(clip_vertex):
            is_active = is_active and i_v < self.n_vertices
        if is_active:
            i_c_start = 0
            n_corners = 4 * self.n_elements
            if qd.static(is_serial and not gather_internal_force):
                for i_v_ in range(self.n_vertices):
                    if qd.static(gather_Ap):
                        self.pcg_state_v[i_b, i_v_].Ap = qd.Vector.zero(gs.qd_float, 3)
                    else:
                        self.elements_v_energy[i_b, i_v_].force = qd.Vector.zero(gs.qd_float, 3)
                        self.pcg_state_v[i_b, i_v_].diag3x3 = qd.Matrix.zero(gs.qd_float, 3, 3)
            if qd.static(not is_serial):
                i_c_start = self.vertices_corners_start[i_v]
                n_corners = self.vertices_corners_start[i_v + 1] - i_c_start

            # On CPU the corners of an element follow each other, sharing its mapping matrix, stress or force
            S = qd.Matrix.zero(gs.qd_float, 4, 3)
            stress = qd.Matrix.zero(gs.qd_float, 3, 3)
            force_internal = qd.Matrix.zero(gs.qd_float, 3, 4)
            for i_c_ in range((n_corners + n_lanes - 1) // n_lanes):
                i_c = i_c_ * n_lanes + i_lane
                if i_c < n_corners:
                    i_corner = i_c
                    if qd.static(not is_serial):
                        i_corner = self.vertices_corners_idx[i_c_start + i_c]
                    i_e = i_corner // 4
                    i_k = i_corner % 4
                    V = self.elements_i[i_e].V
                    i_vs = self.elements_i[i_e].el2v
                    is_new_element = True
                    if qd.static(is_serial):
                        is_new_element = i_k == 0
                    if is_new_element:
                        if qd.static(gather_internal_force):
                            force_internal = self._func_element_internal_force(f, i_e, i_b)
                        else:
                            S = self._func_compute_element_mapping_matrix(i_vs, self.elements_i[i_e].B, i_b)
                        if qd.static(gather_Ap):
                            p = qd.Matrix.cols([self.pcg_state_v[i_b, i_vs[i]].p for i in qd.static(range(4))])
                            _, dP = self.compute_elastic_product(i_e, i_b, S, p)
                            stress = V * damping_beta_factor * dP
                    S_k = S[i_k, :]

                    vec_corner = qd.Vector.zero(gs.qd_float, 3)
                    mat_corner = qd.Matrix.zero(gs.qd_float, 3, 3)
                    if qd.static(gather_internal_force):
                        vec_corner = force_internal[:, i_k]
                    elif qd.static(gather_Ap):
                        vec_corner = stress @ S_k
                    else:
                        vec_corner = -V * self.elements_el_energy[i_b, i_e].gradient @ S_k
                        # Column j of the block is the force that a unit displacement of the vertex along axis j
                        # induces on it.
                        corner_k = qd.Matrix.identity(gs.qd_float, 4)[i_k, :]
                        for j in range(3):
                            _, dP = self.compute_elastic_product(i_e, i_b, S, I_3[:, j].outer_product(corner_k))
                            mat_corner = mat_corner + (V * damping_beta_factor * dP @ S_k).outer_product(I_3[:, j])

                    if qd.static(is_serial):
                        i_v_k = i_vs[i_k]
                        if qd.static(gather_internal_force):
                            for j in qd.static(range(3)):
                                forces[i_b, i_v_k, j] = forces[i_b, i_v_k, j] + vec_corner[j]
                        elif qd.static(gather_Ap):
                            self.pcg_state_v[i_b, i_v_k].Ap = self.pcg_state_v[i_b, i_v_k].Ap + vec_corner
                        else:
                            force = self.elements_v_energy[i_b, i_v_k].force + vec_corner
                            self.elements_v_energy[i_b, i_v_k].force = force
                            diag3x3 = self.pcg_state_v[i_b, i_v_k].diag3x3 + mat_corner
                            self.pcg_state_v[i_b, i_v_k].diag3x3 = diag3x3
                    else:
                        vec_elastic = vec_elastic + vec_corner
                        mat_elastic = mat_elastic + mat_corner
        if qd.static(not is_serial):
            for i in qd.static(range(3)):
                vec_elastic[i] = su.qd_block_sum(vec_elastic[i], LOG2_CORNER_LANES)
                if qd.static(not gather_Ap and not gather_internal_force):
                    for j in qd.static(range(3)):
                        mat_elastic[i, j] = su.qd_block_sum(mat_elastic[i, j], LOG2_CORNER_LANES)

        if qd.static(gather_internal_force):
            # The first lane of a vertex writes its sum, which the walk over the corners adds up directly on CPU
            if qd.static(not is_serial):
                if i_lane == 0:
                    for j in qd.static(range(3)):
                        forces[i_b, i_v, j] = vec_elastic[j]
        elif is_active:
            # The inertia terms of the vertices this thread owns add to their elastic sums
            n_owned = 1 if i_lane == 0 else 0
            if qd.static(is_serial):
                n_owned = self.n_vertices
            for i_o in range(n_owned):
                i_v_o = i_v
                if qd.static(is_serial):
                    i_v_o = i_o
                    vec_elastic = self.elements_v_energy[i_b, i_v_o].force
                    mat_elastic = self.pcg_state_v[i_b, i_v_o].diag3x3
                    if qd.static(gather_Ap):
                        vec_elastic = self.pcg_state_v[i_b, i_v_o].Ap
                mass_over_dt2 = self.elements_v_info[i_v_o].mass_over_dt2
                if qd.static(gather_Ap):
                    p = self.pcg_state_v[i_b, i_v_o].p
                    self.pcg_state_v[i_b, i_v_o].Ap = mass_over_dt2 * (damping_alpha_dt + 1.0) * p + vec_elastic
                else:
                    force_inertia = self._func_vertex_inertia_force(f, i_v_o, i_b)
                    self.elements_v_energy[i_b, i_v_o].force = force_inertia + vec_elastic
                    diag3x3 = mat_elastic + mass_over_dt2 * (damping_alpha_dt + 1.0) * I_3
                    self.pcg_state_v[i_b, i_v_o].diag3x3 = diag3x3
                    # 3-by-3 block Jacobi preconditioner
                    self.pcg_state_v[i_b, i_v_o].prec = diag3x3.inverse()

    @qd.func
    def _func_accumulate_vertices(
        self,
        f: int,
        forces: qd.template(),
        batch_active: qd.template(),
        pcg_counter: qd.template(),
        gather_Ap: qd.template(),
        gather_internal_force: qd.template(),
        update_loop_flag: qd.template(),
    ):
        is_serial = qd.static(self._env_block_dim == 1)
        n_lanes = qd.static(1 if is_serial else 2**LOG2_CORNER_LANES)
        n_env_threads = qd.static(1 if is_serial else self.n_vertices * n_lanes)
        for i_flat in range(self._B * n_env_threads):
            # A PCG iteration starts here: its loop flag drops until an environment iterates on (see
            # '_kernel_batch_solve').
            if qd.static(gather_Ap and update_loop_flag):
                if i_flat == 0:
                    pcg_counter[()] = 0
                    self.solve_iterations_left[None].pcg = self.solve_iterations_left[None].pcg - 1
            i_b = i_flat // n_env_threads
            i_lane = i_flat % n_lanes
            i_v = i_flat // n_lanes % self.n_vertices
            self._func_accumulate_vertex(
                f, i_b, i_v, i_lane, forces, batch_active, gather_Ap, gather_internal_force, clip_vertex=False
            )

    @qd.func
    def _func_init_pcg_env(
        self, f: int, i_b: int, i_t: int, sh_sum: qd.template(), pcg_counter: qd.template(), is_adjoint: qd.template()
    ):
        # The first residual of the PCG solve of an environment, by the threads of its block (see '_func_block_sum'):
        # the forces at the end of the substep, or for the adjoint solve the adjoint of the positions there (see
        # '_implicit_solve_grad' for that solve).
        n_threads = qd.static(self._env_block_dim)
        is_active = self.batch_active[i_b]
        rTr = gs.qd_float(0.0)
        rTz = gs.qd_float(0.0)
        if is_active:
            for i_v_ in range((self.n_vertices + n_threads - 1) // n_threads):
                i_v = i_v_ * n_threads + i_t
                if i_v < self.n_vertices:
                    r = self.elements_v_energy[i_b, i_v].force
                    if qd.static(is_adjoint):
                        r = self.elements_v.grad[f + 1, i_v, i_b].pos
                    z = self.pcg_state_v[i_b, i_v].prec @ r
                    self.pcg_state_v[i_b, i_v].x = qd.Vector.zero(gs.qd_float, 3)
                    self.pcg_state_v[i_b, i_v].r = r
                    self.pcg_state_v[i_b, i_v].z = z
                    self.pcg_state_v[i_b, i_v].p = z
                    rTr = rTr + r.dot(r)
                    rTz = rTz + r.dot(z)
        rTr, rTz = self._func_block_sum(i_t, qd.Vector([rTr, rTz]), sh_sum)
        if i_t == 0:
            if is_active:
                self.pcg_state[i_b].rTr = rTr
                self.pcg_state[i_b].rTz = rTz
            rTr_threshold = self.solve_info[None].pcg_threshold
            if qd.static(self._requires_grad):
                is_first_residual = True
                if qd.static(not is_adjoint):
                    is_first_residual = (
                        self.solve_iterations_left[None].newton == self.solve_info[None].n_newton_iterations
                    )
                if is_first_residual:
                    self.pcg_state[i_b].rTr_threshold = self.solve_info[None].pcg_threshold_rel * rTr
                rTr_threshold = self.pcg_state[i_b].rTr_threshold
            is_pcg_active = is_active and rTr > rTr_threshold
            self.batch_pcg_active[i_b] = is_pcg_active
            if is_pcg_active:
                pcg_counter[()] = 1

    @qd.func
    def _func_init_pcg(self, f: int, pcg_counter: qd.template(), is_adjoint: qd.template()):
        pcg_counter[()] = 0
        n_threads = qd.static(self._env_block_dim)
        qd.loop_config(block_dim=qd.static(n_threads if n_threads > 1 else None))
        for i_flat in range(self._B * n_threads):
            sh_sum = qd.simt.block.SharedArray((2, qd.static((n_threads + 31) // 32)), gs.qd_float)
            i_b = i_flat // n_threads
            i_t = i_flat % n_threads
            self._func_init_pcg_env(f, i_b, i_t, sh_sum, pcg_counter, is_adjoint)
        self.solve_iterations_left[None].pcg = self.solve_info[None].n_pcg_iterations

    @qd.func
    def _func_update_pcg_env(
        self, i_b: int, i_t: int, sh_sum: qd.template(), pcg_counter: qd.template(), update_loop_flag: qd.template()
    ):
        # The step of a PCG iteration of an environment past its Ap, by the threads of its block (see '_func_block_sum')
        n_threads = qd.static(self._env_block_dim)
        is_active = self.batch_pcg_active[i_b]
        rTz = self.pcg_state[i_b].rTz

        # pTAp, then x, r, z and the new rTr, rTz
        pTAp = gs.qd_float(0.0)
        if is_active:
            for i_v_ in range((self.n_vertices + n_threads - 1) // n_threads):
                i_v = i_v_ * n_threads + i_t
                if i_v < self.n_vertices:
                    pTAp = pTAp + self.pcg_state_v[i_b, i_v].p.dot(self.pcg_state_v[i_b, i_v].Ap)
        pTAp = self._func_block_sum(i_t, qd.Vector([pTAp]), sh_sum)[0]
        alpha = gs.qd_float(0.0)
        rTr_new = gs.qd_float(0.0)
        rTz_new = gs.qd_float(0.0)
        if is_active:
            alpha = rTz / pTAp
            for i_v_ in range((self.n_vertices + n_threads - 1) // n_threads):
                i_v = i_v_ * n_threads + i_t
                if i_v < self.n_vertices:
                    p = self.pcg_state_v[i_b, i_v].p
                    r = self.pcg_state_v[i_b, i_v].r - alpha * self.pcg_state_v[i_b, i_v].Ap
                    z = self.pcg_state_v[i_b, i_v].prec @ r
                    self.pcg_state_v[i_b, i_v].x = self.pcg_state_v[i_b, i_v].x + alpha * p
                    self.pcg_state_v[i_b, i_v].r = r
                    self.pcg_state_v[i_b, i_v].z = z
                    rTr_new = rTr_new + r.dot(r)
                    rTz_new = rTz_new + r.dot(z)
        rTr_new, rTz_new = self._func_block_sum(i_t, qd.Vector([rTr_new, rTz_new]), sh_sum)

        # Convergence check, then the new search direction
        rTr_threshold = self.solve_info[None].pcg_threshold
        if qd.static(self._requires_grad):
            rTr_threshold = self.pcg_state[i_b].rTr_threshold
        is_still_active = is_active and rTr_new > rTr_threshold
        beta = gs.qd_float(0.0)
        if is_still_active:
            beta = rTz_new / rTz
            for i_v_ in range((self.n_vertices + n_threads - 1) // n_threads):
                i_v = i_v_ * n_threads + i_t
                if i_v < self.n_vertices:
                    self.pcg_state_v[i_b, i_v].p = self.pcg_state_v[i_b, i_v].z + beta * self.pcg_state_v[i_b, i_v].p
        if i_t == 0:
            if is_active:
                self.pcg_state[i_b].pTAp = pTAp
                self.pcg_state[i_b].alpha = alpha
                self.pcg_state[i_b].rTr_new = rTr_new
                self.pcg_state[i_b].rTz_new = rTz_new
            if is_still_active:
                self.pcg_state[i_b].beta = beta
                self.pcg_state[i_b].rTr = rTr_new
                self.pcg_state[i_b].rTz = rTz_new
                if qd.static(update_loop_flag):
                    if self.solve_iterations_left[None].pcg > 0:
                        pcg_counter[()] = 1
            self.batch_pcg_active[i_b] = is_still_active

    @qd.func
    def _func_pcg_iter(self, pcg_counter: qd.template()):
        self._func_accumulate_vertices(
            f=0,
            forces=None,
            batch_active=self.batch_pcg_active,
            pcg_counter=pcg_counter,
            gather_Ap=True,
            gather_internal_force=False,
            update_loop_flag=True,
        )
        n_threads = qd.static(self._env_block_dim)
        qd.loop_config(block_dim=qd.static(n_threads if n_threads > 1 else None))
        for i_flat in range(self._B * n_threads):
            sh_sum = qd.simt.block.SharedArray((2, qd.static((n_threads + 31) // 32)), gs.qd_float)
            i_b = i_flat // n_threads
            i_t = i_flat % n_threads
            self._func_update_pcg_env(i_b, i_t, sh_sum, pcg_counter, update_loop_flag=True)

    @qd.func
    def _func_init_linesearch(self, f: int, linesearch_counter: qd.template()):
        linesearch_counter[()] = 0
        n_threads = qd.static(self._env_block_dim)
        qd.loop_config(block_dim=qd.static(n_threads if n_threads > 1 else None))
        for i_flat in range(self._B * n_threads):
            i_b = i_flat // n_threads
            i_t = i_flat % n_threads
            sh_sum = qd.simt.block.SharedArray((2, qd.static((n_threads + 31) // 32)), gs.qd_float)
            is_active = self.batch_active[i_b]
            prev_energy = gs.qd_float(0.0)
            m = gs.qd_float(0.0)
            if is_active:
                # Inertia, x_prev, m
                for i_v_ in range((self.n_vertices + n_threads - 1) // n_threads):
                    i_v = i_v_ * n_threads + i_t
                    if i_v < self.n_vertices:
                        pos = self.elements_v[f + 1, i_v, i_b].pos
                        diff = pos - self.elements_v_energy[i_b, i_v].inertia
                        prev_energy = prev_energy + 0.5 * self.elements_v_info[i_v].mass_over_dt2 * diff.dot(diff)
                        self.linesearch_state_v[i_b, i_v].x_prev = pos
                        m = m - self.pcg_state_v[i_b, i_v].x.dot(self.elements_v_energy[i_b, i_v].force)
                # Elastic
                for i_e_ in range((self.n_elements + n_threads - 1) // n_threads):
                    i_e = i_e_ * n_threads + i_t
                    if i_e < self.n_elements:
                        prev_energy = prev_energy + self.elements_el_energy[i_b, i_e].energy * self.elements_i[i_e].V
            prev_energy, m = self._func_block_sum(i_t, qd.Vector([prev_energy, m]), sh_sum)
            if i_t == 0:
                self.batch_linesearch_active[i_b] = is_active
                if is_active:
                    linesearch_counter[()] = 1
                    self.linesearch_state[i_b].prev_energy = prev_energy
                    self.linesearch_state[i_b].step_size = 1.0
                    self.linesearch_state[i_b].m = m
        self.solve_iterations_left[None].linesearch = self.solve_info[None].n_linesearch_iterations

    @qd.func
    def _func_linesearch_iter(self, f: int, linesearch_counter: qd.template()):
        # update pos
        for i_b, i_v in qd.ndrange(self._B, self.n_vertices):
            # A line search iteration starts here: its loop flag drops until an environment iterates on
            if i_b == 0 and i_v == 0:
                linesearch_counter[()] = 0
                self.solve_iterations_left[None].linesearch = self.solve_iterations_left[None].linesearch - 1
            if self.batch_linesearch_active[i_b]:
                self.elements_v[f + 1, i_v, i_b].pos = (
                    self.linesearch_state_v[i_b, i_v].x_prev
                    + self.linesearch_state[i_b].step_size * self.pcg_state_v[i_b, i_v].x
                )

        # compute elastic energy
        self._func_compute_ele_energy(f)

        damping_alpha_dt = self._damping_alpha * self._substep_dt
        n_threads = qd.static(self._env_block_dim)
        qd.loop_config(block_dim=qd.static(n_threads if n_threads > 1 else None))
        for i_flat in range(self._B * n_threads):
            i_b = i_flat // n_threads
            i_t = i_flat % n_threads
            sh_sum = qd.simt.block.SharedArray((2, qd.static((n_threads + 31) // 32)), gs.qd_float)
            is_active = self.batch_linesearch_active[i_b]
            energy = gs.qd_float(0.0)
            if is_active:
                # Inertia and damping
                for i_v_ in range((self.n_vertices + n_threads - 1) // n_threads):
                    i_v = i_v_ * n_threads + i_t
                    if i_v < self.n_vertices:
                        mass_over_dt2 = self.elements_v_info[i_v].mass_over_dt2
                        pos = self.elements_v[f + 1, i_v, i_b].pos
                        diff = pos - self.elements_v_energy[i_b, i_v].inertia
                        energy = energy + 0.5 * mass_over_dt2 * diff.dot(diff)
                        if self._damping_alpha > 0.0:
                            diff = pos - self.elements_v[f, i_v, i_b].pos
                            energy = energy + 0.5 * mass_over_dt2 * diff.dot(diff) * damping_alpha_dt
                # Elastic
                for i_e_ in range((self.n_elements + n_threads - 1) // n_threads):
                    i_e = i_e_ * n_threads + i_t
                    if i_e < self.n_elements:
                        energy = energy + self.elements_el_energy[i_b, i_e].energy * self.elements_i[i_e].V
            energy = self._func_block_sum(i_t, qd.Vector([energy]), sh_sum)[0]

            # check condition
            if i_t == 0 and is_active:
                self.linesearch_state[i_b].energy = energy
                step_size = self.linesearch_state[i_b].step_size
                is_still_active = (
                    energy
                    > self.linesearch_state[i_b].prev_energy
                    + self.solve_info[None].linesearch_c * step_size * self.linesearch_state[i_b].m
                )
                self.batch_linesearch_active[i_b] = is_still_active
                if is_still_active:
                    if self.solve_iterations_left[None].linesearch > 0:
                        linesearch_counter[()] = 1
                    self.linesearch_state[i_b].step_size = step_size * self.solve_info[None].linesearch_tau

    @qd.func
    def _func_skip_linesearch(self, f: int):
        # Inertia, x_prev, m
        for i_b, i_v in qd.ndrange(self._B, self.n_vertices):
            if not self.batch_active[i_b]:
                continue
            self.elements_v[f + 1, i_v, i_b].pos = self.elements_v[f + 1, i_v, i_b].pos + self.pcg_state_v[i_b, i_v].x

    @qd.func
    def _func_newton_head(self, f: int, pcg_counter: qd.template()):
        # The energy gradient, force and preconditioner a Newton iteration starts from, then the first PCG residual
        self._func_compute_ele_energy_gradient(f)
        self._func_accumulate_vertices(
            f,
            forces=None,
            batch_active=self.batch_active,
            pcg_counter=pcg_counter,
            gather_Ap=False,
            gather_internal_force=False,
            update_loop_flag=False,
        )
        self._func_init_pcg(f, pcg_counter, is_adjoint=False)

    @qd.kernel(graph=True)
    def _kernel_batch_solve(
        self,
        f: int,
        newton_counter: qd.types.ndarray(dtype=qd.i32, ndim=0),
        pcg_counter: qd.types.ndarray(dtype=qd.i32, ndim=0),
        linesearch_counter: qd.types.ndarray(dtype=qd.i32, ndim=0),
    ):
        newton_counter[()] = 1
        self.solve_iterations_left[None].newton = self.solve_info[None].n_newton_iterations
        # FIXME: quadrants#1083 - the statements before a graph do_while run again on every pass when the loop body
        # opens with bare statements, so each loop body opens with a for loop.
        while qd.graph.do_while(newton_counter):
            self._func_newton_head(f, pcg_counter)
            self._func_pcg_solve(pcg_counter)
            if qd.static(self._n_linesearch_iterations > 0):
                self._func_init_linesearch(f, linesearch_counter)
                while qd.graph.do_while(linesearch_counter):
                    self._func_linesearch_iter(f, linesearch_counter)
            else:
                self._func_skip_linesearch(f)
            self.solve_iterations_left[None].newton = self.solve_iterations_left[None].newton - 1
            if self.solve_iterations_left[None].newton <= 0:
                newton_counter[()] = 0

    @qd.func
    def _func_pcg_solve(self, pcg_counter: qd.template()):
        if qd.static(self._is_pcg_solved_per_env_block):
            self._func_pcg_solve_per_env_block(pcg_counter)
        else:
            while qd.graph.do_while(pcg_counter):
                self._func_pcg_iter(pcg_counter)

    @qd.func
    def _func_pcg_solve_per_env_block(self, pcg_counter: qd.template()):
        # The whole PCG solve of an environment by one block of threads, which synchronize between the steps of an
        # iteration rather than ending a kernel (see '_is_pcg_solved_per_env_block').
        n_threads = qd.static(self._env_block_dim)
        n_lanes = qd.static(2**LOG2_CORNER_LANES)
        qd.loop_config(block_dim=n_threads)
        for i_flat in range(self._B * n_threads):
            i_b = i_flat // n_threads
            i_t = i_flat % n_threads
            sh_sum = qd.simt.block.SharedArray((2, qd.static((n_threads + 31) // 32)), gs.qd_float)
            for _ in range(self.solve_info[None].n_pcg_iterations):
                # Every thread of the block reads the flag behind the barrier that follows its last write
                if not self.batch_pcg_active[i_b]:
                    break
                for i_item_ in range((self.n_vertices * n_lanes + n_threads - 1) // n_threads):
                    i_item = i_item_ * n_threads + i_t
                    self._func_accumulate_vertex(
                        f=0,
                        i_b=i_b,
                        i_v=i_item // n_lanes,
                        i_lane=i_item % n_lanes,
                        forces=None,
                        batch_active=self.batch_pcg_active,
                        gather_Ap=True,
                        gather_internal_force=False,
                        clip_vertex=True,
                    )
                qd.simt.grid.mem_fence()
                qd.simt.block.sync()
                self._func_update_pcg_env(i_b, i_t, sh_sum, pcg_counter, update_loop_flag=False)
                qd.simt.grid.mem_fence()
                qd.simt.block.sync()

    @qd.kernel(graph=True)
    def _kernel_solve_adjoint(self, f: int, pcg_counter: qd.types.ndarray(dtype=qd.i32, ndim=0)):
        # The adjoint solve of a substep, whose solution seeds the adjoint of the forces (see '_implicit_solve_grad')
        self._func_init_pcg(f, pcg_counter, is_adjoint=True)
        self._func_pcg_solve(pcg_counter)
        for i_b, i_v in qd.ndrange(self._B, self.n_vertices):
            self.elements_v_energy.grad[i_b, i_v].force = self.pcg_state_v[i_b, i_v].x

    @qd.kernel
    def _kernel_compute_implicit_force(self, f: int):
        # The forces at the end of a substep of the implicit solver, whose autodiff alone runs (see
        # '_implicit_solve_grad'). The elements scatter their forces, which autodiff reverses, where the solve gathers.
        for i_b, i_e in qd.ndrange(self._B, self.n_elements):
            _, gradient = self._func_element_energy_gradient(f, i_e, i_b)
            i_vs = self.elements_i[i_e].el2v
            S = self._func_compute_element_mapping_matrix(i_vs, self.elements_i[i_e].B, i_b)
            force = -self.elements_i[i_e].V * gradient @ S.transpose()
            for i_k in qd.static(range(4)):
                qd.atomic_add(self.elements_v_energy[i_b, i_vs[i_k]].force, force[:, i_k])
        for i_b, i_v in qd.ndrange(self._B, self.n_vertices):
            qd.atomic_add(self.elements_v_energy[i_b, i_v].force, self._func_vertex_inertia_force(f, i_v, i_b))

    @qd.kernel
    def _kernel_material_data_grad(self, f: int):
        # The adjoint of the data the materials compute from the start of the substep, which the forces seeded with
        # lambda pull on (see '_implicit_solve_grad'), carried to the start positions element by element.
        for i_b, i_e in qd.ndrange(self._B, self.n_elements):
            i_vs = self.elements_i[i_e].el2v
            S = self._func_compute_element_mapping_matrix(i_vs, self.elements_i[i_e].B, i_b)
            # The forces -V * gradient @ S^T on the vertices dotted with lambda give <gradient, -V * lambda @ S>
            vertices_lambda = qd.Matrix.cols(
                [self.elements_v_energy.grad[i_b, i_vs[i_k]].force for i_k in qd.static(range(4))]
            )
            gradient_grad = -self.elements_i[i_e].V * vertices_lambda @ S
            _, F_start = self._compute_ele_J_F(f, i_e, i_b)
            _, F = self._compute_ele_J_F(f + 1, i_e, i_b)
            dF_damping = qd.Matrix.zero(gs.qd_float, 3, 3)
            if qd.static(self._damping_beta > gs.EPS):
                damping_beta_over_dt = self._damping_beta / self._substep_dt
                dF_damping = damping_beta_over_dt * self._func_element_displacement(f, i_e, i_b) @ S
            mu = self.elements_i[i_e].mu
            lam = self.elements_i[i_e].lam
            actu = self.elements_el[f, i_e, i_b].actu
            m_dir = self.elements_i[i_e].muscle_direction
            F_start_grad = qd.Matrix.zero(gs.qd_float, 3, 3)
            for mat_idx in qd.static(range(len(self._mats))):
                if self.elements_i[i_e].mat_idx == mat_idx:
                    F_start_grad = self._mats[mat_idx].pre_compute_grad(
                        i_e, i_b, mu, lam, F_start, F, dF_damping, gradient_grad, actu, m_dir
                    )
            # F_start = D B, D holding the positions of the first three vertices relative to the last in its columns
            D_grad = F_start_grad @ self.elements_i[i_e].B.transpose()
            for i_k in qd.static(range(3)):
                qd.atomic_add(self.elements_v.grad[f, i_vs[i_k], i_b].pos, D_grad[:, i_k])
            qd.atomic_add(self.elements_v.grad[f, i_vs[3], i_b].pos, -D_grad[:, 0] - D_grad[:, 1] - D_grad[:, 2])

    def _implicit_solve_grad(self, f: int) -> None:
        # The end positions x+ are the root of the forces R(x+; x, v, actu), whose Jacobian with respect to x+ is minus
        # the Hessian H of the incremental energy, symmetric and constant since every material the implicit solver
        # supports has an energy quadratic in x+. The adjoint solve H lambda = g of the adjoint g of x+ runs the PCG
        # of the solve, then the autodiff of the forces seeded with lambda adds lambda^T dR/d(input) to the adjoint of
        # every input but the data the materials compute from the start of the substep, which autodiff takes as
        # constant and the closed-form reverse of the materials carries to the start positions instead. A constrained
        # vertex has no force, so its target takes its adjoint plus the pull of lambda.
        self._kernel_solve_adjoint(f, self._pcg_counter)
        self._kernel_compute_implicit_force.grad(f)
        self._kernel_material_data_grad(f)

    @qd.kernel
    def setup_pos_vel(self, f: qd.i32):
        for i_v, i_b in qd.ndrange(self.n_vertices, self._B):
            # set pos and vel
            self.elements_v[f + 1, i_v, i_b].vel = (
                self.elements_v[f + 1, i_v, i_b].pos - self.elements_v[f, i_v, i_b].pos
            ) / self.substep_dt

    # ------------------------------------------------------------------------------------
    # ------------------------------------ stepping --------------------------------------
    # ------------------------------------------------------------------------------------

    def substep_pre_coupling(self, f):
        if self.is_active:
            # Skip FEM solver step if using IPCCoupler (IPC handles FEM simulation)
            from genesis.engine.couplers import IPCCoupler

            if isinstance(self.sim._coupler, IPCCoupler):
                pass  # IPC coupler handles FEM simulation
            elif self._use_implicit_solver:
                self.precompute_material_data(f)
                self.init_pos_and_inertia(f, self._sim.rigid_solver.dyn_state.links)
                # The solve is reversed through the adjoint of its root, which spares reversing its iterations
                with custom_backward(self._implicit_solve_grad, f):
                    self._kernel_batch_solve(f, self._newton_counter, self._pcg_counter, self._linesearch_counter)
                self.setup_pos_vel(f)
            else:
                self.init_pos_and_vel(f)
                self.compute_vel(f)
                self.apply_uniform_force(f)
                if self._constraints_initialized:
                    self.apply_soft_constraints(f, self._sim.rigid_solver.dyn_state.links)

    def substep_post_coupling(self, f):
        if self.is_active:
            self.compute_pos(f)
            if self._constraints_initialized and not self._use_implicit_solver:
                self.apply_hard_constraints(f, self._sim.rigid_solver.dyn_state.links)

    @qd.kernel
    def copy_frame(self, source: qd.i32, target: qd.i32):
        # Copy pos/vel for all vertices and all batch indices
        for i_v, i_b in qd.ndrange(self.n_vertices_max, self._B):
            self.elements_v[target, i_v, i_b].pos = self.elements_v[source, i_v, i_b].pos
            self.elements_v[target, i_v, i_b].vel = self.elements_v[source, i_v, i_b].vel

    # ------------------------------------------------------------------------------------
    # ----------------------------------- gradient ---------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def data(self):
        """Yield the state of the elements, the frame 0 of which a boundary checkpoint holds (see 'Solver.data')."""
        for name, field in (
            ("elements_v.pos", self.elements_v.pos),
            ("elements_v.vel", self.elements_v.vel),
            ("elements_el.actu", self.elements_el.actu),
        ):
            yield ArrayItem(name, DataKind.STATE, field)

    # ------------------------------------------------------------------------------------
    # --------------------------------------- io -----------------------------------------
    # ------------------------------------------------------------------------------------

    @mutates()
    def set_state(self, f, state, envs_idx=None):
        if self.is_active:
            self._kernel_set_state(f, state.pos, state.vel)
            self.clear_errno(envs_idx)

    def get_state(self, f):
        if self.is_active:
            pos, vel = self._read_data((self.elements_v.pos, self.elements_v.vel), None, None, copy=True)
            state = FEMSolverState(self._scene, pos, vel)
        else:
            state = None
        return state

    def get_state_render(self, f):
        """
        Refresh and return the render geometry of every visual geom, laid out contiguously.

        Returns
        -------
        tuple
            (vverts_pos, vverts_uvs, vfaces_indices) - environment-offset render vertex positions with shape
            (n_vverts, B), their UV coordinates, and the render triangles in global render vertex space.
        """
        if not self.is_active or self._n_vverts == 0:
            return None, None, None

        self._kernel_get_state_render(f)
        return self.vverts_render.pos, self.vverts_uvs, self.vfaces_indices

    def get_forces(self):
        """
        Get the internal elastic force on every vertex, computed from the current state.

        The force of each element follows the material model of the solver in use, and the forces of a body sum to zero.
        Damping, gravity and contact forces are left out. With the IPC coupler, which simulates the FEM entities itself,
        it raises an exception.

        Returns:
            torch.Tensor : shape (B, n_vertices, 3) where B is batch size
        """
        if not self.is_active:
            return None
        # Imported here to avoid a circular dependency (see 'substep_pre_coupling').
        from genesis.engine.couplers import IPCCoupler

        # The IPC coupler simulates the FEM entities with its own models, and leaves the elements of Cloth ones empty.
        if isinstance(self._sim.coupler, IPCCoupler):
            gs.raise_exception("The internal forces of FEM entities are not available with the IPC coupler.")

        # Between two steps, the state sits in frame 0 (see 'Solver.has_frames'). The linear corotated model takes the
        # rotation of each element from this state.
        if self._use_implicit_solver:
            self.precompute_material_data(0)
        forces = torch.zeros((self._B, self.n_vertices, 3), dtype=gs.tc_float, device=gs.device)
        self._kernel_get_forces(0, forces)
        return forces

    @qd.kernel
    def _kernel_get_forces(self, f: int, forces: qd.types.ndarray()):
        self._func_accumulate_vertices(
            f,
            forces,
            batch_active=None,
            pcg_counter=None,
            gather_Ap=False,
            gather_internal_force=True,
            update_loop_flag=False,
        )

    @qd.kernel
    def add_elements(
        self,
        f: qd.i32,
        mat_idx: qd.i32,
        mat_mu: float,
        mat_lam: float,
        mat_rho: float,
        mat_friction_mu: float,
        v_start: qd.i32,
        el_start: qd.i32,
        s_start: qd.i32,
        verts: qd.types.ndarray(),
        elems: qd.types.ndarray(),
        tri2v: qd.types.ndarray(),
        tri2el: qd.types.ndarray(),
    ):
        n_verts_local = verts.shape[0]
        for i_v, i_b in qd.ndrange(n_verts_local, self._B):
            i_global = i_v + v_start
            for j in qd.static(range(3)):
                self.elements_v[f, i_global, i_b].pos[j] = verts[i_v, j]
            self.elements_v[f, i_global, i_b].vel = qd.Vector.zero(gs.qd_float, 3)

        for i_v in range(n_verts_local):
            i_global = i_v + v_start
            self.elements_v_info[i_global].mass = 0.0
            self.elements_v_info[i_global].mass_over_dt2 = 0.0
            self.elements_v_info[i_global].friction_mu = mat_friction_mu

        dt2_inv = 1.0 / (self.substep_dt**2)
        n_elems_local = elems.shape[0]
        for i_e in range(n_elems_local):
            i_global = i_e + el_start

            a = self.elements_v[f, elems[i_e, 0] + v_start, 0].pos
            b = self.elements_v[f, elems[i_e, 1] + v_start, 0].pos
            c = self.elements_v[f, elems[i_e, 2] + v_start, 0].pos
            d = self.elements_v[f, elems[i_e, 3] + v_start, 0].pos
            B_inv = qd.Matrix.cols([a - d, b - d, c - d])
            self.elements_i[i_global].B = B_inv.inverse()
            det = B_inv.determinant()
            # Determinant should be consistently smaller than 0
            if det >= 0.0:
                self.tet_wrong_order[None] = True
            V = qd.abs(det) / 6.0
            self.elements_i[i_global].V = V

            for j in qd.static(range(4)):
                self.elements_i[i_global].el2v[j] = elems[i_e, j] + v_start
            self.elements_i[i_global].mat_idx = mat_idx
            self.elements_i[i_global].mu = mat_mu
            self.elements_i[i_global].lam = mat_lam
            self.elements_i[i_global].friction_mu = mat_friction_mu
            self.elements_i[i_global].mass_scaled = mat_rho * V * self._vol_scale
            self.elements_i[i_global].muscle_group = 0
            self.elements_i[i_global].muscle_direction = qd.Vector([0.0, 0.0, 1.0], dt=gs.qd_float)

        # The elements add their mass to their vertices in order, so the vertex masses are reproducible bitwise
        qd.loop_config(serialize=True)
        for i_e in range(n_elems_local):
            i_global = i_e + el_start
            mass = 0.25 * mat_rho * self.elements_i[i_global].V
            for j in qd.static(range(4)):
                i_v = self.elements_i[i_global].el2v[j]
                self.elements_v_info[i_v].mass = self.elements_v_info[i_v].mass + mass
                self.elements_v_info[i_v].mass_over_dt2 = self.elements_v_info[i_v].mass_over_dt2 + mass * dt2_inv

        for i_v in range(n_verts_local):
            i_global = i_v + v_start
            self.elements_v_info[i_global].mass_inv = 1.0 / self.elements_v_info[i_global].mass

        for i_e, i_b in qd.ndrange(n_elems_local, self._B):
            i_global = i_e + el_start
            self.elements_el[f, i_global, i_b].actu = 0.0

        for i_s in range(tri2v.shape[0]):
            i_global = i_s + s_start
            for j in qd.static(range(3)):
                self.surface[i_global].tri2v[j] = tri2v[i_s, j] + v_start
            self.surface[i_global].tri2el = tri2el[i_s] + el_start

    @qd.kernel
    def add_cloth(
        self,
        f: qd.i32,
        v_start: qd.i32,
        s_start: qd.i32,
        verts: qd.types.ndarray(),
        tri2v: qd.types.ndarray(),
    ):
        """
        Add cloth vertices and surface triangles to the solver, for position tracking and coupling only.

        Cloth elements and mass are owned by the IPC coupler, so the vertex info holds placeholder values and each
        surface triangle references itself as element.
        """
        n_verts_local = verts.shape[0]
        for i_v, i_b in qd.ndrange(n_verts_local, self._B):
            i_global = i_v + v_start
            for j in qd.static(range(3)):
                self.elements_v[f, i_global, i_b].pos[j] = verts[i_v, j]
            self.elements_v[f, i_global, i_b].vel = qd.Vector.zero(gs.qd_float, 3)

        for i_v in range(n_verts_local):
            i_global = i_v + v_start
            self.elements_v_info[i_global].mass = 1.0
            self.elements_v_info[i_global].mass_over_dt2 = 0.0
            self.elements_v_info[i_global].friction_mu = 0.0

        for i_s in range(tri2v.shape[0]):
            i_global = i_s + s_start
            for j in qd.static(range(3)):
                self.surface[i_global].tri2v[j] = tri2v[i_s, j] + v_start
            self.surface[i_global].tri2el = i_global

    @gs.assert_built
    @mutates(StateChange.GEOMETRY)
    def set_verts_pos(self, poss, verts_idx=None, envs_idx=None):
        """
        Set the position of some vertices.

        Parameters
        ----------
        poss : array_like, shape ([M,] [N,] 3)
            Target position of each vertex.
        verts_idx : None | int | array_like, shape ([M,] N), optional
            Global index of the vertices, once for every environment or one row per environment. If None, all vertices
            will be considered. Defaults to None.
        envs_idx : None | int | array_like, shape (M,), optional
            The indices of the environments to set. If None, all environments will be considered. Defaults to None.
        """
        poss, verts_idx, envs_idx = self._sanitize_io_variables(
            poss, verts_idx, self.n_vertices, "verts_idx", envs_idx, (3,), batched_idx=True
        )
        # Between two steps, the state sits in frame 0 (see 'Solver.has_frames')
        self._kernel_set_verts_pos(0, verts_idx, envs_idx, poss)

    def get_verts_pos(self, verts_idx=None, envs_idx=None):
        """Get the position of some vertices (see 'set_verts_pos')."""
        poss, verts_idx, envs_idx = self._sanitize_io_variables(
            None, verts_idx, self.n_vertices, "verts_idx", envs_idx, (3,), batched_idx=True
        )
        # See 'set_verts_pos'
        self._kernel_get_verts_pos(0, verts_idx, envs_idx, poss)
        return poss[0] if self._scene.n_envs == 0 else poss

    @gs.assert_built
    @mutates(StateChange.DYNAMICS)
    def set_verts_vel(self, vels, verts_idx=None, envs_idx=None):
        """Set the velocity of some vertices (see 'set_verts_pos')."""
        vels, verts_idx, envs_idx = self._sanitize_io_variables(
            vels, verts_idx, self.n_vertices, "verts_idx", envs_idx, (3,), batched_idx=True
        )
        # See 'set_verts_pos'
        self._kernel_set_verts_vel(0, verts_idx, envs_idx, vels)

    @gs.assert_built
    @mutates()
    def set_elements_actu(self, actus, elements_idx=None, envs_idx=None, *, n_groups=1):
        """
        Set the actuation of some elements, each element taking the actuation of its muscle group.

        Parameters
        ----------
        actus : array_like, shape ([M,] [N,] [n_groups,])
            Actuation of each muscle group, for each element.
        elements_idx : None | int | array_like, shape ([M,] N), optional
            Global index of the elements (see 'set_verts_pos'). If None, all elements will be considered.
        envs_idx : None | int | array_like, shape (M,), optional
            The indices of the environments to set. If None, all environments will be considered. Defaults to None.
        n_groups : int, optional
            The number of muscle groups. Defaults to 1.
        """
        actus, elements_idx, envs_idx = self._sanitize_io_variables(
            actus, elements_idx, self.n_elements, "elements_idx", envs_idx, (n_groups,), batched_idx=True
        )
        # See 'set_verts_pos'
        self._kernel_set_elements_actu(0, elements_idx, envs_idx, actus, n_groups)

    @mutates(StateChange.MODEL)
    def set_elements_muscle_group(self, muscle_group, elements_idx=None):
        """Set the muscle group of some elements, shared by every environment."""
        muscle_group, elements_idx, _ = self._sanitize_io_variables(
            muscle_group, elements_idx, self.n_elements, "elements_idx", dtype=gs.tc_int, batched=False
        )
        self._kernel_set_elements_muscle_group(elements_idx, muscle_group)

    @mutates(StateChange.MODEL)
    def set_elements_muscle_direction(self, muscle_direction, elements_idx=None):
        """Set the unit direction of the muscle fibers of some elements, shared by every environment."""
        muscle_direction, elements_idx, _ = self._sanitize_io_variables(
            muscle_direction, elements_idx, self.n_elements, "elements_idx", element_shape=(3,), batched=False
        )
        self._kernel_set_elements_muscle_direction(elements_idx, muscle_direction)

    def get_elements_el2v(self, elements_idx=None):
        """Get the global index of the four vertices of some elements."""
        el2v, elements_idx, _ = self._sanitize_io_variables(
            None, elements_idx, self.n_elements, "elements_idx", element_shape=(4,), dtype=gs.tc_int, batched=False
        )
        self._kernel_get_elements_el2v(elements_idx, el2v)
        return el2v

    @gs.assert_built
    @mutates(StateChange.MODEL)
    def set_vertex_constraints(
        self, verts_idx=None, target_poss=None, link_idx=-1, is_soft_constraint=False, stiffness=0.0, envs_idx=None
    ):
        """
        Constrain some vertices to target positions, or to follow a rigid link.

        Parameters
        ----------
        verts_idx : None | int | array_like, shape ([M,] N), optional
            Global index of the vertices (see 'set_verts_pos'). If None, all vertices will be considered.
        target_poss : None | array_like, shape ([M,] [N,] 3), optional
            Target position of each vertex. If None, the current positions are used. Defaults to None.
        link_idx : int, optional
            Global index of the rigid link the vertices follow, keeping their offset to it, or -1 for none.
        is_soft_constraint : bool, optional
            Whether to pull the vertices toward their targets with critically damped springs, rather than setting their
            position and zeroing their velocity. Defaults to False.
        stiffness : float, optional
            Stiffness of the springs of a soft constraint. Defaults to 0.0.
        envs_idx : None | int | array_like, shape (M,), optional
            The indices of the environments to set. If None, all environments will be considered. Defaults to None.
        """
        if self._use_implicit_solver and not self._enable_vertex_constraints:
            gs.raise_exception(
                "This feature is disabled. Please set 'enable_vertex_constraints=True' when using FEM implicit solver."
            )
        # Imported here to avoid a circular dependency (see 'substep_pre_coupling')
        from genesis.engine.couplers import IPCCoupler

        if isinstance(self._sim.coupler, IPCCoupler):
            gs.raise_exception("Vertex constraints are not supported by the IPC coupler.")
        if not self._constraints_initialized:
            self.init_constraints()

        is_target_current = target_poss is None
        target_poss, verts_idx, envs_idx = self._sanitize_io_variables(
            target_poss, verts_idx, self.n_vertices, "verts_idx", envs_idx, (3,), batched_idx=True
        )
        if is_target_current:
            # See 'set_verts_pos'
            self._kernel_get_verts_pos(0, verts_idx, envs_idx, target_poss)

        if link_idx < 0:
            link_init_pos = torch.zeros((len(envs_idx), 3), dtype=gs.tc_float, device=gs.device)
            link_init_quat = torch.zeros((len(envs_idx), 4), dtype=gs.tc_float, device=gs.device)
        else:
            rigid_solver = self._sim.rigid_solver
            links_envs_idx = envs_idx if self._scene.n_envs > 0 else None
            # The attachment is a configuration of the constraint, which carries no gradient
            with torch.no_grad():
                link_init_pos = rigid_solver.get_links_pos(links_idx=link_idx, envs_idx=links_envs_idx)[..., 0, :]
                link_init_quat = rigid_solver.get_links_quat(links_idx=link_idx, envs_idx=links_envs_idx)[..., 0, :]
            link_init_pos = broadcast_tensor(link_init_pos, gs.tc_float, (len(envs_idx), 3)).contiguous()
            link_init_quat = broadcast_tensor(link_init_quat, gs.tc_float, (len(envs_idx), 4)).contiguous()

        # See 'set_verts_pos'
        self._kernel_set_vertex_constraints(
            0, verts_idx, envs_idx, link_idx, is_soft_constraint, stiffness, target_poss, link_init_pos, link_init_quat
        )

    @mutates(StateChange.MODEL)
    def update_constraint_targets(self, target_poss, verts_idx=None, envs_idx=None):
        """Update the target position of some constrained vertices (see 'set_vertex_constraints')."""
        if target_poss is None:
            gs.raise_exception("The target positions of the constraints must be specified.")
        if not self._constraints_initialized:
            gs.logger.warning("Ignoring update_constraint_targets; constraints have not been initialized.")
            return
        target_poss, verts_idx, envs_idx = self._sanitize_io_variables(
            target_poss, verts_idx, self.n_vertices, "verts_idx", envs_idx, (3,), batched_idx=True
        )
        self._kernel_update_constraint_targets(verts_idx, envs_idx, target_poss)

    @mutates(StateChange.MODEL)
    def remove_vertex_constraints(self, verts_idx=None, envs_idx=None):
        """Remove the constraints of some vertices (see 'set_vertex_constraints'), of all of them if None."""
        if not self._constraints_initialized:
            gs.logger.warning("Ignoring remove_vertex_constraints; constraints have not been initialized.")
            return
        _, verts_idx, envs_idx = self._sanitize_io_variables(
            None, verts_idx, self.n_vertices, "verts_idx", envs_idx, batched_idx=True, skip_allocation=True
        )
        self._kernel_remove_specific_constraints(verts_idx, envs_idx)

    @qd.kernel
    def _kernel_set_verts_pos(
        self, f: int, verts_idx: qd.types.ndarray(), envs_idx: qd.types.ndarray(), poss: qd.types.ndarray()
    ):
        for i_v_, i_b_ in qd.ndrange(verts_idx.shape[1], envs_idx.shape[0]):
            i_v = verts_idx[i_b_, i_v_]
            i_b = envs_idx[i_b_]
            # The earlier occurrences of a repeated vertex are -1 (see 'drop_repeated_index')
            if i_v >= 0:
                for k in qd.static(range(3)):
                    self.elements_v[f, i_v, i_b].pos[k] = poss[i_b_, i_v_, k]

    @qd.kernel
    def _kernel_set_verts_vel(
        self, f: int, verts_idx: qd.types.ndarray(), envs_idx: qd.types.ndarray(), vels: qd.types.ndarray()
    ):
        for i_v_, i_b_ in qd.ndrange(verts_idx.shape[1], envs_idx.shape[0]):
            i_v = verts_idx[i_b_, i_v_]
            i_b = envs_idx[i_b_]
            # See '_kernel_set_verts_pos'
            if i_v >= 0:
                for k in qd.static(range(3)):
                    self.elements_v[f, i_v, i_b].vel[k] = vels[i_b_, i_v_, k]

    @qd.kernel
    def _kernel_get_verts_pos(
        self, f: qd.i32, verts_idx: qd.types.ndarray(), envs_idx: qd.types.ndarray(), poss: qd.types.ndarray()
    ):
        for i_v_, i_b_ in qd.ndrange(verts_idx.shape[1], envs_idx.shape[0]):
            i_v = verts_idx[i_b_, i_v_]
            i_b = envs_idx[i_b_]
            for k in qd.static(range(3)):
                poss[i_b_, i_v_, k] = self.elements_v[f, i_v, i_b].pos[k]

    @qd.kernel
    def _kernel_set_elements_actu(
        self,
        f: qd.i32,
        elements_idx: qd.types.ndarray(),
        envs_idx: qd.types.ndarray(),
        actus: qd.types.ndarray(),  # shape [n_envs, n_elements, n_groups]
        n_groups: qd.i32,
    ):
        for i_e_, j_g, i_b_ in qd.ndrange(elements_idx.shape[1], n_groups, envs_idx.shape[0]):
            i_e = elements_idx[i_b_, i_e_]
            i_b = envs_idx[i_b_]
            # See '_kernel_set_verts_pos'
            if i_e >= 0:
                if self.elements_i[i_e].muscle_group == j_g:
                    self.elements_el[f, i_e, i_b].actu = actus[i_b_, i_e_, j_g]

    @qd.kernel
    def _kernel_set_elements_muscle_group(self, elements_idx: qd.types.ndarray(), muscle_group: qd.types.ndarray()):
        for i_e_ in range(elements_idx.shape[0]):
            i_e = elements_idx[i_e_]
            # See '_kernel_set_verts_pos'
            if i_e >= 0:
                self.elements_i[i_e].muscle_group = muscle_group[i_e_]

    @qd.kernel
    def _kernel_set_elements_muscle_direction(
        self, elements_idx: qd.types.ndarray(), muscle_direction: qd.types.ndarray()
    ):
        for i_e_ in range(elements_idx.shape[0]):
            i_e = elements_idx[i_e_]
            # See '_kernel_set_verts_pos'
            if i_e >= 0:
                for j in qd.static(range(3)):
                    self.elements_i[i_e].muscle_direction[j] = muscle_direction[i_e_, j]

    @qd.kernel
    def _kernel_get_elements_el2v(self, elements_idx: qd.types.ndarray(), el2v: qd.types.ndarray()):
        for i_e_ in range(elements_idx.shape[0]):
            i_e = elements_idx[i_e_]
            for j in qd.static(range(4)):
                el2v[i_e_, j] = self.elements_i[i_e].el2v[j]

    @qd.kernel
    def _kernel_get_state_render(self, f: qd.i32):
        for i_vv, i_b in qd.ndrange(self._n_vverts, self._B):
            i_v = self.vverts_info[i_vv].vert_idx
            for j in qd.static(range(3)):
                self.vverts_render[i_vv, i_b].pos[j] = qd.cast(self.elements_v[f, i_v, i_b].pos[j], qd.f32)

    @qd.kernel
    def add_vverts(
        self,
        vvert_start: qd.i32,
        vface_start: qd.i32,
        v_start: qd.i32,
        verts_idx: qd.types.ndarray(),
        uvs: qd.types.ndarray(element_dim=1),
        vfaces: qd.types.ndarray(element_dim=1),
    ):
        n_vverts_local = verts_idx.shape[0]
        for i_vv_ in range(n_vverts_local):
            self.vverts_info[i_vv_ + vvert_start].vert_idx = verts_idx[i_vv_] + v_start

        n_uvs = uvs.shape[0]
        for i_vv_ in range(n_uvs):
            self.vverts_uvs[i_vv_ + vvert_start] = uvs[i_vv_]

        n_vfaces_local = vfaces.shape[0]
        for i_vf_ in range(n_vfaces_local):
            self.vfaces_indices[i_vf_ + vface_start] = vfaces[i_vf_] + vvert_start

    @qd.kernel
    def _kernel_set_state(
        self,
        f: qd.i32,
        pos: qd.types.ndarray(),  # shape [B, n_vertices, 3]
        vel: qd.types.ndarray(),  # shape [B, n_vertices, 3]
    ):
        for i_v, i_b in qd.ndrange(self.n_vertices, self._B):
            for j in qd.static(range(3)):
                self.elements_v[f, i_v, i_b].pos[j] = pos[i_b, i_v, j]
                self.elements_v[f, i_v, i_b].vel[j] = vel[i_b, i_v, j]

    # ------------------------------------------------------------------------------------
    # ----------------------------------- properties -------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def damping(self):
        return self._damping

    @property
    def n_vertices(self):
        return sum([entity.n_vertices for entity in self._entities])

    @property
    def n_elements(self):
        return sum([entity.n_elements for entity in self._entities])

    @property
    def n_surfaces(self):
        return sum([entity.n_surfaces for entity in self.entities])

    @property
    def n_vverts(self):
        return sum([entity.n_vverts for entity in self._entities])

    @property
    def n_vfaces(self):
        return sum([entity.n_vfaces for entity in self._entities])

    @property
    def n_vertices_max(self):
        return self._n_vertices_max

    @property
    def n_elements_max(self):
        return self._n_elements_max

    @property
    def vol_scale(self):
        return self._vol_scale

    @property
    def n_surface_vertices(self):
        return self.surface_vertices.shape[0]

    @property
    def n_surface_elements(self):
        return self.surface_elements.shape[0]

    # ------------------------------------------------------------------------------------
    # -------------------------------- vertex constraints --------------------------------
    # ------------------------------------------------------------------------------------

    @qd.func
    def _func_get_constraint_target(self, i_v: int, i_b: int, links_state: LinksState) -> qd.types.vector(3):
        target_pos = self.vertex_constraints[i_v, i_b].target_pos
        i_l = self.vertex_constraints[i_v, i_b].link_idx
        if i_l >= 0:
            # The target of a vertex following a link is computed from the pose of the link where it is used, rather than
            # stored, so that the gradient flows from the link to the vertex
            offset_quat = qd_transform_quat_by_quat(
                qd_inv_quat(self.vertex_constraints[i_v, i_b].link_init_quat), links_state.quat[i_l, i_b]
            )
            offset_pos = qd_transform_by_quat(self.vertex_constraints[i_v, i_b].link_offset_pos, offset_quat)
            target_pos = links_state.pos[i_l, i_b] + offset_pos
        return target_pos

    @qd.kernel
    def apply_hard_constraints(self, f: qd.i32, links_state: LinksState):
        """Apply hard constraints by directly overriding positions and velocities."""
        for i_v, i_b in qd.ndrange(self.n_vertices, self._B):
            vc = self.vertex_constraints[i_v, i_b]
            if vc.is_constrained and not vc.is_soft_constraint:
                target_pos = self._func_get_constraint_target(i_v, i_b, links_state)
                self.elements_v[f + 1, i_v, i_b].pos = target_pos
                self.elements_v[f + 1, i_v, i_b].vel = qd.Vector.zero(gs.qd_float, 3)
                if not qd_isfinite(target_pos):
                    self.errno[i_b] = self.errno[i_b] | ErrorCode.INVALID_VERTEX_NAN

    @qd.kernel
    def apply_soft_constraints(self, f: qd.i32, links_state: LinksState):
        """Apply soft constraints as spring forces for explicit solver."""
        for i_v, i_b in qd.ndrange(self.n_vertices, self._B):
            vc = self.vertex_constraints[i_v, i_b]
            if vc.is_constrained and vc.is_soft_constraint:
                target_pos = self._func_get_constraint_target(i_v, i_b, links_state)
                pos_error = self.elements_v[f, i_v, i_b].pos - target_pos
                vel_error = self.elements_v[f + 1, i_v, i_b].vel - self.elements_v[f, i_v, i_b].vel
                spring_force = -vc.stiffness * pos_error
                damping_force = -2.0 * qd.math.sqrt(vc.stiffness) * vel_error

                dv = self.substep_dt * (spring_force + damping_force)
                self.elements_v[f + 1, i_v, i_b].vel += dv

    @qd.kernel
    def _kernel_set_vertex_constraints(
        self,
        f: qd.i32,
        verts_idx: qd.types.ndarray(),  # shape [n_selected_envs, n_verts]
        envs_idx: qd.types.ndarray(),  # shape [n_selected_envs]
        link_idx: qd.i32,
        is_soft_constraint: qd.i32,
        stiffness: float,
        target_poss: qd.types.ndarray(),  # shape [n_selected_envs, n_verts, 3]
        link_init_pos: qd.types.ndarray(),  # shape [n_selected_envs, 3]
        link_init_quat: qd.types.ndarray(),  # shape [n_selected_envs, 4]
    ):
        for i_v_, i_b_ in qd.ndrange(verts_idx.shape[1], envs_idx.shape[0]):
            i_b = envs_idx[i_b_]
            i_v = verts_idx[i_b_, i_v_]
            # The earlier occurrences of a repeated vertex are -1 (see 'drop_repeated_index')
            if i_v >= 0:
                self.vertex_constraints[i_v, i_b].is_constrained = True
                self.vertex_constraints[i_v, i_b].is_soft_constraint = qd.cast(is_soft_constraint, gs.qd_bool)
                self.vertex_constraints[i_v, i_b].stiffness = stiffness
                self.vertex_constraints[i_v, i_b].link_idx = link_idx

                cur_pos = self.elements_v[f, i_v, i_b].pos
                for j in qd.static(range(3)):
                    self.vertex_constraints[i_v, i_b].target_pos[j] = target_poss[i_b_, i_v_, j]
                    self.vertex_constraints[i_v, i_b].link_offset_pos[j] = cur_pos[j] - link_init_pos[i_b_, j]
                for j in qd.static(range(4)):
                    self.vertex_constraints[i_v, i_b].link_init_quat[j] = link_init_quat[i_b_, j]

    @qd.kernel
    def _kernel_update_constraint_targets(
        self, verts_idx: qd.types.ndarray(), envs_idx: qd.types.ndarray(), new_target_poss: qd.types.ndarray()
    ):
        for i_v_, i_b_ in qd.ndrange(verts_idx.shape[1], envs_idx.shape[0]):
            i_b = envs_idx[i_b_]
            i_v = verts_idx[i_b_, i_v_]
            # See '_kernel_set_vertex_constraints'
            if i_v >= 0:
                for j in qd.static(range(3)):
                    self.vertex_constraints[i_v, i_b].target_pos[j] = new_target_poss[i_b_, i_v_, j]

    @qd.kernel
    def _kernel_remove_specific_constraints(self, verts_idx: qd.types.ndarray(), envs_idx: qd.types.ndarray()):
        for i_v_, i_b_ in qd.ndrange(verts_idx.shape[1], envs_idx.shape[0]):
            i_b = envs_idx[i_b_]
            i_v = verts_idx[i_b_, i_v_]
            self.vertex_constraints[i_v, i_b].is_constrained = False

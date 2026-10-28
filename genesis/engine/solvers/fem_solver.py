# pylint: disable=no-value-for-parameter

import numpy as np
import torch

import igl

import quadrants as qd

import genesis as gs
import genesis.utils.array_class as array_class
from genesis.engine.boundaries import FloorBoundary
from genesis.engine.entities.fem_entity import FEMEntity
from genesis.engine.materials import FEM
from genesis.engine.state_change import StateChange, mutates
from genesis.engine.states.solvers import FEMSolverState
from genesis.grad.bridge import no_grad
from genesis.utils.geom import qd_transform_by_quat, qd_transform_quat_by_quat
from genesis.utils.misc import broadcast_tensor, qd_to_torch

from .base_solver import GravityMixin, Solver, TimeBasedMixin


@qd.data_oriented
class FEMSolver(GravityMixin, TimeBasedMixin, Solver):
    material_cls = FEM.Base
    has_frames = True

    # ------------------------------------------------------------------------------------
    # --------------------------------- Initialization -----------------------------------
    # ------------------------------------------------------------------------------------

    def __init__(self, scene, sim, options):
        super().__init__(scene, sim, options)

        # options
        self._floor_height = options.floor_height
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

        # use scaled volume for better numerical stability, similar to p_vol_scale in mpm
        self._vol_scale = float(1e4)

        # materials
        self._mats = list()
        self._mats_idx = list()
        self._mats_update_stress = list()
        self._mats_compute_energy_gradient_hessian = list()
        self._mats_compute_energy = list()

        # boundary
        self.setup_boundary()

        # lazy initialization
        self._constraints_initialized = False

    def setup_boundary(self):
        self.boundary = FloorBoundary(height=self._floor_height)

    def init_batch_fields(self):
        self.batch_active = qd.field(dtype=gs.qd_bool, shape=(self._B,), needs_grad=False)
        self.batch_pcg_active = qd.field(dtype=gs.qd_bool, shape=(self._B,), needs_grad=False)
        self.batch_linesearch_active = qd.field(dtype=gs.qd_bool, shape=(self._B,), needs_grad=False)

        pcg_state = qd.types.struct(
            rTr=gs.qd_float,
            rTz=gs.qd_float,
            rTr_new=gs.qd_float,
            rTz_new=gs.qd_float,
            pTAp=gs.qd_float,
            alpha=gs.qd_float,
            beta=gs.qd_float,
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
            V_scaled=gs.qd_float,  # scaled rest volume of the element
            friction_mu=gs.qd_float,  # friction coefficient for contact
            # for muscle
            muscle_group=gs.qd_int,
            muscle_direction=gs.qd_vec3,
        )

        # element state for energy
        element_state_el_energy = qd.types.struct(
            energy=gs.qd_float,  # energy density for the element
            gradient=gs.qd_mat3,  # gradient density for the element, del energy / del F
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

        self.elements_el_hessian = qd.field(shape=(self._B, 3, 3, self.n_elements), dtype=gs.qd_mat3)

        self.elements_v_energy = element_state_v_energy.field(
            shape=(self._B, self.n_vertices),
            needs_grad=False,
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
        mass = igl.massmatrix(pos_np, surface_vertices_mapping[surface_triangles_np])
        surface_vert_mass_np = mass.diagonal().astype(gs.np_float, copy=False)
        self.surface_vert_mass = qd.field(
            dtype=gs.qd_float,
            shape=(len(surface_vertices_np),),
            needs_grad=False,
        )
        self.surface_vert_mass.from_numpy(surface_vert_mass_np)

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
        self.elements_v.grad.fill(0)
        self.elements_el.grad.fill(0)

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
            if self.tet_wrong_order[None]:
                raise RuntimeError(
                    "The order of vertices in the tetrahedral elements is not correct. "
                    "Please check the input mesh or the FEM solver implementation."
                )

        if self.n_vertices_max > 0 and self._enable_vertex_constraints and not self._constraints_initialized:
            self.init_constraints()

        # Kernels of this solver take the solver itself, so gravity has to be a field for them.
        self._build_gravity(as_field=True)

    @property
    def is_active(self):
        # Counted from the entities rather than from what build recorded, so the answer stands before it: the rate
        # every solver integrates at is settled from the active ones, and that is settled before any of them builds.
        return self.n_elements > 0

    def add_entity(
        self, idx, material, morph, surface, visualize_contact=False, name: str | None = None, desc=None
    ) -> "FEMEntity":
        # add material's update methods if not matching any existing material
        exist = False
        for mat in self._mats:
            if material == mat:
                material.idx = mat.idx
                exist = True
                break
        self._mats.append(material)
        if not exist:
            material.idx = len(self._mats_idx)
            self._mats_idx.append(material.idx)
            self._mats_update_stress.append(material.update_stress)
            self._mats_compute_energy_gradient_hessian.append(material.compute_energy_gradient_hessian)
            self._mats_compute_energy.append(material.compute_energy)

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

    @qd.kernel
    def compute_vel(self, f: qd.i32):
        for i_e, i_b in qd.ndrange(self.n_elements, self._B):
            i_v0, i_v1, i_v2, i_v3 = self.elements_i[i_e].el2v
            pos_v0 = self.elements_v[f, i_v0, i_b].pos
            pos_v1 = self.elements_v[f, i_v1, i_b].pos
            pos_v2 = self.elements_v[f, i_v2, i_b].pos
            pos_v3 = self.elements_v[f, i_v3, i_b].pos
            D = qd.Matrix.cols([pos_v0 - pos_v3, pos_v1 - pos_v3, pos_v2 - pos_v3])

            V_scaled = self.elements_i[i_e].V_scaled
            B = self.elements_i[i_e].B
            F = D @ B
            J = F.determinant()

            stress = qd.Matrix.zero(gs.qd_float, 3, 3)
            for mat_idx in qd.static(self._mats_idx):
                if self.elements_i[i_e].mat_idx == mat_idx:
                    stress = self._mats_update_stress[mat_idx](
                        mu=self.elements_i[i_e].mu,
                        lam=self.elements_i[i_e].lam,
                        J=J,
                        F=F,
                        actu=self.elements_el[f, i_e, i_b].actu,
                        m_dir=self.elements_i[i_e].muscle_direction,
                    )

            verts = self.elements_i[i_e].el2v
            mass_scaled = self.elements_i[i_e].mass_scaled
            H_scaled = -V_scaled * stress @ B.transpose()
            for k in qd.static(range(3)):
                force_scaled = qd.Vector([H_scaled[j, k] for j in range(3)])

                # store so forces can be read out
                self.elements_v_energy[i_b, verts[k]].force = force_scaled

                dv = self.substep_dt * force_scaled / mass_scaled
                self.elements_v[f + 1, verts[k], i_b].vel += dv
                self.elements_v[f + 1, verts[3], i_b].vel -= dv

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
        for i_v, i_b in qd.ndrange(self.n_vertices, self._B):
            self.elements_v[f + 1, i_v, i_b].pos = (
                self.substep_dt * self.elements_v[f + 1, i_v, i_b].vel + self.elements_v[f, i_v, i_b].pos
            )
        # With gradients, the substeps of a step run in successive frames, to which the actuation set in frame 0 carries
        # over
        if qd.static(self._requires_grad):
            for i_e, i_b in qd.ndrange(self.n_elements, self._B):
                self.elements_el[f + 1, i_e, i_b].actu = self.elements_el[f, i_e, i_b].actu

    @qd.kernel
    def precompute_material_data(self, f: qd.i32):
        for i_b, i_e in qd.ndrange(self._B, self.n_elements):
            J, F = self._compute_ele_J_F(f, i_e, i_b)  # use last time step's pos to compute
            for mat_idx in qd.static(self._mats_idx):
                if self.elements_i[i_e].mat_idx == mat_idx:
                    self._mats[mat_idx].pre_compute(J=J, F=F, i_e=i_e, i_b=i_b)

    @qd.kernel
    def init_pos_and_inertia(self, f: qd.i32):
        dt2 = self.substep_dt**2
        for i_v, i_b in qd.ndrange(self.n_vertices, self._B):
            if qd.static(self._enable_vertex_constraints):
                if self.vertex_constraints.is_constrained[i_v, i_b]:
                    self.elements_v[f + 1, i_v, i_b].pos = self.vertex_constraints.target_pos[i_v, i_b]
                    self.elements_v_energy[i_b, i_v].inertia = self.vertex_constraints.target_pos[i_v, i_b]
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

    @qd.kernel
    def compute_ele_hessian_gradient(self, f: qd.i32):
        for i_b, i_e in qd.ndrange(self._B, self.n_elements):
            if not self.batch_active[i_b]:
                continue

            J, F = self._compute_ele_J_F(f + 1, i_e, i_b)

            for mat_idx in qd.static(self._mats_idx):
                if self.elements_i[i_e].mat_idx == mat_idx:
                    if self._mats[mat_idx]._hessian_ready:
                        (
                            self.elements_el_energy[i_b, i_e].energy,
                            self.elements_el_energy[i_b, i_e].gradient,
                        ) = self._mats[mat_idx].compute_energy_gradient(
                            mu=self.elements_i[i_e].mu,
                            lam=self.elements_i[i_e].lam,
                            J=J,
                            F=F,
                            actu=self.elements_el[f, i_e, i_b].actu,
                            m_dir=self.elements_i[i_e].muscle_direction,
                            i_e=i_e,
                            i_b=i_b,
                        )
                    else:
                        (
                            self.elements_el_energy[i_b, i_e].energy,
                            self.elements_el_energy[i_b, i_e].gradient,
                        ) = self._mats[mat_idx].compute_energy_gradient_hessian(
                            mu=self.elements_i[i_e].mu,
                            lam=self.elements_i[i_e].lam,
                            J=J,
                            F=F,
                            actu=self.elements_el[f, i_e, i_b].actu,
                            m_dir=self.elements_i[i_e].muscle_direction,
                            i_e=i_e,
                            i_b=i_b,
                            hessian_field=self.elements_el_hessian,
                        )

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

            for mat_idx in qd.static(self._mats_idx):
                if self.elements_i[i_e].mat_idx == mat_idx:
                    self.elements_el_energy[i_b, i_e].energy = self._mats[mat_idx].compute_energy(
                        mu=self.elements_i[i_e].mu,
                        lam=self.elements_i[i_e].lam,
                        J=J,
                        F=F,
                        actu=self.elements_el[f, i_e, i_b].actu,
                        m_dir=self.elements_i[i_e].muscle_direction,
                        i_e=i_e,
                        i_b=i_b,
                    )

            # add linearized damping energy
            if self._damping_beta > gs.EPS:
                damping_beta_over_dt = self._damping_beta / self._substep_dt
                i_vs = self.elements_i[i_e].el2v
                B = self.elements_i[i_e].B
                S = self._func_compute_element_mapping_matrix(i_vs, B, i_b)

                x_diff = qd.Vector.zero(gs.qd_float, 12)
                for i in qd.static(range(4)):
                    x_diff[i * 3 : i * 3 + 3] = (
                        self.elements_v[f + 1, i_vs[i], i_b].pos - self.elements_v[f, i_vs[i], i_b].pos
                    )
                St_x_diff = qd.Vector.zero(gs.qd_float, 9)
                for i, j in qd.static(qd.ndrange(3, 4)):
                    St_x_diff[i * 3 : i * 3 + 3] += S[j, i] * x_diff[j * 3 : j * 3 + 3]

                H_St_x_diff = qd.Vector.zero(gs.qd_float, 9)
                for i, j in qd.static(qd.ndrange(3, 3)):
                    H_St_x_diff[i * 3 : i * 3 + 3] += (
                        self.elements_el_hessian[i_b, i, j, i_e] @ St_x_diff[j * 3 : j * 3 + 3]
                    )

                self.elements_el_energy[i_b, i_e].energy += 0.5 * damping_beta_over_dt * St_x_diff.dot(H_St_x_diff)

    @qd.kernel
    def accumulate_vertex_force_preconditioner(self, f: qd.i32):
        damping_alpha_dt = self._damping_alpha * self._substep_dt
        damping_alpha_factor = damping_alpha_dt + 1.0
        damping_beta_over_dt = self._damping_beta / self._substep_dt
        damping_beta_factor = damping_beta_over_dt + 1.0
        # inertia
        for i_b, i_v in qd.ndrange(self._B, self.n_vertices):
            if not self.batch_active[i_b]:
                continue
            self.elements_v_energy[i_b, i_v].force = -self.elements_v_info[i_v].mass_over_dt2 * (
                (self.elements_v[f + 1, i_v, i_b].pos - self.elements_v_energy[i_b, i_v].inertia)
                + (self.elements_v[f + 1, i_v, i_b].pos - self.elements_v[f, i_v, i_b].pos) * damping_alpha_dt
            )
            self.pcg_state_v[i_b, i_v].diag3x3 = qd.Matrix.zero(gs.qd_float, 3, 3)
            for i in qd.static(range(3)):
                self.pcg_state_v[i_b, i_v].diag3x3[i, i] = (
                    self.elements_v_info[i_v].mass_over_dt2 * damping_alpha_factor
                )

        # elastic
        for i_b, i_e in qd.ndrange(self._B, self.n_elements):
            if not self.batch_active[i_b]:
                continue
            V = self.elements_i[i_e].V
            B = self.elements_i[i_e].B
            gradient = self.elements_el_energy[i_b, i_e].gradient
            i_vs = self.elements_i[i_e].el2v
            S = self._func_compute_element_mapping_matrix(i_vs, B, i_b)
            force = -V * gradient @ S.transpose()

            # atomic
            for i in qd.static(range(4)):
                self.elements_v_energy[i_b, i_vs[i]].force += force[:, i]

            if self._damping_beta > gs.EPS:
                x_diff = qd.Vector.zero(gs.qd_float, 12)
                for i in qd.static(range(4)):
                    x_diff[i * 3 : i * 3 + 3] = (
                        self.elements_v[f + 1, i_vs[i], i_b].pos - self.elements_v[f, i_vs[i], i_b].pos
                    )
                St_x_diff = qd.Vector.zero(gs.qd_float, 9)
                for i, j in qd.static(qd.ndrange(3, 4)):
                    St_x_diff[i * 3 : i * 3 + 3] += S[j, i] * x_diff[j * 3 : j * 3 + 3]

                H_St_x_diff = qd.Vector.zero(gs.qd_float, 9)
                for i, j in qd.static(qd.ndrange(3, 3)):
                    H_St_x_diff[i * 3 : i * 3 + 3] += (
                        self.elements_el_hessian[i_b, i, j, i_e] @ St_x_diff[j * 3 : j * 3 + 3]
                    )
                S_H_St_x_diff = qd.Vector.zero(gs.qd_float, 12)
                for i, j in qd.static(qd.ndrange(4, 3)):
                    S_H_St_x_diff[i * 3 : i * 3 + 3] += S[i, j] * H_St_x_diff[j * 3 : j * 3 + 3]
                for i in qd.static(range(4)):
                    self.elements_v_energy[i_b, i_vs[i]].force += (
                        -damping_beta_over_dt * V * S_H_St_x_diff[i * 3 : i * 3 + 3]
                    )

            # diagonal 3-by-3 block of hessian
            for k, i, j in qd.ndrange(4, 3, 3):
                self.pcg_state_v[i_b, i_vs[k]].diag3x3 += (
                    V * damping_beta_factor * S[k, i] * S[k, j] * self.elements_el_hessian[i_b, i, j, i_e]
                )

        # inverse
        for i_b, i_v in qd.ndrange(self._B, self.n_vertices):
            if not self.batch_active[i_b]:
                continue
            # Use 3-by-3 diagonal block inverse for preconditioner
            self.pcg_state_v[i_b, i_v].prec = self.pcg_state_v[i_b, i_v].diag3x3.inverse()

            # Other options for preconditioner:
            # Uncomment one of the following lines to test different preconditioners
            # Use identity for preconditioner
            # self.pcg_state_v[i_b, i_v].prec = qd.Matrix.identity(gs.qd_float, 3)

            # Use diagonal for preconditioner
            # self.pcg_state_v[i_b, i_v].prec = qd.Matrix([[1 / self.pcg_state_v[i_b, i_v].diag3x3[0, 0], 0, 0],
            #                                            [0, 1 / self.pcg_state_v[i_b, i_v].diag3x3[1, 1], 0],
            #                                            [0, 0, 1 / self.pcg_state_v[i_b, i_v].diag3x3[2, 2]]])

    @qd.func
    def compute_Ap(self):
        damping_alpha_dt = self._damping_alpha * self._substep_dt
        damping_alpha_factor = damping_alpha_dt + 1.0
        damping_beta_over_dt = self._damping_beta / self._substep_dt
        damping_beta_factor = damping_beta_over_dt + 1.0
        for i_b, i_v in qd.ndrange(self._B, self.n_vertices):
            if not self.batch_pcg_active[i_b]:
                continue
            self.pcg_state_v[i_b, i_v].Ap = (
                self.elements_v_info[i_v].mass_over_dt2 * damping_alpha_factor * self.pcg_state_v[i_b, i_v].p
            )

        for i_b, i_e in qd.ndrange(self._B, self.n_elements):
            if not self.batch_pcg_active[i_b]:
                continue
            V = self.elements_i[i_e].V
            B = self.elements_i[i_e].B
            i_vs = self.elements_i[i_e].el2v
            S = self._func_compute_element_mapping_matrix(i_vs, B, i_b)

            p9 = qd.Vector([0.0] * 9, dt=gs.qd_float)

            for i, j in qd.static(qd.ndrange(3, 4)):
                p9[i * 3 : i * 3 + 3] = p9[i * 3 : i * 3 + 3] + S[j, i] * self.pcg_state_v[i_b, i_vs[j]].p

            new_p9 = qd.Vector([0.0] * 9, dt=gs.qd_float)

            for i, j in qd.static(qd.ndrange(3, 3)):
                new_p9[i * 3 : i * 3 + 3] = (
                    new_p9[i * 3 : i * 3 + 3] + self.elements_el_hessian[i_b, i, j, i_e] @ p9[j * 3 : j * 3 + 3]
                )

            # atomic
            for i in qd.static(range(4)):
                self.pcg_state_v[i_b, i_vs[i]].Ap += (
                    (S[i, 0] * new_p9[0:3] + S[i, 1] * new_p9[3:6] + S[i, 2] * new_p9[6:9]) * V * damping_beta_factor
                )

    @qd.kernel
    def init_pcg_solve(self):
        for i_b in range(self._B):
            self.batch_pcg_active[i_b] = self.batch_active[i_b]
            if not self.batch_pcg_active[i_b]:
                continue
            self.pcg_state[i_b].rTr = 0.0
            self.pcg_state[i_b].rTz = 0.0
        for i_b, i_v in qd.ndrange(self._B, self.n_vertices):
            if not self.batch_pcg_active[i_b]:
                continue
            self.pcg_state_v[i_b, i_v].x = 0
            self.pcg_state_v[i_b, i_v].r = self.elements_v_energy[i_b, i_v].force
            self.pcg_state_v[i_b, i_v].z = self.pcg_state_v[i_b, i_v].prec @ self.pcg_state_v[i_b, i_v].r
            self.pcg_state_v[i_b, i_v].p = self.pcg_state_v[i_b, i_v].z
            qd.atomic_add(self.pcg_state[i_b].rTr, self.pcg_state_v[i_b, i_v].r.dot(self.pcg_state_v[i_b, i_v].r))
            qd.atomic_add(self.pcg_state[i_b].rTz, self.pcg_state_v[i_b, i_v].r.dot(self.pcg_state_v[i_b, i_v].z))
        for i_b in range(self._B):
            if not self.batch_pcg_active[i_b]:
                continue
            self.batch_pcg_active[i_b] = self.pcg_state[i_b].rTr > self._pcg_threshold

    @qd.kernel
    def one_pcg_iter(self):
        self.compute_Ap()

        # compute pTAp
        for i_b in range(self._B):
            if not self.batch_pcg_active[i_b]:
                continue
            self.pcg_state[i_b].pTAp = 0.0
        for i_b, i_v in qd.ndrange(self._B, self.n_vertices):
            if not self.batch_pcg_active[i_b]:
                continue
            qd.atomic_add(self.pcg_state[i_b].pTAp, self.pcg_state_v[i_b, i_v].p.dot(self.pcg_state_v[i_b, i_v].Ap))

        # compute alpha and update x, r, z, rTr, rTz
        for i_b in range(self._B):
            if not self.batch_pcg_active[i_b]:
                continue
            self.pcg_state[i_b].alpha = self.pcg_state[i_b].rTz / self.pcg_state[i_b].pTAp
            self.pcg_state[i_b].rTr_new = 0.0
            self.pcg_state[i_b].rTz_new = 0.0
        for i_b, i_v in qd.ndrange(self._B, self.n_vertices):
            if not self.batch_pcg_active[i_b]:
                continue
            self.pcg_state_v[i_b, i_v].x += self.pcg_state[i_b].alpha * self.pcg_state_v[i_b, i_v].p
            self.pcg_state_v[i_b, i_v].r -= self.pcg_state[i_b].alpha * self.pcg_state_v[i_b, i_v].Ap
            self.pcg_state_v[i_b, i_v].z = self.pcg_state_v[i_b, i_v].prec @ self.pcg_state_v[i_b, i_v].r
            qd.atomic_add(self.pcg_state[i_b].rTr_new, self.pcg_state_v[i_b, i_v].r.dot(self.pcg_state_v[i_b, i_v].r))
            qd.atomic_add(self.pcg_state[i_b].rTz_new, self.pcg_state_v[i_b, i_v].r.dot(self.pcg_state_v[i_b, i_v].z))

        # check convergence
        for i_b in range(self._B):
            if not self.batch_pcg_active[i_b]:
                continue
            self.batch_pcg_active[i_b] = self.pcg_state[i_b].rTr_new > self._pcg_threshold

        # update beta, rTr, rTz
        for i_b in range(self._B):
            if not self.batch_pcg_active[i_b]:
                continue
            self.pcg_state[i_b].beta = self.pcg_state[i_b].rTz_new / self.pcg_state[i_b].rTz
            self.pcg_state[i_b].rTr = self.pcg_state[i_b].rTr_new
            self.pcg_state[i_b].rTz = self.pcg_state[i_b].rTz_new

        # update p
        for i_b, i_v in qd.ndrange(self._B, self.n_vertices):
            if not self.batch_pcg_active[i_b]:
                continue
            self.pcg_state_v[i_b, i_v].p = (
                self.pcg_state_v[i_b, i_v].z + self.pcg_state[i_b].beta * self.pcg_state_v[i_b, i_v].p
            )

    def pcg_solve(self):
        self.init_pcg_solve()
        for i in range(self._n_pcg_iterations):
            self.one_pcg_iter()

    @qd.kernel
    def init_linesearch(self, f: qd.i32):
        for i_b in range(self._B):
            self.batch_linesearch_active[i_b] = self.batch_active[i_b]
            if not self.batch_linesearch_active[i_b]:
                continue
            self.linesearch_state[i_b].prev_energy = 0.0
            self.linesearch_state[i_b].step_size = 1.0
            self.linesearch_state[i_b].m = 0.0

        # Inertia, x_prev, m
        for i_b, i_v in qd.ndrange(self._B, self.n_vertices):
            if not self.batch_linesearch_active[i_b]:
                continue
            diff = self.elements_v[f + 1, i_v, i_b].pos - self.elements_v_energy[i_b, i_v].inertia
            self.linesearch_state[i_b].prev_energy += 0.5 * self.elements_v_info[i_v].mass_over_dt2 * diff.dot(diff)
            self.linesearch_state_v[i_b, i_v].x_prev = self.elements_v[f + 1, i_v, i_b].pos
            self.linesearch_state[i_b].m -= self.pcg_state_v[i_b, i_v].x.dot(self.elements_v_energy[i_b, i_v].force)
        # Elastic
        for i_b, i_e in qd.ndrange(self._B, self.n_elements):
            if not self.batch_linesearch_active[i_b]:
                continue
            self.linesearch_state[i_b].prev_energy += self.elements_el_energy[i_b, i_e].energy * self.elements_i[i_e].V

    @qd.kernel
    def one_linesearch_iter(self, f: qd.i32):
        for i_b in range(self._B):
            if not self.batch_linesearch_active[i_b]:
                continue
            self.linesearch_state[i_b].energy = 0.0

        # update pos and compute Inertia energy
        for i_b, i_v in qd.ndrange(self._B, self.n_vertices):
            if not self.batch_linesearch_active[i_b]:
                continue
            self.elements_v[f + 1, i_v, i_b].pos = (
                self.linesearch_state_v[i_b, i_v].x_prev
                + self.linesearch_state[i_b].step_size * self.pcg_state_v[i_b, i_v].x
            )
            diff = self.elements_v[f + 1, i_v, i_b].pos - self.elements_v_energy[i_b, i_v].inertia
            self.linesearch_state[i_b].energy += 0.5 * self.elements_v_info[i_v].mass_over_dt2 * diff.dot(diff)
            # damping
            if self._damping_alpha > 0.0:
                damping_alpha_dt = self._damping_alpha * self._substep_dt
                diff = self.elements_v[f + 1, i_v, i_b].pos - self.elements_v[f, i_v, i_b].pos
                self.linesearch_state[i_b].energy += (
                    0.5 * self.elements_v_info[i_v].mass_over_dt2 * diff.dot(diff) * damping_alpha_dt
                )

        # compute elastic energy
        self._func_compute_ele_energy(f)
        for i_b, i_e in qd.ndrange(self._B, self.n_elements):
            if not self.batch_linesearch_active[i_b]:
                continue
            self.linesearch_state[i_b].energy += self.elements_el_energy[i_b, i_e].energy * self.elements_i[i_e].V

        # check condition
        for i_b in range(self._B):
            if not self.batch_linesearch_active[i_b]:
                continue
            self.batch_linesearch_active[i_b] = (
                self.linesearch_state[i_b].energy
                > self.linesearch_state[i_b].prev_energy
                + self._linesearch_c * self.linesearch_state[i_b].step_size * self.linesearch_state[i_b].m
            )
            if not self.batch_linesearch_active[i_b]:
                continue
            self.linesearch_state[i_b].step_size *= self._linesearch_tau

    @qd.kernel
    def skip_linesearch(self, f: qd.i32):
        # Inertia, x_prev, m
        for i_b, i_v in qd.ndrange(self._B, self.n_vertices):
            if not self.batch_active[i_b]:
                continue
            self.elements_v[f + 1, i_v, i_b].pos = self.elements_v[f + 1, i_v, i_b].pos + self.pcg_state_v[i_b, i_v].x

    def linesearch(self, f: qd.i32):
        """
        Note
        ------
        https://en.wikipedia.org/wiki/Backtracking_line_search#Algorithm
        """
        if self._n_linesearch_iterations <= 0:
            self.skip_linesearch(f)
            return
        self.init_linesearch(f)
        for i in range(self._n_linesearch_iterations):
            self.one_linesearch_iter(f)

    def batch_solve(self, f: qd.i32):
        self.batch_active.fill(True)

        for i in range(self._n_newton_iterations):
            # compute element energy and gradient
            self.compute_ele_hessian_gradient(f)

            # If the hessian is invariant, we only need to compute it once
            for mat_idx in self._mats_idx:
                if self._mats[mat_idx].hessian_invariant:
                    self._mats[mat_idx]._hessian_ready = True

            # accumulate vertex force and preconditioner
            self.accumulate_vertex_force_preconditioner(f)

            # solve for the vertex positions
            self.pcg_solve()

            # line search
            self.linesearch(f)

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
                self.init_pos_and_inertia(f)
                self.batch_solve(f)
                self.setup_pos_vel(f)
            else:
                self.init_pos_and_vel(f)
                self.compute_vel(f)
                self.apply_uniform_force(f)
                if self._constraints_initialized:
                    self.apply_soft_constraints(f)

    def substep_post_coupling(self, f):
        if self.is_active:
            self.compute_pos(f)
            if self._constraints_initialized and not self._use_implicit_solver:
                self.apply_hard_constraints(f)

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
    def is_differentiable(self) -> bool:
        return not self._use_implicit_solver

    @property
    def data(self):
        """Yield the state of the elements, the frame 0 of which a boundary checkpoint holds (see 'Solver.data')."""
        for name, field in (
            ("elements_v.pos", self.elements_v.pos),
            ("elements_v.vel", self.elements_v.vel),
            ("elements_el.actu", self.elements_el.actu),
        ):
            yield array_class.ArrayItem(name, array_class.DataKind.STATE, field)

    # ------------------------------------------------------------------------------------
    # --------------------------------------- io -----------------------------------------
    # ------------------------------------------------------------------------------------

    @mutates()
    def set_state(self, f, state, envs_idx=None):
        if self.is_active:
            self._kernel_set_state(f, state.pos, state.vel)

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
        Get forces on all vertices.

        Returns:
            torch.Tensor : shape (B, n_vertices, 3) where B is batch size
        """
        if not self.is_active:
            return None

        return qd_to_torch(self.elements_v_energy.force, copy=True)

    @qd.kernel
    def add_elements(
        self,
        f: qd.i32,
        mat_idx: qd.i32,
        mat_mu: qd.f32,
        mat_lam: qd.f32,
        mat_rho: qd.f32,
        mat_friction_mu: qd.f32,
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
            V_scaled = V * self._vol_scale
            self.elements_i[i_global].V_scaled = V_scaled

            for j in qd.static(range(4)):
                self.elements_i[i_global].el2v[j] = elems[i_e, j] + v_start
            self.elements_i[i_global].mat_idx = mat_idx
            self.elements_i[i_global].mu = mat_mu
            self.elements_i[i_global].lam = mat_lam
            self.elements_i[i_global].friction_mu = mat_friction_mu
            self.elements_i[i_global].mass_scaled = mat_rho * V_scaled
            for j in qd.static(range(4)):
                mass = 0.25 * mat_rho * V
                self.elements_v_info[self.elements_i[i_global].el2v[j]].mass += mass
                self.elements_v_info[self.elements_i[i_global].el2v[j]].mass_over_dt2 += mass * dt2_inv
            self.elements_i[i_global].muscle_group = 0
            self.elements_i[i_global].muscle_direction = qd.Vector([0.0, 0.0, 1.0], dt=gs.qd_float)

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
        # FIXME: Quadrants 'fill' method is very inefficient. Try using zero-copy if possible.
        if verts_idx is None and envs_idx is None:
            self.vertex_constraints.is_constrained.fill(0)
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
    def floor_height(self):
        return self._floor_height

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

    @qd.kernel
    def _kernel_update_linked_vertex_constraints(
        self,
        links_state: array_class.LinksState,
    ):
        for i_v, i_b in qd.ndrange(self.n_vertices, self._B):
            vc = self.vertex_constraints[i_v, i_b]
            if vc.is_constrained and vc.link_idx >= 0:
                i_l = vc.link_idx
                pos = links_state.pos[i_l, i_b]
                quat = links_state.quat[i_l, i_b]

                offset_pos = vc.link_offset_pos
                offset_quat = qd_transform_quat_by_quat(vc.link_init_quat, quat)
                self.vertex_constraints[i_v, i_b].target_pos = pos + qd_transform_by_quat(offset_pos, offset_quat)

    @no_grad
    @qd.kernel
    def apply_hard_constraints(self, f: qd.i32):
        """Apply hard constraints by directly overriding positions and velocities."""
        for i_v, i_b in qd.ndrange(self.n_vertices, self._B):
            vc = self.vertex_constraints[i_v, i_b]
            if vc.is_constrained and not vc.is_soft_constraint:
                self.elements_v[f + 1, i_v, i_b].pos = vc.target_pos
                self.elements_v[f + 1, i_v, i_b].vel.fill(0.0)

    @no_grad
    @qd.kernel
    def apply_soft_constraints(self, f: qd.i32):
        """Apply soft constraints as spring forces for explicit solver."""
        for i_v, i_b in qd.ndrange(self.n_vertices, self._B):
            vc = self.vertex_constraints[i_v, i_b]
            if vc.is_constrained and vc.is_soft_constraint:
                pos_error = self.elements_v[f, i_v, i_b].pos - vc.target_pos
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
        stiffness: qd.f32,
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

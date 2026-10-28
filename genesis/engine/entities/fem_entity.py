from functools import wraps
from pathlib import Path

import numpy as np
import torch

import igl
import trimesh

import quadrants as qd

import genesis as gs
import genesis.utils.element as eu
import genesis.utils.geom as gu
import genesis.utils.mesh as mu
from genesis.engine.couplers import SAPCoupler
from genesis.engine.entities.rigid_entity import RigidLink
from genesis.engine.states.entities import FEMEntityState
from genesis.options.materials import Cloth, Muscle
from genesis.repr_base import RBC
from genesis.utils.misc import broadcast_tensor, tensor_to_array, unique_rows

from .base_entity import Entity


class FEMVisGeom(RBC):
    """A visual geom of a FEM entity, the deformable counterpart of `RigidVisGeom`.

    It carries the render mesh drawn by the visualizer, decoupled from the simulation mesh, and 'sim_verts_idx',
    the map from each render-mesh vertex to the simulated vertex standing for it (vertices co-located across visual
    geoms or duplicated by texture seams share a single simulated vertex).
    """

    def __init__(self, entity, vvert_start, vface_start, vmesh, sim_verts_idx):
        self._uid = gs.UID()
        self._entity = entity
        self._vvert_start = vvert_start
        self._vface_start = vface_start
        self._vmesh = vmesh
        self._sim_verts_idx = sim_verts_idx

    def get_trimesh(self):
        """The underlying `trimesh.Trimesh` of the render mesh."""
        return self._vmesh.trimesh

    @property
    def uid(self):
        """Unique ID of the vgeom."""
        return self._uid

    @property
    def entity(self):
        """The FEM entity the vgeom belongs to."""
        return self._entity

    @property
    def vmesh(self):
        """The render mesh."""
        return self._vmesh

    @property
    def sim_verts_idx(self):
        """Map from render-mesh vertex index to the entity's simulated vertex index."""
        return self._sim_verts_idx

    @property
    def surface(self):
        """Surface object of the vgeom."""
        return self._vmesh.surface

    @property
    def uvs(self):
        """UV coordinates of the vgeom."""
        return self._vmesh.uvs

    @property
    def metadata(self):
        """Metadata of the render mesh."""
        return self._vmesh.metadata

    @property
    def n_vverts(self):
        """Number of render vertices of the vgeom."""
        return len(self._vmesh.verts)

    @property
    def n_vfaces(self):
        """Number of render faces of the vgeom."""
        return len(self._vmesh.faces)

    @property
    def vvert_start(self):
        """Starting index of the vgeom's render vertices in the FEM solver."""
        return self._vvert_start

    @property
    def vface_start(self):
        """Starting index of the vgeom's render faces in the FEM solver."""
        return self._vface_start

    @property
    def vvert_end(self):
        """Ending index of the vgeom's render vertices in the FEM solver."""
        return self._vvert_start + self.n_vverts

    @property
    def vface_end(self):
        """Ending index of the vgeom's render faces in the FEM solver."""
        return self._vface_start + self.n_vfaces


def assert_muscle(method):
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        if not isinstance(self.material, Muscle):
            gs.raise_exception("This method is only supported by entities with a 'Muscle' material.")
        return method(self, *args, **kwargs)

    return wrapper


@qd.data_oriented
class FEMEntity(Entity):
    """
    A finite element method (FEM)-based entity for deformable simulation.

    This class represents a deformable object using tetrahedral elements. It interfaces with
    the physics solver to handle state updates, checkpointing, gradients, and actuation
    for physics-based simulation in batched environments.

    Parameters
    ----------
    scene : Scene
        The simulation scene that this entity belongs to.
    solver : Solver
        The physics solver instance used for simulation.
    material : gs.materials.Elastic | gs.materials.Cloth | gs.materials.Muscle
        The physical model of the entity and its parameters, the defaults of which its options resolve.
    morph : Morph
        The morph specification that defines the entity's shape.
    surface : Surface
        The surface mesh associated with the entity (for rendering or collision).
    idx : int
        Unique identifier of the entity within the scene.
    model_idx : int
        Index of the constitutive model of the entity in the solver.
    v_start : int, optional
        Starting index of this entity's vertices in the global vertex array (default is 0).
    el_start : int, optional
        Starting index of this entity's elements in the global element array (default is 0).
    s_start : int, optional
        Starting index of this entity's surface triangles in the global surface array (default is 0).
    vvert_start : int, optional
        Starting index of this entity's render vertices in the global render vertex array (default is 0).
    vface_start : int, optional
        Starting index of this entity's render faces in the global render face array (default is 0).
    """

    def __init__(
        self,
        scene,
        solver,
        material,
        morph,
        surface,
        idx,
        model_idx,
        v_start=0,
        el_start=0,
        s_start=0,
        vvert_start=0,
        vface_start=0,
        name: str | None = None,
        *,
        options,
    ):
        super().__init__(idx, scene, morph, solver, material, surface, name=name, options=options)

        self._model_idx = model_idx
        self._v_start = v_start  # offset for vertex index of elements
        self._el_start = el_start  # offset for element index
        self._s_start = s_start  # offset for surface triangles
        self._vvert_start = vvert_start  # offset for render vertices
        self._vface_start = vface_start  # offset for render faces
        self._step_global_added = None
        self.sample()

        if isinstance(self.material, Cloth):
            # For cloth, elements are already surface triangles
            self._surface_tri_np = self.elems
            self._n_surfaces = len(self._surface_tri_np)
            if self._n_surfaces > 0:
                self._n_surface_vertices = len(np.unique(self._surface_tri_np))
            else:
                self._n_surface_vertices = 0
            # For cloth, each triangle is its own "element"
            self._surface_el_np = np.arange(self.elems.shape[0], dtype=gs.np_int)
        else:
            # For volumetric FEM, extract surface triangles from tetrahedral elements
            el2tri = np.array(
                [  # follow the order with correct normal
                    [[v[0], v[2], v[1]], [v[1], v[2], v[3]], [v[0], v[1], v[3]], [v[0], v[3], v[2]]] for v in self.elems
                ],
                dtype=gs.np_int,
            )
            all_tri = el2tri.reshape((-1, 3))
            all_tri_sorted = np.sort(all_tri, axis=1)
            _, unique_idcs, _, cnt = unique_rows(all_tri_sorted)
            unique_tri = all_tri[unique_idcs]
            surface_tri = unique_tri[cnt == 1]

            self._surface_tri_np = surface_tri
            self._n_surfaces = len(self._surface_tri_np)

            if self._n_surfaces > 0:
                self._n_surface_vertices = len(np.unique(self._surface_tri_np))
            else:
                self._n_surface_vertices = 0

            tri2el = np.repeat(np.arange(self.elems.shape[0], dtype=gs.np_int)[:, np.newaxis], 4, axis=1)
            unique_el = tri2el.flat[unique_idcs]
            self._surface_el_np = unique_el[cnt == 1]

        if isinstance(self.sim.coupler, SAPCoupler):
            self.compute_pressure_field()

    # ------------------------------------------------------------------------------------
    # ----------------------------------- basic entity ops -------------------------------
    # ------------------------------------------------------------------------------------

    def set_position(self, pos):
        """
        Set the target position(s) for the FEM entity.

        Parameters
        ----------
        pos : torch.Tensor or array-like
            The desired position(s). Can be:
            - (3,): a single COM offset vector.
            - (n_vertices, 3): per-vertex positions for all vertices.
            - (n_envs, 3): per-environment COM offsets.
            - (n_envs, n_vertices, 3): full batched per-vertex positions.

        Raises
        ------
        Exception
            If the tensor shape is not supported.
        """
        # A value without one entry per vertex places the center of mass, each vertex keeping its initial offset
        shape = np.shape(pos)
        if len(shape) == 1 or (len(shape) == 2 and shape[0] != self.n_vertices):
            pos = self.init_positions_COM_offset + broadcast_tensor(pos, gs.tc_float, (*shape[:-1], 1, 3))
        self.set_verts_pos(pos)

    def set_velocity(self, vel):
        """
        Set the target velocity(ies) for the FEM entity.

        Parameters
        ----------
        vel : torch.Tensor or array-like
            The desired velocity(ies). Can be:
            - (3,): a global velocity vector for all vertices.
            - (n_vertices, 3): per-vertex velocities.
            - (n_envs, 3): per-environment velocities broadcast to all vertices.
            - (n_envs, n_vertices, 3): full batched per-vertex velocities.

        Raises
        ------
        Exception
            If the tensor shape is not supported.
        """
        self.set_verts_vel(vel)

    @assert_muscle
    def set_actuation(self, actu):
        """
        Set the actuation signal for the FEM entity.

        Parameters
        ----------
        actu : torch.Tensor or array-like
            The actuation tensor. Can be:
            - (): a single scalar for all groups.
            - (n_groups,): group-level actuation.
            - (n_envs, n_groups): batch of group-level actuation signals.

        Raises
        ------
        Exception
            If the tensor shape is not supported or per-element actuation is attempted.
        """
        self._assert_active()

        actu = to_gs_tensor(actu)

        is_valid = False
        n_groups = self._options.solver_options.n_groups
        if actu.ndim == 0:
            self._tgt["actu"] = actu.tile((self._sim._B, n_groups))
            is_valid = True
        elif actu.ndim == 1:
            if actu.shape == (n_groups,):
                self._tgt["actu"] = actu[None].tile((self._sim._B, 1))
                is_valid = True
            elif actu.shape == (self.n_elements,):
                gs.raise_exception("Cannot set per-element actuation.")
        elif actu.ndim == 2:
            if actu.shape == (self._sim._B, n_groups):
                self._tgt["actu"] = actu
                is_valid = True
        if not is_valid:
            gs.raise_exception("Tensor shape not supported.")

    def set_muscle(self, muscle_group=None, muscle_direction=None):
        """
        Set the muscle group and/or muscle direction for the FEM entity.

        Parameters
        ----------
        muscle_group : torch.Tensor or array-like, optional
            Tensor of shape (n_elements,) specifying the muscle group ID for each element.

        muscle_direction : torch.Tensor or array-like, optional
            Tensor of shape (n_elements, 3) specifying unit direction vectors for muscle forces.

        Raises
        ------
        AssertionError
            If tensor shapes are incorrect or normalization fails.
        """
        self._assert_active()

        n_groups = self._options.solver_options.n_groups
        max_group_id = muscle_group.max().item()

        muscle_group = broadcast_tensor(muscle_group, gs.tc_int, (self.n_elements,), ("elements_idx",))
        assert isinstance(max_group_id, int) and max_group_id < n_groups

        self.set_muscle_group(muscle_group)

        if muscle_direction is not None:
            muscle_direction = broadcast_tensor(
                muscle_direction, gs.tc_float, (self.n_elements, 3), ("elements_idx", "")
            )
            assert ((1.0 - muscle_direction.norm(dim=-1)).abs() < gs.EPS).all()

            self.set_muscle_direction(muscle_direction)

    def get_state(self):
        solver_state = self._solver.get_state(self._sim.cur_substep_local)
        verts_slice = slice(self._v_start, self._v_start + self.n_vertices)
        return FEMEntityState(
            self, self._sim.cur_step_global, solver_state.pos[:, verts_slice], solver_state.vel[:, verts_slice]
        )

    # ------------------------------------------------------------------------------------
    # ----------------------------------- instantiation ----------------------------------
    # ------------------------------------------------------------------------------------

    def instantiate(self, verts, elems):
        """
        Initialize FEM entity with given vertices and elements.

        Parameters
        ----------
        verts : np.ndarray
            Array of vertex positions with shape (n_vertices, 3).

        elems : np.ndarray
            Array of elements indexing into verts: tetrahedra with shape (n_elements, 4), or surface triangles with
            shape (n_elements, 3) for Cloth material.

        Raises
        ------
        Exception
            If no vertices are provided.
        """
        verts = verts.astype(gs.np_float, copy=False)
        elems = elems.astype(gs.np_int, copy=False)

        # Compose the morph pose offset (e.g. an up-axis conversion) onto the morph orientation, rotating the verts
        # about their COM (the pre-existing morph.quat convention), then translate by the body-frame offset position
        # R(morph.quat) @ offset_pos. NB: pivoting the orientation about the vertex COM differs from the rigid
        # parent-child composition when the mesh COM is not at the morph origin.
        morph_quat = np.array(self._morph.quat, dtype=gs.np_float)
        init_quat = gu.transform_quat_by_quat(np.array(self._morph.offset_quat, dtype=gs.np_float), morph_quat)
        R = gu.quat_to_R(init_quat)
        verts_COM = verts.mean(axis=0)
        init_positions = (verts - verts_COM) @ R.T + verts_COM
        offset_shift = gu.transform_by_quat(np.array(self._morph.offset_pos, dtype=gs.np_float), morph_quat)
        init_positions = init_positions + offset_shift

        if not init_positions.shape[0] > 0:
            gs.raise_exception("Entity has zero vertices.")

        # The vertex options select vertices as the morph states them, relative to its position before its rotation
        self._fixed_verts_idx_local = self._options.get_fixed_verts_idx(verts - self._morph.pos)

        self.init_positions = torch.tensor(init_positions, dtype=gs.tc_float, device=gs.device)
        self.init_positions_COM_offset = torch.tensor(
            init_positions - (verts_COM + offset_shift), dtype=gs.tc_float, device=gs.device
        )

        self.elems = elems

    def sample(self):
        """
        Build the entity's visual geoms and simulation mesh from its morph.

        Each morph sub-mesh becomes a visual geom with its own surface and UVs, while the simulation operates on a
        single welded copy of their vertices, tracked through 'FEMVisGeom.sim_verts_idx': welding and
        tetrahedralization both keep the input vertices first and in order, so these maps remain valid indices into
        the simulated vertices.
        """
        meshes = gs.Mesh.from_morph_surface(self._morph, self._surface)
        surface_verts, surface_faces, verts_maps = mu.merge_submeshes(
            [mesh.verts for mesh in meshes], [mesh.faces for mesh in meshes]
        )
        self._vgeoms = gs.List()
        vvert_start, vface_start = self._vvert_start, self._vface_start
        for mesh, verts_idx in zip(meshes, verts_maps):
            self._vgeoms.append(
                FEMVisGeom(
                    entity=self,
                    vvert_start=vvert_start,
                    vface_start=vface_start,
                    vmesh=mesh,
                    sim_verts_idx=verts_idx,
                )
            )
            vvert_start += len(mesh.verts)
            vface_start += len(mesh.faces)

        if isinstance(self.material, Cloth):
            # Cloth needs no tetrahedralization: the welded surface triangles are the simulation elements.
            verts = surface_verts + self._morph.pos
            elems = surface_faces
        else:
            # Tetgen refinement depends on the absolute coordinates of its input. File meshes are tetrahedralized
            # untranslated so the result, and its on-disk cache, are shared across all placements of the same asset;
            # primitives keep the position baked in, as the simulated rest state is sensitive to the exact refinement.
            is_mesh_morph = isinstance(self._morph, gs.options.morphs.Mesh)
            if not is_mesh_morph:
                surface_verts = surface_verts + self._morph.pos
            surface_trimesh = trimesh.Trimesh(vertices=surface_verts, faces=surface_faces, process=False)
            verts, elems = eu.mesh_to_elements(surface_trimesh, tet_cfg=self.tet_cfg)
            if is_mesh_morph:
                verts = verts + self._morph.pos
            verts, elems = eu.split_all_surface_tets(verts, elems)

        self.instantiate(verts, elems)

    def _add_to_solver(self):
        self._step_global_added = self._sim.cur_step_global
        gs.logger.info(
            f"Entity {self.uid} added. class: {self.__class__.__name__}, morph: {self.morph.__class__.__name__}, "
            f"size: ({self.n_elements}, {self.n_vertices}), material: {self.material}."
        )

        # Convert to appropriate numpy array types
        verts_numpy = tensor_to_array(self.init_positions, dtype=gs.np_float)

        if isinstance(self.material, Cloth):
            self._solver._kernel_add_cloth(
                f=self._sim.cur_substep_local,
                v_start=self._v_start,
                s_start=self._s_start,
                verts=verts_numpy,
                tri2v=self._surface_tri_np,
            )
        else:
            elems_np = self.elems.astype(gs.np_int, copy=False)
            material_params = self._options.material_params
            mat_mu, mat_lam = eu.lame_parameters(material_params.youngs_modulus, material_params.poisson_ratio)
            self._solver._kernel_add_elements(
                f=self._sim.cur_substep_local,
                mat_idx=self._model_idx,
                mat_mu=mat_mu,
                mat_lam=mat_lam,
                mat_rho=material_params.density,
                mat_friction_mu=material_params.friction,
                v_start=self._v_start,
                el_start=self._el_start,
                s_start=self._s_start,
                verts=verts_numpy,
                elems=elems_np,
                tri2v=self._surface_tri_np,
                tri2el=self._surface_el_np,
            )

        for vgeom in self._vgeoms:
            # A vgeom without a texture carries no UVs; an empty array leaves its slice of the solver buffer zeroed.
            uvs = vgeom.uvs
            if uvs is None:
                uvs = np.zeros((0, 2), dtype=gs.np_float)
            self._solver.add_vverts(
                vvert_start=vgeom.vvert_start,
                vface_start=vgeom.vface_start,
                v_start=self._v_start,
                verts_idx=vgeom.sim_verts_idx,
                uvs=uvs,
                vfaces=vgeom.vmesh.faces.astype(gs.np_int, copy=False),
            )

    def compute_pressure_field(self):
        """
        Compute the pressure field for the FEM entity based on its tetrahedral elements.

        For hydroelastic contact: https://drake.mit.edu/doxygen_cxx/group__hydroelastic__user__guide.html

        Notes
        -----
        https://github.com/RobotLocomotion/drake/blob/master/geometry/proximity/make_mesh_field.cc
        TODO: Add margin support
        Drake's implementation of margin seems buggy.
        """
        init_positions = tensor_to_array(self.init_positions)
        signed_distance, *_ = igl.signed_distance(init_positions, init_positions, self._surface_tri_np)
        signed_distance = signed_distance.astype(gs.np_float, copy=False)

        unsigned_distance = np.abs(signed_distance)
        max_distance = np.max(unsigned_distance)
        if max_distance < gs.EPS:
            gs.raise_exception(
                f"Pressure field max distance is too small: {max_distance}. "
                "This might be due to a mesh having no internal vertices."
            )
        # normalize
        self.pressure_field_np = unsigned_distance / max_distance * self._options.solver_options.hydroelastic_modulus

    # ------------------------------------------------------------------------------------
    # ---------------------------- interfacing with solver -------------------------------
    # ------------------------------------------------------------------------------------

    @gs.assert_built
    def set_verts_pos(self, poss, verts_idx_local=None, envs_idx=None):
        """
        Set the position of some vertices.

        Parameters
        ----------
        poss : array_like, shape ([M,] [N,] 3)
            Target position of each vertex.
        verts_idx_local : None | int | array_like, shape ([M,] N)
            Index of the vertices relative to this entity, once for every environment or one row per environment. If
            None, all vertices will be considered. Defaults to None.
        envs_idx : None | int | array_like, shape (M,), optional
            The indices of the environments to set. If None, all environments will be considered. Defaults to None.
        """
        verts_idx = self._get_global_idx(verts_idx_local, self.n_vertices, self._v_start, unsafe=True, batched_idx=True)
        self._solver.set_verts_pos(poss, verts_idx, envs_idx)

    @gs.assert_built
    def set_verts_vel(self, vels, verts_idx_local=None, envs_idx=None):
        """
        Set the velocity of some vertices.

        Parameters
        ----------
        vels : array_like, shape ([M,] [N,] 3)
            Target velocity of each vertex.
        verts_idx_local : None | int | array_like, shape ([M,] N)
            Index of the vertices relative to this entity (see 'set_verts_pos').
        envs_idx : None | int | array_like, shape (M,), optional
            The indices of the environments to set. If None, all environments will be considered. Defaults to None.
        """

        self._solver._kernel_set_elements_pos_grad(
            f=f,
            element_v_start=self._v_start,
            n_vertices=self.n_vertices,
            pos_grad=pos_grad,
        )

    def set_vel(self, f, vel):
        """
        Set element velocities in the solver.

        Parameters
        ----------
        f : int
            Current substep/frame index.

        vel : torch.Tensor
            Tensor of shape (n_envs, n_vertices, 3) containing velocities.
        """

        self._solver._kernel_set_elements_vel(
            f=f,
            element_v_start=self._v_start,
            n_vertices=self.n_vertices,
            vel=vel,
        )

    def set_vel_grad(self, f, vel_grad):
        """
        Set gradient of element velocities in the solver.

        Parameters
        ----------
        f : int
            Current substep/frame index.

        vel_grad : torch.Tensor
            Tensor of shape (n_envs, n_vertices, 3) containing gradients of velocities.
        """

        self._solver._kernel_set_elements_vel_grad(
            f=f,
            element_v_start=self._v_start,
            n_vertices=self.n_vertices,
            vel_grad=vel_grad,
        )

    def set_actu(self, f, actu):
        """
        Set actuation values for elements in the solver.

        Parameters
        ----------
        f : int
            Current substep/frame index.

        actu : torch.Tensor
            Tensor of shape (n_envs, n_groups) specifying actuation values.
        """

        self._solver._kernel_set_elements_actu(
            f=f,
            element_el_start=self._el_start,
            n_elements=self.n_elements,
            n_groups=self._options.solver_options.n_groups,
            actu=actu,
        )

    def set_actu_grad(self, f, actu_grad):
        """
        Set gradient of actuation values in the solver.

        Parameters
        ----------
        f : int
            Current substep/frame index.

        actu_grad : torch.Tensor
            Tensor of shape (n_envs, n_groups) specifying gradients of actuation.
        """

        self._solver._kernel_set_elements_actu(
            f=f,
            element_el_start=self._el_start,
            n_elements=self.n_elements,
            actu_grad=actu_grad,
        )

    def set_active(self, f, active):
        """
        Set the active status of each element.

        Parameters
        ----------
        f : int
            Current substep/frame index.

        active : int
            Activity flag (gs.ACTIVE or gs.INACTIVE).
        """

        self._solver._kernel_set_active(
            f=f,
            element_el_start=self._el_start,
            n_elements=self.n_elements,
            active=active,
        )

    @assert_muscle
    def set_muscle_group(self, muscle_group):
        """
        Set muscle group index for each element.

        Parameters
        ----------
        muscle_group : torch.Tensor
            Tensor of shape (n_elements,) specifying muscle group IDs.
        """
        elements_idx = self._get_global_idx(None, self.n_elements, self._el_start, unsafe=True)
        self._solver.set_elements_muscle_group(muscle_group, elements_idx)

    @assert_muscle
    def set_muscle_direction(self, muscle_direction):
        """
        Set muscle force direction for each element.

        Parameters
        ----------
        muscle_direction : torch.Tensor
            Tensor of shape (n_elements, 3) with unit direction vectors.
        """
        elements_idx = self._get_global_idx(None, self.n_elements, self._el_start, unsafe=True)
        self._solver.set_elements_muscle_direction(muscle_direction, elements_idx)

    def set_vertex_constraints(
        self, verts_idx_local, target_poss=None, link=None, is_soft_constraint=False, stiffness=0.0, envs_idx=None
    ):
        """
        Set vertex constraints for specified vertices.

        Parameters
        ----------
            verts_idx_local : array_like
                List of local vertex indices to constrain.
            target_poss : array_like, shape (len(verts_idx), 3), optional
                List of target positions [x, y, z] for each vertex. If not provided, the initial positions are used.
            link : RigidLink
                Optional rigid link for the vertices to follow, maintaining relative position.
            is_soft_constraint: bool
                By default, use a hard constraint directly sets position and zero velocity.
                A soft constraint uses a spring force to pull the vertex towards the target position.
            stiffness : float
                Specify a spring stiffness for a soft constraint. Critical damping is applied.
            envs_idx : array_like, optional
                List of environment indices to apply the constraints to. If None, applies to all environments.
        """
        if link is not None and not isinstance(link, RigidLink):
            gs.raise_exception("Only RigidLink is supported for vertex constraints.")
        verts_idx = self._get_global_idx(verts_idx_local, self.n_vertices, self._v_start, unsafe=True, batched_idx=True)
        self._solver.set_vertex_constraints(
            verts_idx, target_poss, -1 if link is None else link.idx, is_soft_constraint, stiffness, envs_idx
        )

    def update_constraint_targets(self, verts_idx_local, target_poss, envs_idx=None):
        """Update target positions for existing constraints."""
        verts_idx = self._get_global_idx(verts_idx_local, self.n_vertices, self._v_start, unsafe=True, batched_idx=True)
        self._solver.update_constraint_targets(target_poss, verts_idx, envs_idx)

    def remove_vertex_constraints(self, verts_idx_local=None, envs_idx=None):
        """Remove constraints from the specified vertices and environments, or from all of them if None."""
        verts_idx = self._get_global_idx(verts_idx_local, self.n_vertices, self._v_start, unsafe=True, batched_idx=True)
        self._solver.remove_vertex_constraints(verts_idx, envs_idx)

    def get_el2v(self):
        """
        Retrieve the element-to-vertex mapping.

        Returns
        -------
        el2v : torch.Tensor
            Tensor of shape (n_elements, 4) mapping each element to its local vertex indices.
        """
        elements_idx = self._get_global_idx(None, self.n_elements, self._el_start, unsafe=True)
        return self._solver.get_elements_el2v(elements_idx) - self._v_start

    # ------------------------------------------------------------------------------------
    # --------------------------------- naming methods -----------------------------------
    # ------------------------------------------------------------------------------------

    def _get_morph_identifier(self) -> str:
        morph = self._morph

        if isinstance(morph, gs.morphs.Box):
            return "fem_box"
        if isinstance(morph, gs.morphs.Sphere):
            return "fem_sphere"
        if isinstance(morph, gs.morphs.Cylinder):
            return "fem_cylinder"
        if isinstance(morph, gs.morphs.Mesh):
            return f"fem_{Path(morph.file).stem}"
        return "fem_entity"

    # ------------------------------------------------------------------------------------
    # ----------------------------------- properties -------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def n_vertices(self):
        """Number of vertices in the FEM entity."""
        return len(self.init_positions)

    @property
    def vgeoms(self):
        """The list of visual geoms (`FEMVisGeom`) in the entity, one per morph sub-mesh."""
        return self._vgeoms

    @property
    def n_elements(self):
        """Number of simulation elements: surface triangles for Cloth material, tetrahedra otherwise."""
        return len(self.elems)

    @property
    def n_surfaces(self):
        """Number of surface triangles extracted from the FEM mesh."""
        return self._n_surfaces

    @property
    def v_start(self):
        """Global vertex index offset for this entity."""
        return self._v_start

    @property
    def el_start(self):
        """Global element index offset for this entity."""
        return self._el_start

    @property
    def s_start(self):
        """Global surface triangle index offset for this entity."""
        return self._s_start

    @property
    def n_vverts(self):
        """Number of render vertices in the FEM entity, summed over its visual geoms."""
        return sum(vgeom.n_vverts for vgeom in self._vgeoms)

    @property
    def n_vfaces(self):
        """Number of render faces in the FEM entity, summed over its visual geoms."""
        return sum(vgeom.n_vfaces for vgeom in self._vgeoms)

    @property
    def vvert_start(self):
        """Global render vertex index offset for this entity."""
        return self._vvert_start

    @property
    def vface_start(self):
        """Global render face index offset for this entity."""
        return self._vface_start

    @property
    def vvert_end(self):
        """Global render vertex index past this entity's last one."""
        return self._vvert_start + self.n_vverts

    @property
    def vface_end(self):
        """Global render face index past this entity's last one."""
        return self._vface_start + self.n_vfaces

    @property
    def n_surface_vertices(self):
        """Number of unique vertices involved in surface triangles."""
        return self._n_surface_vertices

    @property
    def surface_triangles(self):
        """Surface triangles of the FEM mesh."""
        return self._surface_tri_np

    @property
    def tet_cfg(self):
        """Configuration of tetrahedralization."""
        tet_cfg = mu.generate_tetgen_config_from_morph(self.morph)
        return tet_cfg

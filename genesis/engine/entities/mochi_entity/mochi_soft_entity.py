# Portions of this file are derived from Meta Platforms, Inc. and affiliates' "mochi" physics library
# (mochi_core / mochi_physics), licensed under the Apache License, Version 2.0.
# SPDX-License-Identifier: Apache-2.0
import os

import numpy as np
import torch
import trimesh

import genesis as gs
import genesis.utils.element as eu
import genesis.utils.geom as gu
import genesis.utils.mesh as mu
from genesis.engine.entities.fem_entity import FEMVisGeom
from genesis.engine.states.cache import QueriedStates
from genesis.engine.states.entities import MochiSoftEntityState
from genesis.utils.misc import tensor_to_array

from ..base_entity import Entity
from .mochi_entity import filter_entity_contacts

TET_NODE_FORMAT = ".node"
TET_VTK_FORMAT = ".vtk"


def load_tet_files(node_path):
    """Vertices and tetrahedra of a tetgen '.node' / '.ele' file pair (0- or 1-based indices)."""
    ele_path = os.path.splitext(node_path)[0] + ".ele"
    if not os.path.exists(ele_path):
        gs.raise_exception(f"Tetrahedral mesh element file not found: '{ele_path}'.")

    def read_rows(path, n_cols):
        rows = []
        with open(path) as f:
            header = None
            for line in f:
                line = line.split("#", 1)[0].strip()
                if not line:
                    continue
                if header is None:
                    header = [int(float(v)) for v in line.split()]
                    continue
                values = line.split()
                rows.append([float(v) for v in values[: 1 + n_cols]])
        if header is None:
            gs.raise_exception(f"Empty tetrahedral mesh file: '{path}'.")
        rows = np.array(rows, dtype=np.float64).reshape((-1, 1 + n_cols))
        return rows[: header[0]]

    nodes = read_rows(node_path, 3)
    elems = read_rows(ele_path, 4)
    index_base = int(nodes[0, 0]) if len(nodes) > 0 else 0
    verts = nodes[:, 1:4]
    tets = elems[:, 1:5].astype(np.int64) - index_base
    if tets.min(initial=0) < 0 or tets.max(initial=-1) >= len(verts):
        gs.raise_exception(f"Tetrahedral mesh '{ele_path}' references vertices outside '{node_path}'.")
    return verts, tets


def load_vtk_tet_files(path):
    """Vertices and tetrahedra of a legacy ASCII VTK unstructured grid holding a tetrahedral mesh (the format Drake
    ships its deformable meshes in)."""
    with open(path) as f:
        lines = [line.split("#", 1)[0] for line in f]

    def read_tokens(start, count):
        tokens = []
        i = start
        while len(tokens) < count and i < len(lines):
            tokens.extend(lines[i].split())
            i += 1
        if len(tokens) < count:
            gs.raise_exception(f"Truncated VTK mesh file: '{path}'.")
        return tokens[:count], i

    verts = cells = cell_types = None
    i = 0
    while i < len(lines):
        parts = lines[i].split()
        if parts and parts[0] == "POINTS":
            values, i = read_tokens(i + 1, 3 * int(parts[1]))
            verts = np.array(values, dtype=np.float64).reshape((-1, 3))
        elif parts and parts[0] == "CELLS":
            values, i = read_tokens(i + 1, int(parts[2]))
            cells = np.array(values, dtype=np.int64)
        elif parts and parts[0] == "CELL_TYPES":
            values, i = read_tokens(i + 1, int(parts[1]))
            cell_types = np.array(values, dtype=np.int64)
        else:
            i += 1
    if verts is None or cells is None or cell_types is None:
        gs.raise_exception(f"'{path}' is not an ASCII VTK unstructured grid (POINTS, CELLS and CELL_TYPES required).")

    tets = []
    offset = 0
    for cell_type in cell_types:
        n_nodes = int(cells[offset])
        if cell_type == 10:
            if n_nodes != 4:
                gs.raise_exception(f"Malformed tetrahedron in VTK mesh file '{path}'.")
            tets.append(cells[offset + 1 : offset + 5])
        offset += 1 + n_nodes
    if not tets:
        gs.raise_exception(f"VTK mesh file '{path}' holds no tetrahedra.")
    tets = np.stack(tets)
    if tets.min() < 0 or tets.max() >= len(verts):
        gs.raise_exception(f"VTK mesh file '{path}' references vertices outside its POINTS block.")
    return verts, tets


def embed_in_tets(points, verts, tets):
    """Element index and barycentric coordinates of every point in the tetrahedral mesh (verts, tets): each point
    takes the tetrahedron whose barycentric coordinates are least negative, so points inside a tetrahedron are exact
    and points outside the coarse mesh extrapolate linearly from the nearest one."""
    points = np.asarray(points, dtype=np.float64)
    verts = np.asarray(verts, dtype=np.float64)
    tets = np.asarray(tets, dtype=np.int64)
    x3 = verts[tets[:, 3]]
    Dm = np.stack([verts[tets[:, k]] - x3 for k in range(3)], axis=-1)
    Dm_inv = np.linalg.inv(Dm)
    elems_idx = np.empty(len(points), dtype=gs.np_int)
    bary = np.empty((len(points), 4), dtype=gs.np_float)
    chunk = max(1, int(2e7) // max(1, len(tets)))
    for start in range(0, len(points), chunk):
        block = points[start : start + chunk]
        # b[i, t, :] barycentric coordinates of point i in tetrahedron t
        b123 = np.einsum("tjk,itk->itj", Dm_inv, block[:, None, :] - x3[None, :, :])
        b = np.concatenate([b123, 1.0 - b123.sum(axis=-1, keepdims=True)], axis=-1)
        best = b.min(axis=-1).argmax(axis=-1)
        rows = np.arange(len(block))
        elems_idx[start : start + chunk] = best
        bary[start : start + chunk] = b[rows, best]
    return elems_idx, bary


class MochiSoftEntity(Entity):
    """
    Deformable body simulated by the MochiSolver: a tetrahedral mesh (`gs.materials.Mochi.Elastic`) or a thin shell
    made of the surface triangles of the morph (`gs.materials.Mochi.Shell`), whose vertex positions are unknowns of the
    implicit solve.

    The solid simulation mesh is tetrahedralized from the morph surface (tetgen options on the morph), or read directly
    from a tetgen '.node' / '.ele' file pair given as a `gs.morphs.Mesh`. The boundary triangles (all triangles of a
    shell) carry the contact samples and the render mesh.
    """

    def __init__(
        self,
        scene,
        solver,
        material,
        morph,
        surface,
        idx,
        idx_in_solver,
        v_start,
        vvert_start,
        vface_start,
        name=None,
    ):
        self._is_shell = isinstance(material, gs.materials.Mochi.Shell)
        self._is_rod = isinstance(material, gs.materials.Mochi.Rod)
        if self._is_rod != isinstance(morph, gs.morphs.Rod):
            gs.raise_exception("A `gs.morphs.Rod` morph requires a `gs.materials.Mochi.Rod` material, and conversely.")
        super().__init__(idx, scene, morph, solver, material, surface, name=name)
        self._idx_in_solver = idx_in_solver
        self._v_start = v_start
        self._vvert_start = vvert_start
        self._vface_start = vface_start
        self._queried_states = QueriedStates()
        self._attachments = []

        self.sample()

        # Boundary triangles (outward winding), all triangles of a shell, none for a rod (its contact samples lie on
        # the centerline).
        if self._is_rod:
            self._surface_tri_np = np.zeros((0, 3), dtype=gs.np_int)
        elif self._is_shell:
            self._surface_tri_np = self.elems
        else:
            self._surface_tri_np = self._boundary_triangles(self.elems)

    def _get_morph_identifier(self) -> str:
        morph = self._morph
        if self._is_rod:
            return "rod"
        prefix = "shell" if self._is_shell else "soft"
        if isinstance(morph, gs.morphs.Box):
            return f"{prefix}_box"
        if isinstance(morph, gs.morphs.Sphere):
            return f"{prefix}_sphere"
        if isinstance(morph, gs.morphs.Cylinder):
            return f"{prefix}_cylinder"
        if isinstance(morph, gs.morphs.Mesh):
            return f"{prefix}_{os.path.splitext(os.path.basename(morph.file))[0]}"
        return f"{prefix}_body"

    # ------------------------------------------------------------------------------------
    # ----------------------------------- instantiation ----------------------------------
    # ------------------------------------------------------------------------------------

    def sample(self):
        """Build the render geoms and the tetrahedral simulation mesh from the morph."""
        morph = self._morph
        if self._is_rod:
            self._sample_rod()
            return
        if isinstance(morph, gs.morphs.Mesh) and morph.file.endswith((TET_NODE_FORMAT, TET_VTK_FORMAT)):
            if self._is_shell:
                gs.raise_exception("Shells are surface meshes: a tetrahedral mesh file cannot be used for a shell.")
            load_files = load_tet_files if morph.file.endswith(TET_NODE_FORMAT) else load_vtk_tet_files
            verts, elems = load_files(morph.file)
            # The morph rotation is applied by `instantiate` (about the vertex centroid, like every other morph).
            verts = verts * np.asarray(morph.scale, dtype=np.float64) + np.asarray(morph.pos, dtype=np.float64)
            self.instantiate(verts, elems)
            surface_tri = self._boundary_triangles(self.elems)
            vmesh = gs.Mesh.from_trimesh(
                trimesh.Trimesh(vertices=self.init_positions, faces=surface_tri, process=False),
                surface=self._surface,
            )
            self._vgeoms = gs.List(
                [
                    FEMVisGeom(
                        entity=self,
                        vvert_start=self._vvert_start,
                        vface_start=self._vface_start,
                        vmesh=vmesh,
                        sim_verts_idx=np.arange(self.n_vertices, dtype=gs.np_int),
                    )
                ]
            )
            return

        meshes = gs.Mesh.from_morph_surface(morph, self._surface)
        surface_verts, surface_faces, verts_maps = mu.merge_submeshes(
            [mesh.verts for mesh in meshes], [mesh.faces for mesh in meshes]
        )
        self._vgeoms = gs.List()
        vvert_start, vface_start = self._vvert_start, self._vface_start
        for mesh, verts_idx in zip(meshes, verts_maps):
            self._vgeoms.append(
                FEMVisGeom(
                    entity=self, vvert_start=vvert_start, vface_start=vface_start, vmesh=mesh, sim_verts_idx=verts_idx
                )
            )
            vvert_start += len(mesh.verts)
            vface_start += len(mesh.faces)

        if self._is_shell:
            # The welded surface triangles are the shell elements.
            self.instantiate(surface_verts + morph.pos, surface_faces)
            return

        # File meshes are tetrahedralized untranslated so that the result and its cache are shared across placements;
        # primitives keep the position baked in, as the rest state is sensitive to the exact refinement.
        is_mesh_morph = isinstance(morph, gs.morphs.Mesh)
        if not is_mesh_morph:
            surface_verts = surface_verts + morph.pos
        surface_trimesh = trimesh.Trimesh(vertices=surface_verts, faces=surface_faces, process=False)
        verts, elems = eu.mesh_to_elements(surface_trimesh, tet_cfg=mu.generate_tetgen_config_from_morph(morph))
        if is_mesh_morph:
            verts = verts + morph.pos
        self.instantiate(verts, elems)

    def _sample_rod(self):
        """Rod nodes and segments from the polyline of the morph, the reference material axes (an arbitrary
        perpendicular of the first segment parallel transported along the rod), and the tube drawn around it."""
        morph = self._morph
        points = np.asarray(morph.points, dtype=gs.np_float) + np.asarray(morph.pos, dtype=gs.np_float)
        n_nodes = len(points)
        segments = np.stack([np.arange(n_nodes - 1), np.arange(1, n_nodes)], axis=-1)
        if morph.is_closed_loop:
            segments = np.concatenate([segments, [[n_nodes - 1, 0]]])
        self.instantiate(points, segments)
        verts = self.init_positions
        tangents = verts[segments[:, 1]] - verts[segments[:, 0]]
        tangents /= np.linalg.norm(tangents, axis=1)[:, None]
        axes = np.zeros_like(tangents)
        seed = np.array([0.0, 0.0, 1.0]) if abs(tangents[0][2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        axes[0] = seed - tangents[0] * seed.dot(tangents[0])
        axes[0] /= np.linalg.norm(axes[0])
        for i_s in range(1, len(segments)):
            # Parallel transport of the axis across the node (minimal rotation between consecutive tangents).
            t0, t1 = tangents[i_s - 1], tangents[i_s]
            c, v = t0.dot(t1), np.cross(t0, t1)
            R = c * np.eye(3) + np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
            R = R + np.outer(v, v) / max(1.0 + c, 1e-300)
            axis = R @ axes[i_s - 1]
            axis -= t1 * axis.dot(t1)
            axes[i_s] = axis / np.linalg.norm(axis)
        self.rod_axes_ref = axes.astype(gs.np_float)
        self.rod_radius = float(morph.radius)

        # Tube: one ring of vertices per node, on the material frame of the segment following the node.
        n_cs = morph.n_cross_section_segments
        angles = np.linspace(0.0, 2.0 * np.pi, n_cs, endpoint=False)
        offsets = np.stack([np.cos(angles), np.sin(angles)], axis=-1) * morph.radius
        ring_nodes = np.repeat(np.arange(n_nodes), n_cs)
        ring_elems = np.repeat(np.minimum(np.arange(n_nodes), len(segments) - 1), n_cs)
        ring_offsets = np.tile(offsets, (n_nodes, 1))
        tube_verts = np.zeros((n_nodes * n_cs, 3), dtype=gs.np_float)
        for i_vv in range(n_nodes * n_cs):
            i_v, i_s = ring_nodes[i_vv], ring_elems[i_vv]
            binormal = np.cross(tangents[i_s], axes[i_s])
            tube_verts[i_vv] = verts[i_v] + ring_offsets[i_vv, 0] * axes[i_s] + ring_offsets[i_vv, 1] * binormal
        faces = []
        n_rings = n_nodes if morph.is_closed_loop else n_nodes - 1
        for i_ring in range(n_rings):
            j_ring = (i_ring + 1) % n_nodes
            for k in range(n_cs):
                a, b = i_ring * n_cs + k, i_ring * n_cs + (k + 1) % n_cs
                c, d = j_ring * n_cs + k, j_ring * n_cs + (k + 1) % n_cs
                faces.append([a, b, d])
                faces.append([a, d, c])
        self.rod_vverts_node = ring_nodes.astype(gs.np_int)
        self.rod_vverts_elem = ring_elems.astype(gs.np_int)
        self.rod_vverts_offset = ring_offsets.astype(gs.np_float)
        vmesh = gs.Mesh.from_trimesh(
            trimesh.Trimesh(vertices=tube_verts, faces=np.array(faces), process=False), surface=self._surface
        )
        self._vgeoms = gs.List(
            [
                FEMVisGeom(
                    entity=self,
                    vvert_start=self._vvert_start,
                    vface_start=self._vface_start,
                    vmesh=vmesh,
                    sim_verts_idx=self.rod_vverts_node,
                )
            ]
        )

    @staticmethod
    def _boundary_triangles(elems):
        """Boundary triangles of positively oriented tetrahedra, with outward winding."""
        el2tri = np.array(
            [[[v[0], v[1], v[2]], [v[0], v[3], v[1]], [v[1], v[3], v[2]], [v[0], v[2], v[3]]] for v in elems],
            dtype=gs.np_int,
        ).reshape((-1, 3))
        _, unique_idcs, cnt = np.unique(np.sort(el2tri, axis=1), axis=0, return_counts=True, return_index=True)
        return el2tri[unique_idcs][cnt == 1]

    def instantiate(self, verts, elems):
        """Set the rest vertex positions (with the morph orientation applied about their centroid and the morph pose
        offset composed on) and the tetrahedra."""
        verts = np.asarray(verts, dtype=gs.np_float)
        elems = np.asarray(elems, dtype=gs.np_int)
        if len(verts) == 0 or len(elems) == 0:
            gs.raise_exception("Entity has no tetrahedra.")
        self._instantiate_verts_COM = verts.mean(axis=0)
        morph_quat = np.array(self._morph.quat, dtype=gs.np_float)
        init_quat = gu.transform_quat_by_quat(np.array(self._morph.offset_quat, dtype=gs.np_float), morph_quat)
        R = gu.quat_to_R(init_quat)
        verts_COM = verts.mean(axis=0)
        init_positions = (verts - verts_COM) @ R.T + verts_COM
        init_positions = init_positions + gu.transform_by_quat(
            np.array(self._morph.offset_pos, dtype=gs.np_float), morph_quat
        )
        if elems.shape[1] == 4:
            # Positive orientation of every tetrahedron (node 3 as origin).
            D = np.stack([init_positions[elems[:, k]] - init_positions[elems[:, 3]] for k in range(3)], axis=-1)
            is_inverted = np.linalg.det(D) < 0.0
            elems = elems.copy()
            elems[is_inverted, 1], elems[is_inverted, 2] = elems[is_inverted, 2], elems[is_inverted, 1]
        self.init_positions = init_positions.astype(gs.np_float)
        self.elems = elems

    # ------------------------------------------------------------------------------------
    # ---------------------------------------- io ----------------------------------------
    # ------------------------------------------------------------------------------------

    def get_state(self):
        s_global = self._sim.cur_step_global
        if s_global in self._queried_states:
            return self._queried_states[s_global][0]
        state = MochiSoftEntityState(self, s_global)
        self._solver.get_soft_entity_state(self, state.pos, state.vel)
        self._queried_states.append(state)
        return state

    def attach_to_link(self, link, verts_idx, stiffness=1e6, damping=0.0):
        """Rigidly attach vertices of this deformable body to a rigid link.

        Each attached vertex is tied by a stiff penalty (inside the implicit solve) to the point of the link it
        coincides with when the scene is built, so place the body at its mounted pose. Forces act on both sides and
        the coupled bodies form one simulation island. Must be called before the scene is built.

        Parameters
        ----------
        link : RigidLink
            A link of a rigid or articulated entity simulated by the MochiSolver.
        verts_idx : array_like
            Indices of the vertices of this entity to attach.
        stiffness : float, optional
            Penalty stiffness k of the energy 1/2 k |c|^2 per vertex, in N/m. Defaults to 1e6 (mochi's constraint
            default).
        damping : float, optional
            Damping d on the violation rate, adding 1/2 (d / dt) |c - c_stage_start|^2 per stage. Defaults to 0.
        """
        if self.is_built:
            gs.raise_exception("Attachments must be registered before the scene is built.")
        verts_idx = np.unique(np.asarray(verts_idx, dtype=gs.np_int).reshape((-1,)))
        if len(verts_idx) == 0 or verts_idx[0] < 0 or verts_idx[-1] >= self.n_vertices:
            gs.raise_exception(f"'verts_idx' must be non-empty vertex indices within [0, {self.n_vertices}).")
        if stiffness <= 0.0 or damping < 0.0:
            gs.raise_exception("'stiffness' must be positive and 'damping' non-negative.")
        self._attachments.append(
            {"link": link, "verts_idx": verts_idx, "stiffness": float(stiffness), "damping": float(damping)}
        )

    @property
    def attachments(self):
        """Rigid-link attachment registrations of this entity."""
        return self._attachments

    def set_visual_mesh(self, file, pos=(0.0, 0.0, 0.0), euler=(0.0, 0.0, 0.0), scale=1.0):
        """Replace the render mesh of this solid by a detailed visual mesh skinned by the simulation tetrahedra.

        Every vertex of the visual mesh is embedded in the tetrahedron whose barycentric coordinates it takes (points
        outside the coarse simulation mesh extrapolate linearly from the nearest one), so the visual surface follows
        the deformation of the simulation mesh - the usual pairing of a fine render mesh with a coarse collision
        mesh. `pos`, `euler` (degrees) and `scale` place the visual mesh in the frame of the simulation mesh file;
        the entity's morph transform then applies to both. Must be called before the scene is built.
        """
        if self.is_built:
            gs.raise_exception("The visual mesh must be set before the scene is built.")
        if self._is_shell or self._is_rod:
            gs.raise_exception("Only solid (tetrahedral) entities can carry an embedded visual mesh.")
        visual_trimesh = trimesh.load(file, force="mesh", process=False)
        quat = gu.xyz_to_quat(np.asarray(euler, dtype=np.float64), rpy=True, degrees=True)
        verts = visual_trimesh.vertices * float(scale) @ gu.quat_to_R(quat).T + np.asarray(pos, dtype=np.float64)
        # Mirror the placement of the simulation mesh: the pre-rotation transform of `sample` (the morph scale and
        # position for file meshes), then the morph rotation about the simulation mesh's centroid of `instantiate`.
        morph = self._morph
        if isinstance(morph, gs.morphs.Mesh):
            verts = verts * np.asarray(morph.scale, dtype=np.float64)
        verts = verts + np.asarray(morph.pos, dtype=np.float64)
        morph_quat = np.array(morph.quat, dtype=gs.np_float)
        init_quat = gu.transform_quat_by_quat(np.array(morph.offset_quat, dtype=gs.np_float), morph_quat)
        com = self._instantiate_verts_COM
        verts = (verts - com) @ gu.quat_to_R(init_quat).T + com
        verts = verts + gu.transform_by_quat(np.array(morph.offset_pos, dtype=gs.np_float), morph_quat)
        elems_idx, bary = embed_in_tets(verts, self.init_positions, self.elems)
        # Keep the loaded mesh's UVs and material; only the vertex positions move into the simulation frame.
        visual_trimesh = visual_trimesh.copy()
        visual_trimesh.vertices = verts
        vmesh = gs.Mesh.from_trimesh(visual_trimesh, surface=self._surface)
        self._vgeoms = gs.List(
            [
                FEMVisGeom(
                    entity=self,
                    vvert_start=self._vvert_start,
                    vface_start=self._vface_start,
                    vmesh=vmesh,
                    sim_verts_idx=None,
                    elems_idx=elems_idx,
                    bary=bary,
                )
            ]
        )

    def get_vertices_position(self, envs_idx=None):
        """World positions of the vertices, shape (n_vertices, 3) or (n_envs, n_vertices, 3)."""
        return self._solver.get_soft_vertices_position(self, envs_idx)

    def get_vertices_velocity(self, envs_idx=None):
        """Velocities of the vertices, shape (n_vertices, 3) or (n_envs, n_vertices, 3)."""
        return self._solver.get_soft_vertices_velocity(self, envs_idx)

    def get_vertices_contact_force(self, envs_idx=None):
        """Net contact force applied to each vertex, shape (n_vertices, 3) or (n_envs, n_vertices, 3)."""
        return self._solver.get_soft_vertices_contact_force(self, envs_idx)

    def get_contacts(self, with_entity=None, exclude_self_contact=False, is_padded=False):
        """
        Contact points of the most recent `scene.step()` involving this body (and `with_entity` if given), with the
        keys documented in `MochiEntity.get_contacts`. On the side of this body a contact point carries the vertices of
        the surface element (`verts_a` / `verts_b`, local to this body, -1 padded) and the weights spreading the force
        over them (`bary_a` / `bary_b`).
        """
        return filter_entity_contacts(self._solver, self, with_entity, exclude_self_contact, is_padded)

    def set_position(self, pos, envs_idx=None):
        """Move the body: a (3,) or (n_envs, 3) offset of the rest positions, or full vertex positions of shape
        (n_vertices, 3) or (n_envs, n_vertices, 3)."""
        pos = torch.as_tensor(tensor_to_array(pos), dtype=gs.tc_float, device=gs.device)
        if pos.shape[-2:] != (self.n_vertices, 3):
            pos = torch.as_tensor(self.init_positions, dtype=gs.tc_float, device=gs.device) + pos[..., None, :]
        self._solver.set_soft_vertices_position(self, pos, envs_idx)

    def set_velocity(self, vel, envs_idx=None):
        """Set the vertex velocities: one (3,) or (n_envs, 3) velocity for all vertices, or per vertex of shape
        (n_vertices, 3) or (n_envs, n_vertices, 3)."""
        vel = torch.as_tensor(tensor_to_array(vel), dtype=gs.tc_float, device=gs.device)
        if vel.shape[-2:] != (self.n_vertices, 3):
            vel = vel[..., None, :].expand(*vel.shape[:-1], self.n_vertices, 3)
        self._solver.set_soft_vertices_velocity(self, vel, envs_idx)

    def _sanitize_vertices_idx(self, verts_idx_local):
        verts_idx = np.atleast_1d(np.asarray(tensor_to_array(verts_idx_local), dtype=gs.np_int))
        if verts_idx.min(initial=0) < 0 or verts_idx.max(initial=-1) >= self.n_vertices:
            gs.raise_exception("Vertex index out of range.")
        return verts_idx

    def set_vertices_fixed(self, verts_idx_local, is_fixed=True, envs_idx=None):
        """Fix the given vertices at their current position (Dirichlet condition), or release them."""
        self._solver.set_soft_vertices_fixed(self, self._sanitize_vertices_idx(verts_idx_local), is_fixed, envs_idx)

    def set_vertices_target(self, verts_idx_local, pos, envs_idx=None):
        """Prescribe the positions the given vertices must reach at the end of the next step (moving Dirichlet
        condition); the vertices become fixed until released with `set_vertices_fixed(..., is_fixed=False)`. `pos`
        has shape (n_verts_idx, 3) or (n_envs, n_verts_idx, 3)."""
        self._solver.set_soft_vertices_target(self, self._sanitize_vertices_idx(verts_idx_local), pos, envs_idx)

    def set_contact_params(
        self,
        *,
        penalty_coefficient=None,
        friction=None,
        penalty_smoothing_half_distance=None,
        penalty_threshold=None,
        friction_falloff_vel=None,
        viscous_friction=None,
        normal_viscous_damping=None,
    ):
        """Update the contact parameters of this body (see `gs.materials.Mochi.Elastic`)."""
        self._solver.set_soft_entity_contact_params(
            self,
            penalty_coefficient=penalty_coefficient,
            friction=friction,
            penalty_smoothing_half_distance=penalty_smoothing_half_distance,
            penalty_threshold=penalty_threshold,
            friction_falloff_vel=friction_falloff_vel,
            viscous_friction=viscous_friction,
            normal_viscous_damping=normal_viscous_damping,
        )

    # ------------------------------------------------------------------------------------
    # ----------------------------------- properties -------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def idx_in_solver(self):
        return self._idx_in_solver

    @property
    def is_shell(self):
        return self._is_shell

    @property
    def is_rod(self):
        return self._is_rod

    @property
    def n_rod_stencils(self):
        """Interior nodes of a rod (all nodes of a closed loop)."""
        if not self._is_rod:
            return 0
        return self.n_vertices if self._morph.is_closed_loop else max(0, self.n_vertices - 2)

    @property
    def n_vertices(self):
        return len(self.init_positions)

    @property
    def n_elements(self):
        return len(self.elems)

    @property
    def n_surfaces(self):
        return len(self._surface_tri_np)

    @property
    def surface_triangles(self):
        return self._surface_tri_np

    @property
    def n_dofs(self):
        return 3 * self.n_vertices

    @property
    def v_start(self):
        return self._v_start

    @property
    def v_end(self):
        return self._v_start + self.n_vertices

    @property
    def vgeoms(self):
        return self._vgeoms

    @property
    def n_vgeoms(self):
        return len(self._vgeoms)

    @property
    def n_vverts(self):
        return sum(vgeom.n_vverts for vgeom in self._vgeoms)

    @property
    def n_vfaces(self):
        return sum(vgeom.n_vfaces for vgeom in self._vgeoms)

    @property
    def vvert_start(self):
        return self._vvert_start

    @property
    def vface_start(self):
        return self._vface_start

    @property
    def volume(self):
        if self._is_shell:
            return 0.0
        D = np.stack(
            [self.init_positions[self.elems[:, k]] - self.init_positions[self.elems[:, 3]] for k in range(3)], -1
        )
        return float(np.abs(np.linalg.det(D)).sum() / 6.0)

    @property
    def area(self):
        """Total area of the surface triangles."""
        tri = self.init_positions[self._surface_tri_np]
        return float(0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=-1).sum())

    @property
    def length(self):
        """Total length of the rod centerline."""
        if not self._is_rod:
            return 0.0
        return float(
            np.linalg.norm(self.init_positions[self.elems[:, 1]] - self.init_positions[self.elems[:, 0]], axis=1).sum()
        )

    @property
    def mass(self):
        if self._is_rod:
            return self.material.resolve(self.rod_radius)["linear_density"] * self.length
        if self._is_shell:
            return self.material.areal_density * self.area
        return self.material.rho * self.volume

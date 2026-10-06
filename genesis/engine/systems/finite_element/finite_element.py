from __future__ import annotations

import numpy as np

import genesis as gs
from genesis.utils.misc import qd_to_numpy


def compute_Dm_inv_2d(
    positions: np.ndarray,
    triangles: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    x0 = positions[triangles[:, 0]]
    x1 = positions[triangles[:, 1]]
    x2 = positions[triangles[:, 2]]
    v01 = x1 - x0
    v02 = x2 - x0
    normals = np.cross(v01, v02)
    norms = np.linalg.norm(normals, axis=1)
    areas = 0.5 * norms
    if np.any(areas < 1e-30):
        index = int(np.flatnonzero(areas < 1e-30)[0])
        raise ValueError(f"Degenerate QCloth triangle {index}: area={areas[index]:.2e}")
    normals = normals / norms[:, None]

    vx = normals[:, 1]
    vy = -normals[:, 0]
    cosine = normals[:, 2]
    rotation = np.tile(np.eye(3, dtype=np.float64), (len(triangles), 1, 1))
    flip = cosine + 1.0 < 1e-12
    regular = ~flip
    rotation[flip, 0, 0] = -1.0
    rotation[flip, 1, 1] = -1.0
    if np.any(regular):
        vx_regular = vx[regular]
        vy_regular = vy[regular]
        cosine_regular = cosine[regular]
        inverse_one_plus_cosine = 1.0 / (1.0 + cosine_regular)
        rotation[regular, 0, 2] += vy_regular
        rotation[regular, 2, 0] -= vy_regular
        rotation[regular, 1, 2] -= vx_regular
        rotation[regular, 2, 1] += vx_regular
        vx_squared = vx_regular * vx_regular * inverse_one_plus_cosine
        vy_squared = vy_regular * vy_regular * inverse_one_plus_cosine
        vx_vy = vx_regular * vy_regular * inverse_one_plus_cosine
        rotation[regular, 0, 0] -= vy_squared
        rotation[regular, 0, 1] += vx_vy
        rotation[regular, 1, 0] += vx_vy
        rotation[regular, 1, 1] -= vx_squared
        rotation[regular, 2, 2] -= (vx_squared + vy_squared) * (1.0 + cosine_regular) * inverse_one_plus_cosine

    rotation_2d = rotation[:, :2, :]
    edge_01 = np.einsum("nij,nj->ni", rotation_2d, v01)
    edge_02 = np.einsum("nij,nj->ni", rotation_2d, v02)
    a = edge_01[:, 0]
    b = edge_02[:, 0]
    c = edge_01[:, 1]
    d = edge_02[:, 1]
    inverse_determinant = 1.0 / (a * d - b * c)
    Dm_inv = np.empty((len(triangles), 2, 2), dtype=np.float64)
    Dm_inv[:, 0, 0] = d * inverse_determinant
    Dm_inv[:, 0, 1] = -b * inverse_determinant
    Dm_inv[:, 1, 0] = -c * inverse_determinant
    Dm_inv[:, 1, 1] = a * inverse_determinant
    return Dm_inv, areas


def compute_shell_lumped_mass(
    positions: np.ndarray,
    triangles: np.ndarray,
    density: float,
    thickness: float,
) -> np.ndarray:
    x0 = positions[triangles[:, 0]]
    x1 = positions[triangles[:, 1]]
    x2 = positions[triangles[:, 2]]
    areas = 0.5 * np.linalg.norm(np.cross(x1 - x0, x2 - x0), axis=1)
    corner_mass = areas * 2.0 * thickness * density / 3.0
    masses = np.zeros(len(positions), dtype=np.float64)
    for corner in range(3):
        np.add.at(masses, triangles[:, corner], corner_mass)
    return masses


def _shell_hinges(triangles: np.ndarray) -> np.ndarray:
    edge_opposites: dict[tuple[int, int], list[int]] = {}
    for triangle in triangles:
        a, b, c = (int(index) for index in triangle)
        for left, right, opposite in ((a, b, c), (b, c, a), (c, a, b)):
            edge = (min(left, right), max(left, right))
            edge_opposites.setdefault(edge, []).append(opposite)
    hinges = [
        (left, right, opposites[0], opposites[1])
        for (left, right), opposites in sorted(edge_opposites.items())
        if len(opposites) == 2
    ]
    return np.asarray(hinges, dtype=np.int32).reshape(-1, 4)


def _quadratic_bending_Q0(rest_hinge_positions: np.ndarray) -> np.ndarray:
    xi, xj, xk, xl = rest_hinge_positions

    def cotangent(left: np.ndarray, right: np.ndarray) -> float:
        cross_norm = np.linalg.norm(np.cross(left, right))
        return float(np.dot(left, right) / cross_norm) if cross_norm > 1e-30 else 0.0

    c01 = cotangent(xj - xi, xk - xi)
    c02 = cotangent(xj - xi, xl - xi)
    c03 = cotangent(xi - xj, xk - xj)
    c04 = cotangent(xi - xj, xl - xj)
    K0 = np.array(
        [c03 + c04, c01 + c02, -c01 - c03, -c02 - c04],
        dtype=np.float64,
    )
    area_0 = 0.5 * np.linalg.norm(np.cross(xj - xi, xk - xi))
    area_1 = 0.5 * np.linalg.norm(np.cross(xj - xi, xl - xi))
    area_sum = area_0 + area_1
    if area_sum <= 1e-30:
        return np.zeros((4, 4), dtype=np.float64)
    return np.outer(K0, K0) / area_sum


def _surface_edges(triangles: np.ndarray) -> np.ndarray:
    edges = np.concatenate(
        (
            triangles[:, (0, 1)],
            triangles[:, (1, 2)],
            triangles[:, (0, 2)],
        ),
        axis=0,
    )
    edges = np.sort(edges, axis=1)
    return np.unique(edges, axis=0).astype(np.int32, copy=False)


def _surface_area_weights(
    positions: np.ndarray,
    triangles: np.ndarray,
    edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_verts = len(positions)
    x0 = positions[triangles[:, 0]]
    x1 = positions[triangles[:, 1]]
    x2 = positions[triangles[:, 2]]
    face_weights = 0.5 * np.linalg.norm(np.cross(x1 - x0, x2 - x0), axis=1)
    corner_weights = face_weights / 3.0

    vert_weights = np.zeros(n_verts, dtype=np.float64)
    for corner in range(3):
        vert_weights += np.bincount(
            triangles[:, corner],
            weights=corner_weights,
            minlength=n_verts,
        )

    edge_weights = np.zeros(len(edges), dtype=np.float64)
    if len(edges) != 0:
        key_scale = n_verts + 1

        def edge_key(left: np.ndarray, right: np.ndarray) -> np.ndarray:
            return np.minimum(left, right) * key_scale + np.maximum(left, right)

        edge_keys = edge_key(edges[:, 0], edges[:, 1])
        order = np.argsort(edge_keys)
        sorted_keys = edge_keys[order]
        for left, right in ((0, 1), (1, 2), (0, 2)):
            triangle_keys = edge_key(triangles[:, left], triangles[:, right])
            positions_in_order = np.searchsorted(sorted_keys, triangle_keys)
            edge_weights += np.bincount(
                order[positions_in_order],
                weights=corner_weights,
                minlength=len(edges),
            )
    return vert_weights, edge_weights, face_weights


class FiniteElement:
    """Graph-native host-side FEM staging owner."""

    def init(self, scene) -> bool:
        entities = [
            entity for entity in scene.fem_solver.entities if isinstance(entity.material, gs.materials.FEM.QCloth)
        ]
        if len(entities) != len(scene.fem_solver.entities):
            raise RuntimeError("The QIPC engine currently accepts only FEM.QCloth FEM entities")
        if not entities:
            return False

        n_envs = scene._B
        gravity = np.asarray(
            qd_to_numpy(scene.fem_solver._gravity, transpose=True, copy=True),
            dtype=np.float64,
        ).reshape(n_envs, 3)

        constrained = None
        is_soft = None
        link_idx = None
        target = None
        if scene.fem_solver._constraints_initialized:
            constrained = np.asarray(
                scene.fem_solver.vertex_constraints.is_constrained.to_numpy(),
                dtype=np.int32,
            )
            is_soft = np.asarray(
                scene.fem_solver.vertex_constraints.is_soft_constraint.to_numpy(),
                dtype=np.int32,
            )
            link_idx = np.asarray(
                scene.fem_solver.vertex_constraints.link_idx.to_numpy(),
                dtype=np.int32,
            )
            target = np.asarray(
                scene.fem_solver.vertex_constraints.target_pos.to_numpy(),
                dtype=np.float64,
            )
            if constrained.shape == (n_envs, scene.fem_solver.n_vertices):
                constrained = constrained.T
                is_soft = is_soft.T
                link_idx = link_idx.T
                target = np.swapaxes(target, 0, 1)

        positions: list[np.ndarray] = []
        velocities: list[np.ndarray] = []
        masses: list[np.ndarray] = []
        is_fixed: list[np.ndarray] = []
        gravities: list[np.ndarray] = []
        triangles: list[np.ndarray] = []
        bridge_vertex: list[np.ndarray] = []
        bridge_environment: list[np.ndarray] = []
        Dm_inv: list[np.ndarray] = []
        rest_areas: list[np.ndarray] = []
        thicknesses: list[np.ndarray] = []
        membrane_mu: list[np.ndarray] = []
        membrane_lambda: list[np.ndarray] = []
        strain_limit_multiplier: list[np.ndarray] = []
        hinge_indices: list[np.ndarray] = []
        hinge_Q0: list[np.ndarray] = []
        hinge_stiffness: list[np.ndarray] = []
        vert_bend_k: list[np.ndarray] = []
        body_ids: list[np.ndarray] = []
        geometry_ids: list[np.ndarray] = []
        source_geometry_ids: list[np.ndarray] = []
        geometry_environments: list[np.ndarray] = []
        body_environment: list[int] = []
        body_vertex_offsets = [0]

        entity_data = []
        for source_geometry_id, entity in enumerate(entities):
            rest_positions = np.asarray(entity.init_positions.cpu(), dtype=np.float64)
            local_triangles = np.asarray(entity.elems, dtype=np.int32).reshape(-1, 3)
            local_Dm_inv, local_rest_areas = compute_Dm_inv_2d(rest_positions, local_triangles)
            local_masses = compute_shell_lumped_mass(
                rest_positions,
                local_triangles,
                entity.material.rho,
                entity.material.thickness,
            )
            fixed = np.zeros((entity.n_vertices, n_envs), dtype=np.int32)
            targets = np.zeros((entity.n_vertices, n_envs, 3), dtype=np.float64)
            if constrained is not None:
                vertex_slice = slice(entity._v_start, entity._v_start + entity.n_vertices)
                fixed = constrained[vertex_slice]
                targets = target[vertex_slice]
                unsupported = (fixed != 0) & ((is_soft[vertex_slice] != 0) | (link_idx[vertex_slice] >= 0))
                if np.any(unsupported):
                    raise RuntimeError("QCloth currently supports only fixed hard vertex constraints")

            material = entity.material
            effective_shear = material.shear_modulus if material.shear_modulus is not None else material.E / 2.0
            bending_E = material.bending_youngs_modulus if material.bending_youngs_modulus is not None else material.E
            physical_thickness = 2.0 * material.thickness
            bending_stiffness = bending_E * physical_thickness**3 / 12.0
            local_hinges = _shell_hinges(local_triangles)
            entity_data.append(
                (
                    entity,
                    source_geometry_id,
                    rest_positions,
                    local_triangles,
                    local_Dm_inv,
                    local_rest_areas,
                    local_masses,
                    fixed,
                    targets,
                    effective_shear,
                    bending_stiffness,
                    local_hinges,
                )
            )

        vertex_offset = 0
        body_index = 0
        for env in range(n_envs):
            for (
                entity,
                source_geometry_id,
                rest_positions,
                local_triangles,
                local_Dm_inv,
                local_rest_areas,
                local_masses,
                fixed,
                targets,
                effective_shear,
                bending_stiffness,
                local_hinges,
            ) in entity_data:
                x = rest_positions.copy()
                x[fixed[:, env] != 0] = targets[fixed[:, env] != 0, env]
                n_entity_verts = entity.n_vertices
                n_entity_tris = len(local_triangles)

                positions.append(x)
                velocities.append(np.zeros_like(x))
                masses.append(local_masses)
                is_fixed.append(fixed[:, env])
                gravities.append(np.tile(gravity[env], (n_entity_verts, 1)))
                triangles.append(local_triangles + vertex_offset)
                Dm_inv.append(local_Dm_inv)
                rest_areas.append(local_rest_areas)
                thicknesses.append(np.full(n_entity_verts, entity.material.thickness, dtype=np.float64))
                membrane_mu.append(np.full(n_entity_tris, effective_shear / 2.0, dtype=np.float64))
                membrane_lambda.append(np.full(n_entity_tris, entity.material.E / 2.0, dtype=np.float64))
                strain_limit_multiplier.append(
                    np.full(n_entity_tris, entity.material.strain_limit_multiplier, dtype=np.float64)
                )
                vert_bend_k.append(np.full(n_entity_verts, bending_stiffness, dtype=np.float64))
                if len(local_hinges) != 0:
                    global_hinges = local_hinges + vertex_offset
                    hinge_indices.append(global_hinges)
                    hinge_Q0.append(
                        np.stack(
                            [_quadratic_bending_Q0(x[hinge]) for hinge in local_hinges],
                            axis=0,
                        )
                    )
                    hinge_stiffness.append(np.full(len(local_hinges), bending_stiffness, dtype=np.float64))

                body_ids.append(np.full(n_entity_verts, body_index, dtype=np.int32))
                geometry_ids.append(np.full(n_entity_verts, body_index, dtype=np.int32))
                source_geometry_ids.append(
                    np.full(
                        n_entity_verts,
                        source_geometry_id,
                        dtype=np.int32,
                    )
                )
                geometry_environments.append(np.full(n_entity_verts, env, dtype=np.int32))
                body_environment.append(env)
                body_vertex_offsets.append(body_vertex_offsets[-1] + n_entity_verts)
                bridge_vertex.append(np.arange(entity._v_start, entity._v_start + n_entity_verts, dtype=np.int32))
                bridge_environment.append(np.full(n_entity_verts, env, dtype=np.int32))
                vertex_offset += n_entity_verts
                body_index += 1

        self.positions = np.ascontiguousarray(np.concatenate(positions), dtype=np.float64)
        self.x_prev = self.positions.copy()
        self.velocities = np.ascontiguousarray(np.concatenate(velocities), dtype=np.float64)
        self.masses = np.ascontiguousarray(np.concatenate(masses), dtype=np.float64)
        self.is_fixed = np.ascontiguousarray(np.concatenate(is_fixed), dtype=np.int32)
        self.gravity = np.ascontiguousarray(np.concatenate(gravities), dtype=np.float64)
        self.tri_indices = np.ascontiguousarray(np.concatenate(triangles), dtype=np.int32)
        self.Dm_inv_2d = np.ascontiguousarray(
            np.concatenate(Dm_inv),
            dtype=np.float64,
        )
        self.rest_areas = np.ascontiguousarray(
            np.concatenate(rest_areas),
            dtype=np.float64,
        )
        self.thicknesses = np.ascontiguousarray(np.concatenate(thicknesses), dtype=np.float64)
        self.membrane_mu = np.ascontiguousarray(np.concatenate(membrane_mu), dtype=np.float64)
        self.membrane_lambda = np.ascontiguousarray(np.concatenate(membrane_lambda), dtype=np.float64)
        self.strain_limit_multiplier = np.ascontiguousarray(
            np.concatenate(strain_limit_multiplier),
            dtype=np.float64,
        )
        self.hinge_indices = (
            np.ascontiguousarray(np.concatenate(hinge_indices), dtype=np.int32)
            if hinge_indices
            else np.empty((0, 4), dtype=np.int32)
        )
        self.hinge_Q0 = (
            np.ascontiguousarray(np.concatenate(hinge_Q0), dtype=np.float64)
            if hinge_Q0
            else np.empty((0, 4, 4), dtype=np.float64)
        )
        self.hinge_stiffness = (
            np.ascontiguousarray(np.concatenate(hinge_stiffness), dtype=np.float64)
            if hinge_stiffness
            else np.empty(0, dtype=np.float64)
        )
        self.vert_bend_k = np.ascontiguousarray(np.concatenate(vert_bend_k), dtype=np.float64)
        self.body_ids = np.ascontiguousarray(np.concatenate(body_ids), dtype=np.int32)
        self.geometry_ids = np.ascontiguousarray(
            np.concatenate(geometry_ids),
            dtype=np.int32,
        )
        self.source_geometry_ids = np.ascontiguousarray(
            np.concatenate(source_geometry_ids),
            dtype=np.int32,
        )
        self.geometry_environments = np.ascontiguousarray(
            np.concatenate(geometry_environments),
            dtype=np.int32,
        )
        self.body_vertex_offsets = np.asarray(body_vertex_offsets, dtype=np.int32)
        self.body_environment = np.asarray(body_environment, dtype=np.int32)
        self.self_collision = np.ones(body_index, dtype=np.int32)
        ignorance_ranges = [0]
        ignorance_body_ids: list[int] = []
        for source_environment in self.body_environment:
            ignored = np.flatnonzero(self.body_environment != source_environment)
            ignorance_body_ids.extend(int(index) for index in ignored)
            ignorance_ranges.append(len(ignorance_body_ids))
        self.body_contact_ignorance_ranges = np.asarray(ignorance_ranges, dtype=np.int32)
        self.body_contact_ignorance_body_ids = np.asarray(ignorance_body_ids, dtype=np.int32)

        self.surf_triangles = self.tri_indices
        self.surf_edges = np.ascontiguousarray(_surface_edges(self.tri_indices), dtype=np.int32)
        self.surf_verts = np.arange(len(self.positions), dtype=np.int32)
        self.vert_dimensions = np.full(len(self.positions), 2, dtype=np.int32)
        self.vert_area_weights, self.edge_area_weights, self.face_area_weights = _surface_area_weights(
            self.positions,
            self.surf_triangles,
            self.surf_edges,
        )
        self.bridge_vertex = np.ascontiguousarray(np.concatenate(bridge_vertex), dtype=np.int32)
        self.bridge_environment = np.ascontiguousarray(
            np.concatenate(bridge_environment),
            dtype=np.int32,
        )
        self.scene_elements_v = entities[0]._solver.elements_v
        self.scene_vertex_constraints = scene.fem_solver.vertex_constraints
        self.scene_frame = entities[0]._sim.cur_substep_local
        self.n_verts = len(self.positions)
        self.n_tris = len(self.tri_indices)
        self.n_bodies = body_index
        return True

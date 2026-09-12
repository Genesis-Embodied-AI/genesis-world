from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import torch
import trimesh

import genesis as gs
import genesis.utils.geom as gu
import genesis.utils.mesh as mu
import genesis.utils.particle as pu
from genesis.ext import pyrender
from genesis.ext.pyrender.jit_render import JITRenderer
from genesis.utils.misc import tensor_to_array, qd_to_numpy

if TYPE_CHECKING:
    from genesis.engine.solvers.kinematic_solver import KinematicSolver
    from genesis.engine.solvers.rigid.rigid_solver import RigidSolver


# Magnitude beyond which a transform entry marks a pose that blew up, which the drawn node then stops following
POSE_MAX_ABS = 1e20


class RigidNodeBatch(NamedTuple):
    """Rigid geoms of one solver drawn in one visual mode, placed together from one conversion of their poses."""

    solver: "RigidSolver | KinematicSolver"
    is_visual: bool
    geoms_idx: list[int]
    geoms_uid: list[int]
    is_plane: list[bool]


class SegmentationColorMap:
    def __init__(self, seed: int = 0, to_torch: bool = False):
        self.seed = seed
        self.to_torch = to_torch
        self.idxc_map = {0: -1}
        self.key_map = {-1: 0}
        self.idxc_to_color = None

    def seg_key_to_idxc(self, seg_key):
        seg_idxc = self.key_map.setdefault(seg_key, len(self.key_map))
        self.idxc_map[seg_idxc] = seg_key
        return seg_idxc

    def seg_idxc_to_key(self, seg_idxc):
        return self.idxc_map[seg_idxc]

    def colorize_seg_idxc_arr(self, seg_idxc_arr):
        return self.idxc_to_color[seg_idxc_arr]

    def generate_seg_colors(self):
        # seg_key: same as entity/link/geom's idx
        # seg_idxc: segmentation index of objects
        # seg_idxc_rgb: colorized seg_idxc internally used by renderer

        # Evenly spaced hues
        num_keys = len(self.key_map)
        hues = np.linspace(0.0, 1.0, num_keys, endpoint=False)
        rng = np.random.default_rng(seed=self.seed)
        rng.shuffle(hues)

        # Fixed saturation/value
        s, v = 0.8, 0.95

        # HSV to RGB conversion
        rgb = np.zeros((num_keys, 3), dtype=np.float32)
        i = (hues * 6).astype(np.int32)
        f = hues * 6 - i
        p = v * (1 - s)
        q = v * (1 - f * s)
        t = v * (1 - (1 - f) * s)
        for k in range(1, num_keys):  # Skip first color to enforce black background
            match i[k] % 6:
                case 0:
                    rgb[k] = (v, t[k], p)
                case 1:
                    rgb[k] = (q[k], v, p)
                case 2:
                    rgb[k] = (p, v, t[k])
                case 3:
                    rgb[k] = (p, q[k], v)
                case 4:
                    rgb[k] = (t[k], p, v)
                case 5:
                    rgb[k] = (v, p, q[k])
        rgb = mu.color_f32_to_u8(rgb)

        # Store the generated map
        if self.to_torch:
            self.idxc_to_color = torch.from_numpy(rgb).to(device=gs.device)
        else:
            self.idxc_to_color = rgb


class RasterizerContext:
    def __init__(self, options):
        self.show_world_frame = options.show_world_frame
        self.world_frame_size = options.world_frame_size
        self.show_link_frame = options.show_link_frame
        self.link_frame_size = options.link_frame_size
        self.show_cameras = options.show_cameras
        self.shadow = options.shadow
        self.plane_reflection = options.plane_reflection
        self.ambient_light = options.ambient_light
        self.background_color = options.background_color
        self.segmentation_level = options.segmentation_level
        self.lights = options.lights
        self.visualize_mpm_boundary = options.visualize_mpm_boundary
        self.visualize_sph_boundary = options.visualize_sph_boundary
        self.visualize_pbd_boundary = options.visualize_pbd_boundary
        self.particle_size_scale = options.particle_size_scale
        self.contact_force_scale = options.contact_force_scale
        self.render_particle_as = options.render_particle_as
        self.rendered_envs_idx = list(options.rendered_envs_idx) if options.rendered_envs_idx is not None else None
        self.split_envs = options.split_envs

        # nodes
        self.world_frame_node = None
        self.link_frame_node = None
        self.link_frame_nodes = dict()
        self.frustum_nodes = dict()  # nodes camera frustums
        self.rigid_nodes = dict()
        # The geoms behind the rigid nodes, grouped by solver and visual mode for placement at update time
        self._rigid_batches: dict[tuple["RigidSolver | KinematicSolver", bool], RigidNodeBatch] = {}
        # (env_idx, geom.uid) -> per-env pyrender node for kinematic entities currently driven by set_vverts
        self.vverts_nodes = dict()
        self._per_env_vverts_entity_uids: set = set()
        self.static_nodes = dict()  # used across all frames
        self.dynamic_nodes = dict()  # nodes that live within single frame
        self.external_nodes = dict()  # nodes added by external user
        self.seg_node_map = dict()
        self.seg_color_map = SegmentationColorMap()

        self.init_meshes()

    def init_meshes(self):
        self.world_frame_shown = False
        self.link_frame_shown = False
        self.camera_frustum_shown = False

        self.world_frame_mesh = mu.create_frame(
            origin_radius=0.012,
            axis_radius=0.005,
            axis_length=self.world_frame_size,
            head_radius=0.01,
            head_length=0.03,
        )

        self.link_frame_mesh = trimesh.creation.axis(origin_size=0.03, axis_radius=0.025, axis_length=1.0)
        self.link_frame_mesh.visual.face_colors[:, :3] = (0.7 * self.link_frame_mesh.visual.face_colors[:, :3]).astype(
            int
        )
        self.link_frame_mesh.vertices *= self.link_frame_size

    def build(self, scene):
        self.scene = scene
        self.sim = scene.sim
        self.visualizer = scene.visualizer

        # Update visuals at this point avoids nasty visual artifacts during Scene build
        self.visualizer.update_visual_states()

        if self.rendered_envs_idx is None:
            self.rendered_envs_idx = list(range(self.sim._B))
        self.rendered_envs_mask = np.ones(len(self.rendered_envs_idx), dtype=np.bool_)

        # pyrender scene
        self._scene = pyrender.Scene(
            ambient_light=self.ambient_light, bg_color=self.background_color, n_envs=len(self.rendered_envs_idx)
        )

        self.jit = JITRenderer(self._scene, [], [], self.scene.envs_offset[self.rendered_envs_idx])

        self.on_lights()

        if self.show_world_frame:
            self.on_world_frame()
        if self.show_link_frame:
            self.on_link_frame()
        if self.show_cameras:
            self.on_camera_frustum()

        self.on_tool()
        self.on_rigid()
        self.on_mpm()
        self.on_sph()
        self.on_pbd()
        self.on_fem()

        # segmentation mapping
        self.seg_color_map.generate_seg_colors()

    def destroy(self):
        self.clear_dynamic_nodes(only_outdated=False)

        for node_registry in (
            self.link_frame_nodes,
            self.frustum_nodes,
            self.rigid_nodes,
            self.vverts_nodes,
            self.static_nodes,
            self.external_nodes,
        ):
            for external_node in node_registry.values():
                self.remove_node(external_node)
            node_registry.clear()
        self._per_env_vverts_entity_uids.clear()

    def reset(self):
        self._t = -1

    def add_node(self, obj, **kwargs):
        with self.scene._visualizer.viewer_lock:
            return self._scene.add(obj, **kwargs)

    def remove_node(self, node):
        if self.scene._visualizer is None:
            self._scene.remove_node(node)
        else:
            with self.scene._visualizer.viewer_lock:
                self._scene.remove_node(node)

    def _get_geom_active_envs_idx(self, geom, rendered_envs_idx):
        """Get the intersection of geom.active_envs_idx (for heterogeneous sim) and rendered_envs_idx.

        For heterogeneous simulation, each geom is only active in certain environments.
        This returns the environments where the geom should actually be rendered.
        """
        geom_active_envs_idx = geom.active_envs_idx
        if geom_active_envs_idx is not None:
            return np.intersect1d(geom_active_envs_idx, rendered_envs_idx)
        return rendered_envs_idx

    def add_rigid_node(self, geom, obj, track_pose=True, **kwargs):
        """Add the node drawing a geom. With 'track_pose' the node follows the geom pose in every rendered environment at
        update time, which a node drawn once for all of them does without."""
        rigid_node = self.add_node(obj, **kwargs)
        self.rigid_nodes[geom.uid] = rigid_node
        if track_pose:
            is_visual = geom.entity.surface.vis_mode == "visual"
            batch_key = (geom.solver, is_visual)
            if batch_key not in self._rigid_batches:
                self._rigid_batches[batch_key] = RigidNodeBatch(geom.solver, is_visual, [], [], [])
            batch = self._rigid_batches[batch_key]
            batch.geoms_idx.append(geom.idx)
            batch.geoms_uid.append(geom.uid)
            batch.is_plane.append(isinstance(geom.entity._morph, gs.morphs.Plane))

        # create segemtation id
        if self.segmentation_level == "geom":
            seg_key = (geom.entity.idx, geom.link.idx, geom.idx)
        elif self.segmentation_level == "link":
            seg_key = (geom.entity.idx, geom.link.idx)
        elif self.segmentation_level == "entity":
            seg_key = geom.entity.idx
        else:
            gs.raise_exception(f"Unsupported segmentation level: {self.segmentation_level}")
        self.create_node_seg(seg_key, rigid_node)

    def add_geom_node(self, geom, geoms_T, material=None):
        """Add the node drawing a rigid geom, one instance per rendered environment and present in the environments
        holding the geom, and return its mesh (None when no rendered environment holds it). 'geoms_T' holds the
        transforms of every geom of its solver in its visual mode, as 'rigid_geoms_T' returns them.

        A horizontal plane wider than the environment spacing is one shared floor: one instance where the plane stands,
        in place of one copy per environment fighting over the same depth once laid out on the grid. Drawn where it
        stands, it lies under every environment in a pass drawing one environment as well.
        """
        geom_envs_idx = self._get_geom_active_envs_idx(geom, self.rendered_envs_idx)
        if len(geom_envs_idx) == 0:
            return None

        entity = geom.entity
        vis_mode = entity.surface.vis_mode
        mesh = geom.get_sdf_trimesh() if "sdf" in vis_mode else geom.get_trimesh()
        geom_T = geoms_T[:, geom.idx]
        envs = np.isin(self.rendered_envs_idx, geom_envs_idx)
        is_shared_floor = False
        if envs.all() and isinstance(entity.main_morph, gs.morphs.Plane):
            plane_normal, plane_size = entity.main_morph.normal, entity.main_morph.plane_size
            is_shared_floor = (
                abs(plane_normal[0]) < gs.EPS
                and abs(plane_normal[1]) < gs.EPS
                and self.scene.env_spacing[0] < plane_size[0]
                and self.scene.env_spacing[1] < plane_size[1]
            )
        if is_shared_floor:
            geom_T = geom_T[:1]
            envs = None
        is_collision = "collision" in vis_mode
        mesh_node = pyrender.Mesh.from_trimesh(
            mesh=mesh,
            poses=geom_T,
            smooth=geom.surface.smooth and not is_collision,
            double_sided=geom.surface.double_sided and not is_collision,
            is_floor=isinstance(entity._morph, gs.morphs.Plane),
            envs=envs,
            material=material,
        )
        self.add_rigid_node(geom, mesh_node, track_pose=not is_shared_floor)
        if isinstance(entity._morph, gs.morphs.Plane):
            self.set_reflection_mat(geom_T)
        return mesh_node

    def remove_rigid_node(self, geom):
        """Remove the node drawing a geom, if any: a geom present in no rendered environment has none."""
        rigid_node = self.rigid_nodes.pop(geom.uid, None)
        if rigid_node is None:
            return
        for batch_key, batch in self._rigid_batches.items():
            if geom.uid in batch.geoms_uid:
                i = batch.geoms_uid.index(geom.uid)
                del batch.geoms_idx[i], batch.geoms_uid[i], batch.is_plane[i]
                if not batch.geoms_uid:
                    del self._rigid_batches[batch_key]
                break
        self.remove_node_seg(rigid_node)
        self.remove_node(rigid_node)

    def add_static_node(self, entity, obj, i_b, **kwargs):
        static_node = self.add_node(obj, **kwargs)
        self.static_nodes[(i_b, entity.uid)] = static_node
        self.create_node_seg(entity.idx, static_node)

    def add_dynamic_node(self, entity, obj, **kwargs):
        if obj:
            dynamic_node = self.add_node(obj, **kwargs)
            self.dynamic_nodes.setdefault(self.scene.sim.cur_step_global, []).append(dynamic_node)
        else:
            dynamic_node = None
        if entity:
            self.create_node_seg(entity.idx, dynamic_node)

    def add_external_node(self, obj, **kwargs):
        # Check if the node has a valid name
        if not hasattr(obj, "name") or not obj.name:
            gs.raise_exception("Node must have a valid 'name' attribute.")

        # Check if the name is already in use
        if obj.name in self.external_nodes:
            gs.raise_exception(f"A node with the name '{obj.name}' already exists.")

        self.external_nodes[obj.name] = self.add_node(obj, **kwargs)

    def clear_dynamic_nodes(self, only_outdated: bool = True):
        for t in tuple(self.dynamic_nodes.keys()):
            if not only_outdated or t < self.scene.sim.cur_step_global:
                for dynamic_node in self.dynamic_nodes.pop(t):
                    self.remove_node_seg(dynamic_node)
                    self.remove_node(dynamic_node)

    def clear_external_node(self, node):
        if node.name in self.external_nodes:
            self.remove_node(self.external_nodes[node.name])
            del self.external_nodes[node.name]

    def clear_external_nodes(self):
        for external_node in self.external_nodes.values():
            self.remove_node(external_node)
        self.external_nodes.clear()

    def set_node_pose(self, node, pose):
        self._scene.set_pose(node, pose)

    def update_camera_frustum(self, camera):
        if self.camera_frustum_shown:
            self.set_node_pose(self.frustum_nodes[camera.uid], camera.transform)

    def on_camera_frustum(self):
        if not self.camera_frustum_shown:
            for camera in self.cameras:
                self.frustum_nodes[camera.uid] = self.add_node(
                    pyrender.Mesh.from_trimesh(
                        mu.create_camera_frustum(camera, color=(1.0, 1.0, 1.0, 0.3)), smooth=False
                    )
                )
            self.camera_frustum_shown = True

    def off_camera_frustum(self):
        if self.camera_frustum_shown:
            for camera in self.cameras:
                self.remove_node(self.frustum_nodes[camera.uid])
            self.frustum_nodes.clear()
            self.camera_frustum_shown = False

    def on_world_frame(self):
        if not self.world_frame_shown:
            mesh = pyrender.Mesh.from_trimesh(self.world_frame_mesh, smooth=True, is_marker=True)
            self.world_frame_node = self.add_node(mesh)
            self.world_frame_shown = True

    def off_world_frame(self):
        if self.world_frame_shown:
            self.remove_node(self.world_frame_node)
            self.world_frame_node = None
            self.world_frame_shown = False

    def _states_T(self, states):
        """4x4 transform of every element of a solver state (links, geoms or visual geoms) in every rendered
        environment, shaped (n_envs, n_elements, 4, 4)."""
        pos = qd_to_numpy(states.pos, self.rendered_envs_idx, transpose=True)
        quat = qd_to_numpy(states.quat, self.rendered_envs_idx, transpose=True)
        return gu.trans_quat_to_T(pos, quat)

    def rigid_geoms_T(self, solver, is_visual):
        """4x4 transform of every geom of the solver (its visual geoms when 'is_visual') in every rendered environment,
        shaped (n_envs, n_geoms, 4, 4)."""
        return self._states_T(solver.dyn_state.vgeoms if is_visual else solver.dyn_state.geoms)

    def _link_frame_grid_T(self):
        """Transforms of the link frames of every rigid solver, laid out on the environment grid, stacked in one array
        for the single node drawing them all when the environments are rendered together."""
        links_T = [self._states_T(solver.dyn_state.links) for solver in self._rigid_solvers()]
        if not links_T:
            return None
        links_T = np.concatenate(links_T, axis=1)
        links_T[..., :3, 3] += self.scene.envs_offset[self.rendered_envs_idx, np.newaxis]
        return links_T.reshape((-1, 4, 4))

    def on_link_frame(self):
        if not self.link_frame_shown:
            if self.split_envs:
                for solver in self._rigid_solvers():
                    links_T = self._states_T(solver.dyn_state.links)
                    for i, link in enumerate(solver.links):
                        mesh = pyrender.Mesh.from_trimesh(
                            mesh=self.link_frame_mesh, poses=links_T[:, i], is_marker=True, envs=self.rendered_envs_mask
                        )
                        self.link_frame_nodes[link.uid] = self.add_node(mesh)
            else:
                links_T = self._link_frame_grid_T()
                if links_T is not None:
                    mesh = pyrender.Mesh.from_trimesh(mesh=self.link_frame_mesh, poses=links_T, is_marker=True)
                    self.link_frame_node = self.add_node(mesh)
            self.link_frame_shown = True

    def off_link_frame(self):
        if self.link_frame_shown:
            if self.split_envs:
                for node in self.link_frame_nodes.values():
                    self.remove_node(node)
                self.link_frame_nodes.clear()
            elif self.link_frame_node is not None:
                self.remove_node(self.link_frame_node)
                self.link_frame_node = None
            self.link_frame_shown = False

    def update_link_frame(self):
        if self.link_frame_shown:
            if self.split_envs:
                for solver in self._rigid_solvers():
                    links_T = self._states_T(solver.dyn_state.links)
                    for i, link in enumerate(solver.links):
                        link_T = links_T[:, i]
                        node = self.link_frame_nodes[link.uid]
                        node.mesh.primitives[0].poses = link_T
                        self.jit.update_buffer(node, "model", link_T.transpose((0, 2, 1)))
            elif self.link_frame_node is not None:
                links_T = self._link_frame_grid_T()
                self.link_frame_node.mesh.primitives[0].poses = links_T
                self.jit.update_buffer(self.link_frame_node, "model", links_T.transpose((0, 2, 1)))

    def on_tool(self):
        if self.sim.tool_solver.is_active:
            for tool_entity in self.sim.tool_solver.entities:
                if tool_entity.mesh is not None:
                    for env_i, idx in enumerate(self.rendered_envs_idx):
                        mesh = trimesh.Trimesh(
                            tool_entity.mesh.raw_vertices,
                            tool_entity.mesh.faces_np.reshape([-1, 3]),
                            tool_entity.mesh.raw_vertex_normals,
                            process=False,
                        )
                        mesh.visual = mu.surface_uvs_to_trimesh_visual(tool_entity.surface, n_verts=len(mesh.vertices))

                        pose = gu.trans_quat_to_T(tool_entity.init_pos, tool_entity.init_quat)
                        double_sided = tool_entity.surface.double_sided
                        self.add_static_node(
                            tool_entity,
                            pyrender.Mesh.from_trimesh(mesh, double_sided=double_sided, envs=env_i),
                            i_b=idx,
                            pose=pose,
                        )

    def update_tool(self):
        if self.sim.tool_solver.is_active:
            for tool_entity in self.sim.tool_solver.entities:
                poss = qd_to_numpy(tool_entity.pos)[self.sim.cur_substep_local]
                quats = qd_to_numpy(tool_entity.quat)[self.sim.cur_substep_local]
                for idx in self.rendered_envs_idx:
                    pose = gu.trans_quat_to_T(poss[idx], quats[idx])
                    self.set_node_pose(self.static_nodes[(idx, tool_entity.uid)], pose=pose)

    def set_reflection_mat(self, geom_T):
        height = geom_T[0, 2, 3]
        self.jit.reflection_mat = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, height * 2],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

    def _rigid_solvers(self):
        """Yield active solvers that manage KinematicEntity-based entities (rigid + kinematic)."""
        if self.sim.rigid_solver.is_active:
            yield self.sim.rigid_solver
        if self.sim.kinematic_solver.is_active:
            yield self.sim.kinematic_solver

    def on_rigid(self):
        # Reuse a single pyrender material per genesis surface so that geoms sharing one surface (e.g. the many
        # textured submeshes of a GLB, which are kept separate to preserve baked convex decompositions) expose the
        # same Texture instances. The renderer then keeps a single host copy and GPU upload instead of one per geom.
        vis_materials = {}
        for solver in self._rigid_solvers():
            # One conversion of the geom poses per visual mode in use
            geoms_T_by_mode = {}
            for entity in solver.entities:
                is_visual = entity.surface.vis_mode == "visual"
                geoms = entity.vgeoms if is_visual else entity.geoms
                if is_visual not in geoms_T_by_mode:
                    geoms_T_by_mode[is_visual] = self.rigid_geoms_T(solver, is_visual)
                geoms_T = geoms_T_by_mode[is_visual]

                for geom in geoms:
                    surface_key = id(geom.surface)
                    mesh_node = self.add_geom_node(geom, geoms_T, material=vis_materials.get(surface_key))
                    if mesh_node is not None:
                        vis_materials.setdefault(surface_key, mesh_node.primitives[0].material)

    def update_rigid(self):
        for solver in self._rigid_solvers():
            for entity in solver.entities:
                if entity.surface.vis_mode == "visual" and entity._morph.enable_custom_vverts:
                    if entity.uid not in self._per_env_vverts_entity_uids:
                        # Seed primitive.positions with the current vverts: buffer updates bypass primitive.positions,
                        # which seeds the bounds of the geometry (shadow map extents)
                        vverts = qd_to_numpy(solver.dyn_state.vverts.pos, self.rendered_envs_idx, transpose=True)
                        custom_offset = entity._custom_vvert_start - entity._vvert_start
                        for geom in entity.vgeoms:
                            self.remove_rigid_node(geom)

                            geom_envs_idx = self._get_geom_active_envs_idx(geom, self.rendered_envs_idx)
                            if len(geom_envs_idx) == 0:
                                continue

                            mesh = geom.get_trimesh()
                            v_start = geom.vvert_start + custom_offset
                            v_end = geom.vvert_end + custom_offset
                            for i_b in geom_envs_idx:
                                env_i = self.rendered_envs_idx.index(i_b)
                                node = self.add_node(
                                    pyrender.Mesh.from_trimesh(
                                        mesh=mesh,
                                        smooth=geom.surface.smooth,
                                        double_sided=geom.surface.double_sided,
                                        envs=env_i,
                                    )
                                )
                                geom_vverts = vverts[env_i, v_start:v_end, :]
                                node.mesh.primitives[0].positions = self._scene.reorder_vertices(
                                    node, geom_vverts.astype(np.float32)
                                )
                                self.vverts_nodes[(i_b, geom.uid)] = node
                                if self.segmentation_level == "geom":
                                    seg_key = (geom.entity.idx, geom.link.idx, geom.idx)
                                elif self.segmentation_level == "link":
                                    seg_key = (geom.entity.idx, geom.link.idx)
                                elif self.segmentation_level == "entity":
                                    seg_key = geom.entity.idx
                                else:
                                    gs.raise_exception(f"Unsupported segmentation level: {self.segmentation_level}")
                                self.create_node_seg(seg_key, node)
                        self._per_env_vverts_entity_uids.add(entity.uid)

                    vverts = qd_to_numpy(solver.dyn_state.vverts.pos, self.rendered_envs_idx, transpose=True)
                    custom_offset = entity._custom_vvert_start - entity._vvert_start
                    for geom in entity.vgeoms:
                        geom_envs_idx = self._get_geom_active_envs_idx(geom, self.rendered_envs_idx)
                        if len(geom_envs_idx) == 0:
                            continue
                        v_start = geom.vvert_start + custom_offset
                        v_end = geom.vvert_end + custom_offset
                        for env_i, i_b in enumerate(self.rendered_envs_idx):
                            if i_b not in geom_envs_idx:
                                continue
                            node = self.vverts_nodes[(i_b, geom.uid)]
                            geom_vverts = vverts[env_i, v_start:v_end, :]
                            update_data = self._scene.reorder_vertices(node, geom_vverts.astype(np.float32))
                            self.jit.update_buffer(node, "pos", update_data)
                            normal_data = self.jit.update_normal(node, update_data)
                            if normal_data is not None:
                                self.jit.update_buffer(node, "normal", normal_data)

        # One conversion of the geom poses per solver and visual mode, and one transposed copy the buffer flush uploads
        # as is: each node then takes a view of both, and the reductions the draw order and the scene bounds need run
        # over the stack
        for batch in self._rigid_batches.values():
            geoms_T = self.rigid_geoms_T(batch.solver, batch.is_visual)[:, batch.geoms_idx].transpose((1, 0, 2, 3))
            # A geom whose pose blew up keeps the transform it was last drawn with, rather than vanishing and dragging
            # the scene bounds, and the shadow map fitted to them, along
            is_pose_valid = (np.abs(geoms_T) < POSE_MAX_ABS).all(axis=(-2, -1))
            if not is_pose_valid.all():
                geoms_T_prev = np.stack([self.rigid_nodes[uid].mesh.primitives[0].poses for uid in batch.geoms_uid])
                geoms_T[~is_pose_valid] = geoms_T_prev[~is_pose_valid]
            models = np.ascontiguousarray(geoms_T.transpose((0, 1, 3, 2)))
            for geom_T, model, geom_uid in zip(geoms_T, models, batch.geoms_uid):
                node = self.rigid_nodes[geom_uid]
                node.mesh._bounds = None
                node.mesh.primitives[0].poses = geom_T
                self.jit.update_buffer(node, "model", model)
            for i_plane in np.flatnonzero(batch.is_plane):
                self.set_reflection_mat(geoms_T[i_plane])

    def update_contact(self):
        if self.sim.rigid_solver.is_active and any(link.visualize_contact for link in self.sim.rigid_solver.links):
            # Extract all contact information at once
            contacts_info_all = self.sim.rigid_solver.collider.get_contacts(as_tensor=False, to_torch=False)

            # Scale contact arrows by the parent link's overall size rather than the individual contacting geom, so
            # they stay legible on links built from many small convex-decomposition pieces. The per-geom init AABBs
            # are in each geom's local frame, so offset their corners by the geom pose to get the link-frame extent.
            # This diagonal is constant, so compute it once and cache it.
            geoms_aabb = qd_to_numpy(self.sim.rigid_solver.geoms_init_AABB)
            links_init_AABB_size = np.zeros(self.sim.rigid_solver.n_links, dtype=gs.np_float)
            for link in self.sim.rigid_solver.links:
                if link.n_geoms == 0:
                    continue
                lower = np.full(3, np.inf, dtype=gs.np_float)
                upper = np.full(3, -np.inf, dtype=gs.np_float)
                for geom in link.geoms:
                    corners = gu.transform_by_trans_quat(geoms_aabb[geom.idx], geom.init_pos, geom.init_quat)
                    lower = np.minimum(lower, corners.min(axis=0))
                    upper = np.maximum(upper, corners.max(axis=0))
                links_init_AABB_size[link.idx] = np.linalg.norm(upper - lower)

            for env_i, batch_idx in enumerate(self.rendered_envs_idx):
                if self.sim.rigid_solver.n_envs > 0:
                    contacts_info = {key: value[batch_idx] for key, value in contacts_info_all.items()}
                else:
                    contacts_info = contacts_info_all

                n_contacts = len(contacts_info["geom_a"])
                if n_contacts == 0:
                    continue

                la_size = links_init_AABB_size[contacts_info["link_a"]]
                lb_size = links_init_AABB_size[contacts_info["link_b"]]
                arrow_scale = np.minimum(la_size, lb_size)
                radius = np.minimum(arrow_scale * 0.04, 0.005)
                contact_pos = contacts_info["position"]
                contact_normal_scaled = contacts_info["normal"] * arrow_scale[:, None]
                contact_force = contacts_info["force"]

                for i_c in range(n_contacts):
                    for link_idx, sign in ((contacts_info["link_a"][i_c], -1), (contacts_info["link_b"][i_c], 1)):
                        if self.sim.rigid_solver.links[link_idx].visualize_contact:
                            self.draw_contact_arrow(
                                pos=contact_pos[i_c], radius=radius[i_c], force=sign * contact_force[i_c], env_idx=env_i
                            )
                            self.draw_debug_arrow(
                                pos=contact_pos[i_c],
                                radius=radius[i_c],
                                vec=-sign * contact_normal_scaled[i_c],
                                color=(0.9, 0.0, 0.8, 1.0),
                                persistent=False,
                                env_idx=env_i,
                            )

    def on_mpm(self):
        if self.sim.mpm_solver.is_active:
            for mpm_entity in self.sim.mpm_solver.entities:
                if mpm_entity.surface.vis_mode in ("recon", "visual"):
                    self.add_dynamic_node(mpm_entity, None)
                elif mpm_entity.surface.vis_mode == "particle":
                    for env_i, idx in enumerate(self.rendered_envs_idx):
                        mesh = mu.create_sphere(
                            self.sim.mpm_solver.particle_radius * self.particle_size_scale, subdivisions=1
                        )
                        mesh.visual = mu.surface_uvs_to_trimesh_visual(mpm_entity.surface, n_verts=len(mesh.vertices))

                        tfs = np.tile(np.eye(4), (mpm_entity.n_particles, 1, 1))
                        tfs[:, :3, 3] = mpm_entity.init_particles
                        self.add_static_node(
                            mpm_entity, pyrender.Mesh.from_trimesh(mesh, smooth=True, poses=tfs, envs=env_i), i_b=idx
                        )

            # boundary
            if self.visualize_mpm_boundary:
                for env_i in range(len(self.rendered_envs_idx)):
                    self.add_node(
                        pyrender.Mesh.from_trimesh(
                            mu.create_box(
                                bounds=np.array(
                                    [self.sim.mpm_solver.boundary.lower, self.sim.mpm_solver.boundary.upper],
                                    dtype=np.float32,
                                ),
                                wireframe=True,
                                color=(1.0, 1.0, 0.0, 1.0),
                            ),
                            smooth=True,
                            envs=env_i,
                        )
                    )

    def update_mpm(self):
        if self.sim.mpm_solver.is_active:
            particles_all = qd_to_numpy(self.sim.mpm_solver.particles_render.pos)
            active_all = qd_to_numpy(self.sim.mpm_solver.particles_render.active).astype(dtype=np.bool_, copy=False)
            vverts_all = qd_to_numpy(self.sim.mpm_solver.vverts_render.pos)
            for mpm_entity in self.sim.mpm_solver.entities:
                for env_i, idx in enumerate(self.rendered_envs_idx):
                    if mpm_entity.surface.vis_mode == "recon":
                        mesh = pu.particles_to_mesh(
                            positions=particles_all[mpm_entity.particle_start : mpm_entity.particle_end, idx][
                                active_all[mpm_entity.particle_start : mpm_entity.particle_end, idx]
                            ],
                            radius=self.sim.mpm_solver.particle_radius,
                            backend=mpm_entity.surface.recon_backend,
                        )
                        mesh.visual = mu.surface_uvs_to_trimesh_visual(mpm_entity.surface, n_verts=len(mesh.vertices))
                        self.add_dynamic_node(mpm_entity, pyrender.Mesh.from_trimesh(mesh, smooth=True, envs=env_i))
                    elif mpm_entity.surface.vis_mode == "particle":
                        tfs = np.tile(np.eye(4), (mpm_entity.n_particles, 1, 1))
                        tfs[:, :3, 3] = particles_all[mpm_entity.particle_start : mpm_entity.particle_end, idx]

                        node = self.static_nodes[(idx, mpm_entity.uid)]
                        self.jit.update_buffer(node, "model", tfs.transpose((0, 2, 1)))

                    elif mpm_entity.surface.vis_mode == "visual":
                        mpm_entity._vmesh.trimesh.vertices = vverts_all[
                            mpm_entity.vvert_start : mpm_entity.vvert_end, idx
                        ]
                        self.add_dynamic_node(
                            mpm_entity,
                            pyrender.Mesh.from_trimesh(
                                mpm_entity.vmesh.trimesh, smooth=mpm_entity.surface.smooth, envs=env_i
                            ),
                        )

    def on_sph(self):
        if self.sim.sph_solver.is_active:
            for sph_entity in self.sim.sph_solver.entities:
                if sph_entity.surface.vis_mode == "recon":
                    self.add_dynamic_node(sph_entity, None)
                elif sph_entity.surface.vis_mode == "particle":
                    for env_i, idx in enumerate(self.rendered_envs_idx):
                        mesh = mu.create_sphere(
                            self.sim.sph_solver.particle_radius * self.particle_size_scale, subdivisions=1
                        )
                        mesh.visual = mu.surface_uvs_to_trimesh_visual(sph_entity.surface, n_verts=len(mesh.vertices))

                        tfs = np.tile(np.eye(4), (sph_entity.n_particles, 1, 1))
                        tfs[:, :3, 3] = sph_entity.init_particles
                        self.add_static_node(
                            sph_entity, pyrender.Mesh.from_trimesh(mesh, smooth=True, poses=tfs, envs=env_i), i_b=idx
                        )

            # boundary
            if self.visualize_sph_boundary:
                for env_i in range(len(self.rendered_envs_idx)):
                    self.add_node(
                        pyrender.Mesh.from_trimesh(
                            mu.create_box(
                                bounds=np.array(
                                    [self.sim.sph_solver.boundary.lower, self.sim.sph_solver.boundary.upper],
                                    dtype=np.float32,
                                ),
                                wireframe=True,
                                color=(0.0, 1.0, 1.0, 1.0),
                            ),
                            smooth=True,
                            envs=env_i,
                        )
                    )

    def update_sph(self):
        if self.sim.sph_solver.is_active:
            particles_all = qd_to_numpy(self.sim.sph_solver.particles_render.pos)
            active_all = qd_to_numpy(self.sim.sph_solver.particles_render.active).astype(dtype=np.bool_, copy=False)

            for sph_entity in self.sim.sph_solver.entities:
                for env_i, idx in enumerate(self.rendered_envs_idx):
                    if sph_entity.surface.vis_mode == "recon":
                        mesh = pu.particles_to_mesh(
                            positions=particles_all[sph_entity.particle_start : sph_entity.particle_end, idx][
                                active_all[sph_entity.particle_start : sph_entity.particle_end, idx]
                            ],
                            radius=self.sim.sph_solver.particle_radius,
                            backend=sph_entity.surface.recon_backend,
                        )
                        mesh.visual = mu.surface_uvs_to_trimesh_visual(sph_entity.surface, n_verts=len(mesh.vertices))
                        self.add_dynamic_node(sph_entity, pyrender.Mesh.from_trimesh(mesh, smooth=True, envs=env_i))
                    elif sph_entity.surface.vis_mode == "particle":
                        tfs = np.tile(np.eye(4), (sph_entity.n_particles, 1, 1))
                        tfs[:, :3, 3] = particles_all[sph_entity.particle_start : sph_entity.particle_end, idx]

                        node = self.static_nodes[(idx, sph_entity.uid)]
                        self.jit.update_buffer(node, "model", tfs.transpose((0, 2, 1)))

    def on_pbd(self):
        if self.sim.pbd_solver.is_active:
            for pbd_entity in self.sim.pbd_solver.entities:
                if pbd_entity.surface.vis_mode == "visual":
                    # Apply surface visual with UVs to the trimesh
                    pbd_entity.vmesh.trimesh.visual = mu.surface_uvs_to_trimesh_visual(
                        pbd_entity.surface, uvs=pbd_entity.vmesh.uvs, n_verts=len(pbd_entity.vmesh.trimesh.vertices)
                    )
                for env_i, idx in enumerate(self.rendered_envs_idx):
                    if pbd_entity.surface.vis_mode == "recon":
                        self.add_dynamic_node(pbd_entity, None)
                    elif pbd_entity.surface.vis_mode == "particle":
                        if self.render_particle_as == "sphere":
                            mesh = mu.create_sphere(
                                self.sim.pbd_solver.particle_radius * self.particle_size_scale, subdivisions=1
                            )
                            mesh.visual = mu.surface_uvs_to_trimesh_visual(
                                pbd_entity.surface, n_verts=len(mesh.vertices)
                            )
                            tfs = np.tile(np.eye(4), (pbd_entity.n_particles, 1, 1))
                            tfs[:, :3, 3] = pbd_entity.init_particles
                            self.add_static_node(
                                pbd_entity,
                                pyrender.Mesh.from_trimesh(mesh, smooth=True, poses=tfs, envs=env_i),
                                i_b=idx,
                            )
                        elif self.render_particle_as == "tet":
                            mesh = mu.create_tets_mesh(
                                pbd_entity.n_particles, self.sim.pbd_solver.particle_radius * self.particle_size_scale
                            )
                            mesh.visual = mu.surface_uvs_to_trimesh_visual(
                                pbd_entity.surface, n_verts=len(mesh.vertices)
                            )
                            pbd_entity._tets_mesh = mesh
                            self.add_static_node(
                                pbd_entity, pyrender.Mesh.from_trimesh(mesh, smooth=False, envs=env_i), i_b=idx
                            )
                    elif pbd_entity.surface.vis_mode == "visual":
                        self.add_static_node(
                            pbd_entity,
                            pyrender.Mesh.from_trimesh(
                                pbd_entity.vmesh.trimesh,
                                smooth=pbd_entity.surface.smooth,
                                double_sided=pbd_entity._surface.double_sided,
                                envs=env_i,
                            ),
                            i_b=idx,
                        )

            # boundary
            if self.visualize_pbd_boundary:
                self.add_node(
                    pyrender.Mesh.from_trimesh(
                        mu.create_box(
                            bounds=np.array(
                                [self.sim.pbd_solver.boundary.lower, self.sim.pbd_solver.boundary.upper],
                                dtype=np.float32,
                            ),
                            wireframe=True,
                            color=(0.0, 1.0, 1.0, 1.0),
                        ),
                        smooth=True,
                    )
                )

    def update_pbd(self):
        if self.sim.pbd_solver.is_active:
            particles_all = qd_to_numpy(self.sim.pbd_solver.particles_render.pos)
            particles_vel_all = qd_to_numpy(self.sim.pbd_solver.particles_render.vel)
            active_all = qd_to_numpy(self.sim.pbd_solver.particles_render.active).astype(dtype=np.bool_, copy=False)
            vverts_all = qd_to_numpy(self.sim.pbd_solver.vverts_render.pos)
            for pbd_entity in self.sim.pbd_solver.entities:
                for env_i, idx in enumerate(self.rendered_envs_idx):
                    particles_env = particles_all[:, idx]
                    particles_vel_env = particles_vel_all[:, idx]
                    active_env = active_all[:, idx]
                    vverts_env = vverts_all[:, idx]

                    if pbd_entity.surface.vis_mode == "recon":
                        positions = particles_env[pbd_entity.particle_start : pbd_entity.particle_end][
                            active_env[pbd_entity.particle_start : pbd_entity.particle_end]
                        ]
                        mesh = pu.particles_to_mesh(
                            positions=positions,
                            radius=self.sim.mpm_solver.particle_radius,
                            backend=pbd_entity.surface.recon_backend,
                        )
                        mesh.visual = mu.surface_uvs_to_trimesh_visual(pbd_entity.surface, n_verts=len(mesh.vertices))
                        self.add_dynamic_node(pbd_entity, pyrender.Mesh.from_trimesh(mesh, smooth=True, envs=env_i))

                    # TODO: need to support multi-env visulaization for tet mode (it's using static node)
                    elif pbd_entity.surface.vis_mode == "particle":
                        if self.render_particle_as == "sphere":
                            tfs = np.tile(np.eye(4), (pbd_entity.n_particles, 1, 1))
                            tfs[:, :3, 3] = particles_env[pbd_entity.particle_start : pbd_entity.particle_end]

                            node = self.static_nodes[(idx, pbd_entity.uid)]
                            self.jit.update_buffer(node, "model", tfs.transpose((0, 2, 1)))

                        elif self.render_particle_as == "tet":
                            new_verts = mu.transform_tets_mesh_verts(
                                pbd_entity._tets_mesh.vertices,
                                positions=particles_env[pbd_entity.particle_start : pbd_entity.particle_end],
                                zs=particles_vel_env[pbd_entity.particle_start : pbd_entity.particle_end],
                            )
                            node = self.static_nodes[(idx, pbd_entity.uid)]
                            update_data = self._scene.reorder_vertices(node, new_verts.astype(np.float32))
                            self.jit.update_buffer(node, "pos", update_data)
                            normal_data = self.jit.update_normal(node, update_data)
                            if normal_data is not None:
                                self.jit.update_buffer(node, "normal", normal_data)
                    elif pbd_entity.surface.vis_mode == "visual":
                        vverts = vverts_env[pbd_entity.vvert_start : pbd_entity.vvert_end]
                        node = self.static_nodes[(idx, pbd_entity.uid)]
                        update_data = self._scene.reorder_vertices(node, vverts.astype(np.float32))
                        self.jit.update_buffer(node, "pos", update_data)
                        normal_data = self.jit.update_normal(node, update_data)
                        if normal_data is not None:
                            self.jit.update_buffer(node, "normal", normal_data)

    def on_fem(self):
        if self.sim.fem_solver.is_active:
            vverts_pos, _, _ = self.sim.fem_solver.get_state_render(self.sim.cur_substep_local)
            vverts_all = qd_to_numpy(vverts_pos, self.rendered_envs_idx, transpose=True)

            for fem_entity in self.sim.fem_solver.entities:
                if fem_entity.surface.vis_mode != "visual":
                    continue

                for i_g, vgeom in enumerate(fem_entity.vgeoms):
                    visual = mu.surface_uvs_to_trimesh_visual(vgeom.surface, uvs=vgeom.uvs, n_verts=vgeom.n_vverts)
                    seg_key = (fem_entity.idx, i_g) if self.segmentation_level == "geom" else fem_entity.idx
                    vverts = vverts_all[:, vgeom.vvert_start : vgeom.vvert_end]
                    for env_i, i_b in enumerate(self.rendered_envs_idx):
                        mesh = trimesh.Trimesh(vverts[env_i], vgeom.vmesh.faces, process=False)
                        mesh.visual = visual
                        node = pyrender.Mesh.from_trimesh(
                            mesh, smooth=vgeom.surface.smooth, double_sided=vgeom.surface.double_sided, envs=env_i
                        )
                        static_node = self.add_node(node)
                        self.static_nodes[(i_b, vgeom.uid)] = static_node
                        self.create_node_seg(seg_key, static_node)

    def update_fem(self):
        if self.sim.fem_solver.is_active:
            vverts_pos, _, _ = self.sim.fem_solver.get_state_render(self.sim.cur_substep_local)
            vverts_all = qd_to_numpy(vverts_pos, self.rendered_envs_idx, transpose=True)

            for fem_entity in self.sim.fem_solver.entities:
                if fem_entity.surface.vis_mode != "visual":
                    continue

                for vgeom in fem_entity.vgeoms:
                    vverts = vverts_all[:, vgeom.vvert_start : vgeom.vvert_end]
                    for env_i, i_b in enumerate(self.rendered_envs_idx):
                        node = self.static_nodes[(i_b, vgeom.uid)]
                        render_verts = vverts[env_i].astype(np.float32, copy=False)
                        update_data = self._scene.reorder_vertices(node, render_verts)
                        self.jit.update_buffer(node, "pos", update_data)
                        normal_data = self.jit.update_normal(node, update_data)
                        if normal_data is not None:
                            self.jit.update_buffer(node, "normal", normal_data)

    def update_sensors(self):
        self.sim._sensor_manager.draw_debug(self)

    def on_lights(self):
        for light in self.lights:
            self.add_light(light)

    def draw_debug_line(self, start, end, radius=0.002, color=(1.0, 0.0, 0.0, 1.0)):
        mesh = mu.create_line(
            tensor_to_array(start, dtype=np.float32), tensor_to_array(end, dtype=np.float32), radius, color
        )
        node = pyrender.Mesh.from_trimesh(mesh, name=f"debug_line_{gs.UID()}", is_marker=True)
        self.add_external_node(node)
        return node

    def draw_debug_arrow(
        self, pos, vec=(0.0, 0.0, 1.0), radius=0.006, color=(1.0, 0.0, 0.0, 0.5), persistent=True, env_idx=None
    ):
        vec = tensor_to_array(vec, dtype=np.float32)
        length = np.linalg.norm(vec)
        if length > gs.EPS:
            mesh = mu.create_arrow(length=length, radius=radius, body_color=color, head_color=color)

            # An arrow of one environment is drawn in that environment alone, where that environment is. An arrow
            # without environment is shared by all of them.
            pos = tensor_to_array(pos)
            if env_idx is not None:
                poses = np.zeros((len(self.rendered_envs_idx), 4, 4), dtype=np.float32)
                gu.trans_R_to_T(pos, gu.z_up_to_R(vec), out=poses[env_idx])
                envs = np.arange(len(self.rendered_envs_idx)) == env_idx
            else:
                poses = np.zeros((1, 4, 4), dtype=np.float32)
                gu.trans_R_to_T(pos, gu.z_up_to_R(vec), out=poses[0])
                envs = None

            node = pyrender.Mesh.from_trimesh(
                mesh, name=f"debug_arrow_{gs.UID()}", poses=poses, is_marker=True, envs=envs
            )
            if persistent:
                self.add_external_node(node)
            else:
                self.add_dynamic_node(None, node)
            return node

    def draw_debug_frame(self, T, axis_length=1.0, origin_size=0.015, axis_radius=0.01, color=None):
        mesh = trimesh.creation.axis(origin_size=origin_size, axis_radius=axis_radius, axis_length=axis_length)
        if color is not None:
            visual = trimesh.visual.ColorVisuals()
            visual._data["vertex_colors"] = np.tile(mu.color_f32_to_u8(color), (len(mesh.vertices), 1))
            mesh.visual = visual

        n_envs = len(self.rendered_envs_idx)
        poses = tensor_to_array(T)
        if poses.ndim != 3:
            poses = np.tile(poses[np.newaxis], (n_envs, 1, 1))
        assert len(poses) == n_envs, "Inconsistent batch size."

        node = pyrender.Mesh.from_trimesh(
            mesh, name=f"debug_frame_{gs.UID()}", poses=poses, is_marker=True, envs=self.rendered_envs_mask
        )
        self.add_external_node(node)
        return node

    def draw_debug_frames(self, poses, axis_length=1.0, origin_size=0.015, axis_radius=0.01, color=None):
        mesh = trimesh.creation.axis(origin_size=origin_size, axis_radius=axis_radius, axis_length=axis_length)
        if color is not None:
            visual = trimesh.visual.ColorVisuals()
            visual._data["vertex_colors"] = np.tile(mu.color_f32_to_u8(color), (len(mesh.vertices), 1))
            mesh.visual = visual
        node = pyrender.Mesh.from_trimesh(mesh, name=f"debug_frame_{gs.UID()}", poses=poses, is_marker=True)
        self.add_external_node(node)
        return node

    def draw_debug_mesh(self, mesh, pos=np.zeros(3), T=None):
        n_envs = len(self.rendered_envs_idx)
        if T is None:
            T = gu.trans_to_T(pos)
        poses = tensor_to_array(T)
        if poses.ndim != 3:
            poses = np.tile(poses[np.newaxis], (n_envs, 1, 1))
        assert len(poses) == n_envs, "Inconsistent batch size."

        node = pyrender.Mesh.from_trimesh(
            mesh, name=f"debug_mesh_{gs.UID()}", poses=poses, is_marker=True, envs=self.rendered_envs_mask
        )
        self.add_external_node(node)
        return node

    def draw_contact_arrow(self, pos, radius=0.005, force=(0, 0, 1), color=(0.0, 0.9, 0.8, 1.0), env_idx=None):
        length = (tensor_to_array(force) * self.contact_force_scale,)
        self.draw_debug_arrow(pos, length, radius, color=color, persistent=False, env_idx=env_idx)

    def draw_debug_sphere(self, pos, radius=0.01, color=(1.0, 0.0, 0.0, 0.5), persistent=True):
        mesh = mu.create_sphere(radius=radius, color=color)

        n_envs = len(self.rendered_envs_idx)
        poses = gu.trans_to_T(tensor_to_array(pos))
        if poses.ndim != 3:
            poses = np.tile(poses[np.newaxis], (n_envs, 1, 1))
        assert len(poses) == n_envs, "Inconsistent batch size."

        node = pyrender.Mesh.from_trimesh(
            mesh,
            name=f"debug_sphere_{gs.UID()}",
            smooth=True,
            poses=poses,
            is_marker=True,
            envs=self.rendered_envs_mask,
        )
        if persistent:
            self.add_external_node(node)
        else:
            self.add_dynamic_node(None, node)
        return node

    def draw_debug_pyramid(self, T, base_width=0.05, base_height=0.05, height=0.05, color=(1.0, 1.0, 1.0, 0.5)):
        """
        Draw a debug pyramid representing a camera frustum.
        Parameters
        ----------
        T: array-like, shape (4, 4), optional
            The transformation matrix.
        base_width: float
            The width of the pyramid base.
        base_height: float
            The height of the pyramid base.
        height: float
            The height of the pyramid (distance from apex to base).
        color: RGBA color tuple
        """
        T = tensor_to_array(T, dtype=np.float32)
        right = T[:3, 0]
        up = T[:3, 1]
        forward = -T[:3, 2]

        base_center = forward * height
        half_width = base_width / 2
        half_height = base_height / 2
        vertices = np.array(
            [
                [0, 0, 0],  # apex
                base_center + half_width * right + half_height * up,  # top-right
                base_center - half_width * right + half_height * up,  # top-left
                base_center - half_width * right - half_height * up,  # bottom-left
                base_center + half_width * right - half_height * up,  # bottom-right
            ]
        )
        faces = np.array(
            [
                # Base (2 triangles) - facing away from apex
                [1, 2, 3],
                [1, 3, 4],
                # Sides (4 triangles from apex to base edges)
                [0, 2, 1],  # left side
                [0, 3, 2],  # back side
                [0, 4, 3],  # right side
                [0, 1, 4],  # front side
            ]
        )

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        mesh.visual.face_colors = np.tile(mu.color_f32_to_u8(color), (len(faces), 1))

        node = pyrender.Mesh.from_trimesh(mesh, name=f"debug_pyramid_{gs.UID()}", smooth=False, is_marker=True)
        self.add_external_node(node)
        return node

    def draw_debug_spheres(self, poss, radius=0.01, color=(1.0, 0.0, 0.0, 0.5), persistent=True):
        mesh = mu.create_sphere(radius=radius, color=color)
        poses = gu.trans_to_T(tensor_to_array(poss))
        node = pyrender.Mesh.from_trimesh(
            mesh, name=f"debug_spheres_{gs.UID()}", smooth=True, poses=poses, is_marker=True
        )
        if persistent:
            self.add_external_node(node)
        else:
            self.add_dynamic_node(None, node)
        return node

    def draw_debug_box(self, bounds, color=(1.0, 0.0, 0.0, 1.0), wireframe=True, wireframe_radius=0.002):
        bounds = tensor_to_array(bounds)
        mesh = mu.create_box(bounds=bounds, wireframe=wireframe, wireframe_radius=wireframe_radius, color=color)
        node = pyrender.Mesh.from_trimesh(mesh, name=f"debug_box_{gs.UID()}", is_marker=True)
        self.add_external_node(node)
        return node

    def draw_debug_points(self, poss, colors=(1.0, 0.0, 0.0, 0.5)):
        poss = tensor_to_array(poss)
        colors = tensor_to_array(colors)
        if len(colors.shape) == 1:
            colors = np.tile(colors, [len(poss), 1])
        elif len(colors.shape) == 2:
            assert colors.shape[0] == len(poss)

        node = pyrender.Mesh.from_points(poss, name=f"debug_box_{gs.UID()}", colors=colors, is_marker=True)
        self.add_external_node(node)
        return node

    def update_debug_objects(self, objs, poses):
        for obj, pose in zip(objs, poses):
            if not any(
                obj.name.startswith(prefix)
                for prefix in ("debug_sphere_", "debug_frame_", "debug_mesh_", "debug_arrow_")
            ):
                gs.raise_exception("This method is only supported by individual spheres, frames, meshes, and arrows.")
            # An object is placed again at the instances it was drawn with: an arrow drawn for every environment at
            # once holds one shared instance where a sphere, a frame and a mesh hold one per environment, and the
            # buffers sized on that count - the model buffer here, the sorting centre in jit_render - take it whole.
            n_instances = len(obj.primitives[0].poses)
            pose = tensor_to_array(pose)
            if pose.ndim != 3:
                pose = np.tile(pose[np.newaxis], (n_instances, 1, 1))
            assert len(pose) == n_instances, "Inconsistent batch size."
            obj.primitives[0].poses = pose
            node = self.external_nodes[obj.name]
            self.jit.update_buffer(node, "model", pose.transpose((0, 2, 1)))

    def clear_debug_object(self, obj):
        self.clear_external_node(obj)

    def clear_debug_objects(self):
        self.clear_external_nodes()

    def update(self, force_render: bool = False):
        # Early return if already updated previously
        if not force_render and self._t >= self.scene.sim.cur_step_global:
            return

        # The interactive viewer draws from its own thread whatever is uploaded here
        with self.scene._visualizer.viewer_lock:
            # Update current time right away
            self._t = self.scene.sim.cur_step_global

            # Remove up old dynamic nodes
            self.clear_dynamic_nodes(only_outdated=True)

            # Force updating rendering-only quantities that are not updated automatically during simulation
            self.visualizer.update_visual_states(force_render)

            # Reset scene bounds to trigger recomputation. They are involved in shadow map.
            self._scene._bounds = None

            self.update_link_frame()
            self.update_tool()
            self.update_rigid()
            self.update_contact()
            self.update_mpm()
            self.update_sph()
            self.update_pbd()
            self.update_fem()
            self.update_sensors()

            # Update camera fructum
            for camera in self.visualizer.cameras:
                self.update_camera_frustum(camera)

    def add_light(self, light):
        # light direction is light pose's -z frame
        if isinstance(light, gs.options.vis.DirectionalLight):
            pose = np.eye(4, dtype=np.float32)
            gu.z_up_to_R(-np.array(light.dir, dtype=np.float32), out=pose[:3, :3])
            self.add_node(pyrender.DirectionalLight(color=light.color, intensity=light.intensity), pose=pose)
        elif isinstance(light, gs.options.vis.PointLight):
            pose = gu.trans_to_T(np.array(light.pos, dtype=np.float32))
            self.add_node(pyrender.PointLight(color=light.color, intensity=light.intensity), pose=pose)
        else:
            gs.raise_exception(f"Unsupported light: {light}")

    def create_node_seg(self, seg_key, seg_node):
        seg_idxc = self.seg_color_map.seg_key_to_idxc(seg_key)
        if seg_node:
            self.seg_node_map[seg_node] = self.seg_idxc_to_idxc_rgb(seg_idxc)

    def remove_node_seg(self, seg_node):
        self.seg_node_map.pop(seg_node, None)

    def seg_idxc_to_idxc_rgb(self, seg_idxc):
        seg_idxc_rgb = np.array(
            [
                (seg_idxc >> 16) & 0xFF,
                (seg_idxc >> 8) & 0xFF,
                seg_idxc & 0xFF,
            ],
            dtype=np.int32,
        )
        return seg_idxc_rgb

    def seg_idxc_rgb_arr_to_idxc_arr(self, seg_idxc_rgb_arr):
        # Combine the RGB components into a single integer
        seg_idxc_rgb_arr = seg_idxc_rgb_arr.astype(np.int64, copy=False)
        return seg_idxc_rgb_arr[..., 0] * (256 * 256) + seg_idxc_rgb_arr[..., 1] * 256 + seg_idxc_rgb_arr[..., 2]

    def colorize_seg_idxc_arr(self, seg_idxc_arr):
        return self.seg_color_map.colorize_seg_idxc_arr(seg_idxc_arr)

    @property
    def cameras(self):
        return self.visualizer.cameras

    @property
    def seg_idxc_map(self):
        return self.seg_color_map.idxc_map

from dataclasses import dataclass, field
from typing import Any, NamedTuple

import numpy as np
import torch

from OpenGL.GL import GL_HALF_FLOAT, GL_TEXTURE_2D_ARRAY, GL_UNSIGNED_BYTE
import trimesh

import genesis as gs
import genesis.utils.geom as gu
import genesis.utils.mesh as mu
from genesis.constants import TEXTURE_MAP_TYPE
from genesis.engine.render_context import (
    Arrows,
    DynamicMesh,
    Frames,
    Lines,
    Points,
    ScopedRenderContext,
    SegmentationKeys,
    Shape,
    StaticMeshBatch,
    TextureVariants,
)
from genesis.engine.state_change import StateChange, Subscriber, Tracker
from genesis.ext import pyrender
from genesis.ext.pyrender.jit_render import JITRenderer
from genesis.typing import IndexType
from genesis.utils.misc import BuildableMixin, tensor_to_array


# Magnitude beyond which a transform entry marks a pose that blew up, which the drawn node then stops following
POSE_MAX_ABS = 1e20

# The levels of 'VisOptions.segmentation_level', in the order of the keys of SegmentationKeys
SEGMENTATION_LEVELS = ("entity", "link", "geom")


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
        self.generate_seg_colors()
        return self.idxc_to_color[seg_idxc_arr]

    def generate_seg_colors(self):
        """Generate the colors of the keys registered since the latest generation, keeping the colors generated before.

        The keys of each generation take evenly spaced hues shuffled among them, the background staying black, so that
        the colors of the keys registered first do not depend on the keys registered after them.
        """
        # seg_key: same as entity/link/geom's idx
        # seg_idxc: segmentation index of objects
        # seg_idxc_rgb: colorized seg_idxc internally used by renderer
        n_colored = 0 if self.idxc_to_color is None else len(self.idxc_to_color)
        num_keys = len(self.key_map) - n_colored
        if num_keys == 0:
            return

        # Evenly spaced hues
        hues = np.linspace(0.0, 1.0, num_keys, endpoint=False)
        rng = np.random.default_rng(seed=self.seed + n_colored)
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
        # The first color of the first generation is the black background
        for k in range(int(n_colored == 0), num_keys):
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
            rgb = torch.from_numpy(rgb).to(device=gs.device)
            self.idxc_to_color = rgb if self.idxc_to_color is None else torch.cat((self.idxc_to_color, rgb))
        else:
            self.idxc_to_color = rgb if self.idxc_to_color is None else np.concatenate((self.idxc_to_color, rgb))


class TextureBinding(NamedTuple):
    """The texture array a dynamic texture of a mesh is drawn from, and the primitives drawing the mesh."""

    texture: TextureVariants
    texture_array: "pyrender.Texture"
    """The array holding the image of each variant, then that of the batched variant in each rendered environment."""
    primitives: list["pyrender.Primitive"]
    tracker: Tracker
    """The writes and resets of the texture since its images and layers were last drawn (see 'TextureVariants')."""


@dataclass(eq=False)
class RowsSelection:
    """The rows of the shapes of some environments holding a rendered environment, and the position of that
    environment among the rendered ones, which every read of these shapes gathers."""

    rows: "np.ndarray | slice"
    envs_pos: np.ndarray
    # Whether the rows are the rendered environments in order
    is_all_rendered: bool
    # Whether the shapes hold a single row drawn once where it stands (see 'Shape.is_shared')
    is_shared: bool = False


@dataclass(eq=False)
class ShapeNodes:
    """The nodes drawing a shape of the render context, and what updating them reads (see 'RasterizerContext')."""

    nodes: list["pyrender.Node | None"]
    # Whether the shape is a marker, which the debug views alone draw and which takes no part in segmentation
    is_marker: bool
    # The rows of the shape the nodes draw, shared by the shapes of the same environments
    selection: RowsSelection
    # The refreshes of the arrays of the shape and the changes of the reads holding its parts, since the nodes were last
    # synced (see 'RenderContext.track').
    refreshes_tracker: Tracker
    reads_tracker: Tracker
    # Whether its scoped context refreshed the shape since its nodes were last updated, while no view drew it
    is_refreshed: bool = False
    # The meshes of a StaticMeshBatch posed per rendered environment: their positions in the batch, their pyrender
    # meshes and primitives, and the positions among them of the floors.
    meshes_posed: np.ndarray | None = None
    meshes: list["pyrender.Mesh"] | None = None
    primitives: list["pyrender.Primitive"] | None = None
    floors_idx: list[int] | None = None
    # The tetrahedra drawing the particles of a Points shape drawn as such
    tets_mesh: trimesh.Trimesh | None = None
    # The rows of a DynamicMesh shape drawn, a change of which makes its nodes anew
    rows_shown: np.ndarray | None = None
    # The views drawing each node (see 'pyrender.ViewFlags')
    nodes_views: np.ndarray | None = None
    # The texture arrays the dynamic textures of the meshes of the shape are drawn from
    textures: list[TextureBinding] = field(default_factory=list)


class RasterizerContext(BuildableMixin):
    def __init__(self, options):
        self.show_world_frame = options.show_world_frame
        self.world_frame_size = options.world_frame_size
        self.show_cameras = options.show_cameras
        self.shadow = options.shadow
        self.plane_reflection = options.plane_reflection
        self.ambient_light = options.ambient_light
        self.background_color = options.background_color
        self.segmentation_level = options.segmentation_level
        self.lights = options.lights
        self.particle_size_scale = options.particle_size_scale
        self.render_particle_as = options.render_particle_as
        self.rendered_envs_idx = list(options.rendered_envs_idx) if options.rendered_envs_idx is not None else None
        self.split_envs = options.split_envs
        # The step of the latest sync (see 'update')
        self._t = -1

        # nodes
        self.world_frame_node = None
        self.frustum_nodes = dict()  # nodes camera frustums
        # The nodes drawing each shape of the render context, and the rows of the shapes of each selection of
        # environments (see '_sync_shapes').
        self._shapes_nodes: dict[Shape, ShapeNodes] = {}
        self._rows_envs: dict[tuple[int, ...], RowsSelection] = {}
        # Whether each view drawn from the context is a debug view (see 'add_view')
        self._views: dict[Any, bool] = {}
        # The nodes drawing the meshes and particles of each part of the scene, by its segmentation keys, and the keys
        # of each such node.
        self.nodes_by_seg_keys: dict[SegmentationKeys, list["pyrender.Node"]] = {}
        self._nodes_seg_keys: dict["pyrender.Node", SegmentationKeys] = {}
        self.seg_node_map = dict()
        self.seg_color_map = SegmentationColorMap()

        self.world_frame_shown = False
        self.camera_frustum_shown = False
        self.world_frame_mesh = mu.create_frame(
            origin_radius=0.012,
            axis_radius=0.005,
            axis_length=self.world_frame_size,
            head_radius=0.01,
            head_length=0.03,
        )

    @gs.assert_unbuilt
    def build(self, scene):
        self.scene = scene
        self.sim = scene.sim
        self.visualizer = scene.visualizer

        # Update visuals at this point avoids nasty visual artifacts during Scene build
        self.visualizer.update_cameras()

        if self.rendered_envs_idx is None:
            self.rendered_envs_idx = list(range(max(self.scene.n_envs, 1)))
        # The position of each rendered environment among them
        self._rendered_envs_pos = {i_b: i_r for i_r, i_b in enumerate(self.rendered_envs_idx)}
        # The rendered environments on the device, which select the images of the dynamic textures they show
        self._rendered_envs_t = torch.tensor(self.rendered_envs_idx, dtype=gs.tc_int, device=gs.device)

        # pyrender scene
        self._scene = pyrender.Scene(
            ambient_light=self.ambient_light, bg_color=self.background_color, n_envs=len(self.rendered_envs_idx)
        )

        self.jit = JITRenderer(self._scene, [], [], self.scene.envs_offset[self.rendered_envs_idx])

        self.on_lights()

        if self.show_world_frame:
            self.on_world_frame()
        if self.show_cameras:
            self.on_camera_frustum()

        # A change of the render context outside a step, such as a write of the appearance of an entity, shows at the
        # next sync, which a sync between two steps skips otherwise (see 'update').
        def on_change(change: StateChange, **indices: IndexType | None) -> None:
            self._t = -1

        self.sim.render_context.subscribe(Subscriber(to=frozenset(StateChange), callback=on_change))

        # The nodes are made at the first update, once the views drawn from the context are known (see 'add_view')
        self._is_built = True

    def destroy(self):
        self._is_built = False
        for shape in tuple(self._shapes_nodes):
            self._remove_shape_nodes(shape)
        for node in self.frustum_nodes.values():
            self.remove_node(node)
        self.frustum_nodes.clear()
        self.nodes_by_seg_keys.clear()
        self._nodes_seg_keys.clear()
        self._views.clear()
        self._rows_envs.clear()

    def reset(self):
        """Sync the nodes at the next update, whatever the step it is called at."""
        self._t = -1

    def add_view(self, view: Any, is_debug: bool) -> None:
        """Record a view drawn from the context, the camera or the viewer, and whether it is a debug view.

        A debug view draws the markers and the scene as the debug read of the render context holds it, the other views
        the scene as it looks (see 'RenderContext.get_shapes'). The nodes of the reads and of the markers some view asks
        for exist from the next sync on, each drawn by the views it is meant for (see 'pyrender.ViewFlags').
        """
        self._views[view] = is_debug
        self.reset()

    def remove_view(self, view: Any) -> None:
        """Forget a view recorded by 'add_view'."""
        self._views.pop(view, None)

    def add_node(self, obj, **kwargs):
        with self.scene.visualizer.viewer_lock:
            return self._scene.add(obj, **kwargs)

    def remove_node(self, node):
        if self.scene.visualizer is None:
            self._scene.remove_node(node)
        else:
            with self.scene.visualizer.viewer_lock:
                self._scene.remove_node(node)

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

    def _sync_shapes(self) -> None:
        """Sync the nodes with the shapes and the markers of the render context in the rendered environments (see
        'RenderContext.get_shapes').

        The read of the scene as it looks is made when some view draws it, and the debug read and the markers when some
        debug view draws them. Without view, the scene as it looks is drawn. A shape returned again keeps its nodes,
        updated once its scoped context refreshed it or changed the reads holding its parts, a new one gets nodes of its
        own, and the nodes of a shape the context no longer returns are removed. The nodes of a shape no view draws,
        such as a debug representation the debug state hides, are made and updated once some view draws it, and kept
        as it is hidden again.
        """
        context = self.sim.render_context
        views_is_debug = tuple(self._views.values()) or (False,)
        has_normal_view, has_debug_view = not all(views_is_debug), any(views_is_debug)
        shapes: dict[Shape, None] = {}
        markers: tuple[Shape, ...] = ()
        if has_normal_view:
            shapes.update(dict.fromkeys(context.get_shapes(self.rendered_envs_idx, debug=False)))
        if has_debug_view:
            shapes.update(dict.fromkeys(context.get_shapes(self.rendered_envs_idx, debug=True)))
            markers = context.get_markers(self.rendered_envs_idx)
            shapes.update(dict.fromkeys(markers))
        for shape in tuple(self._shapes_nodes):
            if shape not in shapes:
                self._remove_shape_nodes(shape)
        markers = set(markers)
        arrays_host: dict[tuple, np.ndarray] = {}
        for shape in shapes:
            reads = shape.reads
            is_drawn = reads is None or has_normal_view and reads.normal.any() or has_debug_view and reads.debug.any()
            shape_nodes = self._shapes_nodes.get(shape)
            if shape_nodes is None:
                if not is_drawn:
                    continue
                # The segmentation keys of the debug read alone take colors of their own, after those of the scene as it
                # looks, whose colors then stay those of a scene without debug view (see 'generate_seg_colors').
                if shape.reads is not None and not shape.reads.normal.any():
                    self.seg_color_map.generate_seg_colors()
                # The shapes of the same environments share their rows, which the reads of a sync are keyed on
                selection = self._rows_envs.get(None if shape.is_shared else shape.envs)
                if selection is None and shape.is_shared:
                    selection = self._rows_envs[None] = RowsSelection(
                        slice(None), np.zeros(1, dtype=gs.np_int), is_all_rendered=True, is_shared=True
                    )
                elif selection is None:
                    envs_pos = np.array([self._rendered_envs_pos.get(i_b, -1) for i_b in shape.envs], dtype=gs.np_int)
                    rows = np.flatnonzero(envs_pos >= 0)
                    envs_pos = envs_pos[rows]
                    if len(rows) == len(shape.envs):
                        rows = slice(None)
                    is_all_rendered = np.array_equal(envs_pos, np.arange(len(self.rendered_envs_idx)))
                    selection = self._rows_envs[shape.envs] = RowsSelection(rows, envs_pos, is_all_rendered)
                shape_nodes = ShapeNodes(
                    nodes=[],
                    is_marker=shape in markers,
                    selection=selection,
                    refreshes_tracker=context.track(shape, StateChange.GEOMETRY),
                    reads_tracker=context.track(shape, StateChange.VISUAL),
                )
                # The nodes are made from the shape as it stands
                shape_nodes.refreshes_tracker.pull()
                shape_nodes.reads_tracker.pull()
                self._sync_shape_nodes(shape, shape_nodes, arrays_host)
                self._shapes_nodes[shape] = shape_nodes
                continue
            # The nodes of a shape no view draws are updated once some view draws it
            if shape_nodes.refreshes_tracker.pull() is not None:
                shape_nodes.is_refreshed = True
            if shape_nodes.is_refreshed and is_drawn:
                self._sync_shape_nodes(shape, shape_nodes, arrays_host)
                shape_nodes.is_refreshed = False
            if shape_nodes.reads_tracker.pull() is not None:
                self._set_nodes_views(shape, shape_nodes)
            for binding in shape_nodes.textures:
                self._update_texture(binding)

    def _remove_nodes(self, shape_nodes: ShapeNodes) -> None:
        for node in shape_nodes.nodes:
            if node is not None:
                self.remove_node_seg(node)
                self.remove_node(node)
                seg_nodes = self.nodes_by_seg_keys.get(self._nodes_seg_keys.pop(node, None))
                if seg_nodes is not None:
                    seg_nodes.remove(node)
        shape_nodes.nodes.clear()
        shape_nodes.textures.clear()

    def _remove_shape_nodes(self, shape: Shape) -> None:
        shape_nodes = self._shapes_nodes.pop(shape)
        self._remove_nodes(shape_nodes)

    def _add_node_seg(self, mesh: "pyrender.Mesh", seg_keys: SegmentationKeys | None) -> "pyrender.Node":
        node = self.add_node(mesh)
        if seg_keys is not None:
            self.nodes_by_seg_keys.setdefault(seg_keys, []).append(node)
            self._nodes_seg_keys[node] = seg_keys
            self.create_node_seg(seg_keys[SEGMENTATION_LEVELS.index(self.segmentation_level)], node)
        return node

    @staticmethod
    def _read(
        tensor: torch.Tensor, shape_nodes: ShapeNodes, arrays_host: dict[tuple, np.ndarray], dtype=None
    ) -> np.ndarray:
        # The rows of the rendered environments of a tensor on the host, read once per sync whatever the shapes viewing
        # it, by the memory they view.
        selection = shape_nodes.selection
        key = (tensor.data_ptr(), tensor.shape, tensor.stride(), selection)
        array = arrays_host.get(key)
        if array is None:
            array = arrays_host[key] = tensor_to_array(tensor[selection.rows], dtype=dtype)
        return array

    def _pad_envs(self, shape_nodes: ShapeNodes, values_T: np.ndarray) -> np.ndarray:
        # The transforms of every rendered environment, from those of the rows of a shape, the identity standing for
        # the environments the shape leaves out.
        if shape_nodes.selection.is_all_rendered:
            return values_T
        n_rendered = len(self.rendered_envs_idx)
        values_T_padded = np.tile(np.eye(4, dtype=values_T.dtype), (n_rendered, *values_T.shape[1:-2], 1, 1))
        values_T_padded[shape_nodes.selection.envs_pos] = values_T
        return values_T_padded

    def _get_envs_mask(self, shape_nodes: ShapeNodes) -> np.ndarray | None:
        # A shared shape is drawn in every view
        if shape_nodes.selection.is_shared:
            return None
        envs_mask = np.zeros(len(self.rendered_envs_idx), dtype=np.bool_)
        envs_mask[shape_nodes.selection.envs_pos] = True
        return envs_mask

    def _sync_shape_nodes(self, shape: Shape, shape_nodes: ShapeNodes, arrays_host: dict[tuple, np.ndarray]) -> None:
        """Make the nodes drawing a shape when it has none or when what they draw changed, and update them otherwise.

        Each kind of shape reads its arrays once, then makes nodes from them or writes them into the buffers of the
        existing nodes, None standing for a part no rendered environment holds.
        """
        nodes = shape_nodes.nodes
        is_marker = shape_nodes.is_marker
        is_new = not nodes
        selection = shape_nodes.selection
        # The rendered environment each row of a shape drawn one node per row stands in, None for a shared shape
        nodes_env = [None] if selection.is_shared else selection.envs_pos.tolist()
        # The nodes are made drawn by no view, the views drawing each being set once they all exist, see
        # '_set_nodes_views'.
        views = 0
        if isinstance(shape, StaticMeshBatch):
            # The transforms of the meshes in each rendered environment, a marker hidden in an environment being placed
            # out of sight there, so that its nodes keep their instances.
            poses_T = gu.trans_quat_to_T(
                self._read(shape.pos, shape_nodes, arrays_host), self._read(shape.quat, shape_nodes, arrays_host)
            )
            if shape.scale is not None:
                poses_T[..., :3, :3] = poses_T[..., :3, :3] @ self._read(shape.scale, shape_nodes, arrays_host)
            if is_marker and shape.is_active is not None:
                is_active = self._read(shape.is_active, shape_nodes, arrays_host, dtype=np.bool_)
                poses_T[..., :3, 3] = np.where(is_active[..., None], poses_T[..., :3, 3], gu.nowhere())
            poses_T = self._pad_envs(shape_nodes, poses_T)
            if is_new:
                # One node per mesh, holding one instance per rendered environment holding it. A single pyrender
                # material is reused per genesis surface, and per trimesh material among the textured meshes (see
                # 'surface_uvs_to_trimesh_visual'), so that meshes sharing either one (e.g. the many textured submeshes
                # of a GLB, kept separate to preserve baked convex decompositions) expose the same Texture instances, of
                # which the renderer keeps a single host copy and GPU upload. A surface keys its textured meshes and its
                # untextured ones (without uvs) apart, since each kind is drawn with a material of its own. A horizontal
                # plane wider than the environment spacing is one shared floor: one instance where the plane stands, in
                # place of one copy per environment fighting over the same depth once laid out on the grid, which lies
                # under every environment in a pass drawing one environment as well.
                # The pyrender material of each surface and of each trimesh material made so far, found by identity
                vis_materials: list[tuple[tuple, "pyrender.Material"]] = []
                envs_mask = self._get_envs_mask(shape_nodes)
                is_active = None
                # A marker hides some of its meshes in some environments by their transforms alone, set above
                if shape.is_active is not None and not is_marker:
                    is_active = np.zeros((len(self.rendered_envs_idx), len(shape)), dtype=np.bool_)
                    is_active[shape_nodes.selection.envs_pos] = self._read(
                        shape.is_active, shape_nodes, arrays_host, dtype=np.bool_
                    )
                for i_m, geometry in enumerate(shape.geometries):
                    envs = envs_mask if is_active is None else is_active[:, i_m]
                    if envs is not None and not envs.any():
                        nodes.append(None)
                        continue
                    pose_T = poses_T[:, i_m]
                    floor = geometry.floor
                    if floor is not None and envs is not None and envs.all():
                        if (
                            abs(floor.normal[0]) < gs.EPS
                            and abs(floor.normal[1]) < gs.EPS
                            and self.scene.env_spacing[0] < floor.size[0]
                            and self.scene.env_spacing[1] < floor.size[1]
                        ):
                            pose_T = pose_T[:1]
                            envs = None
                    verts = tensor_to_array(geometry.verts)
                    mesh = trimesh.Trimesh(
                        verts,
                        tensor_to_array(geometry.faces),
                        vertex_normals=tensor_to_array(geometry.normals),
                        process=False,
                    )
                    if geometry.visual is None:
                        mesh.visual = mu.surface_uvs_to_trimesh_visual(
                            geometry.surface, uvs=geometry.uvs, n_verts=len(verts)
                        )
                    else:
                        mesh.visual = geometry.visual
                    # A marker is drawn with the colors its visual states alone, and a mesh showing a dynamic texture
                    # binds the texture arrays it is drawn from to a material of its own.
                    material_keys = []
                    if geometry.surface is not None and not geometry.textures:
                        is_textured = isinstance(mesh.visual, trimesh.visual.TextureVisuals)
                        material_keys.append((geometry.surface, is_textured))
                        if is_textured:
                            material_keys.append((mesh.visual.material,))
                    keys_material = [
                        next(
                            (
                                material
                                for key_made, material in vis_materials
                                if key_made[0] is key[0] and key_made == key
                            ),
                            None,
                        )
                        for key in material_keys
                    ]
                    materials = [material for material in keys_material if material is not None]
                    mesh_node = pyrender.Mesh.from_trimesh(
                        mesh=mesh,
                        poses=pose_T,
                        smooth=geometry.is_smooth if geometry.surface is None else geometry.surface.smooth,
                        double_sided=geometry.surface is not None and geometry.surface.double_sided,
                        is_floor=floor is not None,
                        envs=envs,
                        material=materials[0] if materials else None,
                        is_marker=is_marker,
                        views=0,
                    )
                    vis_materials += [
                        (key, mesh_node.primitives[0].material)
                        for key, material in zip(material_keys, keys_material)
                        if material is None
                    ]
                    nodes.append(self._add_node_seg(mesh_node, seg_keys=None if is_marker else geometry.seg_keys))
                    if geometry.textures:
                        self._bind_textures(shape_nodes, geometry.textures, mesh_node.primitives[:1])
                    if floor is not None:
                        self.set_reflection_mat(pose_T)
                n_instances = len(poses_T)
                meshes_posed = [
                    i_m
                    for i_m, node in enumerate(nodes)
                    if node is not None and len(node.mesh.primitives[0].poses) == n_instances
                ]
                shape_nodes.meshes_posed = np.array(meshes_posed, dtype=gs.np_int)
                shape_nodes.meshes = [nodes[i_m].mesh for i_m in meshes_posed]
                shape_nodes.primitives = [nodes[i_m].mesh.primitives[0] for i_m in meshes_posed]
                shape_nodes.floors_idx = [
                    i for i, i_m in enumerate(meshes_posed) if shape.geometries[i_m].floor is not None
                ]
            else:
                meshes, primitives = shape_nodes.meshes, shape_nodes.primitives
                if not meshes:
                    return
                # One transposed copy of the poses, which the buffer flush uploads as is: each node then takes a view
                poses_T = poses_T.swapaxes(0, 1)
                if len(meshes) < len(shape):
                    poses_T = poses_T[shape_nodes.meshes_posed]
                # A part whose pose blew up keeps the transform it was last drawn with, rather than vanishing and
                # dragging the scene bounds, and the shadow map fitted to them, along.
                is_pose_valid = (np.abs(poses_T) < POSE_MAX_ABS).all(axis=(-2, -1))
                if not is_pose_valid.all():
                    for primitive, pose_T, is_valid in zip(primitives, poses_T, is_pose_valid):
                        pose_T[~is_valid] = primitive.poses[~is_valid]
                # Each primitive takes a view of the poses laid out contiguously
                poses_T = np.ascontiguousarray(poses_T, dtype=np.float32)
                for mesh, primitive, pose_T in zip(meshes, primitives, poses_T):
                    mesh._bounds = None
                    primitive.poses = pose_T
                self.jit.update_buffer(primitives, "model", np.ascontiguousarray(poses_T.transpose((0, 1, 3, 2))))
                for i in shape_nodes.floors_idx:
                    self.set_reflection_mat(poses_T[i])
        elif isinstance(shape, DynamicMesh):
            if shape.verts_idx is None:
                verts = self._read(shape.verts, shape_nodes, arrays_host)
            else:
                # The vertices meshes share are gathered on the device, which reads them faster than the host does,
                # those of every drawn mesh viewing the same array at once with one read of the device (see
                # 'DynamicMesh.verts_idx').
                key = ("gathered", shape)
                verts = arrays_host.get(key)
                if verts is None:
                    meshes = [
                        mesh
                        for mesh, mesh_nodes in self._shapes_nodes.items()
                        if isinstance(mesh, DynamicMesh)
                        and mesh.verts_idx is not None
                        and mesh.verts is shape.verts
                        and mesh_nodes.selection is selection
                    ]
                    if shape not in meshes:
                        meshes.append(shape)
                    meshes_verts_idx = torch.cat([mesh.verts_idx for mesh in meshes])
                    meshes_verts = tensor_to_array(torch.index_select(shape.verts[selection.rows], 1, meshes_verts_idx))
                    vert_start = 0
                    for mesh in meshes:
                        vert_end = vert_start + len(mesh.verts_idx)
                        arrays_host[("gathered", mesh)] = meshes_verts[:, vert_start:vert_end]
                        vert_start = vert_end
                    verts = arrays_host[key]
            rows_shown = np.ones(len(verts), dtype=np.bool_)
            if shape.is_active is not None:
                rows_shown = self._read(shape.is_active, shape_nodes, arrays_host, dtype=np.bool_)
            # A row shown or hidden makes the nodes anew
            if not is_new and (rows_shown != shape_nodes.rows_shown).any():
                self._remove_nodes(shape_nodes)
                is_new = True
            if is_new:
                # A copy, the host read of a tensor on the host viewing it in place
                shape_nodes.rows_shown = rows_shown.copy()
                faces = tensor_to_array(shape.faces)
                colors = None if shape.colors is None else self._read(shape.colors, shape_nodes, arrays_host)
                for i_r, (env_i, vverts) in enumerate(zip(nodes_env, verts)):
                    if not rows_shown[i_r]:
                        nodes.append(None)
                        continue
                    if colors is None:
                        visual = mu.surface_uvs_to_trimesh_visual(shape.surface, uvs=shape.uvs, n_verts=len(vverts))
                    else:
                        visual = trimesh.visual.ColorVisuals(vertex_colors=mu.color_f32_to_u8(colors[i_r]))
                    mesh_node = pyrender.Mesh.from_trimesh(
                        trimesh.Trimesh(vverts, faces, visual=visual, process=False),
                        smooth=shape.surface is not None and shape.surface.smooth,
                        double_sided=shape.surface is not None and shape.surface.double_sided,
                        envs=env_i,
                        is_marker=is_marker,
                        views=views,
                    )
                    nodes.append(self._add_node_seg(mesh_node, seg_keys=None if is_marker else shape.seg_keys))
                if shape.textures:
                    primitives = [node.mesh.primitives[0] for node in nodes if node is not None]
                    self._bind_textures(shape_nodes, shape.textures, primitives)
            else:
                for node, vverts in zip(nodes, verts):
                    if node is not None:
                        self._update_vertices(node, vverts)
        elif isinstance(shape, Points):
            # An inactive point is placed out of sight, so that the nodes keep one instance per point
            poss = self._read(shape.pos, shape_nodes, arrays_host)
            if shape.is_active is not None:
                # The flags are integers on Metal (see 'gs.qd_bool'), which masks read as indices
                is_active = self._read(shape.is_active, shape_nodes, arrays_host, dtype=np.bool_)
                poss = np.where(is_active[..., None], poss, gu.nowhere())
            # Points drawn as pixels hold their positions in their vertices, which only new nodes take
            if not is_new and shape.radius == 0.0:
                self._remove_nodes(shape_nodes)
                is_new = True
            if is_new:
                if poss.shape[1] == 0:
                    return
                # The particles of an entity are drawn at the scale of the visualization options
                radius = shape.radius if shape.seg_keys is None else shape.radius * self.particle_size_scale
                colors = None if not isinstance(shape.color, torch.Tensor) else tensor_to_array(shape.color)
                if radius == 0.0:
                    sphere = None
                elif shape.surface is None:
                    sphere = mu.create_sphere(radius=radius, color=shape.color)
                else:
                    sphere = mu.create_sphere(radius, subdivisions=1)
                    sphere.visual = mu.surface_uvs_to_trimesh_visual(shape.surface, n_verts=len(sphere.vertices))
                is_tets = not is_marker and self.render_particle_as == "tet"
                if is_tets:
                    shape_nodes.tets_mesh = mu.create_tets_mesh(poss.shape[1], radius)
                    shape_nodes.tets_mesh.visual = mu.surface_uvs_to_trimesh_visual(
                        shape.surface, n_verts=len(shape_nodes.tets_mesh.vertices)
                    )
                    vels = self._read(shape.vel, shape_nodes, arrays_host)
                for i_r, (env_i, pos) in enumerate(zip(nodes_env, poss)):
                    if sphere is None:
                        mesh_node = pyrender.Mesh.from_points(
                            pos, colors=colors, is_marker=is_marker, envs=env_i, views=views
                        )
                    elif is_tets:
                        tets_mesh = shape_nodes.tets_mesh.copy()
                        tets_mesh.vertices = mu.transform_tets_mesh_verts(
                            tets_mesh.vertices, positions=pos, zs=vels[i_r]
                        )
                        mesh_node = pyrender.Mesh.from_trimesh(tets_mesh, smooth=False, envs=env_i, views=views)
                    else:
                        mesh_node = pyrender.Mesh.from_trimesh(
                            sphere, smooth=True, poses=gu.trans_to_T(pos), envs=env_i, is_marker=is_marker, views=views
                        )
                    nodes.append(self._add_node_seg(mesh_node, seg_keys=None if is_marker else shape.seg_keys))
            elif shape_nodes.tets_mesh is not None:
                vels = self._read(shape.vel, shape_nodes, arrays_host)
                for node, pos, vel in zip(nodes, poss, vels):
                    tets_verts = mu.transform_tets_mesh_verts(shape_nodes.tets_mesh.vertices, positions=pos, zs=vel)
                    self._update_vertices(node, tets_verts)
            else:
                for node, pos in zip(nodes, poss):
                    self.jit.update_buffer(node.mesh.primitives, "model", (gu.trans_to_T(pos).transpose((0, 2, 1)),))
        elif isinstance(shape, (Frames, Arrows, Lines)):
            if isinstance(shape, Frames):
                # The transforms of the instances of each node drawing the frames: one node per frame holding one
                # instance per rendered environment when they are drawn apart, a single node holding every frame of
                # every environment placed at its offset otherwise. They scale the frame of unit length, so that a frame
                # resized keeps its nodes.
                nodes_T = gu.trans_quat_to_T(
                    self._read(shape.pos, shape_nodes, arrays_host), self._read(shape.quat, shape_nodes, arrays_host)
                )
                nodes_T[..., :3, :3] *= shape.length
            else:
                # The transform placing the unit mesh of each element in each rendered environment: the unit mesh
                # stretched along its z axis by the length of the element and across by its radius gives the mesh of
                # the element, an element of zero length collapsing to a point.
                if isinstance(shape, Arrows):
                    starts = self._read(shape.pos, shape_nodes, arrays_host)
                    vecs = self._read(shape.vec, shape_nodes, arrays_host)
                else:
                    starts = self._read(shape.start, shape_nodes, arrays_host)
                    vecs = self._read(shape.end, shape_nodes, arrays_host) - starts
                radius = shape.radius
                if isinstance(radius, torch.Tensor):
                    radius = tensor_to_array(x=radius[selection.rows] if radius.ndim == 2 else radius)
                R = gu.z_up_to_R(vecs.astype(np.float32))
                R[..., :, :2] *= np.broadcast_to(radius, vecs.shape[:2])[..., None, None]
                R[..., :, 2] *= np.linalg.norm(vecs, axis=-1)[..., None]
                nodes_T = gu.trans_R_to_T(starts.astype(np.float32), R)
            # An inactive element is placed out of sight, so that its node keeps its instances
            if shape.is_active is not None:
                is_active = self._read(shape.is_active, shape_nodes, arrays_host, dtype=np.bool_)
                nodes_T[..., :3, 3] = np.where(is_active[..., None], nodes_T[..., :3, 3], gu.nowhere())
            if isinstance(shape, Frames) and not (self.split_envs or selection.is_shared):
                envs_idx = np.array(self.rendered_envs_idx)[selection.envs_pos]
                nodes_T[..., :3, 3] += self.scene.envs_offset[envs_idx, np.newaxis]
                nodes_T = nodes_T.reshape((1, -1, 4, 4))
            else:
                nodes_T = self._pad_envs(shape_nodes, nodes_T).swapaxes(0, 1)
            if is_new and isinstance(shape, Frames):
                mesh = trimesh.creation.axis(origin_size=0.03, axis_radius=0.025, axis_length=1.0)
                mesh.visual.face_colors[:, :3] = (0.7 * mesh.visual.face_colors[:, :3]).astype(int)
                envs_mask = self._get_envs_mask(shape_nodes) if self.split_envs else None
                for frame_T in nodes_T:
                    nodes.append(
                        self.add_node(pyrender.Mesh.from_trimesh(mesh, poses=frame_T, is_marker=True, envs=envs_mask))
                    )
            elif is_new:
                # One node per element, holding one instance per rendered environment of the unit mesh it stretches
                colors = None if isinstance(shape.color, tuple) else tensor_to_array(shape.color).tolist()
                envs_mask = self._get_envs_mask(shape_nodes)
                for i_e, element_T in enumerate(nodes_T):
                    color = shape.color if colors is None else tuple(colors[i_e])
                    if isinstance(shape, Arrows):
                        mesh = mu.create_arrow(length=1.0, radius=1.0, body_color=color, head_color=color)
                    else:
                        mesh = mu.create_cylinder(radius=1.0, height=1.0, sections=12, color=color)
                        mesh.apply_translation((0.0, 0.0, 0.5))
                    mesh_node = pyrender.Mesh.from_trimesh(mesh, poses=element_T, is_marker=True, envs=envs_mask)
                    nodes.append(self.add_node(mesh_node))
            else:
                for node, node_T in zip(nodes, nodes_T):
                    node.mesh.primitives[0].poses = node_T
                    self.jit.update_buffer(
                        node.mesh.primitives, "model", (node.mesh.primitives[0].poses.transpose((0, 2, 1)),)
                    )
        if is_new:
            shape_nodes.nodes_views = None
            self._set_nodes_views(shape, shape_nodes)

    def _bind_textures(
        self, shape_nodes: ShapeNodes, textures: tuple[TextureVariants, ...], primitives: list["pyrender.Primitive"]
    ) -> None:
        for texture in textures:
            # A dynamic texture is the base color of the material when the surface draws it as its color, and its
            # emissive texture otherwise (see 'surface_uvs_to_trimesh_visual').
            images = texture.variants
            if texture.batched is not None:
                images = torch.cat((images, texture.batched[self._rendered_envs_t]))
            # The arrays hold the images in the format of their map, which the shaders read from the type of the array
            texture_array = pyrender.Texture(
                source=tensor_to_array(images),
                source_channels="RGBA" if texture.is_base_color else "RGB",
                tex_type=GL_TEXTURE_2D_ARRAY,
                data_format=GL_UNSIGNED_BYTE if texture.map_type == TEXTURE_MAP_TYPE.DIFFUSE else GL_HALF_FLOAT,
            )
            for primitive in primitives:
                if texture.is_base_color:
                    primitive.material.baseColorTexture = texture_array
                else:
                    primitive.material.emissiveTexture = texture_array
                if primitive.texture_layers is None:
                    primitive.texture_layers = np.zeros((len(self.rendered_envs_idx), 2), dtype=np.float32)
            binding = TextureBinding(texture, texture_array, primitives, texture.changelog.track(StateChange.VISUAL))
            shape_nodes.textures.append(binding)
            self._update_texture(binding)

    def _update_texture(self, binding: TextureBinding) -> None:
        """Upload the batched images a write or a reset shows in the rendered environments, and set their layers.

        A texture just bound counts as written in every environment.
        """
        record = binding.tracker.pull()
        if record is None:
            return
        texture = binding.texture
        n_variants = len(texture.variants)
        variants_idx = texture.variants_idx[self._rendered_envs_t]
        is_batched = variants_idx < 0
        is_written = is_batched if record.envs is None else is_batched & record.envs[self._rendered_envs_t]
        if is_written.any():
            layers_idx = n_variants + torch.nonzero(is_written)[:, 0]
            images = texture.batched[self._rendered_envs_t[is_written]]
            binding.texture_array.set_layers(tensor_to_array(layers_idx), tensor_to_array(images))
            self._scene.updated_textures.add(binding.texture_array)
        envs_pos = torch.arange(len(variants_idx), device=gs.device)
        envs_layer = tensor_to_array(torch.where(is_batched, n_variants + envs_pos, variants_idx))
        column = 0 if texture.is_base_color else 1
        for primitive in binding.primitives:
            # A primitive standing in one environment samples its layer whichever row a pass reads
            env_idx = primitive.env_idx
            primitive.texture_layers[:, column] = envs_layer if env_idx is None else envs_layer[env_idx]
            self.jit.update_texture_layers(primitive)

    def _set_nodes_views(self, shape: Shape, shape_nodes: ShapeNodes) -> None:
        # The debug views draw the markers and the parts of the shapes the debug read holds, the other views the parts
        # the read of the scene as it looks holds, each node of a StaticMeshBatch drawing one of its meshes and the
        # nodes of another shape drawing it whole (see 'Shape.reads').
        n_nodes = len(shape_nodes.nodes)
        if shape_nodes.is_marker:
            views = np.full(n_nodes, pyrender.ViewFlags.DEBUG)
        elif shape.reads is None:
            views = np.full(n_nodes, pyrender.ViewFlags.NORMAL | pyrender.ViewFlags.DEBUG)
        else:
            reads = shape.reads
            views = reads.normal * pyrender.ViewFlags.NORMAL | reads.debug * pyrender.ViewFlags.DEBUG
            views = np.broadcast_to(views, n_nodes)
        nodes_changed = (
            range(len(views)) if shape_nodes.nodes_views is None else np.flatnonzero(views != shape_nodes.nodes_views)
        )
        for i_n in nodes_changed:
            node = shape_nodes.nodes[i_n]
            if node is not None:
                node.mesh.views = int(views[i_n])
                self._scene._meshes_updated = True
                # What the views draw changed, which the shadow maps follow (see 'pyrender.Scene.revision')
                self._scene.bump_revision()
        shape_nodes.nodes_views = views

    def _update_vertices(self, node: "pyrender.Node", vverts: np.ndarray) -> None:
        update_data = self._scene.reorder_vertices(node, vverts.astype(np.float32, copy=False))
        self.jit.update_buffer(node.mesh.primitives, "pos", (update_data,))
        normal_data = self.jit.update_normal(node, update_data)
        if normal_data is not None:
            self.jit.update_buffer(node.mesh.primitives, "normal", (normal_data,))

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

    def on_lights(self):
        for light in self.lights:
            self.add_light(light)

    def update(self, force_render: bool = False):
        """
        Sync the nodes with the state of the scene.

        The nodes of a shape are updated once the render context refreshed it, after a change of state of its solver.
        """
        # Early return if already updated previously
        if not force_render and self._t >= self.scene.sim.cur_step_global:
            return

        # The interactive viewer draws from its own thread whatever is uploaded here
        with self.scene.visualizer.viewer_lock:
            # Update current time right away
            self._t = self.scene.sim.cur_step_global

            # Force updating rendering-only quantities that are not updated automatically during simulation
            self.visualizer.update_cameras(force_render)

            # Reset scene bounds to trigger recomputation. They are involved in shadow map.
            self._scene._bounds = None

            self._sync_shapes()

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

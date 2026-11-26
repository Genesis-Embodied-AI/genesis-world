import math

import numpy as np
import torch

import genesis as gs
from genesis.constants import IMAGE_TYPE
from genesis.engine.render_context import DynamicMesh
from genesis.repr_base import RBC
from genesis.utils.misc import BuildableMixin, tensor_to_array

from .camera import Camera
from .rasterizer_context import SEGMENTATION_LEVELS, SegmentationColorMap

# Optional imports for platform-specific functionality
try:
    from gs_madrona.renderer_gs import MadronaBatchRendererAdapter

    _MADRONA_AVAILABLE = True
except ImportError:
    MadronaBatchRendererAdapter = None
    _MADRONA_AVAILABLE = False


def _transform_camera_quat(quat):
    # quat for Madrona needs to be transformed to y-forward
    w, x, y, z = torch.unbind(quat, dim=-1)
    return torch.stack([x + w, x - w, y - z, y + z], dim=-1) / math.sqrt(2.0)


def _make_tensor(data, *, dtype: torch.dtype = torch.float32):
    return torch.tensor(data, dtype=dtype, device=gs.device)


class GenesisGeomRetriever(BuildableMixin):
    def __init__(self, rigid_solver, seg_level):
        self.rigid_solver = rigid_solver
        self.seg_color_map = SegmentationColorMap(to_torch=True)
        self.seg_level = seg_level
        self.geom_idxc = None

        self.default_geom_group = 2
        self.default_enabled_geom_groups = np.array([self.default_geom_group], dtype=np.int32)

    @gs.assert_unbuilt
    def build(self):
        # The visual geoms of the rigid solver, those of the kinematic entities being left out, are the static mesh
        # batches of its scoped context.
        if any(isinstance(shape, DynamicMesh) for shape in self.rigid_solver.render_context.get_shapes()):
            gs.raise_exception(
                "BatchRenderer draws the visual meshes of the rigid entities posed by their link, and no "
                "'gs.surfaces.DynamicVertices'."
            )
        self.geometries = [
            geometry for batch in self.rigid_solver.render_context.get_shapes() for geometry in batch.geometries
        ]
        if any(geometry.textures for geometry in self.geometries):
            gs.raise_exception("BatchRenderer draws no 'gs.textures.DynamicTexture'.")
        self.n_vgeoms = len(self.geometries)
        level = SEGMENTATION_LEVELS.index(self.seg_level)
        self.geom_idxc = torch.tensor(
            [self.seg_color_map.seg_key_to_idxc(geometry.seg_keys[level]) for geometry in self.geometries],
            dtype=torch.int32,
            device=gs.device,
        )

        self._is_built = True

    # FIXME: Use a kernel to do it efficiently
    @gs.assert_built
    def retrieve_rigid_meshes_static(self):
        args = {}
        meshes_verts = [geometry.verts for geometry in self.geometries]
        meshes_faces = [geometry.faces for geometry in self.geometries]
        mesh_vertex_counts = np.array([len(verts) for verts in meshes_verts], dtype=np.int32)
        mesh_face_counts = np.array([len(faces) for faces in meshes_faces], dtype=np.int32)
        mesh_uvs = []
        mesh_uv_offsets = []
        total_uv_size = 0
        for geometry in self.geometries:
            if geometry.uvs is not None:
                mesh_uvs.append(geometry.uvs.astype(np.float32))
                mesh_uv_offsets.append(total_uv_size)
                total_uv_size += len(geometry.uvs)
            else:
                mesh_uv_offsets.append(-1)

        # The adapter reads 32-bit floats and indices
        args["mesh_vertices"] = tensor_to_array(torch.cat(meshes_verts), dtype=np.float32)
        args["mesh_vertex_offsets"] = np.cumsum(mesh_vertex_counts, dtype=np.int32) - mesh_vertex_counts
        args["mesh_faces"] = tensor_to_array(torch.cat(meshes_faces), dtype=np.int32)
        args["mesh_face_offsets"] = np.cumsum(mesh_face_counts, dtype=np.int32) - mesh_face_counts
        args["mesh_texcoords"] = np.concatenate(mesh_uvs, axis=0) if mesh_uvs else np.empty((0, 2), np.float32)
        args["mesh_texcoord_offsets"] = np.array(mesh_uv_offsets, np.int32)
        args["geom_types"] = np.full((self.n_vgeoms,), 7, dtype=np.int32)  # 7 stands for mesh
        args["geom_groups"] = np.full((self.n_vgeoms,), self.default_geom_group, dtype=np.int32)
        args["geom_data_ids"] = np.arange(self.n_vgeoms, dtype=np.int32)
        args["geom_sizes"] = np.ones((self.n_vgeoms, 3), dtype=np.float32)
        args["enabled_geom_groups"] = self.default_enabled_geom_groups

        # Heterogeneous entities: a mesh may exist in only a subset of envs. Emit an optional [num_worlds, num_geoms]
        # visibility mask (1 = visible, 0 = hidden) so each variant renders only where it is active, omitted for
        # homogeneous scenes.
        batches = self.rigid_solver.render_context.get_shapes()
        if any(batch.is_active is not None for batch in batches):
            # The masks view the render context, which lays the environments out last, and their concatenation keeps
            # that layout, while the adapter reads them row by row.
            args["geom_env_mask"] = np.ascontiguousarray(
                np.concatenate(
                    [
                        np.ones((len(batch.envs), len(batch)), dtype=np.int32)
                        if batch.is_active is None
                        else tensor_to_array(batch.is_active, dtype=np.int32)
                        for batch in batches
                    ],
                    axis=1,
                )
            )

        # Retrieve material data
        geom_mat_ids = []
        num_materials = 0
        materials_indices = {}
        mat_rgbas = []
        mat_texture_indices = []
        mat_texture_offsets = []
        total_mat_textures = 0

        num_textures = 0
        texture_indices = {}
        texture_widths = []
        texture_heights = []
        texture_nchans = []
        texture_offsets = []
        texture_data = []
        total_texture_size = 0

        for geometry in self.geometries:
            geom_surface = geometry.surface
            geom_textures = geom_surface.get_rgba(batch=True).textures

            geom_texture_indices = []
            for geom_texture in geom_textures:
                if isinstance(geom_texture, gs.textures.ImageTexture) and geom_texture.image_array is not None:
                    texture_id = (
                        ("path", geom_texture.image_path)
                        if geom_texture.image_path is not None
                        else ("image_array", id(geom_texture.image_array))
                    )
                    if texture_id not in texture_indices:
                        texture_idx = num_textures
                        texture_indices[texture_id] = texture_idx
                        texture_widths.append(geom_texture.image_array.shape[1])
                        texture_heights.append(geom_texture.image_array.shape[0])
                        assert geom_texture.channel == 4
                        texture_nchans.append(geom_texture.channel)
                        texture_offsets.append(total_texture_size)
                        texture_data.append(geom_texture.image_array.flat)
                        num_textures += 1
                        total_texture_size += geom_texture.image_array.size
                    else:
                        texture_idx = texture_indices[texture_id]
                    geom_texture_indices.append(texture_idx)

            # TODO: support batch rgba
            geom_rgbas = [
                geom_texture.image_color if isinstance(geom_texture, gs.textures.ImageTexture) else geom_texture.color
                for geom_texture in geom_textures
            ]
            for i in range(1, len(geom_rgbas)):
                if not np.allclose(geom_rgbas[0], geom_rgbas[i], atol=gs.EPS):
                    gs.logger.warning("Batch Color is not yet supported. Use the first texture's color instead.")
                    break
            geom_rgba = geom_rgbas[0]

            mat_id = None
            if len(geom_texture_indices) == 0:
                geom_rgba_int = (np.array(geom_rgba) * 255.0).astype(np.uint32)
                mat_id = geom_rgba_int[0] << 24 | geom_rgba_int[1] << 16 | geom_rgba_int[2] << 8 | geom_rgba_int[3]
            if mat_id not in materials_indices:
                material_idx = num_materials
                if mat_id is not None:
                    materials_indices[mat_id] = material_idx
                mat_rgbas.append(geom_rgba)
                mat_texture_indices.extend(geom_texture_indices)
                mat_texture_offsets.append(total_mat_textures)
                num_materials += 1
                total_mat_textures += len(geom_texture_indices)
            else:
                material_idx = materials_indices[mat_id]
            geom_mat_ids.append(material_idx)

        args["geom_mat_ids"] = np.array(geom_mat_ids, np.int32)
        args["tex_widths"] = np.array(texture_widths, np.int32)
        args["tex_heights"] = np.array(texture_heights, np.int32)
        args["tex_nchans"] = np.array(texture_nchans, np.int32)
        args["tex_data"] = np.concatenate(texture_data, axis=0) if texture_data else np.array([], np.uint8)
        args["tex_offsets"] = np.array(texture_offsets, np.int64)
        args["mat_rgba"] = np.array(mat_rgbas, np.float32)
        args["mat_tex_ids"] = np.array(mat_texture_indices, np.int32)
        args["mat_tex_offsets"] = np.array(mat_texture_offsets, np.int32)

        return args

    # FIXME: Use a kernel to do it efficiently
    @gs.assert_built
    def retrieve_rigid_property_torch(self, num_worlds):
        geom_rgb = torch.empty((0, self.n_vgeoms), dtype=torch.uint32, device=gs.device)
        geom_mat_ids = torch.full((num_worlds, self.n_vgeoms), -1, dtype=torch.int32, device=gs.device)
        geom_sizes = torch.ones((self.n_vgeoms, 3), dtype=torch.float32, device=gs.device)
        geom_sizes = geom_sizes[None].repeat(num_worlds, 1, 1)
        return geom_mat_ids, geom_rgb, geom_sizes

    # FIXME: Use a kernel to do it efficiently
    @gs.assert_built
    def retrieve_rigid_state_torch(self):
        batches = self.rigid_solver.render_context.get_shapes()
        geom_pos = torch.cat([batch.pos for batch in batches], dim=1)
        geom_rot = torch.cat([batch.quat for batch in batches], dim=1)
        return geom_pos, geom_rot


class Light:
    def __init__(self, pos, dir, color, intensity, directional, castshadow, cutoff, attenuation):
        self._pos = pos
        self._dir = tuple(dir / np.linalg.norm(dir))
        self._color = color
        self._intensity = intensity
        self._directional = directional
        self._castshadow = castshadow
        self._cutoff = cutoff
        self._attenuation = attenuation

    @property
    def pos(self):
        return self._pos

    @property
    def dir(self):
        return self._dir

    @property
    def color(self):
        return self._color

    @property
    def intensity(self):
        return self._intensity

    @property
    def directional(self):
        return self._directional

    @property
    def castshadow(self):
        return self._castshadow

    @property
    def cutoffRad(self):
        return math.radians(self._cutoff)

    @property
    def cutoffDeg(self):
        return self._cutoff

    @property
    def attenuation(self):
        return self._attenuation


class BatchRenderer(BuildableMixin, RBC):
    """
    This class is used to manage batch rendering
    """

    def __init__(self, visualizer, renderer_options, vis_options):
        self._visualizer = visualizer
        self._lights = gs.List()
        self._use_rasterizer = renderer_options.use_rasterizer
        self._renderer = None
        self._geom_retriever = GenesisGeomRetriever(self._visualizer.scene.rigid_solver, vis_options.segmentation_level)
        self._data_cache = {}
        self._t = -1

    def add_light(self, pos, dir, color, intensity, directional, castshadow, cutoff, attenuation):
        self._lights.append(Light(pos, dir, color, intensity, directional, castshadow, cutoff, attenuation))

    @gs.assert_unbuilt
    def build(self):
        """
        Build all cameras in the batch and initialize Moderona renderer
        """
        if not _MADRONA_AVAILABLE:
            gs.raise_exception("Madrona batch renderer is only supported on Linux x86-64.")

        if gs.backend != gs.cuda:
            gs.raise_exception("BatchRenderer requires CUDA backend.")
        gpu_id = gs.device.index if gs.device.index is not None else 0

        # Extract the complete list of non-debug cameras
        self._cameras = gs.List(
            [camera for camera in self._visualizer._cameras if not isinstance(camera, Camera) or not camera.debug]
        )
        if not self._cameras:
            gs.raise_exception("Please add at least one camera when using BatchRender.")
        # The renderer draws a fisheye as an equidistant projection of its field of view, about the centre of the image
        for camera in self._cameras:
            if isinstance(camera, Camera) and camera.model == "fisheye":
                (fx, _, cx), (_, fy, cy), _ = camera.lens.intrinsics
                width, height = camera.res
                if not np.allclose(
                    (*camera.lens.distortion, fy - fx, cx - 0.5 * width, cy - 0.5 * height), 0.0, atol=gs.EPS
                ):
                    gs.raise_exception(
                        "The BatchRenderer draws a fisheye camera as an equidistant projection about the centre of the "
                        "image, which this calibrated fisheye lens is not. Render it with the Rasterizer."
                    )

        # Build the geometry retriever
        self._geom_retriever.build()

        # Make sure that all cameras have identical resolution
        try:
            ((camera_width, camera_height),) = set(camera.res for camera in self._cameras)
        except ValueError as e:
            gs.raise_exception_from("All cameras must have the exact same resolution when using BatchRender.", e)

        self._renderer = MadronaBatchRendererAdapter(
            geom_retriever=self._geom_retriever,
            gpu_id=gpu_id,
            num_worlds=max(self._visualizer.scene.n_envs, 1),
            num_lights=len(self._lights),
            cam_fovs_tensor=_make_tensor([camera.fov for camera in self._cameras]),
            cam_znears_tensor=_make_tensor([camera.near for camera in self._cameras]),
            cam_zfars_tensor=_make_tensor([camera.far for camera in self.cameras]),
            cam_proj_types_tensor=_make_tensor(
                [camera.model == "fisheye" for camera in self._cameras], dtype=torch.uint32
            ),
            batch_render_view_width=camera_width,
            batch_render_view_height=camera_height,
            add_cam_debug_geo=False,
            use_rasterizer=self._use_rasterizer,
        )
        self._renderer.init(
            cam_pos_tensor=torch.stack([torch.atleast_2d(camera.get_pos()) for camera in self._cameras], dim=1),
            cam_rot_tensor=_transform_camera_quat(
                torch.stack([torch.atleast_2d(camera.get_quat()) for camera in self._cameras], dim=1)
            ),
            lights_pos_tensor=_make_tensor([light.pos for light in self._lights]).reshape((-1, 3)),
            lights_dir_tensor=_make_tensor([light.dir for light in self._lights]).reshape((-1, 3)),
            lights_rgb_tensor=_make_tensor([light.color for light in self._lights]).reshape((-1, 3)),
            lights_directional_tensor=_make_tensor([light.directional for light in self._lights], dtype=torch.bool),
            lights_castshadow_tensor=_make_tensor([light.castshadow for light in self._lights], dtype=torch.bool),
            lights_cutoff_tensor=_make_tensor([light.cutoffRad for light in self._lights]),
            lights_attenuation_tensor=_make_tensor([light.attenuation for light in self._lights]),
            lights_intensity_tensor=_make_tensor([light.intensity for light in self._lights]),
        )
        self._is_built = True

    def update_scene(self):
        self._visualizer._context.update()

    def render(self, rgb=True, depth=False, segmentation=False, normal=False, antialiasing=False):
        """
        Render all cameras in the batch.

        Parameters
        ----------
        rgb : bool, optional
            Whether to render the rgb image.
        depth : bool, optional
            Whether to render the depth image.
        segmentation : bool, optional
            Whether to render the segmentation image.
        normal : bool, optional
            Whether to render the normal image.
        antialiasing : bool, optional
            Whether to apply anti-aliasing.

        Returns
        -------
        rgb_arr : tuple of arrays
            The sequence of rgb images associated with each camera.
        depth_arr : tuple of arrays
            The sequence of depth images associated with each camera.
        segmentation_arr : tuple of arrays
            The sequence of segmentation images associated with each camera.
        normal_arr : tuple of arrays
            The sequence of normal images associated with each camera.
        """

        # The images are rendered again once the scene or one of the cameras changed (see 'reset')
        if self._t < self._visualizer.scene.sim.cur_step_global:
            self._data_cache.clear()

        # Fetch available cached data
        request = (rgb, depth, segmentation, normal)
        cache_key = (antialiasing,)
        cached = [self._data_cache.get((img_type, cache_key), None) for img_type in IMAGE_TYPE]

        # Force disabling rendering whenever cached data is already available
        needed = tuple(req and arr is None for req, arr in zip(request, cached))

        # Early return if everything requested is already cached
        if not any(needed):
            return tuple(arr if req else None for req, arr in zip(request, cached))

        # Update scene
        self.update_scene()

        # Render only what is needed (flags still passed to renderer)
        cameras_pos = torch.stack([torch.atleast_2d(camera.get_pos()) for camera in self._cameras], dim=1)
        cameras_quat = torch.stack([torch.atleast_2d(camera.get_quat()) for camera in self._cameras], dim=1)
        cameras_quat = _transform_camera_quat(cameras_quat)
        render_flags = np.array(
            (
                *(
                    needed[img_type]
                    for img_type in (IMAGE_TYPE.RGB, IMAGE_TYPE.DEPTH, IMAGE_TYPE.NORMAL, IMAGE_TYPE.SEGMENTATION)
                ),
                antialiasing,
            ),
            dtype=np.uint32,
        )
        rendered = list(self._renderer.render(cameras_pos, cameras_quat, render_flags))

        # convert seg geom idx to seg_idxc
        if needed[IMAGE_TYPE.SEGMENTATION]:
            seg_geoms = rendered[IMAGE_TYPE.SEGMENTATION]
            mask = seg_geoms != -1
            seg_geoms[mask] = self._geom_retriever.geom_idxc[seg_geoms[mask]]
            seg_geoms[~mask] = 0

        # Post-processing:
        # * Remove alpha channel from RGBA
        # * Squeeze env and channel dims if necessary
        # * Split along camera dim
        for img_type, data in enumerate(rendered):
            if needed[img_type]:
                data = data.swapaxes(0, 1)
                if self._visualizer.scene.n_envs == 0:
                    data = data.squeeze(1)
                rendered[img_type] = tuple(data[..., :3].squeeze(-1))

        # Convert center distance depth to plane distance
        if not self._use_rasterizer and needed[IMAGE_TYPE.DEPTH]:
            rendered[IMAGE_TYPE.DEPTH] = tuple(
                camera.distance_center_to_plane(depth_data)
                for camera, depth_data in zip(self._cameras, rendered[IMAGE_TYPE.DEPTH])
            )

        # Update cache
        self._t = self._visualizer.scene.sim.cur_step_global
        for img_type, data in enumerate(rendered):
            if needed[img_type]:
                self._data_cache[(img_type, cache_key)] = rendered[img_type]

        # Return in the required order, or None if not requested
        return tuple(self._data_cache[(img_type, cache_key)] if request[img_type] else None for img_type in IMAGE_TYPE)

    def colorize_seg_idxc_arr(self, seg_idxc_arr):
        return self._geom_retriever.seg_color_map.colorize_seg_idxc_arr(seg_idxc_arr)

    def destroy(self):
        self._is_built = False
        self._lights.clear()
        self._data_cache.clear()
        if self._renderer is not None:
            del self._renderer.madrona
            self._renderer = None

    def reset(self):
        self._t = -1

    @property
    def lights(self):
        return self._lights

    @property
    def cameras(self):
        return self._cameras

    @property
    def seg_idxc_map(self):
        return self._geom_retriever.seg_color_map.idxc_map

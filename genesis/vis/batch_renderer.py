import math

import numpy as np
import torch

import genesis as gs
from genesis.repr_base import RBC
from genesis.constants import IMAGE_TYPE
from genesis.utils.misc import qd_to_torch

from .camera import Camera
from .rasterizer_context import SegmentationColorMap

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


class GenesisGeomRetriever:
    def __init__(self, rigid_solver, seg_level):
        self.rigid_solver = rigid_solver
        self.seg_color_map = SegmentationColorMap(to_torch=True)
        self.seg_level = seg_level
        self.geom_idxc = None

        self.default_geom_group = 2
        self.default_enabled_geom_groups = np.array([self.default_geom_group], dtype=np.int32)

    def build(self):
        self.n_vgeoms = self.rigid_solver.n_vgeoms
        self.geom_idxc = []
        vgeoms = self.rigid_solver.vgeoms
        for vgeom in vgeoms:
            seg_key = self.get_seg_key(vgeom)
            seg_idxc = self.seg_color_map.seg_key_to_idxc(seg_key)
            self.geom_idxc.append(seg_idxc)
        self.geom_idxc = torch.tensor(self.geom_idxc, dtype=torch.int32, device=gs.device)
        self.seg_color_map.generate_seg_colors()

    def get_seg_key(self, vgeom):
        if self.seg_level == "geom":
            return (vgeom.entity.idx, vgeom.link.idx, vgeom.idx)
        elif self.seg_level == "link":
            return (vgeom.entity.idx, vgeom.link.idx)
        elif self.seg_level == "entity":
            return vgeom.entity.idx
        else:
            gs.raise_exception(f"Unsupported segmentation level: {self.seg_level}")

    # FIXME: Use a kernel to do it efficiently
    def retrieve_rigid_meshes_static(self):
        args = {}
        vgeoms = self.rigid_solver.vgeoms

        # Retrieve geom data
        mesh_vertices = self.rigid_solver.dyn_info.vverts.init_pos.to_numpy()
        mesh_faces = self.rigid_solver.dyn_info.vfaces.vverts_idx.to_numpy()
        mesh_vertex_offsets = self.rigid_solver.dyn_info.vgeoms.vvert_start.to_numpy()
        mesh_face_starts = self.rigid_solver.dyn_info.vgeoms.vface_start.to_numpy()
        mesh_face_ends = self.rigid_solver.dyn_info.vgeoms.vface_end.to_numpy()
        total_uv_size = 0
        mesh_uvs = []
        mesh_uv_offsets = []
        for i in range(self.n_vgeoms):
            mesh_faces[mesh_face_starts[i] : mesh_face_ends[i]] -= mesh_vertex_offsets[i]

        geom_data_ids = []
        for vgeom in vgeoms:
            seg_key = self.get_seg_key(vgeom)
            seg_id = self.seg_color_map.seg_key_to_idxc(seg_key)
            geom_data_ids.append(seg_id)
            if vgeom.uvs is not None:
                mesh_uvs.append(vgeom.uvs.astype(np.float32))
                mesh_uv_offsets.append(total_uv_size)
                total_uv_size += vgeom.uvs.shape[0]
            else:
                mesh_uv_offsets.append(-1)

        args["mesh_vertices"] = mesh_vertices
        args["mesh_vertex_offsets"] = mesh_vertex_offsets
        args["mesh_faces"] = mesh_faces
        args["mesh_face_offsets"] = mesh_face_starts
        args["mesh_texcoords"] = np.concatenate(mesh_uvs, axis=0) if mesh_uvs else np.empty((0, 2), np.float32)
        args["mesh_texcoord_offsets"] = np.array(mesh_uv_offsets, np.int32)
        args["geom_types"] = np.full((self.n_vgeoms,), 7, dtype=np.int32)  # 7 stands for mesh
        args["geom_groups"] = np.full((self.n_vgeoms,), self.default_geom_group, dtype=np.int32)
        args["geom_data_ids"] = np.arange(self.n_vgeoms, dtype=np.int32)
        args["geom_sizes"] = np.ones((self.n_vgeoms, 3), dtype=np.float32)
        args["enabled_geom_groups"] = self.default_enabled_geom_groups

        # Heterogeneous entities: a vgeom variant may exist in only a subset of envs
        # (vgeom.active_envs_mask). Emit an optional [num_worlds, num_geoms] visibility
        # mask (1 = visible, 0 = hidden) so each variant renders only where it is active;
        # omitted for homogeneous scenes (legacy + back-compat with older Madrona).
        if any(vgeom.active_envs_mask is not None for vgeom in vgeoms):
            num_worlds = max(self.rigid_solver.n_envs, 1)
            geom_env_mask = np.ones((num_worlds, self.n_vgeoms), dtype=np.int32)
            for vgeom in vgeoms:
                if vgeom.active_envs_mask is not None:
                    geom_env_mask[:, vgeom.idx] = vgeom.active_envs_mask.detach().cpu().numpy()
            args["geom_env_mask"] = geom_env_mask

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

        for vgeom in vgeoms:
            geom_surface = vgeom.surface
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
    def retrieve_rigid_property_torch(self, num_worlds):
        geom_rgb = torch.empty((0, self.n_vgeoms), dtype=torch.uint32, device=gs.device)
        geom_mat_ids = torch.full((num_worlds, self.n_vgeoms), -1, dtype=torch.int32, device=gs.device)
        geom_sizes = torch.ones((self.n_vgeoms, 3), dtype=torch.float32, device=gs.device)
        geom_sizes = geom_sizes[None].repeat(num_worlds, 1, 1)
        return geom_mat_ids, geom_rgb, geom_sizes

    # FIXME: Use a kernel to do it efficiently
    def retrieve_rigid_state_torch(self):
        geom_pos = qd_to_torch(self.rigid_solver.dyn_state.vgeoms.pos)
        geom_rot = qd_to_torch(self.rigid_solver.dyn_state.vgeoms.quat)
        geom_pos = geom_pos.transpose(0, 1).contiguous()
        geom_rot = geom_rot.transpose(0, 1).contiguous()
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


class BatchRenderer(RBC):
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

        # Build the geometry retriever
        self._geom_retriever.build()

        # Make sure that all cameras have identical resolution
        try:
            ((camera_width, camera_height),) = set(camera.res for camera in self._cameras)
        except ValueError as e:
            gs.raise_exception_from("All cameras must have the exact same resolution when using BatchRender.", e)

        self._renderer = MadronaBatchRendererAdapter(
            geom_retriever=self._geom_retriever,
            gpu_id=gs.device.index if gs.device.index is not None else 0,
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

    def update_scene(self, force_render: bool = False):
        self._visualizer._context.update(force_render)

    def render(self, rgb=True, depth=False, segmentation=False, normal=False, antialiasing=False, force_render=False):
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
        force_render : bool, optional
            Whether to force render the scene.

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

        # Clear cache if requested or necessary
        if force_render or self._t < self._visualizer.scene.sim.cur_step_global:
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
        self.update_scene(force_render)

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
        return tuple(self._data_cache[(img_type, cache_key)] if needed[img_type] else None for img_type in IMAGE_TYPE)

    def colorize_seg_idxc_arr(self, seg_idxc_arr):
        return self._geom_retriever.seg_color_map.colorize_seg_idxc_arr(seg_idxc_arr)

    def destroy(self):
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

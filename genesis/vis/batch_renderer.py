import math
from typing import NamedTuple

import numpy as np
import torch

import genesis as gs
from genesis.constants import IMAGE_TYPE, TEXTURE_MAP_TYPE
from genesis.engine.render_context import DynamicMesh, TextureVariants
from genesis.engine.state_change import StateChange, Subscriber, Tracker
from genesis.typing import IndexType
from genesis.repr_base import RBC
from genesis.options.vis import FisheyeLens
from genesis.utils.geom import pixels_ray_cosine
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


class DynamicTextureBinding(NamedTuple):
    """The materials and textures of the renderer a dynamic texture of a visual geom is drawn from."""

    geom_idx: int
    """The index of the visual geom among those the renderer draws."""
    texture: TextureVariants
    variants_mat_idx: torch.Tensor
    """The material of each image of 'TextureVariants.variants', holding that image alone."""
    batched_mat_idx: int
    """The material holding the batched image of each world, the one of a world drawing its own, -1 for a texture
    without a batched variant."""
    batched_textures_idx: np.ndarray | None
    """The texture holding the batched image of each world, None for a texture without a batched variant."""
    tracker: Tracker
    """The writes and resets of the texture since the renderer last drew them (see 'TextureVariants.changelog')."""


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
        for geometry in self.geometries:
            for texture in geometry.textures:
                if texture.map_type != TEXTURE_MAP_TYPE.DIFFUSE or not texture.is_base_color:
                    gs.raise_exception(
                        "BatchRenderer draws a 'gs.textures.DynamicTexture' as the color of a surface, in its diffuse "
                        "map alone."
                    )
        self.n_vgeoms = len(self.geometries)
        # The materials and textures of the dynamic textures, and the material each world draws each visual geom with,
        # -1 for that of its mesh, which the static meshes and the properties set (see 'retrieve_rigid_meshes_static')
        self.dynamic_textures: list[DynamicTextureBinding] = []
        self.geom_mat_ids: torch.Tensor | None = None
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

        self.dynamic_textures.clear()
        n_worlds = max(self.rigid_solver.n_envs, 1)
        for geom_idx, geometry in enumerate(self.geometries):
            # A dynamic texture holds one material per image it switches between, and the batched variant one material
            # holding the image of each world, which the renderer draws in that world. The material of the mesh is
            # that of the first image, each world drawing the material of the image it shows (see
            # 'retrieve_rigid_property_torch').
            if geometry.textures:
                (texture,) = geometry.textures
                images = tensor_to_array(texture.variants)
                if texture.batched is not None:
                    images = np.concatenate((images, tensor_to_array(texture.batched)))
                height, width = images.shape[1:3]
                textures_idx = np.arange(num_textures, num_textures + len(images), dtype=np.int32)
                for image in images:
                    texture_widths.append(width)
                    texture_heights.append(height)
                    texture_nchans.append(4)
                    texture_offsets.append(total_texture_size)
                    texture_data.append(np.ascontiguousarray(image).reshape(-1))
                    total_texture_size += image.size
                num_textures += len(images)
                n_variants = len(texture.variants)
                variants_mat_idx = np.arange(num_materials, num_materials + n_variants, dtype=np.int32)
                for texture_idx in textures_idx[:n_variants]:
                    mat_rgbas.append((1.0, 1.0, 1.0, 1.0))
                    mat_texture_indices.append(texture_idx)
                    mat_texture_offsets.append(total_mat_textures)
                    total_mat_textures += 1
                num_materials += n_variants
                batched_mat_idx, batched_textures_idx = -1, None
                if texture.batched is not None:
                    batched_textures_idx = textures_idx[n_variants:]
                    assert len(batched_textures_idx) == n_worlds
                    batched_mat_idx = num_materials
                    mat_rgbas.append((1.0, 1.0, 1.0, 1.0))
                    mat_texture_indices.extend(batched_textures_idx)
                    mat_texture_offsets.append(total_mat_textures)
                    total_mat_textures += len(batched_textures_idx)
                    num_materials += 1
                geom_mat_ids.append(int(variants_mat_idx[0]))
                self.dynamic_textures.append(
                    DynamicTextureBinding(
                        geom_idx,
                        texture,
                        torch.as_tensor(variants_mat_idx, dtype=torch.int32, device=gs.device),
                        batched_mat_idx,
                        batched_textures_idx,
                        texture.changelog.track(StateChange.VISUAL),
                    )
                )
                continue

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
        self.geom_mat_ids = torch.full((num_worlds, self.n_vgeoms), -1, dtype=torch.int32, device=gs.device)
        for binding in self.dynamic_textures:
            self.geom_mat_ids[:, binding.geom_idx] = self._get_dynamic_texture_mat_ids(binding)
        geom_sizes = torch.ones((self.n_vgeoms, 3), dtype=torch.float32, device=gs.device)
        geom_sizes = geom_sizes[None].repeat(num_worlds, 1, 1)
        return self.geom_mat_ids, geom_rgb, geom_sizes

    @staticmethod
    def _get_dynamic_texture_mat_ids(binding: DynamicTextureBinding) -> torch.Tensor:
        # The material of the image each world shows, the batched one where its index is -1
        variants_idx = binding.texture.variants_idx
        mat_ids = binding.variants_mat_idx[variants_idx.clamp(min=0)]
        if binding.batched_textures_idx is not None:
            mat_ids = torch.where(variants_idx < 0, binding.batched_mat_idx, mat_ids)
        return mat_ids

    @gs.assert_built
    def update_dynamic_textures(self, renderer: "MadronaBatchRendererAdapter") -> None:
        """Draw the images the dynamic textures show since they were last drawn, writes and resets alike.

        A world showing its batched image takes the pixels written there, and every world the material of the image it
        shows.
        """
        is_mat_changed = False
        for binding in self.dynamic_textures:
            record = binding.tracker.pull()
            if record is None:
                continue
            texture = binding.texture
            self.geom_mat_ids[:, binding.geom_idx] = self._get_dynamic_texture_mat_ids(binding)
            is_mat_changed = True
            if binding.batched_textures_idx is not None:
                is_written = texture.variants_idx < 0
                if record.envs is not None:
                    is_written &= record.envs
                envs_idx = torch.nonzero(is_written).reshape(-1)
                if envs_idx.numel() > 0:
                    renderer.update_textures(
                        binding.batched_textures_idx[tensor_to_array(envs_idx)], texture.batched[envs_idx]
                    )
        if is_mat_changed:
            renderer.set_geom_mat_ids(self.geom_mat_ids)

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
        self._subscriber = None

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
        if any(geometry.textures for geometry in self._geom_retriever.geometries) and not hasattr(
            MadronaBatchRendererAdapter, "set_geom_mat_ids"
        ):
            gs.raise_exception(
                "BatchRenderer draws a 'gs.textures.DynamicTexture' with a version of gs-madrona switching the "
                "materials and writing the textures at runtime. Please upgrade gs-madrona."
            )

        # Make sure that all cameras have identical resolution
        try:
            ((camera_width, camera_height),) = set(camera.res for camera in self._cameras)
        except ValueError as e:
            gs.raise_exception_from("All cameras must have the exact same resolution when using BatchRender.", e)

        # The cosine of the angle between the ray of each pixel of each camera and its line of sight, which converts the
        # depth between the distance along the ray and the depth along the line of sight (see 'render'). The cameras
        # and the camera sensors alike state their projection by their field of view.
        self._cameras_ray_cosine = []
        for camera in self._cameras:
            if camera.model == "fisheye":
                (fx, _, cx), (_, fy, cy), _ = FisheyeLens.equidistant(camera.res, camera.fov).intrinsics
                rays_cosine = pixels_ray_cosine(camera.res, (fx, fy, cx, cy), (0.0, 0.0, 0.0, 0.0))
            else:
                f = 0.5 * camera_height / math.tan(math.radians(0.5 * camera.fov))
                rays_cosine = pixels_ray_cosine(camera.res, (f, f, 0.5 * camera_width, 0.5 * camera_height))
            self._cameras_ray_cosine.append(torch.as_tensor(rays_cosine, dtype=gs.tc_float, device=gs.device))

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

        # A change of the render context outside a step, such as a write of a dynamic texture, shows at the next
        # render, which the images rendered since the last step skip otherwise.
        def on_change(change: StateChange, **indices: IndexType | None) -> None:
            self._t = -1

        self._subscriber = Subscriber(to=frozenset((StateChange.VISUAL,)), callback=on_change)
        self._visualizer.scene.sim.render_context.subscribe(self._subscriber)
        self._is_built = True

    def update_scene(self):
        self._visualizer._context.update()
        self._geom_retriever.update_dynamic_textures(self._renderer)

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

        # The depth of a pinhole camera is along its line of sight, and that of a fisheye camera along the ray of each
        # pixel. The ray tracer gives the distance along the ray, while the rasterizer draws a fisheye from a pinhole
        # image, which gives the depth along the line of sight within the disk the fisheye shows and the far clipping
        # distance beyond it.
        if needed[IMAGE_TYPE.DEPTH]:
            depths = []
            for camera, rays_cosine, depth in zip(self._cameras, self._cameras_ray_cosine, rendered[IMAGE_TYPE.DEPTH]):
                if camera.model == "fisheye":
                    if self._use_rasterizer:
                        depth = torch.where(depth < camera.far * (1.0 - 1e-6), depth / rays_cosine, depth)
                elif not self._use_rasterizer:
                    depth = depth * rays_cosine
                depths.append(depth)
            rendered[IMAGE_TYPE.DEPTH] = tuple(depths)

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
        if self._subscriber is not None:
            self._visualizer.scene.sim.render_context.unsubscribe(self._subscriber)
            self._subscriber = None
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

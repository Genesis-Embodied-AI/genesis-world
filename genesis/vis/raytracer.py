import os
import sys

import numpy as np
import trimesh

import genesis as gs
import genesis.utils.geom as gu
import genesis.utils.mesh as mu
import genesis.utils.misc as miscu
import genesis.utils.particle as pu
from genesis.engine.render_context import ParticlesPhysics, StaticMeshBatch
from genesis.utils.misc import BuildableMixin

try:
    sys.path.append(os.path.join(miscu.get_src_dir(), "ext/LuisaRender/build/bin"))
    import LuisaRenderPy
except ImportError as e:
    gs.raise_exception_from("Failed to import LuisaRenderer.", e)

logging_class = {
    "debug": LuisaRenderPy.LogLevel.DEBUG,
    "info": LuisaRenderPy.LogLevel.INFO,
    "warning": LuisaRenderPy.LogLevel.WARNING,
}
rigid_as_deformable = True
sphere_light_as_mesh = True


class EnvironmentSphere:
    """
    Environment is implemented as a huge env sphere around the scene, with an emission map.
    """

    def __init__(self, radius, pos, quat, surface=None, name=None, file_path="meshes/env_sphere/env_sphere.obj"):
        self.radius = radius
        self.pos = pos
        self.quat = quat
        self.surface = surface
        self.name = str(gs.UID()) if name is None else name
        self.file_path = os.path.join(gs.utils.get_assets_dir(), file_path)

    def add_to_render(self, renderer):
        if self.surface is None:
            return
        env_sphere = mu.load_trimesh(self.file_path)
        env_vertices = env_sphere.vertices * self.radius
        env_faces = env_sphere.faces
        env_normals = env_sphere.vertex_normals
        env_uvs = env_sphere.visual.uv
        env_uvs[:, 1] = 1.0 - env_uvs[:, 1]
        env_transform = gu.trans_quat_to_T(np.array(self.pos), np.array(self.quat))

        renderer.add_surface(self.name, self.surface)
        renderer.add_rigid(self.name, env_vertices, env_faces, env_normals, env_uvs)
        renderer.update_rigid(self.name, env_transform)


class ShapeLight:
    def __init__(self, surface, name=None):
        self.surface = surface
        self.name = str(gs.UID()) if name is None else name

    def add_to_render(self, renderer):
        renderer.add_surface(self.name, self.surface)


class SphereLight(ShapeLight):
    def __init__(self, radius, pos, surface, name=None):
        super().__init__(surface, name)
        self.pos = np.array(pos)
        self.radius = radius

    def add_to_render(self, renderer):
        super().add_to_render(renderer)
        if sphere_light_as_mesh:
            trimesh_sphere = trimesh.creation.icosphere(radius=self.radius)
            renderer.add_rigid(
                name=self.name,
                vertices=np.array(trimesh_sphere.vertices),
                triangles=np.array(trimesh_sphere.faces),
                normals=np.array(trimesh_sphere.vertex_normals),
                uvs=np.array([]),
            )
            renderer.update_rigid(self.name, gs.trans_to_T(self.pos))
        else:
            renderer.add_particles(self.name)
            renderer.update_particles(self.name, self.pos, self.radius)


class MeshLight(ShapeLight):
    def __init__(self, mesh, transform, surface, name=None, revert_dir=False):
        super().__init__(surface, name)
        self.mesh = mesh
        self.transform = transform
        self.revert_dir = revert_dir

    def add_to_render(self, renderer):
        super().add_to_render(renderer)
        if self.revert_dir:
            renderer.add_rigid(
                name=self.name,
                vertices=self.mesh.trimesh.vertices,
                triangles=self.mesh.trimesh.faces[:, ::-1],
                normals=-self.mesh.trimesh.vertex_normals,
                uvs=np.array([]),
            )
        else:
            renderer.add_rigid(
                name=self.name,
                vertices=self.mesh.trimesh.vertices,
                triangles=self.mesh.trimesh.faces,
                normals=self.mesh.trimesh.vertex_normals,
                uvs=np.array([]),
            )
        renderer.update_rigid(self.name, self.transform)


def _get_luisa_thin_lens(fov: float, res: tuple[int, int], lens: "gs.options.ThinLens") -> tuple[float, float]:
    """Return the focal length in millimeters and the f-number that LuisaRender takes to render a thin lens.

    LuisaRender frames a thin-lens camera through a full-frame sensor of 36x24 mm behind a lens of the focal length it
    takes, and sizes the radius of the lens as that focal length over twice the f-number. The focal length it takes is
    the one framing the vertical field of view `fov` at the resolution `res`, and the f-number the one giving the
    radius of the lens `lens`, whose focus distance is set.
    """
    if res[0] > res[1]:
        projected_pixel_size = min(0.036 / res[0], 0.024 / res[1])
    else:
        projected_pixel_size = min(0.036 / res[1], 0.024 / res[0])
    image_dist = res[1] * projected_pixel_size / (2.0 * np.tan(np.deg2rad(fov / 2.0)))
    luisa_focal_len = 1.0 / (1.0 / image_dist + 1.0 / lens.focus_dist)
    return 1000.0 * luisa_focal_len, lens.aperture * luisa_focal_len / lens.focal_len


class Raytracer(BuildableMixin):
    def __init__(self, options, vis_options):
        self.logging_level = options.logging_level
        self.state_limit = options.state_limit
        self.tracing_depth = options.tracing_depth
        self.rr_depth = options.rr_depth
        self.rr_threshold = options.rr_threshold
        self.clamp_normal = options.normal_diff_clamp

        self.rendered_envs_idx = vis_options.rendered_envs_idx

        self._scene = None
        self._shapes = dict()
        self._cameras = dict()
        self.camera_updated = False

        # shape properties
        self.shape_surfaces = dict()
        self.shape_clamp_normals = dict()
        self.shape_surface_only = dict()
        self.shape_foamgens = dict()
        self.shape_geoms = dict()
        # The foam each shape of a rendered environment generates, by the name of the shape (see '_add_foams')
        self._foams_generator: dict[str, pu.FoamGenerator] = {}

        # environment configs
        self.env_sphere = EnvironmentSphere(
            pos=options.env_pos, quat=options.env_quat, radius=options.env_radius, surface=options.env_surface
        )

        # light objects
        self.lights = []
        for light in options.lights:
            light_surface = gs.surfaces.Emission(
                color=tuple(x * light.intensity for x in light.color),
            )
            light_surface.update_texture()
            self.lights.append(SphereLight(radius=light.radius, pos=light.pos, surface=light_surface))

        # backend and device selection: aligning with genesis if possible
        backend = gs.backend.name
        device_index = options.device_index
        if device_index is None:
            # If no device index has been specified, use Torch GPU device ID if any, 0 otherwise
            device_index = 0 if gs.device.type == "cpu" else (gs.device.index or 0)
        if backend == "amdgpu":
            # Luisa does not support HIP for AMD GPU: using DirectX on Windows, falling back to CPU otherwise
            if sys.platform == "win32":
                backend = "dx"
            else:
                backend = "cpu"
                device_index = 0
        elif sys.platform == "darwin":
            backend = "metal"
        self.backend = backend
        self.device_index = device_index

        LuisaRenderPy.init(
            context_path=os.path.dirname(LuisaRenderPy.__file__),
            context_id=str(gs.UID()),
            backend=backend,
            device_index=device_index,
            log_level=logging_class[self.logging_level],
        )

    def add_mesh_light(self, mesh, color, intensity, pos, quat, revert_dir=False, double_sided=False, cutoff=180.0):
        color = np.array(color)
        if color.ndim != 1 or (color.shape[0] != 3 and color.shape[0] != 4):
            gs.raise_exception("Light color should have shape (3,) or (4,).")

        self.lights.append(
            MeshLight(
                mesh=mesh,
                transform=gu.trans_quat_to_T(np.array(pos), np.array(quat)),
                surface=gs.surfaces.Plastic(
                    color=(color[0], color[1], color[2], color[3] if color.shape[0] == 4 else 1.0),
                    emissive=(
                        color[0] * intensity,
                        color[1] * intensity,
                        color[2] * intensity,
                    ),
                    double_sided=double_sided,
                    cutoff=cutoff,
                ),
                name=str(mesh.uid),
                revert_dir=revert_dir,
            )
        )

    @gs.assert_unbuilt
    def build(self, scene):
        # Note that surfaces and entities added after building may not be included in LuisaRender
        self.scene = scene
        self.sim = scene.sim
        self.visualizer = scene.visualizer
        if self.rendered_envs_idx is None:
            self.rendered_envs_idx = list(range(max(self.scene.n_envs, 1)))

        self._scene = LuisaRenderPy.create_scene()
        self._scene.init(
            LuisaRenderPy.Render(
                name=str(scene.uid),
                spectrum=LuisaRenderPy.SRGBSpectrum(),
                integrator=LuisaRenderPy.WavePathIntegrator(
                    log_level=logging_class[self.logging_level],
                    enable_cache=True,
                    max_depth=self.tracing_depth,
                    rr_depth=self.rr_depth,
                    rr_threshold=self.rr_threshold,
                ),
                clamp_normal=self.clamp_normal,
            )
        )

        self.visualizer.update_cameras()

        self.env_sphere.add_to_render(self)
        for light in self.lights:
            light.add_to_render(self)

        # The meshes posed by their solver are added here, the dynamic meshes as they are first drawn, see
        # 'update_scene'.
        for shape in self.sim.render_context.get_shapes(
            self.rendered_envs_idx, debug=False, reconstruct_surfaces=False
        ):
            if isinstance(shape, StaticMeshBatch):
                for geometry in shape.geometries:
                    name = str(geometry.seg_keys.geom)
                    self.add_surface(name, geometry.surface)
                    self.add_rigid_batch(
                        name=name,
                        vertices=miscu.tensor_to_array(geometry.verts),
                        triangles=miscu.tensor_to_array(geometry.faces),
                        normals=miscu.tensor_to_array(geometry.normals),
                        uvs=np.array([]) if geometry.uvs is None else geometry.uvs,
                    )

        self._is_built = True

    def get_transform(self, matrix):
        if matrix is None:
            return None
        assert matrix.shape == (4, 4)
        return LuisaRenderPy.MatrixTransform(np.ascontiguousarray(matrix))

    def get_texture(self, texture):
        if texture is None:
            return None
        if isinstance(texture, gs.textures.DynamicTexture):
            gs.raise_exception("Raytracer draws no 'gs.textures.DynamicTexture'.")

        if isinstance(texture, gs.textures.ColorTexture):
            return LuisaRenderPy.ColorTexture(color=texture.color)
        elif isinstance(texture, gs.textures.ImageTexture):
            image_path = texture.image_path
            image_array = texture.image_array

            if image_array is not None:
                assert image_array.dtype == np.uint8
                return LuisaRenderPy.ImageTexture(
                    image_data=np.ascontiguousarray(image_array).tobytes(),
                    width=image_array.shape[1],
                    height=image_array.shape[0],
                    channel=texture.channel,
                    scale=texture.image_color,
                    encoding=texture.encoding,
                )
            elif image_path is not None:
                return LuisaRenderPy.ImageTexture(file=image_path, scale=texture.image_color, encoding=texture.encoding)
            else:
                gs.raise_exception("Texture type error. Both 'image_path' and 'image_array' are None")
        else:
            gs.raise_exception(f"Texture type error: {type(texture)}")

    def add_surface(self, shape_name, surface):
        """Add the surface of a shape."""
        # add emission
        if surface.emission is not None:
            emission_luisa = LuisaRenderPy.Light(
                name=f"emis_{shape_name}",
                emission=self.get_texture(surface.emission),
                two_sided=False if surface.double_sided is None else surface.double_sided,
                beam_angle=surface.cutoff,
            )
            self._scene.update_emission(emission_luisa)
        else:
            emission_luisa = None

        # add surface
        subsurface_luisa = None
        surface_name = f"surf_{shape_name}"
        if isinstance(surface, gs.surfaces.Plastic):
            surface_luisa = LuisaRenderPy.PlasticSurface(
                name=surface_name,
                roughness=self.get_texture(surface.roughness_texture),
                opacity=self.get_texture(surface.opacity_texture),
                normal_map=self.get_texture(surface.normal_texture),
                kd=self.get_texture(surface.diffuse_texture),
                ks=self.get_texture(surface.specular_texture),
                eta=self.get_texture(gs.textures.ColorTexture(color=(surface.ior,))),
            )
        elif isinstance(surface, gs.surfaces.BSDF):
            surface_luisa = LuisaRenderPy.DisneySurface(
                name=surface_name,
                roughness=self.get_texture(surface.roughness_texture),
                opacity=self.get_texture(surface.opacity_texture),
                normal_map=self.get_texture(surface.normal_texture),
                kd=self.get_texture(surface.diffuse_texture),
                eta=self.get_texture(gs.textures.ColorTexture(color=(surface.ior,))),
                metallic=self.get_texture(surface.metallic_texture),
                specular_trans=self.get_texture(gs.textures.ColorTexture(color=(surface.specular_trans,))),
                diffuse_trans=(
                    None
                    if surface.diffuse_trans == 0.0
                    else self.get_texture(gs.textures.ColorTexture(color=(surface.diffuse_trans,)))
                ),
            )
        elif isinstance(surface, gs.surfaces.Metal):
            surface_luisa = LuisaRenderPy.MetalSurface(
                name=surface_name,
                roughness=self.get_texture(surface.roughness_texture),
                opacity=self.get_texture(surface.opacity_texture),
                normal_map=self.get_texture(surface.normal_texture),
                kd=self.get_texture(surface.diffuse_texture),
                eta=surface.metal_type,
            )
        elif isinstance(surface, gs.surfaces.Glass):
            if surface.subsurface:
                subsurface_luisa = LuisaRenderPy.UniformSubsurface(
                    name=f"subsurf_{shape_name}",
                    thickness=self.get_texture(surface.thickness_texture),
                )
                self._scene.update_subsurface(subsurface_luisa)
            surface_luisa = LuisaRenderPy.GlassSurface(
                name=surface_name,
                roughness=self.get_texture(surface.roughness_texture),
                normal_map=self.get_texture(surface.normal_texture),
                ks=self.get_texture(surface.specular_texture),
                kt=self.get_texture(surface.transmission_texture),
                eta=self.get_texture(gs.textures.ColorTexture(color=(surface.ior,))),
            )
        elif isinstance(surface, gs.surfaces.Emission):
            surface_luisa = None
        else:
            gs.raise_exception(f"Suface type error: {type(surface)}")

        # if surface.subsurface is not None:
        #     self.add_surface(f"{shape_name}_sub", surface.subsurface)
        #     surface_luisa = LuisaRenderPy.LayeredSurface(
        #         name      = f"{surface_name}_layer",
        #         top       = surface_luisa,
        #         bottom    = self.shape_surfaces[f"{shape_name}_sub"][0],
        #         thickness = self.get_texture(surface.thickness_texture),
        #     )

        if surface_luisa is not None:
            self._scene.update_surface(surface_luisa)

        self.shape_surfaces[shape_name] = [surface_luisa, emission_luisa, subsurface_luisa]

        # add clamp normal
        self.shape_clamp_normals[shape_name] = (
            min(self.clamp_normal, surface.normal_diff_clamp) if surface.smooth else 0.0
        )

        # generate foams
        if surface.generate_foam:
            self.shape_foamgens[shape_name] = {
                "radius_scale": surface.foam_options.radius_scale,
                "spray_decay": surface.foam_options.spray_decay,
                "foam_decay": surface.foam_options.foam_decay,
                "bubble_decay": surface.foam_options.bubble_decay,
                "k_foam": surface.foam_options.k_foam,
            }
            self.add_surface(f"{shape_name}_foams", gs.surfaces.Rough(color=surface.foam_options.color))
        else:
            self.shape_foamgens[shape_name] = None

    def add_rigid(self, name, vertices, triangles, normals, uvs, batch_index=None):
        if rigid_as_deformable:
            self.add_deformable(name, batch_index)
            self.shape_geoms[name] = [vertices, triangles, normals, uvs]
        else:
            shape_name = name if batch_index is None else f"{name}_{batch_index}"
            self._shapes[shape_name] = LuisaRenderPy.RigidShape(
                name=shape_name,
                vertices=np.ascontiguousarray(vertices),
                triangles=np.ascontiguousarray(triangles),
                normals=np.ascontiguousarray(normals),
                uvs=np.ascontiguousarray(uvs),
                surface=self.shape_surfaces[name][0],
                emission=self.shape_surfaces[name][1],
                subsurface=self.shape_surfaces[name][2],
                clamp_normal=self.shape_clamp_normals[name],
            )

    def add_rigid_batch(self, name, vertices, triangles, normals, uvs):
        for batch_index in self.rendered_envs_idx:
            self.add_rigid(name, vertices, triangles, normals, uvs, batch_index)

    def update_rigid(self, name, matrix, batch_index=None):
        if rigid_as_deformable:
            vertices = gu.transform_by_T(self.shape_geoms[name][0], matrix)
            triangles = self.shape_geoms[name][1]
            normals = gu.transform_by_R(self.shape_geoms[name][2], matrix[..., :3, :3])
            uvs = self.shape_geoms[name][3]
            self.update_deformable(name, vertices, triangles, normals, uvs, batch_index)
        else:
            shape_name = name if batch_index is None else f"{name}_{batch_index}"
            self._shapes[shape_name].update(
                transform=self.get_transform(matrix),
            )
            self._scene.update_shape(self._shapes[shape_name])

    def update_rigid_batch(self, name, matrices):
        """Place a rigid shape in every rendered environment from its transforms in each of them, laid out on the
        environment grid as the raytracer draws the environments side by side."""
        for env_i, batch_index in enumerate(self.rendered_envs_idx):
            matrix = matrices[env_i].copy()
            matrix[:3, 3] += self.visualizer.scene.envs_offset[batch_index]
            self.update_rigid(name, matrix, batch_index)

    def add_deformable(self, name, batch_index=None):
        shape_name = name if batch_index is None else f"{name}_{batch_index}"
        self._shapes[shape_name] = LuisaRenderPy.DeformableShape(
            name=shape_name,
            surface=self.shape_surfaces[name][0],
            emission=self.shape_surfaces[name][1],
            subsurface=self.shape_surfaces[name][2],
            clamp_normal=self.shape_clamp_normals[name],
        )

    def update_deformable(self, name, vertices, triangles, normals, uvs, batch_index=None):
        shape_name = name if batch_index is None else f"{name}_{batch_index}"
        self._shapes[shape_name].update(
            vertices=np.ascontiguousarray(vertices),
            triangles=np.ascontiguousarray(triangles),
            normals=np.ascontiguousarray(normals),
            uvs=np.ascontiguousarray(uvs),
        )
        self._scene.update_shape(self._shapes[shape_name])

    def add_particles(self, name, batch_index=None):
        shape_name = name if batch_index is None else f"{name}_{batch_index}"
        self._shapes[shape_name] = LuisaRenderPy.ParticlesShape(
            name=shape_name,
            subdivision=0,
            surface=self.shape_surfaces[name][0],
            emission=self.shape_surfaces[name][1],
            subsurface=self.shape_surfaces[name][2],
        )

    def update_particles(self, name, particles, radius, batch_index=None):
        shape_name = name if batch_index is None else f"{name}_{batch_index}"
        self._shapes[shape_name].update(centers=particles, radii=np.array([radius]))
        self._scene.update_shape(self._shapes[shape_name])

    def _add_foams(self, name: str, batch_index: int, radius: float, physics: ParticlesPhysics | None) -> None:
        """Add the foam the particles of a shape generate in a rendered environment.

        Foam requires a surface asking for it (see 'Surface.generate_foam') and the physical setting of the particles
        (see 'Points.physics').
        """
        foam_options = self.shape_foamgens[name]
        if foam_options is None or physics is None:
            return
        self._foams_generator[f"{name}_{batch_index}"] = pu.FoamGenerator(
            particle_radius=radius,
            time_step=self.scene.dt,
            gravity=np.array(physics.gravity),
            lower_bound=physics.lower_bound,
            upper_bound=physics.upper_bound,
            spray_decay=foam_options["spray_decay"],
            foam_decay=foam_options["foam_decay"],
            bubble_decay=foam_options["bubble_decay"],
            k_foam=foam_options["k_foam"],
            foam_density=physics.density,
        )
        self.add_particles(f"{name}_foams", batch_index)

    def add_camera(self, camera):
        camera_name = str(camera.uid)
        if camera.model == "fisheye":
            gs.raise_exception(
                "The RayTracer renders pinhole cameras alone. Render a fisheye camera with the Rasterizer."
            )

        if isinstance(camera.lens, gs.options.ThinLens):
            luisa_focal_len, luisa_aperture = _get_luisa_thin_lens(camera.fov, camera.res, camera.lens)
            self._cameras[camera_name] = LuisaRenderPy.ThinLensCamera(
                name=camera_name,
                pose=self.get_transform(np.eye(4)),
                film=LuisaRenderPy.Film(resolution=camera.res),
                filter=LuisaRenderPy.Filter(),
                spp=camera.spp,
                aperture=luisa_aperture,
                focal_len=luisa_focal_len,
                focus_dis=camera.lens.focus_dist,
            )
        elif camera.model == "pinhole":
            self._cameras[camera_name] = LuisaRenderPy.PinholeCamera(
                name=camera_name,
                pose=self.get_transform(np.eye(4)),
                film=LuisaRenderPy.Film(resolution=camera.res),
                filter=LuisaRenderPy.Filter(),
                spp=camera.spp,
                fov=camera.fov,
            )
        else:
            gs.raise_exception("Invalid camera model.")

        self.camera_updated = True

    def update_camera(self, camera):
        camera_name = str(camera.uid)
        camera_transform = camera.transform

        if isinstance(camera.lens, gs.options.ThinLens):
            luisa_focal_len, luisa_aperture = _get_luisa_thin_lens(camera.fov, camera.res, camera.lens)
            self._cameras[camera_name].update(
                pose=self.get_transform(camera_transform),
                aperture=luisa_aperture,
                focal_len=luisa_focal_len,
                focus_dis=camera.lens.focus_dist,
            )
        elif camera.model == "pinhole":
            self._cameras[camera_name].update(pose=self.get_transform(camera_transform), fov=camera.fov)
        else:
            gs.raise_exception("Invalid camera model.")

        self._scene.update_camera(self._cameras[camera_name], camera.denoise)
        self.camera_updated = True

    def reset(self):
        self._t = -1

    @gs.assert_built
    def update_scene(self):
        if self._t >= self.scene.sim.cur_step_global:
            if self.camera_updated:
                self._scene.update_scene(time=self._t)
                self.camera_updated = False
            return

        # update t
        self._t = self.scene.sim.cur_step_global

        # update variables not used in simulation
        self.visualizer.update_cameras()

        # Every rendered environment is drawn, laid out on the environment grid
        for shape in self.sim.render_context.get_shapes(self.rendered_envs_idx, debug=False):
            # The shapes hold one row per environment of their own, among which the rendered ones are drawn
            envs_row = {i_b: i_r for i_r, i_b in enumerate(shape.envs)}
            rows_env = [(envs_row[i_b], i_b) for i_b in self.rendered_envs_idx if i_b in envs_row]
            rows = [i_r for i_r, _ in rows_env]
            if isinstance(shape, StaticMeshBatch):
                poses_T = gu.trans_quat_to_T(
                    miscu.tensor_to_array(shape.pos[rows]), miscu.tensor_to_array(shape.quat[rows])
                )
                for i_m, geometry in enumerate(shape.geometries):
                    self.update_rigid_batch(str(geometry.seg_keys.geom), poses_T[:, i_m])
                continue
            # A dynamic mesh is a shape per rendered environment showing it, which the skinned mesh of a body and the
            # surface reconstructed once it tears there share.
            name = str(shape.seg_keys.geom)
            if name not in self.shape_surfaces:
                self.add_surface(name, shape.surface)
            # The vertices a mesh shares are gathered as they are uploaded (see 'DynamicMesh.verts_idx')
            verts = miscu.tensor_to_array(shape.verts[rows])
            if shape.verts_idx is not None:
                verts = verts[:, miscu.tensor_to_array(shape.verts_idx)]
            rows_shown = np.ones(len(rows), dtype=np.bool_)
            if shape.is_active is not None:
                rows_shown = miscu.tensor_to_array(shape.is_active[rows], dtype=np.bool_)
            faces = miscu.tensor_to_array(shape.faces)
            uvs = np.array([]) if shape.uvs is None else shape.uvs
            particles = shape.particles
            for (_, i_b), vverts, is_shown in zip(rows_env, verts, rows_shown):
                if not is_shown:
                    continue
                if f"{name}_{i_b}" not in self._shapes:
                    self.add_deformable(name, i_b)
                    if particles is not None:
                        self._add_foams(name, i_b, particles.radius, particles.physics)
                vverts = vverts + self.scene.envs_offset[i_b]
                normals = trimesh.Trimesh(vertices=vverts, faces=faces, process=False).vertex_normals
                self.update_deformable(name, vverts, faces, normals, uvs, i_b)
                foams_generator = self._foams_generator.get(f"{name}_{i_b}")
                if foams_generator is not None:
                    # The particles hold rows for environments of their own (those of the solver for a skinned mesh)
                    i_p = particles.envs.index(i_b)
                    is_active = miscu.tensor_to_array(particles.is_active[i_p], dtype=np.bool_)
                    foam_particles = foams_generator.generate(
                        miscu.tensor_to_array(particles.pos[i_p])[is_active],
                        miscu.tensor_to_array(particles.vel[i_p])[is_active],
                    )
                    foam_radius = particles.radius * self.shape_foamgens[name]["radius_scale"]
                    foam_particles = foam_particles + self.scene.envs_offset[i_b]
                    self.update_particles(f"{name}_foams", foam_particles, foam_radius, i_b)

        # Flush the update buffer.
        self._scene.update_scene(time=self._t)
        self.camera_updated = False

    @gs.assert_built
    def render_camera(self, camera):
        b = self._scene.render_frame(camera=self._cameras[str(camera.uid)])
        img = np.frombuffer(b, dtype=np.uint8).reshape(camera.res[1], camera.res[0], 4)[:, :, :3]
        return img

    def destroy(self):
        self._shapes = dict()
        self._cameras = dict()
        LuisaRenderPy.destroy()
        self._is_built = False

    @property
    def cameras(self):
        return self.visualizer.cameras

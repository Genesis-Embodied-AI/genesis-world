import os
import sys

import numpy as np
import trimesh

import genesis as gs
import genesis.utils.geom as gu
import genesis.utils.mesh as mu
import genesis.utils.misc as miscu
import genesis.utils.particle as pu
from genesis.engine import entities

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
        self._renderer = renderer
        if self.surface is None:
            return
        env_sphere = trimesh.load(self.file_path)
        env_vertices = env_sphere.vertices * self.radius
        env_faces = env_sphere.faces
        env_normals = env_sphere.vertex_normals
        env_uvs = env_sphere.visual.uv
        env_uvs[:, 1] = 1.0 - env_uvs[:, 1]
        env_transform = gu.trans_quat_to_T(np.array(self.pos), np.array(self.quat))
        # self.verts = env_vertices

        renderer.add_surface(self.name, self.surface)
        renderer.add_rigid(self.name, env_vertices, env_faces, env_normals, env_uvs)
        renderer.update_rigid(self.name, env_transform)

    def update_pose(self, transform):
        self._renderer.update_rigid(self.name, transform)


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
            renderer.add_particles(self.name, None)
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


class Raytracer:
    def __init__(self, options, vis_options):
        self.logging_level = options.logging_level
        self.state_limit = options.state_limit
        self.tracing_depth = options.tracing_depth
        self.rr_depth = options.rr_depth
        self.rr_threshold = options.rr_threshold
        self.clamp_normal = options.normal_diff_clamp

        self.render_particle_as = vis_options.render_particle_as
        self.rendered_envs_idx = vis_options.rendered_envs_idx

        self._scene = None
        self._shapes = dict()
        self._cameras = dict()
        self.camera_updated = False

        # shape properties
        self.shape_surfaces = dict()
        self.shape_clamp_normals = dict()
        self.shape_reconstructs = dict()
        self.shape_surface_only = dict()
        self.shape_foamgens = dict()
        self.shape_geoms = dict()
        self.shape_skin_visuals = dict()

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

    def build(self, scene):
        # Note that surfaces and entities added after building may not be included in LuisaRender
        self.scene = scene
        self.sim = scene.sim
        self.visualizer = scene.visualizer
        if self.rendered_envs_idx is None:
            self.rendered_envs_idx = list(range(self.sim._B))

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

        self.visualizer.update_visual_states()

        # state_limit = 0
        # for camera in self._cameras:
        #     state_limit_cam = int(camera.res[0] * camera.res[1] * self.spp)
        #     state_limit = max(state_limit, state_limit_cam)
        # state_limit = min(state_limit, self.state_limit)

        self.env_sphere.add_to_render(self)
        for light in self.lights:
            light.add_to_render(self)

        for entity in self.sim.entities:
            if isinstance(entity, entities.KinematicEntity):
                for geom in entity.vgeoms:
                    self.add_surface(str(geom.uid), geom.surface)
                if isinstance(entity, entities.RigidEntity):
                    for geom in entity.geoms:
                        self.add_surface(str(geom.uid), geom.surface)
            elif isinstance(entity, entities.FEMEntity):
                for vgeom in entity.vgeoms:
                    self.add_surface(str(vgeom.uid), vgeom.surface)
            else:
                self.add_surface(str(entity.uid), entity.surface)

        # tool entities
        if self.sim.tool_solver.is_active:
            for tool_entity in self.sim.tool_solver.entities:
                self.add_rigid(
                    name=str(tool_entity.uid),
                    vertices=tool_entity.mesh.raw_vertices,
                    triangles=tool_entity.mesh.faces_np,
                    normals=tool_entity.mesh.raw_vertex_normals,
                    uvs=np.array([]),
                )

        # rigid entities
        if self.sim.rigid_solver.is_active:
            for rigid_entity in self.sim.rigid_solver.entities:
                if rigid_entity.surface.vis_mode == "visual":
                    geoms = rigid_entity.vgeoms
                else:
                    geoms = rigid_entity.geoms

                for geom in geoms:
                    if "sdf" in rigid_entity.surface.vis_mode:
                        mesh = geom.get_sdf_trimesh()
                    else:
                        mesh = geom.get_trimesh()
                    self.add_rigid_batch(
                        name=str(geom.uid),
                        vertices=mesh.vertices,
                        triangles=mesh.faces,
                        normals=mesh.vertex_normals,
                        uvs=np.array([]) if geom.uvs is None else geom.uvs,
                    )

        # kinematic entities
        if self.sim.kinematic_solver.is_active:
            for kinematic_entity in self.sim.kinematic_solver.entities:
                assert kinematic_entity.surface.vis_mode == "visual"
                for geom in kinematic_entity.vgeoms:
                    mesh = geom.get_trimesh()
                    self.add_rigid_batch(
                        name=str(geom.uid),
                        vertices=mesh.vertices,
                        triangles=mesh.faces,
                        normals=mesh.vertex_normals,
                        uvs=np.array([]) if geom.uvs is None else geom.uvs,
                    )

        # MPM particles
        if self.sim.mpm_solver.is_active:
            for mpm_entity in self.sim.mpm_solver.entities:
                if mpm_entity.surface.vis_mode == "visual":
                    self.add_deformable(str(mpm_entity.uid))
                else:
                    self.add_particles(
                        str(mpm_entity.uid),
                        self.sim.mpm_solver,
                        self.sim.mpm_solver.particle_radius,
                        mpm_entity.material.rho,
                    )

        # SPH particles
        if self.sim.sph_solver.is_active:
            for sph_entity in self.sim.sph_solver.entities:
                self.add_particles(
                    str(sph_entity.uid),
                    self.sim.sph_solver,
                    self.sim.sph_solver.particle_radius,
                    sph_entity.material.rho,
                )

        # PBD entities
        if self.sim.pbd_solver.is_active:
            for pbd_entity in self.sim.pbd_solver.entities:
                if pbd_entity.surface.vis_mode == "visual":
                    self.add_deformable(str(pbd_entity.uid))
                else:
                    if self.render_particle_as == "sphere":
                        self.add_particles(str(pbd_entity.uid), self.sim.pbd_solver)
                    elif self.render_particle_as == "tet":
                        mesh = mu.create_tets_mesh(pbd_entity.n_particles, self.sim.pbd_solver.particle_radius)
                        pbd_entity._tets_mesh = mesh
                        self.add_deformable(str(pbd_entity.uid))

        # FEM entities
        if self.sim.fem_solver.is_active:
            for fem_entity in self.sim.fem_solver.entities:
                if fem_entity.surface.vis_mode == "visual":
                    for vgeom in fem_entity.vgeoms:
                        self.add_deformable(str(vgeom.uid))

    def get_transform(self, matrix):
        if matrix is None:
            return None
        assert matrix.shape == (4, 4)
        return LuisaRenderPy.MatrixTransform(np.ascontiguousarray(matrix))

    def get_texture(self, texture):
        if texture is None:
            return None

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

        # surface reconstruction
        if surface.vis_mode == "recon":
            self.shape_reconstructs[shape_name] = [surface.recon_backend]
        else:
            self.shape_reconstructs[shape_name] = None

        # skinning visual mesh
        if surface.vis_mode == "visual":
            self.shape_skin_visuals[shape_name] = True
        else:
            self.shape_skin_visuals[shape_name] = False

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

    def add_particles(self, name, solver, radius=None, density=None):
        if self.shape_reconstructs[name] is not None:
            self.add_deformable(name)
        else:
            self._shapes[name] = LuisaRenderPy.ParticlesShape(
                name=name,
                subdivision=0,
                surface=self.shape_surfaces[name][0],
                emission=self.shape_surfaces[name][1],
                subsurface=self.shape_surfaces[name][2],
            )

        if self.shape_foamgens[name] is not None:
            # The gravity of the environment being rendered, taken from the solver these particles belong to, since
            # each holds its own and a solver no entity was given never built one.
            envs_idx = self.rendered_envs_idx[:1] if self.sim.n_envs > 0 else None
            gravity = solver.get_gravity(envs_idx)
            if envs_idx is not None:
                gravity = gravity[0]
            self.shape_foamgens[name]["generator"] = pu.init_foam_generator(
                object_id=name,
                particle_radius=radius,
                time_step=self.scene.dt,
                gravity=miscu.tensor_to_array(gravity),
                lower_bound=self.sim.sph_solver.lower_bound,
                upper_bound=self.sim.sph_solver.upper_bound,
                spray_decay=self.shape_foamgens[name]["spray_decay"],
                foam_decay=self.shape_foamgens[name]["foam_decay"],
                bubble_decay=self.shape_foamgens[name]["bubble_decay"],
                k_foam=self.shape_foamgens[name]["k_foam"],
                foam_density=density,  # use fluid density for foam
            )
            self.shape_foamgens[name]["radius"] = radius * self.shape_foamgens[name]["radius_scale"]
            self.add_particles(f"{name}_foams", solver)

    def update_particles(self, name, particles, radius=None, particles_vel=None, particles_radii=None):
        if self.shape_reconstructs[name] is not None:
            mesh = pu.particles_to_mesh(positions=particles, radius=radius, backend=self.shape_reconstructs[name][0])
            self.update_deformable(
                name,
                mesh.vertices,
                mesh.faces,
                mesh.vertex_normals,
                np.array([]),
            )
        else:
            radii = np.array([radius]) if particles_radii is None else particles_radii
            self._shapes[name].update(centers=particles, radii=radii)
            self._scene.update_shape(self._shapes[name])

        if self.shape_foamgens[name] is not None:
            if particles_vel is None:
                gs.raise_exception("Velocities not passed when generating foams.")
            foam_particles = pu.generate_foam_particles(
                generator=self.shape_foamgens[name]["generator"],
                positions=particles,
                velocities=particles_vel,
            )
            self.update_particles(f"{name}_foams", foam_particles, self.shape_foamgens[name]["radius"])

    def add_camera(self, camera):
        camera_name = str(camera.uid)
        camera_model = camera.model

        if camera_model == "pinhole":
            self._cameras[camera_name] = LuisaRenderPy.PinholeCamera(
                name=camera_name,
                pose=self.get_transform(np.eye(4)),
                film=LuisaRenderPy.Film(resolution=camera.res),
                filter=LuisaRenderPy.Filter(),
                spp=camera.spp,
                fov=camera.fov,
            )
        elif camera_model == "thinlens":
            self._cameras[camera_name] = LuisaRenderPy.ThinLensCamera(
                name=camera_name,
                pose=self.get_transform(np.eye(4)),
                film=LuisaRenderPy.Film(resolution=camera.res),
                filter=LuisaRenderPy.Filter(),
                spp=camera.spp,
                aperture=camera.aperture,
                focal_len=camera.focal_len * 1000,  # measure in mms
                focus_dis=camera.focus_dist,
            )
        else:
            gs.raise_exception("Invalid camera model.")

        self.camera_updated = True

    def update_camera(self, camera):
        camera_name = str(camera.uid)
        camera_model = camera.model
        camera_transform = camera.transform

        if camera_model == "pinhole":
            self._cameras[camera_name].update(pose=self.get_transform(camera_transform), fov=camera.fov)
        elif camera_model == "thinlens":
            self._cameras[camera_name].update(
                pose=self.get_transform(camera_transform),
                aperture=camera.aperture,
                focal_len=camera.focal_len * 1000,
                focus_dis=camera.focus_dist,
            )
        else:
            gs.raise_exception("Invalid camera model.")

        self._scene.update_camera(self._cameras[camera_name], camera.denoise)
        self.camera_updated = True

    def reset(self):
        self._t = -1

    def update_scene(self, force_render: bool = False):
        if not force_render and self._t >= self.scene.sim.cur_step_global:
            if self.camera_updated:
                self._scene.update_scene(time=self._t)
                self.camera_updated = False
            return

        # update t
        self._t = self.scene.sim.cur_step_global

        # update variables not used in simulation
        self.visualizer.update_visual_states(force_render)

        # tool entities
        if self.sim.tool_solver.is_active:
            for tool_entity in self.sim.tool_solver.entities:
                pos = tool_entity.pos[self.sim.cur_substep_local].to_numpy()
                quat = tool_entity.quat[self.sim.cur_substep_local].to_numpy()
                T = gu.trans_quat_to_T(pos, quat)
                self.update_rigid(str(tool_entity.uid), T)

        # rigid-like entities (rigid + kinematic)
        for solver in (self.sim.rigid_solver, self.sim.kinematic_solver):
            if not solver.is_active:
                continue

            # One conversion of the geom poses per visual mode in use
            geoms_T_by_mode = {}
            for entity in solver.entities:
                is_visual = entity.surface.vis_mode == "visual"
                geoms = entity.vgeoms if is_visual else entity.geoms
                if is_visual not in geoms_T_by_mode:
                    geoms_T_by_mode[is_visual] = self.visualizer.context.rigid_geoms_T(solver, is_visual)
                geoms_T = geoms_T_by_mode[is_visual]

                for geom in geoms:
                    self.update_rigid_batch(str(geom.uid), geoms_T[:, geom.idx])

        # MPM particles
        if self.sim.mpm_solver.is_active:
            particles_all = self.sim.mpm_solver.particles_render.pos.to_numpy()[:, self.rendered_envs_idx[0]]
            particles_vel_all = self.sim.mpm_solver.particles_render.vel.to_numpy()[:, self.rendered_envs_idx[0]]
            active_all = self.sim.mpm_solver.particles_render.active.to_numpy().astype(bool)[
                :, self.rendered_envs_idx[0]
            ]
            vverts_all = self.sim.mpm_solver.vverts_render.pos.to_numpy()[:, self.rendered_envs_idx[0]]

            for mpm_entity in self.sim.mpm_solver.entities:
                if mpm_entity.surface.vis_mode == "visual":
                    vverts = vverts_all[mpm_entity.vvert_start : mpm_entity.vvert_end]

                    self.update_deformable(
                        str(mpm_entity.uid),
                        vverts,
                        mpm_entity.vmesh.trimesh.faces,
                        mpm_entity.vmesh.trimesh.vertex_normals,
                        np.array([]) if mpm_entity.vmesh.uvs is None else mpm_entity.vmesh.uvs,
                    )
                else:
                    particles = particles_all[mpm_entity.particle_start : mpm_entity.particle_end][
                        active_all[mpm_entity.particle_start : mpm_entity.particle_end]
                    ]
                    particles_vel = particles_vel_all[mpm_entity.particle_start : mpm_entity.particle_end][
                        active_all[mpm_entity.particle_start : mpm_entity.particle_end]
                    ]
                    self.update_particles(
                        str(mpm_entity.uid), particles, self.sim.mpm_solver.particle_radius, particles_vel
                    )

        # SPH particles
        if self.sim.sph_solver.is_active:
            particles_all = self.sim.sph_solver.particles_render.pos.to_numpy()[:, self.rendered_envs_idx[0]]
            particles_vel_all = self.sim.sph_solver.particles_render.vel.to_numpy()[:, self.rendered_envs_idx[0]]
            active_all = self.sim.sph_solver.particles_render.active.to_numpy().astype(bool)[
                :, self.rendered_envs_idx[0]
            ]

            for sph_entity in self.sim.sph_solver.entities:
                particles = particles_all[sph_entity.particle_start : sph_entity.particle_end][
                    active_all[sph_entity.particle_start : sph_entity.particle_end]
                ]
                particles_vel = particles_vel_all[sph_entity.particle_start : sph_entity.particle_end][
                    active_all[sph_entity.particle_start : sph_entity.particle_end]
                ]
                self.update_particles(
                    str(sph_entity.uid), particles, self.sim.sph_solver.particle_radius, particles_vel
                )

        # PBD entities
        if self.sim.pbd_solver.is_active:
            idx = self.rendered_envs_idx[0]
            particles_all = self.sim.pbd_solver.particles_render.pos.to_numpy()[:, idx]
            particles_vel_all = self.sim.pbd_solver.particles_render.vel.to_numpy()[:, idx]
            active_all = self.sim.pbd_solver.particles_render.active.to_numpy().astype(bool)[:, idx]
            vverts_all = self.sim.pbd_solver.vverts_render.pos.to_numpy()[:, idx]

            for pbd_entity in self.sim.pbd_solver.entities:
                if pbd_entity.surface.vis_mode == "visual":
                    vverts = vverts_all[pbd_entity.vvert_start : pbd_entity.vvert_end]

                    self.update_deformable(
                        str(pbd_entity.uid),
                        vverts,
                        pbd_entity.vmesh.trimesh.faces,
                        trimesh.Trimesh(
                            vertices=vverts, faces=pbd_entity.vmesh.trimesh.faces, process=False
                        ).vertex_normals,  # TODO: make this more principled
                        np.array([]) if pbd_entity.vmesh.uvs is None else pbd_entity.vmesh.uvs,
                    )
                else:
                    if self.render_particle_as == "sphere":
                        particles = particles_all[pbd_entity.particle_start : pbd_entity.particle_end][
                            active_all[pbd_entity.particle_start : pbd_entity.particle_end]
                        ]

                        self.update_particles(str(pbd_entity.uid), particles, self.sim.pbd_solver.particle_radius, None)
                    elif self.render_particle_as == "tet":
                        self.update_deformable(
                            str(pbd_entity.uid),
                            mu.transform_tets_mesh_verts(
                                pbd_entity._tets_mesh.vertices,
                                positions=particles_all[pbd_entity.particle_start : pbd_entity.particle_end],
                                zs=particles_vel_all[pbd_entity.particle_start : pbd_entity.particle_end],
                            ).astype(np.float32),
                            pbd_entity._tets_mesh.faces,
                            np.array([]),
                            np.array([]),
                        )

        # FEM entities
        if self.sim.fem_solver.is_active:
            vverts_pos, _, _ = self.sim.fem_solver.get_state_render(self.sim.cur_substep_local)
            vverts_all = miscu.qd_to_numpy(vverts_pos, self.rendered_envs_idx[0], keepdim=False, transpose=True)

            for fem_entity in self.sim.fem_solver.entities:
                if fem_entity.surface.vis_mode != "visual":
                    continue

                for vgeom in fem_entity.vgeoms:
                    render_verts = vverts_all[vgeom.vvert_start : vgeom.vvert_end]
                    vertex_normals = trimesh.Trimesh(
                        vertices=render_verts, faces=vgeom.vmesh.faces, process=False
                    ).vertex_normals
                    self.update_deformable(
                        str(vgeom.uid),
                        render_verts,
                        vgeom.vmesh.faces,
                        vertex_normals,
                        np.array([]) if vgeom.uvs is None else vgeom.uvs,
                    )

        # Flush the update buffer.
        self._scene.update_scene(time=self._t)
        self.camera_updated = False

    def render_camera(self, camera):
        b = self._scene.render_frame(camera=self._cameras[str(camera.uid)])
        img = np.frombuffer(b, dtype=np.uint8).reshape(camera.res[1], camera.res[0], 4)[:, :, :3]
        return img

    def destroy(self):
        self._shapes = dict()
        self._cameras = dict()
        LuisaRenderPy.destroy()

    @property
    def cameras(self):
        return self.visualizer.cameras

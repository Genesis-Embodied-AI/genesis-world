import ctypes
import os
import platform
import subprocess
import sys
import shutil
import tempfile
from multiprocessing import Process, Queue

import igl
import numpy as np
import trimesh

import genesis as gs

from . import geom as gu
from . import mesh as msu
from . import misc as miu

# Make sure ParticleMesherPy shared libary can be found in search path
LD_LIBRARY_PATH = os.path.join(miu.get_src_dir(), "ext/ParticleMesher/ParticleMesherPy")
sys.path.append(LD_LIBRARY_PATH)
os.environ["LD_LIBRARY_PATH"] = ":".join(filter(None, (os.environ.get("LD_LIBRARY_PATH"), LD_LIBRARY_PATH)))

try:
    malloc_trim = ctypes.CDLL(ctypes.util.find_library("c")).malloc_trim
except (AttributeError, TypeError):
    malloc_trim = None


def n_particles_vol(p_size=0.01, volume=1.0):
    return max(1, round(volume / p_size**3))


def n_particles_3D(p_size=0.01, size=(1.0, 1.0, 1.0)):
    return max(1, round(size[0] / p_size)) * max(1, round(size[1] / p_size)) * max(1, round(size[2] / p_size))


def n_particles_1D(p_size=0.01, length=1.0):
    return max(1, round(length / p_size))


def nowhere_particles(n):
    return np.tile(gu.nowhere(), (n, 1))


def trimesh_to_particles_simple(mesh, p_size, sampler):
    """
    Mesh to particles via `random` or `regular` sampler.
    """
    assert sampler in ("random", "regular")

    # compute file name via hashing for caching
    cache = msu.get_ptc_cache(mesh.vertices, mesh.faces, p_size, sampler)

    # loading pre-computed cache if available
    positions = cache.load()
    if positions is not None:
        gs.logger.debug("Sampled particles file (`.ptc`) found in cache.")
        return positions

    with gs.logger.timer(f"Sampling particles with ~<{sampler}>~ sampler and generating `.ptc` file:"):
        # sample a cube first
        box_size = mesh.bounds[1] - mesh.bounds[0]
        box_center = (mesh.bounds[1] + mesh.bounds[0]) / 2
        positions = _box_to_particles(p_size=p_size, pos=box_center, size=box_size, sampler=sampler)
        # reject out-of-boundary particles
        sd, *_ = igl.signed_distance(positions, mesh.vertices, mesh.faces)
        positions = positions[sd < 0.0]

        cache.save(positions)

    return positions


def trimesh_to_particles_pbs(mesh, p_size, sampler, pos=(0, 0, 0)):
    """
    Physics-based particle sampler using the method proposed by Kugelstadt et al. [2021].

    References: https://splishsplash.readthedocs.io/en/latest/VolumeSampling.html
    """
    assert "pbs" in sampler

    if not (sys.platform == "linux" and platform.machine() == "x86_64"):
        gs.raise_exception(f"Physics-based particle sampler '{sampler}' is only supported on Linux x86.")

    # compute file name via hashing for caching
    cache = msu.get_ptc_cache(mesh.vertices, mesh.faces, p_size, sampler)

    # loading pre-computed cache if available
    positions = cache.load()
    if positions is not None:
        gs.logger.debug("Sampled particles file (`.ptc`) found in cache.")
    else:
        with gs.logger.timer(f"Sampling particles with ~<{sampler}>~ sampler and generating `.ptc` file:"):
            sdf_res = int(sampler.split("-")[-1])

            # We scale up a bit the particle size because this method tends to sample denser particles compared to `random` and `regular` samplers.
            scale = 1.104  # Magic number from Pingchuan Ma.
            particle_radius = p_size * scale / 2

            fd, mesh_path = tempfile.mkstemp(suffix=".obj")
            os.close(fd)
            fd, vtk_path = tempfile.mkstemp(suffix=".vtk")
            os.close(fd)
            mesh.export(mesh_path)

            try:
                # Try importing VTK. It would fail on Linux if not graphic server is running.
                import vtk
                from vtk.util.numpy_support import vtk_to_numpy

                # Sample particles
                command = (
                    os.path.join(miu.get_src_dir(), "ext/VolumeSampling"),
                    "-i",
                    mesh_path,
                    "-o",
                    vtk_path,
                    "--no-cache",
                    "--mode",
                    4,
                    "-r",
                    particle_radius,
                    "--res",
                    f"{sdf_res},{sdf_res},{sdf_res}",
                    "--steps",
                    100,  # use bigger value leads to smoother surface
                )
                result = subprocess.run(map(str, command), capture_output=True, text=True)
                if result.stdout:
                    gs.logger.debug(result.stdout)
                if result.stderr:
                    gs.logger.warning(result.stderr)
                if os.path.getsize(vtk_path) == 0:
                    raise OSError("Output VTK file is empty.")

                # Read the generated VTK file
                reader = vtk.vtkUnstructuredGridReader()
                reader.SetFileName(vtk_path)
                reader.Update()
                positions = vtk_to_numpy(reader.GetOutput().GetPoints().GetData())
            except (OSError, ImportError) as e:
                gs.raise_exception_from(f"Physics-based particle sampler '{sampler}' failed.", e)
            finally:
                os.remove(mesh_path)
                os.remove(vtk_path)

            # Clean up the intermediate files
            output_dir = os.path.join(miu.get_src_dir(), "ext/output")
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir, ignore_errors=True)

            # Cache the generated positions
            cache.save(positions)

    positions += np.asarray(pos)

    return positions


def _box_to_particles(p_size, pos, size, sampler):
    """
    Private function to sample particles from a box. This function only supports `random` and `regular` samplers.
    This is a private function that does not consider additional mesh offset or scale.
    """
    size = np.array(size)
    pos = np.array(pos)
    lower = pos - size / 2
    upper = pos + size / 2

    if sampler == "random":
        n_particles = n_particles_3D(p_size, size)
        positions = np.random.uniform(low=lower, high=upper, size=(n_particles, 3))

    elif sampler == "regular":
        n_x = n_particles_1D(p_size, size[0])
        n_y = n_particles_1D(p_size, size[1])
        n_z = n_particles_1D(p_size, size[2])
        p_lower = lower + 0.5 * p_size
        p_upper = upper - 0.5 * p_size
        x = np.linspace(p_lower[0], p_upper[0], n_x)
        y = np.linspace(p_lower[1], p_upper[1], n_y)
        z = np.linspace(p_lower[2], p_upper[2], n_z)
        positions = np.stack(np.meshgrid(x, y, z, indexing="ij"), -1).reshape((-1, 3))

    else:
        gs.raise_exception(f"Unsupported sampler method: {sampler}.")

    return positions


def box_to_particles(p_size=0.01, pos=(0, 0, 0), size=(1, 1, 1), sampler="random"):
    if "pbs" in sampler:
        mesh = trimesh.creation.box(extents=size)

        try:
            positions = trimesh_to_particles_pbs(mesh, p_size, sampler, pos=pos)
        except gs.GenesisException:
            sampler = "random"

    if sampler in ("random", "regular"):
        positions = _box_to_particles(
            p_size=p_size,
            pos=pos,
            size=size,
            sampler=sampler,
        )

    return positions


def cylinder_to_particles(p_size=0.01, pos=(0, 0, 0), radius=0.5, height=1.0, sampler="random"):
    if "pbs" in sampler:
        mesh = trimesh.creation.cylinder(radius=radius, height=height)
        try:
            positions = trimesh_to_particles_pbs(mesh, p_size, sampler, pos=pos)
        except gs.GenesisException:
            sampler = "random"

    if sampler in ("random", "regular"):
        # sample a cube first
        size = np.array([2 * radius, 2 * radius, height])
        positions = _box_to_particles(
            p_size=p_size,
            pos=pos,
            size=size,
            sampler=sampler,
        )
        # reject out-of-boundary particles
        positions_r = np.linalg.norm(positions[:, [0, 1]] - np.array(pos)[[0, 1]], axis=1)
        positions = positions[positions_r <= radius]

    return positions


def sphere_to_particles(p_size=0.01, pos=(0, 0, 0), radius=0.5, sampler="random"):
    if "pbs" in sampler:
        mesh = trimesh.creation.icosphere(radius=radius)
        try:
            positions = trimesh_to_particles_pbs(mesh, p_size, sampler, pos=pos)
        except gs.GenesisException:
            sampler = "random"

    if sampler in ("random", "regular"):
        # sample a cube first
        size = np.array([2 * radius, 2 * radius, 2 * radius])
        positions = _box_to_particles(
            p_size=p_size,
            pos=pos,
            size=size,
            sampler=sampler,
        )
        # reject out-of-boundary particles
        positions_r = np.linalg.norm(positions - np.array(pos), axis=1)
        positions = positions[positions_r <= radius]

    return positions


def shell_to_particles(p_size=0.01, pos=(0, 0, 0), inner_radius=0.5, outer_radius=0.7, sampler="random"):
    if "pbs" in sampler:
        mesh = trimesh.creation.icosphere(radius=outer_radius)
        try:
            positions = trimesh_to_particles_pbs(mesh, p_size, sampler, pos=pos)
        except gs.GenesisException:
            sampler = "random"

    if sampler in ("random", "regular"):
        # sample a cube first
        size = np.array([2 * outer_radius, 2 * outer_radius, 2 * outer_radius])
        positions = _box_to_particles(
            p_size=p_size,
            pos=pos,
            size=size,
            sampler=sampler,
        )
        # reject out-of-boundary particles
        positions_r = np.linalg.norm(positions - np.array(pos), axis=1)
        positions = positions[positions_r <= outer_radius]

    # reject inner particles
    positions_r = np.linalg.norm(positions - np.array(pos), axis=1)
    positions = positions[positions_r >= inner_radius]

    return positions


def _splashsurf_worker(positions, radius, args_dict, result_queue):
    try:
        import pysplashsurf

        mesh_with_data, _ = pysplashsurf.reconstruction_pipeline(
            positions,
            particle_radius=radius * args_dict.get("rscale", 1.0),
            smoothing_length=2.0,
            cube_size=0.8,
            iso_surface_threshold=0.6,
            mesh_smoothing_weights=True,
            mesh_smoothing_iters=int(args_dict.get("smooth", 25)),
            normals_smoothing_iters=10,
            mesh_cleanup=True,
            compute_normals=True,
            multi_threading=True,
        )
        normals = mesh_with_data.point_attributes["normals"]
        vertices = mesh_with_data.mesh.vertices
        triangles = mesh_with_data.mesh.triangles
        result_queue.put_nowait((vertices, triangles, normals))
    except Exception as e:
        result_queue.put_nowait(e)
        raise


def particles_to_mesh(positions, radius, backend):
    def parse_args(backend):
        args_dict = dict()
        args_list = backend.split("-")
        if len(args_list) >= 2:
            args_dict["rscale"] = float(args_list[1])
            args_list = args_list[2:]
            for i in range(0, len(args_list), 2):
                args_dict[args_list[i]] = float(args_list[i + 1])
        return args_dict

    if positions.shape[0] == 0:
        return trimesh.Trimesh(vertices=np.zeros((0, 3)), faces=np.zeros((0, 3)))

    if isinstance(radius, np.ndarray):
        radii = radius
        radius = np.min(radius)
        if "splashsurf" in backend:
            gs.logger.warning(
                "Backend 'splashsurf' does not support specifying individual radius for each particle. Switching to "
                "backend 'openvdb' as a fallback."
            )
            backend = "openvdb"
    else:
        radii = np.array([])

    args_dict = parse_args(backend)

    if "openvdb" in backend:
        if sys.platform != "linux" or sys.version_info[:2] == (3, 9):
            gs.raise_exception("Backend 'openvdb' is only supported on Linux and Python 3.9 specifically.")

        import ParticleMesherPy

        radius_scale = args_dict.get("rscale", 2.0)
        reconstructor = ParticleMesherPy.MeshConstructor(
            ParticleMesherPy.MeshConstructorConfig(
                particle_radius=radius * radius_scale,
                voxel_scale=args_dict.get("vscale", 1.0),
                isovalue=args_dict.get("isovalue", 0.0),
                adaptivity=args_dict.get("adaptivity", 0.01),
            )
        )
        mesh = reconstructor.construct(positions=positions, radii=radii * radius_scale)
        gs.logger.debug(f"[ParticleMehser]: {mesh.info_msg}")
        vertices = mesh.vertices.reshape([-1, 3])
        faces = mesh.triangles.reshape([-1, 3])

        return trimesh.Trimesh(vertices, faces, process=False)
    elif "splashsurf" in backend:
        # FIXME: Running in subprocess or manually reclaiming free-ed head memory is necessary to avoid unbounded growth
        result_queue = Queue()
        if malloc_trim is not None:
            _splashsurf_worker(positions, radius, args_dict, result_queue)
            result = result_queue.get()
            malloc_trim(0)
        else:
            proc = Process(target=_splashsurf_worker, args=(positions, radius, args_dict, result_queue))
            proc.start()
            result = result_queue.get()
            proc.join()
            if proc.exitcode != 0:
                gs.raise_exception_from(f"splashsurf subprocess failed with exit code {proc.exitcode}", result)
        vertices, triangles, normals = result
        mesh = trimesh.Trimesh(vertices=vertices, faces=triangles, face_normals=normals, process=False)
        gs.logger.debug(f"[splashsurf]: reconstruct vertices: {mesh.vertices.shape}, {mesh.faces.shape}")
        return mesh
    else:
        gs.raise_exception(f"Unsupported backend: {backend}.")


def init_foam_generator(
    object_id,
    particle_radius,
    time_step,
    gravity,
    lower_bound,
    upper_bound,
    spray_decay,
    foam_decay,
    bubble_decay,
    k_foam,
    foam_density,
):
    if sys.platform != "linux" or sys.version_info[:2] == (3, 9):
        gs.raise_exception("This method is only supported on Linux and Python 3.9 specfically.")

    import ParticleMesherPy

    min_ke = 0.1 * (particle_radius**3 * 6400) * (2.5**2)
    return ParticleMesherPy.FoamGenerator(
        config=ParticleMesherPy.FoamGeneratorConfig(
            particle_radius=particle_radius,
            voxel_scale=1.0,
            time_step=time_step,
            lower_bound=tuple(lower_bound),
            upper_bound=tuple(upper_bound),
            gravity=tuple(gravity),
            lim_ta=(0.05, 0.5),
            lim_wc=(0.05, 0.5),
            lim_ke=(min_ke, min_ke * 50),
            generate_neighbor_min=5,
            k_foam=k_foam,
            k_ad=0.90,
            spray_decay=spray_decay,
            foam_decay=foam_decay,
            bubble_decay=bubble_decay,
            foam_density=foam_density,
        ),
        object_id=object_id,
    )


def generate_foam_particles(generator, positions, velocities):
    foams = generator.generate_foams(
        positions=positions,
        velocities=velocities,
    )
    gs.logger.debug(f"[ParticleMesher]: {foams.info_msg}")
    return foams.positions.reshape([-1, 3])


def filter_surface(positions, radii, particle_radius, half_width=8.0, radius_scale=1.0):
    if sys.platform != "linux" or sys.version_info[:2] == (3, 9):
        gs.raise_exception("This method is only supported on Linux and Python 3.9 specfically.")

    import ParticleMesherPy

    splitter = ParticleMesherPy.SurfaceSplitter(
        ParticleMesherPy.SurfaceSplitterConfig(
            particle_radius=particle_radius * radius_scale,
            voxel_scale=0.25,
            support_scale=4.0,
            half_width=half_width,
            surface_neighbor_max=20,
        )
    )

    # surface_indices = splitter.split_surface_sdf(positions, radii * radius_scale)
    surface_indices = splitter.split_surface_count(positions)
    gs.logger.debug(
        f"[ParticleMesher]: {surface_indices.info_msg}\n"
        f"\tFrom {positions.shape[0]} to {np.sum(surface_indices.is_surface)}"
    )
    return surface_indices.is_surface

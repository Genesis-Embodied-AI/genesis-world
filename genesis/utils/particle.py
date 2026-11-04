import ctypes
from multiprocessing import Process, Queue

import numba as nb
import numpy as np

import igl
import trimesh
from skimage.measure import marching_cubes

import genesis as gs

from . import geom as gu
from .cache import get_ptc_cache

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

    # The particles the random sampler draws follow the state of the global generator, which 'gs.init' seeds and the
    # sampling advances. An entry is therefore keyed on the state the sampling starts from and holds the state it ends
    # at, which a hit restores, so that a run draws exactly what it would draw without the cache.
    random_state = np.random.get_state() if sampler == "random" else None
    cache = get_ptc_cache(mesh.vertices, mesh.faces, p_size, sampler, random_state)

    entry = cache.load()
    if entry is not None:
        gs.logger.debug("Sampled particles file (`.ptc`) found in cache.")
        positions, random_state_end = entry
        if random_state_end is not None:
            np.random.set_state(random_state_end)
        return positions

    with gs.logger.timer(f"Sampling particles with ~<{sampler}>~ sampler and generating `.ptc` file:"):
        # sample a cube first
        box_size = mesh.bounds[1] - mesh.bounds[0]
        box_center = (mesh.bounds[1] + mesh.bounds[0]) / 2
        positions = _box_to_particles(p_size=p_size, pos=box_center, size=box_size, sampler=sampler)
        # reject out-of-boundary particles
        # See compute_sdf_data in genesis/utils/mesh.py for the sign computation
        sd, *_ = igl.signed_distance(
            positions, mesh.vertices, mesh.faces, sign_type=igl.SIGNED_DISTANCE_TYPE_FAST_WINDING_NUMBER
        )
        positions = positions[sd < 0.0]

        cache.save((positions, np.random.get_state() if sampler == "random" else None))

    return positions


def trimesh_to_particles_pbs(mesh, p_size, sampler, pos=(0, 0, 0)):
    """Sample particles uniformly in the volume of a mesh, following Kugelstadt et al. 2021.

    Particles are seeded on a lattice inside the mesh, then relaxed by a few smoothed-particle hydrodynamics (SPH) steps
    that even out their density while keeping them inside. The sampler is named 'pbs-<res>', 'res' being the number of
    cells per axis of the signed distance field of the mesh that holds the particles inside.
    """
    assert "pbs" in sampler

    # The physics-based sampler draws nothing from the global generator, which its entries leave out of their key
    cache = get_ptc_cache(mesh.vertices, mesh.faces, p_size, sampler, None)
    positions = cache.load()
    if positions is not None:
        gs.logger.debug("Sampled particles file (`.ptc`) found in cache.")
    else:
        with gs.logger.timer(f"Sampling particles with ~<{sampler}>~ sampler and generating `.ptc` file:"):
            sdf_res = int(sampler.split("-")[-1])
            # This method samples denser particles than the 'random' and 'regular' samplers at the same radius
            radius = 1.104 * p_size / 2

            # Signed distance field, positive inside, over the bounding box of the mesh padded by the support radius
            origin = mesh.bounds[0] - 4.0 * radius
            cell_size = (mesh.bounds[1] + 4.0 * radius - origin) / sdf_res
            nodes = origin + cell_size * np.stack(np.meshgrid(*[np.arange(sdf_res + 1)] * 3, indexing="ij"), axis=-1)
            sdf, *_ = igl.signed_distance(nodes.reshape((-1, 3)), mesh.vertices, mesh.faces)
            sdf = -sdf.reshape((sdf_res + 1,) * 3)

            # The particles are seeded slightly closer than their final spacing, on layers of rows offset by a radius
            # every other row, then relaxed towards a uniform density
            seed_radius = 0.95 * radius
            spacing = np.array([2.0 * seed_radius, np.sqrt(3.0) * seed_radius, 2.0 * seed_radius])
            lattice_x, lattice_y, lattice_z = [
                np.arange(mesh.bounds[0][k], mesh.bounds[1][k] + 0.5 * spacing[k], spacing[k]) for k in range(3)
            ]
            lattice_x, lattice_y, lattice_z = [
                lattice_x[lattice_x <= mesh.bounds[1][0]],
                lattice_y[lattice_y <= mesh.bounds[1][1]],
                lattice_z[lattice_z <= mesh.bounds[1][2]],
            ]
            grid_z, grid_y, grid_x = np.meshgrid(lattice_z, np.arange(len(lattice_y)), lattice_x, indexing="ij")
            is_row_odd = grid_y % 2 == 1
            positions = np.stack(
                (
                    grid_x + is_row_odd * seed_radius,
                    lattice_y[grid_y] + seed_radius,
                    grid_z + ~is_row_odd * seed_radius,
                ),
                axis=-1,
            ).reshape((-1, 3))
            positions = _relax_volume_sampling(positions, sdf, origin, cell_size, radius, n_steps=100)
            cache.save(positions)

    return positions + pos


@nb.jit(nopython=True, cache=True)
def _interpolate_sdf(sdf, origin, cell_size, point):
    """Interpolate trilinearly a signed distance field sampled on the nodes of a regular grid, and its gradient.

    Return the distance followed by the 3 components of its gradient, the distance being infinite outside the grid.
    """
    c_x = (point[0] - origin[0]) / cell_size[0]
    c_y = (point[1] - origin[1]) / cell_size[1]
    c_z = (point[2] - origin[2]) / cell_size[2]
    if not (0.0 <= c_x <= sdf.shape[0] - 1 and 0.0 <= c_y <= sdf.shape[1] - 1 and 0.0 <= c_z <= sdf.shape[2] - 1):
        return np.inf, 0.0, 0.0, 0.0
    i_x, i_y, i_z = min(int(c_x), sdf.shape[0] - 2), min(int(c_y), sdf.shape[1] - 2), min(int(c_z), sdf.shape[2] - 2)
    f_x, f_y, f_z = c_x - i_x, c_y - i_y, c_z - i_z
    dist, grad_x, grad_y, grad_z = 0.0, 0.0, 0.0, 0.0
    for o_x in range(2):
        w_x, dw_x = (f_x, 1.0) if o_x else (1.0 - f_x, -1.0)
        for o_y in range(2):
            w_y, dw_y = (f_y, 1.0) if o_y else (1.0 - f_y, -1.0)
            for o_z in range(2):
                w_z, dw_z = (f_z, 1.0) if o_z else (1.0 - f_z, -1.0)
                value = sdf[i_x + o_x, i_y + o_y, i_z + o_z]
                dist += w_x * w_y * w_z * value
                grad_x += dw_x * w_y * w_z * value
                grad_y += w_x * dw_y * w_z * value
                grad_z += w_x * w_y * dw_z * value
    return dist, grad_x / cell_size[0], grad_y / cell_size[1], grad_z / cell_size[2]


@nb.jit(nopython=True, cache=True, parallel=True)
def _relax_volume_sampling(positions, sdf, origin, cell_size, radius, n_steps):
    """Relax particles sampling the volume of a shape towards a uniform density, keeping them inside the shape.

    The shape is the positive region of a signed distance field sampled on a grid (see '_interpolate_sdf'). Each step
    pulls neighbors to a spacing of one diameter (cohesion) and particles near the boundary towards it (adhesion), then
    corrects the density by one pressure iteration of divergence-free SPH, before projecting back inside any particle
    closer to the boundary than its radius. Only the seeds inside the shape are kept.
    """
    diameter = 2.0 * radius
    support_radius = 4.0 * radius
    volume = diameter**3
    density_rest = 1000.0
    mass = volume * density_rest
    cohesion, adhesion, cfl_factor, eps = 4.0, 2.0, 0.5, 1e-5
    # Coincident particles have no direction to push each other along
    dist_min = 1e-6 * radius

    is_inside = np.empty(len(positions), dtype=np.bool_)
    for i in nb.prange(len(positions)):
        is_inside[i] = 0.0 < _interpolate_sdf(sdf, origin, cell_size, positions[i])[0] < np.inf
    positions = positions[is_inside]

    n_points = len(positions)
    if n_points == 0:
        return positions
    densities = np.empty(n_points, dtype=positions.dtype)
    factors = np.empty(n_points, dtype=positions.dtype)
    corrs = np.empty((n_points, 3), dtype=positions.dtype)
    corrs_sqr = np.empty(n_points, dtype=positions.dtype)
    for i_step in range(n_steps):
        neighbors_start, neighbors_idx = _find_neighbors(positions, positions, support_radius)

        for i in nb.prange(n_points):
            density = 0.0
            for i_n in range(neighbors_start[i], neighbors_start[i + 1]):
                j = neighbors_idx[i_n]
                d_x, d_y, d_z = (
                    positions[i, 0] - positions[j, 0],
                    positions[i, 1] - positions[j, 1],
                    positions[i, 2] - positions[j, 2],
                )
                density += _cubic_spline_kernel(np.sqrt(d_x * d_x + d_y * d_y + d_z * d_z), support_radius)
            densities[i] = mass * density

        # Cohesion and adhesion, then the inverse diagonal of the pressure system of divergence-free SPH
        for i in nb.prange(n_points):
            corr_x, corr_y, corr_z = 0.0, 0.0, 0.0
            grad_x, grad_y, grad_z, grad_sqr_sum = 0.0, 0.0, 0.0, 0.0
            for i_n in range(neighbors_start[i], neighbors_start[i + 1]):
                j = neighbors_idx[i_n]
                d_x, d_y, d_z = (
                    positions[i, 0] - positions[j, 0],
                    positions[i, 1] - positions[j, 1],
                    positions[i, 2] - positions[j, 2],
                )
                dist = np.sqrt(d_x * d_x + d_y * d_y + d_z * d_z)
                if dist > dist_min:
                    cohesion_scale = cohesion * mass / densities[j] * (dist - diameter)
                    cohesion_scale *= _cubic_spline_kernel(dist, support_radius) / dist
                    corr_x -= cohesion_scale * d_x
                    corr_y -= cohesion_scale * d_y
                    corr_z -= cohesion_scale * d_z
                    grad_scale = volume * _cubic_spline_kernel_derivative(dist, support_radius)
                    grad_x += grad_scale * d_x / dist
                    grad_y += grad_scale * d_y / dist
                    grad_z += grad_scale * d_z / dist
                    grad_sqr_sum += grad_scale * grad_scale
            sdf_dist, sdf_grad_x, sdf_grad_y, sdf_grad_z = _interpolate_sdf(sdf, origin, cell_size, positions[i])
            if sdf_dist < support_radius:
                adhesion_scale = adhesion * mass / densities[i] * sdf_dist
                adhesion_scale *= _cubic_spline_kernel(abs(sdf_dist), support_radius)
                adhesion_scale /= np.sqrt(sdf_grad_x * sdf_grad_x + sdf_grad_y * sdf_grad_y + sdf_grad_z * sdf_grad_z)
                corr_x -= adhesion_scale * sdf_grad_x
                corr_y -= adhesion_scale * sdf_grad_y
                corr_z -= adhesion_scale * sdf_grad_z
            corrs[i, 0], corrs[i, 1], corrs[i, 2] = corr_x, corr_y, corr_z
            grad_sqr_sum += grad_x * grad_x + grad_y * grad_y + grad_z * grad_z
            factors[i] = -1.0 / grad_sqr_sum if grad_sqr_sum > eps else 0.0

        # One pressure iteration of divergence-free SPH towards the rest density
        for i in nb.prange(n_points):
            k_i = (densities[i] / density_rest - 1.0) * factors[i]
            for i_n in range(neighbors_start[i], neighbors_start[i + 1]):
                j = neighbors_idx[i_n]
                k_sum = k_i + (densities[j] / density_rest - 1.0) * factors[j]
                d_x, d_y, d_z = (
                    positions[i, 0] - positions[j, 0],
                    positions[i, 1] - positions[j, 1],
                    positions[i, 2] - positions[j, 2],
                )
                dist = np.sqrt(d_x * d_x + d_y * d_y + d_z * d_z)
                if dist > dist_min and abs(k_sum) > eps:
                    pressure_scale = k_sum * volume * _cubic_spline_kernel_derivative(dist, support_radius) / dist
                    corrs[i, 0] += pressure_scale * d_x
                    corrs[i, 1] += pressure_scale * d_y
                    corrs[i, 2] += pressure_scale * d_z
            corrs_sqr[i] = corrs[i, 0] ** 2 + corrs[i, 1] ** 2 + corrs[i, 2] ** 2

        # The step is bounded so that no particle moves farther than a fraction of its diameter
        max_corr_sqr = corrs_sqr.max()
        step_size = min(cfl_factor * 0.4 * diameter / np.sqrt(max_corr_sqr), 1.0) if max_corr_sqr > 0.0 else 1.0
        for i in nb.prange(n_points):
            for k in range(3):
                positions[i, k] += step_size * corrs[i, k]
            sdf_dist, sdf_grad_x, sdf_grad_y, sdf_grad_z = _interpolate_sdf(sdf, origin, cell_size, positions[i])
            if sdf_dist < radius:
                projection_scale = radius - sdf_dist
                projection_scale /= np.sqrt(sdf_grad_x * sdf_grad_x + sdf_grad_y * sdf_grad_y + sdf_grad_z * sdf_grad_z)
                positions[i, 0] += projection_scale * sdf_grad_x
                positions[i, 1] += projection_scale * sdf_grad_y
                positions[i, 2] += projection_scale * sdf_grad_z
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
        # The draws are scaled by separate numpy operations, each rounded exactly, so that every platform samples the
        # same positions. The compiled scaling of 'np.random.uniform' rounds differently from one platform to another.
        positions = lower + np.random.random((n_particles, 3)) * (upper - lower)

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
                "backend 'levelset' as a fallback."
            )
            backend = "levelset"
    else:
        radii = np.array([])

    args_dict = parse_args(backend)

    if "levelset" in backend:
        radius_scale = args_dict.get("rscale", 2.0)
        if radii.size == 0:
            radii = np.full(len(positions), radius, dtype=positions.dtype)
        radii = radii * radius_scale
        voxel_size = radius * radius_scale * args_dict.get("vscale", 1.0)
        isovalue = args_dict.get("isovalue", 0.0)
        # The surface is extracted from the distance at the corners of the cells it crosses, which lie within one cell
        # diagonal of it, so the distance only has to be exact within that band around the iso-surface.
        sdf, origin = _rasterize_spheres(positions, radii, voxel_size, band=abs(isovalue) + 2.0 * voxel_size)
        vertices, faces, _, _ = marching_cubes(
            sdf,
            level=isovalue,
            spacing=(voxel_size,) * 3,
            gradient_direction="descent",
            mask=np.abs(sdf - isovalue) < 2.0 * voxel_size,
        )
        return trimesh.Trimesh(origin + vertices, faces, process=False)
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


@nb.jit(nopython=True, cache=True, parallel=True)
def _rasterize_spheres(positions, radii, voxel_size, band):
    """Sample the signed distance to the union of a set of spheres on a regular grid, clamped to a band.

    The grid covers the spheres padded by the band and one more voxel, so that the surface closes inside it. Return the
    distance and the position of the grid node of index (0, 0, 0), the node of index (i, j, k) lying 'voxel_size *
    (i, j, k)' away from it. The distance is exact wherever it lies within 'band' of the surface, and saturates at
    -band inside and +band outside.
    """
    padding = band + voxel_size
    origin = np.empty(3, dtype=positions.dtype)
    shape = np.empty(3, dtype=np.int64)
    for i_d in range(3):
        lower, upper = np.inf, -np.inf
        for i_p in range(len(positions)):
            lower = min(lower, positions[i_p, i_d] - radii[i_p])
            upper = max(upper, positions[i_p, i_d] + radii[i_p])
        origin[i_d] = lower - padding
        shape[i_d] = int(np.ceil((upper + padding - origin[i_d]) / voxel_size)) + 1
    sdf = np.full((shape[0], shape[1], shape[2]), band, dtype=positions.dtype)

    # Each slab of the grid along the first axis is filled by a single thread, from the spheres that reach it. Binning
    # the spheres by slab makes those of neighboring slabs contiguous.
    spheres_slab = np.empty(len(positions), dtype=np.int64)
    for i_p in range(len(positions)):
        spheres_slab[i_p] = int((positions[i_p, 0] - origin[0]) / voxel_size)
    slabs_start = np.zeros(shape[0] + 1, dtype=np.int64)
    for i_p in range(len(positions)):
        slabs_start[spheres_slab[i_p] + 1] += 1
    slabs_start = np.cumsum(slabs_start)
    slabs_fill = slabs_start[:-1].copy()
    spheres_order = np.empty(len(positions), dtype=np.int64)
    for i_p in range(len(positions)):
        spheres_order[slabs_fill[spheres_slab[i_p]]] = i_p
        slabs_fill[spheres_slab[i_p]] += 1

    reach_slabs = int(np.ceil((radii.max() + band) / voxel_size)) + 1
    for i_x in nb.prange(shape[0]):
        x = origin[0] + i_x * voxel_size
        for i_s in range(slabs_start[max(i_x - reach_slabs, 0)], slabs_start[min(i_x + reach_slabs + 1, shape[0])]):
            i_p = spheres_order[i_s]
            # Only the voxels within reach of the sphere are visited, row by row of the disk it cuts in the slab
            reach = radii[i_p] + band
            dx = x - positions[i_p, 0]
            reach_yz_sqr = reach * reach - dx * dx
            if reach_yz_sqr < 0.0:
                continue
            reach_yz = np.sqrt(reach_yz_sqr)
            i_y_min = max(int(np.ceil((positions[i_p, 1] - reach_yz - origin[1]) / voxel_size)), 0)
            i_y_max = min(int(np.floor((positions[i_p, 1] + reach_yz - origin[1]) / voxel_size)), shape[1] - 1)
            for i_y in range(i_y_min, i_y_max + 1):
                dy = origin[1] + i_y * voxel_size - positions[i_p, 1]
                reach_z = np.sqrt(max(reach_yz_sqr - dy * dy, 0.0))
                i_z_min = max(int(np.ceil((positions[i_p, 2] - reach_z - origin[2]) / voxel_size)), 0)
                i_z_max = min(int(np.floor((positions[i_p, 2] + reach_z - origin[2]) / voxel_size)), shape[2] - 1)
                for i_z in range(i_z_min, i_z_max + 1):
                    dz = origin[2] + i_z * voxel_size - positions[i_p, 2]
                    dist = np.sqrt(dx * dx + dy * dy + dz * dz) - radii[i_p]
                    if dist < sdf[i_x, i_y, i_z]:
                        sdf[i_x, i_y, i_z] = max(dist, -band)
    return sdf, origin


@nb.jit(nopython=True, cache=True)
def _build_cell_grid(points, cell_size):
    """Bin a set of points in cubic cells of a given size over their bounding box, sorted by linear cell index.

    Return the origin and the shape of the grid, the offset of the first point of every column of cells along the last
    axis in sorted order followed by the number of points, the sorted linear cell index of every point, the indices of
    the points in that order, and their positions in that order. Only the columns are indexed densely, so that the
    memory follows the extent of the points in 2 dimensions rather than 3.
    """
    origin = np.zeros(3, dtype=points.dtype)
    shape = np.ones(3, dtype=np.int64)
    if len(points) > 0:
        for i_d in range(3):
            origin[i_d] = points[:, i_d].min()
            shape[i_d] = int((points[:, i_d].max() - origin[i_d]) / cell_size) + 1
    cells_key = np.empty(len(points), dtype=np.int64)
    for i_p in range(len(points)):
        i_x = int((points[i_p, 0] - origin[0]) / cell_size)
        i_y = int((points[i_p, 1] - origin[1]) / cell_size)
        i_z = int((points[i_p, 2] - origin[2]) / cell_size)
        cells_key[i_p] = (i_x * shape[1] + i_y) * shape[2] + i_z
    points_order = np.argsort(cells_key, kind="mergesort")
    cells_key_sorted = cells_key[points_order]
    columns_start = np.searchsorted(cells_key_sorted, np.arange(shape[0] * shape[1] + 1) * shape[2])
    return origin, shape, columns_start, cells_key_sorted, points_order, points[points_order]


@nb.jit(nopython=True, cache=True)
def _find_cell_grid_ranges(origin, shape, columns_start, cells_key_sorted, cell_size, query, ranges):
    """Find the points binned in the 27 cells around the cell of a query point, as ranges of their sorted index.

    A cell grid as wide as a support radius holds all the points closer than that radius to the query point. The 3
    cells of a column are contiguous in sorted order, so that one range holds them, found by binary search within the
    points of the column. The ranges are written in 'ranges', of shape (9, 2), and their number is returned.
    """
    i_x = int(np.floor((query[0] - origin[0]) / cell_size))
    i_y = int(np.floor((query[1] - origin[1]) / cell_size))
    i_z = int(np.floor((query[2] - origin[2]) / cell_size))
    i_z_min, i_z_max = max(i_z - 1, 0), min(i_z + 1, shape[2] - 1)
    n_ranges = 0
    if i_z_min <= i_z_max:
        for i_cx in range(max(i_x - 1, 0), min(i_x + 2, shape[0])):
            for i_cy in range(max(i_y - 1, 0), min(i_y + 2, shape[1])):
                i_c = i_cx * shape[1] + i_cy
                i_start, i_end = columns_start[i_c], columns_start[i_c + 1]
                column_keys = cells_key_sorted[i_start:i_end]
                ranges[n_ranges, 0] = i_start + np.searchsorted(column_keys, i_c * shape[2] + i_z_min, side="left")
                ranges[n_ranges, 1] = i_start + np.searchsorted(column_keys, i_c * shape[2] + i_z_max, side="right")
                n_ranges += 1
    return n_ranges


@nb.vectorize(cache=True)
def _cubic_spline_kernel(dist, support_radius):
    """Evaluate the cubic spline kernel of smoothed-particle hydrodynamics (SPH), vanishing at the support radius."""
    q = dist / support_radius
    scale = 8.0 / (np.pi * support_radius**3)
    if q <= 0.5:
        return scale * (6.0 * (q - 1.0) * q * q + 1.0)
    if q <= 1.0:
        return 2.0 * scale * (1.0 - q) ** 3
    return 0.0


@nb.vectorize(cache=True)
def _cubic_spline_kernel_derivative(dist, support_radius):
    """Evaluate the derivative with respect to the distance of the cubic spline kernel (see '_cubic_spline_kernel')."""
    q = dist / support_radius
    scale = 48.0 / (np.pi * support_radius**4)
    if q <= 0.5:
        return scale * q * (3.0 * q - 2.0)
    if q <= 1.0:
        return -scale * (1.0 - q) ** 2
    return 0.0


@nb.jit(nopython=True, cache=True, parallel=True)
def _find_neighbors(points, queries, support_radius):
    """Find the points closer than a support radius to each query point, the query point itself included.

    Return the start offset of the neighbors of each query point, then the concatenated indices of these neighbors in
    'points': the neighbors of query point i are 'neighbors_idx[neighbors_start[i]:neighbors_start[i + 1]]'.
    """
    origin, shape, columns_start, cells_key_sorted, points_order, points_sorted = _build_cell_grid(
        points, support_radius
    )

    # The neighbors are counted first, then written once the output is allocated at its exact size
    neighbors_count = np.empty(len(queries), dtype=np.int64)
    neighbors_start = np.zeros(len(queries) + 1, dtype=np.int64)
    neighbors_idx = np.empty(0, dtype=np.int64)
    for i_pass in range(2):
        for i_q in nb.prange(len(queries)):
            ranges = np.empty((9, 2), dtype=np.int64)
            n_ranges = _find_cell_grid_ranges(
                origin, shape, columns_start, cells_key_sorted, support_radius, queries[i_q], ranges
            )
            n_neighbors = 0
            for i_r in range(n_ranges):
                for i_s in range(ranges[i_r, 0], ranges[i_r, 1]):
                    d_x = queries[i_q, 0] - points_sorted[i_s, 0]
                    d_y = queries[i_q, 1] - points_sorted[i_s, 1]
                    d_z = queries[i_q, 2] - points_sorted[i_s, 2]
                    if d_x * d_x + d_y * d_y + d_z * d_z < support_radius * support_radius:
                        if i_pass == 1:
                            neighbors_idx[neighbors_start[i_q] + n_neighbors] = points_order[i_s]
                        n_neighbors += 1
            neighbors_count[i_q] = n_neighbors
        if i_pass == 0:
            neighbors_start[1:] = np.cumsum(neighbors_count)
            neighbors_idx = np.empty(neighbors_start[-1], dtype=np.int64)
    return neighbors_start, neighbors_idx


@nb.jit(nopython=True, cache=True, parallel=True)
def _advect_foams(
    positions,
    velocities,
    foams_pos,
    foams_vel,
    foams_lifetime,
    gravity,
    lower_bound,
    upper_bound,
    time_step,
    support_radius,
    spray_decay,
    foam_decay,
    bubble_decay,
):
    """Classify each foam particle by the fluid particles around it, then advect it over one time step in place.

    A foam particle with fewer than 6 fluid neighbors is spray, falling under gravity and air damping. One with more
    than 20 is an air bubble, rising by buoyancy and dragged by the fluid. Any other one is foam, moving with the fluid,
    whose velocity is the average of that of its neighbors weighted by the SPH kernel. Near a wall, the neighbors
    missing beyond it are added back in proportion to the volume of the support sphere it cuts off.

    Return the positions, velocities and lifetimes of the foam particles still alive: inside the domain of the fluid,
    before the end of their lifetime, and with at least 2 fluid neighbors.
    """
    origin, shape, columns_start, cells_key_sorted, points_order, points_sorted = _build_cell_grid(
        positions, support_radius
    )
    is_alive = np.empty(len(foams_pos), dtype=np.bool_)
    for i_f in nb.prange(len(foams_pos)):
        ranges = np.empty((9, 2), dtype=np.int64)
        n_ranges = _find_cell_grid_ranges(
            origin, shape, columns_start, cells_key_sorted, support_radius, foams_pos[i_f], ranges
        )
        n_neighbors, weights_sum = 0, 0.0
        fluid_vel_x, fluid_vel_y, fluid_vel_z = 0.0, 0.0, 0.0
        for i_r in range(n_ranges):
            for i_s in range(ranges[i_r, 0], ranges[i_r, 1]):
                d_x = foams_pos[i_f, 0] - points_sorted[i_s, 0]
                d_y = foams_pos[i_f, 1] - points_sorted[i_s, 1]
                d_z = foams_pos[i_f, 2] - points_sorted[i_s, 2]
                dist_sqr = d_x * d_x + d_y * d_y + d_z * d_z
                if dist_sqr < support_radius * support_radius:
                    weight = _cubic_spline_kernel(np.sqrt(dist_sqr), support_radius)
                    i_p = points_order[i_s]
                    fluid_vel_x += weight * velocities[i_p, 0]
                    fluid_vel_y += weight * velocities[i_p, 1]
                    fluid_vel_z += weight * velocities[i_p, 2]
                    weights_sum += weight
                    n_neighbors += 1
        if weights_sum > 0.0:
            fluid_vel_x, fluid_vel_y, fluid_vel_z = (
                fluid_vel_x / weights_sum,
                fluid_vel_y / weights_sum,
                fluid_vel_z / weights_sum,
            )

        is_inside = True
        is_corrected = 6 <= n_neighbors <= 20
        for k in range(3):
            is_inside = is_inside and lower_bound[k] <= foams_pos[i_f, k] <= upper_bound[k]
            for bound_dist in (foams_pos[i_f, k] - lower_bound[k], upper_bound[k] - foams_pos[i_f, k]):
                cap = 1.0 - abs(bound_dist) / support_radius
                if is_corrected and cap > 0.0:
                    n_neighbors += int(np.ceil(n_neighbors * cap * cap * (np.pi / 3.0) * (3.0 - cap)))

        if n_neighbors < 6:
            for k in range(3):
                foams_vel[i_f, k] = 0.9 * (foams_vel[i_f, k] + time_step * gravity[k])
            foams_lifetime[i_f] -= time_step * spray_decay
        elif n_neighbors > 20:
            foams_vel[i_f, 0] += -2.0 * time_step * gravity[0] + 0.8 * (fluid_vel_x - foams_vel[i_f, 0])
            foams_vel[i_f, 1] += -2.0 * time_step * gravity[1] + 0.8 * (fluid_vel_y - foams_vel[i_f, 1])
            foams_vel[i_f, 2] += -2.0 * time_step * gravity[2] + 0.8 * (fluid_vel_z - foams_vel[i_f, 2])
            foams_lifetime[i_f] -= time_step * bubble_decay
        else:
            foams_vel[i_f, 0], foams_vel[i_f, 1], foams_vel[i_f, 2] = fluid_vel_x, fluid_vel_y, fluid_vel_z
            foams_lifetime[i_f] -= time_step * foam_decay
        for k in range(3):
            foams_pos[i_f, k] += time_step * foams_vel[i_f, k]
        is_alive[i_f] = is_inside and foams_lifetime[i_f] > 0.0 and n_neighbors >= 2

    alive_offsets = np.cumsum(is_alive)
    n_alive = alive_offsets[-1] if len(foams_pos) else 0
    alive_pos = np.empty((n_alive, 3), dtype=foams_pos.dtype)
    alive_vel = np.empty((n_alive, 3), dtype=foams_vel.dtype)
    alive_lifetime = np.empty(n_alive, dtype=foams_lifetime.dtype)
    for i_f in nb.prange(len(foams_pos)):
        if is_alive[i_f]:
            i_a = alive_offsets[i_f] - 1
            for k in range(3):
                alive_pos[i_a, k] = foams_pos[i_f, k]
                alive_vel[i_a, k] = foams_vel[i_f, k]
            alive_lifetime[i_a] = foams_lifetime[i_f]
    return alive_pos, alive_vel, alive_lifetime


@nb.jit(nopython=True, cache=True, parallel=True)
def _compute_emission_potentials(positions, velocities, support_radius, particle_mass, eps):
    """Compute the potentials of each fluid particle to trap air and to form a wave crest (Ihmsen et al. 2012).

    Return the number of neighbors of each particle, its surface normal, zero away from the free surface, its
    trapped-air potential (Eq. 2) and its wave-crest curvature (Eq. 4). Distances and normals smaller than 'eps' have
    no direction and contribute nothing.
    """
    neighbors_start, neighbors_idx = _find_neighbors(positions, positions, support_radius)
    n_points = len(positions)
    # The contribution of a particle to its own density is counted twice, once by itself and once among its
    # neighbors. The thresholds of the emission potentials of 'FoamGenerator' are tuned for this density.
    densities = np.empty(n_points, dtype=positions.dtype)
    for i in nb.prange(n_points):
        density = _cubic_spline_kernel(0.0, support_radius)
        for i_n in range(neighbors_start[i], neighbors_start[i + 1]):
            j = neighbors_idx[i_n]
            d_x, d_y, d_z = (
                positions[i, 0] - positions[j, 0],
                positions[i, 1] - positions[j, 1],
                positions[i, 2] - positions[j, 2],
            )
            density += _cubic_spline_kernel(np.sqrt(d_x * d_x + d_y * d_y + d_z * d_z), support_radius)
        densities[i] = particle_mass * density

    # Normals only matter near the free surface, which a particle with many neighbors is far from
    normals = np.zeros((n_points, 3), dtype=positions.dtype)
    for i in nb.prange(n_points):
        if neighbors_start[i + 1] - neighbors_start[i] > 20:
            continue
        n_x, n_y, n_z = 0.0, 0.0, 0.0
        for i_n in range(neighbors_start[i], neighbors_start[i + 1]):
            j = neighbors_idx[i_n]
            d_x, d_y, d_z = (
                positions[i, 0] - positions[j, 0],
                positions[i, 1] - positions[j, 1],
                positions[i, 2] - positions[j, 2],
            )
            dist = np.sqrt(d_x * d_x + d_y * d_y + d_z * d_z)
            if dist > eps:
                scale = particle_mass / densities[j] * _cubic_spline_kernel_derivative(dist, support_radius) / dist
                n_x -= scale * d_x
                n_y -= scale * d_y
                n_z -= scale * d_z
        norm = np.sqrt(n_x * n_x + n_y * n_y + n_z * n_z)
        if norm > eps:
            normals[i, 0], normals[i, 1], normals[i, 2] = n_x / norm, n_y / norm, n_z / norm

    trapped_air = np.zeros(n_points, dtype=positions.dtype)
    curvature = np.zeros(n_points, dtype=positions.dtype)
    for i in nb.prange(n_points):
        for i_n in range(neighbors_start[i], neighbors_start[i + 1]):
            j = neighbors_idx[i_n]
            d_x, d_y, d_z = (
                positions[i, 0] - positions[j, 0],
                positions[i, 1] - positions[j, 1],
                positions[i, 2] - positions[j, 2],
            )
            dist = np.sqrt(d_x * d_x + d_y * d_y + d_z * d_z)
            if dist <= eps:
                continue
            d_x, d_y, d_z = d_x / dist, d_y / dist, d_z / dist
            weight = particle_mass / densities[j] * _cubic_spline_kernel(dist, support_radius)
            v_x, v_y, v_z = (
                velocities[i, 0] - velocities[j, 0],
                velocities[i, 1] - velocities[j, 1],
                velocities[i, 2] - velocities[j, 2],
            )
            v_norm = np.sqrt(v_x * v_x + v_y * v_y + v_z * v_z)
            if v_norm > 1e-6:
                trapped_air[i] += weight * v_norm * (1.0 - (v_x * d_x + v_y * d_y + v_z * d_z) / v_norm)
            else:
                trapped_air[i] += weight * v_norm
            if d_x * normals[i, 0] + d_y * normals[i, 1] + d_z * normals[i, 2] > 0.0:
                n_dot = normals[i, 0] * normals[j, 0] + normals[i, 1] * normals[j, 1] + normals[i, 2] * normals[j, 2]
                curvature[i] += weight * (1.0 - n_dot)
    return np.diff(neighbors_start), normals, trapped_air, curvature


class FoamGenerator:
    """Generate the spray, foam and air bubble particles of a fluid.

    The model follows Ihmsen et al., "Unified spray, foam and air bubbles for particle-based fluids", 2012. Fluid
    particles trapping air, at a wave crest, or moving fast emit foam particles at each step. A foam particle surrounded
    by few fluid particles is spray, falling under gravity and air damping. One surrounded by many is an air bubble,
    rising by buoyancy and dragged by the fluid. Any other one is foam, carried along by the fluid. Every foam particle
    disappears at the end of its lifetime or once it leaves the domain of the fluid.

    Parameters
    ----------
    particle_radius : float
        The radius of the fluid particles.
    time_step : float
        The time elapsed between two consecutive calls to 'generate'.
    gravity : array_like, shape (3,)
        The gravity acceleration.
    lower_bound, upper_bound : array_like, shape (3,)
        The corners of the domain of the fluid.
    spray_decay, foam_decay, bubble_decay : float
        The rate at which the lifetime of each kind of foam particle decreases.
    k_foam : float
        The number of foam particles a fluid particle emits per unit time at most.
    foam_density : float
        The density of the fluid.
    """

    def __init__(
        self,
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
        self.particle_radius = particle_radius
        self.time_step = time_step
        self.gravity = gravity
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.spray_decay = spray_decay
        self.foam_decay = foam_decay
        self.bubble_decay = bubble_decay
        self.k_foam = k_foam
        self.particle_mass = 6.4 * particle_radius**3 * foam_density
        self.support_radius = 4.0 * particle_radius

        self.foams_pos = np.zeros((0, 3), dtype=gs.np_float)
        self.foams_vel = np.zeros((0, 3), dtype=gs.np_float)
        self.foams_lifetime = np.zeros((0,), dtype=gs.np_float)

    def generate(self, positions, velocities):
        """Emit new foam particles from the fluid particles, then advect every foam particle over one time step.

        Return the positions of the foam particles.
        """
        h, m, dt = self.support_radius, self.particle_mass, self.time_step

        # Potentials of each fluid particle to trap air, form a wave crest, and its kinetic energy (Eq. 2, 4 and 7)
        n_neighbors, normals, trapped_air, curvature = _compute_emission_potentials(positions, velocities, h, m, gs.EPS)
        speeds = np.linalg.norm(velocities, axis=-1)
        velocities_dir = np.divide(
            velocities, speeds[:, None], out=np.zeros_like(velocities), where=speeds[:, None] > gs.EPS
        )
        is_crest = (velocities_dir * normals).sum(axis=-1) >= 0.6
        kinetic_energy = 0.5 * m * speeds**2
        ke_min = 0.1 * (self.particle_radius**3 * 6400) * 2.5**2
        i_ta = np.clip((trapped_air - 0.05) / (0.5 - 0.05), 0.0, 1.0)
        i_wc = np.clip((is_crest * curvature - 0.05) / (0.5 - 0.05), 0.0, 1.0)
        i_ke = np.clip((kinetic_energy - ke_min) / (49.0 * ke_min), 0.0, 1.0)
        n_emitted = np.floor(10.0 * (i_ta + i_wc) * self.k_foam * i_ke * dt + 0.5).astype(gs.np_int)
        n_emitted[n_neighbors < 5] = 0

        # New foam particles are spread uniformly in a cylinder around their fluid particle, along its velocity
        emitters_i = np.repeat(np.arange(len(positions)), n_emitted)
        emitters_dir = velocities_dir[emitters_i]
        axis_ref = np.where((np.abs(emitters_dir[:, :1]) > 0.999), np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0]))
        e1 = np.cross(emitters_dir, axis_ref)
        e1 /= np.linalg.norm(e1, axis=-1, keepdims=True)
        e2 = np.cross(emitters_dir, e1)
        samples = np.random.rand(len(emitters_i), 4)
        r = self.particle_radius * np.sqrt(samples[:, 0])
        theta = 2.0 * np.pi * samples[:, 1]
        radial = (r * np.cos(theta))[:, None] * e1 + (r * np.sin(theta))[:, None] * e2
        axial = ((samples[:, 2] - 0.5) * dt * speeds[emitters_i])[:, None] * emitters_dir
        foams_pos = np.concatenate((self.foams_pos, positions[emitters_i] + radial + axial))
        foams_vel = np.concatenate((self.foams_vel, velocities[emitters_i] + radial))
        foams_lifetime = np.concatenate((self.foams_lifetime, 2.0 + 3.0 * i_ke[emitters_i] * samples[:, 3]))

        self.foams_pos, self.foams_vel, self.foams_lifetime = _advect_foams(
            positions,
            velocities,
            foams_pos,
            foams_vel,
            foams_lifetime,
            self.gravity,
            self.lower_bound,
            self.upper_bound,
            dt,
            h,
            self.spray_decay,
            self.foam_decay,
            self.bubble_decay,
        )
        return self.foams_pos


def filter_surface(positions, particle_radius):
    """Return whether each particle lies at the surface of the set, which a particle with few neighbors does."""
    neighbors_start, _ = _find_neighbors(positions, positions, 4.0 * particle_radius)
    return np.diff(neighbors_start) < 20

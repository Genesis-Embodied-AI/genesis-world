import math
import os
from pathlib import Path

import numpy as np

import fast_simplification
import trimesh

import genesis as gs
from genesis.ext.isaacgym import terrain_utils as isaacgym_terrain_utils
from genesis.options.morphs import Terrain

from . import serialization
from .cache import get_gnd_path
from .mesh import load_trimesh
from .misc import tensor_to_array


def parse_terrain(morph: Terrain, surface, step_height: float = math.inf):
    """
    Generate mesh (and height field) according to configurations passed by morph.

    A cell of the height field whose heights differ by more than 'step_height', in meters, is made of flat columns (see
    'convert_heightfield_to_watertight_trimesh').

    ------------------------------------------------------------------------------------------------------
    If morph.height_field is not passed, generate the height field by
        n_subterrains    : Tuple[int, int]     = (3, 3)     # number of subterrains in x and y directions
        subterrain_types : Any                 = [
                ['flat_terrain', 'random_uniform_terrain', 'pyramid_sloped_terrain'],
                ['pyramid_sloped_terrain', 'discrete_obstacles_terrain', 'wave_terrain'],
                ['random_uniform_terrain', 'pyramid_stairs_terrain', 'pyramid_sloped_terrain'],
        ]                                                   # types of subterrains in x and y directions
    If morph.height_field is passed, (n_subterrains, subterrain_size, subterrain_types) will be ignored.
    ------------------------------------------------------------------------------------------------------

    Returns
    --------------------------
    vmesh        : Mesh | None, None when the morph disables visualization
    mesh         : Mesh
    height_field : np.ndarray
    """

    gnd_file_path = None
    if morph.name is not None:
        gnd_file_path = Path(
            get_gnd_path(
                morph.name,
                morph.subterrain_types,
                morph.subterrain_size,
                morph.horizontal_scale,
                morph.vertical_scale,
                morph.n_subterrains,
                morph.subterrain_parameters,
            )
        )

    heightfield = morph.height_field
    if heightfield is None and (gnd_file_path is not None and gnd_file_path.exists()):
        gs.logger.debug(f"Pre-genenerated terrain heightmap '{morph.name}' found in cache.")
        try:
            with open(gnd_file_path, "rb") as file:
                heightfield = serialization.load_record(file, np.ndarray)
        except (ValueError, MemoryError):
            # Do not ignore error in case of corrupted cache, to make sure the user is aware of it
            gs.raise_exception(f"Corrupted cache for terrain heightmap: {gnd_file_path}")

    if heightfield is None:
        subterrain_rows = int(morph.subterrain_size[0] / morph.horizontal_scale + gs.EPS) + 1
        subterrain_cols = int(morph.subterrain_size[1] / morph.horizontal_scale + gs.EPS) + 1
        heightfield = np.full(
            (
                morph.n_subterrains[0] * (subterrain_rows - 1) + 1,
                morph.n_subterrains[1] * (subterrain_cols - 1) + 1,
            ),
            fill_value=float("-inf"),
            dtype=gs.np_float,
        )

        for i, j in zip(*map(np.ravel, np.meshgrid(*map(range, morph.n_subterrains), indexing="ij"))):
            subterrain_type = morph.subterrain_types[i][j]
            params = morph.subterrain_params[subterrain_type]

            new_subterrain = isaacgym_terrain_utils.SubTerrain(
                width=subterrain_rows,
                length=subterrain_cols,
                vertical_scale=morph.vertical_scale,
                horizontal_scale=morph.horizontal_scale,
            )
            if not morph.randomize:
                saved_state = np.random.get_state()
                np.random.seed(0)
            if subterrain_type == "flat_terrain":
                subterrain = None
            elif subterrain_type == "fractal_terrain":
                subterrain = fractal_terrain(
                    new_subterrain,
                    levels=params.get("levels", 8),
                    scale=params.get("scale", 5.0),
                )
            elif subterrain_type == "random_uniform_terrain":
                subterrain = isaacgym_terrain_utils.random_uniform_terrain(
                    new_subterrain,
                    min_height=params.get("min_height", -0.1),
                    max_height=params.get("max_height", 0.1),
                    step=params.get("step", 0.1),
                    downsampled_scale=params.get("downsampled_scale", 0.5),
                )
            elif subterrain_type == "sloped_terrain":
                subterrain = isaacgym_terrain_utils.sloped_terrain(
                    new_subterrain,
                    slope=params.get("slope", -0.5),
                )
            elif subterrain_type == "pyramid_sloped_terrain":
                subterrain = isaacgym_terrain_utils.pyramid_sloped_terrain(
                    new_subterrain,
                    slope=params.get("slope", -0.1),
                )
            elif subterrain_type == "discrete_obstacles_terrain":
                subterrain = isaacgym_terrain_utils.discrete_obstacles_terrain(
                    new_subterrain,
                    max_height=params.get("max_height", 0.05),
                    min_size=params.get("min_size", 1.0),
                    max_size=params.get("max_size", 5.0),
                    num_rects=params.get("num_rects", 20),
                )
            elif subterrain_type == "wave_terrain":
                subterrain = isaacgym_terrain_utils.wave_terrain(
                    new_subterrain,
                    num_waves=params.get("num_waves", 2.0),
                    amplitude=params.get("amplitude", 0.1),
                )
            elif subterrain_type == "stairs_terrain":
                subterrain = isaacgym_terrain_utils.stairs_terrain(
                    new_subterrain,
                    step_width=params.get("step_width", 0.75),
                    step_height=params.get("step_height", -0.1),
                )
            elif subterrain_type == "pyramid_stairs_terrain":
                subterrain = isaacgym_terrain_utils.pyramid_stairs_terrain(
                    new_subterrain,
                    step_width=params.get("step_width", 0.75),
                    step_height=params.get("step_height", -0.1),
                )
            elif subterrain_type == "stepping_stones_terrain":
                subterrain = isaacgym_terrain_utils.stepping_stones_terrain(
                    new_subterrain,
                    stone_size=params.get("stone_size", 1.0),
                    stone_distance=params.get("stone_distance", 0.25),
                    max_height=params.get("max_height", 0.2),
                    platform_size=params.get("platform_size", 0.0),
                )
            else:
                gs.raise_exception(f"Unsupported subterrain type: {subterrain_type}")

            if not morph.randomize:
                np.random.set_state(saved_state)

            if subterrain is None:
                data = np.zeros((subterrain_rows, subterrain_cols), dtype=gs.np_float)
            else:
                data = subterrain.height_field_raw

            subterrain_heightfield = heightfield[
                i * (subterrain_rows - 1) : (i + 1) * (subterrain_rows - 1) + 1,
                j * (subterrain_cols - 1) : (j + 1) * (subterrain_cols - 1) + 1,
            ]
            subterrain_heightfield[:] = np.maximum(subterrain_heightfield, data)

        if gnd_file_path is not None:
            os.makedirs(os.path.dirname(gnd_file_path), exist_ok=True)
            with open(gnd_file_path, "wb") as file:
                serialization.export_record(file, heightfield, np.ndarray)

    need_uvs = getattr(surface, "diffuse_texture", None) is not None
    heightfield = heightfield.astype(gs.np_float)
    tmesh, sdf_tmesh = convert_heightfield_to_watertight_trimesh(
        heightfield,
        horizontal_scale=morph.horizontal_scale,
        vertical_scale=morph.vertical_scale,
        surface=surface,
        step_height=step_height,
        uv_scale=morph.uv_scale if need_uvs else None,
    )
    vmesh = None
    if morph.visualization:
        vmesh = gs.Mesh.from_trimesh(mesh=tmesh, surface=surface, metadata={})
    mesh = gs.Mesh.from_trimesh(
        mesh=tmesh,
        surface=gs.surfaces.Collision(),
        metadata={
            "horizontal_scale": morph.horizontal_scale,
            "sdf_mesh": sdf_tmesh,
            "height_field": heightfield,
        },
    )
    return vmesh, mesh, heightfield


def fractal_terrain(terrain, levels=8, scale=1.0):
    """
    Generates a fractal terrain

    Parameters
        terrain (SubTerrain): the terrain
        levels (int, optional): granurarity of the fractal terrain. Defaults to 8.
        scale (float, optional): scales vertical variation. Defaults to 1.0.
    """
    width = terrain.width
    length = terrain.length
    height = np.zeros((width, length), dtype=gs.np_float)
    for level in range(1, levels + 1):
        step = 2 ** (levels - level)
        for y in range(0, width, step):
            y_skip = (1 + y // step) % 2
            for x in range(step * y_skip, length, step * (1 + y_skip)):
                x_skip = (1 + x // step) % 2
                xref = step * (1 - x_skip)
                yref = step * (1 - y_skip)
                mean = height[y - yref : y + yref + 1 : 2 * step, x - xref : x + xref + 1 : 2 * step].mean()
                variation = 2 ** (-level) * np.random.uniform(-1, 1)
                height[y, x] = mean + scale * variation

    height /= terrain.vertical_scale
    terrain.height_field_raw = height
    return terrain


def convert_heightfield_to_watertight_trimesh(
    height_field_raw, horizontal_scale, vertical_scale, surface, step_height=math.inf, uv_scale=None
):
    """
    Convert a height field to the triangle mesh of its surface and to the closed mesh of the solid below it.

    The surface cuts every cell of the grid into two triangles along its diagonal from sample (i + 1, j) to sample
    (i, j + 1), except the cells whose heights differ by more than the step height, which are made of four flat squares,
    each at the height of its nearest corner, joined by vertical walls. The solid closes the surface with vertical sides
    along the edge of the grid and a flat bottom one meter below its lowest point.

    Parameters
    ----------
    height_field_raw : np.ndarray
        The height of every sample of the grid, in units of the vertical scale.
    horizontal_scale : float
        The distance between two neighboring samples, in meters.
    vertical_scale : float
        The height of one unit of the height field, in meters.
    surface : gs.surfaces.Surface
        The surface of the terrain, whose double-sidedness selects the first mesh returned.
    step_height : float, optional
        The difference of height within a cell, in meters, beyond which the cell is made of flat squares. Defaults to
        infinity, which keeps every cell smooth.
    uv_scale : float, optional
        The scale of the texture coordinates spanning the grid. Defaults to None, which gives no texture coordinates.

    Returns
    -------
    surface_mesh : trimesh.Trimesh
        The surface for a single-sided surface, else the solid, losslessly simplified.
    solid_mesh : trimesh.Trimesh
        The solid.
    """
    num_rows, num_cols = height_field_raw.shape
    n_verts_grid = num_rows * num_cols

    x = np.linspace(0, (num_rows - 1) * horizontal_scale, num_rows)
    y = np.linspace(0, (num_cols - 1) * horizontal_scale, num_cols)
    yy, xx = np.meshgrid(y, x)
    zz = height_field_raw * vertical_scale
    verts_grid = np.stack((xx.flat, yy.flat, zz.flat), axis=-1, dtype=np.float32)
    z_bottom = verts_grid[:, 2].min() - 1.0

    # The corners 00, 10, 01, 11 of every cell, then its kind. A step cell whose heights change along the rows alone, or
    # along the columns alone, is made of two strips across the other axis, and of four columns otherwise (see
    # func_contact_mpr_terrain in narrowphase.py). The heights are compared in the precision of the height field.
    z_00, z_10, z_01, z_11 = zz[:-1, :-1], zz[1:, :-1], zz[:-1, 1:], zz[1:, 1:]
    cells_corner_z = np.stack((z_00, z_10, z_01, z_11))
    is_step = cells_corner_z.max(axis=0) - cells_corner_z.min(axis=0) > zz.dtype.type(step_height)
    is_strips_rows = is_step & (np.abs(z_00 - z_01) <= gs.EPS) & (np.abs(z_10 - z_11) <= gs.EPS)
    is_strips_cols = is_step & ~is_strips_rows & (np.abs(z_00 - z_10) <= gs.EPS) & (np.abs(z_01 - z_11) <= gs.EPS)
    is_columns = is_step & ~is_strips_rows & ~is_strips_cols

    # The two triangles of every cell, in the order of the walk of the collision detection
    cells_grid_idx = np.arange(n_verts_grid).reshape((num_rows, num_cols))[:-1, :-1]
    grid_idx_00, grid_idx_10 = cells_grid_idx, cells_grid_idx + num_cols
    grid_idx_01, grid_idx_11 = cells_grid_idx + 1, cells_grid_idx + num_cols + 1
    cells_faces = np.stack(
        (
            np.stack((grid_idx_00, grid_idx_10, grid_idx_01), axis=-1),
            np.stack((grid_idx_01, grid_idx_10, grid_idx_11), axis=-1),
        ),
        axis=2,
    )

    # A step cell changes height halfway along its edges that cross the axis its heights change along, along every edge
    # for columns, where every face meeting the edge gets a vertex. The edges along the rows join sample (i, j) to sample
    # (i + 1, j), and those along the columns join sample (i, j) to sample (i, j + 1).
    is_cells_mid_rows = is_columns | is_strips_rows
    is_cells_mid_cols = is_columns | is_strips_cols
    has_mid_rows = np.zeros((num_rows - 1, num_cols), dtype=bool)
    has_mid_rows[:, :-1] |= is_cells_mid_rows
    has_mid_rows[:, 1:] |= is_cells_mid_rows
    has_mid_cols = np.zeros((num_rows, num_cols - 1), dtype=bool)
    has_mid_cols[:-1] |= is_cells_mid_cols
    has_mid_cols[1:] |= is_cells_mid_cols

    # The points beyond the samples are identified by their coordinates on the grid of half cells and their height, so
    # that the faces sharing a point are welded at the end. A smooth cell splits its triangles at the midpoints of its
    # edges. The edges of a cell are named after their side, (s)outh at column j, (w)est at row i, (e)ast at row i + 1
    # and (n)orth at column j + 1, and their midpoints are interpolated once, so that both sides read the same height.
    zz_mid_rows = 0.5 * (zz[:-1] + zz[1:])
    zz_mid_cols = 0.5 * (zz[:, :-1] + zz[:, 1:])
    i_cell, j_cell = np.meshgrid(np.arange(num_rows - 1), np.arange(num_cols - 1), indexing="ij")
    cells_points = {
        name: np.stack(point, axis=-1, dtype=np.float64)
        for name, point in (
            ("00", (2 * i_cell, 2 * j_cell, z_00)),
            ("10", (2 * i_cell + 2, 2 * j_cell, z_10)),
            ("01", (2 * i_cell, 2 * j_cell + 2, z_01)),
            ("11", (2 * i_cell + 2, 2 * j_cell + 2, z_11)),
            ("s", (2 * i_cell + 1, 2 * j_cell, zz_mid_rows[:, :-1])),
            ("w", (2 * i_cell, 2 * j_cell + 1, zz_mid_cols[:-1])),
            ("e", (2 * i_cell + 2, 2 * j_cell + 1, zz_mid_cols[1:])),
            ("n", (2 * i_cell + 1, 2 * j_cell + 2, zz_mid_rows[:, 1:])),
        )
    }
    cells_has_mid = {"s": has_mid_rows[:, :-1], "w": has_mid_cols[:-1], "e": has_mid_cols[1:], "n": has_mid_rows[:, 1:]}
    is_smooth_padded = np.pad(~is_step, 1)
    cells_has_smooth_nb = {
        "s": is_smooth_padded[1:-1, :-2],
        "w": is_smooth_padded[:-2, 1:-1],
        "e": is_smooth_padded[2:, 1:-1],
        "n": is_smooth_padded[1:-1, 2:],
    }
    is_face_plain = np.stack(
        (
            ~is_step & ~cells_has_mid["s"] & ~cells_has_mid["w"],
            ~is_step & ~cells_has_mid["e"] & ~cells_has_mid["n"],
        ),
        axis=-1,
    )

    # Every face of the surface beyond the plain triangles goes with the direction it faces: up for a top face, towards
    # the lower side for a vertical face. The pieces of a split triangle come first, for each combination of split edges.
    faces_points, faces_dir = [], []
    for is_split_0, is_split_1, faces in (
        ("s", "w", (("00", "s", "01"), ("s", "10", "01"))),
        ("w", "s", (("00", "10", "w"), ("w", "10", "01"))),
        (("s", "w"), None, (("00", "s", "w"), ("s", "10", "01"), ("s", "01", "w"))),
        ("e", "n", (("01", "10", "e"), ("01", "e", "11"))),
        ("n", "e", (("01", "10", "n"), ("10", "11", "n"))),
        (("e", "n"), None, (("01", "10", "e"), ("01", "e", "n"), ("e", "11", "n"))),
    ):
        if is_split_1 is None:
            is_split = ~is_step & cells_has_mid[is_split_0[0]] & cells_has_mid[is_split_0[1]]
        else:
            is_split = ~is_step & cells_has_mid[is_split_0] & ~cells_has_mid[is_split_1]
        for face in faces:
            faces_points.append(np.stack([cells_points[name][is_split] for name in face], axis=1))
            faces_dir.append(np.tile((0.0, 0.0, 1.0), (len(faces_points[-1]), 1)))

    # The walls, each between two vertical edges given as chains of four points from bottom to top, a missing point
    # repeating the one before it
    walls_chain_0, walls_chain_1, walls_dir = [], [], []

    # The four columns of a step cell, the column of corner (a, b) spanning the half cells from row 2 * i + a and column
    # 2 * j + b, and the walls between two columns of different heights, from the center of the cell to the midpoint of
    # its edge. A wall is split wherever another face ends on it: at the center, at the heights of the other two
    # columns, and at the midpoint, at the height of a smooth cell beyond the edge.
    i_col, j_col = np.nonzero(is_columns)
    cols_corner_z = cells_corner_z[:, i_col, j_col]
    for a, b in ((0, 0), (1, 0), (0, 1), (1, 1)):
        z = cols_corner_z[a + 2 * b]
        square = [
            np.stack((2 * i_col + a + da, 2 * j_col + b + db, z), axis=-1, dtype=np.float64)
            for da, db in ((0, 0), (1, 0), (0, 1), (1, 1))
        ]
        faces_points += [np.stack(square[:3], axis=1), np.stack((square[2], square[1], square[3]), axis=1)]
        faces_dir += [np.tile((0.0, 0.0, 1.0), (len(z), 1))] * 2
    for i_axis, k in ((0, 0), (0, 1), (1, 0), (1, 1)):
        # The wall across axis i_axis between the columns of corners 0 and 1 along it, at corner k along the other axis
        if i_axis == 0:
            corners_wall, corners_other = (2 * k, 2 * k + 1), [2 - 2 * k, 3 - 2 * k]
            edge, z_mid = "sn"[k], zz_mid_rows[i_col, j_col + k]
        else:
            corners_wall, corners_other = (k, k + 2), [1 - k, 3 - k]
            edge, z_mid = "we"[k], zz_mid_cols[i_col + k, j_col]
        z_0, z_1 = cols_corner_z[corners_wall[0]], cols_corner_z[corners_wall[1]]
        z_lo, z_hi = np.minimum(z_0, z_1), np.maximum(z_0, z_1)
        z_other_lo, z_other_hi = np.sort(cols_corner_z[corners_other], axis=0)
        z_center_1 = np.where((z_lo < z_other_lo) & (z_other_lo < z_hi), z_other_lo, z_lo)
        z_center_2 = np.where((z_lo < z_other_hi) & (z_other_hi < z_hi), z_other_hi, z_center_1)
        z_edge_1 = np.where(cells_has_smooth_nb[edge][i_col, j_col], z_mid, z_lo)
        u = 2 * (i_col, j_col)[i_axis] + 1
        v_center = 2 * (j_col, i_col)[i_axis] + 1
        v_edge = v_center + 2 * k - 1
        for chains, v, zs in (
            (walls_chain_0, v_center, (z_lo, z_center_1, z_center_2, z_hi)),
            (walls_chain_1, v_edge, (z_lo, z_edge_1, z_hi, z_hi)),
        ):
            chain = np.stack([np.stack((u, v, z) if i_axis == 0 else (v, u, z), axis=-1) for z in zs], axis=1)
            chains.append(chain.astype(np.float64))
        wall_dir = np.zeros((len(z_0), 3))
        wall_dir[:, i_axis] = np.where(z_0 < z_1, -1.0, 1.0)
        walls_dir.append(wall_dir)

    # The two strips of a step cell whose heights change along one axis alone span the cell across the other axis, each
    # at the height of its side, with the wall between them along the middle of the cell. A strip gets a vertex at the
    # midpoint of its outer edge when a column beyond it changes height there, and the wall is split at either end where
    # a smooth cell meets it. The strips along the columns are those along the rows of the transposed height field.
    for i_axis, (zz_axis, zz_mid_axis, is_strips_axis, has_mid_outer, is_smooth_axis) in enumerate(
        (
            (zz, zz_mid_rows, is_strips_rows, has_mid_cols, is_smooth_padded),
            (zz.T, zz_mid_cols.T, is_strips_cols.T, has_mid_rows.T, is_smooth_padded.T),
        )
    ):
        i_s, j_s = np.nonzero(is_strips_axis)
        for a in (0, 1):
            z_lo_end, z_hi_end = zz_axis[i_s + a, j_s], zz_axis[i_s + a, j_s + 1]
            strip = [
                (2 * i_s + 1, 2 * j_s, z_lo_end),
                (2 * i_s + 1, 2 * j_s + 2, z_hi_end),
                (2 * i_s + 2 * a, 2 * j_s + 2, z_hi_end),
                (2 * i_s + 2 * a, 2 * j_s + 1, z_lo_end),
                (2 * i_s + 2 * a, 2 * j_s, z_lo_end),
            ]
            # A missing outer vertex repeats the corner before it
            strip[3] = tuple(np.where(has_mid_outer[i_s + a, j_s], p, q) for p, q in zip(strip[3], strip[2]))
            strip = [np.stack(p if i_axis == 0 else (p[1], p[0], p[2]), axis=-1).astype(np.float64) for p in strip]
            for i_p in range(1, 4):
                faces_points.append(np.stack((strip[0], strip[i_p], strip[i_p + 1]), axis=1))
                faces_dir.append(np.tile((0.0, 0.0, 1.0), (len(i_s), 1)))
        for chains, j_end, has_smooth_end in (
            (walls_chain_0, j_s, is_smooth_axis[i_s + 1, j_s]),
            (walls_chain_1, j_s + 1, is_smooth_axis[i_s + 1, j_s + 2]),
        ):
            z_0, z_1 = zz_axis[i_s, j_end], zz_axis[i_s + 1, j_end]
            z_lo, z_hi = np.minimum(z_0, z_1), np.maximum(z_0, z_1)
            z_mid = np.where(has_smooth_end, zz_mid_axis[i_s, j_end], z_lo)
            chain = [(2 * i_s + 1, 2 * j_end, z) for z in (z_lo, z_mid, z_hi, z_hi)]
            chain = [np.stack(p if i_axis == 0 else (p[1], p[0], p[2]), axis=-1) for p in chain]
            chains.append(np.stack(chain, axis=1).astype(np.float64))
        wall_dir = np.zeros((len(i_s), 3))
        wall_dir[:, i_axis] = np.where(zz_axis[i_s, j_s] < zz_axis[i_s + 1, j_s], -1.0, 1.0)
        walls_dir.append(wall_dir)

    # Every wall joins the bottom of both chains, climbs the first chain then the second one. Its degenerate faces drop
    # out with the repeated points.
    walls_chain_0, walls_chain_1 = np.concatenate(walls_chain_0), np.concatenate(walls_chain_1)
    walls_dir = np.concatenate(walls_dir)
    is_wall = walls_chain_0[:, 0, 2] < walls_chain_0[:, 3, 2]
    walls_points = np.concatenate((walls_chain_0, walls_chain_1), axis=1)[is_wall]
    for face in ((0, 4, 1), (1, 4, 2), (2, 4, 3), (3, 4, 5), (3, 5, 6), (3, 6, 7)):
        faces_points.append(walls_points[:, face])
        faces_dir.append(walls_dir[is_wall])

    # The vertical triangles closing the gap between the flat halves of the edge of a step cell and the midpoint of the
    # edge, where a smooth cell meets it
    for edge, corners, (di, dj) in (
        ("s", ("00", "10"), (0, -1)),
        ("w", ("00", "01"), (-1, 0)),
        ("e", ("10", "11"), (1, 0)),
        ("n", ("01", "11"), (0, 1)),
    ):
        is_gap = is_step & cells_has_mid[edge] & cells_has_smooth_nb[edge]
        mid = cells_points[edge][is_gap]
        for corner_name in corners:
            corner = cells_points[corner_name][is_gap]
            mid_flat = np.concatenate((mid[:, :2], corner[:, 2:]), axis=-1)
            gap_dir = np.zeros((len(corner), 3))
            gap_dir[:, :2] = np.where(mid[:, 2:] > corner[:, 2:], -1.0, 1.0) * (di, dj)
            faces_points.append(np.stack((corner, mid_flat, mid), axis=1))
            faces_dir.append(gap_dir)

    faces_points = np.concatenate(faces_points)
    faces_dir = np.concatenate(faces_dir)
    faces_normal = np.cross(faces_points[:, 1] - faces_points[:, 0], faces_points[:, 2] - faces_points[:, 0])
    is_flipped = (faces_normal * faces_dir).sum(axis=-1) < 0.0
    faces_points[is_flipped] = faces_points[is_flipped][:, ::-1]

    # The profile of the edge of the grid, counterclockwise from sample (0, 0), where an edge with a midpoint adds it
    # twice, at the height of either sample
    border_points = []
    for i_side, j_side, has_mid_side in (
        (np.arange(num_rows), np.zeros(num_rows, dtype=np.int64), has_mid_rows[:, 0]),
        (np.full(num_cols, num_rows - 1), np.arange(num_cols), has_mid_cols[-1]),
        (np.arange(num_rows)[::-1], np.full(num_rows, num_cols - 1), has_mid_rows[::-1, -1]),
        (np.zeros(num_cols, dtype=np.int64), np.arange(num_cols)[::-1], has_mid_cols[0, ::-1]),
    ):
        z_side = zz[i_side, j_side]
        i_mid, j_mid = i_side[:-1] + i_side[1:], j_side[:-1] + j_side[1:]
        side_points = np.stack(
            (
                np.stack((2 * i_side[:-1], 2 * j_side[:-1], z_side[:-1]), axis=-1),
                np.stack((i_mid, j_mid, z_side[:-1]), axis=-1),
                np.stack((i_mid, j_mid, z_side[1:]), axis=-1),
            ),
            axis=1,
            dtype=np.float64,
        )
        border_points.append(side_points[np.stack((np.ones_like(has_mid_side), has_mid_side, has_mid_side), axis=-1)])
    border_points = np.concatenate(border_points)
    points_next = np.roll(border_points, -1, axis=0)
    is_jump = (border_points[:, :2] == points_next[:, :2]).all(axis=-1)

    # The sides of the solid, a vertical strip under every segment of the profile facing outward. The lower point of a
    # jump of the profile lies on the vertical edge of the strip on its higher side, and splits it.
    point_p, point_q = border_points[~is_jump], points_next[~is_jump]
    point_p_bottom = np.concatenate((point_p[:, :2], np.full((len(point_p), 1), z_bottom)), axis=-1)
    point_q_bottom = np.concatenate((point_q[:, :2], np.full((len(point_q), 1), z_bottom)), axis=-1)
    point_before, point_after = (
        np.roll(border_points, 1, axis=0)[~is_jump],
        np.roll(border_points, -2, axis=0)[~is_jump],
    )
    is_split_p = np.roll(is_jump, 1)[~is_jump] & (point_before[:, 2] < point_p[:, 2])
    is_split_q = np.roll(is_jump, -1)[~is_jump] & (point_after[:, 2] < point_q[:, 2])
    is_plain = ~is_split_p & ~is_split_q
    closure_points = [
        np.stack((point_p, point_p_bottom, point_q_bottom), axis=1)[is_plain],
        np.stack((point_p, point_q_bottom, point_q), axis=1)[is_plain],
        np.stack((point_q, point_p, point_before), axis=1)[is_split_p],
        np.stack((point_q, point_before, point_p_bottom), axis=1)[is_split_p],
        np.stack((point_q, point_p_bottom, point_q_bottom), axis=1)[is_split_p],
        np.stack((point_p, point_p_bottom, point_q_bottom), axis=1)[is_split_q],
        np.stack((point_p, point_q_bottom, point_after), axis=1)[is_split_q],
        np.stack((point_p, point_after, point_q), axis=1)[is_split_q],
    ]
    # The bottom of the solid, a fan around its center facing down
    center_bottom = np.array([[num_rows - 1, num_cols - 1, z_bottom]], dtype=np.float64)
    closure_points.append(
        np.stack(
            (
                np.repeat(center_bottom, len(point_p), axis=0),
                np.roll(point_p_bottom, -1, axis=0),
                point_p_bottom,
            ),
            axis=1,
        )
    )
    closure_points = np.concatenate(closure_points)

    # Weld the points of the faces: a sample keeps its index on the grid, and every other point of the surface comes
    # after the grid, before the points of the solid alone. The points are grouped by their coordinates on the grid of
    # half cells, then by the bits of their height, a negative zero turned positive.
    points_all = np.concatenate((faces_points, closure_points)).reshape((-1, 3))
    lattice_key = (points_all[:, 0] * (2 * num_cols) + points_all[:, 1]).astype(np.int64)
    height_key = (points_all[:, 2] + 0.0).view(np.int64)
    order = np.lexsort((height_key, lattice_key))
    is_first = np.ones(len(order), dtype=bool)
    is_first[1:] = (lattice_key[order[1:]] != lattice_key[order[:-1]]) | (
        height_key[order[1:]] != height_key[order[:-1]]
    )
    points = points_all[order[is_first]]
    points_idx = np.empty(len(order), dtype=np.int64)
    points_idx[order] = np.cumsum(is_first) - 1
    is_sample = (points[:, 0] % 2 == 0) & (points[:, 1] % 2 == 0) & (points[:, 2] > z_bottom)
    is_surface = np.zeros(len(points), dtype=bool)
    is_surface[points_idx[: 3 * len(faces_points)]] = True
    verts_idx = np.empty(len(points), dtype=np.int64)
    verts_idx[is_sample] = (points[is_sample, 0] // 2 * num_cols + points[is_sample, 1] // 2).astype(np.int64)
    is_surface_extra, is_solid_extra = ~is_sample & is_surface, ~is_sample & ~is_surface
    n_surface_extra = np.count_nonzero(is_surface_extra)
    verts_idx[is_surface_extra] = n_verts_grid + np.arange(n_surface_extra)
    verts_idx[is_solid_extra] = n_verts_grid + n_surface_extra + np.arange(np.count_nonzero(is_solid_extra))
    faces = verts_idx[points_idx].reshape((-1, 3))
    is_degenerate = (faces[:, 0] == faces[:, 1]) | (faces[:, 1] == faces[:, 2]) | (faces[:, 2] == faces[:, 0])
    faces_surface = np.concatenate(
        (cells_faces[is_face_plain], faces[: len(faces_points)][~is_degenerate[: len(faces_points)]])
    )
    faces_closure = faces[len(faces_points) :][~is_degenerate[len(faces_points) :]]

    # The coordinates of the points off the grid, interpolated between those of the samples
    x_half = np.empty(2 * num_rows - 1)
    x_half[::2], x_half[1::2] = x, 0.5 * (x[:-1] + x[1:])
    y_half = np.empty(2 * num_cols - 1)
    y_half[::2], y_half[1::2] = y, 0.5 * (y[:-1] + y[1:])
    points_extra = points[np.concatenate((np.flatnonzero(is_surface_extra), np.flatnonzero(is_solid_extra)))]
    verts_extra = np.stack(
        (
            np.interp(points_extra[:, 0], np.arange(len(x_half)), x_half),
            np.interp(points_extra[:, 1], np.arange(len(y_half)), y_half),
            points_extra[:, 2],
        ),
        axis=-1,
        dtype=np.float32,
    )
    verts = np.concatenate((verts_grid, verts_extra))
    verts_surface = verts[: n_verts_grid + n_surface_extra]

    uvs = None
    if uv_scale is not None:
        uvs = np.stack(
            (
                (verts[:, 0] - xx.min()) / (xx.max() - xx.min()) * uv_scale,
                (verts[:, 1] - yy.min()) / (yy.max() - yy.min()) * uv_scale,
            ),
            axis=-1,
            dtype=np.float32,
        )

    if not surface.double_sided:
        visual = None
        if uvs is not None:
            visual = trimesh.visual.TextureVisuals(uv=uvs[: len(verts_surface)])

        # The normals are computed here and handed over, the way trimesh computes them, because trimesh would
        # otherwise keep every intermediate of their computation cached on the surface, several times its size.
        triangles = verts_surface.astype(np.float64)[faces_surface]
        faces_normal_valid, is_face_valid = trimesh.triangles.normals(triangles)
        faces_normal = np.zeros((len(faces_surface), 3))
        faces_normal[is_face_valid] = faces_normal_valid
        verts_normal = trimesh.geometry.weighted_vertex_normals(
            len(verts_surface), faces_surface, faces_normal, trimesh.triangles.angles(triangles)
        )
        vmesh_single = trimesh.Trimesh(
            verts_surface, faces_surface, vertex_normals=verts_normal, visual=visual, process=False
        )

    visual = None if uvs is None else trimesh.visual.TextureVisuals(uv=uvs)
    sdf_mesh = trimesh.Trimesh(verts, np.concatenate((faces_surface, faces_closure)), process=False, visual=visual)

    # This is the mesh used for non-sdf purposes. A single-sided surface uses the top face alone. A double-sided surface
    # uses the full mesh, losslessly simplified to save memory cost for storing verts and faces.
    if not surface.double_sided:
        return vmesh_single, sdf_mesh

    v_simp, f_simp = fast_simplification.simplify(sdf_mesh.vertices, sdf_mesh.faces, target_count=0, lossless=True)

    if uvs is not None:
        idx_map = np.empty(len(v_simp), dtype=np.int64)
        for i, v in enumerate(v_simp):
            dists = np.square(verts - v).sum(axis=1)
            idx_map[i] = np.argmin(dists)

        uv_simp = uvs[idx_map]
        visual = trimesh.visual.TextureVisuals(uv=uv_simp)
    vmesh_full = trimesh.Trimesh(v_simp, f_simp, visual=visual)

    return vmesh_full, sdf_mesh


def mesh_to_heightfield(
    path: str,
    spacing: float | tuple[float, float],
    oversample: int = 1,
    *,
    up_axis: str = "z",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Parameters
    ----------
    path : str
        .glb / .obj / … file containing the terrain mesh.
    spacing : float  |  (float, float)
        Desired grid spacing  Δx  (and  Δy).  Units must match the mesh.
    oversample : int, default 3
        Number of extra rays per coarse‑grid axis.  k = 1 reproduces the
        simple single‑ray method; k ≥ 2 captures peaks inside a cell.
    up_axis : {"z", "y"}, default "z"
        Mesh’s up direction.  If "y" (common in glTF), the function
        rotates the mesh so Z is up before sampling.

    Returns
    -------
    heights : (nx, ny)  float32 ndarray
        Highest elevation per coarse cell (NaN if no hit).
    xs      : (nx,)     float32 ndarray
    ys      : (ny,)     float32 ndarray
        Coordinates of grid lines (cell centres).

    Notes
    -----
    • Memory cost grows as  oversample².
    """
    if np.isscalar(spacing):
        spacing = (spacing, spacing)

    mesh = load_trimesh(path, force="mesh")

    # -------------------------------- axis handling ---------------------------
    if up_axis.lower() == "y":
        # rotate so Z becomes up (‑90° around X)
        T = trimesh.transformations.rotation_matrix(np.deg2rad(-90), [1, 0, 0])
        mesh.apply_transform(T)

    (minx, miny, _), (maxx, maxy, maxz) = mesh.bounds

    # -------------------------------- coarse grid ----------------------------
    dx, dy = spacing
    nx = math.ceil((maxx - minx) / dx) + 1
    ny = math.ceil((maxy - miny) / dy) + 1

    xs = np.linspace(minx, maxx, nx, dtype=np.float32)
    ys = np.linspace(miny, maxy, ny, dtype=np.float32)

    # -------------------------------- fine grid ------------------------------
    fx = nx * oversample
    fy = ny * oversample
    fx_lin = np.linspace(minx, maxx, fx, dtype=np.float32)
    fy_lin = np.linspace(miny, maxy, fy, dtype=np.float32)
    fxx, fyy = np.meshgrid(fx_lin, fy_lin, indexing="ij")

    origins = np.stack((fxx.ravel(), fyy.ravel(), np.full(fxx.size, maxz + 1.0)), axis=-1)
    directions = np.tile([0.0, 0.0, -1.0], (origins.shape[0], 1))

    # -------------------------------- ray cast -------------------------------
    locs, hit_ids, _ = mesh.ray.intersects_location(ray_origins=origins, ray_directions=directions, multiple_hits=False)

    h_fine = np.full(fxx.size, np.nan, dtype=np.float32)
    h_fine[hit_ids] = locs[:, 2]
    h_fine = h_fine.reshape((fx, fy))  # (fx, fy) = (nx*k, ny*k)

    # -------------------------------- down‑sample ----------------------------
    # reshape to (nx, k, ny, k) then take max over the 2 fine axes
    h_coarse = np.nanmax(h_fine.reshape(nx, oversample, ny, oversample).swapaxes(1, 2), axis=(2, 3))

    # change nan to min
    minz = mesh.bounds[0][2]
    h_coarse = np.where(np.isnan(h_coarse), minz, h_coarse)

    return h_coarse, xs, ys

import json
import tempfile
import webbrowser
from pathlib import Path
from typing import NamedTuple

import igl
import numpy as np

from genesis.utils.misc import tensor_to_array


class Crossing(NamedTuple):
    """A genuinely-interpenetrating pair of links among those passed to `get_genuine_interpenetration`."""

    link_a: int
    link_b: int
    depth: float  # separating translation for a crossing, deepest incursion for jammed/contact pairs (metres)


def get_genuine_interpenetration(links, cross_tol=1e-3, n_dir=40, n_bisect=9, is_exact=True):
    """Measure the deepest genuine interpenetration over all link pairs among `links` (each a list of
    `(verts, faces)` collision geoms), as the penetration-depth ground truth a collision algorithm is
    expected to resolve.

    Pairs overlapping within `cross_tol` are contacts reporting their incursion; a crossing pair reads, by
    structure:

    - FULL CONTAINMENT (every vert of one link inside the other): the separation - a swallowed body has no
      free surface to heal toward, so only the extraction resolves (an engulfed sphere reads R + r).
    - PARTIAL ENCLOSURE (a quarter or more of one link's verts buried): min(burial incursion, separation) -
      an oversized apple jammed in a cup reads the wall burial, and a hyper-inflated one enclosing the whole
      cup reads the cheaper translation that slides it off.
    - DENT (the breach touches a single face of each wall): the separation when the breach normal is
      coherent (a directional press retracts by its press depth - the unsigned incursion would escape
      through the far face of a thin wall), min(incursion, separation) when it cancels (a sphere seated all
      around a donut hole reads its 2 mm seat; unseating it is no resolution).
    - PIERCE (some wall carries penetrating verts on both of its faces): the separation, which retracts the
      press or backs the rod out. When separating costs an order of magnitude more than the incursion - a
      bore-fit body poking through its container's wall, where the separation is extraction-scale - the
      resolution is instead the push-back heal: the material protrusion past the pierced wall, marched from
      the antipodal-pair verts along the breach normal (the mean normal of the intruder's penetrating verts,
      coherent only for one-faced breaches), floored by the incursion and capped by the separation.

    Separation means zero material overlap, not unlinked: chain-linked donuts pressed together separate by
    the press depth. Insideness is vertex-sampled via generalized winding numbers (exact on the overlapping
    closed components of convex decompositions, where pseudonormal signed distances break), so each mesh
    must be tessellated finer than its partner's thinnest dimension. Searches run `n_dir` Fibonacci
    directions, bracketed on a geometric ladder and bisected `n_bisect` times.

    Returns `(max_depth, crossings)`: `max_depth` is the largest depth over ALL overlapping pairs, and
    `crossings` lists the pairs deeper than `cross_tol`, deepest first.

    `is_exact` gates the insideness of the ground-truth `overlap` on the exact `igl.winding_number`; set it
    False to use the faster tree approximation, whose platform-dependent error can shift `overlap` on
    borderline geometry (see `inside_of`).
    """
    # Geoms may arrive as torch tensors (verts) or numpy arrays (faces); igl needs float64 verts and int64 faces.
    links = [
        [(tensor_to_array(verts, dtype=np.float64), tensor_to_array(faces, dtype=np.int64)) for verts, faces in geoms]
        for geoms in links
    ]

    # Broad-phase AABB per link, from the full-resolution geoms. A link with no collision geom (e.g. a free-joint
    # base link) gets an inverted AABB so it is rejected against every other link, keeping the link indexing aligned
    # with the caller's list so the returned crossings stay valid indices.
    links_aabb = [
        (
            (np.concatenate([verts for verts, _ in geoms]).min(0), np.concatenate([verts for verts, _ in geoms]).max(0))
            if geoms
            else (np.full(3, np.inf), np.full(3, -np.inf))
        )
        for geoms in links
    ]

    # One concatenated full-resolution mesh per link.
    merged = []
    for geoms in links:
        verts_all, faces_all, offset = [], [], 0
        for verts, faces in geoms:
            verts_all.append(verts)
            faces_all.append(faces + offset)
            offset += len(verts)
        merged.append(
            (np.concatenate(verts_all), np.concatenate(faces_all))
            if verts_all
            else (np.empty((0, 3)), np.empty((0, 3), dtype=np.int64))
        )

    def inside_of(points, verts_other, faces_other, lo_other, hi_other, is_exact=False):
        # Winding-number insideness with an AABB prefilter: a point outside the partner's box can't be inside.
        # `igl.fast_winding_number` is a tree approximation whose error is platform-dependent (BLAS order),
        # not float noise, so it can flip a vertex's insideness across platforms. `is_exact` uses the exact
        # `igl.winding_number` for the one-shot calls that set `overlap`'s ground truth; the hot `separated_at`
        # search keeps the fast one, where a wrong result only costs an extra bisection step.
        is_inside = np.zeros(len(points), dtype=bool)
        is_in_box = ((points >= lo_other) & (points <= hi_other)).all(1)
        if is_in_box.any():
            winding_number = igl.winding_number if is_exact else igl.fast_winding_number
            is_inside[is_in_box] = np.abs(winding_number(verts_other, faces_other, points[is_in_box])) > 0.5
        return is_inside

    # Fibonacci direction sphere and the shift grid (each direction times each probe distance).
    golden = 0.5 * (1.0 + 5.0**0.5)
    i_dir = np.arange(n_dir)
    z_dir = 1.0 - 2.0 * (i_dir + 0.5) / n_dir
    r_dir = np.sqrt(1.0 - z_dir * z_dir)
    azimuth = 2.0 * np.pi * i_dir / golden
    dirs = np.stack([r_dir * np.cos(azimuth), r_dir * np.sin(azimuth), z_dir], axis=1)
    probe = np.array([1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 4e-2, 8e-2, 1.2e-1])

    max_depth = 0.0
    crossings = []
    for i_la, geoms_a in enumerate(links):
        lo_a, hi_a = links_aabb[i_la]
        for i_lb in range(i_la + 1, len(links)):
            # Broad-phase reject via AABB.
            lo_b, hi_b = links_aabb[i_lb]
            if (lo_a > hi_b).any() or (lo_b > hi_a).any():
                continue
            va, fa = merged[i_la]
            vb, fb = merged[i_lb]

            # Incursion magnitudes from unsigned distances gated by winding-number insideness: the
            # pseudonormal sign of igl.signed_distance is unreliable on the overlapping closed components of
            # convex decompositions, while the generalized winding number stays exact.
            is_inside_a0 = inside_of(va, vb, fb, lo_b, hi_b, is_exact=is_exact)
            is_inside_b0 = inside_of(vb, va, fa, lo_a, hi_a, is_exact=is_exact)
            dist_a0, faces_near_a = igl.signed_distance(va, vb, fb)[:2]
            dist_b0, faces_near_b = igl.signed_distance(vb, va, fa)[:2]
            dist_a0 = np.abs(dist_a0)
            dist_b0 = np.abs(dist_b0)
            depth_a0 = np.where(is_inside_a0, dist_a0, 0.0)
            depth_b0 = np.where(is_inside_b0, dist_b0, 0.0)
            overlap = max(depth_a0.max(), depth_b0.max())
            if overlap <= cross_tol:
                # Contact (or cavity containment touching lightly): report the overlap directly.
                max_depth = max(max_depth, overlap)
                continue

            # Dent-vs-pierce classification: a pierced wall carries penetrating verts on BOTH of its
            # faces - a close pair with near-opposite outward normals whose offset is aligned with them
            # (opposite-normal verts offset sideways sit across a breach rim or a hole, not through a
            # wall). Pair distance is capped: beyond ~12 mm two opposite-normal verts belong to opposite
            # sides of a body, not one wall. Near-touching verts count as penetrating here, since a
            # crossing exactly as deep as the wall leaves the far-face verts ON the partner's surface.
            r_pair = min(3.0 * overlap, 12e-3) + 1e-3

            def antipodal_pairs(pen_pts, pen_normals):
                pairs = [np.empty((0, 2), dtype=np.int64)]
                for i_lo in range(0, len(pen_pts), 256):
                    chunk = slice(i_lo, i_lo + 256)
                    diff = pen_pts[None] - pen_pts[chunk, None]
                    dist2 = (diff**2).sum(-1)
                    dots = pen_normals[chunk] @ pen_normals.T
                    is_aligned = np.abs(np.einsum("knj,kj->kn", diff, pen_normals[chunk])) > 0.7 * np.sqrt(dist2)
                    i_v, j_v = np.nonzero((dist2 <= r_pair**2) & (dots < -0.5) & is_aligned)
                    pairs.append(np.stack([i_v + i_lo, j_v], axis=1))
                return np.concatenate(pairs)

            normals_a = igl.per_vertex_normals(va, fa)
            normals_b = igl.per_vertex_normals(vb, fb)
            normals_faces_a = np.cross(va[fa[:, 1]] - va[fa[:, 0]], va[fa[:, 2]] - va[fa[:, 0]])
            normals_faces_a /= np.maximum(np.linalg.norm(normals_faces_a, axis=1)[:, None], 1e-30)
            normals_faces_b = np.cross(vb[fb[:, 1]] - vb[fb[:, 0]], vb[fb[:, 2]] - vb[fb[:, 0]])
            normals_faces_b /= np.maximum(np.linalg.norm(normals_faces_b, axis=1)[:, None], 1e-30)
            is_pen_a = is_inside_a0 | (dist_a0 <= 0.6e-3)
            is_pen_b = is_inside_b0 | (dist_b0 <= 0.6e-3)
            pairs_a = antipodal_pairs(va[is_pen_a], normals_a[is_pen_a])
            pairs_b = antipodal_pairs(vb[is_pen_b], normals_b[is_pen_b])
            has_pierce = bool(len(pairs_a)) or bool(len(pairs_b))

            def separated_at(offsets):
                # A shift separates the pair when no vert of one link samples inside the other. Deeply
                # overlapping shifts are decided by a strided 1/8 vert subsample first (any hit proves
                # overlap); only the undecided shifts pay the full queries, so the result is exact.
                offsets = np.atleast_2d(offsets)
                is_separated = np.ones(len(offsets), dtype=bool)
                for verts_query, sign, verts_other, faces_other, lo_other, hi_other in (
                    (va[::8], 1.0, vb, fb, lo_b, hi_b),
                    (vb[::8], -1.0, va, fa, lo_a, hi_a),
                    (va, 1.0, vb, fb, lo_b, hi_b),
                    (vb, -1.0, va, fa, lo_a, hi_a),
                ):
                    idx = np.flatnonzero(is_separated)
                    if not len(idx):
                        break
                    points = (verts_query[None] + sign * offsets[idx, None]).reshape(-1, 3)
                    is_inside = inside_of(points, verts_other, faces_other, lo_other, hi_other)
                    is_separated[idx[is_inside.reshape(len(idx), -1).any(1)]] = False
                return is_separated

            # Per-side informed escape directions: a side's coherent mean penetrating normal (a one-faced
            # press or bulge) and the local wall axes of its crossings - the normals of the partner faces
            # nearest its antipodal-pair verts, i.e. the faces of the crossed walls themselves. Wall axes
            # carry a pierce whose own vert normals cancel (a stem's ring verts point radially around the
            # stem, and its antipodal-pair axes span the stem's cross-section - both perpendicular to the
            # through-wall direction), and being face-based they survive sparse vert sampling. The axes are
            # clustered (sign-invariant, folded onto each cluster's dominant eigenvector) because one side
            # may cross several walls at once, and each crossing owns its axis and its witness verts - a
            # dominant unrelated contact would otherwise steer the push-back parallel to the pierced wall.
            # Appended to the Fibonacci grid the directions also remove the direction-grid tilt from the
            # separation.
            dirs_informed_sides = []
            is_press_sides = []
            press_probes = []
            clusters_sides = []
            has_press_normal = False
            for normals_pen, verts_pen, faces_near_pen, normals_faces_other, pairs_side, depths_pen in (
                (normals_a[is_inside_a0], va[is_pen_a], faces_near_a[is_pen_a], normals_faces_b, pairs_a, depth_a0),
                (normals_b[is_inside_b0], vb[is_pen_b], faces_near_b[is_pen_b], normals_faces_a, pairs_b, depth_b0),
            ):
                side_dirs = []
                side_is_press = []
                side_clusters = []
                press_probes.append(None)
                if len(normals_pen):
                    normal_mean = normals_pen.mean(0)
                    norm = np.linalg.norm(normal_mean)
                    if norm > 0.5:
                        side_dirs.append(np.outer([1.0, -1.0], normal_mean / norm))
                        side_is_press += [True, True]
                        has_press_normal = True
                        press_probes[-1] = (verts_pen[depths_pen[depths_pen > 0.0].argmax()], normal_mean / norm)
                if len(pairs_side):
                    verts_idx = np.unique(pairs_side)
                    axes_pairs = normals_faces_other[faces_near_pen[verts_idx]]
                    remaining = np.ones(len(axes_pairs), dtype=bool)
                    for _ in range(3):
                        if remaining.sum() < 4:
                            break
                        axes_rem = axes_pairs[remaining]
                        axis_dominant = np.linalg.eigh((axes_rem[:, :, None] * axes_rem[:, None, :]).mean(0))[1][:, -1]
                        in_cluster = remaining & (np.abs(axes_pairs @ axis_dominant) > 0.8)
                        if in_cluster.sum() >= 4:
                            axes_cluster = axes_pairs[in_cluster].copy()
                            axes_cluster[axes_cluster @ axis_dominant < 0.0] *= -1.0
                            axis_mean = axes_cluster.mean(0)
                            axis_mean /= np.linalg.norm(axis_mean)
                            side_dirs.append(np.outer([1.0, -1.0], axis_mean))
                            side_is_press += [False, False]
                            side_clusters.append((axis_mean, verts_pen[verts_idx[in_cluster]]))
                        remaining &= ~in_cluster
                dirs_informed_sides.append(np.concatenate(side_dirs) if side_dirs else np.empty((0, 3)))
                is_press_sides.append(np.array(side_is_press, dtype=bool))
                clusters_sides.append(side_clusters)
            dirs_search = np.concatenate([dirs, *dirs_informed_sides])
            is_press_dir = np.concatenate([np.zeros(len(dirs), dtype=bool), *is_press_sides])

            # Smallest separating translation: bracket each direction's transition on the geometric ladder,
            # then bisect in lockstep; np.inf when no direction separates within the ladder.
            i_first = np.full(len(dirs_search), -1)
            is_active = np.ones(len(dirs_search), dtype=bool)
            for i_r in range(len(probe)):
                if not is_active.any():
                    break
                idx = np.flatnonzero(is_active)
                is_separated = separated_at(dirs_search[idx] * probe[i_r])
                i_first[idx[is_separated]] = i_r
                is_active[idx[is_separated]] = False
            seps_dir = np.full(len(dirs_search), np.inf)
            idx = np.flatnonzero(i_first >= 0)
            if len(idx):
                lo = np.where(i_first[idx] > 0, probe[np.maximum(i_first[idx] - 1, 0)], 0.0)
                hi = probe[i_first[idx]]
                for _ in range(n_bisect):
                    mid = 0.5 * (lo + hi)
                    is_separated = separated_at(dirs_search[idx] * mid[:, None])
                    lo = np.where(is_separated, lo, mid)
                    hi = np.where(is_separated, mid, hi)
                seps_dir[idx] = hi
            separation = seps_dir.min() if len(seps_dir) else np.inf
            seps_press_sides = []
            i_dir = len(dirs)
            for is_press_side in is_press_sides:
                n_side = len(is_press_side)
                seps_side = seps_dir[i_dir : i_dir + n_side][is_press_side]
                seps_press_sides.append(seps_side.min() if len(seps_side) else np.inf)
                i_dir += n_side

            if is_inside_a0.all() or is_inside_b0.all():
                # Full containment: a body with every vert inside the partner has no free surface to heal
                # toward - burial is meaningless and only the extraction resolves (an engulfed sphere reads
                # R + r, however deep it is buried).
                depth = separation if np.isfinite(separation) else overlap
            elif max(is_inside_a0.mean(), is_inside_b0.mean()) >= 0.25:
                # Partial enclosure: unseating or extracting resolves nothing a solver would report - the
                # depth is the burial incursion, unless plainly separating is cheaper.
                depth = min(overlap, separation)
            elif not has_pierce:
                # Dent with a coherent breach normal: a directional press, resolved by retracting along
                # that normal - the separation RESTRICTED to the press directions, not the global minimum
                # (for a presser inside its container the global minimum is the extraction through the
                # mouth, which resolves nothing a solver would report). The retraction is only a press
                # resolution while it fits the wall it presses: a press that produced no pierce cannot run
                # deeper than the local wall thickness, so a larger directional separation means the
                # retraction is blocked (a body wedged inside its container escaping diagonally through
                # the mouth) and the depth is the incursion, as for an incoherent all-around seat.
                depth_press = np.inf
                s_probe = np.arange(1, 121) * 1e-3
                for press_probe, sep_press, (verts_other, faces_other, lo_other, hi_other) in zip(
                    press_probes,
                    seps_press_sides,
                    ((vb, fb, lo_b, hi_b), (va, fa, lo_a, hi_a)),
                ):
                    if press_probe is None or not np.isfinite(sep_press):
                        continue
                    vert_deep, normal_press = press_probe
                    thickness = 0.0
                    for sign in (1.0, -1.0):
                        pts = vert_deep[None] + sign * normal_press * s_probe[:, None]
                        is_inside_probe = inside_of(pts, verts_other, faces_other, lo_other, hi_other)
                        thickness += s_probe[-1] if is_inside_probe.all() else s_probe[np.argmin(is_inside_probe)]
                    if sep_press <= thickness + 2e-3:
                        depth_press = min(depth_press, sep_press)
                depth = depth_press if np.isfinite(depth_press) else min(overlap, separation)
            elif separation <= 10.0 * overlap:
                # Ordinary pierce: presses, rods, beams - the separation IS the resolution (it retracts the
                # press or backs the rod out; the breach normals of fully-crossed walls cancel anyway).
                depth = separation
            else:
                # Extraction-scale containment (separating costs an order of magnitude more than the
                # incursion: a bore-fit body poking through its container's wall): the resolution is the
                # push-back heal - the retreat along the wall axis that brings all of the crossing link's
                # material back through the pierced wall entirely. Projection onto the axis gives it
                # exactly: the retreat for a sign is the span from the wall's far face (the cluster's
                # extreme projection against that sign, so walls stacked along one axis - a beam through
                # both tube walls - are cleared together) to the link's farthest material (its stem tip - a
                # compact nub; a crossed plate measures its own half-extent instead and defers to the
                # separation via the near-parity gate below); the cheaper sign is the side to push back
                # through. Rigid translation feasibility is no
                # yardstick here: a bore-fit body cannot actually translate (any push re-presses the far
                # side of the bore), yet the wall crossing is still healed by exactly the protrusion.
                protrusions = []
                for i_side, side_clusters in enumerate(clusters_sides):
                    if not side_clusters:
                        continue
                    verts_side = (va, vb)[i_side]
                    heal_side = 0.0
                    for axis, verts_cluster in side_clusters:
                        proj_cluster = verts_cluster @ axis
                        proj_side = verts_side @ axis
                        retreat = min(proj_side.max() - proj_cluster.min(), proj_cluster.max() - proj_side.min())
                        heal_side = max(heal_side, retreat)
                    protrusions.append(heal_side)
                if not protrusions:
                    # No side has a coherent escape direction: crossings pressed from every side, a
                    # seat-family jam whose extraction resolves nothing a solver would report.
                    depth = min(overlap, separation)
                else:
                    heal = max(overlap, min(protrusions))
                    if heal < 0.8 * separation:
                        depth = heal
                    else:
                        # Near-parity means the crossing genuinely resolves by sliding out; the separation
                        # is the exact value while the marched heal carries the 1 mm grid quantization.
                        depth = separation
                if not np.isfinite(depth):
                    depth = overlap
            max_depth = max(max_depth, depth)
            if depth > cross_tol:
                crossings.append(Crossing(i_la, i_lb, depth))

    crossings.sort(key=lambda crossing: crossing.depth, reverse=True)
    return max_depth, crossings


def display_collision_pairs(pairs):
    """Open a self-contained interactive 3D viewer of colliding link pairs in the default browser.

    Each entry of `pairs` is `(geoms_a, geoms_b, label)`, where each `geoms` is a list of `(verts, faces)`
    collision meshes (torch or numpy); the geoms of a link are merged and the two links drawn blue and red,
    with a dropdown to switch between pairs. Each bead in the hover overlay marks a surface the cursor ray
    crosses (coloured by link, back walls flagged with a cross); a digit key (numpad or top row) picks the
    N-th surface directly, shift+scroll steps to enclosed surfaces, and clicking two beads reads their
    distance in millimetres.
    """
    data = []
    for geoms_a, geoms_b, label in pairs:
        entry = {"label": label}
        for key, geoms in (("a", geoms_a), ("b", geoms_b)):
            verts_all, faces_all, offset = [], [], 0
            for verts, faces in geoms:
                verts = tensor_to_array(verts, dtype=np.float64)
                faces = tensor_to_array(faces, dtype=np.int64)
                verts_all.append(verts)
                faces_all.append(faces + offset)
                offset += len(verts)
            entry[key] = {"v": np.concatenate(verts_all).tolist(), "f": np.concatenate(faces_all).tolist()}
        data.append(entry)
    template = (Path(__file__).parent / "mesh_pairs_viewer.html").read_text()
    with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False) as file:
        file.write(template.replace("__DATA__", json.dumps(data)))
    webbrowser.open(f"file://{file.name}")

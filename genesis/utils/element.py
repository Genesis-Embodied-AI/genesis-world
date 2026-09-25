import numpy as np
import igl

import genesis as gs

from . import mesh as mu


def mesh_to_elements(mesh, tet_cfg=dict()):
    """Tetrahedralize a surface trimesh, with the result cached on disk keyed on vertices, faces and configuration."""
    cache = mu.get_tet_cache(mesh.vertices, mesh.faces, tet_cfg)

    # loading pre-computed cache if available
    elements = cache.load()
    if elements is not None:
        gs.logger.debug("Tetrahedra file (`.tet`) found in cache.")
        return elements

    with gs.logger.timer(f"Tetrahedralization with configuration {tet_cfg} and generating `.tet` file:"):
        elements = mu.tetrahedralize_mesh(mesh, tet_cfg)
        cache.save(elements)
    return elements


def split_all_surface_tets(verts, elems):
    """
    Splits tetrahedras that have 4 vertices on the surface into 4 smaller tetrahedras.

    This is useful for the hydroelastic contact model.
    """
    F, *_ = igl.boundary_facets(elems)
    on_surface = np.zeros(verts.shape[0], dtype=bool)
    on_surface[F.reshape(-1)] = True
    all_on_surface = np.all(on_surface[elems], axis=1)
    if not all_on_surface.any():
        return verts, elems
    bad_elems = elems[all_on_surface]
    new_verts = np.mean(verts[bad_elems], axis=1, dtype=np.float32)
    new_elems = []
    for idx, (v0, v1, v2, v3) in enumerate(bad_elems, len(verts)):
        new_elems.append([v0, v1, v2, idx])
        new_elems.append([v0, v1, idx, v3])
        new_elems.append([v0, idx, v2, v3])
        new_elems.append([idx, v1, v2, v3])
    new_elems = np.array(new_elems, dtype=np.int32)
    verts = np.concatenate([verts, new_verts], axis=0)
    # remove the bad elements from the original elements
    elems = np.concatenate([elems[~all_on_surface], new_elems], axis=0)
    return verts, elems

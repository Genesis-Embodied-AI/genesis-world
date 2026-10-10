import numpy as np
import pytest
import trimesh

import genesis as gs
from genesis.utils.misc import qd_to_numpy

from ..utils.assertions import assert_allclose


@pytest.mark.precision("64")
def test_embedded_visual_mesh_follows_deformation(tmp_path, show_viewer):
    # A fine icosphere as the visual mesh of a coarse soft cube: the embedded render vertices must reproduce their
    # rest placement exactly at build and follow the barycentric combination of their tetrahedra as the cube deforms.
    visual = trimesh.creation.icosphere(subdivisions=3, radius=0.045)
    visual_path = str(tmp_path / "visual.obj")
    visual.export(visual_path)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=1.0 / 60.0, gravity=(0.0, 0.0, -9.8)),
        mochi_options=gs.options.MochiOptions(n_newton_iterations=8),
        show_viewer=show_viewer,
    )
    scene.add_entity(gs.morphs.Plane(), material=gs.materials.Mochi.Rigid())
    cube = scene.add_entity(
        gs.morphs.Box(size=(0.1, 0.1, 0.1), pos=(0.0, 0.0, 0.2), maxvolume=2e-4, nobisect=False),
        material=gs.materials.Mochi.Elastic(E=2e4, nu=0.4, rho=1000.0),
    )
    cube.set_visual_mesh(file=visual_path, pos=(0.0, 0.005, 0.01), scale=1.5)
    scene.build()
    solver = scene.mochi_solver

    vgeom = cube.vgeoms[0]
    assert vgeom.elems_idx is not None and vgeom.n_vverts == len(visual.vertices)

    def render_verts():
        vverts, _, _ = solver.get_soft_state_render(0)
        return qd_to_numpy(vverts)[vgeom.vvert_start : vgeom.vvert_end, 0].astype(np.float64)

    # The visual mesh is placed in the frame of the simulation mesh (a primitive morph: its local frame plus the
    # morph position).
    expected_rest = visual.vertices * 1.5 + (0.0, 0.005, 0.01) + cube.morph.pos
    assert_allclose(render_verts(), expected_rest, atol=1e-6)

    for _ in range(30):
        scene.step()
    verts_pos = qd_to_numpy(solver.soft_state.verts_pos)[:, 0].astype(np.float64)
    expected = np.einsum("ik,ikj->ij", vgeom.bary, verts_pos[cube.elems[vgeom.elems_idx]])
    assert_allclose(render_verts(), expected, atol=1e-6)
    # The cube has fallen and squashed: the skinned visual mesh moved with it.
    assert render_verts()[:, 2].max() < expected_rest[:, 2].max() - 0.02

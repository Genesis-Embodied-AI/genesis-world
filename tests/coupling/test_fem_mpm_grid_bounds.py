"""The legacy coupler's FEM <-> MPM exchange (LegacyCoupler.fem_surface_force) reads the MPM grid around every FEM
surface vertex; a vertex whose 3x3x3 stencil leaves the grid skips the exchange instead of reading outside it (on
CPU an out-of-bounds read can fault the process). No MPM mass lives outside the grid, so nothing is lost."""

import numpy as np
import pytest

import genesis as gs
from genesis.utils.misc import tensor_to_array

DT, SUBSTEPS = 2e-3, 10
FEM_SIZE, BODY = 0.10, 0.04
GRID = dict(lower_bound=(-0.2, -0.2, 0.0), upper_bound=(0.2, 0.2, 0.4), grid_density=64)


def _scene(fem_bottom, show_viewer, with_fem=True, with_body=True):
    """An FEM block (its bottom at z `fem_bottom`) and an MPM body set down on it, coupled by the legacy coupler's
    FEM-MPM pair; the MPM grid starts at z 0, so a bottom under it puts FEM surface vertices outside the grid."""
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=DT, substeps=SUBSTEPS),
        fem_options=gs.options.FEMOptions(use_implicit_solver=True, damping_alpha=5.0),
        mpm_options=gs.options.MPMOptions(**GRID),
        coupler_options=gs.options.LegacyCouplerOptions(fem_mpm=True),
        show_viewer=show_viewer,
        show_FPS=False,
    )
    fem = body = None
    if with_fem:
        fem = scene.add_entity(
            morph=gs.morphs.Box(size=(FEM_SIZE, FEM_SIZE, FEM_SIZE), pos=(0.0, 0.0, fem_bottom + FEM_SIZE / 2)),
            material=gs.materials.FEM.Elastic(E=2e4, nu=0.4, rho=1000.0),
        )
    if with_body:
        body = scene.add_entity(
            morph=gs.morphs.Box(size=(BODY, BODY, BODY), pos=(0.0, 0.0, 0.15 + BODY / 2)),
            material=gs.materials.MPM.ElastoPlastic(E=5e4, nu=0.3, rho=2000.0),
        )
    scene.build()
    return scene, fem, body


def _fem_pos(fem):
    return tensor_to_array(fem.get_state().pos).reshape(-1, 3)


def _body_pos(body):
    return tensor_to_array(body.get_particles_pos()).reshape(-1, 3)


@pytest.mark.required
def test_a_fem_vertex_under_the_grid_steps_without_a_fault_and_the_rest_still_couples(show_viewer):
    """The block's bottom 3 mm under the MPM grid: its bottom vertices' stencils leave the grid; the scene steps on
    the CPU (before the check this read outside the grid and faulted), every position stays finite, and the body
    set down on the block's top (inside the grid) is held by it through the exchange: it does not fall through."""
    scene, fem, body = _scene(fem_bottom=-0.003, show_viewer=show_viewer)
    bottom0 = _fem_pos(fem)[:, 2].min()
    assert bottom0 < 0.0, bottom0                                            # vertices under the grid's lower bound
    for _ in range(40):
        scene.step()
    fem_pos, body_pos = _fem_pos(fem), _body_pos(body)
    assert np.isfinite(fem_pos).all() and np.isfinite(body_pos).all()
    top = fem_pos[:, 2].max()
    assert body_pos[:, 2].min() > top - 0.015, (body_pos[:, 2].min(), top)   # on the block, not through it
    assert body_pos[:, 2].min() < 0.15, body_pos[:, 2].min()                 # it came down onto the block


@pytest.mark.required
def test_a_fem_block_wholly_outside_the_grid_exchanges_nothing(show_viewer):
    """The block set under the grid altogether (its top 5 cm below the grid's floor): every stencil leaves the grid,
    the exchange is skipped for every vertex, and the body falls exactly as it does with no FEM block in the scene."""
    scene_a, _fem, body_a = _scene(fem_bottom=-0.15, show_viewer=show_viewer)
    scene_b, _none, body_b = _scene(fem_bottom=-0.15, show_viewer=show_viewer, with_fem=False)
    for _ in range(15):
        scene_a.step()
        scene_b.step()
    pa, pb = _body_pos(body_a), _body_pos(body_b)
    assert np.isfinite(pa).all()
    assert np.array_equal(pa, pb), float(np.abs(pa - pb).max())

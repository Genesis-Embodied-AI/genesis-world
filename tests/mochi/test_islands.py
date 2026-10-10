import numpy as np
import pytest

import genesis as gs
from genesis.utils.misc import qd_to_numpy, tensor_to_array

from ..utils.assertions import assert_allclose


def _mochi_scene(show_viewer, dt, **mochi_kwargs):
    return gs.Scene(
        sim_options=gs.options.SimOptions(dt=dt, gravity=(0.0, 0.0, -9.8)),
        mochi_options=gs.options.MochiOptions(**mochi_kwargs),
        show_viewer=show_viewer,
    )


@pytest.mark.precision("64")
@pytest.mark.parametrize("linear_solver", ["auto", "pcg"])
def test_islands_box_piles(show_viewer, linear_solver):
    size = 0.2
    # Three piles of two boxes and two single boxes: 48 degrees of freedom in five islands of at most 12.
    scene = _mochi_scene(
        show_viewer, 1.0 / 60.0, n_newton_iterations=8, linear_solver=linear_solver, dense_solver_max_dofs=12
    )
    scene.add_entity(gs.morphs.Plane(), material=gs.materials.Mochi.Rigid())
    boxes = []
    for x in (-0.6, 0.0, 0.6):
        for z in (0.5 * size, 1.5 * size + 2e-3):
            boxes.append(
                scene.add_entity(
                    gs.morphs.Box(size=(size, size, size), pos=(x, 0.0, z)), material=gs.materials.Mochi.Rigid()
                )
            )
    for x in (-1.2, 1.2):
        boxes.append(
            scene.add_entity(
                gs.morphs.Box(size=(size, size, size), pos=(x, 0.0, 0.5 * size + 1e-3)),
                material=gs.materials.Mochi.Rigid(),
            )
        )
    scene.build()
    solver = scene.mochi_solver
    assert solver.n_dofs_total == 48
    assert solver.mochi_config.has_dense == (linear_solver == "auto")
    for _ in range(120):
        scene.step()
    islands = solver.island_state
    # The static plane is a node of its own, without degrees of freedom.
    assert int(qd_to_numpy(islands.n_islands)[0]) == 6
    assert int(qd_to_numpy(islands.island_max_dofs)[0]) == 12
    assert bool(qd_to_numpy(islands.uses_dense)[0]) == (linear_solver == "auto")
    n_dofs_islands = qd_to_numpy(islands.island_n_dofs)[:6, 0]
    assert sorted(n_dofs_islands.tolist()) == [0, 6, 6, 12, 12, 12]
    dofs = qd_to_numpy(islands.island_dofs)[:, 0]
    assert sorted(dofs.tolist()) == list(range(48))
    # Every pile rests: the net contact force on each box balances its own weight.
    for i_pile in range(3):
        lower, upper = boxes[2 * i_pile], boxes[2 * i_pile + 1]
        mass = float(tensor_to_array(lower.get_mass()))
        for box in (lower, upper):
            assert_allclose(
                tensor_to_array(box.get_links_net_contact_force())[0], (0.0, 0.0, mass * 9.8), rtol=1e-3, atol=1e-3
            )
        assert_allclose(
            float(tensor_to_array(upper.get_pos())[2]) - float(tensor_to_array(lower.get_pos())[2]), size, atol=2e-3
        )
    for box in boxes:
        assert_allclose(box.get_dofs_velocity(), 0.0, atol=1e-4)


@pytest.mark.precision("64")
def test_islands_soft_and_rigid(show_viewer):
    scene = _mochi_scene(show_viewer, 1.0 / 60.0, n_newton_iterations=8, dense_solver_max_dofs=400)
    scene.add_entity(gs.morphs.Plane(), material=gs.materials.Mochi.Rigid())
    support = scene.add_entity(
        gs.morphs.Box(size=(0.3, 0.3, 0.1), pos=(0.0, 0.0, 0.05)),
        material=gs.materials.Mochi.Rigid(collider_type="box"),
    )
    cube = scene.add_entity(
        gs.morphs.Box(size=(0.12, 0.12, 0.12), pos=(0.0, 0.0, 0.17), maxvolume=0.0005, nobisect=False),
        material=gs.materials.Mochi.Elastic(E=1e5, nu=0.45, rho=1000.0),
    )
    loose = scene.add_entity(
        gs.morphs.Box(size=(0.2, 0.2, 0.2), pos=(1.0, 0.0, 0.11)), material=gs.materials.Mochi.Rigid()
    )
    scene.build()
    solver = scene.mochi_solver
    assert solver.mochi_config.has_dense
    for _ in range(120):
        scene.step()
    islands = solver.island_state
    # Islands: {plane}, {support, soft cube}, {loose box}.
    assert int(qd_to_numpy(islands.n_islands)[0]) == 3
    assert int(qd_to_numpy(islands.island_max_dofs)[0]) == 6 + 3 * cube.n_vertices
    assert bool(qd_to_numpy(islands.uses_dense)[0])
    dofs_island = qd_to_numpy(islands.dofs_island)[:, 0]
    support_island = dofs_island[support.dof_start]
    cube_dof_start = solver.n_dofs + 3 * cube.v_start
    assert np.all(dofs_island[cube_dof_start : cube_dof_start + 3 * cube.n_vertices] == support_island)
    assert dofs_island[loose.dof_start] != support_island
    # Rest: the ground carries the support and the soft cube, so the net contact force on the support (ground push
    # minus the cube's weight) balances its own weight; the loose box rests on the ground.
    assert_allclose(cube.get_vertices_velocity(), 0.0, atol=1e-5)
    assert_allclose(
        tensor_to_array(cube.get_vertices_contact_force()).sum(axis=0), (0.0, 0.0, cube.mass * 9.8), atol=2e-3
    )
    assert_allclose(
        tensor_to_array(support.get_links_net_contact_force())[0],
        (0.0, 0.0, float(tensor_to_array(support.get_mass())) * 9.8),
        rtol=1e-3,
        atol=2e-3,
    )
    assert_allclose(loose.get_dofs_velocity(), 0.0, atol=1e-4)

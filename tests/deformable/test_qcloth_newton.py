import numpy as np
import pytest
import trimesh
from quadrants.lang import impl

import genesis as gs
from genesis.engine.systems import build_scene_engine
from genesis.engine.systems.finite_element import QuadraticBending
from genesis.engine.systems.sim_engine import _step_kernel
from genesis.utils.misc import qd_to_numpy, tensor_to_array


def make_grid(path, n=3, size=0.2):
    vertices = np.array(
        [[x * size / (n - 1), y * size / (n - 1), 0.5] for y in range(n) for x in range(n)],
        dtype=np.float64,
    )
    triangles = []
    for y in range(n - 1):
        for x in range(n - 1):
            lower_left = y * n + x
            lower_right = lower_left + 1
            upper_left = lower_left + n
            upper_right = upper_left + 1
            triangles.extend(
                [
                    [lower_left, lower_right, upper_left],
                    [lower_right, upper_right, upper_left],
                ]
            )
    trimesh.Trimesh(vertices, np.asarray(triangles), process=False).export(path)


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("backend", [gs.gpu])
def test_qcloth_zero_hinge_capacity():
    bending = QuadraticBending()
    data = bending.data
    bending.wire_data(
        hinge_indices=np.empty((0, 4), dtype=np.int32),
        bending_stiffness=np.empty(0, dtype=np.float64),
        Q0=np.empty((0, 4, 4), dtype=np.float64),
        vert_bend_k=np.zeros(3, dtype=np.float64),
    )

    assert bending.data is data
    np.testing.assert_array_equal(qd_to_numpy(data.n_hinges), 0)
    np.testing.assert_array_equal(
        qd_to_numpy(data.hinge_indices),
        np.zeros((1, 4), dtype=np.int32),
    )
    np.testing.assert_array_equal(
        qd_to_numpy(data.k),
        np.zeros(1, dtype=np.float64),
    )
    np.testing.assert_array_equal(
        qd_to_numpy(data.Q0),
        np.zeros((1, 16), dtype=np.float64),
    )


@pytest.mark.required
@pytest.mark.parametrize("backend", [gs.gpu])
def test_scene_selects_concrete_simulator(show_viewer):
    from genesis.engine.newton_simulator import NewtonSimulator
    from genesis.engine.simulator import LegacySimulator

    legacy_scene = gs.Scene(show_viewer=show_viewer)
    newton_scene = gs.Scene(
        engine_options=gs.options.NewtonEngineOptions(),
        show_viewer=show_viewer,
    )

    assert isinstance(legacy_scene.sim, LegacySimulator)
    assert isinstance(newton_scene.sim, NewtonSimulator)
    with pytest.raises(gs.GenesisException, match="coupler_options"):
        gs.Scene(
            engine_options=gs.options.NewtonEngineOptions(),
            coupler_options=gs.options.LegacyCouplerOptions(),
            show_viewer=show_viewer,
        )


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("backend", [gs.gpu])
def test_qcloth_graph_step(tmp_path, show_viewer):
    path = tmp_path / "qcloth_grid.obj"
    make_grid(path)
    scene = gs.Scene(
        engine_options=gs.options.NewtonEngineOptions(),
        show_viewer=show_viewer,
    )
    cloth = scene.add_entity(
        morph=gs.morphs.Mesh(file=str(path)),
        material=gs.materials.FEM.QCloth(E=1e4, thickness=1e-3),
    )
    scene.build()
    cloth.set_vertex_constraints([0, 2])

    engine = build_scene_engine(scene)
    initial = qd_to_numpy(engine.fem.x)
    for _ in range(5):
        engine.step()
    # Inspect the graph launch directly: engine.step() performs scene
    # writeback and a scalar failure read after this kernel.
    _step_kernel(
        engine.data,
        engine.checkpoint_never_yield,
        engine.checkpoint_never_yield,
        engine.checkpoint_never_yield,
        engine.global_linear_system_data.matrix.triplet_overflow,
        engine.checkpoint_never_yield,
        engine.checkpoint_never_yield,
        engine.graph_fastcache_key,
    )
    graph_cache_used = impl.get_runtime().prog.get_graph_cache_used_on_last_call()
    graph_num_nodes = impl.get_runtime().prog.get_graph_num_nodes_on_last_call()
    final = qd_to_numpy(engine.fem.x)

    assert graph_cache_used
    assert graph_num_nodes > 0
    assert np.isfinite(final).all()
    np.testing.assert_array_equal(final[[0, 2]], initial[[0, 2]])
    assert final[4, 2] < initial[4, 2]


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("backend", [gs.gpu])
def test_newton_engine_scene_step_and_reset(tmp_path, show_viewer):
    path = tmp_path / "newton_engine_grid.obj"
    make_grid(path)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01),
        engine_options=gs.options.NewtonEngineOptions(),
        show_viewer=show_viewer,
    )
    cloth = scene.add_entity(
        morph=gs.morphs.Mesh(file=str(path)),
        material=gs.materials.FEM.QCloth(E=1e4, thickness=1e-3),
    )
    scene.build()
    first_engine = scene.sim.engine
    cloth.set_vertex_constraints([0, 2])

    initial = np.squeeze(tensor_to_array(cloth.get_state().pos), axis=0).copy()
    for _ in range(3):
        scene.step()
    first_final = np.squeeze(tensor_to_array(cloth.get_state().pos), axis=0).copy()

    assert scene.sim.cur_step_global == 3
    assert scene.sim.engine is first_engine
    assert np.isfinite(first_final).all()
    np.testing.assert_array_equal(first_final[[0, 2]], initial[[0, 2]])
    assert first_final[4, 2] < initial[4, 2]
    assert int(qd_to_numpy(first_engine.frame_failed)) == 0

    scene.reset()
    assert scene.sim.cur_step_global == 0
    assert scene.sim.engine is first_engine
    reset_state = np.squeeze(tensor_to_array(cloth.get_state().pos), axis=0)
    np.testing.assert_allclose(reset_state, initial, rtol=0.0, atol=1e-12)
    for _ in range(3):
        scene.step()
    second_final = np.squeeze(tensor_to_array(cloth.get_state().pos), axis=0)

    assert scene.sim.cur_step_global == 3
    np.testing.assert_allclose(second_final, first_final, rtol=1e-10, atol=1e-12)


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("backend", [gs.gpu])
def test_qcloth_freefall_matches_reference_converged_step(tmp_path, show_viewer):
    path = tmp_path / "qcloth_freefall.obj"
    make_grid(path)
    dt = 0.01
    gravity = np.array([0.0, 0.0, -9.8], dtype=np.float64)
    bending_youngs_modulus = 3.0e3
    thickness = 1.0e-3
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=dt,
            gravity=tuple(gravity),
        ),
        engine_options=gs.options.NewtonEngineOptions(),
        show_viewer=show_viewer,
    )
    scene.add_entity(
        morph=gs.morphs.Mesh(file=str(path)),
        material=gs.materials.FEM.QCloth(
            E=2.0e4,
            shear_modulus=2.0e3,
            thickness=thickness,
            bending_youngs_modulus=bending_youngs_modulus,
        ),
    )
    scene.build()
    engine = build_scene_engine(
        scene,
        contact_config={
            "contact/enable": 0,
            "linear_system/tol_rate": 1.0e-10,
        },
    )

    expected_bending_stiffness = bending_youngs_modulus * (2.0 * thickness) ** 3 / 12.0
    np.testing.assert_allclose(
        qd_to_numpy(engine.fem.quadratic_bending.k),
        expected_bending_stiffness,
        rtol=0.0,
        atol=1.0e-18,
    )

    initial = qd_to_numpy(engine.fem.x)
    engine.step()
    displacement = qd_to_numpy(engine.fem.x) - initial
    np.testing.assert_allclose(
        displacement,
        np.broadcast_to(dt * dt * gravity, displacement.shape),
        rtol=0.0,
        atol=2.0e-14,
    )
    assert engine.get_newton_iters() == 2


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("backend", [gs.gpu])
def test_qcloth_global_managers_two_entities(tmp_path, show_viewer):
    path = tmp_path / "qcloth_grid.obj"
    make_grid(path)
    scene = gs.Scene(
        engine_options=gs.options.NewtonEngineOptions(),
        show_viewer=show_viewer,
    )
    material = gs.materials.FEM.QCloth(E=1e4, thickness=1e-3)
    scene.add_entity(
        morph=gs.morphs.Mesh(file=str(path)),
        material=material,
    )
    scene.add_entity(
        morph=gs.morphs.Mesh(file=str(path), pos=(0.3, 0.0, 0.2)),
        material=material,
    )
    scene.build()

    engine = build_scene_engine(scene)
    np.testing.assert_array_equal(qd_to_numpy(engine.global_vertex_data.n_verts), 18)
    np.testing.assert_array_equal(qd_to_numpy(engine.global_body_data.n_bodies), 2)
    np.testing.assert_array_equal(
        qd_to_numpy(engine.global_body_data.vertex_offsets)[:3],
        np.array([0, 9, 18], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        qd_to_numpy(engine.global_vertex_data.body_id)[:18],
        np.repeat(np.arange(2, dtype=np.int32), 9),
    )
    np.testing.assert_allclose(
        qd_to_numpy(engine.global_vertex_data.positions)[:18],
        qd_to_numpy(engine.fem.x)[:18],
    )
    np.testing.assert_array_equal(qd_to_numpy(engine.global_surface_data.n_surf_triangles), 16)
    np.testing.assert_array_equal(qd_to_numpy(engine.global_surface_data.n_surf_edges), 32)
    np.testing.assert_allclose(
        qd_to_numpy(engine.global_surface_data.face_area_weights)[:16].sum(),
        0.08,
    )

    engine.step()
    np.testing.assert_allclose(
        qd_to_numpy(engine.global_vertex_data.positions)[:18],
        qd_to_numpy(engine.fem.x)[:18],
    )

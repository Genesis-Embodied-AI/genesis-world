from pathlib import Path

import numpy as np
import pytest
import trimesh

import genesis as gs
from genesis.engine.systems import ContactTabular, build_scene_engine
from genesis.engine.systems.contact import CONTACT_CONFIG_DEFAULTS
from genesis.engine.systems.sim_engine import ContactCheckpoint
from genesis.utils.misc import qd_to_numpy


def make_contact_grid(path, n=3, size=0.2, height=0.5):
    vertices = np.array(
        [[x * size / (n - 1), y * size / (n - 1), height] for y in range(n) for x in range(n)],
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


def test_contact_parameter_manifest_defaults():
    assert dict(CONTACT_CONFIG_DEFAULTS) == {
        "contact/enable": 1,
        "contact/d_hat": 0.01,
        "contact/max_step_in_d_hat": -1.0,
        "contact/ccd_bound": "directional",
        "contact/adaptive_kappa_mode": "per-body",
        "contact/adaptive_kappa_tick": "newton",
        "contact/init_collision_pair_capacity": 1_000,
        "contact/ccd_partition": 1,
        "contact/ccd_partition_sv_max_iter": 64,
        "contact/intersection_check": 0,
        "contact/intersection_check_capacity": 1_024,
        "contact/constitution": "auto",
        "friction/eps_v": 1e-2,
        "linear_system/tol_rate": 1e-4,
        "extras/capacity_grow_factor": 1.2,
        "extras/capacity_shrink_threshold": 0.8,
        "extras/pipeline/genesis_serial": 0,
        "extras/ls_forensics/test_energy_bias": 0.0,
        "topo/grow_factor": 1.5,
        "bvh/type": "info_lbvh_batched_dop14",
        "bvh/pt_query": "warp",
        "bvh/ee_query": "dual",
        "bvh/dual/frontier_levels": 0,
        "bvh/dual/target_waves": 24.0,
        "bvh/dual/max_levels": 18,
        "rigid_proxy/globalization": "merit",
        "rigid_proxy/restoration": 1,
        "rigid_proxy/test_merit_energy_bias": 0.0,
        "rigid_forest/fused": 1,
        "extras/rigid_forest/genesis_legacy": 0,
        "extras/sort_reduce/genesis_legacy": 0,
        "extras/bvh/genesis_legacy_fp64_bounds": 0,
        "extras/bvh/genesis_legacy_refit": 0,
    }
    model = ContactTabular().at(0, 0)
    assert model.friction_rate == 0.05
    assert model.resistance == 1e4
    assert model.enable
    assert model.enable_ee


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("backend", [gs.gpu])
@pytest.mark.parametrize(
    "genesis_legacy_sort_reduce",
    (False, True),
)
def test_qcloth_contact_graph_step(
    tmp_path,
    show_viewer,
    genesis_legacy_sort_reduce,
):
    path = tmp_path / "contact_grid.obj"
    make_contact_grid(path, height=0.008)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01),
        coupler_options=gs.options.NewtonCouplerOptions(),
        show_viewer=show_viewer,
    )
    scene.add_entity(
        morph=gs.morphs.Mesh(file=str(path)),
        material=gs.materials.FEM.QCloth(E=1e4, thickness=1e-3),
    )
    scene.build()

    engine = build_scene_engine(
        scene,
        contact_config={
            "extras/sort_reduce/genesis_legacy": int(genesis_legacy_sort_reduce),
        },
        halfplanes=(
            np.array([[0.0, 0.0, 0.0]], dtype=np.float64),
            np.array([[0.0, 0.0, 1.0]], dtype=np.float64),
        ),
    )
    engine.step()

    assert int(qd_to_numpy(engine.frame_failed)) == 0
    assert int(qd_to_numpy(engine.contact.intersection_flag)) == 0
    assert int(qd_to_numpy(engine.contact.n_pairs_ph)) > 0
    assert int(qd_to_numpy(engine.contact.n_active_pairs)) > 0
    assert np.isfinite(qd_to_numpy(engine.fem.x)).all()


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("backend", [gs.gpu])
def test_contact_checkpoint_capacity_growth(tmp_path, show_viewer):
    path = tmp_path / "contact_grid.obj"
    make_contact_grid(path, height=0.008)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01),
        coupler_options=gs.options.NewtonCouplerOptions(),
        show_viewer=show_viewer,
    )
    scene.add_entity(
        morph=gs.morphs.Mesh(file=str(path)),
        material=gs.materials.FEM.QCloth(E=1e4, thickness=1e-3),
    )
    scene.build()
    engine = build_scene_engine(
        scene,
        contact_config={"contact/init_collision_pair_capacity": 1},
        halfplanes=(
            np.array([[0.0, 0.0, 0.0]], dtype=np.float64),
            np.array([[0.0, 0.0, 1.0]], dtype=np.float64),
        ),
    )

    assert engine.contact.pairs_ph.shape[0] > 1
    engine.contact.max_contact_doublets.from_numpy(np.array(1, dtype=np.int32))
    engine.contact.max_contact_triplets.from_numpy(np.array(1, dtype=np.int32))
    engine.global_linear_system.max_triplets.from_numpy(np.array(1, dtype=np.int32))
    engine.step()

    assert int(qd_to_numpy(engine.contact.max_friction_pairs_ph)) >= int(
        qd_to_numpy(engine.contact.n_friction_pairs_ph)
    )
    assert int(qd_to_numpy(engine.contact.max_contact_doublets)) >= int(
        qd_to_numpy(engine.contact.n_counted_doublets)
    ) + int(qd_to_numpy(engine.contact.n_friction_demand_doublets))
    assert int(qd_to_numpy(engine.contact.max_contact_triplets)) >= int(
        qd_to_numpy(engine.contact.n_counted_triplets)
    ) + int(qd_to_numpy(engine.contact.n_friction_demand_triplets))
    assert int(qd_to_numpy(engine.global_linear_system.max_triplets)) >= int(
        qd_to_numpy(engine.global_linear_system.n_triplets)
    )


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("backend", [gs.gpu])
def test_contact_checkpoint_yield_skips_pcg(tmp_path, show_viewer):
    """Pin Quadrants #956's qd.func suffix workaround: a SORT yield must leave PCG completely untouched."""
    path = tmp_path / "contact_grid.obj"
    make_contact_grid(path, height=0.008)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01),
        coupler_options=gs.options.NewtonCouplerOptions(),
        show_viewer=show_viewer,
    )
    scene.add_entity(
        morph=gs.morphs.Mesh(file=str(path)),
        material=gs.materials.FEM.QCloth(E=1e4, thickness=1e-3),
    )
    scene.build()
    engine = build_scene_engine(
        scene,
        contact_config={},
        halfplanes=(
            np.array([[0.0, 0.0, 0.0]], dtype=np.float64),
            np.array([[0.0, 0.0, 1.0]], dtype=np.float64),
        ),
    )

    sentinel = 123
    engine.global_linear_system.max_triplets.from_numpy(np.array(1, dtype=np.int32))
    engine.pcg_solver.linear_pcg.n_iterations.from_numpy(np.array(sentinel, dtype=np.int32))
    status = engine._step_kernel(
        engine.contact.overflow_flag,
        engine.contact.count_overflow_flag,
        engine.contact.contact_padding_overflow,
        engine.global_linear_system.triplet_overflow,
        engine.contact.friction_overflow_flag,
        engine.contact.et_overflow_flag,
        engine.graph_fastcache_key,
    )

    assert status.yielded
    assert status.checkpoint == ContactCheckpoint.SORT
    assert int(qd_to_numpy(engine.pcg_solver.linear_pcg.n_iterations)) == sentinel


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("backend", [gs.gpu])
def test_contact_rejects_initial_cloth_intersection(tmp_path, show_viewer):
    path = tmp_path / "intersecting_grid.obj"
    make_contact_grid(path, height=0.2)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01),
        coupler_options=gs.options.NewtonCouplerOptions(),
        show_viewer=show_viewer,
    )
    material = gs.materials.FEM.QCloth(E=1e4, thickness=1e-3)
    scene.add_entity(morph=gs.morphs.Mesh(file=str(path)), material=material)
    scene.add_entity(morph=gs.morphs.Mesh(file=str(path)), material=material)
    scene.build()

    with pytest.raises(
        RuntimeError,
        match="ET check: initial state detected",
    ) as error:
        build_scene_engine(
            scene,
            contact_config={
                "contact/intersection_check": 1,
                "contact/intersection_check_capacity": 1,
            },
        )
    assert "edge geo_id" in str(error.value)
    assert "face geo_id" in str(error.value)
    assert "fem_solver.entities[" in str(error.value)


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("backend", [gs.gpu])
def test_et_report_resolves_rigid_geometry(tmp_path, show_viewer):
    path = tmp_path / "box_intersecting_grid.obj"
    make_contact_grid(path, n=5, size=0.16, height=0.04)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01),
        coupler_options=gs.options.NewtonCouplerOptions(),
        show_viewer=show_viewer,
    )
    scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.08, 0.08, 0.04),
            size=(0.08, 0.08, 0.08),
            fixed=True,
        ),
        vis_mode="collision",
    )
    scene.add_entity(
        morph=gs.morphs.Mesh(file=str(path)),
        material=gs.materials.FEM.QCloth(E=1e4, thickness=1e-3),
    )
    scene.build()

    with pytest.raises(
        RuntimeError,
        match="ET check: initial state detected",
    ) as error:
        build_scene_engine(
            scene,
            contact_config={"contact/intersection_check": 1},
        )
    message = str(error.value)
    assert "source RIGID" in message
    assert "rigid_solver.geoms[0]" in message


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("backend", [gs.gpu])
def test_crossed_vertical_cloths_drop_without_penetration(show_viewer):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01),
        coupler_options=gs.options.NewtonCouplerOptions(),
        show_viewer=show_viewer,
    )
    asset = Path(__file__).parents[2] / "examples" / "newton_coupling" / "assets" / "qcloth_grid.obj"
    material = gs.materials.FEM.QCloth(
        E=2e4,
        shear_modulus=2e3,
        rho=200.0,
        thickness=1e-3,
        bending_youngs_modulus=1e6,
    )
    scene.add_entity(
        morph=gs.morphs.Mesh(file=str(asset), pos=(0.0, 0.0, 0.23), euler=(90.0, 0.0, 0.0)),
        material=material,
    )
    scene.add_entity(
        morph=gs.morphs.Mesh(file=str(asset), pos=(0.0, 0.0, 0.68), euler=(90.0, 0.0, 90.0)),
        material=material,
    )
    scene.build()
    engine = build_scene_engine(
        scene,
        contact_config={},
        halfplanes=(
            np.array([[0.0, 0.0, 0.0]], dtype=np.float64),
            np.array([[0.0, 0.0, 1.0]], dtype=np.float64),
        ),
    )

    saw_halfplane_contact = False
    saw_cross_cloth_candidate = False
    saw_friction_energy = False
    for _ in range(120):
        engine.step()
        assert int(qd_to_numpy(engine.contact.intersection_flag)) == 0
        saw_halfplane_contact |= int(qd_to_numpy(engine.contact.n_pairs_ph)) > 0
        saw_friction_energy |= float(qd_to_numpy(engine.contact.friction_energy)) > 0.0

        n_pt = int(qd_to_numpy(engine.contact.n_pairs_pt))
        if n_pt:
            pairs = qd_to_numpy(engine.contact.pairs_pt)[:n_pt]
            surface_vertices = qd_to_numpy(engine.global_surface_manager.surf_verts)
            triangles = qd_to_numpy(engine.global_surface_manager.surf_triangles)
            body_ids = qd_to_numpy(engine.global_vertex_manager.body_id)
            for surface_vertex, face in pairs:
                point_body = body_ids[surface_vertices[surface_vertex]]
                face_body = body_ids[triangles[face, 0]]
                if point_body != face_body:
                    saw_cross_cloth_candidate = True
                    break

    assert saw_halfplane_contact
    assert saw_cross_cloth_candidate
    assert saw_friction_energy
    final_positions = qd_to_numpy(engine.fem.x)
    assert np.isfinite(final_positions).all()
    assert final_positions[:, 2].min() > 0.0

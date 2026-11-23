"""Tests for the entity naming system."""

import gc
import itertools
import os

import numpy as np
import pytest
import torch
import trimesh

import genesis as gs
import genesis.utils.geom as gu
import genesis.utils.point_cloud as pc
from genesis.utils.cache import disable_owned_cache_release, enable_owned_cache_release
from genesis.utils.misc import assign_indexed_tensor, broadcast_tensor, indices_to_mask, sanitize_index, tensor_to_array

from ..utils.assertions import assert_allclose, assert_equal


@pytest.mark.required
def test_scene_destroy_cleans_up_simulator():
    scene = gs.Scene(show_viewer=False)
    scene.add_entity(
        morph=gs.morphs.Plane(),
    )
    scene.build()
    scene.step()

    assert scene.is_built

    scene.destroy()

    assert not scene.is_built
    assert scene.visualizer is None


@pytest.mark.required
def test_scene_destroy_idempotent():
    scene = gs.Scene(show_viewer=False)
    scene.add_entity(
        morph=gs.morphs.Plane(),
    )
    scene.build()
    scene.step()

    scene.destroy()
    assert not scene.is_built

    scene.destroy()
    assert not scene.is_built


@pytest.mark.required
def test_rebuild_keeps_cached_data():
    scene = gs.Scene(show_viewer=False)
    scene.add_entity(
        morph=gs.morphs.Mesh(
            file="meshes/duck/duck.obj",
            scale=0.001,
        ),
    )
    scene.build()
    duck_verts = scene.entities[0].vgeoms[0].trimesh.vertices

    # A rebuilt scene shares the data of the destroyed one while its release is disabled, and loads it anew otherwise.
    # The destroyed scene sits in reference cycles, which only a garbage collection frees.
    for is_release_disabled in (True, False):
        if is_release_disabled:
            disable_owned_cache_release()
        scene.destroy()
        scene = gs.Scene(show_viewer=False)
        gc.collect()
        scene.add_entity(
            morph=gs.morphs.Mesh(
                file="meshes/duck/duck.obj",
                scale=0.001,
            ),
        )
        enable_owned_cache_release()
        assert np.shares_memory(scene.entities[0].vgeoms[0].trimesh.vertices, duck_verts) == is_release_disabled


@pytest.mark.required
@pytest.mark.parametrize("raise_before_build", [True, False])
def test_destroy_after_aborted_camera_build(monkeypatch, raise_before_build):
    from genesis.engine.sensors.camera import RasterizerCameraSensor

    scene = gs.Scene(show_viewer=False)
    camera = scene.add_sensor(
        gs.sensors.RasterizerCameraOptions(
            res=(64, 64),
        )
    )

    # The shared metadata outlives its entry in the sensor manager through this reference, so its fields stay readable
    shared_metadata = camera._shared_metadata

    # Inject a bug either at build entry (no metadata population) or after the original build populated the renderer,
    # the context, the sensors and the image cache.
    original_build = RasterizerCameraSensor.build

    def buggy_build(self):
        if not raise_before_build:
            original_build(self)
        raise RuntimeError("injected camera build failure")

    monkeypatch.setattr(RasterizerCameraSensor, "build", buggy_build)

    # Count the teardowns of the shared metadata through an instance-level shadow, which leaves the class and every other
    # metadata instance untouched, and is removed before any finalizer can fire.
    original_destroy = shared_metadata.destroy
    destroy_call_count = [0]

    def tracked_destroy():
        destroy_call_count[0] += 1
        original_destroy()

    shared_metadata.destroy = tracked_destroy
    try:
        # The aborted build is torn down at once, from whatever state it reached
        with pytest.raises(RuntimeError, match="injected camera build failure"):
            scene.build()
        assert destroy_call_count[0] == 1
        assert shared_metadata.renderer is None
        assert shared_metadata.context is None
        assert shared_metadata.sensors is None
        assert shared_metadata.image_cache is None

        # Destroying the scene afterwards tears nothing down twice
        scene.destroy()
    finally:
        del shared_metadata.destroy
    assert destroy_call_count[0] == 1


@pytest.mark.required
def test_auto_and_user_names():
    scene = gs.Scene()

    # Auto-generated name
    box = scene.add_entity(
        gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
        )
    )
    assert box.name.startswith("box_")

    # Multiple identical entities should have unique names
    box2 = scene.add_entity(
        gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
        )
    )
    assert box2.name.startswith("box_")
    assert box.name != box2.name

    # User-specified name
    sphere = scene.add_entity(
        gs.morphs.Sphere(
            radius=0.1,
        ),
        name="my_sphere",
    )
    assert sphere.name == "my_sphere"

    # Duplicate name raises error
    with pytest.raises(Exception, match="already exists"):
        scene.add_entity(
            gs.morphs.Cylinder(
                radius=0.1,
                height=0.2,
            ),
            name="my_sphere",
        )


@pytest.mark.required
def test_get_entity_by_name():
    scene = gs.Scene()

    box = scene.add_entity(
        gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
        ),
        name="test_box",
    )
    assert scene.get_entity(name="test_box") is box

    # Non-existent name raises error
    with pytest.raises(Exception, match="not found"):
        scene.get_entity(name="nonexistent")


@pytest.mark.required
def test_get_entity_by_uid():
    scene = gs.Scene()

    box = scene.add_entity(
        gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
        )
    )

    # Short UID lookup (7-character prefix shown in terminal)
    assert scene.get_entity(uid=box.uid.short()) is box

    # Non-existent UID raises error
    with pytest.raises(Exception, match="not found"):
        scene.get_entity(uid=gs.UID().short())


@pytest.mark.required
def test_entity_names_property():
    scene = gs.Scene()

    # Use "B" then "A" to confirm insertion order (not sorted)
    scene.add_entity(
        gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
        ),
        name="B",
    )
    scene.add_entity(
        gs.morphs.Sphere(
            radius=0.1,
        ),
        name="A",
    )
    assert tuple(scene.entity_names) == ("B", "A")


@pytest.mark.required
def test_urdf_mjcf_names_from_file():
    scene = gs.Scene()

    # URDF: plane.urdf has <robot name="plane">
    urdf_entity = scene.add_entity(
        gs.morphs.URDF(
            file="urdf/plane/plane.urdf",
        )
    )
    assert urdf_entity.name.startswith("plane_")

    # MJCF: panda.xml has <mujoco model="panda">
    mjcf_entity = scene.add_entity(
        gs.morphs.MJCF(
            file="xml/franka_emika_panda/panda.xml",
        )
    )
    assert mjcf_entity.name.startswith("panda_")

    # Multiple URDF entities should have unique names
    urdf_entity2 = scene.add_entity(
        gs.morphs.URDF(
            file="urdf/plane/plane.urdf",
        )
    )
    assert urdf_entity2.name.startswith("plane_")
    assert urdf_entity.name != urdf_entity2.name


@pytest.mark.required
def test_morph_orientation_offset_resolution():
    quat_90z = gu.xyz_to_quat(np.array((0.0, 0.0, 90.0)), rpy=True, degrees=True)

    # An unset offset resolves to identity, whether omitted or passed as the None unset sentinel.
    assert_equal(gs.morphs.Box(size=(0.1, 0.1, 0.1)).offset_quat, (1.0, 0.0, 0.0, 0.0))
    assert_equal(gs.morphs.Box(size=(0.1, 0.1, 0.1), offset_quat=None).offset_quat, (1.0, 0.0, 0.0, 0.0))

    # 'offset_euler' resolves into 'offset_quat'.
    assert_allclose(
        gs.morphs.Box(size=(0.1, 0.1, 0.1), offset_euler=(0.0, 0.0, 90.0)).offset_quat, quat_90z, tol=gs.EPS
    )

    # An explicit 'offset_quat' is kept verbatim.
    assert_equal(
        gs.morphs.Box(size=(0.1, 0.1, 0.1), offset_quat=(0.0, 1.0, 0.0, 0.0)).offset_quat, (0.0, 1.0, 0.0, 0.0)
    )

    # A None 'offset_quat' means unset, so a serializer can forward it unconditionally alongside 'offset_euler';
    # it resolves to 'offset_euler'.
    assert_allclose(
        gs.morphs.Box(size=(0.1, 0.1, 0.1), offset_euler=(0.0, 0.0, 90.0), offset_quat=None).offset_quat,
        quat_90z,
        tol=gs.EPS,
    )

    # 'offset_euler' and 'offset_quat' are mutually exclusive.
    with pytest.raises(Exception, match="'offset_euler' and 'offset_quat' cannot both be set"):
        gs.morphs.Box(size=(0.1, 0.1, 0.1), offset_euler=(0.0, 0.0, 90.0), offset_quat=(0.0, 1.0, 0.0, 0.0))


@pytest.mark.required
@pytest.mark.parametrize(
    ("index", "expected"),
    (
        (-1, (3,)),
        (-5, (-5,)),
        ([-5, -4, -1], (-5, 0, 3)),
        ([-4, -2], (0, 2)),
        ((-3, -1), (1, 3)),
        (np.array((-2, -1), dtype=np.int32), (2, 3)),
        (np.array((-5, -1), dtype=np.int32), (-5, 3)),
        (np.array((1, 3), dtype=np.uint32), (1, 3)),
        (np.array((-2.0, -1.0), dtype=np.float64), (2, 3)),
        (np.arange(4)[::-2], (3, 1)),
        (torch.tensor((-2, -1), dtype=torch.int32), (2, 3)),
        (torch.tensor((-5, -1), dtype=torch.int32), (-5, 3)),
        (torch.tensor((0, 3), dtype=torch.uint8), (0, 3)),
        (torch.tensor((-2.0, -1.0)), (2, 3)),
        (range(-4, 0), (0, 1, 2, 3)),
        (range(-5, 0), (-5, 0, 1, 2, 3)),
        (range(-2, 1), (2, 3, 0)),
        (slice(-2, None), (2, 3)),
        (slice(-1, None, -1), (3, 2, 1, 0)),
        (np.array((False, True, False, True)), (1, 3)),
        (torch.tensor((False, True, False, True)), (1, 3)),
    ),
)
def test_sanitize_index(index, expected):
    assert_equal(sanitize_index(index, -1, 4, 0, "index"), expected)


@pytest.mark.required
@pytest.mark.parametrize(
    "index",
    (
        {0, 1},
        [[0, 1]],
        ["a"],
        np.array((True,), dtype=bool),
        torch.tensor((True,), dtype=torch.bool),
    ),
)
def test_sanitize_index_rejects_invalid_collections(index):
    with pytest.raises(gs.GenesisException):
        sanitize_index(index, -1, 4, 0, "index")


@pytest.mark.required
def test_tensor_read_write_by_mixed_selection():
    tensor_ref = np.arange(20).reshape((4, 5))
    # Every form a selection of one axis takes, with the entries it selects. A trailing None leaves a single axis.
    selections = [
        (
            (None, np.arange(size)),
            (1, np.array((1,))),
            (slice(1, 4), np.arange(1, 4)),
            (range(0, size, 2), np.arange(0, size, 2)),
            ([3, 0, 2], np.array((3, 0, 2))),
            (np.array((3, 0, 2)), np.array((3, 0, 2))),
            (torch.tensor((3, 0, 2), device=gs.device), np.array((3, 0, 2))),
            (torch.tensor((True, False, True, True, False)[:size], device=gs.device), np.array((0, 2, 3))),
        )
        for size in tensor_ref.shape
    ]
    for (selection_0, idx_0), (selection_1, idx_1) in itertools.product(*selections):
        idx = np.ix_(idx_0, idx_1)
        tensor = torch.as_tensor(tensor_ref, dtype=gs.tc_float, device=gs.device)
        assert_equal(tensor[indices_to_mask(selection_0, selection_1, boolean_mask=False)], tensor_ref[idx])

        # A value is a scalar, broadcast along either axis, holds the selected entries, or holds every entry of the
        # axes selected by a mask, the selected entries being written in the order of the selection.
        n_0, n_1 = len(idx_0), len(idx_1)
        blocks = (np.arange(n_1), np.arange(n_0)[:, None], np.arange(n_0 * n_1).reshape((n_0, n_1)))
        values_and_blocks = [(7, 7), *[(block, block) for block in blocks]]
        is_mask = []
        for selection in (selection_0, selection_1):
            is_mask.append(isinstance(selection, torch.Tensor) and selection.dtype == torch.bool)
        if any(is_mask):
            full_value_shape, full_value_idx = [], []
            for size, axis_idx, is_axis_mask in zip(tensor_ref.shape, (idx_0, idx_1), is_mask):
                full_value_shape.append(size if is_axis_mask else len(axis_idx))
                full_value_idx.append(axis_idx if is_axis_mask else np.arange(len(axis_idx)))
            full_value = np.arange(np.prod(full_value_shape)).reshape(full_value_shape)
            values_and_blocks.append((full_value, full_value[np.ix_(*full_value_idx)]))
        mask = indices_to_mask(selection_0, selection_1)
        for dtype in (gs.tc_float, torch.uint32):
            for value, block in values_and_blocks:
                if isinstance(value, int):
                    value_tensor = value
                else:
                    value_tensor = torch.as_tensor(value, dtype=dtype, device=gs.device)
                if value is block:
                    assert_equal(broadcast_tensor(value_tensor, dtype, (n_0, n_1)), np.broadcast_to(block, (n_0, n_1)))
                tensor_written = torch.as_tensor(tensor_ref, dtype=dtype, device=gs.device)
                assign_indexed_tensor(tensor_written, mask, value_tensor)
                expected = tensor_ref.copy()
                expected[idx] = block
                assert_equal(tensor_written, expected)

    # A selection of two dimensions takes one row of entries per selected environment, or a row shared by all of them.
    # A write keeps the last occurrence of an entry repeated in a row, and the last row of a repeated environment.
    for selection_0, idx_0 in (*selections[0], (torch.tensor((3, 0, 3), device=gs.device), np.array((3, 0, 3)))):
        rows = (np.arange(len(idx_0))[:, None] + (4, 1, 4)) % 5
        for rows_idx in (rows, rows[:1]):
            idx_0_block, rows_block = np.broadcast_arrays(idx_0[:, None], rows_idx)
            mask = indices_to_mask(selection_0, rows_idx, boolean_mask=False, pair_rows=True)
            tensor = torch.as_tensor(tensor_ref, dtype=gs.tc_float, device=gs.device)
            assert_equal(tensor[mask], tensor_ref[idx_0_block, rows_block])
            for value in (7, np.arange(rows_block.size).reshape(rows_block.shape)):
                tensor = torch.as_tensor(tensor_ref, dtype=gs.tc_float, device=gs.device)
                value_tensor = value if isinstance(value, int) else torch.as_tensor(value, device=gs.device)
                assign_indexed_tensor(tensor, mask, value_tensor)
                expected = tensor_ref.copy()
                values_block = np.broadcast_to(value, rows_block.shape)
                for i in {i_0: i for i, i_0 in enumerate(idx_0)}.values():
                    for i_0, i_1, entry in zip(idx_0_block[i], rows_block[i], values_block[i]):
                        expected[i_0, i_1] = entry
                assert_equal(tensor, expected)


@pytest.mark.required
def test_init_preserves_torch_defaults():
    assert torch.get_default_device() == torch.device("cpu")
    assert torch.get_default_dtype() == torch.float32


@pytest.mark.required
def test_fps_algorithm_core():
    # Shape, dtype, determinism, anchor-on-no-seed, and invalid n_samples all in one test.
    points = np.random.default_rng(1).random((50, 3))
    out_a = pc.furthest_point_sample(points, 10, seed=42)
    out_b = pc.furthest_point_sample(points, 10, seed=42)
    assert out_a.shape == (10, 3)
    assert out_a.dtype == gs.np_float
    assert_equal(out_a, out_b)

    # With seed=None the first sample is the first input point (deterministic anchor).
    anchor = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    out_anchor = pc.furthest_point_sample(anchor, 3, seed=None)
    assert_allclose(out_anchor[0], anchor[0], tol=1e-5)

    with pytest.raises(gs.GenesisException):
        pc.furthest_point_sample(np.zeros((5, 3)), 10, seed=None)


@pytest.mark.required
def test_fps_mesh_sampling_end_to_end():
    # Shape/dtype/determinism, on-surface points, minimum separation, and box-aligned normals all in one scene.
    mesh = trimesh.creation.box((1.0, 1.0, 1.0))
    points_a = pc.sample_mesh_point_cloud(mesh.vertices, mesh.faces, 16, n_candidates=400, seed=7, use_cache=False)
    points_b = pc.sample_mesh_point_cloud(mesh.vertices, mesh.faces, 16, n_candidates=400, seed=7, use_cache=False)
    assert points_a.shape == (16, 3)
    assert points_a.dtype == gs.np_float
    assert_equal(points_a, points_b)

    _, dist, _ = mesh.nearest.on_surface(points_a)
    assert dist.max() < 1e-5

    pairwise = np.linalg.norm(points_a[:, None, :] - points_a[None, :, :], axis=-1)
    np.fill_diagonal(pairwise, np.inf)
    assert pairwise.min() > 0.2

    # Box face normals are axis-aligned unit vectors. Unit-length + `max(|n|) == 1` is sufficient: by Pythagoras the
    # other two components must be ~0.
    _, normals = pc.sample_mesh_point_cloud(
        mesh.vertices, mesh.faces, 32, n_candidates=800, seed=0, use_cache=False, return_normals=True
    )
    assert_allclose(np.linalg.norm(normals, axis=1), 1.0, tol=1e-5)
    assert_allclose(np.abs(normals).max(axis=1), 1.0, tol=1e-4)


@pytest.mark.cache(False)
@pytest.mark.required
def test_fps_cache_round_trip():
    mesh = trimesh.creation.box((1.0, 1.0, 1.0))
    # Samples lie on the surface, so they follow any move of the vertices. Perturbing the mesh the way re-exporting it
    # would, then getting the very samples of the original back, is what proves they were read from the cache.
    reexported_verts = mesh.vertices + np.random.uniform(-1e-7, 1e-7, mesh.vertices.shape)
    # Run both `return_normals` paths: first call writes the cache; the others hit it; outputs identical.
    for return_normals in (True, False):
        kwargs = dict(
            faces=mesh.faces,
            n_points=5,
            n_candidates=10,
            return_normals=return_normals,
            seed=7,
        )
        first = pc.sample_mesh_point_cloud(verts=mesh.vertices, **kwargs, use_cache=True)
        assert_equal(first, pc.sample_mesh_point_cloud(verts=mesh.vertices, **kwargs, use_cache=True))
        assert_equal(first, pc.sample_mesh_point_cloud(verts=reexported_verts, **kwargs, use_cache=True))


@pytest.mark.required
def test_gs_mesh_sample_point_cloud_wrapper():
    mesh = trimesh.creation.box((0.2, 0.4, 0.6))
    gmesh = gs.Mesh.from_trimesh(mesh)
    points, normals = gmesh.sample_point_cloud(10, n_candidates=300, seed=67, use_cache=False, return_normals=True)
    assert points.shape == (10, 3)
    assert normals.shape == (10, 3)
    _, dist, _ = mesh.nearest.on_surface(points)
    assert dist.max() < 1e-5
    assert_allclose(np.linalg.norm(normals, axis=1), 1.0, tol=1e-5)


@pytest.mark.required
def test_solver_state_change_trackers(show_viewer):
    # Imported lazily: the solver package pulls in quadrants kernels that need gs.qd_float, set only by gs.init.
    from genesis.engine.state_change import StateChange, Subscriber

    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(2.0, -2.0, 1.5),
            camera_lookat=(0.0, 0.0, 0.5),
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(gs.morphs.Plane())
    cube = scene.add_entity(
        gs.morphs.Box(
            size=(0.2, 0.2, 0.2),
            pos=(0.0, 0.0, 0.5),
        ),
    )
    scene.build(n_envs=2)

    solver = scene.rigid_solver

    # An eager subscriber is called back inside the setter, with the environments it was given
    eager_events = []
    solver.changelog.subscribe(
        Subscriber(
            to=frozenset({StateChange.GEOMETRY}),
            callback=lambda change, **indices: eager_events.append((change, indices["envs_idx"])),
        )
    )
    # A tracker reports every environment modified until its first pull, then the changes of its categories since its
    # last pull and the environments they reached
    geometry_tracker = solver.changelog.track(StateChange.GEOMETRY)
    dynamics_tracker = solver.changelog.track(StateChange.DYNAMICS)
    assert geometry_tracker.is_modified()
    assert geometry_tracker.pull().envs is None
    dynamics_tracker.pull()
    assert not geometry_tracker.is_modified()
    assert geometry_tracker.pull() is None

    cube.set_pos([[0.0, 0.0, 1.0], [0.0, 0.0, 2.0]], zero_velocity=False)
    assert eager_events == [(StateChange.GEOMETRY, None)]
    assert not dynamics_tracker.is_modified()
    record = geometry_tracker.pull()
    assert record.changes == frozenset({StateChange.GEOMETRY})
    assert record.envs is None
    assert dynamics_tracker.pull() is None
    assert geometry_tracker.pull() is None

    # A setter targeting some environments reports them alone
    cube.set_pos([[0.0, 0.0, 3.0]], envs_idx=[1], zero_velocity=False)
    assert eager_events[1] == (StateChange.GEOMETRY, [1])
    assert_equal(tensor_to_array(geometry_tracker.pull().envs), [False, True])

    # A velocity setter changes DYNAMICS alone, and a default set_pos both categories
    cube.set_dofs_velocity(0.0)
    assert geometry_tracker.pull() is None
    assert dynamics_tracker.pull().changes == frozenset({StateChange.DYNAMICS})

    # Reads record nothing
    solver.get_links_pos()
    assert geometry_tracker.pull() is None

    geometry_dynamics_tracker = solver.changelog.track(StateChange.GEOMETRY, StateChange.DYNAMICS)
    geometry_dynamics_tracker.pull()
    cube.set_pos([[0.0, 0.0, 4.0], [0.0, 0.0, 5.0]])
    assert geometry_dynamics_tracker.is_modified(StateChange.DYNAMICS)
    assert geometry_dynamics_tracker.pull().changes == frozenset({StateChange.GEOMETRY, StateChange.DYNAMICS})

    # A setter modifies the solver until the next step commits it, which reports every environment modified
    assert solver.changelog.is_modified(StateChange.GEOMETRY)
    scene.step()
    assert not solver.changelog.is_modified()
    assert geometry_tracker.pull().envs is None

    # A whole-state restore reaches every environment
    scene.reset()
    assert geometry_tracker.pull().envs is None


@pytest.mark.parametrize("backend", [None])
def test_per_solver_gravity():
    # Field-backed storage is what a solver whose kernels reach gravity through the solver itself holds, and the array
    # mode is forced to fields here so both kinds of solver are on that storage at once.
    os.environ["GS_ENABLE_NDARRAY"] = "0"
    try:
        gs.init(backend=gs.cpu, seed=0)

        scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                gravity=(0.0, 0.0, -9.81),
            ),
            show_viewer=False,
        )
        scene.add_entity(gs.morphs.Plane())
        scene.add_entity(
            gs.morphs.Box(
                size=(0.4, 0.4, 0.4),
                pos=(0.0, 0.0, 0.5),
            )
        )
        scene.add_entity(
            gs.morphs.Sphere(
                pos=(0.0, 0.0, 0.5),
                radius=0.1,
            ),
            material=gs.materials.MPM.Liquid(),
        )
        scene.build(n_envs=2)

        new_gravity = [0.0, 0.0, -5.0]

        rigid = scene.rigid_solver
        rigid.set_gravity(new_gravity)
        assert_allclose(rigid.get_gravity(), new_gravity, atol=1e-6)

        mpm = scene.mpm_solver
        mpm.set_gravity(new_gravity)
        assert_allclose(mpm.get_gravity(), new_gravity, atol=1e-6)

        # Gravity is per solver, so setting one leaves the other where it was.
        rigid.set_gravity(0.0)
        assert_allclose(rigid.get_gravity(), 0.0, atol=1e-6)
        assert_allclose(mpm.get_gravity(), new_gravity, atol=1e-6)

        # It is per environment too, so naming one leaves the others where they were.
        rigid.set_gravity([0.0, 0.0, -1.0], envs_idx=1)
        assert_allclose(rigid.get_gravity(envs_idx=0), 0.0, atol=1e-6)
        assert_allclose(rigid.get_gravity(envs_idx=1), [0.0, 0.0, -1.0], atol=1e-6)

    finally:
        gs.destroy()
        os.environ.pop("GS_ENABLE_NDARRAY", None)


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 3])
def test_per_env_time(show_viewer, n_envs, tol):
    DT = 0.01
    N_STEPS = 10
    N_MORE_STEPS = 5

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=DT,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.5, 0.0, 1.0),
            camera_lookat=(0.0, 0.0, 0.5),
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(
        morph=gs.morphs.Sphere(
            pos=(0.0, 0.0, 1.0),
        ),
    )
    scene.build(n_envs=n_envs)

    for _ in range(N_STEPS):
        scene.step()

    # A scene of no parallel environments has one clock, reported as the scalar the whole scene runs on.
    if n_envs == 0:
        assert_equal(scene.get_time().ndim, 0)
        assert_allclose(scene.get_time(), N_STEPS * DT, tol=tol)
        scene.reset()
        assert_allclose(scene.get_time(), 0.0, tol=tol)
        return

    # Environments reset independently, so resetting one rewinds its clock alone while the others keep theirs, whether
    # it is named by an index counted from either end, by a selection of several, or by a mask.
    scene.reset(envs_idx=-2)
    assert_allclose(scene.get_time(), [N_STEPS * DT, 0.0, N_STEPS * DT], tol=tol)

    for _ in range(N_MORE_STEPS):
        scene.step()
    times_expected = [(N_STEPS + N_MORE_STEPS) * DT, N_MORE_STEPS * DT, (N_STEPS + N_MORE_STEPS) * DT]
    assert_allclose(scene.get_time(), times_expected, tol=tol)
    assert_allclose(scene.get_time(envs_idx=1), N_MORE_STEPS * DT, tol=tol)

    scene.reset(envs_idx=(0, 2))
    assert_allclose(scene.get_time(), [0.0, N_MORE_STEPS * DT, 0.0], tol=tol)

    for _ in range(N_MORE_STEPS):
        scene.step()
    scene.reset(envs_idx=torch.tensor((False, True, False), device=gs.device))
    assert_allclose(scene.get_time(), [N_MORE_STEPS * DT, 0.0, N_MORE_STEPS * DT], tol=tol)


@pytest.mark.required
def test_derived_substeps(show_viewer, tol):
    GRAVITY = -9.81
    N_STEPS = 10
    SUBSTEPS = 5

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
            gravity=(0.0, 0.0, GRAVITY),
        ),
        rigid_options=gs.options.RigidOptions(
            dt=0.01 / SUBSTEPS,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.5, 0.0, 1.0),
            camera_lookat=(0.0, 0.0, 0.5),
        ),
        show_viewer=show_viewer,
    )
    ball = scene.add_entity(
        morph=gs.morphs.Sphere(
            pos=(0.0, 0.0, 1.0),
        ),
    )
    scene.add_entity(
        morph=gs.morphs.Box(
            pos=(1.0, 0.0, 1.0),
            size=(0.02, 0.02, 0.02),
        ),
        material=gs.materials.PBD.Particle(),
    )
    scene.build()

    # A solver integrates at the interval it was given, as many times per scene step as that interval divides the
    # step dt, and every solver reads the same resolution.
    solver = scene.rigid_solver
    assert_equal(solver.substeps, SUBSTEPS)
    assert_allclose(solver.substep_dt, 0.01 / SUBSTEPS, tol=gs.EPS)

    # A step advances the step dt whatever the rate: the fall accumulates once per substep, at the solver dt, for as
    # many substeps as the ratio gives.
    for _ in range(N_STEPS):
        scene.step()
    n_substeps = N_STEPS * solver.substeps
    expected_z = 1.0 + GRAVITY * solver.substep_dt**2 * n_substeps * (n_substeps + 1) / 2
    assert_allclose(ball.get_pos()[2], expected_z, tol=tol)

    # The substep loop advances every active solver once per iteration, so the interval one of them asks for is the
    # rate they all take, whether they asked for one or not.
    assert_equal(scene.pbd_solver.substeps, SUBSTEPS)

    # The substeps shorthand decides the rate of every solver when none asks for one of its own
    scene_from_substeps = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
            gravity=(0.0, 0.0, GRAVITY),
            substeps=SUBSTEPS,
        ),
        show_viewer=False,
    )
    ball_from_substeps = scene_from_substeps.add_entity(
        morph=gs.morphs.Sphere(
            pos=(0.0, 0.0, 1.0),
        ),
    )
    scene_from_substeps.add_entity(
        morph=gs.morphs.Box(
            pos=(1.0, 0.0, 1.0),
            size=(0.02, 0.02, 0.02),
        ),
        material=gs.materials.PBD.Particle(),
    )
    scene_from_substeps.build()
    assert_equal(scene_from_substeps.rigid_solver.substeps, SUBSTEPS)
    assert_allclose(scene_from_substeps.rigid_solver.substep_dt, 0.01 / SUBSTEPS, tol=gs.EPS)

    # The constraint time constant is floored against the substep the rigid solver integrates at, so the contacts are
    # as stiff as in a scene whose rigid solver asks for that rate itself.
    timeconst, *_ = ball_from_substeps.geoms[0].get_sol_params()
    timeconst_ref, *_ = ball.geoms[0].get_sol_params()
    assert timeconst + gs.EPS >= 2.0 * scene_from_substeps.rigid_solver.substep_dt
    assert_allclose(timeconst, timeconst_ref, tol=gs.EPS)

    # The rate reaches what integrates, not only what reports it: a solver settles its buffers as it builds, so a rate
    # settled afterwards would leave it stepping the scene interval as many times as the shorter one.
    for _ in range(N_STEPS):
        scene_from_substeps.step()
    n_substeps = N_STEPS * SUBSTEPS
    expected_z = 1.0 + GRAVITY * scene_from_substeps.rigid_solver.substep_dt**2 * n_substeps * (n_substeps + 1) / 2
    assert_allclose(ball_from_substeps.get_pos()[2], expected_z, tol=tol)

    # A differentiable scene replays the substeps of a step, as many as the rate a solver dt derives: the gradient of
    # the position over one step reaches the velocity it started from through every one of them.
    scene_differentiable = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
            gravity=(0.0, 0.0, GRAVITY),
            requires_grad=True,
        ),
        rigid_options=gs.options.RigidOptions(
            dt=0.01 / SUBSTEPS,
        ),
        show_viewer=False,
    )
    ball_differentiable = scene_differentiable.add_entity(
        morph=gs.morphs.Sphere(
            pos=(0.0, 0.0, 1.0),
        ),
    )
    scene_differentiable.build()
    assert_equal(scene_differentiable.rigid_solver.substeps, SUBSTEPS)
    velocity = torch.tensor([0.0, 0.0, 0.5], dtype=gs.tc_float, device=gs.device, requires_grad=True)
    ball_differentiable.set_dofs_velocity(velocity, dofs_idx_local=slice(0, 3))
    scene_differentiable.step()
    ball_differentiable.get_pos()[2].backward()
    assert_allclose(velocity.grad, (0.0, 0.0, 0.01), tol=tol)

    # Two solvers asking for intervals that imply different counts have no one rate to advance at, and are rejected.
    with pytest.raises(gs.GenesisException, match="Solvers integrating at different rates"):
        scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=0.01),
            rigid_options=gs.options.RigidOptions(dt=0.005),
            pbd_options=gs.options.PBDOptions(dt=0.0025),
            show_viewer=False,
        )
        scene.add_entity(gs.morphs.Sphere(pos=(0.0, 0.0, 1.0)))
        scene.add_entity(gs.morphs.Box(pos=(1.0, 0.0, 1.0), size=(0.02, 0.02, 0.02)), gs.materials.PBD.Particle())
        scene.build()

    # A rate is settled for the solvers the entities made active, and only for those: what an interval would imply for
    # a solver nothing is simulated with is no reason to refuse the scene it sits in.
    scene_inactive = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
        ),
        rigid_options=gs.options.RigidOptions(
            dt=0.003,
        ),
        show_viewer=False,
    )
    scene_inactive.build()
    assert_equal(scene_inactive.sim.substeps, 1)
    assert_allclose(scene_inactive.sim.substep_dt, 0.01, tol=gs.EPS)
    # A scene holding no active solver steps and reports no env in error
    scene_inactive.step()
    assert_equal(scene_inactive.get_error_envs_mask(), False)

    for interval in (0.003, 0.02):
        with pytest.raises(gs.GenesisException, match="does not divide the step dt"):
            scene = gs.Scene(
                sim_options=gs.options.SimOptions(dt=0.01),
                rigid_options=gs.options.RigidOptions(dt=interval),
                show_viewer=False,
            )
            scene.add_entity(gs.morphs.Sphere(pos=(0.0, 0.0, 1.0)))
            scene.build()

    with pytest.raises(gs.GenesisException, match="conflicting with the requested substeps"):
        scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=0.01, substeps=4),
            rigid_options=gs.options.RigidOptions(dt=0.002),
            show_viewer=False,
        )
        scene.add_entity(gs.morphs.Sphere(pos=(0.0, 0.0, 1.0)))
        scene.build()


@pytest.mark.required
@pytest.mark.parametrize(
    "boxes_size",
    [
        # A light body under a heavy scene aggregate, below the floor quoted on the solver tolerance.
        [1.0, 0.001],
        # A uniformly tiny scene, below the absolute floor the working precision resolves at all.
        pytest.param([0.0002], marks=pytest.mark.precision("32")),
    ],
)
def test_warn_solver_numerical_stability(boxes_size, caplog):
    scene = gs.Scene(show_viewer=False)
    for i, size in enumerate(boxes_size):
        scene.add_entity(
            gs.morphs.Box(
                pos=(float(i), 0.0, 0.5 * size),
                size=(size, size, size),
            ),
        )
    with caplog.at_level("WARNING"):
        scene.build()
    assert any("too small for the constraint solver" in record.message for record in caplog.records)

import os

import igl
import numpy as np
import pytest
import torch
import trimesh

import genesis as gs
import genesis.utils.geom as gu
import genesis.utils.terrain as tu
from genesis.utils.misc import get_assets_dir, tensor_to_array

from ..utils.assertions import assert_allclose


@pytest.mark.parametrize(
    "backend",
    [
        pytest.param(gs.gpu, marks=pytest.mark.required),
        gs.cpu,  # This test takes too much time of CPU (~1000s)
    ],
)
@pytest.mark.parametrize("is_named", [True, False])
def test_generation(is_named, show_viewer, tol):
    TERRAIN_PATTERN = [
        ["flat_terrain", "flat_terrain", "flat_terrain", "flat_terrain", "flat_terrain"],
        ["flat_terrain", "fractal_terrain", "random_uniform_terrain", "sloped_terrain", "flat_terrain"],
        ["flat_terrain", "pyramid_sloped_terrain", "discrete_obstacles_terrain", "wave_terrain", "flat_terrain"],
        ["flat_terrain", "stairs_terrain", "pyramid_stairs_terrain", "stepping_stones_terrain", "flat_terrain"],
        ["flat_terrain", "flat_terrain", "flat_terrain", "flat_terrain", "flat_terrain"],
    ]
    TERRAIN_OFFSET = (10.0, -10.0, -1.0)
    TERRAIN_SIZE = 10.0
    SUBTERRAIN_GRID_SIZE = 15
    OBJ_SIZE = 0.1
    OBJ_HEIGHT_INIT = 0.3
    NUM_OBJ_SQRT = 15

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.006,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(-5.0 + TERRAIN_OFFSET[0], -5.0 + TERRAIN_OFFSET[1], 10.0 + TERRAIN_OFFSET[2]),
            camera_lookat=(5.0 + TERRAIN_OFFSET[0], 5.0 + TERRAIN_OFFSET[1], 0.0 + TERRAIN_OFFSET[2]),
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )
    terrain_kwargs = dict(
        pos=TERRAIN_OFFSET,
        n_subterrains=(len(TERRAIN_PATTERN),) * 2,
        subterrain_size=(TERRAIN_SIZE / len(TERRAIN_PATTERN),) * 2,
        horizontal_scale=TERRAIN_SIZE / len(TERRAIN_PATTERN) / SUBTERRAIN_GRID_SIZE,
        vertical_scale=0.05,
        subterrain_types=TERRAIN_PATTERN,
        randomize=False,
        name="my_terrain" if is_named else None,
    )
    # FIXME: Collision detection is very unstable for 'stepping_stones' pattern.
    terrain = scene.add_entity(
        morph=gs.morphs.Terrain(**terrain_kwargs),
    )
    obj = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(1.0, 1.0, 1.0),
            size=(0.1, 0.1, 0.1),
        ),
        surface=gs.surfaces.Default(
            color=(1.0, 0.0, 0.0, 1.0),
        ),
    )
    scene.build(n_envs=NUM_OBJ_SQRT**2)

    # Spread objects across the entire field
    obj_pos_1d = torch.linspace(OBJ_SIZE / 2, TERRAIN_SIZE - OBJ_SIZE / 2, NUM_OBJ_SQRT)
    obj_pos_init_rel = torch.cartesian_prod(*(obj_pos_1d,) * 2, torch.tensor((OBJ_HEIGHT_INIT,)))
    obj.set_pos(obj_pos_init_rel + torch.tensor(TERRAIN_OFFSET))

    # Drop the objects and simulate for a while.
    for _ in range(600):
        scene.step()

    # Check that objects are not moving anymore
    assert_allclose(obj.get_vel(), 0.0, tol=0.1)

    # Check the the terrain is not entirely flat and has the expected size
    terrain_min_corner, terrain_max_corner = tensor_to_array(terrain.geoms[0].get_AABB()) - TERRAIN_OFFSET
    assert_allclose(terrain_min_corner[:2], 0.0, tol=gs.EPS)
    assert_allclose(terrain_max_corner[:2], TERRAIN_SIZE, tol=gs.EPS)
    assert terrain_min_corner[2] < -1.0  # Stepping stone depth
    assert terrain_max_corner[2] > 0.01  # FIXME: It should not be larger than 'vertical_scale'

    # Check that all objects are in contact with the terrain
    obj_pos = tensor_to_array(obj.get_pos()) - TERRAIN_OFFSET
    terrain_mesh = terrain.geoms[0].mesh
    signed_distance, *_ = igl.signed_distance(obj_pos, terrain_mesh.verts, terrain_mesh.faces)
    assert (signed_distance > 0.0).all()
    assert (signed_distance < 2 * OBJ_SIZE).all()

    # Check if cache is being reloaded as expected
    if is_named:
        scene = gs.Scene()
        terrain_2 = scene.add_entity(
            morph=gs.morphs.Terrain(**{**terrain_kwargs, **dict(randomize=True)}),
        )
        terrain_2_mesh = terrain_2.geoms[0].mesh
        assert_allclose(terrain_mesh.verts, terrain_2_mesh.verts, tol=tol)


@pytest.mark.required
def test_discrete_obstacles():
    scene = gs.Scene()
    terrain = scene.add_entity(
        gs.morphs.Terrain(
            n_subterrains=(1, 1),
            subterrain_size=(6.0, 6.0),
            horizontal_scale=0.5,
            vertical_scale=0.5,
            subterrain_types=[["discrete_obstacles_terrain"]],
            subterrain_parameters={
                "discrete_obstacles_terrain": {
                    "max_height": 1.0,
                    "platform_size": 1.0,
                }
            },
        )
    )
    scene.build()
    height_field = terrain.geoms[0].metadata["height_field"]
    platform = height_field[5:7, 5:7]

    assert height_field.max() == 2.0
    assert height_field.min() == -2.0
    assert (platform < gs.EPS).all()


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
def test_get_terrain_height(n_envs, tol):
    height_field = np.array(
        [
            [0.0, 4.0, 9.0, 15.0],
            [10.0, 20.0, 33.0, 49.0],
            [30.0, 50.0, 73.0, 99.0],
        ],
    )
    quat_yaw = gu.xyz_to_quat(np.array((0.0, 0.0, 90.0)), degrees=True)

    scene = gs.Scene(show_viewer=False)
    terrain = scene.add_entity(
        morph=gs.morphs.Terrain(
            pos=(2.0, -1.0, 0.3),
            quat=quat_yaw,
            batch_fixed_verts=True,
            horizontal_scale=0.5,
            vertical_scale=0.1,
            height_field=height_field,
        ),
    )
    visual_terrain = scene.add_entity(
        morph=gs.morphs.Terrain(
            pos=(30.0, 40.0, 3.0),
            quat=gu.xyz_to_quat(np.array((0.0, 0.0, 37.0)), degrees=True),
            collision=False,
            horizontal_scale=0.25,
            vertical_scale=0.01,
            height_field=np.add.outer(np.arange(3), 2 * np.arange(3)),
        ),
    )
    scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
        ),
    )
    scene.build(n_envs=n_envs)

    initial_height = terrain.get_terrain_height((2.0, -0.5))
    assert_allclose(initial_height, 1.3, tol=tol)

    visual_pose = gu.trans_quat_to_T(np.array(visual_terrain.morph.pos), np.array(visual_terrain.morph.quat))
    visual_position = visual_pose @ np.array((0.5, 0.25, 0.0, 1.0))
    visual_height = visual_terrain.get_terrain_height(visual_position[:2])
    assert_allclose(visual_height, 3.04, tol=tol)

    if n_envs:
        shared_positions = terrain.get_terrain_height((((2.0, -1.0), (2.0, -0.5)),))
        assert_allclose(shared_positions, ((0.3, 1.3), (0.3, 1.3)), tol=tol)

        terrain.set_pos(((10.0, 20.0, 1.0), (-3.0, 4.0, -2.0)), relative=False)
        terrain.set_quat(np.stack((gu.identity_quat(), quat_yaw)), relative=False)
    else:
        terrain.set_pos((10.0, 20.0, 1.0), relative=False)
        terrain.set_quat((1.0, 0.0, 0.0, 0.0), relative=False)

    positions = (
        (10.0, 20.0),
        (10.5, 20.0),
        (10.0, 20.5),
        (10.125, 20.25),
        (10.375, 20.25),
        (11.0, 21.5),
    )
    terrain_mesh = terrain.geoms[0].get_trimesh().copy()
    terrain_mesh.apply_transform(gu.trans_quat_to_T(np.array((10.0, 20.0, 1.0)), gu.identity_quat()))
    ray_origins = np.column_stack((positions, np.full(len(positions), terrain_mesh.bounds[1, 2] + 1.0)))
    ray_directions = np.tile((0.0, 0.0, -1.0), (len(positions), 1))
    locations, ray_ids, _ = terrain_mesh.ray.intersects_location(
        ray_origins=ray_origins, ray_directions=ray_directions, multiple_hits=False
    )
    expected = locations[np.argsort(ray_ids), 2]
    if n_envs:
        heights = terrain.get_terrain_height(positions, envs_idx=[0])

        positions_per_env = ((positions[3], positions[4]), ((-3.25, 4.125), (-3.25, 4.375)))
        heights_per_env = terrain.get_terrain_height(positions_per_env)
        assert_allclose(heights_per_env, ((1.45, 2.1), (-1.55, -0.9)), tol=tol)

        one_position_per_env = terrain.get_terrain_height(((positions_per_env[0][0],), (positions_per_env[1][0],)))
        assert_allclose(one_position_per_env, ((1.45,), (-1.55,)), tol=tol)

        heights_env_1 = terrain.get_terrain_height(positions_per_env[1], envs_idx=1)
        assert_allclose(heights_env_1, (-1.55, -0.9), tol=tol)
    else:
        heights = terrain.get_terrain_height(positions)

    assert_allclose(heights, expected, tol=1e-6)

    envs_idx = [0] if n_envs else None
    boundary_heights = terrain.get_terrain_height(((9.5, 20.0), (9.499, 20.0), (np.nan, 20.0)), envs_idx=envs_idx)
    assert_allclose(boundary_heights[..., 0], 1.0, tol=tol)
    assert torch.isnan(boundary_heights[..., 1:]).all()

    with pytest.raises(gs.GenesisException):
        terrain.get_terrain_height((10.0, 20.0, 1.0), envs_idx=envs_idx)

    terrain.set_quat(gu.xyz_to_quat(np.array((5e-4, 0.0, 0.0))), envs_idx=envs_idx, relative=False)
    height = terrain.get_terrain_height((10.0, 20.0), envs_idx=envs_idx)
    assert_allclose(height, 1.0, tol=tol)

    terrain.set_quat(gu.xyz_to_quat(np.array((2e-3, 0.0, 0.0))), envs_idx=envs_idx, relative=False)
    height = terrain.get_terrain_height((10.0, 20.0), envs_idx=envs_idx)
    assert torch.isnan(height).all()


@pytest.mark.required
def test_subterrain_parameters(show_viewer):
    scene_ref = gs.Scene(show_viewer=show_viewer)
    terrain_ref = scene_ref.add_entity(
        morph=gs.morphs.Terrain(
            n_subterrains=(2, 2),
            subterrain_size=(12.0, 12.0),
            horizontal_scale=0.25,
            subterrain_types="wave_terrain",
        )
    )

    height_ref = terrain_ref.geoms[0].metadata["height_field"]

    scene_test = gs.Scene(show_viewer=show_viewer)
    terrain_test = scene_test.add_entity(
        morph=gs.morphs.Terrain(
            n_subterrains=(2, 2),
            subterrain_size=(12.0, 12.0),
            horizontal_scale=0.25,
            subterrain_types="wave_terrain",
            subterrain_parameters={"wave_terrain": {"amplitude": 0.2}},
        )
    )

    height_test = terrain_test.geoms[0].metadata["height_field"]

    assert_allclose((height_ref * 2.0), height_test, tol=gs.EPS)


def test_mesh_to_heightfield(tmp_path, show_viewer):
    horizontal_scale = 2.0
    path_terrain = os.path.join(get_assets_dir(), "meshes", "terrain_45.obj")

    hf_terrain, xs, ys = tu.mesh_to_heightfield(path_terrain, spacing=horizontal_scale, oversample=1)

    # default heightfield starts at 0, 0, 0
    # translate to the center of the mesh
    translation = np.array([np.nanmin(xs), np.nanmin(ys), 0])

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            gravity=(5, 0, -5),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0, -5, 7),
            camera_lookat=(10, 15, 4),
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )
    terrain_heightfield = scene.add_entity(
        morph=gs.morphs.Terrain(
            horizontal_scale=horizontal_scale,
            vertical_scale=1.0,
            height_field=hf_terrain,
            pos=translation,
        ),
        vis_mode="collision",
    )
    ball = scene.add_entity(
        gs.morphs.Sphere(
            pos=(10, 15, 7),
            radius=1,
        ),
        vis_mode="collision",
    )
    scene.build()

    for i in range(70):
        scene.step()

    # The ball is at rest (on the terrain)
    assert_allclose(ball.get_dofs_velocity(), 0, tol=1e-3)


@pytest.mark.required
def test_box_on_terrain_no_spurious_spin(show_viewer):
    BOX_SIZE = (0.12, 0.06, 0.025)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.005,
        ),
        rigid_options=gs.options.RigidOptions(
            box_box_detection=False,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3.5, -3.5, 2.0),
            camera_lookat=(0.0, 0.0, 0.0),
            camera_fov=30,
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )
    scene.add_entity(
        morph=gs.morphs.Terrain(
            pos=(-1.5, -1.5, 0.0),
            height_field=np.zeros((4, 4), dtype=np.float32),
            horizontal_scale=1.0,
        ),
        surface=gs.surfaces.Default(
            color=(0.4, 0.8, 0.4),
        ),
    )
    box = scene.add_entity(
        morph=(
            gs.morphs.Box(size=BOX_SIZE),
            gs.morphs.MeshSet(
                files=(trimesh.creation.box(extents=BOX_SIZE),),
                decimate=False,
            ),
        ),
        surface=gs.surfaces.Default(
            color=(0.95, 0.2, 0.2),
        ),
    )

    # 4x4 grid spread across the 3 m x 3 m terrain.
    grid = np.linspace(-1.2, 1.2, 4)
    xy = np.stack(np.meshgrid(grid, grid, indexing="ij"), axis=-1).reshape(-1, 2)
    n_envs = xy.shape[0]
    scene.build(n_envs=n_envs)

    z = BOX_SIZE[2] / 2
    pos = np.concatenate([xy, np.full((n_envs, 1), z)], axis=-1).astype(np.float32)
    box.set_pos(torch.from_numpy(pos))
    box.set_dofs_velocity(torch.zeros((n_envs, 6)))
    quat_initial = tensor_to_array(box.get_quat())

    for _ in range(500):
        scene.step()

    quat_delta = gu.transform_quat_by_quat(tensor_to_array(box.get_quat()), gu.inv_quat(quat_initial))
    assert_allclose(gu.quat_to_rotvec(quat_delta), 0.0, tol=0.02)


@pytest.mark.required
def test_multicontact_sphere_vs_terrain(show_viewer, tol):
    GRID_N = 13
    APEX_IDX = GRID_N // 2
    VERTICAL_SCALE = 0.04
    HORIZONTAL_SCALE = 0.05
    SPHERE_RADIUS = 0.1

    ii, jj = np.meshgrid(np.arange(GRID_N), np.arange(GRID_N), indexing="ij")
    hf = (np.abs(ii - APEX_IDX) + np.abs(jj - APEX_IDX)).astype(np.int16)
    terrain_pos = (-APEX_IDX * HORIZONTAL_SCALE, -APEX_IDX * HORIZONTAL_SCALE, 0.0)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.0, -1.0, 0.8),
            camera_lookat=(0.0, 0.0, 0.0),
            camera_fov=30,
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )
    scene.add_entity(
        morph=gs.morphs.Terrain(
            pos=terrain_pos,
            height_field=hf,
            horizontal_scale=HORIZONTAL_SCALE,
            vertical_scale=VERTICAL_SCALE,
        ),
        surface=gs.surfaces.Default(
            color=(0.4, 0.8, 0.4),
        ),
    )
    sphere = scene.add_entity(
        morph=gs.morphs.Sphere(
            radius=SPHERE_RADIUS,
        ),
        surface=gs.surfaces.Default(
            color=(0.95, 0.2, 0.2),
        ),
        visualize_contact=True,
    )

    scene.build()

    sphere.set_pos(torch.tensor([0.0, 0.0, 0.25]))

    for _ in range(80):
        scene.step()
        print(tensor_to_array(sphere.get_dofs_velocity()))

    # Sphere is at rest at the apex of the pit. Equilibrium height is set by the four-wall solid angle: contacts on the
    # opposing pyramid walls push the sphere upward, so it sits above the apex vertex but well below the rim.
    pos_final = tensor_to_array(sphere.get_pos())
    assert_allclose(pos_final[:2], 0.0, tol=2.0 * HORIZONTAL_SCALE)
    assert SPHERE_RADIUS < pos_final[2] < APEX_IDX * VERTICAL_SCALE + SPHERE_RADIUS

    vel_final = tensor_to_array(sphere.get_dofs_velocity())
    assert_allclose(vel_final, 0.0, tol=1e-5)

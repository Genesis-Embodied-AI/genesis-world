import textwrap

import numpy as np
import pytest
import torch

import genesis as gs
from genesis.utils.misc import tensor_to_array
from genesis.vis.viewer_plugins.raycast import Raycaster

from ..utils.assertions import assert_allclose, assert_equal


# A raycast consumer refreshes the solver's forward kinematics before casting; under MuJoCo compatibility the
# solver relies on redoing that itself, geoms included, so the arm below pins that a raycaster leaves the physics
# alone - obstacle_1 falls onto the ground plane and the grid rays must still find its top face where it rests.
@pytest.mark.parametrize(
    "n_envs, enable_mujoco_compatibility", [(0, False), pytest.param(2, True, marks=pytest.mark.required)]
)
def test_hits(show_viewer, n_envs, enable_mujoco_compatibility, tol):
    NUM_RAYS_XY = (3, 5)
    SPHERE_POS = (2.5, 0.5, 1.0)
    BOX_SIZE = 0.05
    RAYCAST_BOX_SIZE = 0.1
    RAYCAST_GRID_SIZE_X = 1.0
    RAYCAST_HEIGHT = 1.0
    GRAZE_FACE_Z = 1.0
    GRAZE_RANGE = 1.0

    scene = gs.Scene(
        rigid_options=gs.options.RigidOptions(
            enable_mujoco_compatibility=enable_mujoco_compatibility,
        ),
        vis_options=gs.options.VisOptions(
            split_envs=False,
            rendered_envs_idx=(0,),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(-3.0, RAYCAST_GRID_SIZE_X * (NUM_RAYS_XY[1] / NUM_RAYS_XY[0]), 2 * RAYCAST_HEIGHT),
            camera_lookat=(1.5, RAYCAST_GRID_SIZE_X * (NUM_RAYS_XY[1] / NUM_RAYS_XY[0]), RAYCAST_HEIGHT),
        ),
        profiling_options=gs.options.ProfilingOptions(
            show_FPS=False,
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(gs.morphs.Plane())

    spherical_sensor = scene.add_entity(
        gs.morphs.Sphere(
            radius=RAYCAST_HEIGHT,
            pos=SPHERE_POS,
            fixed=True,
        ),
    )
    spherical_raycaster = scene.add_sensor(
        gs.sensors.Raycaster(
            pattern=gs.sensors.raycaster.SphericalPattern(
                n_points=NUM_RAYS_XY,
            ),
            entity_idx=spherical_sensor.idx,
            return_world_frame=False,
            draw_debug=show_viewer,
            debug_ray_start_color=(0.0, 0.0, 0.0, 0.0),
            debug_ray_hit_color=(1.0, 0.0, 0.0, 1.0),
        )
    )

    grid_sensor = scene.add_entity(
        gs.morphs.Box(
            size=(RAYCAST_BOX_SIZE, RAYCAST_BOX_SIZE, RAYCAST_BOX_SIZE),
            pos=(0.0, 0.0, RAYCAST_HEIGHT + 0.5 * RAYCAST_BOX_SIZE),
            collision=False,
            fixed=False,
        ),
    )
    grid_res = RAYCAST_GRID_SIZE_X / (NUM_RAYS_XY[0] - 1)
    grid_size_y = grid_res * (NUM_RAYS_XY[1] - 1)
    grid_raycaster = scene.add_sensor(
        gs.sensors.Raycaster(
            pattern=gs.sensors.raycaster.GridPattern(
                resolution=grid_res,
                size=(RAYCAST_GRID_SIZE_X, grid_size_y),
                direction=(0.0, 0.0, -1.0),  # pointing downwards to ground
            ),
            entity_idx=grid_sensor.idx,
            pos_offset=(0.0, 0.0, -0.5 * RAYCAST_BOX_SIZE),
            return_world_frame=True,
            draw_debug=show_viewer,
            debug_ray_start_color=(0.0, 0.0, 0.0, 0.0),
            debug_ray_hit_color=(0.0, 1.0, 0.0, 1.0),
        )
    )
    depth_camera = scene.add_sensor(
        gs.sensors.DepthCamera(
            pattern=gs.sensors.raycaster.DepthCameraPattern(
                res=NUM_RAYS_XY[::-1],
            ),
            entity_idx=spherical_sensor.idx,
            draw_debug=show_viewer,
            debug_ray_start_color=(0.0, 0.0, 0.0, 0.0),
            debug_ray_hit_color=(0.0, 0.0, 1.0, 1.0),
        ),
    )
    depth_camera_depth_only = scene.add_sensor(
        gs.sensors.DepthCamera(
            pattern=gs.sensors.raycaster.DepthCameraPattern(
                res=NUM_RAYS_XY[::-1],
            ),
            entity_idx=spherical_sensor.idx,
            return_points=False,
        ),
    )

    # A ray grazing a face from just below must still find it. The face sits on a power of two, so a mount one float
    # under it starts within half an eps, and the ray rises through the face only ten hit-distances later.
    scene.add_entity(
        gs.morphs.Box(
            size=(BOX_SIZE, BOX_SIZE, GRAZE_FACE_Z),
            pos=(0.0, -RAYCAST_GRID_SIZE_X, 0.5 * GRAZE_FACE_Z),
            fixed=True,
        ),
    )
    graze_origin_z = np.nextafter(gs.np_float(GRAZE_FACE_Z), gs.np_float(0.0))
    graze_sensor = scene.add_entity(
        gs.morphs.Box(
            size=(BOX_SIZE, BOX_SIZE, BOX_SIZE),
            pos=(-(GRAZE_RANGE + 0.5 * BOX_SIZE), -RAYCAST_GRID_SIZE_X, graze_origin_z),
            collision=False,
            fixed=True,
        ),
    )
    graze_raycaster = scene.add_sensor(
        gs.sensors.Raycaster(
            pattern=gs.sensors.raycaster.GridPattern(
                resolution=1.0,
                size=(0.0, 0.0),
                direction=(1.0, 0.0, 0.1 * (GRAZE_FACE_Z - graze_origin_z) / GRAZE_RANGE),
            ),
            entity_idx=graze_sensor.idx,
            max_range=2.0 * GRAZE_RANGE,
            no_hit_value=-1.0,
        )
    )

    obstacle_1 = scene.add_entity(
        gs.morphs.Box(
            size=(BOX_SIZE, BOX_SIZE, BOX_SIZE),
            pos=(grid_res, grid_res, 0.5 * BOX_SIZE),
        ),
        material=gs.materials.Rigid(
            use_visual_raycasting=True,
        ),
    )
    obstacle_2 = scene.add_entity(
        gs.morphs.Box(
            size=(BOX_SIZE, BOX_SIZE, BOX_SIZE),
            pos=(RAYCAST_GRID_SIZE_X, grid_size_y, RAYCAST_HEIGHT + RAYCAST_BOX_SIZE + BOX_SIZE),
            fixed=True,
        ),
    )

    # Build the simulation and do one step
    scene.build(n_envs=n_envs)
    batch_shape = (n_envs,) if n_envs > 0 else ()

    # Validate grid raycast
    for obstacle_pos, sensor_pos, hit_ij in (
        (None, None, (-1, -2)),
        ((grid_res, grid_res, BOX_SIZE), None, (-1, -2)),
        (None, (*(grid_res * (e - 2) for e in NUM_RAYS_XY), RAYCAST_HEIGHT + 0.5 * RAYCAST_BOX_SIZE), (1, 0)),
    ):
        # Update obstacle and/or sensor position if necessary
        if obstacle_pos is not None:
            obstacle_1.set_pos(np.tile(obstacle_pos, (*batch_shape, 1)))
        obstacle_pos = obstacle_1.get_pos()
        if sensor_pos is not None:
            grid_sensor.set_pos(np.tile(sensor_pos, (*batch_shape, 1)))
        scene.sim._sensor_manager.step()
        if show_viewer:
            scene.visualizer.update(force=True)

        # Fetch updated sensor data
        grid_hits = grid_raycaster.read().points
        grid_distances = grid_raycaster.read().distances
        assert grid_distances.shape == (*batch_shape, *NUM_RAYS_XY)

        # Check hits
        grid_sensor_origin = grid_sensor.get_pos()
        x_rel = torch.linspace(-0.5, 0.5, NUM_RAYS_XY[0], dtype=gs.tc_float, device=gs.device)
        y_rel = torch.linspace(-0.5, 0.5, NUM_RAYS_XY[1], dtype=gs.tc_float, device=gs.device)
        x = x_rel * RAYCAST_GRID_SIZE_X + grid_sensor_origin[..., [0]]
        y = y_rel * grid_size_y + grid_sensor_origin[..., [1]]
        # xg, yg = torch.meshgrid(x, y, indexing="ij")
        xg = x.unsqueeze(-1).expand((*batch_shape, -1, NUM_RAYS_XY[1]))
        yg = y.unsqueeze(-2).expand((*batch_shape, NUM_RAYS_XY[0], -1))
        zg = torch.zeros((*batch_shape, *NUM_RAYS_XY), dtype=gs.tc_float, device=gs.device)
        zg[(..., *hit_ij)] = obstacle_pos[..., 2] + 0.5 * BOX_SIZE
        grid_hits_ref = torch.stack([xg, yg, zg], dim=-1)
        assert_allclose(grid_hits, grid_hits_ref, tol=gs.EPS)

        # Check distances
        grid_distances_ref = torch.full(
            (*batch_shape, *NUM_RAYS_XY), RAYCAST_HEIGHT, dtype=gs.tc_float, device=gs.device
        )
        grid_distances_ref[(..., *hit_ij)] = RAYCAST_HEIGHT - obstacle_pos[..., 2] - 0.5 * BOX_SIZE
        assert_allclose(grid_distances, grid_distances_ref, tol=gs.EPS)

    assert_allclose(graze_raycaster.read().distances, GRAZE_RANGE, tol=gs.EPS)

    # Validate spherical raycast
    spherical_distances = spherical_raycaster.read().distances
    assert spherical_distances.shape == (*batch_shape, *NUM_RAYS_XY)
    # Note that the tolerance must be large because the sphere geometry is discretized
    assert_allclose(spherical_distances, RAYCAST_HEIGHT, tol=5e-3)
    # The horizontal scan lines of a full rotation are evenly spaced, the gap between the last and the first included
    spherical_azimuths = torch.deg2rad(torch.linspace(-180.0, 180.0, NUM_RAYS_XY[0] + 1, dtype=gs.tc_float)[:-1])
    spherical_dirs_xy = torch.nn.functional.normalize(spherical_raycaster.read().points[..., :2], dim=-1)
    assert_allclose(spherical_dirs_xy[..., 0], spherical_azimuths.cos()[:, None], tol=tol)
    assert_allclose(spherical_dirs_xy[..., 1], spherical_azimuths.sin()[:, None], tol=tol)
    # Same for a full rotation on either axis, vertical included
    full_rotation_angles = gs.sensors.raycaster.SphericalPattern(fov=(360.0, 360.0), n_points=NUM_RAYS_XY).angles
    for angles, n_points in zip(full_rotation_angles, NUM_RAYS_XY):
        angles_ref = torch.deg2rad(torch.linspace(-180.0, 180.0, n_points + 1, dtype=gs.tc_float)[:-1])
        assert_allclose(angles, angles_ref, tol=tol)

    # Check that we can read image from depth camera
    assert_equal(depth_camera.read_image().shape, batch_shape + NUM_RAYS_XY)
    # Note that the tolerance must be large because the sphere geometry is discretized
    assert_allclose(depth_camera.read_image(), RAYCAST_HEIGHT, tol=5e-3)

    # Check the distances-only depth camera: distances bit-identical to its points-enabled twin
    assert_equal(depth_camera_depth_only.read().distances, depth_camera.read().distances)
    # The points-enabled twin keeps consistent points while packed next to a smaller distances-only cache block
    assert_allclose(depth_camera.read().points.norm(dim=-1), depth_camera.read().distances, tol=gs.EPS)

    # Imperfections must land on the sensor they are set on, whatever the per-sensor cache sizes
    depth_camera_depth_only.set_bias(0.05)
    scene.sim._sensor_manager.step()
    assert_allclose(
        depth_camera_depth_only.read().distances,
        depth_camera_depth_only.read_ground_truth().distances + 0.05,
        tol=gs.EPS,
    )
    assert_equal(depth_camera.read().points, depth_camera.read_ground_truth().points)
    assert_equal(depth_camera.read().distances, depth_camera.read_ground_truth().distances)

    # Simulate for a while and check again that the ray is casted properly
    offset = torch.from_numpy(np.random.rand(*batch_shape, 3)).to(dtype=gs.tc_float, device=gs.device)
    for entity in (grid_sensor, obstacle_1, obstacle_2):
        pos = entity.get_pos() + offset
        if entity is obstacle_2:
            pos[..., 2] = BOX_SIZE / 2
        entity.set_pos(pos)
    grid_sensor_pos = grid_sensor.get_pos()
    for _ in range(60):
        scene.step()
    grid_sensor.set_pos(grid_sensor_pos)
    scene.sim._sensor_manager.step()
    if show_viewer:
        scene.visualizer.update(force=True)

    grid_distances = grid_raycaster.read().distances
    grid_distances_ref = torch.full((*batch_shape, *NUM_RAYS_XY), RAYCAST_HEIGHT, dtype=gs.tc_float, device=gs.device)
    grid_distances_ref[(..., -1, -2)] = RAYCAST_HEIGHT - BOX_SIZE
    grid_distances_ref[(..., *hit_ij)] = RAYCAST_HEIGHT - BOX_SIZE
    grid_distances_ref += offset[..., 2].reshape((*(-1 for e in batch_shape), 1, 1))
    assert_allclose(grid_distances, grid_distances_ref, tol=1e-3)


@pytest.mark.required("sparse")
@pytest.mark.parametrize("n_envs", [2, 0])
@pytest.mark.parametrize("kin_raycastable", [True, False])
def test_against_visual(tmp_path, show_viewer, n_envs, kin_raycastable):
    # Two depth cameras, one per entity:
    #   - cam_kin -> KinematicEntity sphere. When use_visual_raycasting=True the depth camera reads the entity's
    #     visual mesh (including set_vverts overrides, which survive step() until set_vverts(None) hands control
    #     back to FK). When False the kinematic entity is completely ignored by the raycaster.
    #   - cam_rigid -> RigidEntity whose visual mesh (sphere radius 0.2) is intentionally different from its collision
    #     mesh (capsule radius 0.05). With use_visual_raycasting=True the depth must match the visual sphere.
    urdf_path = tmp_path / "vis_diff.urdf"
    urdf_path.write_text(
        textwrap.dedent(
            """
            <robot name="vis_diff">
                <link name="root">
                    <visual>
                        <origin rpy="0 0 0" xyz="0 0 0"/>
                        <geometry>
                            <sphere radius="0.2"/>
                        </geometry>
                    </visual>
                    <collision>
                        <origin rpy="0 0 0" xyz="0 0 0"/>
                        <geometry>
                            <capsule radius="0.05" length="0.05"/>
                        </geometry>
                    </collision>
                </link>
            </robot>
            """
        )
    )

    scene = gs.Scene(
        profiling_options=gs.options.ProfilingOptions(
            show_FPS=False,
        ),
        show_viewer=show_viewer,
    )
    plane = scene.add_entity(gs.morphs.Plane())
    kin_sphere = scene.add_entity(
        morph=gs.morphs.Mesh(
            file="meshes/sphere.obj",
            scale=0.2,
            pos=(0.0, 0.0, 0.5),
            fixed=True,
            enable_custom_vverts=True,
        ),
        material=gs.materials.Kinematic(
            use_visual_raycasting=kin_raycastable,
        ),
    )
    scene.add_entity(
        morph=gs.morphs.URDF(
            file=str(urdf_path),
            pos=(0.0, 0.0, 1.5),
            fixed=True,
        ),
        material=gs.materials.Rigid(
            use_visual_raycasting=True,
        ),
    )
    cam_kin = scene.add_sensor(
        gs.sensors.DepthCamera(
            pattern=gs.sensors.DepthCameraPattern(
                res=(40, 30),
                fov_horizontal=30.0,
            ),
            entity_idx=plane.idx,
            link_idx_local=0,
            pos_offset=(-1.0, 0.0, 0.5),
            euler_offset=(0.0, 0.0, 0.0),
            max_range=5.0,
            return_world_frame=True,
        ),
    )
    cam_rigid = scene.add_sensor(
        gs.sensors.DepthCamera(
            pattern=gs.sensors.DepthCameraPattern(
                res=(40, 30),
                fov_horizontal=30.0,
            ),
            entity_idx=plane.idx,
            link_idx_local=0,
            pos_offset=(-1.0, 0.0, 1.5),
            euler_offset=(0.0, 0.0, 0.0),
            max_range=5.0,
            return_world_frame=True,
        ),
    )
    if n_envs > 0:
        scene.build(n_envs=n_envs)
    else:
        scene.build()
    scene.step()

    # Each camera at x=-1 along its own z-row looks along +x. The center pixel hits the closest point of its target
    # sphere at x=-0.2 -> depth 0.8. For cam_rigid this comes from the visual mesh (not the collision capsule). When
    # the kinematic entity opts out of raycasting, cam_kin sees nothing and returns the no_hit_value (max_range=5.0).
    NO_HIT = 5.0  # max_range
    kin_at_origin = 0.8 if kin_raycastable else NO_HIT
    kin_scaled = 0.6 if kin_raycastable else NO_HIT
    assert_allclose(cam_kin.read_image()[..., 15, 20], kin_at_origin, tol=1e-2)
    assert_allclose(cam_rigid.read_image()[..., 15, 20], 0.8, tol=1e-2)

    # Scale the kinematic sphere by 2x around its center via per-vertex set_vverts. The new radius is 0.4, so the
    # closest point becomes x=-0.4 and the depth at the center pixel drops to 0.6. Scaling perturbs each vvert by a
    # different amount, so only the correct vvert-to-state mapping yields 0.6. cam_rigid is unaffected.
    fk_vverts = tensor_to_array(kin_sphere.get_vverts())
    center = np.array([0.0, 0.0, 0.5], dtype=np.float32)
    kin_sphere.set_vverts((fk_vverts - center) * 2.0 + center)
    scene.step()
    assert_allclose(cam_kin.read_image()[..., 15, 20], kin_scaled, tol=1e-2)
    assert_allclose(cam_rigid.read_image()[..., 15, 20], 0.8, tol=1e-2)

    # Push the kinematic sphere far away. cam_kin should report no_hit_value at the center pixel; cam_rigid still sees
    # the rigid visual sphere.
    kin_sphere.set_vverts((100.0, 100.0, 100.0))
    scene.step()
    assert_allclose(cam_kin.read_image()[..., 15, 20], NO_HIT, tol=gs.EPS)
    assert_allclose(cam_rigid.read_image()[..., 15, 20], 0.8, tol=1e-2)

    # Restoring FK control returns the original hit distance on cam_kin; cam_rigid stays put.
    kin_sphere.set_vverts(None)
    scene.step()
    assert_allclose(cam_kin.read_image()[..., 15, 20], kin_at_origin, tol=1e-2)
    assert_allclose(cam_rigid.read_image()[..., 15, 20], 0.8, tol=1e-2)


@pytest.mark.required
def test_lidar_bvh_parallel_env(show_viewer, tol):
    SHARED_OBSTACLE_1_X = 1.2
    SHARED_OBSTACLE_2_X = 1.3
    scene = gs.Scene(
        vis_options=gs.options.VisOptions(
            rendered_envs_idx=(1,),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1, -5, 3),
            camera_lookat=(1, 0.5, 0),
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(gs.morphs.Plane())

    sensor_mount = scene.add_entity(
        gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(0.0, 0.0, 0.5),
            fixed=True,
            collision=False,
        )
    )
    obstacle_1 = scene.add_entity(
        gs.morphs.Box(
            size=(0.2, 0.2, 0.2),
            pos=(1.0, 0.0, 0.5),
            fixed=True,
        ),
    )
    obstacle_2 = scene.add_entity(
        gs.morphs.Box(
            size=(0.05, 0.4, 0.4),
            pos=(1.0, 0.0, 0.5),
            fixed=True,
        ),
    )

    lidar = scene.add_sensor(
        gs.sensors.Lidar(
            entity_idx=sensor_mount.idx,
            pattern=gs.options.sensors.SphericalPattern(
                n_points=(1, 1),
                fov=(0.0, 0.0),
            ),
            max_range=5.0,
            draw_debug=show_viewer,
            debug_ray_start_color=(0.0, 0.0, 0.0, 0.0),
            debug_ray_hit_color=(1.0, 0.0, 0.0, 1.0),
        )
    )

    scene.build(n_envs=2)

    sensor_positions = np.array([[0.0, 0.0, 0.5], [0.0, 1.0, 0.5]], dtype=gs.np_float)
    obstacle_1_positions = np.array([[1.1, 0.0, 0.5], [2.5, 1.0, 0.5]], dtype=gs.np_float)
    obstacle_2_positions = np.array([[1.4, 0.0, 0.5], [2.2, 1.0, 0.5]], dtype=gs.np_float)
    sensor_mount.set_pos(sensor_positions)
    obstacle_1.set_pos(obstacle_1_positions)
    obstacle_2.set_pos(obstacle_2_positions)

    scene.step()

    distances = lidar.read().distances
    assert distances.shape == (2, 1, 1)
    lidar_distances = distances[:, 0, 0]

    front_positions = np.minimum(obstacle_1_positions[:, 0] - 0.1, obstacle_2_positions[:, 0] - 0.025)
    expected_distances = front_positions - sensor_positions[:, 0]
    assert_allclose(lidar_distances, expected_distances, tol=tol)

    # Make the obstacle geometry identical across envs (sensors still differ in x): the set_pos calls must reach the
    # cast, otherwise it keeps casting against the stale poses.
    shared_sensor_positions = np.array([[0.0, 0.0, 0.5], [0.5, 0.0, 0.5]], dtype=gs.np_float)
    sensor_mount.set_pos(shared_sensor_positions)
    obstacle_1.set_pos((SHARED_OBSTACLE_1_X, 0.0, 0.5))
    obstacle_2.set_pos((SHARED_OBSTACLE_2_X, 0.0, 0.5))

    scene.step()

    shared_distances = lidar.read().distances[:, 0, 0]
    shared_expected = min(SHARED_OBSTACLE_1_X - 0.1, SHARED_OBSTACLE_2_X - 0.025) - shared_sensor_positions[:, 0]
    assert_allclose(shared_distances, shared_expected, tol=tol)


@pytest.mark.parametrize("n_envs", [0, pytest.param(4, marks=pytest.mark.required)])
def test_cast_per_env_geoms(show_viewer, n_envs):
    scene = gs.Scene(show_viewer=show_viewer)
    scene.add_entity(gs.morphs.Plane())
    box = scene.add_entity(
        gs.morphs.Box(
            size=(0.4, 0.4, 0.4),
            pos=(1.0, 0.0, 0.5),
            fixed=True,
        )
    )
    sensor_mount = scene.add_entity(
        gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(0.0, 0.0, 0.5),
            fixed=True,
            collision=False,
        )
    )
    visual_box = scene.add_entity(
        gs.morphs.Box(
            size=(0.4, 0.4, 0.4),
            pos=(6.0, 0.0, 0.5),
            fixed=True,
            collision=False,
        ),
        material=gs.materials.Kinematic(
            use_visual_raycasting=True,
        ),
    )
    movable_box = scene.add_entity(
        gs.morphs.Box(
            size=(0.2, 0.2, 0.2),
            pos=(3.0, 2.0, 0.5),
        )
    )
    lidar = scene.add_sensor(
        gs.sensors.Lidar(
            entity_idx=sensor_mount.idx,
            pattern=gs.options.sensors.SphericalPattern(n_points=(1, 1), fov=(0.0, 0.0)),
            max_range=8.0,
        )
    )
    scene.build(n_envs=n_envs)
    scene.step()

    assert_allclose(lidar.read().distances[..., 0, 0], 0.8, tol=5e-3)

    if n_envs > 0:
        # A per-env set_pos on the fixed box diverges the envs, each env reading its own moved box. The env order is
        # deliberately non-monotonic in x, so any confusion between environments shifts the distances.
        box.set_pos(np.array([[1.0, 0.0, 0.5], [4.0, 0.0, 0.5], [3.0, 0.0, 0.5], [2.0, 0.0, 0.5]], dtype=gs.np_float))
        scene.step()
        assert_allclose(lidar.read().distances[:, 0, 0], (0.8, 3.8, 2.8, 1.8), tol=5e-3)

        # A per-env set_pos on the visual box, with interleaved env membership: the merge picks the visual hit where
        # it is the closest.
        visual_box.set_pos(
            np.array([[6.0, 0.0, 0.5], [6.0, 0.0, 0.5], [1.5, 0.0, 0.5], [6.0, 0.0, 0.5]], dtype=gs.np_float)
        )
        scene.step()
        assert_allclose(lidar.read().distances[:, 0, 0], (0.8, 3.8, 1.3, 1.8), tol=5e-3)

        # Diverge the movable box per env (off the ray corridor), then restore the fixed box to identical poses
        movable_box.set_pos(
            np.array([[3.0, 2.0, 0.5], [4.0, 2.0, 0.5], [5.0, 2.0, 0.5], [6.0, 2.0, 0.5]], dtype=gs.np_float)
        )
        box.set_pos((1.0, 0.0, 0.5))
        scene.step()
        assert_allclose(lidar.read().distances[:, 0, 0], 0.8, tol=5e-3)

        # A reset restores the built geometry
        scene.reset()
        scene.step()
        assert_allclose(lidar.read().distances[:, 0, 0], 0.8, tol=5e-3)


@pytest.mark.required
def test_lidar_cache_offset_parallel_env(show_viewer, tol):
    scene = gs.Scene(show_viewer=show_viewer)

    scene.add_entity(
        morph=gs.morphs.Plane(),
    )
    cube = scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.1, 0.1, 1.0),
            pos=(0.0, 0.0, 0.5),
        ),
    )

    sensors = [
        scene.add_sensor(
            gs.sensors.Raycaster(
                pattern=gs.sensors.raycaster.SphericalPattern(
                    n_points=(2, 2),
                ),
                entity_idx=cube.idx,
                return_world_frame=False,
            )
        ),
        scene.add_sensor(
            gs.sensors.Raycaster(
                pattern=gs.sensors.raycaster.SphericalPattern(
                    n_points=(2, 2),
                ),
                entity_idx=cube.idx,
                return_world_frame=False,
            )
        ),
        scene.add_sensor(
            gs.sensors.Raycaster(
                pattern=gs.sensors.raycaster.SphericalPattern(
                    n_points=(2, 2),
                ),
                entity_idx=cube.idx,
                return_world_frame=False,
            )
        ),
    ]

    scene.build()

    scene.step()
    for sensor in sensors:
        sensor_data = sensor.read()
        assert (sensor_data.distances > gs.EPS).any()
        assert (sensor_data.points.abs() > gs.EPS).any()


@pytest.mark.required
def test_heterogeneous_object(show_viewer, tol):
    scene = gs.Scene(show_viewer=show_viewer)
    scene.add_entity(gs.morphs.Plane())
    sensor_mount = scene.add_entity(
        gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(0.0, 0.0, 0.5),
            fixed=True,
            collision=False,
        )
    )
    # Without per-env geom masking an env casts against the union of all variants (they share one vertex buffer). The
    # variants are concentric obstacles of decreasing near-face distance, so each env's own variant is the farthest
    # hit. A missing mask is then observable as an env shadowing its variant with a nearer one belonging to another env.
    het_obstacle = scene.add_entity(
        morph=(
            gs.morphs.Box(size=(0.2, 0.2, 0.2), pos=(1.0, 0.0, 0.5), fixed=True),
            gs.morphs.Sphere(radius=0.2, pos=(1.0, 0.0, 0.5), fixed=True),
            gs.morphs.Box(size=(0.6, 0.6, 0.6), pos=(1.0, 0.0, 0.5), fixed=True),
        ),
        material=gs.materials.Rigid(
            use_visual_raycasting=True,
        ),
    )
    # A box primitive puts its vgeom at the link origin, so these same-pos variants hold identical vgeom poses and
    # differ only in rest-pose vverts: the active vgeom range of each environment alone tells them apart.
    scene.add_entity(
        morph=(
            gs.morphs.Box(size=(0.2, 0.2, 0.2), pos=(4.0, 0.0, 0.5), fixed=True),
            gs.morphs.Box(size=(0.4, 0.4, 0.4), pos=(4.0, 0.0, 0.5), fixed=True),
            gs.morphs.Box(size=(0.6, 0.6, 0.6), pos=(4.0, 0.0, 0.5), fixed=True),
        ),
        material=gs.materials.Kinematic(
            use_visual_raycasting=True,
        ),
    )
    movable_box = scene.add_entity(
        gs.morphs.Box(
            size=(0.05, 0.4, 0.4),
            pos=(3.0, 2.0, 0.5),
        )
    )
    lidar = scene.add_sensor(
        gs.sensors.Lidar(
            entity_idx=sensor_mount.idx,
            pattern=gs.options.sensors.SphericalPattern(n_points=(1, 1), fov=(0.0, 0.0)),
            max_range=5.0,
            draw_debug=show_viewer,
        )
    )

    scene.build(n_envs=6)
    scene.step()

    # Balanced variant mapping assigns contiguous env blocks: envs (0, 1) -> variant 0, (2, 3) -> 1, (4, 5) -> 2.
    distances = lidar.read().distances[:, 0, 0]
    assert_allclose(distances, (0.9, 0.9, 0.8, 0.8, 0.7, 0.7), tol=5e-3)

    # The movable box slides into the ray's path: its closer hit wins identically in every env, in front of each
    # env's own variant.
    movable_box.set_pos((0.5, 0.0, 0.5))
    scene.step()
    assert_allclose(lidar.read().distances[:, 0, 0], 0.475, tol=5e-3)

    # And back out: each env again sees exactly its own variant.
    movable_box.set_pos((3.0, 2.0, 0.5))
    scene.step()
    assert_allclose(lidar.read().distances[:, 0, 0], (0.9, 0.9, 0.8, 0.8, 0.7, 0.7), tol=5e-3)

    # Sending the rigid variants out of range leaves the kinematic ones as the nearest hit, each env still reading
    # its own.
    het_obstacle.set_pos((10.0, 0.0, 0.5))
    scene.step()
    assert_allclose(lidar.read().distances[:, 0, 0], (3.9, 3.9, 3.8, 3.8, 3.7, 3.7), tol=5e-3)


@pytest.mark.parametrize("n_envs", [0, pytest.param(2, marks=pytest.mark.required)])
def test_cast_fixed_and_moving_geoms(show_viewer, n_envs, tol):
    MOUNT_Z = 3.0
    FIXED_TOP_Z = 1.0
    VISUAL_TOP_Z = 0.5
    MOVABLE_TOP_Z = 2.4  # movable box (edge 0.4) centered at z=2.2
    FAR_POS = (5.0, 5.0, 0.2)
    NO_HIT = -1.0  # below max_range on purpose: a miss must still lose the merge against any real hit
    DT = 0.01
    TACTILE_POS = (10.0, 0.0)  # tactile cluster, laterally away from the ray corridor
    PAD_POS_Z = 0.0025  # pad half-height 0.003: rests 0.5mm into the table so the table becomes a probe candidate
    TACTILE_BOX_BOTTOM_Z = -0.001
    PROBE_RADIUS = 0.005
    PROBE_1_LOCAL_Z = -0.002  # world z=0.0005: 0.5mm above the table top, 1.5mm inside the box above its bottom face
    PROBE_2_LOCAL = (0.018, 0.0, 0.002)  # world z=0.0045: 2mm inside the box's x side face, the nearest face overall

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=DT,
            gravity=(0.0, 0.0, 0.0),
        ),
        show_viewer=show_viewer,
    )
    fixed_box = scene.add_entity(
        gs.morphs.Box(
            size=(1.0, 1.0, 1.0),
            pos=(0.0, 0.0, 0.5),
            fixed=True,
            batch_fixed_verts=False,
        )
    )
    visual_box = scene.add_entity(
        gs.morphs.Box(
            size=(1.0, 1.0, 0.5),
            pos=(0.0, 0.0, 0.25),
            fixed=True,
            collision=False,
        ),
        material=gs.materials.Rigid(
            use_visual_raycasting=True,
        ),
    )
    mount = scene.add_entity(
        gs.morphs.Box(
            size=(0.05, 0.05, 0.05),
            pos=(0.0, 0.0, MOUNT_Z),
            fixed=True,
            collision=False,
        )
    )
    movable_box = scene.add_entity(
        gs.morphs.Box(
            size=(0.4, 0.4, 0.4),
            pos=FAR_POS,
        )
    )
    # Tactile cluster: a movable sensor pad resting 0.5mm into a fixed table with a movable box pressed
    # through it, so its probes hold candidates on both.
    table = scene.add_entity(
        gs.morphs.Box(
            size=(0.4, 0.4, 0.1),
            pos=(*TACTILE_POS, -0.05),
            fixed=True,
            batch_fixed_verts=False,
        )
    )
    pad = scene.add_entity(
        gs.morphs.Box(
            size=(0.04, 0.04, 0.006),
            pos=(*TACTILE_POS, PAD_POS_Z),
        )
    )
    tactile_box = scene.add_entity(
        gs.morphs.Box(
            size=(0.04, 0.04, 0.04),
            pos=(*TACTILE_POS, TACTILE_BOX_BOTTOM_Z + 0.02),
        )
    )
    depth_probe = scene.add_sensor(
        gs.sensors.ContactDepthProbe(
            entity_idx=pad.idx,
            probe_local_pos=((0.0, 0.0, PROBE_1_LOCAL_Z), PROBE_2_LOCAL),
            probe_radius=PROBE_RADIUS,
            contact_depth_query="raycast",
        )
    )
    taxel = scene.add_sensor(
        gs.sensors.KinematicTaxel(
            entity_idx=pad.idx,
            probe_local_pos=((0.0, 0.0, PROBE_1_LOCAL_Z), PROBE_2_LOCAL),
            probe_radius=PROBE_RADIUS,
            normal_stiffness=100.0,
            normal_damping=0.0,
            shear_scalar=0.0,
            twist_scalar=0.0,
            contact_depth_query="raycast",
        )
    )
    sensor = scene.add_sensor(
        gs.sensors.Raycaster(
            pattern=gs.sensors.raycaster.GridPattern(resolution=1.0, size=(0.0, 0.0), direction=(0.0, 0.0, -1.0)),
            entity_idx=mount.idx,
            return_world_frame=False,
            no_hit_value=NO_HIT,
        )
    )
    distances_only_sensor = scene.add_sensor(
        gs.sensors.Raycaster(
            pattern=gs.sensors.raycaster.GridPattern(resolution=1.0, size=(0.0, 0.0), direction=(0.0, 0.0, -1.0)),
            entity_idx=mount.idx,
            return_world_frame=False,
            return_points=False,
            no_hit_value=NO_HIT,
        )
    )
    scene.build(n_envs=n_envs)

    def assert_distances(expected, tol=gs.EPS):
        assert_allclose(sensor.read().distances, expected, tol=tol)
        assert_allclose(distances_only_sensor.read().distances, expected, tol=tol)

    # Movable box parked far away: the ray hits the fixed box top, the closer collision hit surviving the merge with
    # the farther visual hit.
    scene.step()
    assert_distances(MOUNT_Z - FIXED_TOP_Z)

    # Tactile probes pick the globally nearest triangle. The contact response of the step above displaced the
    # interpenetrating movable pair, so their poses are restored, and the candidate contact pairs (pad-table, pad-box)
    # detected anew at them, before sampling the sensors alone. Probe 1 is nearer to the table face (0.5mm outside)
    # than to the box bottom (1.5mm inside): a choice by deepest penetration instead of smallest signed-distance
    # magnitude would report the larger depth of the box. Probe 2 is nearest to the x side face of the box (2mm inside).
    pad.set_pos((*TACTILE_POS, PAD_POS_Z))
    tactile_box.set_pos((*TACTILE_POS, TACTILE_BOX_BOTTOM_Z + 0.02))
    scene.rigid_solver.detect_collision()
    scene.sim._sensor_manager.step()
    probe_1_world_z = PAD_POS_Z + PROBE_1_LOCAL_Z
    assert_allclose(
        depth_probe.read_ground_truth(),
        (PROBE_RADIUS - probe_1_world_z, PROBE_RADIUS + 0.002),
        tol=1e-6,
    )
    # The taxel's contact normal must come from the same nearest triangle: the table's upward face at probe 1 (the
    # box's deeper answer would push along its bottom face's downward normal instead).
    assert (taxel.read_ground_truth().force[..., 0, 2] > gs.EPS).all()
    # Park the tactile pair apart: left interpenetrating, the contact response keeps ejecting them across the scene.
    pad.set_pos((*TACTILE_POS, 1.0))
    tactile_box.set_pos((10.0, 3.0, 1.0))

    # Movable box under the ray, above the fixed box: the closer movable hit wins
    movable_box.set_pos((0.0, 0.0, MOVABLE_TOP_Z - 0.2))
    scene.step()
    assert_distances(MOUNT_Z - MOVABLE_TOP_Z)

    # Movable box parked again: back to the fixed hit.
    movable_box.set_pos(FAR_POS)
    scene.step()
    assert_distances(MOUNT_Z - FIXED_TOP_Z)

    # Teleporting the fixed box lets the ray fall through to the visual box: a first-pass miss followed by a
    # last-pass hit, the ordering that would report NO_HIT if a miss stamped it before the merge completed.
    fixed_box.set_pos((5.0, -5.0, 0.5))
    scene.step()
    assert_distances(MOUNT_Z - VISUAL_TOP_Z)

    # Same ordering with the hit coming from the collision pass instead of the visual one.
    movable_box.set_pos((0.0, 0.0, MOVABLE_TOP_Z - 0.2))
    scene.step()
    assert_distances(MOUNT_Z - MOVABLE_TOP_Z)

    # Move everything out of the ray's path: only a miss on every pass settles to no_hit_value.
    visual_box.set_pos((5.0, 0.0, 0.25))
    movable_box.set_pos(FAR_POS)
    scene.step()
    assert_distances(NO_HIT)

    # Physics-driven motion: a constant downward velocity moves the movable box without any GEOMETRY event, and the
    # cast tracks the fall.
    movable_box.set_pos((0.0, 0.0, MOVABLE_TOP_Z - 0.2))
    movable_box.set_dofs_velocity((0.0, 0.0, -1.0), dofs_idx_local=slice(0, 3))
    scene.step()
    assert_distances(MOUNT_Z - MOVABLE_TOP_Z + DT, tol=tol)
    scene.step()
    assert_distances(MOUNT_Z - MOVABLE_TOP_Z + 2 * DT, tol=tol)

    # A reset restores the initial poses, the teleported fixed box included
    scene.reset()
    scene.step()
    assert_distances(MOUNT_Z - FIXED_TOP_Z)

    # The movable box behind the fixed one: the nearer fixed hit wins
    movable_box.set_pos((0.0, 0.0, -0.5))
    scene.step()
    assert_distances(MOUNT_Z - FIXED_TOP_Z)

    # The viewer picks through the same trees: a ray down from the mount meets the movable box moved in front on the
    # collision meshes, and the visual box on the meshes opted into visual raycasting
    movable_box.set_pos((0.0, 0.0, MOVABLE_TOP_Z - 0.2))
    scene.step()
    for use_visual_geom, top_z in ((False, MOVABLE_TOP_Z), (True, VISUAL_TOP_Z)):
        hit = Raycaster(scene, use_visual_geom=use_visual_geom).cast((0.0, 0.0, MOUNT_Z), (0.0, 0.0, -1.0))
        assert_allclose(hit.distance, MOUNT_Z - top_z, tol=tol)

import math
import xml.etree.ElementTree as ET
from contextlib import nullcontext
from itertools import product

import numpy as np
import pytest
import torch
import trimesh
from scipy.spatial import ConvexHull
from scipy.spatial.qhull import QhullError

import genesis as gs
import genesis.utils.geom as gu
from genesis.utils.misc import tensor_to_array

from ..utils.assertions import assert_allclose, assert_equal
from ..utils.assets import get_hf_dataset
from .conftest import ellipsoid_mjcf


def _capsule_mjcf_path(tmp_path, radius, length, name="capsule"):
    mjcf = ET.Element("mujoco", model=name)
    body = ET.SubElement(ET.SubElement(mjcf, "worldbody"), "body")
    ET.SubElement(body, "geom", type="capsule", size=f"{radius} {0.5 * length}")
    ET.SubElement(body, "joint", type="free")
    path = tmp_path / f"{name}.xml"
    ET.ElementTree(mjcf).write(path)
    return str(path)


def _ellipsoid_mjcf_path(tmp_path, semi_axes):
    path = tmp_path / "ellipsoid.xml"
    ET.ElementTree(ellipsoid_mjcf(semi_axes)).write(path)
    return str(path)


@pytest.mark.required
@pytest.mark.mujoco_compatibility(False)
@pytest.mark.parametrize("mode", range(9))
@pytest.mark.parametrize("model_name", ["collision_edge_cases"])
@pytest.mark.parametrize("gs_solver", [gs.constraint_solver.CG])
@pytest.mark.parametrize("gs_integrator", [gs.integrator.Euler])
@pytest.mark.parametrize("gjk_collision", [True, False])
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
def test_edge_cases(gs_sim, mode):
    qpos_0 = gs_sim.rigid_solver.get_dofs_position()
    for _ in range(200):
        gs_sim.scene.step()

    qvel = gs_sim.rigid_solver.get_dofs_velocity()
    assert_allclose(qvel, 0, atol=1e-2)
    qpos = gs_sim.rigid_solver.get_dofs_position()
    atol = 1e-3 if mode in (4, 6) else 1e-4
    assert_allclose(qpos[[0, 1, 3, 4, 5]], qpos_0[[0, 1, 3, 4, 5]], atol=atol)


@pytest.mark.slow  # ~200s
@pytest.mark.required
@pytest.mark.parametrize("backend", [gs.cpu])
def test_plane_convex(show_viewer, tol):
    for morph in (
        gs.morphs.Plane(),
        gs.morphs.Box(
            pos=(0.5, 0.0, -0.5),
            size=(1.0, 1.0, 1.0),
            fixed=True,
        ),
    ):
        scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=0.001,
            ),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(1.0, -0.5, 0.5),
                camera_lookat=(0.5, 0.0, 0.0),
            ),
            show_viewer=show_viewer,
            show_FPS=False,
        )

        scene.add_entity(morph)

        asset_path = get_hf_dataset(pattern="image_0000_segmented.glb")
        asset = scene.add_entity(
            gs.morphs.Mesh(
                file=f"{asset_path}/image_0000_segmented.glb",
                scale=0.03196910891804585,
                pos=(0.45184245, 0.05020455, 0.02),
                quat=(0.51982231, 0.44427745, 0.49720965, 0.53402704),
            ),
            vis_mode="collision",
            visualize_contact=True,
        )

        scene.build()

        for i in range(500):
            scene.step()
            if i > 400:
                qvel = asset.get_dofs_velocity()
                assert_allclose(qvel, 0, atol=0.14)


@pytest.mark.slow  # ~200s
@pytest.mark.required
@pytest.mark.parametrize("model_name", ["ellipsoid"])
def test_ellipsoid(xml_path, show_viewer):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.02,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.4, 0.4, 0.3),
            camera_lookat=(0.0, 0.0, 0.1),
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(gs.morphs.Plane())
    entity = scene.add_entity(
        gs.morphs.MJCF(
            file=xml_path,
            pos=(0, 0, 0.2),
        ),
        vis_mode="collision",
        visualize_contact=True,
    )
    scene.build()

    entity.set_dofs_velocity(20 * np.random.rand(3), dofs_idx_local=slice(3, 6))
    entity.set_dofs_kv(0.002, dofs_idx_local=slice(3, 6))
    entity.control_dofs_velocity(0.0, dofs_idx_local=slice(3, 6))

    # AABB must match the ellipsoid semi-axes
    aabb = entity.get_AABB()
    aabb_extent = aabb[1] - aabb[0]
    assert_allclose(aabb_extent, (0.10, 0.10, 0.04), atol=1e-3)

    # Free-fall onto plane: ellipsoid must come to rest
    for _ in range(100):
        scene.step()

    assert_allclose(entity.get_dofs_velocity(), 0, tol=5e-3)
    assert (-0.005 < entity.get_AABB()[0, 2] < 0.0).all()
    roll, pitch, _yaw = gu.quat_to_xyz(entity.get_quat(), rpy=True)
    assert_allclose((roll, pitch), (0.0, 0.0), tol=5e-3)


@pytest.mark.parametrize(
    "entity_kind, entity_type, ground_type",
    [
        pytest.param("sphere", "prim", "prim", marks=pytest.mark.required),
        pytest.param("sphere", "prim", "mesh", marks=pytest.mark.required),
        pytest.param("capsule", "prim", "prim", marks=pytest.mark.required),
        pytest.param("capsule", "prim", "mesh", marks=pytest.mark.required),
        pytest.param("cylinder", "prim", "prim", marks=pytest.mark.required),
        pytest.param("cylinder", "prim", "mesh", marks=pytest.mark.required),
        pytest.param("ellipsoid", "prim", "prim", marks=pytest.mark.required),
        pytest.param("ellipsoid", "prim", "mesh", marks=pytest.mark.required),
        ("sphere", "prim", "terrain"),
        ("sphere", "prim", "nonconvex"),
        ("sphere", "mesh", "mesh"),
        ("sphere", "nonconvex", "prim"),
        ("sphere", "nonconvex", "nonconvex"),
        ("sphere", "nonconvex", "plane"),
    ],
)
@pytest.mark.parametrize("gjk_collision", [False, True])
def test_no_drift(gjk_collision, entity_kind, entity_type, ground_type, show_viewer, tmp_path):
    WORLD_TILT_ANGLE = 50.0
    HEIGHT = 0.02
    # The smooth-primitive characteristic length must be small enough to amplify the bias and make drift evident
    SMOOTH_RADIUS = 0.0025
    CYLINDER_HEIGHT = 0.005
    # Smallest semi-axis along body z so the ellipsoid rests on its narrowest cross-section
    ELLIPSOID_SEMI_AXES = (0.0035, 0.0030, SMOOTH_RADIUS)
    BOX_HALF_EXTENT = 0.1
    N_ENVS = 16
    SPHERE_TESSELLATION_SUBDIVISIONS = 3

    # The box and the gravity vector are rotated by the same tilt, which is physically equivalent to the untilted setup.
    tilt_axis = np.array([1.0, 1.0, 0.0]) / math.sqrt(2.0)
    tilt_quat = gu.rotvec_to_quat(math.radians(WORLD_TILT_ANGLE) * tilt_axis)
    R = gu.quat_to_R(tilt_quat)
    box_pos_world = R @ np.array([0.0, 0.0, 0.5 * HEIGHT])
    gravity_world = R @ np.array([0.0, 0.0, -9.81])

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.003,
            gravity=gravity_world,
        ),
        rigid_options=gs.options.RigidOptions(
            use_gjk_collision=gjk_collision,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.25, 0.25, 0.2),
            camera_lookat=(0.0, 0.0, 0.5 * HEIGHT),
            camera_fov=30.0,
        ),
        show_viewer=show_viewer,
    )
    if ground_type in ("mesh", "nonconvex"):
        box_mesh = trimesh.creation.box(extents=(2.0 * BOX_HALF_EXTENT, 2.0 * BOX_HALF_EXTENT, HEIGHT))
        is_ground_convex = ground_type == "mesh"
        box = scene.add_entity(
            morph=gs.morphs.MeshSet(
                files=(box_mesh,),
                pos=box_pos_world,
                quat=tilt_quat,
                convexify=is_ground_convex,
                fixed=True,
            ),
            surface=gs.surfaces.Default(
                smooth=False,
            ),
            visualize_contact=True,
        )
        # Manually overwrite convex flag to forcibly exercise non-convex collision path
        box.geoms[0]._is_convex = is_ground_convex
    elif ground_type == "terrain":
        flat_hf = np.zeros((2, 2), dtype=np.float32)
        terrain_pos_world = R @ np.array([-BOX_HALF_EXTENT, -BOX_HALF_EXTENT, HEIGHT])
        scene.add_entity(
            morph=gs.morphs.Terrain(
                horizontal_scale=2.0 * BOX_HALF_EXTENT,
                vertical_scale=2.0 * BOX_HALF_EXTENT,
                height_field=flat_hf,
                pos=terrain_pos_world,
                quat=tilt_quat,
            ),
            visualize_contact=True,
        )
    elif ground_type == "plane":
        plane_pos_world = R @ np.array([0.0, 0.0, HEIGHT])
        scene.add_entity(
            morph=gs.morphs.Plane(
                pos=plane_pos_world,
                plane_size=(2.0 * BOX_HALF_EXTENT, 2.0 * BOX_HALF_EXTENT),
                quat=tilt_quat,
                fixed=True,
            ),
            visualize_contact=True,
        )
    else:  # if ground_type == "prim":
        scene.add_entity(
            morph=gs.morphs.Box(
                pos=box_pos_world,
                quat=tilt_quat,
                size=(2.0 * BOX_HALF_EXTENT, 2.0 * BOX_HALF_EXTENT, HEIGHT),
                fixed=True,
            ),
            visualize_contact=True,
        )

    if entity_kind == "sphere":
        if entity_kind == "sphere" and entity_type in ("mesh", "nonconvex"):
            sphere_mesh = trimesh.creation.icosphere(
                radius=SMOOTH_RADIUS, subdivisions=SPHERE_TESSELLATION_SUBDIVISIONS
            )
            # Rotate the icosphere so that one face plane is perpendicular to the body -z axis. With the sphere oriented
            # to match the box tilt, this puts that face squarely against the box's top, eliminating the discretization
            # xy shift the sphere would otherwise pick up while rocking onto its nearest supporting feature. We align
            # the face's OUTWARD NORMAL with -z; aligning the centroid direction instead leaves the face plane slightly
            # tilted because for subdivided icosphere faces the centroid is not exactly along the face normal.
            bottom_dir = sphere_mesh.face_normals[int(np.argmin(sphere_mesh.face_normals[:, 2]))]
            cross_axis = np.cross(bottom_dir, np.array([0.0, 0.0, -1.0]))
            sin_t = float(np.linalg.norm(cross_axis))
            if sin_t > 1e-12:
                cross_axis = cross_axis / sin_t
                angle = np.arctan2(sin_t, float(np.dot(bottom_dir, np.array([0.0, 0.0, -1.0]))))
                sphere_mesh.apply_transform(trimesh.transformations.rotation_matrix(angle, cross_axis))
            is_entity_convex = entity_type == "mesh"
            entity = scene.add_entity(
                morph=gs.morphs.MeshSet(
                    files=(sphere_mesh,),
                    convexify=is_entity_convex,
                    decimate=False,
                ),
                vis_mode="collision",
                # visualize_contact=True,
            )
            # Manually overwrite convex flag to forcibly exercise non-convex collision path
            entity.geoms[0]._is_convex = is_entity_convex
        else:
            entity = scene.add_entity(
                morph=gs.morphs.Sphere(
                    radius=SMOOTH_RADIUS,
                ),
            )
    elif entity_kind == "cylinder":
        entity = scene.add_entity(
            morph=gs.morphs.Cylinder(
                radius=SMOOTH_RADIUS,
                height=CYLINDER_HEIGHT,
            ),
        )
    elif entity_kind == "capsule":
        # Two capsule lengths exist as separate entities: the zero-length capsule (sphere-like, used by "vertical-axis"
        # envs because a full-length capsule standing on its cap is a tippy-pencil configuration that is numerically
        # unstable regardless of the bias fix) and the full-length capsule (used by "horizontal-axis" envs, barrel
        # contact). MuJoCo rejects an exact zero length so we use a tiny positive value.
        entity = scene.add_entity(
            morph=(
                gs.morphs.MJCF(
                    file=_capsule_mjcf_path(tmp_path, SMOOTH_RADIUS, gs.EPS, name="capsule_v"),
                ),
                gs.morphs.MJCF(
                    file=_capsule_mjcf_path(tmp_path, SMOOTH_RADIUS, CYLINDER_HEIGHT, name="capsule_h"),
                ),
            )
        )
    else:  # if entity_kind == "ellipsoid":
        entity = scene.add_entity(
            morph=gs.morphs.MJCF(
                file=_ellipsoid_mjcf_path(tmp_path, ELLIPSOID_SEMI_AXES),
            ),
        )
    scene.build(n_envs=N_ENVS)

    # Randomly sample position in local frame.
    # Add small vertical offset to ensure contact at init; otherwise the primitive will sink before bouncing up.
    smooth_xy_local = np.random.uniform(
        low=-(BOX_HALF_EXTENT - 2.0 * SMOOTH_RADIUS),
        high=BOX_HALF_EXTENT - 2.0 * SMOOTH_RADIUS,
        size=(N_ENVS, 2),
    )
    smooth_pos_local = np.concatenate([smooth_xy_local, np.full((N_ENVS, 1), HEIGHT + SMOOTH_RADIUS - 1e-4)], axis=-1)

    # Randomly sample orientation in local frame.
    # Special handling for capsule to ensure stable barrel contact if needed.
    smooth_quat_local = np.random.uniform(low=-1.0, high=1.0, size=(N_ENVS, 4))
    if entity_kind in "cylinder":
        singular_mask = np.ones((N_ENVS,), dtype=np.bool_)
        angle_pitch = 0.5 * np.pi
    elif entity_kind in "ellipsoid":
        singular_mask = np.ones((N_ENVS,), dtype=np.bool_)
        angle_pitch = 0.0
    elif entity_kind == "capsule":
        singular_mask = np.arange(N_ENVS) >= N_ENVS // 2
        angle_pitch = 0.5 * np.pi
    else:
        # A tessellated ball has the resting face aligned above, so it keeps its body axis up and randomises only the
        # yaw, as every other entity here does. Orienting it freely instead stands the polytope on a vertex, which
        # topples into a facet basin and walks the ball an order of magnitude further than the settling it measures.
        is_tessellated = entity_type in ("mesh", "nonconvex")
        singular_mask = np.full((N_ENVS,), is_tessellated)
        angle_pitch = 0.0
    n_singulars = np.sum(singular_mask)
    angle_yaw = np.random.uniform(low=-np.pi, high=np.pi, size=(n_singulars, 1))
    smooth_quat_local[singular_mask] = gu.xyz_to_quat(
        np.concatenate([np.zeros((n_singulars, 1)), np.full((n_singulars, 1), angle_pitch), angle_yaw], axis=-1),
        rpy=True,
    )

    # Convert pose from local to world frame
    smooth_pos_world = smooth_pos_local @ R.T
    smooth_quat_world = gu.transform_quat_by_quat(smooth_quat_local, np.tile(tilt_quat, (N_ENVS, 1)))

    entity.set_pos(smooth_pos_world)
    entity.set_quat(smooth_quat_world)
    if show_viewer:
        scene.visualizer.update()

    for i_step in range(400):
        scene.step()

    pos_local = tensor_to_array(entity.get_pos()) @ R
    # A tessellated ball settles into the basin of whichever facet it rests on, which is what this bound covers: it
    # sits a little over twice the largest displacement any entity here needs, in either precision.
    assert_allclose(pos_local[..., :2], smooth_xy_local, atol=5e-4)


@pytest.mark.required
@pytest.mark.parametrize("precision", ["32"])
def test_mpr_thin_box_stack_no_lateral_phantom(show_viewer, tol):
    scene = gs.Scene(
        rigid_options=gs.options.RigidOptions(
            use_gjk_collision=False,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.1, -0.08, 0.06),
            camera_lookat=(0.0, 0.0, 0.01),
            camera_fov=20,
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(
        gs.morphs.Box(
            pos=(0.0, 0.0, 0.005),
            size=(0.002, 0.02, 0.01),
            fixed=True,
        ),
        surface=gs.surfaces.Default(
            color=(0, 0, 1),
        ),
    )
    box = scene.add_entity(
        gs.morphs.Box(
            pos=(0.0, 0.0, 0.01495),
            size=(0.002, 0.0199, 0.01),
        ),
        surface=gs.surfaces.Default(
            color=(1, 0, 0),
        ),
        visualize_contact=True,
    )
    scene.build()

    scene.step()
    contacts = scene.rigid_solver.collider.get_contacts(to_torch=False)
    normals = contacts["normal"]
    assert len(normals) > 0
    assert_allclose(np.abs(normals[..., 2]), 1, atol=1e2 * tol)

    for _ in range(100):
        scene.step()
    pos = box.get_pos()
    assert_allclose(pos[..., :2], 0, atol=1e1 * tol)
    assert_allclose(pos[..., 2], 0.015, atol=1e1 * tol)


@pytest.mark.required
@pytest.mark.parametrize("precision", ["32"])
@pytest.mark.parametrize("gjk_collision", [True, False])
def test_convex_collision_across_geom_scales(gjk_collision, show_viewer, tol):
    YAW = 1.1
    BOX_SIZE = 16.0
    GEOM_SIZE = 0.016
    # Collision tolerances scale with the smaller geom of a pair, rounding errors with its coordinates. A large size
    # ratio thus drives the tolerance below the rounding error. The grazing box sits where its separation from the face
    # is rounding noise, so contact detection must resolve the pair without any measurable depth to converge on. The
    # pressed box checks that such a pair still yields a contact.
    GRAZING_SPOT = (-3.36, 0.8)
    PRESSED_SPOT = (-3.3, 0.8)
    PRESSED_DEPTH = 0.25 * GEOM_SIZE

    scene = gs.Scene(
        rigid_options=gs.options.RigidOptions(
            use_gjk_collision=gjk_collision,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(6.6, 5.74, 8.85),
            camera_lookat=(6.6, 5.63, 8.8),
            camera_fov=30,
        ),
        show_viewer=show_viewer,
    )
    box = scene.add_entity(
        gs.morphs.Box(
            pos=(0.0, 0.0, 0.5 * BOX_SIZE),
            euler=(0.0, 0.0, math.degrees(YAW)),
            size=(BOX_SIZE, BOX_SIZE, BOX_SIZE),
            fixed=True,
        ),
    )
    geom_grazing = scene.add_entity(
        gs.morphs.Box(
            pos=(0.0, 0.0, 2.0 * BOX_SIZE),
            size=(GEOM_SIZE, GEOM_SIZE, GEOM_SIZE),
        ),
        visualize_contact=True,
        vis_mode="collision",
    )
    geom_pressed = scene.add_entity(
        gs.morphs.Box(
            pos=(0.0, 0.0, 3.0 * BOX_SIZE),
            size=(GEOM_SIZE, GEOM_SIZE, GEOM_SIZE),
        ),
        visualize_contact=True,
        vis_mode="collision",
    )
    scene.build()

    face_offset = 0.5 * (BOX_SIZE + GEOM_SIZE)
    face_normal = torch.tensor([math.cos(YAW), math.sin(YAW), 0.0], dtype=gs.tc_float, device=gs.device)
    for geom, spot, depth in ((geom_grazing, GRAZING_SPOT, 0.0), (geom_pressed, PRESSED_SPOT, PRESSED_DEPTH)):
        pos_local = torch.tensor([face_offset - depth, *spot], dtype=gs.tc_float, device=gs.device)
        geom.set_pos(gu.transform_by_quat(pos_local, box.get_quat()) + box.get_pos())
        geom.set_quat(box.get_quat())

    scene.step()
    contacts = scene.rigid_solver.collider.get_contacts()
    assert_allclose((contacts["normal"] @ face_normal).abs(), 1.0, tol=tol)
    is_pressed = contacts["geom_b"] == geom_pressed.geoms[0].idx
    assert is_pressed.any()
    assert_allclose(contacts["penetration"][is_pressed], PRESSED_DEPTH, tol=tol)
    assert (contacts["penetration"][~is_pressed] >= 0.0).all()
    assert (contacts["penetration"][~is_pressed] <= GEOM_SIZE).all()
    offset = (geom_grazing.get_pos() - box.get_pos()) @ face_normal
    assert face_offset - tol <= offset <= face_offset + GEOM_SIZE


@pytest.mark.slow  # ~200s
@pytest.mark.required
def test_robot_scaling_primitive_collision(show_viewer):
    scene = gs.Scene(
        show_viewer=show_viewer,
        show_FPS=False,
    )
    plane = scene.add_entity(
        gs.morphs.Plane(),
    )
    asset_path = get_hf_dataset(pattern="cross.xml")
    robot = scene.add_entity(
        gs.morphs.MJCF(
            file=f"{asset_path}/cross.xml",
            scale=0.5,
        ),
        vis_mode="collision",
    )
    scene.build()

    robot.set_qpos([0.0, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0, 1.0, -1.0, -1.0, 1.0])
    for _ in range(50):
        scene.step()

    # Robot not moving anymore
    assert_allclose(robot.get_links_vel(), 0.0, atol=5e-3)

    # Robot in contact with the ground
    robot_min_corner, _ = robot.get_AABB()
    assert_allclose(robot_min_corner[2], 0.0, tol=1e-3)


@pytest.mark.slow  # ~200s
@pytest.mark.required
@pytest.mark.parametrize("precision", ["32"])
@pytest.mark.parametrize("backend", [gs.gpu])
def test_contact_forces(show_viewer):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
        ),
        rigid_options=gs.options.RigidOptions(
            # Enabling box-box algorithm to improve code coverage
            box_box_detection=True,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3, -1, 1.5),
            camera_lookat=(0.0, 0.0, 0.5),
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )

    scene.add_entity(
        gs.morphs.Plane(),
    )
    franka = scene.add_entity(
        gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
    )
    cube = scene.add_entity(
        gs.morphs.Box(
            size=(0.04, 0.04, 0.04),
            pos=(0.65, 0.0, 0.02),
        ),
        # visualize_contact=True,
    )
    scene.build(n_envs=5)

    cube_weight = scene.rigid_solver.get_gravity(envs_idx=0) * cube.get_mass()
    motors_dof = np.arange(7)
    fingers_dof = np.arange(7, 9)
    qpos = np.array([-1.0124, 1.5559, 1.3662, -1.6878, -1.5799, 1.7757, 1.4602, 0.04, 0.04])
    franka.set_qpos(qpos)
    scene.step()

    end_effector = franka.get_link("hand")
    qpos = franka.inverse_kinematics(
        link=end_effector,
        pos=np.tile([0.65, 0.0, 0.13], (scene.n_envs, 1)),
        quat=np.tile([0, 1, 0, 0], (scene.n_envs, 1)),
    )
    franka.control_dofs_position(qpos[:, :-2], motors_dof)

    # hold
    for i in range(50):
        scene.step()
    contact_forces = cube.get_links_net_contact_force()
    assert_allclose(contact_forces[:, 0], -cube_weight, atol=1e-5)

    # grasp
    franka.control_dofs_position(qpos[:, :-2], motors_dof)
    franka.control_dofs_position(0.0, fingers_dof)
    for i in range(20):
        scene.step()

    # lift
    qpos = franka.inverse_kinematics(
        link=end_effector,
        pos=np.tile([0.65, 0.0, 0.2], (scene.n_envs, 1)),
        quat=np.tile([0.0, 1, 0, 0], (scene.n_envs, 1)),
    )
    franka.control_dofs_position(qpos[:, :-2], motors_dof)
    for i in range(100):
        scene.step()

    # Check contact forces while randomizing gripper orientations across parallel envs.
    # Note that it is necessary to reset the scene state because the box is slowly falling without noslip solver.
    state = scene.get_state()
    rng = np.random.RandomState(0)
    all_errors = []
    for i_trial in range(10):
        scene.reset(state)

        angles = rng.uniform(-np.deg2rad(45), np.deg2rad(45), size=scene.n_envs).astype(gs.np_float)
        axes = rng.randn(scene.n_envs, 3).astype(gs.np_float)
        perturbs = gu.axis_angle_to_quat(angles, axes)
        lift_quats = gu.transform_quat_by_quat(perturbs, np.tile([0, 1, 0, 0], (scene.n_envs, 1)).astype(gs.np_float))
        qpos = franka.inverse_kinematics(
            link=end_effector,
            pos=np.tile([0.65, 0.0, 0.2], (scene.n_envs, 1)).astype(gs.np_float),
            quat=lift_quats,
        )
        franka.control_dofs_position(qpos[:, :-2], motors_dof)
        franka.control_dofs_position(0.0, fingers_dof)
        for _ in range(160):
            scene.step()

        contact_forces = cube.get_links_net_contact_force()
        errors = torch.linalg.norm(contact_forces[:, 0, :] + cube_weight, ord=float("inf"), dim=-1)
        all_errors.append(errors)
    assert torch.quantile(torch.cat(all_errors), 0.95) < 2e-4


@pytest.mark.slow  # ~200s
@pytest.mark.required
@pytest.mark.xfail(reason="De-duplication of repeated contact points is currently too naive for this test to pass...")
@pytest.mark.parametrize("surface_kind", ["primitive_box", "primitive_plane", "vertex_box", "flat_terrain"])
def test_contact_dedup(surface_kind, show_viewer):
    SPHERE_RADIUS = 0.05
    GROUND_SIZE = 1.0

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.005,
        ),
        show_viewer=show_viewer,
    )
    if surface_kind == "primitive_box":
        scene.add_entity(
            morph=gs.morphs.Box(
                pos=(0.0, 0.0, -0.05),
                size=(GROUND_SIZE, GROUND_SIZE, 0.1),
                fixed=True,
            ),
        )
    elif surface_kind == "primitive_plane":
        scene.add_entity(
            morph=gs.morphs.Plane(
                pos=(0.0, 0.0, 0.0),
            ),
        )
    elif surface_kind == "vertex_box":
        box_mesh = trimesh.creation.box(extents=(GROUND_SIZE, GROUND_SIZE, 0.1))
        scene.add_entity(
            morph=gs.morphs.MeshSet(
                files=(box_mesh,),
                pos=(0.0, 0.0, -0.05),
                fixed=True,
            ),
        )
    elif surface_kind == "flat_terrain":
        flat_hf = np.zeros((16, 16), dtype=np.float32)
        scene.add_entity(
            morph=gs.morphs.Terrain(
                horizontal_scale=0.1,
                vertical_scale=1.0,
                height_field=flat_hf,
                pos=(-0.8, -0.8, 0.0),
            ),
        )
    sphere = scene.add_entity(
        morph=gs.morphs.MeshSet(
            files=(trimesh.creation.icosphere(radius=SPHERE_RADIUS, subdivisions=3),),
            pos=(0.0, 0.0, SPHERE_RADIUS - 1e-4),
            decimate=False,
        ),
        vis_mode="collision",
        visualize_contact=True,
    )
    scene.build()

    for i in range(80):
        scene.step()
        if i == 20:
            sphere.set_dofs_velocity(0.2, dofs_idx_local=sphere.dof_start)
        n_contacts = scene.rigid_solver.collider.collider_state.n_contacts.to_numpy()
        assert np.all(n_contacts == 1), f"Expected 1 contact after dedup, got {n_contacts}"


@pytest.mark.required
@pytest.mark.parametrize("gjk_collision", [True, False])
def test_contact_pruning(gjk_collision, show_viewer):
    GEOM_HALF_SIZE = 0.1
    MARGIN = 1e-4

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.005,
            gravity=(-1.0, -1.0, -1.0),
        ),
        rigid_options=gs.options.RigidOptions(
            contact_pruning_tolerance=0.02,
            # box_box_detection=True,
            use_gjk_collision=gjk_collision,
        ),
        vis_options=gs.options.VisOptions(
            rendered_envs_idx=(0,),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.4, 0.3, 0.3),
            camera_lookat=(0.0, 0.0, 0.0),
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(
        morph=gs.morphs.Box(
            size=(GEOM_HALF_SIZE, 1.0, 1.0),
            pos=(MARGIN - 1.5 * GEOM_HALF_SIZE, 0.0, 0.0),
            fixed=True,
        ),
        surface=gs.surfaces.Default(
            color=(1, 0, 0, 0.8),
        ),
    )
    scene.add_entity(
        morph=gs.morphs.Box(
            size=(1.0, GEOM_HALF_SIZE, 1.0),
            pos=(0.0, MARGIN - 1.5 * GEOM_HALF_SIZE, 0.0),
            fixed=True,
        ),
        surface=gs.surfaces.Default(
            color=(0, 1, 0, 0.8),
        ),
    )
    scene.add_entity(
        morph=gs.morphs.Box(
            size=(1.0, 1.0, GEOM_HALF_SIZE),
            pos=(0.0, 0.0, MARGIN - 1.5 * GEOM_HALF_SIZE),
            fixed=True,
        ),
        surface=gs.surfaces.Default(
            color=(0, 0, 1, 0.8),
        ),
    )

    sub_meshes = []
    for sx, sy, sz in product((-1, 0, +1), repeat=3):
        mesh = trimesh.creation.box(extents=(2 / 3 * GEOM_HALF_SIZE,) * 3)
        mesh.apply_translation((2 / 3 * sx * GEOM_HALF_SIZE, 2 / 3 * sy * GEOM_HALF_SIZE, 2 / 3 * sz * GEOM_HALF_SIZE))
        sub_meshes.append(mesh)
    box = scene.add_entity(
        morph=gs.morphs.MeshSet(
            files=sub_meshes,
        ),
        surface=gs.surfaces.Default(
            smooth=False,
        ),
        visualize_contact=True,
        vis_mode="collision",
    )
    scene.build(n_envs=2)

    for step_idx in range(200):
        scene.step()
        # Within each contact-normal bucket, every surviving contact must be a vertex of the 2D convex hull of
        # contacts' positions projected onto the plane perpendicular to that shared normal. The bucket key is the
        # contact's dominant axial direction (this scene's normals are nearly axial, so axis + sign is enough; we
        # don't need to be fully generic). Redundant (interior or hull-edge-midpoint) contacts and >2-collinear
        # contacts both indicate the pruning kernel left work undone.
        contacts = scene.rigid_solver.collider.get_contacts(to_torch=False)
        for i_b in range(scene.n_envs):
            positions = contacts["position"][i_b]
            normals = contacts["normal"][i_b]
            buckets: dict[tuple[int, int], list[int]] = {}
            for i in range(len(positions)):
                axis = int(np.argmax(np.abs(normals[i])))
                sign = 1 if normals[i][axis] > 0 else -1
                buckets.setdefault((axis, sign), []).append(i)
            for key, idxs in buckets.items():
                if len(idxs) < 3:
                    continue
                other_axes = [a for a in range(3) if a != key[0]]
                proj = positions[idxs][:, other_axes].astype(np.float64)
                diam = float(np.linalg.norm(proj.max(axis=0) - proj.min(axis=0)))
                if diam < 1e-6:
                    continue
                try:
                    # Qhull's E tolerance merges nearly-collinear points into hull edges; without it, float noise on
                    # the order of 1e-6 hides the collinearity that the pruning kernel is supposed to detect.
                    hull = ConvexHull(proj, qhull_options=f"Qt E{diam * 1e-3}")
                    n_hull_vertices = len(hull.vertices)
                except QhullError:
                    raise AssertionError(
                        f"step {step_idx}, bucket axis={key[0]} sign={key[1]}: {len(idxs)} contacts are collinear in "
                        f"the contact plane. The pruning kernel should have kept at most 2 of them."
                    ) from None
                if n_hull_vertices == len(idxs):
                    continue
                non_hull = sorted(set(range(len(idxs))) - set(hull.vertices.tolist()))
                details = "\n".join(
                    f"    [{i}] contact={idxs[i]} pos={positions[idxs[i]]} proj={proj[i]}"
                    f"{'  <-- REDUNDANT' if i in non_hull else ''}"
                    for i in range(len(idxs))
                )
                raise AssertionError(
                    f"step {step_idx}, bucket axis={key[0]} sign={key[1]}: {len(idxs)} surviving contacts but only "
                    f"{n_hull_vertices} are vertices of the bucket's 2D convex hull. The pruning kernel should have "
                    f"dropped these {len(idxs) - n_hull_vertices} redundant contact(s):\n{details}"
                )
    assert_allclose(box.get_pos(), 0.0, atol=2e-3)


@pytest.mark.required
@pytest.mark.precision("32")
@pytest.mark.parametrize("gjk_collision", [False, True])
def test_contact_pruning_authored_decomp(gjk_collision, show_viewer):
    # A central pole carries six concentric rings, capped by a ball seated in the top ring's hole. Each ring collision
    # mesh is pre-decomposed into N_WEDGES convex slices, so stacked pieces touch face-to-face along the vertical axis.
    # Physically only vertical contacts are valid between stacked rings; any lateral contact is a spurious cross-sector
    # overlap of the convex decomposition. The ball rests on the curved hole surface, so it legitimately produces angled
    # normals and is exempt from the vertical-normal and one-per-slice checks.
    N_WEDGES = 16
    BASE_HEIGHT = 0.020
    RING_HEIGHT = 0.020
    BALL_HEIGHT = 0.019
    RINGS_ORDER = (0, 1, 2, 3, 5, 4)

    NUM_CHECKS = 10
    POS_TOL = 2e-3
    # FIXME: The top ball is slightly rotating around z-axis (~0.5degree)
    ROT_TOL = 1e-2

    scene = gs.Scene(
        rigid_options=gs.options.RigidOptions(
            max_collision_pairs=1200,
            use_gjk_collision=gjk_collision,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.4, 0.0, 0.3),
            camera_lookat=(0.0, 0.0, 0.1),
        ),
        show_viewer=show_viewer,
    )
    plane = scene.add_entity(gs.morphs.Plane())
    pole_pos = (0.0, 0.0, BASE_HEIGHT / 2)
    pole = scene.add_entity(
        morph=gs.morphs.URDF(
            file="tower/base_pole.urdf",
            pos=pole_pos,
            file_meshes_are_zup=True,
        ),
        material=gs.materials.Rigid(
            rho=600.0,
        ),
        vis_mode="collision",
    )
    poss_init = [pole_pos]
    rpys_init = [(0.0, 0.0, 0.0)]
    rings = []
    height = BASE_HEIGHT
    for i, ring_idx in enumerate(RINGS_ORDER):
        ring_pos = (0.0, 0.0, height + (RING_HEIGHT - 1e-4) / 2)
        # Alternate rotational offset along z-axis to avoid lateral contacts
        ring_yaw = 180 / N_WEDGES * (i % 2)
        ring = scene.add_entity(
            morph=gs.morphs.URDF(
                file=f"tower/ring_{ring_idx + 1:02d}.urdf",
                pos=ring_pos,
                euler=(0.0, 0.0, ring_yaw),
                file_meshes_are_zup=True,
            ),
            material=gs.materials.Rigid(
                rho=600.0,
            ),
            vis_mode="collision",
            visualize_contact=True,
        )
        rings.append(ring)
        poss_init.append(ring_pos)
        rpys_init.append((0.0, 0.0, np.deg2rad(ring_yaw)))
        height += RING_HEIGHT - 1e-4
    ball_pos = (0.0, 0.0, height + BALL_HEIGHT)
    ball = scene.add_entity(
        morph=gs.morphs.URDF(
            file="tower/ball.urdf",
            pos=ball_pos,
            file_meshes_are_zup=True,
        ),
        material=gs.materials.Rigid(
            rho=600.0,
        ),
        vis_mode="collision",
    )
    poss_init.append(ball_pos)
    rpys_init.append((0.0, 0.0, 0.0))
    scene.build()

    geom_owner = {geom.idx: entity for entity in (plane, pole, *rings, ball) for geom in entity.geoms}
    ring_geoms = {geom.idx for ring in rings for geom in ring.geoms}
    ball_geoms = {geom.idx for geom in ball.geoms}

    # Tiny warm-up to deal with initial penetration (~5e-4)
    for _ in range(2):
        scene.step()

    # Check that the tower stay in place
    for _ in range(20):
        scene.step()
        for entity, pos_init, rpy_init in zip((pole, *rings, ball), poss_init, rpys_init):
            assert_allclose(entity.get_pos(), pos_init, atol=POS_TOL)
            assert_allclose(gu.quat_to_xyz(entity.get_quat(), rpy=True), rpy_init, atol=ROT_TOL)
        # Only check linear velocity at CoM and angular velocity around z-axis.
        # It is robust to loosing a few contact points while still asserting the failure modes that matter.
        assert_allclose(scene.rigid_solver.get_dofs_velocity(dofs_idx=(0, 1, 2, 5)), 0, tol=0.06)

    # A contact step is "ideal" when both invariants hold across all stacked interfaces (the ball seats on a curved
    # hole and is exempt from both):
    #   - normals are vertical: only axial contacts are physical between stacked rings; a lateral normal is a spurious
    #     cross-sector overlap of the convex decomposition,
    #   - pruning collapses each wedge-pair manifold to one contact per slice, so every pole-ring / ring-ring interface
    #     carries at most N_WEDGES contacts (without pruning each manifold would emit many more).
    # Both invariants fail together on a bad step (a spurious lateral overlap also inflates the slice count). MPR keeps
    # the sub-resolution overlaps below the rejection floor on every step; GJK's tighter penetration estimates let one
    # spike above it occasionally in fp32, so it only has to be ideal at least once.
    for _ in range(NUM_CHECKS):
        scene.step()
        contacts = scene.rigid_solver.collider.get_contacts(to_torch=False)
        geom_a, geom_b = contacts["geom_a"], contacts["geom_b"]
        penetration = contacts["penetration"]
        normal_z = contacts["normal"][:, 2]
        interface_counts = {}
        is_vertical = True
        for i in range(len(geom_a)):
            if penetration[i] <= 0.0:
                continue
            a, b = int(geom_a[i]), int(geom_b[i])
            if a in ball_geoms or b in ball_geoms:
                continue
            if abs(normal_z[i]) < 0.5:
                is_vertical = False
            if a in ring_geoms or b in ring_geoms:
                key = frozenset((geom_owner[a], geom_owner[b]))
                interface_counts[key] = interface_counts.get(key, 0) + 1
        # pole-ring0 plus each ring-ring interface up the stack
        is_pruned = len(interface_counts) == len(rings) and all(
            count <= N_WEDGES for count in interface_counts.values()
        )
        assert is_vertical and is_pruned


@pytest.mark.slow  # ~200s
@pytest.mark.required
@pytest.mark.parametrize(
    "model_name",
    [
        "side_by_side_capsules",
        "collinear_capsules",
        "side_by_side_cylinders",
        "collinear_cylinders",
        "collinear_spheres",
    ],
)
def test_contact_pruning_degenerated_hull(model_name, xml_path, show_viewer):
    HEIGHT = 0.02
    BOX_HALFSIZE = 0.15
    PRIM_RADIUS = 0.0025
    PRIM_LENGTH = 0.02
    N_ENVS = 16

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.004,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.25, 0.25, 0.2),
            camera_lookat=(0.0, 0.0, 0.5 * HEIGHT),
            camera_fov=30.0,
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(
        morph=gs.morphs.Box(
            size=(2 * BOX_HALFSIZE, 2 * BOX_HALFSIZE, HEIGHT),
            pos=(0.0, 0.0, 0.5 * HEIGHT),
            fixed=True,
        ),
        visualize_contact=True,
    )
    entity = scene.add_entity(
        morph=gs.morphs.MJCF(
            file=xml_path,
        ),
        surface=gs.surfaces.Default(
            smooth=False,
        ),
    )
    scene.build(n_envs=N_ENVS)

    # Randomly sample position in local frame.
    # Add small vertical offset to ensure contact at init; otherwise the primitive will sink before bouncing up.
    smooth_xy = np.random.uniform(
        low=-(BOX_HALFSIZE - 2.0 * PRIM_LENGTH), high=BOX_HALFSIZE - 2.0 * PRIM_LENGTH, size=(N_ENVS, 2)
    )
    smooth_pos = np.concatenate([smooth_xy, np.full((N_ENVS, 1), HEIGHT + PRIM_RADIUS - 1e-4)], axis=-1)
    entity.set_pos(smooth_pos)

    # Random yaw about world z; capsules/cylinders stay horizontal since their fromto axis lies in the body xy plane.
    angle_yaw = np.random.uniform(low=-np.pi, high=np.pi, size=(N_ENVS, 1))
    smooth_quat = gu.xyz_to_quat(np.concatenate([np.zeros((N_ENVS, 2)), angle_yaw], axis=-1), rpy=True)
    entity.set_quat(smooth_quat)

    if show_viewer:
        scene.visualizer.update()

    for _ in range(20):
        scene.step()
    for _ in range(300):
        scene.step()
        n_contacts = scene.rigid_solver.collider.collider_state.n_contacts.to_numpy()
        assert n_contacts.all()
        if model_name.startswith("side_by_side"):
            assert (n_contacts == 4).all()
        elif model_name == "collinear_spheres":
            assert (n_contacts == 2).all()

    assert_allclose(entity.get_pos()[..., :2], smooth_xy, atol=1e-3)


@pytest.mark.slow("gpu")  # gpu ~250s
@pytest.mark.parametrize(
    "scene_kind, max_collision_pairs, max_contacts, error_pattern",
    [
        # Post-pruning contact budget overflow, with the candidate buffer large enough (2x margin) that it cannot
        # trip first. The automatic budget resolves to 32 contact points per link pair floored at 512, far below
        # what the piled-up bowls produce.
        pytest.param("bowls", 1_000, None, "max number of post-pruning contact points", marks=pytest.mark.required),
        # Candidate contact buffer overflow. The explicit contact budget is clamped down to the buffer size, so only
        # the buffer itself can overflow.
        ("bowls", 150, 1_000, "max number of candidate contact points"),
        # Buffers large enough for the whole pile: no overflow at all. Both values keep a 2x margin over the peaks
        # reached within the stepped window (about 500 colliding geom pairs and 1040 post-pruning contact points).
        ("bowls", 1_000, 2_000, None),
        # Two contacts against a budget of one: the clamp must also run when the contact count is below the pruning
        # gate (n_contacts < 3), in both the serial and the GPU cooperative kernel variants.
        ("spheres", 150, 1, "max number of post-pruning contact points"),
    ],
)
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
def test_num_contact_overflow(scene_kind, max_collision_pairs, max_contacts, error_pattern, show_viewer):
    from genesis.engine.simulator import RATE_CHECK_ERRNO

    N_BOWLS = 4
    scene = gs.Scene(
        rigid_options=gs.options.RigidOptions(
            max_collision_pairs=max_collision_pairs,
            max_contacts=max_contacts,
        ),
        renderer=gs.renderers.Rasterizer(),
        show_viewer=show_viewer,
    )
    scene.add_entity(
        morph=gs.morphs.Plane(),
    )
    if scene_kind == "bowls":
        asset_path = get_hf_dataset(pattern="glb/orange_plastic_bowl.glb")
        for _ in range(N_BOWLS):
            scene.add_entity(
                morph=gs.morphs.Mesh(
                    file=f"{asset_path}/glb/orange_plastic_bowl.glb",
                    pos=(0, 0, 0.5),
                    euler=(90, 0, 0),
                    convexify=True,
                    file_meshes_are_zup=True,
                ),
            )
    else:
        # Non-contacting nonconvex mesh: makes the scene prunable so that the GPU cooperative kernel is exercised.
        scene.add_entity(
            morph=gs.morphs.Mesh(
                file="meshes/duck.obj",
                scale=0.04,
                pos=(5.0, 5.0, 5.0),
                convexify=False,
            ),
        )
        for i in range(2):
            scene.add_entity(
                morph=gs.morphs.Sphere(
                    pos=(0.5 * i, 0.0, 0.0999),
                    radius=0.1,
                ),
            )
    scene.build()
    assert scene.rigid_solver.collider.collider_config.has_prunable_contacts

    # The resolved contact budget must match the documented resolution: 32 contact points per link pair floored at
    # 512 when automatic (every link pair here has more than 32 candidate contact points), the explicit value clamped
    # to the candidate buffer size otherwise. The constraint buffers are sized accordingly, with 4 constraint rows
    # per contact point (all joints are free so there is no joint-limit term).
    solver = scene.rigid_solver
    collider_info = solver.collider.collider_info
    if max_contacts is None:
        n_link_pairs = (N_BOWLS + 1) * N_BOWLS // 2
        expected_max_contacts = max(32 * n_link_pairs, 512)
    else:
        expected_max_contacts = min(max_contacts, int(collider_info.max_candidate_contacts[None]))
    assert int(collider_info.max_contacts[None]) == expected_max_contacts
    expected_len_constraints = 4 * expected_max_contacts + solver.n_dofs + 6 * solver.n_candidate_equalities_
    assert solver.constraint_solver.len_constraints == expected_len_constraints

    # All overflows occur on the very first step (the bowls start fully overlapping, the spheres start resting on the
    # plane), but errno is only polled every RATE_CHECK_ERRNO substeps, so one extra step is required to guarantee
    # that the error gets raised.
    with nullcontext() if error_pattern is None else pytest.raises(gs.GenesisException, match=error_pattern):
        for _ in range(RATE_CHECK_ERRNO + 1):
            scene.step()


@pytest.mark.slow  # ~200s
@pytest.mark.required
def test_filter_neutral_self_collisions(show_viewer):
    scene = gs.Scene(
        rigid_options=gs.options.RigidOptions(
            enable_self_collision=True,
            enable_neutral_collision=False,
            enable_adjacent_collision=False,
        ),
        show_viewer=show_viewer,
    )
    robot = scene.add_entity(
        gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
    )
    sphere = scene.add_entity(
        gs.morphs.Sphere(
            radius=0.08,
        ),
        surface=gs.surfaces.Default(
            color=(0.0, 2.0, 0.0, 1.0),
        ),
    )
    box = scene.add_entity(
        gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
        ),
        surface=gs.surfaces.Default(
            color=(1.0, 0.0, 0.0, 1.0),
        ),
    )
    sphere.attach(robot, "hand")
    scene.build()
    eq_type = scene.rigid_solver.dyn_info.equalities.eq_type.to_numpy()[: scene.rigid_solver.n_equalities, 0]
    eq_obj1id = scene.rigid_solver.dyn_info.equalities.eq_obj1id.to_numpy()[: scene.rigid_solver.n_equalities, 0]
    eq_obj2id = scene.rigid_solver.dyn_info.equalities.eq_obj2id.to_numpy()[: scene.rigid_solver.n_equalities, 0]

    scene.rigid_solver.collider.detection()
    contacts_data = scene.rigid_solver.collider.get_contacts()
    assert ((contacts_data["link_a"] == 12) & (contacts_data["link_b"] == 0)).any()

    for i in range(2):
        for i_ga in range(robot.geom_start, box.geom_start):
            for i_gb in range(i_ga + 1, box.geom_start):
                geom_a = scene.rigid_solver.geoms[i_ga]
                geom_b = scene.rigid_solver.geoms[i_gb]
                link_a = geom_a.link
                link_b = geom_b.link

                if link_a.idx == link_b.idx:
                    continue

                if link_a.is_fixed and link_b.is_fixed:
                    continue

                if (
                    (eq_type == gs.EQUALITY_TYPE.WELD)
                    & (
                        (eq_obj1id == link_a.idx & eq_obj2id == link_b.idx)
                        | (eq_obj1id == link_b.idx & eq_obj2id == link_a.idx)
                    )
                ).any():
                    continue

                is_adjacent = False
                link = link_b
                while link.parent_idx > 0:
                    if link.parent_idx == link_a.idx:
                        is_adjacent = True
                        break
                    if not all(joint.type is gs.JOINT_TYPE.FIXED for joint in link.joints):
                        break
                    link = scene.rigid_solver.links[link.parent_idx]
                if is_adjacent:
                    continue

                verts_a = tensor_to_array(geom_a.get_verts())
                verts_a = (1.0 - 1e-3) * verts_a + 1e-3 * verts_a.mean(axis=0, keepdims=True)
                mesh_a = trimesh.Trimesh(vertices=verts_a, faces=geom_a.init_faces, process=False)
                geom_b = scene.rigid_solver.geoms[i_gb]
                verts_b = tensor_to_array(geom_b.get_verts())
                verts_b = (1.0 - 1e-3) * verts_b + 1e-3 * verts_b.mean(axis=0, keepdims=True)
                mesh_b = trimesh.Trimesh(vertices=verts_b, faces=geom_b.init_faces, process=False)
                is_colliding = mesh_a.contains(mesh_b.vertices).any() or mesh_b.contains(mesh_a.vertices).any()
                assert is_colliding == ({(i_ga, i_gb)} in ({(5, 10)}, {(6, 10)}, {(11, 23)}, {(17, 23)}))
        scene.step()


@pytest.mark.slow  # ~200s
@pytest.mark.required
def test_contype_conaffinity(show_viewer, tol):
    GRAVITY = (0.0, 0.0, -10.0)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            gravity=GRAVITY,
        ),
        show_viewer=show_viewer,
    )

    plane = scene.add_entity(
        gs.morphs.Plane(
            pos=(0.0, 0.0, 0.0),
        )
    )
    box1 = scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.5, 0.5, 0.5),
            pos=(0.0, 0.0, 0.5),
            contype=3,
            conaffinity=3,
        ),
        surface=gs.surfaces.Default(
            color=(1.0, 0.0, 0.0, 1.0),
        ),
    )
    box2 = scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.5, 0.5, 0.5),
            pos=(0.0, 0.0, 1.0),
            contype=2,
            conaffinity=2,
        ),
        surface=gs.surfaces.Default(
            color=(0.0, 1.0, 0.0, 1.0),
        ),
    )
    box3 = scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.5, 0.5, 0.5),
            pos=(0.0, 0.0, 1.5),
            contype=1,
            conaffinity=1,
        ),
        surface=gs.surfaces.Default(
            color=(0.0, 0.0, 1.0, 1.0),
        ),
        visualize_contact=True,
    )
    box4 = scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.5, 0.5, 0.5),
            pos=(0.0, 0.0, 2.0),
            contype=0,
            conaffinity=0,
        ),
        surface=gs.surfaces.Default(
            color=(0.8, 0.8, 0.8, 1.0),
        ),
        visualize_contact=True,
    )
    scene.build()

    for _ in range(80):
        scene.step()

    assert_allclose(box1.get_pos(), (0.0, 0.0, 0.25), atol=5e-4)
    assert_allclose(box2.get_pos(), (0.0, 0.0, 0.75), atol=2e-3)
    assert_allclose(box2.get_pos(), box3.get_pos(), atol=2e-3)
    assert_allclose(scene.rigid_solver.get_links_acc(slice(box4.link_start, box4.link_end)), GRAVITY, atol=tol)


@pytest.mark.required
@pytest.mark.precision("32")
@pytest.mark.parametrize("backend", [gs.gpu])
@pytest.mark.parametrize("contact_pruning_tolerance", [0.02, None], ids=["prune", "noprune"])
@pytest.mark.parametrize(
    "prefer_decomposed_solver",
    [
        pytest.param(False, marks=pytest.mark.use_deterministic_algorithms(False)),
        pytest.param(True, marks=pytest.mark.use_deterministic_algorithms(False)),
        None,
    ],
    ids=["monolith", "decomposed", "deterministic"],
)
def test_gpu_simulation_determinism(prefer_decomposed_solver, contact_pruning_tolerance, monkeypatch, show_viewer):
    # Run-to-run reproducibility on GPU: from an identical initial state, every trial must reproduce a bit-identical
    # trajectory. CPU is serialized and deterministic by construction, so this targets GPU parallel races only
    # (atomic_add slot reservation, parallel reductions, scheduling). The two registered solve implementations are
    # numerically distinct, so each is pinned via prefer_decomposed_solver to isolate physics-kernel determinism per
    # variant. The third case asks for neither and runs under 'use_deterministic_algorithms', whose job is precisely
    # to resolve that choice itself instead of leaving it to a timing measurement that varies with machine load.
    #
    # The authored-decomposition tower is the stress case: stacked rings pre-split into convex wedges produce many
    # multi-contact manifolds per geom pair, exercising the narrowphase, contact pruning, the contact sort, and the
    # contact-coupled solve. The per-step fingerprints are compared in pipeline order so the assertion names the
    # earliest diverging stage, pinpointing the root:
    #   - contact set    -> narrowphase / pruning
    #   - contact order  -> contact sort
    #   - dofs velocity  -> constraint solve
    from genesis.utils.array_class import RigidSimStaticConfig

    if prefer_decomposed_solver is not None:
        init_orig = RigidSimStaticConfig.__init__

        def init_forced(self, *args, **kwargs):
            kwargs["prefer_decomposed_solver"] = int(prefer_decomposed_solver)
            init_orig(self, *args, **kwargs)

        monkeypatch.setattr(RigidSimStaticConfig, "__init__", init_forced)

    N_TRIALS = 8
    N_STEPS = 25
    N_WEDGES = 16
    BASE_HEIGHT = 0.020
    RING_HEIGHT = 0.020
    BALL_HEIGHT = 0.019
    RINGS_ORDER = (0, 1, 2, 3, 5, 4)

    scene = gs.Scene(
        rigid_options=gs.options.RigidOptions(
            max_collision_pairs=1200,
            contact_pruning_tolerance=contact_pruning_tolerance,
            use_gjk_collision=True,
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(gs.morphs.Plane())
    scene.add_entity(
        morph=gs.morphs.URDF(
            file="tower/base_pole.urdf",
            pos=(0.0, 0.0, BASE_HEIGHT / 2),
            file_meshes_are_zup=True,
        ),
        material=gs.materials.Rigid(
            rho=600.0,
        ),
    )
    height = BASE_HEIGHT
    for i, ring_idx in enumerate(RINGS_ORDER):
        scene.add_entity(
            morph=gs.morphs.URDF(
                file=f"tower/ring_{ring_idx + 1:02d}.urdf",
                pos=(0.0, 0.0, height + (RING_HEIGHT - 1e-4) / 2),
                # Alternate rotational offset along z-axis to avoid lateral contacts
                euler=(0.0, 0.0, 180 / N_WEDGES * (i % 2)),
                file_meshes_are_zup=True,
            ),
            material=gs.materials.Rigid(
                rho=600.0,
            ),
        )
        height += RING_HEIGHT - 1e-4
    ball = scene.add_entity(
        morph=gs.morphs.URDF(
            file="tower/ball.urdf",
            pos=(0.0, 0.0, height + BALL_HEIGHT),
            file_meshes_are_zup=True,
        ),
        material=gs.materials.Rigid(
            rho=600.0,
        ),
    )
    scene.build()
    solver = scene.rigid_solver

    # The ball is a sphere seated in the top ring's hole, so every ball contact normal must point radially
    ball_geoms_idx = {geom.idx for geom in ball.geoms}
    ball_center = np.atleast_2d(tensor_to_array(ball.get_pos()))[0]
    solver.collider.detection()
    contacts = solver.collider.get_contacts(to_torch=False)
    geom_a, geom_b = contacts["geom_a"], contacts["geom_b"]
    position, normal, penetration = contacts["position"], contacts["normal"], contacts["penetration"]
    for i in range(len(geom_a)):
        if penetration[i] <= 0.0 or (geom_a[i] not in ball_geoms_idx and geom_b[i] not in ball_geoms_idx):
            continue
        radial = ball_center - position[i]
        radial /= np.linalg.norm(radial)
        cos_angle = min(1.0, abs(np.dot(normal[i], radial)))
        assert np.degrees(np.arccos(cos_angle)) < 15.0

    # trials[trial][step] = (contact_set, contact_order, dofs_velocity, dofs_position)
    trials = []
    for _ in range(N_TRIALS):
        scene.reset()
        steps = []
        for _ in range(N_STEPS):
            scene.step()
            contacts = solver.collider.get_contacts(to_torch=False)
            geom_a, geom_b = contacts["geom_a"], contacts["geom_b"]
            position, normal, penetration = contacts["position"], contacts["normal"], contacts["penetration"]
            contact_order = tuple(
                (geom_a[i], geom_b[i], *position[i], *normal[i], penetration[i]) for i in range(len(geom_a))
            )
            dofs_velocity = tensor_to_array(solver.get_dofs_velocity()).copy()
            dofs_position = tensor_to_array(solver.get_qpos()).copy()
            steps.append((frozenset(contact_order), contact_order, dofs_velocity, dofs_position))
        trials.append(steps)

    ref = trials[0]
    for trial in range(1, N_TRIALS):
        for step in range(N_STEPS):
            ref_set, ref_order, ref_vel, ref_pos = ref[step]
            cur_set, cur_order, cur_vel, cur_pos = trials[trial][step]
            assert cur_set == ref_set
            assert cur_order == ref_order
            assert_equal(cur_vel, ref_vel)
            assert_equal(cur_pos, ref_pos)


@pytest.mark.required
@pytest.mark.xfail(reason="No reliable way to generate nan...")
@pytest.mark.parametrize("mode", [3])
@pytest.mark.parametrize("model_name", ["collision_edge_cases"])
@pytest.mark.parametrize("gs_solver", [gs.constraint_solver.CG])
@pytest.mark.parametrize("gs_integrator", [gs.integrator.Euler])
def test_nan_reset(gs_sim, mode):
    for _ in range(200):
        gs_sim.scene.step()
        qvel = gs_sim.rigid_solver.get_dofs_velocity()
        if torch.isnan(qvel).any():
            break
    else:
        raise AssertionError

    gs_sim.scene.reset()
    for _ in range(5):
        gs_sim.scene.step()
    qvel = gs_sim.rigid_solver.get_dofs_velocity()
    assert not torch.isnan(qvel).any()


@pytest.mark.required
def test_neutral_self_collision_masks_across_merged_entities(merged_overlapping_models, show_viewer):
    # attach() merges the hand into the arm's kinematic tree, and self-collision masking (adjacency and neutral overlap)
    # keys on root_idx, so it must span the merge boundary. The palm geom overlaps the non-adjacent a2 link at the
    # neutral pose, so that cross-entity pair must be masked out.
    arm_xml, hand_xml = merged_overlapping_models
    scene = gs.Scene(
        rigid_options=gs.options.RigidOptions(
            enable_self_collision=True,
            enable_adjacent_collision=False,
        ),
        show_viewer=show_viewer,
    )
    arm = scene.add_entity(
        gs.morphs.MJCF(
            file=arm_xml,
        ),
    )
    hand = scene.add_entity(
        gs.morphs.MJCF(
            file=hand_xml,
        ),
    )
    hand.attach(arm, "tip")
    scene.build()

    # The merged hand shares the arm's kinematic-tree root, so cross-entity pairs are self-collision candidates.
    geoms_root_idx = np.array([geom.link.root_idx for geom in scene.rigid_solver.geoms])
    arm_geoms = [geom.idx for geom in arm.geoms]
    hand_geoms = [geom.idx for geom in hand.geoms]
    assert set(geoms_root_idx[arm_geoms].tolist()) == set(geoms_root_idx[hand_geoms].tolist())

    # The palm overlaps the non-adjacent a2 link at qpos0, so the neutral-overlap check masks this cross-entity pair.
    collision_pair_idx = scene.rigid_solver.collider.collider_info.collision_pair_idx.to_numpy()
    a2_geom = arm.get_link("a2").geoms[0].idx
    palm_geom = hand.get_link("palm").geoms[0].idx
    assert_equal(collision_pair_idx[a2_geom, palm_geom], -1)
    assert_equal(collision_pair_idx[palm_geom, a2_geom], -1)

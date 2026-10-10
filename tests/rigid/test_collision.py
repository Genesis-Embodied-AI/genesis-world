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
    # A tessellated ball resting on a facet stands on the corners of that facet, around a centre of mass that projects
    # slightly off their centroid. The contacts are compliant, so the torque they oppose to a tilt grows with the square
    # of the facet size, while the inertia of the ball grows with the square of its radius. The finest tessellation
    # whose facets still rebalance that offset is used, since a finer one tilts onto the next facet and rolls away.
    SPHERE_TESSELLATION_SUBDIVISIONS = 2

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


@pytest.mark.slow  # ~150s
@pytest.mark.required
@pytest.mark.parametrize("detection", ["mpr", "gjk", "box_box"])
def test_box_contact_minimal_separation(detection, show_viewer, tol):
    # A contact is a point and a normal along which the depth is the smallest displacement separating both boxes. For a
    # box pair, that displacement is the smallest overlap over the 15 separating axes (the face normals of either box and
    # the cross products of their edges), which gives an exact reference however the boxes are posed.
    N_ENVS = 64
    N_ROUNDS = 8
    N_STEPS = 3
    CAMERA_FOV = 60.0
    SCALES = (0.05, 0.2, 1.0, 5.0)
    BOXES_SIZE = np.array(((1.0, 0.6, 0.02), (0.8, 0.05, 0.05), (0.5, 0.4, 0.1), (0.3, 0.3, 0.3)))
    BOX_EULER_ROTS = ((0, 0, 0), (180, 0, 0), (90, 0, 0), (-90, 0, 0), (0, -90, 0), (0, 90, 0))
    # Angle within which the normal of Minkowski Portal Refinement (MPR) follows a separating axis of the pair, as
    # bounded by the convergence of its portal
    MPR_AXIS_ANGLE_TOL = 2e-3

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            gravity=(0.0, 0.0, 0.0),
        ),
        rigid_options=gs.options.RigidOptions(
            use_gjk_collision=detection == "gjk",
            box_box_detection=detection == "box_box",
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.0, 0.0, 0.0),
            camera_lookat=(0.0, 1.0, -1.0),
            camera_fov=CAMERA_FOV,
        ),
        show_viewer=show_viewer,
    )
    # Every box takes each shape in some environments, in an order shuffled per box. The pairs are laid out in the view
    # as a grid, each one away from the camera in proportion to its scale, so that all of them look alike while lying at
    # depths far enough apart for their boxes to never reach each other. The smallest ones also sit near the origin,
    # where rounding errors are the smallest.
    pairs_boxes = []
    for i_s, scale in enumerate(SCALES):
        view_x, view_y = 0.7 * (2 * (i_s % 2) - 1), 0.3 * (2 * (i_s // 2) - 1)
        pair_pos = 3.0 * scale * np.array((view_x, 1.0 + view_y, view_y - 1.0))
        pairs_boxes.append(
            [
                scene.add_entity(
                    morph=[
                        gs.morphs.Box(pos=pair_pos + (0.0, 0.0, 3.0 * scale * i_box), size=scale * box_size)
                        for box_size in np.random.permutation(BOXES_SIZE)
                    ],
                    vis_mode="collision",
                )
                for i_box in range(2)
            ]
        )
    scene.build(n_envs=N_ENVS)
    n_pairs = len(pairs_boxes)
    # The armature bounds the response of the thinnest boxes, whose inertia would otherwise call for a smaller time step
    for boxes in pairs_boxes:
        for box in boxes:
            box.set_dofs_armature(0.1)

    # The boxes are built axis-aligned, so their bounding boxes give the half-size each environment simulates
    boxes_aabb = np.stack(
        [np.stack([tensor_to_array(box.get_AABB()) for box in boxes], axis=1) for boxes in pairs_boxes], axis=1
    )
    boxes_half = 0.5 * (boxes_aabb[..., 1, :] - boxes_aabb[..., 0, :])
    pairs_pos = 0.5 * (boxes_aabb[:, :, 0, 0] + boxes_aabb[:, :, 0, 1])
    pairs_size = np.linalg.norm(boxes_half, axis=-1).sum(axis=-1)
    pairs_scale = np.linalg.norm(boxes_half, axis=-1).min(axis=-1)
    # Multi-contact detection tilts each box of a pair about its first contact by the perturbation angle. A point found
    # in the tilted pose moves by up to this angle times the diameter of its box, as does the overlap along any normal.
    pairs_tol = 2.0 * scene.rigid_solver.collider._mc_perturbation * pairs_size

    links_pair_idx = np.full(scene.rigid_solver.n_links, -1)
    links_box_idx = np.full(scene.rigid_solver.n_links, -1)
    for i_p, boxes in enumerate(pairs_boxes):
        for i_box, box in enumerate(boxes):
            links_pair_idx[box.base_link_idx] = i_p
            links_box_idx[box.base_link_idx] = i_box

    for i_round in range(N_ROUNDS):
        # Fully random poses, faces aligned up to a random yaw and a tiny tilt, edges nearly parallel, exactly aligned. The
        # draw favors the families whose contacts span an edge or a face over the vertex contacts of random poses.
        modes = np.random.choice(4, size=(N_ENVS, n_pairs), p=(0.1, 0.3, 0.4, 0.2))
        quats_rand = gu.random_quaternion(2 * N_ENVS * n_pairs).reshape((N_ENVS, n_pairs, 2, 4))
        quats_face = gu.euler_to_quat(np.take(BOX_EULER_ROTS, np.random.randint(6, size=(N_ENVS, n_pairs, 2)), axis=0))
        tilts = 10.0 ** np.random.uniform(-7.0, -2.0, size=(N_ENVS, n_pairs, 2))
        tilts *= np.where(modes[..., None] == 3, 0.0, np.random.choice((-1.0, 1.0), size=(N_ENVS, n_pairs, 2)))
        yaws = 0.5 * np.pi * np.random.randint(4, size=(N_ENVS, n_pairs))
        yaws += (modes == 2) * 10.0 ** np.random.uniform(-7.0, -2.0, size=(N_ENVS, n_pairs))
        yaws = np.where(modes == 1, np.random.uniform(-np.pi, np.pi, size=(N_ENVS, n_pairs)), yaws)
        quats_tilt = gu.euler_to_quat(np.rad2deg(np.concatenate((tilts, yaws[..., None]), axis=-1)))
        quats_a = np.where(
            ((modes == 0) | (np.random.rand(N_ENVS, n_pairs) < 0.5))[..., None],
            quats_rand[..., 0, :],
            quats_face[..., 0, :],
        )
        quats_b = gu.transform_quat_by_quat(gu.transform_quat_by_quat(quats_tilt, quats_face[..., 1, :]), quats_a)
        quats_b = np.where((modes == 0)[..., None], quats_rand[..., 1, :], quats_b)

        for i_p, boxes in enumerate(pairs_boxes):
            for box, quat in zip(boxes, (quats_a[:, i_p], quats_b[:, i_p])):
                box.set_pos(pairs_pos[:, i_p])
                box.set_quat(quat)

        # Detect the contacts from scratch, then over a few steps along which the boxes separate. The contacts of a step
        # are detected at the poses that precede it.
        for i_step in range(N_STEPS + 1):
            boxes_pos = np.stack(
                [np.stack([tensor_to_array(box.get_pos()) for box in boxes], axis=1) for boxes in pairs_boxes], axis=1
            )
            boxes_R = gu.quat_to_R(
                np.stack(
                    [np.stack([tensor_to_array(box.get_quat()) for box in boxes], axis=1) for boxes in pairs_boxes],
                    axis=1,
                )
            )

            # Separation directions: the face normals of either box, then the cross products of their edges
            boxes_axes = boxes_R.swapaxes(-1, -2)
            edges_cross = np.cross(boxes_axes[..., 0, :, None, :], boxes_axes[..., 1, None, :, :])
            axes = np.concatenate(
                (boxes_axes[..., 0, :, :], boxes_axes[..., 1, :, :], edges_cross.reshape((*modes.shape, 9, 3))), axis=-2
            )
            axes_norm = np.linalg.norm(axes, axis=-1)
            is_axis_valid = axes_norm > gs.EPS
            axes /= np.where(is_axis_valid, axes_norm, 1.0)[..., None]
            # Support of the Minkowski difference of both boxes centred at the origin along each axis
            axes_support = (np.abs(axes[..., None, :, :] @ boxes_R) * boxes_half[..., None, :]).sum(axis=(-1, -3))

            if i_step == 0:
                # Offset the second box along a random direction from touching, by a gap, a graze, or a deep penetration
                dirs = np.random.normal(size=(N_ENVS, n_pairs, 3))
                dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)
                axes_dist = np.where(is_axis_valid, axes_support / np.abs(axes @ dirs[..., None])[..., 0], np.inf)
                dist_touch = axes_dist.min(axis=-1)

                half_min = boxes_half.min(axis=(-1, -2))
                dist_gap = 10.0 ** np.random.uniform(-6.0, -1.0, size=(N_ENVS, n_pairs)) * half_min
                dist_graze = -(10.0 ** np.random.uniform(-7.0, -3.0, size=(N_ENVS, n_pairs))) * half_min
                dist_deep = -np.random.uniform(0.0, 2.0, size=(N_ENVS, n_pairs)) * half_min
                offset_kinds = np.random.choice(3, size=(N_ENVS, n_pairs), p=(0.1, 0.5, 0.4))
                dist = np.choose(offset_kinds, (dist_gap, dist_graze, dist_deep))

                for i_p, (box_a, box_b) in enumerate(pairs_boxes):
                    box_b.set_pos(pairs_pos[:, i_p] + np.maximum(dist_touch + dist, 0.0)[:, i_p, None] * dirs[:, i_p])
                    # The boxes tumble at random while the second one closes in on the first, so that contacts evolve
                    for box, vel_lin in ((box_a, 0.0), (box_b, -dirs[:, i_p])):
                        vel_lin = SCALES[i_p] * (np.random.normal(size=(N_ENVS, 3)) + vel_lin)
                        box.set_dofs_velocity(np.concatenate((vel_lin, np.random.normal(size=(N_ENVS, 3))), axis=-1))
                boxes_pos[..., 1, :] = np.stack([tensor_to_array(box_b.get_pos()) for _, box_b in pairs_boxes], axis=1)

                if show_viewer:
                    scene.visualizer.update()
                scene.rigid_solver.collider.clear()
                scene.rigid_solver.collider.detection()
            else:
                scene.step()
            contacts = scene.rigid_solver.collider.get_contacts(to_torch=False)
            offsets = boxes_pos[..., 1, :] - boxes_pos[..., 0, :]
            axes_depth = np.where(is_axis_valid, axes_support - np.abs(axes @ offsets[..., None])[..., 0], np.inf)
            pairs_depth = axes_depth.min(axis=-1)

            # Geom A and geom B of every contact, with its normal pointing from B to A
            envs_idx, _ = np.nonzero(contacts["link_a"] >= 0)
            links_a, links_b = contacts["link_a"][contacts["link_a"] >= 0], contacts["link_b"][contacts["link_a"] >= 0]
            pairs_idx = links_pair_idx[links_a]
            assert_equal(links_pair_idx[links_b], pairs_idx)
            boxes_ab_idx = np.stack((links_box_idx[links_a], links_box_idx[links_b]), axis=-1)
            contacts_boxes_R = boxes_R[envs_idx[:, None], pairs_idx[:, None], boxes_ab_idx]
            contacts_boxes_half = boxes_half[envs_idx[:, None], pairs_idx[:, None], boxes_ab_idx]
            contacts_boxes_pos = boxes_pos[envs_idx[:, None], pairs_idx[:, None], boxes_ab_idx]
            contacts_tol = pairs_tol[envs_idx, pairs_idx]
            normals_ab = -contacts["normal"][contacts["link_a"] >= 0]
            depths = contacts["penetration"][contacts["link_a"] >= 0]
            positions = contacts["position"][contacts["link_a"] >= 0]

            # The boxes overlap along the normal of a contact by at least its depth. The first contact of a pair takes
            # their minimal overlap along any direction, the other ones being found on the pair tilted by the
            # perturbation, which leaves their normals unconstrained.
            offsets_ab = contacts_boxes_pos[:, 1] - contacts_boxes_pos[:, 0]
            normals_support = np.abs(normals_ab[:, None, None] @ contacts_boxes_R) * contacts_boxes_half[:, :, None]
            normals_overlap = normals_support.sum(axis=(-1, -2, -3)) - (normals_ab * offsets_ab).sum(axis=-1)
            assert (depths <= normals_overlap + contacts_tol).all()
            pairs_overlap_min = np.full((N_ENVS, n_pairs), np.inf)
            np.minimum.at(pairs_overlap_min, (envs_idx, pairs_idx), normals_overlap)
            is_pair_detected = np.isfinite(pairs_overlap_min)
            if detection == "mpr":
                # Minkowski Portal Refinement (MPR) converges on the face of the Minkowski difference crossed by its
                # ray, which may differ from the one of least overlap. Its normal is still the direction of a face or of
                # a pair of edges of both boxes, one of their separating axes, up to the convergence of its portal.
                axes_cos = np.abs((axes[envs_idx, pairs_idx] @ normals_ab[..., None])[..., 0])
                assert (np.arccos(np.minimum(axes_cos.max(axis=-1), 1.0)) < MPR_AXIS_ANGLE_TOL).all()
            else:
                assert (pairs_overlap_min - pairs_depth <= tol * pairs_size)[is_pair_detected].all()

            # The deepest contact of a pair takes the whole overlap along its normal
            pairs_depth_max = np.full((N_ENVS, n_pairs), -np.inf)
            np.maximum.at(pairs_depth_max, (envs_idx, pairs_idx), depths)
            is_deepest = depths >= pairs_depth_max[envs_idx, pairs_idx]
            assert (normals_overlap[is_deepest] - depths[is_deepest] <= contacts_tol[is_deepest]).all()

            # A contact lies midway between its witnesses, each on the surface of its own box. Rebuilt along the normal,
            # the witnesses of GJK and MPR, which lie apart along another direction, may land past the surfaces by up to
            # twice the tolerance, and those of a perturbed contact, whose depth is a lower bound, inside the boxes by
            # up to five times it.
            witnesses_tol_out, witnesses_tol_in = (1.0, 1.0) if detection == "box_box" else (2.0, 5.0)
            witnesses = (
                positions[:, None] + 0.5 * np.array((1.0, -1.0))[:, None] * depths[:, None, None] * normals_ab[:, None]
            )
            witnesses_local = ((witnesses - contacts_boxes_pos)[..., None, :] @ contacts_boxes_R)[..., 0, :]
            witnesses_q = np.abs(witnesses_local) - contacts_boxes_half
            witnesses_sdf = np.linalg.norm(np.maximum(witnesses_q, 0.0), axis=-1)
            witnesses_sdf += np.minimum(witnesses_q.max(axis=-1), 0.0)
            assert (witnesses_sdf <= witnesses_tol_out * contacts_tol[:, None]).all()
            assert (witnesses_sdf >= -witnesses_tol_in * contacts_tol[:, None]).all()

            # Overlapping boxes have contacts, boxes apart have none
            assert is_pair_detected[pairs_depth > pairs_tol].all()
            assert not is_pair_detected[pairs_depth < -pairs_tol].any()


@pytest.mark.required
def test_box_contact_true_penetration(show_viewer, tol):
    # The penetration of two aligned boxes is their least overlap along their common axes
    N_ENVS = 64
    N_ROUNDS = 8
    BOX_SIZE = np.array((4.0, 0.25, 0.25))

    scene = gs.Scene(
        rigid_options=gs.options.RigidOptions(
            use_gjk_collision=True,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.0, -8.0, 4.0),
            camera_lookat=(0.0, 0.0, 0.0),
        ),
        show_viewer=show_viewer,
    )
    box_1, box_2 = (
        scene.add_entity(
            morph=gs.morphs.Box(
                size=BOX_SIZE,
            ),
            vis_mode="collision",
        )
        for _ in range(2)
    )
    scene.build(n_envs=N_ENVS)

    for _ in range(N_ROUNDS):
        # Two identical boxes posed exactly aligned at random orientations, overlapping end to end
        quats = gu.random_quaternion(N_ENVS)
        overlaps = np.random.uniform(0.01, 0.5, size=(N_ENVS, 3)) * BOX_SIZE
        offsets_local = np.random.choice((-1.0, 1.0), size=(N_ENVS, 3)) * (BOX_SIZE - overlaps)
        box_1.set_quat(quats)
        box_2.set_pos(gu.transform_by_quat(offsets_local, quats))
        box_2.set_quat(quats)

        scene.rigid_solver.collider.clear()
        scene.rigid_solver.collider.detection()
        contacts = box_1.get_contacts(with_entity=box_2)
        penetrations = np.where(tensor_to_array(contacts["valid_mask"]), tensor_to_array(contacts["penetration"]), 0.0)
        assert_allclose(penetrations.max(axis=-1), overlaps.min(axis=-1), tol=tol)


@pytest.mark.slow  # ~150s
@pytest.mark.required
@pytest.mark.parametrize(
    "detection",
    [
        pytest.param("mpr", marks=pytest.mark.xfail(reason="Lets some stacks of boxes drift.")),
        pytest.param("gjk", marks=pytest.mark.xfail(reason="Lets some stacks of boxes drift.")),
        "box_box",
    ],
)
def test_box_stacks_stability(detection, show_viewer, tol):
    # Piles of boxes of random shapes, each lying on a random face at a random yaw with a tiny tilt, on fixed bases at
    # several scales, restacked in a new order and pose after every reset. Each pile is statically stable: at every
    # interface, the center of mass of the boxes above lies within the contact polygon, which holds as long as it lies
    # within the footprint of the box below, since the box above contains it too by convexity. Placing the boxes from the
    # top down, the next box only has to cover the center of mass of the boxes already placed.
    N_ENVS = 24
    GRAVITY = 9.81
    TILT = 0.0
    CONSTRAINT_TIMECONST = 0.002
    CONTACT_IMPEDANCE = 0.99
    PRUNING_TOLERANCE = 0.02
    SCALES = (0.1, 2.0)
    BASE_SIZE = np.array((1.5, 1.5, 0.2))
    BOXES_SIZE = np.array(((1.0, 0.6, 0.02), (0.8, 0.05, 0.05), (0.5, 0.4, 0.1), (0.3, 0.3, 0.3)))
    # Rotation laying a box on each of its faces, and the axis of the box that then points up
    BOX_EULER_ROTS = ((0, 0, 0), (180, 0, 0), (90, 0, 0), (-90, 0, 0), (0, -90, 0), (0, 90, 0))
    BOX_UP_AXES = np.array((2, 2, 1, 1, 0, 0))

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.001,
            gravity=(0.0, 0.0, -GRAVITY),
        ),
        rigid_options=gs.options.RigidOptions(
            use_gjk_collision=detection == "gjk",
            box_box_detection=detection == "box_box",
            contact_pruning_tolerance=PRUNING_TOLERANCE,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.0, 0.0, 0.0),
            camera_lookat=(0.0, 1.0, -0.1),
            camera_fov=60.0,
        ),
        show_viewer=show_viewer,
    )
    # Every box takes each shape in some environments, in an order shuffled per box. The boxes are built apart, as the
    # contacts of boxes overlapping at build time overflow the contact budget. The piles are laid out in the view as a
    # row, each one away from the camera in proportion to its scale, so that all of them look alike while lying at
    # depths far enough apart for their boxes, which may overhang the center of mass of the boxes above by half their
    # diagonal, to never reach each other. The camera looks at them from barely above, which keeps every pile close
    # to the horizontal plane through the origin, where the vertical resolution of the coordinates stays far finer
    # than the depth at which the boxes rest.
    piles = []
    for i_s, scale in enumerate(SCALES):
        view_x = 0.5 * (i_s - 0.5 * (len(SCALES) - 1))
        pile_pos = 5.0 * scale * np.array((view_x, 1.0, -0.1)) + (0.0, 0.0, scale * BASE_SIZE[2])
        base = scene.add_entity(
            gs.morphs.Box(
                pos=pile_pos - (0.0, 0.0, 0.5 * scale * BASE_SIZE[2]),
                size=scale * BASE_SIZE,
                fixed=True,
            ),
            vis_mode="collision",
        )
        boxes = [
            scene.add_entity(
                morph=[
                    gs.morphs.Box(pos=(*pile_pos[:2], pile_pos[2] + (i_b + 1) * scale), size=scale * box_size)
                    for box_size in np.random.permutation(BOXES_SIZE)
                ],
                visualize_contact=True,
                vis_mode="collision",
            )
            for i_b in range(len(BOXES_SIZE))
        ]
        piles.append((scale, pile_pos, base, boxes))
    scene.build(n_envs=N_ENVS)

    # Under the default impedance, the residual softness of the contacts tips the tallest piles standing on the
    # narrowest supports at the smallest scale, although they are statically stable. At rest, a contact sinks by about
    # the gravity acceleration times the squared time constant, scaled by (1 - d) / d for an impedance d and by the load
    # it carries: the mass of the boxes it supports times the sum of the inverse masses of the two links it separates,
    # the fixed base adding none. The time constant grows with the square root of the scale, which sinks every pile by a
    # depth in proportion to its size: the piles stay geometrically similar, and this depth stays above the resolution
    # of the coordinates of the largest ones, which stand the farthest from the origin. The boxes are built
    # axis-aligned, so their bounding boxes give the size each environment simulates.
    sol_params = gu.default_solver_params()
    piles_rest_depth, piles_boxes_size, piles_boxes_mass = [], [], []
    for scale, pile_pos, base, boxes in piles:
        sol_params[0] = CONSTRAINT_TIMECONST * np.sqrt(scale / SCALES[0])
        sol_params[2:4] = CONTACT_IMPEDANCE
        for entity in (base, *boxes):
            for geom in entity.geoms:
                geom.set_sol_params(sol_params)
        piles_rest_depth.append(GRAVITY * sol_params[0] ** 2 * (1.0 - CONTACT_IMPEDANCE) / CONTACT_IMPEDANCE)
        piles_boxes_mass.append(np.stack([tensor_to_array(box.get_mass()) for box in boxes]))
        aabbs = np.stack([tensor_to_array(box.get_AABB()) for box in boxes])
        piles_boxes_size.append(aabbs[..., 1, :] - aabbs[..., 0, :])

    for i_phase in range(3):
        if i_phase > 0:
            scene.reset()
        piles_boxes_pos_rest = []
        piles_boxes_up_axis = []
        for (scale, pile_pos, base, boxes), rest_depth, boxes_size, boxes_mass in zip(
            piles, piles_rest_depth, piles_boxes_size, piles_boxes_mass
        ):
            n_boxes = len(boxes)
            # Each box lies on a face no taller than the narrowest side of the face
            is_up_axis = np.arange(3) == BOX_UP_AXES[:, None]
            faces_height = np.where(is_up_axis, boxes_size[..., None, :], 0.0).sum(axis=-1)
            faces_width = np.where(is_up_axis, np.inf, boxes_size[..., None, :]).min(axis=-1)
            faces = np.argmax(np.where(faces_height <= faces_width, np.random.rand(*faces_height.shape), -1.0), axis=-1)
            is_box_up_axis = is_up_axis[faces]
            angles_rp = np.random.uniform(low=-1.0, high=1.0, size=(n_boxes, N_ENVS, 2)) * np.rad2deg(TILT)
            angles_yaw = np.random.uniform(low=-180.0, high=180.0, size=(n_boxes, N_ENVS, 1))
            quats_face = gu.euler_to_quat(np.take(BOX_EULER_ROTS, faces, axis=0))
            quats_yaw = gu.euler_to_quat(np.concatenate((np.zeros_like(angles_rp), angles_yaw), axis=-1))
            quats_tilt = gu.euler_to_quat(np.concatenate((angles_rp, angles_yaw), axis=-1))
            boxes_R_flat = gu.quat_to_R(gu.transform_quat_by_quat(quats_face, quats_yaw))
            boxes_quat = gu.transform_quat_by_quat(quats_face, quats_tilt)
            boxes_height = (is_box_up_axis * boxes_size).sum(axis=-1)

            # Stack the boxes in a random order, the height of each one being where it rests once the pile lies flat
            levels = np.argsort(np.random.rand(n_boxes, N_ENVS), axis=0)
            levels_height = np.take_along_axis(boxes_height, levels, axis=0)
            levels_z_rest = pile_pos[2] + np.cumsum(levels_height, axis=0) - 0.5 * levels_height
            levels_bottom = levels_z_rest - 0.5 * levels_height
            pile_top = levels_bottom[-1] + levels_height[-1]

            # The tilt of every box above moves their center of mass while they settle flat, by up to the tilt times
            # their height, which the margin to the edges of each footprint covers.
            levels_half = np.take_along_axis(0.5 * boxes_size, levels[..., None], axis=0)
            levels_margin = TILT * (pile_top - levels_bottom)[..., None]
            levels_is_up_axis = np.take_along_axis(is_box_up_axis, levels[..., None], axis=0)
            levels_footprint_half = np.where(levels_is_up_axis, 0.0, np.maximum(0.5 * levels_half - levels_margin, 0.0))
            levels_R_flat = np.take_along_axis(boxes_R_flat, levels[..., None, None], axis=0)
            levels_volume = np.take_along_axis(boxes_size.prod(axis=-1), levels, axis=0)
            levels_xy = np.empty((n_boxes, N_ENVS, 2))
            com_xy, com_moment, mass = np.zeros((N_ENVS, 2)), np.zeros((N_ENVS, 2)), np.zeros(N_ENVS)
            for i_l in reversed(range(n_boxes)):
                offset = np.random.uniform(low=-1.0, high=1.0, size=(N_ENVS, 3)) * levels_footprint_half[i_l]
                levels_xy[i_l] = com_xy + (levels_R_flat[i_l, :, :2] @ offset[..., None])[..., 0]
                com_moment += levels_volume[i_l, :, None] * levels_xy[i_l]
                mass += levels_volume[i_l]
                com_xy = com_moment / mass[:, None]
            base_half = np.maximum(0.5 * 0.5 * scale * BASE_SIZE[:2] - TILT * (pile_top - pile_pos[2])[:, None], 0.0)
            levels_xy += pile_pos[:2] + np.random.uniform(low=-1.0, high=1.0, size=(N_ENVS, 2)) * base_half - com_xy

            # Lower every box onto the one below until they touch, then by the depth at which its contact rests under
            # the load it carries. Raised along the vertical, a pair of boxes separates at the lowest height at which
            # one of their separating axes (the face normals of either box and the cross products of their edges)
            # separates them.
            levels_mass = np.take_along_axis(boxes_mass, levels, axis=0)
            levels_mass_above = np.cumsum(levels_mass[::-1], axis=0)[::-1]
            levels_mass_inv = 1.0 / levels_mass + np.concatenate((np.zeros((1, N_ENVS)), 1.0 / levels_mass[:-1]))
            levels_rest_depth = rest_depth * levels_mass_above * levels_mass_inv
            levels_R = np.take_along_axis(gu.quat_to_R(boxes_quat), levels[..., None, None], axis=0)
            levels_z = np.empty((n_boxes, N_ENVS))
            below_R = np.broadcast_to(np.eye(3), (N_ENVS, 3, 3))
            below_half = np.broadcast_to(0.5 * scale * BASE_SIZE, (N_ENVS, 3))
            below_pos = np.broadcast_to(pile_pos - (0.0, 0.0, 0.5 * scale * BASE_SIZE[2]), (N_ENVS, 3))
            for i_l in range(n_boxes):
                pair_R = np.stack((below_R, levels_R[i_l]), axis=1)
                pair_half = np.stack((below_half, levels_half[i_l]), axis=1)
                pair_axes = pair_R.swapaxes(-1, -2)
                edges_cross = np.cross(pair_axes[:, 0, :, None, :], pair_axes[:, 1, None, :, :])
                axes = np.concatenate((pair_axes[:, 0], pair_axes[:, 1], edges_cross.reshape((N_ENVS, 9, 3))), axis=1)
                axes /= np.maximum(np.linalg.norm(axes, axis=-1, keepdims=True), gs.EPS)
                axes *= np.where(axes[..., 2:] < 0.0, -1.0, 1.0)
                axes_support = (np.abs(axes[:, None] @ pair_R) * pair_half[:, :, None]).sum(axis=(-1, -3))
                offset = np.concatenate((levels_xy[i_l] - below_pos[:, :2], -below_pos[:, 2:]), axis=-1)
                axes_height = (axes_support - (axes * offset[:, None]).sum(axis=-1)) / axes[..., 2]
                levels_z[i_l] = np.where(axes[..., 2] > gs.EPS, axes_height, np.inf).min(axis=-1)
                levels_z[i_l] -= levels_rest_depth[i_l]
                below_R, below_half = levels_R[i_l], levels_half[i_l]
                below_pos = np.concatenate((levels_xy[i_l], levels_z[i_l, :, None]), axis=-1)

            boxes_pos = np.empty((n_boxes, N_ENVS, 3))
            np.put_along_axis(
                boxes_pos, levels[..., None], np.concatenate((levels_xy, levels_z[..., None]), -1), axis=0
            )
            boxes_pos_rest = np.empty((n_boxes, N_ENVS, 3))
            np.put_along_axis(
                boxes_pos_rest, levels[..., None], np.concatenate((levels_xy, levels_z_rest[..., None]), -1), axis=0
            )
            piles_boxes_pos_rest.append(boxes_pos_rest)
            piles_boxes_up_axis.append(BOX_UP_AXES[faces])
            for box, pos, quat in zip(boxes, boxes_pos, boxes_quat):
                box.set_pos(pos)
                box.set_quat(quat)

        # At every step, every pile stays where it was placed, its boxes lying flat on each other, and the contacts that
        # the step detects from the poses it starts from cover the patch between each pair of boxes in contact
        for i_step in range(100):
            piles_boxes_pos = [np.stack([tensor_to_array(box.get_pos()) for box in boxes]) for *_, boxes in piles]
            piles_boxes_quat = [np.stack([tensor_to_array(box.get_quat()) for box in boxes]) for *_, boxes in piles]
            scene.step()
            contacts = scene.rigid_solver.collider.get_contacts(to_torch=False)
            for i_pile, (scale, _, base, boxes) in enumerate(piles):
                rest_depth, boxes_size = piles_rest_depth[i_pile], piles_boxes_size[i_pile]
                boxes_pos_rest, boxes_up_axis = piles_boxes_pos_rest[i_pile], piles_boxes_up_axis[i_pile]
                boxes_pos, boxes_quat = piles_boxes_pos[i_pile], piles_boxes_quat[i_pile]
                boxes_R = gu.quat_to_R(boxes_quat)
                boxes_up_z = np.take_along_axis(boxes_R[..., 2, :], boxes_up_axis[..., None], axis=-1)[..., 0]
                boxes_tilt = np.arccos(np.minimum(np.abs(boxes_up_z), 1.0))
                boxes_drift = np.linalg.norm(boxes_pos - boxes_pos_rest, axis=-1) / scale
                assert (boxes_tilt < np.deg2rad(0.5)).all()
                assert (boxes_drift < 5e-3).all()

                # Each link rests on the one below through the faces nearest to the horizontal, whose corners are spanned
                # by its two other axes. The base is the first link of the pile.
                links_idx = np.array([base.base_link_idx] + [box.base_link_idx for box in boxes])
                links_pos = np.concatenate((tensor_to_array(base.get_pos())[None], boxes_pos))
                links_R = np.concatenate((np.broadcast_to(np.eye(3), (1, N_ENVS, 3, 3)), boxes_R))
                links_half = 0.5 * np.concatenate((np.broadcast_to(scale * BASE_SIZE, (1, N_ENVS, 3)), boxes_size))
                links_axes_order = (np.argmax(np.abs(links_R[..., 2, :]), axis=-1)[..., None] + np.arange(3)) % 3
                links_axes = np.take_along_axis(links_R, links_axes_order[..., None, :], axis=-1)
                links_axes_half = np.take_along_axis(links_half, links_axes_order, axis=-1)
                links_normal = links_axes[..., 0] * np.sign(links_axes[..., 2, :1])
                corners_sign = np.array(((-1.0, 1.0, 1.0, -1.0), (-1.0, -1.0, 1.0, 1.0)))
                links_corners = (links_axes[..., 1:] * links_axes_half[..., None, 1:]) @ corners_sign
                links_top = links_pos + links_normal * links_axes_half[..., :1]
                links_bottom = links_pos - links_normal * links_axes_half[..., :1]

                # The patch between two links is the overlap of their faces seen from above. Its depth at a point is the
                # vertical distance between the face planes there. A guard band covering the rounding of either depth
                # tells pressed and lifted corners from touching ones.
                depth_guard = 1e-2 * rest_depth + tol * scale
                for i_b in range(N_ENVS):
                    levels = np.concatenate(((0,), 1 + np.argsort(boxes_pos_rest[:, i_b, 2])))
                    is_contact = contacts["link_a"][i_b] >= 0
                    contacts_link_a = contacts["link_a"][i_b][is_contact]
                    contacts_link_b = contacts["link_b"][i_b][is_contact]
                    contacts_pos = contacts["position"][i_b][is_contact]
                    for i_lower, i_upper in zip(levels[:-1], levels[1:]):
                        # The overlap clips the top face of the lower link by the half-planes of the bottom face of the
                        # upper link, each keeping the points p where 'offset + p @ normal' is non-negative. Its pressed
                        # part clips it further by the half-plane where the depth exceeds the guard band.
                        lower_top, lower_normal = links_top[i_lower, i_b], links_normal[i_lower, i_b]
                        upper_bottom, upper_normal = links_bottom[i_upper, i_b], links_normal[i_upper, i_b]
                        lower_slope = -lower_normal[:2] / lower_normal[2]
                        upper_slope = -upper_normal[:2] / upper_normal[2]
                        lower_offset = lower_top[2] - lower_top[:2] @ lower_slope
                        upper_offset = upper_bottom[2] - upper_bottom[:2] @ upper_slope
                        depth_normal = lower_slope - upper_slope
                        depth_offset = lower_offset - upper_offset

                        clip = (links_bottom[i_upper, i_b, :2, None] + links_corners[i_upper, i_b, :2]).T
                        clip_edges = np.roll(clip, -1, axis=0) - clip
                        half_normals = np.stack((-clip_edges[:, 1], clip_edges[:, 0]), axis=-1)
                        half_normals *= np.sign(((clip.mean(axis=0) - clip) * half_normals).sum(axis=-1))[:, None]
                        half_offsets = -(half_normals * clip).sum(axis=-1)
                        half_planes = [*zip(half_normals, half_offsets), (depth_normal, depth_offset - depth_guard)]
                        patch = (links_top[i_lower, i_b, :2, None] + links_corners[i_lower, i_b, :2]).T
                        for i_half, (half_normal, half_offset) in enumerate(half_planes):
                            if i_half == len(half_planes) - 1:
                                patch_overlap = patch
                            patch_side = half_offset + patch @ half_normal
                            patch_clipped = []
                            for point, point_next, side, side_next in zip(
                                patch, np.roll(patch, -1, axis=0), patch_side, np.roll(patch_side, -1)
                            ):
                                if side >= 0.0:
                                    patch_clipped.append(point)
                                if side * side_next < 0.0:
                                    patch_clipped.append(point + (point_next - point) * side / (side - side_next))
                            patch = np.array(patch_clipped).reshape((-1, 2))
                        patch_pressed, patch = patch, patch_overlap
                        # Faces resting flat on each other have no part deeper than the guard band, all of it touching
                        if len(patch_pressed) < 3:
                            patch_pressed = patch
                        patch_depth = depth_offset + patch @ depth_normal

                        pair_links_idx = links_idx[[i_lower, i_upper]]
                        is_pair = np.isin(contacts_link_a, pair_links_idx) & np.isin(contacts_link_b, pair_links_idx)
                        pair_pos = contacts_pos[is_pair, :2]

                        # When at most four corners of the patch touch the other face, each of those pressed into it
                        # carries a contact, unless the pruning drops it: the triangle a corner forms with its two
                        # neighbors then covers less than the pruning tolerance of the patch area. Detection merges the
                        # contacts closer than the multi-contact tolerance, which scales with the smaller box of the
                        # pair, so a contact stands for every corner within this distance.
                        if (patch_depth > -depth_guard).sum() <= 4:
                            patch_prev, patch_next = np.roll(patch, 1, axis=0), np.roll(patch, -1, axis=0)
                            corners_edges = np.stack((patch - patch_prev, patch_next - patch), axis=-2)
                            corners_area = 0.5 * np.abs(np.linalg.det(corners_edges))
                            patch_area = 0.5 * np.abs(np.linalg.det(np.stack((patch, patch_next), axis=-2)).sum())
                            is_kept = corners_area > PRUNING_TOLERANCE * patch_area
                            is_pressed = patch_depth > depth_guard
                            corners_pair_dist = np.linalg.norm(patch[:, None] - pair_pos, axis=-1)
                            corners_dist = corners_pair_dist.min(axis=-1, initial=np.inf)
                            pair_scale = np.linalg.norm(links_half[[i_lower, i_upper], i_b], axis=-1).min()
                            corners_tol = scene.rigid_solver.collider._mc_tolerance * pair_scale
                            assert (corners_dist[is_pressed & is_kept] < corners_tol).all()

                        # Otherwise, detection reports only a few points per pair, so their hull only has to span the
                        # pressed part of the patch: its smallest width is compared with that of this part.
                        else:
                            assert len(pair_pos) >= 3
                            widths = []
                            for points in (pair_pos, patch_pressed):
                                points_diff = (points[:, None] - points[None]).reshape((-1, 2))
                                points_diff = points_diff[np.linalg.norm(points_diff, axis=-1) > gs.EPS]
                                directions = np.stack((-points_diff[:, 1], points_diff[:, 0]), axis=-1)
                                directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
                                projections = points @ directions.T
                                widths.append((projections.max(axis=0) - projections.min(axis=0)).min())
                            assert widths[0] > 0.25 * widths[1]


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

    asset_path = get_hf_dataset(pattern="meshes/*_hull.obj")

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
    # The lamp spans a meter while the finger presses it by less than a millimeter, so the rounding errors of the
    # Minkowski difference reach the depth convergence tolerance and cut dents into the polytope that refines the depth.
    # The finger is placed where the search would end on a dent.
    finger = scene.add_entity(
        gs.morphs.Mesh(
            file=f"{asset_path}/meshes/inspire_finger_hull.obj",
            pos=(0.2282334562357352, -0.24876064442406798, -3.140536243429785),
            quat=(0.5830528595970937, 0.23922303491173352, -0.7549261541988322, 0.18140618564379823),
            decimate=False,
        ),
        visualize_contact=True,
        vis_mode="collision",
    )
    lamp = scene.add_entity(
        gs.morphs.Mesh(
            file=f"{asset_path}/meshes/floor_lamp_hull.obj",
            pos=(-0.13053986430168152, -0.09062329679727554, -3.08468234539032),
            quat=(-0.04015674069523811, 0.15293818712234497, 0.14172174036502838, 0.9771961569786072),
        ),
        vis_mode="collision",
    )
    scene.build()

    face_offset = 0.5 * (BOX_SIZE + GEOM_SIZE)
    face_normal = torch.tensor([math.cos(YAW), math.sin(YAW), 0.0], dtype=gs.tc_float, device=gs.device)
    for geom, spot, depth in ((geom_grazing, GRAZING_SPOT, 0.0), (geom_pressed, PRESSED_SPOT, PRESSED_DEPTH)):
        pos_local = torch.tensor([face_offset - depth, *spot], dtype=gs.tc_float, device=gs.device)
        geom.set_pos(gu.transform_by_quat(pos_local, box.get_quat()) + box.get_pos())
        geom.set_quat(box.get_quat())

    # The exact depth of the hull pair is the distance from the origin to the boundary of their Minkowski difference
    finger_verts = tensor_to_array(finger.geoms[0].get_verts(), dtype=np.float64)
    lamp_verts = tensor_to_array(lamp.geoms[0].get_verts(), dtype=np.float64)
    hull_depth = -ConvexHull((finger_verts[:, None] - lamp_verts).reshape((-1, 3))).equations[:, 3].max()

    scene.step()
    contacts = scene.rigid_solver.collider.get_contacts()
    is_box = contacts["geom_a"] == box.geoms[0].idx
    assert_allclose((contacts["normal"][is_box] @ face_normal).abs(), 1.0, tol=tol)
    is_pressed = contacts["geom_b"] == geom_pressed.geoms[0].idx
    assert is_pressed.any()
    assert_allclose(contacts["penetration"][is_pressed], PRESSED_DEPTH, tol=tol)
    assert (contacts["penetration"][is_box & ~is_pressed] >= 0.0).all()
    assert (contacts["penetration"][is_box & ~is_pressed] <= GEOM_SIZE).all()
    is_lamp = contacts["geom_b"] == lamp.geoms[0].idx
    assert is_lamp.any()
    assert_allclose(contacts["penetration"][is_lamp].max(), hull_depth, atol=1e-7)
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

    # The pruning groups the contacts by link pair, which must stay apart however large and close their indices, so the
    # box comes after a sphere apart from everything and two walls come after the box
    scene.add_entity(
        morph=gs.morphs.Sphere(
            pos=(10.0, 10.0, 10.0),
            radius=0.01,
            fixed=True,
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
    scene.build(n_envs=2)
    # The box sits in the corner of the three walls in the first environment, and on the edge of the two walls after it
    # in the second one, where its only contacts come from the two link pairs of closest indices
    box.set_pos(((0.0, 0.0, 0.0), (0.55, 0.0, 0.0)))

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
                    hull = ConvexHull(proj, qhull_options="Qt")
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
    assert_allclose(box.get_pos(), ((0.0, 0.0, 0.0), (0.55, 0.0, 0.0)), atol=2e-3)


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
            assert (n_contacts >= 4).all()
        elif model_name == "collinear_spheres":
            assert (n_contacts == 2).all()

    assert_allclose(entity.get_pos()[..., :2], smooth_xy, atol=1e-3)


@pytest.mark.slow("gpu")  # gpu ~250s
@pytest.mark.parametrize(
    "scene_kind, max_collision_pairs, max_contacts, error_pattern, is_raised_by_build",
    [
        # Post-pruning contact budget overflow, with the candidate buffer large enough (2x margin) that it cannot
        # trip first. The automatic budget resolves to 32 contact points per link pair floored at 512, far below
        # what the bowls produce once they pile up. Its phase is left unpinned: the coincident bowls put the contact
        # count of the step taken by the build close to the budget, on either side depending on rounding.
        pytest.param(
            "bowls", 1_000, None, "max number of post-pruning contact points", None, marks=pytest.mark.required
        ),
        # Candidate contact buffer overflow. The explicit contact budget is clamped down to the buffer size, so only
        # the buffer itself can overflow. Its phase is left unpinned, for the same reason as above.
        ("bowls", 150, 1_000, "max number of candidate contact points", None),
        # Broad phase candidate pair overflow on the step taken by the build, the bowls starting fully overlapping.
        pytest.param(
            "bowls", 20, None, "max number of broad phase candidate contact pairs", True, marks=pytest.mark.required
        ),
        # Buffers large enough for the whole pile: no overflow at all. Both values keep a 2x margin over the peaks
        # reached within the stepped window (about 500 colliding geom pairs and 1040 post-pruning contact points).
        ("bowls", 1_000, 2_000, None, False),
        # Two contacts against a budget of one, from spheres resting on the plane at build: the clamp must also run
        # below the pruning gate (n_contacts < 3), in both the serial and the GPU cooperative kernel variants.
        ("spheres", 150, 1, "max number of post-pruning contact points", True),
    ],
)
@pytest.mark.parametrize("use_hibernation", [False, True])
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
def test_num_contact_overflow(
    scene_kind, max_collision_pairs, max_contacts, error_pattern, is_raised_by_build, use_hibernation, show_viewer
):
    from genesis.engine.simulator import RATE_CHECK_ERRNO

    N_BOWLS = 4
    scene = gs.Scene(
        rigid_options=gs.options.RigidOptions(
            max_collision_pairs=max_collision_pairs,
            max_contacts=max_contacts,
            use_hibernation=use_hibernation,
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

    with nullcontext() if error_pattern is None else pytest.raises(gs.GenesisException, match=error_pattern):
        scene.build()
        assert scene.rigid_solver.collider.collider_config.has_prunable_contacts

        # Contact budget as documented for 'RigidOptions.max_contacts' (32 contact points per link pair floored at 512),
        # each contact point taking 4 constraint rows under the default pyramidal friction cone.
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

        # errno is only polled every RATE_CHECK_ERRNO substeps, so one extra step is required to guarantee that an
        # error triggered by the first steps gets raised.
        for _ in range(RATE_CHECK_ERRNO + 1):
            scene.step()

    # An error raised by the build leaves the scene destroyed, one raised by a step leaves it built.
    if is_raised_by_build is not None:
        assert scene.is_built is not is_raised_by_build


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
            noslip_iterations=2,
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

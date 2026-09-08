import math
import os
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import torch
from PIL import Image

import genesis as gs
import genesis.utils.geom as gu
from genesis.ext import urdfpy
from genesis.utils import urdf as uu
from genesis.utils.misc import get_assets_dir, qd_to_numpy, tensor_to_array

from ..utils.assertions import assert_allclose, assert_equal
from ..utils.assets import get_hf_dataset


@pytest.mark.required
@pytest.mark.parametrize("model_name", ["depth_first_tree_mjcf", "depth_first_tree_urdf"])
def test_depth_first_link_ordering(xml_path, model_name, show_viewer):
    # Links must be parsed depth-first so every subtree - hence every free body's DOFs - occupies a contiguous index
    # range. The per-tree mass-matrix factorization relies on this so a multi-body file costs the same as the
    # equivalent separate entities.
    scene = gs.Scene(show_viewer=show_viewer)
    morph = gs.morphs.MJCF(file=xml_path) if model_name.endswith("mjcf") else gs.morphs.URDF(file=xml_path, fixed=True)
    entity = scene.add_entity(morph)
    scene.build(n_envs=0)

    parents = [link.parent_idx for link in entity.links]
    n_links = len(parents)
    children: dict[int, list[int]] = {i: [] for i in range(n_links)}
    for i, parent in enumerate(parents):
        if parent != -1:
            children[parent].append(i)
    for i in range(n_links):
        subtree = []
        stack = [i]
        while stack:
            link = stack.pop()
            subtree.append(link)
            stack.extend(children[link])
        assert sorted(subtree) == list(range(i, i + len(subtree))), f"subtree at link {i} is not contiguous"


@pytest.mark.slow  # ~250s
@pytest.mark.required
def test_mjcf_parsing_with_include():
    scene = gs.Scene()
    robot1 = scene.add_entity(
        morph=gs.morphs.MJCF(
            file="xml/franka_emika_panda/scene.xml",
        )
    )
    robot2 = scene.add_entity(
        morph=gs.morphs.MJCF(
            file="xml/franka_emika_panda/panda.xml",
        )
    )
    robot3 = scene.add_entity(
        morph=gs.morphs.MJCF(
            file="xml/franka_sim/franka_panda.xml",
        )
    )
    scene.build()
    assert_allclose(robot1.get_qpos(), robot2.get_qpos(), tol=gs.EPS)
    assert_allclose(robot1.get_qpos(), robot3.get_qpos(), tol=gs.EPS)


@pytest.mark.required
def test_mjcf_include_rewrites_asset_mesh_and_skips_default_mesh(mjcf_include_default_and_asset_mesh):
    scene_path, extents = mjcf_include_default_and_asset_mesh

    scene = gs.Scene()
    entity = scene.add_entity(
        gs.morphs.MJCF(
            file=scene_path,
        )
    )
    scene.build()

    (geom,) = entity.geoms
    assert geom.type == gs.GEOM_TYPE.MESH
    aabb = geom.get_AABB()
    assert_allclose(aabb[1] - aabb[0], extents, tol=5e-8)


@pytest.mark.required
def test_ground_plane_preservation(box_plan):
    mjcf = ET.tostring(box_plan, encoding="unicode")

    scene = gs.Scene()
    entity_with_ground = scene.add_entity(
        gs.morphs.MJCF(
            file=mjcf,
            exclude_ground_plane=False,
        )
    )
    entity_without_ground = scene.add_entity(
        gs.morphs.MJCF(
            file=mjcf,
            exclude_ground_plane=True,
        )
    )
    scene.build()

    assert_equal(sum(geom.type == gs.GEOM_TYPE.PLANE for geom in entity_with_ground.geoms), 1)
    assert_equal(sum(geom.type == gs.GEOM_TYPE.PLANE for geom in entity_without_ground.geoms), 0)
    assert_equal(sum(geom.type == gs.GEOM_TYPE.BOX for geom in entity_without_ground.geoms), 1)


@pytest.mark.slow  # ~200s
@pytest.mark.required
def test_urdf_parsing(show_viewer, tol):
    POS_OFFSET = 0.8
    WOLRD_QUAT = np.array([1.0, 1.0, -0.3, +0.3])
    DOOR_JOINT_DAMPING = 1.5

    scene = gs.Scene(
        show_viewer=show_viewer,
        show_FPS=False,
    )
    asset_path = get_hf_dataset(pattern="microwave/*")
    entities = {}
    for i, (fixed, merge_fixed_links) in enumerate(
        ((False, False), (False, True), (True, False), (True, True)),
    ):
        entity = scene.add_entity(
            morph=gs.morphs.URDF(
                file=f"{asset_path}/microwave/microwave.urdf",
                fixed=fixed,
                merge_fixed_links=merge_fixed_links,
                pos=(0.0, (i - 1.5) * POS_OFFSET, 0.0),
                quat=tuple(WOLRD_QUAT / np.linalg.norm(WOLRD_QUAT)),
            ),
            vis_mode="collision",
        )
        entities[(fixed, merge_fixed_links)] = entity
    scene.build()

    # four microwaves have four different root_idx
    root_idx_all = [link.root_idx for link in scene.rigid_solver.links]
    assert len(set(root_idx_all)) == 4

    def _check_entity_positions(expected_y_spacing, tol):
        # The four parsing configs are laid out 'expected_y_spacing' apart in y, so their world AABBs must coincide once
        # that spacing is removed. AABBs are world-frame, so this check is independent of the relative-getter frame.
        nonlocal entities
        AABB_all = []
        for key in ((False, False), (False, True), (True, False), (True, True)):
            AABB = np.array(
                [
                    [np.inf, np.inf, np.inf],
                    [-np.inf, -np.inf, -np.inf],
                ]
            )
            for geom in entities[key].geoms:
                AABB_i = tensor_to_array(geom.get_AABB())
                AABB[0] = np.minimum(AABB[0], AABB_i[0])
                AABB[1] = np.maximum(AABB[1], AABB_i[1])
            AABB_all.append(AABB)
        AABB_diff = np.diff(AABB_all, axis=0)
        AABB_diff[..., 1] -= expected_y_spacing
        assert_allclose(AABB_diff, 0.0, tol=tol)

    # Check that `set_pos` / `set_quat` applies the same transform in all cases. Both frames place every config at the
    # same pose, so the world AABBs coincide with no residual spacing.
    for relative in (False, True):
        for key in ((False, False), (False, True), (True, False), (True, True)):
            entities[key].set_pos(np.array([0.5, 0.0, 0.0]), relative=relative)
            entities[key].set_quat(np.array([0.0, 0.0, 0.0, 1.0]), relative=relative)
        if show_viewer:
            scene.visualizer.update()
        _check_entity_positions(0.0, tol=tol)

    # Check that `set_qpos` applies the same absolute transform in all cases. The fixed roots have no free joint to
    # take a base pose via qpos, so they are placed at the matching absolute world pose with the (relative=False)
    # setters. All four configs then sit POS_OFFSET apart in y, as at creation.
    door_angle = np.array([1.1])
    world_quat = tuple(WOLRD_QUAT / np.linalg.norm(WOLRD_QUAT))
    for i, key in enumerate(((False, False), (False, True))):
        qpos = np.concatenate(((0.0, (i - 1.5) * POS_OFFSET, 0.0), world_quat, door_angle))
        entities[key].set_qpos(qpos)
    for i, key in enumerate(((True, False), (True, True))):
        config_y = ((i + 2) - 1.5) * POS_OFFSET
        entities[key].set_pos(np.array([0.0, config_y, 0.0]), relative=False)
        entities[key].set_quat(np.array(world_quat), relative=False)
        entities[key].set_qpos(door_angle)
    if show_viewer:
        scene.visualizer.update()
    _check_entity_positions(POS_OFFSET, tol=tol)

    # Add dof damping to stabilitze the physics
    for key in ((False, False), (False, True), (True, False), (True, True)):
        entities[key].set_dofs_damping(entities[key].get_dofs_damping() + DOOR_JOINT_DAMPING)

    # Make sure that the dynamics of the door is the same in all cases
    door_vel = np.array([-0.2])
    entities[(False, False)].set_dofs_velocity(door_vel, 6)
    entities[(False, True)].set_dofs_velocity(door_vel, 6)
    entities[(True, False)].set_dofs_velocity(door_vel)
    entities[(True, True)].set_dofs_velocity(door_vel)
    link_1 = entities[(True, True)].link_start
    for key in ((False, False), (False, True)):
        link_2 = entities[key].link_start
        scene.rigid_solver.add_weld_constraint(link_1, link_2)

    for i in range(2000):
        scene.step()
        door_pos_all = (
            entities[(False, False)].get_dofs_position(6),
            entities[(False, True)].get_dofs_position(6),
            entities[(True, False)].get_dofs_position(0),
            entities[(True, True)].get_dofs_position(0),
        )
        door_pos_diff = torch.diff(torch.concatenate(door_pos_all))
        assert_allclose(door_pos_diff, 0, tol=5e-3)
    assert_allclose(scene.rigid_solver.dyn_state.dofs.vel.to_numpy(), 0.0, tol=1e-3)
    _check_entity_positions(POS_OFFSET, tol=2e-3)


@pytest.mark.slow  # ~200s
@pytest.mark.required
def test_parsing_inertia_defaults(
    undefined_inertia,
    undefined_inertia_arm,
    degenerate_inertials,
    zero_density_marker_mjcf,
    implicit_inertial_origin_chain,
    show_viewer,
    tol,
    caplog,
):
    GEOM_POS = (0.0, 0.0, 0.09)
    INERTIAL_POS = (0.0, 0.0, 0.11)
    TIP_MASS = 1e-5
    BOB_INERTIA_PER_MASS = 1e-3
    INERTIA = (
        (0.11, 0.01, 0.02),
        (0.01, 0.22, 0.03),
        (0.02, 0.03, 0.30),
    )
    # Principal moments per unit mass of the two geometries a recovered inertia is derived from.
    SPHERE_INERTIA_PER_MASS = 2.0 * 0.06**2 / 5.0
    BOX_INERTIA_PER_MASS = 2.0 * 0.2**2 / 12.0
    GRAVITY = (0.0, 0.0, -9.81)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
            gravity=GRAVITY,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.5, -3.5, 1.5),
            camera_lookat=(0.0, 0.0, 0.2),
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(gs.morphs.Plane())

    # merge_fixed_links=False keeps every fixed link as a link of its own, so the test can check its inertial.
    entity = scene.add_entity(
        morph=gs.morphs.URDF(
            file=degenerate_inertials,
            pos=(0.0, 0.0, 0.1),
            merge_fixed_links=False,
        ),
    )
    # The same asset welded to the world, so that the links the world carries resolve without a geometry estimate.
    entity_welded = scene.add_entity(
        morph=gs.morphs.URDF(
            file=degenerate_inertials,
            pos=(0.0, -4.0, 0.1),
            fixed=True,
            merge_fixed_links=False,
        ),
    )
    # The same asset welded and then attached to a moving link, so that the links the world carried resolve at attach.
    carrier_welded = scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.2, 0.2, 0.2),
            pos=(0.0, -8.0, 1.0),
        ),
    )
    entity_mounted = scene.add_entity(
        morph=gs.morphs.URDF(
            file=degenerate_inertials,
            pos=(0.0, -8.0, 1.3),
            fixed=True,
            batch_fixed_verts=True,
            merge_fixed_links=False,
        ),
    )
    entity_mounted.attach(carrier_welded, parent_link_name=carrier_welded.base_link.name, pos=(0.0, 0.0, 0.2))
    entity_zero_density = scene.add_entity(
        morph=gs.morphs.MJCF(
            file=zero_density_marker_mjcf,
        ),
    )
    entity_chain_unmerged = scene.add_entity(
        morph=gs.morphs.URDF(
            file=implicit_inertial_origin_chain,
            pos=(0.0, 2.0, 0.5),
            merge_fixed_links=False,
        ),
    )
    # The two merged chains are built from one in-memory model, which parsing must leave intact for the second one.
    chain_root = ET.fromstring(implicit_inertial_origin_chain)
    chain_model = urdfpy.URDF._from_xml(chain_root, chain_root, get_assets_dir())
    entity_chain_merged = scene.add_entity(
        morph=gs.morphs.URDF(
            file=chain_model,
            pos=(0.8, 2.0, 0.5),
        ),
    )
    entity_chain_unmerged_unaligned = scene.add_entity(
        morph=gs.morphs.URDF(
            file=implicit_inertial_origin_chain,
            pos=(0.8, 1.0, 0.5),
            align=False,
            merge_fixed_links=False,
        ),
    )
    entity_chain_merged_unaligned = scene.add_entity(
        morph=gs.morphs.URDF(
            file=chain_model,
            pos=(1.6, 1.0, 0.5),
            align=False,
        ),
    )
    # The asset states no inertial for its base link. Attachment makes that base movable, so Genesis computes its
    # mass from geometry at robot density, the same value the free-standing copy gets.
    carrier = scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.2, 0.2, 0.2),
            pos=(2.4, 0.0, 1.0),
        ),
    )
    mounted_arm = scene.add_entity(
        morph=gs.morphs.URDF(
            file=undefined_inertia_arm,
            pos=(2.4, 0.0, 1.3),
            fixed=True,
            batch_fixed_verts=True,
        ),
    )
    mounted_arm.attach(carrier, parent_link_name=carrier.base_link.name, pos=(0.0, 0.0, 0.2))
    # A link with visual geometry only gets its mass from that geometry.
    mounted_drawn = scene.add_entity(
        morph=gs.morphs.URDF(
            file=undefined_inertia,
            pos=(4.0, 0.0, 1.3),
            collision=False,
            fixed=True,
            batch_fixed_verts=True,
            align=False,
        ),
    )
    mounted_drawn.attach(carrier, parent_link_name=carrier.base_link.name, pos=(0.8, 0.0, 0.2))
    free_arm = scene.add_entity(
        morph=gs.morphs.URDF(
            file=undefined_inertia_arm,
            pos=(3.2, 0.0, 1.0),
        ),
    )
    free_drawn = scene.add_entity(
        morph=gs.morphs.URDF(
            file=undefined_inertia,
            pos=(4.8, 0.0, 1.0),
            collision=False,
            align=False,
        ),
    )
    # An entity attached beneath another shares its root, so attaching that one onto a moving link makes both bases
    # movable at once. Each is estimated at the density of its own entity. The three stand consecutively, since an
    # entity created between attached ones is rejected.
    stacked_base = scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.2, 0.2, 0.2),
            pos=(6.0, 0.0, 1.0),
        ),
    )
    stacked_middle = scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(6.0, 0.0, 1.3),
            fixed=True,
            batch_fixed_verts=True,
        ),
        material=gs.materials.Rigid(
            rho=2000.0,
        ),
    )
    stacked_tip = scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(6.0, 0.0, 1.5),
            fixed=True,
            batch_fixed_verts=True,
        ),
        material=gs.materials.Rigid(
            rho=8000.0,
        ),
    )
    stacked_tip.attach(stacked_middle, parent_link_name=stacked_middle.base_link.name, pos=(0.0, 0.0, 0.2))
    stacked_middle.attach(stacked_base, parent_link_name=stacked_base.base_link.name, pos=(0.0, 0.0, 0.2))

    with caplog.at_level("WARNING"):
        scene.build()

    # A 0.1 m cube of each entity's own density, rather than both taking the density of the one they hang from.
    assert_allclose(stacked_middle.base_link.get_mass(), 2000.0 * 0.1**3, tol=gs.EPS)
    assert_allclose(stacked_tip.base_link.get_mass(), 8000.0 * 0.1**3, tol=gs.EPS)

    # The link 'no_inertial' has no inertial element, so its whole inertial is the geometry estimate. Every link carries
    # the same sphere, so this mass is the reference for the recovered masses below.
    estimate = entity.get_link("no_inertial").desc
    assert_allclose(estimate.inertial_pos, GEOM_POS, tol=tol)
    assert_allclose(np.linalg.eigvalsh(estimate.inertia) / estimate.mass, SPHERE_INERTIA_PER_MASS, tol=tol)

    # The tensors are stored in their principal frame, whose parsed quaternion is markedly less accurate than the
    # tensor itself, so compare the rotation-invariant principal moments rather than the tensor rotated back.
    for link, expected_mass, expected_com, expected_inertia_per_mass in (
        # The root states a zero mass and a valid inertia, and the geometry estimate replaces both. Its fixed child
        # 'massless_marker' keeps its zero mass, so the root has no mass-bearing child.
        (entity.base_link, estimate.mass, GEOM_POS, SPHERE_INERTIA_PER_MASS),
        # A zero mass beside an authored center of mass keeps that center of mass, as the geometry supplies no inertia.
        (entity.get_link("massless_marker"), gs.EPS, INERTIAL_POS, 0.0),
        # An omitted inertial origin places the center of mass at the link frame.
        (entity.get_link("implicit_origin"), 2.5, (0.0, 0.0, 0.0), np.linalg.eigvalsh(INERTIA) / 2.5),
        # Geometry supplies the inertia scaled to the authored mass, and the link keeps its authored center of mass.
        (entity.get_link("zero_inertia"), 2.5, INERTIAL_POS, SPHERE_INERTIA_PER_MASS),
        # The fixed link 'tip' contributes to its parent composite, so geometry recovers its zero inertia at the
        # authored mass.
        (entity.get_link("tip"), TIP_MASS, GEOM_POS, SPHERE_INERTIA_PER_MASS),
        # The fixed child 'bob' carries the mass, so 'connector' keeps a zero inertia and a zero mass floored at gs.EPS.
        (entity.get_link("connector"), gs.EPS, GEOM_POS, 0.0),
        (entity.get_link("bob"), 1.0, GEOM_POS, BOB_INERTIA_PER_MASS),
        # The geom of 'marker' has a zero density and 'hull' carries the mass, so the marker keeps a zero mass.
        (entity_zero_density.get_link("hull"), 8.0, (0.0, 0.0, 0.0), BOX_INERTIA_PER_MASS),
        (entity_zero_density.get_link("marker"), gs.EPS, (0.0, 0.0, 0.0), 0.0),
    ):
        assert_allclose(link.desc.mass, expected_mass, rtol=1e-6, err_msg=link.name)
        assert_allclose(link.desc.inertial_pos, expected_com, tol=tol, err_msg=link.name)
        assert_allclose(
            np.linalg.eigvalsh(link.desc.inertia) / link.desc.mass,
            expected_inertia_per_mass,
            tol=tol,
            err_msg=link.name,
        )
    # The links the joints still move resolve exactly as in the free copy. The two the world carries keep the asset's
    # values, degenerate as they are: a zero mass and no inertia. Attached, every link resolves as in the free copy.
    for link, link_welded, link_mounted in zip(entity.links, entity_welded.links, entity_mounted.links):
        assert_allclose(link_mounted.desc.mass, link.desc.mass, tol=gs.EPS, err_msg=link.name)
        assert_allclose(link_mounted.desc.inertial_pos, link.desc.inertial_pos, tol=gs.EPS, err_msg=link.name)
        assert_allclose(link_mounted.desc.inertia, link.desc.inertia, tol=gs.EPS, err_msg=link.name)
        if link_welded.is_fixed:
            assert_equal(link_welded.desc.mass, 0.0, err_msg=link.name)
            assert_equal(link_welded.desc.inertia, 0.0, err_msg=link.name)
            continue
        assert_equal(link_welded.desc.mass, link.desc.mass, err_msg=link.name)
        assert_equal(link_welded.desc.inertial_pos, link.desc.inertial_pos, err_msg=link.name)
        assert_equal(link_welded.desc.inertia, link.desc.inertia, err_msg=link.name)

    # Every asset above is parsed by MuJoCo, a zero or missing inertial included.
    assert not any("legacy URDF parser" in record.getMessage() for record in caplog.records)

    # Resolving the center of mass to the link frame can place it outside the geometry, which stays worth reporting.
    # Only the link whose geometry is offset qualifies, once per copy of the robot.
    dubious_com_records = [record for record in caplog.records if "dubious center of mass" in record.getMessage()]
    assert len(dubious_com_records) == 3

    # Every link of an aligned free body keeps its own mass, so each link reads its authored mass. The two totals are
    # accumulated by independent code paths, so their agreement is bounded by that cross-path floor.
    assert_allclose(entity_chain_unmerged.get_links_mass(), (2.5, 1.5), tol=gs.EPS)
    assert_allclose(entity_chain_unmerged.get_mass(), entity_chain_merged.get_mass(), rtol=5e-7)

    # Merged and unmerged copies are the same rigid body, so one step from rest under the same wrench gives both the
    # same motion. This holds with the frame anchored on the center of mass and with the frame at the link origin. The
    # wrench acts on the last link, the fixed child when kept, at one world point above the base. Armature, damping and
    # a velocity servo sit on every free-joint DOF.
    ARMATURE = (0.5, 0.4, 0.3, 0.01, 0.02, 0.015)
    DAMPING = (2.0, 1.0, 3.0, 0.05, 0.1, 0.02)
    KV = (4.0, 3.0, 5.0, 0.2, 0.3, 0.1)
    VEL_TARGET = (0.5, -0.2, 0.3, 1.0, -2.0, 0.5)
    FORCE, TORQUE = (3.0, 0.0, 0.0), (0.4, -0.2, 0.1)
    for entity_chain in (
        entity_chain_unmerged,
        entity_chain_merged,
        entity_chain_unmerged_unaligned,
        entity_chain_merged_unaligned,
    ):
        entity_chain.set_dofs_armature(ARMATURE)
        entity_chain.set_dofs_damping(DAMPING)
        entity_chain.set_dofs_kv(KV)
        entity_chain.control_dofs_velocity(VEL_TARGET)
        entity_chain.apply_links_external_wrench(
            force=FORCE,
            torque=TORQUE,
            links_idx_local=entity_chain.n_links - 1,
            pos=(*entity_chain.morph.pos[:2], 0.8),
        )
    # From rest, one implicit step solves the augmented mass matrix against the generalized force. Both are written in
    # the coordinates of the free joint: origin velocity along the world axes, angular velocity along the link axes. The
    # single unaligned link is the case with every coupling in it.
    desc = entity_chain_merged_unaligned.base_link.desc
    link_R = gu.quat_to_R(tensor_to_array(entity_chain_merged_unaligned.get_quat()))
    com = link_R @ desc.inertial_pos
    inertia = link_R @ gu.quat_to_R(desc.inertial_quat) @ desc.inertia @ gu.quat_to_R(desc.inertial_quat).T @ link_R.T
    com_skew = np.array([[0.0, -com[2], com[1]], [com[2], 0.0, -com[0]], [-com[1], com[0], 0.0]])
    mass_mat = np.block(
        [
            [desc.mass * np.eye(3), -desc.mass * com_skew @ link_R],
            [desc.mass * link_R.T @ com_skew, link_R.T @ (inertia - desc.mass * com_skew @ com_skew) @ link_R],
        ]
    )
    mass_mat += np.diag(ARMATURE) + scene.dt * np.diag(np.add(DAMPING, KV))
    origin = tensor_to_array(entity_chain_merged_unaligned.get_pos())
    arm = np.array((*entity_chain_merged_unaligned.morph.pos[:2], 0.8)) - origin
    force = np.concatenate([FORCE + desc.mass * np.array(GRAVITY), link_R.T @ (TORQUE + np.cross(arm, FORCE))])
    force += np.multiply(KV, VEL_TARGET)
    scene.step()
    # The midpoint rule takes the velocity products at the midpoint velocity, the implicit update at the initial one.
    # From rest the two differ by h * |dv|^2 / 4, the tolerance here.
    assert_allclose(
        entity_chain_merged_unaligned.get_dofs_velocity(), scene.dt * np.linalg.solve(mass_mat, force), tol=1e-5
    )
    # The two copies compose the same body through independent code paths, so their agreement is bounded by that
    # cross-path floor, or by the working precision when coarser.
    for entity_chain, entity_chain_ref in (
        (entity_chain_unmerged, entity_chain_merged),
        (entity_chain_unmerged_unaligned, entity_chain_merged_unaligned),
    ):
        assert_allclose(entity_chain.get_dofs_velocity(), entity_chain_ref.get_dofs_velocity(), tol=max(tol, 5e-7))
        assert_allclose(entity_chain.get_quat(), entity_chain_ref.get_quat(), tol=max(tol, 5e-7))
        assert_allclose(
            entity_chain.get_pos() - torch.tensor(entity_chain.morph.pos),
            entity_chain_ref.get_pos() - torch.tensor(entity_chain_ref.morph.pos),
            tol=max(tol, 5e-7),
        )

    # Attachment resolves the base like any movable link, so its mass matches the same asset added free. Both
    # collision and visual geometry feed the computation.
    assert_allclose(mounted_arm.base_link.get_mass(), free_arm.base_link.get_mass(), tol=gs.EPS)
    assert_allclose(mounted_drawn.get_mass(), free_drawn.get_mass(), tol=gs.EPS)
    # Genesis applies the robot density because the asset is articulated. The convex hull of the authored sphere
    # loses a little volume, so the measured mass lands slightly below the closed form.
    assert_allclose(mounted_arm.base_link.get_mass(), 1500.0 * 4.0 / 3.0 * np.pi * 0.06**3, rtol=0.1)

    # The rest on the spheres after the fall checks that the recovered inertias give stable dynamics. The position
    # tolerance covers the steady-state penetration of the soft contact.
    for _ in range(40):
        scene.step()
    assert_allclose(entity.get_dofs_velocity(), 0.0, atol=1e-4)
    assert_allclose(entity.get_pos(), (0.0, 0.0, -0.03), atol=5e-4)


@pytest.mark.slow  # ~200s
@pytest.mark.required
@pytest.mark.parametrize("urdf_path", ["chain.urdf", "dual_arms_glb/dual_arms_glb.urdf", "dual_arms_primitives.urdf"])
@pytest.mark.parametrize("fixed", [False, True])
def test_urdf_parsing_merge_fixed_links(urdf_path, fixed, show_viewer, tol):
    POS = (0.0, -0.2, 0.5)
    EULER = (0.0, 90.0, 45.0)

    scene = gs.Scene(show_viewer=show_viewer)
    urdf_rootdir = os.path.dirname(urdf_path)
    asset_path = get_hf_dataset(pattern=os.path.join(urdf_rootdir, "*") if urdf_rootdir else urdf_path)
    robot_1 = scene.add_entity(
        gs.morphs.URDF(
            file=os.path.join(asset_path, urdf_path),
            pos=POS,
            euler=EULER,
            fixed=fixed,
            recompute_inertia=True,
            merge_fixed_links=False,
        ),
        surface=gs.surfaces.Default(
            color=(1, 0, 0, 0.5),
        ),
    )
    robot_2 = scene.add_entity(
        gs.morphs.URDF(
            file=os.path.join(asset_path, urdf_path),
            pos=POS,
            euler=EULER,
            fixed=fixed,
            recompute_inertia=True,
            merge_fixed_links=True,
        ),
        surface=gs.surfaces.Default(
            color=(0, 1, 0, 0.5),
        ),
    )
    scene.build()

    assert_allclose(robot_1.get_pos(), POS, tol=tol)
    assert_allclose(robot_1.get_quat(), gu.euler_to_quat(EULER), tol=tol)

    for _ in range(2):
        assert_allclose(robot_1.get_pos(), robot_2.get_pos(), tol=tol)
        assert_allclose(robot_1.get_quat(), robot_2.get_quat(), tol=tol)
        for link_2 in robot_2.links:
            link_1 = robot_1.get_link(link_2.name)
            assert_allclose(link_1.get_pos(), link_2.get_pos(), tol=tol)
            quat_1, quat_2 = link_1.get_quat(), link_2.get_quat()
            if quat_1[0] * quat_2[0] < 0.0:
                quat_2[:] *= -1.0
            assert_allclose(quat_1, quat_2, tol=tol)

        pos0 = np.random.rand(3)
        quat0 = np.random.rand(4)
        for robot in (robot_1, robot_2):
            robot.set_pos(pos0)
            robot.set_quat(quat0)

    com_robot_1, com_robot_2 = scene.rigid_solver.get_links_root_COM(
        links_idx=(robot_1.base_link_idx, robot_2.base_link_idx)
    )
    assert_allclose(com_robot_1, com_robot_2, tol=5e-7)


@pytest.mark.required
@pytest.mark.parametrize("model_name", ["box_freejoint_offset"])
def test_mjcf_parsing_merge_fixed_links(xml_path, show_viewer):
    # get_pos must reflect set_qpos for MJCF robots with a freejoint and a non-zero initial body position.
    POS = (1.0, 2.0, 3.0)
    QUAT = (0.0, 1.0, 0.0, 0.0)

    scene = gs.Scene(show_viewer=show_viewer)
    robot = scene.add_entity(
        gs.morphs.MJCF(
            file=xml_path,
        )
    )
    scene.build()

    assert_allclose(robot.get_pos(), (0.0, 0.0, 1.0), tol=gs.EPS)
    assert_allclose(robot.get_quat(), np.array([1.0, 0.0, 0.0, 1.0]) / math.sqrt(2), tol=gs.EPS)

    robot.set_qpos((*POS, *QUAT), qs_idx_local=slice(None, 7))
    assert_allclose(robot.get_pos(), POS, tol=gs.EPS)
    assert_allclose(robot.get_quat(), QUAT, tol=gs.EPS)

    scene.reset()
    assert_allclose(robot.get_pos(), (0.0, 0.0, 1.0), tol=gs.EPS)
    assert_allclose(robot.get_quat(), np.array([1.0, 0.0, 0.0, 1.0]) / math.sqrt(2), tol=gs.EPS)

    robot.set_pos(POS)
    robot.set_quat(QUAT)
    assert_allclose(robot.get_pos(), POS, tol=gs.EPS)
    assert_allclose(robot.get_quat(), QUAT, tol=gs.EPS)


@pytest.mark.slow  # ~200s
@pytest.mark.required
def test_urdf_capsule(tmp_path, show_viewer, tol):
    urdf_path = tmp_path / "capsule.urdf"
    with open(urdf_path, "w") as f:
        f.write(
            """
            <robot name="urdf_robot">
                <link name="base_link">
                    <inertial>
                        <origin rpy="0 0 0" xyz="0 0 0"/>
                        <mass value=".1"/>
                        <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>
                    </inertial>
                    <collision>
                        <origin rpy="0 0 0" xyz="0 0 0"/>
                        <geometry>
                            <capsule length="0.1" radius="0.02"/>
                        </geometry>
                    </collision>
                    <visual>
                        <origin rpy="0 0 0" xyz="0 0 0"/>
                        <geometry>
                            <capsule length="0.06" radius="0.03"/>
                        </geometry>
                    </visual>
                </link>
            </robot>
            """
        )

    scene = gs.Scene(show_viewer=show_viewer)
    scene.add_entity(gs.morphs.Plane())
    robot = scene.add_entity(
        gs.morphs.URDF(
            file=urdf_path,
            pos=(0.0, 0.0, 0.3),
        ),
        vis_mode="collision",
    )
    scene.build()

    (geom,) = robot.geoms
    assert geom.type == gs.GEOM_TYPE.CAPSULE
    assert_allclose(geom.data[:2], (0.02, 0.1), tol=gs.EPS)

    for _ in range(40):
        scene.step()
    geom_verts = tensor_to_array(geom.get_verts())
    assert np.linalg.norm(geom_verts - (0.0, 0.0, 0.0), axis=-1, ord=np.inf).min() < 1e-3
    assert np.linalg.norm(geom_verts - (0.0, 0.0, 0.14), axis=-1, ord=np.inf).min() < 1e-3

    (vgeom,) = robot.vgeoms
    vgeom_verts = tensor_to_array(vgeom.get_vverts())
    # Visual is a capsule (length=0.06, radius=0.03, total height 0.12) centered on the link, so after
    # the collision capsule settles against the plane (link at z=0.07), the visual spans z in [0.01, 0.13].
    assert np.linalg.norm(vgeom_verts - (0.0, 0.0, 0.01), axis=-1, ord=np.inf).min() < 1e-3
    assert np.linalg.norm(vgeom_verts - (0.0, 0.0, 0.13), axis=-1, ord=np.inf).min() < 1e-3


@pytest.mark.required
@pytest.mark.parametrize("model_name", ["pendulum_with_joint_dynamics", "joint_with_partial_dynamics"])
@pytest.mark.parametrize("joint_damping, joint_friction", [(1.0, 2.0), (1.0, None), (None, 2.0)])
def test_urdf_joint_dynamics(joint_damping, joint_friction, xml_path):
    # A <dynamics> tag authoring only one of damping/friction leaves the other one unset. It must fall back to zero
    # like an absent tag rather than reaching the solver as a null value.
    scene = gs.Scene()
    robot = scene.add_entity(
        gs.morphs.URDF(
            file=xml_path,
            pos=(0, 0, 0.8),
            convexify=True,
        ),
    )
    scene.build()
    expected_damping = 0.0 if joint_damping is None else joint_damping
    expected_frictionloss = 0.0 if joint_friction is None else joint_friction
    assert_allclose(robot.joints[0].desc.dofs_damping, 0.0, tol=gs.EPS)
    assert_allclose(robot.joints[1].desc.dofs_damping, expected_damping, tol=gs.EPS)
    assert_allclose(robot.joints[0].desc.dofs_frictionloss, 0.0, tol=gs.EPS)
    assert_allclose(robot.joints[1].desc.dofs_frictionloss, expected_frictionloss, tol=gs.EPS)


@pytest.mark.slow  # ~200s
@pytest.mark.required
@pytest.mark.parametrize("model_name", ["freeflyer_mjcf", "freeflyer_urdf"])
def test_default_armature_freeflyer(xml_path):
    DEFAULT_ARMATURE = 1000.0

    morph_class = gs.morphs.URDF if xml_path.endswith(".urdf") else gs.morphs.MJCF
    morph = morph_class(
        file=xml_path,
        default_armature=DEFAULT_ARMATURE,
    )
    morph_without_armature = morph_class(
        file=xml_path,
        pos=(1.0, 0.0, 0.0),
    )

    scene = gs.Scene()
    robot = scene.add_entity(morph)
    robot_without_armature = scene.add_entity(morph_without_armature)
    scene.build()

    armature = robot.get_dofs_armature()
    assert_allclose(armature[:6], 0.0, tol=gs.EPS)
    assert_allclose(armature[6], DEFAULT_ARMATURE, tol=gs.EPS)
    if xml_path.endswith(".xml"):
        assert_allclose(armature[7], 42.0, tol=gs.EPS)
        assert_allclose(armature[8], 0.0002, tol=gs.EPS)

    # Rotor inertia adds to the effective inertia of the joint it is applied to, so it must lower the inverse weight
    # the solver derives from it. An inverse weight parsed before the armature was applied leaves the rotor inertia
    # out of every constraint it scales.
    assert robot.get_dofs_invweight()[6] < robot_without_armature.get_dofs_invweight()[6]


@pytest.mark.slow  # ~200s
@pytest.mark.required
def test_xacro_loading(xacro_robot, show_viewer, tol):
    scene = gs.Scene(show_viewer=show_viewer)

    # Load with default args (mass=1.0, length=0.4)
    morph = gs.morphs.URDF(
        file=xacro_robot,
        fixed=True,
        merge_fixed_links=False,
    )

    entity = scene.add_entity(morph)

    # Load again with overridden mass via xacro_args
    heavy = scene.add_entity(
        gs.morphs.URDF(
            file=xacro_robot,
            fixed=True,
            merge_fixed_links=False,
            xacro_args={"link_mass": "5.0"},
        ),
    )
    scene.build()

    # Entity name from <robot name="xacro_chain">
    assert entity.name.startswith("xacro_chain_")

    # Three links (base_link + child_link + mesh_link), one revolute DOF
    assert entity.n_links == 3
    assert [l.name for l in entity.links] == ["base_link", "child_link", "mesh_link"]
    assert entity.n_dofs == 1
    assert entity.links[1].joints[0].type == gs.JOINT_TYPE.REVOLUTE

    # Geom types: cylinder on first two links, mesh on third
    assert entity.links[0].geoms[0].type == gs.GEOM_TYPE.CYLINDER
    assert entity.links[1].geoms[0].type == gs.GEOM_TYPE.CYLINDER
    assert entity.links[2].geoms[0].type == gs.GEOM_TYPE.MESH

    # Mass check: of the three links at 1.0 each (default) or 5.0 each (overridden), the base is fixed to the world
    # and is carried with no mass of its own, so the body weighs what the dynamics moves.
    assert_allclose(entity.get_mass(), 2.0, tol=tol)
    assert_allclose(heavy.get_mass(), 10.0, tol=tol)


@pytest.mark.required
@pytest.mark.required
@pytest.mark.parametrize("overwrite", [False, True])
def test_color_overwrite(overwrite, urdf_with_external_assets, show_viewer):
    scene = gs.Scene(show_viewer=show_viewer)
    box = scene.add_entity(
        gs.morphs.URDF(
            file="urdf/blue_box/model.urdf",
            convexify=False,
        ),
        surface=gs.surfaces.Default(
            color=(1.0, 0.0, 0.0, 1.0) if overwrite else None,
        ),
    )
    asset_path = get_hf_dataset(pattern="chain.urdf")
    chain = scene.add_entity(
        gs.morphs.URDF(
            file=f"{asset_path}/chain.urdf",
        ),
        surface=gs.surfaces.Default(
            color=(1.0, 0, 0, 1.0) if overwrite else None,
        ),
    )
    axis = scene.add_entity(
        gs.morphs.Mesh(
            file="meshes/axis.obj",
            convexify=False,
        ),
        surface=gs.surfaces.Default(
            color=(1.0, 0.0, 0.0, 1.0) if overwrite else None,
        ),
    )
    asset_path = get_hf_dataset(pattern="work_table.glb")
    table = scene.add_entity(
        gs.morphs.Mesh(
            file=f"{asset_path}/work_table.glb",
            convexify=False,
        ),
        surface=gs.surfaces.Default(
            color=(1.0, 0.0, 0.0, 1.0) if overwrite else None,
        ),
    )
    asset_path = get_hf_dataset(pattern="humanoid.xml")
    humanoid = scene.add_entity(
        gs.morphs.MJCF(
            file=f"{asset_path}/humanoid.xml",
        ),
        surface=gs.surfaces.Default(
            color=(1.0, 0.0, 0.0, 1.0) if overwrite else None,
        ),
    )
    textured = scene.add_entity(
        gs.morphs.URDF(
            file=urdf_with_external_assets,
        ),
        surface=gs.surfaces.Default(
            color=(1.0, 0.0, 0.0, 1.0) if overwrite else None,
        ),
    )
    if show_viewer:
        scene.build()

    for vgeom in box.vgeoms:
        assert vgeom.vmesh.metadata["is_visual_overwritten"] == overwrite
        visual = vgeom.vmesh.trimesh.visual
        assert visual.defined
        color = np.unique(visual.vertex_colors, axis=0)
        assert_equal(color, (255, 0, 0, 255) if overwrite else (0, 0, 255, 255))

    for vgeom in chain.vgeoms:
        assert vgeom.vmesh.metadata["is_visual_overwritten"] == overwrite
        visual = vgeom.vmesh.trimesh.visual
        assert visual.defined
        color = np.unique(visual.vertex_colors, axis=0)
        assert_equal(color, (255, 0, 0, 255) if overwrite else (51, 51, 51, 255))

    for vgeom in humanoid.vgeoms:
        # FIXME: The original material is lost because the visuals are collision geometries that has been duplicated as
        # visual to circumvent the lack of dedicated visuals.
        is_true_visual = vgeom.vmesh.metadata["name"] == "nose"
        assert vgeom.vmesh.metadata["is_visual_overwritten"] == overwrite or not is_true_visual
        visual = vgeom.vmesh.trimesh.visual
        assert visual.defined
        color = np.unique(visual.vertex_colors, axis=0)
        if is_true_visual:
            if overwrite:
                assert_equal(color, (255, 0, 0, 255))
            else:
                with pytest.raises(AssertionError):
                    assert_equal(color, (128, 128, 128, 255))
        else:
            assert_equal(color, (255, 0, 0, 255) if overwrite else (128, 128, 128, 255))

    for vgeom in axis.vgeoms:
        assert vgeom.vmesh.metadata["is_visual_overwritten"] == overwrite
        visual = vgeom.vmesh.trimesh.visual
        assert visual.defined
        color = np.unique(visual.vertex_colors, axis=0)
        if overwrite:
            assert_equal(color, (255, 0, 0, 255))
        else:
            assert_equal(color, [[0, 0, 178, 255], [0, 178, 0, 255], [178, 0, 0, 255], [255, 255, 255, 255]])

    for vgeom in table.vgeoms:
        assert vgeom.vmesh.metadata["is_visual_overwritten"] == overwrite
        visual = vgeom.vmesh.trimesh.visual
        assert visual.defined
        if overwrite:
            color = np.unique(visual.vertex_colors, axis=0)
            assert_equal(color, (255, 0, 0, 255))

    texture_path = ET.fromstring(urdf_with_external_assets).find("material/texture").get("filename")
    texture = np.array(Image.open(texture_path).convert("RGB"))
    sphere_vgeom, quad_vgeom = textured.vgeoms
    for vgeom in (sphere_vgeom, quad_vgeom):
        assert vgeom.vmesh.metadata["is_visual_overwritten"] == overwrite
        assert vgeom.vmesh.trimesh.visual.defined
    if overwrite:
        for vgeom in (sphere_vgeom, quad_vgeom):
            color = np.unique(vgeom.vmesh.trimesh.visual.vertex_colors, axis=0)
            assert_equal(color, (255, 0, 0, 255))
    else:
        assert_equal(np.array(quad_vgeom.vmesh.trimesh.visual.material.image.convert("RGB")), texture)
        # A visual without texture coordinates takes the mean color of the texture
        color = np.unique(sphere_vgeom.vmesh.trimesh.visual.vertex_colors, axis=0)
        assert_equal(color, (*texture.mean(axis=(0, 1)), 255))

    for entity in scene.entities:
        for geom in entity.geoms:
            assert geom.mesh.metadata["is_visual_overwritten"]
            visual = geom.mesh.trimesh.visual
            assert visual.defined
            color = np.unique(visual.vertex_colors, axis=0)
            # Collision geometry meshes have randomized colors with partial transparency to ease debugging
            with pytest.raises(AssertionError):
                assert_equal(color, (255, 0, 0, 255))


@pytest.mark.required
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
@pytest.mark.parametrize(
    "xml_path",
    [
        pytest.param("xml/franka_emika_panda/panda.xml", marks=pytest.mark.slow),
        "urdf/go2/urdf/go2.urdf",
        "urdf/panda_bullet/panda.urdf",
    ],
)
def test_robot_scale_and_dofs_armature(xml_path, monkeypatch, tol):
    ROBOT_SCALES = (1.0, 0.2, 5.0)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            gravity=(0, 0, -10.0),
        ),
        rigid_options=gs.options.RigidOptions(
            enable_collision=False,
        ),
        show_viewer=False,
        show_FPS=False,
    )
    files = [xml_path]
    if xml_path.endswith(".urdf"):
        # The asset is also loaded as one in-memory model shared by an entity at every scale, which parsing must leave
        # intact for the next one. An in-memory model resolves its relative mesh paths against the working directory.
        urdf_path = os.path.join(get_assets_dir(), xml_path)
        monkeypatch.chdir(os.path.dirname(urdf_path))
        files.append(urdfpy.URDF.load(urdf_path))
    robots_scale = []
    for scale in ROBOT_SCALES:
        for file in files:
            morph_kwargs = dict(file=file, scale=scale)
            if xml_path.endswith(".xml"):
                morph = gs.morphs.MJCF(**morph_kwargs)
            else:
                morph = gs.morphs.URDF(**morph_kwargs)
            scene.add_entity(morph)
            robots_scale.append(scale)
    scene.build()

    # Disable armature because it messes up with the mass matrix.
    # It is also a good opportunity to check that it updates 'invweight' and meaninertia accordingly.
    attr_orig = {}
    for scale, robot in zip(robots_scale, scene.entities):
        links_invweight = robot.get_links_invweight()
        dofs_invweight = robot.get_dofs_invweight()
        robot.set_dofs_armature(torch.ones((robot.n_dofs,), dtype=gs.tc_float, device=gs.device))
        assert torch.all(robot.get_dofs_invweight() < 1.0)
        with pytest.raises(AssertionError):
            assert_allclose(robot.get_dofs_invweight(), dofs_invweight, tol=tol)
        with pytest.raises(AssertionError):
            assert_allclose(robot.get_links_invweight(), links_invweight, tol=tol)
        robot.set_dofs_armature(torch.zeros((robot.n_dofs,), dtype=gs.tc_float, device=gs.device))
        links_invweight = robot.get_links_invweight()
        dofs_invweight = robot.get_dofs_invweight()
        qpos = np.random.rand(robot.n_dofs)
        robot.set_dofs_position(qpos)
        robot.set_dofs_armature(torch.zeros((robot.n_dofs,), dtype=gs.tc_float, device=gs.device))
        assert_allclose(robot.get_dofs_invweight(), dofs_invweight, tol=gs.EPS)
        assert_allclose(robot.get_links_invweight(), links_invweight, tol=gs.EPS)
        scene.reset()
        assert_allclose(robot.get_dofs_invweight(), dofs_invweight, tol=gs.EPS)
        assert_allclose(robot.get_links_invweight(), links_invweight, tol=gs.EPS)

        mass = robot.get_mass() / scale**3
        attr_orig.setdefault("mass", mass)
        assert_allclose(mass, attr_orig["mass"], tol=tol)

        inertia = np.stack([link.desc.inertia for link in robot.links], axis=0) / scale**5
        attr_orig.setdefault("inertia", inertia)
        assert_allclose(inertia, attr_orig["inertia"], tol=tol)

        joint_pos = np.stack([joint.pos for joint in robot.joints], axis=0) / scale
        attr_orig.setdefault("joint_pos", joint_pos)
        assert_allclose(joint_pos, attr_orig["joint_pos"], tol=tol)

        links_pos = robot.get_links_pos() / scale
        attr_orig.setdefault("links_pos", links_pos)
        assert_allclose(links_pos, attr_orig["links_pos"], tol=tol)

        # Check that links and dofs invweight are approximately valid.
        # Note that assessing whether the value is truly correct would be quite tricky.
        # FIXME: The tolerance must be very high when using 32bits precision. This means that our computation of the
        # inverse mass matrix has poor numerical robustness due to ill conditioning of the mass matrix. This is
        # concerning as it would impact the numerical stability of constraint solving, and by extension of the entire
        # rigid body dynamics.
        tol_ = tol if gs.backend == gs.cpu else 2e-3
        attr_orig.setdefault("links_invweight", links_invweight)
        attr_orig.setdefault("dofs_invweight", dofs_invweight)
        if scale > 1.0:
            scale_ratio_min, scale_ratio_max = scale**3, scale**5
        else:
            scale_ratio_min, scale_ratio_max = scale**5, scale**3
        assert torch.all(scale_ratio_min * links_invweight - tol_ < attr_orig["links_invweight"])
        assert torch.all(attr_orig["links_invweight"] < scale_ratio_max * links_invweight + tol_)
        dofs_invweight = robot.get_dofs_invweight()
        assert torch.all(scale_ratio_min * dofs_invweight - tol_ < attr_orig["dofs_invweight"])
        assert torch.all(attr_orig["dofs_invweight"] < scale_ratio_max * dofs_invweight + tol_)

    # Make sure that we are scaling bounds properly for linear joints
    # TODO: None of the robots being tested for now have linear joints...
    # TODO: Scaling of bounds depending on the type of joint should be explicitly checked.
    for robot in scene.entities:
        dofs_lower_bound, dofs_upper_bound = robot.get_dofs_limit()
        robot.set_dofs_position(dofs_lower_bound)
    scene.step()
    qf_passive = scene.rigid_solver.dyn_state.dofs.qf_passive.to_numpy()
    assert_allclose(qf_passive, 0.0, tol=tol)


@pytest.mark.slow  # ~200s
@pytest.mark.required
def test_mesh_primitive_COM(show_viewer):
    scene = gs.Scene(
        profiling_options=gs.options.ProfilingOptions(
            show_FPS=False,
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(
        gs.morphs.Plane(),
    )
    bunny = scene.add_entity(
        gs.morphs.Mesh(
            file="meshes/bunny.obj",
            pos=(-1.0, -1.0, 0.55),
        ),
        vis_mode="collision",
        visualize_contact=True,
    )
    cube = scene.add_entity(
        gs.morphs.Box(
            size=(0.5, 0.5, 0.5),
            pos=(1.0, 1.0, 0.55),
        ),
        vis_mode="collision",
        visualize_contact=True,
    )
    twin = scene.add_entity(
        gs.morphs.Mesh(
            file="meshes/bunny.obj",
            pos=(-1.0, 1.0, 0.55),
        ),
    )

    scene.build()

    # Entities built from one asset share a single copy of the collision geometry and its derived data. A convex
    # decomposition holds hundreds of kilobytes of it, so a thousand bunnies cost the memory of one.
    geometry, twin_geometry = bunny.geoms[0].mesh, twin.geoms[0].mesh
    assert twin_geometry.verts is geometry.verts
    assert twin_geometry.faces is geometry.faces
    assert twin_geometry.get_unique_edges() is geometry.get_unique_edges()
    assert twin_geometry.get_vert_adjacency() is geometry.get_vert_adjacency()
    assert twin_geometry.inertial is geometry.inertial

    rigid = scene.sim.rigid_solver
    for _ in range(50):
        scene.step()
    scene.rigid_solver.update_vgeoms()

    _, bunny_COM, cube_COM, _ = rigid.get_links_pos(ref=gs.link_ref_frame.link_COM)
    _, root_bunny_COM, root_cube_COM, _ = rigid.get_links_pos(ref=gs.link_ref_frame.root_COM)
    assert_allclose(bunny_COM, bunny.get_links_pos(links_idx_local=[0], ref=gs.link_ref_frame.link_COM), atol=gs.EPS)
    assert_allclose(cube_COM, cube.get_links_pos(links_idx_local=[0], ref=gs.link_ref_frame.link_COM), atol=gs.EPS)
    assert_allclose(root_bunny_COM, bunny_COM, atol=gs.EPS)
    assert_allclose(root_cube_COM, cube_COM, atol=gs.EPS)

    bunny_vgeom = bunny.vgeoms[0]
    bunny_vgeom_COM = tensor_to_array(bunny_vgeom.get_pos()) + bunny_vgeom.vmesh.trimesh.center_mass
    assert_allclose(scene.rigid_solver.get_dofs_velocity(), 0.0, atol=0.05)
    assert_allclose(bunny_COM, bunny_vgeom_COM, tol=5e-3)
    assert_allclose(cube_COM[2], 0.25, atol=2e-3)


@pytest.mark.slow  # ~200s
@pytest.mark.required
def test_align_mesh(show_viewer, tol):
    INIT_POS = (0.0, 0.0, 0.1)

    mango_path = get_hf_dataset(pattern="glb/mango.glb")
    bowl_path = get_hf_dataset(pattern="glb/orange_plastic_bowl.glb")

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.8, 0.8, 1.6),
            camera_lookat=(0.0, 0.0, 0.0),
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(gs.morphs.Plane())
    mango_morph = gs.morphs.Mesh(
        file=f"{mango_path}/glb/mango.glb",
        scale=0.045,
        pos=INIT_POS,
        align=True,
    )
    mango = scene.add_entity(
        mango_morph,
        material=gs.materials.Rigid(
            rho=1000.0,
        ),
        vis_mode="collision",
        visualize_contact=True,
    )
    ghost_mango = scene.add_entity(
        mango_morph,
        material=gs.materials.Kinematic(),
    )
    # Heterogeneous entity sharing one link across a bowl and a mango variant. Each variant must be aligned to its own
    # geometry (link origin at that variant's COM) and report its own offset per environment. The bowl carries a
    # hard-coded offset to check it composes with the alignment, and the mango variant must end up aligned exactly
    # like the standalone mango above.
    HET_POS = (0.5, 0.0, 0.1)
    het_obj = scene.add_entity(
        morph=(
            gs.morphs.Mesh(
                file=f"{bowl_path}/glb/orange_plastic_bowl.glb",
                scale=0.5,
                pos=HET_POS,
                offset_euler=(30.0, 0.0, 0.0),
                align=True,
            ),
            gs.morphs.Mesh(
                file=f"{mango_path}/glb/mango.glb",
                scale=0.045,
                pos=HET_POS,
                align=True,
            ),
        ),
        material=gs.materials.Rigid(
            rho=1000.0,
        ),
    )
    scene.build(n_envs=2)

    # Geoms/vgeoms are alignment-transparent, so their world pose equals the morph pose, and without a morph offset
    # the relative (user-frame) pose matches it too.
    geom, vgeom = mango.geoms[0], mango.vgeoms[0]
    for relative in (False, True):
        assert_allclose(geom.get_pos(relative=relative), INIT_POS, atol=1e-3)
        assert_allclose(geom.get_quat(relative=relative), gu.identity_quat(), atol=1e-3)
        assert_allclose(vgeom.get_pos(relative=relative), INIT_POS, atol=1e-3)
        assert_allclose(vgeom.get_quat(relative=relative), gu.identity_quat(), atol=1e-3)

    # The relative (user-frame) base pose strips the alignment back to the morph pose.
    assert_allclose(mango.get_pos(relative=True), INIT_POS, tol=tol)
    assert_allclose(mango.get_quat(relative=True), gu.identity_quat(), tol=tol)
    # The world-frame base pose places the link frame at the geometry COM and principal axes.
    assert_allclose(
        mango.get_links_pos(links_idx_local=[0], ref=gs.link_ref_frame.link_COM, relative=False),
        mango.get_links_pos(links_idx_local=[0], ref=gs.link_ref_frame.link_origin, relative=False),
        tol=tol,
    )
    geom_inertia_i = qd_to_numpy(scene.rigid_solver.dyn_state.links.cinr_inertial, transpose=True)[0, 1]
    geom_quat = tensor_to_array(mango.get_quat(relative=False))
    assert_allclose(gu.R_to_xyz(gu.quat_to_R(geom_quat) @ uu.principal_axes_rot(geom_inertia_i).T), 0.0, tol=tol)

    # Both variants (env 0 bowl, env 1 mango) strip their own offset back to the user pose, and each variant's link
    # origin sits at its own COM. The bowl recovers the user frame despite its hard-coded offset, proving the offset
    # composes with the alignment.
    for i_env in (0, 1):
        assert_allclose(het_obj.get_pos(relative=True, envs_idx=i_env), HET_POS, tol=tol)
        assert_allclose(het_obj.get_quat(relative=True, envs_idx=i_env), gu.identity_quat(), tol=tol)
        assert_allclose(
            het_obj.get_links_pos(links_idx_local=[0], ref=gs.link_ref_frame.link_COM, envs_idx=i_env, relative=False),
            het_obj.get_links_pos(
                links_idx_local=[0], ref=gs.link_ref_frame.link_origin, envs_idx=i_env, relative=False
            ),
            tol=tol,
        )

    # The two variants have different geometry, so their aligned world origins differ.
    with np.testing.assert_raises(AssertionError):
        assert_allclose(
            het_obj.get_pos(relative=False, envs_idx=0), het_obj.get_pos(relative=False, envs_idx=1), tol=tol
        )

    # The heterogeneous mango variant (env 1) is aligned exactly like the standalone mango: the world<-user offset
    # (COM shift and principal-axis rotation) matches, independent of the base placement.
    het_mango_offset = het_obj.get_pos(relative=False, envs_idx=1) - het_obj.get_pos(relative=True, envs_idx=1)
    mango_offset = mango.get_pos(relative=False, envs_idx=0) - mango.get_pos(relative=True, envs_idx=0)
    assert_allclose(het_mango_offset, mango_offset, tol=tol)
    assert_allclose(het_obj.get_quat(relative=False, envs_idx=1), mango.get_quat(relative=False, envs_idx=0), tol=tol)

    # Same qpos on rigid and kinematic entities must yield matching vAABB
    qpos = (0.3, -0.2, 1.0, 0.6, 0.5, 0.3, 0.0)
    mango.set_qpos(qpos)
    ghost_mango.set_qpos(qpos)
    assert_allclose(mango.get_vAABB(), ghost_mango.get_vAABB(), tol=gs.EPS)
    scene.reset()

    # Simulate
    for _ in range(650):
        scene.step()

    assert_allclose(mango.get_dofs_velocity(dofs_idx_local=(0, 1, 2)), 0, tol=0.01)
    assert_allclose(mango.get_dofs_velocity(dofs_idx_local=(3, 4, 5)), 0, tol=0.08)
    min_z = mango.get_AABB()[:, 0, 2]
    assert ((-1e-3 < min_z) & (min_z < 0.0)).all()


@pytest.mark.slow  # ~200s
@pytest.mark.required
def test_align_urdf(show_viewer, tol):
    INIT_POS = (0.0, 0.0, 0.7)

    asset_path = get_hf_dataset(pattern="fork/*")

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.8, 0.8, 0.6),
            camera_lookat=(-0.3, 0.0, 0.0),
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(gs.morphs.Plane())
    fork_morph = gs.morphs.URDF(
        file=f"{asset_path}/fork/fork.urdf",
        pos=INIT_POS,
        euler=(0.0, 90.0, 0.0),
    )
    fork = scene.add_entity(
        fork_morph,
        vis_mode="collision",
        visualize_contact=True,
    )
    ghost_fork = scene.add_entity(
        fork_morph,
        material=gs.materials.Kinematic(),
    )
    scene.build()

    # The relative (user-frame) base pose strips the alignment back to the morph pose, while the world-frame base
    # pose has its link frame origin at the collision geometry COM (auto-align for basic rigid objects).
    assert_allclose(fork.get_pos(relative=True), INIT_POS, tol=tol)
    assert_allclose(fork.get_links_pos(ref=gs.link_ref_frame.link_COM), fork.get_pos(relative=False), tol=tol)

    # Same qpos on rigid and kinematic entities must yield matching vAABB
    qpos = (0.3, -0.2, 1.0, 0.6, 0.5, 0.3, 0.0)
    fork.set_qpos(qpos)
    ghost_fork.set_qpos(qpos)
    assert_allclose(fork.get_vAABB(), ghost_fork.get_vAABB(), tol=gs.EPS)
    scene.reset()

    # Simulate with initial angular velocity to check numerical stability
    fork.set_dofs_velocity(10.0, dofs_idx_local=slice(3, 6))
    for _ in range(200):
        scene.step()

    assert_allclose(fork.get_dofs_velocity(), 0, tol=0.05)
    assert (-0.002 < fork.get_AABB()[0, 2] < 0.0).all()


@pytest.mark.required
def test_align_mixed_mass_raises():
    # Mixing a user-specified mass with a geometry-estimated one in an aligned free body makes the anchor density-
    # dependent (so rigid and kinematic could align differently) and must raise. The fixed joint with
    # merge_fixed_links=False keeps the child a distinct fixed link with unspecified mass while the base specifies one.
    urdf = _build_two_link_revolute_urdf(
        "mixed_mass_align",
        "box",
        {"size": "0.06 0.06 0.06"},
        links_inertial=[{"mass": 1.0, "ixx": 0.01, "iyy": 0.01, "izz": 0.01, "origin_xyz": "0 0 0"}, None],
        joint_type="fixed",
    )
    for material in (gs.materials.Rigid(), gs.materials.Kinematic()):
        scene = gs.Scene(
            show_viewer=False,
            show_FPS=False,
        )
        with pytest.raises(gs.GenesisException, match="geometry-estimated link masses"):
            scene.add_entity(gs.morphs.URDF(file=urdf, align=True, merge_fixed_links=False), material=material)


@pytest.mark.required
def test_align_preserves_link_local_frames(aligned_link_frame_urdfs, show_viewer, tol):
    TORQUE = (0.3, 0.8, -0.5)
    OFFSET_EULER = (20.0, 35.0, 50.0)
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.0, -4.0, 2.0),
            camera_lookat=(1.0, 0.0, 0.5),
        ),
        show_viewer=show_viewer,
    )
    entity_aligned = scene.add_entity(
        morph=gs.morphs.URDF(
            pos=(0.0, 0.0, 0.5),
            offset_euler=OFFSET_EULER,
            file=aligned_link_frame_urdfs[1],
            align=True,
            merge_fixed_links=False,
        ),
        vis_mode="collision",
    )
    entity_unaligned = scene.add_entity(
        morph=gs.morphs.URDF(
            pos=(1.0, 0.0, 0.5),
            offset_euler=OFFSET_EULER,
            file=aligned_link_frame_urdfs[1],
            align=False,
            merge_fixed_links=False,
        ),
        vis_mode="collision",
    )
    entity_heterogeneous_aligned = scene.add_entity(
        morph=tuple(
            gs.morphs.URDF(
                pos=(2.0, 0.0, 0.5),
                offset_euler=OFFSET_EULER,
                file=file,
                align=True,
                merge_fixed_links=False,
            )
            for file in aligned_link_frame_urdfs
        ),
        vis_mode="collision",
    )
    # The unaligned twin of each aligned entity pins the authored behavior that alignment must preserve.
    entity_heterogeneous_unaligned = scene.add_entity(
        morph=tuple(
            gs.morphs.URDF(
                pos=(3.0, 0.0, 0.5),
                offset_euler=OFFSET_EULER,
                file=file,
                align=False,
                merge_fixed_links=False,
            )
            for file in aligned_link_frame_urdfs
        ),
        vis_mode="collision",
    )
    # A heterogeneous entity needs one environment per morph variant.
    scene.build(n_envs=2)

    # The fixture geoms sit at non-identity poses relative to their links and the morph offset is a rotation that
    # does not commute with them. Otherwise the geom-frame conjugation would degenerate to the plain morph offset
    # and a corrupted re-expression would still pass. A convex-decomposed mesh is useless here: its sub-geoms keep
    # an identity frame (geometry lives in the vertices). The user orientation is identity, so the world<-user offset
    # rotates each base-link geom about the link origin:
    # geom_world_pos = U_pos + R(offset) * (geom_user_pos - U_pos) and geom_world_quat = offset * geom_user_quat.
    # Child-link geoms carry no user-frame offset (morph offsets belong to floating base links), so their world
    # pose is pinned from the authored chain of the unaligned twin's description instead.
    offset_quat = gu.xyz_to_quat(np.array(OFFSET_EULER), rpy=True, degrees=True)
    # 'transform_quat_by_quat' requires operands of matching shapes.
    batched_offset_quat = np.tile(offset_quat, (scene.n_envs, 1))
    base_desc, child_desc = (link.desc for link in entity_unaligned.links)
    child_geom_desc = entity_unaligned.geoms[1].desc
    child_geom_user_pos = child_desc.pos + gu.transform_by_quat(child_geom_desc.pos, child_desc.quat)
    child_geom_quat = gu.transform_quat_by_quat(child_geom_desc.quat, child_desc.quat)
    child_world_quat = gu.transform_quat_by_quat(child_geom_quat, offset_quat)
    for entity in (entity_aligned, entity_unaligned):
        assert_allclose(gu.quat_to_xyz(entity.get_quat()), 0.0, tol=tol)
        u_pos = tensor_to_array(entity.get_pos())
        base_geom, child_geom = entity.geoms
        base_user_pos = tensor_to_array(base_geom.get_pos(relative=True))
        base_user_quat = tensor_to_array(base_geom.get_quat(relative=True))
        base_world_pos = u_pos + gu.transform_by_quat(base_user_pos - u_pos, offset_quat)
        base_world_quat = gu.transform_quat_by_quat(base_user_quat, batched_offset_quat)
        assert_allclose(base_geom.get_pos(relative=False), base_world_pos, tol=tol)
        assert_allclose(base_geom.get_quat(relative=False), base_world_quat, tol=tol)
        child_world_pos = u_pos + gu.transform_by_quat(child_geom_user_pos, offset_quat)
        assert_allclose(child_geom.get_pos(relative=False), child_world_pos, tol=tol)
        assert_allclose(child_geom.get_quat(relative=False), child_world_quat, tol=tol)
    # Alignment must not move any collision or visual geom in the world: the aligned twin matches the unaligned one.
    for aligned_geom, unaligned_geom in zip(
        (*entity_aligned.geoms, *entity_aligned.vgeoms),
        (*entity_unaligned.geoms, *entity_unaligned.vgeoms),
        strict=True,
    ):
        assert_allclose(
            aligned_geom.get_pos(relative=False) - entity_aligned.get_pos(),
            unaligned_geom.get_pos(relative=False) - entity_unaligned.get_pos(),
            tol=tol,
        )
        assert_allclose(aligned_geom.get_quat(relative=False), unaligned_geom.get_quat(relative=False), tol=tol)

    # The child COM stays at its authored point while alignment moves the root to the composite COM. The authored
    # values come from the unaligned twin's description, since the aligned description is the quantity under test.
    child_com_user = child_desc.pos + gu.transform_by_quat(child_desc.inertial_pos, child_desc.quat)
    com_offset = gu.transform_by_quat(child_com_user, offset_quat)
    composite_user = base_desc.mass * base_desc.inertial_pos + child_desc.mass * child_com_user
    composite_user /= base_desc.mass + child_desc.mass
    anchor_offset = gu.transform_by_quat(composite_user, offset_quat)
    for entity, expected_anchor in ((entity_aligned, anchor_offset), (entity_unaligned, 0.0)):
        entity_pos = tensor_to_array(entity.get_pos())
        child_com = entity.get_links_pos(links_idx_local=1, ref=gs.link_ref_frame.link_COM)[..., 0, :]
        assert_allclose(child_com, entity_pos + com_offset, tol=tol)
        assert_allclose(entity.get_pos(relative=False), entity_pos + expected_anchor, tol=tol)
    # Per-environment variants: alignment must preserve the unaligned twin's COM offsets and composite anchor.
    aligned_com, unaligned_com = (
        entity.get_links_pos(links_idx_local=1, ref=gs.link_ref_frame.link_COM)[..., 0, :] - entity.get_pos()
        for entity in (entity_heterogeneous_aligned, entity_heterogeneous_unaligned)
    )
    assert_allclose(aligned_com, unaligned_com, tol=tol)
    masses = entity_heterogeneous_unaligned.get_links_mass()
    coms = entity_heterogeneous_unaligned.get_links_pos(ref=gs.link_ref_frame.link_COM)
    composite = (masses[..., None] * coms).sum(dim=-2) / masses.sum(dim=-1)[..., None]
    assert_allclose(
        entity_heterogeneous_aligned.get_pos(relative=False),
        composite - entity_heterogeneous_unaligned.get_pos() + entity_heterogeneous_aligned.get_pos(),
        tol=tol,
    )
    # The inverse weight of a link derives from its COM, so the child's must match the unaligned twin as well.
    twins = ((entity_aligned, entity_unaligned), (entity_heterogeneous_aligned, entity_heterogeneous_unaligned))
    for aligned_entity, unaligned_entity in twins:
        assert_allclose(
            aligned_entity.get_links_invweight(links_idx_local=1),
            unaligned_entity.get_links_invweight(links_idx_local=1),
            tol=tol,
        )

    # A pure local torque isolates the inertial-frame orientation because it has no application-point arm.
    for entity in scene.entities:
        entity.apply_links_external_wrench(torque=TORQUE, links_idx_local=1, ref=gs.link_ref_frame.link_COM, local=True)
    scene.step()
    for aligned_entity, unaligned_entity in twins:
        assert_allclose(
            aligned_entity.get_links_ang(links_idx_local=1), unaligned_entity.get_links_ang(links_idx_local=1), tol=tol
        )


def _build_two_link_revolute_urdf(
    name, geom_tag=None, geom_attribs=None, *, links_geoms=None, links_inertial=None, joint_type="prismatic"
):
    """Build a 2-link URDF file (prismatic joint by default) and return its path.

    Geometry can be specified either uniformly via (geom_tag, geom_attribs) — applied identically
    to all links — or per-link via links_geoms for full control.

    Parameters
    ----------
    links_geoms : list of list of (tag, attribs, origin_xyz) or None
        Per-link geometry specs. Each link gets a list of (tag, attribs, origin_xyz) tuples.
    links_inertial : list of (dict or None) or None
        Per-link inertial overrides. Each dict may contain 'mass', 'ixx', 'iyy', 'izz',
        'ixy', 'ixz', 'iyz', 'origin_xyz'. A None entry leaves that link's inertial unspecified
        (recomputed from geometry); None for the whole list does so for every link.
    joint_type : str
        Type of the joint between the two links ('prismatic', 'revolute', 'fixed', ...). A fixed joint makes the
        second link a fixed child of the first (a single rigid body).
    """
    robot = ET.Element("robot", name=name)

    link_defs = [("base", None), ("moving", "0.1 0 0")]
    for i_link, (link_name, default_origin_xyz) in enumerate(link_defs):
        link = ET.SubElement(robot, "link", name=link_name)
        if links_geoms is not None:
            geoms = links_geoms[i_link]
        else:
            geoms = [(geom_tag, geom_attribs, default_origin_xyz)]
        for tag, attribs, origin_xyz in geoms:
            for group_tag in ("visual", "collision"):
                group = ET.SubElement(link, group_tag)
                geom_el = ET.SubElement(group, "geometry")
                ET.SubElement(geom_el, tag, **attribs)
                if origin_xyz:
                    ET.SubElement(group, "origin", xyz=origin_xyz)
        if links_inertial and links_inertial[i_link] is not None:
            inertial_props = links_inertial[i_link]
            inertial = ET.SubElement(link, "inertial")
            ET.SubElement(inertial, "mass", value=str(inertial_props["mass"]))
            ET.SubElement(inertial, "origin", xyz=inertial_props["origin_xyz"])
            ET.SubElement(
                inertial,
                "inertia",
                ixx=str(inertial_props.get("ixx", 0)),
                ixy=str(inertial_props.get("ixy", 0)),
                ixz=str(inertial_props.get("ixz", 0)),
                iyy=str(inertial_props.get("iyy", 0)),
                iyz=str(inertial_props.get("iyz", 0)),
                izz=str(inertial_props.get("izz", 0)),
            )

    joint = ET.SubElement(robot, "joint", name="joint1", type=joint_type)
    ET.SubElement(joint, "parent", link="base")
    ET.SubElement(joint, "child", link="moving")
    ET.SubElement(joint, "origin", xyz="0.1 0 0")
    if joint_type != "fixed":
        ET.SubElement(joint, "axis", xyz="1 0 0")
        ET.SubElement(joint, "limit", lower="-1.0", upper="1.0", effort="100", velocity="1.0")

    return urdfpy.URDF._from_xml(robot, robot, get_assets_dir())


def _build_free_body_urdf(name, com_xyz):
    """Build a single free-floating link URDF with a box geom and an off-center COM, returning its path."""
    robot = ET.Element("robot", name=name)
    link = ET.SubElement(robot, "link", name="body")
    for group_tag in ("visual", "collision"):
        group = ET.SubElement(link, group_tag)
        geom_el = ET.SubElement(group, "geometry")
        ET.SubElement(geom_el, "box", size="0.04 0.04 0.04")
    inertial = ET.SubElement(link, "inertial")
    ET.SubElement(inertial, "mass", value="0.5")
    ET.SubElement(inertial, "origin", xyz=com_xyz)
    ET.SubElement(inertial, "inertia", ixx="1e-3", iyy="1e-3", izz="1e-3", ixy="0", ixz="0", iyz="0")
    return urdfpy.URDF._from_xml(robot, robot, get_assets_dir())


def _build_wrapped_free_body_urdf(name, child_mass):
    """Build a free body whose root link is empty (no geom, no inertial) and whose mass and geometry live on a fixed
    child link, returning its path. Exercises the empty-free-root wrapping a fixed massive child topology."""
    robot = ET.Element("robot", name=name)
    ET.SubElement(robot, "link", name="root")
    child = ET.SubElement(robot, "link", name="payload")
    for group_tag in ("visual", "collision"):
        group = ET.SubElement(child, group_tag)
        geom_el = ET.SubElement(group, "geometry")
        ET.SubElement(geom_el, "box", size="0.04 0.04 0.04")
    inertial = ET.SubElement(child, "inertial")
    ET.SubElement(inertial, "mass", value=str(child_mass))
    ET.SubElement(inertial, "origin", xyz="0 0 0")
    ET.SubElement(inertial, "inertia", ixx="1e-3", iyy="1e-3", izz="1e-3", ixy="0", ixz="0", iyz="0")
    joint = ET.SubElement(robot, "joint", name="weld", type="fixed")
    ET.SubElement(joint, "parent", link="root")
    ET.SubElement(joint, "child", link="payload")
    ET.SubElement(joint, "origin", xyz="0 0 0")
    return urdfpy.URDF._from_xml(robot, robot, get_assets_dir())


@pytest.mark.slow  # ~250s
@pytest.mark.required
def test_align_heterogeneous_inertial(show_viewer, tol):
    GRAVITY = -9.81

    # Variant A: sphere mesh collision with explicit inertial properties per link
    sphere_base_mass, sphere_moving_mass = 0.5, 0.3
    sphere_base_com = (0.0, 0.01, 0.0)
    sphere_moving_com = (0.05, 0.0, 0.0)
    sphere_base_inertia_diag = 1e-4
    sphere_moving_inertia_diag = 5e-5
    urdf_spheres = _build_two_link_revolute_urdf(
        "two_sphere_revolute",
        "mesh",
        {"filename": os.path.join(get_assets_dir(), "meshes", "sphere.obj"), "scale": "0.08 0.08 0.08"},
        links_inertial=[
            {
                "mass": sphere_base_mass,
                "ixx": sphere_base_inertia_diag,
                "iyy": sphere_base_inertia_diag,
                "izz": sphere_base_inertia_diag,
                "origin_xyz": " ".join(map(str, sphere_base_com)),
            },
            {
                "mass": sphere_moving_mass,
                "ixx": sphere_moving_inertia_diag,
                "iyy": sphere_moving_inertia_diag,
                "izz": sphere_moving_inertia_diag,
                "origin_xyz": " ".join(map(str, sphere_moving_com)),
            },
        ],
    )

    # Variant B: 2 half-height box primitives per link.
    # Setting zero inertial to test recomputation from geometry for non-primary morph.
    half_box = {"size": "0.04 0.04 0.02"}
    urdf_boxes = _build_two_link_revolute_urdf(
        "two_box_revolute",
        links_geoms=[
            [("box", half_box, "0 0 0.01"), ("box", half_box, "0 0 -0.01")],
            [("box", half_box, "0.1 0 0.01"), ("box", half_box, "0.1 0 -0.01")],
        ],
    )

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            gravity=(0.0, 0.0, GRAVITY),
        ),
        rigid_options=gs.options.RigidOptions(
            # Allow specifying different controller gains for each env
            batch_dofs_info=True,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.0, 1.0, 1.0),
            camera_lookat=(0.0, 0.0, 0.0),
        ),
        show_viewer=show_viewer,
    )

    scene.add_entity(gs.morphs.Plane())
    # align=True is requested but must be ignored for these articulated robots: a free base with a DOF-bearing child is
    # not a single rigid body, so its link frames and joint-space mass coupling must be left intact (aligning it would
    # misplace the moving child and drop the base coupling). The link-spacing and settling assertions below verify this.
    het_morph = (
        gs.morphs.URDF(file=urdf_spheres, pos=(0.5, 0, 0.08), align=True),
        gs.morphs.URDF(file=urdf_boxes, pos=(0, 0, 0.02), align=True),
    )
    het_obj = scene.add_entity(
        morph=het_morph,
        material=gs.materials.Rigid(
            rho=200.0,
            friction=1e-2,
        ),
    )
    # Kinematic heterogeneous URDF entity (same variants, different positions)
    het_kin = scene.add_entity(
        morph=het_morph,
        material=gs.materials.Kinematic(),
        surface=gs.surfaces.Default(
            color=(0.0, 0.0, 1.0, 0.4),
        ),
    )
    # Free-floating single-link URDF objects with different off-center COMs. Unlike the articulated robots above (whose
    # requested alignment is ignored), each variant is a basic rigid object, so its link frame is moved to its own COM.
    FREE_POS = (3.0, 0.0, 0.2)
    free_morph = (
        gs.morphs.URDF(file=_build_free_body_urdf("free_body_a", "0.02 0 0"), pos=FREE_POS, align=True),
        gs.morphs.URDF(file=_build_free_body_urdf("free_body_b", "0 0 0.03"), pos=FREE_POS, align=True),
    )
    free_het = scene.add_entity(
        morph=free_morph,
        material=gs.materials.Rigid(
            rho=200.0,
        ),
    )
    # Kinematic counterpart of the aligned free bodies. The COM/principal anchoring is applied in the base entity, so
    # for the same qpos a kinematic visualization and the rigid body it tracks must place identical world geometry.
    free_kin = scene.add_entity(
        morph=free_morph,
        material=gs.materials.Kinematic(),
    )
    # A free entity whose two variants are identical, so the solver resolves a single shared offset and takes the
    # broadcast path (the base link offset) rather than the per-env variant offset. That shared base offset must still
    # carry the COM anchoring, else the relative getter cannot strip the alignment on this path.
    dup_morph = (
        gs.morphs.URDF(file=_build_free_body_urdf("free_dup_a", "0.02 0 0"), pos=FREE_POS, align=True),
        gs.morphs.URDF(file=_build_free_body_urdf("free_dup_b", "0.02 0 0"), pos=FREE_POS, align=True),
    )
    free_dup = scene.add_entity(
        morph=dup_morph,
        material=gs.materials.Rigid(),
    )
    # Free bodies whose root link is empty and whose mass lives on a fixed child (merge_fixed_links=False keeps the
    # wrapper). The empty root reads the gs.EPS placeholder and the child its authored mass.
    WRAP_MASS_A, WRAP_MASS_B = 0.5, 0.25
    wrapped_morph = (
        gs.morphs.URDF(
            file=_build_wrapped_free_body_urdf("wrap_a", WRAP_MASS_A),
            pos=(5.0, 0.0, 0.2),
            align=True,
            merge_fixed_links=False,
        ),
        gs.morphs.URDF(
            file=_build_wrapped_free_body_urdf("wrap_b", WRAP_MASS_B),
            pos=(5.0, 0.0, 0.2),
            align=True,
            merge_fixed_links=False,
        ),
    )
    free_wrapped = scene.add_entity(
        morph=wrapped_morph,
    )
    scene.build(n_envs=4, env_spacing=(0.0, 0.5))

    # Same absolute qpos must map to the same world geometry for the aligned rigid body and its kinematic counterpart;
    # an unanchored kinematic entity would interpret the qpos in a different link frame and diverge.
    free_qpos = (1.0, 0.5, 0.8, 0.6, 0.5, 0.3, 0.0)
    free_het.set_qpos(free_qpos)
    free_kin.set_qpos(free_qpos)
    assert_allclose(free_het.get_vAABB(), free_kin.get_vAABB(), tol=tol)
    scene.reset()

    # Each free-body variant (env 0 variant A, env 2 variant B) is aligned to its own COM: the link origin coincides
    # with the COM, and the relative getter strips the alignment back to the user pose.
    for i_env in (0, 2):
        assert_allclose(
            free_het.get_links_pos(links_idx_local=[0], ref=gs.link_ref_frame.link_COM, envs_idx=i_env, relative=False),
            free_het.get_links_pos(
                links_idx_local=[0], ref=gs.link_ref_frame.link_origin, envs_idx=i_env, relative=False
            ),
            tol=tol,
        )
        assert_allclose(free_het.get_pos(relative=True, envs_idx=i_env), FREE_POS, tol=tol)

    # The duplicate-variant entity takes the broadcast offset path; its relative getter must still strip the shared COM
    # anchoring back to the user pose in every env (a non-anchored base offset would leave it COM-shifted).
    assert_allclose(
        free_dup.get_links_pos(links_idx_local=[0], ref=gs.link_ref_frame.link_COM, relative=False),
        free_dup.get_links_pos(links_idx_local=[0], ref=gs.link_ref_frame.link_origin, relative=False),
        tol=tol,
    )
    assert_allclose(free_dup.get_pos(relative=True), FREE_POS, tol=tol)
    assert_allclose(gu.quat_to_xyz(free_dup.get_quat(relative=True)), 0.0, tol=tol)

    # Relative set_pos on a boolean-masked subset of envs: each selected env's relative getter must report its target
    # back, stripping its own per-variant offset.
    mask = torch.tensor([True, False, True, False], device=gs.device)
    free_new_pos = torch.tensor([[1.0, 0.0, 0.5], [2.0, 0.0, 0.5]], dtype=gs.tc_float, device=gs.device)
    free_het.set_pos(free_new_pos, envs_idx=mask, relative=True)
    free_pos = free_het.get_pos(relative=True)
    assert_allclose(free_pos[[0, 2]], free_new_pos, tol=tol)
    assert_allclose(free_pos[[1, 3]], FREE_POS, tol=tol)

    # Joint structure: both variants share the same joints (root_joint + joint1)
    assert len(het_obj.joints) == 2
    assert len(het_obj.links) == 2
    assert het_obj.get_qpos().shape == (4, 8)  # free joint (7) + prismatic (1)
    assert het_obj.get_dofs_velocity().shape == (4, 7)  # free joint (6) + prismatic (1)

    # Check that kinematic vAABB matches rigid
    assert_allclose(het_kin.get_vAABB(), het_obj.get_vAABB(), tol=1e-3)

    # Verify initial z-positions match per-variant morph.pos (z is unaffected by env_spacing)
    het_pos_init = het_obj.get_pos()
    assert_allclose(het_pos_init[0, 2], 0.08, tol=tol)
    assert_allclose(het_pos_init[1, 2], 0.08, tol=tol)
    assert_allclose(het_pos_init[2, 2], 0.02, tol=tol)
    assert_allclose(het_pos_init[3, 2], 0.02, tol=tol)
    # Variant B has x-offset relative to variant A
    assert_allclose(het_pos_init[0, 0] - het_pos_init[2, 0], 0.5, tol=tol)
    assert_allclose(het_pos_init[1, 0] - het_pos_init[3, 0], 0.5, tol=tol)
    het_links_pos_init = het_obj.get_links_pos(relative=False)
    assert_allclose(het_links_pos_init.diff(dim=-2), (0.1, 0, 0), tol=tol)

    # Same-variant envs produce identical results (balanced block [A, A, B, B])
    het_pos = het_obj.get_pos()
    het_qpos = het_obj.get_qpos()
    assert_allclose(het_pos[0], het_pos[1], tol=tol)
    assert_allclose(het_pos[2], het_pos[3], tol=tol)
    assert_allclose(het_qpos[0], het_qpos[1], tol=tol)
    assert_allclose(het_qpos[2], het_qpos[3], tol=tol)

    # Different-variant envs produce different results
    with pytest.raises(AssertionError):
        assert_allclose(het_pos[0], het_pos[2], tol=tol)
    with pytest.raises(AssertionError):
        assert_allclose(het_qpos[0], het_qpos[2], tol=tol)

    # Mass differs between variants
    mass = het_obj.get_mass()
    assert mass.shape == (scene.n_envs,)
    assert_allclose(mass[0], mass[1], tol=tol)
    assert_allclose(mass[2], mass[3], tol=tol)
    assert not torch.allclose(mass[0], mass[2], atol=tol, rtol=tol), "Variant A and B masses should differ"
    # Variant B total mass should match the explicit URDF inertial values
    assert_allclose(mass[0], sphere_base_mass + sphere_moving_mass, tol=tol)

    # CoM position: variant B should match explicit URDF inertial origin_xyz
    com_pos = het_obj.get_links_pos(ref=gs.link_ref_frame.link_COM, relative=False)
    origin_pos = het_obj.get_links_pos(ref=gs.link_ref_frame.link_origin, relative=False)
    com_offset = com_pos - origin_pos
    # Variant A: CoM offset matches URDF inertial origin
    assert_allclose(com_offset[0, 0], sphere_base_com, tol=tol)
    assert_allclose(com_offset[0, 1], sphere_moving_com, tol=tol)
    assert_allclose(com_offset[1, 0], sphere_base_com, tol=tol)
    assert_allclose(com_offset[1, 1], sphere_moving_com, tol=tol)
    # Variant B: symmetric split boxes => CoM at link origin for base, at geometry center for moving
    assert_allclose(com_offset[2, 0], 0.0, tol=tol)
    assert_allclose(com_offset[2, 1, 0], 0.1, tol=tol)  # x-offset of moving link geometry
    # Same-variant consistency
    assert_allclose(com_offset[2], com_offset[3], tol=tol)
    # CoM differs between variants on base link (non-zero y-offset for variant B)
    with pytest.raises(AssertionError):
        assert_allclose(com_offset[0, 0], com_offset[2, 0], tol=tol)

    # Inertia matrix: variant B should match explicit URDF values
    links_idx = slice(het_obj.link_start, het_obj.link_end)
    inertial_i = qd_to_numpy(scene.rigid_solver.dyn_info.links.inertial_i, None, links_idx, transpose=True)
    # Variant A: diagonal inertia matches URDF
    assert_allclose(inertial_i[[0, 1], 0], np.eye(3) * sphere_base_inertia_diag, tol=tol)
    assert_allclose(inertial_i[[0, 1], 1], np.eye(3) * sphere_moving_inertia_diag, tol=tol)
    # Variant B: Recomputed inertia from geometry
    assert_allclose(inertial_i[[2, 3]], np.eye(3) * ((0.04**5 / 6.0) * het_obj.material.rho), tol=tol)
    # Same-variant consistency
    assert_allclose(inertial_i[2], inertial_i[3], tol=tol)
    # Variants differ
    with pytest.raises(AssertionError):
        assert_allclose(inertial_i[0, 0], inertial_i[2, 0], tol=tol)

    # The root (link 0) reads the gs.EPS placeholder of a geometry-free link, the child (link 1) its authored mass. The
    # environments hold [A, A, B, B].
    wrapped_mass = free_wrapped.get_links_mass()
    assert_allclose(wrapped_mass[:, 0], gs.EPS, tol=gs.EPS)
    assert_allclose(wrapped_mass[[0, 1], 1], WRAP_MASS_A, tol=gs.EPS)
    assert_allclose(wrapped_mass[[2, 3], 1], WRAP_MASS_B, tol=gs.EPS)

    # Check contacts
    for i in range(4):
        for _ in range(10):
            scene.step()
        pos = het_obj.get_pos()
        assert_allclose(pos[:2, [0, 2]], het_pos_init[:2, [0, 2]], tol=1e-3)
        assert_allclose(pos[:2, 1], het_pos_init[:2, 1], tol=0.02)
        assert_allclose(pos[2:], het_pos_init[2:], tol=2e-4)
        het_obj.set_quat(gu.euler_to_quat((90 * i, 0, 0)))

    # Apply control and simulate for a while
    target_dof_pos = np.array((0.05, 0.1, 0.01, 0.02), dtype=gs.np_float)
    het_obj.set_dofs_kp((1000.0, 1000.0, 100.0, 100.0), dofs_idx_local=-1)
    het_obj.set_dofs_kv((100.0, 100.0, 10.0, 10.0), dofs_idx_local=-1)
    het_obj.control_dofs_position(target_dof_pos, dofs_idx_local=-1)
    for _ in range(100):
        scene.step()

    # Velocity should be near zero (settled)
    assert_allclose(het_obj.get_vel(), 0.0, tol=0.05)

    # All objects should be near their initial z-positions (settled on ground)
    pos = het_obj.get_pos()
    assert_allclose(pos[..., 2], het_pos_init[..., 2], tol=1e-3)

    # Check that dof position is correct
    dof_pos = het_obj.get_dofs_position()
    assert_allclose(dof_pos[..., -1], target_dof_pos, tol=1e-3)
    het_links_pos = het_obj.get_links_pos(relative=False)
    assert_allclose(het_links_pos[..., 1, 0] - het_links_pos[..., 0, 0], target_dof_pos + 0.1, tol=1e-3)
    assert_allclose(het_links_pos[..., 1, 1:], het_links_pos[..., 0, 1:], tol=5e-3)

    # Check that the acceleration is matching the analytical formula
    links_mass = qd_to_numpy(scene.rigid_solver.dyn_info.links.inertial_mass, None, links_idx, transpose=True)
    force = np.zeros((scene.n_envs, 2, 3))
    force[..., 2] = -links_mass * GRAVITY
    het_obj.set_pos((0, 0, 0.2))
    het_obj.control_dofs_force(0.0, dofs_idx_local=-1)
    scene.step()
    assert_allclose(het_obj.get_links_acc()[..., 2], GRAVITY, tol=tol)
    het_obj.zero_all_dofs_velocity()
    for _ in range(10):
        scene.rigid_solver.apply_links_external_wrench(force, links_idx=links_idx, ref=gs.link_ref_frame.link_COM)
        scene.step()
        assert_allclose(het_obj.get_links_acc(), 0.0, tol=tol)


def create_two_free_bodies_mjcf(name, pos_a, geom_a, pos_b, geom_b):
    """Helper to create an MJCF with two free root bodies, each a single box geom offset from its body origin."""
    mjcf = ET.Element("mujoco", model=name)
    worldbody = ET.SubElement(mjcf, "worldbody")
    body_a = ET.SubElement(worldbody, "body", name="a", pos=f"{pos_a[0]} {pos_a[1]} {pos_a[2]}")
    ET.SubElement(body_a, "joint", name="a_free", type="free")
    ET.SubElement(
        body_a, "geom", type="box", size="0.05 0.05 0.05", pos=f"{geom_a[0]} {geom_a[1]} {geom_a[2]}", density="1000"
    )
    body_b = ET.SubElement(worldbody, "body", name="b", pos=f"{pos_b[0]} {pos_b[1]} {pos_b[2]}")
    ET.SubElement(body_b, "joint", name="b_free", type="free")
    ET.SubElement(
        body_b, "geom", type="box", size="0.05 0.08 0.03", pos=f"{geom_b[0]} {geom_b[1]} {geom_b[2]}", density="1000"
    )
    return mjcf


@pytest.mark.required
def test_multi_root_offset(show_viewer, tol):
    # To exercise per-root offset tracking the entity MUST hold more than one free root (one MJCF, several free
    # bodies) with DISTINCT per-root geometry, so each root gets its own alignment offset; a single root - or
    # identical roots - would not surface the cross-contamination bug (one root's offset leaking into another).
    BODY_A_POS = (1.0, 0.0, 0.5)
    BODY_B_POS = (-1.0, 0.0, 0.5)
    GEOM_A_POS = (0.02, 0.01, 0.0)
    GEOM_B_POS = (0.0, 0.03, 0.02)

    scene = gs.Scene(
        show_viewer=show_viewer,
        show_FPS=False,
    )
    entity = scene.add_entity(
        gs.morphs.MJCF(
            file=ET.tostring(
                create_two_free_bodies_mjcf("two_bodies", BODY_A_POS, GEOM_A_POS, BODY_B_POS, GEOM_B_POS),
                encoding="unicode",
            ),
            align=True,
        ),
    )
    scene.build()

    link_a, link_b = entity.links
    # Each root reports its own user-specified pose in the relative frame, independent of the other root.
    assert_allclose(link_a.get_pos(), BODY_A_POS, tol=tol)
    assert_allclose(link_b.get_pos(), BODY_B_POS, tol=tol)
    assert_allclose(link_a.get_quat(), gu.identity_quat(), tol=tol)
    assert_allclose(link_b.get_quat(), gu.identity_quat(), tol=tol)

    # The world frame carries each root's own COM shift (the box center), confirming the offsets are not shared.
    assert_allclose(link_a.get_pos(relative=False), np.add(BODY_A_POS, GEOM_A_POS), tol=tol)
    assert_allclose(link_b.get_pos(relative=False), np.add(BODY_B_POS, GEOM_B_POS), tol=tol)

    # Both roots free-fall under gravity: the relative getter tracks each user frame, holding x/y and dropping z
    # equally (free fall is mass-independent).
    for _ in range(20):
        scene.step()
    assert_allclose(link_a.get_pos()[..., :2], BODY_A_POS[:2], tol=tol)
    assert_allclose(link_b.get_pos()[..., :2], BODY_B_POS[:2], tol=tol)
    assert_allclose(link_a.get_pos()[..., 2], link_b.get_pos()[..., 2], tol=tol)


@pytest.mark.required
@pytest.mark.parametrize("is_fixed", [False, True])
@pytest.mark.parametrize("merge_fixed_links", [False, True])
def test_merge_entities(is_fixed, merge_fixed_links, show_viewer, tol, monkeypatch):
    # Force parallelism on CPU to trigger any cross-entity race condition
    if gs.backend == gs.cpu:
        monkeypatch.setenv("GS_PARA_LEVEL", "2")
        monkeypatch.setenv("QD_NUM_THREADS", "3")

    EULER_OFFSET = (0, 0, 45)
    TOOL_MOUNT_POS = (0.0, 0.0, 0.05)
    TOOL_MOUNT_QUAT = (math.cos(math.pi / 8), math.sin(math.pi / 8), 0.0, 0.0)  # 45 deg about x
    GHOST_MOUNT_POS = (0.0, 0.6, 0.3)
    GHOST_MOUNT_OFFSET = (0.0, 0.0, 0.1)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
        ),
        rigid_options=gs.options.RigidOptions(
            enable_self_collision=True,
            enable_neutral_collision=True,
            enable_adjacent_collision=False,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0, -3.5, 2.5),
            camera_lookat=(0.0, 0.0, 0.5),
        ),
        show_viewer=show_viewer,
    )

    scene.add_entity(gs.morphs.Plane())

    franka = scene.add_entity(
        gs.morphs.URDF(
            file="urdf/panda_bullet/panda_nohand.urdf",
            merge_fixed_links=False,
            fixed=True,
        ),
        vis_mode="collision",
    )
    hand = scene.add_entity(
        gs.morphs.URDF(
            file="urdf/panda_bullet/hand.urdf",
            euler=EULER_OFFSET,
            fixed=is_fixed,
            merge_fixed_links=merge_fixed_links,
            batch_fixed_verts=is_fixed,
        ),
        vis_mode="collision",
    )
    ghost_mount = scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.05, 0.05, 0.05),
            pos=GHOST_MOUNT_POS,
        ),
        material=gs.materials.Kinematic(),
    )
    ghost_tool = scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.02, 0.02, 0.02),
            pos=(2.0, 2.0, 2.0),
        ),
        material=gs.materials.Kinematic(),
    )
    tool = scene.add_entity(
        gs.morphs.Sphere(
            radius=0.005,
            # A conflicting morph pose: the explicit mounting transform below must override it.
            pos=(1.0, -2.0, 3.0),
        ),
    )
    box = scene.add_entity(
        gs.morphs.Box(
            size=(0.02, 0.02, 0.02),
            pos=(0.3, 0.0, 0.01),
        ),
    )
    with pytest.raises(gs.GenesisException):
        franka.attach(hand, "right_finger")
    # A merged pair is one kinematic tree, numbered within one solver, so the two entities must share one.
    with pytest.raises(gs.GenesisException):
        ghost_tool.attach(hand, "right_finger")
    ghost_tool.attach(ghost_mount, ghost_mount.base_link.name, pos=GHOST_MOUNT_OFFSET)
    # Omitting the mounting transform keeps the child's morph pose acting as the mount. The kinematic pair above
    # stands ahead of this one, so a mount re-indexes the entities its own solver holds after it.
    hand.attach(franka, "attachment")
    # Malformed mounting transforms (wrong shape, or a zero-length quaternion) raise before any kinematic-tree
    # mutation, leaving the entity attachable.
    with pytest.raises(gs.GenesisException):
        tool.attach(hand, "right_finger", pos=(0.0, 0.0))
    with pytest.raises(gs.GenesisException):
        tool.attach(hand, "right_finger", quat=(1.0, 0.0, 0.0))
    with pytest.raises(gs.GenesisException):
        tool.attach(hand, "right_finger", quat=(0.0, 0.0, 0.0, 0.0))
    tool.attach(hand, "right_finger", pos=TOOL_MOUNT_POS, quat=TOOL_MOUNT_QUAT)
    scene.build()
    with pytest.raises(gs.GenesisException):
        box.attach(hand, "right_finger")

    # Make sure that collision between hand base link and franka attachment point has been filtered out as adjacent
    collision_pair_idx = scene.rigid_solver.collider.collider_info.collision_pair_idx.to_numpy()
    assert collision_pair_idx[franka.get_link("attachment").idx, hand.base_link_idx] == -1

    with pytest.raises(gs.GenesisException):
        hand.set_pos(0.0)
    with pytest.raises(gs.GenesisException):
        hand.set_quat(0.0)

    # The free box is dynamically isolated from the robot, so its lateral position must stay put while the
    # gripper actuates. Attaching the floating-base hand re-indexes joints by dropping its free base joint;
    # the hand's mimic (joint-equality) references must follow that re-indexing, otherwise they alias this
    # box's free-joint DOFs and the corrupted constraint drags the box sideways as the fingers move.
    box_pos_init = box.get_pos()

    franka.control_dofs_position([-1, 0.8, 1, -2, 1, 0.5, -0.5])
    hand.control_dofs_position([0.04, 0.04])
    for _ in range(30):
        scene.step()

    assert_allclose(box.get_pos()[..., :2], box_pos_init[..., :2], tol=1e-3)

    attach_link = franka.get_link("attachment")
    assert_allclose(attach_link.get_pos(), hand.links[0].get_pos(), tol=gs.EPS)
    offset_quat = gu.transform_quat_by_quat(hand.links[0].get_quat(), gu.inv_quat(attach_link.get_quat()))
    assert_allclose(gu.quat_to_xyz(offset_quat, rpy=False, degrees=True), EULER_OFFSET, tol=tol)
    for link in hand.links[slice(0, None) if merge_fixed_links else slice(1, -1)]:
        assert torch.linalg.norm(link.get_pos() - attach_link.get_pos(), dim=-1) < 0.08
    if not merge_fixed_links:
        assert_allclose(torch.linalg.norm(hand.links[-1].get_pos() - attach_link.get_pos(), dim=-1), 0.105, tol=tol)

    # The tool's explicit mounting transform overrides its conflicting morph pose: it sits at (pos, quat) in the
    # right-finger frame rather than at the finger origin.
    finger_link = hand.get_link("right_finger")
    expected_tool_pos = finger_link.get_pos() + gu.transform_by_quat(
        torch.tensor(TOOL_MOUNT_POS, dtype=gs.tc_float, device=gs.device), finger_link.get_quat()
    )
    expected_tool_quat = gu.transform_quat_by_quat(
        torch.tensor(TOOL_MOUNT_QUAT, dtype=gs.tc_float, device=gs.device), finger_link.get_quat()
    )
    assert_allclose(tool.get_pos(), expected_tool_pos, tol=tol)
    assert_allclose(tool.get_quat(), expected_tool_quat, tol=tol)

    # A kinematic entity attaches onto another one the same way. The mounting transform places it in the parent
    # link frame, as for a rigid pair.
    assert_allclose(ghost_tool.get_pos(), np.add(GHOST_MOUNT_POS, GHOST_MOUNT_OFFSET), tol=gs.EPS)
    assert ghost_tool.desc.attachment.entity_name == ghost_mount.name

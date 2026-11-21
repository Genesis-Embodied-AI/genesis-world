import numpy as np
import pytest
import torch

import genesis as gs
import genesis.utils.geom as gu
from genesis.utils.misc import tensor_to_array

from ..utils.assertions import assert_allclose, assert_equal
from ..utils.assets import get_hf_dataset


@pytest.mark.required
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
@pytest.mark.parametrize("broadphase_traversal", [gs.broadphase_traversal.SAP, gs.broadphase_traversal.ALL_VS_ALL])
def test_physics_parity(
    fixed_base_dual_arm,
    fixed_base_dual_arm_high_damping,
    fixed_base_dual_arm_chained,
    fixed_base_single_arm_with_forearms,
    broadphase_traversal,
    show_viewer,
    tol,
):
    # Uses the fixed-child mesh objects from 'test_convexify' (offset center of mass, distinct mass) so the per-env
    # parity check exercises the inertia alignment, not just trivially-symmetric primitives.
    N_STEPS = 100
    # The arms of every dual arm variant are still falling apart from each other after these steps. Once they press
    # against each other, friction leaves a whole range of rest configurations to choose from, which the rounding of
    # each simulation picks, so the dual arms are compared with their references before.
    N_STEPS_FALLING = 40
    DROP_HEIGHT = 0.2
    VARIANTS = (("mug_1", "output.xml"), ("donut_0", "output.xml"), ("cup_2", "model.xml"), ("apple_15", "model.xml"))
    # Divergent per-variant yaw offset, stripped to identity by the relative getter and carried by the world frame.
    # Applied identically to the homogeneous reference so the dynamics still match.
    OFFSET_EULERS = ((0.0, 0.0, 30.0), (0.0, 0.0, -45.0), (0.0, 0.0, 90.0), (0.0, 0.0, -120.0))
    # Distinct per-variant placement, dispatched per environment.
    POSITIONS = ((0.0, 0.0, DROP_HEIGHT), (0.2, 0.0, DROP_HEIGHT), (0.0, 0.2, DROP_HEIGHT), (0.2, 0.2, DROP_HEIGHT))
    # The homogeneous references live in the same scene, offset far enough that no entity interacts with those of
    # another offset: a single build compiles one kernel set instead of one per scene. The references at the offset
    # 'i_env' replicate the environment 'i_env' of the heterogeneous entities.
    REFERENCE_OFFSETS = ((10.0, 0.0, 0.0), (20.0, 0.0, 0.0), (30.0, 0.0, 0.0), (40.0, 0.0, 0.0))
    # Beside the objects, the arms of an articulated heterogeneous entity fall against each other, and cubes of a
    # heterogeneous pool rest on its torso, whose top lies 0.05 above its center at unit scale. The dual arm variants
    # differ in their shoulder damping, scale and fixed base pose. The dual arm pool is smaller than the cube pool, so
    # that every dual arm variant meets several cube variants: environments 0-1 carry the first dual arm variant, and
    # environments 2-3 the second one. The geoms of a variant an environment does not carry stay at the initial pose of
    # the variant it carries, so a cube starting on the torso leaves the larger cube variants penetrating the torso in
    # its environment, where they must not collide.
    DUAL_ARM_POSITIONS = ((1.0, 0.0, 0.5), (1.0, 0.1, 0.6))
    DUAL_ARM_EULERS = ((0.0, 0.0, 0.0), (0.0, 0.0, 90.0))
    DUAL_ARM_SCALES = (1.0, 1.25)
    CUBE_SIZES = (0.04, 0.05, 0.06, 0.07)
    CUBE_POSITIONS = tuple(
        (pos[0], pos[1], pos[2] + 0.05 * scale + 0.5 * cube_size)
        for pos, scale, cube_size in zip(
            np.repeat(DUAL_ARM_POSITIONS, 2, axis=0), np.repeat(DUAL_ARM_SCALES, 2), CUBE_SIZES
        )
    )

    # Arms whose variants differ in their kinematic trees: the primary hangs its right arm from its left one, the second
    # variant has a single arm extended by two forearms, and the third hangs both arms from the torso. The entity holds
    # the union of their trees: the first forearm shares the link of the right arm of the primary, which hangs from the
    # left arm too, while the second forearm and the right arm hanging from the torso add links, the latter renamed.
    # The arms hang low enough to swing into the ground, while a position control drives every dof of the entity.
    ARMS_POS = (2.0, 0.0, 0.2)
    ARMS_ENVS_VARIANT = (0, 0, 1, 2)
    ARMS_LINKS_NAME = ("torso", "left_arm", "right_arm", "left_forearm_1", "right_arm_2")
    # Per variant, the links and the dofs of the entity standing for its own, in its own order
    ARMS_LINKS_IDX = ((0, 1, 2), (0, 1, 2, 3), (0, 1, 4))
    ARMS_DOFS_IDX = ((0, 1), (0, 1, 2), (0, 3))
    ARMS_KP = 20.0
    ARMS_TARGET = 1.2

    dual_arm_files = (fixed_base_dual_arm, fixed_base_dual_arm_high_damping)
    arms_files = (fixed_base_dual_arm_chained, fixed_base_single_arm_with_forearms, fixed_base_dual_arm)
    asset_files = tuple(f"{get_hf_dataset(pattern=f'{name}/*')}/{name}/{xml}" for name, xml in VARIANTS)

    # One homogeneous reference entity per variant plus a single heterogeneous entity dispatching one variant per
    # environment, all in one scene.
    scene = gs.Scene(
        rigid_options=gs.options.RigidOptions(
            broadphase_traversal=broadphase_traversal,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.0, -2.2, 1.0),
            camera_lookat=(1.0, 0.05, 0.25),
        ),
        show_viewer=show_viewer,
    )
    plane = scene.add_entity(gs.morphs.Plane())
    ref_dual_arms = []
    for i_env, offset in enumerate(REFERENCE_OFFSETS):
        ref_dual_arms.append(
            scene.add_entity(
                gs.morphs.URDF(
                    file=dual_arm_files[i_env // 2],
                    pos=np.add(DUAL_ARM_POSITIONS[i_env // 2], offset),
                    euler=DUAL_ARM_EULERS[i_env // 2],
                    scale=DUAL_ARM_SCALES[i_env // 2],
                    fixed=True,
                ),
            )
        )
    het_dual_arm = scene.add_entity(
        morph=tuple(
            gs.morphs.URDF(file=file, pos=pos, euler=euler, scale=scale, fixed=True)
            for file, pos, euler, scale in zip(dual_arm_files, DUAL_ARM_POSITIONS, DUAL_ARM_EULERS, DUAL_ARM_SCALES)
        )
    )
    ref_cubes = []
    for cube_size, pos, offset in zip(CUBE_SIZES, CUBE_POSITIONS, REFERENCE_OFFSETS):
        ref_cubes.append(
            scene.add_entity(
                gs.morphs.Box(
                    size=(cube_size, cube_size, cube_size),
                    pos=(pos[0] + offset[0], pos[1] + offset[1], pos[2] + offset[2]),
                ),
            )
        )
    het_cube = scene.add_entity(
        morph=tuple(
            gs.morphs.Box(size=(cube_size, cube_size, cube_size), pos=pos)
            for cube_size, pos in zip(CUBE_SIZES, CUBE_POSITIONS)
        )
    )
    ref_objs = []
    for file, pos, offset_euler, offset in zip(asset_files, POSITIONS, OFFSET_EULERS, REFERENCE_OFFSETS):
        ref_objs.append(
            scene.add_entity(
                gs.morphs.MJCF(
                    file=file,
                    pos=(pos[0] + offset[0], pos[1] + offset[1], pos[2] + offset[2]),
                    offset_euler=offset_euler,
                ),
            )
        )
    het_obj = scene.add_entity(
        morph=tuple(
            gs.morphs.MJCF(file=file, pos=pos, offset_euler=offset_euler)
            for file, pos, offset_euler in zip(asset_files, POSITIONS, OFFSET_EULERS)
        )
    )
    ref_arms = []
    for i_variant, offset in zip(ARMS_ENVS_VARIANT, REFERENCE_OFFSETS):
        ref_arms.append(
            scene.add_entity(
                gs.morphs.URDF(
                    file=arms_files[i_variant],
                    pos=np.add(ARMS_POS, offset),
                    fixed=True,
                ),
            )
        )
    het_arms = scene.add_entity(
        morph=tuple(gs.morphs.URDF(file=file, pos=ARMS_POS, fixed=True) for file in arms_files),
    )
    scene.build(n_envs=len(VARIANTS))

    # Each link and joint of the arms is carried by the environments whose variant has a link standing for it
    assert tuple(link.name for link in het_arms.links) == ARMS_LINKS_NAME
    for i_l, link in enumerate(het_arms.links):
        envs_idx = [i_env for i_env, i_variant in enumerate(ARMS_ENVS_VARIANT) if i_l in ARMS_LINKS_IDX[i_variant]]
        assert_equal(link.active_envs_idx, envs_idx)
        for joint in link.joints:
            assert_equal(joint.active_envs_idx, envs_idx)

    # The position control drives every dof of the arms, those of the links an environment does not carry included
    het_arms.set_dofs_kp(ARMS_KP)
    het_arms.control_dofs_position(np.full((len(VARIANTS), het_arms.n_dofs), ARMS_TARGET))
    for i_env, ref_arm in enumerate(ref_arms):
        ref_arm.set_dofs_kp(ARMS_KP)
        ref_arm.control_dofs_position(np.full((len(VARIANTS), ref_arm.n_dofs), ARMS_TARGET))

    # At init each variant sits at its own placement; the relative getter strips its offset (and inertial alignment) to
    # identity in the user frame, while the world frame matches the homogeneous reference's world orientation.
    assert_allclose(gu.quat_to_xyz(het_obj.get_quat(relative=True)), 0.0, tol=tol)
    assert_allclose(het_obj.get_pos(), POSITIONS, tol=tol)
    # Matching the reference in both frames validates that the inertial alignment is applied identically to the
    # heterogeneous entity and the homogeneous references.
    for relative in (True, False):
        ref_quats = torch.cat(
            [ref_obj.get_quat(envs_idx=[i_env], relative=relative) for i_env, ref_obj in enumerate(ref_objs)]
        )
        assert_allclose(het_obj.get_quat(relative=relative), ref_quats, tol=tol)

    for _ in range(N_STEPS_FALLING):
        scene.step()

    # Every environment simulates the link frames and joints of the dual arm variant it carries, as its reference does
    ref_dual_arm_qpos = torch.cat([ref.get_qpos(envs_idx=[i_env]) for i_env, ref in enumerate(ref_dual_arms)])
    assert_allclose(het_dual_arm.get_qpos(), ref_dual_arm_qpos, tol=tol)
    ref_links_pos = torch.cat([ref.get_links_pos(envs_idx=[i_env]) for i_env, ref in enumerate(ref_dual_arms)])
    assert_allclose(ref_links_pos - het_dual_arm.get_links_pos(), np.expand_dims(REFERENCE_OFFSETS, 1), tol=tol)
    ref_dofs_damping = torch.cat([ref.get_dofs_damping(envs_idx=[i_env]) for i_env, ref in enumerate(ref_dual_arms)])
    assert_allclose(het_dual_arm.get_dofs_damping(), ref_dofs_damping, tol=gs.EPS)

    # Every environment simulates the arms of its variant as its reference does, through contacts with the ground, and
    # the dofs of the links it does not carry stay at rest and receive no force.
    arms_qpos = tensor_to_array(het_arms.get_qpos())
    arms_dofs_vel = tensor_to_array(het_arms.get_dofs_velocity())
    arms_dofs_force = tensor_to_array(het_arms.get_dofs_control_force())
    arms_links_pos = tensor_to_array(het_arms.get_links_pos())
    for i_env, (i_variant, ref_arm, offset) in enumerate(zip(ARMS_ENVS_VARIANT, ref_arms, REFERENCE_OFFSETS)):
        dofs_idx = list(ARMS_DOFS_IDX[i_variant])
        assert_allclose(arms_qpos[i_env, dofs_idx], ref_arm.get_qpos(envs_idx=i_env), tol=tol)
        assert_allclose(arms_dofs_vel[i_env, dofs_idx], ref_arm.get_dofs_velocity(envs_idx=i_env), tol=tol)
        assert_allclose(arms_dofs_force[i_env, dofs_idx], ref_arm.get_dofs_control_force(envs_idx=i_env), tol=tol)
        links_pos = arms_links_pos[i_env, list(ARMS_LINKS_IDX[i_variant])]
        assert_allclose(tensor_to_array(ref_arm.get_links_pos(envs_idx=i_env)) - links_pos, offset, tol=tol)
        dofs_is_padding = np.ones(het_arms.n_dofs, dtype=bool)
        dofs_is_padding[dofs_idx] = False
        for value in (arms_qpos, arms_dofs_vel, arms_dofs_force):
            assert_equal(value[i_env, dofs_is_padding], 0.0)
    assert tensor_to_array(het_arms.get_contacts(with_entity=plane)["valid_mask"]).any(axis=-1).all()

    for _ in range(N_STEPS - N_STEPS_FALLING):
        scene.step()

    # After the drop each environment matches the homogeneous reference of its variant in pose, velocity and mass.
    ref_pos = torch.cat([ref_obj.get_pos(envs_idx=[i_env]) for i_env, ref_obj in enumerate(ref_objs)])
    ref_vel = torch.cat([ref_obj.get_vel(envs_idx=[i_env]) for i_env, ref_obj in enumerate(ref_objs)])
    # Batch layout and translation change rounding, which can change which nearly equivalent corners pruning retains.
    # The velocity comparison allows their accumulated effect over the drop.
    assert_allclose(ref_pos - het_obj.get_pos(), REFERENCE_OFFSETS, tol=1e-5)
    assert_allclose(het_obj.get_vel(), ref_vel, atol=1e-3)
    assert_allclose(het_obj.get_mass(), torch.cat([ref_obj.get_mass(envs_idx=0) for ref_obj in ref_objs]), tol=tol)

    # The arms stop against each other instead of swinging through each other. FIXME: pytorch#TBD - 'any' over an
    # empty dimension returns uninitialized memory on MPS, so the contacts are reduced on the host.
    assert tensor_to_array(het_dual_arm.get_contacts(with_entity=het_dual_arm)["valid_mask"]).any(axis=-1).all()

    # Each cube rests on the torso of the dual arm variant its environment carries, as its reference does
    ref_cube_pos = torch.cat([ref_cube.get_pos(envs_idx=[i_env]) for i_env, ref_cube in enumerate(ref_cubes)])
    assert_allclose(ref_cube_pos - het_cube.get_pos(), REFERENCE_OFFSETS, tol=tol)
    torso_top = het_dual_arm.get_link("torso").get_AABB()[:, 1, 2]
    assert_allclose(het_cube.get_pos()[:, 2] - torso_top, 0.5 * np.array(CUBE_SIZES), tol=2e-4)

    # The variants are genuinely distinct: their masses are not all equal.
    with pytest.raises(AssertionError):
        assert_allclose(het_obj.get_mass(), het_obj.get_mass()[0], tol=tol)


@pytest.mark.required
def test_variant_inertia_matches_standalone(undefined_inertia, implicit_inertial_origin, tol):
    # A variant must resolve whatever its asset leaves unspecified exactly as the same asset loaded on its own. The
    # second URDF omits its inertial origin, which resolves to the link frame and keeps the authored mass and inertia
    # tensor, while the first has no inertial element at all and falls back to its geometry. These cannot be folded
    # into 'test_physics_parity': its variants come from MJCF, whose root joint carries a format-determined name that
    # no URDF variant can match.
    scene = gs.Scene(show_viewer=False)
    files = (undefined_inertia, implicit_inertial_origin)
    ref_objs = [
        scene.add_entity(
            gs.morphs.URDF(
                file=file,
                pos=(0.0, 0.5 * i_file, 0.1),
            ),
        )
        for i_file, file in enumerate(files)
    ]
    het_obj = scene.add_entity(
        morph=tuple(gs.morphs.URDF(file=file, pos=(0.5, 0.5 * i_file, 0.1)) for i_file, file in enumerate(files)),
    )
    scene.build(n_envs=len(files))

    assert_allclose(het_obj.get_mass(), torch.cat([ref_obj.get_mass(envs_idx=0) for ref_obj in ref_objs]), tol=tol)


@pytest.mark.required
def test_fewer_envs_than_variants():
    # With n_envs < n_variants, environment i gets variant i and the variants beyond n_envs stay unused.
    scene = gs.Scene(show_viewer=False)
    scene.add_entity(gs.morphs.Plane())

    # 4 variants with different positions but only 2 environments
    morphs_heterogeneous = [
        gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=(0.0, 0.0, 0.1)),
        gs.morphs.Box(size=(0.03, 0.03, 0.03), pos=(0.1, 0.0, 0.15)),
        gs.morphs.Box(size=(0.02, 0.02, 0.02), pos=(0.2, 0.0, 0.2)),
        gs.morphs.Sphere(radius=0.02, pos=(0.3, 0.0, 0.25)),
    ]
    het_obj = scene.add_entity(
        morph=morphs_heterogeneous,
    )

    # Building with only 2 environments should work - each env gets a unique variant
    scene.build(n_envs=2)

    # Verify mass - env 0 gets variant 0 (0.04 box), env 1 gets variant 1 (0.03 box)
    mass = het_obj.get_mass()
    assert mass.shape == (scene.n_envs,)
    # Different box sizes should have different masses
    assert mass[0] != mass[1]


@pytest.mark.slow  # ~200s
@pytest.mark.required
def test_aabb(undefined_inertia, show_viewer, tol):
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.15, -0.2, 0.17),
            camera_lookat=(0.045, 0.0, 0.12),
        ),
        show_viewer=show_viewer,
    )

    # Box and sphere with different sizes and positions
    morphs_heterogeneous = (
        gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=(0.0, 0.0, 0.1)),
        gs.morphs.Sphere(radius=0.01, pos=(0.1, 0.0, 0.15)),
    )
    het_obj = scene.add_entity(
        morph=morphs_heterogeneous,
    )
    # Fixed boxes of different sizes and poses, the second one turned by 45 degrees about the vertical, holding a
    # single copy of their vertices for every environment
    FIXED_POSITIONS = ((0.05, 0.08, 0.12), (0.05, 0.12, 0.1))
    het_fixed = scene.add_entity(
        morph=(
            gs.morphs.Box(size=(0.02, 0.02, 0.02), pos=FIXED_POSITIONS[0], fixed=True, batch_fixed_verts=False),
            gs.morphs.Box(
                size=(0.03, 0.03, 0.03),
                pos=FIXED_POSITIONS[1],
                euler=(0.0, 0.0, 45.0),
                fixed=True,
                batch_fixed_verts=False,
            ),
        ),
    )
    # Fixed spheres from a file, which drops their joint, at two scales
    FIXED_URDF_POS = (0.05, 0.4, 0.1)
    het_fixed_urdf = scene.add_entity(
        morph=tuple(
            gs.morphs.URDF(file=undefined_inertia, pos=FIXED_URDF_POS, scale=scale, fixed=True) for scale in (1.0, 2.0)
        ),
    )
    # A camera rendering a sphere environment follows the entity from before the build
    CAMERA_POS = (0.3, -0.4, 0.3)
    camera = scene.add_camera(
        pos=CAMERA_POS,
        lookat=(0.1, 0.0, 0.15),
        env_idx=2,
    )
    camera.follow_entity(het_obj)
    # 4 envs: envs 0-1 get box, envs 2-3 get sphere
    scene.build(n_envs=4)

    # Scaling a variant scales its whole geometry about the fixed base
    urdf_aabb = tensor_to_array(het_fixed_urdf.get_AABB()) - np.array(FIXED_URDF_POS)
    assert_allclose(urdf_aabb[[2, 3]], 2.0 * urdf_aabb[[0, 1]], tol=tol)

    # The camera keeps its pose, the offset it follows the entity at being measured from the sphere it renders. It
    # reports its pose in single precision.
    camera.update_following()
    assert_allclose(camera.pos, CAMERA_POS, tol=1e-6)

    # Per-variant morph.pos should be correctly applied
    pos = het_obj.get_pos()
    assert_allclose(pos[[0, 1]], (0.0, 0.0, 0.1), tol=tol)
    assert_allclose(pos[[2, 3]], (0.1, 0.0, 0.15), tol=tol)

    # The AABB of a fixed heterogeneous entity bounds the box each environment carries
    envs_center = np.repeat(FIXED_POSITIONS, 2, axis=0)
    envs_half_extent = np.repeat([(0.01, 0.01, 0.01), (0.015 * np.sqrt(2.0), 0.015 * np.sqrt(2.0), 0.015)], 2, axis=0)
    fixed_aabb = np.stack((envs_center - envs_half_extent, envs_center + envs_half_extent), axis=-2)
    assert_allclose(het_fixed.get_AABB(), fixed_aabb, tol=tol)
    assert_allclose(het_fixed.base_link.get_AABB(), fixed_aabb, tol=tol)

    # get_AABB should return correct shapes
    aabb = het_obj.get_AABB()
    assert aabb.shape == (scene.n_envs, 2, 3)  # (n_envs, min/max, xyz)
    for i in range(scene.n_envs):
        assert_allclose(aabb[i], het_obj.get_AABB(i), tol=gs.EPS)

    # Box envs should have same AABB, sphere envs should have same AABB
    assert_allclose(aabb[0], aabb[1], tol=gs.EPS)
    assert_allclose(aabb[2], aabb[3], tol=gs.EPS)

    # Box and sphere should have different AABBs (different sizes)
    with pytest.raises(AssertionError):
        assert_allclose(aabb[0], aabb[2], tol=1e-3)

    # get_vAABB should also work
    vaabb = het_obj.get_vAABB()
    assert vaabb.shape == (scene.n_envs, 2, 3)  # (n_envs, min/max, xyz) - same as AABB
    for i in range(scene.n_envs):
        assert_allclose(vaabb[i], het_obj.get_vAABB(i), tol=gs.EPS)

    # vAABB should have same structure as AABB (box envs same, sphere envs same)
    assert_allclose(vaabb[0], vaabb[1], tol=gs.EPS)
    assert_allclose(vaabb[2], vaabb[3], tol=gs.EPS)
    with pytest.raises(AssertionError):
        assert_allclose(vaabb[0], vaabb[2], tol=1e-3)

    # AABB and vAABB sizes should be approximately equal for each environment
    aabb_size_box = aabb[0, 1] - aabb[0, 0]
    vaabb_size_box = vaabb[0, 1] - vaabb[0, 0]
    assert_allclose(aabb_size_box, vaabb_size_box, tol=tol)

    aabb_size_sphere = aabb[2, 1] - aabb[2, 0]
    vaabb_size_sphere = vaabb[2, 1] - vaabb[2, 0]
    assert_allclose(aabb_size_sphere, vaabb_size_sphere, tol=1e-3)  # Allow small tolerance for decimation

    # The AABBs maintained by the collision detection only bound the variant each environment carries
    scene.step()
    assert_allclose(het_obj.get_AABB(allow_fast_approx=True), het_obj.get_AABB(), tol=1e-3)


@pytest.mark.parametrize("backend", [gs.gpu])  # Grasping physics requires GPU
def test_pick_heterogenous_objects(show_viewer):
    scene = gs.Scene(show_viewer=show_viewer)
    scene.add_entity(gs.morphs.Plane())
    franka = scene.add_entity(
        morph=gs.morphs.MJCF(
            file="xml/franka_emika_panda/panda.xml",
        )
    )

    # 4 geometry variants: env i -> variant i
    # Sizes: box0=0.04, box1=0.02, sphere0=0.03, sphere1=0.025 (radius for spheres)
    # Note: spheres need larger radius to be reliably grasped by the Franka gripper
    sizes = [0.04, 0.02, 0.03, 0.025]  # box0, box1, sphere0, sphere1
    het_obj = scene.add_entity(
        morph=[
            gs.morphs.Box(size=(sizes[0],) * 3, pos=(0.65, 0.0, 0.02)),
            gs.morphs.Box(size=(sizes[1],) * 3, pos=(0.65, 0.0, 0.02)),
            gs.morphs.Sphere(radius=sizes[2], pos=(0.65, 0.0, 0.02)),
            gs.morphs.Sphere(radius=sizes[3], pos=(0.65, 0.0, 0.02)),
        ]
    )
    scene.build(n_envs=4, env_spacing=(1, 1))

    # Expected CoM z at rest: half-height for boxes, radius for spheres
    expected_com_z = np.array([sizes[0] / 2, sizes[1] / 2, sizes[2], sizes[3]])

    motors_dof = np.arange(7)
    fingers_dof = np.arange(7, 9)
    init_qpos = np.array([[-1.0124, 1.5559, 1.3662, -1.6878, -1.5799, 1.7757, 1.4602, 0.04, 0.04]] * 4)

    # Set PD gains for finger joints (MJCF raw params have negligible control gain)
    franka.set_dofs_kp([100.0, 100.0], fingers_dof)
    franka.set_dofs_kv([10.0, 10.0], fingers_dof)

    # Initialize robot position
    franka.set_qpos(init_qpos)
    scene.step()

    # Test 1: CoM at rest matches expected heights based on shape
    # Control robot to hold position while objects settle
    for _ in range(30):
        franka.control_dofs_position(init_qpos[:, :7], motors_dof)
        franka.control_dofs_position(init_qpos[:, 7:9], fingers_dof)
        scene.step()
    assert_allclose(het_obj.get_pos()[:, 2], expected_com_z, tol=0.005)

    # Move to grasp position
    end_effector = franka.get_link("hand")
    qpos_grasp = franka.inverse_kinematics(
        link=end_effector,
        pos=np.array([[0.65, 0.0, 0.135]] * scene.n_envs),
        quat=np.array([[0, 1, 0, 0]] * scene.n_envs),
    )

    # Hold - approach with gripper open
    for _ in range(50):
        franka.control_dofs_position(qpos_grasp[:, :7], motors_dof)
        franka.control_dofs_position(np.array([[0.04, 0.04]] * scene.n_envs), fingers_dof)
        scene.step()

    # Grasp - close gripper
    for _ in range(50):
        franka.control_dofs_position(qpos_grasp[:, :7], motors_dof)
        franka.control_dofs_position(np.array([[0.0, 0.0]] * scene.n_envs), fingers_dof)
        scene.step()

    # Test 2: Gripper width matches object size (box width or sphere diameter)
    gripper_qpos = franka.get_qpos()[:, 7:9]
    gripper_widths = (gripper_qpos[:, 0] + gripper_qpos[:, 1]).cpu().numpy()
    expected_grip_widths = np.array([sizes[0], sizes[1], 2 * sizes[2], 2 * sizes[3]])  # box size or sphere diameter
    assert_allclose(gripper_widths, expected_grip_widths, tol=0.005)

    # Record positions before lifting
    pre_lift_z = het_obj.get_pos()[:, 2]

    # Lift
    qpos_lift = franka.inverse_kinematics(
        link=end_effector,
        pos=np.array([[0.65, 0.0, 0.3]] * scene.n_envs),
        quat=np.array([[0, 1, 0, 0]] * scene.n_envs),
    )
    for _ in range(50):
        franka.control_dofs_position(qpos_lift[:, :7], motors_dof)
        franka.control_dofs_position(np.array([[0.0, 0.0]] * scene.n_envs), fingers_dof)
        scene.step()

    # Test 3: All 4 objects were lifted
    post_lift_z = het_obj.get_pos()[:, 2]
    lift_deltas = tensor_to_array(post_lift_z - pre_lift_z)
    assert np.all(lift_deltas > 0.05), f"All objects should be lifted (deltas={lift_deltas})"


@pytest.mark.required
def test_invalid_variant_specification_raises():
    # Restricting variants to a single family is what keeps the inertial parsed per variant aligned with the variant
    # index: only URDF and MJCF variants record one, so admitting a basic object among them would shift every
    # subsequent lookup onto the wrong variant.
    MJCF_SPHERE = (
        '<mujoco><worldbody><body><joint type="free"/><geom type="sphere" size="0.1"/></body></worldbody></mujoco>'
    )

    scene = gs.Scene(show_viewer=False)

    morphs_heterogeneous = (
        gs.morphs.Box(size=(1.0, 1.0, 1.0)),
        gs.morphs.Box(size=(1.0, 1.0, 1.0)),
    )

    # PBD material should raise an exception
    with pytest.raises(gs.GenesisException, match="created from one morph"):
        scene.add_entity(
            morph=morphs_heterogeneous,
            material=gs.materials.PBD.Cloth(),
        )

    # A morph family that cannot be a variant at all
    with pytest.raises(gs.GenesisException, match="only support Primitive, Mesh, URDF and MJCF"):
        scene.add_entity(
            morph=(gs.morphs.Box(size=(1.0, 1.0, 1.0)), gs.morphs.Terrain()),
        )

    # A basic object and an articulated robot in the same entity
    with pytest.raises(gs.GenesisException, match="must be consistent"):
        scene.add_entity(
            morph=(gs.morphs.Box(size=(1.0, 1.0, 1.0)), gs.morphs.MJCF(file=MJCF_SPHERE)),
        )


@pytest.mark.required
def test_morph_property_raises():
    scene = gs.Scene(show_viewer=False)

    single_morph = gs.morphs.Box(size=(0.1, 0.1, 0.1))
    single_obj = scene.add_entity(
        morph=single_morph,
    )

    rigid_morphs_heterogeneous = (
        gs.morphs.Box(size=(0.1, 0.1, 0.1)),
        gs.morphs.Cylinder(radius=0.05, height=0.2),
    )
    rigid_obj = scene.add_entity(
        morph=rigid_morphs_heterogeneous,
    )
    kinematic_morphs_heterogeneous = (
        gs.morphs.Box(size=(0.2, 0.2, 0.2)),
        gs.morphs.Sphere(radius=0.1),
    )
    kinematic_obj = scene.add_entity(
        morph=kinematic_morphs_heterogeneous,
        material=gs.materials.Kinematic(),
    )
    tool = scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.05, 0.05, 0.05),
        ),
    )

    assert single_obj.morph is single_morph
    assert rigid_obj.main_morph is rigid_morphs_heterogeneous[0]
    assert list(rigid_obj.morphs) == list(rigid_morphs_heterogeneous)
    with pytest.raises(gs.GenesisException, match=r"Heterogeneous.*\.morphs") as exc_info:
        _ = rigid_obj.morph
    assert ".main_morph" in str(exc_info.value)

    assert kinematic_obj.main_morph is kinematic_morphs_heterogeneous[0]
    assert list(kinematic_obj.morphs) == list(kinematic_morphs_heterogeneous)
    with pytest.raises(gs.GenesisException, match=r"Heterogeneous.*\.morphs"):
        _ = kinematic_obj.morph

    # Attaching is unsupported both from a heterogeneous entity (as the reparented child) and onto one (as parent).
    with pytest.raises(gs.GenesisException, match="[Hh]eterogeneous"):
        rigid_obj.attach(single_obj)
    with pytest.raises(gs.GenesisException, match="[Hh]eterogeneous"):
        tool.attach(rigid_obj)


@pytest.mark.required
def test_root_joint_mismatch(fixed_base_dual_arm):
    scene = gs.Scene(show_viewer=False)

    # A fixed base and a free base decide differently which links the world carries
    with pytest.raises(gs.GenesisException, match="root link"):
        scene.add_entity(
            morph=[
                gs.morphs.URDF(file=fixed_base_dual_arm, fixed=True),
                gs.morphs.URDF(file=fixed_base_dual_arm, fixed=False),
            ]
        )

import numpy as np
import pytest
import torch

import genesis as gs
import genesis.utils.geom as gu
from genesis.utils.misc import tensor_to_array

from ..utils.assertions import assert_allclose
from ..utils.assets import get_hf_dataset


@pytest.mark.required
def test_physics_parity(show_viewer, tol):
    # Uses the fixed-child mesh objects from 'test_convexify' (offset center of mass, distinct mass) so the per-env
    # parity check exercises the inertia alignment, not just trivially-symmetric primitives.
    N_STEPS = 100
    DROP_HEIGHT = 0.2
    VARIANTS = (("mug_1", "output.xml"), ("donut_0", "output.xml"), ("cup_2", "model.xml"), ("apple_15", "model.xml"))
    # Divergent per-variant yaw offset, stripped to identity by the relative getter and carried by the world frame.
    # Applied identically to the homogeneous reference so the dynamics still match.
    OFFSET_EULERS = ((0.0, 0.0, 30.0), (0.0, 0.0, -45.0), (0.0, 0.0, 90.0), (0.0, 0.0, -120.0))
    # Distinct per-variant placement, dispatched per environment.
    POSITIONS = ((0.0, 0.0, DROP_HEIGHT), (0.2, 0.0, DROP_HEIGHT), (0.0, 0.2, DROP_HEIGHT), (0.2, 0.2, DROP_HEIGHT))
    # The homogeneous references live in the same scene, offset far enough that no entity ever interacts with
    # another: a single build compiles one kernel set instead of one per scene.
    REFERENCE_OFFSETS = ((10.0, 0.0, 0.0), (20.0, 0.0, 0.0), (30.0, 0.0, 0.0), (40.0, 0.0, 0.0))

    asset_files = tuple(f"{get_hf_dataset(pattern=f'{name}/*')}/{name}/{xml}" for name, xml in VARIANTS)

    # One homogeneous reference entity per variant plus a single heterogeneous entity dispatching one variant per
    # environment, all in one scene.
    scene = gs.Scene(show_viewer=show_viewer)
    scene.add_entity(gs.morphs.Plane())
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
            gs.morphs.MJCF(
                file=file,
                pos=pos,
                offset_euler=offset_euler,
            )
            for file, pos, offset_euler in zip(asset_files, POSITIONS, OFFSET_EULERS)
        )
    )
    scene.build(n_envs=len(VARIANTS))

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

    for _ in range(N_STEPS):
        scene.step()

    # After the drop each environment matches the homogeneous reference of its variant in pose, velocity and mass.
    ref_pos = torch.cat([ref_obj.get_pos(envs_idx=[i_env]) for i_env, ref_obj in enumerate(ref_objs)])
    ref_vel = torch.cat([ref_obj.get_vel(envs_idx=[i_env]) for i_env, ref_obj in enumerate(ref_objs)])
    # Both are held loosely, and the rate more so than the pose: sharing one batch with the other variants puts the
    # heterogeneous entity through a different arithmetic than its own reference.
    assert_allclose(ref_pos - het_obj.get_pos(), REFERENCE_OFFSETS, tol=1e-5)
    assert_allclose(het_obj.get_vel(), ref_vel, tol=2e-4)
    assert_allclose(het_obj.get_mass(), torch.cat([ref_obj.get_mass(envs_idx=0) for ref_obj in ref_objs]), tol=tol)

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
        morph=tuple(
            gs.morphs.URDF(
                file=file,
                pos=(0.5, 0.5 * i_file, 0.1),
            )
            for i_file, file in enumerate(files)
        ),
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
def test_aabb(tol):
    scene = gs.Scene(show_viewer=False)
    scene.add_entity(gs.morphs.Plane())

    # Box and sphere with different sizes and positions
    morphs_heterogeneous = (
        gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=(0.0, 0.0, 0.1)),
        gs.morphs.Sphere(radius=0.01, pos=(0.1, 0.0, 0.15)),
    )
    het_obj = scene.add_entity(
        morph=morphs_heterogeneous,
    )
    # 4 envs: envs 0-1 get box, envs 2-3 get sphere
    scene.build(n_envs=4)

    # Per-variant morph.pos should be correctly applied
    pos = het_obj.get_pos()
    assert_allclose(pos[[0, 1]], (0.0, 0.0, 0.1), tol=tol)
    assert_allclose(pos[[2, 3]], (0.1, 0.0, 0.15), tol=tol)

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


# 30s
@pytest.mark.slow  # ~250s
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


@pytest.mark.slow  # ~200s
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
def test_articulated_structure_mismatch():
    scene = gs.Scene(show_viewer=False)
    scene.add_entity(gs.morphs.Plane())

    # two_cube_revolute has 1 revolute joint; two_link_arm has 2 continuous joints
    with pytest.raises(gs.GenesisException):
        scene.add_entity(
            morph=[
                gs.morphs.URDF(file="urdf/simple/two_cube_revolute.urdf", pos=(0, 0, 0.1)),
                gs.morphs.URDF(file="urdf/simple/two_link_arm.urdf", pos=(0, 0, 0.1)),
            ]
        )

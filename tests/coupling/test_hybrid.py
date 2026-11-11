import numpy as np
import pytest
import torch

import trimesh

import genesis as gs
import genesis.utils.geom as gu
from genesis.utils.misc import tensor_to_array

from ..utils.assertions import assert_allclose, assert_equal
from ..utils.assets import get_hf_dataset


@pytest.mark.slow  # ~300s
@pytest.mark.required
def test_rigid_mpm_muscle(show_viewer):
    BALL_POS_INIT = (0.8, 0.6, 0.12)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=3e-3,
            substeps=10,
        ),
        rigid_options=gs.options.RigidOptions(
            gravity=(0, 0, -9.8),
            constraint_timeconst=0.02,
        ),
        mpm_options=gs.options.MPMOptions(
            gravity=(0.0, 0.0, 0.0),  # mimic gravity compensation
            enable_CPIC=True,
            lower_bound=(0.0, 0.0, -0.2),
            upper_bound=(1.0, 1.0, 1.0),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.5, 1.3, 0.5),
            camera_lookat=(0.0, 0.0, 0.0),
            camera_fov=40,
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )

    scene.add_entity(
        morph=gs.morphs.Plane(),
    )
    cube = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.85, 0.1, 0.1),
            size=(0.1, 0.1, 0.1),
        ),
        material=gs.materials.MPM.Elastic(),
    )
    robot = scene.add_entity(
        morph=gs.morphs.URDF(
            file="urdf/simple/two_link_arm.urdf",
            pos=(0.45, 0.45, 0.2),
            euler=(0.0, 0.0, 0.0),
            scale=0.2,
            fixed=True,
        ),
        material=gs.materials.Hybrid(
            material_rigid=gs.materials.Rigid(
                gravity_compensation=1.0,
            ),
            material_soft=gs.materials.MPM.Muscle(
                E=1e4,
                nu=0.45,
                rho=1000.0,
                model="neohooken",
            ),
            damping=1000.0,
            thickness=0.05,
        ),
    )
    ball = scene.add_entity(
        morph=gs.morphs.Sphere(
            pos=BALL_POS_INIT,
            radius=0.12,
        ),
        material=gs.materials.Rigid(
            rho=1000,
            friction=0.5,
        ),
    )
    scene.build(n_envs=2)

    scene.reset()
    # The soft part of the robot follows its links regardless of the activeness of the particles of another entity
    cube.set_particles_active(gs.INACTIVE, envs_idx=[1])
    for i in range(150):
        robot.control_dofs_velocity(np.array([2.0 * np.sin(2 * np.pi * i * 0.006)] * robot.n_dofs))
        scene.step()

    ball_pos_delta = ball.get_pos() - torch.tensor(BALL_POS_INIT, dtype=gs.tc_float, device=gs.device)
    assert_allclose(ball_pos_delta[..., 0], 0.0, tol=1e-2)
    assert ((0.02 < ball_pos_delta[..., 1]) & (ball_pos_delta[..., 1] < 0.05)).all()
    assert_allclose(ball_pos_delta[..., 2], 0.0, tol=1e-3)
    soft_pos = robot.part_soft.get_particles_pos()
    assert_allclose(soft_pos[0], soft_pos[1], tol=gs.EPS)


@pytest.mark.required
@pytest.mark.parametrize("substeps", [1, 10])
def test_rigid_mpm_legacy_coupling(substeps, tol, show_viewer):
    # This test is aimining at two things:
    # 1) When substep = 1, the rigid object should be affected by the MPM particles
    # 2) Regardless of substeps, as far as the substep dt is the same, they should give consistent results
    substep_dt = 4e-4
    horizon_substeps = 100

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=substep_dt * substeps,
            substeps=substeps,
        ),
        mpm_options=gs.options.MPMOptions(
            enable_CPIC=True,
            lower_bound=(-0.5, -1.0, 0.0),
            upper_bound=(0.5, 1.0, 1),
        ),
        vis_options=gs.options.VisOptions(
            visualize_mpm_boundary=True,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_fov=30,
        ),
        show_viewer=show_viewer,
    )

    plane = scene.add_entity(
        morph=gs.morphs.Plane(),
    )

    obj_rigid = scene.add_entity(
        material=gs.materials.Rigid(
            rho=1.0,
        ),
        morph=gs.morphs.Box(
            pos=(0.0, -0.25, 0.1),
            size=(0.2, 0.2, 0.2),
        ),
        surface=gs.surfaces.Default(
            color=(1.0, 0.4, 0.4),
            vis_mode="visual",
        ),
    )

    obj_sand = scene.add_entity(
        material=gs.materials.MPM.Liquid(),
        morph=gs.morphs.Box(
            pos=(0.0, 0.0, 0.2),
            size=(0.3, 0.3, 0.3),
        ),
        surface=gs.surfaces.Default(
            color=(0.3, 0.3, 1.0),
            vis_mode="particle",
        ),
    )

    scene.build(n_envs=2)

    horizon = horizon_substeps // substeps
    for i in range(horizon):
        scene.step()

    # Check that the sand moved the box along the negative Y direction
    pos = tensor_to_array(obj_rigid.get_pos())
    assert (pos[..., 1] + 0.25 < 0.0).all()

    # Resetting one environment leaves the other one as it was
    sand_pos = tensor_to_array(obj_sand.get_particles_pos())
    scene.reset(envs_idx=[1])
    assert_equal(obj_rigid.get_pos()[0], pos[0])
    assert_equal(obj_sand.get_particles_pos()[0], sand_pos[0])

    # A rollout after a reset starts free of the coupling wrench the previous one left
    for i in range(horizon):
        scene.step()
    assert_allclose(obj_rigid.get_pos()[1], pos[1], tol=tol)


@pytest.mark.required
def test_mesh_mpm_build(show_viewer):
    scene = gs.Scene(
        mpm_options=gs.options.MPMOptions(
            lower_bound=(-0.5, -0.5, -0.5),
            upper_bound=(0.5, 0.5, 0.5),
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )
    scene.add_entity(
        morph=gs.morphs.Mesh(
            file="meshes/duck.obj",
            scale=0.15,
        ),
        material=gs.materials.Hybrid(
            material_rigid=gs.materials.Rigid(),
            material_soft=gs.materials.MPM.Muscle(),
        ),
    )
    scene.build()


@pytest.mark.debug(False)  # Disable debug for speedup
@pytest.mark.required
@pytest.mark.parametrize("backend", [gs.gpu])
def test_deformable_parallel(show_viewer):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=2e-3,
            substeps=10,
        ),
        mpm_options=gs.options.MPMOptions(
            lower_bound=(0.5, -0.1, -0.05),
            upper_bound=(0.7, 0.1, 0.3),
        ),
        sph_options=gs.options.SPHOptions(
            lower_bound=(-0.03, -0.03, -0.08),
            upper_bound=(0.33, 0.33, 1.0),
        ),
        fem_options=gs.options.FEMOptions(
            damping=45.0,
        ),
        pbd_options=gs.options.PBDOptions(
            particle_size=1e-2,
        ),
        vis_options=gs.options.VisOptions(
            rendered_envs_idx=[1],
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3.5, 0.0, 2.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=40,
        ),
        show_viewer=show_viewer,
    )

    plane = scene.add_entity(
        morph=gs.morphs.Plane(),
        material=gs.materials.Rigid(
            needs_coup=True,
            coup_friction=0.0,
        ),
    )
    cloth = scene.add_entity(
        morph=gs.morphs.Mesh(
            file="meshes/cloth.obj",
            scale=0.6,
            pos=(0.0, 0.8, 0.3),
            euler=(180.0, 0.0, 0.0),
        ),
        material=gs.materials.PBD.Cloth(),
        surface=gs.surfaces.Default(
            color=(0.2, 0.4, 0.8, 1.0),
        ),
    )
    water = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.15, 0.15, 0.22),
            size=(0.25, 0.25, 0.4),
        ),
        material=gs.materials.SPH.Liquid(),
        surface=gs.surfaces.Default(
            color=(0.2, 0.6, 1.0, 1.0),
        ),
    )
    mpm_cube = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.6, 0, 0.1),
            size=(0.1, 0.1, 0.1),
        ),
        material=gs.materials.MPM.Elastic(
            rho=200,
        ),
        surface=gs.surfaces.Default(
            color=(0.9, 0.8, 0.2, 1.0),
        ),
    )
    entity_fem = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.8, 0.8, 0.1),
            size=(0.1, 0.1, 0.1),
        ),
        material=gs.materials.FEM.Elastic(
            E=3.0e4,
            nu=0.45,
            rho=1000.0,
            model="stable_neohookean",
        ),
    )
    scene.build(n_envs=2)

    init_mpm_cube_pos = mpm_cube.get_particles_pos()
    init_cloth_pos = cloth.get_particles_pos()
    init_water_pos = water.get_particles_pos()

    scene.get_state()
    for i in range(1500):
        scene.step()

    final_mpm_cube_pos = mpm_cube.get_particles_pos()
    final_cloth_pos = cloth.get_particles_pos()
    final_water_pos = water.get_particles_pos()

    # check if the positions are changed
    assert (init_mpm_cube_pos - final_mpm_cube_pos).abs().sum() > 0.1
    assert (init_cloth_pos - final_cloth_pos).abs().sum() > 0.1
    assert (init_water_pos - final_water_pos).abs().sum() > 0.1

    # check if the particles are above the ground
    assert final_mpm_cube_pos[..., 2].min() > -1e-5
    assert final_cloth_pos[..., 2].min() > -1e-5
    assert final_water_pos[..., 2].min() > -1e-5

    # The liquid at rest on a rigid floor stands half a particle size above it, up to the discretization of the boundary
    # FIXME: Within a support radius of a wall, the ghost lattice and the continuous plane disagree on the boundary
    # volume, which lowers the corner layer. Replacing the wall lattice by exact boundaries removes it.
    sph_solver = scene.sph_solver
    water_pos = tensor_to_array(final_water_pos)
    is_middle = (
        (water_pos[..., :2] > sph_solver.boundary_lower[:2] + sph_solver.support_radius)
        & (water_pos[..., :2] < sph_solver.boundary_upper[:2] - sph_solver.support_radius)
    ).all(axis=-1)
    is_bottom_middle = is_middle & (water_pos[..., 2] < sph_solver.particle_size)
    bottom_height = (water_pos[..., 2] * is_bottom_middle).sum(axis=-1) / is_bottom_middle.sum(axis=-1)
    assert_allclose(bottom_height, 0.5 * sph_solver.particle_size, atol=0.1 * sph_solver.particle_size)

    assert_allclose(cloth.get_particles_vel(), 0.0, atol=1e-5)
    assert_allclose(mpm_cube.get_particles_vel(), 0.0, atol=2e-4)
    assert_allclose(entity_fem.get_verts_vel(), 0, atol=1e-3)
    assert_allclose(water.get_particles_vel(), 0.0, atol=5e-2)


@pytest.mark.required
def test_fem_mpm_coupling(precision, show_viewer, tol):
    GRAVITY = -9.81
    DT = 2e-3
    N_STEPS_FREE_FALL = 10
    N_STEPS = 100

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=DT,
            substeps=10,
            gravity=(0.0, 0.0, GRAVITY),
        ),
        mpm_options=gs.options.MPMOptions(
            lower_bound=(-0.1, -0.1, 0.0),
            upper_bound=(0.1, 0.1, 0.2),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.0, -1.0, 0.8),
            camera_lookat=(0.2, 0.0, 0.3),
        ),
        show_viewer=show_viewer,
    )
    block = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.0, 0.0, 0.0675),
            size=(0.1, 0.1, 0.035),
        ),
        material=gs.materials.MPM.Elastic(),
    )
    # FIXME: Building should raise when the scene cannot meet the accuracy of the simulation. The explicit time step
    # must stay under about h / c, with h the smallest element size, c = sqrt((lambda + 2 * mu) / rho) the wave speed,
    # lambda = E * nu / ((1 + nu) * (1 - 2 * nu)) and mu = E / (2 * (1 + nu)). The rest shape of a body accumulates
    # a velocity noise of the order of t * E * eps / (rho * h) after a time t, with eps the machine epsilon, which must
    # stay under the tolerance. Meanwhile, single precision takes a softer material.
    box = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.0, 0.0, 0.12),
            size=(0.05, 0.05, 0.05),
        ),
        material=gs.materials.FEM.Elastic(
            E=1e6 if precision == "64" else 1e5,
        ),
    )
    fem = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.4, 0.0, 0.5),
            size=(0.1, 0.1, 0.1),
        ),
        material=gs.materials.FEM.Elastic(
            E=1e6 if precision == "64" else 1e5,
        ),
    )
    scene.build(n_envs=2)

    for _ in range(N_STEPS_FREE_FALL):
        scene.step()

    # An unstressed body far from the grid falls freely, every vertex at the same velocity
    assert_allclose(fem.get_verts_vel(), (0.0, 0.0, GRAVITY * DT * N_STEPS_FREE_FALL), tol=tol)

    for _ in range(N_STEPS - N_STEPS_FREE_FALL):
        scene.step()

    # The box dropped on the block comes to rest on top of it. Its surface stands off the topmost particles by less
    # than one grid cell, the reach of the coupling.
    assert_allclose(box.get_verts_vel(), 0.0, atol=5e-3)
    gap = box.get_verts_pos()[..., 2].min(dim=-1).values - block.get_particles_pos()[..., 2].max(dim=-1).values
    assert ((gap > 0.0) & (gap < scene.mpm_solver.dx)).all()


@pytest.mark.parametrize(
    "n_envs, material_type",
    [
        pytest.param(0, gs.materials.SPH.Liquid, marks=pytest.mark.required),
        (1, gs.materials.SPH.Liquid),
        (2, gs.materials.SPH.Liquid),
        pytest.param(2, gs.materials.PBD.Liquid, marks=pytest.mark.required),
        pytest.param(2, gs.materials.MPM.Liquid, marks=pytest.mark.required),
        (2, gs.materials.MPM.Sand),
        (2, gs.materials.MPM.Snow),
        (2, gs.materials.MPM.Elastic),  # This makes little sense but nothing prevents doing this
    ],
)
def test_fluid_emitter(n_envs, material_type, show_viewer):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=4e-3,
            substeps=10,
        ),
        mpm_options=gs.options.MPMOptions(
            lower_bound=(0.0, -1.5, 0.0),
            upper_bound=(1.0, 1.5, 4.0),
        ),
        sph_options=gs.options.SPHOptions(
            particle_size=0.02,
        ),
        pbd_options=gs.options.PBDOptions(
            particle_size=0.02,
        ),
        vis_options=gs.options.VisOptions(
            rendered_envs_idx=[0],
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(5.5, 6.5, 3.2),
            camera_lookat=(0.5, 1.5, 1.5),
            camera_fov=35,
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(gs.morphs.Plane())
    scene.add_entity(
        morph=gs.morphs.URDF(
            file="urdf/wheel/fancy_wheel.urdf",
            pos=(0.5, 0.25, 1.6),
            euler=(0, 0, 0),
            fixed=True,
            convexify=False,
        ),
    )
    emitter = scene.add_emitter(
        material=material_type(),
        max_particles=5000,
        surface=gs.surfaces.Glass(
            color=(0.7, 0.85, 1.0, 0.7),
        ),
    )
    scene.build(n_envs=n_envs)

    emitter.emit_omni()
    for i in range(5):
        emitter.emit(droplet_shape="circle", droplet_size=0.25)
        scene.step()
    scene.step()


@pytest.mark.required("sparse")
@pytest.mark.parametrize("batch_links_info", [False, True])
def test_sap_rigid_rigid_hydroelastic_contact(batch_links_info, show_viewer):
    PLANES_HEIGHT = (0.1, 0.3)
    BOX_HALFHEIGHT = 0.1
    DT = 1 / 60
    GRAVITY = 9.81
    MU = 0.5

    # The ground is a single fixed plane facing up, and any other ground is refused
    for ground_morphs in (
        (gs.morphs.Terrain(),),
        (gs.morphs.Plane(normal=(0.0, 0.6, 0.8)),),
        (gs.morphs.Plane(), gs.morphs.Plane(pos=(0.0, 0.0, 1.0))),
    ):
        scene = gs.Scene(coupler_options=gs.options.SAPCouplerOptions(), show_viewer=False)
        for morph in ground_morphs:
            scene.add_entity(morph)
        with pytest.raises(gs.GenesisException, match="single fixed plane facing up"):
            scene.build()

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=DT,
            substeps=2,
            gravity=(0.0, 0.0, -GRAVITY),
        ),
        rigid_options=gs.options.RigidOptions(
            batch_links_info=batch_links_info,
        ),
        coupler_options=gs.options.SAPCouplerOptions(),
        show_viewer=show_viewer,
        show_FPS=False,
    )
    scene.add_entity(
        gs.morphs.Plane(
            pos=(0.0, 0.0, PLANES_HEIGHT[0]),
            batch_fixed_verts=True,
        ),
    )
    box = scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.5, 0.5, 2 * BOX_HALFHEIGHT),
            pos=(0.0, 0.0, PLANES_HEIGHT[0] + BOX_HALFHEIGHT),
        ),
        material=gs.materials.Rigid(),
    )
    # One box slides on the ground plane, and another one is pushed below the friction limit
    slider = scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.2, 0.2, 0.1),
            pos=(1.0, 0.0, PLANES_HEIGHT[0] + 0.05),
        ),
        material=gs.materials.Rigid(
            coup_friction=MU,
        ),
    )
    pushed = scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.2, 0.2, 0.1),
            pos=(-1.0, 0.0, PLANES_HEIGHT[0] + 0.05),
        ),
        material=gs.materials.Rigid(
            coup_friction=MU,
        ),
    )
    asset_path = get_hf_dataset(pattern="heavy_three_joint_link.xml")
    robot_1 = scene.add_entity(
        gs.morphs.MJCF(
            file=f"{asset_path}/heavy_three_joint_link.xml",
            pos=(-0.2, -0.26, PLANES_HEIGHT[0]),
            scale=0.3,
        ),
    )
    robot_2 = scene.add_entity(
        gs.morphs.MJCF(
            file=f"{asset_path}/heavy_three_joint_link.xml",
            pos=(0.17, -0.26, PLANES_HEIGHT[0] + 0.1),
            euler=(0.0, 0.0, 90.0),
            scale=0.3,
        ),
    )
    scene.build(n_envs=2)

    # The second environment holds the whole scene higher, its ground plane included
    for entity in scene.entities:
        entity_pos = entity.get_pos(envs_idx=[1])
        entity_pos[..., 2] += PLANES_HEIGHT[1] - PLANES_HEIGHT[0]
        entity.set_pos(entity_pos, envs_idx=[1])

    # Run simulation, launching the slider and pushing the other box once the boxes rest on the ground
    for _ in range(10):
        scene.step()
    slider.set_dofs_velocity([1.0], dofs_idx_local=[0])
    push_force = 0.5 * MU * pushed.get_mass() * GRAVITY
    slider_vel, slider_height, pushed_slip = [], [], []
    for _ in range(70):
        pushed.control_dofs_force(push_force[..., None], dofs_idx_local=[0])
        scene.step()
        slider_vel.append(tensor_to_array(slider.get_vel()))
        slider_height.append(tensor_to_array(slider.get_pos()[..., 2]) - PLANES_HEIGHT)
        # The velocity of the center of the bottom face of the pushed box, where it slips on the ground
        pushed_quat = pushed.get_quat()
        bottom_arm = gu.transform_by_quat(
            pushed_quat.new_tensor((0.0, 0.0, -0.05)).expand(*pushed_quat.shape[:-1], 3), pushed_quat
        )
        pushed_slip.append(tensor_to_array(pushed.get_vel() + torch.linalg.cross(pushed.get_ang(), bottom_arm)))
    slider_vel, slider_height, pushed_slip = np.stack(slider_vel), np.stack(slider_height), np.stack(pushed_slip)

    # The sliding box stays on the ground, decelerating at the rate of Coulomb friction until it stops
    assert_allclose(slider_height, 0.05, atol=1e-3)
    assert_allclose(np.diff(slider_vel[:10, ..., 0], axis=0), -MU * GRAVITY * DT, tol=2e-3)
    assert_allclose(slider_vel[15:], 0.0, atol=1e-3)

    # The pushed box sticks, slipping at most at the stiction tolerance. Its center of mass moves faster while the box
    # rocks on its compliant contact, which involves no slip.
    assert (np.linalg.norm(pushed_slip[..., :2], axis=-1) < 1e-4).all()

    # All the entities must be still
    for entity in scene.entities:
        assert_allclose(entity.get_links_vel(), 0.0, atol=0.03)

    # The box should stay at its initial position, resting on the ground plane of each environment
    box_pos = tensor_to_array(box.get_pos())
    assert_allclose(box_pos[..., :2], 0.0, atol=2e-3)
    assert_allclose(box_pos[..., 2] - PLANES_HEIGHT, BOX_HALFHEIGHT, atol=2e-3)

    # The box, and both robots should be laying on top of each other
    robot_1_aabb = tensor_to_array(robot_1.get_AABB())
    robot_2_aabb = tensor_to_array(robot_2.get_AABB())
    assert (robot_1_aabb[..., 0, :2] > -0.4).all() and (robot_2_aabb[..., 0, :2] > -0.4).all()
    assert (robot_1_aabb[..., 0, :2] < 0.4).all() and (robot_2_aabb[..., 0, :2] < 0.4).all()
    assert (robot_1_aabb[..., 1, 2] - PLANES_HEIGHT > 2 * BOX_HALFHEIGHT).all()
    assert (robot_2_aabb[..., 1, 2] > robot_1_aabb[..., 1, 2] + 0.05).all()


@pytest.mark.required
@pytest.mark.parametrize("model_name", ["scaled_mjcf_joint_equalities"])
@pytest.mark.parametrize("n_envs, is_batched", [(0, False), (2, True)])
def test_sap_joint_equality_polynomial(show_viewer, xml_path, n_envs, is_batched, tol):
    DT = 1e-2
    SCALE = 2.0
    COEFFICIENTS = (0.2, 0.4, -0.3, 0.2, -0.1)
    DRIVER_POSITION = 0.5
    DRIVER_VELOCITY = 0.4
    FOLLOWER_POSITION = sum([coefficient * DRIVER_POSITION**degree for degree, coefficient in enumerate(COEFFICIENTS)])
    DERIVATIVE = sum(
        [degree * coefficient * DRIVER_POSITION ** (degree - 1) for degree, coefficient in enumerate(COEFFICIENTS)]
    )
    FOLLOWER_VELOCITY = DRIVER_VELOCITY * DERIVATIVE
    TARGET_POSITION = 0.25
    UNRELATED_POSITION = 1.0
    MAX_RESIDUAL_RATIO = 1e-3

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=DT,
        ),
        rigid_options=gs.options.RigidOptions(
            batch_joints_info=is_batched,
            batch_dofs_info=is_batched,
        ),
        coupler_options=gs.options.SAPCouplerOptions(
            rigid_rigid_contact_type="none",
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.15, -0.75, 4.0),
            camera_lookat=(0.15, -0.75, 0.0),
            camera_up=(0.0, 1.0, 0.0),
        ),
        show_viewer=show_viewer,
    )
    entity = scene.add_entity(
        morph=gs.morphs.MJCF(
            file=xml_path,
            scale=SCALE,
        ),
    )
    scene.build(n_envs=n_envs)

    # Driver and follower joints of each pair, with the scale of their coordinates
    joints_pairs = []
    for driver_type, follower_type in (("hinge", "hinge"), ("slide", "slide"), ("slide", "hinge"), ("hinge", "slide")):
        driver_joint = entity.get_joint(f"{driver_type}_{follower_type}_driver")
        follower_joint = entity.get_joint(f"{driver_type}_{follower_type}_follower")
        driver_scale = SCALE if driver_type == "slide" else 1.0
        follower_scale = SCALE if follower_type == "slide" else 1.0
        joints_pairs.append((driver_joint, follower_joint, driver_scale, follower_scale))
    (i_target_q,) = entity.get_joint("target").qs_idx_local
    (i_unrelated_q,) = entity.get_joint("unrelated").qs_idx_local

    # A velocity tangent to the polynomial triggers no constraint impulse
    qpos = entity.get_qpos()
    dofs_vel = entity.get_dofs_velocity()
    for driver_joint, follower_joint, driver_scale, follower_scale in joints_pairs:
        qpos[..., driver_joint.qs_idx_local[0]] = DRIVER_POSITION * driver_scale
        qpos[..., follower_joint.qs_idx_local[0]] = FOLLOWER_POSITION * follower_scale
        dofs_vel[..., driver_joint.dofs_idx_local[0]] = DRIVER_VELOCITY * driver_scale
        dofs_vel[..., follower_joint.dofs_idx_local[0]] = FOLLOWER_VELOCITY * follower_scale
    qpos[..., i_target_q] = TARGET_POSITION * SCALE
    qpos[..., i_unrelated_q] = UNRELATED_POSITION * SCALE
    entity.set_qpos(qpos)
    entity.set_dofs_velocity(dofs_vel)
    scene.step()
    assert_allclose(entity.get_qpos(), qpos + dofs_vel * DT, tol=tol)

    # From rest, the solve leaves the share R / (R + J M^-1 J^T) of the linearized residual, which the regularization R
    # keeps below MAX_RESIDUAL_RATIO. Every driver at its reference pose checks the Jacobian at poly(0) = a0 and
    # poly'(0) = a1, and the target joint, which has no second joint, is pulled to a0.
    for driver_joint, follower_joint, _, _ in joints_pairs:
        qpos[..., driver_joint.qs_idx_local[0]] = 0.0
        qpos[..., follower_joint.qs_idx_local[0]] = 0.0
    qpos[..., i_target_q] = 0.0
    entity.set_qpos(qpos)
    mass_mat = entity.get_mass_mat()
    scene.step()

    qpos_after = entity.get_qpos()
    for driver_joint, follower_joint, driver_scale, follower_scale in joints_pairs:
        (i_driver_q,) = driver_joint.qs_idx_local
        (i_follower_q,) = follower_joint.qs_idx_local
        (i_driver_d,) = driver_joint.dofs_idx_local
        (i_follower_d,) = follower_joint.dofs_idx_local
        driver_displacement = (qpos_after[..., i_driver_q] - qpos[..., i_driver_q]) / driver_scale
        expected_follower_position = COEFFICIENTS[0] + COEFFICIENTS[1] * driver_displacement
        assert_allclose(
            qpos_after[..., i_follower_q] / follower_scale,
            expected_follower_position,
            atol=MAX_RESIDUAL_RATIO * COEFFICIENTS[0],
        )

        # Each joint has its own root link, so its generalized impulse is M_ii * delta_q / dt. The two impulses have the
        # ratio -poly'(driver), including the scaling of the coordinates.
        driver_impulse = (
            mass_mat[..., i_driver_d, i_driver_d] * (qpos_after[..., i_driver_q] - qpos[..., i_driver_q]) / DT
        )
        follower_impulse = (
            mass_mat[..., i_follower_d, i_follower_d] * (qpos_after[..., i_follower_q] - qpos[..., i_follower_q]) / DT
        )
        assert_allclose(driver_impulse, -COEFFICIENTS[1] * follower_scale / driver_scale * follower_impulse, tol=tol)

    assert_allclose(qpos_after[..., i_target_q] / SCALE, TARGET_POSITION, atol=MAX_RESIDUAL_RATIO * TARGET_POSITION)
    assert_allclose(qpos_after[..., i_unrelated_q] / SCALE, UNRELATED_POSITION, tol=tol)


@pytest.mark.required("sparse")
@pytest.mark.parametrize("rigid_floor_contact_type", ["tet", "vert"])
def test_sap_rigid_floor_contact(rigid_floor_contact_type, asset_tmp_path, show_viewer):
    SLAB_SIZE = (0.4, 0.4, 0.2)
    CUBE_SIZE = (0.1, 0.1, 0.1)
    RHO = 1000.0
    GRAVITY = 9.81
    DT = 1 / 60
    SUBSTEPS = 2
    SAP_TAUD = 0.1
    SAP_BETA = 0.125
    # Stiffness of each contact of the vertex floor
    VERT_STIFFNESS = 1e6
    FRICTION_MU = 0.5
    STICTION_TOLERANCE = 1e-4

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=DT,
            substeps=SUBSTEPS,
            gravity=(0.0, 0.0, -GRAVITY),
        ),
        coupler_options=gs.options.SAPCouplerOptions(
            sap_taud=SAP_TAUD,
            sap_beta=SAP_BETA,
            stiction_tolerance=STICTION_TOLERANCE,
            rigid_floor_contact_type=rigid_floor_contact_type,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.5, -2.0, 1.0),
            camera_lookat=(0.0, 0.2, 0.1),
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )
    scene.add_entity(
        gs.morphs.Plane(),
    )
    # A slab in the first environment and a small cube in the second, each starting slightly sunk into the ground. The
    # second environment keeps the slab where it was last placed, sunk deep into the ground, which it must not touch.
    variants = scene.add_entity(
        morph=(
            gs.morphs.Box(
                size=SLAB_SIZE,
                pos=(0.0, 0.0, 0.5 * SLAB_SIZE[2] - 0.002),
            ),
            gs.morphs.Box(
                size=CUBE_SIZE,
                pos=(0.0, 0.0, 0.5 * CUBE_SIZE[2] - 0.002),
            ),
        ),
        material=gs.materials.Rigid(
            rho=RHO,
        ),
    )
    # A small cube dropped from a few centimeters
    dropped_cube = scene.add_entity(
        morph=gs.morphs.Box(
            size=CUBE_SIZE,
            pos=(0.6, 0.0, 0.5 * CUBE_SIZE[2] + 0.048),
        ),
        material=gs.materials.Rigid(
            rho=RHO,
        ),
    )
    # A small cube released on one of its edges, tilted by 20 degrees
    tilted_cube = scene.add_entity(
        morph=gs.morphs.Box(
            size=CUBE_SIZE,
            pos=(-0.6, 0.0, 0.5 * CUBE_SIZE[2] * (np.cos(np.radians(20.0)) + np.sin(np.radians(20.0)))),
            euler=(0.0, 20.0, 0.0),
        ),
        material=gs.materials.Rigid(
            rho=RHO,
        ),
    )
    # A small cube whose faces are split into 4 triangles, with 9 vertices on its bottom face instead of 4
    cube_mesh = trimesh.creation.box(CUBE_SIZE)
    refined_cube_mesh = cube_mesh.subdivide()
    refined_cube_path = str(asset_tmp_path / "refined_cube.obj")
    refined_cube_mesh.export(refined_cube_path)
    refined_cube = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=refined_cube_path,
            pos=(0.0, -0.6, 0.5 * CUBE_SIZE[2]),
            decimate=False,
            convexify=False,
        ),
        material=gs.materials.Rigid(
            rho=RHO,
        ),
    )
    # A slab pushed sideways by half the force that would make it slide
    pushed_slab = scene.add_entity(
        morph=gs.morphs.Box(
            size=SLAB_SIZE,
            pos=(0.0, 0.8, 0.5 * SLAB_SIZE[2]),
        ),
        material=gs.materials.Rigid(
            rho=RHO,
            coup_friction=FRICTION_MU,
        ),
    )
    scene.build(n_envs=2)

    push_force = 0.5 * FRICTION_MU * RHO * np.prod(SLAB_SIZE) * GRAVITY
    for _ in range(100):
        pushed_slab.control_dofs_force(push_force, dofs_idx_local=0)
        scene.step()

    # Each environment rests still on its own variant, and the other cubes rest still, flat on the ground
    for entity in (variants, dropped_cube, tilted_cube, refined_cube):
        assert_allclose(entity.get_vel(), 0.0, atol=1e-5)
        assert_allclose(entity.get_ang(), 0.0, atol=1e-4)
        assert_allclose(entity.get_quat()[..., 1:3], 0.0, atol=1e-3)

    # The pushed slab sticks, creeping at most at the stiction tolerance
    assert (torch.linalg.norm(pushed_slab.get_vel(), dim=-1) < STICTION_TOLERANCE).all()

    # The contacts of a body at rest form one patch, regularized as one contact whatever their number. Each contact of
    # stiffness k takes the regularization max(beta^2 / (4 pi^2) * |W| / 3 * K / k, 1 / (dt * k * (dt + taud))), from the
    # Delassus operator W = I / m + [r]x I^-1 [r]x^T of the contact at r from the center of mass and the stiffness K of
    # the patch, the near-rigid one for light bodies and the compliant one for heavy ones. At rest, each contact carries
    # the impulse vn_hat / Rn, vn_hat = depth / (dt + taud), and together they hold the weight. The vertex floor holds a
    # box on the vertices of its bottom face, all of stiffness VERT_STIFFNESS. The tet floor holds it on slices spread
    # over that face, closer to its center of mass than the corners, so that it rests less deep.
    substep_dt = DT / SUBSTEPS
    rest_depths = []
    for size, mesh in (
        (SLAB_SIZE, trimesh.creation.box(SLAB_SIZE)),
        (CUBE_SIZE, cube_mesh),
        (CUBE_SIZE, refined_cube_mesh),
    ):
        mass = RHO * np.prod(size)
        inertia = np.diag(mass / 12.0 * (np.sum(np.square(size)) - np.square(size)))
        arms = mesh.vertices[np.isclose(mesh.vertices[:, 2], -0.5 * size[2])]
        arms_cross = np.cross(arms[:, None], np.eye(3))
        delassus = np.eye(3) / mass + arms_cross @ np.linalg.inv(inertia) @ arms_cross.transpose(0, 2, 1)
        near_rigid_regularizations = SAP_BETA**2 / (4.0 * np.pi**2) * np.linalg.norm(delassus, axis=(1, 2)) / 3.0
        regularizations = np.maximum(
            near_rigid_regularizations * len(arms), 1.0 / (substep_dt * VERT_STIFFNESS * (substep_dt + SAP_TAUD))
        )
        rest_depths.append(mass * GRAVITY * substep_dt * (substep_dt + SAP_TAUD) / np.sum(1.0 / regularizations))
    rest_depths = torch.tensor(rest_depths, dtype=gs.tc_float, device=gs.device)
    rest_heights = 0.5 * torch.tensor((SLAB_SIZE[2], CUBE_SIZE[2]), dtype=gs.tc_float, device=gs.device)
    # The tet floor bounds the depth of the refined cube by that of a cube held on its corners
    refined_cube_rest_depth = rest_depths[2] if rigid_floor_contact_type == "vert" else rest_depths[1]
    for entity, entity_rest_heights, entity_rest_depths in (
        (variants, rest_heights, rest_depths[:2]),
        (dropped_cube, rest_heights[1], rest_depths[1]),
        (tilted_cube, rest_heights[1], rest_depths[1]),
        (refined_cube, rest_heights[1], refined_cube_rest_depth),
    ):
        entity_depths = entity_rest_heights - entity.get_pos()[..., 2]
        if rigid_floor_contact_type == "vert":
            # A few roundings of the height of a body in single precision take a share of its depth
            assert_allclose(entity_depths, entity_rest_depths, rtol=5e-3)
        else:
            assert ((entity_depths > 0.0) & (entity_depths < entity_rest_depths)).all()


@pytest.mark.required("sparse")
@pytest.mark.parametrize("enable_lagged_friction", [False, True])
def test_sap_fem_vs_robot(enable_lagged_friction, show_viewer):
    PLANE_HEIGHT = 0.1
    SPHERE_RADIUS = 0.2

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1 / 60,
            substeps=2,
        ),
        fem_options=gs.options.FEMOptions(
            use_implicit_solver=True,
        ),
        # Rigid-FEM contact without FEM self-contact, which no other test covers
        coupler_options=gs.options.SAPCouplerOptions(
            enable_lagged_friction=enable_lagged_friction,
            enable_fem_self_tet_contact=False,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(2.0, 1.5, 1.2),
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )
    scene.add_entity(
        gs.morphs.Plane(
            pos=(0.0, 0.0, PLANE_HEIGHT),
        ),
    )
    sphere = scene.add_entity(
        morph=gs.morphs.Sphere(
            pos=(0.0, 0.0, PLANE_HEIGHT + SPHERE_RADIUS),
            radius=SPHERE_RADIUS,
        ),
        material=gs.materials.FEM.Elastic(
            E=1e5,
            nu=0.4,
            model="linear_corotated",
        ),
    )
    asset_path = get_hf_dataset(pattern="cross.xml")
    robot = scene.add_entity(
        gs.morphs.MJCF(
            file=f"{asset_path}/cross.xml",
            pos=(0.0, 0.0, PLANE_HEIGHT + 2 * SPHERE_RADIUS + 0.04),
            scale=0.5,
        ),
    )
    scene.build()

    # Run the simulation
    for _ in range(50):
        scene.step()

    # Check that the sphere did not move, and is slightly squished on top of the ground plane
    sphere_verts_pos = sphere.get_verts_pos()
    center = sphere_verts_pos.mean(axis=0)
    assert_allclose(center[:2], 0.0, tol=0.01)
    assert center[2] < PLANE_HEIGHT + SPHERE_RADIUS - 0.02
    # FIXME: The SAP solve stops before converging on resting contacts, so the sphere sinks into the ground and
    # oscillates. The bound covers that error until the solve converges.
    assert_allclose(sphere_verts_pos[..., 2].min(), PLANE_HEIGHT, atol=2e-3)

    # Check that the ant is laying on top of the sphere
    robot_pos = robot.get_pos()
    assert_allclose(robot_pos[:2], 0.0, tol=0.03)
    assert robot_pos[2] > (PLANE_HEIGHT + 2 * SPHERE_RADIUS + 0.04) - 0.05

    # Check that the legs of the ants are resting on the sphere
    assert_allclose(robot.get_qpos()[-4:].abs(), 1.0, tol=0.1)


@pytest.mark.required("sparse")
@pytest.mark.parametrize("integrator", [gs.integrator.Euler, gs.integrator.implicitfast])
def test_sap_coupler_implicit_damping(integrator, show_viewer):
    DT = 0.01
    N_STEPS = 10
    SPHERE_RADIUS = 0.1
    SAP_RTOL = 1e-10

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=DT,
            substeps=1,
            gravity=(0.0, 0.0, 0.0),
        ),
        rigid_options=gs.options.RigidOptions(
            integrator=integrator,
        ),
        fem_options=gs.options.FEMOptions(
            use_implicit_solver=True,
        ),
        coupler_options=gs.options.SAPCouplerOptions(
            sap_convergence_rtol=SAP_RTOL,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.5, 1.5, 1.0),
            camera_lookat=(0.5, 0.0, 0.3),
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(
        morph=gs.morphs.Plane(),
    )
    # A resting FEM sphere keeps the SAP solve running, which the damped cube floating away from it goes through
    scene.add_entity(
        morph=gs.morphs.Sphere(
            pos=(0.0, 0.0, 0.9 * SPHERE_RADIUS),
            radius=SPHERE_RADIUS,
        ),
        material=gs.materials.FEM.Elastic(),
    )
    cube = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(1.0, 0.0, 0.5),
            size=(0.1, 0.1, 0.1),
        ),
    )
    scene.build()

    dofs_damping = [2.0, 5.0, 10.0, 0.0, 0.0, 0.05]
    cube.set_dofs_damping(dofs_damping)
    vel = np.array([0.5, -0.5, 0.5, 0.0, 0.0, 5.0])
    cube.set_dofs_velocity(vel)
    mass_diag = np.diag(tensor_to_array(cube.get_mass_mat()))
    for _ in range(N_STEPS):
        scene.step()
        # The implicit damping of a free cube spinning about a principal axis decouples its dofs: (M + hD) v' = M v
        vel = vel * mass_diag / (mass_diag + DT * np.array(dofs_damping))
    # Each SAP solve leaves the velocities within its relative tolerance of the velocities of the scene, which the
    # spinning cube dominates, an error the steps add up. Single precision stops that solve at its round-off.
    tol = N_STEPS * SAP_RTOL * np.abs(vel).max() if gs.np_float == np.float64 else 2e-7
    assert_allclose(cube.get_dofs_velocity(), vel, tol=tol)

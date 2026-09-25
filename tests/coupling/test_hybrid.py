import platform
import sys

import numpy as np
import pytest
import torch

import genesis as gs
from genesis.utils.misc import tensor_to_array

from ..utils.assertions import assert_allclose
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
    scene.build()

    scene.reset()
    for i in range(150):
        robot.control_dofs_velocity(np.array([2.0 * np.sin(2 * np.pi * i * 0.006)] * robot.n_dofs))
        scene.step()

    ball_pos_delta = ball.get_pos() - torch.tensor(BALL_POS_INIT, dtype=gs.tc_float, device=gs.device)
    assert_allclose(ball_pos_delta[..., 0], 0.0, tol=1e-2)
    assert ((0.02 < ball_pos_delta[1]) & (ball_pos_delta[1] < 0.05)).all()
    assert_allclose(ball_pos_delta[..., 2], 0.0, tol=1e-3)


@pytest.mark.required
@pytest.mark.parametrize("substeps", [1, 10])
def test_rigid_mpm_legacy_coupling(substeps, show_viewer):
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

    scene.build()

    horizon = horizon_substeps // substeps
    for i in range(horizon):
        scene.step()

    # Check that the sand moved the box along the negative Y direction
    pos = tensor_to_array(obj_rigid.get_pos())
    assert pos[1] + 0.25 < 0.0


@pytest.mark.slow  # ~250s
@pytest.mark.required
@pytest.mark.skipif(
    sys.platform == "linux" and platform.machine() == "aarch64",
    reason="pygel3d is not supported on Linux ARM (aarch64)",
)
def test_mesh_mpm_build(show_viewer):
    # FIXME: This test is crashing on Linux (x86 & aarch64) Github-hosted runners

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


@pytest.mark.slow  # ~200s
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

    assert_allclose(cloth.get_particles_vel(), 0.0, atol=1e-5)
    assert_allclose(mpm_cube.get_particles_vel(), 0.0, atol=2e-4)
    assert_allclose(entity_fem._solver.get_state(0).vel, 0, atol=1e-3)
    assert_allclose(water.get_particles_vel(), 0.0, atol=5e-2)


@pytest.mark.required
@pytest.mark.parametrize(
    "n_envs, material_type",
    [
        (0, gs.materials.SPH.Liquid),
        (1, gs.materials.SPH.Liquid),
        (2, gs.materials.SPH.Liquid),
        (2, gs.materials.PBD.Liquid),
        (2, gs.materials.MPM.Liquid),
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


@pytest.mark.required
@pytest.mark.parametrize("precision", ["64"])
def test_sap_rigid_rigid_hydroelastic_contact(show_viewer):
    BOX_POS = (0.0, 0.0, 0.1)
    BOX_HALFHEIGHT = 0.1

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1 / 60,
            substeps=2,
        ),
        coupler_options=gs.options.SAPCouplerOptions(
            sap_convergence_atol=1e-10,
            sap_convergence_rtol=1e-10,
            pcg_threshold=1e-10,
            linesearch_ftol=1e-10,
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )
    plane = scene.add_entity(
        gs.morphs.Plane(
            collision=False,
        ),
    )
    box = scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.5, 0.5, 2 * BOX_HALFHEIGHT),
            pos=(0.0, 0.0, BOX_HALFHEIGHT),
        ),
        material=gs.materials.Rigid(),
    )
    asset_path = get_hf_dataset(pattern="heavy_three_joint_link.xml")
    robot_1 = scene.add_entity(
        gs.morphs.MJCF(
            file=f"{asset_path}/heavy_three_joint_link.xml",
            pos=(-0.2, -0.26, 0.0),
            scale=0.3,
        ),
    )
    robot_2 = scene.add_entity(
        gs.morphs.MJCF(
            file=f"{asset_path}/heavy_three_joint_link.xml",
            pos=(0.17, -0.26, 0.1),
            euler=(0.0, 0.0, 90.0),
            scale=0.3,
        ),
    )
    scene.build()

    # Run simulation
    for _ in range(80):
        scene.step()

    # All the entities must be still
    for entity in scene.entities:
        assert_allclose(entity.get_links_vel(), 0.0, atol=0.03)

    # The box should stay at its initial position
    assert_allclose(box.get_pos(), (0.0, 0.0, BOX_HALFHEIGHT), atol=2e-3)

    # The box, and both robots should be laying on top of each other
    robot_1_min_corner, robot_1_max_corner = robot_1.get_AABB()
    robot_2_min_corner, robot_2_max_corner = robot_2.get_AABB()
    assert (robot_1_min_corner[:2] > -0.4).all() and (robot_2_min_corner[:2] > -0.4).all()
    assert (robot_1_min_corner[:2] < 0.4).all() and (robot_2_min_corner[:2] < 0.4).all()
    assert robot_1_max_corner[2] > 2 * BOX_HALFHEIGHT
    assert robot_2_max_corner[2] > robot_1_max_corner[2] + 0.05


@pytest.mark.slow  # ~200s
@pytest.mark.required
@pytest.mark.parametrize("precision", ["64"])
def test_sap_fem_vs_robot(show_viewer):
    SPHERE_RADIUS = 0.2

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1 / 60,
            substeps=2,
        ),
        fem_options=gs.options.FEMOptions(
            use_implicit_solver=True,
        ),
        coupler_options=gs.options.SAPCouplerOptions(),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(2.0, 1.5, 1.2),
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )
    plane = scene.add_entity(
        gs.morphs.Plane(
            collision=False,
        ),
    )
    sphere = scene.add_entity(
        morph=gs.morphs.Sphere(
            pos=(0.0, 0.0, SPHERE_RADIUS),
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
            pos=(0.0, 0.0, 2 * SPHERE_RADIUS + 0.04),
            scale=0.5,
        ),
    )
    scene.build()

    # Run the simulation
    for _ in range(50):
        scene.step()

    # Check that the sphere did not move, and the slightly squished
    state = sphere.get_state()
    center = state.pos.mean(axis=(0, 1))
    assert_allclose(center[:2], 0.0, tol=0.01)
    assert center[2] < SPHERE_RADIUS - 0.02

    # Check that the ant is laying on top of the sphere
    robot_pos = robot.get_pos()
    assert_allclose(robot_pos[:2], 0.0, tol=0.03)
    assert robot_pos[2] > (2 * SPHERE_RADIUS + 0.04) - 0.05

    # Check that the legs of the ants are resting on the sphere
    assert_allclose(robot.get_qpos()[-4:].abs(), 1.0, tol=0.1)

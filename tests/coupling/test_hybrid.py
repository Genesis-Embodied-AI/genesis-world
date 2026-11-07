import numpy as np
import pytest
import torch

import genesis as gs
from genesis.utils.misc import tensor_to_array

from ..utils.assertions import assert_allclose
from ..utils.assets import get_hf_dataset


@pytest.mark.slow  # ~60s
@pytest.mark.required
def test_rigid_mpm_muscle(show_viewer):
    BALL_POS_INIT = (0.8, 0.6, 0.12)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1e-2,
        ),
        rigid_options=gs.options.RigidOptions(
            gravity=(0, 0, -9.8),
            constraint_timeconst=0.02,
        ),
        mpm_options=gs.options.MPMOptions(
            gravity=(0.0, 0.0, 0.0),  # mimic gravity compensation
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
                sampler="regular",
            ),
            # The coupling scales the velocity the links impose on the particles by exp(-dt damping)
            damping=300.0,
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
    for i in range(60):
        robot.control_dofs_velocity(np.array([4.5 * np.sin(2 * np.pi * i * 0.02)] * robot.n_dofs))
        scene.step()

    # The arm drives the muscle into the ball, which rolls away on the plane, mostly along +y
    ball_pos_delta = ball.get_pos() - torch.tensor(BALL_POS_INIT, dtype=gs.tc_float, device=gs.device)
    assert (ball_pos_delta[..., 1] > 0.05).all()
    assert (ball_pos_delta[..., 0].abs() < 0.4 * ball_pos_delta[..., 1]).all()
    assert_allclose(ball_pos_delta[..., 2], 0.0, tol=1e-3)


@pytest.mark.required
def test_mpm_on_terrain(show_viewer):
    # The terrain is the only Rigid geom, so it is in no Rigid collision pair
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1e-2,
        ),
        mpm_options=gs.options.MPMOptions(
            lower_bound=(0.0, 0.0, -0.2),
            upper_bound=(2.0, 2.0, 1.0),
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(
        morph=gs.morphs.Terrain(
            n_subterrains=(1, 1),
            subterrain_size=(2.0, 2.0),
            subterrain_types="flat_terrain",
        ),
        material=gs.materials.Rigid(),
    )
    block = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(1.0, 1.0, 0.3),
            size=(0.2, 0.2, 0.2),
        ),
        material=gs.materials.MPM.Elastic(
            sampler="regular",
        ),
    )
    scene.build()

    for _ in range(60):
        scene.step()
    pos = tensor_to_array(block.get_particles_pos())
    assert np.isfinite(pos).all()
    dx = 1.0 / 64
    assert pos[..., 2].min() > -0.5 * dx


@pytest.mark.required
def test_reject_unsupported_mpm_coupling():
    liquid = (gs.morphs.Box(pos=(0.0, 0.0, 0.3), size=(0.1, 0.1, 0.1)), gs.materials.MPM.Liquid())
    box = (gs.morphs.Box(pos=(0.3, 0.0, 0.1), size=(0.1, 0.1, 0.1)), gs.materials.Rigid())
    soft = (gs.morphs.Box(pos=(-0.3, 0.0, 0.1), size=(0.1, 0.1, 0.1)), gs.materials.FEM.Elastic())
    terrain = (gs.morphs.Terrain(), gs.materials.Rigid())
    for entities, options, requires_grad, match in (
        ((liquid, box), {"friction_cone": gs.friction_cone.elliptic}, False, "elliptic"),
        ((liquid, box), {"constraint_solver": gs.constraint_solver.CG}, False, "Newton solver"),
        ((liquid, box), {"use_hibernation": True}, False, "hibernation"),
        ((liquid, box), {"noslip_iterations": 5}, False, "noslip"),
        ((liquid, soft), {}, False, "implicit solver"),
        ((liquid, terrain), {}, True, "terrain"),
    ):
        scene = gs.Scene(
            sim_options=gs.options.SimOptions(requires_grad=requires_grad),
            rigid_options=gs.options.RigidOptions(**options),
            show_viewer=False,
        )
        for morph, material in entities:
            scene.add_entity(morph=morph, material=material)
        with pytest.raises(gs.GenesisException, match=match):
            scene.build()


@pytest.mark.required
@pytest.mark.cpu_threads(1)
def test_rigid_mpm_legacy_coupling(tol, show_viewer):
    # The liquid collapsing against the rigid box pushes it
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1e-2,
            substeps=1,
        ),
        mpm_options=gs.options.MPMOptions(
            lower_bound=(-0.5, -1.0, 0.0),
            upper_bound=(0.5, 1.0, 1),
        ),
        newton_options=gs.options.NewtonOptions(
            strict=True,
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
            rho=1000.0,
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
        material=gs.materials.MPM.Liquid(
            sampler="regular",
        ),
        # Off the faces of the grid cells, where the rounding of a position would pick the cell, so that the rollout
        # after the reset repeats the first
        morph=gs.morphs.Box(
            pos=(0.003, 0.002, 0.201),
            size=(0.3, 0.3, 0.3),
        ),
        surface=gs.surfaces.Default(
            color=(0.3, 0.3, 1.0),
            vis_mode="particle",
        ),
    )

    scene.build()

    for _ in range(10):
        scene.step()

    # Check that the liquid moved the box along the negative Y direction
    pos = tensor_to_array(obj_rigid.get_pos())
    assert pos[1] + 0.25 < 0.0

    # A rollout after a reset starts free of the coupling wrench the previous one left
    scene.reset()
    for _ in range(10):
        scene.step()
    assert_allclose(obj_rigid.get_pos(), pos, tol=tol)


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
        # The implicit MPM solver is stable at one substep per step, which the explicit SPH solver splits into subcycles
        sim_options=gs.options.SimOptions(
            dt=2e-3,
            substeps=1,
        ),
        mpm_options=gs.options.MPMOptions(
            lower_bound=(0.5, -0.1, -0.05),
            upper_bound=(0.7, 0.1, 0.3),
        ),
        sph_options=gs.options.SPHOptions(
            n_subcycles=10,
            lower_bound=(-0.03, -0.03, -0.08),
            upper_bound=(0.33, 0.33, 1.0),
        ),
        # The implicit MPM solver couples with the implicit FEM solver only. Its mass-proportional Rayleigh damping
        # matches the velocity damping of the explicit FEM solver.
        fem_options=gs.options.FEMOptions(
            use_implicit_solver=True,
            damping_alpha=45.0,
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
            sampler="regular",
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
        # The implicit FEM solver implements the energy Hessian of the linear models only
        material=gs.materials.FEM.Elastic(
            E=3.0e4,
            nu=0.45,
            rho=1000.0,
            model="linear_corotated",
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
    # The implicit MPM solve stops once its velocity update falls below the Newton tolerance, which bounds the velocity
    # of the cube at rest
    assert_allclose(mpm_cube.get_particles_vel(), 0.0, atol=scene.options.newton.velocity_tol)
    assert_allclose(entity_fem._solver.get_state(0).vel, 0, atol=1e-3)
    assert_allclose(water.get_particles_vel(), 0.0, atol=5e-2)


@pytest.mark.required
def test_fem_outside_mpm_grid(show_viewer):
    # The joint solve of MPM and FEM moves the vertices together, so the FEM velocities hold its tolerance
    VELOCITY_TOL = 1e-3

    def fall(with_mpm):
        scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=1e-2,
            ),
            mpm_options=gs.options.MPMOptions(
                lower_bound=(-0.1, -0.1, 0.0),
                upper_bound=(0.1, 0.1, 0.2),
            ),
            # The implicit MPM couples with the implicit FEM alone, in their joint solve
            fem_options=gs.options.FEMOptions(
                use_implicit_solver=True,
            ),
            newton_options=gs.options.NewtonOptions(
                velocity_tol=VELOCITY_TOL,
            ),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(1.0, -1.0, 0.8),
                camera_lookat=(0.2, 0.0, 0.3),
            ),
            show_viewer=show_viewer,
        )
        if with_mpm:
            scene.add_entity(
                morph=gs.morphs.Box(
                    pos=(0.0, 0.0, 0.1),
                    size=(0.05, 0.05, 0.05),
                ),
                material=gs.materials.MPM.Elastic(
                    sampler="regular",
                ),
            )
        fem = scene.add_entity(
            morph=gs.morphs.Box(
                pos=(0.4, 0.0, 0.5),
                size=(0.1, 0.1, 0.1),
            ),
            material=gs.materials.FEM.Elastic(
                E=1e6,
            ),
        )
        scene.build(n_envs=2)
        for _ in range(10):
            scene.step()
        return fem.get_state().vel

    # A body far from the grid moves in the joint solve as it does in the solve of FEM alone
    assert_allclose(fall(with_mpm=True), fall(with_mpm=False), tol=VELOCITY_TOL)


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
    # The implicit MPM takes one large step, while SPH and PBD need small substeps
    if issubclass(material_type, gs.materials.MPM.Base):
        sim_options = gs.options.SimOptions(dt=1e-2, substeps=1)
    else:
        sim_options = gs.options.SimOptions(dt=4e-3, substeps=10)
    scene = gs.Scene(
        sim_options=sim_options,
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
@pytest.mark.parametrize("batch_links_info", [False, True])
def test_sap_rigid_rigid_hydroelastic_contact(batch_links_info, show_viewer):
    PLANES_HEIGHT = (0.1, 0.3)
    BOX_HALFHEIGHT = 0.1

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
            dt=1 / 60,
            substeps=2,
        ),
        rigid_options=gs.options.RigidOptions(
            batch_links_info=batch_links_info,
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

    # Run simulation
    for _ in range(80):
        scene.step()

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
@pytest.mark.parametrize("precision", ["64"])
def test_sap_fem_vs_robot(show_viewer):
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
    state = sphere.get_state()
    center = state.pos.mean(axis=(0, 1))
    assert_allclose(center[:2], 0.0, tol=0.01)
    assert center[2] < PLANE_HEIGHT + SPHERE_RADIUS - 0.02
    # FIXME: The SAP solve stops before converging on resting contacts, so the sphere sinks into the ground and
    # oscillates. The bound covers that error until the solve converges.
    assert_allclose(state.pos[..., 2].min(), PLANE_HEIGHT, atol=2e-3)

    # Check that the ant is laying on top of the sphere
    robot_pos = robot.get_pos()
    assert_allclose(robot_pos[:2], 0.0, tol=0.03)
    assert robot_pos[2] > (PLANE_HEIGHT + 2 * SPHERE_RADIUS + 0.04) - 0.05

    # Check that the legs of the ants are resting on the sphere
    assert_allclose(robot.get_qpos()[-4:].abs(), 1.0, tol=0.1)

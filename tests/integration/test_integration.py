import numpy as np
import pytest

import genesis as gs

from ..utils.assertions import assert_allclose
from ..utils.assets import get_hf_dataset


@pytest.mark.slow("gpu")  # gpu ~250s
@pytest.mark.parametrize("mode", [0, 1, 2])
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
def test_pick_and_place(mode, show_viewer):
    # Add DoF armature to improve numerical stability if not using 'approximate_implicitfast' integrator.
    #
    # This is necessary because the first-order correction term involved in the implicit integration schemes
    # 'implicitfast' and 'Euler' are only able to stabilize each entity independently, from the forces that were
    # obtained from the instable accelerations. As a result, everything is fine as long as the entities are not
    # interacting with each other, but it induces unrealistic motion otherwise. In this case, the acceleration of the
    # cube being lifted is based on the acceleration that the gripper would have without implicit damping.
    #
    # The only way to correct this would be to take into account the derivative of the Jacobian of the constraints in
    # the first-order correction term. Doing this is challenging and would significantly increase the computation cost.
    #
    # In practice, it is more common to just go for a higher order integrator such as RK4.
    if mode == 0:
        integrator = gs.integrator.approximate_implicitfast
        substeps = 1
        armature = 0.0
    elif mode == 1:
        integrator = gs.integrator.implicitfast
        substeps = 4
        armature = 0.0
    elif mode == 2:
        integrator = gs.integrator.Euler
        substeps = 1
        armature = 2.0

    # Create and build the scene
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
            substeps=substeps,
        ),
        rigid_options=gs.options.RigidOptions(
            integrator=integrator,
            box_box_detection=True,
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )
    scene.add_entity(
        gs.morphs.Plane(),
    )
    cube = scene.add_entity(
        gs.morphs.Box(
            size=(0.05, 0.05, 0.05),
            pos=(0.65, 0.0, 0.025),
        ),
        surface=gs.surfaces.Plastic(
            color=(1, 0, 0),
        ),
    )
    scene.add_entity(
        gs.morphs.Box(
            size=(0.05, 0.05, 0.05),
            pos=(0.4, 0.2, 0.025),
            fixed=True,
        ),
        surface=gs.surfaces.Plastic(
            color=(0, 1, 0),
        ),
    )
    franka = scene.add_entity(
        gs.morphs.MJCF(
            file="xml/franka_emika_panda/panda.xml",
        ),
        visualize_contact=True,
        vis_mode="collision",
    )
    scene.build()

    franka.set_dofs_armature(franka.get_dofs_armature() + armature)

    motors_dof = np.arange(7)
    fingers_dof = np.arange(7, 9)
    end_effector = franka.get_link("hand")

    # set control gains
    franka.set_dofs_kp(
        np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]),
    )
    franka.set_dofs_kv(
        np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]),
    )
    franka.set_dofs_force_range(
        np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
        np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]),
    )

    # move to pre-grasp pose
    qpos = franka.inverse_kinematics(
        link=end_effector,
        pos=np.array([0.65, 0.0, 0.22]),
        quat=np.array([0, 1, 0, 0]),
    )
    # gripper open pos
    qpos[-2:] = 0.04
    path = franka.plan_path(qpos_goal=qpos, num_waypoints=300, resolution=0.05, max_retry=10)
    # execute the planned path
    franka.control_dofs_position(np.array([0.15, 0.15]), fingers_dof)
    for waypoint in path:
        franka.control_dofs_position(waypoint)
        scene.step()

    # Get more time to the robot to reach the last waypoint
    for i in range(120):
        scene.step()

    # reach
    qpos = franka.inverse_kinematics(
        link=end_effector,
        pos=np.array([0.65, 0.0, 0.13]),
        quat=np.array([0, 1, 0, 0]),
    )
    franka.control_dofs_position(qpos[:-2], motors_dof)
    for i in range(60):
        scene.step()

    # grasp
    franka.control_dofs_position(qpos[:-2], motors_dof)
    franka.control_dofs_force(np.array([-1.0, -1.0]), fingers_dof)
    for i in range(50):
        scene.step()

    # lift
    qpos = franka.inverse_kinematics(
        link=end_effector,
        pos=np.array([0.65, 0.0, 0.28]),
        quat=np.array([0, 1, 0, 0]),
    )
    franka.control_dofs_position(qpos[:-2], motors_dof)
    for i in range(50):
        scene.step()

    # reach
    qpos = franka.inverse_kinematics(
        link=end_effector,
        pos=np.array([0.4, 0.2, 0.2]),
        quat=np.array([0, 1, 0, 0]),
    )
    path = franka.plan_path(
        qpos_goal=qpos,
        num_waypoints=100,
        resolution=0.05,
        max_retry=10,
        ee_link_name="hand",
        with_entity=cube,
    )
    for waypoint in path:
        franka.control_dofs_position(waypoint[:-2], motors_dof)
        scene.step()

    # Get more time to the robot to reach the last waypoint
    for i in range(50):
        scene.step()

    # release
    franka.control_dofs_position(np.array([0.15, 0.15]), fingers_dof)

    for i in range(180):
        scene.step()
        if i > 150:
            qvel = cube.get_dofs_velocity()
            assert_allclose(qvel, 0, atol=0.02)

    qpos = cube.get_dofs_position()
    assert_allclose(qpos[2], 0.075, atol=2e-3)


@pytest.mark.slow("gpu")  # gpu ~250s
@pytest.mark.required
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
def test_hanging_rigid_cable(show_viewer, tol):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.002,
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )
    robot = scene.add_entity(
        gs.morphs.MJCF(
            file="xml/cable.xml",
        ),
    )
    scene.build()

    links_pos_0 = scene.rigid_solver.dyn_state.links.pos.to_numpy()[:, 0]
    links_quat_0 = scene.rigid_solver.dyn_state.links.quat.to_numpy()[:, 0]
    links_quat_0 /= np.linalg.norm(links_quat_0, axis=-1, keepdims=True)

    robot.set_dofs_position(robot.get_dofs_position())
    if show_viewer:
        scene.visualizer.update()
    for _ in range(100):
        scene.step()

    links_pos_f = scene.rigid_solver.dyn_state.links.pos.to_numpy()[:, 0]
    links_quat_f = scene.rigid_solver.dyn_state.links.quat.to_numpy()[:, 0]
    links_quat_f /= np.linalg.norm(links_quat_f, axis=-1, keepdims=True)
    links_quat_err = 2.0 * np.arccos(np.minimum(np.abs(np.sum(links_quat_f * links_quat_0, axis=-1)), 1.0))

    # FIXME: Why it is not possible to achieve better accuracy?
    assert_allclose(links_pos_0, links_pos_f, tol=1e-3)
    assert_allclose(links_quat_err, 0.0, tol=1e-3)


@pytest.mark.slow  # ~200s
@pytest.mark.parametrize("primitive_type", ["box", "sphere"])
@pytest.mark.parametrize("precision", ["64"])
def test_franka_panda_grasp_fem_entity(primitive_type, show_viewer):
    if gs.use_ndarray:
        pytest.skip("SAPCoupler does not support ndarray yet.")

    GRAPPER_POS_START = (0.65, 0.0, 0.13)
    GRAPPER_POS_END = (0.65, 0.0, 0.18)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1.0 / 60,
            substeps=2,
        ),
        rigid_options=gs.options.RigidOptions(
            enable_self_collision=False,
        ),
        fem_options=gs.options.FEMOptions(
            use_implicit_solver=True,
            pcg_threshold=1e-10,
        ),
        coupler_options=gs.options.SAPCouplerOptions(
            sap_convergence_atol=1e-10,
            sap_convergence_rtol=1e-10,
            pcg_threshold=1e-10,
            linesearch_ftol=1e-10,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.3, 0.0, 0.15),
            camera_lookat=(0.65, 0.0, 0.15),
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )

    franka = scene.add_entity(
        gs.morphs.MJCF(
            file="xml/franka_emika_panda/panda.xml",
        ),
        material=gs.materials.Rigid(
            friction=1.0,
            coup_friction=1.0,
        ),
    )
    # Only allow finger contact to accelerate
    for geom in franka.geoms:
        if "finger" not in geom.link.name:
            geom.desc.contype = 0
            geom.desc.conaffinity = 0
    if primitive_type == "sphere":
        obj = scene.add_entity(
            morph=gs.morphs.Sphere(
                pos=(0.65, 0.0, 0.02),
                radius=0.02,
            ),
            material=gs.materials.FEM.Elastic(
                E=1e5,
                nu=0.4,
                friction_mu=1.0,
                model="linear_corotated",
            ),
        )
    else:  # primitive_type == "box":
        asset_path = get_hf_dataset(pattern="meshes/cube8.obj")
        obj = scene.add_entity(
            morph=gs.morphs.Mesh(
                file=f"{asset_path}/meshes/cube8.obj",
                pos=(0.65, 0.0, 0.02),
                scale=0.02,
            ),
            material=gs.materials.FEM.Elastic(
                friction_mu=1.0,
                model="linear_corotated",
            ),
        )
    scene.build()

    motors_dof = np.arange(7)
    fingers_dof = np.arange(7, 9)
    end_effector = franka.get_link("hand")

    # init
    franka.set_qpos((-1.0124, 1.5559, 1.3662, -1.6878, -1.5799, 1.7757, 1.4602, 0.04, 0.04))
    box_pos_0 = obj.get_state().pos.mean(dim=-2)

    # hold
    qpos = franka.inverse_kinematics(link=end_effector, pos=GRAPPER_POS_START, quat=(0, 1, 0, 0))
    franka.control_dofs_position(qpos[motors_dof], motors_dof)
    for i in range(15):
        scene.step()

    # grasp
    for i in range(10):
        franka.control_dofs_force(np.array([-1.0, -1.0]), fingers_dof)
        scene.step()

    # lift and wait for while to give enough time for the robot to stop shaking
    qpos = franka.inverse_kinematics(link=end_effector, pos=GRAPPER_POS_END, quat=(0, 1, 0, 0))
    franka.control_dofs_position(qpos[motors_dof], motors_dof)
    for i in range(65):
        franka.control_dofs_force(np.array([-1.0, -1.0]), fingers_dof)
        scene.step()

    # Check that the box has moved by the expected delta, without slipping
    box_pos_f = obj.get_state().pos.mean(dim=-2)
    assert_allclose(box_pos_f - box_pos_0, np.array(GRAPPER_POS_END) - np.array(GRAPPER_POS_START), tol=5e-3)

    # wait for a while
    for i in range(25):
        franka.control_dofs_force(np.array([-1.0, -1.0]), fingers_dof)
        scene.step()
    box_pos_post = obj.get_state().pos.mean(dim=-2)
    assert_allclose(box_pos_f, box_pos_post, atol=1e-3)

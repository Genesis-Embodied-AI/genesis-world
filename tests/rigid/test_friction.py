import numpy as np
import pytest
import torch
import trimesh
from scipy.optimize import brentq

import genesis as gs
import genesis.utils.geom as gu
from genesis.utils.misc import tensor_to_array

from ..utils.assertions import assert_allclose, assert_equal
from ..utils.assets import get_hf_dataset
from ..utils.mujoco_parity import simulate_and_check_mujoco_consistency


@pytest.mark.required
@pytest.mark.parametrize("model_name", ["hinge_slide"])
@pytest.mark.parametrize("gs_solver", [gs.constraint_solver.CG, gs.constraint_solver.Newton])
@pytest.mark.parametrize("gs_integrator", [gs.integrator.implicitfast, gs.integrator.Euler])
@pytest.mark.parametrize("backend", [gs.cpu])
def test_frictionloss(gs_sim, mj_sim, tol):
    qvel = np.array([0.7, -0.9])
    simulate_and_check_mujoco_consistency(gs_sim, mj_sim, qvel=qvel, num_steps=2000, tol=tol)

    # Check that final velocity is almost zero
    gs_qvel = gs_sim.rigid_solver.dyn_state.dofs.vel.to_numpy()
    assert_allclose(gs_qvel, 0.0, tol=1e-2)


@pytest.mark.required
@pytest.mark.parametrize("model_name", ["hinge_slide"])
@pytest.mark.parametrize("gs_solver", [gs.constraint_solver.CG, gs.constraint_solver.Newton])
@pytest.mark.parametrize("gs_integrator", [gs.integrator.implicitfast, gs.integrator.Euler])
def test_set_dofs_frictionloss_physics(gs_sim, tol):
    (robot,) = gs_sim.entities

    initial_velocity = np.array([1.0, 0.0])
    robot.set_dofs_velocity(initial_velocity)

    robot.set_dofs_frictionloss(np.array([0.0, 0.0]))
    frictionloss = robot.get_dofs_frictionloss()
    assert_allclose(frictionloss, np.array([0.0, 0.0]), atol=tol)
    for _ in range(10):
        gs_sim.step()
    velocity_zero = gs_sim.rigid_solver.dyn_state.dofs.vel.to_numpy()[:, 0]

    robot.set_dofs_velocity(initial_velocity)
    robot.set_dofs_frictionloss(np.array([1.0, 0.0]))
    frictionloss = robot.get_dofs_frictionloss()
    assert_allclose(frictionloss, np.array([1.0, 0.0]), atol=tol)
    for _ in range(10):
        gs_sim.step()
    velocity_high = gs_sim.rigid_solver.dyn_state.dofs.vel.to_numpy()[:, 0]

    np.testing.assert_array_less(velocity_high[0], velocity_zero[0])
    np.testing.assert_array_less(velocity_high[1], velocity_zero[1])

    robot.set_dofs_velocity(initial_velocity)
    robot.set_dofs_frictionloss(np.array([0.5]), dofs_idx_local=[0])
    frictionloss = robot.get_dofs_frictionloss(dofs_idx_local=[0])
    assert_allclose(frictionloss, np.array([0.5]), atol=tol)
    for _ in range(10):
        gs_sim.step()
    velocity_medium = gs_sim.rigid_solver.dyn_state.dofs.vel.to_numpy()[:, 0]

    np.testing.assert_array_less(velocity_high[0], velocity_medium[0])
    np.testing.assert_array_less(velocity_medium[0], velocity_zero[0])

    friction_effect = velocity_zero[0] - velocity_high[0]
    np.testing.assert_array_less(tol, friction_effect)

    slide_friction_effect = velocity_zero[1] - velocity_high[1]
    np.testing.assert_array_less(tol, slide_friction_effect)


@pytest.mark.slow  # ~200s
@pytest.mark.required
def test_frictionloss_advanced(show_viewer, tol):
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.4, 0.7, 1.4),
            camera_lookat=(0.6, 0.0, 0.0),
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )
    scene.add_entity(gs.morphs.Plane())
    asset_path = get_hf_dataset(pattern="SO101/*")
    robot = scene.add_entity(
        morph=gs.morphs.MJCF(
            file=f"{asset_path}/SO101/so101_new_calib.xml",
        ),
        # vis_mode="collision",
    )
    box = scene.add_entity(
        gs.morphs.Box(
            pos=(0.1, 0.0, 0.6),
            size=(0.025, 0.025, 0.025),
        ),
    )
    scene.build()

    scene.reset()
    for _ in range(230):
        scene.step()

    assert_allclose(robot.get_contacts()["position"][:, 2].min(), 0.0, tol=1e-4)
    assert_allclose(robot.get_AABB()[0, 2], 0.0, tol=2e-4)
    box_pos = box.get_pos()
    assert box_pos[0] > 0.4

    # This is to check collision detection is working correctly on Apple Metal.
    # The box should collide with the robot and roll on the ground within a reasonable range without not blow up.
    assert_allclose(box_pos[1:], 0.0, tol=0.05)
    assert_allclose(box.get_dofs_velocity(), 0.0, tol=50 * tol)


@pytest.mark.slow("gpu")  # gpu ~250s
@pytest.mark.debug(False)  # Disable debug for speedup
@pytest.mark.parametrize(
    "backend, mode, friction, n_boxes, solver, scale, mesh_boxes",
    [
        # Two floating boxes (the original noslip scenario): a balanced half fraction of the backend x friction x scale
        # x geometry matrix, in which every axis value appears four times and every pair of values twice.
        pytest.param(gs.cpu, "noslip", 0.5, 2, gs.constraint_solver.Newton, 0.04, False, marks=pytest.mark.required),
        (gs.cpu, "noslip", 0.5, 2, gs.constraint_solver.Newton, 1.0, True),
        pytest.param(gs.cpu, "noslip", 2.0, 2, gs.constraint_solver.Newton, 0.04, True, marks=pytest.mark.required),
        (gs.cpu, "noslip", 2.0, 2, gs.constraint_solver.Newton, 1.0, False),
        (gs.gpu, "noslip", 0.5, 2, gs.constraint_solver.Newton, 0.04, True),
        pytest.param(gs.gpu, "noslip", 0.5, 2, gs.constraint_solver.Newton, 1.0, False, marks=pytest.mark.required),
        (gs.gpu, "noslip", 2.0, 2, gs.constraint_solver.Newton, 0.04, False),
        pytest.param(gs.gpu, "noslip", 2.0, 2, gs.constraint_solver.Newton, 1.0, True, marks=pytest.mark.required),
        # Constraint solver coverage: the CG configurations document the baseline users can expect from CG. It holds the
        # chain of two boxes; the chain of three at the same pushes is beyond its convergence and stays on Newton.
        (gs.gpu, "elliptic", 2.0, 2, gs.constraint_solver.CG, 1.0, False),
        # At this low friction the push is so strong that every interface satisfies the balance criterion of the test
        # body, and the full contact manifold holds with no orientation control at all.
        (gs.cpu, "elliptic", 0.25, 2, gs.constraint_solver.Newton, 1.0, False),
        (gs.cpu, "elliptic", 0.25, 3, gs.constraint_solver.CG, 1.0, False),
        # Three floating boxes: the longer friction chain that both mechanisms must hold. At 18 DOF the chain turns
        # contact islands on and, past the 16 DOF cooperative threshold on GPU, engages the decomposed solver arm; the
        # elliptic configurations without islands are covered by test_elliptic_cone_coulomb_isotropy. CG takes the
        # lightly loaded configurations, and the mesh configurations at small scale cover scale and mesh contacts.
        pytest.param(gs.cpu, "elliptic", 2.0, 3, gs.constraint_solver.Newton, 1.0, False, marks=pytest.mark.required),
        (gs.cpu, "elliptic", 0.5, 3, gs.constraint_solver.Newton, 0.04, True),
        pytest.param(gs.gpu, "elliptic", 2.0, 3, gs.constraint_solver.Newton, 1.0, False, marks=pytest.mark.required),
        (gs.gpu, "elliptic", 0.5, 3, gs.constraint_solver.Newton, 0.04, True),
        pytest.param(gs.cpu, "noslip", 2.0, 3, gs.constraint_solver.Newton, 0.04, True, marks=pytest.mark.required),
        (gs.cpu, "noslip", 0.5, 3, gs.constraint_solver.CG, 1.0, False),
        pytest.param(gs.gpu, "noslip", 2.0, 3, gs.constraint_solver.Newton, 1.0, False, marks=pytest.mark.required),
    ],
)
def test_static_friction(mode, friction, n_boxes, solver, scale, mesh_boxes, show_viewer, asset_tmp_path):
    # A stack of n_boxes floating boxes is pressed sideways against a fixed wall and must stay static. Both mechanisms
    # that suppress friction creep are exercised: noslip (pyramidal cone with noslip post-iterations) and the elliptic
    # cone (high tangential impedance). Regularized friction alone lets the stack slide slowly under a sustained push.
    GRAVITY = -9.81
    # SAFETY_FACTOR scales the applied push above the theoretical minimum (weight / mu) that keeps the stack from
    # sliding. The pyramidal cone inscribes the true cone and its regularized friction creeps, so noslip needs about 2.5
    # times the minimum. The elliptic cone enforces the exact Coulomb limit and holds down to about 1.08, below which
    # the friction chain no longer transmits the push to the fixed wall. Residual creep shrinks monotonically with
    # impratio and the default value holds with margin.
    SAFETY_FACTOR = 1.1 if mode == "elliptic" else 2.5
    # The noslip pass count is the one where the creep converges on every backend and chain length, rather than the
    # fewest that happens to hold the bound, because anyone reading this test will take the value as a recommendation.
    # Below that count the residual creep varies with the number of passes without shrinking.
    NOSLIP_ITERATIONS = 20
    CG_ITERATIONS = 100
    # Cross-section growth per box toward the wall, so that every interface has its own face height and its own balance
    # criterion (see the tilt analysis below).
    TAPER = 0.3

    scene = gs.Scene(
        rigid_options=gs.options.RigidOptions(
            constraint_solver=solver,
            iterations=CG_ITERATIONS if mode == "elliptic" and solver == gs.constraint_solver.CG else 50,
            noslip_iterations=NOSLIP_ITERATIONS if mode == "noslip" else 0,
            friction_cone=gs.friction_cone.elliptic if mode == "elliptic" else gs.friction_cone.pyramidal,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=((0.5 * n_boxes + 4) * scale, (n_boxes + 1.5) * scale, 3 * scale),
            camera_lookat=(0.5 * n_boxes * scale, 0.0, 0.0),
        ),
        show_viewer=show_viewer,
    )

    for i in range(n_boxes + 1):
        box_size = (scale, scale * (1 + TAPER * (2 - i)), scale * (1 + TAPER * (2 - i)))
        if mesh_boxes:
            mesh_path = str(asset_tmp_path / f"static_friction_box_{scale}_{i}.obj")
            trimesh.creation.box(extents=box_size).export(mesh_path, file_type="obj")
            morph = gs.morphs.Mesh(
                file=mesh_path,
                fixed=(i == 0),
            )
            kinematic_morph = gs.morphs.Mesh(
                file=mesh_path,
                pos=(i * scale, 0, 0),
            )
        else:
            morph = gs.morphs.Box(
                size=box_size,
                fixed=(i == 0),
            )
            kinematic_morph = gs.morphs.Box(
                size=box_size,
                pos=(i * scale, 0, 0),
            )
        scene.add_entity(
            morph,
            material=gs.materials.Rigid(
                rho=200.0,
                friction=friction,
            ),
            vis_mode="collision",
            visualize_contact=True,
        )
        scene.add_entity(
            kinematic_morph,
            material=gs.materials.Kinematic(),
        )

    boxes = scene.rigid_solver.entities
    floating_boxes = boxes[1:]
    contacts_link_a = torch.arange(n_boxes, device=gs.device).repeat_interleave(4)
    scene.build()

    # The solver arms are provably exercised: on GPU the cooperative decomposed arm engages once the chain reaches the
    # 16-DOF threshold (3 boxes); prefer_decomposed_solver is pinned by the test infra (1 on GPU, 0 on CPU).
    rigid_solver = scene.sim.rigid_solver
    if gs.backend != gs.cpu:
        assert rigid_solver.rigid_config.enable_cooperative_constraint_kernels == (6 * n_boxes >= 16)
        assert rigid_solver.rigid_config.prefer_decomposed_solver == (6 * n_boxes >= 16)

    # Force needed to hold the floating boxes static without slipping
    # Native floats: the equilibrium below solves for the rest penetrations with scipy, which needs its residual to
    # come back as a float, and every mass reaches it through the inverse masses and the force targets.
    masses = [float(box.get_mass()) for box in floating_boxes]
    total_mass = sum(masses)
    force_x = (total_mass * GRAVITY) / friction

    # The weights hanging outward of interface i apply a torque 0.5 * scale * (V_i + V_{i+1}) that tilts box i forward.
    # The contact can balance that torque as long as the offset of its pressure resultant, which is the torque divided
    # by the normal force, stays within a quarter of the face height. Beyond that the face starts to open at the top.
    # The stack therefore stands on its own only when every interface satisfies this criterion. Otherwise the torque is
    # cancelled externally: a constant compensating torque suffices when the contact damps the tilt oscillation, and an
    # orientation controller whose pitch target is offset by moment / kp takes over when it does not.
    is_tilt_balanced = all(
        2.0 * (sum(masses[i:]) + sum(masses[i + 1 :])) * friction / (SAFETY_FACTOR * total_mass) <= 1 - TAPER * i
        for i in range(n_boxes)
    )
    # An elliptic stack that satisfies the criterion everywhere stands with no assistance on either solver. When the
    # criterion is violated the elliptic contact still damps the tilt oscillation, so a constant torque is enough.
    is_tilt_damped = mode == "elliptic"
    is_unassisted = is_tilt_balanced and mode == "elliptic"

    # Start every box at its static equilibrium instead of dropping it onto the stack, since the landing transient
    # proves nothing that the holding phase does not. The rest force of a contact is f(d) = k * imp(d)^2 * d / ((1 -
    # imp(d)) * inv_w) with the translation-only inverse weight, and friction still bootstraps at the first step.
    timeconst, dampratio, dmin, dmax, width, mid, power = tensor_to_array(floating_boxes[0].geoms[0].get_sol_params())
    k_stiff = 1.0 / (dmax * dmax * timeconst * timeconst * dampratio * dampratio)
    push = -SAFETY_FACTOR * force_x
    inv_mass = [1.0 / masses[k] + (1.0 / masses[k - 1] if k > 0 else 0.0) for k in range(n_boxes)]
    corner_z = [0.5 * scale * (1 + TAPER * (1 - j)) for j in range(n_boxes)]
    tilt_torques = [
        0.5 * scale * (sum(masses[i:]) + sum(masses[i + 1 :])) * -GRAVITY if is_unassisted else 0.0
        for i in range(n_boxes)
    ]

    # Every interface transmits the whole push. Walking from the outer end, the tilt torques determine how the two
    # contact corners of each interface share it, and inverting the monotone rest force at each corner turns that share
    # into a penetration.
    corner_diff = np.zeros(n_boxes)
    for j in range(n_boxes - 1, -1, -1):
        m_out = corner_z[j + 1] * corner_diff[j + 1] if j < n_boxes - 1 else 0.0
        corner_diff[j] = (m_out - tilt_torques[j]) / corner_z[j]

    def rest_force_error(d, inv_w, target):
        x = min(d / width, 1.0)
        y = x**power / mid ** (power - 1) if x < mid else 1.0 - (1.0 - x) ** power / (1.0 - mid) ** (power - 1)
        imp = dmin + y * (dmax - dmin)
        return 2.0 * k_stiff * imp**2 * d / ((1.0 - imp) * inv_w) - target

    x_off, theta = [0.0], [0.0]
    for j in range(n_boxes):
        hi = width + 0.5 * push * (1.0 - dmax) * inv_mass[j] / (k_stiff * dmax * dmax)
        pens = [
            brentq(rest_force_error, 0.0, hi, args=(inv_mass[j], target))
            for target in (0.5 * (push - corner_diff[j]), 0.5 * (push + corner_diff[j]))
        ]
        x_off.append(x_off[-1] - 0.5 * (pens[0] + pens[1]))
        theta.append(theta[-1] - 0.5 * (pens[1] - pens[0]) / corner_z[j])
    q_eq = np.zeros(2 * n_boxes)
    q_eq[0::2] = x_off[1:]
    q_eq[1::2] = theta[1:]
    boxes_pos_init = [(0.0, 0.0, 0.0)]
    for k, box in enumerate(floating_boxes, start=1):
        x_eq = k * scale + q_eq[2 * (k - 1)]
        box.set_pos([x_eq, 0.0, 0.0])
        box.set_quat(gu.xyz_to_quat(np.array([0.0, q_eq[2 * k - 1], 0.0]), rpy=True))
        boxes_pos_init.append((x_eq, 0.0, 0.0))

    # Push the furthest floating box toward the fixed wall
    floating_boxes[-1].control_dofs_force(SAFETY_FACTOR * force_x, dofs_idx_local=0)

    if not is_unassisted:
        kp = 1000.0 * total_mass
        for i, box in enumerate(floating_boxes):
            tau_pitch = 0.5 * scale * (sum(masses[i:]) + sum(masses[i + 1 :])) * GRAVITY
            if is_tilt_damped:
                box.control_dofs_force(tau_pitch, dofs_idx_local=4)
            else:
                box.set_dofs_kp(kp, dofs_idx_local=slice(3, 6))
                box.set_dofs_kv(100.0 * total_mass, dofs_idx_local=slice(3, 6))
                box.control_dofs_position([0.0, tau_pitch / kp, 0.0], dofs_idx_local=slice(3, 6))

    # Hold under the sustained push for 20 seconds
    for _ in range(2000):
        scene.step()
        assert_equal(rigid_solver.collider.get_contacts()["link_a"], contacts_link_a)

    # The floating boxes stay where the equilibrium put them. Drift is measured per contact, comparing each box against
    # the one supporting it and the first against the fixed wall, so that a slip is attributed to the contact where it
    # happens instead of accumulating down the chain.
    boxes_pos_ref = torch.as_tensor(boxes_pos_init, dtype=gs.tc_float, device=gs.device)
    drift = torch.stack([box.get_pos() for box in boxes]) - boxes_pos_ref
    drift = torch.diff(drift, dim=0)
    if mode == "noslip":
        atol_x = 1e-2 if solver == gs.constraint_solver.Newton else 5e-3
        atol_y = (5e-4 if solver == gs.constraint_solver.Newton else 5e-5) * scale + 2e-5
        atol_z = 2e-3 if solver == gs.constraint_solver.Newton else 2e-4
    else:
        atol_x = (2e-5 if solver == gs.constraint_solver.Newton else 5e-6) * scale
        atol_y = (1e-5 if solver == gs.constraint_solver.Newton else 5e-5) * scale + 2e-7
        atol_z = 2e-3
    assert_allclose(drift[..., 0], 0.0, atol=atol_x)
    assert_allclose(drift[..., 1], 0.0, atol=atol_y)
    assert_allclose(drift[..., 2], 0.0, atol=atol_z)

    # Drop the force below the theoretical threshold; the stack loses its brace and falls
    floating_boxes[-1].control_dofs_force(0.95 * force_x, dofs_idx_local=0)
    for _ in range(300):
        scene.step()
    for box in floating_boxes:
        _, _, box_z = box.get_pos()
        assert box_z < -scale


@pytest.mark.required
def test_static_hold_unaffected_by_press_on_separate_body(show_viewer):
    # The convergence thresholds must stay independent of the forces applied elsewhere in the environment, so a box
    # held against a wall by friction keeps its hold whether or not a body half a metre away is pressed.
    GRAVITY = -9.81
    BOX = 0.06
    WALL = (0.02, 0.4, 0.5)
    FRICTION = 0.25
    SAFETY = 1.25  # push over the Coulomb minimum; tilt-balanced while SAFETY >= 2 * FRICTION
    NOSLIP_ITERATIONS = 20  # leaves the hold resting on a converged solve rather than on regularized creep
    PRESS_SIZE = 0.3
    PRESS_KP = 1.0e5
    PRESS_DEPTH = 0.25  # commanded below the floor, so the press reaches thousands of times the presser's own weight
    N_SETTLE = 30
    N_PRESS = 40

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            gravity=(0.0, 0.0, GRAVITY),
        ),
        rigid_options=gs.options.RigidOptions(
            noslip_iterations=NOSLIP_ITERATIONS,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.3, -1.4, 0.6),
            camera_lookat=(0.3, 0.0, 0.25),
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(
        gs.morphs.Plane(),
        material=gs.materials.Rigid(
            friction=FRICTION,
        ),
    )
    scene.add_entity(
        gs.morphs.Box(
            size=WALL,
            pos=(0.0, 0.0, 0.5 * WALL[2]),
            fixed=True,
        ),
        material=gs.materials.Rigid(
            friction=FRICTION,
        ),
    )
    held_box = scene.add_entity(
        gs.morphs.Box(
            size=(BOX, BOX, BOX),
            pos=(0.5 * (WALL[0] + BOX), 0.0, 0.75 * WALL[2]),
        ),
        material=gs.materials.Rigid(
            friction=FRICTION,
        ),
    )
    presser = scene.add_entity(
        gs.morphs.Box(
            size=(PRESS_SIZE, PRESS_SIZE, PRESS_SIZE),
            pos=(0.5, 0.0, 0.5 * PRESS_SIZE),
        ),
        material=gs.materials.Rigid(
            friction=FRICTION,
        ),
    )
    scene.build(n_envs=2)

    held_box.control_dofs_force(-SAFETY * held_box.get_mass() * -GRAVITY / FRICTION, dofs_idx_local=0)
    for _ in range(N_SETTLE):
        scene.step()
    hold_z = held_box.get_pos()[..., 2]

    press_mass = presser.get_mass()
    presser.set_dofs_kp(PRESS_KP * press_mass, dofs_idx_local=[2])
    presser.set_dofs_kv(0.1 * PRESS_KP * press_mass, dofs_idx_local=[2])
    presser.control_dofs_position([[0.5 * PRESS_SIZE], [0.5 * PRESS_SIZE - PRESS_DEPTH]], dofs_idx_local=[2])
    for _ in range(N_PRESS):
        scene.step()

    # The contact carries the press instead of letting the presser through the floor.
    assert_allclose(presser.get_pos()[..., 2], 0.5 * PRESS_SIZE, atol=0.02 * PRESS_SIZE)
    # A friction solve truncated on the press loses the hold, and the box slides down the wall to the floor.
    slip = held_box.get_pos()[..., 2] - hold_z
    assert_allclose(slip, 0.0, atol=0.05 * BOX)
    assert_allclose(slip[1], slip[0], atol=1e-5)

    # A geom asked for its friction reports what the solver rubs it with, while the property reports what it was built
    # with, which is the value a scene rebuilt from the same material would hold again.
    geom = presser.geoms[0]
    geom.set_friction(0.37)
    geom.set_friction_rolling(0.02)
    assert_allclose(geom.get_friction(), scene.rigid_solver.get_geoms_friction(geom.idx), tol=gs.EPS)
    assert_allclose(geom.get_friction(), 0.37, tol=gs.EPS)
    assert_allclose(geom.get_friction_rolling(), 0.02, tol=gs.EPS)
    assert_allclose(geom.desc.friction, FRICTION, tol=gs.EPS)
    assert_allclose(geom.get_sol_params(), scene.rigid_solver.get_sol_params(geoms_idx=geom.idx)[0], tol=gs.EPS)


@pytest.mark.required
@pytest.mark.parametrize(
    "sparse_solve",
    [
        # Beyond the default arms, the explicit-sparse config pins the elliptic per-island skyline factor on CPU (a lone
        # box is a single island) and the GPU sparse build, which rebuilds with the cone baked in each iteration since
        # the CPU-only incremental cone update is compiled out.
        None,
        True,
    ],
)
def test_elliptic_cone_coulomb_isotropy(sparse_solve, show_viewer):
    # With the box yaw and the tangential center-of-mass force in independent random directions across parallel envs, a
    # box on a plane must slide above the Coulomb threshold |F_t| = mu*N and hold static below it, identically per env.
    GRAVITY = -9.81
    MU = 1.0
    DT = 0.005
    N_ENVS = 16

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=DT,
            gravity=(0.0, 0.0, GRAVITY),
        ),
        rigid_options=gs.options.RigidOptions(
            friction_cone=gs.friction_cone.elliptic,
            sparse_solve=sparse_solve,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.0, 1.0, 0.7),
            camera_lookat=(0.0, 0.0, 0.1),
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(
        gs.morphs.Plane(),
        material=gs.materials.Rigid(
            friction=MU,
        ),
    )
    box = scene.add_entity(
        gs.morphs.Box(
            size=(0.2, 0.2, 0.2),
            pos=(0.0, 0.0, 0.1),
        ),
        material=gs.materials.Rigid(
            friction=MU,
        ),
    )
    scene.build(n_envs=N_ENVS)
    mass = box.get_mass()
    normal_force = MU * mass * (-GRAVITY)

    yaw = 2.0 * torch.pi * torch.rand(N_ENVS, device=gs.device)
    direction = 2.0 * torch.pi * torch.rand(N_ENVS, device=gs.device)
    zeros = torch.zeros(N_ENVS, device=gs.device)
    quat = torch.stack((torch.cos(0.5 * yaw), zeros, zeros, torch.sin(0.5 * yaw)), dim=1)
    force_dir = torch.stack((torch.cos(direction), torch.sin(direction)), dim=1)

    def settle():
        box.control_dofs_force(0.0, dofs_idx_local=[0, 1])
        box.set_pos((0.0, 0.0, 0.1))
        box.set_quat(quat)
        box.set_dofs_velocity(
            torch.cat(
                (0.02 * torch.randn(N_ENVS, 2, device=gs.device), torch.zeros(N_ENVS, 4, device=gs.device)), dim=1
            )
        )
        # Hold each orientation so the CoM force slides the box instead of tipping it about the contact.
        box.set_dofs_kp(1.0e3 * mass, dofs_idx_local=slice(3, 6))
        box.set_dofs_kv(1.0e2 * mass, dofs_idx_local=slice(3, 6))
        box.control_dofs_position(box.get_dofs_position(dofs_idx_local=slice(3, 6)), dofs_idx_local=slice(3, 6))
        for _ in range(25):
            scene.step()

    # Above the Coulomb threshold: the box slides, and the elliptic cone makes the sliding acceleration identical in
    # every direction. Skip the initial transient, then measure the acceleration over a fixed window.
    settle()
    box.control_dofs_force(1.5 * normal_force * force_dir, dofs_idx_local=[0, 1])
    for _ in range(10):
        scene.step()
    vel_0 = box.get_dofs_velocity(dofs_idx_local=[0, 1])
    for _ in range(20):
        scene.step()
    vel_1 = box.get_dofs_velocity(dofs_idx_local=[0, 1])
    accel = torch.linalg.norm(vel_1 - vel_0, dim=1) / (20 * DT)
    # Coulomb caps the friction opposing the 1.5x push at mu * N, leaving exactly half the weight to accelerate the
    # box. A contact whose normal force answers the tangential demand instead brakes harder than that and lands
    # short, which the spread below cannot see since it scales with its own mean.
    assert_allclose(accel, 0.5 * MU * (-GRAVITY), rtol=0.01)
    # The elliptic spread measures ~1e-5 relative; the pyramidal cone's anisotropy spreads it to ~0.5.
    assert accel.std() < 5e-5 * accel.mean()

    # Below the Coulomb threshold: friction holds the box static in every direction, with no slow tangential creep.
    # The elliptic residual measures ~1e-5; the pyramidal cone's regularized friction creeps at ~1e-3.
    settle()
    box.control_dofs_force(0.4 * normal_force * force_dir, dofs_idx_local=[0, 1])
    for _ in range(40):
        scene.step()
    assert (torch.linalg.norm(box.get_dofs_velocity(dofs_idx_local=[0, 1]), dim=1) < 5e-5).all()


@pytest.mark.required
@pytest.mark.parametrize("friction_cone", [gs.friction_cone.pyramidal, gs.friction_cone.elliptic])
@pytest.mark.parametrize("n_envs", [0, 2])
def test_torsional_friction_spin_down_rate(friction_cone, n_envs, show_viewer):
    # I * dw/dt = -friction_torsional * m * g with I = 2/5 m r^2: the saturated spin-down rate is
    # friction_torsional * g / (0.4 r^2), mass-independent. The elliptic cone tracks this exact Coulomb bound once
    # fully slipping; the pyramidal cone's regularized friction decays below it, so only the rate ordering is
    # asserted. The plane's zero coefficient is inert under the pair-by-maximum rule, and a zero-coefficient sphere
    # keeps the weight-only contact force of a torsional-free scene.
    GRAVITY = 9.81
    DT = 0.01
    RADIUS = 0.1
    W0 = 3.0

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=DT,
            gravity=(0.0, 0.0, -GRAVITY),
        ),
        rigid_options=gs.options.RigidOptions(
            friction_cone=friction_cone,
            enable_torsional_friction=True,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.75, -1.8, 0.8),
            camera_lookat=(0.75, 0.0, 0.1),
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(
        gs.morphs.Plane(),
        material=gs.materials.Rigid(
            friction_torsional=0.0,
        ),
    )
    spheres_friction_torsional = (0.0, 0.0001, 0.002, 0.005)
    spheres = []
    for i_s, friction_torsional in enumerate(spheres_friction_torsional):
        spheres.append(
            scene.add_entity(
                gs.morphs.Sphere(
                    radius=RADIUS,
                    pos=(0.5 * i_s, 0.0, RADIUS),
                ),
                material=gs.materials.Rigid(
                    friction_torsional=friction_torsional,
                ),
            )
        )
    scene.build(n_envs=n_envs)

    for _ in range(10):
        scene.step()
    for sphere in spheres:
        sphere.set_dofs_velocity([0.0, 0.0, 0.0, 0.0, 0.0, W0])
    # Let the contact reference dynamics settle into the fully slipping regime before measuring the rate.
    for _ in range(5):
        scene.step()
    w_start = [sphere.get_dofs_velocity()[..., 5] for sphere in spheres]
    for _ in range(10):
        scene.step()
    spin_downs = []
    for sphere, friction_torsional, w_0 in zip(spheres, spheres_friction_torsional, w_start):
        spin_down = w_0 - sphere.get_dofs_velocity()[..., 5]
        spin_downs.append(spin_down)
        if friction_cone == gs.friction_cone.elliptic or friction_torsional == 0.0:
            spin_down_rate = friction_torsional * GRAVITY / (0.4 * RADIUS**2)
            assert_allclose(spin_down, spin_down_rate * 10 * DT, rtol=0.05, atol=1e-3)
    for spin_down_slow, spin_down_fast in zip(spin_downs[1:], spin_downs[2:]):
        assert (spin_down_slow < spin_down_fast).all()
    assert_allclose(
        torch.linalg.norm(spheres[0].get_links_net_contact_force(), dim=-1).sum(dim=-1),
        spheres[0].get_mass() * GRAVITY,
        rtol=0.01,
    )

    # A runtime coefficient update takes effect immediately: the zero-coefficient sphere, still spinning at W0, now
    # decays exactly like the sphere that carried the same coefficient from the start.
    spheres[0].set_friction_torsional(spheres_friction_torsional[-1])
    for _ in range(5):
        scene.step()
    w_runtime_start = spheres[0].get_dofs_velocity()[..., 5]
    for _ in range(10):
        scene.step()
    assert_allclose(w_runtime_start - spheres[0].get_dofs_velocity()[..., 5], spin_downs[-1], rtol=0.05, atol=1e-3)


@pytest.mark.required
@pytest.mark.parametrize("friction_cone", [gs.friction_cone.pyramidal, gs.friction_cone.elliptic])
@pytest.mark.parametrize("n_envs", [0, 2])
def test_rolling_friction_deceleration_rate(friction_cone, n_envs, show_viewer):
    # A sphere rolling without slipping decelerates only through the rolling resistance torque friction_rolling *
    # m * g: with the rolling constraint v = w * r and I = 2/5 m r^2, dv/dt = -(5/7) * friction_rolling * g / r,
    # mass-independent. The elliptic cone tracks this exact Coulomb bound; the pyramidal cone decays below it, so
    # only the rate ordering is asserted. The plane's zero coefficient is inert under the pair-by-maximum rule, and
    # a zero-coefficient sphere keeps rolling freely with the weight-only contact force of a rolling-free scene.
    GRAVITY = 9.81
    DT = 0.01
    RADIUS = 0.1
    V0 = 0.5

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=DT,
            gravity=(0.0, 0.0, -GRAVITY),
        ),
        rigid_options=gs.options.RigidOptions(
            friction_cone=friction_cone,
            enable_torsional_friction=True,
            enable_rolling_friction=True,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.75, -1.8, 0.8),
            camera_lookat=(0.75, 0.0, 0.1),
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(
        gs.morphs.Plane(),
        material=gs.materials.Rigid(
            friction_torsional=0.0,
            friction_rolling=0.0,
        ),
    )
    spheres_friction_rolling = (0.0, 0.001, 0.004)
    spheres = []
    for i_s, friction_rolling in enumerate(spheres_friction_rolling):
        spheres.append(
            scene.add_entity(
                gs.morphs.Sphere(
                    radius=RADIUS,
                    pos=(0.5 * i_s, 0.0, RADIUS),
                ),
                material=gs.materials.Rigid(
                    friction_rolling=friction_rolling,
                ),
            )
        )
    scene.build(n_envs=n_envs)

    for _ in range(10):
        scene.step()
    for sphere in spheres:
        sphere.set_dofs_velocity([V0, 0.0, 0.0, 0.0, V0 / RADIUS, 0.0])
    # Let the contact settle into steady rolling before measuring the deceleration.
    for _ in range(5):
        scene.step()
    v_start = [sphere.get_dofs_velocity()[..., 0] for sphere in spheres]
    for _ in range(10):
        scene.step()
    slow_downs = []
    for sphere, friction_rolling, v_0 in zip(spheres, spheres_friction_rolling, v_start):
        slow_down = v_0 - sphere.get_dofs_velocity()[..., 0]
        slow_downs.append(slow_down)
        if friction_cone == gs.friction_cone.elliptic or friction_rolling == 0.0:
            deceleration = (5.0 / 7.0) * friction_rolling * GRAVITY / RADIUS
            assert_allclose(slow_down, deceleration * 10 * DT, rtol=0.05, atol=1e-3)
    for slow_down_slow, slow_down_fast in zip(slow_downs[:-1], slow_downs[1:]):
        assert (slow_down_slow < slow_down_fast).all()
    assert_allclose(
        torch.linalg.norm(spheres[0].get_links_net_contact_force(), dim=-1).sum(dim=-1),
        spheres[0].get_mass() * GRAVITY,
        rtol=0.01,
    )


# The mesh box at the larger scale is the demanding case, so it is the required one: it reaches a face normal's support
# through the sampled table rather than analytically, and its measured spreads sit closest to their bounds. The two ends
# of the sweep are four orders of magnitude apart, which is what forces every tolerance the step goes through to be
# relative to the quantity it bounds.
@pytest.mark.parametrize(
    "is_box_mesh, scale",
    [
        (False, 1.0),
        (True, 0.01),
        (True, 0.02),
        (True, 0.05),
        (True, 0.1),
        (True, 1.0),
        pytest.param(True, 100.0, marks=pytest.mark.required),
    ],
)
@pytest.mark.parametrize("contact_resolution", [gs.contact_resolution.convex, gs.contact_resolution.signorini])
def test_elliptic_cone_push_isotropy(contact_resolution, is_box_mesh, scale, precision, show_viewer, tol):
    N_ENVS = 8
    FRICTION = 0.5
    is_signorini = contact_resolution == gs.contact_resolution.signorini
    is_fp64 = gs.np_float == np.float64
    # In single precision the solve's residuals stop tracking the scene below the static-friction battery's small
    # scale, and rotated copies of a 'convex' scene genuinely diverge there; the sweep documents that floor by
    # holding every scale above it.
    if not is_fp64 and not is_signorini and scale < 0.04:
        pytest.skip("single-precision convex solve accuracy does not track the scene scale below 0.04")
    # Every length below is quoted at unit scale and multiplied by it, and gravity with it, leaving the motion
    # geometrically similar at any scale so one scene serves the whole sweep and each bound takes the power of the scale
    # its own quantity carries. What the sweep holds is that the tolerances contact detection compares against are
    # relative to the pair of geoms it is given, over two orders of magnitude of geometry and eight of force.
    GRAVITY = 9.81 * scale
    # Every body starts a ten-thousandth of its height into the ground, so contact 0 carries real depth from the first
    # step instead of grazing at the acceptance boundary, where rounding decides which copy detects it.
    BOX_POS = (0.0, 0.0, (1 - 1e-4) * 0.02 * scale)
    BOX_SIZE = (0.1 * scale, 0.2 * scale, 0.04 * scale)
    # The pillar pushes the box below its centre of mass so the box slides rather than tipping: level with it, the box
    # verges on lifting a leading corner and each env's own rounding decides where it settles.
    PILLAR_HEIGHT = 0.04 * scale
    PILLAR_RADIUS = 0.0316 * scale
    # Pusher path in the box's local frame; the shared +y offset gives the push a lever arm that spins the box, and the
    # height places the pillar rather than driving it: its plane contact carrying its weight holds it there.
    PUSH_START_LOCAL = (-0.15 * scale, 0.03 * scale, (1 - 1e-4) * 0.5 * PILLAR_HEIGHT)
    PUSH_END_LOCAL = (0.02 * scale, 0.03 * scale, (1 - 1e-4) * 0.5 * PILLAR_HEIGHT)
    # Each bound is the worst spread measured across every backend, array layout and debug mode, with headroom of about
    # two on a 1-2-5 grid, per precision and per contact resolution: 'signorini' holds a stable manifold while 'convex'
    # couples the normal force with the tangential demand (see contact_resolution in genesis/constants.py), which costs
    # its bounds the difference. Anything that is not rounding exceeds them by orders of magnitude. Each bound carries
    # its quantity's power of the scale, which is zero for a direction or an angular rate with time held fixed, and a
    # bound shared by several quantities covers the largest.
    LENGTH_TOL = (0.2 if is_fp64 else (0.1 if is_signorini else 0.5)) * tol * scale
    DIRECTION_TOL = (0.5 if is_signorini and not is_fp64 else 2.0) * tol
    # The velocities and forces are where the constraint solve leaves its residual, and only 'signorini' pins them
    # tightly enough for a comparison to certify more than their order of magnitude, so they are compared under it
    # alone; the spread between two converged orientations measures several tens of the solver tolerance.
    LIN_VEL_TOL = 2.0 * tol * scale
    ANG_VEL_TOL = (50.0 if is_fp64 else 10.0) * tol
    # Force carries the stiffness gain on top, and coplanar contacts of one pair share the load with a null space the
    # solve may resolve anywhere inside, so the bound covers the split. A mass times an acceleration takes three powers
    # of the scale from the mass and one from gravity; a torque one more from its lever arm.
    FORCE_TOL = (500.0 if is_fp64 else 50.0) * tol * scale**4
    TORQUE_TOL = (10.0 if is_fp64 else 5.0) * tol * scale**5
    # How far either body may sit from the plane resting under its own weight, how still it must end, and how far the
    # pusher may sit from the height its stance gives it and the yaw it was commanded to. 'signorini' resolves the depth
    # on its own and meets each bound to a fraction of a thousandth; 'convex' charges a sliding contact's tangential
    # residual to the normal direction (see contact_resolution in genesis/constants.py), so the body keeps leaving the
    # plane and falling back, and where in that cycle the run ends costs its bounds an order or two.
    GROUND_TOL = (2e-4 if is_signorini else 2e-2) * scale
    REST_LIN_VEL_TOL = ((5e-5 if is_signorini else 1e-4) if is_fp64 else 1e-3) * scale
    REST_ANG_VEL_TOL = (5e-4 if is_fp64 else 2e-3) if is_signorini else (2e-3 if is_fp64 else 5e-3)
    REST_LENGTH_TOL = (1e-4 if is_signorini else 1e-2) * scale
    REST_TILT_TOL = (5e-4 if is_fp64 else 1e-3) if is_signorini else 5e-2

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.005,
            gravity=(0.0, 0.0, -GRAVITY),
        ),
        rigid_options=gs.options.RigidOptions(
            friction_cone=gs.friction_cone.elliptic,
            contact_resolution=contact_resolution,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.7 * scale, 0.7 * scale, 0.8 * scale),
            camera_lookat=(0.0, 0.0, 0.05 * scale),
        ),
        show_viewer=show_viewer,
    )
    ground = scene.add_entity(
        gs.morphs.Plane(),
        material=gs.materials.Rigid(
            friction=FRICTION,
        ),
    )
    # The box is swept over both geometries because they reach the support of a face normal, a direction where several
    # vertices tie, by different means: the primitive resolves it analytically and the mesh reads the sampled table.
    # Either may pick a different tied vertex for a rotated copy, and only sweeping both holds each manifold.
    if is_box_mesh:
        box_morph = gs.morphs.MeshSet(
            files=(trimesh.creation.box(extents=BOX_SIZE),),
            pos=BOX_POS,
        )
    else:
        box_morph = gs.morphs.Box(
            pos=BOX_POS,
            size=BOX_SIZE,
        )
    box = scene.add_entity(
        box_morph,
        material=gs.materials.Rigid(
            friction=FRICTION,
        ),
        visualize_contact=True,
        vis_mode="collision",
    )
    # The pusher is a pillar with a triangular cross-section, standing on three corners fixed in its own frame; a
    # circular one rests on a degenerate manifold whose sampled points follow the world frame, anisotropic before the
    # box is ever touched.
    pusher = scene.add_entity(
        gs.morphs.MeshSet(
            files=(trimesh.creation.cylinder(radius=PILLAR_RADIUS, height=PILLAR_HEIGHT, sections=3),),
            pos=PUSH_START_LOCAL,
        ),
        material=gs.materials.Rigid(
            friction=FRICTION,
        ),
        surface=gs.surfaces.Default(
            smooth=False,
        ),
        visualize_contact=True,
        vis_mode="collision",
    )
    scene.build(n_envs=N_ENVS, env_spacing=(0.3 * scale, 0.3 * scale))

    yaw = 2.0 * torch.pi * torch.arange(N_ENVS, device=gs.device) / N_ENVS
    yaw_euler = torch.stack((torch.zeros_like(yaw), torch.zeros_like(yaw), yaw), dim=1)
    box_quat = gu.xyz_to_quat(yaw_euler, rpy=True)
    box.set_quat(box_quat)

    # Rotate the local pusher path into each env's world frame by the box yaw, and PD-control the pusher's full pose.
    push_start = gu.transform_by_quat(torch.tensor(PUSH_START_LOCAL, device=gs.device).repeat(N_ENVS, 1), box_quat)
    push_end = gu.transform_by_quat(torch.tensor(PUSH_END_LOCAL, device=gs.device).repeat(N_ENVS, 1), box_quat)
    pusher.set_pos(push_start)
    pusher.set_quat(box_quat)
    # Quoted per unit mass, the linear gains are accelerations per unit error, fixed so the pusher tracks the same
    # path at any scale; the angular ones act on the inertia, two powers of length ahead of the mass, and carry that
    # difference.
    pusher_mass = float(pusher.get_mass())
    pusher.set_dofs_kp(2000.0 * pusher_mass, dofs_idx_local=[0, 1])
    pusher.set_dofs_kv(200.0 * pusher_mass, dofs_idx_local=[0, 1])
    pusher.set_dofs_kp(5000.0 * pusher_mass * scale**2, dofs_idx_local=[5])
    pusher.set_dofs_kv(500.0 * pusher_mass * scale**2, dofs_idx_local=[5])

    # Let the box resolve its initial ground contact before the push starts, so the two transients do not couple.
    scene.step()

    # Only the horizontal path and the yaw are driven: holding the height would carry the plane contact whatever the
    # solve does with it, and holding roll and pitch would keep the stance flat however the contacts load it, which is
    # what the manifold below is there to check.
    pusher.control_dofs_position(push_end[:, :2], dofs_idx_local=[0, 1])
    # The pillar is held at its env's own yaw, so every env simulates one rigidly rotated copy of the same scene.
    pusher.control_dofs_position(yaw_euler[:, 2:], dofs_idx_local=[5])
    # Every quantity is compared at every step rather than only the settled pose: a difference that appears once is
    # amplified by the steps that follow, so the end state cannot say which quantity broke first.
    box_quat_inv = gu.inv_quat(box_quat)
    expected_manifold = {
        (ground.geoms[0].idx, box.geoms[0].idx): 4,
        (ground.geoms[0].idx, pusher.geoms[0].idx): 3,
        (box.geoms[0].idx, pusher.geoms[0].idx): 2,
    }
    is_manifold_complete = False
    n_count_matched = 0
    N_STEPS = 160
    for i_step in range(N_STEPS):
        scene.step()

        contacts = scene.rigid_solver.collider.get_contacts(as_tensor=False, to_torch=True)
        counts = [len(positions) for positions in contacts["position"]]
        # What this sweep asserts is contact MANIFOLD invariance, which holds whenever contact 0 carries real depth:
        # single-point detection invariance (tie-breaking which of several equivalent supports contact 0 lands on) is
        # deliberately dropped, since the perturbation-based multi-contact followed by redundant contact pruning
        # recovers the same patch from any of them and enforcing it buys nothing while growing complexity, runtime and
        # an endless tail of edge cases. Under 'convex' a grazing contact enters at zero depth and carries no force, and
        # rounding decides at which step each rotated copy admits it: the manifold is compared at the steps where the
        # counts agree, and such steps are held rare after the loop.
        is_count_matched = counts == counts[:1] * N_ENVS
        n_count_matched += is_count_matched
        if is_signorini:
            assert is_count_matched, f"contact count differs across envs at step {i_step}: {counts}"
        # Which contacts the scene has, once it has them all, not merely how many: each geom pair stands on the corners
        # of one of its geoms, so the per-pair count is a property of the shapes alone and holds for the rest of the
        # run. Only the resolution holding a sliding body on the plane earns this; in flight there is no manifold.
        if is_signorini:
            pair_counts = [
                {
                    pair: int(((contacts["geom_a"][i_env] == pair[0]) & (contacts["geom_b"][i_env] == pair[1])).sum())
                    for pair in expected_manifold
                }
                for i_env in range(N_ENVS)
            ]
            is_manifold_complete = is_manifold_complete or pair_counts[0] == expected_manifold
            if is_manifold_complete:
                for i_env, pair_count in enumerate(pair_counts):
                    assert pair_count == expected_manifold, (
                        f"the manifold lost a contact at step {i_step}, env {i_env}: "
                        f"{pair_count} against {expected_manifold}"
                    )
        # The box leaves the ground for a step at the smallest scale, which leaves the whole scene with nothing
        # to compare while the pose and the velocities below still say the envs agree.
        if counts[0] and is_count_matched:
            blocks = []
            for i_env in range(N_ENVS):
                quat = box_quat_inv[i_env].expand(counts[i_env], 4)
                columns = (
                    gu.transform_by_quat(contacts["position"][i_env], quat),
                    gu.transform_by_quat(contacts["normal"][i_env], quat),
                    gu.transform_by_quat(contacts["force"][i_env], quat),
                    contacts["penetration"][i_env][:, None],
                    contacts["geom_a"][i_env][:, None],
                    contacts["geom_b"][i_env][:, None],
                )
                blocks.append(torch.cat([column.to(gs.tc_float) for column in columns], dim=1))
            paired = torch.stack(blocks)
            # Compared row by row in the order the collider reports them: every geom pair's contacts are ordered by
            # their position in one of the pair's own frames, so a scene and any rotated copy report the same contacts
            # in the same order. That the rows line up at all is itself part of what is checked.
            assert_equal(
                paired[:, :, 10:], paired[0, :, 10:], err_msg=f"contact pairs or their order differ at step {i_step}"
            )
            for key, columns, atol in (
                ("position", slice(0, 3), LENGTH_TOL),
                ("normal", slice(3, 6), DIRECTION_TOL),
                ("penetration", slice(9, 10), LENGTH_TOL),
                *((("force", slice(6, 9), FORCE_TOL),) if is_signorini else ()),
            ):
                values = paired[:, :, columns]
                assert_allclose(values, values[0], atol=atol, err_msg=f"contact {key} differs at step {i_step}")
            # Net wrench per geom pair about the world origin. Coplanar contacts of one pair share the load with a null
            # space the solve may resolve anywhere inside, so the individual forces are not determined while their
            # resultant is: comparing both says which of the two any difference lives in.
            if is_signorini:
                for pair in ((0, 1), (0, 2), (1, 2)):
                    rows = (paired[0, :, 10] == pair[0]) & (paired[0, :, 11] == pair[1])
                    if not rows.any():
                        continue
                    net_force = paired[:, rows, 6:9].sum(dim=1)
                    net_torque = torch.cross(paired[:, rows, 0:3], paired[:, rows, 6:9], dim=-1).sum(dim=1)
                    assert_allclose(
                        net_force, net_force[0], atol=FORCE_TOL, err_msg=f"net force differs at step {i_step}"
                    )
                    assert_allclose(
                        net_torque, net_torque[0], atol=TORQUE_TOL, err_msg=f"net torque differs at step {i_step}"
                    )
        # Dropping through the plane, or leaving it altogether, would leave eight envs identically wrong with every
        # comparison above still green.
        for entity in (box, pusher):
            assert_allclose(entity.get_AABB()[:, 0, 2], 0.0, atol=GROUND_TOL, err_msg=f"off the ground, step {i_step}")

        # Every free joint carries a linear half in world axes and an angular half in its own body frame, so only
        # the former is de-rotated; the latter is already invariant under a rotation of the whole scene.
        if is_signorini:
            velocity = scene.rigid_solver.get_dofs_velocity().reshape((N_ENVS, -1, 6))
            quat = box_quat_inv[:, None].expand(N_ENVS, velocity.shape[1], 4)
            linear = gu.transform_by_quat(velocity[:, :, :3], quat)
            assert_allclose(linear, linear[0], atol=LIN_VEL_TOL, err_msg=f"linear velocity differs at step {i_step}")
            angular = velocity[:, :, 3:]
            assert_allclose(angular, angular[0], atol=ANG_VEL_TOL, err_msg=f"angular velocity differs at step {i_step}")

    # Rounding decides the step at which a grazing row is admitted (see the manifold comment in the loop), so single
    # steps may skip the comparison. Those steps staying rare is what says only rows carrying no force ever differ, and
    # the allowance covers one row entering and leaving once.
    if not is_signorini:
        assert N_STEPS - n_count_matched <= 2

    # The box and pusher come to rest on the plane by the end, the pusher at the height its stance gives it and flat
    # on the yaw it was commanded to: sinking would bury it in the plane, tilting would lift a corner of its stance.
    velocity = scene.rigid_solver.get_dofs_velocity().reshape((N_ENVS, -1, 6))
    assert_allclose(velocity[:, :, :3], 0.0, atol=REST_LIN_VEL_TOL)
    assert_allclose(velocity[:, :, 3:], 0.0, atol=REST_ANG_VEL_TOL)
    assert_allclose(pusher.get_pos()[:, 2], 0.5 * PILLAR_HEIGHT, atol=REST_LENGTH_TOL)
    assert_allclose(
        gu.transform_quat_by_quat(pusher.get_quat(), gu.inv_quat(box_quat)), (1.0, 0.0, 0.0, 0.0), atol=REST_TILT_TOL
    )

    # The final box pose in its own initial frame is identical across every initial yaw.
    rel_pos = gu.transform_by_quat(box.get_pos() - torch.tensor(BOX_POS, device=gs.device), gu.inv_quat(box_quat))
    rel_yaw = gu.quat_to_xyz(gu.transform_quat_by_quat(box.get_quat(), gu.inv_quat(box_quat)), rpy=True)[:, 2]
    # A push that moved the box hardly at all would satisfy the comparison below without exercising anything.
    assert (rel_pos[:, 0] > 0.01 * scale).all() and (rel_yaw.abs() > 0.05).all()
    assert_allclose(rel_pos, rel_pos.mean(dim=0), atol=LENGTH_TOL)
    assert_allclose(rel_yaw, rel_yaw.mean(), atol=DIRECTION_TOL)


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
def test_kinetic_friction(n_envs, show_viewer):
    GRAVITY = 9.81
    SIZE = 0.1
    # A cube resting flat tips once friction can torque it over its leading edge, from mu = width / (2 * com_height).
    TIP_FRICTION = SIZE / (2.0 * 0.5 * SIZE)
    # Friction coefficient and launch speed per box, spanning both sides of that threshold. 0.01 is the smallest
    # coefficient the contact model applies, and the plane wears it so the pairwise max leaves each box its own.
    BOXES = ((0.01, 6.0), (0.25, 6.0), (0.5, 2.0), (0.5, 6.0), (0.5, 20.0), (0.9, 6.0), (1.2, 6.0), (2.0, 6.0))
    N_SETTLE = 40
    N_SLIDE = 40

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.005,
            gravity=(0.0, 0.0, -GRAVITY),
        ),
        rigid_options=gs.options.RigidOptions(
            friction_cone=gs.friction_cone.elliptic,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.0, -1.5, 0.6),
            camera_lookat=(1.0, 0.3, 0.05),
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(
        gs.morphs.Plane(),
        material=gs.materials.Rigid(
            friction=0.01,
        ),
    )
    # Spaced far enough apart never to interact, so each box is its own contact island.
    boxes = [
        scene.add_entity(
            gs.morphs.Box(
                pos=(0.0, 4.0 * SIZE * i_box, 0.5 * SIZE),
                size=(SIZE, SIZE, SIZE),
            ),
            material=gs.materials.Rigid(
                friction=friction,
            ),
        )
        for i_box, (friction, _) in enumerate(BOXES)
    ]
    scene.build(n_envs=n_envs)

    for _ in range(N_SETTLE):
        scene.step()

    height_0 = torch.stack([box.get_pos()[..., 2] for box in boxes])
    for box, (_, speed) in zip(boxes, BOXES):
        velocity = box.get_dofs_velocity()
        velocity[..., 0] = speed
        box.set_dofs_velocity(velocity)
    speed_0 = torch.stack([box.get_dofs_velocity()[..., 0] for box in boxes])

    height_min, height_max, tilt_max = height_0, height_0, torch.zeros_like(height_0)
    for _ in range(N_SLIDE):
        scene.step()
        height = torch.stack([box.get_pos()[..., 2] for box in boxes])
        height_min = torch.minimum(height_min, height)
        height_max = torch.maximum(height_max, height)
        # Angle between the body up-axis and world up, from the quaternion's rotation of e_z.
        quat = torch.stack([box.get_quat() for box in boxes])
        up_z = 1.0 - 2.0 * (quat[..., 1] ** 2 + quat[..., 2] ** 2)
        tilt_max = torch.maximum(tilt_max, torch.rad2deg(torch.arccos(up_z.clamp(-1.0, 1.0))))
    speed_1 = torch.stack([box.get_dofs_velocity()[..., 0] for box in boxes])

    friction = torch.tensor([friction for friction, _ in BOXES], device=gs.device).reshape(-1, *(1,) * (n_envs > 0))
    is_sliding = (friction < TIP_FRICTION).expand_as(height_0)

    # Below the threshold the box slides flat; above it, friction torques it over its leading edge.
    assert_allclose(tilt_max[is_sliding], 0.0, atol=0.1)
    assert (tilt_max[~is_sliding] > 45.0).all()

    # Penetration answers to the load alone, so a sliding box sits exactly as deep as a resting one. Every box here
    # carries the same weight, so holding all of them to their common resting height pins the penetration against
    # both the friction coefficient and the sliding speed at once. The bound is set by the box closest to the
    # tipping threshold, whose contact force genuinely leans toward its leading edge and lifts the centre of mass.
    assert_allclose(height_min[is_sliding], height_0[is_sliding], atol=5e-7)
    assert_allclose(height_max[is_sliding], height_0[is_sliding], atol=5e-7)

    # The height above measures this through the geometry; the contact force states it directly. A sliding contact
    # carries the weight and nothing more, however much tangential force the slide is asking of it.
    normal_force = torch.stack([box.get_links_net_contact_force().sum(dim=-2)[..., 2] for box in boxes])
    assert_allclose(normal_force[is_sliding], boxes[0].get_mass() * GRAVITY, rtol=0.01)

    # Each box decelerates at its own mu * g, independently of how fast it was launched.
    deceleration = (speed_0 - speed_1) / (N_SLIDE * scene.sim.dt)
    assert_allclose(deceleration[is_sliding], (friction * GRAVITY).expand_as(height_0)[is_sliding], rtol=0.01)

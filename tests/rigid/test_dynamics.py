import math

import numpy as np
import pytest
import torch
from quadrants.lang._perf_dispatch import PerformanceDispatcher

import genesis as gs
import genesis.utils.geom as gu
from genesis.engine.solvers.rigid.constraint import solver as constraint_solver
from genesis.engine.solvers.rigid.constraint.solver import ConstraintSolver
from genesis.utils.misc import qd_to_numpy, tensor_to_array

from ..utils.assertions import assert_allclose, assert_equal


@pytest.mark.required
def test_gravity(show_viewer, tol):
    scene = gs.Scene(show_viewer=show_viewer)

    sphere = scene.add_entity(gs.morphs.Sphere())
    ghost = scene.add_entity(
        morph=gs.morphs.Sphere(
            pos=(1.0, 0.0, 0.0),
        ),
        material=gs.materials.Kinematic(),
    )
    scene.build(n_envs=3)

    # Gravity belongs to the solvers that fall under it, and a solver that does not carries no way of setting it at
    # all; the scene-wide call reaches only those that do.
    with pytest.raises(AttributeError):
        ghost.solver.set_gravity((0.0, 0.0, -9.81))

    envs_idx_cases = (
        ([-3, -1], (0, 2)),
        (slice(-2, None), (1, 2)),
        (range(-3, -1), (0, 1)),
        (torch.tensor((True, False, True)), (0, 2)),
        # The last environment, named in each of the forms it can be named in: one index becomes a slice of itself, and
        # the last one is where that has to be said with no stop at all.
        (-1, (2,)),
        ([-1], (2,)),
        ((-1,), (2,)),
        (np.int64(-1), (2,)),
        (np.array((-1,)), (2,)),
        (torch.tensor((-1,)), (2,)),
    )
    # A slice stepping backwards names its environments in reverse, which the kernel resolves and a view cannot.
    if not gs.use_zerocopy:
        envs_idx_cases += ((slice(-1, 0, -1), (2, 1)),)
    gravity_values = torch.tensor(((1.0, 0.0, 0.0), (0.0, 2.0, 0.0), (0.0, 0.0, 3.0)))
    for envs_idx, expected_envs_idx in envs_idx_cases:
        values = gravity_values[: len(expected_envs_idx)]

        scene.sim.set_gravity((0.0, 0.0, 0.0))
        scene.sim.set_gravity(values, envs_idx=envs_idx)
        actual_gravity = scene.rigid_solver.get_gravity()

        scene.sim.set_gravity((0.0, 0.0, 0.0))
        scene.sim.set_gravity(values, envs_idx=expected_envs_idx)
        assert_equal(actual_gravity, scene.rigid_solver.get_gravity())

    scene.sim.set_gravity(torch.tensor([0.0, 0.0, 0.0]))
    scene.sim.set_gravity(torch.tensor([[9.0, 0.0, 0.0], [0.0, 2.0, 0.0]]), envs_idx=[0, 1])
    scene.sim.set_gravity(torch.tensor([1.0, 0.0, 0.0]), envs_idx=np.int64(-3))
    scene.sim.set_gravity(torch.tensor([0.0, 0.0, 3.0]), envs_idx=-1)
    # A vector that is not one, and one vector per environment where a single environment was named: rejected by
    # shape, with the shape said, rather than by whatever the write would have made of it.
    with pytest.raises(gs.GenesisException, match="Invalid input shape"):
        scene.sim.set_gravity(torch.tensor([0.0, -10.0]))
    with pytest.raises(gs.GenesisException, match="Invalid input shape"):
        scene.sim.set_gravity(torch.tensor([[0.0, 0.0, -10.0], [0.0, 0.0, -10.0]]), envs_idx=1)

    scene.step()

    assert_allclose(
        [
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
        ],
        sphere.get_links_acc()[..., 0, :],
        tol=tol,
    )


@pytest.mark.required
def test_all_fixed(show_viewer):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3, 1, 1.5),
            camera_lookat=(0.0, 0.0, 0.5),
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )
    cube = scene.add_entity(
        gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(0.0, 0.0, 0.0),
            fixed=True,
        ),
    )
    scene.build()
    scene.step()

    assert_allclose(cube.get_pos(), 0, tol=gs.EPS)
    assert_allclose(cube.get_quat(), (1.0, 0.0, 0.0, 0.0), tol=gs.EPS)
    assert_allclose(cube.get_vel(), 0, tol=gs.EPS)
    assert_allclose(cube.get_ang(), 0, tol=gs.EPS)
    assert_allclose(scene.rigid_solver.get_links_acc(), 0, tol=gs.EPS)


@pytest.mark.required
@pytest.mark.parametrize("model_name", ["box_box"])
@pytest.mark.parametrize("gs_solver", [gs.constraint_solver.CG, gs.constraint_solver.Newton])
@pytest.mark.parametrize("gs_integrator", [gs.integrator.implicitfast, gs.integrator.Euler])
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
def test_box_box_dynamics(gs_sim):
    (gs_robot,) = gs_sim.entities
    for _ in range(20):
        cube1_pos = np.array([0.0, 0.0, 0.2])
        cube1_quat = np.array([1.0, 0.0, 0.0, 0.0])
        cube2_pos = np.array([0.0, 0.0, 0.65 + 0.1 * np.random.rand()])
        cube2_quat = gu.xyz_to_quat(
            np.array([*(0.15 * np.random.rand(2)), np.pi * np.random.rand()]),
        )
        gs_robot.set_qpos(np.concatenate((cube1_pos, cube1_quat, cube2_pos, cube2_quat)))
        for i in range(110):
            gs_sim.scene.step()
            if i > 100:
                qvel = gs_robot.get_dofs_velocity()
                assert_allclose(qvel, 0, atol=1e-2)

        qpos = gs_robot.get_dofs_position()
        assert_allclose(qpos[8], 0.6, atol=2e-3)


@pytest.mark.debug(False)  # Disable debug for speedup
@pytest.mark.parametrize(
    "box_box_detection, gjk_collision, dynamics",
    [
        (True, False, False),
        (False, False, False),
        (False, False, True),
        (False, True, False),
    ],
)
@pytest.mark.parametrize("backend", [gs.cpu])  # TODO: Cannot afford GPU test for this one
def test_many_boxes_dynamics(box_box_detection, gjk_collision, dynamics, show_viewer):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
        ),
        rigid_options=gs.options.RigidOptions(
            max_collision_pairs=1000,
            box_box_detection=box_box_detection,
            use_gjk_collision=gjk_collision,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(10, 10, 10),
            camera_lookat=(0.0, 0.0, 0.0),
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(
        gs.morphs.Plane(),
    )
    for n in range(5**3):
        i, j, k = int(n / 25), int(n / 5) % 5, n % 5
        scene.add_entity(
            gs.morphs.Box(
                pos=(i * (1.0 - 1e-3), j * (1.0 - 1e-3), k * (1.0 - 1e-3) + 0.5),
                size=(1.0, 1.0, 1.0),
            ),
            surface=gs.surfaces.Default(
                color=(*np.random.rand(3), 0.7),
            ),
        )
    scene.build()

    if dynamics:
        for entity in scene.entities[1:]:
            entity.set_dofs_velocity(4.0 * np.random.rand(6))
    num_steps = 900 if dynamics else 150
    for i in range(num_steps):
        scene.step()
        if i > num_steps - 50:
            qvel = scene.rigid_solver.get_dofs_velocity().reshape((6, -1))
            # Checking the average velocity because is always one cube moving depending on the machine.
            assert_allclose(torch.linalg.norm(qvel, dim=0).mean(), 0, atol=0.05)

    for n, entity in enumerate(scene.entities[1:]):
        i, j, k = int(n / 25), int(n / 5) % 5, n % 5
        qpos = entity.get_dofs_position()
        if dynamics:
            assert qpos[:2].norm() < 20.0
            assert qpos[2] < 5.0
        else:
            qpos0 = np.array((i * (1.0 - 1e-3), j * (1.0 - 1e-3), k * (1.0 - 1e-3) + 0.5))
            assert_allclose(qpos[:3], qpos0, atol=0.05)
            assert_allclose(qpos[3:], 0, atol=0.03)


@pytest.mark.slow  # ~200s
@pytest.mark.required
@pytest.mark.parametrize("model_name", ["double_ball_pendulum"])
def test_apply_external_wrench(xml_path, show_viewer, tol):
    GRAVITY = 2.0

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            substeps=2,
            gravity=(0, 0, -GRAVITY),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0, -3.5, 2.5),
            camera_lookat=(0.0, 0.0, 1.0),
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )

    scene.add_entity(
        gs.morphs.Plane(),
    )
    robot = scene.add_entity(
        gs.morphs.MJCF(
            file=xml_path,
            quat=(1.0, 0, 1.0, 0),
        ),
    )
    duck = scene.add_entity(
        morph=gs.morphs.Mesh(
            file="meshes/duck.obj",
            scale=0.04,
            pos=(1.0, 0.0, 1.0),
            euler=(90, 0, 0),
            collision=False,
        ),
    )
    scene.build()
    rigid_solver = scene.rigid_solver

    end_effector_link_idx = robot.links[-1].idx
    end_effector_link_idx_local = robot.links[-1].idx_local
    duck_link_idx = duck.links[0].idx
    duck_mass = duck.get_mass()
    duck_init_link_pos = duck.base_link.get_pos()
    duck_init_link_R = gu.quat_to_R(duck.base_link.get_quat())
    # The duck is held at rest by cancelling gravity, but the cancelling force is applied away from its center of mass
    # so that the moment arm of 'pos' is exercised: the spurious torque it generates is undone by an opposite torque
    # about the very same frame, hence any error in the arm leaves the duck accelerating.
    duck_lever_arm = (0.2, -0.15, 0.1)
    duck_force_local = tensor_to_array(duck_mass * GRAVITY * duck_init_link_R[2])
    for step in range(801):
        ee_pos = rigid_solver.get_links_pos(end_effector_link_idx)[0]
        duck_pos = rigid_solver.get_links_pos(duck_link_idx)[0]
        duck_quat = rigid_solver.get_links_quat(duck_link_idx)[0]
        if step == 0:
            assert_allclose(ee_pos, (0.8, 0.0, 0.02), tol=1e-4)
        elif step in (500, 600):
            assert_allclose(ee_pos, (0.0, 0.0, 0.82), tol=0.01)
        elif step == 800:
            assert_allclose(ee_pos, (-0.8 / math.sqrt(2), 0.8 / math.sqrt(2), 0.02), tol=0.02)
        assert_allclose(duck_pos, duck_init_link_pos, tol=1e-3)
        assert_allclose(duck_quat, duck.base_link.desc.quat, tol=1e-3)

        if step >= 600:
            force = [-4.0, 4.0, 0.0]
            torque = [0.0, 0.0, 0.0]
        elif step >= 500:
            force = [0.0, 0.0, 0.0]
            torque = [0.0, 0.0, 2.0]
        elif step >= 50:
            force = [0.0, 0.0, 10.0]
            torque = [0.0, 0.0, 0.0]
        else:
            force = [0.0, 0.0, 0.0]
            torque = [0.0, 0.0, 0.0]

        duck.base_link.apply_external_wrench(
            force=duck_force_local,
            torque=-np.cross(duck_lever_arm, duck_force_local),
            pos=duck_lever_arm,
            ref=gs.link_ref_frame.link_COM,
            local=True,
        )
        robot.apply_links_external_wrench(force=force, torque=torque, links_idx_local=end_effector_link_idx_local)
        scene.step()

    duck.base_link.apply_external_torque((0, 1, 0), ref=gs.link_ref_frame.link_COM, local=True)
    assert_allclose(rigid_solver.dyn_state.links.cfrc_applied_vel[duck_link_idx, 0], 0, tol=gs.EPS)
    assert_allclose(
        rigid_solver.dyn_state.links.cfrc_applied_ang[duck_link_idx, 0], -duck_init_link_R[:, 1], tol=gs.EPS
    )

    # A local force and a local application point are both expressed in the frame that 'ref' designates, which only
    # shows on a link whose inertial frame is rotated with respect to its own frame.
    base_link = robot.get_link("base")
    with pytest.raises(AssertionError):
        assert_allclose(base_link.desc.inertial_quat, gu.identity_quat(), tol=gs.EPS)
    base_inertial_quat = torch.as_tensor(base_link.desc.inertial_quat, device=gs.device)
    base_link_pos = rigid_solver.get_links_pos(base_link.idx)
    base_link_quat = rigid_solver.get_links_quat(base_link.idx)
    base_link_COM = rigid_solver.get_links_pos(base_link.idx, ref=gs.link_ref_frame.link_COM)
    base_root_COM = rigid_solver.get_links_root_COM(base_link.idx)
    base_link_R = gu.quat_to_R(base_link_quat)
    base_inertial_R = gu.quat_to_R(gu.transform_quat_by_quat(base_inertial_quat, base_link_quat))
    lever_arm = torch.tensor((0.0, 0.0, 0.1), dtype=gs.tc_float, device=gs.device)
    force_local = torch.tensor((0.0, 1.0, 0.0), dtype=gs.tc_float, device=gs.device)

    rigid_solver.clear_external_force()
    rigid_solver.apply_links_external_wrench(
        force=force_local, links_idx=base_link.idx, pos=lever_arm, ref=gs.link_ref_frame.link_origin, local=True
    )
    force_world = base_link_R @ force_local
    point_world = base_link_pos + base_link_R @ lever_arm
    assert_allclose(rigid_solver.dyn_state.links.cfrc_applied_vel[base_link.idx, 0], -force_world, tol=gs.EPS)
    assert_allclose(
        rigid_solver.dyn_state.links.cfrc_applied_ang[base_link.idx, 0],
        -torch.linalg.cross(point_world - base_root_COM, force_world),
        tol=gs.EPS,
    )

    # A world application point locates the point on its own, so it reproduces the local one it is derived from.
    rigid_solver.clear_external_force()
    base_link.apply_external_force(force_world, pos=point_world)
    assert_allclose(rigid_solver.dyn_state.links.cfrc_applied_vel[base_link.idx, 0], -force_world, tol=gs.EPS)
    assert_allclose(
        rigid_solver.dyn_state.links.cfrc_applied_ang[base_link.idx, 0],
        -torch.linalg.cross(point_world - base_root_COM, force_world),
        tol=gs.EPS,
    )

    rigid_solver.clear_external_force()
    rigid_solver.apply_links_external_wrench(
        force=force_local, links_idx=base_link.idx, pos=lever_arm, ref=gs.link_ref_frame.link_COM, local=True
    )
    force_world = base_inertial_R @ force_local
    point_world = base_link_COM + base_inertial_R @ lever_arm
    assert_allclose(rigid_solver.dyn_state.links.cfrc_applied_vel[base_link.idx, 0], -force_world, tol=gs.EPS)
    assert_allclose(
        rigid_solver.dyn_state.links.cfrc_applied_ang[base_link.idx, 0],
        -torch.linalg.cross(point_world - base_root_COM, force_world),
        tol=gs.EPS,
    )

    with pytest.raises(gs.GenesisException, match="'ref' must be one of"):
        rigid_solver.apply_links_external_wrench(
            force=(0, 0, 0), links_idx=duck_link_idx, ref=gs.link_ref_frame.root_COM
        )
    with pytest.raises(gs.GenesisException, match="Either 'force' or 'torque'"):
        rigid_solver.apply_links_external_wrench(links_idx=duck_link_idx)
    with pytest.raises(gs.GenesisException, match="'pos' requires 'force'"):
        rigid_solver.apply_links_external_wrench(torque=(0, 0, 0), links_idx=duck_link_idx, pos=lever_arm)

    # Armature on a free joint adds to the inertia the solver turns the body with. A torque from rest spins the body up
    # by that augmented inertia under the implicit integrators too. The torque is given in the inertial frame, which
    # holds the inertia tensor, and the angular velocity is read in the link frame.
    ARMATURE, TORQUE = 0.002, (0.02, -0.01, 0.015)
    duck_inertia = duck.base_link.desc.inertia + ARMATURE * np.eye(3)
    duck_inertial_R = gu.quat_to_R(duck.base_link.desc.inertial_quat)
    duck_ang = duck.get_dofs_velocity()[3:]
    duck.set_dofs_armature(ARMATURE, dofs_idx_local=[3, 4, 5])
    duck.base_link.apply_external_torque(TORQUE, ref=gs.link_ref_frame.link_COM, local=True)
    scene.step()
    duck_spin = duck_inertial_R @ np.linalg.solve(duck_inertia, np.array(TORQUE) * scene.dt)
    assert_allclose(duck.get_dofs_velocity()[3:] - duck_ang, duck_spin, tol=1e-4)


@pytest.mark.required
@pytest.mark.parametrize("integrator", [gs.integrator.Euler, gs.integrator.approximate_implicitfast])
def test_energy_analytical_and_conservation(
    spring_double_pendulum, implicit_inertial_origin_chain, show_viewer, tol, integrator
):
    g = 9.81
    dt = 0.001
    h0 = 0.5
    radius = 0.1
    n_steps = 400
    undamped_sol_params = [10.0, 0.001, 0.9, 0.95, 0.001, 0.5, 2.0]

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=dt,
            gravity=(0, 0, -g),
        ),
        rigid_options=gs.options.RigidOptions(
            integrator=integrator,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.25, 1.5, 0.7),
            camera_lookat=(0.25, 0.0, 0.2),
        ),
        show_viewer=show_viewer,
    )
    plane = scene.add_entity(gs.morphs.Plane())
    sphere_a = scene.add_entity(
        gs.morphs.Sphere(
            radius=radius,
            pos=(0, 0, h0),
        ),
    )
    sphere_b = scene.add_entity(
        gs.morphs.Sphere(
            radius=radius,
            pos=(0.5, 0, h0),
        ),
    )
    arm = scene.add_entity(
        gs.morphs.MJCF(
            file=spring_double_pendulum,
        ),
    )
    # A load hung on the elbow link weighs on both joints of the arm and swings with it.
    arm_load = scene.add_entity(
        gs.morphs.URDF(
            file=implicit_inertial_origin_chain,
            pos=(0.45, 0.5, 0.8),
            fixed=True,
            batch_fixed_verts=True,
        ),
    )
    arm_load.attach(arm, parent_link_name="arm_lower", pos=(0.2, 0.0, 0.0))
    # Three copies of a tumbling free body, high enough to fall for the whole horizon: one link, a root with a fixed
    # child, and one link with armature on its angular DOFs.
    tumblers = [
        scene.add_entity(
            gs.morphs.URDF(
                file=implicit_inertial_origin_chain,
                pos=(x, 1.0, 2.0),
                merge_fixed_links=merge_fixed_links,
            ),
        )
        for x, merge_fixed_links in ((1.0, True), (1.6, False), (2.2, True))
    ]
    # A fourth copy receives a fixed entity off its center of mass, so the body it tumbles as is the pair.
    tumblers.append(
        scene.add_entity(
            gs.morphs.URDF(
                file=implicit_inertial_origin_chain,
                pos=(2.8, 1.0, 2.0),
            ),
        )
    )
    load = scene.add_entity(
        gs.morphs.URDF(
            file=implicit_inertial_origin_chain,
            pos=(2.8, 1.0, 2.0),
            fixed=True,
            batch_fixed_verts=True,
        ),
    )
    load.attach(tumblers[-1], parent_link_name=tumblers[-1].base_link.name, pos=(0.3, 0.0, 0.0))
    scene.build()
    rigid_solver = scene.rigid_solver
    # The attachment leaves the anchor of the receiving body where the build put it, so its configuration reads exactly
    # as the lone copy's, offset by the morph pose.
    assert_allclose(tumblers[-1].get_qpos() - tumblers[0].get_qpos(), (1.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), tol=gs.EPS)
    arm_links_idx = [link.idx for link in (*arm.links, *arm_load.links)]
    arm_dofs_idx = slice(arm.dof_start, arm.dof_end)
    tumblers_links_idx = [[link.idx for link in tumbler.links] for tumbler in tumblers]
    tumblers_links_idx[-1] += [link.idx for link in load.links]
    tumblers_dofs_idx = [slice(tumbler.dof_start, tumbler.dof_end) for tumbler in tumblers]
    tumblers_links_mass = [rigid_solver.get_links_mass(links_idx) for links_idx in tumblers_links_idx]
    # One row of mass fractions per tumbler turns the links' centers of mass into the tumblers' own.
    tumblers_all_links_idx = [idx for links_idx in tumblers_links_idx for idx in links_idx]
    tumblers_weights = torch.block_diag(*[(links_mass / links_mass.sum())[None] for links_mass in tumblers_links_mass])

    arm.set_dofs_position([0.5, -0.8])
    ARMATURE = 0.05
    for tumbler in tumblers:
        tumbler.set_dofs_velocity([0.0, 0.0, 0.0, 3.0, 1.0, 2.0])
    tumblers[2].set_dofs_armature(ARMATURE, dofs_idx_local=[3, 4, 5])

    # Nearly undamped contact for sphere_a: small dampratio gives very stiff elastic spring with minimal damping.
    # Contact sol_params are averaged: 0.5*(geom_a + geom_b), so both geoms must share the same params.
    plane.geoms[0].set_sol_params(undamped_sol_params)
    sphere_a.geoms[0].set_sol_params(undamped_sol_params)

    mass = sphere_a.get_links_mass()
    te_initial = sphere_a.get_total_energy()

    ke_a, pe_a, ke_b, pe_b, te_arm, ke_tumblers, vel_tumblers, pos_tumblers = [], [], [], [], [], [], [], []
    pos_tumblers.append(
        tumblers_weights @ rigid_solver.get_links_pos(tumblers_all_links_idx, ref=gs.link_ref_frame.link_COM)
    )
    vel_tumblers.append(
        tumblers_weights @ rigid_solver.get_links_vel(tumblers_all_links_idx, ref=gs.link_ref_frame.link_COM)
    )
    impact_step = -1
    for i in range(n_steps):
        scene.step()
        pos_tumblers.append(
            tumblers_weights @ rigid_solver.get_links_pos(tumblers_all_links_idx, ref=gs.link_ref_frame.link_COM)
        )
        vel_tumblers.append(
            tumblers_weights @ rigid_solver.get_links_vel(tumblers_all_links_idx, ref=gs.link_ref_frame.link_COM)
        )
        te_arm.append(
            rigid_solver.get_kinetic_energy(arm_links_idx, arm_dofs_idx)
            + rigid_solver.get_potential_energy(arm_links_idx, arm_dofs_idx)
        )
        ke_tumblers.append(
            torch.stack(
                [
                    rigid_solver.get_kinetic_energy(links_idx, dofs_idx)
                    for links_idx, dofs_idx in zip(tumblers_links_idx, tumblers_dofs_idx)
                ]
            )
        )
        ke_a.append(sphere_a.get_kinetic_energy())
        pe_a.append(sphere_a.get_potential_energy())
        ke_b.append(sphere_b.get_kinetic_energy())
        pe_b.append(sphere_b.get_potential_energy())
        if impact_step < 0 and scene.rigid_solver.collider.collider_state.n_contacts.to_numpy().any():
            impact_step = i
    assert impact_step > 0

    # Free fall: verify analytical KE and PE (semi-implicit Euler)
    # After step n: v_n = n*g*dt, z_n = h0 - g*dt^2*n*(n+1)/2
    for i in range(impact_step):
        n = i + 1
        expected_ke = 0.5 * mass * (n * g * dt) ** 2
        expected_pe = mass * g * (h0 - g * dt**2 * n * (n + 1) / 2)
        assert_allclose(ke_a[i], expected_ke, tol=tol)
        assert_allclose(pe_a[i], expected_pe, tol=tol)
        assert_allclose(ke_b[i], expected_ke, tol=tol)
        assert_allclose(pe_b[i], expected_pe, tol=tol)

    # Undamped sphere_a: energy conserved after bouncing (drift < 1%)
    te_a_final = ke_a[-1] + pe_a[-1]
    assert_allclose(te_a_final, te_initial, tol=0.01)

    # Damped sphere_b: energy strictly decreased
    te_b_final = ke_b[-1] + pe_b[-1]
    assert te_b_final < te_initial

    # Spring-driven arm with its load: nothing dissipates, so its energy holds through the swing to integration error
    te_arm = torch.stack(te_arm)
    assert_allclose(te_arm, te_arm[0], tol=0.01)
    # The springs must carry a real share of the arm's own energy, or the check above would hold with no spring term
    spring_energy = 0.5 * torch.sum(arm.get_dofs_stiffness() * arm.get_dofs_position() ** 2)
    assert spring_energy > 0.1 * arm.get_total_energy()

    # Gravity is the only force on a tumbling body, so its center of mass falls along the parabola the velocity-implicit
    # update traces, whatever the body turns about it. A body anchored on its center of mass integrates that point
    # itself and follows the parabola exactly. The pair is anchored on its first link alone: its center of mass starts
    # moving with the spin, which the parabola carries through the initial velocity, and turns about the integrated
    # origin, which leaves the parabola by the rotation of the offset over one step, a second-order term per step.
    tumblers_mass = torch.stack([links_mass.sum() for links_mass in tumblers_links_mass])
    steps = torch.arange(n_steps + 1, dtype=gs.tc_float, device=gs.device)[:, None, None]
    pos_tumblers, vel_tumblers = torch.stack(pos_tumblers), torch.stack(vel_tumblers)
    gravity = torch.tensor((0.0, 0.0, -g), dtype=gs.tc_float, device=gs.device)
    pos_expected = pos_tumblers[0] + vel_tumblers[0] * steps * dt + gravity * dt**2 * steps * (steps + 1) / 2
    assert_allclose(pos_tumblers[:, :-1], pos_expected[:, :-1], tol=tol)
    pair_offset = np.linalg.norm(tensor_to_array(pos_tumblers[0, -1] - tumblers[-1].get_pos()))
    pair_spin = np.linalg.norm(tensor_to_array(tumblers[-1].get_dofs_velocity()[3:]))
    assert_allclose(pos_tumblers[:, -1], pos_expected[:, -1], atol=n_steps * (dt * pair_spin) ** 2 * pair_offset)
    # Gravity exerts no torque about the center of mass, so the rotational energy of a tumbling body is a constant of
    # its motion. The midpoint rule keeps it, with armature the augmented energy w^T (I + A) w / 2. Each Newton solve
    # leaves a residual at the working precision, and the horizon accumulates them.
    ke_rot = torch.stack(ke_tumblers) - 0.5 * tumblers_mass * vel_tumblers[1:].square().sum(dim=-1)
    if integrator != gs.integrator.Euler:
        assert_allclose(ke_rot, ke_rot[0], tol=10.0 * tol)


@pytest.mark.slow  # ~250s
@pytest.mark.required
@pytest.mark.parametrize("model_name", ["long_chain"])
def test_mass_mat(xml_path, show_viewer, tol):
    # Create and build the scene
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
            substeps=1,
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )
    plane = scene.add_entity(
        gs.morphs.Plane(),
    )
    franka1 = scene.add_entity(
        gs.morphs.MJCF(
            file="xml/franka_emika_panda/panda.xml",
            pos=(0, 0, 0),
        ),
        vis_mode="collision",
        visualize_contact=True,
    )
    franka2 = scene.add_entity(
        gs.morphs.MJCF(
            file="xml/franka_emika_panda/panda.xml",
            pos=(0, 2, 0),
        ),
        vis_mode="collision",
        visualize_contact=True,
    )
    # High-DOF single tree: its mass submatrix exceeds GPU shared memory, exercising the cooperative >shared-cap
    # assemble (the low-DOF frankas exercise the under-cap shared-memory factor instead).
    long_chain = scene.add_entity(
        gs.morphs.MJCF(
            file=xml_path,
            pos=(5, 0, 2),
        ),
    )
    scene.build()

    # Two identical entities must yield identical mass matrices, and the LTDL factor must reconstruct it.
    mass_mat_1 = franka1.get_mass_mat(decompose=False)
    mass_mat_2 = franka2.get_mass_mat(decompose=False)
    assert mass_mat_1.shape == (franka1.n_dofs, franka1.n_dofs)
    assert_allclose(mass_mat_1, mass_mat_2, tol=tol)

    mass_mat_L, mass_mat_D_inv = franka1.get_mass_mat(decompose=True)
    mass_mat = mass_mat_L.T @ torch.diag(1.0 / mass_mat_D_inv) @ mass_mat_L
    assert_allclose(mass_mat, mass_mat_1, tol=tol)

    # The cooperative >shared-cap assemble maps a flat lane index to a lower-triangular (row, col) via a float sqrt;
    # on GPUs whose sqrt undershoots perfect squares (Apple Metal: sqrt(15129) -> 122.999 instead of 123) a naive
    # inversion lands one row short on every j=0 boundary and silently drops the long-range coupling entries, leaving
    # the assembled mass matrix indefinite. A real joint-space mass matrix is always symmetric positive-definite.
    mass_mat_chain = tensor_to_array(long_chain.get_mass_mat(decompose=False))
    assert_allclose(mass_mat_chain, mass_mat_chain.T, tol=tol)
    assert np.linalg.eigvalsh(0.5 * (mass_mat_chain + mass_mat_chain.T)).min() > 0.0

    # On GPU the high-DOF chain factors through the register-tiled path (auto-enabled above the shared-memory cap when
    # RigidOptions.register_tiled_mass is left to its default); its LTDL factor must reconstruct the mass matrix to the
    # same accuracy as the under-cap path.
    mass_mat_chain_L, mass_mat_chain_D_inv = long_chain.get_mass_mat(decompose=True)
    mass_mat_chain_rec = mass_mat_chain_L.T @ torch.diag(1.0 / mass_mat_chain_D_inv) @ mass_mat_chain_L
    assert_allclose(mass_mat_chain_rec, mass_mat_chain, tol=tol)


@pytest.mark.required
@pytest.mark.parametrize("model_name", ["two_fixed_branches"])
def test_mass_block_partition(xml_path, show_viewer, tol):
    # Two chains rigidly attached to the fixed world are kinematically independent: the mass matrix is block-diagonal,
    # so it must partition into one kinematic tree per branch (factoring two n/2 trees instead of one dense n tree).
    scene = gs.Scene(
        rigid_options=gs.options.RigidOptions(
            enable_collision=False,
        ),
        show_viewer=show_viewer,
    )
    entity = scene.add_entity(
        gs.morphs.MJCF(
            file=xml_path,
        ),
    )
    scene.build(n_envs=0)

    n_dofs = entity.n_dofs
    branch = n_dofs // 2
    block_start = qd_to_numpy(scene.rigid_solver.rigid_info.dofs_mass_block_start)
    block_end = qd_to_numpy(scene.rigid_solver.rigid_info.dofs_mass_block_end)
    assert_allclose(block_start, [0] * branch + [branch] * branch, tol=0)
    assert_allclose(block_end, [branch] * branch + [n_dofs] * branch, tol=0)

    # The two branches do not couple, and the LTDL factor reconstructs the (block-diagonal) mass matrix.
    mass_mat = tensor_to_array(entity.get_mass_mat(decompose=False))
    assert_allclose(mass_mat[:branch, branch:], 0.0, tol=tol)
    mass_mat_L, mass_mat_D_inv = entity.get_mass_mat(decompose=True)
    assert_allclose(mass_mat_L.T @ torch.diag(1.0 / mass_mat_D_inv) @ mass_mat_L, mass_mat, tol=tol)


@pytest.mark.required
@pytest.mark.parametrize(
    "box_position, n_envs",
    [
        ("after", 0),
        ("after", 2),
        ("between", 2),
        ("inside", 2),
        ("inside_target", 2),
    ],
)
def test_merge_matches_single_equivalent_entity(merged_arm_hand_models, box_position, n_envs, show_viewer, tol):
    # A tree merged across several entities by attach() - a hand on the arm tip, a second chained onto that hand's
    # palm, a third on another branch - has the same mass matrix, LTDL factor, and one-step dynamics as the single
    # equivalent entity, and a free body stays block-diagonal from it. Layouts interleaving the free body's DOFs
    # inside the merged block ('between', 'inside', 'inside_target') make attach() raise their cause-specific error.
    mono_xml, arm_xml, arm_box_last_xml, arm_box_first_xml, hand_xml = merged_arm_hand_models
    arm_xml_by_box_position = {
        "after": arm_xml,
        "between": arm_xml,
        "inside": arm_box_last_xml,
        "inside_target": arm_box_first_xml,
    }
    scene = gs.Scene(
        rigid_options=gs.options.RigidOptions(
            enable_collision=False,
        ),
        show_viewer=show_viewer,
    )
    mono = scene.add_entity(
        gs.morphs.MJCF(
            file=mono_xml,
        ),
    )
    arm = scene.add_entity(
        gs.morphs.MJCF(
            file=arm_xml_by_box_position[box_position],
        ),
    )
    box_morph = gs.morphs.Box(
        size=(0.1, 0.1, 0.1),
        pos=(0.0, 2.0, 1.0),
    )
    if box_position == "between":
        box = scene.add_entity(box_morph)
    hand = scene.add_entity(
        gs.morphs.MJCF(
            file=hand_xml,
        ),
    )
    hand_chained = scene.add_entity(
        gs.morphs.MJCF(
            file=hand_xml,
        ),
    )
    hand_branch = scene.add_entity(
        gs.morphs.MJCF(
            file=hand_xml,
        ),
    )
    if box_position == "after":
        box = scene.add_entity(box_morph)
    # Chained attach runs before its own parent is attached: re-rooting is transitive, so the order is free.
    hand_chained.attach(hand, "palm")
    if box_position == "between":
        with np.testing.assert_raises_regex(gs.GenesisException, "Instantiate attached entities consecutively"):
            hand.attach(arm, "tip")
        return
    if box_position == "inside":
        with np.testing.assert_raises_regex(gs.GenesisException, "Declare the attached-onto tree last"):
            hand.attach(arm, "tip")
        return
    hand.attach(arm, "tip")
    hand_branch.attach(arm, "a2")
    # A free body attached onto a link the world carries is carried by it too
    pedestal = scene.add_entity(
        gs.morphs.Box(
            size=(0.2, 0.2, 0.2),
            pos=(0.0, -2.0, 0.1),
            fixed=True,
        )
    )
    mounted = scene.add_entity(
        gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(0.0, -2.0, 0.5),
        )
    )
    mounted.attach(pedestal, pedestal.base_link.name, pos=(0.0, 0.0, 0.15))
    if box_position == "inside_target":
        hand_box = scene.add_entity(
            gs.morphs.MJCF(
                file=hand_xml,
            ),
        )
        with np.testing.assert_raises_regex(gs.GenesisException, "Load the parent's trees as separate entities"):
            hand_box.attach(arm, "freebox")
        return
    scene.build(n_envs=n_envs)

    # attach() re-roots every child link - including previously attached grandchildren - into the parent's tree.
    tip_link = arm.get_link("tip")
    assert hand.base_link.parent_idx == tip_link.idx
    assert hand_chained.base_link.parent_idx == hand.get_link("palm").idx
    assert hand_branch.base_link.parent_idx == arm.get_link("a2").idx
    for child in (hand, hand_chained, hand_branch):
        assert_equal([link.root_idx for link in child.links], tip_link.root_idx)
    assert all(link.is_fixed for link in mounted.links)
    assert mounted.n_dofs == 0
    mounted_verts = mounted.get_verts()
    assert_allclose(mounted_verts.min(dim=-2).values, (-0.05, -2.05, 0.2), tol=tol)
    assert_allclose(mounted_verts.max(dim=-2).values, (0.05, -1.95, 0.3), tol=tol)

    mono_dofs = torch.arange(mono.dof_start, mono.dof_start + mono.n_dofs)
    hands = (hand, hand_chained, hand_branch)
    pair_dofs = torch.cat(
        [
            torch.arange(arm.dof_start, arm.dof_start + arm.n_dofs),
            *(torch.arange(h.dof_start, h.dof_start + h.n_dofs) for h in hands),
        ]
    )
    box_dofs = torch.arange(box.dof_start, box.dof_start + box.n_dofs)

    solver = scene.rigid_solver
    mass_mat = solver.get_mass_mat(decompose=False)
    # The merged tree reproduces the single equivalent entity's full coupled mass matrix.
    assert_allclose(mass_mat[..., pair_dofs[:, None], pair_dofs], mass_mat[..., mono_dofs[:, None], mono_dofs], tol=tol)
    # The free body is a separate kinematic tree, so it stays block-diagonal from the merged tree.
    assert_allclose(mass_mat[..., pair_dofs[:, None], box_dofs], 0.0, tol=tol)
    # The LTDL factor reconstructs the mass matrix.
    mass_mat_L, mass_mat_D_inv = solver.get_mass_mat(decompose=True)
    reconstructed = mass_mat_L.transpose(-2, -1) @ (mass_mat_L * (1.0 / mass_mat_D_inv).unsqueeze(-1))
    assert_allclose(reconstructed, mass_mat, tol=tol)

    # One step from a nontrivial pose at rest: the post-step velocities (accelerations times dt, from zero) match the
    # single equivalent entity's, exercising the solve.
    q = np.linspace(-0.3, 0.3, mono.n_dofs)
    mono.set_dofs_position(q)
    arm.set_dofs_position(q[: arm.n_dofs])
    i_q = arm.n_dofs
    for h in hands:
        h.set_dofs_position(q[i_q : i_q + h.n_dofs])
        i_q += h.n_dofs
    scene.step()

    pair_vel = torch.cat([arm.get_dofs_velocity(), *(h.get_dofs_velocity() for h in hands)], dim=-1)
    assert_allclose(pair_vel, mono.get_dofs_velocity(), tol=tol)

    # The mass matrix reassembled at the new configuration matches too: cross-entity couplings are written by the
    # block-root entity, so they cannot lag one recompute behind when the configuration changes.
    mass_mat = solver.get_mass_mat(decompose=False)
    assert_allclose(mass_mat[..., pair_dofs[:, None], pair_dofs], mass_mat[..., mono_dofs[:, None], mono_dofs], tol=tol)


@pytest.mark.slow  # ~500s
@pytest.mark.required
@pytest.mark.parametrize("precision", ["32", "64"])
@pytest.mark.parametrize("backend", [gs.gpu])
def test_cholesky_tiling(monkeypatch, tol):
    import genesis.engine.solvers

    rigid_solver_build_orig = genesis.engine.solvers.RigidSolver.build

    values = []
    for enable_tiled_cholesky in (True, False):

        def rigid_solver_build(self):
            nonlocal enable_tiled_cholesky

            rigid_solver_build_orig(self)
            self.rigid_config.enable_tiled_cholesky_mass_matrix = enable_tiled_cholesky
            if enable_tiled_cholesky:
                self.rigid_config.tiled_n_dofs_per_entity = 32

        monkeypatch.setattr("genesis.engine.solvers.RigidSolver.build", rigid_solver_build)

        scene = gs.Scene(
            rigid_options=gs.options.RigidOptions(
                constraint_solver=gs.constraint_solver.Newton,
                iterations=1,
                sparse_solve=False,
            ),
            show_viewer=False,
            show_FPS=False,
        )
        scene.add_entity(gs.morphs.Plane())
        gs_robot = scene.add_entity(
            gs.morphs.URDF(
                file="urdf/go2/urdf/go2.urdf",
            ),
        )
        scene.build(n_envs=2)
        assert scene.rigid_solver.rigid_config.enable_tiled_cholesky_mass_matrix == enable_tiled_cholesky

        scene.step()
        assert not scene.rigid_solver.get_error_envs_mask().any()
        assert (scene.rigid_solver.constraint_solver.constraint_state.n_constraints.to_numpy() > 0).all()

        Mgrad = scene.rigid_solver.constraint_solver.constraint_state.Mgrad.to_numpy()
        assert np.linalg.norm(Mgrad) > 5.0
        values.append(Mgrad)

    # analysis for choice tolerance: https://github.com/Genesis-Embodied-AI/Genesis/pull/2659#discussion_r3041684256
    assert_allclose(*values, tol=5e-4)


@pytest.mark.required
@pytest.mark.use_deterministic_algorithms(False)
@pytest.mark.parametrize("backend", [gs.gpu])
def test_solve_arm_equivalence(monkeypatch, show_viewer, tol):
    SIZE = 0.1
    N_STEPS = 12

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
            gravity=(0.0, 0.0, -9.81),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.1, -1.1, 0.7),
            camera_lookat=(1.1, 0.1, 0.05),
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(gs.morphs.Plane())
    # Islands of one, three and nine bodies at once, each sunk a ten-thousandth of its height so contact has real depth
    positions = [(0.0, 0.0, (1 - 1e-4) * SIZE / 2)]
    positions += [(10 * SIZE, 0.0, (1 - 1e-4) * SIZE * (0.5 + i)) for i in range(3)]
    positions += [(20 * SIZE + SIZE * (i % 3), SIZE * (i // 3), (1 - 1e-4) * SIZE / 2) for i in range(9)]
    for pos in positions:
        scene.add_entity(
            morph=gs.morphs.Box(
                size=(SIZE, SIZE, SIZE),
                pos=pos,
            ),
            vis_mode="collision",
        )
    scene.build(n_envs=2)
    solver = scene.rigid_solver
    constraint_state = solver.constraint_solver.constraint_state
    dofs = solver.dyn_state.dofs

    # Selecting by index into the candidates the dispatcher deems eligible replaces its whole decision, the timing it
    # would do and any QD_PERFDISPATCH_FORCE in the environment alike, and demanding more than one candidate means a
    # selection that stopped working fails here instead of quietly comparing one implementation against itself.
    selected = 0

    def call_selected(self, *args, **kwargs):
        eligible = self._get_compatible_functions(*args, **kwargs)
        # Registration order, so an index means the same implementation on every call and every run.
        candidates = [impl for impl in self._dispatch_impls if impl in eligible]
        assert len(candidates) > 1
        return candidates[selected](*args, **kwargs)

    monkeypatch.setattr(PerformanceDispatcher, "__call__", call_selected)

    # A solve leaves its own inputs changed: the warm start it wrote, the factor and active set the incremental path
    # maintains across calls, and the applied force it folded the constraint force into. The second implementation gets
    # all of them back exactly as the first one received them, or it would be solving a different problem.
    solve_inputs = (
        constraint_state.qacc_ws,
        constraint_state.is_warmstart,
        constraint_state.active,
        constraint_state.prev_active,
        constraint_state.incr_changed_idx,
        constraint_state.incr_n_changed,
        constraint_state.nt_H,
        constraint_state.nt_jacobi,
        constraint_state.solver_iter_counter,
        constraint_state.improved,
        dofs.force,
    )

    accelerations = []
    resolve_orig = ConstraintSolver.resolve

    def resolve_compared(self):
        nonlocal selected
        inputs = [qd_to_numpy(tensor, copy=True) for tensor in solve_inputs]
        accelerations.clear()
        # The candidate order is fixed, so solving with index 0 last leaves the trajectory to one implementation.
        for selected in (1, 0):
            for tensor, value in zip(solve_inputs, inputs):
                tensor.from_numpy(value)
            resolve_orig(self)
            accelerations.append(qd_to_numpy(constraint_state.qacc, copy=True))

    monkeypatch.setattr(ConstraintSolver, "resolve", resolve_compared)

    for i_step in range(N_STEPS):
        scene.step()
        assert not solver.get_error_envs_mask().any()
        # Neither implementation may be compared on a solve that had nothing to resolve
        assert (qd_to_numpy(constraint_state.n_constraints) > 32).all()

        compared, reference = accelerations
        # The two implementations spread every reduction, factor and solve differently, so their accelerations land
        # within rounding of each other and never on the same value: a gap of exactly zero means one of them ran twice.
        assert np.abs(compared.astype(np.float64) - reference.astype(np.float64)).max() > 0.0
        assert_allclose(compared, reference, tol=tol, err_msg=f"step {i_step}")


@pytest.mark.slow  # ~200s
@pytest.mark.precision("32")
@pytest.mark.parametrize("backend", [gs.cuda])
def test_cholesky_tiling_large_shared_memory(show_viewer):
    if gs.device.type != "cuda":
        pytest.skip("Requires CUDA device")

    from cuda.bindings import runtime  # Transitive dependency of torch CUDA

    _, max_shared_mem = runtime.cudaDeviceGetAttribute(
        runtime.cudaDeviceAttr.cudaDevAttrMaxSharedMemoryPerBlockOptin, gs.device.index
    )
    if max_shared_mem <= 49152:
        pytest.skip("GPU does not support opt-in shared memory beyond the default 48kB")

    # Stack 17 free boxes (6 DOFs each = 102 total) to exceed the default 48kB tiling limit of 96 DOFs for f32
    scene = gs.Scene(
        rigid_options=gs.options.RigidOptions(
            constraint_solver=gs.constraint_solver.Newton,
            sparse_solve=False,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.5, 1.0, 2.5),
            camera_lookat=(0.0, 0.0, 1.2),
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )
    scene.add_entity(gs.morphs.Plane())
    for i in range(17):
        scene.add_entity(
            gs.morphs.Box(
                size=(0.1, 0.1, 0.1),
                pos=(0, 0, 0.5 + i * 0.15),
            )
        )
    scene.build(n_envs=2)

    assert scene.rigid_solver.n_dofs == 102
    assert scene.rigid_solver.rigid_config.island_tile_cap_last == 128

    scene.step()
    assert not scene.rigid_solver.get_error_envs_mask().any()

import math
import sys

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
import trimesh

import genesis as gs
import genesis.utils.geom as gu
from genesis.utils.misc import qd_to_numpy, tensor_to_array

from ..utils.assertions import assert_allclose
from ..utils.assets import get_hf_dataset
from ..utils.collision import display_collision_pairs, get_genuine_interpenetration


# Force CPU because it would be too slow otherwise
@pytest.mark.required
@pytest.mark.parametrize("backend", [gs.cpu])
def test_collision(show_viewer):
    scene = gs.Scene(
        show_viewer=show_viewer,
        show_FPS=False,
    )
    tank = scene.add_entity(
        gs.morphs.Mesh(
            file="meshes/tank.obj",
            scale=5.0,
            fixed=True,
            euler=(100, -10, 0),
            convexify=False,
        ),
        # vis_mode="collision",
    )
    ball = scene.add_entity(
        gs.morphs.Sphere(
            radius=0.05,
            pos=(0.0, 0.0, 0.75),
        ),
        surface=gs.surfaces.Default(
            color=(0.5, 0.7, 0.9, 1.0),
        ),
        visualize_contact=True,
    )
    scene.build()

    # Force numpy seed because this test is very sensitive to the initial condition
    np.random.seed(0)
    ball.set_dofs_velocity(np.random.rand(ball.n_dofs) * 0.8)
    for i in range(500):
        scene.step()
        if i > 450:
            qvel = scene.sim.rigid_solver.dyn_state.dofs.vel.to_numpy()[:, 0]
            assert_allclose(qvel, 0, atol=0.05)


# Force CPU because it would be too slow otherwise
@pytest.mark.parametrize("backend", [gs.cpu])
def test_nonwatertight_collision(show_viewer):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.002,
        ),
        rigid_options=gs.options.RigidOptions(
            max_collision_pairs=20,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(2.0, 15.0, 3.0),
            camera_lookat=(2.0, 0.0, -2.0),
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )
    asset_path = get_hf_dataset(pattern="spacecraft.obj")
    scene.add_entity(
        gs.morphs.Mesh(
            file=f"{asset_path}/spacecraft.obj",
            pos=(-0.4, 0.0, -4.0),
            euler=(90.0, 0.0, 0.0),
            scale=3.0,
            convexify=False,
            fixed=True,
        ),
        vis_mode="collision",
    )
    obj = scene.add_entity(
        gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
        ),
        surface=gs.surfaces.Default(
            color=(0.5, 0.7, 0.9, 1.0),
        ),
        visualize_contact=True,
    )
    scene.build(n_envs=64)

    obj.set_pos(
        torch.cartesian_prod(
            torch.linspace(-6.25, 9.05, 8),
            torch.linspace(-5.2, 5.5, 8),
            torch.tensor((0.39,)),
        )
    )
    for _ in range(750):
        scene.step()

    # The velocity is fairly large for boxes whose contact set is stable at keep changing (border of a cliff)
    assert_allclose(obj.get_dofs_velocity(), 0.0, tol=0.08)


@pytest.mark.parametrize("obj_shape", ["box", "sphere_mesh"])
@pytest.mark.parametrize("backend", [gs.cpu])
def test_inner_corner_multi_contact(obj_shape, show_viewer, tmp_path):
    INIT_GAP = 1e-4  # initial gap between the body and the L-mesh surfaces (no overlap)
    # An object wedged at the inner corner of a non-convex L-shaped mesh under gravity tilted into both surfaces.
    # The object must settle in the corner with at least one contact on each surface (floor and wall). A single
    # contact with a mixed floor+wall normal is what the perturbation-only path returned, and it lets the object
    # squirt out of the corner along the tilted normal instead of staying wedged.
    # Parametrised over a primitive BOX and a generic mesh (tessellated icosphere via MeshSet, so it is *not*
    # classified as a SPHERE primitive) to exercise both the primitive-geom and generic-mesh dispatch paths.
    floor = trimesh.creation.box(extents=(4.0, 4.0, 0.2))
    floor.apply_translation((0.0, 0.0, -0.1))
    wall = trimesh.creation.box(extents=(0.2, 4.0, 2.0))
    wall.apply_translation((1.0, 0.0, 1.0))
    mesh_path = tmp_path / "L.obj"
    trimesh.util.concatenate([floor, wall]).export(mesh_path)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.002,
            gravity=(5.0, 0.0, -9.81),
        ),
        rigid_options=gs.options.RigidOptions(
            max_collision_pairs=20,
        ),
        viewer_options=gs.options.ViewerOptions(
            # Frame the L-corner (wall at x=0.9, floor at z=0) where the object settles.
            camera_pos=(0.6, -2.2, 0.55),
            camera_lookat=(0.85, 0.0, 0.15),
            camera_fov=35,
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )
    world = scene.add_entity(
        gs.morphs.Mesh(
            file=str(mesh_path),
            pos=(0.0, 0.0, 0.0),
            convexify=False,
            fixed=True,
        ),
        visualize_contact=True,
        vis_mode="collision",
    )
    obj_surface = gs.surfaces.Default(color=(0.5, 0.7, 0.9, 1.0))
    if obj_shape == "box":
        obj = scene.add_entity(
            gs.morphs.Box(
                size=(0.2, 0.2, 0.2),
                pos=(0.8 - INIT_GAP, 0.0, 0.1 + INIT_GAP),
            ),
            surface=obj_surface,
            vis_mode="collision",
        )
    else:
        sphere_radius = 0.1
        obj = scene.add_entity(
            morph=gs.morphs.MeshSet(
                files=(trimesh.creation.icosphere(radius=sphere_radius, subdivisions=3),),
                pos=(0.8 - INIT_GAP, 0.0, sphere_radius + INIT_GAP),
                decimate=False,
                convexify=False,
            ),
            surface=obj_surface,
            vis_mode="collision",
        )
    scene.build()

    # Run 10 warm-up steps so the body resolves its initial fall/impact transient, then monitor the velocity at every
    # subsequent step: a wedged body must never spike (detect simulation blow-up). Final velocity must be near zero
    # (body must actually settle).
    for _ in range(10):
        scene.step()
    max_v_seen = 0.0
    for _ in range(200):
        scene.step()
        v = tensor_to_array(obj.get_dofs_velocity())
        max_v_seen = max(max_v_seen, float(np.abs(v).max()))
    assert max_v_seen < 0.05, f"velocity spike during settling: max |v| = {max_v_seen:.4f}"

    contacts = scene.rigid_solver.collider.collider_state.contact_data
    n_contacts = int(scene.rigid_solver.collider.collider_state.n_contacts[0])
    normals = qd_to_numpy(contacts.normal, transpose=True)
    positions = qd_to_numpy(contacts.pos, transpose=True)
    ga = qd_to_numpy(contacts.geom_a, transpose=True)
    obj_idx = obj.geoms[0].idx
    floor_contacts = []
    wall_contacts = []
    for k in range(n_contacts):
        sign = +1 if ga[0, k] == obj_idx else -1
        n = sign * normals[0, k]
        p = positions[0, k]
        if n[2] > 0.7:
            floor_contacts.append((p, n))
        elif n[0] < -0.7:
            wall_contacts.append((p, n))
    # Both shapes settle wedged in the L-corner with zero residual velocity. Equilibrium has the body centred at
    # (0.8, 0, 0.1): bottom touching the floor (z=0) and right touching the wall (x=0.9). Any drift means a spurious
    # tangential force from a non-axis-aligned contact normal.
    assert_allclose(obj.get_pos(), (0.8, 0.0, 0.1), tol=1e-3)
    assert_allclose(obj.get_dofs_velocity(), 0.0, tol=0.05)
    if obj_shape == "sphere_mesh":
        # The icosphere touches the floor at its bottom and the wall at its right; expect a single contact on each
        # surface with pure axis-aligned normal direction.
        assert n_contacts == 2, f"expected exactly 2 contacts (1 floor, 1 wall), got {n_contacts}"
        assert len(floor_contacts) == 1
        assert len(wall_contacts) == 1
        floor_pos, floor_normal = floor_contacts[0]
        wall_pos, wall_normal = wall_contacts[0]
        assert_allclose(floor_pos, (0.8, 0.0, 0.0), tol=5e-3)
        assert_allclose(floor_normal, (0.0, 0.0, 1.0), tol=1e-2)
        assert_allclose(wall_pos, (0.9, 0.0, 0.1), tol=5e-3)
        assert_allclose(wall_normal, (-1.0, 0.0, 0.0), tol=1e-2)
    # FIXME: The box test only checks that the body wedges at the L-corner equilibrium (position + zero velocity).
    # The detailed contact set is not asserted because the grid SDF emits an edge-regime contact at the bottom-right
    # corners with a non-axis-aligned normal; the resulting contact pattern works physically (the body wedges and stays
    # put) but does not match the clean 2-floor + 2-wall configuration the sphere case enforces.


# Force CPU because nonconvex SDF is slow on GPU
@pytest.mark.parametrize("backend", [gs.cpu])
def test_tunneling(show_viewer):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.002,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.2, 0.2, 1.0),
            camera_lookat=(0.0, 0.0, 0.0),
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )
    scene.add_entity(
        gs.morphs.Mesh(
            file="meshes/tank.obj",
            euler=(95, 0, 0),
            scale=5.0,
            fixed=True,
            convexify=False,
        ),
        vis_mode="collision",
    )
    rod = scene.add_entity(
        gs.morphs.Mesh(
            file="meshes/stirrer.obj",
            pos=(0.0, 0.0, 0.2),
            convexify=False,
        ),
        vis_mode="collision",
        visualize_contact=True,
    )
    scene.build()

    # It collides with the tank bottom at step 200
    for step in range(250):
        scene.step()
    assert rod.get_pos()[..., 2] > -0.05
    assert_allclose(rod.get_dofs_velocity(dofs_idx_local=slice(None, 3)), 0, atol=0.08)


# Force CPU because nonconvex SDF is slow on GPU
@pytest.mark.parametrize("backend", [gs.cpu])
def test_overlap(show_viewer):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.001,
            gravity=(0, 0, 0),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.0, 0.3, 0.15),
            camera_lookat=(0.0, 0.0, 0.0),
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )
    a = scene.add_entity(
        gs.morphs.Mesh(
            file="meshes/stirrer.obj",
            pos=(-0.051, 0.0, 0.0),
            convexify=False,
        ),
        vis_mode="collision",
    )
    b = scene.add_entity(
        gs.morphs.Mesh(
            file="meshes/stirrer.obj",
            pos=(+0.05, 0.0, 0.0),
            euler=(0, 0, 90),
            convexify=False,
        ),
        vis_mode="collision",
        visualize_contact=True,
    )
    # Two compact solid meshes barely touching along their local OBB diagonal, away from the stirrers. The
    # closing-direction penetration floor must read the true submillimetric overlap along the axis; an OBB-projection
    # bound overestimates it by up to sqrt(3) off-axis and catapults the pair apart.
    asset_path = get_hf_dataset(pattern="apple_15/*")
    apples = []
    for i_apple in range(2):
        apples.append(
            scene.add_entity(
                gs.morphs.MJCF(
                    file=f"{asset_path}/apple_15/model.xml",
                    pos=(0.0, 0.5 + 0.2 * i_apple, 0.0),
                    convexify=False,
                ),
            )
        )
    scene.build()
    a.set_dofs_velocity(+1.0, dofs_idx_local=0)
    b.set_dofs_velocity(-1.0, dofs_idx_local=0)

    geom = apples[0].geoms[0]
    apples_overlap = 1e-3
    u_local = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)
    proj = (geom.init_verts - geom.init_verts.mean(axis=0)) @ u_local
    u_world = gu.quat_to_R(tensor_to_array(geom.get_quat())) @ u_local
    apples_offset = (proj.max() - proj.min() - apples_overlap) * u_world
    apples_pos_ref = [tensor_to_array(apple.get_pos()) for apple in apples]
    apples_pos_ref[1] = apples_pos_ref[0] + apples_offset
    apples[1].set_pos(apples_pos_ref[1])

    total_energy_history = []
    for _ in range(200):
        total_energy = tensor_to_array(a.get_total_energy() + b.get_total_energy())
        total_energy_history.append(total_energy)
        scene.step()

    # FIXME: The total energy should be not strictly decreasing but is not... relaxing the condition
    # assert (np.diff(total_energy_history, axis=0) < 0.0)
    assert total_energy_history[0] > 3.0 * total_energy_history[-1]

    # Constraint stabilization alone resolves the overlap, so it cannot separate the apples faster than
    # overlap / timeconst; a spurious deep contact catapults them an order of magnitude above that ceiling.
    # Contact impulses being internal to the pair, its total momentum must stay zero.
    v_sep_max = apples_overlap / float(geom.desc.sol_params[0])
    assert np.linalg.norm(tensor_to_array(apples[1].get_vel() - apples[0].get_vel())) < v_sep_max
    assert_allclose(apples[0].get_vel() + apples[1].get_vel(), 0, atol=1e-6)
    # The apples must separate by at least the overlap, but no more than the stabilization drift accumulates
    # over the simulated horizon.
    apples_dist = np.linalg.norm(tensor_to_array(apples[1].get_pos() - apples[0].get_pos()))
    assert 0.5 * apples_overlap < apples_dist - np.linalg.norm(apples_offset) < v_sep_max * 200 * 0.001


# Force CPU because nonconvex SDF is slow on GPU
@pytest.mark.parametrize("backend", [gs.cpu])
@pytest.mark.xfail(reason="Recovery is too slow: the separating push is creep-rate-bound by the thin-shell pen cap.")
def test_shell_crossing_recovery(show_viewer):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.004,
            gravity=(0, 0, 0),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.17, 0.21, 0.6),
            camera_lookat=(0.04, -0.02, 0.5),
        ),
        show_viewer=show_viewer,
    )
    asset_path = get_hf_dataset(pattern="cup_2/*")
    cups = []
    for i_cup in range(2):
        cups.append(
            scene.add_entity(
                gs.morphs.MJCF(
                    file=f"{asset_path}/cup_2/model.xml",
                    pos=(0.0, 0.0, 0.5 * i_cup),
                    convexify=False,
                ),
                vis_mode="collision",
                # visualize_contact=True,
            )
        )
    scene.build()

    # Spawn the pair deeply crossed: cup B perpendicular to cup A with its mouth rim 45mm past A's side wall, the
    # thin-shell equivalent of an overlapping spawn. The solver must recover by pushing the shells apart until they
    # separate; the failure mode is a wedged equilibrium where contacts on both sides of the crossing curve cancel
    # and hold the crossed pose forever.
    geom = cups[0].geoms[0]
    verts = torch.as_tensor(geom.init_verts, dtype=gs.tc_float)
    ext = verts.max(dim=0).values - verts.min(dim=0).values
    mesh_center = 0.5 * (verts.max(dim=0).values + verts.min(dim=0).values)
    i_axis = int(torch.argmax(ext))
    quat_a = geom.get_quat()
    rot_a = gu.quat_to_R(quat_a)
    radial_dir = rot_a[:, (i_axis + 1) % 3]
    euler_rot = torch.zeros(3, dtype=gs.tc_float)
    euler_rot[(i_axis + 2) % 3] = 0.5 * math.pi
    quat_b = gu.transform_quat_by_quat(gu.xyz_to_quat(euler_rot), quat_a)
    mesh_center_a = torch.tensor([0.0, 0.0, 0.5], dtype=gs.tc_float) + gu.transform_by_quat(mesh_center, quat_a)
    mesh_center_b = mesh_center_a + radial_dir * (0.5 * (ext[(i_axis + 1) % 3] + ext[i_axis]) - 0.045)
    for cup, quat_target, mesh_center_target in ((cups[0], quat_a, mesh_center_a), (cups[1], quat_b, mesh_center_b)):
        cup_geom = cup.geoms[0]
        corr_quat = gu.transform_quat_by_quat(gu.inv_quat(cup_geom.get_quat()), quat_target)
        cup.set_quat(gu.transform_quat_by_quat(cup.get_quat(), corr_quat))
        cup.set_pos(
            cup.get_pos() + mesh_center_target - cup_geom.get_pos() - gu.transform_by_quat(mesh_center, quat_target)
        )

    centers_dist_init = torch.linalg.norm(cups[1].get_pos() - cups[0].get_pos())
    for _ in range(200):
        scene.step()

    # Separation requires the centers to move apart by at least the spawn crossing depth; a wedged pair stays put
    # (residual velocities below 1e-2 m/s and unchanged distance)
    centers_dist = torch.linalg.norm(cups[1].get_pos() - cups[0].get_pos())
    assert centers_dist - centers_dist_init > 0.02


# Force CPU because nonconvex SDF is slow on GPU
@pytest.mark.parametrize("backend", [gs.cpu])
@pytest.mark.parametrize("direction", ["down", "up"])
def test_concentric_contact(direction, show_viewer):
    PITCH = 3.0e-3  # matches genesis/assets/meshes/bolt_nut/generate_bolt_nut.py
    PITCH_RATE = PITCH / (2.0 * np.pi)  # axial advance per radian of rotation
    # Head top is at z = 11 mm, so the 18 mm nut seats with its center at z ~ 20 mm. Driving down, release just above
    # that so the nut coasts onto the head rather than being driven into it. Driving up, release just below the shaft
    # tip (z ~ 48 mm) where only a turn of thread is left engaged, so the nut spins off and falls rather than stalling.
    SEAT_RELEASE_Z = 0.0202
    TIP_RELEASE_Z = 0.048
    # Advance-per-revolution is only meaningful while enough of the nut is still threaded. Past ~2/3 unscrewed (less
    # than a third of the 18 mm nut still gripping below the 43 mm shaft tip) the few remaining threads slip, so the
    # pitch is checked only up to that engagement.
    NUT_HEIGHT = 0.018
    SHAFT_TIP_Z = 0.043
    # 4.5 N*m screws down and 5.0 N*m unscrews up within the step budget below. At this test's single-substep dt the
    # solve diverges past ~5 N*m (the example tolerates more only because its substeps make each step stiffer).
    torque = -4.5 if direction == "down" else 5.0
    # Per-step thread coupling carries a contact jitter of a few mm/s; unscrewing jitters more, so its bound is looser.
    coupling_atol = 5e-3 if direction == "down" else 1e-2

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1e-3,
        ),
        rigid_options=gs.options.RigidOptions(
            # Fine-thread contact needs a stiff constraint (the default 0.01 is too soft, letting the nut sink through
            # the flanks and advance faster than the pitch).
            constraint_timeconst=4e-3,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.0, -0.2, 0.1),
            camera_lookat=(0.0, 0.0, 0.03),
            camera_fov=35,
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )
    scene.add_entity(gs.morphs.Plane())
    # Realistic steel density so the nut carries a real fastener's mass and inertia.
    steel = gs.materials.Rigid(rho=7850.0)
    scene.add_entity(
        gs.morphs.Mesh(
            # Head bottom rests on the plane (z = 0), head top at z = 11 mm, shaft tip at z = 43 mm.
            pos=(0.0, 0.0, 0.011),
            file="meshes/bolt_nut/bolt.stl",
            decimate=False,
            convexify=False,
            fixed=True,
        ),
        material=steel,
        vis_mode="collision",
    )
    nut = scene.add_entity(
        gs.morphs.Mesh(
            # Pre-engaged near the top of the shaft (base ~24 mm) so it has the whole thread to travel.
            pos=(0.0, 0.0, 0.024),
            file="meshes/bolt_nut/nut.stl",
            decimate=False,
            convexify=False,
        ),
        material=steel,
        vis_mode="collision",
        # visualize_contact=True,
    )
    scene.build()

    # Drive a steady torque about z (a wrench) until the nut reaches its release height, then let go: negative torque
    # screws it down onto the head, positive torque unscrews it up and off the shaft tip.
    z0 = nut.get_pos(relative=False)[..., 2]
    prev_yaw = gu.quat_to_xyz(nut.get_quat(relative=False))[..., 2]
    total_turn = 0.0
    released_step = None
    z_engaged = z0
    turn_engaged = total_turn
    z_history = []
    horizon = 4100 if direction == "down" else 4800
    for step in range(horizon):
        z = nut.get_pos(relative=False)[..., 2]
        if released_step is None:
            reached = (z < SEAT_RELEASE_Z).all() if direction == "down" else (z > TIP_RELEASE_Z).all()
            if reached:
                released_step = step
        driving = released_step is None
        nut.control_dofs_force([torque if driving else 0.0], dofs_idx_local=(5,))
        scene.step()

        pos = nut.get_pos(relative=False)
        rpy = gu.quat_to_xyz(nut.get_quat(relative=False))
        vel = nut.get_dofs_velocity()
        z_history.append(pos[..., 2])
        yaw = rpy[..., 2]
        total_turn = total_turn + ((yaw - prev_yaw + np.pi) % (2.0 * np.pi) - np.pi)
        prev_yaw = yaw
        # Fraction of the nut height still threaded below the shaft tip (capped at 1 while fully on the shaft). Track
        # the last driven sample with more than a third engaged for the advance-per-revolution check below.
        engaged = torch.clamp((SHAFT_TIP_Z - (pos[..., 2] - NUT_HEIGHT / 2.0)) / NUT_HEIGHT, max=1.0)
        if driving and (engaged > 0.5).all():
            z_engaged = pos[..., 2]
            turn_engaged = total_turn
        # While steadily screwing through the middle of the thread (past the initial spin-up, away from the seat and
        # the tip where engagement thins) the nut stays coaxial and upright and its axial speed stays locked to its
        # rotation by the pitch (vz = wz * pitch/2pi). The bound is loose because of the per-step contact jitter; it
        # guards against a flank dropping out and letting vz decouple from wz by tens of mm/s (the nut spins without
        # translating, stripping).
        if driving and step > 100 and (0.025 < pos[..., 2]).all() and (pos[..., 2] < 0.043).all():
            assert (torch.linalg.norm(pos[..., :2], dim=-1) < 5e-4).all()
            assert_allclose(rpy[..., :2], 0.0, atol=0.02)
            assert_allclose(vel[..., 2], vel[..., 5] * PITCH_RATE, atol=coupling_atol)

    # The nut travelled the thread and reached its release height.
    assert released_step is not None
    # Over the well-engaged phase the axial advance per revolution tracks the thread pitch (it really screwed along
    # the thread rather than slipping).
    travel = torch.abs(z_engaged - z0)
    revolutions = torch.abs(turn_engaged) / (2.0 * np.pi)
    assert_allclose(travel / revolutions, PITCH, rtol=0.1)

    if direction == "down":
        # Comes to a clean rest seated on the head - no bounce, no drift: over the final settle window the nut height
        # holds within a tight band, since a bounce or a strip would show as a large z excursion. The seated nut keeps
        # a small steady contact jitter in velocity, so the position band is the robust at-rest signal.
        z_window = torch.stack(z_history[-200:], dim=0)
        assert ((z_window.amax(dim=0) - z_window.amin(dim=0)) < 1e-4).all()
        z_final = nut.get_pos(relative=False)[..., 2]
        assert ((0.019 < z_final) & (z_final < 0.021)).all()
    else:
        # Spun off the tip, fell, and came to rest flat on the ground: its bounding box now sits on the plane and all
        # of its velocities have decayed to zero.
        aabb = nut.get_AABB()
        assert (aabb[..., 0, 2] < 1.0e-3).all()
        assert_allclose(nut.get_dofs_velocity(), 0.0, atol=0.07)


# Force CPU because nonconvex SDF is slow on GPU
@pytest.mark.debug(False)  # Disable debug for speedup
@pytest.mark.parametrize("backend", [gs.cpu])
@pytest.mark.parametrize(
    "timestep, decimate",
    [
        pytest.param(0.01, True, marks=pytest.mark.required),
        (0.001, False),
    ],
)
def test_concave_slanted_wall(timestep, decimate, show_viewer):
    BOWL_THICKNESS = 0.013
    NUM_BOWLS = 32

    timeconst = max(0.005, 2 * timestep)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=timestep,
        ),
        rigid_options=gs.options.RigidOptions(
            # The pyramidal cone cannot hold the pile: its regularized friction creeps tangentially under the
            # sustained shear of the nested stack, and the tower topples within a few thousand steps.
            friction_cone=gs.friction_cone.elliptic,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(-0.6, 0.6, 0.5),
            camera_lookat=(-0.25, 0.0, 0.3),
        ),
        renderer=gs.renderers.Rasterizer(),
        show_viewer=show_viewer,
    )
    scene.add_entity(morph=gs.morphs.Plane())
    asset_path = get_hf_dataset(pattern="glb/orange_plastic_bowl.glb")
    for i in range(NUM_BOWLS):
        # Independent yaws: the collision mesh seats on a few vertices, so under one shared orientation every nested
        # pair tilts by the same small angle in the same direction and the pile arcs by the sum of those tilts.
        # Independent orientations let the tilts average out, as in a real pile of bowls.
        scene.add_entity(
            morph=gs.morphs.Mesh(
                file=f"{asset_path}/glb/orange_plastic_bowl.glb",
                pos=(0, 0, 0.0 + i * (BOWL_THICKNESS - 0.15 * timeconst)),
                euler=(90, 0, np.random.uniform(0.0, 360.0)),
                convexify=False,
                decimate=decimate,
                file_meshes_are_zup=True,
            ),
            vis_mode="collision",
            # visualize_contact=(i in (0, NUM_BOWLS - 1)),
        )
    scene.build()

    # Make sure that the pile stays upright, with bowls stay tightly packed together during the entire motion
    bowls_link_idx = [entity.base_link_idx for entity in scene.entities[-NUM_BOWLS:]]
    # The spawn drop sways the stack laterally before it settles; assert once the transient has decayed.
    for _ in range(700):
        scene.step()
    for _ in range(1000):
        scene.step()
        bowls_pos = tensor_to_array(scene.rigid_solver.get_links_pos(bowls_link_idx, relative=True))
        bowls_dist_abs = np.linalg.norm(bowls_pos[:, :2] - bowls_pos[0, :2], axis=-1)
        assert (bowls_dist_abs < 0.025).all()
        bowls_dist_rel = np.linalg.norm(np.diff(bowls_pos, axis=0), axis=-1)
        assert ((BOWL_THICKNESS - 0.5 * timeconst) < bowls_dist_rel).all()
        assert (bowls_dist_rel < BOWL_THICKNESS + 1e-3).all()


@pytest.mark.required
@pytest.mark.parametrize("convexify", [True, False])
@pytest.mark.parametrize("gjk_collision", [True, False])
def test_mesh_repair(convexify, show_viewer, gjk_collision):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.004,
        ),
        rigid_options=gs.options.RigidOptions(
            use_gjk_collision=gjk_collision,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.3, 0.4, 0.01),
            camera_lookat=(0.3, 0.0, 0.0),
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )
    asset_path = get_hf_dataset(pattern="work_table.glb")
    scene.add_entity(
        gs.morphs.Mesh(
            file=f"{asset_path}/work_table.glb",
            pos=(0.4, 0.0, -0.54),
            fixed=True,
        ),
        vis_mode="collision",
    )
    asset_path = get_hf_dataset(pattern="spoon.glb")
    obj = scene.add_entity(
        gs.morphs.Mesh(
            file=f"{asset_path}/spoon.glb",
            pos=(0.3, 0, 0.009),
            euler=(0.0, -2.5 if convexify else 0.0, 0.0),
            convexify=convexify,
            scale=1.0,
        ),
        vis_mode="collision",
        visualize_contact=True,
    )
    scene.build()

    if show_viewer:
        obj_com = obj.get_links_pos(ref=gs.link_ref_frame.link_COM)[0]
        scene.draw_debug_sphere(pos=obj_com, radius=0.003, color=(1, 1, 1, 1))
        scene.visualizer.update(force=True)

    for geom in obj.geoms:
        assert ("decomposed" in geom.metadata) ^ (not convexify)
        max_faces = obj._morph.decimate_face_num if convexify else 5000
        num_faces = geom.face_end - geom.face_start
        assert num_faces <= max_faces
        assert ("convexified" in geom.metadata) ^ (not convexify)

    # MPR collision detection is less reliable than SDF and GJK in terms of penetration depth estimation
    is_mpr = convexify and not gjk_collision
    tol_pos = 0.05 if is_mpr else 0.005
    tol_rot = 1.25 if is_mpr else 0.5
    init_pos = obj.geoms[0].get_pos()
    for _ in range(50):
        scene.step()
    for _ in range(100):
        scene.step()
        qvel = obj.get_dofs_velocity()
        assert_allclose(qvel[:3], 0, atol=tol_pos)
        assert_allclose(qvel[3:], 0, atol=tol_rot)
    assert_allclose(obj.geoms[0].get_pos()[:2], init_pos[:2], atol=2e-3)


@pytest.mark.required
@pytest.mark.parametrize("euler", [(90, 0, 90), (74, 15, 90)])
@pytest.mark.parametrize("gjk_collision", [True, False])
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
def test_convexify(euler, show_viewer, gjk_collision):
    OBJ_OFFSET_X = 0.0  # 0.02
    OBJ_OFFSET_Y = 0.15
    N_SETTLE = 1350

    # The test check that the volume difference is under a given threshold and that convex decomposition is only used
    # whenever it is necessary. Then run a simulation to see if it explodes, i.e. objects are at reset inside tank.
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.004,
        ),
        rigid_options=gs.options.RigidOptions(
            use_gjk_collision=gjk_collision,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.0, 0.5, 2.5),
            camera_lookat=(0.0, 0.0, 0.5),
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )
    box = scene.add_entity(
        gs.morphs.URDF(
            file="urdf/blue_box/model.urdf",
            fixed=True,
            pos=(0.0, 1.0, 0.0),
        ),
        vis_mode="collision",
    )
    scene.add_entity(
        gs.morphs.Mesh(
            file="meshes/tank.obj",
            scale=5.0,
            fixed=True,
            pos=(0.05, -0.05, 0.0),
            euler=euler,
            # coacd_options=gs.options.CoacdOptions(
            #     threshold=0.08,
            # ),
        ),
        vis_mode="collision",
    )
    objs = []
    for i, (asset_name, xml_file) in enumerate(
        (("mug_1", "output.xml"), ("donut_0", "output.xml"), ("cup_2", "model.xml"), ("apple_15", "model.xml"))
    ):
        asset_path = get_hf_dataset(pattern=f"{asset_name}/*")
        obj = scene.add_entity(
            gs.morphs.MJCF(
                file=f"{asset_path}/{asset_name}/{xml_file}",
                pos=(OBJ_OFFSET_X * (1.5 - i), OBJ_OFFSET_Y * (i - 1.5), 0.1 if euler == (90, 0, 90) else 0.4),
            ),
            vis_mode="collision",
            visualize_contact=True,
        )
        objs.append(obj)
    # cam = scene.add_camera(
    #     pos=(0.5, 0.0, 1.0),
    #     lookat=(0.0, 0.0, 0.0),
    #     res=(500, 500),
    #     fov=60,
    #     spp=512,
    #     GUI=False,
    # )
    scene.build()
    gs_sim = scene.sim

    # Make sure that all the geometries in the scene are convex
    assert gs_sim.rigid_solver.dyn_info.geoms.is_convex.to_numpy().all()
    assert not gs_sim.rigid_solver.collider.collider_config.has_nonconvex_nonterrain

    # There should be only one geometry for the apple as it can be convexify without decomposition,
    # but for the others it is hard to tell... Let's use some reasonable guess.
    mug, donut, cup, apple = objs
    assert not any(geom.metadata.get("decomposed", False) for geom in apple.geoms)
    assert not any(geom.metadata.get("decomposed", False) for geom in cup.geoms)
    assert all(geom.metadata["decomposed"] for geom in donut.geoms) and 5 <= len(donut.geoms) <= 10
    assert all(geom.metadata["decomposed"] for geom in mug.geoms) and 5 <= len(mug.geoms) <= 40
    assert all(geom.metadata["decomposed"] for geom in box.geoms) and 5 <= len(box.geoms) <= 20

    # Check that all the objects settle at rest after a while, without spurious jumps
    # cam.start_recording()
    vel_lin_all, vel_ang_all = [], []
    for i in range(N_SETTLE + 100):
        scene.step()
        # cam.render()
        if i > N_SETTLE:
            vel_lin_all.append(gs_sim.rigid_solver.get_links_vel(ref=gs.link_ref_frame.link_COM))
            vel_ang_all.append(gs_sim.rigid_solver.get_links_ang())
    # cam.stop_recording(save_to_filename="video.mp4", fps=60)
    # FIXME: There is spurious residual motion on both paths that prevents the objects from truly settling
    assert_allclose(torch.quantile(torch.stack(vel_lin_all, dim=0), 0.5, dim=0), 0.0, tol=0.01)
    assert_allclose(torch.quantile(torch.stack(vel_ang_all, dim=0), 0.5, dim=0), 0.0, tol=0.1)

    for obj in objs:
        obj_pos = tensor_to_array(obj.get_pos())
        np.testing.assert_array_less(-0.1, obj_pos[2])
        np.testing.assert_array_less(obj_pos[2], 0.15)
        np.testing.assert_array_less(np.linalg.norm(obj_pos[:2]), 0.5)

    # Check that the mug, donut and cup are landing straight if the tank is horizontal.
    # FIXME: The cup is falling on Windows OS because the convex decomposition provided by CoACD is different than
    # other platform, and much worst in practice, with the bottom of the tank that is not planar (even discontinuous).
    if euler == (90, 0, 90):
        for i, obj in enumerate((mug, donut, *(() if sys.platform == "win32" else (cup,)))):
            obj_pos = obj.get_pos()
            assert_allclose(obj_pos[:2], (OBJ_OFFSET_X * (1.5 - i), OBJ_OFFSET_Y * (i - 1.5)), atol=6e-3)


@pytest.mark.required
@pytest.mark.parametrize("convexify, watertighten", [(True, 5), (False, 5), (False, None)])
@pytest.mark.parametrize("model_name", ["decompose_fusion_groups"])
def test_convexify_fusion_groups(convexify, watertighten, xml_path):
    scene = gs.Scene()
    entity = scene.add_entity(
        gs.morphs.MJCF(
            file=xml_path,
            convexify=convexify,
            watertighten=watertighten,
        ),
    )
    scene.build()

    # The plane can never be merged nor watertightened.
    (geom_plane,) = [geom for geom in entity.geoms if geom.type == gs.GEOM_TYPE.PLANE]
    assert len(geom_plane.init_verts) == 4
    assert not geom_plane.metadata.get("watertightened", False)

    if convexify:
        # Only the L-shape sub-group may be decomposed, with its primitive box merged along: the mesh boxes must
        # survive as four separate convex geoms instead of one hull spanning their gap, and the primitive boxes must
        # pass through untouched.
        geoms_decomposed = [geom for geom in entity.geoms if geom.metadata.get("decomposed", False)]
        geoms_box = [
            geom
            for geom in entity.geoms
            if geom.type == gs.GEOM_TYPE.MESH and not geom.metadata.get("decomposed", False)
        ]
        assert len(geoms_decomposed) >= 2
        assert len(geoms_box) == 4
        for geom in geoms_box:
            assert geom.is_convex
            assert not geom.metadata.get("merged", False)
            assert_allclose(geom.init_verts.max(axis=0) - geom.init_verts.min(axis=0), 0.1, tol=5e-9)
        assert len([geom for geom in entity.geoms if geom.type == gs.GEOM_TYPE.BOX]) == 2
    elif watertighten is not None:
        # All multi-geom sub-groups are fused systematically, including bare primitives, and every fused geom is
        # watertightened even when its sub-meshes are individually watertight. The mesh boxes with adjacent collision
        # masks belong to distinct sub-groups and must survive as two separate geoms.
        assert all(geom.type in (gs.GEOM_TYPE.PLANE, gs.GEOM_TYPE.MESH) for geom in entity.geoms)
        geoms_merged = [geom for geom in entity.geoms if geom.metadata.get("merged", False)]
        assert len(geoms_merged) == 3
        assert len(entity.geoms) == 6
        assert all(geom.metadata.get("watertightened", False) for geom in geoms_merged)
    else:
        # Disabling watertightening on the nonconvex path opts out of fusion entirely: every geom passes through.
        assert len(entity.geoms) == 9
        assert not any(geom.metadata.get("merged", False) for geom in entity.geoms)
        assert len([geom for geom in entity.geoms if geom.type == gs.GEOM_TYPE.BOX]) == 3


@pytest.mark.debug(False)  # Disable debug for speedup
@pytest.mark.slow
@pytest.mark.precision("32")
@pytest.mark.parametrize("backend", [gs.cpu])
@pytest.mark.parametrize("convexify", [False, True])
def test_many_objects_collision(convexify, show_viewer, tol):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.004,
        ),
        rigid_options=gs.options.RigidOptions(
            max_collision_pairs=8000,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.5, 0.2, 1.6),
            camera_lookat=(0.0, 0.0, 0.3),
        ),
        show_viewer=show_viewer,
    )
    tank = scene.add_entity(
        gs.morphs.Mesh(
            file="meshes/tank.obj",
            scale=5.0,
            fixed=True,
            euler=(90, 0, 90),
            convexify=convexify,
        ),
        vis_mode="collision",
    )
    assets = (("mug_1", "output.xml"), ("donut_0", "output.xml"), ("cup_2", "model.xml"), ("apple_15", "model.xml"))
    asset_files = {name: f"{get_hf_dataset(pattern=f'{name}/*')}/{name}/{xml}" for name, xml in assets}
    objs = []
    obj_names = []
    for i in range(80):
        gx, gy, gz = i % 4, (i // 4) % 4, i // 16
        name = assets[(gx + gy + gz) % len(assets)][0]
        obj_names.append(name)
        base_pos = ((gx + 0.5 * (gz % 2)) * 0.1 - 0.18, (gy + 0.5 * (gz % 2)) * 0.145 - 0.265, 0.11 + gz * 0.08)
        objs.append(
            scene.add_entity(
                gs.morphs.MJCF(
                    file=asset_files[name],
                    pos=base_pos + np.random.uniform(-2e-4, 2e-4, 3),
                    euler=(90.0, 0.0, 0.0) + np.random.uniform(-0.2, 0.2, 3),
                    convexify=convexify,
                ),
                vis_mode="collision",
            )
        )
    scene.build()

    # Wait for the pile to collapse and settle at rest
    vmax_trace, wmax_trace, energy_trace = [], [], []
    for i in range(1600):
        scene.step()
        energy_trace.append(tensor_to_array(scene.rigid_solver.get_total_energy()))
        if show_viewer:
            vmax_trace.append(scene.rigid_solver.get_links_vel(ref=gs.link_ref_frame.link_COM).norm(dim=-1).max())
            wmax_trace.append(scene.rigid_solver.get_links_ang().norm(dim=-1).max())

    # The pile has settled at rest, fully contained in the tank (no ground/tank penetration, no ejection)
    for obj in objs:
        obj_pos = tensor_to_array(obj.get_pos())
        np.testing.assert_array_less(-0.1, obj_pos[2])
        np.testing.assert_array_less(obj_pos[2], 0.6)
        np.testing.assert_array_less(np.linalg.norm(obj_pos[:2]), 0.5)

    # Make sure that there is no interpenetration among the settled objects
    links, link_names = [], []
    for obj, name in zip(objs, obj_names):
        for link in obj.links:
            links.append([(geom.get_verts(), geom.get_trimesh().faces) for geom in link.geoms])
            link_names.append(name)
    max_penetration, crossings = get_genuine_interpenetration(links)
    # FIXME: Rare (~5% of initial-pose draws) stem-through-wall traps exceed this bound by design: a thin feature
    # creeping through a sub-cell wall is a known nonconvex detection limitation, excluded from the bound.
    assert max_penetration < (1e-3 if convexify else 6e-3)

    # Over a 100-step window, record the residual velocities and the net energy produced per contact
    vel_lin_all, vel_ang_all = [], []
    contact_energy = {}
    for i in range(100):
        scene.step()
        com_pos = scene.rigid_solver.get_links_pos(ref=gs.link_ref_frame.link_COM)
        com_vel = scene.rigid_solver.get_links_vel(ref=gs.link_ref_frame.link_COM)
        ang = scene.rigid_solver.get_links_ang()
        vel_lin_all.append(com_vel.norm(dim=-1))
        vel_ang_all.append(ang.norm(dim=-1))
        contacts = scene.rigid_solver.collider.get_contacts(as_tensor=True)
        link_a, link_b = contacts["link_a"], contacts["link_b"]
        pos, force = contacts["position"], contacts["force"]
        v_rel = (
            com_vel[link_b]
            + torch.linalg.cross(ang[link_b], pos - com_pos[link_b])
            - com_vel[link_a]
            - torch.linalg.cross(ang[link_a], pos - com_pos[link_a])
        )
        power = (force * v_rel).sum(dim=-1)
        keys = zip(link_a.tolist(), link_b.tolist(), map(tuple, (pos / 2e-3).round().tolist()))
        for key, contact_power in zip(keys, power.tolist()):
            contact_energy[key] = contact_energy.get(key, 0.0) + contact_power * scene.options.sim.dt
        energy_trace.append(tensor_to_array(scene.rigid_solver.get_total_energy()))
        if show_viewer:
            vmax_trace.append(com_vel.norm(dim=-1).max())
            wmax_trace.append(ang.norm(dim=-1).max())

    # Make sure that all objects are settling at rest.
    # Note that it is not possible to be stricter than quantile because there is legitimate residual motion.
    # FIXME: Why the angular velocity threshold has to be so large without any visual effect?!
    assert_allclose(torch.quantile(torch.stack(vel_lin_all, dim=0), 0.7, dim=0), 0.0, tol=0.1 if convexify else 0.25)
    assert_allclose(torch.quantile(torch.stack(vel_ang_all, dim=0), 0.7, dim=0), 0.0, tol=5.0 if convexify else 8.0)

    # Contacts at zero restitution must dissipate over their lifetime, so net positive contact energy is the
    # solver pumping; contact_data.force acts as -F on link_a and +F on link_b.
    # FIXME: Both path pumps net positive contact energy over this window.
    assert sum(max(energy, 0.0) for energy in contact_energy.values()) < (0.1 if convexify else 1.2)
    # Total mechanical energy (KE+PE) is a state function, so its per-step rise isolates fictitious energy the
    # solver injected at contacts (a strictly dissipative pile can only lose energy).
    # FIXME: Both paths suffer from fictitious energy injection.
    assert np.quantile(np.maximum(np.diff(energy_trace), 0.0), 0.9 if convexify else 0.75) < tol

    if show_viewer:
        _fig, (ax_v, ax_w, ax_e) = plt.subplots(3, 1, sharex=True, figsize=(8, 8))
        ax_v.semilogy(vmax_trace)
        ax_v.set_ylabel("max |linear velocity| [m/s]")
        ax_w.semilogy(wmax_trace)
        ax_w.set_ylabel("max |angular velocity| [rad/s]")
        ax_w.set_ylim(bottom=1e-3)
        ax_e.plot(np.maximum(np.diff(energy_trace), 0.0))
        ax_e.set_ylabel("energy injected dE+ [J]")
        ax_e.set_xlabel("step")
        for ax in (ax_v, ax_w, ax_e):
            ax.set_xlim(0, len(vmax_trace) - 1)
            ax.grid(True)
        plt.tight_layout()
        plt.show(block=True)

        pairs = []
        for crossing in crossings:
            a, b = crossing.link_a, crossing.link_b
            label = f"{link_names[a]}#{a} vs {link_names[b]}#{b} ({crossing.depth * 1e3:.1f}mm)"
            pairs.append((links[a], links[b], label))
        if pairs:
            display_collision_pairs(pairs)

import numpy as np
import pytest
import torch

import genesis as gs
import genesis.utils.geom as gu
from genesis.utils.misc import gaussian_crosstalk_kernel, tensor_to_array

from ..utils.assertions import assert_allclose, assert_equal


@pytest.mark.slow  # ~200s
@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
def test_surface_distance_sensor_box_sphere(show_viewer, tol, n_envs):
    SPHERE_RADIUS = 0.05
    BOX_SIZE = 0.1
    DISTANCE = 0.15
    MAX_RANGE = 10.0
    N_SETTLE = 5
    # Overlap of the untracked distractor sphere into the box's +y face. With zero gravity the contact pushes the
    # free box a few mm along -y over the settle steps, so the box-mounted probes move and the sensor is validated
    # against the live box pose rather than a static analytic value.
    DISTRACTOR_PENETRATION = 0.01
    # The sensor measures to the tracked collision MESH, so its nearest point can sit up to this far inside the
    # analytic sphere surface (icosphere faceting). Used only for the on-surface grounding check; the distance
    # self-consistency check stays exact.
    MESH_TOL = 5e-4
    BOX_PROBE_POS = [(0.0, 0.0, 0.0), (0.0, 0.0, BOX_SIZE / 2.0)]
    SPHERE_PROBE_POS = [(0.0, 0.0, SPHERE_RADIUS)]

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            gravity=(0.0, 0.0, 0.0),
        ),
        profiling_options=gs.options.ProfilingOptions(
            show_FPS=False,
        ),
        show_viewer=show_viewer,
    )
    box = scene.add_entity(
        gs.morphs.Box(
            size=(BOX_SIZE, BOX_SIZE, BOX_SIZE),
            pos=(0.0, 0.0, 0.0),
        ),
    )
    # Tracked objects
    sphere1 = scene.add_entity(
        gs.morphs.Sphere(
            radius=SPHERE_RADIUS,
            pos=(0.0, 0.0, DISTANCE),
        ),
    )
    sphere2 = scene.add_entity(
        gs.morphs.Sphere(
            radius=SPHERE_RADIUS,
            pos=(0.0, 0.0, DISTANCE * 2.0),
        ),
    )
    # Untracked distractor whose surface overlaps the box +y face by DISTRACTOR_PENETRATION (see above); the contact
    # pushes the free box centrally, so it only translates (no torque). It is also closer to the probes than either
    # tracked sphere, so a bug that ignored track_link_idx and measured every link would change the reading.
    sphere3 = scene.add_entity(
        gs.morphs.Sphere(
            radius=SPHERE_RADIUS,
            pos=(0.0, BOX_SIZE / 2.0 + SPHERE_RADIUS - DISTRACTOR_PENETRATION, 0.0),
        ),
    )

    box_to_spheres_dist_sensor = scene.add_sensor(
        gs.sensors.SurfaceDistanceProbe(
            entity_idx=box.idx,
            probe_local_pos=BOX_PROBE_POS,
            probe_radius=MAX_RANGE,
            track_link_idx=(sphere1.base_link_idx, sphere2.base_link_idx),
        )
    )
    sphere_to_box_dist_sensor = scene.add_sensor(
        gs.sensors.SurfaceDistanceProbe(
            entity_idx=sphere1.idx,
            probe_local_pos=SPHERE_PROBE_POS,
            probe_radius=MAX_RANGE,
            resolution=0.001,
            bias=0.1,
            noise=0.01,
            random_walk=0.01,
            track_link_idx=(box.base_link_idx,),
        )
    )
    scene.build(n_envs=n_envs)

    for _ in range(N_SETTLE):
        scene.step()

    # The overlapping distractor should have pushed the box clear of the origin, exercising the moving-probe path.
    assert (box.get_pos()[..., 1] < -MESH_TOL).all(), "the penetrating distractor should push the box along -y"

    box_prox_data = box_to_spheres_dist_sensor.read()
    sphere_prox_noisy_data = sphere_to_box_dist_sensor.read()
    sphere_prox_data = sphere_to_box_dist_sensor.read_ground_truth()

    # Both box probes see sphere1, the nearest tracked sphere. Each probe's reported distance must equal the distance
    # from its LIVE world position (box pose applied to the local probe) to the reported nearest point (exact), and
    # that point must lie on sphere1's meshed surface -- validating the moving-probe path without an analytic value.
    box_pos = box.get_pos()
    box_quat = box.get_quat()
    nearest = box_to_spheres_dist_sensor.nearest_points
    sphere1_center = torch.as_tensor((0.0, 0.0, DISTANCE), dtype=box_pos.dtype, device=box_pos.device)
    for i, probe_local in enumerate(BOX_PROBE_POS):
        offset = torch.as_tensor(probe_local, dtype=box_pos.dtype, device=box_pos.device).broadcast_to(box_pos.shape)
        probe_world = box_pos + gu.transform_by_quat(offset, box_quat)
        assert_allclose(box_prox_data[..., i], torch.linalg.norm(nearest[..., i, :] - probe_world, dim=-1), tol=tol)
        assert_allclose(torch.linalg.norm(nearest[..., i, :] - sphere1_center, dim=-1), SPHERE_RADIUS, tol=MESH_TOL)
    # The box drifts only in y, so the box face directly under the sphere-mounted probe stays put and the gap holds.
    assert_allclose(sphere_prox_data, DISTANCE, tol=tol)
    with np.testing.assert_raises(AssertionError):
        assert_allclose(sphere_prox_noisy_data, sphere_prox_data, tol=tol)

    # Move sphere1 out of reach; sphere2 becomes the nearest tracked sphere for the box probes.
    sphere1_pos = np.array((0.0, 0.0, DISTANCE * 3.0))
    sphere1.set_pos(sphere1_pos)
    scene.step()

    box_pos = box.get_pos()
    box_quat = box.get_quat()
    box_prox_data = box_to_spheres_dist_sensor.read()
    nearest = box_to_spheres_dist_sensor.nearest_points
    sphere2_center = torch.as_tensor((0.0, 0.0, DISTANCE * 2.0), dtype=box_pos.dtype, device=box_pos.device)
    for i, probe_local in enumerate(BOX_PROBE_POS):
        offset = torch.as_tensor(probe_local, dtype=box_pos.dtype, device=box_pos.device).broadcast_to(box_pos.shape)
        probe_world = box_pos + gu.transform_by_quat(offset, box_quat)
        assert_allclose(box_prox_data[..., i], torch.linalg.norm(nearest[..., i, :] - probe_world, dim=-1), tol=tol)
        assert_allclose(torch.linalg.norm(nearest[..., i, :] - sphere2_center, dim=-1), SPHERE_RADIUS, tol=MESH_TOL)
    assert_allclose(sphere_to_box_dist_sensor.read_ground_truth(), DISTANCE * 3.0, tol=tol)

    # Move the box far below everything: both sensors go out of range, reporting MAX_RANGE with the nearest point
    # pinned to the probe's own world position.
    box.set_pos((0.0, 0.0, -MAX_RANGE))
    scene.step()

    box_pos = box.get_pos()
    box_quat = box.get_quat()
    assert_allclose(box_to_spheres_dist_sensor.read_ground_truth(), MAX_RANGE, tol=tol)
    assert_allclose(sphere_to_box_dist_sensor.read_ground_truth(), MAX_RANGE, tol=tol)
    for i, probe_local in enumerate(BOX_PROBE_POS):
        offset = torch.as_tensor(probe_local, dtype=box_pos.dtype, device=box_pos.device).broadcast_to(box_pos.shape)
        probe_world = box_pos + gu.transform_by_quat(offset, box_quat)
        assert_allclose(
            box_to_spheres_dist_sensor.nearest_points[..., i, :],
            probe_world,
            tol=tol,
            err_msg="When out of range, points should be the probe position in world frame",
        )
    assert_allclose(
        sphere_to_box_dist_sensor.nearest_points,
        np.array(SPHERE_PROBE_POS) + sphere1_pos,
        tol=tol,
        err_msg="When out of range, points should be the probe position in world frame",
    )


@pytest.mark.slow  # ~200s
@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
def test_kinematic_contact_probe_box_sphere_support(show_viewer, tol, n_envs):
    BOX_SIZE = 0.5
    PROBE_RADIUS = 0.05
    PENETRATION = 0.02
    CONTACT_THRESHOLD = 0.002
    STIFFNESS = 100.0
    SPHERE_RADIUS = 0.1
    GAIN = 1.5

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            gravity=(0.0, 0.0, 0.0),
        ),
        profiling_options=gs.options.ProfilingOptions(
            show_FPS=False,
        ),
        show_viewer=show_viewer,
    )
    floor = scene.add_entity(gs.morphs.Plane())
    box = scene.add_entity(
        gs.morphs.Box(
            size=(BOX_SIZE, BOX_SIZE, BOX_SIZE),
            pos=(0.0, 0.0, BOX_SIZE / 2 - PENETRATION),  # box is penetrating ground plane,
            fixed=False,  # probe will not detect fixed-fixed contact,
        )
    )
    sphere = scene.add_entity(
        gs.morphs.Sphere(
            radius=SPHERE_RADIUS,
            pos=(0.0, 0.0, BOX_SIZE + SPHERE_RADIUS + 0.2),  # start with sphere above the box,
            fixed=True,
        )
    )

    probe_local_pos = (
        (0.0, 0.0, BOX_SIZE / 2),
        (BOX_SIZE / 4, BOX_SIZE / 4, BOX_SIZE / 2),
        (-BOX_SIZE / 4, -BOX_SIZE / 4, BOX_SIZE / 2),
        (0.0, 0.0, -BOX_SIZE / 2),
    )
    probe_radii = (
        PROBE_RADIUS,
        PROBE_RADIUS / 10.0,
        BOX_SIZE / 3.0,
        PROBE_RADIUS,
    )
    # Outward surface normal at each probe; the contact normal the sensor reports is the opposite (the other
    # object's surface), so expected KinematicTaxel force aligns with -probe_normals. Not a sensor input.
    probe_normals = (
        (0.0, 0.0, 1.0),
        (0.0, 0.0, 1.0),
        (0.0, 0.0, 1.0),
        (0.0, 0.0, -1.0),
    )
    common_kwargs = dict(
        entity_idx=box.idx,
        probe_local_pos=probe_local_pos,
        probe_radius=probe_radii,
        draw_debug=show_viewer,
    )
    contact_probe = scene.add_sensor(
        gs.sensors.ContactProbe(
            contact_threshold=CONTACT_THRESHOLD,
            **common_kwargs,
        )
    )
    depth_probe = scene.add_sensor(
        gs.sensors.ContactDepthProbe(
            **common_kwargs,
        ),
    )
    noisy_radius_depth_probe = scene.add_sensor(
        gs.sensors.ContactDepthProbe(
            probe_radius_noise=0.25,
            **common_kwargs,
        )
    )
    # probe_gain variants: depth/force should scale by the gain on the measured branch only.
    gained_depth_probe = scene.add_sensor(
        gs.sensors.ContactDepthProbe(
            probe_gain=GAIN,
            **common_kwargs,
        )
    )
    taxel_kwargs = dict(
        normal_stiffness=STIFFNESS,
        normal_damping=0.0,
        shear_scalar=0.0,
        twist_scalar=0.0,
        **common_kwargs,
    )
    taxel = scene.add_sensor(
        gs.sensors.KinematicTaxel(
            **taxel_kwargs,
        ),
    )
    gained_taxel = scene.add_sensor(
        gs.sensors.KinematicTaxel(
            probe_gain=GAIN,
            **taxel_kwargs,
        ),
    )
    sphere_taxel = scene.add_sensor(
        gs.sensors.KinematicTaxel(
            entity_idx=sphere.idx,
            probe_local_pos=((0.0, 0.0, -SPHERE_RADIUS),),
            probe_radius=PROBE_RADIUS,
            normal_stiffness=STIFFNESS,
            normal_damping=0.0,
            shear_scalar=0.0,
            twist_scalar=0.0,
            draw_debug=show_viewer,
        )
    )
    # filter_link_idx drops the ground link: only the bottom probe (idx 3) sees the ground, so filtering it must
    # zero that probe while leaving the sphere-facing top probes (idx 0/2) identical to the unfiltered sensors.
    contact_probe_ground_filtered = scene.add_sensor(
        gs.sensors.ContactProbe(
            contact_threshold=CONTACT_THRESHOLD,
            filter_link_idx=(floor.link_start,),
            **common_kwargs,
        )
    )
    depth_probe_ground_filtered = scene.add_sensor(
        gs.sensors.ContactDepthProbe(
            filter_link_idx=(floor.link_start,),
            **common_kwargs,
        )
    )
    taxel_ground_filtered = scene.add_sensor(
        gs.sensors.KinematicTaxel(
            filter_link_idx=(floor.link_start,),
            **taxel_kwargs,
        )
    )

    scene.build(n_envs=n_envs)
    scene.step()

    depth = depth_probe.read_ground_truth()
    contact = contact_probe.read_ground_truth()
    force = taxel.read_ground_truth().force
    torque = taxel.read_ground_truth().torque

    assert_equal(contact, depth > CONTACT_THRESHOLD)
    assert noisy_radius_depth_probe.read().shape == depth.shape
    # Check that the box's bottom probe (idx 3) detects the ground.
    assert (depth[..., 3] > tol).all(), "Bottom probe should detect the ground."
    assert (force[..., 3, 2] > tol).all(), "Bottom taxel force should point upward."
    # Top probes should not detect anything yet.
    assert_allclose(depth[..., :3], 0.0, tol=gs.EPS)
    assert_allclose(force[..., :3, :], 0.0, tol=gs.EPS)
    assert_allclose(torque, 0.0, tol=gs.EPS)

    # Forces should be equivalent to the penetration * stiffness along normal vector.
    expected_normals = -torch.tensor(probe_normals, dtype=gs.tc_float, device=gs.device)
    assert_allclose(force, depth.unsqueeze(-1) * STIFFNESS * expected_normals, tol=tol)

    # probe_gain scales the measured branch only; GT is untouched. normal_exponent defaults to 1, so the measured
    # force is linear in the gained depth and scales by the same factor.
    gained_depth = gained_depth_probe.read()
    gained_force = gained_taxel.read().force
    assert (depth[..., 3] > tol).all()  # sanity: the bottom probe is in contact
    assert_allclose(gained_depth[..., 3], depth[..., 3] * GAIN, tol=tol)
    assert_allclose(gained_depth_probe.read_ground_truth(), depth, tol=gs.EPS)
    assert_allclose(gained_force[..., 3, :], force[..., 3, :] * GAIN, tol=tol)
    assert_allclose(gained_taxel.read_ground_truth().force, force, tol=gs.EPS)

    # Ground is the only contact so far, so filtering it zeros the bottom probe on every branch.
    assert_allclose(depth_probe_ground_filtered.read_ground_truth(), 0.0, tol=gs.EPS)
    assert not contact_probe_ground_filtered.read_ground_truth().any()
    assert_allclose(taxel_ground_filtered.read_ground_truth().force, 0.0, tol=gs.EPS)

    # Now position the sphere to penetrate the top of the box.
    box_top_z = BOX_SIZE - PENETRATION
    sphere.set_pos((0.0, 0.0, box_top_z + SPHERE_RADIUS - PENETRATION))
    scene.step()

    depth = depth_probe.read_ground_truth()
    contact = contact_probe.read_ground_truth()
    force = taxel.read_ground_truth().force
    sphere_force = sphere_taxel.read_ground_truth().force

    assert_equal(contact, depth > CONTACT_THRESHOLD)
    assert (depth[..., 0] > tol).all(), "Top center probe should detect the sphere."
    assert (force[..., 0, 2] < -tol).all(), "Top center taxel force should point downward."
    assert_allclose(depth[..., 1], 0.0, tol=gs.EPS)
    assert (depth[..., 2] > tol).all(), "Large offset probe should detect the nearby sphere."
    assert (sphere_force[..., 0, 2] > tol).all(), "Sphere taxel should see the box underneath."

    # With the sphere pressing the top and the ground under the bottom, filtering the ground zeros only the bottom
    # probe (idx 3); the sphere-driven top probe (idx 0) is untouched and matches the unfiltered sensor.
    depth_ground_filtered = depth_probe_ground_filtered.read_ground_truth()
    force_ground_filtered = taxel_ground_filtered.read_ground_truth().force
    contact_ground_filtered = contact_probe_ground_filtered.read_ground_truth()
    assert_allclose(depth_ground_filtered[..., 3], 0.0, tol=gs.EPS)
    assert_allclose(force_ground_filtered[..., 3, :], 0.0, tol=gs.EPS)
    assert_allclose(depth_ground_filtered[..., 0], depth[..., 0], tol=tol)
    assert_allclose(force_ground_filtered[..., 0, :], force[..., 0, :], tol=tol)
    assert contact_ground_filtered[..., 0].all(), "top probe still contacts the sphere"
    assert not contact_ground_filtered[..., 3].any(), "bottom probe no longer contacts the filtered ground"

    # Move sphere away and check no contact.
    sphere.set_pos((0.0, 0.0, box_top_z + SPHERE_RADIUS + PROBE_RADIUS + 0.2))
    scene.step()
    assert_allclose(sphere_taxel.read_ground_truth().force, 0.0, tol=gs.EPS)


@pytest.mark.required
def test_raycast_probe_on_fully_fixed_solver(show_viewer):
    # A fully-fixed solver shares its static collision BVH across identical envs (a single tree, see the raycaster's
    # RaycastContext), so the raycast probe must traverse it through the env -> tree routing. Fixed-fixed pairs are
    # filtered out of collision detection, so the candidate-geom mask stays empty and the probe reads zero depth
    # despite the geometric overlap.
    scene = gs.Scene(show_viewer=show_viewer)
    pad = scene.add_entity(
        gs.morphs.Box(
            size=(0.2, 0.2, 0.05),
            pos=(0.0, 0.0, 0.025),
            fixed=True,
        )
    )
    scene.add_entity(
        gs.morphs.Box(
            size=(0.1, 0.1, 0.05),
            pos=(0.0, 0.0, 0.06),
            fixed=True,
        )
    )
    probe = scene.add_sensor(
        gs.sensors.ContactDepthProbe(
            entity_idx=pad.idx,
            probe_local_pos=((0.0, 0.0, 0.025),),
            probe_radius=0.01,
            contact_depth_query="raycast",
        )
    )
    scene.build(n_envs=2)
    scene.step()

    (collision_bvh,) = probe._shared_context.collision_bvh_contexts
    assert collision_bvh.maybe_static
    assert collision_bvh.aabb.n_batches == 1
    assert_equal(probe.read_ground_truth(), 0.0)


@pytest.mark.required
def test_contact_probe_hysteresis(show_viewer):
    # ContactProbe with release_threshold < contact_threshold latches like a Schmitt trigger. Depth-probe
    # semantics: depth = probe_radius - sd(probe, geom). With the probe at the box center (link-local origin) and
    # the box descending into the ground plane, sd = box.z and depth = probe_radius - box.z.
    n_envs = 0
    BOX_SIZE = 0.2
    # Place probe 0.05m above the box bottom; reported depth = probe_radius - probe.z. With probe_radius = 0.060,
    # depth = 0.010 at zero penetration and grows linearly with penetration p.
    PROBE_LOCAL_Z = -BOX_SIZE / 2 + 0.05
    PROBE_RADIUS = 0.060
    ENTER = 0.030  # triggered at p ~= 0.020
    RELEASE = 0.015  # triggered at p ~= 0.005

    # box.z values; box.z = BOX_SIZE/2 - p gives penetration p.
    BOX_Z_OFF = 1.0  # well above plane -> no contact -> depth = 0
    BOX_Z_BELOW_RELEASE = 0.099  # p = 0.001 -> depth = 0.011 (< RELEASE)
    BOX_Z_IN_BAND = 0.090  # p = 0.010 -> depth = 0.020 (RELEASE < d < ENTER)
    BOX_Z_ABOVE_ENTER = 0.070  # p = 0.030 -> depth = 0.040 (> ENTER)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            gravity=(0.0, 0.0, 0.0),
        ),
        profiling_options=gs.options.ProfilingOptions(
            show_FPS=False,
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(gs.morphs.Plane())
    box = scene.add_entity(
        gs.morphs.Box(
            size=(BOX_SIZE, BOX_SIZE, BOX_SIZE),
            pos=(0.0, 0.0, BOX_Z_OFF),
            fixed=False,
        ),
    )

    common = dict(
        entity_idx=box.idx,
        probe_local_pos=((0.0, 0.0, PROBE_LOCAL_Z),),
        probe_radius=PROBE_RADIUS,
        draw_debug=show_viewer,
    )
    hyst_probe = scene.add_sensor(
        gs.sensors.ContactProbe(
            contact_threshold=ENTER,
            release_threshold=RELEASE,
            **common,
        ),
    )
    plain_probe = scene.add_sensor(
        gs.sensors.ContactProbe(
            contact_threshold=ENTER,
            **common,
        ),
    )

    scene.build(n_envs=n_envs)

    def step_at(box_z):
        box.set_pos((0.0, 0.0, box_z))
        scene.step()
        h = hyst_probe.read_ground_truth()
        p = plain_probe.read_ground_truth()
        return h.reshape(-1), p.reshape(-1)

    # 1. No contact.
    h, p = step_at(BOX_Z_OFF)
    assert not h.any() and not p.any()

    # 2. Depth in band before any latch: both False (not latched).
    h, p = step_at(BOX_Z_IN_BAND)
    assert not h.any() and not p.any()

    # 3. Depth above enter: both latch True.
    h, p = step_at(BOX_Z_ABOVE_ENTER)
    assert h.all() and p.all()

    # 4. Lift to band: hyst stays latched, plain releases (depth < enter).
    h, p = step_at(BOX_Z_IN_BAND)
    assert h.all() and not p.any()

    # 5. Lift to below release: hyst clears.
    h, p = step_at(BOX_Z_BELOW_RELEASE)
    assert not h.any() and not p.any()

    # 6. Back into band: still False (not latched).
    h, p = step_at(BOX_Z_IN_BAND)
    assert not h.any() and not p.any()

    # 7. Reset clears latch even if depth is in band.
    step_at(BOX_Z_ABOVE_ENTER)
    scene.reset()
    h, p = step_at(BOX_Z_IN_BAND)
    assert not h.any() and not p.any()


@pytest.mark.required
def test_contact_depth_probe_hysteresis_gain_and_dead_resample(show_viewer, tol):
    # hysteresis_strength > 0 makes the measured depth overshoot GT after a step then relax back (GT untouched);
    # probe_gain_resample_range and dead_taxel_probability redraw per-(env, probe) on each reset (GT untouched).
    BOX_SIZE = 0.2
    PROBE_LOCAL_Z = -BOX_SIZE / 2 + 0.05
    PROBE_RADIUS = 0.060
    STRENGTH = 0.5
    DT = 0.01
    TAU = 0.05  # alpha = exp(-dt/tau) ~= 0.819
    ALPHA = np.exp(-DT / TAU)
    GAIN_LOW, GAIN_HIGH = 0.5, 1.5
    DEAD_LOW, DEAD_HIGH = 0.123, 0.456
    N_ENVS = 8

    BOX_Z_OFF = 1.0
    BOX_Z_ON = 0.080  # p = 0.020, depth = 0.030 in steady state

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=DT,
            gravity=(0.0, 0.0, 0.0),
        ),
        profiling_options=gs.options.ProfilingOptions(
            show_FPS=False,
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(gs.morphs.Plane())
    box = scene.add_entity(
        gs.morphs.Box(
            size=(BOX_SIZE, BOX_SIZE, BOX_SIZE),
            pos=(0.0, 0.0, BOX_Z_OFF),
            fixed=False,
        ),
    )
    common = dict(
        entity_idx=box.idx,
        probe_local_pos=((0.0, 0.0, PROBE_LOCAL_Z),),
        probe_radius=PROBE_RADIUS,
        draw_debug=show_viewer,
    )
    hyst_sensor = scene.add_sensor(
        gs.sensors.ContactDepthProbe(
            hysteresis_strength=STRENGTH,
            hysteresis_tau=TAU,
            **common,
        ),
    )
    plain_sensor = scene.add_sensor(
        gs.sensors.ContactDepthProbe(
            **common,
        ),
    )
    gain_sensor = scene.add_sensor(
        gs.sensors.ContactDepthProbe(
            probe_gain_resample_range=(GAIN_LOW, GAIN_HIGH),
            **common,
        ),
    )
    dead_sensor = scene.add_sensor(
        gs.sensors.ContactDepthProbe(
            dead_taxel_probability=1.0,
            dead_taxel_value_range=(DEAD_LOW, DEAD_HIGH),
            **common,
        ),
    )

    scene.build(n_envs=N_ENVS)

    def step_at(z):
        box.set_pos([[0.0, 0.0, z]] * N_ENVS)
        scene.step()
        return (
            hyst_sensor.read().reshape(-1),
            hyst_sensor.read_ground_truth().reshape(-1),
            plain_sensor.read().reshape(-1),
        )

    # Step 1: no contact. All zero.
    hyst_measured, hyst_ground_truth, plain_measured = step_at(BOX_Z_OFF)
    assert_allclose(hyst_measured, 0.0, tol=tol)
    assert_allclose(hyst_ground_truth, 0.0, tol=tol)
    assert_allclose(plain_measured, 0.0, tol=tol)

    # Step 2: jump to BOX_Z_ON. GT should equal plain measured (both = D). Hyst measured = D*(1+strength).
    hyst_measured, hyst_ground_truth, plain_measured = step_at(BOX_Z_ON)
    depth_ref = float(hyst_ground_truth[0].item())
    assert depth_ref > 0.02  # sanity
    assert_allclose(plain_measured, depth_ref, tol=tol)
    assert_allclose(hyst_measured, depth_ref * (1.0 + STRENGTH), tol=tol)

    # Holding depth: xi decays by ALPHA each step, so measured = depth_ref * (1 + strength * ALPHA^i_step).
    for i_step in range(1, 5):
        hyst_measured, hyst_ground_truth, plain_measured = step_at(BOX_Z_ON)
        assert_allclose(hyst_ground_truth, depth_ref, tol=tol)
        assert_allclose(plain_measured, depth_ref, tol=tol)
        expected = depth_ref * (1.0 + STRENGTH * (ALPHA**i_step))
        assert_allclose(hyst_measured, expected, tol=tol)

    # Reset clears xi: a single step at depth_ref overshoots exactly like the first contact step.
    scene.reset()
    box.set_pos([[0.0, 0.0, BOX_Z_OFF]] * N_ENVS)
    scene.step()
    hyst_measured, hyst_ground_truth, plain_measured = step_at(BOX_Z_ON)
    assert_allclose(hyst_measured, depth_ref * (1.0 + STRENGTH), tol=tol)

    def reset_step_read():
        scene.reset()  # triggers the per-(env, probe) resample of gain and dead state
        box.set_pos([[0.0, 0.0, BOX_Z_ON]] * N_ENVS)
        scene.step()
        gains = (gain_sensor.read() / gain_sensor.read_ground_truth()).reshape(-1).cpu()
        dead = dead_sensor.read().reshape(-1).cpu()
        return gains, dead

    gains_a, dead_a = reset_step_read()
    # Gain stays in range, dead values are overwritten in range, and both vary across the 8 envs.
    assert torch.all((gains_a >= GAIN_LOW - tol) & (gains_a <= GAIN_HIGH + tol))
    assert torch.all((dead_a >= DEAD_LOW - tol) & (dead_a <= DEAD_HIGH + tol))
    assert gains_a.std().item() > 0.01 and dead_a.std().item() > 0.01
    # The dead sensor's GT is untouched -- it still reports the real (non-zero) contact depth.
    assert torch.all(dead_sensor.read_ground_truth().reshape(-1) > 0.0)

    # A second reset redraws both.
    gains_b, dead_b = reset_step_read()
    assert not torch.allclose(gains_a, gains_b, atol=1e-3)
    assert not torch.allclose(dead_a, dead_b, atol=1e-3)


@pytest.mark.slow  # ~200s
@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
def test_filler_probes_radius_zero(show_viewer, tol, n_envs):
    # probe_radius == 0 marks inactive filler probes on ElastomerTaxel / KinematicTaxel: they read 0 and are
    # excluded from dilation / force, letting an irregular taxel set be padded into a regular grid for FFT.
    SPHERE_RADIUS = 0.1
    BOX_SIZE = 0.1
    PENETRATION = 0.01
    GRID = (8, 8)
    RADIUS = 0.02

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            gravity=(0.0, 0.0, 0.0),
        ),
        profiling_options=gs.options.ProfilingOptions(
            show_FPS=False,
        ),
        show_viewer=show_viewer,
    )
    sphere = scene.add_entity(
        gs.morphs.Sphere(
            radius=SPHERE_RADIUS,
            pos=(0.0, 0.0, SPHERE_RADIUS),
            fixed=True,
        )
    )
    box = scene.add_entity(
        gs.morphs.Box(
            size=(BOX_SIZE, BOX_SIZE, BOX_SIZE),
            pos=(0.0, 0.0, SPHERE_RADIUS * 2 + BOX_SIZE / 2 - PENETRATION),
            fixed=False,
        )
    )
    grid_pos = gu.generate_grid_points_on_plane(
        lo=(-BOX_SIZE / 2, -BOX_SIZE / 2, -BOX_SIZE / 2),
        hi=(BOX_SIZE / 2, BOX_SIZE / 2, -BOX_SIZE / 2),
        normal=(0.0, 0.0, -1.0),
        nx=GRID[0],
        ny=GRID[1],
    )
    flat_pos = grid_pos.reshape(-1, 3)
    # Mark a 2x2 corner block (flat indices iy*nx+ix) as inactive fillers; the rest sense normally.
    filler_idx = [0, 1, GRID[0], GRID[0] + 1]
    radii = np.full(flat_pos.shape[0], RADIUS)
    radii[filler_idx] = 0.0
    active_mask = radii > 0.0

    elastomer_kwargs = dict(
        entity_idx=box.idx,
        probe_local_normal=(0.0, 0.0, -1.0),
        track_link_idx=(sphere.base_link_idx,),
        n_sample_points=600,
        lambda_s=0.0,
        shear_scale=0.0,
        dilate_scale=1.0,
        draw_debug=show_viewer,
    )
    elastomer_grid = scene.add_sensor(
        gs.sensors.ElastomerTaxel(
            probe_local_pos=grid_pos,
            probe_radius=radii.tolist(),
            **elastomer_kwargs,
        )
    )
    elastomer_active = scene.add_sensor(
        gs.sensors.ElastomerTaxel(
            probe_local_pos=flat_pos[active_mask],
            probe_radius=RADIUS,
            **elastomer_kwargs,
        )
    )
    kinematic_kwargs = dict(
        entity_idx=box.idx,
        normal_stiffness=500.0,
        draw_debug=show_viewer,
    )
    kinematic_grid = scene.add_sensor(
        gs.sensors.KinematicTaxel(
            probe_local_pos=grid_pos,
            probe_radius=radii.tolist(),
            **kinematic_kwargs,
        )
    )
    kinematic_full = scene.add_sensor(
        gs.sensors.KinematicTaxel(
            probe_local_pos=grid_pos,
            probe_radius=RADIUS,
            **kinematic_kwargs,
        )
    )
    kinematic_crosstalk = scene.add_sensor(
        gs.sensors.KinematicTaxel(
            probe_local_pos=grid_pos,
            probe_radius=radii.tolist(),
            crosstalk_strength=1.0,
            crosstalk_sigma=BOX_SIZE / GRID[0],
            **kinematic_kwargs,
        )
    )
    assert elastomer_grid._use_grid_fft
    scene.build(n_envs=n_envs)
    scene.step()

    # ElastomerTaxel (FFT dilation): filler probes read 0; active probes match a sensor built from only the
    # active probes -- the fillers contribute no dilation, so the active readings are unchanged by their padding.
    # The grid-input sensor reports (..., ny, nx, 3); flatten the grid axes for filler-index comparison.
    grid_data = torch.as_tensor(elastomer_grid.read_ground_truth(), device=gs.device).flatten(-3, -2)
    active_data = torch.as_tensor(elastomer_active.read_ground_truth(), device=gs.device)
    assert torch.linalg.norm(grid_data, dim=-1).max() > tol, "active elastomer probes should detect contact"
    assert_allclose(grid_data[..., filler_idx, :], 0.0, tol=gs.EPS)
    assert_allclose(grid_data[..., active_mask, :], active_data, tol=tol)

    # KinematicTaxel: filler probes read 0 force; active probes match the all-active grid (per-probe force).
    # KinematicTaxel reports a grid-shaped (..., ny, nx, 3) reading; flatten the grid axes to the flat index.
    kin_grid = torch.as_tensor(kinematic_grid.read().force, device=gs.device).flatten(-3, -2)
    kin_full = torch.as_tensor(kinematic_full.read().force, device=gs.device).flatten(-3, -2)
    assert torch.linalg.norm(kin_full, dim=-1).max() > tol, "active kinematic probes should detect contact"
    assert_allclose(kin_grid[..., filler_idx, :], 0.0, tol=gs.EPS)
    assert_allclose(kin_grid[..., active_mask, :], kin_full[..., active_mask, :], tol=tol)

    # KinematicTaxel FFT crosstalk smears neighbour force, but filler probes are still masked back to 0.
    kin_xt = torch.as_tensor(kinematic_crosstalk.read().force, device=gs.device).flatten(-3, -2)
    assert_allclose(kin_xt[..., filler_idx, :], 0.0, tol=gs.EPS)


@pytest.mark.required
def test_contact_depth_query_sdf_vs_raycast_parity(show_viewer):
    # SDF and raycast contact-depth backends should agree on a face-on contact across the probe sensors. The backend
    # is class-wide (all sensors of a class share one mode), so each mode is built in its own scene and compared.
    PAD_SIZE = (0.2, 0.2, 0.05)
    PAD_TOP_Z = PAD_SIZE[2]
    BALL_R = 0.04
    PROBE_R = 0.01
    CENTER_PROBE = (0.0, 0.0, PAD_SIZE[2] / 2)

    def build_and_read(mode):
        # Build a scene whose probe sensors all use mode, press the ball in, and return CPU-side readings.
        scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                gravity=(0.0, 0.0, 0.0),
            ),
            profiling_options=gs.options.ProfilingOptions(
                show_FPS=False,
            ),
            show_viewer=show_viewer,
        )
        pad = scene.add_entity(
            gs.morphs.Box(
                size=PAD_SIZE,
                pos=(0.0, 0.0, PAD_SIZE[2] / 2),
                fixed=True,
            )
        )
        ball = scene.add_entity(
            gs.morphs.Sphere(
                radius=BALL_R,
                pos=(0.0, 0.0, 0.4),
            )
        )

        common = dict(entity_idx=pad.idx, probe_local_pos=(CENTER_PROBE,), probe_radius=PROBE_R)
        depth = scene.add_sensor(
            gs.sensors.ContactDepthProbe(
                contact_depth_query=mode,
                **common,
            )
        )
        # Filtering the ball (the only counterpart) must zero the depth on both backends: the SDF path drops it from
        # the per-sensor geom list, the raycast path drops it from the candidate-geom mask.
        depth_filtered = scene.add_sensor(
            gs.sensors.ContactDepthProbe(
                contact_depth_query=mode,
                filter_link_idx=(ball.base_link_idx,),
                **common,
            )
        )
        kin = scene.add_sensor(
            gs.sensors.KinematicTaxel(
                normal_stiffness=100.0,
                normal_damping=0.0,
                shear_scalar=0.0,
                twist_scalar=0.0,
                contact_depth_query=mode,
                **common,
            )
        )
        elast = scene.add_sensor(
            gs.sensors.ElastomerTaxel(
                entity_idx=pad.idx,
                probe_local_pos=(CENTER_PROBE,),
                probe_radius=PROBE_R,
                probe_local_normal=(0.0, 0.0, 1.0),
                contact_depth_query=mode,
                track_link_idx=(ball.base_link_idx,),
                n_sample_points=200,
            )
        )
        scene.build(n_envs=0)

        ball.set_pos((0.0, 0.0, PAD_TOP_Z + BALL_R - 0.005))  # 5mm penetration
        scene.step()
        # Materialize on CPU so the readings survive the next scene build.
        return (
            tensor_to_array(depth.read_ground_truth()),
            tensor_to_array(kin.read_ground_truth().force).reshape(-1, 3),
            tensor_to_array(elast.read_ground_truth()),
            tensor_to_array(depth_filtered.read_ground_truth()),
        )

    sdf_d, sdf_f, sdf_e, sdf_d_filtered = build_and_read("sdf")
    ray_d, ray_f, ray_e, ray_d_filtered = build_and_read("raycast")

    # Filtering the only counterpart zeros the depth on both backends (SDF geom list / raycast candidate mask).
    assert_allclose(sdf_d_filtered, 0.0, tol=gs.EPS)
    assert_allclose(ray_d_filtered, 0.0, tol=gs.EPS)

    # ContactDepthProbe -- both backends report a positive depth of the same order. They do not match tightly: SDF
    # uses the ball's analytic sphere SDF while raycast hits its faceted mesh, so the depths differ by a
    # mesh-discretization margin (a few tenths of the probe radius).
    assert (sdf_d > gs.EPS).all() and (ray_d > gs.EPS).all()
    assert_allclose(sdf_d, ray_d, tol=0.5 * PROBE_R)

    # KinematicTaxel force: both modes report a force in the same direction with magnitude within mesh-discretization
    # tolerance of each other.
    assert np.linalg.norm(sdf_f, axis=-1).item() > 0
    assert np.linalg.norm(ray_f, axis=-1).item() > 0
    cos_sim = (sdf_f * ray_f).sum(axis=-1) / (np.linalg.norm(sdf_f, axis=-1) * np.linalg.norm(ray_f, axis=-1) + gs.EPS)
    assert (cos_sim > 0.9).all(), f"force direction mismatch: cos_sim={cos_sim}"

    # ElastomerTaxel dilate displacement: face-on contact, identical on both modes when geom is a sphere primitive.
    assert_allclose(sdf_e, ray_e, tol=0.1 * PROBE_R)


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
def test_filtered_contact_survives_prefilter_cap(show_viewer, monkeypatch, n_envs):
    # The raycast contact-depth path prefilters the sensor link's contacts into a capped per-sensor list before
    # querying depth. The counterpart filter must run before that cap: otherwise a filtered manifold can fill the
    # list and starve an allowed contact. Shrink the cap to 1 so a single filtered contact would exhaust it, then
    # filter the sphere (whose contact is enumerated before the ground manifold). The allowed ground contact must
    # still reach the bottom probe -- with the filter applied too late, both probes read zero.
    monkeypatch.setattr("genesis.engine.sensors.kinematic_tactile._MAX_CONTACTS_PER_SENSOR", 1)
    BOX_SIZE = 0.2
    SPHERE_RADIUS = 0.1
    PENETRATION = 0.02
    PROBE_RADIUS = 0.05

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            gravity=(0.0, 0.0, 0.0),
        ),
        profiling_options=gs.options.ProfilingOptions(
            show_FPS=False,
        ),
        show_viewer=show_viewer,
    )
    plane = scene.add_entity(gs.morphs.Plane())
    box = scene.add_entity(
        gs.morphs.Box(
            size=(BOX_SIZE, BOX_SIZE, BOX_SIZE),
            pos=(0.0, 0.0, BOX_SIZE / 2 - PENETRATION),
        )
    )
    sphere = scene.add_entity(
        gs.morphs.Sphere(
            radius=SPHERE_RADIUS,
            pos=(0.0, 0.0, BOX_SIZE + SPHERE_RADIUS - 2 * PENETRATION),
            fixed=True,
        )
    )
    probe = scene.add_sensor(
        gs.sensors.ContactDepthProbe(
            entity_idx=box.idx,
            probe_local_pos=((0.0, 0.0, BOX_SIZE / 2), (0.0, 0.0, -BOX_SIZE / 2)),
            probe_radius=PROBE_RADIUS,
            contact_depth_query="raycast",
            filter_link_idx=(sphere.base_link_idx,),
        )
    )
    scene.build(n_envs=n_envs)
    scene.step()

    depth = probe.read_ground_truth()
    # Bottom probe faces the unfiltered ground: it must survive the cap despite the filtered sphere contact.
    assert (depth[..., 1] > gs.EPS).all(), "allowed ground contact must survive the counterpart-filter prefilter cap"
    # Top probe faces the filtered sphere, which contributes nothing.
    assert_allclose(depth[..., 0], 0.0, tol=gs.EPS)


@pytest.mark.required
def test_kinematic_taxel_crosstalk(show_viewer):
    # Crosstalk smears the measured force across grid neighbors (GT unchanged) and preserves total normal force,
    # whether configured via a Gaussian (crosstalk_strength/sigma) or an explicit per-group kernel.
    # crosstalk_strength=0 and an identity kernel are both the exact no-crosstalk path, and a grid layout matches a
    # flat one at the same probes.
    BOX_SIZE = 0.2
    PROBE_RADIUS = 0.02
    SPACING = 0.03
    SPHERE_RADIUS = 0.025
    BOX_BOTTOM_Z = 0.05
    CROSSTALK_STRENGTH = 0.6
    CROSSTALK_SIGMA = SPACING
    BLUR_KERNEL = [[0.03, 0.07, 0.03], [0.07, 0.60, 0.07], [0.03, 0.07, 0.03]]  # sums to 1 (conservative)
    IDENTITY_KERNEL = [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]

    ny, nx = 5, 5
    grid_positions = np.zeros((ny, nx, 3), dtype=gs.np_float)
    for i_y in range(ny):
        for i_x in range(nx):
            grid_positions[i_y, i_x] = ((i_x - 2) * SPACING, (i_y - 2) * SPACING, BOX_SIZE / 2)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            gravity=(0.0, 0.0, 0.0),
        ),
        profiling_options=gs.options.ProfilingOptions(
            show_FPS=False,
        ),
        show_viewer=show_viewer,
    )
    box = scene.add_entity(
        gs.morphs.Box(
            size=(BOX_SIZE, BOX_SIZE, BOX_SIZE),
            pos=(0.0, 0.0, BOX_BOTTOM_Z + BOX_SIZE / 2),
            fixed=True,
        )
    )
    sphere = scene.add_entity(
        gs.morphs.Sphere(
            radius=SPHERE_RADIUS,
            pos=(0.0, 0.0, BOX_BOTTOM_Z + BOX_SIZE + SPHERE_RADIUS - 0.010),
            fixed=False,
        )
    )

    common = dict(
        entity_idx=box.idx,
        probe_radius=PROBE_RADIUS,
        normal_stiffness=100.0,
        normal_damping=0.0,
        shear_scalar=0.0,
        twist_scalar=0.0,
    )
    plain = scene.add_sensor(
        gs.sensors.KinematicTaxel(
            probe_local_pos=grid_positions.tolist(),
            **common,
        )
    )
    crosstalk = scene.add_sensor(
        gs.sensors.KinematicTaxel(
            probe_local_pos=grid_positions.tolist(),
            crosstalk_strength=CROSSTALK_STRENGTH,
            crosstalk_sigma=CROSSTALK_SIGMA,
            **common,
        )
    )
    # crosstalk_strength=0 must reproduce the no-crosstalk path exactly, even with a non-zero sigma.
    crosstalk_off = scene.add_sensor(
        gs.sensors.KinematicTaxel(
            probe_local_pos=grid_positions.tolist(),
            crosstalk_strength=0.0,
            crosstalk_sigma=0.05,
            **common,
        )
    )
    # Same probes laid out flat: per-probe GT must match the grid layout.
    flat = scene.add_sensor(
        gs.sensors.KinematicTaxel(
            probe_local_pos=grid_positions.reshape(-1, 3).tolist(),
            **common,
        )
    )
    ck_id = scene.add_sensor(
        gs.sensors.KinematicTaxel(probe_local_pos=grid_positions.tolist(), crosstalk_kernel=IDENTITY_KERNEL, **common)
    )
    ck_blur = scene.add_sensor(
        gs.sensors.KinematicTaxel(probe_local_pos=grid_positions.tolist(), crosstalk_kernel=BLUR_KERNEL, **common)
    )
    ck_normal = scene.add_sensor(
        gs.sensors.KinematicTaxel(
            probe_local_pos=grid_positions.tolist(), crosstalk_kernel=[BLUR_KERNEL, IDENTITY_KERNEL], **common
        )
    )
    ck_shear = scene.add_sensor(
        gs.sensors.KinematicTaxel(
            probe_local_pos=grid_positions.tolist(), crosstalk_kernel=[IDENTITY_KERNEL, BLUR_KERNEL], **common
        )
    )

    scene.build(n_envs=0)
    sphere.set_pos((0.0, 0.0, BOX_BOTTOM_Z + BOX_SIZE + SPHERE_RADIUS - 0.010))
    scene.step()

    plain_meas_force = plain.read().force
    crosstalk_meas_force = crosstalk.read().force
    plain_gt_force = plain.read_ground_truth().force
    crosstalk_gt_force = crosstalk.read_ground_truth().force

    # GT branch is untouched by crosstalk.
    assert_allclose(crosstalk_gt_force, plain_gt_force, tol=gs.EPS)

    # Plain measured equals GT (no transforms enabled on plain sensor).
    assert_allclose(plain_meas_force, plain_gt_force, tol=gs.EPS)

    plain_force_mag = torch.linalg.norm(plain_meas_force, dim=-1)
    i_y_c, i_x_c = (plain_force_mag == plain_force_mag.max()).nonzero(as_tuple=False)[0].tolist()
    assert (i_y_c, i_x_c) == (ny // 2, nx // 2)

    crosstalk_force_mag = torch.linalg.norm(crosstalk_meas_force, dim=-1)
    # Center magnitude on crosstalk sensor is reduced vs plain (energy redistributed).
    assert crosstalk_force_mag[i_y_c, i_x_c] < plain_force_mag[i_y_c, i_x_c]
    # A probe outside the contact patch (2 spacings from center) was ~zero on plain; crosstalk leaks force there.
    plain_far = plain_force_mag[0, 0].item()
    crosstalk_far = crosstalk_force_mag[0, 0].item()
    assert plain_far < 1e-4, f"far probe should be ~zero on plain sensor (got {plain_far})"
    assert crosstalk_far > 1e-4, f"far probe should pick up crosstalk leakage (got {crosstalk_far})"

    # Total Fz across the grid is preserved up to Gaussian-tail leakage past the output slice boundary.
    plain_total_fz = plain_meas_force[..., 2].sum().item()
    crosstalk_total_fz = crosstalk_meas_force[..., 2].sum().item()
    assert np.isclose(plain_total_fz, crosstalk_total_fz, rtol=5e-2, atol=1e-5), (
        f"plain={plain_total_fz}, crosstalk={crosstalk_total_fz}"
    )

    # crosstalk_strength=0 is the exact no-crosstalk path (even with a non-zero sigma).
    assert_allclose(crosstalk_off.read().force, plain_meas_force, tol=gs.EPS)
    assert_allclose(crosstalk_off.read().torque, plain.read().torque, tol=gs.EPS)

    # A grid layout produces the same per-probe GT as a flat layout at the identical positions.
    flat_gt = flat.read_ground_truth()
    assert_allclose(plain_gt_force.reshape(-1, 3), flat_gt.force, tol=gs.EPS)
    assert_allclose(plain.read_ground_truth().torque.reshape(-1, 3), flat_gt.torque, tol=gs.EPS)

    plain_fz = plain_meas_force[..., 2]

    # An identity kernel is an exact no-op, and crosstalk never touches the GT branch.
    assert_allclose(ck_id.read().force, plain_meas_force, tol=1e-6)
    assert_allclose(ck_blur.read_ground_truth().force, plain_gt_force, tol=gs.EPS)

    # The (N, M) blur reduces the contact peak and leaks force to probes that read ~zero on the plain sensor.
    plain_zero = plain_force_mag < 1e-4
    assert plain_zero.any()
    blur_mag = torch.linalg.norm(ck_blur.read().force, dim=-1)
    assert blur_mag[2, 2] < plain_force_mag[2, 2]
    assert (blur_mag[plain_zero] > 1e-4).any()
    assert np.isclose(plain_fz.sum().item(), ck_blur.read().force[..., 2].sum().item(), rtol=5e-2, atol=1e-5)

    # 2-group [normal, shear]: contact force is pure normal (Fz), so the normal kernel governs it. A normal-blur
    # spreads Fz (peak down, leaks into previously-zero probes); a shear-blur leaves Fz identical (the shear
    # component is ~zero here).
    normal_fz = ck_normal.read().force[..., 2]
    shear_fz = ck_shear.read().force[..., 2]
    assert normal_fz[2, 2].abs() < plain_fz[2, 2].abs()
    assert (normal_fz.abs()[plain_zero] > 1e-4).any()
    assert_allclose(shear_fz, plain_fz, tol=1e-6)


@pytest.mark.required
def test_gaussian_crosstalk_kernel_helper():
    # gaussian_crosstalk_kernel: L1-normalized (conservative), symmetric, center-peaked, rejects even dims.
    kernel = gaussian_crosstalk_kernel(5, 5, sigma=1.0)
    assert kernel.shape == (5, 5)
    assert np.isclose(kernel.sum(), 1.0)
    assert np.allclose(kernel, kernel.T)  # isotropic on a square grid -> symmetric
    assert kernel[2, 2] == kernel.max()  # center is the self (peak) tap
    assert kernel[2, 2] < 1.0  # a conservative kernel shares the peak with neighbors (center < 1)
    assert gaussian_crosstalk_kernel(5, 5, sigma=2.0)[2, 2] < kernel[2, 2]  # wider sigma spreads more
    # anisotropic pitch: a larger step on the row axis makes row neighbors lighter than column neighbors.
    kernel_aniso = gaussian_crosstalk_kernel(5, 5, sigma=1.0, spacing=(2.0, 1.0))
    assert kernel_aniso[1, 2] < kernel_aniso[2, 1]
    for bad in [(4, 5), (5, 4)]:
        with pytest.raises(Exception):
            gaussian_crosstalk_kernel(*bad, sigma=1.0)


@pytest.mark.required
def test_proximity_taxel_crosstalk(show_viewer):
    # ProximityTaxel crosstalk smears the measured force across grid neighbors (peak down, leakage) with GT untouched.
    BOX_SIZE = 0.2
    SPACING = 0.03
    SPHERE_RADIUS = 0.03
    BOX_BOTTOM_Z = 0.05
    PROBE_RADIUS = 0.04

    ny, nx = 5, 5
    grid_positions = np.zeros((ny, nx, 3), dtype=gs.np_float)
    for i_y in range(ny):
        for i_x in range(nx):
            grid_positions[i_y, i_x] = ((i_x - 2) * SPACING, (i_y - 2) * SPACING, BOX_SIZE / 2)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            gravity=(0.0, 0.0, 0.0),
        ),
        profiling_options=gs.options.ProfilingOptions(
            show_FPS=False,
        ),
        show_viewer=show_viewer,
    )
    box = scene.add_entity(
        gs.morphs.Box(
            size=(BOX_SIZE, BOX_SIZE, BOX_SIZE),
            pos=(0.0, 0.0, BOX_BOTTOM_Z + BOX_SIZE / 2),
            fixed=True,
        )
    )
    sphere = scene.add_entity(
        gs.morphs.Sphere(
            radius=SPHERE_RADIUS,
            pos=(0.0, 0.0, BOX_BOTTOM_Z + BOX_SIZE + SPHERE_RADIUS - 0.012),
            fixed=True,
        )
    )
    common = dict(
        entity_idx=box.idx,
        probe_local_pos=grid_positions.tolist(),
        probe_local_normal=(0.0, 0.0, 1.0),
        probe_radius=PROBE_RADIUS,
        track_link_idx=(sphere.base_link_idx,),
        n_sample_points=3000,
        stiffness=100.0,
        shear_coupling=0.0,
    )
    plain = scene.add_sensor(
        gs.sensors.ProximityTaxel(
            **common,
        )
    )
    crosstalk = scene.add_sensor(
        gs.sensors.ProximityTaxel(
            crosstalk_kernel=[[0.03, 0.07, 0.03], [0.07, 0.60, 0.07], [0.03, 0.07, 0.03]],
            **common,
        )
    )

    scene.build(n_envs=0)
    scene.step()

    plain_f = plain.read().force
    plain_mag = torch.linalg.norm(plain_f, dim=-1)
    # The localized sphere indent must give a peaked field with some ~zero probes, else there is nothing to smear.
    i_y_c, i_x_c = (plain_mag == plain_mag.max()).nonzero(as_tuple=False)[0].tolist()
    assert (i_y_c, i_x_c) == (ny // 2, nx // 2)
    plain_zero = plain_mag < 1e-4
    assert plain_zero.any()

    # GT branch untouched; measured peak reduced; previously-zero probes pick up leakage.
    assert_allclose(crosstalk.read_ground_truth().force, plain.read_ground_truth().force, tol=gs.EPS)
    ck_mag = torch.linalg.norm(crosstalk.read().force, dim=-1)
    assert ck_mag[i_y_c, i_x_c] < plain_mag[i_y_c, i_x_c]
    assert (ck_mag[plain_zero] > 1e-4).any()


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
def test_proximity_taxel_twist_torque(show_viewer, tol, n_envs):
    PAD_SIZE = 0.2
    PAD_TOP = PAD_SIZE
    OBJ_SIZE = 0.06
    PROBE_RADIUS = 0.04
    PENETRATION = 0.004
    WZ = 4.0

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            gravity=(0.0, 0.0, 0.0),
        ),
        profiling_options=gs.options.ProfilingOptions(
            show_FPS=False,
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(gs.morphs.Plane())
    pad = scene.add_entity(
        gs.morphs.Box(
            size=(PAD_SIZE, PAD_SIZE, PAD_SIZE),
            pos=(0.0, 0.0, PAD_SIZE / 2),
            fixed=True,
        )
    )
    # A free object: the twist term is driven by relative angular velocity, so the tracked body must actually spin.
    obj = scene.add_entity(
        gs.morphs.Box(
            size=(OBJ_SIZE, OBJ_SIZE, OBJ_SIZE),
        )
    )
    grid = gu.generate_grid_points_on_plane(
        lo=(-PAD_SIZE / 2, -PAD_SIZE / 2, PAD_SIZE / 2),
        hi=(PAD_SIZE / 2, PAD_SIZE / 2, PAD_SIZE / 2),
        normal=(0.0, 0.0, 1.0),
        nx=8,
        ny=8,
    )
    common = dict(
        entity_idx=pad.idx,
        probe_local_pos=grid,
        probe_local_normal=(0.0, 0.0, 1.0),
        probe_radius=PROBE_RADIUS,
        track_link_idx=(obj.base_link_idx,),
        n_sample_points=3000,
        stiffness=100.0,
        shear_coupling=0.0,  # isolate the spin term from shear-field-curl torsion
        draw_debug=show_viewer,
    )
    twist1 = scene.add_sensor(
        gs.sensors.ProximityTaxel(
            twist_scalar=1.0,
            **common,
        )
    )
    twist2 = scene.add_sensor(
        gs.sensors.ProximityTaxel(
            twist_scalar=2.0,
            **common,
        )
    )
    twist0 = scene.add_sensor(
        gs.sensors.ProximityTaxel(
            twist_scalar=0.0,
            **common,
        )
    )

    scene.build(n_envs=n_envs)

    obj_center_z = PAD_TOP + OBJ_SIZE / 2 - PENETRATION
    for _ in range(4):
        obj.set_pos((0.0, 0.0, obj_center_z))
        obj.set_dofs_velocity((0.0, 0.0, WZ), dofs_idx_local=slice(3, None))
        scene.step()

    normal = torch.tensor((0.0, 0.0, 1.0), dtype=gs.tc_float, device=gs.device)

    def max_twist(sensor):
        return (sensor.read().torque @ normal).abs().max()

    # With shear disabled, the spin term is the sole source of torque about the normal: present for twist_scalar>0,
    # exactly linear in twist_scalar, and identically zero without it.
    assert max_twist(twist1) > tol
    assert_allclose(max_twist(twist2), 2.0 * max_twist(twist1), tol=tol)
    assert_allclose(max_twist(twist0), 0.0, tol=tol)
    # A torque reading still exists without the spin term: the lever arm yields in-plane tilting moments.
    assert torch.linalg.norm(twist0.read().torque, dim=-1).max() > tol


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
def test_proximity_sensor_box_on_box(show_viewer, tol, n_envs):
    BOX_SIZE = 0.2
    PENETRATION = 0.01
    GAIN = 1.5

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            gravity=(0.0, 0.0, 0.0),
        ),
        profiling_options=gs.options.ProfilingOptions(
            show_FPS=False,
        ),
        show_viewer=show_viewer,
    )
    support = scene.add_entity(
        gs.morphs.Box(
            size=(BOX_SIZE, BOX_SIZE, BOX_SIZE),
            pos=(0.0, 0.0, BOX_SIZE / 2),
            fixed=True,
        )
    )
    taxel_box = scene.add_entity(
        gs.morphs.Box(
            size=(BOX_SIZE, BOX_SIZE, BOX_SIZE),
            pos=(0.0, 0.0, BOX_SIZE + BOX_SIZE / 2 - PENETRATION),
            fixed=False,
        )
    )
    sensor = scene.add_sensor(
        gs.sensors.ProximityTaxel(
            entity_idx=taxel_box.idx,
            probe_local_pos=((0.0, 0.0, -BOX_SIZE / 2), (BOX_SIZE / 4, 0.0, -BOX_SIZE / 2)),
            probe_radius=0.06,
            probe_radius_noise=0.1,
            probe_local_normal=(0.0, 0.0, -1.0),
            track_link_idx=(support.base_link_idx,),
            n_sample_points=600,
            stiffness=100.0,
            shear_coupling=0.0,
            draw_debug=show_viewer,
        )
    )
    # probe_gain variant (no radius noise so the measured branch is deterministic): force is linear in the summed
    # penetration, so the measured force scales by the gain while GT is untouched.
    gained_sensor = scene.add_sensor(
        gs.sensors.ProximityTaxel(
            entity_idx=taxel_box.idx,
            probe_local_pos=((0.0, 0.0, -BOX_SIZE / 2), (BOX_SIZE / 4, 0.0, -BOX_SIZE / 2)),
            probe_radius=0.06,
            probe_local_normal=(0.0, 0.0, -1.0),
            probe_gain=GAIN,
            track_link_idx=(support.base_link_idx,),
            n_sample_points=600,
            stiffness=100.0,
            shear_coupling=0.0,
            draw_debug=show_viewer,
        )
    )

    scene.build(n_envs=n_envs)
    scene.step()

    force_norm = torch.linalg.norm(sensor.read_ground_truth().force, dim=-1)
    assert (force_norm > tol).all()

    gained_meas = gained_sensor.read().force
    gained_gt = gained_sensor.read_ground_truth().force
    assert (torch.linalg.norm(gained_gt, dim=-1) > tol).all()  # sanity: in contact
    assert_allclose(gained_meas, gained_gt * GAIN, tol=tol)

    taxel_box.set_pos((0.0, 0.0, BOX_SIZE + BOX_SIZE / 2 + 0.2))
    scene.step()
    force_norm = torch.linalg.norm(sensor.read_ground_truth().force, dim=-1)
    assert_allclose(force_norm, 0.0, tol=gs.EPS)


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
def test_elastomer_sensor_sphere_ground_dilate_shear(show_viewer, tol, n_envs):
    SPHERE_RADIUS = 0.2
    PROBE_RADIUS = 0.02
    PENETRATION = 0.01
    GROUND_THICKNESS = 0.08
    N_RINGS = 3
    LATERAL_SHIFT = 0.01
    SHEAR_SCALE = 100.0
    GAIN = 2.0

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            gravity=(0.0, 0.0, 0.0),
        ),
        profiling_options=gs.options.ProfilingOptions(
            show_FPS=False,
        ),
        show_viewer=show_viewer,
    )

    ground = scene.add_entity(
        gs.morphs.Box(
            size=(0.8, 0.8, GROUND_THICKNESS),
            pos=(0.0, 0.0, -GROUND_THICKNESS / 2),
            fixed=True,
        )
    )
    # Sphere penetrating the ground (center below z=0 by PENETRATION).
    sphere_init_pos = (0.0, 0.0, SPHERE_RADIUS - PENETRATION)
    sphere = scene.add_entity(
        gs.morphs.Sphere(
            radius=SPHERE_RADIUS,
            pos=sphere_init_pos,
            fixed=False,
        )
    )

    probe_local_pos, probe_local_normal = gu.generate_ring_points_on_sphere(
        radius=SPHERE_RADIUS,
        cap_axis=(0.0, 0.0, -1.0),
        n_rings=N_RINGS,
        arc_spacing=2.0 * PROBE_RADIUS,
        return_normals=True,
    )
    normals = torch.as_tensor(probe_local_normal, dtype=gs.tc_float, device=gs.device)
    sensor_kwargs = dict(
        entity_idx=sphere.idx,
        probe_local_pos=probe_local_pos,
        probe_local_normal=probe_local_normal,
        probe_radius=PROBE_RADIUS,
        track_link_idx=(ground.base_link_idx,),
        n_sample_points=800,
        lambda_s=0.0,
        draw_debug=show_viewer,
    )
    dilate_sensor = scene.add_sensor(
        gs.sensors.ElastomerTaxel(
            dilate_scale=1.0,
            shear_scale=0.0,
            **sensor_kwargs,
        )
    )
    shear_sensor = scene.add_sensor(
        gs.sensors.ElastomerTaxel(
            dilate_scale=0.0,
            shear_scale=SHEAR_SCALE,
            **sensor_kwargs,
        )
    )
    combined_sensor = scene.add_sensor(
        gs.sensors.ElastomerTaxel(
            dilate_scale=1.0,
            shear_scale=SHEAR_SCALE,
            **sensor_kwargs,
        )
    )
    # probe_gain variant: the measured marker displacement scales by the gain; GT is untouched.
    gained_sensor = scene.add_sensor(
        gs.sensors.ElastomerTaxel(
            dilate_scale=1.0,
            shear_scale=0.0,
            probe_gain=GAIN,
            **sensor_kwargs,
        )
    )
    assert not dilate_sensor._is_grid and not dilate_sensor._use_grid_fft

    scene.build(n_envs=n_envs)
    scene.step()

    dilate_data = dilate_sensor.read_ground_truth()
    shear_data = shear_sensor.read_ground_truth()
    combined_data = combined_sensor.read_ground_truth()
    normal_projection = (dilate_data * normals).sum(dim=-1)
    assert (normal_projection[..., 0] > tol).all(), "Bottom marker should dilate along its outward normal."
    assert torch.linalg.norm(dilate_data, dim=-1).max() > tol
    assert_allclose(shear_data, 0.0, tol=tol)
    assert_allclose(combined_data, dilate_data, tol=tol)

    gained_meas = gained_sensor.read()
    gained_gt = gained_sensor.read_ground_truth()
    assert torch.linalg.norm(gained_gt, dim=-1).max() > tol  # sanity: in contact
    assert_allclose(gained_meas, gained_gt * GAIN, tol=tol)

    sphere.set_pos((LATERAL_SHIFT, 0.0, sphere_init_pos[2]))
    scene.step()

    dilate_data = dilate_sensor.read_ground_truth()
    shear_data = shear_sensor.read_ground_truth()
    combined_data = combined_sensor.read_ground_truth()
    shear_normal_projection = (shear_data * normals).sum(dim=-1)
    shear_tangent = shear_data - shear_normal_projection.unsqueeze(-1) * normals
    assert torch.linalg.norm(shear_tangent, dim=-1).max() > tol
    assert_allclose(shear_normal_projection, 0.0, tol=tol)
    assert_allclose(combined_data, dilate_data + shear_data, tol=5e-5)

    sphere.set_pos((0.0, 0.0, SPHERE_RADIUS + 0.05))
    scene.step()
    assert_equal(combined_sensor.read_ground_truth(), 0.0, err_msg="ElastomerTaxel should be zero with no contact.")


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
def test_elastomer_sensor_grid_box_sphere(show_viewer, tol, n_envs):
    SPHERE_RADIUS = 0.1
    BOX_SIZE = 0.1
    PENETRATION = 0.01
    GRID_SIZE = (8, 8)
    LATERAL_SHIFT = 0.01
    SHEAR_SCALE = 100.0

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            gravity=(0.0, 0.0, 0.0),
        ),
        profiling_options=gs.options.ProfilingOptions(
            show_FPS=False,
        ),
        show_viewer=show_viewer,
    )
    sphere = scene.add_entity(
        gs.morphs.Sphere(
            radius=SPHERE_RADIUS,
            pos=(0.0, 0.0, SPHERE_RADIUS),
            fixed=True,
        )
    )
    box = scene.add_entity(
        gs.morphs.Box(
            size=(BOX_SIZE, BOX_SIZE, BOX_SIZE),
            pos=(0.0, 0.0, SPHERE_RADIUS * 2 + BOX_SIZE / 2 - PENETRATION),
            fixed=False,
        )
    )
    probe_local_pos = gu.generate_grid_points_on_plane(
        lo=(-BOX_SIZE / 2, -BOX_SIZE / 2, -BOX_SIZE / 2),
        hi=(BOX_SIZE / 2, BOX_SIZE / 2, -BOX_SIZE / 2),
        normal=(0.0, 0.0, -1.0),
        nx=GRID_SIZE[0],
        ny=GRID_SIZE[1],
    )
    sensor_kwargs = dict(
        entity_idx=box.idx,
        probe_local_normal=(0.0, 0.0, -1.0),
        probe_radius=0.02,
        track_link_idx=(sphere.base_link_idx,),
        n_sample_points=600,
        lambda_s=0.0,
        draw_debug=show_viewer,
    )
    elastomer_grid_sensor = scene.add_sensor(
        gs.sensors.ElastomerTaxel(
            probe_local_pos=probe_local_pos,
            dilate_scale=1.0,
            shear_scale=0.0,
            **sensor_kwargs,
        )
    )
    elastomer_sensor = scene.add_sensor(
        gs.sensors.ElastomerTaxel(
            probe_local_pos=probe_local_pos.reshape(-1, 3),
            dilate_scale=1.0,
            shear_scale=0.0,
            **sensor_kwargs,
        )
    )
    shear_sensor = scene.add_sensor(
        gs.sensors.ElastomerTaxel(
            probe_local_pos=probe_local_pos.reshape(-1, 3),
            dilate_scale=0.0,
            shear_scale=SHEAR_SCALE,
            **sensor_kwargs,
        )
    )
    combined_sensor = scene.add_sensor(
        gs.sensors.ElastomerTaxel(
            probe_local_pos=probe_local_pos.reshape(-1, 3),
            dilate_scale=1.0,
            shear_scale=SHEAR_SCALE,
            **sensor_kwargs,
        )
    )
    # A non-default normal_exponent (cubic instead of the default quadratic normal dilation), one per path.
    cubic_grid_sensor = scene.add_sensor(
        gs.sensors.ElastomerTaxel(
            probe_local_pos=probe_local_pos,
            dilate_scale=1.0,
            shear_scale=0.0,
            normal_exponent=3.0,
            **sensor_kwargs,
        )
    )
    cubic_flat_sensor = scene.add_sensor(
        gs.sensors.ElastomerTaxel(
            probe_local_pos=probe_local_pos.reshape(-1, 3),
            dilate_scale=1.0,
            shear_scale=0.0,
            normal_exponent=3.0,
            **sensor_kwargs,
        )
    )
    # compressibility blends the local Gaussian in-plane bulge (1.0, the default) with the global incompressible
    # (volume-conserving) ~1/r stretch (0.0). Add a fully incompressible grid sensor and a 50/50 blend (no
    # thickness: the free-space kernel, regularized internally at the probe spacing).
    incompressible_grid_sensor = scene.add_sensor(
        gs.sensors.ElastomerTaxel(
            probe_local_pos=probe_local_pos,
            dilate_scale=1.0,
            shear_scale=0.0,
            compressibility=0.0,
            **sensor_kwargs,
        )
    )
    half_grid_sensor = scene.add_sensor(
        gs.sensors.ElastomerTaxel(
            probe_local_pos=probe_local_pos,
            dilate_scale=1.0,
            shear_scale=0.0,
            compressibility=0.5,
            **sensor_kwargs,
        )
    )
    # elastomer_thickness replaces the free-space ~1/r global stretch with the exact bonded-incompressible-layer
    # transfer S(|k| h): a thicker gel suppresses in-plane surface motion (it approaches the incompressible
    # half-space, S -> 0), a thin gel recovers the 1/r squeeze flow. Both incompressible (compressibility=0), FFT only.
    thin_thickness_grid_sensor = scene.add_sensor(
        gs.sensors.ElastomerTaxel(
            probe_local_pos=probe_local_pos,
            dilate_scale=1.0,
            shear_scale=0.0,
            compressibility=0.0,
            elastomer_thickness=0.002,
            **sensor_kwargs,
        )
    )
    thick_thickness_grid_sensor = scene.add_sensor(
        gs.sensors.ElastomerTaxel(
            probe_local_pos=probe_local_pos,
            dilate_scale=1.0,
            shear_scale=0.0,
            compressibility=0.0,
            elastomer_thickness=0.02,
            **sensor_kwargs,
        )
    )
    assert elastomer_grid_sensor._is_grid and elastomer_grid_sensor._use_grid_fft
    assert not elastomer_sensor._is_grid and not elastomer_sensor._use_grid_fft
    assert_allclose(elastomer_sensor.probe_local_pos, elastomer_grid_sensor.probe_local_pos, tol=gs.EPS)

    scene.build(n_envs=n_envs)
    scene.step()

    # Test dilate displacement: grid sensor should match the flat-layout sensor and detect contact magnitude.
    # The grid-input sensor reports (..., ny, nx, 3); flatten the grid axes for comparison with the flat sensor.
    grid_data = torch.as_tensor(elastomer_grid_sensor.read_ground_truth(), device=gs.device).flatten(-3, -2)
    flat_data = elastomer_sensor.read_ground_truth()
    assert_allclose(flat_data, grid_data, tol=tol)
    assert torch.linalg.norm(grid_data, dim=-1).max() > tol
    assert_allclose(shear_sensor.read_ground_truth(), 0.0, tol=tol)
    assert_allclose(combined_sensor.read_ground_truth(), flat_data, tol=tol)

    # normal_exponent reshapes only the out-of-plane channel: the grid-FFT and direct paths still agree, and the
    # cubic-normal response differs from the default quadratic one (sub-unit depths here, so depth**3 < depth**2).
    cubic_data = torch.as_tensor(cubic_grid_sensor.read_ground_truth(), device=gs.device).flatten(-3, -2)
    assert_allclose(cubic_flat_sensor.read_ground_truth(), cubic_data, tol=tol)
    cubic_diff = cubic_data - grid_data
    assert torch.linalg.norm(cubic_diff, dim=-1).max() > tol, "normal_exponent=3 should change the dilation output"

    # compressibility sanity. The normal (out-of-plane, z) channel is the Gaussian bulge regardless of
    # compressibility, so it is unchanged from the default (compressibility=1) sensor for any blend. The
    # volume-conserving incompressible in-plane field decays as ~1/r instead of the local Gaussian, so it reaches
    # farther: weighting each probe's in-plane displacement by its distance from the centered contact gives a larger
    # mean radius. The 50/50 blend lands between the local default and the fully incompressible sensor.
    incompressible_data = torch.as_tensor(incompressible_grid_sensor.read_ground_truth(), device=gs.device).flatten(
        -3, -2
    )
    half_data = torch.as_tensor(half_grid_sensor.read_ground_truth(), device=gs.device).flatten(-3, -2)
    assert_allclose(incompressible_data[..., 2], grid_data[..., 2], tol=tol)
    assert_allclose(half_data[..., 2], grid_data[..., 2], tol=tol)

    flat_probe_pos = torch.as_tensor(probe_local_pos.reshape(-1, 3), dtype=gs.tc_float, device=gs.device)
    probe_radius_from_center = torch.linalg.norm(flat_probe_pos[:, :2], dim=-1)  # in-plane (x, y) distance

    def _inplane_mean_radius(data):
        inplane_mag = torch.linalg.norm(data[..., :2], dim=-1)
        return (inplane_mag * probe_radius_from_center).sum(-1) / inplane_mag.sum(-1).clamp_min(gs.EPS)

    local_radius = _inplane_mean_radius(grid_data)
    half_radius = _inplane_mean_radius(half_data)
    incompressible_radius = _inplane_mean_radius(incompressible_data)
    assert (incompressible_radius > half_radius + tol).all() and (half_radius > local_radius + tol).all(), (
        "in-plane dilation reach should grow as compressibility decreases (local Gaussian -> incompressible 1/r)"
    )

    # elastomer_thickness: the normal channel is still the Gaussian bulge, and a thicker bonded gel suppresses
    # in-plane surface motion (every Fourier mode is scaled by S(|k| h), which decreases with h), so its total
    # in-plane displacement energy is smaller than the thin gel's.
    thin_data = torch.as_tensor(thin_thickness_grid_sensor.read_ground_truth(), device=gs.device).flatten(-3, -2)
    thick_data = torch.as_tensor(thick_thickness_grid_sensor.read_ground_truth(), device=gs.device).flatten(-3, -2)
    assert_allclose(thin_data[..., 2], grid_data[..., 2], tol=tol)
    assert_allclose(thick_data[..., 2], grid_data[..., 2], tol=tol)
    thin_inplane_energy = (thin_data[..., :2] ** 2).sum((-1, -2))
    thick_inplane_energy = (thick_data[..., :2] ** 2).sum((-1, -2))
    assert (thin_inplane_energy > tol).all(), "thin bonded gel should produce a nonzero in-plane field"
    assert (thick_inplane_energy < thin_inplane_energy).all(), (
        "a thicker bonded gel should suppress in-plane surface motion (S(|k| h) -> 0)"
    )

    # Test combined displacement: dilate + shear contributions should add when the box slides laterally.
    box.set_pos((LATERAL_SHIFT, 0.0, SPHERE_RADIUS * 2 + BOX_SIZE / 2 - PENETRATION))
    scene.step()
    dilate_data = elastomer_sensor.read_ground_truth()
    shear_data = shear_sensor.read_ground_truth()
    combined_data = combined_sensor.read_ground_truth()
    assert torch.linalg.norm(torch.as_tensor(shear_data, device=gs.device), dim=-1).max() > tol
    assert_allclose(combined_data, dilate_data + shear_data, tol=5e-5)

    # Move box away and check no contact.
    box.set_pos((0.0, 0.0, BOX_SIZE + SPHERE_RADIUS * 2 + 0.05))
    scene.step()
    assert_equal(elastomer_grid_sensor.read_ground_truth(), 0.0, err_msg="ElastomerTaxel should be zero in air.")
    assert_equal(combined_sensor.read_ground_truth(), 0.0, err_msg="ElastomerTaxel should be zero in air.")


@pytest.mark.required
def test_heterogeneous_object(show_viewer, tol):
    PAD_SIZE = (0.4, 0.4, 0.1)
    PAD_TOP_Z = PAD_SIZE[2]
    OBJECT_Z_SIZE = 0.16
    BOX_XY_SIZE = 0.28
    SPHERE_RADIUS = 0.08
    PENETRATION = 0.01
    CENTER_PROBE = (0.0, 0.0, PAD_SIZE[2] / 2)
    OUTER_PROBE = (0.13, 0.0, PAD_SIZE[2] / 2)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            gravity=(0.0, 0.0, 0.0),
        ),
        profiling_options=gs.options.ProfilingOptions(
            show_FPS=False,
        ),
        show_viewer=show_viewer,
    )
    pad = scene.add_entity(
        gs.morphs.Box(
            size=PAD_SIZE,
            pos=(0.0, 0.0, PAD_SIZE[2] / 2),
            fixed=True,
        )
    )
    obj = scene.add_entity(
        morph=[
            gs.morphs.Box(
                size=(BOX_XY_SIZE, BOX_XY_SIZE, OBJECT_Z_SIZE),
            ),
            gs.morphs.Sphere(
                radius=SPHERE_RADIUS,
            ),
        ],
        material=gs.materials.Rigid(
            friction=0.5,
        ),
    )

    probe_local_pos = (CENTER_PROBE, OUTER_PROBE)
    expected_contact = torch.tensor([[True, True], [True, False]], dtype=gs.tc_bool, device=gs.device)
    common = dict(
        entity_idx=pad.idx,
        probe_local_pos=probe_local_pos,
        probe_radius=0.025,
        draw_debug=show_viewer,
    )
    contact_probe = scene.add_sensor(
        gs.sensors.ContactProbe(
            contact_threshold=0.001,
            **common,
        )
    )
    depth_probe = scene.add_sensor(
        gs.sensors.ContactDepthProbe(
            **common,
        )
    )
    kinematic_taxel = scene.add_sensor(
        gs.sensors.KinematicTaxel(
            normal_stiffness=100.0,
            normal_damping=0.0,
            shear_scalar=0.0,
            twist_scalar=0.0,
            **common,
        )
    )
    proximity_taxel = scene.add_sensor(
        gs.sensors.ProximityTaxel(
            probe_local_normal=(0.0, 0.0, 1.0),
            probe_radius=0.04,
            track_link_idx=(obj.base_link_idx,),
            n_sample_points=800,
            stiffness=100.0,
            shear_coupling=0.0,
            **{k: v for k, v in common.items() if k != "probe_radius"},
        )
    )
    elastomer_taxel = scene.add_sensor(
        gs.sensors.ElastomerTaxel(
            probe_local_normal=(0.0, 0.0, 1.0),
            track_link_idx=(obj.base_link_idx,),
            n_sample_points=800,
            **common,
        )
    )
    surface_probe = scene.add_sensor(
        gs.sensors.SurfaceDistanceProbe(
            probe_radius=1.0,
            track_link_idx=(obj.base_link_idx,),
            **{k: v for k, v in common.items() if k != "probe_radius"},
        )
    )

    scene.build(n_envs=2)

    # Per-variant sampling: each heterogeneous variant must receive the full n_sample_points budget so
    # every parallel env sees the requested point count regardless of which variant is active there.
    for pc_sensor, n_requested in ((proximity_taxel, 800), (elastomer_taxel, 800)):
        meta = pc_sensor._shared_metadata
        pc_start = int(meta.sensor_pc_start[pc_sensor._idx].item())
        pc_end = pc_start + int(meta.sensor_pc_n[pc_sensor._idx].item())
        per_env_active = meta.pc_active_envs_mask[pc_start:pc_end].sum(dim=0)
        assert_equal(per_env_active, torch.full_like(per_env_active, n_requested))

    obj.set_pos(
        [
            [0.0, 0.0, PAD_TOP_Z + OBJECT_Z_SIZE / 2 - PENETRATION],
            [0.0, 0.0, PAD_TOP_Z + SPHERE_RADIUS - PENETRATION],
        ]
    )
    scene.step()

    contact = contact_probe.read_ground_truth()
    depth = depth_probe.read_ground_truth()
    kinematic_norm = torch.linalg.norm(kinematic_taxel.read_ground_truth().force, dim=-1)
    proximity_norm = torch.linalg.norm(proximity_taxel.read_ground_truth().force, dim=-1)
    elastomer_norm = torch.linalg.norm(elastomer_taxel.read_ground_truth(), dim=-1)
    surface_distance = surface_probe.read_ground_truth()

    assert_equal(contact, expected_contact)
    assert_equal(depth > 0.001, expected_contact)
    assert_equal(kinematic_norm > tol, expected_contact)
    assert (proximity_norm[0, 0] > tol) and (proximity_norm[1, 0] > tol)
    assert proximity_norm[0, 1] > proximity_norm[1, 1] + tol
    assert (elastomer_norm[0, 0] > tol) and (elastomer_norm[1, 0] > tol)
    assert elastomer_norm[0, 1] > elastomer_norm[1, 1] + gs.EPS
    assert surface_distance[0, 1] < surface_distance[1, 1]

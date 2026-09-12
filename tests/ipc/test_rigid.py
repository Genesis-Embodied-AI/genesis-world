import math
from itertools import permutations
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

try:
    import uipc
except ImportError:
    pytest.skip("IPC Coupler is not supported because 'uipc' module is not available.", allow_module_level=True)

from uipc import builtin

import genesis as gs
import genesis.utils.geom as gu
from genesis.utils.misc import tensor_to_array, geometric_mean, harmonic_mean

from ..conftest import TOL_SINGLE
from ..utils.assertions import assert_allclose
from ..utils.assets import get_hf_dataset
from .utils import (
    find_ipc_geometries,
    get_ipc_merged_geometry,
    get_ipc_positions,
    get_ipc_rigid_links_idx,
)

if TYPE_CHECKING:
    from genesis.engine.couplers import IPCCoupler


@pytest.mark.required
@pytest.mark.parametrize("n_envs, n_substeps", [(0, 1), (2, 1), (0, 2), (2, 2)])
def test_objects_freefall(n_envs, n_substeps, show_viewer):
    from genesis.engine.entities import FEMEntity

    # The interval one advance of the IPC world covers, asked for as the dt the solvers integrate at, so that a step
    # holding several substeps must reach the coupler as the substep rate to integrate the same free fall.
    DT = 0.002
    GRAVITY = np.array([0.0, 0.0, -9.8], dtype=gs.np_float)
    NUM_SUBSTEPS = 30

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=DT * n_substeps,
            gravity=GRAVITY,
        ),
        rigid_options=gs.options.RigidOptions(
            dt=DT,
        ),
        fem_options=gs.options.FEMOptions(
            dt=DT,
        ),
        coupler_options=gs.options.IPCCouplerOptions(
            contact_d_hat=0.01,
            enable_rigid_rigid_contact=False,
            two_way_coupling=True,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(2.2, 3.2, 1.5),
            camera_lookat=(0.0, 0.0, 1.1),
        ),
        show_viewer=show_viewer,
    )

    asset_path = get_hf_dataset(pattern="IPC/grid20x20.obj")
    cloth = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=f"{asset_path}/IPC/grid20x20.obj",
            scale=1.5,
            pos=(0.0, 0.0, 1.5),
            euler=(0, 0, 0),
        ),
        material=gs.materials.FEM.Cloth(
            E=1e5,
            nu=0.499,
            rho=200,
            thickness=0.001,
            bending_stiffness=50.0,
        ),
        surface=gs.surfaces.Plastic(
            color=(0.3, 0.5, 0.8, 1.0),
        ),
    )

    box = scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.2, 0.2, 0.2),
            pos=(0.0, 0.0, 0.6),
        ),
        material=gs.materials.Rigid(
            rho=500.0,
            coup_type="ipc_only",
        ),
        surface=gs.surfaces.Plastic(
            color=(0.8, 0.3, 0.2, 0.8),
        ),
    )

    sphere = scene.add_entity(
        morph=gs.morphs.Sphere(
            radius=0.08,
            pos=(0.5, 0.0, 0.1),
        ),
        material=gs.materials.FEM.Elastic(
            E=1.0e5,
            nu=0.3,
            rho=1000.0,
            model="stable_neohookean",
        ),
        surface=gs.surfaces.Plastic(
            color=(0.2, 0.8, 0.3, 0.8),
        ),
    )

    scene.build(n_envs=n_envs)
    assert scene.sim is not None
    assert scene.sim.substeps == n_substeps
    coupler = cast("IPCCoupler", scene.sim.coupler)

    envs_idx = range(max(scene.n_envs, 1))

    ipc_links_idx = get_ipc_rigid_links_idx(scene, env_idx=0)
    assert box.base_link_idx in ipc_links_idx
    assert box.base_link in coupler._abd_slots_by_link

    # Verify that geometries are present in IPC for each environment
    cloth_entity_idx = scene.sim.fem_solver.entities.index(cloth)
    box_entity_idx = scene.sim.rigid_solver.entities.index(box)
    sphere_entity_idx = scene.sim.fem_solver.entities.index(sphere)
    objs_kwargs = {
        obj: dict(solver_type=solver_type, idx=idx)
        for obj, solver_type, idx in (
            (cloth, "cloth", cloth_entity_idx),
            (box, "rigid", box_entity_idx),
            (sphere, "fem", sphere_entity_idx),
        )
    }
    for obj_kwargs in objs_kwargs.values():
        for env_idx in envs_idx:
            assert len(find_ipc_geometries(scene, **obj_kwargs, env_idx=env_idx)) == 1

    # Get initial state
    p_0 = {obj: get_ipc_positions(scene, **obj_kwargs, envs_idx=envs_idx) for obj, obj_kwargs in objs_kwargs.items()}
    v_0 = {obj: np.zeros_like(p_0[obj]) for obj in objs_kwargs.keys()}

    # Run simulation and validate dynamics equations at each step
    p_prev, v_prev = p_0.copy(), v_0.copy()
    for _i in range(NUM_SUBSTEPS // n_substeps):
        # Move forward in time
        scene.step()

        for obj, obj_kwargs in objs_kwargs.items():
            # Get new position
            p_i = get_ipc_positions(scene, **obj_kwargs, envs_idx=envs_idx)

            # Estimate velocity by finite difference over the step, which the Euler scheme makes the mean over its
            # substeps: (x_{n+m} - x_n) / (m * DT)
            v_i = (p_i - p_prev[obj]) / (n_substeps * DT)

            # Compute estimated position and velocity, accumulating one Euler substep at a time
            expected_v = v_prev[obj] + GRAVITY * DT * (n_substeps + 1) / 2
            expected_p = p_prev[obj] + expected_v * (n_substeps * DT)

            # Update for next iteration
            p_prev[obj], v_prev[obj] = p_i, v_prev[obj] + GRAVITY * DT * n_substeps

            # FIXME: This test does not pass for sphere entity...
            if obj is sphere:
                continue

            # Validate displacement and velocity assuming Euler scheme
            assert_allclose(v_i, expected_v, atol=1e-3)
            assert_allclose(p_i, expected_p, tol=TOL_SINGLE)

    for obj in objs_kwargs.keys():
        # Validate centroid consistency
        ipc_centroid = p_prev[obj].mean(axis=-2)
        gs_centroid = obj.get_state().pos.mean(axis=-2)
        assert_allclose(ipc_centroid, gs_centroid, atol=TOL_SINGLE)

        # Validate centroidal total displacement: 0.5 * GRAVITY * t * (t + DT)
        # FEM entities (cloth) deform during freefall, causing small centroid drift — use looser tolerance.
        p_delta = p_prev[obj] - p_0[obj]
        expected_displacement = 0.5 * GRAVITY * NUM_SUBSTEPS * (NUM_SUBSTEPS + 1) * DT**2
        assert_allclose(p_delta.mean(axis=-2), expected_displacement, tol=2e-3 if isinstance(obj, FEMEntity) else 1e-3)

        # FIXME: This test does not pass for sphere entity...
        if obj is sphere:
            continue

        # Validate vertex-based total displacement
        assert_allclose(p_delta, expected_displacement, tol=TOL_SINGLE)


@pytest.mark.slow  # ~200s
@pytest.mark.parametrize("n_envs", [0, 2])
def test_ground_clearance(n_envs, show_viewer):
    GRAVITY = np.array([0.0, 0.0, -9.8], dtype=gs.np_float)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.005,
            gravity=GRAVITY,
        ),
        coupler_options=gs.options.IPCCouplerOptions(
            contact_d_hat=0.01,
            contact_resistance=1e6,
            enable_rigid_rigid_contact=False,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.5, 0.0, 0.1),
            camera_lookat=(0.0, 0.0, 0.0),
        ),
        show_viewer=show_viewer,
    )

    scene.add_entity(
        gs.morphs.Plane(),
        material=gs.materials.Rigid(
            rho=200.0,
            coup_type="ipc_only",
        ),
    )

    cubes = []
    for y, resistance in ((-0.4, 1e2), (-0.2, 1e3), (0.0, 1e4), (0.2, 1e5), (0.4, 1e6)):
        cube = scene.add_entity(
            gs.morphs.Box(
                pos=(0.0, y, 0.05),
                size=(0.08, 0.08, 0.08),
            ),
            material=gs.materials.Rigid(
                rho=200.0,
                coup_friction=0.0,
                coup_type="ipc_only",
                contact_resistance=resistance,
            ),
        )
        cubes.append(cube)

    scene.build(n_envs=n_envs)

    initial_positions = np.stack([tensor_to_array(cube.get_pos()) for cube in cubes], axis=-2)

    dist = []
    for _ in range(70):
        scene.step()
    for _ in range(20):
        scene.step()
        dist.append(np.stack([tensor_to_array(cube.get_verts())[..., 2].min(axis=-1) for cube in cubes], axis=-1))
    dist = np.stack(dist, axis=-1)

    final_positions = np.stack([tensor_to_array(cube.get_pos()) for cube in cubes], axis=-2)

    # No lateral driving force in x/y; drift should stay small.
    assert_allclose(initial_positions[..., :2], final_positions[..., :2], atol=TOL_SINGLE)

    # Make sure that it reaches equilibrium
    assert_allclose(dist[..., -1], dist[..., -2], tol=TOL_SINGLE)

    # Larger contact resistance should produce larger ground clearance (less penetration/compression).
    assert (np.diff(dist, axis=-2) > TOL_SINGLE).all()


@pytest.mark.parametrize("n_envs", [0, 2])
def test_ground_sliding(n_envs, show_viewer):
    GRAVITY = np.array([5.0, 0.0, -10.0], dtype=gs.np_float)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
            gravity=GRAVITY,
        ),
        coupler_options=gs.options.IPCCouplerOptions(
            contact_d_hat=0.01,
            enable_rigid_rigid_contact=False,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3.5, 2.0, 1.5),
            camera_lookat=(1.0, -0.5, 0.0),
        ),
        show_viewer=show_viewer,
    )

    scene.add_entity(
        gs.morphs.Plane(),
        material=gs.materials.Rigid(
            coup_friction=0.25,
            coup_type="ipc_only",
        ),
    )

    cubes = []
    for y, mu in ((-0.4, 0.0), (-0.2, 0.01), (0.0, 0.04), (0.2, 0.09), (0.4, 0.16)):
        cube = scene.add_entity(
            gs.morphs.Box(
                pos=(0.0, y, 0.12),
                size=(0.08, 0.08, 0.08),
            ),
            material=gs.materials.Rigid(
                coup_friction=mu,
                coup_type="ipc_only",
            ),
        )
        cubes.append(cube)

    scene.build(n_envs=n_envs)

    initial_positions = np.stack([tensor_to_array(cube.get_pos()) for cube in cubes], axis=-2)
    for _ in range(100):
        scene.step()
    final_positions = np.stack([tensor_to_array(cube.get_pos()) for cube in cubes], axis=-2)

    # Coarse non-penetration sanity check
    assert (final_positions[..., 2] > 0.0).all()

    # Distance from ground should be friction-independent
    assert_allclose(np.diff(final_positions[..., 2], axis=-1), 0.0, tol=TOL_SINGLE)

    # No y-axis driving force: lateral drift should be minimal
    assert_allclose(initial_positions[..., 1], final_positions[..., 1], tol=TOL_SINGLE)

    # All cubes should move along +x under tilted gravity.
    assert ((final_positions[..., 0] - initial_positions[..., 0]) > 0.5).all()

    # Lower coup_friction should slide farther, so x should strictly decrease as mu increases.
    assert (np.diff(final_positions[..., ::-1, 0], axis=-1) > 0.2).all()


@pytest.mark.slow  # ~250s
@pytest.mark.parametrize("enable_rigid_rigid_contact", [False, True])
def test_contact_pair_friction_resistance(enable_rigid_rigid_contact):
    from genesis.engine.entities import RigidEntity

    scene = gs.Scene(
        coupler_options=gs.options.IPCCouplerOptions(
            contact_resistance=36.0,
            enable_rigid_rigid_contact=enable_rigid_rigid_contact,
        ),
        show_viewer=False,
    )

    plane = scene.add_entity(
        gs.morphs.Plane(),
        material=gs.materials.Rigid(
            coup_type="ipc_only",
        ),
    )
    rigid_a = scene.add_entity(
        gs.morphs.Box(
            pos=(0.0, 0.0, 0.12),
            size=(0.05, 0.05, 0.05),
        ),
        material=gs.materials.Rigid(
            coup_friction=0.25,
            coup_type="ipc_only",
            contact_resistance=9.0,
        ),
    )
    rigid_b = scene.add_entity(
        gs.morphs.Box(
            pos=(0.2, 0.0, 0.12),
            size=(0.05, 0.05, 0.05),
        ),
        material=gs.materials.Rigid(
            coup_friction=0.64,
            coup_type="ipc_only",
            contact_resistance=16.0,
        ),
    )
    rigid_c = scene.add_entity(
        gs.morphs.Box(
            pos=(-0.2, 0.0, 0.12),
            size=(0.05, 0.05, 0.05),
        ),
        material=gs.materials.Rigid(
            coup_friction=0.16,
            coup_type="ipc_only",
            contact_resistance=None,
        ),
    )
    fem = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.4, 0.0, 0.12),
            size=(0.05, 0.05, 0.05),
        ),
        material=gs.materials.FEM.Elastic(
            E=5e4,
            nu=0.35,
            rho=1000.0,
            friction_mu=0.49,
            contact_resistance=25.0,
        ),
    )

    scene.build()
    assert scene.sim is not None
    coupler = cast("IPCCoupler", scene.sim.coupler)

    tab = coupler._ipc_scene.contact_tabular()
    for entities in permutations((plane, rigid_a, rigid_b, rigid_c, fem), 2):
        elems_idx = []
        frictions = []
        resistances = []
        for entity in entities:
            if isinstance(entity, RigidEntity):
                if entity is plane:
                    elem = coupler._ipc_ground_contacts[entity]
                else:
                    elem = coupler._ipc_abd_contacts[entity]
                friction = entity.material.coup_friction
            else:
                elem = coupler._ipc_fem_contacts[entity]
                friction = entity.material.friction_mu
            resistance = entity.material.contact_resistance or coupler.options.contact_resistance
            elems_idx.append(elem.id())
            frictions.append(friction)
            resistances.append(resistance)
        model = tab.at(*elems_idx)
        assert model.friction_rate() == pytest.approx(geometric_mean(*frictions))
        assert model.resistance() == pytest.approx(harmonic_mean(*resistances))
        assert model.is_enabled() ^ (
            all(isinstance(entity, RigidEntity) and entity is not plane for entity in entities)
            and not enable_rigid_rigid_contact
        )


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
def test_objects_colliding(n_envs, show_viewer):
    DT = 0.02
    CONTACT_MARGIN = 0.01
    GRAVITY = np.array([0.0, 0.0, -9.8], dtype=gs.np_float)
    NUM_STEPS = 90

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=DT,
            gravity=GRAVITY,
        ),
        coupler_options=gs.options.IPCCouplerOptions(
            contact_d_hat=CONTACT_MARGIN,
            enable_rigid_rigid_contact=False,
            two_way_coupling=True,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(2.0, 2.0, 0.1),
            camera_lookat=(0.0, 0.0, 0.1),
        ),
        show_viewer=show_viewer,
    )

    scene.add_entity(
        gs.morphs.Plane(),
        material=gs.materials.Rigid(
            coup_friction=0.5,
            coup_type="ipc_only",
        ),
    )

    asset_path = get_hf_dataset(pattern="IPC/grid20x20.obj")
    cloth = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=f"{asset_path}/IPC/grid20x20.obj",
            scale=1.5,
            pos=(0.0, 0.0, 0.2),
            euler=(90, 0, 0),
        ),
        material=gs.materials.FEM.Cloth(
            E=1e5,
            nu=0.499,
            rho=200,
            thickness=0.001,
            bending_stiffness=50.0,
        ),
        surface=gs.surfaces.Plastic(
            color=(0.3, 0.5, 0.8, 1.0),
        ),
    )

    box = scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(-0.25, 0.0, 0.1),
        ),
        material=gs.materials.Rigid(
            rho=500.0,
            coup_friction=0.3,
            coup_type="ipc_only",
        ),
        surface=gs.surfaces.Plastic(
            color=(0.8, 0.3, 0.2, 0.8),
        ),
    )

    sphere = scene.add_entity(
        morph=gs.morphs.Sphere(
            radius=0.08,
            pos=(0.25, 0.0, 0.1),
        ),
        material=gs.materials.FEM.Elastic(
            E=1.0e3,
            nu=0.3,
            rho=1000.0,
            friction_mu=0.3,
            model="stable_neohookean",
        ),
        surface=gs.surfaces.Plastic(
            color=(0.2, 0.8, 0.3, 0.8),
        ),
    )

    scene.build(n_envs=n_envs)
    assert scene.sim is not None
    envs_idx = range(max(scene.n_envs, 1))

    # Run simulation and validate dynamics equations at each step
    objs_kwargs = {
        obj: dict(solver_type=solver_type, idx=idx)
        for obj, solver_type, idx in (
            (cloth, "cloth", scene.sim.fem_solver.entities.index(cloth)),
            (box, "rigid", scene.sim.rigid_solver.entities.index(box)),
            (sphere, "fem", scene.sim.fem_solver.entities.index(sphere)),
        )
    }
    p_history = {obj: [] for obj in objs_kwargs.keys()}
    for _i in range(NUM_STEPS):
        scene.step()

        for obj, obj_kwargs in objs_kwargs.items():
            p_i = get_ipc_positions(scene, **obj_kwargs, envs_idx=envs_idx)
            p_history[obj].append(p_i)

    cloth_p_history = np.stack(p_history[cloth], axis=-3)
    for obj in objs_kwargs.keys():
        obj_p_history = np.stack(p_history[obj], axis=-3)

        # Make sure that all vertices are laying on the ground
        assert (obj_p_history[..., 2] < 1.5 * CONTACT_MARGIN).any()
        assert (obj_p_history[..., 2] > 0.0).all()

        # Check that the objects did not fly away (5cm)
        obj_delta_history = np.linalg.norm((obj_p_history - obj_p_history[..., [0], :, :])[..., :2], axis=-1)
        assert_allclose(obj_delta_history, 0.0, atol=0.1)

        # Make sure that all objects reached steady state
        obj_disp_history = np.linalg.norm(np.diff(obj_p_history[..., -10:, :, :], axis=-3), axis=-1)
        assert_allclose(obj_disp_history, 0.0, tol=5e-3)

        # Make sure that the cloth is laying on all objects (at least one vertex above the others)
        if obj is cloth:
            continue
        assert (obj_p_history[..., 2].max(axis=-1) < cloth_p_history[..., 2].max(axis=-1)).all()


@pytest.mark.slow  # ~200s
@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
def test_momentum_conservation(n_envs, show_viewer):
    DT = 0.001
    DURATION = 0.30
    CONTACT_MARGIN = 0.01
    VELOCITY = np.array([4.0, 0.0, 0.0], dtype=gs.np_float)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=DT,
            gravity=(0.0, 0.0, 0.0),
        ),
        coupler_options=gs.options.IPCCouplerOptions(
            contact_d_hat=CONTACT_MARGIN,
            constraint_strength_translation=1,
            constraint_strength_rotation=1,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.5, 1.3, 0.6),
            camera_lookat=(0.2, 0.0, 0.3),
        ),
        show_viewer=show_viewer,
    )

    blob = scene.add_entity(
        morph=gs.morphs.Sphere(
            pos=(0.3, 0.0, 0.4),
            radius=0.1,
        ),
        material=gs.materials.FEM.Elastic(
            E=1.0e5,
            nu=0.45,
            rho=1000.0,
            friction_mu=0.0,
            model="stable_neohookean",
        ),
    )

    rigid_cube = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.0, 0.0, 0.4),
            size=(0.1, 0.1, 0.1),
            euler=(0, 0, 0),
        ),
        material=gs.materials.Rigid(
            rho=1000,
            coup_type="two_way_soft_constraint",
        ),
        surface=gs.surfaces.Plastic(
            color=(0.8, 0.2, 0.2, 0.8),
        ),
    )

    scene.build(n_envs=n_envs)
    assert scene.sim is not None
    coupler = cast("IPCCoupler", scene.sim.coupler)

    rigid_cube.set_dofs_velocity((*VELOCITY, 0.0, 0.0, 0.0))

    fem_entity_idx = scene.sim.fem_solver.entities.index(blob)
    assert len(find_ipc_geometries(scene, solver_type="fem", idx=fem_entity_idx, env_idx=0)) == 1

    rigid_link = rigid_cube.base_link
    ipc_links_idx = get_ipc_rigid_links_idx(scene, env_idx=0)
    assert rigid_link.idx in ipc_links_idx
    assert rigid_link in coupler._abd_slots_by_link

    cube_mass = tensor_to_array(rigid_cube.get_mass())

    # Read actual FEM mass from IPC geometry (mesh mass != analytical sphere mass due to tet discretization).
    blob_radius = blob.morph.radius
    blob_rho = blob.material.rho
    blob_analytical_mass = (4.0 / 3.0) * np.pi * blob_radius**3 * blob_rho
    (fem_raw_geo,) = find_ipc_geometries(scene, solver_type="fem", idx=fem_entity_idx, env_idx=0)
    fem_mass_density = fem_raw_geo.meta().find(builtin.mass_density).view().item()
    fem_merged_geo = get_ipc_merged_geometry(scene, solver_type="fem", idx=fem_entity_idx, env_idx=0)
    fem_vertex_volumes = fem_merged_geo.vertices().find(builtin.volume).view().reshape(-1)
    blob_mass = float(np.sum(fem_vertex_volumes) * fem_mass_density)
    assert_allclose(blob_mass, blob_analytical_mass, rtol=0.01)

    total_p_history = []
    momentum_0 = VELOCITY * cube_mass[..., None]

    dist_min = np.array(float("inf"))
    fem_positions_prev = None  # FEM initial velocity is zero
    for step in range(int(DURATION / DT)):
        cube_vel = tensor_to_array(
            rigid_cube.get_links_vel(links_idx_local=0, ref=gs.link_ref_frame.link_COM)[..., 0, :]
        )
        rigid_linear_momentum = cube_mass[..., None] * cube_vel

        fem_proc_geo = get_ipc_merged_geometry(scene, solver_type="fem", idx=fem_entity_idx, env_idx=0)
        fem_positions = fem_proc_geo.positions().view().squeeze(axis=-1)
        if fem_positions_prev is not None:
            fem_velocities = (fem_positions - fem_positions_prev) / DT
        else:
            fem_velocities = np.zeros_like(fem_positions)
        fem_positions_prev = fem_positions

        # Make sure that rigid and fem are not penetrating each other
        fem_aabb_min, fem_aabb_max = fem_positions.min(axis=-2), fem_positions.max(axis=-2)
        rigid_aabb = tensor_to_array(rigid_cube.get_AABB())
        rigid_aabb_min, rigid_aabb_max = rigid_aabb[..., 0, :], rigid_aabb[..., 1, :]
        overlap = np.minimum(fem_aabb_max, rigid_aabb_max) - np.maximum(rigid_aabb_min, fem_aabb_min)
        dist_min = np.minimum(dist_min, -overlap.min(axis=-1))
        assert (dist_min > 0.0).all()

        volume_attr = fem_proc_geo.vertices().find(builtin.volume)
        fem_vertex_masses = volume_attr.view().reshape(-1) * fem_mass_density
        assert_allclose(np.sum(fem_vertex_masses), blob_mass, tol=TOL_SINGLE)
        fem_linear_momentum = np.sum(fem_vertex_masses[:, np.newaxis] * fem_velocities, axis=0)

        # Before collision: FEM should have zero momentum, rigid should carry all momentum.
        if step < int(DURATION / 10 / DT):
            assert_allclose(fem_linear_momentum, 0.0, atol=TOL_SINGLE)
            assert_allclose(rigid_linear_momentum, momentum_0, tol=TOL_SINGLE)

        total_linear_momentum = rigid_linear_momentum + fem_linear_momentum
        total_p_history.append(total_linear_momentum)

        scene.step()

    # Make sure the objects bounced on each other
    assert (dist_min < 1.5 * CONTACT_MARGIN).all()
    assert (cube_vel[..., 0] < -0.5).all()
    assert (fem_velocities[..., 0].mean(axis=-1) > 0.5).all()

    # Check total momentum conservation.
    # NOTE : The tet mesh's contact-facing vertices (x < -0.05) have a z-mean of -0.00138 due to TetGen's asymmetric
    # Steiner point insertion, causing an asymmetric contact force distribution during the x-direction collision.
    # This z-bias produces a net -z impulse, resulting in the observed z-momentum leak.
    assert_allclose(total_p_history, momentum_0, tol=0.001)


@pytest.mark.slow  # ~200s
@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
@pytest.mark.parametrize(
    "coup_type, fixed",
    [("two_way_soft_constraint", True), ("two_way_soft_constraint", False), ("external_articulation", True)],
)
@pytest.mark.parametrize("joint_type", ["revolute", "prismatic"])
def test_single_joint(n_envs, coup_type, joint_type, fixed, show_viewer):
    DT = 0.01
    GRAVITY = np.array([0.0, 0.0, -9.8], dtype=gs.np_float)
    POS = (0, 0, 0.5)
    FREQ = 1.0
    SCALE = 0.5 if joint_type == "revolute" else 0.1
    CONTACT_MARGIN = 0.01

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=DT,
            gravity=GRAVITY,
        ),
        rigid_options=gs.options.RigidOptions(
            enable_collision=False,
        ),
        coupler_options=gs.options.IPCCouplerOptions(
            newton_tolerance=1e-2,
            newton_translation_tolerance=1e-2,
            newton_semi_implicit_enable=False,
            linear_system_tolerance=1e-3,
            contact_d_hat=CONTACT_MARGIN,
            constraint_strength_translation=1,
            constraint_strength_rotation=1,
            enable_rigid_rigid_contact=False,
            two_way_coupling=True,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.0, 1.0, 0.8),
            camera_lookat=(0.0, 0.0, 0.3),
        ),
        show_viewer=show_viewer,
    )

    scene.add_entity(
        gs.morphs.Plane(),
        material=gs.materials.Rigid(
            coup_friction=0.5,
            coup_type="ipc_only",
        ),
    )

    robot = scene.add_entity(
        morph=gs.morphs.URDF(
            file=f"urdf/simple/two_cube_{joint_type}.urdf",
            pos=POS,
            fixed=fixed,
        ),
        material=gs.materials.Rigid(
            coup_type=coup_type,
        ),
    )

    scene.build(n_envs=n_envs)
    assert scene.sim is not None
    coupler = cast("IPCCoupler", scene.sim.coupler)

    envs_idx = range(max(scene.n_envs, 1))

    robot.set_dofs_kp(500.0, dofs_idx_local=-1)
    robot.set_dofs_kv(50.0, dofs_idx_local=-1)

    moving_link = robot.get_link("moving")
    ipc_links_idx = get_ipc_rigid_links_idx(scene, env_idx=0)
    assert moving_link.idx in ipc_links_idx
    assert moving_link in coupler._abd_slots_by_link
    if coup_type == "two_way_soft_constraint":
        assert moving_link in coupler._abd_data_by_link
    elif coup_type == "external_articulation":
        art_data = coupler._articulation_data_by_entity[robot]
        assert len(art_data.articulation_slots) == max(scene.n_envs, 1)
        if fixed:
            assert not coupler._abd_data_by_link

    dist_min = np.array(float("inf"))
    cur_dof_pos_history, target_dof_pos_history = [], []
    gs_transform_history, ipc_transform_history = [], []
    for _ in range(int(1 / (DT * FREQ))):
        # Apply sinusoidal target position
        target_dof_pos = SCALE * np.sin((2 * math.pi * FREQ) * scene.sim.cur_t)
        target_dof_vel = SCALE * (2 * math.pi * FREQ) * np.cos((2 * math.pi * FREQ) * scene.sim.cur_t)
        robot.control_dofs_position_velocity(target_dof_pos, target_dof_vel, dofs_idx_local=-1)

        # Store the current and target position / velocity
        cur_dof_pos = tensor_to_array(robot.get_dofs_position(dofs_idx_local=-1)[..., 0])
        cur_dof_pos_history.append(cur_dof_pos)
        target_dof_pos_history.append(target_dof_pos)

        # Make sure the robot never went through the ground
        if not fixed:
            robot_verts = tensor_to_array(robot.get_verts())
            dist_min = np.minimum(dist_min, robot_verts[..., 2].min(axis=-1))
            # FIXME: For some reason it actually can...
            assert (dist_min > -0.1).all()

        scene.step()

        if coup_type == "two_way_soft_constraint" or not fixed:
            for env_idx in envs_idx:
                abd_data = coupler._abd_data_by_link[moving_link][env_idx]
                gs_transform = coupler._abd_transforms_by_link[moving_link][env_idx]
                ipc_transform = abd_data.transform
                # FIXME: Why the tolerance is must so large if no fixed ?!
                assert_allclose(gs_transform[:3, 3], ipc_transform[:3, 3], atol=TOL_SINGLE if fixed else 0.2)
                assert_allclose(
                    gu.R_to_xyz(gs_transform[:3, :3] @ ipc_transform[:3, :3].T), 0.0, atol=1e-4 if fixed else 0.3
                )
                gs_transform_history.append(gs_transform)
                ipc_transform_history.append(ipc_transform)
    cur_dof_pos_history = np.stack(cur_dof_pos_history, axis=-1)
    target_dof_pos_history = np.stack(target_dof_pos_history, axis=-1)

    for env_idx in envs_idx if scene.n_envs > 0 else (slice(None),):
        corr = np.corrcoef(cur_dof_pos_history[env_idx], target_dof_pos_history)[0, 1]
        assert corr > 1.0 - 5e-3
    assert_allclose(
        cur_dof_pos_history - cur_dof_pos_history[..., [0]],
        target_dof_pos_history - target_dof_pos_history[..., [0]],
        tol=0.03,
    )
    assert_allclose(np.ptp(cur_dof_pos_history, axis=-1), 2 * SCALE, tol=0.05)

    if gs_transform_history:
        gs_pos_history, gs_quat_history = gu.T_to_trans_quat(np.stack(gs_transform_history, axis=0))
        ipc_pos_history, ipc_quat_history = gu.T_to_trans_quat(np.stack(ipc_transform_history, axis=0))
        pos_err_history = np.linalg.norm(ipc_pos_history - gs_pos_history, axis=-1)
        rot_err_history = np.linalg.norm(
            gu.quat_to_rotvec(gu.transform_quat_by_quat(gs.inv_quat(gs_quat_history), ipc_quat_history)), axis=-1
        )
        assert (np.percentile(pos_err_history, 90, axis=0) < 1e-2).all()
        assert (np.percentile(rot_err_history, 90, axis=0) < 5e-2).all()

    # Make sure the robot bounced on the ground or stayed in place
    if fixed:
        assert_allclose(robot.get_pos(), POS, atol=TOL_SINGLE)
    else:
        assert (dist_min < 1.5 * CONTACT_MARGIN).all()


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
@pytest.mark.parametrize("constraint_strength", [1, 100])
def test_apply_forces_base_link(n_envs, constraint_strength, show_viewer):
    DT = 0.002
    FREQ = 2.0
    SCALE = 0.1
    GRAVITY = np.array([0.0, 0.0, -9.8], dtype=gs.np_float)
    POS = (0.5, 0.0, 0.0)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=DT,
            gravity=GRAVITY,
        ),
        coupler_options=gs.options.IPCCouplerOptions(
            constraint_strength_translation=constraint_strength,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.5, -0.5, 0.3),
            camera_lookat=(0.25, 0.0, 0.0),
        ),
        show_viewer=show_viewer,
    )

    box = scene.add_entity(
        gs.morphs.Box(
            size=(0.05, 0.05, 0.05),
            pos=POS,
        ),
        material=gs.materials.Rigid(
            coup_type="two_way_soft_constraint",
        ),
    )

    scene.build(n_envs=n_envs)
    assert scene.sim is not None

    box.set_dofs_kp(50000.0)
    box.set_dofs_kv(500.0)

    z_actual, z_target = [], []
    for _ in range(int(1 / (DT * FREQ))):
        t = scene.sim.cur_t
        target_z = SCALE * math.sin((2 * math.pi * FREQ) * t)
        target_vz = SCALE * (2 * math.pi * FREQ) * math.cos((2 * math.pi * FREQ) * t)
        box.control_dofs_position_velocity(target_z, target_vz, dofs_idx_local=2)
        scene.step()
        z_target.append(target_z)
        z_actual.append(tensor_to_array(box.get_pos()[..., 2]))

    z_actual = np.array(z_actual)
    z_target = np.array(z_target)
    if z_actual.ndim > 1:
        z_target = z_target[:, np.newaxis]
    assert_allclose(z_actual, z_target, atol=0.005)

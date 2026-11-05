import weakref

import numpy as np
import pytest
import torch
import trimesh

import quadrants as qd

import genesis as gs

from ..utils.assertions import assert_allclose, assert_equal


@pytest.mark.required
def test_particle_constraints(show_viewer, tol):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1e-2,
        ),
        mpm_options=gs.options.MPMOptions(
            grid_density=64,
            lower_bound=(-1.0, -1.0, 0.0),
            upper_bound=(1.0, 1.0, 1.0),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.5, -1.2, 1.0),
            camera_lookat=(0.0, 0.3, 0.35),
            camera_fov=32,
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )
    scene.add_entity(gs.morphs.Plane())
    rigid_box = scene.add_entity(
        gs.morphs.Box(
            pos=(0, 0, 0.55),
            size=(0.12, 0.12, 0.05),
            fixed=True,
        ),
    )
    mpm_cube = scene.add_entity(
        material=gs.materials.MPM.Elastic(
            E=5e4,
            nu=0.3,
            rho=1000,
            sampler="regular",
        ),
        morph=gs.morphs.Box(
            pos=(0, 0, 0.35),
            size=(0.15, 0.15, 0.15),
        ),
    )
    # Two copies of one box offset along its local y axis, each placed with its own member pose
    box = trimesh.creation.box(
        extents=(0.04, 0.12, 0.04), transform=trimesh.transformations.translation_matrix((0.0, 0.1, 0.0))
    )
    mpm_mesh_set = scene.add_entity(
        morph=gs.morphs.MeshSet(
            files=(box, box),
            poss=((0.5, 0.5, 0.3), (-0.5, 0.5, 0.3)),
            eulers=((0.0, 0.0, 90.0), (90.0, 0.0, 0.0)),
        ),
        material=gs.materials.MPM.Elastic(
            sampler="regular",
        ),
    )
    scene.build(n_envs=2)

    # Each member is rotated about its own position: the box offset lands at -x for the member rotated about z and at
    # +z for the one rotated about x, and the long axis follows. A regular lattice spans the box minus one particle.
    particles = mpm_mesh_set.init_particles
    particle_size = mpm_mesh_set.particle_size
    long_span, short_span = 0.12 - particle_size, 0.04 - particle_size
    particles_z90 = particles[particles[:, 0] > 0.0]
    particles_x90 = particles[particles[:, 0] < 0.0]
    assert_allclose(particles_z90.mean(axis=0), (0.4, 0.5, 0.3), tol=tol)
    assert_allclose(np.ptp(particles_z90, axis=0), (long_span, short_span, short_span), tol=tol)
    assert_allclose(particles_x90.mean(axis=0), (-0.5, 0.5, 0.4), tol=tol)
    assert_allclose(np.ptp(particles_x90, axis=0), (short_span, short_span, long_span), tol=tol)
    # The visual mesh carries the same member poses
    assert_allclose(mpm_mesh_set.vmesh.verts.min(axis=0), (-0.52, 0.48, 0.28), tol=tol)
    assert_allclose(mpm_mesh_set.vmesh.verts.max(axis=0), (0.46, 0.52, 0.46), tol=tol)
    # Positioning the set as a whole moves the mean member position there, keeping the members' layout
    mpm_mesh_set.set_position((0.0, 0.5, 0.5))
    assert_allclose(mpm_mesh_set.get_particles_pos(), particles + (0.0, 0.0, 0.2), tol=tol)

    # Test get_particles_in_bbox - returns (n_envs, n_particles) mask
    mask = mpm_cube.get_particles_in_bbox((-0.08, -0.08, 0.41), (0.08, 0.08, 0.44))
    assert mask.shape == (2, mpm_cube.n_particles), "mask should be (n_envs, n_particles)"
    assert mask.any(), "bbox should select some particles"
    assert not mask.all(), "bbox should not select all particles"

    # Attach and test following
    link_idx = rigid_box.links[0].idx
    mpm_cube.set_particle_constraints(mask, link_idx, stiffness=1e5)
    initial_rigid_pos = rigid_box.get_pos()
    initial_mpm_x = mpm_cube.get_particles_pos()[:, mask[0], 0].mean()

    pos_diff = torch.tensor([0.2, 0, 0], device=gs.device)
    rigid_box.set_pos(initial_rigid_pos + pos_diff, zero_velocity=False)
    for _ in range(30):
        scene.step()

    # The attached particles follow the link, the springs keeping every particle finite whatever their stiffness
    assert mpm_cube.get_state().vel.isfinite().all()
    mpm_diff = mpm_cube.get_particles_pos()[:, mask[0], 0].mean() - initial_mpm_x
    # The springs deflect under the pull of the rest of the block, so the attached particles lag the link by a physical
    # offset that precision does not bound
    assert_allclose(mpm_diff, pos_diff[0], tol=1e-4)


@pytest.mark.required
@pytest.mark.parametrize("variant", ["svd", "no_svd"])
def test_perf_dispatch(variant, show_viewer):
    if variant == "svd":
        elastic_model = "corotation"
        liquid_viscous = True
    else:
        elastic_model = "neohooken"
        liquid_viscous = False

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1e-2,
        ),
        mpm_options=gs.options.MPMOptions(
            grid_density=64,
            lower_bound=(-0.4, -0.4, -0.05),
            upper_bound=(0.4, 0.4, 0.5),
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(
        morph=gs.morphs.Plane(),
    )
    elastic = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(-0.1, 0.0, 0.2),
            size=(0.08, 0.08, 0.08),
        ),
        material=gs.materials.MPM.Elastic(
            E=3e5,
            nu=0.2,
            rho=200.0,
            model=elastic_model,
            sampler="regular",
        ),
    )
    liquid = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.15, 0.0, 0.2),
            size=(0.08, 0.08, 0.08),
        ),
        material=gs.materials.MPM.Liquid(
            viscous=liquid_viscous,
            sampler="regular",
        ),
    )
    scene.build(n_envs=2)

    init_elastic_pos = elastic.get_particles_pos()
    init_liquid_pos = liquid.get_particles_pos()

    for _ in range(20):
        scene.step()

    final_elastic_pos = elastic.get_particles_pos()
    final_liquid_pos = liquid.get_particles_pos()

    # Gravity acted: particles moved down on average.
    assert init_elastic_pos[..., 2].mean() - final_elastic_pos[..., 2].mean() > 0.05
    assert init_liquid_pos[..., 2].mean() - final_liquid_pos[..., 2].mean() > 0.05

    # The ground (plane at z=0) holds the particles. Its soft contact rests at a penetration set by its impedance, so
    # only a particle sinking past half a grid cell has gone through it.
    dx = 1.0 / 64
    assert final_elastic_pos[..., 2].min() > -0.5 * dx
    assert final_liquid_pos[..., 2].min() > -0.5 * dx


def _falling_block(n_envs, show_viewer):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1e-2,
        ),
        mpm_options=gs.options.MPMOptions(
            lower_bound=(-0.5, -0.5, -0.1),
            upper_bound=(0.5, 0.5, 0.5),
            grid_density=32,
        ),
        # Converged far below the rounding the order of the sums of the reductions leaves, so that a solve does not
        # stop an iteration apart from the same solve in another batch
        newton_options=gs.options.NewtonOptions(
            velocity_tol=1e-10,
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(
        morph=gs.morphs.Plane(),
    )
    # Off the faces of the grid cells: the regular lattice of a block centered on a node puts particles on faces, whose
    # cell the rounding of their position then picks, and the motion jumps with it (see MPM_FD_EPS in test_hybrid_push)
    block = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.003, 0.002, 0.1),
            size=(0.1, 0.1, 0.1),
        ),
        material=gs.materials.MPM.Elastic(
            sampler="regular",
        ),
    )
    scene.build(n_envs=n_envs)
    return scene, block


@pytest.mark.required
# fp32 cannot resolve the Newton tolerance the comparison of two batches needs
@pytest.mark.precision("64")
def test_envs_solve_apart(precision, show_viewer):
    # The environments of a batch run their own Newton solves: thrown at different speeds, the block of each one moves
    # as it does alone, the solves of the block hitting the plane running up to four times as many iterations as those
    # of the block at rest
    velocities = ((0.0, 0.0, 0.0), (0.0, 0.0, -1.0), (1.0, 0.0, -0.5))
    scene, block = _falling_block(len(velocities), show_viewer)
    block.set_velocity(velocities)
    for _ in range(10):
        scene.step()
    pos = block.get_particles_pos()
    scene.destroy()
    for i_b, velocity in enumerate(velocities):
        scene, block = _falling_block(0, show_viewer)
        block.set_velocity(velocity)
        for _ in range(10):
            scene.step()
        assert_allclose(pos[i_b], block.get_particles_pos(), tol=1e-11)
        scene.destroy()


@pytest.mark.required
# fp32 cannot resolve the Newton tolerance the comparison of two batches needs
@pytest.mark.precision("64")
def test_halt_on_nan(precision, show_viewer):
    scene, block = _falling_block(2, show_viewer)
    scene.step()
    pos_init = block.get_particles_pos()

    # A velocity that is not finite in the second environment fails its Newton solve alone: its particles keep their
    # state, and the first environment steps on as it does without it
    vel = block.get_particles_vel()
    vel[1, 0] = float("nan")
    block.set_particles_vel(vel)
    scene.step()
    assert_equal(scene.get_error_envs_mask(), (False, True))
    pos = block.get_particles_pos()
    assert_equal(pos[1], pos_init[1])
    assert pos[0].isfinite().all()
    reference_scene, reference_block = _falling_block(0, show_viewer)
    for _ in range(2):
        reference_scene.step()
    assert_allclose(pos[0], reference_block.get_particles_pos(), tol=1e-12)

    # The error raises at the beginning of the next step on CPU, and at worst of the one after it on GPU
    with pytest.raises(gs.GenesisException, match="Newton solve produced non-finite values"):
        for _ in range(1 if gs.backend == gs.cpu else 2):
            scene.step()

    # Resetting the scene clears the error, so the simulation proceeds
    scene.reset()
    scene.step()
    assert block.get_particles_pos().isfinite().all()
    assert not scene.get_error_envs_mask().any()


@pytest.mark.required
@pytest.mark.precision("64")
def test_halt_on_nan_freezes_participants(precision, show_viewer):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1e-2,
            gravity=(0.0, 0.0, 0.0),
        ),
        mpm_options=gs.options.MPMOptions(
            lower_bound=(-0.5, -0.5, -0.1),
            upper_bound=(0.5, 0.5, 0.5),
            grid_density=32,
        ),
        fem_options=gs.options.FEMOptions(
            use_implicit_solver=True,
        ),
        pbd_options=gs.options.PBDOptions(
            particle_size=1e-2,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.0, -1.0, 0.7),
            camera_lookat=(0.0, 0.0, 0.2),
            camera_fov=32,
        ),
        show_viewer=show_viewer,
    )
    block = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(-0.15, 0.0, 0.15),
            size=(0.1, 0.1, 0.1),
        ),
        material=gs.materials.MPM.Elastic(
            sampler="regular",
        ),
    )
    rigid_box = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.25, 0.0, 0.3),
            size=(0.05, 0.05, 0.05),
        ),
    )
    fem_box = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.0, 0.3, 0.3),
            size=(0.06, 0.06, 0.06),
        ),
        material=gs.materials.FEM.Elastic(),
    )
    cloth = scene.add_entity(
        morph=gs.morphs.Mesh(
            file="meshes/cloth.obj",
            scale=0.2,
            pos=(0.0, -0.3, 0.3),
        ),
        material=gs.materials.PBD.Cloth(),
    )
    scene.build(n_envs=2)

    # A velocity that is not finite in the second environment fails its Newton solve alone: every participant of the
    # solve keeps the state of that environment, and moves the other one at its velocity as it does without the failure
    states_by_run = []
    for is_nan_injected in (True, False):
        scene.reset()
        rigid_box.set_dofs_velocity((0.1, 0.0, 0.0), dofs_idx_local=slice(0, 3))
        fem_box.set_velocity((0.1, 0.0, 0.0))
        cloth.set_velocity((0.1, 0.0, 0.0))
        states = []
        for i_step in range(2):
            scene.step()
            states.append(
                tuple(
                    state.clone()
                    for state in (
                        rigid_box.get_pos(),
                        rigid_box.get_quat(),
                        rigid_box.get_vel(),
                        rigid_box.get_ang(),
                        fem_box.get_state().pos,
                        fem_box.get_state().vel,
                        cloth.get_particles_pos(),
                        cloth.get_particles_vel(),
                    )
                )
            )
            if is_nan_injected and i_step == 0:
                vel = block.get_particles_vel()
                vel[1, 0] = float("nan")
                block.set_particles_vel(vel)
        states_by_run.append(states)
        if is_nan_injected:
            assert_equal(scene.get_error_envs_mask(), (False, True))
    states_nan, states_clean = states_by_run
    for state_init, state_end in zip(*states_nan):
        assert_allclose(state_end[1], state_init[1], tol=1e-12)
    for state_nan, state_clean in zip(states_nan[1], states_clean[1]):
        assert_allclose(state_nan[0], state_clean[0], tol=1e-11)
    for i_pos in (0, 4, 6):
        assert (states_nan[1][i_pos][0, ..., 0] - states_nan[0][i_pos][0, ..., 0] > 5e-4).all()

    # The snapshots PBD keeps of its substep start are freed with the scene, as the other data of the solve
    ndarrays = {item.name: item.array for item in scene.sim.newton_solver.data if isinstance(item.array, qd.Ndarray)}
    assert sum(name.endswith((".pos_begin", ".vel_begin")) for name in ndarrays) == 2
    # Every coupling of the scene holds the rows of the sorted particles
    names = [item.name for item in scene.sim.newton_solver.data]
    assert all(any(name.startswith(prefix) for name in names) for prefix in ("rigid_rows.", "fem_rows.", "pbd_rows."))
    scene.destroy()
    assert all(array.arr is None for array in ndarrays.values())


@pytest.mark.required
@pytest.mark.parametrize(("has_plane", "has_rigid_dofs"), [(True, True), (True, False), (False, False)])
def test_destroy_frees_data(has_plane, has_rigid_dofs, show_viewer):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1e-2,
        ),
        mpm_options=gs.options.MPMOptions(
            lower_bound=(-0.3, -0.3, -0.1),
            upper_bound=(0.3, 0.3, 0.4),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.9, -0.9, 0.6),
            camera_lookat=(0.0, 0.0, 0.15),
            camera_fov=40,
        ),
        show_viewer=show_viewer,
    )
    if has_plane:
        scene.add_entity(
            morph=gs.morphs.Plane(),
        )
    mpm_box = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.0, 0.0, 0.05),
            size=(0.1, 0.1, 0.1),
        ),
        material=gs.materials.MPM.Elastic(
            sampler="regular",
        ),
    )
    if has_rigid_dofs:
        scene.add_entity(
            morph=gs.morphs.Box(
                pos=(0.0, 0.0, 0.15),
                size=(0.05, 0.05, 0.05),
            ),
        )
    scene.build(n_envs=2)
    arrays_built = [
        weakref.ref(item.array._unwrap())
        for item in scene.sim.newton_solver.data
        if item.name.startswith(("grid.", "newton.", "linear.")) and ".errors." not in item.name
    ]
    mpm_box.set_position(mpm_box.get_particles_pos() * 3.0)
    for _ in range(3):
        scene.step()

    # The data of the implicit step and of the joint Newton solve with Rigid, which kernels compiled for the scene keep
    # referencing, are freed with the scene, including the Rigid data read by the contact of a Rigid without dofs
    items = [*scene.sim.mpm_solver.data, *scene.sim.newton_solver.data]
    roots = {item.name.split(".")[0] for item in items}
    assert {"particles", "particles_info", "grid", "newton", "linear", "control"} <= roots
    # The rows of a coupling exist with the coupling alone, and the Rigid ones with any Rigid geom
    assert ("participants" in roots) == has_plane
    assert any(item.name.startswith("rigid_rows.") for item in items) == has_plane
    assert not any(item.name.startswith(("fem_rows.", "pbd_rows.")) for item in items)
    assert any(item.name.startswith("rigid_data.") for item in items) == has_plane
    assert any(".query." in item.name for item in items) == has_plane
    assert any(item.name.startswith("newton_info.") for item in items) == has_plane
    arrays = [item.array._unwrap() if isinstance(item.array, qd.Tensor) else item.array for item in items]
    ndarrays = [array for array in arrays if isinstance(array, qd.Ndarray)]
    assert ndarrays

    # Only the particle data, the material table and the error flags of the grid are fields in performance mode, which
    # a destroy cannot free
    names_field = [item.name for item, array in zip(items, arrays) if not isinstance(array, qd.Ndarray)]
    assert all(
        name.startswith(("particles", "mpm_materials.", "grid.errors.")) or name.endswith(".geoms_init_AABB")
        for name in names_field
    )

    # Spreading the particles grows the grid, the Newton state and the linear system, which replaces the structs built
    # with the scene. No array of the walk is one of theirs, and each is reached once
    arrays_replaced = [array() for array in arrays_built if array() is not None]
    assert not {id(array) for array in arrays_replaced} & {id(array) for array in arrays}
    assert all(array.arr is None for array in arrays_replaced)
    assert len({id(array) for array in arrays}) == len(arrays)

    scene.destroy()
    assert all(array.arr is None for array in ndarrays)


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.cpu_threads(1)
@pytest.mark.parametrize("backend", [gs.cpu])
@pytest.mark.parametrize("has_rigid_dofs", [True, False])
def test_grow_grid_and_matrix_matches_full_capacity(has_rigid_dofs, show_viewer):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1e-2,
        ),
        mpm_options=gs.options.MPMOptions(
            lower_bound=(-0.3, -0.3, -0.1),
            upper_bound=(0.3, 0.3, 0.4),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.9, -0.9, 0.6),
            camera_lookat=(0.0, 0.0, 0.15),
            camera_fov=40,
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(
        morph=gs.morphs.Plane(),
    )
    mpm_box = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.0, 0.0, 0.05),
            size=(0.1, 0.1, 0.1),
        ),
        material=gs.materials.MPM.Elastic(
            sampler="regular",
        ),
    )
    if has_rigid_dofs:
        scene.add_entity(
            morph=gs.morphs.Box(
                pos=(0.0, 0.0, 0.15),
                size=(0.05, 0.05, 0.05),
            ),
        )
    scene.build(n_envs=2)

    # The first run spreads the particles of a scene sized to their start, which grows its storage. The second run, from
    # the same state, finds the capacity the first one grew to, which no longer grows: no run can be compared with a
    # full capacity from a build of its own, since a capacity only grows
    arrays = {item.name: item.array for item in scene.sim.newton_solver.data}
    capacities = [(arrays["grid.block_coords"].shape[0], arrays["linear.matrix.triplet_row"].shape[0])]
    states = []
    for _ in range(2):
        scene.reset()
        mpm_box.set_position(mpm_box.get_particles_pos() * 3.0)
        for _ in range(3):
            scene.step()
        states.append((mpm_box.get_particles_pos().clone(), mpm_box.get_particles_vel().clone()))
        arrays = {item.name: item.array for item in scene.sim.newton_solver.data}
        capacities.append((arrays["grid.block_coords"].shape[0], arrays["linear.matrix.triplet_row"].shape[0]))

    assert capacities[1][0] > capacities[0][0] and capacities[1][1] > capacities[0][1]
    assert capacities[2] == capacities[1]
    assert_equal(states[0][0], states[1][0])
    assert_equal(states[0][1], states[1][1])


@pytest.mark.required
# The SAP coupler, one of the configurations refused, supports 64 bits precision alone
@pytest.mark.precision("64")
def test_reject_unsupported_configurations():
    liquid = gs.morphs.Box(pos=(0.0, 0.0, 0.2), size=(0.1, 0.1, 0.1))
    for options, requires_grad, force_field, match in (
        (
            {"mpm_options": gs.options.MPMOptions(lower_bound=(-0.5, -0.5, 0.0), upper_bound=(600.0, 0.5, 0.5))},
            False,
            None,
            "sparse grid",
        ),
        ({"coupler_options": gs.options.SAPCouplerOptions()}, False, None, "LegacyCouplerOptions"),
        ({"mpm_options": gs.options.MPMOptions(liquid_smoothing_velocity=100.0)}, False, None, "smoothing band"),
        ({}, True, gs.force_fields.Noise(), "Noise"),
    ):
        scene = gs.Scene(sim_options=gs.options.SimOptions(requires_grad=requires_grad), **options, show_viewer=False)
        scene.add_entity(morph=liquid, material=gs.materials.MPM.Liquid())
        if force_field is not None:
            scene.add_force_field(force_field)
        with pytest.raises(gs.GenesisException, match=match):
            scene.build()


@pytest.mark.slow  # ~300s from a cold cache, the Newton graph compiling in ~200s
@pytest.mark.required
@pytest.mark.precision("64")
def test_newton_graph_matches_split(show_viewer):
    # The Newton graph runs the stages of Rigid's dofs with those of MPM
    positions = []
    for newton_graph in ("split", "newton"):
        scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=1e-2,
            ),
            mpm_options=gs.options.MPMOptions(
                lower_bound=(-0.5, -0.5, -0.1),
                upper_bound=(0.5, 0.5, 0.6),
                grid_density=32,
            ),
            newton_options=gs.options.NewtonOptions(
                # Converged far below its default tolerance, the solve stops at the same iteration on every path
                velocity_tol=1e-10,
                graph=newton_graph,
            ),
            show_viewer=show_viewer,
        )
        scene.add_entity(
            morph=gs.morphs.Plane(),
        )
        block = scene.add_entity(
            morph=gs.morphs.Box(
                pos=(0.0, 0.0, 0.08),
                size=(0.12, 0.12, 0.12),
            ),
            material=gs.materials.MPM.Elastic(
                E=5e4,
                nu=0.3,
                sampler="regular",
            ),
        )
        scene.add_entity(
            morph=gs.morphs.Box(
                pos=(0.0, 0.0, 0.2),
                size=(0.06, 0.06, 0.06),
            ),
        )
        scene.build(n_envs=2)
        for _ in range(10):
            scene.step()
        positions.append(block.get_particles_pos())
    assert_allclose(*positions, tol=1e-13)

import numpy as np
import pytest
import torch
import trimesh

import genesis as gs

from ..utils.assertions import assert_allclose, assert_equal


@pytest.mark.required
def test_particle_constraints(show_viewer, tol):
    # A density that single precision cannot represent
    RHO = 1000.1

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
            rho=RHO,
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

    # The mass of the particles follows the density at the precision of the simulation
    assert_allclose(mpm_cube.get_mass(), mpm_cube.n_particles * scene.mpm_solver.particle_volume_real * RHO, tol=tol)

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
    dx = scene.mpm_solver.dx
    assert final_elastic_pos[..., 2].min() > -0.5 * dx
    assert final_liquid_pos[..., 2].min() > -0.5 * dx


@pytest.mark.required
# fp32 cannot resolve the Newton tolerance the comparison of two batches needs
@pytest.mark.precision("64")
def test_envs_solve_apart(precision, show_viewer):
    # Thrown at different speeds, so that their Newton solves run different numbers of iterations, the block of each
    # environment of a batch moves as it does alone. The number of environments is set at build, so the batch and each
    # environment alone are scenes of their own.
    VELOCITIES = ((0.0, 0.0, 0.0), (0.0, 0.0, -1.0), (1.0, 0.0, -0.5))
    for i_b in (None, *range(len(VELOCITIES))):
        scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=1e-2,
            ),
            mpm_options=gs.options.MPMOptions(
                lower_bound=(-0.5, -0.5, -0.1),
                upper_bound=(0.5, 0.5, 0.5),
                grid_density=32,
            ),
            # Converged far below the rounding the order of the sums of the reductions leaves, so that a solve does
            # not stop an iteration apart from the same solve in another batch
            newton_options=gs.options.NewtonOptions(
                velocity_tol=1e-10,
            ),
            show_viewer=show_viewer,
        )
        scene.add_entity(
            morph=gs.morphs.Plane(),
            vis_mode="collision",
        )
        # Off the faces of the grid cells, where the rounding of the position of a particle would pick its cell
        block = scene.add_entity(
            morph=gs.morphs.Box(
                pos=(0.003, 0.002, 0.1),
                size=(0.1, 0.1, 0.1),
            ),
            material=gs.materials.MPM.Elastic(
                sampler="regular",
            ),
        )
        scene.build(n_envs=len(VELOCITIES) if i_b is None else 0)
        block.set_velocity(VELOCITIES if i_b is None else VELOCITIES[i_b])
        for _ in range(10):
            scene.step()
        if i_b is None:
            pos_batch = block.get_particles_pos()
        else:
            assert_allclose(pos_batch[i_b], block.get_particles_pos(), tol=1e-11)
        scene.destroy()


@pytest.mark.required
def test_halt_on_nan(show_viewer):
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
        vis_mode="collision",
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
    # solve keeps the state of that environment, and moves the other one at its velocity
    rigid_box.set_dofs_velocity((0.1, 0.0, 0.0), dofs_idx_local=slice(0, 3))
    for entity in (fem_box, cloth, block):
        entity.set_velocity((0.1, 0.0, 0.0))
    scene.step()
    vel = block.get_particles_vel()
    vel[1, 0] = float("nan")
    block.set_particles_vel(vel)
    positions = (rigid_box.get_pos, lambda: fem_box.get_state().pos, cloth.get_particles_pos, block.get_particles_pos)
    positions_start = [position().clone() for position in positions]
    scene.step()
    for position, position_start in zip(positions, positions_start):
        assert_allclose(position()[1], position_start[1], tol=gs.EPS)
        assert (position()[0, ..., 0] - position_start[0, ..., 0] > 5e-4).all()
    assert_equal(scene.get_error_envs_mask(), (False, True))

    # The error raises at the beginning of the next step on CPU, and at worst of the one after it on GPU
    with pytest.raises(gs.GenesisException, match="Newton solve produced non-finite values"):
        for _ in range(1 if gs.backend == gs.cpu else 2):
            scene.step()

    # Resetting the scene clears the error, so the simulation proceeds
    scene.reset()
    scene.step()
    assert not scene.get_error_envs_mask().any()


@pytest.mark.required
# The SAP coupler, one of the configurations refused, supports 64 bits precision alone
@pytest.mark.precision("64")
def test_reject_unsupported_configurations():
    liquid = gs.morphs.Box(
        pos=(0.0, 0.0, 0.2),
        size=(0.1, 0.1, 0.1),
    )
    for options, requires_grad, force_field, match in (
        (
            {
                "mpm_options": gs.options.MPMOptions(
                    lower_bound=(-0.5, -0.5, 0.0),
                    upper_bound=(600.0, 0.5, 0.5),
                ),
            },
            False,
            None,
            "sparse grid",
        ),
        ({"coupler_options": gs.options.SAPCouplerOptions()}, False, None, "LegacyCouplerOptions"),
        (
            {
                "mpm_options": gs.options.MPMOptions(
                    liquid_smoothing_velocity=100.0,
                ),
            },
            False,
            None,
            "smoothing band",
        ),
        ({}, True, gs.force_fields.Noise(), "Noise"),
    ):
        scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                requires_grad=requires_grad,
            ),
            **options,
            show_viewer=False,
        )
        scene.add_entity(
            morph=liquid,
            material=gs.materials.MPM.Liquid(),
        )
        if force_field is not None:
            scene.add_force_field(force_field)
        with pytest.raises(gs.GenesisException, match=match):
            scene.build()


@pytest.mark.slow
@pytest.mark.required
@pytest.mark.precision("64")
def test_newton_graph_matches_split(show_viewer):
    # The Newton graph runs the stages of Rigid's dofs with those of MPM. The graph is set at build, so each mode is a
    # scene of its own.
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
            vis_mode="collision",
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
            vis_mode="collision",
        )
        scene.build(n_envs=2)
        for _ in range(10):
            scene.step()
        positions.append(block.get_particles_pos())
    assert_allclose(*positions, tol=1e-13)

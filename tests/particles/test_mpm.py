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
            dt=2e-3,
            substeps=20,
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
            dt=2e-3,
            substeps=10,
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
        ),
    )
    liquid = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.15, 0.0, 0.2),
            size=(0.08, 0.08, 0.08),
        ),
        material=gs.materials.MPM.Liquid(
            viscous=liquid_viscous,
        ),
    )
    scene.build(n_envs=2)

    # Aggregate SVD flag must match the variant for the dispatch path under test to actually run.
    assert scene.sim.mpm_solver.needs_svd == (variant == "svd")

    init_elastic_pos = elastic.get_particles_pos()
    init_liquid_pos = liquid.get_particles_pos()

    for _ in range(100):
        scene.step()

    final_elastic_pos = elastic.get_particles_pos()
    final_liquid_pos = liquid.get_particles_pos()

    # Gravity acted: particles moved down on average.
    assert init_elastic_pos[..., 2].mean() - final_elastic_pos[..., 2].mean() > 0.05
    assert init_liquid_pos[..., 2].mean() - final_liquid_pos[..., 2].mean() > 0.05

    # No ground penetration (plane is at z=0). Non-viscous liquid spreads thin enough that a sub-grid-cell penetration
    # is normal for the MPM coupling, so allow up to 2mm.
    assert final_elastic_pos[..., 2].min() > -1e-3
    assert final_liquid_pos[..., 2].min() > -2e-3


@pytest.mark.required
@pytest.mark.parametrize("precision", ["32"])
def test_halt_on_nan(show_viewer):
    scene = gs.Scene(
        mpm_options=gs.options.MPMOptions(
            lower_bound=(-0.5, -0.5, 0.0),
            upper_bound=(0.5, 0.5, 1.0),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.6, 0.6, 0.5),
            camera_lookat=(0.0, 0.0, 0.2),
        ),
        show_viewer=show_viewer,
    )
    block = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.0, 0.0, 0.3),
            size=(0.1, 0.1, 0.1),
        ),
        material=gs.materials.MPM.Elastic(
            E=1e9,
        ),
    )
    scene.build(n_envs=2)

    # The explicit integration of a stiff block diverges, the block of the second env first since it hits the floor
    # faster. The diverging env is flagged and frozen before its state stops being finite, while the other proceeds.
    block.set_velocity(((0.0, 0.0, 0.0), (0.0, 0.0, -5.0)))
    for _ in range(10):
        scene.step()
        assert block.get_state().pos.isfinite().all()
        if scene.get_error_envs_mask().any():
            break
    assert_equal(scene.get_error_envs_mask(), (False, True))

    # The error raises at the beginning of the next step on CPU, and at worst of the one after it on GPU
    with pytest.raises(gs.GenesisException, match="Invalid MPM particle velocities"):
        for _ in range(1 if gs.backend == gs.cpu else 2):
            scene.step()

    # Resetting the scene clears the error, so the simulation proceeds
    scene.reset()
    scene.step()
    assert block.get_state().pos.isfinite().all()
    assert not scene.get_error_envs_mask().any()

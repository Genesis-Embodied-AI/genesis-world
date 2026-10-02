import numpy as np
import pytest
import torch
import trimesh

import genesis as gs
from genesis.engine.states.solvers import MPMSolverState

from ..utils.assertions import assert_allclose, assert_equal


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
@pytest.mark.parametrize("model", ["corotation", "neohooken"])
def test_particles_stress(model, n_envs, show_viewer, tol):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.001,
            substeps=2,
            substeps_local=8,
            requires_grad=True,
        ),
        mpm_options=gs.options.MPMOptions(
            grid_density=32,
            lower_bound=(-0.3, -0.3, 0.0),
            upper_bound=(0.3, 0.3, 0.6),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.5, -0.6, 0.65),
            camera_lookat=(-0.04, 0.05, 0.3),
            camera_fov=35,
        ),
        show_viewer=show_viewer,
    )
    liquid = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(-0.15, 0.0, 0.3),
            size=(0.04, 0.04, 0.04),
        ),
        material=gs.materials.MPM.Liquid(
            sampler="regular",
        ),
    )
    scene.add_entity(
        morph=gs.morphs.Box(
            pos=(-0.15, 0.1, 0.3),
            size=(0.04, 0.04, 0.04),
        ),
        material=liquid.material,
    )
    elastic = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.1, 0.0, 0.3),
            size=(0.06, 0.06, 0.06),
        ),
        material=gs.materials.MPM.Elastic(
            E=1200.0,
            nu=0.25,
            sampler="regular",
            model=model,
        ),
        surface=gs.surfaces.Default(
            vis_mode="particle",
        ),
    )
    scene.build(n_envs=n_envs)
    snapshot_initial = scene.get_state()
    assert_equal(elastic.get_particles_stress(), 0.0)
    with pytest.raises(gs.GenesisException, match="MPM.Elastic"):
        liquid.get_particles_stress()

    snapshot = scene.get_state()
    state = next(state for state in snapshot if isinstance(state, MPMSolverState))
    particles_slice = slice(elastic.particle_start, elastic.particle_start + elastic.n_particles)
    rotation = torch.tensor(
        [
            [0.8, -0.6, 0.0],
            [0.6, 0.8, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=gs.tc_float,
        device=gs.device,
    )
    stretches = torch.tensor([1.2, 0.8, 1.1], dtype=gs.tc_float, device=gs.device)
    deformation = torch.diag(stretches)
    with torch.no_grad():
        for i_b in range(max(n_envs, 1)):
            state.F[i_b, particles_slice] = (1.0 - 0.1 * i_b) * deformation
            state.F[i_b, elastic.particle_start + elastic.n_particles // 2 : particles_slice.stop] = (
                (1.0 - 0.1 * i_b) * rotation @ deformation
            )
    scene.reset(state=snapshot)
    stress_before = elastic.get_particles_stress()
    assert stress_before.shape == ((n_envs,) if n_envs else ()) + (elastic.n_particles, 3, 3)
    stress_saved = stress_before.clone()
    stress_diagonal = stress_before[..., 0, :, :]
    stress_rotated = stress_before[..., -1, :, :]
    assert_allclose(stress_rotated / 1200.0, rotation @ stress_diagonal @ rotation.T / 1200.0, tol=tol)

    # Differentiate the strain energy independently of the solver's constitutive stress function
    for _ in range(2):
        deformation = elastic.get_state().F.detach().clone().requires_grad_(True)
        volume_ratio = torch.linalg.det(deformation)
        if model == "corotation":
            energy = 480.0 * (torch.linalg.svdvals(deformation) - 1.0).square().sum(dim=-1)
            energy += 240.0 * (volume_ratio - 1.0).square()
        else:
            energy = 240.0 * (deformation.square().sum(dim=(-2, -1)) - 3.0)
            energy += -480.0 * volume_ratio.log() + 240.0 * volume_ratio.log().square()
        piola = torch.autograd.grad(energy.sum(), deformation)[0]
        stress_reference = (piola @ deformation.transpose(-2, -1) / volume_ratio[..., None, None]).detach()
        stress = elastic.get_particles_stress()
        assert_allclose(stress / 1200.0, stress_reference / 1200.0, tol=tol)
        assert not stress.requires_grad
        if n_envs:
            assert_equal(elastic.get_particles_stress(envs_idx=[1, 0]), stress.flip(0))
            assert_equal(elastic.get_particles_stress(envs_idx=-1), stress[-1:])
            assert_equal(elastic.get_particles_stress(envs_idx=[]), stress[:0])
            assert elastic.get_particles_stress(envs_idx=[]).shape == (0, elastic.n_particles, 3, 3)
        scene.step()

    assert_equal(stress_before, stress_saved)
    scene.reset(state=snapshot_initial)
    assert_equal(elastic.get_particles_stress(), 0.0)
    state = next(state for state in snapshot_initial if isinstance(state, MPMSolverState))
    with torch.no_grad():
        state.F[:, particles_slice] = rotation
    scene.reset(state=snapshot_initial)
    assert_allclose(elastic.get_particles_stress() / 1200.0, 0.0, tol=tol)
    with torch.no_grad():
        state.F[:, elastic.particle_start, 0, 0] = -1.0
    scene.reset(state=snapshot_initial)
    with pytest.raises(gs.GenesisException, match="positive"):
        elastic.get_particles_stress()
    state.active[:, elastic.particle_start] = 0
    scene.reset(state=snapshot_initial)
    assert_allclose(elastic.get_particles_stress() / 1200.0, 0.0, tol=tol)


@pytest.mark.required
def test_particle_constraints(show_viewer, tol):
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
            rho=1000,
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
    mpm_mesh_set.process_input()
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

    mpm_diff = mpm_cube.get_particles_pos()[:, mask[0], 0].mean() - initial_mpm_x
    assert mpm_diff > pos_diff[0] * 0.3, f"MPM should follow rigid link. Got {mpm_diff:.3f}"


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

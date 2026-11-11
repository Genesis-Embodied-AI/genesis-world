# Tests for SPH simulation - initial density and pressure validation
#
# Background:
# SPH is sensitive to the initial particle distribution, as it directly determines the initial density and pressure fields.
# To ensure numerical stability, particles must be initialized using a regular sampler that enforces near-uniform spacing.
# Irregular samplers (e.g. pbs, random) introduce local density fluctuations at initialization, which lead to large
# spurious pressure forces and can cause the simulation to become unstable or diverge.

import pickle

import numpy as np
import pytest

import genesis as gs
from genesis.utils.misc import qd_to_numpy, tensor_to_array

from ..utils.assertions import assert_allclose, assert_equal


def create_sph_scene(show_viewer, particle_size=0.01, gravity=-9.81, mu=0.005):
    """Create a basic SPH scene with regular sampler."""
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=4e-3,
            substeps=10,
            gravity=(0.0, 0.0, gravity),
        ),
        sph_options=gs.options.SPHOptions(
            particle_size=particle_size,
            lower_bound=(-0.5, -0.5, 0.0),
            upper_bound=(0.5, 0.5, 1),
        ),
        show_viewer=show_viewer,
    )

    scene.add_entity(
        morph=gs.morphs.Plane(),
    )

    liquid = scene.add_entity(
        options=gs.entities.SPH(
            morph=gs.morphs.Box(
                pos=(0.0, 0.0, 0.65),
                size=(0.4, 0.4, 0.4),
            ),
            surface=gs.surfaces.Default(
                color=(0.4, 0.8, 1.0),
                vis_mode="particle",
            ),
            material=gs.materials.SPH.Liquid(
                mu=mu,
                sampler="regular",
            ),
        ),
    )

    scene.build()

    return scene, liquid


@pytest.fixture(scope="function")
def sph_scene(show_viewer):
    """Fixture to create a basic SPH scene with regular sampler."""
    return create_sph_scene(show_viewer)


def get_sph_density(scene, n_particles):
    """
    Helper function to compute and retrieve density from SPH solver.

    This triggers the density computation kernel and returns the computed values.

    Parameters
    ----------
    scene : gs.Scene
        The built scene containing SPH solver.
    n_particles : int
        Number of particles to retrieve.

    Returns
    -------
    densities : np.ndarray
        Array of density values for all particles.
    """
    from genesis.engine.solvers.sph_solver import kernel_compute_rho_and_dfsph_factor, kernel_reorder_particles

    sph_solver = scene.sph_solver

    # Reorder particles into spatial grid cells for efficient neighbor queries,
    # then compute density. This mirrors what happens in substep_pre_coupling.
    # We must reorder first because kernel_compute_rho_and_dfsph_factor operates on particles_reordered.
    kernel_reorder_particles(sph_solver.sph_state)
    kernel_compute_rho_and_dfsph_factor(sph_solver.sph_state, sph_solver.sph_info, sph_solver.sph_config)

    # Get density values from reordered particles (where kernel_compute_rho_and_dfsph_factor stores results)
    return qd_to_numpy(sph_solver.sph_state.particles_reordered.rho)[:n_particles, 0]


@pytest.mark.required
def test_initial_density_regular_sampler(show_viewer, tol):
    # A regular sampler places particles at uniform spacing: interior particles see a consistent near-rest
    # density while boundary particles sit lower (fewer neighbors inside the kernel support), so the mean lands
    # below the rest density with a bounded spread.
    scene, _ = create_sph_scene(show_viewer)
    sph_solver = scene.sph_solver
    n_particles = sph_solver.n_particles
    rho0 = qd_to_numpy(sph_solver.sph_info.particles.rho)[0]

    # Get initial density values
    densities = get_sph_density(scene, n_particles)

    # Test 1: Mean density should be within reasonable bounds
    # For regular sampler, mean is typically 70-90% of rho0 due to boundary effects
    density_mean = densities.mean()
    assert 0.5 * rho0 < density_mean < rho0, (
        f"Mean initial density {density_mean:.2f} is outside expected range "
        f"[{0.5 * rho0:.2f}, {rho0:.2f}] for rest density {rho0}"
    )

    # Test 2: Density standard deviation should be bounded
    # Regular sampler produces relatively uniform density distribution
    density_std = densities.std()
    assert density_std < 0.1 * rho0, (
        f"Density standard deviation {density_std:.2f} is too high "
        f"(expected < {0.1 * rho0:.2f} for rest density {rho0})"
    )

    # Test 3: No extreme density values
    assert densities.min() > 0.3 * rho0, (
        f"Minimum density {densities.min():.2f} is too low (expected > {0.3 * rho0:.2f})"
    )
    assert densities.max() < 1.5 * rho0, (
        f"Maximum density {densities.max():.2f} is too high (expected < {1.5 * rho0:.2f})"
    )


@pytest.mark.required
def test_initial_pressure_regular_sampler(show_viewer, tol):
    # Regular sampling yields densities at or below the rest density, so the density solve finds no compression to
    # correct and the initial pressure is zero. A nonzero initial pressure would inject spurious forces that
    # destabilize the simulation. The liquid is inviscid, which leaves the stable timestep unbounded.
    scene, _ = create_sph_scene(show_viewer, mu=0.0)
    sph_solver = scene.sph_solver
    rho0 = qd_to_numpy(sph_solver.sph_info.particles.rho)[0]

    densities = get_sph_density(scene, sph_solver.n_particles)
    assert (densities <= rho0).all(), (
        f"Maximum initial density {densities.max():.6f} exceeds the rest density {rho0}, which shouldn't happen with "
        f"regular sampler."
    )


@pytest.mark.required
def test_simulation_stability_regular_sampler(show_viewer, tol):
    GRAVITY = -9.81

    scene, liquid = create_sph_scene(show_viewer, gravity=GRAVITY)
    sph_solver = scene.sph_solver
    n_particles = sph_solver.n_particles

    # Get initial positions for comparison
    initial_pos = liquid.get_particles_pos().clone()

    # The upper quarter of the liquid on one side slides through the rest, so that the viscosity acts between particles
    # of different densities along the free surface.
    particles_pos = tensor_to_array(initial_pos)
    is_sliding = (particles_pos[:, 0] < particles_pos[:, 0].mean()) & (particles_pos[:, 2] > particles_pos[:, 2].mean())
    particles_vel = np.zeros((n_particles, 3))
    particles_vel[is_sliding, 0] = 1.0
    liquid.set_particles_vel(particles_vel)

    # Run simulation for a few steps
    num_steps = 5
    max_velocities = []
    for step in range(num_steps):
        scene.step()
        pos = liquid.get_particles_pos()
        vel = liquid.get_particles_vel()
        assert not pos.isnan().any(), f"NaN values detected in positions at step {step}"
        assert not pos.isinf().any(), f"Inf values detected in positions at step {step}"
        assert not vel.isnan().any(), f"NaN values detected in velocities at step {step}"
        assert not vel.isinf().any(), f"Inf values detected in velocities at step {step}"

        # Track maximum velocity
        max_vel = vel.norm(dim=-1).max()
        max_velocities.append(max_vel)

    # Maximum velocity should be bounded (not exploding)
    # For stable simulation with gravity, velocities should be reasonable
    max_velocity_overall = max(max_velocities)
    assert max_velocity_overall < 10.0, (
        f"Maximum velocity {max_velocity_overall:.2f} exceeds reasonable bounds. This may indicate instability."
    )

    # Get final positions
    final_pos = liquid.get_particles_pos()

    # Check particles haven't moved too far (not exploding)
    displacement = (final_pos - initial_pos).norm(dim=-1).max()
    assert displacement < 1.0, (
        f"Maximum displacement {displacement:.4f} is too large after {num_steps} steps. This may indicate instability."
    )

    # The liquid is still far above the floor, so it is in free fall: the particle interactions are internal and
    # cancel out in the momentum balance, leaving the mean velocity to keep its initial value and to gain gravity times
    # the elapsed time. The tolerance absorbs the momentum drift of the surface tension, which pushes each particle
    # along its own normal to the free surface.
    assert_allclose(
        liquid.get_particles_vel().mean(dim=0),
        particles_vel.mean(axis=0) + (0.0, 0.0, GRAVITY * num_steps * scene.dt),
        tol=2e-6,
    )


@pytest.mark.required
@pytest.mark.parametrize("particle_size", [0.01, 0.02])
def test_density_consistency_different_particle_sizes(show_viewer, particle_size, tol):
    # The SPH kernel support scales with the particle size, so the density distribution normalized by the rest
    # density must be independent of the particle size.
    scene, _ = create_sph_scene(
        show_viewer,
        particle_size=particle_size,
    )

    sph_solver = scene.sph_solver
    n_particles = sph_solver.n_particles
    rho0 = qd_to_numpy(sph_solver.sph_info.particles.rho)[0]

    # Get density values
    densities = get_sph_density(scene, n_particles)

    # Normalized metrics should be similar regardless of particle size
    normalized_mean = densities.mean() / rho0
    normalized_std = densities.std() / rho0

    # Mean density should be in similar range (50-100% of rho0)
    assert 0.5 < normalized_mean < 1.0, (
        f"Normalized mean density {normalized_mean:.3f} is outside expected range [0.5, 1.0] "
        f"for particle_size={particle_size}"
    )

    # Standard deviation should be bounded
    assert normalized_std < 0.1, (
        f"Normalized density std {normalized_std:.3f} is too high for particle_size={particle_size}"
    )

    # No particle is compressed, so the initial pressure is still zero
    assert (densities <= rho0).all(), (
        f"Maximum initial density {densities.max():.6f} exceeds the rest density for particle_size={particle_size}"
    )


@pytest.mark.required
def test_resume_from_pickle(show_viewer):
    LOWER_BOUND, UPPER_BOUND = (-0.5, -0.5, 0.0), (0.5, 0.5, 1.0)
    LIQUID_SIZE = (0.2, 0.2, 0.2)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=4e-3,
            substeps=10,
        ),
        sph_options=gs.options.SPHOptions(
            particle_size=0.02,
            lower_bound=LOWER_BOUND,
            upper_bound=UPPER_BOUND,
        ),
        pbd_options=gs.options.PBDOptions(
            particle_size=0.02,
            lower_bound=LOWER_BOUND,
            upper_bound=UPPER_BOUND,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.5, 0.0, 0.8),
            camera_lookat=(0.0, 0.0, 0.2),
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(
        morph=gs.morphs.Plane(),
    )
    sph_liquid = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(-0.25, 0.0, 0.2),
            size=LIQUID_SIZE,
        ),
        material=gs.materials.SPH.Liquid(
            sampler="regular",
        ),
    )
    pbd_liquid = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.25, 0.0, 0.2),
            size=LIQUID_SIZE,
        ),
        material=gs.materials.PBD.Liquid(
            sampler="regular",
        ),
    )
    scene.build(n_envs=2)

    # The liquids are in motion when the state is saved, and the pressure the density solve of DFSPH carries over from
    # one substep to the next is part of the state a restore resumes from
    for _ in range(10):
        scene.step()
    # Pickling the scene rejects particle entities, which have no description yet, so the simulator state goes alone
    checkpoint_bytes = pickle.dumps(scene.sim.__getstate__())
    for _ in range(10):
        scene.step()
    entities_state = [(entity.get_particles_pos(), entity.get_particles_vel()) for entity in (sph_liquid, pbd_liquid)]
    scene.sim.__setstate__(pickle.loads(checkpoint_bytes))
    for _ in range(10):
        scene.step()
    for entity, (pos, vel) in zip((sph_liquid, pbd_liquid), entities_state):
        assert_allclose(entity.get_particles_pos(), pos, tol=gs.EPS)
        assert_allclose(entity.get_particles_vel(), vel, tol=gs.EPS)


@pytest.mark.required
def test_dfsph_resting_liquid_stability(show_viewer):
    LIQUID_HEIGHT = 0.1
    GRAVITY = -9.81

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=2e-3,
            substeps=10,
            gravity=(0.0, 0.0, GRAVITY),
        ),
        sph_options=gs.options.SPHOptions(
            lower_bound=(-0.1, -0.1, 0.0),
            upper_bound=(0.1, 0.1, 0.5),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.5, -0.5, 0.3),
            camera_lookat=(0.0, 0.0, 0.05),
            camera_fov=40,
        ),
        show_viewer=show_viewer,
    )
    liquid = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.0, 0.0, 0.5 * LIQUID_HEIGHT),
            size=(0.2, 0.2, LIQUID_HEIGHT),
        ),
        material=gs.materials.SPH.Liquid(),
    )
    scene.build(n_envs=2)

    # The liquid starts at rest at its rest density, so its particles only move as it settles under its own weight. The
    # pressure the density solve carries over from one substep to the next must not push the particles apart.
    for _ in range(50):
        scene.step()
        assert (liquid.get_particles_vel().norm(dim=-1) < 5e-2).all()


@pytest.mark.required
# The wide column runs on the GPU alone, as it takes minutes on the CPU
@pytest.mark.parametrize("liquid_width, backend", [(0.3, gs.gpu), (0.02, gs.cpu), (0.02, gs.gpu)])
def test_pressure_carries_column_weight(show_viewer, liquid_width):
    GRAVITY = 9.81
    PARTICLE_SIZE = 0.02
    LIQUID_SIZE = (0.3, liquid_width, 0.5)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1e-2,
            substeps=10,
            gravity=(0.0, 0.0, -GRAVITY),
        ),
        sph_options=gs.options.SPHOptions(
            particle_size=PARTICLE_SIZE,
            lower_bound=(-0.5 * LIQUID_SIZE[0], -0.5 * LIQUID_SIZE[1], 0.0),
            upper_bound=(0.5 * LIQUID_SIZE[0], 0.5 * LIQUID_SIZE[1], 1.0),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.9, 0.0, 0.5),
            camera_lookat=(0.0, 0.0, 0.25),
            camera_fov=40,
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(
        morph=gs.morphs.Plane(),
    )
    liquid = scene.add_entity(
        options=gs.entities.SPH(
            morph=gs.morphs.Box(
                pos=(0.0, 0.0, 0.5 * LIQUID_SIZE[2]),
                size=LIQUID_SIZE,
            ),
            material=gs.materials.SPH.Liquid(
                sampler="regular",
            ),
        ),
    )
    scene.build(n_envs=2)

    # The pressure at the bottom of a settled column carries the weight of the liquid above it. The column keeps
    # sloshing, so the pressure only stays within a loose bound at every step once the column has settled, while its
    # mean over whole periods of the sloshing matches the hydrostatic pressure. The periods are delimited by the steps
    # where the pressure rises through its mean, which cancels the acoustic mode of the column whatever its frequency.
    sph_solver = scene.sph_solver
    particle_weight = qd_to_numpy(sph_solver.sph_info.particles.mass)[0] * GRAVITY
    footprint_area = LIQUID_SIZE[0] * LIQUID_SIZE[1]
    weight_per_area = sph_solver.n_particles * particle_weight / footprint_area
    # Under gravity tilted away from the vertical, the free surface settles normal to gravity, and the walls carry the
    # hydrostatic pressure of the depth along gravity. The second environment starts at this equilibrium, each column of
    # the lattice of the liquid filled up to the free surface over it, the columns gaining what others lose. The walls
    # along y of a slab thinner than the support radius hold all its particles, which the ratio would then average over.
    is_tilt_checked = liquid_width > 2 * PARTICLE_SIZE
    tilt = np.radians(20.0)
    gravity_tilted = GRAVITY * np.array([np.sin(tilt), 0.0, -np.cos(tilt)])
    if is_tilt_checked:
        sph_solver.set_gravity(gravity_tilted, envs_idx=1)
        particles_pos = tensor_to_array(liquid.get_particles_pos())[1]
        columns_pos = np.unique(particles_pos[:, :2], axis=0)
        layers_height = particles_pos[:, 2].min() + np.arange(0.0, 2.0 * LIQUID_SIZE[2], PARTICLE_SIZE)
        is_filled = layers_height < LIQUID_SIZE[2] + columns_pos[:, :1] * np.tan(tilt)
        columns_layers_pos = np.concatenate(
            [
                np.broadcast_to(columns_pos[:, None], (*is_filled.shape, 2)),
                np.broadcast_to(layers_height[:, None], (*is_filled.shape, 1)),
            ],
            axis=-1,
        )
        liquid.set_particles_pos(columns_layers_pos[is_filled][None], envs_idx=[1])
    # Every other column of the lattice starts a twentieth of a particle size higher, which seeds the packing of
    # staggered columns that the liquid settles into, so that the checks see a settled liquid.
    particles_pos = tensor_to_array(liquid.get_particles_pos())
    is_column_odd = np.round(particles_pos[..., :2] / PARTICLE_SIZE).sum(axis=-1) % 2 > 0.5
    particles_pos[..., 2] += 0.05 * PARTICLE_SIZE * is_column_odd
    liquid.set_particles_pos(particles_pos)
    rho_rest = qd_to_numpy(sph_solver.sph_info.particles.rho)[0]
    p_bottoms, p_middles, p_tops, p_hydrostatics, tilted_ratios = [], [], [], [], []
    # A late reordering of the packing drops a particle at most from its site on the cubic lattice into a hollow of the
    # close-packed layer below, by the difference of their layer spacings at the same density, in free fall.
    speed_max = np.sqrt(2.0 * GRAVITY * (1.0 - 4.0 ** (1.0 / 3.0) / np.sqrt(3.0)) * PARTICLE_SIZE)
    n_steps = 150
    for i_step in range(n_steps):
        scene.step()
        if is_tilt_checked and i_step >= 100:
            pressures = qd_to_numpy(sph_solver.sph_state.particles_reordered.p, transpose=True)[1]
            positions = qd_to_numpy(sph_solver.sph_state.particles_reordered.pos, transpose=True)[1]
            heights = positions @ (-gravity_tilted / GRAVITY)
            depths = heights.max() + 0.5 * PARTICLE_SIZE - heights
            is_wall = (positions[:, 2] < PARTICLE_SIZE) | (
                np.abs(positions[:, :2]) > 0.5 * np.array(LIQUID_SIZE[:2]) - PARTICLE_SIZE
            ).any(axis=-1)
            is_deep_wall = is_wall & (depths > 4 * PARTICLE_SIZE)
            tilted_ratios.append(pressures[is_deep_wall].mean() / (rho_rest * GRAVITY * depths[is_deep_wall]).mean())
        if i_step >= 100:
            assert (liquid.get_particles_vel()[0].norm(dim=-1) < speed_max).all()
            pressures = qd_to_numpy(sph_solver.sph_state.particles_reordered.p, transpose=True)[0]
            heights = qd_to_numpy(sph_solver.sph_state.particles_reordered.pos, transpose=True)[0, :, 2]
            p_bottom = pressures[heights < 0.05].mean()
            if i_step >= 130:
                assert_allclose(p_bottom, weight_per_area, rtol=0.5)
            p_bottoms.append(p_bottom)
            p_middles.append(pressures[(heights >= 0.2) & (heights < 0.25)].mean())
            p_tops.append(pressures[heights >= 0.3].mean())
            # The weight of the liquid above each particle, half its own layer included, spread over the footprint
            ranks = np.argsort(np.argsort(-heights))
            p_hydrostatics.append(((ranks[heights < 0.05] + 0.5) * particle_weight / footprint_area).mean())
    p_bottoms, p_middles, p_tops, p_hydrostatics = map(np.array, (p_bottoms, p_middles, p_tops, p_hydrostatics))
    (rising_idx,) = np.nonzero((p_bottoms[:-1] < p_bottoms.mean()) & (p_bottoms[1:] >= p_bottoms.mean()))
    window = slice(rising_idx[0] + 1, rising_idx[-1] + 1)

    # The kernel moments of the staggered packing set the pressure gradient, and the layers next to the floor shift the
    # bottom pressure.
    assert_allclose(p_bottoms[window].mean(), p_hydrostatics[window].mean(), rtol=0.03)

    if is_tilt_checked:
        assert_allclose(np.mean(tilted_ratios), 1.0, atol=1e-2)

    # The staggered packing sums the kernel lower than the lattice, so the liquid compresses until it reaches its rest
    # density again, its top layer sinking at most half a layer, while the seeded columns start higher.
    settlement_max = 0.5 * 4.73e-3 * LIQUID_SIZE[2] + 0.5 * PARTICLE_SIZE / round(LIQUID_SIZE[2] / PARTICLE_SIZE)
    particles_height_mean = liquid.get_particles_pos()[..., 2].mean(dim=-1)
    assert (particles_height_mean > 0.5 * LIQUID_SIZE[2] - settlement_max).all()
    assert (particles_height_mean < 0.5 * LIQUID_SIZE[2] + 0.025 * PARTICLE_SIZE).all()

    # The pressure must decrease with height, following the hydrostatic gradient. The factors are lower bounds
    # on the ratio of the hydrostatic depths at the band centers, with slack for pressure fluctuations.
    assert p_bottoms[window].mean() > 1.5 * p_middles[window].mean()
    assert p_middles[window].mean() > 2.0 * p_tops[window].mean()

    # Stepping from a restored state must reproduce the continuous run bit-identically, whatever the order in which
    # parallel threads sort the particles into the cells of the neighbour search.
    state = scene.get_state()
    for _ in range(5):
        scene.step()
    pos = liquid.get_particles_pos().clone()
    scene.reset(state)
    for _ in range(5):
        scene.step()
    assert_equal(liquid.get_particles_pos(), pos)

    # The trajectory of an environment must not depend on what the other environments simulate alongside it.
    scene.reset(state)
    liquid.set_particles_vel((0.0, 0.0, 0.5), envs_idx=[1])
    for _ in range(5):
        scene.step()
    assert_equal(liquid.get_particles_pos(envs_idx=[0]), pos[[0]])

    # The walls under a shallow pool carry the weight of that pool alone, so it moves like a pool far from any other
    # liquid, whatever lies in the columns that the hash grid wraps onto its own, here a deep pool one period of the
    # default hash grid away (150 cells of twice the particle size).
    scene_pools = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1e-2,
            substeps=10,
            gravity=(0.0, 0.0, -GRAVITY),
        ),
        sph_options=gs.options.SPHOptions(
            particle_size=PARTICLE_SIZE,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3.0, -6.0, 2.0),
            camera_lookat=(3.0, 0.0, 0.0),
            camera_fov=40,
        ),
        show_viewer=show_viewer,
    )
    scene_pools.add_entity(
        morph=gs.morphs.Box(
            pos=(0.0, 0.0, 0.15),
            size=(0.3, 0.3, 0.3),
        ),
        material=gs.materials.SPH.Liquid(
            sampler="regular",
        ),
    )
    wrapped_pool = scene_pools.add_entity(
        morph=gs.morphs.Box(
            pos=(6.0, 0.0, 0.03),
            size=(0.3, 0.3, 0.06),
        ),
        material=gs.materials.SPH.Liquid(
            sampler="regular",
        ),
    )
    isolated_pool = scene_pools.add_entity(
        morph=gs.morphs.Box(
            pos=(0.0, 1.0, 0.03),
            size=(0.3, 0.3, 0.06),
        ),
        material=gs.materials.SPH.Liquid(
            sampler="regular",
        ),
    )
    scene_pools.build(n_envs=2)
    for _ in range(3):
        scene_pools.step()
        assert_allclose(wrapped_pool.get_particles_vel(), isolated_pool.get_particles_vel(), atol=1e-3)

# Tests for SPH simulation - initial density and pressure validation
#
# Background:
# SPH is sensitive to the initial particle distribution, as it directly determines the initial density and pressure fields.
# To ensure numerical stability, particles must be initialized using a regular sampler that enforces near-uniform spacing.
# Irregular samplers (e.g. pbs, random) introduce local density fluctuations at initialization, which lead to large
# spurious pressure forces and can cause the simulation to become unstable or diverge.

import pytest
import numpy as np

import genesis as gs
from genesis.utils.misc import qd_to_numpy


def create_sph_scene(show_viewer, particle_size=0.01, pressure_solver="WCSPH"):
    """Create a basic SPH scene with regular sampler."""
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=4e-3,
            substeps=10,
        ),
        sph_options=gs.options.SPHOptions(
            particle_size=particle_size,
            pressure_solver=pressure_solver,
            lower_bound=(-0.5, -0.5, 0.0),
            upper_bound=(0.5, 0.5, 1),
        ),
        show_viewer=show_viewer,
    )

    scene.add_entity(
        morph=gs.morphs.Plane(),
    )

    liquid = scene.add_entity(
        material=gs.materials.SPH.Liquid(
            sampler="regular",
        ),
        morph=gs.morphs.Box(
            pos=(0.0, 0.0, 0.65),
            size=(0.4, 0.4, 0.4),
        ),
        surface=gs.surfaces.Default(
            color=(0.4, 0.8, 1.0),
            vis_mode="particle",
        ),
    )

    scene.build()

    return scene, liquid


@pytest.fixture(scope="function")
def sph_scene(show_viewer):
    """Fixture to create a basic SPH scene with regular sampler."""
    return create_sph_scene(show_viewer)


def get_sph_density_pressure(scene, n_particles):
    """
    Helper function to compute and retrieve density and pressure from SPH solver.

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
    pressures : np.ndarray
        Array of pressure values for all particles.
    """
    sph_solver = scene.sph_solver

    # Reorder particles into spatial grid cells for efficient neighbor queries,
    # then compute density. This mirrors what happens in substep_pre_coupling.
    # We must reorder first because _kernel_compute_rho operates on particles_reordered.
    sph_solver._kernel_reorder_particles(0)
    sph_solver._kernel_compute_rho(0)

    # Get density values from reordered particles (where _kernel_compute_rho stores results)
    densities = sph_solver.particles_reordered.rho.to_numpy()[:n_particles, 0]

    # Get material properties for pressure computation
    rho0 = sph_solver.particles_info[0].rho
    stiffness = sph_solver.particles_info[0].stiffness
    exponent = sph_solver.particles_info[0].exponent

    # Compute pressure using the same formula as the solver:
    # p = stiffness * ((max(rho, rho0) / rho0)^exponent - 1)
    densities_clamped = np.maximum(densities, rho0)
    pressures = stiffness * (np.power(densities_clamped / rho0, exponent) - 1.0)

    return densities, pressures


@pytest.mark.required
@pytest.mark.parametrize("pressure_solver", ["WCSPH", "DFSPH"])
def test_initial_density_regular_sampler(show_viewer, pressure_solver, tol):
    # A regular sampler places particles at uniform spacing: interior particles see a consistent near-rest
    # density while boundary particles sit lower (fewer neighbors inside the kernel support), so the mean lands
    # below the rest density with a bounded spread.
    scene, _ = create_sph_scene(show_viewer, pressure_solver=pressure_solver)
    sph_solver = scene.sph_solver
    n_particles = sph_solver.n_particles
    rho0 = sph_solver.particles_info[0].rho

    # Get initial density values
    densities, _ = get_sph_density_pressure(scene, n_particles)

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
@pytest.mark.parametrize("pressure_solver", ["WCSPH", "DFSPH"])
def test_initial_pressure_regular_sampler(show_viewer, pressure_solver, tol):
    # Regular sampling yields densities at or below the rest density, and the equation of state clamps density
    # to the rest density, so the initial pressure must be exactly zero. A nonzero initial pressure would inject
    # spurious forces that destabilize the simulation.
    scene, _ = create_sph_scene(show_viewer, pressure_solver=pressure_solver)
    sph_solver = scene.sph_solver
    n_particles = sph_solver.n_particles
    rho0 = sph_solver.particles_info[0].rho
    stiffness = sph_solver.particles_info[0].stiffness

    # Get initial pressure values
    densities, pressures = get_sph_density_pressure(scene, n_particles)

    # Test: For regular sampler, initial density should be <= rho0
    # which means pressure should be 0 (clamped)
    # All particles should have near-zero pressure initially
    assert np.allclose(pressures, 0.0, atol=1e-6), (
        f"Initial pressure is not zero. Mean: {pressures.mean():.6f}, "
        f"Max: {pressures.max():.6f}. "
        f"This indicates density > rho0 for some particles, which shouldn't happen "
        f"with regular sampler."
    )

    # Also verify that density is indeed <= rho0 for most particles
    fraction_above_rho0 = (densities > rho0).sum() / n_particles
    assert fraction_above_rho0 < 0.01, (
        f"{fraction_above_rho0 * 100:.1f}% of particles have density > rho0. "
        f"Regular sampler should produce density <= rho0 for almost all particles."
    )


@pytest.mark.required
@pytest.mark.parametrize("pressure_solver", ["WCSPH", "DFSPH"])
def test_simulation_stability_regular_sampler(show_viewer, pressure_solver, tol):
    scene, liquid = create_sph_scene(show_viewer, pressure_solver=pressure_solver)
    sph_solver = scene.sph_solver
    n_particles = sph_solver.n_particles

    # Get initial positions for comparison
    initial_pos = liquid.get_particles_pos().clone()

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


@pytest.mark.required
@pytest.mark.parametrize("pressure_solver", ["WCSPH", "DFSPH"])
@pytest.mark.parametrize("particle_size", [0.01, 0.02])
def test_density_consistency_different_particle_sizes(show_viewer, pressure_solver, particle_size, tol):
    # The SPH kernel support scales with the particle size, so the density distribution normalized by the rest
    # density must be independent of the particle size.
    scene, _ = create_sph_scene(
        show_viewer,
        particle_size=particle_size,
        pressure_solver=pressure_solver,
    )

    sph_solver = scene.sph_solver
    n_particles = sph_solver.n_particles
    rho0 = sph_solver.particles_info[0].rho

    # Get density values
    densities, pressures = get_sph_density_pressure(scene, n_particles)

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

    # Pressure should still be zero
    assert np.allclose(pressures, 0.0, atol=1e-6), (
        f"Initial pressure is not zero for particle_size={particle_size}. Mean: {pressures.mean():.6f}"
    )


# =============================================================================
# DFSPH (Divergence-Free SPH) Pressure Solver Tests
# =============================================================================


# TODO: Add more tests for validating the underlying physical behavior of DFSPH.
@pytest.mark.required
def test_dfsph_simulation_builds_and_runs(show_viewer):
    scene, liquid = create_sph_scene(show_viewer, pressure_solver="DFSPH")

    assert scene.sph_solver is not None
    assert scene.sph_solver.n_particles > 0

    # Run a few simulation steps
    for _ in range(3):
        scene.step()

    pos = liquid.get_particles_pos()
    assert pos.shape[0] == scene.sph_solver.n_particles


@pytest.mark.required
@pytest.mark.parametrize("pressure_solver", ["WCSPH", "DFSPH"])
def test_pressure_carries_column_weight(show_viewer, pressure_solver):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1e-2,
            substeps=10,
            gravity=(0.0, 0.0, -9.81),
        ),
        sph_options=gs.options.SPHOptions(
            particle_size=0.02,
            pressure_solver=pressure_solver,
            lower_bound=(-0.15, -0.15, 0.0),
            upper_bound=(0.15, 0.15, 1.0),
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
    scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.0, 0.0, 0.25),
            size=(0.3, 0.3, 0.5),
        ),
        material=gs.materials.SPH.Liquid(
            sampler="regular",
        ),
    )
    scene.build()

    for _ in range(100):
        scene.step()

    sph_solver = scene.sph_solver
    pressures = qd_to_numpy(sph_solver.particles_reordered.p, transpose=True)[0]
    heights = qd_to_numpy(sph_solver.particles_reordered.pos, transpose=True)[0, :, 2]

    # The pressure at the bottom of a settled column carries the weight of the whole column, which is exactly
    # the particle count times the particle weight spread over the column footprint. The tolerance absorbs the
    # residual sloshing of the column and two boundary effects that pull in opposite directions: the wall
    # pressure force from the floor relieves part of the compression that the WCSPH equation of state reads,
    # while the positive-part clamp of the DFSPH density solve rectifies velocity noise into extra pressure.
    weight_per_area = sph_solver.n_particles * sph_solver.particles_info[0].mass * 9.81 / (0.3 * 0.3)
    p_bottom = pressures[heights < 0.05].mean()
    assert 0.4 * weight_per_area < p_bottom < 1.5 * weight_per_area

    # The pressure must decrease with height, following the hydrostatic gradient. The factors are lower bounds
    # on the ratio of the hydrostatic depths at the band centers, with slack for pressure fluctuations.
    p_middle = pressures[(heights >= 0.2) & (heights < 0.25)].mean()
    p_top = pressures[heights >= 0.3].mean()
    assert p_bottom > 1.5 * p_middle
    assert p_middle > 2.0 * p_top

import numpy as np
import pytest

import genesis as gs
import genesis.utils.particle as pu

# A capacity small enough for the recycling corner cases, and emission sizes that vary from call to call. The cursor
# reaches 8 after three calls, so the fourth batch of 3 particles must recycle to 0 instead of addressing index 10 of
# a 10-particle entity.
MAX_PARTICLES = 10
EMIT_SIZES = (3, 3, 2, 3)
EXPECTED_INDICES = ((0, 1, 2), (3, 4, 5), (6, 7), (0, 1, 2))


@pytest.mark.required
@pytest.mark.parametrize("emit_omni", [False, True])
def test_emit_recycles_before_capacity_overflow(monkeypatch, emit_omni):
    # Deterministic droplet cardinalities, as the real ones depend on the droplet geometry, the sampler and the
    # particle size.
    emit_sizes = iter(EMIT_SIZES)

    def sample_droplet(*args, **kwargs):
        n_particles = next(emit_sizes)
        positions = np.full((n_particles, 3), 0.5, dtype=gs.np_float)
        positions[:, 0] += 0.01 * np.arange(n_particles, dtype=gs.np_float)
        return positions

    if emit_omni:
        monkeypatch.setattr(pu, "shell_to_particles", sample_droplet)
    else:
        monkeypatch.setattr(pu, "cylinder_to_particles", sample_droplet)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=1e-2, substeps=1, gravity=(0.0, 0.0, 0.0)),
        pbd_options=gs.options.PBDOptions(particle_size=0.05),
        show_viewer=False,
    )
    emitter = scene.add_emitter(material=gs.materials.PBD.Particle(), max_particles=MAX_PARTICLES)
    scene.build()

    # Record the indices and the values of every write, as out-of-range indices are silently dropped or corrupt
    # neighbouring particles in the solver kernels instead of raising. The public getters tell whether every write
    # actually landed in the solver state, which the recorded arguments alone cannot.
    written = []
    for method_name, write_particles in (
        ("set_particles_pos", emitter.entity.set_particles_pos),
        ("set_particles_vel", emitter.entity.set_particles_vel),
        ("set_particles_active", emitter.entity.set_particles_active),
    ):

        def record_and_write(values, particles_idx_local=None, write_particles=write_particles, written=written):
            written.append((particles_idx_local, values))
            return write_particles(values, particles_idx_local)

        monkeypatch.setattr(emitter.entity, method_name, record_and_write)

    for _ in EMIT_SIZES:
        if emit_omni:
            emitter.emit_omni(source_radius=0.1)
        else:
            emitter.emit(droplet_shape="circle", droplet_size=0.25, droplet_length=0.1)

    assert emitter.next_particle == len(EXPECTED_INDICES[-1]), (
        f"Emitter cursor should have recycled before the last emission and ended at {len(EXPECTED_INDICES[-1])}, "
        f"got {emitter.next_particle}."
    )
    assert len(written) == 3 * len(EMIT_SIZES), f"Expected {3 * len(EMIT_SIZES)} particle writes, got {len(written)}."

    # The state the solver actually holds, read through the public getters.
    all_poss = emitter.entity.get_particles_pos().cpu().numpy()
    all_vels = emitter.entity.get_particles_vel().cpu().numpy()
    all_actives = emitter.entity.get_particles_active().cpu().numpy()

    written_particles = set()
    for i_emission, expected_indices in enumerate(EXPECTED_INDICES):
        emission_writes = written[3 * i_emission : 3 * (i_emission + 1)]
        for written_idx, _ in emission_writes:
            assert written_idx.tolist() == list(expected_indices), (
                f"Emission #{i_emission + 1} wrote particles {written_idx.tolist()} instead of "
                f"{list(expected_indices)}."
            )
            written_particles.update(written_idx.tolist())

        # Positions and velocities land where the emission sent them, and the emitted particles are active.
        idx = emission_writes[0][0].cpu().numpy()
        np.testing.assert_allclose(
            all_poss[idx],
            emission_writes[0][1],
            atol=1e-6,
            err_msg=f"Emission #{i_emission + 1} positions did not land in the solver state.",
        )
        np.testing.assert_allclose(
            all_vels[idx],
            np.broadcast_to(emission_writes[1][1], (len(idx), 3)),
            atol=1e-6,
            err_msg=f"Emission #{i_emission + 1} velocities did not land in the solver state.",
        )
        np.testing.assert_array_equal(
            all_actives[idx],
            True,
            err_msg=f"Emission #{i_emission + 1} left its particles inactive.",
        )

    # Particles no emission addressed stay inactive.
    unwritten = np.setdiff1d(np.arange(MAX_PARTICLES), np.fromiter(written_particles, dtype=int))
    np.testing.assert_array_equal(
        all_actives[unwritten],
        False,
        err_msg="Particles no emission addressed should stay inactive.",
    )

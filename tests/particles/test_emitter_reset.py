# Regression tests for emitter episode-local state on reset (issue #3401).
#
# When 'droplet_length' is omitted, 'Emitter.emit' accumulates the length emitted so far in
# '_acc_droplet_len' and only emits once it crosses one particle size. 'Emitter.reset' must clear
# that accumulator: 'Scene.reset' calls it on every episode restart (genesis/engine/scene.py), so
# a leaked accumulator makes the first emission of the new episode fire using the previous
# episode's partial droplet length.

import pytest

import genesis as gs


# dt = 0.04 and particle size 0.1: every sub-threshold 'emit' call accumulates 0.04, so two calls
# (0.08) stay below the threshold and a third one (0.12) crosses it.
@pytest.fixture
def emitter_scene():
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.04,
            substeps=1,
        ),
        sph_options=gs.options.SPHOptions(
            particle_size=0.1,
            lower_bound=(0.0, 0.0, 0.0),
            upper_bound=(1.0, 1.0, 1.0),
        ),
        show_viewer=False,
    )
    emitter = scene.add_emitter(
        material=gs.materials.SPH.Liquid(),
        max_particles=100,
    )
    scene.build()
    return scene, emitter


def emit_step(emitter):
    emitter.emit(
        droplet_shape="circle",
        droplet_size=0.5,
        pos=(0.5, 0.5, 0.5),
    )


def n_active_particles(emitter):
    return emitter.entity.get_particles_active().sum().item()


@pytest.mark.required
def test_reset_clears_droplet_length_accumulator(emitter_scene):
    scene, emitter = emitter_scene

    # The first episode accumulates 0.08 without emitting.
    emit_step(emitter)
    emit_step(emitter)
    assert n_active_particles(emitter) == 0

    emitter.reset()

    # A fresh emitter would have accumulated 0.04 < 0.1 here and emitted nothing. The leaked
    # accumulator instead reaches 0.12 >= 0.1 and emits right away.
    emit_step(emitter)
    assert n_active_particles(emitter) == 0

    # The accumulator must still work within the new episode: crossing the threshold emits.
    emit_step(emitter)
    emit_step(emitter)
    assert n_active_particles(emitter) > 0


@pytest.mark.required
def test_scene_reset_clears_droplet_length_accumulator(emitter_scene):
    scene, emitter = emitter_scene

    emit_step(emitter)
    emit_step(emitter)
    assert n_active_particles(emitter) == 0

    scene.reset()

    emit_step(emitter)
    assert n_active_particles(emitter) == 0

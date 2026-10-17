import pytest

from .benchmark import scenes

# mochi's own fp64 probe after the benchmark protocol (5 warm-up steps and 3 windows of 30 steps) and the deviation of
# its default fp32 build from it: the natural floor for judging Genesis in fp32. The bound is four times that floor plus
# a millimetre (Genesis fp32 lands at 2.7 times the floor on the t-shirt, at the floor on the duck).
MOCHI_PROBE = {
    "soft_duck": (-0.002267668808590001, 2.6e-9),
    "cloth_tshirt": (0.2098612281289445, 3.64e-3),
}


@pytest.mark.precision("32")
@pytest.mark.parametrize("scene_name", sorted(MOCHI_PROBE))
def test_fp32_probe_within_mochi_floor(scene_name, show_viewer):
    scene, probe = scenes.build_genesis(scene_name, n_envs=1, show_viewer=show_viewer)
    for _ in range(5 + 3 * 30):
        scene.step()
    reference, floor = MOCHI_PROBE[scene_name]
    assert abs(float(probe()) - reference) <= 4.0 * floor + 1e-3

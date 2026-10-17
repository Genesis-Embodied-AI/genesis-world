import pytest

import genesis as gs

# Regression tests for https://github.com/Genesis-Embodied-AI/genesis-world/issues/3440
# The option tests only exercise options construction, so they run without pyuipc and without a
# GPU. The passthrough test builds a real IPC scene, so it additionally requires a CUDA GPU.


@pytest.mark.required
def test_ipc_options_accept_backend_default_values():
    """libuipc supports 'fused_pcg' and 'info_stackless_bvh' and they are the pyuipc 0.0.28 defaults."""
    options = gs.options.IPCCouplerOptions(
        linear_system_solver="fused_pcg",
        collision_detection_method="info_stackless_bvh",
    )
    assert options.linear_system_solver == "fused_pcg"
    assert options.collision_detection_method == "info_stackless_bvh"


@pytest.mark.required
def test_ipc_options_accept_each_choice_individually():
    for solver in ("linear_pcg", "fused_pcg", "direct"):
        assert gs.options.IPCCouplerOptions(linear_system_solver=solver).linear_system_solver == solver
    for method in ("linear_bvh", "info_stackless_bvh", "spatial_hash"):
        assert gs.options.IPCCouplerOptions(collision_detection_method=method).collision_detection_method == method


@pytest.mark.required
def test_ipc_options_none_defaults_unchanged():
    """None must still mean 'use the libuipc default'."""
    options = gs.options.IPCCouplerOptions()
    assert options.linear_system_solver is None
    assert options.collision_detection_method is None


@pytest.mark.required
def test_ipc_options_reject_invalid_values():
    with pytest.raises(gs.GenesisException):
        gs.options.IPCCouplerOptions(linear_system_solver="unsupported_solver")
    with pytest.raises(gs.GenesisException):
        gs.options.IPCCouplerOptions(collision_detection_method="unsupported_method")


def test_ipc_options_passthrough_to_ipc_scene():
    """With pyuipc available, the explicit choices must reach the built IPC scene unchanged."""
    pytest.importorskip("uipc")

    scene = gs.Scene(
        coupler_options=gs.options.IPCCouplerOptions(
            linear_system_solver="fused_pcg",
            collision_detection_method="info_stackless_bvh",
        ),
        show_viewer=False,
    )
    scene.add_entity(
        gs.morphs.Plane(),
        material=gs.materials.Rigid(
            needs_coup=False,
        ),
    )
    scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(0, 0, 0.5),
        ),
        material=gs.materials.Rigid(
            needs_coup=False,
        ),
    )
    scene.build()

    ipc_config = scene.sim.coupler._ipc_scene.config()
    assert ipc_config["linear_system"]["solver"] == "fused_pcg"
    assert ipc_config["collision_detection"]["method"] == "info_stackless_bvh"

import os
import pathlib
import signal
import subprocess
import sys
import time

import numpy as np
import pytest
import quadrants as qd

import genesis as gs

RET_SUCCESS = 42
RET_SKIP = 43

FILE_PATH = pathlib.Path(__file__)
MODULE_ROOT_DIR = FILE_PATH.parents[2]
MODULE = ".".join((FILE_PATH.parents[1].name, FILE_PATH.parent.name, FILE_PATH.stem))


@pytest.mark.parametrize("backend", [None])  # Disable genesis initialization at worker level
@pytest.mark.parametrize(
    "order",
    [
        (True, False, True),
        (False, True, False),
    ],
    ids=["ndarray-field-ndarray", "field-ndarray-field"],
)
def test_switching(backend, order):
    # Each cycle rebuilds a box-on-plane scene and verifies the quadrants type wrappers re-resolve for the
    # newly selected backend.
    for cycle_idx, use_nd in enumerate(order):
        old_val = os.environ.get("GS_ENABLE_NDARRAY")
        os.environ["GS_ENABLE_NDARRAY"] = "1" if use_nd else "0"

        try:
            gs.init(backend=gs.cpu, seed=0)

            assert gs.use_ndarray == use_nd, f"Cycle {cycle_idx}: expected use_ndarray={use_nd}, got {gs.use_ndarray}"

            from genesis.utils.array_class import V, V_VEC, _tensor_backend

            expected_backend = qd.Backend.NDARRAY if use_nd else qd.Backend.FIELD
            assert _tensor_backend() == expected_backend, (
                f"Cycle {cycle_idx}: expected _tensor_backend()={expected_backend}, got {_tensor_backend()}"
            )

            t = V(qd.i32, (4,))
            t.fill(cycle_idx + 1)
            arr = t.to_numpy()
            np.testing.assert_array_equal(arr, np.full(4, cycle_idx + 1))

            v = V_VEC(3, qd.f32, (2,))
            assert v.to_numpy().shape == (2, 3), f"Cycle {cycle_idx}: unexpected V_VEC shape {v.to_numpy().shape}"

            scene = gs.Scene(show_viewer=False)
            scene.add_entity(gs.morphs.Plane())
            scene.add_entity(
                morph=gs.morphs.Box(
                    size=(0.4, 0.4, 0.4),
                    pos=(0.0, 0.0, 0.5),
                ),
            )
            scene.build()
            for _ in range(10):
                scene.step()

        finally:
            gs.destroy()
            if old_val is None:
                os.environ.pop("GS_ENABLE_NDARRAY", None)
            else:
                os.environ["GS_ENABLE_NDARRAY"] = old_val


def _basic_sim_child(args: list[str]):
    """Child process: init genesis, build a scene, step, destroy."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str, choices=["cpu", "gpu"], default="cpu")
    args = parser.parse_args(args)

    backend = getattr(gs.constants.backend, args.backend)
    try:
        gs.utils.get_device(backend)
    except gs.GenesisException:
        print(f"Backend '{backend}' not available on this machine", file=sys.stderr)
        sys.exit(RET_SKIP)

    gs.init(backend=backend, precision="32")

    scene = gs.Scene(show_viewer=False)
    scene.add_entity(gs.morphs.Plane())
    scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.4, 0.4, 0.4),
            pos=(0.0, 0.0, 0.5),
        ),
    )
    scene.build()
    for _ in range(10):
        scene.step()

    sys.exit(RET_SUCCESS)


@pytest.mark.parametrize("backend", [None])
@pytest.mark.parametrize("test_backend", ["cpu"])
@pytest.mark.parametrize("use_ndarray", [False, True])
def test_basic_sim_subprocess(test_backend: str, use_ndarray: bool):
    cmd_line = [
        sys.executable,
        "-m",
        MODULE,
        _basic_sim_child.__name__,
        "--backend",
        test_backend,
    ]
    env = dict(os.environ)
    env["GS_ENABLE_NDARRAY"] = "1" if use_ndarray else "0"

    proc = subprocess.run(cmd_line, capture_output=True, text=True, encoding="utf-8", env=env, cwd=MODULE_ROOT_DIR)
    if proc.returncode == RET_SKIP:
        pytest.skip(proc.stderr)
    elif proc.returncode != RET_SUCCESS:
        print(proc.stdout)
        print("-" * 100)
        print(proc.stderr)
    assert proc.returncode == RET_SUCCESS


def _termination_signal_child(args: list[str]):
    """Child process: build a scene, then ask the operating system to terminate itself."""
    gs.init(backend=gs.constants.backend.cpu, precision="32")

    scene = gs.Scene(show_viewer=False)
    scene.add_entity(gs.morphs.Plane())
    scene.build()

    os.kill(os.getpid(), signal.SIGTERM)

    # Getting here means something intercepted the signal instead of letting it through.
    time.sleep(10.0)
    sys.exit(RET_SUCCESS)


@pytest.mark.parametrize("backend", [None])
@pytest.mark.skipif(sys.platform == "win32", reason="Windows has no signal-based process termination status.")
def test_process_terminates_on_signal():
    cmd_line = [sys.executable, "-m", MODULE, _termination_signal_child.__name__]

    proc = subprocess.run(cmd_line, capture_output=True, text=True, encoding="utf-8", cwd=MODULE_ROOT_DIR)
    if proc.returncode != -signal.SIGTERM:
        print(proc.stdout)
        print("-" * 100)
        print(proc.stderr)
    assert proc.returncode == -signal.SIGTERM


if __name__ == "__main__":
    globals()[sys.argv[1]](sys.argv[2:])

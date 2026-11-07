import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from .conftest import SKIP_NO_GPU


EXAMPLES_DIR = Path(__file__).parents[1] / "examples"

# Training a policy, or evaluating the policy a training produces, is out of reach of the unit tests, and the other
# modules hold code shared by examples
IGNORE_SCRIPT_PATTERNS = {"*_train.py", "*_eval.py", "*_env.py", "behavior_cloning.py", "quadcopter_controller.py"}

# Map example scripts or directories to their required optional dependencies.
# Directory keys apply recursively to all scripts within that directory.
EXAMPLE_DEPENDENCIES = {
    "usd": ["pxr"],  # USD examples require the usd-core package (provides pxr module)
    "ipc": ["uipc"],  # Requires pyuipc package (provides uipc module)
    "single_franka_batch_render.py": ["gs_madrona"],
    "demo.py": ["LuisaRenderPy"],
}
# Examples running on a CUDA GPU alone, which the CPU fallback of 'gs.gpu' cannot stand in for
CUDA_SCRIPT_NAMES = {"single_franka_batch_render.py"}

TIMEOUT = 600


pytestmark = [
    pytest.mark.examples,
]


def _discover_examples():
    if not EXAMPLES_DIR.exists():
        raise ValueError(f"Example directory '{EXAMPLES_DIR}' does not exist.")

    return [
        path
        for path in sorted(EXAMPLES_DIR.rglob("*.py"))
        if not any([path.match(pattern) for pattern in IGNORE_SCRIPT_PATTERNS])
    ]


@pytest.mark.examples
@pytest.mark.parametrize("backend", [None])  # Disable genesis initialization at worker level
@pytest.mark.parametrize("file", _discover_examples(), ids=lambda p: p.relative_to(EXAMPLES_DIR).as_posix())
def test_example(file: Path):
    # Check for required optional dependencies (script-level and inherited from parent dirs)
    rel = file.relative_to(EXAMPLES_DIR)
    module_deps = list(EXAMPLE_DEPENDENCIES.get(rel.name, []))
    for parent in rel.parents:
        if parent != Path("."):
            module_deps.extend(EXAMPLE_DEPENDENCIES.get(parent.as_posix(), []))

    for module_name in module_deps:
        pytest.importorskip(module_name, reason=f"Python module '{module_name}' not installed.")
    # A ROCm build of torch reports its AMD GPUs through the CUDA API too
    if file.name in CUDA_SCRIPT_NAMES and not (torch.cuda.is_available() and torch.version.cuda is not None):
        pytest.skip(SKIP_NO_GPU)

    # Disable keyboard control and monitoring when running the unit tests
    env = os.environ.copy()
    env["PYNPUT_BACKEND"] = "dummy"

    path_rel = file.relative_to(EXAMPLES_DIR).as_posix()
    try:
        result = subprocess.run(
            [sys.executable, str(file)], env=env, capture_output=True, text=True, check=False, timeout=TIMEOUT
        )
    except subprocess.TimeoutExpired as e:
        err_msg = f"Timeout running example {path_rel}."
        if e.stdout is not None:
            err_msg += f"\n\n--- STDOUT ---\n{e.stdout.decode()}"
        if e.stderr is not None:
            err_msg += f"\n\n--- STDERR ---\n{e.stderr.decode()}"
        pytest.fail(err_msg)

    if result.returncode != 0:
        pytest.fail(
            f"Failed to run example {path_rel} (Exit Code {result.returncode}).\n\n"
            f"--- STDOUT ---\n{result.stdout}\n\n--- STDERR ---\n{result.stderr}"
        )

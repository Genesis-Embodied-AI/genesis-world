import os
import subprocess
import sys
import time
from pathlib import Path

import psutil
import pytest


EXAMPLES_DIR = Path(__file__).parents[1] / "examples"

ALLOW_PATTERNS = {
    "collision/**/*.py",
    "coupling/**/*.py",
    "deformable/**/*.py",
    "drone/interactive_drone.py",
    "drone/fly_route.py",
    "fluid/**/*.py",
    "ipc/**/*.py",
    "kinematic/**/*.py",
    "rendering/**/*.py",
    "rigid/**/*.py",
    "sap_coupling/**/*.py",
    "sensors/**/*.py",
    "tutorials/**/*.py",
    "usd/**/*.py",
    "viewer_plugin/**/*.py",
    "gui/**/*.py",
}
IGNORE_SCRIPT_NAMES = {
    "ddp_multi_gpu.py",
    "multi_gpu.py",
    "single_franka_batch_render.py",  # FIXME: segfault on exit
    "fem_cube_linked_with_arm.py",  # FIXME: segfault on exit (corrupted double-linked list)
    "differentiable_push.py",
    # Raytraced showcase: needs the optional LuisaRender backend and opens a GUI window to be watched
    "demo.py",
}
if sys.platform != "linux":
    IGNORE_SCRIPT_NAMES |= {
        "cut_dragon.py",
    }

# Map example scripts or directories to their required optional dependencies.
# Directory keys apply recursively to all scripts within that directory.
EXAMPLE_DEPENDENCIES = {
    "usd": ["pxr"],  # USD examples require the usd-core package (provides pxr module)
    "ipc": ["uipc"],  # Requires pyuipc package (provides uipc module)
}

TIMEOUT = 600
# Period of the memory polling of a running example, short enough to catch a growth before it exhausts the machine
MEMORY_POLL_PERIOD = 0.5


pytestmark = [
    pytest.mark.examples,
]


def _discover_examples():
    if not EXAMPLES_DIR.exists():
        raise ValueError(f"Example directory '{EXAMPLES_DIR}' does not exist.")

    files = []
    for pattern in ALLOW_PATTERNS:
        for path in EXAMPLES_DIR.glob(pattern):
            if path.name not in IGNORE_SCRIPT_NAMES:
                files.append(path)

    return sorted(files)


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

    # Disable keyboard control and monitoring when running the unit tests
    env = os.environ.copy()
    env["PYNPUT_BACKEND"] = "dummy"

    # Every example running in parallel gets an even share of the memory of the machine. An example going beyond it is
    # killed and reported, rather than exhausting the memory of the machine and taking down every process running on it.
    memory_limit = psutil.virtual_memory().total / int(os.environ.get("PYTEST_XDIST_WORKER_COUNT", "1"))

    path_rel = file.relative_to(EXAMPLES_DIR).as_posix()
    process = subprocess.Popen(
        [sys.executable, str(file)], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    process_tree = psutil.Process(process.pid)
    deadline = time.monotonic() + TIMEOUT
    memory_peak, err_msg = 0, None
    while True:
        try:
            stdout, stderr = process.communicate(timeout=MEMORY_POLL_PERIOD)
            break
        except subprocess.TimeoutExpired:
            pass
        memory = 0
        for proc in (process_tree, *process_tree.children(recursive=True)):
            try:
                memory += proc.memory_info().rss
            except psutil.NoSuchProcess:
                pass
        memory_peak = max(memory_peak, memory)
        if memory > memory_limit:
            err_msg = f"Example {path_rel} exceeded its share of memory ({memory_limit / 1024**3:.1f} GiB)."
        elif time.monotonic() > deadline:
            err_msg = f"Timeout running example {path_rel}."
        if err_msg is not None:
            for proc in (*process_tree.children(recursive=True), process_tree):
                try:
                    proc.kill()
                except psutil.NoSuchProcess:
                    pass
            stdout, stderr = process.communicate()
            break

    if err_msg is None and process.returncode != 0:
        err_msg = f"Failed to run example {path_rel} (Exit Code {process.returncode})."
    if err_msg is not None:
        pytest.fail(
            f"{err_msg} Peak memory: {memory_peak / 1024**3:.1f} GiB.\n\n"
            f"--- STDOUT ---\n{stdout}\n\n--- STDERR ---\n{stderr}"
        )

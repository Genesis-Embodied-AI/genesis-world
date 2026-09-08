import ctypes
import gc
import logging
import os
import shutil
import subprocess
import sys
import warnings
from argparse import SUPPRESS
from enum import Enum
from io import BytesIO
from pathlib import Path

import setproctitle
import psutil
import pyglet
import pytest
from _pytest.mark import Expression, MarkMatcher
from PIL import Image
from syrupy.extensions.image import PNGImageSnapshotExtension

from tests.gpu_info import detect_gpu_backend

# Determine whether EGL driver is available, which is what backs pyglet's own headless mode
has_egl = True
try:
    pyglet.lib.load_library("EGL")
except ImportError:
    has_egl = False

# Determine whether a screen is available, which only the interactive viewer needs. Resolving a display binds
# pyglet's windowing backend and would rule out its headless one, so the environment answers where headless exists.
if has_egl:
    has_display = bool(os.environ.get("DISPLAY")) if sys.platform.startswith("linux") else False
else:
    try:
        has_display = bool(pyglet.display.get_display().get_screens())
    except Exception:
        has_display = False
    if not has_display and sys.platform == "darwin":
        # A display asleep, as while the screen is locked, stops being active but stays online, and Genesis opens
        # the viewer window on it all the same, which 'genesis.utils.misc.get_default_screen' takes care of.
        from pyglet.libs.darwin.cocoapy import quartz

        num_displays = ctypes.c_uint32()
        quartz.CGGetOnlineDisplayList(0, None, ctypes.byref(num_displays))
        has_display = bool(num_displays.value)

# Determine whether the optional 'imgui-bundle' dependency is available
try:
    import imgui_bundle  # noqa: F401

    is_imgui_bundle_supported = True
except ImportError:
    is_imgui_bundle_supported = False

# Forcibly disable Mujoco OpenGL to avoid conflicts with Genesis
os.environ["MUJOCO_GL"] = "0"

# Forcibly disable tqdm to avoid random crashes on the MacOS CI
os.environ["TQDM_DISABLE"] = "1"

# pyglet must be configured in headless mode before importing Genesis if necessary.
# Note that environment variables are used instead of global options to ease option propagation to subprocesses.
if not has_display and has_egl:
    pyglet.options["headless"] = True
    os.environ["PYGLET_HEADLESS"] = "1"

IS_INTERACTIVE_VIEWER_AVAILABLE = has_display or has_egl

TOL_SINGLE = 5e-5
TOL_DOUBLE = 1e-9


# Canonical skip reason registry.
# When a test is skipped during setup with one of these reasons, the skip trace location is
# normalized to point to the definition line below instead of scattered test function locations.
# This makes the short test summary group identical skip reasons into a single line.
_CANONICAL_SKIP_LINES = {}


def _skip_reason(reason):
    _CANONICAL_SKIP_LINES[reason] = sys._getframe(1).f_lineno
    return reason


SKIP_NO_GPU = _skip_reason("No GPU available on this machine")
SKIP_METAL_64BIT = _skip_reason("Apple Metal GPU does not support 64bits precision.")
SKIP_NDARRAY_PERFORMANCE_MODE = _skip_reason(
    "Skipping unit tests requiring performance mode when running with Quadrants dynamic array mode."
)
SKIP_BACKEND_UNAVAILABLE = _skip_reason("Backend not available on this machine")
SKIP_NO_MADRONA = _skip_reason("BatchRenderer is not supported because 'gs_madrona' is not available.")
SKIP_NO_LUISA = _skip_reason("RayTracer is not supported because 'LuisaRenderPy' is not available.")
SKIP_NO_VIEWER = _skip_reason("Interactive viewer not supported on this platform.")
SKIP_NO_OMNIVERSE_KIT = _skip_reason("omniverse-kit support not available")
SKIP_NO_IMGUI_BUNDLE = _skip_reason("'imgui-bundle' is not available.")


def is_mem_monitoring_supported():
    if not sys.platform.startswith("linux"):
        return False, "mem-monitoring only supported on linux"

    backend = detect_gpu_backend()
    if backend is not None:
        return True, None

    return False, "no supported GPU backend detected"


def pytest_make_parametrize_id(config, val, argname):
    if isinstance(val, Enum):
        return val.name
    if isinstance(val, type):
        return ".".join((val.__module__, val.__name__))
    return f"{val}"


@pytest.hookimpl(tryfirst=True)
def pytest_cmdline_main(config: pytest.Config) -> None:
    # Make sure that no unsupported markers have been specified in CLI
    declared_markers = set(name for spec in config.getini("markers") if (name := spec.split(":")[0]) != "forked")
    try:
        eval(config.option.markexpr, {"__builtins__": {}}, {key: None for key in declared_markers})
    except NameError as e:
        raise pytest.UsageError(f"Unknown marker in CLI expression: '{e.name}'")

    # Benchmarks are opt-in and exclusive: they run only via '-m benchmarks' alone. Exclude benchmarks from
    # any expression that does not name them so they never run implicitly (e.g. '-m "not slow"' would match
    # them since benchmarks are not slow), and reject combining the 'benchmarks' marker with anything else.
    markexpr = config.option.markexpr
    if markexpr and "benchmarks" not in markexpr:
        config.option.markexpr = f"({markexpr}) and not benchmarks"
    elif (
        markexpr
        and markexpr.strip() != "benchmarks"
        and Expression.compile(markexpr).evaluate(MarkMatcher.from_markers((pytest.mark.benchmarks,)))
    ):
        raise pytest.UsageError(
            "The 'benchmarks' marker is exclusive and cannot be combined with other markers; "
            "run benchmarks with '-m benchmarks' alone."
        )

    # Only launch memory monitor from the main process, not from xdist workers
    mem_filepath = config.getoption("--mem-monitoring-filepath")
    if mem_filepath and not os.environ.get("PYTEST_XDIST_WORKER"):
        supported, reason = is_mem_monitoring_supported()
        if not supported:
            raise pytest.UsageError(f"--mem-monitoring-filepath is not supported on this platform: {reason}")
        subprocess.Popen(
            [
                sys.executable,
                "tests/monitor_test_mem.py",
                "--die-with-parent",
                "--out-file",
                mem_filepath,
            ]
        )

    # Make sure that benchmarks are running on GPU and the number of workers if valid
    expr = Expression.compile(config.option.markexpr)
    is_benchmarks = expr.evaluate(MarkMatcher.from_markers((pytest.mark.benchmarks,)))
    if is_benchmarks:
        # Make sure that GPU backend is enforced
        backend = config.getoption("--backend")
        if backend == "cpu":
            raise ValueError("Running benchmarks on CPU is not supported.")
        config.option.backend = "gpu"

    # Force disabling forked for non-linux systems
    if not sys.platform.startswith("linux"):
        config.option.forked = False

    # Snapshot regeneration must run serially: syrupy writes updated snapshots at session end in the main process, so
    # it is incompatible with xdist and pytest-forked. Coerce the default '-n auto' to serial like the viewer does,
    # but reject an explicitly parallel or forked run.
    if config.getoption("--snapshot-update", False):
        if config.option.forked or (isinstance(config.option.numprocesses, int) and config.option.numprocesses > 0):
            raise pytest.UsageError(
                "'--snapshot-update' requires serial execution; run with '-n 0' and without '--forked'."
            )
        config.option.numprocesses = 0

    # Force disabling distributed framework if interactive viewer is enabled
    show_viewer = config.getoption("--vis", IS_INTERACTIVE_VIEWER_AVAILABLE)
    if show_viewer:
        config.option.numprocesses = 0

    # Force disabling reruns if debugger is enabled
    is_pdb_enabled = config.getoption("--pdb")
    if is_pdb_enabled:
        config.option.reruns = 0

    # Keep the viewer off the developer's desktop when it is not wanted, the variable carrying the request to the
    # xdist workers.
    if not show_viewer:
        os.environ["GS_HEADLESS"] = "1"

    # Make sure that the number of workers is not too large if specified
    if isinstance(config.option.numprocesses, int):
        max_workers = max(pytest_xdist_auto_num_workers(config), 1)
        if config.option.numprocesses > max_workers:
            raise ValueError(f"The number of workers cannot exceed '{max_workers}' on this machine.")


def pytest_sessionstart(session: pytest.Session) -> None:
    config = session.config

    # Properly configure Quadrants std out stream right away to avoid significant performance penalty (~10%)
    # Note that this variable must be set in the main thread BEFORE spawning the distributed workers, otherwise
    # the variable will be set incorrectly. Although, Genesis is already setting this env variable properly at import,
    # relying on this mechanism is fragile.
    os.environ.setdefault("QD_ENABLE_PYBUF", "0" if sys.stdout is sys.__stdout__ else "1")

    # Enforce special environment variable before importing test modules if distributed framework is enabled
    worker_id = os.environ.get("PYTEST_XDIST_WORKER")
    if worker_id and worker_id.startswith("gw"):
        # Enforce GPU affinity
        gpu_indices = _get_gpu_indices()
        if gpu_indices:
            worker_num = int(worker_id[2:])
            gpu_index = gpu_indices[worker_num % len(gpu_indices)]
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
            os.environ["QD_VISIBLE_DEVICE"] = str(gpu_index)

        # Limit CPU threading
        expr = Expression.compile(config.option.markexpr)
        is_benchmarks = expr.evaluate(MarkMatcher.from_markers((pytest.mark.benchmarks,)))
        if is_benchmarks:
            # FIXME: Enabling multi-threading in benchmark is making compile time estimation unreliable
            num_cpu_per_worker = "1"
        else:
            physical_core_count = psutil.cpu_count(logical=config.option.logical)
            num_workers = int(os.environ["PYTEST_XDIST_WORKER_COUNT"])
            num_cpu_per_worker = str(max(int(physical_core_count / num_workers), 1))
        os.environ["QD_NUM_THREADS"] = num_cpu_per_worker
        os.environ["OMP_NUM_THREADS"] = num_cpu_per_worker
        os.environ["OPENBLAS_NUM_THREADS"] = num_cpu_per_worker
        os.environ["MKL_NUM_THREADS"] = num_cpu_per_worker
        os.environ["VECLIB_MAXIMUM_THREADS"] = num_cpu_per_worker
        os.environ["NUMEXPR_NUM_THREADS"] = num_cpu_per_worker
        os.environ["NUMBA_NUM_THREADS"] = num_cpu_per_worker

    # Avoid numba cache collision between sessions and workers.
    # Must be set before numba is imported, so it cannot live in a fixture.
    basetemp = config._tmp_path_factory.getbasetemp()
    os.environ["NUMBA_CACHE_DIR"] = str(basetemp / "numba-cache")


def _get_gpu_indices():
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible_devices is not None:
        return tuple(map(int, cuda_visible_devices.split(",")))

    if sys.platform == "linux":
        backend = detect_gpu_backend()
        if backend is not None:
            device_count = backend.get_device_count()
            if device_count > 0:
                return tuple(range(device_count))

        warnings.warn(
            "No GPU backend detected (neither NVML nor AMD SMI); multi-GPU support will be disabled.",
            stacklevel=2,
        )

    return (0,)


def _torch_get_gpu_idx(device):
    # The caller only invokes this for a CUDA device, so torch is using this GPU and its identity must be
    # confirmable. Returns the resolved physical device index, or -1 when it cannot be confirmed (no GPU
    # management library, or a UUID unknown to it), which the caller turns into a hard error rather than
    # letting an unverified device through.
    import torch

    device_uuid = str(torch.cuda.get_device_properties(device).uuid)

    backend = detect_gpu_backend()
    if backend is None:
        return -1

    return backend.get_device_index_from_uuid(device_uuid)


def _get_egl_index(gpu_index):
    from OpenGL import EGL
    from OpenGL.EGL.NV.device_cuda import EGL_CUDA_DEVICE_NV

    # Get the list of Nvidia GPU that are visible
    nvidia_gpu_indices = _get_gpu_indices()

    # Define some ctypes for convenience
    EGLDeviceEXT = ctypes.c_void_p
    EGLAttrib = ctypes.c_ssize_t
    EGLint = ctypes.c_int
    EGLuint = ctypes.c_uint

    # Load EGL extension functions dynamically
    EGLuint = ctypes.c_uint
    eglQueryDevicesEXT_addr = EGL.eglGetProcAddress(b"eglQueryDevicesEXT")
    if not eglQueryDevicesEXT_addr:
        raise RuntimeError("eglQueryDevicesEXT not available")
    eglQueryDevicesEXT = ctypes.CFUNCTYPE(EGLuint, EGLint, ctypes.POINTER(EGLDeviceEXT), ctypes.POINTER(EGLint))(
        eglQueryDevicesEXT_addr
    )
    eglQueryDeviceAttribEXT_addr = EGL.eglGetProcAddress(b"eglQueryDeviceAttribEXT")
    if not eglQueryDeviceAttribEXT_addr:
        raise RuntimeError("eglQueryDeviceAttribEXT not available")
    eglQueryDeviceAttribEXT = ctypes.CFUNCTYPE(EGLuint, EGLDeviceEXT, EGLint, ctypes.POINTER(EGLAttrib))(
        eglQueryDeviceAttribEXT_addr
    )

    # Query EGL devices
    num_devices = EGLint()
    eglQueryDevicesEXT(0, None, ctypes.byref(num_devices))
    devices = (EGLDeviceEXT * num_devices.value)()
    eglQueryDevicesEXT(num_devices, devices, ctypes.byref(num_devices))
    egl_map = {}
    for i in range(num_devices.value):
        dev = devices[i]
        cuda_id = EGLAttrib()
        if eglQueryDeviceAttribEXT(dev, EGL_CUDA_DEVICE_NV, ctypes.byref(cuda_id)):
            egl_map[nvidia_gpu_indices[cuda_id.value]] = i

    return egl_map[gpu_index]


def pytest_xdist_auto_num_workers(config):
    # Get available memory (RAM & VRAM) and number of cores
    physical_core_count = psutil.cpu_count(logical=config.option.logical)
    ram_memory = psutil.virtual_memory().total / 1024**3
    if sys.platform == "darwin":
        # On Apple ARM, cpu and gpu are part of the same physical device with unified memory
        num_gpus = 1
        vram_memory = ram_memory
    else:
        # Cannot rely on 'torch' because this would force loading devices before configuring CUDA device visibility
        devices_vram_memory = None
        backend = detect_gpu_backend()
        if backend is not None:
            devices_vram_memory = backend.get_device_vram_mib()

        if devices_vram_memory:
            assert len(set(devices_vram_memory)) == 1, "Heterogeneous GPU devices not supported."
            num_gpus = len(devices_vram_memory)
            vram_memory = sum(devices_vram_memory) / 1024
        else:
            # FIXME: There is no easy way for Intel ARC device. Ignore device visibility issue for now...
            import torch

            if torch.xpu.is_available():
                num_gpus = torch.xpu.device_count()
                vram_memory = 0.0
                for device_idx in range(num_gpus):
                    device = torch.device("xpu", device_idx)
                    device_property = torch.cuda.get_device_properties(device)
                    vram_memory += device_property.total_memory / 1024**3
            else:
                # Ignore VRAM if no GPU is available
                num_gpus = 0
                vram_memory = float("inf")

    # Compute the default number of workers based on available RAM, VRAM, and number of physical cores.
    # Note that if `forked` is not enabled, up to 7.5Gb per worker is necessary on Linux because Quadrants
    # does not completely release memory between each test.
    if sys.platform == "darwin":
        ram_memory_per_worker = vram_memory_per_worker = 3.0
    elif config.option.forked:
        ram_memory_per_worker = 5.5
        vram_memory_per_worker = 1.8
    else:
        ram_memory_per_worker = 7.5
        vram_memory_per_worker = 2.5
    num_workers = min(
        physical_core_count,
        max(ram_memory / ram_memory_per_worker, 1),
        max(vram_memory / vram_memory_per_worker, 1),
    )

    # Special treatment for benchmarks
    expr = Expression.compile(config.option.markexpr)
    is_benchmarks = expr.evaluate(MarkMatcher.from_markers((pytest.mark.benchmarks,)))
    if is_benchmarks:
        num_cpu_per_gpu = 8
        num_workers = min(
            num_workers,
            num_gpus,
            max(physical_core_count / num_cpu_per_gpu, 1),
        )

    return int(num_workers)


@pytest.hookimpl(hookwrapper=True)
def pytest_collection_modifyitems(config, items):
    # Resolve backend-conditional 'slow' markers before pytest applies '-m' selection.
    # '@pytest.mark.slow' without argument means slow on every backend, whereas
    # '@pytest.mark.slow("gpu")' (any subset of backend names) means slow only on those backends.
    # The marker is dropped from items running on a non-matching backend, so '-m "not slow"' keeps
    # them and the slow-first scheduling below treats them as fast.
    default_backend = config.getoption("--backend") or "cpu"
    for item in items:
        mark = item.get_closest_marker("slow")
        if mark is None or not mark.args:
            continue
        # 'callspec' only exists on parametrized items; fall back to the session backend otherwise.
        callspec = getattr(item, "callspec", None)
        if callspec is not None and "backend" in callspec.params:
            value = callspec.params["backend"]
            backend = value.name if isinstance(value, Enum) else str(value)
        else:
            backend = default_backend
        if backend not in mark.args:
            item.own_markers = [own for own in item.own_markers if own.name != "slow"]

    yield

    # Run slow tests first
    slow = [item for item in items if item.get_closest_marker("slow") is not None]
    fast = [item for item in items if item.get_closest_marker("slow") is None]

    max_workers = config.option.numprocesses
    if max_workers is None:
        max_workers = int(os.environ["PYTEST_XDIST_WORKER_COUNT"])
    max_workers = max(max_workers, 1)

    buckets = [[] for _ in range(max_workers)]
    for idx, item in enumerate(slow + fast):
        bucket_idx = idx % max_workers
        buckets[bucket_idx].append(item)
    items[:] = [item for bucket in sorted(buckets, key=len) for item in bucket]


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    # Convert all quadrants and torch UserWarning as errors
    warnings.filterwarnings("error", category=UserWarning, module="torch")
    warnings.filterwarnings(
        "default", message=r".*The .grad attribute of a Tensor that is not a leaf Tensor is being accessed..*"
    )
    warnings.filterwarnings(
        "default", message=r".*not currently supported on the MPS backend and will fall back to run on the CPU.*"
    )
    warnings.filterwarnings("default", message=r"\s*.*cuda capability.*")
    warnings.filterwarnings("error", category=UserWarning, module="quadrants")
    warnings.filterwarnings("default", message=r".*cannot create weak reference to 'tuple' object.*")

    # Include test name in process title
    test_name = item.nodeid.replace(" ", "")
    dtype = "field" if os.environ.get("GS_ENABLE_NDARRAY", "1") == "0" else "ndarray"
    test_name = test_name[:-1] + f"-{dtype}]"
    setproctitle.setproctitle(f"pytest: {test_name}")

    # Match CUDA device with EGL device.
    # Note that this must be done here instead of 'pytest_cmdline_main', otherwise it will segfault when using
    # 'pytest-forked', because EGL instances are not allowed to cross thread boundaries.
    worker_id = os.environ.get("PYTEST_XDIST_WORKER")
    if worker_id and worker_id.startswith("gw"):
        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cuda_visible_devices is not None:
            gpu_index = int(cuda_visible_devices)
            if has_egl:
                try:
                    os.environ["EGL_DEVICE_ID"] = str(_get_egl_index(gpu_index))
                except (AttributeError, KeyError):
                    # AttributeError: CUDA is not supported on this machine
                    # KeyError: The selected GPU does not support CUDA
                    pass


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Normalize skip trace locations so identical reasons stack into a single summary line.

    By default, pytest attributes setup-phase skips to the test item's location, scattering
    identical reasons across many lines. This hook rewrites the location to the canonical
    definition line in conftest.py for all registered skip reasons.
    """
    outcome = yield
    report = outcome.get_result()
    if report.skipped and isinstance(report.longrepr, tuple):
        _, _, reason = report.longrepr
        # pytest may prefix the reason with "Skipped: " - strip it for matching
        bare_reason = reason.removeprefix("Skipped: ")
        lineno = _CANONICAL_SKIP_LINES.get(bare_reason)
        if (
            lineno is None
            and bare_reason.startswith("Backend '")
            and bare_reason.endswith("' not available on this machine")
        ):
            lineno = _CANONICAL_SKIP_LINES[SKIP_BACKEND_UNAVAILABLE]
        if lineno is not None:
            report.longrepr = (os.path.relpath(__file__), lineno, reason)


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Print a plain list of failed test IDs at the end of the run.

    Replaces pytest's default 'FAILED test_id - msg' short summary lines (which in verbose mode
    duplicate the FAILURES section) with a compact, easy-to-scan ID-only list.
    """
    failed = terminalreporter.stats.get("failed")
    if not failed:
        return
    terminalreporter.write_sep("=", "Failed tests")
    fullwidth = terminalreporter._tw.fullwidth
    for report in failed:
        reprcrash = getattr(report.longrepr, "reprcrash", None)
        msg = " ".join(reprcrash.message.split()) if reprcrash is not None else ""
        if not msg:
            terminalreporter.write_line(report.nodeid)
            continue
        line = f"{report.nodeid} - {msg}"
        if len(line) > fullwidth:
            line = line[: fullwidth - 3] + "..."
        terminalreporter.write_line(line)


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--backend", action="store", default=None, help="Default simulation backend.")
    parser.addoption(
        "--logical", action="store_true", default=False, help="Consider logical cores in default number of workers."
    )
    if IS_INTERACTIVE_VIEWER_AVAILABLE:
        parser.addoption("--vis", action="store_true", default=False, help="Enable interactive viewer.")
    parser.addoption("--dev", action="store_true", default=False, help="Enable genesis debug mode.")
    supported, _reason = is_mem_monitoring_supported()
    help_text = (
        "Run memory monitoring, and store results to mem_monitoring_filepath. CUDA on linux ONLY."
        if supported
        else SUPPRESS
    )
    parser.addoption("--mem-monitoring-filepath", type=str, help=help_text)
    parser.addoption(
        "--speed-test-filepath",
        type=str,
        default="speed_test.txt",
        help="Base filepath for speed test reports (default: speed_test.txt).",
    )


@pytest.fixture(scope="session")
def show_viewer(pytestconfig):
    return pytestconfig.getoption("--vis", IS_INTERACTIVE_VIEWER_AVAILABLE)


@pytest.fixture(scope="session")
def backend(pytestconfig):
    return pytestconfig.getoption("--backend") or "cpu"


@pytest.fixture(scope="session")
def asset_tmp_path(tmp_path_factory):
    return tmp_path_factory.mktemp("assets")


@pytest.fixture
def tol():
    import numpy as np
    import genesis as gs

    return TOL_DOUBLE if gs.np_float == np.float64 else TOL_SINGLE


@pytest.fixture
def precision(request, backend):
    import genesis as gs

    precision = None
    for mark in request.node.iter_markers("precision"):
        if mark.args:
            if precision is not None:
                pytest.fail("'precision' can only be specified once.")
            (precision,) = mark.args
    if precision is None:
        # Only default to 64bits precision when running the unit tests on CPU backend
        expr = Expression.compile(request.config.option.markexpr)
        is_benchmarks = expr.evaluate(MarkMatcher.from_markers((pytest.mark.benchmarks,)))
        if isinstance(backend, str):
            backend = getattr(gs.constants.backend, backend)
        precision = "64" if not is_benchmarks and backend == gs.cpu else "32"
    return precision


@pytest.fixture
def mujoco_compatibility(request):
    mujoco_compatibility = None
    for mark in request.node.iter_markers("mujoco_compatibility"):
        if mark.args:
            if mujoco_compatibility is not None:
                pytest.fail("'mujoco_compatibility' can only be specified once.")
            (mujoco_compatibility,) = mark.args
    if mujoco_compatibility is None:
        mujoco_compatibility = True
    return mujoco_compatibility


@pytest.fixture
def friction_cone(request):
    import genesis as gs

    friction_cone = None
    for mark in request.node.iter_markers("friction_cone"):
        if mark.args:
            if friction_cone is not None:
                pytest.fail("'friction_cone' can only be specified once.")
            (friction_cone,) = mark.args
    if friction_cone is None:
        friction_cone = gs.friction_cone.pyramidal
    return friction_cone


@pytest.fixture
def adjacent_collision(request):
    adjacent_collision = None
    for mark in request.node.iter_markers("adjacent_collision"):
        if mark.args:
            if adjacent_collision is not None:
                pytest.fail("'adjacent_collision' can only be specified once.")
            (adjacent_collision,) = mark.args
    if adjacent_collision is None:
        adjacent_collision = False
    return adjacent_collision


@pytest.fixture
def gjk_collision(request):
    gjk_collision = None
    for mark in request.node.iter_markers("gjk_collision"):
        if mark.args:
            if gjk_collision is not None:
                pytest.fail("'gjk_collision' can only be specified once.")
            (gjk_collision,) = mark.args
    if gjk_collision is None:
        gjk_collision = False
    return gjk_collision


@pytest.fixture
def friction_torsional(request):
    friction_torsional = None
    for mark in request.node.iter_markers("friction_torsional"):
        if mark.args:
            if friction_torsional is not None:
                pytest.fail("'friction_torsional' can only be specified once.")
            (friction_torsional,) = mark.args
    if friction_torsional is None:
        friction_torsional = False
    return friction_torsional


@pytest.fixture
def friction_rolling(request):
    friction_rolling = None
    for mark in request.node.iter_markers("friction_rolling"):
        if mark.args:
            if friction_rolling is not None:
                pytest.fail("'friction_rolling' can only be specified once.")
            (friction_rolling,) = mark.args
    if friction_rolling is None:
        friction_rolling = False
    return friction_rolling


@pytest.fixture
def merge_fixed_links(request):
    merge_fixed_links = None
    for mark in request.node.iter_markers("merge_fixed_links"):
        if mark.args:
            if merge_fixed_links is not None:
                pytest.fail("'merge_fixed_links' can only be specified once.")
            (merge_fixed_links,) = mark.args
    if merge_fixed_links is None:
        merge_fixed_links = True
    return merge_fixed_links


@pytest.fixture
def multi_contact(request):
    multi_contact = None
    for mark in request.node.iter_markers("multi_contact"):
        if mark.args:
            if multi_contact is not None:
                pytest.fail("'multi_contact' can only be specified once.")
            (multi_contact,) = mark.args
    if multi_contact is None:
        multi_contact = True
    return multi_contact


@pytest.fixture
def dof_damping(request):
    dof_damping = None
    for mark in request.node.iter_markers("dof_damping"):
        if mark.args:
            if dof_damping is not None:
                pytest.fail("'dof_damping' can only be specified once.")
            (dof_damping,) = mark.args
    if dof_damping is None:
        dof_damping = False
    return dof_damping


@pytest.fixture
def cache(request):
    cache = None
    for mark in request.node.iter_markers("cache"):
        if mark.args:
            if cache is not None:
                pytest.fail("'cache' can only be specified once.")
            (cache,) = mark.args
    if cache is None:
        cache = True
    return cache


@pytest.fixture
def performance_mode(request):
    performance_mode = None
    for mark in request.node.iter_markers("performance_mode"):
        if mark.args:
            if performance_mode is not None:
                pytest.fail("'performance_mode' can only be specified once.")
            (performance_mode,) = mark.args
    return performance_mode


@pytest.fixture
def debug(request):
    debug = None
    for mark in request.node.iter_markers("debug"):
        if mark.args:
            if debug is not None:
                pytest.fail("'debug' can only be specified once.")
            (debug,) = mark.args
    return debug


@pytest.fixture
def use_deterministic_algorithms(request):
    use_deterministic_algorithms = None
    for mark in request.node.iter_markers("use_deterministic_algorithms"):
        if mark.args:
            if use_deterministic_algorithms is not None:
                pytest.fail("'use_deterministic_algorithms' can only be specified once.")
            (use_deterministic_algorithms,) = mark.args
    if use_deterministic_algorithms is None:
        use_deterministic_algorithms = True
    return use_deterministic_algorithms


@pytest.fixture(scope="function", autouse=True)
def initialize_genesis(
    request, monkeypatch, tmp_path, backend, precision, performance_mode, debug, cache, use_deterministic_algorithms
):
    import genesis as gs

    # Early return if backend is None
    if backend is None:
        yield
        return

    # Convert backend from string to enum if necessary
    if isinstance(backend, str):
        backend = getattr(gs.constants.backend, backend)

    dev_mode = request.config.getoption("--dev")
    logging_level = request.config.getoption("--log-cli-level")
    if logging_level is None:
        logging_level = logging.DEBUG if dev_mode else logging.INFO
    if debug is None:
        debug = dev_mode

    if not cache:
        monkeypatch.setenv("QD_OFFLINE_CACHE", "0")
        # FIXME: Must set temporary cache even if caching is forcibly disabled because this flag is not always honored
        monkeypatch.setenv("QD_OFFLINE_CACHE_FILE_PATH", str(tmp_path / ".cache" / "quadrants"))
        monkeypatch.setenv("GS_CACHE_FILE_PATH", str(tmp_path / ".cache" / "genesis"))

        # Wipe worker-specific cache entirely since there is no way to disable it
        numba_cache_dir = Path(os.environ["NUMBA_CACHE_DIR"])
        basetemp = request.config._tmp_path_factory.getbasetemp()
        assert numba_cache_dir.is_relative_to(basetemp)
        shutil.rmtree(numba_cache_dir, ignore_errors=True)

    # Redirect name terrain cache directory to some test-local temporary location to avoid conflict and persistence
    monkeypatch.setattr("genesis.utils.misc.get_gnd_cache_dir", lambda: str(tmp_path / ".cache" / "terrain"))

    try:
        # Skip if requested backend is not available
        try:
            gs.utils.get_device(backend)
        except gs.GenesisException:
            pytest.skip(f"Backend '{backend}' not available on this machine")

        # Skip test if not supported by this machine
        if sys.platform == "darwin" and backend != gs.cpu:
            if os.environ.get("QD_ENABLE_METAL", "1") != "0" and precision == "64":
                pytest.skip(SKIP_METAL_64BIT)

        # Skip test if performance mode is required but 'GS_ENABLE_NDARRAY' != '0' because it cannot be updated
        if performance_mode is not None and ((os.environ.get("GS_ENABLE_NDARRAY", "1") == "0") ^ performance_mode):
            pytest.skip(SKIP_NDARRAY_PERFORMANCE_MODE)

        gs.init(
            backend=backend,
            precision=precision,
            debug=debug,
            seed=0,
            logging_level=logging_level,
            performance_mode=performance_mode,
            use_deterministic_algorithms=use_deterministic_algorithms,
        )
        gc.collect()

        if gs.backend != gs.cpu and gs.device.index is not None:
            # The device torch selected must be one this worker is allowed to use. Anything else - including a
            # -1 meaning the device could not be confirmed - fails hard rather than letting an unverified device
            # through, on every platform.
            device_idx = _torch_get_gpu_idx(gs.device.index)
            if device_idx not in _get_gpu_indices():
                raise RuntimeError(f"Invalid CUDA GPU device, got {device_idx}, not in {_get_gpu_indices()}.")

        if backend != gs.cpu and gs.backend == gs.cpu:
            pytest.skip(SKIP_NO_GPU)

        yield
    finally:
        gs.destroy()
        # Double garbage collection is over-zealous since gstaichi 2.2.1 but let's do it anyway
        gc.collect()
        gc.collect()


@pytest.fixture
def mj_sim(
    xml_path,
    gs_solver,
    gs_integrator,
    merge_fixed_links,
    multi_contact,
    adjacent_collision,
    gjk_collision,
    friction_cone,
):
    from .utils.simulators import build_mujoco_sim

    return build_mujoco_sim(
        xml_path,
        gs_solver,
        gs_integrator,
        merge_fixed_links,
        multi_contact,
        adjacent_collision,
        gjk_collision,
        friction_cone=friction_cone,
    )


@pytest.fixture
def gs_sim(
    xml_paths,
    gs_solver,
    gs_integrator,
    merge_fixed_links,
    multi_contact,
    mujoco_compatibility,
    adjacent_collision,
    gjk_collision,
    friction_cone,
    friction_torsional,
    friction_rolling,
    show_viewer,
    mj_sim,
):
    from .utils.simulators import build_genesis_sim

    return build_genesis_sim(
        xml_paths,
        gs_solver,
        gs_integrator,
        merge_fixed_links,
        multi_contact,
        mujoco_compatibility,
        adjacent_collision,
        gjk_collision,
        show_viewer,
        mj_sim,
        friction_cone=friction_cone,
        friction_torsional=friction_torsional,
        friction_rolling=friction_rolling,
    )


@pytest.fixture(scope="session")
def cube_verts_and_faces():
    cx, cy, cz = (0.0, 0.0, 0.0)
    edge_length = 1.0

    h = edge_length / 2.0

    verts = [
        (cx - h, cy - h, cz - h),  # v0
        (cx + h, cy - h, cz - h),  # v1
        (cx + h, cy + h, cz - h),  # v2
        (cx - h, cy + h, cz - h),  # v3
        (cx - h, cy - h, cz + h),  # v4
        (cx + h, cy - h, cz + h),  # v5
        (cx + h, cy + h, cz + h),  # v6
        (cx - h, cy + h, cz + h),  # v7
    ]

    faces = [
        (1, 2, 3, 4),
        (5, 6, 7, 8),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 4, 8, 7),
        (4, 1, 5, 8),
    ]
    return verts, faces


@pytest.fixture(scope="session")
def box_obj_path(asset_tmp_path, cube_verts_and_faces):
    """Fixture that generates a temporary cube .obj file"""
    verts, faces = cube_verts_and_faces

    filename = str(asset_tmp_path / "fixture_box_obj_path.obj")
    with open(filename, "w", encoding="utf-8") as f:
        for x, y, z in verts:
            f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        f.write("\n")
        for a, b, c, d in faces:
            f.write(f"f {a} {b} {c} {d}\n")

    return filename


class PixelMatchSnapshotExtension(PNGImageSnapshotExtension):
    _std_err_threshold: float | None = None
    _ratio_err_threshold: float | None = None
    _blurred_kernel_size: int | None = None

    def matches(self, *, serialized_data, snapshot_data) -> bool:
        # Imported here rather than at module top: conftest must not be coupled to any other test module at load
        # time, as that can cause hard-to-debug side effects.
        from .utils.assertions import IMG_BLUR_KERNEL_SIZE, IMG_NUM_ERR_THR, IMG_STD_ERR_THR, assert_pixel_match

        std_err_threshold = IMG_STD_ERR_THR if self._std_err_threshold is None else self._std_err_threshold
        ratio_err_threshold = IMG_NUM_ERR_THR if self._ratio_err_threshold is None else self._ratio_err_threshold
        blurred_kernel_size = IMG_BLUR_KERNEL_SIZE if self._blurred_kernel_size is None else self._blurred_kernel_size
        try:
            assert_pixel_match(
                Image.open(BytesIO(serialized_data)),
                Image.open(BytesIO(snapshot_data)),
                std_err_threshold=std_err_threshold,
                ratio_err_threshold=ratio_err_threshold,
                blurred_kernel_size=blurred_kernel_size,
            )
        except AssertionError:
            return False
        return True


@pytest.fixture
def png_snapshot(request, snapshot):
    snapshot_obj = snapshot.use_extension(PixelMatchSnapshotExtension)
    snapshot_dir = Path(PixelMatchSnapshotExtension.dirname(test_location=snapshot_obj.test_location))
    snapshot_name = PixelMatchSnapshotExtension.get_snapshot_name(test_location=snapshot_obj.test_location)

    # The brackets a parametrized test carries in its name are a character class to any glob, so they must be escaped
    # for the pattern to match the files that name belongs to
    snapshot_pattern = "".join(f"[{char}]" if char in ("[", "]") else char for char in snapshot_name) + "*"

    must_update_snapshot = request.config.getoption("--snapshot-update")
    if must_update_snapshot:
        for path in (Path(snapshot_dir.parent) / snapshot_dir.name).glob(snapshot_pattern):
            assert path.is_file()
            path.unlink()
    else:
        from .utils.assets import get_hf_dataset

        # The snapshots repository mirrors the tests tree ('rendering/__snapshots__/test_offscreen/...'), so the
        # snapshot directory path relative to the tests root is also its path in the repository.
        tests_dir = Path(__file__).parent
        get_hf_dataset(
            pattern=f"{snapshot_dir.relative_to(tests_dir).as_posix()}/{snapshot_pattern}",
            repo_name="snapshots",
            local_dir=tests_dir,
        )

    return snapshot_obj


@pytest.fixture
def trajectory_snapshot(request) -> Path:
    from genesis.recorders.trajectory import TRAJECTORY_FORMAT

    from .utils.assets import get_hf_dataset

    # The path is the one syrupy gives a single-file snapshot of the test. The test reads the file itself, so syrupy
    # leaves it out of its count of unused snapshots.
    path = request.path.parent / "__snapshots__" / request.path.stem / f"{request.node.name}{TRAJECTORY_FORMAT}"
    # The brackets of a parametrized test name are escaped for the same reason as in 'png_snapshot'
    snapshot_pattern = "".join(f"[{char}]" if char in ("[", "]") else char for char in path.name)
    tests_dir = Path(__file__).parent
    get_hf_dataset(
        pattern=f"{path.parent.relative_to(tests_dir).as_posix()}/{snapshot_pattern}",
        repo_name="snapshots",
        local_dir=tests_dir,
    )
    return path

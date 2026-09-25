import io
import os
import sys
import atexit
import logging as _logging
import traceback
import weakref
from typing import Callable
from warnings import warn
from contextlib import redirect_stdout

# Import quadrants while collecting its output without printing directly
_qd_outputs = io.StringIO()

os.environ.setdefault("QD_ENABLE_PYBUF", "0" if sys.stdout is sys.__stdout__ else "1")

with redirect_stdout(_qd_outputs):
    import quadrants as qd

try:
    import torch
except ImportError as e:
    raise ImportError(
        "'torch' module not available. Please install pytorch manually: https://pytorch.org/get-started/locally/"
    ) from e
import numpy as np

from .constants import backend as _gs_backend
from .logging import Logger
from .version import __version__
from .utils import redirect_libc_stderr, set_random_seed, get_device


_IS_OLD_TORCH = tuple(map(int, torch.__version__.split(".")[:2])) < (2, 8)
# FIXME: qd.Field does not support zero-copy on Metal for 'torch<=2.9.1'.
# See: https://github.com/pytorch/pytorch/pull/168193
_TORCH_MPS_SUPPORT_DLPACK_FIELD = torch.torch_version.TorchVersion(torch.__version__.split("+", 1)[0]) > (2, 9, 1)
if _IS_OLD_TORCH:
    warn("'torch<2.8.0' is not supported. Please upgrade pytorch manually: https://pytorch.org/get-started/locally/")


# Global state
_initialized: bool = False
_scene_registry: list[weakref.ReferenceType["Scene"]] = []
_module_registry: set[tuple[Callable[[], None], Callable[[], None]]] = set()
_theme: str | None = None
logger: Logger | None = None
device: torch.device | None = None
backend: _gs_backend | None = None
use_ndarray: bool | None = None
use_zerocopy: bool | None = None
use_deterministic_algorithms: bool | None = None
debug: bool | None = None
EPS: float | None = None


########################## init ##########################
def init(
    *,
    backend=None,
    precision="32",
    logging_level=None,
    debug=False,
    seed=None,
    eps=1e-15,
    theme="dark",
    logger_verbose_time=False,
    performance_mode=False,
    use_deterministic_algorithms=False,
):
    global _initialized
    if _initialized:
        raise_exception("Genesis already initialized.")

    # Make sure everything is properly destroyed, just in case initialization failed previously
    destroy()

    # Update theme if valid
    global _theme
    if theme not in ("dark", "light", "dumb"):
        raise_exception(f"Unsupported theme: ~~<{theme}>~~")
    _theme = theme

    # Make sure that specified arch and precision are supported
    if precision not in ("32", "64"):
        raise_exception(f"Unsupported precision type: ~~<{precision}>~~")

    # Initialize the logger and print greeting message
    global logger
    if logging_level is None:
        logging_level = _logging.DEBUG if debug else _logging.INFO
    logger = Logger(logging_level, logger_verbose_time)

    try:
        columns, _lines = os.get_terminal_size()
    except OSError:
        columns = 80
    wave_width = (columns - logger.INFO_length - 11) // 2
    if wave_width % 2 == 0:
        wave_width -= 1
    wave_width = max(0, min(38, wave_width))
    bar_width = wave_width * 2 + 9
    wave = ("┈┉" * wave_width)[:wave_width]
    logger.info(f"~<╭{'─' * (bar_width)}╮>~")
    logger.info(f"~<│{wave}>~ ~~~~<Genesis>~~~~ ~<{wave}│>~")
    logger.info(f"~<╰{'─' * (bar_width)}╯>~")

    # Get device and backend
    global device
    if backend is None and debug:
        backend_candidates = [_gs_backend.cpu]
    elif backend is None or backend == _gs_backend.gpu:
        backend_candidates = [_gs_backend.cuda, _gs_backend.amdgpu, _gs_backend.metal, _gs_backend.cpu]
    else:
        backend_candidates = [backend]
    while backend_candidates:
        _backend = backend_candidates.pop(0)
        if os.environ.get(f"QD_ENABLE_{_backend.name.upper()}", "1") == "0":
            continue
        try:
            device, device_name, total_mem, _backend = get_device(_backend)
            if backend == _gs_backend.gpu and _backend == _gs_backend.cpu:
                logger.warning(f"Backend ~~<{backend}>~~ not available on this machine. Falling back to CPU.")
            backend = _backend
            break
        except GenesisException as e:
            if not backend_candidates:
                raise_exception_from(f"Backend ~~<{_backend}>~~ not available on this machine.", e)
    globals()["backend"] = backend

    # Fallback to Torch CPU device if requested
    if backend != _gs_backend.cpu and os.environ.get("GS_TORCH_FORCE_CPU_DEVICE") == "1":
        device, device_name, total_mem, _backend = get_device(_gs_backend.cpu)

    # Configure Quadrants fast cache and array type
    global use_ndarray, use_zerocopy
    is_ndarray_disabled = os.environ.get("GS_ENABLE_NDARRAY", "1") == "0"
    _use_ndarray = not (is_ndarray_disabled or performance_mode)
    use_ndarray = _use_ndarray

    # Unlike dynamic vs static array mode, and fastcache, zero-copy can be toggle on/off between init without issue
    _use_zerocopy = bool(int(os.environ["GS_ENABLE_ZEROCOPY"])) if "GS_ENABLE_ZEROCOPY" in os.environ else None
    if _use_zerocopy:
        if backend == _gs_backend.metal and not _use_ndarray and not _TORCH_MPS_SUPPORT_DLPACK_FIELD:
            raise_exception("Zero-copy not supported for static array mode on Apple Metal if 'torch<=2.9.1'.")
        if (backend == _gs_backend.metal and device.type != "mps") or (
            backend in (_gs_backend.cuda, _gs_backend.amdgpu) and device.type != "cuda"
        ):
            raise_exception(
                f"Genesis backend '{backend}' not consistent with Torch device type '{device.type}'. Zero-copy "
                "not supported."
            )

    if (
        (backend == gs.cpu and device.type == "cpu")
        or (backend in (_gs_backend.cuda, _gs_backend.amdgpu) and device.type == "cuda")
        or (backend == _gs_backend.metal and device.type == "mps" and (_use_ndarray or _TORCH_MPS_SUPPORT_DLPACK_FIELD))
    ):
        if _use_zerocopy is None:
            _use_zerocopy = True
    else:
        if _use_zerocopy:
            raise_exception(f"Zero-copy not supported on {backend} backend.")
    use_zerocopy = bool(_use_zerocopy)

    # Reproducing a rollout means settling the runtime-measured choices the simulation would otherwise keep revisiting
    # (see prefer_decomposed_solver in rigid_solver.py), at the cost of the throughput they were buying, hence opt-in.
    globals()["use_deterministic_algorithms"] = use_deterministic_algorithms
    globals()["debug"] = debug

    # Define the right dtypes in accordance with selected backend and precision
    global qd_float, np_float, tc_float
    if precision == "32":
        qd_float = qd.f32
        np_float = np.float32
        tc_float = torch.float32
    else:  # precision == "64":
        if backend == _gs_backend.metal:
            raise_exception("64bits precision is not supported on Apple Metal GPU.")
        qd_float = qd.f64
        np_float = np.float64
        tc_float = torch.float64

    # All int uses 32-bit precision, unless under special circumstances
    global qd_int, np_int, tc_int
    qd_int = qd.i32
    np_int = np.int32
    tc_int = torch.int32

    # Bool
    # Note that `qd.u1` is broken on Apple Metal.
    global qd_bool, np_bool, tc_bool
    if backend == _gs_backend.metal:
        qd_bool = qd.i32
        np_bool = np.int32
        tc_bool = torch.int32
    else:
        qd_bool = qd.u1
        np_bool = np.bool_
        tc_bool = torch.bool

    # Let's use GLSL convention: https://learnwebgl.brown37.net/12_shader_language/glsl_data_types.html
    global qd_vec2, qd_vec3, qd_vec4, qd_vec6, qd_vec7, qd_vec11, qd_mat3, qd_mat4, qd_ivec2, qd_ivec3, qd_ivec4
    qd_vec2 = qd.types.vector(2, qd_float)
    qd_vec3 = qd.types.vector(3, qd_float)
    qd_vec4 = qd.types.vector(4, qd_float)
    qd_vec6 = qd.types.vector(6, qd_float)
    qd_vec7 = qd.types.vector(7, qd_float)
    qd_vec11 = qd.types.vector(11, qd_float)
    qd_mat3 = qd.types.matrix(3, 3, qd_float)
    qd_mat4 = qd.types.matrix(4, 4, qd_float)
    qd_ivec2 = qd.types.vector(2, qd_int)
    qd_ivec3 = qd.types.vector(3, qd_int)
    qd_ivec4 = qd.types.vector(4, qd_int)

    # Update torch default dtype and device, just in case
    torch.set_default_device(device)
    torch.set_default_dtype(tc_float)

    # Define smallest float that is considered non-zero
    global EPS
    EPS = float(max(eps, np.finfo(np_float).eps))

    # Configure and initialize quadrants
    qd_init_kwargs = {}
    if logger.level == _logging.CRITICAL:
        qd_init_kwargs.update(log_level=qd.ERROR)
    elif logger.level == _logging.ERROR:
        qd_init_kwargs.update(log_level=qd.ERROR)
    elif logger.level == _logging.WARNING:
        qd_init_kwargs.update(log_level=qd.WARN)
    elif logger.level == _logging.INFO:
        qd_init_kwargs.update(log_level=qd.WARN)
    elif logger.level == _logging.DEBUG:
        qd_init_kwargs.update(log_level=qd.INFO)
    if debug:
        if backend != _gs_backend.cpu:
            logger.warning("Debug mode is partially supported for GPU backend.")
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.info("Beware running Genesis in debug mode dramatically reduces runtime speed.")

    # FIXME: Enforcing Quadrants num threads to 1 by default when running on CPU
    # because it significantly improve performance.
    qd_num_cpu_threads = os.environ.get("QD_NUM_THREADS")
    if qd_num_cpu_threads is not None:
        qd_init_kwargs.update(
            cpu_max_num_threads=int(qd_num_cpu_threads),
            num_compile_threads=int(qd_num_cpu_threads),
        )
    else:
        qd_init_kwargs.update(
            cpu_max_num_threads=1,
        )

    if seed is not None:
        global SEED
        SEED = seed
        set_random_seed(SEED)
        qd_init_kwargs.update(
            random_seed=seed,
        )

    # init quadrants
    qd_debug = debug and (os.environ.get("QD_DEBUG") != "0")
    with redirect_stdout(_qd_outputs):
        qd.init(
            arch=getattr(qd, backend.name),
            enable_fallback=False,
            # Add a (hidden) mechanism to forcible disable Quadrants debug mode as it is still a bit experimental
            debug=qd_debug and backend == _gs_backend.cpu,
            check_out_of_bound=debug and backend != _gs_backend.metal,
            # force_scalarize_matrix=True for speeding up kernel compilation
            # FIXME: Turning off 'force_scalarize_matrix' is causing numerical instabilities ('nan') on MacOS
            force_scalarize_matrix=True,
            # FIXME: Turning off 'advanced_optimization' is causing issues on MacOS
            advanced_optimization=True,
            # This improves runtime speed by around 1%-5%, while it makes compilation up to 6x slower
            cfg_optimization=False,
            fast_math=not debug,
            default_ip=qd_int,
            default_fp=qd_float,
            # This feature is necessary to support auto-diff with non-static for-loop:
            # * Up to 500MiB static upper-bound mem alloc per kernel before switching to tight runtime-based bound
            ad_stack_sparse_threshold_bytes=200_000_000,
            ad_stack_experimental_enabled=True,
            **qd_init_kwargs,
        )

    # Disable debug checks for quadrants
    qd.lang._template_mapper.__builtins__["__debug__"] = qd_debug

    logger.info(
        f"Running on ~~<[{device_name}]>~~ with backend ~~<{backend}>~~. Device memory: ~~<{total_mem:.2f}>~~ GB."
    )

    for qd_output in _qd_outputs.getvalue().splitlines():
        logger.debug(qd_output)
    _qd_outputs.truncate(0)
    _qd_outputs.seek(0)

    # Redirect Quadrants logging messages to unify logging management
    for qd_name, gs_name in (
        ("debug", "debug"),
        ("trace", "debug"),
        ("info", "debug"),
        ("warn", "info"),
        ("error", "warning"),
        ("critical", "error"),
    ):
        setattr(qd._logging, qd_name, getattr(logger, gs_name))

    # Dealing with default backend
    if use_ndarray:
        logger.debug("[Quadrants] Enabling Quadrants dynamic array type to avoid scene-specific compilation.")
    if backend == _gs_backend.amdgpu:
        logger.debug("[Quadrants] Beware AMD GPU backend is still experimental and may be unstable.")

    if _IS_OLD_TORCH:
        logger.warning(
            "'torch<2.8.0' is not supported. Please upgrade pytorch manually: https://pytorch.org/get-started/locally/"
        )
    elif backend == _gs_backend.metal and not _TORCH_MPS_SUPPORT_DLPACK_FIELD:
        logger.warning(
            "'torch<2.9.1' does not supported zero-copy on Apple Metal. Consider upgrading pytorch to improve "
            "runtime performance: https://pytorch.org/get-started/locally/"
        )

    msg_options = ", ".join(
        f"{name}: ~~<{val}>~~"
        for name, val in (
            ("🔖 version", __version__),
            ("🎨 theme", theme),
            ("🌱 seed", seed),
            ("🐛 debug", bool(debug)),
            ("📏 precision", precision),
            ("🔥 performance", bool(performance_mode)),
            ("💬 verbose", _logging.getLevelName(logger.level)),
        )
    )
    logger.info(f"🚀 Genesis initialized. {msg_options}")

    if _use_zerocopy is None:
        logger.warning(
            "Zero-copy mode not enabled because Genesis backend is not supported or Torch device is not consistent "
            "with it. This will reduce performance."
        )

    atexit.register(destroy)
    _initialized = True

    # Initialize all externally registered modules
    for init_fun, _destroy_fun in _module_registry:
        init_fun()


########################## destroy ##########################


def destroy():
    """
    This call destroys all scenes, releasing all gpu memories allocated and runtime data, then forces caching of
    compiled kernels.

    Note that gs.init() needs to be called again to re-initialize Genesis after destroy.
    """
    # Early return if not initialized
    global _initialized
    if not _initialized:
        return

    # Do not consider Genesis as initialized at this point
    _initialized = False

    # Unregister at-exit callback that is not longer relevant.
    # This is important when `init` / `destroy` is called multiple times, which is typically the case for unit tests.
    atexit.unregister(destroy)

    # Display any buffered error message if logger is configured
    global logger
    if logger:
        logger.info("💤 Exiting Genesis and caching compiled kernels...")

    # Destroy all scenes. A weakref that no longer resolves means the scene was already garbage-collected (and its
    # resources released), so there is nothing left to destroy - skip it rather than asserting.
    global _scene_registry
    for scene_ref in _scene_registry.copy():
        scene = scene_ref()
        if scene is not None:
            scene.destroy()

    # Release every module-level asset cache (parsed meshes, baked RGBA textures, ...) so the large arrays they hold
    # for destroyed scenes are not retained globally.
    _clear_caches()

    # Destroy all externally registered modules
    for _init_fun, destroy_fun in _module_registry:
        destroy_fun()

    # Reset Quadrants
    qd.reset()

    # Restore original Quadrants logging facilities
    for qd_name, qd_level in (
        ("debug", qd._logging.DEBUG),
        ("trace", qd._logging.TRACE),
        ("info", qd._logging.INFO),
        ("warn", qd._logging.WARN),
        ("error", qd._logging.ERROR),
        ("critical", qd._logging.CRITICAL),
    ):
        setattr(qd._logging, qd_name, qd._logging._get_logging(qd_level))

    # Delete logger
    logger.removeHandler(logger.handler)
    logger = None

    # Clear global state
    global _theme, device, backend, EPS
    _theme = None
    device = None
    backend = None
    EPS = None


####################### External module registration hook #######################


def register_external_module(init_fun: Callable[[], None], destroy_fun: Callable[[], None]) -> None:
    assert isinstance(init_fun, Callable) and isinstance(destroy_fun, Callable)
    _module_registry.add((init_fun, destroy_fun))

    # Call init right away if Genesis is already initialized
    if gs._initialized:
        init_fun()


def unregister_external_module(init_fun: Callable[[], None], destroy_fun: Callable[[], None]) -> None:
    assert isinstance(init_fun, Callable) and isinstance(destroy_fun, Callable)
    _module_registry.remove((init_fun, destroy_fun))


########################## Exception and exit handling ##########################


class GenesisException(Exception):
    pass


def _custom_excepthook(exctype, value, tb):
    print("".join(traceback.format_exception(exctype, value, tb)))

    # Log the exception right before exit if possible
    global logger
    try:
        logger.error(f"{exctype.__name__}: {value}")
    except (AttributeError, NameError):
        # Logger may not be configured at this point
        pass


# Set the custom excepthook to handle GenesisException
sys.excepthook = _custom_excepthook

########################## shortcut imports for users ##########################

from .ext import _trimesh_patch
from .utils.misc import get_src_dir as _get_src_dir
from .utils.misc import clear_caches as _clear_caches

# Eagerly load native extensions under redirected stderr to silence dlopen-time noise (e.g. macOS
# objc duplicate-class warnings when several libraries ship their own copy of GLFW).
with open(os.devnull, "w") as stderr, redirect_libc_stderr(stderr):
    try:
        from pygel3d import graph, hmesh
    except OSError as e:
        # Import may fail because of missing system dependencies (libGLU.so.1).
        # This is not blocking because it is only an issue for hybrid entities.
        pass

    try:
        import imgui_bundle  # noqa: F401
    except ImportError:
        pass

    try:
        sys.path.append(os.path.join(_get_src_dir(), "ext/LuisaRender/build/bin"))
        import LuisaRenderPy as _LuisaRenderPy
    except ImportError:
        pass

from .constants import (
    IntEnum,
    JOINT_TYPE,
    GEOM_TYPE,
    EQUALITY_TYPE,
    CTRL_MODE,
    PARA_LEVEL,
    ACTIVE,
    INACTIVE,
    integrator,
    constraint_solver,
    friction_cone,
    contact_resolution,
    broadphase_traversal,
    link_ref_frame,
)

from .utils.uid import UID
from .utils import tools
from .utils.geom import *
from .utils.misc import assert_built, assert_unbuilt, assert_initialized, raise_exception, raise_exception_from

from .options import morphs
from .options import sensors
from .options import renderers
from .options import surfaces
from .options import textures

from .datatypes import List
from .grad.creation_ops import *

from .engine import states, materials, force_fields
from .engine.mesh import Mesh
from .engine.scene import Scene

from . import recorders

for name, member in _gs_backend.__members__.items():
    globals()[name] = member

import argparse
import gc
import os
import pathlib
import subprocess
import sys

import pytest
import numpy as np

import genesis as gs
from genesis.utils.misc import qd_to_torch

from ..utils.assertions import assert_allclose


RET_SUCCESS = 42
RET_SKIP = 43

FILE_PATH = pathlib.Path(__file__)
MODULE_ROOT_DIR = FILE_PATH.parents[2]
MODULE = ".".join((FILE_PATH.parents[1].name, FILE_PATH.parent.name, FILE_PATH.stem))


def _initialize_genesis(backend: gs.constants.backend | str):
    if isinstance(backend, str):
        backend = getattr(gs.constants.backend, backend)

    # Skip if requested backend is not available
    try:
        gs.utils.get_device(backend)
    except gs.GenesisException:
        print(f"Backend '{backend}' not available on this machine", file=sys.stderr)
        sys.exit(RET_SKIP)

    gs.init(backend=backend, precision="32")

    if backend != gs.cpu and gs.backend == gs.cpu:
        print("No GPU available on this machine", file=sys.stderr)
        sys.exit(RET_SKIP)


@pytest.mark.parametrize("batch_shape", [(2, 3, 5), ()])
@pytest.mark.parametrize(
    "qd_type_spec, arg_shape",
    [
        (("field", "scalar"), ()),
        (("field", "vector"), (7,)),
        (("field", "matrix"), (7, 1)),
        (("field", "matrix"), (7, 11)),
        (("ndarray", "scalar"), ()),
        (("ndarray", "vector"), (7,)),
        (("ndarray", "matrix"), (7, 1)),
        (("ndarray", "matrix"), (7, 11)),
    ],
)
def test_to_torch(qd_type_spec, batch_shape, arg_shape):
    import quadrants as qd

    for _ in range(10):
        QD_TYPE_MAP = {
            ("field", "scalar"): qd.field,
            ("field", "vector"): qd.Vector.field,
            ("field", "matrix"): qd.Matrix.field,
            ("ndarray", "scalar"): qd.ndarray,
            ("ndarray", "vector"): qd.Vector.ndarray,
            ("ndarray", "matrix"): qd.Matrix.ndarray,
        }

        np_arg = np.asarray(np.random.rand(*batch_shape, *arg_shape), dtype=np.float32)
        qd_arg = QD_TYPE_MAP[qd_type_spec](*arg_shape, dtype=qd.f32, shape=batch_shape)
        qd_arg.from_numpy(np_arg)
        assert_allclose(qd_to_torch(qd_arg), qd_arg.to_numpy(), tol=gs.EPS)

        # Restart quadrants runtime
        arch_idx = int(qd.cfg.arch)
        debug = qd.cfg.debug
        qd.reset()
        qd.init(arch=qd._lib.core.Arch(arch_idx), debug=debug)
        gc.collect()


def gs_static_child(args: list[str]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str, choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--enable-multi-contact", action="store_true")
    parser.add_argument("--expected-num-contacts", type=int, required=True)
    parser.add_argument("--expected-use-src-ll-cache", type=int, required=True)
    parser.add_argument("--expected-src-ll-cache-hit", type=int, required=True)
    args = parser.parse_args(args)

    _initialize_genesis(backend=args.backend)

    scene = gs.Scene(
        rigid_options=gs.options.RigidOptions(
            enable_multi_contact=args.enable_multi_contact,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1, 1, 0.5),
            camera_lookat=(0.0, 0.0, 0.0),
        ),
        show_viewer=False,
    )
    scene.add_entity(
        gs.morphs.Plane(),
    )
    scene.add_entity(
        gs.morphs.Box(
            size=(0.4, 0.4, 0.4),
            pos=(0.0, 0.0, 0.18),
        )
    )
    scene.build()

    scene.rigid_solver.collider.detection()
    actual_contacts = scene.rigid_solver.collider.collider_state.n_contacts.to_numpy()
    assert actual_contacts == args.expected_num_contacts
    if scene.rigid_solver.collider.collider_config.has_non_box_plane_convex_convex:
        from genesis.engine.solvers.rigid.collider import _func_narrowphase_contact0

        kernel_to_check = _func_narrowphase_contact0
    else:
        from genesis.engine.solvers.rigid.collider import func_narrow_phase_convex_specializations

        kernel_to_check = func_narrow_phase_convex_specializations

    assert kernel_to_check._primal.src_ll_cache_observations.cache_key_generated == args.expected_use_src_ll_cache
    assert kernel_to_check._primal.src_ll_cache_observations.cache_validated == args.expected_src_ll_cache_hit
    assert kernel_to_check._primal.src_ll_cache_observations.cache_loaded == args.expected_src_ll_cache_hit

    sys.exit(RET_SUCCESS)


@pytest.mark.required
@pytest.mark.parametrize("backend", [None])  # Disable genesis initialization at worker level
@pytest.mark.parametrize("test_backend", ["cpu", "gpu"])
@pytest.mark.parametrize("use_ndarray", [False, True])
@pytest.mark.parametrize("enable_multicontact, expected_num_contacts", [(False, 1), (True, 4)])
def test_static(
    enable_multicontact: bool,
    expected_num_contacts: int,
    test_backend: str,
    use_ndarray: bool,
    tmp_path: pathlib.Path,
) -> None:
    for it in range(3):
        cmd_line = [
            sys.executable,
            "-m",
            MODULE,
            gs_static_child.__name__,
            "--expected-num-contacts",
            str(expected_num_contacts),
            "--expected-use-src-ll-cache",
            "1" if use_ndarray else "0",
            "--expected-src-ll-cache-hit",
            "1" if use_ndarray and it > 0 else "0",
            "--backend",
            test_backend,
        ]
        if enable_multicontact:
            cmd_line += ["--enable-multi-contact"]
        env = dict(os.environ)
        env.pop("GS_ENABLE_ZEROCOPY", None)
        env["GS_ENABLE_NDARRAY"] = "1" if use_ndarray else "0"
        env["QD_OFFLINE_CACHE"] = "1"
        env["QD_OFFLINE_CACHE_FILE_PATH"] = str(tmp_path)

        proc = subprocess.run(cmd_line, capture_output=True, text=True, encoding="utf-8", env=env, cwd=MODULE_ROOT_DIR)
        return_code = proc.returncode
        if return_code == RET_SKIP:
            pytest.skip(proc.stderr)
        elif return_code != RET_SUCCESS:
            print(proc.stdout)
            print("-" * 100)
            print(proc.stderr)
        assert return_code == RET_SUCCESS


def gs_num_envs_child(args: list[str]):
    import quadrants as qd

    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str, choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--n_envs", type=int, required=True)
    parser.add_argument("--expected-use-src-ll-cache", action="store_true")
    parser.add_argument("--expected-src-ll-cache-hit", action="store_true")
    parser.add_argument("--expected-fe-ll-cache-hit", action="store_true")
    args = parser.parse_args(args)

    _initialize_genesis(backend=args.backend)

    scene = gs.Scene(show_viewer=False)
    scene.add_entity(
        gs.morphs.Plane(),
    )
    scene.add_entity(
        gs.morphs.Box(
            size=(0.4, 0.4, 0.4),
            pos=(0.0, 0.0, 0.18),
        )
    )
    scene.build(n_envs=args.n_envs, env_spacing=(0.5, 0.5))

    scene.rigid_solver.collider.detection()
    qd.sync()

    from genesis.engine.solvers.rigid.rigid_solver import kernel_step_1

    assert kernel_step_1._primal.fe_ll_cache_observations.cache_hit == args.expected_fe_ll_cache_hit
    assert kernel_step_1._primal.src_ll_cache_observations.cache_key_generated == args.expected_use_src_ll_cache
    assert kernel_step_1._primal.src_ll_cache_observations.cache_loaded == args.expected_src_ll_cache_hit

    sys.exit(RET_SUCCESS)


@pytest.mark.required
@pytest.mark.parametrize("backend", [None])  # Disable genesis initialization at worker level
@pytest.mark.parametrize("test_backend", ["cpu", "gpu"])
@pytest.mark.parametrize("use_ndarray", [False, True])
def test_num_envs(use_ndarray: bool, test_backend: str, tmp_path: pathlib.Path) -> None:
    # Change n_envs each time, and check effect on reading from cache
    for it, n_envs in enumerate([3, 5, 7]):
        cmd_line = [
            sys.executable,
            "-m",
            MODULE,
            gs_num_envs_child.__name__,
            "--backend",
            test_backend,
            "--n_envs",
            str(n_envs),
        ]
        env = dict(os.environ)
        env.pop("GS_ENABLE_ZEROCOPY", None)
        env["GS_ENABLE_NDARRAY"] = "1" if use_ndarray else "0"
        env["QD_OFFLINE_CACHE"] = "1"
        env["QD_OFFLINE_CACHE_FILE_PATH"] = str(tmp_path)
        # Fastcache is always on. For ndarray, fastcache (src-ll-cache) kicks in so fe-ll-cache is never reached.
        # For fields, fastcache silently falls back but n_envs changes each iteration so no fe-ll-cache hit either.
        expected_fe_ll_cache_hit = False
        expected_use_src_ll_cache = use_ndarray
        expected_src_ll_cache_hit = use_ndarray and it > 0
        if expected_fe_ll_cache_hit:
            cmd_line += ["--expected-fe-ll-cache-hit"]
        if expected_use_src_ll_cache:
            cmd_line += ["--expected-use-src-ll-cache"]
        if expected_src_ll_cache_hit:
            cmd_line += ["--expected-src-ll-cache-hit"]
        proc = subprocess.run(cmd_line, capture_output=True, text=True, encoding="utf-8", env=env, cwd=MODULE_ROOT_DIR)
        return_code = proc.returncode
        if return_code == RET_SKIP:
            pytest.skip(proc.stderr)
        elif return_code != RET_SUCCESS:
            print(proc.stdout)
            print("-" * 100)
            print(proc.stderr)
        assert return_code == RET_SUCCESS


def change_scene(args: list[str]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str, choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--n_objs", type=int, required=True)
    parser.add_argument("--n_envs", type=int, required=True)
    parser.add_argument("--expected-src-ll-cache-hit", type=int, required=True)
    args = parser.parse_args(args)

    _initialize_genesis(backend=args.backend)

    scene = gs.Scene(
        rigid_options=gs.options.RigidOptions(
            sparse_solve=False,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1, 1, 0.5),
            camera_lookat=(0.0, 0.0, 0.0),
        ),
        profiling_options=gs.options.ProfilingOptions(
            show_FPS=False,
        ),
        show_viewer=False,
    )
    scene.add_entity(
        gs.morphs.Plane(),
    )
    for i_obj in range(args.n_objs):
        scene.add_entity(
            gs.morphs.Box(
                size=(0.4, 0.4, 0.4),
                pos=(0.0, 0.5 * i_obj, 0.5),
            )
        )
    scene.build(n_envs=args.n_envs)

    for _ in range(60):
        scene.step()

    qpos = scene.sim.rigid_solver.get_qpos()
    if args.n_envs > 0:
        assert qpos.ndim == 2
        assert qpos.shape[0] == args.n_envs
    else:
        assert qpos.ndim == 1
    assert qpos.shape[-1] == args.n_objs * 7

    z = qpos.reshape((*qpos.shape[:-1], args.n_objs, 7))[..., 2]
    assert_allclose(z, 0.2, atol=1e-3)

    from genesis.engine.solvers.rigid.rigid_solver import kernel_step_1

    assert kernel_step_1._primal.src_ll_cache_observations.cache_validated == args.expected_src_ll_cache_hit
    assert kernel_step_1._primal.src_ll_cache_observations.cache_loaded == args.expected_src_ll_cache_hit

    sys.exit(RET_SUCCESS)


@pytest.mark.slow  # ~200s
@pytest.mark.required
@pytest.mark.parametrize("backend", [None])  # Disable genesis initialization at worker level
@pytest.mark.parametrize(
    "test_backend, list_n_objs_n_envs",
    [
        ("gpu", [(3, 0), (4, 1)]),
        ("gpu", [(3, 3), (4, 4)]),
        ("cpu", [(2, 0), (4, 4)]),
    ],
)
def test_ndarray_no_compile(
    list_n_objs_n_envs: list[tuple[int, int]], test_backend: str, tmp_path: pathlib.Path
) -> None:
    # Iterate to make sure stuff is really being read from cache
    for i, (n_objs, n_envs) in enumerate(list_n_objs_n_envs):
        cmd_line = [
            sys.executable,
            "-m",
            MODULE,
            change_scene.__name__,
            "--n_objs",
            str(n_objs),
            "--n_envs",
            str(n_envs),
            "--expected-src-ll-cache-hit",
            "1" if i > 0 else "0",
            "--backend",
            test_backend,
        ]
        env = dict(os.environ)
        env.pop("GS_ENABLE_ZEROCOPY", None)
        env["GS_ENABLE_NDARRAY"] = "1"
        env["QD_OFFLINE_CACHE"] = "1"
        env["QD_OFFLINE_CACHE_FILE_PATH"] = str(tmp_path)

        proc = subprocess.run(cmd_line, capture_output=True, text=True, encoding="utf-8", env=env, cwd=MODULE_ROOT_DIR)
        return_code = proc.returncode
        if return_code == RET_SKIP:
            pytest.skip(proc.stderr)
        elif return_code != RET_SUCCESS:
            print(proc.stdout)
            print("-" * 100)
            print(proc.stderr)
        assert return_code == RET_SUCCESS


@pytest.mark.parametrize("n_dofs", [32, 62, 64, 92])
def test_linear_to_lower_tri(n_dofs):
    # Metal's GPU sqrt can return slightly-below-exact results for perfect squares (e.g. sqrt(11881) ->
    # ~108.999 instead of 109), so the sqrt-based triangular index formula must post-correct the row index
    # against off-by-one selection.
    import quadrants as qd
    from genesis.engine.solvers.rigid.constraint.solver import linear_to_lower_tri

    n_lower_tri = n_dofs * (n_dofs + 1) // 2
    result_row = qd.field(dtype=qd.i32, shape=(n_lower_tri,))
    result_col = qd.field(dtype=qd.i32, shape=(n_lower_tri,))

    @qd.kernel
    def compute_indices():
        for i_pair in range(n_lower_tri):
            i_d1, i_d2 = linear_to_lower_tri(i_pair)
            result_row[i_pair] = i_d1
            result_col[i_pair] = i_d2

    compute_indices()
    qd.sync()

    rows = result_row.to_numpy()
    cols = result_col.to_numpy()

    expected_row = np.zeros(n_lower_tri, dtype=np.int32)
    expected_col = np.zeros(n_lower_tri, dtype=np.int32)
    idx = 0
    for r in range(n_dofs):
        for c in range(r + 1):
            expected_row[idx] = r
            expected_col[idx] = c
            idx += 1

    np.testing.assert_array_equal(rows, expected_row)
    np.testing.assert_array_equal(cols, expected_col)


# The following lines are critical for the test to work
if __name__ == "__main__":
    globals()[sys.argv[1]](sys.argv[2:])

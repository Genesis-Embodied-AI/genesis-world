"""Time Genesis' MochiSolver on the benchmark scenes.

    python tests/mochi/benchmark/bench_genesis.py rigid [--backend cpu|gpu] [--precision 64|32] [--n-envs B] [--profile]

Protocol (shared with bench_mochi.py): warm-up steps, then `--n-windows` windows of `--n-steps` steps; the best and the
mean window are reported in ms/step. Every run also records the solver's memory (per environment and static, from
`memory_report`), the process' device memory (NVML, when available), the Newton and conjugate-gradient
iteration counts and the usage of the bounded contact lists. `--profile` adds the quadrants kernel profiler (launches per
step, kernel time, top kernels), `--hits` reads the contact-hit lists back (counts, redundancy of the self-contact
couplings), `--cprofile` profiles the Python side of a step. Results are written as JSON into `--out` (default `results/`).
"""

import argparse
import contextlib
import cProfile
import dataclasses
import io
import json
import os
import pstats
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import scenes

MIB = 2**20


def vram_mib():
    """(device memory of this process, total device memory) in MiB, or (None, None) without NVML/amdsmi."""
    try:
        from gpu_info import detect_gpu_backend

        backend = detect_gpu_backend()
        if backend is None:
            return None, None
        return backend.get_per_process_vram_mib().get(os.getpid()), backend.get_device_vram_mib()[0]
    except (ImportError, OSError, RuntimeError, KeyError, IndexError):
        return None, None


def memory_report(solver):
    """Bytes held by the arrays of the solver: totals, the per-environment and static parts and every field by size.

    An array counts as per environment when one of its axes has the batch size, which is unambiguous from two
    environments on.
    """
    import numpy as np
    import torch
    from quadrants.lang.util import to_numpy_type

    _B = solver._B

    def tensor_bytes(tensor):
        if hasattr(tensor, "get_member_field") and hasattr(tensor, "keys"):
            keys = tensor.keys() if callable(tensor.keys) else tensor.keys
            return sum(tensor_bytes(tensor.get_member_field(key))[0] for key in keys), tuple(tensor.shape)
        shape = tuple(int(n) for n in tensor.shape)
        element_shape = tuple(int(n) for n in (getattr(tensor, "element_shape", None) or ()))
        n_bytes = np.prod(shape, dtype=np.int64) * np.prod(element_shape, dtype=np.int64)
        return int(n_bytes) * np.dtype(to_numpy_type(tensor.dtype)).itemsize, shape

    def is_tensor(value):
        return hasattr(value, "shape") and hasattr(value, "dtype") and not isinstance(value, (np.ndarray, torch.Tensor))

    def walk(obj, prefix, fields, seen):
        if obj is None or id(obj) in seen:
            return
        seen.add(id(obj))
        if dataclasses.is_dataclass(obj):
            items = [(field.name, getattr(obj, field.name)) for field in dataclasses.fields(obj)]
        elif hasattr(obj, "__dict__"):
            items = list(vars(obj).items())
        else:
            return
        for name, value in items:
            if is_tensor(value):
                n_bytes, shape = tensor_bytes(value)
                fields.append((f"{prefix}.{name}", n_bytes, shape))
            elif dataclasses.is_dataclass(value) or type(value).__module__.startswith(("genesis.", "quadrants.")):
                if not isinstance(value, (str, bytes, np.ndarray, torch.Tensor)):
                    walk(value, f"{prefix}.{name}", fields, seen)

    fields = []
    seen = set()
    for name in (
        "mochi_info",
        "mochi_state",
        "contact_state",
        "island_state",
        "eq_info",
        "eq_state",
        "soft_info",
        "soft_state",
        "hit_readback",
        "dyn_state",
        "dyn_info",
        "rigid_info",
        "geoms_init_AABB",
        "_soft_vverts_render",
        "_errno",
    ):
        value = getattr(solver, name, None)
        if is_tensor(value):
            n_bytes, shape = tensor_bytes(value)
            fields.append((name, n_bytes, shape))
        else:
            walk(value, name, fields, seen)
    if solver.sdf is not None:
        walk(solver.sdf._sdf_info, "sdf_info", fields, seen)
    total = sum(n_bytes for _, n_bytes, _ in fields)
    per_env_bytes = sum(n_bytes for _, n_bytes, shape in fields if _B >= 2 and _B in shape)
    return {
        "total_bytes": total,
        "per_env_bytes": per_env_bytes // _B if _B >= 2 else None,
        "static_bytes": total - per_env_bytes if _B >= 2 else None,
        "n_envs": _B,
        "fields": sorted(
            ({"name": name, "bytes": n_bytes, "shape": shape} for name, n_bytes, shape in fields),
            key=lambda item: -item["bytes"],
        ),
    }


def hit_statistics(solver):
    """Counts of the contact-hit lists of the last assembly (first environment) and the redundancy of the point-cloud
    couplings: unique (vertex of the sample's triangle, collider vertex) pairs per hit-vertex product."""
    from genesis.utils.misc import qd_to_numpy

    if not solver.has_soft:
        return {}
    # The lists hold the last assembly of the step, which may be a skipped line-search trial: re-assemble at the
    # accepted iterate (the contact-recording pass evaluates every environment).
    solver._record_contacts()
    soft_state = solver.soft_state
    stats = {
        "n_soft_hits": int(qd_to_numpy(soft_state.n_soft_hits)[0]),
        "n_soft_collider_hits": int(qd_to_numpy(soft_state.n_sc_hits)[0]),
        "n_point_cloud_hits": int(qd_to_numpy(soft_state.n_pc_hits)[0]),
    }
    n_pc = stats["n_point_cloud_hits"]
    if n_pc > 0:
        kind = qd_to_numpy(soft_state.pc_hit_kind_a)[:n_pc, 0]
        sample = qd_to_numpy(soft_state.pc_hit_sample_a)[:n_pc, 0]
        vert_b = qd_to_numpy(soft_state.pc_hit_vert_b)[:n_pc, 0]
        tri = qd_to_numpy(solver.soft_info.samples_verts)[:, :3]
        is_soft = kind == 1
        pairs = set()
        for i_s, i_v in zip(sample[is_soft], vert_b[is_soft]):
            for i_w in tri[i_s]:
                pairs.add((int(i_w), int(i_v)))
        stats["n_point_cloud_hits_soft"] = int(is_soft.sum())
        stats["point_cloud_pair_redundancy"] = len(pairs) / max(1, 3 * int(is_soft.sum()))
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scene", choices=sorted(scenes.SCENES))
    parser.add_argument("--backend", default="cpu", choices=["cpu", "gpu"])
    parser.add_argument("--precision", default="64", choices=["32", "64"])
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--n-steps", type=int, default=30)
    parser.add_argument("--n-windows", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--profile", action="store_true", help="quadrants kernel profiler: launches and kernel times")
    parser.add_argument("--hits", action="store_true", help="read back the contact-hit lists")
    parser.add_argument("--cprofile", action="store_true", help="profile the Python side of the step")
    parser.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
    parser.add_argument("--tag", default="", help="suffix of the result file name")
    parser.add_argument("--option", action="append", default=[], help="MochiOptions override, e.g. linear_solver=pcg")
    parser.add_argument("--cold", action="store_true", help="disable the quadrants offline cache: measures compilation")
    parser.add_argument("--compile-threads", type=int, default=None, help="quadrants num_compile_threads (default 4)")
    args = parser.parse_args()

    import quadrants as qd

    if args.profile or args.cold or args.compile_threads is not None:
        _qd_init = qd.init

        def _init(**kwargs):
            if args.profile:
                kwargs["kernel_profiler"] = True
            if args.cold:
                kwargs["offline_cache"] = False
            if args.compile_threads is not None:
                kwargs["num_compile_threads"] = args.compile_threads
            return _qd_init(**kwargs)

        qd.init = _init

    import genesis as gs

    gs.init(backend=gs.gpu if args.backend == "gpu" else gs.cpu, precision=args.precision, logging_level="warning")

    overrides = {}
    for item in args.option:
        key, value = item.split("=", 1)
        overrides[key] = json.loads(value) if value[0] in '0123456789-[{tfn"' else value
    t0 = time.perf_counter()
    scene, probe = scenes.build_genesis(args.scene, n_envs=args.n_envs, **overrides)
    build_time = time.perf_counter() - t0
    solver = scene.mochi_solver

    # The first step compiles (or loads from the cache) every kernel of the step; with --cold it is the compile time.
    t0 = time.perf_counter()
    scene.step()
    qd.sync()
    first_step_time = time.perf_counter() - t0
    for _ in range(args.warmup - 1):
        scene.step()
    qd.sync()
    warmup_time = time.perf_counter() - t0

    windows = []
    for _ in range(args.n_windows):
        t0 = time.perf_counter()
        for _ in range(args.n_steps):
            scene.step()
        qd.sync()
        windows.append((time.perf_counter() - t0) / args.n_steps * 1e3)
    info = solver.get_convergence_info()
    memory = memory_report(solver)
    vram_process, vram_device = vram_mib()
    mem_per_env = memory["per_env_bytes"] / MIB if memory["per_env_bytes"] is not None else None
    max_envs = None
    if mem_per_env and vram_process is not None and vram_device is not None:
        # Memory not proportional to the batch (CUDA context, compiled kernels, static arrays) is what the process
        # holds beyond the batched arrays; the rest of the device is available for more environments.
        fixed = max(0.0, vram_process - args.n_envs * mem_per_env)
        max_envs = int((vram_device - fixed) // mem_per_env)
    result = {
        "engine": "genesis",
        "scene": args.scene,
        "backend": args.backend,
        "precision": args.precision,
        "n_envs": args.n_envs,
        "n_steps": args.n_steps,
        "n_windows": args.n_windows,
        "warmup": args.warmup,
        "ms_per_step_best": min(windows),
        "ms_per_step_mean": sum(windows) / len(windows),
        "ms_per_step_windows": windows,
        "us_per_env_step_best": min(windows) / max(1, args.n_envs) * 1e3,
        "build_s": build_time,
        "warmup_s": warmup_time,
        "first_step_s": first_step_time,
        "cold": args.cold,
        "compile_threads": args.compile_threads,
        "step_kernel": solver._resolve_step_kernel(),
        "n_dofs_total": int(solver.n_dofs_total),
        "n_samples": int(solver.n_samples),
        "n_soft_verts": int(solver.n_soft_verts),
        "n_soft_elems": int(solver.n_soft_elems),
        "n_shell_elems": int(solver.n_shell_elems),
        "n_rod_elems": int(solver.n_rod_elems),
        "newton_iterations_last": int(info["n_iter"][0]),
        "newton_iterations_max": int(info["n_iter"].max()),
        "pcg_iterations_last": int(info["n_pcg_iter"][0]),
        "pcg_iterations_max": int(info["n_pcg_iter"].max()),
        "mem_mib_total": memory["total_bytes"] / MIB,
        "mem_mib_per_env": mem_per_env,
        "mem_mib_static": memory["static_bytes"] / MIB if memory["static_bytes"] is not None else None,
        "mem_top": [
            {"name": field["name"], "mib": field["bytes"] / MIB, "shape": list(field["shape"])}
            for field in memory["fields"][:12]
        ],
        "vram_mib_process": vram_process,
        "vram_mib_device": vram_device,
        "max_envs_estimate": max_envs,
        "contact_capacity_usage": {k: list(v) for k, v in solver.get_contact_capacity_usage().items()},
        "probe": probe(),
        "options": overrides,
    }
    if args.hits:
        result["hits"] = hit_statistics(solver)
    if args.profile:
        qd.profiler.clear_kernel_profiler_info()
        t0 = time.perf_counter()
        for _ in range(args.n_steps):
            scene.step()
        qd.sync()
        wall = time.perf_counter() - t0
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            qd.profiler.print_kernel_profiler_info("count")
        table = buffer.getvalue()
        rows = re.findall(r"\[\s*([\d.]+)%\s+([\d.]+) s\s+(\d+)x \|\s+([\d.]+)\s+([\d.]+)\s+([\d.]+) ms\] (\S+)", table)
        result.update(
            profile_ms_per_step=wall / args.n_steps * 1e3,
            kernel_ms_per_step=qd.profiler.get_kernel_profiler_total_time() / args.n_steps * 1e3,
            launches_per_step=sum(int(r[2]) for r in rows) / args.n_steps,
            top_kernels=[
                {
                    "name": r[6],
                    "percent": float(r[0]),
                    "calls_per_step": int(r[2]) / args.n_steps,
                    "avg_ms": float(r[4]),
                }
                for r in rows[:15]
            ],
        )
    if args.cprofile:
        profiler = cProfile.Profile()
        profiler.enable()
        for _ in range(args.n_steps):
            scene.step()
        qd.sync()
        profiler.disable()
        buffer = io.StringIO()
        pstats.Stats(profiler, stream=buffer).sort_stats("tottime").print_stats(20)
        result["cprofile"] = buffer.getvalue()
    os.makedirs(args.out, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    path = os.path.join(args.out, f"genesis_{args.scene}_{args.backend}_fp{args.precision}_B{args.n_envs}{tag}.json")
    with open(path, "w") as fp:
        json.dump(result, fp, indent=2)
    mem_text = f"{mem_per_env:.2f} MiB/env" if mem_per_env is not None else f"{result['mem_mib_total']:.1f} MiB total"
    vram_text = f", vram {vram_process} MiB" if vram_process is not None else ""
    print(
        f"[genesis] {args.scene} {args.backend} fp{args.precision} B={args.n_envs}: "
        f"best {result['ms_per_step_best']:.3f} ms/step ({result['us_per_env_step_best']:.1f} us/env-step), "
        f"mean {result['ms_per_step_mean']:.3f} ms/step, newton {result['newton_iterations_last']}, "
        f"pcg {result['pcg_iterations_last']}, {mem_text}{vram_text}, probe {result['probe']:.4f}, "
        f"{result['step_kernel']} first step {result['first_step_s']:.1f} s{' (cold)' if args.cold else ''}"
        + (
            f", launches/step {result['launches_per_step']:.1f}, kernel {result['kernel_ms_per_step']:.3f} ms/step"
            if args.profile
            else ""
        )
    )
    print(f"    capacity usage (max over envs / capacity): {result['contact_capacity_usage']}")
    if args.hits:
        print(f"    hits: {result['hits']}")
    if args.profile:
        for row in result["top_kernels"][:8]:
            print(
                f"    {row['percent']:5.1f}%  {row['calls_per_step']:7.1f}/step  {row['avg_ms']:.3f} ms  {row['name']}"
            )
    if args.cprofile:
        print(result["cprofile"])
    print(f"[genesis] wrote {path}")


if __name__ == "__main__":
    main()

"""Time the original mochi engine on the benchmark scenes (run with the superdex environment's Python).

    SUPERDEX_PRECISION=double python tests/mochi/benchmark/bench_mochi.py rigid [--threads 0]

Same protocol as bench_genesis.py. The rigid and articulated scenes need the meshes exported by bench_genesis.py.
"""

import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import scenes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scene", choices=sorted(k for k, v in scenes.SCENES.items() if not v.get("genesis_only")))
    parser.add_argument("--threads", type=int, default=0, help="mochi worker threads: 0 single-threaded, -1 auto")
    parser.add_argument("--n-steps", type=int, default=30)
    parser.add_argument("--n-windows", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    from superdex import physics

    physics.initialize(num_worker_threads=args.threads)
    precision = "64" if physics.uses_double_precision() else "32"
    t0 = time.perf_counter()
    scene, probe = scenes.build_mochi(args.scene)
    build_time = time.perf_counter() - t0
    for _ in range(args.warmup):
        scene.step(scenes.DT)
    windows = []
    for _ in range(args.n_windows):
        t0 = time.perf_counter()
        for _ in range(args.n_steps):
            scene.step(scenes.DT)
        windows.append((time.perf_counter() - t0) / args.n_steps * 1e3)
    stats = scene.get_solver_stats()

    # Linear-solver iteration counts: the solver stats do not expose them, the verbose Newton log does. One extra
    # window with verbose logging (untimed), the messages caught by the log callback.
    linear_iterations = []
    newton_iterations = []
    pattern_linear = re.compile(r"Linear solver converged after (\d+) iterations")
    pattern_newton = re.compile(r"Newton-Raphson iteration (\d+)")

    def on_log(*args):
        text = " ".join(str(arg) for arg in args)
        match = pattern_linear.search(text)
        if match:
            linear_iterations[-1] += int(match.group(1))
        match = pattern_newton.search(text)
        if match:
            newton_iterations[-1] = max(newton_iterations[-1], int(match.group(1)) + 1)

    params = scene.get_solver_params()
    params.non_linear_solver.verbosity = physics.VerbosityLevel.VERBOSE
    scene.set_solver_params(params)
    physics.set_log_callback(on_log)
    for _ in range(args.n_steps):
        linear_iterations.append(0)
        newton_iterations.append(0)
        scene.step(scenes.DT)
    params.non_linear_solver.verbosity = physics.VerbosityLevel.WARNING
    scene.set_solver_params(params)
    result = {
        "engine": "mochi",
        "scene": args.scene,
        "backend": "cpu",
        "precision": precision,
        "threads": args.threads,
        "n_envs": 1,
        "n_steps": args.n_steps,
        "n_windows": args.n_windows,
        "warmup": args.warmup,
        "ms_per_step_best": min(windows),
        "ms_per_step_mean": sum(windows) / len(windows),
        "ms_per_step_windows": windows,
        "build_s": build_time,
        "newton_iterations_last": int(getattr(stats, "max_non_linear_iters", -1)),
        "linear_iterations_per_step": sum(linear_iterations) / max(1, len(linear_iterations)),
        "newton_iterations_per_step": sum(newton_iterations) / max(1, len(newton_iterations)),
        "probe": probe(),
    }
    os.makedirs(args.out, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    path = os.path.join(args.out, f"mochi_{args.scene}_cpu_fp{precision}_T{args.threads}{tag}.json")
    with open(path, "w") as fp:
        json.dump(result, fp, indent=2)
    print(
        f"[mochi] {args.scene} fp{precision} threads={args.threads}: best {result['ms_per_step_best']:.3f} ms/step, "
        f"mean {result['ms_per_step_mean']:.3f} ms/step, newton {result['newton_iterations_last']}, "
        f"linear iterations/step {result['linear_iterations_per_step']:.1f}, probe {result['probe']:.4f}"
    )
    print(f"[mochi] wrote {path}")
    physics.shutdown()


if __name__ == "__main__":
    main()

"""Run bench_genesis.py over a list of batch sizes and print the scaling table.

    python tests/mochi/benchmark/sweep.py cloth_arm --backend gpu --precision 32 --n-envs 1,64,256,1024,4096 [--tag x]

Every batch size runs in its own process (the solver's arrays are allocated once per build); the sweep stops at the
first failing run (typically out of device memory) and reports what fitted.
"""

import argparse
import json
import os
import subprocess
import sys

BENCH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench_genesis.py")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scene")
    parser.add_argument("--backend", default="gpu")
    parser.add_argument("--precision", default="32")
    parser.add_argument("--n-envs", default="1,64,256,1024,4096")
    parser.add_argument("--tag", default="")
    parser.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
    args, extra = parser.parse_known_args()

    rows = []
    for n_envs in [int(x) for x in args.n_envs.split(",")]:
        command = [
            sys.executable,
            BENCH,
            args.scene,
            "--backend",
            args.backend,
            "--precision",
            args.precision,
            "--n-envs",
            str(n_envs),
            "--out",
            args.out,
            *extra,
        ]
        if args.tag:
            command += ["--tag", args.tag]
        status = subprocess.run(command, check=False)
        tag = f"_{args.tag}" if args.tag else ""
        path = os.path.join(args.out, f"genesis_{args.scene}_{args.backend}_fp{args.precision}_B{n_envs}{tag}.json")
        if status.returncode != 0 or not os.path.exists(path):
            print(f"[sweep] {args.scene} B={n_envs} failed (exit {status.returncode}); stopping the sweep")
            break
        with open(path) as fp:
            rows.append(json.load(fp))
    print(f"\n[sweep] {args.scene} {args.backend} fp{args.precision}")
    print("| B | ms/step | us per env-step | MiB/env | process VRAM MiB | newton | pcg |")
    print("|---|---|---|---|---|---|---|")
    for row in rows:
        mem = f"{row['mem_mib_per_env']:.2f}" if row.get("mem_mib_per_env") is not None else "-"
        vram = row.get("vram_mib_process")
        print(
            f"| {row['n_envs']} | {row['ms_per_step_best']:.2f} | {row['us_per_env_step_best']:.1f} | {mem} | "
            f"{vram if vram is not None else '-'} | {row['newton_iterations_last']} | {row['pcg_iterations_last']} |"
        )


if __name__ == "__main__":
    main()

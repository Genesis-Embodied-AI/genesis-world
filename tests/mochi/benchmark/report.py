"""Merge the JSON results of bench_genesis.py and bench_mochi.py into markdown tables.

python tests/mochi/benchmark/report.py [results_dir] [--precision 64] [--threads 0]

The first table compares one environment on the CPU with the original engine; the second lists the GPU fp32 scaling of
every scene that was swept over batch sizes (ms/step, us per environment step, MiB per environment).
"""

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import scenes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "results", nargs="?", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    )
    parser.add_argument("--precision", default="64", help="precision of the CPU comparison")
    parser.add_argument("--threads", type=int, default=0, help="mochi thread setting used as the reference")
    args = parser.parse_args()

    # the most recent result of each configuration wins
    runs = []
    for path in sorted(glob.glob(os.path.join(args.results, "*.json")), key=os.path.getmtime):
        with open(path) as fp:
            runs.append(json.load(fp))

    def latest_gpu_runs(name):
        by_envs = {
            r["n_envs"]: r
            for r in runs
            if r["engine"] == "genesis"
            and r["scene"] == name
            and r["backend"] == "gpu"
            and r["precision"] == "32"
            and not r.get("options")
        }
        return [by_envs[n_envs] for n_envs in sorted(by_envs)]

    print(
        f"| scene | mochi fp{args.precision} (T={args.threads}) ms/step | Genesis CPU fp{args.precision} ms/step | ratio "
        "| launches/step | CG iterations/step (mochi / Genesis) | MiB/env | notes |"
    )
    print("|---|---|---|---|---|---|---|---|")
    for name in scenes.SCENES:
        mochi = [
            r
            for r in runs
            if r["engine"] == "mochi"
            and r["scene"] == name
            and r["precision"] == args.precision
            and r["threads"] == args.threads
        ]
        gen_cpu = [
            r
            for r in runs
            if r["engine"] == "genesis"
            and r["scene"] == name
            and r["backend"] == "cpu"
            and r["precision"] == args.precision
            and r["n_envs"] == 1
            and not r.get("options")
        ]
        # the reference engine's time is the best over its runs (its iteration counts come from the latest run)
        mochi_ms = min(r["ms_per_step_best"] for r in mochi) if mochi else None
        m = f"{mochi_ms:.3f}" if mochi else "-"
        g = f"{gen_cpu[-1]['ms_per_step_best']:.3f}" if gen_cpu else "-"
        ratio = f"{gen_cpu[-1]['ms_per_step_best'] / mochi_ms:.1f}x" if mochi and gen_cpu else "-"
        # the launch count comes from the most recent profiled run of the configuration
        profiled = [r for r in gen_cpu if "launches_per_step" in r]
        launches = f"{profiled[-1]['launches_per_step']:.0f}" if profiled else "-"
        cg_mochi = "-"
        if mochi and "linear_iterations_per_step" in mochi[-1]:
            cg_mochi = f"{mochi[-1]['linear_iterations_per_step']:.0f}"
        cg_genesis = "-"
        if gen_cpu and "pcg_iterations_last" in gen_cpu[-1]:
            cg_genesis = f"{gen_cpu[-1]['pcg_iterations_last']}"
        with_memory = [
            r for r in runs if r["engine"] == "genesis" and r["scene"] == name and r.get("mem_mib_per_env") is not None
        ]
        memory = f"{with_memory[-1]['mem_mib_per_env']:.2f}" if with_memory else "-"
        notes = []
        if gen_cpu:
            notes.append(f"{gen_cpu[-1]['n_dofs_total']} dofs, newton {gen_cpu[-1]['newton_iterations_last']}")
        if mochi:
            notes.append(f"mochi newton {mochi[-1]['newton_iterations_last']}")
        print(
            f"| {name} | {m} | {g} | {ratio} | {launches} | {cg_mochi} / {cg_genesis} | {memory} | {'; '.join(notes)} |"
        )

    print()
    print("GPU fp32 scaling (best window; MiB/env from the solver's memory report; VRAM = this process' device memory)")
    print()
    print("| scene | B | ms/step | us per env-step | MiB/env | process VRAM MiB | newton | CG iterations/step |")
    print("|---|---|---|---|---|---|---|---|")
    for name in scenes.SCENES:
        for r in latest_gpu_runs(name):
            mem = f"{r['mem_mib_per_env']:.2f}" if r.get("mem_mib_per_env") is not None else "-"
            vram = r.get("vram_mib_process")
            us_env = r.get("us_per_env_step_best", r["ms_per_step_best"] / max(1, r["n_envs"]) * 1e3)
            print(
                f"| {name} | {r['n_envs']} | {r['ms_per_step_best']:.2f} | {us_env:.1f} | {mem} | "
                f"{vram if vram is not None else '-'} | {r['newton_iterations_last']} | "
                f"{r.get('pcg_iterations_last', '-')} |"
            )


if __name__ == "__main__":
    main()

"""Plot a Probability Lab export against a published, UNMATCHED real-world reference.

This compares outcome frequencies, not trajectories or simulator accuracy. The
blue point is a published aggregate estimate, never a fabricated real trial path.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from examples.probability.model import wilson_interval


REAL_COLOR = "#2274A5"
SIM_COLOR = "#DB6726"
REFERENCE = {
    "title": "Fair coins tend to land on the same side they started: Evidence from 350,757 flips",
    "authors": "Bartos et al.",
    "version": "arXiv:2310.04153v4, 2025-04-17, abstract and Methods",
    "url": "https://arxiv.org/html/2310.04153v4",
    "accepted_flips": 350757,
    "heads_estimate": 0.500,
    "heads_interval_95": [0.498, 0.502],
    "interval_type": "Published Bayesian credible interval, rounded in the abstract",
    "protocol": "Human flips caught in hand; failed catches repeated; many coins and people",
}


def summarize_export(data):
    """Check trial accounting before computing conditional and unconditional rates."""
    if data.get("config", {}).get("kind") != "coin" or data.get("labels") != ["Heads", "Tails"]:
        raise ValueError("Use a coin export with Heads/Tails labels.")
    trials = data.get("trials", [])
    if not trials:
        raise ValueError("The export contains no completed trials.")
    outcomes = [trial["outcome"] for trial in trials]
    if any(type(outcome) is not int or outcome not in (-1, 0, 1) for outcome in outcomes):
        raise ValueError("Coin outcomes must be integer -1 (unresolved), 0, or 1.")
    counts = [outcomes.count(0), outcomes.count(1), outcomes.count(-1)]
    if data.get("counts") != counts[:2] or data.get("unresolved") != counts[2]:
        raise ValueError("Stored totals disagree with the trial records.")
    resolved = counts[0] + counts[1]
    return {
        "total": len(trials), "resolved": resolved,
        "counts_heads_tails_unresolved": counts,
        "heads_given_resolved": counts[0] / resolved if resolved else None,
        "heads_given_resolved_interval_95": list(wilson_interval(counts[0], resolved)) if resolved else None,
        "unresolved_fraction": counts[2] / len(trials),
        "config": data["config"],
    }


def create_chart(export_path, output_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    export_path, output_path = Path(export_path), Path(output_path)
    raw = export_path.read_bytes()
    data = json.loads(raw)
    summary = summarize_export(data)
    n, resolved = summary["total"], summary["resolved"]
    counts = np.array(summary["counts_heads_tails_unresolved"])
    p = counts / n
    intervals = np.array([wilson_interval(int(k), n) for k in counts])
    real_p = REFERENCE["heads_estimate"]
    real_low, real_high = REFERENCE["heads_interval_95"]

    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False}):
        fig, (left, right) = plt.subplots(1, 2, figsize=(14, 7.5), gridspec_kw={"width_ratios": [1.05, 1]})
        fig.subplots_adjust(top=0.73, bottom=0.34, left=0.085, right=0.97, wspace=0.32)
        fig.suptitle("Real coin tosses vs your Genesis experiment", x=0.085, y=0.965,
                     ha="left", fontsize=23, weight="bold")
        fig.text(0.085, 0.907, "PUBLISHED REFERENCE  /  DIFFERENT CONDITIONS  /  NOT AN ACCURACY TEST",
                 color="#8B4513", fontsize=11, weight="bold")
        fig.text(0.085, 0.857,
                 "Blue: real coins caught in the hand.  Orange: simulated coins landing on a floor.\n"
                 "Neither the coin geometry nor the launch/contact conditions are matched.",
                 color="#465366", linespacing=1.6)

        left.errorbar(real_p, 1, xerr=[[real_p - real_low], [real_high - real_p]],
                      fmt="o", color=REAL_COLOR, capsize=6, markersize=9, linewidth=2)
        left.text(0.03, 0.88, "Real: 50.0%  [49.8%, 50.2%]\n350,757 accepted catches; published 95% credible interval",
                  transform=left.transAxes, color=REAL_COLOR, fontsize=10)
        if resolved:
            estimate = summary["heads_given_resolved"]
            low, high = summary["heads_given_resolved_interval_95"]
            left.errorbar(estimate, 0, xerr=[[estimate - low], [high - estimate]],
                          fmt="o", color=SIM_COLOR, capsize=6, markersize=9, linewidth=2)
            sim_text = (f"Genesis: {estimate:.1%}  [{low:.1%}, {high:.1%}]\n"
                        f"{counts[0]} heads / {resolved} resolved tosses; 95% Wilson interval")
        else:
            sim_text = "Genesis: no resolved tosses; heads rate is undefined"
        left.text(0.03, 0.11, sim_text, transform=left.transAxes, color=SIM_COLOR, fontsize=10)
        left.axvline(0.5, color="#64748B", linestyle="--", alpha=0.6, zorder=0)
        left.set(title="A   Heads among accepted / resolved tosses", xlabel="Heads probability (%)",
                 yticks=[1, 0], yticklabels=["Real", "Genesis"], ylim=(-0.6, 1.65), xlim=(0, 1))
        left.xaxis.set_major_formatter(PercentFormatter(1))
        left.grid(axis="x", alpha=0.15)

        right.bar(np.arange(3), p, color=SIM_COLOR, width=0.55,
                  yerr=np.maximum(0, np.array([p - intervals[:, 0], intervals[:, 1] - p])),
                  capsize=5, ecolor="#475569", label="Genesis: all recorded trials")
        right.patches[2].set_hatch("///")
        for i, (count, fraction, interval) in enumerate(zip(counts, p, intervals)):
            right.text(i, interval[1] + 0.045, f"{count} / {n}\n{fraction:.1%}",
                       ha="center", fontsize=11, color="#334155")
        right.set(title="B   Keep unresolved tosses visible", ylabel="Share of all simulated tosses (%)",
                  xticks=np.arange(3), xticklabels=["Heads", "Tails", "Unresolved"], ylim=(0, 1.13))
        right.yaxis.set_major_formatter(PercentFormatter(1))
        right.set_yticks(np.arange(0, 1.01, 0.2))
        right.legend(loc="upper left", fontsize=9, frameon=False)
        right.grid(axis="y", alpha=0.15)

        horizon = summary["config"]["horizon"]
        fig.text(0.085, 0.20,
                 f"{summary['unresolved_fraction']:.1%} unresolved by {horizon:g} s. "
                 "This category can include still-moving coins and edge/ambiguous poses.\n"
                 "The real study repeated failed catches, so it supplies no comparable floor-edge or timeout rate.",
                 fontsize=11, color="#334155", linespacing=1.7)
        fig.text(0.085, 0.115,
                 "Intervals describe different estimands and methods; overlap is not evidence of physical equivalence.\n"
                 "Trajectory errors, bounce differences and settling-time differences require matched tracking data.",
                 fontsize=10, color="#465366", linespacing=1.7)
        fig.text(0.085, 0.045,
                 "Real source: Bartos et al., arXiv:2310.04153v4 (abstract, Methods).  "
                 f"Genesis: {export_path.name}\nNo synthetic measurements or illustrative outcomes are plotted.",
                 fontsize=8.5, color="#64748B", linespacing=1.6)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path.with_suffix(".png"), dpi=180, facecolor="white")
        fig.savefig(output_path.with_suffix(".svg"), facecolor="white")
        plt.close(fig)

    report = {"comparison_kind": "unmatched_published_outcome_reference", "illustrative": False,
              "matched_conditions": False, "real_reference": REFERENCE, "simulation": summary,
              "export_path": str(export_path.resolve()), "export_sha256": hashlib.sha256(raw).hexdigest(),
              "source_run_versions": data.get("versions"), "source_run_commit": data.get("genesis_commit"),
              "source_run_hashes": data.get("app_source_sha256"),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "interpretation": "No measured trajectory errors, accuracy claim or equivalence test; different protocols."}
    output_path.with_suffix(".json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", required=True, help="Probability Lab coin JSON export")
    parser.add_argument("--out", default="out/probability/published-comparison.png")
    args = parser.parse_args()
    report = create_chart(args.export, args.out)
    print(json.dumps(report["simulation"], indent=2))
    print(f"Saved PNG, SVG and provenance JSON at {Path(args.out).with_suffix('')}")


if __name__ == "__main__":
    main()

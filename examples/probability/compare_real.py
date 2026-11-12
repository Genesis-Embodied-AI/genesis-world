"""Matched coin/die replay and comparison; see COMPARISON.md for the measurement contract.

The demo is synthetic on BOTH sides. `simulate` and `drop-check` invoke Genesis. Nothing in
this module turns existing endpoint-only Probability Lab exports into real data.
"""

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from examples.probability.model import classify


# State columns: COM position, body-to-world quaternion (wxyz), WORLD v and omega.
COLUMNS = "x_m y_m z_m qw qx qy qz vx_m_s vy_m_s vz_m_s wx_rad_s wy_rad_s wz_rad_s".split()
DIE_NORMALS = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0], [0, -1, 0], [-1, 0, 0], [0, 0, -1]])
REAL_COLOR, SIM_COLOR = "#2274A5", "#DB6726"


@dataclass(frozen=True)
class Coin:
    kind = "coin"
    outcomes = ("heads", "tails", "edge", "unresolved")
    mass_kg: float
    radius_m: float
    thickness_m: float

    def __post_init__(self):
        if not all(np.isfinite(x) and x > 0 for x in asdict(self).values()):
            raise ValueError("Coin dimensions and mass must be positive and finite.")

    @property
    def inertia(self):
        # Uniform solid cylinder, with its symmetry axis along body +Z.
        transverse = self.mass_kg * (3 * self.radius_m**2 + self.thickness_m**2) / 12
        return (transverse, transverse, self.mass_kg * self.radius_m**2 / 2)

    @property
    def half_height(self):
        return self.thickness_m / 2

    @property
    def dimensions_label(self):
        return f"radius {self.radius_m * 1000:g} mm · thickness {self.thickness_m * 1000:g} mm"


@dataclass(frozen=True)
class Die:
    kind = "die"
    outcomes = ("1", "2", "3", "4", "5", "6", "unresolved")
    mass_kg: float
    edge_m: float

    def __post_init__(self):
        if not all(np.isfinite(x) and x > 0 for x in asdict(self).values()):
            raise ValueError("Die edge length and mass must be positive and finite.")

    @property
    def inertia(self):
        return (self.mass_kg * self.edge_m**2 / 6,) * 3

    @property
    def half_height(self):
        return self.edge_m / 2

    @property
    def dimensions_label(self):
        return f"edge {self.edge_m * 1000:g} mm · ideal uniform cube"


def shape_from_manifest(manifest):
    kind = manifest.get("kind", "coin")  # Original coin manifests remain valid.
    if kind not in ("coin", "die"):
        raise ValueError("Manifest kind must be coin or die.")
    return (Coin if kind == "coin" else Die)(**manifest[kind])


@dataclass(frozen=True)
class Rules:
    contact_gap_m: float = 0.001
    minimum_bounce_clearance_m: float = 0.002
    minimum_flight_s: float = 0.01
    rest_linear_m_s: float = 0.015
    rest_angular_rad_s: float = 0.15
    rest_duration_s: float = 0.25
    face_cosine: float = 0.985
    edge_cosine: float = 0.1
    maximum_tracking_gap_s: float = 0.02

    def __post_init__(self):
        if not all(np.isfinite(x) and x > 0 for x in asdict(self).values()):
            raise ValueError("Analysis thresholds must be positive and finite.")
        if not 0 < self.edge_cosine < self.face_cosine < 1:
            raise ValueError("Require 0 < edge_cosine < face_cosine < 1.")
        if self.minimum_bounce_clearance_m <= self.contact_gap_m:
            raise ValueError("Bounce clearance must exceed the contact tolerance.")


def validate_track(t, state, name="track"):
    if len(t) < 3 or state.shape != (len(t), 13):
        raise ValueError(f"{name}: need >=3 time samples and 13 state columns.")
    if not np.isfinite(t).all() or not np.isfinite(state).all():
        raise ValueError(f"{name}: missing or nonfinite measurements.")
    if abs(t[0]) > 1e-8 or np.any(np.diff(t) <= 0):
        raise ValueError(f"{name}: start at release t=0, with strictly increasing timestamps.")
    norms = np.linalg.norm(state[:, 3:7], axis=1)
    if np.any(abs(norms - 1) > 0.01):
        raise ValueError(f"{name}: quaternion norms must be within 1% of unity.")
    state = state.copy()
    state[:, 3:7] /= norms[:, None]
    return np.asarray(t, dtype=float), state


def read_tracks(path):
    tracks = {}
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = ["trial_id", "t_s", *COLUMNS]
        if not set(required) <= set(reader.fieldnames or []):
            raise ValueError("Tracking CSV requires columns: " + ",".join(required))
        for row in reader:
            trial = row["trial_id"].strip()
            if not trial:
                raise ValueError("Empty trial_id.")
            tracks.setdefault(trial, []).append([float(row[k]) for k in required[1:]])
    if not tracks:
        raise ValueError("Tracking CSV is empty.")
    # Preserve file order, including within trials; do not silently sort bad time stamps.
    return {key: validate_track(np.array(rows)[:, 0], np.array(rows)[:, 1:], key)
            for key, rows in tracks.items()}


def write_tracks(path, tracks):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["trial_id", "t_s", *COLUMNS])
        for trial, (t, state) in tracks.items():
            writer.writerows([trial, float(ti), *map(float, xi)] for ti, xi in zip(t, state))


def read_manifest(path):
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if manifest.get("data_kind") not in ("measured", "illustrative"):
        raise ValueError("Set data_kind explicitly to measured or illustrative.")
    for key in ("source", "floor_material", "tracking_method"):
        if not isinstance(manifest.get(key), str) or not manifest[key].strip():
            raise ValueError(f"Manifest needs a nonempty {key}.")
    shape_from_manifest(manifest)
    Rules(**manifest.get("analysis", {}))
    profiles = manifest.get("contact_profiles", [])
    if not profiles or len({p["name"] for p in profiles}) != len(profiles):
        raise ValueError("Provide nonempty, uniquely named contact_profiles.")
    for p in profiles:
        if not p["name"].replace("_", "").replace("-", "").isalnum():
            raise ValueError("Profile names must contain only letters, digits, '_' or '-'.")
        for key in ("friction", "timeconst_s", "dampratio", "rolling_m", "torsional_m"):
            if not np.isfinite(p[key]) or p[key] < 0:
                raise ValueError(f"Invalid contact parameter {key}.")
        if not 0.01 <= p["friction"] <= 5 or min(p["timeconst_s"], p["dampratio"]) <= 0:
            raise ValueError("Friction must be in [0.01,5]; solref entries must be positive.")
    return manifest


def align_track(t, state, target):
    if target[0] < t[0] - 1e-10 or target[-1] > t[-1] + 1e-10:
        raise ValueError("Extrapolation beyond observed data is forbidden.")
    target = np.clip(target, t[0], t[-1])
    result = np.column_stack([np.interp(target, t, state[:, i]) for i in range(13)])
    result[:, 3:7] = Slerp(t, Rotation.from_quat(state[:, 3:7], scalar_first=True))(target).as_quat(
        scalar_first=True
    )
    return result


def orientation_error_deg(a, b):
    # Normalize in float64: float32 rounding before acos creates false ~0.03° errors.
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return np.degrees(2 * np.arccos(np.clip(abs(np.sum(a * b, axis=-1)), 0, 1)))


def support_height(quaternions, shape):
    """Vertical distance from COM to lowest point; return support and body-Z dot up."""
    q = np.asarray(quaternions, dtype=float)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    nz = np.clip(1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2), -1, 1)
    if isinstance(shape, Die):
        # A cube's support is h*(|R_zx|+|R_zy|+|R_zz|), not a constant half-edge.
        support = shape.edge_m / 2 * abs(Rotation.from_quat(q, scalar_first=True).as_matrix()[:, 2, :]).sum(axis=1)
    else:
        support = shape.radius_m * np.sqrt(np.maximum(0, 1 - nz**2)) + shape.thickness_m / 2 * abs(nz)
    return support, nz


def clearance(state, shape):
    """Lowest point of the ideal object above a horizontal z=0 plane, in metres."""
    support, nz = support_height(state[:, 3:7], shape)
    return state[:, 2] - support, nz


def metrics(t, state, shape, rules):
    gap, nz = clearance(state, shape)
    contact = gap <= rules.contact_gap_m
    bounce_heights, bounce_clearances, bounce_times = [], [], []
    # Count complete contact -> flight -> contact episodes, excluding the launch flight.
    seen_contact, flight_start = False, None
    for i, touching in enumerate(contact):
        if touching:
            seen_contact = True
            if flight_start is not None:
                start = flight_start
                if (t[i] - t[start] >= rules.minimum_flight_s
                        and np.max(gap[start:i + 1]) >= rules.minimum_bounce_clearance_m):
                    bounce_heights.append(float(np.max(state[start:i + 1, 2])))
                    bounce_clearances.append(float(np.max(gap[start:i + 1])))
                    bounce_times.append(float(t[start]))
                flight_start = None
        elif seen_contact and flight_start is None:
            flight_start = i

    quiet = (contact & (np.linalg.norm(state[:, 7:10], axis=1) < rules.rest_linear_m_s)
             & (np.linalg.norm(state[:, 10:13], axis=1) < rules.rest_angular_rad_s))
    # Require rest to persist to the observation end, not just at an airborne apex.
    moving = np.flatnonzero(~quiet)
    rest_start = int(moving[-1] + 1) if len(moving) else 0
    settled = rest_start < len(t) and t[-1] - t[rest_start] >= rules.rest_duration_s
    outcome = "unresolved"
    if settled:
        if isinstance(shape, Die):
            face = int(classify(state[-1, 3:7], "die", alignment=rules.face_cosine)[0])
            outcome = str(face + 1) if face >= 0 else "unresolved"
        elif nz[-1] >= rules.face_cosine:
            outcome = "heads"
        elif nz[-1] <= -rules.face_cosine:
            outcome = "tails"
        elif abs(nz[-1]) <= rules.edge_cosine:
            outcome = "edge"
    return {
        "bounce_count": len(bounce_heights), "bounce_heights_m": bounce_heights,
        "bounce_clearances_m": bounce_clearances, "bounce_times_s": bounce_times,
        "incomplete_rebound": flight_start is not None,
        "settling_time_s": float(t[rest_start]) if settled else None,
        "outcome": outcome,
    }


def wilson(count, n):
    p = np.asarray(count) / n
    z = 1.959963984540054
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return np.maximum(0, center - half), np.minimum(1, center + half)


def bootstrap_mean(values, seed=817, repetitions=2000):
    """Pointwise interval over independent toss IDs, never over frames or replay seeds."""
    values = np.asarray(values)
    mean = values.mean(axis=0)
    if len(values) < 2:
        return mean, np.full_like(mean, np.nan), np.full_like(mean, np.nan)
    rng = np.random.default_rng(seed)
    # Multinomial weights avoid allocating [bootstrap, trial, time] arrays.
    weights = rng.multinomial(len(values), np.full(len(values), 1 / len(values)), size=repetitions)
    boot = weights @ values / len(values)
    low, high = np.quantile(boot, [0.025, 0.975], axis=0)
    return mean, low, high


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_run(path, ids, t, states, metadata):
    np.savez_compressed(path, trial_ids=np.array(ids), t_s=t, states=states,
                        metadata=np.array(json.dumps(metadata)))


def plot_comparison(reference, manifest_path, run_path, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    manifest = read_manifest(manifest_path)
    shape, rules = shape_from_manifest(manifest), Rules(**manifest.get("analysis", {}))
    tracks = read_tracks(reference)
    with np.load(run_path, allow_pickle=False) as bundle:
        ids, t, simulated = bundle["trial_ids"].tolist(), bundle["t_s"], bundle["states"]
        metadata = json.loads(str(bundle["metadata"]))
    if len(ids) != len(set(ids)) or set(ids) != set(tracks):
        raise ValueError("Real and simulated trial IDs must match exactly, without duplicates.")
    if simulated.shape != (len(ids), len(t), 13):
        raise ValueError("Simulation shape does not match the trial/time labels.")
    if metadata["reference_sha256"] != file_hash(reference):
        raise ValueError("Reference CSV changed since simulation; replay it again.")
    if metadata["manifest_sha256"] != file_hash(manifest_path):
        raise ValueError("Manifest changed since simulation; replay it again.")
    measured = []
    for i, trial in enumerate(ids):
        validate_track(t, simulated[i], f"simulation {trial}")
        ti, xi = tracks[trial]
        if np.diff(ti).max() > rules.maximum_tracking_gap_s:
            raise ValueError(f"{trial}: tracking gap exceeds the declared maximum.")
        measured.append(align_track(ti, xi, t))
    measured = np.array(measured)
    if not np.allclose(measured[:, 0], simulated[:, 0], atol=2e-6, rtol=2e-6):
        # q and -q represent the same pose; use an explicit rotation check below.
        same_linear = np.allclose(measured[:, 0, [0, 1, 2, 7, 8, 9, 10, 11, 12]],
                                  simulated[:, 0, [0, 1, 2, 7, 8, 9, 10, 11, 12]], atol=2e-6, rtol=2e-6)
        if not same_linear or np.max(orientation_error_deg(measured[:, 0, 3:7], simulated[:, 0, 3:7])) > 0.001:
            raise ValueError("Initial poses and velocities do not match the measured releases.")
    illustrative = manifest["data_kind"] != "measured" or metadata["engine"] != "Genesis"
    labels = ("Real (placeholder)" if manifest["data_kind"] != "measured" else "Real",
              "Genesis (placeholder)" if metadata["engine"] != "Genesis" else "Genesis")
    groups = [[metrics(t, x, shape, rules) for x in states] for states in (measured, simulated)]
    n = len(ids)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(3, 2, figsize=(13.6, 12.8))
    fig.subplots_adjust(top=0.865, bottom=0.155, left=0.085, right=0.97, hspace=0.57, wspace=0.28)
    subject = "Dice rolls" if isinstance(shape, Die) else "Coin tosses"
    fig.suptitle(f"{subject} | Real-world reference vs Genesis", x=0.085, y=0.977,
                 ha="left", fontsize=23, weight="bold")
    if manifest["data_kind"] != "measured" and metadata["engine"] != "Genesis":
        badge = "ILLUSTRATIVE ONLY — both series synthetic; no measured data or Genesis result"
    elif manifest["data_kind"] != "measured":
        badge = "ILLUSTRATIVE INPUT — synthetic reference; orange is an actual Genesis replay"
    elif metadata["engine"] != "Genesis":
        badge = "ILLUSTRATIVE SIMULATION — the orange series is a placeholder, not a Genesis result"
    else:
        badge = "MEASURED REFERENCE — paired replay; one fixed contact profile and replay seed"
    fig.text(0.085, 0.942, badge, color="#9A3412" if illustrative else "#175C41", fontsize=11, weight="bold")
    profile = metadata["profile"]
    fig.text(0.085, 0.912, f"{n} paired tosses  |  {profile['name']}  |  seed {metadata['seed']}  |  "
             f"mass {shape.mass_kg * 1000:g} g · {shape.dimensions_label}",
             color="#465366", fontsize=10)

    errors = (np.linalg.norm(simulated[:, :, :3] - measured[:, :, :3], axis=2) * 1000,
              orientation_error_deg(simulated[:, :, 3:7], measured[:, :, 3:7]))
    for ax, values, title, unit in zip(axes[0], errors,
            ("A  Position error against tracked reference", "B  Full orientation error against reference"),
            ("Euclidean COM error (mm)", "Geodesic rotation error (degrees)")):
        mean, low, high = bootstrap_mean(values)
        ax.plot(t, np.zeros_like(t), color=REAL_COLOR, linestyle="--", label=labels[0] + " reference")
        ax.plot(t, mean, color=SIM_COLOR, label=labels[1] + " error", linewidth=2)
        ax.fill_between(t, low, high, color=SIM_COLOR, alpha=0.2)
        ax.set(title=title, xlabel="Time since release (s)", ylabel=unit, xlim=(t[0], t[-1]))
        ax.legend(fontsize=8)
        ax.grid(alpha=0.16)

    ax = axes[1, 0]
    for j, (group, color, label) in enumerate(zip(groups, (REAL_COLOR, SIM_COLOR), labels)):
        mean, low, high = bootstrap_mean([m["bounce_count"] for m in group])
        ax.bar(j, mean, color=color, width=0.5, label=label)
        if np.isfinite(low):
            ax.errorbar(j, mean, yerr=[[max(0, mean - low)], [max(0, high - mean)]], color="#334155", capsize=5)
        censored = sum(m["incomplete_rebound"] for m in group)
        ax.annotate(f"mean {mean:.2f}\n{censored} open rebounds", (j, mean), xytext=(0, 13),
                    textcoords="offset points", ha="center", fontsize=9)
    ax.set(title="C  Completed rebounds within the window", ylabel="Mean bounce count (count / toss)",
           xticks=[0, 1], xticklabels=labels)
    ax.set_ylim(0, max(1, max(m["bounce_count"] for g in groups for m in g)) + 0.9)
    ax.grid(axis="y", alpha=0.16)

    ax = axes[1, 1]
    max_bounces = max(m["bounce_count"] for g in groups for m in g)
    shown = min(5, max_bounces)
    for j, (group, color, label) in enumerate(zip(groups, (REAL_COLOR, SIM_COLOR), labels)):
        for k in range(shown):
            heights = [m["bounce_heights_m"][k] * 1000 for m in group if m["bounce_count"] > k]
            if not heights:
                continue
            mean, low, high = bootstrap_mean(heights)
            x = k + 1 + (j - 0.5) * 0.28
            ax.bar(x, mean, width=0.26, color=color, label=label if k == 0 else None)
            if np.isfinite(low):
                ax.errorbar(x, mean, yerr=[[max(0, mean - low)], [max(0, high - mean)]], color="#334155", capsize=3)
            ax.annotate(f"n={len(heights)}", (x, mean), xytext=(0, 8 + j * 10),
                        textcoords="offset points", ha="center", fontsize=7)
    if not shown:
        ax.text(0.5, 0.5, "No completed rebounds above the threshold", transform=ax.transAxes, ha="center")
    else:
        ax.legend(fontsize=8)
        ax.margins(y=0.3)
    ax.set(title="D  Rebound apex heights (conditional on a bounce)", xlabel="Completed rebound number (first 5)",
           ylabel="COM apex above floor (mm)", xticks=range(1, shown + 1))
    ax.grid(axis="y", alpha=0.16)

    ax = axes[2, 0]
    for group, color, label in zip(groups, (REAL_COLOR, SIM_COLOR), labels):
        settling = np.array([m["settling_time_s"] if m["settling_time_s"] is not None else np.inf for m in group])
        counts = (settling[:, None] <= t).sum(axis=0)
        low, high = wilson(counts, n)
        ax.step(t, counts / n * 100, where="post", color=color, linewidth=2,
                label=f"{label} ({np.isfinite(settling).sum()}/{n} settled)")
        ax.fill_between(t, low * 100, high * 100, color=color, alpha=0.12, step="post")
    ax.set(title="E  Settling time across tosses", xlabel="Settling onset after release (s)",
           ylabel="Tosses settled by time t (%)", ylim=(0, 105), xlim=(0, t[-1]))
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.16)

    ax = axes[2, 1]
    for j, (group, color, label) in enumerate(zip(groups, (REAL_COLOR, SIM_COLOR), labels)):
        counts = np.array([sum(m["outcome"] == category for m in group) for category in shape.outcomes])
        p = counts / n
        low, high = wilson(counts, n)
        ax.bar(np.arange(len(shape.outcomes)) + (j - 0.5) * 0.36, p * 100, width=0.35, color=color, label=label,
               yerr=np.maximum(0, np.array([p - low, high - p])) * 100, capsize=3, ecolor="#334155")
    ax.set(title="F  Final outcomes (all tosses stay in denominator)", ylabel="Share of tosses (%)",
           xlabel=f"Outcome at {t[-1]:g} s", xticks=np.arange(len(shape.outcomes)),
           xticklabels=[x.title() for x in shape.outcomes])
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.16)

    fig.text(0.085, 0.035,
             "95% intervals: bootstrap over toss IDs (A–D); marginal Wilson (E–F). "
             "Bands are pointwise, not simultaneous.\n"
             "Open rebounds and unsettled tosses are censored at the shared horizon. "
             "Final faces require rest. No contact profiles or replay seeds are pooled.\n"
             + ("All differences in this preview are generated examples. They are not evidence about Genesis accuracy."
                if illustrative else "Intervals describe variation across tosses; camera calibration and tracking uncertainty are not included."),
             fontsize=9, color="#465366", linespacing=1.7)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, facecolor="white")
    fig.savefig(output.with_suffix(".svg"), facecolor="white")
    plt.close(fig)
    report = {"illustrative": illustrative, "kind": shape.kind, "n_pairs": n, "common_horizon_s": float(t[-1]),
              "run": metadata, "analysis": asdict(rules), "trials": []}
    for i, trial in enumerate(ids):
        report["trials"].append({"trial_id": trial, "real": groups[0][i], "simulation": groups[1][i],
                                 "mean_position_error_mm": float(errors[0][i].mean()),
                                 "mean_orientation_error_deg": float(errors[1][i].mean())})
    output.with_suffix(".json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return report


def object_xml(shape):
    root = ET.Element("mujoco", model=f"matched_{shape.kind}")
    ET.SubElement(root, "compiler", angle="radian", inertiafromgeom="false")
    body = ET.SubElement(ET.SubElement(root, "worldbody"), "body", name=shape.kind)
    ET.SubElement(body, "freejoint")
    ET.SubElement(body, "inertial", pos="0 0 0", mass=str(shape.mass_kg),
                  diaginertia=" ".join(map(str, shape.inertia)))
    if isinstance(shape, Die):
        ET.SubElement(body, "geom", type="box", size=" ".join([str(shape.edge_m / 2)] * 3))
    else:
        ET.SubElement(body, "geom", type="cylinder", size=f"{shape.radius_m} {shape.thickness_m / 2}")
    return root


def run_genesis(initial, shape, profile, dt, substeps, horizon, sample_dt, seed, backend):
    import genesis as gs
    from genesis.utils.misc import tensor_to_array

    if not all(np.isfinite(v) for v in (dt, horizon, sample_dt)):
        raise ValueError("Timing parameters must be finite.")
    if dt <= 0 or substeps < 1 or horizon <= 0 or sample_dt < dt or sample_dt > horizon:
        raise ValueError("Require dt>0, substeps>=1, and dt<=sample_dt<=horizon.")
    stride = round(sample_dt / dt)
    if not np.isclose(stride * dt, sample_dt, atol=1e-10, rtol=0):
        raise ValueError("sample_dt must be an integer multiple of dt.")
    if profile["timeconst_s"] < 2 * dt / substeps:
        raise ValueError("Contact timeconst must be >= twice the substep, to avoid silent solver clamping.")
    if not gs._initialized:
        gs.init(backend=gs.gpu if backend == "gpu" else gs.cpu, seed=seed, logging_level="warning",
                precision="32", use_deterministic_algorithms=True)
    gs.set_random_seed(seed)
    scene = gs.Scene(sim_options=gs.options.SimOptions(dt=dt, substeps=substeps, gravity=(0, 0, -9.81)),
                     rigid_options=gs.options.RigidOptions(enable_torsional_friction=True,
                         enable_rolling_friction=True, iterations=50, tolerance=1e-8,
                         integrator=gs.integrator.approximate_implicitfast,
                         constraint_solver=gs.constraint_solver.Newton, friction_cone=gs.friction_cone.pyramidal),
                     show_viewer=False)
    material = dict(friction=profile["friction"], friction_rolling=profile["rolling_m"],
                    friction_torsional=profile["torsional_m"])
    floor = scene.add_entity(morph=gs.morphs.Plane(), material=gs.materials.Rigid(**material))
    body = scene.add_entity(morph=gs.morphs.MJCF(file=object_xml(shape)), material=gs.materials.Rigid(**material))
    params = [profile["timeconst_s"], profile["dampratio"], 0.9, 0.95, 0.001, 0.5, 2.0]
    for entity in (floor, body):
        for geom in entity.geoms:
            geom.set_sol_params(params.copy())
    try:
        scene.build(n_envs=len(initial))
        body.set_pos(initial[:, :3])
        body.set_quat(initial[:, 3:7])
        # Free-joint rotational DOFs are BODY-local; tracked omega is WORLD-frame.
        local_omega = Rotation.from_quat(initial[:, 3:7], scalar_first=True).inv().apply(initial[:, 10:13])
        body.set_dofs_velocity(np.column_stack((initial[:, 7:10], local_omega)))
        def snapshot():
            return np.column_stack([tensor_to_array(getter()).copy()
                                    for getter in (body.get_pos, body.get_quat, body.get_vel, body.get_ang)])

        samples = [snapshot()]
        steps = int(np.floor((horizon + 1e-10) / sample_dt)) * stride
        for step in range(1, steps + 1):
            scene.step(update_visualizer=False)
            if step % stride == 0:
                samples.append(snapshot())
        effective = {f"{shape.kind}_sol_params": tensor_to_array(body.geoms[0].get_sol_params()).tolist(),
                     "floor_sol_params": tensor_to_array(floor.geoms[0].get_sol_params()).tolist(),
                     "mass_kg": float(body.get_mass().sum().item()),
                     "genesis_version": gs.__version__, "backend": gs.backend.name,
                     "device": str(gs.device), "precision": str(np.dtype(gs.np_float).itemsize * 8),
                     "deterministic_algorithms": bool(gs.use_deterministic_algorithms),
                     "libraries": {name: version(name) for name in ("numpy", "scipy", "torch", "matplotlib")},
                     "solver": {"integrator": "approximate_implicitfast", "constraint_solver": "Newton",
                                "friction_cone": "pyramidal", "iterations": 50, "tolerance": 1e-8}}
        return np.arange(len(samples)) * sample_dt, np.stack(samples, axis=1), effective
    finally:
        scene.destroy()


def simulate(args):
    manifest = read_manifest(args.manifest)
    tracks = read_tracks(args.reference)
    shape, rules = shape_from_manifest(manifest), Rules(**manifest.get("analysis", {}))
    horizon = min(t[-1] for t, x in tracks.values())
    if args.horizon is not None:
        if not np.isfinite(args.horizon) or args.horizon <= 0:
            raise ValueError("Horizon must be positive and finite.")
        horizon = min(horizon, args.horizon)
    if horizon < rules.rest_duration_s:
        raise ValueError("Shared recording window is shorter than rest confirmation duration.")
    for trial, (t, x) in tracks.items():
        if np.diff(t).max() > rules.maximum_tracking_gap_s:
            raise ValueError(f"{trial}: tracking gap exceeds maximum_tracking_gap_s.")
        if clearance(x[:1], shape)[0][0] <= rules.contact_gap_m:
            raise ValueError(f"{trial}: release must start in free flight, without floor contact.")
    if len(set(args.seeds)) != len(args.seeds) or any(s < 0 or s >= 2**32 for s in args.seeds):
        raise ValueError("Use distinct seeds in [0, 2**32).")
    initial = np.stack([x[0] for t, x in tracks.values()])
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False).stdout.strip()
    summaries = []
    for profile in manifest["contact_profiles"]:
        for seed in args.seeds:
            print(f"Replaying {len(initial)} releases: {profile['name']}, seed={seed}, horizon={horizon:g}s", flush=True)
            t, states, effective = run_genesis(initial, shape, profile, args.dt, args.substeps,
                                               horizon, args.sample_dt, seed, args.backend)
            metadata = {"engine": "Genesis", "profile": profile, "seed": seed, "dt_s": args.dt,
                        "substeps": args.substeps, "sample_dt_s": args.sample_dt,
                        "data_kind": manifest["data_kind"], "reference_sha256": file_hash(args.reference),
                        "manifest_sha256": file_hash(args.manifest), "script_sha256": file_hash(__file__),
                        "git_revision": revision, "python": platform.python_version(), **effective}
            path = output / f"{profile['name']}-seed{seed}.npz"
            save_run(path, list(tracks), t, states, metadata)
            group = [metrics(t, x, shape, rules) for x in states]
            summaries.append({"run": path.name, "metadata": metadata, "n": len(group),
                              "outcomes": {k: sum(m["outcome"] == k for m in group) for k in shape.outcomes}})
    (output / "runs.json").write_text(json.dumps(summaries, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Saved {len(summaries)} separate runs. Use plot to compare one profile and seed at a time.")


def drop_check(args):
    """Characterize rebound settings; this does NOT fit them to real measurements."""
    manifest = read_manifest(args.manifest)
    shape, rules = shape_from_manifest(manifest), Rules(**manifest.get("analysis", {}))
    heights = np.asarray(args.heights, dtype=float)
    if not np.isfinite(heights).all() or np.any(heights <= rules.minimum_bounce_clearance_m):
        raise ValueError("Drop heights must be finite and exceed the bounce detection threshold.")
    initial = np.zeros((len(heights), 13))
    initial[:, 2] = heights + shape.half_height
    initial[:, 3] = 1
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    reports = []
    for profile in manifest["contact_profiles"]:
        print(f"Checking flat, non-spinning drops: {profile['name']}", flush=True)
        t, states, effective = run_genesis(initial, shape, profile, args.dt, args.substeps,
                                           args.horizon, args.sample_dt, args.seed, args.backend)
        records = []
        for height, state in zip(heights, states):
            m = metrics(t, state, shape, rules)
            rebound = m["bounce_clearances_m"][0] if m["bounce_clearances_m"] else None
            # This height-ratio estimate applies only to these flat, non-spinning drops.
            records.append({"drop_clearance_m": float(height), "rebound_clearance_m": rebound,
                            "effective_restitution_height_ratio": float(np.sqrt(rebound / height)) if rebound else None,
                            "status": "completed rebound" if rebound else "no complete rebound detected; e unknown",
                            "metrics": m})
        reports.append({"profile": profile, "drops": records, "effective_solver": effective})
        np.savez_compressed(output / f"{profile['name']}-drop-check.npz", t_s=t, states=states)
    metadata = {"purpose": "Simulator response characterization; not a real-material calibration",
                "manifest_sha256": file_hash(args.manifest), "script_sha256": file_hash(__file__),
                "dt_s": args.dt, "substeps": args.substeps, "sample_dt_s": args.sample_dt,
                "seed": args.seed, "analysis": asdict(rules), "profiles": reports}
    (output / "restitution-drop-check.json").write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")


def toy_track(t, shape, theta0, spin, height, speed, bounce_factor, drift, initial_quat=None):
    """Deliberately simplified analytic drawing fixture, NOT a rigid-body solver."""
    g = 9.81
    hit = (speed + np.sqrt(speed**2 + 2 * g * height)) / g
    gap = np.maximum(0, height + speed * t - 0.5 * g * t**2)
    impact_speed = g * hit - speed
    start, v = hit, impact_speed * bounce_factor
    while v * v / (2 * g) > 0.001:
        flight = (t >= start) & (t <= start + 2 * v / g)
        elapsed = t[flight] - start
        gap[flight] = v * elapsed - 0.5 * g * elapsed**2
        start += 2 * v / g
        v *= bounce_factor
    gap[t > start] = 0
    theta = theta0 + spin * t + drift * t**2
    # Synthetic relaxation toward the nearest face; this is not a physical contact law.
    settle_begin = max(hit, start - 0.12)
    target = np.round((theta0 + spin * settle_begin + drift * settle_begin**2) / np.pi) * np.pi
    weight = np.clip((t - settle_begin) / 0.3, 0, 1)
    weight = weight**2 * (3 - 2 * weight)
    if isinstance(shape, Die):
        axis = np.array([1, 0.65, 0.35])
        axis /= np.linalg.norm(axis)
        rotations = Rotation.from_rotvec((spin * t + drift * t**2)[:, None] * axis) * Rotation.from_quat(
            initial_quat, scalar_first=True
        )
        at_contact = rotations[min(np.searchsorted(t, settle_begin), len(t) - 1)]
        face = int(classify(at_contact.as_quat(scalar_first=True), "die", alignment=0)[0])
        normal = at_contact.apply(DIE_NORMALS[face])
        correction_axis = np.cross(normal, [0, 0, 1])
        angle = np.arccos(np.clip(normal[2], -1, 1))
        correction = correction_axis / max(np.linalg.norm(correction_axis), 1e-12) * angle
        target_rotation = Rotation.from_rotvec(correction) * at_contact
        residual = (target_rotation * rotations.inv()).as_rotvec()
        rotations = Rotation.from_rotvec(weight[:, None] * residual) * rotations
        q = rotations.as_quat(scalar_first=True)
        interval_omega = (rotations[1:] * rotations[:-1].inv()).as_rotvec() / np.diff(t)[:, None]
        omega = np.vstack((interval_omega[0], (interval_omega[:-1] + interval_omega[1:]) / 2, interval_omega[-1]))
    else:
        theta = theta * (1 - weight) + target * weight
        q = Rotation.from_euler("x", theta[:, None]).as_quat(scalar_first=True)
        omega = np.column_stack((np.gradient(theta, t, edge_order=2), np.zeros((len(t), 2))))
    support, _ = support_height(q, shape)
    p = np.column_stack((0.015 * np.minimum(t, start), np.zeros_like(t), gap + support))
    v = np.gradient(p, t, axis=0, edge_order=2)
    return np.column_stack((p, q, v, omega))


def demo(args):
    if args.trials < 2:
        raise ValueError("Demo requires at least two tosses.")
    output = Path(args.out or ("out/probability/real-comparison-demo" if args.kind == "coin"
                              else "out/probability/real-comparison-die-demo"))
    output.mkdir(parents=True, exist_ok=True)
    shape = Coin(0.02, 0.045, 0.008) if args.kind == "coin" else Die(0.08, 0.08)
    profile = {"name": "illustrative_contact", "friction": 0.6, "timeconst_s": 0.02,
               "dampratio": 1.0, "rolling_m": 0.0001, "torsional_m": 0.005}
    manifest = {"data_kind": "illustrative", "source": "Synthetic analytic chart fixture; NOT camera measurements",
                "floor_material": "PLACEHOLDER smooth rigid plane; physical material not measured",
                "tracking_method": "Synthetic positions and quaternions; numerical derivatives",
                "kind": shape.kind, shape.kind: asdict(shape), "analysis": asdict(Rules()),
                "contact_profiles": [
                    {**profile, "name": f"mu{mu:g}_d{damping:g}", "friction": mu, "dampratio": damping}
                    for mu in (0.3, 0.6) for damping in (0.5, 1.0, 2.0)
                ]}
    # Keep profile names suitable for filenames (decimal points become 'p').
    for item in manifest["contact_profiles"]:
        item["name"] = item["name"].replace(".", "p")
    t = np.arange(0, 2.5 + 0.0025, 0.005)
    rng = np.random.default_rng(20261111)
    reference, sims = {}, []
    for i in range(args.trials):
        theta0, spin, height, speed = rng.uniform(-np.pi, np.pi), rng.uniform(13, 22), rng.uniform(0.25, 0.45), rng.uniform(0.5, 0.9)
        bounce = rng.uniform(0.25, 0.4)
        initial_quat = Rotation.random(random_state=rng).as_quat(scalar_first=True) if args.kind == "die" else None
        real = toy_track(t, shape, theta0, spin, height, speed, bounce, 0, initial_quat)
        sim = toy_track(t, shape, theta0, spin, height, speed, bounce + 0.07, rng.uniform(-0.7, 0.7), initial_quat)
        sim[0] = real[0]  # Exact same synthetic release measurement.
        reference[f"demo_{i + 1:03d}"] = (t, real)
        sims.append(sim)
    reference_path, manifest_path = output / "reference.csv", output / "experiment.json"
    write_tracks(reference_path, reference)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    run_path = output / "ILLUSTRATIVE-simulation.npz"
    save_run(run_path, list(reference), t, np.array(sims),
             {"engine": "synthetic illustration, NOT Genesis", "profile": profile, "seed": 20261111,
              "reference_sha256": file_hash(reference_path), "manifest_sha256": file_hash(manifest_path)})
    plot_comparison(reference_path, manifest_path, run_path, output / "ILLUSTRATIVE-comparison.png")
    print(f"Created illustrative chart and input templates in {output}. No measured data or Genesis run used.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("demo", help="Synthetic chart and input templates; no Genesis execution")
    p.add_argument("--out", help="Default: separate coin and die example folders under out/probability")
    p.add_argument("--kind", choices=("coin", "die"), default="coin")
    p.add_argument("--trials", type=int, default=100)
    p.set_defaults(func=demo)
    p = commands.add_parser("simulate", help="Genesis replay of every measured release, per contact profile and seed")
    p.add_argument("--reference", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--backend", choices=("cpu", "gpu"), default="cpu")
    p.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33])
    p.add_argument("--dt", type=float, default=0.001)
    p.add_argument("--substeps", type=int, default=2)
    p.add_argument("--sample-dt", type=float, default=0.005)
    p.add_argument("--horizon", type=float, help="Optional cap; always bounded by the shortest real recording")
    p.set_defaults(func=simulate)
    p = commands.add_parser("drop-check", help="Measure each contact profile's effective rebound in flat drops")
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--heights", type=float, nargs="+", default=[0.1, 0.2])
    p.add_argument("--backend", choices=("cpu", "gpu"), default="cpu")
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--dt", type=float, default=0.001)
    p.add_argument("--substeps", type=int, default=2)
    p.add_argument("--sample-dt", type=float, default=0.001)
    p.add_argument("--horizon", type=float, default=2.0)
    p.set_defaults(func=drop_check)
    p = commands.add_parser("plot", help="Plot one profile and seed, with no pooling or extrapolation")
    p.add_argument("--reference", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--out", required=True, help="Output PNG path; SVG and per-trial JSON are also saved")
    p.set_defaults(func=lambda a: plot_comparison(a.reference, a.manifest, a.run, a.out))
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

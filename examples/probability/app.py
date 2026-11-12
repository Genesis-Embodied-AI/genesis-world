"""Run with python -m examples.probability.app -g (or Run-Probability-Lab.cmd)."""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import multiprocessing as mp
import platform
import queue
import subprocess
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import genesis as gs
import numpy as np
from imgui_bundle import hello_imgui, imgui, immapp, immvision, implot
from PIL import Image

from examples.probability.model import Config, Forecast, Results, sample_launch, wilson_interval
from examples.probability.vectors import (
    ANGULAR_COLOR, CONTACT_COLOR, FLOOR_COLOR, INITIAL_ANGULAR_COLOR, INITIAL_LINEAR_COLOR, LINEAR_COLOR,
    NORMAL_COLOR, SLIP_COLOR, SLIP_THRESHOLD, STICK_COLOR, TRAIL_COLOR, Layers,
)
from examples.probability.worker import Command, Frame, select_frame, worker
from examples.probability.world import World


def export_results(results: Results, forecasts: list[Forecast], output: Path, gpu: bool) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    prefix = output / f"{results.config.kind}-{stamp}"
    source = Path(__file__).parent
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=source, capture_output=True, text=True, check=False)
    data = {
        "created_utc": stamp,
        "config": asdict(results.config),
        "backend": "cuda" if gpu else "cpu",
        "python": platform.python_version(),
        "versions": {name: importlib.metadata.version(name) for name in ("genesis-world", "torch", "numpy", "scipy")},
        "genesis_commit": revision.stdout.strip(),
        "app_source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.glob("*.py")
        },
        "labels": results.config.labels,
        "counts": results.counts.tolist(),
        "unresolved": results.unresolved,
        "chi_square_p_exploratory": results.p_value,
        "trials": [asdict(trial) for trial in results.trials],
        "forecasts": [asdict(forecast) for forecast in forecasts],
        "interpretation": "Face frequencies condition on valid settled outcomes. Edge/timeouts are separate. "
                          "Forecasts are Gaussian state-error scenarios, not calibrated real-world probabilities.",
    }
    prefix.with_suffix(".json").write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
    with prefix.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["trial", "seed", "outcome", "seconds", "source"])
        for i, trial in enumerate(results.trials, 1):
            label = results.config.labels[trial.outcome] if trial.outcome >= 0 else "unresolved"
            writer.writerow([i, trial.seed, label, trial.seconds, trial.source])
    return prefix.with_suffix(".json")


class Dashboard:
    def __init__(self, config: Config, gpu: bool, output: Path, smoke: bool):
        self.config = config
        self.gpu, self.output, self.smoke = gpu, output, smoke
        self.results = Results(config)
        self.forecasts: list[Forecast] = []
        self.current_forecasts: list[Forecast] = []
        self.frame: Frame | None = None
        self.terminal_frame: Frame | None = None
        self.experiment_id = 0
        self.roll_id = 0
        self.smoke_motion_during_forecast = False
        self.texture = None
        self.busy = True
        self.forecasting = False
        self.failed = False
        self.status = "Starting Genesis. The first launch compiles kernels."
        self.auto_forecast = True
        self.layers = Layers()
        self.graph_height = 360
        self.expanded_graphs = False
        self.fit_graphs = True
        self.batch_count = 128
        self.kind_index = 0 if config.kind == "coin" else 1
        self.protocol_index = 0 if config.protocol == "isotropic" else 1
        self.seed_text = str(config.seed)
        self.height, self.spin, self.friction = config.height, config.spin, config.friction
        self.saved_path = ""
        self.roll_seed = None
        self.smoke_stage = 0
        self.smoke_finished_at = None
        self.smoke_paused_at = None
        self.started = time.monotonic()
        context = mp.get_context("spawn")
        self.commands, self.events, self.frames = context.Queue(), context.Queue(), context.Queue(maxsize=2)
        self.process = context.Process(target=worker, args=(config, gpu, self.commands, self.events, self.frames))
        self.process.start()
        self.params = hello_imgui.RunnerParams()
        self.params.app_window_params.window_title = "Genesis | Probability Lab - Continuous motion"
        if smoke:
            self.params.app_window_params.window_title += " (automated check)"
        self.params.app_window_params.window_geometry.size = (1360, 900)
        self.params.ini_disable = True
        self.params.fps_idling.fps_idle = 30
        self.params.callbacks.show_gui = self.draw
        self.params.callbacks.post_init = self.style

    def style(self):
        imgui.style_colors_dark()
        style = imgui.get_style()
        style.window_padding = (18, 14)
        style.frame_padding = (9, 6)
        style.item_spacing = (9, 7)
        style.frame_rounding = 5
        style.child_rounding = 8

    def send(self, command: Command):
        if command.action in ("configure", "roll"):
            if command.action == "configure":
                self.experiment_id += 1
                self.roll_id = 0
            else:
                self.roll_id += 1
            self.current_forecasts = []
            self.frame = self.terminal_frame = None
            self.roll_seed = None
            self.forecasting = False
        if command.action in ("configure", "batch"):
            self.busy = True
        self.commands.put(replace(command, experiment_id=self.experiment_id, roll_id=self.roll_id))

    def save(self):
        self.saved_path = str(export_results(self.results, self.forecasts, self.output, self.gpu))

    def apply_settings(self):
        try:
            seed = int(self.seed_text)
            if not 0 <= seed < 2**32:
                raise ValueError("seed outside uint32 range")
        except ValueError:
            self.status = "Seed must be an integer from 0 to 4294967295. No settings were changed."
            return
        if self.results.trials or self.forecasts:
            self.save()
        self.config = replace(self.config, kind=("coin", "die")[self.kind_index],
                              protocol=("isotropic", "face_up")[self.protocol_index], seed=seed,
                              height=self.height, spin=self.spin, friction=self.friction)
        self.results = Results(self.config)
        self.forecasts = []
        self.current_forecasts = []
        self.roll_seed = None
        self.frame = None
        self.terminal_frame = None
        self.send(Command("configure", config=self.config))

    def receive_events(self):
        while True:
            try:
                event = self.events.get_nowait()
            except queue.Empty:
                break
            if event.kind == "error":
                self.failed = True
                self.busy = False
                self.status = event.message
                continue
            if event.experiment_id != self.experiment_id:
                continue
            if event.kind == "trials":
                self.results.trials.extend(event.trials)
            elif event.kind == "forecast":
                self.forecasts.append(event.forecast)
            if event.roll_id != self.roll_id:
                continue
            if event.kind == "busy":
                self.busy = True
            elif event.kind == "ready":
                self.busy = False
            elif event.kind == "forecasting":
                self.forecasting = True
            elif event.kind in ("forecast_idle", "forecast_cancelled"):
                self.forecasting = False
            elif event.kind == "forecast":
                self.current_forecasts.append(event.forecast)
            elif event.kind == "roll":
                self.roll_seed = event.seed
            elif event.kind == "terminal":
                self.terminal_frame = event.frame
            if event.message:
                self.status = event.message
                print(f"{event.kind}: {event.message}", flush=True)

    def poll(self):
        self.receive_events()
        newest = select_frame(self.frame, self.terminal_frame, self.experiment_id, self.roll_id)
        self.terminal_frame = None
        while True:
            try:
                candidate = self.frames.get_nowait()
            except queue.Empty:
                break
            newest = select_frame(newest, candidate, self.experiment_id, self.roll_id)
        if newest is not None and newest is not self.frame:
            if (self.smoke and self.forecasting and self.frame is not None
                    and newest.seconds > self.frame.seconds):
                self.smoke_motion_during_forecast = True
            self.frame = newest
            if self.texture is None:
                self.texture = immvision.GlTexture(newest.rgb)
            else:
                self.texture.update_from_image(newest.rgb)
        if not self.process.is_alive() and not self.failed:
            self.failed = True
            self.status = f"Physics worker stopped (exit {self.process.exitcode}). See the launcher log."

    def controls(self):
        imgui.separator_text("EXPERIMENT")
        imgui.begin_disabled(self.busy or self.failed or self.smoke)
        imgui.set_next_item_width(175)
        _, self.kind_index = imgui.combo("Object", self.kind_index, ["Coin: H / T", "Die: 1 to 6"])
        imgui.same_line()
        imgui.set_next_item_width(240)
        _, self.protocol_index = imgui.combo("Launch", self.protocol_index,
                                            ["Random orientation (symmetric)", "Heads / face 1 starts up"])
        imgui.set_next_item_width(175)
        _, self.seed_text = imgui.input_text("Seed", self.seed_text)
        imgui.same_line()
        if imgui.button("Apply / reset experiment"):
            self.apply_settings()
        expanded = imgui.collapsing_header("Physical settings (apply to start a new experiment)")
        if expanded:
            _, self.height = imgui.slider_float("Launch height (m)", self.height, 0.1, 1.5)
            _, self.spin = imgui.slider_float("Spin magnitude (rad/s)", self.spin, 0, 50)
            _, self.friction = imgui.slider_float("Sliding friction", self.friction, 0.01, 2)
            imgui.text_wrapped(f"Gravity 9.81 m/s^2; fixed dt {self.config.dt} s, two substeps. "
                               "Settings do not force a fair outcome.")
        imgui.end_disabled()
        if self.config.protocol == "face_up":
            imgui.text_colored((1, .75, .3, 1), "Face-up launch breaks the symmetric reference assumptions.")

    def vector_controls(self):
        imgui.separator_text("LIVE VECTORS / LEGEND")
        entries = (
            ("floor_normals", "Floor normals", FLOOR_COLOR, self.layers.floor_normals),
            ("initial_pose", "Start ghost + RGB axes", TRAIL_COLOR, self.layers.initial_pose),
            ("trail", "COM trail", TRAIL_COLOR, self.layers.trail),
            ("initial_linear", "Initial v (m/s)", INITIAL_LINEAR_COLOR, self.layers.initial_linear),
            ("initial_angular", "Initial omega (rad/s)", INITIAL_ANGULAR_COLOR, self.layers.initial_angular),
            ("contact_points", "Contact points", CONTACT_COLOR, self.layers.contact_points),
            ("linear", "Current v (m/s)", LINEAR_COLOR, self.layers.linear),
            ("angular", "Current omega (rad/s)", ANGULAR_COLOR, self.layers.angular),
            ("normal_forces", "Normal forces (N)", NORMAL_COLOR, self.layers.normal_forces),
            ("friction_forces", "Friction forces (N)", SLIP_COLOR, self.layers.friction_forces),
        )
        changed_layers = False
        if imgui.begin_table("vector_layers", 3, imgui.TableFlags_.sizing_stretch_same):
            for i, (field, label, color, value) in enumerate(entries):
                if i % 3 == 0:
                    imgui.table_next_row()
                imgui.table_set_column_index(i % 3)
                imgui.push_style_color(imgui.Col_.text, color[:3] + (1.,))
                changed, enabled = imgui.checkbox(label, value)
                imgui.pop_style_color()
                if changed:
                    self.layers = replace(self.layers, **{field: enabled})
                    changed_layers = True
            imgui.end_table()
        imgui.text_colored(STICK_COLOR, "Green: sticking*  |  Orange: slipping*")
        imgui.text_wrapped(f"*Measured tangential contact speed threshold: {SLIP_THRESHOLD:g} m/s; "
                           "a diagnostic, not a solver mode flag. Axes: X red, Y green, Z blue.")
        imgui.text(f"Scales: v {self.layers.velocity_scale:g} m per m/s; "
                   f"omega {self.layers.angular_scale:g} m per rad/s; F {self.layers.force_scale:g} m per N.")
        imgui.text_disabled(f"Normals are unit directions drawn 0.08 m long. Arrow length capped at "
                            f"{self.layers.arrow_limit:g} m; numeric values are never capped.")
        if imgui.collapsing_header("Adjust vector scales"):
            for field, label, value, lower, upper in (
                ("velocity_scale", "Velocity scale (m per m/s)", self.layers.velocity_scale, .02, .5),
                ("angular_scale", "Spin scale (m per rad/s)", self.layers.angular_scale, .002, .05),
                ("force_scale", "Force scale (m per N)", self.layers.force_scale, .02, 2.),
            ):
                changed, value = imgui.slider_float(label, value, lower, upper)
                if changed:
                    self.layers = replace(self.layers, **{field: value})
                    changed_layers = True
        if changed_layers:
            self.send(Command("layers", layers=self.layers))

    def physical_readout(self):
        if self.frame is None:
            return
        data = self.frame.diagnostics
        imgui.separator_text("MEASURED PHYSICS / WORLD FRAME")
        imgui.text(f"Floor: plane  |  normal ({data.floor_normal[0]:.2f}, {data.floor_normal[1]:.2f}, "
                   f"{data.floor_normal[2]:.2f})")
        imgui.text_disabled("Checker squares: 0.10 m. Color texture has no height or roughness geometry.")
        imgui.text(f"Static / kinetic friction: shared mu = {data.friction:.3f} (dimensionless)")
        imgui.text_disabled(f"Floor mu {data.floor_friction:.3f}; object mu {data.entity_friction:.3f}. "
                            "Genesis has no separate mu_s / mu_k settings here.")
        imgui.text_colored(LINEAR_COLOR, f"v = ({data.linear[0]:+.3f}, {data.linear[1]:+.3f}, "
                          f"{data.linear[2]:+.3f}) m/s   |v| = {np.linalg.norm(data.linear):.3f}")
        imgui.text_colored(ANGULAR_COLOR, f"omega = ({data.angular[0]:+.2f}, {data.angular[1]:+.2f}, "
                          f"{data.angular[2]:+.2f}) rad/s   |omega| = {np.linalg.norm(data.angular):.3f}")
        if imgui.collapsing_header("Starting pose and velocity (frozen for this throw)"):
            p, q = data.initial_center, data.initial_quaternion
            imgui.text(f"COM0 = ({p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f}) m")
            imgui.text(f"q0 [w,x,y,z] = ({q[0]:.4f}, {q[1]:.4f}, {q[2]:.4f}, {q[3]:.4f})")
            for label, value, unit in (("v0", data.initial_linear, "m/s"),
                                       ("omega0", data.initial_angular, "rad/s")):
                imgui.text(f"{label} = ({value[0]:+.3f}, {value[1]:+.3f}, {value[2]:+.3f}) {unit} "
                           f" |{label}| = {np.linalg.norm(value):.3f}")
        imgui.text(f"Floor contacts: {len(data.contacts)}  |  sample step {data.contact_time:.2f} s")
        imgui.text_disabled("Forces/points: latest solver substep. Velocities: current frame state.")
        if not data.contacts:
            imgui.text_disabled("No floor contact: no contact-force arrows.")
        elif imgui.begin_table("contact_readout", 5, imgui.TableFlags_.row_bg | imgui.TableFlags_.borders_inner_h):
            for heading in ("Point xyz (m)", "|Fn| (N)", "|Ft| (N)", "Slip (m/s)", "State*"):
                imgui.table_setup_column(heading)
            imgui.table_headers_row()
            for contact in data.contacts:
                p = contact.position
                values = (f"{p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}",
                          f"{np.linalg.norm(contact.normal_force):.4f}",
                          f"{np.linalg.norm(contact.friction_force):.4f}",
                          f"{np.linalg.norm(contact.slip_velocity):.4f}", contact.regime)
                imgui.table_next_row()
                for col, value in enumerate(values):
                    imgui.table_set_column_index(col)
                    imgui.text(value)
            imgui.end_table()
        force_max = max((max(np.linalg.norm(c.normal_force), np.linalg.norm(c.friction_force))
                         for c in data.contacts), default=0)
        if force_max * self.layers.force_scale > self.layers.arrow_limit:
            imgui.text_colored(SLIP_COLOR, "Force arrows are capped; see full magnitudes in the table.")

    def simulation(self, width: float):
        imgui.text_colored((.35, .85, .84, 1), "01  /  PHYSICS WORLD")
        imgui.same_line()
        imgui.text_disabled("CUDA" if self.gpu else "CPU")
        imgui.text("COIN FLIP" if self.config.kind == "coin" else "DIE ROLL")
        if self.frame is not None:
            imgui.same_line()
            imgui.text_disabled(f"{self.frame.phase}  |  t = {self.frame.seconds:.2f} s")
        if self.texture is not None and self.frame is not None:
            size = (width - 25, (width - 25) * self.frame.rgb.shape[0] / self.frame.rgb.shape[1])
            imgui.image(imgui.ImTextureRef(self.texture.texture_id), size)
            speed = np.linalg.norm(self.frame.velocity[:3])
            spin = np.linalg.norm(self.frame.velocity[3:])
            imgui.text(f"Height {self.frame.position[2]:.3f} m    Speed {speed:.2f} m/s    Spin {spin:.2f} rad/s")
            contacts = self.frame.diagnostics.contacts
            normal = np.sum([c.normal_force for c in contacts], axis=0) if contacts else np.zeros(3)
            tangent = np.sum([c.friction_force for c in contacts], axis=0) if contacts else np.zeros(3)
            imgui.text(f"{len(contacts)} contacts   |sum Fn| {np.linalg.norm(normal):.4f} N   "
                       f"|sum Ft| {np.linalg.norm(tangent):.4f} N   mu {self.frame.diagnostics.friction:.3f}")
        else:
            imgui.dummy((width - 25, 290))
            imgui.text_wrapped(self.status)
        imgui.begin_disabled(self.busy or self.failed or self.smoke)
        if imgui.button("Flip coin" if self.config.kind == "coin" else "Roll die"):
            self.send(Command("roll", auto_forecast=self.auto_forecast))
        imgui.same_line()
        if imgui.button("Pause / resume"):
            self.send(Command("pause"))
        imgui.same_line()
        if imgui.button("Step"):
            self.send(Command("step"))
        imgui.same_line()
        if imgui.button("Forecast now"):
            self.send(Command("forecast"))
        imgui.same_line()
        if imgui.button("Zoom in"):
            self.send(Command("camera", amount=1))
        imgui.same_line()
        if imgui.button("Zoom out"):
            self.send(Command("camera", amount=0))
        _, self.auto_forecast = imgui.checkbox("Auto forecasts near 0.00, 0.30 and 0.65 seconds", self.auto_forecast)
        imgui.set_next_item_width(120)
        _, self.batch_count = imgui.input_int("Trials", self.batch_count)
        self.batch_count = max(1, min(self.batch_count, 10000))
        imgui.same_line()
        if imgui.button("Run batch"):
            self.send(Command("batch", amount=self.batch_count))
        imgui.same_line()
        if imgui.button("Export JSON + CSV"):
            self.save()
        imgui.end_disabled()
        if not self.smoke and (self.busy or self.forecasting) and imgui.button("Cancel calculation"):
            self.send(Command("cancel"))
        self.vector_controls()
        if imgui.collapsing_header("Force magnitudes, floor and starting-pose details"):
            self.physical_readout()
        if imgui.collapsing_header("Experiment settings / reset"):
            self.controls()
        if self.saved_path:
            imgui.text_wrapped(f"Saved: {self.saved_path}")
        imgui.text_disabled("Close this window to stop. Completed trials are saved automatically.")

    def forecast_plot(self, height: float):
        labels = self.config.labels
        reference = 1 / len(labels)
        if implot.begin_plot("Forecast over simulated time", (-1, height)):
            implot.setup_axes("Observation time (s)", "P(face)")
            implot.setup_legend(implot.Location_.north, implot.LegendFlags_.horizontal | implot.LegendFlags_.outside)
            max_time = max(1.0, max((forecast.at_time for forecast in self.current_forecasts), default=0.0))
            if self.fit_graphs:
                implot.setup_axes_limits(0, max_time, 0, 1, imgui.Cond_.always)
            if self.current_forecasts:
                x = np.array([f.at_time for f in self.current_forecasts], dtype=float)
                spec = implot.Spec(line_weight=2.5, marker=implot.Marker_.circle)
                for i, label in enumerate(labels):
                    implot.plot_line(label, x, np.array([f.probabilities[i] for f in self.current_forecasts]), spec)
                implot.plot_line("Unresolved", x,
                                 np.array([f.unresolved / f.samples for f in self.current_forecasts]), spec)
            implot.plot_line("Fair", np.array([0., max_time]), np.array([reference, reference]))
            implot.end_plot()

    def frequency_plot(self, height: float):
        labels = self.config.labels
        reference = 1 / len(labels)
        total = int(self.results.counts.sum())
        if implot.begin_plot("Frequency across completed trials", (-1, height)):
            implot.setup_axes("Valid settled trials", "Frequency")
            implot.setup_legend(implot.Location_.north, implot.LegendFlags_.horizontal | implot.LegendFlags_.outside)
            if self.fit_graphs:
                implot.setup_axes_limits(1, max(10, total), 0, 1, imgui.Cond_.always)
            outcomes = np.array([trial.outcome for trial in self.results.trials if trial.outcome >= 0])
            if total:
                x = np.arange(1, total + 1, dtype=float)
                for i, label in enumerate(labels):
                    implot.plot_line(label, x, np.cumsum(outcomes == i) / x, implot.Spec(line_weight=2.5))
            implot.plot_line("Fair", np.array([1., float(max(10, total))]), np.array([reference, reference]))
            implot.end_plot()

    def probabilities(self):
        labels = self.config.labels
        reference = 1 / len(labels)
        imgui.text_colored((1, .77, .35, 1), "02  /  PROBABILITY")
        imgui.text(f"Fair reference: {reference:.2%} per face")
        imgui.text_wrapped("A reference to test, not a rule imposed on the physics.")
        if imgui.button("Enlarge graphs"):
            self.expanded_graphs = True
        imgui.same_line()
        _, self.fit_graphs = imgui.checkbox("Auto-fit axes", self.fit_graphs)
        imgui.set_next_item_width(180)
        _, self.graph_height = imgui.slider_int("Minimum chart height", self.graph_height, 280, 600)
        chart_height = max(self.graph_height, imgui.get_window_height() * 0.29)
        imgui.separator_text("CURRENT-ROLL FORECAST")
        if self.current_forecasts:
            forecast = self.current_forecasts[-1]
            imgui.text(f"Observation t={forecast.at_time:.2f}s  |  {forecast.samples} futures")
            columns = min(3, len(labels))
            if imgui.begin_table("forecast_faces", columns, imgui.TableFlags_.sizing_stretch_same):
                for i, (label, probability) in enumerate(zip(labels, forecast.probabilities)):
                    if i % columns == 0:
                        imgui.table_next_row()
                    imgui.table_set_column_index(i % columns)
                    imgui.progress_bar(probability, (-1, 18), f"{label}  {probability:.1%}")
                imgui.end_table()
            unresolved_p = forecast.unresolved / forecast.samples
            imgui.progress_bar(unresolved_p, (-1, 18), f"Unresolved  {unresolved_p:.1%}")
        else:
            imgui.text_wrapped("Flip / roll to calculate possible futures from the observed physical state.")
        imgui.text_disabled(f"State error: {self.config.position_sigma * 1000:g} mm, "
                            f"{np.degrees(self.config.angle_sigma):.2f} deg, "
                            f"{self.config.velocity_sigma:g} m/s, {self.config.spin_sigma:g} rad/s")
        self.forecast_plot(chart_height)
        imgui.separator_text("OBSERVED OUTCOMES")
        counts = self.results.counts
        total = int(counts.sum())
        imgui.text(f"{len(self.results.trials)} trials  |  {total} valid  |  {self.results.unresolved} unresolved")
        if imgui.begin_table("frequencies", 4, imgui.TableFlags_.row_bg | imgui.TableFlags_.borders_inner_h):
            for heading in ("Face", "Count", "Frequency", "95% interval"):
                imgui.table_setup_column(heading)
            imgui.table_headers_row()
            for label, count in zip(labels, counts):
                low, high = wilson_interval(int(count), total)
                values = (label, str(count), f"{count / total:.1%}" if total else "--",
                          f"{low:.1%} - {high:.1%}" if total else "--")
                imgui.table_next_row()
                for i, value in enumerate(values):
                    imgui.table_set_column_index(i)
                    imgui.text(value)
            imgui.end_table()
        self.frequency_plot(chart_height)
        p = self.results.p_value
        imgui.text(f"Chi-square p = {p:.3f} (exploratory)" if p is not None
                   else "More valid trials needed for chi-square.")
        imgui.text_wrapped("Frequencies condition on valid faces. Intervals are per face, not simultaneous. "
                           "A large p-value does not prove fairness.")
        imgui.text_wrapped("Forecasts assume Gaussian measurement errors. They are model estimates; real-world "
                           "calibration has not been established.")

    def draw(self):
        self.poll()
        imgui.text_colored((.93, .94, .96, 1), "GENESIS  /  PROBABILITY LAB")
        imgui.same_line()
        imgui.text_disabled("Physics -> observations -> probabilities")
        imgui.text_wrapped(self.status)
        if self.smoke:
            imgui.text_disabled("Automated validation is running; this window closes when it finishes.")
        imgui.separator()
        width = imgui.get_content_region_avail().x
        left = width * .56
        imgui.begin_child("Simulation", (left, 0), imgui.ChildFlags_.borders)
        self.simulation(left - 20)
        imgui.end_child()
        imgui.same_line()
        imgui.begin_child("Data", (0, 0), imgui.ChildFlags_.borders)
        self.probabilities()
        imgui.end_child()
        if self.expanded_graphs:
            imgui.set_next_window_size((1200, 900), imgui.Cond_.first_use_ever)
            imgui.set_next_window_bg_alpha(1.0)
            visible, self.expanded_graphs = imgui.begin("Expanded probability graphs", self.expanded_graphs)
            if visible:
                imgui.text("Resize this window. Turn off Auto-fit axes to pan and zoom the plots.")
                _, self.fit_graphs = imgui.checkbox("Auto-fit axes##expanded", self.fit_graphs)
                height = max(260, (imgui.get_content_region_avail().y - 20) / 2)
                self.forecast_plot(height)
                self.frequency_plot(height)
            imgui.end()
        if self.smoke:
            if self.failed or time.monotonic() - self.started > 360:
                self.failed = True
                self.params.app_shall_exit = True
            elif not self.busy and self.frame is not None:
                if self.smoke_stage == 0:
                    self.send(Command("roll"))
                    self.smoke_stage = 1
                elif self.smoke_stage == 1 and self.frame.seconds >= 0.09:
                    self.send(Command("pause"))
                    self.smoke_stage = 2
                elif self.smoke_stage == 2 and self.frame.phase == "Paused":
                    self.smoke_paused_at = self.frame.seconds
                    self.send(Command("step"))
                    self.smoke_stage = 3
                elif self.smoke_stage == 3 and self.frame.seconds > self.smoke_paused_at:
                    if not np.isclose(self.frame.seconds - self.smoke_paused_at, self.config.dt):
                        raise RuntimeError("Single step did not advance by exactly one physics timestep.")
                    self.send(Command("forecast"))
                    self.smoke_stage = 4
                elif self.smoke_stage == 4:
                    self.send(Command("pause"))
                    self.smoke_stage = 5
                elif self.smoke_stage == 5 and len(self.results.trials) == 1 and not self.forecasting:
                    self.send(Command("batch", amount=64))
                    self.smoke_stage = 6
                elif self.smoke_stage == 6 and len(self.results.trials) == 65:
                    self.save()
                    self.smoke_finished_at = time.monotonic()
                    self.smoke_stage = 7
                elif self.smoke_stage == 7 and time.monotonic() - self.smoke_finished_at > 2:
                    self.params.app_shall_exit = True

    def run(self):
        addons = immapp.AddOnsParams()
        addons.with_implot = True
        try:
            immapp.run(self.params, addons)
        finally:
            self.commands.put(Command("stop"))
            # Drain while joining: multiprocessing cannot flush a final image to an unread pipe.
            deadline = time.monotonic() + 5
            while self.process.is_alive() and time.monotonic() < deadline:
                self.receive_events()
                while True:
                    try:
                        self.frames.get_nowait()
                    except queue.Empty:
                        break
                self.process.join(timeout=0.02)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=5)
            self.receive_events()
            if self.results.trials or self.forecasts:
                self.save()
        if self.smoke:
            screenshot = hello_imgui.final_app_window_screenshot()
            self.output.mkdir(parents=True, exist_ok=True)
            Image.fromarray(screenshot).save(self.output / f"{self.config.kind}-dashboard.png")
            if (self.failed or self.smoke_stage != 7 or len(self.forecasts) < 4
                    or not self.smoke_motion_during_forecast or self.process.exitcode != 0):
                raise RuntimeError(f"GUI smoke incomplete (worker exit {self.process.exitcode}): {self.status}")
            print(f"GUI smoke passed: {len(self.results.trials)} trials, "
                  f"{len(self.forecasts)} forecasts; motion continued during forecasts; {self.saved_path}")


def main():
    parser = argparse.ArgumentParser(description="Genesis coin and die probability laboratory")
    parser.add_argument("-g", "--gpu", action="store_true", help="Use CUDA instead of CPU.")
    parser.add_argument("--kind", choices=("coin", "die"), default="coin", help="Object to simulate.")
    parser.add_argument("--seed", type=int, default=2026, help="Experiment random seed.")
    parser.add_argument("--batch", type=int, default=0, help="Run this many trials without opening the interface.")
    parser.add_argument("--dt", type=float, default=0.01, help="Physics step in seconds (two substeps).")
    parser.add_argument("--smoke", action="store_true",
                        help="Exercise the real GUI, roll, forecasts and 64 trials, then exit.")
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("out/probability"),
                        help="Experiment export directory.")
    args = parser.parse_args()
    config = Config(kind=args.kind, seed=args.seed, dt=args.dt)
    if not 0 <= args.batch <= 10000:
        parser.error("--batch must be between 0 and 10000")
    if args.batch:
        gs.init(backend=gs.cuda if args.gpu else gs.cpu, seed=config.seed, logging_level="warning")
        world = World(config, config.forecast_samples)
        results = Results(config)
        rng = np.random.default_rng(config.seed)
        try:
            while len(results.trials) < args.batch:
                n = min(config.forecast_samples, args.batch - len(results.trials))
                seeds = rng.integers(0, 2**32, size=n).tolist()
                padded = seeds + [0] * (config.forecast_samples - n)
                results.trials.extend(world.run(sample_launch(config, padded), padded)[:n])
            path = export_results(results, [], args.output_dir, args.gpu)
            print(json.dumps({"counts": results.counts.tolist(), "unresolved": results.unresolved,
                              "chi_square_p": results.p_value, "report": str(path)}))
        finally:
            world.close()
            gs.destroy()
    else:
        Dashboard(config, args.gpu, args.output_dir, args.smoke).run()


if __name__ == "__main__":
    main()

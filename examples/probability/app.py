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
from examples.probability.worker import Command, Frame, worker
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
        self.texture = None
        self.busy = True
        self.failed = False
        self.status = "Starting Genesis. The first launch compiles kernels."
        self.auto_forecast = True
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
        self.params.app_window_params.window_title = "Genesis | Probability Lab"
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
        if command.action in ("configure", "forecast", "batch"):
            self.busy = True
        self.commands.put(command)

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
            if event.kind == "busy":
                self.busy = True
            elif event.kind == "ready":
                self.busy = False
            elif event.kind == "error":
                self.failed = True
                self.busy = False
            elif event.kind == "trials":
                self.results.trials.extend(event.trials)
            elif event.kind == "forecast":
                self.forecasts.append(event.forecast)
                self.current_forecasts.append(event.forecast)
            elif event.kind == "roll":
                self.roll_seed = event.seed
                self.current_forecasts = []
                self.terminal_frame = None
                self.frame = None
            elif event.kind == "terminal":
                self.terminal_frame = event.frame
            if event.message:
                self.status = event.message
                print(f"{event.kind}: {event.message}", flush=True)

    def poll(self):
        self.receive_events()
        newest = None
        while True:
            try:
                newest = self.frames.get_nowait()
            except queue.Empty:
                break
        if self.terminal_frame is not None and (newest is None or newest.seconds <= self.terminal_frame.seconds):
            newest = self.terminal_frame
        self.terminal_frame = None
        expected_seed = self.config.seed if self.roll_seed is None else self.roll_seed
        if newest is not None and (newest.seed != expected_seed or newest.kind != self.config.kind):
            newest = None
        if newest is not None and self.frame is not None and newest.seconds < self.frame.seconds:
            newest = None
        if newest is not None:
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
        imgui.begin_disabled(self.busy or self.failed)
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
        else:
            imgui.dummy((width - 25, 290))
            imgui.text_wrapped(self.status)
        imgui.begin_disabled(self.busy or self.failed)
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
        if self.busy and imgui.button("Cancel calculation"):
            self.send(Command("cancel"))
        self.controls()
        if self.saved_path:
            imgui.text_wrapped(f"Saved: {self.saved_path}")
        imgui.text_disabled("Close this window to stop. Completed trials are saved automatically.")

    def probabilities(self):
        labels = self.config.labels
        reference = 1 / len(labels)
        imgui.text_colored((1, .77, .35, 1), "02  /  PROBABILITY")
        imgui.text(f"Fair reference: {reference:.2%} per face")
        imgui.text_wrapped("A reference to test, not a rule imposed on the physics.")
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
        if implot.begin_plot("Forecast over simulated time", (-1, 260)):
            implot.setup_axes("Observation time (s)", "P(face)")
            implot.setup_legend(implot.Location_.north, implot.LegendFlags_.horizontal | implot.LegendFlags_.outside)
            max_time = max(1.0, self.frame.seconds if self.frame is not None else 1.0)
            implot.setup_axes_limits(0, max_time, 0, 1, imgui.Cond_.always)
            if self.current_forecasts:
                x = np.array([f.at_time for f in self.current_forecasts], dtype=float)
                for i, label in enumerate(labels):
                    implot.plot_line(label, x, np.array([f.probabilities[i] for f in self.current_forecasts]))
            implot.plot_line("Fair", np.array([0., max_time]), np.array([reference, reference]))
            implot.end_plot()
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
        if implot.begin_plot("Frequency across completed trials", (-1, 260)):
            implot.setup_axes("Valid settled trials", "Frequency")
            implot.setup_legend(implot.Location_.north, implot.LegendFlags_.horizontal | implot.LegendFlags_.outside)
            implot.setup_axes_limits(1, max(10, total), 0, 1, imgui.Cond_.always)
            outcomes = np.array([trial.outcome for trial in self.results.trials if trial.outcome >= 0])
            if total:
                x = np.arange(1, total + 1, dtype=float)
                for i, label in enumerate(labels):
                    implot.plot_line(label, x, np.cumsum(outcomes == i) / x)
            implot.plot_line("Fair", np.array([1., float(max(10, total))]), np.array([reference, reference]))
            implot.end_plot()
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
        imgui.separator()
        width = imgui.get_content_region_avail().x
        left = width * .64
        imgui.begin_child("Simulation", (left, 0), imgui.ChildFlags_.borders)
        self.simulation(left - 20)
        imgui.end_child()
        imgui.same_line()
        imgui.begin_child("Data", (0, 0), imgui.ChildFlags_.borders)
        self.probabilities()
        imgui.end_child()
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
                elif self.smoke_stage == 5 and len(self.results.trials) == 1:
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
            if self.failed or self.smoke_stage != 7 or len(self.forecasts) < 4 or self.process.exitcode != 0:
                raise RuntimeError(f"GUI smoke incomplete (worker exit {self.process.exitcode}): {self.status}")
            print(f"GUI smoke passed: {len(self.results.trials)} trials, "
                  f"{len(self.forecasts)} forecasts; {self.saved_path}")


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

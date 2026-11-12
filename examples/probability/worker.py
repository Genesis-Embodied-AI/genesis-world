"""A separate process keeps native rendering and physics off the UI thread."""

import queue
import time
import traceback
from collections import deque
from collections.abc import Generator
from dataclasses import dataclass, replace

import genesis as gs
import numpy as np

from examples.probability.model import Config, Forecast, State, Trial, perturb_observation, sample_launch
from examples.probability.vectors import Diagnostics, Layers
from examples.probability.world import World


@dataclass
class Command:
    action: str
    config: Config | None = None
    amount: int = 0
    auto_forecast: bool = True
    layers: Layers | None = None
    experiment_id: int = 0
    roll_id: int = 0


@dataclass
class Event:
    kind: str
    message: str = ""
    trials: list[Trial] | None = None
    forecast: Forecast | None = None
    seed: int = 0
    frame: "Frame | None" = None
    experiment_id: int = 0
    roll_id: int = 0


@dataclass
class Frame:
    rgb: np.ndarray
    seconds: float
    position: list[float]
    velocity: list[float]
    phase: str
    seed: int
    kind: str
    diagnostics: Diagnostics
    experiment_id: int = 0
    roll_id: int = 0
    sequence: int = 0


def select_frame(current: Frame | None, candidate: Frame | None, experiment_id: int, roll_id: int) -> Frame | None:
    """Merge both transport queues without accepting old runs or rewinding equal-time updates."""
    if current is not None and (current.experiment_id, current.roll_id) != (experiment_id, roll_id):
        current = None
    if candidate is not None and (candidate.experiment_id, candidate.roll_id) == (experiment_id, roll_id):
        if current is None or candidate.sequence > current.sequence:
            return candidate
    return current


class Engine:
    def __init__(self, config: Config, commands, events, frames):
        self.commands, self.events, self.frames = commands, events, frames
        self.world: World | None = None
        self.shadow: World | None = None
        self.stopping = False
        self.pending: list[Command] = []
        self.forecasts: deque[Generator[float, None, None]] = deque()
        self.experiment_id = 0
        self.roll_id = 0
        self.frame_sequence = 0
        self.layers = Layers()
        self.configure(config)

    def emit(self, event: Event):
        self.events.put(replace(event, experiment_id=self.experiment_id, roll_id=self.roll_id))

    def configure(self, config: Config):
        self.cancel_forecasts()
        self.emit(Event("busy", "Building the physics world (first launch compiles GPU kernels)..."))
        if self.world is not None:
            self.world.close()
        if self.shadow is not None:
            self.shadow.close()
        self.shadow = None
        self.config = config
        self.seed = config.seed
        self.launch_rng = np.random.default_rng(config.seed)
        self.forecast_rng = np.random.default_rng(config.seed ^ 0xA5A5A5A5)
        self.world = World(config, 1, render=True)
        self.initial = sample_launch(config, [config.seed])
        self.world.reset(self.initial)
        self.world.step()
        self.world.reset(self.initial)
        self.phase = "Ready"
        self.rolling = False
        self.active = False
        self.paused = False
        self.auto_forecast = True
        self.next_checkpoint = 0
        self.camera_offset = np.array([0.7, -0.9, 0.6])
        self.publish_frame()
        self.emit(Event("busy", "Preparing the forecast world before the first toss..."))
        self.shadow = World(config, config.forecast_samples)
        self.shadow.reset(perturb_observation(config, self.initial, config.seed))
        self.shadow.step()
        self.emit(Event("ready", "Ready. Start a throw or run a batch."))

    def publish_frame(self, terminal: bool = False) -> Frame:
        state = self.world.state()
        diagnostics = self.world.diagnostics()
        target = diagnostics.center
        if self.layers.initial_pose or self.layers.initial_linear or self.layers.initial_angular:
            target = (diagnostics.center + diagnostics.initial_center) / 2
        separation = np.linalg.norm(diagnostics.center - target)
        offset = self.camera_offset * max(1.0, separation / 0.3)
        self.world.camera.set_pose(pos=target + offset, lookat=target)
        self.frame_sequence += 1
        frame = Frame(self.world.image(diagnostics, self.layers), self.world.elapsed, state.position[0].tolist(),
                      state.velocity[0].tolist(), self.phase, self.seed, self.config.kind, diagnostics,
                      self.experiment_id, self.roll_id, self.frame_sequence)
        if terminal:
            self.emit(Event("terminal", frame=frame))
            return frame
        try:
            self.frames.put(frame, timeout=0.2)
        except queue.Full:
            # Frames are expendable; result events use a separate lossless queue.
            pass
        return frame

    def progress(self, seconds: float) -> bool:
        keep_going = True
        while True:
            try:
                command = self.commands.get_nowait()
            except queue.Empty:
                break
            if command.action == "stop":
                self.stopping = True
                keep_going = False
            elif command.action == "cancel":
                keep_going = False
            else:
                self.pending.append(command)
        return keep_going

    def request_forecast(self):
        if not self.active:
            self.emit(Event("status", "Start a flip or roll to forecast its outcome."))
            return
        if not self.forecasts:
            self.emit(Event("forecasting", "Calculating forecasts while the simulation continues..."))
        observed = self.world.state()
        seed = int(self.forecast_rng.integers(0, 2**32))
        self.forecasts.append(self.run_forecast(observed, self.world.elapsed, self.seed, seed))

    def run_forecast(self, observed: State, at_time: float, trial_seed: int, seed: int):
        initial = perturb_observation(self.config, observed, seed)
        horizon = max(0.0, self.config.horizon - at_time)
        trials = yield from self.shadow.run_steps(initial, [seed] * self.config.forecast_samples, horizon)
        faces = np.array([trial.outcome for trial in trials])
        counts = np.bincount(faces[faces >= 0], minlength=len(self.config.labels))
        forecast = Forecast(at_time, trial_seed, horizon, seed, counts.tolist(), int((faces < 0).sum()), len(trials),
                            (counts / len(trials)).tolist(), observed.position[0].tolist(),
                            observed.quaternion[0].tolist(), observed.velocity[0].tolist())
        self.emit(Event("forecast", forecast=forecast))

    def cancel_forecasts(self, message: str = ""):
        if self.forecasts:
            while self.forecasts:
                self.forecasts.popleft().close()
            self.emit(Event("forecast_cancelled", message))

    def batch(self, amount: int):
        # A batch takes exclusive use of the same shadow world.
        self.cancel_forecasts()
        self.emit(Event("busy", f"Running {amount} independent physical trials..."))
        completed = 0
        while completed < amount and not self.stopping:
            n = min(self.config.forecast_samples, amount - completed)
            seeds = self.launch_rng.integers(0, 2**32, size=n).tolist()
            # Pad only the final GPU chunk; unused scenarios never enter the statistics.
            padded_seeds = seeds + [0] * (self.config.forecast_samples - n)
            trials = self.shadow.run(sample_launch(self.config, padded_seeds), padded_seeds, self.progress)
            if not trials:
                break
            completed += n
            self.emit(Event("trials", f"Batch: {completed}/{amount}", trials=trials[:n]))
        self.emit(Event("ready", f"Batch finished: {completed}/{amount} trials recorded."))

    def advance(self, render: bool = True):
        at_rest, faces = self.world.step()
        if at_rest[0] or self.world.elapsed >= self.config.horizon:
            state = self.world.state()
            outcome = int(faces[0]) if at_rest[0] else -1
            trial = Trial(self.seed, outcome, self.world.elapsed, self.initial.position[0].tolist(),
                          self.initial.quaternion[0].tolist(), self.initial.velocity[0].tolist(),
                          state.position[0].tolist(), state.quaternion[0].tolist(), state.velocity[0].tolist(),
                          source="visible")
            self.emit(Event("trials", trials=[trial]))
            self.phase = self.config.labels[outcome] if outcome >= 0 else "Unresolved / edge / timeout"
            self.rolling = self.active = False
            self.publish_frame(terminal=True)
        if render:
            self.publish_frame()

    def handle(self, command: Command):
        if command.action not in ("stop", "configure"):
            if command.experiment_id != self.experiment_id:
                return
            if command.action != "roll" and command.roll_id != self.roll_id:
                return
        if command.action == "stop":
            self.stopping = True
        elif command.action == "cancel":
            self.cancel_forecasts("Forecast cancelled.")
        elif command.action == "configure":
            self.experiment_id = command.experiment_id
            self.roll_id = command.roll_id
            self.configure(command.config)
        elif command.action == "roll":
            self.cancel_forecasts()
            self.roll_id = command.roll_id
            self.seed = int(self.launch_rng.integers(0, 2**32))
            self.initial = sample_launch(self.config, [self.seed])
            self.world.reset(self.initial)
            self.rolling = self.active = True
            self.paused = False
            self.auto_forecast = command.auto_forecast
            self.next_checkpoint = 0
            self.phase = "In motion"
            self.emit(Event("roll", seed=self.seed))
            self.publish_frame()
        elif command.action == "pause" and self.active:
            self.paused = not self.paused
            self.phase = "Paused" if self.paused else "In motion"
            self.publish_frame()
        elif command.action == "step" and self.active:
            self.paused = True
            self.phase = "Paused"
            self.advance()
        elif command.action == "forecast":
            self.request_forecast()
        elif command.action == "batch":
            self.batch(command.amount)
        elif command.action == "camera":
            self.camera_offset = np.array([0.45, -0.6, 0.45]) if command.amount else np.array([0.9, -1.2, 0.8])
            self.publish_frame()
        elif command.action == "layers":
            assert command.layers is not None
            self.layers = command.layers
            self.publish_frame()
            self.emit(Event("layers", "Vector layers updated."))

    def run(self):
        checkpoints = (0.0, 0.3, 0.65)
        while not self.stopping:
            started = time.perf_counter()
            has_work = (self.rolling and not self.paused) or self.forecasts
            if self.pending:
                self.handle(self.pending.pop(0))
            else:
                try:
                    self.handle(self.commands.get(timeout=0.001 if has_work else 0.05))
                except queue.Empty:
                    pass
            if self.rolling and not self.paused and not self.stopping:
                if (self.auto_forecast and self.next_checkpoint < len(checkpoints)
                        and self.world.elapsed >= checkpoints[self.next_checkpoint]):
                    self.request_forecast()
                    self.next_checkpoint += 1
                # Keep animation running during forecasts, retaining exact checkpoint boundaries.
                for _ in range(3):
                    if self.rolling:
                        self.advance(render=False)
                    if (self.auto_forecast and self.next_checkpoint < len(checkpoints)
                            and self.world.elapsed >= checkpoints[self.next_checkpoint]):
                        break
                if self.rolling:
                    self.publish_frame()
            # Use the time left before the next frame, with a small minimum slice if rendering used the frame budget.
            deadline = max(started + 0.03, time.perf_counter() + 0.005)
            while self.forecasts and not self.stopping and time.perf_counter() < deadline:
                try:
                    next(self.forecasts[0])
                except StopIteration:
                    self.forecasts.popleft()
                    if not self.forecasts:
                        self.emit(Event("forecast_idle", "Forecast updated."))
            time.sleep(max(0, 0.03 - (time.perf_counter() - started)))
        self.cancel_forecasts()
        self.world.close()
        if self.shadow is not None:
            self.shadow.close()


def worker(config: Config, gpu: bool, commands, events, frames):
    frames.cancel_join_thread()
    try:
        gs.init(backend=gs.cuda if gpu else gs.cpu, seed=config.seed, logging_level="warning")
        Engine(config, commands, events, frames).run()
    except Exception:
        # Crossing a process boundary: show the complete diagnostic in the UI and its log.
        events.put(Event("error", traceback.format_exc()))
        traceback.print_exc()
    finally:
        gs.destroy()

import collections
import contextlib
import math
import os
import time

import numpy as np
import quadrants as qd
from PIL import Image

import genesis as gs
from genesis.utils.misc import get_entry_point_name


def animate(imgs, filename=None, fps=60):
    """
    Create a video from a list of images.

    Args:
        imgs (list): List of input images.
        filename (str, optional): Name of the output video file. If not provided, the name will default to the name
            of the script the process was launched from, with a timestamp and '.mp4' extension.
    """
    assert isinstance(imgs, list)
    if len(imgs) == 0:
        gs.logger.warning("No image to save.")
        return

    if filename is None:
        filename = f"{get_entry_point_name()}_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
    os.makedirs(os.path.abspath(os.path.dirname(filename)), exist_ok=True)

    gs.logger.info(f'Saving video to ~<"{filename}">~...')
    from moviepy import ImageSequenceClip

    imgs = ImageSequenceClip(imgs, fps=fps)
    imgs.write_videofile(
        filename,
        fps=fps,
        logger=None,
        codec="libx264",
        preset="ultrafast",
        # ffmpeg_params=["-crf", "0"],
    )
    gs.logger.info("Video saved.")


def save_img_arr(arr, filename="img.png"):
    assert isinstance(arr, np.ndarray)
    os.makedirs(os.path.abspath(os.path.dirname(filename)), exist_ok=True)
    img = Image.fromarray(arr)
    img.save(filename)
    gs.logger.info(f"Image saved to ~<{filename}>~.")


class Timer:
    def __init__(self, skip=False, level=0, qd_sync=False):
        self.accu_log = dict()
        self.skip = skip
        self.level = level
        self.qd_sync = qd_sync
        self.msg_width = 0
        self.reset()

    def reset(self):
        self.just_reset = True
        if self.level == 0 and not self.skip:
            try:
                column, _lines = os.get_terminal_size()
            except OSError:
                column = 80
            print("─" * column)
        if self.qd_sync and not self.skip:
            qd.sync()
        self.prev_time = self.init_time = time.perf_counter()

    def _stamp(self, msg="", _ratio=1.0):
        if self.skip:
            return

        if self.qd_sync:
            qd.sync()

        self.cur_time = time.perf_counter()
        self.msg_width = max(self.msg_width, len(msg))
        step_time = 1000 * (self.cur_time - self.prev_time) * _ratio
        accu_time = 1000 * (self.cur_time - self.init_time) * _ratio

        if msg not in self.accu_log:
            self.accu_log[msg] = [1, step_time, accu_time]
        else:
            self.accu_log[msg][0] += 1
            self.accu_log[msg][1] += step_time
            self.accu_log[msg][2] += accu_time

        if self.level > 0:
            prefix = " │  " * (self.level - 1)
            if self.just_reset:
                prefix += " ╭──"
            else:
                prefix += " ├──"
        else:
            prefix = ""

        print(
            f"{prefix}[{msg.ljust(self.msg_width)}] step: {step_time:5.3f}ms | accu: {accu_time:5.3f}ms | step_avg: {self.accu_log[msg][1] / self.accu_log[msg][0]:5.3f}ms | accu_avg: {self.accu_log[msg][2] / self.accu_log[msg][0]:5.3f}ms"
        )

        self.prev_time = time.perf_counter()
        self.just_reset = False

    def stamp(self, msg="", _ratio=1.0):
        return
        if self.skip:
            return

        if self.qd_sync:
            qd.sync()

        self.cur_time = time.perf_counter()
        self.msg_width = max(self.msg_width, len(msg))
        step_time = 1000 * (self.cur_time - self.prev_time) * _ratio
        accu_time = 1000 * (self.cur_time - self.init_time) * _ratio

        if msg not in self.accu_log:
            self.accu_log[msg] = [1, step_time, accu_time]
        else:
            self.accu_log[msg][0] += 1
            self.accu_log[msg][1] += step_time
            self.accu_log[msg][2] += accu_time

        if self.level > 0:
            prefix = " │  " * (self.level - 1)
            if self.just_reset:
                prefix += " ╭──"
            else:
                prefix += " ├──"
        else:
            prefix = ""

        print(
            f"{prefix}[{msg.ljust(self.msg_width)}] step: {step_time:5.3f}ms | accu: {accu_time:5.3f}ms | step_avg: {self.accu_log[msg][1] / self.accu_log[msg][0]:5.3f}ms | accu_avg: {self.accu_log[msg][2] / self.accu_log[msg][0]:5.3f}ms"
        )

        self.prev_time = time.perf_counter()
        self.just_reset = False


timers = dict()


def create_timer(name=None, new=False, level=0, qd_sync=False, skip_first_call=False):
    if name is None:
        return Timer()
    else:
        if name in timers and not new:
            timer = timers[name]
            timer.skip = False
            timer.reset()
            return timer
        else:
            timer = Timer(skip=skip_first_call, level=level, qd_sync=qd_sync)
            timers[name] = timer
            return timer


class Rate:
    """Fixed-frequency loop limiter: call ``sleep`` once per iteration to hold the loop at ``rate`` Hz.

    Each wake-up is scheduled against an ideal clock advanced by exactly one period per tick, rather than from the
    actual (over-slept) wake time, so ``time.sleep`` overshoot does not accumulate and the average rate stays on target.
    When an iteration runs longer than one period the limiter does not sleep and resets its schedule from the current
    time, so a loop that cannot keep up simply runs as fast as it can without building a sleep debt to burn off
    afterwards.
    """

    def __init__(self, rate):
        self.rate = rate
        self.period = 1.0 / rate
        self.next_time = time.perf_counter() + self.period

    def sleep(self):
        now = time.perf_counter()
        sleep_duration = self.next_time - now
        if sleep_duration > 0:
            time.sleep(sleep_duration)
            self.next_time += self.period
        else:
            self.next_time = now + self.period


class FPSTracker:
    """Times the phases of a stepped process and logs its achieved step rate.

    A step opens with `start`, its phases run under `phase`, named freely, or between `start_phase` and the next
    `start_phase` or `stop_phase` where a `with` block would indent too much, and `step` closes it: the wall time of
    every phase timed in the step and of the whole step, under the name `total`, is kept, and `timings` averages the
    latest `timings_window` of them. When `log` is set, the step rate is logged over fixed wall-clock windows: the
    per-window rate is the actual step count divided by the actual window duration (so it is phase-stable, unlike
    dividing a raw count by a smoothed time), then lightly EMA-smoothed for readability.
    """

    def __init__(
        self,
        n_envs,
        alpha=0.95,
        minimum_interval_seconds: float | None = 0.05,
        timings_window: int = 1,
        log: bool = True,
    ):
        self.n_envs = n_envs
        self.alpha = alpha
        self.minimum_interval_seconds = minimum_interval_seconds
        self.log = log
        self.window_start = None
        self.steps_since_last_print: int = 0
        self.fps_ema = None
        self.total_fps = 0.0
        self._step_start = None
        self._phase_open: tuple[str, float] | None = None
        self._phases_time: dict[str, float] = {}
        self._timings_history: collections.deque[dict[str, float]] = collections.deque(maxlen=timings_window)

    def start(self):
        """Open a step: the phases timed from here on belong to it."""
        self._step_start = time.perf_counter()
        self._phase_open = None
        self._phases_time.clear()

    def start_phase(self, phase: str):
        """Time the code that follows as part of the phase, until the next `start_phase` or `stop_phase`, which lets
        a loop time each of its iterations under its own phase without a `with` block. A phase running from a previous
        `start_phase` stops here."""
        self.stop_phase()
        self._phase_open = (phase, time.perf_counter())

    def stop_phase(self):
        """Stop the phase `start_phase` opened, adding its time to what the phase already took in this step."""
        if self._phase_open is not None:
            phase, tic = self._phase_open
            self._phases_time[phase] = self._phases_time.get(phase, 0.0) + time.perf_counter() - tic
            self._phase_open = None

    @contextlib.contextmanager
    def phase(self, phase: str):
        """Time the enclosed code as part of the phase of the current step, adding to the time the phase already
        took in this step. Blocks nest, each adding to its own phase."""
        tic = time.perf_counter()
        try:
            yield
        finally:
            self._phases_time[phase] = self._phases_time.get(phase, 0.0) + time.perf_counter() - tic

    def step(self, count: bool = True, current_time: float | None = None) -> float | None:
        """Close the step and keep its timings. A counted step enters the step rate, which is logged once the
        wall-clock window is long enough for a stable estimate, and returned then. current_time stands in for the
        clock, for a step whose phases were not timed."""
        self.stop_phase()
        if current_time is None:
            current_time = time.perf_counter()
        if self._step_start is not None:
            self._phases_time["total"] = current_time - self._step_start
        self._timings_history.append(dict(self._phases_time))
        if not self.log or not count:
            return None

        if self.window_start is None:
            self.window_start = current_time
            return None

        self.steps_since_last_print += 1

        # Accumulate until the window is long enough to give a stable estimate.
        window_dt = current_time - self.window_start
        if self.minimum_interval_seconds and window_dt < self.minimum_interval_seconds:
            return None

        window_fps = self.steps_since_last_print / window_dt
        self.fps_ema = window_fps if self.fps_ema is None else self.alpha * self.fps_ema + (1 - self.alpha) * window_fps

        if self.n_envs > 0:
            self.total_fps = self.fps_ema * self.n_envs
            gs.logger.info(
                f"Running at ~<{self.total_fps:,.2f}>~ FPS (~<{self.fps_ema:.2f}>~ FPS per env, ~<{self.n_envs}>~ envs)."
            )
        else:
            self.total_fps = self.fps_ema
            gs.logger.info(f"Running at ~<{self.fps_ema:.2f}>~ FPS.")

        self.window_start = current_time
        self.steps_since_last_print = 0
        return self.total_fps

    @property
    def timings(self) -> dict[str, float]:
        """Wall time in seconds of every phase timed in the latest `timings_window` steps, averaged over the steps that
        timed it."""
        phases_time = collections.defaultdict(list)
        for step_phases_time in self._timings_history:
            for phase, phase_time in step_phases_time.items():
                phases_time[phase].append(phase_time)
        return {phase: sum(phase_times) / len(phase_times) for phase, phase_times in phases_time.items()}

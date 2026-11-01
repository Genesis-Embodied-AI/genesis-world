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

    Each wake-up is scheduled against an ideal clock advanced by exactly the periods slept per tick, rather than from
    the actual (over-slept) wake time, so ``time.sleep`` overshoot does not accumulate and the average rate stays on
    target. When an iteration runs longer than the periods it sleeps, the limiter does not sleep and resets its schedule
    from the current time, so a loop that cannot keep up simply runs as fast as it can without building a sleep debt to
    burn off afterwards. ``period`` may be changed between two ticks, the next one following the new value.
    """

    def __init__(self, rate):
        self.period = 1.0 / rate
        self.last_time = time.perf_counter()

    @property
    def rate(self):
        """The frequency in Hz the loop is held at."""
        return 1.0 / self.period

    @rate.setter
    def rate(self, rate):
        self.period = 1.0 / rate

    def advance(self, n_periods=1):
        """Schedule the next tick 'n_periods' periods after the previous one, and return the time left until it.

        The time left is 0 when the loop is behind, and the schedule then starts anew from the current time. A caller
        that waits the time left in a way of its own, such as keeping a window responsive (see 'Viewer.sleep'), uses it
        in place of 'sleep'.
        """
        now = time.perf_counter()
        next_time = self.last_time + n_periods * self.period
        if next_time > now:
            self.last_time = next_time
            return next_time - now
        self.last_time = now
        return 0.0

    def sleep(self, n_periods=1):
        """Sleep until 'n_periods' periods have passed since the previous tick, or not at all if they already have."""
        duration = self.advance(n_periods)
        if duration > 0.0:
            time.sleep(duration)


class RunningMean:
    """Mean of a scalar fed one sample at a time, over the latest samples or exponentially weighted.

    Exactly one of 'window' and 'alpha' is given. With 'window', 'value' is the mean of the latest 'window' samples,
    NaN until that many have arrived, and it follows a change fully after that many samples. With 'alpha', it is the
    exponential moving average, the value so far weighing 'alpha' and each new sample '1 - alpha', starting from the
    first sample and NaN before it: it never forgets a sample entirely, and costs nothing per sample whatever the
    smoothing.
    """

    def __init__(self, window: int | None = None, alpha: float | None = None):
        if (window is None) == (alpha is None):
            gs.raise_exception("Exactly one of 'window' and 'alpha' must be given.")
        if window is not None and window < 1:
            gs.raise_exception(f"The window of a mean holds one sample at least, got {window}.")
        self._alpha = alpha
        self._samples: collections.deque[float] = collections.deque(maxlen=window)
        self._value = math.nan

    def update(self, sample: float):
        """Add a sample."""
        if self._alpha is not None:
            self._value = (
                sample if math.isnan(self._value) else self._alpha * self._value + (1.0 - self._alpha) * sample
            )
        else:
            self._samples.append(sample)
            if len(self._samples) == self._samples.maxlen:
                self._value = sum(self._samples) / len(self._samples)

    @property
    def value(self) -> float:
        """The mean of the samples so far, NaN until there are enough of them (see the class)."""
        return self._value


class StepTimer:
    """Times the phases of the steps of a stepped process.

    A step opens with `start`, its phases run under `phase`, named freely, or between `start_phase` and the next
    `start_phase` or `stop_phase` where a `with` block would indent too much, and `step` closes it. A closed step leaves
    in `timings`, until the next one, the wall time of every phase it timed, of the whole step under `step`, and of the
    period from the start of the previous step to its own under `period`. The period covers the time spent between the
    two steps as well, so its inverse is the rate the loop achieves. A step that `start` opens and `step` never closes
    leaves no trace, as the next `start` discards it. The first step has no period. The code timing a phase declares it
    with `add_phases` before the first step, so that a log laid out once lists every phase a step may time.
    """

    def __init__(self):
        self._step_start: float | None = None
        self._previous_step_start: float | None = None
        self._phase_open: tuple[str, float] | None = None
        self._phases_time: dict[str, float] = {}
        self._timings: dict[str, float] = {}
        self._phases: list[str] = ["step", "period"]

    @property
    def phases(self) -> tuple[str, ...]:
        """Every phase a step may time, `step` and `period` first, in declaration order (see `add_phases`)."""
        return tuple(self._phases)

    def add_phases(self, *phases: str):
        """Declare phases the steps time, ahead of the first step."""
        self._phases.extend(phase for phase in phases if phase not in self._phases)

    @property
    def timings(self) -> dict[str, float]:
        """Wall time in seconds of every phase the latest recorded step timed, by phase, empty before any."""
        return self._timings

    def start(self):
        """Open a step: the phases timed from here on belong to it."""
        self._step_start = time.perf_counter()
        self._phase_open = None
        self._phases_time.clear()

    def start_phase(self, phase: str):
        """Time the code that follows as part of a phase, until the next `start_phase` or `stop_phase`.

        It lets a loop time each of its iterations under its own phase without a `with` block. A phase running from a
        previous `start_phase` stops here.
        """
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
        """Time the enclosed code as part of a phase of the current step, adding to what the phase already took.

        Blocks nest, each adding to its own phase.
        """
        tic = time.perf_counter()
        try:
            yield
        finally:
            self._phases_time[phase] = self._phases_time.get(phase, 0.0) + time.perf_counter() - tic

    def step(self):
        """Close the step, keeping its timings in `timings` until the next one."""
        self.stop_phase()
        self._phases_time["step"] = time.perf_counter() - self._step_start
        if self._previous_step_start is not None:
            self._phases_time["period"] = self._step_start - self._previous_step_start
        self._timings = dict(self._phases_time)
        self._previous_step_start = self._step_start

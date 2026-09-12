import queue
import threading
import time
from typing import TYPE_CHECKING, Callable, Generic, TypeVar

import genesis as gs
from genesis.options.recorders import RecorderOptions
from genesis.utils import data_to_array

if TYPE_CHECKING:
    from .recorder_manager import RecorderManager


T = TypeVar("T")


class Recorder(Generic[T]):
    """
    Base class for all recorders.

    Note that modifying the signature of this class in recorder implementations should be avoided since instantiation is
    done through the RecorderManager.
    """

    def __init__(self, manager: "RecorderManager", options: RecorderOptions, data_func: Callable[[], T]):
        self._options = options
        self._manager = manager
        self._data_func = data_func
        self._steps_per_sample = 1
        self._error: Exception | None = None
        self._is_built = False
        self._is_recording = False
        self._is_alive = True

        self._data_queue: queue.Queue | None = None
        self._processor_thread: threading.Thread | None = None

        if options.hz:
            steps_per_sample_float = 1.0 / (options.hz * manager._step_dt)
            steps_per_sample = max(1, round(steps_per_sample_float))
            if abs(steps_per_sample_float - steps_per_sample) > gs.EPS:
                gs.logger.warning(
                    f"[Recorder] hz={options.hz} is not an integer multiple of step size of step dt. "
                    f"Using hz={1.0 / steps_per_sample / manager._step_dt} instead."
                )
            self._steps_per_sample = steps_per_sample

    # =============================== methods to implement ===============================

    @gs.assert_unbuilt
    def build(self):
        """
        Build the recorder, e.g. by initializing variables and creating widgets or file handles.
        """
        self._is_built = True

    @gs.assert_built
    def process(self, data, cur_time):
        """
        Process each incoming data sample.

        Parameters
        ----------
        data: Any
            The data to be processed.
        cur_time: float
            The current time of the simulation.
        """
        raise NotImplementedError(f"[{type(self).__name__}] process() is not implemented.")

    @gs.assert_built
    def cleanup(self):
        """
        Cleanup all resources, e.g. by closing widgets or files.

        This method is called when recording is stopped by `scene.stop_recording()`.
        """
        raise NotImplementedError(f"[{type(self).__name__}] cleanup() is not implemented.")

    @gs.assert_built
    def reset(self, envs_idx=None):
        """
        Reset the recorder, e.g. by flushing stored data.

        This method is called when the scene is reset by `scene.reset()`.

        Parameters
        ----------
        envs_idx: array_like, optional
            The indices of the environments to reset. If None, all environments are reset.
        """
        if self.run_in_thread:
            # sync the thread to ensure all data is processed
            self.sync()

    def _get_frame(self):
        """Return the frame of one recorded step from 'data_func' as host numpy arrays, so 'process' handles host data
        only. A subclass overrides this when its 'data_func' returns the source it reads the frame from.
        """
        return data_to_array(self._data_func())

    @property
    def run_in_thread(self) -> bool:
        """
        Whether to run the recorder in a background thread.

        Running in a background thread allows for processing data without blocking the main thread, so this is
        encouraged for most recorders (simply `return True`), but implementers should check that the recorder is
        thread-safe on all devices (threading on macOS tends to be less supported).
        """
        raise NotImplementedError(f"[{type(self).__name__}] run_in_thread is not implemented.")

    # =============================== recording  ===============================

    def _process_data_loop(self):
        """Background thread that processes and outputs data."""
        if self._data_queue is None:
            return

        while self._is_recording or not self._data_queue.empty():
            try:
                data, timestamp = self._data_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            try:
                self.process(data, timestamp)
            except Exception as error:
                # Forwarded to the stepping thread, which raises it at its next step or sync (see 'step'). The recorder
                # stops recording for good, as it does when a sample fails on the stepping thread (see 'record'). The
                # samples left in the queue are marked done, so 'sync' returns.
                self._error = error
                self._is_alive = False
                self._data_queue.task_done()
                while True:
                    try:
                        self._data_queue.get_nowait()
                        self._data_queue.task_done()
                    except queue.Empty:
                        return
            self._data_queue.task_done()

    @gs.assert_built
    def start(self):
        """Start the recording thread if run_in_thread is True."""
        self._is_recording = True
        if self.run_in_thread:
            self.start_thread()

    @gs.assert_built
    def stop(self):
        """Stop the recording thread and cleanup resources."""
        self._is_recording = False
        if self.run_in_thread:
            self.join_thread()
        self.cleanup()
        # The failure the background thread recorded is raised on the calling thread, once
        if self._error is not None:
            error, self._error = self._error, None
            gs.raise_exception_from(f"[{type(self).__name__}] Processing failed in the background: {error}", error)

    @gs.assert_built
    def join_thread(self):
        """Wait for the processor thread to finish."""
        if self._processor_thread is not None:
            self._processor_thread.join()
            self._processor_thread = None
            self._data_queue = None

    @gs.assert_built
    def start_thread(self):
        """Wait for the processor thread to finish."""
        if self._processor_thread is None:
            self._data_queue = queue.Queue(maxsize=self._options.buffer_size)
            self._processor_thread = threading.Thread(target=self._process_data_loop)
            self._processor_thread.start()
            try:
                # Note that it is necessary to rely on private threading atexit because functions registered using
                # 'atexit.register' are called AFTER 'threading._shutdown', causing deadlock for non-daemon threads.
                threading._register_atexit(self.stop)
            except AttributeError:
                pass
        else:
            gs.logger.warning(f"[{type(self).__name__}] start_thread(): Processor thread already exists.")

    @gs.assert_built
    def sync(self, timeout: float | None = None):
        """
        Wait until the data queue is empty.

        Parameters
        ----------
        timeout: float | None
            The maximum time to wait for the data queue to be empty. If None, wait indefinitely.
            If the timeout is reached, an exception is raised.
        """
        timestep = min(0.1, timeout) if timeout is not None else 0.1
        if self._data_queue is not None:
            if timeout is not None:
                start_time = time.time()

            # A frame taken off the queue is still being processed until the worker marks its task done. A failure
            # leaves no task pending (see '_process_data_loop'), so it is raised here, once, before and while waiting.
            while True:
                if self._error is not None:
                    error, self._error = self._error, None
                    gs.raise_exception_from(
                        f"[{type(self).__name__}] Processing failed in the background: {error}", error
                    )
                if not self._data_queue.unfinished_tasks:
                    break
                if timeout is not None and time.time() - start_time > timeout:
                    gs.raise_exception(f"[{type(self).__name__}] sync(): Timeout waiting for data queue to be empty.")

                dt = min(timestep, (start_time + timeout) - time.time()) if timeout is not None else timestep
                if dt > 0.0:
                    time.sleep(dt)

    @gs.assert_built
    def step(self, global_step: int):
        """Record the step when the sampling rate selects it, raising what the background thread reported."""
        if self._error is not None:
            error, self._error = self._error, None
            gs.raise_exception_from(f"[{type(self).__name__}] Processing failed in the background: {error}", error)
        if self._is_recording and global_step % self._steps_per_sample == 0:
            self.record(global_step)

    @gs.assert_built
    def record(self, global_step: int):
        """Record the current state whatever the sampling rate, such as the state a run stops or resets at."""
        if not self._is_recording or not self._is_alive:
            return

        global_time = global_step * self._manager._step_dt
        # A sample that raises ends the recorder for good, on this thread as on a worker (see '_process_data_loop'): a
        # reset restarts the others (see 'RecorderManager.reset') and the stop flushes what it wrote.
        try:
            # Sanitize to CPU numpy once at the boundary: process() code never has to branch on torch vs numpy, and no
            # GPU tensor is ever queued for cross-thread access in threaded mode.
            data = self._get_frame()
            if not self.run_in_thread:
                self.process(data, global_time)
                return
        except Exception:
            self._is_alive = False
            raise

        # threaded mode: put data in queue
        try:
            self._data_queue.put((data, global_time), block=False)
            return
        except queue.Full:
            try:
                self._data_queue.put((data, global_time), timeout=self._options.buffer_full_wait_time)
                return
            except queue.Full:
                pass

        gs.logger.debug("[Recorder] Data queue is full, dropping oldest data sample.")
        try:
            self._data_queue.get_nowait()
            # A dropped sample counts as done, since 'sync' waits for every sample taken off the queue
            self._data_queue.task_done()
        except queue.Empty:
            # Queue became empty between operations, just put the data
            pass
        finally:
            self._data_queue.put_nowait((data, global_time))

    @property
    def is_built(self) -> bool:
        return self._is_built

    @property
    def is_alive(self) -> bool:
        """Whether no sample or write of the recorder raised. A recorder that raised records no more, a reset restarts
        the others (see 'RecorderManager.reset'), and the stop flushes what it wrote.
        """
        return self._is_alive

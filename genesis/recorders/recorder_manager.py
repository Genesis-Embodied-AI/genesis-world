from typing import TYPE_CHECKING, Any, Callable

import genesis as gs

if TYPE_CHECKING:
    from .base_recorder import Recorder, RecorderOptions


class RecorderManager:
    """
    Manage the creation, processing, and cleanup of all data recorders.

    Parameters
    ----------
    step_dt: float
        The simulation time step.
    """

    RECORDER_TYPES_MAP = {}

    def __init__(self, step_dt: float):
        self._step_dt = step_dt

        self._recorders: list["Recorder"] = []
        self._is_recording = False
        self._is_built = False

    @gs.assert_unbuilt
    def add_recorder(self, data_func: Callable[[], Any], rec_options: "RecorderOptions") -> "Recorder":
        """
        Automatically read and process data. See RecorderOptions for more details.

        Parameters
        ----------
        data_func: Callable[[], Any]
            A callable with no arguments that returns the data a step records. A recorder that overrides
            'Recorder._get_frame' receives from it the object it reads the frame from (the trajectory recorder receives
            a TrajectorySource).
        rec_options: RecorderOptions
            The options for the recorder which determines how the data is recorded and processed.

        Returns
        -------
        recorder : Recorder
            The created recorder object.
        """
        recorder_cls = RecorderManager.RECORDER_TYPES_MAP[type(rec_options)]
        recorder = recorder_cls(self, rec_options, data_func)
        self._recorders.append(recorder)
        return recorder

    @gs.assert_unbuilt
    def build(self):
        """Start data recording."""
        # A recorder that fails to build leaves none of the recorders started before it running
        started = []
        try:
            for recorder in self._recorders:
                recorder.build()
                recorder.start()
                started.append(recorder)
        except Exception:
            # Every started recorder is stopped before a failure is raised: the first stop that raised, with the build
            # failure as its context, or the build failure itself
            stop_error = None
            for recorder in started:
                try:
                    recorder.stop()
                except Exception as error:
                    if stop_error is None:
                        stop_error = error
            if stop_error is not None:
                raise stop_error
            raise
        self._is_recording = True
        self._is_built = True

    @gs.assert_built
    def stop(self):
        """Stop and complete data recording."""
        if not self._is_recording:
            gs.logger.warning("[DataRecorder] Ignoring stop(): data recording is not active.")
        else:
            self._is_recording = False
            # Every recorder is stopped and flushed before the first failure is raised
            error = None
            for recorder in self._recorders:
                try:
                    recorder.stop()
                except Exception as recorder_error:
                    if error is None:
                        error = recorder_error
            self._recorders.clear()
            if error is not None:
                raise error

    @gs.assert_built
    def reset(self, envs_idx=None):
        for recorder in self._recorders:
            if recorder.is_alive:
                recorder.reset(envs_idx)
                recorder.start()

    @gs.assert_built
    def step(self, global_step: int):
        """
        Increment the step count and process data from each recording configuration.

        In threaded mode, data is put in queues. In non-threaded mode, data is processed synchronously.
        """
        if not self._is_recording:
            return

        for recorder in self._recorders:
            recorder.step(global_step)

    @gs.assert_built
    def record(self, global_step: int):
        """Record the current state through every recorder, whatever its sampling rate."""
        if not self._is_recording:
            return

        # Every recorder gets the state before the first failure is raised: it is the state each log ends with
        error = None
        for recorder in self._recorders:
            try:
                recorder.record(global_step)
            except Exception as recorder_error:
                if error is None:
                    error = recorder_error
        if error is not None:
            raise error

    @property
    def recorders(self) -> "list[Recorder]":
        return gs.List(self._recorders)

    @property
    def is_recording(self) -> bool:
        return self._is_recording

    @property
    def is_built(self) -> bool:
        return self._is_built


def register_recording(options_cls: type["RecorderOptions"]):
    def _impl(recorder_cls: type["Recorder"]):
        RecorderManager.RECORDER_TYPES_MAP[options_cls] = recorder_cls
        return recorder_cls

    return _impl

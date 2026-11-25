import functools
import logging
import threading
import weakref
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch

import genesis as gs
from genesis.utils.misc import BuildableMixin

if TYPE_CHECKING:
    from .base_recorder import Recorder, RecorderOptions

LOGGER = logging.getLogger(__name__)


class RecorderManager(BuildableMixin):
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

        try:
            # A run that ends without 'stop_recording' has its recordings stopped here. The hook runs before every
            # 'atexit' callback, among them the one of 'pyplot', which puts the base canvas back on each figure and
            # leaves the plotters nothing to draw on, and the one of Genesis, which destroys the scenes. The sample of
            # the state the stop records stays out of a recording stopped this way, since an encoder closes its file in
            # a hook of its own, registered at its first frame and therefore ahead of this one.
            def stop_at_exit(manager_ref):
                manager = manager_ref()
                if manager is not None and manager._is_built and manager._is_recording:
                    try:
                        manager.stop()
                    except Exception as error:
                        # 'threading' runs its exit hooks without catching anything, so a failure raised here would
                        # skip the hooks left and the join of the threads of the run
                        (gs.logger or LOGGER).error(f"[DataRecorder] Stopping the recordings at exit failed: {error}")

            threading._register_atexit(functools.partial(stop_at_exit, weakref.ref(self)))
        except AttributeError:
            pass

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
            # A recorder may stop another one it feeds, so they are destroyed once all of them stopped
            for recorder in self._recorders:
                recorder.destroy()
            self._recorders.clear()
            if error is not None:
                raise error

    def destroy(self) -> None:
        self._is_built = False

    @gs.assert_built
    def reset(self, envs_idx=None):
        for recorder in self._recorders:
            if recorder.is_alive:
                recorder.reset(envs_idx)

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
    @torch.no_grad()
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


def register_recording(options_cls: type["RecorderOptions"]):
    def _impl(recorder_cls: type["Recorder"]):
        RecorderManager.RECORDER_TYPES_MAP[options_cls] = recorder_cls
        return recorder_cls

    return _impl

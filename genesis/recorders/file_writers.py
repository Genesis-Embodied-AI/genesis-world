import csv
import logging
import os
from collections import defaultdict

import numpy as np

import genesis as gs
from genesis.options.recorders import (
    CSVFile as CSVFileWriterOptions,
    NPZFile as NPZFileWriterOptions,
    VideoFile as VideoFileWriterOptions,
)
from genesis.utils.video_encoder import VideoEncoder

from .base_recorder import Recorder
from .recorder_manager import register_recording

LOGGER = logging.getLogger(__name__)


class BaseFileWriter(Recorder):
    """
    Base class for file writers.

    """

    def build(self):
        super().build()

        os.makedirs(os.path.abspath(os.path.dirname(self._options.filename)), exist_ok=True)
        self._initialize_writer()

    def _get_filename(self):
        return self._options.filename

    def _initialize_writer(self):
        pass


@register_recording(VideoFileWriterOptions)
class VideoFileWriter(BaseFileWriter):
    encoder: "VideoEncoder | None"

    def build(self):
        self.encoder = None

        # Left fractional on purpose: the sampling period is not a whole number of frames per second in general, and
        # rounding it would make the video drift from the pace it was sampled at.
        if self._options.fps is None:
            self.fps = 1.0 / (self._steps_per_sample * self._manager._step_dt)
        else:
            self.fps = self._options.fps

        super().build()

    def _initialize_writer(self):
        self.encoder = VideoEncoder(
            self._get_filename(),
            self.fps,
            name=self._options.name,
            codec=self._options.codec,
            bitrate=self._options.bitrate,
            codec_options=self._options.codec_options,
            is_threaded=True,
        )

    def process(self, data, cur_time):
        # The copy is unconditional, which is what lets the frame be encoded asynchronously while the caller is free to
        # reuse its own buffer.
        self.encoder.write(data.copy())

    def cleanup(self):
        try:
            self.encoder.close()
        except gs.GenesisException as e:
            # Cleaning up carries on regardless, since this runs while every recorder of the scene is being stopped
            # and a video failing to encode must not leave the others open. The encoding error was already reported.
            gs.logger.warning(f"{e}")

    @property
    def run_in_thread(self) -> bool:
        # Encoding already happens on the encoder's own thread, whose queue is bounded and blocks the producer rather
        # than growing without limit or dropping the oldest frame.
        return False


@register_recording(CSVFileWriterOptions)
class CSVFileWriter(BaseFileWriter):
    def _initialize_writer(self):
        self.wrote_data = False
        self.file_handle = open(self._get_filename(), "w", encoding="utf-8", newline="")
        self.csv_writer = csv.writer(self.file_handle)

    def _sanitize_to_list(self, value):
        if isinstance(value, np.ndarray):
            return value.reshape((-1,)).tolist()
        elif isinstance(value, (int, float, bool)):
            return [value]
        elif isinstance(value, (list, tuple)):
            return value
        else:
            gs.raise_exception(f"[{type(self).__name__}] Unsupported data type: {type(value)}")

    def process(self, data, cur_time):
        row_data = [cur_time]
        if isinstance(data, dict):
            for value in data.values():
                row_data.extend(self._sanitize_to_list(value))
        else:
            row_data.extend(self._sanitize_to_list(data))

        if not self.wrote_data:  # write header
            header = ["timestamp"]
            if self._options.header:
                header.extend(self._options.header)
            else:
                if isinstance(data, dict):
                    for key, val in data.items():
                        if hasattr(val, "__len__"):
                            header.extend([f"{key}_{i}" for i in range(len(self._sanitize_to_list(val)))])
                        else:
                            header.append(key)
                else:
                    header.extend([f"data_{i}" for i in range(1, len(row_data))])
            if len(header) != len(row_data):
                gs.raise_exception(f"[{type(self).__name__}] header length does not match data length.")
            self.csv_writer.writerow(header)

        self.wrote_data = True
        self.csv_writer.writerow(row_data)
        if self._options.save_every_write:
            self.file_handle.flush()

    def cleanup(self):
        if self.file_handle:
            if self.wrote_data:
                self.file_handle.close()
                (gs.logger or LOGGER).info(f'[CSVFileWriter] Saved to ~<"{self._get_filename()}">~.')
            else:
                self.file_handle.close()
                os.remove(self._get_filename())  # delete empty file

    @property
    def run_in_thread(self) -> bool:
        return True


@register_recording(NPZFileWriterOptions)
class NPZFileWriter(BaseFileWriter):
    def build(self):
        self.all_data: dict[str, list] = defaultdict(list)

        super().build()

    def process(self, data, cur_time):
        self.all_data["timestamp"].append(cur_time)
        if isinstance(data, dict):
            for key, value in data.items():
                assert isinstance(value, (int, float, bool, list, tuple, np.ndarray))
                self.all_data[key].append(value)
        else:
            self.all_data["data"].append(np.asarray(data))

    def cleanup(self):
        filename = self._get_filename()
        if self.all_data["timestamp"]:  # at least one data point was collected
            try:
                np.savez_compressed(filename, **self.all_data)
            except ValueError as error:
                (gs.logger or LOGGER).warning(f"NPZFileWriter: saving as dtype=object due to ValueError: {error}")
                np.savez_compressed(filename, **{k: np.array(v, dtype=object) for k, v in self.all_data.items()})
            (gs.logger or LOGGER).info(
                f'[NPZFileWriter] Saved data with keys {list(self.all_data.keys())} to ~<"{filename}">~.'
            )
            self.all_data.clear()

    @property
    def run_in_thread(self) -> bool:
        return True

import logging
import os
import queue
import tempfile
import threading
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

import numpy as np

import av

import genesis as gs


LOGGER = logging.getLogger(__name__)

# Ordered by preference: hardware encoders first, software last.
H264_CODEC_CANDIDATES = (
    "h264_videotoolbox",  # macOS hardware
    "h264_nvenc",  # NVIDIA hardware
    "h264_vaapi",  # Linux VA-API hardware
    "h264_qsv",  # Intel Quick Sync
    "libx264",  # Software fallback
)

# Tuned for latency rather than compression ratio, so that encoding never becomes the bottleneck of a simulation.
H264_CODEC_OPTIONS = {
    "h264_videotoolbox": {"realtime": "1"},
    "h264_nvenc": {"preset": "p1", "tune": "ull"},
    "h264_vaapi": {},
    "h264_qsv": {"preset": "veryfast"},
    "libx264": {"preset": "veryfast", "tune": "zerolatency"},
}

# Bounds how far encoding may lag behind capture before the caller has to wait for it, and with it the memory held by
# frames not encoded yet.
ENCODER_QUEUE_SIZE = 10

# How long a full queue is waited on before checking that the encoding thread is still alive to drain it.
ENCODER_QUEUE_TIMEOUT = 0.1


@lru_cache(maxsize=None)
def _probe_h264_codec(codec: str, width: int, height: int) -> bool:
    """
    Test whether a codec can actually encode a single frame at the given resolution.

    Some hardware encoders (e.g. NVENC) reject small resolutions, so the actual target resolution must be tested.
    """
    # Use mkstemp instead of NamedTemporaryFile because Windows cannot open a NamedTemporaryFile from another handle
    path = None
    container = None
    try:
        fd, path = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)
        container = av.open(path, mode="w")
        stream = container.add_stream(codec, rate=30)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        frame = av.VideoFrame(width, height, "yuv420p")
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
        return True
    except (av.error.FFmpegError, ValueError):  # FFmpegError: codec/permission failures, ValueError: invalid parameters
        return False
    finally:
        # Closing before removing, since Windows refuses to delete a file that is still held open
        if container is not None:
            container.close()
        if path is not None:
            os.remove(path)


class VideoEncoder:
    """
    Encode frames to a video file one by one, as they are provided.

    Every frame is muxed to the container straight away instead of being accumulated, so the memory footprint is a
    single frame no matter how long the recording lasts.

    The container is created when the first frame is written rather than upfront, because hardware encoders reject
    some resolutions and the codec can only be validated against the true frame size. A video that never receives a
    frame therefore leaves no file behind.

    Parameters
    ----------
    filename : str
        Path of the output video file, ending in '.mp4'.
    fps : float
        Framerate the video is encoded at, in frames per second.
    name : str, optional
        Title metadata of the video. Defaults to the stem of `filename`.
    codec : str, optional
        Encoder to use. Defaults to the fastest one the machine supports at the frame resolution.
    bitrate : float, optional
        Bitrate of the video, in Mbit/s. The higher, the better the quality. Defaults to 1.0.
    codec_options : dict[str, str], optional
        Low-level options forwarded to ffmpeg. Defaults to latency-oriented settings for the selected codec.
    is_threaded : bool, optional
        Whether to encode in a background thread, so that `write` only pays for handing the frame over. The caller
        must not mutate a frame after passing it to `write`, since it is consumed asynchronously. Defaults to False.
    """

    def __init__(
        self,
        filename,
        fps,
        name="",
        codec="",
        bitrate=1.0,
        codec_options=None,
        is_threaded=False,
    ):
        self._filename = filename
        self._fps = fps
        self._name = name or Path(filename).stem
        self._codec = codec
        self._bitrate = bitrate
        self._codec_options = codec_options
        self._is_threaded = is_threaded

        self._container = None
        self._stream = None
        self._frame = None
        self._buffer = None
        self._row_bytes = 0

        self._queue = queue.Queue(maxsize=ENCODER_QUEUE_SIZE) if is_threaded else None
        self._thread = None
        self._is_encoding_over = False

    def _encode_loop(self):
        while True:
            frame = self._queue.get()
            if frame is None:
                # Reaching the sentinel is the only way out that leaves the video complete, so it is what tells a
                # worker that finished its queue from one that died on a frame, whenever that happened.
                self._is_encoding_over = True
                return
            self._encode(frame)

    def _enqueue(self, frame):
        # A worker that died on a failed encode drains nothing, so the wait is bounded and its liveness rechecked
        # rather than blocking forever on a full queue. The failure itself is reported by the threading excepthook.
        while True:
            if not self._thread.is_alive():
                gs.raise_exception(f'Video encoding stopped for "{self._filename}". See the error reported above.')
            try:
                self._queue.put(frame, timeout=ENCODER_QUEUE_TIMEOUT)
                return
            except queue.Full:
                pass

    def _open(self, frame):
        is_color = frame.ndim == 3 and frame.shape[-1] == 3
        height, width, *_ = frame.shape
        # 4:2:0 chroma subsampling halves both axes, so encoders reject an odd width or height. The frame is padded to
        # the next even size rather than cropped, keeping every pixel the camera rendered.
        height_even, width_even = height + height % 2, width + width % 2

        codec = self._codec
        if not codec:
            for candidate in H264_CODEC_CANDIDATES:
                if candidate in av.codecs_available and _probe_h264_codec(candidate, width_even, height_even):
                    codec = candidate
                    break
            else:
                gs.raise_exception(
                    "No supported H.264 codec found. Please install libx264 or specify a codec explicitly."
                )
        codec_options = self._codec_options or H264_CODEC_OPTIONS.get(codec, {})

        gs.logger.debug(
            f"Starting video recording using codec '{codec}' ({codec_options}) at {width}x{height} {self._fps}fps."
        )

        os.makedirs(os.path.abspath(os.path.dirname(self._filename)), exist_ok=True)
        self._container = av.open(self._filename, mode="w")
        self._container.metadata["title"] = self._name

        # ffmpeg expresses the framerate as a rational, so a fractional rate such as the one induced by a step size
        # that does not divide it evenly is preserved rather than rounded.
        self._stream = self._container.add_stream(codec, rate=Fraction(self._fps).limit_denominator(1000))
        self._stream.width, self._stream.height = (width_even, height_even)
        self._stream.pix_fmt = "yuv420p"
        self._stream.bit_rate = int(self._bitrate * (8 * 1024**2))
        self._stream.codec_context.options = codec_options

        # Frame storage is created once and reused, so encoding a frame costs a single copy. Rows are padded to the
        # alignment the encoder asks for, so the plane is viewed as raw bytes per row and only the leading ones that
        # the image occupies are ever written: 'line_size' is not a whole number of pixels in general.
        self._frame = av.VideoFrame(width_even, height_even, "rgb24" if is_color else "gray8")
        frame_plane = self._frame.planes[0]
        self._buffer = np.asarray(memoryview(frame_plane)).reshape((height_even, frame_plane.line_size))
        self._row_bytes = width * (3 if is_color else 1)
        # The padding is written once, since every frame only ever fills the leading rows and columns
        self._buffer[...] = 0

    def _encode(self, frame):
        if self._buffer is None:
            self._open(frame)

        self._buffer[: frame.shape[0], : self._row_bytes] = frame.reshape((frame.shape[0], -1))
        for packet in self._stream.encode(self._frame):
            self._container.mux(packet)

    def write(self, frame):
        """
        Add one frame to the video, as a grayscale [H, W] or color [H, W, RGB] array of integer type.

        The frame must be a CPU array already: any device transfer is the caller's responsibility, so that it happens
        on the thread owning the rendering context rather than on the encoding thread. When threaded, the frame must
        not be mutated after this call, since it is encoded asynchronously.
        """
        # Every frame is validated, not just the first one: the reusable frame storage is 8-bit, so anything else
        # would be truncated on assignment instead of raising. Validating here rather than at encoding time also
        # reports the error on the calling thread, where it can propagate, rather than on the encoding thread.
        is_color = frame.ndim == 3 and frame.shape[-1] == 3
        if frame.ndim != 2 + is_color or not np.issubdtype(frame.dtype, np.integer):
            gs.raise_exception("Frames must be either grayscale [H, W] or color [H, W, RGB] of integer type.")

        if self._is_threaded:
            if self._thread is None:
                self._is_encoding_over = False
                self._thread = threading.Thread(target=self._encode_loop)
                self._thread.start()
                try:
                    # Same private threading atexit as the recorders use, and for the same reason: functions
                    # registered through 'atexit.register' run after 'threading._shutdown', which deadlocks on
                    # non-daemon threads. It also finalizes the video when the interpreter exits before teardown
                    # had a chance to run.
                    threading._register_atexit(self.close)
                except AttributeError:
                    pass
            self._enqueue(frame)
        else:
            self._encode(frame)

    def close(self):
        """
        Flush the frames still held by the encoder and finalize the video file. Does nothing if already closed.
        """
        # A worker that never reached its sentinel died on a failed encode, so the frames it was given are lost and
        # the video must be reported as failed rather than saved.
        is_encoding_lost = False
        if self._thread is not None:
            while self._thread.is_alive():
                try:
                    self._queue.put(None, timeout=ENCODER_QUEUE_TIMEOUT)
                    break
                except queue.Full:
                    pass
            self._thread.join()
            self._thread = None
            is_encoding_lost = not self._is_encoding_over

        if self._container is not None:
            # 'stream' is None if no frame was ever written, in which case there is nothing to flush.
            if self._stream is not None and not is_encoding_lost:
                for packet in self._stream.encode(None):
                    self._container.mux(packet)
            self._container.close()

            if not is_encoding_lost:
                (gs.logger or LOGGER).info(f'Video saved to "~<{self._filename}>~".')

            self._container = None
            self._stream = None
            self._frame = None
            self._buffer = None

        if is_encoding_lost:
            gs.raise_exception(f'Video encoding failed for "{self._filename}". See the error reported above.')

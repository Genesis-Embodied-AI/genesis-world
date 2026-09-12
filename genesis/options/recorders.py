from typing import Annotated, Any

import av
from pydantic import BeforeValidator, Field, StrictBool

import genesis as gs
from genesis.typing import (
    NonNegativeInt,
    PathType,
    PositiveFloat,
    PositiveInt,
    PositiveVec2IType,
    StrArrayType,
    Vec3FArrayType,
    Vec3FType,
)

from .options import Options


class RecorderOptions(Options):
    """
    Options for recording simulation data by automatically sampling data from a data source, e.g. a sensor.

    Parameters
    ----------
    hz: float, optional
        The frequency at which to sample data, in Hz (samples per second).
        If None, the data will be sampled every step.
    buffer_size: int, optional
        Applicable when run_in_thread is True. The size of the data queue buffer.
        Defaults to 0, which means infinite size.
    buffer_full_wait_time: float, optional
        Applicable when run_in_thread is True. The time to wait for buffer space to become available when the
        buffer is full. Defaults to 0.1 seconds.
    """

    hz: PositiveFloat | None = None
    buffer_size: NonNegativeInt = 0
    buffer_full_wait_time: PositiveFloat = 0.1


class BaseFileWriterOptions(RecorderOptions):
    """
    Base class for file writer options.

    Parameters
    ----------
    filename: str
        The path of the output file.
    """

    filename: PathType


class VideoFile(BaseFileWriterOptions):
    """
    Stream video frames to file using PyAV.

    The PyAV writer streams data directly to the file instead of buffering it in memory. Incoming data should either be
    grayscale [H, W] or color [H, W, RGB] where values are uint8 (0, 255).

    Parameters
    ----------
    filename : str
        The path of the output video file ending in ".mp4".
    name : str
        The name of the video. Note that it may be different from filename. If empty, then filename will be used as a
        fallback. Default to "".
    fps : int, optional
        Frames per second for the video. Defaults to the data collection Hz ("real-time").
    codec : str, optional
        The codec to use for the video file. Defaults to "libx264".
    bitrate: float
        The bitrate of the video. This higher the better the quality of the video.
        Defaults to 1.0.
    codec_options: dict[str, str]
        Additional low-level codec options that will be pass to ffmpeg. Empty by default.
    """

    filename: PathType = Field(pattern=r"(?i).*\.mp4$")
    fps: PositiveInt | None = None
    name: str = ""
    codec: str = ""
    bitrate: float = 1.0
    codec_options: dict[str, str] = Field(default_factory=dict)

    def model_post_init(self, context: Any) -> None:
        if self.codec and self.codec not in av.codecs_available:
            gs.raise_exception(f"[{type(self).__name__}] Codec '{self.codec}' not supported.")


class CSVFile(BaseFileWriterOptions):
    """
    Writes to a .csv file using `csv.writer`.

    Can handle any array-like or dict[str, array-like] output, e.g. from sensors.
    Values must be N-dimensional tensors, arrays or scalars (np.generic, int, float, str)
    If the data or header is a dict, it cannot be further nested. Values are processed in order.

    Parameters
    ----------
    filename : str
        The name of the CSV file to save the data.
    header : tuple[str] | None, optional
        Column headers for the CSV file. It should match the format of the incoming data, where each scalar value has
        an associated header. If the data is a dict, the header should match the total length of the number of values
        after flattening the values.
    save_every_write: bool, optional
        Whether to flush the data to disk as soon as new data is recieved. Defaults to False.
    """

    filename: PathType = Field(pattern=r"(?i).*\.csv$")
    header: tuple[str, ...] | None = Field(default=None, strict=False)
    save_every_write: StrictBool = False


class NPZFile(BaseFileWriterOptions):
    """
    Buffers all data and writes to a .npz file at cleanup.

    Can handle any numeric or array-like or dict[str, array-like] data, e.g. from sensors.

    Parameters
    ----------
    filename : str
        The name of the .npz file to save the data.
    """

    filename: PathType = Field(pattern=r"(?i).*\.npz$")


class TrajectoryFile(BaseFileWriterOptions):
    """
    Record the scene state at every sampled step into a '.gstraj' file, which 'Scene.load_trajectory' opens to seek and
    replay. Recording starts with the build and stops with 'Scene.stop_recording'.

    The file contains the scene as 'Scene.export' writes it, then one frame per sampled step, the last being the state
    the recording stopped at. Every frame is kept: the recorder runs on the stepping thread and compresses and writes
    each completed chunk on a thread of its own while the simulation steps on. A chunk waits for the write of the
    previous one, so recording slows the simulation only when the disk cannot keep up, and a crash keeps every completed
    chunk.

    Exact mode records everything a step reads or writes: replay is bit-for-bit and the simulation resumes from any
    frame. Compressed mode records the model parameters, configuration, velocities, accelerations, control inputs,
    contacts and constraint forces. Its file is several times smaller, which matters for batched scenes, and a load
    recomputes the poses of links and geoms, which may differ at the last bits.

    Parameters
    ----------
    filename : str
        The '.gstraj' file to write.
    exact : bool, optional
        Whether to record in exact mode. If None, resolved based on the number of environments: exact for a single
        environment, compressed for a batched scene, with a warning. Defaults to None.
    chunk_size : int, optional
        Frames per chunk. Larger chunks compress better and seek slower, and a crash loses at most one. Defaults to 64.
    max_size : int, optional
        Bytes the file may grow to. A record that would pass it is dropped, the file is closed as a valid log, and the
        next step raises, so a run left recording fills the disk by at most this much. Raise it for a long run whose
        size is planned. If None, the file grows with the run. Defaults to 2 GiB.
    """

    filename: PathType = Field(pattern=r"(?i).*\.gstraj$")
    exact: StrictBool | None = None
    chunk_size: PositiveInt = 64
    max_size: PositiveInt | None = 2**31


class BasePlotterOptions(RecorderOptions):
    """
    Base class for plot visualization.

    Parameters
    ----------
    title: str
        The title of the plot.
    window_size: tuple[int, int]
        The size of the window in pixels.
    save_to_filename: str | None
        If provided, the animation will be saved to a file with the given filename.
    show_window: bool | None
        Whether to show the window. If not provided, it will be set to True if a display is connected, False otherwise.
    """

    title: str = ""
    window_size: PositiveVec2IType = (800, 600)
    save_to_filename: PathType | None = None
    show_window: StrictBool | None = None


class LinePlotterMixinOptions(Options):
    """
    Mixin class for live line plot visualization of scalar data.

    The recorded data_func should return scalar data (single scalar, a tuple of scalars, or a dict with string keys and
    scalar or tuple of scalars as values).

    Parameters
    ----------
    labels: tuple[str] | dict[str, tuple[str]] | None
        The labels for the plot. The length of the labels should match the length of the data.
        If a dict is provided, the data should also be a dict of tuples of strings that match the length of the data.
        The keys will be used as subplot titles and the values will be used as labels within each subplot.
    x_label: str, optional
        Label for the horizontal axis.
    y_label: str, optional
        Label for the vertical axis.
    history_length: int
        The maximum number of previous data to store.
    """

    labels: Annotated[
        tuple[str, ...] | dict[str, tuple[str, ...]] | None,
        BeforeValidator(
            lambda v: v if v is None else ({k: tuple(val) for k, val in v.items()} if isinstance(v, dict) else tuple(v))
        ),
    ] = None
    x_label: str = ""
    y_label: str = ""
    history_length: PositiveInt = 100


class PyQtLinePlot(BasePlotterOptions, LinePlotterMixinOptions):
    """
    Live line plot visualization of data using PyQtGraph.

    The recorded data_func should return scalar data (single scalar, a tuple of scalars, or a dict with string keys and
    scalar or tuple of scalars as values).

    Parameters
    ----------
    title: str
        The title of the plot.
    window_size: tuple[int, int]
        The size of the window in pixels.
    save_to_filename: str | None
        If provided, the animation will be saved to a file with the given filename.
    show_window: bool | None
        Whether to show the window. If not provided, it will be set to True if a display is connected, False otherwise.
    labels: tuple[str] | dict[str, tuple[str]] | None
        The labels for the plot. The length of the labels should match the length of the data.
        If a dict is provided, the data should also be a dict of tuples of strings that match the length of the data.
        The keys will be used as subplot titles and the values will be used as labels within each subplot.
    x_label: str, optional
        Label for the horizontal axis.
    y_label: str, optional
        Label for the vertical axis.
    history_length: int
        The maximum number of previous data to store.
    """

    pass


class MPLLinePlot(BasePlotterOptions, LinePlotterMixinOptions):
    """
    Live line plot visualization of data using matplotlib.

    The recorded data_func should return scalar data (single scalar, a tuple of scalars, or a dict with string keys and
    scalar or tuple of scalars as values).

    Parameters
    ----------
    title: str
        The title of the plot.
    window_size: tuple[int, int]
        The size of the window in pixels.
    save_to_filename: str | None
        If provided, the animation will be saved to a file with the given filename.
    show_window: bool | None
        Whether to show the window. If not provided, it will be set to True if a display is connected, False otherwise.
    labels: tuple[str] | dict[str, tuple[str]] | None
        The labels for the plot. The length of the labels should match the length of the data.
        If a dict is provided, the data should also be a dict of tuples of strings that match the length of the data.
        The keys will be used as subplot titles and the values will be used as labels within each subplot.
    x_label: str, optional
        Label for the horizontal axis.
    y_label: str, optional
        Label for the vertical axis.
    history_length: int
        The maximum number of previous data to store.
    """

    pass


class MPLImagePlot(BasePlotterOptions):
    """
    Live visualization of image data using matplotlib.

    The image data should be an array-like object with shape (H, W), (H, W, 1), (H, W, 3), or (H, W, 4).

    Parameters
    ----------
    title: str
        The title of the plot.
    window_size: tuple[int, int]
        The size of the window in pixels.
    save_to_filename: str | None
        If provided, the animation will be saved to a file with the given filename.
    show_window: bool | None
        Whether to show the window. If not provided, it will be set to True if a display is connected, False otherwise.
    """

    pass


class MPLVectorFieldPlot(BasePlotterOptions):
    """
    Live visualization of 3D vectors projected onto a 2D plane, colored by magnitude.

    At initialization, provide the normal direction (view axis) and the 3D positions of each vector.
    The data_func should return an array of shape (N, 3) with the 3D vector at each position (e.g. displacement or force).

    Parameters
    ----------
    title: str
        The title of the plot.
    window_size: tuple[int, int]
        The size of the window in pixels.
    positions: array-like of shape (N, 3)
        The 3D positions of each vector (e.g. probe positions in link-local frame).
    normal: tuple[float, float, float]
        The normal direction for projection (view axis). Vectors and positions are projected onto the plane
        perpendicular to this axis. Default: (0, 0, 1).
    scale_factor: float, optional
        The scale factor to apply to the vectors. Defaults to 0.1.
    max_magnitude: float, optional
        Maximum magnitude for the colorbar (colors are fixed to [0, max_magnitude]). Defaults to 1.0.
    subplot_titles: StrArrayType | None, optional
        If provided, the figure holds one subplot per title (K subplots in a near-square grid), all sharing
        ``positions``; the data_func then returns shape ``(K, N, 3)`` -- one vector field per subplot. ``None``
        (default) is a single plot whose data_func returns ``(N, 3)``.
    twist_scale_factor: float | None, optional
        When set, overlays a curved rotation arrow at each position showing the twist about the view ``normal``
        (the ``twist_vectors . normal`` component), with arc radius scaled by this factor -- useful for reading a
        rotational quantity (e.g. per-taxel torque) alongside the straight vectors. The data_func then returns a
        pair ``(vectors, twist_vectors)`` instead of a single array, each shaped as above. The cost is a busier plot
        and a second colorbar. ``None`` (default) draws only the straight vectors.
    twist_max_magnitude: float, optional
        Range for the diverging twist colorbar (colors fixed to [-twist_max_magnitude, +twist_max_magnitude], centered
        at zero). Only used when ``twist_scale_factor`` is set. Defaults to 1.0.
    save_to_filename: str | None
        If provided, the animation will be saved to a file with the given filename.
    show_window: bool | None
        Whether to show the window. If not provided, it will be set to True if a display is connected, False otherwise.
    """

    positions: Vec3FArrayType
    normal: Vec3FType = (0.0, 0.0, 1.0)
    scale_factor: PositiveFloat = 1.0
    max_magnitude: PositiveFloat = 1.0
    subplot_titles: StrArrayType | None = None
    twist_scale_factor: PositiveFloat | None = None
    twist_max_magnitude: PositiveFloat = 1.0

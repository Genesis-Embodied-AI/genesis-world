import io
import itertools
import logging
import math
import threading
import time
from collections import defaultdict
from collections.abc import Sequence
from functools import partial
from typing import Any, Callable, TypeVar

import numpy as np
from PIL import Image

import genesis as gs
import genesis.utils.geom as gu
from genesis.options.recorders import (
    BasePlotterOptions,
    LinePlotterMixinOptions,
    MPLImagePlot as MPLImagePlotterOptions,
    MPLLinePlot as MPLLinePlotterOptions,
    MPLVectorFieldPlot as MPLVectorFieldPlotterOptions,
    PyQtLinePlot as PyQtLinePlotterOptions,
)
from genesis.utils import data_to_array, has_display

from .base_recorder import Recorder
from .recorder_manager import RecorderManager, register_recording

IS_PYQTGRAPH_AVAILABLE = False
try:
    import pyqtgraph as pg

    IS_PYQTGRAPH_AVAILABLE = True
except ImportError:
    pass

IS_MATPLOTLIB_AVAILABLE = False
try:
    import matplotlib as mpl

    IS_MATPLOTLIB_AVAILABLE = tuple(map(int, mpl.__version__.replace("+", ".").split(".")[:3])) >= (3, 7, 0)
except ImportError:
    pass


LOGGER = logging.getLogger(__name__)


MPL_PLOTTER_RESCALE_MIN_X = 0.5
MPL_PLOTTER_RESCALE_RATIO_X = 0.15
MPL_PLOTTER_RESCALE_RATIO_Y = 0.15
MPL_PLOTTER_WINDOW_REFRESH_PERIOD = 0.1

COLORS = itertools.cycle(("r", "g", "b", "c", "m", "y"))


T = TypeVar("T")


class BasePlotter(Recorder):
    def __init__(self, manager: "RecorderManager", options: BasePlotterOptions, data_func: Callable[[], T]):
        if options.show_window is None:
            options.show_window = has_display()

        super().__init__(manager, options, data_func)
        self._frames_buffer: list[np.ndarray] = []

    def build(self):
        super().build()

        self.video_writer = None
        if self._options.save_to_filename:
            # The plotter renders its frame on the stepping thread before the writer samples it, so one is there
            self.video_writer = self._manager.add_recorder(
                data_func=partial(self._frames_buffer.pop, 0),
                rec_options=gs.recorders.VideoFile(
                    filename=self._options.save_to_filename,
                    hz=self._options.hz,
                ),
            )

    def process(self, data, cur_time):
        # Update plot
        self._update_plot()

        # Render frame if necessary
        if self._options.save_to_filename:
            self._frames_buffer.append(self.get_image_array())

    def cleanup(self):
        if self.video_writer is not None:
            self.video_writer.stop()
            self._frames_buffer.clear()
            self.video_writer = None

    def _update_plot(self):
        """
        Update plot.
        """
        raise NotImplementedError(f"[{type(self).__name__}] _update_plot() is not implemented.")

    def get_image_array(self):
        """
        Capture the plot image as a video frame.

        Returns
        -------
        image_array : np.ndarray
            The RGB image as a numpy array.
        """
        raise NotImplementedError(f"[{type(self).__name__}] get_image_array() is not implemented.")


class LinePlotHelper:
    """
    Helper class that manages line plot data.

    Use composition pattern.
    """

    def __init__(self, options: LinePlotterMixinOptions, data: dict[str, Sequence] | Sequence):
        self.x_data: list[float] = []
        self.y_data: defaultdict[str, defaultdict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        self._history_length = options.history_length

        # Note that these attributes will be set during first data processing or initialization
        self._is_dict_data: bool | None = None
        self._subplot_structure: dict[str, tuple[str, ...]] = {}

        if isinstance(data, dict):
            self._is_dict_data = True

            if options.labels is not None:
                assert isinstance(options.labels, dict), (
                    f"[{type(self).__name__}] Labels must be a dict when data is a dict"
                )
                assert set(options.labels.keys()) == set(data.keys()), (
                    f"[{type(self).__name__}] Label keys must match data keys"
                )

                for key in data.keys():
                    data_values = np.atleast_1d(data[key])
                    label_values = options.labels[key]
                    assert len(label_values) == len(data_values), (
                        f"[{type(self).__name__}] Label count must match data count for key '{key}'"
                    )
                    self._subplot_structure[key] = tuple(label_values)
            else:
                self._subplot_structure = {}
                for key, values in data.items():
                    values = np.atleast_1d(values)
                    self._subplot_structure[key] = tuple(f"{key}_{i}" for i in range(len(values)))
        else:
            self._is_dict_data = False
            data = np.atleast_1d(data)

            if options.labels is not None:
                labels = options.labels if isinstance(options.labels, Sequence) else (options.labels,)
                assert len(labels) == len(data), f"[{type(self).__name__}] Label count must match data count"
                plot_labels = tuple(labels)
            else:
                plot_labels = tuple(f"data_{i}" for i in range(len(data)))

            self._subplot_structure = {"main": plot_labels}

    def clear_data(self):
        self.x_data.clear()
        self.y_data.clear()

    def process(self, data, cur_time):
        """Process new data point and update plot."""

        if self._is_dict_data:
            processed_data = {}
            for key, values in data.items():
                if key not in self._subplot_structure:
                    continue  # skip keys not included in subplot structure
                values = np.atleast_1d(values)
                processed_data[key] = values
        else:
            data = np.atleast_1d(data)
            processed_data = {"main": data}

        # Update time data
        self.x_data.append(cur_time)

        # Update y data for each subplot
        for subplot_key, subplot_data in processed_data.items():
            channel_labels = self._subplot_structure[subplot_key]
            if len(subplot_data) != len(channel_labels):
                gs.logger.warning(
                    f"[{type(self).__name__}] Data length ({len(subplot_data)}) doesn't match "
                    f"expected number of channels ({len(channel_labels)}) for subplot '{subplot_key}', skipping..."
                )
                continue

            for i, channel_label in enumerate(channel_labels):
                if i < len(subplot_data):
                    self.y_data[subplot_key][channel_label].append(float(subplot_data[i]))

        # Maintain rolling history window
        if len(self.x_data) > self._history_length:
            self.x_data.pop(0)
            for subplot_key in self.y_data:
                for channel_label in self.y_data[subplot_key]:
                    try:
                        self.y_data[subplot_key][channel_label].pop(0)
                    except IndexError:
                        break  # empty, nothing to do.

    @property
    def history_length(self):
        return self._history_length

    @property
    def is_dict_data(self):
        return self._is_dict_data

    @property
    def subplot_structure(self):
        return self._subplot_structure


class BasePyQtPlotter(BasePlotter):
    """
    Base class for PyQt based plotters.
    """

    def __init__(self, manager: "RecorderManager", options: BasePlotterOptions, data_func: Callable[[], T]):
        super().__init__(manager, options, data_func)
        if threading.current_thread() is not threading.main_thread():
            gs.raise_exception("Impossible to run PyQtPlotter in background thread.")

    def build(self):
        if not IS_PYQTGRAPH_AVAILABLE:
            gs.raise_exception(
                f"{type(self).__name__} pyqtgraph is not installed. Please install it with `pip install pyqtgraph`."
            )

        super().build()

        self.app: pg.QtWidgets.QApplication | None = None
        self.widget: pg.GraphicsLayoutWidget | None = None
        self.plot_widgets: list[pg.PlotWidget] = []
        if not pg.QtWidgets.QApplication.instance():
            self.app = pg.QtWidgets.QApplication([])
        else:
            self.app = pg.QtWidgets.QApplication.instance()

        self.widget = pg.GraphicsLayoutWidget(show=self._options.show_window, title=self._options.title)
        if self._options.show_window:
            gs.logger.info(f"[{type(self).__name__}] created PyQtGraph window")
        self.widget.resize(*self._options.window_size)

    def cleanup(self):
        super().cleanup()

        if self.widget:
            try:
                self.widget.close()
                (gs.logger or LOGGER).debug(f"[{type(self).__name__}] closed PyQtGraph window")
            except Exception as e:
                (gs.logger or LOGGER).warning(f"[{type(self).__name__}] Error closing window: {e}")
            finally:
                self.plot_widgets.clear()
                self.widget = None

    @property
    def run_in_thread(self) -> bool:
        # Qt widgets are driven from the thread that owns the application, which is the stepping thread here
        return False

    def get_image_array(self):
        """
        Capture the plot image as a video frame.

        Returns
        -------
        image_array : np.ndarray
            The image as a numpy array in (b,g,r,a) format.
        """
        pixmap = self.widget.grab()
        qimage = pixmap.toImage()

        # pyqtgraph provides imageToArray but it always outputs (b,g,r,a) format
        # https://pyqtgraph.readthedocs.io/en/latest/api_reference/functions.html#pyqtgraph.functions.imageToArray
        return pg.imageToArray(qimage, copy=True, transpose=True)


@register_recording(PyQtLinePlotterOptions)
class PyQtLinePlotter(BasePyQtPlotter):
    def build(self):
        super().build()

        self.line_plot = LinePlotHelper(options=self._options, data=data_to_array(self._data_func()))
        self.curves: dict[str, list[pg.PlotCurveItem]] = {}

        # create plots for each subplot
        for subplot_idx, (subplot_key, channel_labels) in enumerate(self.line_plot.subplot_structure.items()):
            # add new row if not the first plot
            if subplot_idx > 0:
                self.widget.nextRow()

            plot_widget = self.widget.addPlot(title=subplot_key if self.line_plot.is_dict_data else self._options.title)
            plot_widget.setLabel("bottom", self._options.x_label)
            plot_widget.setLabel("left", self._options.y_label)
            plot_widget.showGrid(x=True, y=True, alpha=0.3)
            plot_widget.addLegend()

            # create lines for this subplot
            subplot_curves = []

            for color, channel_label in zip(COLORS, channel_labels):
                curve = plot_widget.plot(pen=pg.mkPen(color=color, width=2), name=channel_label)
                subplot_curves.append(curve)

            self.plot_widgets.append(plot_widget)
            if self._options.show_window:
                plot_widget.show()

            self.curves[subplot_key] = subplot_curves

    def process(self, data, cur_time):
        self.line_plot.process(data, cur_time)
        super().process(data, cur_time)

    def _update_plot(self):
        # update all curves
        for subplot_key, curves in self.curves.items():
            channel_labels = self.line_plot.subplot_structure[subplot_key]
            for curve, channel_label in zip(curves, channel_labels):
                curve.setData(x=self.line_plot.x_data, y=self.line_plot.y_data[subplot_key][channel_label])

        if self.app:
            self.app.processEvents()

    def cleanup(self):
        super().cleanup()
        self.line_plot.clear_data()
        self.curves.clear()


class BaseMPLPlotter(BasePlotter):
    """
    Base class for matplotlib based plotters.
    """

    def __init__(self, manager: "RecorderManager", options: BasePlotterOptions, data_func: Callable[[], T]):
        super().__init__(manager, options, data_func)
        if threading.current_thread() is not threading.main_thread():
            gs.raise_exception("Impossible to run MPLPlotter in background thread.")

    def build(self):
        if not IS_MATPLOTLIB_AVAILABLE:
            gs.raise_exception(
                f"{type(self).__name__} matplotlib is not installed. Please install it with `pip install matplotlib>=3.7.0`."
            )

        super().build()

        import matplotlib.pyplot as plt

        self.fig: plt.Figure | None = None
        self.axes: list[plt.Axes] = []
        self._background: Any = None
        self.window_refresh_time = -math.inf

        # matplotlib figsize uses inches
        dpi = mpl.rcParams.get("figure.dpi", 100)
        self.figsize = (self._options.window_size[0] / dpi, self._options.window_size[1] / dpi)

    def _make_subplot_grid(self, n_subplots: int, titles: "Sequence[str] | None" = None) -> "list":
        """
        Create ``self.fig`` with ``n_subplots`` axes in a near-square grid and return them as a flat list (one figure,
        many subplots).

        Unused cells in the grid are hidden, the figure title is set from ``options.title``, and each axis gets the
        matching entry of ``titles`` when provided. Subclasses fill the returned axes and then call
        ``_cache_background()``.
        """
        import matplotlib.pyplot as plt

        n_rows = max(1, int(np.floor(np.sqrt(n_subplots))))
        n_cols = int(np.ceil(n_subplots / n_rows))
        self.fig, axes = plt.subplots(n_rows, n_cols, figsize=self.figsize, squeeze=False, constrained_layout=True)
        axes = list(axes.ravel())
        for ax in axes[n_subplots:]:
            ax.set_visible(False)
        self.axes = axes[:n_subplots]
        if titles is not None:
            for ax, title in zip(self.axes, titles):
                ax.set_title(title)
        self.fig.suptitle(self._options.title)
        return self.axes

    def _cache_background(self):
        """Draw the figure and cache its full background region for fast blitting."""
        self.fig.canvas.draw()
        self._background = self.fig.canvas.copy_from_bbox(self.fig.bbox)

    def on_resize(self, event=None):
        """Re-cache the blit background after a resize."""
        if self.fig is not None and self.axes:
            self._cache_background()

    def _show_fig(self):
        if self._options.show_window:
            self.fig.show()
            gs.logger.info(f"[{type(self).__name__}] created matplotlib window")

    def flush_events(self):
        """Repaint the plot window with the latest drawing, at most once per MPL_PLOTTER_WINDOW_REFRESH_PERIOD."""
        # Repainting the window costs about as much as drawing the figure, while the samples arrive at the simulation
        # rate, which may run far ahead of real time. The drawing stays in the canvas, so a skipped repaint is caught
        # up by the next one.
        now = time.perf_counter()
        if now - self.window_refresh_time < MPL_PLOTTER_WINDOW_REFRESH_PERIOD:
            return
        self.fig.canvas.flush_events()
        self.window_refresh_time = now

    def cleanup(self):
        """Clean up matplotlib resources."""
        super().cleanup()

        # Logger may not be available anymore
        logger_exists = hasattr(gs, "logger")

        if self.fig is not None:
            try:
                import matplotlib.pyplot as plt

                plt.close(self.fig)
                if logger_exists:
                    (gs.logger or LOGGER).debug(f"[{type(self).__name__}] Closed matplotlib window")
            except Exception as e:
                if logger_exists:
                    (gs.logger or LOGGER).warning(f"[{type(self).__name__}] Error closing window: {e}")
            finally:
                self.fig = None

    def get_image_array(self):
        """
        Capture the plot image as a video frame.

        Returns
        -------
        image_array : np.ndarray
            The RGB image as a numpy array.
        """
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        if isinstance(self.fig.canvas, FigureCanvasAgg):
            # Read internal buffer
            width, height = self.fig.canvas.get_width_height(physical=True)
            # The alpha channel is read as padding, so that the resampling below skips premultiplying it.
            img = Image.frombuffer("RGBX", (width, height), self.fig.canvas.buffer_rgba())

            # Rescale image if necessary. A high-DPI canvas holds a whole multiple of the window size along each axis,
            # which PIL reduces much faster than it resamples to an arbitrary size.
            window_width, window_height = self._options.window_size
            if (width, height) != (window_width, window_height):
                if width % window_width == 0 and height % window_height == 0:
                    img = img.reduce((width // window_width, height // window_height))
                else:
                    img = img.resize((window_width, window_height), resample=Image.BOX)
            rgb_array = np.asarray(img.convert("RGB"))
        else:
            # Slower but more generic fallback only if necessary
            buffer = io.BytesIO()
            self.fig.canvas.print_figure(buffer, format="png", dpi="figure")
            buffer.seek(0)
            img = Image.open(buffer)
            rgb_array = np.asarray(img.convert("RGB"))

        return rgb_array

    @property
    def run_in_thread(self) -> bool:
        # Drawing is Python code, so it cannot overlap the stepping thread's hold on the interpreter lock: a worker
        # thread only adds the per-sample queue transfer and the switching between threads, which slows stepping down.
        return False


@register_recording(MPLLinePlotterOptions)
class MPLLinePlotter(BaseMPLPlotter):
    def build(self):
        super().build()

        self.line_plot = LinePlotHelper(options=self._options, data=data_to_array(self._data_func()))

        import matplotlib.pyplot as plt

        self.axes: list[plt.Axes] = []
        self.lines: dict[str, list[plt.Line2D]] = {}
        self.caches_bbox: list[Any] = []
        self.cache_xmax: float = -1
        self.x_max_plot: float = -1

        # Create figure and subplots
        n_subplots = len(self.line_plot.subplot_structure)
        if n_subplots == 1:
            self.fig, ax = plt.subplots(figsize=self.figsize)
            self.axes = [ax]
        else:
            self.fig, axes = plt.subplots(n_subplots, 1, figsize=self.figsize, sharex=True, constrained_layout=True)
            self.axes = axes if isinstance(axes, (list, tuple, np.ndarray)) else [axes]
        self.fig.suptitle(self._options.title)

        # Create lines for each subplot
        for subplot_idx, (subplot_key, channel_labels) in enumerate(self.line_plot.subplot_structure.items()):
            ax = self.axes[subplot_idx]
            ax.set_xlabel(self._options.x_label)
            ax.set_ylabel(self._options.y_label)
            ax.grid(True, alpha=0.3)

            if self.line_plot.is_dict_data and n_subplots > 1:
                ax.set_title(subplot_key)

            subplot_lines = []

            for color, channel_label in zip(COLORS, channel_labels):
                (line,) = ax.plot([], [], color=color, label=channel_label, linewidth=2)
                subplot_lines.append(line)

            self.lines[subplot_key] = subplot_lines

        # Legend must be outside, otherwise it will not play well with blitting
        self.fig.legend(ncol=sum(map(len, self.lines.values())), loc="outside lower center")
        self.fig.canvas.draw()

        for ax in self.axes:
            self.caches_bbox.append(self.fig.canvas.copy_from_bbox(ax.bbox))

        self._show_fig()

    def process(self, data, cur_time):
        self.line_plot.process(data, cur_time)
        super().process(data, cur_time)

    def _update_plot(self):
        # Update limits for each subplot if necessary
        limits_changed = False
        if len(self.line_plot.x_data) > 1:
            # First, check if the limits on y-axis must be extended to display all the available data
            subplots_ylim_data = []
            must_update_limit_y = False
            for ax, subplot_key in zip(self.axes, self.lines.keys()):
                subplot_y_data = self.line_plot.y_data[subplot_key]
                subplot_ylim_data = None
                if subplot_y_data:
                    all_y_values = list(itertools.chain.from_iterable(subplot_y_data.values()))
                    subplot_ylim_data = y_min_data, y_max_data = min(all_y_values), max(all_y_values)
                    y_min_plot, y_max_plot = ax.get_ylim()
                    if y_min_data < y_min_plot or y_max_plot < y_max_data:
                        must_update_limit_y = True
                subplots_ylim_data.append(subplot_ylim_data)

            # Next, adjust the limits on x-axis if they must be extended or adjusting y-axis is already planned. The
            # upper limit is kept as set here, since the axis reports it padded and a comparison against the padded
            # value would call for a rescale on every sample.
            x_limits_changed = False
            x_min_data, x_max_data = self.line_plot.x_data[0], self.line_plot.x_data[-1]
            if must_update_limit_y or self.x_max_plot < x_max_data:
                x_min_plot = max(0.0, x_min_data)
                self.x_max_plot = x_max_data + max(
                    MPL_PLOTTER_RESCALE_RATIO_X * (x_max_data - x_min_data), MPL_PLOTTER_RESCALE_MIN_X
                )
                ax.set_xlim((x_min_plot - gs.EPS, self.x_max_plot + gs.EPS))
                x_limits_changed = True

            # Finally, adjust the limits on y-axis if either x- or y-axis must be extended
            if x_limits_changed or must_update_limit_y:
                for ax, subplot_ylim_data in zip(self.axes, subplots_ylim_data):
                    if subplot_ylim_data is not None:
                        y_min_data, y_max_data = subplot_ylim_data
                        y_min_plot = y_min_data - MPL_PLOTTER_RESCALE_RATIO_Y * (y_max_data - y_min_data)
                        y_max_plot = y_max_data + MPL_PLOTTER_RESCALE_RATIO_Y * (y_max_data - y_min_data)
                        ax.set_ylim((y_min_plot - gs.EPS, y_max_plot + gs.EPS))
                limits_changed = True

        # Must redraw the entire figure if the limits have changed
        if limits_changed:
            self.fig.canvas.draw()

        # Update background if the entire figure has been updated, or the buffer size has been exceeded
        if limits_changed or (len(self.line_plot.x_data) > 1 and self.cache_xmax < self.line_plot.x_data[0] + gs.EPS):
            self.caches_bbox = [self.fig.canvas.copy_from_bbox(ax.bbox) for ax in self.axes]
            self.cache_xmax = self.line_plot.x_data[-2]

        # Update lines for each subplot
        for ax, cache_bbox, (subplot_key, subplot_lines) in zip(self.axes, self.caches_bbox, self.lines.items()):
            # Restore background and update line data for this subplot
            self.fig.canvas.restore_region(cache_bbox)

            # Update lines
            channel_labels = self.line_plot.subplot_structure[subplot_key]
            for line, channel_label in zip(subplot_lines, channel_labels):
                y_data = self.line_plot.y_data[subplot_key][channel_label]
                line.set_data(self.line_plot.x_data, y_data)
                ax.draw_artist(line)

            # Blit the updated subplot
            self.fig.canvas.blit(ax.bbox)

        self.flush_events()

    def cleanup(self):
        super().cleanup()
        self.line_plot.clear_data()
        self.lines.clear()
        self.caches_bbox.clear()
        self.cache_xmax = -1
        self.x_max_plot = -1


@register_recording(MPLImagePlotterOptions)
class MPLImagePlotter(BaseMPLPlotter):
    """
    Live image viewer using matplotlib.

    The image data should be an array-like object with shape (H, W), (H, W, 1), (H, W, 3), or (H, W, 4).
    """

    def build(self):
        super().build()

        import matplotlib.pyplot as plt

        self.image_plot = None
        self.background = None

        self.fig, self.ax = plt.subplots(figsize=self.figsize)
        self.fig.tight_layout(pad=0)
        self.ax.set_axis_off()
        self.fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        self.image_plot = self.ax.imshow(np.zeros((1, 1)), cmap="plasma", origin="upper", aspect="auto")
        self._show_fig()

    def process(self, data, cur_time):
        """Process new image data and update display."""
        img_data = np.asarray(data)

        vmin, vmax = np.min(img_data), np.max(img_data)

        current_vmin, current_vmax = self.image_plot.get_clim()
        if vmin != current_vmin or vmax != current_vmax:
            self.image_plot.set_clim(vmin, vmax)
            self.fig.canvas.draw()
            self.background = self.fig.canvas.copy_from_bbox(self.ax.bbox)

        self.fig.canvas.restore_region(self.background)
        self.image_plot.set_data(img_data)
        self.ax.draw_artist(self.image_plot)
        self.fig.canvas.blit(self.ax.bbox)

        self.flush_events()

    def cleanup(self):
        super().cleanup()

        self.ax = None
        self.image_plot = None
        self.background = None


def _project_to_plane(normal: np.ndarray, *arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    """Project 3D arrays onto the plane perpendicular to normal."""
    uv = np.stack(gu.orthogonals(normal / np.linalg.norm(normal)), axis=1)
    return tuple(data @ uv for data in arrays)


@register_recording(MPLVectorFieldPlotterOptions)
class MPLVectorFieldPlotter(BaseMPLPlotter):
    """
    Live 3D vector field viewer: projects positions and vectors onto a 2D plane and plots arrows colored by magnitude.

    The data_func returns an array of shape (N, 3) with the 3D vector at each position. When ``subplot_titles`` is
    set (K titles), the figure holds K subplots sharing the same positions, and the data_func instead returns shape
    (K, N, 3) -- one vector field per subplot (e.g. one per environment).

    When ``twist_scale_factor`` is set, a curved rotation arrow is overlaid at each position for the twist about the
    view normal (the ``twist_vectors . normal`` component), and the data_func instead returns a pair
    ``(vectors, twist_vectors)`` with each entry shaped as above. Positive twist (right-hand rule about ``normal``)
    sweeps counter-clockwise; the arc is colored by signed twist on a diverging colorbar centered at zero.
    """

    def build(self):
        super().build()

        from matplotlib.collections import LineCollection

        opts = self._options
        positions = np.array(opts.positions, dtype=float)
        if positions.ndim != 2 or positions.shape[1] != 3:
            gs.raise_exception(f"[{type(self).__name__}] positions must have shape (N, 3), got {positions.shape}.")
        normal = np.array(opts.normal, dtype=float)
        n_norm = np.linalg.norm(normal)
        if normal.size != 3 or n_norm < gs.EPS:
            gs.raise_exception(f"[{type(self).__name__}] normal must be a non-zero 3D vector.")
        normal = normal / n_norm

        (xy,) = _project_to_plane(normal, positions)
        (x_min, y_min), (x_max, y_max) = xy.min(axis=0), xy.max(axis=0)
        margin = 0.1 * max(np.max(np.ptp(xy, axis=0)), gs.EPS)

        titles = opts.subplot_titles
        axes = self._make_subplot_grid(len(titles) if titles else 1, titles)

        self._positions = positions
        self._normal = normal
        self._scale_factor = opts.scale_factor
        self._max_magnitude = opts.max_magnitude
        self._twist_scale_factor = opts.twist_scale_factor
        self._twist_max_magnitude = opts.twist_max_magnitude
        self._scatters = []
        self._quivers = []
        self._twist_arcs = []
        self._twist_heads = []
        n = len(xy)

        # Sign that maps positive twist (right-hand rule about normal) to a counter-clockwise sweep in the projected
        # basis, independent of the handedness of the orthogonals() basis.
        u, v = gu.orthogonals(normal)
        self._twist_sign = 1.0 if np.dot(np.cross(u, v), normal) > 0.0 else -1.0

        for ax in axes:
            ax.set_xlim(x_min - margin, x_max + margin)
            ax.set_ylim(y_min - margin, y_max + margin)
            ax.set_aspect("equal")
            ax.set_axis_off()
            self._scatters.append(
                ax.scatter(
                    xy[:, 0], xy[:, 1], s=8, c=np.zeros(n), cmap="plasma", vmin=0, vmax=self._max_magnitude, zorder=0
                )
            )
            self._quivers.append(
                ax.quiver(
                    xy[:, 0],
                    xy[:, 1],
                    np.zeros(n),
                    np.zeros(n),
                    np.zeros(n),
                    cmap="plasma",
                    clim=(0, self._max_magnitude),
                    zorder=1,
                    scale_units="xy",
                    scale=1,
                    pivot="mid" if self._twist_scale_factor is not None else "tail",
                )
            )
            if self._twist_scale_factor is not None:
                arcs = LineCollection([], cmap="coolwarm", zorder=2)
                arcs.set_clim(-self._twist_max_magnitude, self._twist_max_magnitude)
                arcs.set_array(np.zeros(n))
                ax.add_collection(arcs)
                self._twist_arcs.append(arcs)
                self._twist_heads.append(
                    ax.quiver(
                        xy[:, 0],
                        xy[:, 1],
                        np.zeros(n),
                        np.zeros(n),
                        np.zeros(n),
                        cmap="coolwarm",
                        clim=(-self._twist_max_magnitude, self._twist_max_magnitude),
                        zorder=3,
                        scale_units="xy",
                        scale=1,
                    )
                )
            for artist in (self._scatters[-1], self._quivers[-1], *self._twist_arcs[-1:], *self._twist_heads[-1:]):
                artist.set_clip_box(ax.bbox)
                artist.set_clip_on(True)
            # The zero-length twist-head quiver renders as a dot per taxel; animate it (and the arcs) so it is not
            # baked into the cached background and left ghosting behind the live scatter.
            for artist in (*self._twist_arcs[-1:], *self._twist_heads[-1:]):
                artist.set_animated(True)
        self.fig.colorbar(self._quivers[-1], ax=axes, label="Magnitude")
        if self._twist_scale_factor is not None:
            self.fig.colorbar(self._twist_arcs[-1], ax=axes, label="Twist")
        self._cache_background()
        self._show_fig()

        self.fig.canvas.mpl_connect("resize_event", self.on_resize)

    def process(self, data, cur_time):
        """Process new vector data and update each subplot's quiver (and the twist overlay when enabled)."""
        is_twist = self._twist_scale_factor is not None
        vectors_data, twist_data = data if is_twist else (data, None)
        # Promote a bare (N, 3) field to a single-subplot (K, N, 3) stack.
        vectors_all = np.asarray(vectors_data)
        if vectors_all.ndim == 2:
            vectors_all = vectors_all[None]
        twist_all = None
        if is_twist:
            twist_all = np.asarray(twist_data)
            if twist_all.ndim == 2:
                twist_all = twist_all[None]

        n = len(self._positions)
        if vectors_all.ndim != 3 or vectors_all.shape[1:] != (n, 3):
            return
        if vectors_all.shape[0] != len(self.axes) or self._background is None:
            return
        if is_twist and twist_all.shape != vectors_all.shape:
            return

        # Blit the whole figure, not per-axes: a stroke clipped at an axes box can bleed a pixel past it, and a
        # per-axes blit never restores that sliver, so it would accumulate as residue.
        self.fig.canvas.restore_region(self._background)
        for i_ax, (ax, scatter, quiver, vectors) in enumerate(
            zip(self.axes, self._scatters, self._quivers, vectors_all)
        ):
            magnitudes = np.linalg.norm(vectors, axis=-1)
            xy, uv = _project_to_plane(self._normal, self._positions, vectors)
            scatter.set_offsets(xy)
            scatter.set_array(magnitudes)
            quiver.set_offsets(xy)
            quiver.set_UVC(*(uv * self._scale_factor).T)
            quiver.set_array(magnitudes)
            ax.draw_artist(scatter)
            ax.draw_artist(quiver)
            if is_twist:
                arcs, heads = self._twist_arcs[i_ax], self._twist_heads[i_ax]
                twist = twist_all[i_ax] @ self._normal
                radius = self._twist_scale_factor * np.abs(twist)
                direction = self._twist_sign * np.sign(twist)
                # 270-degree arc per taxel; the open quarter marks the rotation and leaves room for the head.
                ang = direction[:, None] * np.linspace(0.0, 1.5 * np.pi, 16)[None, :]
                xs = xy[:, 0][:, None] + radius[:, None] * np.cos(ang)
                ys = xy[:, 1][:, None] + radius[:, None] * np.sin(ang)
                segments = np.stack((xs, ys), axis=-1)
                arcs.set_segments(list(segments))
                arcs.set_array(twist)
                heads.set_offsets(segments[:, -2])
                heads.set_UVC(segments[:, -1, 0] - segments[:, -2, 0], segments[:, -1, 1] - segments[:, -2, 1], twist)
                ax.draw_artist(arcs)
                ax.draw_artist(heads)
        self.fig.canvas.blit(self.fig.bbox)
        self.flush_events()

    def cleanup(self):
        super().cleanup()
        self._scatters = None
        self._quivers = None
        self._twist_arcs = None
        self._twist_heads = None
        self._positions = None
        self._normal = None
        self._scale_factor = None
        self._max_magnitude = None
        self._twist_scale_factor = None
        self._twist_max_magnitude = None
        self._twist_sign = None

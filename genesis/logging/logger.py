import logging
import os
import sys
import threading
from contextlib import contextmanager

import genesis as gs
from genesis.styles import THEME, colors, formats, style
from genesis.version import __version__

from .time_elapser import TimeElapser


class GenesisFormatter(logging.Formatter):
    def __init__(self, verbose_time=True):
        super().__init__()

        self.mapping = {
            logging.DEBUG: colors.GREEN,
            logging.INFO: colors.BLUE,
            logging.WARNING: colors.YELLOW,
            logging.ERROR: colors.RED,
            logging.CRITICAL: colors.RED,
        }

        if verbose_time:
            self.TIME = "%(asctime)s.%(msecs)03d"
            self.DATE_FORMAT = "%y-%m-%d %H:%M:%S"
            self.INFO_length = 41
        else:
            self.TIME = "%(asctime)s"
            self.DATE_FORMAT = "%H:%M:%S"
            self.INFO_length = 28

        self.LEVEL = "%(levelname)s"
        self.MESSAGE = "%(message)s"

        self.last_output = ""
        self.last_color = ""

    def format(self, record):
        self.last_color = self.mapping.get(record.levelno)
        log_fmt = f"{style.prefix(self.last_color, self.TIME, self.LEVEL, record.levelno)}{self.MESSAGE}{formats.RESET}"
        formatter = logging.Formatter(log_fmt, datefmt=self.DATE_FORMAT)
        msg = style.markup(formatter.format(record), self.last_color)
        self.last_output = msg
        return msg


class Logger:
    def __init__(self, logging_level, verbose_time, theme):
        if isinstance(logging_level, str):
            logging_level = logging_level.upper()

        # The theme is set before the formatter is created, which reads the colors of its levels once and for all.
        if theme not in tuple(THEME):
            gs.raise_exception(f"Unsupported theme: ~~<{theme}>~~")
        style.theme = THEME(theme)

        self._logger = logging.getLogger("genesis")
        self._logger.setLevel(logging_level)
        # Records are already printed by the handler added below. Propagating them to the root logger would print them
        # a second time, unformatted, whenever the application configures root logging (e.g. 'logging.basicConfig').
        self._logger.propagate = False

        self._formatter = GenesisFormatter(verbose_time)

        self._handler = logging.StreamHandler(sys.stdout)
        self._handler.setLevel(logging_level)
        self._handler.setFormatter(self._formatter)
        self._logger.addHandler(self._handler)

        self._stream = self._handler.stream
        self._is_new_line = True

        self.timer_lock = threading.Lock()

    def addFilter(self, filter):
        self._logger.addFilter(filter)

    def removeFilter(self, filter):
        self._logger.removeFilter(filter)

    def removeHandler(self, handler):
        self._logger.removeHandler(handler)

    @property
    def INFO_length(self):
        return self._formatter.INFO_length

    @contextmanager
    def log_wrapper(self):
        self.timer_lock.acquire()

        # swap with timer output
        if not self._is_new_line and not self._stream.closed:
            self._stream.write("\r")
        try:
            yield
        finally:
            self._is_new_line = True
            self.timer_lock.release()

    @contextmanager
    def lock_timer(self):
        self.timer_lock.acquire()
        try:
            yield
        finally:
            self.timer_lock.release()

    def log(self, level, msg, *args, **kwargs):
        with self.log_wrapper():
            self._logger.log(level, msg, *args, **kwargs)

    def debug(self, message):
        with self.log_wrapper():
            self._logger.debug(message)

    def info(self, message):
        with self.log_wrapper():
            self._logger.info(message)

    def warning(self, message):
        with self.log_wrapper():
            self._logger.warning(message)

    def error(self, message):
        with self.log_wrapper():
            self._logger.error(message)

    def critical(self, message):
        with self.log_wrapper():
            self._logger.critical(message)

    def banner(self, device_name, backend, total_mem, seed, debug, precision, performance_mode):
        """Log the greeting banner of Genesis, followed by the device it runs on and the options it was initialized with.

        Parameters
        ----------
        device_name : str
            The name of the device Genesis runs on.
        backend : gs.backend
            The backend Genesis runs on.
        total_mem : float
            The memory of the device, in GB.
        seed : int | None
            The seed of the random number generators, if any.
        debug : bool
            Whether Genesis runs in debug mode.
        precision : str
            The floating point precision, either '32' or '64'.
        performance_mode : bool
            Whether Genesis runs in performance mode.

        The raw theme leaves out the box and the emojis.
        """
        is_decorated = style.theme is not THEME.raw
        if is_decorated:
            try:
                columns, _lines = os.get_terminal_size()
            except OSError:
                columns = 80
            wave_width = (columns - self.INFO_length - 11) // 2
            if wave_width % 2 == 0:
                wave_width -= 1
            wave_width = max(0, min(38, wave_width))
            bar_width = wave_width * 2 + 9
            wave = ("┈┉" * wave_width)[:wave_width]
            self.info(f"~<╭{'─' * (bar_width)}╮>~")
            self.info(f"~<│{wave}>~ ~~~~<Genesis>~~~~ ~<{wave}│>~")
            self.info(f"~<╰{'─' * (bar_width)}╯>~")

        self.info(
            f"Running on ~~<[{device_name}]>~~ with backend ~~<{backend}>~~. Device memory: ~~<{total_mem:.2f}>~~ GB."
        )

        msg_options = ", ".join(
            f"{f'{emoji} ' if is_decorated else ''}{name}: ~~<{val}>~~"
            for emoji, name, val in (
                ("🔖", "version", __version__),
                ("🎨", "theme", style.theme.value),
                ("🌱", "seed", seed),
                ("🐛", "debug", bool(debug)),
                ("📏", "precision", precision),
                ("🔥", "performance", bool(performance_mode)),
                ("💬", "verbose", logging.getLevelName(self.level)),
            )
        )
        self.info(f"{'🚀 ' if is_decorated else ''}Genesis initialized. {msg_options}")

    def shutdown(self):
        """Log the exit line of Genesis."""
        self.info(f"{'💤 ' if style.theme is not THEME.raw else ''}Exiting Genesis and caching compiled kernels...")

    def raw(self, message):
        self._stream.write(style.markup(message, self._formatter.last_color))
        self._stream.flush()
        if message.endswith("\n"):
            self._is_new_line = True
        else:
            self._is_new_line = False

    def timer(self, msg, refresh_rate=10, end_msg=""):
        self.info(msg)
        return TimeElapser(self, refresh_rate, end_msg)

    @property
    def handler(self):
        return self._handler

    @property
    def last_output(self):
        return self._formatter.last_output

    @property
    def level(self):
        return self._logger.level

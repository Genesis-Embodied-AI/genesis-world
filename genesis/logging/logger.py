import logging
import os
import sys

import genesis as gs
from genesis.styles import THEME, colors, formats, style


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

    def format(self, record):
        color = self.mapping.get(record.levelno)
        log_fmt = f"{style.prefix(color, self.TIME, self.LEVEL, record.levelno)}{self.MESSAGE}{formats.RESET}"
        formatter = logging.Formatter(log_fmt, datefmt=self.DATE_FORMAT)
        return style.markup(formatter.format(record), color)


class Logger:
    def __init__(self, logging_level, verbose_time, theme):
        if isinstance(logging_level, str):
            logging_level = logging_level.upper()

        # The theme is set before the formatter is created, which reads the colors of its levels once and for all.
        if theme not in THEME:
            gs.raise_exception(f"Unsupported theme: ~<{theme}>~")
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

    def addFilter(self, filter):
        self._logger.addFilter(filter)

    def removeFilter(self, filter):
        self._logger.removeFilter(filter)

    def removeHandler(self, handler):
        self._logger.removeHandler(handler)

    @property
    def INFO_length(self):
        return self._formatter.INFO_length

    def log(self, level, msg, *args, **kwargs):
        self._logger.log(level, msg, *args, **kwargs)

    def debug(self, message):
        self._logger.debug(message)

    def info(self, message):
        self._logger.info(message)

    def warning(self, message):
        self._logger.warning(message)

    def error(self, message):
        self._logger.error(message)

    def critical(self, message):
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

        self.info(f"Running on ~<[{device_name}]>~ with backend ~<{backend}>~. Device memory: ~<{total_mem:.2f}>~ GB.")

        msg_options = ", ".join(
            f"{f'{emoji} ' if is_decorated else ''}{name}: ~<{val}>~"
            for emoji, name, val in (
                ("🔖", "version", gs.__version__),
                ("🎨", "theme", style.theme.name),
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

    @property
    def handler(self):
        return self._handler

    @property
    def level(self):
        return self._logger.level

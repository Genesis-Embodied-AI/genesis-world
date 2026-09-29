import sys
import logging
import threading
from contextlib import contextmanager

from genesis.styles import colors, formats

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

    def colored_fmt(self, color):
        self.last_color = color
        return f"{color}[Genesis] [{self.TIME}] [{self.LEVEL}] {self.MESSAGE}{formats.RESET}"

    def extra_fmt(self, msg):
        msg = msg.replace("~~~~<", colors.MINT + formats.BOLD + formats.ITALIC)
        msg = msg.replace("~~~<", colors.MINT + formats.ITALIC)
        msg = msg.replace("~~<", colors.MINT + formats.UNDERLINE)
        msg = msg.replace("~<", colors.MINT)

        msg = msg.replace(">~~~~", formats.RESET + self.last_color)
        msg = msg.replace(">~~~", formats.RESET + self.last_color)
        msg = msg.replace(">~~", formats.RESET + self.last_color)
        msg = msg.replace(">~", formats.RESET + self.last_color)

        return msg

    def format(self, record):
        log_fmt = self.colored_fmt(self.mapping.get(record.levelno))
        formatter = logging.Formatter(log_fmt, datefmt=self.DATE_FORMAT)
        msg = self.extra_fmt(formatter.format(record))
        self.last_output = msg
        return msg


class Logger:
    def __init__(self, logging_level, verbose_time):
        if isinstance(logging_level, str):
            logging_level = logging_level.upper()

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

    def raw(self, message):
        self._stream.write(self._formatter.extra_fmt(message))
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

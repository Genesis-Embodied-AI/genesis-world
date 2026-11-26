import logging

from genesis.styles import colors, formats, style


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

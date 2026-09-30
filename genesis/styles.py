import logging
import re

from genesis.constants import IntEnum


class THEME(IntEnum):
    """Theme of the text Genesis prints.

    'dark' and 'light' color the text for a terminal of that background. 'raw' prints plain text with a compact prefix
    and no decoration, for log files, continuous integration and coding agents, at the cost of the visual cues.
    """

    dark = 0
    light = 1
    raw = 2


class STYLE:
    """Theme of the text Genesis prints, set by the logger, and the prefix of the log records it implies."""

    def __init__(self) -> None:
        self.theme: THEME | None = None

    @property
    def is_colored(self):
        return self.theme in (THEME.dark, THEME.light)

    def prefix(self, color, time, level, levelno):
        match self.theme:
            case THEME.raw:
                # Almost every record is INFO, so only the other levels are named.
                return f"[Genesis {time}{'' if levelno == logging.INFO else f' {level}'}] "
            case _:
                return f"{color}[Genesis] [{time}] [{level}] "

    def markup(self, msg, color):
        """Render the emphasis markup of a message in the theme, with 'color' the color of the text around it.

        The markup nests four levels of emphasis, from '~<...>~' to '~~~~<...>~~~~'. The raw theme renders the enclosed
        text as is.
        """
        msg = msg.replace("~~~~<", colors.MINT + formats.BOLD + formats.ITALIC)
        msg = msg.replace("~~~<", colors.MINT + formats.ITALIC)
        msg = msg.replace("~~<", colors.MINT + formats.UNDERLINE)
        msg = msg.replace("~<", colors.MINT)

        msg = msg.replace(">~~~~", formats.RESET + color)
        msg = msg.replace(">~~~", formats.RESET + color)
        msg = msg.replace(">~~", formats.RESET + color)
        msg = msg.replace(">~", formats.RESET + color)

        return msg


class COLORS:
    # Reference:
    # https://talyian.github.io/ansicolors/
    # https://bixense.com/clicolors/
    def __init__(self) -> None:
        pass

    @property
    def GREEN(self):
        match style.theme:
            case THEME.dark:
                return "\x1b[38;5;119m"
            case THEME.light:
                return "\x1b[38;5;2m"
            case _:
                return ""

    @property
    def BLUE(self):
        match style.theme:
            case THEME.dark:
                return "\x1b[38;5;159m"
            case THEME.light:
                return "\x1b[38;5;17m"
            case _:
                return ""

    @property
    def YELLOW(self):
        match style.theme:
            case THEME.dark:
                return "\x1b[38;5;226m"
            case THEME.light:
                return "\x1b[38;5;3m"
            case _:
                return ""

    @property
    def RED(self):
        match style.theme:
            case THEME.dark:
                return "\x1b[38;5;9m"
            case THEME.light:
                return "\x1b[38;5;1m"
            case _:
                return ""

    @property
    def CORN(self):
        match style.theme:
            case THEME.dark:
                return "\x1b[38;5;11m"
            case THEME.light:
                return "\x1b[38;5;178m"
            case _:
                return ""

    @property
    def GRAY(self):
        match style.theme:
            case THEME.dark:
                return "\x1b[38;5;247m"
            case THEME.light:
                return "\x1b[38;5;239m"
            case _:
                return ""

    @property
    def MINT(self):
        match style.theme:
            case THEME.dark:
                return "\x1b[38;5;121m"
            case THEME.light:
                return "\x1b[38;5;23m"
            case _:
                return ""


class FORMATS:
    def __init__(self) -> None:
        pass

    @property
    def BOLD(self):
        return "\x1b[1m" if style.is_colored else ""

    @property
    def ITALIC(self):
        return "\x1b[3m" if style.is_colored else ""

    @property
    def UNDERLINE(self):
        return "\x1b[4m" if style.is_colored else ""

    @property
    def RESET(self):
        return "\x1b[0m" if style.is_colored else ""


def styless(text):
    pattern = re.compile(r"\x1b\[(\d+)(?:;\d+)*m")
    return pattern.sub("", text)


style = STYLE()
colors = COLORS()
formats = FORMATS()

import enum
import logging
import re


class THEME(str, enum.Enum):
    """Theme of the text Genesis prints.

    'dark' and 'light' color the text for a terminal of that background. 'raw' prints plain text with a compact prefix
    and no decoration, for log files, continuous integration and coding agents, at the cost of the visual cues.
    """

    dark = "dark"
    light = "light"
    raw = "raw"


# Non-greedy, so that the text of a markup may end with a '>' of its own (e.g. '~type<<gs.morphs.Box>>~').
MARKUP_PATTERN = re.compile(r"~(?P<kind>name|uid|idx|value|type|path|title|frame)<(?P<text>.*?)>~")


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
        """Render the markup of a message in the theme, with 'color' the color of the text around it.

        The markup '~kind<...>~' tags the enclosed text with its kind: 'name' for the name of an object, 'uid' for its
        unique identifier, 'idx' for an index, 'value' for any other value, 'type' for the name of a class, 'path' for a
        file or a directory, 'title' and 'frame' for the banner. The colored themes highlight the enclosed text, in
        italic for names and uids and in bold italic for titles. The raw theme quotes names and prints anything else as
        is.
        """
        return MARKUP_PATTERN.sub(lambda match: self.render(match["kind"], match["text"], color), msg)

    def render(self, kind, text, color):
        """Render the text of a given markup kind in the theme, with 'color' the color of the text around it."""
        match self.theme, kind:
            case THEME.raw, "name":
                return f"'{text}'"
            case THEME.raw, _:
                return text
            case _, "name" | "uid":
                return f"{colors.MINT}{formats.ITALIC}{text}{formats.RESET}{color}"
            case _, "title":
                return f"{colors.MINT}{formats.BOLD}{formats.ITALIC}{text}{formats.RESET}{color}"
            case _:
                return f"{colors.MINT}{text}{formats.RESET}{color}"


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

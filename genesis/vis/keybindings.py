from collections.abc import Callable
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

import genesis as gs


class LabeledIntEnum(IntEnum):
    def __new__(cls, value, label):
        obj = int.__new__(cls, value)
        obj._value_ = value
        obj._label = label
        return obj

    def __str__(self) -> str:
        return self._label


class Key(LabeledIntEnum):
    """
    Key codes for keyboard keys.

    These are compatible with the pyglet key codes.
    https://github.com/pyglet/pyglet/blob/master/pyglet/window/key.py
    """

    # fmt: off

    # ASCII commands
    BACKSPACE     = 0xff08, "backspace"
    TAB           = 0xff09, "tab"
    LINEFEED      = 0xff0a, "linefeed"
    CLEAR         = 0xff0b, "clear"
    RETURN        = 0xff0d, "return"
    ENTER         = 0xff0d, "enter" # synonym
    PAUSE         = 0xff13, "pause"
    SCROLLLOCK    = 0xff14, "scrolllock"
    SYSREQ        = 0xff15, "sysreq"
    ESCAPE        = 0xff1b, "escape"

    # Cursor control and motion
    HOME          = 0xff50, "home"
    LEFT          = 0xff51, "left"
    UP            = 0xff52, "up"
    RIGHT         = 0xff53, "right"
    DOWN          = 0xff54, "down"
    PAGEUP        = 0xff55, "pageup"
    PAGEDOWN      = 0xff56, "pagedown"
    END           = 0xff57, "end"
    BEGIN         = 0xff58, "begin"

    # Misc functions
    DELETE        = 0xffff, "delete"
    SELECT        = 0xff60, "select"
    PRINT         = 0xff61, "print"
    EXECUTE       = 0xff62, "execute"
    INSERT        = 0xff63, "insert"
    UNDO          = 0xff65, "undo"
    REDO          = 0xff66, "redo"
    MENU          = 0xff67, "menu"
    FIND          = 0xff68, "find"
    CANCEL        = 0xff69, "cancel"
    HELP          = 0xff6a, "help"
    BREAK         = 0xff6b, "break"
    MODESWITCH    = 0xff7e, "modeswitch"
    SCRIPTSWITCH  = 0xff7e, "scriptswitch"
    FUNCTION      = 0xffd2, "function"

    # Number pad
    NUMLOCK       = 0xff7f, "numlock"
    NUM_SPACE     = 0xff80, "num_space"
    NUM_TAB       = 0xff89, "num_tab"
    NUM_ENTER     = 0xff8d, "num_enter"
    NUM_F1        = 0xff91, "num_f1"
    NUM_F2        = 0xff92, "num_f2"
    NUM_F3        = 0xff93, "num_f3"
    NUM_F4        = 0xff94, "num_f4"
    NUM_HOME      = 0xff95, "num_home"
    NUM_LEFT      = 0xff96, "num_left"
    NUM_UP        = 0xff97, "num_up"
    NUM_RIGHT     = 0xff98, "num_right"
    NUM_DOWN      = 0xff99, "num_down"
    NUM_PRIOR     = 0xff9a, "num_prior"
    NUM_PAGE_UP   = 0xff9a, "num_page_up"
    NUM_NEXT      = 0xff9b, "num_next"
    NUM_PAGE_DOWN = 0xff9b, "num_page_down"
    NUM_END       = 0xff9c, "num_end"
    NUM_BEGIN     = 0xff9d, "num_begin"
    NUM_INSERT    = 0xff9e, "num_insert"
    NUM_DELETE    = 0xff9f, "num_delete"
    NUM_EQUAL     = 0xffbd, "num_equal"
    NUM_MULTIPLY  = 0xffaa, "num_multiply"
    NUM_ADD       = 0xffab, "num_add"
    NUM_SEPARATOR = 0xffac, "num_separator"
    NUM_SUBTRACT  = 0xffad, "num_subtract"
    NUM_DECIMAL   = 0xffae, "num_decimal"
    NUM_DIVIDE    = 0xffaf, "num_divide"

    NUM_0         = 0xffb0, "num_0"
    NUM_1         = 0xffb1, "num_1"
    NUM_2         = 0xffb2, "num_2"
    NUM_3         = 0xffb3, "num_3"
    NUM_4         = 0xffb4, "num_4"
    NUM_5         = 0xffb5, "num_5"
    NUM_6         = 0xffb6, "num_6"
    NUM_7         = 0xffb7, "num_7"
    NUM_8         = 0xffb8, "num_8"
    NUM_9         = 0xffb9, "num_9"

    # Function keys
    F1            = 0xffbe, "f1"
    F2            = 0xffbf, "f2"
    F3            = 0xffc0, "f3"
    F4            = 0xffc1, "f4"
    F5            = 0xffc2, "f5"
    F6            = 0xffc3, "f6"
    F7            = 0xffc4, "f7"
    F8            = 0xffc5, "f8"
    F9            = 0xffc6, "f9"
    F10           = 0xffc7, "f10"
    F11           = 0xffc8, "f11"
    F12           = 0xffc9, "f12"
    F13           = 0xffca, "f13"
    F14           = 0xffcb, "f14"
    F15           = 0xffcc, "f15"
    F16           = 0xffcd, "f16"
    F17           = 0xffce, "f17"
    F18           = 0xffcf, "f18"
    F19           = 0xffd0, "f19"
    F20           = 0xffd1, "f20"
    F21           = 0xffd2, "f21"
    F22           = 0xffd3, "f22"
    F23           = 0xffd4, "f23"
    F24           = 0xffd5, "f24"

    # Modifiers
    LSHIFT        = 0xffe1, "left_shift"
    RSHIFT        = 0xffe2, "right_shift"
    LCTRL         = 0xffe3, "left_ctrl"
    RCTRL         = 0xffe4, "right_ctrl"
    CAPSLOCK      = 0xffe5, "capslock"
    LMETA         = 0xffe7, "left_meta"
    RMETA         = 0xffe8, "right_meta"
    LALT          = 0xffe9, "left_alt"
    RALT          = 0xffea, "right_alt"
    LWINDOWS      = 0xffeb, "left_windows"
    RWINDOWS      = 0xffec, "right_windows"
    LCOMMAND      = 0xffed, "left_command"
    RCOMMAND      = 0xffee, "right_command"
    LOPTION       = 0xffef, "left_option"
    ROPTION       = 0xfff0, "right_option"

    # Latin-1
    SPACE         = 0x020, "space"
    EXCLAMATION   = 0x021, "!"
    DOUBLEQUOTE   = 0x022, "\""
    HASH          = 0x023, "#"
    POUND         = 0x023, "#"  # synonym
    DOLLAR        = 0x024, "$"
    PERCENT       = 0x025, "%"
    AMPERSAND     = 0x026, "&"
    APOSTROPHE    = 0x027, "'"
    PARENLEFT     = 0x028, "("
    PARENRIGHT    = 0x029, ")"
    ASTERISK      = 0x02a, "*"
    PLUS          = 0x02b, "+"
    COMMA         = 0x02c, ","
    MINUS         = 0x02d, "-"
    PERIOD        = 0x02e, "."
    SLASH         = 0x02f, "/"
    _0            = 0x030, "0"
    _1            = 0x031, "1"
    _2            = 0x032, "2"
    _3            = 0x033, "3"
    _4            = 0x034, "4"
    _5            = 0x035, "5"
    _6            = 0x036, "6"
    _7            = 0x037, "7"
    _8            = 0x038, "8"
    _9            = 0x039, "9"
    COLON         = 0x03a, ":"
    SEMICOLON     = 0x03b, ";"
    LESS          = 0x03c, "<"
    EQUAL         = 0x03d, "="
    GREATER       = 0x03e, ">"
    QUESTION      = 0x03f, "?"
    AT            = 0x040, "@"
    BRACKETLEFT   = 0x05b, "["
    BACKSLASH     = 0x05c, "\\"
    BRACKETRIGHT  = 0x05d, "]"
    ASCIICIRCUM   = 0x05e, "^"
    UNDERSCORE    = 0x05f, "_"
    GRAVE         = 0x060, "`"
    QUOTELEFT     = 0x060, "`"
    A             = 0x061, "a"
    B             = 0x062, "b"
    C             = 0x063, "c"
    D             = 0x064, "d"
    E             = 0x065, "e"
    F             = 0x066, "f"
    G             = 0x067, "g"
    H             = 0x068, "h"
    I             = 0x069, "i"
    J             = 0x06a, "j"
    K             = 0x06b, "k"
    L             = 0x06c, "l"
    M             = 0x06d, "m"
    N             = 0x06e, "n"
    O             = 0x06f, "o"
    P             = 0x070, "p"
    Q             = 0x071, "q"
    R             = 0x072, "r"
    S             = 0x073, "s"
    T             = 0x074, "t"
    U             = 0x075, "u"
    V             = 0x076, "v"
    W             = 0x077, "w"
    X             = 0x078, "x"
    Y             = 0x079, "y"
    Z             = 0x07a, "z"
    BRACELEFT     = 0x07b, "{"
    BAR           = 0x07c, "|"
    BRACERIGHT    = 0x07d, "}"
    ASCIITILDE    = 0x07e, "~"
    # fmt: on


class KeyMod(LabeledIntEnum):
    # fmt: off
    SHIFT      = 1 << 0, "shift"
    CTRL       = 1 << 1, "ctrl"
    ALT        = 1 << 2, "alt"
    CAPSLOCK   = 1 << 3, "capslock"
    NUMLOCK    = 1 << 4, "numlock"
    WINDOWS    = 1 << 5, "windows"
    COMMAND    = 1 << 6, "command"
    OPTION     = 1 << 7, "option"
    SCROLLLOCK = 1 << 8, "scrolllock"
    FUNCTION   = 1 << 9, "function"
    # fmt: on


class KeyAction(LabeledIntEnum):
    """
    PRESS is triggered once when the key is initially pressed down.
    HOLD is triggered repeatedly while the key is held down.
    RELEASE is triggered once when the key is released.
    """

    PRESS = 0, "press"
    HOLD = 1, "hold"
    RELEASE = 2, "release"


class MouseButton(LabeledIntEnum):
    LEFT = 1 << 0, "left"
    MIDDLE = 1 << 1, "middle"
    RIGHT = 1 << 2, "right"


def get_key_hash(key_code: int, modifiers: int | None, action: KeyAction) -> int:
    """Generate a unique hash for a key combination.

    Parameters
    ----------
    key_code: int
        The key code as an int.
    modifiers : int | None
        The modifier keys pressed, as an int with bit flags, or None to ignore modifiers.
    action : KeyAction
        The type of key action (press, hold, release).

    Returns
    -------
    int
        A unique hash for this key combination.
    """
    return hash((key_code, modifiers, action))


@dataclass
class Keybind:
    """
    A keybinding with an associated callback.

    Parameters
    ----------
    name : str
        The name of the keybind.
    key : Key
        The key code for the keybind.
    key_action : KeyAction
        The type of key action (press, hold, release).
    key_mods : tuple[KeyMod] | None
        The modifier keys required for the keybind. If None, modifiers are ignored.
    callback : Callable[[], None] | None
        The function to call when the keybind is activated.
    args : tuple
        Positional arguments to pass to the callback.
    kwargs : dict
        Keyword arguments to pass to the callback.
    protected : bool
        Whether this keybind can be overwritten without being deleted first.
    allow_overload : bool
        Whether this key can be shared with keybinds of a different action type (e.g. PRESS
        and RELEASE on the same key). When False, any other keybind using the same key is
        treated as a conflict: replaced if ``overwrite=True`` and the other keybind is not
        protected, otherwise an error is raised.
    """

    name: str
    key: Key
    key_action: KeyAction = KeyAction.RELEASE
    key_mods: tuple[KeyMod] | None = None
    callback: Callable[..., None] | None = None
    args: tuple[Any, ...] = ()
    kwargs: dict = field(default_factory=dict)
    protected: bool = False
    allow_overload: bool = True

    _modifiers: int | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        if self.key_mods is not None:
            self._modifiers = 0
            for mod in self.key_mods:
                self._modifiers |= mod
        if self.kwargs is None:
            self.kwargs = {}

    def __hash__(self) -> int:
        """Generate a unique hash for the keybind based on key code and modifiers."""
        return get_key_hash(self.key, self._modifiers, self.key_action)


class Keybindings:
    def __init__(self, keybinds: tuple[Keybind, ...] = ()):
        self._keybinds: dict[str, Keybind] = {}
        for kb in keybinds:
            self._keybinds[kb.name] = kb

    def register(self, keybind: Keybind, overwrite: bool) -> None:
        for name, other_keybind in list(self._keybinds.items()):
            if hash(keybind) == hash(other_keybind):
                msg = (
                    f"Key '{keybind.key}' already assigned to '{other_keybind.name}' with overlapping action "
                    f"'{other_keybind.key_action}'."
                )
            elif (not keybind.allow_overload or not other_keybind.allow_overload) and keybind.key == other_keybind.key:
                msg = (
                    f"Key '{keybind.key}' already assigned to '{other_keybind.name}' with action "
                    f"'{other_keybind.key_action}'."
                )
            elif keybind.name == name:
                msg = (
                    f"Name '{name}' is already assigned to another key '{other_keybind.key}' with action "
                    f"'{other_keybind.key_action}'."
                )
            else:
                continue
            if not overwrite or other_keybind.protected:
                if other_keybind.protected:
                    msg = f"{msg} This keybind is protected and cannot be overwritten."
                raise ValueError(msg)
            gs.logger.warning(f"{msg}. Overwriting keybind.")
            self.remove(name)
        self._keybinds[keybind.name] = keybind

    def remove(self, name: str) -> None:
        if name not in self._keybinds:
            raise ValueError(f"No keybind found with name '{name}'.")
        self._keybinds.pop(name)

    def rebind(
        self,
        name: str,
        new_key: Key | None,
        new_key_mods: tuple[KeyMod] | None,
        new_key_action: KeyAction | None = None,
    ) -> None:
        if name not in self._keybinds:
            raise ValueError(f"No keybind found with name '{name}'.")
        kb = self._keybinds[name]
        self._keybinds[name] = Keybind(
            name=kb.name,
            key=new_key or kb.key,
            key_action=new_key_action or kb.key_action,
            key_mods=new_key_mods,
            callback=kb.callback,
            args=kb.args,
            kwargs=kb.kwargs,
        )

    def get(self, key: int, modifiers: int, key_action: KeyAction) -> Keybind | None:
        key_hash = get_key_hash(key, modifiers, key_action)
        for keybind in self._keybinds.values():
            if hash(keybind) == key_hash:
                return keybind

        # Try ignoring modifiers (for keybinds where modifiers=None)
        key_hash_no_mods = get_key_hash(key, None, key_action)
        for keybind in self._keybinds.values():
            if hash(keybind) == key_hash_no_mods:
                return keybind

        return None

    def get_by_name(self, name: str) -> Keybind | None:
        return self._keybinds.get(name)

    def __len__(self) -> int:
        return len(self._keybinds)

    @property
    def keybinds(self) -> tuple[Keybind, ...]:
        """Return a tuple of all registered Keybinds."""
        return tuple(self._keybinds.values())

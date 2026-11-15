import weakref
from typing import TYPE_CHECKING

import genesis as gs

if TYPE_CHECKING:
    from genesis.engine.scene import Scene


# The messages each scene has logged, dropped with the scene. Keyed by the scene itself, since an id gets reused.
_SCENES_WARNINGS: "weakref.WeakKeyDictionary[Scene, set[str]]" = weakref.WeakKeyDictionary()


def warn_once(scene: "Scene", message: str) -> None:
    """Log a warning the first time a scene asks for it.

    The record of the warnings already logged for 'scene' is released with it, so another scene logs them again.
    """
    messages = _SCENES_WARNINGS.setdefault(scene, set())
    if message not in messages:
        messages.add(message)
        gs.logger.warning(message)

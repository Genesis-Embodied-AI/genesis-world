import weakref

import genesis as gs

# The messages each object has logged, dropped with the object. Keyed by the object itself, since an id gets reused.
_OWNERS_WARNINGS: weakref.WeakKeyDictionary[object, set[str]] = weakref.WeakKeyDictionary()


def warn_once(owner: object, message: str) -> None:
    """Log a warning the first time an object asks for it.

    The caller passes itself as 'owner'. The record of the warnings 'owner' has logged is released with it, so another
    object logs them again.
    """
    messages = _OWNERS_WARNINGS.setdefault(owner, set())
    if message not in messages:
        messages.add(message)
        gs.logger.warning(message)

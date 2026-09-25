from ..base import EntityT, MaterialOptions


class Base(MaterialOptions[EntityT]):
    """
    The base class of PBD materials.

    Note
    ----
    This class should *not* be instantiated directly.
    """

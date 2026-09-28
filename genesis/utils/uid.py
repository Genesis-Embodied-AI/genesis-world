import uuid

from genesis.repr_base import RBC


class UID(RBC):
    def __init__(self) -> None:
        self.uid = uuid.uuid4().hex

    def _repr_brief(self):
        return self.__repr__()

    def __repr__(self) -> str:
        return f"{self.__repr_name__()}('{self.full()}')"

    def __format__(self, format_spec) -> str:
        return f"<{self.short()}>"

    def __str__(self) -> str:
        return self.uid

    def full(self) -> str:
        return f"{self.uid[:7]}-{self.uid[7:]}"

    def short(self) -> str:
        return self.uid[:7]

    def match(self, other: str, short_only: bool = False) -> bool:
        """
        Check if this UID matches the given string.

        Parameters
        ----------
        other : str
            The string to compare against.
        short_only : bool, optional
            If True, only compare the short (7-character) version of the UID.
            If False (default), compare the full UID.

        Returns
        -------
        bool
            True if the UID matches the given string.
        """
        if short_only:
            return self.short() == other
        # repr and full() insert a hyphen after 7 characters. The stored
        # hex does not. Both strings identify this UID.
        return other == self.uid or other == self.full()

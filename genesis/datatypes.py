import logging
import os
from typing import SupportsIndex, get_args, overload

from pydantic import GetCoreSchemaHandler
from pydantic_core import core_schema

import genesis as gs
from genesis.repr_base import RBC
from genesis.styles import colors, formats, styless


LOGGER = logging.getLogger(__name__)


class List[T: RBC](RBC, list[T]):
    """
    Custom list with more informative repr.

    Elements in the list should also inherit from RBC.
    """

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: type["List[T]"], handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        (item_type,) = get_args(source_type) or (RBC,)
        return core_schema.no_info_after_validator_function(
            cls,  # wraps the validated list in gs.List(...)
            core_schema.list_schema(handler.generate_schema(item_type)),
        )

    @overload
    def __getitem__(self, i: SupportsIndex, /) -> T: ...

    @overload
    def __getitem__(self, s: slice, /) -> list[T]: ...

    def __getitem__(self, item):
        result = super().__getitem__(item)
        if isinstance(item, slice):
            return self.__class__(result)
        return result

    def _repr_elem(self, elem: T, common_length: int = 0) -> str:
        spaces = common_length - len(elem.__repr_name__())
        repr_str = " " * spaces + elem._repr_brief()
        return repr_str

    def _repr_elem_colorized(self, elem: T, common_length: int = 0) -> str:
        content = self._repr_elem(elem, common_length)
        idx = content.find(">")
        formatted_content = f"{colors.BLUE}{formats.ITALIC}{content[: idx + 1]}{formats.RESET}{content[idx + 1 :]}"
        idx = formatted_content.find(":")
        if idx >= 0:
            formatted_content = f"{formatted_content[:idx]}{colors.GRAY}:{colors.MINT}{formatted_content[idx + 1 :]}"
        formatted_content += formats.RESET
        return formatted_content

    def _repr_brief(self) -> str:
        repr_str = f"{self.__repr_name__()}(len={len(self)}, ["
        if len(self) >= 15:
            repr_str += "\n"
            for element in self[:9]:
                repr_str += f"    {self._repr_elem(element)},\n"
            repr_str += "    ...\n"
            for element in self[-1:]:
                repr_str += f"    {self._repr_elem(element)},\n"
        elif self:
            repr_str += "\n"
            for element in self:
                repr_str += f"    {self._repr_elem(element)},\n"
        repr_str += "])"
        return repr_str

    def __repr__colorized__(self) -> str:
        repr_str = f"{colors.BLUE}{self.__repr_name__()}(len={colors.MINT}{formats.UNDERLINE}{len(self)}{formats.RESET}{colors.BLUE}, ["

        is_verbose = getattr(gs, "logger", LOGGER).level <= logging.DEBUG
        common_length = max((len(obj.__repr_name__()) for obj in self), default=0)

        if not self:
            repr_str += f"{colors.BLUE}])"
        elif len(self) < 15 or is_verbose:
            repr_str += "\n"
            for element in self:
                repr_str += f"    {self._repr_elem_colorized(element, common_length)}{colors.GRAY},{formats.RESET}\n"
            repr_str += f"{colors.BLUE}]){formats.RESET}"
        else:
            repr_str += "\n"
            for element in self[:9]:
                repr_str += f"    {self._repr_elem_colorized(element, common_length)}{colors.GRAY},{formats.RESET}\n"
            repr_str += f"    {colors.GRAY}...{formats.RESET}\n"
            for element in self[-1:]:
                repr_str += f"    {self._repr_elem_colorized(element, common_length)}{colors.GRAY},{formats.RESET}\n"
            repr_str += f"{colors.BLUE}]){formats.RESET}"

        min_len = 15
        if self:
            first_line = styless(repr_str.split("\n")[1])

            common_ancestors = set.intersection(*[set(type(obj).__mro__) for obj in self])
            if common_ancestors:
                common_class_name = next(
                    base_cls.__repr_name__(self[0])
                    for base_cls in type(self[0]).__mro__
                    if base_cls in common_ancestors
                )
                header = f"{self.__repr_name__()} of {common_class_name}"
            else:
                header = f"{self.__repr_name__()}"
            header_len = len(first_line)
            line_len = max(min_len, header_len - len(header) - 2)
            try:
                columns, _lines = os.get_terminal_size()
                line_len = min(line_len, columns - len(header) - 2)
            except OSError:
                pass
        else:
            first_line = styless(repr_str)
            header = self.__repr_name__()
            line_len = min_len

        left_line_len = line_len // 2
        right_line_len = line_len - left_line_len

        repr_str = (
            f"{colors.CORN}{'─' * left_line_len} {formats.BOLD}{formats.ITALIC}{header}{formats.RESET} {colors.CORN}{'─' * right_line_len}{formats.RESET}\n"
            + repr_str
        )

        return repr_str

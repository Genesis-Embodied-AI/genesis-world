from typing import Any
from typing_extensions import Self

from pydantic import BaseModel, ConfigDict, ValidationError

import genesis as gs
import genesis.utils.repr as ru
from genesis.repr_base import RBC
from genesis.styles import colors, formats


class Options(RBC, BaseModel):
    """
    This is the base class for all `gs.options.*` classes. An `Options` object is a group of parameters for setting a
    specific component in the scene.

    Note
    ----
    This class should *not* be instantiated directly.

    Tip
    ----
    We build multiple classes based on this concept throughout Genesis, such as `gs.options.morphs`, `gs.renderers`,
    `gs.surfaces`, and `gs.textures`. Note that some of them, although inheriting from `Options`, are accessible
    directly under the `gs` namespace for convenience.
    """

    model_config = ConfigDict(strict=True, extra="forbid", validate_default=True)

    def __init__(self, /, **data: Any) -> None:
        # format pydantic error message to be more informative yet concise
        try:
            super().__init__(**data)
        except ValidationError as e:
            self._validation_error(e)

    def __setattr__(self, name: str, value) -> None:
        try:
            super().__setattr__(name, value)
        except ValidationError as e:
            self._validation_error(e)

    def _validation_error(self, exception: ValidationError) -> None:
        traces = [f"Validation error for {RBC.__repr_name__(self)}:"]

        # Aggregate invalid type errors
        err_invalid_infos = {}
        # Track the required length of fixed-size sequences reported as too short. Pydantic emits one 'missing'
        # error per absent element (e.g. loc ['pos', 2] for a 2-tuple given a 3-tuple field), so the maximum
        # missing index plus one yields the expected length.
        err_too_short_lengths = {}
        for err in exception.errors():
            err_type, (attr, *index), msg, value = err["type"], err["loc"], err["msg"], err.get("input")
            if msg.startswith("Input should be a valid "):
                info = err_invalid_infos.setdefault(attr, {"type": {}})
                info["type"].setdefault(tuple(index), []).append(msg[24:])
            elif attr in err_invalid_infos and err_type == "too_short":
                err_invalid_infos[attr]["value"] = value
            elif err_type == "missing" and index:
                err_too_short_lengths[attr] = max(err_too_short_lengths.get(attr, 0), index[0] + 1)

        # Format all errors without early stopping
        filtered_attrs = set()
        for err in exception.errors():
            err_type, (attr, *index), msg, value = err["type"], err["loc"], err["msg"], err.get("input")
            attr_indexed = f"{attr}{index}" if index else attr

            if attr in filtered_attrs:
                continue
            if err_type == "extra_forbidden":
                trace = f"Unrecognized attribute '{attr}'."
            elif err_type in ("frozen_instance", "frozen_field"):
                trace = f"{msg[0].lower()}{msg[1:]}."
            elif err_type == "missing" and attr in err_too_short_lengths:
                filtered_attrs.add(attr)
                trace = (
                    f"Invalid attribute '{attr}': should be a sequence of length "
                    f"{err_too_short_lengths[attr]}. Got {repr(value)}."
                )
            elif err_type == "missing":
                trace = f"Missing attribute '{attr}'."
            elif attr in err_invalid_infos:
                filtered_attrs.add(attr)
                info = err_invalid_infos[attr]
                value = info.get("value", value)
                if len(info["type"]) == 1:
                    ((indices, (candidate_type_msg,)),) = info["type"].items()
                    if indices:
                        attr = f"{attr}{list(indices)}"
                    trace = f"Invalid attribute '{attr}': should be a valid {candidate_type_msg}. Got {repr(value)}."
                else:
                    indices, candidate_types = zip(*info["type"].items())
                    (*candidate_types, last_candidate_type) = set(e for types in candidate_types for e in types)
                    attr = f"{attr}{{{'|'.join(map(str, map(list, indices)))}}}"
                    if candidate_types:
                        candidate_type_msg = f"{', '.join(candidate_types)}, or {last_candidate_type}"
                    else:
                        candidate_type_msg = last_candidate_type
                    trace = f"Invalid attribute '{attr}': should be valid {candidate_type_msg}s. Got {repr(value)}."
            else:
                trace = f"Invalid attribute '{attr_indexed}': {msg[0].lower()}{msg[1:]}. Got {repr(value)}."
            traces.append(trace)

        # Gather all error messages as once
        if len(traces) > 2:
            trace_msg = "\n".join(f"* {msg}" for msg in traces)
        else:
            trace_msg = " ".join(traces)

        gs.raise_exception_from(trace_msg, None)

    def model_copy_from(self, other: BaseModel, override: bool = False) -> Self:
        # 'dict(self)' keeps a nested option object intact, where 'model_dump' recurses it into a plain dict. The
        # construction below performs no validation, so it would keep such a dict as a dict.
        self_fields = set(self.__class__.model_fields)
        other_dump = {k: v for k, v in dict(other).items() if k in self_fields}
        self_dump = dict(self)
        # Do not include default None
        for field, value in tuple(self_dump.items()):
            if value is None and field not in self.model_fields_set:
                del self_dump[field]
        merged = {**self_dump, **other_dump} if override else {**other_dump, **self_dump}
        # Constructing the copy would mark every merged field as set, including the ones filled in from 'other',
        # while 'model_fields_set' tells a value the user gave from a default. 'model_construct' takes that set
        # explicitly and still runs 'model_post_init'. It performs no validation, which the merged values have no need
        # of: they were validated, and the defaults among them resolved, as 'self' and 'other' were built.
        return self.__class__.model_construct(_fields_set=set(self.model_fields_set), **merged)

    def __repr__colorized__(self) -> str:
        repr_items = tuple(self.__repr_args__())
        max_attr_len = max((len(attr) for attr, _value in repr_items if attr is not None), default=0)

        repr_str = f"{colors.CORN}{'─' * (max_attr_len + 3)} {formats.BOLD}{formats.ITALIC}{self.__repr_name__()}{formats.RESET} {colors.CORN}{'─' * (max_attr_len + 3)}\n"

        for attr, value in repr_items:
            formatted_str = f"{colors.BLUE}'{attr}'{formats.RESET}"

            content = ru.brief(value)
            idx = content.find(">")
            formatted_content = f"{colors.MINT}{formats.ITALIC}{content[: idx + 1]}{formats.RESET}{colors.MINT}{content[idx + 1 :]}{formats.RESET}"
            # in case it's multi-line
            formatted_content = formatted_content.replace("\n", "\n" + " " * (max_attr_len + 4))

            repr_str += f"{formatted_str:>{max_attr_len + 17}}{colors.GRAY}:{formats.RESET} {formatted_content}\n"

        return repr_str

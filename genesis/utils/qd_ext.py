"""Genesis extensions and documented workarounds for Quadrants.

Graph Pipelines use GlobalDataContext and may keep Actions/Schedules as plain Python objects. Systems remain
``@qd.data_oriented`` for now because legacy non-Pipeline member kernels discover data through paths such as
``Entity -> solver -> System -> Data`` and have no context argument yet.
"""

import dataclasses
from typing import Iterable

import quadrants as qd
from quadrants.lang._ndarray import Ndarray


@qd.data_oriented
class GlobalDataContext:
    """Top-level Quadrants parameter-discovery context over an explicit sequence of device-data parameters.

    Quadrants predeclares ndarrays only when they are reachable from a top-level kernel template through a dataclass or
    ``@qd.data_oriented`` object. It does not discover data through arbitrary Python containers or callable-bound
    arguments. Passing this context first lets later code reach the same ndarray objects by identity.
    """

    def __init__(self, data: Iterable[object] = ()) -> None:
        for index, value in enumerate(data):
            if isinstance(value, (tuple, list, dict)):
                raise TypeError("GlobalDataContext parameters must be separate objects, not composite containers.")
            if isinstance(value, type):
                raise TypeError("GlobalDataContext parameters must be instances, not classes.")
            if not (
                isinstance(value, (Ndarray, qd.Tensor))
                or dataclasses.is_dataclass(value)
                or getattr(value, "_data_oriented", False)
            ):
                raise TypeError(
                    "GlobalDataContext expected Tensor, ndarray, dataclass or @qd.data_oriented object, "
                    f"got {type(value).__name__}."
                )
            setattr(self, f"data_{index}", value)


@qd.func(requires_top_level=True)
def invoke_stage_actions(actions: qd.template(), call_args: qd.template()):
    """Invoke frozen StageActions without losing graph-stage boundaries.

    Quadrants does not natively lower an Action collection. A runtime loop would demote the top-level loops of each
    ``requires_top_level`` function, so the Action tuple must be unrolled with ``qd.static``. Parallel Action
    collections cannot use this helper: ``qd.graph.parallel_context`` is legal only syntactically inside a graph kernel,
    so their static unrolling must remain at that kernel call site.
    """
    for action in qd.static(actions):
        action.invoke(call_args)


def expose_data_sequence(owner: object, name: str, values: Iterable[object]) -> None:
    """Expose composite-held data through named attributes Quadrants can traverse.

    Quadrants' struct-ndarray predeclaration walks dataclass and data-oriented attributes but not list/tuple/dict
    members. Owners that intentionally store a sequence therefore expose each item under a stable named alias.
    """
    for index, value in enumerate(values):
        setattr(owner, f"_qd_{name}_{index}", value)

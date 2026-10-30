from __future__ import annotations

import quadrants as qd

from genesis.utils.data import Data, DataKind, Size, StaticConfig, field, requires

ITEMS = Size("items")


# The structs referenced before they are defined resolve once the first instance is built
class Outer(Data, kind=DataKind.CONSTANT):
    inner: Inner
    shared_info: SharedInfo = requires()
    scratch: Scratch = field(lazy=True)
    values: qd.Tensor = field(float, shape=(ITEMS,))


class Inner(Data, kind=DataKind.CONSTANT, shape=()):
    value: qd.Tensor = field(float)


class SharedInfo(Data, kind=DataKind.CONSTANT, shape=()):
    value: qd.Tensor = field(float)


class Scratch(Data, kind=DataKind.SCRATCH):
    buffer: qd.Tensor = field(float, shape=(ITEMS,))


class HookedInfo(Data, kind=DataKind.CONSTANT, shape=(), into=(f"{__name__}.Outer", "hooked_info")):
    value: qd.Tensor = field(float)


# A static config resolves its annotations when defined, after the configs it holds
class NestedConfig(StaticConfig):
    is_enabled: bool = False


class Config(StaticConfig):
    count: int = 3
    nested: NestedConfig = NestedConfig()

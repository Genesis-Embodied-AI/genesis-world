"""
Cloth material for the graph-native Newton coupling runtime.

Unlike :class:`Cloth`, this material is consumed only by
``genesis.engine.systems``. The existing IPC-coupler material and API remain
unchanged.
"""

from typing import Annotated

from pydantic import Field

from genesis.typing import NonNegativeFloat, PositiveFloat, ValidFloat

from .base import Base


class QCloth(Base):
    """Baraff-Witkin membrane plus quadratic shell bending."""

    E: PositiveFloat = 1e4
    nu: Annotated[ValidFloat, Field(gt=-1.0, lt=0.5)] = 0.3
    rho: PositiveFloat = 200.0
    thickness: PositiveFloat = 1e-3
    shear_modulus: PositiveFloat | None = None
    strain_limit_multiplier: NonNegativeFloat = 100.0
    bending_youngs_modulus: PositiveFloat | None = None

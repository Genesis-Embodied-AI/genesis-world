from typing import TYPE_CHECKING, Annotated, ClassVar

from pydantic import Field

from genesis.constants import STRUCTURE
from genesis.typing import NonNegativeFloat, PositiveFloat, ValidFloat

from ..base import Material

if TYPE_CHECKING:
    from genesis.engine.entities.fem_entity import FEMEntity


class Base(Material["FEMEntity"]):
    """
    The base class of FEM materials.

    Note
    ----
    This class should *not* be instantiated directly.

    Parameters
    ----------
    E : float, optional
        Young's modulus, which controls stiffness. Default is 1e6.
    nu : float, optional
        Poisson ratio, describing the material's volume change under stress. Default is 0.2.
    rho : float, optional
        Material density (kg/m^3). Default is 1000.
    hydroelastic_modulus : float, optional
        Hydroelastic modulus for hydroelastic contact. Default is 1e7.
    friction_mu : float, optional
        Friction coefficient. Default is 0.1.
    contact_resistance : float | None, optional
        IPC contact resistance/stiffness override for this material. ``None`` means
        use the IPC coupler global default (``IPCCouplerOptions.contact_resistance``).
    """

    structure: ClassVar[STRUCTURE] = STRUCTURE.DEFORMABLE

    E: PositiveFloat = 1e6
    nu: Annotated[ValidFloat, Field(gt=-1.0, lt=0.5)] = 0.2
    rho: PositiveFloat = 1000.0
    hydroelastic_modulus: PositiveFloat = 1e7
    friction_mu: NonNegativeFloat = 0.1
    contact_resistance: PositiveFloat | None = None

    @property
    def mu(self) -> float:
        """The second Lame parameter, derived from the Young's modulus and the Poisson ratio."""
        return self.E / (2.0 * (1.0 + self.nu))

    @property
    def lam(self) -> float:
        """The first Lame parameter, derived from the Young's modulus and the Poisson ratio."""
        return self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))

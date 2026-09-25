import platform
import re
import sys
from typing import TYPE_CHECKING, Annotated, Any, Literal

import quadrants as qd
from pydantic import BeforeValidator, Field, PrivateAttr
from pydantic_core import PydanticCustomError

import genesis as gs
from genesis.typing import PositiveFloat, StrictInt, ValidFloat

from ..base import Material

if TYPE_CHECKING:
    from genesis.engine.entities.mpm_entity import MPMEntity

_SAMPLER_PATTERN = re.compile(r"^pbs(-\d+)?$|^random$|^regular$")


def _validate_sampler(v: str) -> str:
    if not isinstance(v, str) or not _SAMPLER_PATTERN.match(v):
        raise PydanticCustomError(
            "invalid_sampler",
            "Input should be 'pbs', 'pbs-<seed>', 'random', or 'regular'",
            {"value": v},
        )
    return v


SamplerType = Annotated[str, BeforeValidator(_validate_sampler)]
DEFAULT_SAMPLER = "pbs" if (sys.platform == "linux" and platform.machine() == "x86_64") else "random"


@qd.data_oriented
class Base(Material["MPMEntity"]):
    """
    The base class of MPM materials.

    Note
    ----
    This class should *not* be instantiated directly.

    Parameters
    ----------
    E : float, optional
        Young's modulus. Default is 1e6.
    nu : float, optional
        Poisson ratio. Default is 0.2.
    rho : float, optional
        Density (kg/m³). Default is 1000.
    lam : float, optional
        The first Lame's parameter. Default is None, computed from E and nu.
    mu : float, optional
        The second Lame's parameter. Default is None, computed from E and nu.
    sampler : str, optional
        Particle sampler ('pbs', 'regular', 'random'). Default is platform-dependent.
    """

    E: PositiveFloat = 1e6
    nu: Annotated[ValidFloat, Field(gt=-1.0, lt=0.5)] = 0.2
    rho: PositiveFloat = 1000.0
    lam: ValidFloat | None = None
    mu: ValidFloat | None = None
    sampler: SamplerType = DEFAULT_SAMPLER

    # Dispatch fields - set by subclass model_post_init, not user-specified. needs_svd flips the solver's forward-mode
    # SVD kernel off when no material in the scene reads U/V/S from SVD(F_tmp) (e.g. non-viscous liquid, neohooken
    # elastic).
    update_F_S_Jp: Any = Field(default=None, exclude=True, repr=False)
    update_stress: Any = Field(default=None, exclude=True, repr=False)
    needs_svd: bool = Field(default=True, exclude=True, repr=False)

    # Auto-generated fields.
    idx: StrictInt | None = Field(default=None, exclude=True)

    # Internal solver state - does not belong in an options class, kept pending larger refactor.
    _default_Jp: float = PrivateAttr(default=1.0)

    def model_post_init(self, context: Any) -> None:
        # Resolve Lame parameters
        if self.mu is None:
            self.mu = self.E / (2.0 * (1.0 + self.nu))
        if self.lam is None:
            self.lam = self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))

        # Set dispatch defaults
        if self.update_F_S_Jp is None:
            self.update_F_S_Jp = self._update_F_S_Jp_noop
        if self.update_stress is None:
            self.update_stress = self._update_stress_default

    @qd.func
    def _update_F_S_Jp_noop(self, J, F_tmp, U, S, V, Jp):
        raise NotImplementedError

    @qd.func
    def _update_stress_default(self, U, S, V, F_tmp, F_new, J, Jp, actu, m_dir):
        stress = 2 * self.mu * (F_new - U @ V.transpose()) @ F_new.transpose() + qd.Matrix.identity(
            gs.qd_float, 3
        ) * self.lam * J * (J - 1)
        return stress

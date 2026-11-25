from typing import TYPE_CHECKING, Annotated, ClassVar

from pydantic import ConfigDict, Field, ValidationInfo, field_validator

import quadrants as qd
from quadrants.lang._quadrants_callable import QuadrantsCallable

import genesis as gs
from genesis.constants import STRUCTURE
from genesis.typing import PositiveFloat, ValidFloat

from ..base import Material, SamplerType

if TYPE_CHECKING:
    from genesis.engine.entities.mpm_entity import MPMEntity


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
        Particle sampler. Defaults to 'pbs', which relaxes the particles to a uniform density inside the shape, at the
        cost of a slower first sampling of each shape, cached afterwards. 'regular' places them on a grid, which leaves
        the surface stair-stepped. 'random' draws them uniformly, which leaves local density fluctuations.
    """

    # The kernel functions of a material are methods, which pydantic would otherwise take for fields
    model_config = ConfigDict(ignored_types=(QuadrantsCallable,))

    structure: ClassVar[STRUCTURE] = STRUCTURE.TEARABLE

    # The plastic volume ratio of a particle at rest
    _default_Jp: ClassVar[float] = 1.0

    E: PositiveFloat = 1e6
    nu: Annotated[ValidFloat, Field(gt=-1.0, lt=0.5)] = 0.2
    rho: PositiveFloat = 1000.0
    lam: ValidFloat | None = None
    mu: ValidFloat | None = None
    sampler: SamplerType = "pbs"

    # The Lame parameters left out derive from 'E' and 'nu', validated before them as they are declared first. Either
    # missing from 'info.data' failed its own validation, which pydantic reports.
    @field_validator("mu")
    @classmethod
    def _resolve_mu(cls, mu: float | None, info: ValidationInfo) -> float | None:
        if mu is None and info.data.keys() >= {"E", "nu"}:
            mu = info.data["E"] / (2.0 * (1.0 + info.data["nu"]))
        return mu

    @field_validator("lam")
    @classmethod
    def _resolve_lam(cls, lam: float | None, info: ValidationInfo) -> float | None:
        if lam is None and info.data.keys() >= {"E", "nu"}:
            E, nu = info.data["E"], info.data["nu"]
            lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        return lam

    @property
    def needs_svd(self) -> bool:
        """Whether the material reads the singular value decomposition of the deformation gradient."""
        return True

    @qd.func
    def update_F_S_Jp(self, J, F_tmp, U, S, V, Jp, eps):
        """Return the deformation gradient, its singular values and the plastic volume ratio once plasticity applies.

        `F_tmp` is the deformation gradient of the substep, `U`, `S` and `V` its singular value decomposition, `J` its
        determinant, `Jp` the plastic volume ratio of the particle and `eps` the tolerance of the norms.
        """
        raise NotImplementedError

    @qd.func
    def update_stress(self, U, S, V, F_tmp, F_new, J, Jp, actu, m_dir):
        """Return the Kirchhoff stress of a particle of deformation gradient `F_new` (see 'update_F_S_Jp').

        `actu` is the actuation of the particle and `m_dir` the direction of its muscle fibers.
        """
        stress = 2 * self.mu * (F_new - U @ V.transpose()) @ F_new.transpose() + qd.Matrix.identity(
            gs.qd_float, 3
        ) * self.lam * J * (J - 1)
        return stress

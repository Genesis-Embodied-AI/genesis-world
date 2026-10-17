from typing import TYPE_CHECKING, Annotated, Any, Literal

from pydantic import Field

from genesis.typing import NonNegativeFloat, PositiveFloat, ValidFloat

from .base import Base

if TYPE_CHECKING:
    from genesis.engine.entities.mochi_entity import MochiSoftEntity

ElasticModel = Literal["stable_neohookean", "stvk", "linear"]
SoftColliderType = Literal["auto", "sdf", "none"]


class Elastic(Base["MochiSoftEntity"]):
    """
    Deformable (tetrahedral finite element) material simulated by the MochiSolver.

    The body is discretized into linear tetrahedra whose vertex positions are unknowns of the same implicit Newton
    solve as the rigid bodies. Contact acts on quadrature samples of the boundary triangles against the rigid bodies'
    signed distance fields, with the same smooth penalty and friction model (see `Mochi.Base`).

    Parameters
    ----------
    E : float, optional
        Young's modulus in Pa. Default is 1e5.
    nu : float, optional
        Poisson's ratio. Default is 0.45.
    rho : float, optional
        Density in kg/m^3. Default is 1000.
    model : str, optional
        Constitutive model: "stable_neohookean" (Smith et al. 2018, robust to inversion), "stvk" (Saint
        Venant-Kirchhoff) or "linear" (small strain). Default is "stable_neohookean".
    mass_damping : float, optional
        Mass-proportional (Rayleigh) damping coefficient in 1/s. Default is 0.
    stiffness_damping : float, optional
        Stiffness-proportional (Kelvin-Voigt) damping coefficient in s, acting through the rest-state elastic tangent
        on the rate of the Green strain. Default is 0.
    friction, penalty_coefficient, penalty_smoothing_half_distance, penalty_threshold, friction_falloff_vel,
    viscous_friction, normal_viscous_damping, max_alignment_normals, has_gravity, contact_layer : optional
        Contact parameters, see `Mochi.Base`.
    collider_type : str, optional
        Whether other bodies' sample points collide against this body: "sdf" (and "auto") builds a signed distance
        field of the rest shape that is queried through the deformed tetrahedra (contact only registers for points
        inside the body, the activation threshold is zero), "none" makes the body collide only through its own samples.
        Default is "auto".
    """

    E: PositiveFloat = 1e5
    nu: Annotated[ValidFloat, Field(gt=-1.0, lt=0.5)] = 0.45
    rho: PositiveFloat = 1000.0
    model: ElasticModel = "stable_neohookean"
    mass_damping: NonNegativeFloat = 0.0
    stiffness_damping: NonNegativeFloat = 0.0
    collider_type: SoftColliderType = "auto"

    # Lame parameters, derived from E and nu.
    mu: float = Field(default=0.0, init=False, repr=False)
    lam: float = Field(default=0.0, init=False, repr=False)

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        self.mu = self.E / (2.0 * (1.0 + self.nu))
        self.lam = self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))

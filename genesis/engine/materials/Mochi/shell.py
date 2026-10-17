from typing import TYPE_CHECKING, Annotated, Any, Literal

from pydantic import Field, StrictBool

import genesis as gs
from genesis.typing import NonNegativeFloat, PositiveFloat, ValidFloat

from .base import Base

if TYPE_CHECKING:
    from genesis.engine.entities.mochi_entity import MochiSoftEntity

ShellColliderType = Literal["auto", "point_cloud", "none"]


class Shell(Base["MochiSoftEntity"]):
    """
    Thin shell (cloth-like) material simulated by the MochiSolver.

    The surface triangles of the morph are the finite elements: a Saint Venant-Kirchhoff membrane on the metric of the
    mid-surface and a Koiter-type bending term on its discrete curvature, both thickness-integrated. The stiffnesses
    derive from the 3D isotropic parameters and the thickness (plane stress): membrane shear `mu t`, membrane
    `lambda_ps t` with `lambda_ps = E nu / (1 - nu^2)`, bending `D nu` and `D (1 - nu)` with the plate rigidity
    `D = E t^3 / (12 (1 - nu^2))`, areal density `rho t`; each can be overridden.

    Parameters
    ----------
    E : float, optional
        Young's modulus in Pa. Default is 1e4.
    nu : float, optional
        Poisson's ratio. Default is 0.3.
    rho : float, optional
        Density in kg/m^3. Default is 200.
    thickness : float, optional
        Shell thickness in m. Default is 0.001.
    membrane_mu, membrane_lambda : float, optional
        Thickness-integrated membrane stiffnesses in N/m, overriding the values derived from E, nu and the thickness.
    bending_alpha, bending_beta : float, optional
        Bending stiffnesses in N m, overriding the values derived from E, nu and the thickness.
    mass_damping : float, optional
        Mass-proportional damping coefficient in 1/s. Default is 0.
    stiffness_damping : float, optional
        Stiffness-proportional (Rayleigh) damping coefficient in s. Default is 0.
    friction, penalty_coefficient, penalty_smoothing_half_distance, penalty_threshold, friction_falloff_vel,
    viscous_friction, normal_viscous_damping, max_alignment_normals, has_gravity, contact_layer : optional
        Contact parameters, see `Mochi.Base`.
    collider_type : str, optional
        Whether other bodies collide against this shell: "point_cloud" (and "auto") places a sphere of radius
        `collider_radius` at every vertex, "none" makes the shell collide only through its own samples. Default is
        "auto".
    collider_radius : float, optional
        Radius in m of the collider spheres at the vertices. Default is 0.01.
    self_contact : bool, optional
        Whether the samples of this shell collide against its own collider spheres (point-cloud collider only).
        Default is False.
    self_contact_exclusion_ratio : float, optional
        Samples whose rest position lies within `collider_radius * self_contact_exclusion_ratio` plus the penalty
        threshold of a vertex do not collide with the sphere of that vertex (neighbors along the shell); the rest
        distance to the second-nearest vertices must exceed this range plus the contact band. Must be larger than 1.
        Default is 1.5.
    """

    E: PositiveFloat = 1e4
    nu: Annotated[ValidFloat, Field(gt=-1.0, lt=0.5)] = 0.3
    rho: PositiveFloat = 200.0
    thickness: PositiveFloat = 1e-3
    membrane_mu: PositiveFloat | None = None
    membrane_lambda: ValidFloat | None = None
    bending_alpha: ValidFloat | None = None
    bending_beta: PositiveFloat | None = None
    mass_damping: NonNegativeFloat = 0.0
    stiffness_damping: NonNegativeFloat = 0.0
    collider_type: ShellColliderType = "auto"
    collider_radius: PositiveFloat = 1e-2
    self_contact: StrictBool = False
    self_contact_exclusion_ratio: Annotated[ValidFloat, Field(gt=1.0)] = 1.5

    # Areal density, derived from rho and the thickness.
    areal_density: float = Field(default=0.0, init=False, repr=False)

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        t = self.thickness
        mu = self.E / (2.0 * (1.0 + self.nu))
        lam_ps = self.E * self.nu / (1.0 - self.nu * self.nu)
        D = t**3 / 12.0 * self.E / (1.0 - self.nu * self.nu)
        if self.membrane_mu is None:
            self.membrane_mu = t * mu
        if self.membrane_lambda is None:
            self.membrane_lambda = t * lam_ps
        if self.bending_alpha is None:
            self.bending_alpha = D * self.nu
        if self.bending_beta is None:
            self.bending_beta = D * (1.0 - self.nu)
        if self.membrane_lambda <= -self.membrane_mu:
            gs.raise_exception("`membrane_lambda` must be larger than -`membrane_mu`.")
        if self.bending_alpha <= -0.5 * self.bending_beta:
            gs.raise_exception("`bending_alpha` must be larger than -`bending_beta` / 2.")
        self.areal_density = t * self.rho

from typing import TYPE_CHECKING, Annotated, Literal

import numpy as np
from pydantic import Field, StrictBool

from genesis.typing import NonNegativeFloat, PositiveFloat, ValidFloat

from .base import Base

if TYPE_CHECKING:
    from genesis.engine.entities.mochi_entity import MochiSoftEntity

RodColliderType = Literal["auto", "point_cloud", "none"]


class Rod(Base["MochiSoftEntity"]):
    """
    Rod (cable, rope, wire) material simulated by the MochiSolver as a discrete elastic rod: stretching of the
    segments, bending and twisting at the interior nodes.

    The stiffnesses derive from the 3D isotropic parameters and the radius of the circular cross-section: axial
    `E A`, flexural `E I` (both principal directions), torsional `G J` with `G = E / (2 (1 + nu))`, linear density
    `rho A` and linear rotational inertia `rho J`, with `A = pi r^2`, `I = pi r^4 / 4`, `J = pi r^4 / 2`; each can be
    overridden.

    Parameters
    ----------
    E : float, optional
        Young's modulus in Pa. Default is 1e7.
    nu : float, optional
        Poisson's ratio. Default is 0.3.
    rho : float, optional
        Density in kg/m^3. Default is 1000.
    axial_stiffness : float, optional
        Stretching stiffness `E A` in N, overriding the derived value.
    flexural_stiffness : float, optional
        Bending stiffness `E I` in N m^2 (both directions), overriding the derived value.
    torsional_stiffness : float, optional
        Twisting stiffness `G J` in N m^2, overriding the derived value.
    linear_density : float, optional
        Mass per unit length in kg/m, overriding the derived value.
    linear_rotational_inertia : float, optional
        Rotational inertia per unit length about the centerline in kg m, overriding the derived value.
    mass_damping : float, optional
        Mass-proportional damping coefficient in 1/s. Default is 0.
    stiffness_damping : float, optional
        Stiffness-proportional (Rayleigh) damping coefficient in s. Default is 0.
    friction, penalty_coefficient, penalty_smoothing_half_distance, penalty_threshold, friction_falloff_vel,
    viscous_friction, normal_viscous_damping, max_alignment_normals, has_gravity, contact_layer : optional
        Contact parameters, see `Mochi.Base`.
    collider_type : str, optional
        Whether other bodies collide against this rod: "point_cloud" (and "auto") places a sphere of the rod radius at
        every node, "none" makes the rod collide only through its own samples. Default is "auto".
    self_contact : bool, optional
        Whether the samples of this rod collide against its own collider spheres (point-cloud collider only).
        Default is False.
    self_contact_exclusion_ratio : float, optional
        Samples whose rest position lies within `collider_radius * self_contact_exclusion_ratio` plus the penalty
        threshold of a vertex do not collide with the sphere of that vertex (neighbors along the rod); the rest
        distance to the second-nearest vertices must exceed this range plus the contact band. Must be larger than 1.
        Default is 1.5.
    """

    E: PositiveFloat = 1e7
    nu: Annotated[ValidFloat, Field(gt=-1.0, lt=0.5)] = 0.3
    rho: PositiveFloat = 1000.0
    axial_stiffness: PositiveFloat | None = None
    flexural_stiffness: PositiveFloat | None = None
    torsional_stiffness: PositiveFloat | None = None
    linear_density: PositiveFloat | None = None
    linear_rotational_inertia: NonNegativeFloat | None = None
    mass_damping: NonNegativeFloat = 0.0
    stiffness_damping: NonNegativeFloat = 0.0
    collider_type: RodColliderType = "auto"
    self_contact: StrictBool = False
    self_contact_exclusion_ratio: Annotated[ValidFloat, Field(gt=1.0)] = 1.5

    def resolve(self, radius):
        """Stiffnesses and inertia per unit length of a circular cross-section of the given radius (the overrides win)."""
        area = np.pi * radius**2
        second_moment = np.pi * radius**4 / 4.0
        shear_modulus = self.E / (2.0 * (1.0 + self.nu))
        return {
            "axial_stiffness": self.axial_stiffness if self.axial_stiffness is not None else self.E * area,
            "flexural_stiffness": (
                self.flexural_stiffness if self.flexural_stiffness is not None else self.E * second_moment
            ),
            "torsional_stiffness": (
                self.torsional_stiffness
                if self.torsional_stiffness is not None
                else shear_modulus * 2.0 * second_moment
            ),
            "linear_density": self.linear_density if self.linear_density is not None else self.rho * area,
            "linear_rotational_inertia": (
                self.linear_rotational_inertia
                if self.linear_rotational_inertia is not None
                else self.rho * 2.0 * second_moment
            ),
        }

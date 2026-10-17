from typing import Annotated

from pydantic import Field, StrictBool

from genesis.typing import NonNegativeFloat, PositiveFloat, ValidFloat

from ..base import EntityT, Material


class Base(Material[EntityT]):
    """
    Base class of the materials simulated by the MochiSolver, holding the contact parameters they share.

    Contact between two bodies is a smooth penalty on the signed distance of sample points placed on one body's
    collision surface to the other body's signed distance field (SDF). Every pair parameter is combined from the two
    bodies' values by geometric mean, except the smoothing distance, the activation threshold and the alignment
    threshold, which are read from the collider body alone.

    Parameters
    ----------
    friction : float, optional
        Coulomb friction coefficient. Default is 0.5.
    penalty_coefficient : float, optional
        Contact stiffness in Pa/m: pressure per unit of penetration once the penalty ramp is fully active. Higher
        values reduce penetration at the cost of a stiffer, more ill-conditioned Newton system. Default is 1e9.
    penalty_smoothing_half_distance : float, optional
        Half-width in meters of the smooth ramp between zero contact pressure and the linear regime. The pressure
        ramps up over twice this distance below the activation threshold, so a wider ramp gives softer, better
        conditioned contact and a narrower one sharper contact onset. Default is 0.005.
    penalty_threshold : float, optional
        Signed distance in meters at which contact pressure starts to build up. Positive values start pushing bodies
        apart before they touch (a contact skin), negative values allow some interpenetration before responding.
        Default is 0.001.
    friction_falloff_vel : float, optional
        Sliding speed in m/s below which the Coulomb friction force is regularized towards zero. Larger values give
        smoother, more robust stick-slip transitions with more creep under static load; smaller values approach exact
        Coulomb friction with a stiffer system. Default is 0.01.
    viscous_friction : float, optional
        Tangential viscous friction coefficient in s/m, multiplying the normal force and the sliding velocity.
        Default is 0.
    normal_viscous_damping : float, optional
        Normal viscous damping coefficient in s/m, multiplying the normal force and the approach velocity. This is the
        mechanism controlling the coefficient of restitution: 0 gives fully elastic penalty contact, larger values
        dissipate more impact energy. Default is 0.
    max_alignment_normals : float, optional
        Cosine in [-1, 1] between the colliding surface normal and the collider gradient above which the contact is
        disabled, so that a body embedded past the far side of a thin collider can escape rather than being trapped.
        Default is 0.
    has_gravity : bool, optional
        Whether gravity acts on this body. Default is True.
    contact_layer : str, optional
        Name of the contact layer of this body. Contact between two layers can be disabled at the solver level.
        Default is "default".

    Note
    ----
    This class should *not* be instantiated directly.
    """

    friction: NonNegativeFloat = 0.5
    penalty_coefficient: PositiveFloat = 1e9
    penalty_smoothing_half_distance: NonNegativeFloat = 5e-3
    penalty_threshold: ValidFloat = 1e-3
    friction_falloff_vel: NonNegativeFloat = 1e-2
    viscous_friction: NonNegativeFloat = 0.0
    normal_viscous_damping: NonNegativeFloat = 0.0
    max_alignment_normals: Annotated[ValidFloat, Field(ge=-1.0, le=1.0)] = 0.0
    has_gravity: StrictBool = True
    contact_layer: str = "default"

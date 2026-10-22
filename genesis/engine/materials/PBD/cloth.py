from genesis.typing import NonNegativeFloat, PositiveFloat, ValidFloat

from .base import Base


class Cloth(Base):
    """
    The cloth material class for PBD.

    Parameters
    ----------
    rho : float, optional
        The density of the cloth. Default is 4.0.
        Note that this is kg/m², not kg/m³, as cloth is a 2D material, so the entity mass will be calculated as rho * surface_area.
    static_friction : float, optional
        Static friction coefficient. Default is 0.15.
    kinetic_friction : float, optional
        Kinetic friction coefficient. Default is 0.15.
    stretch_compliance : float, optional
        The stretch compliance (m/N). Default is 1e-7.
    bending_compliance : float, optional
        The bending compliance (rad/N). Default is 1e-5.
    stretch_relaxation : float, optional
        The stretch relaxation. Smaller value weakens the stretch constraint. Default is 0.3.
    bending_relaxation : float, optional
        The bending relaxation. Smaller value weakens the bending constraint. Default is 0.1.
    air_resistance : float, optional
        The air resistance. Damping force due to air drag. Default is 1e-3.
    """

    rho: PositiveFloat = 4.0
    static_friction: NonNegativeFloat = 0.15
    kinetic_friction: NonNegativeFloat = 0.15
    stretch_compliance: NonNegativeFloat = 1e-7
    bending_compliance: NonNegativeFloat = 1e-5
    stretch_relaxation: ValidFloat = 0.3
    bending_relaxation: ValidFloat = 0.1
    air_resistance: NonNegativeFloat = 1e-3

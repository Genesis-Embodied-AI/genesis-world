from genesis.typing import NonNegativeFloat, PositiveFloat, ValidFloat

from .base import Base


class Elastic(Base):
    """
    The 3D elastic volumetric material class for PBD.

    Parameters
    ----------
    rho : float, optional
        The density of the elastic material (kg/m³). Default is 1000.0.
    static_friction : float, optional
        Static friction coefficient. Default is 0.15.
    kinetic_friction : float, optional
        Kinetic friction coefficient. Default is 0.15.
    stretch_compliance : float, optional
        The stretch compliance (m/N). Default is 0.0.
    bending_compliance : float, optional
        The bending compliance (rad/N). Default is 0.0.
    volume_compliance : float, optional
        The volume compliance (m³/N). Default is 0.0.
    stretch_relaxation : float, optional
        The stretch relaxation. Default is 0.1.
    bending_relaxation : float, optional
        The bending relaxation. Default is 0.1.
    volume_relaxation : float, optional
        The volume relaxation. Default is 0.1.
    """

    rho: PositiveFloat = 1000.0
    static_friction: NonNegativeFloat = 0.15
    kinetic_friction: NonNegativeFloat = 0.15
    stretch_compliance: NonNegativeFloat = 0.0
    bending_compliance: NonNegativeFloat = 0.0
    volume_compliance: NonNegativeFloat = 0.0
    stretch_relaxation: ValidFloat = 0.1
    bending_relaxation: ValidFloat = 0.1
    volume_relaxation: ValidFloat = 0.1

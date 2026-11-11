"""Cloth material, a thin shell of the surface triangles of a mesh."""

from genesis.typing import PositiveFloat

from .base import Base


class Cloth(Base):
    """
    Cloth material for thin shell/membrane simulation. Only accepts surface mesh morphs.

    Parameters
    ----------
    E : float, optional
        Young's modulus (Pa), controlling stiffness. Default is 1e4 (10 kPa).
    rho : float, optional
        Material density (kg/m^3). Default is 200 (typical fabric).
    thickness : float, optional
        Shell thickness (m). Default is 0.001 (1mm).
    """

    E: PositiveFloat = 1e4
    rho: PositiveFloat = 200.0
    thickness: PositiveFloat = 0.001

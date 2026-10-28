from pydantic import StrictBool

from genesis.typing import NonNegativeFloat, PositiveInt

from .base import Material


class Tool(Material):
    """
    Material for tool entities.

    Parameters
    ----------
    friction : float, optional
        Friction coefficient. Default is 0.0.
    coup_softness : float, optional
        Softness of coupling interaction. Default is 0.01.
    collision : bool, optional
        Whether the tool participates in collision. Default is True.
    sdf_res : int, optional
        Resolution of the SDF grid. Default is 128.
    """

    friction: NonNegativeFloat = 0.0
    coup_softness: NonNegativeFloat = 0.01
    collision: StrictBool = True
    sdf_res: PositiveInt = 128

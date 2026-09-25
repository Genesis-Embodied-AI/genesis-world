from typing import TYPE_CHECKING, Callable

from pydantic import StrictBool

from genesis.typing import NonNegativeFloat, ValidFloat

from .base import MaterialOptions

if TYPE_CHECKING:
    from genesis.engine.entities.hybrid_entity import HybridEntity


class Hybrid(MaterialOptions["HybridEntity"]):
    """
    The class for hybrid body material (soft skin actuated by inner rigid skeleton).

    Parameters
    ----------
    material_rigid : MaterialOptions
        The material of the rigid body.
    material_soft : MaterialOptions
        The material of the soft body.
    use_default_coupling : bool, optional
        Whether to use default solver coupling. Default is False.
    damping : float, optional
        Damping coefficient between soft and rigid. Default is 0.0.
    thickness : float, optional
        The thickness to instantiate soft skin. Default is 0.05.
    soft_dv_coef : float, optional
        The coefficient to apply delta velocity from rigid to soft. Default is 0.01.
    func_instantiate_rigid_from_soft : callable, optional
        The function to instantiate rigid body from the geometry of soft body. Default is None.
    func_instantiate_soft_from_rigid : callable, optional
        The function to instantiate soft body from the geometry of rigid body. Default is None.
    func_instantiate_rigid_soft_association : callable, optional
        The function that determines the association of the rigid and the soft body. Default is None.
    """

    material_rigid: MaterialOptions = ...
    material_soft: MaterialOptions = ...
    use_default_coupling: StrictBool = False
    damping: NonNegativeFloat = 0.0
    thickness: ValidFloat = 0.05
    soft_dv_coef: ValidFloat = 0.01
    func_instantiate_rigid_from_soft: Callable | None = None
    func_instantiate_soft_from_rigid: Callable | None = None
    func_instantiate_rigid_soft_association: Callable | None = None

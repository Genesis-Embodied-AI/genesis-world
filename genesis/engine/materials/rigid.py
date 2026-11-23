from typing import TYPE_CHECKING, Annotated, Any

from pydantic import Field, StrictBool

import genesis as gs
from genesis.typing import NonNegativeFloat, PositiveFloat, StrictInt, ValidFloat

from .kinematic import Kinematic

if TYPE_CHECKING:
    from genesis.engine.entities.rigid_entity import RigidEntity


class Rigid(Kinematic["RigidEntity"]):
    """
    The Rigid class represents a material used in rigid body simulation.

    Note
    ----
    This class is intended for use with the rigid solver and provides parameters
    relevant to physical interactions such as friction, density, and signed distance fields (SDFs).

    Parameters
    ----------
    use_visual_raycasting : bool, optional
        See Kinematic. Default is False.
    rho : float or None, optional
        The density of the material used to estimate mass if necessary. When None, the default depends on context:
        1000 kg/m^3 if MuJoCo compatibility is enabled (``RigidOptions.enable_mujoco_compatibility``),
        otherwise 600 kg/m^3 for basic rigid objects (mug, table...) vs 1500 kg/m^3 for poly-articulated
        robots. Default is None.
    friction : float, optional
        Friction coefficient within the rigid solver. If None, a default of 1.0 may be used or parsed from file.
    friction_torsional : float, optional
        Torsional friction coefficient, resisting relative spin about the contact normal. Expressed in meters, as it
        stands for the effective contact patch radius over which sliding friction acts. Only effective when torsional
        friction is enabled at the scene level (see 'RigidOptions.enable_torsional_friction'). If None, parsed from
        file when available (MJCF), otherwise 0.005. Default is None.
    friction_rolling : float, optional
        Rolling friction coefficient, resisting rolling about the two contact tangent axes. Expressed in meters, like
        the torsional coefficient. Only effective when rolling friction is enabled at the scene level (see
        'RigidOptions.enable_rolling_friction'). If None, parsed from file when available (MJCF), otherwise 0.0001.
        Default is None.
    needs_coup : bool, optional
        Whether the material participates in coupling with other solvers. Default is True.
    coup_friction : float, optional
        Friction used during coupling. Must be non-negative. Default is 0.1.
    coup_softness : float, optional
        Softness of coupling interaction. Must be non-negative. Default is 0.002.
    coup_restitution : float, optional
        Restitution coefficient in collision coupling. Should be between 0 and 1. Default is 0.0. The coupling with SPH
        liquids depends on none of these three parameters.
    sdf_cell_size : float, optional
        Cell size in SDF grid in meters. Defines grid resolution. Default is 0.005.
    sdf_min_res : int, optional
        Minimum resolution of the SDF grid. Must be at least 16. Default is 32.
    sdf_max_res : int, optional
        Maximum resolution of the SDF grid. Must be >= sdf_min_res. Default is 128.
    gravity_compensation : float, optional
        Compensation factor for gravity. 1.0 cancels gravity. Default is 0.
    """

    use_visual_raycasting: StrictBool = False

    rho: ValidFloat | None = None
    friction: Annotated[ValidFloat, Field(ge=0.01, le=5.0)] | None = None
    friction_torsional: Annotated[ValidFloat, Field(ge=0.0)] | None = None
    friction_rolling: Annotated[ValidFloat, Field(ge=0.0)] | None = None
    needs_coup: StrictBool = True
    coup_friction: NonNegativeFloat = 0.1
    coup_softness: NonNegativeFloat = 0.002
    coup_restitution: Annotated[ValidFloat, Field(ge=0.0, le=1.0)] = 0.0
    sdf_cell_size: PositiveFloat = 0.005
    sdf_min_res: Annotated[StrictInt, Field(ge=16)] = 32
    sdf_max_res: Annotated[StrictInt, Field(ge=16)] = 128
    gravity_compensation: ValidFloat = 0.0

    def model_post_init(self, context: Any) -> None:
        if self.sdf_min_res > self.sdf_max_res:
            gs.raise_exception("`sdf_min_res` must be smaller than or equal to `sdf_max_res`.")

        if self.coup_restitution != 0:
            gs.logger.warning("Non-zero `coup_restitution` could lead to instability. Use with caution.")

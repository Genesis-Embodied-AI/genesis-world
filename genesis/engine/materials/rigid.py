from typing import TYPE_CHECKING, Annotated, Any, Literal

import numpy as np

from pydantic import Field, StrictBool, model_validator

import genesis as gs
from genesis.typing import NonNegativeFloat, PositiveFloat, StrArrayType, StrictInt, ValidFloat
from genesis.utils import geom as gu

from .base import Material
from .kinematic import Kinematic

if TYPE_CHECKING:
    from genesis.engine.entities.rigid_entity import RigidEntity

CoupType = Literal["two_way_soft_constraint", "external_articulation", "ipc_only"]


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
        Sliding friction coefficient. Exact pairs declared through 'Scene.set_friction_pair' take precedence,
        followed by the higher-priority material, then 'RigidOptions.friction_combine' for equal priorities.
        If None, parsed from the asset when available, otherwise 1.0. Defaults to None.
    friction_torsional : float, optional
        Torsional friction coefficient, resisting relative spin about the contact normal. Expressed in meters, as it
        stands for the effective contact patch radius over which sliding friction acts. Resolved for a contact as the
        sliding coefficient is. Only effective when torsional friction is enabled at the scene level (see
        'RigidOptions.enable_torsional_friction'). If None, parsed from file when available (MJCF), otherwise 0.005.
        Default is None.
    friction_rolling : float, optional
        Rolling friction coefficient, resisting rolling about the two contact tangent axes. Expressed in meters, like
        the torsional coefficient, and resolved for a contact the same way. Only effective when rolling friction is
        enabled at the scene level (see 'RigidOptions.enable_rolling_friction'). If None, parsed from file when
        available (MJCF), otherwise 0.0001. Default is None.
    needs_coup : bool, optional
        Whether the material participates in coupling with other solvers. Default is True.
    contact_priority : int, optional
        A higher priority imposes this material's complete friction row on a contact. Equal priorities use the
        scene's combine rule. Exact material pairs take precedence. Defaults to 0.
    coup_friction : float, optional
        Friction used during coupling. Must be non-negative. Default is 0.1.
    coup_softness : float, optional
        Softness of coupling interaction. Must be non-negative. Default is 0.002.
    coup_restitution : float, optional
        Restitution coefficient in collision coupling. Should be between 0 and 1. Default is 0.0.
    sdf_cell_size : float, optional
        Cell size in SDF grid in meters. Defines grid resolution. Default is 0.005.
    sdf_min_res : int, optional
        Minimum resolution of the SDF grid. Must be at least 16. Default is 32.
    sdf_max_res : int, optional
        Maximum resolution of the SDF grid. Must be >= sdf_min_res. Default is 128.
    gravity_compensation : float, optional
        Compensation factor for gravity. 1.0 cancels gravity. Default is 0.
    coup_type : str or None, optional
        Coupling mode for this entity. Only used by the IPC coupler. Requires ``needs_coup=True``.
        If None, auto-selected based on entity type: ``'external_articulation'`` for fixed-base
        articulated robots, ``'two_way_soft_constraint'`` for floating-base robots, and
        ``'ipc_only'`` for non-articulated objects. Valid values:
          - 'two_way_soft_constraint': Two-way soft coupling.
          - 'external_articulation': Joint-level coupling for articulated bodies. Joint positions will be coupled at
            the DOF level.
          - 'ipc_only': IPC controls entity, transforms copied to Genesis (one-way). Only supported by rigid
            non-articulated objects.
        Default is None.
    coup_links : tuple of str or None, optional
        Tuple of link names to include in coupling. When set, only the named links participate
        in coupling; other links are excluded. Only supported with needs_coup=True and
        ``two_way_soft_constraint`` type in IPC. Default is None.
    enable_coup_collision : bool, optional
        Whether coupler collision is enabled for this entity's links. Only used by the IPC coupler.
        Unlike ``needs_coup=False`` (which removes the entity from the coupler entirely), setting this to
        False keeps the entity in the coupler for coupling forces but disables contact response. Default is True.
    coup_collision_links : tuple of str or None, optional
        Tuple of link names whose geoms participate in coupler collision. Only used by the IPC coupler.
        Only effective when ``enable_coup_collision=True``. If None, all coupled links have collision.
        When set, only the named links get coupler collision; other links are marked no-collision.
        Default is None.
    contact_resistance : float or None, optional
        IPC coupling contact resistance/stiffness override for this entity. ``None`` means use
        ``IPCCouplerOptions.contact_resistance``. Default is None.
    """

    use_visual_raycasting: StrictBool = False

    rho: ValidFloat | None = None
    friction: Annotated[ValidFloat, Field(ge=0.0)] | None = None
    friction_torsional: Annotated[ValidFloat, Field(ge=0.0)] | None = None
    friction_rolling: Annotated[ValidFloat, Field(ge=0.0)] | None = None
    contact_priority: StrictInt = 0
    needs_coup: StrictBool = True
    coup_friction: NonNegativeFloat = 0.1
    coup_softness: NonNegativeFloat = 0.002
    coup_restitution: Annotated[ValidFloat, Field(ge=0.0, le=1.0)] = 0.0
    sdf_cell_size: PositiveFloat = 0.005
    sdf_min_res: Annotated[StrictInt, Field(ge=16)] = 32
    sdf_max_res: Annotated[StrictInt, Field(ge=16)] = 128
    gravity_compensation: ValidFloat = 0.0
    coup_type: CoupType | None = None
    coup_links: StrArrayType | None = None
    enable_coup_collision: StrictBool = True
    coup_collision_links: StrArrayType | None = None
    contact_resistance: PositiveFloat | None = None

    @model_validator(mode="before")
    @classmethod
    def _resolve_defaults(cls, data: dict) -> dict:
        # ipc_only entities have their dynamics fully controlled by IPC (gravity + collision).
        # Genesis gravity must be disabled to avoid double-counting.
        if data.get("coup_type") == "ipc_only":
            grav_comp = data.get("gravity_compensation")
            if grav_comp is not None and grav_comp != 0.0:
                gs.raise_exception(
                    "`gravity_compensation` must be 0 with coup_type='ipc_only'. "
                    "IPC controls gravity for ipc_only entities."
                )
            data["gravity_compensation"] = 0.0
        return data

    def model_post_init(self, context: Any) -> None:
        if self.coup_type is not None and not self.needs_coup:
            gs.raise_exception(
                "`coup_type` is only supported with needs_coup=True. "
                f"Got needs_coup={self.needs_coup}, coup_type={self.coup_type!r}."
            )

        if self.coup_links is not None and (
            not self.needs_coup or self.coup_type not in (None, "two_way_soft_constraint")
        ):
            gs.raise_exception(
                "`coup_links` is only supported with needs_coup=True and "
                f"'two_way_soft_constraint' type in IPC. "
                f"Got needs_coup={self.needs_coup}, coup_type={self.coup_type!r}."
            )

        if self.coup_collision_links is not None and not self.enable_coup_collision:
            gs.raise_exception(
                "`coup_collision_links` is only effective when `enable_coup_collision=True`. "
                "Set `enable_coup_collision=False` to disable collision for all links."
            )

        if self.sdf_min_res > self.sdf_max_res:
            gs.raise_exception("`sdf_min_res` must be smaller than or equal to `sdf_max_res`.")

        if self.coup_restitution != 0:
            gs.logger.warning("Non-zero `coup_restitution` could lead to instability. Use with caution.")


class RigidMaterial(Material[Rigid]):
    """
    A rigid material registered on a scene, as returned by 'Scene.add_material' and held by 'entity.material'.

    Every entity built from the same handle shares one material identity, which is the granularity contact parameters
    resolved between two materials are keyed on.

    'friction' gives this material's resolved sliding, torsional and rolling coefficients. A contact resolves those
    against the other material and any exact pair, then applies both geoms' per-environment friction factors.
    """

    @property
    def friction(self) -> tuple[float, float, float]:
        """Get the resolved sliding, torsional and rolling coefficients of this material."""
        return (
            gu.default_friction() if self.options.friction is None else self.options.friction,
            gu.default_friction_torsional()
            if self.options.friction_torsional is None
            else self.options.friction_torsional,
            gu.default_friction_rolling() if self.options.friction_rolling is None else self.options.friction_rolling,
        )

    @property
    def contact_priority(self) -> int:
        """Get the priority used to select this material's contact parameters."""
        return self.options.contact_priority

    def set_friction(self, friction=None, *, torsional=None, rolling=None):
        """Set this material's friction and update its contacts in every environment.

        Coefficients left None retain their current values. Exact material pairs retain their declared coefficients.
        """
        values = (friction, torsional, rolling)
        if any(value is not None and (not np.isfinite(value) or value < 0.0) for value in values):
            gs.raise_exception("Friction coefficients must be finite and non-negative.")
        for name, value in zip(("friction", "friction_torsional", "friction_rolling"), values):
            if value is not None:
                setattr(self.options, name, value)
        if self.scene.is_built:
            self.scene.sim.rigid_solver.collider.update_friction_pairs()

    # ------------------------------------------------------------------------------------
    # ----------------------------------- properties -------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def use_visual_raycasting(self) -> bool:
        return self._options.use_visual_raycasting

    @property
    def rho(self):
        return self._options.rho

    @property
    def needs_coup(self) -> bool:
        return self._options.needs_coup

    @property
    def coup_friction(self) -> float:
        return self._options.coup_friction

    @property
    def coup_softness(self) -> float:
        return self._options.coup_softness

    @property
    def coup_restitution(self) -> float:
        return self._options.coup_restitution

    @property
    def sdf_cell_size(self) -> float:
        return self._options.sdf_cell_size

    @property
    def sdf_min_res(self) -> int:
        return self._options.sdf_min_res

    @property
    def sdf_max_res(self) -> int:
        return self._options.sdf_max_res

    @property
    def gravity_compensation(self) -> float:
        return self._options.gravity_compensation

    @property
    def coup_type(self):
        return self._options.coup_type

    @property
    def coup_links(self):
        return self._options.coup_links

    @property
    def enable_coup_collision(self) -> bool:
        return self._options.enable_coup_collision

    @property
    def coup_collision_links(self):
        return self._options.coup_collision_links

    @property
    def contact_resistance(self):
        return self._options.contact_resistance

"""Every option a scene is created with, gathered as one object."""

from pydantic import Field, model_validator

import genesis as gs

from .options import Options
from .profiling import ProfilingOptions
from .renderers import Rasterizer, RendererOptions
from .solvers import (
    BaseCouplerOptions,
    FEMOptions,
    KinematicOptions,
    LegacyCouplerOptions,
    MPMOptions,
    PBDOptions,
    RigidOptions,
    SFOptions,
    SimOptions,
    SPHOptions,
    ToolOptions,
)
from .vis import ViewerOptions, VisOptions


class SceneOptions(Options):
    """Every option a scene is created with, as one object.

    Each field bears the name of the 'Scene' argument that carries it, without the '_options' suffix. Each states its
    own default and falls back to it when given None, so a scene names an option only to override it, and every field
    holds one once the options exist.

    Every option inherits from the simulation options the quantities it declares and leaves unset, which in practice
    is the solvers, since no other option shares a field with them. That resolution happens here, so every consumer
    reads the resolved value.
    """

    sim: SimOptions = Field(default_factory=SimOptions)
    tool: ToolOptions = Field(default_factory=ToolOptions)
    rigid: RigidOptions = Field(default_factory=RigidOptions)
    kinematic: KinematicOptions = Field(default_factory=KinematicOptions)
    mpm: MPMOptions = Field(default_factory=MPMOptions)
    sph: SPHOptions = Field(default_factory=SPHOptions)
    fem: FEMOptions = Field(default_factory=FEMOptions)
    sf: SFOptions = Field(default_factory=SFOptions)
    pbd: PBDOptions = Field(default_factory=PBDOptions)
    coupler: BaseCouplerOptions = Field(default_factory=LegacyCouplerOptions)
    vis: VisOptions = Field(default_factory=VisOptions)
    viewer: ViewerOptions = Field(default_factory=ViewerOptions)
    profiling: ProfilingOptions = Field(default_factory=ProfilingOptions)
    renderer: RendererOptions = Field(default_factory=Rasterizer)

    @model_validator(mode="before")
    @classmethod
    def _given_or_declared(cls, values):
        """Give a field passed as None its declared default, exactly like a field the caller omits.

        'Scene.__init__' passes every field, as None where the caller gave nothing, so the declared default is
        reached here rather than by omission.
        """
        if isinstance(values, dict):
            return {name: value for name, value in values.items() if value is not None}
        return values

    def model_post_init(self, context) -> None:
        # Validate rigid_options against sim_options
        if self.rigid.box_box_detection is None:
            self.rigid.box_box_detection = not self.sim.requires_grad
        elif self.rigid.box_box_detection and self.sim.requires_grad:
            gs.raise_exception(
                "`rigid_options.box_box_detection` cannot be True when `sim_options.requires_grad` is True."
            )
        if self.rigid.use_gjk_collision is None:
            self.rigid.use_gjk_collision = self.sim.requires_grad
        elif not self.rigid.use_gjk_collision and self.sim.requires_grad:
            gs.raise_exception(
                "`rigid_options.use_gjk_collision` cannot be False when `sim_options.requires_grad` is True."
            )
        if self.rigid.enable_mujoco_compatibility and self.sim.requires_grad:
            gs.raise_exception(
                "`rigid_options.enable_mujoco_compatibility` cannot be True when `sim_options.requires_grad` is True."
            )

        # Each option inherits from the simulation options the values it leaves unset, on fields it declares.
        # 'model_copy_from' fills only fields both sides declare, so one loop covers every option by name.
        for name, option in dict(self).items():
            if name != "sim":
                self.__dict__[name] = option.model_copy_from(self.sim)

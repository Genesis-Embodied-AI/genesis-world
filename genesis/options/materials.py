"""Materials, each stating the parameters of one physical law, whichever solver simulates the entity."""

from typing import Annotated, Literal

from pydantic import Field, model_validator

import genesis as gs
from genesis.typing import NonNegativeFloat, PositiveFloat, ValidFloat

from .options import Options

PoissonRatio = Annotated[ValidFloat, Field(gt=-1.0, lt=0.5)]


class BaseMaterial(Options):
    """The physical law of an entity and its parameters, whichever solver simulates it.

    A concrete subclass names one law, such as ``gs.materials.Elastic``. The entity options decide the solver, and each
    solver reads the parameters it supports. A parameter left to None takes the default of the solver simulating the
    entity, and the entity options raise an error on a parameter given explicitly that their solver cannot honour.

    Note
    ----
    This class should *not* be instantiated directly.

    Parameters
    ----------
    density : float | None, optional
        The mass density, in kg/m^3. If None, the solver decides.
    """

    density: PositiveFloat | None = None


class Material(BaseMaterial):
    """The material of a rigid entity, whose geometry never deforms.

    Parameters
    ----------
    density : float | None, optional
        The density used to estimate the mass of a link that states none, in kg/m^3. An explicit value replaces the
        densities an asset authors per geom as well. If None, the asset decides where it states a density, and
        otherwise 1000 kg/m^3 under MuJoCo compatibility (see ``RigidOptions.enable_mujoco_compatibility``), 1500
        kg/m^3 for an articulated robot and 600 kg/m^3 for a basic object such as a mug or a table. Default is None.
    friction : float | None, optional
        The sliding friction coefficient, which a rigid entity bounds (see ``RigidEntityOptions``). If None, the asset
        decides where it states one, and otherwise 1.0 for a rigid entity. Default is None.
    friction_torsional : float | None, optional
        The torsional friction coefficient, resisting the relative spin about the contact normal. Expressed in meters,
        as it stands for the radius of the contact patch over which sliding friction acts. Only effective when
        torsional friction is enabled (see ``RigidOptions.enable_torsional_friction``). If None, the asset decides
        where it states one (MJCF), and otherwise 0.005. Default is None.
    friction_rolling : float | None, optional
        The rolling friction coefficient, resisting the rolling about both contact tangent axes. Expressed in meters,
        like the torsional coefficient. Only effective when rolling friction is enabled (see
        ``RigidOptions.enable_rolling_friction``). If None, the asset decides where it states one (MJCF), and
        otherwise 0.0001. Default is None.
    """

    friction: NonNegativeFloat | None = None
    friction_torsional: Annotated[ValidFloat, Field(ge=0.0)] | None = None
    friction_rolling: Annotated[ValidFloat, Field(ge=0.0)] | None = None


class Elastic(BaseMaterial):
    """A solid that recovers its rest shape once the load is removed, such as rubber or soft tissue.

    The finite element method (FEM), the material point method (MPM) and position-based dynamics (PBD) simulate it.
    PBD expresses stiffness through its own solver options, so it rejects an explicit Young's modulus or Poisson ratio.

    Parameters
    ----------
    density : float, optional
        The mass density, in kg/m^3. Default is 1000.
    friction : float | None, optional
        The contact friction coefficient. If None, 0.1 under FEM and 0.15 under PBD. MPM has no contact friction of its
        own and rejects an explicit value. Default is None.
    youngs_modulus : float, optional
        The Young's modulus, in Pa. A stiffer solid deforms less but needs a finer time step to stay stable. Default is
        1e6.
    poisson_ratio : float, optional
        The Poisson ratio, between -1 and 0.5. Close to 0.5, the solid keeps its volume, at the cost of a stiffer
        response that needs a finer time step. Default is 0.2.
    """

    density: PositiveFloat = 1000.0
    friction: NonNegativeFloat | None = None
    youngs_modulus: PositiveFloat = 1e6
    poisson_ratio: PoissonRatio = 0.2


class Cloth(BaseMaterial):
    """A thin sheet that stretches and bends, such as fabric or paper.

    The finite element method (FEM) simulates it as a shell under the IPC coupler, and position-based dynamics (PBD)
    as a mesh of particles. PBD expresses stretching and bending through its own solver options, so it rejects an
    explicit Young's modulus, Poisson ratio or bending stiffness.

    Parameters
    ----------
    density : float | None, optional
        The mass density of the sheet, in kg/m^3, so that the mass per unit area is ``density * thickness``. If None,
        200 kg/m^3 under FEM, and 4.0 kg/m^2 of mass per unit area under PBD. Default is None.
    friction : float | None, optional
        The contact friction coefficient. If None, 0.1 under FEM and 0.15 under PBD. Default is None.
    youngs_modulus : float, optional
        The Young's modulus of the sheet, in Pa. A stiffer sheet stretches less. Default is 1e4.
    poisson_ratio : float, optional
        The Poisson ratio of the sheet, between -1 and 0.5. Default is 0.49.
    thickness : float, optional
        The thickness of the sheet, in m. It scales the mass per unit area given a density, and under FEM the
        resistance to bending as well, so PBD rejects an explicit value without a density. Default is 0.001.
    bending_stiffness : float | None, optional
        The resistance of the sheet to bending. If None, the sheet folds freely, like a thin fabric. Default is None.
    """

    friction: NonNegativeFloat | None = None
    youngs_modulus: PositiveFloat = 1e4
    poisson_ratio: PoissonRatio = 0.49
    thickness: PositiveFloat = 0.001
    bending_stiffness: NonNegativeFloat | None = None


class ElastoPlastic(BaseMaterial):
    """A solid that deforms elastically up to a yield limit and permanently beyond it, such as dough or clay.

    The material point method (MPM) simulates it.

    Parameters
    ----------
    density : float, optional
        The mass density, in kg/m^3. Default is 1000.
    youngs_modulus : float, optional
        The Young's modulus, in Pa. Default is 1e6.
    poisson_ratio : float, optional
        The Poisson ratio, between -1 and 0.5. Default is 0.2.
    yield_criterion : str, optional
        The law deciding when the solid yields. 'von_mises' yields once the shear stress exceeds 'yield_stress',
        whatever the pressure, as metals and dough do. 'stretch_clamp' yields once a principal stretch leaves the
        bounds 'critical_compression' and 'critical_stretch', as a crumbly material does. Each criterion rejects an
        explicit value of the parameters of the other. Default is 'von_mises'.
    yield_stress : float, optional
        The shear stress at which the solid yields under 'von_mises', in Pa. Default is 1e4.
    critical_compression : float, optional
        The relative compression at which the solid yields under 'stretch_clamp'. Default is 0.025.
    critical_stretch : float, optional
        The relative stretch at which the solid yields under 'stretch_clamp'. Default is 0.0045.
    """

    density: PositiveFloat = 1000.0
    youngs_modulus: PositiveFloat = 1e6
    poisson_ratio: PoissonRatio = 0.2
    yield_criterion: Literal["von_mises", "stretch_clamp"] = "von_mises"
    yield_stress: PositiveFloat = 1e4
    critical_compression: NonNegativeFloat = 2.5e-2
    critical_stretch: NonNegativeFloat = 4.5e-3

    @model_validator(mode="after")
    def _check_yield_parameters(self) -> "ElastoPlastic":
        if self.yield_criterion == "von_mises":
            names = sorted({"critical_compression", "critical_stretch"} & self.model_fields_set)
        else:
            names = sorted({"yield_stress"} & self.model_fields_set)
        if names:
            gs.raise_exception(f"Yield criterion '{self.yield_criterion}' ignores {', '.join(names)}.")
        return self


class Liquid(BaseMaterial):
    """A fluid that flows under any shear load, such as water or honey.

    The material point method (MPM), smoothed particle hydrodynamics (SPH) and position-based dynamics (PBD) simulate
    it. PBD keeps its liquid incompressible and expresses viscosity through its own solver options, so it rejects an
    explicit bulk modulus, viscosity or surface tension.

    Parameters
    ----------
    density : float, optional
        The rest density, in kg/m^3. Default is 1000.
    bulk_modulus : float | None, optional
        The resistance to compression, in Pa. A larger value keeps the volume closer to constant but needs a finer
        time step to stay stable. If None, about 2.8e5 Pa under MPM and 3.5e5 Pa under SPH. Default is None.
    viscosity : float | None, optional
        The dynamic viscosity, in Pa*s. A larger value damps the flow, as honey does. MPM applies it as a shear modulus
        of viscosity / substep_dt over each substep, which also stiffens the liquid against compression, the more so
        at a finer substep. If None, 0 under MPM, which simulates an inviscid liquid, and 0.005 under SPH. Default is
        None.
    surface_tension : float | None, optional
        The surface tension coefficient, which pulls the free surface together. SPH alone simulates it. If None, 0.01
        under SPH. Default is None.
    """

    density: PositiveFloat = 1000.0
    bulk_modulus: PositiveFloat | None = None
    viscosity: NonNegativeFloat | None = None
    surface_tension: NonNegativeFloat | None = None


class Sand(BaseMaterial):
    """A granular material whose grains slide over each other once the shear load overcomes their internal friction.

    The material point method (MPM) simulates it.

    Parameters
    ----------
    density : float, optional
        The mass density, in kg/m^3. Default is 1000.
    youngs_modulus : float, optional
        The Young's modulus, in Pa. Default is 1e6.
    poisson_ratio : float, optional
        The Poisson ratio, between -1 and 0.5. Default is 0.2.
    friction_angle : float, optional
        The internal friction angle, in degrees. A larger angle holds steeper piles, as wet sand does. Default is 45.
    """

    density: PositiveFloat = 1000.0
    youngs_modulus: PositiveFloat = 1e6
    poisson_ratio: PoissonRatio = 0.2
    friction_angle: PositiveFloat = 45.0


class Snow(BaseMaterial):
    """A compressible solid that yields under load and hardens as it compacts.

    The material point method (MPM) simulates it.

    Parameters
    ----------
    density : float, optional
        The mass density, in kg/m^3. Default is 1000.
    youngs_modulus : float, optional
        The Young's modulus, in Pa. Default is 1e6.
    poisson_ratio : float, optional
        The Poisson ratio, between -1 and 0.5. Default is 0.2.
    critical_compression : float, optional
        The relative compression at which the snow yields. Default is 0.025.
    critical_stretch : float, optional
        The relative stretch at which the snow yields. Default is 0.0045.
    """

    density: PositiveFloat = 1000.0
    youngs_modulus: PositiveFloat = 1e6
    poisson_ratio: PoissonRatio = 0.2
    critical_compression: NonNegativeFloat = 2.5e-2
    critical_stretch: NonNegativeFloat = 4.5e-3


class Muscle(BaseMaterial):
    """An elastic solid that also contracts along a fiber direction when actuated.

    The finite element method (FEM) and the material point method (MPM) simulate it. The entity groups its elements
    or particles into muscle groups (see the 'n_groups' solver option of its entity options), each actuated on its own.

    Parameters
    ----------
    density : float, optional
        The mass density, in kg/m^3. Default is 1000.
    friction : float | None, optional
        The contact friction coefficient. If None, 0.1 under FEM. MPM has no contact friction of its own and rejects an
        explicit value. Default is None.
    youngs_modulus : float, optional
        The Young's modulus, in Pa. It scales the active contraction as well. Default is 1e6.
    poisson_ratio : float, optional
        The Poisson ratio, between -1 and 0.5. Default is 0.2.
    """

    density: PositiveFloat = 1000.0
    friction: NonNegativeFloat | None = None
    youngs_modulus: PositiveFloat = 1e6
    poisson_ratio: PoissonRatio = 0.2

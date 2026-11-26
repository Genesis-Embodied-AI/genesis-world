from collections.abc import Sequence
from typing import TYPE_CHECKING, Annotated, Any, NamedTuple

import numpy as np

from pydantic import BeforeValidator, Field, StrictBool, ValidatorFunctionWrapHandler, field_validator, model_validator

import genesis as gs
from genesis.typing import (
    FArrayType,
    Grid3DFloatType,
    IArrayType,
    LaxVec3FType,
    NonNegativeFloat,
    NonNegativeInt,
    OptionalIArrayType,
    PositiveFArrayType,
    PositiveFGridType,
    PositiveFloat,
    PositiveVec3IType,
    RotationMatrixType,
    UnitInterval,
    UnitIntervalVec3Type,
    UnitIntervalVec4Type,
    UnitVec3FArrayType,
    UnitVec3FGridType,
    UnitVec3FType,
    Vec2FType,
    Vec3FArrayType,
    Vec3FGridType,
    Vec3FType,
    Vec4FType,
    is_sequence,
    objects_idx_type,
)

from ..options import Options
from .raycaster import DepthCameraPattern, RaycastPattern

if TYPE_CHECKING:
    from genesis.engine.entities import Entity
    from genesis.engine.entities.rigid_entity import RigidLink
    from genesis.engine.simulator import Simulator
    from genesis.engine.sensors.base_sensor import Sensor
    from genesis.engine.sensors.contact_force import ContactForceSensor, ContactSensor
    from genesis.engine.sensors.imu import IMUSensor
    from genesis.engine.sensors.raycaster import RaycasterSensor
    from genesis.engine.sensors.surface_distance_probe import SurfaceDistanceProbeSensor
    from genesis.engine.sensors.temperature import TemperatureGridSensor

    NonNegativeUnboundedFloat = float
    LaxNonNegativeUnboundedVec3FType = Vec3FType | float
    EntityIdx = Entity | int
    OptionalRigidLinksIdx = Sequence[RigidLink | int]
else:
    NonNegativeUnboundedFloat = Annotated[float, Field(ge=0, strict=False)]
    LaxNonNegativeUnboundedVec3FType = Annotated[
        tuple[NonNegativeUnboundedFloat, NonNegativeUnboundedFloat, NonNegativeUnboundedFloat],
        BeforeValidator(lambda v: v if is_sequence(v) else (v,) * 3),
        Field(strict=False),
    ]
    # The classes are looked up lazily, since the engine imports these options before defining them
    EntityIdx = objects_idx_type(lambda: gs.engine.entities.Entity, many=False, allow_int=True)
    OptionalRigidLinksIdx = objects_idx_type(
        lambda: gs.engine.entities.rigid_entity.RigidLink, many=True, allow_empty=True, allow_int=True
    )
CrossCouplingAxisType = RotationMatrixType | UnitIntervalVec3Type | float


def _check_len_match(value, expected_len: int, name: str, ref_name: str):
    if isinstance(value, Sequence) and len(value) != expected_len:
        gs.raise_exception(
            f"{name} must have the same length as {ref_name} when {name} is array-like. "
            f"Got {len(value)} {name} and {expected_len} {ref_name}."
        )


def _resolve_tracked_links(cls_name: str, data: Any) -> Any:
    # 'tracked_links' is resolved into 'track_link_idx', so that the options hold indices alone
    if not isinstance(data, dict) or "tracked_links" not in data:
        return data
    if "track_link_idx" in data:
        gs.logger.warning(f"{cls_name}: Both tracked_links and track_link_idx were provided, using tracked_links.")
    data = dict(data)
    links = data.pop("tracked_links")
    if not is_sequence(links):
        gs.raise_exception(f"{cls_name}: tracked_links must be a sequence of links, got {type(links).__name__}.")
    for link in links:
        if not isinstance(link, gs.engine.entities.rigid_entity.RigidLink):
            gs.raise_exception(f"{cls_name}: tracked_links holds a {type(link).__name__}, which is no rigid link.")
    data["track_link_idx"] = tuple(link.idx for link in links)
    return data


class SensorOptions[SensorT: "Sensor"](Options):
    """
    Base class for all sensor options.

    Each sensor should have their own options class that inherits from this class.
    The associated sensor class registers itself via ``Sensor.__init_subclass__`` when parameterized
    with this options class, e.g. ``class MySensor(Sensor[MyOptions, MyMetadata, MyData]): ...``

    Parameters
    ----------
    history_length : NonNegativeInt
        The length of the history to store. Defaults to 0 (no history).
    delay : float, optional
        The read delay time in seconds. Data read will be outdated by this amount. Defaults to 0.0 (no delay).
    jitter : float, optional
        The jitter in seconds modeled as a random additive delay sampled uniformly in ``[0, jitter)`` each step.
        Jitter cannot be greater than delay.
    draw_debug : bool
        Whether the readings of the sensor are drawn as markers, which the interactive viewer and the debug cameras
        draw. Defaults to False.
    """

    history_length: NonNegativeInt = 0
    delay: NonNegativeFloat = 0.0
    jitter: NonNegativeFloat = 0.0
    draw_debug: StrictBool = False
    # -1 means not link-attached, so that SensorManager can sort uniformly
    entity_idx: EntityIdx | None = None

    @field_validator("entity_idx", mode="wrap")
    @classmethod
    def _normalize_entity_idx(cls, value: Any, handler: ValidatorFunctionWrapHandler) -> int:
        entity_idx = -1 if value is None else handler(value)
        if entity_idx < -1:
            gs.raise_exception(f"{cls.__name__}: entity_idx must be an entity, an index >= 0 or None, got {value}.")
        return entity_idx

    def model_post_init(self, context: Any) -> None:
        if self.jitter > self.delay:
            gs.raise_exception(f"{type(self).__name__}: Jitter must be less than or equal to read delay.")

    def _validate(self, sim: "Simulator"):
        # Checks needing the simulator. The ones on the values alone belong to model_post_init.
        if self.delay > 0:
            delay_ratio = self.delay / sim.dt
            delay_ts = round(delay_ratio)
            if not np.isclose(delay_ratio, delay_ts, atol=gs.EPS):
                gs.logger.warning(
                    f"{type(self).__name__}: Read delay should be a multiple of the simulation time step. Got "
                    f"{self.delay} and {sim.dt}. Actual read delay will be {delay_ts * sim.dt}."
                )


class KinematicSensorOptionsMixin[SensorT: "Sensor"](SensorOptions[SensorT]):
    """
    Base options class for sensors attached to a KinematicEntity (or any subclass, including RigidEntity).

    Use this base for sensors whose output is purely kinematic and does not depend on physics-derived quantities like
    contact forces or inertial dynamics.

    Parameters
    ----------
    link : KinematicLink | None, optional
        The link to which this sensor is attached, or None for a static sensor. It sets 'entity_idx' and
        'link_idx_local', and takes precedence over them when both are given. Defaults to None.
    entity_idx : Entity | int | None, optional
        Deprecated, use 'link'. The entity to which this sensor is attached, or its index in the scene, or None for a
        static sensor. Ignored if 'link' is also given. Defaults to None.
    link_idx_local : int, optional
        Deprecated, use 'link'. The index in 'entity_idx' of the link to which this sensor is attached. Ignored if
        'link' is also given. Defaults to 0, the base link.
    pos_offset : array-like[float, float, float], optional
        The positional offset of the sensor from the link.
    euler_offset : array-like[float, float, float], optional
        The rotational offset of the sensor from the link in degrees.
    """

    link_idx_local: NonNegativeInt = 0
    pos_offset: Vec3FType = (0.0, 0.0, 0.0)
    euler_offset: Vec3FType = (0.0, 0.0, 0.0)

    @model_validator(mode="before")
    @classmethod
    def _resolve_link(cls, data: Any) -> Any:
        # 'link' is resolved into the index fields, so that the options hold indices alone
        if not isinstance(data, dict):
            return data
        if "link" in data:
            if "entity_idx" in data or "link_idx_local" in data:
                gs.logger.warning(
                    f"{cls.__name__}: Both link and entity_idx or link_idx_local were provided, using link."
                )
            data = dict(data)
            link = data.pop("link")
            data.pop("link_idx_local", None)
            if link is None:
                data["entity_idx"] = None
            else:
                if not isinstance(link, gs.engine.entities.rigid_entity.KinematicLink):
                    gs.raise_exception(f"{cls.__name__}: link must be a link, got {type(link).__name__}.")
                data.update(entity_idx=link.entity.idx, link_idx_local=link.idx_local)
        return data

    def _validate(self, sim: "Simulator"):
        from genesis.engine.entities import KinematicEntity

        super()._validate(sim)
        if self.entity_idx >= 0:
            if self.entity_idx >= len(sim.entities):
                gs.raise_exception(f"Invalid entity index {self.entity_idx}.")
            entity = sim.entities[self.entity_idx]
            if not isinstance(entity, KinematicEntity):
                gs.raise_exception(f"Entity at index {self.entity_idx} is not a KinematicEntity.")
            if not 0 <= self.link_idx_local < entity.n_links:
                gs.raise_exception(f"Invalid link index {self.link_idx_local} for entity {self.entity_idx}.")


class RigidSensorOptionsMixin[SensorT: "Sensor"](KinematicSensorOptionsMixin[SensorT]):
    """
    Options for sensors that require a RigidEntity specifically (e.g. contact, contact force, IMU, tactile).

    Any sensor whose output depends on physics quantities (contact pairs, friction, inertial dynamics) belongs
    here.
    """

    def _validate(self, sim: "Simulator"):
        from genesis.engine.entities import RigidEntity

        super()._validate(sim)
        if self.entity_idx >= 0:
            entity = sim.entities[self.entity_idx]
            if not isinstance(entity, RigidEntity):
                gs.raise_exception(f"Entity at index {self.entity_idx} is not a RigidEntity.")


class RigidEntitySensorOptionsMixin[SensorT: "Sensor"](RigidSensorOptionsMixin[SensorT]):
    """
    Options for a sensor that measures nothing without a RigidEntity.

    The attachment is mandatory: entity_idx must refer to an existing RigidEntity, static sensors are not allowed. Such
    are the joint-space sensors, bound to a whole entity, and the sensors measuring a quantity of their link (contacts,
    contact force, temperature).

    The link offset parameters are inherited from RigidSensorOptionsMixin but ignored by joint-space sensors.
    """

    def _validate(self, sim: "Simulator"):
        super()._validate(sim)
        if self.entity_idx < 0:
            gs.raise_exception(f"{type(self).__name__} requires entity_idx >= 0, got {self.entity_idx}.")


class ContactFilterOptionsMixin[SensorT: "Sensor"](RigidSensorOptionsMixin[SensorT]):
    """
    Shared options for the contact sensors whose reading can be scoped to ignore contacts with chosen counterpart
    links -- Contact, ContactForce, and the contact-driven tactile probes (ContactProbe, ContactDepthProbe,
    KinematicTaxel).

    Parameters
    ----------
    filter_link_idx : Sequence[RigidLink | int], optional
        Links of rigid entities, or their global indices in the rigid solver. Contacts with the sensor link whose other
        participant is one of these links are ignored: the sensor behaves as if those counterparts were not touching.
        Use it to scope a reading to a chosen set of contacts (e.g. force on a foot from the ground, excluding
        self-contact). Default is empty, so every contact with the sensor link counts.
    """

    filter_link_idx: OptionalRigidLinksIdx = ()

    def _validate(self, sim: "Simulator"):
        super()._validate(sim)
        if self.filter_link_idx:
            n_links = sim.rigid_solver.n_links
            if np.any(np.array(self.filter_link_idx) < 0) or np.any(np.array(self.filter_link_idx) >= n_links):
                gs.raise_exception(
                    f"{type(self).__name__}: filter_link_idx must be in [0, {n_links}). Got {self.filter_link_idx}."
                )


class SimpleSensorOptions[SensorT: "Sensor"](SensorOptions[SensorT]):
    """
    Options carrying SimpleSensor's imperfection parameters.

    Interpreted by ``_apply_hardware_imperfections`` as perturbations introduced by the embedded sampler when it
    snapshots the sensor into shared memory. Inherited by every ``SimpleSensor``-derived options class; Camera
    (deriving from ``Sensor`` directly) stays on plain ``SensorOptions``.

    Parameters
    ----------
    resolution : float | array-like[float, ...], optional
        The measurement resolution of the sensor (smallest increment of change in the sensor reading).
        Default is 0.0, which means no quantization is applied.
    bias : float | array-like[float, ...], optional
        The constant additive bias of the sensor.
    noise : float | array-like[float, ...], optional
        The standard deviation of the additive white noise.
    random_walk : float | array-like[float, ...], optional
        The standard deviation of the random walk, which acts as accumulated bias drift.
    """

    resolution: FArrayType | float = 0.0
    bias: FArrayType | float = 0.0
    noise: FArrayType | float = 0.0
    random_walk: FArrayType | float = 0.0


class ProbeSensorOptionsMixin[SensorT: "Sensor"](SensorOptions[SensorT]):
    """
    Base options class for sensors that use local probe points.

    Parameters
    ----------
    probe_local_pos : array-like[array-like[float, float, float]] or shape ``(M, N, 3)`` grid
        Probe positions in link-local frame. Either a flat ``(N, 3)`` set or a 2D grid ``(M, N, 3)``; the
        ``read()`` output is reshaped back to match this layout.
    probe_radius : float | array-like[float] or shape ``(M, N)`` grid
        Probe sensing radius in meters. A scalar is shared by every probe; an array (or grid) must match the
        layout of ``probe_local_pos``.
    probe_radius_noise : float
        Additive radius noise in meters used by kernels whose measured branch depends on effective probe radius.
    debug_probe_color : array-like[float, float, float]
        RGB color for debug probe spheres (no alpha; the center sphere is drawn opaque and the outer sphere uses
        ``debug_probe_sphere_opacity``).
    debug_probe_center_radius : float
        Radius in meters of the small opaque marker sphere drawn at each probe position.
    debug_probe_sphere_opacity : float
        Alpha (0..1) of the outer translucent sphere drawn at each probe's sensing radius. Set to ``0.0`` to skip.
    """

    probe_local_pos: Vec3FArrayType | Vec3FGridType = ((0.0, 0.0, 0.0),)
    probe_radius: PositiveFloat | PositiveFArrayType | PositiveFGridType = 0.01
    probe_radius_noise: NonNegativeFloat = 0.0
    debug_probe_color: UnitIntervalVec3Type = (0.2, 0.4, 1.0)
    debug_probe_center_radius: PositiveFloat = 0.0008
    debug_probe_sphere_opacity: UnitInterval = 0.3

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        n_probes = int(np.prod(np.asarray(self.probe_local_pos).shape[:-1]))
        if isinstance(self.probe_radius, Sequence):
            if np.asarray(self.probe_radius).size != n_probes:
                gs.raise_exception(
                    f"probe_radius shape {np.asarray(self.probe_radius).shape} must contain "
                    f"{n_probes} entries to match probe_local_pos."
                )


class ProbesWithNormalSensorOptionsMixin[SensorT: "Sensor"](ProbeSensorOptionsMixin[SensorT]):
    """
    Probe options for sensors that also define one normal per probe, or one shared normal.
    """

    probe_local_normal: UnitVec3FType | UnitVec3FArrayType | UnitVec3FGridType = (0.0, 0.0, 1.0)

    @property
    def _is_probe_local_normal_required(self) -> bool:
        """Override in subclasses where ``probe_local_normal`` is only consumed by an opt-in mode (e.g. raycast
        contact-depth queries).

        When ``False``, the per-probe shape validation in ``model_post_init`` is skipped -- sensors that never read
        the normal don't surface confusing length errors for the default value.
        """
        return True

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        if not self._is_probe_local_normal_required:
            return
        n_probes = int(np.prod(np.asarray(self.probe_local_pos).shape[:-1]))
        normals = np.asarray(self.probe_local_normal)
        if normals.ndim > 1 and normals.size // 3 != n_probes:
            gs.raise_exception(
                "probe_local_normal must be one normal or contain one normal per probe. "
                f"Got normal shape {normals.shape} for {n_probes} probes."
            )


class JointTorque(RigidEntitySensorOptionsMixin["JointTorqueSensor"], SimpleSensorOptions["JointTorqueSensor"]):
    """
    Actuator output effort sensor for rigid entities (torque for revolute DOFs, force for prismatic DOFs).

    Models the generalized effort at each joint's gearbox output shaft:

        actuator_force = tau_control - armature * qacc + tau_frictionloss + tau_damping

    where ``qacc`` is the constraint-solved joint acceleration, ``armature`` the per-DOF armature inertia,
    ``tau_frictionloss`` the Coulomb friction constraint effort (negative when opposing motion), and
    ``tau_damping = -damping * vel`` the viscous passive effort. Gravity, Coriolis and contact loads are thus
    captured implicitly.

    Parameters
    ----------
    entity_idx : RigidEntity | int
        The rigid entity to sense, or its index in the scene.
    dofs_idx_local : array-like[int] | None, optional
        Local DOF indices within the entity. ``None`` (default) selects all DOFs.
    """

    dofs_idx_local: OptionalIArrayType | None = None

    def _validate(self, sim: "Simulator"):
        super()._validate(sim)
        entity = sim.entities[self.entity_idx]
        if entity.n_dofs == 0:
            gs.raise_exception(f"JointTorque: entity at index {self.entity_idx} has no DOFs.")
        if self.dofs_idx_local is not None and any(i < 0 or i >= entity.n_dofs for i in self.dofs_idx_local):
            gs.raise_exception(
                f"JointTorque: dofs_idx_local contains out-of-range indices for entity with {entity.n_dofs} DOFs."
            )


class Contact(
    RigidEntitySensorOptionsMixin["ContactSensor"],
    ContactFilterOptionsMixin["ContactSensor"],
    SimpleSensorOptions["ContactSensor"],
):
    """
    Sensor that returns bool based on whether associated RigidLink is in contact.

    Parameters
    ----------
    threshold : float, optional
        The bool-conversion threshold applied at read time to the underlying float contact magnitude
        (kernel produces float). A bin reads ``True`` iff its magnitude exceeds this value. Default
        ``0.0`` so any positive magnitude registers as contact.
    debug_sphere_radius : float, optional
        The radius of the debug sphere. Defaults to 0.05.
    debug_color : array-like[float, float, float, float], optional
        The rgba color of the debug sphere. Defaults to (1.0, 0.0, 1.0, 0.5).
    """

    threshold: NonNegativeFloat = 0.0
    debug_sphere_radius: PositiveFloat = 0.05
    debug_color: UnitIntervalVec4Type = (1.0, 0.0, 1.0, 0.5)


class ContactForce(
    RigidEntitySensorOptionsMixin["ContactForceSensor"],
    ContactFilterOptionsMixin["ContactForceSensor"],
    SimpleSensorOptions["ContactForceSensor"],
):
    """
    Sensor that returns the total contact force being applied to the associated RigidLink in its local frame.

    Parameters
    ----------
    min_force : float | array-like[float, float, float], optional
        The minimum detectable absolute force per each axis. Values below this will be treated as 0. Default is 0.
    max_force : float | array-like[float, float, float], optional
        The maximum output absolute force per each axis. Values above this will be clipped. Default is infinity.
    debug_color : array-like[float, float, float, float], optional
        The rgba color of the debug arrow. Defaults to (1.0, 0.0, 1.0, 0.5).
    debug_scale : float, optional
        The scale factor for the debug force arrow. Defaults to 0.01.
    """

    resolution: LaxVec3FType = 0.0

    min_force: LaxNonNegativeUnboundedVec3FType = 0.0
    max_force: LaxNonNegativeUnboundedVec3FType = np.inf

    debug_color: UnitIntervalVec4Type = (1.0, 0.0, 1.0, 0.5)
    debug_scale: PositiveFloat = 0.01

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        if np.any(np.array(self.max_force) <= np.array(self.min_force)):
            gs.raise_exception(f"min_force should be less than max_force, got: {self.min_force} and {self.max_force}")


class TemperatureProperties(NamedTuple):
    """
    Material properties for temperature sensor.

    Parameters
    ----------
    base_temperature: float
        The base temperature of the material in Celsius.
    conductivity: float
        The conductivity of the material in W/(m*K)
    density: float
        The density of the material in kilograms per cubic meter.
    specific_heat: float
        The specific heat of the material in J/(kg*C).
    emissivity: float
        The emissivity of the material, between 0 and 1.
    """

    base_temperature: float = 21.0
    conductivity: float = 50.0
    density: float = 1000.0
    specific_heat: float = 1.0
    emissivity: float = 0.9


class TemperatureGrid(
    RigidEntitySensorOptionsMixin["TemperatureGridSensor"], SimpleSensorOptions["TemperatureGridSensor"]
):
    """
    Sensor that returns the temperature in Celsius of the associated RigidLink in its local frame.

    Temperature is computed based on object contacts and their material properties provided to these options.

    Parameters
    ----------
    properties_dict: dict[int, TemperatureProperties]
        A dictionary which maps link indices to their temperature-related material properties. Key `-1` is
        used as the default for links not present in the dict; if omitted, unlisted links are ignored in contacts.
        This parameter is shared across all Temperature sensors (dicts will be merged).
    ambient_temperature: float
        The ambient temperature in Celsius. Default is 21 degrees C.
        This parameter is shared across all Temperature sensors (the last one set will be used).
    convection_coefficient: float
        Convection coefficient h in W/(m^2*K) for surface cooling. Default 1.0.
        This parameter is shared across all Temperature sensors (the last one set will be used).
    simulate_all_link_temperatures: bool
        If True, the temperatures of all links with temperature properties will be simulated.
        When False, other links are treated as adiabatic (no heat transfer, always at base temperature).
        This parameter is shared across all Temperature sensors (setting True for one sets it for all).
    grid_size: tuple[int, int, int]
        The size of the grid in the x, y, and z directions which determines the sensor resolution by spatially
        discretizing the bounding box of the rigid entity link.
    heat_generation: Grid3DFloatType | None
        The heat generation rate in Watts per square meter for each cell in the grid.
    sensor_time_constant: float
        The time constant of the sensor in seconds.
    contact_depth_weight: float
        The weight of the contact depth in the temperature calculation.
    debug_temperature_range: tuple[float, float], optional
        The range of temperatures to visualize in the debug mode. Defaults to (0.0, 100.0).
    """

    properties_dict: dict[int, TemperatureProperties] = Field(default_factory=dict)
    ambient_temperature: float | None = None
    convection_coefficient: float | None = None
    simulate_all_link_temperatures: bool = False

    grid_size: PositiveVec3IType = (1, 1, 1)
    heat_generation: Grid3DFloatType | None = None
    sensor_time_constant: NonNegativeFloat = 0.0
    contact_depth_weight: NonNegativeFloat = 1.0
    debug_temperature_range: Vec2FType = (0.0, 100.0)


class IMU(RigidSensorOptionsMixin["IMUSensor"], SimpleSensorOptions["IMUSensor"]):
    """
    IMU sensor returns the linear acceleration (accelerometer) and angular velocity (gyroscope)
    of the associated entity link.

    Parameters
    ----------
    acc_resolution : float, optional
        The measurement resolution of the accelerometer (smallest increment of change in the sensor reading).
        Default is 0.0, which means no quantization is applied.
    acc_cross_axis_coupling : float | array-like[float, float, float] | array-like with shape (3,3)
        Accelerometer axes alignment as a 3x3 rotation matrix, where diagonal elements represent alignment (0.0 to 1.0)
        for each axis, and off-diagonal elements account for cross-axis misalignment effects.
        - If a scalar is provided (float), all off-diagonal elements are set to the scalar value.
        - If a 3-element vector is provided (array-like[float, float, float]), off-diagonal elements are set.
        - If a full 3x3 matrix is provided, it is used directly.
    acc_bias : array-like[float, float, float]
        The constant additive bias for each axis of the accelerometer.
    acc_noise : array-like[float, float, float]
        The standard deviation of the white noise for each axis of the accelerometer.
    acc_random_walk : array-like[float, float, float]
        The standard deviation of the random walk, which acts as accumulated bias drift.
    gyro_resolution : float, optional
        The measurement resolution of the gyroscope (smallest increment of change in the sensor reading).
        Default is 0.0, which means no quantization is applied.
    gyro_cross_axis_coupling : float | array-like[float, float, float] | array-like with shape (3,3)
        Gyroscope axes alignment as a 3x3 rotation matrix, similar to `acc_cross_axis_coupling`.
    gyro_bias : array-like[float, float, float]
        The constant additive bias for each axis of the gyroscope.
    gyro_noise : array-like[float, float, float]
        The standard deviation of the white noise for each axis of the gyroscope.
    gyro_random_walk : array-like[float, float, float]
        The standard deviation of the bias drift for each axis of the gyroscope.
    mag_resolution : float, optional
        The measurement resolution of the magnetometer (smallest increment of change in the sensor reading).
        Default is 0.0, which means no quantization is applied.
    mag_cross_axis_coupling : float | array-like[float, float, float] | array-like with shape (3,3)
        Magnetometer axes alignment as a 3x3 rotation matrix, similar to `acc_cross_axis_coupling`.
    mag_bias : array-like[float, float, float]
        The constant additive bias for each axis of the magnetometer.
    mag_noise : array-like[float, float, float]
        The standard deviation of the white noise for each axis of the gyroscope.
    mag_random_walk : array-like[float, float, float]
        The standard deviation of the bias drift for each axis of the magnetometer.
    debug_acc_color : array-like[float, float, float, float], optional
        The rgba color of the debug acceleration arrow. Defaults to (1.0, 0.0, 0.0, 0.6).
    debug_acc_scale: float, optional
        The scale factor for the debug acceleration arrow. Defaults to 0.01.
    debug_gyro_color : array-like[float, float, float, float], optional
        The rgba color of the debug gyroscope arrow. Defaults to (0.0, 1.0, 0.0, 0.6).
    debug_gyro_scale: float, optional
        The scale factor for the debug gyroscope arrow. Defaults to 0.01.
    debug_mag_color : array-like[float, float, float, float], optional
        The rgba color of the debug magnetometer arrow. Defaults to (0.0, 0.0, 1.0, 0.6).
    debug_mag_scale: float, optional
        The scale factor for the debug magnetometer arrow. Defaults to 0.01.
    """

    # Accelerometer
    acc_resolution: LaxVec3FType = 0.0
    acc_cross_axis_coupling: CrossCouplingAxisType = 0.0
    acc_noise: LaxVec3FType = 0.0
    acc_bias: LaxVec3FType = 0.0
    acc_random_walk: LaxVec3FType = 0.0

    # Gyroscope
    gyro_resolution: LaxVec3FType = 0.0
    gyro_cross_axis_coupling: CrossCouplingAxisType = 0.0
    gyro_noise: LaxVec3FType = 0.0
    gyro_bias: LaxVec3FType = 0.0
    gyro_random_walk: LaxVec3FType = 0.0

    # Magnetometer
    mag_resolution: LaxVec3FType = 0.0
    mag_cross_axis_coupling: CrossCouplingAxisType = 0.0
    mag_noise: LaxVec3FType = 0.0
    mag_bias: LaxVec3FType = 0.0
    mag_random_walk: LaxVec3FType = 0.0
    magnetic_field: LaxVec3FType = (0.0, 0.0, 0.5)

    debug_acc_color: UnitIntervalVec4Type = (1.0, 0.0, 0.0, 0.6)
    debug_acc_scale: PositiveFloat = 0.01
    debug_gyro_color: UnitIntervalVec4Type = (0.0, 1.0, 0.0, 0.6)
    debug_gyro_scale: PositiveFloat = 0.01
    debug_mag_color: UnitIntervalVec4Type = (0.0, 0.0, 1.0, 0.6)
    debug_mag_scale: PositiveFloat = 0.5

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)

        # FIXME: Resolution should be made private or converted to properties in mixin to prevent setting them directly
        self.resolution = self.acc_resolution + self.gyro_resolution + self.mag_resolution
        self.bias = self.acc_bias + self.gyro_bias + self.mag_bias
        self.random_walk = self.acc_random_walk + self.gyro_random_walk + self.mag_random_walk
        self.noise = self.acc_noise + self.gyro_noise + self.mag_noise


class SurfaceDistanceProbe(
    RigidSensorOptionsMixin["SurfaceDistanceProbeSensor"],
    SimpleSensorOptions["SurfaceDistanceProbeSensor"],
    ProbeSensorOptionsMixin["SurfaceDistanceProbeSensor"],
):
    """
    Surface distance probe that reports nearest distances from probe positions to tracked mesh surfaces.

    The read() output will provide the distances, and the nearest points can be accessed with `sensor.nearest_points`.

    Attached to a rigid entity link. Takes a list of local probe positions and a list of global link indices
    to track; for each probe, outputs the distance and nearest point (world frame) to the closest mesh
    surface among the tracked links. If no mesh is within max_range, reports max_range and the probe
    position as nearest point.

    Parameters
    ----------
    probe_local_pos : array-like[array-like[float, float, float]]
        Probe positions in link-local frame. One (x, y, z) per probe.
    probe_radius : float | array-like[float]
        Maximum sensing range in meters. When no mesh is within this distance, distance is clamped to the probe
        radius and nearest points is the probe position. Default: 0.5. Also controls the outer debug sphere.
    tracked_links : Sequence[RigidLink], optional
        The links of rigid entities whose mesh geoms are used for distance queries. It sets 'track_link_idx', and
        takes precedence over it when both are given. Defaults to none.
    track_link_idx : array-like[int], optional
        Deprecated, use 'tracked_links'. The global indices in the rigid solver of the tracked links. Defaults to none.
    """

    probe_radius: PositiveFArrayType | PositiveFloat = 0.5
    track_link_idx: IArrayType = Field(default_factory=tuple)

    @model_validator(mode="before")
    @classmethod
    def _resolve_tracked_links(cls, data: Any) -> Any:
        return _resolve_tracked_links(cls.__name__, data)

    def _validate(self, sim: "Simulator"):
        super()._validate(sim)
        n_links = sim.rigid_solver.n_links
        for i, link_idx in enumerate(self.track_link_idx):
            if not (0 <= link_idx < n_links):
                gs.raise_exception(
                    f"SurfaceDistanceProbe track_link_idx[{i}]={link_idx} is out of range [0, {n_links})."
                )


class Raycaster(KinematicSensorOptionsMixin["RaycasterSensor"], SimpleSensorOptions["RaycasterSensor"]):
    """
    Raycaster sensor that performs ray casting to get distance measurements and point clouds.

    Parameters
    ----------
    pattern: RaycastPatternOptions
        The raycasting pattern for the sensor.
    min_range : float, optional
        The minimum sensing range in meters. Defaults to 0.0.
    max_range : float, optional
        The maximum sensing range in meters. Defaults to 20.0.
    no_hit_value : float, optional
        The value to return for no hit. Defaults to max_range if not specified.
    return_world_frame : bool, optional
        Whether to return points in the world frame. Defaults to False (local frame).
    return_points : bool, optional
        Whether to return the per-ray hit points. Defaults to True. When False, ``read().points`` is None and only
        the hit distances are measured, cutting the sensor's memory footprint and per-step cost to about a quarter.
        Pick False for distance-only sensing (e.g. depth images); keep True when the point cloud is needed.
    debug_sphere_radius: float, optional
        The radius of each debug sphere drawn in the scene. Defaults to 0.02.
    debug_ray_start_color: array-like[float, float, float, float], optional
        The color of each debug ray start sphere drawn in the scene. Defaults to (0.5, 0.5, 1.0, 1.0).
    debug_ray_hit_color: array-like[float, float, float, float], optional
        The color of each debug ray hit point sphere drawn in the scene. Defaults to (1.0, 0.5, 0.5, 1.0).
    """

    pattern: RaycastPattern
    min_range: NonNegativeFloat = 0.0
    max_range: PositiveFloat = 20.0
    no_hit_value: float | None = None
    return_world_frame: StrictBool = False
    return_points: StrictBool = True

    debug_sphere_radius: PositiveFloat = 0.02
    debug_ray_start_color: Vec4FType = (0.5, 0.5, 1.0, 1.0)
    debug_ray_hit_color: Vec4FType = (1.0, 0.5, 0.5, 1.0)

    def model_post_init(self, context: Any) -> None:
        if self.no_hit_value is None:
            self.no_hit_value = self.max_range
        if self.max_range <= self.min_range:
            gs.raise_exception(
                f"[{type(self).__name__}] max_range {self.max_range} should be greater than min_range {self.min_range}."
            )


class DepthCamera(Raycaster):
    """
    Depth camera that uses ray casting to obtain depth images.

    Parameters
    ----------
    pattern: DepthCameraPattern
        The raycasting pattern configuration for the sensor.
    """

    pattern: DepthCameraPattern

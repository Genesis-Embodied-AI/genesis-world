from typing import TYPE_CHECKING, Annotated, Any, Generic, NamedTuple, Sequence, TypeVar

import numpy as np
from pydantic import BeforeValidator, Field, StrictBool, StrictInt, field_validator

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
)

from ..options import Options
from .raycaster import DepthCameraPattern, RaycastPattern

if TYPE_CHECKING:
    from genesis.engine.scene import Scene
    from genesis.engine.sensors.base_sensor import Sensor
    from genesis.engine.sensors.contact_force import ContactForceSensor, ContactSensor
    from genesis.engine.sensors.imu import IMUSensor
    from genesis.engine.sensors.raycaster import RaycasterSensor
    from genesis.engine.sensors.surface_distance_probe import SurfaceDistanceProbeSensor
    from genesis.engine.sensors.temperature import TemperatureGridSensor

    NonNegativeUnboundedFloat = float
    LaxNonNegativeUnboundedVec3FType = Vec3FType | float
else:
    NonNegativeUnboundedFloat = Annotated[float, Field(ge=0, strict=False)]
    LaxNonNegativeUnboundedVec3FType = Annotated[
        tuple[NonNegativeUnboundedFloat, NonNegativeUnboundedFloat, NonNegativeUnboundedFloat],
        BeforeValidator(lambda v: v if is_sequence(v) else (v,) * 3),
        Field(strict=False),
    ]
CrossCouplingAxisType = RotationMatrixType | UnitIntervalVec3Type | float


SensorT = TypeVar("SensorT", bound="Sensor")


def _check_len_match(value, expected_len: int, name: str, ref_name: str):
    if isinstance(value, Sequence) and len(value) != expected_len:
        gs.raise_exception(
            f"{name} must have the same length as {ref_name} when {name} is array-like. "
            f"Got {len(value)} {name} and {expected_len} {ref_name}."
        )


class SensorOptions(Options, Generic[SensorT]):
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
        If True and visualizer is active, the sensor will draw debug shapes in the scene. Defaults to False.
    """

    history_length: NonNegativeInt = 0
    delay: NonNegativeFloat = 0.0
    jitter: NonNegativeFloat = 0.0
    draw_debug: StrictBool = False
    # -1 means not link-attached. None is accepted from users and normalized to -1 so SensorManager can sort uniformly.
    entity_idx: StrictInt = Field(default=-1, ge=-1)

    @field_validator("entity_idx", mode="before")
    @classmethod
    def _normalize_entity_idx(cls, value):
        return -1 if value is None else value

    def model_post_init(self, context: Any) -> None:
        if self.jitter > self.delay:
            gs.raise_exception(f"{type(self).__name__}: Jitter must be less than or equal to read delay.")

    def validate_scene(self, scene: "Scene"):
        """
        Validate the sensor options values before the sensor is added to the scene.

        Use pydantic's model_post_init() for validation that does not require scene context.
        """
        assert scene.sim is not None
        if self.delay > 0:
            delay_ratio = self.delay / scene.sim.dt
            delay_ts = round(delay_ratio)
            if not np.isclose(delay_ratio, delay_ts, atol=gs.EPS):
                gs.logger.warning(
                    f"{type(self).__name__}: Read delay should be a multiple of the simulation time step. Got "
                    f"{self.delay} and {scene.sim.dt}. Actual read delay will be {delay_ts * scene.sim.dt}."
                )


class KinematicSensorOptionsMixin(SensorOptions[SensorT]):
    """
    Base options class for sensors attached to a KinematicEntity (or any subclass, including RigidEntity).

    Use this base for sensors whose output is purely kinematic and does not depend on physics-derived quantities like
    contact forces or inertial dynamics.

    Parameters
    ----------
    entity_idx : int
        The global entity index of the entity to which this sensor is attached. -1 or None for static sensors.
    link_idx_local : int, optional
        The local index of the link of the entity to which this sensor is attached.
    pos_offset : array-like[float, float, float], optional
        The positional offset of the sensor from the link.
    euler_offset : array-like[float, float, float], optional
        The rotational offset of the sensor from the link in degrees.
    """

    link_idx_local: NonNegativeInt = 0
    pos_offset: Vec3FType = (0.0, 0.0, 0.0)
    euler_offset: Vec3FType = (0.0, 0.0, 0.0)

    def validate_scene(self, scene: "Scene"):
        from genesis.engine.entities import KinematicEntity

        super().validate_scene(scene)
        if self.entity_idx >= 0:
            if self.entity_idx >= len(scene.entities):
                gs.raise_exception(f"Invalid entity index {self.entity_idx}.")
            entity = scene.entities[self.entity_idx]
            if not isinstance(entity, KinematicEntity):
                gs.raise_exception(f"Entity at index {self.entity_idx} is not a KinematicEntity.")
            if self.link_idx_local >= entity.n_links:
                gs.raise_exception(f"Invalid link index {self.link_idx_local} for entity {self.entity_idx}.")


class RigidSensorOptionsMixin(KinematicSensorOptionsMixin[SensorT]):
    """
    Options for sensors that require a RigidEntity specifically (e.g. contact, contact force, IMU, tactile).

    Any sensor whose output depends on physics quantities (contact pairs, friction, inertial dynamics) belongs
    here.
    """

    def validate_scene(self, scene: "Scene"):
        from genesis.engine.entities import RigidEntity

        super().validate_scene(scene)
        if self.entity_idx >= 0:
            entity = scene.entities[self.entity_idx]
            if not isinstance(entity, RigidEntity):
                gs.raise_exception(f"Entity at index {self.entity_idx} is not a RigidEntity.")


class RigidEntitySensorOptionsMixin(RigidSensorOptionsMixin[SensorT]):
    """
    Options for a sensor bound to a whole RigidEntity (e.g. joint-space sensors), where the attachment is mandatory:
    entity_idx must refer to an existing RigidEntity, static sensors are not allowed.

    The link offset parameters are inherited from RigidSensorOptionsMixin but ignored by joint-space sensors.
    """

    def validate_scene(self, scene: "Scene"):
        super().validate_scene(scene)
        if self.entity_idx < 0:
            gs.raise_exception(f"{type(self).__name__} requires entity_idx >= 0, got {self.entity_idx}.")


class ContactFilterOptionsMixin(RigidSensorOptionsMixin[SensorT]):
    """
    Shared options for the contact sensors whose reading can be scoped to ignore contacts with chosen counterpart
    links -- Contact, ContactForce, and the contact-driven tactile probes (ContactProbe, ContactDepthProbe,
    KinematicTaxel).

    Parameters
    ----------
    filter_link_idx : array-like[int], optional
        Global rigid link indices (solver link space). Contacts with the sensor link whose other participant is one
        of these links are ignored: the sensor behaves as if those counterparts were not touching. Use it to scope a
        reading to a chosen set of contacts (e.g. force on a foot from the ground, excluding self-contact). Default
        is empty, so every contact with the sensor link counts.
    """

    filter_link_idx: OptionalIArrayType = Field(default_factory=tuple)

    def validate_scene(self, scene: "Scene"):
        super().validate_scene(scene)
        if self.filter_link_idx:
            n_links = scene.sim.rigid_solver.n_links
            if np.any(np.array(self.filter_link_idx) < 0) or np.any(np.array(self.filter_link_idx) >= n_links):
                gs.raise_exception(
                    f"{type(self).__name__}: filter_link_idx must be in [0, {n_links}). Got {self.filter_link_idx}."
                )


class SimpleSensorOptions(SensorOptions[SensorT]):
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


class ProbeSensorOptionsMixin(SensorOptions[SensorT]):
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


class ProbesWithNormalSensorOptionsMixin(ProbeSensorOptionsMixin[SensorT]):
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
    entity_idx : int
        Scene-level index of the RigidEntity to sense. Must be >= 0.
    dofs_idx_local : array-like[int] | None, optional
        Local DOF indices within the entity. ``None`` (default) selects all DOFs.
    """

    dofs_idx_local: OptionalIArrayType | None = None

    def validate_scene(self, scene: "Scene"):
        super().validate_scene(scene)
        entity = scene.entities[self.entity_idx]
        if entity.n_dofs == 0:
            gs.raise_exception(f"JointTorque: entity at index {self.entity_idx} has no DOFs.")
        if self.dofs_idx_local is not None and any(i < 0 or i >= entity.n_dofs for i in self.dofs_idx_local):
            gs.raise_exception(
                f"JointTorque: dofs_idx_local contains out-of-range indices for entity with {entity.n_dofs} DOFs."
            )


class Contact(ContactFilterOptionsMixin["ContactSensor"], SimpleSensorOptions["ContactSensor"]):
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


class ContactForce(ContactFilterOptionsMixin["ContactForceSensor"], SimpleSensorOptions["ContactForceSensor"]):
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


class TemperatureGrid(RigidSensorOptionsMixin["TemperatureGridSensor"], SimpleSensorOptions["TemperatureGridSensor"]):
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
    track_link_idx : array-like[int]
        Global link indices (solver link space) whose mesh geoms are used for distance queries.
    """

    probe_radius: PositiveFArrayType | PositiveFloat = 0.5
    track_link_idx: IArrayType = Field(default_factory=tuple)

    def validate_scene(self, scene: "Scene"):
        super().validate_scene(scene)
        n_links = scene.sim.rigid_solver.n_links
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

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from pydantic import Field, StrictBool

import genesis as gs
from genesis.typing import (
    FArrayType,
    FGridType,
    IArrayType,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFArrayType,
    PositiveFloat,
    PositiveInt,
    PositiveVec2FType,
    UnitIntervalVec3Type,
    UnitIntervalVec4Type,
    Vec2FType,
)

from .options import (
    ContactFilterOptionsMixin,
    ProbeSensorOptionsMixin,
    ProbesWithNormalSensorOptionsMixin,
    RigidSensorOptionsMixin,
    SensorOptions,
    SensorT,
    SimpleSensorOptions,
    _check_len_match,
)

if TYPE_CHECKING:
    from genesis.engine.sensors.kinematic_tactile import (
        ContactDepthProbeSensor,
        ContactProbeSensor,
        ElastomerTaxelSensor,
        KinematicTaxelSensor,
        ProximityTaxelSensor,
    )


def _validate_filler_probe_radius(probe_radius, sensor_name: str) -> None:
    """
    Validate a ``probe_radius`` that permits 0-valued (inactive padding for grid) entries.
    """
    radii = np.atleast_1d(np.asarray(probe_radius, dtype=float))
    if np.any(radii < 0.0):
        gs.raise_exception(f"{sensor_name} probe_radius entries must be non-negative. Got {probe_radius}.")
    if not np.any(radii > 0.0):
        gs.raise_exception(f"{sensor_name} requires at least one positive probe_radius. Got {probe_radius}.")


class ViscoelasticHysteresisOptionsMixin(SensorOptions[SensorT]):
    """
    Single-Maxwell viscoelastic hysteresis applied on the measured branch only.

    Output equals ``x + hysteresis_strength * xi``, where ``xi`` is a per-cache-column state with
    ``xi_k = exp(-dt / hysteresis_tau) * xi_{k-1} + (x_k - x_{k-1})``. Equilibrium gain is 1 (steady-state output =
    steady-state input). On a step input, output transiently overshoots by ``strength``, decaying with time constant
    ``tau``. On cyclic input this gives a loading-unloading loop in output-vs-input space.

    Parameters
    ----------
    hysteresis_strength : float, optional
        Dimensionless ratio of the Maxwell branch to the equilibrium branch (``E_1 / E_inf`` with ``E_inf = 1``).
        ``0`` disables hysteresis. Default ``0``.
    hysteresis_tau : float, optional
        Relaxation time constant in seconds. Must be positive when ``hysteresis_strength > 0``.
    """

    hysteresis_strength: NonNegativeFloat = 0.0
    hysteresis_tau: NonNegativeFloat = 0.0

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        if self.hysteresis_strength > 0.0 and self.hysteresis_tau <= 0.0:
            gs.raise_exception(
                f"hysteresis_tau ({self.hysteresis_tau}) must be > 0 when hysteresis_strength "
                f"({self.hysteresis_strength}) > 0."
            )


def _validate_crosstalk_kernel(kernel) -> None:
    """
    Validate an explicit crosstalk kernel: 2D ``(N, M)`` or 3D ``(G, N, M)`` with ``G`` in {1, 2, 3}, odd spatial dims
    (center tap), and finite entries.

    The conservation (sum ~ 1) check is a build-time warning in ``tactile_shared.build_crosstalk_kernels`` (where the
    logger is available); the self tap is intentionally allowed to be < 1, which is how a conservative kernel shares a
    peak with its neighbors.
    """
    arr = np.asarray(kernel, dtype=float)
    if arr.ndim not in (2, 3):
        gs.raise_exception(f"crosstalk_kernel must be 2D (N, M) or 3D (G, N, M); got shape {arr.shape}.")
    if arr.ndim == 3 and arr.shape[0] not in (1, 2, 3):
        gs.raise_exception(f"crosstalk_kernel leading (group) dim must be 1, 2, or 3; got {arr.shape[0]}.")
    n_rows, n_cols = int(arr.shape[-2]), int(arr.shape[-1])
    if n_rows % 2 == 0 or n_cols % 2 == 0:
        gs.raise_exception(f"crosstalk_kernel spatial dims must be odd (center tap); got ({n_rows}, {n_cols}).")
    if not np.all(np.isfinite(arr)):
        gs.raise_exception("crosstalk_kernel entries must be finite.")


class SpatialCrosstalkOptionsMixin(SensorOptions[SensorT]):
    """
    Grid spatial crosstalk applied on the measured branch: each taxel's force/torque bleeds onto its grid neighbors,
    modeling mechanical coupling through the sensor's compliant layer.

    Requires a 2D grid ``probe_local_pos`` (shape ``(ny, nx, 3)`` with non-degenerate spacing); a 0-radius filler probe
    stays zero.

    Configure it one of two ways. A **Gaussian** blur via ``crosstalk_strength`` + ``crosstalk_sigma`` (one isotropic
    kernel on all 6 force/torque channels). Or an **explicit kernel** via ``crosstalk_kernel`` -- a measured
    point-spread function used as-is, optionally per force/torque mode. ``genesis.utils.misc.gaussian_crosstalk_kernel``
    builds a kernel of a given size.

    Parameters
    ----------
    crosstalk_strength : float, optional
        Gaussian crosstalk mixing fraction. ``0`` (default) disables; ``1`` is a pure Gaussian blur with sigma
        ``crosstalk_sigma``. Mutually exclusive with ``crosstalk_kernel``.
    crosstalk_sigma : float, optional
        Gaussian standard deviation in meters (same units as ``probe_local_pos`` spacing). Must be > 0 when
        ``crosstalk_strength > 0``.
    crosstalk_kernel : array-like, optional
        Explicit convolution kernel in taxel-grid-cell units (odd dims; center = self weight), used as-is so the
        weights encode self-vs-neighbor coupling and conservation. Shape selects channel grouping: ``(N, M)`` one
        kernel for all 6 channels; ``(2, N, M)`` ``[normal force, shear force + torque]``; ``(3, N, M)``
        ``[normal force, shear force, torque]``. Normal/shear are split along the grid normal. Mutually exclusive
        with ``crosstalk_strength``.
    """

    crosstalk_strength: NonNegativeFloat = 0.0
    crosstalk_sigma: NonNegativeFloat = 0.0
    # 2D (N, M) or 3D (G, N, M) float array; shape is validated in model_post_init (pydantic's nested-tuple unions
    # do not cleanly disambiguate the 2D/3D float case).
    crosstalk_kernel: Any | None = None

    @property
    def is_crosstalk_enabled(self) -> bool:
        return self.crosstalk_strength > 0.0 or self.crosstalk_kernel is not None

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        if self.crosstalk_strength > 0.0 and self.crosstalk_kernel is not None:
            gs.raise_exception("Set only one of crosstalk_strength (Gaussian) or crosstalk_kernel (explicit).")
        if self.crosstalk_strength > 0.0 and self.crosstalk_sigma <= 0.0:
            gs.raise_exception(
                f"crosstalk_sigma ({self.crosstalk_sigma}) must be > 0 when crosstalk_strength "
                f"({self.crosstalk_strength}) > 0."
            )
        if self.crosstalk_kernel is not None:
            _validate_crosstalk_kernel(self.crosstalk_kernel)


class TactileProbeSensorOptionsMixin(ProbeSensorOptionsMixin[SensorT]):
    """
    Tactile probe sensors estimate contact from geometric depth queries (SDF or raycast) around each probe position
    rather than reading the physics solver's contact impulses, so they sense at arbitrary probe locations without
    affecting simulation.

    Parameters
    ----------
    debug_contact_color: array-like[float, float, float]
        RGB color of the debug probe spheres while in contact.
    probe_gain : float | array-like[float], optional
        Per-taxel multiplicative gain applied to the measured-branch contact depth. Default ``1.0`` (no gain). Accepts
        a scalar (applied to all probes) or an array matching the probe count. Force/torque scale as
        ``gain**normal_exponent`` because the spring-damper sees the gained depth.
    probe_gain_resample_range : (float, float), optional
        If set, the per-probe gain is resampled uniformly in ``(low, high)`` on every ``scene.reset()``. Disables the
        static ``probe_gain`` after the first reset. Default ``None`` (no resampling; gain stays at initial value).
    dead_taxel_probability : float, optional
        Per-probe Bernoulli probability that the taxel becomes dead on each ``scene.reset()``. Default ``0.0``
        (no dead taxels). When set, the intermediate-cache value for dead probes is overwritten by a fresh
        per-probe uniform sample in ``dead_taxel_value_range`` (the same value fills every output channel of that
        probe) at the hardware-imperfections stage; the GT branch is untouched.
    dead_taxel_value_range : (float, float), optional
        Uniform range for the dead value sampled per probe on reset. Default ``(0.0, 0.0)``.
    contact_depth_query : {"sdf", "raycast"} or None, optional
        Per-probe contact-depth backend. ``"sdf"`` queries the per-geom analytic SDF grid (fast, exact for primitives,
        requires SDF activation). ``"raycast"`` walks the rigid solver's per-frame collision-mesh BVH and takes the
        signed distance to the nearest candidate triangle (sign from the triangle's face normal, negative inside),
        so ``pen = R - signed_distance`` matches the SDF backend while handling arbitrary meshes uniformly (shares
        the BVH with ``RaycasterSensor``). ``None`` (default) defers the choice: all sensors of the same class must
        agree, and the resolved mode is ``"sdf"`` if no sensor of that class sets it.
    """

    debug_contact_color: UnitIntervalVec3Type = (1.0, 0.2, 0.0)

    probe_gain: PositiveFArrayType | PositiveFloat = 1.0
    probe_gain_resample_range: PositiveVec2FType | None = None
    dead_taxel_probability: NonNegativeFloat = 0.0
    dead_taxel_value_range: Vec2FType = (0.0, 0.0)
    contact_depth_query: Literal["sdf", "raycast"] | None = None

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        n_probes = int(np.prod(np.asarray(self.probe_local_pos).shape[:-1]))
        _check_len_match(self.probe_gain, n_probes, "probe_gain", "probe_local_pos")

        if self.probe_gain_resample_range is not None:
            low, high = float(self.probe_gain_resample_range[0]), float(self.probe_gain_resample_range[1])
            if low > high:
                gs.raise_exception(f"probe_gain_resample_range must satisfy low <= high. Got ({low}, {high}).")
        if self.dead_taxel_probability > 1.0:
            gs.raise_exception(f"dead_taxel_probability must be in [0, 1]. Got {self.dead_taxel_probability}.")
        low, high = float(self.dead_taxel_value_range[0]), float(self.dead_taxel_value_range[1])
        if low > high:
            gs.raise_exception(f"dead_taxel_value_range must satisfy low <= high. Got ({low}, {high}).")


class PointCloudTactileSensorMixin(TactileProbeSensorOptionsMixin[SensorT]):
    """
    Options mixin for tactile sensors that sample a point cloud from tracked link meshes.

    Parameters
    ----------
    track_link_idx : array-like[int]
        Global link indices whose mesh geometry is used to sample a point cloud from.
    n_sample_points: int | array-like[int]
        Total FPS samples split across ``track_link_idx``, or one count per tracked link. Per-variant
        counts are not supported: when a tracked link belongs to a heterogeneous entity, the per-link
        count is allocated to every variant on that link (so each parallel environment sees the full
        count regardless of which variant is active).
    use_visual_mesh : bool
        Whether to use the visual mesh when sampling the point cloud.
    debug_point_cloud_color : array-like[float, float, float, float]
        The rgba color of the debug tracked object point cloud spheres.
    debug_point_cloud_radius : float
        The radius of the debug tracked object point cloud spheres.
    """

    track_link_idx: IArrayType = Field(default_factory=tuple)
    n_sample_points: IArrayType | NonNegativeInt = 500
    use_visual_mesh: StrictBool = True

    debug_point_cloud_color: UnitIntervalVec4Type = (1.0, 0.8, 0.0, 1.0)
    debug_point_cloud_radius: PositiveFloat = 0.002


class ContactHysteresisOptionsMixin(SensorOptions[SensorT]):
    """
    Schmitt-trigger contact gate shared by the stateful tactile sensors, in penetration-depth units.

    Depth is in meters and backend-agnostic (both the ``"sdf"`` and ``"raycast"`` contact-depth backends produce the
    same depth): the gate latches ON when depth >= ``contact_threshold`` and releases when depth <=
    ``release_threshold``, and the band between them suppresses chatter. What the gate controls is sensor-specific
    (ContactProbe: the boolean output; ElastomerTaxel: the shear anchor state).

    Parameters
    ----------
    contact_threshold : float
        Penetration depth (meters) at or above which the gate latches on.
    release_threshold : float, optional
        Penetration depth (meters) at or below which the gate releases. May be negative (the surface must
        *separate* by that margin to release). Must be <= ``contact_threshold``. ``None`` (default) equals
        ``contact_threshold`` (no hysteresis).
    """

    contact_threshold: NonNegativeFloat = 0.0001
    release_threshold: float | None = None

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        if self.release_threshold is not None and self.release_threshold > self.contact_threshold:
            gs.raise_exception(
                f"release_threshold ({self.release_threshold}) must be <= contact_threshold ({self.contact_threshold})."
            )


class ContactProbe(
    ContactFilterOptionsMixin["ContactProbeSensor"],
    SimpleSensorOptions["ContactProbeSensor"],
    TactileProbeSensorOptionsMixin["ContactProbeSensor"],
    ViscoelasticHysteresisOptionsMixin["ContactProbeSensor"],
    ContactHysteresisOptionsMixin["ContactProbeSensor"],
):
    """
    Returns boolean contact per probe based on the contact depth threshold, gated by the ``contact_threshold`` /
    ``release_threshold`` Schmitt trigger (see ``ContactHysteresisOptionsMixin``).

    Note
    ----
    The depth query only runs against geometry the rigid solver reports as in contact with the sensor's link (the
    depth itself comes from an SDF/raycast query, not the contact impulse, but contact *existence* is gated by the
    solver). Since the solver skips collision between two fixed entities, a sensor on a fixed entity will not detect
    contacts with other fixed entities.

    Parameters
    ----------
    probe_radius : float | array-like[float] or shape ``(M, N)`` grid
        Probe sensing radius in meters. A scalar is shared by every probe; an array (or grid) must match the
        layout of ``probe_local_pos``. Array entries of ``0`` mark inactive filler probes -- they always read
        ``False`` and skip the SDF query -- so an irregular taxel set can be padded into a regular grid.
    """

    # Permits 0-valued (inactive filler) entries; see _validate_filler_probe_radius.
    probe_radius: PositiveFloat | FArrayType | FGridType = 0.01

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        _validate_filler_probe_radius(self.probe_radius, "ContactProbe")


class ContactDepthProbe(
    ContactFilterOptionsMixin["ContactDepthProbeSensor"],
    SimpleSensorOptions["ContactDepthProbeSensor"],
    TactileProbeSensorOptionsMixin["ContactDepthProbeSensor"],
    ViscoelasticHysteresisOptionsMixin["ContactDepthProbeSensor"],
):
    """
    Returns contact depth in meters per probe.

    Note
    ----
    The depth query only runs against geometry the rigid solver reports as in contact with the sensor's link (the
    depth itself comes from an SDF/raycast query, not the contact impulse, but contact *existence* is gated by the
    solver). Since the solver skips collision between two fixed entities, a sensor on a fixed entity will not detect
    contacts with other fixed entities.
    """


class KinematicTaxel(
    ContactFilterOptionsMixin["KinematicTaxelSensor"],
    SimpleSensorOptions["KinematicTaxelSensor"],
    TactileProbeSensorOptionsMixin["KinematicTaxelSensor"],
    ViscoelasticHysteresisOptionsMixin["KinematicTaxelSensor"],
    SpatialCrosstalkOptionsMixin["KinematicTaxelSensor"],
):
    """
    A tactile sensor which estimates force and torque per taxel by querying contact depth within the radius of the
    probe positions along a rigid entity link and the relative velocity of the probe and the entity in contact.

    The force and torque are aligned with the contact surface normal ``n`` at each probe -- the SDF gradient in
    ``"sdf"`` mode, or the nearest-triangle face normal in ``"raycast"`` mode. The returned force is a spring-damper
    estimate based on contact depth and relative motion:
        v_n = dot(relative_velocity, n) * n
        v_t = relative_velocity - v_n
        s = penetration ** normal_exponent
        F = (normal_stiffness * s * n) + (normal_damping * s * dot(relative_velocity, n) * n) - (shear_scalar * v_t)
        T = cross(probe_local_pos, F) - twist_scalar * dot(relative_angular_velocity, n) * n
    as opposed to the actual impulse force on the link from the contact obtained from the physics solver.

    Note
    ----
    The depth query only runs against geometry the rigid solver reports as in contact with the sensor's link (the
    force/torque come from the spring-damper estimate above, not the contact impulse, but contact *existence* is
    gated by the solver). Since the solver skips collision between two fixed entities, a sensor on a fixed entity
    will not detect contacts with other fixed entities.

    ``probe_local_pos`` may be either an arbitrary set of probes with shape ``(N, 3)`` or a grid-shaped set with shape
    ``(M, N, 3)``. Regular planar grids enable spatial crosstalk on the measured branch (see ``crosstalk_strength``).
    A probe whose ``probe_radius`` is 0 is treated as an inactive filler -- it reads 0 force/torque and is skipped --
    so an irregular taxel set can be padded into a regular grid for crosstalk.

    Parameters
    ----------
    probe_radius : float | array-like[float]
        Probe sensing radius in meters. A scalar is shared by every probe; an array must match the probe count.
        Array entries of 0 mark inactive filler probes (see the grid note above); at least one must be positive.
    normal_stiffness : float
        Stiffness for normal force estimation based on contact penetration depth and spring-damper model.
    normal_damping : float
        Damping for normal force estimation based on contact penetration depth and spring-damper model.
    normal_exponent : float
        Exponent for contact force estimation based on contact penetration depth and nonlinear spring-damper model.
        Default is 1.0, which means linear spring-damper model. Use 1.5 for Hertzian (spherical) contact.
    shear_scalar : float, optional
        Coefficient for shear force estimation based on relative linear velocity of the probe and entity in contact.
    twist_scalar : float, optional
        Coefficient for twist torque estimation based on relative angular velocity of the probe and entity in contact.

    See ``SpatialCrosstalkOptionsMixin`` for the measured-branch spatial crosstalk parameters
    (``crosstalk_strength`` / ``crosstalk_sigma`` / ``crosstalk_kernel``); a regular planar grid is required.
    """

    # Permits 0-valued (inactive filler) entries; see _validate_filler_probe_radius.
    probe_radius: PositiveFloat | FArrayType | FGridType = 0.01

    normal_stiffness: NonNegativeFloat = 1000.0
    normal_damping: NonNegativeFloat = 1.0
    normal_exponent: NonNegativeFloat = 1.0
    shear_scalar: NonNegativeFloat = 1.0
    twist_scalar: NonNegativeFloat = 1.0

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)

        _validate_filler_probe_radius(self.probe_radius, "KinematicTaxel")
        if self.normal_exponent < 1.0:
            gs.raise_exception(f"normal_exponent must be greater than or equal to 1.0. Got {self.normal_exponent}.")


class ElastomerTaxel(
    RigidSensorOptionsMixin["ElastomerTaxelSensor"],
    SimpleSensorOptions["ElastomerTaxelSensor"],
    PointCloudTactileSensorMixin["ElastomerTaxelSensor"],
    ProbesWithNormalSensorOptionsMixin["ElastomerTaxelSensor"],
    ViscoelasticHysteresisOptionsMixin["ElastomerTaxelSensor"],
    ContactHysteresisOptionsMixin["ElastomerTaxelSensor"],
):
    """
    An elastomer tactile sensor that implements HydroShear-style marker displacement from Genesis SDF queries.

    The tracked rigid links are sampled into indenter on-surface points for shear history, while marker dilation is
    queried directly from the tracked geometry SDF.

    Note
    ----
    ``probe_local_pos`` may be either an arbitrary set of probes with shape ``(N, 3)`` or a grid-shaped set with shape
    ``(M, N, 3)``. Regular planar grids with one shared normal use FFT acceleration for dilation; other layouts use the
    direct dilation path. Shear is computed directly. A probe whose ``probe_radius`` is 0 is treated as an inactive
    filler -- it reads 0 and is excluded from dilation/shear -- so an irregular taxel set can be padded into a
    regular grid for FFT acceleration.

    Note
    ----
    ``probe_gain`` is applied to ElastomerTaxel as a post-step linear scale of the measured marker displacement
    (the dilation kernel writes a single shared field for both branches). This is exact for the tangential
    dilation and shear components but approximate for the normal dilation term, which scales as
    ``depth**normal_exponent`` and would ideally scale as ``gain**normal_exponent`` rather than ``gain``. For
    gains near 1 the error is small.

    Parameters
    ----------
    probe_local_pos: array-like[array-like[float, float, float]], shape (N, 3) or (M, N, 3)
        Probe positions in link-local frame.
    probe_local_normal : array-like[float, float, float] or array-like[array-like[float, float, float]]
        Unit direction(s) in link-local frame: one normal for all probes, or one normal per probe matching
        ``probe_local_pos``.
    probe_radius : float | array-like[float]
        Probe sensing radius in meters. A scalar is shared by every probe; an array must match the probe count.
        Array entries of 0 mark inactive filler probes (see the grid note above); at least one must be positive.
    track_link_idx : array-like[int]
        Global rigid link indices whose collision geometry is queried by SDF and whose mesh is sampled for shear.
    n_sample_points: int | array-like[int]
        Total surface samples split across ``track_link_idx``, or one count per tracked link.
    compressibility : float
        In-plane dilation kernel as a mix of a local and a global response, in ``[0, 1]``. ``1`` (default) is fully
        compressible: a concentrated *local* bulge from the ``exp(-lambda_d * r^2)`` Gaussian, with no far-field
        reach. ``0`` is fully incompressible: the in-plane displacement is the gradient of the inverse-Laplacian of
        the indentation depth field (``~ r_hat / r``) -- the *global* volume-conserving stretch that reaches the
        whole sensor, but with a soft center. Intermediate values superimpose a local bulge of weight
        ``compressibility`` on a global stretch of weight ``1 - compressibility`` (each kernel peak-normalized first
        so the weight is meaningful), giving both a sharp local bulge and the global stretch. The endpoints ``0`` and
        ``1`` skip the unused kernel. The normal (out-of-plane) channel keeps the ``lambda_d`` Gaussian bulge for any
        value.
    elastomer_thickness : float
        Gel layer thickness in meters, bottom face bonded to a rigid backing (the Dirichlet condition a flat-slab
        FEM would use). When > 0 (and ``compressibility < 1``), the global in-plane response is that of an
        incompressible elastic layer of this thickness instead of the free-space ``1/r``: indentation features much
        smaller than the thickness produce little in-plane surface motion (incompressible half-space limit),
        wavelengths comparable to the thickness respond most, and the long-range field recovers the ``1/r`` squeeze
        flow. Grid (FFT) layouts use the exact spectral layer solution; non-grid (direct) layouts approximate it by
        regularizing the ``1/r`` kernel at scale ``h`` (``r_hat * r / (r^2 + h^2)``). ``0`` (default) keeps the
        free-space kernel, regularized at the probe spacing (a numerical guard, not a physical scale -- set the
        thickness to control the global response physically).
    lambda_d: float
        Falloff coefficient (in 1/m^2) for the Gaussian dilation kernel ``exp(-lambda_d * r^2)`` (larger = sharper,
        more localized; smaller = broader). It sets the width of the local in-plane bulge (weighted by
        ``compressibility``) and always sets the normal (out-of-plane) bulge width.
    lambda_s: float
        Gaussian falloff coefficient (in 1/m^2) for the shear kernel ``exp(-lambda_s * r^2)`` that spreads each
        anchored tracked-surface point's tangential displacement to nearby probes. Larger values keep shear tightly
        local to the contact patch; smaller values produce a softer, more diffuse shear response.
    dilate_scale: float
        Scalar gain applied to dilation displacement.
    shear_scale: float
        Scalar gain applied to shear displacement.
    normal_exponent: float
        Exponent of the penetration-depth power law for the normal (out-of-plane) marker dilation: the normal
        bulge scales as ``depth ** normal_exponent``. Must be >= 1.0. Default ``2.0`` (the HydroShear quadratic
        normal response). Tangential dilation and shear stay linear in depth regardless of this value.
    contact_threshold / release_threshold : float
        Schmitt-trigger gate for the **shear anchor** state (see ``ContactHysteresisOptionsMixin``): a tracked
        surface point starts anchoring shear when its penetration into the elastomer reaches ``contact_threshold``
        and releases when its depth drops to ``release_threshold`` (default ``-1e-4``: it must *separate* by 0.1 mm).
        Only consumed when ``shear_scale > 0``; backend-agnostic (sdf and raycast depths alike).

    Note
    ----
    Genesis reuses rigid-body SDFs for HydroShear queries. For non-analytic tracked meshes, the collision geometry
    should be watertight enough for signed-distance preprocessing, and the attached elastomer link's collision geometry
    should represent the compliant contact surface.
    """

    # Permits 0-valued (inactive filler) entries; see _validate_filler_probe_radius.
    probe_radius: PositiveFloat | FArrayType | FGridType = 0.01

    lambda_d: NonNegativeFloat = 700.0
    lambda_s: NonNegativeFloat = 300.0
    dilate_scale: NonNegativeFloat = 1.0
    shear_scale: NonNegativeFloat = 1.0
    normal_exponent: NonNegativeFloat = 2.0
    compressibility: NonNegativeFloat = 1.0
    elastomer_thickness: NonNegativeFloat = 0.0

    # Shear-anchor gate defaults (see ContactHysteresisOptionsMixin): anchor at 10 um penetration, release only
    # once separated by 0.1 mm.
    contact_threshold: NonNegativeFloat = 1e-5
    release_threshold: float | None = -1e-4

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        _validate_filler_probe_radius(self.probe_radius, "ElastomerTaxel")
        if len(self.track_link_idx) == 0:
            gs.raise_exception("ElastomerTaxel requires at least one tracked link in track_link_idx.")
        if self.normal_exponent < 1.0:
            gs.raise_exception(f"normal_exponent must be greater than or equal to 1.0. Got {self.normal_exponent}.")
        if self.compressibility > 1.0:
            gs.raise_exception(f"compressibility must be in [0, 1]. Got {self.compressibility}.")


class ProximityTaxel(
    RigidSensorOptionsMixin["ProximityTaxelSensor"],
    SimpleSensorOptions["ProximityTaxelSensor"],
    PointCloudTactileSensorMixin["ProximityTaxelSensor"],
    ProbesWithNormalSensorOptionsMixin["ProximityTaxelSensor"],
    ViscoelasticHysteresisOptionsMixin["ProximityTaxelSensor"],
    SpatialCrosstalkOptionsMixin["ProximityTaxelSensor"],
):
    """
    A tactile sensor which estimates force and torque per taxel from proximity to point clouds sampled on tracked
    meshes within a **spherical** sensing volume of nominal ``probe_radius`` around each taxel.

    For each taxel, every tracked point inside that sphere contributes a penetration depth ``P_i = R_eff - ||p_i - o||``
    where ``R_eff`` is drawn each simulation step when ``probe_radius_noise`` is non-zero (additive uniform noise
    in meters around the sensing radius, clipped nonnegative). Normal force is aligned with ``probe_local_normal``;
    shear uses tangential relative velocity. Generic SimpleSensor imperfections (bias, resolution, etc.) still apply.
    Outputs are in link-local frame.

    Parameters
    ----------
    probe_local_normal : array-like[array-like[float, float, float]]
        Unit direction(s) for the normal force channel in link-local frame: one ``(3,)`` for all taxels, or one row per
        taxel matching ``probe_local_pos``. Default ``(0, 0, 1)``.
    stiffness : float
        Linear spring stiffness (N/m) scaling summed penetration depths into total reported force.
    shear_coupling : float
        Scales penetration-weighted tangential slip ``sum_i P_i * v_{t,i}`` into a shear force contribution (see
        sensor documentation). Set to ``0.0`` to disable shear and use only the normal channel.
    twist_scalar : float
        Scales the relative angular velocity about the contact normal into a torsional (spin) torque
        ``-twist_scalar * dot(relative_angular_velocity, n) * n``. This is the only source of torque about the normal;
        all remaining torque is in-plane tilting moment. Keep it positive to sense twisting slip, or set to ``0.0`` to
        report those tilting moments alone.
    density_scalar : int
        Reference point count for normalizing summed penetrations against tracked cloud size
        (scale is ``density_scalar / max(N_pc, 1)`` for this sensor's tracked samples).

    See ``SpatialCrosstalkOptionsMixin`` for the measured-branch spatial crosstalk parameters
    (``crosstalk_strength`` / ``crosstalk_sigma`` / ``crosstalk_kernel``); a regular planar grid ``probe_local_pos``
    is required to enable it.
    """

    stiffness: NonNegativeFloat = 100.0
    shear_coupling: NonNegativeFloat = 0.0
    twist_scalar: NonNegativeFloat = 1.0
    density_scalar: PositiveInt = 100

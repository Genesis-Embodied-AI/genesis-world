"""Experiment definitions; probabilities come from counted physical outcomes."""

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import chisquare


@dataclass(frozen=True)
class Config:
    kind: str = "coin"
    seed: int = 2026
    dt: float = 0.01
    horizon: float = 6.0
    friction: float = 0.6
    height: float = 0.45
    spin: float = 18.0
    protocol: str = "isotropic"
    forecast_samples: int = 64
    # Gaussian standard deviations defining the observation uncertainty model.
    position_sigma: float = 0.0005
    angle_sigma: float = 0.02
    velocity_sigma: float = 0.03
    spin_sigma: float = 0.3

    def __post_init__(self):
        if self.kind not in ("coin", "die") or self.protocol not in ("isotropic", "face_up"):
            raise ValueError("Choose coin/die and isotropic/face_up.")
        if not 0 <= self.seed < 2**32 or not 8 <= self.forecast_samples <= 256:
            raise ValueError("Seed must be uint32 and forecast samples must be 8..256.")
        values = np.array([self.dt, self.horizon, self.friction, self.height, self.spin,
                           self.position_sigma, self.angle_sigma, self.velocity_sigma, self.spin_sigma])
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("Physical parameters must be finite and nonnegative.")
        if not 0.002 <= self.dt <= 0.02 or not 1 <= self.horizon <= 12:
            raise ValueError("Use dt 0.002..0.02 s and horizon 1..12 s.")
        if not 0.01 <= self.friction <= 2 or not 0.1 <= self.height <= 1.5 or self.spin > 50:
            raise ValueError("Friction, height, or spin is outside the supported range.")

    @property
    def labels(self) -> tuple[str, ...]:
        return ("Heads", "Tails") if self.kind == "coin" else ("1", "2", "3", "4", "5", "6")


@dataclass
class State:
    position: np.ndarray
    quaternion: np.ndarray  # wxyz, local frame to world
    velocity: np.ndarray  # free-joint DOFs: world translation, local angular velocity


@dataclass
class Trial:
    seed: int
    outcome: int  # zero-based face; -1 means unresolved (edge or timeout)
    seconds: float
    initial_position: list[float]
    initial_quaternion: list[float]
    initial_velocity: list[float]
    final_position: list[float]
    final_quaternion: list[float]
    final_velocity: list[float]
    source: str = "batch"


@dataclass
class Forecast:
    at_time: float
    trial_seed: int
    horizon_remaining: float
    seed: int
    counts: list[int]
    unresolved: int
    samples: int
    probabilities: list[float]
    observed_position: list[float]
    observed_quaternion: list[float]
    observed_velocity: list[float]


@dataclass
class Results:
    config: Config
    trials: list[Trial] = field(default_factory=list)

    @property
    def counts(self) -> np.ndarray:
        outcomes = np.array([trial.outcome for trial in self.trials], dtype=int)
        return np.bincount(outcomes[outcomes >= 0], minlength=len(self.config.labels))

    @property
    def unresolved(self) -> int:
        return sum(trial.outcome < 0 for trial in self.trials)

    @property
    def p_value(self) -> float | None:
        counts = self.counts
        # The asymptotic chi-square approximation needs sufficient expected counts.
        return float(chisquare(counts).pvalue) if counts.sum() >= 5 * len(counts) else None


def sample_launch(config: Config, seeds: list[int]) -> State:
    """One RNG per trial makes launch states independent of batching/chunk size."""
    poses, quats, velocities = [], [], []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        quat = Rotation.random(random_state=rng).as_quat(scalar_first=True)
        if config.protocol == "face_up":
            quat = np.array([1.0, 0.0, 0.0, 0.0])
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        poses.append([0.0, 0.0, config.height])
        quats.append(quat)
        velocities.append(np.r_[rng.normal(0, 0.06, 2), 0.8, axis * config.spin])
    return State(np.asarray(poses), np.asarray(quats), np.asarray(velocities))


def perturb_observation(config: Config, observed: State, seed: int) -> State:
    """Independent Gaussian sensor-error scenarios, not a fitted Bayesian posterior."""
    rng = np.random.default_rng(seed)
    n = config.forecast_samples
    position = observed.position[0] + rng.normal(0, config.position_sigma, (n, 3))
    perturbation = Rotation.from_rotvec(rng.normal(0, config.angle_sigma, (n, 3)))
    quaternion = (Rotation.from_quat(observed.quaternion[0], scalar_first=True) * perturbation).as_quat(
        scalar_first=True
    )
    sigma = np.array([config.velocity_sigma] * 3 + [config.spin_sigma] * 3)
    velocity = observed.velocity[0] + rng.normal(size=(n, 6)) * sigma
    return State(position, quaternion, velocity)


def classify(quaternions: np.ndarray, kind: str, alignment: float = 0.985) -> np.ndarray:
    """Identify upward face only; the caller must independently require rest."""
    if kind not in ("coin", "die"):
        raise ValueError("Unknown object kind.")
    rotation = Rotation.from_quat(np.atleast_2d(quaternions), scalar_first=True).as_matrix()
    normals = np.array([[0, 0, 1], [0, 0, -1]]) if kind == "coin" else np.array(
        [[0, 0, 1], [1, 0, 0], [0, 1, 0], [0, -1, 0], [-1, 0, 0], [0, 0, -1]]
    )
    upward = rotation[:, 2, :] @ normals.T
    faces = np.argmax(upward, axis=1)
    return np.where(upward.max(axis=1) >= alignment, faces, -1)


def wilson_interval(count: int, total: int) -> tuple[float, float]:
    """Marginal 95% Wilson interval; exploratory, not simultaneous/sequential coverage."""
    if not 0 <= count <= total:
        raise ValueError("Count must be between zero and total.")
    if total == 0:
        return 0.0, 1.0
    z2 = 1.959963984540054**2
    p = count / total
    center = (p + z2 / (2 * total)) / (1 + z2 / total)
    radius = np.sqrt(z2 * (p * (1 - p) / total + z2 / (4 * total**2))) / (1 + z2 / total)
    return max(0.0, center - radius), min(1.0, center + radius)

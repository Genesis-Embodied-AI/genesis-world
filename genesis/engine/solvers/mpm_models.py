"""Constitutive models of the material point method (MPM) solver, bound from the material and the solver options."""

import math
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, ClassVar

import quadrants as qd

import genesis as gs
from genesis.options.materials import Elastic, ElastoPlastic, Liquid, Muscle, Sand, Snow
from genesis.utils.element import lame_parameters

if TYPE_CHECKING:
    from genesis.options.entities import MPMEntityOptions


@qd.data_oriented
class MPMModel:
    """The elastic model shared by every MPM material, under the Lame parameters of the material.

    'needs_svd' states whether the model reads the singular value decomposition of the deformation gradient, which the
    solver skips when no model of the scene reads it. 'default_Jp' is the plastic volume ratio a particle starts from.
    """

    def __init__(self, youngs_modulus: float, poisson_ratio: float) -> None:
        self.mu, self.lam = lame_parameters(youngs_modulus, poisson_ratio)
        self.needs_svd = True
        self.default_Jp = 1.0

    @qd.func
    def update_F_S_Jp(self, J, F_tmp, U, S, V, Jp):
        raise NotImplementedError

    @qd.func
    def update_stress(self, U, S, V, F_tmp, F_new, J, Jp, actu, m_dir):
        stress = 2 * self.mu * (F_new - U @ V.transpose()) @ F_new.transpose() + qd.Matrix.identity(
            gs.qd_float, 3
        ) * self.lam * J * (J - 1)
        return stress


@qd.data_oriented
class MPMElasticModel(MPMModel):
    """The stress of an Elastic material, under the stress model its entity solver options choose."""

    def __init__(self, youngs_modulus: float, poisson_ratio: float, model: str) -> None:
        super().__init__(youngs_modulus, poisson_ratio)
        self.update_F_S_Jp = self._update_F_S_Jp_elastic
        # The 'corotation' model keeps the stress of the base model, which reads U @ V.T from the decomposition of F_tmp
        if model == "neohooken":
            self.update_stress = self._update_stress_neohooken
            # The Neo-Hookean stress only reads F_tmp and J = det(F_tmp)
            self.needs_svd = False

    @qd.func
    def _update_F_S_Jp_elastic(self, J, F_tmp, U, S, V, Jp):
        F_new = F_tmp
        S_new = S
        Jp_new = Jp
        return F_new, S_new, Jp_new

    @qd.func
    def _update_stress_neohooken(self, U, S, V, F_tmp, F_new, J, Jp, actu, m_dir):
        stress = self.mu * (F_tmp @ F_tmp.transpose()) + qd.Matrix.identity(gs.qd_float, 3) * (
            self.lam * qd.log(J) - self.mu
        )
        return stress


@qd.data_oriented
class MPMMuscleModel(MPMElasticModel):
    """The model of a Muscle material: its elastic model plus an active stress along the fiber direction.

    The active stress scales with the Young's modulus of the material.
    """

    def __init__(self, youngs_modulus: float, poisson_ratio: float, model: str) -> None:
        super().__init__(youngs_modulus, poisson_ratio, model)
        self.stiffness = youngs_modulus
        self._update_stress_without_actuation = self.update_stress
        self.update_stress = self._update_stress_with_actuation

    @qd.func
    def _update_stress_with_actuation(self, U, S, V, F_tmp, F_new, J, Jp, actu, m_dir):
        stress = self._update_stress_without_actuation(U, S, V, F_tmp, F_new, J, Jp, actu, m_dir)

        AAt = m_dir.outer_product(m_dir)
        stress += self.stiffness * actu * F_tmp @ AAt @ F_tmp.transpose()

        return stress


@qd.data_oriented
class MPMElastoPlasticModel(MPMModel):
    """The model of an ElastoPlastic material, yielding under its yield criterion."""

    def __init__(
        self,
        youngs_modulus: float,
        poisson_ratio: float,
        yield_criterion: str,
        yield_stress: float,
        critical_compression: float,
        critical_stretch: float,
    ) -> None:
        super().__init__(youngs_modulus, poisson_ratio)
        self.use_von_mises = yield_criterion == "von_mises"
        self.yield_stress = yield_stress
        self.critical_compression = critical_compression
        self.critical_stretch = critical_stretch

    @qd.func
    def update_F_S_Jp(self, J, F_tmp, U, S, V, Jp):
        F_new = qd.Matrix.zero(gs.qd_float, 3, 3)
        S_new = qd.Matrix.zero(gs.qd_float, 3, 3)
        if qd.static(self.use_von_mises):
            # The lower bound on the singular values prevents NaN in the logarithm
            S_new = qd.max(S, 0.05)
            epsilon = qd.Vector([qd.log(S_new[0, 0]), qd.log(S_new[1, 1]), qd.log(S_new[2, 2])])
            epsilon_hat = epsilon - (epsilon.sum() / 3)
            epsilon_hat_norm = epsilon_hat.norm(gs.EPS)
            delta_gamma = epsilon_hat_norm - self.yield_stress / (2 * self.mu)

            if delta_gamma > 0:
                epsilon -= (delta_gamma / epsilon_hat_norm) * epsilon_hat
                S_new = qd.Matrix.zero(gs.qd_float, 3, 3)
                for d in qd.static(range(3)):
                    S_new[d, d] = qd.exp(epsilon[d])
                F_new = U @ S_new @ V.transpose()
            else:
                F_new = F_tmp

        else:
            S_new = qd.Matrix.zero(gs.qd_float, 3, 3)
            for d in qd.static(range(3)):
                S_new[d, d] = min(max(S[d, d], 1 - self.critical_compression), 1 + self.critical_stretch)
            F_new = U @ S_new @ V.transpose()

        Jp_new = Jp
        return F_new, S_new, Jp_new


@qd.data_oriented
class MPMLiquidModel(MPMModel):
    """The model of a Liquid material, whose deformation resets to a pure volume change at every substep.

    The bulk modulus is the first Lame parameter, the shear modulus being zero. A viscous liquid takes the shear modulus
    viscosity / substep_dt, applied to the deformation of one substep. It resists the strain rate as a viscosity would,
    and its volumetric part also stiffens the liquid against compression, more so at a finer substep.
    """

    def __init__(
        self,
        youngs_modulus: float,
        poisson_ratio: float,
        bulk_modulus: float | None,
        viscosity: float | None,
        substep_dt: float,
    ) -> None:
        # The bulk modulus defaults to the first Lame parameter of the Young's modulus and the Poisson ratio
        super().__init__(youngs_modulus, poisson_ratio)
        if bulk_modulus is not None:
            self.lam = bulk_modulus
        self.mu = 0.0 if viscosity is None else viscosity / substep_dt
        # The term 2 * mu * (F_tmp - U @ V.T) @ F_tmp.T is the only reader of the decomposition, and vanishes with mu
        self.needs_svd = self.mu > 0.0

    @qd.func
    def update_F_S_Jp(self, J, F_tmp, U, S, V, Jp):
        F_new = qd.Matrix.identity(gs.qd_float, 3) * qd.pow(J, 1.0 / 3.0)
        S_new = S
        Jp_new = Jp
        return F_new, S_new, Jp_new

    @qd.func
    def update_stress(self, U, S, V, F_tmp, F_new, J, Jp, actu, m_dir):
        stress = 2 * self.mu * (F_tmp - U @ V.transpose()) @ F_tmp.transpose() + qd.Matrix.identity(
            gs.qd_float, 3
        ) * self.lam * J * (J - 1)
        return stress


@qd.data_oriented
class MPMSandModel(MPMModel):
    """The model of a Sand material, yielding under the Drucker-Prager criterion of its internal friction angle."""

    def __init__(self, youngs_modulus: float, poisson_ratio: float, friction_angle: float) -> None:
        super().__init__(youngs_modulus, poisson_ratio)
        self.default_Jp = 0.0
        sin_phi = math.sin(math.radians(friction_angle))
        self.alpha = math.sqrt(2 / 3) * 2 * sin_phi / (3 - sin_phi)

    @qd.func
    def _sand_projection(self, S, Jp):
        S_out = qd.Matrix.zero(gs.qd_float, 3, 3)
        epsilon = qd.Vector.zero(gs.qd_float, 3)
        for i in qd.static(range(3)):
            epsilon[i] = qd.log(max(abs(S[i, i]), 1e-4))
            S_out[i, i] = 1
        tr = epsilon.sum() + Jp
        epsilon_hat = epsilon - tr / 3
        epsilon_hat_norm = epsilon_hat.norm(gs.EPS)

        Jp_new = gs.qd_float(0.0)
        if tr >= 0.0:
            Jp_new = tr
        else:
            Jp_new = 0.0
            delta_gamma = epsilon_hat_norm + (3 * self.lam + 2 * self.mu) / (2 * self.mu) * tr * self.alpha
            for i in qd.static(range(3)):
                S_out[i, i] = qd.exp(epsilon[i] - max(0, delta_gamma) / epsilon_hat_norm * epsilon_hat[i])

        return S_out, Jp_new

    @qd.func
    def update_F_S_Jp(self, J, F_tmp, U, S, V, Jp):
        S_new, Jp_new = self._sand_projection(S, Jp)
        F_new = U @ S_new @ V.transpose()
        return F_new, S_new, Jp_new

    @qd.func
    def update_stress(self, U, S, V, F_tmp, F_new, J, Jp, actu, m_dir):
        log_S_sum = gs.qd_float(0.0)
        center = qd.Matrix.zero(gs.qd_float, 3, 3)
        for i in qd.static(range(3)):
            log_S_sum += qd.log(S[i, i])
            center[i, i] = 2.0 * self.mu * qd.log(S[i, i]) * (1 / S[i, i])
        for i in qd.static(range(3)):
            center[i, i] += self.lam * log_S_sum * (1 / S[i, i])
        stress = U @ center @ V.transpose() @ F_new.transpose()
        return stress


@qd.data_oriented
class MPMSnowModel(MPMModel):
    """The model of a Snow material, clamping its principal stretches and hardening as it compacts."""

    def __init__(
        self, youngs_modulus: float, poisson_ratio: float, critical_compression: float, critical_stretch: float
    ) -> None:
        super().__init__(youngs_modulus, poisson_ratio)
        self.critical_compression = critical_compression
        self.critical_stretch = critical_stretch

    @qd.func
    def update_F_S_Jp(self, J, F_tmp, U, S, V, Jp):
        S_new = qd.Matrix.zero(gs.qd_float, 3, 3)
        Jp_new = Jp
        for d in qd.static(range(3)):
            S_new[d, d] = min(max(S[d, d], 1 - self.critical_compression), 1 + self.critical_stretch)
            Jp_new *= S[d, d] / S_new[d, d]
        F_new = U @ S_new @ V.transpose()
        return F_new, S_new, Jp_new

    @qd.func
    def update_stress(self, U, S, V, F_tmp, F_new, J, Jp, actu, m_dir):
        # The hardening coefficient stiffens the snow as it compacts
        h = qd.exp(10 * (1.0 - Jp))
        mu, lam = self.mu * h, self.lam * h

        r = U @ V.transpose()
        stress = 2 * mu * (F_new - r) @ F_new.transpose() + qd.Matrix.identity(gs.qd_float, 3) * lam * J * (J - 1)

        return stress


@dataclass(frozen=True)
class MPMModelKey:
    """The parameters of the model of an entity, equal for the entities sharing one compiled model.

    Each subclass holds the constructor parameters of its model class, which keys of another subclass never equal.
    """

    model_cls: ClassVar[type[MPMModel]]
    youngs_modulus: float
    poisson_ratio: float


@dataclass(frozen=True)
class MPMElasticModelKey(MPMModelKey):
    model_cls: ClassVar[type[MPMModel]] = MPMElasticModel
    model: str


@dataclass(frozen=True)
class MPMMuscleModelKey(MPMModelKey):
    model_cls: ClassVar[type[MPMModel]] = MPMMuscleModel
    model: str


@dataclass(frozen=True)
class MPMElastoPlasticModelKey(MPMModelKey):
    model_cls: ClassVar[type[MPMModel]] = MPMElastoPlasticModel
    yield_criterion: str
    yield_stress: float
    critical_compression: float
    critical_stretch: float


@dataclass(frozen=True)
class MPMLiquidModelKey(MPMModelKey):
    model_cls: ClassVar[type[MPMModel]] = MPMLiquidModel
    bulk_modulus: float | None
    viscosity: float | None


@dataclass(frozen=True)
class MPMSandModelKey(MPMModelKey):
    model_cls: ClassVar[type[MPMModel]] = MPMSandModel
    friction_angle: float


@dataclass(frozen=True)
class MPMSnowModelKey(MPMModelKey):
    model_cls: ClassVar[type[MPMModel]] = MPMSnowModel
    critical_compression: float
    critical_stretch: float


def get_mpm_model_key(options: "MPMEntityOptions") -> MPMModelKey:
    """Return what decides the model of an entity, so that the entities sharing it share one compiled model.

    The key holds the parameters of the model of the material, the defaults of the solver filled in (see
    'MPMEntityOptions.material_params').
    """
    material, params = options.material, options.material_params
    youngs_modulus, poisson_ratio = params.youngs_modulus, params.poisson_ratio
    if isinstance(material, Elastic):
        return MPMElasticModelKey(youngs_modulus, poisson_ratio, params.model)
    if isinstance(material, Muscle):
        return MPMMuscleModelKey(youngs_modulus, poisson_ratio, params.model)
    if isinstance(material, ElastoPlastic):
        return MPMElastoPlasticModelKey(
            youngs_modulus,
            poisson_ratio,
            material.yield_criterion,
            material.yield_stress,
            material.critical_compression,
            material.critical_stretch,
        )
    if isinstance(material, Liquid):
        return MPMLiquidModelKey(youngs_modulus, poisson_ratio, material.bulk_modulus, material.viscosity)
    if isinstance(material, Sand):
        return MPMSandModelKey(youngs_modulus, poisson_ratio, material.friction_angle)
    return MPMSnowModelKey(youngs_modulus, poisson_ratio, material.critical_compression, material.critical_stretch)


def build_mpm_model(key: MPMModelKey, substep_dt: float) -> MPMModel:
    """Bind the model that a key returned by 'get_mpm_model_key' names.

    A viscous Liquid needs the substep of the solver (see 'MPMLiquidModel').
    """
    if isinstance(key, MPMLiquidModelKey):
        return MPMLiquidModel(**asdict(key), substep_dt=substep_dt)
    return key.model_cls(**asdict(key))

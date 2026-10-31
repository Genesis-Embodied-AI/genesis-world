"""Constitutive models of the finite element method (FEM) solver, bound from the material and the solver options."""

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, ClassVar

import quadrants as qd

import genesis as gs
from genesis.options.materials import Cloth, Elastic, Muscle

if TYPE_CHECKING:
    from genesis.options.entities import FEMEntityOptions


@qd.data_oriented
class FEMModel:
    """The constitutive model of a material without volumetric response, whose physics the IPC coupler owns.

    The explicit solver compiles the stress of every model into its static loop over models, even one no element
    executes, so the stress of this model returns zero rather than raise, which quadrants cannot compile.
    """

    def __init__(self) -> None:
        self.hessian_invariant = False
        self.is_hessian_ready = False

    def build(self, fem_solver) -> None:
        pass

    @qd.func
    def pre_compute(self, J, F, i_e, i_b):
        pass

    @qd.func
    def update_stress(self, mu, lam, J, F, actu, m_dir):
        return qd.Matrix.zero(gs.qd_float, 3, 3)

    @qd.func
    def compute_energy_gradient_hessian(self, mu, lam, J, F, actu, m_dir, i_e, i_b, hessian_field):
        raise NotImplementedError

    @qd.func
    def compute_energy_gradient(self, mu, lam, J, F, actu, m_dir, i_e, i_b):
        raise NotImplementedError

    @qd.func
    def compute_energy(self, mu, lam, J, F, actu, m_dir, i_e, i_b):
        raise NotImplementedError


@qd.data_oriented
class FEMElasticModel(FEMModel):
    """The stress and energy of an Elastic material, under the constitutive model its entity solver options choose.

    The Hessian of the elastic energy is constant under the linear model alone, which lets the implicit solver compute
    it once and reuse it at every step.
    """

    def __init__(self, model: str) -> None:
        super().__init__()
        self.hessian_invariant = model == "linear"
        # The rotation of each element under the linear corotated model, allocated once the element count is known
        self._R = None
        if model == "linear":
            self.update_stress = self._update_stress_linear
            self.compute_energy_gradient_hessian = self._compute_energy_gradient_hessian_linear
            self.compute_energy_gradient = self._compute_energy_gradient_linear
            self.compute_energy = self._compute_energy_linear
        elif model == "stable_neohookean":
            self.update_stress = self._update_stress_stable_neohookean
            self.compute_energy_gradient_hessian = self._compute_energy_gradient_hessian_stable_neohookean
            self.compute_energy_gradient = self._compute_energy_gradient_stable_neohookean
            self.compute_energy = self._compute_energy_stable_neohookean
        elif model == "linear_corotated":
            self.build = self._build_linear_corotated
            self.pre_compute = self._pre_compute_linear_corotated
            self.update_stress = self._update_stress_linear_corotated
            self.compute_energy_gradient_hessian = self._compute_energy_gradient_hessian_linear_corotated
            self.compute_energy_gradient = self._compute_energy_gradient_linear_corotated
            self.compute_energy = self._compute_energy_linear_corotated

    def _build_linear_corotated(self, fem_solver) -> None:
        self._R = qd.field(dtype=gs.qd_mat3, shape=(fem_solver._B, fem_solver.n_elements))

    @qd.func
    def _pre_compute_linear_corotated(self, J, F, i_e, i_b):
        U, S, V = qd.svd(F)
        R = U @ V.transpose()
        self._R[i_b, i_e] = R

    # --- Linear model ---

    @qd.func
    def _update_stress_linear(self, mu, lam, J, F, actu, m_dir):
        I = qd.Matrix.identity(dt=gs.qd_float, n=3)
        stress = mu * (F + F.transpose() - 2 * I) + lam * (F - I).trace() * I
        return stress

    @qd.func
    def _compute_energy_gradient_hessian_linear(self, mu, lam, J, F, actu, m_dir, i_e, i_b, hessian_field):
        I = qd.Matrix.identity(dt=gs.qd_float, n=3)
        eps = 0.5 * (F + F.transpose()) - I
        trEps = eps.trace()
        energy = mu * eps.norm_sqr() + 0.5 * lam * trEps**2
        gradient = 2.0 * mu * eps + lam * trEps * I

        for i in qd.static(qd.grouped(qd.ndrange(3, 3))):
            hessian_field[i_b, i, i_e].fill(0.0)

        for i, k in qd.static(qd.ndrange(3, 3)):
            hessian_field[i_b, i, i, i_e][k, k] = mu

        hessian_field[i_b, 0, 0, i_e][0, 0] += mu + lam
        hessian_field[i_b, 1, 1, i_e][1, 1] += mu + lam
        hessian_field[i_b, 2, 2, i_e][2, 2] += mu + lam

        hessian_field[i_b, 0, 1, i_e][1, 0] = hessian_field[i_b, 1, 0, i_e][0, 1] = mu
        hessian_field[i_b, 0, 2, i_e][2, 0] = hessian_field[i_b, 2, 0, i_e][0, 2] = mu
        hessian_field[i_b, 1, 2, i_e][2, 1] = hessian_field[i_b, 2, 1, i_e][1, 2] = mu

        hessian_field[i_b, 0, 1, i_e][0, 1] = hessian_field[i_b, 0, 2, i_e][0, 2] = lam
        hessian_field[i_b, 1, 0, i_e][1, 0] = hessian_field[i_b, 2, 0, i_e][2, 0] = lam
        hessian_field[i_b, 1, 2, i_e][1, 2] = hessian_field[i_b, 2, 1, i_e][2, 1] = lam
        return energy, gradient

    @qd.func
    def _compute_energy_gradient_linear(self, mu, lam, J, F, actu, m_dir, i_e, i_b):
        I = qd.Matrix.identity(dt=gs.qd_float, n=3)
        eps = 0.5 * (F + F.transpose()) - I
        trEps = eps.trace()
        energy = mu * eps.norm_sqr() + 0.5 * lam * trEps**2
        gradient = 2.0 * mu * eps + lam * trEps * I
        return energy, gradient

    @qd.func
    def _compute_energy_linear(self, mu, lam, J, F, actu, m_dir, i_e, i_b):
        I = qd.Matrix.identity(dt=gs.qd_float, n=3)
        eps = 0.5 * (F + F.transpose()) - I
        trEps = eps.trace()
        energy = mu * eps.norm_sqr() + 0.5 * lam * trEps**2
        return energy

    # --- Stable Neo-Hookean model ---

    @qd.func
    def _update_stress_stable_neohookean(self, mu, lam, J, F, actu, m_dir):
        IC = F.norm_sqr()
        dJdF0 = F[:, 1].cross(F[:, 2])
        dJdF1 = F[:, 2].cross(F[:, 0])
        dJdF2 = F[:, 0].cross(F[:, 1])
        dJdF = qd.Matrix.cols([dJdF0, dJdF1, dJdF2])
        alpha = 1 + 0.75 * mu / lam
        stress = mu * (1 - 1 / (IC + 1)) * F + lam * (J - alpha) * dJdF
        return stress

    @qd.func
    def _compute_energy_gradient_hessian_stable_neohookean(self, mu, lam, J, F, actu, m_dir, i_e, i_b, hessian_field):
        raise NotImplementedError("Hessian computation is not implemented for stable_neohookean model.")

    @qd.func
    def _compute_energy_gradient_stable_neohookean(self, mu, lam, J, F, actu, m_dir, i_e, i_b):
        gs.raise_exception("Gradient computation is not implemented for stable_neohookean model.")

    @qd.func
    def _compute_energy_stable_neohookean(self, mu, lam, J, F, actu, m_dir, i_e, i_b):
        _lambda = lam + mu
        _alpha = 1.0 + mu / _lambda
        Ic = F.norm_sqr()
        Jminus1 = J - _alpha
        energy = 0.5 * (mu * (Ic - 3.0) + _lambda * Jminus1**2)
        return energy

    # --- Linear corotated model ---

    @qd.func
    def _update_stress_linear_corotated(self, mu, lam, J, F, actu, m_dir):
        gs.raise_exception("Linear corotated stress update is not implemented yet.")

    @qd.func
    def _compute_energy_gradient_hessian_linear_corotated(self, mu, lam, J, F, actu, m_dir, i_e, i_b, hessian_field):
        R = self._R[i_b, i_e]
        F_hat = R.transpose() @ F
        eps = 0.5 * (F_hat + F_hat.transpose())
        for i in qd.static(range(3)):
            eps[i, i] -= 1.0
        trEps = eps.trace()
        energy = mu * eps.norm_sqr() + 0.5 * lam * trEps**2
        gradient = 2.0 * mu * R @ eps + lam * trEps * R

        for i in qd.static(qd.grouped(qd.ndrange(3, 3))):
            hessian_field[i_b, i, i_e].fill(0.0)

        for i, k in qd.static(qd.ndrange(3, 3)):
            hessian_field[i_b, i, i, i_e][k, k] = mu

        for i, j, alpha, beta in qd.ndrange(3, 3, 3, 3):
            hessian_field[i_b, j, beta, i_e][i, alpha] += mu * R[i, beta] * R[alpha, j] + lam * R[alpha, beta] * R[i, j]

        return energy, gradient

    @qd.func
    def _compute_energy_gradient_linear_corotated(self, mu, lam, J, F, actu, m_dir, i_e, i_b):
        F_hat = self._R[i_b, i_e].transpose() @ F
        eps = 0.5 * (F_hat + F_hat.transpose())
        for i in qd.static(range(3)):
            eps[i, i] -= 1.0
        trEps = eps.trace()
        energy = mu * eps.norm_sqr() + 0.5 * lam * trEps**2
        gradient = 2.0 * mu * self._R[i_b, i_e] @ eps + lam * trEps * self._R[i_b, i_e]
        return energy, gradient

    @qd.func
    def _compute_energy_linear_corotated(self, mu, lam, J, F, actu, m_dir, i_e, i_b):
        F_hat = self._R[i_b, i_e].transpose() @ F
        eps = 0.5 * (F_hat + F_hat.transpose())
        for i in qd.static(range(3)):
            eps[i, i] -= 1.0
        trEps = eps.trace()
        energy = mu * eps.norm_sqr() + 0.5 * lam * trEps**2
        return energy


@qd.data_oriented
class FEMMuscleModel(FEMElasticModel):
    """The model of a Muscle material: its elastic model plus an active stress along the fiber direction.

    The active stress scales with the Young's modulus of the material.
    """

    def __init__(self, model: str, stiffness: float) -> None:
        super().__init__(model)
        self.stiffness = stiffness
        self._update_stress_without_actuation = self.update_stress
        self.update_stress = self._update_stress_with_actuation

    @qd.func
    def _update_stress_with_actuation(self, mu, lam, J, F, actu, m_dir):
        stress = self._update_stress_without_actuation(mu, lam, J, F, actu, m_dir)

        l = (F @ m_dir).norm(1e-12)
        mmT = m_dir.outer_product(m_dir)
        stress += self.stiffness * (actu / l) * F @ mmT

        return stress


@dataclass(frozen=True)
class FEMModelKey:
    """The parameters of the model of an entity, equal for the entities sharing one compiled model.

    Each subclass holds the constructor parameters of its model class, which keys of another subclass never equal.
    """

    model_cls: ClassVar[type[FEMModel]]


@dataclass(frozen=True)
class FEMClothModelKey(FEMModelKey):
    model_cls: ClassVar[type[FEMModel]] = FEMModel


@dataclass(frozen=True)
class FEMElasticModelKey(FEMModelKey):
    model_cls: ClassVar[type[FEMModel]] = FEMElasticModel
    model: str


@dataclass(frozen=True)
class FEMMuscleModelKey(FEMModelKey):
    model_cls: ClassVar[type[FEMModel]] = FEMMuscleModel
    model: str
    stiffness: float


def get_fem_model_key(options: "FEMEntityOptions") -> FEMModelKey:
    """Return what decides the model of an entity, so that the entities sharing it share one compiled model."""
    material, model = options.material, options.solver_options.model
    if isinstance(material, Cloth):
        return FEMClothModelKey()
    if isinstance(material, Muscle):
        return FEMMuscleModelKey(model, material.youngs_modulus)
    return FEMElasticModelKey(model)


def build_fem_model(key: FEMModelKey) -> FEMModel:
    """Bind the model that a key returned by 'get_fem_model_key' names."""
    return key.model_cls(**asdict(key))

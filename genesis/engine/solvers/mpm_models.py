"""Constitutive models of the material point method (MPM) solver, bound from the material and the solver options."""

import quadrants as qd

import genesis as gs
from genesis.options.materials import Elastic, ElastoPlastic, Liquid, Muscle, Sand, Snow


@qd.data_oriented
class MPMModel:
    """The elastic model shared by every MPM material, under the parameters each particle holds.

    'needs_svd' states whether the model reads the singular value decomposition of the deformation gradient, which the
    solver skips when no model of the scene reads it. 'default_Jp' is the plastic volume ratio a particle starts from.
    Each particle holds the Lame parameters and the yield parameters of its material, the latter zero for a law that
    does not yield (see 'MPMEntity._add_particles_to_solver').

    Parameters
    ----------
    model : str | None
        The variant of the law, compiled in: the stress model of an Elastic or a Muscle material, and the yield
        criterion of an ElastoPlastic one. None for the other materials.
    """

    def __init__(self, model: str | None) -> None:
        self.needs_svd = True
        self.default_Jp = 1.0

    @qd.func
    def update_F_S_Jp(
        self, J, F_tmp, U, S, V, Jp, mu, lam, yield_stress, critical_compression, critical_stretch, friction_alpha
    ):
        raise NotImplementedError

    @qd.func
    def update_stress(self, U, S, V, F_tmp, F_new, J, Jp, actu, m_dir, mu, lam):
        return self._compute_corotated_stress(U, V, F_new, J, mu, lam)

    @qd.func
    def _compute_corotated_stress(self, U, V, F, J, mu, lam):
        return 2 * mu * (F - U @ V.transpose()) @ F.transpose() + qd.Matrix.identity(gs.qd_float, 3) * lam * J * (J - 1)


@qd.data_oriented
class MPMElasticModel(MPMModel):
    """The stress of an Elastic material, under the stress model its entity solver options choose."""

    def __init__(self, model: str | None) -> None:
        super().__init__(model)
        # The 'corotation' model keeps the stress of the base model, which reads U @ V.T from the decomposition of F_tmp
        if model == gs.mpm_model.neohookean:
            self.update_stress = self._update_stress_neohookean
            # The Neo-Hookean stress only reads F_tmp and J = det(F_tmp)
            self.needs_svd = False

    @qd.func
    def update_F_S_Jp(
        self, J, F_tmp, U, S, V, Jp, mu, lam, yield_stress, critical_compression, critical_stretch, friction_alpha
    ):
        return F_tmp, S, Jp

    @qd.func
    def _update_stress_neohookean(self, U, S, V, F_tmp, F_new, J, Jp, actu, m_dir, mu, lam):
        stress = mu * (F_tmp @ F_tmp.transpose()) + qd.Matrix.identity(gs.qd_float, 3) * (lam * qd.log(J) - mu)
        return stress


@qd.data_oriented
class MPMMuscleModel(MPMElasticModel):
    """The model of a Muscle material: its elastic model plus an active stress along the fiber direction.

    The active stress scales with the Young's modulus of the material, recovered from the Lame parameters.
    """

    def __init__(self, model: str | None) -> None:
        super().__init__(model)
        self._update_stress_without_actuation = self.update_stress
        self.update_stress = self._update_stress_with_actuation

    @qd.func
    def _update_stress_with_actuation(self, U, S, V, F_tmp, F_new, J, Jp, actu, m_dir, mu, lam):
        stress = self._update_stress_without_actuation(U, S, V, F_tmp, F_new, J, Jp, actu, m_dir, mu, lam)

        youngs_modulus = mu * (3 * lam + 2 * mu) / (lam + mu)
        AAt = m_dir.outer_product(m_dir)
        stress += youngs_modulus * actu * F_tmp @ AAt @ F_tmp.transpose()

        return stress


@qd.data_oriented
class MPMElastoPlasticModel(MPMModel):
    """The model of an ElastoPlastic material, yielding under its yield criterion."""

    def __init__(self, model: str | None) -> None:
        super().__init__(model)
        self.use_von_mises = model == gs.yield_criterion.von_mises

    @qd.func
    def update_F_S_Jp(
        self, J, F_tmp, U, S, V, Jp, mu, lam, yield_stress, critical_compression, critical_stretch, friction_alpha
    ):
        F_new = qd.Matrix.zero(gs.qd_float, 3, 3)
        S_new = qd.Matrix.zero(gs.qd_float, 3, 3)
        if qd.static(self.use_von_mises):
            # The lower bound on the singular values prevents NaN in the logarithm
            S_new = qd.max(S, 0.05)
            epsilon = qd.Vector([qd.log(S_new[0, 0]), qd.log(S_new[1, 1]), qd.log(S_new[2, 2])])
            epsilon_hat = epsilon - (epsilon.sum() / 3)
            epsilon_hat_norm = epsilon_hat.norm(gs.EPS)
            delta_gamma = epsilon_hat_norm - yield_stress / (2 * mu)

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
                S_new[d, d] = min(max(S[d, d], 1 - critical_compression), 1 + critical_stretch)
            F_new = U @ S_new @ V.transpose()

        Jp_new = Jp
        return F_new, S_new, Jp_new


@qd.data_oriented
class MPMLiquidModel(MPMModel):
    """The model of a Liquid material, whose deformation resets to a pure volume change at every substep.

    The bulk modulus is the first Lame parameter, the shear modulus being zero. A viscous liquid takes the shear modulus
    viscosity / substep_dt, applied to the deformation of one substep. It resists the strain rate as a viscosity would,
    and its volumetric part also stiffens the liquid against compression, more so at a finer substep. An inviscid
    liquid leaves the decomposition of the deformation gradient unread, and a viscous one reads it (see
    'MPMSolver.build'). The particles of a liquid hold these moduli (see 'MPMEntity._add_particles_to_solver').
    """

    def __init__(self, model: str | None) -> None:
        super().__init__(model)
        self.needs_svd = False

    @qd.func
    def update_F_S_Jp(
        self, J, F_tmp, U, S, V, Jp, mu, lam, yield_stress, critical_compression, critical_stretch, friction_alpha
    ):
        F_new = qd.Matrix.identity(gs.qd_float, 3) * qd.pow(J, 1.0 / 3.0)
        S_new = S
        Jp_new = Jp
        return F_new, S_new, Jp_new

    @qd.func
    def update_stress(self, U, S, V, F_tmp, F_new, J, Jp, actu, m_dir, mu, lam):
        # The viscosity resists the deformation of the substep, which F_tmp holds before its reset to a volume change
        return self._compute_corotated_stress(U, V, F_tmp, J, mu, lam)


@qd.data_oriented
class MPMSandModel(MPMModel):
    """The model of a Sand material, yielding under the Drucker-Prager criterion of its internal friction angle."""

    def __init__(self, model: str | None) -> None:
        super().__init__(model)
        self.default_Jp = 0.0

    @qd.func
    def update_F_S_Jp(
        self, J, F_tmp, U, S, V, Jp, mu, lam, yield_stress, critical_compression, critical_stretch, friction_alpha
    ):
        S_new = qd.Matrix.zero(gs.qd_float, 3, 3)
        epsilon = qd.Vector.zero(gs.qd_float, 3)
        for i in qd.static(range(3)):
            epsilon[i] = qd.log(max(abs(S[i, i]), 1e-4))
            S_new[i, i] = 1
        tr = epsilon.sum() + Jp
        epsilon_hat = epsilon - tr / 3
        epsilon_hat_norm = epsilon_hat.norm(gs.EPS)

        Jp_new = gs.qd_float(0.0)
        if tr >= 0.0:
            Jp_new = tr
        else:
            Jp_new = 0.0
            delta_gamma = epsilon_hat_norm + (3 * lam + 2 * mu) / (2 * mu) * tr * friction_alpha
            for i in qd.static(range(3)):
                S_new[i, i] = qd.exp(epsilon[i] - max(0, delta_gamma) / epsilon_hat_norm * epsilon_hat[i])

        F_new = U @ S_new @ V.transpose()
        return F_new, S_new, Jp_new

    @qd.func
    def update_stress(self, U, S, V, F_tmp, F_new, J, Jp, actu, m_dir, mu, lam):
        log_S_sum = gs.qd_float(0.0)
        center = qd.Matrix.zero(gs.qd_float, 3, 3)
        for i in qd.static(range(3)):
            log_S_sum += qd.log(S[i, i])
            center[i, i] = 2.0 * mu * qd.log(S[i, i]) * (1 / S[i, i])
        for i in qd.static(range(3)):
            center[i, i] += lam * log_S_sum * (1 / S[i, i])
        stress = U @ center @ V.transpose() @ F_new.transpose()
        return stress


@qd.data_oriented
class MPMSnowModel(MPMModel):
    """The model of a Snow material, clamping its principal stretches and hardening as it compacts."""

    @qd.func
    def update_F_S_Jp(
        self, J, F_tmp, U, S, V, Jp, mu, lam, yield_stress, critical_compression, critical_stretch, friction_alpha
    ):
        S_new = qd.Matrix.zero(gs.qd_float, 3, 3)
        Jp_new = Jp
        for d in qd.static(range(3)):
            S_new[d, d] = min(max(S[d, d], 1 - critical_compression), 1 + critical_stretch)
            Jp_new *= S[d, d] / S_new[d, d]
        F_new = U @ S_new @ V.transpose()
        return F_new, S_new, Jp_new

    @qd.func
    def update_stress(self, U, S, V, F_tmp, F_new, J, Jp, actu, m_dir, mu, lam):
        # The hardening coefficient stiffens the snow as it compacts
        h = qd.exp(10 * (1.0 - Jp))
        return self._compute_corotated_stress(U, V, F_new, J, mu * h, lam * h)


# The model class of each material, which compiles one model per variant (see 'MPMModel')
MPM_MODEL_CLS_BY_MATERIAL = {
    Elastic: MPMElasticModel,
    Muscle: MPMMuscleModel,
    ElastoPlastic: MPMElastoPlasticModel,
    Liquid: MPMLiquidModel,
    Sand: MPMSandModel,
    Snow: MPMSnowModel,
}

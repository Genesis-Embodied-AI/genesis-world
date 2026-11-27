"""
The constitutive models of the materials of the material point method (MPM) solver, bound to their options classes
through 'MaterialPhysics'.
"""

import math
from typing import ClassVar

import quadrants as qd

import genesis as gs
from genesis.engine.material_physics import MaterialPhysics
from genesis.engine.materials import MPM


@qd.func
def elastic_stress(
    J: float,
    U: qd.types.matrix(3, 3),
    V: qd.types.matrix(3, 3),
    F_tmp: qd.types.matrix(3, 3),
    F_new: qd.types.matrix(3, 3),
    mu: float,
    lam: float,
    corotated: qd.template(),
):
    """
    Return the Kirchhoff stress of an elastic material, of the fixed corotated model or of the neo-Hookean one.

    Parameters
    ----------
    J : float
        The determinant of the advected deformation gradient, which the neo-Hookean stress reads.
    U : qd.types.matrix(3, 3)
        The left rotation of the singular value decomposition of the advected deformation gradient.
    V : qd.types.matrix(3, 3)
        The right rotation of the singular value decomposition of the advected deformation gradient.
    F_tmp : qd.types.matrix(3, 3)
        The advected deformation gradient, which the neo-Hookean stress reads.
    F_new : qd.types.matrix(3, 3)
        The deformation gradient once plasticity applies, which the fixed corotated stress reads.
    mu : float
        The second Lame parameter.
    lam : float
        The first Lame parameter.
    corotated : bool
        Whether the stress is the fixed corotated one, which reads the rotation `U @ V^T`, or the neo-Hookean one.
    """
    stress = qd.Matrix.zero(gs.qd_float, 3, 3)
    if qd.static(corotated):
        stress = 2 * mu * (F_new - U @ V.transpose()) @ F_new.transpose() + qd.Matrix.identity(
            gs.qd_float, 3
        ) * lam * J * (J - 1)
    else:
        stress = mu * (F_tmp @ F_tmp.transpose()) + qd.Matrix.identity(gs.qd_float, 3) * (lam * qd.log(J) - mu)
    return stress


@qd.func
def clamp_singular_values(
    U: qd.types.matrix(3, 3),
    S: qd.types.matrix(3, 3),
    V: qd.types.matrix(3, 3),
    yield_lower: float,
    yield_higher: float,
):
    """
    Return the deformation gradient and its singular values once clamped to the elastic range of a material.

    Parameters
    ----------
    U : qd.types.matrix(3, 3)
        The left rotation of the singular value decomposition of the advected deformation gradient.
    S : qd.types.matrix(3, 3)
        The diagonal matrix of the singular values of the advected deformation gradient.
    V : qd.types.matrix(3, 3)
        The right rotation of the singular value decomposition of the advected deformation gradient.
    yield_lower : float
        The compression bound, which clamps the singular values from below at `1 - yield_lower`.
    yield_higher : float
        The stretch bound, which clamps the singular values from above at `1 + yield_higher`.
    """
    S_new = qd.Matrix.zero(gs.qd_float, 3, 3)
    for d in qd.static(range(3)):
        S_new[d, d] = min(max(S[d, d], 1 - yield_lower), 1 + yield_higher)
    F_new = U @ S_new @ V.transpose()
    return F_new, S_new


@qd.data_oriented
class MPMMaterialPhysics[MaterialT: MPM.Base](MaterialPhysics[MaterialT]):
    """
    The constitutive model of a material of the material point method (MPM) solver.

    The solver builds one instance per distinct material, whose parameters the kernels compile as constants.
    """

    PHYSICS_TYPES_MAP: ClassVar[dict[type[MPM.Base], type["MPMMaterialPhysics"]]] = {}

    # The plastic volume ratio of a particle at rest
    default_Jp: ClassVar[float] = 1.0

    @property
    def needs_svd(self) -> bool:
        """Whether the model reads the singular value decomposition U, S, V of the advected deformation gradient."""
        return True

    @qd.func
    def update_F_S_Jp(
        self,
        J: float,
        F_tmp: qd.types.matrix(3, 3),
        U: qd.types.matrix(3, 3),
        S: qd.types.matrix(3, 3),
        V: qd.types.matrix(3, 3),
        Jp: float,
        eps: float,
    ):
        """
        Return the deformation gradient, its singular values and the plastic volume ratio once plasticity applies.

        By default the material is purely elastic, and the substep keeps all three.

        Parameters
        ----------
        J : float
            The determinant of the advected deformation gradient.
        F_tmp : qd.types.matrix(3, 3)
            The deformation gradient advected over the substep.
        U : qd.types.matrix(3, 3)
            The left rotation of the singular value decomposition of `F_tmp`.
        S : qd.types.matrix(3, 3)
            The diagonal matrix of the singular values of `F_tmp`.
        V : qd.types.matrix(3, 3)
            The right rotation of the singular value decomposition of `F_tmp`.
        Jp : float
            The plastic volume ratio the substep starts from.
        eps : float
            The tolerance of the norms, which keeps their gradient finite at zero.
        """
        F_new = F_tmp
        S_new = S
        Jp_new = Jp
        return F_new, S_new, Jp_new

    @qd.func
    def update_stress(
        self,
        U: qd.types.matrix(3, 3),
        S: qd.types.matrix(3, 3),
        V: qd.types.matrix(3, 3),
        F_tmp: qd.types.matrix(3, 3),
        F_new: qd.types.matrix(3, 3),
        J: float,
        Jp: float,
        actu: float,
        m_dir: qd.types.vector(3),
    ):
        """
        Return the Kirchhoff stress of a particle, by default the one of the fixed corotated model.

        Parameters
        ----------
        U : qd.types.matrix(3, 3)
            The left rotation of the singular value decomposition of `F_tmp`.
        S : qd.types.matrix(3, 3)
            The diagonal matrix of the singular values once plasticity applies (see 'update_F_S_Jp').
        V : qd.types.matrix(3, 3)
            The right rotation of the singular value decomposition of `F_tmp`.
        F_tmp : qd.types.matrix(3, 3)
            The deformation gradient advected over the substep.
        F_new : qd.types.matrix(3, 3)
            The deformation gradient once plasticity applies (see 'update_F_S_Jp').
        J : float
            The determinant of `F_tmp`.
        Jp : float
            The plastic volume ratio the substep starts from.
        actu : float
            The actuation of the particle.
        m_dir : qd.types.vector(3)
            The direction of the muscle fibers of the particle, along which the actuation pulls.
        """
        return elastic_stress(J, U, V, F_tmp, F_new, self.material.mu, self.material.lam, corotated=True)


class ElasticPhysics(MPMMaterialPhysics[MPM.Elastic]):
    @property
    def needs_svd(self) -> bool:
        # The neo-Hookean stress reads the advected deformation gradient and its determinant alone
        return self.material.model == "corotation"

    @qd.func
    def update_stress(
        self,
        U: qd.types.matrix(3, 3),
        S: qd.types.matrix(3, 3),
        V: qd.types.matrix(3, 3),
        F_tmp: qd.types.matrix(3, 3),
        F_new: qd.types.matrix(3, 3),
        J: float,
        Jp: float,
        actu: float,
        m_dir: qd.types.vector(3),
    ):
        return elastic_stress(
            J, U, V, F_tmp, F_new, self.material.mu, self.material.lam, corotated=self.material.model == "corotation"
        )


class MusclePhysics(MPMMaterialPhysics[MPM.Muscle]):
    @property
    def needs_svd(self) -> bool:
        # See 'ElasticPhysics.needs_svd'
        return self.material.model == "corotation"

    @qd.func
    def update_stress(
        self,
        U: qd.types.matrix(3, 3),
        S: qd.types.matrix(3, 3),
        V: qd.types.matrix(3, 3),
        F_tmp: qd.types.matrix(3, 3),
        F_new: qd.types.matrix(3, 3),
        J: float,
        Jp: float,
        actu: float,
        m_dir: qd.types.vector(3),
    ):
        stress = elastic_stress(
            J, U, V, F_tmp, F_new, self.material.mu, self.material.lam, corotated=self.material.model == "corotation"
        )

        AAt = m_dir.outer_product(m_dir)
        stress += self.material.E * actu * F_tmp @ AAt @ F_tmp.transpose()

        return stress


class ElastoPlasticPhysics(MPMMaterialPhysics[MPM.ElastoPlastic]):
    @qd.func
    def update_F_S_Jp(
        self,
        J: float,
        F_tmp: qd.types.matrix(3, 3),
        U: qd.types.matrix(3, 3),
        S: qd.types.matrix(3, 3),
        V: qd.types.matrix(3, 3),
        Jp: float,
        eps: float,
    ):
        F_new = qd.Matrix.zero(gs.qd_float, 3, 3)
        S_new = qd.Matrix.zero(gs.qd_float, 3, 3)
        if qd.static(self.material.use_von_mises):
            # The singular values are bounded away from zero, where their logarithm is undefined
            S_new = qd.max(S, 0.05)
            epsilon = qd.Vector([qd.log(S_new[0, 0]), qd.log(S_new[1, 1]), qd.log(S_new[2, 2])])
            epsilon_hat = epsilon - (epsilon.sum() / 3)
            epsilon_hat_norm = epsilon_hat.norm(eps)
            delta_gamma = epsilon_hat_norm - self.material.von_mises_yield_stress / (2 * self.material.mu)

            # The material yields
            if delta_gamma > 0:
                epsilon -= (delta_gamma / epsilon_hat_norm) * epsilon_hat
                S_new = qd.Matrix.zero(gs.qd_float, 3, 3)
                for d in qd.static(range(3)):
                    S_new[d, d] = qd.exp(epsilon[d])
                F_new = U @ S_new @ V.transpose()
            else:
                F_new = F_tmp

        else:
            F_new, S_new = clamp_singular_values(U, S, V, self.material.yield_lower, self.material.yield_higher)

        Jp_new = Jp
        return F_new, S_new, Jp_new


class SnowPhysics(MPMMaterialPhysics[MPM.Snow]):
    @qd.func
    def update_F_S_Jp(
        self,
        J: float,
        F_tmp: qd.types.matrix(3, 3),
        U: qd.types.matrix(3, 3),
        S: qd.types.matrix(3, 3),
        V: qd.types.matrix(3, 3),
        Jp: float,
        eps: float,
    ):
        F_new, S_new = clamp_singular_values(U, S, V, self.material.yield_lower, self.material.yield_higher)
        Jp_new = Jp
        for d in qd.static(range(3)):
            Jp_new *= S[d, d] / S_new[d, d]
        return F_new, S_new, Jp_new

    @qd.func
    def update_stress(
        self,
        U: qd.types.matrix(3, 3),
        S: qd.types.matrix(3, 3),
        V: qd.types.matrix(3, 3),
        F_tmp: qd.types.matrix(3, 3),
        F_new: qd.types.matrix(3, 3),
        J: float,
        Jp: float,
        actu: float,
        m_dir: qd.types.vector(3),
    ):
        # Hardening coefficient: material harder when compressed
        h = qd.exp(10 * (1.0 - Jp))
        return elastic_stress(J, U, V, F_tmp, F_new, mu=self.material.mu * h, lam=self.material.lam * h, corotated=True)


class LiquidPhysics(MPMMaterialPhysics[MPM.Liquid]):
    @property
    def needs_svd(self) -> bool:
        # The shear term of a viscous liquid alone reads U and V
        return self.material.viscous

    @qd.func
    def update_F_S_Jp(
        self,
        J: float,
        F_tmp: qd.types.matrix(3, 3),
        U: qd.types.matrix(3, 3),
        S: qd.types.matrix(3, 3),
        V: qd.types.matrix(3, 3),
        Jp: float,
        eps: float,
    ):
        F_new = qd.Matrix.identity(gs.qd_float, 3) * qd.pow(J, 1.0 / 3.0)
        S_new = S
        Jp_new = Jp
        return F_new, S_new, Jp_new

    @qd.func
    def update_stress(
        self,
        U: qd.types.matrix(3, 3),
        S: qd.types.matrix(3, 3),
        V: qd.types.matrix(3, 3),
        F_tmp: qd.types.matrix(3, 3),
        F_new: qd.types.matrix(3, 3),
        J: float,
        Jp: float,
        actu: float,
        m_dir: qd.types.vector(3),
    ):
        stress = qd.Matrix.zero(gs.qd_float, 3, 3)
        if qd.static(self.material.viscous):
            # The shear term reads the advected deformation gradient, since plasticity leaves the liquid a pure dilation
            stress = elastic_stress(J, U, V, F_tmp, F_tmp, self.material.mu, self.material.lam, corotated=True)
        else:
            stress = qd.Matrix.identity(gs.qd_float, 3) * self.material.lam * J * (J - 1)
        return stress


class SandPhysics(MPMMaterialPhysics[MPM.Sand]):
    # The plastic volume ratio of the sand stores the volume correction of its projection, none at rest
    default_Jp: ClassVar[float] = 0.0

    def __init__(self, material: MPM.Sand) -> None:
        super().__init__(material)
        sin_phi = math.sin(math.radians(material.friction_angle))
        self.alpha = math.sqrt(2 / 3) * 2 * sin_phi / (3 - sin_phi)

    @qd.func
    def update_F_S_Jp(
        self,
        J: float,
        F_tmp: qd.types.matrix(3, 3),
        U: qd.types.matrix(3, 3),
        S: qd.types.matrix(3, 3),
        V: qd.types.matrix(3, 3),
        Jp: float,
        eps: float,
    ):
        S_new = qd.Matrix.zero(gs.qd_float, 3, 3)
        epsilon = qd.Vector.zero(gs.qd_float, 3)
        for i in qd.static(range(3)):
            epsilon[i] = qd.log(max(abs(S[i, i]), 1e-4))
            S_new[i, i] = 1
        tr = epsilon.sum() + Jp
        epsilon_hat = epsilon - tr / 3
        epsilon_hat_norm = epsilon_hat.norm(eps)

        Jp_new = gs.qd_float(0.0)
        if tr >= 0.0:
            Jp_new = tr
        else:
            Jp_new = 0.0
            delta_gamma = (
                epsilon_hat_norm
                + (3 * self.material.lam + 2 * self.material.mu) / (2 * self.material.mu) * tr * self.alpha
            )
            for i in qd.static(range(3)):
                S_new[i, i] = qd.exp(epsilon[i] - max(0, delta_gamma) / epsilon_hat_norm * epsilon_hat[i])

        F_new = U @ S_new @ V.transpose()
        return F_new, S_new, Jp_new

    @qd.func
    def update_stress(
        self,
        U: qd.types.matrix(3, 3),
        S: qd.types.matrix(3, 3),
        V: qd.types.matrix(3, 3),
        F_tmp: qd.types.matrix(3, 3),
        F_new: qd.types.matrix(3, 3),
        J: float,
        Jp: float,
        actu: float,
        m_dir: qd.types.vector(3),
    ):
        log_S_sum = gs.qd_float(0.0)
        center = qd.Matrix.zero(gs.qd_float, 3, 3)
        for i in qd.static(range(3)):
            log_S_sum += qd.log(S[i, i])
            center[i, i] = 2.0 * self.material.mu * qd.log(S[i, i]) * (1 / S[i, i])
        for i in qd.static(range(3)):
            center[i, i] += self.material.lam * log_S_sum * (1 / S[i, i])
        stress = U @ center @ V.transpose() @ F_new.transpose()
        return stress

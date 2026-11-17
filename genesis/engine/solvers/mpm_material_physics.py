"""
The constitutive models of the materials of the material point method (MPM) solver.

A user extends the solver with a subclass of an options class of 'gs.materials.MPM' and a physics for it,
`class MyPhysics(MPMMaterialPhysics[MyMaterial])` (see 'MaterialPhysics').
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

    The fixed corotated stress reads the deformation gradient `F_new` once plasticity applies and the rotation `U @ V^T`
    of the singular value decomposition of the advected deformation gradient. The neo-Hookean stress reads the advected
    deformation gradient `F_tmp` and its determinant `J` alone. `mu` and `lam` are the Lame parameters.
    """
    stress = qd.Matrix.zero(gs.qd_float, 3, 3)
    if qd.static(corotated):
        stress = 2 * mu * (F_new - U @ V.transpose()) @ F_new.transpose() + qd.Matrix.identity(
            gs.qd_float, 3
        ) * lam * J * (J - 1)
    else:
        stress = mu * (F_tmp @ F_tmp.transpose()) + qd.Matrix.identity(gs.qd_float, 3) * (lam * qd.log(J) - mu)
    return stress


@qd.data_oriented
class MPMMaterialPhysics[MaterialT: MPM.Base](MaterialPhysics[MaterialT]):
    """
    The constitutive model of a material of the material point method (MPM) solver.

    The solver builds one instance per distinct material, whose parameters the kernels compile as constants.
    """

    PHYSICS_TYPES_MAP: ClassVar[dict[type[MPM.Base], type["MPMMaterialPhysics"]]] = {}

    @property
    def needs_svd(self) -> bool:
        """Whether the model reads the singular value decomposition U, S, V of the advected deformation gradient."""
        return True

    @property
    def default_Jp(self) -> float:
        """The plastic volume ratio of a particle at rest."""
        return 1.0

    @qd.func
    def update_F_S_Jp(self, J, F_tmp, U, S, V, Jp, eps):
        """
        Return the deformation gradient, its singular values and the plastic volume ratio once plasticity applies.

        `F_tmp` is the deformation gradient advected over the substep, of determinant `J` and singular value
        decomposition `U`, `S`, `V`. `Jp` is the plastic volume ratio the substep starts from. By default the material
        is purely elastic, and the substep keeps all three.
        """
        F_new = F_tmp
        S_new = S
        Jp_new = Jp
        return F_new, S_new, Jp_new

    @qd.func
    def update_stress(self, U, S, V, F_tmp, F_new, J, Jp, actu, m_dir):
        """
        Return the Kirchhoff stress of a particle, by default the one of the fixed corotated model.

        `F_new` and `S` are the deformation gradient and its singular values once plasticity applies (see
        'update_F_S_Jp'). `actu` is the actuation of the particle along its muscle direction `m_dir`.
        """
        return elastic_stress(J, U, V, F_tmp, F_new, self.material.mu, self.material.lam, corotated=True)


class ElasticPhysics(MPMMaterialPhysics[MPM.Elastic]):
    @property
    def needs_svd(self) -> bool:
        # The neo-Hookean stress reads the advected deformation gradient and its determinant alone
        return self.material.model == "corotation"

    @qd.func
    def update_stress(self, U, S, V, F_tmp, F_new, J, Jp, actu, m_dir):
        return elastic_stress(
            J, U, V, F_tmp, F_new, self.material.mu, self.material.lam, corotated=self.material.model == "corotation"
        )


class MusclePhysics(MPMMaterialPhysics[MPM.Muscle]):
    @property
    def needs_svd(self) -> bool:
        # See 'ElasticPhysics.needs_svd'
        return self.material.model == "corotation"

    @qd.func
    def update_stress(self, U, S, V, F_tmp, F_new, J, Jp, actu, m_dir):
        stress = elastic_stress(
            J, U, V, F_tmp, F_new, self.material.mu, self.material.lam, corotated=self.material.model == "corotation"
        )

        AAt = m_dir.outer_product(m_dir)
        stress += self.material.E * actu * F_tmp @ AAt @ F_tmp.transpose()

        return stress


class ElastoPlasticPhysics(MPMMaterialPhysics[MPM.ElastoPlastic]):
    @qd.func
    def update_F_S_Jp(self, J, F_tmp, U, S, V, Jp, eps):
        F_new = qd.Matrix.zero(gs.qd_float, 3, 3)
        S_new = qd.Matrix.zero(gs.qd_float, 3, 3)
        if qd.static(self.material.use_von_mises):
            S_new = qd.max(S, 0.05)  # to prevent NaN
            epsilon = qd.Vector([qd.log(S_new[0, 0]), qd.log(S_new[1, 1]), qd.log(S_new[2, 2])])
            epsilon_hat = epsilon - (epsilon.sum() / 3)
            epsilon_hat_norm = epsilon_hat.norm(eps)
            delta_gamma = epsilon_hat_norm - self.material.von_mises_yield_stress / (2 * self.material.mu)

            if delta_gamma > 0:  # Yields
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
                S_new[d, d] = min(max(S[d, d], 1 - self.material.yield_lower), 1 + self.material.yield_higher)
            F_new = U @ S_new @ V.transpose()

        Jp_new = Jp
        return F_new, S_new, Jp_new


class SnowPhysics(MPMMaterialPhysics[MPM.Snow]):
    @qd.func
    def update_F_S_Jp(self, J, F_tmp, U, S, V, Jp, eps):
        S_new = qd.Matrix.zero(gs.qd_float, 3, 3)
        Jp_new = Jp
        for d in qd.static(range(3)):
            S_new[d, d] = min(max(S[d, d], 1 - self.material.yield_lower), 1 + self.material.yield_higher)
            Jp_new *= S[d, d] / S_new[d, d]
        F_new = U @ S_new @ V.transpose()
        return F_new, S_new, Jp_new

    @qd.func
    def update_stress(self, U, S, V, F_tmp, F_new, J, Jp, actu, m_dir):
        # Hardening coefficient: material harder when compressed
        h = qd.exp(10 * (1.0 - Jp))
        mu, lam = self.material.mu * h, self.material.lam * h

        r = U @ V.transpose()
        stress = 2 * mu * (F_new - r) @ F_new.transpose() + qd.Matrix.identity(gs.qd_float, 3) * lam * J * (J - 1)

        return stress


class LiquidPhysics(MPMMaterialPhysics[MPM.Liquid]):
    @property
    def needs_svd(self) -> bool:
        # The shear term of a viscous liquid alone reads U and V
        return self.material.viscous

    @qd.func
    def update_F_S_Jp(self, J, F_tmp, U, S, V, Jp, eps):
        F_new = qd.Matrix.identity(gs.qd_float, 3) * qd.pow(J, 1.0 / 3.0)
        S_new = S
        Jp_new = Jp
        return F_new, S_new, Jp_new

    @qd.func
    def update_stress(self, U, S, V, F_tmp, F_new, J, Jp, actu, m_dir):
        stress = qd.Matrix.identity(gs.qd_float, 3) * self.material.lam * J * (J - 1)
        if qd.static(self.material.viscous):
            stress = 2 * self.material.mu * (F_tmp - U @ V.transpose()) @ F_tmp.transpose() + stress
        return stress


class SandPhysics(MPMMaterialPhysics[MPM.Sand]):
    def __init__(self, material: MPM.Sand) -> None:
        super().__init__(material)
        sin_phi = math.sin(math.radians(material.friction_angle))
        self.alpha = math.sqrt(2 / 3) * 2 * sin_phi / (3 - sin_phi)

    @property
    def default_Jp(self) -> float:
        # The plastic volume ratio of the sand stores the volume correction of its projection, none at rest
        return 0.0

    @qd.func
    def update_F_S_Jp(self, J, F_tmp, U, S, V, Jp, eps):
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
    def update_stress(self, U, S, V, F_tmp, F_new, J, Jp, actu, m_dir):
        log_S_sum = gs.qd_float(0.0)
        center = qd.Matrix.zero(gs.qd_float, 3, 3)
        for i in qd.static(range(3)):
            log_S_sum += qd.log(S[i, i])
            center[i, i] = 2.0 * self.material.mu * qd.log(S[i, i]) * (1 / S[i, i])
        for i in qd.static(range(3)):
            center[i, i] += self.material.lam * log_S_sum * (1 / S[i, i])
        stress = U @ center @ V.transpose() @ F_new.transpose()
        return stress

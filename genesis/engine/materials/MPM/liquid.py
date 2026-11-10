from typing import Any

import quadrants as qd
from pydantic import StrictBool

import genesis as gs

from .base import Base
from .elastic import func_cofactor, func_cofactor_jacobian


@qd.data_oriented
class Liquid(Base):
    """
    The liquid material class for MPM.

    Parameters
    ----------
    E : float, optional
        Young's modulus. Default is 1e6.
    nu : float, optional
        Poisson ratio. Default is 0.2.
    rho : float, optional
        Density (kg/m³). Default is 1000.
    viscous : bool, optional
        Whether the liquid is viscous. Simply sets mu to zero when non-viscous. Default is False.
    """

    viscous: StrictBool = False

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        if not self.viscous:
            self.mu = 0.0
        self.update_F_S_Jp = self._update_F_S_Jp_liquid
        self.update_stress = self._update_stress_liquid
        # Viscous liquid uses U @ V.T in the 2*mu*(F_tmp - U@V.T)@F_tmp.T term; when mu==0 that term vanishes and SVD
        # is pure waste.
        self.needs_svd = self.viscous

    @qd.func
    def _update_F_S_Jp_liquid(self, J, F_tmp, U, S, V, Jp):
        F_new = qd.Matrix.identity(gs.qd_float, 3) * qd.pow(J, 1.0 / 3.0)
        S_new = S
        Jp_new = Jp
        return F_new, S_new, Jp_new

    @qd.func
    def _update_stress_liquid(self, U, S, V, F_tmp, F_new, J, Jp, actu, m_dir):
        stress = 2 * self.mu * (F_tmp - U @ V.transpose()) @ F_tmp.transpose() + qd.Matrix.identity(
            gs.qd_float, 3
        ) * self.lam * J * (J - 1)
        return stress


@qd.func
def func_liquid_volume_terms(J: float, lam: float, band: float):
    """
    Return the volume term of the energy density of a liquid, and its first and second derivatives in the volume ratio.

    A liquid resists compression alone. Within `band` below the rest volume, the term rises as a quintic in the
    compression that meets the full term with its value and its first two derivatives at the edge of the band, so that
    the stiffness of a liquid near rest fades in smoothly from zero.
    """
    psi = gs.qd_float(0.0)
    dpsi = gs.qd_float(0.0)
    d2psi = gs.qd_float(0.0)
    compression = 1.0 - J
    if compression >= band and compression > 0.0:
        psi = 0.25 * lam * (J * J - 1.0 - 2.0 * qd.log(J))
        dpsi = 0.5 * lam * (J - 1.0 / J)
        d2psi = 0.5 * lam * (1.0 + 1.0 / (J * J))
    elif compression > 0.0:
        J_band = 1.0 - band
        value = 0.25 * lam * (J_band * J_band - 1.0 - 2.0 * qd.log(J_band))
        # The slope and the curvature of the full term in the compression at the edge of the band, scaled by the band
        slope = -0.5 * lam * (J_band - 1.0 / J_band) * band
        curvature = 0.5 * lam * (1.0 + 1.0 / (J_band * J_band)) * band * band
        c3 = 10.0 * value - 4.0 * slope + 0.5 * curvature
        c4 = 7.0 * slope - 15.0 * value - curvature
        c5 = 0.5 * curvature - 3.0 * slope + 6.0 * value
        t = compression / band
        psi = t * t * t * (c3 + t * (c4 + t * c5))
        dpsi = -t * t * (3.0 * c3 + t * (4.0 * c4 + t * 5.0 * c5)) / band
        d2psi = t * (6.0 * c3 + t * (12.0 * c4 + t * 20.0 * c5)) / (band * band)
    return psi, dpsi, d2psi


@qd.func
def func_liquid_principal(sigma: qd.types.vector(3), mu: float, lam: float, band: float):
    """
    Return the energy density of a liquid, its gradient and its Hessian in the signed singular values.

    The energy sums a viscous shear term of modulus `mu` and the volume term (see 'func_liquid_volume_terms').
    """
    cofactor = func_cofactor(sigma)
    psi_volume, dpsi, d2psi = func_liquid_volume_terms(sigma[0] * sigma[1] * sigma[2], lam, band)
    psi = mu * (sigma - 1.0).dot(sigma - 1.0) + psi_volume
    g = 2.0 * mu * (sigma - 1.0) + dpsi * cofactor
    A = 2.0 * mu * qd.Matrix.identity(gs.qd_float, 3)
    A += d2psi * cofactor.outer_product(cofactor) + dpsi * func_cofactor_jacobian(sigma)
    return psi, g, A


@qd.func
def func_liquid_return_map(F: qd.types.matrix(3, 3), J_min: float):
    """
    Return the deformation gradient of a liquid once it forgets its shape, its volume ratio kept within its range.

    The volume ratio stays between `J_min` and the rest volume, a liquid resisting compression alone.
    """
    J = qd.min(qd.max(F.determinant(), J_min), 1.0)
    return qd.pow(J, 1.0 / 3.0) * qd.Matrix.identity(gs.qd_float, 3)

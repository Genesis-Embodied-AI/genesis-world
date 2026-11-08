from typing import TYPE_CHECKING, Any

import quadrants as qd
from pydantic import Field, PrivateAttr

import genesis as gs
from genesis.typing import PositiveInt, ValidFloat

from .elastic import Elastic

if TYPE_CHECKING:
    from genesis.engine.solvers.fem_solver import FEMSolver


@qd.data_oriented
class Muscle(Elastic):
    """
    The muscle material class for FEM.

    Parameters
    ----------
    E : float, optional
        Young's modulus, which controls stiffness. Default is 1e6.
    nu : float, optional
        Poisson ratio. Default is 0.2.
    rho : float, optional
        Material density (kg/m^3). Default is 1000.
    model : str, optional
        Constitutive model ('linear' or 'stable_neohookean'). Default is 'linear'.
    n_groups : int, optional
        Number of muscle groups. Default is 1.
    friction_mu : float, optional
        Contact friction coefficient. Default is 0.1.
    contact_resistance : float | None, optional
        IPC contact resistance/stiffness override. Default is None.
    """

    n_groups: PositiveInt = 1

    # Auto-generated, equal to E.
    stiffness: ValidFloat = Field(default=0.0, exclude=True)

    # The small number added to the squared length of a fiber before taking its root, which keeps it positive
    _fiber_length_eps: Any = PrivateAttr(default=None)
    # The implicit solver applies the active stress of the fibers along their direction at the start of a substep, as
    # the explicit one does, which leaves the energy quadratic in the positions at its end
    _fibers_dir: Any = PrivateAttr(default=None)

    _update_stress_without_actuation: Any = PrivateAttr(default=None)
    _build_without_actuation: Any = PrivateAttr(default=None)
    _pre_compute_without_actuation: Any = PrivateAttr(default=None)
    _pre_compute_grad_without_actuation: Any = PrivateAttr(default=None)
    _compute_energy_gradient_without_actuation: Any = PrivateAttr(default=None)
    _compute_energy_without_actuation: Any = PrivateAttr(default=None)

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)

        self.stiffness = self.E
        self._update_stress_without_actuation = self.update_stress
        self.update_stress = self._update_stress_with_actuation
        self._build_without_actuation = self.build
        self.build = self._build_with_actuation
        self._pre_compute_without_actuation = self.pre_compute
        self.pre_compute = self._pre_compute_with_actuation
        self._pre_compute_grad_without_actuation = self.pre_compute_grad
        self.pre_compute_grad = self._pre_compute_grad_with_actuation
        self._compute_energy_gradient_without_actuation = self.compute_energy_gradient
        self.compute_energy_gradient = self._compute_energy_gradient_with_actuation
        self._compute_energy_without_actuation = self.compute_energy
        self.compute_energy = self._compute_energy_with_actuation

    def _build_with_actuation(self, fem_solver: "FEMSolver") -> None:
        self._build_without_actuation(fem_solver)
        self._fiber_length_eps = qd.field(dtype=gs.qd_float, shape=())
        self._fiber_length_eps[None] = 1e-12
        if fem_solver._use_implicit_solver:
            self._fibers_dir = qd.field(dtype=gs.qd_vec3, shape=(fem_solver._B, fem_solver.n_elements))

    @qd.func
    def _pre_compute_with_actuation(
        self, i_e: int, i_b: int, J: float, F: qd.types.matrix(3, 3), m_dir: qd.types.vector(3)
    ):
        self._pre_compute_without_actuation(i_e, i_b, J, F, m_dir)
        fiber = F @ m_dir
        self._fibers_dir[i_b, i_e] = fiber / fiber.norm(self._fiber_length_eps[None])

    @qd.func
    def _pre_compute_grad_with_actuation(
        self,
        i_e: int,
        i_b: int,
        mu: float,
        lam: float,
        F_start: qd.types.matrix(3, 3),
        F: qd.types.matrix(3, 3),
        dF_damping: qd.types.matrix(3, 3),
        gradient_grad: qd.types.matrix(3, 3),
        actu: float,
        m_dir: qd.types.vector(3),
    ):
        F_start_grad = self._pre_compute_grad_without_actuation(
            i_e, i_b, mu, lam, F_start, F, dF_damping, gradient_grad, actu, m_dir
        )
        # The fiber direction enters the energy gradient through the active stress stiffness * actu * fiber_dir m^T
        fiber_dir_grad = self.stiffness * actu * gradient_grad @ m_dir
        fiber = F_start @ m_dir
        fiber_length = fiber.norm(self._fiber_length_eps[None])
        fiber_dir = fiber / fiber_length
        fiber_grad = (fiber_dir_grad - fiber_dir.dot(fiber_dir_grad) * fiber_dir) / fiber_length
        return F_start_grad + fiber_grad.outer_product(m_dir)

    @qd.func
    def _compute_energy_gradient_with_actuation(
        self,
        i_e: int,
        i_b: int,
        mu: float,
        lam: float,
        J: float,
        F: qd.types.matrix(3, 3),
        actu: float,
        m_dir: qd.types.vector(3),
    ):
        energy, gradient = self._compute_energy_gradient_without_actuation(i_e, i_b, mu, lam, J, F, actu, m_dir)
        stress = self.stiffness * actu * self._fibers_dir[i_b, i_e].outer_product(m_dir)
        return energy + (stress * F).sum(), gradient + stress

    @qd.func
    def _compute_energy_with_actuation(
        self,
        i_e: int,
        i_b: int,
        mu: float,
        lam: float,
        J: float,
        F: qd.types.matrix(3, 3),
        actu: float,
        m_dir: qd.types.vector(3),
    ):
        energy = self._compute_energy_without_actuation(i_e, i_b, mu, lam, J, F, actu, m_dir)
        return energy + self.stiffness * actu * self._fibers_dir[i_b, i_e].dot(F @ m_dir)

    @qd.func
    def _update_stress_with_actuation(
        self,
        mu: float,
        lam: float,
        J: float,
        F: qd.types.matrix(3, 3),
        actu: float,
        m_dir: qd.types.vector(3),
    ):
        stress = self._update_stress_without_actuation(mu, lam, J, F, actu, m_dir)

        l = (F @ m_dir).norm(self._fiber_length_eps[None])
        mmT = m_dir.outer_product(m_dir)
        stress += self.stiffness * (actu / l) * F @ mmT

        return stress

from typing import TYPE_CHECKING, Annotated, Any

import quadrants as qd
from pydantic import Field

import genesis as gs
from genesis.typing import NonNegativeFloat, PositiveFloat, StrictInt, ValidFloat

from ..base import Material

if TYPE_CHECKING:
    from genesis.engine.entities.fem_entity import FEMEntity


@qd.data_oriented
class Base(Material["FEMEntity"]):
    """
    The base class of FEM materials.

    Note
    ----
    This class should *not* be instantiated directly.

    Parameters
    ----------
    E : float, optional
        Young's modulus, which controls stiffness. Default is 1e6.
    nu : float, optional
        Poisson ratio, describing the material's volume change under stress. Default is 0.2.
    rho : float, optional
        Material density (kg/m^3). Default is 1000.
    hydroelastic_modulus : float, optional
        Hydroelastic modulus for hydroelastic contact. Default is 1e7.
    friction_mu : float, optional
        Friction coefficient. Default is 0.1.
    contact_resistance : float | None, optional
        IPC contact resistance/stiffness override for this material. ``None`` means
        use the IPC coupler global default (``IPCCouplerOptions.contact_resistance``).
    """

    E: PositiveFloat = 1e6
    nu: Annotated[ValidFloat, Field(gt=-1.0, lt=0.5)] = 0.2
    rho: PositiveFloat = 1000.0
    hydroelastic_modulus: PositiveFloat = 1e7
    friction_mu: NonNegativeFloat = 0.1
    contact_resistance: PositiveFloat | None = None

    # Dispatch fields, set by subclass model_post_init and never by the user.
    build: Any = Field(default=None, exclude=True, repr=False)
    # The data the implicit solver computes from the deformation gradient F_start at the start of a substep, and its
    # reverse, which returns the adjoint of F_start from the adjoint of the energy gradient at the end of the substep,
    # which the data enters along with the change dF_damping of the deformation gradient driving the stiffness damping.
    pre_compute: Any = Field(default=None, exclude=True, repr=False)
    pre_compute_grad: Any = Field(default=None, exclude=True, repr=False)
    update_stress: Any = Field(default=None, exclude=True, repr=False)
    # The product of the Hessian of the energy density with a change of the deformation gradient, which the implicit
    # solver takes in place of the Hessian itself.
    hessian_product: Any = Field(default=None, exclude=True, repr=False)
    compute_energy_gradient: Any = Field(default=None, exclude=True, repr=False)
    compute_energy: Any = Field(default=None, exclude=True, repr=False)

    # Auto-generated fields, computed in model_post_init and never set by the user.
    mu: ValidFloat = Field(default=0.0, exclude=True)
    lam: ValidFloat = Field(default=0.0, exclude=True)
    idx: StrictInt | None = Field(default=None, exclude=True)

    def model_post_init(self, context: Any) -> None:
        self.mu = self.E / (2.0 * (1.0 + self.nu))
        self.lam = self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))

        # Set dispatch defaults
        if self.build is None:
            self.build = self._build_noop
        if self.pre_compute is None:
            self.pre_compute = self._pre_compute_noop
        if self.pre_compute_grad is None:
            self.pre_compute_grad = self._pre_compute_grad_noop
        if self.update_stress is None:
            self.update_stress = self._update_stress_noop
        if self.hessian_product is None:
            self.hessian_product = self._hessian_product_noop
        if self.compute_energy_gradient is None:
            self.compute_energy_gradient = self._compute_energy_gradient_noop
        if self.compute_energy is None:
            self.compute_energy = self._compute_energy_noop

    def _build_noop(self, fem_solver):
        pass

    @qd.func
    def _pre_compute_noop(self, i_e: int, i_b: int, J: float, F: qd.types.matrix(3, 3), m_dir: qd.types.vector(3)):
        pass

    @qd.func
    def _pre_compute_grad_noop(
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
        return qd.Matrix.zero(gs.qd_float, 3, 3)

    # A material without a constitutive model (Cloth, whose physics the IPC coupler owns) dispatches stress here.
    # The explicit solver compiles this into its static per-material loop even when no element ever executes it, so
    # it must return a zero response rather than raise, which quadrants cannot compile.
    @qd.func
    def _update_stress_noop(
        self,
        mu: float,
        lam: float,
        J: float,
        F: qd.types.matrix(3, 3),
        actu: float,
        m_dir: qd.types.vector(3),
    ):
        return qd.Matrix.zero(gs.qd_float, 3, 3)

    @qd.func
    def _hessian_product_noop(self, i_e: int, i_b: int, mu: float, lam: float, dF: qd.types.matrix(3, 3)):
        raise NotImplementedError

    @qd.func
    def _compute_energy_gradient_noop(
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
        raise NotImplementedError

    @qd.func
    def _compute_energy_noop(
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
        raise NotImplementedError

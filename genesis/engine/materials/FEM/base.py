from typing import TYPE_CHECKING, Annotated, Any

import quadrants as qd
from pydantic import Field, PrivateAttr, StrictBool

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
        Material density (kg/m³). Default is 1000.
    hydroelastic_modulus : float, optional
        Hydroelastic modulus for hydroelastic contact. Default is 1e7.
    friction_mu : float, optional
        Friction coefficient. Default is 0.1.
    contact_resistance : float | None, optional
        IPC contact resistance/stiffness override for this material. ``None`` means
        use the IPC coupler global default (``IPCCouplerOptions.contact_resistance``).
    hessian_invariant : bool, optional
        If True, Hessian is computed only once. Default is False.
    """

    E: PositiveFloat = 1e6
    nu: Annotated[ValidFloat, Field(gt=-1.0, lt=0.5)] = 0.2
    rho: PositiveFloat = 1000.0
    hydroelastic_modulus: PositiveFloat = 1e7
    friction_mu: NonNegativeFloat = 0.1
    contact_resistance: PositiveFloat | None = None
    hessian_invariant: StrictBool = False

    # Dispatch fields — set by subclass model_post_init, not user-specified.
    build: Any = Field(default=None, exclude=True, repr=False)
    pre_compute: Any = Field(default=None, exclude=True, repr=False)
    update_stress: Any = Field(default=None, exclude=True, repr=False)
    compute_energy_gradient_hessian: Any = Field(default=None, exclude=True, repr=False)
    compute_energy_gradient: Any = Field(default=None, exclude=True, repr=False)
    compute_energy: Any = Field(default=None, exclude=True, repr=False)

    # Auto-generated fields — computed in model_post_init, not user-specified.
    mu: ValidFloat = Field(default=0.0, exclude=True)
    lam: ValidFloat = Field(default=0.0, exclude=True)
    idx: StrictInt | None = Field(default=None, exclude=True)

    # Internal solver state — does not belong in an options class, kept pending larger refactor.
    _hessian_ready: bool = PrivateAttr(default=False)

    def model_post_init(self, context: Any) -> None:
        self.mu = self.E / (2.0 * (1.0 + self.nu))
        self.lam = self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))

        # Set dispatch defaults
        if self.build is None:
            self.build = self._build_noop
        if self.pre_compute is None:
            self.pre_compute = self._pre_compute_noop
        if self.update_stress is None:
            self.update_stress = self._update_stress_noop
        if self.compute_energy_gradient_hessian is None:
            self.compute_energy_gradient_hessian = self._compute_energy_gradient_hessian_noop
        if self.compute_energy_gradient is None:
            self.compute_energy_gradient = self._compute_energy_gradient_noop
        if self.compute_energy is None:
            self.compute_energy = self._compute_energy_noop

    def _build_noop(self, fem_solver):
        pass

    @qd.func
    def _pre_compute_noop(self, J, F, i_e, i_b):
        pass

    # A material without a constitutive model (Cloth, whose physics the IPC coupler owns) dispatches stress here.
    # The explicit solver compiles this into its static per-material loop even when no element ever executes it, so
    # it must return a zero response rather than raise, which quadrants cannot compile.
    @qd.func
    def _update_stress_noop(self, mu, lam, J, F, actu, m_dir):
        return qd.Matrix.zero(gs.qd_float, 3, 3)

    @qd.func
    def _compute_energy_gradient_hessian_noop(self, mu, lam, J, F, actu, m_dir, i_e, i_b, hessian_field):
        raise NotImplementedError

    @qd.func
    def _compute_energy_gradient_noop(self, mu, lam, J, F, actu, m_dir, i_e, i_b):
        raise NotImplementedError

    @qd.func
    def _compute_energy_noop(self, mu, lam, J, F, actu, m_dir, i_e, i_b):
        raise NotImplementedError

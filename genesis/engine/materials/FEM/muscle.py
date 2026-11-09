from typing import Any

from pydantic import Field

import quadrants as qd

from genesis.typing import PositiveInt, ValidFloat

from .elastic import Elastic


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
        Constitutive model ('linear', 'stable_neohookean' or 'linear_corotated'). Default is 'linear'.
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

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        self.stiffness = self.E


# The active stress of the fibers of a muscle adds to the stress of its elastic model. The explicit solver applies it
# along the fibers as they deform, while the implicit one applies it along their direction at the start of a substep,
# which leaves the energy quadratic in the positions at its end.


@qd.func
def func_muscle_fiber_dir(F: qd.types.matrix(3, 3), m_dir: qd.types.vector(3), fiber_length_eps: float):
    """
    Return the direction of the fibers of a muscle element once deformed.

    `m_dir` is the rest direction of the fibers. `fiber_length_eps` is added to the squared length of a fiber before
    taking its root, which keeps it positive.
    """
    fiber = F @ m_dir
    return fiber / fiber.norm(fiber_length_eps)


@qd.func
def func_muscle_stress(
    stiffness: float, actu: float, F: qd.types.matrix(3, 3), m_dir: qd.types.vector(3), fiber_length_eps: float
):
    """Return the active stress of the fibers of a muscle element along their deformed direction."""
    l = (F @ m_dir).norm(fiber_length_eps)
    return stiffness * (actu / l) * F @ m_dir.outer_product(m_dir)


@qd.func
def func_muscle_fiber_stress(stiffness: float, actu: float, m_dir: qd.types.vector(3), fiber_dir: qd.types.vector(3)):
    """
    Return the active stress of the fibers of a muscle element along their direction at the start of a substep.

    It adds to the gradient of the energy density, the energy density gaining its scalar product with the deformation
    gradient.
    """
    return stiffness * actu * fiber_dir.outer_product(m_dir)


@qd.func
def func_muscle_fiber_dir_grad(
    stiffness: float,
    actu: float,
    F_start: qd.types.matrix(3, 3),
    m_dir: qd.types.vector(3),
    gradient_grad: qd.types.matrix(3, 3),
    fiber_length_eps: float,
):
    """
    Return the adjoint of the deformation gradient at the start of a substep that the direction of the fibers carries.

    The direction enters the gradient of the energy density through the active stress stiffness * actu * fiber_dir m^T,
    whose adjoint `gradient_grad` is.
    """
    fiber_dir_grad = stiffness * actu * gradient_grad @ m_dir
    fiber = F_start @ m_dir
    fiber_length = fiber.norm(fiber_length_eps)
    fiber_dir = fiber / fiber_length
    fiber_grad = (fiber_dir_grad - fiber_dir.dot(fiber_dir_grad) * fiber_dir) / fiber_length
    return fiber_grad.outer_product(m_dir)

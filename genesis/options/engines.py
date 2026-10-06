"""Options selecting the concrete simulator implementation owned by a Scene."""

from genesis.typing import NonNegativeFloat, PositiveFloat

from .options import Options


class BaseEngineOptions(Options):
    """Base class for Scene simulation-engine options."""


class LegacyEngineOptions(BaseEngineOptions):
    """Use the existing solver-substep and inter-solver coupling runtime."""


class NewtonEngineOptions(BaseEngineOptions):
    """Use the graph-native QCloth, Rigid, and Consistent IPC runtime.

    Parameters
    ----------
    contact_d_hat : float, optional
        IPC activation distance in meters. Defaults to 1e-3.
    contact_friction_mu : float, optional
        Global Coulomb friction coefficient. Defaults to 1.0.
    contact_resistance : float, optional
        Global IPC barrier resistance. Defaults to 1e4.
    contact_eps_velocity : float, optional
        Friction velocity regularization in meters per second. Defaults to 1e-2.
    """

    contact_d_hat: PositiveFloat = 1e-3
    contact_friction_mu: NonNegativeFloat = 1.0
    contact_resistance: PositiveFloat = 1e4
    contact_eps_velocity: PositiveFloat = 1e-2

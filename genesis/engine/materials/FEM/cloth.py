"""
Cloth material for IPC-based cloth simulation.

This material is used with FEMEntity and IPCCoupler for shell/membrane simulation.
"""

from typing import Literal

from genesis.typing import NonNegativeFloat, PositiveFloat

from .base import Base


class Cloth(Base):
    """
    Cloth material for thin shell/membrane simulation using IPC.

    This material is designed for cloth, fabric, and other thin flexible materials.
    It uses shell-based FEM formulation (NeoHookeanShell) in the IPC backend.

    When used with FEMEntity, it signals to IPCCoupler that this entity should be
    treated as a 2D shell (cloth) rather than a 3D volumetric FEM object.

    Parameters
    ----------
    E : float, optional
        Young's modulus (Pa), controlling stiffness. Default is 1e4 (10 kPa).
    nu : float, optional
        Poisson's ratio, describing volume change under stress.
        Default is 0.49 (nearly incompressible).
    rho : float, optional
        Material density (kg/m³). Default is 200 (typical fabric).
    thickness : float, optional
        Shell thickness (m). Default is 0.001 (1mm).
    bending_stiffness : float, optional
        Bending resistance coefficient. If None, no bending resistance.
        Default is None.
    model : str, optional
        FEM material model (not used for cloth, kept for compatibility).
        Default is "stable_neohookean".
    friction_mu : float, optional
        Friction coefficient. Default is 0.1.
    contact_resistance : float | None, optional
        IPC contact resistance/stiffness override. ``None`` uses the IPC coupler
        global default. Default is None.

    Notes
    -----
    - Only works with IPCCoupler enabled
    - Requires GPU backend
    - Only accepts surface mesh morphs (Mesh, etc.)
    - Uses FEMEntity infrastructure but simulated as 2D shell in IPC

    Examples
    --------
    >>> cloth = scene.add_entity(
    ...     options=gs.options.FEMEntityOptions(
    ...         morph=gs.morphs.Mesh(file="cloth.obj"),
    ...         material=gs.materials.FEM.Cloth(
    ...             E=10e3, nu=0.49, rho=200, thickness=0.001, bending_stiffness=10.0
    ...         ),
    ...     ),
    ... )
    """

    E: PositiveFloat = 1e4
    nu: PositiveFloat = 0.49
    rho: PositiveFloat = 200.0
    thickness: PositiveFloat = 0.001
    bending_stiffness: NonNegativeFloat | None = None
    model: Literal["linear", "stable_neohookean", "linear_corotated"] = "stable_neohookean"

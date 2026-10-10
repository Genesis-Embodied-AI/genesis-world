from typing import Any, Literal

import numpy as np
from pydantic import PrivateAttr, StrictBool, model_validator

import genesis as gs
from genesis.typing import NonNegativeFloat, NonNegativeInt, PositiveFloat, PositiveInt, UnitVec4FType, Vec3FType

from .options import Options

############################ Top level: simulator and coupler ############################
"""
Simulator options specifies the global settings for the simulator and the coupler options specifies whether the
coupling between pairs of solvers is enabled.
"""


class SimOptions(Options):
    """
    Options configuring the top-level simulator.

    Note
    ----
    1. `SimOptions` specifies the global settings for the simulator. Some parameters exist both in `SimOptions` and
    `SolverOptions`. In this case, if such parameters are given in `SolverOptions`, it will override the one specified
    in `SimOptions` for this specific solver. For example, if `dt` is only given in `SimOptions`, it will be shared by
    all the solvers, while a solver given its own `dt` integrates over that interval instead, and the number of
    substeps of every solver follows from it.

    2. In differentiable mode, `substeps_local` must be divisible by `substeps`, as external command is input per
    `step`, but `substep`. If `requires_grad` is False, we can use arbitrary `substeps_local`.

    Parameters
    ----------
    dt : float, optional
        Time duration for each simulation step in seconds. Defaults to 1e-2.
    substeps : int, optional
        Number of substeps per simulation step, i.e. how many times each solver integrates per `scene.step()`. More
        substeps buy accuracy and stability, at a runtime cost that grows linearly with the count in the worst case
        though sub-linearly in practice. Setting both this and a solver `dt` that implies a different count raises an
        exception. Defaults to 1.
    substeps_local : int, optional
        Number of substeps stored in GPU memory. Defaults to None. This is used for differentiable mode.
    gravity : tuple, optional
        Gravity force in N/kg. Defaults to (0.0, 0.0, -9.81).
    floor_height : float, optional
        Height of the floor in meters. Defaults to 0.0.
    requires_grad : bool, optional
        Whether to enable differentiable mode. Defaults to False.
    use_hydroelastic_contact : bool, optional
        Whether to use hydroelastic contact. Defaults to False.
    """

    dt: PositiveFloat = 1e-2
    substeps: PositiveInt = 1
    substeps_local: PositiveInt | None = None  # number of substeps stored in GPU memory
    gravity: Vec3FType = (0.0, 0.0, -9.81)
    floor_height: float = 0.0
    requires_grad: StrictBool = False

    _steps_local: int | None = PrivateAttr(default=None)

    @model_validator(mode="before")
    @classmethod
    def _resolve_substeps(cls, data: dict) -> dict:
        if data.get("substeps_local") is None:
            # use 1 to save gpu memory when not in differentiable mode
            data["substeps_local"] = data.get("substeps", 1) if data.get("requires_grad", False) else 1
        return data

    def model_post_init(self, context: Any) -> None:
        if self.requires_grad:
            if self.substeps_local % self.substeps != 0:
                gs.raise_exception("`substeps_local` must be divisible by `substeps` when `requires_grad` is True.")
            else:
                self._steps_local = int(self.substeps_local / self.substeps)
        else:
            self._steps_local = None


class BaseCouplerOptions(Options):
    """
    Base class for all coupler options.
    """


class LegacyCouplerOptions(BaseCouplerOptions):
    """
    Options configuring the inter-solver coupling.

    Parameters
    ----------
    rigid_mpm : bool, optional
        Whether to enable coupling between rigid and MPM solvers. Defaults to True.
    rigid_sph : bool, optional
        Whether to enable coupling between rigid and SPH solvers. Defaults to True.
    rigid_pbd : bool, optional
        Whether to enable coupling between rigid and PBD solvers. Defaults to True.
    rigid_fem : bool, optional
        Whether to enable coupling between rigid and FEM solvers. Defaults to True.
    mpm_sph : bool, optional
        Whether to enable coupling between MPM and SPH solvers. Defaults to True.
    mpm_pbd : bool, optional
        Whether to enable coupling between MPM and PBD solvers. Defaults to True.
    fem_mpm : bool, optional
        Whether to enable coupling between FEM and MPM solvers. Defaults to True.
    fem_sph : bool, optional
        Whether to enable coupling between FEM and SPH solvers. Defaults to True.
    """

    rigid_mpm: StrictBool = True
    rigid_sph: StrictBool = True
    rigid_pbd: StrictBool = True
    rigid_fem: StrictBool = True
    mpm_sph: StrictBool = True
    mpm_pbd: StrictBool = True
    fem_mpm: StrictBool = True
    fem_sph: StrictBool = True


class SAPCouplerOptions(BaseCouplerOptions):
    """
    Options configuring the inter-solver coupling for the Semi-Analytic Primal (SAP) contact solver used in Drake.

    Note
    ----
    Paper reference: https://arxiv.org/abs/2110.10107
    Drake reference: https://drake.mit.edu/release_notes/v1.5.0.html

    Parameters
    ----------
    n_sap_iterations : int, optional
        Number of iterations for the SAP solver. Defaults to 5.
    n_pcg_iterations : int, optional
        Number of iterations for the Preconditioned Conjugate Gradient solver. Defaults to 100.
    n_linesearch_iterations : int, optional
        Max number of iterations for the line search solver. Defaults to 10.
    sap_convergence_atol : float, optional
        Absolute tolerance for SAP convergence. Defaults to 1e-6.
    sap_convergence_rtol : float, optional
        Relative tolerance for SAP convergence. Defaults to 1e-5.
    sap_taud : float, optional
        Dissipation time scale for SAP. Defaults to 0.1.
    sap_beta : float, optional
        Normal regularization parameter for SAP. Defaults to 1.0.
    sap_sigma : float, optional
        Friction regularization parameter for SAP. Defaults to 1e-3.
    pcg_threshold : float, optional
        Threshold for the Preconditioned Conjugate Gradient solver. Defaults to 1e-6.
    linesearch_ftol : float, optional
        Line search sufficient value close to zero for exact linesearch. Defaults to 1e-6.
    linesearch_max_step_size : float, optional
        Maximum step size for exact linesearch. Defaults to 1.5.
    hydroelastic_stiffness : float, optional
        Stiffness for hydroelastic contact. Defaults to 1e8.
    point_contact_stiffness : float, optional
        Stiffness for point contact. Defaults to 1e8.
    fem_floor_contact_type : str, optional
        Type of contact against the floor. Defaults to "tet". Can be "tet", "vert", or "none".
        TET would be the default choice for most cases.
        VERT would be preferable when the mesh is very coarse, such as a single cube or a tetrahedron.
    enable_fem_self_tet_contact : bool, optional
        Whether to use tetrahedral based self-contact. Defaults to True.
    rigid_rigid_type : str, optional
        Type of contact between rigid bodies. Defaults to "tet". Can be "tet", "vert", or "none".
    rigid_floor_contact_type : str, optional
        Type of contact against the floor. Defaults to "tet". Can be "tet", "vert", or "none".
        Tet would be the default choice for most cases.
        Vert would be preferable when the mesh is very coarse, such as a single cube or a tetrahedron.
    enable_rigid_fem_contact : bool, optional
        Whether to enable coupling between rigid and FEM solvers. Defaults to True.
    """

    n_sap_iterations: PositiveInt = 5
    n_pcg_iterations: PositiveInt = 100
    n_linesearch_iterations: PositiveInt = 10
    sap_convergence_atol: PositiveFloat = 1e-6
    sap_convergence_rtol: PositiveFloat = 1e-5
    sap_taud: PositiveFloat = 0.1
    sap_beta: PositiveFloat = 1.0
    sap_sigma: PositiveFloat = 1e-3
    pcg_threshold: PositiveFloat = 1e-6
    linesearch_ftol: PositiveFloat = 1e-6
    linesearch_max_step_size: PositiveFloat = 1.5
    hydroelastic_stiffness: PositiveFloat = 1e8
    point_contact_stiffness: PositiveFloat = 1e8
    fem_floor_contact_type: Literal["tet", "vert", "none"] = "tet"
    enable_fem_self_tet_contact: StrictBool = True
    rigid_floor_contact_type: Literal["tet", "vert", "none"] = "tet"
    enable_rigid_fem_contact: StrictBool = True
    rigid_rigid_contact_type: Literal["tet", "vert", "none"] = "tet"


class IPCCouplerOptions(BaseCouplerOptions):
    """
    Options configuring the Incremental Potential Contact (IPC) coupler.

    Time step, gravity, and differentiable simulation mode are derived from ``SimOptions``
    (``dt``, ``gravity``, ``requires_grad``) and should not be set here.

    Parameters
    ----------
    Newton Solver Options
    ---------------------
    newton_max_iterations : int, optional
        Maximum iterations for Newton solver. Defaults to None (use libuipc default: 1024).
    newton_min_iterations : int, optional
        Minimum iterations for Newton solver. Defaults to None (use libuipc default: 1).
    newton_tolerance : float, optional
        Velocity tolerance for Newton solver convergence. Defaults to None (use libuipc default: 0.05).
    newton_ccd_tolerance : float, optional
        CCD (Continuous Collision Detection) tolerance for Newton solver. Defaults to None (use libuipc default: 1.0).
    newton_use_adaptive_tolerance : bool, optional
        Whether Newton solver should use adaptive tolerance. Defaults to None (use libuipc default: False).
    newton_translation_tolerance : float, optional
        Translation rate tolerance for Newton solver. Defaults to None (use libuipc default: 0.1).
    newton_semi_implicit_enable : bool, optional
        Whether to enable semi-implicit Newton solver mode. Defaults to None (use libuipc default: False).
    newton_semi_implicit_beta_tolerance : float, optional
        Beta tolerance for semi-implicit Newton solver. Defaults to None (use libuipc default: 1e-3).

    Line Search Options
    -------------------
    n_linesearch_iterations : int, optional
        Maximum iterations for line search. Defaults to None (use libuipc default: 8).
    linesearch_report_energy : bool, optional
        Whether to report energy during line search. Defaults to None (use libuipc default: False).

    Linear System Options
    ---------------------
    linear_system_solver : str, optional
        Linear system solver type. Options: 'linear_pcg', 'direct', etc. Defaults to None (use libuipc default: 'linear_pcg').
    linear_system_tolerance : float, optional
        Tolerance for linear system solver. Defaults to None (use libuipc default: 1e-3).

    Contact Options
    ---------------
    contact_enable : bool, optional
        Whether to enable contact detection. Defaults to None (use libuipc default: True).
    contact_d_hat : float, optional
        Contact distance threshold. Defaults to None (use libuipc default: 0.01).
    contact_friction_enable : bool, optional
        Whether to enable friction in contact. Defaults to None (use libuipc default: True).
    contact_resistance : float, optional
        Ground/default contact resistance/stiffness. It is used for ground contact pairs and
        as the per-entity fallback when a material does not define ``contact_resistance``.
        For ground pairs, it is combined with entity ``material.contact_resistance`` via
        geometric mean. Defaults to 1e9.
    contact_eps_velocity : float, optional
        Epsilon velocity for contact. Defaults to None (use libuipc default: 0.01).
    contact_constitution : str, optional
        Contact constitution model. Options: 'ipc', 'isometric'. Defaults to None (use libuipc default: 'ipc').

    Collision Detection Options
    ---------------------------
    collision_detection_method : str, optional
        Collision detection method. Options: 'linear_bvh', 'spatial_hash', etc. Defaults to None (use libuipc default: 'linear_bvh').

    CFL Options
    -----------
    cfl_enable : bool, optional
        Whether to enable CFL (Courant-Friedrichs-Lewy) condition. Defaults to None (use libuipc default: False).

    Sanity Check Options
    --------------------
    sanity_check_enable : bool, optional
        Whether to enable sanity checks. Defaults to None (use libuipc default: True).

    Genesis Coupling Options
    ------------------------
    constraint_strength_translation : float, optional
        Translation strength for IPC soft transform constraint coupling.
        Higher values create stiffer position coupling between Genesis rigid bodies and IPC ABD objects.
        Defaults to 100.0.
    constraint_strength_rotation : float, optional
        Rotation strength for IPC soft transform constraint coupling.
        Higher values create stiffer orientation coupling between Genesis rigid bodies and IPC ABD objects.
        Defaults to 100.0.
    enable_rigid_ground_contact : bool, optional
        Whether to enable ground contact in IPC system. When False, objects in IPC will not collide
        with the ground plane. Defaults to True.
    enable_rigid_rigid_contact : bool, optional
        Whether to enable contact detection between rigid bodies (ABD objects) in the IPC system.
        When False, only soft-soft and soft-rigid collisions are detected by IPC; rigid-rigid
        collisions within IPC are skipped. Defaults to True.
    two_way_coupling : bool, optional
        Whether to apply coupling forces/torques from IPC back to Genesis rigid bodies. Defaults to True.
    enable_rigid_dofs_sync : bool, optional
        Whether to synchronize the IPC reference DOF state with Genesis each step for
        external_articulation entities. When True, IPC gets tighter coupling with Genesis joint
        state but may amplify small divergences. When False, IPC uses its own DOF reference
        without per-step updates. Defaults to False.
    free_base_driven_by_ipc : bool, optional
        For external_articulation with non-fixed base: whether base link is fully driven by IPC physics.
        When False, base link uses SoftTransformConstraint controlled by Genesis. When True, base link
        is fully driven by IPC physics. Defaults to False.
    _show_ipc_gui : bool, optional
        [Dev/debug] Enable the libuipc built-in polyscope GUI viewer for inspecting the IPC scene.
        Defaults to False.
    """

    # Newton solver options (None = use libuipc default)
    newton_max_iterations: PositiveInt | None = None
    newton_min_iterations: PositiveInt | None = None
    newton_tolerance: PositiveFloat | None = None
    newton_ccd_tolerance: PositiveFloat | None = None
    newton_use_adaptive_tolerance: StrictBool | None = None
    newton_translation_tolerance: PositiveFloat | None = None
    newton_semi_implicit_enable: StrictBool | None = None
    newton_semi_implicit_beta_tolerance: PositiveFloat | None = None

    # Line search options (None = use libuipc default)
    n_linesearch_iterations: PositiveInt | None = None
    linesearch_report_energy: StrictBool | None = None

    # Linear system options (None = use libuipc default)
    linear_system_solver: Literal["linear_pcg", "direct"] | None = None
    linear_system_tolerance: PositiveFloat | None = None

    # Contact options
    contact_enable: StrictBool | None = None
    contact_d_hat: PositiveFloat | None = None
    contact_friction_enable: StrictBool | None = None
    contact_resistance: PositiveFloat = 1e9
    contact_eps_velocity: PositiveFloat | None = None
    contact_constitution: Literal["ipc", "isometric"] | None = None

    # Collision detection options
    collision_detection_method: Literal["linear_bvh", "spatial_hash"] | None = None

    # CFL options
    cfl_enable: StrictBool | None = None

    # Sanity check options
    sanity_check_enable: StrictBool | None = None

    # Genesis coupling options
    constraint_strength_translation: PositiveFloat = 100.0
    constraint_strength_rotation: PositiveFloat = 100.0
    enable_rigid_ground_contact: StrictBool = True
    enable_rigid_rigid_contact: StrictBool = True
    two_way_coupling: StrictBool = True
    enable_rigid_dofs_sync: StrictBool = False
    free_base_driven_by_ipc: StrictBool = False

    _show_ipc_gui: bool = PrivateAttr(default=False)

    def __init__(self, *, _show_ipc_gui: StrictBool = False, **data) -> None:
        super().__init__(**data)
        self._show_ipc_gui = bool(_show_ipc_gui)


############################ Solvers inside simulator ############################
"""
Parameters in these solver-specific options will override SimOptions if available.
"""


class TimeBasedMixin(Options):
    """
    A mixin adding the integration interval `dt` to the options of the solvers that integrate over one.

    Parameters
    ----------
    dt : float, optional
        The interval this solver integrates over, in seconds. It must divide `SimOptions.dt` an integer number of
        times, and that quotient is the number of substeps this solver runs per scene step. A shorter interval buys
        accuracy and stability where the motion is stiff or fast, at a runtime cost that grows linearly with the number
        of substeps in the worst case though sub-linearly in practice. Every active solver advances together, so an
        interval that disagrees with another solver's, or with `SimOptions.substeps`, raises an exception. If none, this
        solver integrates over the interval the other options settle on. Defaults to None.
    """

    dt: PositiveFloat | None = None


class GravityMixin(Options):
    """
    A mixin adding `gravity` to the options of the solvers that accelerate their bodies under it.

    Parameters
    ----------
    gravity : tuple, optional
        The acceleration applied to the bodies of this solver, in m/s^2. Each solver has its own, so one subsystem
        can be simulated weightless next to another. If none, the value carried by `SimOptions` is used. Defaults to
        None.
    """

    gravity: Vec3FType | None = None


class KinematicOptions(Options):
    """
    Options configuring the KinematicSolver (visualization-only solver).

    KinematicSolver is a lightweight solver for ghost/reference entities that only computes
    forward kinematics for visualization. No collision, physics integration, or constraint
    solving is performed.

    Parameters
    ----------
    batch_links_info : bool, optional
        Whether to batch link info. Automatically enabled for heterogeneous simulation. Defaults to False.
    batch_dofs_info : bool, optional
        Whether to batch DOF info. Automatically enabled for heterogeneous simulation. Defaults to False.
    IK_max_targets : int, optional
        Maximum number of IK targets. Increasing this doesn't affect IK solving speed, but will increase memory usage.
        Defaults to 6.
    """

    batch_links_info: StrictBool = False
    batch_joints_info: StrictBool = False
    batch_dofs_info: StrictBool = False
    IK_max_targets: PositiveInt = 6


class ToolOptions(TimeBasedMixin):
    """
    Options configuring the ToolSolver.

    Note
    ----
    ToolEntity is a simplified form of RigidEntity. It supports one way tool->other coupling, but has *no* internal dynamics and can only be created from a single mesh. This is a temporal workaround for differentiable rigid-soft interaction. This solver will be removed once differentiable mode is supported by the RigidSolver.

    Parameters
    ----------
    floor_height : float, optional
        Height of the floor in meters. Defaults to 0.0.
    """

    floor_height: float | None = None


class RigidOptions(GravityMixin, TimeBasedMixin):
    """
    Options configuring the RigidSolver.

    Parameters
    ----------
    enable_collision : bool, optional
        Whether to enable collision detection. Defaults to True.
    enable_joint_limit : bool, optional
        Whether to enable joint limit. Defaults to True.
    enable_self_collision : bool, optional
        Whether to enable self collision within each entity. Defaults to True.
    enable_neutral_collision : bool, optional
        Whether to enable self collision occurring in neutral configuration (qpos0) within each entity. Defaults to
        False.
    enable_adjacent_collision : bool, optional
        Whether to enable collision between successive parent-child body pairs within each entity. Defaults to False.
    disable_constraint: bool, optional
        Whether to disable all constraints. Defaults to False.
    max_collision_pairs : int, optional
        Maximum number of collision pairs. Defaults to 100.
    max_contacts : int, optional
        Maximum number of simultaneous contact points per environment that the constraint solver can handle, which
        determines the size of the contact constraint buffers (3 to 10 constraint rows per contact point depending on
        'friction_cone', 'enable_torsional_friction', and 'enable_rolling_friction'). Defaults to None.

        This limit applies to the final contact points after pruning, not to the candidate contact points that
        collision detection can emit (see 'max_collision_pairs'). Exceeding it at runtime halts the simulation with
        an error. None resolves it automatically: the pre-pruning worst case or, when contact pruning is enabled
        (see 'contact_pruning_tolerance'), 32 contact points per candidate link pair but no less than 512, whichever
        is smaller.
    integrator : gs.integrator, optional
        Integrator type. Current supported integrators are 'gs.integrator.Euler', 'gs.integrator.implicitfast' and
        'gs.integrator.approximate_implicitfast'. 'Euler' and 'implicitfast' are consistent with their Mujoco
        counterpart. 'approximate_implicitfast' is an even faster approximation of 'implicitfast', which avoid
        computing the inverse mass matrix twice by considering the first order correction terms of the implicit
        integration scheme systematically, including for computing the acceleration resulting from the constraints
        and external forces. Although this approximation is wrong in theory, it works reasonably well in practice.
        Defaults to 'approximate_implicitfast'.
    IK_max_targets : int, optional
        Maximum number of IK targets. Increasing this doesn't affect IK solving speed, but will increase memory usage.
        Defaults to 6.
    batch_links_info : bool, optional
        Whether the model parameters of a link, such as its mass or its inertia, are stored per environment rather
        than shared by the whole batch. Storing them per environment is what lets each environment carry its own
        values, which domain randomization needs, and what makes a per-environment write possible at all. It costs one
        copy of every link parameter per environment, in memory and in the bandwidth to read it, which slows down the
        memory-bound kernels. Automatically enabled for heterogeneous simulation. Defaults to False.
    batch_joints_info : bool, optional
        Whether the model parameters of a joint are stored per environment rather than shared by the whole batch,
        with the same tradeoff as `batch_links_info`. Automatically enabled for heterogeneous simulation. Defaults to
        False.
    batch_dofs_info : bool, optional
        Whether the model parameters of a degree of freedom are stored per environment rather than shared by the
        whole batch, with the same tradeoff as `batch_links_info`. Automatically enabled for heterogeneous simulation.
        Defaults to False.
    constraint_solver : gs.constraint_solver, optional
        Constraint solver type. Current supported constraint solvers are 'gs.constraint_solver.CG' (conjugate gradient)
        and 'gs.constraint_solver.Newton' (Newton's method). Defaults to 'Newton'.
    iterations : int, optional
        Maximum number of iterations of the constraint solver, which exits early once its tolerance is met. A batch of
        parallel environments waits for its slowest one on every step, so raising the bound buys accuracy on the steps
        whose hardest contacts never converge at the price of every such step. Defaults to 25.
    tolerance : float, optional
        Tolerance for the constraint solver. If None, resolved based on the floating-point precision selected via
        `gs.init(precision=...)`: 1e-5 for single precision ("32") and 1e-8 for double precision ("64"). Defaults
        to None.
    ls_iterations : int, optional
        Number of line search iterations for the constraint solver. Defaults to 50.
    ls_tolerance : float, optional
        Tolerance for the line search. Defaults to 1e-2.
    noslip_iterations : int, optional
        Number of iterations for the noslip solver. Defaults to 0 (disabled).
        noslip is a post-processing step after the main solver to suppress slip/drift.
        Recommended to set this value to 5 for manipulation tasks or when slip/drift is a big problem.
        This option should only be enabled if necessary because it is experimental and will slow down the simulation.
    noslip_tolerance : float, optional
        Tolerance for the noslip solver. Defaults to 1e-6.
    friction_cone : gs.friction_cone, optional
        Contact friction cone model, trading numerical robustness for physical accuracy. 'gs.friction_cone.pyramidal'
        (default) is robust and easy to solve; 'gs.friction_cone.elliptic' is the exact isotropic cone, harder to solve
        but paired with a high 'impratio' it holds resting stacks without slow tangential creep. See 'gs.friction_cone'
        for the description of each model. Unsupported with the noslip solver or differentiable simulation.
    contact_resolution : gs.contact_resolution, optional
        How a contact's normal force and friction force are resolved against each other.
        'gs.contact_resolution.signorini' bounds friction against the normal force the contact has developed, so sliding
        never inflates it and a body launched horizontally decelerates at mu * g instead of lifting off, at the cost of
        extra solver iterations. 'gs.contact_resolution.convex' poses the contact as a single convex program, which
        converges more predictably on stiff scenes but lets fast sliding buy normal force. See 'gs.contact_resolution'
        for the description of each model. Defaults to None, resolving to 'signorini' with the elliptic cone and the
        Newton solver, and 'convex' otherwise - the pyramidal cone's rows do not separate, and the conjugate gradient
        solver does not reach the fixed point. Always 'convex' when 'enable_mujoco_compatibility' is set.
    enable_torsional_friction : bool, optional
        Whether contacts also resist relative spin about their normal, with strength set per geometry by the material
        option 'friction_torsional' (see 'gs.materials.Rigid'). Enable it when spin resistance matters - a grasped
        object twisting in a gripper, a top spinning in place - motions a point contact transmits no torque against,
        so they persist indefinitely otherwise. The extra spin resistance slows down the constraint solve on every
        contact, including those where spin is irrelevant. Defaults to False.
    enable_rolling_friction : bool, optional
        Whether contacts also resist rolling, with strength set per geometry by the material option 'friction_rolling'
        (see 'gs.materials.Rigid'). Enable it when rolling resistance matters - a ball or wheel coasting to rest, a
        cylinder settling on a slope - motions a point contact otherwise never slows down. The extra rolling
        resistance slows down the constraint solve on every contact, more so than torsional friction (two extra axes),
        and requires 'enable_torsional_friction'. Defaults to False.
    impratio : float, optional
        Ratio of tangential (friction) to normal constraint impedance at contacts. Raising it above 1 stiffens
        friction so resting stacks and piles hold their pose under sustained shear, at the cost of a slower solve that
        turns numerically unstable once pushed too far - a stiffness-versus-stability tradeoff, so use the smallest
        value that holds the contacts. It matters mainly with the elliptic cone, which stiffens friction alone while
        leaving the normal contact response at its own impedance. Defaults to None, resolving to 100 with the elliptic
        cone (1 when 'enable_mujoco_compatibility' is set) and 1 otherwise.
    sparse_solve : bool, optional
        Whether to exploit sparsity (skyline-envelope Cholesky) in the constraint solver.

        Defaults to None, which resolves automatically: enabled on the CPU backend (and not under MuJoCo compatibility)
        when the scene has block structure - at least two DOF-carrying bodies or at least two free joints - so the
        Hessian band stays much tighter than its dimension. Never enabled on GPU, where the dense tiled factorization
        is faster. Set True or False to override the automatic choice; True is ignored with a warning on GPU.
    contact_resolve_time : float, optional
        Please note that this option will be deprecated in a future version. Use 'constraint_timeconst'
        instead.
    constraint_timeconst : float | None
        Time constant of the constraint response, in seconds, used for every geom that does not carry one of its
        own. The smaller it is, the stiffer the constraint, down to a floor of twice the integration interval, below
        which the solve becomes unstable. Set it to None to leave those geoms at that floor: as stiff as the timestep
        allows, and what a model authoring its own values expects, at the cost of contacts that respond more abruptly. This parameter is called
        'timeconst' in Mujoco (https://mujoco.readthedocs.io/en/latest/modeling.html#solver-parameters). Defaults to
        0.01.
    use_hibernation : bool, optional
        Whether to put bodies that have come to rest to sleep, so the solver skips them until they are disturbed. It
        quietly has no effect on a body that is differentiable, prunable, or under no-slip friction. Defaults to False.
    hibernation_thresh_vel : float, optional
        Velocity tolerance for hibernation, in meters per second: a body sleeps once its maximum DOF speed stays below
        this for a few consecutive steps, and a whole island sleeps once all its bodies are ready. Each rotational DOF
        is weighted by the body's swept radius, so the tolerance is a single linear speed that applies uniformly to
        translation and rotation. If None, it is set to 1e-4 when MuJoCo compatibility is enabled (matching MuJoCo's
        default) and 2e-3 otherwise. Defaults to None.
    max_dynamic_constraints : int, optional
        Maximum number of dynamic constraints (like suction cup). Defaults to 8.
    use_gjk_collision: bool, optional
        Whether to use GJK for collision detection instead of MPR. More stable but much slower. Defaults to
        `sim_options.requires_grad`.
    enable_contact_patch: bool, optional
        Whether to recover the full contact patch from the touching faces inside GJK, in a single detection pass,
        instead of through perturbed re-detections. The contact patch is cheaper and reports the exact contact
        polygon, but it is discouraged: it is less reliable than the perturbation-based detection, which is extremely
        robust at the cost of extra detection passes. Requires GJK collision detection, and raises otherwise. If
        None, it is enabled when MuJoCo compatibility is enabled together with GJK and multi-contact, and disabled
        otherwise. Defaults to None.
    broadphase_traversal : gs.broadphase_traversal, optional
        Broadphase traversal strategy. ``SAP`` (sweep-and-prune) or ``ALL_VS_ALL`` (parallel pair iteration). Defaults
        to ``None`` (auto: ``SAP`` on CPU or with heterogeneous entities, ``ALL_VS_ALL`` on GPU otherwise). See
        ``gs.broadphase_traversal`` for details on each strategy.

    Warning
    -------
    Hibernation hasn't been robustly tested and will be fully supported soon.
    """

    enable_collision: StrictBool = True
    enable_joint_limit: StrictBool = True
    enable_self_collision: StrictBool = True
    enable_neutral_collision: StrictBool = False
    enable_adjacent_collision: StrictBool = False
    disable_constraint: StrictBool = False
    max_collision_pairs: NonNegativeInt = 150
    max_contacts: NonNegativeInt | None = None
    multiplier_collision_broad_phase: PositiveInt = 8
    integrator: gs.integrator = gs.integrator.approximate_implicitfast
    IK_max_targets: PositiveInt = 6

    # batching info
    batch_links_info: StrictBool = False
    batch_joints_info: StrictBool = False
    batch_dofs_info: StrictBool = False

    # constraint solver
    constraint_solver: gs.constraint_solver = gs.constraint_solver.Newton
    iterations: PositiveInt = 25
    tolerance: PositiveFloat | None = None
    ls_iterations: PositiveInt = 50
    ls_tolerance: PositiveFloat = 1e-2
    noslip_iterations: NonNegativeInt = 0
    noslip_tolerance: PositiveFloat = 1e-6
    friction_cone: gs.friction_cone = gs.friction_cone.pyramidal
    contact_resolution: gs.contact_resolution | None = None
    enable_torsional_friction: StrictBool = False
    enable_rolling_friction: StrictBool = False
    impratio: PositiveFloat | None = None
    contact_pruning_tolerance: PositiveFloat | None = 0.02
    sparse_solve: StrictBool | None = None
    constraint_timeconst: PositiveFloat | None = 0.01
    box_box_detection: StrictBool = False

    # hibernation threshold
    use_hibernation: StrictBool = False
    hibernation_thresh_vel: PositiveFloat | None = None

    # for dynamic properties
    max_dynamic_constraints: NonNegativeInt = 8

    # Experimental options mainly intended for debug purpose and unit tests
    enable_multi_contact: StrictBool = True
    enable_mujoco_compatibility: StrictBool = False

    # GJK collision detection
    use_gjk_collision: StrictBool | None = None
    enable_contact_patch: StrictBool | None = None

    # broadphase configuration
    broadphase_traversal: gs.broadphase_traversal | None = None

    def __init__(self, *, contact_resolve_time: float | None = None, use_contact_island: bool | None = None, **data):
        super().__init__(**data)
        if contact_resolve_time is not None:
            gs.logger.warning("'contact_resolve_time' is deprecated. Use 'constraint_timeconst' instead.")
        if use_contact_island is not None:
            if not use_contact_island:
                gs.raise_exception(
                    "'use_contact_island=False' is not supported: the constraint solver always solves the contact "
                    "islands of the scene."
                )
            gs.logger.warning("'use_contact_island' is deprecated and has no effect.")

    def model_post_init(self, context):
        super().model_post_init(context)
        if self.contact_pruning_tolerance is not None and self.enable_mujoco_compatibility:
            if "contact_pruning_tolerance" in self.model_fields_set:
                gs.raise_exception(
                    "'contact_pruning_tolerance' is not supported when 'enable_mujoco_compatibility' is True"
                )
            # User did not explicitly request pruning, silently disable to guarantee mujoco compatibility
            self.contact_pruning_tolerance = None
        if self.friction_cone == gs.friction_cone.elliptic and self.noslip_iterations > 0:
            gs.raise_exception("The elliptic friction cone is not supported with the noslip solver.")
        if self.enable_rolling_friction and not self.enable_torsional_friction:
            gs.raise_exception("'enable_rolling_friction' requires 'enable_torsional_friction'.")


class MPMOptions(GravityMixin, TimeBasedMixin):
    """
    Options configuring the MPMSolver.

    Note
    ----
    MPM is a hybrid lagrangian-eulerian method for simulating soft materials. In the eulerian phase, it uses a grid representation. The `upper_bound` and `lower_bound` specify the simulation domain, but a safety padding will be added to the actual grid boundary. Therefore, the actual boundary could be slightly tighter than the specified one. Note that the size of the domain affects the performance of the simulation, hence you should set it as tight as possible.

    Parameters
    ----------
    particle_size : float, optional
        Particle diameter in meters. If not given, we will compute `particle_size` based on `grid_density`, where `particle_size` will be linearly proportional to the grid cell size. A reference value is `particle_size = 0.01` for `grid_density = 64`. Defaults to None.
    grid_density : float, optional
        Number of grid cells per meter. Defaults to 64.
    enable_CPIC : bool, optional
        Whether to enable CPIC (Compatible Particle-in-Cell) to support coupling with thin objects. Defaults to False.
    lower_bound : tuple, shape (3,), optional
        Lower bound of the simulation domain. Defaults to (-1.0, -1.0, 0.0).
    upper_bound : tuple, shape (3,), optional
        Upper bound of the simulation domain. Defaults to (1.0, 1.0, 1.0).
    use_sparse_grid : bool, optional
        This option is deprecated.
    leaf_block_size : int, optional
        This option is deprecated.
    """

    particle_size: PositiveFloat | None = None  # in meters. Will be computed automatically if it's None.
    grid_density: PositiveFloat = 64
    enable_CPIC: StrictBool = False

    # These will later be converted to discrete grid bound. The actual grid boundary could be slightly tighter.
    lower_bound: Vec3FType = (-1.0, -1.0, 0.0)
    upper_bound: Vec3FType = (1.0, 1.0, 1.0)

    def __init__(self, *, use_sparse_grid: bool = False, leaf_block_size: int = 8, **data):
        super().__init__(**data)
        if use_sparse_grid:
            gs.logger.warning("'use_sparse_grid' is deprecated and has no effect.")
        if leaf_block_size != 8:
            gs.logger.warning("'leaf_block_size' is deprecated and has no effect.")

    @model_validator(mode="before")
    @classmethod
    def _resolve_defaults(cls, data: dict) -> dict:
        if data.get("particle_size") is None:
            data["particle_size"] = 0.01 * 64.0 / data.get("grid_density", 64)
        return data

    def model_post_init(self, context: Any) -> None:
        if not np.all(np.array(self.upper_bound) > np.array(self.lower_bound)):
            gs.raise_exception("Invalid pair of upper_bound and lower_bound.")


class SPHOptions(GravityMixin, TimeBasedMixin):
    """
    Options configuring the SPHSolver.

    Note
    ----
    If spatial hashing parameters are not given, we will compute them automatically this way: For `hash_grid_cell_size`, we will set it to be the `support_radius`, which is essentially 2 * `particle_size`. For `hash_grid_res`, if a small bound is given, it's used for the hash grid; otherwise, we use a default value of a 150^3 cube. Any grid bigger than that will results in too many cells hence not ideal.

    Parameters
    ----------
    particle_size : float, optional
        Particle diameter in meters. Defaults to 0.02.
    pressure_solver : str, optional
        Pressure solver type. Current supported pressure solvers are 'WCSPH' and 'DFSPH'. Defaults to 'WCSPH'.
    lower_bound : tuple, shape (3,), optional
        Lower bound of the simulation domain. Defaults to (-100.0, -100.0, 0.0).
    upper_bound : tuple, shape (3,), optional
        Upper bound of the simulation domain. Defaults to (100.0, 100.0, 100.0).
    hash_grid_res : tuple, optional
        Size of the spatially-repetitive spatial hashing grid in meters. If none, it will be computed automatically. Defaults to None.
    hash_grid_cell_size : float, optional
        Size of the lattic cell of the spatial hashing grid in meters. This should be at least 2 * `particle_size`. If none, it will be computed automatically. Defaults to None.
    max_divergence_error : float, optional
        Maximum divergence error for DFSPH. Defaults to 0.1.
    max_density_error_percent : float, optional
        Maximum density error *percent* for DFSPH, so 0.1 means 0.1%. Defaults to 0.05.
    max_divergence_solver_iterations : int, optional
        Maximum number of iterations for the divergence solver. Defaults to 100.
    max_density_solver_iterations : int, optional
        Maximum number of iterations for the density solver. Defaults to 100.
    """

    particle_size: PositiveFloat = 0.02
    pressure_solver: Literal["WCSPH", "DFSPH"] = "WCSPH"

    lower_bound: Vec3FType = (-100.0, -100.0, 0.0)
    upper_bound: Vec3FType = (100.0, 100.0, 100.0)

    # spatial hashing
    hash_grid_res: Vec3FType | None = None  # size of the spatially-repetitive hash grid in meters
    hash_grid_cell_size: PositiveFloat | None = None  # size of the cubic cell in meters

    # DFSPH parameters
    max_divergence_error: PositiveFloat = 0.1
    max_density_error_percent: PositiveFloat = 0.05  # This is percent
    max_divergence_solver_iterations: PositiveInt = 100
    max_density_solver_iterations: PositiveInt = 100

    _support_radius: float = PrivateAttr(default=0.0)
    _hash_grid_res: np.ndarray = PrivateAttr(default=None)

    @model_validator(mode="before")
    @classmethod
    def _resolve_defaults(cls, data: dict) -> dict:
        particle_size = data.get("particle_size", 0.02)
        support_radius = 2 * particle_size
        if data.get("hash_grid_cell_size") is None:
            data["hash_grid_cell_size"] = support_radius
        return data

    def model_post_init(self, context: Any) -> None:
        if not np.all(np.array(self.upper_bound) > np.array(self.lower_bound)):
            gs.raise_exception("Invalid pair of upper_bound and lower_bound.")

        self._support_radius = 2 * self.particle_size

        if self.hash_grid_cell_size < self._support_radius:
            gs.raise_exception("`hash_grid_cell_size` should not be smaller than 2 * `particle_size`.")

        if self.hash_grid_res is None:
            max_hash_grid_res = np.ceil(
                (np.array(self.upper_bound) - np.array(self.lower_bound)) / self.hash_grid_cell_size
            )
            self._hash_grid_res = np.minimum(max_hash_grid_res, 150).astype(int).tolist()
        else:
            self._hash_grid_res = np.ceil(np.array(self.hash_grid_res) / self.hash_grid_cell_size).astype(int).tolist()


class PBDOptions(GravityMixin, TimeBasedMixin):
    """
    Options configuring the PBDSolver.

    Note
    ----
    If spatial hashing parameters are not given, we will compute them automatically this way: For `hash_grid_cell_size`, we will set it to be 1.25 * `particle_size`. For `hash_grid_res`, if a small bound is given, it's used for the hash grid; otherwise, we use a default value of a 150^3 cube. Any grid bigger than that will results in too many cells hence not ideal.

    Parameters
    ----------
    max_stretch_solver_iterations : int, optional
        Maximum number of iterations for the solving stretch constraints. Defaults to 4.
    max_bending_solver_iterations : int, optional
        Maximum number of iterations for the solving bending constraints. Defaults to 1.
    max_volume_solver_iterations : int, optional
        Maximum number of iterations for the solving volume constraints. Defaults to 1.
    max_density_solver_iterations : int, optional
        Maximum number of iterations for the solving density constraints. Defaults to 1.
    max_viscosity_solver_iterations : int, optional
        Maximum number of iterations for the solving viscosity constraints. Defaults to 1.
    particle_size : float, optional
        Particle diameter in meters. Defaults to 1e-2.
    hash_grid_res : tuple, optional
        Size of the spatially-repetitive spatial hashing grid in meters. If none, it will be computed automatically. Defaults to None.
    hash_grid_cell_size : float, optional
        Size of the lattic cell of the spatial hashing grid in meters. This should be at least 1.25 * `particle_size`. If none, it will be computed automatically. Defaults to None.
    lower_bound : tuple, shape (3,), optional
        Lower bound of the simulation domain. Defaults to (-100.0, -100.0, 0.0).
    upper_bound : tuple, shape (3,), optional
        Upper bound of the simulation domain. Defaults to (100.0, 100.0, 100.0).
    """

    # constraints solving iterations
    max_stretch_solver_iterations: PositiveInt = 4
    max_bending_solver_iterations: PositiveInt = 1
    max_volume_solver_iterations: PositiveInt = 1
    max_density_solver_iterations: PositiveInt = 1
    max_viscosity_solver_iterations: PositiveInt = 1

    # self collision
    particle_size: PositiveFloat = 1e-2

    # spatial hashing
    hash_grid_res: Vec3FType | None = None  # size of the spatially-repetitive hash grid in meters
    hash_grid_cell_size: PositiveFloat | None = None  # size of the cubic cell in meters

    lower_bound: Vec3FType = (-100.0, -100.0, 0.0)
    upper_bound: Vec3FType = (100.0, 100.0, 100.0)

    _hash_grid_res: np.ndarray = PrivateAttr(default=None)

    @model_validator(mode="before")
    @classmethod
    def _resolve_defaults(cls, data: dict) -> dict:
        particle_size = data.get("particle_size", 1e-2)
        # NOTE: 1.25 is a safety factor, as inside one single substep, multiple substages can change the position of
        # the particles but we only do spatial hashing once. The grid cell needs to be a bit bigger so that neighbours
        # are not missed.
        if data.get("hash_grid_cell_size") is None:
            data["hash_grid_cell_size"] = 1.25 * particle_size
        return data

    def model_post_init(self, context: Any) -> None:
        if not np.all(np.array(self.upper_bound) > np.array(self.lower_bound)):
            gs.raise_exception("Invalid pair of upper_bound and lower_bound.")

        if self.hash_grid_cell_size < 1.25 * self.particle_size:
            gs.raise_exception("`hash_grid_cell_size` should not be smaller than 1.25 * `particle_size`.")

        if self.hash_grid_res is None:
            max_hash_grid_res = np.ceil(
                (np.array(self.upper_bound) - np.array(self.lower_bound)) / self.hash_grid_cell_size
            )
            self._hash_grid_res = np.minimum(max_hash_grid_res, 150).astype(int).tolist()
        else:
            self._hash_grid_res = np.ceil(np.array(self.hash_grid_res) / self.hash_grid_cell_size).astype(int).tolist()


class FEMOptions(GravityMixin, TimeBasedMixin):
    """
    Options configuring the FEMSolver.

    Note
    ----
    - Damping coefficients are used to control the damping effect in the simulation.
    They are used in the Rayleigh Damping model, which is a common damping model in FEM simulations.
    Reference: https://doc.comsol.com/5.5/doc/com.comsol.help.sme/sme_ug_modeling.05.083.html
    - TODO Move it to material parameters in the future instead of solver options.

    Parameters
    ----------
    damping : float, optional
        Damping factor. Defaults to 0.0.
    floor_height : float, optional
        Height of the floor in meters. If none, it will inherit from `SimOptions`. Defaults to None.
    use_implicit_solver : bool, optional
        Whether to use the implicit solver. Defaults to False.
        Implicit solver is a more stable solver for FEM. It can be used with a large time step.
    n_newton_iterations : int, optional
        Maximum number of Newton iterations. Defaults to 1. Only used when `use_implicit_solver` is True.
    n_pcg_iterations : int, optional
        Maximum number of PCG iterations. Defaults to 500. Only used when `use_implicit_solver` is True.
    n_linesearch_iterations : int, optional
        Maximum number of line search iterations. Defaults to 0. Only used when `use_implicit_solver` is True.
    newton_dx_threshold : float, optional
        Threshold for the Newton solver. Defaults to 1e-6. Only used when `use_implicit_solver` is True.
    pcg_threshold : float, optional
        Threshold for the PCG solver. Defaults to 1e-6. Only used when `use_implicit_solver` is True.
    linesearch_c : float, optional
        Line search sufficient decrease parameter. Defaults to 1e-4. Only used when `use_implicit_solver` is True.
    linesearch_tau : float, optional
        Line search step size reduction factor. Defaults to 0.5. Only used when `use_implicit_solver` is True.
    damping_alpha : float, optional
        Rayleigh Damping factor for the implicit solver. Defaults to 0.5. Only used when `use_implicit_solver` is True.
    damping_beta : float, optional
        Rayleigh Damping factor for the implicit solver. Defaults to 5e-4. Only used when `use_implicit_solver` is True.
    enable_vertex_constraints : bool, optional
        Whether to enable vertex constraints. Defaults to False.
    """

    damping: NonNegativeFloat = 0.0
    floor_height: float | None = None
    use_implicit_solver: StrictBool = False
    n_newton_iterations: PositiveInt = 1
    n_pcg_iterations: PositiveInt = 500
    n_linesearch_iterations: NonNegativeInt = 0
    newton_dx_threshold: PositiveFloat = 1e-6
    pcg_threshold: PositiveFloat = 1e-6
    linesearch_c: PositiveFloat = 1e-4
    linesearch_tau: PositiveFloat = 0.5
    damping_alpha: NonNegativeFloat = 0.5
    damping_beta: NonNegativeFloat = 5e-4
    enable_vertex_constraints: StrictBool = False


class MochiOptions(GravityMixin, TimeBasedMixin):
    """
    Options configuring the MochiSolver.

    MochiSolver is a fully-implicit solver: at every substep it solves a single nonlinear system in which the inertia
    of every rigid body and a smooth signed-distance penalty contact model (regularized Coulomb friction, viscous
    normal damping) are assembled together. There is no separate collision response stage: contact is re-detected at
    every Newton iterate and enters the same residual and Hessian as the inertia, which is what keeps large time steps
    stable and is the prerequisite for coupling deformable bodies into the same system without a coupler.

    Note
    ----
    Double precision (`gs.init(precision="64")`) is recommended. The default contact stiffness of 1e9 Pa/m combined
    with a 1 mm activation threshold makes the Newton system ill-conditioned in single precision.

    Parameters
    ----------
    integrator : str, optional
        Time integration scheme: "backward_euler" (first order, strongly damped) or "bdf2" (second order, closer to
        energy-conserving; needs the previous two steps so the first step of a fresh or reset scene falls back to
        backward Euler). Defaults to "backward_euler".
    use_newton_euler_inertia : bool, optional
        Whether the rotational inertia enters as the Newton-Euler residual `I dw/dt + w x I w` instead of the
        variational merit of the rotation. The merit form derives from a potential and is what the line search
        monitors; the Newton-Euler form is exact for gyroscopic effects but its Hessian is approximate. Defaults to
        False.
    n_newton_iterations : int, optional
        Maximum number of Newton iterations per substep. Defaults to 4.
    newton_abs_tol : float, optional
        Absolute tolerance on the mass-weighted residual norm (unit acceleration under gravity gives a norm of order
        one). Defaults to 1e-3.
    newton_rel_tol : float, optional
        Relative tolerance on the mass-weighted residual norm with respect to its value at the first iteration.
        Defaults to 1e-6.
    explosion_control : bool, optional
        Whether a substep whose residual grows beyond `explosion_rel_tol` times its initial value or beyond
        `explosion_abs_tol` is flagged as diverged: the affected environment is reset to its previous pose with zero
        velocity and an error is raised at the next error check. Defaults to True.
    explosion_abs_tol : float, optional
        Absolute residual norm above which the solve is considered diverged. Defaults to 1e9.
    explosion_rel_tol : float, optional
        Residual growth factor above which the solve is considered diverged. Defaults to 1e4.
    linesearch_type : str, optional
        Step acceptance rule: "residual_norm" accepts the first trial whose residual norm does not exceed the current
        one, "armijo" requires a sufficient decrease of the incremental potential (costs an extra energy assembly per
        trial), "none" always takes the full Newton step. Defaults to "residual_norm".
    n_linesearch_iterations : int, optional
        Maximum number of step halvings per Newton iteration. The last trial is kept even if it did not improve.
        Defaults to 4.
    linesearch_alpha : float, optional
        Step size reduction factor between two line search trials. Defaults to 0.5.
    linesearch_wolfe1 : float, optional
        Sufficient decrease parameter of the Armijo rule. Defaults to 1e-4.
    linear_solver : str, optional
        Linear solver for the Newton system: "ldlt" (dense Cholesky of every simulation island, exact, cubic in the
        number of degrees of freedom of the island), "pcg" (block-Jacobi preconditioned conjugate gradient, linear per
        iteration), or "auto" (dense when the largest island of the environment has at most `dense_solver_max_dofs`
        degrees of freedom, PCG otherwise). Bodies coupled by the contact candidates of the step form an island.
        Defaults to "auto".
    dense_solver_max_dofs : int, optional
        Largest island solved with the dense solver under "auto". Defaults to 50.
    dense_matrix_max_dofs : int, optional
        Largest total number of degrees of freedom for which the dense matrix of the system is allocated (memory
        quadratic in this number per environment); beyond it every environment is solved with PCG. Defaults to 256.
    n_pcg_iterations : int, optional
        Maximum number of conjugate gradient iterations. If None, the number of degrees of freedom capped at 1000.
        Defaults to None.
    pcg_rel_tol : float, optional
        Relative tolerance of the conjugate gradient solve. Ignored under the "adaptive" tolerance strategy.
        Defaults to 1e-5.
    pcg_abs_tol : float, optional
        Absolute floor of the conjugate gradient stopping test, on the norm of the preconditioned residual (the
        residual scaled by the inverse of the Hessian diagonal, i.e. a displacement). A solve stops as soon as that
        norm drops below the floor, whatever the relative tolerance asks for; mochi's default. Set to 0 to disable.
        Defaults to 1e-9.
    linear_tolerance_strategy : str, optional
        How tightly each Newton step solves its linear system: "constant" always solves to `pcg_rel_tol`, so the
        accuracy of a substep is set by `pcg_rel_tol` and `n_newton_iterations` alone; "adaptive" starts each substep
        at a loose tolerance and tightens it as the nonlinear residual drops, spending far fewer conjugate gradient
        iterations per Newton step but leaving more truncation error in the step it takes, which can cost an extra
        Newton iteration on scenes that would otherwise converge in one. Prefer "adaptive" (mochi's default policy)
        when the conjugate gradient dominates the substep and a relative accuracy of order 1e-5 is enough, "constant"
        when accuracy per substep matters more than the cost of reaching it. Defaults to "adaptive".
    friction_model : str, optional
        Regularization of the Coulomb friction force around zero sliding velocity: "c1" has compact support (exact
        Coulomb beyond `friction_falloff_vel`), "cinf" is smooth everywhere (never exactly Coulomb, better
        conditioned). Defaults to "c1".
    use_fitted_friction_hessian : bool, optional
        Whether the friction Hessian uses a quadratic fit that is the same in every tangential direction. The exact
        Hessian converges faster close to the solution but can stall the Newton iterations at the stick-slip
        transition. Defaults to True.
    friction_with_collider_normal : bool, optional
        Whether the friction plane is defined by the collider's distance gradient (True) or by the colliding surface
        normal (False). Defaults to True.
    fade_friction : bool, optional
        Whether friction fades out as the colliding surface normal and the collider gradient become aligned, i.e. as a
        sample point passes through the far side of a thin collider. Defaults to True.
    implicit_normal_force_for_dissipation : bool, optional
        Whether friction and damping scale with the normal force evaluated at the current iterate instead of the one
        recovered at the start of the step. The implicit form is required for an accurate coefficient of restitution
        through normal damping; the explicit form is cheaper and smoother. Defaults to False.
    boundary_element_type : str, optional
        Quadrature rule placing contact sample points on the collision triangles: "P1Q1" (centroid), "P1Q3" (3 points
        per triangle, degree 2), "P1Q6" (6 points, degree 4). More points resolve contact patches better at a
        proportional cost. Defaults to "P1Q3".
    equality_stiffness : float, optional
        Stiffness of the penalty enforcing the equality constraints of the articulations (connect, weld and joint
        couplings; loop closures): the constraint violation is penalized by `0.5 * k * |c|^2`. Defaults to 1e6.
    equality_damping : float, optional
        Damping of the equality constraint penalty, `0.5 * (d / dt) * |c - c_prev|^2`. Defaults to 0.
    max_contact_pairs_per_env : int, optional
        Capacity of the list of (link, collider geom) pairs whose bounding boxes overlap within a substep. If None, the
        number of possible pairs. Defaults to None.
    broadphase_margin : float, optional
        Absolute padding of the per-step conservative bounding boxes in meters. Defaults to 0.01.
    spatial_hash_bins_per_item : int, optional
        Bins of the spatial hash that locates the collider spheres of shells and rods per inserted sphere, rounded
        up to a power of two; more bins shorten the chains a query walks at the cost of memory (4 bytes per bin per
        environment). The tetrahedra of solids use a bounding-box hierarchy instead. Defaults to 2.
    max_soft_hits_per_sample : int, optional
        Capacity of the list of contacts between deformable boundary samples and rigid colliders, per sample.
        Exceeding it halts the simulation with an error. Defaults to 2.
    max_deformable_collider_hits_per_query : int, optional
        Capacity of the list of contacts between sample points (rigid or deformable) and the tetrahedra of the
        deformable solid colliders, per sample point. Defaults to 2.
    max_point_cloud_hits_per_query : int, optional
        Capacity of the list of contacts between deformable samples and the collider spheres of shells and rods,
        per deformable sample (rigid samples get `max_soft_hits_per_sample` slots each). If None, 4 when a shell or
        rod has self-contact (a sample then sees the spheres of the opposing layer of its own body) and 2 otherwise;
        the capacity is a per-sample average over a shared list, and exceeding it halts with an error naming this
        option. Defaults to None.
    record_contacts : bool, optional
        Whether individual contact points can be read back through `entity.get_contacts()`; their buffers are
        allocated at the first readback. Defaults to True.
    step_kernel : str, optional
        How a step is executed: "monolith" runs the whole step of every environment in one kernel (one thread per
        environment, one launch per step, no host round trips), "pipeline" runs each stage as its own kernel with the
        host driving the loops, "graph" runs the step as one graph-launched kernel whose Newton, line-search and
        conjugate-gradient loops run on the device with the stages parallel over items and environments (one launch
        per step). "auto" picks the monolith on the CPU and, on the GPU, for rigid scenes of at most 64 degrees of
        freedom, the pipeline otherwise. The graph kernel is opt-in: on GPUs without device-side graph conditionals
        (before compute capability 9.0) the runtime replays its loop bodies from the host with one flag readback per
        round, which runs at the pipeline's speed, and the single module compiles several times slower than the
        pipeline's kernels (minutes for a cloth with self-contact and an arm). Defaults to "auto".
    graph_pcg_unroll : int, optional
        Conjugate-gradient iterations per round of the graph step kernel's inner loop (each round costs one flag
        readback on GPUs without device-side graph conditionals; each unrolled iteration is compiled once more).
        Defaults to 1.
    joint_limit_stiffness : float, optional
        Stiffness in N/m (N*m/rad) of the penalty holding revolute and prismatic joints inside their range. Joint
        limits are soft: the violation at rest is the limit torque divided by this stiffness. Higher values reduce the
        violation but stiffen the Newton system. Defaults to 1e4 (the original engine defaults to 100).
    joint_limit_damping : float, optional
        Damping in N*s/m (N*m*s/rad) of the joint limit penalty, resisting the velocity of the violation. Defaults to 0.
    batch_links_info : bool, optional
        Whether to batch link info. Defaults to False.
    batch_joints_info : bool, optional
        Whether to batch joint info. Defaults to False.
    batch_dofs_info : bool, optional
        Whether to batch DOF info. Defaults to False.
    IK_max_targets : int, optional
        Maximum number of simultaneous target links of an inverse-kinematics solve (the scratch buffers are quadratic
        in it and allocated at the first solve). Defaults to 6.
    """

    IK_max_targets: PositiveInt = 6
    integrator: Literal["backward_euler", "bdf2"] = "backward_euler"
    use_newton_euler_inertia: StrictBool = False
    n_newton_iterations: PositiveInt = 4
    newton_abs_tol: PositiveFloat = 1e-3
    newton_rel_tol: PositiveFloat = 1e-6
    explosion_control: StrictBool = True
    explosion_abs_tol: PositiveFloat = 1e9
    explosion_rel_tol: PositiveFloat = 1e4
    linesearch_type: Literal["none", "residual_norm", "armijo"] = "residual_norm"
    n_linesearch_iterations: NonNegativeInt = 4
    linesearch_alpha: PositiveFloat = 0.5
    linesearch_wolfe1: PositiveFloat = 1e-4
    linear_solver: Literal["auto", "ldlt", "pcg"] = "auto"
    dense_solver_max_dofs: PositiveInt = 50
    dense_matrix_max_dofs: PositiveInt = 256
    n_pcg_iterations: PositiveInt | None = None
    pcg_rel_tol: PositiveFloat = 1e-5
    pcg_abs_tol: NonNegativeFloat = 1e-9
    linear_tolerance_strategy: Literal["constant", "adaptive"] = "adaptive"
    friction_model: Literal["c1", "cinf"] = "c1"
    use_fitted_friction_hessian: StrictBool = True
    friction_with_collider_normal: StrictBool = True
    fade_friction: StrictBool = True
    implicit_normal_force_for_dissipation: StrictBool = False
    boundary_element_type: Literal["P1Q1", "P1Q3", "P1Q6"] = "P1Q3"
    equality_stiffness: PositiveFloat = 1e6
    equality_damping: NonNegativeFloat = 0.0
    max_contact_pairs_per_env: PositiveInt | None = None
    spatial_hash_bins_per_item: PositiveInt = 2
    max_soft_hits_per_sample: PositiveInt = 2
    max_deformable_collider_hits_per_query: PositiveInt = 2
    max_point_cloud_hits_per_query: PositiveInt | None = None
    broadphase_margin: NonNegativeFloat = 0.01
    record_contacts: StrictBool = True
    step_kernel: Literal["auto", "monolith", "pipeline", "graph"] = "auto"
    graph_pcg_unroll: PositiveInt = 1
    joint_limit_stiffness: PositiveFloat = 1e4
    joint_limit_damping: NonNegativeFloat = 0.0
    batch_links_info: StrictBool = False
    batch_joints_info: StrictBool = False
    batch_dofs_info: StrictBool = False

    def model_post_init(self, context: Any) -> None:
        if not (0.0 < self.linesearch_alpha < 1.0):
            gs.raise_exception("`linesearch_alpha` must be strictly between 0 and 1.")


class SFOptions(TimeBasedMixin):
    """
    Options configuring the SFSolver.
    """

    res: PositiveInt = 128
    solver_iters: PositiveInt = 500
    decay: PositiveFloat = 0.99

    T_low: float = 1.0
    T_high: float = 0.0

    inlet_pos: Vec3FType = (0.6, 0.0, 0.1)
    inlet_vel: Vec3FType = (0.0, 0.0, 1.0)
    inlet_quat: UnitVec4FType = (1.0, 0.0, 0.0, 0.0)
    inlet_s: PositiveFloat = 400.0

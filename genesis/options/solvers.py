from typing import Any, Literal

import numpy as np
from pydantic import PrivateAttr, StrictBool, model_validator

import genesis as gs
from genesis.typing import NonNegativeFloat, NonNegativeInt, PositiveFloat, PositiveInt, Vec3FType

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
    `SimOptions` specifies the global settings for the simulator. Some parameters exist both in `SimOptions` and
    `SolverOptions`. In this case, if such parameters are given in `SolverOptions`, it will override the one specified
    in `SimOptions` for this specific solver. For example, if `dt` is only given in `SimOptions`, it will be shared by
    all the solvers, while a solver given its own `dt` integrates over that interval instead, and the number of
    substeps of every solver follows from it.

    Parameters
    ----------
    dt : float, optional
        Time duration for each simulation step in seconds. Defaults to 1e-2.
    substeps : int, optional
        Number of substeps per simulation step, i.e. how many times each solver integrates per `scene.step()`. More
        substeps buy accuracy and stability, at a runtime cost that grows linearly with the count in the worst case
        though sub-linearly in practice. Setting both this and a solver `dt` that implies a different count raises an
        exception. Defaults to 1.
    gravity : tuple, optional
        Gravity force in N/kg. Defaults to (0.0, 0.0, -9.81).
    requires_grad : bool, optional
        Whether to enable differentiable mode. Defaults to False.
    use_hydroelastic_contact : bool, optional
        Whether to use hydroelastic contact. Defaults to False.
    """

    dt: PositiveFloat = 1e-2
    substeps: PositiveInt = 1
    gravity: Vec3FType = (0.0, 0.0, -9.81)
    requires_grad: StrictBool = False


class BaseCouplerOptions(Options):
    """
    Base class for all coupler options.
    """

    pass


class LegacyCouplerOptions(BaseCouplerOptions):
    """
    Options configuring the inter-solver coupling.

    Parameters
    ----------
    rigid_mpm : bool, optional
        Whether to enable coupling between rigid and MPM solvers. Defaults to True.
    rigid_sph : bool, optional
        Whether to enable coupling between rigid and SPH solvers. Defaults to True. The liquid sampled inside a coupled
        rigid body, or within half a particle size of its surface, is removed when the scene is built, with a warning.
        Moving liquid inside a coupled body afterwards, for instance by setting its position, is unsupported.
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
        Whether to enable coupling between FEM and SPH solvers. Defaults to True. The liquid sampled inside a coupled
        FEM body, or within half a particle size of its surface, is removed when the scene is built, with a warning.
        Moving liquid inside a coupled body afterwards, for instance by setting its position, is unsupported.
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
        Maximum number of Newton iterations of the SAP solver in each substep, which stops earlier once it converges. A
        smaller value bounds the cost of a substep, at the risk of returning velocities that do not satisfy the contacts
        and that inject energy into resting contacts. Defaults to 100.
    n_pcg_iterations : int, optional
        Number of iterations for the Preconditioned Conjugate Gradient solver. Defaults to 100.
    n_linesearch_iterations : int, optional
        Maximum number of iterations of the exact line search of each Newton iteration, which stops earlier once it
        meets its tolerance, usually within a few iterations. A smaller value bounds the cost of a Newton iteration, at
        the risk of taking a step short of the minimum of the cost along the Newton direction, which can stall the
        Newton iterations. Defaults to 100.
    sap_convergence_atol : float, optional
        Absolute tolerance of the SAP solve on the norm of the gradient of its cost, in momentum units, added to the
        relative one. A larger value stops the solves of scenes moving slowly earlier, at the cost of accuracy near
        rest. Defaults to None, which leaves the relative tolerance alone.
    sap_convergence_rtol : float, optional
        Tolerance of the SAP solve on the norm of the gradient of its cost relative to the momentum of the scene. A
        smaller value solves the contacts more accurately, so that resting contacts slip less within a step, at the cost
        of more Newton iterations per substep. Defaults to None, which takes 1e-4 in double precision and 1e-3 in single
        precision: single precision with its default runs fastest, while double precision solves about ten times more
        accurately for a quarter to two thirds more Newton iterations. A solve also stops once its iterations no longer
        decrease its cost, which bounds the accuracy any value can reach to the precision of the simulation.
    sap_taud : float, optional
        Dissipation time scale for SAP. Defaults to 0.1.
    sap_beta : float, optional
        Stiffness of the contacts of rigid bodies, as the time a resting body takes to settle on what it rests on, in
        substeps. A smaller value makes the contacts stiffer, so that bodies sink less into each other and rock less
        under a push, at the cost of a less robust solve that may take more iterations. A larger value softens them:
        the penetration grows with the square of this value. On the vertex floor (`rigid_floor_contact_type="vert"`),
        the fixed stiffness of each vertex also bounds how stiff the contact of a heavy body gets. Defaults to 0.125,
        which keeps resting bodies less than a tenth of a millimeter deep at usual substeps.
    sap_sigma : float, optional
        Friction regularization parameter for SAP. Defaults to 1e-3.
    enable_lagged_friction : bool, optional
        Whether the friction of each contact scales with its normal force of the previous substep rather than with the
        normal force solved within the substep. Lagged friction keeps sliding bodies on the surface they slide on,
        decelerating at the rate of Coulomb friction, where SAP friction lifts them off it at a speed growing with
        their sliding speed. In exchange, the friction lags one substep behind changes of the normal force, so that it
        grips harder on impact, and sticking contacts under load creep at a speed up to `stiction_tolerance`. Defaults
        to True.
    stiction_tolerance : float, optional
        Sliding speed below which lagged friction holds a contact as sticking, in m/s. A smaller value lets sticking
        contacts creep less, at the cost of more Newton iterations per substep. Ignored without lagged friction.
        Defaults to 1e-4.
    pcg_threshold : float, optional
        Threshold of the Preconditioned Conjugate Gradient solver on the squared residual of each Newton step, relative
        to its initial value. A smaller value takes fewer and more accurate Newton steps, each costing more conjugate
        gradient iterations, for the same converged result. A value below the square of the precision of the simulation
        acts as that square. Defaults to 1e-2.
    linesearch_ftol : float, optional
        Line search sufficient value close to zero for exact linesearch. Defaults to 1e-6.
    linesearch_max_step_size : float, optional
        Maximum step size for exact linesearch. Defaults to 1.5.
    hydroelastic_stiffness : float, optional
        Stiffness for hydroelastic contact. Defaults to 1e8.
    point_contact_stiffness : float, optional
        Stiffness for point contact. Defaults to 1e8.
    fem_floor_contact_type : str, optional
        Type of contact against the floor. Defaults to "tet". Can be "tet", "vert", or "none". The floor is the rigid
        ground plane of the scene, which must be a single fixed plane facing up. A scene without one has no floor.
        TET would be the default choice for most cases.
        VERT would be preferable when the mesh is very coarse, such as a single cube or a tetrahedron.
    enable_fem_self_tet_contact : bool, optional
        Whether to use tetrahedral based self-contact. Defaults to True.
    rigid_rigid_type : str, optional
        Type of contact between rigid bodies. Defaults to "tet". Can be "tet", "vert", or "none".
    rigid_floor_contact_type : str, optional
        Type of contact against the floor, as for `fem_floor_contact_type`. Defaults to "tet". Can be "tet", "vert",
        or "none".
        Tet would be the default choice for most cases.
        Vert would be preferable when the mesh is very coarse, such as a single cube or a tetrahedron.
    enable_rigid_fem_contact : bool, optional
        Whether to enable coupling between rigid and FEM solvers. Defaults to True.
    max_collision_pairs : int, optional
        Maximum number of collision pairs of each kind of contact in each environment: the pairs of overlapping bounding
        boxes the contact detection tests, and the pairs of elements, or of an element and the floor, that may touch. A
        larger value covers larger or more finely meshed contact patches, at the cost of memory proportional to it and
        to the number of environments, capped by the most pairs the scene can produce. Exceeding it at runtime halts
        the simulation with an error naming this option. Defaults to 32768.
    max_contacts : int, optional
        Maximum number of contact points of each kind of contact in each environment. A larger value covers larger or
        more finely meshed contact patches, at the cost of memory proportional to it and to the number of environments,
        capped by the most contact points the scene can produce. Exceeding it at runtime halts the simulation with an
        error naming this option. Defaults to 4096.
    """

    n_sap_iterations: PositiveInt = 100
    n_pcg_iterations: PositiveInt = 100
    n_linesearch_iterations: PositiveInt = 100
    sap_convergence_atol: PositiveFloat | None = None
    sap_convergence_rtol: PositiveFloat | None = None
    sap_taud: PositiveFloat = 0.1
    sap_beta: PositiveFloat = 0.125
    sap_sigma: PositiveFloat = 1e-3
    enable_lagged_friction: StrictBool = True
    stiction_tolerance: PositiveFloat = 1e-4
    pcg_threshold: PositiveFloat = 1e-2
    linesearch_ftol: PositiveFloat = 1e-6
    linesearch_max_step_size: PositiveFloat = 1.5
    hydroelastic_stiffness: PositiveFloat = 1e8
    point_contact_stiffness: PositiveFloat = 1e8
    fem_floor_contact_type: Literal["tet", "vert", "none"] = "tet"
    enable_fem_self_tet_contact: StrictBool = True
    rigid_floor_contact_type: Literal["tet", "vert", "none"] = "tet"
    enable_rigid_fem_contact: StrictBool = True
    rigid_rigid_contact_type: Literal["tet", "vert", "none"] = "tet"
    max_collision_pairs: NonNegativeInt = 32768
    max_contacts: NonNegativeInt = 4096


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
        Maximum number of dynamic constraints (like suction cup) per environment. Adding one past it raises. Defaults
        to 8.
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
    use_implicit_solver : bool, optional
        Whether to use the implicit solver, which stays stable at a time step far larger than the explicit solver
        allows, and resolves the contact of the particles with the rigid entities within the same solve as their motion
        (see NewtonOptions). Each of its substeps costs several iterations, which the larger time step pays back.
        Differentiable simulation, CPIC, custom materials and the contact with FEM, PBD and SPH entities require the
        explicit solver. The contact with moving rigid entities requires the pyramidal friction cone and the Newton
        constraint solver of the rigid solver, with hibernation and noslip iterations off (see RigidOptions). A muscle
        only contracts under the implicit solver: a negative actuation acts as zero. Defaults to False.
    wall_timeconst : float, optional
        Time constant in seconds over which the walls of the domain push back the particles that penetrate them, raised
        to twice the substep when smaller. A smaller value gives stiffer walls that the particles penetrate less, but
        more iterations of the solve near the walls. Defaults to 0.01. Only used when `use_implicit_solver` is True.
    liquid_smoothing_velocity : float, optional
        Velocity in m/s under which the resistance of a liquid to compression fades in, its stiffness rising from zero
        to its bulk modulus over the compression the liquid undergoes at that velocity over a substep. A larger value
        lets the solve of a liquid near rest take fewer iterations, but softens the liquid more. If None, resolved based
        on the tolerance of the solve: 'NewtonOptions.velocity_tol', the smallest velocity the solve resolves. Defaults
        to None. Only used when `use_implicit_solver` is True.
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

    use_implicit_solver: StrictBool = False
    wall_timeconst: PositiveFloat = 1e-2
    liquid_smoothing_velocity: NonNegativeFloat | None = None

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
        if self.use_implicit_solver and self.enable_CPIC:
            gs.raise_exception("'enable_CPIC' requires the explicit solver, got 'use_implicit_solver=True'.")


class SPHOptions(GravityMixin, TimeBasedMixin):
    """
    Options configuring the SPHSolver, which keeps the liquid incompressible with divergence-free smoothed particle
    hydrodynamics (DFSPH).

    Note
    ----
    If spatial hashing parameters are not given, we will compute them automatically this way: For `hash_grid_cell_size`, we will set it to be the `support_radius`, which is essentially 2 * `particle_size`. For `hash_grid_res`, if a small bound is given, it's used for the hash grid; otherwise, we use a default value of a 150^3 cube. Any grid bigger than that will results in too many cells hence not ideal.

    Parameters
    ----------
    particle_size : float, optional
        Particle diameter in meters. Defaults to 0.02.
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
        Average density error that DFSPH allows at the end of a substep, in percent of the rest density, so 0.1 means
        0.1%. A lower value keeps the liquid closer to incompressible and its pressure steadier from one substep to the
        next, at the cost of more solver iterations per substep. A larger value saves iterations, but lets the pressure
        swing between substeps, which can shake a liquid at rest and the bodies it carries. Defaults to 0.01.
    max_divergence_solver_iterations : int, optional
        Maximum number of iterations for the divergence solver. Defaults to 100.
    max_density_solver_iterations : int, optional
        Maximum number of iterations for the density solver. Defaults to 100.
    """

    particle_size: PositiveFloat = 0.02

    lower_bound: Vec3FType = (-100.0, -100.0, 0.0)
    upper_bound: Vec3FType = (100.0, 100.0, 100.0)

    # spatial hashing
    hash_grid_res: Vec3FType | None = None  # size of the spatially-repetitive hash grid in meters
    hash_grid_cell_size: PositiveFloat | None = None  # size of the cubic cell in meters

    # DFSPH parameters
    max_divergence_error: PositiveFloat = 0.1
    max_density_error_percent: PositiveFloat = 0.01
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
        Threshold for the PCG solver. Defaults to 1e-6. Only used when `use_implicit_solver` is True, without
        `requires_grad`: a differentiable scene solves to the precision of the simulation instead, which its gradients
        need, at the cost of more iterations.
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


class NewtonOptions(Options):
    """
    Options of the iterative solve of the implicit material point method (MPM) solver and of the cloth, which moves
    the particles or the cloth over every substep together with the rigid entities in contact with them.

    The implicit MPM solver is enabled by `MPMOptions.use_implicit_solver`, and the cloth is solved this way whenever
    the scene holds a `gs.materials.FEM.Cloth` entity.

    Parameters
    ----------
    velocity_tol : float, optional
        Convergence tolerance of the solve, in m/s: the solve of a substep stops once an iteration changes the velocity
        of every MPM particle, cloth vertex and rigid entity in contact with them by less than this value. A smaller value
        gives a more accurate motion at the cost of more iterations per substep, and a larger one a faster but less
        accurate one. Defaults to 5e-2.
    max_iterations : int, optional
        Iterations of a substep at most. A solve reaching it stops short of 'velocity_tol' and logs a warning. Defaults
        to 1024.
    line_search_max_iterations : int, optional
        Trial steps of an iteration at most, each one shorter than the previous, the last one being kept. More trials
        let an iteration progress where the motion is hard to resolve, at the cost of more work per iteration. Defaults
        to 12.
    line_search_shrink : float, optional
        Factor of the step of an iteration between two trials, strictly between 0 and 1. A smaller factor reaches an
        acceptable step in fewer trials, while a larger one keeps a longer step at the cost of more trials. Defaults to
        0.5.
    pcg_max_iterations : int, optional
        Iterations at most of the inner linear solve of an iteration. A larger cap lets a stiff scene converge, at the
        cost of a longer worst-case substep. Defaults to 1024.
    pcg_tol : float, optional
        Relative accuracy of the inner linear solve of an iteration. A looser value makes every iteration cheaper, the
        next iterations correcting its inexact result, at the cost of more iterations. Defaults to 1e-4.
    failure_check_interval : int, optional
        Substeps between two checks of whether a solve reached 'max_iterations', which logs a warning. A smaller
        interval reports it sooner, at the cost of a slower simulation. Defaults to 32.
    """

    velocity_tol: PositiveFloat = 5e-2
    max_iterations: PositiveInt = 1024
    line_search_max_iterations: PositiveInt = 12
    line_search_shrink: PositiveFloat = 0.5
    pcg_max_iterations: PositiveInt = 1024
    pcg_tol: PositiveFloat = 1e-4
    failure_check_interval: PositiveInt = 32

    def model_post_init(self, context: Any) -> None:
        if self.line_search_shrink >= 1.0:
            gs.raise_exception(f"'line_search_shrink' must be below 1, got {self.line_search_shrink}.")


class IPCOptions(Options):
    """
    Options of the incremental potential contact (IPC) of the cloth, which keeps it from crossing the surfaces it
    touches.

    The contact acts within a band of width 'd_hat' beyond the contact thickness of the cloth (see
    `gs.materials.FEM.Cloth`), its repulsion growing without bound as the gap closes, so the cloth never reaches the
    thickness. The contact of the cloth with itself and with other cloths is reliable in double precision,
    `gs.init(precision='64')`: in single precision, cloths at rest on one another creep and their motion depends on the
    order in which their contacts are found.

    Parameters
    ----------
    d_hat : float, optional
        Width of the contact band, in m. A wider band starts pushing the cloth away earlier, which spreads the contact
        over more substeps and eases the solve, at the cost of a visible gap: a cloth at rest hovers within the band.
        Defaults to 1e-2.
    friction_eps_v : float, optional
        Sliding velocity below which the friction of a contact eases into static friction, in m/s. A smaller value holds
        a resting cloth more firmly in place at the cost of a stiffer, harder solve, while a larger one lets it creep.
        Defaults to 1e-2.
    ccd_bound : str, optional
        How a step of the solve finds how far the cloth may move before touching: 'directional' bounds the motion by how
        fast the gap between two surfaces closes, 'additive' by how far each surface moves in any direction. Both stop
        at the same contact, 'directional' in fewer passes for cloth sliding over a surface. Defaults to 'directional'.
    ccd_partition : bool, optional
        Whether every group of cloths in contact with one another moves by a step of its own, as far as its own contacts
        allow. Off, a contact anywhere in a scene shortens the step of every cloth in it, which slows the solve of
        scenes with many separate cloths, without changing the motion it converges to. Defaults to True.
    adaptive_kappa_mode : str, optional
        Whether the stiffness of the contact grows where a cloth presses so hard into it that the gap nearly closes:
        'per-vertex' stiffens the contacts of the vertices concerned, and relaxes them once they stay calm, 'off' keeps
        the stiffness of the cloth material. A stiffer contact holds a heavy load at a wider gap, which keeps the solve
        fast, at the cost of a contact force that depends on the past of the load. Defaults to 'per-vertex'.
    adaptive_kappa_tick : str, optional
        When the stiffness of the contact grows: 'newton' within the solve of a substep, as soon as a gap closes,
        'frame' once per substep. Growing within the solve answers a sudden load at once, at the cost of a few more
        iterations of that substep. Defaults to 'newton'.
    adaptive_kappa_gap_ratio : float, optional
        Gap below which a contact stiffens, as a fraction of 'd_hat' beyond the contact thickness. A larger ratio keeps
        cloths farther apart under load, at the cost of stiffer contacts. Defaults to 0.01.
    adaptive_kappa_grow : float, optional
        Factor of the stiffness each time a contact stiffens, above 1. A larger factor opens a closing gap in fewer
        steps, at the cost of a harder solve right after. Defaults to 2.
    adaptive_kappa_max_scale : float, optional
        Largest factor of the stiffness of the cloth material that a contact reaches. A larger one holds heavier loads,
        at the cost of a harder solve under them. Defaults to 128.
    adaptive_kappa_calm_time : float, optional
        Time a stiffened contact holds its stiffness once its gap reopens, in s, before it relaxes. A longer time holds
        a repeated load steadily, at the cost of a stiff contact that outlives it. Defaults to 0.3.
    adaptive_kappa_relax_time : float, optional
        Time constant of the relaxation of a stiffened contact back to the stiffness of the cloth material, in s. A
        longer time relaxes more smoothly and more slowly. Defaults to 0.4.
    adaptive_kappa_hysteresis : float, optional
        Factor of 'adaptive_kappa_gap_ratio' up to which a contact keeps its stiffness without relaxing, at least 1. A
        larger factor avoids stiffening and relaxing in turn under a varying load. Defaults to 2.
    init_collision_pair_capacity : int, optional
        Pairs of surfaces near enough to touch that the scene holds at first. More pairs grow the storage as the scene
        needs, at the cost of repeating the search that found them, so a larger value saves that work in crowded scenes
        at the cost of memory. Defaults to 1000.
    capacity_grow_factor : float, optional
        Factor of the storage of the pairs of surfaces over the pairs found when it grows, above 1. A larger factor
        grows less often at the cost of memory. Defaults to 1.2.
    intersection_check : bool, optional
        Whether building the scene checks that no cloth crosses itself or another cloth, and raises an exception naming
        the crossing edge and triangle. The solve cannot separate cloths that start crossed, so the check finds the
        mistake at once, at the cost of a slower build. Defaults to False.
    vertex_constraint_strength_ratio : float, optional
        Stiffness of the pull of a hard vertex constraint of the cloth toward its target, as a ratio of the mass of the
        vertex (see `FEMEntity.set_vertex_constraints`). The pull moves the vertex within the solve, so that the contact
        stops it short of crossing a surface. A larger ratio reaches a moving target in fewer iterations, at the cost of
        a stiffer solve while the vertex is held off its target. Defaults to 1e4.
    vertex_constraint_fix_tol : float, optional
        Distance from its target within which a hard vertex constraint of the cloth holds the vertex fixed, as a
        fraction of `NewtonOptions.velocity_tol` times the substep. A held vertex leaves the solve, which speeds it up.
        A larger fraction holds the vertex sooner, at the cost of a larger offset from its target. Defaults to 0.2.
    """

    d_hat: PositiveFloat = 1e-2
    friction_eps_v: PositiveFloat = 1e-2
    ccd_bound: Literal["directional", "additive"] = "directional"
    ccd_partition: StrictBool = True
    adaptive_kappa_mode: Literal["per-vertex", "off"] = "per-vertex"
    adaptive_kappa_tick: Literal["newton", "frame"] = "newton"
    adaptive_kappa_gap_ratio: PositiveFloat = 0.01
    adaptive_kappa_grow: PositiveFloat = 2.0
    adaptive_kappa_max_scale: PositiveFloat = 128.0
    adaptive_kappa_calm_time: NonNegativeFloat = 0.3
    adaptive_kappa_relax_time: PositiveFloat = 0.4
    adaptive_kappa_hysteresis: PositiveFloat = 2.0
    init_collision_pair_capacity: PositiveInt = 1000
    capacity_grow_factor: PositiveFloat = 1.2
    intersection_check: StrictBool = False
    vertex_constraint_strength_ratio: PositiveFloat = 1e4
    vertex_constraint_fix_tol: PositiveFloat = 0.2

    def model_post_init(self, context: Any) -> None:
        if not self.adaptive_kappa_grow > 1.0:
            gs.raise_exception(f"'adaptive_kappa_grow' must be above 1, got {self.adaptive_kappa_grow}.")
        if not self.capacity_grow_factor > 1.0:
            gs.raise_exception(f"'capacity_grow_factor' must be above 1, got {self.capacity_grow_factor}.")
        if self.adaptive_kappa_hysteresis < 1.0:
            gs.raise_exception(f"'adaptive_kappa_hysteresis' must be at least 1, got {self.adaptive_kappa_hysteresis}.")

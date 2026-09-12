import enum

URDF_FORMAT = ".urdf"
XACRO_FORMAT = ".xacro"
MJCF_FORMAT = ".xml"
# Root tags identifying the format of inline XML content passed as 'FileMorph.file'.
XML_ROOT_TAG_TO_FORMAT = {"mujoco": MJCF_FORMAT, "robot": URDF_FORMAT}
GLTF_FORMATS = (".glb", ".gltf")
MESH_FORMATS = (".obj", ".stl", ".dae", *GLTF_FORMATS)
USD_FORMATS = (".usd", ".usda", ".usdc", ".usdz")

# dynamic loading
ACTIVE = 1
INACTIVE = 0


class IntEnum(enum.IntEnum):
    def __repr__(self):
        return f"<gs.{self.__class__.__name__}.{self.name}: {self.value}>"

    def __format__(self, format_spec):
        return f"<{self.name}: {self.value}>"


# geom type in rigid solver
class GEOM_TYPE(IntEnum):
    # Beware PLANE must be the first geometry type as this is assumed by MPR collision detection.
    PLANE = 0
    SPHERE = 1
    ELLIPSOID = 2
    CYLINDER = 3
    CAPSULE = 4
    BOX = 5
    MESH = 6
    TERRAIN = 7


# joint type in rigid solver, ranked by number of dofs
class JOINT_TYPE(IntEnum):
    FIXED = 0
    REVOLUTE = 1
    PRISMATIC = 2
    SPHERICAL = 3
    FREE = 4


class EQUALITY_TYPE(IntEnum):
    CONNECT = 0
    WELD = 1
    JOINT = 2


class CTRL_MODE(IntEnum):
    POSITION = 0
    VELOCITY = 1
    FORCE = 2


######### User accessible constants do not capitalize #########
# rigid solver intergrator
class integrator(IntEnum):
    Euler = 0
    implicitfast = 1
    approximate_implicitfast = 2


# rigid solver constraint solver
class constraint_solver(IntEnum):
    CG = 0
    Newton = 1


# rigid solver contact friction cone
class friction_cone(IntEnum):
    """
    Contact friction cone model, trading numerical robustness for physical accuracy.

    'pyramidal' (the default) approximates the friction cone by a pyramid: robust and easy to solve, but the
    approximation makes friction anisotropic (the effective limit depends on the sliding direction). 'elliptic' is
    the exact cone: friction is isotropic and bounded by its true Euclidean limit sqrt(f_t1^2 + f_t2^2) <= mu * f_n
    in every direction, and with a high 'impratio' it holds resting stacks without the slow tangential creep of
    regularized friction, in return for being harder to solve and more sensitive numerically. Prefer pyramidal for
    robustness; choose elliptic when isotropic friction or firm static friction matters - e.g. objects that must stay
    put at rest instead of slowly creeping.
    """

    pyramidal = 0
    elliptic = 1


# rigid solver contact resolution
class contact_resolution(IntEnum):
    """
    How a contact's normal force and friction force are resolved against each other.

    'convex' poses the whole contact as a single smooth convex cost and lets the solver trade the normal residual
    against the tangential one. Because the friction limit mu * f_n bounds the pair jointly, a contact sliding fast
    enough that its friction rows demand more force than the cone allows can be answered by raising f_n instead: a body
    launched horizontally then lifts off a flat floor, by more the faster it slides. In exchange the whole problem stays
    one convex program, which converges predictably on stiff articulated chains and high mass ratios.

    'signorini' bounds friction against the normal force the contact has actually developed, so that force is set by the
    contact's own normal state rather than by tangential demand, and sliding can never inflate it - a sliding body
    decelerates at mu * g and stays down at any speed. Contacts are resolved by successive approximation, costing extra
    solver iterations and giving up the single-convex-program guarantee. Prefer it whenever sliding contact matters;
    choose 'convex' for parity with engines built on that formulation, or if a stiff scene converges better under it.

    'signorini' requires the elliptic friction cone, whose rows separate into a normal row and a friction disc - the
    pyramidal cone mixes the normal direction into every row and admits no such split - and the Newton constraint
    solver, the only one that reaches the fixed point of the resulting successive approximation. It implements the
    Coulomb complementarity problem eq. (C.22) of Alexis Duburcq, "Learning and Optimization of the Locomotion with an
    Exoskeleton for Paraplegic People", PhD thesis, Universite Paris Sciences et Lettres, 2022 (HAL tel-04166955),
    Appendix C, whose Signorini condition is what forbids the normal force from absorbing tangential demand.
    """

    convex = 0
    signorini = 1


# rigid solver broadphase traversal strategy
class broadphase_traversal(IntEnum):
    """
    Strategy for broad-phase collision detection in the rigid solver.

    Broad-phase quickly eliminates geometry pairs that cannot collide before the more expensive narrow-phase runs.

    At init time, geometry pairs that can never collide are filtered out (same-link, fixed-vs-fixed, contype/conaffinity
    mismatch, etc.), producing a list of *valid pairs*.  The number of valid pairs can be up to O(n_geoms^2) but is
    typically much smaller after filtering.  The two strategies differ in how they search these valid pairs each step:

    Attributes
    ----------
    SAP : int
        Sweep-and-prune.  Sorts geometry AABBs along one axis (O(n_geoms log n_geoms)) then only checks pairs that
        overlap on that axis.  The sort and sweep are single-threaded, which utilizes GPU cores poorly. However the
        cost per step is only O(n_geoms log n_geoms + k) where k is the number of axis-overlapping pairs — typically
        much less than the full set of valid pairs.
    ALL_VS_ALL : int
        Checks every valid pair every step (AABB overlap test), dispatching them in parallel across GPU threads.  Cost
        per step is O(n_valid_pairs) which is efficient on GPU when the pair count is moderate, but becomes expensive
        in scenes with many geometries since the valid pair count grows quadratically. Does not support hibernation or
        heterogeneous entities at this time.

    Notes
    -----
    ``RigidOptions.broadphase_traversal`` defaults to ``None``, which lets the solver choose automatically:

    - **CPU backend** → ``SAP`` (sequential sweep is efficient on CPU).
    - **GPU backend** → ``ALL_VS_ALL`` (parallel pair checking is faster).
    - **GPU with hibernation or heterogeneous entities** → ``SAP``
      (``ALL_VS_ALL`` is not compatible with these features).
    """

    SAP = 0
    ALL_VS_ALL = 1


class link_ref_frame(IntEnum):
    """
    Reference frame at which a per-link quantity is expressed.

    Each member fixes an origin, about which spatial quantities are translated, together with an orientation, by which
    quantities given in local coordinates are rotated. 'root_COM' is the center of mass of the whole kinematic tree the
    link belongs to, with world-aligned axes; the solver stores link velocities and generalized forces there natively,
    so it is the only frame reached without a moment arm. 'link_COM' is the center of mass of the link, with the axes of
    its inertial frame. 'link_origin' is the origin of the link, with the axes of the link frame.
    """

    root_COM = 0
    link_COM = 1
    link_origin = 2


# backend
class backend(IntEnum):
    cpu = 0
    gpu = 1
    cuda = 2
    amdgpu = 3
    metal = 4

    def __format__(self, format_spec):
        return f"gs.{self.name}"


# image types for visualization
class IMAGE_TYPE(IntEnum):
    RGB = 0
    DEPTH = 1
    SEGMENTATION = 2
    NORMAL = 3

    def __format__(self, format_spec):
        return self.name


# parallelize
class PARA_LEVEL(IntEnum):
    NEVER = 0  # when using cpu
    PARTIAL = 1  # when using gpu for non-batched scene
    ALL = 2  # when using gpu for batched scene

"""
Constants and enums for the collider module.
"""

from enum import IntEnum


class CCD_ALGORITHM_CODE(IntEnum):
    """Convex collision detection algorithm codes."""

    # Our MPR (with SDF)
    MPR = 0
    # MuJoCo MPR
    MJ_MPR = 1
    # Our GJK
    GJK = 2
    # MuJoCo GJK
    MJ_GJK = 3


class RETURN_CODE(IntEnum):
    """
    Return codes for the general subroutines used in GJK and EPA algorithms.
    """

    SUCCESS = 0
    FAIL = 1


class GJK_RETURN_CODE(IntEnum):
    """
    Return codes for the GJK algorithm.
    """

    SEPARATED = 0
    INTERSECT = 1
    NUM_ERROR = 2


class GJK_STAGE(IntEnum):
    """
    Stage of the safe Gilbert-Johnson-Keerthi (GJK) algorithm that the next support query of a pair serves.
    """

    # One of the 4 fixed-direction vertices of the initial tetrahedron
    INITIAL_VERTEX = 0
    # Replacement of an initial vertex that would make an invalid simplex, along the normal of the triangle so far
    VERTEX_SEARCH = 1
    # Replacement of the vertex of the simplex facing the origin the least
    ITERATE = 2
    # No query left
    DONE = 3


class PORTAL_STATUS(IntEnum):
    """
    What the penetration depth of a contact is worth, and whether the portal behind it may be reused (perturbation
    reconstruction, EPA seeding). Each value names the depth rather than the portal's health, since that is what every
    consumer decides on.
    """

    # No portal exists: the contact is computed in closed form (plane, capsule, sphere) or by the MPR centres fallback,
    # so there is nothing for a refinement to improve. Also what an unwritten slot reads as.
    NONE = 0
    # MPR hit its iteration cap, so the depth means nothing
    UNCONVERGED = 1
    # The origin projects outside the converged portal, or the portal is degenerate, so the depth is read off the plane
    # of a triangle that is not the exact contact face
    APPROXIMATE = 2
    # The origin projects inside the converged portal, so the depth is exact (Theorem 4.2). The only status whose portal
    # may be reused.
    EXACT = 3


class MPR_STAGE(IntEnum):
    """
    Stage of the Minkowski portal refinement (MPR) that the next support query of a pair serves.
    """

    # Probe of the three axes when the two centres coincide within the absolute length resolution of the solver, which
    # leaves the ray undefined. The axis of largest Minkowski support extent gives the portal the most room to enclose
    # the origin.
    AXIS_PROBE = 0
    # First portal vertex, along the ray from the centre of the Minkowski difference toward the origin
    PORTAL_VERTEX_1 = 1
    # Second portal vertex, normal to the ray and the first vertex
    PORTAL_VERTEX_2 = 2
    # Third portal vertex, searched until the portal faces the origin
    PORTAL_VERTEX_3 = 3
    # Refinement of the portal until it encloses the origin
    REFINE = 4
    # Refinement of the enclosing portal until it reaches the surface of the Minkowski difference
    PENETRATION = 5
    # No query left
    DONE = 6


class MPR_SUPPORT_TYPE(IntEnum):
    """
    Feature of the Minkowski difference supporting the contact found by the Minkowski portal refinement (MPR).
    """

    # No contact: a portal vertex falls short of the origin, the search of the third portal vertex gives up, or the
    # refinement finds no portal enclosing the origin
    NONE = -1
    # Contact read off the portal refined up to the surface of the Minkowski difference
    FACE = 0
    # The first portal vertex vanishes, the geometries touching at a single point
    VERTEX = 1
    # The first portal vertex is collinear with the centre of the Minkowski difference, which leaves a segment as portal
    EDGE = 2


class CONTACT_ORDER(IntEnum):
    """
    Order in which the contact pruning sorts the contacts of an environment or of one of its link-pair buckets.
    """

    # By link pair, with the link indices of each contact, to group them into buckets
    LINK_PAIR = 0
    # By the intrinsic data of each contact: geom pair, frame-local position, then penetration
    INTRINSIC = 1
    # By the projection of each contact on the plane of its bucket, lexicographically
    POSITION = 2


class MULTICONTACT_SLOT(IntEnum):
    """
    What a candidate slot of the split multi-contact pass holds, which decides how the gather accepts it.
    """

    # No contact, either a detection that found none or one that did not run
    EMPTY = 0
    # The first contact of the pair, accepted as it is
    BASE = 1
    # A contact of a perturbed detection, kept unless it repeats a contact already kept
    PERTURBED = 2


class MULTICONTACT_ENTRY(IntEnum):
    """
    Where contact 0 of a pair queued for the split multi-contact pass stands, which decides what each stage of the pass
    runs for it (see func_narrowphase_multicontact).
    """

    # Stored in slot 0 without Gilbert-Johnson-Keerthi (GJK): the Minkowski portal refinement (MPR) seed of the contact0
    # kernel, or the analytic detection that the first stage of the pass runs in place of GJK for a pair it resolves
    # analytically
    SEED = 0
    # Left to the first stage of the pass, which detects it with Gilbert-Johnson-Keerthi (GJK)
    GJK = 1
    # Detected with Gilbert-Johnson-Keerthi (GJK) by the first stage and stored in slot 0
    DETECTED = 2
    # Nothing left to detect: Gilbert-Johnson-Keerthi (GJK) found no contact, a manifold that holds every contact, or a
    # pair without multi-contact
    DONE = 3


class EPA_POLY_INIT_RETURN_CODE(IntEnum):
    """
    Return codes for the EPA polytope initialization.
    """

    SUCCESS = 0
    P2_NONCONVEX = 1
    P2_FALLBACK3 = 2
    P3_BAD_NORMAL = 3
    P3_INVALID_V4 = 4
    P3_INVALID_V5 = 5
    P3_MISSING_ORIGIN = 6
    P3_ORIGIN_ON_FACE = 7
    P4_FALLBACK3 = 9

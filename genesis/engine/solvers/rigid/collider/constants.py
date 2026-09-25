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


class PORTAL_STATUS(IntEnum):
    """
    What the penetration depth of a contact is worth, and whether the portal behind it may be reused (perturbation
    reconstruction, EPA seeding). Each value names the depth rather than the portal's health, since that is what every
    consumer decides on.

    NONE: no portal exists - the contact is computed in closed form (plane, capsule, sphere) or by the MPR centres
    fallback, so there is nothing for a refinement to improve. Also what an unwritten slot reads as.
    UNCONVERGED: MPR hit its iteration cap, so the depth means nothing.
    EXTRAPOLATED: the origin's projection falls so far beyond the portal triangle that the depth is read off an
    extrapolation of its plane, or the triangle is degenerate. Untrustworthy.
    LOWER_BOUND: the origin's projection falls just outside the triangle, so the depth is a valid lower bound of the
    true one (Theorem 4.3), but the portal is not the exact contact face.
    EXACT: the origin projects inside the converged portal, so the depth is exact (Theorem 4.2). The only status whose
    portal may be reused.
    """

    NONE = 0
    UNCONVERGED = 1
    EXTRAPOLATED = 2
    LOWER_BOUND = 3
    EXACT = 4


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

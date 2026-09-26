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
    """

    # No portal exists: the contact is computed in closed form (plane, capsule, sphere) or by the MPR centres fallback,
    # so there is nothing for a refinement to improve. Also what an unwritten slot reads as.
    NONE = 0
    # MPR hit its iteration cap, so the depth means nothing
    UNCONVERGED = 1
    # The origin's projection falls so far beyond the portal triangle that the depth is read off an extrapolation of its
    # plane, or the triangle is degenerate. Untrustworthy.
    EXTRAPOLATED = 2
    # The origin's projection falls just outside the triangle, so the depth is a valid lower bound of the true one
    # (Theorem 4.3), but the portal is not the exact contact face
    LOWER_BOUND = 3
    # The origin projects inside the converged portal, so the depth is exact (Theorem 4.2). The only status whose portal
    # may be reused.
    EXACT = 4


class MULTICONTACT_SLOT(IntEnum):
    """
    What a candidate slot of the split multi-contact pass holds, which decides how the gather accepts it.
    """

    # No contact, either a detection that found none or one that did not run
    EMPTY = 0
    # The first contact of the pair, accepted as it is
    BASE = 1
    # A perturbed contact whose recovered penetration is exact, discarded as soon as it is non-positive
    EXACT = 2
    # A perturbed contact whose recovered penetration is first-order, kept within a negative tolerance
    APPROX = 3


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

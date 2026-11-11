"""Consistent incremental potential contact (IPC) barrier on squared distances, offset by a contact thickness.

The barrier of a pair at squared distance D is b(r) = kappa (r - 1)^2 ln(r)^2 with r = (D - xi^2) / (d_hat^2 + 2 d_hat xi),
xi the contact thickness of the pair and d_hat the activation distance beyond it. It is active for xi^2 < D < (d_hat +
xi)^2, where 0 < r < 1, and vanishes with its first derivative at the outer bound. The gradient and the Hessian with
respect to the stencil vertices come from the squared distance D and its derivatives, the Hessian being projected onto
the positive semi-definite cone.
"""

import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu

from .distance import func_edge_edge_cross_norm2, func_edge_edge_flagged_distance2_hessian


@qd.func
def func_pair_barrier_parameters(
    d_hat_a: float, d_hat_b: float, thicknesses_a, thicknesses_b, area_weight_a: float, area_weight_b: float
):
    """Compute the activation distance, the contact thickness and the area weight of a pair of primitives.

    Each side passes the activation distance shared by its vertices, the thicknesses of its vertices as a vector, and
    its area weight (the area a vertex or an edge stands for on the surface). The pair takes the mean activation
    distance, the sum of the largest thickness of each side, and the smaller area weight times d_hat / 4. The energy of
    the pair is its barrier times the area weight times the squared time step. A point against a half-plane takes the
    parameters of the point, with the area weight times d_hat.
    """
    d_hat = (d_hat_a + d_hat_b) / 2.0
    thickness = thicknesses_a.max() + thicknesses_b.max()
    area_weight = qd.min(area_weight_a, area_weight_b) * d_hat / 4.0
    return d_hat, thickness, area_weight


@qd.func
def func_barrier(distance2: float, d_hat: float, thickness: float, kappa: float):
    """Compute the barrier and its first and second derivatives with respect to the squared distance.

    The squared distance must lie strictly inside the active band (see the module docstring). Returns the energy, the
    first derivative and the second derivative.
    """
    d_tilde = distance2 - thickness * thickness
    d_hat_tilde2 = d_hat * d_hat + 2.0 * d_hat * thickness
    ratio = d_tilde / d_hat_tilde2
    log_ratio = qd.log(ratio)
    ratio_minus_one = ratio - 1.0
    log_term = log_ratio + (ratio - 1.0) / ratio
    energy = kappa * ratio_minus_one * ratio_minus_one * log_ratio * log_ratio
    first = 2.0 * kappa * (ratio - 1.0) * log_ratio * log_term / d_hat_tilde2
    second = (2.0 * kappa * (log_term * log_term + (ratio * ratio - 1.0) * log_ratio / (ratio * ratio))) / (
        d_hat_tilde2 * d_hat_tilde2
    )
    return energy, first, second


@qd.func
def func_barrier_normal_force(distance2: float, d_hat: float, thickness: float, kappa: float) -> float:
    """Compute the normal force of a barrier, the opposite of its derivative with respect to the unsquared distance.

    It is positive inside the active band, and scales the friction of the pair.
    """
    _, first, _ = func_barrier(distance2, d_hat, thickness, kappa)
    return -first * 2.0 * qd.sqrt(distance2)


@qd.func
def func_edge_edge_mollifier_threshold(
    ea0_rest: qd.types.vector(3),
    ea1_rest: qd.types.vector(3),
    eb0_rest: qd.types.vector(3),
    eb1_rest: qd.types.vector(3),
    mollifier_ratio: float,
) -> float:
    """Compute the squared cross norm below which two edges are mollified, from their rest positions.

    It is 'mollifier_ratio' times the product of the squared rest lengths of the two edges. Below it, the edge-edge
    barrier is multiplied by a mollifier that ramps it down to zero as the edges become parallel, where the distance
    between their lines and the classification of their closest features are ill-conditioned.
    """
    return mollifier_ratio * (ea0_rest - ea1_rest).norm_sqr() * (eb0_rest - eb1_rest).norm_sqr()


@qd.func
def func_mollified_barrier(
    distance2: float, cross_norm2: float, mollifier_threshold: float, d_hat: float, thickness: float, kappa: float
) -> float:
    """Compute the barrier of an edge-edge pair multiplied by its mollifier.

    The mollifier is m(c) = 2 c / e - (c / e)^2 of the squared cross norm c of the edges (see
    func_edge_edge_cross_norm2) and of the mollifier threshold e (see func_edge_edge_mollifier_threshold), which ramps
    from 0 for parallel edges to 1 at the threshold with a zero slope there.
    """
    energy, _, _ = func_barrier(distance2, d_hat, thickness, kappa)
    mollifier = -(cross_norm2 * cross_norm2) / (mollifier_threshold * mollifier_threshold) + (
        2.0 * cross_norm2 / mollifier_threshold
    )
    return mollifier * energy


@qd.func
def _func_barrier_hessian_eigenvalue(
    ratio: float, d_tilde: float, d_hat_tilde2: float, kappa: float, hessian_clamp_ratio: float
) -> float:
    """Compute the curvature of the barrier along the unsquared distance, normalized by the barrier width.

    It is the eigenvalue of the barrier Hessian along the gradient of the distance in the deformation-gradient form of
    the barrier. Below 'hessian_clamp_ratio', the ratio is frozen at that value, so that the stiffness of a pair close to
    contact stops growing and does not collapse the Newton steps of the other pairs.
    """
    clamped_ratio = qd.select(d_tilde < hessian_clamp_ratio * d_hat_tilde2, hessian_clamp_ratio, ratio)
    log_ratio = qd.log(clamped_ratio)
    log_term = log_ratio + (clamped_ratio - 1.0) / clamped_ratio
    return (
        8.0
        * kappa
        * (
            clamped_ratio * log_term * log_term
            + (clamped_ratio * clamped_ratio - 1.0) * log_ratio / clamped_ratio
            + (clamped_ratio - 1.0) * log_ratio * log_term * 0.5
        )
    )


@qd.func
def func_barrier_gradient_hessian(
    distance2: float,
    distance2_gradient,
    d_hat: float,
    thickness: float,
    kappa: float,
    hessian_clamp_ratio: float,
):
    """Compute the gradient and the positive semi-definite Hessian of the barrier with respect to the stencil vertices.

    'distance2_gradient' is the gradient of the squared distance over the stencil, of any size. The gradient is the
    first derivative of the barrier times the gradient of the squared distance. The Hessian is the rank-one outer
    product of the gradient of the squared distance, scaled by the curvature of the barrier along the unsquared
    distance, frozen at its value for the ratio 'hessian_clamp_ratio' below it. Returns the gradient vector and the
    Hessian matrix.
    """
    d_tilde = distance2 - thickness * thickness
    d_hat_tilde2 = d_hat * d_hat + 2.0 * d_hat * thickness
    ratio = d_tilde / d_hat_tilde2
    _, first, _ = func_barrier(distance2, d_hat, thickness, kappa)
    eigenvalue = _func_barrier_hessian_eigenvalue(ratio, d_tilde, d_hat_tilde2, kappa, hessian_clamp_ratio)
    gradient = first * distance2_gradient
    hessian = (eigenvalue / (4.0 * d_tilde * d_hat_tilde2)) * distance2_gradient.outer_product(distance2_gradient)
    return gradient, hessian


@qd.func
def func_mollified_barrier_gradient_hessian(
    flag: int,
    ea0: qd.types.vector(3),
    ea1: qd.types.vector(3),
    eb0: qd.types.vector(3),
    eb1: qd.types.vector(3),
    distance2: float,
    distance2_gradient: qd.types.vector(12),
    mollifier_threshold: float,
    d_hat: float,
    thickness: float,
    kappa: float,
    rank_tolerance: float,
    eigen_tolerance: float,
    max_sweeps: int,
):
    """Compute the gradient and the positive semi-definite Hessian of the mollified edge-edge barrier m(c) b(D).

    The barrier is the one of func_mollified_barrier. 'flag' is the distance flag of the edges (see
    func_edge_edge_distance_flag), 'distance2' and 'distance2_gradient' the squared distance of their closest features
    and its gradient over (ea0, ea1, eb0, eb1). Both derivatives keep every term of the product rule,
        G = b grad m + m b' grad D,
        H = b Hess m + b' (grad m grad D^T + grad D grad m^T) + m (b'' grad D grad D^T + b' Hess D),
    because toward parallel edges the curvature is dominated by b m' Hess c, the resistance of the barrier to the
    relative rotation of the edges. The Hessian is projected onto the positive semi-definite cone (see qd_project_psd,
    which takes 'rank_tolerance', 'eigen_tolerance' and 'max_sweeps'), then symmetrized. Returns the gradient vector and
    the Hessian matrix.
    """
    PROJECTION_MAX_RANK = qd.static(6)
    cross_norm2, cross_norm2_gradient, cross_norm2_hessian = func_edge_edge_cross_norm2(ea0, ea1, eb0, eb1)
    distance2_hessian = func_edge_edge_flagged_distance2_hessian(flag, ea0, ea1, eb0, eb1)
    energy, first, second = func_barrier(distance2, d_hat, thickness, kappa)

    inverse_threshold = 1.0 / mollifier_threshold
    mollifier = -(cross_norm2 * cross_norm2) * inverse_threshold * inverse_threshold + (
        2.0 * cross_norm2 * inverse_threshold
    )
    mollifier_first = -2.0 * cross_norm2 * inverse_threshold * inverse_threshold + 2.0 * inverse_threshold
    mollifier_second = -2.0 * inverse_threshold * inverse_threshold
    mollifier_gradient = mollifier_first * cross_norm2_gradient
    mollifier_hessian = (
        mollifier_second * cross_norm2_gradient.outer_product(cross_norm2_gradient)
        + mollifier_first * cross_norm2_hessian
    )

    gradient = energy * mollifier_gradient + (mollifier * first) * distance2_gradient
    coupling = mollifier_gradient.outer_product(distance2_gradient)
    hessian = (
        energy * mollifier_hessian
        + first * (coupling + coupling.transpose())
        + mollifier * (second * distance2_gradient.outer_product(distance2_gradient) + first * distance2_hessian)
    )
    hessian = gu.qd_project_psd(hessian, PROJECTION_MAX_RANK, rank_tolerance, eigen_tolerance, max_sweeps)
    # The scatter of the Hessian into the system keeps one triangle of the off-diagonal blocks but the whole diagonal
    # ones, so that a projection symmetric only up to rounding would add an asymmetric part
    return gradient, 0.5 * (hessian + hessian.transpose())


@qd.func
def func_point_halfplane_barrier_gradient_hessian(
    distance: float, normal: qd.types.vector(3), d_hat: float, thickness: float, kappa: float
):
    """Compute the gradient and the positive semi-definite Hessian of the barrier of a point against a half-plane.

    'distance' is the signed distance normal . (x - plane_point) of the point to the plane, and 'normal' the unit normal
    of the half-plane. The Hessian is the exact one along the normal, its coefficient being clamped at zero. Returns the
    gradient vector and the Hessian matrix of the barrier with respect to the point.
    """
    distance2 = distance * distance
    _, first, second = func_barrier(distance2, d_hat, thickness, kappa)
    coefficient = qd.max(4.0 * distance2 * second + 2.0 * first, 0.0)
    return (first * 2.0 * distance) * normal, coefficient * normal.outer_product(normal)

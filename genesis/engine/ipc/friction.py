"""Lagged smoothed Coulomb friction of an IPC contact stencil.

The friction of a pair is mu * lambda * f0(|u|), lambda the normal force of its barrier at the start of the time step
and u the tangential relative displacement of its closest points over the step. The tangent plane and the closest points
are frozen at the start of the step (the lagged positions): a frame gives an orthonormal 3x2 tangent basis T and the
weights w_k of the stencil vertices in the relative displacement, which is u = T^T sum_k w_k dx_k for the displacements
dx_k of the vertices from their lagged positions. The smoothing f0 is C1, linear in |u| beyond the smoothing length
eps_vh (sliding) and quadratic below it (sticking).
"""

import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu


@qd.func
def func_point_triangle_friction_frame(
    p: qd.types.vector(3), t0: qd.types.vector(3), t1: qd.types.vector(3), t2: qd.types.vector(3)
):
    """Compute the tangent basis of a point-triangle pair and the weights of its vertices (p, t0, t1, t2).

    The basis spans the plane of the triangle, its first axis along the edge t0-t1. The weights are 1 for the point and
    minus the barycentric coordinates of its projection on the plane of the triangle for the triangle vertices.
    """
    edge0 = t1 - t0
    edge1 = t2 - t0
    basis = qd.Matrix.cols([edge0.normalized(), edge0.cross(edge1).cross(edge0).normalized()])
    edges = qd.Matrix.rows([edge0, edge1])
    coords = (edges @ edges.transpose()).inverse() @ (edges @ (p - t0))
    return basis, qd.Vector([1.0, -1.0 + coords[0] + coords[1], -coords[0], -coords[1]])


@qd.func
def func_edge_edge_friction_frame(
    ea0: qd.types.vector(3), ea1: qd.types.vector(3), eb0: qd.types.vector(3), eb1: qd.types.vector(3)
):
    """Compute the tangent basis of an edge-edge pair and the weights of its vertices (ea0, ea1, eb0, eb1).

    The basis spans the plane parallel to both edges, its first axis along edge A. The weights interpolate the closest
    points of the lines of the two edges, positively on edge A and negatively on edge B. Parallel edges have no such
    plane: their friction is left out by the caller (see func_edge_edge_mollifier_threshold).
    """
    edge_a = ea1 - ea0
    edge_b = eb1 - eb0
    offset = ea0 - eb0
    basis = qd.Matrix.cols([edge_a.normalized(), edge_a.cross(edge_b).cross(edge_a).normalized()])
    coupling = -edge_b.dot(edge_a)
    system = qd.Matrix([[edge_a.norm_sqr(), coupling], [coupling, edge_b.norm_sqr()]])
    coords = system.inverse() @ qd.Vector([-offset.dot(edge_a), offset.dot(edge_b)])
    return basis, qd.Vector([1.0 - coords[0], coords[0], coords[1] - 1.0, -coords[1]])


@qd.func
def func_point_edge_friction_frame(p: qd.types.vector(3), e0: qd.types.vector(3), e1: qd.types.vector(3)):
    """Compute the tangent basis of a point-edge pair and the weights of its vertices (p, e0, e1).

    The basis is the edge direction and the normal of the plane through the point and the edge. The weights are 1 for
    the point and minus the linear coordinates of its projection on the line of the edge.
    """
    edge = e1 - e0
    basis = qd.Matrix.cols([edge.normalized(), edge.cross(p - e0).normalized()])
    coord = (p - e0).dot(edge) / edge.norm_sqr()
    return basis, qd.Vector([1.0, coord - 1.0, -coord])


@qd.func
def func_point_point_friction_frame(p0: qd.types.vector(3), p1: qd.types.vector(3)):
    """Compute the tangent basis of a point-point pair, orthogonal to the line through the points, and their weights.

    The weights are 1 and -1.
    """
    tangent0, tangent1 = gu.qd_orthogonals((p1 - p0).normalized())
    return qd.Matrix.cols([tangent0, tangent1]), qd.Vector([1.0, -1.0])


@qd.func
def func_point_halfplane_friction_frame(normal: qd.types.vector(3)):
    """Compute the tangent basis of a point against a half-plane of unit normal, and the weight of the point, 1."""
    tangent0, tangent1 = gu.qd_orthogonals(normal)
    return qd.Matrix.cols([tangent0, tangent1]), qd.Vector([1.0])


@qd.func
def _func_tangent_displacement(basis: qd.types.matrix(3, 2), weights, displacements):
    relative_displacement = qd.Vector.zero(gs.qd_float, 3)
    for i_v in qd.static(range(weights.n)):
        relative_displacement += weights[i_v] * displacements[i_v, :]
    return basis.transpose() @ relative_displacement


@qd.func
def func_friction_energy(
    basis: qd.types.matrix(3, 2), weights, displacements, normal_force: float, mu: float, eps_vh: float
) -> float:
    """Compute the friction energy of a stencil over the time step.

    'basis' and 'weights' come from the frame of the stencil, 'displacements' holds the displacements of its vertices
    from their lagged positions as rows, 'normal_force' is the lagged normal force of the pair (see
    func_barrier_normal_force), 'mu' the friction coefficient and 'eps_vh' the smoothing length, the product of the
    smoothing velocity and the time step.
    """
    tangent_displacement2 = _func_tangent_displacement(basis, weights, displacements).norm_sqr()
    eps_vh2 = eps_vh * eps_vh
    smoothed_norm = qd.sqrt(tangent_displacement2)
    if tangent_displacement2 < eps_vh2:
        smoothed_norm = (
            tangent_displacement2 * (-qd.sqrt(tangent_displacement2) / 3.0 + eps_vh) / eps_vh2 + eps_vh / 3.0
        )
    return mu * normal_force * smoothed_norm


@qd.func
def func_friction_gradient_hessian(
    basis: qd.types.matrix(3, 2), weights, displacements, normal_force: float, mu: float, eps_vh: float
):
    """Compute the gradient and the positive semi-definite Hessian of the friction energy of a stencil.

    The arguments are those of func_friction_energy. In the tangent plane, the Hessian is of rank one across the
    displacement when sliding, the scaled identity at rest, and full rank in between, positive definite in all three
    regimes. Both are lifted to the stencil through the weights of its vertices. Returns the gradient vector and the
    Hessian matrix over the stencil.
    """
    tangent_displacement = _func_tangent_displacement(basis, weights, displacements)
    tangent_displacement2 = tangent_displacement.norm_sqr()
    eps_vh2 = eps_vh * eps_vh
    scale = mu * normal_force

    # Derivative of the smoothed norm divided by the norm, and Hessian in the tangent plane
    slope = gs.qd_float(0.0)
    hessian_tangent = qd.Matrix.zero(gs.qd_float, 2, 2)
    if tangent_displacement2 >= eps_vh2:
        slope = 1.0 / qd.sqrt(tangent_displacement2)
        across = qd.Vector([-tangent_displacement[1], tangent_displacement[0]])
        hessian_tangent = (across * (scale * slope / tangent_displacement2)).outer_product(across)
    else:
        slope = (-qd.sqrt(tangent_displacement2) + 2.0 * eps_vh) / eps_vh2
        if tangent_displacement2 <= 0.0:
            hessian_tangent = (scale * slope) * qd.Matrix.identity(gs.qd_float, 2)
        else:
            curvature = (-1.0 / eps_vh2) / qd.sqrt(tangent_displacement2)
            hessian_tangent = (tangent_displacement * curvature).outer_product(tangent_displacement)
            hessian_tangent = (hessian_tangent + slope * qd.Matrix.identity(gs.qd_float, 2)) * scale
    gradient_tangent = scale * slope * tangent_displacement

    gradient_block = basis @ gradient_tangent
    hessian_block = basis @ hessian_tangent @ basis.transpose()
    n = qd.static(3 * weights.n)
    gradient = qd.Vector([weights[i // 3] * gradient_block[i % 3] for i in qd.static(range(n))])
    hessian = qd.Matrix(
        [
            [weights[i // 3] * weights[j // 3] * hessian_block[i % 3, j % 3] for j in qd.static(range(n))]
            for i in qd.static(range(n))
        ]
    )
    return gradient, hessian

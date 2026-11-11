"""Distances between the primitives of an IPC contact stencil, their closest features and their derivatives.

A stencil lists its vertices in a fixed order: the point then the triangle for point-triangle (PT), the two vertices of
edge A then the two of edge B for edge-edge (EE), the point then the edge for point-edge (PE). The closest feature of a
stencil is encoded as a distance flag, a bit mask of its active vertices, bit k standing for vertex k of the stencil.
Two active vertices make a point-point (PP) feature, three a point-edge one and four the full PT or EE primitive.

The classifiers branch on quantities that are exactly zero for axis-aligned, coplanar or parallel geometry, so the side
of a tie they land on depends on how each product and sum is rounded. They reproduce the classification of a strict IEEE
evaluation only when the kernel is compiled without contraction into fused multiply-adds, that is with 'fast_math' off.
"""

import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu


@qd.func
def func_point_triangle_distance_flag(
    p: qd.types.vector(3), t0: qd.types.vector(3), t1: qd.types.vector(3), t2: qd.types.vector(3)
) -> int:
    """Classify the feature of a triangle closest to a point, as a distance flag over the stencil (p, t0, t1, t2).

    The point is projected on each edge in the plane of the triangle, its coordinates along the edge and across it
    being solved by Cramer's rule. The flag is a PE feature when the projection falls inside an edge on its outer side,
    a PP feature when the point lies in the corner region of a vertex, and the whole PT stencil otherwise. A triangle of
    zero area gives non-finite coordinates, which classify as PT.
    """
    normal = (t1 - t0).cross(t2 - t0)
    params_along = qd.Vector.zero(gs.qd_float, 3)
    params_across = qd.Vector.zero(gs.qd_float, 3)
    corners = (t0, t1, t2)
    for k in qd.static(range(3)):
        edge = corners[(k + 1) % 3] - corners[k]
        offset = p - corners[k]
        # Columns (edge, edge x normal, normal), whose determinants are the triple products with this vector
        binormal = edge.cross(normal).cross(normal)
        det = edge.dot(binormal)
        params_along[k] = offset.dot(binormal) / det
        params_across[k] = edge.dot(offset.cross(normal)) / det

    flag = 0xF
    if params_along[0] > 0.0 and params_along[0] < 1.0 and params_across[0] >= 0.0:
        flag = 0x7
    elif params_along[1] > 0.0 and params_along[1] < 1.0 and params_across[1] >= 0.0:
        flag = 0xD
    elif params_along[2] > 0.0 and params_along[2] < 1.0 and params_across[2] >= 0.0:
        flag = 0xB
    elif params_along[0] <= 0.0 and params_along[2] >= 1.0:
        flag = 0x3
    elif params_along[1] <= 0.0 and params_along[0] >= 1.0:
        flag = 0x5
    elif params_along[2] <= 0.0 and params_along[1] >= 1.0:
        flag = 0x9
    return flag


@qd.func
def func_edge_edge_distance_flag(
    ea0: qd.types.vector(3),
    ea1: qd.types.vector(3),
    eb0: qd.types.vector(3),
    eb1: qd.types.vector(3),
    parallel_tolerance: float,
) -> int:
    """Classify the features of two edges closest to each other, as a distance flag over the stencil (ea0, ea1, eb0, eb1).

    The closest points of the two supporting lines are clamped to the segments by the canonical segment-segment
    routine. Edges whose squared sine of angle is at most 'parallel_tolerance' have an indeterminate parameter along
    edge A, pinned to its first vertex, after which the clamp of the parameter along edge B and the re-clamp along edge
    A still select the closest feature. Crossing coplanar edges classify as EE at zero distance.
    """
    u = ea1 - ea0
    v = eb1 - eb0
    w = ea0 - eb0
    a = u.dot(u)
    b = u.dot(v)
    c = v.dot(v)
    d = u.dot(w)
    e = v.dot(w)
    denom = a * c - b * b
    s_num = b * e - c * d

    t_num = a * e - b * d
    t_denom = denom
    flag = 0xF
    is_parallel = a > 0.0 and c > 0.0 and denom <= parallel_tolerance * (a * c)
    if is_parallel or s_num <= 0.0:
        t_num = e
        t_denom = c
        flag = 0xD
    elif s_num >= denom:
        t_num = e + b
        t_denom = c
        flag = 0xE

    if t_num <= 0.0:
        flag = 0x5 if -d <= 0.0 else (0x6 if -d >= a else 0x7)
    elif t_num >= t_denom:
        s_num_end = -d + b
        flag = 0x9 if s_num_end <= 0.0 else (0xA if s_num_end >= a else 0xB)
    return flag


@qd.func
def func_point_edge_distance_flag(p: qd.types.vector(3), e0: qd.types.vector(3), e1: qd.types.vector(3)) -> int:
    """Classify the feature of an edge closest to a point, as a distance flag over the stencil (p, e0, e1).

    The flag is PE when the projection of the point falls strictly inside the edge, and PP with the nearer vertex
    otherwise. A collapsed edge picks the nearer of its two vertices, the first one on a tie.
    """
    edge = e1 - e0
    offset = p - e0
    edge_norm2 = edge.dot(edge)
    flag = 0x1
    if edge_norm2 <= 0.0:
        offset_end = p - e1
        flag = flag | (0x2 if offset.dot(offset) <= offset_end.dot(offset_end) else 0x4)
    else:
        ratio = edge.dot(offset) / edge_norm2
        if ratio < 1.0:
            flag = flag | 0x2
        if ratio > 0.0:
            flag = flag | 0x4
    return flag


@qd.func
def func_edge_edge_distance_type(
    ea0: qd.types.vector(3),
    ea1: qd.types.vector(3),
    eb0: qd.types.vector(3),
    eb1: qd.types.vector(3),
    parallel_tolerance: float,
) -> int:
    """Classify the features of two edges closest to each other by projecting parallel edges on each other.

    Same distance flag encoding as func_edge_edge_distance_flag, which it differs from on parallel and collapsed edges.
    Edges whose squared sine of angle is below 'parallel_tolerance' project the vertices of edge B on the line of edge
    A, which picks the overlap feature directly, and a collapsed edge degenerates to its first vertex. This is the
    classifier of the additive continuous collision detection (CCD), whose distance must stay a true segment distance
    for edges of any length.
    """
    u = ea1 - ea0
    v = eb1 - eb0
    w = ea0 - eb0
    a = u.norm_sqr()
    b = u.dot(v)
    c = v.norm_sqr()
    d = u.dot(w)
    e = v.dot(w)
    denom = a * c - b * b

    # Bits of the closest feature of each edge: its first vertex, its second one, or both for its interior
    bits_a = 0x3
    bits_b = 0xC
    if a <= 0.0 or c <= 0.0:
        if a <= 0.0:
            bits_a = 0x1
        if c <= 0.0:
            bits_b = 0x4
    elif u.cross(v).norm_sqr() < parallel_tolerance * a * c:
        alpha = (eb0 - ea0).dot(u) / u.norm_sqr()
        beta = (eb1 - ea0).dot(u) / u.norm_sqr()
        if alpha < 0.0:
            bits_a = 0x3 if (0.0 <= beta and beta <= 1.0) else 0x1
            bits_b = 0x4 if beta <= alpha else (0x8 if beta <= 1.0 else 0xC)
        elif alpha > 1.0:
            bits_a = 0x3 if (0.0 <= beta and beta <= 1.0) else 0x2
            bits_b = 0x4 if beta >= alpha else (0x8 if 0.0 <= beta else 0xC)
        else:
            bits_b = 0x4
    else:
        s_num = b * e - c * d
        t_num = a * e - b * d
        t_denom = denom
        if s_num <= 0.0:
            t_num = e
            t_denom = c
            bits_a = 0x1
        elif s_num >= denom:
            t_num = e + b
            t_denom = c
            bits_a = 0x2
        if t_num <= 0.0:
            bits_a = 0x1 if -d <= 0.0 else (0x2 if -d >= a else 0x3)
            bits_b = 0x4
        elif t_num >= t_denom:
            bits_a = 0x1 if (-d + b) <= 0.0 else (0x2 if (-d + b) >= a else 0x3)
            bits_b = 0x8
    return bits_a | bits_b


@qd.func
def func_point_edge_distance2(p: qd.types.vector(3), e0: qd.types.vector(3), e1: qd.types.vector(3)) -> float:
    """Compute the squared distance from a point to the line through an edge."""
    return (e0 - p).cross(e1 - p).norm_sqr() / (e1 - e0).norm_sqr()


@qd.func
def func_point_triangle_distance2(
    p: qd.types.vector(3), t0: qd.types.vector(3), t1: qd.types.vector(3), t2: qd.types.vector(3)
) -> float:
    """Compute the squared distance from a point to the plane of a triangle."""
    normal = (t1 - t0).cross(t2 - t0)
    offset = (p - t0).dot(normal)
    return offset * offset / normal.norm_sqr()


@qd.func
def func_edge_edge_distance2(
    ea0: qd.types.vector(3), ea1: qd.types.vector(3), eb0: qd.types.vector(3), eb1: qd.types.vector(3)
) -> float:
    """Compute the squared distance between the lines through two edges."""
    normal = (ea1 - ea0).cross(eb1 - eb0)
    offset = (eb0 - ea0).dot(normal)
    return offset * offset / normal.norm_sqr()


@qd.func
def _func_plane_distance2_gradient(offset: qd.types.vector(3), u: qd.types.vector(3), v: qd.types.vector(3)):
    """Compute the gradient of the squared distance (offset . n)^2 / |n|^2 of a point to a plane, n = u x v.

    Returns the gradients with respect to 'offset', 'u' and 'v' as the rows of a matrix. With s = offset . n and
    r = offset - (s / |n|^2) n the in-plane part of the offset, they are 2 s n / |n|^2, 2 s (v x r) / |n|^2 and
    2 s (r x u) / |n|^2.
    """
    normal = u.cross(v)
    normal_norm2 = normal.norm_sqr()
    offset_normal = offset.dot(normal)
    scale = 2.0 * offset_normal / normal_norm2
    offset_in_plane = offset - (offset_normal / normal_norm2) * normal
    return qd.Matrix.rows([scale * normal, scale * v.cross(offset_in_plane), scale * offset_in_plane.cross(u)])


@qd.func
def func_point_edge_distance2_gradient(p: qd.types.vector(3), e0: qd.types.vector(3), e1: qd.types.vector(3)):
    """Compute the gradient of the squared distance from a point to the line through an edge, over (p, e0, e1)."""
    offset0 = e0 - p
    offset1 = e1 - p
    edge = e1 - e0
    edge_norm2 = edge.norm_sqr()
    cross = offset0.cross(offset1)
    distance2 = cross.norm_sqr() / edge_norm2
    grad_e0 = (2.0 * offset1.cross(cross) + 2.0 * distance2 * edge) / edge_norm2
    grad_e1 = (2.0 * cross.cross(offset0) - 2.0 * distance2 * edge) / edge_norm2
    grad_p = -(grad_e0 + grad_e1)
    blocks = (grad_p, grad_e0, grad_e1)
    return qd.Vector([blocks[k // 3][k % 3] for k in qd.static(range(9))])


@qd.func
def func_point_triangle_distance2_gradient(
    p: qd.types.vector(3), t0: qd.types.vector(3), t1: qd.types.vector(3), t2: qd.types.vector(3)
):
    """Compute the gradient of the squared distance from a point to the plane of a triangle, over (p, t0, t1, t2)."""
    grads = _func_plane_distance2_gradient(p - t0, t1 - t0, t2 - t0)
    blocks = (grads[0, :], -(grads[0, :] + grads[1, :] + grads[2, :]), grads[1, :], grads[2, :])
    return qd.Vector([blocks[k // 3][k % 3] for k in qd.static(range(12))])


@qd.func
def func_edge_edge_distance2_gradient(
    ea0: qd.types.vector(3), ea1: qd.types.vector(3), eb0: qd.types.vector(3), eb1: qd.types.vector(3)
):
    """Compute the gradient of the squared distance between the lines through two edges, over (ea0, ea1, eb0, eb1)."""
    grads = _func_plane_distance2_gradient(eb0 - ea0, ea1 - ea0, eb1 - eb0)
    blocks = (-(grads[0, :] + grads[1, :]), grads[1, :], grads[0, :] - grads[2, :], grads[2, :])
    return qd.Vector([blocks[k // 3][k % 3] for k in qd.static(range(12))])


@qd.func
def _func_cross_norm2_hessian(u: qd.types.vector(3), v: qd.types.vector(3)):
    """Compute the Hessian of |u x v|^2 = |u|^2 |v|^2 - (u . v)^2 over (u, v)."""
    identity = qd.Matrix.identity(gs.qd_float, 3)
    block_uu = 2.0 * (v.norm_sqr() * identity - v.outer_product(v))
    block_vv = 2.0 * (u.norm_sqr() * identity - u.outer_product(u))
    block_uv = 4.0 * u.outer_product(v) - 2.0 * v.outer_product(u) - 2.0 * u.dot(v) * identity
    hessian = qd.Matrix.zero(gs.qd_float, 6, 6)
    for i in qd.static(range(3)):
        for j in qd.static(range(3)):
            hessian[i, j] = block_uu[i, j]
            hessian[i, 3 + j] = block_uv[i, j]
            hessian[3 + i, j] = block_uv[j, i]
            hessian[3 + i, 3 + j] = block_vv[i, j]
    return hessian


@qd.func
def _func_vertex_hessian(variable_hessian, coefficients: qd.template()):
    """Map a Hessian over difference vectors of a stencil to its vertices.

    Each variable is a signed sum of vertices, 'coefficients[i_v][i_x]' being the coefficient (-1, 0 or 1) of vertex i_v
    in variable i_x.
    """
    n_v, n_x = qd.static(len(coefficients), len(coefficients[0]))
    hessian = qd.Matrix.zero(gs.qd_float, 3 * n_v, 3 * n_v)
    for i_v in qd.static(range(n_v)):
        for j_v in qd.static(range(n_v)):
            for i_x in qd.static(range(n_x)):
                for j_x in qd.static(range(n_x)):
                    sign = qd.static(coefficients[i_v][i_x] * coefficients[j_v][j_x])
                    for i in qd.static(range(3)):
                        for j in qd.static(range(3)):
                            if qd.static(sign > 0):
                                hessian[3 * i_v + i, 3 * j_v + j] = (
                                    hessian[3 * i_v + i, 3 * j_v + j] + variable_hessian[3 * i_x + i, 3 * j_x + j]
                                )
                            elif qd.static(sign < 0):
                                hessian[3 * i_v + i, 3 * j_v + j] = (
                                    hessian[3 * i_v + i, 3 * j_v + j] - variable_hessian[3 * i_x + i, 3 * j_x + j]
                                )
    return hessian


@qd.func
def func_point_edge_distance2_hessian(p: qd.types.vector(3), e0: qd.types.vector(3), e1: qd.types.vector(3)):
    """Compute the Hessian of the squared distance from a point to the line through an edge, over (p, e0, e1).

    With x = e0 - p, e = e1 - e0, N = |x x e|^2 and L = |e|^2, the squared distance is D = N / L and its Hessian over
    (x, e) is (Hess N - D Hess L - grad D grad L^T - grad L grad D^T) / L.
    """
    offset = e0 - p
    edge = e1 - e0
    edge_norm2 = edge.norm_sqr()
    cross = offset.cross(edge)
    distance2 = cross.norm_sqr() / edge_norm2
    grad_offset = 2.0 * edge.cross(cross) / edge_norm2
    grad_edge = (2.0 * cross.cross(offset) - 2.0 * distance2 * edge) / edge_norm2
    coupling_edge = grad_edge.outer_product(2.0 * edge)
    block_edge = 2.0 * distance2 * qd.Matrix.identity(gs.qd_float, 3) + coupling_edge + coupling_edge.transpose()
    block_offset_edge = grad_offset.outer_product(2.0 * edge)
    variable_hessian = _func_cross_norm2_hessian(offset, edge) / edge_norm2
    for i in qd.static(range(3)):
        for j in qd.static(range(3)):
            variable_hessian[i, 3 + j] = variable_hessian[i, 3 + j] - block_offset_edge[i, j] / edge_norm2
            variable_hessian[3 + i, j] = variable_hessian[3 + i, j] - block_offset_edge[j, i] / edge_norm2
            variable_hessian[3 + i, 3 + j] = variable_hessian[3 + i, 3 + j] - block_edge[i, j] / edge_norm2
    return _func_vertex_hessian(variable_hessian, ((-1, 0), (1, -1), (0, 1)))


@qd.func
def func_edge_edge_distance2_hessian(
    ea0: qd.types.vector(3), ea1: qd.types.vector(3), eb0: qd.types.vector(3), eb1: qd.types.vector(3)
):
    """Compute the Hessian of the squared distance between the lines through two edges, over (ea0, ea1, eb0, eb1).

    With r = eb0 - ea0, u = ea1 - ea0, v = eb1 - eb0, n = u x v, s = r . n, q = |n|^2 and a = s / q, the squared
    distance is D = s^2 / q and its Hessian over (r, u, v) is (2 / q) w w^T + 2 a Hess s - a^2 Hess q, with
    w = grad s - a grad q = (n, v x (r - 2 a n), (r - 2 a n) x u). It stays bounded as the lines meet (s -> 0).
    """
    offset = eb0 - ea0
    u = ea1 - ea0
    v = eb1 - eb0
    normal = u.cross(v)
    normal_norm2 = normal.norm_sqr()
    ratio = offset.dot(normal) / normal_norm2
    shifted = offset - 2.0 * ratio * normal
    blocks = (normal, v.cross(shifted), shifted.cross(u))
    w = qd.Vector([blocks[k // 3][k % 3] for k in qd.static(range(9))])
    variable_hessian = (2.0 / normal_norm2) * w.outer_product(w)
    # s = r . (u x v) is trilinear: its Hessian has the off-diagonal blocks Hess_uv = -[r]x, Hess_ur = [v]x and
    # Hess_vr = -[u]x, [.]x being the skew-symmetric matrix of a cross product
    skews = (gu.qd_vec_to_skew(offset), gu.qd_vec_to_skew(v), gu.qd_vec_to_skew(u))
    cross_hessian = _func_cross_norm2_hessian(u, v)
    for i in qd.static(range(3)):
        for j in qd.static(range(3)):
            variable_hessian[3 + i, 6 + j] = variable_hessian[3 + i, 6 + j] - 2.0 * ratio * skews[0][i, j]
            variable_hessian[6 + i, 3 + j] = variable_hessian[6 + i, 3 + j] + 2.0 * ratio * skews[0][i, j]
            variable_hessian[3 + i, j] = variable_hessian[3 + i, j] + 2.0 * ratio * skews[1][i, j]
            variable_hessian[i, 3 + j] = variable_hessian[i, 3 + j] - 2.0 * ratio * skews[1][i, j]
            variable_hessian[6 + i, j] = variable_hessian[6 + i, j] - 2.0 * ratio * skews[2][i, j]
            variable_hessian[i, 6 + j] = variable_hessian[i, 6 + j] + 2.0 * ratio * skews[2][i, j]
    for i in qd.static(range(6)):
        for j in qd.static(range(6)):
            variable_hessian[3 + i, 3 + j] = variable_hessian[3 + i, 3 + j] - ratio * ratio * cross_hessian[i, j]
    return _func_vertex_hessian(variable_hessian, ((-1, -1, 0), (0, 1, 0), (1, 0, -1), (0, 0, 1)))


@qd.func
def func_edge_edge_cross_norm2(
    ea0: qd.types.vector(3), ea1: qd.types.vector(3), eb0: qd.types.vector(3), eb1: qd.types.vector(3)
):
    """Compute the squared norm of the cross product of two edges, and its gradient and Hessian over (ea0, ea1, eb0, eb1).

    It vanishes for parallel edges, which is where the mollifier of the edge-edge barrier ramps the barrier down.
    Returns the squared cross norm, its gradient vector and its Hessian matrix.
    """
    u = ea1 - ea0
    v = eb1 - eb0
    cross = u.cross(v)
    grad_u = 2.0 * v.cross(cross)
    grad_v = 2.0 * cross.cross(u)
    blocks = (-grad_u, grad_u, -grad_v, grad_v)
    gradient = qd.Vector([blocks[k // 3][k % 3] for k in qd.static(range(12))])
    hessian = _func_vertex_hessian(_func_cross_norm2_hessian(u, v), ((-1, 0), (1, 0), (0, -1), (0, 1)))
    return cross.norm_sqr(), gradient, hessian


@qd.func
def _func_select_vertex(
    i_v: int, x0: qd.types.vector(3), x1: qd.types.vector(3), x2: qd.types.vector(3), x3: qd.types.vector(3)
):
    return qd.select(i_v == 0, x0, qd.select(i_v == 1, x1, qd.select(i_v == 2, x2, x3)))


@qd.func
def _func_scatter_blocks(i_v0: int, i_v1: int, i_v2: int, blocks):
    """Place the 3D blocks of a sub-stencil at their vertex slots in a 4-vertex stencil, zero elsewhere.

    The first 'blocks.n // 3' of the slots are used.
    """
    gradient = qd.Vector.zero(gs.qd_float, 12)
    slots = (i_v0, i_v1, i_v2)
    for i_v in qd.static(range(4)):
        for i_s in qd.static(range(blocks.n // 3)):
            if slots[i_s] == i_v:
                for k in qd.static(range(3)):
                    gradient[3 * i_v + k] = blocks[3 * i_s + k]
    return gradient


@qd.func
def _func_scatter_hessian_blocks(i_v0: int, i_v1: int, i_v2: int, blocks):
    """Place the 3x3 blocks of the Hessian of a sub-stencil at their vertex slots in a 4-vertex stencil, zero elsewhere.

    The first 'blocks.n // 3' of the slots are used.
    """
    hessian = qd.Matrix.zero(gs.qd_float, 12, 12)
    slots = qd.Vector([i_v0, i_v1, i_v2], dt=gs.qd_int)
    for i_s in qd.static(range(blocks.n // 3)):
        for j_s in qd.static(range(blocks.n // 3)):
            for i in qd.static(range(3)):
                for j in qd.static(range(3)):
                    hessian[3 * slots[i_s] + i, 3 * slots[j_s] + j] = blocks[3 * i_s + i, 3 * j_s + j]
    return hessian


@qd.func
def _func_edge_edge_feature_slots(flag: int):
    """Find the stencil slots of the closest features of an edge-edge distance flag of two or three active vertices.

    A PE feature gives the slot of its point, the active vertex of the edge with one inactive vertex, then the slots of
    the other edge. A PP feature gives the slot of the vertex of edge A, the one of edge B, then a dummy slot.
    """
    i_va = 0 if flag & 0x1 else 1
    i_vb = 2 if flag & 0x4 else 3
    i_v0, i_v1, i_v2 = i_va, i_vb, 0
    if qd.math.popcnt(flag) == 3:
        is_point_on_a = (flag & 0x3) != 0x3
        i_v0 = i_va if is_point_on_a else i_vb
        i_v1 = 2 if is_point_on_a else 0
        i_v2 = i_v1 + 1
    return i_v0, i_v1, i_v2


@qd.func
def _func_point_point_gradient(x0: qd.types.vector(3), x1: qd.types.vector(3)):
    offset = 2.0 * (x0 - x1)
    blocks = (offset, -offset)
    return qd.Vector([blocks[k // 3][k % 3] for k in qd.static(range(6))])


@qd.func
def func_point_triangle_flagged_distance2(
    flag: int, p: qd.types.vector(3), t0: qd.types.vector(3), t1: qd.types.vector(3), t2: qd.types.vector(3)
):
    """Compute the squared distance of the closest feature of a point-triangle stencil, and its gradient.

    The feature is given by its distance flag (see func_point_triangle_distance_flag). Returns the squared distance and
    its gradient over (p, t0, t1, t2), zero at the inactive vertices.
    """
    distance2 = gs.qd_float(0.0)
    gradient = qd.Vector.zero(gs.qd_float, 12)
    n_active = qd.math.popcnt(flag)
    if n_active == 4:
        distance2 = func_point_triangle_distance2(p, t0, t1, t2)
        gradient = func_point_triangle_distance2_gradient(p, t0, t1, t2)
    else:
        # The lowest and highest active triangle vertices, the same one for a PP feature
        i_v_first = 1 if flag & 0x2 else (2 if flag & 0x4 else 3)
        i_v_last = 3 if flag & 0x8 else (2 if flag & 0x4 else 1)
        e0 = _func_select_vertex(i_v_first, p, t0, t1, t2)
        e1 = _func_select_vertex(i_v_last, p, t0, t1, t2)
        if n_active == 3:
            distance2 = func_point_edge_distance2(p, e0, e1)
            gradient = _func_scatter_blocks(0, i_v_first, i_v_last, func_point_edge_distance2_gradient(p, e0, e1))
        else:
            distance2 = (p - e0).norm_sqr()
            gradient = _func_scatter_blocks(0, i_v_first, 0, _func_point_point_gradient(p, e0))
    return distance2, gradient


@qd.func
def func_edge_edge_flagged_distance2(
    flag: int, ea0: qd.types.vector(3), ea1: qd.types.vector(3), eb0: qd.types.vector(3), eb1: qd.types.vector(3)
):
    """Compute the squared distance of the closest features of an edge-edge stencil, and its gradient.

    The features are given by their distance flag (see func_edge_edge_distance_flag). A PE feature takes as point the
    active vertex of the edge with one inactive vertex. Returns the squared distance and its gradient over
    (ea0, ea1, eb0, eb1), zero at the inactive vertices.
    """
    distance2 = gs.qd_float(0.0)
    gradient = qd.Vector.zero(gs.qd_float, 12)
    n_active = qd.math.popcnt(flag)
    i_v0, i_v1, i_v2 = _func_edge_edge_feature_slots(flag)
    x0 = _func_select_vertex(i_v0, ea0, ea1, eb0, eb1)
    x1 = _func_select_vertex(i_v1, ea0, ea1, eb0, eb1)
    if n_active == 4:
        distance2 = func_edge_edge_distance2(ea0, ea1, eb0, eb1)
        gradient = func_edge_edge_distance2_gradient(ea0, ea1, eb0, eb1)
    elif n_active == 3:
        x2 = _func_select_vertex(i_v2, ea0, ea1, eb0, eb1)
        distance2 = func_point_edge_distance2(x0, x1, x2)
        gradient = _func_scatter_blocks(i_v0, i_v1, i_v2, func_point_edge_distance2_gradient(x0, x1, x2))
    else:
        distance2 = (x0 - x1).norm_sqr()
        gradient = _func_scatter_blocks(i_v0, i_v1, 0, _func_point_point_gradient(x0, x1))
    return distance2, gradient


@qd.func
def func_edge_edge_flagged_distance2_hessian(
    flag: int, ea0: qd.types.vector(3), ea1: qd.types.vector(3), eb0: qd.types.vector(3), eb1: qd.types.vector(3)
):
    """Compute the Hessian of the squared distance of the closest features of an edge-edge stencil.

    The features are given by their distance flag, as for func_edge_edge_flagged_distance2. Returns the Hessian over
    (ea0, ea1, eb0, eb1), zero at the inactive vertices.
    """
    hessian = qd.Matrix.zero(gs.qd_float, 12, 12)
    n_active = qd.math.popcnt(flag)
    i_v0, i_v1, i_v2 = _func_edge_edge_feature_slots(flag)
    if n_active == 4:
        hessian = func_edge_edge_distance2_hessian(ea0, ea1, eb0, eb1)
    elif n_active == 3:
        x0 = _func_select_vertex(i_v0, ea0, ea1, eb0, eb1)
        x1 = _func_select_vertex(i_v1, ea0, ea1, eb0, eb1)
        x2 = _func_select_vertex(i_v2, ea0, ea1, eb0, eb1)
        hessian = _func_scatter_hessian_blocks(i_v0, i_v1, i_v2, func_point_edge_distance2_hessian(x0, x1, x2))
    else:
        identity = 2.0 * qd.Matrix.identity(gs.qd_float, 3)
        blocks = qd.Matrix.zero(gs.qd_float, 6, 6)
        for i in qd.static(range(3)):
            for j in qd.static(range(3)):
                blocks[i, j] = identity[i, j]
                blocks[i, 3 + j] = -identity[i, j]
                blocks[3 + i, j] = -identity[i, j]
                blocks[3 + i, 3 + j] = identity[i, j]
        hessian = _func_scatter_hessian_blocks(i_v0, i_v1, 0, blocks)
    return hessian


@qd.func
def func_point_edge_flagged_distance2(flag: int, p: qd.types.vector(3), e0: qd.types.vector(3), e1: qd.types.vector(3)):
    """Compute the squared distance of the closest feature of a point-edge stencil, and its gradient.

    The feature is given by its distance flag (see func_point_edge_distance_flag). Returns the squared distance and its
    gradient over (p, e0, e1), zero at the inactive vertex.
    """
    distance2 = gs.qd_float(0.0)
    gradient = qd.Vector.zero(gs.qd_float, 9)
    if qd.math.popcnt(flag) == 3:
        distance2 = func_point_edge_distance2(p, e0, e1)
        gradient = func_point_edge_distance2_gradient(p, e0, e1)
    else:
        is_first_vertex = (flag & 0x2) != 0
        vertex = qd.select(is_first_vertex, e0, e1)
        distance2 = (p - vertex).norm_sqr()
        grad_p = 2.0 * (p - vertex)
        for k in qd.static(range(3)):
            gradient[k] = grad_p[k]
            gradient[3 + k] = qd.select(is_first_vertex, -grad_p[k], 0.0)
            gradient[6 + k] = qd.select(is_first_vertex, 0.0, -grad_p[k])
    return distance2, gradient


@qd.func
def func_edge_edge_closest_points(
    ea0: qd.types.vector(3), ea1: qd.types.vector(3), eb0: qd.types.vector(3), eb1: qd.types.vector(3)
):
    """Find the closest points of two segments, by clamping the closest points of their lines to the segments.

    A segment shorter than 1e-150 is treated as a point, and parallel segments pin the parameter along edge A to its
    first vertex before the clamp. Returns the distance and the closest points on edge A and on edge B.
    """
    dir_a = ea1 - ea0
    dir_b = eb1 - eb0
    offset = ea0 - eb0
    a = dir_a.norm_sqr()
    e = dir_b.norm_sqr()
    f = dir_b.dot(offset)
    c = dir_a.dot(offset)
    b = dir_a.dot(dir_b)
    s = gs.qd_float(0.0)
    t = gs.qd_float(0.0)
    if a <= 1e-300 and e <= 1e-300:
        s = 0.0
        t = 0.0
    elif a <= 1e-300:
        t = qd.min(qd.max(f / e, 0.0), 1.0)
    elif e <= 1e-300:
        s = qd.min(qd.max(-c / a, 0.0), 1.0)
    else:
        denom = a * e - b * b
        if denom > 0.0:
            s = qd.min(qd.max((b * f - c * e) / denom, 0.0), 1.0)
        t = (b * s + f) / e
        if t < 0.0:
            t = 0.0
            s = qd.min(qd.max(-c / a, 0.0), 1.0)
        elif t > 1.0:
            t = 1.0
            s = qd.min(qd.max((b - c) / a, 0.0), 1.0)
    closest_a = ea0 + s * dir_a
    closest_b = eb0 + t * dir_b
    return (closest_a - closest_b).norm(), closest_a, closest_b


@qd.func
def func_point_triangle_closest_point(
    p: qd.types.vector(3), t0: qd.types.vector(3), t1: qd.types.vector(3), t2: qd.types.vector(3)
):
    """Find the point of a triangle closest to a point, by locating it in the Voronoi regions of the triangle features.

    Returns the distance and the closest point on the triangle.
    """
    ab = t1 - t0
    ac = t2 - t0
    ap = p - t0
    bp = p - t1
    cp = p - t2
    d1 = ab.dot(ap)
    d2 = ac.dot(ap)
    d3 = ab.dot(bp)
    d4 = ac.dot(bp)
    d5 = ab.dot(cp)
    d6 = ac.dot(cp)
    vc = d1 * d4 - d3 * d2
    vb = d5 * d2 - d1 * d6
    va = d3 * d6 - d5 * d4
    closest = t0
    if d1 <= 0.0 and d2 <= 0.0:
        closest = t0
    elif d3 >= 0.0 and d4 <= d3:
        closest = t1
    elif vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        closest = t0 + (d1 / (d1 - d3)) * ab
    elif d6 >= 0.0 and d5 <= d6:
        closest = t2
    elif vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        closest = t0 + (d2 / (d2 - d6)) * ac
    elif va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        closest = t1 + ((d4 - d3) / ((d4 - d3) + (d5 - d6))) * (t2 - t1)
    else:
        denom = 1.0 / (va + vb + vc)
        closest = t0 + ab * (vb * denom) + ac * (vc * denom)
    return (p - closest).norm(), closest


@qd.func
def func_point_edge_closest_point(p: qd.types.vector(3), e0: qd.types.vector(3), e1: qd.types.vector(3)):
    """Find the point of a segment closest to a point. A segment shorter than 1e-150 is treated as its first vertex.

    Returns the distance and the closest point on the segment.
    """
    edge = e1 - e0
    edge_norm2 = edge.norm_sqr()
    t = gs.qd_float(0.0)
    if edge_norm2 > 1e-300:
        t = qd.min(qd.max((p - e0).dot(edge) / edge_norm2, 0.0), 1.0)
    closest = e0 + t * edge
    return (p - closest).norm(), closest


@qd.func
def func_closest_points(points_a, points_b):
    """Find the closest points of the two sides of a contact stencil, each given as the rows of a matrix.

    The shapes of the sides select the primitives: one point against three (point-triangle), two against two
    (edge-edge), one against two (point-edge) or one against one (point-point). Returns the distance and the closest
    points on side A and on side B.
    """
    n_a, n_b = qd.static(points_a.n, points_b.n)
    closest_a = points_a[0, :]
    closest_b = points_b[0, :]
    distance = gs.qd_float(0.0)
    if qd.static(n_a == 1 and n_b == 3):
        distance, closest_b = func_point_triangle_closest_point(
            points_a[0, :], points_b[0, :], points_b[1, :], points_b[2, :]
        )
    elif qd.static(n_a == 2 and n_b == 2):
        distance, closest_a, closest_b = func_edge_edge_closest_points(
            points_a[0, :], points_a[1, :], points_b[0, :], points_b[1, :]
        )
    elif qd.static(n_a == 1 and n_b == 2):
        distance, closest_b = func_point_edge_closest_point(points_a[0, :], points_b[0, :], points_b[1, :])
    else:
        distance = (points_a[0, :] - points_b[0, :]).norm()
    return distance, closest_a, closest_b

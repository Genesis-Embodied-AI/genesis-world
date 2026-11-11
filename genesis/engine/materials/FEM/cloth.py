"""
Cloth material, simulated as a thin shell by the Newton solver.

The membrane is the Baraff-Witkin model with strain limiting and the bending a quadratic or a dihedral-angle model over
the hinges of the mesh (two triangles sharing an edge). The energies are functions of values, the solver scaling them
by the rest size of an element and the squared substep.
"""

import math
from typing import Literal

import quadrants as qd

import genesis as gs
from genesis.typing import NonNegativeFloat, PositiveFloat

from .base import Base


class Cloth(Base):
    """
    Cloth material for thin shell/membrane simulation, solved by the Newton solver of the scene with its contacts with
    itself, with the other cloths and with the fixed planes of the scene (see 'gs.options.IPCOptions'). Only accepts
    surface mesh morphs.

    The membrane resists stretch and shear separately (Baraff-Witkin), its stretch stiffening sharply past the rest
    length (strain limiting), with no Poisson effect. The bending resists the folding of every pair of adjacent
    triangles. The contact between cloths is reliable in double precision, `gs.init(precision='64')`, which a scene in
    single precision recommends with a warning.

    Parameters
    ----------
    E : float, optional
        Effective Young's modulus of the membrane in stretch (Pa). Default is 1e4.
    rho : float, optional
        Material density (kg/m^3). Default is 200 (typical fabric).
    thickness : float, optional
        Half thickness of the shell (m): the shell is 2 * thickness thick for its mass and its bending. Default is
        0.001.
    shear_modulus : float | None, optional
        Effective shear modulus of the membrane (Pa). None takes E / 2, the isotropic value without Poisson effect.
        Default is None.
    strain_limit_multiplier : float, optional
        Strength of the strain limiting, relative to the stretch stiffness: the stretch energy gains a cubic term past
        the rest length. A larger value keeps the cloth closer to inextensible at the cost of a stiffer, harder solve,
        and 0 disables it. Default is 100.
    bending : str | None, optional
        Bending model: "quadratic" (cheap, exact for a flat rest shape under small folds) or "hinge" (the dihedral
        angle of every hinge, exact for large folds and curved rest shapes, at the cost of a harder solve), or None for
        no bending. Default is "quadratic".
    bending_E : float | None, optional
        Young's modulus of the bending (Pa), over the full thickness of the shell. None takes E. Default is None.
    contact_thickness : float | None, optional
        Gap the contacts keep between the cloth and the other surfaces (m). None takes the thickness. A thicker gap
        keeps the cloth farther from what it touches, while 0 lets a cloth start right on a surface. Default is None.
    friction_mu : float, optional
        Friction coefficient of the contacts. Default is 0.05.
    contact_resistance : float, optional
        Stiffness of the contacts. A stiffer contact keeps the cloth closer to the contact thickness at the cost of a
        harder solve. Default is 1e4.

    Examples
    --------
    >>> cloth = scene.add_entity(
    ...     morph=gs.morphs.Mesh(file="cloth.obj"),
    ...     material=gs.materials.FEM.Cloth(E=10e3, rho=200, thickness=0.001),
    ... )
    """

    E: PositiveFloat = 1e4
    rho: PositiveFloat = 200.0
    thickness: PositiveFloat = 0.001
    shear_modulus: PositiveFloat | None = None
    strain_limit_multiplier: NonNegativeFloat = 100.0
    bending: Literal["quadratic", "hinge"] | None = "quadratic"
    bending_E: NonNegativeFloat | None = None
    contact_thickness: NonNegativeFloat | None = None
    friction_mu: NonNegativeFloat = 0.05
    contact_resistance: PositiveFloat = 1e4


# ------------------------------------------------------------------------------------
# ------------------------------------ membrane --------------------------------------
# ------------------------------------------------------------------------------------

# The deformation gradient F of a triangle maps its rest plane to the world, F = [x1 - x0, x2 - x0] D^-1 with D the
# edges of the triangle at rest in its own plane. Its columns f_u and f_v are the images of the two rest axes, and its
# Hessians act on vec(F) = [f_u; f_v].


@qd.func
def func_triangle_deformation_gradient(
    x0: qd.types.vector(3), x1: qd.types.vector(3), x2: qd.types.vector(3), rest_inv: qd.types.matrix(2, 2)
):
    """Compute the 3x2 deformation gradient of a triangle from its vertices and the inverse of its rest edges."""
    return qd.Matrix.cols([x1 - x0, x2 - x0]) @ rest_inv


@qd.func
def func_baraff_witkin_energy(
    F: qd.types.matrix(3, 2), stretch_stiffness: float, shear_stiffness: float, strain_limit_multiplier: float
) -> float:
    """
    Compute the Baraff-Witkin energy density of a membrane with strain limiting.

    The stretch of each rest axis is |f| - 1, its energy the stretch squared plus 'strain_limit_multiplier' times its
    cube under tension. The shear is f_u . f_v, its energy its square. The density is 'stretch_stiffness' times the
    stretch energy plus 'shear_stiffness' times the shear energy.
    """
    f_u = qd.Vector([F[0, 0], F[1, 0], F[2, 0]])
    f_v = qd.Vector([F[0, 1], F[1, 1], F[2, 1]])
    stretch = qd.Vector([f_u.norm(), f_v.norm()]) - 1.0
    tension = qd.max(stretch, 0.0)
    shear = f_u.dot(f_v)
    stretch_energy = stretch.dot(stretch) + strain_limit_multiplier * (tension * tension).dot(tension)
    return stretch_stiffness * stretch_energy + shear_stiffness * shear * shear


@qd.func
def func_baraff_witkin_gradient_hessian(
    F: qd.types.matrix(3, 2), stretch_stiffness: float, shear_stiffness: float, strain_limit_multiplier: float
):
    """
    Compute the gradient and the positive semi-definite Hessian of the Baraff-Witkin energy density in the deformation
    gradient.

    The stretch of an axis keeps its curvature along f, and its curvature across f under tension only, where it is
    positive. The shear Hessian is projected analytically onto its positive eigenspace. Returns the 3x2 gradient and
    the 6x6 Hessian in vec(F).
    """
    EPS = qd.static(1e-30)
    f_u = qd.Vector([F[0, 0], F[1, 0], F[2, 0]])
    f_v = qd.Vector([F[0, 1], F[1, 1], F[2, 1]])
    shear = f_u.dot(f_v)

    # Stretch of each axis, independent of the other one
    gradient_u = qd.Vector.zero(gs.qd_float, 3)
    gradient_v = qd.Vector.zero(gs.qd_float, 3)
    hessian = qd.Matrix.zero(gs.qd_float, 6, 6)
    for i_axis in qd.static(range(2)):
        f = f_u if i_axis == 0 else f_v
        length = f.norm()
        if length > EPS:
            coefficient = 1.0 - 1.0 / length
            across = gs.qd_float(0.0)
            along = gs.qd_float(2.0)
            if length > 1.0:
                coefficient += 1.5 * strain_limit_multiplier * (length + 1.0 / length - 2.0)
                across = (2.0 + 3.0 * strain_limit_multiplier * (length - 1.0)) * (length - 1.0) / length
                along = (2.0 + 3.0 * strain_limit_multiplier * (length * length - 1.0)) / length
            direction = f / length
            block = across * qd.Matrix.identity(gs.qd_float, 3) + along * direction.outer_product(direction)
            if i_axis == 0:
                gradient_u = 2.0 * coefficient * f
            else:
                gradient_v = 2.0 * coefficient * f
            for j, k in qd.static(qd.ndrange(3, 3)):
                hessian[3 * i_axis + j, 3 * i_axis + k] = stretch_stiffness * block[j, k]
    gradient = stretch_stiffness * qd.Matrix.cols([gradient_u, gradient_v]) + (2.0 * shear_stiffness * shear) * (
        qd.Matrix.cols([f_v, f_u])
    )

    # Shear: the exact Hessian 2 (g g^T + I6 S), with S swapping the two columns and g = vec(F S), keeps its two
    # eigenvalues of the eigenspace of S of the sign of I6 and its largest one, along q0
    swap = qd.Matrix.zero(gs.qd_float, 6, 6)
    for j in qd.static(range(3)):
        swap[j, j + 3] = 1.0
        swap[j + 3, j] = 1.0
    g = qd.Vector([f_v[0], f_v[1], f_v[2], f_u[0], f_u[1], f_u[2]])
    norm2 = f_u.norm_sqr() + f_v.norm_sqr()
    eigenvalue_max = 0.5 * (norm2 + qd.sqrt(norm2 * norm2 + 12.0 * shear * shear))
    q0 = shear * (swap @ g) + eigenvalue_max * g
    q0_norm = q0.norm()
    if q0_norm > EPS:
        q0 = q0 / q0_norm
        sign = qd.select(shear >= 0.0, 1.0, -1.0)
        eigenspace = 0.5 * (qd.Matrix.identity(gs.qd_float, 6) + sign * swap)
        projected = eigenspace @ q0
        projected_norm2 = projected.norm_sqr()
        shear_hessian = eigenvalue_max * q0.outer_product(q0)
        if projected_norm2 > EPS:
            shear_hessian += qd.abs(shear) * (eigenspace - projected.outer_product(projected) / projected_norm2)
        hessian += (2.0 * shear_stiffness) * shear_hessian
    return gradient, hessian


@qd.func
def func_triangle_gradient_hessian(rest_inv: qd.types.matrix(2, 2), gradient_F, hessian_F):
    """
    Map the gradient and the Hessian of a function of the deformation gradient of a triangle to its vertices.

    'gradient_F' and 'hessian_F' are those of func_baraff_witkin_gradient_hessian. Returns the 9-vector gradient and
    the 9x9 Hessian in the vertices (x0, x1, x2).
    """
    # Row k holds the derivative of each column of F with respect to vertex k, a multiple of the identity, and the
    # Jacobian of vec(F) in the vertices is its Kronecker product with the identity
    shape = qd.Matrix([[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]]) @ rest_inv
    jacobian = qd.Matrix(
        [
            [qd.select(i % 3 == j % 3, shape[j // 3, i // 3], 0.0) for j in qd.static(range(9))]
            for i in qd.static(range(6))
        ]
    )
    gradient_vec = qd.Vector([gradient_F[i % 3, i // 3] for i in qd.static(range(6))])
    return jacobian.transpose() @ gradient_vec, jacobian.transpose() @ hessian_F @ jacobian


# ------------------------------------------------------------------------------------
# ------------------------------------- bending --------------------------------------
# ------------------------------------------------------------------------------------

# A hinge is two triangles sharing an edge, its vertices given as the rows of a 4x3 matrix: the quadratic model takes
# them as (edge0, edge1, wing0, wing1) and the dihedral model as (wing0, edge0, edge1, wing1). Gradients and Hessians
# are over the stacked vertex positions of the hinge.


@qd.func
def func_quadratic_bending_energy(
    positions: qd.types.matrix(4, 3), Q: qd.types.matrix(4, 4), stiffness: float
) -> float:
    """
    Compute the quadratic bending energy of a hinge, 1/2 k sum_ab Q_ab x_a . x_b.

    'Q' is the bending form of the hinge at rest (Bergou et al. 2006), symmetric and positive semi-definite, of zero row
    sums, and 'stiffness' k its flexural rigidity.
    """
    # Zero row sums make the energy translation invariant: positions relative to the first vertex give the same value
    # without the cancellation of absolute coordinates, which buries the decrease of the line search in rounding.
    positions_rel = qd.Matrix(
        [[positions[i, k] - positions[0, k] for k in qd.static(range(3))] for i in qd.static(range(4))]
    )
    return 0.5 * stiffness * (positions_rel * (Q @ positions_rel)).sum()


@qd.func
def func_quadratic_bending_gradient_hessian(
    positions: qd.types.matrix(4, 3), Q: qd.types.matrix(4, 4), stiffness: float
):
    """
    Compute the gradient and the Hessian of the quadratic bending energy of a hinge (see func_quadratic_bending_energy).

    The Hessian is k Q_ab times the identity, constant and positive semi-definite. Returns the 12-vector gradient and
    the 12x12 Hessian.
    """
    # Relative positions, as in func_quadratic_bending_energy
    positions_rel = qd.Matrix(
        [[positions[i, k] - positions[0, k] for k in qd.static(range(3))] for i in qd.static(range(4))]
    )
    vertices_gradient = stiffness * (Q @ positions_rel)
    gradient = qd.Vector([vertices_gradient[i // 3, i % 3] for i in qd.static(range(12))])
    hessian = qd.Matrix(
        [
            [qd.select(i % 3 == j % 3, stiffness * Q[i // 3, j // 3], 0.0) for j in qd.static(range(12))]
            for i in qd.static(range(12))
        ]
    )
    return gradient, hessian


@qd.func
def func_dihedral_angle(
    x0: qd.types.vector(3), x1: qd.types.vector(3), x2: qd.types.vector(3), x3: qd.types.vector(3)
) -> float:
    """
    Compute the signed dihedral angle of a hinge (wing0, edge0, edge1, wing1), zero for a flat hinge.

    It is the angle between the normals (x1 - x0) x (x2 - x0) and (x2 - x3) x (x1 - x3) of its two triangles, signed
    about the edge x1 -> x2.
    """
    normal0 = (x1 - x0).cross(x2 - x0)
    normal1 = (x2 - x3).cross(x1 - x3)
    edge = x2 - x1
    return qd.atan2(normal0.cross(normal1).dot(edge) / edge.norm(), normal0.dot(normal1))


@qd.func
def func_dihedral_bending_energy(angle: float, rest_angle: float, weight: float, stiffness: float) -> float:
    """
    Compute the dihedral bending energy of a hinge, w k (theta - theta_rest)^2.

    'weight' w is the squared rest length of the edge over the rest area of the two triangles, and 'stiffness' k half the
    flexural rigidity.
    """
    deviation = angle - rest_angle
    return weight * stiffness * deviation * deviation


@qd.func
def _func_projected_spectrum(F: qd.types.matrix(4, 4), eigenvalues: qd.types.vector(4)):
    """
    Project a symmetric 4x4 matrix of known eigenvalues onto its positive eigenspace, without its eigenvectors.

    The projection sums its positive eigenvalues times the spectral projector onto their eigenspace, the Lagrange
    product of (F - lam_j I) / (lam_i - lam_j) over the other eigenvalues. Eigenvalues below 1e-7 of the largest one
    count as zero, which keeps the denominators large, and two positive eigenvalues closer than 1e-9 of it share one
    projector, onto their joint eigenspace.
    """
    RATIO_ZERO = qd.static(1e-7)
    RATIO_MERGED = qd.static(1e-9)
    scale = qd.abs(eigenvalues).max()
    identity = qd.Matrix.identity(gs.qd_float, 4)
    projected = qd.Matrix.zero(gs.qd_float, 4, 4)
    for i in qd.static(range(4)):
        is_kept = eigenvalues[i] > RATIO_ZERO * scale
        for j in qd.static(range(i)):
            if eigenvalues[j] > RATIO_ZERO * scale and qd.abs(eigenvalues[i] - eigenvalues[j]) < RATIO_MERGED * scale:
                is_kept = False
        if is_kept:
            projector = identity
            for j in qd.static(range(4)):
                if qd.static(j != i):
                    is_merged = (
                        eigenvalues[j] > RATIO_ZERO * scale
                        and qd.abs(eigenvalues[i] - eigenvalues[j]) < RATIO_MERGED * scale
                    )
                    if not is_merged:
                        projector = projector @ (F - eigenvalues[j] * identity) / (eigenvalues[i] - eigenvalues[j])
            projected += eigenvalues[i] * projector
    return projected


@qd.func
def func_dihedral_bending_gradient_hessian(
    positions: qd.types.matrix(4, 3), rest_angle: float, weight: float, stiffness: float
):
    """
    Compute the gradient and the positive semi-definite Hessian of the dihedral bending energy of a hinge (see
    func_dihedral_bending_energy).

    The Hessian is projected analytically (Wang et al. 2023, "Stable Discrete Bending by Analytic Eigensystem"): it is
    Z F Z^T in an invariant subspace Z spanned by eight products of a vertex weight vector with a geometric direction,
    its 8x8 core F being two 4x4 blocks of closed-form eigenvalues, each clamped to its positive part. A degenerate hinge
    (zero edge, flat triangle, or wing on the edge) gives a zero gradient and Hessian. Returns the 12-vector gradient
    and the 12x12 Hessian in the order of the rows of 'positions'.
    """
    EPS = qd.static(1e-30)
    gradient = qd.Vector.zero(gs.qd_float, 12)
    hessian = qd.Matrix.zero(gs.qd_float, 12, 12)
    # The analysis takes the edge first then the wings: (edge0, edge1, wing0, wing1)
    p0 = qd.Vector([positions[1, 0], positions[1, 1], positions[1, 2]])
    p1 = qd.Vector([positions[2, 0], positions[2, 1], positions[2, 2]])
    p2 = qd.Vector([positions[0, 0], positions[0, 1], positions[0, 2]])
    p3 = qd.Vector([positions[3, 0], positions[3, 1], positions[3, 2]])
    edge = p1 - p0
    length = edge.norm()
    normal0 = edge.cross(p2 - p0)
    normal1 = (p0 - p1).cross(p3 - p1)
    area0 = normal0.norm()
    area1 = normal1.norm()
    altitude = (p2 - p0) - (p2 - p0).dot(edge) / (length * length) * edge
    altitude_norm = altitude.norm()
    if length > EPS and area0 > EPS and area1 > EPS and altitude_norm > EPS:
        direction = edge / length
        unit_normal0 = normal0 / area0
        unit_normal1 = normal1 / area1
        inward = -altitude / altitude_norm
        coord0 = (p2 - p0).dot(direction) / length
        coord1 = (p3 - p0).dot(direction) / length
        # The angle of the analysis is pi minus the dihedral angle, pi for a flat hinge
        angle = math.pi - qd.atan2(normal0.cross(normal1).dot(direction), normal0.dot(normal1))
        rest = math.pi - rest_angle
        stiffness_angle = 2.0 * stiffness
        moment = stiffness_angle * (angle - rest)

        weights0 = qd.Vector([coord0 - 1.0, -coord0, 1.0, 0.0]) * (length / area0)
        weights1 = qd.Vector([coord1 - 1.0, -coord1, 0.0, 1.0]) * (length / area1)
        weights_edge = qd.Vector([1.0, -1.0, 0.0, 0.0]) / length
        sin = qd.sin(angle)
        cos = qd.cos(angle)
        sin2 = 2.0 * sin * cos
        cos2 = cos * cos - sin * sin
        core0 = qd.Matrix(
            [
                [stiffness_angle, -stiffness_angle * cos, stiffness_angle * sin, moment],
                [
                    -stiffness_angle * cos,
                    stiffness_angle * cos * cos - moment * sin2,
                    -0.5 * stiffness_angle * sin2 - moment * cos2,
                    0.0,
                ],
                [
                    stiffness_angle * sin,
                    -0.5 * stiffness_angle * sin2 - moment * cos2,
                    stiffness_angle * sin * sin + moment * sin2,
                    0.0,
                ],
                [moment, 0.0, 0.0, 0.0],
            ]
        )
        core1 = moment * qd.Matrix(
            [
                [-sin * cos, sin * sin, -cos, 1.0],
                [sin * sin, sin * cos, sin, 0.0],
                [-cos, sin, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
            ]
        )
        radius = qd.sqrt(stiffness_angle * stiffness_angle + moment * moment)
        root0 = qd.sqrt(sin * sin + 4.0 * (1.0 - cos))
        root1 = qd.sqrt(sin * sin + 4.0 * (1.0 + cos))
        eigenvalues0 = qd.Vector([stiffness_angle + radius, moment, stiffness_angle - radius, -moment])
        eigenvalues1 = 0.5 * moment * qd.Vector([sin + root0, root1 - sin, sin - root0, -sin - root1])

        # The subspace of each block, column c being the product of a weight vector over the vertices with a direction
        for i_block in range(2):
            core = core0
            eigenvalues = eigenvalues0
            if i_block == 1:
                core = core1
                eigenvalues = eigenvalues1
            projected = _func_projected_spectrum(core, eigenvalues)
            basis = qd.Matrix.zero(gs.qd_float, 12, 4)
            for i_v, d in qd.static(qd.ndrange(4, 3)):
                if i_block == 0:
                    basis[3 * i_v + d, 0] = weights0[i_v] * unit_normal0[d]
                    basis[3 * i_v + d, 1] = weights1[i_v] * unit_normal0[d]
                    basis[3 * i_v + d, 2] = weights1[i_v] * inward[d]
                    basis[3 * i_v + d, 3] = weights0[i_v] * inward[d]
                else:
                    basis[3 * i_v + d, 0] = weights_edge[i_v] * unit_normal0[d]
                    basis[3 * i_v + d, 1] = weights_edge[i_v] * inward[d]
                    basis[3 * i_v + d, 2] = weights1[i_v] * direction[d]
                    basis[3 * i_v + d, 3] = weights0[i_v] * direction[d]
            hessian += basis @ projected @ basis.transpose()
        hessian *= weight
        for i_v, d in qd.static(qd.ndrange(4, 3)):
            gradient[3 * i_v + d] = (
                weight * moment * (weights0[i_v] * unit_normal0[d] + weights1[i_v] * unit_normal1[d])
            )

    # Back from (edge0, edge1, wing0, wing1) to the order of the rows of 'positions'
    ORDER = qd.static((2, 0, 1, 3))
    gradient_out = qd.Vector([gradient[3 * ORDER[i // 3] + i % 3] for i in qd.static(range(12))])
    hessian_out = qd.Matrix(
        [
            [hessian[3 * ORDER[i // 3] + i % 3, 3 * ORDER[j // 3] + j % 3] for j in qd.static(range(12))]
            for i in qd.static(range(12))
        ]
    )
    return gradient_out, hessian_out

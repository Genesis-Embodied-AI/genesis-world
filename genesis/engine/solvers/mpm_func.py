"""Layer-0 mathematics and Layer-1 stages of the implicit material point method (MPM) solver."""

import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu
from genesis.engine.numerics.bcoo_matrix import func_add_triplet, func_write_triplet
from genesis.engine.numerics.fsr_reduce import add_by_key, head_segmented_reduce_add, max_by_key
from genesis.engine.numerics.newton_control import NewtonControl
from genesis.engine.numerics.open_addressing_hash import (
    EMPTY_KEY,
    func_insert_hash,
    func_lookup_hash,
    func_reset_hash,
)
from genesis.engine.numerics.soft_contact import soft_row
from genesis.utils.array_class import ErrorCode, LinksState
from genesis.utils.misc import qd_debug_asserts, qd_isfinite

from .mpm_data import (
    BLOCK_CELLS,
    PYRAMID_ROWS,
    REDUCE_BLOCK,
    STENCIL_BLOCKS,
    MpmGridScratch,
    MpmInfo,
    MpmMaterialTable,
    MpmNewtonScratch,
    MpmParticles,
    MpmParticlesInfo,
    MpmParticlesNG,
    MpmWallPairs,
)
from .newton_contact import append_pair
from .newton_linear import LinearBlock, NewtonLinearScratch, discard_block

# Biased signed range of block coordinates packed into 14 bits per axis
COORD_BIAS = 8192
# Range of the cell coordinates of a particle whose block and positive neighbour blocks all pack into the key
CELL_MIN = -4 * COORD_BIAS
CELL_END = 4 * (COORD_BIAS - 1)
# Environments a particle key holds, in the bits above the block key and the 6-bit cell index
MAX_KEY_ENVS = 1 << 16

# Energy families of the material table
FAMILY_FIXED_COROTATED = 0
FAMILY_STABLE_NEOHOOKEAN = 1
FAMILY_EXTENDED_HENCKY = 2
FAMILY_LIQUID = 3

# Return maps of the material table, applied in G2P
PLASTIC_NONE = 0
PLASTIC_VON_MISES = 1
PLASTIC_CLAMP = 2
PLASTIC_SNOW = 3
PLASTIC_SAND = 4
PLASTIC_LIQUID = 5

# Columns of the parameter-major material table
PARAM_MU = 0
PARAM_LAM = 1
# Stiffness of the muscle fibre term, zero for other materials
PARAM_FIBRE = 2
PARAM_YIELD_LOWER = 3
PARAM_YIELD_HIGHER = 4
PARAM_YIELD_STRESS = 5
PARAM_SAND_ALPHA = 6
N_MATERIAL_PARAMS = 7

# Singular value below which the extended Hencky energy continues log(sigma) by its tangent line
HENCKY_SIGMA_MIN = 0.05
# Singular value below which the fixed corotated and Stable Neo-Hookean energies resist further compression, and the
# stiffness of that resistance in units of the Young's modulus (Stomakhin et al. 2012, see compression_penalty)
EXTENSION_SIGMA_MIN = 0.4
EXTENSION_STIFFNESS = 20.0
# Singular values below which the return maps clamp before taking a log, those of the explicit materials
VON_MISES_SIGMA_MIN = 0.05
SAND_SIGMA_MIN = 1e-4
# Range of the volume ratio the liquid return map keeps
LIQUID_J_MIN = 0.2
# Fraction of the step to the first zero volume of a liquid particle along the Newton direction, beyond which the line
# search does not start, so that every trial state keeps the volume ratio of the liquid positive
INVERSION_STEP_SLACK = 0.9
# Hardening exponent of snow, mu and lam scaled by exp(SNOW_HARDENING (1 - Jp))
SNOW_HARDENING = 10.0

# Pairs of principal axes, in the order of the twist and flip eigenvalues
_PAIRS = ((0, 1), (0, 2), (1, 2))

# Upper-triangular node pairs (a <= b) of the 8-node stencil, in the order of a task's 36 Hessian blocks
_STENCIL_PAIR_A = tuple(a for a in range(8) for b in range(a, 8))
_STENCIL_PAIR_B = tuple(b for a in range(8) for b in range(a, 8))

# Half extent of the inverted box of a leaf that must meet no query of a coupling's tree, finite so that its center
# gets a Morton code
EMPTY_BOX_HALF = 1e10

# Lanes of a warp
_WARP = 32


# ============================================== Layer 0: shape and keys ==============================================


@qd.func
def node_offset(a: qd.template()):
    """Offset of the a-th node of a cell from its lower corner, bits 2, 1 and 0 of a giving the x, y and z offsets."""
    return qd.Vector([(a >> 2) & 1, (a >> 1) & 1, a & 1], dt=gs.qd_int)


@qd.func
def block_of(cell):
    """Block coordinates of a cell or node coordinate (floor division by the block size, valid for negatives)."""
    return cell >> 2


@qd.func
def cell_local(cell):
    """Index of a cell or node coordinate inside its block."""
    return ((cell[0] & 3) << 4) | ((cell[1] & 3) << 2) | (cell[2] & 3)


@qd.func
def local_offset(local: int):
    """Inverse of cell_local: offset of a local index inside its block."""
    return qd.Vector([(local >> 4) & 3, (local >> 2) & 3, local & 3], dt=gs.qd_int)


@qd.func
def pack_block(i_b: int, block):
    """Key of a block of an environment: environment index from bit 42, then 14 biased bits per block coordinate."""
    return (
        (qd.u64(i_b) << qd.u64(42))
        | (qd.u64(block[0] + COORD_BIAS) << qd.u64(28))
        | (qd.u64(block[1] + COORD_BIAS) << qd.u64(14))
        | qd.u64(block[2] + COORD_BIAS)
    )


@qd.func
def unpack_block(key: qd.u64):
    """Environment and block coordinates of a block key."""
    i_b = gs.qd_int(key >> qd.u64(42))
    block = qd.Vector(
        [
            gs.qd_int((key >> qd.u64(28)) & qd.u64(0x3FFF)) - COORD_BIAS,
            gs.qd_int((key >> qd.u64(14)) & qd.u64(0x3FFF)) - COORD_BIAS,
            gs.qd_int(key & qd.u64(0x3FFF)) - COORD_BIAS,
        ],
        dt=gs.qd_int,
    )
    return i_b, block


@qd.func
def particle_key(i_b: int, cell):
    """Sort key of a particle: its block key followed by the 6-bit local index of its cell."""
    return (pack_block(i_b, block_of(cell)) << qd.u64(6)) | qd.u64(cell_local(cell))


@qd.func
def flag_env_error(i_b: int, grid: MpmGridScratch, code: qd.template()):
    """Raise an error code for one environment, which the solver reports at the beginning of the next step."""
    qd.atomic_or(grid.errors.errno[i_b], qd.static(int(code)))


@qd.func
def flag_all_envs_error(grid: MpmGridScratch, mpm_info: MpmInfo, code: qd.template()):
    """Raise an error code for every environment, for a failure of the solve they share."""
    for i_b in range(mpm_info.n_envs[()]):
        flag_env_error(i_b, grid, code)


@qd.func
def cofactor(v: qd.types.vector(3)):
    """Products of the two other components of every component, the gradient of v0 v1 v2."""
    return qd.Vector([v[1] * v[2], v[0] * v[2], v[0] * v[1]], dt=gs.qd_float)


@qd.func
def cofactor_jacobian(v: qd.types.vector(3)):
    """Jacobian of cofactor(v), the third component off the diagonal and zero on it."""
    return qd.Matrix(
        [
            [0.0, v[2], v[1]],
            [v[2], 0.0, v[0]],
            [v[1], v[0], 0.0],
        ],
        dt=gs.qd_float,
    )


@qd.func
def diagonal(v: qd.types.vector(3)):
    return qd.Matrix(
        [
            [v[0], 0.0, 0.0],
            [0.0, v[1], 0.0],
            [0.0, 0.0, v[2]],
        ],
        dt=gs.qd_float,
    )


@qd.func
def diagonal_of(matrix: qd.types.matrix(3, 3)):
    return qd.Vector([matrix[0, 0], matrix[1, 1], matrix[2, 2]], dt=gs.qd_float)


@qd.func
def q1_axes(offset, fx: qd.types.vector(3), inv_dx: float):
    """Weights and weight derivatives along the three axes of the node of offset `offset` in {0, 1}^3 of a cell, at the
    point of offset `fx` from its lower corner in cell units."""
    offset_float = offset.cast(gs.qd_float)
    return offset_float * fx + (1.0 - offset_float) * (1.0 - fx), (2.0 * offset_float - 1.0) * inv_dx


@qd.func
def q1_node_weight(offset, fx: qd.types.vector(3), inv_dx: float):
    """Trilinear weight and weight gradient of the node of offset `offset` in {0, 1}^3 of a cell, at the point of offset
    `fx` from its lower corner in cell units, in [0, 1)^3, `inv_dx` being the inverse cell size."""
    axis_w, axis_dw = q1_axes(offset, fx, inv_dx)
    return axis_w[0] * axis_w[1] * axis_w[2], axis_dw * cofactor(axis_w)


@qd.func
def q1_stencil(fx: qd.types.vector(3), inv_dx: float):
    """Trilinear weights and weight gradients of the 8 nodes of a cell (see q1_node_weight), node a in row a."""
    w = qd.Vector.zero(gs.qd_float, 8)
    grad_w = qd.Matrix.zero(gs.qd_float, 8, 3)
    for a in qd.static(range(8)):
        w_a, grad_w_a = q1_node_weight(node_offset(a), fx, inv_dx)
        w[a] = w_a
        for d in qd.static(range(3)):
            grad_w[a, d] = grad_w_a[d]
    return w, grad_w


@qd.func
def q1_weight_hessian(fx: qd.types.vector(3), inv_dx: float, a: qd.template()):
    """Hessian of the trilinear weight of node `a` of a cell with respect to the particle position (see q1_stencil):
    off the diagonal, the product of the axis derivatives along i and j and the axis weight along the third axis."""
    axis_w, axis_dw = q1_axes(node_offset(a), fx, inv_dx)
    return axis_dw.outer_product(axis_dw) * cofactor_jacobian(axis_w)


# ============================================= Layer 0: gather lanes ==============================================


def gather_lanes() -> int:
    """Lanes that gather the particles of one node: the 32 lanes of a warp on a GPU, one thread on a CPU, which has no
    lanes to share a node over."""
    return 1 if gs.backend == gs.cpu else _WARP


@qd.func
def gather_lane():
    i_lane = gs.qd_int(0)
    if qd.static(gather_lanes() > 1):
        i_lane = gs.qd_int(qd.simt.subgroup.invocation_id()) & (_WARP - 1)
    return i_lane


@qd.func
def lanes_sum(value):
    """Sum over the gather lanes by a fixed butterfly, so every lane holds the same, reproducible total."""
    if qd.static(gather_lanes() > 1):
        for level in qd.static(range(5)):
            value = value + qd.simt.subgroup.shuffle_xor(value, qd.u32(16 >> level))
    return value


@qd.func
def lanes_max(value):
    if qd.static(gather_lanes() > 1):
        for level in qd.static(range(5)):
            value = qd.max(value, qd.simt.subgroup.shuffle_xor(value, qd.u32(16 >> level)))
    return value


@qd.func
def cell_segment(i_b: int, cell, i_lane: int, grid: MpmGridScratch):
    """Range of the sorted particles of cell `cell` of an environment, looked up by lane 0 and broadcast to the gather
    lanes."""
    i_ps_start = 0
    i_ps_end = 0
    if i_lane == 0:
        i_blk = func_lookup_hash(pack_block(i_b, block_of(cell)), grid.hash)
        if i_blk >= 0:
            i_cell_slot = i_blk * BLOCK_CELLS + cell_local(cell)
            i_ps_start = grid.cells_segment_start[i_cell_slot]
            i_ps_end = grid.cells_segment_end[i_cell_slot]
    if qd.static(gather_lanes() > 1):
        i_ps_start = qd.simt.subgroup.shuffle(i_ps_start, qd.u32(0))
        i_ps_end = qd.simt.subgroup.shuffle(i_ps_end, qd.u32(0))
    return i_ps_start, i_ps_end


# ============================================== Layer 0: constitution ================================================


@qd.func
def compression_penalty(sigma: qd.types.vector(3), mu: float, lam: float):
    """Compression penalty of the fixed corotated and Stable Neo-Hookean energies, with its gradient and Hessian."""
    sigma_min = gs.qd_float(EXTENSION_SIGMA_MIN)
    young_modulus = gs.qd_float(0.0)
    if lam + mu > 0.0:
        young_modulus = mu * (3.0 * lam + 2.0 * mu) / (lam + mu)
    stiffness = EXTENSION_STIFFNESS * young_modulus
    deficit = qd.min(sigma - sigma_min, 0.0)
    return (
        0.5 * stiffness * deficit.dot(deficit),
        stiffness * deficit,
        stiffness * (sigma < sigma_min).cast(gs.qd_float),
    )


@qd.func
def stable_neohookean_parameters(mu: float, lam: float):
    """mu_s, lam_s, alpha and the rest energy of the Stable Neo-Hookean energy (see stable_neohookean_principal)."""
    mu_s = 4.0 / 3.0 * mu
    lam_s = lam + 5.0 / 6.0 * mu
    alpha = gs.qd_float(1.0)
    if lam_s > 0.0:
        alpha = 1.0 + mu_s / lam_s - mu_s / (4.0 * lam_s)
    rest = 0.5 * lam_s * (1.0 - alpha) ** 2 - 0.5 * mu_s * qd.log(4.0)
    return mu_s, lam_s, alpha, rest


@qd.func
def extended_hencky_principal(sigma: qd.types.vector(3), mu: float, lam: float):
    """Energy density, gradient g = dpsi/dsigma and Hessian A = d2psi/dsigma2 of the extended Hencky energy."""
    sigma_min = qd.static(HENCKY_SIGMA_MIN)
    log_stretch = qd.Vector.zero(gs.qd_float, 3)
    dlog_stretch = qd.Vector.zero(gs.qd_float, 3)
    d2log_stretch = qd.Vector.zero(gs.qd_float, 3)
    for k in qd.static(range(3)):
        if sigma[k] >= sigma_min:
            log_stretch[k] = qd.log(sigma[k])
            dlog_stretch[k] = 1.0 / sigma[k]
            d2log_stretch[k] = -1.0 / (sigma[k] * sigma[k])
        else:
            log_stretch[k] = qd.log(sigma_min) + (sigma[k] - sigma_min) / sigma_min
            dlog_stretch[k] = 1.0 / sigma_min
    trace = log_stretch.sum()
    tau = 2.0 * mu * log_stretch + lam * trace
    A = lam * dlog_stretch.outer_product(dlog_stretch) + diagonal(
        2.0 * mu * dlog_stretch * dlog_stretch + tau * d2log_stretch
    )
    return mu * log_stretch.dot(log_stretch) + 0.5 * lam * trace * trace, tau * dlog_stretch, A


@qd.func
def stable_neohookean_principal(sigma: qd.types.vector(3), mu: float, lam: float):
    """Energy density, gradient and Hessian of the Stable Neo-Hookean energy (Smith et al. 2018, Eq. 14) in the signed
    singular values, defined for inverted states."""
    mu_s, lam_s, alpha, rest = stable_neohookean_parameters(mu, lam)
    J = sigma[0] * sigma[1] * sigma[2]
    cof = cofactor(sigma)
    I_C = sigma.dot(sigma)
    barrier = 1.0 / (I_C + 1.0)
    psi = 0.5 * mu_s * (I_C - 3.0) + 0.5 * lam_s * (J - alpha) ** 2 - 0.5 * mu_s * qd.log(I_C + 1.0) - rest
    g = mu_s * (1.0 - barrier) * sigma + lam_s * (J - alpha) * cof
    A = mu_s * (1.0 - barrier) * qd.Matrix.identity(gs.qd_float, 3)
    A += 2.0 * mu_s * barrier * barrier * sigma.outer_product(sigma)
    A += lam_s * (cof.outer_product(cof) + (J - alpha) * cofactor_jacobian(sigma))
    return psi, g, A


@qd.func
def liquid_volume_terms(J: float, lam: float, band: float):
    """Volume term of the liquid energy density and its first and second derivatives in J."""
    psi = gs.qd_float(0.0)
    dpsi = gs.qd_float(0.0)
    d2psi = gs.qd_float(0.0)
    x = 1.0 - J
    if x >= band and x > 0.0:
        psi = 0.25 * lam * (J * J - 1.0 - 2.0 * qd.log(J))
        dpsi = 0.5 * lam * (J - 1.0 / J)
        d2psi = 0.5 * lam * (1.0 + 1.0 / (J * J))
    elif x > 0.0:
        J_band = 1.0 - band
        value = 0.25 * lam * (J_band * J_band - 1.0 - 2.0 * qd.log(J_band))
        # Slope and curvature in x at x = band, scaled by band and band^2
        slope = -0.5 * lam * (J_band - 1.0 / J_band) * band
        curvature = 0.5 * lam * (1.0 + 1.0 / (J_band * J_band)) * band * band
        c3 = 10.0 * value - 4.0 * slope + 0.5 * curvature
        c4 = 7.0 * slope - 15.0 * value - curvature
        c5 = 0.5 * curvature - 3.0 * slope + 6.0 * value
        t = x / band
        psi = t * t * t * (c3 + t * (c4 + t * c5))
        dpsi = -t * t * (3.0 * c3 + t * (4.0 * c4 + t * 5.0 * c5)) / band
        d2psi = t * (6.0 * c3 + t * (12.0 * c4 + t * 20.0 * c5)) / (band * band)
    return psi, dpsi, d2psi


@qd.func
def liquid_principal(sigma: qd.types.vector(3), mu: float, lam: float, band: float):
    """Energy density, gradient and Hessian of the liquid in the signed singular values."""
    cof = cofactor(sigma)
    psi_volume, dpsi, d2psi = liquid_volume_terms(sigma[0] * sigma[1] * sigma[2], lam, band)
    psi = mu * (sigma - 1.0).dot(sigma - 1.0) + psi_volume
    g = 2.0 * mu * (sigma - 1.0) + dpsi * cof
    A = 2.0 * mu * qd.Matrix.identity(gs.qd_float, 3) + d2psi * cof.outer_product(cof) + dpsi * cofactor_jacobian(sigma)
    return psi, g, A


@qd.func
def det_cubic(F0: qd.types.matrix(3, 3), dF: qd.types.matrix(3, 3)):
    """Coefficients c0, c1, c2, c3 of det(F0 + t dF) in powers of t, by the multilinear expansion over the columns."""
    a = [F0[:, c] for c in qd.static(range(3))]
    b = [dF[:, c] for c in qd.static(range(3))]
    c0 = a[0].dot(a[1].cross(a[2]))
    c1 = b[0].dot(a[1].cross(a[2])) + a[0].dot(b[1].cross(a[2])) + a[0].dot(a[1].cross(b[2]))
    c2 = a[0].dot(b[1].cross(b[2])) + b[0].dot(a[1].cross(b[2])) + b[0].dot(b[1].cross(a[2]))
    c3 = b[0].dot(b[1].cross(b[2]))
    return c0, c1, c2, c3


@qd.func
def cubic_first_root(c0: float, c1: float, c2: float, c3: float, t_max: float):
    """Smallest root in (0, t_max] of c0 + c1 t + c2 t^2 + c3 t^3 given c0 > 0, or 2 t_max where it stays positive."""
    # Extrema in (0, t_max), the absent ones at t_max, in ascending order
    t_low = t_max
    t_high = t_max
    a = 3.0 * c3
    b = 2.0 * c2
    c = c1
    if qd.abs(a) > 0.0:
        discriminant = b * b - 4.0 * a * c
        if discriminant > 0.0:
            root_sqrt = qd.sqrt(discriminant)
            r1 = (-b - root_sqrt) / (2.0 * a)
            r2 = (-b + root_sqrt) / (2.0 * a)
            if r1 > 0.0 and r1 < t_max:
                t_low = r1
            if r2 > 0.0 and r2 < t_max:
                t_high = r2
    elif qd.abs(b) > 0.0:
        r = -c / b
        if r > 0.0 and r < t_max:
            t_low = r
    if t_high < t_low:
        t_swap = t_low
        t_low = t_high
        t_high = t_swap
    breakpoints = qd.Vector([t_low, t_high, t_max], dt=gs.qd_float)
    root = 2.0 * t_max
    is_found = False
    lo = gs.qd_float(0.0)
    for k in qd.static(range(3)):
        hi = breakpoints[k]
        g_lo = c0 + lo * (c1 + lo * (c2 + lo * c3))
        g_hi = c0 + hi * (c1 + hi * (c2 + hi * c3))
        if not is_found and g_lo > 0.0 and g_hi <= 0.0:
            is_found = True
            left = lo
            right = hi
            for _ in range(64):
                mid = 0.5 * (left + right)
                if c0 + mid * (c1 + mid * (c2 + mid * c3)) > 0.0:
                    left = mid
                else:
                    right = mid
            root = 0.5 * (left + right)
        lo = hi
    return root


@qd.func
def fixed_corotated_principal(sigma: qd.types.vector(3), mu: float, lam: float):
    """Energy density, gradient and Hessian of the fixed corotated energy in the signed singular values."""
    J = sigma[0] * sigma[1] * sigma[2]
    cof = cofactor(sigma)
    psi = mu * (sigma - 1.0).dot(sigma - 1.0) + 0.5 * lam * (J - 1.0) ** 2
    g = 2.0 * mu * (sigma - 1.0) + lam * (J - 1.0) * cof
    A = 2.0 * mu * qd.Matrix.identity(gs.qd_float, 3)
    A += lam * (cof.outer_product(cof) + (J - 1.0) * cofactor_jacobian(sigma))
    return psi, g, A


@qd.func
def isotropic_pair_eigenvalues_exact(sigma: qd.types.vector(3), g: qd.types.vector(3), A: qd.types.matrix(3, 3)):
    """Twist and flip eigenvalues of an isotropic energy for the axis pairs (0,1), (0,2), (1,2)."""
    eps = gs.qd_float(1e-6)
    twist = qd.Vector.zero(gs.qd_float, 3)
    flip = qd.Vector.zero(gs.qd_float, 3)
    for k in qd.static(range(3)):
        i, j = qd.static(_PAIRS[k])
        denominator_flip = sigma[i] - sigma[j]
        if qd.abs(denominator_flip) > eps:
            flip[k] = (g[i] - g[j]) / denominator_flip
        else:
            flip[k] = A[i, i] - A[i, j]
        denominator_twist = sigma[i] + sigma[j]
        if qd.abs(denominator_twist) < eps:
            denominator_twist = eps if denominator_twist >= 0.0 else -eps
        twist[k] = (g[i] + g[j]) / denominator_twist
    return twist, flip


@qd.func
def isotropic_first_piola_differential(
    U: qd.types.matrix(3, 3),
    V: qd.types.matrix(3, 3),
    sigma: qd.types.vector(3),
    g: qd.types.vector(3),
    A: qd.types.matrix(3, 3),
    dF: qd.types.matrix(3, 3),
):
    """Unprojected differential dP = (dP/dF) : dF of the first Piola-Kirchhoff stress of an isotropic energy at F = U
    diag(sigma) V^T, with `g` and `A` its principal gradient and Hessian."""
    dF_hat = U.transpose() @ dF @ V
    twist, flip = isotropic_pair_eigenvalues_exact(sigma, g, A)
    dP_hat = diagonal(A @ qd.Vector([dF_hat[0, 0], dF_hat[1, 1], dF_hat[2, 2]], dt=gs.qd_float))
    for k in qd.static(range(3)):
        i, j = qd.static(_PAIRS[k])
        alpha = 0.5 * (flip[k] + twist[k])
        beta = 0.5 * (flip[k] - twist[k])
        dP_hat[i, j] = alpha * dF_hat[i, j] + beta * dF_hat[j, i]
        dP_hat[j, i] = beta * dF_hat[i, j] + alpha * dF_hat[j, i]
    return U @ dP_hat @ V.transpose()


@qd.func
def isotropic_projected_modes(
    sigma: qd.types.vector(3), g: qd.types.vector(3), A: qd.types.matrix(3, 3), is_exact: qd.template() = False
):
    """Symmetric positive definite (SPD) projection of the principal Hessian of an isotropic energy, as a scaling block
    and per-pair modes."""
    S = 0.5 * (A + A.transpose())
    twist, flip = isotropic_pair_eigenvalues_exact(sigma, g, A)
    if qd.static(not is_exact):
        eigenvalues, Q = qd.sym_eig(S, gs.qd_float)
        S = qd.Matrix.zero(gs.qd_float, 3, 3)
        for k in qd.static(range(3)):
            eigenvector = Q[:, k]
            S += qd.max(eigenvalues[k], 0.0) * eigenvector.outer_product(eigenvector)
        twist = qd.max(twist, 0.0)
        flip = qd.max(flip, 0.0)
    return S, 0.5 * (flip + twist), 0.5 * (flip - twist)


@qd.func
def isotropic_pair_block(
    r_a: qd.types.vector(3),
    r_b: qd.types.vector(3),
    S: qd.types.matrix(3, 3),
    alpha: qd.types.vector(3),
    beta: qd.types.vector(3),
):
    """Hessian block Z_ab of a node pair of the stencil in the V frame, from the modes of isotropic_projected_modes."""
    Z = S * r_a.outer_product(r_b)
    for k in qd.static(range(3)):
        i, j = qd.static(_PAIRS[k])
        Z[i, i] += alpha[k] * r_a[j] * r_b[j]
        Z[j, j] += alpha[k] * r_a[i] * r_b[i]
        Z[i, j] += beta[k] * r_a[j] * r_b[i]
        Z[j, i] += beta[k] * r_a[i] * r_b[j]
    return Z


# ========================================= Layer 1: per-item stencil access ==========================================


@qd.func
def material_param(i_mat: int, materials: MpmMaterialTable, mpm_info: MpmInfo, param: qd.template()):
    return materials.material_table[param * mpm_info.n_materials[()] + i_mat]


@qd.func
def material_params(i_mat: int, Jp: float, materials: MpmMaterialTable, mpm_info: MpmInfo):
    """Lame parameters of a material slot for a particle, hardened by its plastic state for snow."""
    mu = material_param(i_mat, materials, mpm_info, PARAM_MU)
    lam = material_param(i_mat, materials, mpm_info, PARAM_LAM)
    if materials.materials_plastic[i_mat] == PLASTIC_SNOW:
        hardening = qd.exp(qd.static(SNOW_HARDENING) * (1.0 - Jp))
        mu *= hardening
        lam *= hardening
    return mu, lam


@qd.func
def elastic_principal(
    i_mat: int, sigma: qd.types.vector(3), mu: float, lam: float, materials: MpmMaterialTable, mpm_info: MpmInfo
):
    """Energy density, principal gradient and Hessian of the energy family of a material slot."""
    psi = gs.qd_float(0.0)
    g = qd.Vector.zero(gs.qd_float, 3)
    A = qd.Matrix.zero(gs.qd_float, 3, 3)
    family = materials.materials_family[i_mat]
    if family == FAMILY_EXTENDED_HENCKY:
        psi, g, A = extended_hencky_principal(sigma, mu, lam)
    elif family == FAMILY_LIQUID:
        psi, g, A = liquid_principal(sigma, mu, lam, mpm_info.liquid_band[()])
    else:
        if family == FAMILY_STABLE_NEOHOOKEAN:
            psi, g, A = stable_neohookean_principal(sigma, mu, lam)
        else:
            psi, g, A = fixed_corotated_principal(sigma, mu, lam)
        penalty, g_penalty, A_penalty = compression_penalty(sigma, mu, lam)
        psi += penalty
        g += g_penalty
        A += diagonal(A_penalty)
    return psi, g, A


@qd.func
def active_fibre_stiffness(i_mat: int, actu: float, materials: MpmMaterialTable, mpm_info: MpmInfo):
    """Active stiffness of the muscle fibre term of a particle, its fibre stiffness times the positive part of the
    actuation `actu`, and its derivative in `actu`."""
    stiffness = material_param(i_mat, materials, mpm_info, PARAM_FIBRE)
    return stiffness * qd.max(actu, 0.0), qd.select(actu > 0.0, stiffness, 0.0)


@qd.func
def fibre_terms(
    i_mat: int,
    actu: float,
    muscle_direction: qd.types.vector(3),
    F_trial: qd.types.matrix(3, 3),
    materials: MpmMaterialTable,
    mpm_info: MpmInfo,
):
    """Energy density, first Piola-Kirchhoff stress and Hessian curvature of the muscle fibre term of a particle."""
    stiffness, _ = active_fibre_stiffness(i_mat, actu, materials, mpm_info)
    F_fibre = F_trial @ muscle_direction
    return 0.5 * stiffness * F_fibre.dot(F_fibre), stiffness * F_fibre.outer_product(muscle_direction), stiffness


@qd.func
def return_map_principal(
    i_mat: int, kind: int, sigma: qd.types.vector(3), Jp: float, materials: MpmMaterialTable, mpm_info: MpmInfo
):
    """Principal values of a plastic return map that acts on the singular values, with the derivatives of its
    reverse."""
    h = sigma
    Jp_new = Jp
    is_changed = True
    dh = qd.Matrix.identity(gs.qd_float, 3)
    dh_dJp = qd.Vector.zero(gs.qd_float, 3)
    dJp_new_dsigma = qd.Vector.zero(gs.qd_float, 3)
    dJp_new_dJp = gs.qd_float(1.0)
    mu = material_param(i_mat, materials, mpm_info, PARAM_MU)
    eye = qd.Matrix.identity(gs.qd_float, 3)
    if kind == PLASTIC_VON_MISES:
        # log h = mean(eps) 1 + r eps_hat / |eps_hat| on yield, r = yield stress / (2 mu)
        eps = qd.log(qd.max(sigma, qd.static(VON_MISES_SIGMA_MIN)))
        eps_hat = eps - eps.sum() / 3.0
        eps_hat_norm = eps_hat.norm(gs.EPS)
        r = material_param(i_mat, materials, mpm_info, PARAM_YIELD_STRESS) / (2.0 * mu)
        delta_gamma = eps_hat_norm - r
        is_changed = delta_gamma > 0.0
        if is_changed:
            h = qd.exp(eps - (delta_gamma / eps_hat_norm) * eps_hat)
            is_log = sigma > qd.static(VON_MISES_SIGMA_MIN)
            deps = qd.select(is_log, 1.0 / qd.select(is_log, sigma, 1.0), 0.0)
            dlog = 1.0 / 3.0 + r * ((eye - 1.0 / 3.0) / eps_hat_norm - eps_hat.outer_product(eps_hat) / eps_hat_norm**3)
            dh = h.outer_product(deps) * dlog
    elif kind == PLASTIC_SAND:
        lam = material_param(i_mat, materials, mpm_info, PARAM_LAM)
        is_log = qd.abs(sigma) > qd.static(SAND_SIGMA_MIN)
        deps = qd.select(is_log, 1.0 / qd.select(is_log, sigma, 1.0), 0.0)
        eps = qd.log(qd.max(qd.abs(sigma), qd.static(SAND_SIGMA_MIN)))
        trace = eps.sum() + Jp
        eps_hat = eps - trace / 3.0
        eps_hat_norm = eps_hat.norm(gs.EPS)
        h = qd.Vector([1.0, 1.0, 1.0], dt=gs.qd_float)
        dh = qd.Matrix.zero(gs.qd_float, 3, 3)
        if trace >= 0.0:
            Jp_new = trace
            dJp_new_dsigma = deps
        else:
            Jp_new = 0.0
            dJp_new_dJp = 0.0
            c = (3.0 * lam + 2.0 * mu) / (2.0 * mu) * material_param(i_mat, materials, mpm_info, PARAM_SAND_ALPHA)
            delta_gamma = eps_hat_norm + c * trace
            h = qd.exp(eps - qd.max(delta_gamma, 0.0) / eps_hat_norm * eps_hat)
            if delta_gamma <= 0.0:
                dh = diagonal(h * deps)
            else:
                # log h = trace / 3 1 - c trace u with u = eps_hat / |eps_hat|, the differential of u being
                # du = (I - u u^T) / |eps_hat| d_eps_hat with d_eps_hat = (I - 1 1^T / 3) d_eps - 1 / 3 d_Jp
                u = eps_hat / eps_hat_norm
                tangent = eye - u.outer_product(u)
                scale = c * trace / eps_hat_norm
                ones = qd.Vector([1.0, 1.0, 1.0], dt=gs.qd_float)
                dlog = 1.0 / 3.0 - c * u.outer_product(ones) - scale * (tangent @ (eye - 1.0 / 3.0))
                dh = h.outer_product(deps) * dlog
                dh_dJp = h * (1.0 / 3.0 - c * u + scale / 3.0 * (tangent @ ones))
    else:
        lower = 1.0 - material_param(i_mat, materials, mpm_info, PARAM_YIELD_LOWER)
        higher = 1.0 + material_param(i_mat, materials, mpm_info, PARAM_YIELD_HIGHER)
        h = qd.min(qd.max(sigma, lower), higher)
        is_inside = (sigma > lower).cast(gs.qd_float) * (sigma < higher).cast(gs.qd_float)
        dh = diagonal(is_inside)
        if kind == PLASTIC_SNOW:
            # Jp_new = Jp prod(sigma) / prod(h)
            h_product = h[0] * h[1] * h[2]
            dJp_new_dJp = (sigma[0] * sigma[1] * sigma[2]) / h_product
            Jp_new = Jp * dJp_new_dJp
            dJp_new_dsigma = Jp * cofactor(sigma) / h_product - Jp_new * is_inside / h
    return h, Jp_new, is_changed, dh, dh_dJp, dJp_new_dsigma, dJp_new_dJp


@qd.func
def return_map(
    i_mat: int, kind: int, F: qd.types.matrix(3, 3), Jp: float, materials: MpmMaterialTable, mpm_info: MpmInfo
):
    """Plastic return map of a material slot, applied to the deformation gradient and the plastic state after the
    grid-to-particle (G2P) transfer."""
    F_new = F
    Jp_new = Jp
    if kind == PLASTIC_LIQUID:
        J = qd.min(qd.max(F.determinant(), qd.static(LIQUID_J_MIN)), 1.0)
        F_new = qd.pow(J, 1.0 / 3.0) * qd.Matrix.identity(gs.qd_float, 3)
    elif kind != PLASTIC_NONE:
        U, S, V = gu.qd_svd(F)
        sigma = diagonal_of(S)
        h, Jp_new, is_changed, dh, dh_dJp, dJp_new_dsigma, dJp_new_dJp = return_map_principal(
            i_mat, kind, sigma, Jp, materials, mpm_info
        )
        if is_changed:
            F_new = U @ diagonal(h) @ V.transpose()
    return F_new, Jp_new


@qd.func
def particle_of_sorted(i_ps: int, grid: MpmGridScratch, mpm_info: MpmInfo):
    """Particle index, environment index and cell of a sorted particle."""
    i_flat = grid.sort_ids[i_ps]
    i_b = i_flat // mpm_info.n_particles[()]
    i_p = i_flat - i_b * mpm_info.n_particles[()]
    key = grid.sort_keys[i_ps]
    _, block = unpack_block(key >> qd.u64(6))
    cell = block * 4 + local_offset(gs.qd_int(key & qd.u64(63)))
    return i_p, i_b, cell


@qd.func
def stencil_node_slots(i_b: int, cell, grid: MpmGridScratch):
    """Node slots of the 8 nodes of a cell."""
    slots = qd.Vector.zero(gs.qd_int, 8)
    for a in qd.static(range(8)):
        node_coords = cell + node_offset(a)
        i_blk = func_lookup_hash(pack_block(i_b, block_of(node_coords)), grid.hash)
        if qd.static(qd_debug_asserts()):
            assert i_blk >= 0, "MPM grid: no block holds stencil node (%d, %d, %d) of env %d" % (
                node_coords[0],
                node_coords[1],
                node_coords[2],
                i_b,
            )
        if i_blk < 0:
            flag_env_error(i_b, grid, ErrorCode.INVALID_MPM_GRID)
            i_blk = 0
        slots[a] = i_blk * BLOCK_CELLS + cell_local(node_coords)
    return slots


@qd.func
def trial_deformation(
    f: int, i_p: int, i_b: int, cell, slots, particles: MpmParticles, newton: MpmNewtonScratch, mpm_info: MpmInfo
):
    """Trial deformation gradient F_trial = (I + sum_a u_a grad w_a^T) F of a particle, with its stencil gradients and
    its deformation gradient F."""
    fx = particles.pos[f, i_p, i_b] * mpm_info.inv_dx[()] - cell.cast(gs.qd_float)
    w, grad_w = q1_stencil(fx, mpm_info.inv_dx[()])
    F = particles.F[f, i_p, i_b]
    return (qd.Matrix.identity(gs.qd_float, 3) + stencil_gradient(slots, grad_w, newton.u)) @ F, grad_w, F


@qd.func
def particle_elastic_stress(
    f: int,
    i_p: int,
    i_b: int,
    F_trial: qd.types.matrix(3, 3),
    particles: MpmParticles,
    particles_info: MpmParticlesInfo,
    materials: MpmMaterialTable,
    mpm_info: MpmInfo,
):
    """Principal decomposition and first Piola-Kirchhoff stress of the elastic energy of a particle."""
    U, S, V = gu.qd_svd(F_trial)
    sigma = diagonal_of(S)
    i_mat = particles_info.material_idx[i_p]
    mu, lam = material_params(i_mat, particles.Jp[f, i_p, i_b], materials, mpm_info)
    psi, g, A = elastic_principal(i_mat, sigma, mu, lam, materials, mpm_info)
    P = U @ diagonal(g) @ V.transpose()
    return i_mat, U, sigma, V, g, A, P


@qd.func
def pyramid_directions(normal: qd.types.vector(3), friction: float):
    """Directions of the four rows of the friction pyramid around a contact normal, as the rows of a 4 x 3 matrix."""
    tangent1, tangent2 = gu.qd_orthogonals(normal)
    return qd.Matrix.rows(
        [
            normal - friction * tangent1,
            normal + friction * tangent1,
            normal - friction * tangent2,
            normal + friction * tangent2,
        ]
    )


@qd.func
def soft_row_terms(direction: qd.types.vector(3), stiffness: float, target: float, u_rel: qd.types.vector(3)):
    """Energy, gradient and Hessian in `u_rel` of a soft contact row, 1/2 stiffness min(0, gap)^2 of the gap
    direction . u_rel - target."""
    energy = gs.qd_float(0.0)
    grad = qd.Vector.zero(gs.qd_float, 3)
    hess = qd.Matrix.zero(gs.qd_float, 3, 3)
    gap = direction.dot(u_rel) - target
    if stiffness > 0.0 and gap < 0.0:
        energy = 0.5 * stiffness * gap * gap
        grad = stiffness * gap * direction
        hess = stiffness * direction.outer_product(direction)
    return energy, grad, hess


@qd.func
def pyramid_terms(
    normal: qd.types.vector(3),
    friction: float,
    stiffness: qd.types.vector(PYRAMID_ROWS),
    target: qd.types.vector(PYRAMID_ROWS),
    u_rel: qd.types.vector(3),
):
    """Energy, gradient and Hessian in `u_rel` of the soft rows of a friction pyramid (see pyramid_directions and
    soft_row_terms), row k taking `stiffness[k]` and `target[k]`."""
    directions = pyramid_directions(normal, friction)
    energy = gs.qd_float(0.0)
    grad = qd.Vector.zero(gs.qd_float, 3)
    hess = qd.Matrix.zero(gs.qd_float, 3, 3)
    for k in qd.static(range(PYRAMID_ROWS)):
        energy_k, grad_k, hess_k = soft_row_terms(directions[k, :], stiffness[k], target[k], u_rel)
        energy += energy_k
        grad += grad_k
        hess += hess_k
    return energy, grad, hess


# ============================================== Layer 1: grid rebuild ================================================


@qd.func(requires_top_level=True)
def func_compose_particle_keys(
    f: int, particles: MpmParticles, particles_ng: MpmParticlesNG, grid: MpmGridScratch, mpm_info: MpmInfo
):
    """Write the sort key of every particle, environment-major."""
    qd.loop_config(name="mpm_compose_particle_keys")
    for i_flat in range(mpm_info.n_envs[()] * mpm_info.n_particles[()]):
        i_b = i_flat // mpm_info.n_particles[()]
        i_p = i_flat - i_b * mpm_info.n_particles[()]
        key = qd.u64(EMPTY_KEY)
        if particles_ng.active[f, i_p, i_b]:
            pos = particles.pos[f, i_p, i_b]
            # Fast math lets the compiler assume that the result of an arithmetic operation is finite, which folds a
            # test of it away: only the loaded position can be tested for being finite
            x = pos * mpm_info.inv_dx[()]
            is_in_range = qd_isfinite(pos) and ((x >= CELL_MIN) & (x < CELL_END)).all()
            if qd.static(qd_debug_asserts()):
                assert is_in_range, "MPM grid: particle %d of env %d is not finite or out of the grid range" % (
                    i_p,
                    i_b,
                )
            if is_in_range:
                key = particle_key(i_b, qd.floor(x).cast(gs.qd_int))
            else:
                flag_env_error(i_b, grid, ErrorCode.OVERFLOW_MPM_GRID_RANGE)
        grid.sort_keys[i_flat] = key
        grid.sort_ids[i_flat] = i_flat
        if i_flat == 0:
            grid.n_sort_keys[()] = mpm_info.n_envs[()] * mpm_info.n_particles[()]


@qd.func(requires_top_level=True)
def func_append_particle_blocks(grid: MpmGridScratch, mpm_info: MpmInfo):
    """Count the valid sorted particles and append the key of every distinct particle block."""
    for _ in range(1):
        grid.n_valid[()] = 0
        grid.n_particle_blocks[()] = 0
    qd.loop_config(name="mpm_append_particle_blocks")
    for i_ps in range(mpm_info.n_envs[()] * mpm_info.n_particles[()]):
        key = grid.sort_keys[i_ps]
        if key != qd.u64(EMPTY_KEY):
            if i_ps == mpm_info.n_envs[()] * mpm_info.n_particles[()] - 1 or grid.sort_keys[i_ps + 1] == qd.u64(
                EMPTY_KEY
            ):
                grid.n_valid[()] = i_ps + 1
            block_key = key >> qd.u64(6)
            if i_ps == 0 or (grid.sort_keys[i_ps - 1] >> qd.u64(6)) != block_key:
                i_blk = qd.atomic_add(grid.n_particle_blocks[()], 1)
                if i_blk < grid.particle_blocks_key.shape[0]:
                    grid.particle_blocks_key[i_blk] = block_key
    for _ in range(1):
        n_valid = grid.n_valid[()]
        grid.n_valid_chunks[()] = (n_valid + _WARP - 1) // _WARP
        n_particle_blocks = qd.min(grid.n_particle_blocks[()], grid.particle_blocks_key.shape[0])
        grid.n_neighbor_keys[()] = 8 * n_particle_blocks
        if grid.n_particle_blocks[()] > grid.particle_blocks_key.shape[0]:
            grid.has_grid_overflow[()] = 1


@qd.func(requires_top_level=True)
def func_gen_neighbor_keys(grid: MpmGridScratch):
    """Every particle block activates its positive-octant neighbour blocks, which hold the nodes its cells touch."""
    qd.loop_config(name="mpm_gen_neighbor_keys")
    for i_key in range(grid.n_neighbor_keys[()]):
        i_b, block = unpack_block(grid.particle_blocks_key[i_key >> 3])
        octant = i_key & 7
        grid.neighbors_key[i_key] = pack_block(
            i_b, block + qd.Vector([(octant >> 2) & 1, (octant >> 1) & 1, octant & 1], dt=gs.qd_int)
        )
        grid.neighbors_source_idx[i_key] = i_key


@qd.func(requires_top_level=True)
def func_block_flags(grid: MpmGridScratch):
    qd.loop_config(name="mpm_block_flags")
    for i_key in range(grid.n_neighbor_keys[()]):
        is_new_block = 0
        if i_key == 0 or grid.neighbors_key[i_key] != grid.neighbors_key[i_key - 1]:
            is_new_block = 1
        grid.neighbors_is_new_block[i_key] = is_new_block


@qd.func(requires_top_level=True)
def func_publish_blocks(grid: MpmGridScratch):
    """Give every distinct block the rank of its sorted key as slot, and insert it in the hash table."""
    for _ in range(1):
        grid.grid_demand[()] = 0
    qd.loop_config(name="mpm_publish_blocks")
    for i_key in range(grid.n_neighbor_keys[()]):
        if grid.neighbors_is_new_block[i_key] != 0:
            i_blk = grid.neighbors_block_idx[i_key]
            if i_blk < grid.blocks_key.shape[0]:
                grid.blocks_key[i_blk] = grid.neighbors_key[i_key]
                if not func_insert_hash(grid.neighbors_key[i_key], i_blk, grid.hash):
                    i_b, block = unpack_block(grid.neighbors_key[i_key])
                    flag_env_error(i_b, grid, ErrorCode.INVALID_MPM_GRID)
        if i_key == grid.n_neighbor_keys[()] - 1:
            grid.grid_demand[()] = grid.neighbors_block_idx[i_key] + grid.neighbors_is_new_block[i_key]


@qd.func(requires_top_level=True)
def func_cell_flags(grid: MpmGridScratch):
    """Flag the first sorted particle of every cell and of every task."""
    qd.loop_config(name="mpm_cell_flags")
    for i_ps in range(grid.n_valid[()]):
        is_cell_start = 0
        is_task_start = 0
        if i_ps == 0 or grid.sort_keys[i_ps] != grid.sort_keys[i_ps - 1]:
            is_cell_start = 1
        if qd.static(gs.backend == gs.cpu):
            is_task_start = is_cell_start
        elif is_cell_start != 0 or i_ps % _WARP == 0:
            is_task_start = 1
        grid.sort_is_cell_start[i_ps] = is_cell_start
        grid.sort_is_task_start[i_ps] = is_task_start


@qd.func(requires_top_level=True)
def func_publish_cells(grid: MpmGridScratch):
    """Write the cell starts and the cell and task counts, after the exclusive sums of the cell and task flags."""
    for _ in range(1):
        grid.n_active_cells[()] = 0
        grid.n_tasks[()] = 0
    qd.loop_config(name="mpm_publish_cells")
    for i_ps in range(grid.n_valid[()]):
        if grid.sort_is_cell_start[i_ps] != 0:
            i_cell = grid.sort_cell_idx[i_ps]
            grid.cells_start[i_cell] = i_ps
        if i_ps == grid.n_valid[()] - 1:
            grid.n_active_cells[()] = grid.sort_cell_idx[i_ps] + grid.sort_is_cell_start[i_ps]
            grid.n_tasks[()] = grid.sort_task_idx[i_ps] + grid.sort_is_task_start[i_ps]
    for _ in range(1):
        n_active_cells = grid.n_active_cells[()]
        grid.cells_start[n_active_cells] = grid.n_valid[()]


@qd.func(requires_top_level=True)
def func_reset_cell_segs(grid: MpmGridScratch):
    qd.loop_config(name="mpm_reset_cell_segs")
    for i_cell_slot in range(grid.n_blocks[()] * BLOCK_CELLS):
        grid.cells_segment_start[i_cell_slot] = 0
        grid.cells_segment_end[i_cell_slot] = 0


@qd.func(requires_top_level=True)
def func_fill_cell_segs(grid: MpmGridScratch):
    """Fill the dense table from cell slot to the range of its sorted particles."""
    qd.loop_config(name="mpm_fill_cell_segs")
    for i_cell in range(grid.n_active_cells[()]):
        i_ps_start = grid.cells_start[i_cell]
        key = grid.sort_keys[i_ps_start]
        i_blk = func_lookup_hash(key >> qd.u64(6), grid.hash)
        is_missing = i_blk < 0 and grid.has_grid_overflow[()] == 0
        if qd.static(qd_debug_asserts()):
            assert not is_missing, "MPM grid: no block holds the cell of sorted particle %d" % i_ps_start
        if is_missing:
            i_b, block = unpack_block(key >> qd.u64(6))
            flag_env_error(i_b, grid, ErrorCode.INVALID_MPM_GRID)
        if i_blk >= 0:
            i_cell_slot = i_blk * BLOCK_CELLS + gs.qd_int(key & qd.u64(63))
            grid.cells_segment_start[i_cell_slot] = i_ps_start
            grid.cells_segment_end[i_cell_slot] = grid.cells_start[i_cell + 1]


@qd.func(requires_top_level=True)
def func_publish_grid_demand(grid: MpmGridScratch):
    """Publish the live block count, clamped to the capacity, and raise the overflow flag past it."""
    for _ in range(1):
        demand = grid.grid_demand[()]
        n_blocks = qd.min(demand, grid.blocks_key.shape[0])
        grid.n_blocks[()] = n_blocks
        if demand > grid.blocks_key.shape[0]:
            grid.has_grid_overflow[()] = 1


@qd.func(requires_top_level=True)
def func_reset_grid(grid: MpmGridScratch):
    for _ in range(1):
        grid.has_grid_overflow[()] = 0
    func_reset_hash(grid.hash)


# ================================================ Layer 1: transfers =================================================


@qd.func(requires_top_level=True)
def func_p2g(
    f: int,
    dt: float,
    gravity: qd.template(),
    particles: MpmParticles,
    grid: MpmGridScratch,
    particles_info: MpmParticlesInfo,
    mpm_info: MpmInfo,
):
    """Particle-to-grid (P2G) transfer of the node mass and free velocity, gathered per node."""
    n_lanes = qd.static(gather_lanes())
    qd.loop_config(name="mpm_p2g", block_dim=128)
    for i_th in range(grid.n_blocks[()] * BLOCK_CELLS * n_lanes):
        i_lane = gather_lane()
        i_n = i_th // n_lanes
        i_b, block = unpack_block(grid.blocks_key[i_n // BLOCK_CELLS])
        node_coords = block * 4 + local_offset(i_n % BLOCK_CELLS)
        x_n = node_coords.cast(gs.qd_float) * mpm_info.dx[()]
        mass = gs.qd_float(0.0)
        momentum = qd.Vector.zero(gs.qd_float, 3)
        is_fixed = 0
        for a in qd.static(range(8)):
            offset = node_offset(a)
            cell = node_coords - offset
            i_ps_start, i_ps_end = cell_segment(i_b, cell, i_lane, grid)
            for i_chunk_ in range((i_ps_end - i_ps_start + n_lanes - 1) // n_lanes):
                i_ps = i_ps_start + i_chunk_ * n_lanes + i_lane
                if i_ps < i_ps_end:
                    particle_mass, particle_momentum, is_free = apic_node_contribution(
                        f, i_ps, offset, dt, cell, x_n, gravity, particles, grid, particles_info, mpm_info
                    )
                    mass += particle_mass
                    momentum += particle_momentum
                    if not is_free:
                        is_fixed = 1
        mass = lanes_sum(mass)
        for d in qd.static(range(3)):
            momentum[d] = lanes_sum(momentum[d])
        is_fixed = lanes_max(is_fixed)
        if i_lane == 0:
            publish_node_mass_velocity(i_n, mass, momentum, is_fixed, grid)


@qd.func
def apic_node_contribution(
    f: int,
    i_ps: int,
    offset,
    dt: float,
    cell,
    x_n: qd.types.vector(3),
    gravity: qd.template(),
    particles: MpmParticles,
    grid: MpmGridScratch,
    particles_info: MpmParticlesInfo,
    mpm_info: MpmInfo,
):
    """Mass and affine particle-in-cell (APIC) momentum that a sorted particle brings to one node of its stencil, and
    whether the particle is free."""
    i_flat = grid.sort_ids[i_ps]
    i_b = i_flat // mpm_info.n_particles[()]
    i_p = i_flat - i_b * mpm_info.n_particles[()]
    x_p = particles.pos[f, i_p, i_b]
    fx = x_p * mpm_info.inv_dx[()] - cell.cast(gs.qd_float)
    w, grad_w = q1_node_weight(offset, fx, mpm_info.inv_dx[()])
    mass = w * (particles_info.mass[i_p] / mpm_info.mass_scale[()])
    v_p = particles.vel[f, i_p, i_b] + dt * gravity[i_b] + particles.C[f, i_p, i_b] @ (x_n - x_p)
    return mass, mass * v_p, particles_info.free[i_p]


@qd.func
def publish_node_mass_velocity(
    i_n: int, mass: float, momentum: qd.types.vector(3), is_fixed: int, grid: MpmGridScratch
):
    grid.nodes_mass[i_n] = mass
    grid.nodes_is_fixed[i_n] = is_fixed
    v = qd.Vector.zero(gs.qd_float, 3)
    if mass > 0.0 and is_fixed == 0:
        v = momentum / mass
    grid.nodes_vel[i_n] = v


@qd.func(requires_top_level=True)
def func_apply_force_fields(t: float, dt: float, grid: MpmGridScratch, force_fields: qd.template(), mpm_info: MpmInfo):
    """nodes_vel_free = nodes_vel + dt * the accelerations of the scene force fields at the node."""
    qd.loop_config(name="mpm_apply_force_fields")
    for i_n in range(grid.n_blocks[()] * BLOCK_CELLS):
        i_b, block = unpack_block(grid.blocks_key[i_n // BLOCK_CELLS])
        x_n = (block * 4 + local_offset(i_n % BLOCK_CELLS)).cast(gs.qd_float) * mpm_info.dx[()]
        v = grid.nodes_vel[i_n]
        if grid.nodes_mass[i_n] > 0.0 and grid.nodes_is_fixed[i_n] == 0:
            acceleration = qd.Vector.zero(gs.qd_float, 3)
            for i_ff in qd.static(range(len(force_fields.force_fields))):
                acceleration += force_fields.force_fields[i_ff].get_acc(x_n, v, t, i=-1)
            v += dt * acceleration
        grid.nodes_vel_free[i_n] = v


@qd.func(requires_top_level=True)
def func_node_diag(
    f: int,
    particles: MpmParticles,
    grid: MpmGridScratch,
    newton: MpmNewtonScratch,
    particles_info: MpmParticlesInfo,
    materials: MpmMaterialTable,
    mpm_info: MpmInfo,
):
    """Diagonal block of every node of the physical Hessian (kinetic and elastic, see func_assemble)."""
    qd.loop_config(name="mpm_kinetic_node_diag")
    for i_n in range(grid.n_blocks[()] * BLOCK_CELLS):
        grid.nodes_diagonal[i_n] = kinetic_block(i_n, grid)
    qd.loop_config(name="mpm_elastic_node_diag")
    for i_ps in range(grid.n_valid[()]):
        slots, movable, r_frame, U, S_block, alpha, beta, iso = particle_stencil_terms(
            f, i_ps, mpm_info.substep_dt[()], particles, grid, newton, particles_info, materials, mpm_info, False
        )
        for a in qd.static(range(8)):
            qd.atomic_add(
                grid.nodes_diagonal[slots[a]], stencil_pair_block(a, a, r_frame, U, S_block, alpha, beta, iso)
            )


@qd.func(requires_top_level=True)
def func_contact_invweight(
    f: int,
    particles: MpmParticles,
    grid: MpmGridScratch,
    newton: MpmNewtonScratch,
    particles_info: MpmParticlesInfo,
    materials: MpmMaterialTable,
    mpm_info: MpmInfo,
):
    """Compute the inverse weight matrix of every valid particle of the first environment for its contact rows."""
    func_node_diag(f, particles, grid, newton, particles_info, materials, mpm_info)
    qd.loop_config(name="mpm_contact_invweight")
    for i_ps in range(grid.n_valid[()]):
        i_p, i_b, cell = particle_of_sorted(i_ps, grid, mpm_info)
        if i_b == 0:
            slots = stencil_node_slots(i_b, cell, grid)
            fx = particles.pos[f, i_p, i_b] * mpm_info.inv_dx[()] - cell.cast(gs.qd_float)
            w, grad_w = q1_stencil(fx, mpm_info.inv_dx[()])
            invweight = qd.Matrix.zero(gs.qd_float, 3, 3)
            for a in qd.static(range(8)):
                invweight += w[a] * w[a] * grid.nodes_diagonal[slots[a]].inverse()
            particles_info.contact_invweight[i_p] = 0.5 * (invweight + invweight.transpose())


@qd.func
def contact_invweight(
    f: int, i_p: int, i_b: int, n: qd.types.vector(3), particles: MpmParticles, particles_info: MpmParticlesInfo
):
    """Inverse weight of a particle along a direction, its rest-frame matrix rotated along with the material."""
    U, S, V = gu.qd_svd(particles.F[f, i_p, i_b])
    n_rest = V @ (U.transpose() @ n)
    return n_rest.dot(particles_info.contact_invweight[i_p] @ n_rest)


@qd.func
def contact_band_width(approach, dt: float, mpm_info: MpmInfo):
    """Gap within which a particle gets a contact row at the substep start."""
    return 2.0 * dt * qd.max(approach, 0.0) + mpm_info.dx[()]


@qd.func
def band_row(sol_params: qd.types.vector(7), phi: float, vel: float, diag_coef: float, dt: float):
    """Stiffness and target of a soft row on the displacement relative to the obstacle."""
    # The edge of the range is phi = 0 at rest, chosen at runtime so that `soft_row` is inlined once
    phi_row = gs.qd_float(0.0)
    vel_row = gs.qd_float(0.0)
    if phi <= 0.0:
        phi_row = phi
        vel_row = vel
    stiffness, target = soft_row(sol_params, phi_row, vel_row, diag_coef, dt)
    if phi > 0.0:
        target = -phi
    return stiffness, target


@qd.func(requires_top_level=True)
def func_build_wall_pairs(
    f: int,
    dt: float,
    particles: MpmParticles,
    grid: MpmGridScratch,
    pairs: MpmWallPairs,
    particles_info: MpmParticlesInfo,
    mpm_info: MpmInfo,
):
    """Build the frictionless domain-wall pairs of every valid particle."""
    for _ in range(1):
        pairs.n_pairs[()] = 0
    qd.loop_config(name="mpm_build_wall_pairs")
    for i_ps in range(grid.n_valid[()]):
        i_p, i_b, cell = particle_of_sorted(i_ps, grid, mpm_info)
        x_p = particles.pos[f, i_p, i_b]
        v_p = particles.vel[f, i_p, i_b]
        sol_params = mpm_info.wall_sol_params[()]
        # Distance to the nearer wall plane of each axis, positive inside the domain, and the sign of the inward normal
        gaps_upper = mpm_info.wall_upper[()] - x_p
        is_upper = gaps_upper < x_p - mpm_info.wall_lower[()]
        gaps = qd.select(is_upper, gaps_upper, x_p - mpm_info.wall_lower[()])
        signs = 1.0 - 2.0 * is_upper.cast(gs.qd_float)
        vels = signs * v_p
        is_near = gaps < contact_band_width(-vels, dt, mpm_info)
        # The rotation of the inverse weight (see contact_invweight) is computed once per particle near a wall: along
        # the inward normal n = +-e_d, n^T R W R^T n is the quadratic form of W on column d of R^T
        if is_near.any():
            U, S, V = gu.qd_svd(particles.F[f, i_p, i_b])
            rotation_t = V @ U.transpose()
            for d in qd.static(range(3)):
                if is_near[d]:
                    n_rest = rotation_t[:, d]
                    invweight = n_rest.dot(particles_info.contact_invweight[i_p] @ n_rest)
                    stiffness, target = band_row(sol_params, gaps[d], vels[d], invweight, dt)
                    i_pair = append_pair(pairs.n_pairs, pairs.sorted_idx.shape[0])
                    if i_pair >= 0:
                        pairs.sorted_idx[i_pair] = i_ps
                        pairs.axis[i_pair] = d
                        pairs.sign[i_pair] = signs[d]
                        pairs.stiffness[i_pair] = stiffness
                        pairs.target[i_pair] = target


@qd.func(requires_top_level=True)
def func_g2p_advect(
    f: int,
    dt: float,
    particles: MpmParticles,
    particles_ng: MpmParticlesNG,
    grid: MpmGridScratch,
    newton: MpmNewtonScratch,
    particles_info: MpmParticlesInfo,
    materials: MpmMaterialTable,
    mpm_info: MpmInfo,
    frozen: qd.template(),
):
    """Advance every particle to the next frame: grid-to-particle (G2P) transfer, advection, deformation gradient update
    and plastic return map."""
    qd.loop_config(name="mpm_g2p_advect")
    for i_ps in range(grid.n_valid[()]):
        i_p, i_b, cell = particle_of_sorted(i_ps, grid, mpm_info)
        if frozen[i_b] == 0:
            slots = stencil_node_slots(i_b, cell, grid)
            pos = particles.pos[f, i_p, i_b]
            fx = pos * mpm_info.inv_dx[()] - cell.cast(gs.qd_float)
            w, grad_w = q1_stencil(fx, mpm_info.inv_dx[()])
            vel = particle_node_sum(slots, w, newton.u) / dt
            C = stencil_gradient(slots, grad_w, newton.u) / dt
            particles.pos[f + 1, i_p, i_b] = pos + dt * vel
            particles.vel[f + 1, i_p, i_b] = vel
            particles.C[f + 1, i_p, i_b] = C
            i_mat = particles_info.material_idx[i_p]
            F_new, Jp_new = return_map(
                i_mat,
                materials.materials_plastic[i_mat],
                (qd.Matrix.identity(gs.qd_float, 3) + dt * C) @ particles.F[f, i_p, i_b],
                particles.Jp[f, i_p, i_b],
                materials,
                mpm_info,
            )
            particles.F[f + 1, i_p, i_b] = F_new
            particles.Jp[f + 1, i_p, i_b] = Jp_new
            particles.actu[f + 1, i_p, i_b] = particles.actu[f, i_p, i_b]

    qd.loop_config(name="mpm_carry_inactive")
    for i_p, i_b in qd.ndrange(mpm_info.n_particles[()], mpm_info.n_envs[()]):
        if not particles_ng.active[f, i_p, i_b] or frozen[i_b] != 0:
            particles.pos[f + 1, i_p, i_b] = particles.pos[f, i_p, i_b]
            particles.vel[f + 1, i_p, i_b] = particles.vel[f, i_p, i_b]
            particles.C[f + 1, i_p, i_b] = particles.C[f, i_p, i_b]
            particles.F[f + 1, i_p, i_b] = particles.F[f, i_p, i_b]
            particles.Jp[f + 1, i_p, i_b] = particles.Jp[f, i_p, i_b]
            particles.actu[f + 1, i_p, i_b] = particles.actu[f, i_p, i_b]
        particles_ng.active[f + 1, i_p, i_b] = particles_ng.active[f, i_p, i_b]


# ================================================== Layer 1: Newton ==================================================


@qd.func
def constraint_row_terms(i_ps: int, x: qd.types.vector(3), dx: qd.types.vector(3), newton: MpmNewtonScratch):
    """Energy, gradient and Hessian scale of the constraint spring of a sorted particle."""
    spring = newton.constraint_spring[i_ps]
    damping = newton.constraint_damping[i_ps]
    gap = x + dx - newton.constraint_target[i_ps]
    energy = 0.5 * spring * gap.norm_sqr() + 0.5 * damping * dx.norm_sqr()
    grad = spring * gap + damping * dx
    return energy, grad, spring + damping


@qd.func(requires_top_level=True)
def func_build_constraint_rows(
    grid: MpmGridScratch,
    newton: MpmNewtonScratch,
    links_state: LinksState,
    particles_info: MpmParticlesInfo,
    mpm_info: MpmInfo,
    constraints: qd.template(),
):
    """Build the spring of every constrained particle as a term of the incremental potential on its trial
    displacement."""
    dt = mpm_info.substep_dt[()]
    qd.loop_config(name="mpm_build_constraint_rows")
    for i_ps in range(grid.n_valid[()]):
        i_p, i_b, cell = particle_of_sorted(i_ps, grid, mpm_info)
        spring = gs.qd_float(0.0)
        damping = gs.qd_float(0.0)
        target = qd.Vector.zero(gs.qd_float, 3)
        if constraints[i_p, i_b].is_constrained:
            target = constraints[i_p, i_b].target_pos
            i_l = constraints[i_p, i_b].link_idx
            if i_l >= 0:
                target = gu.qd_transform_by_trans_quat(
                    constraints[i_p, i_b].link_local_pos, links_state.pos[i_l, i_b], links_state.quat[i_l, i_b]
                )
            stiffness = constraints[i_p, i_b].stiffness
            mass = particles_info.mass[i_p] / mpm_info.mass_scale[()]
            spring = dt * dt * stiffness
            damping = dt * 2.0 * qd.sqrt(stiffness * mass)
        newton.constraint_spring[i_ps] = spring
        newton.constraint_damping[i_ps] = damping
        newton.constraint_target[i_ps] = target


@qd.func
def particle_trial_displacement(
    f: int, i_p: int, i_b: int, cell, slots, particles: MpmParticles, newton: MpmNewtonScratch, mpm_info: MpmInfo
):
    """Trial displacement sum_a w_a u_a of a particle, with its stencil weights."""
    fx = particles.pos[f, i_p, i_b] * mpm_info.inv_dx[()] - cell.cast(gs.qd_float)
    w, grad_w = q1_stencil(fx, mpm_info.inv_dx[()])
    return particle_node_sum(slots, w, newton.u), w


# ============================================ Layer 1: contact vertices ============================================


@qd.func
def particle_stencil(i_ps: int, grid: MpmGridScratch, newton: MpmNewtonScratch, mpm_info: MpmInfo):
    """Environment, node slots and weights of the stencil of a sorted particle, a fixed node taking a zero weight."""
    i_p, i_b, cell = particle_of_sorted(i_ps, grid, mpm_info)
    slots = stencil_node_slots(i_b, cell, grid)
    w, _ = q1_stencil(newton.particles_pos[i_ps] * mpm_info.inv_dx[()] - cell.cast(gs.qd_float), mpm_info.inv_dx[()])
    for a in qd.static(range(8)):
        if grid.nodes_is_fixed[slots[a]] != 0:
            w[a] = 0.0
    return i_b, slots, w


@qd.func
def particle_node_sum(slots, w: qd.types.vector(8), nodes: qd.template()):
    """Sum of the node vectors of a stencil, weighted by its weights."""
    value = qd.Vector.zero(gs.qd_float, 3)
    for a in qd.static(range(8)):
        value += w[a] * nodes[slots[a]]
    return value


@qd.func
def stencil_dots(slots, nodes: qd.template(), y: qd.types.vector(3)):
    """Dot products of a vector with the node vectors of a stencil."""
    return qd.Vector([nodes[slots[a]].dot(y) for a in qd.static(range(8))], dt=gs.qd_float)


@qd.func
def stencil_gradient(slots, grad_w: qd.types.matrix(8, 3), nodes: qd.template()):
    """Gradient sum_a v_a grad w_a^T of the node vectors of a stencil, with the weight gradients in the rows of
    `grad_w`."""
    gradient = qd.Matrix.zero(gs.qd_float, 3, 3)
    for a in qd.static(range(8)):
        gradient += nodes[slots[a]].outer_product(grad_w[a, :])
    return gradient


@qd.func
def add_particle_node_vector(slots, w: qd.types.vector(8), y: qd.types.vector(3), nodes: qd.template()):
    """Add w_a y to the vector of every node a of a stencil."""
    for a in qd.static(range(8)):
        qd.atomic_add(nodes[slots[a]], w[a] * y)


@qd.func
def wall_normal(axis: int, sign: float):
    """Inward normal sign e_axis of a wall pair."""
    return sign * qd.Vector([gs.qd_float(axis == d) for d in qd.static(range(3))], dt=gs.qd_float)


@qd.func(requires_top_level=True)
def func_stage_particle_pos(
    f: int, particles: MpmParticles, grid: MpmGridScratch, newton: MpmNewtonScratch, mpm_info: MpmInfo
):
    """Copy the position of every valid particle at the substep start to its sorted slot (see particle_stencil)."""
    qd.loop_config(name="mpm_stage_particle_pos")
    for i_ps in range(grid.n_valid[()]):
        i_p, i_b, cell = particle_of_sorted(i_ps, grid, mpm_info)
        newton.particles_pos[i_ps] = particles.pos[f, i_p, i_b]


@qd.func(requires_top_level=True)
def func_newton_reset(grid: MpmGridScratch, newton: MpmNewtonScratch):
    """Reset the Newton state to u = 0, the configuration at the start of the substep, which is free of penetration."""
    qd.loop_config(name="mpm_newton_reset")
    for i_n in range(grid.n_blocks[()] * BLOCK_CELLS):
        newton.u[i_n] = qd.Vector.zero(gs.qd_float, 3)
        newton.u_start[i_n] = qd.Vector.zero(gs.qd_float, 3)
        newton.du[i_n] = qd.Vector.zero(gs.qd_float, 3)


@qd.func(requires_top_level=True)
def func_report_extent(grid: MpmGridScratch, rows: LinearBlock):
    """Report the rows of the nodes and their triplets: one kinetic diagonal per node and STENCIL_BLOCKS per elastic
    task."""
    for _ in range(1):
        n_nodes = grid.n_blocks[()] * BLOCK_CELLS
        rows.n_rows[()] = n_nodes
        rows.n_triplets[()] = n_nodes + STENCIL_BLOCKS * grid.n_tasks[()]


@qd.func
def kinetic_block(i_n: int, grid: MpmGridScratch):
    """Kinetic diagonal block of a node, m I, or I on a fixed or massless node."""
    mass = grid.nodes_mass[i_n]
    scale = gs.qd_float(1.0)
    if mass > 0.0 and grid.nodes_is_fixed[i_n] == 0:
        scale = mass
    return scale * qd.Matrix.identity(gs.qd_float, 3)


@qd.func(requires_top_level=True)
def func_kinetic_assemble(grid: MpmGridScratch, linear: NewtonLinearScratch, rows: LinearBlock):
    """Write the kinetic diagonal block and the environment of the row of every node."""
    qd.loop_config(name="mpm_kinetic_assemble")
    for i_n in range(grid.n_blocks[()] * BLOCK_CELLS):
        i_r = rows.row_start[()] + i_n
        func_write_triplet(
            rows.triplet_start[()] + i_n, i_r, i_r, kinetic_block(i_n, grid), linear.matrix, is_symmetric=True
        )
        i_b, block = unpack_block(grid.blocks_key[i_n // BLOCK_CELLS])
        linear.pcg.rows_env[i_r] = i_b


@qd.func(requires_top_level=True)
def func_elastic_assemble(
    f: int,
    dt: float,
    particles: MpmParticles,
    grid: MpmGridScratch,
    newton: MpmNewtonScratch,
    linear: NewtonLinearScratch,
    rows: LinearBlock,
    particles_info: MpmParticlesInfo,
    materials: MpmMaterialTable,
    mpm_info: MpmInfo,
    is_exact: qd.template() = False,
):
    """Assemble the elastic and contact Hessian blocks and the gradient factors of every task."""
    is_lanes = qd.static(gather_lanes() > 1)
    pair_a = qd.Vector(_STENCIL_PAIR_A, dt=gs.qd_int)
    pair_b = qd.Vector(_STENCIL_PAIR_B, dt=gs.qd_int)
    qd.loop_config(name="mpm_elastic_assemble", block_dim=128)
    for i_ps in range(grid.n_valid_chunks[()] * _WARP):
        n_valid = grid.n_valid[()]
        is_valid = i_ps < n_valid
        # A lane past the particles takes the terms of the last one and zeroes its blocks, every lane of the warp taking
        # part in the reductions
        i_ps_clamped = qd.min(i_ps, n_valid - 1)
        is_head = is_valid and grid.sort_is_task_start[i_ps_clamped] != 0
        i_task = grid.sort_task_idx[i_ps_clamped] + grid.sort_is_task_start[i_ps_clamped] - 1
        i_t_start = rows.triplet_start[()] + grid.n_blocks[()] * BLOCK_CELLS + STENCIL_BLOCKS * i_task
        i_ps_end = i_ps_clamped + 1
        is_tail = 0
        if qd.static(is_lanes):
            if gather_lane() == _WARP - 1 or i_ps >= n_valid - 1:
                is_tail = 1
            elif grid.sort_is_task_start[i_ps + 1] != 0:
                is_tail = 1
        else:
            i_ps_end = grid.cells_start[i_task + 1] if is_head else i_ps_clamped
        for j_ps in range(i_ps_clamped, i_ps_end):
            slots, movable, r_frame, U, S_block, alpha, beta, iso = particle_stencil_terms(
                f, j_ps, dt, particles, grid, newton, particles_info, materials, mpm_info, is_exact
            )
            for i_pair in range(STENCIL_BLOCKS):
                a = pair_a[i_pair]
                b = pair_b[i_pair]
                block = gs.qd_float(is_valid) * stencil_pair_block(a, b, r_frame, U, S_block, alpha, beta, iso)
                if qd.static(is_lanes):
                    for j, k in qd.static(qd.ndrange(3, 3)):
                        block[j, k] = head_segmented_reduce_add(block[j, k], is_tail)
                if is_head:
                    i_ra = rows.row_start[()] + slots[a]
                    i_rb = rows.row_start[()] + slots[b]
                    if grid.nodes_is_fixed[slots[a]] != 0 or grid.nodes_is_fixed[slots[b]] != 0:
                        if j_ps == i_ps_clamped:
                            discard_block(i_t_start + i_pair, linear)
                    elif j_ps == i_ps_clamped:
                        func_write_triplet(i_t_start + i_pair, i_ra, i_rb, block, linear.matrix, is_symmetric=True)
                    else:
                        func_add_triplet(i_t_start + i_pair, i_ra, i_rb, block, linear.matrix, is_symmetric=True)


@qd.func
def particle_stencil_terms(
    f: int,
    i_ps: int,
    dt: float,
    particles: MpmParticles,
    grid: MpmGridScratch,
    newton: MpmNewtonScratch,
    particles_info: MpmParticlesInfo,
    materials: MpmMaterialTable,
    mpm_info: MpmInfo,
    is_exact: qd.template(),
):
    """Terms of the stencil blocks of a sorted particle (see stencil_pair_block), after staging its gradient factor and
    the gradient of its constraint spring for the gradient gather."""
    i_p, i_b, cell = particle_of_sorted(i_ps, grid, mpm_info)
    slots = stencil_node_slots(i_b, cell, grid)
    movable = qd.Vector([gs.qd_float(grid.nodes_is_fixed[slots[a]] == 0) for a in qd.static(range(8))], dt=gs.qd_float)
    F_trial, grad_w, F = trial_deformation(f, i_p, i_b, cell, slots, particles, newton, mpm_info)
    i_mat, U, sigma, V, g, A, P = particle_elastic_stress(
        f, i_p, i_b, F_trial, particles, particles_info, materials, mpm_info
    )
    scale = mpm_info.particle_volume[()] * dt * dt
    muscle_direction = particles_info.muscle_direction[i_p]
    psi_fibre, P_fibre, fibre_curvature = fibre_terms(
        i_mat, particles.actu[f, i_p, i_b], muscle_direction, F_trial, materials, mpm_info
    )
    newton.particles_stress_factor[i_ps] = scale * (P + P_fibre) @ F.transpose()
    S_block, alpha, beta = isotropic_projected_modes(sigma, g, A, is_exact)
    dx, w = particle_trial_displacement(f, i_p, i_b, cell, slots, particles, newton, mpm_info)
    energy_spring, grad_spring, hess_spring = constraint_row_terms(i_ps, particles.pos[f, i_p, i_b], dx, newton)
    newton.particles_spring_grad[i_ps] = grad_spring
    r_frame = movable.outer_product(qd.Vector([1.0, 1.0, 1.0], dt=gs.qd_float)) * (grad_w @ F @ V)
    fibre_weights = qd.sqrt(scale * fibre_curvature) * (r_frame @ (V.transpose() @ muscle_direction))
    iso = qd.Matrix.cols([fibre_weights, qd.sqrt(hess_spring) * movable * w])
    return slots, movable, r_frame, U, scale * S_block, scale * alpha, scale * beta, iso


@qd.func
def stencil_pair_block(
    a: int,
    b: int,
    r_frame: qd.types.matrix(8, 3),
    U: qd.types.matrix(3, 3),
    S_block: qd.types.matrix(3, 3),
    alpha: qd.types.vector(3),
    beta: qd.types.vector(3),
    iso: qd.types.matrix(8, 2),
):
    """Block (a, b) of the stencil of one particle from its terms (see particle_stencil_terms), zero on a fixed node."""
    r_a = qd.Vector([r_frame[a, d] for d in qd.static(range(3))], dt=gs.qd_float)
    r_b = qd.Vector([r_frame[b, d] for d in qd.static(range(3))], dt=gs.qd_float)
    block = U @ isotropic_pair_block(r_a, r_b, S_block, alpha, beta) @ U.transpose()
    return block + (iso[a, 0] * iso[b, 0] + iso[a, 1] * iso[b, 1]) * qd.Matrix.identity(gs.qd_float, 3)


@qd.func(requires_top_level=True)
def func_gradient_gather(
    f: int,
    dt: float,
    particles: MpmParticles,
    grid: MpmGridScratch,
    newton: MpmNewtonScratch,
    linear: NewtonLinearScratch,
    rows: LinearBlock,
    mpm_info: MpmInfo,
):
    """Gather the gradient b_rhs = m (u - dt v_free) + sum_p (G_p grad w + w g_spring_p) of every node."""
    n_lanes = qd.static(gather_lanes())
    qd.loop_config(name="mpm_gradient_gather", block_dim=128)
    for i_th in range(grid.n_blocks[()] * BLOCK_CELLS * n_lanes):
        i_lane = gather_lane()
        i_n = i_th // n_lanes
        i_b, block = unpack_block(grid.blocks_key[i_n // BLOCK_CELLS])
        node_coords = block * 4 + local_offset(i_n % BLOCK_CELLS)
        grad = qd.Vector.zero(gs.qd_float, 3)
        for a in qd.static(range(8)):
            offset = node_offset(a)
            cell = node_coords - offset
            i_ps_start, i_ps_end = cell_segment(i_b, cell, i_lane, grid)
            for i_chunk_ in range((i_ps_end - i_ps_start + n_lanes - 1) // n_lanes):
                i_ps = i_ps_start + i_chunk_ * n_lanes + i_lane
                if i_ps < i_ps_end:
                    grad = add_particle_node_gradient(f, i_ps, offset, grad, cell, particles, grid, newton, mpm_info)
        for d in qd.static(range(3)):
            grad[d] = lanes_sum(grad[d])
        if i_lane == 0:
            publish_node_gradient(i_n, grad, dt, grid, newton, linear, rows)


@qd.func
def add_particle_node_gradient(
    f: int,
    i_ps: int,
    offset,
    grad: qd.types.vector(3),
    cell,
    particles: MpmParticles,
    grid: MpmGridScratch,
    newton: MpmNewtonScratch,
    mpm_info: MpmInfo,
):
    """Add the gradient contribution G_p grad w + w g_spring_p of a sorted particle at one node of its stencil."""
    i_flat = grid.sort_ids[i_ps]
    i_b = i_flat // mpm_info.n_particles[()]
    i_p = i_flat - i_b * mpm_info.n_particles[()]
    fx = particles.pos[f, i_p, i_b] * mpm_info.inv_dx[()] - cell.cast(gs.qd_float)
    w, grad_w = q1_node_weight(offset, fx, mpm_info.inv_dx[()])
    return grad + w * newton.particles_spring_grad[i_ps] + newton.particles_stress_factor[i_ps] @ grad_w


@qd.func
def publish_node_gradient(
    i_n: int,
    grad: qd.types.vector(3),
    dt: float,
    grid: MpmGridScratch,
    newton: MpmNewtonScratch,
    linear: NewtonLinearScratch,
    rows: LinearBlock,
):
    value = qd.Vector.zero(gs.qd_float, 3)
    if grid.nodes_is_fixed[i_n] == 0:
        u = newton.u[i_n]
        value = grad + grid.nodes_mass[i_n] * (u - dt * grid.nodes_vel_free[i_n])
    linear.b_rhs[rows.row_start[()] + i_n] = value


@qd.func(requires_top_level=True)
def func_assemble(
    f: int,
    dt: float,
    particles: MpmParticles,
    grid: MpmGridScratch,
    newton: MpmNewtonScratch,
    linear: NewtonLinearScratch,
    rows: LinearBlock,
    particles_info: MpmParticlesInfo,
    materials: MpmMaterialTable,
    mpm_info: MpmInfo,
    is_exact: qd.template() = False,
):
    """Write the gradient and Hessian of the nodes into their rows and triplets of the linear system."""
    func_kinetic_assemble(grid, linear, rows)
    func_elastic_assemble(f, dt, particles, grid, newton, linear, rows, particles_info, materials, mpm_info, is_exact)
    func_gradient_gather(f, dt, particles, grid, newton, linear, rows, mpm_info)


@qd.func(requires_top_level=True)
def func_negate_du(grid: MpmGridScratch, newton: MpmNewtonScratch, linear: NewtonLinearScratch, rows: LinearBlock):
    qd.loop_config(name="mpm_negate_du")
    for i_n in range(grid.n_blocks[()] * BLOCK_CELLS):
        newton.du[i_n] = -linear.solution[rows.row_start[()] + i_n]


@qd.func(requires_top_level=True)
def func_record_start_point(grid: MpmGridScratch, newton: MpmNewtonScratch):
    qd.loop_config(name="mpm_record_start_point")
    for i_n in range(grid.n_blocks[()] * BLOCK_CELLS):
        newton.u_start[i_n] = newton.u[i_n]


@qd.func(requires_top_level=True)
def func_contribute_newton_max_displacement(
    f: int,
    particles: MpmParticles,
    grid: MpmGridScratch,
    newton: MpmNewtonScratch,
    control: NewtonControl,
    mpm_info: MpmInfo,
):
    """Reduce into the Newton control of every environment the largest |du| component of its particles, sum_a w_a
    du_a."""
    qd.loop_config(name="mpm_contribute_newton_max_disp", block_dim=REDUCE_BLOCK)
    for i_ps in range(grid.n_valid_chunks[()] * _WARP):
        displacement = gs.qd_float(0.0)
        env_key = gs.qd_int(-1)
        if i_ps < grid.n_valid[()]:
            _, i_b, _ = particle_of_sorted(i_ps, grid, mpm_info)
            env_key = i_b
            displacement = particle_max_displacement(f, i_ps, particles, grid, newton, mpm_info)
        max_by_key(displacement, env_key, control.max_displacement)


@qd.func(requires_top_level=True)
def func_cap_inversion_step(
    f: int,
    particles: MpmParticles,
    grid: MpmGridScratch,
    newton: MpmNewtonScratch,
    control: NewtonControl,
    particles_info: MpmParticlesInfo,
    materials: MpmMaterialTable,
    mpm_info: MpmInfo,
):
    """Cap the start step of the line search below the first zero volume of a liquid particle along the Newton
    direction."""
    t_max = 1.0 / INVERSION_STEP_SLACK
    qd.loop_config(name="mpm_cap_inversion_step")
    for i_ps in range(grid.n_valid[()]):
        i_p, i_b, cell = particle_of_sorted(i_ps, grid, mpm_info)
        i_mat = particles_info.material_idx[i_p]
        if materials.materials_family[i_mat] == FAMILY_LIQUID:
            slots = stencil_node_slots(i_b, cell, grid)
            F_trial, grad_w, F = trial_deformation(f, i_p, i_b, cell, slots, particles, newton, mpm_info)
            c0, c1, c2, c3 = det_cubic(F_trial, stencil_gradient(slots, grad_w, newton.du) @ F)
            if c0 > 0.0:
                root = cubic_first_root(c0, c1, c2, c3, t_max)
                if root <= t_max:
                    qd.atomic_min(control.alpha[i_b], INVERSION_STEP_SLACK * root)


@qd.func
def particle_max_displacement(
    f: int, i_ps: int, particles: MpmParticles, grid: MpmGridScratch, newton: MpmNewtonScratch, mpm_info: MpmInfo
):
    """Largest |du| component of a sorted particle, from its interpolated displacement update sum_a w_a du_a."""
    i_p, i_b, cell = particle_of_sorted(i_ps, grid, mpm_info)
    slots = stencil_node_slots(i_b, cell, grid)
    w, _ = q1_stencil(particles.pos[f, i_p, i_b] * mpm_info.inv_dx[()] - cell.cast(gs.qd_float), mpm_info.inv_dx[()])
    return qd.abs(particle_node_sum(slots, w, newton.du)).max()


@qd.func(requires_top_level=True)
def func_step_forward(grid: MpmGridScratch, newton: MpmNewtonScratch, control: NewtonControl):
    """Move the nodes of every environment to the trial point of its line-search step."""
    qd.loop_config(name="mpm_step_forward")
    for i_n in range(grid.n_blocks[()] * BLOCK_CELLS):
        i_b, block = unpack_block(grid.blocks_key[i_n // BLOCK_CELLS])
        newton.u[i_n] = newton.u_start[i_n] + control.alpha[i_b] * newton.du[i_n]


@qd.func(requires_top_level=True)
def func_elastic_energy(
    f: int,
    dt: float,
    particles: MpmParticles,
    grid: MpmGridScratch,
    newton: MpmNewtonScratch,
    energies: qd.template(),
    particles_info: MpmParticlesInfo,
    materials: MpmMaterialTable,
    mpm_info: MpmInfo,
):
    """Add V dt^2 psi(F_trial) and the constraint spring energy of the sorted particles into the entry of their
    environment in `energies`, the particles being sorted by environment (see add_by_key)."""
    qd.loop_config(name="mpm_elastic_energy", block_dim=REDUCE_BLOCK)
    for i_ps in range(grid.n_valid_chunks[()] * _WARP):
        energy = gs.qd_float(0.0)
        env_key = gs.qd_int(-1)
        if i_ps < grid.n_valid[()]:
            _, i_b, _ = particle_of_sorted(i_ps, grid, mpm_info)
            env_key = i_b
            energy = particle_energy(f, i_ps, dt, particles, grid, newton, particles_info, materials, mpm_info)
        add_by_key(energy, env_key, energies)


@qd.func
def particle_energy(
    f: int,
    i_ps: int,
    dt: float,
    particles: MpmParticles,
    grid: MpmGridScratch,
    newton: MpmNewtonScratch,
    particles_info: MpmParticlesInfo,
    materials: MpmMaterialTable,
    mpm_info: MpmInfo,
):
    """V dt^2 psi(F_trial) and the constraint spring energy of sorted particle `i_ps`."""
    i_p, i_b, cell = particle_of_sorted(i_ps, grid, mpm_info)
    slots = stencil_node_slots(i_b, cell, grid)
    F_trial, grad_w, F = trial_deformation(f, i_p, i_b, cell, slots, particles, newton, mpm_info)
    i_mat = particles_info.material_idx[i_p]
    mu, lam = material_params(i_mat, particles.Jp[f, i_p, i_b], materials, mpm_info)
    _, S, _ = gu.qd_svd(F_trial)
    psi, g, A = elastic_principal(i_mat, diagonal_of(S), mu, lam, materials, mpm_info)
    psi_fibre, P_fibre, fibre_curvature = fibre_terms(
        i_mat, particles.actu[f, i_p, i_b], particles_info.muscle_direction[i_p], F_trial, materials, mpm_info
    )
    psi += psi_fibre
    dx, w = particle_trial_displacement(f, i_p, i_b, cell, slots, particles, newton, mpm_info)
    energy_spring, grad_spring, hess_spring = constraint_row_terms(i_ps, particles.pos[f, i_p, i_b], dx, newton)
    return mpm_info.particle_volume[()] * dt * dt * psi + energy_spring


@qd.func(requires_top_level=True)
def func_kinetic_energy(dt: float, grid: MpmGridScratch, newton: MpmNewtonScratch, energies: qd.template()):
    """Add 1/2 m |u - dt v_free|^2 of the free nodes into the entry of their environment in `energies`, reduced as in
    func_elastic_energy. The nodes of a block, of one environment, fill whole warps."""
    qd.loop_config(name="mpm_kinetic_energy", block_dim=REDUCE_BLOCK)
    for i_n in range(grid.n_blocks[()] * BLOCK_CELLS):
        i_b, _ = unpack_block(grid.blocks_key[i_n // BLOCK_CELLS])
        add_by_key(node_kinetic_energy(i_n, dt, grid, newton), i_b, energies)


@qd.func
def node_kinetic_energy(i_n: int, dt: float, grid: MpmGridScratch, newton: MpmNewtonScratch):
    """1/2 m |u - dt v_free|^2 of a node, zero for a fixed node."""
    energy = gs.qd_float(0.0)
    if grid.nodes_is_fixed[i_n] == 0:
        u = newton.u[i_n]
        energy = 0.5 * grid.nodes_mass[i_n] * (u - dt * grid.nodes_vel_free[i_n]).norm_sqr()
    return energy

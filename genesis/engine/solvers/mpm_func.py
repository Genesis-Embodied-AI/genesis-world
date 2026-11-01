"""Layer-0 mathematics and Layer-1 stages of the implicit material point method (MPM) solver.

Every function here is a '@qd.func'. Layer-0 functions map values to values. Layer-1 functions take the solver data
(particle data, `MpmGrid`, `MpmNewton`, `MpmLinearSystem`, `NewtonControl`, `MpmMaterialTable`) and the scene values of
`MpmParams`. Those marked 'requires_top_level' are the stages that a graph kernel or a kernel of the solver calls. Per
substep, the grid node displacement u minimizes the incremental potential

    E(u) = sum_a 1/2 m_a |u_a - dt v_a|^2 + dt^2 sum_p V_p psi(F_trial_p),   F_trial_p = (I + sum_a u_a grad w_a^T) F_p

with trilinear (Q1) shape functions and affine particle-in-cell (APIC) transfers. The block-sparse hash grid is rebuilt
every substep.

The stages that combine the values of several particles or nodes (the gathers of the particle-to-grid (P2G) transfer and
of the gradient, the assembly of the stencil blocks, the brick inversion, the reductions of the energy and of the
displacement update) have one implementation per backend, `_gpu` and `_cpu`, a dispatcher picking one at compile time.
On a GPU they map a node, a 32-particle chunk or a brick to a warp, whose lanes combine their values by shuffles and
ballots, and a reduction to a block. The CPU backend of Quadrants has none of these operations and runs a few dozen
threads: each thread takes a whole node, cell or brick and walks it serially, in the same fixed order, which needs
neither lanes nor atomics. The stages that map each particle or node on its own are the same for both backends.
"""

import quadrants as qd

import genesis as gs
import genesis.utils.array_class as array_class
import genesis.utils.geom as gu
from genesis.engine.numerics.block_reduce import REDUCE_BLOCK_DIM, write_block_partial
from genesis.engine.numerics.dynamic_exclusive_sum import func_dynamic_exclusive_sum
from genesis.engine.numerics.dynamic_radix_sort import func_dynamic_radix_sort
from genesis.engine.numerics.fsr_reduce import head_segmented_reduce_add
from genesis.engine.numerics.linear_pcg import func_linear_pcg_direction_update, func_linear_pcg_matvec_update
from genesis.engine.numerics.open_addressing_hash import EMPTY_KEY, func_reset_hash
from genesis.engine.numerics.soft_contact import soft_row
from genesis.utils.misc import qd_debug_asserts, qd_isfinite

from .mpm_data import (
    BLOCK_CELLS,
    BRICK_DIM,
    BRICK_INVERT_WARPS,
    BRICK_NODES,
    BRICK_SCALARS,
    BRICK_SYM_BLOCKS,
    BRICKS_PER_BLOCK,
    CONTACT_ROWS,
    STENCIL_BLOCKS,
    WALL_ROWS,
    MpmMaterialTable,
    MpmParticles,
    MpmParticlesInfo,
    MpmParticlesNG,
)

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
# Hardening exponent of snow, mu and lam scaled by exp(SNOW_HARDENING (1 - Jp))
SNOW_HARDENING = 10.0

# Pairs of principal axes, in the order of the twist and flip eigenvalues
_PAIRS = ((0, 1), (0, 2), (1, 2))

# Upper-triangular node pairs (a <= b) of the 8-node stencil, in the order of a task's 36 Hessian blocks
_STENCIL_PAIR_A = tuple(a for a in range(8) for b in range(a, 8))
_STENCIL_PAIR_B = tuple(b for a in range(8) for b in range(a, 8))

# Register tiling of the brick inversion: the 48 x 48 matrix as an 8 x 8 grid of 6 x 6 tiles. Lanes below
# _BRICK_OFF_TILES own one off-diagonal tile each (36 entries), the others two diagonal tiles each (2 x 21 upper
# entries), which makes _BRICK_SLOTS entries per lane at most.
_BRICK_TILE = 6
_BRICK_TILES = BRICK_DIM // _BRICK_TILE
_BRICK_OFF_TILES = _BRICK_TILES * (_BRICK_TILES - 1) // 2
_BRICK_TRI = _BRICK_TILE * (_BRICK_TILE + 1) // 2
_BRICK_SLOTS = 2 * _BRICK_TRI
# Upper pair (i <= j) of the slots of a diagonal tile
_BRICK_TRI_I = tuple(i for i in range(_BRICK_TILE) for j in range(i, _BRICK_TILE))
_BRICK_TRI_J = tuple(j for i in range(_BRICK_TILE) for j in range(i, _BRICK_TILE))
# Shift of the unit diagonal after equilibration, which shifts every diagonal entry by the same fraction of itself
_BRICK_REGULARIZATION = 1e-6
_BRICK_FINITE_LIMIT = float("inf")

# Lanes of a warp
_WARP = 32


# ============================================== Layer 0: shape and keys ==============================================


@qd.func
def node_offset(a: qd.template()):
    """Offset of the a-th node of a cell from its lower corner, bits 2, 1 and 0 of a giving the x, y and z offsets."""
    return qd.Vector([(a >> 2) & 1, (a >> 1) & 1, a & 1], dt=qd.i32)


@qd.func
def block_of(cell):
    """Block coordinates of a cell or node coordinate (floor division by the block size, valid for negatives)."""
    return qd.Vector([(cell[d] - (cell[d] & 3)) // 4 for d in qd.static(range(3))], dt=qd.i32)


@qd.func
def cell_local(cell):
    """Index of a cell or node coordinate inside its block."""
    return ((cell[0] & 3) << 4) | ((cell[1] & 3) << 2) | (cell[2] & 3)


@qd.func
def local_offset(local):
    """Inverse of cell_local: offset of a local index inside its block."""
    return qd.Vector([(local >> 4) & 3, (local >> 2) & 3, local & 3], dt=qd.i32)


@qd.func
def pack_block(env, block):
    """Key of a block of an environment: environment index from bit 42, then 14 biased bits per block coordinate."""
    return (
        (qd.u64(env) << qd.u64(42))
        | (qd.u64(block[0] + COORD_BIAS) << qd.u64(28))
        | (qd.u64(block[1] + COORD_BIAS) << qd.u64(14))
        | qd.u64(block[2] + COORD_BIAS)
    )


@qd.func
def unpack_block(key):
    """Environment and block coordinates of a block key."""
    env = qd.i32(key >> qd.u64(42))
    block = qd.Vector(
        [
            qd.i32((key >> qd.u64(28)) & qd.u64(0x3FFF)) - COORD_BIAS,
            qd.i32((key >> qd.u64(14)) & qd.u64(0x3FFF)) - COORD_BIAS,
            qd.i32(key & qd.u64(0x3FFF)) - COORD_BIAS,
        ],
        dt=qd.i32,
    )
    return env, block


@qd.func
def particle_key(env, cell):
    """Sort key of a particle: its block key followed by the 6-bit local index of its cell."""
    return (pack_block(env, block_of(cell)) << qd.u64(6)) | qd.u64(cell_local(cell))


@qd.func
def flag_env_error(
    grid: qd.template(),  # MpmGrid
    env,
    code: qd.template(),  # array_class.ErrorCode
):
    """Raise an error code for one environment, which the solver reports at the beginning of the next step."""
    qd.atomic_or(grid.errno[env], qd.static(int(code)))


@qd.func
def flag_all_envs_error(
    grid: qd.template(),  # MpmGrid
    params: qd.template(),  # MpmParams
    code: qd.template(),  # array_class.ErrorCode
):
    """Raise an error code for every environment, for a failure of the solve they share.

    The caller runs it on one thread.
    """
    for i_b in range(params.n_envs[()]):
        flag_env_error(grid, i_b, code)


@qd.func
def q1_stencil(fx, inv_dx):
    """Trilinear weights and weight gradients of the 8 nodes of a cell at a point inside it.

    `fx` is the offset of the point from the lower corner of the cell in cell units, in [0, 1)^3, and `inv_dx` the
    inverse cell size.
    """
    w = qd.Vector.zero(gs.qd_float, 8)
    grad_w = qd.Matrix.zero(gs.qd_float, 8, 3)
    for a in qd.static(range(8)):
        axis_w = qd.Vector.zero(gs.qd_float, 3)
        axis_dw = qd.Vector.zero(gs.qd_float, 3)
        for d in qd.static(range(3)):
            if qd.static((a >> (2 - d)) & 1):
                axis_w[d] = fx[d]
                axis_dw[d] = inv_dx
            else:
                axis_w[d] = 1.0 - fx[d]
                axis_dw[d] = -inv_dx
        w[a] = axis_w[0] * axis_w[1] * axis_w[2]
        grad_w[a, 0] = axis_dw[0] * axis_w[1] * axis_w[2]
        grad_w[a, 1] = axis_w[0] * axis_dw[1] * axis_w[2]
        grad_w[a, 2] = axis_w[0] * axis_w[1] * axis_dw[2]
    return w, grad_w


@qd.func
def q1_weight_hessian(fx, a: qd.template(), inv_dx):
    """Hessian of the trilinear weight of one node of a cell with respect to the particle position (see q1_stencil).

    For node `a` of the cell, entry (i, j) off the diagonal is the product of the axis derivatives along i and j and the
    axis weight along the third axis. The diagonal is zero, the weight being linear along each axis.
    """
    axis_w = qd.Vector.zero(gs.qd_float, 3)
    axis_dw = qd.Vector.zero(gs.qd_float, 3)
    for d in qd.static(range(3)):
        if qd.static((a >> (2 - d)) & 1):
            axis_w[d] = fx[d]
            axis_dw[d] = inv_dx
        else:
            axis_w[d] = 1.0 - fx[d]
            axis_dw[d] = -inv_dx
    H = qd.Matrix.zero(gs.qd_float, 3, 3)
    for i, j in qd.static(qd.ndrange(3, 3)):
        if qd.static(i != j):
            H[i, j] = axis_dw[i] * axis_dw[j] * axis_w[3 - i - j]
    return H


@qd.func
def q1_node_weight(fx, o, inv_dx):
    """Trilinear weight and weight gradient of one node of a cell at a point inside it (see q1_stencil).

    `o` is the offset of the node in {0, 1}^3 and `fx` the offset of the point from the lower corner, in cell units.
    """
    axis_w = qd.Vector.zero(gs.qd_float, 3)
    axis_dw = qd.Vector.zero(gs.qd_float, 3)
    for d in qd.static(range(3)):
        if o[d] != 0:
            axis_w[d] = fx[d]
            axis_dw[d] = inv_dx
        else:
            axis_w[d] = 1.0 - fx[d]
            axis_dw[d] = -inv_dx
    w = axis_w[0] * axis_w[1] * axis_w[2]
    grad_w = qd.Vector(
        [
            axis_dw[0] * axis_w[1] * axis_w[2],
            axis_w[0] * axis_dw[1] * axis_w[2],
            axis_w[0] * axis_w[1] * axis_dw[2],
        ],
        dt=gs.qd_float,
    )
    return w, grad_w


# ============================================= Layer 0: warp reductions ==============================================


@qd.func
def warp_sum(value):
    """Sum over the 32 lanes of a warp by a fixed butterfly, so every lane holds the same, reproducible total."""
    for level in qd.static(range(5)):
        value = value + qd.simt.subgroup.shuffle_xor(value, qd.u32(16 >> level))
    return value


@qd.func
def warp_max(value):
    for level in qd.static(range(5)):
        value = qd.max(value, qd.simt.subgroup.shuffle_xor(value, qd.u32(16 >> level)))
    return value


# ============================================== Layer 0: constitution ================================================


@qd.func
def signed_singular_values(F):
    """Singular values of a deformation gradient by descending magnitude, the last one signed by its determinant.

    They are the square roots of the eigenvalues of F^T F in closed form (the trigonometric solution of the
    characteristic cubic), which skips the rotations of a full singular value decomposition since the energy needs the
    singular values alone.
    """
    A = F.transpose() @ F
    p1 = A[0, 1] * A[0, 1] + A[0, 2] * A[0, 2] + A[1, 2] * A[1, 2]
    eigenvalues = qd.Vector([A[0, 0], A[1, 1], A[2, 2]], dt=gs.qd_float)
    if p1 > 0.0:
        q = (A[0, 0] + A[1, 1] + A[2, 2]) / 3.0
        p2 = (A[0, 0] - q) ** 2 + (A[1, 1] - q) ** 2 + (A[2, 2] - q) ** 2 + 2.0 * p1
        p = qd.sqrt(p2 / 6.0)
        B = (A - q * qd.Matrix.identity(gs.qd_float, 3)) / p
        r = qd.min(qd.max(B.determinant() / 2.0, -1.0), 1.0)
        phi = qd.acos(r) / 3.0
        e1 = q + 2.0 * p * qd.cos(phi)
        e3 = q + 2.0 * p * qd.cos(phi + 2.0943951023931953)
        eigenvalues = qd.Vector([e1, 3.0 * q - e1 - e3, e3], dt=gs.qd_float)
    else:
        # Already diagonal: sort the eigenvalues in descending order
        for i in qd.static(range(2)):
            for j in qd.static(range(2 - i)):
                if eigenvalues[j] < eigenvalues[j + 1]:
                    eigenvalue_next = eigenvalues[j + 1]
                    eigenvalues[j + 1] = eigenvalues[j]
                    eigenvalues[j] = eigenvalue_next
    sigma = qd.Vector([qd.sqrt(qd.max(eigenvalues[i], 0.0)) for i in qd.static(range(3))], dt=gs.qd_float)
    if F.determinant() < 0.0:
        sigma[2] = -sigma[2]
    return sigma


@qd.func
def compression_penalty(sigma, mu, lam):
    """Compression penalty of the fixed corotated and Stable Neo-Hookean energies, with its gradient and Hessian.

    The penalty is k / 2 |d|^2 with d = min(sigma - s, 0), s = EXTENSION_SIGMA_MIN and k = EXTENSION_STIFFNESS times the
    Young's modulus of the Lame parameters `mu` and `lam`. Its gradient is k d and its diagonal Hessian is k where
    sigma_i < s.

    In the material point method (MPM) a particle pushes its nodes by its Kirchhoff stress,
    tau_i = sigma_i dpsi/dsigma_i, which vanishes as the particle collapses whenever dpsi/dsigma stays bounded, as it
    does for both energies. A crushed particle then exerts no restoring force, its stiffness vanishing too, and the
    material stays crushed. The penalty is the stiffness term of the C1 extension of Stomakhin et al. 2012 (Sec. 4.1),
    with their s and k. Their extension also replaces the energy below s by its Taylor polynomial at s, which an energy
    undefined across inversion needs. Both energies here are defined there, and keeping them whole keeps the sum bounded
    below. The Taylor polynomial of the fixed corotated energy is unbounded below under a large stretch of another axis,
    which raises the curvature that the polynomial drops.
    """
    s = gs.qd_float(EXTENSION_SIGMA_MIN)
    young_modulus = gs.qd_float(0.0)
    if lam + mu > 0.0:
        young_modulus = mu * (3.0 * lam + 2.0 * mu) / (lam + mu)
    k = EXTENSION_STIFFNESS * young_modulus
    d = qd.min(sigma - s, 0.0)
    curvature = qd.Vector.zero(gs.qd_float, 3)
    for i in qd.static(range(3)):
        if sigma[i] < s:
            curvature[i] = k
    return 0.5 * k * d.dot(d), k * d, curvature


@qd.func
def stable_neohookean_parameters(mu, lam):
    """mu_s, lam_s, alpha and the rest energy of the Stable Neo-Hookean energy (see stable_neohookean_principal)."""
    mu_s = 4.0 / 3.0 * mu
    lam_s = lam + 5.0 / 6.0 * mu
    alpha = gs.qd_float(1.0)
    if lam_s > 0.0:
        alpha = 1.0 + mu_s / lam_s - mu_s / (4.0 * lam_s)
    rest = 0.5 * lam_s * (1.0 - alpha) ** 2 - 0.5 * mu_s * qd.log(4.0)
    return mu_s, lam_s, alpha, rest


@qd.func
def fixed_corotated_energy(sigma, mu, lam):
    """Energy density of the fixed corotated energy with its compression penalty (see fixed_corotated_principal)."""
    J = sigma[0] * sigma[1] * sigma[2]
    penalty, g_penalty, A_penalty = compression_penalty(sigma, mu, lam)
    return mu * (sigma - 1.0).dot(sigma - 1.0) + 0.5 * lam * (J - 1.0) ** 2 + penalty


@qd.func
def extended_hencky_principal(sigma, mu, lam):
    """Energy density, gradient g = dpsi/dsigma and Hessian A = d2psi/dsigma2 of the extended Hencky energy.

    psi = mu |l|^2 + lam / 2 (sum l)^2 with l_i = log(sigma_i), continued below HENCKY_SIGMA_MIN by its tangent line
    log(s) + (sigma_i - s) / s, which keeps the energy C1 and defined for inverted states (sigma_i <= 0). With the
    principal Kirchhoff stress tau = 2 mu l + lam (sum l) 1: g_i = tau_i l'_i and
    A = diag(l') (2 mu I + lam 11^T) diag(l') + diag(tau l''), l'' = -1 / sigma^2 above s and 0 below.
    """
    s = qd.static(HENCKY_SIGMA_MIN)
    l = qd.Vector.zero(gs.qd_float, 3)
    dl = qd.Vector.zero(gs.qd_float, 3)
    ddl = qd.Vector.zero(gs.qd_float, 3)
    for i in qd.static(range(3)):
        if sigma[i] >= s:
            l[i] = qd.log(sigma[i])
            dl[i] = 1.0 / sigma[i]
            ddl[i] = -1.0 / (sigma[i] * sigma[i])
        else:
            l[i] = qd.log(s) + (sigma[i] - s) / s
            dl[i] = 1.0 / s
    trace = l[0] + l[1] + l[2]
    psi = mu * l.dot(l) + 0.5 * lam * trace * trace
    tau = 2.0 * mu * l + lam * trace
    g = tau * dl
    A = qd.Matrix.zero(gs.qd_float, 3, 3)
    for i, j in qd.static(qd.ndrange(3, 3)):
        A[i, j] = dl[i] * (2.0 * mu * (1.0 if i == j else 0.0) + lam) * dl[j]
        if qd.static(i == j):
            A[i, j] += tau[i] * ddl[i]
    return psi, g, A


@qd.func
def stable_neohookean_principal(sigma, mu, lam, is_penalized: qd.template() = True):
    """Energy density, gradient and Hessian of the Stable Neo-Hookean energy (Smith et al. 2018, Eq. 14) in the signed
    singular values, defined for inverted states.

    psi = mu_s / 2 (I_C - 3) + lam_s / 2 (J - alpha)^2 - mu_s / 2 log(I_C + 1), shifted to vanish at rest, with
    I_C = |sigma|^2. Without the regularized origin barrier log(I_C + 1), a particle crushed to F = 0 sits in a stable
    basin, its forces vanishing and its Hessian mu I (Sec. 3.3), which a collapsed material never leaves.
    alpha = 1 + mu_s / lam_s - mu_s / (4 lam_s) keeps the rest state stress-free, and mu_s = 4/3 mu and
    lam_s = lam + 5/6 mu give the small-strain response of the Lame parameters mu and lam (Sec. 3.4).

    The barrier only makes F = 0 meta-stable: the Kirchhoff stress of a uniformly crushed particle still vanishes as the
    cube of its size, which leaves a crushed material point method (MPM) material without a restoring force. With
    `is_penalized`, the compression penalty adds one (see compression_penalty).
    """
    mu_s, lam_s, alpha, rest = stable_neohookean_parameters(mu, lam)
    J = sigma[0] * sigma[1] * sigma[2]
    cofactor = qd.Vector([sigma[1] * sigma[2], sigma[0] * sigma[2], sigma[0] * sigma[1]], dt=gs.qd_float)
    I_C = sigma.dot(sigma)
    barrier = 1.0 / (I_C + 1.0)
    psi = 0.5 * mu_s * (I_C - 3.0) + 0.5 * lam_s * (J - alpha) ** 2 - 0.5 * mu_s * qd.log(I_C + 1.0) - rest
    g = mu_s * (1.0 - barrier) * sigma + lam_s * (J - alpha) * cofactor
    A = qd.Matrix.zero(gs.qd_float, 3, 3)
    for i in qd.static(range(3)):
        A[i, i] = (
            mu_s * (1.0 - barrier) + 2.0 * mu_s * barrier * barrier * sigma[i] * sigma[i] + lam_s * cofactor[i] ** 2
        )
    for k in qd.static(range(3)):
        i, j = qd.static(_PAIRS[k])
        axis_other = qd.static(3 - _PAIRS[k][0] - _PAIRS[k][1])
        A[i, j] = 2.0 * mu_s * barrier * barrier * sigma[i] * sigma[j] + lam_s * (
            cofactor[i] * cofactor[j] + (J - alpha) * sigma[axis_other]
        )
        A[j, i] = A[i, j]
    if qd.static(is_penalized):
        penalty, g_penalty, A_penalty = compression_penalty(sigma, mu, lam)
        psi += penalty
        g += g_penalty
        for i in qd.static(range(3)):
            A[i, i] += A_penalty[i]
    return psi, g, A


@qd.func
def liquid_principal(sigma, mu, lam):
    """Energy density, gradient and Hessian of the liquid in the signed singular values, without compression penalty.

    The volume term lam / 2 (J - 1)^2 is one-sided: it acts under compression (J < 1) and vanishes under expansion, C1
    at J = 1. The fixed corotated shear term mu |sigma - 1|^2 models viscosity, with mu zero for an inviscid liquid.
    """
    lam_compressed = gs.qd_float(0.0)
    if sigma[0] * sigma[1] * sigma[2] < 1.0:
        lam_compressed = lam
    return fixed_corotated_principal(sigma, mu, lam_compressed, is_penalized=False)


@qd.func
def fixed_corotated_principal(sigma, mu, lam, is_penalized: qd.template() = True):
    """Energy density, gradient and Hessian of the fixed corotated energy in the signed singular values.

    psi = mu |sigma - 1|^2 + lam / 2 (J - 1)^2 with J = sigma_0 sigma_1 sigma_2, defined for inverted states too. With
    `is_penalized`, the compression penalty keeps a crushed particle pushing back (see compression_penalty).
    """
    J = sigma[0] * sigma[1] * sigma[2]
    cofactor = qd.Vector([sigma[1] * sigma[2], sigma[0] * sigma[2], sigma[0] * sigma[1]], dt=gs.qd_float)
    psi = mu * (sigma - 1.0).dot(sigma - 1.0) + 0.5 * lam * (J - 1.0) ** 2
    g = 2.0 * mu * (sigma - 1.0) + lam * (J - 1.0) * cofactor
    A = qd.Matrix.zero(gs.qd_float, 3, 3)
    for i in qd.static(range(3)):
        A[i, i] = 2.0 * mu + lam * cofactor[i] * cofactor[i]
    for k in qd.static(range(3)):
        i, j = qd.static(_PAIRS[k])
        axis_other = qd.static(3 - _PAIRS[k][0] - _PAIRS[k][1])
        A[i, j] = lam * (cofactor[i] * cofactor[j] + (J - 1.0) * sigma[axis_other])
        A[j, i] = A[i, j]
    if qd.static(is_penalized):
        penalty, g_penalty, A_penalty = compression_penalty(sigma, mu, lam)
        psi += penalty
        g += g_penalty
        for i in qd.static(range(3)):
            A[i, i] += A_penalty[i]
    return psi, g, A


@qd.func
def isotropic_pair_eigenvalues(sigma, g, A):
    """Twist and flip eigenvalues of an isotropic energy, clamped at zero (see isotropic_pair_eigenvalues_exact)."""
    twist, flip = isotropic_pair_eigenvalues_exact(sigma, g, A)
    return qd.max(twist, 0.0), qd.max(flip, 0.0)


@qd.func
def isotropic_pair_eigenvalues_exact(sigma, g, A):
    """Twist and flip eigenvalues of an isotropic energy for the axis pairs (0,1), (0,2), (1,2).

    The twist is (g_i + g_j) / (sigma_i + sigma_j) and the flip (g_i - g_j) / (sigma_i - sigma_j) (Smith et al. 2019).
    The flip takes its limit A_ii - A_ij on equal singular values. The twist denominator, which an inverted state can
    bring to zero, is kept at least 1e-6 in magnitude.
    """
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
def isotropic_first_piola_differential(U, V, sigma, g, A, dF):
    """Unprojected differential of the first Piola-Kirchhoff stress of an isotropic energy for a deformation increment.

    It is dP = (dP/dF) : dF at F = U diag(sigma) V^T, with `g` and `A` the principal gradient and Hessian. In the
    singular frame dF_hat = U^T dF V, the diagonal moves by A dF_hat_ii and each axis pair (i, j) by the unprojected
    flip/twist block dP_hat_ij = alp dF_hat_ij + bet dF_hat_ji, with alp = (flip + twist) / 2 and
    bet = (flip - twist) / 2. This is the operator whose projection `isotropic_pair_block` assembles.
    """
    dF_hat = U.transpose() @ dF @ V
    twist, flip = isotropic_pair_eigenvalues_exact(sigma, g, A)
    dP_hat = qd.Matrix.zero(gs.qd_float, 3, 3)
    for i in qd.static(range(3)):
        for j in qd.static(range(3)):
            dP_hat[i, i] += A[i, j] * dF_hat[j, j]
    for k in qd.static(range(3)):
        i, j = qd.static(_PAIRS[k])
        alp = 0.5 * (flip[k] + twist[k])
        bet = 0.5 * (flip[k] - twist[k])
        dP_hat[i, j] = alp * dF_hat[i, j] + bet * dF_hat[j, i]
        dP_hat[j, i] = bet * dF_hat[i, j] + alp * dF_hat[j, i]
    return U @ dP_hat @ V.transpose()


@qd.func
def isotropic_projected_modes(sigma, g, A, is_exact: qd.template() = False):
    """Symmetric positive definite (SPD) projection of the principal Hessian of an isotropic energy, as a scaling block
    and per-pair modes.

    Returns the clamped scaling block S = sum_k max(l_k, 0) a_k a_k^T of sym(A), with (l_k, a_k) the eigenpairs of
    sym(A), and per axis pair (0,1), (0,2), (1,2) the flip/twist combinations alp = (flip + twist) / 2 and
    bet = (flip - twist) / 2 of the clamped flip and twist eigenvalues. With `is_exact`, sym(A) and the eigenvalues stay
    unclamped, which yields the Hessian itself.
    """
    S = 0.5 * (A + A.transpose())
    twist = qd.Vector.zero(gs.qd_float, 3)
    flip = qd.Vector.zero(gs.qd_float, 3)
    if qd.static(is_exact):
        twist, flip = isotropic_pair_eigenvalues_exact(sigma, g, A)
    else:
        eigenvalues, Q = qd.sym_eig(S, gs.qd_float)
        S = qd.Matrix.zero(gs.qd_float, 3, 3)
        for k in qd.static(range(3)):
            eigenvector = qd.Vector([Q[d, k] for d in qd.static(range(3))], dt=gs.qd_float)
            S += qd.max(eigenvalues[k], 0.0) * eigenvector.outer_product(eigenvector)
        twist, flip = isotropic_pair_eigenvalues(sigma, g, A)
    return S, 0.5 * (flip + twist), 0.5 * (flip - twist)


@qd.func
def isotropic_pair_block(r_a, r_b, S, alp, bet):
    """Hessian block Z_ab of a node pair of the stencil in the V frame, from the modes of isotropic_projected_modes.

    With the stencil vectors r_a = V^T F^T grad w_a and r_b of the two nodes, Z_ab is diag(r_a) S diag(r_b) plus the
    four-entry flip/twist update of each axis pair. The world block is U Z_ab U^T times the particle volume and dt^2.
    """
    Z = qd.Matrix.zero(gs.qd_float, 3, 3)
    for i, j in qd.static(qd.ndrange(3, 3)):
        Z[i, j] = r_a[i] * S[i, j] * r_b[j]
    for k in qd.static(range(3)):
        i, j = qd.static(_PAIRS[k])
        Z[i, i] += alp[k] * r_a[j] * r_b[j]
        Z[j, j] += alp[k] * r_a[i] * r_b[i]
        Z[i, j] += bet[k] * r_a[j] * r_b[i]
        Z[j, i] += bet[k] * r_a[i] * r_b[j]
    return Z


# ============================================= Layer 0: brick layout =================================================


@qd.func
def brick_of(s):
    """Brick of a node slot, a brick holding 2 x 2 x 4 nodes of a block and a block holding four bricks."""
    return ((s >> 6) << 2) | (((s >> 5) & 1) << 1) | ((s >> 3) & 1)


@qd.func
def brick_node(s):
    """Node of a slot within its brick, increasing with the slot inside a brick so that a block with row <= column maps
    to a node pair a <= b."""
    return (((s >> 4) & 1) << 3) | (((s >> 2) & 1) << 2) | (s & 3)


@qd.func
def brick_slot(brick, node):
    """Inverse of (brick_of, brick_node)."""
    x = (((brick >> 1) & 1) << 1) | ((node >> 3) & 1)
    y = ((brick & 1) << 1) | ((node >> 2) & 1)
    return ((brick >> 2) << 6) | (x << 4) | (y << 2) | (node & 3)


@qd.func
def brick_sym_index(a, b):
    """Packed index of an upper-triangular node-pair block (a <= b) within a brick."""
    return BRICK_NODES * a - a * (a + 1) // 2 + b


@qd.func
def brick_entry_index(base, r, c):
    """Index in the packed storage of a brick of one entry of its 48 x 48 matrix.

    `base` is the start of the storage of the brick and (r, c) the entry, or (c, r) when that one is stored: the entries
    of the upper node blocks and of the upper triangle of a diagonal node block are stored.
    """
    a = r // 3
    b = c // 3
    i = r % 3
    j = c % 3
    index = 0
    if a < b:
        index = base + brick_sym_index(a, b) * 9 + 3 * i + j
    elif a > b:
        index = base + brick_sym_index(b, a) * 9 + 3 * j + i
    else:
        index = base + brick_sym_index(a, a) * 9 + 3 * qd.min(i, j) + qd.max(i, j)
    return index


@qd.func
def brick_slot_entry(
    s: qd.template(),  # int, the register slot
    diag_lane,
    P0,
    Q0,
):
    """Validity flag, row and column of a register slot of a lane in the brick inversion.

    A diagonal lane (`diag_lane`) holds the upper entries of its two diagonal tiles starting at P0 and Q0, and an
    off-diagonal lane its tile at rows from P0 and columns from Q0.
    """
    is_valid = 1
    r = 0
    c = 0
    if qd.static(s < _BRICK_TRI):
        if diag_lane:
            r = P0 + _BRICK_TRI_I[s]
            c = P0 + _BRICK_TRI_J[s]
        else:
            r = P0 + s // _BRICK_TILE
            c = Q0 + s % _BRICK_TILE
    elif qd.static(s < _BRICK_TILE * _BRICK_TILE):
        if diag_lane:
            r = Q0 + _BRICK_TRI_I[s - _BRICK_TRI]
            c = Q0 + _BRICK_TRI_J[s - _BRICK_TRI]
        else:
            r = P0 + s // _BRICK_TILE
            c = Q0 + s % _BRICK_TILE
    else:
        is_valid = qd.i32(diag_lane)
        r = Q0 + _BRICK_TRI_I[s - _BRICK_TRI]
        c = Q0 + _BRICK_TRI_J[s - _BRICK_TRI]
    return is_valid, r, c


# ========================================= Layer 1: per-item stencil access ==========================================


@qd.func
def material_param(
    materials: MpmMaterialTable,
    params: qd.template(),  # MpmParams
    param: qd.template(),  # int, a PARAM_* column
    slot,
):
    return materials.material_table[param * params.n_materials[()] + slot]


@qd.func
def material_params(
    materials: MpmMaterialTable,
    params: qd.template(),  # MpmParams
    slot,
    Jp,
):
    """Lame parameters of a material slot for a particle, hardened by its plastic state for snow.

    Snow scales them by exp(SNOW_HARDENING (1 - Jp)), with `Jp` the plastic state at the substep start, which stays
    frozen over the Newton solve.
    """
    mu = material_param(materials, params, PARAM_MU, slot)
    lam = material_param(materials, params, PARAM_LAM, slot)
    if materials.mat_plastic[slot] == PLASTIC_SNOW:
        hardening = qd.exp(qd.static(SNOW_HARDENING) * (1.0 - Jp))
        mu *= hardening
        lam *= hardening
    return mu, lam


@qd.func
def elastic_principal(
    materials: MpmMaterialTable,
    slot,
    sigma,
    mu,
    lam,
):
    """Energy density, principal gradient and Hessian of the energy family of a material slot."""
    psi = gs.qd_float(0.0)
    g = qd.Vector.zero(gs.qd_float, 3)
    A = qd.Matrix.zero(gs.qd_float, 3, 3)
    family = materials.mat_family[slot]
    if family == FAMILY_STABLE_NEOHOOKEAN:
        psi, g, A = stable_neohookean_principal(sigma, mu, lam)
    elif family == FAMILY_EXTENDED_HENCKY:
        psi, g, A = extended_hencky_principal(sigma, mu, lam)
    elif family == FAMILY_LIQUID:
        psi, g, A = liquid_principal(sigma, mu, lam)
    else:
        psi, g, A = fixed_corotated_principal(sigma, mu, lam)
    return psi, g, A


@qd.func
def elastic_energy(
    materials: MpmMaterialTable,
    slot,
    sigma,
    mu,
    lam,
):
    """Energy density of the energy family of a material slot, from the signed singular values."""
    psi = gs.qd_float(0.0)
    family = materials.mat_family[slot]
    if family == FAMILY_FIXED_COROTATED:
        psi = fixed_corotated_energy(sigma, mu, lam)
    else:
        psi, g, A = elastic_principal(materials, slot, sigma, mu, lam)
    return psi


@qd.func
def fibre_terms(
    materials: MpmMaterialTable,
    params: qd.template(),  # MpmParams
    slot,
    actu,
    m,
    F_tr,
):
    """Energy density, first Piola-Kirchhoff stress and Hessian curvature of the muscle fibre term of a particle.

    The term is 1/2 k a |F_trial m|^2, with k the fibre stiffness of the material, a the actuation `actu` and m the
    fibre direction. Its stress is k a (F_trial m) m^T and its Hessian k a (m m^T) (x) I, whose curvature is clamped to
    max(k a, 0) since an extension (k a < 0) makes it concave. All three vanish for a material without fibre stiffness.
    """
    ka = material_param(materials, params, PARAM_FIBRE, slot) * actu
    Fm = F_tr @ m
    return 0.5 * ka * Fm.dot(Fm), ka * Fm.outer_product(m), qd.max(ka, 0.0)


@qd.func
def return_map(
    materials: MpmMaterialTable,
    params: qd.template(),  # MpmParams
    slot,
    kind,
    F,
    Jp,
):
    """Plastic return map of a material slot, applied to the deformation gradient and the plastic state after the
    grid-to-particle (G2P) transfer.

    The map `kind` is von Mises in log strain or the singular-value clamp (ElastoPlastic), the clamp with the volume
    ratio it removes accumulated in Jp (Snow), Drucker-Prager with the log volume correction in Jp (Sand), or the
    dilation projection to a volume ratio in [LIQUID_J_MIN, 1] (Liquid). The formulas match the explicit materials.
    """
    F_new = F
    Jp_new = Jp
    if kind == PLASTIC_LIQUID:
        J = qd.min(qd.max(F.determinant(), qd.static(LIQUID_J_MIN)), 1.0)
        F_new = qd.pow(J, 1.0 / 3.0) * qd.Matrix.identity(gs.qd_float, 3)
    elif kind != PLASTIC_NONE:
        U, S, V = qd.svd(F, gs.qd_float)
        sigma = qd.Vector([S[0, 0], S[1, 1], S[2, 2]], dt=gs.qd_float)
        sigma_new = sigma
        is_changed = True
        mu = material_param(materials, params, PARAM_MU, slot)
        if kind == PLASTIC_VON_MISES:
            eps = qd.log(qd.max(sigma, qd.static(VON_MISES_SIGMA_MIN)))
            eps_hat = eps - eps.sum() / 3.0
            eps_hat_norm = eps_hat.norm(gs.EPS)
            delta_gamma = eps_hat_norm - material_param(materials, params, PARAM_YIELD_STRESS, slot) / (2.0 * mu)
            is_changed = delta_gamma > 0.0
            if is_changed:
                sigma_new = qd.exp(eps - (delta_gamma / eps_hat_norm) * eps_hat)
        elif kind == PLASTIC_SAND:
            lam = material_param(materials, params, PARAM_LAM, slot)
            eps = qd.log(qd.max(qd.abs(sigma), qd.static(SAND_SIGMA_MIN)))
            trace = eps.sum() + Jp
            eps_hat = eps - trace / 3.0
            eps_hat_norm = eps_hat.norm(gs.EPS)
            sigma_new = qd.Vector([1.0, 1.0, 1.0], dt=gs.qd_float)
            if trace >= 0.0:
                Jp_new = trace
            else:
                Jp_new = 0.0
                alpha = material_param(materials, params, PARAM_SAND_ALPHA, slot)
                delta_gamma = eps_hat_norm + (3.0 * lam + 2.0 * mu) / (2.0 * mu) * trace * alpha
                sigma_new = qd.exp(eps - qd.max(delta_gamma, 0.0) / eps_hat_norm * eps_hat)
        else:
            lower = 1.0 - material_param(materials, params, PARAM_YIELD_LOWER, slot)
            higher = 1.0 + material_param(materials, params, PARAM_YIELD_HIGHER, slot)
            sigma_new = qd.min(qd.max(sigma, lower), higher)
            if kind == PLASTIC_SNOW:
                Jp_new = Jp * (sigma[0] * sigma[1] * sigma[2]) / (sigma_new[0] * sigma_new[1] * sigma_new[2])
        if is_changed:
            F_new = (
                U
                @ qd.Matrix(
                    [[sigma_new[0], 0.0, 0.0], [0.0, sigma_new[1], 0.0], [0.0, 0.0, sigma_new[2]]], dt=gs.qd_float
                )
                @ V.transpose()
            )
    return F_new, Jp_new


@qd.func
def particle_of_sorted(
    grid: qd.template(),  # MpmGrid
    params: qd.template(),  # MpmParams
    s,
):
    """Particle index, environment index (from its id and from its sort key) and cell of a sorted particle."""
    gid = grid.sort_ids[s]
    i_b = gid // params.n_particles[()]
    i_p = gid - i_b * params.n_particles[()]
    key = grid.sort_keys[s]
    env, block = unpack_block(key >> qd.u64(6))
    cell = block * 4 + local_offset(qd.i32(key & qd.u64(63)))
    return i_p, i_b, env, cell


@qd.func
def stencil_node_slots(
    grid: qd.template(),  # MpmGrid
    env,
    cell,
):
    """Node slots of the 8 nodes of a cell.

    Every node exists, since a particle block activates its neighbours. A missing block fails the assertion in debug
    mode, and otherwise raises INVALID_MPM_GRID and reads block 0 instead of reading out of bounds.
    """
    slots = qd.Vector.zero(qd.i32, 8)
    for a in qd.static(range(8)):
        node = cell + node_offset(a)
        block_slot = grid.hash.lookup(pack_block(env, block_of(node)))
        if qd.static(qd_debug_asserts()):
            assert block_slot >= 0, "MPM grid: no block holds stencil node (%d, %d, %d) of env %d" % (
                node[0],
                node[1],
                node[2],
                env,
            )
        if block_slot < 0:
            flag_env_error(grid, env, array_class.ErrorCode.INVALID_MPM_GRID)
            block_slot = 0
        slots[a] = block_slot * BLOCK_CELLS + cell_local(node)
    return slots


@qd.func
def trial_deformation(
    particles: MpmParticles,
    grid: qd.template(),  # MpmGrid
    newton: qd.template(),  # MpmNewton
    params: qd.template(),  # MpmParams
    f,
    i_p,
    i_b,
    cell,
    slots,
):
    """Trial deformation gradient F_trial = (I + sum_a u_a grad w_a^T) F of a particle, with its stencil gradients and
    its deformation gradient F."""
    fx = particles.pos[f, i_p, i_b] * params.inv_dx[()] - cell.cast(gs.qd_float)
    w, grad_w = q1_stencil(fx, params.inv_dx[()])
    grad_u = qd.Matrix.zero(gs.qd_float, 3, 3)
    for a in qd.static(range(8)):
        u_a = qd.Vector([newton.u[3 * slots[a] + d] for d in qd.static(range(3))], dt=gs.qd_float)
        grad_u += u_a.outer_product(qd.Vector([grad_w[a, d] for d in qd.static(range(3))], dt=gs.qd_float))
    F = particles.F[f, i_p, i_b]
    return (qd.Matrix.identity(gs.qd_float, 3) + grad_u) @ F, grad_w, F


# ============================================== Layer 1: grid rebuild ================================================


@qd.func(requires_top_level=True)
def func_compose_particle_keys(
    particles: MpmParticles,
    particles_ng: MpmParticlesNG,
    grid: qd.template(),  # MpmGrid
    params: qd.template(),  # MpmParams
    f,
):
    """Write the sort key of every particle, environment-major.

    An inactive particle takes the empty key and sorts last. A position that is not finite or whose cell falls outside
    [CELL_MIN, CELL_END) fails the assertion in debug mode, and otherwise raises OVERFLOW_MPM_GRID_RANGE and takes the
    empty key, which leaves it out of the grid."""
    qd.loop_config(name="mpm_compose_particle_keys")
    for gid in range(params.n_envs[()] * params.n_particles[()]):
        i_b = gid // params.n_particles[()]
        i_p = gid - i_b * params.n_particles[()]
        key = qd.u64(EMPTY_KEY)
        if particles_ng.active[f, i_p, i_b]:
            pos = particles.pos[f, i_p, i_b]
            # Fast math lets the compiler assume that the result of an arithmetic operation is finite, which folds a
            # test of it away: only the loaded position can be tested for being finite
            is_in_range = qd_isfinite(pos)
            x = pos * params.inv_dx[()]
            for d in qd.static(range(3)):
                if not (x[d] >= CELL_MIN and x[d] < CELL_END):
                    is_in_range = False
            if qd.static(qd_debug_asserts()):
                assert is_in_range, "MPM grid: particle %d of env %d is not finite or out of the grid range" % (
                    i_p,
                    i_b,
                )
            if is_in_range:
                key = particle_key(i_b, qd.floor(x).cast(qd.i32))
            else:
                flag_env_error(grid, i_b, array_class.ErrorCode.OVERFLOW_MPM_GRID_RANGE)
        grid.sort_keys[gid] = key
        grid.sort_ids[gid] = gid
        if gid == 0:
            grid.n_sort_keys[()] = params.n_envs[()] * params.n_particles[()]


@qd.func(requires_top_level=True)
def func_sort_particle_keys(
    grid: qd.template(),  # MpmGrid
    params: qd.template(),  # MpmParams
):
    func_dynamic_radix_sort(
        grid.particle_sort,
        grid.sort_keys,
        grid.sort_keys_tmp,
        grid.sort_ids,
        grid.sort_ids_tmp,
        params.n_envs[()] * params.n_particles[()],
    )


@qd.func(requires_top_level=True)
def func_append_particle_blocks(
    grid: qd.template(),  # MpmGrid
    params: qd.template(),  # MpmParams
):
    """Count the valid sorted particles and append the key of every distinct particle block."""
    for _ in range(1):
        grid.n_valid[()] = 0
        grid.n_particle_blocks[()] = 0
    qd.loop_config(name="mpm_append_particle_blocks")
    for s in range(params.n_envs[()] * params.n_particles[()]):
        key = grid.sort_keys[s]
        if key != qd.u64(EMPTY_KEY):
            if s == params.n_envs[()] * params.n_particles[()] - 1 or grid.sort_keys[s + 1] == qd.u64(EMPTY_KEY):
                grid.n_valid[()] = s + 1
            block_key = key >> qd.u64(6)
            if s == 0 or (grid.sort_keys[s - 1] >> qd.u64(6)) != block_key:
                index = qd.atomic_add(grid.n_particle_blocks[()], 1)
                if index < grid.pblock_keys.shape[0]:
                    grid.pblock_keys[index] = block_key
    for _ in range(1):
        n_valid = grid.n_valid[()]
        grid.n_valid_chunks[()] = (n_valid + _WARP - 1) // _WARP
        grid.n_valid_reduce_blocks[()] = (n_valid + REDUCE_BLOCK_DIM - 1) // REDUCE_BLOCK_DIM
        n_particle_blocks = qd.min(grid.n_particle_blocks[()], grid.pblock_keys.shape[0])
        grid.n_neighbor_keys[()] = 8 * n_particle_blocks
        if grid.n_particle_blocks[()] > grid.pblock_keys.shape[0]:
            grid.grid_overflow[()] = 1


@qd.func(requires_top_level=True)
def func_gen_neighbor_keys(
    grid: qd.template(),  # MpmGrid
):
    """Every particle block activates its positive-octant neighbour blocks, which hold the nodes its cells touch."""
    qd.loop_config(name="mpm_gen_neighbor_keys")
    for i in range(grid.n_neighbor_keys[()]):
        env, block = unpack_block(grid.pblock_keys[i >> 3])
        o = i & 7
        grid.nbr_keys[i] = pack_block(env, block + qd.Vector([(o >> 2) & 1, (o >> 1) & 1, o & 1], dt=qd.i32))
        grid.nbr_perm[i] = i


@qd.func(requires_top_level=True)
def func_sort_neighbor_keys(
    grid: qd.template(),  # MpmGrid
):
    func_dynamic_radix_sort(
        grid.block_sort, grid.nbr_keys, grid.nbr_keys_tmp, grid.nbr_perm, grid.nbr_perm_tmp, grid.n_neighbor_keys[()]
    )


@qd.func(requires_top_level=True)
def func_block_flags(
    grid: qd.template(),  # MpmGrid
):
    qd.loop_config(name="mpm_block_flags")
    for i in range(grid.n_neighbor_keys[()]):
        flag = 0
        if i == 0 or grid.nbr_keys[i] != grid.nbr_keys[i - 1]:
            flag = 1
        grid.block_flags[i] = flag


@qd.func(requires_top_level=True)
def func_publish_blocks(
    grid: qd.template(),  # MpmGrid
):
    """Give every distinct block the rank of its sorted key as slot, and insert it in the hash table.

    Runs after the exclusive sum of the block flags into the block ranks."""
    for _ in range(1):
        grid.grid_demand[()] = 0
    qd.loop_config(name="mpm_publish_blocks")
    for i in range(grid.n_neighbor_keys[()]):
        if grid.block_flags[i] != 0:
            slot = grid.block_ranks[i]
            if slot < grid.block_coords.shape[0]:
                grid.block_coords[slot] = grid.nbr_keys[i]
                if not grid.hash.insert(grid.nbr_keys[i], slot):
                    env, block = unpack_block(grid.nbr_keys[i])
                    flag_env_error(grid, env, array_class.ErrorCode.INVALID_MPM_GRID)
        if i == grid.n_neighbor_keys[()] - 1:
            grid.grid_demand[()] = grid.block_ranks[i] + grid.block_flags[i]


@qd.func(requires_top_level=True)
def func_cell_flags(
    grid: qd.template(),  # MpmGrid
):
    """Flag the first sorted particle of every cell and of every task.

    On a GPU, a task is the part of a cell inside one 32-particle chunk, which the warp of the chunk assembles. On a CPU
    backend, a task is a whole cell, which one thread assembles, so that the task ranks equal the cell ranks.
    """
    qd.loop_config(name="mpm_cell_flags")
    for s in range(grid.n_valid[()]):
        cell_flag = 0
        task_flag = 0
        if s == 0 or grid.sort_keys[s] != grid.sort_keys[s - 1]:
            cell_flag = 1
        if qd.static(gs.backend == gs.cpu):
            task_flag = cell_flag
        elif cell_flag != 0 or s % _WARP == 0:
            task_flag = 1
        grid.cell_flags[s] = cell_flag
        grid.task_flags[s] = task_flag


@qd.func(requires_top_level=True)
def func_publish_cells(
    grid: qd.template(),  # MpmGrid
):
    """Write the cell starts and the cell and task counts, after the exclusive sums of the cell and task flags."""
    for _ in range(1):
        grid.n_active_cells[()] = 0
        grid.n_tasks[()] = 0
    qd.loop_config(name="mpm_publish_cells")
    for s in range(grid.n_valid[()]):
        if grid.cell_flags[s] != 0:
            i_cell = grid.cell_ranks[s]
            grid.cell_starts[i_cell] = s
        if s == grid.n_valid[()] - 1:
            grid.n_active_cells[()] = grid.cell_ranks[s] + grid.cell_flags[s]
            grid.n_tasks[()] = grid.task_ranks[s] + grid.task_flags[s]
    for _ in range(1):
        n_active_cells = grid.n_active_cells[()]
        grid.cell_starts[n_active_cells] = grid.n_valid[()]


@qd.func(requires_top_level=True)
def func_reset_cell_segs(
    grid: qd.template(),  # MpmGrid
):
    qd.loop_config(name="mpm_reset_cell_segs")
    for slot in range(grid.n_blocks[()] * BLOCK_CELLS):
        grid.cell_seg_start[slot] = 0
        grid.cell_seg_end[slot] = 0


@qd.func(requires_top_level=True)
def func_fill_cell_segs(
    grid: qd.template(),  # MpmGrid
):
    """Fill the dense table from cell slot to the range of its sorted particles.

    Past an overflow, the blocks beyond the capacity are missing from the grid until it grows and builds again.
    Otherwise the grid holds the block of every particle cell: a missing one fails the assertion in debug mode, or else
    raises INVALID_MPM_GRID and leaves its cell out.
    """
    qd.loop_config(name="mpm_fill_cell_segs")
    for r in range(grid.n_active_cells[()]):
        start = grid.cell_starts[r]
        key = grid.sort_keys[start]
        block_slot = grid.hash.lookup(key >> qd.u64(6))
        is_missing = block_slot < 0 and grid.grid_overflow[()] == 0
        if qd.static(qd_debug_asserts()):
            assert not is_missing, "MPM grid: no block holds the cell of sorted particle %d" % start
        if is_missing:
            env, block = unpack_block(key >> qd.u64(6))
            flag_env_error(grid, env, array_class.ErrorCode.INVALID_MPM_GRID)
        if block_slot >= 0:
            slot = block_slot * BLOCK_CELLS + qd.i32(key & qd.u64(63))
            grid.cell_seg_start[slot] = start
            grid.cell_seg_end[slot] = grid.cell_starts[r + 1]


@qd.func(requires_top_level=True)
def func_publish_grid_demand(
    grid: qd.template(),  # MpmGrid
):
    """Publish the live block count, clamped to the capacity, and raise the overflow flag past it."""
    for _ in range(1):
        demand = grid.grid_demand[()]
        n_blocks = qd.min(demand, grid.block_coords.shape[0])
        grid.n_blocks[()] = n_blocks
        grid.n_node_reduce_blocks[()] = (n_blocks * BLOCK_CELLS + REDUCE_BLOCK_DIM - 1) // REDUCE_BLOCK_DIM
        if demand > grid.block_coords.shape[0]:
            grid.grid_overflow[()] = 1


@qd.func(requires_top_level=True)
def func_reset_grid(
    grid: qd.template(),  # MpmGrid
):
    for _ in range(1):
        grid.grid_overflow[()] = 0
    func_reset_hash(grid.hash)


@qd.func(requires_top_level=True)
def func_build_grid(
    particles: MpmParticles,
    particles_ng: MpmParticlesNG,
    grid: qd.template(),  # MpmGrid
    params: qd.template(),  # MpmParams
    f,
):
    """Rebuild the block-sparse grid of the substep from the particles, its topology fixed until the next substep."""
    func_reset_grid(grid)
    func_compose_particle_keys(particles, particles_ng, grid, params, f)
    func_sort_particle_keys(grid, params)
    func_append_particle_blocks(grid, params)
    func_gen_neighbor_keys(grid)
    func_sort_neighbor_keys(grid)
    func_block_flags(grid)
    func_dynamic_exclusive_sum(grid.block_scan, grid.block_flags, grid.block_ranks, grid.n_neighbor_keys[()])
    func_publish_blocks(grid)
    func_publish_grid_demand(grid)
    func_cell_flags(grid)
    func_dynamic_exclusive_sum(grid.cell_scan, grid.cell_flags, grid.cell_ranks, grid.n_valid[()])
    func_dynamic_exclusive_sum(grid.task_scan, grid.task_flags, grid.task_ranks, grid.n_valid[()])
    func_publish_cells(grid)
    func_reset_cell_segs(grid)
    func_fill_cell_segs(grid)


# ================================================ Layer 1: transfers =================================================


@qd.func(requires_top_level=True)
def func_p2g(
    particles: MpmParticles,
    particles_info: MpmParticlesInfo,
    grid: qd.template(),  # MpmGrid
    params: qd.template(),  # MpmParams
    gravity: qd.template(),  # qd.Tensor of per-environment gravity, shape (B,) of vec3
    f,
    dt,
):
    """Particle-to-grid (P2G) transfer of the node mass and free velocity, gathered per node.

    The velocity is the affine particle-in-cell (APIC) one, gravity included. A node touched by a non-free particle is
    fixed. The gather sums the particles of a node in a fixed order, which a scatter of atomic adds would vary from run
    to run. A GPU spreads the particles of a node over the lanes of a warp and sums them by shuffles, and a CPU thread
    walks them all itself."""
    if qd.static(gs.backend == gs.cpu):
        func_p2g_cpu(particles, particles_info, grid, params, gravity, f, dt)
    else:
        func_p2g_gpu(particles, particles_info, grid, params, gravity, f, dt)


@qd.func
def apic_node_contribution(
    particles: MpmParticles,
    particles_info: MpmParticlesInfo,
    grid: qd.template(),  # MpmGrid
    params: qd.template(),  # MpmParams
    gravity: qd.template(),  # qd.Tensor of per-environment gravity, shape (B,) of vec3
    f,
    dt,
    s,
    c,
    o,
    x_n,
):
    """Mass and affine particle-in-cell (APIC) momentum that a sorted particle brings to one node of its stencil, and
    whether the particle is free.

    `s` is the sorted particle, of cell `c`, and the node is c + o, at position `x_n`.
    """
    gid = grid.sort_ids[s]
    i_b = gid // params.n_particles[()]
    i_p = gid - i_b * params.n_particles[()]
    x_p = particles.pos[f, i_p, i_b]
    fx = x_p * params.inv_dx[()] - c.cast(gs.qd_float)
    w, grad_w = q1_node_weight(fx, o, params.inv_dx[()])
    m = w * (particles_info.mass[i_p] / params.mass_scale[()])
    v_p = particles.vel[f, i_p, i_b] + dt * gravity[i_b] + particles.C[f, i_p, i_b] @ (x_n - x_p)
    return m, m * v_p, particles_info.free[i_p]


@qd.func
def publish_node_mass_velocity(
    grid: qd.template(),  # MpmGrid
    node,
    mass,
    mom,
    fixed,
):
    grid.node_mass[node] = mass
    grid.node_fixed[node] = fixed
    for d in qd.static(range(3)):
        v = gs.qd_float(0.0)
        if mass > 0.0 and fixed == 0:
            v = mom[d] / mass
        grid.node_v[3 * node + d] = v


@qd.func(requires_top_level=True)
def func_p2g_cpu(
    particles: MpmParticles,
    particles_info: MpmParticlesInfo,
    grid: qd.template(),  # MpmGrid
    params: qd.template(),  # MpmParams
    gravity: qd.template(),  # qd.Tensor of per-environment gravity, shape (B,) of vec3
    f,
    dt,
):
    """func_p2g by one thread per node, the threads splitting the nodes, each walking the particles of the 8 cells
    around its node in order."""
    qd.loop_config(name="mpm_p2g_cpu")
    for node in range(grid.n_blocks[()] * BLOCK_CELLS):
        env, block = unpack_block(grid.block_coords[node // BLOCK_CELLS])
        n = block * 4 + local_offset(node % BLOCK_CELLS)
        x_n = n.cast(gs.qd_float) * params.dx[()]
        mass = gs.qd_float(0.0)
        momentum = qd.Vector.zero(gs.qd_float, 3)
        is_fixed = 0
        for a in qd.static(range(8)):
            o = node_offset(a)
            c = n - o
            block_slot = grid.hash.lookup(pack_block(env, block_of(c)))
            if block_slot >= 0:
                cell_slot = block_slot * BLOCK_CELLS + cell_local(c)
                for s in range(grid.cell_seg_start[cell_slot], grid.cell_seg_end[cell_slot]):
                    m, m_v, is_free = apic_node_contribution(
                        particles, particles_info, grid, params, gravity, f, dt, s, c, o, x_n
                    )
                    mass += m
                    momentum += m_v
                    if not is_free:
                        is_fixed = 1
        publish_node_mass_velocity(grid, node, mass, momentum, is_fixed)


@qd.func(requires_top_level=True)
def func_p2g_gpu(
    particles: MpmParticles,
    particles_info: MpmParticlesInfo,
    grid: qd.template(),  # MpmGrid
    params: qd.template(),  # MpmParams
    gravity: qd.template(),  # qd.Tensor of per-environment gravity, shape (B,) of vec3
    f,
    dt,
):
    """func_p2g by one warp per node.

    Lane 0 resolves the particle range of each of the 8 cells around the node and broadcasts it. The lanes stride the
    range and a butterfly reduces the warp."""
    qd.loop_config(name="mpm_p2g", block_dim=128)
    for t in range(grid.n_blocks[()] * BLOCK_CELLS * _WARP):
        lane = qd.i32(qd.simt.subgroup.invocation_id()) & (_WARP - 1)
        node = t // _WARP
        env, block = unpack_block(grid.block_coords[node // BLOCK_CELLS])
        n = block * 4 + local_offset(node % BLOCK_CELLS)
        x_n = n.cast(gs.qd_float) * params.dx[()]

        mass = gs.qd_float(0.0)
        momentum = qd.Vector.zero(gs.qd_float, 3)
        is_fixed = 0
        for a in qd.static(range(8)):
            o = node_offset(a)
            c = n - o
            lo = 0
            hi = 0
            if lane == 0:
                block_slot = grid.hash.lookup(pack_block(env, block_of(c)))
                if block_slot >= 0:
                    cell_slot = block_slot * BLOCK_CELLS + cell_local(c)
                    lo = grid.cell_seg_start[cell_slot]
                    hi = grid.cell_seg_end[cell_slot]
            lo = qd.simt.subgroup.shuffle(lo, qd.u32(0))
            hi = qd.simt.subgroup.shuffle(hi, qd.u32(0))
            for i_chunk_ in range((hi - lo + _WARP - 1) // _WARP):
                s = lo + i_chunk_ * _WARP + lane
                if s < hi:
                    m, m_v, is_free = apic_node_contribution(
                        particles, particles_info, grid, params, gravity, f, dt, s, c, o, x_n
                    )
                    mass += m
                    momentum += m_v
                    if not is_free:
                        is_fixed = 1
        mass = warp_sum(mass)
        for d in qd.static(range(3)):
            momentum[d] = warp_sum(momentum[d])
        is_fixed = warp_max(is_fixed)
        if lane == 0:
            publish_node_mass_velocity(grid, node, mass, momentum, is_fixed)


@qd.func(requires_top_level=True)
def func_apply_force_fields(
    grid: qd.template(),  # MpmGrid
    force_fields: qd.template(),  # MpmForceFields
    params: qd.template(),  # MpmParams
    t,
    dt,
):
    """node_v_free = node_v + dt * the accelerations of the scene force fields at the node."""
    qd.loop_config(name="mpm_apply_force_fields")
    for node in range(grid.n_blocks[()] * BLOCK_CELLS):
        env, block = unpack_block(grid.block_coords[node // BLOCK_CELLS])
        x_n = (block * 4 + local_offset(node % BLOCK_CELLS)).cast(gs.qd_float) * params.dx[()]
        v = qd.Vector([grid.node_v[3 * node + d] for d in qd.static(range(3))], dt=gs.qd_float)
        if grid.node_mass[node] > 0.0 and grid.node_fixed[node] == 0:
            acceleration = qd.Vector.zero(gs.qd_float, 3)
            for i_ff in qd.static(range(len(force_fields.force_fields))):
                acceleration += force_fields.force_fields[i_ff].get_acc(x_n, v, t, i=-1)
            v += dt * acceleration
        for d in qd.static(range(3)):
            grid.node_v_free[3 * node + d] = v[d]


@qd.func(requires_top_level=True)
def func_fix_nodes(
    grid: qd.template(),  # MpmGrid
):
    """Hold every displacement component of a node touched by a non-free particle."""
    qd.loop_config(name="mpm_fix_nodes")
    for node in range(grid.n_blocks[()] * BLOCK_CELLS):
        for d in qd.static(range(3)):
            grid.dof_fixed[3 * node + d] = grid.node_fixed[node]


@qd.func(requires_top_level=True)
def func_reset_contact_rows(
    grid: qd.template(),  # MpmGrid
    newton: qd.template(),  # MpmNewton
):
    """Clear the contact rows of every valid particle, so that an assembly before the rows are built is physical."""
    qd.loop_config(name="mpm_reset_contact_rows")
    for s in range(grid.n_valid[()]):
        newton.contact_link[s] = -1
        newton.contact_tri[s] = -1
        newton.contact_pbd[s] = -1
        for r in qd.static(range(CONTACT_ROWS)):
            newton.contact_D[CONTACT_ROWS * s + r] = 0.0
        for d in qd.static(range(3)):
            newton.contact_dr[3 * s + d] = 0.0


@qd.func(requires_top_level=True)
def func_gather_node_diag(
    grid: qd.template(),  # MpmGrid
    linear: qd.template(),  # MpmLinearSystem
):
    """Diagonal block of every node of the reduced Hessian, assembled before any contact row exists."""
    qd.loop_config(name="mpm_gather_node_diag")
    for entry in range(linear.matrix.bcoo_nnz[()]):
        row, col, block = linear.matrix.read_bcoo(entry)
        if row == col:
            node = row - linear.row_offset
            for i, j in qd.static(qd.ndrange(3, 3)):
                grid.node_diag[9 * node + 3 * i + j] = block[i, j]


@qd.func(requires_top_level=True)
def func_contact_invweight(
    particles: MpmParticles,
    particles_info: MpmParticlesInfo,
    grid: qd.template(),  # MpmGrid
    linear: qd.template(),  # MpmLinearSystem
    params: qd.template(),  # MpmParams
    f,
):
    """Compute the inverse weight matrix of every valid particle of the first environment for its contact rows.

    It comes from the physical Hessian (kinetic and elastic) at u = 0 that the matrix holds, assembled in the
    configuration of frame `f`. It is W = sum_a w_a^2 H_aa^-1 over the stencil nodes of the particle, whose n^T W n
    along the normal n of a row is the block-diagonal approximation of the Delassus diagonal J H^-1 J^T, in the frame of
    the rest configuration (see contact_invweight). It plays the part of the inverse weight of a rigid link, so that a
    row pushes the material the particle is embedded in rather than the particle mass alone. Like it, it is a model
    constant of the rest configuration (see MPMSolver._init_contact_invweight): the stiffness of a row depends on the
    state through its normal and the rotation of the material alone."""
    func_gather_node_diag(grid, linear)
    qd.loop_config(name="mpm_contact_invweight")
    for s in range(grid.n_valid[()]):
        i_p, i_b, env, cell = particle_of_sorted(grid, params, s)
        if i_b == 0:
            slots = stencil_node_slots(grid, env, cell)
            fx = particles.pos[f, i_p, i_b] * params.inv_dx[()] - cell.cast(gs.qd_float)
            w, grad_w = q1_stencil(fx, params.inv_dx[()])
            W = qd.Matrix.zero(gs.qd_float, 3, 3)
            for a in qd.static(range(8)):
                H = qd.Matrix.zero(gs.qd_float, 3, 3)
                for i, j in qd.static(qd.ndrange(3, 3)):
                    H[i, j] = grid.node_diag[9 * slots[a] + 3 * i + j]
                W += w[a] * w[a] * H.inverse()
            particles_info.contact_invweight[i_p] = 0.5 * (W + W.transpose())


@qd.func
def contact_invweight(
    particles: MpmParticles,
    particles_info: MpmParticlesInfo,
    f,
    i_p,
    i_b,
    n,
):
    """Inverse weight of a particle along a direction, its rest-frame matrix rotated along with the material.

    It is n^T R W R^T n for direction `n` in frame `f`, with W the inverse weight matrix of particle `i_p` in the rest
    frame (see func_contact_invweight) and R the rotation of the polar decomposition F = R S of its deformation
    gradient, which carries the rest frame along with the material."""
    U, S, V = qd.svd(particles.F[f, i_p, i_b], gs.qd_float)
    n_rest = V @ (U.transpose() @ n)
    return n_rest.dot(particles_info.contact_invweight[i_p] @ n_rest)


@qd.func
def contact_band_width(
    params: qd.template(),  # MpmParams
    approach,
    dt,
):
    """Gap within which a particle gets a contact row at the substep start.

    The band is twice the distance that the particle closes over the substep at normal speed `approach`, plus one cell
    for the deformation of the entity it belongs to."""
    return 2.0 * dt * qd.max(approach, 0.0) + params.dx[()]


@qd.func
def band_row(
    sol_params,
    phi,
    vel,
    diag_coef,
    dt,
):
    """Stiffness D and target t of a soft row on the displacement relative to the obstacle.

    `phi` is the signed gap and `vel` the normal relative velocity, positive when separating. Within the contact range
    (phi <= 0), the row is the soft constraint of Rigid (`soft_row`). Short of it (phi > 0), the row lets the particle
    close the gap freely and resists past it with the stiffness of the edge of the range: t = -phi, so that the row
    energy is zero until the gap closes. Its reference acceleration would pull towards the obstacle there, so it is left
    out, and the next substep takes the row over with it once the particle is in range.
    """
    D = gs.qd_float(0.0)
    t = gs.qd_float(0.0)
    if phi <= 0.0:
        D, t = soft_row(sol_params, phi, vel, diag_coef, dt)
    else:
        D, t_edge = soft_row(sol_params, phi=0.0, vel=0.0, diag_coef=diag_coef, dt=dt)
        t = -phi
    return D, t


@qd.func(requires_top_level=True)
def func_build_wall_rows(
    particles: MpmParticles,
    particles_info: MpmParticlesInfo,
    grid: qd.template(),  # MpmGrid
    newton: qd.template(),  # MpmNewton
    params: qd.template(),  # MpmParams
    f,
    dt,
):
    """Build the frictionless domain-wall rows of every valid particle, after `func_reset_contact_rows`.

    A particle beyond a wall plane `params.wall_lower` / `wall_upper`, or short of the nearer one by less than
    `contact_band_width`, gets one soft row along the inward wall normal by `band_row`, with Rigid's impedance of
    `params.wall_sol_params` and the particle's inverse weight (see func_contact_invweight).
    """
    qd.loop_config(name="mpm_build_wall_rows")
    for s in range(grid.n_valid[()]):
        i_p, i_b, env, cell = particle_of_sorted(grid, params, s)
        x_p = particles.pos[f, i_p, i_b]
        v_p = particles.vel[f, i_p, i_b]
        sol_params = qd.Vector([params.wall_sol_params[i] for i in qd.static(range(7))], dt=gs.qd_float)
        for d in qd.static(range(3)):
            # Distance to the nearer wall plane, positive inside the domain, and the inward normal
            phi = x_p[d] - params.wall_lower[d]
            sign = gs.qd_float(1.0)
            if params.wall_upper[d] - x_p[d] < phi:
                phi = params.wall_upper[d] - x_p[d]
                sign = -1.0
            vel = sign * v_p[d]
            if phi < contact_band_width(params, -vel, dt):
                n = qd.Vector.zero(gs.qd_float, 3)
                n[d] = sign
                invweight = contact_invweight(particles, particles_info, f, i_p, i_b, n)
                D, t = band_row(sol_params, phi, vel, invweight, dt)
                k = CONTACT_ROWS * s + d
                for e in qd.static(range(3)):
                    newton.contact_dir[3 * k + e] = 0.0
                newton.contact_dir[3 * k + d] = sign
                newton.contact_D[k] = D
                newton.contact_t[k] = t


@qd.func(requires_top_level=True)
def func_grid_velocity(
    grid: qd.template(),  # MpmGrid
    newton: qd.template(),  # MpmNewton
    dt,
):
    qd.loop_config(name="mpm_grid_velocity")
    for i in range(3 * grid.n_blocks[()] * BLOCK_CELLS):
        grid.node_v_solved[i] = newton.u[i] / dt


@qd.func(requires_top_level=True)
def func_g2p_advect(
    particles: MpmParticles,
    particles_ng: MpmParticlesNG,
    particles_info: MpmParticlesInfo,
    materials: MpmMaterialTable,
    grid: qd.template(),  # MpmGrid
    params: qd.template(),  # MpmParams
    f,
    dt,
):
    """Advance every particle to the next frame: grid-to-particle (G2P) transfer, advection, deformation gradient update
    and plastic return map.

    The velocity is the particle-in-cell (PIC) one and the velocity gradient the affine particle-in-cell (APIC) one.
    Valid particles run in sorted order for locality, and inactive ones carry their state over unchanged.
    """
    qd.loop_config(name="mpm_g2p_advect")
    for s in range(grid.n_valid[()]):
        i_p, i_b, env, cell = particle_of_sorted(grid, params, s)
        slots = stencil_node_slots(grid, env, cell)
        pos = particles.pos[f, i_p, i_b]
        fx = pos * params.inv_dx[()] - cell.cast(gs.qd_float)
        w, grad_w = q1_stencil(fx, params.inv_dx[()])
        vel = qd.Vector.zero(gs.qd_float, 3)
        C = qd.Matrix.zero(gs.qd_float, 3, 3)
        for a in qd.static(range(8)):
            v_a = qd.Vector([grid.node_v_solved[3 * slots[a] + d] for d in qd.static(range(3))], dt=gs.qd_float)
            vel += w[a] * v_a
            C += v_a.outer_product(qd.Vector([grad_w[a, d] for d in qd.static(range(3))], dt=gs.qd_float))
        particles.pos[f + 1, i_p, i_b] = pos + dt * vel
        particles.vel[f + 1, i_p, i_b] = vel
        particles.C[f + 1, i_p, i_b] = C
        slot = particles_info.material_idx[i_p]
        F_new, Jp_new = return_map(
            materials,
            params,
            slot,
            materials.mat_plastic[slot],
            (qd.Matrix.identity(gs.qd_float, 3) + dt * C) @ particles.F[f, i_p, i_b],
            particles.Jp[f, i_p, i_b],
        )
        particles.F[f + 1, i_p, i_b] = F_new
        particles.Jp[f + 1, i_p, i_b] = Jp_new
        particles.actu[f + 1, i_p, i_b] = particles.actu[f, i_p, i_b]

    qd.loop_config(name="mpm_carry_inactive")
    for i_p, i_b in qd.ndrange(params.n_particles[()], params.n_envs[()]):
        if not particles_ng.active[f, i_p, i_b]:
            particles.pos[f + 1, i_p, i_b] = particles.pos[f, i_p, i_b]
            particles.vel[f + 1, i_p, i_b] = particles.vel[f, i_p, i_b]
            particles.C[f + 1, i_p, i_b] = particles.C[f, i_p, i_b]
            particles.F[f + 1, i_p, i_b] = particles.F[f, i_p, i_b]
            particles.Jp[f + 1, i_p, i_b] = particles.Jp[f, i_p, i_b]
            particles.actu[f + 1, i_p, i_b] = particles.actu[f, i_p, i_b]
        particles_ng.active[f + 1, i_p, i_b] = particles_ng.active[f, i_p, i_b]


# ================================================== Layer 1: Newton ==================================================


@qd.func
def contact_row_terms(
    newton: qd.template(),  # MpmNewton
    s,
    dx,
    r_lo: qd.template(),  # int
    r_hi: qd.template(),  # int
):
    """Energy, gradient and Hessian of the active soft contact rows of a sorted particle at a trial displacement.

    The rows are [r_lo, r_hi) of sorted particle `s`, and `dx` is the trial displacement of the particle for the wall
    rows, relative to its contact point on the link for the Rigid rows. A row is active where it is violated at dx, as
    in the Newton solver of Rigid."""
    energy = gs.qd_float(0.0)
    grad = qd.Vector.zero(gs.qd_float, 3)
    hess = qd.Matrix.zero(gs.qd_float, 3, 3)
    for r in qd.static(range(r_lo, r_hi)):
        k = CONTACT_ROWS * s + r
        D = newton.contact_D[k]
        if D > 0.0:
            J = qd.Vector([newton.contact_dir[3 * k + d] for d in qd.static(range(3))], dt=gs.qd_float)
            c = J.dot(dx) - newton.contact_t[k]
            if c < 0.0:
                energy += 0.5 * D * c * c
                grad += D * c * J
                hess += D * J.outer_product(J)
    return energy, grad, hess


@qd.func
def constraint_row_terms(
    newton: qd.template(),  # MpmNewton
    s,
    x,
    dx,
):
    """Energy, gradient and Hessian scale of the constraint spring of a sorted particle.

    `x` is the position of sorted particle `s` and `dx` its trial displacement (see func_build_constraint_rows). All
    three are zero for an unconstrained particle."""
    spring = newton.constraint_spring[s]
    damping = newton.constraint_damping[s]
    target = qd.Vector([newton.constraint_target[3 * s + d] for d in qd.static(range(3))], dt=gs.qd_float)
    gap = x + dx - target
    energy = 0.5 * spring * gap.norm_sqr() + 0.5 * damping * dx.norm_sqr()
    grad = spring * gap + damping * dx
    return energy, grad, spring + damping


@qd.func(requires_top_level=True)
def func_build_constraint_rows(
    particles_info: MpmParticlesInfo,
    grid: qd.template(),  # MpmGrid
    newton: qd.template(),  # MpmNewton
    params: qd.template(),  # MpmParams
    constraints: qd.template(),  # qd.StructField of the particle constraints (see MPMSolver.set_particle_constraints)
    links_state: array_class.LinksState,
):
    """Build the spring of every constrained particle as a term of the incremental potential on its trial displacement.

    The term is 1/2 dt^2 k |x + Delta - T|^2 + 1/2 dt c |Delta|^2 on the trial displacement Delta: the backward Euler
    step of the spring force -k (x' - T) and of the critically damped drag c = 2 sqrt(k m) on its velocity Delta / dt,
    unconditionally stable whatever k. T is the target at the start of the substep, placed by the pose of its link for
    an attached particle. The spring acts on the particle alone: the link feels no reaction."""
    dt = params.substep_dt[()]
    qd.loop_config(name="mpm_build_constraint_rows")
    for s in range(grid.n_valid[()]):
        i_p, i_b, env, cell = particle_of_sorted(grid, params, s)
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
            k = constraints[i_p, i_b].stiffness
            mass = particles_info.mass[i_p] / params.mass_scale[()]
            spring = dt * dt * k
            damping = dt * 2.0 * qd.sqrt(k * mass)
        newton.constraint_spring[s] = spring
        newton.constraint_damping[s] = damping
        for d in qd.static(range(3)):
            newton.constraint_target[3 * s + d] = target[d]


@qd.func
def particle_trial_disp(
    particles: MpmParticles,
    newton: qd.template(),  # MpmNewton
    params: qd.template(),  # MpmParams
    f,
    i_p,
    i_b,
    cell,
    slots,
):
    """Trial displacement sum_a w_a u_a of a particle, with its stencil weights."""
    fx = particles.pos[f, i_p, i_b] * params.inv_dx[()] - cell.cast(gs.qd_float)
    w, grad_w = q1_stencil(fx, params.inv_dx[()])
    dx = qd.Vector.zero(gs.qd_float, 3)
    for a in qd.static(range(8)):
        dx += w[a] * qd.Vector([newton.u[3 * slots[a] + d] for d in qd.static(range(3))], dt=gs.qd_float)
    return dx, w


@qd.func(requires_top_level=True)
def func_newton_reset(
    grid: qd.template(),  # MpmGrid
    newton: qd.template(),  # MpmNewton
):
    """Reset the Newton state to u = 0, the configuration at the start of the substep, which is free of penetration (see
    newton_start_inertial in mpm_data.py)."""
    qd.loop_config(name="mpm_newton_reset")
    for i in range(3 * grid.n_blocks[()] * BLOCK_CELLS):
        newton.u[i] = 0.0
        newton.u_temp[i] = 0.0
        newton.du[i] = 0.0


@qd.func(requires_top_level=True)
def func_newton_start(
    grid: qd.template(),  # MpmGrid
    newton: qd.template(),  # MpmNewton
    params: qd.template(),  # MpmParams
    dt,
):
    """Start the Newton solve at the inertial prediction u = dt v_free instead of u = 0 (func_newton_reset).

    `params.newton_start_inertial` enables it, and the fixed components stay at zero. Runs after the contact rows are
    built and before the start energy."""
    qd.loop_config(name="mpm_newton_start")
    for i in range(3 * grid.n_blocks[()] * BLOCK_CELLS):
        if params.newton_start_inertial[()] != 0 and grid.dof_fixed[i] == 0:
            newton.u[i] = dt * grid.node_v_free[i]
            newton.u_temp[i] = newton.u[i]


@qd.func(requires_top_level=True)
def func_report_extent(
    grid: qd.template(),  # MpmGrid
    linear: qd.template(),  # MpmLinearSystem
):
    """Report the live rows and triplet demand: one kinetic diagonal per node and STENCIL_BLOCKS per elastic task."""
    for _ in range(1):
        n_nodes = grid.n_blocks[()] * BLOCK_CELLS
        linear.pcg.n_rows[()] = linear.row_offset + n_nodes
        linear.matrix.report_triplet_demand(n_nodes + STENCIL_BLOCKS * grid.n_tasks[()])


@qd.func(requires_top_level=True)
def func_check_loop_capacity(
    grid: qd.template(),  # MpmGrid
    linear: qd.template(),  # MpmLinearSystem
    params: qd.template(),  # MpmParams
):
    """Check that the storage of the matrix holds the triplets of the assemblies of the Newton loop.

    These assemblies write the storage that the assembly before the loop grew for the grid of the substep, and nothing
    grows it inside the graph. An overflow there drops triplets, which fails the assertion in debug mode and otherwise
    raises INVALID_MPM_GRID for every environment. Runs after func_report_extent."""
    for _ in range(1):
        is_overflow = linear.matrix.triplet_overflow[()] != 0 or grid.grid_overflow[()] != 0
        if qd.static(qd_debug_asserts()):
            assert not is_overflow, "MPM Newton loop: the triplet demand %d exceeds the storage of %d triplets" % (
                linear.matrix.triplet_demand[()],
                linear.matrix.max_triplets[()],
            )
        if is_overflow:
            flag_all_envs_error(grid, params, array_class.ErrorCode.INVALID_MPM_GRID)


@qd.func(requires_top_level=True)
def func_kinetic_assemble(
    grid: qd.template(),  # MpmGrid
    linear: qd.template(),  # MpmLinearSystem
):
    """Write the diagonal block of every node, m I, or I on a fixed or massless node.

    A fixed node also gets a zero gradient and no stencil coupling, so its row is the identity and it stays put.
    """
    qd.loop_config(name="mpm_kinetic_assemble")
    for node in range(grid.n_blocks[()] * BLOCK_CELLS):
        mass = grid.node_mass[node]
        diagonal = qd.Matrix.identity(gs.qd_float, 3)
        if mass > 0.0 and grid.node_fixed[node] == 0:
            diagonal = mass * qd.Matrix.identity(gs.qd_float, 3)
        row = linear.row_offset + node
        linear.matrix.write_triplet(node, row, row, diagonal)


@qd.func(requires_top_level=True)
def func_elastic_assemble(
    particles: MpmParticles,
    particles_info: MpmParticlesInfo,
    materials: MpmMaterialTable,
    grid: qd.template(),  # MpmGrid
    newton: qd.template(),  # MpmNewton
    linear: qd.template(),  # MpmLinearSystem
    params: qd.template(),  # MpmParams
    f,
    dt,
    is_exact: qd.template() = False,
):
    """Assemble the elastic and contact Hessian blocks and the gradient factors of every task.

    The particles of a task share the 8-node stencil of their cell, and the task writes each of the 36 upper-triangular
    stencil blocks once, summed over its particles. Every particle also stages V dt^2 P(F_trial) F^T and the gradient of
    its active contact rows for the gradient gather. The rows act on the trial displacement sum_a w_a u_a, so their
    Hessian enters block (a, b) as w_a w_b H. Rows and columns of fixed components are dropped from the blocks. The
    elastic blocks are projected to symmetric positive definite (SPD), or exact with `is_exact`, which only an adjoint
    solve at the converged displacement takes.

    On a GPU, the warp of a task sums each block over the task by a segmented shuffle reduction. On a CPU, which has no
    lanes to share a cell over, the thread of a task adds the blocks of its particles in order (see func_cell_flags).
    """
    if qd.static(gs.backend == gs.cpu):
        func_elastic_assemble_cpu(particles, particles_info, materials, grid, newton, linear, params, f, dt, is_exact)
    else:
        func_elastic_assemble_gpu(particles, particles_info, materials, grid, newton, linear, params, f, dt, is_exact)


@qd.func
def particle_stencil_terms(
    particles: MpmParticles,
    particles_info: MpmParticlesInfo,
    materials: MpmMaterialTable,
    grid: qd.template(),  # MpmGrid
    newton: qd.template(),  # MpmNewton
    params: qd.template(),  # MpmParams
    f,
    dt,
    s,
    is_exact: qd.template(),
):
    """Terms of the stencil blocks of a sorted particle (see stencil_pair_block), after staging its gradient factor and
    contact-row gradients for the gradient gather."""
    i_p, i_b, env, cell = particle_of_sorted(grid, params, s)
    slots = stencil_node_slots(grid, env, cell)
    movable = qd.Matrix.zero(gs.qd_float, 8, 3)
    for a in qd.static(range(8)):
        for d in qd.static(range(3)):
            movable[a, d] = gs.qd_float(grid.dof_fixed[3 * slots[a] + d] == 0)
    F_tr, grad_w, F = trial_deformation(particles, grid, newton, params, f, i_p, i_b, cell, slots)
    U, S, V = qd.svd(F_tr, gs.qd_float)
    sigma = qd.Vector([S[0, 0], S[1, 1], S[2, 2]], dt=gs.qd_float)
    slot = particles_info.material_idx[i_p]
    mu, lam = material_params(materials, params, slot, particles.Jp[f, i_p, i_b])
    psi, g, A = elastic_principal(materials, slot, sigma, mu, lam)
    scale = params.particle_volume[()] * dt * dt
    P = U @ qd.Matrix([[g[0], 0.0, 0.0], [0.0, g[1], 0.0], [0.0, 0.0, g[2]]], dt=gs.qd_float) @ V.transpose()
    m = particles_info.muscle_direction[i_p]
    psi_fibre, P_fibre, fibre_curvature = fibre_terms(materials, params, slot, particles.actu[f, i_p, i_b], m, F_tr)
    P += P_fibre
    if qd.static(is_exact):
        fibre_curvature = material_param(materials, params, PARAM_FIBRE, slot) * particles.actu[f, i_p, i_b]
    # Fibre direction in the V frame
    fibre_frame = V.transpose() @ m
    G = scale * P @ F.transpose()
    for i, j in qd.static(qd.ndrange(3, 3)):
        newton.g_p[9 * s + 3 * i + j] = G[i, j]
    S_block, alp, bet = isotropic_projected_modes(sigma, g, A, is_exact)
    # Stencil vectors r_a = V^T F^T grad w_a in the V frame (see isotropic_pair_block)
    r_frame = qd.Matrix.zero(gs.qd_float, 8, 3)
    VtFt = V.transpose() @ F.transpose()
    for a in qd.static(range(8)):
        r_a = VtFt @ qd.Vector([grad_w[a, d] for d in qd.static(range(3))], dt=gs.qd_float)
        for d in qd.static(range(3)):
            r_frame[a, d] = r_a[d]
    dx, w = particle_trial_disp(particles, newton, params, f, i_p, i_b, cell, slots)
    dr = qd.Vector([newton.contact_dr[3 * s + d] for d in qd.static(range(3))], dt=gs.qd_float)
    energy_wall, grad_wall, hess_wall = contact_row_terms(newton, s, dx, 0, WALL_ROWS)
    energy_rigid, grad_rigid, hess_rigid = contact_row_terms(newton, s, dx - dr, WALL_ROWS, CONTACT_ROWS)
    energy_spring, grad_spring, hess_spring = constraint_row_terms(newton, s, particles.pos[f, i_p, i_b], dx)
    for d in qd.static(range(3)):
        newton.g_contact[3 * s + d] = grad_wall[d] + grad_rigid[d] + grad_spring[d]
        newton.g_rigid[3 * s + d] = grad_rigid[d]
    for i, j in qd.static(qd.ndrange(3, 3)):
        newton.h_rigid[9 * s + 3 * i + j] = hess_rigid[i, j]
    hess_contact = hess_wall + hess_rigid + hess_spring * qd.Matrix.identity(gs.qd_float, 3)
    return slots, movable, r_frame, U, S_block, alp, bet, fibre_curvature, fibre_frame, scale, w, hess_contact


@qd.func
def stencil_pair_block(
    a,
    b,
    movable,
    r_frame,
    U,
    S_block,
    alp,
    bet,
    fibre_curvature,
    fibre_frame,
    scale,
    w,
    hess_contact,
):
    """Block (a, b) of the stencil of one particle from its terms (see particle_stencil_terms)."""
    r_a = qd.Vector([r_frame[a, d] for d in qd.static(range(3))], dt=gs.qd_float)
    r_b = qd.Vector([r_frame[b, d] for d in qd.static(range(3))], dt=gs.qd_float)
    block = (scale * U) @ isotropic_pair_block(r_a, r_b, S_block, alp, bet) @ U.transpose()
    # The fibre term in u: F_trial m moves by (grad w_a . F m) u_a, with grad w_a . F m = r_a . V^T m
    block += (scale * fibre_curvature * r_a.dot(fibre_frame) * r_b.dot(fibre_frame)) * qd.Matrix.identity(
        gs.qd_float, 3
    )
    block += (w[a] * w[b]) * hess_contact
    for i, j in qd.static(qd.ndrange(3, 3)):
        block[i, j] *= movable[a, i] * movable[b, j]
    return block


@qd.func(requires_top_level=True)
def func_elastic_assemble_cpu(
    particles: MpmParticles,
    particles_info: MpmParticlesInfo,
    materials: MpmMaterialTable,
    grid: qd.template(),  # MpmGrid
    newton: qd.template(),  # MpmNewton
    linear: qd.template(),  # MpmLinearSystem
    params: qd.template(),  # MpmParams
    f,
    dt,
    is_exact: qd.template(),
):
    """func_elastic_assemble by one thread per task, a whole cell (see func_cell_flags).

    The threads split the tasks. The first particle of a task writes its 36 blocks, and each later one adds its own to
    them in order."""
    pair_a = qd.Vector(_STENCIL_PAIR_A, dt=qd.i32)
    pair_b = qd.Vector(_STENCIL_PAIR_B, dt=qd.i32)
    qd.loop_config(name="mpm_elastic_assemble_cpu")
    for s in range(grid.n_valid[()]):
        if grid.task_flags[s] != 0:
            i_task = grid.task_ranks[s]
            base = grid.n_blocks[()] * BLOCK_CELLS + STENCIL_BLOCKS * i_task
            # The task ranks equal the cell ranks, so the particles of the task end at the start of the next cell
            for t in range(s, grid.cell_starts[i_task + 1]):
                slots, movable, r_frame, U, S_block, alp, bet, fibre_curvature, fibre_frame, scale, w, hess_contact = (
                    particle_stencil_terms(
                        particles, particles_info, materials, grid, newton, params, f, dt, t, is_exact
                    )
                )
                for pair in range(STENCIL_BLOCKS):
                    a = pair_a[pair]
                    b = pair_b[pair]
                    block = stencil_pair_block(
                        a,
                        b,
                        movable,
                        r_frame,
                        U,
                        S_block,
                        alp,
                        bet,
                        fibre_curvature,
                        fibre_frame,
                        scale,
                        w,
                        hess_contact,
                    )
                    row = linear.row_offset + slots[a]
                    col = linear.row_offset + slots[b]
                    if t == s:
                        linear.matrix.write_triplet(base + pair, row, col, block)
                    else:
                        linear.matrix.add_triplet(base + pair, row, col, block)


@qd.func(requires_top_level=True)
def func_elastic_assemble_gpu(
    particles: MpmParticles,
    particles_info: MpmParticlesInfo,
    materials: MpmMaterialTable,
    grid: qd.template(),  # MpmGrid
    newton: qd.template(),  # MpmNewton
    linear: qd.template(),  # MpmLinearSystem
    params: qd.template(),  # MpmParams
    f,
    dt,
    is_exact: qd.template(),
):
    """func_elastic_assemble by one warp per 32 sorted particles: a head-segmented warp reduction sums each block over
    the task, and the head lane of the task writes it."""
    qd.loop_config(name="mpm_elastic_assemble", block_dim=128)
    for s in range(grid.n_valid_chunks[()] * _WARP):
        lane = qd.i32(qd.simt.subgroup.invocation_id()) & (_WARP - 1)
        n_valid = grid.n_valid[()]
        is_valid = s < n_valid

        slots = qd.Vector.zero(qd.i32, 8)
        movable = qd.Matrix.zero(gs.qd_float, 8, 3)
        r_frame = qd.Matrix.zero(gs.qd_float, 8, 3)
        U = qd.Matrix.identity(gs.qd_float, 3)
        S_block = qd.Matrix.zero(gs.qd_float, 3, 3)
        alp = qd.Vector.zero(gs.qd_float, 3)
        bet = qd.Vector.zero(gs.qd_float, 3)
        fibre_curvature = gs.qd_float(0.0)
        fibre_frame = qd.Vector.zero(gs.qd_float, 3)
        scale = gs.qd_float(0.0)
        w = qd.Vector.zero(gs.qd_float, 8)
        hess_contact = qd.Matrix.zero(gs.qd_float, 3, 3)
        is_head = 0
        is_tail = 0
        task = 0
        if is_valid:
            terms = particle_stencil_terms(
                particles, particles_info, materials, grid, newton, params, f, dt, s, is_exact
            )
            slots = terms[0]
            movable = terms[1]
            r_frame = terms[2]
            U = terms[3]
            S_block = terms[4]
            alp = terms[5]
            bet = terms[6]
            fibre_curvature = terms[7]
            fibre_frame = terms[8]
            scale = terms[9]
            w = terms[10]
            hess_contact = terms[11]
            is_head = grid.task_flags[s]
            task = grid.task_ranks[s] + grid.task_flags[s] - 1
            if lane == _WARP - 1 or s == n_valid - 1:
                is_tail = 1
            elif grid.task_flags[s + 1] != 0:
                is_tail = 1

        n_nodes = grid.n_blocks[()] * BLOCK_CELLS
        pair_a = qd.Vector(_STENCIL_PAIR_A, dt=qd.i32)
        pair_b = qd.Vector(_STENCIL_PAIR_B, dt=qd.i32)
        for pair in range(STENCIL_BLOCKS):
            a = pair_a[pair]
            b = pair_b[pair]
            block = qd.Matrix.zero(gs.qd_float, 3, 3)
            if is_valid:
                block = stencil_pair_block(
                    a, b, movable, r_frame, U, S_block, alp, bet, fibre_curvature, fibre_frame, scale, w, hess_contact
                )
            task_block = qd.Matrix.zero(gs.qd_float, 3, 3)
            for i, j in qd.static(qd.ndrange(3, 3)):
                task_block[i, j] = head_segmented_reduce_add(block[i, j], is_tail)
            if is_valid and is_head != 0:
                linear.matrix.write_triplet(
                    n_nodes + STENCIL_BLOCKS * task + pair,
                    linear.row_offset + slots[a],
                    linear.row_offset + slots[b],
                    task_block,
                )


@qd.func(requires_top_level=True)
def func_gradient_gather(
    particles: MpmParticles,
    grid: qd.template(),  # MpmGrid
    newton: qd.template(),  # MpmNewton
    linear: qd.template(),  # MpmLinearSystem
    params: qd.template(),  # MpmParams
    f,
    dt,
):
    """Gather the gradient b_rhs = m (u - dt v_free) + sum_p (G_p grad w + w g_contact_p) of every node.

    The gradient is zero on fixed components. The particles of a node are summed in a fixed order, by the lanes of a
    warp on a GPU and by one thread on a CPU (see func_p2g)."""
    if qd.static(gs.backend == gs.cpu):
        func_gradient_gather_cpu(particles, grid, newton, linear, params, f, dt)
    else:
        func_gradient_gather_gpu(particles, grid, newton, linear, params, f, dt)


@qd.func
def add_particle_node_gradient(
    grad,
    particles: MpmParticles,
    grid: qd.template(),  # MpmGrid
    newton: qd.template(),  # MpmNewton
    params: qd.template(),  # MpmParams
    f,
    s,
    c,
    o,
):
    """Add the gradient contribution G_p grad w + w g_contact_p of a sorted particle at one node of its stencil.

    `s` is the sorted particle, of cell `c`, and the node is c + o. Returns `grad` plus the contribution.
    """
    gid = grid.sort_ids[s]
    i_b = gid // params.n_particles[()]
    i_p = gid - i_b * params.n_particles[()]
    fx = particles.pos[f, i_p, i_b] * params.inv_dx[()] - c.cast(gs.qd_float)
    w, grad_w = q1_node_weight(fx, o, params.inv_dx[()])
    for i in qd.static(range(3)):
        grad[i] += w * newton.g_contact[3 * s + i]
        for j in qd.static(range(3)):
            grad[i] += newton.g_p[9 * s + 3 * i + j] * grad_w[j]
    return grad


@qd.func
def publish_node_gradient(
    grid: qd.template(),  # MpmGrid
    newton: qd.template(),  # MpmNewton
    linear: qd.template(),  # MpmLinearSystem
    node,
    grad,
    dt,
):
    mass = grid.node_mass[node]
    for d in qd.static(range(3)):
        value = gs.qd_float(0.0)
        if grid.dof_fixed[3 * node + d] == 0:
            value = grad[d] + mass * (newton.u[3 * node + d] - dt * grid.node_v_free[3 * node + d])
        linear.b_rhs[3 * (linear.row_offset + node) + d] = value


@qd.func(requires_top_level=True)
def func_gradient_gather_cpu(
    particles: MpmParticles,
    grid: qd.template(),  # MpmGrid
    newton: qd.template(),  # MpmNewton
    linear: qd.template(),  # MpmLinearSystem
    params: qd.template(),  # MpmParams
    f,
    dt,
):
    """func_gradient_gather by one thread per node, the threads splitting the nodes, each walking the particles of the 8
    cells around its node in order."""
    qd.loop_config(name="mpm_gradient_gather_cpu")
    for node in range(grid.n_blocks[()] * BLOCK_CELLS):
        env, block = unpack_block(grid.block_coords[node // BLOCK_CELLS])
        n = block * 4 + local_offset(node % BLOCK_CELLS)
        grad = qd.Vector.zero(gs.qd_float, 3)
        for a in qd.static(range(8)):
            o = node_offset(a)
            c = n - o
            block_slot = grid.hash.lookup(pack_block(env, block_of(c)))
            if block_slot >= 0:
                cell_slot = block_slot * BLOCK_CELLS + cell_local(c)
                for s in range(grid.cell_seg_start[cell_slot], grid.cell_seg_end[cell_slot]):
                    grad = add_particle_node_gradient(grad, particles, grid, newton, params, f, s, c, o)
        publish_node_gradient(grid, newton, linear, node, grad, dt)


@qd.func(requires_top_level=True)
def func_gradient_gather_gpu(
    particles: MpmParticles,
    grid: qd.template(),  # MpmGrid
    newton: qd.template(),  # MpmNewton
    linear: qd.template(),  # MpmLinearSystem
    params: qd.template(),  # MpmParams
    f,
    dt,
):
    """func_gradient_gather by one warp per node (see func_p2g_gpu)."""
    qd.loop_config(name="mpm_gradient_gather", block_dim=128)
    for t in range(grid.n_blocks[()] * BLOCK_CELLS * _WARP):
        lane = qd.i32(qd.simt.subgroup.invocation_id()) & (_WARP - 1)
        node = t // _WARP
        env, block = unpack_block(grid.block_coords[node // BLOCK_CELLS])
        n = block * 4 + local_offset(node % BLOCK_CELLS)

        grad = qd.Vector.zero(gs.qd_float, 3)
        for a in qd.static(range(8)):
            o = node_offset(a)
            c = n - o
            lo = 0
            hi = 0
            if lane == 0:
                block_slot = grid.hash.lookup(pack_block(env, block_of(c)))
                if block_slot >= 0:
                    cell_slot = block_slot * BLOCK_CELLS + cell_local(c)
                    lo = grid.cell_seg_start[cell_slot]
                    hi = grid.cell_seg_end[cell_slot]
            lo = qd.simt.subgroup.shuffle(lo, qd.u32(0))
            hi = qd.simt.subgroup.shuffle(hi, qd.u32(0))
            for i_chunk_ in range((hi - lo + _WARP - 1) // _WARP):
                s = lo + i_chunk_ * _WARP + lane
                if s < hi:
                    grad = add_particle_node_gradient(grad, particles, grid, newton, params, f, s, c, o)
        for d in qd.static(range(3)):
            grad[d] = warp_sum(grad[d])
        if lane == 0:
            publish_node_gradient(grid, newton, linear, node, grad, dt)


@qd.func(requires_top_level=True)
def func_assemble(
    particles: MpmParticles,
    particles_info: MpmParticlesInfo,
    materials: MpmMaterialTable,
    grid: qd.template(),  # MpmGrid
    newton: qd.template(),  # MpmNewton
    linear: qd.template(),  # MpmLinearSystem
    params: qd.template(),  # MpmParams
    f,
    dt,
    is_exact: qd.template() = False,
):
    func_kinetic_assemble(grid, linear)
    func_elastic_assemble(particles, particles_info, materials, grid, newton, linear, params, f, dt, is_exact)
    func_gradient_gather(particles, grid, newton, linear, params, f, dt)


@qd.func(requires_top_level=True)
def func_reset_brick(
    grid: qd.template(),  # MpmGrid
    linear: qd.template(),  # MpmLinearSystem
):
    qd.loop_config(name="mpm_brick_reset")
    for k in range(grid.n_blocks[()] * BRICKS_PER_BLOCK * BRICK_SCALARS):
        linear.brick[k] = 0.0


@qd.func(requires_top_level=True)
def func_gather_brick(
    linear: qd.template(),  # MpmLinearSystem
):
    """Store every reduced upper-triangular block whose row and column nodes share a brick into its packed storage.

    Blocks coupling two bricks are dropped (block-diagonal Schwarz restriction)."""
    qd.loop_config(name="mpm_brick_visit")
    for entry in range(linear.matrix.bcoo_nnz[()]):
        row, col, block = linear.matrix.read_bcoo(entry)
        local_row = row - linear.row_offset
        local_col = col - linear.row_offset
        if local_row >= 0:
            brick = brick_of(local_row)
            if brick_of(local_col) == brick:
                base = (brick * BRICK_SYM_BLOCKS + brick_sym_index(brick_node(local_row), brick_node(local_col))) * 9
                for i, j in qd.static(qd.ndrange(3, 3)):
                    linear.brick[base + 3 * i + j] = block[i, j]


@qd.func(requires_top_level=True)
def func_invert_brick(
    grid: qd.template(),  # MpmGrid
    linear: qd.template(),  # MpmLinearSystem
):
    """Invert every brick's 48 x 48 matrix in place and record which of its nodes are coupled inside the brick.

    The sweep of a pivot reads the pivot column that every entry depends on. A GPU spreads the entries of a brick over
    the registers of a warp, shares the pivot column and row through per-warp shared memory and steps the lanes together
    (see func_invert_brick_gpu). The CPU backend has neither, and a CPU thread sweeps a whole brick on its packed
    storage, the column kept in scratch."""
    if qd.static(gs.backend == gs.cpu):
        func_invert_brick_cpu(grid, linear)
    else:
        func_invert_brick_gpu(grid, linear)


@qd.func(requires_top_level=True)
def func_invert_brick_cpu(
    grid: qd.template(),  # MpmGrid
    linear: qd.template(),  # MpmLinearSystem
):
    """func_invert_brick_gpu by one thread per brick, in place on the packed storage of its upper triangle.

    Only the entries between coupled nodes are equilibrated and swept: a decoupled node has no entry with another one,
    so the sweep of a coupled pivot leaves the coupled entries as the full sweep does. The equilibration and the
    snapshot of the pivot column live in `linear.brick_scratch`."""
    qd.loop_config(name="mpm_brick_invert_cpu")
    for brick in range(grid.n_blocks[()] * BRICKS_PER_BLOCK):
        base = brick * BRICK_SCALARS
        scale_start = brick * 2 * BRICK_DIM
        column_start = scale_start + BRICK_DIM
        bits = 0
        for a in range(BRICK_NODES):
            for b in range(a + 1, BRICK_NODES):
                block_base = base + brick_sym_index(a, b) * 9
                for k in qd.static(range(9)):
                    if linear.brick[block_base + k] != 0.0:
                        bits |= (1 << a) | (1 << b)
        linear.brick_mask[brick] = bits

        if bits != 0:
            for r in range(BRICK_DIM):
                if (bits & (1 << (r // 3))) != 0:
                    diagonal = linear.brick[brick_entry_index(base, r, r)]
                    linear.brick_scratch[scale_start + r] = 1.0 / qd.sqrt(1.0 if diagonal == 0.0 else diagonal)
            for r in range(BRICK_DIM):
                if (bits & (1 << (r // 3))) != 0:
                    for c in range(r, BRICK_DIM):
                        if (bits & (1 << (c // 3))) != 0:
                            index = brick_entry_index(base, r, c)
                            x = linear.brick[index]
                            if r == c and x == 0.0:
                                x = 1.0
                            x *= linear.brick_scratch[scale_start + r] * linear.brick_scratch[scale_start + c]
                            if r == c:
                                x += _BRICK_REGULARIZATION
                            linear.brick[index] = x

            is_broken = 0
            for k in range(BRICK_DIM):
                if (bits & (1 << (k // 3))) != 0:
                    for r in range(BRICK_DIM):
                        if (bits & (1 << (r // 3))) != 0:
                            linear.brick_scratch[column_start + r] = linear.brick[brick_entry_index(base, r, k)]
                    d = linear.brick_scratch[column_start + k]
                    inv_d = 1.0 / d
                    if not (d > 0.0 and d < _BRICK_FINITE_LIMIT):
                        is_broken = 1
                    for r in range(BRICK_DIM):
                        if (bits & (1 << (r // 3))) != 0:
                            a_rk = linear.brick_scratch[column_start + r]
                            for c in range(r, BRICK_DIM):
                                if (bits & (1 << (c // 3))) != 0:
                                    index = brick_entry_index(base, r, c)
                                    qc = linear.brick_scratch[column_start + c] * inv_d
                                    value = linear.brick[index] - a_rk * qc
                                    if r == k:
                                        value = -inv_d if c == k else qc
                                    elif c == k:
                                        value = a_rk * inv_d
                                    linear.brick[index] = value

            for r in range(BRICK_DIM):
                if (bits & (1 << (r // 3))) != 0:
                    for c in range(r, BRICK_DIM):
                        if (bits & (1 << (c // 3))) != 0:
                            index = brick_entry_index(base, r, c)
                            value = -linear.brick[index]
                            if is_broken != 0:
                                value = 1.0 if r == c else 0.0
                            value *= linear.brick_scratch[scale_start + r] * linear.brick_scratch[scale_start + c]
                            linear.brick[index] = value
                            # A diagonal node block is stored whole: mirror its lower entry
                            if r // 3 == c // 3 and r != c:
                                block_base = base + brick_sym_index(r // 3, r // 3) * 9
                                linear.brick[block_base + (c % 3) * 3 + r % 3] = value

        # Decoupled nodes: their inverse is the inverse of their own block
        for a in range(BRICK_NODES):
            if (bits & (1 << a)) == 0:
                block_base = base + brick_sym_index(a, a) * 9
                D = qd.Matrix.zero(gs.qd_float, 3, 3)
                entry_max = gs.qd_float(0.0)
                for i, j in qd.static(qd.ndrange(3, 3)):
                    D[i, j] = linear.brick[block_base + 3 * i + j]
                    entry_max = qd.max(entry_max, qd.abs(D[i, j]))
                if entry_max == 0.0:
                    D = qd.Matrix.identity(gs.qd_float, 3)
                D_inv = D.inverse()
                for i, j in qd.static(qd.ndrange(3, 3)):
                    linear.brick[block_base + 3 * i + j] = D_inv[i, j]


@qd.func(requires_top_level=True)
def func_invert_brick_gpu(
    grid: qd.template(),  # MpmGrid
    linear: qd.template(),  # MpmLinearSystem
):
    """Invert every brick's 48 x 48 matrix in place by the symmetric sweep operator (Goodnight 1979), one warp per
    brick, and record which of its nodes are coupled inside the brick.

    Sweeping pivot k maps a_kk -> -1/a_kk, a_ik -> a_ik / a_kk, a_ij -> a_ij - a_ik a_kj / a_kk, and sweeping every
    pivot of a symmetric positive definite (SPD) matrix yields -A^-1. The 1176 upper entries live in registers as 6 x 6
    tiles of the 8 x 8 tile grid: lanes 0-27 own the 28 off-diagonal tiles, lanes 28-31 two diagonal tiles each. The
    pivot column and row go through double-buffered per-warp shared memory. The matrix is equilibrated to a unit
    diagonal and shifted by _BRICK_REGULARIZATION. Only the coupled nodes are swept, and the inverse of a decoupled node
    is the inverse of its own 3 x 3 block. A non-positive or non-finite pivot falls back to the Jacobi inverse.
    """
    qd.loop_config(name="mpm_brick_invert", block_dim=_WARP * BRICK_INVERT_WARPS)
    for t in range(grid.n_blocks[()] * BRICKS_PER_BLOCK * _WARP):
        lane = qd.i32(qd.simt.subgroup.invocation_id()) & (_WARP - 1)
        i_warp = qd.simt.block.thread_idx() // _WARP
        brick = t // _WARP
        sh_col = qd.simt.block.SharedArray((BRICK_INVERT_WARPS * 2 * BRICK_DIM,), gs.qd_float)
        sh_row = qd.simt.block.SharedArray((BRICK_INVERT_WARPS * 2 * BRICK_DIM,), gs.qd_float)
        sh_scale = qd.simt.block.SharedArray((BRICK_INVERT_WARPS * BRICK_DIM,), gs.qd_float)
        sh_pivots = qd.simt.block.SharedArray((BRICK_INVERT_WARPS * BRICK_DIM,), qd.i32)
        base = brick * BRICK_SCALARS

        is_diag_lane = lane >= _BRICK_OFF_TILES
        P0 = 0
        Q0 = 0
        if is_diag_lane:
            P0 = 2 * _BRICK_TILE * (lane - _BRICK_OFF_TILES)
            Q0 = P0 + _BRICK_TILE
        else:
            tile = lane
            tile_row = 0
            for _ in qd.static(range(_BRICK_TILES - 1)):
                if tile >= _BRICK_TILES - 1 - tile_row:
                    tile -= _BRICK_TILES - 1 - tile_row
                    tile_row += 1
            P0 = _BRICK_TILE * tile_row
            Q0 = _BRICK_TILE * (tile_row + 1 + tile)

        # Load the entries of the lane, which also scans the in-brick off-diagonal coupling
        x = qd.Vector.zero(gs.qd_float, _BRICK_SLOTS)
        bits = 0
        for s in qd.static(range(_BRICK_SLOTS)):
            is_valid, r, c = brick_slot_entry(s, is_diag_lane, P0, Q0)
            if is_valid:
                x[s] = linear.brick[base + brick_sym_index(r // 3, c // 3) * 9 + (r % 3) * 3 + c % 3]
                if r // 3 != c // 3 and x[s] != 0.0:
                    bits |= (1 << (r // 3)) | (1 << (c // 3))
                if r == c:
                    sh_scale[i_warp * BRICK_DIM + r] = x[s]
        for level in qd.static(range(5)):
            bits |= qd.simt.subgroup.shuffle_xor(bits, qd.u32(1 << level))
        if lane == 0:
            linear.brick_mask[brick] = bits

        if bits != 0:
            if lane < BRICK_NODES and (bits & (1 << lane)) != 0:
                rank = qd.i32(qd.math.popcnt(qd.u32(bits & ((1 << lane) - 1))))
                for q in qd.static(range(3)):
                    sh_pivots[i_warp * BRICK_DIM + 3 * rank + q] = 3 * lane + q
            n_pivots = 3 * qd.i32(qd.math.popcnt(qd.u32(bits)))

            # Turn the raw diagonal held in sh_scale into the equilibration
            qd.simt.subgroup.sync()
            qd.simt.subgroup.mem_fence()
            for half in qd.static(range(2)):
                r = lane + half * _WARP
                if r < BRICK_DIM:
                    diagonal = sh_scale[i_warp * BRICK_DIM + r]
                    sh_scale[i_warp * BRICK_DIM + r] = 1.0 / qd.sqrt(1.0 if diagonal == 0.0 else diagonal)
            qd.simt.subgroup.sync()
            qd.simt.subgroup.mem_fence()

            for s in qd.static(range(_BRICK_SLOTS)):
                is_valid, r, c = brick_slot_entry(s, is_diag_lane, P0, Q0)
                if is_valid:
                    if r == c and x[s] == 0.0:
                        x[s] = 1.0
                    x[s] *= sh_scale[i_warp * BRICK_DIM + r] * sh_scale[i_warp * BRICK_DIM + c]
                    if r == c:
                        x[s] += _BRICK_REGULARIZATION

            is_broken = 0
            for p in range(n_pivots):
                i_buffer = (i_warp * 2 + (p & 1)) * BRICK_DIM
                k = sh_pivots[i_warp * BRICK_DIM + p]
                for s in qd.static(range(_BRICK_SLOTS)):
                    is_valid, r, c = brick_slot_entry(s, is_diag_lane, P0, Q0)
                    if is_valid and (r == k or c == k):
                        sh_col[i_buffer + qd.select(r == k, c, r)] = x[s]
                qd.simt.subgroup.sync()
                qd.simt.subgroup.mem_fence()

                d = sh_col[i_buffer + k]
                inv_d = 1.0 / d
                if not (d > 0.0 and d < _BRICK_FINITE_LIMIT):
                    is_broken = 1
                sh_row[i_buffer + lane] = sh_col[i_buffer + lane] * inv_d
                if lane < BRICK_DIM - _WARP:
                    sh_row[i_buffer + lane + _WARP] = sh_col[i_buffer + lane + _WARP] * inv_d
                qd.simt.subgroup.sync()
                qd.simt.subgroup.mem_fence()

                col_p = qd.Vector([sh_col[i_buffer + P0 + i] for i in qd.static(range(_BRICK_TILE))], dt=gs.qd_float)
                row_p = qd.Vector([sh_row[i_buffer + P0 + i] for i in qd.static(range(_BRICK_TILE))], dt=gs.qd_float)
                col_q = qd.Vector([sh_col[i_buffer + Q0 + i] for i in qd.static(range(_BRICK_TILE))], dt=gs.qd_float)
                row_q = qd.Vector([sh_row[i_buffer + Q0 + i] for i in qd.static(range(_BRICK_TILE))], dt=gs.qd_float)
                for s in qd.static(range(_BRICK_SLOTS)):
                    is_valid, r, c = brick_slot_entry(s, is_diag_lane, P0, Q0)
                    if is_valid:
                        # a = a_rk, qc = a_kc / a_kk, qr = a_rk / a_kk
                        a = gs.qd_float(0.0)
                        qc = gs.qd_float(0.0)
                        qr = gs.qd_float(0.0)
                        if qd.static(s < _BRICK_TRI):
                            if is_diag_lane:
                                a = col_p[_BRICK_TRI_I[s]]
                                qc = row_p[_BRICK_TRI_J[s]]
                                qr = row_p[_BRICK_TRI_I[s]]
                            else:
                                a = col_p[s // _BRICK_TILE]
                                qc = row_q[s % _BRICK_TILE]
                                qr = row_p[s // _BRICK_TILE]
                        elif qd.static(s < _BRICK_TILE * _BRICK_TILE):
                            if is_diag_lane:
                                a = col_q[_BRICK_TRI_I[s - _BRICK_TRI]]
                                qc = row_q[_BRICK_TRI_J[s - _BRICK_TRI]]
                                qr = row_q[_BRICK_TRI_I[s - _BRICK_TRI]]
                            else:
                                a = col_p[s // _BRICK_TILE]
                                qc = row_q[s % _BRICK_TILE]
                                qr = row_p[s // _BRICK_TILE]
                        else:
                            a = col_q[_BRICK_TRI_I[s - _BRICK_TRI]]
                            qc = row_q[_BRICK_TRI_J[s - _BRICK_TRI]]
                            qr = row_q[_BRICK_TRI_I[s - _BRICK_TRI]]
                        g = x[s] - a * qc
                        row_k = qd.select(c == k, -inv_d, qc)
                        x[s] = qd.select(r == k, row_k, qd.select(c == k, qr, g))

            # In place: every load of this brick finished before the sweep
            for s in qd.static(range(_BRICK_SLOTS)):
                is_valid, r, c = brick_slot_entry(s, is_diag_lane, P0, Q0)
                if is_valid and (bits & (1 << (r // 3))) != 0 and (bits & (1 << (c // 3))) != 0:
                    value = -x[s]
                    if is_broken != 0:
                        value = qd.select(r == c, gs.qd_float(1.0), gs.qd_float(0.0))
                    value *= sh_scale[i_warp * BRICK_DIM + r] * sh_scale[i_warp * BRICK_DIM + c]
                    block_base = base + brick_sym_index(r // 3, c // 3) * 9
                    linear.brick[block_base + (r % 3) * 3 + c % 3] = value
                    if r // 3 == c // 3 and r != c:
                        linear.brick[block_base + (c % 3) * 3 + r % 3] = value

        # Decoupled nodes: no in-brick block joins them to another node, so their inverse is that of their own block
        if lane < BRICK_NODES and (bits & (1 << lane)) == 0:
            block_base = base + brick_sym_index(lane, lane) * 9
            D = qd.Matrix.zero(gs.qd_float, 3, 3)
            entry_max = gs.qd_float(0.0)
            for i, j in qd.static(qd.ndrange(3, 3)):
                D[i, j] = linear.brick[block_base + 3 * i + j]
                entry_max = qd.max(entry_max, qd.abs(D[i, j]))
            if entry_max == 0.0:
                D = qd.Matrix.identity(gs.qd_float, 3)
            D_inv = D.inverse()
            for i, j in qd.static(qd.ndrange(3, 3)):
                linear.brick[block_base + 3 * i + j] = D_inv[i, j]
        qd.simt.subgroup.sync()


@qd.func(requires_top_level=True)
def func_apply_brick(
    grid: qd.template(),  # MpmGrid
    linear: qd.template(),  # MpmLinearSystem
):
    """Apply the brick preconditioner z = M^-1 r over the material point method (MPM) rows, one thread per row.

    M is the block diagonal of the bricks. A GPU stages the r of a brick in block-shared memory once for its 48 rows
    (see func_apply_brick_gpu). The CPU backend has no shared memory, and its rows read r directly."""
    if qd.static(gs.backend == gs.cpu):
        func_apply_brick_cpu(grid, linear)
    else:
        func_apply_brick_gpu(grid, linear)


@qd.func(requires_top_level=True)
def func_apply_brick_cpu(
    grid: qd.template(),  # MpmGrid
    linear: qd.template(),  # MpmLinearSystem
):
    """func_apply_brick_gpu reading r from the rows directly, without the shared copy of the brick's r."""
    qd.loop_config(name="mpm_brick_apply_cpu")
    for t in range(grid.n_blocks[()] * BRICKS_PER_BLOCK * BRICK_DIM):
        brick = t // BRICK_DIM
        i = t - brick * BRICK_DIM
        a = i // 3
        p = i % 3
        base = brick * BRICK_SCALARS
        bits = linear.brick_mask[brick]
        z_entry = gs.qd_float(0.0)
        if (bits & (1 << a)) != 0:
            for b in range(BRICK_NODES):
                if (bits & (1 << b)) != 0:
                    block_base = base + brick_sym_index(qd.min(a, b), qd.max(a, b)) * 9
                    row_b = 3 * (linear.row_offset + brick_slot(brick, b))
                    for q in qd.static(range(3)):
                        m = gs.qd_float(0.0)
                        if a <= b:
                            m = linear.brick[block_base + p * 3 + q]
                        else:
                            m = linear.brick[block_base + q * 3 + p]
                        z_entry += m * linear.pcg.r[row_b + q]
        else:
            block_base = base + brick_sym_index(a, a) * 9
            row_a = 3 * (linear.row_offset + brick_slot(brick, a))
            for q in qd.static(range(3)):
                z_entry += linear.brick[block_base + p * 3 + q] * linear.pcg.r[row_a + q]
        linear.pcg.z[3 * (linear.row_offset + brick_slot(brick, a)) + p] = z_entry


@qd.func(requires_top_level=True)
def func_apply_brick_gpu(
    grid: qd.template(),  # MpmGrid
    linear: qd.template(),  # MpmLinearSystem
):
    """func_apply_brick by 48 threads per brick, one per row.

    A block of 96 threads serves two bricks, since a block size must be a multiple of the warp size. A coupled row uses
    the dense inverse over the coupled nodes, and a decoupled row its own 3 x 3 inverse.
    """
    qd.loop_config(name="mpm_brick_apply", block_dim=2 * BRICK_DIM)
    for t in range(grid.n_blocks[()] * BRICKS_PER_BLOCK * BRICK_DIM):
        thread = qd.simt.block.thread_idx()
        local = thread // BRICK_DIM
        i = thread - local * BRICK_DIM
        brick = t // BRICK_DIM
        a = i // 3
        p = i % 3
        sh_r = qd.simt.block.SharedArray((2 * BRICK_DIM,), gs.qd_float)
        s = brick_slot(brick, a)
        sh_r[thread] = linear.pcg.r[3 * (linear.row_offset + s) + p]
        qd.simt.block.sync()

        base = brick * BRICK_SCALARS
        bits = linear.brick_mask[brick]
        z_entry = gs.qd_float(0.0)
        if (bits & (1 << a)) != 0:
            for b in range(BRICK_NODES):
                if (bits & (1 << b)) != 0:
                    block_base = base + brick_sym_index(qd.min(a, b), qd.max(a, b)) * 9
                    for q in qd.static(range(3)):
                        m = gs.qd_float(0.0)
                        if a <= b:
                            m = linear.brick[block_base + p * 3 + q]
                        else:
                            m = linear.brick[block_base + q * 3 + p]
                        z_entry += m * sh_r[local * BRICK_DIM + b * 3 + q]
        else:
            block_base = base + brick_sym_index(a, a) * 9
            for q in qd.static(range(3)):
                z_entry += linear.brick[block_base + p * 3 + q] * sh_r[local * BRICK_DIM + a * 3 + q]
        linear.pcg.z[3 * (linear.row_offset + s) + p] = z_entry


@qd.func(requires_top_level=True)
def func_solve_brick_pcg(
    grid: qd.template(),  # MpmGrid
    linear: qd.template(),  # MpmLinearSystem
):
    """Run the preconditioned conjugate gradient (PCG) iteration loop with the brick preconditioner, after
    func_init_linear_pcg_direction armed it."""
    while qd.graph.do_while(linear.pcg.cond):
        func_linear_pcg_matvec_update(linear.pcg, linear.matrix, linear.solution)
        func_apply_brick(grid, linear)
        func_linear_pcg_direction_update(linear.pcg)


@qd.func(requires_top_level=True)
def func_negate_du(
    grid: qd.template(),  # MpmGrid
    newton: qd.template(),  # MpmNewton
    linear: qd.template(),  # MpmLinearSystem
):
    qd.loop_config(name="mpm_negate_du")
    for i in range(3 * grid.n_blocks[()] * BLOCK_CELLS):
        newton.du[i] = -linear.solution[3 * linear.row_offset + i]


@qd.func(requires_top_level=True)
def func_record_start_point(
    grid: qd.template(),  # MpmGrid
    newton: qd.template(),  # MpmNewton
):
    qd.loop_config(name="mpm_record_start_point")
    for i in range(3 * grid.n_blocks[()] * BLOCK_CELLS):
        newton.u_temp[i] = newton.u[i]


@qd.func(requires_top_level=True)
def func_contribute_newton_max_disp(
    particles: MpmParticles,
    grid: qd.template(),  # MpmGrid
    newton: qd.template(),  # MpmNewton
    control: qd.template(),  # NewtonControl
    params: qd.template(),  # MpmParams
    f,
):
    """Reduce into the Newton control the largest |du| component that the particles see through sum_a w_a du_a.

    A GPU takes a block maximum, then one atomic max per block, and a CPU backend one atomic max per particle. The
    maximum is taken over the particles rather than the nodes: a node reached by a sliver of a particle stencil carries
    a vanishing mass and Hessian diagonal, so the preconditioned conjugate gradient (PCG) leaves its du loose while it
    moves no particle."""
    if qd.static(gs.backend == gs.cpu):
        qd.loop_config(name="mpm_contribute_newton_max_disp_cpu")
        for s in range(grid.n_valid[()]):
            qd.atomic_max(control.max_disp[()], particle_max_disp(particles, grid, newton, params, f, s))
    else:
        qd.loop_config(name="mpm_contribute_newton_max_disp", block_dim=REDUCE_BLOCK_DIM)
        for s in range(grid.n_valid_reduce_blocks[()] * REDUCE_BLOCK_DIM):
            disp = gs.qd_float(0.0)
            if s < grid.n_valid[()]:
                disp = particle_max_disp(particles, grid, newton, params, f, s)
            block_max = qd.simt.block.reduce_max(disp, REDUCE_BLOCK_DIM, gs.qd_float)
            if qd.simt.block.thread_idx() == 0:
                qd.atomic_max(control.max_disp[()], block_max)


@qd.func
def particle_max_disp(
    particles: MpmParticles,
    grid: qd.template(),  # MpmGrid
    newton: qd.template(),  # MpmNewton
    params: qd.template(),  # MpmParams
    f,
    s,
):
    """Largest |du| component of a sorted particle, from its interpolated displacement update sum_a w_a du_a."""
    i_p, i_b, env, cell = particle_of_sorted(grid, params, s)
    slots = stencil_node_slots(grid, env, cell)
    w, _ = q1_stencil(particles.pos[f, i_p, i_b] * params.inv_dx[()] - cell.cast(gs.qd_float), params.inv_dx[()])
    ddx = qd.Vector.zero(gs.qd_float, 3)
    for a in qd.static(range(8)):
        ddx += w[a] * qd.Vector([newton.du[3 * slots[a] + d] for d in qd.static(range(3))], dt=gs.qd_float)
    disp = gs.qd_float(0.0)
    for d in qd.static(range(3)):
        disp = qd.max(disp, qd.abs(ddx[d]))
    return disp


@qd.func(requires_top_level=True)
def func_step_forward(
    grid: qd.template(),  # MpmGrid
    newton: qd.template(),  # MpmNewton
    control: qd.template(),  # NewtonControl
):
    qd.loop_config(name="mpm_step_forward")
    for i in range(3 * grid.n_blocks[()] * BLOCK_CELLS):
        newton.u[i] = newton.u_temp[i] + control.alpha[()] * newton.du[i]


@qd.func(requires_top_level=True)
def func_elastic_energy_partial(
    particles: MpmParticles,
    particles_info: MpmParticlesInfo,
    materials: MpmMaterialTable,
    grid: qd.template(),  # MpmGrid
    newton: qd.template(),  # MpmNewton
    params: qd.template(),  # MpmParams
    f,
    dt,
):
    """Write the per-block partial sums of V dt^2 psi(F_trial), of the contact-row energy and of the constraint spring
    energy over the sorted particles.

    A GPU takes a block reduction and a CPU the ordered sum of one thread per block, both bitwise reproducible."""
    if qd.static(gs.backend == gs.cpu):
        qd.loop_config(name="mpm_elastic_energy_partial_cpu")
        for block in range(grid.n_valid_reduce_blocks[()]):
            total = gs.qd_float(0.0)
            end = qd.min(block * REDUCE_BLOCK_DIM + REDUCE_BLOCK_DIM, grid.n_valid[()])
            for s in range(block * REDUCE_BLOCK_DIM, end):
                total += particle_energy(particles, particles_info, materials, grid, newton, params, f, dt, s)
            newton.energy_partials[block] = total
    else:
        qd.loop_config(name="mpm_elastic_energy_partial", block_dim=REDUCE_BLOCK_DIM)
        for s in range(grid.n_valid_reduce_blocks[()] * REDUCE_BLOCK_DIM):
            energy = gs.qd_float(0.0)
            if s < grid.n_valid[()]:
                energy = particle_energy(particles, particles_info, materials, grid, newton, params, f, dt, s)
            write_block_partial(energy, newton.energy_partials, s // REDUCE_BLOCK_DIM, gs.qd_float)


@qd.func
def particle_energy(
    particles: MpmParticles,
    particles_info: MpmParticlesInfo,
    materials: MpmMaterialTable,
    grid: qd.template(),  # MpmGrid
    newton: qd.template(),  # MpmNewton
    params: qd.template(),  # MpmParams
    f,
    dt,
    s,
):
    """V dt^2 psi(F_trial), the contact-row energy and the constraint spring energy of the sorted particle s."""
    i_p, i_b, env, cell = particle_of_sorted(grid, params, s)
    slots = stencil_node_slots(grid, env, cell)
    F_tr, grad_w, F = trial_deformation(particles, grid, newton, params, f, i_p, i_b, cell, slots)
    slot = particles_info.material_idx[i_p]
    mu, lam = material_params(materials, params, slot, particles.Jp[f, i_p, i_b])
    psi = elastic_energy(materials, slot, signed_singular_values(F_tr), mu, lam)
    psi_fibre, P_fibre, fibre_curvature = fibre_terms(
        materials, params, slot, particles.actu[f, i_p, i_b], particles_info.muscle_direction[i_p], F_tr
    )
    psi += psi_fibre
    dx, w = particle_trial_disp(particles, newton, params, f, i_p, i_b, cell, slots)
    dr = qd.Vector([newton.contact_dr[3 * s + d] for d in qd.static(range(3))], dt=gs.qd_float)
    energy_wall, grad_wall, hess_wall = contact_row_terms(newton, s, dx, 0, WALL_ROWS)
    energy_rigid, grad_rigid, hess_rigid = contact_row_terms(newton, s, dx - dr, WALL_ROWS, CONTACT_ROWS)
    energy_spring, grad_spring, hess_spring = constraint_row_terms(newton, s, particles.pos[f, i_p, i_b], dx)
    return params.particle_volume[()] * dt * dt * psi + energy_wall + energy_rigid + energy_spring


@qd.func(requires_top_level=True)
def func_kinetic_energy_partial(
    grid: qd.template(),  # MpmGrid
    newton: qd.template(),  # MpmNewton
    dt,
):
    """Write the per-block partial sums of 1/2 m |u - dt v_free|^2 over the free components, after the particle partials
    (see func_elastic_energy_partial)."""
    if qd.static(gs.backend == gs.cpu):
        qd.loop_config(name="mpm_kinetic_energy_partial_cpu")
        for block in range(grid.n_node_reduce_blocks[()]):
            total = gs.qd_float(0.0)
            end = qd.min(block * REDUCE_BLOCK_DIM + REDUCE_BLOCK_DIM, grid.n_blocks[()] * BLOCK_CELLS)
            for node in range(block * REDUCE_BLOCK_DIM, end):
                total += node_kinetic_energy(grid, newton, dt, node)
            n_particle_partials = newton.n_particle_partials[()]
            newton.energy_partials[n_particle_partials + block] = total
    else:
        qd.loop_config(name="mpm_kinetic_energy_partial", block_dim=REDUCE_BLOCK_DIM)
        for node in range(grid.n_node_reduce_blocks[()] * REDUCE_BLOCK_DIM):
            energy = gs.qd_float(0.0)
            if node < grid.n_blocks[()] * BLOCK_CELLS:
                energy = node_kinetic_energy(grid, newton, dt, node)
            write_block_partial(
                energy, newton.energy_partials, newton.n_particle_partials[()] + node // REDUCE_BLOCK_DIM, gs.qd_float
            )


@qd.func
def node_kinetic_energy(
    grid: qd.template(),  # MpmGrid
    newton: qd.template(),  # MpmNewton
    dt,
    node,
):
    """1/2 m |u - dt v_free|^2 of a node over its free components."""
    energy = gs.qd_float(0.0)
    for d in qd.static(range(3)):
        if grid.dof_fixed[3 * node + d] == 0:
            r = newton.u[3 * node + d] - dt * grid.node_v_free[3 * node + d]
            energy += 0.5 * grid.node_mass[node] * r * r
    return energy


@qd.func(requires_top_level=True)
def func_energy_final(
    grid: qd.template(),  # MpmGrid
    newton: qd.template(),  # MpmNewton
    control: qd.template(),  # NewtonControl
    is_trial,
):
    """Sum the particle and node partials in a fixed order into the start or the trial energy: one block on a GPU, one
    thread on a CPU."""
    if qd.static(gs.backend == gs.cpu):
        for _ in range(1):
            n_particle_partials = newton.n_particle_partials[()]
            total = gs.qd_float(0.0)
            for i in range(grid.n_valid_reduce_blocks[()]):
                total += newton.energy_partials[i]
            for i in range(grid.n_node_reduce_blocks[()]):
                total += newton.energy_partials[n_particle_partials + i]
            if is_trial:
                control.energy_trial[()] = total
            else:
                control.energy_start[()] = total
    else:
        qd.loop_config(name="mpm_energy_final", block_dim=REDUCE_BLOCK_DIM)
        for tid in range(REDUCE_BLOCK_DIM):
            n_valid_reduce_blocks = grid.n_valid_reduce_blocks[()]
            n_node_reduce_blocks = grid.n_node_reduce_blocks[()]
            n_particle_partials = newton.n_particle_partials[()]
            value = gs.qd_float(0.0)
            for i_chunk_ in range((n_valid_reduce_blocks + REDUCE_BLOCK_DIM - 1) // REDUCE_BLOCK_DIM):
                i = i_chunk_ * REDUCE_BLOCK_DIM + tid
                if i < n_valid_reduce_blocks:
                    value += newton.energy_partials[i]
            for i_chunk_ in range((n_node_reduce_blocks + REDUCE_BLOCK_DIM - 1) // REDUCE_BLOCK_DIM):
                i = i_chunk_ * REDUCE_BLOCK_DIM + tid
                if i < n_node_reduce_blocks:
                    value += newton.energy_partials[n_particle_partials + i]
            total = qd.simt.block.reduce_add(value, REDUCE_BLOCK_DIM, gs.qd_float)
            if tid == 0:
                if is_trial:
                    control.energy_trial[()] = total
                else:
                    control.energy_start[()] = total


@qd.func(requires_top_level=True)
def func_energy(
    particles: MpmParticles,
    particles_info: MpmParticlesInfo,
    materials: MpmMaterialTable,
    grid: qd.template(),  # MpmGrid
    newton: qd.template(),  # MpmNewton
    control: qd.template(),  # NewtonControl
    params: qd.template(),  # MpmParams
    f,
    dt,
    is_trial,
):
    func_elastic_energy_partial(particles, particles_info, materials, grid, newton, params, f, dt)
    func_kinetic_energy_partial(grid, newton, dt)
    func_energy_final(grid, newton, control, is_trial)

"""PFPx Jacobians (@qd.func device version).

Ports the pinned reference ``gipc_pfpx.h``. Each function writes into an output template array.
PT/EE: (12, 9), PP: (6,), PE: (9, 4). Only the nonzero column is populated.
"""

from __future__ import annotations

import quadrants as qd


@qd.func
def gipc_pfpx_pt(
    x0x: qd.f64,
    x0y: qd.f64,
    x0z: qd.f64,
    x1x: qd.f64,
    x1y: qd.f64,
    x1z: qd.f64,
    x2x: qd.f64,
    x2y: qd.f64,
    x2z: qd.f64,
    x3x: qd.f64,
    x3y: qd.f64,
    x3z: qd.f64,
    d_hat_sqrt: qd.f64,
    out: qd.template(),  # qd.Vector
):
    """PT PFPx: writes 12 values into out[0..11] (column 8 of 12x9 matrix)."""
    t2 = 1.0 / d_hat_sqrt
    t12 = -x1x + x0x
    t13 = -x1y + x0y
    t14 = -x1z + x0z
    t15 = -x2x + x1x
    t16 = -x2y + x1y
    t17 = -x2z + x1z
    t18 = -x3x + x1x
    t19 = -x3y + x1y
    t20 = -x3z + x1z
    t21 = -x3x + x2x
    t22 = -x3y + x2y
    t23 = -x3z + x2z
    t33 = t15 * t19 - t16 * t18
    t34 = t15 * t20 - t17 * t18
    t35 = t16 * t20 - t17 * t19
    t44 = 1.0 / (t33 * t33 + t34 * t34 + t35 * t35)
    t46 = t14 * t33 + t12 * t35 - t13 * t34
    t47 = t46 * t46
    t49 = 1.0 / qd.sqrt(t44 * t47)

    out[0] = t2 * t35 * t44 * t46 * t49
    out[1] = -t2 * t34 * t44 * t46 * t49
    out[2] = t2 * t33 * t44 * t46 * t49

    b_d = t44 * t44 * t47
    t44_m = t44 * t46
    t47n = t2 * t49

    out[3] = t47n * (b_d * (t22 * t33 * 2.0 + t23 * t34 * 2.0) + t44_m * ((t35 + t13 * t23) - t14 * t22) * 2.0) * -0.5
    out[4] = t47n * (b_d * (t21 * t33 * 2.0 - t23 * t35 * 2.0) + t44_m * ((t34 + t12 * t23) - t14 * t21) * 2.0) / 2.0
    out[5] = t47n * (b_d * (t21 * t34 * 2.0 + t22 * t35 * 2.0) - t44_m * ((t33 + t12 * t22) - t13 * t21) * 2.0) / 2.0
    out[6] = t47n * (t44_m * (t13 * t20 - t14 * t19) * 2.0 + b_d * (t19 * t33 * 2.0 + t20 * t34 * 2.0)) / 2.0
    out[7] = t47n * (t44_m * (t12 * t20 - t14 * t18) * 2.0 + b_d * (t18 * t33 * 2.0 - t20 * t35 * 2.0)) * -0.5
    out[8] = t47n * (t44_m * (t12 * t19 - t13 * t18) * 2.0 - b_d * (t18 * t34 * 2.0 + t19 * t35 * 2.0)) / 2.0
    out[9] = t47n * (t44_m * (t13 * t17 - t14 * t16) * 2.0 + b_d * (t16 * t33 * 2.0 + t17 * t34 * 2.0)) * -0.5
    out[10] = t47n * (t44_m * (t12 * t17 - t14 * t15) * 2.0 + b_d * (t15 * t33 * 2.0 - t17 * t35 * 2.0)) / 2.0
    out[11] = t47n * (t44_m * (t12 * t16 - t13 * t15) * 2.0 - b_d * (t15 * t34 * 2.0 + t16 * t35 * 2.0)) * -0.5


@qd.func
def gipc_pfpx_ee(
    x0x: qd.f64,
    x0y: qd.f64,
    x0z: qd.f64,
    x1x: qd.f64,
    x1y: qd.f64,
    x1z: qd.f64,
    x2x: qd.f64,
    x2y: qd.f64,
    x2z: qd.f64,
    x3x: qd.f64,
    x3y: qd.f64,
    x3z: qd.f64,
    d_hat_sqrt: qd.f64,
    out: qd.template(),  # qd.Vector
):
    """EE PFPx: writes 12 values into out[0..11] (column 8 of 12x9 matrix)."""
    t12 = -x1x + x0x
    t13 = -x1y + x0y
    t14 = -x1z + x0z
    t15 = -x2x + x0x
    t16 = -x2y + x0y
    t17 = -x2z + x0z
    t18 = -x3x + x2x
    t19 = -x3y + x2y
    t20 = -x3z + x2z
    t33 = t15 * t19
    t34 = t16 * t18
    t35 = t15 * t20
    t36 = t17 * t18
    t37 = t16 * t20
    t38 = t17 * t19
    t45 = t12 * t19 - t13 * t18
    t46 = t12 * t20 - t14 * t18
    t47 = t13 * t20 - t14 * t19
    t76 = 1.0 / (t45 * t45 + t46 * t46 + t47 * t47)
    t78 = t17 * t45 + t15 * t47 - t16 * t46
    t77 = t76 * t76
    t79 = t78 * t78
    t82 = (t13 * t45 * 2.0 + t14 * t46 * 2.0) * t77 * t79
    t83 = (t12 * t46 * 2.0 + t13 * t47 * 2.0) * t77 * t79
    t84 = (t19 * t45 * 2.0 + t20 * t46 * 2.0) * t77 * t79
    t85 = (t18 * t46 * 2.0 + t19 * t47 * 2.0) * t77 * t79
    t86 = (t12 * t45 * 2.0 - t14 * t47 * 2.0) * t77 * t79
    t87 = (t18 * t45 * 2.0 - t20 * t47 * 2.0) * t77 * t79
    t78_m = t78 * t76
    t77_inv = 1.0 / d_hat_sqrt * (1.0 / qd.sqrt(t76 * t79))

    out[0] = t77_inv * (t84 - t78_m * ((-t37 + t38) + t47) * 2.0) * -0.5
    out[1] = t77_inv * (t87 - t78_m * ((-t35 + t36) + t46) * 2.0) / 2.0
    out[2] = t77_inv * (t85 + t78_m * ((-t33 + t34) + t45) * 2.0) / 2.0
    out[3] = t77_inv * (t84 + t78_m * (t37 - t38) * 2.0) / 2.0
    out[4] = t77_inv * (t87 + t78_m * (t35 - t36) * 2.0) * -0.5
    out[5] = t77_inv * (t85 - t78_m * (t33 - t34) * 2.0) * -0.5

    t18_v = t13 * t17 - t14 * t16
    out[6] = t77_inv * (t82 - t78_m * (t18_v + t47) * 2.0) / 2.0
    t20_v = t12 * t17 - t14 * t15
    out[7] = t77_inv * (t86 - t78_m * (t20_v + t46) * 2.0) * -0.5
    t19_v = t12 * t16 - t13 * t15
    out[8] = t77_inv * (t83 + t78_m * (t19_v + t45) * 2.0) * -0.5
    out[9] = t77_inv * (t82 - t78_m * t18_v * 2.0) * -0.5
    out[10] = t77_inv * (t86 - t78_m * t20_v * 2.0) / 2.0
    out[11] = t77_inv * (t83 + t78_m * t19_v * 2.0) / 2.0


@qd.func
def gipc_pfpx_pp(
    x0x: qd.f64,
    x0y: qd.f64,
    x0z: qd.f64,
    x1x: qd.f64,
    x1y: qd.f64,
    x1z: qd.f64,
    d_hat_sqrt: qd.f64,
    out: qd.template(),  # qd.Vector
):
    """PP PFPx: writes 6 values into out[0..5]."""
    t8 = 1.0 / d_hat_sqrt
    t25 = -x1x + x0x
    t26 = -x1y + x0y
    t27 = -x1z + x0z
    t25_inv = 1.0 / qd.sqrt(t25 * t25 + t26 * t26 + t27 * t27)
    r0 = t8 * (x0x * 2.0 - x1x * 2.0) * t25_inv / 2.0
    r1 = t8 * (x0y * 2.0 - x1y * 2.0) * t25_inv / 2.0
    r2 = t8 * (x0z * 2.0 - x1z * 2.0) * t25_inv / 2.0
    out[0] = r0
    out[1] = r1
    out[2] = r2
    out[3] = -r0
    out[4] = -r1
    out[5] = -r2


@qd.func
def gipc_pfpx_pe(
    x0x: qd.f64,
    x0y: qd.f64,
    x0z: qd.f64,
    x1x: qd.f64,
    x1y: qd.f64,
    x1z: qd.f64,
    x2x: qd.f64,
    x2y: qd.f64,
    x2z: qd.f64,
    d_hat_sqrt: qd.f64,
    out: qd.template(),  # qd.Vector
):
    """PE PFPx: writes 9 values into out[0..8] (column 3 of 9x4 matrix)."""
    t8 = 1.0 / d_hat_sqrt
    t18 = -x1x + x0x
    t19 = -x1y + x0y
    t20 = -x1z + x0z
    t21 = -x2x + x0x
    t22 = -x2y + x0y
    t23 = -x2z + x0z
    t24 = -x2x + x1x
    t25 = -x2y + x1y
    t26 = -x2z + x1z
    t43 = 1.0 / (t24 * t24 + t25 * t25 + t26 * t26)
    t45 = t18 * t22 - t19 * t21
    t46 = t18 * t23 - t20 * t21
    t47 = t19 * t23 - t20 * t22
    t44 = t43 * t43
    t51 = t45 * t45 + t46 * t46 + t47 * t47
    t54 = (x1x * 2.0 - x2x * 2.0) * t44 * t51
    t55 = (x1y * 2.0 - x2y * 2.0) * t44 * t51
    t56 = (x1z * 2.0 - x2z * 2.0) * t44 * t51
    t44_sqrt = 1.0 / qd.sqrt(t43 * t51)

    t51_sc = t8 * t43 * t44_sqrt
    out[0] = t51_sc * (t25 * t45 * 2.0 + t26 * t46 * 2.0) / 2.0
    out[1] = t51_sc * (t24 * t45 * 2.0 - t26 * t47 * 2.0) * -0.5
    out[2] = t51_sc * (t24 * t46 * 2.0 + t25 * t47 * 2.0) * -0.5

    t51_sc2 = t8 * t44_sqrt
    out[3] = t51_sc2 * (t54 + t43 * (t22 * t45 * 2.0 + t23 * t46 * 2.0)) * -0.5
    out[4] = t51_sc2 * (t55 - t43 * (t21 * t45 * 2.0 - t23 * t47 * 2.0)) * -0.5
    out[5] = t51_sc2 * (t56 - t43 * (t21 * t46 * 2.0 + t22 * t47 * 2.0)) * -0.5
    out[6] = t51_sc2 * (t54 + t43 * (t19 * t45 * 2.0 + t20 * t46 * 2.0)) / 2.0
    out[7] = t51_sc2 * (t55 - t43 * (t18 * t45 * 2.0 - t20 * t47 * 2.0)) / 2.0
    out[8] = t51_sc2 * (t56 - t43 * (t18 * t46 * 2.0 + t19 * t47 * 2.0)) / 2.0


# Mollified PFPx variants from the same pinned-reference header.
@qd.func
def gipc_pfpx_ee_mollified(
    x0x: qd.f64,
    x0y: qd.f64,
    x0z: qd.f64,
    x1x: qd.f64,
    x1y: qd.f64,
    x1z: qd.f64,
    x2x: qd.f64,
    x2y: qd.f64,
    x2z: qd.f64,
    x3x: qd.f64,
    x3y: qd.f64,
    x3z: qd.f64,
    d_hat_sqrt: qd.f64,
    c4: qd.template(),  # qd.Vector
    c8: qd.template(),  # qd.Vector
):
    """EE mollified PFPx columns 4 and 8."""
    t12 = x0x - x1x
    t13 = x0y - x1y
    t14 = x0z - x1z
    t15 = x0x - x2x
    t16 = x0y - x2y
    t17 = x0z - x2z
    t18 = x2x - x3x
    t19 = x2y - x3y
    t20 = x2z - x3z
    t33 = t15 * t19
    t34 = t16 * t18
    t35 = t15 * t20
    t36 = t17 * t18
    t37 = t16 * t20
    t38 = t17 * t19
    t45 = t12 * t19 - t13 * t18
    t46 = t12 * t20 - t14 * t18
    t47 = t13 * t20 - t14 * t19
    t69 = t13 * t45 * 2.0 + t14 * t46 * 2.0
    t70 = t12 * t46 * 2.0 + t13 * t47 * 2.0
    t71 = t19 * t45 * 2.0 + t20 * t46 * 2.0
    t72 = t18 * t46 * 2.0 + t19 * t47 * 2.0
    t19_sq = t45 * t45 + t46 * t46 + t47 * t47
    t73 = t12 * t45 * 2.0 - t14 * t47 * 2.0
    t74 = t18 * t45 * 2.0 - t20 * t47 * 2.0
    t76 = 1.0 / t19_sq
    t19_sq_inv = 1.0 / qd.sqrt(t19_sq)
    t79 = t17 * t45 + t15 * t47 - t16 * t46
    t77 = t76 * t76
    t80 = t79 * t79
    t81 = t69 * t19_sq_inv / 2.0
    t82 = t70 * t19_sq_inv / 2.0
    t83 = t71 * t19_sq_inv / 2.0
    t84 = t72 * t19_sq_inv / 2.0
    t85 = t73 * t19_sq_inv / 2.0
    t86 = t74 * t19_sq_inv / 2.0
    t89 = t69 * t77 * t80
    t90 = t70 * t77 * t80
    t20_reuse = t71 * t77 * t80
    t18_reuse = t72 * t77 * t80
    t71_reuse = t73 * t77 * t80
    t19_reuse = t74 * t77 * t80

    t70_r = t76 * t79
    t69_r = 1.0 / d_hat_sqrt * (1.0 / qd.sqrt(t76 * t80))

    c4[0] = t83
    c8[0] = t69_r * (t20_reuse - t70_r * ((-t37 + t38) + t47) * 2.0) * -0.5
    c4[1] = -t86
    c8[1] = t69_r * (t19_reuse - t70_r * ((-t35 + t36) + t46) * 2.0) / 2.0
    c4[2] = -t84
    c8[2] = t69_r * (t18_reuse + t70_r * ((-t33 + t34) + t45) * 2.0) / 2.0
    c4[3] = -t83
    c8[3] = t69_r * (t20_reuse + t70_r * (t37 - t38) * 2.0) / 2.0
    c4[4] = t86
    c8[4] = t69_r * (t19_reuse + t70_r * (t35 - t36) * 2.0) * -0.5
    c4[5] = t84
    c8[5] = t69_r * (t18_reuse - t70_r * (t33 - t34) * 2.0) * -0.5

    t20_v = t13 * t17 - t14 * t16
    c4[6] = -t81
    c8[6] = t69_r * (t89 - t70_r * (t20_v + t47) * 2.0) / 2.0
    t18_v = t12 * t17 - t14 * t15
    c4[7] = t85
    c8[7] = t69_r * (t71_reuse - t70_r * (t18_v + t46) * 2.0) * -0.5
    t19_v = t12 * t16 - t13 * t15
    c4[8] = t82
    c8[8] = t69_r * (t90 + t70_r * (t19_v + t45) * 2.0) * -0.5
    c4[9] = t81
    c8[9] = t69_r * (t89 - t70_r * t20_v * 2.0) * -0.5
    c4[10] = -t85
    c8[10] = t69_r * (t71_reuse - t70_r * t18_v * 2.0) / 2.0
    c4[11] = -t82
    c8[11] = t69_r * (t90 + t70_r * t19_v * 2.0) / 2.0


@qd.func
def gipc_pfpx_pp_mollified(
    x0x: qd.f64,
    x0y: qd.f64,
    x0z: qd.f64,
    x1x: qd.f64,
    x1y: qd.f64,
    x1z: qd.f64,
    x2x: qd.f64,
    x2y: qd.f64,
    x2z: qd.f64,
    x3x: qd.f64,
    x3y: qd.f64,
    x3z: qd.f64,
    d_hat_sqrt: qd.f64,
    c4: qd.template(),  # qd.Vector
    c8: qd.template(),  # qd.Vector
):
    """PP mollified PFPx columns 4 and 8.

    The PP distance depends only on ``x0``/``x1``, so ``c8`` is zero on the
    trailing six rows (the pinned reference writes those zeros explicitly).
    """
    t8 = 1.0 / d_hat_sqrt
    t49_orig = x0x - x1x
    t82_orig = x0y - x1y
    t81_orig = x0z - x1z
    t24 = x0x - x2x
    t25 = x0y - x2y
    t26 = x0z - x2z
    t27 = x1x - x3x
    t28 = x1y - x3y
    t29 = x1z - x3z
    t46 = t24 * t28 - t25 * t27
    t47 = t24 * t29 - t26 * t27
    t48 = t25 * t29 - t26 * t28
    t49_inv = 1.0 / qd.sqrt(t49_orig * t49_orig + t82_orig * t82_orig + t81_orig * t81_orig)
    t67 = t8 * (x0x * 2.0 - x1x * 2.0) * t49_inv / 2.0
    t68 = t8 * (x0y * 2.0 - x1y * 2.0) * t49_inv / 2.0
    t69 = t8 * (x0z * 2.0 - x1z * 2.0) * t49_inv / 2.0
    t49_inv2 = 1.0 / qd.sqrt(t46 * t46 + t47 * t47 + t48 * t48)
    t78 = (t25 * t46 * 2.0 + t26 * t47 * 2.0) * t49_inv2 / 2.0
    t8_new = (t24 * t47 * 2.0 + t25 * t48 * 2.0) * t49_inv2 / 2.0
    t25_new = (t28 * t46 * 2.0 + t29 * t47 * 2.0) * t49_inv2 / 2.0
    t81_new = (t27 * t47 * 2.0 + t28 * t48 * 2.0) * t49_inv2 / 2.0
    t82_new = (t24 * t46 * 2.0 - t26 * t48 * 2.0) * t49_inv2 / 2.0
    t49_new = (t27 * t46 * 2.0 - t29 * t48 * 2.0) * t49_inv2 / 2.0

    c4[0] = t25_new
    c8[0] = t67
    c4[1] = -t49_new
    c8[1] = t68
    c4[2] = -t81_new
    c8[2] = t69
    c4[3] = -t78
    c8[3] = -t67
    c4[4] = t82_new
    c8[4] = -t68
    c4[5] = t8_new
    c8[5] = -t69
    c4[6] = -t25_new
    c4[7] = t49_new
    c4[8] = t81_new
    c4[9] = t78
    c4[10] = -t82_new
    c4[11] = -t8_new
    for i in qd.static(range(6, 12)):
        c8[i] = qd.f64(0.0)


@qd.func
def gipc_pfpx_pe_mollified(
    x0x: qd.f64,
    x0y: qd.f64,
    x0z: qd.f64,
    x1x: qd.f64,
    x1y: qd.f64,
    x1z: qd.f64,
    x2x: qd.f64,
    x2y: qd.f64,
    x2z: qd.f64,
    x3x: qd.f64,
    x3y: qd.f64,
    x3z: qd.f64,
    d_hat_sqrt: qd.f64,
    c4: qd.template(),  # qd.Vector
    c8: qd.template(),  # qd.Vector
):
    """PE mollified PFPx columns 4 and 8.

    The PE distance is independent of ``x3``, so ``c8`` is zero on the trailing
    three rows.
    """
    t8 = 1.0 / d_hat_sqrt
    t21 = x0x - x1x
    t22 = x0y - x1y
    t23 = x0z - x1z
    t24 = x0x - x2x
    t25 = x0y - x2y
    t26 = x0z - x2z
    t27 = x0x - x3x
    t28 = x1x - x2x
    t29 = x0y - x3y
    t30 = x1y - x2y
    t31 = x0z - x3z
    t32 = x1z - x2z
    t58 = 1.0 / (t28 * t28 + t30 * t30 + t32 * t32)
    t60 = t21 * t25 - t22 * t24
    t61 = t21 * t26 - t23 * t24
    t62 = t22 * t26 - t23 * t25
    t63 = t27 * t30 - t28 * t29
    t64 = t27 * t32 - t28 * t31
    t65 = t29 * t32 - t30 * t31
    t59 = t58 * t58
    t92 = t60 * t60 + t61 * t61 + t62 * t62
    t94 = 1.0 / qd.sqrt(t63 * t63 + t64 * t64 + t65 * t65)
    t97 = (x1x * 2.0 - x2x * 2.0) * t59 * t92
    t98 = (x1y * 2.0 - x2y * 2.0) * t59 * t92
    t99 = (x1z * 2.0 - x2z * 2.0) * t59 * t92
    t96 = 1.0 / qd.sqrt(t58 * t92)
    t102 = (t29 * t63 * 2.0 + t31 * t64 * 2.0) * t94 / 2.0
    t103 = (t30 * t63 * 2.0 + t32 * t64 * 2.0) * t94 / 2.0
    t104 = (t27 * t64 * 2.0 + t29 * t65 * 2.0) * t94 / 2.0
    t64_new = (t28 * t64 * 2.0 + t30 * t65 * 2.0) * t94 / 2.0
    t29_new = (t27 * t63 * 2.0 - t31 * t65 * 2.0) * t94 / 2.0
    t59_new = (t28 * t63 * 2.0 - t32 * t65 * 2.0) * t94 / 2.0

    t92_sc = t8 * t58 * t96
    c4[0] = t103
    c8[0] = t92_sc * (t30 * t60 * 2.0 + t32 * t61 * 2.0) / 2.0
    c4[1] = -t59_new
    c8[1] = t92_sc * (t28 * t60 * 2.0 - t32 * t62 * 2.0) * -0.5
    c4[2] = -t64_new
    c8[2] = t92_sc * (t28 * t61 * 2.0 + t30 * t62 * 2.0) * -0.5

    t92_sc2 = t8 * t96
    c4[3] = -t102
    c8[3] = t92_sc2 * (t97 + t58 * (t25 * t60 * 2.0 + t26 * t61 * 2.0)) * -0.5
    c4[4] = t29_new
    c8[4] = t92_sc2 * (t98 - t58 * (t24 * t60 * 2.0 - t26 * t62 * 2.0)) * -0.5
    c4[5] = t104
    c8[5] = t92_sc2 * (t99 - t58 * (t24 * t61 * 2.0 + t25 * t62 * 2.0)) * -0.5
    c4[6] = t102
    c8[6] = t92_sc2 * (t97 + t58 * (t22 * t60 * 2.0 + t23 * t61 * 2.0)) / 2.0
    c4[7] = -t29_new
    c8[7] = t92_sc2 * (t98 - t58 * (t21 * t60 * 2.0 - t23 * t62 * 2.0)) / 2.0
    c4[8] = -t104
    c8[8] = t92_sc2 * (t99 - t58 * (t21 * t61 * 2.0 + t22 * t62 * 2.0)) / 2.0
    c4[9] = -t103
    c4[10] = t59_new
    c4[11] = t64_new
    for i in qd.static(range(9, 12)):
        c8[i] = qd.f64(0.0)

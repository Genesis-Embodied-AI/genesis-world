"""Project a symmetric 2x2 matrix to its nearest PSD matrix (@qd.func).

Device counterpart of ``make_pd.py``, specialised to the 2x2 block that the
mollified barrier Hessian projects (the pinned reference's ``gipc_make_pd<double, 2>``).

The pinned reference reaches this through ``Eigen::SelfAdjointEigenSolver::computeDirect``, which
for 2x2 is itself a closed form: eigenvalues ``tr/2 +- sqrt(((a-c)/2)^2 + b^2)``
and, for the eigenvector, whichever of ``(b, l-a)`` / ``(l-c, b)`` has the larger
norm (Eigen picks by the same comparison, expressed on the trace-shifted
matrix). The transcription below is that closed form. Eigen additionally
normalises by the largest shifted coefficient before taking the root, which
moves results by at most an ulp -- far under the 1e-11 tolerance the golden
assembly comparison uses -- so the scaling step is dropped.
"""

from __future__ import annotations

import quadrants as qd


@qd.func
def gipc_make_pd(
    a: qd.f64,
    b: qd.f64,
    c: qd.f64,
    out: qd.template(),  # qd.Matrix
):
    """PSD-project symmetric ``[[a, b], [b, c]]`` into ``out = (a', b', c')``.

    Negative eigenvalues are clamped to zero. A matrix that is already PSD is
    returned untouched, matching the pinned reference's early return -- so the common case never
    round-trips through the eigen-decomposition.
    """
    half_tr = 0.5 * (a + c)
    half_diff = 0.5 * (a - c)
    radius = qd.sqrt(half_diff * half_diff + b * b)
    lam_min = half_tr - radius
    lam_max = half_tr + radius

    if lam_min >= 0.0:
        out[0] = a
        out[1] = b
        out[2] = c
    elif lam_max <= 0.0:
        out[0] = qd.f64(0.0)
        out[1] = qd.f64(0.0)
        out[2] = qd.f64(0.0)
    else:
        # Rank-1 reconstruction from the surviving eigenpair. Of the two
        # equivalent eigenvector forms, take the longer one: their norms are
        # |radius -+ half_diff| and one of those is always >= radius > 0, so the
        # normalisation below can never divide by a cancelled-out zero.
        v0 = b
        v1 = radius - half_diff
        w0 = radius + half_diff
        w1 = b
        if w0 * w0 + w1 * w1 > v0 * v0 + v1 * v1:
            v0 = w0
            v1 = w1
        s = lam_max / (v0 * v0 + v1 * v1)
        out[0] = s * v0 * v0
        out[1] = s * v0 * v1
        out[2] = s * v1 * v1

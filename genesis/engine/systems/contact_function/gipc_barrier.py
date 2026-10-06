from __future__ import annotations

import quadrants as qd


@qd.func
def gipc_barrier_energy(D: qd.f64, d_hat: qd.f64, xi: qd.f64, kappa: qd.f64) -> qd.f64:
    xi_sq = xi * xi
    d_tilde = D - xi_sq
    dH_tilde = d_hat * d_hat + 2.0 * d_hat * xi
    r = d_tilde / dH_tilde
    lnr = qd.log(r)
    rm1 = r - 1.0
    return kappa * rm1 * rm1 * lnr * lnr


@qd.func
def gipc_barrier_first_derivative(D: qd.f64, d_hat: qd.f64, xi: qd.f64, kappa: qd.f64) -> qd.f64:
    xi_sq = xi * xi
    d_tilde = D - xi_sq
    dH_tilde = d_hat * d_hat + 2.0 * d_hat * xi
    r = d_tilde / dH_tilde
    lnr = qd.log(r)
    A = lnr + (r - 1.0) / r
    return 2.0 * kappa * (r - 1.0) * lnr * A / dH_tilde


@qd.func
def gipc_barrier_second_derivative(D: qd.f64, d_hat: qd.f64, xi: qd.f64, kappa: qd.f64) -> qd.f64:
    xi_sq = xi * xi
    d_tilde = D - xi_sq
    dH_tilde = d_hat * d_hat + 2.0 * d_hat * xi
    r = d_tilde / dH_tilde
    lnr = qd.log(r)
    A = lnr + (r - 1.0) / r
    return 2.0 * kappa * (A * A + (r * r - 1.0) * lnr / (r * r)) / (dH_tilde * dH_tilde)


@qd.func
def gipc_normal_force(D: qd.f64, d_hat: qd.f64, xi: qd.f64, kappa: qd.f64) -> qd.f64:
    return -gipc_barrier_first_derivative(D, d_hat, xi, kappa) * 2.0 * qd.sqrt(D)


@qd.func
def gipc_barrier_energy_mollified(
    D: qd.f64,
    d_hat: qd.f64,
    xi: qd.f64,
    kappa: qd.f64,
    I1: qd.f64,
    eps_x: qd.f64,
) -> qd.f64:
    mollifier = -(I1 * I1) / (eps_x * eps_x) + 2.0 * I1 / eps_x
    return mollifier * gipc_barrier_energy(D, d_hat, xi, kappa)

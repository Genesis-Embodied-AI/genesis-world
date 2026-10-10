import numpy as np
import pytest
import quadrants as qd

import genesis as gs
from genesis.engine.solvers.mochi.rod import (
    func_rod_axial,
    func_rod_bend_twist,
    func_rod_bend_twist_measures,
    func_rod_parallel_transport,
)
from genesis.utils.misc import tensor_to_array

from ..utils.assertions import assert_allclose


def _mochi_scene(show_viewer, dt, gravity=(0.0, 0.0, -9.8), **mochi_kwargs):
    return gs.Scene(
        sim_options=gs.options.SimOptions(dt=dt, gravity=gravity),
        mochi_options=gs.options.MochiOptions(**mochi_kwargs),
        show_viewer=show_viewer,
    )


def _straight_rod_points(n_segments, length):
    return np.stack([np.linspace(0.0, length, n_segments + 1), np.zeros(n_segments + 1), np.zeros(n_segments + 1)], -1)


@qd.kernel
def _kernel_measures(x: qd.types.ndarray(), a: qd.types.ndarray(), L: float, tiny: float, out: qd.types.ndarray()):
    x0 = qd.Vector([x[0, 0], x[0, 1], x[0, 2]])
    x1 = qd.Vector([x[1, 0], x[1, 1], x[1, 2]])
    x2 = qd.Vector([x[2, 0], x[2, 1], x[2, 2]])
    a0 = qd.Vector([a[0, 0], a[0, 1], a[0, 2]])
    a1 = qd.Vector([a[1, 0], a[1, 1], a[1, 2]])
    ka, kb, tw = func_rod_bend_twist_measures(x0, x1, x2, a0, a1, L, tiny)
    out[0] = ka
    out[1] = kb
    out[2] = tw


@qd.kernel
def _kernel_bend_twist(
    x: qd.types.ndarray(),
    a: qd.types.ndarray(),
    L: float,
    ref: qd.types.ndarray(),
    EI1: float,
    EI2: float,
    GJ: float,
    tiny: float,
    energy: qd.types.ndarray(),
    res: qd.types.ndarray(),
    K: qd.types.ndarray(),
):
    x0 = qd.Vector([x[0, 0], x[0, 1], x[0, 2]])
    x1 = qd.Vector([x[1, 0], x[1, 1], x[1, 2]])
    x2 = qd.Vector([x[2, 0], x[2, 1], x[2, 2]])
    a0 = qd.Vector([a[0, 0], a[0, 1], a[0, 2]])
    a1 = qd.Vector([a[1, 0], a[1, 1], a[1, 2]])
    e, r, Km, _ka, _kb, _tw = func_rod_bend_twist(
        x0, x1, x2, a0, a1, L, ref[0], ref[1], ref[2], ref[0], ref[1], ref[2], EI1, EI2, GJ, 0.0, tiny, True
    )
    energy[0] = e
    for p in qd.static(range(11)):
        res[p] = r[p]
        for q in qd.static(range(11)):
            K[p, q] = Km[p, q]


@qd.kernel
def _kernel_transport(
    n0: qd.types.ndarray(), n: qd.types.ndarray(), v: qd.types.ndarray(), tiny: float, out: qd.types.ndarray()
):
    r = func_rod_parallel_transport(qd.Vector([n0[0], n0[1], n0[2]]), qd.Vector([n[0], n[1], n[2]]), tiny) @ qd.Vector(
        [v[0], v[1], v[2]]
    )
    for k in qd.static(range(3)):
        out[k] = r[k]


@qd.kernel
def _kernel_axial(x: qd.types.ndarray(), L: float, EA: float, out: qd.types.ndarray(), T: qd.types.ndarray()):
    x0 = qd.Vector([x[0, 0], x[0, 1], x[0, 2]])
    x1 = qd.Vector([x[1, 0], x[1, 1], x[1, 2]])
    e, g, Tm = func_rod_axial(x0, x1, L, EA, 0.0, 0.0, 2.2e-16, True)
    out[0] = e
    for k in qd.static(range(3)):
        out[1 + k] = g[k]
        for l in qd.static(range(3)):
            T[k, l] = Tm[k, l]


def _unit(v):
    return v / np.linalg.norm(v)


def _transport(n0, n, v):
    out = np.zeros(3)
    _kernel_transport(
        np.ascontiguousarray(n0),
        np.ascontiguousarray(n),
        np.ascontiguousarray(v),
        float(np.finfo(gs.np_float).tiny),
        out,
    )
    return out


def _measures(x, a, L):
    out = np.zeros(3)
    _kernel_measures(np.ascontiguousarray(x), np.ascontiguousarray(a), float(L), float(np.finfo(gs.np_float).tiny), out)
    return out


def _bend_twist(x, a, L, ref, params):
    energy, res, K = np.zeros(1), np.zeros(11), np.zeros((11, 11))
    _kernel_bend_twist(
        np.ascontiguousarray(x),
        np.ascontiguousarray(a),
        float(L),
        np.ascontiguousarray(ref),
        *params,
        float(np.finfo(gs.np_float).tiny),
        energy,
        res,
        K,
    )
    return energy[0], res, K


def _apply_stencil_dofs(x, a, d):
    """Apply an 11-coordinate increment: positions, and twists through transported and rotated axes."""
    x_new = x.copy()
    x_new[0] += d[0:3]
    x_new[1] += d[4:7]
    x_new[2] += d[8:11]
    a_new = np.zeros_like(a)
    for s, theta in ((0, d[3]), (1, d[7])):
        t_old = _unit(x[s + 1] - x[s])
        t_new = _unit(x_new[s + 1] - x_new[s])
        axis = _transport(t_old, t_new, a[s])
        a_new[s] = np.cos(theta) * axis + np.sin(theta) * np.cross(t_new, axis)
    return x_new, a_new


@pytest.mark.required
@pytest.mark.precision("64")
def test_rod_stencil_finite_difference():
    rng = np.random.default_rng(5)
    X = np.array([[0.0, 0.0, 0.0], [0.1, 0.01, 0.0], [0.19, 0.05, 0.02]])
    A = np.zeros((2, 3))
    for s in range(2):
        t = _unit(X[s + 1] - X[s])
        v = np.array([0.0, 0.0, 1.0])
        A[s] = _unit(v - v.dot(t) * t)
    A[1] = np.cos(0.3) * A[1] + np.sin(0.3) * np.cross(_unit(X[2] - X[1]), A[1])
    L = 0.5 * (np.linalg.norm(X[1] - X[0]) + np.linalg.norm(X[2] - X[1]))
    ref = _measures(X, A, L)
    params = (0.01, 0.02, 0.005)
    h = 1e-6
    # Rest: no energy or residual, and the Gauss-Newton tangent is the exact Hessian.
    energy, res, K = _bend_twist(X, A, L, ref, params)
    assert_allclose(energy, 0.0, atol=1e-20)
    assert_allclose(res, 0.0, atol=1e-12)
    K_fd = np.zeros((11, 11))
    for p in range(11):
        d = np.zeros(11)
        d[p] = h
        x_p, a_p = _apply_stencil_dofs(X, A, d)
        x_m, a_m = _apply_stencil_dofs(X, A, -d)
        _, r_p, _ = _bend_twist(x_p, a_p, L, ref, params)
        _, r_m, _ = _bend_twist(x_m, a_m, L, ref, params)
        K_fd[:, p] = (r_p - r_m) / (2.0 * h)
    assert_allclose(K, K_fd, atol=1e-7 * np.abs(K).max(), rtol=0.0)
    assert np.linalg.eigvalsh(K).min() > -1e-9 * np.abs(K).max()
    # Deformed (bent and twisted): the residual is the gradient of the energy.
    x = X + 0.02 * rng.standard_normal(X.shape)
    a = np.zeros_like(A)
    for s in range(2):
        a[s] = _transport(_unit(X[s + 1] - X[s]), _unit(x[s + 1] - x[s]), A[s])
    a[0] = np.cos(0.2) * a[0] + np.sin(0.2) * np.cross(_unit(x[1] - x[0]), a[0])
    energy, res, K = _bend_twist(x, a, L, ref, params)
    res_fd = np.zeros(11)
    for p in range(11):
        d = np.zeros(11)
        d[p] = h
        x_p, a_p = _apply_stencil_dofs(x, a, d)
        x_m, a_m = _apply_stencil_dofs(x, a, -d)
        e_p, _, _ = _bend_twist(x_p, a_p, L, ref, params)
        e_m, _, _ = _bend_twist(x_m, a_m, L, ref, params)
        res_fd[p] = (e_p - e_m) / (2.0 * h)
    assert_allclose(res, res_fd, atol=1e-8 * np.abs(res).max(), rtol=0.0)
    assert np.linalg.eigvalsh(K).min() > -1e-9 * np.abs(K).max()
    # Axial: gradient and tangent.
    x2 = np.array([[0.0, 0.0, 0.0], [0.13, 0.02, -0.01]])
    L2, EA = 0.1, 100.0
    out, T = np.zeros(4), np.zeros((3, 3))
    _kernel_axial(np.ascontiguousarray(x2), L2, EA, out, T)
    g_fd, T_fd = np.zeros(3), np.zeros((3, 3))
    for k in range(3):
        d = np.zeros((2, 3))
        d[1, k] = h
        out_p, out_m = np.zeros(4), np.zeros(4)
        _kernel_axial(np.ascontiguousarray(x2 + d), L2, EA, out_p, np.zeros((3, 3)))
        _kernel_axial(np.ascontiguousarray(x2 - d), L2, EA, out_m, np.zeros((3, 3)))
        g_fd[k] = (out_p[0] - out_m[0]) / (2.0 * h)
        T_fd[:, k] = (out_p[1:] - out_m[1:]) / (2.0 * h)
    assert_allclose(out[1:], g_fd, atol=1e-8 * np.abs(out[1:]).max(), rtol=0.0)
    assert_allclose(T, T_fd, atol=1e-8 * np.abs(T).max(), rtol=0.0)


@pytest.mark.required
@pytest.mark.precision("64")
def test_rod_free_fall(show_viewer):
    dt, g = 0.01, 9.8
    # The free fall is checked to round-off, which needs the linear solves tight (the default adaptive tolerance
    # leaves the truncation error of the first Newton step in the accepted iterate).
    scene = _mochi_scene(show_viewer, dt, gravity=(0.0, 0.0, -g), linear_tolerance_strategy="constant", pcg_abs_tol=0.0)
    rod = scene.add_entity(
        gs.morphs.Rod(points=_straight_rod_points(20, 1.0), radius=0.01, pos=(0.0, 0.0, 1.0), euler=(0.0, 20.0, 0.0)),
        material=gs.materials.Mochi.Rod(E=1e7, nu=0.3, rho=1000.0),
    )
    scene.build()
    assert rod.is_rod
    assert scene.mochi_solver.n_dofs_total == 3 * 21 + 20
    assert_allclose(rod.mass, 1000.0 * np.pi * 0.01**2 * 1.0, tol=1e-9)
    rest = tensor_to_array(rod.get_vertices_position())
    n_steps = 30
    for _ in range(n_steps):
        scene.step()
    pos = tensor_to_array(rod.get_vertices_position())
    assert_allclose(pos[:, 2].mean() - rest[:, 2].mean(), -g * dt * dt * n_steps * (n_steps + 1) / 2, tol=1e-8)
    assert_allclose(tensor_to_array(rod.get_vertices_velocity())[:, 2], -g * dt * n_steps, tol=1e-6)
    assert_allclose(pos - pos.mean(axis=0), rest - rest.mean(axis=0), tol=1e-5)


@pytest.mark.required
def test_rod_degenerate_segment(show_viewer):
    n_segments, length = 8, 0.8
    scene = _mochi_scene(show_viewer, 0.01, n_newton_iterations=8)
    rod = scene.add_entity(
        gs.morphs.Rod(points=_straight_rod_points(n_segments, length), radius=0.01, pos=(0.0, 0.0, 1.0)),
        material=gs.materials.Mochi.Rod(E=1e7, nu=0.3, rho=1000.0),
    )
    scene.build()
    rest = tensor_to_array(rod.get_vertices_position())
    # Collapsing the root segment onto a point leaves its tangent, its parallel transport and its curvature binormal
    # defined only through the smallest-normal floor of the rod stencils.
    rod.set_vertices_target([0, 1], np.stack([rest[0], rest[0]]))
    for _ in range(10):
        scene.step()
    assert np.isfinite(tensor_to_array(rod.get_vertices_position())).all()
    assert np.isfinite(tensor_to_array(rod.get_vertices_velocity())).all()


@pytest.mark.precision("64")
def test_rod_cantilever(show_viewer):
    E, rho, radius, length, n_segments = 1e10, 1000.0, 0.01, 1.0, 40
    scene = _mochi_scene(show_viewer, 0.01, n_newton_iterations=8)
    rod = scene.add_entity(
        gs.morphs.Rod(points=_straight_rod_points(n_segments, length), radius=radius, pos=(0.0, 0.0, 1.0)),
        material=gs.materials.Mochi.Rod(E=E, nu=0.3, rho=rho, mass_damping=5.0),
    )
    scene.build()
    rest = tensor_to_array(rod.get_vertices_position())
    # Clamp: the first two nodes fix the position and the direction of the root.
    rod.set_vertices_fixed([0, 1])
    for _ in range(400):
        scene.step()
    pos = tensor_to_array(rod.get_vertices_position())
    assert_allclose(pos[:2], rest[:2], tol=1e-12)
    assert_allclose(rod.get_vertices_velocity(), 0.0, atol=1e-4)
    # Small-deflection Euler-Bernoulli cantilever under its own weight: w L^4 / (8 E I), with the clamp at the second
    # node shortening the free length by one segment.
    params = rod.material.resolve(radius)
    w = params["linear_density"] * 9.8
    free_length = length * (n_segments - 1) / n_segments
    deflection = rest[-1, 2] - pos[-1, 2]
    # The discrete curvature converges to the beam theory at first order in the segment length (10% at 20 segments,
    # 5% at 40).
    assert_allclose(deflection, w * free_length**4 / (8.0 * params["flexural_stiffness"]), rtol=0.08)


def _penalty_rest_distance(k, h, thr, pressure):
    """Distance at which the smoothed penalty k * phi(d) * phi'(d) balances the given pressure (bisection)."""

    def force(d):
        xi = -d - (h - thr)
        y = xi / h
        if abs(xi) <= h:
            phi = 3.0 * h / 16.0 + (0.5 + 3.0 * y / 8.0 - y**3 / 16.0) * xi
            dphi = min(max(0.5 + 0.75 * y - 0.25 * y**3, 0.0), 1.0)
        else:
            phi = max(0.0, xi)
            dphi = float(xi > 0.0)
        return k * phi * dphi

    lo, hi = thr - 2.0 * h, thr
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if force(mid) > pressure:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


@pytest.mark.precision("64")
def test_rod_on_plane(show_viewer):
    n_segments, length, radius, threshold = 20, 1.0, 0.01, 1e-3
    penalty_coefficient, half_distance = 1e9, 5e-3
    scene = _mochi_scene(show_viewer, 0.01, n_newton_iterations=8, newton_abs_tol=1e-9, newton_rel_tol=1e-12)
    scene.add_entity(
        gs.morphs.Plane(),
        material=gs.materials.Mochi.Rigid(
            penalty_coefficient=penalty_coefficient,
            penalty_smoothing_half_distance=half_distance,
            penalty_threshold=threshold,
        ),
    )
    rod = scene.add_entity(
        gs.morphs.Rod(points=_straight_rod_points(n_segments, length), radius=radius, pos=(0.0, 0.0, 0.02)),
        material=gs.materials.Mochi.Rod(
            E=1e7,
            nu=0.3,
            rho=1000.0,
            penalty_coefficient=penalty_coefficient,
            penalty_smoothing_half_distance=half_distance,
            penalty_threshold=threshold,
        ),
    )
    scene.build()
    rest = tensor_to_array(rod.get_vertices_position())
    for _ in range(150):
        scene.step()
    pos = tensor_to_array(rod.get_vertices_position())
    # Contact samples sit on the centerline with a 1-D measure (no radius offset): the straight rod comes to rest where
    # the smoothed penalty per unit length balances its weight per unit length.
    weight_per_length = rod.material.resolve(radius)["linear_density"] * 9.8
    z_rest = _penalty_rest_distance(penalty_coefficient, half_distance, threshold, weight_per_length)
    assert threshold - 2.0 * half_distance < z_rest < threshold
    assert_allclose(pos[:, 2], z_rest, atol=1e-6)
    assert_allclose(pos[:, :2], rest[:, :2], atol=1e-4)
    assert_allclose(rod.get_vertices_velocity(), 0.0, atol=1e-5)
    assert_allclose(
        tensor_to_array(rod.get_vertices_contact_force()).sum(axis=0), (0.0, 0.0, rod.mass * 9.8), atol=1e-6
    )


@pytest.mark.precision("64")
def test_rigid_ball_on_ropes(show_viewer):
    n_segments, length, rope_radius, ball_radius, gap = 40, 1.0, 0.02, 0.05, 0.03
    # The force balance is checked to 1e-3 N, below what the default Newton tolerance guarantees for the ball.
    scene = _mochi_scene(show_viewer, 0.01, n_newton_iterations=8, newton_abs_tol=1e-6)
    ropes = []
    for y in (-gap, gap):
        ropes.append(
            scene.add_entity(
                gs.morphs.Rod(points=_straight_rod_points(n_segments, length), radius=rope_radius, pos=(0.0, y, 0.5)),
                material=gs.materials.Mochi.Rod(E=1e9, nu=0.3, rho=1000.0, mass_damping=2.0),
            )
        )
    ball = scene.add_entity(
        gs.morphs.Sphere(radius=ball_radius, pos=(0.5, 0.0, 0.5 + ball_radius + rope_radius + 0.01)),
        material=gs.materials.Mochi.Rigid(rho=500.0),
    )
    scene.build()
    for rope in ropes:
        rope.set_vertices_fixed([0, n_segments])
    for _ in range(200):
        scene.step()
    ball_pos = tensor_to_array(ball.get_pos())
    # The ball rests in the cradle formed by the two ropes, touching the collider spheres carried by the rope nodes.
    for rope in ropes:
        rope_pos = tensor_to_array(rope.get_vertices_position())
        i_node = np.argmin(np.abs(rope_pos[:, 0] - ball_pos[0]))
        dist = np.linalg.norm(ball_pos - rope_pos[i_node])
        assert ball_radius < dist < ball_radius + rope_radius
        # The ropes sag under the ball and stay pinned at their ends.
        assert rope_pos[i_node, 2] < 0.5 - 1e-3
        assert_allclose(rope_pos[[0, n_segments], 2], 0.5, tol=1e-12)
    assert_allclose(ball_pos[:2], (0.5, 0.0), atol=1e-3)
    ball_vel = tensor_to_array(ball.get_dofs_velocity())
    assert_allclose(ball_vel[:3], 0.0, atol=1e-4)
    assert_allclose(ball_vel[3:], 0.0, atol=1e-2)
    ball_mass = float(tensor_to_array(ball.get_mass()))
    assert_allclose(tensor_to_array(ball.get_links_net_contact_force())[0], (0.0, 0.0, ball_mass * 9.8), atol=1e-3)


@pytest.mark.required
@pytest.mark.precision("64")
def test_rod_band_preconditioner_exact_in_contact(show_viewer):
    # The banded per-rod Cholesky includes the contact blocks on the rod's vertices: in a scene with no other
    # coupling the preconditioner is the exact inverse of the rod's Hessian and every Newton solve converges in one
    # conjugate-gradient iteration, resting on the plane included (without the contact blocks a rod pressed into a
    # collider costs 3-5 iterations per solve).
    scene = _mochi_scene(show_viewer, 0.01, n_newton_iterations=8)
    scene.add_entity(gs.morphs.Plane(), material=gs.materials.Mochi.Rigid())
    rod = scene.add_entity(
        gs.morphs.Rod(points=_straight_rod_points(20, 1.0), radius=0.01, pos=(0.0, 0.0, 0.05)),
        material=gs.materials.Mochi.Rod(E=1e7, nu=0.3, rho=1000.0),
    )
    scene.build()
    in_contact_steps = 0
    for _ in range(60):
        scene.step()
        info = scene.mochi_solver.get_convergence_info()
        n_iter = int(tensor_to_array(info["n_iter"]).ravel()[0])
        n_pcg = int(tensor_to_array(info["n_pcg_iter"]).ravel()[0])
        assert n_pcg <= n_iter + 1
        if len(rod.get_contacts()["position"]) > 0:
            in_contact_steps += 1
    assert in_contact_steps > 30

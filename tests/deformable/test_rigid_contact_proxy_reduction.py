"""Pinned CGQ dense-reference gates for reduced rigid-proxy KKT algebra."""

from __future__ import annotations

import numpy as np


def _skew(value):
    x, y, z = np.asarray(value, dtype=np.float64)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def _so3_exp(rotation):
    rotation = np.asarray(rotation, dtype=np.float64)
    angle = float(np.linalg.norm(rotation))
    hat = _skew(rotation)
    if angle < 1e-8:
        return np.eye(3) + hat + 0.5 * hat @ hat
    return np.eye(3) + np.sin(angle) / angle * hat + (1.0 - np.cos(angle)) / (angle * angle) * (hat @ hat)


def _so3_log(rotation):
    cosine = np.clip(0.5 * (np.trace(rotation) - 1.0), -1.0, 1.0)
    angle = float(np.arccos(cosine))
    vee = np.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ]
    )
    if angle < 1e-8:
        return 0.5 * vee
    return 0.5 * angle / np.sin(angle) * vee


def _left_jacobian_inverse(rotation):
    theta_sq = float(np.dot(rotation, rotation))
    hat = _skew(rotation)
    if theta_sq < 1e-10:
        coefficient = 1.0 / 12.0 + theta_sq / 720.0
    else:
        theta = np.sqrt(theta_sq)
        half = 0.5 * theta
        coefficient = (1.0 - half / np.tan(half)) / theta_sq
    return np.eye(3) - 0.5 * hat + coefficient * hat @ hat


def _pose_constraint(
    proxy_position,
    proxy_rotation,
    target_position,
    target_rotation,
):
    return np.concatenate(
        (
            proxy_position - target_position,
            _so3_log(proxy_rotation @ target_rotation.T),
        )
    )


def test_pose_constraint_jacobian_matches_finite_difference():
    rng = np.random.default_rng(7163)
    n_q = 5
    target_position = rng.normal(size=3)
    target_rotation = _so3_exp(rng.normal(size=3) * 0.4)
    proxy_position = target_position + rng.normal(size=3) * 0.03
    proxy_rotation = _so3_exp(rng.normal(size=3) * 0.2) @ target_rotation
    link_jacobian = rng.normal(size=(6, n_q))
    constraint = _pose_constraint(
        proxy_position,
        proxy_rotation,
        target_position,
        target_rotation,
    )
    proxy_jacobian = np.eye(6)
    proxy_jacobian[3:, 3:] = _left_jacobian_inverse(constraint[3:])
    target_jacobian = link_jacobian.copy()
    target_jacobian[3:, :] = _left_jacobian_inverse(-constraint[3:]) @ target_jacobian[3:, :]
    analytic = np.column_stack((proxy_jacobian, -target_jacobian))

    delta = 1e-6
    finite_difference = np.zeros_like(analytic)
    for column in range(6 + n_q):
        plus_proxy_position = proxy_position.copy()
        minus_proxy_position = proxy_position.copy()
        plus_proxy_rotation = proxy_rotation.copy()
        minus_proxy_rotation = proxy_rotation.copy()
        plus_target_position = target_position.copy()
        minus_target_position = target_position.copy()
        plus_target_rotation = target_rotation.copy()
        minus_target_rotation = target_rotation.copy()
        if column < 3:
            plus_proxy_position[column] += delta
            minus_proxy_position[column] -= delta
        elif column < 6:
            axis = np.zeros(3)
            axis[column - 3] = delta
            plus_proxy_rotation = _so3_exp(axis) @ proxy_rotation
            minus_proxy_rotation = _so3_exp(-axis) @ proxy_rotation
        else:
            direction = np.zeros(n_q)
            direction[column - 6] = delta
            target_twist = link_jacobian @ direction
            plus_target_position += target_twist[:3]
            minus_target_position -= target_twist[:3]
            plus_target_rotation = _so3_exp(target_twist[3:]) @ target_rotation
            minus_target_rotation = _so3_exp(-target_twist[3:]) @ target_rotation
        plus = _pose_constraint(
            plus_proxy_position,
            plus_proxy_rotation,
            plus_target_position,
            plus_target_rotation,
        )
        minus = _pose_constraint(
            minus_proxy_position,
            minus_proxy_rotation,
            minus_target_position,
            minus_target_rotation,
        )
        finite_difference[:, column] = (plus - minus) / (2.0 * delta)

    np.testing.assert_allclose(
        analytic,
        finite_difference,
        rtol=5e-7,
        atol=5e-8,
    )


def _random_kkt(seed: int = 93):
    rng = np.random.default_rng(seed)
    n_other = 5
    n_proxy = 12
    n_q = 7
    n_full = n_other + n_proxy + n_q
    factor = rng.normal(size=(n_full, n_full))
    hessian = factor.T @ factor + 0.5 * np.eye(n_full)
    gradient = rng.normal(size=n_full)
    proxy_jacobian = np.zeros((n_proxy, n_proxy))
    for pair in range(n_proxy // 6):
        block = np.eye(6)
        block[3:, 3:] += rng.normal(size=(3, 3)) * 0.03
        begin = pair * 6
        proxy_jacobian[begin : begin + 6, begin : begin + 6] = block
    link_jacobian = rng.normal(size=(n_proxy, n_q))
    constraint = rng.normal(size=n_proxy) * 0.02
    full_jacobian = np.column_stack(
        (
            np.zeros((n_proxy, n_other)),
            proxy_jacobian,
            -link_jacobian,
        )
    )
    return (
        hessian,
        gradient,
        proxy_jacobian,
        link_jacobian,
        constraint,
        full_jacobian,
        n_other,
    )


def _reduce(
    hessian,
    gradient,
    proxy_jacobian,
    link_jacobian,
    constraint,
    n_other,
):
    n_proxy = proxy_jacobian.shape[0]
    n_q = link_jacobian.shape[1]
    inverse = np.linalg.inv(proxy_jacobian)
    particular = np.zeros(hessian.shape[0])
    particular[n_other : n_other + n_proxy] = -inverse @ constraint
    prolongation = np.zeros((hessian.shape[0], n_other + n_q))
    prolongation[:n_other, :n_other] = np.eye(n_other)
    prolongation[n_other : n_other + n_proxy, n_other:] = inverse @ link_jacobian
    prolongation[n_other + n_proxy :, n_other:] = np.eye(n_q)
    reduced_hessian = prolongation.T @ hessian @ prolongation
    reduced_gradient = prolongation.T @ (gradient + hessian @ particular)
    return particular, prolongation, reduced_hessian, reduced_gradient


def _solve_dense_kkt(hessian, gradient, jacobian, constraint):
    system = np.block(
        [
            [hessian, jacobian.T],
            [jacobian, np.zeros((len(constraint), len(constraint)))],
        ]
    )
    solution = np.linalg.solve(
        system,
        -np.concatenate((gradient, constraint)),
    )
    return solution[: hessian.shape[0]]


def test_nullspace_reduction_matches_dense_kkt():
    (
        hessian,
        gradient,
        proxy_jacobian,
        link_jacobian,
        constraint,
        full_jacobian,
        n_other,
    ) = _random_kkt()
    particular, prolongation, reduced_hessian, reduced_gradient = _reduce(
        hessian,
        gradient,
        proxy_jacobian,
        link_jacobian,
        constraint,
        n_other,
    )

    np.testing.assert_allclose(full_jacobian @ prolongation, 0.0, atol=2e-14)
    np.testing.assert_allclose(
        full_jacobian @ particular,
        -constraint,
        atol=2e-14,
    )
    np.testing.assert_allclose(
        reduced_hessian,
        reduced_hessian.T,
        atol=2e-13,
    )
    assert np.linalg.eigvalsh(reduced_hessian).min() > 0.0

    reduced_step = np.linalg.solve(reduced_hessian, -reduced_gradient)
    full_step = particular + prolongation @ reduced_step
    dense_step = _solve_dense_kkt(
        hessian,
        gradient,
        full_jacobian,
        constraint,
    )
    np.testing.assert_allclose(full_step, dense_step, rtol=2e-12, atol=2e-12)


def test_nonzero_residual_rhs_requires_h_particular():
    (
        hessian,
        gradient,
        proxy_jacobian,
        link_jacobian,
        constraint,
        full_jacobian,
        n_other,
    ) = _random_kkt(seed=191)
    particular, prolongation, reduced_hessian, _ = _reduce(
        hessian,
        gradient,
        proxy_jacobian,
        link_jacobian,
        constraint,
        n_other,
    )
    wrong_gradient = prolongation.T @ gradient
    wrong_step = particular + prolongation @ np.linalg.solve(
        reduced_hessian,
        -wrong_gradient,
    )
    dense_step = _solve_dense_kkt(
        hessian,
        gradient,
        full_jacobian,
        constraint,
    )
    assert np.linalg.norm(wrong_step - dense_step) > 1e-3


def test_elastic_slack_reduction_matches_full_augmented_system():
    (
        hessian,
        gradient,
        proxy_jacobian,
        link_jacobian,
        constraint,
        full_jacobian,
        n_other,
    ) = _random_kkt(seed=731)
    particular, prolongation, _, _ = _reduce(
        hessian,
        gradient,
        proxy_jacobian,
        link_jacobian,
        constraint,
        n_other,
    )
    rng = np.random.default_rng(882)
    factor = rng.normal(size=proxy_jacobian.shape)
    metric = factor.T @ factor + 0.2 * np.eye(proxy_jacobian.shape[0])
    dual = rng.normal(size=proxy_jacobian.shape[0])
    normal_map = np.zeros((hessian.shape[0], proxy_jacobian.shape[0]))
    normal_map[n_other : n_other + proxy_jacobian.shape[0]] = np.linalg.inv(proxy_jacobian)
    transform = np.column_stack((prolongation, normal_map))
    metric_reduced = np.zeros((transform.shape[1], transform.shape[1]))
    metric_reduced[-len(dual) :, -len(dual) :] = metric
    dual_reduced = np.zeros(transform.shape[1])
    dual_reduced[-len(dual) :] = dual
    reduced_hessian = transform.T @ hessian @ transform + metric_reduced
    reduced_gradient = transform.T @ (gradient + hessian @ particular) + dual_reduced

    reduced_step = np.linalg.solve(reduced_hessian, -reduced_gradient)
    full_step = particular + transform @ reduced_step
    augmented_hessian = hessian + full_jacobian.T @ metric @ full_jacobian
    augmented_gradient = gradient + full_jacobian.T @ (dual + metric @ constraint)
    dense_step = np.linalg.solve(augmented_hessian, -augmented_gradient)

    np.testing.assert_allclose(
        full_jacobian @ normal_map,
        np.eye(len(dual)),
        atol=2e-14,
    )
    np.testing.assert_allclose(full_step, dense_step, rtol=3e-12, atol=3e-12)
    np.testing.assert_allclose(
        constraint + full_jacobian @ full_step,
        reduced_step[-len(dual) :],
        atol=3e-13,
    )


def _two_revolute_fk(configuration, lengths):
    first, second = configuration
    first_rotation = _so3_exp(np.array([0.0, 0.0, first]))
    translation = first_rotation @ np.array([lengths[0], 0.0, 0.0])
    rotation = first_rotation @ _so3_exp(np.array([0.0, 0.0, second]))
    return translation, rotation


def _two_revolute_point(configuration, lengths, local_point):
    translation, rotation = _two_revolute_fk(configuration, lengths)
    return translation + rotation @ local_point


def _two_revolute_tangent(configuration, direction, lengths):
    first_rotation = _so3_exp(np.array([0.0, 0.0, configuration[0]]))
    link_arm = first_rotation @ np.array([lengths[0], 0.0, 0.0])
    axis = np.array([0.0, 0.0, 1.0])
    linear = direction[0] * np.cross(axis, link_arm)
    angular = axis * (direction[0] + direction[1])
    return linear, angular


def _two_revolute_curvature_bound(direction, lengths, local_point):
    first, second = np.abs(direction)
    reach = float(np.linalg.norm(local_point))
    speed = second * reach
    reach = lengths[0] + reach
    speed += first * reach
    return 2.0 * (first + second) * speed


def _proxy_screw_point(
    translation,
    rotation,
    linear,
    angular,
    local_point,
    alpha,
):
    lever = rotation @ local_point
    return translation + alpha * linear + _so3_exp(alpha * angular) @ lever


def test_two_revolute_screw_fk_defect_bound_is_conservative():
    rng = np.random.default_rng(4419)
    alphas = np.linspace(0.0, 1.0, 33)
    for _ in range(400):
        configuration = rng.uniform(-1.5, 1.5, size=2)
        direction = rng.uniform(-1.2, 1.2, size=2)
        lengths = rng.uniform(0.1, 0.8, size=2)
        local_point = rng.normal(size=3)
        local_point *= rng.uniform(0.01, lengths[1]) / np.linalg.norm(local_point)

        translation, rotation = _two_revolute_fk(
            configuration,
            lengths,
        )
        linear, angular = _two_revolute_tangent(
            configuration,
            direction,
            lengths,
        )
        forest_bound = _two_revolute_curvature_bound(
            direction,
            lengths,
            local_point,
        )
        proxy_bound = float(np.dot(angular, angular) * np.linalg.norm(local_point))
        total_bound = forest_bound + proxy_bound

        for alpha in alphas:
            proxy_point = _proxy_screw_point(
                translation,
                rotation,
                linear,
                angular,
                local_point,
                alpha,
            )
            mechanism_point = _two_revolute_point(
                configuration + alpha * direction,
                lengths,
                local_point,
            )
            mismatch = np.linalg.norm(proxy_point - mechanism_point)
            assert mismatch <= 0.5 * total_bound * alpha * alpha + 2e-12

"""Generate the mochi reference trajectories. Run with the superdex environment:

    SUPERDEX_PRECISION=double python -m tests.mochi.reference.gen_reference [case ...]

Scenes are Y-up (mochi convention); positions are stored as recorded.
"""

import sys
from pathlib import Path

import numpy as np

CUBE_COORDINATES = np.array(
    [
        [-1, -1, -1],
        [1, -1, -1],
        [1, 1, -1],
        [-1, 1, -1],
        [-1, -1, 1],
        [1, -1, 1],
        [1, 1, 1],
        [-1, 1, 1],
    ],
    dtype=np.float64,
)
CUBE_CONNECTIVITY = np.array(
    [
        [0, 2, 1],
        [0, 3, 2],
        [4, 5, 6],
        [4, 6, 7],
        [0, 4, 7],
        [0, 7, 3],
        [1, 2, 6],
        [1, 6, 5],
        [0, 1, 5],
        [0, 5, 4],
        [2, 3, 7],
        [2, 7, 6],
    ],
    dtype=np.int32,
)


def _cube_shape(physics, size):
    coords = (0.5 * size * CUBE_COORDINATES).astype(np.float64).reshape((-1,))
    return physics.create_tri_mesh_shape(coordinates=coords, connectivity=CUBE_CONNECTIVITY.reshape((-1,)))


def _contact_params(physics, friction=0.5, normal_damping=0.0):
    params = physics.ContactParams()
    params.coulomb_friction_coefficient = friction
    params.normal_viscous_damping_coefficient = normal_damping
    return params


def _set_solver(
    physics, scene, *, max_iter, integrator="backward_euler", implicit_normal_force=False, abs_tol=None, rel_tol=None
):
    params = scene.get_solver_params()
    params.non_linear_solver.max_iter = max_iter
    if abs_tol is not None:
        params.non_linear_solver.abs_tol = abs_tol
    if rel_tol is not None:
        params.non_linear_solver.rel_tol = rel_tol
    params.integration_method = (
        physics.IntegrationMethod.BDF2 if integrator == "bdf2" else physics.IntegrationMethod.BACKWARD_EULER
    )
    params.experimental_eval.implicit_normal_force_for_dissipation = implicit_normal_force
    scene.set_solver_params(params)


def _actor_state(actor):
    transform = actor.get_center_of_mass_transform()
    return (
        np.array(transform.translation.tolist() if hasattr(transform.translation, "tolist") else transform.translation),
        np.array(actor.get_linear_velocity()),
        np.array(actor.get_angular_velocity()),
    )


def case_box_drop_plane(physics):
    """Unit cube (density 1000) dropped from height 1 onto a plane, g = 10, dt = 0.05, backward Euler, 40 steps."""
    dt, n_steps = 0.05, 40
    scene = physics.create_scene("box_drop_plane")
    scene.set_gravity([0.0, -10.0, 0.0])
    _set_solver(physics, scene, max_iter=4)
    plane = scene.create_rigid_actor(
        name="plane",
        shape=physics.create_plane_shape(normal=[0.0, 1.0, 0.0], distance=0.0),
        is_static=True,
        contact=_contact_params(physics),
    )
    box = scene.create_rigid_actor(
        name="box",
        shape=_cube_shape(physics, 1.0),
        density=1000.0,
        world_from_local=physics.TransformRT(translation=[0.0, 1.0, 0.0]),
        collider_type=physics.ColliderType.BOX,
        contact=_contact_params(physics),
    )
    box.register_query(physics.QueryType.TOTAL_CONTACT_FORCE)
    pos, vel, force = [], [], []
    for _ in range(n_steps):
        scene.step(dt)
        p, v, _ = _actor_state(box)
        pos.append(p)
        vel.append(v)
        force.append(np.array(box.get_contact_force_world()))
    return {"dt": dt, "gravity": 10.0, "pos": np.array(pos), "vel": np.array(vel), "contact_force": np.array(force)}


def case_two_boxes_stack(physics):
    """Cube of side 1 resting on the plane, cube of side 0.5 dropped from 0.05 above it, dt = 0.05, 40 steps."""
    dt, n_steps = 0.05, 40
    scene = physics.create_scene("two_boxes_stack")
    scene.set_gravity([0.0, -10.0, 0.0])
    _set_solver(physics, scene, max_iter=4)
    scene.create_rigid_actor(
        name="plane",
        shape=physics.create_plane_shape(normal=[0.0, 1.0, 0.0], distance=0.0),
        is_static=True,
        contact=_contact_params(physics),
    )
    bottom = scene.create_rigid_actor(
        name="bottom",
        shape=_cube_shape(physics, 1.0),
        density=1000.0,
        world_from_local=physics.TransformRT(translation=[0.0, 0.5, 0.0]),
        collider_type=physics.ColliderType.BOX,
        contact=_contact_params(physics),
    )
    top = scene.create_rigid_actor(
        name="top",
        shape=_cube_shape(physics, 0.5),
        density=1000.0,
        world_from_local=physics.TransformRT(translation=[0.0, 1.3, 0.0]),
        collider_type=physics.ColliderType.BOX,
        contact=_contact_params(physics),
    )
    top.register_query(physics.QueryType.TOTAL_CONTACT_FORCE)
    bottom.register_query(physics.QueryType.TOTAL_CONTACT_FORCE)
    pos_bottom, pos_top, force_top, force_bottom = [], [], [], []
    for _ in range(n_steps):
        scene.step(dt)
        pos_bottom.append(_actor_state(bottom)[0])
        pos_top.append(_actor_state(top)[0])
        force_top.append(np.array(top.get_contact_force_world()))
        force_bottom.append(np.array(bottom.get_contact_force_world()))
    return {
        "dt": dt,
        "pos_bottom": np.array(pos_bottom),
        "pos_top": np.array(pos_top),
        "contact_force_top": np.array(force_top),
        "contact_force_bottom": np.array(force_bottom),
    }


def case_box_slide_friction(physics):
    """Flat box (0.4 x 0.05 x 0.4) resting on the plane, launched at 2 m/s along x, mu = 0.5, dt = 0.01."""
    dt, n_settle, n_steps = 0.01, 20, 60
    scene = physics.create_scene("box_slide_friction")
    scene.set_gravity([0.0, -10.0, 0.0])
    _set_solver(physics, scene, max_iter=12)
    scene.create_rigid_actor(
        name="plane",
        shape=physics.create_plane_shape(normal=[0.0, 1.0, 0.0], distance=0.0),
        is_static=True,
        contact=_contact_params(physics),
    )
    coords = (CUBE_COORDINATES * np.array([0.2, 0.025, 0.2])).reshape((-1,))
    shape = physics.create_tri_mesh_shape(coordinates=coords, connectivity=CUBE_CONNECTIVITY.reshape((-1,)))
    box = scene.create_rigid_actor(
        name="box",
        shape=shape,
        density=1000.0,
        world_from_local=physics.TransformRT(translation=[0.0, 0.025, 0.0]),
        collider_type=physics.ColliderType.BOX,
        contact=_contact_params(physics),
    )
    for _ in range(n_settle):
        scene.step(dt)
    box.set_velocity([2.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    pos, vel = [], []
    for _ in range(n_steps):
        scene.step(dt)
        p, v, _ = _actor_state(box)
        pos.append(p)
        vel.append(v)
    return {"dt": dt, "pos": np.array(pos), "vel": np.array(vel)}


def case_double_pendulum(physics):
    """Two 1 m links on revolute joints about x hanging from a fixed base, each carrying a 0.1 m cube (density 1000)
    at its lower end, released from (0.6, -0.3) rad; g = 10, dt = 0.01, backward Euler, 8 Newton iterations."""
    dt, n_steps, length, cube = 0.01, 200, 1.0, 0.1
    scene = physics.create_scene("double_pendulum")
    scene.set_gravity([0.0, -10.0, 0.0])
    _set_solver(physics, scene, max_iter=8, abs_tol=1e-10, rel_tol=1e-12)
    coords = (0.5 * cube * CUBE_COORDINATES + np.array([0.0, -length, 0.0])).reshape((-1,))
    cube_shape = physics.create_tri_mesh_shape(coordinates=coords, connectivity=CUBE_CONNECTIVITY.reshape((-1,)))
    base_shape = physics.create_tri_mesh_shape(
        coordinates=(0.01 * CUBE_COORDINATES).reshape((-1,)), connectivity=CUBE_CONNECTIVITY.reshape((-1,))
    )
    identity = physics.TransformRT()
    links = [
        physics.ArticulatedLinkParams(
            name="base",
            parent_link=-1,
            parent_joint_from_link=identity,
            shape=base_shape,
            collider_type=physics.ColliderType.NONE,
        ),
        physics.ArticulatedLinkParams(
            name="link_0",
            parent_link=0,
            parent_joint_from_link=identity,
            shape=cube_shape,
            density=1000.0,
            collider_type=physics.ColliderType.NONE,
        ),
        physics.ArticulatedLinkParams(
            name="link_1",
            parent_link=1,
            parent_joint_from_link=identity,
            shape=cube_shape,
            density=1000.0,
            collider_type=physics.ColliderType.NONE,
        ),
    ]
    joints = [
        physics.ArticulatedJointParams(
            name="root", type=physics.ArticulatedJointType.HARD, parent_link_from_joint=identity
        ),
        physics.ArticulatedJointParams(
            name="joint_0",
            type=physics.ArticulatedJointType.REVOLUTE,
            parent_link_from_joint=identity,
            axis=[1.0, 0.0, 0.0],
        ),
        physics.ArticulatedJointParams(
            name="joint_1",
            type=physics.ArticulatedJointType.REVOLUTE,
            parent_link_from_joint=physics.TransformRT(translation=[0.0, -length, 0.0]),
            axis=[1.0, 0.0, 0.0],
        ),
    ]
    actor = scene.create_articulated_actor(
        name="pendulum", world_from_root=physics.TransformRT(translation=[0.0, 3.0, 0.0]), joints=joints, links=links
    )
    actor.set_articulated_pose_from_joints(np.array([0.6, -0.3]))
    angles, velocities = [], []
    pose = np.zeros(2)
    joint_vel = np.zeros(2)
    for _ in range(n_steps):
        scene.step(dt)
        actor.get_articulated_pose(pose)
        actor.get_articulated_joint_velocities(joint_vel)
        angles.append(pose.copy())
        velocities.append(joint_vel.copy())
    return {"dt": dt, "angles": np.array(angles), "velocities": np.array(velocities)}


def structured_cube_tets(n_cells, size, center):
    """Vertices and tetrahedra of a cube split into n_cells^3 cells of 6 tetrahedra each (Freudenthal), Z-up frame."""
    axis = np.linspace(-0.5 * size, 0.5 * size, n_cells + 1)
    grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1) + np.asarray(center)
    verts = grid.reshape((-1, 3))

    def vid(i, j, k):
        return (i * (n_cells + 1) + j) * (n_cells + 1) + k

    tets = []
    for i in range(n_cells):
        for j in range(n_cells):
            for k in range(n_cells):
                corner = np.array([i, j, k])
                for perm in ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)):
                    p1 = corner.copy()
                    p1[perm[0]] += 1
                    p2 = p1.copy()
                    p2[perm[1]] += 1
                    tets.append([vid(*corner), vid(*p1), vid(*p2), vid(*(corner + 1))])
    return verts.astype(np.float64), np.array(tets, dtype=np.int64)


def _soft_world_positions(actor):
    root = actor.get_root_transform()
    translation = np.array(root.translation.tolist() if hasattr(root.translation, "tolist") else root.translation)
    x, y, z, w = np.array(root.rotation.tolist() if hasattr(root.rotation, "tolist") else root.rotation)
    R = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    local = np.array(actor.get_node_positions_local()).reshape((-1, 3))
    return local @ R.T + translation


def case_soft_cube_drop(physics):
    """Soft cube (side 0.2 m, 3x3x3 vertices, 48 tetrahedra, E = 1e5, nu = 0.45, rho = 1000) dropped from height 0.3 m
    onto a plane with friction 0.5, g = 9.8, dt = 1/60, backward Euler, 90 steps, tight Newton tolerances."""
    dt, n_steps = 1.0 / 60.0, 90
    # Built Z-up (Genesis frame), then rotated into the Y-up scene: (x, y, z) -> (x, z, -y).
    verts_zup, tets = structured_cube_tets(2, 0.2, (0.0, 0.0, 0.3))
    verts = np.stack([verts_zup[:, 0], verts_zup[:, 2], -verts_zup[:, 1]], axis=-1)
    scene = physics.create_scene("soft_cube_drop")
    scene.set_gravity([0.0, -9.8, 0.0])
    _set_solver(physics, scene, max_iter=8, abs_tol=1e-10, rel_tol=1e-12)
    scene.create_rigid_actor(
        name="plane",
        shape=physics.create_plane_shape(normal=[0.0, 1.0, 0.0], distance=0.0),
        is_static=True,
        contact=_contact_params(physics),
    )
    material = physics.SoftMaterialParams()
    material.type = physics.SoftMaterialType.NEO_HOOKEAN
    material.neo_hookean.youngs_modulus = 1e5
    material.neo_hookean.poisson_ratio = 0.45
    material.density = 1000.0
    cube = scene.create_soft_actor(
        name="cube",
        shape=physics.create_tet_mesh_shape(coordinates=verts.reshape((-1,)), connectivity=tets.reshape((-1,))),
        material=material,
        contact=_contact_params(physics),
    )
    # mochi recenters the local frame of a soft actor every step: world positions follow from the root transform.
    cube.register_query(physics.QueryType.NODE_POSITIONS)
    positions = []
    for _ in range(n_steps):
        scene.step(dt)
        positions.append(_soft_world_positions(cube))
    return {
        "dt": dt,
        "gravity": 9.8,
        "rest_positions": verts,
        "tets": tets,
        "positions": np.array(positions),
    }


def structured_sheet(n_cells, size, center):
    """Vertices and triangles of a square sheet in the x-y plane split into n_cells^2 cells of 2 triangles (Z-up frame,
    consistently wound with normals along +z)."""
    axis = np.linspace(-0.5 * size, 0.5 * size, n_cells + 1)
    X, Y = np.meshgrid(axis, axis, indexing="ij")
    verts = np.stack([X.reshape(-1), Y.reshape(-1), np.zeros(X.size)], axis=-1) + np.asarray(center)
    faces = []
    for i in range(n_cells):
        for j in range(n_cells):
            a = i * (n_cells + 1) + j
            b = a + 1
            c = a + (n_cells + 1)
            d = c + 1
            faces.append([a, b, d])
            faces.append([a, d, c])
    return verts.astype(np.float64), np.array(faces, dtype=np.int64)


def case_cloth_drop(physics):
    """Square shell (0.4 m, 4x4 cells, E = 1e4, nu = 0.3, rho = 200, t = 1 mm) dropped from 0.3 m onto a plane with
    friction 0.5, g = 9.8, dt = 1/60, backward Euler, 90 steps, tight Newton tolerances."""
    dt, n_steps = 1.0 / 60.0, 90
    verts_zup, faces = structured_sheet(4, 0.4, (0.0, 0.0, 0.3))
    verts = np.stack([verts_zup[:, 0], verts_zup[:, 2], -verts_zup[:, 1]], axis=-1)
    scene = physics.create_scene("cloth_drop")
    scene.set_gravity([0.0, -9.8, 0.0])
    _set_solver(physics, scene, max_iter=8, abs_tol=1e-10, rel_tol=1e-12)
    scene.create_rigid_actor(
        name="plane",
        shape=physics.create_plane_shape(normal=[0.0, 1.0, 0.0], distance=0.0),
        is_static=True,
        contact=_contact_params(physics),
    )
    params = physics.experimental.ShellActorParams()
    params.name = "cloth"
    params.shape = physics.create_tri_mesh_shape(coordinates=verts.reshape((-1,)), connectivity=faces.reshape((-1,)))
    params.material = physics.experimental.shell_material_params_from3d_isotropic(1e4, 0.3, 200.0, 1e-3)
    params.contact = _contact_params(physics)
    params.collider_type = physics.ColliderType.NONE
    cloth = physics.experimental.create_shell_actor(scene, params)
    cloth.register_query(physics.QueryType.NODE_POSITIONS)
    positions = []
    for _ in range(n_steps):
        scene.step(dt)
        positions.append(_soft_world_positions(cloth))
    return {
        "dt": dt,
        "gravity": 9.8,
        "rest_positions": verts,
        "faces": faces,
        "positions": np.array(positions),
    }


CASES = {
    "box_drop_plane": case_box_drop_plane,
    "double_pendulum": case_double_pendulum,
    "two_boxes_stack": case_two_boxes_stack,
    "box_slide_friction": case_box_slide_friction,
    "soft_cube_drop": case_soft_cube_drop,
    "cloth_drop": case_cloth_drop,
}


def generate(name, data_dir):
    from superdex import physics

    physics.initialize(num_worker_threads=0)
    try:
        data = CASES[name](physics)
    finally:
        physics.shutdown()
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    np.savez(data_dir / f"{name}.npz", **data)
    return data


if __name__ == "__main__":
    names = sys.argv[1:] or list(CASES)
    for name in names:
        data = generate(name, Path(__file__).parent / "data")
        print(name, {key: np.asarray(value).shape for key, value in data.items()})

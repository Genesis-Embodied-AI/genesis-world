"""Benchmark scenes shared by Genesis' MochiSolver and the original mochi engine.

Every scene is described once, in Genesis' Z-up world, and built twice: `build_genesis` returns a Genesis scene,
`build_mochi` the same bodies (same meshes, same coordinates, same materials, same solver caps) in the original engine's
Y-up world. Rigid primitives are meshed by Genesis; their meshes are exported to `assets/generated/` by the Genesis
build so that the mochi build carries the very same triangles (run `bench_genesis.py` once before `bench_mochi.py`).
"""

import os
import tempfile

import numpy as np

ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
GENERATED = os.path.join(ASSETS, "generated")

DT = 1.0 / 60.0
GRAVITY = 9.8
DENSITY = 1000.0

# Newton iteration caps: both engines stop on the same residual tolerances, the cap only matters when they do not
# converge (the cloth keeps mochi's own t-shirt example setting).
SCENES = {
    "rigid": {"newton_cap": 20, "description": "sphere and cube dropped onto a static table over the ground"},
    "articulated": {"newton_cap": 20, "description": "double pendulum on a limited rail striking a ball on the ground"},
    "equalities": {
        "newton_cap": 20,
        "description": "two links on a compliant world pivot and a compliant spherical joint",
    },
    "soft_duck": {"newton_cap": 20, "description": "tetrahedral duck (1899 nodes, neo-Hookean) dropped onto a plane"},
    "cloth_tshirt": {
        "newton_cap": 2,
        "description": "t-shirt shell (3593 nodes) with self-contact falling onto a plane",
    },
    "rod_helix": {"newton_cap": 20, "description": "helical spring (129 nodes) hanging from its first node"},
    "franka": {
        "newton_cap": 20,
        "description": "Franka arm on a plane next to a box (Genesis only)",
        "genesis_only": True,
    },
    # Reinforcement-learning shaped scenes (mochi's default Newton budget): a Franka arm descends onto a deformable and
    # closes its fingers; every environment runs the same scripted motion.
    "cloth_arm": {
        "newton_cap": 4,
        "description": "Franka arm pressing a 31x31 cloth with self-contact lying on the ground (Genesis only)",
        "genesis_only": True,
    },
    "soft_gripper": {
        "newton_cap": 4,
        "description": "Franka gripper closing on a tetrahedral cube (Genesis only)",
        "genesis_only": True,
    },
    "rope_arm": {
        "newton_cap": 4,
        "description": "Franka arm pressing a 64-node rope lying on the ground (Genesis only)",
        "genesis_only": True,
    },
}

# --- rigid -----------------------------------------------------------------------------------------------------------
TABLE_SIZE = (1.6, 1.0, 0.1)
TABLE_POS = (0.0, 0.0, 0.75)
SPHERE_RADIUS = 0.2
SPHERE_POS = (-0.5, 0.0, 1.5)
CUBE_SIZE = 0.4
CUBE_POS = (0.5, 0.1, 1.6)

# --- articulated (mochi's double pendulum on a rail; Y-up sizes (x, y, z) -> Z-up (x, z, y) by the symmetry of the boxes)
ROOT_HEIGHT = 0.75
RAIL_SCALE = (0.5, 0.05, 0.05)
CART_SCALE = (0.075, 0.075, 0.075)
ARM_SCALE = (0.03, 0.3, 0.03)
BALL_RADIUS = 0.05
BALL_POS = (0.15, 0.0, 0.05)
BALL_DENSITY = 500.0
RAIL_LIMIT = 0.2
RAIL_LIMIT_STIFFNESS = 250.0
RAIL_LIMIT_DAMPING = 8.8
RAIL_VISCOUS_FRICTION = 0.018
RAIL_ARMATURE = 0.125
SEED_JOINT_VELOCITIES = (0.3, 4.2, 0.0, 0.0, 0.0)

# --- equalities (mochi's constrained double pendulum) -----------------------------------------------------------------
LINK_LENGTH = 0.25
LINK_THICKNESS = 0.025
PIVOT_ANCHOR = (0.0, 0.0, 0.5)
CONSTRAINT_STIFFNESS = 2.5e4
CONSTRAINT_DAMPING = 3.5

# --- deformables --------------------------------------------------------------------------------------------------------
DUCK_POS = (-0.5, 1.0, 1.0)
DUCK_E, DUCK_NU, DUCK_RHO = 1e5, 0.45, 1000.0
TSHIRT_POS = (-0.5, 0.0, 0.1)
TSHIRT_E, TSHIRT_NU, TSHIRT_RHO, TSHIRT_THICKNESS = 1e5, 0.25, 1000.0, 2e-3
TSHIRT_CONTACT_RADIUS = 1.5e-2
HELIX_POS = (0.0, 0.0, 1.0)
ROD_RADIUS = 1e-2
ROD_E = 1e9
ROD_G = 1e9
ROD_RHO = 1000.0

# --- RL scenes: Franka arm + deformable (poses from examples/mochi/articulated_arm.py: the hand starts ~0.30 m above the
# work point and is driven to ~0.13 m, where the closing fingertips reach the ground) -----------------------------------
FRANKA_XML = "xml/franka_emika_panda/panda.xml"
FRANKA_START_QPOS = (-1.0418, 1.3805, 1.5885, -1.7021, -1.3798, 1.5783, 1.4467, 0.04, 0.04)
FRANKA_TARGET_QPOS = (-1.0095, 1.5617, 1.3595, -1.6830, -1.5855, 1.7817, 1.4595, 0.04, 0.04)
FRANKA_MOTORS = np.arange(7)
FRANKA_FINGERS = np.arange(7, 9)
FRANKA_FINGER_FORCE = -1.0
WORK_XY = (0.65, 0.0)
CLOTH_CELLS, CLOTH_SIZE, CLOTH_HEIGHT = 30, 0.3, 0.01
CLOTH_E, CLOTH_NU, CLOTH_RHO, CLOTH_THICKNESS = 2e4, 0.3, 200.0, 1e-3
CLOTH_CONTACT_RADIUS, CLOTH_FRICTION, CLOTH_PENALTY = 5e-3, 0.6, 1e7
SOFT_CUBE_SIZE, SOFT_CUBE_MAXVOLUME = 0.05, 6e-8
SOFT_CUBE_E, SOFT_CUBE_NU, SOFT_CUBE_RHO = 2e4, 0.4, 800.0
ROPE_NODES, ROPE_LENGTH, ROPE_RADIUS = 64, 0.5, 5e-3
ROPE_E, ROPE_G, ROPE_RHO = 1e7, 1e7, 1000.0


def rod_material_params(radius=ROD_RADIUS, E=ROD_E, G=ROD_G, rho=ROD_RHO):
    area = np.pi * radius**2
    second_moment = 0.25 * np.pi * radius**4
    polar_moment = 0.5 * np.pi * radius**4
    return {
        "axial_stiffness": E * area,
        "flexural_stiffness": E * second_moment,
        "torsional_stiffness": G * polar_moment,
        "linear_density": rho * area,
        "linear_rotational_inertia": rho * polar_moment,
    }


def _sheet_obj(directory, n_cells, size):
    """Square shell mesh of n_cells x n_cells cells (two triangles each) centered at the origin, written as an .obj."""
    import trimesh

    axis = np.linspace(-0.5 * size, 0.5 * size, n_cells + 1)
    X, Y = np.meshgrid(axis, axis, indexing="ij")
    verts = np.stack([X.reshape(-1), Y.reshape(-1), np.zeros(X.size)], axis=-1)
    faces = []
    for i in range(n_cells):
        for j in range(n_cells):
            a = i * (n_cells + 1) + j
            faces.append([a, a + 1, a + n_cells + 2])
            faces.append([a, a + n_cells + 2, a + n_cells + 1])
    path = os.path.join(directory, f"sheet_{n_cells}.obj")
    trimesh.Trimesh(vertices=verts, faces=np.array(faces), process=False).export(path)
    return path


def z_up_to_y_up(points):
    """Genesis world (Z up) -> mochi world (Y up): (x, y, z) -> (x, z, -y)."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim == 1:
        return np.array([points[0], points[2], -points[1]])
    return np.stack([points[:, 0], points[:, 2], -points[:, 1]], axis=-1)


def load_tet_mesh(path):
    """Vertices and tetrahedra of a tetgen `.node`/`.ele` pair (1-based indices)."""
    with open(path) as fp:
        n = int(fp.readline().split()[0])
        verts = np.array([[float(x) for x in fp.readline().split()[1:4]] for _ in range(n)])
    with open(os.path.splitext(path)[0] + ".ele") as fp:
        n = int(fp.readline().split()[0])
        tets = np.array([[int(x) for x in fp.readline().split()[1:5]] for _ in range(n)]) - 1
    return verts, tets


def load_obj(path):
    verts, faces = [], []
    with open(path) as fp:
        for line in fp:
            if line.startswith("v "):
                verts.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                faces.append([int(x.split("/")[0]) - 1 for x in line.split()[1:4]])
    return np.array(verts), np.array(faces)


def mochi_options(name):
    """`gs.options.MochiOptions` keyword arguments of a scene."""
    spec = SCENES[name]
    kwargs = {"n_newton_iterations": spec["newton_cap"]}
    if name == "articulated":
        kwargs.update(joint_limit_stiffness=RAIL_LIMIT_STIFFNESS, joint_limit_damping=RAIL_LIMIT_DAMPING)
    elif name == "equalities":
        kwargs.update(equality_stiffness=CONSTRAINT_STIFFNESS, equality_damping=CONSTRAINT_DAMPING)
    elif name == "cloth_tshirt":
        kwargs.update(implicit_normal_force_for_dissipation=True)
    return kwargs


PENDULUM_XML = """
<mujoco model="pendulum_on_rail">
  <worldbody>
    <body name="rail" pos="0 0 {root_z}">
      <geom type="box" size="{rail_hx} {rail_hz} {rail_hy}"/>
      <body name="cart" pos="0 0 {cart_joint_z}">
        <joint name="rail" type="slide" axis="1 0 0" range="{rail_min} {rail_max}" damping="{rail_damping}"
               armature="{rail_armature}"/>
        <geom type="box" size="{cart_hx} {cart_hz} {cart_hy}" pos="0 0 {cart_center_z}"/>
        <body name="upper_arm" pos="0 0 {upper_joint_z}">
          <joint name="upper_swing" type="hinge" axis="0 1 0"/>
          <geom type="box" size="{arm_hx} {arm_hz} {arm_hy}" pos="0 0 {arm_center_z}"/>
          <body name="lower_arm" pos="0 0 {lower_joint_z}">
            <joint name="lower_swing" type="ball"/>
            <geom type="box" size="{arm_hx} {arm_hz} {arm_hy}" pos="0 0 {arm_center_z}"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

LINKS_XML = """
<mujoco model="constrained_links">
  <worldbody>
    <body name="anchor" pos="{ax} {ay} {az}">
      <geom type="sphere" size="0.01" contype="0" conaffinity="0"/>
    </body>
    <body name="link1" pos="{x1} {ay} {az}">
      <freejoint/>
      <geom type="box" size="{hl} {hw} {hw}"/>
    </body>
    <body name="link2" pos="{x2} {ay} {az}">
      <freejoint/>
      <geom type="box" size="{hl} {hw} {hw}"/>
    </body>
  </worldbody>
  <equality>
    <connect body1="link1" body2="anchor" anchor="{neg_hl} 0 0"/>
    <connect body1="link1" body2="link2" anchor="{hl} 0 0"/>
  </equality>
</mujoco>
"""


def pendulum_xml():
    return PENDULUM_XML.format(
        root_z=ROOT_HEIGHT,
        rail_hx=RAIL_SCALE[0] / 2,
        rail_hy=RAIL_SCALE[1] / 2,
        rail_hz=RAIL_SCALE[2] / 2,
        cart_joint_z=-RAIL_SCALE[1] / 2,
        rail_min=-RAIL_LIMIT,
        rail_max=RAIL_LIMIT,
        rail_damping=RAIL_VISCOUS_FRICTION,
        rail_armature=RAIL_ARMATURE,
        cart_hx=CART_SCALE[0] / 2,
        cart_hy=CART_SCALE[1] / 2,
        cart_hz=CART_SCALE[2] / 2,
        cart_center_z=-CART_SCALE[1] / 2,
        upper_joint_z=-CART_SCALE[1],
        arm_hx=ARM_SCALE[0] / 2,
        arm_hy=ARM_SCALE[1] / 2,
        arm_hz=ARM_SCALE[2] / 2,
        arm_center_z=-ARM_SCALE[1] / 2,
        lower_joint_z=-ARM_SCALE[1],
    )


def links_xml():
    return LINKS_XML.format(
        ax=PIVOT_ANCHOR[0],
        ay=PIVOT_ANCHOR[1],
        az=PIVOT_ANCHOR[2],
        x1=PIVOT_ANCHOR[0] + LINK_LENGTH / 2,
        x2=PIVOT_ANCHOR[0] + 1.5 * LINK_LENGTH,
        hl=LINK_LENGTH / 2,
        neg_hl=-LINK_LENGTH / 2,
        hw=LINK_THICKNESS / 2,
    )


def _write(directory, name, text):
    path = os.path.join(directory, name)
    with open(path, "w") as fp:
        fp.write(text)
    return path


def _export_mesh(name, entity):
    """Save the collision mesh of a Genesis primitive (link frame) for the mochi build."""
    os.makedirs(GENERATED, exist_ok=True)
    geom = entity.geoms[0]
    np.savez(
        os.path.join(GENERATED, f"{name}.npz"), verts=np.asarray(geom.init_verts), faces=np.asarray(geom.init_faces)
    )


def _load_generated_mesh(name):
    path = os.path.join(GENERATED, f"{name}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} is missing: run bench_genesis.py on this scene first to export the meshes.")
    data = np.load(path)
    return data["verts"], data["faces"]


def _as_points(tensor):
    """(n, 3) numpy view of a Genesis position tensor of any batch layout."""
    if hasattr(tensor, "cpu"):
        tensor = tensor.cpu().numpy()
    return np.asarray(tensor, dtype=np.float64).reshape(-1, 3)


# --- Genesis -----------------------------------------------------------------------------------------------------------


def build_genesis(name, n_envs=1, show_viewer=False, **mochi_kwargs):
    """Build a benchmark scene in Genesis. Returns (scene, probe) where probe() gives one scalar of the state (the
    height of the moving body) for sanity checks and trajectory comparisons."""
    import genesis as gs

    spec = SCENES[name]
    kwargs = mochi_options(name)
    kwargs.update(mochi_kwargs)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=DT, gravity=(0.0, 0.0, -GRAVITY)),
        mochi_options=gs.options.MochiOptions(**kwargs),
        show_viewer=show_viewer,
    )
    post_build = []
    exports = []
    if name == "rigid":
        scene.add_entity(gs.morphs.Plane(), material=gs.materials.Mochi.Rigid())
        table = scene.add_entity(
            gs.morphs.Box(size=TABLE_SIZE, pos=TABLE_POS, fixed=True), material=gs.materials.Mochi.Rigid()
        )
        sphere = scene.add_entity(
            gs.morphs.Sphere(radius=SPHERE_RADIUS, pos=SPHERE_POS), material=gs.materials.Mochi.Rigid(rho=DENSITY)
        )
        cube = scene.add_entity(
            gs.morphs.Box(size=(CUBE_SIZE,) * 3, pos=CUBE_POS), material=gs.materials.Mochi.Rigid(rho=DENSITY)
        )
        exports = [("rigid_table", table), ("rigid_sphere", sphere), ("rigid_cube", cube)]

        def probe():
            return float(_as_points(cube.get_pos())[0, 2])
    elif name == "articulated":
        plane = scene.add_entity(gs.morphs.Plane(), material=gs.materials.Mochi.Rigid())
        directory = tempfile.mkdtemp(prefix="mochi_bench_")
        pendulum = scene.add_entity(
            gs.morphs.MJCF(file=_write(directory, "pendulum.xml", pendulum_xml())),
            material=gs.materials.Mochi.Rigid(rho=DENSITY),
        )
        ball = scene.add_entity(
            gs.morphs.Sphere(radius=BALL_RADIUS, pos=BALL_POS), material=gs.materials.Mochi.Rigid(rho=BALL_DENSITY)
        )
        exports = [("articulated_ball", ball)]
        post_build.append(lambda: pendulum.set_dofs_velocity(np.array(SEED_JOINT_VELOCITIES)))
        post_build.append(lambda: scene.mochi_solver.enable_entity_contact(pendulum, plane, False))

        def probe():
            return float(_as_points(ball.get_pos())[0, 0])
    elif name == "equalities":
        directory = tempfile.mkdtemp(prefix="mochi_bench_")
        links = scene.add_entity(
            gs.morphs.MJCF(file=_write(directory, "links.xml", links_xml())),
            material=gs.materials.Mochi.Rigid(rho=DENSITY),
        )

        def probe():
            return float(_as_points(links.links[-1].get_pos())[0, 2])
    elif name == "soft_duck":
        scene.add_entity(gs.morphs.Plane(), material=gs.materials.Mochi.Rigid())
        duck = scene.add_entity(
            gs.morphs.Mesh(file=os.path.join(ASSETS, "duck_1899.node"), pos=DUCK_POS),
            material=gs.materials.Mochi.Elastic(E=DUCK_E, nu=DUCK_NU, rho=DUCK_RHO),
        )

        def probe():
            return float(_as_points(duck.get_vertices_position())[:, 2].min())
    elif name == "cloth_tshirt":
        scene.add_entity(gs.morphs.Plane(), material=gs.materials.Mochi.Rigid())
        shirt = scene.add_entity(
            gs.morphs.Mesh(file=os.path.join(ASSETS, "tshirt_3593.obj"), pos=TSHIRT_POS),
            material=gs.materials.Mochi.Shell(
                E=TSHIRT_E,
                nu=TSHIRT_NU,
                rho=TSHIRT_RHO,
                thickness=TSHIRT_THICKNESS,
                collider_radius=TSHIRT_CONTACT_RADIUS,
                self_contact=True,
            ),
        )

        def probe():
            return float(_as_points(shirt.get_vertices_position())[:, 2].max())
    elif name == "rod_helix":
        points = np.load(os.path.join(ASSETS, "helix_129.npy"))
        scene.add_entity(gs.morphs.Plane(), material=gs.materials.Mochi.Rigid())
        rod = scene.add_entity(
            gs.morphs.Rod(points=points, radius=ROD_RADIUS, pos=HELIX_POS),
            material=gs.materials.Mochi.Rod(collider_type="none", **rod_material_params()),
        )
        post_build.append(lambda: rod.set_vertices_fixed([0]))

        def probe():
            return float(_as_points(rod.get_vertices_position())[-1, 2])
    elif name == "franka":
        scene.add_entity(gs.morphs.Plane(), material=gs.materials.Mochi.Rigid())
        franka = scene.add_entity(
            gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"), material=gs.materials.Mochi.Rigid()
        )
        scene.add_entity(
            gs.morphs.Box(size=(0.05, 0.05, 0.05), pos=(0.5, 0.0, 0.025)), material=gs.materials.Mochi.Rigid()
        )

        def probe():
            return float(_as_points(franka.get_link("hand").get_pos())[0, 2])
    elif name in ("cloth_arm", "soft_gripper", "rope_arm"):
        scene.add_entity(gs.morphs.Plane(), material=gs.materials.Mochi.Rigid())
        franka = scene.add_entity(
            gs.morphs.MJCF(file=FRANKA_XML), material=gs.materials.Mochi.Rigid(friction=1.0, viscous_friction=1.0)
        )
        if name == "cloth_arm":
            directory = tempfile.mkdtemp(prefix="mochi_bench_")
            cloth = scene.add_entity(
                gs.morphs.Mesh(file=_sheet_obj(directory, CLOTH_CELLS, CLOTH_SIZE), pos=(*WORK_XY, CLOTH_HEIGHT)),
                material=gs.materials.Mochi.Shell(
                    E=CLOTH_E,
                    nu=CLOTH_NU,
                    rho=CLOTH_RHO,
                    thickness=CLOTH_THICKNESS,
                    friction=CLOTH_FRICTION,
                    collider_radius=CLOTH_CONTACT_RADIUS,
                    penalty_coefficient=CLOTH_PENALTY,
                    self_contact=True,
                ),
            )

            def probe():
                return float(_as_points(cloth.get_vertices_position())[:, 2].max())
        elif name == "soft_gripper":
            cube = scene.add_entity(
                gs.morphs.Box(
                    size=(SOFT_CUBE_SIZE,) * 3,
                    pos=(*WORK_XY, 0.5 * SOFT_CUBE_SIZE),
                    maxvolume=SOFT_CUBE_MAXVOLUME,
                    nobisect=False,
                ),
                material=gs.materials.Mochi.Elastic(
                    E=SOFT_CUBE_E, nu=SOFT_CUBE_NU, rho=SOFT_CUBE_RHO, friction=1.0, viscous_friction=1.0
                ),
            )

            def probe():
                return float(_as_points(cube.get_vertices_position())[:, 2].max())
        else:
            # The rope lies on the ground along x through the work point; the descending hand presses its middle.
            xs = WORK_XY[0] + np.linspace(-0.5 * ROPE_LENGTH, 0.5 * ROPE_LENGTH, ROPE_NODES)
            points = np.stack([xs, np.full(ROPE_NODES, WORK_XY[1]), np.full(ROPE_NODES, ROPE_RADIUS + 1e-3)], axis=-1)
            rope = scene.add_entity(
                gs.morphs.Rod(points=points, radius=ROPE_RADIUS),
                material=gs.materials.Mochi.Rod(**rod_material_params(ROPE_RADIUS, ROPE_E, ROPE_G, ROPE_RHO)),
            )

            def probe():
                return float(_as_points(rope.get_vertices_position())[ROPE_NODES // 2, 2])

        post_build.append(lambda: franka.set_qpos(np.array(FRANKA_START_QPOS)))
        post_build.append(
            lambda: franka.control_dofs_position(np.array(FRANKA_TARGET_QPOS)[FRANKA_MOTORS], FRANKA_MOTORS)
        )
        post_build.append(lambda: franka.control_dofs_force(np.full(2, FRANKA_FINGER_FORCE), FRANKA_FINGERS))
    else:
        raise ValueError(f"unknown scene {name}")

    scene.build(n_envs=n_envs)
    for entity_name, entity in exports:
        _export_mesh(entity_name, entity)
    for fn in post_build:
        fn()
    return scene, probe


# --- mochi (original engine) ------------------------------------------------------------------------------------------


def _unit_cube_tri_mesh(scale):
    """Corner-anchored box [0, sx] x [0, sy] x [0, sz] as 8 vertices / 12 triangles (outward winding), as mochi's
    examples build their links."""
    sx, sy, sz = scale
    coordinates = np.array(
        [[0, 0, 0], [sx, 0, 0], [sx, sy, 0], [0, sy, 0], [0, 0, sz], [sx, 0, sz], [sx, sy, sz], [0, sy, sz]],
        dtype=np.float64,
    )
    connectivity = np.array(
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
    return coordinates, connectivity


def _tri_mesh_shape(physics, verts_z_up, faces):
    verts = z_up_to_y_up(verts_z_up)
    return physics.create_tri_mesh_shape(
        coordinates=np.ascontiguousarray(verts, dtype=np.float64).reshape(-1),
        connectivity=np.ascontiguousarray(faces, dtype=np.int32).reshape(-1),
    )


def _mochi_world_nodes(actor):
    """World positions of a mochi deformable actor's nodes (root transform composed with the local positions)."""
    transform = actor.get_root_transform()
    x, y, z, w = np.asarray(transform.rotation, dtype=np.float64)
    R = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    local = np.asarray(actor.get_node_positions_local(), dtype=np.float64).reshape(-1, 3)
    return local @ R.T + np.asarray(transform.translation, dtype=np.float64)


def build_mochi(name):
    """Build a benchmark scene in the original mochi engine (call `physics.initialize` first). Returns (scene, probe)."""
    from superdex import physics

    spec = SCENES[name]
    if spec.get("genesis_only"):
        raise ValueError(f"scene {name} has no mochi counterpart")
    scene = physics.create_scene(f"benchmark_{name}")
    scene.set_gravity([0.0, -GRAVITY, 0.0])
    params = scene.get_solver_params()
    params.non_linear_solver.max_iter = spec["newton_cap"]
    params.non_linear_solver.line_search_type = physics.LineSearchType.RESIDUAL_NORM
    if name == "cloth_tshirt":
        params.experimental_eval.implicit_normal_force_for_dissipation = True
    scene.set_solver_params(params)

    def translation(pos_z_up):
        return physics.TransformRT(translation=list(z_up_to_y_up(pos_z_up)))

    def plane():
        shape = physics.create_plane_shape(normal=[0.0, 1.0, 0.0], distance=0.0)
        return scene.create_rigid_actor(name="ground", layer="Environment", shape=shape, is_static=True)

    if name == "rigid":
        plane()
        verts, faces = _load_generated_mesh("rigid_table")
        scene.create_rigid_actor(
            name="table",
            shape=_tri_mesh_shape(physics, verts, faces),
            is_static=True,
            world_from_local=translation(TABLE_POS),
            collider_type=physics.ColliderType.BOX,
        )
        verts, faces = _load_generated_mesh("rigid_sphere")
        scene.create_rigid_actor(
            name="sphere",
            shape=_tri_mesh_shape(physics, verts, faces),
            density=DENSITY,
            world_from_local=translation(SPHERE_POS),
            collider_type=physics.ColliderType.SPHERE,
        )
        verts, faces = _load_generated_mesh("rigid_cube")
        cube = scene.create_rigid_actor(
            name="cube",
            shape=_tri_mesh_shape(physics, verts, faces),
            density=DENSITY,
            world_from_local=translation(CUBE_POS),
            collider_type=physics.ColliderType.BOX,
        )

        def probe():
            return float(cube.get_root_transform().translation[1])
    elif name == "articulated":
        plane()

        def box(scale):
            coordinates, connectivity = _unit_cube_tri_mesh(scale)
            return physics.create_tri_mesh_shape(
                coordinates=coordinates.reshape(-1), connectivity=connectivity.reshape(-1)
            )

        rail_shape, cart_shape, arm_shape = box(RAIL_SCALE), box(CART_SCALE), box(ARM_SCALE)
        # Corner-anchored boxes centered on their inbound joint in x/z, hanging down -y (mochi's own example layout).
        joints = [
            physics.ArticulatedJointParams(name="CeilingWeld", type=physics.ArticulatedJointType.HARD),
            physics.ArticulatedJointParams(
                name="Rail",
                type=physics.ArticulatedJointType.PRISMATIC,
                axis=[1.0, 0.0, 0.0],
                parent_link_from_joint=physics.TransformRT(translation=[RAIL_SCALE[0] / 2, 0.0, RAIL_SCALE[2] / 2]),
                min_limit=[-RAIL_LIMIT, 0.0, 0.0],
                max_limit=[RAIL_LIMIT, 0.0, 0.0],
                limit_stiffness=RAIL_LIMIT_STIFFNESS,
                limit_damping=RAIL_LIMIT_DAMPING,
                friction=physics.ArticulatedJointFrictionParams(viscous=RAIL_VISCOUS_FRICTION),
                inertia=RAIL_ARMATURE,
            ),
            physics.ArticulatedJointParams(
                name="UpperSwing",
                type=physics.ArticulatedJointType.REVOLUTE,
                axis=[0.0, 0.0, -1.0],
                parent_link_from_joint=physics.TransformRT(translation=[CART_SCALE[0] / 2, 0.0, CART_SCALE[2] / 2]),
            ),
            physics.ArticulatedJointParams(
                name="LowerSwing",
                type=physics.ArticulatedJointType.SPHERICAL,
                parent_link_from_joint=physics.TransformRT(translation=[ARM_SCALE[0] / 2, 0.0, ARM_SCALE[2] / 2]),
            ),
        ]

        def link(link_name, parent, shape, scale, layer, hang):
            offset = [-scale[0] / 2, -scale[1] if hang else -scale[1] / 2, -scale[2] / 2]
            return physics.ArticulatedLinkParams(
                name=link_name,
                parent_link=parent,
                parent_joint_from_link=physics.TransformRT(translation=offset),
                shape=shape,
                collider_type=physics.ColliderType.BOX,
                layer=layer,
                density=DENSITY,
            )

        links = [
            link("RailHousing", -1, rail_shape, RAIL_SCALE, "Pendulum", False),
            link("Cart", 0, cart_shape, CART_SCALE, "Pendulum", True),
            link("UpperArm", 1, arm_shape, ARM_SCALE, "Pendulum", True),
            link("LowerArm", 2, arm_shape, ARM_SCALE, "EndEffector", True),
        ]
        art_params = physics.ArticulatedActorParams(name="DoublePendulumOnRail")
        art_params.world_from_root = physics.TransformRT(translation=[0.0, ROOT_HEIGHT, 0.0])
        art_params.joints = joints
        art_params.links = links
        articulation = scene.create_articulated_actor(art_params)
        for shape in (rail_shape, cart_shape, arm_shape):
            physics.release_shape(shape)
        np_real = np.float64 if physics.uses_double_precision() else np.float32
        articulation.set_articulated_joint_velocities(velocities=np.array(SEED_JOINT_VELOCITIES, dtype=np_real))
        verts, faces = _load_generated_mesh("articulated_ball")
        ball = scene.create_rigid_actor(
            name="Ball",
            layer="Ball",
            shape=_tri_mesh_shape(physics, verts, faces),
            density=BALL_DENSITY,
            world_from_local=translation(BALL_POS),
            collider_type=physics.ColliderType.SPHERE,
        )
        scene.enable_layer_contact_symmetric("Pendulum", "Pendulum", enable=False)
        scene.enable_layer_contact_symmetric("Pendulum", "Ball", enable=False)
        scene.enable_layer_contact_symmetric("Pendulum", "EndEffector", enable=False)
        scene.enable_layer_contact_symmetric("Pendulum", "Environment", enable=False)
        scene.enable_layer_contact_symmetric("EndEffector", "Environment", enable=False)

        def probe():
            return float(ball.get_root_transform().translation[0])
    elif name == "equalities":
        coordinates, connectivity = _unit_cube_tri_mesh((LINK_LENGTH, LINK_THICKNESS, LINK_THICKNESS))
        link_shape = physics.create_tri_mesh_shape(
            coordinates=coordinates.reshape(-1), connectivity=connectivity.reshape(-1)
        )
        half_w = LINK_THICKNESS / 2
        end_a = [0.0, half_w, half_w]
        end_b = [LINK_LENGTH, half_w, half_w]
        pivot = list(z_up_to_y_up(PIVOT_ANCHOR))
        link1_origin = [p - e for p, e in zip(pivot, end_a)]
        link2_origin = [link1_origin[0] + LINK_LENGTH, link1_origin[1], link1_origin[2]]
        link1 = scene.create_rigid_actor(
            name="Link1",
            layer="Link",
            shape=link_shape,
            density=DENSITY,
            world_from_local=physics.TransformRT(translation=link1_origin),
            collider_type=physics.ColliderType.BOX,
        )
        link2 = scene.create_rigid_actor(
            name="Link2",
            layer="Link",
            shape=link_shape,
            density=DENSITY,
            world_from_local=physics.TransformRT(translation=link2_origin),
            collider_type=physics.ColliderType.BOX,
        )
        scene.enable_layer_contact_symmetric("Link", "Link", enable=False)
        scene.create_rigid_pivot_position_constraint(
            physics.RigidPivotPositionConstraintParams(
                actor=link1.get_handle(),
                local_position=end_a,
                target_position=pivot,
                stiffness=CONSTRAINT_STIFFNESS,
                damping=CONSTRAINT_DAMPING,
                saturation=-1.0,
            )
        )
        scene.create_rigid_spherical_joint_constraint(
            physics.RigidSphericalJointConstraintParams(
                actor_a=link1.get_handle(),
                actor_b=link2.get_handle(),
                local_pos_a=end_b,
                local_pos_b=end_a,
                stiffness=CONSTRAINT_STIFFNESS,
                damping=CONSTRAINT_DAMPING,
                saturation=-1.0,
            )
        )

        def probe():
            return float(link2.get_root_transform().translation[1])
    elif name == "soft_duck":
        plane()
        verts, tets = load_tet_mesh(os.path.join(ASSETS, "duck_1899.node"))
        shape = physics.create_tet_mesh_shape(
            coordinates=np.ascontiguousarray(z_up_to_y_up(verts)).reshape(-1),
            connectivity=np.ascontiguousarray(tets, dtype=np.int32).reshape(-1),
        )
        material = physics.SoftMaterialParams()
        material.density = DUCK_RHO
        material.neo_hookean.youngs_modulus = DUCK_E
        material.neo_hookean.poisson_ratio = DUCK_NU
        duck = scene.create_soft_actor(
            name="duck", shape=shape, material=material, world_from_local=translation(DUCK_POS)
        )
        duck.register_query(physics.QueryType.NODE_POSITIONS)

        def probe():
            return float(_mochi_world_nodes(duck)[:, 1].min())
    elif name == "cloth_tshirt":
        plane()
        verts, faces = load_obj(os.path.join(ASSETS, "tshirt_3593.obj"))
        shape = _tri_mesh_shape(physics, verts, faces)
        material = physics.experimental.shell_material_params_from3d_isotropic(
            TSHIRT_E, TSHIRT_NU, TSHIRT_RHO, TSHIRT_THICKNESS
        )
        shell_params = physics.experimental.ShellActorParams(
            name="tshirt", shape=shape, material=material, world_from_local=translation(TSHIRT_POS)
        )
        shell_params.point_cloud_collider.radius = TSHIRT_CONTACT_RADIUS
        shell_params.point_cloud_collider.self_contact = True
        shirt = physics.experimental.create_shell_actor(scene, shell_params)
        shirt.register_query(physics.QueryType.NODE_POSITIONS)

        def probe():
            return float(_mochi_world_nodes(shirt)[:, 1].max())
    elif name == "rod_helix":
        plane()
        points = z_up_to_y_up(np.load(os.path.join(ASSETS, "helix_129.npy")))
        n = len(points)
        segments = np.stack([np.arange(n - 1), np.arange(1, n)], axis=-1)
        shape = physics.create_mesh_shape(
            physics.MeshData(
                nodes_per_element=2,
                coordinates=np.ascontiguousarray(points).reshape(-1),
                connectivity=np.ascontiguousarray(segments, dtype=np.int32).reshape(-1),
            )
        )
        params = rod_material_params()
        material = physics.experimental.RodMaterialParams(
            linear_density=params["linear_density"],
            linear_rotational_inertia=params["linear_rotational_inertia"],
            axial_stiffness=params["axial_stiffness"],
            torsional_stiffness=params["torsional_stiffness"],
            flexural_stiffness=[params["flexural_stiffness"], params["flexural_stiffness"]],
        )
        rod_params = physics.experimental.RodActorParams(
            name="helix", shape=shape, world_from_local=translation(HELIX_POS), material=material
        )
        rod = physics.experimental.create_rod_actor(scene, rod_params)
        first = list(points[0] + z_up_to_y_up(HELIX_POS))
        scene.create_deformable_node_position_constraint(
            actor=rod.get_handle(), node_index=0, position=first, stiffness=params["axial_stiffness"]
        )

        # Node position queries are not available for rods; the probe reports the root height instead.
        def probe():
            return float(rod.get_root_transform().translation[1])
    else:
        raise ValueError(f"unknown scene {name}")
    return scene, probe

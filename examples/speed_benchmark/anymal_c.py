import numpy as np

import genesis as gs

# The benchmark measures throughput at 30000 parallel environments, which needs a GPU.
gs.init(backend=gs.gpu, performance_mode=True)


scene = gs.Scene(
    sim_options=gs.options.SimOptions(
        dt=0.01,
    ),
    rigid_options=gs.options.RigidOptions(
        constraint_solver=gs.constraint_solver.Newton,
    ),
    show_viewer=False,
)

scene.add_entity(
    gs.morphs.Plane(),
)
robot = scene.add_entity(
    gs.morphs.URDF(
        file="urdf/anymal_c/urdf/anymal_c.urdf",
        pos=(0, 0, 0.8),
    ),
)
n_envs = 30000
scene.build(n_envs=n_envs)

joints_name = (
    "RH_HAA",
    "LH_HAA",
    "RF_HAA",
    "LF_HAA",
    "RH_HFE",
    "LH_HFE",
    "RF_HFE",
    "LF_HFE",
    "RH_KFE",
    "LH_KFE",
    "RF_KFE",
    "LF_KFE",
)
motors_dof_idx = [robot.get_joint(name).dofs_idx_local[0] for name in joints_name]

robot.set_dofs_kp(np.full(12, 1000), motors_dof_idx)
robot.control_dofs_position(np.zeros((n_envs, 12)), motors_dof_idx)

# Speed: 14.4M FPS
for i in range(1000):
    scene.step()

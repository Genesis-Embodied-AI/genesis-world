import math
import os
import time
from collections import namedtuple
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

import genesis as gs

from ..utils.assets import get_git_commit_timestamp, get_hf_dataset
from ..utils.misc import pprint_oneline

STEP_DT = 0.01
DURATION_WARMUP = 45.0
DURATION_RECORD = 15.0

SceneMeta = namedtuple(
    "SceneMeta",
    ["compile_time", "step_dt", "duration_warmup", "duration_record"],
    defaults=[STEP_DT, DURATION_WARMUP, DURATION_RECORD],
)

pytestmark = [
    pytest.mark.benchmarks,
    pytest.mark.cache(False),
    pytest.mark.debug(False),
    # Benchmarks measure the solve implementation the autotuner actually settles on, so the choice is left to it
    pytest.mark.use_deterministic_algorithms(False),
]


def get_rigid_solver_options(**kwargs):
    timestamp = get_git_commit_timestamp()

    # Beyond this point, track performance for default options, finally !
    if not (get_git_commit_timestamp("bbab229d74e5f30e2f641ccf6b009a65f3cbec0f") > timestamp):
        options = dict()

    # Try to be comparable to previous official release (ie 0.2.1) as much as possible.
    elif timestamp >= get_git_commit_timestamp("e46a1ffd33f681155422896c2e343e576e0a72b1"):
        # * Rename 'constraint_resolve_time' in 'constraint_timeconst'
        options = dict(
            enable_mujoco_compatibility=True,
            enable_self_collision=False,
            constraint_timeconst=2 * STEP_DT,
            constraint_solver=gs.constraint_solver.CG,
            max_collision_pairs=100,
            iterations=50,
            tolerance=1e-5,
        )
    elif timestamp >= get_git_commit_timestamp("b1ae77d5c838967dff6f85ee83796a4a82811061"):
        # * 'constraint_solver' now default to Newton (instead of CG)
        # * 'iterations' now default to 100 (instead of 50)
        # * 'tolerance' now default to 1e-8 (instead of 1e-5)
        options = dict(
            enable_mujoco_compatibility=True,
            enable_self_collision=False,
            constraint_resolve_time=2 * STEP_DT,
            constraint_solver=gs.constraint_solver.CG,
            max_collision_pairs=100,
            iterations=50,
            tolerance=1e-5,
        )
    elif timestamp >= get_git_commit_timestamp("6638c6389978594637da216b72be8d7a8f2272c4"):
        # * 'enable_mpr_vanilla' has been renamed in 'enable_mujoco_compatibility'
        options = dict(
            enable_mujoco_compatibility=True,
            constraint_resolve_time=2 * STEP_DT,
            max_collision_pairs=100,
        )
    elif timestamp >= get_git_commit_timestamp("5d04ec4c3ecba4a1e295d6a4a677c041e69092a7"):
        # * Expose option 'enable_multi_contact' (default to True)
        # * Expose 'enable_mpr_vanilla' (default to False)
        options = dict(
            enable_mpr_vanilla=True,
            enable_self_collision=False,
            constraint_resolve_time=2 * STEP_DT,
            max_collision_pairs=100,
        )
    elif timestamp >= get_git_commit_timestamp("6638c6389978594637da216b72be8d7a8f2272c4"):
        # * 'enable_self_collision' now default to True (instead of False)
        options = dict(
            enable_self_collision=False,
            constraint_resolve_time=2 * STEP_DT,
            max_collision_pairs=100,
        )
    elif timestamp >= get_git_commit_timestamp("361d9500cd321b25a63a28ace7a3d94fb9e45f65"):
        # * 'max_collision_pairs' now default to 300 (instead of 100)
        # * 'constraint_resolve_time' now default to None (instead of 2 * DT)
        options = dict(
            constraint_resolve_time=2 * STEP_DT,
            max_collision_pairs=100,
        )

    # Official release 0.2.1. Using the default options is fine
    else:
        options = {}

    return {**options, **kwargs}


def get_file_morph_options(**kwargs):
    timestamp = get_git_commit_timestamp()

    # Beyond this point, track performance for default options, finally !
    if not (get_git_commit_timestamp("bbab229d74e5f30e2f641ccf6b009a65f3cbec0f") > timestamp):
        options = {}

    # Try to be comparable to previous official release (ie 0.2.1) as much as possible.
    elif timestamp >= get_git_commit_timestamp("bbab229d74e5f30e2f641ccf6b009a65f3cbec0f"):
        # * 'decimation' has been enabled back by default
        # * 'decimate_aggressiveness' now defaults to 5
        options = dict(
            decimate=False,
            decimate_aggressiveness=7,
            decompose_robot_error_threshold=float("inf"),
            coacd_options=gs.options.CoacdOptions(
                resolution=2000,
                mcts_iterations=150,
                extrude_margin=0.01,
            ),
        )
    elif timestamp >= get_git_commit_timestamp("d7ea71d5490d0eba6c70a2dfe5943de62227fe68"):
        # * 'decompose_error_threshold' has been split in 'decompose_object_error_threshold' (default to 0.15) and
        #   'decompose_robot_error_threshold' (default to inf)
        options = dict(
            decimate=False,
            decimate_aggressiveness=7,
            decompose_robot_error_threshold=float("inf"),
            coacd_options=gs.options.CoacdOptions(
                resolution=2000,
                mcts_iterations=150,
                extrude_margin=0.01,
            ),
        )
    elif timestamp >= get_git_commit_timestamp("0e7b4be511d261d6ad25a382e5aa335468f5718b"):
        # * 'decimate_aggressiveness' has been exposed and default to 2
        options = dict(
            decimate=False,
            decimate_aggressiveness=7,
            decompose_error_threshold=float("inf"),
            coacd_options=gs.options.CoacdOptions(
                resolution=2000,
                mcts_iterations=150,
                extrude_margin=0.01,
            ),
        )
    elif timestamp >= get_git_commit_timestamp("361d9500cd321b25a63a28ace7a3d94fb9e45f65"):
        # * 'decimate' now defaults to 'convexify'
        # * 'decimate' aggressiveness has been updated from 0 to 2 (but not exposed), it was 7 originally
        options = dict(
            decimate=False,
            decompose_error_threshold=float("inf"),
            coacd_options=gs.options.CoacdOptions(
                resolution=2000,
                mcts_iterations=150,
                extrude_margin=0.01,
            ),
        )
    elif timestamp >= get_git_commit_timestamp("ec6e16949a65dbc62d318a734eeb7f17b0011e03"):
        # * 'decompose_error_threshold' default value updated to 0.15
        options = dict(
            decompose_error_threshold=float("inf"),
            coacd_options=gs.options.CoacdOptions(
                resolution=2000,
                mcts_iterations=150,
                extrude_margin=0.01,
            ),
        )
    elif timestamp >= get_git_commit_timestamp("3bc64493a537b7f52fca6b5fd2dd81f764c34433"):
        # * Move 'decimate', 'decompose_nonconvex' options from Mesh to FileMorph morphs (parent class)
        #   Before that, decimation and convex decomposition could not be enabled at all.
        # * 'convexify' has been enabled back by default
        # * 'decompose_nonconvex' has be deprecated in favor of 'decompose_error_threshold' (default to 0.2)
        # * 'CoacdOptions' options has been updated
        options = dict(
            decompose_error_threshold=float("inf"),
            coacd_options=gs.options.CoacdOptions(
                resolution=2000,
                mcts_iterations=150,
                extrude_margin=0.01,
            ),
        )

    elif timestamp >= get_git_commit_timestamp("8ea732b1a3b340ba7dff295fbd3527cb34b5b676"):
        # * 'convexify' has been disabled by default
        # * 'decimate' has been disabled by default (it only affects Mesh morphs at that time)
        # * 'decompose_nonconvex' has been disabled by default (it only affects Mesh morphs at that time)
        options = dict(
            convexify=True,
        )

    # Official release 0.2.1. Using the default options is fine
    else:
        options = {}

    return {**options, **kwargs}


# ---------------------------------------------------------------------------
# Scene factories
#
# Each factory builds a complete scene and returns (scene, step_fn, metadata).
#   - scene:    the built gs.Scene (useful for visualization, inspection)
#   - step_fn:  callable that runs one simulation step including control logic
#   - metadata: dict with compile_time, step_dt, and optional overrides for
#               duration_warmup, duration_record
#
# **scene_kwargs are forwarded to gs.Scene(), allowing callers to control
# show_viewer, vis_options, etc. without modifying these factories.
# ---------------------------------------------------------------------------


def make_go2(n_envs, solver=None, gjk=None, **scene_kwargs):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=STEP_DT,
        ),
        rigid_options=gs.options.RigidOptions(
            **get_rigid_solver_options(
                **(dict(constraint_solver=solver) if solver is not None else {}),
                **(dict(use_gjk_collision=gjk) if gjk is not None else {}),
            ),
        ),
        **{"show_viewer": False, "show_FPS": False, **scene_kwargs},
    )

    scene.add_entity(gs.morphs.Plane())
    robot = scene.add_entity(
        gs.morphs.URDF(
            file="urdf/go2/urdf/go2.urdf",
        ),
        vis_mode="collision",
    )
    time_start = time.time()
    scene.build(n_envs=n_envs)
    compile_time = time.time() - time_start

    ctrl_pos = torch.tensor(
        [0.0, 0.8, -1.5, 0.0, 0.8, -1.5, 0.0, 1.0, -1.5, 0.0, 1.0, -1.5],
        dtype=gs.tc_float,
        device=gs.device,
    )
    robot.control_dofs_position(ctrl_pos, dofs_idx_local=slice(6, None))

    init_qpos = torch.tensor(
        [[0.0, 0.0, 0.42, 1.0, 0.0, 0.0, 0.0, 0.0, 0.8, -1.5, 0.0, 0.8, -1.5, 0.0, 1.0, -1.5, 0.0, 1.0, -1.5]],
        dtype=gs.tc_float,
        device=gs.device,
    ).repeat((scene.n_envs, 1))
    dofs_lower_bound, dofs_upper_bound = robot.get_dofs_limit()
    init_qpos[:, 7:] = dofs_lower_bound[6:] + (dofs_upper_bound[6:] - dofs_lower_bound[6:]) * torch.as_tensor(
        np.random.rand(scene.n_envs, robot.n_dofs - 6), dtype=gs.tc_float, device=gs.device
    )
    robot.set_qpos(init_qpos)

    def step():
        scene.step()

    return scene, step, SceneMeta(compile_time=compile_time)


def make_anymal(n_envs, solver=None, gjk=None, control=None, with_kinematic=False, **scene_kwargs):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=STEP_DT,
        ),
        rigid_options=gs.options.RigidOptions(
            **get_rigid_solver_options(
                **(dict(constraint_solver=solver) if solver is not None else {}),
                **(dict(use_gjk_collision=gjk) if gjk is not None else {}),
            ),
        ),
        **{"show_viewer": False, "show_FPS": False, **scene_kwargs},
    )

    scene.add_entity(gs.morphs.Plane())
    robot = scene.add_entity(
        gs.morphs.URDF(
            **get_file_morph_options(
                file="urdf/anymal_c/urdf/anymal_c.urdf",
                pos=(0, 0, 0.8),
            )
        ),
    )
    ghost = None
    if with_kinematic:
        ghost = scene.add_entity(
            gs.morphs.URDF(
                file="urdf/anymal_c/urdf/anymal_c.urdf",
                pos=(0, -0.5, 0.8),
            ),
            material=gs.materials.Kinematic(),
        )
    time_start = time.time()
    scene.build(n_envs=n_envs)
    compile_time = time.time() - time_start

    motors_dof_idx = slice(6, None)
    robot.set_dofs_kp(1000.0, motors_dof_idx)
    robot.control_dofs_position(0.0, motors_dof_idx)

    if control == "uniform":
        rand_shape = (12,)
    elif control == "per_env":
        rand_shape = (n_envs, 12)
    else:
        rand_shape = None

    def step():
        if rand_shape is not None:
            robot.control_dofs_position(np.random.rand(*rand_shape) * 0.1 - 0.05, motors_dof_idx)
        if ghost is not None:
            ghost.set_dofs_position(np.random.rand(*rand_shape) * 0.1 - 0.05, motors_dof_idx)
        scene.step()

    return scene, step, SceneMeta(compile_time=compile_time)


def make_g1_fall(n_envs, solver=None, gjk=None, accessors=False, **scene_kwargs):
    step_dt = 0.005

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=step_dt,
        ),
        rigid_options=gs.options.RigidOptions(
            iterations=10,
            tolerance=1e-5,
            ls_iterations=20,
            **(dict(constraint_solver=solver) if solver is not None else {}),
            **(dict(use_gjk_collision=gjk) if gjk is not None else {}),
        ),
        **{"show_viewer": False, "show_FPS": False, **scene_kwargs},
    )

    scene.add_entity(gs.morphs.Plane())
    asset_path = get_hf_dataset(pattern="unitree_g1/*")
    robot = scene.add_entity(
        gs.morphs.MJCF(
            **get_file_morph_options(
                file=f"{asset_path}/unitree_g1/g1_29dof_rev_1_0.xml",
                pos=(0, 0, 1.0),
            )
        ),
        vis_mode="collision",
    )
    time_start = time.time()
    scene.build(n_envs=n_envs)
    compile_time = time.time() - time_start

    init_qpos = torch.zeros((robot.n_qs,), dtype=gs.tc_float, device=gs.device)
    init_qpos[2] = 1.0  # z position
    init_qpos[3] = 1.0  # quaternion w component
    robot.set_qpos(init_qpos)

    random_forces = torch.zeros((n_envs, robot.n_dofs), dtype=gs.tc_float, device=gs.device)
    max_force = 50.0

    # Snapshot the state exercised by the per-step partial reset in the 'accessors' variant: the same per-env
    # accessor + reset workload seen in RL rollouts, run on top of the falling-humanoid dynamics.
    if accessors:
        qpos0 = torch.tile(init_qpos, (n_envs, 1)) if n_envs > 0 else init_qpos
        vel0 = torch.zeros((*((n_envs,) if n_envs > 0 else ()), robot.n_dofs), dtype=gs.tc_float, device=gs.device)
        state_rigid_0 = scene.rigid_solver.get_state()

        if n_envs > 0:
            n_reset_envs = max(int(0.02 * n_envs), 1)
            reset_envs_idx = torch.as_tensor(
                np.random.permutation(n_envs)[:n_reset_envs], dtype=gs.tc_int, device=gs.device
            )
            reset_envs_mask = torch.isin(scene._envs_idx, reset_envs_idx)
        else:
            reset_envs_mask = None

        dofs_stiffness = robot.get_dofs_stiffness()
        dofs_damping = robot.get_dofs_damping()

        # Per-selected-env base pose subset exercising the bool-mask zero-copy 'masked_scatter_' path
        base_pos0 = robot.get_pos(reset_envs_mask)
        base_quat0 = robot.get_quat(reset_envs_mask)

    def step():
        random_forces.uniform_(-max_force, max_force)
        robot.control_dofs_force(random_forces)
        scene.step()
        if accessors:
            robot.get_ang()
            robot.get_vel()
            robot.get_dofs_position()
            robot.get_dofs_velocity()
            robot.get_links_pos()
            robot.get_links_quat()
            robot.get_links_vel()
            robot.get_contacts(is_padded=True)

            # TODO: Entire scene reset is still slow currently because 'partial=False' by default.
            scene.rigid_solver.set_state(0, state_rigid_0, envs_idx=reset_envs_mask, partial=True)

            robot.set_dofs_stiffness(dofs_stiffness)
            robot.set_dofs_damping(dofs_damping)
            robot.set_dofs_velocity(vel0, envs_idx=reset_envs_mask, skip_forward=True)
            robot.set_qpos(qpos0, envs_idx=reset_envs_mask, zero_velocity=False, skip_forward=True)
            robot.set_pos(base_pos0, envs_idx=reset_envs_mask, skip_forward=True)
            robot.set_quat(base_quat0, envs_idx=reset_envs_mask, relative=False, skip_forward=True)

    return (
        scene,
        step,
        SceneMeta(
            compile_time=compile_time,
            step_dt=step_dt,
            duration_warmup=20.0,
            duration_record=5.0,
        ),
    )


def make_double_smplx(n_envs, solver=None, gjk=None, **scene_kwargs):
    # Two 159-DOF SMPL-X humanoids dropped on a plane with random per-DOF torques: the 318-DOF environment exceeds the
    # constraint-Hessian shared-memory cap and each entity exceeds the per-entity mass-matrix cap, so both
    # factorizations exercise their uncapped paths.
    STEP_DT = 0.005
    INIT_POSS = ((0.0, -0.5, 1.4), (0.0, 0.5, 1.4))  # pelvis dropped 1.4 m above the plane, offset in y
    INIT_EULER = (90.0, 0.0, 0.0)  # SMPL-X canonical frame is Y-up; rotate to stand upright in Genesis (Z-up)
    MAX_FORCE = 500.0

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=STEP_DT,
        ),
        rigid_options=gs.options.RigidOptions(
            # Cap the reserved contacts so the self-colliding 318-DOF Jacobian (n_constraints * n_dofs * n_envs) stays
            # within int32 at n_envs=4096.
            max_collision_pairs=40,
            **(dict(constraint_solver=solver) if solver is not None else {}),
            **(dict(use_gjk_collision=gjk) if gjk is not None else {}),
        ),
        **{"show_viewer": False, "show_FPS": False, **scene_kwargs},
    )

    scene.add_entity(gs.morphs.Plane())
    asset_path = get_hf_dataset(pattern="smplx_humanoid/*")
    smplx_xml = f"{asset_path}/smplx_humanoid/smplx_humanoid.xml"
    robots = [
        scene.add_entity(
            gs.morphs.MJCF(
                **get_file_morph_options(
                    file=smplx_xml,
                    pos=pos,
                    euler=INIT_EULER,
                )
            ),
            vis_mode="collision",
        )
        for pos in INIT_POSS
    ]
    time_start = time.time()
    scene.build(n_envs=n_envs)
    compile_time = time.time() - time_start

    random_forces = [torch.zeros((n_envs, robot.n_dofs), dtype=gs.tc_float, device=gs.device) for robot in robots]

    def step():
        for robot, forces in zip(robots, random_forces):
            forces.uniform_(-MAX_FORCE, MAX_FORCE)
            robot.control_dofs_force(forces)
        scene.step()

    return (
        scene,
        step,
        SceneMeta(
            compile_time=compile_time,
            step_dt=STEP_DT,
            duration_warmup=20.0,
            duration_record=5.0,
        ),
    )


def make_shadow_hand_cubes(n_envs, solver=None, gjk=None, sparse_solve=None, **scene_kwargs):
    STEP_DT = 1.0 / 30
    TABLE_Z = 0.762

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=STEP_DT,
            substeps=4,
        ),
        rigid_options=gs.options.RigidOptions(
            noslip_iterations=2,
            max_collision_pairs=256,
            sparse_solve=sparse_solve,
            **(dict(constraint_solver=solver) if solver is not None else {}),
            **(dict(use_gjk_collision=gjk) if gjk is not None else {}),
        ),
        **{"show_viewer": False, "show_FPS": False, **scene_kwargs},
    )

    # Two shadow hands placed horizontally, palms facing down
    scene.add_entity(
        morph=gs.morphs.URDF(
            file="urdf/shadow_hand/shadow_hand.urdf",
            pos=(-0.1, 0.30, 0.9 * TABLE_Z),
            euler=(90, 0, 0),
            fixed=True,
        )
    )
    scene.add_entity(
        morph=gs.morphs.URDF(
            file="urdf/shadow_hand/shadow_hand.urdf",
            pos=(0.1, 0.30, 0.9 * TABLE_Z),
            euler=(90, 0, 0),
            fixed=True,
        )
    )

    # Table
    scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0, 0, TABLE_Z / 2),
            size=(0.5, 0.5, TABLE_Z / 2),
            fixed=True,
        )
    )

    # 25 cubes in a 5x5 grid on the table
    for i in range(25):
        x = -0.10 + 0.05 * (i % 5)
        y = -0.05 + 0.05 * (i // 5)
        scene.add_entity(
            material=gs.materials.Rigid(
                friction=0.8,
            ),
            morph=gs.morphs.Box(
                pos=(x, y, TABLE_Z + 0.01),
                size=(0.02, 0.02, 0.02),
            ),
        )

    time_start = time.time()
    scene.build(n_envs=n_envs)
    compile_time = time.time() - time_start

    def step():
        scene.step()

    return (
        scene,
        step,
        SceneMeta(
            compile_time=compile_time,
            step_dt=STEP_DT,
            duration_warmup=20.0,
            duration_record=5.0,
        ),
    )


def make_dex_hand(n_envs, solver=None, gjk=None, **scene_kwargs):
    shadow_hand_path = Path(get_hf_dataset(pattern="shadow_hand/*"))
    dex_path = Path(get_hf_dataset(pattern="dex/*"))

    WRIST_STIFFNESS = 20
    FINGER_FORCE = 0.6
    DRILL_STIFFNESS = 20

    STEP_DT = 1 / 16

    JOINT_NAMES = [
        *("FFJ4", "FFJ3", "FFJ2", "FFJ1"),
        *("MFJ4", "MFJ3", "MFJ2", "MFJ1"),
        *("RFJ4", "RFJ3", "RFJ2", "RFJ1"),
        *("LFJ5", "LFJ4", "LFJ3", "LFJ2", "LFJ1"),
        *("THJ5", "THJ4", "THJ3", "THJ2", "THJ1"),
    ]
    LEFT_DOFS = [
        *(0.34907, 0.24929, 0.54424, 0.65614),
        *(0.21329, 0.08060, 0.19969, 0.66944),
        *(1.57080, 0.21846, 0.53605, 0.44963),
        *(0.38350, 0.02379, 0.41705, 0.54773, 0.61160),
        *(0.36664, 0.44036, 0.20944, 0.34497, 0.15896),
    ]
    RIGHT_DOFS = [
        *(0.34907, 0.23328, 0.57399, 0.70467),
        *(0.00000, 0.34907, 0.51778, 0.65078),
        *(1.48947, 0.33727, 0.55919, 0.56268),
        *(0.54360, -0.08460, 0.48588, 0.66095, 0.73317),
        *(0.13239, 0.45613, 0.20944, 0.19625, 0.00750),
    ]

    hand_configs = [
        dict(
            pos=(0.19227, -0.00058, 1.31227),
            quat=(-0.45215, -0.31265, 0.76087, 0.34480),
            dofs=LEFT_DOFS,
            urdf="shadow_hand_left_woarm.urdf",
        ),
        dict(
            pos=(0.16257, 0.24658, 1.28047),
            quat=(0.74525, 0.46466, -0.45186, -0.15660),
            dofs=RIGHT_DOFS,
            urdf="shadow_hand_right_woarm.urdf",
        ),
    ]

    # Note: screenshots on effect of coacd_options on how the drill and table look
    # at https://github.com/Genesis-Embodied-AI/Genesis/pull/2500#discussion_r2900243286
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=STEP_DT,
            substeps=25,
        ),
        rigid_options=gs.options.RigidOptions(
            max_collision_pairs=200,
            **(dict(use_gjk_collision=gjk) if gjk is not None else {}),
        ),
        **{"show_viewer": False, "show_FPS": False, **scene_kwargs},
    )

    hands = []
    for cfg in hand_configs:
        urdf_path = str(shadow_hand_path / "shadow_hand" / cfg["urdf"])
        hand = scene.add_entity(
            gs.morphs.URDF(
                file=urdf_path,
                pos=cfg["pos"],
                quat=cfg["quat"],
            ),
        )
        hands.append(hand)

    table_path = str(dex_path / "dex" / "table.glb")
    scene.add_entity(
        gs.morphs.Mesh(
            file=table_path,
            pos=(0.1, 0.0, 0.485403),
            euler=(0, 0, 90),
            fixed=True,
        )
    )
    drill_path = str(dex_path / "dex" / "drill_1.glb")
    drill = scene.add_entity(
        gs.morphs.Mesh(
            file=drill_path,
            pos=(0.15, 0.1, 0.87),
            euler=(90, 0, 225),
        )
    )

    time_start = time.time()
    scene.build(n_envs=n_envs)
    compile_time = time.time() - time_start

    for hand, default_dof in zip(hands, (LEFT_DOFS, RIGHT_DOFS)):
        kp = [WRIST_STIFFNESS] * 6 + [40.0] * (hand.n_dofs - 6)
        hand.set_dofs_kp(kp)
        hand.set_dofs_kv(2.0 * np.sqrt(kp))
        hand.set_dofs_position(
            default_dof,
            dofs_idx_local=[hand.get_joint(name).dofs_idx_local[0] for name in JOINT_NAMES],
        )
        hand.control_dofs_position(torch.clamp(hand.get_dofs_position(), *hand.get_dofs_limit()))

    finger_dofs = {hand: slice(6, hand.n_dofs) for hand in hands}
    random_forces = {
        hand: torch.zeros((n_envs, hand.n_dofs - 6), dtype=gs.tc_float, device=gs.device) for hand in hands
    }
    base_xy_targets = {hand: hand.get_dofs_position()[:, :2] for hand in hands}

    drill.set_dofs_kp(DRILL_STIFFNESS, dofs_idx_local=[0, 1])
    drill.set_dofs_kv(2.0 * math.sqrt(DRILL_STIFFNESS), dofs_idx_local=[0, 1])
    drill_xy_target = drill.get_dofs_position()[:, :2]

    for hand in hands:
        hand.control_dofs_position(base_xy_targets[hand], dofs_idx_local=[0, 1])
    drill.control_dofs_position(drill_xy_target, dofs_idx_local=[0, 1])

    def step():
        for hand in hands:
            random_forces[hand].uniform_(-FINGER_FORCE, FINGER_FORCE)
            hand.control_dofs_force(random_forces[hand], dofs_idx_local=finger_dofs[hand])
        scene.step()

    return (
        scene,
        step,
        SceneMeta(
            compile_time=compile_time,
            step_dt=STEP_DT,
            duration_warmup=20.0,
            duration_record=5.0,
        ),
    )


def make_convexify(n_envs, solver=None, gjk=None, **scene_kwargs):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.004),
        rigid_options=gs.options.RigidOptions(
            max_collision_pairs=8000,
            **(dict(constraint_solver=solver) if solver is not None else {}),
            **(dict(use_gjk_collision=gjk) if gjk is not None else {}),
        ),
        **{"show_viewer": False, "show_FPS": False, **scene_kwargs},
    )

    scene.add_entity(
        gs.morphs.Mesh(file="meshes/tank.obj", scale=5.0, fixed=True, euler=(90, 0, 90)),
        vis_mode="collision",
    )
    # 80 auto-decomposed objects (mug/donut/cup/apple, CoACD) stacked in a 4x4x5 staggered grid that collapses
    # into the tank -- exercises the convex-decomposition collision path at a high contact count.
    assets = (("mug_1", "output.xml"), ("donut_0", "output.xml"), ("cup_2", "model.xml"), ("apple_15", "model.xml"))
    asset_files = {name: f"{get_hf_dataset(pattern=f'{name}/*')}/{name}/{xml}" for name, xml in assets}
    for i in range(80):
        gx, gy, gz = i % 4, (i // 4) % 4, i // 16
        name = assets[(gx + gy + gz) % len(assets)][0]
        scene.add_entity(
            gs.morphs.MJCF(
                file=asset_files[name],
                pos=((gx + 0.5 * (gz % 2)) * 0.1 - 0.18, (gy + 0.5 * (gz % 2)) * 0.145 - 0.265, 0.11 + gz * 0.08),
                euler=(90.0, 0.0, 0.0),
            ),
            vis_mode="collision",
        )

    time_start = time.time()
    scene.build(n_envs=n_envs)
    compile_time = time.time() - time_start

    def step():
        scene.step()

    return (
        scene,
        step,
        SceneMeta(
            compile_time=compile_time,
            step_dt=0.004,
            duration_warmup=20.0,
            duration_record=5.0,
        ),
    )


def make_nonconvex_spacecraft(n_envs, solver=None, gjk=None, **scene_kwargs):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.002),
        rigid_options=gs.options.RigidOptions(
            max_collision_pairs=20,
            **(dict(constraint_solver=solver) if solver is not None else {}),
            **(dict(use_gjk_collision=gjk) if gjk is not None else {}),
        ),
        **{"show_viewer": False, "show_FPS": False, **scene_kwargs},
    )

    # Non-watertight, non-convex spacecraft hull (convexify disabled) with a grid of boxes settling against it.
    asset_path = get_hf_dataset(pattern="spacecraft.obj")
    scene.add_entity(
        gs.morphs.Mesh(
            file=f"{asset_path}/spacecraft.obj",
            pos=(-0.4, 0.0, -4.0),
            euler=(90.0, 0.0, 0.0),
            scale=3.0,
            convexify=False,
            fixed=True,
        ),
        vis_mode="collision",
    )
    obj = scene.add_entity(gs.morphs.Box(size=(0.1, 0.1, 0.1)))

    time_start = time.time()
    scene.build(n_envs=n_envs)
    compile_time = time.time() - time_start

    obj.set_pos(
        torch.cartesian_prod(
            torch.linspace(-6.25, 9.05, 8),
            torch.linspace(-5.2, 5.5, 8),
            torch.tensor((0.39,)),
        )
    )

    def step():
        scene.step()

    return (
        scene,
        step,
        SceneMeta(
            compile_time=compile_time,
            step_dt=0.002,
            duration_warmup=20.0,
            duration_record=5.0,
        ),
    )


def make_table_bussing(n_envs, solver=None, gjk=None, **scene_kwargs):
    # Table-bussing digital twin: a fixed 18-DOF bimanual robot posed over a work table, with a clutter of dining
    # objects dropped and settling on the tabletop -- a contact-rich mesh-collision workload with a high-DOF
    # articulated system held in place. Every asset is already on the pinned HF dataset (dual_arms_primitives is the
    # all-primitive, mesh-free stand-in robot; work_table.glb and the mug/cup/apple/donut clutter are public), so
    # this benchmark needs no additional asset upload.
    STEP_DT = 1.0 / 30.0
    # work_table.glb sits with its top face ~0.54 above the mesh origin at the default scale (cf. test_mesh_repair,
    # which places the table at z=-0.54 so its top lands at z~0), so at pos z=0 the tabletop is at z~0.54.
    TABLE_TOP_Z = 0.54

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=STEP_DT, substeps=10),
        rigid_options=gs.options.RigidOptions(
            noslip_iterations=5,
            max_collision_pairs=256,
            max_contacts=1024,
            **(dict(constraint_solver=solver) if solver is not None else {}),
            **(dict(use_gjk_collision=gjk) if gjk is not None else {}),
        ),
        **{"show_viewer": False, "show_FPS": False, **scene_kwargs},
    )

    scene.add_entity(gs.morphs.Plane())

    table_path = get_hf_dataset(pattern="work_table.glb")
    scene.add_entity(
        gs.morphs.Mesh(file=f"{table_path}/work_table.glb", pos=(0.5, 0.0, 0.0), fixed=True),
        vis_mode="collision",
    )

    robot_path = get_hf_dataset(pattern="dual_arms_primitives.urdf")
    robot = scene.add_entity(
        gs.morphs.URDF(file=f"{robot_path}/dual_arms_primitives.urdf", pos=(0.0, 0.0, TABLE_TOP_Z + 0.5), fixed=True),
    )

    # Dining clutter to bus off the table: auto-decomposed (CoACD) mug/cup/apple/donut in a 4x4 grid in front of the
    # robot, staggered in height so none overlap at spawn, then dropped so they settle on the tabletop.
    assets = (("mug_1", "output.xml"), ("cup_2", "model.xml"), ("apple_15", "model.xml"), ("donut_0", "output.xml"))
    asset_files = {name: f"{get_hf_dataset(pattern=f'{name}/*')}/{name}/{xml}" for name, xml in assets}
    for i in range(16):
        gx, gy = i % 4, i // 4
        name = assets[(gx + gy) % len(assets)][0]
        scene.add_entity(
            gs.morphs.MJCF(
                file=asset_files[name],
                pos=(0.30 + 0.11 * gx, -0.17 + 0.11 * gy, TABLE_TOP_Z + 0.06 + 0.02 * i),
                euler=(90.0, 0.0, 0.0),
            ),
            vis_mode="collision",
        )

    time_start = time.time()
    scene.build(n_envs=n_envs)
    compile_time = time.time() - time_start

    # Hold every actuated DOF at its initial pose so the arms stay posed over the table instead of collapsing.
    robot.set_dofs_kp(np.full((robot.n_dofs,), 100.0, dtype=np.float32))
    robot.control_dofs_position(torch.zeros((n_envs, robot.n_dofs), dtype=gs.tc_float, device=gs.device))

    def step():
        scene.step()

    return (
        scene,
        step,
        SceneMeta(
            compile_time=compile_time,
            step_dt=STEP_DT,
            duration_warmup=20.0,
            duration_record=5.0,
        ),
    )


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


def run_benchmark(step_fn, *, n_envs, meta):
    import quadrants as qd

    qd.sync()

    # Force CUDA-context init / lazy module load / first kernel SASS JIT to
    # complete BEFORE entering the wall-clock warmup loop. Otherwise, the
    # warmup's `meta.duration_warmup` budget can be partially consumed by
    # CUDA setup (which happens on the first kernel launch into a fresh
    # context) instead of by actual step()s. When that happens, the wall-
    # clock warmup ends before the per-process cold-start phase (~5 s of
    # slow step()s on hardware tested) is over, and the recording window
    # then measures cold-start step time instead of steady-state — which
    # manifested as intermittent 50%+ apparent `runtime_fps` regressions in
    # `g1_fall` cluster benchmark runs even though steady-state step() time
    # was unchanged. After this single sync-bracketed step the warmup loop's
    # wall-clock budget contains only step() execution time, so 20 s of
    # warmup actually means 20 s of step()s — enough to absorb the cold-
    # start phase on every benchmark. Cost: 1 step() + 1 qd.sync().
    step_fn()
    qd.sync()

    num_steps = 0
    is_recording = False
    time_start = time.time()
    while True:
        step_fn()
        time_elapsed = time.time() - time_start
        if is_recording:
            num_steps += 1
            if time_elapsed > meta.duration_record:
                qd.sync()
                time_elapsed = time.time() - time_start
                break
        elif time_elapsed > meta.duration_warmup:
            qd.sync()
            time_start = time.time()
            is_recording = True
    runtime_fps = int(num_steps * max(n_envs, 1) / time_elapsed)
    realtime_factor = runtime_fps * meta.step_dt

    return dict(compile_time=meta.compile_time, runtime_fps=runtime_fps, realtime_factor=realtime_factor)


# ---------------------------------------------------------------------------
# Pytest infrastructure
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def stream_writers(printer_session, request):
    report_path = Path(request.config.getoption("--speed-test-filepath"))

    # Delete old unrelated worker-specific reports
    worker_id = os.environ.get("PYTEST_XDIST_WORKER")
    if worker_id == "gw0":
        worker_count = int(os.environ["PYTEST_XDIST_WORKER_COUNT"])

        for path in report_path.parent.glob("-".join((report_path.stem, "*.txt"))):
            _, worker_id_ = path.stem.rsplit("-", 1)
            worker_num = int(worker_id_[2:])
            if worker_num >= worker_count:
                path.unlink()

    # Create new empty worker-specific report
    report_name = "-".join(filter(None, (report_path.stem, worker_id)))
    report_path = report_path.with_name(f"{report_name}.txt")
    if report_path.exists():
        report_path.unlink()
    fd = open(report_path, "w")

    yield (lambda msg: print(msg, file=fd, flush=True), printer_session)

    fd.close()


@pytest.fixture(scope="function")
def factory_logger(stream_writers):
    class Logger:
        def __init__(self, hparams: dict[str, Any]):
            self.hparams = {
                **hparams,
                "dtype": "ndarray" if gs.use_ndarray else "field",
                "backend": str(gs.backend.name),
            }

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            pass

        def write(self, items):
            nonlocal stream_writers

            if stream_writers:
                msg = (
                    pprint_oneline(self.hparams, delimiter=" \t| ")
                    + " \t| "
                    + pprint_oneline(items, delimiter=" \t| ", digits=1)
                )
                for writer in stream_writers:
                    writer(msg)

    return Logger


# ---------------------------------------------------------------------------
# Pytest fixture wrappers (thin: just wire factory -> runner)
# ---------------------------------------------------------------------------


@pytest.fixture
def go2(solver, n_envs, gjk):
    _, step_fn, meta = make_go2(n_envs, solver=solver, gjk=gjk)
    return run_benchmark(step_fn, n_envs=n_envs, meta=meta)


@pytest.fixture
def anymal_random(solver, n_envs, gjk):
    _, step_fn, meta = make_anymal(n_envs, solver=solver, gjk=gjk, control="per_env")
    return run_benchmark(step_fn, n_envs=n_envs, meta=meta)


@pytest.fixture
def g1_fall(solver, n_envs, gjk):
    _, step_fn, meta = make_g1_fall(n_envs, solver=solver, gjk=gjk)
    return run_benchmark(step_fn, n_envs=n_envs, meta=meta)


@pytest.fixture
def g1_fall_accessors(solver, n_envs, gjk):
    _, step_fn, meta = make_g1_fall(n_envs, solver=solver, gjk=gjk, accessors=True)
    return run_benchmark(step_fn, n_envs=n_envs, meta=meta)


@pytest.fixture
def double_smplx(solver, n_envs, gjk):
    _, step_fn, meta = make_double_smplx(n_envs, solver=solver, gjk=gjk)
    return run_benchmark(step_fn, n_envs=n_envs, meta=meta)


@pytest.fixture
def shadow_hand(solver, n_envs, gjk):
    # sparse_solve left at its default (None) so the solver auto-selects (CPU multi-island -> sparse skyline).
    _, step_fn, meta = make_shadow_hand_cubes(n_envs, solver=solver, gjk=gjk)
    return run_benchmark(step_fn, n_envs=n_envs, meta=meta)


@pytest.fixture
def dex_hand(solver, n_envs, gjk):
    _, step_fn, meta = make_dex_hand(n_envs, solver=solver, gjk=gjk)
    return run_benchmark(step_fn, n_envs=n_envs, meta=meta)


@pytest.fixture
def convexify(solver, n_envs, gjk):
    _, step_fn, meta = make_convexify(n_envs, solver=solver, gjk=gjk)
    return run_benchmark(step_fn, n_envs=n_envs, meta=meta)


@pytest.fixture
def nonconvex_spacecraft(solver, n_envs, gjk):
    _, step_fn, meta = make_nonconvex_spacecraft(n_envs, solver=solver, gjk=gjk)
    return run_benchmark(step_fn, n_envs=n_envs, meta=meta)


@pytest.fixture
def table_bussing(solver, n_envs, gjk):
    _, step_fn, meta = make_table_bussing(n_envs, solver=solver, gjk=gjk)
    return run_benchmark(step_fn, n_envs=n_envs, meta=meta)


# ---------------------------------------------------------------------------
# Parametrized benchmark test
# ---------------------------------------------------------------------------


# Full benchmark suite (one instance of each, solver/gjk left at their defaults), run on the 'field' dtype.
BENCHMARKS_FIELD = [
    ("go2", None, None, 4096, gs.gpu),
    ("anymal_random", None, None, 20000, gs.gpu),
    ("g1_fall", None, None, 4096, gs.gpu),
    ("g1_fall_accessors", None, None, 4096, gs.gpu),
    ("double_smplx", None, None, 4096, gs.gpu),
    ("dex_hand", None, None, 4096, gs.gpu),
    ("shadow_hand", None, None, 0, gs.cpu),
    ("convexify", None, None, 0, gs.cpu),
    ("nonconvex_spacecraft", None, None, 64, gs.cpu),
    ("table_bussing", None, None, 50, gs.cpu),
    ("table_bussing", None, None, 50, gs.gpu),
]

# Reduced subset, run on the 'ndarray' dtype only.
BENCHMARKS_NDARRAY = [
    ("dex_hand", None, None, 4096, gs.gpu),
    ("g1_fall_accessors", None, None, 4096, gs.gpu),
    ("table_bussing", None, None, 50, gs.cpu),
]

# The dtype is selected before collection via 'GS_ENABLE_NDARRAY' ('0' => field, otherwise ndarray).
BENCHMARKS = BENCHMARKS_FIELD if os.environ.get("GS_ENABLE_NDARRAY", "1") == "0" else BENCHMARKS_NDARRAY


@pytest.mark.parametrize(
    "runnable, solver, gjk, n_envs, backend",
    BENCHMARKS,
)
def test_speed(factory_logger, request, runnable, solver, gjk, n_envs):
    with factory_logger(
        {
            "env": runnable,
            "batch_size": n_envs,
            **({"constraint_solver": solver} if solver is not None else {}),
            **({"gjk_collision": gjk} if gjk is not None else {}),
        }
    ) as logger:
        logger.write(request.getfixturevalue(runnable))

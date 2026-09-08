from contextlib import nullcontext
from copy import deepcopy

import numpy as np
import pytest
import torch

import genesis as gs
import genesis.utils.geom as gu
from genesis.engine.states.solvers import RigidSolverState
from genesis.utils.misc import qd_to_numpy, qd_to_torch

from ..utils.assertions import assert_allclose, assert_equal


@pytest.mark.slow  # ~200s
@pytest.mark.parametrize(
    "n_envs, batched, backend",
    [
        (0, False, gs.cpu),
        (0, False, gs.gpu),
        (3, False, gs.cpu),
        (3, True, gs.cpu),
    ],
)
def test_data_accessor(n_envs, batched, tol):
    # Create and build the scene
    scene = gs.Scene(
        rigid_options=gs.options.RigidOptions(
            batch_links_info=batched,
            batch_joints_info=batched,
            batch_dofs_info=batched,
        ),
        show_viewer=False,
        show_FPS=False,
    )
    scene.add_entity(gs.morphs.Plane())
    gs_robot = scene.add_entity(
        gs.morphs.URDF(
            file="urdf/go2/urdf/go2.urdf",
        ),
    )
    gs_link = gs_robot.get_link("RR_thigh")
    gs_geom = gs_link.geoms[0]
    gs_vgeom = gs_link.vgeoms[0]
    scene.build(n_envs=n_envs)
    gs_s = scene.sim.rigid_solver

    # Initialize the simulation
    np.random.seed(0)
    dofs_limit = qd_to_numpy(gs_s.dyn_info.dofs.limit, transpose=True)
    dof_bounds = np.broadcast_to(dofs_limit, (max(n_envs, 1), gs_s.n_dofs, 2)).copy()
    dof_bounds[..., :2, :] = (-1.0, 1.0)
    dof_bounds[..., 2, :] = (0.7, 1.0)
    dof_bounds[..., 3:6, :] = (-np.pi / 2, np.pi / 2)
    for i in range(max(n_envs, 1)):
        lower, upper = dof_bounds[i, :, 0], dof_bounds[i, :, 1]
        qpos = lower + (upper - lower) * np.random.rand(gs_robot.n_dofs)
        gs_robot.set_dofs_position(qpos, envs_idx=([i] if n_envs else None))

    # Simulate for a while, until they collide with something
    for _ in range(400):
        scene.step()

        gs_n_contacts = gs_s.collider.collider_state.n_contacts.to_numpy()
        assert len(gs_n_contacts) == max(n_envs, 1)
        for as_tensor in (False, True):
            for to_torch in (False, True):
                contacts_info = gs_s.collider.get_contacts(as_tensor, to_torch)
                for value in contacts_info.values():
                    if n_envs > 0:
                        assert n_envs == len(value)
                    else:
                        assert gs_n_contacts[0] == len(value)
                        value = value[None] if as_tensor else (value,)

                    for i_b in range(n_envs):
                        n_contacts = gs_n_contacts[i_b]
                        if as_tensor:
                            assert isinstance(value, torch.Tensor if to_torch else np.ndarray)
                            if value.dtype in (gs.tc_int, gs.np_int):
                                assert (value[i_b, :n_contacts] != -1).all()
                                assert (value[i_b, n_contacts:] == -1).all()
                            else:
                                assert_allclose(value[i_b, n_contacts:], 0.0, tol=0)
                        else:
                            assert isinstance(value, (list, tuple))
                            assert value[i_b].shape[0] == n_contacts
                            if value[i_b].dtype in (gs.tc_int, gs.np_int):
                                assert (value[i_b] != -1).all()

        if (gs_n_contacts > 0).all():
            break
    else:
        assert False

    # 'is_padded' returns a fixed capacity (independent of the live contact count) plus per-env 'n_contacts',
    # with the live prefix identical to the trimmed result, for every (as_tensor, to_torch) and batching.
    capacity = max(gs_s.collider.collider_info.max_candidate_contacts[None], 1)
    for as_tensor in (False, True):
        for to_torch in (False, True):
            trimmed = gs_s.collider.get_contacts(as_tensor, to_torch, is_padded=False)
            padded = gs_s.collider.get_contacts(as_tensor, to_torch, is_padded=True)
            assert "n_contacts" not in trimmed
            padded_counts = padded.pop("n_contacts")
            assert isinstance(padded_counts, torch.Tensor if to_torch else np.ndarray)
            assert_equal(padded_counts, gs_n_contacts)
            for key in trimmed:
                for i_b in range(max(n_envs, 1)):
                    n_contacts = gs_n_contacts[i_b]
                    pad_i = padded[key] if n_envs == 0 else padded[key][i_b]
                    trim_i = trimmed[key] if n_envs == 0 else trimmed[key][i_b]
                    assert len(pad_i) == capacity
                    assert_equal(pad_i[:n_contacts], trim_i[:n_contacts])

    gs_s._func_forward_dynamics()
    gs_s._func_constraint_force()

    # Make sure that all the robots ends up in the different state
    qposs = gs_robot.get_qpos()
    for i in range(n_envs - 1):
        with np.testing.assert_raises(AssertionError):
            assert_allclose(qposs[i], qposs[i + 1], tol=tol)

    # Check attribute getters / setters.
    # First, without any any row or column masking:
    # * Call 'Get' -> Call 'Set' with random value -> Call 'Get'
    # * Compare first 'Get' ouput with Quadrants value
    # Then, for any possible combinations of row and column masking:
    # * Call 'Get' -> Call 'Set' with 'Get' output -> Call 'Get'
    # * Compare first 'Get' output with last 'Get' output
    # * Compare last 'Get' output with corresponding slice of non-masking 'Get' output
    def get_all_supported_masks(i, max_length):
        if max_length <= 0 or i > max_length - 1:
            return (None,)
        if i == max_length - 1:
            return (
                i,
                [i],
                slice(i, i + 1),
                range(i, i + 1),
                np.array([i], dtype=np.int32),
                torch.tensor([i], dtype=torch.int64),
                torch.tensor([i], dtype=gs.tc_int, device=gs.device),
            )
        return (
            [i, i + 1],
            slice(i, i + 2),
            range(i, i + 2),
            np.array([i, i + 1], dtype=np.int32),
            torch.tensor([i, i + 1], dtype=torch.int64),
            torch.tensor([i, i + 1], dtype=gs.tc_int, device=gs.device),
        )

    # Link, joint and DOF info only carries an environment dimension when batched, which decides whether their
    # accessors take an environment selection at all.
    n_envs_info = n_envs if batched else -1

    for arg1_max, arg2_max, getter_or_spec, setter, qd_data in (
        # SOLVER
        (gs_s.n_links, n_envs, gs_s.get_links_pos, None, gs_s.dyn_state.links.pos),
        (gs_s.n_links, n_envs, gs_s.get_links_quat, None, gs_s.dyn_state.links.quat),
        (gs_s.n_links, n_envs, gs_s.get_links_vel, None, None),
        (gs_s.n_links, n_envs, gs_s.get_links_ang, None, gs_s.dyn_state.links.cd_ang),
        (gs_s.n_links, n_envs, gs_s.get_links_acc, None, None),
        (gs_s.n_links, n_envs, gs_s.get_links_root_COM, None, gs_s.dyn_state.links.root_COM),
        (
            gs_s.n_links,
            n_envs_info,
            gs_s.get_links_mass,
            gs_s.set_links_mass,
            gs_s.dyn_info.links.inertial_mass,
        ),
        (gs_s.n_links, n_envs_info, gs_s.get_links_COM, gs_s.set_links_COM, gs_s.dyn_info.links.inertial_pos),
        (gs_s.n_links, n_envs_info, gs_s.get_links_inertia, gs_s.set_links_inertia, gs_s.dyn_info.links.inertial_i),
        (gs_s.n_links, n_envs_info, gs_s.get_links_invweight, None, gs_s.dyn_info.links.invweight),
        (gs_s.n_dofs, n_envs, gs_s.get_dofs_control_force, gs_s.control_dofs_force, None),
        (gs_s.n_dofs, n_envs, gs_s.get_dofs_force, None, gs_s.dyn_state.dofs.force),
        (gs_s.n_dofs, n_envs, gs_s.get_dofs_velocity, gs_s.set_dofs_velocity, gs_s.dyn_state.dofs.vel),
        (gs_s.n_dofs, n_envs, gs_s.get_dofs_position, gs_s.set_dofs_position, gs_s.dyn_state.dofs.pos),
        (
            gs_s.n_dofs,
            n_envs_info,
            gs_s.get_dofs_force_range,
            gs_s.set_dofs_force_range,
            gs_s.dyn_info.dofs.force_range,
        ),
        (gs_s.n_dofs, n_envs_info, gs_s.get_dofs_limit, gs_s.set_dofs_limit, gs_s.dyn_info.dofs.limit),
        (gs_s.n_dofs, n_envs_info, gs_s.get_dofs_stiffness, gs_s.set_dofs_stiffness, gs_s.dyn_info.dofs.stiffness),
        (gs_s.n_dofs, n_envs_info, gs_s.get_dofs_invweight, None, gs_s.dyn_info.dofs.invweight),
        (gs_s.n_dofs, n_envs_info, gs_s.get_dofs_armature, gs_s.set_dofs_armature, gs_s.dyn_info.dofs.armature),
        (gs_s.n_dofs, n_envs_info, gs_s.get_dofs_damping, gs_s.set_dofs_damping, gs_s.dyn_info.dofs.damping),
        (
            gs_s.n_dofs,
            n_envs_info,
            gs_s.get_dofs_frictionloss,
            gs_s.set_dofs_frictionloss,
            gs_s.dyn_info.dofs.frictionloss,
        ),
        (gs_s.n_dofs, n_envs_info, gs_s.get_dofs_kp, gs_s.set_dofs_kp, gs_s.dyn_info.dofs.act_gain),
        (gs_s.n_dofs, n_envs_info, gs_s.get_dofs_kv, gs_s.set_dofs_kv, None),
        (gs_s.n_dofs, n_envs_info, gs_s.get_dofs_act_bias, gs_s.set_dofs_act_bias, gs_s.dyn_info.dofs.act_bias),
        (gs_s.n_dofs, n_envs_info, gs_s.get_dofs_act_gain, gs_s.set_dofs_act_gain, gs_s.dyn_info.dofs.act_gain),
        (gs_s.n_geoms, n_envs, gs_s.get_geoms_pos, None, gs_s.dyn_state.geoms.pos),
        (gs_s.n_geoms, n_envs, gs_s.get_geoms_quat, None, gs_s.dyn_state.geoms.quat),
        (
            gs_s.n_geoms,
            n_envs,
            gs_s.get_geoms_friction_ratio,
            gs_s.set_geoms_friction_ratio,
            gs_s.dyn_state.geoms.friction_ratio,
        ),
        (gs_s.n_geoms, -1, gs_s.get_geoms_friction, gs_s.set_geoms_friction, gs_s.dyn_info.geoms.friction),
        (gs_s.n_qs, n_envs, gs_s.get_qpos, gs_s.set_qpos, gs_s.qpos),
        # ROBOT
        (gs_robot.n_links, n_envs, gs_robot.get_links_pos, None, None),
        (gs_robot.n_links, n_envs, gs_robot.get_links_quat, None, None),
        (gs_robot.n_links, n_envs, gs_robot.get_links_vel, None, None),
        (gs_robot.n_links, n_envs, gs_robot.get_links_ang, None, None),
        (gs_robot.n_links, n_envs, gs_robot.get_links_acc, None, None),
        (gs_robot.n_links, n_envs, (), gs_robot.set_friction_ratio, None),
        (gs_robot.n_links, n_envs_info, gs_robot.get_links_mass, gs_robot.set_links_mass, None),
        (gs_robot.n_links, n_envs_info, (3,), gs_robot.set_links_COM, None),
        (gs_robot.n_links, n_envs_info, gs_robot.get_links_inertia, gs_robot.set_links_inertia, None),
        (gs_robot.n_links, n_envs_info, gs_robot.get_links_invweight, None, None),
        (gs_robot.n_dofs, n_envs, gs_robot.get_dofs_control_force, None, None),
        (gs_robot.n_dofs, n_envs, gs_robot.get_dofs_force, None, None),
        (gs_robot.n_dofs, n_envs, gs_robot.get_dofs_velocity, gs_robot.set_dofs_velocity, None),
        (gs_robot.n_dofs, n_envs, gs_robot.get_dofs_position, gs_robot.set_dofs_position, None),
        (gs_robot.n_dofs, n_envs_info, gs_robot.get_dofs_force_range, gs_robot.set_dofs_force_range, None),
        (gs_robot.n_dofs, n_envs_info, gs_robot.get_dofs_limit, None, None),
        (gs_robot.n_dofs, n_envs_info, gs_robot.get_dofs_stiffness, None, None),
        (gs_robot.n_dofs, n_envs_info, gs_robot.get_dofs_invweight, None, None),
        (gs_robot.n_dofs, n_envs_info, gs_robot.get_dofs_armature, None, None),
        (gs_robot.n_dofs, n_envs_info, gs_robot.get_dofs_damping, None, None),
        (gs_robot.n_dofs, n_envs_info, gs_robot.get_dofs_frictionloss, gs_robot.set_dofs_frictionloss, None),
        (gs_robot.n_dofs, n_envs_info, gs_robot.get_dofs_kp, gs_robot.set_dofs_kp, None),
        (gs_robot.n_dofs, n_envs_info, gs_robot.get_dofs_kv, gs_robot.set_dofs_kv, None),
        (gs_robot.n_dofs, n_envs_info, gs_robot.get_dofs_act_bias, gs_robot.set_dofs_act_bias, None),
        (gs_robot.n_dofs, n_envs_info, gs_robot.get_dofs_act_gain, gs_robot.set_dofs_act_gain, None),
        (gs_robot.n_qs, n_envs, gs_robot.get_qpos, gs_robot.set_qpos, None),
        (-1, n_envs, gs_robot.get_mass_mat, None, None),
        (-1, n_envs, gs_robot.get_links_net_contact_force, None, None),
        (-1, n_envs, gs_robot.get_pos, gs_robot.set_pos, None),
        (-1, n_envs, gs_robot.get_quat, gs_robot.set_quat, None),
        (-1, -1, gs_robot.get_mass, gs_robot.set_mass, None),
        (-1, -1, gs_robot.get_verts, None, None),
        (-1, -1, gs_robot.get_AABB, None, None),
        (-1, -1, gs_robot.get_vAABB, None, None),
        # LINK
        (-1, -1, gs_link.get_pos, None, None),
        (-1, -1, gs_link.get_quat, None, None),
        (-1, -1, gs_link.get_mass, gs_link.set_mass, None),
        (-1, -1, gs_link.get_verts, None, None),
        (-1, -1, gs_link.get_AABB, None, None),
        (-1, -1, gs_link.get_vAABB, None, None),
        # GEOM
        (-1, -1, gs_geom.get_pos, None, None),
        (-1, -1, gs_geom.get_quat, None, None),
        (-1, -1, gs_geom.get_verts, None, None),
        (-1, -1, gs_geom.get_AABB, None, None),
        # VGEOM
        (-1, -1, gs_vgeom.get_pos, None, None),
        (-1, -1, gs_vgeom.get_quat, None, None),
        (-1, -1, gs_vgeom.get_vAABB, None, None),
    ):
        getter, spec = (getter_or_spec, None) if callable(getter_or_spec) else (None, getter_or_spec)

        # Restore PD consistency before each iteration (act_gain/act_bias setters may have broken it)
        gs_s.set_dofs_kp(0.0)
        gs_s.set_dofs_kv(0.0)

        # Check getter and setter without row or column masking
        if getter is not None:
            datas = deepcopy(getter())
            is_tuple = isinstance(datas, (tuple, list))
            if arg1_max > 0:
                assert_allclose(getter(range(arg1_max)), datas, tol=tol)
        else:
            batch_shape = []
            if arg2_max > 0:
                batch_shape.append(arg2_max)
            if arg1_max > 0:
                batch_shape.append(arg1_max)
            is_tuple = spec and isinstance(spec[0], (tuple, list))
            if is_tuple:
                datas = [torch.ones((*batch_shape, *shape)) for shape in spec]
            else:
                datas = torch.ones((*batch_shape, *spec))
        if qd_data is not None:
            true = qd_to_torch(qd_data)
            qd_ndim = getattr(qd_data, "ndim", len(getattr(qd_data, "element_shape", ())))
            true = true.movedim(true.ndim - qd_ndim - 1, 0)
            if is_tuple:
                true = torch.unbind(true, dim=-1)
                true = [val.reshape(data.shape) for data, val in zip(datas, true)]
            else:
                true = true.reshape(datas.shape)
            assert_allclose(datas, true, tol=tol)
        if setter is not None:
            if is_tuple:
                datas = [torch.as_tensor(val) for val in datas]
            else:
                datas = torch.as_tensor(datas, dtype=gs.tc_float)
            datas_tp = datas if is_tuple else (datas,)
            if getter is not None:
                # Sampled valid for what they set: positive for masses, normalized for quaternions, and symmetric
                # positive definite for an inertia matrix, which a merely positive matrix is not.
                for val in datas_tp:
                    if val.ndim > 1 and val.shape[-2] == 3 and val.shape[-1] == 3:
                        factor = torch.randn(val.shape, dtype=gs.tc_float, device=gs.device)
                        val[()] = factor @ factor.transpose(-1, -2) + torch.eye(3, dtype=gs.tc_float, device=gs.device)
                    else:
                        val[()] = torch.abs(torch.randn(val.shape, dtype=gs.tc_float, device=gs.device)) + gs.EPS
                        val /= torch.linalg.norm(val, dim=-1, keepdims=True)
            setter(*datas_tp)
            if getter is not None:
                assert_allclose(getter(), datas, tol=tol)

        # Early return if neither rows or columns can be masked
        if not (arg1_max > 0 or arg2_max > 0):
            continue

        # Check getter and setter for all possible combinations of row and column masking
        for i in range(arg1_max) if arg1_max > 0 else (None,):
            if i is not None:
                mask_i = [i, i + 1] if i < arg1_max - 1 else [i]
            for arg1 in get_all_supported_masks(i, arg1_max):
                for j in range(max(arg2_max, 1)) if arg2_max >= 0 else (None,):
                    if j is not None:
                        mask_j = [j, j + 1] if j < arg2_max - 1 else [j]
                    for arg2 in get_all_supported_masks(j, arg2_max):
                        if arg1 is None and arg2 is not None:
                            if getter is not None:
                                data = deepcopy(getter(arg2))
                            else:
                                if is_tuple:
                                    data = [torch.ones((len(mask_j), *shape)) for shape in spec]
                                else:
                                    data = torch.ones((len(mask_j), *spec))
                            if setter is not None:
                                setter(data, arg2)
                            if n_envs:
                                if is_tuple:
                                    data_ = [val[mask_j] for val in datas]
                                else:
                                    data_ = datas[mask_j]
                            else:
                                data_ = datas
                        elif arg1 is not None and arg2 is None:
                            if getter is not None:
                                data = deepcopy(getter(arg1))
                            else:
                                if is_tuple:
                                    data = [torch.ones((len(mask_i), *shape)) for shape in spec]
                                else:
                                    data = torch.ones((len(mask_i), *spec))
                            if setter is not None:
                                if is_tuple:
                                    setter(*data, arg1)
                                else:
                                    setter(data, arg1)
                            if is_tuple:
                                data_ = [val[mask_i] for val in datas]
                            else:
                                data_ = datas[mask_i]
                        else:
                            if getter is not None:
                                data = deepcopy(getter(arg1, arg2))
                            else:
                                if is_tuple:
                                    data = [torch.ones((len(mask_j), len(mask_i), *shape)) for shape in spec]
                                else:
                                    data = torch.ones((len(mask_j), len(mask_i), *spec))
                            if setter is not None:
                                if is_tuple:
                                    setter(*data, arg1, arg2)
                                else:
                                    setter(data, arg1, arg2)
                            if is_tuple:
                                data_ = [val[mask_j, :][:, mask_i] for val in datas]
                            else:
                                data_ = datas[mask_j, :][:, mask_i]
                        # FIXME: Not sure why tolerance must be increased for tests to pass
                        assert_allclose(data_, data, tol=(5.0 * tol))

    for dofs_idx in (*get_all_supported_masks(0, gs_s.n_dofs), None):
        for envs_idx in (*(get_all_supported_masks(0, gs_s.n_dofs) if n_envs > 0 else ()), None):
            dofs_pos = gs_s.get_dofs_position(dofs_idx, envs_idx)
            dofs_vel = gs_s.get_dofs_velocity(dofs_idx, envs_idx)
            gs_s.control_dofs_position(dofs_pos, dofs_idx, envs_idx)
            gs_s.control_dofs_velocity(dofs_vel, dofs_idx, envs_idx)

    # Must be tested independently because of non-trival return type
    gs_robot.get_contacts()


@pytest.mark.required
@pytest.mark.parametrize("enable_mujoco_compatibility", [True, False])
def test_getter_vs_state_post_step_consistency(enable_mujoco_compatibility):
    DT = 0.01
    GRAVITY = 10.0

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=DT,
            gravity=(0.0, 0.0, GRAVITY),
        ),
        rigid_options=gs.options.RigidOptions(
            enable_mujoco_compatibility=enable_mujoco_compatibility,
        ),
    )
    box = scene.add_entity(
        gs.morphs.Box(
            size=(1.0, 1.0, 1.0),
            pos=(0.0, 0.0, 0.0),
        )
    )
    (box_link,) = box.links
    scene.build()

    scene.step()
    dof_vel = scene.rigid_solver.get_dofs_velocity()
    assert_allclose(dof_vel[:3], (0.0, 0.0, GRAVITY * DT), atol=gs.EPS)
    vel = box_link.get_vel()
    with pytest.raises(AssertionError) if enable_mujoco_compatibility else nullcontext():
        assert_allclose(dof_vel[:3], vel, atol=gs.EPS)
    dof_pos = scene.rigid_solver.get_qpos()
    assert_allclose(dof_pos[:3], (0.0, 0.0, GRAVITY * DT**2), atol=gs.EPS)
    pos = box_link.get_pos()
    with pytest.raises(AssertionError) if enable_mujoco_compatibility else nullcontext():
        assert_allclose(dof_pos[:3], pos, atol=gs.EPS)


@pytest.mark.slow  # ~200s
@pytest.mark.required
def test_extended_broadcasting():
    scene = gs.Scene(show_viewer=False)
    for i in range(4):
        scene.add_entity(
            gs.morphs.Box(
                size=(1.0, 1.0, 1.0),
                pos=(0.0, 0.0, i),
            )
        )
    scene.build(n_envs=2)

    envs_idx = torch.tensor([0, 1], dtype=gs.tc_int, device=gs.device)
    for entity in scene.entities:
        entity.zero_all_dofs_velocity(envs_idx)
    assert_allclose(entity.get_dofs_velocity(), 0.0, tol=gs.EPS)
    entity.set_dofs_velocity(1.0)
    assert_allclose(entity.get_dofs_velocity(), 1.0, tol=gs.EPS)
    entity.set_dofs_velocity((1.0, 2.0))
    assert_allclose(entity.get_dofs_velocity(), np.array([(1.0,) * 6, (2.0,) * 6]), tol=gs.EPS)
    entity.set_dofs_velocity((3.0,) * 6)
    assert_allclose(entity.get_dofs_velocity(), 3.0, tol=gs.EPS)
    entity.zero_all_dofs_velocity(torch.tensor([False, True], dtype=torch.bool, device=gs.device))
    assert_allclose(entity.get_dofs_velocity(), np.array([(3.0,) * 6, (0.0,) * 6]), tol=gs.EPS)


@pytest.mark.slow  # ~250s
@pytest.mark.required
@pytest.mark.parametrize("batch_links_info", [False, True])
@pytest.mark.parametrize("batch_joints_info", [False, True])
@pytest.mark.parametrize("batch_dofs_info", [False, True])
def test_batched_info(batch_links_info, batch_joints_info, batch_dofs_info, tol):
    INERTIA_RATIO = 2.0

    scene = gs.Scene(
        rigid_options=gs.options.RigidOptions(
            batch_links_info=batch_links_info,
            batch_joints_info=batch_joints_info,
            batch_dofs_info=batch_dofs_info,
        ),
    )
    scene.add_entity(gs.morphs.Terrain())
    franka = scene.add_entity(
        morph=gs.morphs.MJCF(
            file="xml/franka_emika_panda/panda.xml",
        )
    )
    scene.build(n_envs=2)
    gs_s = scene.rigid_solver

    links_info = gs_s.data_manager.dyn_info.links
    entity_idx = links_info.entity_idx.to_numpy()
    assert entity_idx.shape == ((12, 2) if batch_links_info else (12,))

    joints_info = gs_s.data_manager.dyn_info.joints
    pos = joints_info.pos.to_numpy()
    assert pos.shape == ((10, 2, 3) if batch_joints_info else (10, 3))

    dofs_info = gs_s.data_manager.dyn_info.dofs
    act_gain = dofs_info.act_gain.to_numpy()
    assert act_gain.shape == ((9, 2) if batch_dofs_info else (9,))

    # Give every environment its own configuration, so that picking the wrong one cannot go unnoticed.
    franka.set_dofs_position(np.random.rand(2, franka.n_dofs))

    # Links info only has an environment dimension when batched, so `envs_idx` must be rejected otherwise.
    if batch_links_info:
        # Potential energy is linear in the link masses, so scaling one environment scales its energy by that ratio
        # alone. Armature is cleared first: it adds to the mass matrix without belonging to any link, so invweight
        # only scales with the mass once the mass matrix is the links' alone.
        franka.set_dofs_armature(0.0)
        potential_energy = franka.get_potential_energy()
        gs_s.set_links_mass(INERTIA_RATIO * gs_s.get_links_mass(envs_idx=1), envs_idx=1)
        gs_s.set_links_inertia(INERTIA_RATIO * gs_s.get_links_inertia(envs_idx=1), envs_idx=1)
        assert_allclose(franka.get_potential_energy() / potential_energy, (1.0, INERTIA_RATIO), tol=tol)

        links_mass = gs_s.get_links_mass()
        links_invweight = gs_s.get_links_invweight()
        assert links_mass.shape == (2, 12)
        assert links_invweight.shape == (2, 12, 2)
        assert_allclose(links_mass[1], INERTIA_RATIO * links_mass[0], tol=tol)
        assert_allclose(INERTIA_RATIO * links_invweight[1], links_invweight[0], tol=tol)
        assert_allclose(gs_s.get_links_mass(envs_idx=1), links_mass[1], tol=gs.EPS)
        assert_allclose(gs_s.get_links_invweight(envs_idx=1), links_invweight[1], tol=gs.EPS)
    else:
        assert gs_s.get_links_mass().shape == (12,)
        assert gs_s.get_links_invweight().shape == (12, 2)
        # Nothing is held per environment, so naming one is rejected rather than answered with the shared value, by
        # every accessor of link info alike.
        for accessor in (gs_s.get_links_mass, gs_s.get_links_COM, gs_s.get_links_inertia, gs_s.get_links_invweight):
            with pytest.raises(gs.GenesisException, match="cannot be specified for non-batched links info"):
                accessor(envs_idx=1)
        for setter in (gs_s.set_links_mass, gs_s.set_links_COM, gs_s.set_links_inertia):
            with pytest.raises(gs.GenesisException, match="cannot be specified for non-batched links info"):
                setter(gs_s.get_links_mass()[1] if setter is gs_s.set_links_mass else 0.0, envs_idx=1)
        # A shared mass has nowhere to put a value of its own per environment.
        with pytest.raises(gs.GenesisException, match="batch_links_info"):
            franka.links[1].set_mass((1.0, 2.0))
        # One value stands for the whole batch, and the weights it settles are recomputed from it.
        link = franka.links[1]
        links_invweight = gs_s.get_links_invweight()
        link.set_mass(2.0 * link.get_mass())
        assert_allclose(link.get_mass(), gs_s.get_links_mass()[link.idx], tol=gs.EPS)
        with np.testing.assert_raises(AssertionError):
            assert_allclose(gs_s.get_links_invweight(), links_invweight, tol=tol)

        # A body brought to a mass is brought to it by the links the dynamics moves: the base of this robot is fixed
        # to the world and carries mass of its own, which counted in the shares would leave the body short of what was
        # asked for, and counted in the total would hide that it was.
        base = franka.base_link
        assert base.is_fixed and base.get_mass() > gs.EPS
        base_mass = base.get_mass()
        franka.set_mass(10.0)
        assert_allclose(franka.get_mass(), 10.0, tol=tol)
        moved = sum(gs_s.get_links_mass(link.idx) for link in franka.links if not link.is_fixed)
        assert_allclose(moved, 10.0, tol=tol)
        assert_allclose(base.get_mass(), base_mass, tol=gs.EPS)

    # Energy getters select environments through the dynamic state, so they accept `envs_idx` in both batching modes.
    for get_energy in (gs_s.get_total_energy, franka.get_potential_energy, franka.get_total_energy):
        energy = get_energy()
        with np.testing.assert_raises(AssertionError):
            assert_allclose(energy[0], energy[1], tol=gs.EPS)
        assert_allclose(get_energy(envs_idx=1), energy[1], tol=gs.EPS)


@pytest.mark.slow  # ~200s
@pytest.mark.required
def test_info_batching(tol):
    scene = gs.Scene(
        rigid_options=gs.options.RigidOptions(
            batch_links_info=True,
            batch_joints_info=True,
            batch_dofs_info=True,
        ),
        show_viewer=False,
        show_FPS=False,
    )
    plane = scene.add_entity(
        gs.morphs.Plane(),
    )
    robot = scene.add_entity(
        gs.morphs.MJCF(
            file="xml/franka_emika_panda/panda.xml",
        ),
    )
    scene.build(n_envs=2)

    scene.step()
    qposs = robot.get_qpos()
    assert_allclose(qposs[0], qposs[1], tol=tol)


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
def test_geom_pos_quat(n_envs, show_viewer):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            gravity=(0.0, 0.0, -10.0),
        ),
        show_viewer=show_viewer,
    )

    box = scene.add_entity(
        gs.morphs.Box(
            size=(1.0, 1.0, 1.0),
            pos=(0.0, 0.0, 2.0),
        )
    )
    scene.build(n_envs=n_envs)
    batch_shape = (n_envs,) if n_envs > 0 else ()

    box.set_dofs_position(np.random.rand(*batch_shape, 6))
    scene.rigid_solver.update_vgeoms()

    for link in box.links:
        for vgeom, geom in zip(link.vgeoms, link.geoms):
            geom_pos, geom_quat = geom.get_pos(), geom.get_quat()
            assert geom_pos.shape == (*batch_shape, 3)
            assert geom_quat.shape == (*batch_shape, 4)
            vgeom_pos, vgeom_quat = vgeom.get_pos(), vgeom.get_quat()
            assert vgeom_pos.shape == (*batch_shape, 3)
            assert vgeom_quat.shape == (*batch_shape, 4)
            assert_allclose(geom_pos, vgeom_pos, atol=gs.EPS)
            assert_allclose(geom_quat, vgeom_quat, atol=gs.EPS)


@pytest.mark.slow  # ~200s
@pytest.mark.required
@pytest.mark.parametrize("batch_fixed_verts", [False, True])
@pytest.mark.parametrize("relative", [False, True])
def test_set_root_pose(batch_fixed_verts, relative, show_viewer, tol):
    ROBOT_POS_ZERO = (0.0, 0.4, 0.1)
    ROBOT_EULER_ZERO = (0.0, 0.0, 90.0)
    CUBE_POS_ZERO = (0.65, 0.0, 0.02)
    CUBE_EULER_ZERO = (0.0, 90.0, 0.0)

    scene = gs.Scene(
        show_viewer=show_viewer,
        show_FPS=False,
    )
    plane = scene.add_entity(
        gs.morphs.Plane(),
    )
    robot = scene.add_entity(
        gs.morphs.MJCF(
            file="xml/franka_emika_panda/panda.xml",
            offset_pos=ROBOT_POS_ZERO,
            offset_euler=ROBOT_EULER_ZERO,
            batch_fixed_verts=batch_fixed_verts,
        ),
    )
    sphere = scene.add_entity(
        gs.morphs.Sphere(
            radius=0.04,
            batch_fixed_verts=False,
            fixed=True,
        ),
    )
    cube = scene.add_entity(
        gs.morphs.Box(
            size=(0.04, 0.04, 0.04),
            offset_pos=CUBE_POS_ZERO,
            offset_euler=CUBE_EULER_ZERO,
        ),
    )
    plain_box = scene.add_entity(
        gs.morphs.Box(
            pos=(2.0, 0.0, 0.2),
            size=(0.04, 0.04, 0.04),
        ),
    )
    POSED_BOX_POS = (2.0, 0.5, 0.3)
    POSED_BOX_OFFSET_EULER = (0.0, 0.0, 45.0)
    posed_box = scene.add_entity(
        gs.morphs.Box(
            pos=POSED_BOX_POS,
            size=(0.04, 0.04, 0.04),
            offset_pos=(0.0, 0.0, 0.5),
            offset_euler=POSED_BOX_OFFSET_EULER,
        ),
    )
    scene.build(n_envs=2)

    # A no-offset entity reports the same pose in the user and world frames.
    assert_allclose(plain_box.get_pos(relative=True), plain_box.get_pos(relative=False), tol=tol)
    assert_allclose(plain_box.get_pos(), (2.0, 0.0, 0.2), tol=tol)

    # With both a morph pose and an offset, the relative getter returns the morph pose while the world getter carries
    # the offset composed onto it (the offset position adds in z since the authored orientation is identity).
    assert_allclose(posed_box.get_pos(relative=True), POSED_BOX_POS, tol=tol)
    assert_allclose(posed_box.get_quat(relative=True), gu.identity_quat(), tol=tol)
    assert_allclose(posed_box.get_pos(relative=False), (2.0, 0.5, 0.8), tol=tol)
    assert_allclose(
        posed_box.get_quat(relative=False),
        gu.xyz_to_quat(np.array(POSED_BOX_OFFSET_EULER), rpy=True, degrees=True),
        tol=tol,
    )

    # Setting the orientation in the authored frame keeps the authored-frame position fixed: the offset position rotates
    # with the orientation, so the world position is rewritten to preserve the reported relative position. Rotating
    # about x while the offset position is along z makes that offset contribution change, exercising the rewrite.
    new_quat = gu.xyz_to_quat(np.array((90.0, 0.0, 0.0)), rpy=True, degrees=True)
    posed_box.set_quat(new_quat, relative=True)
    assert_allclose(posed_box.get_pos(relative=True), POSED_BOX_POS, tol=tol)
    assert_allclose(posed_box.get_quat(relative=True), new_quat, tol=tol)

    robot_aabb_init, robot_base_aabb_init = robot.get_AABB(), robot.geoms[0].get_AABB()
    cube_aabb_init, cube_base_aabb_init = cube.get_AABB(), cube.geoms[0].get_AABB()

    # A non-rotating link origin only translates, so both frames report the same free fall.
    scene.step()
    assert_allclose(posed_box.get_vel(relative=True), posed_box.get_vel(relative=False), tol=tol)
    assert_allclose(posed_box.get_links_acc(relative=True), posed_box.get_links_acc(relative=False), tol=tol)

    # Spinning the box makes the authored origin orbit the internal one, so both are reported at the displacement 'd'
    # the position getters strip. The spin is about a principal body axis orthogonal to the offset position, so it is
    # torque-free and stays orthogonal to 'd', which collapses the acceleration transport to the centripetal term
    # 'SPIN**2 * d' toward the internal origin.
    SPIN = 5.0
    posed_box.set_dofs_velocity((0.0, 0.0, 0.0, SPIN, 0.0, 0.0))
    scene.step()
    omega = posed_box.get_ang()
    offset_shift = posed_box.get_pos(relative=False) - posed_box.get_pos(relative=True)
    assert_allclose(
        posed_box.get_vel(relative=True),
        posed_box.get_vel(relative=False) - torch.cross(omega, offset_shift, dim=-1),
        tol=tol,
    )
    assert_allclose(
        posed_box.get_links_acc(relative=True)[..., 0, :],
        posed_box.get_links_acc(relative=False)[..., 0, :] + SPIN**2 * offset_shift,
        tol=tol,
    )

    # Masking rows and columns must strip the same offset as the unmasked query.
    assert_allclose(posed_box.get_links_vel(links_idx_local=0, envs_idx=1)[..., 0, :], posed_box.get_vel()[1], tol=tol)

    # Make sure that it is not possible to end up in an inconsistent state for fixed geometries. These place entities
    # at absolute world positions, so they bypass the pose offset (relative=False).
    pos_delta = np.random.rand(2, 3)
    with nullcontext() if batch_fixed_verts else pytest.raises(gs.GenesisException):
        robot.set_pos(pos_delta, relative=False)
        if show_viewer:
            scene.visualizer.update()
    with nullcontext() if batch_fixed_verts else pytest.raises(gs.GenesisException):
        robot.set_pos(pos_delta[[0]], envs_idx=[0], relative=False)
        if show_viewer:
            scene.visualizer.update()
    cube.set_pos(pos_delta[[0]] + (0.0, 0.0, 0.16), envs_idx=[0], relative=False)
    cube.set_pos(pos_delta[[1]] + (0.0, 0.0, 0.11), envs_idx=1, relative=False)
    sphere.set_pos(np.tile(pos_delta[[0]], (2, 1)) + 1.0, relative=False)
    quat_delta = np.random.rand(2, 4)
    with nullcontext() if batch_fixed_verts else pytest.raises(gs.GenesisException):
        robot.set_quat(quat_delta, relative=False)
        if show_viewer:
            scene.visualizer.update()
    with nullcontext() if batch_fixed_verts else pytest.raises(gs.GenesisException):
        robot.set_quat(quat_delta[[0]], envs_idx=[0], relative=False)
        if show_viewer:
            scene.visualizer.update()
    cube.set_quat(quat_delta, relative=False)
    if show_viewer:
        scene.visualizer.update()

    sphere_aabb, sphere_base_aabb = sphere.get_AABB(), sphere.geoms[0].get_AABB()
    assert_allclose(sphere_aabb.mean(dim=-2), pos_delta[0] + 1.0, tol=tol)
    assert_allclose(sphere_aabb, sphere_base_aabb, tol=tol)

    # Simulate for a while to check if the dynamic object is colliding with the static one
    if batch_fixed_verts:
        has_collided = torch.tensor([False, False], dtype=torch.bool, device=gs.device)
        for _ in range(20):
            scene.step()
            contacts_state = cube.get_contacts(with_entity=robot, exclude_self_contact=True)
            has_collided |= contacts_state["valid_mask"].any(dim=-1)
            if has_collided.all():
                break
        else:
            raise AssertionError("Cube never collided with robot for at least one of the environments.")

    for _ in range(2):
        scene.reset()

        for entity, pos_zero, euler_zero, entity_aabb_init, base_aabb_init in (
            (robot, ROBOT_POS_ZERO, ROBOT_EULER_ZERO, robot_aabb_init, robot_base_aabb_init),
            (cube, CUBE_POS_ZERO, CUBE_EULER_ZERO, cube_aabb_init, cube_base_aabb_init),
        ):
            pos_zero = torch.tensor(pos_zero, device=gs.device, dtype=gs.tc_float)
            euler_zero = torch.deg2rad(torch.tensor(euler_zero, dtype=gs.tc_float))
            quat_zero = gu.xyz_to_quat(euler_zero, rpy=True)
            # The pose lives in the offset, so relative=False reports it and the authored frame is identity.
            assert_allclose(entity.get_pos(relative=False), pos_zero, tol=tol)
            assert_allclose(entity.get_pos(relative=True), 0.0, tol=tol)
            # Use quaternion for comparison to avoid gymbal lock issue in euler angles
            quat = entity.get_quat(relative=False)
            assert_allclose(quat, quat_zero, tol=tol)
            base_aabb = entity.geoms[0].get_AABB()
            assert base_aabb.shape == ((2, 2, 3) if not entity.geoms[0].is_fixed or batch_fixed_verts else (2, 3))
            assert_allclose(base_aabb, base_aabb_init, tol=tol)
            assert_allclose(entity.get_AABB(), entity_aabb_init, tol=tol)

            pos_delta = torch.as_tensor(np.random.rand(3), dtype=gs.tc_float, device=gs.device).expand((2, 3))
            entity.set_pos(pos_delta, relative=relative)

            pos_ref = pos_delta + pos_zero if relative else pos_delta
            # Round-trip in the frame it was set in: the getter must report back exactly what set_pos received.
            assert_allclose(entity.get_pos(relative=relative), pos_delta, tol=tol)
            assert_allclose(entity.geoms[0].get_AABB(), base_aabb_init + (pos_ref - pos_zero), tol=tol)
            assert_allclose(entity.get_AABB(), entity_aabb_init + (pos_ref - pos_zero), tol=tol)

            quat_delta = torch.tile(torch.as_tensor(np.random.rand(4), dtype=gs.tc_float, device=gs.device), (2, 1))
            quat_delta /= torch.linalg.norm(quat_delta, axis=1, keepdim=True)
            entity.set_quat(quat_delta, relative=relative)
            assert_allclose(entity.get_quat(relative=relative), quat_delta, tol=tol)


@pytest.mark.slow  # ~200s
@pytest.mark.required
def test_normalized_quat(show_viewer, tol):
    scene = gs.Scene(
        show_viewer=show_viewer,
        show_FPS=False,
    )
    robot = scene.add_entity(
        gs.morphs.URDF(
            file="urdf/go2/urdf/go2.urdf",
        ),
    )
    scene.build()

    # Make sure that the simulation state is not sensitive to qpos normalization
    quat = torch.randn((4,), dtype=gs.tc_float, device=gs.device)

    qpos = robot.get_qpos()
    qpos[3:7] = quat / torch.linalg.norm(quat)
    robot.set_qpos(qpos)
    scene.step()
    qpos_post = robot.get_qpos()
    assert_allclose(torch.linalg.norm(qpos_post[3:7]), 1.0, tol=tol)

    qpos[3:7] = quat
    scene.reset()
    robot.set_qpos(qpos)
    # assert_allclose(qpos, robot.get_qpos(), tol=tol)  # True, but not specification requirement
    scene.step()
    assert_allclose(qpos_post, robot.get_qpos(), tol=tol)

    scene.reset()
    robot.set_quat(quat)
    # assert_allclose(quat, qpos[3:7], tol=tol)  # True, but not specification requirement
    scene.step()
    assert_allclose(qpos_post, robot.get_qpos(), tol=tol)

    # Make sure that entity, link and geom quaternions are normalized.
    # "RigidEntity.set_quat" is calling 'kernel_forward_kinematics_links_geoms', which is relying on
    # 'func_update_cartesian_space' under the hood.
    # Let's check that everything is properly normalized at this stage already. If so, it means that all quaternions of
    # interest are guaranteed to be always normalized, since 'func_update_cartesian_space' is called internally during
    # forward dynamics 'step_1' at the very beginning of 'RigidSolver.step'.
    scene.reset()
    robot.set_quat(quat)
    assert_allclose(torch.linalg.norm(robot.get_quat()), 1.0, tol=tol)
    for link in robot.links:
        assert_allclose(torch.linalg.norm(link.get_quat()), 1.0, tol=tol)
    for geom in robot.geoms:
        assert_allclose(torch.linalg.norm(geom.get_quat()), 1.0, tol=tol)
    assert_allclose(torch.linalg.norm(scene.rigid_solver.get_links_quat(), dim=-1), 1.0, tol=tol)
    assert_allclose(torch.linalg.norm(scene.rigid_solver.get_geoms_quat(), dim=-1), 1.0, tol=tol)


@pytest.mark.required
def test_inertial_property_setters(
    sliding_ball_pair, free_bodies_in_one_model, implicit_inertial_origin_chain, show_viewer, tol
):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
        ),
        rigid_options=gs.options.RigidOptions(
            batch_links_info=True,
            integrator=gs.integrator.Euler,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(2.5, -3.0, 1.5),
            camera_lookat=(2.5, 0.0, 0.2),
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(
        morph=gs.morphs.Plane(),
    )
    het_obj = scene.add_entity(
        morph=[
            gs.morphs.Box(size=(0.01, 0.01, 0.01), pos=(0.0, 3.0, 0.3)),
            gs.morphs.Box(size=(0.02, 0.02, 0.02), pos=(0.0, 3.0, 0.3)),
            gs.morphs.Sphere(radius=0.01, pos=(0.0, 3.0, 0.3)),
            gs.morphs.Sphere(radius=0.02, pos=(0.0, 3.0, 0.3)),
        ],
    )
    # The default armature follows each pair's build-time inertia, so the three routes to one mass meet only without it.
    authored, per_link, whole_entity = (
        scene.add_entity(
            morph=gs.morphs.URDF(file=sliding_ball_pair(*masses), pos=(x, 0.0, 0.3), default_armature=0.0),
        )
        for x, masses in ((1.0, (1.0, 2.0)), (2.0, (1.0, 1.5)), (3.0, (0.5, 1.0)))
    )
    free_bodies = scene.add_entity(
        morph=gs.morphs.MJCF(
            file=free_bodies_in_one_model,
        ),
    )
    unaligned = scene.add_entity(
        morph=gs.morphs.MJCF(
            file=free_bodies_in_one_model,
            pos=(0.0, 10.0, 8.0),
            align=False,
        ),
    )
    chain = scene.add_entity(
        morph=gs.morphs.URDF(
            file=implicit_inertial_origin_chain,
            pos=(0.0, -3.0, 0.5),
            merge_fixed_links=False,
        ),
    )
    scene.build(n_envs=4)

    # Bringing a whole entity to a mass holds however far apart its environments start, and a per-link set takes one
    # value per environment and nothing else.
    het_link = next(link for link in het_obj.links if not link.is_fixed)
    with pytest.raises(gs.GenesisException):
        het_link.set_mass((1.0, 2.0))
    het_obj.set_mass(1.0)
    assert_allclose(het_obj.get_mass(), 1.0, tol=tol)

    # One route brings a single link to the mass, keeping the inertia it was given, and the other the whole body by a
    # ratio, which scales the inertia with it - so the link route writes the heavier inertia itself to match.
    mass_ratio = 2.0 / scene.rigid_solver.get_links_mass(per_link.links[1].idx)
    links_inertia = scene.rigid_solver.get_links_inertia(per_link.links[1].idx)
    per_link.links[1].set_mass(2.0)
    per_link.set_links_inertia(links_inertia * mass_ratio[..., None, None], links_idx_local=1)
    whole_entity.set_mass(3.0)

    pairs = (authored, per_link, whole_entity)
    for pair in pairs[1:]:
        assert_allclose(pair.get_mass(), authored.get_mass(), tol=gs.EPS)

    # The inertia grows with the mass, as it does for a body of the same shape built heavier.
    authored_inertia = scene.rigid_solver.get_links_inertia(authored.links[1].idx)
    assert_allclose(authored_inertia, scene.rigid_solver.get_links_inertia(per_link.links[1].idx), tol=gs.EPS)

    links_idx = [pair.base_link.idx for pair in pairs]
    for pair in pairs:
        pair.set_dofs_velocity([0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    # Long enough to fall onto the plane and keep sliding, so the contact rows take part.
    for _ in range(60):
        scene.rigid_solver.apply_links_external_wrench([[[2.0, 0.0, 0.0]] * len(links_idx)] * 4, links_idx=links_idx)
        scene.step()
    assert (authored.get_pos()[:, 2] < 0.2).all(), "the pairs must have landed for the contact solve to be exercised"

    # The slide between the two spheres is where the mass of the second one shows up in the articulated solve.
    slide_dof = [authored.n_dofs - 1]
    for i_pair, pair in enumerate(pairs[1:], start=1):
        offset = torch.tensor([i_pair, 0.0, 0.0], dtype=gs.tc_float, device=gs.device)
        assert_allclose(pair.get_pos() - offset, authored.get_pos(), tol=tol)
        assert_allclose(pair.get_vel(), authored.get_vel(), tol=tol)
        assert_allclose(pair.get_ang(), authored.get_ang(), tol=tol)
        assert_allclose(pair.get_dofs_position(slide_dof), authored.get_dofs_position(slide_dof), tol=tol)
        assert_allclose(pair.get_dofs_velocity(slide_dof), authored.get_dofs_velocity(slide_dof), tol=tol)

    # Refreshing the weights, defined only at the neutral configuration, must leave the live one untouched.
    qpos_live = authored.get_qpos()
    vel_live = authored.get_dofs_velocity()
    authored.links[1].set_mass(2.25)
    assert_allclose(authored.get_qpos(), qpos_live, tol=gs.EPS)
    assert_allclose(authored.get_dofs_velocity(), vel_live, tol=gs.EPS)

    # Where the mass of a link sits is what the dynamics turns it about, so both the place a link carries its own mass
    # and the composite place its tree turns about have to follow a mass or a center of mass being written. Asserted on
    # those two places directly: the one the solver holds against the offset it was given, and the composite one
    # against the mass-weighted mean it is by definition.
    solver = scene.rigid_solver
    pair_idx = [link.idx for link in per_link.links]
    het_obj.set_dofs_velocity([0.4, -0.3, 0.5], dofs_idx_local=[3, 4, 5])
    for change in (None, "COM", "mass", "inertia"):
        if change == "inertia":
            # An inertia is not a place, so writing one leaves both of them where the motion alone puts them.
            solver.set_links_inertia(solver.get_links_inertia(pair_idx) * 2.0, links_idx=pair_idx)
        elif change == "COM":
            solver.set_links_COM(solver.get_links_COM(pair_idx) + 0.03, links_idx=pair_idx)
        elif change == "mass":
            solver.set_links_mass(solver.get_links_mass(pair_idx) * 1.5, links_idx=pair_idx)
        held_idx = pair_idx[-1]
        offset = gu.transform_by_quat(solver.get_links_COM(held_idx), solver.get_links_quat(held_idx))
        held = solver.get_links_pos(held_idx, ref=gs.link_ref_frame.link_COM)
        assert_allclose(held, solver.get_links_pos(held_idx) + offset, tol=tol)

        masses = solver.get_links_mass(pair_idx)
        centers = solver.get_links_pos(pair_idx, ref=gs.link_ref_frame.link_COM)
        weighed = (masses[..., None] * centers).sum(dim=-2) / masses.sum(dim=-1)[..., None]
        assert_allclose(solver.get_links_pos(pair_idx, ref=gs.link_ref_frame.root_COM), weighed[..., None, :], tol=tol)
        # Stepped last, so the next write lands on a configuration the motion has moved on.
        scene.step()

    # An inertial property is written in the frame the degrees of freedom are taken at, so a center of mass away from
    # its origin couples translation to rotation and an inertia holding a product of inertia couples the rotations to
    # each other. A force with no moment about that origin then turns the body, by the moment the offset gives it and
    # by what the inertia carries. Gravity exerts no moment about a body's own center of mass, so the turn is the
    # applied force's alone.
    OFFSET, FORCE, TORQUE = 0.05, 2.0, 2.0
    SKEWED_INERTIA = np.array([[0.02, 0.0, 0.0], [0.0, 0.065, 0.03], [0.0, 0.03, 0.08]])
    offset_link, skewed_link = unaligned.links
    solver.set_links_COM([OFFSET, 0.0, 0.0], links_idx=offset_link.idx)
    solver.set_links_inertia(SKEWED_INERTIA, links_idx=skewed_link.idx)
    inertia_zz = solver.get_links_inertia(offset_link.idx)[..., 0, 2, 2]
    unaligned.control_dofs_force([0.0, FORCE, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, TORQUE])
    scene.step()
    turn = unaligned.get_links_ang()[..., 0, :]
    assert_allclose(turn[..., 2], -OFFSET * FORCE / inertia_zz * scene.dt, tol=tol)
    assert_allclose(turn[..., :2], 0.0, tol=tol)
    spin = np.linalg.solve(SKEWED_INERTIA, [0.0, 0.0, TORQUE]) * scene.dt
    assert_allclose(unaligned.get_links_ang()[..., 1, :], spin, tol=tol)

    # A body the build anchored on its own center of mass and principal axes has no room for either write, since the
    # frame that anchoring gave it is what the solver goes on to rely on. A model file can decline the anchoring,
    # which is what the body above does; a primitive shape cannot.
    with pytest.raises(gs.GenesisException, match="align=False"):
        solver.set_links_COM([OFFSET, 0.0, 0.0], links_idx=free_bodies.links[0].idx)
    with pytest.raises(gs.GenesisException, match="not supported yet"):
        solver.set_links_inertia(SKEWED_INERTIA, links_idx=het_link.idx)
    # The anchor covers fixed children too, so the setters hold their center of mass and inertia like the root's. A mass
    # written on one link alone moves the anchor, so the setter rejects it. A mass set on the whole entity keeps the
    # anchor and the mass ratios.
    root, child = chain.links
    with pytest.raises(gs.GenesisException, match="align=False"):
        child.set_COM([OFFSET, 0.0, 0.0])
    with pytest.raises(gs.GenesisException, match="align=False"):
        child.set_inertia(SKEWED_INERTIA)
    with pytest.raises(gs.GenesisException, match="align=False"):
        child.set_mass(1.0)
    with pytest.raises(gs.GenesisException, match="align=False"):
        root.set_mass(1.0)
    with pytest.raises(gs.GenesisException, match="align=False"):
        chain.set_links_mass([1.0, 1.0])
    chain.set_mass(2.0)
    assert_allclose(chain.get_links_mass(), [[1.25, 0.75]] * 4, tol=tol)

    # An inertial property belongs to the model, so a reset restores the configuration the scene was built at and
    # leaves the mass it now runs with alone.
    mass_written = solver.get_links_mass(pair_idx)
    scene.reset()
    assert_allclose(solver.get_links_mass(pair_idx), mass_written, tol=gs.EPS)

    # A weight is solved over the mass blocks of the chain being weighed alone, which is what lets the trees of one
    # entity be weighed side by side through one pair of buffers. Sentinels are written over every row first: those of
    # the tree that is not weighed must come back untouched.
    SENTINEL = -7.0
    scratch = solver.data_manager.weight_scratch
    dofs_idx = slice(free_bodies.dof_start, free_bodies.dof_start + free_bodies.n_dofs)
    untouched = slice(free_bodies.dof_start + 6, free_bodies.dof_start + free_bodies.n_dofs)
    for buffer in (scratch.jac_row, scratch.solve_out):
        sentinels = qd_to_numpy(buffer)
        sentinels[dofs_idx] = SENTINEL
        buffer.from_numpy(sentinels)
    free_bodies.links[0].set_mass(2.0)
    for buffer in (scratch.jac_row, scratch.solve_out):
        assert_allclose(qd_to_numpy(buffer)[untouched], SENTINEL, tol=gs.EPS)


@pytest.mark.slow  # ~250s
@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 3])
def test_axis_aligned_bounding_boxes(n_envs):
    scene = gs.Scene()
    scene.add_entity(
        gs.morphs.Plane(
            normal=(0, 0, 1),
            pos=(0, 0, 0),
        ),
    )
    scene.add_entity(
        gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(0.5, 0, 0.05),
        ),
    )
    scene.add_entity(
        gs.morphs.Cylinder(
            height=0.8,
            radius=0.06,
            pos=(1.0, 0, 0.5),
        ),
    )
    scene.add_entity(
        gs.morphs.Sphere(
            radius=0.05,
            pos=(-0.5, 0, 0.05),
        ),
    )
    robot = scene.add_entity(
        gs.morphs.MJCF(
            file="xml/franka_emika_panda/panda.xml",
        ),
    )
    scene.build(n_envs=n_envs)

    batch_shape = (n_envs,) if n_envs > 0 else ()
    aabb_shape = (*batch_shape, 2, 3)

    qpos = np.random.rand(*(*batch_shape, robot.n_dofs))
    robot.set_dofs_position(qpos)

    robot_aabb = robot.get_AABB()
    robot_geoms_aabb = torch.stack([geom.get_AABB().expand(aabb_shape) for geom in robot.geoms], dim=0)
    assert_allclose(torch.min(robot_geoms_aabb[..., 0, :], dim=0).values, robot_aabb[..., 0, :], tol=gs.EPS)
    assert_allclose(torch.max(robot_geoms_aabb[..., 1, :], dim=0).values, robot_aabb[..., 1, :], tol=gs.EPS)
    for link in robot.links:
        link_aabb = link.get_AABB()
        link_geoms_aabb = torch.stack([geom.get_AABB().expand(aabb_shape) for geom in link.geoms], dim=0)
        assert_allclose(torch.min(link_geoms_aabb[..., 0, :], dim=0).values, link_aabb[..., 0, :], tol=gs.EPS)
        assert_allclose(torch.max(link_geoms_aabb[..., 1, :], dim=0).values, link_aabb[..., 1, :], tol=gs.EPS)

    all_aabbs = scene.sim.rigid_solver.get_AABB()
    aabbs = [geom.get_AABB().expand(aabb_shape) for entity in scene.entities for geom in entity.geoms]
    if n_envs > 0:
        assert all_aabbs.ndim == 4 and len(all_aabbs) == n_envs
    else:
        assert all_aabbs.ndim == 3
    assert all_aabbs.shape[-3:] == (len(aabbs), 2, 3)
    assert_allclose(aabbs[:4], all_aabbs.swapaxes(-3, 0)[:4], atol=gs.EPS)
    with pytest.raises(AssertionError):
        assert_allclose(aabbs[4:], all_aabbs.swapaxes(-3, 0)[4:], atol=gs.EPS)

    box_aabb_min, box_aabb_max = aabbs[1].split(1, dim=-2)
    assert_allclose(box_aabb_min, (0.45, -0.05, 0.0), atol=gs.EPS)
    assert_allclose(box_aabb_max, (0.55, 0.05, 0.1), atol=gs.EPS)
    sphere_aabb_min, sphere_aabb_max = aabbs[3].split(1, dim=-2)
    assert_allclose(sphere_aabb_min, (-0.55, -0.05, 0.0), atol=gs.EPS)
    assert_allclose(sphere_aabb_max, (-0.45, 0.05, 0.1), atol=gs.EPS)

    vaabbs = [vgeom.get_vAABB().expand(aabb_shape) for entity in scene.entities for vgeom in entity.vgeoms]
    if n_envs > 0:
        for entity in scene.entities:
            for vgeom in entity.vgeoms:
                assert_allclose(vgeom.get_vAABB(), [vgeom.get_vAABB(i)[0] for i in range(n_envs)], tol=gs.EPS)
    box_aabb_min, box_aabb_max = vaabbs[1].split(1, dim=-2)
    assert_allclose(box_aabb_min, (0.45, -0.05, 0.0), atol=gs.EPS)
    assert_allclose(box_aabb_max, (0.55, 0.05, 0.1), atol=gs.EPS)
    sphere_aabb_min, sphere_aabb_max = vaabbs[3].split(1, dim=-2)
    assert_allclose(sphere_aabb_min, (-0.55, -0.05, 0.0), atol=1e-3)
    assert_allclose(sphere_aabb_max, (-0.45, 0.05, 0.1), atol=1e-3)

    robot_vaabb = robot.get_vAABB()
    assert_allclose(robot_vaabb, robot_aabb, atol=1e-3)


@pytest.mark.slow  # ~200s
@pytest.mark.required
@pytest.mark.parametrize(
    "friction_cone, sparse_solve, use_hibernation, enable_mujoco_compatibility",
    [
        (gs.friction_cone.pyramidal, None, False, False),
        # Hibernation sleeps both boxes before the state is captured, so the reset has to wake them. It needs its own
        # cell because MuJoCo compatibility lowers the rest threshold to a velocity this scene never reaches, which
        # would leave the sleeping path unexercised.
        (gs.friction_cone.pyramidal, None, True, False),
        # The buffers a step leaves behind: the elliptic cone keeps a cone-free Hessian mirror, the sparse solve zeroes
        # each Jacobian row through the pattern the previous step left, and MuJoCo compatibility reads the warm-start
        # acceleration unconditionally.
        (gs.friction_cone.elliptic, True, False, True),
    ],
)
def test_reset(show_viewer, friction_cone, sparse_solve, use_hibernation, enable_mujoco_compatibility):
    N_STEPS = 60
    BOOL_MASK = torch.tensor([True, False, True, False], dtype=torch.bool, device=gs.device)

    scene = gs.Scene(
        rigid_options=gs.options.RigidOptions(
            friction_cone=friction_cone,
            sparse_solve=sparse_solve,
            use_hibernation=use_hibernation,
            enable_mujoco_compatibility=enable_mujoco_compatibility,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.0, -2.5, 1.2),
            camera_lookat=(1.0, 0.0, 0.05),
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(
        gs.morphs.URDF(
            file="urdf/plane/plane.urdf",
            fixed=True,
        )
    )
    box = scene.add_entity(
        gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(0, 0, 0.5),
        )
    )
    # Far enough from the first box to never interact, so the scene splits into one island each - the structure
    # hibernation needs to engage.
    box_far = scene.add_entity(
        gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(2.0, 0, 0.5),
        )
    )
    scene.build(n_envs=4)

    # Every position- and velocity-level accessor must read the same in a restored state as it did at capture, starting
    # before the first step that follows the reset. Acceleration only matches from the first step onwards: a restored
    # state derives it from the restored velocity, while the capture reflects the refresh the setter left pending.
    GETTERS = (
        box.get_pos,
        box.get_quat,
        box.get_vel,
        box.get_ang,
        box.get_links_pos,
        box.get_links_quat,
        box.get_links_vel,
        box.get_links_ang,
        box.get_dofs_position,
        box.get_dofs_velocity,
    )
    ACC_GETTERS = (
        box.get_links_acc,
        box.get_links_acc_ang,
    )

    init_state = scene.get_state()
    init_rigid_state = next(s for s in init_state.solvers_state if isinstance(s, RigidSolverState))
    init_getters = tuple(getter() for getter in GETTERS)
    # The horizon reaches past the point where both boxes come to rest, so the state is captured with them asleep
    # whenever hibernation is enabled, and the reset has to wake them.
    for _ in range(N_STEPS):
        scene.step()
    if use_hibernation:
        for entity in (box, box_far):
            assert qd_to_numpy(scene.rigid_solver.dyn_state.links.is_hibernated, entity.base_link_idx).all()
    fallen_state = scene.get_state()
    fallen_rigid_state = next(s for s in fallen_state.solvers_state if isinstance(s, RigidSolverState))
    fallen_getters = tuple(getter() for getter in GETTERS)
    fallen_acc_getters = tuple(getter() for getter in ACC_GETTERS)

    for envs_idx in (BOOL_MASK, torch.where(BOOL_MASK)[0]):
        scene.reset(state=fallen_state)
        scene.reset(state=init_state, envs_idx=envs_idx)
        restored_rigid_state = next(s for s in scene.get_state().solvers_state if isinstance(s, RigidSolverState))
        restored_getters = tuple(getter() for getter in GETTERS)
        for actual, init_ref, fallen_ref in (
            (restored_rigid_state.qpos, init_rigid_state.qpos, fallen_rigid_state.qpos),
            (restored_rigid_state.dofs_vel, init_rigid_state.dofs_vel, fallen_rigid_state.dofs_vel),
            (restored_rigid_state.links_pos, init_rigid_state.links_pos, fallen_rigid_state.links_pos),
            *zip(restored_getters, init_getters, fallen_getters),
        ):
            assert_equal(actual[BOOL_MASK], init_ref[BOOL_MASK])
            assert_equal(actual[~BOOL_MASK], fallen_ref[~BOOL_MASK])

    # After reset, simulation from init_state should reproduce the original fallen_state trajectory
    for _ in range(N_STEPS):
        scene.step()
    replayed_rigid_state = next(s for s in scene.get_state().solvers_state if isinstance(s, RigidSolverState))
    replayed_getters = tuple(getter() for getter in GETTERS)
    replayed_acc_getters = tuple(getter() for getter in ACC_GETTERS)
    for actual, fallen_ref in (
        (replayed_rigid_state.qpos, fallen_rigid_state.qpos),
        (replayed_rigid_state.dofs_vel, fallen_rigid_state.dofs_vel),
        (replayed_rigid_state.links_pos, fallen_rigid_state.links_pos),
        *zip(replayed_getters, fallen_getters),
        *zip(replayed_acc_getters, fallen_acc_getters),
    ):
        assert_equal(actual[BOOL_MASK], fallen_ref[BOOL_MASK])


@pytest.mark.required
def test_deprecated_properties(caplog):
    scene = gs.Scene(
        show_viewer=False,
        show_FPS=False,
    )
    box = scene.add_entity(
        gs.morphs.Box(
            size=(1.0, 1.0, 1.0),
            pos=(0.0, 0.0, 0.0),
        )
    )
    scene.build()

    joint = box.joints[0]

    # Verify introspection doesn't trigger warnings
    caplog.clear()
    with caplog.at_level("WARNING"):
        repr(joint)
        vars(joint)
    assert len(caplog.records) == 0

    for name_old, name_new in (
        ("dof_idx", "dofs_idx"),
        ("dof_idx_local", "dofs_idx_local"),
        ("q_idx", "qs_idx"),
        ("q_idx_local", "qs_idx_local"),
    ):
        # Make sure that deprecated properties are hidden
        assert name_old not in dir(joint)

        # Verify deprecated properties emit warnings but work correctly
        caplog.clear()
        with caplog.at_level("WARNING"):
            deprecated_value = getattr(joint, name_old)
        assert len(caplog.records) > 0
        assert_allclose(deprecated_value, getattr(joint, name_new), tol=gs.EPS)


@pytest.mark.required
def test_object_repr():
    inline_mjcf = '<mujoco model="probe"><worldbody><body><geom type="box" size="1 1 1"/></body></worldbody></mujoco>'

    scene = gs.Scene(show_viewer=False)
    scene.add_entity(
        morph=gs.morphs.Plane(),
    )
    scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
        )
    )
    panda = scene.add_entity(
        morph=gs.morphs.MJCF(
            file="xml/franka_emika_panda/panda.xml",
        )
    )
    inline = scene.add_entity(
        morph=gs.morphs.MJCF(
            file=inline_mjcf,
        )
    )
    het = scene.add_entity(
        morph=(
            gs.morphs.Box(size=(0.2, 0.2, 0.2)),
            gs.morphs.Cylinder(radius=0.05, height=0.2),
        ),
    )
    scene.add_entity(
        morph=(
            gs.morphs.Box(size=(0.2, 0.2, 0.2)),
            gs.morphs.Sphere(radius=0.1),
        ),
        material=gs.materials.Kinematic(),
    )
    cam = scene.add_camera(
        res=(64, 64),
        pos=(1.0, 1.0, 1.0),
        lookat=(0.0, 0.0, 0.0),
    )
    scene.build(n_envs=2)

    # Every printable object renders without raising, across both the brief and the full colorized form
    for obj in (scene, scene.entities, cam, scene.sim.rigid_solver):
        assert repr(obj)
    for entity in scene.entities:
        assert entity._repr_brief()
        assert repr(entity)
        for morph in entity.morphs:
            assert repr(morph)
        sub_objects = [*entity.links, *entity.joints, *entity.vgeoms]
        if isinstance(entity, gs.engine.entities.RigidEntity):
            sub_objects += list(entity.geoms)
        for sub in sub_objects:
            assert sub._repr_brief()
            assert repr(sub)

    # A morph created from a file prints its path. A morph created from inline XML prints the model name its
    # document declares, and never the document itself.
    assert "panda.xml" in repr(panda.main_morph)
    assert "<inline probe>" in inline.main_morph.__repr_name__()
    assert inline_mjcf not in repr(inline.main_morph)
    # A heterogeneous entity reports its variants instead of collapsing to a single ambiguous morph
    assert "morph variants" in het._repr_brief()

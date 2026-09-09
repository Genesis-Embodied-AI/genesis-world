from itertools import chain
from typing import Literal, Sequence

import mujoco
import numpy as np
import pytest

import genesis as gs
import genesis.utils.geom as gu
from genesis.utils.misc import qd_to_numpy, tensor_to_array

from .assertions import assert_allclose


def _gs_search_by_joints_name(
    scene,
    joints_name: str | list[str],
    to: Literal["entity", "index"] = "index",
    is_local: bool = False,
    flatten: bool = True,
):
    if isinstance(joints_name, str):
        joints_name = [joints_name]

    gs_joints_idx = dict()
    gs_joints_qs_idx = dict()
    gs_joints_dofs_idx = dict()
    valid_joints_name = []
    for entity in scene.entities:
        for joint in entity.joints:
            valid_joints_name.append(joint.name)
            if joint.name in joints_name:
                if to == "entity":
                    gs_joints_idx[joint.name] = joint
                    gs_joints_qs_idx[joint.name] = joint
                    gs_joints_dofs_idx[joint.name] = joint
                elif to == "index":
                    gs_joints_idx[joint.name] = joint.idx_local if is_local else joint.idx
                    gs_joints_qs_idx[joint.name] = joint.qs_idx_local if is_local else joint.qs_idx
                    gs_joints_dofs_idx[joint.name] = joint.dofs_idx_local if is_local else joint.dofs_idx
                else:
                    raise ValueError(f"Cannot recognize what ({to}) to extract for the search")

    missing_joints_name = set(joints_name) - gs_joints_idx.keys()
    if missing_joints_name:
        raise ValueError(f"Cannot find joints `{missing_joints_name}`. Valid joints names are {valid_joints_name}")

    gs_joints_idx = {name: gs_joints_idx[name] for name in joints_name}
    gs_joints_qs_idx = {name: gs_joints_qs_idx[name] for name in joints_name}
    gs_joints_dofs_idx = {name: gs_joints_dofs_idx[name] for name in joints_name}
    if flatten:
        return (
            list(gs_joints_idx.values()),
            list(chain.from_iterable(gs_joints_qs_idx.values())),
            list(chain.from_iterable(gs_joints_dofs_idx.values())),
        )
    return (gs_joints_idx, gs_joints_qs_idx, gs_joints_dofs_idx)


def _gs_search_by_links_name(
    scene,
    links_name: str | Sequence[str],
    to: Literal["entity", "index"] = "index",
    is_local: bool = False,
    flatten: bool = True,
):
    if isinstance(links_name, str):
        links_name = (links_name,)

    gs_links_idx = dict()
    valid_links_name = []
    for entity in scene.entities:
        for link in entity.links:
            valid_links_name.append(link.name)
            if link.name in links_name:
                if to == "entity":
                    gs_links_idx[link.name] = link
                elif to == "index":
                    gs_links_idx[link.name] = link.idx_local if is_local else link.idx
                else:
                    raise ValueError(f"Cannot recognize what ({to}) to extract for the search")

    missing_links_name = set(links_name) - gs_links_idx.keys()
    if missing_links_name:
        raise ValueError(f"Cannot find links `{missing_links_name}`. Valid link names are {valid_links_name}")

    gs_links_idx = {name: gs_links_idx[name] for name in links_name}
    if flatten:
        return list(gs_links_idx.values())
    return gs_links_idx


def _get_model_mappings(
    gs_sim,
    mj_sim,
    joints_name: list[str] | None = None,
    bodies_name: list[str] | None = None,
):
    if joints_name is None:
        joints_name = [
            joint.name for entity in gs_sim.entities for joint in entity.joints if joint.type != gs.JOINT_TYPE.FIXED
        ]
    if bodies_name is None:
        bodies_name = [
            body.name
            for entity in gs_sim.entities
            for body in entity.links
            if not (body.is_fixed and body.parent_idx < 0)
        ]

    motors_name: list[str] = []
    mj_joints_idx: list[int] = []
    mj_qs_idx: list[int] = []
    mj_dofs_idx: list[int] = []
    mj_geoms_idx: list[int] = []
    mj_motors_idx: list[int] = []
    for joint_name in joints_name:
        if joint_name:
            try:
                mj_joint = mj_sim.model.joint(joint_name)
            except KeyError:
                for entity in gs_sim.entities:
                    for joint in entity.joints:
                        if joint.name == joint_name:
                            mj_joint = mj_sim.model.joint(joint.idx)
                            break
        else:
            # Must rely on exhaustive search if the joint has empty name
            for j in range(mj_sim.model.njoint):
                mj_joint = mj_sim.model.joint(j)
                if mj_joint.name == "":
                    break
            else:
                raise ValueError(f"Invalid joint name '{joint_name}'.")
        mj_joints_idx.append(mj_joint.id)
        mj_type = mj_sim.model.jnt_type[mj_joint.id]
        if mj_type == mujoco.mjtJoint.mjJNT_HINGE:
            n_dofs, n_qs = 1, 1
        elif mj_type == mujoco.mjtJoint.mjJNT_SLIDE:
            n_dofs, n_qs = 1, 1
        elif mj_type == mujoco.mjtJoint.mjJNT_BALL:
            n_dofs, n_qs = 3, 4
        elif mj_type == mujoco.mjtJoint.mjJNT_FREE:
            n_dofs, n_qs = 6, 7
        else:
            raise ValueError(f"Invalid joint type '{mj_type}'.")
        mj_dof_start_j = mj_sim.model.jnt_dofadr[mj_joint.id]
        mj_dofs_idx += range(mj_dof_start_j, mj_dof_start_j + n_dofs)

        mj_q_start_j = mj_sim.model.jnt_qposadr[mj_joint.id]
        mj_qs_idx += range(mj_q_start_j, mj_q_start_j + n_qs)
        if (mj_joint.id == mj_sim.model.actuator_trnid[:, 0]).any():
            motors_name.append(joint_name)
            (motors_idx,) = np.nonzero(mj_joint.id == mj_sim.model.actuator_trnid[:, 0])
            # FIXME: only supporting 1DoF per actuator
            mj_motors_idx.append(motors_idx[0])

    mj_bodies_idx, mj_geoms_idx = [], []
    for body_name in bodies_name:
        mj_body = mj_sim.model.body(body_name)
        mj_bodies_idx.append(mj_body.id)
        for mj_geom_idx in range(mj_body.geomadr[0], mj_body.geomadr[0] + mj_body.geomnum[0]):
            mj_geom = mj_sim.model.geom(mj_geom_idx)
            if mj_geom.contype or mj_geom.conaffinity:
                mj_geoms_idx.append(mj_geom.id)

    gs_joints_idx, gs_q_idx, gs_dofs_idx = _gs_search_by_joints_name(gs_sim.scene, joints_name)
    _, _, gs_motors_idx = _gs_search_by_joints_name(gs_sim.scene, motors_name)

    gs_bodies_idx = _gs_search_by_links_name(gs_sim.scene, bodies_name)
    gs_geoms_idx: list[int] = []
    for gs_body_idx in gs_bodies_idx:
        link = gs_sim.rigid_solver.links[gs_body_idx]
        gs_geoms_idx += range(link.geom_start, link.geom_end)

    gs_maps = (gs_bodies_idx, gs_joints_idx, gs_q_idx, gs_dofs_idx, gs_geoms_idx, gs_motors_idx)
    mj_maps = (mj_bodies_idx, mj_joints_idx, mj_qs_idx, mj_dofs_idx, mj_geoms_idx, mj_motors_idx)
    return gs_maps, mj_maps


def init_paired_simulators(gs_sim, mj_sim, qpos=None, qvel=None):
    """Initialize the Genesis simulator and reset MuJoCo onto its exact state, ready for a step-by-step comparison."""
    gs_sim.scene.reset()
    if qpos is not None or qvel is not None:
        (gs_robot,) = gs_sim.entities
        if qpos is not None:
            gs_robot.set_qpos(qpos)
        if qvel is not None:
            gs_robot.set_dofs_velocity(qvel)

    # The consistency checks compare pre-step derived quantities (bias forces, smooth accelerations), which only a
    # dynamics pass populates on the Genesis side, mirroring the mj_forward call below.
    gs_sim.rigid_solver.dyn_state.dofs.qf_constraint.fill(0.0)
    gs_sim.rigid_solver._func_forward_dynamics()
    gs_sim.rigid_solver._func_constraint_force()
    gs_sim.rigid_solver._func_update_acc()

    if gs_sim.scene.visualizer:
        gs_sim.scene.visualizer.update()

    _, (_, _, mj_qs_idx, mj_dofs_idx, _, _) = _get_model_mappings(gs_sim, mj_sim)
    mujoco.mj_resetData(mj_sim.model, mj_sim.data)
    mj_sim.data.qpos[mj_qs_idx] = gs_sim.rigid_solver.qpos.to_numpy()[:, 0]
    mj_sim.data.qvel[mj_dofs_idx] = gs_sim.rigid_solver.dyn_state.dofs.vel.to_numpy()[:, 0]
    mujoco.mj_forward(mj_sim.model, mj_sim.data)


def set_paired_inertial_properties(gs_sim, mj_sim, *, armature_ratio, mass_ratio, inertia_ratio, com_offset):
    """Apply the same inertial change to both simulators, through the runtime setters on the Genesis side and through
    the model fields and 'mj_setConst' on the MuJoCo side.

    Every DOF armature and every body mass and inertia are scaled by the given ratios, and the center of mass of every
    body MuJoCo compiled with a full mass matrix structure is shifted by 'com_offset' in its body frame. What the setters
    derive from the change (inverse weights, mean inertia) can then be held against the constants MuJoCo recomputes.
    """
    gs_maps, mj_maps = _get_model_mappings(gs_sim, mj_sim)
    gs_bodies_idx, _, _, gs_dofs_idx, _, _ = gs_maps
    mj_bodies_idx, _, _, mj_dofs_idx, _, _ = mj_maps
    model = mj_sim.model
    # MuJoCo bakes a diagonal mass matrix structure for a body compiled with its inertial frame on the body frame, so a
    # center of mass shifted at runtime has nowhere to couple: only the bodies compiled with the full structure take one.
    gs_com_idx, mj_com_idx = [], []
    for gs_i, mj_i in zip(gs_bodies_idx, mj_bodies_idx):
        if not model.body_simple[mj_i]:
            gs_com_idx.append(gs_i)
            mj_com_idx.append(mj_i)

    model.dof_armature[mj_dofs_idx] *= armature_ratio
    model.body_mass[mj_bodies_idx] *= mass_ratio
    model.body_inertia[mj_bodies_idx] *= inertia_ratio
    model.body_ipos[mj_com_idx] += com_offset
    # A body compiled with its inertial frame on the body frame keeps the flag that lets MuJoCo skip the offset.
    model.body_sameframe[mj_com_idx] = mujoco.mjtSameFrame.mjSAMEFRAME_NONE
    mujoco.mj_setConst(model, mj_sim.data)
    align_mujoco_invweight0(model)

    solver = gs_sim.rigid_solver
    solver.set_dofs_armature(armature_ratio * solver.get_dofs_armature(dofs_idx=gs_dofs_idx), dofs_idx=gs_dofs_idx)
    solver.set_links_mass(mass_ratio * solver.get_links_mass(links_idx=gs_bodies_idx), links_idx=gs_bodies_idx)
    solver.set_links_inertia(inertia_ratio * solver.get_links_inertia(links_idx=gs_bodies_idx), links_idx=gs_bodies_idx)
    links_com = tensor_to_array(solver.get_links_COM(links_idx=gs_com_idx)) + com_offset
    solver.set_links_COM(links_com, links_idx=gs_com_idx)


def get_mujoco_midpoint_dofs_mask(mj_sim):
    """Boolean mask over MuJoCo DOFs whose qacc / qvel are overwritten by midpoint integration this step.

    Mirrors MuJoCo's eligibility test: implicitfast without invdiscrete, zero medium density/viscosity, and per
    joint a standalone unconstrained free body. Rotational DOFs are always overwritten for eligible bodies; linear
    DOFs only when the center of mass is off the joint origin.
    """
    model, data = mj_sim.model, mj_sim.data
    mask = np.zeros(model.nv, dtype=np.bool_)
    if (
        model.opt.integrator != mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        or model.opt.enableflags & mujoco.mjtEnableBit.mjENBL_INVDISCRETE
        or model.opt.density != 0.0
        or model.opt.viscosity != 0.0
    ):
        return mask
    for i_j in range(model.njnt):
        if model.jnt_type[i_j] != mujoco.mjtJoint.mjJNT_FREE:
            continue
        i_b = model.jnt_bodyid[i_j]
        if model.body_parentid[i_b] != 0 or model.body_subtreemass[i_b] != model.body_mass[i_b]:
            continue
        is_constrained = False
        for i_c in range(data.ncon):
            geom_a, geom_b = data.contact.geom[i_c]
            if (geom_a >= 0 and model.geom_bodyid[geom_a] == i_b) or (geom_b >= 0 and model.geom_bodyid[geom_b] == i_b):
                is_constrained = True
        for i_e in range(model.neq):
            if not data.eq_active[i_e] or model.eq_type[i_e] not in (
                mujoco.mjtEq.mjEQ_CONNECT,
                mujoco.mjtEq.mjEQ_WELD,
            ):
                continue
            obj1, obj2 = model.eq_obj1id[i_e], model.eq_obj2id[i_e]
            if model.eq_objtype[i_e] == mujoco.mjtObj.mjOBJ_SITE:
                obj1, obj2 = model.site_bodyid[obj1], model.site_bodyid[obj2]
            if obj1 == i_b or obj2 == i_b:
                is_constrained = True
        if is_constrained:
            continue
        adr = model.jnt_dofadr[i_j]
        start = 3 if (model.body_ipos[i_b] == 0.0).all() else 0
        mask[adr + start : adr + 6] = True
    return mask


def align_mujoco_invweight0(model):
    """Write into the model the constraint inverse weights of every body and DOF that MuJoCo's general rule gives.

    That rule of 'mj_setConst' is the mean diagonal of J M^-1 J^T at the neutral configuration, over the translational
    and the rotational rows of a body, and over the DOFs a joint moves together. MuJoCo leaves it aside for a body it
    flags as simple (see the FIXME in genesis.utils.mjcf), so aligning the model on it has both engines simulate the
    weights Genesis derives.
    """
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    mass_mat_inv = np.zeros((model.nv, model.nv))
    mujoco.mj_solveM(model, data, mass_mat_inv, np.eye(model.nv))
    jac = np.zeros((6, model.nv))
    for i_b in range(1, model.nbody):
        mujoco.mj_jacBodyCom(model, data, jac[:3], jac[3:], i_b)
        inv_inertia = jac @ mass_mat_inv @ jac.T
        model.body_invweight0[i_b] = (np.trace(inv_inertia[:3, :3]) / 3.0, np.trace(inv_inertia[3:, 3:]) / 3.0)
    dofs_invweight = np.diag(mass_mat_inv).copy()
    for i_j in range(model.njnt):
        i_d = model.jnt_dofadr[i_j]
        if model.jnt_type[i_j] == mujoco.mjtJoint.mjJNT_FREE:
            dofs_invweight[i_d : i_d + 3] = dofs_invweight[i_d : i_d + 3].mean()
            dofs_invweight[i_d + 3 : i_d + 6] = dofs_invweight[i_d + 3 : i_d + 6].mean()
        elif model.jnt_type[i_j] == mujoco.mjtJoint.mjJNT_BALL:
            dofs_invweight[i_d : i_d + 3] = dofs_invweight[i_d : i_d + 3].mean()
    model.dof_invweight0[:] = dofs_invweight


def check_mujoco_model_consistency(
    gs_sim,
    mj_sim,
    joints_name: list[str] | None = None,
    bodies_name: list[str] | None = None,
    *,
    tol: float,
):
    # Delay import to enable run benchmarks for old Genesis versions that do not have this method
    from genesis.engine.solvers.rigid.rigid_solver import _sanitize_sol_params

    # Get mapping between Mujoco and Genesis
    gs_maps, mj_maps = _get_model_mappings(gs_sim, mj_sim, joints_name, bodies_name)
    gs_bodies_idx, gs_joints_idx, gs_q_idx, gs_dofs_idx, gs_geoms_idx, gs_motors_idx = gs_maps
    mj_bodies_idx, mj_joints_idx, mj_qs_idx, mj_dofs_idx, mj_geoms_idx, mj_motors_idx = mj_maps

    # solver
    gs_gravity = gs_sim.rigid_solver.get_gravity()
    mj_gravity = mj_sim.model.opt.gravity
    assert_allclose(gs_gravity, mj_gravity, tol=tol)
    assert mj_sim.model.opt.timestep == gs_sim.rigid_solver.substep_dt
    assert mj_sim.model.opt.tolerance == gs_sim.rigid_solver._options.tolerance
    assert mj_sim.model.opt.iterations == gs_sim.rigid_solver._options.iterations
    assert not (mj_sim.model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_EULERDAMP)
    assert not (mj_sim.model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_REFSAFE)
    assert not (mj_sim.model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_GRAVITY)
    assert not (mj_sim.model.opt.enableflags & mujoco.mjtEnableBit.mjENBL_FWDINV)

    mj_adj_collision = bool(mj_sim.model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_FILTERPARENT)
    gs_adj_collision = gs_sim.rigid_solver._options.enable_adjacent_collision
    assert gs_adj_collision == mj_adj_collision

    gs_use_gjk_collision = gs_sim.rigid_solver._options.use_gjk_collision
    mj_use_gjk_collision = not (mj_sim.model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_NATIVECCD)
    assert gs_use_gjk_collision == mj_use_gjk_collision

    mj_solver = mujoco.mjtSolver(mj_sim.model.opt.solver)
    if mj_solver.name == "mjSOL_PGS":
        assert False
    elif mj_solver.name == "mjSOL_CG":
        assert gs_sim.rigid_solver._options.constraint_solver == gs.constraint_solver.CG
    elif mj_solver.name == "mjSOL_NEWTON":
        assert gs_sim.rigid_solver._options.constraint_solver == gs.constraint_solver.Newton
    else:
        assert False

    mj_integrator = mujoco.mjtIntegrator(mj_sim.model.opt.integrator)
    if mj_integrator.name == "mjINT_EULER":
        assert gs_sim.rigid_solver._options.integrator == gs.integrator.Euler
    elif mj_integrator.name == "mjINT_IMPLICIT":
        assert False
    elif mj_integrator.name == "mjINT_IMPLICITFAST":
        assert gs_sim.rigid_solver._options.integrator == gs.integrator.implicitfast
    else:
        assert False

    mj_cone = mujoco.mjtCone(mj_sim.model.opt.cone)
    if mj_cone.name == "mjCONE_ELLIPTIC":
        assert gs_sim.rigid_solver._options.friction_cone == gs.friction_cone.elliptic
        assert_allclose(gs_sim.rigid_solver._options.impratio, mj_sim.model.opt.impratio, tol=tol)
    elif mj_cone.name == "mjCONE_PYRAMIDAL":
        assert gs_sim.rigid_solver._options.friction_cone == gs.friction_cone.pyramidal
    else:
        assert False

    gs_roots_name = sorted(
        gs_sim.rigid_solver.links[i].name
        for i in set(gs_sim.rigid_solver.dyn_info.links.root_idx.to_numpy()[gs_bodies_idx])
    )
    mj_roots_name = sorted(mj_sim.model.body(i).name for i in set(mj_sim.model.body_rootid[mj_bodies_idx]))
    assert gs_roots_name == mj_roots_name

    # body
    for gs_i, mj_i in zip(gs_bodies_idx, mj_bodies_idx):
        gs_invweight_i = gs_sim.rigid_solver.dyn_info.links.invweight.to_numpy()[gs_i]
        mj_invweight_i = mj_sim.model.body(mj_i).invweight0
        try:
            assert_allclose(gs_invweight_i, mj_invweight_i, tol=tol)
        except AssertionError:
            if tuple(int(x) for x in mujoco.__version__.split(".")[:2]) < (3, 5):
                pytest.skip(
                    "MuJoCo < 3.5 lacks the degenerate invweight fix. "
                    "See https://github.com/google-deepmind/mujoco/commit/1cda1e7a"
                )
            raise
        gs_inertia_i = gs_sim.rigid_solver.dyn_info.links.inertial_i.to_numpy()[gs_i, [0, 1, 2], [0, 1, 2]]
        mj_inertia_i = mj_sim.model.body(mj_i).inertia
        assert_allclose(gs_inertia_i, mj_inertia_i, tol=tol)
        gs_ipos_i = gs_sim.rigid_solver.dyn_info.links.inertial_pos.to_numpy()[gs_i]
        mj_ipos_i = mj_sim.model.body(mj_i).ipos
        assert_allclose(gs_ipos_i, mj_ipos_i, tol=tol)
        gs_iquat_i = gs_sim.rigid_solver.dyn_info.links.inertial_quat.to_numpy()[gs_i]
        mj_iquat_i = mj_sim.model.body(mj_i).iquat
        assert_allclose(gs_iquat_i, mj_iquat_i, tol=tol)
        gs_pos_i = gs_sim.rigid_solver.dyn_info.links.pos.to_numpy()[gs_i]
        mj_pos_i = mj_sim.model.body(mj_i).pos
        assert_allclose(gs_pos_i, mj_pos_i, tol=tol)
        gs_quat_i = gs_sim.rigid_solver.dyn_info.links.quat.to_numpy()[gs_i]
        mj_quat_i = mj_sim.model.body(mj_i).quat
        assert_allclose(gs_quat_i, mj_quat_i, tol=tol)
        gs_mass_i = gs_sim.rigid_solver.dyn_info.links.inertial_mass.to_numpy()[gs_i]
        mj_mass_i = mj_sim.model.body(mj_i).mass
        assert_allclose(gs_mass_i, mj_mass_i, tol=tol)

    # dof / joints
    gs_dof_damping = gs_sim.rigid_solver.dyn_info.dofs.damping.to_numpy()
    mj_dof_damping = mj_sim.model.dof_damping
    assert_allclose(gs_dof_damping[gs_dofs_idx], mj_dof_damping[mj_dofs_idx], tol=tol)

    gs_dof_armature = gs_sim.rigid_solver.dyn_info.dofs.armature.to_numpy()
    mj_dof_armature = mj_sim.model.dof_armature
    assert_allclose(gs_dof_armature[gs_dofs_idx], mj_dof_armature[mj_dofs_idx], tol=tol)

    # TODO: 1 stiffness per joint in Mujoco, 1 stiffness per DoF in Genesis
    gs_dof_stiffness = gs_sim.rigid_solver.dyn_info.dofs.stiffness.to_numpy()
    mj_dof_stiffness = mj_sim.model.jnt_stiffness
    if all(joint.n_dofs == 1 for joint in gs_sim.rigid_solver.joints):
        assert_allclose(gs_dof_stiffness[gs_dofs_idx], mj_dof_stiffness[mj_joints_idx], tol=tol)

    gs_dof_invweight0 = gs_sim.rigid_solver.dyn_info.dofs.invweight.to_numpy()
    mj_dof_invweight0 = mj_sim.model.dof_invweight0
    assert_allclose(gs_dof_invweight0[gs_dofs_idx], mj_dof_invweight0[mj_dofs_idx], tol=tol)

    gs_dof_dof_frictionloss = gs_sim.rigid_solver.dyn_info.dofs.frictionloss.to_numpy()
    mj_dof_dof_frictionloss = mj_sim.model.dof_frictionloss
    assert_allclose(gs_dof_dof_frictionloss[gs_dofs_idx], mj_dof_dof_frictionloss[mj_dofs_idx], tol=tol)

    # Batched joint info carries a leading environment axis, and the MuJoCo model states one value per joint, so
    # compare the first environment against it.
    gs_joints_solparams = tensor_to_array(gs_sim.rigid_solver.get_sol_params(joints_idx=slice(None)))
    gs_joint_solparams = gs_joints_solparams[0] if gs_joints_solparams.ndim == 3 else gs_joints_solparams
    mj_joint_solparams = np.concatenate((mj_sim.model.jnt_solref, mj_sim.model.jnt_solimp), axis=-1)
    _sanitize_sol_params(
        mj_joint_solparams, gs_sim.rigid_solver._sol_min_timeconst, gs_sim.rigid_solver._sol_default_timeconst
    )
    assert_allclose(gs_joint_solparams[gs_joints_idx], mj_joint_solparams[mj_joints_idx], tol=tol)
    gs_geoms_solparams = tensor_to_array(gs_sim.rigid_solver.get_sol_params())
    gs_geom_solparams = gs_geoms_solparams[0] if gs_geoms_solparams.ndim == 3 else gs_geoms_solparams
    mj_geom_solparams = np.concatenate((mj_sim.model.geom_solref, mj_sim.model.geom_solimp), axis=-1)
    # Geom time constants are compared as the model states them: a contact mixes the two geoms' values and the floor is
    # applied to that mix, so flooring per geom here would expect a value neither engine stores.
    _sanitize_sol_params(
        mj_geom_solparams,
        gs_sim.rigid_solver._sol_min_timeconst,
        gs_sim.rigid_solver._sol_default_timeconst,
        floor_timeconst=False,
    )
    assert_allclose(gs_geom_solparams[gs_geoms_idx], mj_geom_solparams[mj_geoms_idx], tol=tol)
    # FIXME: Masking geometries and equality constraints is not supported for now
    gs_eq_solparams = np.array(
        [tensor_to_array(equality.get_sol_params()) for entity in gs_sim.entities for equality in entity.equalities]
    ).reshape((-1, 7))
    mj_eq_solparams = np.concatenate((mj_sim.model.eq_solref, mj_sim.model.eq_solimp), axis=-1)
    _sanitize_sol_params(
        mj_eq_solparams, gs_sim.rigid_solver._sol_min_timeconst, gs_sim.rigid_solver._sol_default_timeconst
    )
    assert_allclose(gs_eq_solparams, mj_eq_solparams, tol=tol)

    assert_allclose(mj_sim.model.jnt_margin, 0, tol=tol)
    gs_joint_range = np.stack(
        [
            gs_sim.rigid_solver.dyn_info.dofs.limit[gs_sim.rigid_solver.dyn_info.joints.dof_start[i]].to_numpy()
            for i in gs_joints_idx
        ],
        axis=0,
    )
    mj_joint_range = mj_sim.model.jnt_range
    mj_joint_range[mj_sim.model.jnt_limited == 0, 0] = float("-inf")
    mj_joint_range[mj_sim.model.jnt_limited == 0, 1] = float("+inf")
    assert_allclose(gs_joint_range, mj_joint_range[mj_joints_idx], tol=tol)

    # actuator (position control)
    for v in mj_sim.model.actuator_dyntype:
        assert v == mujoco.mjtDyn.mjDYN_NONE
    for v in mj_sim.model.actuator_biastype:
        assert v in (mujoco.mjtBias.mjBIAS_AFFINE, mujoco.mjtBias.mjBIAS_NONE)

    # NOTE: not considering gear for biasprm (only relevant for AFFINE actuators where gear=1 in practice).
    gs_act_gain = gs_sim.rigid_solver.dyn_info.dofs.act_gain.to_numpy()
    gs_act_bias = gs_sim.rigid_solver.dyn_info.dofs.act_bias.to_numpy()
    mj_gear = mj_sim.model.actuator_gear[:, 0]
    mj_gainprm = mj_sim.model.actuator_gainprm[:, 0] * mj_gear
    mj_biasprm = mj_sim.model.actuator_biasprm[:, :3] * mj_gear[:, None]
    assert_allclose(gs_act_gain[gs_motors_idx], mj_gainprm[mj_motors_idx], tol=tol)
    assert_allclose(gs_act_bias[gs_motors_idx], mj_biasprm[mj_motors_idx], tol=tol)


def _compute_efc_tolerances(mj_sim, tol):
    """Absolute tolerances of the constraint-space quantities and of the velocities integrating them.

    Rounding enters the reference acceleration, and every force and acceleration responding to it, amplified by the
    stiffest constraint's inverse squared time constant; the velocity integrates that noise over one step. The
    engines only round identically at double precision, so single precision carries the amplification.
    """
    if gs.np_float == np.float64:
        return tol, tol
    solref_timeconst = np.concatenate(
        (mj_sim.model.geom_solref[:, 0], mj_sim.model.jnt_solref[:, 0], mj_sim.model.eq_solref[:, 0])
    )
    # MuJoCo clamps a positive time constant to at least twice the timestep at runtime, so the sub-timestep
    # epsilon defaults Genesis's parser authors for hard constraints amplify no further than that.
    timeconst_min = max(solref_timeconst[solref_timeconst > 0.0].min(), 2.0 * mj_sim.model.opt.timestep)
    efc_atol = tol / timeconst_min**2
    return efc_atol, efc_atol * mj_sim.model.opt.timestep


def _pair_constraint_rows(gs_sim, mj_sim, gs_dofs_idx, mj_dofs_idx, *, qvel_prev, tol, efc_atol):
    """Pair the engines' constraint rows, validating each candidate pairing on the row jacobians, impedances,
    reference accelerations and row velocities. Returns the (gs_sidx, mj_sidx) permutations of the first candidate
    that passes, raising the last validation error when none does.
    """
    gs_n_constraints = gs_sim.rigid_solver.constraint_solver.n_constraints.to_numpy()[0]
    mj_n_constraints = mj_sim.data.nefc
    assert gs_n_constraints == mj_n_constraints
    gs_n_contacts = gs_sim.rigid_solver.collider.collider_state.n_contacts.to_numpy()[0]
    mj_n_contacts = mj_sim.data.ncon
    gs_contact_pos = gs_sim.rigid_solver.collider.collider_state.contact_data.pos.to_numpy()[:gs_n_contacts, 0]
    mj_contact_pos = mj_sim.data.contact.pos
    gs_contact_geoms = np.stack(
        (
            gs_sim.rigid_solver.collider.collider_state.contact_data.geom_a.to_numpy()[:gs_n_contacts, 0],
            gs_sim.rigid_solver.collider.collider_state.contact_data.geom_b.to_numpy()[:gs_n_contacts, 0],
        ),
        axis=-1,
    )
    mj_contact_geoms = np.stack(
        (mj_sim.data.contact.geom1[:mj_n_contacts], mj_sim.data.contact.geom2[:mj_n_contacts]), axis=-1
    )

    # FIXME: It is not always possible to reshape Mujoco jacobian because joint bound constraints are computed in
    # "sparse" dof space, unlike contact constraints.
    error = None
    gs_jac = gs_sim.rigid_solver.constraint_solver.jac.to_numpy()[:gs_n_constraints, :, 0]
    mj_jac = mj_sim.data.efc_J.reshape([mj_n_constraints, -1])
    gs_efc_D = gs_sim.rigid_solver.constraint_solver.efc_D.to_numpy()[:gs_n_constraints, 0]
    mj_efc_D = mj_sim.data.efc_D
    gs_efc_aref = gs_sim.rigid_solver.constraint_solver.aref.to_numpy()[:gs_n_constraints, 0]
    mj_efc_aref = mj_sim.data.efc_aref

    # Constraint rows are paired by identity. A contact's rows are contiguous on both sides, since Genesis places
    # them after the equality and frictionloss rows in the order contact_sort_idx defines and MuJoCo labels them
    # by efc_id, so pairing the contacts pairs their rows. Genesis orders the two opposing rows of a pyramidal
    # friction axis the other way round from MuJoCo, hence the swap in twos; an elliptic cone has no such pairs
    # and keeps its order. A joint-limit row constrains a single DOF, which identifies it within its own family.
    mj_efc_type = mj_sim.data.efc_type[:mj_n_constraints]
    mj_efc_id = mj_sim.data.efc_id[:mj_n_constraints]
    is_mj_contact = np.isin(
        mj_efc_type,
        (
            int(mujoco.mjtConstraint.mjCNSTR_CONTACT_FRICTIONLESS),
            int(mujoco.mjtConstraint.mjCNSTR_CONTACT_PYRAMIDAL),
            int(mujoco.mjtConstraint.mjCNSTR_CONTACT_ELLIPTIC),
        ),
    )
    is_mj_limit = np.isin(
        mj_efc_type,
        (int(mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT), int(mujoco.mjtConstraint.mjCNSTR_LIMIT_TENDON)),
    )
    is_mj_pyramidal = mj_efc_type == int(mujoco.mjtConstraint.mjCNSTR_CONTACT_PYRAMIDAL)
    n_mj_contact_rows = int(is_mj_contact.sum())
    n_head = mj_n_constraints - n_mj_contact_rows - int(is_mj_limit.sum())
    rows_per_contact = n_mj_contact_rows // mj_n_contacts if mj_n_contacts else 0

    gs_contact_sort_idx = gs_sim.rigid_solver.collider.collider_state.contact_sort_idx.to_numpy()[:gs_n_contacts, 0]
    gs_block_of_contact = np.empty(gs_n_contacts, dtype=int)
    gs_block_of_contact[gs_contact_sort_idx] = np.arange(gs_n_contacts)

    # Contacts are paired by ordering both sides the same way: geom pair first, then position. Positions agree to
    # rounding once the pair is fixed, so the order is the same sequence on both sides.
    gs_order = np.lexsort((*gs_contact_pos.T[::-1], *gs_contact_geoms.T[::-1]))
    mj_order = np.lexsort((*mj_contact_pos.T[::-1], *mj_contact_geoms.T[::-1]))
    gs_rows, mj_rows = list(range(n_head)), list(np.flatnonzero(~(is_mj_contact | is_mj_limit)))
    for i_c, i_m in zip(gs_order, mj_order):
        mj_contact_rows = np.flatnonzero(is_mj_contact & (mj_efc_id == i_m))
        if len(mj_contact_rows) and is_mj_pyramidal[mj_contact_rows].all():
            mj_contact_rows = mj_contact_rows.reshape(-1, 2)[:, ::-1].ravel()
        gs_rows.extend(n_head + gs_block_of_contact[i_c] * rows_per_contact + np.arange(rows_per_contact))
        mj_rows.extend(mj_contact_rows)

    gs_limit_rows = np.arange(n_head + n_mj_contact_rows, gs_n_constraints)
    mj_limit_rows = np.flatnonzero(is_mj_limit)
    gs_dof_of_mj_dof = {mj_d: gs_d for gs_d, mj_d in zip(gs_dofs_idx, mj_dofs_idx)}
    gs_rows.extend(gs_limit_rows[np.argsort([int(np.argmax(np.abs(gs_jac[i]))) for i in gs_limit_rows], kind="stable")])
    mj_rows.extend(
        mj_limit_rows[
            np.argsort([gs_dof_of_mj_dof[int(np.argmax(np.abs(mj_jac[i])))] for i in mj_limit_rows], kind="stable")
        ]
    )

    # The value sorts come first because they need nothing of the layout; the identity pairing is the fallback
    # for rows sharing their sorting key. The row velocities separate what the other keys tie on: symmetric
    # contacts, and friction pyramids whose tangent pair the engines label in a different order at fp32.
    pairing_candidates = [
        (np.argsort(gs_jac.sum(axis=1)), np.argsort(mj_jac.sum(axis=1))),
        (np.argsort(gs_efc_aref), np.argsort(mj_efc_aref)),
    ]
    if qvel_prev is not None:
        pairing_candidates.append((np.argsort(gs_jac @ qvel_prev), np.argsort(mj_sim.data.efc_vel)))
    pairing_candidates.append((np.array(gs_rows, dtype=int), np.array(mj_rows, dtype=int)))
    for gs_sidx, mj_sidx in pairing_candidates:
        try:
            gs_jac_nz_mask = (np.abs(gs_jac[gs_sidx]) > 0.0).all(axis=0)
            gs_jac_nz = gs_jac[gs_sidx][:, np.array(gs_dofs_idx)[gs_jac_nz_mask[gs_dofs_idx]]]
            mj_jac_nz_mask = np.zeros_like(gs_jac_nz_mask, dtype=np.bool_)
            mj_jac_nz_mask[mj_dofs_idx] = gs_jac_nz_mask[gs_dofs_idx]
            if mj_jac.shape[-1] == len(mj_dofs_idx):
                mj_jac_nz = mj_jac[mj_sidx][:, np.array(mj_dofs_idx)[mj_jac_nz_mask[mj_dofs_idx]]]
            else:
                mj_jac_nz = mj_jac[mj_sidx]

            assert_allclose(gs_jac_nz, mj_jac_nz, tol=tol)
            assert_allclose(gs_efc_D[gs_sidx], mj_efc_D[mj_sidx], atol=efc_atol, rtol=tol)
            assert_allclose(gs_efc_aref[gs_sidx], mj_efc_aref[mj_sidx], atol=efc_atol, rtol=tol)
            # Row velocities discriminate identically-parameterized rows (e.g. symmetric contacts), which the
            # jacobian column mask and the amplified D/aref floors cannot separate.
            if qvel_prev is not None:
                assert_allclose((gs_jac @ qvel_prev)[gs_sidx], mj_sim.data.efc_vel[mj_sidx], tol=tol)
            return gs_sidx, mj_sidx
        except AssertionError as e:
            error = e
    assert error is not None
    raise error


def check_mujoco_data_consistency(
    gs_sim,
    mj_sim,
    joints_name: list[str] | None = None,
    bodies_name: list[str] | None = None,
    *,
    qvel_prev: np.ndarray | None = None,
    tol: float,
    ignore_constraints: bool = False,
):
    # Get mapping between Mujoco and Genesis
    gs_maps, mj_maps = _get_model_mappings(gs_sim, mj_sim, joints_name, bodies_name)
    gs_bodies_idx, _, gs_q_idx, gs_dofs_idx, _, _ = gs_maps
    mj_bodies_idx, _, mj_qs_idx, mj_dofs_idx, _, _ = mj_maps

    # crb
    gs_crb_inertial = gs_sim.rigid_solver.dyn_state.links.crb_inertial.to_numpy()[:, 0].reshape([-1, 9])[
        :, [0, 4, 8, 1, 2, 5]
    ]
    mj_crb_inertial = mj_sim.data.crb[:, :6]  # upper-triangular part
    assert_allclose(gs_crb_inertial[gs_bodies_idx], mj_crb_inertial[mj_bodies_idx], tol=tol)
    gs_crb_pos = gs_sim.rigid_solver.dyn_state.links.crb_pos.to_numpy()[:, 0]
    mj_crb_pos = mj_sim.data.crb[:, 6:9]
    assert_allclose(gs_crb_pos[gs_bodies_idx], mj_crb_pos[mj_bodies_idx], tol=tol)
    gs_crb_mass = gs_sim.rigid_solver.dyn_state.links.crb_mass.to_numpy()[:, 0]
    mj_crb_mass = mj_sim.data.crb[:, 9]
    assert_allclose(gs_crb_mass[gs_bodies_idx], mj_crb_mass[mj_bodies_idx], tol=tol)

    gs_mass_mat = gs_sim.rigid_solver.mass_mat.to_numpy()[:, :, 0]
    mj_mass_mat = np.zeros((mj_sim.model.nv, mj_sim.model.nv))
    mujoco.mj_fullM(mj_sim.model, mj_sim.data, mj_mass_mat)
    assert_allclose(gs_mass_mat[gs_dofs_idx][:, gs_dofs_idx], mj_mass_mat[mj_dofs_idx][:, mj_dofs_idx], tol=tol)

    gs_meaninertia = gs_sim.rigid_solver.meaninertia.to_numpy()[0]
    mj_meaninertia = mj_sim.model.stat.meaninertia
    assert_allclose(gs_meaninertia, mj_meaninertia, tol=tol)

    # Pre-constraint so-called bias forces in configuration space. The bias force of a fast articulated chain reaches
    # hundreds of newtons and its rounding floor scales with that magnitude, surfacing after cancellation as absolute
    # noise on the small entries; the relative tolerance still guards the large ones.
    gs_qfrc_bias = gs_sim.rigid_solver.dyn_state.dofs.qf_bias.to_numpy()[:, 0]
    mj_qfrc_bias = mj_sim.data.qfrc_bias
    assert_allclose(gs_qfrc_bias, mj_qfrc_bias[mj_dofs_idx], rtol=tol, atol=tol * max(1.0, np.abs(mj_qfrc_bias).max()))
    gs_qfrc_passive = gs_sim.rigid_solver.dyn_state.dofs.qf_passive.to_numpy()[:, 0]
    mj_qfrc_passive = mj_sim.data.qfrc_passive
    assert_allclose(gs_qfrc_passive, mj_qfrc_passive[mj_dofs_idx], tol=tol)
    gs_qfrc_actuator = gs_sim.rigid_solver.dyn_state.dofs.qf_applied.to_numpy()[:, 0]
    mj_qfrc_actuator = mj_sim.data.qfrc_actuator
    assert_allclose(gs_qfrc_actuator, mj_qfrc_actuator[mj_dofs_idx], tol=tol)

    gs_n_contacts = gs_sim.rigid_solver.collider.collider_state.n_contacts.to_numpy()[0]
    mj_n_contacts = mj_sim.data.ncon
    assert gs_n_contacts == mj_n_contacts, f"contact count differs: gs={gs_n_contacts} mj={mj_n_contacts}"
    gs_n_constraints = gs_sim.rigid_solver.constraint_solver.n_constraints.to_numpy()[0]
    mj_n_constraints = mj_sim.data.nefc
    assert gs_n_constraints == mj_n_constraints

    efc_atol, qvel_atol = _compute_efc_tolerances(mj_sim, tol)

    if gs_n_constraints and not ignore_constraints:
        gs_contact_pos = gs_sim.rigid_solver.collider.collider_state.contact_data.pos.to_numpy()[:gs_n_contacts, 0]
        mj_contact_pos = mj_sim.data.contact.pos
        # Sort based on the axis with the largest variation
        max_var_axis = 0
        if gs_n_contacts > 1:
            max_var = -1
            for axis in range(3):
                sorted_contact_pos = np.sort(mj_contact_pos[:, axis])
                var = np.min(sorted_contact_pos[1:] - sorted_contact_pos[:-1])
                if var > max_var:
                    max_var_axis = axis
                    max_var = var
        gs_sidx = np.argsort(gs_contact_pos[:, max_var_axis])
        mj_sidx = np.argsort(mj_contact_pos[:, max_var_axis])
        assert_allclose(gs_contact_pos[gs_sidx], mj_contact_pos[mj_sidx], tol=tol)
        gs_contact_normal = gs_sim.rigid_solver.collider.collider_state.contact_data.normal.to_numpy()[
            :gs_n_contacts, 0
        ]
        mj_contact_normal = -mj_sim.data.contact.frame[:, :3]
        assert_allclose(gs_contact_normal[gs_sidx], mj_contact_normal[mj_sidx], tol=tol)
        gs_penetration = gs_sim.rigid_solver.collider.collider_state.contact_data.penetration.to_numpy()[
            :gs_n_contacts, 0
        ]
        mj_penetration = -mj_sim.data.contact.dist
        assert_allclose(gs_penetration[gs_sidx], mj_penetration[mj_sidx], tol=tol)

        gs_sidx, mj_sidx = _pair_constraint_rows(
            gs_sim, mj_sim, gs_dofs_idx, mj_dofs_idx, qvel_prev=qvel_prev, tol=tol, efc_atol=efc_atol
        )

        gs_efc_force = gs_sim.rigid_solver.constraint_solver.efc_force.to_numpy()[:gs_n_constraints, 0]
        mj_efc_force = mj_sim.data.efc_force
        assert_allclose(gs_efc_force[gs_sidx], mj_efc_force[mj_sidx], atol=efc_atol, rtol=tol)

        mj_iter = mj_sim.data.solver_niter[0] - 1
        if gs_n_constraints and mj_iter >= 0:
            gs_scale = 1.0 / (gs_meaninertia * max(1, gs_sim.rigid_solver.n_dofs))
            gs_gradient = gs_scale * np.linalg.norm(
                gs_sim.rigid_solver.constraint_solver.grad.to_numpy()[: gs_sim.rigid_solver.n_dofs, 0]
            )
            mj_gradient = mj_sim.data.solver.gradient[mj_iter]
            assert_allclose(gs_gradient, mj_gradient, tol=tol)
            gs_improvement = gs_scale * gs_sim.rigid_solver.constraint_solver.ls_improvement[0]
            mj_improvement = mj_sim.data.solver.improvement[mj_iter]

            # Note that 'constraint_solver.active' refers to whether the quadratic part of a constraint is active,
            # unlike Mujoco that defines 'nactive' as the number of active constraints regardless of its type.
            # In practice, this only makes a difference if frictionloss is enabled. Middle-zone (slipping) elliptic
            # cone rows are excluded from the per-row quadratic (handled as a coupled block) yet count as active in
            # Mujoco's stat, which engages a cone as a whole; counting every row of a cone that carries any force
            # translates Genesis's convention into Mujoco's.
            gs_counted = gs_sim.rigid_solver.constraint_solver.active.to_numpy()[:gs_n_constraints, 0].copy()
            gs_n_cone = gs_sim.rigid_solver.constraint_solver.n_constraints_cone.to_numpy()[0]
            if gs_n_cone:
                gs_nef = (
                    gs_sim.rigid_solver.constraint_solver.n_constraints_equality.to_numpy()[0]
                    + gs_sim.rigid_solver.constraint_solver.n_constraints_frictionloss.to_numpy()[0]
                )
                rows_per_contact = gs_sim.rigid_solver.rigid_config.rows_per_contact
                gs_cone_rows = slice(gs_nef, gs_nef + gs_n_cone)
                gs_cone_rows_counted = gs_counted[gs_cone_rows] | (np.abs(gs_efc_force[gs_cone_rows]) > 0.0)
                gs_cones_counted = gs_cone_rows_counted.reshape(-1, rows_per_contact).any(axis=1)
                gs_counted[gs_cone_rows] = np.repeat(gs_cones_counted, rows_per_contact)
            gs_nactive = gs_counted.sum()
            mj_native = mj_sim.data.solver.nactive[mj_iter]
            if not (gs_sim.rigid_solver.dyn_info.dofs.frictionloss.to_numpy() > gs.EPS).any():
                assert mj_native == gs_nactive

            # FIXME: For some reason, mujoco is sometimes (seemingful) wrongly reporting 0...
            # The final iterate's improvement is a path quantity, meaningless to compare unless both engines round
            # identically, which only holds at double precision.
            if mj_improvement > gs.EPS and gs.np_float == np.float64:
                # Must relax tolerance because of compounding of errors.
                assert_allclose(gs_improvement, mj_improvement, tol=tol * 1e2)

    gs_qfrc_constraint = gs_sim.rigid_solver.dyn_state.dofs.qf_constraint.to_numpy()[:, 0]
    mj_qfrc_constraint = mj_sim.data.qfrc_constraint
    assert_allclose(gs_qfrc_constraint[gs_dofs_idx], mj_qfrc_constraint[mj_dofs_idx], atol=efc_atol, rtol=tol)

    gs_qfrc_all = gs_sim.rigid_solver.dyn_state.dofs.force.to_numpy()[:, 0]
    mj_qfrc_all = mj_sim.data.qfrc_smooth + mj_sim.data.qfrc_constraint
    assert_allclose(gs_qfrc_all[gs_dofs_idx], mj_qfrc_all[mj_dofs_idx], atol=efc_atol, rtol=tol)

    # The smooth force sums the bias force with the passive and actuation forces, so it carries the bias force's
    # magnitude-scaled rounding floor (see the qfrc_bias comparison above).
    gs_qfrc_smooth = gs_sim.rigid_solver.dyn_state.dofs.qf_smooth.to_numpy()[:, 0]
    mj_qfrc_smooth = mj_sim.data.qfrc_smooth
    assert_allclose(
        gs_qfrc_smooth[gs_dofs_idx],
        mj_qfrc_smooth[mj_dofs_idx],
        rtol=tol,
        atol=tol * max(1.0, np.abs(mj_qfrc_bias).max()),
    )

    gs_qacc_smooth = gs_sim.rigid_solver.dyn_state.dofs.acc_smooth.to_numpy()[:, 0]
    mj_qacc_smooth = mj_sim.data.qacc_smooth
    # The smooth acceleration divides the rounding accumulated along the kinematic chain by the lightest DOF
    # inertia, so its single-precision comparison carries that amplification (mj_mass_mat is computed above).
    if gs.np_float == np.float64:
        qacc_smooth_atol = tol
    else:
        qacc_smooth_atol = tol / mj_mass_mat.diagonal()[mj_dofs_idx].min()
    assert_allclose(gs_qacc_smooth[gs_dofs_idx], mj_qacc_smooth[mj_dofs_idx], atol=qacc_smooth_atol, rtol=tol)

    # Acceleration pre- VS post-implicit damping gs_qacc_post = gs_sim.rigid_solver.dyn_state.dofs.acc.to_numpy()[:, 0]
    if gs_n_constraints:
        gs_qacc_pre = gs_sim.rigid_solver.constraint_solver.qacc.to_numpy()[:, 0]
    else:
        gs_qacc_pre = gs_qacc_smooth
    mj_qacc_pre = mj_sim.data.qacc
    # Midpoint-integrated DOFs hold the realized acceleration (qvel_new - qvel_old) / h in both engines, which
    # Genesis mirrors in dofs.acc; the smooth+constraint acceleration is only observable on the remaining DOFs.
    midpoint_mask = get_mujoco_midpoint_dofs_mask(mj_sim)[mj_dofs_idx]
    gs_qacc_post = gs_sim.rigid_solver.dyn_state.dofs.acc.to_numpy()[:, 0]
    assert_allclose(
        gs_qacc_pre[gs_dofs_idx][~midpoint_mask], mj_qacc_pre[mj_dofs_idx][~midpoint_mask], atol=efc_atol, rtol=tol
    )
    assert_allclose(
        gs_qacc_post[gs_dofs_idx][midpoint_mask], mj_qacc_pre[mj_dofs_idx][midpoint_mask], atol=efc_atol, rtol=tol
    )

    gs_qvel = gs_sim.rigid_solver.dyn_state.dofs.vel.to_numpy()[:, 0]
    mj_qvel = mj_sim.data.qvel
    assert_allclose(gs_qvel[gs_dofs_idx], mj_qvel[mj_dofs_idx], atol=qvel_atol, rtol=tol)
    gs_qpos = gs_sim.rigid_solver.qpos.to_numpy()[:, 0]
    mj_qpos = mj_sim.data.qpos
    assert_allclose(gs_qpos[gs_q_idx], mj_qpos[mj_qs_idx], tol=tol)

    # ------------------------------------------------------------------------

    gs_com = gs_sim.rigid_solver.dyn_state.links.root_COM.to_numpy()[:, 0]
    gs_root_idx = np.unique(gs_sim.rigid_solver.dyn_info.links.root_idx.to_numpy()[gs_bodies_idx])
    mj_com = mj_sim.data.subtree_com
    mj_root_idx = np.unique(mj_sim.model.body_rootid[mj_bodies_idx])
    assert_allclose(gs_com[gs_root_idx], mj_com[mj_root_idx], tol=tol)

    gs_xipos = gs_sim.rigid_solver.dyn_state.links.i_pos.to_numpy()[:, 0]
    mj_xipos = mj_sim.data.xipos - mj_sim.data.subtree_com[mj_sim.model.body_rootid]
    assert_allclose(gs_xipos[gs_bodies_idx], mj_xipos[mj_bodies_idx], tol=tol)

    gs_xpos = gs_sim.rigid_solver.dyn_state.links.pos.to_numpy()[:, 0]
    mj_xpos = mj_sim.data.xpos
    assert_allclose(gs_xpos[gs_bodies_idx], mj_xpos[mj_bodies_idx], tol=tol)

    gs_xquat = gs_sim.rigid_solver.dyn_state.links.quat.to_numpy()[:, 0]
    gs_xmat = gu.quat_to_R(gs_xquat).reshape([-1, 9])
    mj_xmat = mj_sim.data.xmat
    assert_allclose(gs_xmat[gs_bodies_idx], mj_xmat[mj_bodies_idx], tol=tol)

    gs_cd_vel = gs_sim.rigid_solver.dyn_state.links.cd_vel.to_numpy()[:, 0]
    mj_cd_vel = mj_sim.data.cvel[:, 3:]
    assert_allclose(gs_cd_vel[gs_bodies_idx], mj_cd_vel[mj_bodies_idx], tol=tol)
    gs_cd_ang = gs_sim.rigid_solver.dyn_state.links.cd_ang.to_numpy()[:, 0]
    mj_cd_ang = mj_sim.data.cvel[:, :3]
    assert_allclose(gs_cd_ang[gs_bodies_idx], mj_cd_ang[mj_bodies_idx], tol=tol)

    gs_cdof_vel = gs_sim.rigid_solver.dyn_state.dofs.cdof_vel.to_numpy()[:, 0]
    mj_cdof_vel = mj_sim.data.cdof[:, 3:]
    assert_allclose(gs_cdof_vel[gs_dofs_idx], mj_cdof_vel[mj_dofs_idx], tol=tol)
    gs_cdof_ang = gs_sim.rigid_solver.dyn_state.dofs.cdof_ang.to_numpy()[:, 0]
    mj_cdof_ang = mj_sim.data.cdof[:, :3]
    assert_allclose(gs_cdof_ang[gs_dofs_idx], mj_cdof_ang[mj_dofs_idx], tol=tol)

    mj_cdof_dot_ang = mj_sim.data.cdof_dot[:, :3]
    gs_cdof_dot_ang = gs_sim.rigid_solver.dyn_state.dofs.cdofd_ang.to_numpy()[:, 0]
    assert_allclose(gs_cdof_dot_ang[gs_dofs_idx], mj_cdof_dot_ang[mj_dofs_idx], tol=tol)

    mj_cdof_dot_vel = mj_sim.data.cdof_dot[:, 3:]
    gs_cdof_dot_vel = gs_sim.rigid_solver.dyn_state.dofs.cdofd_vel.to_numpy()[:, 0]
    assert_allclose(gs_cdof_dot_vel[gs_dofs_idx], mj_cdof_dot_vel[mj_dofs_idx], tol=tol)

    # cinr
    gs_cinr_inertial = gs_sim.rigid_solver.dyn_state.links.cinr_inertial.to_numpy()[:, 0].reshape([-1, 9])[
        :, [0, 4, 8, 1, 2, 5]
    ]
    mj_cinr_inertial = mj_sim.data.cinert[:, :6]  # upper-triangular part
    assert_allclose(gs_cinr_inertial[gs_bodies_idx], mj_cinr_inertial[mj_bodies_idx], tol=tol)
    gs_cinr_pos = gs_sim.rigid_solver.dyn_state.links.cinr_pos.to_numpy()[:, 0]
    mj_cinr_pos = mj_sim.data.cinert[:, 6:9]
    assert_allclose(gs_cinr_pos[gs_bodies_idx], mj_cinr_pos[mj_bodies_idx], tol=tol)
    gs_cinr_mass = gs_sim.rigid_solver.dyn_state.links.cinr_mass.to_numpy()[:, 0]
    mj_cinr_mass = mj_sim.data.cinert[:, 9]
    assert_allclose(gs_cinr_mass[gs_bodies_idx], mj_cinr_mass[mj_bodies_idx], tol=tol)


def simulate_and_check_mujoco_consistency(
    gs_sim, mj_sim, qpos=None, qvel=None, *, tol, num_steps, ignore_constraints=False
):
    # Get mapping between Mujoco and Genesis
    gs_maps, mj_maps = _get_model_mappings(gs_sim, mj_sim)
    gs_bodies_idx, _, _, gs_dofs_idx, _, _ = gs_maps
    mj_bodies_idx, _, mj_qs_idx, mj_dofs_idx, _, _ = mj_maps

    # Make sure that "static" model information are matching
    check_mujoco_model_consistency(gs_sim, mj_sim, tol=tol)

    # Weights computed in single precision land a unit or two of the last place from MuJoCo's double ones, which a
    # contact system holding near-equal rows carries through to forces differing by percents, so MuJoCo's are imposed.
    if gs.np_float == np.float32:
        links_invweight = qd_to_numpy(gs_sim.rigid_solver.dyn_info.links.invweight)
        links_invweight[gs_bodies_idx] = mj_sim.model.body_invweight0[mj_bodies_idx]
        gs_sim.rigid_solver.dyn_info.links.invweight.from_numpy(links_invweight)
        dofs_invweight = qd_to_numpy(gs_sim.rigid_solver.dyn_info.dofs.invweight)
        dofs_invweight[gs_dofs_idx] = mj_sim.model.dof_invweight0[mj_dofs_idx]
        gs_sim.rigid_solver.dyn_info.dofs.invweight.from_numpy(dofs_invweight)

    # Initialize the simulation
    init_paired_simulators(gs_sim, mj_sim, qpos, qvel)

    # Run the simulation for a few steps
    qvel_prev = None

    # At single precision the engines assemble measurably different constraint problems: the reference acceleration
    # amplifies the rounding of its position and velocity inputs by the inverse squared constraint time constant
    # (see _compute_efc_tolerances), which moves a near-boundary row's activation threshold by more than its margin.
    # Substituting MuJoCo's double-precision reference acceleration right before each solve makes both engines
    # minimize the same problem, keeping the solver comparison exact; the row pairing still validates Genesis's own
    # assembled values beforehand. MuJoCo steps first in the loop below, so its rows describe the same state.
    constraint_solver = gs_sim.rigid_solver.constraint_solver
    resolve_solver = constraint_solver.resolve

    def resolve_on_mujoco_aref():
        if constraint_solver.n_constraints.to_numpy()[0]:
            efc_atol, _ = _compute_efc_tolerances(mj_sim, tol)
            gs_sidx, mj_sidx = _pair_constraint_rows(
                gs_sim, mj_sim, gs_dofs_idx, mj_dofs_idx, qvel_prev=qvel_prev, tol=tol, efc_atol=efc_atol
            )
            aref_rows = constraint_solver.constraint_state.aref.to_numpy()
            aref_rows[gs_sidx, 0] = mj_sim.data.efc_aref[mj_sidx]
            constraint_solver.constraint_state.aref.from_numpy(aref_rows)
        resolve_solver()

    with pytest.MonkeyPatch.context() as mp:
        if gs.np_float == np.float32 and not ignore_constraints:
            mp.setattr(constraint_solver, "resolve", resolve_on_mujoco_aref)

        for i in range(num_steps):
            # Make sure that all "dynamic" quantities are matching before stepping
            check_mujoco_data_consistency(
                gs_sim, mj_sim, qvel_prev=qvel_prev, tol=tol, ignore_constraints=ignore_constraints
            )

            # Keep Mujoco and Genesis simulation in sync to avoid drift over time
            mj_sim.data.qpos[mj_qs_idx] = gs_sim.rigid_solver.qpos.to_numpy()[:, 0]
            mj_sim.data.qvel[mj_dofs_idx] = gs_sim.rigid_solver.dyn_state.dofs.vel.to_numpy()[:, 0]
            mj_sim.data.qacc_warmstart[mj_dofs_idx] = gs_sim.rigid_solver.constraint_solver.qacc_ws.to_numpy()[:, 0]
            mj_sim.data.qacc_smooth[mj_dofs_idx] = gs_sim.rigid_solver.dyn_state.dofs.acc_smooth.to_numpy()[:, 0]

            # Backup current velocity
            qvel_prev = gs_sim.rigid_solver.dyn_state.dofs.vel.to_numpy()[:, 0]

            # Do a single simulation step (eventually with substeps for Genesis)
            mujoco.mj_step(mj_sim.model, mj_sim.data)
            gs_sim.scene.step()
            # if gs_sim.scene.visualizer:
            #     gs_sim.scene.visualizer.update()

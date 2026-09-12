"""Manual reverse-mode kernels for the rigid backward pass.

They hand-compute the Jacobian-transpose products of forward stages whose reverse Quadrants automatic differentiation
silently drops (forward kinematics, forward velocity) or cannot express (the mass solve, reversed via the implicit
function theorem). The chain-rule building blocks (quaternion product, quaternion transform, rotation-vector
conversion, motion cross product) live next to their forwards in genesis/utils/geom.py as the *_grad_* adjoint funcs.
"""

import quadrants as qd

import genesis as gs
import genesis.utils.array_class as array_class
import genesis.utils.geom as gu


@qd.kernel(fastcache=True)
def kernel_manual_forward_kinematics_bw(
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    """Manual reverse of kernel_forward_kinematics_replay: link pos / quat grads back to qpos and parent-pose grads.

    Iterates each entity's links leaf to root in one launch, so a child's parent pos / quat grad write lands before
    the parent's own iteration consumes it, and within each link reverses the full joint chain. A link may carry
    several joints (e.g. a planar floating base = slide-x + slide-z + hinge-y on one link); the forward composes them
    in sequence and caches the per-joint intermediate pose in dyn_state.links.{pos,quat}_bw[i_l, k]: slot 0 is the
    parent pose composed with the link's fixed offset, slot k+1 the pose after joint k, slot n_joints the final link
    pose. The reverse seeds the grad on the final pose, reverses joint k for k = n_joints-1 .. 0 (each step consumes
    the grad on slot k+1 and emits qpos.grad for that joint plus the grad on slot k), then reverses the base
    composition into the parent's pose grad. Each joint also feeds dyn_state.joints.{xanchor,xaxis} downstream
    (forward velocity), so their accumulated grads are folded back through slot k as well.
    """
    qd.loop_config(
        name="manual_fk_only_bw",
        serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL),
    )
    for i_e, i_b in qd.ndrange(dyn_info.entities.n_links.shape[0], dyn_state.links.pos.shape[1]):
        n_in_e = dyn_info.entities.n_links[i_e]
        for i_l_rev in range(n_in_e):
            i_l = dyn_info.entities.link_end[i_e] - 1 - i_l_rev
            I_l = [i_l, i_b] if qd.static(rigid_config.batch_links_info) else i_l
            i_parent = dyn_info.links.parent_idx[I_l]
            n_joints = dyn_info.links.joint_end[I_l] - dyn_info.links.joint_start[I_l]

            # Grad seeded on the final link pose (= slot n_joints). Carried
            # backward through the joint chain; after the loop it holds the grad
            # on slot 0 (arm base).
            g_pos = dyn_state.links.pos.grad[i_l, i_b]
            g_quat = dyn_state.links.quat.grad[i_l, i_b]

            for k_rev in range(n_joints):
                k = n_joints - 1 - k_rev
                i_j = dyn_info.links.joint_start[I_l] + k
                I_j = [i_j, i_b] if qd.static(rigid_config.batch_joints_info) else i_j
                joint_type = dyn_info.joints.type[I_j]
                q_start = dyn_info.joints.q_start[I_j]
                dof_start = dyn_info.joints.dof_start[I_j]
                I_d = [dof_start, i_b] if qd.static(rigid_config.batch_dofs_info) else dof_start

                # Input pose to joint k (slot k), cached by the forward replay.
                pos_in = dyn_state.links.pos_bw[i_l, k, i_b]
                quat_in = dyn_state.links.quat_bw[i_l, k, i_b]
                joint_pos_off = dyn_info.joints.pos[I_j]
                xanchor_grad = dyn_state.joints.xanchor.grad[i_j, i_b]
                xaxis_grad = dyn_state.joints.xaxis.grad[i_j, i_b]

                if joint_type == gs.JOINT_TYPE.FREE:
                    # Final pose is set absolutely from qpos (slot in unused); xanchor = qpos[0:3]. The forward also
                    # publishes the pose as dofs.pos = (pos, quat_to_xyz(quat)), whose grad (accumulated by the
                    # force-stage reverses through position control / joint stiffness) folds back here.
                    for j in qd.static(range(3)):
                        rigid_info.qpos.grad[q_start + j, i_b] = (
                            rigid_info.qpos.grad[q_start + j, i_b]
                            + g_pos[j]
                            + xanchor_grad[j]
                            + dyn_state.dofs.pos.grad[dof_start + j, i_b]
                        )
                    # The forward normalizes the raw qpos quaternion (quat_ = q / |q|) before writing the link
                    # quaternion, so the gradient must pass through the Jacobian of q / |q|:
                    # J^T g = (g - qhat (qhat . g)) / |q|. Copying g straight into the raw entries leaves a spurious
                    # radial component that disagrees with finite differences even at unit length.
                    q_raw = qd.Vector(
                        [
                            rigid_info.qpos[q_start + 3, i_b],
                            rigid_info.qpos[q_start + 4, i_b],
                            rigid_info.qpos[q_start + 5, i_b],
                            rigid_info.qpos[q_start + 6, i_b],
                        ],
                        dt=gs.qd_float,
                    )
                    q_norm = q_raw.norm()
                    qhat = q_raw / q_norm
                    g_xyz = qd.Vector(
                        [
                            dyn_state.dofs.pos.grad[dof_start + 3, i_b],
                            dyn_state.dofs.pos.grad[dof_start + 4, i_b],
                            dyn_state.dofs.pos.grad[dof_start + 5, i_b],
                        ],
                        dt=gs.qd_float,
                    )
                    g_quat_eff = g_quat + gu.qd_quat_to_xyz_grad_quat(qhat, rigid_info.EPS[None], g_xyz)
                    g_quat_raw = (g_quat_eff - qhat * qhat.dot(g_quat_eff)) / q_norm
                    for j in qd.static(range(4)):
                        rigid_info.qpos.grad[q_start + 3 + j, i_b] = (
                            rigid_info.qpos.grad[q_start + 3 + j, i_b] + g_quat_raw[j]
                        )
                    for j in qd.static(range(6)):
                        dyn_state.dofs.pos.grad[dof_start + j, i_b] = 0.0
                    g_pos = qd.Vector([0.0, 0.0, 0.0], dt=gs.qd_float)
                    g_quat = qd.Vector([0.0, 0.0, 0.0, 0.0], dt=gs.qd_float)

                elif joint_type == gs.JOINT_TYPE.REVOLUTE:
                    axis = dyn_info.dofs.motion_ang[I_d]
                    angle = rigid_info.qpos[q_start, i_b] - rigid_info.qpos0[q_start, i_b]
                    rotvec = axis * angle
                    qloc = gu.qd_rotvec_to_quat(rotvec, rigid_info.EPS[None])
                    # quat_out = transform_quat_by_quat(qloc, quat_in) = quat_mul(quat_in, qloc)
                    quat_out = gu.qd_transform_quat_by_quat(qloc, quat_in)

                    # pos_out = xanchor - transform(joint_pos_off, quat_out)
                    # xanchor = transform(joint_pos_off, quat_in) + pos_in
                    gq_out = g_quat - gu.qd_transform_by_quat_grad_quat(joint_pos_off, quat_out, g_pos)
                    g_qloc = gu.qd_quat_mul_grad_rhs(quat_in, qloc, gq_out)
                    g_quat_in_apply = gu.qd_quat_mul_grad_lhs(quat_in, qloc, gq_out)
                    rotvec_grad = gu.qd_rotvec_to_quat_grad_rotvec(rotvec, rigid_info.EPS[None], g_qloc)
                    angle_grad = axis[0] * rotvec_grad[0] + axis[1] * rotvec_grad[1] + axis[2] * rotvec_grad[2]
                    # The forward also publishes the angle as dofs.pos[dof_start] = qpos - qpos0, whose grad
                    # (accumulated by the force-stage reverses) folds back linearly.
                    rigid_info.qpos.grad[q_start, i_b] = (
                        rigid_info.qpos.grad[q_start, i_b] + angle_grad + dyn_state.dofs.pos.grad[dof_start, i_b]
                    )
                    dyn_state.dofs.pos.grad[dof_start, i_b] = 0.0

                    # grad into xanchor = g_pos (from pos_out) + downstream xanchor_grad
                    g_xanchor = g_pos + xanchor_grad
                    g_quat_in = (
                        g_quat_in_apply
                        + gu.qd_transform_by_quat_grad_quat(joint_pos_off, quat_in, g_xanchor)
                        + gu.qd_transform_by_quat_grad_quat(axis, quat_in, xaxis_grad)
                    )
                    g_pos = g_xanchor
                    g_quat = g_quat_in

                elif joint_type == gs.JOINT_TYPE.PRISMATIC:
                    axis = dyn_info.dofs.motion_vel[I_d]
                    displacement = rigid_info.qpos[q_start, i_b] - rigid_info.qpos0[q_start, i_b]
                    xaxis = gu.qd_transform_by_quat(axis, quat_in)
                    # pos_out = pos_in + xaxis * displacement ; quat_out = quat_in
                    displacement_grad = xaxis[0] * g_pos[0] + xaxis[1] * g_pos[1] + xaxis[2] * g_pos[2]
                    # The forward also publishes the displacement as dofs.pos[dof_start] = qpos - qpos0, whose grad
                    # (accumulated by the force-stage reverses) folds back linearly.
                    rigid_info.qpos.grad[q_start, i_b] = (
                        rigid_info.qpos.grad[q_start, i_b] + displacement_grad + dyn_state.dofs.pos.grad[dof_start, i_b]
                    )
                    dyn_state.dofs.pos.grad[dof_start, i_b] = 0.0
                    g_xaxis = qd.Vector(
                        [
                            g_pos[0] * displacement + xaxis_grad[0],
                            g_pos[1] * displacement + xaxis_grad[1],
                            g_pos[2] * displacement + xaxis_grad[2],
                        ],
                        dt=gs.qd_float,
                    )
                    # pos_out never reads xanchor here (pos_out = pos_in + xaxis * displacement), so only the
                    # downstream xanchor consumer's grad flows through xanchor = T(joint_pos_off, quat_in) + pos_in
                    # into quat_in; g_pos itself reaches pos_in directly.
                    g_quat_in = (
                        g_quat
                        + gu.qd_transform_by_quat_grad_quat(axis, quat_in, g_xaxis)
                        + gu.qd_transform_by_quat_grad_quat(joint_pos_off, quat_in, xanchor_grad)
                    )
                    g_pos = g_pos + xanchor_grad
                    g_quat = g_quat_in

                elif joint_type == gs.JOINT_TYPE.SPHERICAL:
                    # qloc = qpos[q_start:q_start+4] (direct); quat_out = quat_mul(quat_in, qloc).
                    # axis defaults to [0,0,1] (xaxis = transform(axis, quat_in)).
                    axis = qd.Vector([0.0, 0.0, 1.0], dt=gs.qd_float)
                    qloc = qd.Vector(
                        [
                            rigid_info.qpos[q_start, i_b],
                            rigid_info.qpos[q_start + 1, i_b],
                            rigid_info.qpos[q_start + 2, i_b],
                            rigid_info.qpos[q_start + 3, i_b],
                        ],
                        dt=gs.qd_float,
                    )
                    quat_out = gu.qd_transform_quat_by_quat(qloc, quat_in)
                    gq_out = g_quat - gu.qd_transform_by_quat_grad_quat(joint_pos_off, quat_out, g_pos)
                    g_qloc = gu.qd_quat_mul_grad_rhs(quat_in, qloc, gq_out)
                    g_quat_in_apply = gu.qd_quat_mul_grad_lhs(quat_in, qloc, gq_out)
                    # The forward also publishes the joint coordinate as dofs.pos = quat_to_xyz(qloc), whose grad
                    # (accumulated by the force-stage reverses) folds back through the Euler extraction.
                    g_xyz = qd.Vector(
                        [
                            dyn_state.dofs.pos.grad[dof_start, i_b],
                            dyn_state.dofs.pos.grad[dof_start + 1, i_b],
                            dyn_state.dofs.pos.grad[dof_start + 2, i_b],
                        ],
                        dt=gs.qd_float,
                    )
                    g_qloc = g_qloc + gu.qd_quat_to_xyz_grad_quat(qloc, rigid_info.EPS[None], g_xyz)
                    for j in qd.static(range(4)):
                        rigid_info.qpos.grad[q_start + j, i_b] = rigid_info.qpos.grad[q_start + j, i_b] + g_qloc[j]
                    for j in qd.static(range(3)):
                        dyn_state.dofs.pos.grad[dof_start + j, i_b] = 0.0
                    g_xanchor = g_pos + xanchor_grad
                    g_quat_in = (
                        g_quat_in_apply
                        + gu.qd_transform_by_quat_grad_quat(joint_pos_off, quat_in, g_xanchor)
                        + gu.qd_transform_by_quat_grad_quat(axis, quat_in, xaxis_grad)
                    )
                    g_pos = g_xanchor
                    g_quat = g_quat_in

                else:  # gs.JOINT_TYPE.FIXED - pose passes through unchanged.
                    pass

                for j in qd.static(range(3)):
                    dyn_state.joints.xanchor.grad[i_j, i_b][j] = 0.0
                    dyn_state.joints.xaxis.grad[i_j, i_b][j] = 0.0

            # Reverse the arm-base composition (slot 0):
            #   arm_base_pos  = parent_pos + transform(link_offset_pos, parent_quat)
            #   arm_base_quat = quat_mul(parent_quat, link_offset_quat)
            # propagating slot-0 grad (g_pos, g_quat) into the parent's pose grad.
            if i_parent != -1:
                parent_quat = dyn_state.links.quat[i_parent, i_b]
                link_off_pos = dyn_info.links.pos[I_l]
                link_off_quat = dyn_info.links.quat[I_l]
                parent_quat_grad_from_pos = gu.qd_transform_by_quat_grad_quat(link_off_pos, parent_quat, g_pos)
                parent_quat_grad_from_quat = gu.qd_quat_mul_grad_lhs(parent_quat, link_off_quat, g_quat)
                for j in qd.static(range(3)):
                    dyn_state.links.pos.grad[i_parent, i_b][j] = dyn_state.links.pos.grad[i_parent, i_b][j] + g_pos[j]
                for j in qd.static(range(4)):
                    dyn_state.links.quat.grad[i_parent, i_b][j] = (
                        dyn_state.links.quat.grad[i_parent, i_b][j]
                        + parent_quat_grad_from_pos[j]
                        + parent_quat_grad_from_quat[j]
                    )

            # The forward skips writing fixed root links' pose so users can overwrite it (see
            # func_forward_kinematics_entity): their pose is persistent input state whose grad must keep
            # accumulating across substeps instead of being consumed here.
            if not (i_parent == -1 and dyn_info.links.is_fixed[I_l]):
                for j in qd.static(range(3)):
                    dyn_state.links.pos.grad[i_l, i_b][j] = 0.0
                for j in qd.static(range(4)):
                    dyn_state.links.quat.grad[i_l, i_b][j] = 0.0


@qd.func
def func_cd_contraction_bw(
    i_b,
    i_l,
    i_slot,
    dof_lo,
    dof_hi,
    dyn_state: array_class.DynState,
):
    """Reverse of the forward-velocity contraction atomic_add(cd_*_bw[i_l, i_slot], cdof_*[i_d] * vel[i_d]) over
    dofs [dof_lo, dof_hi): accumulates cd_*_bw[i_slot].grad into cdof_{ang,vel}.grad and vel.grad."""
    g_cd_vel = dyn_state.links.cd_vel_bw.grad[i_l, i_slot, i_b]
    g_cd_ang = dyn_state.links.cd_ang_bw.grad[i_l, i_slot, i_b]
    for i_d in range(dof_lo, dof_hi):
        dof_vel = dyn_state.dofs.vel[i_d, i_b]
        cdof_vel = dyn_state.dofs.cdof_vel[i_d, i_b]
        cdof_ang = dyn_state.dofs.cdof_ang[i_d, i_b]
        for k in qd.static(range(3)):
            dyn_state.dofs.cdof_vel.grad[i_d, i_b][k] = (
                dyn_state.dofs.cdof_vel.grad[i_d, i_b][k] + g_cd_vel[k] * dof_vel
            )
            dyn_state.dofs.cdof_ang.grad[i_d, i_b][k] = (
                dyn_state.dofs.cdof_ang.grad[i_d, i_b][k] + g_cd_ang[k] * dof_vel
            )
        dot_vel = cdof_vel[0] * g_cd_vel[0] + cdof_vel[1] * g_cd_vel[1] + cdof_vel[2] * g_cd_vel[2]
        dot_ang = cdof_ang[0] * g_cd_ang[0] + cdof_ang[1] * g_cd_ang[1] + cdof_ang[2] * g_cd_ang[2]
        dyn_state.dofs.vel.grad[i_d, i_b] = dyn_state.dofs.vel.grad[i_d, i_b] + dot_vel + dot_ang


@qd.kernel(fastcache=True)
def kernel_manual_forward_velocity_bw(
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    """Manual reverse of kernel_forward_velocity: link velocity grads back to dof velocity and cdof grads.

    Consumes the grad seeds on cd_vel / cd_ang, their per-joint caches cd_{vel,ang}_bw and cdofd_{ang,vel}, and
    accumulates into dyn_state.dofs.{vel,cdof_ang,cdof_vel}.grad plus the parent links' cd_{vel,ang}.grad (the
    cross-link chain matching the forward replay's cd_*_bw[i_l, 0] = parent cd_*).
    """
    qd.loop_config(
        name="manual_forward_velocity_bw",
        serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL),
    )
    for i_e, i_b in qd.ndrange(dyn_info.entities.n_links.shape[0], dyn_state.links.pos.shape[1]):
        n_in_e = dyn_info.entities.n_links[i_e]
        # Leaf -> root iteration so each link's cd_*_bw[0].grad (which
        # accumulates into parent.cd_*.grad) is propagated *before* the
        # parent's own iteration uses it.
        for i_l_rev in range(n_in_e):
            i_l = dyn_info.entities.link_end[i_e] - 1 - i_l_rev
            I_l = [i_l, i_b] if qd.static(rigid_config.batch_links_info) else i_l
            n_joints = dyn_info.links.joint_end[I_l] - dyn_info.links.joint_start[I_l]
            i_parent = dyn_info.links.parent_idx[I_l]

            # --- Step 1 reverse: cd_*[i_l].grad -> cd_*_bw[i_l, n_joints].grad
            for k in qd.static(range(3)):
                dyn_state.links.cd_vel_bw.grad[i_l, n_joints, i_b][k] = (
                    dyn_state.links.cd_vel_bw.grad[i_l, n_joints, i_b][k] + dyn_state.links.cd_vel.grad[i_l, i_b][k]
                )
                dyn_state.links.cd_ang_bw.grad[i_l, n_joints, i_b][k] = (
                    dyn_state.links.cd_ang_bw.grad[i_l, n_joints, i_b][k] + dyn_state.links.cd_ang.grad[i_l, i_b][k]
                )
            # consume cd_vel/cd_ang.grad[i_l]
            for k in qd.static(range(3)):
                dyn_state.links.cd_vel.grad[i_l, i_b][k] = 0.0
                dyn_state.links.cd_ang.grad[i_l, i_b][k] = 0.0

            # --- Step 2: iterate joints in reverse
            for i_j_rev in range(n_joints):
                i_j_ = n_joints - 1 - i_j_rev
                i_j = i_j_ + dyn_info.links.joint_start[I_l]
                I_j = [i_j, i_b] if qd.static(rigid_config.batch_joints_info) else i_j
                joint_type = dyn_info.joints.type[I_j]
                dof_start = dyn_info.joints.dof_start[I_j]
                dof_end = dyn_info.joints.dof_end[I_j]
                i_slot = i_j_
                i_slot_next = i_j_ + 1

                # --- Reverse the joint-space contraction into cd_*_bw[next].
                # Forward (FREE contracts only its angular dofs here; the linear ones go through slot curr):
                #   _vel = cdof_vel[d] * vel[d];  atomic_add(cd_vel_bw[next], _vel)
                #   _ang = cdof_ang[d] * vel[d];  atomic_add(cd_ang_bw[next], _ang)
                dof_lo = dof_start + 3 if joint_type == gs.JOINT_TYPE.FREE else dof_start
                func_cd_contraction_bw(i_b, i_l, i_slot_next, dof_lo, dof_end, dyn_state)
                g_cd_vel_next = dyn_state.links.cd_vel_bw.grad[i_l, i_slot_next, i_b]
                g_cd_ang_next = dyn_state.links.cd_ang_bw.grad[i_l, i_slot_next, i_b]

                # --- Reverse the slot copy cd_*_bw[next] = cd_*_bw[curr]
                for k in qd.static(range(3)):
                    dyn_state.links.cd_vel_bw.grad[i_l, i_slot, i_b][k] = (
                        dyn_state.links.cd_vel_bw.grad[i_l, i_slot, i_b][k] + g_cd_vel_next[k]
                    )
                    dyn_state.links.cd_ang_bw.grad[i_l, i_slot, i_b][k] = (
                        dyn_state.links.cd_ang_bw.grad[i_l, i_slot, i_b][k] + g_cd_ang_next[k]
                    )
                # consume next
                for k in qd.static(range(3)):
                    dyn_state.links.cd_vel_bw.grad[i_l, i_slot_next, i_b][k] = 0.0
                    dyn_state.links.cd_ang_bw.grad[i_l, i_slot_next, i_b][k] = 0.0

                # --- Reverse motion_cross_motion:
                # Forward: (cdofd_ang[i_d], cdofd_vel[i_d]) =
                #     motion_cross_motion(cd_ang_bw[curr], cd_vel_bw[curr], cdof_ang[i_d], cdof_vel[i_d])
                # over the same dof range as the contraction above (FREE applies it to its angular dofs only).
                s_ang_primal = dyn_state.links.cd_ang_bw[i_l, i_slot, i_b]
                s_vel_primal = dyn_state.links.cd_vel_bw[i_l, i_slot, i_b]
                for i_d in range(dof_lo, dof_end):
                    g_cdofd_ang = dyn_state.dofs.cdofd_ang.grad[i_d, i_b]
                    g_cdofd_vel = dyn_state.dofs.cdofd_vel.grad[i_d, i_b]
                    cdof_ang = dyn_state.dofs.cdof_ang[i_d, i_b]
                    cdof_vel = dyn_state.dofs.cdof_vel[i_d, i_b]
                    g_cd_ang, g_cd_vel, g_cdof_ang, g_cdof_vel = gu.motion_cross_motion_grad(
                        s_ang_primal, s_vel_primal, cdof_ang, cdof_vel, g_cdofd_ang, g_cdofd_vel
                    )
                    for k in qd.static(range(3)):
                        dyn_state.links.cd_ang_bw.grad[i_l, i_slot, i_b][k] = (
                            dyn_state.links.cd_ang_bw.grad[i_l, i_slot, i_b][k] + g_cd_ang[k]
                        )
                        dyn_state.links.cd_vel_bw.grad[i_l, i_slot, i_b][k] = (
                            dyn_state.links.cd_vel_bw.grad[i_l, i_slot, i_b][k] + g_cd_vel[k]
                        )
                        dyn_state.dofs.cdof_ang.grad[i_d, i_b][k] = (
                            dyn_state.dofs.cdof_ang.grad[i_d, i_b][k] + g_cdof_ang[k]
                        )
                        dyn_state.dofs.cdof_vel.grad[i_d, i_b][k] = (
                            dyn_state.dofs.cdof_vel.grad[i_d, i_b][k] + g_cdof_vel[k]
                        )
                    # consume cdofd_*.grad[i_d]
                    for k in qd.static(range(3)):
                        dyn_state.dofs.cdofd_ang.grad[i_d, i_b][k] = 0.0
                        dyn_state.dofs.cdofd_vel.grad[i_d, i_b][k] = 0.0
                if joint_type == gs.JOINT_TYPE.FREE:
                    # Linear dofs: the forward writes constant-zero cdofd, so the reverse is a no-op; consume the
                    # grads so they do not leak into the next substep.
                    for i_3 in qd.static(range(3)):
                        i_d = dof_start + i_3
                        for k in qd.static(range(3)):
                            dyn_state.dofs.cdofd_ang.grad[i_d, i_b][k] = 0.0
                            dyn_state.dofs.cdofd_vel.grad[i_d, i_b][k] = 0.0

                # --- Reverse the FREE linear contraction into cd_*_bw[curr].
                # Forward (FREE linear pre-motion_cross_motion): for i_3=0..2 at i_d = dof_start + i_3,
                #   _vel = cdof_vel[i_d] * vel[i_d];  atomic_add(cd_vel_bw[curr], _vel)
                #   _ang = cdof_ang[i_d] * vel[i_d];  atomic_add(cd_ang_bw[curr], _ang)
                if joint_type == gs.JOINT_TYPE.FREE:
                    func_cd_contraction_bw(i_b, i_l, i_slot, dof_start, dof_start + 3, dyn_state)

            # --- Step 1 (initial cvel setup) reverse:
            # Forward: cd_*_bw[i_l, 0, i_b] = parent.cd_*[i_parent, i_b] (if i_parent != -1) else 0
            # Reverse: parent.cd_*.grad[i_parent] += cd_*_bw[i_l, 0].grad; consume slot 0
            g_cd_vel_slot0 = dyn_state.links.cd_vel_bw.grad[i_l, 0, i_b]
            g_cd_ang_slot0 = dyn_state.links.cd_ang_bw.grad[i_l, 0, i_b]
            if i_parent != -1:
                for k in qd.static(range(3)):
                    dyn_state.links.cd_vel.grad[i_parent, i_b][k] = (
                        dyn_state.links.cd_vel.grad[i_parent, i_b][k] + g_cd_vel_slot0[k]
                    )
                    dyn_state.links.cd_ang.grad[i_parent, i_b][k] = (
                        dyn_state.links.cd_ang.grad[i_parent, i_b][k] + g_cd_ang_slot0[k]
                    )
            # consume slot 0
            for k in qd.static(range(3)):
                dyn_state.links.cd_vel_bw.grad[i_l, 0, i_b][k] = 0.0
                dyn_state.links.cd_ang_bw.grad[i_l, 0, i_b][k] = 0.0


@qd.kernel(fastcache=True)
def kernel_manual_compute_qacc_bw(
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    """Manual backward for func_compute_qacc via the implicit function theorem (IFT).

    Forward chain (func_compute_qacc):
        acc_smooth = M^{-1} . force   (per-block LDL^T solve in func_solve_mass)
        acc[i]     = acc_smooth[i]    (identity copy)

    Reverse chain (manual, by IFT and symmetry of M = L^T D L):
        acc_smooth.grad += acc.grad   (reverse of the identity copy; acc.grad is then consumed since the forward
                                       copy overwrites acc)
        force_contrib    = M^{-1} . acc_smooth.grad   (M is symmetric, so M^{-T} = M^{-1})
        force.grad      += force_contrib
        mass_mat[i, i].grad += -force_contrib[i] * acc_smooth[i]
        mass_mat[i, j].grad += -(force_contrib[i] * acc_smooth[j] + force_contrib[j] * acc_smooth[i])    (i > j)
    mass_mat is stored lower-triangular with the upper half implicit by symmetry, so each off-diagonal parameter
    combines the chain terms of both its (i, j) and (j, i) occurrences. The forward factored the dense mass_mat into
    mass_mat_L / mass_mat_D_inv already, so only mass_mat.grad is touched here; this kernel is the single place the
    backward path populates it, and kernel_forward_dynamics_without_qacc.grad then reverses it into link poses.

    Like func_solve_mass_block, the triangular solves and the IFT outer product are restricted to the mass blocks
    rooted in each entity (see entities_mass_block_dof_start in array_class.py): elimination never crosses a block,
    and cross-block mass entries are structural zeros whose grads must stay zero.
    """
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_e, i_b in qd.ndrange(dyn_info.entities.n_links.shape[0], dyn_state.dofs.force.shape[1]):
        if rigid_info.mass_mat_mask[i_e, i_b]:
            blocks_dof_start = rigid_info.entities_mass_block_dof_start[i_e]
            blocks_dof_end = rigid_info.entities_mass_block_dof_end[i_e]

            # Reverse of acc[i] = acc_smooth[i]: drain acc.grad into the acc_smooth.grad seed, stashed in
            # acc_smooth_bw[0] as the input of the LDL^T reverse solve. acc.grad is consumed since the forward copy
            # overwrites acc.
            for i_d in range(blocks_dof_start, blocks_dof_end):
                dyn_state.dofs.acc_smooth_bw[0, i_d, i_b] = (
                    dyn_state.dofs.acc_smooth.grad[i_d, i_b] + dyn_state.dofs.acc.grad[i_d, i_b]
                )
                dyn_state.dofs.acc.grad[i_d, i_b] = 0.0
                dyn_state.dofs.acc_smooth.grad[i_d, i_b] = 0.0

            # Step 1: solve L^T . u = seed (input from [0], output to [1])
            #   u[i] = seed[i] - sum_{j>i} L[j,i] * u[j]
            for i_d_ in range(blocks_dof_end - blocks_dof_start):
                i_d = blocks_dof_end - i_d_ - 1
                block_end = rigid_info.dofs_mass_block_end[i_d]
                curr = dyn_state.dofs.acc_smooth_bw[0, i_d, i_b]
                for j_d in range(i_d + 1, block_end):
                    curr = curr - rigid_info.mass_mat_L[j_d, i_d, i_b] * dyn_state.dofs.acc_smooth_bw[1, j_d, i_b]
                dyn_state.dofs.acc_smooth_bw[1, i_d, i_b] = curr

            # Step 2: v = D^{-1} . u (output to [0], overwriting input)
            for i_d in range(blocks_dof_start, blocks_dof_end):
                dyn_state.dofs.acc_smooth_bw[0, i_d, i_b] = (
                    dyn_state.dofs.acc_smooth_bw[1, i_d, i_b] * rigid_info.mass_mat_D_inv[i_d, i_b]
                )

            # Step 3: solve L . delta = v (input from [0], output to [1])
            #   delta[i] = v[i] - sum_{j<i} L[i,j] * delta[j]
            for i_d in range(blocks_dof_start, blocks_dof_end):
                block_start = rigid_info.dofs_mass_block_start[i_d]
                curr = dyn_state.dofs.acc_smooth_bw[0, i_d, i_b]
                for j_d in range(block_start, i_d):
                    curr = curr - rigid_info.mass_mat_L[i_d, j_d, i_b] * dyn_state.dofs.acc_smooth_bw[1, j_d, i_b]
                dyn_state.dofs.acc_smooth_bw[1, i_d, i_b] = curr

            # Accumulate into force.grad.
            for i_d in range(blocks_dof_start, blocks_dof_end):
                dyn_state.dofs.force.grad[i_d, i_b] = (
                    dyn_state.dofs.force.grad[i_d, i_b] + dyn_state.dofs.acc_smooth_bw[1, i_d, i_b]
                )

            # IFT seed for mass_mat.grad, restricted to each lower-triangular in-block pair (see the docstring).
            for i_d in range(blocks_dof_start, blocks_dof_end):
                block_start = rigid_info.dofs_mass_block_start[i_d]
                force_contrib_i = dyn_state.dofs.acc_smooth_bw[1, i_d, i_b]
                acc_smooth_i = dyn_state.dofs.acc_smooth[i_d, i_b]
                rigid_info.mass_mat.grad[i_d, i_d, i_b] = (
                    rigid_info.mass_mat.grad[i_d, i_d, i_b] - force_contrib_i * acc_smooth_i
                )
                for j_d in range(block_start, i_d):
                    force_contrib_j = dyn_state.dofs.acc_smooth_bw[1, j_d, i_b]
                    acc_smooth_j = dyn_state.dofs.acc_smooth[j_d, i_b]
                    rigid_info.mass_mat.grad[i_d, j_d, i_b] = rigid_info.mass_mat.grad[i_d, j_d, i_b] - (
                        force_contrib_i * acc_smooth_j + force_contrib_j * acc_smooth_i
                    )

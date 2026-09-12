import quadrants as qd

import genesis as gs
import genesis.utils.array_class as array_class
import genesis.utils.geom as gu


@qd.func
def linear_to_lower_tri(i_pair: qd.i32, strict: qd.template() = False):
    """Convert a linear index into (row, col) of a lower-triangular matrix.

    Maps i_pair -> (i_d1, i_d2) over the lower triangle in the order (0,0), (1,0), (1,1), (2,0), ...
    (i_pair = i_d1 * (i_d1 + 1) / 2 + i_d2, i_d2 in [0, i_d1]). When ``strict`` the diagonal is excluded, mapping over
    the strict lower triangle (1,0), (2,0), (2,1), (3,0), ... (i_pair = i_d1 * (i_d1 - 1) / 2 + i_d2, i_d2 in [0, i_d1)).

    Uses a float sqrt with an integer post-correction to handle GPUs whose sqrt is not correctly rounded for perfect
    squares (observed on Apple Metal where e.g. sqrt(11881) returns ~108.999 instead of 109). Without it the row index
    lands one short on every j=0 boundary, silently dropping those matrix entries.
    """
    offset = qd.static(1.0 if strict else -1.0)
    i_d1 = qd.cast(qd.floor((qd.sqrt(qd.cast(8 * i_pair + 1, gs.qd_float)) + offset) / 2.0), qd.i32)
    i_d2 = qd.i32(0)
    if qd.static(strict):
        if (i_d1 + 1) * i_d1 // 2 <= i_pair:
            i_d1 = i_d1 + 1
        i_d2 = i_pair - i_d1 * (i_d1 - 1) // 2
    else:
        if (i_d1 + 1) * (i_d1 + 2) // 2 <= i_pair:
            i_d1 = i_d1 + 1
        i_d2 = i_pair - i_d1 * (i_d1 + 1) // 2
    return i_d1, i_d2


@qd.func
def func_wakeup_island(
    i_island,
    i_b,
    dyn_state: array_class.DynState,
    constraint_state: array_class.ConstraintState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    # Wake a hibernated component-island as a unit: every link in the island (and its DOFs and geoms) is revived and
    # appended to the awake lists, and the owning entities' flags are cleared. Waking the whole island clears its
    # daisy-chain links, which would otherwise keep re-connecting the woken links to their previous island at the next
    # partition build.
    if i_island >= 0:
        for li in range(constraint_state.island.link_slices.n[i_island, i_b]):
            link_ref = constraint_state.island.link_slices.start[i_island, i_b] + li
            i_l = constraint_state.island.link_id[link_ref, i_b]

            # Atomically claim the link by clearing its hibernation flag and reading the previous value. Only the
            # caller that observes the True->False transition appends it to the awake lists. A plain read-check-set
            # would let several wake threads targeting the same link (redundant grid threads a backend may launch, or
            # several triggers in one step) all pass the guard and append the link/DOFs once each, corrupting counts.
            was_hibernated = qd.atomic_exchange(dyn_state.links.is_hibernated[i_l, i_b], 0)

            if was_hibernated:
                constraint_state.island.hibernated_next_link[i_l, i_b] = -1
                dyn_state.links.awake_steps[i_l, i_b] = 0

                n_awake_links = qd.atomic_add(rigid_info.n_awake_links[i_b], 1)
                rigid_info.awake_links[n_awake_links, i_b] = i_l

                link_I = [i_l, i_b] if qd.static(rigid_config.batch_links_info) else i_l
                n_dofs = dyn_info.links.n_dofs[link_I]
                if n_dofs > 0:
                    base_dof_idx = dyn_info.links.dof_start[link_I]
                    base_awake_dof_idx = qd.atomic_add(rigid_info.n_awake_dofs[i_b], n_dofs)
                    for i in range(n_dofs):
                        i_d = base_dof_idx + i
                        dyn_state.dofs.is_hibernated[i_d, i_b] = False
                        rigid_info.awake_dofs[base_awake_dof_idx + i, i_b] = i_d

                for i_g in range(dyn_info.links.geom_start[link_I], dyn_info.links.geom_end[link_I]):
                    dyn_state.geoms.is_hibernated[i_g, i_b] = False

                # The entity owning this link now has an awake link; claim it for awake_entities exactly once.
                i_e = dyn_info.links.entity_idx[link_I]
                was_entity_hibernated = qd.atomic_exchange(dyn_state.entities.is_hibernated[i_e, i_b], 0)
                if was_entity_hibernated:
                    n_awake_entities = qd.atomic_add(rigid_info.n_awake_entities[i_b], 1)
                    rigid_info.awake_entities[n_awake_entities, i_b] = i_e


# --------------------------------------------------------------------------------------
# Initialization kernels
# --------------------------------------------------------------------------------------


@qd.kernel(fastcache=True)
def kernel_init_dof_fields(
    entity_idx: qd.types.ndarray(),
    dofs_motion_ang: qd.types.ndarray(),
    dofs_motion_vel: qd.types.ndarray(),
    dofs_limit: qd.types.ndarray(),
    dofs_invweight: qd.types.ndarray(),
    dofs_stiffness: qd.types.ndarray(),
    dofs_damping: qd.types.ndarray(),
    dofs_frictionloss: qd.types.ndarray(),
    dofs_armature: qd.types.ndarray(),
    dofs_act_gain: qd.types.ndarray(),
    dofs_act_bias: qd.types.ndarray(),
    dofs_force_range: qd.types.ndarray(),
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    n_dofs = dyn_state.dofs.ctrl_mode.shape[0]
    _B = dyn_state.dofs.ctrl_mode.shape[1]

    for I_d in qd.grouped(dyn_info.dofs.invweight):
        i_d = I_d[0]  # batching (if any) will be the second dim

        for j in qd.static(range(3)):
            dyn_info.dofs.motion_ang[I_d][j] = dofs_motion_ang[i_d, j]
            dyn_info.dofs.motion_vel[I_d][j] = dofs_motion_vel[i_d, j]
            dyn_info.dofs.act_bias[I_d][j] = dofs_act_bias[i_d, j]

        for j in qd.static(range(2)):
            dyn_info.dofs.limit[I_d][j] = dofs_limit[i_d, j]
            dyn_info.dofs.force_range[I_d][j] = dofs_force_range[i_d, j]

        dyn_info.dofs.armature[I_d] = dofs_armature[i_d]
        dyn_info.dofs.invweight[I_d] = dofs_invweight[i_d]
        dyn_info.dofs.stiffness[I_d] = dofs_stiffness[i_d]
        dyn_info.dofs.damping[I_d] = dofs_damping[i_d]
        dyn_info.dofs.frictionloss[I_d] = dofs_frictionloss[i_d]
        dyn_info.dofs.act_gain[I_d] = dofs_act_gain[i_d]
        dyn_info.dofs.entity_idx[I_d] = entity_idx[i_d]

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_d, i_b in qd.ndrange(n_dofs, _B):
        dyn_state.dofs.ctrl_mode[i_d, i_b] = gs.CTRL_MODE.FORCE
        dyn_state.dofs.ctrl_force[i_d, i_b] = gs.qd_float(0.0)

    if qd.static(rigid_config.use_hibernation):
        qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
        for i_d, i_b in qd.ndrange(n_dofs, _B):
            dyn_state.dofs.is_hibernated[i_d, i_b] = False
            rigid_info.awake_dofs[i_d, i_b] = i_d

        qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.ALL))
        for i_b in range(_B):
            rigid_info.n_awake_dofs[i_b] = n_dofs


@qd.kernel(fastcache=True)
def kernel_reset_hibernation(
    envs_idx: qd.types.ndarray(),
    dyn_state: array_class.DynState,
    constraint_state: array_class.ConstraintState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    # Wake every body in the given envs and rebuild the compact awake lists. A scene whose state is set (reset or
    # set_state) must resume fully awake: the restored positions and velocities are a discontinuity, and any body left
    # hibernated would stay frozen and never be re-simulated. DOFs are gathered per link (so a DOF-less scene reports
    # zero awake DOFs even though its DOF buffers are padded to at least one slot).
    n_links = dyn_state.links.is_hibernated.shape[0]
    n_geoms = dyn_state.geoms.is_hibernated.shape[0]
    n_entities = dyn_state.entities.is_hibernated.shape[0]
    max_islands = constraint_state.island.is_hibernated.shape[0]

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.ALL))
    for i_b_ in range(envs_idx.shape[0]):
        i_b = envs_idx[i_b_]
        n_awake_dofs = 0
        for i_l in range(n_links):
            dyn_state.links.is_hibernated[i_l, i_b] = False
            dyn_state.links.awake_steps[i_l, i_b] = 0
            constraint_state.island.hibernated_next_link[i_l, i_b] = -1
            rigid_info.awake_links[i_l, i_b] = i_l
            link_I = [i_l, i_b] if qd.static(rigid_config.batch_links_info) else i_l
            for i_d_ in range(dyn_info.links.n_dofs[link_I]):
                i_d = dyn_info.links.dof_start[link_I] + i_d_
                dyn_state.dofs.is_hibernated[i_d, i_b] = False
                rigid_info.awake_dofs[n_awake_dofs, i_b] = i_d
                n_awake_dofs = n_awake_dofs + 1
        rigid_info.n_awake_links[i_b] = n_links
        rigid_info.n_awake_dofs[i_b] = n_awake_dofs
        for i_g in range(n_geoms):
            dyn_state.geoms.is_hibernated[i_g, i_b] = False
        for i_e in range(n_entities):
            dyn_state.entities.is_hibernated[i_e, i_b] = False
            rigid_info.awake_entities[i_e, i_b] = i_e
        rigid_info.n_awake_entities[i_b] = n_entities
        for i_island in range(max_islands):
            constraint_state.island.is_hibernated[i_island, i_b] = 0


@qd.kernel(fastcache=True)
def kernel_init_link_fields(
    links_parent_idx: qd.types.ndarray(),
    links_root_idx: qd.types.ndarray(),
    links_q_start: qd.types.ndarray(),
    links_dof_start: qd.types.ndarray(),
    links_joint_start: qd.types.ndarray(),
    links_q_end: qd.types.ndarray(),
    links_dof_end: qd.types.ndarray(),
    links_joint_end: qd.types.ndarray(),
    links_entity_idx: qd.types.ndarray(),
    links_geom_start: qd.types.ndarray(),
    links_geom_end: qd.types.ndarray(),
    links_vgeom_start: qd.types.ndarray(),
    links_vgeom_end: qd.types.ndarray(),
    links_is_fixed: qd.types.ndarray(),
    links_pos: qd.types.ndarray(),
    links_quat: qd.types.ndarray(),
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    n_links = links_parent_idx.shape[0]
    _B = dyn_state.links.pos.shape[1]

    for I_l in qd.grouped(dyn_info.links.parent_idx):
        i_l = I_l[0]

        dyn_info.links.parent_idx[I_l] = links_parent_idx[i_l]
        dyn_info.links.root_idx[I_l] = links_root_idx[i_l]
        dyn_info.links.q_start[I_l] = links_q_start[i_l]
        dyn_info.links.joint_start[I_l] = links_joint_start[i_l]
        dyn_info.links.dof_start[I_l] = links_dof_start[i_l]
        dyn_info.links.q_end[I_l] = links_q_end[i_l]
        dyn_info.links.dof_end[I_l] = links_dof_end[i_l]
        dyn_info.links.joint_end[I_l] = links_joint_end[i_l]
        dyn_info.links.n_dofs[I_l] = links_dof_end[i_l] - links_dof_start[i_l]
        dyn_info.links.is_fixed[I_l] = links_is_fixed[i_l]
        dyn_info.links.entity_idx[I_l] = links_entity_idx[i_l]
        dyn_info.links.geom_start[I_l] = links_geom_start[i_l]
        dyn_info.links.geom_end[I_l] = links_geom_end[i_l]
        dyn_info.links.vgeom_start[I_l] = links_vgeom_start[i_l]
        dyn_info.links.vgeom_end[I_l] = links_vgeom_end[i_l]

        for j in qd.static(range(4)):
            dyn_info.links.quat[I_l][j] = links_quat[i_l, j]

        for j in qd.static(range(3)):
            dyn_info.links.pos[I_l][j] = links_pos[i_l, j]

    for i_l, i_b in qd.ndrange(n_links, _B):
        I_l = [i_l, i_b] if qd.static(rigid_config.batch_links_info) else i_l

        # Update state for root fixed link. Their state will not be updated in forward kinematics later but can be manually changed by user.
        if dyn_info.links.parent_idx[I_l] == -1 and dyn_info.links.is_fixed[I_l]:
            for j in qd.static(range(4)):
                dyn_state.links.quat[i_l, i_b][j] = links_quat[i_l, j]

            for j in qd.static(range(3)):
                dyn_state.links.pos[i_l, i_b][j] = links_pos[i_l, j]

    if qd.static(rigid_config.use_hibernation):
        qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
        for i_l, i_b in qd.ndrange(n_links, _B):
            dyn_state.links.is_hibernated[i_l, i_b] = False
            rigid_info.awake_links[i_l, i_b] = i_l

        qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.ALL))
        for i_b in range(_B):
            rigid_info.n_awake_links[i_b] = n_links


@qd.kernel(fastcache=True)
def kernel_init_link_dynamics(
    links_invweight: qd.types.ndarray(),
    links_inertial_pos: qd.types.ndarray(),
    links_inertial_quat: qd.types.ndarray(),
    links_inertial_i: qd.types.ndarray(),
    links_inertial_mass: qd.types.ndarray(),
    # Quadrants variables
    dyn_info: array_class.DynInfo,
):
    """Write each simulated link's inertial frame, inertia, mass and inverse weight."""
    for I_l in qd.grouped(dyn_info.links.invweight):
        i_l = I_l[0]

        for j in qd.static(range(2)):
            dyn_info.links.invweight[I_l][j] = links_invweight[i_l, j]

        for j in qd.static(range(4)):
            dyn_info.links.inertial_quat[I_l][j] = links_inertial_quat[i_l, j]

        for j in qd.static(range(3)):
            dyn_info.links.inertial_pos[I_l][j] = links_inertial_pos[i_l, j]

        dyn_info.links.inertial_mass[I_l] = links_inertial_mass[i_l]
        for j1, j2 in qd.static(qd.ndrange(3, 3)):
            dyn_info.links.inertial_i[I_l][j1, j2] = links_inertial_i[i_l, j1, j2]


@qd.kernel(fastcache=True)
def kernel_update_heterogeneous_links_vgeom(
    i_l: qd.i32,
    links_vgeom_start: qd.types.ndarray(),
    links_vgeom_end: qd.types.ndarray(),
    # Quadrants variables
    dyn_info: array_class.DynInfo,
):
    """Update per-environment links vgeom for heterogeneous entities."""
    _B = links_vgeom_start.shape[0]

    for i_b in range(_B):
        dyn_info.links.vgeom_start[i_l, i_b] = links_vgeom_start[i_b]
        dyn_info.links.vgeom_end[i_l, i_b] = links_vgeom_end[i_b]


@qd.kernel(fastcache=True)
def kernel_update_heterogeneous_link_info(
    i_l: qd.i32,
    links_geom_start: qd.types.ndarray(),
    links_geom_end: qd.types.ndarray(),
    links_vgeom_start: qd.types.ndarray(),
    links_vgeom_end: qd.types.ndarray(),
    links_inertial_mass: qd.types.ndarray(),
    links_inertial_pos: qd.types.ndarray(),
    links_inertial_quat: qd.types.ndarray(),
    links_inertial_i: qd.types.ndarray(),
    # Quadrants variables
    dyn_info: array_class.DynInfo,
):
    """Update per-environment link info for heterogeneous entities."""
    _B = links_geom_start.shape[0]

    for i_b in range(_B):
        dyn_info.links.geom_start[i_l, i_b] = links_geom_start[i_b]
        dyn_info.links.geom_end[i_l, i_b] = links_geom_end[i_b]
        dyn_info.links.vgeom_start[i_l, i_b] = links_vgeom_start[i_b]
        dyn_info.links.vgeom_end[i_l, i_b] = links_vgeom_end[i_b]
        dyn_info.links.inertial_mass[i_l, i_b] = links_inertial_mass[i_b]

        for j in qd.static(range(3)):
            dyn_info.links.inertial_pos[i_l, i_b][j] = links_inertial_pos[i_b, j]

        for j in qd.static(range(4)):
            dyn_info.links.inertial_quat[i_l, i_b][j] = links_inertial_quat[i_b, j]

        for j1, j2 in qd.static(qd.ndrange(3, 3)):
            dyn_info.links.inertial_i[i_l, i_b][j1, j2] = links_inertial_i[i_b, j1, j2]


@qd.kernel(fastcache=True)
def kernel_init_joint_fields(
    joints_q_start: qd.types.ndarray(),
    joints_dof_start: qd.types.ndarray(),
    joints_q_end: qd.types.ndarray(),
    joints_dof_end: qd.types.ndarray(),
    joints_type: qd.types.ndarray(),
    joints_sol_params: qd.types.ndarray(),
    joints_pos: qd.types.ndarray(),
    dyn_info: array_class.DynInfo,
    rigid_config: qd.template(),
):
    for I_j in qd.grouped(dyn_info.joints.type):
        i_j = I_j[0]

        dyn_info.joints.type[I_j] = joints_type[i_j]
        dyn_info.joints.q_start[I_j] = joints_q_start[i_j]
        dyn_info.joints.dof_start[I_j] = joints_dof_start[i_j]
        dyn_info.joints.q_end[I_j] = joints_q_end[i_j]
        dyn_info.joints.dof_end[I_j] = joints_dof_end[i_j]
        dyn_info.joints.n_dofs[I_j] = joints_dof_end[i_j] - joints_dof_start[i_j]

        for j in qd.static(range(7)):
            dyn_info.joints.sol_params[I_j][j] = joints_sol_params[i_j, j]
        for j in qd.static(range(3)):
            dyn_info.joints.pos[I_j][j] = joints_pos[i_j, j]


@qd.kernel(fastcache=True)
def kernel_init_vert_fields(
    verts_geom_idx: qd.types.ndarray(),
    verts_state_idx: qd.types.ndarray(),
    verts: qd.types.ndarray(),
    faces: qd.types.ndarray(),
    edges: qd.types.ndarray(),
    normals: qd.types.ndarray(),
    init_center_pos: qd.types.ndarray(),
    is_fixed: qd.types.ndarray(),
    dyn_info: array_class.DynInfo,
    rigid_config: qd.template(),
):
    n_verts = verts.shape[0]
    n_faces = faces.shape[0]
    n_edges = edges.shape[0]

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_v in range(n_verts):
        for j in qd.static(range(3)):
            dyn_info.verts.init_pos[i_v][j] = verts[i_v, j]
            dyn_info.verts.init_normal[i_v][j] = normals[i_v, j]
            dyn_info.verts.init_center_pos[i_v][j] = init_center_pos[i_v, j]

        dyn_info.verts.geom_idx[i_v] = verts_geom_idx[i_v]
        dyn_info.verts.verts_state_idx[i_v] = verts_state_idx[i_v]
        dyn_info.verts.is_fixed[i_v] = is_fixed[i_v]

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_f in range(n_faces):
        for j in qd.static(range(3)):
            dyn_info.faces.verts_idx[i_f][j] = faces[i_f, j]
        dyn_info.faces.geom_idx[i_f] = verts_geom_idx[faces[i_f, 0]]

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_ed in range(n_edges):
        dyn_info.edges.v0[i_ed] = edges[i_ed, 0]
        dyn_info.edges.v1[i_ed] = edges[i_ed, 1]
        # minus = verts_info.init_pos[edges[i_ed, 0]] - verts_info.init_pos[edges[i_ed, 1]]
        # edges_info.length[i_ed] = minus.norm()
        # FIXME: the line below does not work
        dyn_info.edges.length[i_ed] = (
            dyn_info.verts.init_pos[edges[i_ed, 0]] - dyn_info.verts.init_pos[edges[i_ed, 1]]
        ).norm()


@qd.kernel(fastcache=True)
def kernel_init_vvert_fields(
    vverts_vgeom_idx: qd.types.ndarray(),
    vverts_state_idx: qd.types.ndarray(),
    vverts: qd.types.ndarray(),
    vfaces: qd.types.ndarray(),
    vnormals: qd.types.ndarray(),
    dyn_info: array_class.DynInfo,
    rigid_config: qd.template(),
):
    n_vverts = vverts.shape[0]
    n_vfaces = vfaces.shape[0]

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_vv in range(n_vverts):
        for j in qd.static(range(3)):
            dyn_info.vverts.init_pos[i_vv][j] = vverts[i_vv, j]
            dyn_info.vverts.init_vnormal[i_vv][j] = vnormals[i_vv, j]

        dyn_info.vverts.vgeom_idx[i_vv] = vverts_vgeom_idx[i_vv]
        dyn_info.vverts.vverts_state_idx[i_vv] = vverts_state_idx[i_vv]

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_vf in range(n_vfaces):
        for j in qd.static(range(3)):
            dyn_info.vfaces.vverts_idx[i_vf][j] = vfaces[i_vf, j]
        dyn_info.vfaces.vgeom_idx[i_vf] = vverts_vgeom_idx[vfaces[i_vf, 0]]


@qd.kernel(fastcache=True)
def kernel_init_geom_fields(
    geoms_link_idx: qd.types.ndarray(),
    geoms_vert_start: qd.types.ndarray(),
    geoms_face_start: qd.types.ndarray(),
    geoms_edge_start: qd.types.ndarray(),
    geoms_verts_state_start: qd.types.ndarray(),
    geoms_vert_end: qd.types.ndarray(),
    geoms_face_end: qd.types.ndarray(),
    geoms_edge_end: qd.types.ndarray(),
    geoms_verts_state_end: qd.types.ndarray(),
    geoms_pos: qd.types.ndarray(),
    geoms_center: qd.types.ndarray(),
    geoms_quat: qd.types.ndarray(),
    geoms_type: qd.types.ndarray(),
    geoms_friction: qd.types.ndarray(),
    geoms_friction_torsional: qd.types.ndarray(),
    geoms_friction_rolling: qd.types.ndarray(),
    geoms_sol_params: qd.types.ndarray(),
    geoms_data: qd.types.ndarray(),
    geoms_is_convex: qd.types.ndarray(),
    geoms_needs_coup: qd.types.ndarray(),
    geoms_contype: qd.types.ndarray(),
    geoms_conaffinity: qd.types.ndarray(),
    geoms_coup_softness: qd.types.ndarray(),
    geoms_coup_friction: qd.types.ndarray(),
    geoms_coup_restitution: qd.types.ndarray(),
    geoms_is_fixed: qd.types.ndarray(),
    geoms_is_decomp: qd.types.ndarray(),
    geoms_is_hollow: qd.types.ndarray(),
    geoms_init_AABB: array_class.GeomsInitAABB,  # TODO: move to rigid global info
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_config: qd.template(),
):
    n_geoms = geoms_pos.shape[0]
    _B = dyn_state.geoms.friction_ratio.shape[1]

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_g in range(n_geoms):
        for j in qd.static(range(3)):
            dyn_info.geoms.pos[i_g][j] = geoms_pos[i_g, j]
            dyn_info.geoms.center[i_g][j] = geoms_center[i_g, j]

        for j in qd.static(range(4)):
            dyn_info.geoms.quat[i_g][j] = geoms_quat[i_g, j]

        for j in qd.static(range(7)):
            dyn_info.geoms.data[i_g][j] = geoms_data[i_g, j]
            dyn_info.geoms.sol_params[i_g][j] = geoms_sol_params[i_g, j]

        dyn_info.geoms.vert_start[i_g] = geoms_vert_start[i_g]
        dyn_info.geoms.vert_end[i_g] = geoms_vert_end[i_g]
        dyn_info.geoms.vert_num[i_g] = geoms_vert_end[i_g] - geoms_vert_start[i_g]

        dyn_info.geoms.face_start[i_g] = geoms_face_start[i_g]
        dyn_info.geoms.face_end[i_g] = geoms_face_end[i_g]
        dyn_info.geoms.face_num[i_g] = geoms_face_end[i_g] - geoms_face_start[i_g]

        dyn_info.geoms.edge_start[i_g] = geoms_edge_start[i_g]
        dyn_info.geoms.edge_end[i_g] = geoms_edge_end[i_g]
        dyn_info.geoms.edge_num[i_g] = geoms_edge_end[i_g] - geoms_edge_start[i_g]

        dyn_info.geoms.verts_state_start[i_g] = geoms_verts_state_start[i_g]
        dyn_info.geoms.verts_state_end[i_g] = geoms_verts_state_end[i_g]

        dyn_info.geoms.link_idx[i_g] = geoms_link_idx[i_g]
        dyn_info.geoms.type[i_g] = geoms_type[i_g]
        dyn_info.geoms.friction[i_g] = geoms_friction[i_g]
        dyn_info.geoms.friction_torsional[i_g] = geoms_friction_torsional[i_g]
        dyn_info.geoms.friction_rolling[i_g] = geoms_friction_rolling[i_g]

        dyn_info.geoms.is_convex[i_g] = geoms_is_convex[i_g]
        dyn_info.geoms.is_hollow[i_g] = geoms_is_hollow[i_g]
        dyn_info.geoms.needs_coup[i_g] = geoms_needs_coup[i_g]
        dyn_info.geoms.contype[i_g] = geoms_contype[i_g]
        dyn_info.geoms.conaffinity[i_g] = geoms_conaffinity[i_g]

        dyn_info.geoms.coup_softness[i_g] = geoms_coup_softness[i_g]
        dyn_info.geoms.coup_friction[i_g] = geoms_coup_friction[i_g]
        dyn_info.geoms.coup_restitution[i_g] = geoms_coup_restitution[i_g]

        dyn_info.geoms.is_fixed[i_g] = geoms_is_fixed[i_g]
        dyn_info.geoms.is_decomposed[i_g] = geoms_is_decomp[i_g]

        # compute init AABB.
        # Beware the ordering the this corners is critical and MUST NOT be changed as this order is used elsewhere in
        # the codebase, e.g. overlap estimation between two convex geometries using there bounding boxes. For
        # primitives, use exact analytical AABB bounds rather than tessellated mesh vertices, which are inscribed in
        # the true surface. Using the mesh AABB would shrink the box and cause broadphase to false-negative shallow
        # penetrations.
        lower = gu.qd_vec3(qd.math.inf)
        upper = gu.qd_vec3(-qd.math.inf)
        geom_type = geoms_type[i_g]
        if geom_type == gs.GEOM_TYPE.SPHERE:
            radius = geoms_data[i_g, 0]
            lower = qd.Vector([-radius, -radius, -radius], dt=gs.qd_float)
            upper = qd.Vector([radius, radius, radius], dt=gs.qd_float)
        elif geom_type == gs.GEOM_TYPE.ELLIPSOID:
            a = geoms_data[i_g, 0]
            b = geoms_data[i_g, 1]
            c = geoms_data[i_g, 2]
            lower = qd.Vector([-a, -b, -c], dt=gs.qd_float)
            upper = qd.Vector([a, b, c], dt=gs.qd_float)
        elif geom_type == gs.GEOM_TYPE.CAPSULE:
            radius = geoms_data[i_g, 0]
            half_length = 0.5 * geoms_data[i_g, 1]
            lower = qd.Vector([-radius, -radius, -(half_length + radius)], dt=gs.qd_float)
            upper = qd.Vector([radius, radius, half_length + radius], dt=gs.qd_float)
        elif geom_type == gs.GEOM_TYPE.CYLINDER:
            radius = geoms_data[i_g, 0]
            half_length = 0.5 * geoms_data[i_g, 1]
            lower = qd.Vector([-radius, -radius, -half_length], dt=gs.qd_float)
            upper = qd.Vector([radius, radius, half_length], dt=gs.qd_float)
        else:
            for i_v in range(geoms_vert_start[i_g], geoms_vert_end[i_g]):
                lower = qd.min(lower, dyn_info.verts.init_pos[i_v])
                upper = qd.max(upper, dyn_info.verts.init_pos[i_v])
        geoms_init_AABB[i_g, 0] = qd.Vector([lower[0], lower[1], lower[2]], dt=gs.qd_float)
        geoms_init_AABB[i_g, 1] = qd.Vector([lower[0], lower[1], upper[2]], dt=gs.qd_float)
        geoms_init_AABB[i_g, 2] = qd.Vector([lower[0], upper[1], lower[2]], dt=gs.qd_float)
        geoms_init_AABB[i_g, 3] = qd.Vector([lower[0], upper[1], upper[2]], dt=gs.qd_float)
        geoms_init_AABB[i_g, 4] = qd.Vector([upper[0], lower[1], lower[2]], dt=gs.qd_float)
        geoms_init_AABB[i_g, 5] = qd.Vector([upper[0], lower[1], upper[2]], dt=gs.qd_float)
        geoms_init_AABB[i_g, 6] = qd.Vector([upper[0], upper[1], lower[2]], dt=gs.qd_float)
        geoms_init_AABB[i_g, 7] = qd.Vector([upper[0], upper[1], upper[2]], dt=gs.qd_float)

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_g, i_b in qd.ndrange(n_geoms, _B):
        dyn_state.geoms.friction_ratio[i_g, i_b] = 1.0


@qd.kernel(fastcache=True)
def kernel_init_vgeom_fields(
    vgeoms_link_idx: qd.types.ndarray(),
    vgeoms_vvert_start: qd.types.ndarray(),
    vgeoms_vface_start: qd.types.ndarray(),
    vgeoms_vvert_end: qd.types.ndarray(),
    vgeoms_vface_end: qd.types.ndarray(),
    vgeoms_pos: qd.types.ndarray(),
    vgeoms_quat: qd.types.ndarray(),
    vgeoms_color: qd.types.ndarray(),
    dyn_info: array_class.DynInfo,
    rigid_config: qd.template(),
):
    n_vgeoms = vgeoms_pos.shape[0]

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_vg in range(n_vgeoms):
        for j in qd.static(range(3)):
            dyn_info.vgeoms.pos[i_vg][j] = vgeoms_pos[i_vg, j]

        for j in qd.static(range(4)):
            dyn_info.vgeoms.quat[i_vg][j] = vgeoms_quat[i_vg, j]

        dyn_info.vgeoms.vvert_start[i_vg] = vgeoms_vvert_start[i_vg]
        dyn_info.vgeoms.vvert_end[i_vg] = vgeoms_vvert_end[i_vg]
        dyn_info.vgeoms.vvert_num[i_vg] = vgeoms_vvert_end[i_vg] - vgeoms_vvert_start[i_vg]

        dyn_info.vgeoms.vface_start[i_vg] = vgeoms_vface_start[i_vg]
        dyn_info.vgeoms.vface_end[i_vg] = vgeoms_vface_end[i_vg]
        dyn_info.vgeoms.vface_num[i_vg] = vgeoms_vface_end[i_vg] - vgeoms_vface_start[i_vg]

        dyn_info.vgeoms.link_idx[i_vg] = vgeoms_link_idx[i_vg]
        for j in qd.static(range(4)):
            dyn_info.vgeoms.color[i_vg][j] = vgeoms_color[i_vg, j]


@qd.kernel(fastcache=True)
def kernel_init_entity_fields(
    entities_dof_start: qd.types.ndarray(),
    entities_dof_end: qd.types.ndarray(),
    entities_link_start: qd.types.ndarray(),
    entities_link_end: qd.types.ndarray(),
    entities_geom_start: qd.types.ndarray(),
    entities_geom_end: qd.types.ndarray(),
    entities_gravity_compensation: qd.types.ndarray(),
    entities_is_local_collision_mask: qd.types.ndarray(),
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    n_entities = entities_dof_start.shape[0]

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_e in range(n_entities):
        dyn_info.entities.dof_start[i_e] = entities_dof_start[i_e]
        dyn_info.entities.dof_end[i_e] = entities_dof_end[i_e]
        dyn_info.entities.n_dofs[i_e] = entities_dof_end[i_e] - entities_dof_start[i_e]

        dyn_info.entities.link_start[i_e] = entities_link_start[i_e]
        dyn_info.entities.link_end[i_e] = entities_link_end[i_e]
        dyn_info.entities.n_links[i_e] = entities_link_end[i_e] - entities_link_start[i_e]

        dyn_info.entities.geom_start[i_e] = entities_geom_start[i_e]
        dyn_info.entities.geom_end[i_e] = entities_geom_end[i_e]
        dyn_info.entities.n_geoms[i_e] = entities_geom_end[i_e] - entities_geom_start[i_e]

        dyn_info.entities.gravity_compensation[i_e] = entities_gravity_compensation[i_e]
        dyn_info.entities.is_local_collision_mask[i_e] = entities_is_local_collision_mask[i_e]

    if qd.static(rigid_config.use_hibernation):
        _B = dyn_state.entities.is_hibernated.shape[1]

        qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
        for i_e, i_b in qd.ndrange(n_entities, _B):
            dyn_state.entities.is_hibernated[i_e, i_b] = False
            rigid_info.awake_entities[i_e, i_b] = i_e

        qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.ALL))
        for i_b in range(_B):
            rigid_info.n_awake_entities[i_b] = n_entities


@qd.kernel(fastcache=True)
def kernel_init_equality_fields(
    equalities_type: qd.types.ndarray(),
    equalities_eq_obj1id: qd.types.ndarray(),
    equalities_eq_obj2id: qd.types.ndarray(),
    equalities_eq_data: qd.types.ndarray(),
    equalities_eq_type: qd.types.ndarray(),
    equalities_sol_params: qd.types.ndarray(),
    # Quadrants variables
    dyn_info: array_class.DynInfo,
    rigid_config: qd.template(),
):
    n_equalities = equalities_eq_obj1id.shape[0]
    _B = dyn_info.equalities.eq_obj1id.shape[1]

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_eq, i_b in qd.ndrange(n_equalities, _B):
        dyn_info.equalities.eq_obj1id[i_eq, i_b] = equalities_eq_obj1id[i_eq]
        dyn_info.equalities.eq_obj2id[i_eq, i_b] = equalities_eq_obj2id[i_eq]
        dyn_info.equalities.eq_type[i_eq, i_b] = equalities_eq_type[i_eq]
        for j in qd.static(range(11)):
            dyn_info.equalities.eq_data[i_eq, i_b][j] = equalities_eq_data[i_eq, j]
        for j in qd.static(range(7)):
            dyn_info.equalities.sol_params[i_eq, i_b][j] = equalities_sol_params[i_eq, j]


# --------------------------------------------------------------------------------------
# External force kernels
# --------------------------------------------------------------------------------------


@qd.kernel(fastcache=True)
def kernel_apply_links_external_wrench(
    links_idx: qd.types.ndarray(),
    envs_idx: qd.types.ndarray(),
    force: qd.types.ndarray(),
    torque: qd.types.ndarray(),
    dyn_state: array_class.DynState,
    rigid_config: qd.template(),
    ref: qd.template(),
    local: qd.template(),
):
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.ALL))
    for i_l_, i_b_ in qd.ndrange(links_idx.shape[0], envs_idx.shape[0]):
        force_i = qd.Vector([force[i_b_, i_l_, 0], force[i_b_, i_l_, 1], force[i_b_, i_l_, 2]], dt=gs.qd_float)
        torque_i = qd.Vector([torque[i_b_, i_l_, 0], torque[i_b_, i_l_, 1], torque[i_b_, i_l_, 2]], dt=gs.qd_float)
        func_apply_link_external_wrench(
            links_idx[i_l_],
            envs_idx[i_b_],
            qd.Vector.zero(gs.qd_float, 3),
            force_i,
            torque_i,
            dyn_state,
            ref,
            local,
            has_pos=False,
        )


@qd.kernel(fastcache=True)
def kernel_apply_links_external_wrench_at_pos(
    links_idx: qd.types.ndarray(),
    envs_idx: qd.types.ndarray(),
    pos: qd.types.ndarray(),
    force: qd.types.ndarray(),
    torque: qd.types.ndarray(),
    dyn_state: array_class.DynState,
    rigid_config: qd.template(),
    ref: qd.template(),
    local: qd.template(),
):
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.ALL))
    for i_l_, i_b_ in qd.ndrange(links_idx.shape[0], envs_idx.shape[0]):
        pos_i = qd.Vector([pos[i_b_, i_l_, 0], pos[i_b_, i_l_, 1], pos[i_b_, i_l_, 2]], dt=gs.qd_float)
        force_i = qd.Vector([force[i_b_, i_l_, 0], force[i_b_, i_l_, 1], force[i_b_, i_l_, 2]], dt=gs.qd_float)
        torque_i = qd.Vector([torque[i_b_, i_l_, 0], torque[i_b_, i_l_, 1], torque[i_b_, i_l_, 2]], dt=gs.qd_float)
        func_apply_link_external_wrench(
            links_idx[i_l_], envs_idx[i_b_], pos_i, force_i, torque_i, dyn_state, ref, local, has_pos=True
        )


@qd.func
def func_apply_coupling_force(link_idx, env_idx, pos, force, links_state: array_class.LinksState):
    torque = (pos - links_state.root_COM[link_idx, env_idx]).cross(force)
    links_state.cfrc_coupling_ang[link_idx, env_idx] -= torque
    links_state.cfrc_coupling_vel[link_idx, env_idx] -= force


@qd.kernel(fastcache=True)
def kernel_wakeup_coupled_links(
    dyn_state: array_class.DynState,
    constraint_state: array_class.ConstraintState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    """
    Wake the hibernated islands of links holding a pending coupling force.

    Couplers accumulate forces into cfrc_coupling_* directly, bypassing the wake-aware public force API, while
    forward dynamics only consumes the forces of awake entities before clearing the field. Waking the receiving
    islands before the forces are consumed keeps hibernated links responsive to the other solvers and preserves
    the momentum exchange whose opposite half has already been applied on the coupled side.
    """
    qd.loop_config(serialize=rigid_config.para_level < gs.PARA_LEVEL.PARTIAL)
    for i_l, i_b in qd.ndrange(dyn_state.links.is_hibernated.shape[0], dyn_state.links.is_hibernated.shape[1]):
        if dyn_state.links.is_hibernated[i_l, i_b] and (
            dyn_state.links.cfrc_coupling_vel[i_l, i_b].norm_sqr() > 0
            or dyn_state.links.cfrc_coupling_ang[i_l, i_b].norm_sqr() > 0
        ):
            func_wakeup_island(
                constraint_state.island.links_island_idx[i_l, i_b],
                i_b,
                dyn_state,
                constraint_state,
                dyn_info,
                rigid_info,
                rigid_config,
            )


@qd.func
def func_apply_link_external_wrench(
    link_idx,
    env_idx,
    pos,
    force,
    torque,
    dyn_state: array_class.DynState,
    ref: qd.template(),
    local: qd.template(),
    has_pos: qd.template(),
):
    # The generalized force is expressed about the root COM, so a linear force adds the moment of the arm going from
    # the root COM to its application point, while a torque is a free couple.
    arm = qd.Vector.zero(gs.qd_float, 3)
    if qd.static(ref == gs.link_ref_frame.link_COM):
        if qd.static(local):
            i_quat = dyn_state.links.i_quat[link_idx, env_idx]
            force = gu.qd_transform_by_quat(force, i_quat)
            torque = gu.qd_transform_by_quat(torque, i_quat)
            if qd.static(has_pos):
                pos = gu.qd_transform_by_quat(pos, i_quat)
        arm = dyn_state.links.i_pos[link_idx, env_idx]
    if qd.static(ref == gs.link_ref_frame.link_origin):
        if qd.static(local):
            quat = dyn_state.links.quat[link_idx, env_idx]
            force = gu.qd_transform_by_quat(force, quat)
            torque = gu.qd_transform_by_quat(torque, quat)
            if qd.static(has_pos):
                pos = gu.qd_transform_by_quat(pos, quat)
        arm = dyn_state.links.pos[link_idx, env_idx] - dyn_state.links.root_COM[link_idx, env_idx]
    if qd.static(has_pos):
        # A local point is an offset from the reference frame origin, whose own arm has just been computed, while a
        # world point already locates the application point and makes the reference frame irrelevant.
        if qd.static(local):
            arm = arm + pos
        else:
            arm = pos - dyn_state.links.root_COM[link_idx, env_idx]

    dyn_state.links.cfrc_applied_vel[link_idx, env_idx] -= force
    dyn_state.links.cfrc_applied_ang[link_idx, env_idx] -= torque + arm.cross(force)


@qd.func
def func_clear_external_force(
    dyn_state: array_class.DynState, rigid_info: array_class.RigidInfo, rigid_config: qd.template()
):
    n_links = dyn_state.links.pos.shape[0]
    _B = dyn_state.links.pos.shape[1]

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.ALL))
    for i_0, i_b in qd.ndrange(1, _B) if qd.static(rigid_config.use_hibernation) else qd.ndrange(n_links, _B):
        for i_1 in (
            range(rigid_info.n_awake_links[i_b]) if qd.static(rigid_config.use_hibernation) else qd.static(range(1))
        ):
            i_l = rigid_info.awake_links[i_1, i_b] if qd.static(rigid_config.use_hibernation) else i_0
            dyn_state.links.cfrc_applied_ang[i_l, i_b] = qd.Vector.zero(gs.qd_float, 3)
            dyn_state.links.cfrc_applied_vel[i_l, i_b] = qd.Vector.zero(gs.qd_float, 3)


# --------------------------------------------------------------------------------------
# Render transform kernels
# --------------------------------------------------------------------------------------


@qd.kernel(fastcache=True)
def kernel_bit_reduction(tensor: qd.Tensor) -> qd.i32:
    flag = qd.i32(0)
    for i in range(tensor.shape[0]):
        flag = qd.atomic_or(flag, tensor[i])
    return flag


@qd.kernel(fastcache=True)
def kernel_set_zero(envs_idx: qd.types.ndarray(), tensor: qd.Tensor):
    for i_b_ in range(envs_idx.shape[0]):
        tensor[i_b_] = 0


@qd.func
def func_atomic_add_if(I, value, field: qd.Tensor, cond: qd.template()):
    if qd.static(cond):
        qd.atomic_add(field[I], value)
    return value


@qd.func
def func_add_safe_backward(I, value, field: qd.Tensor, cond: qd.template()):
    # Use (expensive) atomic add in backward for differentiability -- when there is race condition on the field to
    # write, use atomic add directly. For reference, see official Quadrants documentation:
    # https://docs.taichi-lang.org/docs/differentiable_programming#global-data-access-rules
    if qd.static(cond):
        qd.atomic_add(field[I], value)
    else:
        field[I] = field[I] + value


@qd.func
def func_read_field_if(I, value, field: qd.Tensor, cond: qd.template()):
    return field[I] if qd.static(cond) else value


@qd.func
def func_write_field_if(I, value, field: qd.Tensor, cond: qd.template()):
    if qd.static(cond):
        field[I] = value
    return value


@qd.func
def func_write_and_read_field_if(I, value, field: qd.Tensor, cond: qd.template()):
    if qd.static(cond):
        field[I] = value
    return field[I] if qd.static(cond) else value


@qd.func
def func_check_index_range(idx: qd.i32, min: qd.i32, max: qd.i32, cond: qd.template()):
    # Conditionally check if the index is in the range [min, max) to save computational cost
    return (idx >= min and idx < max) if qd.static(cond) else True


@qd.kernel(fastcache=True)
def kernel_clear_external_force(
    dyn_state: array_class.DynState, rigid_info: array_class.RigidInfo, rigid_config: qd.template()
):
    func_clear_external_force(dyn_state, rigid_info, rigid_config)


from genesis.utils.deprecated_module_wrapper import create_virtual_deprecated_module

create_virtual_deprecated_module(__name__, "genesis.engine.solvers.rigid.rigid_solver_util_decomp")

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import quadrants as qd

import genesis as gs
from genesis.engine.solvers.rigid.abd.inverse_kinematics import (
    func_forward_kinematics_scratch,
)
from genesis.utils import geom as gu

from .bcoo_operations import sym_bcoo_spmv_naive
from .rigid_contact_proxy_kkt import (
    expand_proxy_twist,
    expand_slack_twist,
    restrict_proxy_wrench,
    restrict_slack_wrench,
    rigid_contact_proxy_skew,
)
from .rigid_system import RigidSystem
from .sim_system import SimData, SimSystem

_FUSED_MAX_TREE_SIZE = 64
_FUSED_MAX_TREES = 8

if TYPE_CHECKING:
    from genesis.engine.solvers.rigid.rigid_solver import RigidSolver

    from .rigid_contact_proxy import RigidContactProxySystem


@qd.data_oriented  # WORKAROUND: Quadrants bound @qd.func self must be data-oriented.
class RigidJointForestSystem(SimSystem):
    """Organize the minimal-coordinate link/proxy forest transform."""

    @qd.data_oriented
    class Data(SimData):
        """Complete forest topology and numerical scratch."""

        n_bodies: qd.Ndarray
        n_mechanism_bodies: qd.Ndarray
        n_edges: qd.Ndarray
        n_trees: qd.Ndarray
        max_tree_size: qd.Ndarray
        max_depth: qd.Ndarray
        n_levels: qd.Ndarray
        total_dof: qd.Ndarray
        proxy_dof_offset: qd.Ndarray
        parent_body: qd.Ndarray
        parent_edge: qd.Ndarray
        depth: qd.Ndarray
        tree_id: qd.Ndarray
        depth_start: qd.Ndarray
        depth_order: qd.Ndarray
        tree_roots: qd.Ndarray
        tree_body_start: qd.Ndarray
        tree_body_list: qd.Ndarray
        tree_local_index: qd.Ndarray
        edge_parent: qd.Ndarray
        edge_child: qd.Ndarray
        edge_dof_index: qd.Ndarray
        root_dof_index: qd.Ndarray
        body_twist: qd.Ndarray
        body_wrench: qd.Ndarray
        physical_p: qd.Ndarray
        physical_Ap: qd.Ndarray
        articulated_inertia: qd.Ndarray
        body_inertia: qd.Ndarray
        edge_u: qd.Ndarray
        edge_hessian: qd.Ndarray
        edge_basis: qd.Ndarray
        edge_arm: qd.Ndarray
        edge_d: qd.Ndarray
        root_inverse: qd.Ndarray
        precond_force: qd.Ndarray
        precond_a: qd.Ndarray
        precond_velocity: qd.Ndarray
        kkt_proxy_diagonal: qd.Ndarray
        endpoint_qpos: qd.Tensor
        endpoint_link_pos: qd.Tensor
        endpoint_link_quat: qd.Tensor
        endpoint_joint_xanchor: qd.Tensor
        endpoint_joint_xaxis: qd.Tensor
        endpoint_t: qd.Tensor
        endpoint_quat: qd.Tensor

    def __init__(self) -> None:
        super().__init__()
        self.data = self.Data()
        self._rigid_solver: RigidSolver | None = None
        self._total_dof: int | None = None
        self._n_rigid_bodies: int | None = None
        self._proxy_dof_offset: int | None = None
        self.fused_enabled: bool = True
        self.genesis_legacy_enabled: bool = False
        self.use_fused_tree_path: bool = False
        self.n_links: int = 0

    def wire_data(
        self,
        rigid_solver: RigidSolver,
        *,
        total_dof: int,
        n_rigid_bodies: int,
        proxy_dof_offset: int,
        fused_enabled: bool = True,
        genesis_legacy_enabled: bool = False,
    ) -> None:
        self.n_links = int(rigid_solver.n_links)
        self.fused_enabled = bool(fused_enabled)
        self.genesis_legacy_enabled = bool(genesis_legacy_enabled)
        self._rigid_solver = rigid_solver
        self._total_dof = total_dof
        self._n_rigid_bodies = n_rigid_bodies
        self._proxy_dof_offset = proxy_dof_offset

    def init(self) -> None:
        if (
            self._rigid_solver is None
            or self._total_dof is None
            or self._n_rigid_bodies is None
            or self._proxy_dof_offset is None
        ):
            raise RuntimeError("RigidJointForestSystem data has not been wired")
        _populate_rigid_joint_forest_data(
            self,
            self._rigid_solver,
            self.rigid.data,
            total_dof=self._total_dof,
            n_rigid_bodies=self._n_rigid_bodies,
            proxy_dof_offset=self._proxy_dof_offset,
            proxy_data=self.contact_proxy.data,
            n_proxy_pairs=self.contact_proxy.n_pairs,
            fused_enabled=self.fused_enabled,
            genesis_legacy_enabled=self.genesis_legacy_enabled,
        )
        self._rigid_solver = None
        self._total_dof = None
        self._n_rigid_bodies = None
        self._proxy_dof_offset = None

    def build(self) -> None:
        from .global_linear_system import GlobalLinearSystem
        from .pcg_solver import PCGSolver
        from .rigid_contact_proxy import RigidContactProxySystem

        self.rigid = self.require(RigidSystem)
        self.contact_proxy = self.find(RigidContactProxySystem)
        if self.contact_proxy is None:
            raise RuntimeError("Registered RigidJointForestSystem requires RigidContactProxySystem")
        self.linear_system_system = self.require(GlobalLinearSystem)
        pcg_solver = self.require(PCGSolver)
        self.pcg_operator_action = self.create_action(
            pcg_apply_operator,
            self,
            self.rigid.data,
            self.contact_proxy.data,
        )
        self.pcg_preconditioner_action = self.create_action(
            pcg_apply_preconditioner,
            self,
            self.rigid.data,
            self.contact_proxy.data,
        )
        pcg_solver.on_reduced_solve(
            self.pcg_operator_action,
            self.pcg_preconditioner_action,
        )

    @qd.func(requires_top_level=True)
    def on_compute_endpoint_fk(self):
        compute_endpoint_fk(self.data, self.rigid.data, self.contact_proxy.data)

    @qd.func(requires_top_level=True)
    def on_prepare_particular(self):
        prepare_particular(self.data, self.rigid.data, self.contact_proxy.data)

    @qd.func(requires_top_level=True)
    def on_particular_spmv(self):
        particular_spmv(
            self.data,
            self.rigid.data,
            self.contact_proxy.data,
            self.linear_system_system.data,
        )

    @qd.func(requires_top_level=True)
    def on_project_physical_rhs(self):
        project_physical_rhs(
            self.data,
            self.rigid.data,
            self.contact_proxy.data,
            self.linear_system_system.data,
            self.genesis_legacy_enabled,
            self.use_fused_tree_path,
            self.n_links,
        )

    @qd.func(requires_top_level=True)
    def on_build_preconditioner(self):
        build_preconditioner(
            self,
            self.rigid.data,
            self.contact_proxy.data,
            self.linear_system_system.data,
        )

    @qd.func(requires_top_level=True)
    def on_expand_solution(self):
        expand_solution(
            self,
            self.rigid.data,
            self.contact_proxy.data,
            self.linear_system_system.data.x_sol,
        )

    @qd.func(requires_top_level=True)
    def on_compute_merit_directional_derivative(self):
        compute_merit_directional_derivative(
            self.data,
            self.rigid.data,
            self.contact_proxy.data,
            self.linear_system_system.data,
        )


def _populate_rigid_joint_forest_data(
    system: RigidJointForestSystem,
    rigid_solver,
    rigid_data: RigidSystem.Data,
    *,
    total_dof: int,
    n_rigid_bodies: int,
    proxy_dof_offset: int,
    proxy_data: RigidContactProxySystem.Data | None,
    n_proxy_pairs: int,
    fused_enabled: bool = True,
    genesis_legacy_enabled: bool = False,
) -> None:
    data = system.data
    n_links = rigid_solver.n_links
    n_instances = rigid_solver._B
    parent_links = []
    link_depths = []
    edge_children = []
    edge_dof_indices = []
    root_dof_indices = []
    for link in rigid_solver.links:
        parent_links.append(link.parent_idx)
        link_depths.append(0 if link.parent_idx < 0 else link_depths[link.parent_idx] + 1)
        moving_joints = [joint for joint in link.joints if joint.n_dofs > 0]
        if link.parent_idx < 0:
            root_dof = -1
            if moving_joints:
                if (
                    len(moving_joints) != 1
                    or moving_joints[0].type != gs.JOINT_TYPE.FREE
                    or moving_joints[0].n_dofs != 6
                ):
                    raise RuntimeError("RigidJointForestSystem requires a fixed or six-DOF free Genesis root")
                root_dof = moving_joints[0].dof_start
            root_dof_indices.append(root_dof)
        else:
            root_dof_indices.append(-1)
            if moving_joints:
                if (
                    len(moving_joints) != 1
                    or moving_joints[0].n_dofs != 1
                    or moving_joints[0].type
                    not in (
                        gs.JOINT_TYPE.REVOLUTE,
                        gs.JOINT_TYPE.PRISMATIC,
                    )
                ):
                    raise RuntimeError(
                        "RigidJointForestSystem requires scalar revolute or prismatic Genesis forest edges"
                    )
                edge_children.append(link.idx)
                edge_dof_indices.append(moving_joints[0].dof_start)
    if proxy_data is None:
        raise RuntimeError("RigidJointForestSystem requires RigidContactProxySystem data")
    n_mechanism_bodies = n_links * n_instances
    if n_rigid_bodies < n_mechanism_bodies:
        raise ValueError("RigidJointForestSystem body count is smaller than Genesis link count")
    if total_dof < 0 or proxy_dof_offset < 0 or proxy_dof_offset > total_dof:
        raise ValueError("RigidJointForestSystem global DOF layout is invalid")
    required = proxy_dof_offset + n_proxy_pairs * 6
    if required > total_dof:
        raise ValueError("RigidJointForestSystem proxy dummy rows exceed global DOF layout")

    parents = np.full(n_mechanism_bodies, -1, dtype=np.int32)
    depths = np.zeros(n_mechanism_bodies, dtype=np.int32)
    for environment in range(n_instances):
        body_offset = environment * n_links
        for link, parent_link in enumerate(parent_links):
            body = body_offset + link
            if parent_link >= 0:
                parents[body] = body_offset + parent_link
                depths[body] = link_depths[link]

    children = [[] for _ in range(n_mechanism_bodies)]
    for body, parent in enumerate(parents):
        if parent >= 0:
            children[int(parent)].append(body)
    tree_roots = np.flatnonzero(parents < 0).astype(np.int32)
    tree_ids = np.full(n_rigid_bodies, -1, dtype=np.int32)
    tree_body_starts = [0]
    tree_body_order = []
    tree_local_index = np.full(n_rigid_bodies, -1, dtype=np.int32)
    max_tree_size = 0
    for tree, root in enumerate(tree_roots):
        stack = [int(root)]
        tree_bodies = []
        while stack:
            body = stack.pop()
            tree_bodies.append(body)
            stack.extend(reversed(children[body]))
        for local, body in enumerate(tree_bodies):
            tree_ids[body] = tree
            tree_local_index[body] = local
        tree_body_order.extend(tree_bodies)
        tree_body_starts.append(len(tree_body_order))
        max_tree_size = max(max_tree_size, len(tree_bodies))

    depth_order = np.argsort(depths, kind="stable").astype(np.int32)
    depth_counts = np.bincount(depths, minlength=n_links)
    depth_starts = np.zeros(n_links + 1, dtype=np.int32)
    np.cumsum(depth_counts, out=depth_starts[1:])
    n_trees = len(tree_roots)
    system.use_fused_tree_path = (
        fused_enabled
        and not genesis_legacy_enabled
        and n_trees <= _FUSED_MAX_TREES
        and max_tree_size <= _FUSED_MAX_TREE_SIZE
    )

    n_edges = len(edge_children) * n_instances
    parent_edge = np.full(n_mechanism_bodies, -1, dtype=np.int32)
    edge_parent = np.zeros(max(n_edges, 1), dtype=np.int32)
    edge_child = np.zeros(max(n_edges, 1), dtype=np.int32)
    edge_dof_index = np.zeros(max(n_edges, 1), dtype=np.int32)
    root_dof_index = np.full(n_mechanism_bodies, -1, dtype=np.int32)
    edge = 0
    for environment in range(n_instances):
        body_offset = environment * n_links
        dof_offset = environment * rigid_solver.n_dofs
        for child, dof in zip(
            edge_children,
            edge_dof_indices,
            strict=True,
        ):
            body = body_offset + child
            parent_edge[body] = edge
            edge_parent[edge] = parents[body]
            edge_child[edge] = body
            edge_dof_index[edge] = dof_offset + dof
            edge += 1
        for link, dof in enumerate(root_dof_indices):
            if dof >= 0:
                root_dof_index[body_offset + link] = dof_offset + dof

    capacity = max(n_rigid_bodies, 1)
    edge_capacity = max(n_edges, 1)
    tree_capacity = max(n_trees, 1)
    mechanism_capacity = max(n_mechanism_bodies, 1)
    data.n_bodies = qd.ndarray(qd.i32, shape=())
    data.n_mechanism_bodies = qd.ndarray(qd.i32, shape=())
    data.n_edges = qd.ndarray(qd.i32, shape=())
    data.n_trees = qd.ndarray(qd.i32, shape=())
    data.max_tree_size = qd.ndarray(qd.i32, shape=())
    data.max_depth = qd.ndarray(qd.i32, shape=())
    data.n_levels = qd.ndarray(qd.i32, shape=())
    data.total_dof = qd.ndarray(qd.i32, shape=())
    data.proxy_dof_offset = qd.ndarray(qd.i32, shape=())
    data.parent_body = qd.ndarray(qd.i32, shape=(capacity,))
    data.parent_edge = qd.ndarray(qd.i32, shape=(capacity,))
    data.depth = qd.ndarray(qd.i32, shape=(capacity,))
    data.tree_id = qd.ndarray(qd.i32, shape=(capacity,))
    data.depth_start = qd.ndarray(qd.i32, shape=(n_links + 1,))
    data.depth_order = qd.ndarray(qd.i32, shape=(mechanism_capacity,))
    data.tree_roots = qd.ndarray(qd.i32, shape=(tree_capacity,))
    data.tree_body_start = qd.ndarray(qd.i32, shape=(tree_capacity + 1,))
    data.tree_body_list = qd.ndarray(qd.i32, shape=(mechanism_capacity,))
    data.tree_local_index = qd.ndarray(qd.i32, shape=(capacity,))
    data.edge_parent = qd.ndarray(qd.i32, shape=(edge_capacity,))
    data.edge_child = qd.ndarray(qd.i32, shape=(edge_capacity,))
    data.edge_dof_index = qd.ndarray(qd.i32, shape=(edge_capacity,))
    data.root_dof_index = qd.ndarray(qd.i32, shape=(capacity,))
    data.body_twist = qd.ndarray(qd.f64, shape=(capacity, 6))
    data.body_wrench = qd.ndarray(qd.f64, shape=(capacity, 6))
    data.endpoint_qpos = qd.tensor(
        qd.f64,
        shape=rigid_data.rigid_info.qpos.shape,
    )
    data.endpoint_link_pos = qd.Vector.tensor(
        3,
        qd.f64,
        shape=rigid_data.dyn_state.links.pos.shape[:2],
    )
    data.endpoint_link_quat = qd.Vector.tensor(
        4,
        qd.f64,
        shape=rigid_data.dyn_state.links.quat.shape[:2],
    )
    data.endpoint_joint_xanchor = qd.Vector.tensor(
        3,
        qd.f64,
        shape=rigid_data.dyn_state.joints.xanchor.shape[:2],
    )
    data.endpoint_joint_xaxis = qd.Vector.tensor(
        3,
        qd.f64,
        shape=rigid_data.dyn_state.joints.xaxis.shape[:2],
    )
    data.endpoint_t = qd.Vector.tensor(
        3,
        qd.f64,
        shape=(max(n_mechanism_bodies, 1),),
    )
    data.endpoint_quat = qd.Vector.tensor(
        4,
        qd.f64,
        shape=(max(n_mechanism_bodies, 1),),
    )
    data.physical_p = qd.ndarray(qd.f64, shape=(max(total_dof, 1),))
    data.physical_Ap = qd.ndarray(qd.f64, shape=(max(total_dof, 1),))
    data.articulated_inertia = qd.ndarray(
        qd.f64,
        shape=(max(n_mechanism_bodies, 1), 6, 6),
    )
    data.body_inertia = qd.ndarray(
        qd.f64,
        shape=(max(n_mechanism_bodies, 1), 6, 6),
    )
    data.edge_u = qd.ndarray(qd.f64, shape=(edge_capacity, 6))
    data.edge_hessian = qd.ndarray(
        qd.f64,
        shape=(edge_capacity, 6, 6),
    )
    data.edge_basis = qd.ndarray(qd.f64, shape=(edge_capacity, 6))
    data.edge_arm = qd.ndarray(qd.f64, shape=(edge_capacity, 3))
    data.edge_d = qd.ndarray(qd.f64, shape=(edge_capacity,))
    data.root_inverse = qd.ndarray(
        qd.f64,
        shape=(max(n_mechanism_bodies, 1), 6, 6),
    )
    data.precond_force = qd.ndarray(
        qd.f64,
        shape=(max(n_mechanism_bodies, 1), 6),
    )
    data.precond_a = qd.ndarray(qd.f64, shape=(edge_capacity,))
    data.precond_velocity = qd.ndarray(
        qd.f64,
        shape=(max(n_mechanism_bodies, 1), 6),
    )
    data.kkt_proxy_diagonal = qd.ndarray(
        qd.f64,
        shape=(capacity, 6, 6),
    )
    data.n_bodies.from_numpy(np.array(n_rigid_bodies, dtype=np.int32))
    data.n_mechanism_bodies.from_numpy(np.array(n_mechanism_bodies, dtype=np.int32))
    data.n_edges.from_numpy(np.array(n_edges, dtype=np.int32))
    data.n_trees.from_numpy(np.array(n_trees, dtype=np.int32))
    data.max_tree_size.from_numpy(np.array(max_tree_size, dtype=np.int32))
    max_depth = int(depths.max(initial=0))
    data.max_depth.from_numpy(np.array(max_depth, dtype=np.int32))
    data.n_levels.from_numpy(np.array(max_depth + 1, dtype=np.int32))
    data.total_dof.from_numpy(np.array(total_dof, dtype=np.int32))
    data.proxy_dof_offset.from_numpy(np.array(proxy_dof_offset, dtype=np.int32))
    data.parent_body.from_numpy(np.pad(parents, (0, capacity - len(parents)), constant_values=-1))
    data.parent_edge.from_numpy(
        np.pad(
            parent_edge,
            (0, capacity - len(parent_edge)),
            constant_values=-1,
        )
    )
    data.depth.from_numpy(np.pad(depths, (0, capacity - len(depths))))
    data.tree_id.from_numpy(tree_ids)
    data.depth_start.from_numpy(depth_starts)
    data.depth_order.from_numpy(
        np.pad(
            depth_order,
            (0, mechanism_capacity - len(depth_order)),
        )
    )
    data.tree_roots.from_numpy(
        np.pad(
            tree_roots,
            (0, tree_capacity - len(tree_roots)),
        )
    )
    data.tree_body_start.from_numpy(
        np.pad(
            np.asarray(tree_body_starts, dtype=np.int32),
            (0, tree_capacity + 1 - len(tree_body_starts)),
            mode="edge",
        )
    )
    data.tree_body_list.from_numpy(
        np.pad(
            np.asarray(tree_body_order, dtype=np.int32),
            (0, mechanism_capacity - len(tree_body_order)),
        )
    )
    data.tree_local_index.from_numpy(tree_local_index)
    data.edge_parent.from_numpy(edge_parent)
    data.edge_child.from_numpy(edge_child)
    data.edge_dof_index.from_numpy(edge_dof_index)
    data.root_dof_index.from_numpy(
        np.pad(
            root_dof_index,
            (0, capacity - len(root_dof_index)),
            constant_values=-1,
        )
    )
    data.body_twist.from_numpy(np.zeros((capacity, 6), dtype=np.float64))
    data.body_wrench.from_numpy(np.zeros((capacity, 6), dtype=np.float64))
    data.endpoint_qpos.from_numpy(np.zeros(rigid_data.rigid_info.qpos.shape, dtype=np.float64))
    data.endpoint_link_pos.from_numpy(
        np.zeros(
            tuple(rigid_data.dyn_state.links.pos.shape) + (3,),
            dtype=np.float64,
        )
    )
    endpoint_link_quat = np.zeros(
        tuple(rigid_data.dyn_state.links.quat.shape) + (4,),
        dtype=np.float64,
    )
    endpoint_link_quat[..., 0] = 1.0
    data.endpoint_link_quat.from_numpy(endpoint_link_quat)
    data.endpoint_joint_xanchor.from_numpy(
        np.zeros(
            tuple(rigid_data.dyn_state.joints.xanchor.shape) + (3,),
            dtype=np.float64,
        )
    )
    data.endpoint_joint_xaxis.from_numpy(
        np.zeros(
            tuple(rigid_data.dyn_state.joints.xaxis.shape) + (3,),
            dtype=np.float64,
        )
    )
    data.endpoint_t.from_numpy(np.zeros((max(n_mechanism_bodies, 1), 3), dtype=np.float64))
    endpoint_quat = np.zeros((max(n_mechanism_bodies, 1), 4), dtype=np.float64)
    endpoint_quat[:, 0] = 1.0
    data.endpoint_quat.from_numpy(endpoint_quat)
    data.physical_p.from_numpy(np.zeros(max(total_dof, 1), dtype=np.float64))
    data.physical_Ap.from_numpy(np.zeros(max(total_dof, 1), dtype=np.float64))
    data.articulated_inertia.from_numpy(np.zeros((max(n_mechanism_bodies, 1), 6, 6), dtype=np.float64))
    data.body_inertia.from_numpy(np.zeros((max(n_mechanism_bodies, 1), 6, 6), dtype=np.float64))
    data.edge_u.from_numpy(np.zeros((edge_capacity, 6), dtype=np.float64))
    data.edge_hessian.from_numpy(np.zeros((edge_capacity, 6, 6), dtype=np.float64))
    data.edge_basis.from_numpy(np.zeros((edge_capacity, 6), dtype=np.float64))
    data.edge_arm.from_numpy(np.zeros((edge_capacity, 3), dtype=np.float64))
    data.edge_d.from_numpy(np.zeros(edge_capacity, dtype=np.float64))
    data.root_inverse.from_numpy(np.zeros((max(n_mechanism_bodies, 1), 6, 6), dtype=np.float64))
    data.precond_force.from_numpy(np.zeros((max(n_mechanism_bodies, 1), 6), dtype=np.float64))
    data.precond_a.from_numpy(np.zeros(edge_capacity, dtype=np.float64))
    data.precond_velocity.from_numpy(np.zeros((max(n_mechanism_bodies, 1), 6), dtype=np.float64))
    data.kkt_proxy_diagonal.from_numpy(np.zeros((capacity, 6, 6), dtype=np.float64))


@qd.func
def _body_point_twist(
    data: qd.template(),  # RigidJointForestSystem.Data
    rigid: qd.template(),  # RigidSystem.Data
    contact_proxy: qd.template(),  # RigidContactProxySystem.Data
    link,
    environment,
    joint,
    dof,
):
    joint_index = [joint, environment] if qd.static(rigid.rigid_config.batch_joints_info) else joint
    dof_index = [dof, environment] if qd.static(rigid.rigid_config.batch_dofs_info) else dof
    joint_type = rigid.dyn_info.joints.type[joint_index]
    dof_start = rigid.dyn_info.joints.dof_start[joint_index]
    angular = qd.Vector.zero(qd.f64, 3)
    linear = qd.Vector.zero(qd.f64, 3)
    if joint_type == gs.JOINT_TYPE.REVOLUTE:
        angular = data.endpoint_joint_xaxis[joint, environment]
        linear = angular.cross(
            data.endpoint_t[environment * rigid.n_links[()] + link] - data.endpoint_joint_xanchor[joint, environment]
        )
    elif joint_type == gs.JOINT_TYPE.PRISMATIC:
        linear = data.endpoint_joint_xaxis[joint, environment]
    elif joint_type == gs.JOINT_TYPE.FREE:
        local_dof = dof - dof_start
        if local_dof < 3:
            linear[local_dof] = 1.0
        else:
            angular = rigid.dyn_info.dofs.motion_ang[dof_index]
            linear = angular.cross(
                data.endpoint_t[environment * rigid.n_links[()] + link]
                - data.endpoint_joint_xanchor[joint, environment]
            )
    result = qd.Vector.zero(qd.f64, 6)
    for axis in qd.static(range(3)):
        result[axis] = linear[axis]
        result[axis + 3] = angular[axis]
    return result


@qd.func(requires_top_level=True)
def compute_endpoint_fk(
    data: qd.template(),  # RigidJointForestSystem.Data
    rigid: qd.template(),  # RigidSystem.Data
    contact_proxy: qd.template(),  # RigidContactProxySystem.Data
):
    h = rigid.h[()]
    for q, environment in qd.ndrange(
        data.endpoint_qpos.shape[0],
        rigid.n_instances[()],
    ):
        data.endpoint_qpos[q, environment] = rigid.rigid_info.qpos[
            q,
            environment,
        ]

    for link, environment in qd.ndrange(
        rigid.n_links[()],
        rigid.n_instances[()],
    ):
        link_index = [link, environment] if qd.static(rigid.rigid_config.batch_links_info) else link
        for joint in range(
            rigid.dyn_info.links.joint_start[link_index],
            rigid.dyn_info.links.joint_end[link_index],
        ):
            joint_index = [joint, environment] if qd.static(rigid.rigid_config.batch_joints_info) else joint
            joint_type = rigid.dyn_info.joints.type[joint_index]
            q_start = rigid.dyn_info.joints.q_start[joint_index]
            q_end = rigid.dyn_info.joints.q_end[joint_index]
            dof_start = rigid.dyn_info.joints.dof_start[joint_index]
            if joint_type == gs.JOINT_TYPE.FREE:
                for axis in qd.static(range(3)):
                    velocity = (
                        rigid.dyn_state.dofs.vel[dof_start + axis, environment]
                        + h
                        * rigid.constraint_state.qacc[
                            dof_start + axis,
                            environment,
                        ]
                    )
                    data.endpoint_qpos[q_start + axis, environment] = (
                        rigid.rigid_info.qpos[q_start + axis, environment] + h * velocity
                    )
                angular = qd.Vector.zero(qd.f64, 3)
                for axis in qd.static(range(3)):
                    angular[axis] = h * (
                        rigid.dyn_state.dofs.vel[
                            dof_start + axis + 3,
                            environment,
                        ]
                        + h
                        * rigid.constraint_state.qacc[
                            dof_start + axis + 3,
                            environment,
                        ]
                    )
                delta = gu.qd_rotvec_to_quat(
                    angular,
                    rigid.rigid_info.EPS[None],
                )
                current = qd.Vector(
                    [
                        rigid.rigid_info.qpos[q_start + 3, environment],
                        rigid.rigid_info.qpos[q_start + 4, environment],
                        rigid.rigid_info.qpos[q_start + 5, environment],
                        rigid.rigid_info.qpos[q_start + 6, environment],
                    ]
                )
                endpoint = gu.qd_transform_quat_by_quat(delta, current)
                for axis in qd.static(range(4)):
                    data.endpoint_qpos[q_start + 3 + axis, environment] = endpoint[axis]
            elif joint_type == gs.JOINT_TYPE.SPHERICAL:
                angular = qd.Vector.zero(qd.f64, 3)
                for axis in qd.static(range(3)):
                    angular[axis] = h * (
                        rigid.dyn_state.dofs.vel[
                            dof_start + axis,
                            environment,
                        ]
                        + h
                        * rigid.constraint_state.qacc[
                            dof_start + axis,
                            environment,
                        ]
                    )
                delta = gu.qd_rotvec_to_quat(
                    angular,
                    rigid.rigid_info.EPS[None],
                )
                current = qd.Vector(
                    [
                        rigid.rigid_info.qpos[q_start, environment],
                        rigid.rigid_info.qpos[q_start + 1, environment],
                        rigid.rigid_info.qpos[q_start + 2, environment],
                        rigid.rigid_info.qpos[q_start + 3, environment],
                    ]
                )
                endpoint = gu.qd_transform_quat_by_quat(delta, current)
                for axis in qd.static(range(4)):
                    data.endpoint_qpos[q_start + axis, environment] = endpoint[axis]
            elif joint_type != gs.JOINT_TYPE.FIXED:
                for local_q in range(q_end - q_start):
                    data.endpoint_qpos[q_start + local_q, environment] = rigid.rigid_info.qpos[
                        q_start + local_q, environment
                    ] + h * (
                        rigid.dyn_state.dofs.vel[
                            dof_start + local_q,
                            environment,
                        ]
                        + h
                        * rigid.constraint_state.qacc[
                            dof_start + local_q,
                            environment,
                        ]
                    )

    for entity, environment in qd.ndrange(
        rigid.dyn_info.entities.link_start.shape[0],
        rigid.n_instances[()],
    ):
        func_forward_kinematics_scratch(
            environment,
            environment,
            environment,
            entity,
            0,
            0,
            0,
            data.endpoint_qpos,
            data.endpoint_link_pos,
            data.endpoint_link_quat,
            data.endpoint_joint_xanchor,
            data.endpoint_joint_xaxis,
            rigid.dyn_state,
            rigid.dyn_info,
            rigid.rigid_info,
            rigid.rigid_config,
        )

    for body in range(data.n_mechanism_bodies[()]):
        environment = body // rigid.n_links[()]
        link = body - environment * rigid.n_links[()]
        link_index = [link, environment] if qd.static(rigid.rigid_config.batch_links_info) else link
        link_position = data.endpoint_link_pos[link, environment]
        link_quaternion = data.endpoint_link_quat[link, environment]
        inertial_position = rigid.dyn_info.links.inertial_pos[link_index]
        inertial_quaternion = rigid.dyn_info.links.inertial_quat[link_index]
        endpoint_position = link_position + gu.qd_transform_by_quat(
            inertial_position,
            link_quaternion,
        )
        endpoint_quaternion = gu.qd_transform_quat_by_quat(
            inertial_quaternion,
            link_quaternion,
        )
        for axis in qd.static(range(3)):
            data.endpoint_t[body][axis] = endpoint_position[axis]
        for axis in qd.static(range(4)):
            data.endpoint_quat[body][axis] = endpoint_quaternion[axis]

    for edge in range(data.n_edges[()]):
        child = data.edge_child[edge]
        parent = data.edge_parent[edge]
        environment = child // rigid.n_links[()]
        link = child - environment * rigid.n_links[()]
        dof = data.edge_dof_index[edge] - environment * rigid.n_dofs_per_instance[()]
        link_index = [link, environment] if qd.static(rigid.rigid_config.batch_links_info) else link
        basis = qd.Vector.zero(qd.f64, 6)
        for joint in range(
            rigid.dyn_info.links.joint_start[link_index],
            rigid.dyn_info.links.joint_end[link_index],
        ):
            joint_index = [joint, environment] if qd.static(rigid.rigid_config.batch_joints_info) else joint
            if dof >= rigid.dyn_info.joints.dof_start[joint_index] and dof < rigid.dyn_info.joints.dof_end[joint_index]:
                basis = _body_point_twist(
                    data,
                    rigid,
                    contact_proxy,
                    link,
                    environment,
                    joint,
                    dof,
                )
        arm = data.endpoint_t[child] - data.endpoint_t[parent]
        for component in qd.static(range(6)):
            data.edge_basis[edge, component] = basis[component]
        for axis in qd.static(range(3)):
            data.edge_arm[edge, axis] = arm[axis]


@qd.func
def forest_expand_body_p_cached(
    data: qd.template(),  # RigidJointForestSystem.Data
    rigid: qd.template(),  # RigidSystem.Data
    contact_proxy: qd.template(),  # RigidContactProxySystem.Data
    reduced: qd.template(),  # qd.Ndarray
    body,
):
    environment = body // rigid.n_links[()]
    link = body - environment * rigid.n_links[()]
    parent = data.parent_body[body]
    twist = qd.Vector.zero(qd.f64, 6)
    if parent >= 0:
        edge = data.parent_edge[body]
        parent_linear = qd.Vector.zero(qd.f64, 3)
        parent_angular = qd.Vector.zero(qd.f64, 3)
        for axis in qd.static(range(3)):
            parent_linear[axis] = data.body_twist[parent, axis]
            parent_angular[axis] = data.body_twist[parent, axis + 3]
        offset = data.endpoint_t[body] - data.endpoint_t[parent]
        if edge >= 0:
            offset = qd.Vector(
                [
                    data.edge_arm[edge, 0],
                    data.edge_arm[edge, 1],
                    data.edge_arm[edge, 2],
                ]
            )
        shifted_linear = parent_linear + parent_angular.cross(offset)
        for axis in qd.static(range(3)):
            twist[axis] = shifted_linear[axis]
            twist[axis + 3] = parent_angular[axis]

        if edge >= 0:
            reduced_index = rigid.dof_offset[()] + data.edge_dof_index[edge]
            coefficient = reduced[reduced_index]
            for component in qd.static(range(6)):
                twist[component] = twist[component] + coefficient * data.edge_basis[edge, component]
    else:
        link_index = [link, environment] if qd.static(rigid.rigid_config.batch_links_info) else link
        for joint in range(
            rigid.dyn_info.links.joint_start[link_index],
            rigid.dyn_info.links.joint_end[link_index],
        ):
            joint_index = [joint, environment] if qd.static(rigid.rigid_config.batch_joints_info) else joint
            for dof in range(
                rigid.dyn_info.joints.dof_start[joint_index],
                rigid.dyn_info.joints.dof_end[joint_index],
            ):
                reduced_index = rigid.dof_offset[()] + environment * rigid.n_dofs_per_instance[()] + dof
                basis = _body_point_twist(
                    data,
                    rigid,
                    contact_proxy,
                    link,
                    environment,
                    joint,
                    dof,
                )
                coefficient = reduced[reduced_index]
                twist = twist + coefficient * basis
    for component in qd.static(range(6)):
        data.body_twist[body, component] = twist[component]


@qd.func
def forest_expand_body_p(
    data: qd.template(),  # RigidJointForestSystem.Data
    rigid: qd.template(),  # RigidSystem.Data
    contact_proxy: qd.template(),  # RigidContactProxySystem.Data
    reduced: qd.template(),  # qd.Ndarray
    body,
):
    environment = body // rigid.n_links[()]
    link = body - environment * rigid.n_links[()]
    parent = data.parent_body[body]
    twist = qd.Vector.zero(qd.f64, 6)
    if parent >= 0:
        parent_linear = qd.Vector.zero(qd.f64, 3)
        parent_angular = qd.Vector.zero(qd.f64, 3)
        for axis in qd.static(range(3)):
            parent_linear[axis] = data.body_twist[parent, axis]
            parent_angular[axis] = data.body_twist[parent, axis + 3]
        offset = data.endpoint_t[body] - data.endpoint_t[parent]
        shifted_linear = parent_linear + parent_angular.cross(offset)
        for axis in qd.static(range(3)):
            twist[axis] = shifted_linear[axis]
            twist[axis + 3] = parent_angular[axis]

    link_index = [link, environment] if qd.static(rigid.rigid_config.batch_links_info) else link
    for joint in range(
        rigid.dyn_info.links.joint_start[link_index],
        rigid.dyn_info.links.joint_end[link_index],
    ):
        joint_index = [joint, environment] if qd.static(rigid.rigid_config.batch_joints_info) else joint
        for dof in range(
            rigid.dyn_info.joints.dof_start[joint_index],
            rigid.dyn_info.joints.dof_end[joint_index],
        ):
            reduced_index = rigid.dof_offset[()] + environment * rigid.n_dofs_per_instance[()] + dof
            basis = _body_point_twist(
                data,
                rigid,
                contact_proxy,
                link,
                environment,
                joint,
                dof,
            )
            coefficient = reduced[reduced_index]
            twist = twist + coefficient * basis
    for component in qd.static(range(6)):
        data.body_twist[body, component] = twist[component]


@qd.func(requires_top_level=True)
def genesis_legacy_expand_reduced_direction(
    data: qd.template(),  # RigidJointForestSystem.Data
    rigid: qd.template(),  # RigidSystem.Data
    contact_proxy: qd.template(),  # RigidContactProxySystem.Data
    reduced: qd.template(),  # qd.Ndarray
    n_links: qd.template(),  # int
):
    for level in qd.static(range(n_links)):
        qd.loop_config(name="genesis_legacy_expand_level")
        for body in range(data.n_mechanism_bodies[()]):
            if level < data.n_levels[()] and data.depth[body] == level:
                forest_expand_body_p(data, rigid, contact_proxy, reduced, body)


@qd.func(requires_top_level=True)
def forest_expand_level_p(
    data: qd.template(),  # RigidJointForestSystem.Data
    rigid: qd.template(),  # RigidSystem.Data
    contact_proxy: qd.template(),  # RigidContactProxySystem.Data
    reduced: qd.template(),  # qd.Ndarray
    n_links: qd.template(),  # int
):
    for level in qd.static(range(n_links)):
        qd.loop_config(name="forest_expand_level_p")
        for order_index in range(
            data.depth_start[level],
            data.depth_start[level + 1],
        ):
            forest_expand_body_p_cached(
                data,
                rigid,
                contact_proxy,
                reduced,
                data.depth_order[order_index],
            )


@qd.func(requires_top_level=True)
def forest_expand_tree_p(
    data: qd.template(),  # RigidJointForestSystem.Data
    rigid: qd.template(),  # RigidSystem.Data
    contact_proxy: qd.template(),  # RigidContactProxySystem.Data
    reduced: qd.template(),  # qd.Ndarray
):
    qd.loop_config(name="forest_expand_tree_p", block_dim=128)
    for tree in range(8):
        if tree < data.n_trees[()]:
            begin = data.tree_body_start[tree]
            end = data.tree_body_start[tree + 1]
            for order_index in range(begin, end):
                forest_expand_body_p_cached(
                    data,
                    rigid,
                    contact_proxy,
                    reduced,
                    data.tree_body_list[order_index],
                )


@qd.func(requires_top_level=True)
def expand_reduced_direction(
    data: qd.template(),  # RigidJointForestSystem.Data
    rigid: qd.template(),  # RigidSystem.Data
    contact_proxy: qd.template(),  # RigidContactProxySystem.Data
    reduced: qd.template(),  # qd.Ndarray
    genesis_legacy_enabled: qd.template(),  # bool
    use_fused_tree_path: qd.template(),  # bool
    n_links: qd.template(),  # int
):
    for body in range(data.n_bodies[()]):
        for component in qd.static(range(6)):
            data.body_twist[body, component] = 0.0
    expand_reduced_direction_from_zero(
        data,
        rigid,
        contact_proxy,
        reduced,
        genesis_legacy_enabled,
        use_fused_tree_path,
        n_links,
    )


@qd.func(requires_top_level=True)
def expand_mechanism_direction(
    data: qd.template(),  # RigidJointForestSystem.Data
    rigid: qd.template(),  # RigidSystem.Data
    contact_proxy: qd.template(),  # RigidContactProxySystem.Data
    reduced: qd.template(),  # qd.Ndarray
    genesis_legacy_enabled: qd.template(),  # bool
    use_fused_tree_path: qd.template(),  # bool
    n_links: qd.template(),  # int
):
    if qd.static(genesis_legacy_enabled):
        genesis_legacy_expand_reduced_direction(data, rigid, contact_proxy, reduced, n_links)
    elif qd.static(use_fused_tree_path):
        forest_expand_tree_p(data, rigid, contact_proxy, reduced)
    else:
        forest_expand_level_p(data, rigid, contact_proxy, reduced, n_links)


@qd.func(requires_top_level=True)
def expand_reduced_direction_from_zero(
    data: qd.template(),  # RigidJointForestSystem.Data
    rigid: qd.template(),  # RigidSystem.Data
    contact_proxy: qd.template(),  # RigidContactProxySystem.Data
    reduced: qd.template(),  # qd.Ndarray
    genesis_legacy_enabled: qd.template(),  # bool
    use_fused_tree_path: qd.template(),  # bool
    n_links: qd.template(),  # int
):
    expand_mechanism_direction(
        data,
        rigid,
        contact_proxy,
        reduced,
        genesis_legacy_enabled,
        use_fused_tree_path,
        n_links,
    )
    for pair in range(contact_proxy.n_pairs[()]):
        mechanism = contact_proxy.mechanism_body[pair]
        proxy = contact_proxy.proxy_body[pair]
        tangent = qd.Matrix.zero(qd.f64, 6, 6)
        mechanism_twist = qd.Vector.zero(qd.f64, 6)
        for row in qd.static(range(6)):
            mechanism_twist[row] = data.body_twist[mechanism, row]
            for column in qd.static(range(6)):
                tangent[row, column] = contact_proxy.tangent_map[pair, row, column]
        proxy_twist = expand_proxy_twist(tangent, mechanism_twist)
        for component in qd.static(range(6)):
            data.body_twist[proxy, component] = proxy_twist[component]


@qd.func(requires_top_level=True)
def clear_body_wrench(
    data: qd.template(),  # RigidJointForestSystem.Data
    rigid: qd.template(),  # RigidSystem.Data
    contact_proxy: qd.template(),  # RigidContactProxySystem.Data
):
    for body in range(data.n_bodies[()]):
        for component in qd.static(range(6)):
            data.body_wrench[body, component] = 0.0


@qd.func(requires_top_level=True)
def restrict_proxy_wrenches(
    data: qd.template(),  # RigidJointForestSystem.Data
    rigid: qd.template(),  # RigidSystem.Data
    contact_proxy: qd.template(),  # RigidContactProxySystem.Data
):
    for pair in range(contact_proxy.n_pairs[()]):
        mechanism = contact_proxy.mechanism_body[pair]
        proxy = contact_proxy.proxy_body[pair]
        tangent = qd.Matrix.zero(qd.f64, 6, 6)
        proxy_wrench = qd.Vector.zero(qd.f64, 6)
        for row in qd.static(range(6)):
            proxy_wrench[row] = data.body_wrench[proxy, row]
            for column in qd.static(range(6)):
                tangent[row, column] = contact_proxy.tangent_map[pair, row, column]
        mapped = restrict_proxy_wrench(tangent, proxy_wrench)
        for component in qd.static(range(6)):
            qd.atomic_add(data.body_wrench[mechanism, component], mapped[component])


@qd.func
def forest_project_body_Ap_cached(
    data: qd.template(),  # RigidJointForestSystem.Data
    rigid: qd.template(),  # RigidSystem.Data
    contact_proxy: qd.template(),  # RigidContactProxySystem.Data
    reduced: qd.template(),  # qd.Ndarray
    body,
    use_atomics: qd.template(),  # bool
):
    environment = body // rigid.n_links[()]
    link = body - environment * rigid.n_links[()]
    wrench = qd.Vector.zero(qd.f64, 6)
    for component in qd.static(range(6)):
        wrench[component] = data.body_wrench[body, component]

    parent = data.parent_body[body]
    if parent >= 0:
        edge = data.parent_edge[body]
        if edge >= 0:
            contribution = qd.f64(0.0)
            for component in qd.static(range(6)):
                contribution = contribution + data.edge_basis[edge, component] * wrench[component]
            reduced_index = rigid.dof_offset[()] + data.edge_dof_index[edge]
            if qd.static(use_atomics):
                qd.atomic_add(reduced[reduced_index], contribution)
            else:
                reduced[reduced_index] = reduced[reduced_index] + contribution

        offset = data.endpoint_t[body] - data.endpoint_t[parent]
        if edge >= 0:
            offset = qd.Vector(
                [
                    data.edge_arm[edge, 0],
                    data.edge_arm[edge, 1],
                    data.edge_arm[edge, 2],
                ]
            )
        force = qd.Vector([wrench[0], wrench[1], wrench[2]])
        torque = qd.Vector([wrench[3], wrench[4], wrench[5]])
        parent_torque = torque + offset.cross(force)
        for axis in qd.static(range(3)):
            if qd.static(use_atomics):
                qd.atomic_add(data.body_wrench[parent, axis], force[axis])
                qd.atomic_add(
                    data.body_wrench[parent, axis + 3],
                    parent_torque[axis],
                )
            else:
                data.body_wrench[parent, axis] = data.body_wrench[parent, axis] + force[axis]
                data.body_wrench[parent, axis + 3] = data.body_wrench[parent, axis + 3] + parent_torque[axis]
    else:
        link_index = [link, environment] if qd.static(rigid.rigid_config.batch_links_info) else link
        for joint in range(
            rigid.dyn_info.links.joint_start[link_index],
            rigid.dyn_info.links.joint_end[link_index],
        ):
            joint_index = [joint, environment] if qd.static(rigid.rigid_config.batch_joints_info) else joint
            for dof in range(
                rigid.dyn_info.joints.dof_start[joint_index],
                rigid.dyn_info.joints.dof_end[joint_index],
            ):
                basis = _body_point_twist(
                    data,
                    rigid,
                    contact_proxy,
                    link,
                    environment,
                    joint,
                    dof,
                )
                reduced_index = rigid.dof_offset[()] + environment * rigid.n_dofs_per_instance[()] + dof
                contribution = basis.dot(wrench)
                if qd.static(use_atomics):
                    qd.atomic_add(reduced[reduced_index], contribution)
                else:
                    reduced[reduced_index] = reduced[reduced_index] + contribution


@qd.func
def forest_project_body_Ap(
    data: qd.template(),  # RigidJointForestSystem.Data
    rigid: qd.template(),  # RigidSystem.Data
    contact_proxy: qd.template(),  # RigidContactProxySystem.Data
    reduced: qd.template(),  # qd.Ndarray
    body,
    use_atomics: qd.template(),  # bool
):
    environment = body // rigid.n_links[()]
    link = body - environment * rigid.n_links[()]
    wrench = qd.Vector.zero(qd.f64, 6)
    for component in qd.static(range(6)):
        wrench[component] = data.body_wrench[body, component]

    link_index = [link, environment] if qd.static(rigid.rigid_config.batch_links_info) else link
    for joint in range(
        rigid.dyn_info.links.joint_start[link_index],
        rigid.dyn_info.links.joint_end[link_index],
    ):
        joint_index = [joint, environment] if qd.static(rigid.rigid_config.batch_joints_info) else joint
        for dof in range(
            rigid.dyn_info.joints.dof_start[joint_index],
            rigid.dyn_info.joints.dof_end[joint_index],
        ):
            basis = _body_point_twist(
                data,
                rigid,
                contact_proxy,
                link,
                environment,
                joint,
                dof,
            )
            reduced_index = rigid.dof_offset[()] + environment * rigid.n_dofs_per_instance[()] + dof
            contribution = basis.dot(wrench)
            if qd.static(use_atomics):
                qd.atomic_add(reduced[reduced_index], contribution)
            else:
                reduced[reduced_index] = reduced[reduced_index] + contribution

    parent = data.parent_body[body]
    if parent >= 0:
        offset = data.endpoint_t[body] - data.endpoint_t[parent]
        force = qd.Vector([wrench[0], wrench[1], wrench[2]])
        torque = qd.Vector([wrench[3], wrench[4], wrench[5]])
        parent_torque = torque + offset.cross(force)
        for axis in qd.static(range(3)):
            if qd.static(use_atomics):
                qd.atomic_add(data.body_wrench[parent, axis], force[axis])
                qd.atomic_add(
                    data.body_wrench[parent, axis + 3],
                    parent_torque[axis],
                )
            else:
                data.body_wrench[parent, axis] = data.body_wrench[parent, axis] + force[axis]
                data.body_wrench[parent, axis + 3] = data.body_wrench[parent, axis + 3] + parent_torque[axis]


@qd.func(requires_top_level=True)
def genesis_legacy_restrict_body_wrenches(
    data: qd.template(),  # RigidJointForestSystem.Data
    rigid: qd.template(),  # RigidSystem.Data
    contact_proxy: qd.template(),  # RigidContactProxySystem.Data
    reduced: qd.template(),  # qd.Ndarray
    n_links: qd.template(),  # int
):
    for reverse_level in qd.static(range(n_links)):
        level = data.max_depth[()] - reverse_level
        qd.loop_config(name="genesis_legacy_project_level")
        for body in range(data.n_mechanism_bodies[()]):
            if reverse_level < data.n_levels[()] and data.depth[body] == level:
                forest_project_body_Ap(data, rigid, contact_proxy, reduced, body, True)


@qd.func(requires_top_level=True)
def forest_project_level_Ap(
    data: qd.template(),  # RigidJointForestSystem.Data
    rigid: qd.template(),  # RigidSystem.Data
    contact_proxy: qd.template(),  # RigidContactProxySystem.Data
    reduced: qd.template(),  # qd.Ndarray
    n_links: qd.template(),  # int
):
    for reverse_level in qd.static(range(n_links)):
        level = n_links - reverse_level - 1
        qd.loop_config(name="forest_project_level_Ap")
        for order_index in range(
            data.depth_start[level],
            data.depth_start[level + 1],
        ):
            forest_project_body_Ap_cached(
                data,
                rigid,
                contact_proxy,
                reduced,
                data.depth_order[order_index],
                True,
            )


@qd.func(requires_top_level=True)
def forest_project_tree_Ap(
    data: qd.template(),  # RigidJointForestSystem.Data
    rigid: qd.template(),  # RigidSystem.Data
    contact_proxy: qd.template(),  # RigidContactProxySystem.Data
    reduced: qd.template(),  # qd.Ndarray
):
    qd.loop_config(name="forest_project_tree_Ap", block_dim=128)
    for tree in range(8):
        if tree < data.n_trees[()]:
            begin = data.tree_body_start[tree]
            end = data.tree_body_start[tree + 1]
            for reverse_index in range(end - begin):
                order_index = end - reverse_index - 1
                forest_project_body_Ap_cached(
                    data,
                    rigid,
                    contact_proxy,
                    reduced,
                    data.tree_body_list[order_index],
                    False,
                )


@qd.func(requires_top_level=True)
def restrict_body_wrenches(
    data: qd.template(),  # RigidJointForestSystem.Data
    rigid: qd.template(),  # RigidSystem.Data
    contact_proxy: qd.template(),  # RigidContactProxySystem.Data
    reduced: qd.template(),  # qd.Ndarray
    genesis_legacy_enabled: qd.template(),  # bool
    use_fused_tree_path: qd.template(),  # bool
    n_links: qd.template(),  # int
):
    if qd.static(genesis_legacy_enabled):
        genesis_legacy_restrict_body_wrenches(data, rigid, contact_proxy, reduced, n_links)
    elif qd.static(use_fused_tree_path):
        forest_project_tree_Ap(data, rigid, contact_proxy, reduced)
    else:
        forest_project_level_Ap(data, rigid, contact_proxy, reduced, n_links)


@qd.func(requires_top_level=True)
def prepare_particular(
    data: qd.template(),  # RigidJointForestSystem.Data
    rigid: qd.template(),  # RigidSystem.Data
    contact_proxy: qd.template(),  # RigidContactProxySystem.Data
):
    for dof in range(data.total_dof[()]):
        data.physical_p[dof] = 0.0
    for pair in range(contact_proxy.n_pairs[()]):
        for component in qd.static(range(6)):
            data.physical_p[data.proxy_dof_offset[()] + pair * 6 + component] = contact_proxy.particular[
                pair, component
            ]


@qd.func(requires_top_level=True)
def particular_spmv(
    data: qd.template(),  # RigidJointForestSystem.Data
    rigid: qd.template(),  # RigidSystem.Data
    contact_proxy: qd.template(),  # RigidContactProxySystem.Data
    linear_system_data: qd.template(),  # GlobalLinearSystem.Data
):
    for dof in range(data.total_dof[()]):
        data.physical_Ap[dof] = 0.0
    sym_bcoo_spmv_naive(linear_system_data.matrix, data.physical_p, data.physical_Ap)


@qd.func(requires_top_level=True)
def project_physical_rhs(
    data: qd.template(),  # RigidJointForestSystem.Data
    rigid: qd.template(),  # RigidSystem.Data
    contact_proxy: qd.template(),  # RigidContactProxySystem.Data
    linear_system_data: qd.template(),  # GlobalLinearSystem.Data
    genesis_legacy_enabled: qd.template(),  # bool
    use_fused_tree_path: qd.template(),  # bool
    n_links: qd.template(),  # int
):
    for dof in range(data.proxy_dof_offset[()]):
        linear_system_data.write_rhs(dof, linear_system_data.read_rhs(dof) + data.physical_Ap[dof])

    clear_body_wrench(data, rigid, contact_proxy)
    for pair in range(contact_proxy.n_pairs[()]):
        proxy = contact_proxy.proxy_body[pair]
        offset = data.proxy_dof_offset[()] + pair * 6
        proxy_wrench = qd.Vector.zero(qd.f64, 6)
        normal = qd.Matrix.zero(qd.f64, 6, 6)
        for row in qd.static(range(6)):
            proxy_wrench[row] = linear_system_data.read_rhs(offset + row) + data.physical_Ap[offset + row]
            data.body_wrench[proxy, row] = proxy_wrench[row]
            for column in qd.static(range(6)):
                normal[row, column] = contact_proxy.normal_map[pair, row, column]

        if contact_proxy.restoration_active[()] != 0:
            slack_rhs = restrict_slack_wrench(normal, proxy_wrench) + qd.Vector(
                [
                    contact_proxy.lambda_[pair, 0],
                    contact_proxy.lambda_[pair, 1],
                    contact_proxy.lambda_[pair, 2],
                    contact_proxy.lambda_[pair, 3],
                    contact_proxy.lambda_[pair, 4],
                    contact_proxy.lambda_[pair, 5],
                ]
            )
            for component in qd.static(range(6)):
                linear_system_data.write_rhs(offset + component, slack_rhs[component])
        else:
            for component in qd.static(range(6)):
                linear_system_data.write_rhs(offset + component, 0.0)
    restrict_proxy_wrenches(data, rigid, contact_proxy)
    restrict_body_wrenches(
        data,
        rigid,
        contact_proxy,
        linear_system_data.b_rhs,
        genesis_legacy_enabled,
        use_fused_tree_path,
        n_links,
    )


@qd.func(requires_top_level=True)
def pcg_apply_operator(
    system: qd.template(),  # RigidJointForestSystem
    rigid: qd.template(),  # RigidSystem.Data
    contact_proxy: qd.template(),  # RigidContactProxySystem.Data
    linear_system_data: qd.template(),  # GlobalLinearSystem.Data
    direction: qd.template(),  # qd.Ndarray
    output: qd.template(),  # qd.Ndarray
):
    data = system.data
    for dof in range(linear_system_data.total_dof[()]):
        data.physical_Ap[dof] = qd.f64(0.0)
    prepare_physical_direction(
        data,
        rigid,
        contact_proxy,
        direction,
        system.genesis_legacy_enabled,
        system.use_fused_tree_path,
        system.n_links,
    )
    sym_bcoo_spmv_naive(linear_system_data.matrix, data.physical_p, data.physical_Ap)
    finish_reduced_spmv(
        data,
        rigid,
        contact_proxy,
        direction,
        output,
        system.genesis_legacy_enabled,
        system.use_fused_tree_path,
        system.n_links,
    )


@qd.func(requires_top_level=True)
def prepare_physical_direction(
    data: qd.template(),  # RigidJointForestSystem.Data
    rigid: qd.template(),  # RigidSystem.Data
    contact_proxy: qd.template(),  # RigidContactProxySystem.Data
    reduced: qd.template(),  # qd.Ndarray
    genesis_legacy_enabled: qd.template(),  # bool
    use_fused_tree_path: qd.template(),  # bool
    n_links: qd.template(),  # int
):
    for dof in range(data.total_dof[()]):
        data.physical_p[dof] = reduced[dof]
        if dof < data.n_bodies[()] * 6:
            body = dof // 6
            component = dof - body * 6
            data.body_twist[body, component] = 0.0
    expand_mechanism_direction(
        data,
        rigid,
        contact_proxy,
        reduced,
        genesis_legacy_enabled,
        use_fused_tree_path,
        n_links,
    )
    for pair in range(contact_proxy.n_pairs[()]):
        mechanism = contact_proxy.mechanism_body[pair]
        proxy = contact_proxy.proxy_body[pair]
        tangent = qd.Matrix.zero(qd.f64, 6, 6)
        normal = qd.Matrix.zero(qd.f64, 6, 6)
        mechanism_twist = qd.Vector.zero(qd.f64, 6)
        slack = qd.Vector.zero(qd.f64, 6)
        offset = data.proxy_dof_offset[()] + pair * 6
        for row in qd.static(range(6)):
            mechanism_twist[row] = data.body_twist[mechanism, row]
            slack[row] = reduced[offset + row]
            for column in qd.static(range(6)):
                tangent[row, column] = contact_proxy.tangent_map[
                    pair,
                    row,
                    column,
                ]
                normal[row, column] = contact_proxy.normal_map[pair, row, column]
        value = expand_proxy_twist(tangent, mechanism_twist)
        for component in qd.static(range(6)):
            data.body_twist[proxy, component] = value[component]
        if contact_proxy.restoration_active[()] != 0:
            value = value + expand_slack_twist(normal, slack)
        for component in qd.static(range(6)):
            data.physical_p[offset + component] = value[component]


@qd.func(requires_top_level=True)
def finish_reduced_spmv(
    data: qd.template(),  # RigidJointForestSystem.Data
    rigid: qd.template(),  # RigidSystem.Data
    contact_proxy: qd.template(),  # RigidContactProxySystem.Data
    reduced: qd.template(),  # qd.Ndarray
    result: qd.template(),  # qd.Ndarray
    genesis_legacy_enabled: qd.template(),  # bool
    use_fused_tree_path: qd.template(),  # bool
    n_links: qd.template(),  # int
):
    for dof in range(data.proxy_dof_offset[()]):
        result[dof] = result[dof] + data.physical_Ap[dof]

    # RigidSystem contributes Genesis native curvature. This projection
    # adds only the physical BCOO/contact wrench to generalized rows.
    clear_body_wrench(data, rigid, contact_proxy)
    for pair in range(contact_proxy.n_pairs[()]):
        mechanism = contact_proxy.mechanism_body[pair]
        proxy = contact_proxy.proxy_body[pair]
        offset = data.proxy_dof_offset[()] + pair * 6
        proxy_wrench = qd.Vector.zero(qd.f64, 6)
        tangent = qd.Matrix.zero(qd.f64, 6, 6)
        normal = qd.Matrix.zero(qd.f64, 6, 6)
        metric = qd.Matrix.zero(qd.f64, 6, 6)
        slack = qd.Vector.zero(qd.f64, 6)
        for row in qd.static(range(6)):
            proxy_wrench[row] = data.physical_Ap[offset + row]
            data.body_wrench[proxy, row] = proxy_wrench[row]
            slack[row] = reduced[offset + row]
            for column in qd.static(range(6)):
                tangent[row, column] = contact_proxy.tangent_map[
                    pair,
                    row,
                    column,
                ]
                normal[row, column] = contact_proxy.normal_map[pair, row, column]
                metric[row, column] = contact_proxy.metric[pair, row, column]
        mapped = restrict_proxy_wrench(tangent, proxy_wrench)
        for component in qd.static(range(6)):
            qd.atomic_add(
                data.body_wrench[mechanism, component],
                mapped[component],
            )
        if contact_proxy.restoration_active[()] != 0:
            slack_result = restrict_slack_wrench(normal, proxy_wrench) + metric @ slack
            for component in qd.static(range(6)):
                result[offset + component] = result[offset + component] + slack_result[component]
        else:
            for component in qd.static(range(6)):
                result[offset + component] = result[offset + component] + reduced[offset + component]
    restrict_body_wrenches(
        data,
        rigid,
        contact_proxy,
        result,
        genesis_legacy_enabled,
        use_fused_tree_path,
        n_links,
    )


@qd.func(requires_top_level=True)
def forest_inertia_wrench(
    data: qd.template(),  # RigidJointForestSystem.Data
    rigid: qd.template(),  # RigidSystem.Data
    contact_proxy: qd.template(),  # RigidContactProxySystem.Data
):
    qd.loop_config(name="forest_inertia_wrench")
    for body in range(data.n_bodies[()]):
        for row in qd.static(range(6)):
            value = qd.f64(0.0)
            if body < data.n_mechanism_bodies[()]:
                for column in qd.static(range(6)):
                    value = value + data.body_inertia[body, row, column] * data.body_twist[body, column]
            data.body_wrench[body, row] = value


@qd.func(requires_top_level=True)
def forest_control_matvec(
    data: qd.template(),  # RigidJointForestSystem.Data
    rigid: qd.template(),  # RigidSystem.Data
    contact_proxy: qd.template(),  # RigidContactProxySystem.Data
    reduced: qd.template(),  # qd.Ndarray
    result: qd.template(),  # qd.Ndarray
):
    qd.loop_config(name="forest_control_matvec")
    for dof, environment in qd.ndrange(
        rigid.n_dofs_per_instance[()],
        rigid.n_instances[()],
    ):
        dof_index = [dof, environment] if qd.static(rigid.rigid_config.batch_dofs_info) else dof
        augmentation = rigid.dyn_info.dofs.armature[dof_index] + rigid.h[()] * rigid.dyn_info.dofs.damping[dof_index]
        if rigid.dyn_state.dofs.ctrl_mode[dof, environment] <= gs.CTRL_MODE.VELOCITY:
            augmentation = augmentation - rigid.dyn_info.dofs.act_bias[dof_index][2] * rigid.h[()]
        reduced_index = rigid.dof_offset[()] + environment * rigid.n_dofs_per_instance[()] + dof
        result[reduced_index] = result[reduced_index] + augmentation * reduced[reduced_index]


@qd.func(requires_top_level=True)
def expand_solution(
    system: qd.template(),  # RigidJointForestSystem
    rigid: qd.template(),  # RigidSystem.Data
    contact_proxy: qd.template(),  # RigidContactProxySystem.Data
    solution: qd.template(),  # qd.Ndarray
):
    data = system.data
    expand_reduced_direction(
        data,
        rigid,
        contact_proxy,
        solution,
        system.genesis_legacy_enabled,
        system.use_fused_tree_path,
        system.n_links,
    )
    for pair in range(contact_proxy.n_pairs[()]):
        proxy = contact_proxy.proxy_body[pair]
        offset = data.proxy_dof_offset[()] + pair * 6
        normal = qd.Matrix.zero(qd.f64, 6, 6)
        slack = qd.Vector.zero(qd.f64, 6)
        for row in qd.static(range(6)):
            slack[row] = solution[offset + row]
            for column in qd.static(range(6)):
                normal[row, column] = contact_proxy.normal_map[pair, row, column]
        value = qd.Vector.zero(qd.f64, 6)
        for component in qd.static(range(6)):
            value[component] = contact_proxy.particular[pair, component] - data.body_twist[proxy, component]
        if contact_proxy.restoration_active[()] != 0:
            value = value - expand_slack_twist(normal, slack)
            for component in qd.static(range(6)):
                contact_proxy.slack[pair, component] = -slack[component]
            qd.atomic_max(
                contact_proxy.restoration_max_slack[()],
                slack.norm(),
            )
        else:
            for component in qd.static(range(6)):
                contact_proxy.slack[pair, component] = 0.0
        for component in qd.static(range(6)):
            contact_proxy.dq[pair, component] = value[component]


@qd.func
def curvature_bound(
    data: qd.template(),  # RigidJointForestSystem.Data
    rigid: qd.template(),  # RigidSystem.Data
    contact_proxy: qd.template(),  # RigidContactProxySystem.Data
    body,
    radius,
):
    reach = radius
    speed = qd.f64(0.0)
    angular_speed = qd.f64(0.0)
    node = body
    while node >= 0 and data.parent_body[node] >= 0:
        parent = data.parent_body[node]
        environment = node // rigid.n_links[()]
        link = node - environment * rigid.n_links[()]
        link_index = [link, environment] if qd.static(rigid.rigid_config.batch_links_info) else link
        child_anchor_distance = qd.f64(0.0)
        parent_anchor_distance = (data.endpoint_t[node] - data.endpoint_t[parent]).norm()
        has_joint = False
        has_rotation = False
        has_prismatic = False
        for joint in range(
            rigid.dyn_info.links.joint_start[link_index],
            rigid.dyn_info.links.joint_end[link_index],
        ):
            joint_index = [joint, environment] if qd.static(rigid.rigid_config.batch_joints_info) else joint
            joint_type = rigid.dyn_info.joints.type[joint_index]
            has_joint = True
            has_rotation = has_rotation or (
                joint_type == gs.JOINT_TYPE.REVOLUTE or joint_type == gs.JOINT_TYPE.SPHERICAL
            )
            has_prismatic = has_prismatic or (joint_type == gs.JOINT_TYPE.PRISMATIC)
            anchor = data.endpoint_joint_xanchor[joint, environment]
            child_anchor_distance = qd.max(
                child_anchor_distance,
                (data.endpoint_t[node] - anchor).norm(),
            )
            parent_anchor_distance = qd.max(
                parent_anchor_distance,
                (data.endpoint_t[parent] - anchor).norm(),
            )

        parent_linear = qd.Vector.zero(qd.f64, 3)
        parent_angular = qd.Vector.zero(qd.f64, 3)
        child_linear = qd.Vector.zero(qd.f64, 3)
        child_angular = qd.Vector.zero(qd.f64, 3)
        for axis in qd.static(range(3)):
            parent_linear[axis] = data.body_twist[parent, axis]
            parent_angular[axis] = data.body_twist[parent, axis + 3]
            child_linear[axis] = data.body_twist[node, axis]
            child_angular[axis] = data.body_twist[node, axis + 3]
        arm = data.endpoint_t[node] - data.endpoint_t[parent]
        relative_linear = child_linear - parent_linear - parent_angular.cross(arm)
        relative_angular = child_angular - parent_angular
        rotation_rate = relative_angular.norm()
        prismatic_rate = relative_linear.norm()
        reach_from_joint = reach + child_anchor_distance
        if has_rotation:
            speed = speed + rotation_rate * reach_from_joint
            angular_speed = angular_speed + rotation_rate
        if has_prismatic:
            speed = speed + prismatic_rate
        if not has_joint:
            parent_anchor_distance = arm.norm()
        travel = qd.f64(0.0)
        if has_prismatic:
            travel = prismatic_rate
        reach = parent_anchor_distance + travel + reach_from_joint
        node = parent

    root_angular = qd.Vector.zero(qd.f64, 3)
    for axis in qd.static(range(3)):
        root_angular[axis] = data.body_twist[node, axis + 3]
    root_angular_speed = root_angular.norm()
    angular_speed = angular_speed + root_angular_speed
    nontranslational_speed = speed + root_angular_speed * reach
    return 2.0 * angular_speed * nontranslational_speed


@qd.func
def _motion_matrix(
    data: qd.template(),  # RigidJointForestSystem.Data
    rigid: qd.template(),  # RigidSystem.Data
    contact_proxy: qd.template(),  # RigidContactProxySystem.Data
    arm: qd.template(),  # qd.Vector
):
    motion = qd.Matrix.identity(qd.f64, 6)
    skew = rigid_contact_proxy_skew(arm)
    for row in qd.static(range(3)):
        for column in qd.static(range(3)):
            motion[row, column + 3] = -skew[row, column]
    return motion


@qd.func
def _root_basis(
    data: qd.template(),  # RigidJointForestSystem.Data
    rigid: qd.template(),  # RigidSystem.Data
    contact_proxy: qd.template(),  # RigidContactProxySystem.Data
    body,
):
    basis_matrix = qd.Matrix.zero(qd.f64, 6, 6)
    environment = body // rigid.n_links[()]
    link = body - environment * rigid.n_links[()]
    first_dof = data.root_dof_index[body] - environment * rigid.n_dofs_per_instance[()]
    link_index = [link, environment] if qd.static(rigid.rigid_config.batch_links_info) else link
    for column in qd.static(range(6)):
        dof = first_dof + column
        value = qd.Vector.zero(qd.f64, 6)
        for joint in range(
            rigid.dyn_info.links.joint_start[link_index],
            rigid.dyn_info.links.joint_end[link_index],
        ):
            joint_index = [joint, environment] if qd.static(rigid.rigid_config.batch_joints_info) else joint
            if dof >= rigid.dyn_info.joints.dof_start[joint_index] and dof < rigid.dyn_info.joints.dof_end[joint_index]:
                value = _body_point_twist(
                    data,
                    rigid,
                    contact_proxy,
                    link,
                    environment,
                    joint,
                    dof,
                )
        for row in qd.static(range(6)):
            basis_matrix[row, column] = value[row]
    return basis_matrix


@qd.func(requires_top_level=True)
def build_preconditioner(
    system: qd.template(),  # RigidJointForestSystem
    rigid: qd.template(),  # RigidSystem.Data
    contact_proxy: qd.template(),  # RigidContactProxySystem.Data
    linear_system_data: qd.template(),  # GlobalLinearSystem.Data
):
    data = system.data
    for body in range(data.n_mechanism_bodies[()]):
        environment = body // rigid.n_links[()]
        link = body - environment * rigid.n_links[()]
        link_index = [link, environment] if qd.static(rigid.rigid_config.batch_links_info) else link
        mass = rigid.dyn_info.links.inertial_mass[link_index]
        inertia = rigid.dyn_info.links.inertial_i[link_index]
        rotation = gu.qd_quat_to_R(
            data.endpoint_quat[body],
            qd.f64(1.0e-12),
        )
        world_inertia = rotation @ inertia @ rotation.transpose()
        for row in qd.static(range(6)):
            for column in qd.static(range(6)):
                value = qd.f64(0.0)
                if qd.static(row < 3 and column < 3 and row == column):
                    value = mass
                elif qd.static(row >= 3 and column >= 3):
                    value = world_inertia[row - 3, column - 3]
                data.articulated_inertia[body, row, column] = value
                data.body_inertia[body, row, column] = value
                data.root_inverse[body, row, column] = 0.0

    for body in range(data.n_bodies[()]):
        for row in qd.static(range(6)):
            for column in qd.static(range(6)):
                data.kkt_proxy_diagonal[body, row, column] = 0.0

    for edge in range(data.n_edges[()]):
        child = data.edge_child[edge]
        parent = data.edge_parent[edge]
        environment = child // rigid.n_links[()]
        link = child - environment * rigid.n_links[()]
        dof = data.edge_dof_index[edge] - environment * rigid.n_dofs_per_instance[()]
        link_index = [link, environment] if qd.static(rigid.rigid_config.batch_links_info) else link
        basis = qd.Vector.zero(qd.f64, 6)
        for joint in range(
            rigid.dyn_info.links.joint_start[link_index],
            rigid.dyn_info.links.joint_end[link_index],
        ):
            joint_index = [joint, environment] if qd.static(rigid.rigid_config.batch_joints_info) else joint
            if dof >= rigid.dyn_info.joints.dof_start[joint_index] and dof < rigid.dyn_info.joints.dof_end[joint_index]:
                basis = _body_point_twist(
                    data,
                    rigid,
                    contact_proxy,
                    link,
                    environment,
                    joint,
                    dof,
                )
        arm = data.endpoint_t[child] - data.endpoint_t[parent]
        for component in qd.static(range(6)):
            data.edge_basis[edge, component] = basis[component]
            data.edge_u[edge, component] = 0.0
        for axis in qd.static(range(3)):
            data.edge_arm[edge, axis] = arm[axis]
        data.edge_d[edge] = 0.0
        data.precond_a[edge] = 0.0
        for row in qd.static(range(6)):
            for column in qd.static(range(6)):
                data.edge_hessian[edge, row, column] = 0.0

    matrix = linear_system_data.matrix
    proxy_block_base = data.proxy_dof_offset[()] // 3
    proxy_block_end = proxy_block_base + contact_proxy.n_pairs[()] * 2
    for index in range(matrix.bcoo_nnz[()]):
        block_row, block_col, hessian = matrix.read_bcoo(index)
        if (
            block_row >= proxy_block_base
            and block_row < proxy_block_end
            and block_col >= proxy_block_base
            and block_col < proxy_block_end
        ):
            local_row = block_row - proxy_block_base
            local_col = block_col - proxy_block_base
            pair_row = local_row // 2
            pair_col = local_col // 2
            row_block = local_row - pair_row * 2
            col_block = local_col - pair_col * 2

            proxy_row = contact_proxy.proxy_body[pair_row]
            proxy_col = contact_proxy.proxy_body[pair_col]
            if proxy_row == proxy_col:
                for row in qd.static(range(3)):
                    for column in qd.static(range(3)):
                        qd.atomic_add(
                            data.kkt_proxy_diagonal[
                                proxy_row,
                                row_block * 3 + row,
                                col_block * 3 + column,
                            ],
                            hessian[row, column],
                        )
                        if block_row != block_col:
                            qd.atomic_add(
                                data.kkt_proxy_diagonal[
                                    proxy_row,
                                    col_block * 3 + column,
                                    row_block * 3 + row,
                                ],
                                hessian[row, column],
                            )

            tangent_row = qd.Matrix.zero(qd.f64, 3, 6)
            tangent_col = qd.Matrix.zero(qd.f64, 3, 6)
            for row in qd.static(range(3)):
                for column in qd.static(range(6)):
                    tangent_row[row, column] = contact_proxy.tangent_map[
                        pair_row,
                        row_block * 3 + row,
                        column,
                    ]
                    tangent_col[row, column] = contact_proxy.tangent_map[
                        pair_col,
                        col_block * 3 + row,
                        column,
                    ]
            mapped = tangent_row.transpose() @ hessian @ tangent_col
            owner_row = contact_proxy.mechanism_body[pair_row]
            owner_col = contact_proxy.mechanism_body[pair_col]
            if owner_row == owner_col:
                for row in qd.static(range(6)):
                    for column in qd.static(range(6)):
                        qd.atomic_add(
                            data.articulated_inertia[
                                owner_row,
                                row,
                                column,
                            ],
                            mapped[row, column],
                        )
                        if block_row != block_col:
                            qd.atomic_add(
                                data.articulated_inertia[
                                    owner_row,
                                    column,
                                    row,
                                ],
                                mapped[row, column],
                            )
            elif data.parent_body[owner_col] == owner_row:
                edge = data.parent_edge[owner_col]
                if edge >= 0:
                    for row in qd.static(range(6)):
                        for column in qd.static(range(6)):
                            qd.atomic_add(
                                data.edge_hessian[
                                    edge,
                                    row,
                                    column,
                                ],
                                mapped[row, column],
                            )
            elif data.parent_body[owner_row] == owner_col:
                edge = data.parent_edge[owner_row]
                if edge >= 0:
                    for row in qd.static(range(6)):
                        for column in qd.static(range(6)):
                            qd.atomic_add(
                                data.edge_hessian[
                                    edge,
                                    column,
                                    row,
                                ],
                                mapped[row, column],
                            )

    for reverse_level in qd.static(range(system.n_links)):
        level = data.max_depth[()] - reverse_level
        for body in range(data.n_mechanism_bodies[()]):
            if reverse_level < data.max_depth[()] and data.depth[body] == level:
                parent = data.parent_body[body]
                articulated = qd.Matrix.zero(qd.f64, 6, 6)
                for row in qd.static(range(6)):
                    for column in qd.static(range(6)):
                        articulated[row, column] = data.articulated_inertia[body, row, column]
                arm = data.endpoint_t[body] - data.endpoint_t[parent]
                motion = _motion_matrix(data, rigid, contact_proxy, arm)
                propagated = motion.transpose() @ articulated @ motion
                edge = data.parent_edge[body]
                if edge >= 0:
                    edge_hessian = qd.Matrix.zero(qd.f64, 6, 6)
                    basis = qd.Vector.zero(qd.f64, 6)
                    for row in qd.static(range(6)):
                        basis[row] = data.edge_basis[edge, row]
                        for column in qd.static(range(6)):
                            edge_hessian[row, column] = data.edge_hessian[edge, row, column]
                    u = (motion.transpose() @ articulated + edge_hessian) @ basis
                    environment = body // rigid.n_links[()]
                    dof = data.edge_dof_index[edge] - environment * rigid.n_dofs_per_instance[()]
                    dof_index = [dof, environment] if qd.static(rigid.rigid_config.batch_dofs_info) else dof
                    actuator_damping = qd.f64(0.0)
                    if (
                        rigid.dyn_state.dofs.ctrl_mode[
                            dof,
                            environment,
                        ]
                        <= gs.CTRL_MODE.VELOCITY
                    ):
                        actuator_damping = -rigid.dyn_info.dofs.act_bias[dof_index][2] * rigid.h[()]
                    pivot = qd.max(
                        basis.dot(articulated @ basis)
                        + rigid.dyn_info.dofs.armature[dof_index]
                        + rigid.h[()] * rigid.dyn_info.dofs.damping[dof_index]
                        + actuator_damping,
                        qd.f64(1.0e-12),
                    )
                    edge_motion = edge_hessian @ motion
                    propagated = propagated + edge_motion + edge_motion.transpose() - u.outer_product(u) / pivot
                    for component in qd.static(range(6)):
                        data.edge_u[edge, component] = u[component]
                    data.edge_d[edge] = pivot
                for row in qd.static(range(6)):
                    for column in qd.static(range(6)):
                        qd.atomic_add(
                            data.articulated_inertia[parent, row, column],
                            propagated[row, column],
                        )

    for body in range(data.n_mechanism_bodies[()]):
        if data.parent_body[body] < 0 and data.root_dof_index[body] >= 0:
            articulated = qd.Matrix.zero(qd.f64, 6, 6)
            for row in qd.static(range(6)):
                for column in qd.static(range(6)):
                    articulated[row, column] = data.articulated_inertia[
                        body,
                        row,
                        column,
                    ]
            basis = _root_basis(data, rigid, contact_proxy, body)
            root_hessian = basis.transpose() @ articulated @ basis
            environment = body // rigid.n_links[()]
            first_dof = data.root_dof_index[body] - environment * rigid.n_dofs_per_instance[()]
            for dof in qd.static(range(6)):
                dof_index = (
                    [first_dof + dof, environment] if qd.static(rigid.rigid_config.batch_dofs_info) else first_dof + dof
                )
                actuator_damping = qd.f64(0.0)
                if (
                    rigid.dyn_state.dofs.ctrl_mode[
                        first_dof + dof,
                        environment,
                    ]
                    <= gs.CTRL_MODE.VELOCITY
                ):
                    actuator_damping = -rigid.dyn_info.dofs.act_bias[dof_index][2] * rigid.h[()]
                root_hessian[dof, dof] = (
                    root_hessian[dof, dof]
                    + rigid.dyn_info.dofs.armature[dof_index]
                    + rigid.h[()] * rigid.dyn_info.dofs.damping[dof_index]
                    + actuator_damping
                )
            inverse = root_hessian.inverse()
            for row in qd.static(range(6)):
                for column in qd.static(range(6)):
                    data.root_inverse[body, row, column] = inverse[
                        row,
                        column,
                    ]

    for pair in range(contact_proxy.n_pairs[()]):
        if contact_proxy.restoration_active[()] != 0:
            proxy = contact_proxy.proxy_body[pair]
            diagonal = qd.Matrix.zero(qd.f64, 6, 6)
            normal = qd.Matrix.zero(qd.f64, 6, 6)
            metric = qd.Matrix.zero(qd.f64, 6, 6)
            for row in qd.static(range(6)):
                for column in qd.static(range(6)):
                    diagonal[row, column] = data.kkt_proxy_diagonal[
                        proxy,
                        row,
                        column,
                    ]
                    normal[row, column] = contact_proxy.normal_map[
                        pair,
                        row,
                        column,
                    ]
                    metric[row, column] = contact_proxy.metric[
                        pair,
                        row,
                        column,
                    ]
            inverse = (normal.transpose() @ diagonal @ normal + metric).inverse()
            for row in qd.static(range(6)):
                for column in qd.static(range(6)):
                    data.kkt_proxy_diagonal[
                        proxy,
                        row,
                        column,
                    ] = inverse[row, column]


@qd.func(requires_top_level=True)
def forest_precond_apply_tree_shared(
    data: qd.template(),  # RigidJointForestSystem.Data
    rigid: qd.template(),  # RigidSystem.Data
    contact_proxy: qd.template(),  # RigidContactProxySystem.Data
    residual: qd.template(),  # qd.Ndarray
    result: qd.template(),  # qd.Ndarray
):
    qd.loop_config(name="forest_precond_apply_tree_shared", block_dim=32)
    for task in range(data.n_trees[()] * 32):
        tree = task // 32
        lane = qd.simt.block.thread_idx()
        tree_force = qd.simt.block.SharedArray((64, 6), qd.f64)
        tree_velocity = qd.simt.block.SharedArray((64, 6), qd.f64)
        root_rhs = qd.simt.block.SharedArray((6,), qd.f64)
        root_solution = qd.simt.block.SharedArray((6,), qd.f64)
        scalar = qd.simt.block.SharedArray((1,), qd.f64)

        if tree < data.n_trees[()]:
            begin = data.tree_body_start[tree]
            size = data.tree_body_start[tree + 1] - begin
            flat = lane
            while flat < size * 6:
                local = flat // 6
                component = flat - local * 6
                tree_force[local, component] = 0.0
                tree_velocity[local, component] = 0.0
                flat = flat + 32
            if lane < 6:
                root_rhs[lane] = 0.0
                root_solution[lane] = 0.0
            qd.simt.block.sync()

            for reverse_index in range(size - 1):
                local = size - reverse_index - 1
                child = data.tree_body_list[begin + local]
                parent = data.parent_body[child]
                parent_local = data.tree_local_index[parent]
                edge = data.parent_edge[child]
                if lane == 0:
                    edge_rhs = qd.f64(0.0)
                    if edge >= 0:
                        dof = rigid.dof_offset[()] + data.edge_dof_index[edge]
                        edge_rhs = residual[dof]
                        for component in qd.static(range(6)):
                            edge_rhs = edge_rhs + data.edge_basis[edge, component] * tree_force[local, component]
                        data.precond_a[edge] = edge_rhs
                    scalar[0] = edge_rhs
                qd.simt.block.sync()

                if lane < 6:
                    value = tree_force[local, lane]
                    if lane >= 3:
                        arm = qd.Vector(
                            [
                                data.endpoint_t[child][0] - data.endpoint_t[parent][0],
                                data.endpoint_t[child][1] - data.endpoint_t[parent][1],
                                data.endpoint_t[child][2] - data.endpoint_t[parent][2],
                            ]
                        )
                        child_force = qd.Vector(
                            [
                                tree_force[local, 0],
                                tree_force[local, 1],
                                tree_force[local, 2],
                            ]
                        )
                        value = value + arm.cross(child_force)[lane - 3]
                    if edge >= 0:
                        value = value - data.edge_u[edge, lane] * (scalar[0] / data.edge_d[edge])
                    tree_force[parent_local, lane] = tree_force[parent_local, lane] + value
                qd.simt.block.sync()

            root = data.tree_body_list[begin]
            root_dof = data.root_dof_index[root]
            root_basis = qd.Matrix.zero(qd.f64, 6, 6)
            if lane < 6 and root_dof >= 0:
                root_basis = _root_basis(data, rigid, contact_proxy, root)
                projected_rhs = residual[rigid.dof_offset[()] + root_dof + lane]
                for component in qd.static(range(6)):
                    projected_rhs = projected_rhs + root_basis[component, lane] * tree_force[0, component]
                root_rhs[lane] = projected_rhs
            qd.simt.block.sync()

            if lane < 6 and root_dof >= 0:
                solved = qd.f64(0.0)
                for component in qd.static(range(6)):
                    solved = solved + data.root_inverse[root, lane, component] * root_rhs[component]
                root_solution[lane] = solved
                result[rigid.dof_offset[()] + root_dof + lane] = solved
            qd.simt.block.sync()

            if lane < 6:
                velocity = qd.f64(0.0)
                if root_dof >= 0:
                    for component in qd.static(range(6)):
                        velocity = velocity + root_basis[lane, component] * root_solution[component]
                tree_velocity[0, lane] = velocity
            qd.simt.block.sync()

            for local_offset in range(size - 1):
                local = local_offset + 1
                child = data.tree_body_list[begin + local]
                parent = data.parent_body[child]
                parent_local = data.tree_local_index[parent]
                edge = data.parent_edge[child]
                if lane < 6:
                    transported = tree_velocity[parent_local, lane]
                    if lane < 3:
                        parent_angular = qd.Vector(
                            [
                                tree_velocity[parent_local, 3],
                                tree_velocity[parent_local, 4],
                                tree_velocity[parent_local, 5],
                            ]
                        )
                        arm = qd.Vector(
                            [
                                data.endpoint_t[child][0] - data.endpoint_t[parent][0],
                                data.endpoint_t[child][1] - data.endpoint_t[parent][1],
                                data.endpoint_t[child][2] - data.endpoint_t[parent][2],
                            ]
                        )
                        transported = transported + parent_angular.cross(arm)[lane]
                    tree_velocity[local, lane] = transported
                qd.simt.block.sync()

                if lane == 0:
                    edge_velocity = qd.f64(0.0)
                    if edge >= 0:
                        projection = qd.f64(0.0)
                        for component in qd.static(range(6)):
                            projection = (
                                projection + data.edge_u[edge, component] * tree_velocity[parent_local, component]
                            )
                        edge_velocity = (data.precond_a[edge] - projection) / data.edge_d[edge]
                        result[rigid.dof_offset[()] + data.edge_dof_index[edge]] = edge_velocity
                    scalar[0] = edge_velocity
                qd.simt.block.sync()

                if lane < 6 and edge >= 0:
                    tree_velocity[local, lane] = tree_velocity[local, lane] + data.edge_basis[edge, lane] * scalar[0]
                qd.simt.block.sync()

            flat = lane
            while flat < size * 6:
                local = flat // 6
                component = flat - local * 6
                body = data.tree_body_list[begin + local]
                data.precond_force[body, component] = tree_force[
                    local,
                    component,
                ]
                data.precond_velocity[body, component] = tree_velocity[
                    local,
                    component,
                ]
                flat = flat + 32


@qd.func(requires_top_level=True)
def forest_precond_apply_level(
    data: qd.template(),  # RigidJointForestSystem.Data
    rigid: qd.template(),  # RigidSystem.Data
    contact_proxy: qd.template(),  # RigidContactProxySystem.Data
    residual: qd.template(),  # qd.Ndarray
    result: qd.template(),  # qd.Ndarray
    n_links: qd.template(),  # int
):
    for dof in range(rigid.n_storage_dofs[()]):
        result[rigid.dof_offset[()] + dof] = 0.0
    for body in range(data.n_mechanism_bodies[()]):
        for component in qd.static(range(6)):
            data.precond_force[body, component] = 0.0
            data.precond_velocity[body, component] = 0.0

    for reverse_level in qd.static(range(n_links)):
        level = data.max_depth[()] - reverse_level
        for body in range(data.n_mechanism_bodies[()]):
            if reverse_level < data.max_depth[()] and data.depth[body] == level:
                parent = data.parent_body[body]
                force = qd.Vector.zero(qd.f64, 6)
                for component in qd.static(range(6)):
                    force[component] = data.precond_force[body, component]
                arm = data.endpoint_t[body] - data.endpoint_t[parent]
                motion = _motion_matrix(data, rigid, contact_proxy, arm)
                propagated = motion.transpose() @ force
                edge = data.parent_edge[body]
                if edge >= 0:
                    basis = qd.Vector.zero(qd.f64, 6)
                    u = qd.Vector.zero(qd.f64, 6)
                    for component in qd.static(range(6)):
                        basis[component] = data.edge_basis[edge, component]
                        u[component] = data.edge_u[edge, component]
                    dof = rigid.dof_offset[()] + data.edge_dof_index[edge]
                    edge_rhs = residual[dof] + basis.dot(force)
                    data.precond_a[edge] = edge_rhs
                    propagated = propagated - u * (edge_rhs / data.edge_d[edge])
                for component in qd.static(range(6)):
                    qd.atomic_add(
                        data.precond_force[parent, component],
                        propagated[component],
                    )

    for body in range(data.n_mechanism_bodies[()]):
        root_dof = data.root_dof_index[body]
        if data.parent_body[body] < 0 and root_dof >= 0:
            basis = _root_basis(data, rigid, contact_proxy, body)
            inverse = qd.Matrix.zero(qd.f64, 6, 6)
            force = qd.Vector.zero(qd.f64, 6)
            rhs = qd.Vector.zero(qd.f64, 6)
            for row in qd.static(range(6)):
                force[row] = data.precond_force[body, row]
                rhs[row] = residual[rigid.dof_offset[()] + root_dof + row]
                for column in qd.static(range(6)):
                    inverse[row, column] = data.root_inverse[
                        body,
                        row,
                        column,
                    ]
            solved = inverse @ (rhs + basis.transpose() @ force)
            velocity = basis @ solved
            for component in qd.static(range(6)):
                result[rigid.dof_offset[()] + root_dof + component] = solved[component]
                data.precond_velocity[body, component] = velocity[component]

    for level_slot in qd.static(range(n_links)):
        level = level_slot + 1
        for body in range(data.n_mechanism_bodies[()]):
            if level_slot < data.max_depth[()] and data.depth[body] == level:
                parent = data.parent_body[body]
                parent_velocity = qd.Vector.zero(qd.f64, 6)
                for component in qd.static(range(6)):
                    parent_velocity[component] = data.precond_velocity[parent, component]
                arm = data.endpoint_t[body] - data.endpoint_t[parent]
                velocity = _motion_matrix(data, rigid, contact_proxy, arm) @ parent_velocity
                edge = data.parent_edge[body]
                if edge >= 0:
                    basis = qd.Vector.zero(qd.f64, 6)
                    u = qd.Vector.zero(qd.f64, 6)
                    for component in qd.static(range(6)):
                        basis[component] = data.edge_basis[edge, component]
                        u[component] = data.edge_u[edge, component]
                    edge_velocity = (data.precond_a[edge] - u.dot(parent_velocity)) / data.edge_d[edge]
                    velocity = velocity + basis * edge_velocity
                    result[rigid.dof_offset[()] + data.edge_dof_index[edge]] = edge_velocity
                for component in qd.static(range(6)):
                    data.precond_velocity[body, component] = velocity[component]

    for dof in range(
        rigid.n_dofs[()],
        rigid.n_storage_dofs[()],
    ):
        offset = rigid.dof_offset[()] + dof
        result[offset] = residual[offset]

    for pair in range(contact_proxy.n_pairs[()]):
        offset = data.proxy_dof_offset[()] + pair * 6
        if contact_proxy.restoration_active[()] == 0:
            for component in qd.static(range(6)):
                result[offset + component] = residual[offset + component]
        else:
            proxy = contact_proxy.proxy_body[pair]
            inverse = qd.Matrix.zero(qd.f64, 6, 6)
            rhs = qd.Vector.zero(qd.f64, 6)
            for row in qd.static(range(6)):
                rhs[row] = residual[offset + row]
                for column in qd.static(range(6)):
                    inverse[row, column] = data.kkt_proxy_diagonal[
                        proxy,
                        row,
                        column,
                    ]
            value = inverse @ rhs
            for component in qd.static(range(6)):
                result[offset + component] = value[component]


@qd.func(requires_top_level=True)
def pcg_apply_preconditioner(
    system: qd.template(),  # RigidJointForestSystem
    rigid: qd.template(),  # RigidSystem.Data
    contact_proxy: qd.template(),  # RigidContactProxySystem.Data
    residual: qd.template(),  # qd.Ndarray
    output: qd.template(),  # qd.Ndarray
):
    apply_preconditioner(system, rigid, contact_proxy, residual, output)


@qd.func(requires_top_level=True)
def apply_preconditioner(
    system: qd.template(),  # RigidJointForestSystem
    rigid: qd.template(),  # RigidSystem.Data
    contact_proxy: qd.template(),  # RigidContactProxySystem.Data
    residual: qd.template(),  # qd.Ndarray
    result: qd.template(),  # qd.Ndarray
):
    data = system.data
    if qd.static(system.use_fused_tree_path):
        forest_precond_apply_tree_shared(data, rigid, contact_proxy, residual, result)
        for dof in range(
            rigid.n_dofs[()],
            rigid.n_storage_dofs[()],
        ):
            offset = rigid.dof_offset[()] + dof
            result[offset] = residual[offset]

        for pair in range(contact_proxy.n_pairs[()]):
            offset = data.proxy_dof_offset[()] + pair * 6
            if contact_proxy.restoration_active[()] == 0:
                for component in qd.static(range(6)):
                    result[offset + component] = residual[offset + component]
            else:
                proxy = contact_proxy.proxy_body[pair]
                inverse = qd.Matrix.zero(qd.f64, 6, 6)
                rhs = qd.Vector.zero(qd.f64, 6)
                for row in qd.static(range(6)):
                    rhs[row] = residual[offset + row]
                    for column in qd.static(range(6)):
                        inverse[row, column] = data.kkt_proxy_diagonal[
                            proxy,
                            row,
                            column,
                        ]
                value = inverse @ rhs
                for component in qd.static(range(6)):
                    result[offset + component] = value[component]
    else:
        forest_precond_apply_level(data, rigid, contact_proxy, residual, result, system.n_links)


@qd.func(requires_top_level=True)
def compute_merit_directional_derivative(
    data: qd.template(),  # RigidJointForestSystem.Data
    rigid: qd.template(),  # RigidSystem.Data
    contact_proxy: qd.template(),  # RigidContactProxySystem.Data
    linear_system_data: qd.template(),  # GlobalLinearSystem.Data
):
    for _ in range(1):
        contact_proxy.merit_gtd[()] = 0.0
    for dof in range(data.proxy_dof_offset[()]):
        qd.atomic_add(
            contact_proxy.merit_gtd[()],
            -contact_proxy.merit_gradient[dof] * linear_system_data.read_solution(dof),
        )
    for pair in range(contact_proxy.n_pairs[()]):
        offset = data.proxy_dof_offset[()] + pair * 6
        contribution = qd.f64(0.0)
        for component in qd.static(range(6)):
            contribution = contribution + (
                contact_proxy.merit_gradient[offset + component] * contact_proxy.dq[pair, component]
            )
        qd.atomic_add(contact_proxy.merit_gtd[()], contribution)

from __future__ import annotations

import numpy as np
import quadrants as qd

import genesis as gs
from genesis.engine.solvers.rigid.abd.inverse_kinematics import (
    func_forward_kinematics_scratch,
)
from genesis.utils import geom as gu

from .rigid_contact_proxy import RigidContactProxySystem
from .rigid_contact_proxy_kkt import (
    expand_proxy_twist,
    expand_slack_twist,
    restrict_proxy_wrench,
    restrict_slack_wrench,
    rigid_contact_proxy_skew,
)
from .rigid_system import RigidSystem
from .sim_system import SimSystem

_FUSED_MAX_TREE_SIZE = 64
_FUSED_MAX_TREES = 8


@qd.data_oriented
class RigidJointForestSystem(SimSystem):
    """Genesis minimal-coordinate link/proxy prolongation and restriction."""

    def __init__(self, rigid_solver) -> None:
        super().__init__()
        self.is_initialized_host = False
        self.fused_enabled = True
        self.genesis_legacy_enabled = False
        self.use_fused_tree_path = False
        self.n_links_host = rigid_solver.n_links
        self.n_instances_host = rigid_solver._B
        parent_link = []
        depth = []
        edge_child = []
        edge_dof_index = []
        root_dof_index = []
        for link in rigid_solver.links:
            parent_link.append(link.parent_idx)
            depth.append(0 if link.parent_idx < 0 else depth[link.parent_idx] + 1)
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
                root_dof_index.append(root_dof)
            else:
                root_dof_index.append(-1)
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
                    edge_child.append(link.idx)
                    edge_dof_index.append(moving_joints[0].dof_start)
        self.parent_link_host = tuple(parent_link)
        self.depth_host = tuple(depth)
        self.edge_child_host = tuple(edge_child)
        self.edge_dof_index_host = tuple(edge_dof_index)
        self.root_dof_index_host = tuple(root_dof_index)
        self.n_edges_per_instance_host = len(edge_child)
        self.has_contact_proxy = False

    def configure(self, fused_enabled: bool) -> None:
        if self.is_initialized_host:
            raise RuntimeError("RigidJointForestSystem is already initialized")
        self.fused_enabled = bool(fused_enabled)

    def configure_genesis_legacy(self, enabled: bool) -> None:
        if self.is_initialized_host:
            raise RuntimeError("RigidJointForestSystem is already initialized")
        self.genesis_legacy_enabled = bool(enabled)

    @property
    def selected_path(self) -> str:
        if self.genesis_legacy_enabled:
            return "genesis_legacy"
        if self.use_fused_tree_path:
            return "cgq_tree"
        return "cgq_level"

    def build(self) -> None:
        self.rigid = self.require(RigidSystem)
        self.contact_proxy = self.find(RigidContactProxySystem)
        self.has_contact_proxy = self.contact_proxy is not None

    def init(
        self,
        total_dof: int,
        n_rigid_bodies: int,
        proxy_dof_offset: int,
    ) -> None:
        if self.is_initialized_host:
            raise RuntimeError("RigidJointForestSystem is already initialized")
        n_links = self.n_links_host
        n_instances = self.n_instances_host
        n_mechanism_bodies = n_links * n_instances
        if n_rigid_bodies < n_mechanism_bodies:
            raise ValueError("RigidJointForestSystem body count is smaller than Genesis link count")
        if total_dof < 0 or proxy_dof_offset < 0 or proxy_dof_offset > total_dof:
            raise ValueError("RigidJointForestSystem global DOF layout is invalid")
        if self.has_contact_proxy:
            required = proxy_dof_offset + self.contact_proxy.n_pairs_host * 6
            if required > total_dof:
                raise ValueError("RigidJointForestSystem proxy dummy rows exceed global DOF layout")

        parents = np.full(n_mechanism_bodies, -1, dtype=np.int32)
        depths = np.zeros(n_mechanism_bodies, dtype=np.int32)
        for environment in range(n_instances):
            body_offset = environment * n_links
            for link, parent_link in enumerate(self.parent_link_host):
                body = body_offset + link
                if parent_link >= 0:
                    parents[body] = body_offset + parent_link
                    depths[body] = self.depth_host[link]

        children = [[] for _ in range(n_mechanism_bodies)]
        for body, parent in enumerate(parents):
            if parent >= 0:
                children[int(parent)].append(body)
        tree_roots_host = np.flatnonzero(parents < 0).astype(np.int32)
        tree_ids = np.full(n_rigid_bodies, -1, dtype=np.int32)
        tree_body_start_host = [0]
        tree_body_list_host = []
        tree_local_index = np.full(n_rigid_bodies, -1, dtype=np.int32)
        max_tree_size = 0
        for tree, root in enumerate(tree_roots_host):
            stack = [int(root)]
            tree_bodies = []
            while stack:
                body = stack.pop()
                tree_bodies.append(body)
                stack.extend(reversed(children[body]))
            for local, body in enumerate(tree_bodies):
                tree_ids[body] = tree
                tree_local_index[body] = local
            tree_body_list_host.extend(tree_bodies)
            tree_body_start_host.append(len(tree_body_list_host))
            max_tree_size = max(max_tree_size, len(tree_bodies))

        depth_order_host = np.argsort(depths, kind="stable").astype(np.int32)
        depth_counts = np.bincount(depths, minlength=n_links)
        depth_start_host = np.zeros(n_links + 1, dtype=np.int32)
        np.cumsum(depth_counts, out=depth_start_host[1:])
        n_trees = len(tree_roots_host)
        self.use_fused_tree_path = (
            self.fused_enabled
            and not self.genesis_legacy_enabled
            and n_trees <= _FUSED_MAX_TREES
            and max_tree_size <= _FUSED_MAX_TREE_SIZE
        )

        n_edges = self.n_edges_per_instance_host * n_instances
        parent_edge = np.full(n_mechanism_bodies, -1, dtype=np.int32)
        edge_parent = np.zeros(max(n_edges, 1), dtype=np.int32)
        edge_child = np.zeros(max(n_edges, 1), dtype=np.int32)
        edge_dof_index = np.zeros(max(n_edges, 1), dtype=np.int32)
        root_dof_index = np.full(n_mechanism_bodies, -1, dtype=np.int32)
        edge = 0
        for environment in range(n_instances):
            body_offset = environment * n_links
            dof_offset = environment * self.rigid.n_dofs_per_instance_host
            for child, dof in zip(
                self.edge_child_host,
                self.edge_dof_index_host,
                strict=True,
            ):
                body = body_offset + child
                parent_edge[body] = edge
                edge_parent[edge] = parents[body]
                edge_child[edge] = body
                edge_dof_index[edge] = dof_offset + dof
                edge += 1
            for link, dof in enumerate(self.root_dof_index_host):
                if dof >= 0:
                    root_dof_index[body_offset + link] = dof_offset + dof

        capacity = max(n_rigid_bodies, 1)
        edge_capacity = max(n_edges, 1)
        tree_capacity = max(n_trees, 1)
        mechanism_capacity = max(n_mechanism_bodies, 1)
        self.n_bodies = qd.ndarray(qd.i32, shape=())
        self.n_mechanism_bodies = qd.ndarray(qd.i32, shape=())
        self.n_edges = qd.ndarray(qd.i32, shape=())
        self.n_trees = qd.ndarray(qd.i32, shape=())
        self.max_tree_size = qd.ndarray(qd.i32, shape=())
        self.max_depth = qd.ndarray(qd.i32, shape=())
        self.n_levels = qd.ndarray(qd.i32, shape=())
        self.total_dof = qd.ndarray(qd.i32, shape=())
        self.proxy_dof_offset = qd.ndarray(qd.i32, shape=())
        self.parent_body = qd.ndarray(qd.i32, shape=(capacity,))
        self.parent_edge = qd.ndarray(qd.i32, shape=(capacity,))
        self.depth = qd.ndarray(qd.i32, shape=(capacity,))
        self.tree_id = qd.ndarray(qd.i32, shape=(capacity,))
        self.depth_start = qd.ndarray(qd.i32, shape=(n_links + 1,))
        self.depth_order = qd.ndarray(qd.i32, shape=(mechanism_capacity,))
        self.tree_roots = qd.ndarray(qd.i32, shape=(tree_capacity,))
        self.tree_body_start = qd.ndarray(qd.i32, shape=(tree_capacity + 1,))
        self.tree_body_list = qd.ndarray(qd.i32, shape=(mechanism_capacity,))
        self.tree_local_index = qd.ndarray(qd.i32, shape=(capacity,))
        self.edge_parent = qd.ndarray(qd.i32, shape=(edge_capacity,))
        self.edge_child = qd.ndarray(qd.i32, shape=(edge_capacity,))
        self.edge_dof_index = qd.ndarray(qd.i32, shape=(edge_capacity,))
        self.root_dof_index = qd.ndarray(qd.i32, shape=(capacity,))
        self.body_twist = qd.ndarray(qd.f64, shape=(capacity, 6))
        self.body_wrench = qd.ndarray(qd.f64, shape=(capacity, 6))
        self.endpoint_qpos = qd.tensor(
            qd.f64,
            shape=self.rigid.rigid_info.qpos.shape,
        )
        self.endpoint_link_pos = qd.Vector.tensor(
            3,
            qd.f64,
            shape=self.rigid.dyn_state.links.pos.shape[:2],
        )
        self.endpoint_link_quat = qd.Vector.tensor(
            4,
            qd.f64,
            shape=self.rigid.dyn_state.links.quat.shape[:2],
        )
        self.endpoint_joint_xanchor = qd.Vector.tensor(
            3,
            qd.f64,
            shape=self.rigid.dyn_state.joints.xanchor.shape[:2],
        )
        self.endpoint_joint_xaxis = qd.Vector.tensor(
            3,
            qd.f64,
            shape=self.rigid.dyn_state.joints.xaxis.shape[:2],
        )
        self.endpoint_t = qd.Vector.tensor(
            3,
            qd.f64,
            shape=(max(n_mechanism_bodies, 1),),
        )
        self.endpoint_quat = qd.Vector.tensor(
            4,
            qd.f64,
            shape=(max(n_mechanism_bodies, 1),),
        )
        self.physical_p = qd.ndarray(qd.f64, shape=(max(total_dof, 1),))
        self.physical_Ap = qd.ndarray(qd.f64, shape=(max(total_dof, 1),))
        self.articulated_inertia = qd.ndarray(
            qd.f64,
            shape=(max(n_mechanism_bodies, 1), 6, 6),
        )
        self.body_inertia = qd.ndarray(
            qd.f64,
            shape=(max(n_mechanism_bodies, 1), 6, 6),
        )
        self.edge_u = qd.ndarray(qd.f64, shape=(edge_capacity, 6))
        self.edge_hessian = qd.ndarray(
            qd.f64,
            shape=(edge_capacity, 6, 6),
        )
        self.edge_basis = qd.ndarray(qd.f64, shape=(edge_capacity, 6))
        self.edge_arm = qd.ndarray(qd.f64, shape=(edge_capacity, 3))
        self.edge_d = qd.ndarray(qd.f64, shape=(edge_capacity,))
        self.root_inverse = qd.ndarray(
            qd.f64,
            shape=(max(n_mechanism_bodies, 1), 6, 6),
        )
        self.precond_force = qd.ndarray(
            qd.f64,
            shape=(max(n_mechanism_bodies, 1), 6),
        )
        self.precond_a = qd.ndarray(qd.f64, shape=(edge_capacity,))
        self.precond_velocity = qd.ndarray(
            qd.f64,
            shape=(max(n_mechanism_bodies, 1), 6),
        )
        self.kkt_proxy_diagonal = qd.ndarray(
            qd.f64,
            shape=(capacity, 6, 6),
        )
        self.n_bodies.from_numpy(np.array(n_rigid_bodies, dtype=np.int32))
        self.n_mechanism_bodies.from_numpy(np.array(n_mechanism_bodies, dtype=np.int32))
        self.n_edges.from_numpy(np.array(n_edges, dtype=np.int32))
        self.n_trees.from_numpy(np.array(n_trees, dtype=np.int32))
        self.max_tree_size.from_numpy(np.array(max_tree_size, dtype=np.int32))
        max_depth = int(depths.max(initial=0))
        self.max_depth.from_numpy(np.array(max_depth, dtype=np.int32))
        self.n_levels.from_numpy(np.array(max_depth + 1, dtype=np.int32))
        self.total_dof.from_numpy(np.array(total_dof, dtype=np.int32))
        self.proxy_dof_offset.from_numpy(np.array(proxy_dof_offset, dtype=np.int32))
        self.parent_body.from_numpy(np.pad(parents, (0, capacity - len(parents)), constant_values=-1))
        self.parent_edge.from_numpy(
            np.pad(
                parent_edge,
                (0, capacity - len(parent_edge)),
                constant_values=-1,
            )
        )
        self.depth.from_numpy(np.pad(depths, (0, capacity - len(depths))))
        self.tree_id.from_numpy(tree_ids)
        self.depth_start.from_numpy(depth_start_host)
        self.depth_order.from_numpy(
            np.pad(
                depth_order_host,
                (0, mechanism_capacity - len(depth_order_host)),
            )
        )
        self.tree_roots.from_numpy(
            np.pad(
                tree_roots_host,
                (0, tree_capacity - len(tree_roots_host)),
            )
        )
        self.tree_body_start.from_numpy(
            np.pad(
                np.asarray(tree_body_start_host, dtype=np.int32),
                (0, tree_capacity + 1 - len(tree_body_start_host)),
                mode="edge",
            )
        )
        self.tree_body_list.from_numpy(
            np.pad(
                np.asarray(tree_body_list_host, dtype=np.int32),
                (0, mechanism_capacity - len(tree_body_list_host)),
            )
        )
        self.tree_local_index.from_numpy(tree_local_index)
        self.edge_parent.from_numpy(edge_parent)
        self.edge_child.from_numpy(edge_child)
        self.edge_dof_index.from_numpy(edge_dof_index)
        self.root_dof_index.from_numpy(
            np.pad(
                root_dof_index,
                (0, capacity - len(root_dof_index)),
                constant_values=-1,
            )
        )
        self.body_twist.from_numpy(np.zeros((capacity, 6), dtype=np.float64))
        self.body_wrench.from_numpy(np.zeros((capacity, 6), dtype=np.float64))
        self.endpoint_qpos.from_numpy(np.zeros(self.rigid.rigid_info.qpos.shape, dtype=np.float64))
        self.endpoint_link_pos.from_numpy(
            np.zeros(
                tuple(self.rigid.dyn_state.links.pos.shape) + (3,),
                dtype=np.float64,
            )
        )
        endpoint_link_quat = np.zeros(
            tuple(self.rigid.dyn_state.links.quat.shape) + (4,),
            dtype=np.float64,
        )
        endpoint_link_quat[..., 0] = 1.0
        self.endpoint_link_quat.from_numpy(endpoint_link_quat)
        self.endpoint_joint_xanchor.from_numpy(
            np.zeros(
                tuple(self.rigid.dyn_state.joints.xanchor.shape) + (3,),
                dtype=np.float64,
            )
        )
        self.endpoint_joint_xaxis.from_numpy(
            np.zeros(
                tuple(self.rigid.dyn_state.joints.xaxis.shape) + (3,),
                dtype=np.float64,
            )
        )
        self.endpoint_t.from_numpy(np.zeros((max(n_mechanism_bodies, 1), 3), dtype=np.float64))
        endpoint_quat = np.zeros((max(n_mechanism_bodies, 1), 4), dtype=np.float64)
        endpoint_quat[:, 0] = 1.0
        self.endpoint_quat.from_numpy(endpoint_quat)
        self.physical_p.from_numpy(np.zeros(max(total_dof, 1), dtype=np.float64))
        self.physical_Ap.from_numpy(np.zeros(max(total_dof, 1), dtype=np.float64))
        self.articulated_inertia.from_numpy(np.zeros((max(n_mechanism_bodies, 1), 6, 6), dtype=np.float64))
        self.body_inertia.from_numpy(np.zeros((max(n_mechanism_bodies, 1), 6, 6), dtype=np.float64))
        self.edge_u.from_numpy(np.zeros((edge_capacity, 6), dtype=np.float64))
        self.edge_hessian.from_numpy(np.zeros((edge_capacity, 6, 6), dtype=np.float64))
        self.edge_basis.from_numpy(np.zeros((edge_capacity, 6), dtype=np.float64))
        self.edge_arm.from_numpy(np.zeros((edge_capacity, 3), dtype=np.float64))
        self.edge_d.from_numpy(np.zeros(edge_capacity, dtype=np.float64))
        self.root_inverse.from_numpy(np.zeros((max(n_mechanism_bodies, 1), 6, 6), dtype=np.float64))
        self.precond_force.from_numpy(np.zeros((max(n_mechanism_bodies, 1), 6), dtype=np.float64))
        self.precond_a.from_numpy(np.zeros(edge_capacity, dtype=np.float64))
        self.precond_velocity.from_numpy(np.zeros((max(n_mechanism_bodies, 1), 6), dtype=np.float64))
        self.kkt_proxy_diagonal.from_numpy(np.zeros((capacity, 6, 6), dtype=np.float64))

        self.n_entities_host = self.rigid.dyn_info.entities.link_start.shape[0]
        self.is_initialized_host = True

    @qd.func
    def _body_point_twist(self, link, environment, joint, dof):
        joint_index = [joint, environment] if qd.static(self.rigid.rigid_config.batch_joints_info) else joint
        dof_index = [dof, environment] if qd.static(self.rigid.rigid_config.batch_dofs_info) else dof
        joint_type = self.rigid.dyn_info.joints.type[joint_index]
        dof_start = self.rigid.dyn_info.joints.dof_start[joint_index]
        angular = qd.Vector.zero(qd.f64, 3)
        linear = qd.Vector.zero(qd.f64, 3)
        if joint_type == gs.JOINT_TYPE.REVOLUTE:
            angular = self.endpoint_joint_xaxis[joint, environment]
            linear = angular.cross(
                self.endpoint_t[environment * self.rigid.n_links[()] + link]
                - self.endpoint_joint_xanchor[joint, environment]
            )
        elif joint_type == gs.JOINT_TYPE.PRISMATIC:
            linear = self.endpoint_joint_xaxis[joint, environment]
        elif joint_type == gs.JOINT_TYPE.FREE:
            local_dof = dof - dof_start
            if local_dof < 3:
                linear[local_dof] = 1.0
            else:
                angular = self.rigid.dyn_info.dofs.motion_ang[dof_index]
                linear = angular.cross(
                    self.endpoint_t[environment * self.rigid.n_links[()] + link]
                    - self.endpoint_joint_xanchor[joint, environment]
                )
        result = qd.Vector.zero(qd.f64, 6)
        for axis in qd.static(range(3)):
            result[axis] = linear[axis]
            result[axis + 3] = angular[axis]
        return result

    @qd.func(requires_top_level=True)
    def compute_endpoint_fk(self):
        h = self.rigid.h
        for q, environment in qd.ndrange(
            self.endpoint_qpos.shape[0],
            self.n_instances_host,
        ):
            self.endpoint_qpos[q, environment] = self.rigid.rigid_info.qpos[
                q,
                environment,
            ]

        for link, environment in qd.ndrange(
            self.rigid.n_links[()],
            self.n_instances_host,
        ):
            link_index = [link, environment] if qd.static(self.rigid.rigid_config.batch_links_info) else link
            for joint in range(
                self.rigid.dyn_info.links.joint_start[link_index],
                self.rigid.dyn_info.links.joint_end[link_index],
            ):
                joint_index = [joint, environment] if qd.static(self.rigid.rigid_config.batch_joints_info) else joint
                joint_type = self.rigid.dyn_info.joints.type[joint_index]
                q_start = self.rigid.dyn_info.joints.q_start[joint_index]
                q_end = self.rigid.dyn_info.joints.q_end[joint_index]
                dof_start = self.rigid.dyn_info.joints.dof_start[joint_index]
                if joint_type == gs.JOINT_TYPE.FREE:
                    for axis in qd.static(range(3)):
                        velocity = (
                            self.rigid.dyn_state.dofs.vel[dof_start + axis, environment]
                            + h
                            * self.rigid.constraint_state.qacc[
                                dof_start + axis,
                                environment,
                            ]
                        )
                        self.endpoint_qpos[q_start + axis, environment] = (
                            self.rigid.rigid_info.qpos[q_start + axis, environment] + h * velocity
                        )
                    angular = qd.Vector.zero(qd.f64, 3)
                    for axis in qd.static(range(3)):
                        angular[axis] = h * (
                            self.rigid.dyn_state.dofs.vel[
                                dof_start + axis + 3,
                                environment,
                            ]
                            + h
                            * self.rigid.constraint_state.qacc[
                                dof_start + axis + 3,
                                environment,
                            ]
                        )
                    delta = gu.qd_rotvec_to_quat(
                        angular,
                        self.rigid.rigid_info.EPS[None],
                    )
                    current = qd.Vector(
                        [
                            self.rigid.rigid_info.qpos[q_start + 3, environment],
                            self.rigid.rigid_info.qpos[q_start + 4, environment],
                            self.rigid.rigid_info.qpos[q_start + 5, environment],
                            self.rigid.rigid_info.qpos[q_start + 6, environment],
                        ]
                    )
                    endpoint = gu.qd_transform_quat_by_quat(delta, current)
                    for axis in qd.static(range(4)):
                        self.endpoint_qpos[q_start + 3 + axis, environment] = endpoint[axis]
                elif joint_type == gs.JOINT_TYPE.SPHERICAL:
                    angular = qd.Vector.zero(qd.f64, 3)
                    for axis in qd.static(range(3)):
                        angular[axis] = h * (
                            self.rigid.dyn_state.dofs.vel[
                                dof_start + axis,
                                environment,
                            ]
                            + h
                            * self.rigid.constraint_state.qacc[
                                dof_start + axis,
                                environment,
                            ]
                        )
                    delta = gu.qd_rotvec_to_quat(
                        angular,
                        self.rigid.rigid_info.EPS[None],
                    )
                    current = qd.Vector(
                        [
                            self.rigid.rigid_info.qpos[q_start, environment],
                            self.rigid.rigid_info.qpos[q_start + 1, environment],
                            self.rigid.rigid_info.qpos[q_start + 2, environment],
                            self.rigid.rigid_info.qpos[q_start + 3, environment],
                        ]
                    )
                    endpoint = gu.qd_transform_quat_by_quat(delta, current)
                    for axis in qd.static(range(4)):
                        self.endpoint_qpos[q_start + axis, environment] = endpoint[axis]
                elif joint_type != gs.JOINT_TYPE.FIXED:
                    for local_q in range(q_end - q_start):
                        self.endpoint_qpos[q_start + local_q, environment] = self.rigid.rigid_info.qpos[
                            q_start + local_q, environment
                        ] + h * (
                            self.rigid.dyn_state.dofs.vel[
                                dof_start + local_q,
                                environment,
                            ]
                            + h
                            * self.rigid.constraint_state.qacc[
                                dof_start + local_q,
                                environment,
                            ]
                        )

        for task in range(self.n_entities_host * self.n_instances_host):
            environment = task // self.n_entities_host
            entity = task - environment * self.n_entities_host
            func_forward_kinematics_scratch(
                environment,
                environment,
                environment,
                entity,
                0,
                0,
                0,
                self.endpoint_qpos,
                self.endpoint_link_pos,
                self.endpoint_link_quat,
                self.endpoint_joint_xanchor,
                self.endpoint_joint_xaxis,
                self.rigid.dyn_state,
                self.rigid.dyn_info,
                self.rigid.rigid_info,
                self.rigid.rigid_config,
            )

        for body in range(self.n_mechanism_bodies[()]):
            environment = body // self.rigid.n_links[()]
            link = body - environment * self.rigid.n_links[()]
            link_index = [link, environment] if qd.static(self.rigid.rigid_config.batch_links_info) else link
            link_position = self.endpoint_link_pos[link, environment]
            link_quaternion = self.endpoint_link_quat[link, environment]
            inertial_position = self.rigid.dyn_info.links.inertial_pos[link_index]
            inertial_quaternion = self.rigid.dyn_info.links.inertial_quat[link_index]
            endpoint_position = link_position + gu.qd_transform_by_quat(
                inertial_position,
                link_quaternion,
            )
            endpoint_quaternion = gu.qd_transform_quat_by_quat(
                inertial_quaternion,
                link_quaternion,
            )
            for axis in qd.static(range(3)):
                self.endpoint_t[body][axis] = endpoint_position[axis]
            for axis in qd.static(range(4)):
                self.endpoint_quat[body][axis] = endpoint_quaternion[axis]

        for edge in range(self.n_edges[()]):
            child = self.edge_child[edge]
            parent = self.edge_parent[edge]
            environment = child // self.rigid.n_links[()]
            link = child - environment * self.rigid.n_links[()]
            dof = self.edge_dof_index[edge] - environment * self.rigid.n_dofs_per_instance[()]
            link_index = [link, environment] if qd.static(self.rigid.rigid_config.batch_links_info) else link
            basis = qd.Vector.zero(qd.f64, 6)
            for joint in range(
                self.rigid.dyn_info.links.joint_start[link_index],
                self.rigid.dyn_info.links.joint_end[link_index],
            ):
                joint_index = [joint, environment] if qd.static(self.rigid.rigid_config.batch_joints_info) else joint
                if (
                    dof >= self.rigid.dyn_info.joints.dof_start[joint_index]
                    and dof < self.rigid.dyn_info.joints.dof_end[joint_index]
                ):
                    basis = self._body_point_twist(
                        link,
                        environment,
                        joint,
                        dof,
                    )
            arm = self.endpoint_t[child] - self.endpoint_t[parent]
            for component in qd.static(range(6)):
                self.edge_basis[edge, component] = basis[component]
            for axis in qd.static(range(3)):
                self.edge_arm[edge, axis] = arm[axis]

    @qd.func
    def forest_expand_body_p_cached(self, reduced: qd.template(), body):
        environment = body // self.rigid.n_links[()]
        link = body - environment * self.rigid.n_links[()]
        parent = self.parent_body[body]
        twist = qd.Vector.zero(qd.f64, 6)
        if parent >= 0:
            edge = self.parent_edge[body]
            parent_linear = qd.Vector.zero(qd.f64, 3)
            parent_angular = qd.Vector.zero(qd.f64, 3)
            for axis in qd.static(range(3)):
                parent_linear[axis] = self.body_twist[parent, axis]
                parent_angular[axis] = self.body_twist[parent, axis + 3]
            offset = self.endpoint_t[body] - self.endpoint_t[parent]
            if edge >= 0:
                offset = qd.Vector(
                    [
                        self.edge_arm[edge, 0],
                        self.edge_arm[edge, 1],
                        self.edge_arm[edge, 2],
                    ]
                )
            shifted_linear = parent_linear + parent_angular.cross(offset)
            for axis in qd.static(range(3)):
                twist[axis] = shifted_linear[axis]
                twist[axis + 3] = parent_angular[axis]

            if edge >= 0:
                reduced_index = self.rigid.dof_offset[()] + self.edge_dof_index[edge]
                coefficient = reduced[reduced_index]
                for component in qd.static(range(6)):
                    twist[component] = twist[component] + coefficient * self.edge_basis[edge, component]
        else:
            link_index = [link, environment] if qd.static(self.rigid.rigid_config.batch_links_info) else link
            for joint in range(
                self.rigid.dyn_info.links.joint_start[link_index],
                self.rigid.dyn_info.links.joint_end[link_index],
            ):
                joint_index = [joint, environment] if qd.static(self.rigid.rigid_config.batch_joints_info) else joint
                for dof in range(
                    self.rigid.dyn_info.joints.dof_start[joint_index],
                    self.rigid.dyn_info.joints.dof_end[joint_index],
                ):
                    reduced_index = self.rigid.dof_offset[()] + environment * self.rigid.n_dofs_per_instance[()] + dof
                    basis = self._body_point_twist(
                        link,
                        environment,
                        joint,
                        dof,
                    )
                    coefficient = reduced[reduced_index]
                    twist = twist + coefficient * basis
        for component in qd.static(range(6)):
            self.body_twist[body, component] = twist[component]

    @qd.func
    def forest_expand_body_p(self, reduced: qd.template(), body):
        environment = body // self.rigid.n_links[()]
        link = body - environment * self.rigid.n_links[()]
        parent = self.parent_body[body]
        twist = qd.Vector.zero(qd.f64, 6)
        if parent >= 0:
            parent_linear = qd.Vector.zero(qd.f64, 3)
            parent_angular = qd.Vector.zero(qd.f64, 3)
            for axis in qd.static(range(3)):
                parent_linear[axis] = self.body_twist[parent, axis]
                parent_angular[axis] = self.body_twist[parent, axis + 3]
            offset = self.endpoint_t[body] - self.endpoint_t[parent]
            shifted_linear = parent_linear + parent_angular.cross(offset)
            for axis in qd.static(range(3)):
                twist[axis] = shifted_linear[axis]
                twist[axis + 3] = parent_angular[axis]

        link_index = [link, environment] if qd.static(self.rigid.rigid_config.batch_links_info) else link
        for joint in range(
            self.rigid.dyn_info.links.joint_start[link_index],
            self.rigid.dyn_info.links.joint_end[link_index],
        ):
            joint_index = [joint, environment] if qd.static(self.rigid.rigid_config.batch_joints_info) else joint
            for dof in range(
                self.rigid.dyn_info.joints.dof_start[joint_index],
                self.rigid.dyn_info.joints.dof_end[joint_index],
            ):
                reduced_index = self.rigid.dof_offset[()] + environment * self.rigid.n_dofs_per_instance[()] + dof
                basis = self._body_point_twist(
                    link,
                    environment,
                    joint,
                    dof,
                )
                coefficient = reduced[reduced_index]
                twist = twist + coefficient * basis
        for component in qd.static(range(6)):
            self.body_twist[body, component] = twist[component]

    @qd.func(requires_top_level=True)
    def genesis_legacy_expand_reduced_direction(self, reduced: qd.template()):
        for level in qd.static(range(self.n_links_host)):
            qd.loop_config(name="genesis_legacy_expand_level")
            for body in range(self.n_mechanism_bodies[()]):
                if level < self.n_levels[()] and self.depth[body] == level:
                    self.forest_expand_body_p(reduced, body)

    @qd.func(requires_top_level=True)
    def forest_expand_level_p(self, reduced: qd.template()):
        for level in qd.static(range(self.n_links_host)):
            qd.loop_config(name="forest_expand_level_p")
            for order_index in range(
                self.depth_start[level],
                self.depth_start[level + 1],
            ):
                self.forest_expand_body_p_cached(
                    reduced,
                    self.depth_order[order_index],
                )

    @qd.func(requires_top_level=True)
    def forest_expand_tree_p(self, reduced: qd.template()):
        qd.loop_config(name="forest_expand_tree_p", block_dim=128)
        for tree in range(8):
            if tree < self.n_trees[()]:
                begin = self.tree_body_start[tree]
                end = self.tree_body_start[tree + 1]
                for order_index in range(begin, end):
                    self.forest_expand_body_p_cached(
                        reduced,
                        self.tree_body_list[order_index],
                    )

    @qd.func(requires_top_level=True)
    def expand_reduced_direction(self, reduced: qd.template()):
        for body in range(self.n_bodies[()]):
            for component in qd.static(range(6)):
                self.body_twist[body, component] = 0.0
        self.expand_reduced_direction_from_zero(reduced)

    @qd.func(requires_top_level=True)
    def expand_mechanism_direction(self, reduced: qd.template()):
        if qd.static(self.genesis_legacy_enabled):
            self.genesis_legacy_expand_reduced_direction(reduced)
        elif qd.static(self.use_fused_tree_path):
            self.forest_expand_tree_p(reduced)
        else:
            self.forest_expand_level_p(reduced)

    @qd.func(requires_top_level=True)
    def expand_reduced_direction_from_zero(self, reduced: qd.template()):
        self.expand_mechanism_direction(reduced)
        if qd.static(self.has_contact_proxy):
            for pair in range(self.contact_proxy.n_pairs[()]):
                mechanism = self.contact_proxy.mechanism_body[pair]
                proxy = self.contact_proxy.proxy_body[pair]
                tangent = qd.Matrix.zero(qd.f64, 6, 6)
                mechanism_twist = qd.Vector.zero(qd.f64, 6)
                for row in qd.static(range(6)):
                    mechanism_twist[row] = self.body_twist[mechanism, row]
                    for column in qd.static(range(6)):
                        tangent[row, column] = self.contact_proxy.tangent_map[pair, row, column]
                proxy_twist = expand_proxy_twist(tangent, mechanism_twist)
                for component in qd.static(range(6)):
                    self.body_twist[proxy, component] = proxy_twist[component]

    @qd.func(requires_top_level=True)
    def clear_body_wrench(self):
        for body in range(self.n_bodies[()]):
            for component in qd.static(range(6)):
                self.body_wrench[body, component] = 0.0

    @qd.func(requires_top_level=True)
    def restrict_proxy_wrenches(self):
        if qd.static(self.has_contact_proxy):
            for pair in range(self.contact_proxy.n_pairs[()]):
                mechanism = self.contact_proxy.mechanism_body[pair]
                proxy = self.contact_proxy.proxy_body[pair]
                tangent = qd.Matrix.zero(qd.f64, 6, 6)
                proxy_wrench = qd.Vector.zero(qd.f64, 6)
                for row in qd.static(range(6)):
                    proxy_wrench[row] = self.body_wrench[proxy, row]
                    for column in qd.static(range(6)):
                        tangent[row, column] = self.contact_proxy.tangent_map[pair, row, column]
                mapped = restrict_proxy_wrench(tangent, proxy_wrench)
                for component in qd.static(range(6)):
                    qd.atomic_add(self.body_wrench[mechanism, component], mapped[component])

    @qd.func
    def forest_project_body_Ap_cached(
        self,
        reduced: qd.template(),
        body,
        use_atomics: qd.template(),
    ):
        environment = body // self.rigid.n_links[()]
        link = body - environment * self.rigid.n_links[()]
        wrench = qd.Vector.zero(qd.f64, 6)
        for component in qd.static(range(6)):
            wrench[component] = self.body_wrench[body, component]

        parent = self.parent_body[body]
        if parent >= 0:
            edge = self.parent_edge[body]
            if edge >= 0:
                contribution = qd.f64(0.0)
                for component in qd.static(range(6)):
                    contribution = contribution + self.edge_basis[edge, component] * wrench[component]
                reduced_index = self.rigid.dof_offset[()] + self.edge_dof_index[edge]
                if qd.static(use_atomics):
                    qd.atomic_add(reduced[reduced_index], contribution)
                else:
                    reduced[reduced_index] = reduced[reduced_index] + contribution

            offset = self.endpoint_t[body] - self.endpoint_t[parent]
            if edge >= 0:
                offset = qd.Vector(
                    [
                        self.edge_arm[edge, 0],
                        self.edge_arm[edge, 1],
                        self.edge_arm[edge, 2],
                    ]
                )
            force = qd.Vector([wrench[0], wrench[1], wrench[2]])
            torque = qd.Vector([wrench[3], wrench[4], wrench[5]])
            parent_torque = torque + offset.cross(force)
            for axis in qd.static(range(3)):
                if qd.static(use_atomics):
                    qd.atomic_add(self.body_wrench[parent, axis], force[axis])
                    qd.atomic_add(
                        self.body_wrench[parent, axis + 3],
                        parent_torque[axis],
                    )
                else:
                    self.body_wrench[parent, axis] = self.body_wrench[parent, axis] + force[axis]
                    self.body_wrench[parent, axis + 3] = self.body_wrench[parent, axis + 3] + parent_torque[axis]
        else:
            link_index = [link, environment] if qd.static(self.rigid.rigid_config.batch_links_info) else link
            for joint in range(
                self.rigid.dyn_info.links.joint_start[link_index],
                self.rigid.dyn_info.links.joint_end[link_index],
            ):
                joint_index = [joint, environment] if qd.static(self.rigid.rigid_config.batch_joints_info) else joint
                for dof in range(
                    self.rigid.dyn_info.joints.dof_start[joint_index],
                    self.rigid.dyn_info.joints.dof_end[joint_index],
                ):
                    basis = self._body_point_twist(
                        link,
                        environment,
                        joint,
                        dof,
                    )
                    reduced_index = self.rigid.dof_offset[()] + environment * self.rigid.n_dofs_per_instance[()] + dof
                    contribution = basis.dot(wrench)
                    if qd.static(use_atomics):
                        qd.atomic_add(reduced[reduced_index], contribution)
                    else:
                        reduced[reduced_index] = reduced[reduced_index] + contribution

    @qd.func
    def forest_project_body_Ap(
        self,
        reduced: qd.template(),
        body,
        use_atomics: qd.template(),
    ):
        environment = body // self.rigid.n_links[()]
        link = body - environment * self.rigid.n_links[()]
        wrench = qd.Vector.zero(qd.f64, 6)
        for component in qd.static(range(6)):
            wrench[component] = self.body_wrench[body, component]

        link_index = [link, environment] if qd.static(self.rigid.rigid_config.batch_links_info) else link
        for joint in range(
            self.rigid.dyn_info.links.joint_start[link_index],
            self.rigid.dyn_info.links.joint_end[link_index],
        ):
            joint_index = [joint, environment] if qd.static(self.rigid.rigid_config.batch_joints_info) else joint
            for dof in range(
                self.rigid.dyn_info.joints.dof_start[joint_index],
                self.rigid.dyn_info.joints.dof_end[joint_index],
            ):
                basis = self._body_point_twist(
                    link,
                    environment,
                    joint,
                    dof,
                )
                reduced_index = self.rigid.dof_offset[()] + environment * self.rigid.n_dofs_per_instance[()] + dof
                contribution = basis.dot(wrench)
                if qd.static(use_atomics):
                    qd.atomic_add(reduced[reduced_index], contribution)
                else:
                    reduced[reduced_index] = reduced[reduced_index] + contribution

        parent = self.parent_body[body]
        if parent >= 0:
            offset = self.endpoint_t[body] - self.endpoint_t[parent]
            force = qd.Vector([wrench[0], wrench[1], wrench[2]])
            torque = qd.Vector([wrench[3], wrench[4], wrench[5]])
            parent_torque = torque + offset.cross(force)
            for axis in qd.static(range(3)):
                if qd.static(use_atomics):
                    qd.atomic_add(self.body_wrench[parent, axis], force[axis])
                    qd.atomic_add(
                        self.body_wrench[parent, axis + 3],
                        parent_torque[axis],
                    )
                else:
                    self.body_wrench[parent, axis] = self.body_wrench[parent, axis] + force[axis]
                    self.body_wrench[parent, axis + 3] = self.body_wrench[parent, axis + 3] + parent_torque[axis]

    @qd.func(requires_top_level=True)
    def genesis_legacy_restrict_body_wrenches(self, reduced: qd.template()):
        for reverse_level in qd.static(range(self.n_links_host)):
            level = self.max_depth[()] - reverse_level
            qd.loop_config(name="genesis_legacy_project_level")
            for body in range(self.n_mechanism_bodies[()]):
                if reverse_level < self.n_levels[()] and self.depth[body] == level:
                    self.forest_project_body_Ap(reduced, body, True)

    @qd.func(requires_top_level=True)
    def forest_project_level_Ap(self, reduced: qd.template()):
        for reverse_level in qd.static(range(self.n_links_host)):
            level = self.n_links_host - reverse_level - 1
            qd.loop_config(name="forest_project_level_Ap")
            for order_index in range(
                self.depth_start[level],
                self.depth_start[level + 1],
            ):
                self.forest_project_body_Ap_cached(
                    reduced,
                    self.depth_order[order_index],
                    True,
                )

    @qd.func(requires_top_level=True)
    def forest_project_tree_Ap(self, reduced: qd.template()):
        qd.loop_config(name="forest_project_tree_Ap", block_dim=128)
        for tree in range(8):
            if tree < self.n_trees[()]:
                begin = self.tree_body_start[tree]
                end = self.tree_body_start[tree + 1]
                for reverse_index in range(end - begin):
                    order_index = end - reverse_index - 1
                    self.forest_project_body_Ap_cached(
                        reduced,
                        self.tree_body_list[order_index],
                        False,
                    )

    @qd.func(requires_top_level=True)
    def restrict_body_wrenches(self, reduced: qd.template()):
        if qd.static(self.genesis_legacy_enabled):
            self.genesis_legacy_restrict_body_wrenches(reduced)
        elif qd.static(self.use_fused_tree_path):
            self.forest_project_tree_Ap(reduced)
        else:
            self.forest_project_level_Ap(reduced)

    @qd.func(requires_top_level=True)
    def prepare_particular(self):
        for dof in range(self.total_dof[()]):
            self.physical_p[dof] = 0.0
        if qd.static(self.has_contact_proxy):
            for pair in range(self.contact_proxy.n_pairs[()]):
                for component in qd.static(range(6)):
                    self.physical_p[self.proxy_dof_offset[()] + pair * 6 + component] = self.contact_proxy.particular[
                        pair, component
                    ]

    @qd.func(requires_top_level=True)
    def particular_spmv(self, linear_system: qd.template()):
        for dof in range(self.total_dof[()]):
            self.physical_Ap[dof] = 0.0
        linear_system.spmv(self.physical_p, self.physical_Ap)

    @qd.func(requires_top_level=True)
    def project_physical_rhs(self, linear_system: qd.template()):
        for dof in range(self.proxy_dof_offset[()]):
            linear_system.b_rhs[dof] = linear_system.b_rhs[dof] + self.physical_Ap[dof]

        self.clear_body_wrench()
        if qd.static(self.has_contact_proxy):
            for pair in range(self.contact_proxy.n_pairs[()]):
                proxy = self.contact_proxy.proxy_body[pair]
                offset = self.proxy_dof_offset[()] + pair * 6
                proxy_wrench = qd.Vector.zero(qd.f64, 6)
                normal = qd.Matrix.zero(qd.f64, 6, 6)
                for row in qd.static(range(6)):
                    proxy_wrench[row] = linear_system.b_rhs[offset + row] + self.physical_Ap[offset + row]
                    self.body_wrench[proxy, row] = proxy_wrench[row]
                    for column in qd.static(range(6)):
                        normal[row, column] = self.contact_proxy.normal_map[pair, row, column]

                if self.contact_proxy.restoration_active[()] != 0:
                    slack_rhs = restrict_slack_wrench(normal, proxy_wrench) + qd.Vector(
                        [
                            self.contact_proxy.lambda_[pair, 0],
                            self.contact_proxy.lambda_[pair, 1],
                            self.contact_proxy.lambda_[pair, 2],
                            self.contact_proxy.lambda_[pair, 3],
                            self.contact_proxy.lambda_[pair, 4],
                            self.contact_proxy.lambda_[pair, 5],
                        ]
                    )
                    for component in qd.static(range(6)):
                        linear_system.b_rhs[offset + component] = slack_rhs[component]
                else:
                    for component in qd.static(range(6)):
                        linear_system.b_rhs[offset + component] = 0.0
        self.restrict_proxy_wrenches()
        self.restrict_body_wrenches(linear_system.b_rhs)

    @qd.func(requires_top_level=True)
    def prepare_physical_direction(self, reduced: qd.template()):
        for dof in range(self.total_dof[()]):
            self.physical_p[dof] = reduced[dof]
            if dof < self.n_bodies[()] * 6:
                body = dof // 6
                component = dof - body * 6
                self.body_twist[body, component] = 0.0
        self.expand_mechanism_direction(reduced)
        if qd.static(self.has_contact_proxy):
            for pair in range(self.contact_proxy.n_pairs[()]):
                mechanism = self.contact_proxy.mechanism_body[pair]
                proxy = self.contact_proxy.proxy_body[pair]
                tangent = qd.Matrix.zero(qd.f64, 6, 6)
                normal = qd.Matrix.zero(qd.f64, 6, 6)
                mechanism_twist = qd.Vector.zero(qd.f64, 6)
                slack = qd.Vector.zero(qd.f64, 6)
                offset = self.proxy_dof_offset[()] + pair * 6
                for row in qd.static(range(6)):
                    mechanism_twist[row] = self.body_twist[mechanism, row]
                    slack[row] = reduced[offset + row]
                    for column in qd.static(range(6)):
                        tangent[row, column] = self.contact_proxy.tangent_map[
                            pair,
                            row,
                            column,
                        ]
                        normal[row, column] = self.contact_proxy.normal_map[pair, row, column]
                value = expand_proxy_twist(tangent, mechanism_twist)
                for component in qd.static(range(6)):
                    self.body_twist[proxy, component] = value[component]
                if self.contact_proxy.restoration_active[()] != 0:
                    value = value + expand_slack_twist(normal, slack)
                for component in qd.static(range(6)):
                    self.physical_p[offset + component] = value[component]

    @qd.func(requires_top_level=True)
    def finish_reduced_spmv(
        self,
        reduced: qd.template(),
        result: qd.template(),
    ):
        for dof in range(self.proxy_dof_offset[()]):
            result[dof] = result[dof] + self.physical_Ap[dof]

        # RigidSystem contributes Genesis native curvature. This projection
        # adds only the physical BCOO/contact wrench to generalized rows.
        self.clear_body_wrench()
        if qd.static(self.has_contact_proxy):
            for pair in range(self.contact_proxy.n_pairs[()]):
                mechanism = self.contact_proxy.mechanism_body[pair]
                proxy = self.contact_proxy.proxy_body[pair]
                offset = self.proxy_dof_offset[()] + pair * 6
                proxy_wrench = qd.Vector.zero(qd.f64, 6)
                tangent = qd.Matrix.zero(qd.f64, 6, 6)
                normal = qd.Matrix.zero(qd.f64, 6, 6)
                metric = qd.Matrix.zero(qd.f64, 6, 6)
                slack = qd.Vector.zero(qd.f64, 6)
                for row in qd.static(range(6)):
                    proxy_wrench[row] = self.physical_Ap[offset + row]
                    self.body_wrench[proxy, row] = proxy_wrench[row]
                    slack[row] = reduced[offset + row]
                    for column in qd.static(range(6)):
                        tangent[row, column] = self.contact_proxy.tangent_map[
                            pair,
                            row,
                            column,
                        ]
                        normal[row, column] = self.contact_proxy.normal_map[pair, row, column]
                        metric[row, column] = self.contact_proxy.metric[pair, row, column]
                mapped = restrict_proxy_wrench(tangent, proxy_wrench)
                for component in qd.static(range(6)):
                    qd.atomic_add(
                        self.body_wrench[mechanism, component],
                        mapped[component],
                    )
                if self.contact_proxy.restoration_active[()] != 0:
                    slack_result = restrict_slack_wrench(normal, proxy_wrench) + metric @ slack
                    for component in qd.static(range(6)):
                        result[offset + component] = result[offset + component] + slack_result[component]
                else:
                    for component in qd.static(range(6)):
                        result[offset + component] = result[offset + component] + reduced[offset + component]
        self.restrict_body_wrenches(result)

    @qd.func(requires_top_level=True)
    def forest_inertia_wrench(self):
        qd.loop_config(name="forest_inertia_wrench")
        for body in range(self.n_bodies[()]):
            for row in qd.static(range(6)):
                value = qd.f64(0.0)
                if body < self.n_mechanism_bodies[()]:
                    for column in qd.static(range(6)):
                        value = value + self.body_inertia[body, row, column] * self.body_twist[body, column]
                self.body_wrench[body, row] = value

    @qd.func(requires_top_level=True)
    def forest_control_matvec(
        self,
        reduced: qd.template(),
        result: qd.template(),
    ):
        qd.loop_config(name="forest_control_matvec")
        for dof, environment in qd.ndrange(
            self.rigid.n_dofs_per_instance[()],
            self.rigid.n_instances[()],
        ):
            dof_index = [dof, environment] if qd.static(self.rigid.rigid_config.batch_dofs_info) else dof
            augmentation = (
                self.rigid.dyn_info.dofs.armature[dof_index]
                + self.rigid.h * self.rigid.dyn_info.dofs.damping[dof_index]
            )
            if self.rigid.dyn_state.dofs.ctrl_mode[dof, environment] <= gs.CTRL_MODE.VELOCITY:
                augmentation = augmentation - self.rigid.dyn_info.dofs.act_bias[dof_index][2] * self.rigid.h
            reduced_index = self.rigid.dof_offset[()] + environment * self.rigid.n_dofs_per_instance[()] + dof
            result[reduced_index] = result[reduced_index] + augmentation * reduced[reduced_index]

    @qd.func(requires_top_level=True)
    def expand_solution(self, solution: qd.template()):
        self.expand_reduced_direction(solution)
        if qd.static(self.has_contact_proxy):
            for pair in range(self.contact_proxy.n_pairs[()]):
                proxy = self.contact_proxy.proxy_body[pair]
                offset = self.proxy_dof_offset[()] + pair * 6
                normal = qd.Matrix.zero(qd.f64, 6, 6)
                slack = qd.Vector.zero(qd.f64, 6)
                for row in qd.static(range(6)):
                    slack[row] = solution[offset + row]
                    for column in qd.static(range(6)):
                        normal[row, column] = self.contact_proxy.normal_map[pair, row, column]
                value = qd.Vector.zero(qd.f64, 6)
                for component in qd.static(range(6)):
                    value[component] = (
                        self.contact_proxy.particular[pair, component] - self.body_twist[proxy, component]
                    )
                if self.contact_proxy.restoration_active[()] != 0:
                    value = value - expand_slack_twist(normal, slack)
                    for component in qd.static(range(6)):
                        self.contact_proxy.slack[pair, component] = -slack[component]
                    qd.atomic_max(
                        self.contact_proxy.restoration_max_slack[()],
                        slack.norm(),
                    )
                else:
                    for component in qd.static(range(6)):
                        self.contact_proxy.slack[pair, component] = 0.0
                for component in qd.static(range(6)):
                    self.contact_proxy.dq[pair, component] = value[component]

    @qd.func
    def curvature_bound(self, body, radius):
        reach = radius
        speed = qd.f64(0.0)
        angular_speed = qd.f64(0.0)
        node = body
        while node >= 0 and self.parent_body[node] >= 0:
            parent = self.parent_body[node]
            environment = node // self.rigid.n_links[()]
            link = node - environment * self.rigid.n_links[()]
            link_index = [link, environment] if qd.static(self.rigid.rigid_config.batch_links_info) else link
            child_anchor_distance = qd.f64(0.0)
            parent_anchor_distance = (self.endpoint_t[node] - self.endpoint_t[parent]).norm()
            has_joint = False
            has_rotation = False
            has_prismatic = False
            for joint in range(
                self.rigid.dyn_info.links.joint_start[link_index],
                self.rigid.dyn_info.links.joint_end[link_index],
            ):
                joint_index = [joint, environment] if qd.static(self.rigid.rigid_config.batch_joints_info) else joint
                joint_type = self.rigid.dyn_info.joints.type[joint_index]
                has_joint = True
                has_rotation = has_rotation or (
                    joint_type == gs.JOINT_TYPE.REVOLUTE or joint_type == gs.JOINT_TYPE.SPHERICAL
                )
                has_prismatic = has_prismatic or (joint_type == gs.JOINT_TYPE.PRISMATIC)
                anchor = self.endpoint_joint_xanchor[joint, environment]
                child_anchor_distance = qd.max(
                    child_anchor_distance,
                    (self.endpoint_t[node] - anchor).norm(),
                )
                parent_anchor_distance = qd.max(
                    parent_anchor_distance,
                    (self.endpoint_t[parent] - anchor).norm(),
                )

            parent_linear = qd.Vector.zero(qd.f64, 3)
            parent_angular = qd.Vector.zero(qd.f64, 3)
            child_linear = qd.Vector.zero(qd.f64, 3)
            child_angular = qd.Vector.zero(qd.f64, 3)
            for axis in qd.static(range(3)):
                parent_linear[axis] = self.body_twist[parent, axis]
                parent_angular[axis] = self.body_twist[parent, axis + 3]
                child_linear[axis] = self.body_twist[node, axis]
                child_angular[axis] = self.body_twist[node, axis + 3]
            arm = self.endpoint_t[node] - self.endpoint_t[parent]
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
            root_angular[axis] = self.body_twist[node, axis + 3]
        root_angular_speed = root_angular.norm()
        angular_speed = angular_speed + root_angular_speed
        nontranslational_speed = speed + root_angular_speed * reach
        return 2.0 * angular_speed * nontranslational_speed

    @qd.func
    def _motion_matrix(self, arm: qd.template()):
        motion = qd.Matrix.identity(qd.f64, 6)
        skew = rigid_contact_proxy_skew(arm)
        for row in qd.static(range(3)):
            for column in qd.static(range(3)):
                motion[row, column + 3] = -skew[row, column]
        return motion

    @qd.func
    def _root_basis(self, body):
        basis_matrix = qd.Matrix.zero(qd.f64, 6, 6)
        environment = body // self.rigid.n_links[()]
        link = body - environment * self.rigid.n_links[()]
        first_dof = self.root_dof_index[body] - environment * self.rigid.n_dofs_per_instance[()]
        link_index = [link, environment] if qd.static(self.rigid.rigid_config.batch_links_info) else link
        for column in qd.static(range(6)):
            dof = first_dof + column
            value = qd.Vector.zero(qd.f64, 6)
            for joint in range(
                self.rigid.dyn_info.links.joint_start[link_index],
                self.rigid.dyn_info.links.joint_end[link_index],
            ):
                joint_index = [joint, environment] if qd.static(self.rigid.rigid_config.batch_joints_info) else joint
                if (
                    dof >= self.rigid.dyn_info.joints.dof_start[joint_index]
                    and dof < self.rigid.dyn_info.joints.dof_end[joint_index]
                ):
                    value = self._body_point_twist(
                        link,
                        environment,
                        joint,
                        dof,
                    )
            for row in qd.static(range(6)):
                basis_matrix[row, column] = value[row]
        return basis_matrix

    @qd.func(requires_top_level=True)
    def build_preconditioner(self, linear_system: qd.template()):
        for body in range(self.n_mechanism_bodies[()]):
            environment = body // self.rigid.n_links[()]
            link = body - environment * self.rigid.n_links[()]
            link_index = [link, environment] if qd.static(self.rigid.rigid_config.batch_links_info) else link
            mass = self.rigid.dyn_info.links.inertial_mass[link_index]
            inertia = self.rigid.dyn_info.links.inertial_i[link_index]
            rotation = gu.qd_quat_to_R(
                self.endpoint_quat[body],
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
                    self.articulated_inertia[body, row, column] = value
                    self.body_inertia[body, row, column] = value
                    self.root_inverse[body, row, column] = 0.0

        for body in range(self.n_bodies[()]):
            for row in qd.static(range(6)):
                for column in qd.static(range(6)):
                    self.kkt_proxy_diagonal[body, row, column] = 0.0

        for edge in range(self.n_edges[()]):
            child = self.edge_child[edge]
            parent = self.edge_parent[edge]
            environment = child // self.rigid.n_links[()]
            link = child - environment * self.rigid.n_links[()]
            dof = self.edge_dof_index[edge] - environment * self.rigid.n_dofs_per_instance[()]
            link_index = [link, environment] if qd.static(self.rigid.rigid_config.batch_links_info) else link
            basis = qd.Vector.zero(qd.f64, 6)
            for joint in range(
                self.rigid.dyn_info.links.joint_start[link_index],
                self.rigid.dyn_info.links.joint_end[link_index],
            ):
                joint_index = [joint, environment] if qd.static(self.rigid.rigid_config.batch_joints_info) else joint
                if (
                    dof >= self.rigid.dyn_info.joints.dof_start[joint_index]
                    and dof < self.rigid.dyn_info.joints.dof_end[joint_index]
                ):
                    basis = self._body_point_twist(
                        link,
                        environment,
                        joint,
                        dof,
                    )
            arm = self.endpoint_t[child] - self.endpoint_t[parent]
            for component in qd.static(range(6)):
                self.edge_basis[edge, component] = basis[component]
                self.edge_u[edge, component] = 0.0
            for axis in qd.static(range(3)):
                self.edge_arm[edge, axis] = arm[axis]
            self.edge_d[edge] = 0.0
            self.precond_a[edge] = 0.0
            for row in qd.static(range(6)):
                for column in qd.static(range(6)):
                    self.edge_hessian[edge, row, column] = 0.0

        if qd.static(self.has_contact_proxy):
            proxy_block_base = self.proxy_dof_offset[()] // 3
            proxy_block_end = proxy_block_base + self.contact_proxy.n_pairs[()] * 2
            for index in range(linear_system.bcoo_nnz[()]):
                block_row = linear_system.bcoo_row[index]
                block_col = linear_system.bcoo_col[index]
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
                    hessian = qd.Matrix.zero(qd.f64, 3, 3)
                    for row in qd.static(range(3)):
                        for column in qd.static(range(3)):
                            hessian[row, column] = linear_system.bcoo_val[index * 9 + row * 3 + column]

                    proxy_row = self.contact_proxy.proxy_body[pair_row]
                    proxy_col = self.contact_proxy.proxy_body[pair_col]
                    if proxy_row == proxy_col:
                        for row in qd.static(range(3)):
                            for column in qd.static(range(3)):
                                qd.atomic_add(
                                    self.kkt_proxy_diagonal[
                                        proxy_row,
                                        row_block * 3 + row,
                                        col_block * 3 + column,
                                    ],
                                    hessian[row, column],
                                )
                                if block_row != block_col:
                                    qd.atomic_add(
                                        self.kkt_proxy_diagonal[
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
                            tangent_row[row, column] = self.contact_proxy.tangent_map[
                                pair_row,
                                row_block * 3 + row,
                                column,
                            ]
                            tangent_col[row, column] = self.contact_proxy.tangent_map[
                                pair_col,
                                col_block * 3 + row,
                                column,
                            ]
                    mapped = tangent_row.transpose() @ hessian @ tangent_col
                    owner_row = self.contact_proxy.mechanism_body[pair_row]
                    owner_col = self.contact_proxy.mechanism_body[pair_col]
                    if owner_row == owner_col:
                        for row in qd.static(range(6)):
                            for column in qd.static(range(6)):
                                qd.atomic_add(
                                    self.articulated_inertia[
                                        owner_row,
                                        row,
                                        column,
                                    ],
                                    mapped[row, column],
                                )
                                if block_row != block_col:
                                    qd.atomic_add(
                                        self.articulated_inertia[
                                            owner_row,
                                            column,
                                            row,
                                        ],
                                        mapped[row, column],
                                    )
                    elif self.parent_body[owner_col] == owner_row:
                        edge = self.parent_edge[owner_col]
                        if edge >= 0:
                            for row in qd.static(range(6)):
                                for column in qd.static(range(6)):
                                    qd.atomic_add(
                                        self.edge_hessian[
                                            edge,
                                            row,
                                            column,
                                        ],
                                        mapped[row, column],
                                    )
                    elif self.parent_body[owner_row] == owner_col:
                        edge = self.parent_edge[owner_row]
                        if edge >= 0:
                            for row in qd.static(range(6)):
                                for column in qd.static(range(6)):
                                    qd.atomic_add(
                                        self.edge_hessian[
                                            edge,
                                            column,
                                            row,
                                        ],
                                        mapped[row, column],
                                    )

        for reverse_level in qd.static(range(self.n_links_host)):
            level = self.max_depth[()] - reverse_level
            for body in range(self.n_mechanism_bodies[()]):
                if reverse_level < self.max_depth[()] and self.depth[body] == level:
                    parent = self.parent_body[body]
                    articulated = qd.Matrix.zero(qd.f64, 6, 6)
                    for row in qd.static(range(6)):
                        for column in qd.static(range(6)):
                            articulated[row, column] = self.articulated_inertia[body, row, column]
                    arm = self.endpoint_t[body] - self.endpoint_t[parent]
                    motion = self._motion_matrix(arm)
                    propagated = motion.transpose() @ articulated @ motion
                    edge = self.parent_edge[body]
                    if edge >= 0:
                        edge_hessian = qd.Matrix.zero(qd.f64, 6, 6)
                        basis = qd.Vector.zero(qd.f64, 6)
                        for row in qd.static(range(6)):
                            basis[row] = self.edge_basis[edge, row]
                            for column in qd.static(range(6)):
                                edge_hessian[row, column] = self.edge_hessian[edge, row, column]
                        u = (motion.transpose() @ articulated + edge_hessian) @ basis
                        environment = body // self.rigid.n_links[()]
                        dof = self.edge_dof_index[edge] - environment * self.rigid.n_dofs_per_instance[()]
                        dof_index = [dof, environment] if qd.static(self.rigid.rigid_config.batch_dofs_info) else dof
                        actuator_damping = qd.f64(0.0)
                        if (
                            self.rigid.dyn_state.dofs.ctrl_mode[
                                dof,
                                environment,
                            ]
                            <= gs.CTRL_MODE.VELOCITY
                        ):
                            actuator_damping = -self.rigid.dyn_info.dofs.act_bias[dof_index][2] * self.rigid.h
                        pivot = qd.max(
                            basis.dot(articulated @ basis)
                            + self.rigid.dyn_info.dofs.armature[dof_index]
                            + self.rigid.h * self.rigid.dyn_info.dofs.damping[dof_index]
                            + actuator_damping,
                            qd.f64(1.0e-12),
                        )
                        edge_motion = edge_hessian @ motion
                        propagated = propagated + edge_motion + edge_motion.transpose() - u.outer_product(u) / pivot
                        for component in qd.static(range(6)):
                            self.edge_u[edge, component] = u[component]
                        self.edge_d[edge] = pivot
                    for row in qd.static(range(6)):
                        for column in qd.static(range(6)):
                            qd.atomic_add(
                                self.articulated_inertia[parent, row, column],
                                propagated[row, column],
                            )

        for body in range(self.n_mechanism_bodies[()]):
            if self.parent_body[body] < 0 and self.root_dof_index[body] >= 0:
                articulated = qd.Matrix.zero(qd.f64, 6, 6)
                for row in qd.static(range(6)):
                    for column in qd.static(range(6)):
                        articulated[row, column] = self.articulated_inertia[
                            body,
                            row,
                            column,
                        ]
                basis = self._root_basis(body)
                root_hessian = basis.transpose() @ articulated @ basis
                environment = body // self.rigid.n_links[()]
                first_dof = self.root_dof_index[body] - environment * self.rigid.n_dofs_per_instance[()]
                for dof in qd.static(range(6)):
                    dof_index = (
                        [first_dof + dof, environment]
                        if qd.static(self.rigid.rigid_config.batch_dofs_info)
                        else first_dof + dof
                    )
                    actuator_damping = qd.f64(0.0)
                    if (
                        self.rigid.dyn_state.dofs.ctrl_mode[
                            first_dof + dof,
                            environment,
                        ]
                        <= gs.CTRL_MODE.VELOCITY
                    ):
                        actuator_damping = -self.rigid.dyn_info.dofs.act_bias[dof_index][2] * self.rigid.h
                    root_hessian[dof, dof] = (
                        root_hessian[dof, dof]
                        + self.rigid.dyn_info.dofs.armature[dof_index]
                        + self.rigid.h * self.rigid.dyn_info.dofs.damping[dof_index]
                        + actuator_damping
                    )
                inverse = root_hessian.inverse()
                for row in qd.static(range(6)):
                    for column in qd.static(range(6)):
                        self.root_inverse[body, row, column] = inverse[
                            row,
                            column,
                        ]

        if qd.static(self.has_contact_proxy):
            for pair in range(self.contact_proxy.n_pairs[()]):
                if self.contact_proxy.restoration_active[()] != 0:
                    proxy = self.contact_proxy.proxy_body[pair]
                    diagonal = qd.Matrix.zero(qd.f64, 6, 6)
                    normal = qd.Matrix.zero(qd.f64, 6, 6)
                    metric = qd.Matrix.zero(qd.f64, 6, 6)
                    for row in qd.static(range(6)):
                        for column in qd.static(range(6)):
                            diagonal[row, column] = self.kkt_proxy_diagonal[
                                proxy,
                                row,
                                column,
                            ]
                            normal[row, column] = self.contact_proxy.normal_map[
                                pair,
                                row,
                                column,
                            ]
                            metric[row, column] = self.contact_proxy.metric[
                                pair,
                                row,
                                column,
                            ]
                    inverse = (normal.transpose() @ diagonal @ normal + metric).inverse()
                    for row in qd.static(range(6)):
                        for column in qd.static(range(6)):
                            self.kkt_proxy_diagonal[
                                proxy,
                                row,
                                column,
                            ] = inverse[row, column]

    @qd.func(requires_top_level=True)
    def forest_precond_apply_tree_shared(
        self,
        residual: qd.template(),
        result: qd.template(),
    ):
        qd.loop_config(name="forest_precond_apply_tree_shared", block_dim=32)
        for task in range(self.n_trees[()] * 32):
            tree = task // 32
            lane = qd.simt.block.thread_idx()
            tree_force = qd.simt.block.SharedArray((64, 6), qd.f64)
            tree_velocity = qd.simt.block.SharedArray((64, 6), qd.f64)
            root_rhs = qd.simt.block.SharedArray((6,), qd.f64)
            root_solution = qd.simt.block.SharedArray((6,), qd.f64)
            scalar = qd.simt.block.SharedArray((1,), qd.f64)

            if tree < self.n_trees[()]:
                begin = self.tree_body_start[tree]
                size = self.tree_body_start[tree + 1] - begin
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
                    child = self.tree_body_list[begin + local]
                    parent = self.parent_body[child]
                    parent_local = self.tree_local_index[parent]
                    edge = self.parent_edge[child]
                    if lane == 0:
                        edge_rhs = qd.f64(0.0)
                        if edge >= 0:
                            dof = self.rigid.dof_offset[()] + self.edge_dof_index[edge]
                            edge_rhs = residual[dof]
                            for component in qd.static(range(6)):
                                edge_rhs = edge_rhs + self.edge_basis[edge, component] * tree_force[local, component]
                            self.precond_a[edge] = edge_rhs
                        scalar[0] = edge_rhs
                    qd.simt.block.sync()

                    if lane < 6:
                        value = tree_force[local, lane]
                        if lane >= 3:
                            arm = qd.Vector(
                                [
                                    self.endpoint_t[child][0] - self.endpoint_t[parent][0],
                                    self.endpoint_t[child][1] - self.endpoint_t[parent][1],
                                    self.endpoint_t[child][2] - self.endpoint_t[parent][2],
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
                            value = value - self.edge_u[edge, lane] * (scalar[0] / self.edge_d[edge])
                        tree_force[parent_local, lane] = tree_force[parent_local, lane] + value
                    qd.simt.block.sync()

                root = self.tree_body_list[begin]
                root_dof = self.root_dof_index[root]
                root_basis = qd.Matrix.zero(qd.f64, 6, 6)
                if lane < 6 and root_dof >= 0:
                    root_basis = self._root_basis(root)
                    projected_rhs = residual[self.rigid.dof_offset[()] + root_dof + lane]
                    for component in qd.static(range(6)):
                        projected_rhs = projected_rhs + root_basis[component, lane] * tree_force[0, component]
                    root_rhs[lane] = projected_rhs
                qd.simt.block.sync()

                if lane < 6 and root_dof >= 0:
                    solved = qd.f64(0.0)
                    for component in qd.static(range(6)):
                        solved = solved + self.root_inverse[root, lane, component] * root_rhs[component]
                    root_solution[lane] = solved
                    result[self.rigid.dof_offset[()] + root_dof + lane] = solved
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
                    child = self.tree_body_list[begin + local]
                    parent = self.parent_body[child]
                    parent_local = self.tree_local_index[parent]
                    edge = self.parent_edge[child]
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
                                    self.endpoint_t[child][0] - self.endpoint_t[parent][0],
                                    self.endpoint_t[child][1] - self.endpoint_t[parent][1],
                                    self.endpoint_t[child][2] - self.endpoint_t[parent][2],
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
                                    projection + self.edge_u[edge, component] * tree_velocity[parent_local, component]
                                )
                            edge_velocity = (self.precond_a[edge] - projection) / self.edge_d[edge]
                            result[self.rigid.dof_offset[()] + self.edge_dof_index[edge]] = edge_velocity
                        scalar[0] = edge_velocity
                    qd.simt.block.sync()

                    if lane < 6 and edge >= 0:
                        tree_velocity[local, lane] = (
                            tree_velocity[local, lane] + self.edge_basis[edge, lane] * scalar[0]
                        )
                    qd.simt.block.sync()

                flat = lane
                while flat < size * 6:
                    local = flat // 6
                    component = flat - local * 6
                    body = self.tree_body_list[begin + local]
                    self.precond_force[body, component] = tree_force[
                        local,
                        component,
                    ]
                    self.precond_velocity[body, component] = tree_velocity[
                        local,
                        component,
                    ]
                    flat = flat + 32

    @qd.func(requires_top_level=True)
    def forest_precond_apply_level(
        self,
        residual: qd.template(),
        result: qd.template(),
    ):
        for dof in range(self.rigid.n_storage_dofs[()]):
            result[self.rigid.dof_offset[()] + dof] = 0.0
        for body in range(self.n_mechanism_bodies[()]):
            for component in qd.static(range(6)):
                self.precond_force[body, component] = 0.0
                self.precond_velocity[body, component] = 0.0

        for reverse_level in qd.static(range(self.n_links_host)):
            level = self.max_depth[()] - reverse_level
            for body in range(self.n_mechanism_bodies[()]):
                if reverse_level < self.max_depth[()] and self.depth[body] == level:
                    parent = self.parent_body[body]
                    force = qd.Vector.zero(qd.f64, 6)
                    for component in qd.static(range(6)):
                        force[component] = self.precond_force[body, component]
                    arm = self.endpoint_t[body] - self.endpoint_t[parent]
                    motion = self._motion_matrix(arm)
                    propagated = motion.transpose() @ force
                    edge = self.parent_edge[body]
                    if edge >= 0:
                        basis = qd.Vector.zero(qd.f64, 6)
                        u = qd.Vector.zero(qd.f64, 6)
                        for component in qd.static(range(6)):
                            basis[component] = self.edge_basis[edge, component]
                            u[component] = self.edge_u[edge, component]
                        dof = self.rigid.dof_offset[()] + self.edge_dof_index[edge]
                        edge_rhs = residual[dof] + basis.dot(force)
                        self.precond_a[edge] = edge_rhs
                        propagated = propagated - u * (edge_rhs / self.edge_d[edge])
                    for component in qd.static(range(6)):
                        qd.atomic_add(
                            self.precond_force[parent, component],
                            propagated[component],
                        )

        for body in range(self.n_mechanism_bodies[()]):
            root_dof = self.root_dof_index[body]
            if self.parent_body[body] < 0 and root_dof >= 0:
                basis = self._root_basis(body)
                inverse = qd.Matrix.zero(qd.f64, 6, 6)
                force = qd.Vector.zero(qd.f64, 6)
                rhs = qd.Vector.zero(qd.f64, 6)
                for row in qd.static(range(6)):
                    force[row] = self.precond_force[body, row]
                    rhs[row] = residual[self.rigid.dof_offset[()] + root_dof + row]
                    for column in qd.static(range(6)):
                        inverse[row, column] = self.root_inverse[
                            body,
                            row,
                            column,
                        ]
                solved = inverse @ (rhs + basis.transpose() @ force)
                velocity = basis @ solved
                for component in qd.static(range(6)):
                    result[self.rigid.dof_offset[()] + root_dof + component] = solved[component]
                    self.precond_velocity[body, component] = velocity[component]

        for level_slot in qd.static(range(self.n_links_host)):
            level = level_slot + 1
            for body in range(self.n_mechanism_bodies[()]):
                if level_slot < self.max_depth[()] and self.depth[body] == level:
                    parent = self.parent_body[body]
                    parent_velocity = qd.Vector.zero(qd.f64, 6)
                    for component in qd.static(range(6)):
                        parent_velocity[component] = self.precond_velocity[parent, component]
                    arm = self.endpoint_t[body] - self.endpoint_t[parent]
                    velocity = self._motion_matrix(arm) @ parent_velocity
                    edge = self.parent_edge[body]
                    if edge >= 0:
                        basis = qd.Vector.zero(qd.f64, 6)
                        u = qd.Vector.zero(qd.f64, 6)
                        for component in qd.static(range(6)):
                            basis[component] = self.edge_basis[edge, component]
                            u[component] = self.edge_u[edge, component]
                        edge_velocity = (self.precond_a[edge] - u.dot(parent_velocity)) / self.edge_d[edge]
                        velocity = velocity + basis * edge_velocity
                        result[self.rigid.dof_offset[()] + self.edge_dof_index[edge]] = edge_velocity
                    for component in qd.static(range(6)):
                        self.precond_velocity[body, component] = velocity[component]

        for dof in range(
            self.rigid.n_dofs[()],
            self.rigid.n_storage_dofs[()],
        ):
            offset = self.rigid.dof_offset[()] + dof
            result[offset] = residual[offset]

        if qd.static(self.has_contact_proxy):
            for pair in range(self.contact_proxy.n_pairs[()]):
                offset = self.proxy_dof_offset[()] + pair * 6
                if self.contact_proxy.restoration_active[()] == 0:
                    for component in qd.static(range(6)):
                        result[offset + component] = residual[offset + component]
                else:
                    proxy = self.contact_proxy.proxy_body[pair]
                    inverse = qd.Matrix.zero(qd.f64, 6, 6)
                    rhs = qd.Vector.zero(qd.f64, 6)
                    for row in qd.static(range(6)):
                        rhs[row] = residual[offset + row]
                        for column in qd.static(range(6)):
                            inverse[row, column] = self.kkt_proxy_diagonal[
                                proxy,
                                row,
                                column,
                            ]
                    value = inverse @ rhs
                    for component in qd.static(range(6)):
                        result[offset + component] = value[component]

    @qd.func(requires_top_level=True)
    def apply_preconditioner(
        self,
        residual: qd.template(),
        result: qd.template(),
    ):
        if qd.static(self.use_fused_tree_path):
            self.forest_precond_apply_tree_shared(residual, result)
            for dof in range(
                self.rigid.n_dofs[()],
                self.rigid.n_storage_dofs[()],
            ):
                offset = self.rigid.dof_offset[()] + dof
                result[offset] = residual[offset]

            if qd.static(self.has_contact_proxy):
                for pair in range(self.contact_proxy.n_pairs[()]):
                    offset = self.proxy_dof_offset[()] + pair * 6
                    if self.contact_proxy.restoration_active[()] == 0:
                        for component in qd.static(range(6)):
                            result[offset + component] = residual[offset + component]
                    else:
                        proxy = self.contact_proxy.proxy_body[pair]
                        inverse = qd.Matrix.zero(qd.f64, 6, 6)
                        rhs = qd.Vector.zero(qd.f64, 6)
                        for row in qd.static(range(6)):
                            rhs[row] = residual[offset + row]
                            for column in qd.static(range(6)):
                                inverse[row, column] = self.kkt_proxy_diagonal[
                                    proxy,
                                    row,
                                    column,
                                ]
                        value = inverse @ rhs
                        for component in qd.static(range(6)):
                            result[offset + component] = value[component]
        else:
            self.forest_precond_apply_level(residual, result)

    @qd.func(requires_top_level=True)
    def compute_merit_directional_derivative(
        self,
        linear_system: qd.template(),
    ):
        for _ in range(1):
            self.contact_proxy.merit_gtd[()] = 0.0
        for dof in range(self.proxy_dof_offset[()]):
            qd.atomic_add(
                self.contact_proxy.merit_gtd[()],
                -self.contact_proxy.merit_gradient[dof] * linear_system.x_sol[dof],
            )
        for pair in range(self.contact_proxy.n_pairs[()]):
            offset = self.proxy_dof_offset[()] + pair * 6
            contribution = qd.f64(0.0)
            for component in qd.static(range(6)):
                contribution = contribution + (
                    self.contact_proxy.merit_gradient[offset + component] * self.contact_proxy.dq[pair, component]
                )
            qd.atomic_add(self.contact_proxy.merit_gtd[()], contribution)

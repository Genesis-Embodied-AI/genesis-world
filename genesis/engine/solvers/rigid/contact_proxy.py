"""
The rigid contact proxies of the links in the contact of the Newton solver (see genesis/engine/solvers/ipc_contact.py
and genesis/engine/ipc/se3.py).

Every link holding collision geoms of an entity that couples with the other solvers ('RigidMaterial(needs_coup)') owns a
proxy: a massless copy of the pose of the frame of the link, the only pose the contact sees. The vertices of the proxy
are those of the collision geoms of the link in that frame, published into the contact surface after the cloth. The
proxy of a fixed link takes the pose of its link at the start of every substep.
"""

from typing import NamedTuple

import numpy as np

import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu
from genesis.engine.solvers.ipc_contact import IPCState, IPCSurfacePrimitives, surface_area_weights
from genesis.engine.solvers.rigid.data import DynState
from genesis.utils.data import ENVS, Data, DataKind, Size, field

# The proxies, one per link holding collision geoms, and their vertices
PROXIES = Size("proxies")
PROXY_VERTICES = Size("proxy_vertices")


class ContactProxies(NamedTuple):
    """
    The rigid contact proxies of a scene at build (see the module docstring), one row per proxy or per vertex.

    A proxy carries its link, whether its link is fixed, and its surface radius, the largest distance of its vertices
    from the origin of its link. A vertex carries its proxy and its position in the frame of the link. `surface` holds
    the contact primitives of the proxies, their vertices in the order of the vertices of the proxies.
    """

    links_idx: np.ndarray
    is_fixed: np.ndarray
    surface_radius: np.ndarray
    vertices_proxy_idx: np.ndarray
    vertices_local_pos: np.ndarray
    surface: IPCSurfacePrimitives


class IPCProxiesInfo(Data, kind=DataKind.CONSTANT, shape=(PROXIES,)):
    """The constant parameters of the proxies (see ContactProxies)."""

    link_idx: qd.Tensor = field(int)
    is_fixed: qd.Tensor = field(bool)
    surface_radius: qd.Tensor = field(float)


class IPCProxyVerticesInfo(Data, kind=DataKind.CONSTANT, shape=(PROXY_VERTICES,)):
    """The vertices of the proxies, their proxy and their position in the frame of its link."""

    proxy_idx: qd.Tensor = field(int)
    local_pos: qd.Tensor = field((float, 3))


class IPCProxiesState(Data, kind=DataKind.STATE, shape=(PROXIES, ENVS)):
    """The pose of every proxy, accepted by the last substep, and its pose at the start of the current one."""

    pos: qd.Tensor = field((float, 3))
    quat: qd.Tensor = field((float, 4))
    pos_start: qd.Tensor = field((float, 3))
    quat_start: qd.Tensor = field((float, 4))


def build_contact_proxies(links_geoms: list, n_bodies_start: int, d_hat: float) -> ContactProxies:
    """
    Build the rigid contact proxies of the links holding collision geoms (see ContactProxies).

    `links_geoms` lists, per link owning a proxy, the link, the rank of the contact body of its entity among the rigid
    entities with moving proxies, and its collision geoms. A proxy stands for the merged meshes of its geoms in the
    frame of its link. A vertex of a fixed link belongs to no contact body (-1), the vertices of the other links to body
    `n_bodies_start` plus the rank of their entity. The contact vertices of a proxy are rigid, of no thickness and of no
    resistance (the cloth sets the resistance of their pairs), with the friction coefficient of their geom for coupling.
    """
    links_idx, is_fixed, surface_radius = [], [], []
    vertices_proxy_idx, vertices_local_pos, vertices_body_idx, vertices_friction_mu = [], [], [], []
    tris_all = []
    n_verts = 0
    for i_proxy, (link, i_body, geoms) in enumerate(links_geoms):
        proxy_verts = []
        for geom in geoms:
            verts = gu.transform_by_trans_quat(geom.init_verts, geom.init_pos, geom.init_quat)
            tris_all.append(geom.init_faces + n_verts)
            proxy_verts.append(verts)
            vertices_friction_mu.append(np.full(len(verts), geom.coup_friction, dtype=gs.np_float))
            n_verts += len(verts)
        proxy_verts = np.concatenate(proxy_verts)
        links_idx.append(link.idx)
        is_fixed.append(link.is_fixed)
        surface_radius.append(np.linalg.norm(proxy_verts, axis=-1).max())
        vertices_proxy_idx.append(np.full(len(proxy_verts), i_proxy, dtype=gs.np_int))
        vertices_local_pos.append(proxy_verts)
        vertices_body_idx.append(np.full(len(proxy_verts), -1 if link.is_fixed else n_bodies_start + i_body))
    pos = np.concatenate(vertices_local_pos)
    tris = np.concatenate(tris_all)
    areas, edges_verts_idx, edges_area_weight, edges_rest_length2, vertices_area_weight = surface_area_weights(
        pos, tris
    )
    surface = IPCSurfacePrimitives(
        vertices_thickness=np.zeros(n_verts, dtype=gs.np_float),
        vertices_d_hat=np.full(n_verts, d_hat, dtype=gs.np_float),
        vertices_area_weight=vertices_area_weight,
        vertices_resistance=np.zeros(n_verts, dtype=gs.np_float),
        vertices_friction_mu=np.concatenate(vertices_friction_mu),
        vertices_body_idx=np.concatenate(vertices_body_idx).astype(gs.np_int),
        vertices_is_rigid=np.ones(n_verts, dtype=bool),
        edges_verts_idx=edges_verts_idx,
        edges_area_weight=edges_area_weight,
        edges_rest_length2=edges_rest_length2,
        triangles_verts_idx=tris,
        triangles_area_weight=areas,
    )
    return ContactProxies(
        links_idx=np.array(links_idx, dtype=gs.np_int),
        is_fixed=np.array(is_fixed, dtype=bool),
        surface_radius=np.array(surface_radius, dtype=gs.np_float),
        vertices_proxy_idx=np.concatenate(vertices_proxy_idx),
        vertices_local_pos=pos.astype(gs.np_float),
        surface=surface,
    )


@qd.func(requires_top_level=True)
def func_proxies_publish_contact(
    vertex_start: int,
    ipc_state: IPCState,
    proxies_state: IPCProxiesState,
    dyn_state: DynState,
    proxies_info: IPCProxiesInfo,
    proxy_vertices_info: IPCProxyVerticesInfo,
):
    """
    Publish the vertices of the proxies at the start of the substep into their range of the contact surface, from the
    contact vertex `vertex_start` on (see IPCVertices in genesis/engine/solvers/ipc_contact.py).

    The proxy of a fixed link takes the pose of its link, and the others start at their accepted pose. A vertex stands
    at the pose of its proxy at the start, its lagged position too, its proxy being its pivot, without rotation. The
    vertices of a fixed link take no contact terms.
    """
    for i_p, i_b in qd.ndrange(*proxies_state.pos.shape):
        pos = proxies_state.pos[i_p, i_b]
        quat = proxies_state.quat[i_p, i_b]
        if proxies_info.is_fixed[i_p]:
            i_l = proxies_info.link_idx[i_p]
            pos = dyn_state.links.pos[i_l, i_b]
            quat = dyn_state.links.quat[i_l, i_b]
            proxies_state.pos[i_p, i_b] = pos
            proxies_state.quat[i_p, i_b] = quat
        proxies_state.pos_start[i_p, i_b] = pos
        proxies_state.quat_start[i_p, i_b] = quat
    for i_pv, i_b in qd.ndrange(proxy_vertices_info.proxy_idx.shape[0], proxies_state.pos.shape[1]):
        i_p = proxy_vertices_info.proxy_idx[i_pv]
        pos = proxies_state.pos_start[i_p, i_b]
        x = pos + gu.qd_transform_by_quat(proxy_vertices_info.local_pos[i_pv], proxies_state.quat_start[i_p, i_b])
        i_v = vertex_start + i_pv
        ipc_state.vertices.pos[i_v, i_b] = x
        ipc_state.vertices.pos_lagged[i_v, i_b] = x
        ipc_state.vertices.displacement[i_v, i_b] = qd.Vector.zero(gs.qd_float, 3)
        ipc_state.vertices.is_fixed[i_v, i_b] = proxies_info.is_fixed[i_p]
        ipc_state.vertices.row[i_v, i_b] = 0
        ipc_state.vertices.pivot[i_v, i_b] = pos
        ipc_state.vertices.pivot_displacement[i_v, i_b] = qd.Vector.zero(gs.qd_float, 3)
        ipc_state.vertices.rotation[i_v, i_b] = qd.Vector.zero(gs.qd_float, 3)


@qd.kernel(fastcache=True)
def kernel_proxies_init_pose(proxies_state: IPCProxiesState, dyn_state: DynState, proxies_info: IPCProxiesInfo):
    """Set the pose of every proxy to the pose of its link, its constraint vanishing (see genesis/engine/ipc/se3.py)."""
    for i_p, i_b in qd.ndrange(*proxies_state.pos.shape):
        i_l = proxies_info.link_idx[i_p]
        proxies_state.pos[i_p, i_b] = dyn_state.links.pos[i_l, i_b]
        proxies_state.quat[i_p, i_b] = dyn_state.links.quat[i_l, i_b]

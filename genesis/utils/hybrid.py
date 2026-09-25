import hashlib
import os
import platform
import sys
import time
from itertools import combinations

import networkx as nx
import numpy as np
from matplotlib.patches import FancyArrowPatch

import genesis as gs

from .misc import get_gel_cache_dir


def load_hmesh(fpath: str):
    if sys.platform == "linux" and platform.machine() == "aarch64":
        raise ImportError("pygel3d is not supported on Linux ARM (aarch64).")
    from pygel3d import hmesh

    return hmesh.load(fpath)


def trimesh_to_gelmesh(tmesh):
    if sys.platform == "linux" and platform.machine() == "aarch64":
        raise ImportError("pygel3d is not supported on Linux ARM (aarch64).")
    from pygel3d import hmesh

    return hmesh.Manifold.from_triangles(vertices=tmesh.vertices, faces=tmesh.faces)


def get_gel_path(positions, nodes, sampling):
    hasher = hashlib.sha256()
    hasher.update(positions.tobytes())
    hasher.update(nodes.tobytes())
    hasher.update(str(sampling).encode())
    return os.path.join(get_gel_cache_dir(), f"{hasher.hexdigest()}.gel")


def skeletonization(mesh, sampling=True, verbose=False):
    if sys.platform == "linux" and platform.machine() == "aarch64":
        raise ImportError("pygel3d is not supported on Linux ARM (aarch64).")
    from pygel3d import hmesh, graph

    assert isinstance(mesh, hmesh.Manifold), "The input mesh of skeletonization should be pygel3d.hmesh.Manifold"
    g = graph.from_mesh(mesh)
    if verbose:
        tic = time.time()
    gel_file_path = get_gel_path(g.positions(), np.asarray(g.nodes()), sampling=sampling)
    if os.path.exists(gel_file_path):
        gs.logger.debug("Skeleton (`.gel`) found in cache.")
        graph_gel = graph.load(gel_file_path)
    else:
        with gs.logger.timer("Convert mesh to skeleton:"):
            graph_gel = graph.LS_skeleton(g, sampling=sampling)

        os.makedirs(os.path.dirname(gel_file_path), exist_ok=True)
        graph.save(gel_file_path, graph_gel)
    if verbose:
        toc = time.time()
        print(f"Skeletonization time {toc - tic}")

    return graph_gel


def reduce_graph(G, straight_thresh=10):
    assert nx.get_node_attributes(G, "angles") != {}

    # determine which node to be dropped
    nodes_drop = []
    for node in G.nodes():
        degree_two = G.degree(node) == 2

        angles = G.nodes[node]["angles"]
        if len(angles) > 0:
            max_angles = max(angles)
            max_angles_deg = np.rad2deg(max_angles)
            straight = (np.abs(max_angles_deg - 180) < straight_thresh) or (
                np.abs(max_angles_deg - 0) < straight_thresh
            )
        else:
            straight = False

        drop = degree_two and straight
        if drop:
            nodes_drop.append(node)

    # construct reduced graph
    G_reduced = nx.Graph()
    for node in G.nodes():
        if node in nodes_drop:
            continue
        G_reduced.add_node(node)
        G_reduced.nodes[node].update(
            dict(
                pos=G.nodes[node]["pos"],
            )
        )  # pass on node attribute; only pos is valid after reduction

    ref_node = list(G_reduced.nodes())[0]
    for node in G_reduced.nodes():
        if node == ref_node:
            continue
        path = nx.shortest_path(
            G, source=ref_node, target=node
        )  # NOTE: if there is loop, we pick shortest path to the reference node
        node_cur = ref_node
        for node_on_path in path:
            if node_on_path in G_reduced.nodes():
                G_reduced.add_edge(node_cur, node_on_path)
                node_cur = node_on_path

    return G_reduced


def check_graph(G):
    ccs = [v for v in nx.connected_components(G)]
    n_ccs = len(ccs)
    assert n_ccs == 1, f"Invalid graph with more than 1 ({n_ccs}) connected components"


def compute_graph_attribute(G, G_pos):
    # edge attributes
    edge_attrs = dict(
        vec=dict(),
        unit_vec=dict(),
        length=dict(),
    )
    for edge in G.edges():
        n1, n2 = edge  # by convention n1 == node
        vec = G_pos[n2] - G_pos[n1]
        unit_vec = norm_vec(vec)
        length = np.linalg.norm(vec)
        edge_attrs["vec"][edge] = vec
        edge_attrs["unit_vec"][edge] = unit_vec
        edge_attrs["length"][edge] = length

    for name, values in edge_attrs.items():
        nx.set_edge_attributes(G, values=values, name=name)

    # node attributes
    node_attrs = dict(
        pos=dict(),
        angles=dict(),
        edge_pairs_for_angle=dict(),
    )
    for node in G.nodes():
        # node position
        node_attrs["pos"][node] = G_pos[node]

        # node angle
        edges = []
        edge_unit_vecs = []
        for edge in G.edges(node):
            edges.append(edge)
            edge_unit_vecs.append(G.edges[edge]["unit_vec"])

        angles = []
        edge_pairs_for_angle = []
        for i1, i2 in combinations(range(len(edge_unit_vecs)), r=2):
            vec_1 = edge_unit_vecs[i1]
            vec_2 = edge_unit_vecs[i2]
            ang = np.arccos(np.clip(np.dot(vec_1, vec_2), -1.0, 1.0))
            angles.append(ang)
            edge_pairs_for_angle.append((edges[i1], edges[i2]))
        node_attrs["angles"][node] = angles
        node_attrs["edge_pairs_for_angle"][node] = edge_pairs_for_angle

    for name, values in node_attrs.items():
        nx.set_node_attributes(G, values=values, name=name)


def norm_vec(vec):
    vec = np.array(vec)
    return vec / np.linalg.norm(vec)


def gel_graph_to_nx_graph(gel_graph, use_largest_cc=True):
    G = nx.Graph()
    for node in gel_graph.nodes():
        for neighbor in gel_graph.neighbors(node):
            G.add_edge(node, neighbor)

    if use_largest_cc:
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()

    return G


def graph_to_tree(G):
    assert nx.get_node_attributes(G, "pos") != {}

    pos = {n: G.nodes[n]["pos"] for n in G.nodes()}
    src_node = max(pos.items(), key=lambda x: x[1][2])  # x is (k,v)
    src_node = src_node[0]  # get key only

    T = nx.minimum_spanning_tree(G)
    Gout = nx.DiGraph()
    for edge in nx.bfs_edges(T, source=src_node):
        Gout.add_edge(*edge)

        Gout.nodes[edge[0]].update(G.nodes[edge[0]])
        Gout.nodes[edge[1]].update(G.nodes[edge[1]])

    return Gout, src_node


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        # Importing mpl_toolkits is very slow and not used very often. Let's delay import.
        from mpl_toolkits.mplot3d import proj3d

        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

        return np.min(zs)


def plot_nxgraph(
    G,
    pos=None,
    plot_arrow=True,
    use_tick_labels=False,
    show=True,
    figax=None,
    node_color=None,
    node_size=100,
    plot_node_num=True,
):
    # Importing matplotlib is very slow and not used very often. Let's delay import.
    import matplotlib.pyplot as plt

    if pos is None:
        pos = nx.spring_layout(G, dim=3, seed=779)

    node_xyz = np.array([pos[v] for v in sorted(G)])
    edge_xyz = np.array([(pos[u], pos[v]) for u, v in G.edges()])

    # Create the 3D figure
    if figax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig, ax = figax

    # Plot the nodes - alpha is scaled by "depth" automatically
    scatter_kwargs = dict(s=node_size, ec="w")
    if node_color is not None:
        scatter_kwargs["c"] = node_color
    ax.scatter(*node_xyz.T, **scatter_kwargs)

    if plot_node_num:
        for i, node in enumerate(sorted(G)):
            ax.text(*node_xyz[i], f"{i}", color="red")

    # Plot the edges
    for vizedge in edge_xyz:
        ax.plot(*vizedge.T, color="tab:gray")
        if plot_arrow:
            a = Arrow3D(
                [vizedge[0, 0], vizedge[1, 0]],
                [vizedge[0, 1], vizedge[1, 1]],
                [vizedge[0, 2], vizedge[1, 2]],
                mutation_scale=20,
                lw=1.5,
                arrowstyle="-|>",
                color="tab:gray",
            )
            ax.add_artist(a)

    def _format_axes(ax):
        """Visualization options for the 3D axes."""
        # Turn gridlines off
        ax.grid(False)
        # Suppress tick labels
        if not use_tick_labels:
            for dim in (ax.xaxis, ax.yaxis, ax.zaxis):
                dim.set_ticks([])
        ax.axis("equal")
        # Set axes labels
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

    _format_axes(ax)
    if show:
        fig.tight_layout()
        plt.show()

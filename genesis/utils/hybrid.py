import hashlib
import os
from itertools import combinations

import numpy as np

import networkx as nx

import genesis as gs

from .cache import get_gel_cache_dir


def trimesh_to_gelmesh(tmesh):
    from pygel3d import hmesh

    return hmesh.Manifold.from_triangles(vertices=tmesh.vertices, faces=tmesh.faces)


def skeletonization(mesh, sampling=True):
    from pygel3d import hmesh, graph

    assert isinstance(mesh, hmesh.Manifold), "The input mesh of skeletonization should be pygel3d.hmesh.Manifold"
    g = graph.from_mesh(mesh)
    hasher = hashlib.sha256()
    hasher.update(g.positions().tobytes())
    hasher.update(np.asarray(g.nodes()).tobytes())
    hasher.update(str(sampling).encode())
    gel_file_path = os.path.join(get_gel_cache_dir(), f"{hasher.hexdigest()}.gel")
    if os.path.exists(gel_file_path):
        gs.logger.debug("Skeleton (`.gel`) found in cache.")
        graph_gel = graph.load(gel_file_path)
    else:
        with gs.logger.timer("Convert mesh to skeleton:"):
            graph_gel = graph.LS_skeleton(g, sampling=sampling)

        os.makedirs(os.path.dirname(gel_file_path), exist_ok=True)
        graph.save(gel_file_path, graph_gel)

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


def compute_graph_attribute(G, G_pos):
    for n1, n2 in G.edges():
        vec = G_pos[n2] - G_pos[n1]
        G.edges[n1, n2]["unit_vec"] = vec / np.linalg.norm(vec)

    for node in G.nodes():
        G.nodes[node]["pos"] = G_pos[node]
        edge_unit_vecs = [G.edges[edge]["unit_vec"] for edge in G.edges(node)]
        G.nodes[node]["angles"] = [
            np.arccos(np.clip(np.dot(vec_1, vec_2), -1.0, 1.0)) for vec_1, vec_2 in combinations(edge_unit_vecs, r=2)
        ]


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

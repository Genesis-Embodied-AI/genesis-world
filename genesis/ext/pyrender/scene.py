"""The scene graph of the rasterizer."""

import networkx as nx
import numpy as np

from .camera import Camera
from .light import Light
from .mesh import Mesh
from .node import Node
from .utils import format_color_vector


CORNER_INDICES = np.array(
    [
        [0, 1, 2],
        [3, 1, 2],
        [3, 4, 2],
        [0, 4, 2],
        [0, 1, 5],
        [3, 1, 5],
        [3, 4, 5],
        [0, 4, 5],
    ]
)


class Scene:
    """The hierarchy of the nodes that the rasterizer draws, each placed relative to its parent.

    Parameters
    ----------
    bg_color : (4,) float, optional
        The color of the background. Defaults to white.
    ambient_light : (3,) float, optional
        The color of the ambient light. Defaults to none.
    n_envs : int, optional
        The number of environments rendered. Defaults to 1.
    """

    def __init__(self, bg_color=None, ambient_light=None, n_envs=1):
        self.bg_color = np.ones(4) if bg_color is None else format_color_vector(bg_color, 4)
        self.ambient_light = np.zeros(3) if ambient_light is None else format_color_vector(ambient_light, 3)
        self.n_envs = n_envs

        self.nodes = set()
        # Rank of each node in the order it was added, breaking ties when sorting nodes for rendering. Sets iterate in
        # an order that follows object addresses, so without it two nodes at the same distance from the camera get
        # blended in an order that changes from one process to the next.
        self._node_ranks = {}
        self._node_rank_next = 0
        self._revision = 0

        self.mesh_nodes = set()
        self.light_nodes = set()
        self._camera_nodes = set()
        self._main_camera_node = None
        self._bounds = None

        self.has_meshes_changed = False
        """bool : Whether a mesh was added, removed or changed since the renderer last rebuilt its primitives."""
        self.updated_textures = set()
        """set of Texture : The textures whose source changed since the last render, uploaded before drawing."""

        # Transform tree
        self._digraph = nx.DiGraph()
        self._digraph.add_node("world")
        self._path_cache = {}

    @property
    def revision(self):
        """int : Counter of the changes to what the scene draws.

        It advances when a node is added or removed, when a mesh, a light or a node carrying them is moved, and when
        a GPU buffer update is queued. A consumer keeps the revision it last synced to and repeats its preparation
        (context bookkeeping, pose upload, shadow maps) only once the scene moved past it, so the renderers sharing
        a scene do that work once per change.
        """
        return self._revision

    def bump_revision(self):
        """Advance the revision for a change made outside the scene API, such as a GPU buffer update."""
        self._revision += 1

    @property
    def meshes(self):
        """set of Mesh : The meshes of the scene."""
        return {node.mesh for node in self.mesh_nodes}

    @property
    def lights(self):
        """set of Light : The lights of the scene."""
        return {node.light for node in self.light_nodes}

    @property
    def main_camera_node(self):
        """Node : The node of the camera that the scene is rendered from."""
        return self._main_camera_node

    @main_camera_node.setter
    def main_camera_node(self, value):
        assert value in self.nodes
        self._main_camera_node = value

    @property
    def bounds(self):
        """(2, 3) float : The axis-aligned bounds of the scene, a floor counting by its centre and a marker left out."""
        if self._bounds is None:
            corners = []
            for mesh_node in self.mesh_nodes:
                mesh = mesh_node.mesh
                if mesh.is_marker:
                    continue
                if any(primitive.is_floor for primitive in mesh.primitives):
                    corners_local = mesh.centroid[np.newaxis]
                else:
                    corners_local = mesh.bounds.reshape(-1)[CORNER_INDICES]
                pose = self.get_pose(mesh_node)
                corners.append(corners_local @ pose[:3, :3].T + pose[:3, 3])
            if corners:
                corners = np.concatenate(corners, axis=0)
                self._bounds = np.stack((np.min(corners, axis=0), np.max(corners, axis=0)), axis=0)
            else:
                self._bounds = np.zeros((2, 3))
        return self._bounds

    @bounds.setter
    def bounds(self, value):
        # A renderer holding every mesh and pose stacked computes the bounds in one pass and stores them here, which
        # spares recomputing them node by node
        self._bounds = value

    @property
    def centroid(self):
        """(3,) float : The centre of the axis-aligned bounds of the scene."""
        return np.mean(self.bounds, axis=0)

    @property
    def scale(self):
        """float : The length of the diagonal of the axis-aligned bounds of the scene."""
        return max(np.linalg.norm(self.bounds[1] - self.bounds[0]), 1e-7)

    def add(self, obj, pose=None, parent_node=None):
        """Add a mesh, a light or a camera to the scene, in a node of its own.

        Parameters
        ----------
        obj : Mesh or Light or Camera or Node
            The object to add, or the node to add as it is.
        pose : (4, 4) float, optional
            The pose of the node relative to its parent. Defaults to the identity.
        parent_node : Node, optional
            The parent of the node, already in the scene, None for a root node. Defaults to None.

        Returns
        -------
        node : Node
            The node added.
        """
        if isinstance(obj, Node):
            node = obj
        else:
            assert isinstance(obj, (Mesh, Light, Camera))
            node = Node(
                mesh=obj if isinstance(obj, Mesh) else None,
                light=obj if isinstance(obj, Light) else None,
                camera=obj if isinstance(obj, Camera) else None,
                matrix=pose,
            )
        self.add_node(node, parent_node=parent_node)
        return node

    def add_node(self, node, parent_node=None):
        """Add a node and its children to the scene.

        Parameters
        ----------
        node : Node
            The node to add, absent from the scene.
        parent_node : Node, optional
            The parent of the node, already in the scene, None for a root node. Defaults to None.
        """
        assert node not in self.nodes
        assert parent_node is None or parent_node in self.nodes
        self.nodes.add(node)
        self._node_ranks[node] = self._node_rank_next
        self._node_rank_next += 1

        if node.mesh is not None:
            self.has_meshes_changed = True
            self.mesh_nodes.add(node)
        if node.light is not None:
            self.light_nodes.add(node)
        if node.camera is not None:
            self._camera_nodes.add(node)
            if self._main_camera_node is None:
                self._main_camera_node = node

        if parent_node is None:
            parent_node = "world"
        elif node not in parent_node.children:
            parent_node.children.append(node)
        self._digraph.add_node(node)
        self._digraph.add_edge(node, parent_node)

        for child in node.children:
            self.add_node(child, node)

        self._path_cache = {}
        self._bounds = None
        self._revision += 1

    def has_node(self, node):
        """Tell whether a node is in the scene.

        Parameters
        ----------
        node : Node
            The node to look for.
        """
        return node in self.nodes

    def remove_node(self, node):
        """Remove a node and its children from the scene.

        Parameters
        ----------
        node : Node
            The node to remove.
        """
        parent = next(self._digraph.neighbors(node))
        self._remove_node(node)
        if isinstance(parent, Node):
            parent.children.remove(node)
        self._path_cache = {}
        self._bounds = None
        self._revision += 1

    def get_pose(self, node):
        """Return the pose of a node in the world frame.

        Parameters
        ----------
        node : Node
            The node, in the scene.

        Returns
        -------
        pose : (4, 4) float
            The homogeneous transform from the frame of the node to the world frame.
        """
        assert node in self.nodes
        path = self._path_cache.get(node)
        if path is None:
            path = nx.shortest_path(self._digraph, node, "world")
            self._path_cache[node] = path
        pose = path[0].matrix
        for parent_node in path[1:-1]:
            pose = parent_node.matrix @ pose
        return pose

    def set_pose(self, node, pose):
        """Set the pose of a node relative to its parent.

        Parameters
        ----------
        node : Node
            The node, in the scene.
        pose : (4, 4) float
            The homogeneous transform from the frame of the node to that of its parent.
        """
        node.matrix = pose
        if node.mesh is not None:
            self._bounds = None
        # Moving a camera changes the viewpoint only. Moving a mesh or a light changes what every renderer draws.
        if node.mesh is not None or node.light is not None:
            self._revision += 1

    def _remove_node(self, node):
        self.nodes.remove(node)
        del self._node_ranks[node]
        for child in node.children:
            self._remove_node(child)
        self._digraph.remove_node(node)

        if node.mesh is not None:
            self.has_meshes_changed = True
            self.mesh_nodes.remove(node)
        if node.light is not None:
            self.light_nodes.remove(node)
        if node.camera is not None:
            self._camera_nodes.remove(node)
            if self._main_camera_node is node:
                self._main_camera_node = next(iter(self._camera_nodes), None)

    def sorted_mesh_nodes(self, env_idx=-1):
        """Return the nodes to render, in the order they are drawn from the viewpoint of an environment.

        Opaque nodes come first, nearest to farthest so that the depth test rejects what they hide as early as possible.
        Transparent ones follow, farthest to nearest, the only order their blending is correct in. Both are keyed on the
        depth of the centre of the mesh along the view axis, which decides which of two nodes lies behind the other.
        Nodes at equal depth, such as debug arrows sharing an origin, keep the order they were added in, so that
        rendering the same scene twice gives the same image.

        Parameters
        ----------
        env_idx : int, optional
            The environment whose camera pose to sort from, -1 for the first one. Defaults to -1.
        """
        nodes = list(self.mesh_nodes)
        if not nodes:
            return nodes

        cam_pose = self.get_pose(self.main_camera_node)
        if cam_pose.ndim == 3:
            cam_pose = cam_pose[max(env_idx, 0)]

        # A camera looks down its own -Z axis, so depth along the view axis grows away from it
        view_axis = -cam_pose[:3, 2]

        poses = np.stack([self.get_pose(node) for node in nodes])
        centres = np.einsum("nij,nj->ni", poses[:, :3, :3], [node.mesh.centroid for node in nodes]) + poses[:, :3, 3]
        depths = (centres - cam_pose[:3, 3]) @ view_axis
        is_transparent = np.fromiter((node.mesh.is_transparent for node in nodes), bool, len(nodes))
        ranks = np.fromiter((self._node_ranks[node] for node in nodes), int, len(nodes))

        # Sorting the two groups at once keeps the keys computed a single time, the last one being the primary
        order = np.lexsort((ranks, np.where(is_transparent, -depths, depths), is_transparent))
        return [nodes[i] for i in order]

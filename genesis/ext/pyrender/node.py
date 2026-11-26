"""Nodes of the scene graph of the rasterizer."""

import numpy as np


class Node:
    """A node of the scene graph, holding a camera, a mesh or a light at a pose relative to its parent.

    Parameters
    ----------
    camera : Camera, optional
        The camera of the node. Defaults to None.
    mesh : Mesh, optional
        The mesh of the node. Defaults to None.
    light : Light, optional
        The light of the node. Defaults to None.
    matrix : (4, 4) or (n_envs, 4, 4) float, optional
        The homogeneous transform from the frame of the node to that of its parent, one per environment for a node
        placed in each environment. Defaults to the identity.
    children : list of Node, optional
        The children of the node. Defaults to none.
    """

    def __init__(self, camera=None, mesh=None, light=None, matrix=None, children=None):
        self.camera = camera
        self.mesh = mesh
        self.light = light
        self.matrix = np.eye(4) if matrix is None else matrix
        self.children = [] if children is None else children

    @property
    def matrix(self):
        """(4, 4) or (n_envs, 4, 4) float : A copy of the transform from the frame of the node to that of its parent."""
        return self._matrix.copy()

    @matrix.setter
    def matrix(self, value):
        # The poses of the scene are set every frame, so the node takes the array it is given rather than a copy
        self._matrix = value

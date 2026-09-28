"""Utilities for URDF parsing."""

import os
import xml.etree.ElementTree as ET

import trimesh
import numpy as np

import genesis as gs


def rpy_to_matrix(coords):
    """Convert roll-pitch-yaw coordinates to a 3x3 homogenous rotation matrix.

    The roll-pitch-yaw axes in a typical URDF are defined as a
    rotation of ``r`` radians around the x-axis followed by a rotation of
    ``p`` radians around the y-axis followed by a rotation of ``y`` radians
    around the z-axis. These are the Z1-Y2-X3 Tait-Bryan angles. See
    Wikipedia_ for more information.

    .. _Wikipedia: https://en.wikipedia.org/wiki/Euler_angles#Rotation_matrix

    Parameters
    ----------
    coords : (3,) float
        The roll-pitch-yaw coordinates in order (x-rot, y-rot, z-rot).

    Returns
    -------
    R : (3,3) float
        The corresponding homogenous 3x3 rotation matrix.
    """
    coords = np.asanyarray(coords, dtype=np.float64)
    c3, c2, c1 = np.cos(coords)
    s3, s2, s1 = np.sin(coords)

    return np.array(
        [
            [c1 * c2, (c1 * s2 * s3) - (c3 * s1), (s1 * s3) + (c1 * c3 * s2)],
            [c2 * s1, (c1 * c3) + (s1 * s2 * s3), (c3 * s1 * s2) - (c1 * s3)],
            [-s2, c2 * s3, c2 * c3],
        ],
        dtype=np.float64,
    )


def matrix_to_rpy(R, solution=1):
    """Convert a 3x3 transform matrix to roll-pitch-yaw coordinates.

    The roll-pitchRyaw axes in a typical URDF are defined as a
    rotation of ``r`` radians around the x-axis followed by a rotation of
    ``p`` radians around the y-axis followed by a rotation of ``y`` radians
    around the z-axis. These are the Z1-Y2-X3 Tait-Bryan angles. See
    Wikipedia_ for more information.

    .. _Wikipedia: https://en.wikipedia.org/wiki/Euler_angles#Rotation_matrix

    There are typically two possible roll-pitch-yaw coordinates that could have
    created a given rotation matrix. Specify ``solution=1`` for the first one
    and ``solution=2`` for the second one.

    Parameters
    ----------
    R : (3,3) float
        A 3x3 homogenous rotation matrix.
    solution : int
        Either 1 or 2, indicating which solution to return.

    Returns
    -------
    coords : (3,) float
        The roll-pitch-yaw coordinates in order (x-rot, y-rot, z-rot).
    """
    R = np.asanyarray(R, dtype=np.float64)
    r = 0.0
    p = 0.0
    y = 0.0

    if np.abs(R[2, 0]) >= 1.0 - 1e-12:
        y = 0.0
        if R[2, 0] < 0:
            p = np.pi / 2
            r = np.arctan2(R[0, 1], R[0, 2])
        else:
            p = -np.pi / 2
            r = np.arctan2(-R[0, 1], -R[0, 2])
    else:
        if solution == 1:
            p = -np.arcsin(R[2, 0])
        else:
            p = np.pi + np.arcsin(R[2, 0])
        r = np.arctan2(R[2, 1] / np.cos(p), R[2, 2] / np.cos(p))
        y = np.arctan2(R[1, 0] / np.cos(p), R[0, 0] / np.cos(p))

    return np.array([r, p, y], dtype=np.float64)


def matrix_to_xyz_rpy(matrix):
    """Convert a 4x4 homogenous matrix to xyzrpy coordinates.

    Parameters
    ----------
    matrix : (4,4) float
        The homogenous transform matrix.

    Returns
    -------
    xyz_rpy : (6,) float
        The xyz_rpy vector.
    """
    xyz = matrix[:3, 3]
    rpy = matrix_to_rpy(matrix[:3, :3])
    return np.hstack((xyz, rpy))


def xyz_rpy_to_matrix(xyz_rpy):
    """Convert xyz_rpy coordinates to a 4x4 homogenous matrix.

    Parameters
    ----------
    xyz_rpy : (6,) float
        The xyz_rpy vector.

    Returns
    -------
    matrix : (4,4) float
        The homogenous transform matrix.
    """
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, 3] = xyz_rpy[:3]
    matrix[:3, :3] = rpy_to_matrix(xyz_rpy[3:])
    return matrix


def parse_origin(node, *, default):
    """Find the ``origin`` subelement of an XML node and convert it
    into a 4x4 homogenous transformation matrix.

    Parameters
    ----------
    node : :class`lxml.etree.Element`
        An XML node which (optionally) has a child node with the ``origin``
        tag.

    Returns
    -------
    matrix : (4,4) float
        The 4x4 homogneous transform matrix that corresponds to this node's
        ``origin`` child. Defaults to the identity matrix if no ``origin``
        child was found.
    """
    origin_node = node.find("origin")
    if not default and origin_node is None:
        return

    matrix = np.eye(4, dtype=np.float64)
    if origin_node is not None:
        if "xyz" in origin_node.attrib:
            matrix[:3, 3] = np.fromstring(origin_node.attrib["xyz"], sep=" ")
        if "rpy" in origin_node.attrib:
            rpy = np.fromstring(origin_node.attrib["rpy"], sep=" ")
            matrix[:3, :3] = rpy_to_matrix(rpy)
    return matrix


def unparse_origin(matrix):
    """Turn a 4x4 homogenous matrix into an ``origin`` XML node.

    Parameters
    ----------
    matrix : (4,4) float
        The 4x4 homogneous transform matrix to convert into an ``origin``
        XML node.

    Returns
    -------
    node : :class:`xml.etree.ElementTree.Element`
        An XML node whose tag is ``origin``. The node has two attributes:

        - ``xyz`` - A string with three space-delimited floats representing
          the translation of the origin.
        - ``rpy`` - A string with three space-delimited floats representing
          the rotation of the origin.
    """
    node = ET.Element("origin")
    node.attrib["xyz"] = "{} {} {}".format(*matrix[:3, 3])
    node.attrib["rpy"] = "{} {} {}".format(*matrix_to_rpy(matrix[:3, :3]))
    return node


def get_filename(base_path, file_path):
    """Formats a file path correctly for URDF loading.

    Parameters
    ----------
    base_path : str
        The base path to the URDF's folder.
    file_path : str
        The path to the file.

    Returns
    -------
    resolved : str
        The resolved filepath -- just the normal ``file_path`` if it was an
        absolute path, otherwise that path joined to ``base_path``.
    """
    if file_path.startswith("package://"):
        file_path = file_path.replace("package://", "")
        fn = os.path.join(base_path, file_path)
        if not os.path.isfile(fn):
            base_path_ = base_path.split("/")
            file_path_ = file_path.split("/")
            if file_path_[0] in base_path_:
                idx = base_path_.index(file_path_[0])
                fn = "/".join(base_path_[:idx] + file_path_)
    else:
        if not os.path.isabs(file_path):
            fn = os.path.join(base_path, file_path)
        else:
            fn = file_path

    if not os.path.isfile(fn):
        gs.raise_exception(f"Asset file not found: {fn}")

    return fn


def load_meshes(filename):
    """Loads triangular meshes from a file.

    Parameters
    ----------
    filename : str
        Path to the mesh file.

    Returns
    -------
    meshes : list of :class:`~trimesh.base.Trimesh`
        The meshes loaded from the file.
    """
    meshes = trimesh.load(filename, process=False)

    if isinstance(meshes, trimesh.Scene):
        # FIXME: Scene.dump() has bug that uses copy without include_cache=True,
        #  it will lose the vertex normals.
        results = []
        for node_name in meshes.graph.nodes_geometry:
            transform, geometry_name = meshes.graph[node_name]
            current = meshes.geometry[geometry_name].copy(include_cache=True)
            if isinstance(current, trimesh.Trimesh):
                current.apply_transform(transform)
                current.metadata["name"] = geometry_name
                current.metadata["node"] = node_name
                results.append(current)
        meshes = results

    if isinstance(meshes, (list, tuple, set)):
        meshes = list(meshes)
        if len(meshes) == 0:
            raise ValueError("At least one mesh must be present in file")
        for r in meshes:
            if not isinstance(r, trimesh.Trimesh):
                raise TypeError("Could not load meshes from file")
    elif isinstance(meshes, trimesh.Trimesh):
        meshes = [meshes]
    else:
        raise ValueError("Unable to load mesh from file")

    return meshes


def configure_origin(value, *, default):
    """Convert a value into a 4x4 transform matrix.

    Parameters
    ----------
    value : None, (6,) float, or (4,4) float
        The value to turn into the matrix.
        If (6,), interpreted as xyzrpy coordinates.

    Returns
    -------
    matrix : (4,4) float or None
        The created matrix.
    """
    if value is None:
        return np.eye(4, dtype=np.float64) if default else None
    if isinstance(value, (list, tuple, np.ndarray)):
        value = np.asanyarray(value, dtype=np.float64)
        if value.shape == (6,):
            value = xyz_rpy_to_matrix(value)
        elif value.shape != (4, 4):
            raise ValueError("Origin must be specified as a 4x4 homogenous transformation matrix")
    else:
        raise TypeError("Invalid type for origin, expect 4x4 matrix")
    return value

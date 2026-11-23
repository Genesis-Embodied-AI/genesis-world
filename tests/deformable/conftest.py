import xml.etree.ElementTree as ET

import pytest

from ..conftest import lazy_imports

# Imported on first use (see pytest_sessionstart in tests/conftest.py)
with lazy_imports():
    import numpy as np

    import igl


@pytest.fixture(scope="session")
def cloth_grid_obj_factory(asset_tmp_path):
    """Return a function that writes a flat 1 m x 1 m square cloth in the xy plane and returns the path of its file.

    The function takes the number n of squares along each side, each square split into two triangles. The vertex of
    index j * (n + 1) + i lies at x = i / n - 0.5, y = j / n - 0.5.
    """

    def write_cloth_grid(n):
        verts = np.array([[i / n - 0.5, j / n - 0.5, 0.0] for j in range(n + 1) for i in range(n + 1)])
        faces = []
        for j in range(n):
            for i in range(n):
                v00 = j * (n + 1) + i
                faces += [[v00, v00 + 1, v00 + n + 1], [v00 + 1, v00 + n + 2, v00 + n + 1]]
        path = str(asset_tmp_path / f"cloth_grid_{n}.obj")
        igl.write_triangle_mesh(path, verts, np.array(faces))
        return path

    return write_cloth_grid


@pytest.fixture(scope="session")
def cloth_grid_obj_path(cloth_grid_obj_factory):
    """Write a flat 1 m x 1 m square cloth in the xy plane, split into 8 x 8 squares of two triangles each."""
    return cloth_grid_obj_factory(8)


@pytest.fixture(scope="session")
def cloth_folded_obj_path(asset_tmp_path):
    """Write a 0.3 m wide cloth folded in two about the y axis: a lower layer at z = 0.1, 0.3 m long toward -x, a half
    circle of radius 0.03 m, and an upper layer back above the lower one, 24 x 6 squares of two triangles each.

    The vertex of index k along the length of a row lies above the one of index 24 - k for k < 10.
    """
    n_length, n_width, flat_length, radius = 24, 6, 0.3, 0.03
    path = []
    for s in np.linspace(0.0, 2.0 * flat_length + np.pi * radius, n_length + 1):
        if s <= flat_length:
            path.append((s - flat_length, 0.1))
        elif s <= flat_length + np.pi * radius:
            angle = (s - flat_length) / radius - 0.5 * np.pi
            path.append((radius * np.cos(angle), 0.1 + radius + radius * np.sin(angle)))
        else:
            path.append((flat_length + np.pi * radius - s, 0.1 + 2.0 * radius))
    verts = np.array([[x, (j / n_width - 0.5) * 0.3, z] for j in range(n_width + 1) for x, z in path])
    faces = []
    for j in range(n_width):
        for i in range(n_length):
            v00 = j * (n_length + 1) + i
            faces += [[v00, v00 + 1, v00 + n_length + 1], [v00 + 1, v00 + n_length + 2, v00 + n_length + 1]]
    path = str(asset_tmp_path / "cloth_folded.obj")
    igl.write_triangle_mesh(path, verts, np.array(faces))
    return path


@pytest.fixture(scope="session")
def coupled_sliders():
    """Build an MJCF holding a heavy and a light sphere, each on its own vertical slide joint.

    A joint equality ties the heavy slider to twice the displacement of the light one.
    """
    mjcf = ET.Element("mujoco", model="coupled_sliders")
    worldbody = ET.SubElement(mjcf, "worldbody")
    for i_body, mass in enumerate((1.0, 0.01)):
        body = ET.SubElement(worldbody, "body", name=f"slider{i_body}", pos=f"{-10.0 - i_body} 0.0 0.5")
        ET.SubElement(body, "joint", name=f"slide{i_body}", type="slide", axis="0 0 1")
        ET.SubElement(body, "geom", type="sphere", size="0.05", mass=str(mass))
    ET.SubElement(ET.SubElement(mjcf, "equality"), "joint", joint1="slide0", joint2="slide1", polycoef="0 2 0 0 0")
    return mjcf

import xml.etree.ElementTree as ET

import pytest


@pytest.fixture(scope="session")
def free_box():
    """Build an MJCF holding one free box, whose density decides the mass as a primitive's material would.

    A model file rather than a primitive morph, because a primitive is anchored on its own center of mass and the
    solver keeps that frame, which leaves no room to move the center of mass afterwards. Loading this with
    `align=False` keeps the authored frame and allows the write.
    """

    def build(half_size, density):
        mjcf = ET.Element("mujoco", model="free_box")
        worldbody = ET.SubElement(mjcf, "worldbody")
        body = ET.SubElement(worldbody, "body", name="box")
        ET.SubElement(body, "freejoint")
        ET.SubElement(body, "geom", type="box", size=" ".join(map(str, half_size)), density=str(density))
        return ET.tostring(mjcf, encoding="unicode")

    return build

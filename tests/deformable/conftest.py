import xml.etree.ElementTree as ET

import pytest


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

import os
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import trimesh

from genesis.utils.misc import get_assets_dir


@pytest.fixture
def xml_path(request, tmp_path, model_name):
    """The model Mujoco is built from, which holds every subtree, however Genesis groups them into entities."""
    # An asset-relative path passes through to the asset resolver; a bare name is the fixture generating the model.
    if "/" in model_name:
        return model_name
    mjcf = request.getfixturevalue(model_name)
    file_name = f"{model_name}.urdf" if mjcf.tag == "robot" else f"{model_name}.xml"
    file_path = str(tmp_path / file_name)
    ET.ElementTree(mjcf).write(file_path, encoding="utf-8", xml_declaration=True)
    return file_path


@pytest.fixture
def xml_paths(request, tmp_path, xml_path):
    """The models the Genesis scene is assembled from, one entity per model, which is the Mujoco model by default.

    A test marked 'split_entities' gets the model of 'model_name' split into one model per independent subtree, so that
    the Genesis scene holds one entity per subtree while Mujoco holds them all in one model. Every top-level body of
    the worldbody roots a subtree of its own, and the geoms of the worldbody itself root nothing, so they make a model
    of their own. Each part keeps the headers of the source model, its defaults and assets included.

    Splitting changes which entity a body belongs to, and with it how Genesis pairs contacts, so it is asked for by the
    tests that compare what must not depend on the grouping, rather than applied to every model.
    """
    if next(request.node.iter_markers("split_entities"), None) is None:
        return (xml_path,)

    model_name = request.getfixturevalue("model_name")
    mjcf = request.getfixturevalue(model_name)
    parts = []
    worldbody = mjcf.find("worldbody")
    geoms = worldbody.findall("geom")
    for children in [[body] for body in worldbody.findall("body")] + ([geoms] if geoms else []):
        part = ET.Element(mjcf.tag, mjcf.attrib)
        part.extend(header for header in mjcf if header.tag != "worldbody")
        ET.SubElement(part, "worldbody").extend(children)
        parts.append(part)

    paths = []
    for i_part, part in enumerate(parts):
        file_path = str(tmp_path / f"{model_name}_{i_part}.xml")
        ET.ElementTree(part).write(file_path, encoding="utf-8", xml_declaration=True)
        paths.append(file_path)
    return tuple(paths)


def _build_plane_contact_model(model_name, condim, friction, plane_size):
    """Generate the shared skeleton of the plane-contact MJCF models: one contact default applying to every geom and
    a plane floor, with the free bodies appended by _add_free_body."""
    mjcf = ET.Element("mujoco", model=model_name)
    ET.SubElement(mjcf, "option", timestep="0.01")
    default = ET.SubElement(mjcf, "default")
    ET.SubElement(default, "geom", contype="1", conaffinity="1", condim=condim, friction=friction)
    worldbody = ET.SubElement(mjcf, "worldbody")
    ET.SubElement(worldbody, "geom", type="plane", name="floor", pos="0. 0. 0.", size=plane_size)
    return mjcf


def _add_free_body(mjcf, name, geom_type, geom_size, pos, rgba=None):
    """Append a free-floating body with a single geom to a plane-contact MJCF model."""
    body = ET.SubElement(mjcf.find("worldbody"), "body", name=name, pos=pos)
    geom_kwargs = {} if rgba is None else {"rgba": rgba}
    ET.SubElement(body, "geom", type=geom_type, size=geom_size, pos="0. 0. 0.", **geom_kwargs)
    ET.SubElement(body, "joint", name=f"{name}_root", type="free")


@pytest.fixture(scope="session")
def box_plan():
    """Generate an MJCF model for a box on a plane."""
    mjcf = _build_plane_contact_model("box_plan", condim="3", friction="1. 0.5 0.5", plane_size="40. 40. 40.")
    _add_free_body(mjcf, name="box", geom_type="box", geom_size="0.2 0.2 0.2", pos="0. 0. 0.3")
    return mjcf


def _build_free_box_model(model_name, boxes):
    """Generate an MJCF model holding the given free boxes and nothing else, as (name, pos) pairs."""
    mjcf = ET.Element("mujoco", model=model_name)
    ET.SubElement(mjcf, "option", timestep="0.01")
    worldbody = ET.SubElement(mjcf, "worldbody")
    for name, pos in boxes:
        body = ET.SubElement(worldbody, "body", name=name, pos=pos)
        ET.SubElement(body, "geom", type="box", size="0.2 0.2 0.2", pos="0. 0. 0.")
        ET.SubElement(body, "joint", name=f"{name}_root", type="free")
    return mjcf


FREE_BOXES = (("box_left", "-0.5 0. 1."), ("box_right", "0.5 0. 1."))


@pytest.fixture(scope="session")
def two_free_boxes():
    """Generate an MJCF model for two free boxes, which Mujoco holds as one model."""
    return _build_free_box_model("two_free_boxes", FREE_BOXES)


@pytest.fixture(scope="session")
def sphere_plane_roll():
    """Generate an MJCF model for a sphere rolling on a plane, with torsional and rolling friction (condim=6)."""
    mjcf = _build_plane_contact_model(
        "sphere_plane_roll", condim="6", friction="1. 0.005 0.002", plane_size="10. 10. 10."
    )
    _add_free_body(mjcf, name="sphere", geom_type="sphere", geom_size="0.1", pos="0. 0. 0.1")
    return mjcf


@pytest.fixture(scope="session")
def sphere_plane_spin():
    """Generate an MJCF model for a sphere spinning in place on a plane, with torsional friction (condim=4)."""
    mjcf = _build_plane_contact_model("sphere_plane_spin", condim="4", friction="1. 0.005 0.", plane_size="10. 10. 10.")
    _add_free_body(mjcf, name="sphere", geom_type="sphere", geom_size="0.1", pos="0. 0. 0.1")
    return mjcf


@pytest.fixture(scope="session")
def humanoid_ball_floor():
    """The humanoid model with its floor plane swapped for a large sphere. The reference engine resolves plane-capsule
    pairs with a dedicated primitive whose analytic second cap contact the perturbation-based multi-contact cannot
    place exactly, and any flat floor leaves a lying capsule's contact position degenerate along its axis; the
    sphere-capsule pair is a single-contact analytic primitive in both engines and its curvature keeps the closest
    point unique."""
    mjcf = ET.parse(os.path.join(get_assets_dir(), "xml/humanoid.xml")).getroot()
    floor = mjcf.find(".//worldbody/geom[@name='floor']")
    # The sphere top sits a hair above the plane it replaces, turning the model's exact-touch rest pose into a real
    # penetration that both engines detect from the first step. The radius keeps the floor locally flat while holding
    # the contact coordinates near unit magnitude, where single precision still resolves them finely.
    floor.attrib.update(type="sphere", size="5", pos="0 0 -4.9999999")
    return mjcf


@pytest.fixture(scope="session")
def tet_meshball():
    """Generate an MJCF model of three icosphere meshes falling onto a fixed tetrahedron mesh, with both meshes
    embedded as procedural vertex lists so the two engines consume identical geometry."""
    ball = trimesh.creation.icosphere(subdivisions=1, radius=0.1)
    mjcf = ET.Element("mujoco", model="tet_meshball")
    ET.SubElement(mjcf, "option", gravity="0 0 -9.81")
    default = ET.SubElement(mjcf, "default")
    ET.SubElement(default, "joint", damping="0.05")
    ET.SubElement(default, "geom", friction="1 0.005 0.0001", density="500")
    asset = ET.SubElement(mjcf, "asset")
    ET.SubElement(asset, "mesh", name="tet", vertex="1 1 1 -1 1 -1 -1 -1 1 1 -1 -1")
    ET.SubElement(asset, "mesh", name="icosphere", vertex=" ".join(str(c) for v in ball.vertices for c in v))
    worldbody = ET.SubElement(mjcf, "worldbody")
    ET.SubElement(worldbody, "geom", name="tet", type="mesh", mesh="tet")
    # The first ball lands off the tetrahedron's symmetry plane: a centered drop leaves the deepest face an exact
    # tie between two mirror faces, whose resolution is platform rounding in both engines.
    for i, pos in enumerate(("0.02 0 1.2", "0.3 0 1.2", "0.3 0.29 1.2")):
        body = ET.SubElement(worldbody, "body", name=f"ball{i + 1}", pos=pos)
        ET.SubElement(body, "joint", name=f"root{i + 1}", type="free")
        ET.SubElement(body, "geom", name=f"ball{i + 1}_geom", type="mesh", mesh="icosphere")
    return mjcf


@pytest.fixture(scope="session")
def mimic_hinges():
    mjcf = ET.Element("mujoco", model="mimic_hinges")
    ET.SubElement(mjcf, "compiler", angle="degree")
    ET.SubElement(mjcf, "option", timestep="0.01")
    worldbody = ET.SubElement(mjcf, "worldbody")
    parent = ET.SubElement(worldbody, "body", name="parent", pos="0 0 1.0")
    child1 = ET.SubElement(parent, "body", name="child1", pos="0.5 0 0")
    ET.SubElement(child1, "geom", type="capsule", size="0.05 0.2", rgba="0.9 0.1 0.1 1")
    ET.SubElement(child1, "joint", type="hinge", name="joint1", axis="0 1 0", range="-45 45")
    child2 = ET.SubElement(parent, "body", name="child2", pos="0 0.5 0")
    ET.SubElement(child2, "geom", type="capsule", size="0.05 0.2", rgba="0.1 0.1 0.9 1")
    ET.SubElement(child2, "joint", type="hinge", name="joint2", axis="0 1 0", range="-45 45")
    equality = ET.SubElement(mjcf, "equality")
    ET.SubElement(equality, "joint", name="joint_equality", joint1="joint1", joint2="joint2")
    return mjcf


@pytest.fixture(scope="session")
def scaled_mjcf_joint_equalities():
    mjcf = ET.Element("mujoco", model="scaled_mjcf_joint_equalities")
    worldbody = ET.SubElement(mjcf, "worldbody")
    equality = ET.SubElement(mjcf, "equality")
    for driver_type, follower_type in (
        ("hinge", "hinge"),
        ("slide", "slide"),
        ("slide", "hinge"),
        ("hinge", "slide"),
    ):
        pair_name = f"{driver_type}_{follower_type}"
        driver_body = ET.SubElement(worldbody, "body", name=f"{pair_name}_driver_body")
        ET.SubElement(driver_body, "joint", name=f"{pair_name}_driver", type=driver_type, axis="1 0 0")
        ET.SubElement(driver_body, "geom", type="sphere", size="0.05", mass="1", contype="0", conaffinity="0")
        follower_body = ET.SubElement(worldbody, "body", name=f"{pair_name}_follower_body")
        ET.SubElement(follower_body, "joint", name=f"{pair_name}_follower", type=follower_type, axis="1 0 0")
        ET.SubElement(follower_body, "geom", type="sphere", size="0.05", mass="1", contype="0", conaffinity="0")
        ET.SubElement(
            equality,
            "joint",
            name=f"{pair_name}_coupling",
            joint1=f"{pair_name}_follower",
            joint2=f"{pair_name}_driver",
            polycoef="0.2 0.4 -0.3 0.2 -0.1",
        )
    return ET.tostring(mjcf, encoding="unicode")


@pytest.fixture(scope="session")
def single_joint_equality():
    mjcf = ET.Element("mujoco", model="single_joint_equality")
    worldbody = ET.SubElement(mjcf, "worldbody")
    for name in ("target", "unrelated"):
        body = ET.SubElement(worldbody, "body", name=f"{name}_body")
        ET.SubElement(body, "joint", name=name, type="slide", axis="1 0 0")
        ET.SubElement(body, "geom", type="sphere", size="0.05", mass="1", contype="0", conaffinity="0")
    equality = ET.SubElement(mjcf, "equality")
    ET.SubElement(equality, "joint", name="fixed_target", joint1="target", polycoef="0.25 1 0 0 0")
    return ET.tostring(mjcf, encoding="unicode")


@pytest.fixture(scope="session")
def scaled_urdf_mimic():
    robot = ET.Element("robot", name="scaled_urdf_mimic")
    ET.SubElement(robot, "link", name="base")
    joint_type_pairs = (
        ("revolute", "revolute"),
        ("prismatic", "prismatic"),
        ("prismatic", "revolute"),
        ("revolute", "prismatic"),
    )
    for driver_type, follower_type in joint_type_pairs:
        pair_name = f"{driver_type}_{follower_type}"
        for role in ("driver", "follower"):
            link = ET.SubElement(robot, "link", name=f"{pair_name}_{role}_link")
            inertial = ET.SubElement(link, "inertial")
            ET.SubElement(inertial, "mass", value="1.0")
            ET.SubElement(inertial, "inertia", ixx="0.01", ixy="0.0", ixz="0.0", iyy="0.01", iyz="0.0", izz="0.01")

        driver = ET.SubElement(robot, "joint", name=f"{pair_name}_driver_joint", type=driver_type)
        ET.SubElement(driver, "parent", link="base")
        ET.SubElement(driver, "child", link=f"{pair_name}_driver_link")
        ET.SubElement(driver, "axis", xyz="0 0 1")
        ET.SubElement(driver, "limit", lower="-10", upper="10", effort="100", velocity="100")

        follower = ET.SubElement(robot, "joint", name=f"{pair_name}_follower_joint", type=follower_type)
        ET.SubElement(follower, "parent", link=f"{pair_name}_driver_link")
        ET.SubElement(follower, "child", link=f"{pair_name}_follower_link")
        ET.SubElement(follower, "axis", xyz="0 0 1")
        ET.SubElement(follower, "limit", lower="-10", upper="10", effort="100", velocity="100")
        ET.SubElement(follower, "mimic", joint=f"{pair_name}_driver_joint", multiplier="2.0", offset="0.25")
    return ET.tostring(robot, encoding="unicode")


@pytest.fixture(scope="session")
def box_box():
    """Generate an MJCF model for two boxes stacked on a plane."""
    mjcf = _build_plane_contact_model("box_box", condim="3", friction="1. 0.5 0.5", plane_size="40. 40. 40.")
    _add_free_body(mjcf, name="box1", geom_type="box", geom_size="0.2 0.2 0.2", pos="0. 0. 0.2", rgba="0 1 0 0.4")
    _add_free_body(mjcf, name="box2", geom_type="box", geom_size="0.2 0.2 0.2", pos="0. 0. 0.8", rgba="0 0 1 0.4")
    return mjcf


@pytest.fixture
def collision_edge_cases(asset_tmp_path, mode):
    assets = {}
    for i, box_size in enumerate(((0.8, 0.8, 0.04), (0.04, 0.04, 0.005))):
        tmesh = trimesh.creation.box(extents=np.array(box_size) * 2)
        mesh_path = str(asset_tmp_path / f"box{i}.obj")
        tmesh.export(mesh_path, file_type="obj")
        assets[f"box{i}"] = mesh_path

    mjcf = ET.Element("mujoco", model="one_box")
    ET.SubElement(mjcf, "option", timestep="0.005")
    default = ET.SubElement(mjcf, "default")
    ET.SubElement(default, "geom", contype="1", conaffinity="1", condim="3", friction="1. 0.5 0.5")

    asset = ET.SubElement(mjcf, "asset")
    for name, mesh_path in assets.items():
        ET.SubElement(asset, "mesh", name=name, refpos="0 0 0", refquat="1 0 0 0", file=mesh_path)

    worldbody = ET.SubElement(mjcf, "worldbody")

    if mode == 0:
        ET.SubElement(worldbody, "geom", type="box", size="0.8 0.8 0.04", pos="0. 0. 0.", rgba="0 1 0 0.4")
        box1_body = ET.SubElement(worldbody, "body", name="box1", pos="0.0 0.0 0.7")
        ET.SubElement(box1_body, "geom", type="box", size="0.04 0.04 0.005", pos="-0.758 -0.758 0.", rgba="0 0 1 0.4")
    elif mode == 1:
        ET.SubElement(worldbody, "geom", type="box", size="0.8 0.8 0.04", pos="0. 0. 0.", rgba="0 1 0 0.4")
        box1_body = ET.SubElement(worldbody, "body", name="box1", pos="-0.758 -0.758 0.7")
        ET.SubElement(box1_body, "geom", type="box", size="0.04 0.04 0.005", pos="0. 0. 0.", rgba="0 0 1 0.4")
    elif mode == 2:
        ET.SubElement(worldbody, "geom", type="box", size="0.8 0.8 0.04", pos="0. 0. 0.", rgba="0 1 0 0.4")
        box1_body = ET.SubElement(worldbody, "body", name="box1", pos="-0.758 -0.758 1.1")
        ET.SubElement(box1_body, "geom", type="box", size="0.04 0.04 0.005", pos="0. 0. 0.", rgba="0 0 1 0.4")
    elif mode == 3:
        ET.SubElement(worldbody, "geom", type="box", size="0.8 0.8 0.04", pos="0. 0. 0.", rgba="0 1 0 0.4")
        box1_body = ET.SubElement(worldbody, "body", name="box1", pos="0.0 0.0 0.7")
        ET.SubElement(box1_body, "geom", type="mesh", mesh="box1", pos="-0.758 -0.758 0.", rgba="0 0 1 0.4")
    elif mode == 4:
        ET.SubElement(worldbody, "geom", type="mesh", mesh="box0", pos="0. 0. 0.", rgba="0 1 0 0.4")
        box1_body = ET.SubElement(worldbody, "body", name="box1", pos="0.0 0.0 0.7")
        ET.SubElement(box1_body, "geom", type="mesh", mesh="box1", pos="-0.758 -0.758 0.", rgba="0 0 1 0.4")
    elif mode == 5:
        ET.SubElement(worldbody, "geom", type="mesh", mesh="box0", pos="0. 0. 0.", rgba="0 1 0 0.4")
        box1_body = ET.SubElement(worldbody, "body", name="box1", pos="-0.758 -0.758 0.7")
        ET.SubElement(box1_body, "geom", type="mesh", mesh="box1", pos="0. 0. 0.", rgba="0 0 1 0.4")
    elif mode == 6:
        ET.SubElement(worldbody, "geom", type="mesh", mesh="box0", pos="0. 0. 0.", rgba="0 1 0 0.4")
        box1_body = ET.SubElement(worldbody, "body", name="box1", pos="-0.758 -0.758 1.1")
        ET.SubElement(box1_body, "geom", type="mesh", mesh="box1", pos="0. 0. 0.", rgba="0 0 1 0.4")
    elif mode == 7:
        ET.SubElement(worldbody, "geom", type="mesh", mesh="box1", pos=" 0.758  0.758 0.", rgba="0 1 0 0.4")
        ET.SubElement(worldbody, "geom", type="mesh", mesh="box1", pos="-0.758 -0.758 0.", rgba="0 1 0 0.4")
        ET.SubElement(worldbody, "geom", type="mesh", mesh="box1", pos=" 0.758 -0.758 0.", rgba="0 1 0 0.4")
        ET.SubElement(worldbody, "geom", type="mesh", mesh="box1", pos="-0.758  0.758 0.", rgba="0 1 0 0.4")
        box1_body = ET.SubElement(worldbody, "body", name="box1", pos="0. 0. 0.7")
        ET.SubElement(box1_body, "geom", type="mesh", mesh="box0", pos="0. 0. 0.", rgba="0 0 1 0.4")
    elif mode == 8:
        ET.SubElement(worldbody, "geom", type="mesh", mesh="box1", pos=" 0.762  0.762 0.", rgba="0 1 0 0.4")
        ET.SubElement(worldbody, "geom", type="mesh", mesh="box1", pos="-0.762 -0.762 0.", rgba="0 1 0 0.4")
        ET.SubElement(worldbody, "geom", type="mesh", mesh="box1", pos=" 0.762 -0.762 0.", rgba="0 1 0 0.4")
        ET.SubElement(worldbody, "geom", type="mesh", mesh="box1", pos="-0.762  0.762 0.", rgba="0 1 0 0.4")
        box1_body = ET.SubElement(worldbody, "body", name="box1", pos="0. 0. 0.7")
        ET.SubElement(box1_body, "geom", type="mesh", mesh="box0", pos="0. 0. 0.", rgba="0 0 1 0.4")
    else:
        raise ValueError("Invalid mode")

    ET.SubElement(box1_body, "joint", name="root", type="free")

    return mjcf


@pytest.fixture(scope="session")
def decompose_fusion_groups(asset_tmp_path):
    """Generate an MJCF model of a single static link mixing several contact-parameter sub-groups: a plane, a
    nonconvex L-shaped mesh (hull error ~0.3, above the 0.15 threshold) with a small primitive box touching its inner
    corner, two disjoint convex mesh boxes (0.01 gap along x) with a different friction, two disjoint primitive boxes
    with yet another friction, and two mesh boxes with adjacent large collision masks that must never be grouped
    together."""
    lshape = trimesh.util.concatenate(
        [
            trimesh.creation.box(extents=(0.2, 0.1, 0.1)),
            trimesh.creation.box(
                extents=(0.1, 0.1, 0.3), transform=trimesh.transformations.translation_matrix((0.05, 0.0, 0.2))
            ),
        ]
    )
    lshape.export(asset_tmp_path / "lshape.obj")
    trimesh.creation.box(extents=(0.1, 0.1, 0.1)).export(asset_tmp_path / "small_box.obj")

    mjcf = ET.Element("mujoco", model="decompose_fusion_groups")
    asset = ET.SubElement(mjcf, "asset")
    ET.SubElement(asset, "mesh", name="lshape", file=str(asset_tmp_path / "lshape.obj"))
    ET.SubElement(asset, "mesh", name="small_box", file=str(asset_tmp_path / "small_box.obj"))
    worldbody = ET.SubElement(mjcf, "worldbody")
    ET.SubElement(worldbody, "geom", type="plane", size="5 5 0.1")
    ET.SubElement(worldbody, "geom", type="mesh", mesh="lshape", pos="0 0.5 1")
    ET.SubElement(worldbody, "geom", type="box", size="0.01 0.01 0.01", pos="-0.01 0.5 1.06")
    ET.SubElement(worldbody, "geom", type="mesh", mesh="small_box", pos="-0.055 0 1", friction="0.5")
    ET.SubElement(worldbody, "geom", type="mesh", mesh="small_box", pos="0.055 0 1", friction="0.5")
    ET.SubElement(worldbody, "geom", type="box", size="0.02 0.02 0.02", pos="0.3 0 1", friction="0.8")
    ET.SubElement(worldbody, "geom", type="box", size="0.02 0.02 0.02", pos="0.37 0 1", friction="0.8")
    ET.SubElement(worldbody, "geom", type="mesh", mesh="small_box", pos="0 -0.5 1", contype="16777216")
    ET.SubElement(worldbody, "geom", type="mesh", mesh="small_box", pos="0.15 -0.5 1", contype="16777217")
    return mjcf


@pytest.fixture(scope="session")
def mjcf_include_default_and_asset_mesh(asset_tmp_path):
    """Scene MJCF that <include>s a subdirectory model mixing a file-less <default> mesh class with a real <asset>
    mesh, so the include preprocessing must rewrite only the asset mesh path (relative to the included file). Returns
    the scene file path and the authored box extents."""
    extents = (0.2, 0.4, 0.6)
    include_dir = asset_tmp_path / "mjcf_include"
    include_dir.mkdir(exist_ok=True)
    trimesh.creation.box(extents=extents).export(include_dir / "box.obj")

    robot = ET.Element("mujoco", model="robot")
    default = ET.SubElement(robot, "default")
    ET.SubElement(default, "mesh", maxhullvert="64")
    asset = ET.SubElement(robot, "asset")
    ET.SubElement(asset, "mesh", name="box", file="box.obj")
    worldbody = ET.SubElement(robot, "worldbody")
    ET.SubElement(worldbody, "geom", type="mesh", mesh="box")
    ET.ElementTree(robot).write(include_dir / "robot.xml", encoding="utf-8", xml_declaration=True)

    scene_mjcf = ET.Element("mujoco", model="scene")
    ET.SubElement(scene_mjcf, "include", file="mjcf_include/robot.xml")
    scene_path = str(asset_tmp_path / "mjcf_include_scene.xml")
    ET.ElementTree(scene_mjcf).write(scene_path, encoding="utf-8", xml_declaration=True)
    return scene_path, extents


@pytest.fixture(scope="session")
def two_aligned_hinges():
    mjcf = ET.Element("mujoco", model="two_aligned_hinges")
    ET.SubElement(mjcf, "option", timestep="0.05")
    default = ET.SubElement(mjcf, "default")
    ET.SubElement(default, "geom", contype="1", conaffinity="1", condim="3")
    worldbody = ET.SubElement(mjcf, "worldbody")
    link0 = ET.SubElement(worldbody, "body", name="body0")
    ET.SubElement(link0, "geom", type="capsule", fromto="0 0 0 0.5 0 0", size="0.05")
    ET.SubElement(link0, "joint", type="hinge", name="joint0", axis="0 0 1")
    link1 = ET.SubElement(link0, "body", name="body1", pos="0.5 0 0")
    ET.SubElement(link1, "geom", type="capsule", fromto="0 0 0 0.5 0 0", size="0.05")
    ET.SubElement(link1, "joint", type="hinge", name="joint1", axis="0 0 1")
    return mjcf


def _build_chain_capsule_hinge(asset_tmp_path, enable_mesh):
    if enable_mesh:
        mesh_path = str(asset_tmp_path / "capsule.obj")
        tmesh = trimesh.creation.icosphere(radius=1.0, subdivisions=1)
        tmesh.apply_transform(np.diag([0.05, 0.05, 0.25, 1]))
        tmesh.export(mesh_path, file_type="obj")

    mjcf = ET.Element("mujoco", model="two_stick_robot")
    ET.SubElement(mjcf, "option", timestep="0.05")
    default = ET.SubElement(mjcf, "default")
    ET.SubElement(default, "geom", contype="1", conaffinity="1", condim="3")
    if enable_mesh:
        asset = ET.SubElement(mjcf, "asset")
        ET.SubElement(asset, "mesh", name="capsule", refpos="0 0 -0.25", refquat="0.707 0 -0.707 0", file=mesh_path)
    worldbody = ET.SubElement(mjcf, "worldbody")
    link0 = ET.SubElement(worldbody, "body", name="body1", pos="0.1 0.2 0.0", quat="0.707 0 0.707 0")
    if enable_mesh:
        ET.SubElement(link0, "geom", type="mesh", mesh="capsule", rgba="0 0 1 0.3")
    else:
        ET.SubElement(link0, "geom", type="capsule", fromto="0 0 0 0.5 0 0", size="0.05", rgba="0 0 1 0.3")
    link1 = ET.SubElement(link0, "body", name="body2", pos="0.5 0.2 0.0", quat="0.92388 0 0 0.38268")
    if enable_mesh:
        ET.SubElement(link1, "geom", type="mesh", mesh="capsule")
    else:
        ET.SubElement(link1, "geom", type="capsule", fromto="0 0 0 0.5 0 0", size="0.05")
    ET.SubElement(link1, "joint", type="hinge", name="joint1", axis="0 0 1", pos="0.0 0.0 0.0")
    link2 = ET.SubElement(link1, "body", name="body3", pos="0.5 0.2 0.0", quat="0.92388 0 0.38268 0.0")
    if enable_mesh:
        ET.SubElement(link2, "geom", type="mesh", mesh="capsule")
    else:
        ET.SubElement(link2, "geom", type="capsule", fromto="0 0 0 0.5 0 0", size="0.05")
    ET.SubElement(link2, "joint", type="hinge", name="joint2", axis="0 1 0")
    return mjcf


@pytest.fixture(scope="session")
def chain_capsule_hinge_mesh(asset_tmp_path):
    return _build_chain_capsule_hinge(asset_tmp_path, enable_mesh=True)


@pytest.fixture(scope="session")
def chain_capsule_hinge_capsule(asset_tmp_path):
    return _build_chain_capsule_hinge(asset_tmp_path, enable_mesh=False)


def _build_multi_pendulum(n, joint_damping, joint_friction):
    """Generate an URDF model of a multi-link pendulum with n segments."""
    urdf = ET.Element("robot", name="multi_pendulum")

    # Base link
    ET.SubElement(urdf, "link", name="base")

    parent_link = "base"
    for i in range(n):
        # Continuous joint between parent and this arm
        joint = ET.SubElement(urdf, "joint", name=f"PendulumJoint_{i}", type="continuous")
        ET.SubElement(joint, "origin", xyz="0.0 0.0 0.0", rpy="0.0 0.0 0.0")
        ET.SubElement(joint, "axis", xyz="1 0 0")
        ET.SubElement(joint, "parent", link=parent_link)
        ET.SubElement(joint, "child", link=f"PendulumArm_{i}")
        ET.SubElement(joint, "limit", effort=str(100.0 * (n - i)), velocity="30.0")
        dynamics = ET.SubElement(joint, "dynamics")
        if joint_damping is not None:
            dynamics.set("damping", str(joint_damping))
        if joint_friction is not None:
            dynamics.set("friction", str(joint_friction))

        # Arm link
        arm = ET.SubElement(urdf, "link", name=f"PendulumArm_{i}")
        visual = ET.SubElement(arm, "visual")
        ET.SubElement(visual, "origin", xyz="0.0 0.0 0.5", rpy="0.0 0.0 0.0")
        geometry = ET.SubElement(visual, "geometry")
        ET.SubElement(geometry, "box", size="0.01 0.01 1.0")
        material = ET.SubElement(visual, "material", name="")
        ET.SubElement(material, "color", rgba="0.0 0.0 1.0 1.0")
        inertial = ET.SubElement(arm, "inertial")
        ET.SubElement(inertial, "origin", xyz="0.0 0.0 0.0", rpy="0.0 0.0 0.0")
        ET.SubElement(inertial, "mass", value="0.0")
        ET.SubElement(inertial, "inertia", ixx="0.0", ixy="0.0", ixz="0.0", iyy="0.0", iyz="0.0", izz="0.0")

        # Fixed joint to the mass
        joint2 = ET.SubElement(urdf, "joint", name=f"PendulumMassJoint_{i}", type="fixed")
        ET.SubElement(joint2, "origin", xyz="0.0 0.0 1.0", rpy="0.0 0.0 0.0")
        ET.SubElement(joint2, "parent", link=f"PendulumArm_{i}")
        ET.SubElement(joint2, "child", link=f"PendulumMass_{i}")

        # Mass link
        mass = ET.SubElement(urdf, "link", name=f"PendulumMass_{i}")
        visual = ET.SubElement(mass, "visual")
        ET.SubElement(visual, "origin", xyz="0.0 0.0 0.0", rpy="0.0 0.0 0.0")
        geometry = ET.SubElement(visual, "geometry")
        ET.SubElement(geometry, "sphere", radius="0.06")
        material = ET.SubElement(visual, "material", name="")
        ET.SubElement(material, "color", rgba="0.0 0.0 1.0 1.0")
        inertial = ET.SubElement(mass, "inertial")
        ET.SubElement(inertial, "origin", xyz="0.0 0.0 0.0", rpy="0.0 0.0 0.0")
        ET.SubElement(inertial, "mass", value="1.0")
        ET.SubElement(inertial, "inertia", ixx="1e-12", ixy="0.0", ixz="0.0", iyy="1e-12", iyz="0.0", izz="1e-12")

        parent_link = f"PendulumMass_{i}"

    return urdf


@pytest.fixture
def pendulum_with_joint_dynamics(joint_damping, joint_friction):
    return _build_multi_pendulum(n=1, joint_damping=joint_damping, joint_friction=joint_friction)


@pytest.fixture(scope="session")
def pendulum():
    return _build_multi_pendulum(n=1, joint_damping=0.0, joint_friction=0.0)


@pytest.fixture(scope="session")
def double_pendulum():
    return _build_multi_pendulum(n=2, joint_damping=0.0, joint_friction=0.0)


def _add_sphere_link(urdf, link_name, geom_pos, mass=None, inertia=None):
    """Append a link carrying a single sphere geometry offset by 'geom_pos' from the link frame, optionally authoring
    an inertial element whose origin is left omitted."""
    link = ET.SubElement(urdf, "link", name=link_name)
    if mass is not None:
        inertial = ET.SubElement(link, "inertial")
        ET.SubElement(inertial, "mass", value=str(mass))
        ixx, ixy, ixz, iyy, iyz, izz = (str(value) for value in inertia)
        ET.SubElement(inertial, "inertia", ixx=ixx, ixy=ixy, ixz=ixz, iyy=iyy, iyz=iyz, izz=izz)
    for tag in ("visual", "collision"):
        geom_prop = ET.SubElement(link, tag)
        ET.SubElement(geom_prop, "origin", xyz=geom_pos, rpy="0.0 0.0 0.0")
        ET.SubElement(ET.SubElement(geom_prop, "geometry"), "sphere", radius="0.06")


@pytest.fixture
def joint_with_partial_dynamics(joint_damping, joint_friction):
    # Inertia is deliberately left undefined: recomputing it from geometry is the path along which an unset
    # <dynamics> attribute reaches the solver instead of being replaced beforehand.
    urdf = ET.Element("robot", name="joint_with_partial_dynamics")
    _add_sphere_link(urdf, "base_link", "0.0 0.0 0.0")
    _add_sphere_link(urdf, "PendulumArm_0", "0.0 0.0 0.09")
    joint = ET.SubElement(urdf, "joint", name="PendulumJoint_0", type="continuous")
    ET.SubElement(joint, "origin", xyz="0.0 0.0 0.0", rpy="0.0 0.0 0.0")
    ET.SubElement(joint, "axis", xyz="1 0 0")
    ET.SubElement(joint, "parent", link="base_link")
    ET.SubElement(joint, "child", link="PendulumArm_0")
    ET.SubElement(joint, "limit", effort="100.0", velocity="30.0")
    dynamics = ET.SubElement(joint, "dynamics")
    if joint_damping is not None:
        dynamics.set("damping", str(joint_damping))
    if joint_friction is not None:
        dynamics.set("friction", str(joint_friction))
    return urdf


@pytest.fixture(scope="session")
def undefined_inertia():
    """Generate a URDF with a single link that has no inertial element."""
    urdf = ET.Element("robot", name="undefined_inertia")
    _add_sphere_link(urdf, "base_link", "0.0 0.0 0.09")
    return ET.tostring(urdf, encoding="unicode")


@pytest.fixture(scope="session")
def implicit_inertial_origin():
    """Generate a URDF with an authored inertia whose origin is omitted. Its geometry is offset far enough from the
    link frame that the resolved center of mass falls outside the geometry bounding box."""
    urdf = ET.Element("robot", name="implicit_inertial_origin")
    _add_sphere_link(urdf, "base_link", "0.0 0.0 0.09", mass=2.5, inertia=(0.11, 0.01, 0.02, 0.22, 0.03, 0.30))
    return ET.tostring(urdf, encoding="unicode")


@pytest.fixture(scope="session")
def sliding_ball_pair():
    """Build a URDF of two spheres joined by a prismatic joint, each carrying the requested mass and the inertia of a
    sphere of that mass, so a pair authored at one set of masses can be compared against a pair brought to it."""

    def build(base_mass, tip_mass):
        radius = 0.06
        urdf = ET.Element("robot", name="sliding_ball_pair")
        for name, mass in (("base_link", base_mass), ("tip_link", tip_mass)):
            inertia = 0.4 * mass * radius**2
            _add_sphere_link(urdf, name, "0.0 0.0 0.0", mass=mass, inertia=(inertia, 0.0, 0.0, inertia, 0.0, inertia))
        joint = ET.SubElement(urdf, "joint", name="slide", type="prismatic")
        ET.SubElement(joint, "parent", link="base_link")
        ET.SubElement(joint, "child", link="tip_link")
        ET.SubElement(joint, "origin", xyz="0.0 0.0 0.2", rpy="0.0 0.0 0.0")
        ET.SubElement(joint, "axis", xyz="0.0 0.0 1.0")
        ET.SubElement(joint, "limit", lower="-0.1", upper="0.1", effort="100.0", velocity="10.0")
        return ET.tostring(urdf, encoding="unicode")

    return build


@pytest.fixture(scope="session")
def free_bodies_in_one_model():
    """Generate an MJCF holding two free bodies, i.e. one entity of two kinematic trees sharing its mass blocks."""
    mjcf = ET.Element("mujoco")
    worldbody = ET.SubElement(mjcf, "worldbody")
    for i_body, mass in enumerate((1.0, 3.0)):
        body = ET.SubElement(worldbody, "body", name=f"free_{i_body}", pos=f"{i_body} 4.0 0.4")
        ET.SubElement(body, "freejoint")
        ET.SubElement(body, "geom", type="box", size="0.05 0.05 0.05", mass=str(mass))
    return ET.tostring(mjcf, encoding="unicode")


@pytest.fixture(scope="session")
def implicit_inertial_origin_chain():
    """Generate a URDF with two fixed-jointed links, each authoring an inertia whose origin is omitted."""
    urdf = ET.Element("robot", name="implicit_inertial_origin_chain")
    _add_sphere_link(urdf, "base_link", "0.0 0.0 0.0", mass=2.5, inertia=(0.11, 0.01, 0.02, 0.22, 0.03, 0.30))
    _add_sphere_link(urdf, "tip_link", "0.0 0.0 0.0", mass=1.5, inertia=(0.05, 0.0, 0.01, 0.07, 0.0, 0.09))
    joint = ET.SubElement(urdf, "joint", name="weld", type="fixed")
    ET.SubElement(joint, "parent", link="base_link")
    ET.SubElement(joint, "child", link="tip_link")
    ET.SubElement(joint, "origin", xyz="0.0 0.0 0.3", rpy="0.0 0.0 0.0")
    return ET.tostring(urdf, encoding="unicode")


@pytest.fixture(scope="session")
def double_ball_pendulum():
    mjcf = ET.Element("mujoco", model="double_ball_pendulum")

    default = ET.SubElement(mjcf, "default")
    ET.SubElement(default, "joint", armature="0.1", damping="0.5")

    worldbody = ET.SubElement(mjcf, "worldbody")
    base = ET.SubElement(worldbody, "body", name="base", pos="-0.02 0.0 0.0")
    ET.SubElement(base, "joint", name="joint1", type="ball")
    ET.SubElement(
        base, "geom", name="link1_geom", type="capsule", size="0.02", fromto="0 0 0 0 0 0.5", rgba="0.8 0.2 0.2 1.0"
    )
    link2 = ET.SubElement(base, "body", name="link2", pos="0 0 0.5")
    ET.SubElement(link2, "joint", name="joint2", type="ball")
    ET.SubElement(
        link2, "geom", name="link2_geom", type="capsule", size="0.02", fromto="0 0 0 0 0 0.3", rgba="0.2 0.8 0.2 1.0"
    )
    ee = ET.SubElement(link2, "body", name="end_effector", pos="0 0 0.3")
    ET.SubElement(ee, "geom", name="ee_geom", type="sphere", size="0.02", density="200", rgba="1.0 0.8 0.2 1.0")
    ET.SubElement(
        ee,
        "geom",
        name="marker",
        type="sphere",
        contype="0",
        conaffinity="0",
        size="0.01",
        density="0",
        pos="0 -0.02 0",
        rgba="0.0 0.0 0.0 1.0",
    )

    return mjcf


@pytest.fixture(scope="session")
def long_chain():
    # Single kinematic tree with enough DOFs that its mass submatrix exceeds GPU shared memory, so the cooperative
    # >shared-cap mass assemble runs - the path whose lower-triangular linear-index inversion must stay exact on GPUs
    # with an imprecise sqrt.
    mjcf = ET.Element("mujoco", model="long_chain")
    ET.SubElement(mjcf, "compiler", angle="radian")
    worldbody = ET.SubElement(mjcf, "worldbody")
    body = ET.SubElement(worldbody, "body", name="root", pos="0 0 2")
    ET.SubElement(body, "geom", type="sphere", size="0.03", density="500")
    for i in range(128):
        body = ET.SubElement(body, "body", name=f"l{i}", pos="0 0 0.1")
        ET.SubElement(body, "joint", name=f"j{i}", type="hinge", axis=("1 0 0", "0 1 0", "0 0 1")[i % 3], damping="0.1")
        ET.SubElement(body, "geom", type="capsule", fromto="0 0 0 0 0 0.1", size="0.02", density="500")
    return mjcf


@pytest.fixture(scope="session")
def two_fixed_branches():
    # One entity whose worldbody holds two independent chains, each rigidly attached to the (fixed) world. Their DOFs
    # are kinematically decoupled, so the mass matrix is block-diagonal and must partition into one block per branch.
    mjcf = ET.Element("mujoco", model="two_fixed_branches")
    ET.SubElement(mjcf, "compiler", angle="radian")
    worldbody = ET.SubElement(mjcf, "worldbody")
    for name, x in (("a", 0.0), ("b", 1.0)):
        body = ET.SubElement(worldbody, "body", name=f"{name}root", pos=f"{x} 0 1")
        ET.SubElement(body, "geom", type="capsule", fromto="0 0 0 0 0 0.06", size="0.02", density="500")
        for i in range(4):
            body = ET.SubElement(body, "body", name=f"{name}{i}", pos="0 0 0.06")
            ET.SubElement(body, "joint", name=f"j{name}{i}", type="hinge", axis="0 1 0", damping="0.1")
            ET.SubElement(body, "geom", type="capsule", fromto="0 0 0 0 0 0.06", size="0.02", density="500")
    return mjcf


@pytest.fixture(scope="session")
def hinge_slide():
    mjcf = ET.Element("mujoco", model="hinge_slide")

    default = ET.SubElement(mjcf, "default")
    ET.SubElement(default, "joint", damping="0.01")

    worldbody = ET.SubElement(mjcf, "worldbody")
    base = ET.SubElement(worldbody, "body", name="pendulum", pos="0.15 0.0 0.0")
    ET.SubElement(base, "joint", name="hinge", type="hinge", axis="0 1 0", frictionloss="0.08")
    ET.SubElement(base, "geom", name="geom1", type="capsule", size="0.02", fromto="0.0 0.0 0.0 0.1 0.0 0.0")
    link1 = ET.SubElement(base, "body", name="link1", pos="0.1 0.0 0.0")
    ET.SubElement(link1, "joint", name="slide", type="slide", axis="1 0 0", frictionloss="0.3", stiffness="200.0")
    ET.SubElement(link1, "geom", name="geom2", type="capsule", size="0.015", fromto="-0.1 0.0 0.0 0.1 0.0 0.0")

    return mjcf


def ellipsoid_mjcf(semi_axes, body_name="obj", joint_name="root"):
    a, b, c = semi_axes
    mjcf = ET.Element("mujoco", model="ellipsoid")
    worldbody = ET.SubElement(mjcf, "worldbody")
    body = ET.SubElement(worldbody, "body", name=body_name, pos="0 0 0.0")
    ET.SubElement(body, "joint", name=joint_name, type="free")
    ET.SubElement(body, "geom", type="ellipsoid", size=f"{a} {b} {c}")
    return mjcf


@pytest.fixture(scope="session")
def ellipsoid():
    return ellipsoid_mjcf((0.05, 0.05, 0.02))


@pytest.fixture(scope="session")
def general_actuator():
    """Generate an MJCF model with mixed actuator types: PD, general, and non-actuated."""
    mjcf = ET.Element("mujoco", model="general_actuator")
    ET.SubElement(mjcf, "option", timestep="0.01")
    worldbody = ET.SubElement(mjcf, "worldbody")
    body1 = ET.SubElement(worldbody, "body", name="link1", pos="0 0 1")
    ET.SubElement(body1, "joint", name="hinge_pd", type="hinge", axis="0 1 0", damping="0.5")
    ET.SubElement(body1, "geom", type="capsule", size="0.05 0.3", mass="1.0")
    body2 = ET.SubElement(body1, "body", name="link2", pos="0 0 -0.6")
    ET.SubElement(body2, "joint", name="hinge_general", type="hinge", axis="0 1 0", damping="0.3")
    ET.SubElement(body2, "geom", type="capsule", size="0.04 0.2", mass="0.5")
    body3 = ET.SubElement(body2, "body", name="link3", pos="0 0 -0.4")
    ET.SubElement(body3, "joint", name="hinge_motor", type="hinge", axis="0 1 0", damping="0.2")
    ET.SubElement(body3, "geom", type="capsule", size="0.03 0.15", mass="0.3")
    actuator = ET.SubElement(mjcf, "actuator")
    ET.SubElement(actuator, "position", name="act_pd", joint="hinge_pd", kp="100")
    ET.SubElement(
        actuator,
        "general",
        name="act_general",
        joint="hinge_general",
        gainprm="20 0 0",
        biastype="affine",
        biasprm="0.5 -10 -1",
    )
    ET.SubElement(actuator, "motor", name="act_motor", joint="hinge_motor", gear="5")
    return mjcf


@pytest.fixture(scope="session")
def compound_joint():
    mjcf = ET.Element("mujoco", model="compound_joint")
    ET.SubElement(mjcf, "compiler", angle="radian")
    ET.SubElement(mjcf, "option", gravity="0 0 0")
    worldbody = ET.SubElement(mjcf, "worldbody")
    seg1 = ET.SubElement(worldbody, "body", name="seg1", pos="0 0 0")
    ET.SubElement(seg1, "joint", name="j_x", type="hinge", axis="1 0 0")
    ET.SubElement(seg1, "joint", name="j_y", type="hinge", axis="0 1 0")
    ET.SubElement(seg1, "geom", type="capsule", size="0.02", fromto="0 0 0 0 0 0.4")
    seg2 = ET.SubElement(seg1, "body", name="seg2", pos="0 0 0.4")
    ET.SubElement(seg2, "joint", name="j_z", type="hinge", axis="0 0 1")
    ET.SubElement(seg2, "geom", type="capsule", size="0.02", fromto="0 0 0 0 0 0.4")
    return mjcf


@pytest.fixture(scope="session")
def depth_first_tree_mjcf():
    # A kinematic tree where breadth-first and depth-first orderings differ: root A has a child A1, and a sibling root
    # B has none, so depth-first visits A, A1, B (A's subtree contiguous) while breadth-first would give A, B, A1.
    mjcf = ET.Element("mujoco", model="depth_first_tree")
    worldbody = ET.SubElement(mjcf, "worldbody")
    a = ET.SubElement(worldbody, "body", name="A", pos="0 0 1")
    ET.SubElement(a, "freejoint")
    ET.SubElement(a, "geom", type="box", size="0.05 0.05 0.05")
    a1 = ET.SubElement(a, "body", name="A1", pos="0.15 0 0")
    ET.SubElement(a1, "joint", type="hinge", axis="0 0 1")
    ET.SubElement(a1, "geom", type="box", size="0.05 0.05 0.05")
    b = ET.SubElement(worldbody, "body", name="B", pos="1 0 1")
    ET.SubElement(b, "freejoint")
    ET.SubElement(b, "geom", type="box", size="0.05 0.05 0.05")
    return mjcf


@pytest.fixture(scope="session")
def depth_first_tree_urdf():
    # Same shape as depth_first_tree_mjcf but single-rooted (URDF): base -> {A, B}, A -> A1.
    robot = ET.Element("robot", name="depth_first_tree")
    for name in ("base", "A", "A1", "B"):
        link = ET.SubElement(robot, "link", name=name)
        inertial = ET.SubElement(link, "inertial")
        ET.SubElement(inertial, "mass", value="1.0")
        ET.SubElement(inertial, "inertia", ixx="0.01", iyy="0.01", izz="0.01", ixy="0", ixz="0", iyz="0")
        collision = ET.SubElement(link, "collision")
        ET.SubElement(ET.SubElement(collision, "geometry"), "box", size="0.1 0.1 0.1")
    for joint_name, parent, child in (("j_A", "base", "A"), ("j_A1", "A", "A1"), ("j_B", "base", "B")):
        joint = ET.SubElement(robot, "joint", name=joint_name, type="revolute")
        ET.SubElement(joint, "parent", link=parent)
        ET.SubElement(joint, "child", link=child)
        ET.SubElement(joint, "origin", xyz="0 0 0.2")
        ET.SubElement(joint, "axis", xyz="0 0 1")
        ET.SubElement(joint, "limit", lower="-1", upper="1", effort="10", velocity="10")
    return robot


def _build_primitive_pair_mjcf(prim_type, radius, length, offset, name=None):
    """Generate an MJCF model of two primitives attached to a single free body."""
    mjcf = ET.Element("mujoco", model=name or f"{prim_type}_pair")
    body = ET.SubElement(ET.SubElement(mjcf, "worldbody"), "body")
    for sign in (-0.5, 0.5):
        cx, cy, cz = sign * offset[0], sign * offset[1], sign * offset[2]
        if prim_type == "sphere":
            ET.SubElement(
                body,
                "geom",
                type="sphere",
                pos=f"{cx} {cy} {cz}",
                size=str(radius),
            )
        else:
            ET.SubElement(
                body,
                "geom",
                type=prim_type,
                fromto=f"{cx - 0.5 * length} {cy} {cz} {cx + 0.5 * length} {cy} {cz}",
                size=str(radius),
            )
    ET.SubElement(body, "joint", type="free")
    return mjcf


@pytest.fixture(scope="session")
def side_by_side_capsules():
    return _build_primitive_pair_mjcf("capsule", radius=0.0025, length=0.02, offset=(0.0, 0.0025, 0.0))


@pytest.fixture(scope="session")
def collinear_capsules():
    return _build_primitive_pair_mjcf("capsule", radius=0.0025, length=0.02, offset=(0.02, 0.0, 0.0))


@pytest.fixture(scope="session")
def side_by_side_cylinders():
    return _build_primitive_pair_mjcf("cylinder", radius=0.0025, length=0.02, offset=(0.0, 0.0025, 0.0))


@pytest.fixture(scope="session")
def collinear_cylinders():
    return _build_primitive_pair_mjcf("cylinder", radius=0.0025, length=0.02, offset=(0.02, 0.0, 0.0))


@pytest.fixture(scope="session")
def collinear_spheres():
    return _build_primitive_pair_mjcf("sphere", radius=0.0025, length=0.0, offset=(0.005, 0.0, 0.0))


@pytest.fixture(scope="session")
def box_freejoint_offset():
    mjcf = ET.Element("mujoco", model="test_freejoint")
    worldbody = ET.SubElement(mjcf, "worldbody")

    base_body = ET.SubElement(worldbody, "body", name="base", pos="0 0 1.0", quat="1.0 0 0 1.0")
    ET.SubElement(base_body, "freejoint", name="root")
    ET.SubElement(base_body, "inertial", pos="0 0 0", mass="1.0", diaginertia="0.01 0.01 0.01")
    ET.SubElement(base_body, "geom", type="box", size="0.05 0.05 0.05")

    child_body = ET.SubElement(base_body, "body", name="child", pos="0 0 0.1")
    ET.SubElement(child_body, "inertial", pos="0 0 0", mass="0.5", diaginertia="0.001 0.001 0.001")
    ET.SubElement(child_body, "joint", name="joint1", type="hinge", axis="0 1 0")
    ET.SubElement(child_body, "geom", type="box", size="0.03 0.03 0.05")

    return mjcf


@pytest.fixture(scope="session")
def freeflyer_mjcf():
    mjcf = ET.Element("mujoco", model="freeflyer")
    default = ET.SubElement(mjcf, "default")
    default_authored = ET.SubElement(default, "default", {"class": "authored"})
    ET.SubElement(default_authored, "joint", armature="0.0002")
    worldbody = ET.SubElement(mjcf, "worldbody")
    body = ET.SubElement(worldbody, "body", name="base", pos="0 0 1")
    ET.SubElement(body, "joint", type="free")
    ET.SubElement(body, "inertial", pos="0 0 0", mass="1.0", diaginertia="0.01 0.01 0.01")
    ET.SubElement(body, "geom", type="sphere", size="0.05")
    child = ET.SubElement(body, "body", name="child", pos="0 0 0.1")
    ET.SubElement(child, "joint", type="hinge", axis="0 1 0")
    ET.SubElement(child, "inertial", pos="0 0 0", mass="0.5", diaginertia="0.001 0.001 0.001")
    ET.SubElement(child, "geom", type="sphere", size="0.02")
    grandchild = ET.SubElement(child, "body", name="grandchild", pos="0 0 0.1")
    ET.SubElement(grandchild, "joint", type="slide", axis="1 0 0", armature="42.0")
    ET.SubElement(grandchild, "inertial", pos="0 0 0", mass="0.1", diaginertia="0.0001 0.0001 0.0001")
    ET.SubElement(grandchild, "geom", type="sphere", size="0.01")
    greatgrandchild = ET.SubElement(grandchild, "body", name="greatgrandchild", pos="0 0 0.1")
    ET.SubElement(greatgrandchild, "joint", {"type": "slide", "axis": "0 1 0", "class": "authored"})
    ET.SubElement(greatgrandchild, "inertial", pos="0 0 0", mass="0.1", diaginertia="0.0001 0.0001 0.0001")
    ET.SubElement(greatgrandchild, "geom", type="sphere", size="0.01")
    return mjcf


@pytest.fixture(scope="session")
def freeflyer_urdf():
    robot = ET.Element("robot", name="freeflyer")
    ET.SubElement(robot, "link", name="world")
    base_link = ET.SubElement(robot, "link", name="base_link")
    inertial = ET.SubElement(base_link, "inertial")
    ET.SubElement(inertial, "origin", rpy="0 0 0", xyz="0 0 0")
    ET.SubElement(inertial, "mass", value="1.0")
    ET.SubElement(inertial, "inertia", ixx="0.01", ixy="0", ixz="0", iyy="0.01", iyz="0", izz="0.01")
    collision = ET.SubElement(base_link, "collision")
    ET.SubElement(ET.SubElement(collision, "geometry"), "sphere", radius="0.05")
    root_joint = ET.SubElement(robot, "joint", name="root", type="floating")
    ET.SubElement(root_joint, "parent", link="world")
    ET.SubElement(root_joint, "child", link="base_link")
    child_link = ET.SubElement(robot, "link", name="child_link")
    child_inertial = ET.SubElement(child_link, "inertial")
    ET.SubElement(child_inertial, "origin", rpy="0 0 0", xyz="0 0 0")
    ET.SubElement(child_inertial, "mass", value="0.5")
    ET.SubElement(child_inertial, "inertia", ixx="0.001", ixy="0", ixz="0", iyy="0.001", iyz="0", izz="0.001")
    child_collision = ET.SubElement(child_link, "collision")
    ET.SubElement(ET.SubElement(child_collision, "geometry"), "sphere", radius="0.02")
    arm_joint = ET.SubElement(robot, "joint", name="arm", type="revolute")
    ET.SubElement(arm_joint, "parent", link="base_link")
    ET.SubElement(arm_joint, "child", link="child_link")
    ET.SubElement(arm_joint, "origin", rpy="0 0 0", xyz="0 0 0.1")
    ET.SubElement(arm_joint, "axis", xyz="0 1 0")
    ET.SubElement(arm_joint, "limit", lower="-3.14", upper="3.14", effort="10", velocity="10")
    return robot


@pytest.fixture
def xacro_robot(tmp_path):
    """Generate a XACRO file with a two-link chain using macros, properties, overridable args, and a mesh geometry."""
    XACRO_NS = "http://www.ros.org/wiki/xacro"
    ET.register_namespace("xacro", XACRO_NS)

    # Symlink a mesh file into the tmp directory so the xacro can reference it with a relative path
    mesh_src = os.path.join(get_assets_dir(), "meshes", "sphere.obj")
    mesh_dir = tmp_path / "meshes"
    mesh_dir.mkdir()
    (mesh_dir / "sphere.obj").symlink_to(mesh_src)

    robot = ET.Element("robot", name="xacro_chain")

    # Overridable args with defaults
    ET.SubElement(robot, f"{{{XACRO_NS}}}arg", name="link_mass", default="1.0")
    ET.SubElement(robot, f"{{{XACRO_NS}}}arg", name="link_length", default="0.4")

    # Properties derived from args
    ET.SubElement(robot, f"{{{XACRO_NS}}}property", name="mass", value="$(arg link_mass)")
    ET.SubElement(robot, f"{{{XACRO_NS}}}property", name="length", value="$(arg link_length)")
    ET.SubElement(robot, f"{{{XACRO_NS}}}property", name="radius", value="0.05")

    # Macro for a cylindrical link with inertial
    macro = ET.SubElement(robot, f"{{{XACRO_NS}}}macro", name="cyl_link", params="name")
    link = ET.SubElement(macro, "link", name="${name}")
    inertial = ET.SubElement(link, "inertial")
    ET.SubElement(inertial, "mass", value="${mass}")
    ET.SubElement(inertial, "inertia", ixx="0.01", ixy="0", ixz="0", iyy="0.01", iyz="0", izz="0.001")
    visual = ET.SubElement(link, "visual")
    ET.SubElement(ET.SubElement(visual, "geometry"), "cylinder", radius="${radius}", length="${length}")
    collision = ET.SubElement(link, "collision")
    ET.SubElement(ET.SubElement(collision, "geometry"), "cylinder", radius="${radius}", length="${length}")

    # Macro for a mesh link (uses relative path)
    mesh_macro = ET.SubElement(robot, f"{{{XACRO_NS}}}macro", name="mesh_link", params="name")
    mesh_link = ET.SubElement(mesh_macro, "link", name="${name}")
    mesh_inertial = ET.SubElement(mesh_link, "inertial")
    ET.SubElement(mesh_inertial, "mass", value="${mass}")
    ET.SubElement(mesh_inertial, "inertia", ixx="0.01", ixy="0", ixz="0", iyy="0.01", iyz="0", izz="0.001")
    for tag in ("visual", "collision"):
        group = ET.SubElement(mesh_link, tag)
        ET.SubElement(ET.SubElement(group, "geometry"), "mesh", filename="meshes/sphere.obj", scale="0.05 0.05 0.05")

    # Instantiate: two cylinder links + one mesh link
    ET.SubElement(robot, f"{{{XACRO_NS}}}cyl_link", name="base_link")
    ET.SubElement(robot, f"{{{XACRO_NS}}}cyl_link", name="child_link")
    ET.SubElement(robot, f"{{{XACRO_NS}}}mesh_link", name="mesh_link")

    # Revolute joint: base_link -> child_link
    joint = ET.SubElement(robot, "joint", name="joint_0", type="revolute")
    ET.SubElement(joint, "parent", link="base_link")
    ET.SubElement(joint, "child", link="child_link")
    ET.SubElement(joint, "origin", xyz="0 0 ${length}")
    ET.SubElement(joint, "axis", xyz="0 1 0")
    ET.SubElement(joint, "limit", lower="-1.57", upper="1.57", effort="100", velocity="1")

    # Fixed joint: child_link -> mesh_link
    joint2 = ET.SubElement(robot, "joint", name="joint_1", type="fixed")
    ET.SubElement(joint2, "parent", link="child_link")
    ET.SubElement(joint2, "child", link="mesh_link")
    ET.SubElement(joint2, "origin", xyz="0 0 ${length}")

    file_path = str(tmp_path / "two_link.urdf.xacro")
    ET.ElementTree(robot).write(file_path, encoding="utf-8", xml_declaration=True)
    return file_path


@pytest.fixture(scope="session")
def merged_arm_hand_models():
    """MJCF models built from shared fragments so the merged entities are kinematically identical to the single
    equivalent entity by construction: an arm (fixed base -> a1 -> a2 -> tip, hinges about z, links along +x so the
    neutral tip frame is identity) and a branching hand (palm + 4 fingers x 3 hinges = 12 DOFs).

    Returns (monolith, arm_only, arm_free_box_last, arm_free_box_first, hand_only): the monolith splices three hands
    rigidly - one under the tip, one chained under the first hand's palm, and one under a2 (a second branch of the
    same tree) - declared in that depth-first order so its DOF layout matches attaching three hand entities in the
    same creation order, each with an identity offset (child morph pos 0 == coincident with the parent link). The
    arm_free_box variants additionally declare a free body after or before the arm, making it a two-tree entity.
    """

    def _finger(parent, name, y):
        b0 = ET.SubElement(parent, "body", name=f"{name}_0", pos=f"0.05 {y} 0")
        ET.SubElement(b0, "joint", name=f"{name}_j0", type="hinge", axis="0 1 0", pos="0 0 0")
        ET.SubElement(b0, "geom", type="box", size="0.02 0.008 0.008", mass="0.05")
        b1 = ET.SubElement(b0, "body", name=f"{name}_1", pos="0.04 0 0")
        ET.SubElement(b1, "joint", name=f"{name}_j1", type="hinge", axis="0 1 0", pos="0 0 0")
        ET.SubElement(b1, "geom", type="box", size="0.02 0.008 0.008", mass="0.04")
        b2 = ET.SubElement(b1, "body", name=f"{name}_2", pos="0.04 0 0")
        ET.SubElement(b2, "joint", name=f"{name}_j2", type="hinge", axis="0 1 0", pos="0 0 0")
        ET.SubElement(b2, "geom", type="box", size="0.02 0.008 0.008", mass="0.03")

    def _palm(parent, is_root, prefix=""):
        palm = ET.SubElement(parent, "body", name=f"{prefix}palm", pos="0 0 0")
        if is_root:
            ET.SubElement(palm, "freejoint")
        ET.SubElement(palm, "geom", type="box", size="0.03 0.05 0.02", mass="0.2")
        for i, y in enumerate((-0.03, -0.01, 0.01, 0.03)):
            _finger(palm, f"{prefix}f{i}", y)
        return palm

    def _arm_tip(worldbody):
        base = ET.SubElement(worldbody, "body", name="base", pos="0 0 0.5")
        ET.SubElement(base, "geom", type="capsule", fromto="0 0 0 0.2 0 0", size="0.03", mass="1.0")
        a2 = ET.SubElement(base, "body", name="a2", pos="0.2 0 0")
        ET.SubElement(a2, "joint", name="a1", type="hinge", axis="0 0 1", pos="0 0 0")
        ET.SubElement(a2, "geom", type="capsule", fromto="0 0 0 0.2 0 0", size="0.03", mass="1.0")
        tip = ET.SubElement(a2, "body", name="tip", pos="0.2 0 0")
        ET.SubElement(tip, "joint", name="a2", type="hinge", axis="0 0 1", pos="0 0 0")
        ET.SubElement(tip, "geom", type="capsule", fromto="0 0 0 0.02 0 0", size="0.02", mass="0.2")
        return a2, tip

    def _free_box(worldbody):
        box = ET.SubElement(worldbody, "body", name="freebox", pos="0 2 1")
        ET.SubElement(box, "freejoint")
        ET.SubElement(box, "geom", type="box", size="0.05 0.05 0.05", mass="0.2")

    def _arm_model(free_box_position):
        mjcf = ET.Element("mujoco")
        wb = ET.SubElement(mjcf, "worldbody")
        if free_box_position == "first":
            _free_box(wb)
        _arm_tip(wb)
        if free_box_position == "last":
            _free_box(wb)
        return ET.tostring(mjcf, encoding="unicode")

    mono_mjcf = ET.Element("mujoco")
    mono_a2, mono_tip = _arm_tip(ET.SubElement(mono_mjcf, "worldbody"))
    mono_h1_palm = _palm(mono_tip, is_root=False, prefix="h1_")
    _palm(mono_h1_palm, is_root=False, prefix="h3_")
    _palm(mono_a2, is_root=False, prefix="h2_")

    hand_mjcf = ET.Element("mujoco")
    _palm(ET.SubElement(hand_mjcf, "worldbody"), is_root=True)
    return (
        ET.tostring(mono_mjcf, encoding="unicode"),
        _arm_model(free_box_position=None),
        _arm_model(free_box_position="last"),
        _arm_model(free_box_position="first"),
        ET.tostring(hand_mjcf, encoding="unicode"),
    )


@pytest.fixture(scope="session")
def merged_overlapping_models():
    """An arm (fixed base -> a2 -> tip) and a floating-base hand whose palm geom, once attached to the tip, overlaps
    the arm's a2 link (which is NOT adjacent to the palm) in the neutral configuration.

    Returns (arm, hand). Used to check that self-collision / neutral-overlap masking spans the attach merge boundary.
    """
    arm = ET.Element("mujoco")
    wb = ET.SubElement(arm, "worldbody")
    base = ET.SubElement(wb, "body", name="base", pos="0 0 0.5")
    ET.SubElement(base, "geom", type="capsule", fromto="0 0 0 0.2 0 0", size="0.03", mass="1.0")
    a2 = ET.SubElement(base, "body", name="a2", pos="0.2 0 0")
    ET.SubElement(a2, "joint", name="a1", type="hinge", axis="0 0 1")
    ET.SubElement(a2, "geom", type="capsule", fromto="0 0 0 0.2 0 0", size="0.03", mass="1.0")
    tip = ET.SubElement(a2, "body", name="tip", pos="0.2 0 0")
    ET.SubElement(tip, "joint", name="a2", type="hinge", axis="0 0 1")
    ET.SubElement(tip, "geom", type="capsule", fromto="0 0 0 0.02 0 0", size="0.02", mass="0.2")

    hand = ET.Element("mujoco")
    palm = ET.SubElement(ET.SubElement(hand, "worldbody"), "body", name="palm", pos="0 0 0")
    ET.SubElement(palm, "freejoint")
    # Long box reaching back from the tip over the (non-adjacent) a2 link.
    ET.SubElement(palm, "geom", type="box", size="0.15 0.03 0.03", pos="-0.1 0 0", mass="0.2")
    return ET.tostring(arm, encoding="unicode"), ET.tostring(hand, encoding="unicode")


@pytest.fixture(scope="session")
def spring_double_pendulum():
    """Generate an MJCF model for an undamped two-link arm whose hinges are held by stiff springs.

    Its mass matrix depends on the elbow angle, so the arm only conserves energy if the kinetic term is evaluated at
    the current configuration, and the springs store a share of the energy large enough to dominate rounding.
    """
    mjcf = ET.Element("mujoco")
    upper = ET.SubElement(ET.SubElement(mjcf, "worldbody"), "body", name="arm_upper", pos="0.25 0.5 0.8")
    ET.SubElement(upper, "joint", name="shoulder", type="hinge", axis="0 1 0", stiffness="20.0", damping="0")
    ET.SubElement(upper, "geom", type="capsule", fromto="0 0 0 0.2 0 0", size="0.02", density="1000")
    lower = ET.SubElement(upper, "body", name="arm_lower", pos="0.2 0 0")
    ET.SubElement(lower, "joint", name="elbow", type="hinge", axis="0 1 0", stiffness="20.0", damping="0")
    ET.SubElement(lower, "geom", type="capsule", fromto="0 0 0 0.2 0 0", size="0.02", density="1000")
    return ET.tostring(mjcf, encoding="unicode")

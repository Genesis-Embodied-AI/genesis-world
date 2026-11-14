import xml.etree.ElementTree as ET

import numpy as np
import pytest
import torch
import trimesh

import genesis as gs
from genesis.engine.scene import SCENE_FORMAT
from genesis.utils import serialization

from ..utils.assertions import assert_allclose


@pytest.fixture
def two_box_free_mjcf():
    mjcf = ET.Element("mujoco", model="two_box_free")
    worldbody = ET.SubElement(mjcf, "worldbody")
    body = ET.SubElement(worldbody, "body", name="body", pos="0 0 0")
    ET.SubElement(body, "freejoint")
    for x in (-0.2, 0.2):
        ET.SubElement(body, "geom", type="box", size="0.1 0.1 0.1", pos=f"{x} 0 0")
    return ET.tostring(mjcf, encoding="unicode")


@pytest.fixture
def sphere_free_mjcf():
    mjcf = ET.Element("mujoco", model="sphere_free")
    worldbody = ET.SubElement(mjcf, "worldbody")
    body = ET.SubElement(worldbody, "body", name="body", pos="0 0 0")
    ET.SubElement(body, "freejoint")
    ET.SubElement(body, "geom", type="sphere", size="0.1")
    return ET.tostring(mjcf, encoding="unicode")


@pytest.fixture
def torus_box_sphere_mjcf(asset_tmp_path):
    torus_path = asset_tmp_path / "torus.obj"
    trimesh.creation.torus(major_radius=0.1, minor_radius=0.03).export(torus_path)
    mjcf = ET.Element("mujoco", model="torus_box_sphere")
    asset = ET.SubElement(mjcf, "asset")
    ET.SubElement(asset, "mesh", name="torus", file=str(torus_path))
    worldbody = ET.SubElement(mjcf, "worldbody")
    body = ET.SubElement(worldbody, "body", name="body", pos="0 0 0")
    ET.SubElement(body, "freejoint")
    ET.SubElement(body, "geom", name="torus", type="mesh", mesh="torus")
    ET.SubElement(body, "geom", name="top_cap", type="box", size="0.03 0.03 0.03", pos="0 0 0.1")
    ET.SubElement(body, "geom", name="bottom_cap[1]", type="sphere", size="0.03", pos="0 0 -0.1", friction="0.8")
    ET.SubElement(body, "geom", name="ring", type="mesh", mesh="torus", pos="0 0.3 0")
    tip = ET.SubElement(body, "body", name="tip", pos="0 0 0.2")
    ET.SubElement(tip, "joint", type="hinge")
    ET.SubElement(tip, "geom", name="tip", type="capsule", size="0.02 0.03")
    return ET.tostring(mjcf, encoding="unicode")


@pytest.mark.required
def test_entity_options(tmp_path, show_viewer, tol, two_box_free_mjcf, sphere_free_mjcf, torus_box_sphere_mjcf):
    DT = 1e-2
    N_STEPS = 10
    PUSH_FORCE = 50.0

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=DT,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(2.0, -4.0, 2.0),
            camera_lookat=(2.0, 0.0, 0.6),
        ),
        show_viewer=show_viewer,
    )
    # The root link holds no collision geom, so the geom of 'arm2' is the second one of the entity
    arm_options = gs.entities.Rigid(
        morph=gs.morphs.URDF(
            file="urdf/simple/two_link_arm.urdf",
            pos=(0.0, -1.0, 0.6),
            fixed=True,
            recompute_inertia=True,
        ),
        material=gs.materials.Rigid(
            rho=500.0,
            friction=0.5,
        ),
    )
    arm_options.select(gs.indices.Geom.at(1)).set_friction(1.5)
    arm_options.select(gs.indices.Link.subtree("arm1")).set_friction(0.9).set_friction_rolling(0.003)
    arm_options.select(gs.indices.Link.subtree("arm1") & gs.indices.Link.named("arm2")).set(
        gs.options.Geom(rho=2000.0, friction=0.8)
    )
    arm_options.select(gs.indices.Link.named("arm1")).set_friction(0.6)
    # Nesting a link selection in another keeps it at the link level, so the later selection of the subtree applies
    arm_options.select(gs.indices.Link.subtree("arm1")).select(gs.indices.Link.named("arm2")).set_friction_torsional(
        0.05
    )
    arm_options.select(gs.indices.Link.subtree("base")).set_friction_torsional(0.02)
    arm = scene.add_entity(options=arm_options)
    spheres = []
    for i, friction in enumerate((0.5, 1.5)):
        sphere_options = gs.entities.Rigid(
            morph=gs.morphs.Mesh(
                file="meshes/sphere.obj",
                pos=(1.0 + i, -1.0, 0.6),
                scale=0.1,
                fixed=True,
            ),
        )
        sphere_options.select(gs.indices.Geom.at(slice(None))).set_friction(friction)
        spheres.append(scene.add_entity(options=sphere_options))
    # The torus is decomposed into convex pieces
    mixed_geoms_options = gs.entities.Rigid(
        morph=gs.morphs.MJCF(
            file=torus_box_sphere_mjcf,
            pos=(4.0, 1.0, 0.6),
            decompose_robot_error_threshold=0.0,
        ),
        material=gs.materials.Rigid(
            friction=0.3,
            friction_torsional=0.01,
            friction_rolling=0.0005,
        ),
    )
    caps = gs.indices.Geom.named("top_cap") | gs.indices.Geom.named("bottom_cap[[]1]")
    mixed_geoms_options.select(caps).set(gs.options.Geom(friction_torsional=0.05, friction_rolling=0.002))
    # A nested ordinal counts the geoms of the selection alone, so this selects the second cap
    mixed_geoms_options.select(caps).select(gs.indices.Geom.at(1)).set_friction_rolling(0.004)
    # An ordinal under links counts the geoms of each link apart, so this selects the torus and the capsule
    mixed_geoms_options.select(gs.indices.Link.subtree("body")).select(gs.indices.Geom.at(0)).set_friction(1.2)
    with mixed_geoms_options.select(gs.indices.Link.named("body")) as body:
        body.select(gs.indices.Geom.named("tor*")).set_friction(0.7)
        body.select(gs.indices.Geom.at(1)).set_friction(1.5)
        body.select(gs.indices.Geom.at(slice(1, 3))).set_friction_torsional(0.03)
    # A density given as None stands for the default density on geoms that still fuse with each other
    mixed_geoms_options.select(gs.indices.Link.named("body")).set(gs.options.Geom(rho=None))
    # A field given as None stands for the value of the asset, overriding the material
    mixed_geoms_options.select(gs.indices.Geom.named("bottom_cap[[]1]")).set(gs.options.Geom(friction=None))
    # The fifth collision geom of the entity is the capsule of the second link
    mixed_geoms_options.select(gs.indices.Geom.at(4)).set_friction_rolling(0.001)
    mixed_geoms_body = scene.add_entity(options=mixed_geoms_options)
    # Geom densities weigh in the mass of an aligned body alone, whose frame stays where its kinematic ghost puts it
    dense_body_options = gs.entities.Rigid(
        morph=gs.morphs.MJCF(
            file=two_box_free_mjcf,
            pos=(3.0, 1.0, 0.6),
            align=True,
            recompute_inertia=True,
        ),
    )
    with dense_body_options.select(gs.indices.Link.named("body")) as body:
        body.select(gs.indices.Geom.at(0)).set_rho(8000.0)
        body.select(gs.indices.Geom.at(1)).set_rho(1000.0)
    dense_body = scene.add_entity(options=dense_body_options)
    # A density set on a body whose asset gives its mass changes nothing, so the body stays aligned
    stated_mass_body_options = gs.entities.Rigid(
        morph=gs.morphs.MJCF(
            file=two_box_free_mjcf,
            pos=(3.0, 2.0, 0.6),
            align=True,
        ),
    )
    stated_mass_body_options.select(gs.indices.Geom.at(0)).set_rho(8000.0)
    stated_mass_body = scene.add_entity(options=stated_mass_body_options)
    dense_body_ghost = scene.add_entity(
        options=gs.entities.Kinematic(
            morph=gs.morphs.MJCF(
                file=two_box_free_mjcf,
                pos=(3.0, 1.0, 0.6),
                align=True,
            ),
        ),
    )
    box_options = gs.entities.Rigid(
        morph=gs.morphs.Box(
            pos=(5.0, 0.0, 0.0),
            size=(0.1, 0.1, 0.1),
        ),
        material=gs.materials.Rigid(
            rho=500.0,
        ),
    )
    box_options.select(gs.indices.Geom.at(0)).set_friction(2.0).end().select(gs.indices.Geom.at([0])).set_friction(1.0)
    # A density given as None stands for that of the asset, and the box states none, so the default density applies
    box_options.select(gs.indices.Geom.at(0)).set(gs.options.Geom(rho=None))
    box = scene.add_entity(options=box_options)
    # A fixed link estimates its inertia once a moving parent carries it, with the density its geoms resolved to
    fixed_box_options = gs.entities.Rigid(
        morph=gs.morphs.Box(
            pos=(5.0, 0.0, 0.2),
            size=(0.1, 0.1, 0.1),
            fixed=True,
        ),
        material=gs.materials.Rigid(
            rho=500.0,
        ),
    )
    fixed_box_options.select(gs.indices.Geom.at(0)).set(gs.options.Geom(rho=None))
    fixed_box = scene.add_entity(options=fixed_box_options)
    fixed_box.attach(box, box.base_link.name, pos=(0.0, 0.0, 0.2))
    box_options.select(gs.indices.Geom.at(0)).set_friction(3.0)
    with pytest.raises(gs.GenesisException, match="frozen"):
        box.options.select(gs.indices.Geom.at(0)).set_friction(3.0)

    exported = tmp_path / f"selections{SCENE_FORMAT}"
    rigid_options = gs.entities.Rigid(
        morph=gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
        ),
    )
    rigid_options.select(gs.indices.Link.subtree("box_*")).set_friction(1.0)
    with rigid_options.select(gs.indices.Link.named("box_baselink")) as box_link:
        box_link.select(gs.indices.Geom.at([0])).set_rho(1000.0)
        box_link.select(gs.indices.Geom.at(slice(0, 1))).set_friction_rolling(0.001)
    rigid_options.select(gs.indices.Geom.named("box*") | gs.indices.Geom.named("*")).set_friction(2.0)
    fem_options = gs.entities.FEM(
        morph=gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
        ),
        material=gs.materials.FEM.Elastic(),
    )
    corners = gs.indices.Vertex.at(0) | gs.indices.Vertex.closest_to(pos=(0.05, 0.05, 0.05), max_distance=0.01)
    fem_options.select(corners).set_is_fixed(True)
    fem_options.select(gs.indices.Vertex.at(slice(0, 4)) & (gs.indices.Vertex.at(0) | gs.indices.Vertex.at(1))).set(
        gs.options.Vertex(is_fixed=False)
    )
    fem_options.select(gs.indices.Vertex.in_box(lower=(0.0, 0.0, 0.0), upper=(0.1, 0.1, 0.1))).select(
        gs.indices.Vertex.at(0)
    ).set_is_fixed(True)
    serialization.export(exported, {"rigid": rigid_options, "fem": fem_options})
    loaded = serialization.load(exported, {"rigid": gs.entities.Rigid, "fem": gs.entities.FEM})
    assert loaded["rigid"].model_dump() == rigid_options.model_dump()
    assert loaded["fem"].model_dump() == fem_options.model_dump()

    with pytest.raises(gs.GenesisException, match="gs.entities.FEM"):
        gs.entities.MPM(morph=gs.morphs.Sphere(), material=gs.materials.FEM.Elastic())
    with pytest.raises(gs.GenesisException, match="gs.entities.Rigid"):
        gs.entities.Kinematic(morph=gs.morphs.Sphere(), material=gs.materials.Rigid())

    # A subclass a user declares admits the materials of its base, which the shorthand keeps giving to the base
    class UserRigid(gs.entities.Rigid):
        pass

    with pytest.raises(gs.GenesisException, match="gs.entities.Kinematic"):
        UserRigid(morph=gs.morphs.Sphere(), material=gs.materials.Kinematic())
    assert type(gs.Scene().add_entity(gs.morphs.Sphere()).options) is gs.entities.Rigid
    with pytest.raises(gs.GenesisException, match="either entity options"):
        scene.add_entity(morph=gs.morphs.Sphere(), options=gs.entities.Rigid(morph=gs.morphs.Sphere()))
    with pytest.raises(gs.GenesisException, match="got gs.indices.Vertex"):
        gs.entities.Rigid(morph=gs.morphs.Box(size=(0.1, 0.1, 0.1))).select(gs.indices.Vertex.at(0)).set_friction(1.0)
    with pytest.raises(gs.GenesisException, match="got gs.indices.Link"):
        box_options.select(gs.indices.Geom.at(0)).select(gs.indices.Link.named("box_baselink"))
    with pytest.raises(gs.GenesisException, match="frozen"):
        box.options.name = "renamed_box"
    with pytest.raises(gs.GenesisException, match="frozen"):
        box_options.selection_blocks = ()
    # A copy records again, deep or not
    box.options.model_copy(deep=True).select(gs.indices.Geom.at(0)).set_friction(3.0)
    with pytest.raises(gs.GenesisException, match="at least 1"):
        gs.entities.Rigid(morph=())
    with pytest.raises(gs.GenesisException, match="sets no quantity"):
        box_options.select(gs.indices.Geom.at(0)).set(gs.options.Geom())
    out_of_range_options = box_options.model_copy()
    out_of_range_options.select(gs.indices.Link.named("box_baselink")).select(gs.indices.Geom.at(1)).set_friction(1.0)
    with pytest.raises(gs.GenesisException, match="selects no valid geom among 1"):
        scene.add_entity(options=out_of_range_options)
    disjoint_options = arm.options.model_copy()
    disjoint_options.select(gs.indices.Link.named("arm1") & gs.indices.Link.named("arm2")).set_friction(1.0)
    with pytest.raises(gs.GenesisException, match="holds no collision geom"):
        scene.add_entity(options=disjoint_options)
    unknown_link_options = arm.options.model_copy()
    unknown_link_options.select(gs.indices.Link.named("arm3")).set_friction(1.0)
    with pytest.raises(gs.GenesisException, match="links_to_keep"):
        scene.add_entity(options=unknown_link_options)
    rootless_options = arm.options.model_copy()
    rootless_options.select(gs.indices.Link.named("base")).select(gs.indices.Geom.at(0)).set_friction(1.0)
    with pytest.raises(gs.GenesisException, match="holds no collision geom"):
        scene.add_entity(options=rootless_options)
    # Every heterogeneous variant holds what the entity selects
    variants_options = gs.entities.Rigid(
        morph=(gs.morphs.MJCF(file=two_box_free_mjcf), gs.morphs.MJCF(file=sphere_free_mjcf))
    )
    variants_options.select(gs.indices.Geom.at(1)).set_friction(1.0)
    with pytest.raises(gs.GenesisException, match="selects no valid geom among 1"):
        scene.add_entity(options=variants_options)
    # Only rigid entities draw contacts, which every kind may decline
    gs.Scene().add_entity(gs.morphs.Sphere(), material=gs.materials.Kinematic(), visualize_contact=False)
    with pytest.raises(gs.GenesisException, match="only applies to rigid entities"):
        gs.Scene().add_entity(gs.morphs.Sphere(), material=gs.materials.FEM.Elastic(), visualize_contact=True)

    scene.build(n_envs=2)

    # The mass of each box is its volume, 0.2^3 m^3, times the density set on its geom
    assert_allclose(dense_body.get_mass(), (8000.0 + 1000.0) * 0.2**3, tol=tol)
    assert stated_mass_body.base_link.aligned
    assert_allclose(stated_mass_body.get_mass(), 2 * 1000.0 * 0.2**3, tol=tol)
    assert_allclose(dense_body.get_qpos(), dense_body_ghost.get_qpos(), tol=tol)
    # A force on the translation of the body acts at its frame, off its center of mass, so it spins the body
    dense_body.control_dofs_force([0.0, 0.0, PUSH_FORCE, 0.0, 0.0, 0.0])
    for _ in range(N_STEPS):
        scene.step()

    # The mass of each link is the volume of its box, 1 x 0.1 x 0.1 m^3, times the density set on it
    assert_allclose(arm.get_link("arm1").get_mass(), 500.0 * 0.01, tol=tol)
    assert_allclose(arm.get_link("arm2").get_mass(), 2000.0 * 0.01, tol=tol)
    assert_allclose(arm.get_link("arm1").geoms[0].get_friction(), 0.6, tol=gs.EPS)
    assert_allclose(arm.get_link("arm2").geoms[0].get_friction(), 1.5, tol=gs.EPS)
    assert_allclose(box.geoms[0].get_friction(), 1.0, tol=gs.EPS)
    # The default density of an object, 600 kg/m^3, times the volume of the box, 0.1^3 m^3
    assert_allclose(box.get_mass(), 600.0 * 0.1**3, tol=tol)
    assert_allclose(fixed_box.get_mass(), 600.0 * 0.1**3, tol=tol)
    for geom in arm.geoms:
        assert_allclose(geom.get_friction_torsional(), 0.02, tol=gs.EPS)
        assert_allclose(geom.get_friction_rolling(), 0.003, tol=gs.EPS)
    for sphere, friction in zip(spheres, (0.5, 1.5)):
        for geom in sphere.geoms:
            assert_allclose(geom.get_friction(), friction, tol=gs.EPS)
    # The frame of the body sits at the centroid of its two boxes, 0.2 m on either side
    boxes_mass, boxes_x = np.array([8000.0, 1000.0]) * 0.2**3, np.array([-0.2, 0.2])
    com_x = boxes_mass @ boxes_x / boxes_mass.sum()
    inertia_yy = boxes_mass @ ((boxes_x - com_x) ** 2 + 2.0 * 0.2**2 / 12.0)
    assert_allclose(dense_body.get_dofs_velocity()[..., 4], com_x * PUSH_FORCE / inertia_yy * N_STEPS * DT, tol=5e-2)
    # The ring, given no options, keeps the friction of the entity, its pieces fusing with none of the torus
    meshes_friction = torch.stack(
        [geom.get_friction() for geom in mixed_geoms_body.geoms if geom.type == gs.GEOM_TYPE.MESH]
    )
    assert len(meshes_friction) > 2
    assert_allclose(meshes_friction.unique(), (0.3, 0.7), tol=gs.EPS)
    geoms_friction = {
        gs.GEOM_TYPE.MESH: (None, 0.01, 0.0005),
        gs.GEOM_TYPE.BOX: (1.5, 0.03, 0.002),
        gs.GEOM_TYPE.SPHERE: (0.8, 0.03, 0.004),
        gs.GEOM_TYPE.CAPSULE: (1.2, 0.01, 0.001),
    }
    for geom in mixed_geoms_body.geoms:
        friction, friction_torsional, friction_rolling = geoms_friction[geom.type]
        if friction is not None:
            assert_allclose(geom.get_friction(), friction, tol=gs.EPS)
        assert_allclose(geom.get_friction_torsional(), friction_torsional, tol=gs.EPS)
        assert_allclose(geom.get_friction_rolling(), friction_rolling, tol=gs.EPS)

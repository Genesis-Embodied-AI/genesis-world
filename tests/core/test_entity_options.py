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
    ET.SubElement(body, "geom", name="bottom_cap", type="sphere", size="0.03", pos="0 0 -0.1")
    ET.SubElement(body, "geom", name="ring", type="mesh", mesh="torus", pos="0 0.3 0")
    tip = ET.SubElement(body, "body", name="tip", pos="0 0 0.2")
    ET.SubElement(tip, "joint", type="hinge")
    ET.SubElement(tip, "geom", name="tip", type="capsule", size="0.02 0.03")
    return ET.tostring(mjcf, encoding="unicode")


@pytest.mark.required
def test_reject_material_for_solver():
    scene = gs.Scene()
    # The material of another solver is refused, naming the entity options admitting it
    with pytest.raises(gs.GenesisException, match="gs.entities.FEM"):
        gs.entities.MPM(morph=gs.morphs.Sphere(), material=gs.materials.FEM.Elastic())
    with pytest.raises(gs.GenesisException, match="gs.entities.Rigid"):
        gs.entities.Kinematic(morph=gs.morphs.Sphere(), material=gs.materials.Rigid())

    # A subclass a user declares admits the materials of its base, which the shorthand keeps giving to the base
    class UserRigid(gs.entities.Rigid):
        pass

    with pytest.raises(gs.GenesisException, match="gs.entities.Kinematic"):
        UserRigid(morph=gs.morphs.Sphere(), material=gs.materials.Kinematic())
    assert type(scene.add_entity(gs.morphs.Sphere()).options) is gs.entities.Rigid
    with pytest.raises(gs.GenesisException, match="either entity options"):
        scene.add_entity(morph=gs.morphs.Sphere(), options=gs.entities.Rigid(morph=gs.morphs.Sphere()))


@pytest.mark.required
def test_selection_precedence(tmp_path, caplog, show_viewer, tol, two_box_free_mjcf, torus_box_sphere_mjcf):
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
    # Link and geom options override the material of the entity on the geoms they select, the geom options winning
    arm = scene.add_entity(
        options=gs.entities.Rigid(
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
            links={
                # An ordinal selects in every link of the subtree holding collision geoms, the root holding none
                gs.LinkIndex.subtree("base"): gs.options.Link(
                    geoms={
                        gs.GeomIndex.at(0): gs.options.Geom(
                            friction_torsional=0.02,
                        ),
                    },
                ),
                gs.LinkIndex.subtree("arm1"): gs.options.Link(
                    friction_rolling=0.003,
                ),
                gs.LinkIndex.named("arm2"): gs.options.Link(
                    rho=2000.0,
                    friction=0.8,
                    geoms={
                        gs.GeomIndex.at(0): gs.options.Geom(
                            friction=1.5,
                        ),
                    },
                ),
            },
        ),
    )
    # Entities sharing one collision mesh keep the geom options each of them gives
    spheres = [
        scene.add_entity(
            options=gs.entities.Rigid(
                morph=gs.morphs.Mesh(
                    file="meshes/sphere.obj",
                    pos=(1.0 + i, -1.0, 0.6),
                    scale=0.1,
                    fixed=True,
                ),
                geoms={
                    gs.GeomIndex.named("*"): gs.options.Geom(
                        friction=friction,
                    ),
                },
            ),
        )
        for i, friction in enumerate((0.5, 1.5))
    ]
    # The convex pieces of a decomposed mesh all take the options of the mesh
    mixed_geoms_body = scene.add_entity(
        options=gs.entities.Rigid(
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
            links={
                # The geom name selects in every link of the subtree, of which the parent has none of that name
                gs.LinkIndex.subtree("body"): gs.options.Link(
                    geoms={
                        gs.GeomIndex.named("tip"): gs.options.Geom(
                            friction=1.2,
                        ),
                    },
                ),
                gs.LinkIndex.named("body"): gs.options.Link(
                    geoms={
                        gs.GeomIndex.named("tor*"): gs.options.Geom(
                            friction=0.7,
                        ),
                        gs.GeomIndex.at([False, True, False, False]): gs.options.Geom(
                            friction=1.5,
                        ),
                        gs.GeomIndex.at(slice(1, 3)): gs.options.Geom(
                            friction_torsional=0.03,
                        ),
                    },
                ),
            },
            # An ordinal of the entity counts its collision geoms over every link, so the fifth one is the capsule
            geoms={
                gs.GeomIndex.named("*_cap"): gs.options.Geom(
                    friction_torsional=0.05,
                    friction_rolling=0.002,
                ),
                gs.GeomIndex.at(4): gs.options.Geom(
                    friction_rolling=0.001,
                ),
            },
        ),
    )
    # Geom densities weigh in the mass of an aligned body alone, whose frame stays where its kinematic ghost puts it
    dense_body = scene.add_entity(
        options=gs.entities.Rigid(
            morph=gs.morphs.MJCF(
                file=two_box_free_mjcf,
                pos=(3.0, 1.0, 0.6),
                align=True,
                recompute_inertia=True,
            ),
            links={
                gs.LinkIndex.named("body"): gs.options.Link(
                    geoms={
                        gs.GeomIndex.at(0): gs.options.Geom(
                            rho=8000.0,
                        ),
                        gs.GeomIndex.at(1): gs.options.Geom(
                            rho=1000.0,
                        ),
                    },
                ),
            },
        ),
    )
    dense_body_ghost = scene.add_entity(
        options=gs.entities.Kinematic(
            morph=gs.morphs.MJCF(
                file=two_box_free_mjcf,
                pos=(3.0, 1.0, 0.6),
                align=True,
            ),
        ),
    )

    # Every kind of selection travels through a scene file
    exported = tmp_path / f"selections{SCENE_FORMAT}"
    rigid_options = gs.entities.Rigid(
        morph=gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
        ),
        links={
            gs.LinkIndex.subtree("box_*"): gs.options.Link(),
            gs.LinkIndex.named("box_baselink"): gs.options.Link(
                geoms={
                    gs.GeomIndex.at([0]): gs.options.Geom(),
                    gs.GeomIndex.at([True]): gs.options.Geom(),
                    gs.GeomIndex.at(slice(0, 1)): gs.options.Geom(),
                    gs.GeomIndex.named("box*"): gs.options.Geom(),
                },
            ),
        },
        geoms={
            gs.GeomIndex.named("*"): gs.options.Geom(),
        },
    )
    fem_options = gs.entities.FEM(
        morph=gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
        ),
        material=gs.materials.FEM.Elastic(),
        vertices={
            gs.VertexIndex.at(0) | gs.VertexIndex.closest_to(pos=(0.05, 0.05, 0.05), max_distance=0.01): (
                gs.options.Vertex()
            ),
            gs.VertexIndex.in_box(lower=(0.0, 0.0, 0.0), upper=(0.1, 0.1, 0.1)): gs.options.Vertex(),
        },
    )
    serialization.export(exported, {"rigid": rigid_options, "fem": fem_options})
    loaded = serialization.load(exported, {"rigid": gs.entities.Rigid, "fem": gs.entities.FEM})
    assert loaded["rigid"].model_dump() == rigid_options.model_dump()
    assert loaded["fem"].model_dump() == fem_options.model_dump()

    with caplog.at_level("WARNING"):
        overlapping_box = scene.add_entity(
            options=gs.entities.Rigid(
                morph=gs.morphs.Box(
                    pos=(5.0, 0.0, 0.0),
                    size=(0.1, 0.1, 0.1),
                ),
                links={
                    gs.LinkIndex.named("box_baselink"): gs.options.Link(
                        geoms={
                            gs.GeomIndex.at(0): gs.options.Geom(
                                friction=2.0,
                            ),
                            gs.GeomIndex.at([True]): gs.options.Geom(
                                friction=1.0,
                            ),
                        },
                    ),
                },
            )
        )
    assert "equal specificity give 'friction'" in caplog.text
    with pytest.raises(gs.GenesisException, match="selects no valid geom"):
        scene.add_entity(
            options=gs.entities.Rigid(
                morph=gs.morphs.Box(pos=(5.0, 0.0, 0.0), size=(0.1, 0.1, 0.1)),
                links={
                    gs.LinkIndex.named("box_baselink"): gs.options.Link(
                        geoms={gs.GeomIndex.at(1): gs.options.Geom(friction=1.0)}
                    )
                },
            )
        )
    with pytest.raises(gs.GenesisException, match="No collision geom of the entity"):
        scene.add_entity(
            options=arm.options.model_copy(update={"geoms": {gs.GeomIndex.named("leg*"): gs.options.Geom()}})
        )
    with pytest.raises(gs.GenesisException, match="links_to_keep"):
        scene.add_entity(
            options=arm.options.model_copy(update={"links": {gs.LinkIndex.named("arm3"): gs.options.Link()}})
        )
    with pytest.raises(gs.GenesisException, match="selects no valid geom among 2"):
        scene.add_entity(
            options=arm.options.model_copy(update={"geoms": {gs.GeomIndex.at(2): gs.options.Geom(friction=1.0)}})
        )

    scene.build(n_envs=2)
    with pytest.raises(gs.GenesisException, match="read-only"):
        arm.options.morph.pos = (9.0, 9.0, 9.0)

    # The mass of each box is its volume, 0.2^3 m^3, times the density its geom options give
    assert_allclose(dense_body.get_mass(), (8000.0 + 1000.0) * 0.2**3, tol=tol)
    assert_allclose(dense_body.get_qpos(), dense_body_ghost.get_qpos(), tol=tol)
    # A force on the translation of the body acts at its frame, off its center of mass, so it spins the body
    dense_body.control_dofs_force([0.0, 0.0, PUSH_FORCE, 0.0, 0.0, 0.0])
    for _ in range(N_STEPS):
        scene.step()

    # The mass of each link is the volume of its box, 1 x 0.1 x 0.1 m^3, times the density its options resolve to
    assert_allclose(arm.get_link("arm1").get_mass(), 500.0 * 0.01, tol=tol)
    assert_allclose(arm.get_link("arm2").get_mass(), 2000.0 * 0.01, tol=tol)
    assert_allclose(arm.get_link("arm1").geoms[0].get_friction(), 0.5, tol=gs.EPS)
    assert_allclose(arm.get_link("arm2").geoms[0].get_friction(), 1.5, tol=gs.EPS)
    assert_allclose(overlapping_box.geoms[0].get_friction(), 2.0, tol=gs.EPS)
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
        gs.GEOM_TYPE.SPHERE: (0.3, 0.03, 0.002),
        gs.GEOM_TYPE.CAPSULE: (1.2, 0.01, 0.001),
    }
    for geom in mixed_geoms_body.geoms:
        friction, friction_torsional, friction_rolling = geoms_friction[geom.type]
        if friction is not None:
            assert_allclose(geom.get_friction(), friction, tol=gs.EPS)
        assert_allclose(geom.get_friction_torsional(), friction_torsional, tol=gs.EPS)
        assert_allclose(geom.get_friction_rolling(), friction_rolling, tol=gs.EPS)

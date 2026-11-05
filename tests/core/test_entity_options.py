import xml.etree.ElementTree as ET

import numpy as np
import pytest
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
    ET.SubElement(asset, "material", name="cap_paint", rgba="0.5 0.5 0.5 1")
    worldbody = ET.SubElement(mjcf, "worldbody")
    body = ET.SubElement(worldbody, "body", name="body", pos="0 0 0")
    ET.SubElement(body, "freejoint")
    ET.SubElement(body, "geom", name="torus", type="mesh", mesh="torus")
    ET.SubElement(body, "geom", name="top_cap", type="box", size="0.03 0.03 0.03", pos="0 0 0.1", material="cap_paint")
    ET.SubElement(body, "geom", name="bottom_cap", type="sphere", size="0.03", pos="0 0 -0.1")
    tip = ET.SubElement(body, "body", name="tip", pos="0 0 0.2")
    ET.SubElement(tip, "joint", type="hinge")
    ET.SubElement(tip, "geom", name="tip", type="capsule", size="0.02 0.03")
    return ET.tostring(mjcf, encoding="unicode")


@pytest.mark.required
def test_options_class_selects_solver(show_viewer, tol):
    DT = 1e-2
    N_STEPS = 10
    GRAVITY = 9.81

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=DT,
            substeps=10,
            gravity=(0.0, 0.0, -GRAVITY),
        ),
        fem_options=gs.options.FEMOptions(
            use_implicit_solver=True,
        ),
        mpm_options=gs.options.MPMOptions(
            grid_density=16,
            particle_size=0.05,
            lower_bound=(-0.5, -0.5, 0.0),
            upper_bound=(2.0, 1.5, 1.0),
        ),
        sph_options=gs.options.SPHOptions(
            particle_size=0.05,
        ),
        pbd_options=gs.options.PBDOptions(
            particle_size=0.05,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.75, -1.5, 1.5),
            camera_lookat=(0.75, 0.55, 0.5),
        ),
        show_viewer=show_viewer,
    )
    options_classes = [
        cls
        for name, cls in vars(gs.entities).items()
        if isinstance(cls, type)
        and issubclass(cls, gs.entities.BaseEntity)
        and cls is not gs.entities.BaseEntity
        and name.isidentifier()
        and not name.startswith("_")
    ]
    entities = []
    for i, options_cls in enumerate(options_classes):
        pos = (0.15 + 0.4 * (i % 4), 0.15 + 0.4 * (i // 4), 0.6)
        options_kwargs = {
            "morph": gs.morphs.Box(
                pos=pos,
                size=(0.2, 0.2, 0.2),
            ),
        }
        if issubclass(options_cls, gs.entities.PBDCloth):
            options_kwargs.update(
                morph=gs.morphs.Mesh(
                    file="meshes/cloth.obj",
                    pos=pos,
                    scale=0.2,
                ),
            )
        if issubclass(options_cls, gs.entities.FEM):
            options_kwargs.update(
                material=gs.materials.Elastic(),
            )
        if issubclass(options_cls, gs.entities.MPM):
            options_kwargs.update(
                material=gs.materials.Elastic(
                    youngs_modulus=3e5,
                ),
                solver_options=gs.options.MPMEntitySolverOptions(
                    sampler="regular",
                ),
            )
        if issubclass(options_cls, gs.entities.SPH):
            options_kwargs.update(
                solver_options=gs.options.SPHEntitySolverOptions(
                    sampler="regular",
                ),
            )
        if issubclass(options_cls, gs.entities.PBDLiquid):
            options_kwargs.update(
                solver_options=gs.options.PBDLiquidEntitySolverOptions(
                    sampler="regular",
                ),
            )
        if issubclass(options_cls, gs.entities.PBDParticle):
            options_kwargs.update(
                solver_options=gs.options.PBDParticleEntitySolverOptions(
                    sampler="regular",
                ),
            )
        if issubclass(options_cls, gs.entities.Hybrid):
            options_kwargs.update(
                morph=gs.morphs.URDF(
                    file="urdf/simple/two_link_arm.urdf",
                    pos=pos,
                    scale=0.2,
                ),
                material_soft=gs.materials.Muscle(),
            )
        entities.append(scene.add_entity(options=options_cls(**options_kwargs)))

    # A material parameter the solver of the entity cannot honor is refused rather than ignored, and the shorthand for
    # rigid entities names the classes admitting a deformable material
    with pytest.raises(gs.GenesisException, match="youngs_modulus"):
        gs.entities.PBDElastic(morph=gs.morphs.Sphere(), material=gs.materials.Elastic(youngs_modulus=1e6))
    with pytest.raises(gs.GenesisException, match="gs.entities.FEM or gs.entities.MPM"):
        scene.add_entity(morph=gs.morphs.Sphere(), material=gs.materials.Elastic())
    with pytest.raises(gs.GenesisException, match="SPH or gs.entities.PBDLiquid or gs.entities.PBDParticle"):
        scene.add_entity(morph=gs.morphs.Sphere(), material=gs.materials.Liquid())
    with pytest.raises(gs.GenesisException, match="either entity options"):
        scene.add_entity(morph=gs.morphs.Sphere(), options=gs.entities.Rigid(morph=gs.morphs.Sphere()))

    # Entities ignoring gravity tell the kinematic solver apart from the others
    scene.build(n_envs=2)
    measured_entities = [entity for entity in entities if not isinstance(entity, gs.engine.entities.HybridEntity)]
    # The drop is measured from the height before stepping, since the mean of the vertices of a tetrahedralized box
    # starts off its center by an amount that depends on the mesh.
    entities_z = []
    for n_steps in (0, N_STEPS):
        for _ in range(n_steps):
            scene.step()
        entities_z.append([])
        for entity in measured_entities:
            if isinstance(entity, gs.engine.entities.KinematicEntity):
                pos = entity.get_pos()
            elif isinstance(entity, gs.engine.entities.particle_entity.ParticleEntity):
                pos = entity.get_particles_pos().mean(dim=-2)
            elif isinstance(entity, gs.engine.entities.FEMEntity):
                pos = entity.get_state().pos.mean(dim=-2)
            else:
                pos = entity.get_state().pos
            entities_z[-1].append(pos[..., 2])
    for entity, z_start, z_end in zip(measured_entities, *entities_z):
        drop = z_start - z_end
        if isinstance(entity.options, gs.entities.Kinematic):
            assert_allclose(drop, 0.0, tol=tol)
        else:
            # The mean of the particles or vertices stands in for the center of mass, so it falls at gravity approximately
            assert_allclose(drop, 0.5 * GRAVITY * (N_STEPS * DT) ** 2, rtol=5e-2)


@pytest.mark.required
def test_selection_precedence(tmp_path, show_viewer, tol, two_box_free_mjcf, torus_box_sphere_mjcf):
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
            material=gs.materials.Material(
                density=500.0,
                friction=0.5,
            ),
            links={
                gs.LinkIndex.subtree("arm1"): gs.options.Link(
                    material=gs.materials.Material(
                        friction_rolling=0.003,
                    ),
                ),
                "arm2": gs.options.Link(
                    material=gs.materials.Material(
                        density=2000.0,
                        friction=0.8,
                    ),
                    geoms={
                        gs.GeomIndex.at(0): gs.options.Geom(
                            material=gs.materials.Material(
                                friction=1.5,
                            ),
                        ),
                    },
                ),
            },
            geoms={
                gs.GeomIndex.all(): gs.options.Geom(
                    material=gs.materials.Material(
                        friction_torsional=0.02,
                    ),
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
                    gs.GeomIndex.all(): gs.options.Geom(
                        material=gs.materials.Material(
                            friction=friction,
                        ),
                    ),
                },
            ),
        )
        for i, friction in enumerate((0.5, 1.5))
    ]
    # A geom selection counts and names the collision geoms of a link as the asset states them, so the convex pieces of
    # a decomposed mesh all take the options of the mesh. A name, a mask, a slice and, for the entity, a glob each
    # select their geoms.
    mixed_geoms_body = scene.add_entity(
        options=gs.entities.Rigid(
            morph=gs.morphs.MJCF(
                file=torus_box_sphere_mjcf,
                pos=(4.0, 1.0, 0.6),
                decompose_robot_error_threshold=0.0,
            ),
            material=gs.materials.Material(
                friction=0.3,
                friction_torsional=0.01,
                friction_rolling=0.0005,
            ),
            links={
                # The geom name selects in every link of the subtree, of which the parent has none of that name
                gs.LinkIndex.subtree("body"): gs.options.Link(
                    geoms={
                        gs.GeomIndex.named("tip"): gs.options.Geom(
                            material=gs.materials.Material(
                                friction=1.2,
                            ),
                        ),
                    },
                ),
                "body": gs.options.Link(
                    surface=gs.surfaces.Default(
                        color=(1.0, 0.0, 0.0),
                    ),
                    vgeoms={
                        gs.VGeomIndex.with_material("cap_*"): gs.options.VGeom(
                            surface=gs.surfaces.Default(
                                color=(0.0, 1.0, 0.0),
                            ),
                        ),
                    },
                    geoms={
                        gs.GeomIndex.named("tor*"): gs.options.Geom(
                            material=gs.materials.Material(
                                friction=0.7,
                            ),
                        ),
                        gs.GeomIndex.at([False, True, False]): gs.options.Geom(
                            material=gs.materials.Material(
                                friction=1.5,
                            ),
                        ),
                        gs.GeomIndex.at(slice(1, 3)): gs.options.Geom(
                            material=gs.materials.Material(
                                friction_torsional=0.03,
                            ),
                        ),
                    },
                ),
            },
            geoms={
                gs.GeomIndex.named("*_cap"): gs.options.Geom(
                    material=gs.materials.Material(
                        friction_rolling=0.002,
                    ),
                ),
            },
            vgeoms={
                gs.VGeomIndex.named("tip"): gs.options.VGeom(
                    surface=gs.surfaces.Default(
                        color=(0.0, 0.0, 1.0),
                    ),
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
                "body": gs.options.Link(
                    geoms={
                        gs.GeomIndex.at(0): gs.options.Geom(
                            material=gs.materials.Material(
                                density=8000.0,
                            ),
                        ),
                        gs.GeomIndex.at(1): gs.options.Geom(
                            material=gs.materials.Material(
                                density=1000.0,
                            ),
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
                surface=gs.surfaces.Default(
                    color=(1.0, 0.0, 0.0),
                ),
                vgeoms={
                    gs.VGeomIndex.with_material("lcd*"): gs.options.VGeom(
                        surface=gs.surfaces.Emission(
                            emissive=(0.0, 1.0, 0.0),
                        ),
                    ),
                },
            ),
        },
        geoms={
            gs.GeomIndex.all(): gs.options.Geom(),
        },
        vgeoms={
            gs.VGeomIndex.at(0): gs.options.VGeom(),
            gs.VGeomIndex.named("box*"): gs.options.VGeom(),
        },
    )
    fem_options = gs.entities.FEM(
        morph=gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
        ),
        material=gs.materials.Elastic(),
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

    with pytest.raises(gs.GenesisException, match="equal specificity"):
        scene.add_entity(
            options=gs.entities.Rigid(
                morph=gs.morphs.Box(pos=(5.0, 0.0, 0.0), size=(0.1, 0.1, 0.1)),
                geoms={
                    gs.GeomIndex.all(): gs.options.Geom(material=gs.materials.Material(friction=1.0)),
                    gs.GeomIndex.at(0): gs.options.Geom(material=gs.materials.Material(friction=2.0)),
                },
            )
        )
    with pytest.raises(gs.GenesisException, match="selects no valid geom"):
        scene.add_entity(
            options=gs.entities.Rigid(
                morph=gs.morphs.Box(pos=(5.0, 0.0, 0.0), size=(0.1, 0.1, 0.1)),
                geoms={gs.GeomIndex.at(1): gs.options.Geom(material=gs.materials.Material(friction=1.0))},
            )
        )
    with pytest.raises(gs.GenesisException, match="equal specificity"):
        scene.add_entity(
            options=arm.options.model_copy(
                update={
                    "links": {
                        "arm*": gs.options.Link(material=gs.materials.Material(friction=1.0)),
                        "arm2": gs.options.Link(material=gs.materials.Material(friction=2.0)),
                    }
                }
            )
        )
    with pytest.raises(gs.GenesisException, match="No collision geom of the entity"):
        scene.add_entity(
            options=arm.options.model_copy(
                update={"geoms": {gs.GeomIndex.named("leg*"): gs.options.Geom(material=gs.materials.Material())}}
            )
        )
    with pytest.raises(gs.GenesisException, match="equal specificity"):
        scene.add_entity(
            options=arm.options.model_copy(
                update={
                    "links": {
                        "arm*": gs.options.Link(surface=gs.surfaces.Default()),
                        "arm2": gs.options.Link(surface=gs.surfaces.Rough()),
                    }
                }
            )
        )
    with pytest.raises(gs.GenesisException, match="No visual geom of the entity has a material"):
        scene.add_entity(
            options=mixed_geoms_body.options.model_copy(
                update={"vgeoms": {gs.VGeomIndex.with_material("lcd"): gs.options.VGeom()}}
            )
        )
    with pytest.raises(gs.GenesisException, match="links_to_keep"):
        scene.add_entity(
            options=arm.options.model_copy(update={"links": {gs.LinkIndex.named("arm3"): gs.options.Link()}})
        )
    for update in (
        {"geoms": {gs.GeomIndex.at(0): gs.options.Geom(material=gs.materials.Material(friction=1.0))}},
        {"vgeoms": {gs.VGeomIndex.at(0): gs.options.VGeom(surface=gs.surfaces.Default())}},
    ):
        with pytest.raises(gs.GenesisException, match="mesh or a primitive"):
            scene.add_entity(options=arm.options.model_copy(update=update))
    with pytest.raises(gs.GenesisException, match="two_way_soft_constraint"):
        gs.entities.Rigid(
            morph=gs.morphs.Sphere(),
            solver_options=gs.options.RigidEntitySolverOptions(coup_type="ipc_only"),
            links={"base": gs.options.Link(solver_options=gs.options.RigidLinkSolverOptions(needs_coup=False))},
        )
    with pytest.raises(gs.GenesisException, match="coupled link"):
        gs.entities.Hybrid(
            morph=gs.morphs.URDF(file="urdf/simple/two_link_arm.urdf"),
            material_soft=gs.materials.Muscle(),
            solver_options=gs.options.HybridEntitySolverOptions(
                rigid=gs.options.RigidEntitySolverOptions(coup_type="two_way_soft_constraint")
            ),
        )

    scene.build(n_envs=2)
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
    # The subtree under 'arm1' holds 'arm2' as well
    for geom in arm.geoms:
        assert_allclose(geom.get_friction_torsional(), 0.02, tol=gs.EPS)
        assert_allclose(geom.get_friction_rolling(), 0.003, tol=gs.EPS)
    for sphere, friction in zip(spheres, (0.5, 1.5)):
        for geom in sphere.geoms:
            assert_allclose(geom.get_friction(), friction, tol=gs.EPS)
    # The frame of the body sits at the centroid of its two boxes, 0.2 m on either side, and its center of mass towards
    # the denser one
    boxes_mass, boxes_x = np.array([8000.0, 1000.0]) * 0.2**3, np.array([-0.2, 0.2])
    com_x = boxes_mass @ boxes_x / boxes_mass.sum()
    inertia_yy = boxes_mass @ ((boxes_x - com_x) ** 2 + 2.0 * 0.2**2 / 12.0)
    assert_allclose(dense_body.get_dofs_velocity()[..., 4], com_x * PUSH_FORCE / inertia_yy * N_STEPS * DT, tol=5e-2)
    # The pieces of the decomposed torus take the options of its name, the box those of the mask and of the slice, and
    # the sphere those of the slice over the friction of the entity, both caps taking the glob of the entity and the
    # capsule of the child link the name its subtree selection gives
    torus_pieces = [geom for geom in mixed_geoms_body.geoms if geom.type == gs.GEOM_TYPE.MESH]
    assert len(torus_pieces) > 1
    geoms_friction = {
        gs.GEOM_TYPE.MESH: (0.7, 0.01, 0.0005),
        gs.GEOM_TYPE.BOX: (1.5, 0.03, 0.002),
        gs.GEOM_TYPE.SPHERE: (0.3, 0.03, 0.002),
        gs.GEOM_TYPE.CAPSULE: (1.2, 0.01, 0.0005),
    }
    for geom in mixed_geoms_body.geoms:
        friction, friction_torsional, friction_rolling = geoms_friction[geom.type]
        assert_allclose(geom.get_friction(), friction, tol=gs.EPS)
        assert_allclose(geom.get_friction_torsional(), friction_torsional, tol=gs.EPS)
        assert_allclose(geom.get_friction_rolling(), friction_rolling, tol=gs.EPS)
    # The link surface colors the visual geoms of the body, the visual geom options the cap of the selected material
    # over it, and the entity selection by name the capsule of the child link
    vgeoms_color = {
        "torus": (1.0, 0.0, 0.0),
        "top_cap": (0.0, 1.0, 0.0),
        "bottom_cap": (1.0, 0.0, 0.0),
        "tip": (0.0, 0.0, 1.0),
    }
    for vgeom in mixed_geoms_body.vgeoms:
        assert_allclose(vgeom.surface.get_rgba().color[:3], vgeoms_color[vgeom.metadata["name"]], tol=gs.EPS)

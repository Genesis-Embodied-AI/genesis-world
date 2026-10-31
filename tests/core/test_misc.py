"""Tests for the entity naming system."""

import gc
import os
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import torch
import trimesh

import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu
import genesis.utils.point_cloud as pc
from genesis.engine.scene import SCENE_FORMAT
from genesis.utils import serialization
from genesis.utils.cache import disable_owned_cache_release, enable_owned_cache_release
from genesis.utils.misc import indices_to_mask, sanitize_index, tensor_to_array

from ..utils.assertions import assert_allclose, assert_equal


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
def three_objects_obj(asset_tmp_path):
    # Three objects, of which two share a material
    (asset_tmp_path / "three_objects.mtl").write_text("newmtl plastic\nKd 0.5 0.5 0.5\nnewmtl lcd\nKd 0.0 0.0 0.0\n")
    lines = ["mtllib three_objects.mtl"]
    for i, (name, material) in enumerate((("body", "plastic"), ("knob", "plastic"), ("screen", "lcd"))):
        lines += [f"o {name}", f"usemtl {material}"]
        lines += [f"v {0.1 * i + dx} {dy} 0.0" for dx, dy in ((0.0, 0.0), (0.05, 0.0), (0.05, 0.05), (0.0, 0.05))]
        lines += [f"f {4 * i + 1} {4 * i + 2} {4 * i + 3}", f"f {4 * i + 1} {4 * i + 3} {4 * i + 4}"]
    obj_path = asset_tmp_path / "three_objects.obj"
    obj_path.write_text("\n".join(lines) + "\n")
    return obj_path


@pytest.fixture
def three_objects_urdf(three_objects_obj):
    robot = ET.Element("robot", name="panel")
    link = ET.SubElement(robot, "link", name="panel")
    visual = ET.SubElement(link, "visual")
    ET.SubElement(ET.SubElement(visual, "geometry"), "mesh", filename=str(three_objects_obj))
    collision = ET.SubElement(link, "collision")
    ET.SubElement(ET.SubElement(collision, "geometry"), "box", size="0.25 0.05 0.01")
    return ET.tostring(robot, encoding="unicode")


@pytest.fixture
def lamp_urdf(asset_tmp_path):
    # A visual of a single object, and a visual whose asset has no material, which the URDF material names
    bulb_path = asset_tmp_path / "bulb.obj"
    bulb_path.write_text("o bulb\nv 0.0 0.0 0.0\nv 0.05 0.0 0.0\nv 0.0 0.05 0.0\nf 1 2 3\n")
    shade_path = asset_tmp_path / "shade.glb"
    trimesh.creation.box(extents=(0.05, 0.05, 0.05)).export(shade_path)
    robot = ET.Element("robot", name="lamp")
    link = ET.SubElement(robot, "link", name="lamp")
    for mesh_path in (bulb_path, shade_path):
        visual = ET.SubElement(link, "visual")
        ET.SubElement(ET.SubElement(visual, "geometry"), "mesh", filename=str(mesh_path))
    material = ET.SubElement(visual, "material", name="steel")
    ET.SubElement(material, "color", rgba="0.5 0.5 0.5 1.0")
    collision = ET.SubElement(link, "collision")
    ET.SubElement(ET.SubElement(collision, "geometry"), "box", size="0.05 0.05 0.05")
    return ET.tostring(robot, encoding="unicode")


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
def test_entity_options(
    tmp_path,
    show_viewer,
    tol,
    two_box_free_mjcf,
    torus_box_sphere_mjcf,
    three_objects_obj,
    three_objects_urdf,
    lamp_urdf,
):
    DT = 1e-2
    SUBSTEPS = 10
    N_STEPS = 10
    GRAVITY = 9.81

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=DT,
            substeps=SUBSTEPS,
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
        for cls in vars(gs.options).values()
        if isinstance(cls, type) and issubclass(cls, gs.options.EntityOptions) and cls is not gs.options.EntityOptions
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
        if issubclass(options_cls, gs.options.PBDClothEntityOptions):
            options_kwargs.update(
                morph=gs.morphs.Mesh(
                    file="meshes/cloth.obj",
                    pos=pos,
                    scale=0.2,
                ),
            )
        if issubclass(options_cls, gs.options.FEMEntityOptions):
            options_kwargs.update(
                material=gs.materials.Elastic(),
            )
        if issubclass(options_cls, gs.options.MPMEntityOptions):
            options_kwargs.update(
                material=gs.materials.Elastic(
                    youngs_modulus=3e5,
                ),
                solver_options=gs.options.MPMEntitySolverOptions(
                    sampler="regular",
                ),
            )
        if issubclass(options_cls, gs.options.SPHEntityOptions):
            options_kwargs.update(
                solver_options=gs.options.SPHEntitySolverOptions(
                    sampler="regular",
                ),
            )
        if issubclass(options_cls, gs.options.PBDLiquidEntityOptions):
            options_kwargs.update(
                solver_options=gs.options.PBDLiquidEntitySolverOptions(
                    sampler="regular",
                ),
            )
        if issubclass(options_cls, gs.options.PBDParticleEntityOptions):
            options_kwargs.update(
                solver_options=gs.options.PBDParticleEntitySolverOptions(
                    sampler="regular",
                ),
            )
        if issubclass(options_cls, gs.options.HybridEntityOptions):
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
        gs.options.PBDElasticEntityOptions(morph=gs.morphs.Sphere(), material=gs.materials.Elastic(youngs_modulus=1e6))
    with pytest.raises(gs.GenesisException, match="gs.options.FEMEntityOptions or gs.options.MPMEntityOptions"):
        scene.add_entity(morph=gs.morphs.Sphere(), material=gs.materials.Elastic())
    with pytest.raises(gs.GenesisException, match="either entity options"):
        scene.add_entity(morph=gs.morphs.Sphere(), options=gs.options.RigidEntityOptions(morph=gs.morphs.Sphere()))

    # Link and geom options override the material of the entity on the geoms they select, the geom options winning
    arm = scene.add_entity(
        options=gs.options.RigidEntityOptions(
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
                gs.LinkIndex.subtree("arm1"): gs.options.LinkOptions(
                    material=gs.materials.Material(
                        friction_rolling=0.003,
                    ),
                ),
                "arm2": gs.options.LinkOptions(
                    material=gs.materials.Material(
                        density=2000.0,
                        friction=0.8,
                    ),
                    geoms={
                        gs.GeomIndex.at(0): gs.options.GeomOptions(
                            material=gs.materials.Material(
                                friction=1.5,
                            ),
                        ),
                    },
                ),
            },
            geoms={
                gs.GeomIndex.all(): gs.options.GeomOptions(
                    material=gs.materials.Material(
                        friction_torsional=0.02,
                    ),
                ),
            },
        ),
    )
    # Vertex options fix the vertices they select at their initial positions. A box selects in the frame of the morph,
    # which the rotation of the morph turns into a side face of the entity.
    fixed_vertex_options = gs.options.VertexOptions(
        solver_options=gs.options.FEMVertexSolverOptions(
            is_fixed=True,
        ),
    )
    rotated_box = scene.add_entity(
        options=gs.options.FEMEntityOptions(
            morph=gs.morphs.Box(
                pos=(1.5, -1.0, 0.6),
                size=(0.2, 0.2, 0.2),
                euler=(90.0, 0.0, 0.0),
            ),
            material=gs.materials.Elastic(
                youngs_modulus=1e4,
            ),
            vertices={
                gs.VertexIndex.in_box(lower=(-0.1, -0.1, 0.1), upper=(0.1, 0.1, 0.1)): fixed_vertex_options,
            },
        ),
    )
    # The union of an ordinal selection of two top corners and of the vertices closest to the other two pins the whole
    # top face. The ordinals come from the same box added to a scene that is not built.
    probe_box = gs.Scene().add_entity(
        options=gs.options.FEMEntityOptions(
            morph=gs.morphs.Box(
                size=(0.2, 0.2, 0.2),
            ),
            material=gs.materials.Elastic(),
        ),
    )
    probe_verts_pos = tensor_to_array(probe_box.init_positions)
    top_verts_idx = np.flatnonzero(probe_verts_pos[:, 2] > 0.1 - 1e-6)
    top_face_selection = gs.VertexIndex.at(top_verts_idx[:2])
    for corner_pos in probe_verts_pos[top_verts_idx[2:]]:
        top_face_selection = top_face_selection | gs.VertexIndex.closest_to(pos=corner_pos, max_distance=1e-3)
    hanging_box = scene.add_entity(
        options=gs.options.FEMEntityOptions(
            morph=gs.morphs.Box(
                pos=(2.0, -1.0, 0.6),
                size=(0.2, 0.2, 0.2),
            ),
            material=gs.materials.Elastic(
                youngs_modulus=1e4,
            ),
            vertices={
                top_face_selection: fixed_vertex_options,
            },
        ),
    )
    # Entities sharing one collision mesh keep the geom options each of them gives
    spheres = [
        scene.add_entity(
            options=gs.options.RigidEntityOptions(
                morph=gs.morphs.Mesh(
                    file="meshes/sphere.obj",
                    pos=(3.0 + i, -1.0, 0.6),
                    scale=0.1,
                    fixed=True,
                ),
                geoms={
                    gs.GeomIndex.all(): gs.options.GeomOptions(
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
        options=gs.options.RigidEntityOptions(
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
                gs.LinkIndex.subtree("body"): gs.options.LinkOptions(
                    geoms={
                        gs.GeomIndex.named("tip"): gs.options.GeomOptions(
                            material=gs.materials.Material(
                                friction=1.2,
                            ),
                        ),
                    },
                ),
                "body": gs.options.LinkOptions(
                    surface=gs.surfaces.Default(
                        color=(1.0, 0.0, 0.0),
                    ),
                    vgeoms={
                        gs.VGeomIndex.with_material("cap_*"): gs.options.VGeomOptions(
                            surface=gs.surfaces.Default(
                                color=(0.0, 1.0, 0.0),
                            ),
                        ),
                    },
                    geoms={
                        gs.GeomIndex.named("tor*"): gs.options.GeomOptions(
                            material=gs.materials.Material(
                                friction=0.7,
                            ),
                        ),
                        gs.GeomIndex.at([False, True, False]): gs.options.GeomOptions(
                            material=gs.materials.Material(
                                friction=1.5,
                            ),
                        ),
                        gs.GeomIndex.at(slice(1, 3)): gs.options.GeomOptions(
                            material=gs.materials.Material(
                                friction_torsional=0.03,
                            ),
                        ),
                    },
                ),
            },
            geoms={
                gs.GeomIndex.named("*_cap"): gs.options.GeomOptions(
                    material=gs.materials.Material(
                        friction_rolling=0.002,
                    ),
                ),
            },
            vgeoms={
                gs.VGeomIndex.named("tip"): gs.options.VGeomOptions(
                    surface=gs.surfaces.Default(
                        color=(0.0, 0.0, 1.0),
                    ),
                ),
            },
        ),
    )
    # The objects of an OBJ file stay apart as visual geoms named after them, whether the file is a mesh or the visual
    # of a URDF link
    knob_surface = gs.surfaces.Default(
        color=(0.0, 1.0, 0.0),
    )
    objects_mesh = scene.add_entity(
        options=gs.options.RigidEntityOptions(
            morph=gs.morphs.Mesh(
                file=str(three_objects_obj),
                pos=(5.0, 2.0, 0.6),
                fixed=True,
            ),
            vgeoms={
                gs.VGeomIndex.named("knob"): gs.options.VGeomOptions(
                    surface=knob_surface,
                ),
            },
        ),
    )
    objects_urdf = scene.add_entity(
        options=gs.options.RigidEntityOptions(
            morph=gs.morphs.URDF(
                file=three_objects_urdf,
                pos=(5.0, 3.0, 0.6),
                fixed=True,
            ),
            links={
                "panel": gs.options.LinkOptions(
                    vgeoms={
                        gs.VGeomIndex.named("knob"): gs.options.VGeomOptions(
                            surface=knob_surface,
                        ),
                    },
                ),
            },
        ),
    )
    # The visual geom of an OBJ file of a single object is named after it, and a URDF material selects the visual geom
    # whose asset has no material
    lamp = scene.add_entity(
        options=gs.options.RigidEntityOptions(
            morph=gs.morphs.URDF(
                file=lamp_urdf,
                pos=(5.0, 4.0, 0.6),
                fixed=True,
            ),
            vgeoms={
                gs.VGeomIndex.named("bulb"): gs.options.VGeomOptions(
                    surface=knob_surface,
                ),
                gs.VGeomIndex.with_material("steel"): gs.options.VGeomOptions(
                    surface=gs.surfaces.Default(
                        color=(0.0, 0.0, 1.0),
                    ),
                ),
            },
        ),
    )
    # An OBJ material selects the visual geom it is assigned to by its name in the material library
    boat = scene.add_entity(
        options=gs.options.RigidEntityOptions(
            morph=gs.morphs.Mesh(
                file="meshes/boat/boat.obj",
                pos=(5.0, 1.0, 0.6),
                scale=0.1,
                fixed=True,
            ),
            vgeoms={
                gs.VGeomIndex.with_material("material_2"): gs.options.VGeomOptions(
                    surface=gs.surfaces.Default(
                        color=(0.0, 0.0, 1.0),
                    ),
                ),
            },
        ),
    )
    # Every kind of selection travels through a scene file
    exported = tmp_path / f"selections{SCENE_FORMAT}"
    rigid_options = gs.options.RigidEntityOptions(
        morph=gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
        ),
        links={
            gs.LinkIndex.subtree("box_*"): gs.options.LinkOptions(),
            gs.LinkIndex.named("box_baselink"): gs.options.LinkOptions(
                geoms={
                    gs.GeomIndex.at([0]): gs.options.GeomOptions(),
                    gs.GeomIndex.at([True]): gs.options.GeomOptions(),
                    gs.GeomIndex.at(slice(0, 1)): gs.options.GeomOptions(),
                    gs.GeomIndex.named("box*"): gs.options.GeomOptions(),
                },
                surface=gs.surfaces.Default(
                    color=(1.0, 0.0, 0.0),
                ),
                vgeoms={
                    gs.VGeomIndex.with_material("lcd*"): gs.options.VGeomOptions(
                        surface=gs.surfaces.Emission(
                            emissive=(0.0, 1.0, 0.0),
                        ),
                    ),
                },
            ),
        },
        geoms={
            gs.GeomIndex.all(): gs.options.GeomOptions(),
        },
        vgeoms={
            gs.VGeomIndex.at(0): gs.options.VGeomOptions(),
            gs.VGeomIndex.named("box*"): gs.options.VGeomOptions(),
        },
    )
    fem_options = gs.options.FEMEntityOptions(
        morph=gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
        ),
        material=gs.materials.Elastic(),
        vertices={
            gs.VertexIndex.at(0) | gs.VertexIndex.closest_to(pos=(0.05, 0.05, 0.05), max_distance=0.01): (
                gs.options.VertexOptions()
            ),
            gs.VertexIndex.in_box(lower=(0.0, 0.0, 0.0), upper=(0.1, 0.1, 0.1)): gs.options.VertexOptions(),
        },
    )
    serialization.export(exported, {"rigid": rigid_options, "fem": fem_options})
    loaded = serialization.load(exported, {"rigid": gs.options.RigidEntityOptions, "fem": gs.options.FEMEntityOptions})
    assert loaded["rigid"].links == rigid_options.links
    assert loaded["rigid"].geoms == rigid_options.geoms
    assert loaded["rigid"].vgeoms == rigid_options.vgeoms
    assert loaded["fem"].vertices == fem_options.vertices
    assert loaded["rigid"].model_dump() == rigid_options.model_dump()
    assert loaded["fem"].model_dump() == fem_options.model_dump()

    # Geom densities weigh in the mass of an aligned body alone, whose frame stays where its kinematic ghost puts it
    dense_body = scene.add_entity(
        options=gs.options.RigidEntityOptions(
            morph=gs.morphs.MJCF(
                file=two_box_free_mjcf,
                pos=(3.0, 1.0, 0.6),
                align=True,
                recompute_inertia=True,
            ),
            links={
                "body": gs.options.LinkOptions(
                    geoms={
                        gs.GeomIndex.at(0): gs.options.GeomOptions(
                            material=gs.materials.Material(
                                density=8000.0,
                            ),
                        ),
                        gs.GeomIndex.at(1): gs.options.GeomOptions(
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
        options=gs.options.KinematicEntityOptions(
            morph=gs.morphs.MJCF(
                file=two_box_free_mjcf,
                pos=(3.0, 1.0, 0.6),
                align=True,
            ),
        ),
    )
    with pytest.raises(gs.GenesisException, match="vertices"):
        gs.options.FEMEntityOptions(
            morph=gs.morphs.Sphere(),
            material=gs.materials.Cloth(),
            vertices={gs.VertexIndex.at(0): gs.options.VertexOptions()},
        )
    with pytest.raises(gs.GenesisException, match="two_way_soft_constraint"):
        gs.options.RigidEntityOptions(
            morph=gs.morphs.Sphere(),
            solver_options=gs.options.RigidEntitySolverOptions(coup_type="ipc_only"),
            links={"base": gs.options.LinkOptions(solver_options=gs.options.RigidLinkSolverOptions(needs_coup=False))},
        )
    with pytest.raises(gs.GenesisException, match="coupled link"):
        gs.options.HybridEntityOptions(
            morph=gs.morphs.URDF(file="urdf/simple/two_link_arm.urdf"),
            material_soft=gs.materials.Muscle(),
            solver_options=gs.options.HybridEntitySolverOptions(
                rigid=gs.options.RigidEntitySolverOptions(coup_type="two_way_soft_constraint")
            ),
        )
    with pytest.raises(gs.GenesisException, match="enable_vertex_constraints"):
        gs.Scene(
            fem_options=gs.options.FEMOptions(use_implicit_solver=True, enable_vertex_constraints=False)
        ).add_entity(options=rotated_box.options)
    for vertex_index, match in (
        (gs.VertexIndex.closest_to(pos=(0.1, 0.1, 0.2), max_distance=0.05), "max_distance"),
        (gs.VertexIndex.in_box(lower=(0.2, 0.2, 0.2), upper=(0.3, 0.3, 0.3)), "No vertex"),
        (gs.VertexIndex.at(probe_verts_pos.shape[0]), "no valid vertex"),
    ):
        with pytest.raises(gs.GenesisException, match=match):
            gs.Scene().add_entity(
                options=hanging_box.options.model_copy(update={"vertices": {vertex_index: fixed_vertex_options}})
            )
    with pytest.raises(gs.GenesisException, match="equal specificity"):
        scene.add_entity(
            options=gs.options.RigidEntityOptions(
                morph=gs.morphs.Box(pos=(5.0, 0.0, 0.0), size=(0.1, 0.1, 0.1)),
                geoms={
                    gs.GeomIndex.all(): gs.options.GeomOptions(material=gs.materials.Material(friction=1.0)),
                    gs.GeomIndex.at(0): gs.options.GeomOptions(material=gs.materials.Material(friction=2.0)),
                },
            )
        )
    with pytest.raises(gs.GenesisException, match="selects no valid geom"):
        scene.add_entity(
            options=gs.options.RigidEntityOptions(
                morph=gs.morphs.Box(pos=(5.0, 0.0, 0.0), size=(0.1, 0.1, 0.1)),
                geoms={gs.GeomIndex.at(1): gs.options.GeomOptions(material=gs.materials.Material(friction=1.0))},
            )
        )
    with pytest.raises(gs.GenesisException, match="equal specificity"):
        scene.add_entity(
            options=arm.options.model_copy(
                update={
                    "links": {
                        "arm*": gs.options.LinkOptions(material=gs.materials.Material(friction=1.0)),
                        "arm2": gs.options.LinkOptions(material=gs.materials.Material(friction=2.0)),
                    }
                }
            )
        )
    with pytest.raises(gs.GenesisException, match="No collision geom of the entity"):
        scene.add_entity(
            options=arm.options.model_copy(
                update={"geoms": {gs.GeomIndex.named("leg*"): gs.options.GeomOptions(material=gs.materials.Material())}}
            )
        )
    with pytest.raises(gs.GenesisException, match="equal specificity"):
        scene.add_entity(
            options=arm.options.model_copy(
                update={
                    "links": {
                        "arm*": gs.options.LinkOptions(surface=gs.surfaces.Default()),
                        "arm2": gs.options.LinkOptions(surface=gs.surfaces.Rough()),
                    }
                }
            )
        )
    with pytest.raises(gs.GenesisException, match="No visual geom of the entity has a material"):
        scene.add_entity(
            options=mixed_geoms_body.options.model_copy(
                update={"vgeoms": {gs.VGeomIndex.with_material("lcd"): gs.options.VGeomOptions()}}
            )
        )
    with pytest.raises(gs.GenesisException, match="links_to_keep"):
        scene.add_entity(
            options=arm.options.model_copy(update={"links": {gs.LinkIndex.named("arm3"): gs.options.LinkOptions()}})
        )
    for update in (
        {"geoms": {gs.GeomIndex.at(0): gs.options.GeomOptions(material=gs.materials.Material(friction=1.0))}},
        {"vgeoms": {gs.VGeomIndex.at(0): gs.options.VGeomOptions(surface=gs.surfaces.Default())}},
    ):
        with pytest.raises(gs.GenesisException, match="mesh or a primitive"):
            scene.add_entity(options=arm.options.model_copy(update=update))

    # The class of the options picks the solver, which entities ignoring gravity tell apart from the others
    scene.build(n_envs=2)
    # The mass of each box is its volume, 0.2^3 m^3, times the density its geom options give
    assert_allclose(dense_body.get_mass(), (8000.0 + 1000.0) * 0.2**3, tol=tol)
    assert_allclose(dense_body.get_qpos(), dense_body_ghost.get_qpos(), tol=tol)
    # A force on the translation of the body acts at its frame, off its center of mass, so it spins the body
    PUSH_FORCE = 50.0
    dense_body.control_dofs_force([0.0, 0.0, PUSH_FORCE, 0.0, 0.0, 0.0])
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
        if isinstance(entity.options, gs.options.KinematicEntityOptions):
            assert_allclose(drop, 0.0, tol=tol)
        else:
            # The mean of the particles or vertices stands in for the center of mass, so it falls at gravity approximately
            assert_allclose(drop, 0.5 * GRAVITY * (N_STEPS * DT) ** 2, rtol=5e-2)

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
    assert sorted(vgeom.metadata["material_name"] for vgeom in boat.vgeoms) == [
        "material_0",
        "material_1",
        "material_2",
    ]
    for vgeom in boat.vgeoms:
        is_selected = vgeom.metadata["material_name"] == "material_2"
        assert isinstance(vgeom.surface.get_rgba(), gs.textures.ColorTexture) == is_selected
    for entity in (objects_mesh, objects_urdf):
        vgeoms_name = sorted(vgeom.metadata["name"] for vgeom in entity.vgeoms)
        assert vgeoms_name == ["body", "knob", "screen"]
        for vgeom in entity.vgeoms:
            is_knob_color = np.allclose(vgeom.surface.get_rgba().color[:3], (0.0, 1.0, 0.0), atol=gs.EPS)
            assert is_knob_color == (vgeom.metadata["name"] == "knob")
    lamp_vgeoms_color = [vgeom.surface.get_rgba().color[:3] for vgeom in lamp.vgeoms]
    assert_allclose(lamp_vgeoms_color, ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0)), tol=gs.EPS)
    # The fixed vertices, on the face at the top of each box in the frame of its morph, hold the box while the opposite
    # face sags under gravity
    for box in (rotated_box, hanging_box):
        init_verts_pos = tensor_to_array(box.init_positions)
        morph_verts_pos = (init_verts_pos - box.options.morph.pos) @ gu.quat_to_R(np.array(box.options.morph.quat))
        verts_pos = tensor_to_array(box.get_state().pos)
        is_fixed = morph_verts_pos[:, 2] > morph_verts_pos[:, 2].max() - 1e-3
        assert_allclose(verts_pos[:, is_fixed], init_verts_pos[is_fixed], tol=tol)
        is_opposite = morph_verts_pos[:, 2] < morph_verts_pos[:, 2].min() + 1e-3
        assert (init_verts_pos[is_opposite, 2] - verts_pos[:, is_opposite, 2] > 1e-2).all()


@pytest.mark.required
def test_scene_destroy_cleans_up_simulator():
    scene = gs.Scene(show_viewer=False)
    scene.add_entity(
        morph=gs.morphs.Plane(),
    )
    scene.build()
    scene.step()

    assert scene._sim is not None

    scene.destroy()

    assert scene._sim is None
    assert scene._visualizer is None


@pytest.mark.required
def test_scene_destroy_idempotent():
    scene = gs.Scene(show_viewer=False)
    scene.add_entity(
        morph=gs.morphs.Plane(),
    )
    scene.build()
    scene.step()

    scene.destroy()
    assert scene._sim is None

    scene.destroy()
    assert scene._sim is None


@pytest.mark.required
def test_rebuild_keeps_cached_data():
    scene = gs.Scene(show_viewer=False)
    scene.add_entity(
        morph=gs.morphs.Mesh(
            file="meshes/duck/duck.obj",
            scale=0.001,
        ),
    )
    scene.build()
    duck_verts = scene.entities[0].vgeoms[0].trimesh.vertices

    # A rebuilt scene shares the data of the destroyed one while its release is disabled, and loads it anew otherwise.
    # The destroyed scene sits in reference cycles, which only a garbage collection frees.
    for is_release_disabled in (True, False):
        if is_release_disabled:
            disable_owned_cache_release()
        scene.destroy()
        scene = gs.Scene(show_viewer=False)
        gc.collect()
        scene.add_entity(
            morph=gs.morphs.Mesh(
                file="meshes/duck/duck.obj",
                scale=0.001,
            ),
        )
        enable_owned_cache_release()
        assert np.shares_memory(scene.entities[0].vgeoms[0].trimesh.vertices, duck_verts) == is_release_disabled


@pytest.mark.required
@pytest.mark.parametrize("raise_before_build", [True, False])
def test_destroy_after_aborted_camera_build(monkeypatch, raise_before_build):
    from genesis.engine.sensors.camera import RasterizerCameraSensor

    scene = gs.Scene(show_viewer=False)
    camera = scene.add_sensor(
        gs.sensors.RasterizerCameraOptions(
            res=(64, 64),
        )
    )

    # Capture the shared metadata reference now; SensorManager.destroy() drops its dict entry,
    # but the dataclass instance itself stays alive through our local reference so we can
    # inspect its fields after teardown.
    shared_metadata = camera._shared_metadata

    # Inject a bug either at build entry (no metadata population) or after the original build
    # has populated renderer / context / sensors / image_cache.
    original_build = RasterizerCameraSensor.build

    def buggy_build(self):
        if not raise_before_build:
            original_build(self)
        raise RuntimeError("injected camera build failure")

    monkeypatch.setattr(RasterizerCameraSensor, "build", buggy_build)

    with pytest.raises(RuntimeError, match="injected camera build failure"):
        scene.build()

    if raise_before_build:
        assert shared_metadata.renderer is None
    else:
        assert shared_metadata.renderer is not None
        assert shared_metadata.context is not None
        assert shared_metadata.sensors is not None
        assert shared_metadata.image_cache is not None

    # Track shared_metadata.destroy() invocations via instance-level shadow. Assigning to the
    # instance __dict__ takes precedence over class-level lookup for this instance only, so
    # neither the class nor any other metadata instance is affected. The `del` reverts the
    # instance to plain class-level lookup before any finalizer can fire.
    original_destroy = shared_metadata.destroy
    destroy_call_count = [0]

    def tracked_destroy():
        destroy_call_count[0] += 1
        original_destroy()

    shared_metadata.destroy = tracked_destroy
    try:
        scene.destroy()
    finally:
        del shared_metadata.destroy

    assert destroy_call_count[0] == 1
    assert shared_metadata.renderer is None
    assert shared_metadata.context is None
    assert shared_metadata.sensors is None
    assert shared_metadata.image_cache is None


@pytest.mark.required
def test_auto_and_user_names():
    scene = gs.Scene()

    # Auto-generated name
    box = scene.add_entity(
        gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
        )
    )
    assert box.name.startswith("box_")

    # Multiple identical entities should have unique names
    box2 = scene.add_entity(
        gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
        )
    )
    assert box2.name.startswith("box_")
    assert box.name != box2.name

    # User-specified name
    sphere = scene.add_entity(
        gs.morphs.Sphere(
            radius=0.1,
        ),
        name="my_sphere",
    )
    assert sphere.name == "my_sphere"

    # Duplicate name raises error
    with pytest.raises(Exception, match="already exists"):
        scene.add_entity(
            gs.morphs.Cylinder(
                radius=0.1,
                height=0.2,
            ),
            name="my_sphere",
        )


@pytest.mark.required
def test_get_entity_by_name():
    scene = gs.Scene()

    box = scene.add_entity(
        gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
        ),
        name="test_box",
    )
    assert scene.get_entity(name="test_box") is box

    # Non-existent name raises error
    with pytest.raises(Exception, match="not found"):
        scene.get_entity(name="nonexistent")


@pytest.mark.required
def test_get_entity_by_uid():
    scene = gs.Scene()

    box = scene.add_entity(
        gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
        )
    )

    # Short UID lookup (7-character prefix shown in terminal)
    assert scene.get_entity(uid=box.uid.short()) is box

    # Non-existent UID raises error
    with pytest.raises(Exception, match="not found"):
        scene.get_entity(uid=gs.UID().short())


@pytest.mark.required
def test_entity_names_property():
    scene = gs.Scene()

    # Use "B" then "A" to confirm insertion order (not sorted)
    scene.add_entity(
        gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
        ),
        name="B",
    )
    scene.add_entity(
        gs.morphs.Sphere(
            radius=0.1,
        ),
        name="A",
    )
    assert tuple(scene.entity_names) == ("B", "A")


@pytest.mark.required
def test_urdf_mjcf_names_from_file():
    scene = gs.Scene()

    # URDF: plane.urdf has <robot name="plane">
    urdf_entity = scene.add_entity(
        gs.morphs.URDF(
            file="urdf/plane/plane.urdf",
        )
    )
    assert urdf_entity.name.startswith("plane_")

    # MJCF: panda.xml has <mujoco model="panda">
    mjcf_entity = scene.add_entity(
        gs.morphs.MJCF(
            file="xml/franka_emika_panda/panda.xml",
        )
    )
    assert mjcf_entity.name.startswith("panda_")

    # Multiple URDF entities should have unique names
    urdf_entity2 = scene.add_entity(
        gs.morphs.URDF(
            file="urdf/plane/plane.urdf",
        )
    )
    assert urdf_entity2.name.startswith("plane_")
    assert urdf_entity.name != urdf_entity2.name


@pytest.mark.required
def test_morph_orientation_offset_resolution():
    quat_90z = gu.xyz_to_quat(np.array((0.0, 0.0, 90.0)), rpy=True, degrees=True)

    # An unset offset resolves to identity, whether omitted or passed as the None unset sentinel.
    assert_equal(gs.morphs.Box(size=(0.1, 0.1, 0.1)).offset_quat, (1.0, 0.0, 0.0, 0.0))
    assert_equal(gs.morphs.Box(size=(0.1, 0.1, 0.1), offset_quat=None).offset_quat, (1.0, 0.0, 0.0, 0.0))

    # 'offset_euler' resolves into 'offset_quat'.
    assert_allclose(
        gs.morphs.Box(size=(0.1, 0.1, 0.1), offset_euler=(0.0, 0.0, 90.0)).offset_quat, quat_90z, tol=gs.EPS
    )

    # An explicit 'offset_quat' is kept verbatim.
    assert_equal(
        gs.morphs.Box(size=(0.1, 0.1, 0.1), offset_quat=(0.0, 1.0, 0.0, 0.0)).offset_quat, (0.0, 1.0, 0.0, 0.0)
    )

    # A None 'offset_quat' means unset, so a serializer can forward it unconditionally alongside 'offset_euler';
    # it resolves to 'offset_euler'.
    assert_allclose(
        gs.morphs.Box(size=(0.1, 0.1, 0.1), offset_euler=(0.0, 0.0, 90.0), offset_quat=None).offset_quat,
        quat_90z,
        tol=gs.EPS,
    )

    # 'offset_euler' and 'offset_quat' are mutually exclusive.
    with pytest.raises(Exception, match="'offset_euler' and 'offset_quat' cannot both be set"):
        gs.morphs.Box(size=(0.1, 0.1, 0.1), offset_euler=(0.0, 0.0, 90.0), offset_quat=(0.0, 1.0, 0.0, 0.0))


@pytest.mark.required
def test_coacd_options_pca_validation():
    gs.options.CoacdOptions(pca=False)
    with pytest.raises(gs.GenesisException, match="pca=True"):
        gs.options.CoacdOptions(pca=True)


@pytest.mark.required
@pytest.mark.parametrize(
    ("index", "expected"),
    (
        (-1, (3,)),
        (-5, (-5,)),
        ([-5, -4, -1], (-5, 0, 3)),
        ([-4, -2], (0, 2)),
        ((-3, -1), (1, 3)),
        (np.array((-2, -1), dtype=np.int32), (2, 3)),
        (np.array((-5, -1), dtype=np.int32), (-5, 3)),
        (np.array((1, 3), dtype=np.uint32), (1, 3)),
        (np.array((-2.0, -1.0), dtype=np.float64), (2, 3)),
        (np.arange(4)[::-2], (3, 1)),
        (torch.tensor((-2, -1), dtype=torch.int32), (2, 3)),
        (torch.tensor((-5, -1), dtype=torch.int32), (-5, 3)),
        (torch.tensor((0, 3), dtype=torch.uint8), (0, 3)),
        (torch.tensor((-2.0, -1.0)), (2, 3)),
        (range(-4, 0), (0, 1, 2, 3)),
        (range(-5, 0), (-5, 0, 1, 2, 3)),
        (range(-2, 1), (2, 3, 0)),
        (slice(-2, None), (2, 3)),
        (slice(-1, None, -1), (3, 2, 1, 0)),
        (np.array((False, True, False, True)), (1, 3)),
        (torch.tensor((False, True, False, True)), (1, 3)),
    ),
)
def test_sanitize_index(index, expected):
    assert_equal(sanitize_index(index, -1, 4, 0, "index"), expected)


@pytest.mark.required
@pytest.mark.parametrize(
    "index",
    (
        {0, 1},
        [[0, 1]],
        ["a"],
        np.array((True,), dtype=bool),
        torch.tensor((True,), dtype=torch.bool),
    ),
)
def test_sanitize_index_rejects_invalid_collections(index):
    with pytest.raises(gs.GenesisException):
        sanitize_index(index, -1, 4, 0, "index")


@pytest.mark.required
@pytest.mark.parametrize("as_boolean", (False, True))
def test_indices_to_mask_selects_the_cross_product(as_boolean):
    buf = torch.arange(20, dtype=torch.int32).reshape((4, 5))
    if as_boolean:
        envs_idx = torch.tensor((False, True, False, True))
        dofs_idx = torch.tensor((True, False, True, False, False))
    else:
        envs_idx = torch.tensor((1, 3), dtype=torch.int32)
        dofs_idx = torch.tensor((0, 2), dtype=torch.int32)

    # Selecting on two axes takes every combination, rather than pairing the two selections elementwise, which for
    # these inputs would give the two values on the diagonal.
    assert_equal(buf[indices_to_mask(envs_idx, dofs_idx)], torch.tensor(((5, 7), (15, 17)), dtype=torch.int32))
    assert_equal(buf[indices_to_mask(envs_idx)], buf[torch.tensor((1, 3))])


@pytest.mark.required
def test_init_preserves_torch_defaults():
    assert torch.get_default_device() == torch.device("cpu")
    assert torch.get_default_dtype() == torch.float32


@pytest.mark.required
def test_fps_algorithm_core():
    # Shape, dtype, determinism, anchor-on-no-seed, and invalid n_samples all in one test.
    points = np.random.default_rng(1).random((50, 3))
    out_a = pc.furthest_point_sample(points, 10, seed=42)
    out_b = pc.furthest_point_sample(points, 10, seed=42)
    assert out_a.shape == (10, 3)
    assert out_a.dtype == gs.np_float
    assert_equal(out_a, out_b)

    # With seed=None the first sample is the first input point (deterministic anchor).
    anchor = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    out_anchor = pc.furthest_point_sample(anchor, 3, seed=None)
    assert_allclose(out_anchor[0], anchor[0], tol=1e-5)

    with pytest.raises(gs.GenesisException):
        pc.furthest_point_sample(np.zeros((5, 3)), 10, seed=None)


@pytest.mark.required
def test_fps_mesh_sampling_end_to_end():
    # Shape/dtype/determinism, on-surface points, minimum separation, and box-aligned normals all in one scene.
    mesh = trimesh.creation.box((1.0, 1.0, 1.0))
    points_a = pc.sample_mesh_point_cloud(mesh.vertices, mesh.faces, 16, n_candidates=400, seed=7, use_cache=False)
    points_b = pc.sample_mesh_point_cloud(mesh.vertices, mesh.faces, 16, n_candidates=400, seed=7, use_cache=False)
    assert points_a.shape == (16, 3)
    assert points_a.dtype == gs.np_float
    assert_equal(points_a, points_b)

    _, dist, _ = mesh.nearest.on_surface(points_a)
    assert dist.max() < 1e-5

    pairwise = np.linalg.norm(points_a[:, None, :] - points_a[None, :, :], axis=-1)
    np.fill_diagonal(pairwise, np.inf)
    assert pairwise.min() > 0.2

    # Box face normals are axis-aligned unit vectors. Unit-length + `max(|n|) == 1` is sufficient: by Pythagoras the
    # other two components must be ~0.
    _, normals = pc.sample_mesh_point_cloud(
        mesh.vertices, mesh.faces, 32, n_candidates=800, seed=0, use_cache=False, return_normals=True
    )
    assert_allclose(np.linalg.norm(normals, axis=1), 1.0, tol=1e-5)
    assert_allclose(np.abs(normals).max(axis=1), 1.0, tol=1e-4)


@pytest.mark.cache(False)
@pytest.mark.required
def test_fps_cache_round_trip():
    mesh = trimesh.creation.box((1.0, 1.0, 1.0))
    # Samples lie on the surface, so they follow any move of the vertices. Perturbing the mesh the way re-exporting it
    # would, then getting the very samples of the original back, is what proves they were read from the cache.
    reexported_verts = mesh.vertices + np.random.uniform(-1e-7, 1e-7, mesh.vertices.shape)
    # Run both `return_normals` paths: first call writes the cache; the others hit it; outputs identical.
    for return_normals in (True, False):
        kwargs = dict(
            faces=mesh.faces,
            n_points=5,
            n_candidates=10,
            return_normals=return_normals,
            seed=7,
        )
        first = pc.sample_mesh_point_cloud(verts=mesh.vertices, **kwargs, use_cache=True)
        assert_equal(first, pc.sample_mesh_point_cloud(verts=mesh.vertices, **kwargs, use_cache=True))
        assert_equal(first, pc.sample_mesh_point_cloud(verts=reexported_verts, **kwargs, use_cache=True))


@pytest.mark.required
def test_gs_mesh_sample_point_cloud_wrapper():
    mesh = trimesh.creation.box((0.2, 0.4, 0.6))
    gmesh = gs.Mesh.from_trimesh(mesh)
    points, normals = gmesh.sample_point_cloud(10, n_candidates=300, seed=67, use_cache=False, return_normals=True)
    assert points.shape == (10, 3)
    assert normals.shape == (10, 3)
    _, dist, _ = mesh.nearest.on_surface(points)
    assert dist.max() < 1e-5
    assert_allclose(np.linalg.norm(normals, axis=1), 1.0, tol=1e-5)


@pytest.fixture
def two_link_fixed_urdf():
    robot = ET.Element("robot", name="two_link_fixed")
    ET.SubElement(robot, "link", name="base")
    child = ET.SubElement(robot, "link", name="child")
    collision = ET.SubElement(child, "collision")
    geometry = ET.SubElement(collision, "geometry")
    ET.SubElement(geometry, "box", size="0.1 0.1 0.1")
    joint = ET.SubElement(robot, "joint", name="weld", type="fixed")
    ET.SubElement(joint, "parent", link="base")
    ET.SubElement(joint, "child", link="child")
    ET.SubElement(joint, "origin", xyz="0 0 0.1")
    return ET.tostring(robot, encoding="unicode")


@pytest.mark.required
def test_solver_state_change_subscribers(show_viewer, two_link_fixed_urdf):
    # Imported lazily: the solver package pulls in quadrants kernels that need gs.qd_float, set only by gs.init.
    from genesis.engine.state_change import StateChange, Subscriber

    scene = gs.Scene(show_viewer=show_viewer)
    plane = scene.add_entity(gs.morphs.Plane())
    # A fixed entity whose collision geometry lives on a fixed child link, the common URDF/MJCF layout: teleporting
    # it through its base link must reach subscribers watching the child.
    tower = scene.add_entity(
        gs.morphs.URDF(
            file=two_link_fixed_urdf,
            pos=(2.0, 0.0, 0.0),
            fixed=True,
            merge_fixed_links=False,
        )
    )
    cube = scene.add_entity(
        gs.morphs.Box(
            size=(0.2, 0.2, 0.2),
            pos=(0.0, 0.0, 0.5),
        ),
    )
    scene.build(n_envs=2)

    solver = scene.sim.rigid_solver

    # Eager mode: a callback fires immediately on each matching change and nothing is retained.
    eager_events = []
    eager = Subscriber(
        to=frozenset({StateChange.GEOMETRY}),
        callback=lambda change, envs_idx: eager_events.append((change, envs_idx)),
    )
    solver.subscribe(eager)
    # Lazy mode: matching changes accumulate on the Subscriber handle until cleared.
    lazy = Subscriber(to=frozenset({StateChange.GEOMETRY}))
    solver.subscribe(lazy)
    # A DYNAMICS-only subscriber must stay silent on GEOMETRY changes (filter).
    dynamics = Subscriber(to=frozenset({StateChange.DYNAMICS}))
    solver.subscribe(dynamics)

    # zero_velocity=False isolates the pure GEOMETRY change (a default set_pos also zeroes velocity; see below).
    cube.set_pos([[0.0, 0.0, 1.0], [0.0, 0.0, 2.0]], zero_velocity=False)
    # Eager fired once with the right category; envs_idx forwarded verbatim (None == every env).
    assert len(eager_events) == 1
    assert eager_events[0][0] is StateChange.GEOMETRY
    assert eager_events[0][1] is None
    # Lazy accumulated the category; the DYNAMICS subscriber saw nothing; eager retains nothing.
    assert lazy.pending == frozenset({StateChange.GEOMETRY})
    assert dynamics.pending == frozenset()
    assert eager.pending == frozenset()

    # A targeted setter forwards the exact env subset to the eager callback.
    cube.set_pos([[0.0, 0.0, 3.0]], envs_idx=[1], zero_velocity=False)
    assert len(eager_events) == 2
    forwarded = eager_events[1][1]
    assert forwarded is not None
    assert int(np.atleast_1d(tensor_to_array(forwarded))[0]) == 1

    # Lazy state is idempotent across repeated changes and resets on clear().
    assert lazy.pending == frozenset({StateChange.GEOMETRY})
    lazy.clear()
    assert lazy.pending == frozenset()

    # A velocity setter is a DYNAMICS change only: it wakes the DYNAMICS subscriber, not the GEOMETRY ones (setting a
    # velocity does not move the surface).
    cube.set_dofs_velocity([0.0] * cube.n_dofs)
    assert dynamics.pending == frozenset({StateChange.DYNAMICS})
    assert lazy.pending == frozenset()
    assert len(eager_events) == 2

    # Reads never notify.
    solver.get_links_pos()
    solver.get_links_quat()
    assert len(eager_events) == 2
    assert lazy.pending == frozenset()

    # Physics integration mutates state through kernels, not a tagged method, so it never notifies.
    scene.step()
    assert len(eager_events) == 2
    assert lazy.pending == frozenset()

    # A default set_pos both moves the link and zeroes its velocity, so a subscriber listening for either category
    # receives both - the accumulated union of every change the call produced.
    both = Subscriber(to=frozenset({StateChange.GEOMETRY, StateChange.DYNAMICS}))
    solver.subscribe(both)
    cube.set_pos([[0.0, 0.0, 4.0], [0.0, 0.0, 5.0]])
    assert both.pending == frozenset({StateChange.GEOMETRY, StateChange.DYNAMICS})

    # A link-filtered subscriber only wakes on changes that can affect its links. The plane's base link is fixed, so
    # the cube's base-pose setter (an explicit, disjoint link selection) and its configuration-space setters (which
    # can never displace a link without degrees of freedom) both stay silent.
    plane_watcher = Subscriber(to=frozenset({StateChange.GEOMETRY}), links_filter=[plane.base_link_idx])
    solver.subscribe(plane_watcher)
    cube.set_pos([[0.0, 0.0, 6.0], [0.0, 0.0, 7.0]], zero_velocity=False)
    cube.set_qpos([[0.0, 0.0, 8.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 9.0, 1.0, 0.0, 0.0, 0.0]])
    assert plane_watcher.pending == frozenset()
    # A base-pose setter naming a watched link wakes it.
    plane.set_pos((0.0, 0.0, -0.1))
    assert plane_watcher.pending == frozenset({StateChange.GEOMETRY})
    plane_watcher.clear()
    # A base-pose setter with the implicit selection (every base link, the watched fixed one included) wakes it
    # too. The pose is env-agnostic: an env-specific pose on a fixed link would require batch_fixed_verts.
    solver.set_base_links_pos(solver.get_links_pos(links_idx=solver._base_links_idx)[0])
    assert plane_watcher.pending == frozenset({StateChange.GEOMETRY})
    plane_watcher.clear()
    # A base-pose setter reaches the named links' whole kinematic subtree (forward kinematics carries the new pose
    # to every descendant): teleporting the fixed tower through its base link wakes a watcher of its fixed child
    # link, where the collision geometry lives.
    child_link = next(link for link in tower.links if link.name == "child")
    child_watcher = Subscriber(to=frozenset({StateChange.GEOMETRY}), links_filter=[child_link.idx])
    solver.subscribe(child_watcher)
    cube.set_pos([[0.0, 0.0, 10.0], [0.0, 0.0, 11.0]], zero_velocity=False)
    assert child_watcher.pending == frozenset()
    tower.set_pos((2.0, 1.0, 0.0))
    assert child_watcher.pending == frozenset({StateChange.GEOMETRY})

    # A whole-state restore may move any link, so the unbounded reach always passes the filter.
    scene.reset()
    assert plane_watcher.pending == frozenset({StateChange.GEOMETRY})


@pytest.mark.parametrize("backend", [None])
def test_per_solver_gravity():
    # Field-backed storage is what a solver whose kernels reach gravity through the solver itself holds, and the array
    # mode is forced to fields here so both kinds of solver are on that storage at once.
    os.environ["GS_ENABLE_NDARRAY"] = "0"
    try:
        gs.init(backend=gs.cpu, seed=0)

        scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                gravity=(0.0, 0.0, -9.81),
            ),
            show_viewer=False,
        )
        scene.add_entity(gs.morphs.Plane())
        scene.add_entity(
            gs.morphs.Box(
                size=(0.4, 0.4, 0.4),
                pos=(0.0, 0.0, 0.5),
            )
        )
        scene.add_entity(
            options=gs.options.MPMEntityOptions(
                morph=gs.morphs.Sphere(
                    pos=(0.0, 0.0, 0.5),
                    radius=0.1,
                ),
                material=gs.materials.Liquid(),
            ),
        )
        scene.build(n_envs=2)

        new_gravity = [0.0, 0.0, -5.0]

        rigid = scene.sim.rigid_solver
        rigid.set_gravity(new_gravity)
        assert_allclose(rigid.get_gravity(), new_gravity, atol=1e-6)

        mpm = scene.sim.mpm_solver
        mpm.set_gravity(new_gravity)
        assert_allclose(mpm.get_gravity(), new_gravity, atol=1e-6)

        # Gravity is per solver, so setting one leaves the other where it was.
        rigid.set_gravity(0.0)
        assert_allclose(rigid.get_gravity(), 0.0, atol=1e-6)
        assert_allclose(mpm.get_gravity(), new_gravity, atol=1e-6)

        # It is per environment too, so naming one leaves the others where they were.
        rigid.set_gravity([0.0, 0.0, -1.0], envs_idx=1)
        assert_allclose(rigid.get_gravity(envs_idx=0), 0.0, atol=1e-6)
        assert_allclose(rigid.get_gravity(envs_idx=1), [0.0, 0.0, -1.0], atol=1e-6)

    finally:
        gs.destroy()
        os.environ.pop("GS_ENABLE_NDARRAY", None)


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 3])
def test_per_env_time(show_viewer, n_envs, tol):
    DT = 0.01
    N_STEPS = 10
    N_MORE_STEPS = 5

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=DT,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.5, 0.0, 1.0),
            camera_lookat=(0.0, 0.0, 0.5),
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(
        morph=gs.morphs.Sphere(
            pos=(0.0, 0.0, 1.0),
        ),
    )
    scene.build(n_envs=n_envs)

    for _ in range(N_STEPS):
        scene.step()

    # A scene of no parallel environments has one clock, reported as the scalar the whole scene runs on.
    if n_envs == 0:
        assert_equal(scene.get_time().ndim, 0)
        assert_allclose(scene.get_time(), N_STEPS * DT, tol=tol)
        scene.reset()
        assert_allclose(scene.get_time(), 0.0, tol=tol)
        return

    # Environments reset independently, so resetting one rewinds its clock alone while the others keep theirs, whether
    # it is named by an index counted from either end, by a selection of several, or by a mask.
    scene.reset(envs_idx=-2)
    assert_allclose(scene.get_time(), [N_STEPS * DT, 0.0, N_STEPS * DT], tol=tol)

    for _ in range(N_MORE_STEPS):
        scene.step()
    times_expected = [(N_STEPS + N_MORE_STEPS) * DT, N_MORE_STEPS * DT, (N_STEPS + N_MORE_STEPS) * DT]
    assert_allclose(scene.get_time(), times_expected, tol=tol)
    assert_allclose(scene.get_time(envs_idx=1), N_MORE_STEPS * DT, tol=tol)

    scene.reset(envs_idx=(0, 2))
    assert_allclose(scene.get_time(), [0.0, N_MORE_STEPS * DT, 0.0], tol=tol)

    for _ in range(N_MORE_STEPS):
        scene.step()
    scene.reset(envs_idx=torch.tensor((False, True, False), device=gs.device))
    assert_allclose(scene.get_time(), [N_MORE_STEPS * DT, 0.0, N_MORE_STEPS * DT], tol=tol)


@pytest.mark.required
def test_derived_substeps(show_viewer, tol):
    GRAVITY = -9.81
    N_STEPS = 10
    SUBSTEPS = 5

    @qd.data_oriented
    class NullJet:
        @qd.func
        def get_tan_dir(self, t: float):
            return qd.Vector([0.0, 0.0, 1.0], dt=gs.qd_float)

        @qd.func
        def get_factor(self, i, j, k, dx: float, t: float):
            return 0.0

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
            gravity=(0.0, 0.0, GRAVITY),
        ),
        rigid_options=gs.options.RigidOptions(
            dt=0.01 / SUBSTEPS,
        ),
        sf_options=gs.options.SFOptions(
            dt=0.01 / SUBSTEPS,
            res=16,
            solver_iters=2,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.5, 0.0, 1.0),
            camera_lookat=(0.0, 0.0, 0.5),
        ),
        show_viewer=show_viewer,
    )
    ball = scene.add_entity(
        morph=gs.morphs.Sphere(
            pos=(0.0, 0.0, 1.0),
        ),
    )
    sf_solver = scene.sim.sf_solver
    sf_solver.set_jets([NullJet()])
    scene.build()

    # A solver integrates at the interval it was given, as many times per scene step as that interval divides the
    # step dt, and the bodies and the grid read the same resolution.
    solver = scene.rigid_solver
    assert_equal(solver.substeps, SUBSTEPS)
    assert_allclose(solver.substep_dt, 0.01 / SUBSTEPS, tol=gs.EPS)

    # A step advances the step dt whatever the rate: the fall accumulates once per substep, at the solver dt, for as
    # many substeps as the ratio gives.
    t_built = sf_solver.t
    for _ in range(N_STEPS):
        scene.step()
    n_substeps = N_STEPS * solver.substeps
    expected_z = 1.0 + GRAVITY * solver.substep_dt**2 * n_substeps * (n_substeps + 1) / 2
    assert_allclose(ball.get_pos()[2], expected_z, tol=tol)

    # Solvers integrating on a grid rather than on bodies read the interval off the same resolution, so their own
    # clock advances one step dt per scene step however many substeps the ratio gives them.
    assert_equal(sf_solver.substeps, SUBSTEPS)
    assert_allclose(sf_solver.t - t_built, N_STEPS * 0.01, tol=gs.EPS)

    # The substeps shorthand decides the rate when no solver asks for one of its own, which is settled as a solver is
    # made rather than as a scene is built.
    scene_from_substeps = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
            substeps=4,
        ),
        show_viewer=False,
    )
    assert_equal(scene_from_substeps.rigid_solver.substeps, 4)
    assert_allclose(scene_from_substeps.rigid_solver.substep_dt, 0.0025, tol=gs.EPS)

    # The substep loop advances every active solver once per iteration, so the interval one of them asks for is the
    # rate they all take, whether they asked for one or not.
    scene_from_solver_dt = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
        ),
        sf_options=gs.options.SFOptions(
            dt=0.01 / SUBSTEPS,
            res=16,
            solver_iters=2,
        ),
        show_viewer=False,
    )
    ball_at_derived_rate = scene_from_solver_dt.add_entity(
        morph=gs.morphs.Sphere(
            pos=(0.0, 0.0, 1.0),
        ),
    )
    scene_from_solver_dt.sim.sf_solver.set_jets([NullJet()])
    scene_from_solver_dt.build()
    assert_equal(scene_from_solver_dt.rigid_solver.substeps, SUBSTEPS)
    assert_allclose(scene_from_solver_dt.rigid_solver.substep_dt, 0.01 / SUBSTEPS, tol=gs.EPS)

    # The interval reaches what integrates, not only what reports it: a solver settles its buffers as it builds, so a
    # rate inherited afterwards would leave it stepping the scene interval as many times as the shorter one.
    for _ in range(N_STEPS):
        scene_from_solver_dt.step()
    n_substeps = N_STEPS * SUBSTEPS
    expected_z = 1.0 + GRAVITY * scene_from_solver_dt.rigid_solver.substep_dt**2 * n_substeps * (n_substeps + 1) / 2
    assert_allclose(ball_at_derived_rate.get_pos()[2], expected_z, tol=tol)

    # A differentiable scene holds the frames of the substeps of a step, as many as the rate a solver dt derives: the
    # gradient of the position over one step reaches the velocity it started from through every one of them.
    scene_differentiable = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
            gravity=(0.0, 0.0, GRAVITY),
            requires_grad=True,
        ),
        rigid_options=gs.options.RigidOptions(
            dt=0.01 / SUBSTEPS,
        ),
        show_viewer=False,
    )
    ball_differentiable = scene_differentiable.add_entity(
        morph=gs.morphs.Sphere(
            pos=(0.0, 0.0, 1.0),
        ),
    )
    scene_differentiable.build()
    assert_equal(scene_differentiable.rigid_solver.substeps, SUBSTEPS)
    velocity = torch.tensor([0.0, 0.0, 0.5], dtype=gs.tc_float, device=gs.device, requires_grad=True)
    ball_differentiable.set_dofs_velocity(velocity, dofs_idx_local=slice(0, 3))
    scene_differentiable.step()
    ball_differentiable.get_pos()[2].backward()
    assert_allclose(velocity.grad, (0.0, 0.0, 0.01), tol=tol)

    # Two solvers asking for intervals that imply different counts have no one rate to advance at, and are rejected.
    with pytest.raises(gs.GenesisException, match="Solvers integrating at different rates"):
        scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=0.01),
            rigid_options=gs.options.RigidOptions(dt=0.005),
            sf_options=gs.options.SFOptions(dt=0.0025, res=16, solver_iters=2),
            show_viewer=False,
        )
        scene.add_entity(gs.morphs.Sphere(pos=(0.0, 0.0, 1.0)))
        scene.sim.sf_solver.set_jets([NullJet()])
        scene.build()

    # A rate is settled for the solvers the entities made active, and only for those: what an interval would imply for
    # a solver nothing is simulated with is no reason to refuse the scene it sits in.
    scene_inactive = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
        ),
        rigid_options=gs.options.RigidOptions(
            dt=0.003,
        ),
        show_viewer=False,
    )
    scene_inactive.build()
    assert_equal(scene_inactive.sim.substeps, 1)
    assert_allclose(scene_inactive.sim.substep_dt, 0.01, tol=gs.EPS)
    # A scene holding no active solver steps and reports no env in error
    scene_inactive.step()
    assert_equal(scene_inactive.get_error_envs_mask(), False)

    for interval in (0.003, 0.02):
        with pytest.raises(gs.GenesisException, match="does not divide the step dt"):
            scene = gs.Scene(
                sim_options=gs.options.SimOptions(dt=0.01),
                rigid_options=gs.options.RigidOptions(dt=interval),
                show_viewer=False,
            )
            scene.add_entity(gs.morphs.Sphere(pos=(0.0, 0.0, 1.0)))
            scene.build()

    with pytest.raises(gs.GenesisException, match="conflicting with the requested substeps"):
        scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=0.01, substeps=4),
            rigid_options=gs.options.RigidOptions(dt=0.002),
            show_viewer=False,
        )
        scene.add_entity(gs.morphs.Sphere(pos=(0.0, 0.0, 1.0)))
        scene.build()


@pytest.mark.required
@pytest.mark.parametrize(
    "boxes_size",
    [
        # A light body under a heavy scene aggregate, below the floor quoted on the solver tolerance.
        [1.0, 0.001],
        # A uniformly tiny scene, below the absolute floor the working precision resolves at all.
        pytest.param([0.0002], marks=pytest.mark.precision("32")),
    ],
)
def test_warn_solver_numerical_stability(boxes_size, caplog):
    scene = gs.Scene(show_viewer=False)
    for i, size in enumerate(boxes_size):
        scene.add_entity(
            gs.morphs.Box(
                pos=(float(i), 0.0, 0.5 * size),
                size=(size, size, size),
            ),
        )
    with caplog.at_level("WARNING"):
        scene.build()
    assert any("too small for the constraint solver" in record.message for record in caplog.records)

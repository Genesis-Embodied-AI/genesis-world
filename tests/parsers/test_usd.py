"""
Test USD parsing and comparison with compared scenes.

This module tests that USD files can be parsed correctly and that scenes
loaded from USD files match equivalent scenes loaded from compared files.
"""

import os
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
import pytest

try:
    from pxr import Usd
except ImportError as e:
    pytest.skip("USD is not supported because 'pxr' module is not available.", allow_module_level=True)

from pxr import Gf, Sdf, UsdGeom, UsdPhysics
from genesis.utils.usd import UsdContext, HAS_OMNIVERSE_KIT_SUPPORT
from ..conftest import SKIP_NO_OMNIVERSE_KIT

import genesis as gs
from genesis.engine.scene import SCENE_FORMAT
from genesis.utils.misc import tensor_to_array
from genesis.utils.serialization import MANIFEST_NAME

from ..utils.assertions import assert_allclose, assert_equal
from ..utils.assets import get_hf_dataset
from .conftest import (
    USD_COLOR_TOL,
    build_mesh_scene,
    build_mjcf_scene,
    build_usd_scene,
    compare_mesh_scene,
    compare_scene,
)


# ==================== Primitive Tests ====================


@pytest.mark.slow  # ~450s
@pytest.mark.required
@pytest.mark.parametrize("model_name", ["all_primitives_mjcf"])
@pytest.mark.parametrize("scale", [1.0, 2.0])
def test_primitives_mjcf_vs_usd(xml_path, all_primitives_usd, scale, tol):
    mjcf_scene = build_mjcf_scene(xml_path, scale=scale)
    usd_scene = build_usd_scene(all_primitives_usd, scale=scale)
    compare_scene(mjcf_scene, usd_scene, tol=tol)


# ==================== Joint Tests ====================


@pytest.mark.slow  # ~350s
@pytest.mark.required
@pytest.mark.parametrize("model_name", ["all_joints_mjcf"])
@pytest.mark.parametrize("scale", [1.0, 2.0])
@pytest.mark.parametrize(
    "all_joints_usd", [True, False], indirect=True, ids=["with_articulation_root", "without_articulation_root"]
)
def test_joints_mjcf_vs_usd(xml_path, all_joints_usd, scale, tol):
    mjcf_scene = build_mjcf_scene(xml_path, scale=scale)
    usd_scene = build_usd_scene(all_joints_usd, scale=scale)

    # Compare entire scenes - this will check all joints via compare_joints
    compare_scene(mjcf_scene, usd_scene, tol=tol)


@pytest.mark.required
@pytest.mark.parametrize("model_name", ["usd/sneaker_airforce", "usd/RoughnessTest"])
def test_visual_parse(model_name):
    glb_asset_path = get_hf_dataset(pattern=f"{model_name}.glb")
    glb_file = os.path.join(glb_asset_path, f"{model_name}.glb")
    usd_asset_path = get_hf_dataset(pattern=f"{model_name}.usdz")
    usd_file = os.path.join(usd_asset_path, f"{model_name}.usdz")

    mesh_scene = build_mesh_scene(glb_file, scale=1.0)
    usd_scene = build_usd_scene(usd_file, scale=1.0, vis_mode="visual", is_stage=False)

    compare_mesh_scene(mesh_scene, usd_scene, tol=5e-6)


@pytest.mark.required
@pytest.mark.parametrize("usd_file", ["usd/nodegraph.usda"])
def test_parse_nodegraph(usd_file):
    asset_path = get_hf_dataset(pattern=usd_file)
    usd_file = os.path.join(asset_path, usd_file)

    usd_scene = build_usd_scene(usd_file, scale=1.0, vis_mode="visual", is_stage=False)

    texture0 = usd_scene.entities[0].vgeoms[0].vmesh.surface.diffuse_texture
    texture1 = usd_scene.entities[0].vgeoms[1].vmesh.surface.diffuse_texture
    assert isinstance(texture0, gs.textures.ColorTexture)
    assert isinstance(texture1, gs.textures.ColorTexture)
    assert_allclose(texture0.color, (0.8, 0.2, 0.2), rtol=USD_COLOR_TOL)
    assert_allclose(texture1.color, (0.2, 0.6, 0.9), rtol=USD_COLOR_TOL)


@pytest.mark.required
def test_usdz_packaged_texture_resolution(usdz_packaged_texture_usd):
    scene_file, texture_image = usdz_packaged_texture_usd
    usd_scene = build_usd_scene(scene_file, scale=1.0, vis_mode="visual", fixed=True)
    texture = usd_scene.entities[0].vgeoms[0].vmesh.surface.diffuse_texture
    assert isinstance(texture, gs.textures.ImageTexture)
    assert_equal(texture.image_array, texture_image)


@pytest.mark.slow  # ~400s
@pytest.mark.required
@pytest.mark.parametrize(
    "usd_file", ["usd/WoodenCrate/WoodenCrate_D1_1002.usda", "usd/franka_mocap_teleop/table_scene.usd"]
)
@pytest.mark.parametrize("backend", [gs.cuda])
@pytest.mark.skipif(not HAS_OMNIVERSE_KIT_SUPPORT, reason=SKIP_NO_OMNIVERSE_KIT)
def test_bake(usd_file, tmp_path):
    asset_path = get_hf_dataset(pattern=os.path.join(os.path.dirname(usd_file), "*"), local_dir=tmp_path)
    usd_fullpath = os.path.join(asset_path, usd_file)

    is_stage = usd_file == "usd/franka_mocap_teleop/table_scene.usd"
    usd_scene = build_usd_scene(
        usd_fullpath,
        scale=1.0,
        vis_mode="visual",
        is_stage=is_stage,
        fixed=True,
    )

    is_any_baked = False
    for vgeom in usd_scene.entities[0].vgeoms:
        bake_success = vgeom.vmesh.metadata["bake_success"]
        assert bake_success
        is_any_baked |= bake_success
    assert is_any_baked


@pytest.mark.required
@pytest.mark.parametrize("scale", [1.0, 2.0])
def test_massapi_invalid_defaults_mjcf_vs_usd(asset_tmp_path, scale):
    # USD Physics MassAPI defines attributes with sentinel default values - centerOfMass (-inf, -inf, -inf),
    # principalAxes (0, 0, 0, 0), diagonalInertia (0, 0, 0), mass (0) - that must be treated as unset and
    # recomputed from geometry, matching an MJCF scene without inertial element. Leaving only 'centerOfMass' at
    # its sentinel keeps the authored mass and inertia and places the center of mass at the link frame, matching an
    # MJCF body whose inertial frame is the body origin. Leaving only 'diagonalInertia' at its sentinel keeps the
    # authored mass and center of mass and takes the inertia from geometry.
    MASS = 64.0
    HALF_EXTENT = 0.2
    COM = (0.1, 0.0, 0.0)
    BOX_INERTIA = MASS * 2.0 * (2.0 * HALF_EXTENT) ** 2 / 12.0
    mjcf = ET.Element("mujoco", model="massapi_test")

    worldbody = ET.SubElement(mjcf, "worldbody")

    floor = ET.SubElement(worldbody, "body", name="/worldbody/floor")
    ET.SubElement(floor, "geom", type="plane", pos="0. 0. 0.", size="40. 40. 40.")

    box = ET.SubElement(worldbody, "body", name="/worldbody/test_box", pos="0. 0. 0.3")
    ET.SubElement(box, "geom", type="box", size="0.2 0.2 0.2", pos="0. 0. 0.")
    ET.SubElement(box, "joint", name="/worldbody/test_box_joint", type="free")

    inertia_box = ET.SubElement(worldbody, "body", name="/worldbody/inertia_box", pos="0.6 0. 0.3")
    ET.SubElement(inertia_box, "inertial", pos="0. 0. 0.", mass="64.", diaginertia="1.5 1.7 2.0")
    ET.SubElement(inertia_box, "geom", type="box", size="0.2 0.2 0.2", pos="0. 0. 0.")
    ET.SubElement(inertia_box, "joint", name="/worldbody/inertia_box_joint", type="free")

    com_box = ET.SubElement(worldbody, "body", name="/worldbody/com_box", pos="1.2 0. 0.3")
    ET.SubElement(
        com_box,
        "inertial",
        pos=f"{COM[0]} {COM[1]} {COM[2]}",
        mass=str(MASS),
        diaginertia=f"{BOX_INERTIA} {BOX_INERTIA} {BOX_INERTIA}",
    )
    ET.SubElement(com_box, "geom", type="box", size="0.2 0.2 0.2", pos="0. 0. 0.")
    ET.SubElement(com_box, "joint", name="/worldbody/com_box_joint", type="free")

    xml_tree = ET.ElementTree(mjcf)
    xml_file = str(asset_tmp_path / "massapi_test.xml")
    xml_tree.write(xml_file, encoding="utf-8", xml_declaration=True)

    usd_file = str(asset_tmp_path / "massapi_test.usda")

    stage = Usd.Stage.CreateNew(usd_file)
    UsdGeom.SetStageUpAxis(stage, "Z")
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    root_prim = stage.DefinePrim("/worldbody", "Xform")
    stage.SetDefaultPrim(root_prim)

    floor = UsdGeom.Plane.Define(stage, "/worldbody/floor")
    floor.GetAxisAttr().Set("Z")
    floor.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.0))
    floor.GetWidthAttr().Set(80.0)
    floor.GetLengthAttr().Set(80.0)
    UsdPhysics.CollisionAPI.Apply(floor.GetPrim())

    box = UsdGeom.Cube.Define(stage, "/worldbody/test_box")
    box.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.3))
    box.GetSizeAttr().Set(0.4)  # 0.2 half-extent * 2

    box_joint = UsdPhysics.Joint.Define(stage, "/worldbody/test_box_joint")
    box_joint.CreateBody0Rel().SetTargets([root_prim.GetPath()])
    box_joint.CreateBody1Rel().SetTargets([box.GetPrim().GetPath()])
    box_joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    box_joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))

    box_rigid = UsdPhysics.RigidBodyAPI.Apply(box.GetPrim())
    box_rigid.GetKinematicEnabledAttr().Set(False)

    UsdPhysics.MassAPI.Apply(box.GetPrim())

    inertia_box = UsdGeom.Cube.Define(stage, "/worldbody/inertia_box")
    inertia_box.AddTranslateOp().Set(Gf.Vec3d(0.6, 0.0, 0.3))
    inertia_box.GetSizeAttr().Set(0.4)  # 0.2 half-extent * 2

    inertia_box_joint = UsdPhysics.Joint.Define(stage, "/worldbody/inertia_box_joint")
    inertia_box_joint.CreateBody0Rel().SetTargets([root_prim.GetPath()])
    inertia_box_joint.CreateBody1Rel().SetTargets([inertia_box.GetPrim().GetPath()])
    inertia_box_joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    inertia_box_joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))

    UsdPhysics.RigidBodyAPI.Apply(inertia_box.GetPrim()).GetKinematicEnabledAttr().Set(False)

    inertia_mass_api = UsdPhysics.MassAPI.Apply(inertia_box.GetPrim())
    inertia_mass_api.CreateMassAttr(64.0)
    inertia_mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(1.5, 1.7, 2.0))

    com_box = UsdGeom.Cube.Define(stage, "/worldbody/com_box")
    com_box.AddTranslateOp().Set(Gf.Vec3d(1.2, 0.0, 0.3))
    com_box.GetSizeAttr().Set(2.0 * HALF_EXTENT)

    com_box_joint = UsdPhysics.Joint.Define(stage, "/worldbody/com_box_joint")
    com_box_joint.CreateBody0Rel().SetTargets([root_prim.GetPath()])
    com_box_joint.CreateBody1Rel().SetTargets([com_box.GetPrim().GetPath()])
    com_box_joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    com_box_joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))

    UsdPhysics.RigidBodyAPI.Apply(com_box.GetPrim()).GetKinematicEnabledAttr().Set(False)

    com_mass_api = UsdPhysics.MassAPI.Apply(com_box.GetPrim())
    com_mass_api.CreateMassAttr(MASS)
    com_mass_api.CreateCenterOfMassAttr(Gf.Vec3f(*COM))

    stage.Save()

    mjcf_scene = build_mjcf_scene(xml_file, scale=scale)
    usd_scene = build_usd_scene(usd_file, scale=scale)

    # FIXME: Why does the tolerance has to be so lax for this unit test to pass?!
    compare_scene(mjcf_scene, usd_scene, tol=1e-5)


@pytest.mark.required
def test_uv_size_mismatch_no_crash(asset_tmp_path):
    # Nvidia Omniverse tolerates USD meshes with mismatched UV sizes, so the parser must too.
    usd_file = str(asset_tmp_path / "uv_mismatch.usda")

    stage = Usd.Stage.CreateNew(usd_file)
    UsdGeom.SetStageUpAxis(stage, "Z")
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    root_prim = stage.DefinePrim("/root", "Xform")
    stage.SetDefaultPrim(root_prim)

    # Create a simple triangle mesh with intentionally mismatched UVs
    mesh = UsdGeom.Mesh.Define(stage, "/root/mesh")
    mesh.GetPointsAttr().Set([Gf.Vec3f(0, 0, 0), Gf.Vec3f(1, 0, 0), Gf.Vec3f(0, 1, 0), Gf.Vec3f(1, 1, 0)])
    mesh.GetFaceVertexIndicesAttr().Set([0, 1, 2, 1, 3, 2])
    mesh.GetFaceVertexCountsAttr().Set([3, 3])

    # Add UVs with intentionally wrong count (5 UVs for 4 vertices / 6 face-vertex indices)
    primvar_api = UsdGeom.PrimvarsAPI(mesh.GetPrim())
    uv_primvar = primvar_api.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
    uv_primvar.Set([Gf.Vec2f(0, 0), Gf.Vec2f(1, 0), Gf.Vec2f(0, 1), Gf.Vec2f(1, 1), Gf.Vec2f(0.5, 0.5)])

    UsdPhysics.RigidBodyAPI.Apply(mesh.GetPrim())
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())

    stage.Save()

    # This should NOT raise an exception — it should warn and discard UVs
    usd_scene = build_usd_scene(usd_file, scale=1.0, vis_mode="collision", fixed=True)
    assert len(usd_scene.entities) > 0


@pytest.mark.required
@pytest.mark.parametrize("model_name", ["pure_rigid_usd"])
@pytest.mark.parametrize("fixed", [False, True])
def test_pure_rigid_body_fixed(usd_scene, fixed):
    assert len(usd_scene.entities) == 1
    entity = usd_scene.entities[0]
    expected_dofs = 0 if fixed else 6
    assert entity.n_dofs == expected_dofs
    assert entity.n_links == 1


@pytest.mark.required
@pytest.mark.parametrize("model_name", ["collision_only_rigid_usd"])
@pytest.mark.parametrize("fixed,expected_dofs", [(None, 0), (False, 6), (True, 0)])
def test_collision_only_fixed_override(usd_scene, expected_dofs):
    assert len(usd_scene.entities) == 1
    entity = usd_scene.entities[0]
    assert entity.n_dofs == expected_dofs


@pytest.mark.required
def test_visual_collision_parsing(visual_collision_usd, tmp_path):
    usd_scene = build_usd_scene(visual_collision_usd, scale=1.0, fixed=True)
    assert len(usd_scene.entities) == 1
    entity = usd_scene.entities[0]
    link = entity.base_link
    # 2 collision geoms (Collider1 + Collider2), invisible site_marker excluded
    assert link.n_geoms == 2
    # 1 visual geom (Visual1), invisible site_marker excluded
    assert link.n_vgeoms == 1

    # The export carries the descriptions the parse resolved, collision and visual geometry included. The USD
    # context and its directory stay behind.
    exported = tmp_path / f"stage{SCENE_FORMAT}"
    usd_scene.export(exported)
    with zipfile.ZipFile(exported) as archive:
        assert os.path.dirname(visual_collision_usd) not in archive.read(MANIFEST_NAME).decode()
    restored = gs.Scene.load(exported)
    restored.build()
    restored_link = restored.entities[0].base_link
    assert (restored_link.n_geoms, restored_link.n_vgeoms) == (link.n_geoms, link.n_vgeoms)
    assert_equal(restored.entities[0].get_mass(), entity.get_mass())


@pytest.mark.required
def test_ur10_visual_fallback():
    asset_path = get_hf_dataset(pattern="usd/UniversalRobots/UR10/*")
    usd_file = os.path.join(asset_path, "usd/UniversalRobots/UR10/ur10_instanceable.usd")
    usd_scene = build_usd_scene(usd_file, scale=1.0, fixed=True)
    assert len(usd_scene.entities) == 1
    entity = usd_scene.entities[0]
    for link in entity.links:
        if link.n_geoms > 0:
            assert link.n_vgeoms > 0, f"Link {link.name} has collision but no visual geometry (missing fallback)."


@pytest.mark.slow  # ~250s
@pytest.mark.required
def test_humanoid_generic_joint_detection():
    asset_path = get_hf_dataset(pattern="usd/Humanoid/*")
    usd_file = os.path.join(asset_path, "usd/Humanoid/humanoid.usd")
    usd_scene = build_usd_scene(usd_file, scale=1.0, fixed=True)
    assert len(usd_scene.entities) == 1
    entity = usd_scene.entities[0]
    for joint in entity.joints:
        if joint.type == gs.JOINT_TYPE.FREE:
            assert False, f"Joint {joint.name} is FREE but should be SPHERICAL or REVOLUTE."
    spherical_joints = [j for j in entity.joints if j.type == gs.JOINT_TYPE.SPHERICAL]
    assert len(spherical_joints) > 0, "No SPHERICAL joints found — generic PhysicsJoint detection failed."


@pytest.mark.required
@pytest.mark.parametrize("model_name", ["negative_scale_rigid_usd"])
def test_negative_scale_reflection(usd_scene):
    assert len(usd_scene.entities) == 1
    entity = usd_scene.entities[0]
    assert entity.n_links == 1
    assert entity.n_geoms >= 1

    box_geom = next(g for g in entity.geoms if g.type == gs.GEOM_TYPE.BOX)
    assert_allclose(box_geom.data[:3], (1.0, 1.0, 1.0), tol=gs.EPS)


@pytest.mark.required
@pytest.mark.parametrize("model_name", ["nested_collision_joint_usd"])
def test_nested_collision_joint_targets(usd_scene):
    assert len(usd_scene.entities) == 1
    entity = usd_scene.entities[0]
    assert entity.n_links == 2
    assert entity.n_joints == 1

    # localPos1 is in the collision-child frame; anchor must land at the child offset in link space.
    joint = entity.joints[0]
    assert_allclose(joint.pos, (0.0, 0.0, 0.5), tol=gs.EPS)


@pytest.mark.required
@pytest.mark.parametrize("model_name", ["physics_material_usd"])
def test_physics_material_friction_and_density(usd_scene, physics_material_usd):
    assert len(usd_scene.entities) == 3
    entities = {entity.links[0].name: entity for entity in usd_scene.entities}

    # Dynamic friction (0.6) is preferred over static (0.8); restitution (0.4) is dropped.
    assert_allclose(entities["/root/material_body"].geoms[0].desc.friction, 0.6, tol=5e-8)
    # An authored density belongs to its geom and feeds the link inertial. The entity material here states a
    # density, so the resolution drops every authored one.
    assert entities["/root/material_body"].geoms[0].desc.density is None
    assert entities["/root/frictionless_body"].geoms[0].desc.density is None
    # An explicitly authored dynamic_friction = 0 is honored (frictionless collider).
    assert_allclose(entities["/root/frictionless_body"].geoms[0].desc.friction, 0.0, tol=gs.EPS)

    # The explicitly set entity material density (rho=1000 in build_usd_scene) overrides the authored
    # per-geom densities, as material friction does for authored frictions: unit cubes weigh 1000 kg.
    assert_allclose(entities["/root/material_body"].get_mass(), 1000.0, tol=gs.EPS)
    assert_allclose(entities["/root/density_body"].get_mass(), 1000.0, tol=gs.EPS)

    # Without an explicit material density, the authored physics-material density (300) and MassAPI
    # density (500) drive the link mass; recompute_inertia re-derives from geometry and authored
    # densities keep driving that estimate.
    scene = gs.Scene()
    entities = scene.add_stage(
        morph=gs.morphs.USD(
            file=physics_material_usd,
            recompute_inertia=True,
        ),
    )
    scene.build()
    entities = {entity.links[0].name: entity for entity in entities}
    assert_allclose(entities["/root/material_body"].get_mass(), 300.0, tol=gs.EPS)
    assert_allclose(entities["/root/density_body"].get_mass(), 500.0, tol=gs.EPS)
    # This entity material leaves density unset, so each geom keeps its authored value. A link freed from the
    # world then recomputes its inertial at that value rather than at the entity default.
    assert_allclose(entities["/root/material_body"].geoms[0].desc.density, 300.0, tol=gs.EPS)
    assert_allclose(entities["/root/density_body"].geoms[0].desc.density, 500.0, tol=gs.EPS)
    assert entities["/root/frictionless_body"].geoms[0].desc.density is None


@pytest.mark.required
def test_align_anchor_with_geom_densities(density_align_usd):
    scene = gs.Scene()
    body = scene.add_entity(
        gs.morphs.USD(
            file=density_align_usd,
            prim_path="/root/uniform_body",
            align=True,
        ),
    )
    ghost = scene.add_entity(
        gs.morphs.USD(
            file=density_align_usd,
            prim_path="/root/uniform_body",
            align=True,
        ),
        material=gs.materials.Kinematic(),
    )
    scene.build()
    # Two unit cubes at x = -0.5 / +0.5 with densities 100 / 300: mass 400, center of mass at
    # x = 0.25. The aligned body frame anchors at the density-weighted center of mass, identically
    # for the rigid body and its kinematic ghost.
    assert_allclose(body.get_mass(), 400.0, tol=gs.EPS)
    assert_allclose(body.base_link.get_pos(relative=False), (0.25, 0.0, 0.0), tol=gs.EPS)
    assert_allclose(body.base_link.desc.inertial_pos, 0.0, tol=gs.EPS)
    assert_allclose(ghost.base_link.get_pos(relative=False), (0.25, 0.0, 0.0), tol=gs.EPS)


@pytest.mark.required
@pytest.mark.parametrize("align, rho", [(True, None), (None, 1000.0), (None, None)])
def test_align_requires_all_or_none_geom_densities(density_align_usd, align, rho):
    scene = gs.Scene()

    # A density authored on only part of a link's geoms leaves the anchor sensitive to the density the others take.
    # No kinematic entity holds such a density (see '_align_free_roots'), so an explicit align=True raises. The
    # estimate is resolved as the entity is added, and it raises from there.
    if align:
        with pytest.raises(gs.GenesisException, match="with and without an authored density"):
            scene.add_entity(
                gs.morphs.USD(file=density_align_usd, prim_path="/root/mixed_body", align=align),
                material=gs.materials.Rigid(rho=rho),
            )
        return

    body = scene.add_entity(
        gs.morphs.USD(
            file=density_align_usd,
            prim_path="/root/mixed_body",
            align=align,
        ),
        material=gs.materials.Rigid(
            rho=rho,
        ),
    )

    # An explicitly set material density overrides the authored per-geom density, leaving a plain
    # uniform-density body for which auto-alignment proceeds; otherwise auto-alignment quietly declines
    # and the density-less geom follows the entity material's density.
    scene.build()
    assert body.base_link.aligned == (rho is not None)
    if rho is not None:
        assert_allclose(body.get_mass(), 2000.0, tol=gs.EPS)


@pytest.mark.required
@pytest.mark.parametrize("model_name", ["collision_approximation_usd"])
def test_collision_approximations(usd_scene, collision_approximation_usd):
    assert len(usd_scene.entities) == 6
    entities = {entity.links[0].name: entity for entity in usd_scene.entities}

    box_geom = next(g for g in entities["/root/bounding_cube_body"].geoms if g.type == gs.GEOM_TYPE.BOX)
    assert_allclose(box_geom.data[:3], (2.0, 1.0, 1.0), tol=gs.EPS)

    sphere_geom = next(g for g in entities["/root/bounding_sphere_body"].geoms if g.type == gs.GEOM_TYPE.SPHERE)
    # Half the AABB diagonal of a 2x1x1 box: 0.5 * sqrt(4+1+1).
    assert_allclose(sphere_geom.data[0], 0.5 * np.sqrt(6.0), tol=gs.EPS)

    # convexHull forces a single convex collision geom even though morph.convexify is False.
    hull_entity = entities["/root/convex_hull_body"]
    assert hull_entity.n_geoms == 1
    assert hull_entity.geoms[0].mesh.trimesh.is_convex

    # sdf maps to the SDF-based nonconvex mesh path: a single exact concave geom.
    sdf_entity = entities["/root/sdf_body"]
    assert sdf_entity.n_geoms == 1
    assert not sdf_entity.geoms[0].mesh.trimesh.is_convex

    # An authored 'none' pins the exact mesh: morph-level decimation only applies to colliders
    # without an authored approximation.
    scene = gs.Scene()
    entities = scene.add_stage(
        morph=gs.morphs.USD(
            file=collision_approximation_usd,
            fixed=True,
            convexify=False,
            decimate=True,
        ),
    )
    scene.build()
    entities = {entity.links[0].name: entity for entity in entities}
    assert len(entities["/root/raw_body"].geoms[0].mesh.trimesh.faces) == 768
    assert len(entities["/root/plain_body"].geoms[0].mesh.trimesh.faces) < 768


@pytest.mark.required
@pytest.mark.parametrize("model_name", ["collision_filtering_usd"])
def test_collision_filtering_masks(usd_scene):
    assert len(usd_scene.entities) == 2
    entities = {entity.links[0].name: entity for entity in usd_scene.entities}

    # Masks synthesized from in-model filtering only apply within the entity, so collision against
    # other entities (e.g. a ground plane) is preserved.
    for entity in usd_scene.entities:
        assert entity.is_local_collision_mask
        for geom in entity.geoms:
            assert geom.contype or geom.conaffinity

    col_a, col_b = entities["/root/pair_body"].geoms
    assert ((col_a.contype & col_b.conaffinity) | (col_b.contype & col_a.conaffinity)) == 0

    col_c, col_d, col_e = entities["/root/group_body"].geoms
    assert ((col_c.contype & col_d.conaffinity) | (col_d.contype & col_c.conaffinity)) == 0
    assert ((col_c.contype & col_e.conaffinity) | (col_e.contype & col_c.conaffinity)) != 0
    assert ((col_d.contype & col_e.conaffinity) | (col_e.contype & col_d.conaffinity)) != 0


@pytest.mark.required
def test_filtered_entity_still_collides_with_ground(collision_filtering_usd, show_viewer):
    scene = gs.Scene(
        show_viewer=show_viewer,
    )
    scene.add_entity(gs.morphs.Plane())
    entities = scene.add_stage(
        morph=gs.morphs.USD(
            file=collision_filtering_usd,
            pos=(0.0, 0.0, 0.05),
        ),
    )
    scene.build()

    for _ in range(50):
        scene.step()

    # Both filtered bodies must be stopped by the ground plane, resting on it without penetration.
    for entity in entities:
        aabb_min, _aabb_max = tensor_to_array(entity.get_AABB())
        assert -5e-4 < aabb_min[-1] < 0.0


@pytest.mark.required
def test_cross_entity_filtering_not_applied(cross_entity_filtering_usd, caplog):
    with caplog.at_level("WARNING"):
        usd_scene = build_usd_scene(cross_entity_filtering_usd, scale=1.0, fixed=True)
    assert any("cross-entity" in record.getMessage() for record in caplog.records)
    # The pair keeps default masks: filtering across entities cannot be expressed, so it still collides.
    assert len(usd_scene.entities) == 2
    geom_a, geom_b = (entity.geoms[0] for entity in usd_scene.entities)
    assert ((geom_a.contype & geom_b.conaffinity) | (geom_b.contype & geom_a.conaffinity)) != 0


@pytest.mark.slow  # ~250s
@pytest.mark.required
def test_oriented_capsule(oriented_capsule_usd, show_viewer, tol):
    scene = gs.Scene(
        show_viewer=show_viewer,
    )
    scene.add_entity(gs.morphs.Plane())
    capsule = scene.add_entity(
        morph=gs.morphs.USD(
            file=oriented_capsule_usd,
            pos=(0, 0, 0.5),
        ),
        material=gs.materials.Rigid(
            rho=1000.0,
        ),
    )
    scene.build()

    # Capsule with axis=X must have non-identity quat encoding the axis orientation, and visual/collision quats must
    # match (mesh is Z-aligned, quat orients it — MuJoCo convention).
    assert capsule.n_geoms >= 1
    assert capsule.n_vgeoms >= 1
    for vgeom, geom in zip(capsule.vgeoms, capsule.geoms):
        assert_allclose(vgeom.init_quat, geom.init_quat, tol=gs.EPS)
        # Capsule with axis=X has identity quat — axis rotation not composed into geom_Q.
        with pytest.raises(AssertionError):
            assert_allclose(geom.get_quat(), (1.0, 0.0, 0.0, 0.0), atol=tol)

    # Drop on ground plane — must settle above ground (no penetration) and actually fall
    for _ in range(50):
        scene.step()

    capsule_aabb_min, _capsule_aabb_max = tensor_to_array(
        scene.rigid_solver.get_AABB(entities_idx=(capsule.idx,))[..., 0, :]
    )
    capsule_aabb_min_z = float(capsule_aabb_min[-1])
    assert -5e-4 < capsule_aabb_min_z < 0.0


@pytest.mark.slow  # ~200s
@pytest.mark.required
def test_ant_capsule_axis_collision(show_viewer):
    scene = gs.Scene(
        show_viewer=show_viewer,
    )
    scene.add_entity(gs.morphs.Plane())
    asset_path = get_hf_dataset(pattern="usd/Ant/*")
    ant = scene.add_entity(
        morph=gs.morphs.USD(
            file=os.path.join(asset_path, "usd/Ant/ant.usd"),
            pos=(0, 0, 0.75),
        ),
        material=gs.materials.Rigid(
            rho=1000.0,
        ),
    )
    scene.build()

    # Assert visual/collision quats match for all capsule geoms
    for link in ant.links:
        for vgeom, geom in zip(link.vgeoms, link.geoms):
            assert_allclose(vgeom.get_pos(), geom.get_pos(), tol=1e-5)

    # Step and verify no ground penetration
    for _ in range(50):
        scene.step()

    ant_aabb_min, _capsule_aabb_max = tensor_to_array(ant.get_AABB())
    ant_aabb_min_z = float(ant_aabb_min[-1])
    assert -0.001 < ant_aabb_min_z < 0.001

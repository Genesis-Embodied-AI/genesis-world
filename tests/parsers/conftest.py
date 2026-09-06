import io
import os
import xml.etree.ElementTree as ET
from functools import partial, partialmethod

import numpy as np
import pygltflib
import pytest
import trimesh
from PIL import Image

import genesis as gs
import genesis.utils.geom as gu

from genesis.utils.misc import get_assets_dir

from ..utils.assertions import assert_allclose, assert_equal
from ..utils.assets import get_hf_dataset

try:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, UsdUtils

    from genesis.utils.usd import UsdContext
except ImportError:
    pass


@pytest.fixture
def mesh_path(mesh_file):
    path = os.path.join(get_assets_dir(), mesh_file)
    if not os.path.exists(path):
        path = os.path.join(get_hf_dataset(pattern=mesh_file), mesh_file)
    return path


@pytest.fixture
def mesh_urdf(mesh_path):
    """URDF content wrapping the mesh as the visual geometry of a one-link robot."""
    robot = ET.Element("robot", name="model")
    geometry = ET.SubElement(ET.SubElement(ET.SubElement(robot, "link", name="base"), "visual"), "geometry")
    ET.SubElement(geometry, "mesh", filename=mesh_path)
    return ET.tostring(robot, encoding="unicode")


@pytest.fixture
def glb_path(glb_file, accessor_zero_attribute, asset_tmp_path):
    """Path to an existing GLB or a generated mesh with the selected attribute at accessor zero."""
    if accessor_zero_attribute is None:
        return os.path.join(get_hf_dataset(pattern=glb_file), glb_file)

    # Authored shading normals differ from the triangle's geometric normal
    mesh = trimesh.Trimesh(
        vertices=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        faces=[[0, 1, 2]],
        vertex_normals=[[1.0, 0.0, 0.0]] * 3,
        visual=trimesh.visual.TextureVisuals(
            uv=[[0.125, 0.25], [0.375, 0.5], [0.625, 0.75]],
            material=trimesh.visual.material.PBRMaterial(baseColorTexture=Image.new("RGB", (2, 2), "white")),
        ),
        process=False,
    )
    path = asset_tmp_path / f"accessor_zero_{accessor_zero_attribute}.glb"
    mesh.export(path, include_normals=True)
    glb = pygltflib.GLTF2().load(path)
    primitive = glb.meshes[0].primitives[0]
    attributes = primitive.attributes
    first_accessor = attributes.NORMAL if accessor_zero_attribute == "NORMAL" else attributes.TEXCOORD_0
    # Reorder the accessor table while keeping every reference attached to its authored data
    order = [first_accessor] + [i for i in range(len(glb.accessors)) if i != first_accessor]
    glb.accessors = [glb.accessors[i] for i in order]
    attributes.POSITION = order.index(attributes.POSITION)
    attributes.NORMAL = order.index(attributes.NORMAL)
    attributes.TEXCOORD_0 = order.index(attributes.TEXCOORD_0)
    primitive.indices = order.index(primitive.indices)
    if accessor_zero_attribute == "TEXCOORD_1":
        attributes.TEXCOORD_1 = attributes.TEXCOORD_0
        glb.materials[primitive.material].pbrMetallicRoughness.baseColorTexture.texCoord = 1
    glb.save_binary(str(path))
    return str(path)


# Conversion from .usd to .glb significantly affects precision
USD_COLOR_TOL = 1e-07


USD_NORMALS_TOL = 1e-02


# UsdPhysics declares every joint attribute as single precision, so an anchor, a limit or a drive gain authored in
# double comes back rounded from a USD file and no parser can recover it. Comparing joints against the model they were
# authored from is bounded by that rounding, whatever precision the simulation itself runs at, which is why they are
# the only quantities not held to the session tolerance.
USD_PHYSICS_TOL = 5e-07


def extract_mesh(gs_mesh):
    """Extract vertices, normals, uvs, and faces from a gs.Mesh object."""
    vertices = gs_mesh.trimesh.vertices
    normals = gs_mesh.trimesh.vertex_normals
    uvs = gs_mesh.trimesh.visual.uv
    faces = gs_mesh.trimesh.faces

    indices = np.lexsort(
        [
            uvs[:, 1],
            uvs[:, 0],
            normals[:, 2],
            normals[:, 1],
            normals[:, 0],
            vertices[:, 2],
            vertices[:, 1],
            vertices[:, 0],
        ]
    )

    vertices = vertices[indices]
    normals = normals[indices]
    uvs = uvs[indices]
    invdices = np.argsort(indices)
    faces = invdices[faces]
    return vertices, faces, normals, uvs


def check_gs_meshes(gs_mesh1, gs_mesh2, mesh_name, vertices_tol, normals_tol):
    """Check if two gs.Mesh objects are equal."""
    vertices1, faces1, normals1, uvs1 = extract_mesh(gs_mesh1)
    vertices2, faces2, normals2, uvs2 = extract_mesh(gs_mesh2)

    assert_allclose(vertices1, vertices2, atol=vertices_tol, err_msg=f"Vertices match failed in mesh {mesh_name}.")
    assert_equal(faces1, faces2, err_msg=f"Faces match failed in mesh {mesh_name}.")
    assert_allclose(normals1, normals2, atol=normals_tol, err_msg=f"Normals match failed in mesh {mesh_name}.")
    assert_allclose(uvs1, uvs2, rtol=gs.EPS, err_msg=f"UVs match failed in mesh {mesh_name}.")


def check_gs_tm_meshes(gs_mesh, tm_mesh, mesh_name, vertices_tol, normals_tol):
    """Check if a gs.Mesh object and a trimesh.Trimesh object are equal."""
    assert_allclose(
        tm_mesh.vertices,
        gs_mesh.trimesh.vertices,
        tol=vertices_tol,
        err_msg=f"Vertices match failed in mesh {mesh_name}.",
    )
    assert_equal(
        tm_mesh.faces,
        gs_mesh.trimesh.faces,
        err_msg=f"Faces match failed in mesh {mesh_name}.",
    )
    assert_allclose(
        tm_mesh.vertex_normals,
        gs_mesh.trimesh.vertex_normals,
        tol=normals_tol,
        err_msg=f"Normals match failed in mesh {mesh_name}.",
    )
    if not isinstance(tm_mesh.visual, trimesh.visual.color.ColorVisuals):
        assert_allclose(
            tm_mesh.visual.uv,
            gs_mesh.trimesh.visual.uv,
            rtol=gs.EPS,
            err_msg=f"UVs match failed in mesh {mesh_name}.",
        )


def check_gs_tm_textures(gs_texture, tm_color, tm_image, default_value, dim, material_name, texture_name):
    """Check if a gs.Texture object and a trimesh.Texture object are equal."""
    if isinstance(gs_texture, gs.textures.ColorTexture):
        tm_color = tm_color or (default_value,) * dim
        assert_allclose(
            tm_color,
            gs_texture.color,
            rtol=gs.EPS,
            err_msg=f"Color mismatch for material {material_name} in {texture_name}.",
        )
    elif isinstance(gs_texture, gs.textures.ImageTexture):
        tm_color = tm_color or (1.0,) * dim
        assert_allclose(
            tm_color,
            gs_texture.image_color,
            rtol=gs.EPS,
            err_msg=f"Color mismatch for material {material_name} in {texture_name}.",
        )
        assert_equal(
            tm_image,
            gs_texture.image_array,
            err_msg=f"Texture mismatch for material {material_name} in {texture_name}.",
        )


def check_gs_textures(gs_texture1, gs_texture2, default_value, material_name, texture_name):
    """Check if two gs.Texture objects are equal."""
    if gs_texture1 is None:
        gs_texture1, gs_texture2 = gs_texture2, gs_texture1
    if gs_texture1 is not None:
        gs_texture1 = gs_texture1.check_simplify()
    if gs_texture2 is not None:
        gs_texture2 = gs_texture2.check_simplify()

    if isinstance(gs_texture1, gs.textures.ColorTexture):
        gs_color2 = (default_value,) * len(gs_texture1.color) if gs_texture2 is None else gs_texture2.color
        assert_allclose(
            gs_texture1.color,
            gs_color2,
            rtol=gs.EPS,
            err_msg=f"Color mismatch for material {material_name} in {texture_name}.",
        )
    elif isinstance(gs_texture1, gs.textures.ImageTexture):
        assert isinstance(gs_texture2, gs.textures.ImageTexture)
        assert_allclose(
            gs_texture1.image_color,
            gs_texture2.image_color,
            rtol=gs.EPS,
            err_msg=f"Color mismatch for material {material_name} in {texture_name}.",
        )
        assert_equal(
            gs_texture1.image_array,
            gs_texture2.image_array,
            err_msg=f"Texture mismatch for material {material_name} in {texture_name}.",
        )
    else:
        assert gs_texture1 is None and gs_texture2 is None, (
            f"Both textures should be None for material {material_name} in {texture_name}."
        )


def check_gs_surfaces(gs_surface1, gs_surface2, material_name):
    """Check if two gs.Surface objects are equal."""
    check_gs_textures(gs_surface1.texture, gs_surface2.texture, 1.0, material_name, "color")
    check_gs_textures(gs_surface1.opacity_texture, gs_surface2.opacity_texture, 1.0, material_name, "opacity")
    check_gs_textures(gs_surface1.roughness_texture, gs_surface2.roughness_texture, 1.0, material_name, "roughness")
    check_gs_textures(gs_surface1.metallic_texture, gs_surface2.metallic_texture, 0.0, material_name, "metallic")
    check_gs_textures(gs_surface1.normal_texture, gs_surface2.normal_texture, 0.0, material_name, "normal")
    check_gs_textures(gs_surface1.emissive_texture, gs_surface2.emissive_texture, 0.0, material_name, "emissive")


def to_array(s: str) -> np.ndarray:
    """Convert a string of space-separated floats to a numpy array."""
    return np.array([float(x) for x in s.split()])


def compare_links(compared_links, usd_links, tol):
    """Compare links between two scenes."""
    # Check number of links
    assert len(compared_links) == len(usd_links)

    # Create dictionaries keyed by link name for comparison
    compared_links_by_name = {link.name: link for link in compared_links}
    usd_links_by_name = {link.name: link for link in usd_links}

    # Create index to name mappings for parent comparison
    compared_idx_to_name = {i: link.name for i, link in enumerate(compared_links)}
    usd_idx_to_name = {i: link.name for i, link in enumerate(usd_links)}

    # Check that we have matching link names
    compared_link_names = set(compared_links_by_name.keys())
    usd_link_names = set(usd_links_by_name.keys())
    assert compared_link_names == usd_link_names

    # Compare all link properties by name
    for link_name in sorted(compared_link_names):
        compared_link = compared_links_by_name[link_name]
        usd_link = usd_links_by_name[link_name]
        err_msg = f"Properties mismatched for link {link_name}"

        # Compare link properties
        assert_allclose(compared_link.desc.pos, usd_link.desc.pos, tol=tol, err_msg=err_msg)
        assert_allclose(compared_link.desc.quat, usd_link.desc.quat, tol=tol, err_msg=err_msg)
        assert compared_link.is_fixed == usd_link.is_fixed, err_msg
        assert len(compared_link.geoms) == len(usd_link.geoms), err_msg
        assert compared_link.n_joints == usd_link.n_joints, err_msg
        assert len(compared_link.vgeoms) == len(usd_link.vgeoms), err_msg

        # Compare parent link by name (mapping indices to names)
        compared_parent_idx = compared_link.parent_idx
        usd_parent_idx = usd_link.parent_idx
        if compared_parent_idx == -1:
            compared_parent_name = None
        else:
            compared_parent_name = compared_idx_to_name.get(compared_parent_idx, f"<unknown idx {compared_parent_idx}>")
        if usd_parent_idx == -1:
            usd_parent_name = None
        else:
            usd_parent_name = usd_idx_to_name.get(usd_parent_idx, f"<unknown idx {usd_parent_idx}>")
        assert compared_parent_name == usd_parent_name, err_msg

        # Compare inertial properties if available
        assert_allclose(compared_link.desc.inertial_pos, usd_link.desc.inertial_pos, tol=tol, err_msg=err_msg)
        assert_allclose(compared_link.desc.inertial_quat, usd_link.desc.inertial_quat, tol=tol, err_msg=err_msg)

        # Skip mass and inertia checks for fixed links - they're not used in simulation
        if not compared_link.is_fixed:
            assert_allclose(compared_link.desc.mass, usd_link.desc.mass, atol=tol, err_msg=err_msg)
            assert_allclose(compared_link.desc.inertia, usd_link.desc.inertia, atol=tol, err_msg=err_msg)


def compare_joints(compared_joints, usd_joints):
    """Compare joints between two scenes, within what UsdPhysics can represent (see USD_PHYSICS_TOL)."""
    # Check number of joints
    assert len(compared_joints) == len(usd_joints)

    # Create dictionaries keyed by joint name for comparison
    compared_joints_by_name = {joint.name: joint for joint in compared_joints}
    usd_joints_by_name = {joint.name: joint for joint in usd_joints}

    # Check that we have matching joint names
    compared_joint_names = set(compared_joints_by_name.keys())
    usd_joint_names = set(usd_joints_by_name.keys())
    assert compared_joint_names == usd_joint_names

    # Compare all joint properties by name
    for joint_name in sorted(compared_joint_names):
        compared_joint = compared_joints_by_name[joint_name]
        usd_joint = usd_joints_by_name[joint_name]

        # Compare joint properties
        assert compared_joint.type == usd_joint.type
        err_msg = f"Properties mismatched for joint type {compared_joint.type}"
        assert_joint_allclose = partial(assert_allclose, tol=USD_PHYSICS_TOL, err_msg=err_msg)

        assert_joint_allclose(compared_joint.pos, usd_joint.pos)
        assert_joint_allclose(compared_joint.quat, usd_joint.quat)
        assert compared_joint.n_qs == usd_joint.n_qs, err_msg
        assert compared_joint.n_dofs == usd_joint.n_dofs, err_msg

        # Compare initial qpos
        assert_joint_allclose(compared_joint.init_qpos, usd_joint.init_qpos)

        # Skip mass/inertia-dependent property checks for fixed joints - they're not used in simulation
        if compared_joint.type != gs.JOINT_TYPE.FIXED:
            # Compare dof limits
            assert_joint_allclose(compared_joint.desc.dofs_limit, usd_joint.desc.dofs_limit)

            # Compare dof motion properties
            assert_joint_allclose(compared_joint.desc.dofs_motion_ang, usd_joint.desc.dofs_motion_ang)
            assert_joint_allclose(compared_joint.desc.dofs_motion_vel, usd_joint.desc.dofs_motion_vel)
            assert_joint_allclose(compared_joint.desc.dofs_frictionloss, usd_joint.desc.dofs_frictionloss)
            assert_joint_allclose(compared_joint.desc.dofs_stiffness, usd_joint.desc.dofs_stiffness)
            assert_joint_allclose(compared_joint.desc.dofs_force_range, usd_joint.desc.dofs_force_range)
            assert_joint_allclose(compared_joint.desc.dofs_damping, usd_joint.desc.dofs_damping)
            assert_joint_allclose(compared_joint.desc.dofs_armature, usd_joint.desc.dofs_armature)

            # Compare dof control properties
            assert_joint_allclose(compared_joint.desc.dofs_act_gain, usd_joint.desc.dofs_act_gain)
            assert_joint_allclose(compared_joint.desc.dofs_act_bias, usd_joint.desc.dofs_act_bias)


def compare_geoms(compared_geoms, usd_geoms, tol):
    """Compare geoms between two scenes."""
    assert len(compared_geoms) == len(usd_geoms)

    # Sort geoms by link name for consistent comparison
    compared_geoms_sorted = sorted(compared_geoms, key=lambda g: (g.link.name, g.idx))
    usd_geoms_sorted = sorted(usd_geoms, key=lambda g: (g.link.name, g.idx))

    for compared_geom, usd_geom in zip(compared_geoms_sorted, usd_geoms_sorted):
        assert compared_geom.type == usd_geom.type
        err_msg = f"Properties mismatched for geom type {compared_geom.type}"

        assert_allclose(compared_geom.init_pos, usd_geom.init_pos, tol=tol, err_msg=err_msg)
        assert_allclose(compared_geom.init_quat, usd_geom.init_quat, tol=tol, err_msg=err_msg)
        assert_allclose(compared_geom.get_AABB(), usd_geom.get_AABB(), tol=tol, err_msg=err_msg)


def compare_vgeoms(compared_vgeoms, usd_vgeoms, tol):
    """Compare visual geoms between two scenes."""
    assert len(compared_vgeoms) == len(usd_vgeoms)

    # Sort geoms by link name for consistent comparison
    compared_vgeoms_sorted = sorted(compared_vgeoms, key=lambda g: g.vmesh.metadata["name"])
    usd_vgeoms_sorted = sorted(usd_vgeoms, key=lambda g: g.vmesh.metadata["name"].split("/")[-1])

    for compared_vgeom, usd_vgeom in zip(compared_vgeoms_sorted, usd_vgeoms_sorted):
        compared_vgeom_pos, compared_vgeom_quat = gu.transform_pos_quat_by_trans_quat(
            compared_vgeom.init_pos,
            compared_vgeom.init_quat,
            compared_vgeom.link.desc.pos,
            compared_vgeom.link.desc.quat,
        )
        usd_vgeom_pos, usd_vgeom_quat = gu.transform_pos_quat_by_trans_quat(
            usd_vgeom.init_pos, usd_vgeom.init_quat, usd_vgeom.link.desc.pos, usd_vgeom.link.desc.quat
        )
        compared_vgeom_T = gu.trans_quat_to_T(compared_vgeom_pos, compared_vgeom_quat)
        usd_vgeom_T = gu.trans_quat_to_T(usd_vgeom_pos, usd_vgeom_quat)

        compared_vgeom_mesh = compared_vgeom.vmesh.copy()
        usd_vgeom_mesh = usd_vgeom.vmesh.copy()
        mesh_name = usd_vgeom_mesh.metadata["name"]
        compared_vgeom_mesh.apply_transform(compared_vgeom_T)
        usd_vgeom_mesh.apply_transform(usd_vgeom_T)
        check_gs_meshes(compared_vgeom_mesh, usd_vgeom_mesh, mesh_name, tol, USD_NORMALS_TOL)

        compared_vgeom_surface = compared_vgeom_mesh.surface
        usd_vgeom_surface = usd_vgeom_mesh.surface
        check_gs_surfaces(compared_vgeom_surface, usd_vgeom_surface, mesh_name)


def compare_scene(compared_scene: gs.Scene, usd_scene: gs.Scene, tol: float):
    """Compare structure and data between compared scene and USD scene."""
    compared_entities = compared_scene.entities
    usd_entities = usd_scene.entities

    compared_geoms = [geom for entity in compared_entities for geom in entity.geoms]
    usd_geoms = [geom for entity in usd_entities for geom in entity.geoms]
    compare_geoms(compared_geoms, usd_geoms, tol=tol)

    compared_joints = [joint for entity in compared_entities for joint in entity.joints]
    usd_joints = [joint for entity in usd_entities for joint in entity.joints]
    compare_joints(compared_joints, usd_joints)

    compared_links = [link for entity in compared_entities for link in entity.links]
    usd_links = [link for entity in usd_entities for link in entity.links]
    compare_links(compared_links, usd_links, tol=tol)


def compare_mesh_scene(compared_scene: gs.Scene, usd_scene: gs.Scene, tol: float):
    """Compare mesh data between mesh scene and USD scene."""
    compared_entities = compared_scene.entities
    usd_entities = usd_scene.entities
    compared_vgeoms = [vgeom for entity in compared_entities for vgeom in entity.vgeoms]
    usd_vgeoms = [vgeom for entity in usd_entities for vgeom in entity.vgeoms]
    compare_vgeoms(compared_vgeoms, usd_vgeoms, tol=tol)


def build_mjcf_scene(xml_path: str, scale: float):
    """Build a MJCF scene from its file path."""
    # Create MJCF scene
    mjcf_scene = gs.Scene()

    mjcf_scene.add_entity(
        gs.morphs.MJCF(
            file=xml_path,
            scale=scale,
            convexify=False,
            decimate=False,
            align=False,
        ),
        material=gs.materials.Rigid(
            rho=1000.0,
        ),
    )

    mjcf_scene.build()
    return mjcf_scene


def build_usd_scene(
    usd_file: str,
    scale: float,
    vis_mode: str = "collision",
    is_stage: bool = True,
    fixed: bool | None = None,
    show_viewer: bool = False,
):
    """Build a USD scene from its file path."""
    # Create USD scene
    scene = gs.Scene(
        show_viewer=show_viewer,
    )

    kwargs = dict(
        morph=gs.morphs.USD(
            file=usd_file,
            scale=scale,
            fixed=fixed,
            convexify=False,
            decimate=False,
            watertighten=None,
            align=False,
        ),
        material=gs.materials.Rigid(
            rho=1000.0,
        ),
        vis_mode=vis_mode,
    )

    # A stage is read as authored rather than through baked assets an earlier run left on disk
    with pytest.MonkeyPatch.context() as patched:
        patched.setattr(UsdContext, "__init__", partialmethod(UsdContext.__init__, use_bake_cache=False))
        if is_stage:
            scene.add_stage(**kwargs)
        else:
            scene.add_entity(**kwargs)

    # Note that it is necessary to build the scene because spatial inertia of some geometries may not be specified.
    # In such a case, it will be estimated from the geometry during build (RigidLink._build to be specific).
    scene.build()

    return scene


def build_mesh_scene(mesh_file: str, scale: float):
    """Build a mesh scene from its file path."""
    mesh_scene = gs.Scene()
    mesh_morph = gs.morphs.Mesh(
        file=mesh_file,
        scale=scale,
        file_meshes_are_zup=True,
        merge_submeshes_for_collision=False,
        group_by_material=False,
        convexify=False,
        decimate=False,
        align=False,
    )
    mesh_scene.add_entity(
        mesh_morph,
        material=gs.materials.Rigid(
            rho=1000.0,
        ),
    )
    mesh_scene.build()
    return mesh_scene


@pytest.fixture
def xml_path(request, tmp_path, model_name):
    """Create a temporary MJCF/XML file from the fixture."""
    mjcf = request.getfixturevalue(model_name)
    xml_tree = ET.ElementTree(mjcf)
    file_name = f"{model_name}.xml"
    file_path = str(tmp_path / file_name)
    xml_tree.write(file_path, encoding="utf-8", xml_declaration=True)
    return file_path


@pytest.fixture
def scale():
    return 1.0


@pytest.fixture
def fixed():
    return None


@pytest.fixture
def usd_scene(request, model_name, scale, fixed):
    """Build a USD scene from the USD file provided by the fixture named 'model_name'."""
    return build_usd_scene(request.getfixturevalue(model_name), scale=scale, fixed=fixed)


@pytest.fixture(scope="session")
def emissive_material_variants_glb(asset_tmp_path):
    """Path to a GLB with three materials, each on distinct base/emissive texCoord sets: a base-color atlas (red) on
    texCoord 0 with an emissive atlas on texCoord 1, a flat base color with an emissive atlas on texCoord 1, and a
    KHR_materials_unlit material whose red base atlas stands in for the unlit imagery. The red base atlas is index 0."""
    images = []
    for color in (np.array([220, 30, 30], np.uint8), np.array([30, 220, 30], np.uint8)):
        buffer = io.BytesIO()
        Image.fromarray(np.broadcast_to(color, (8, 8, 3)).copy()).save(buffer, format="PNG")
        images.append(buffer.getvalue())

    blob = b""
    buffer_views = []
    for data in images:
        blob += b"\x00" * ((4 - len(blob) % 4) % 4)
        buffer_views.append(pygltflib.BufferView(buffer=0, byteOffset=len(blob), byteLength=len(data)))
        blob += data

    gltf = pygltflib.GLTF2(
        materials=[
            pygltflib.Material(
                pbrMetallicRoughness=pygltflib.PbrMetallicRoughness(
                    baseColorTexture=pygltflib.TextureInfo(index=0, texCoord=0)
                ),
                emissiveTexture=pygltflib.TextureInfo(index=1, texCoord=1),
                emissiveFactor=[1.0, 1.0, 1.0],
            ),
            pygltflib.Material(
                pbrMetallicRoughness=pygltflib.PbrMetallicRoughness(baseColorFactor=[0.5, 0.5, 0.5, 1.0]),
                emissiveTexture=pygltflib.TextureInfo(index=1, texCoord=1),
                emissiveFactor=[1.0, 1.0, 1.0],
            ),
            pygltflib.Material(
                pbrMetallicRoughness=pygltflib.PbrMetallicRoughness(
                    baseColorTexture=pygltflib.TextureInfo(index=0, texCoord=0)
                ),
                extensions={"KHR_materials_unlit": {}},
            ),
        ],
        textures=[pygltflib.Texture(source=0), pygltflib.Texture(source=1)],
        images=[pygltflib.Image(bufferView=i, mimeType="image/png") for i in range(2)],
        bufferViews=buffer_views,
        buffers=[pygltflib.Buffer(byteLength=len(blob))],
    )
    gltf.set_binary_blob(blob)
    path = asset_tmp_path / "emissive_material_variants.glb"
    gltf.save_binary(str(path))
    return str(path)


@pytest.fixture
def material_mjcf(tmp_path):
    """Generate an MJCF model with materials and geom-level colors."""
    mjcf = ET.Element("mujoco", model="materials")
    default = ET.SubElement(mjcf, "default")
    ET.SubElement(default, "joint", armature="0.0")

    # Define materials with different properties (at top level, not in default)
    asset = ET.SubElement(mjcf, "asset")
    ET.SubElement(
        asset,
        "material",
        name="red_material",
        rgba="1.0 0.0 0.0 0.6",
        specular="0.5",
        shininess="0.3",
    )

    worldbody = ET.SubElement(mjcf, "worldbody")
    floor = ET.SubElement(worldbody, "body", name="/worldbody/floor")
    ET.SubElement(floor, "geom", type="plane", pos="0. 0. 0.", size="40. 40. 40.")

    # Box with red material (material-level rgba)
    box1 = ET.SubElement(worldbody, "body", name="/worldbody/box1", pos="-0.3 0. 0.3")
    ET.SubElement(
        box1,
        "geom",
        type="box",
        size="0.2 0.2 0.2",
        pos="0. 0. 0.",
        material="red_material",
        contype="0",
        conaffinity="0",
    )
    ET.SubElement(box1, "joint", name="/worldbody/box1_joint", type="free")

    # Box with geom-level rgba (no material, tests geom-level color)
    box2 = ET.SubElement(worldbody, "body", name="/worldbody/box2", pos="0.0 0. 0.6")
    ET.SubElement(
        box2,
        "geom",
        type="box",
        size="0.2 0.2 0.2",
        pos="0. 0. 0.",
        rgba="0.0 1.0 0.0 1.0",
        contype="0",
        conaffinity="0",
    )
    ET.SubElement(box2, "joint", name="/worldbody/box2_joint", type="free")

    # Write to temporary file
    xml_tree = ET.ElementTree(mjcf)
    file_path = str(tmp_path / "material_mjcf.xml")
    xml_tree.write(file_path, encoding="utf-8", xml_declaration=True)
    return file_path


@pytest.fixture
def normals_mjcf(tmp_path):
    # MuJoCo packs vertices and normals in separately-addressed blocks. The first mesh has a different number of
    # vertices than normals (a flat-shaded bipyramid: shared positions, per-face normals), which shifts the second
    # mesh's normal block away from its vertex block. The second mesh (a smooth icosphere, whose normals are exactly
    # radial) is the one whose normals must be read at the correct offset and routed through the per-face normal
    # indices, so it is the comparison target.
    n_sides = 24
    radius, half_height = 0.3, 0.4
    angles = 2.0 * np.pi * np.arange(n_sides) / n_sides
    ring = np.stack([radius * np.cos(angles), radius * np.sin(angles), np.zeros(n_sides)], axis=1)
    bipyr_verts = np.vstack([ring, (0.0, 0.0, half_height), (0.0, 0.0, -half_height)])
    top_idx, bot_idx = n_sides, n_sides + 1
    bipyr_faces = []
    for k in range(n_sides):
        a, b = k, (k + 1) % n_sides
        bipyr_faces.append((a, b, top_idx))
        bipyr_faces.append((b, a, bot_idx))
    bipyr_normals = []
    for a, b, c in bipyr_faces:
        normal = np.cross(bipyr_verts[b] - bipyr_verts[a], bipyr_verts[c] - bipyr_verts[a])
        bipyr_normals.append(normal / np.linalg.norm(normal))
    obj_lines = [f"v {v[0]} {v[1]} {v[2]}" for v in bipyr_verts]
    obj_lines += [f"vn {n[0]} {n[1]} {n[2]}" for n in bipyr_normals]
    for i, (a, b, c) in enumerate(bipyr_faces):
        obj_lines.append(f"f {a + 1}//{i + 1} {b + 1}//{i + 1} {c + 1}//{i + 1}")
    bipyr_path = tmp_path / "bipyr.obj"
    bipyr_path.write_text("\n".join(obj_lines) + "\n")

    ico_path = tmp_path / "ico.obj"
    trimesh.creation.icosphere(radius=0.3, subdivisions=3).export(str(ico_path), include_normals=True)

    mjcf = ET.Element("mujoco", model="normals")
    asset = ET.SubElement(mjcf, "asset")
    ET.SubElement(asset, "mesh", name="bipyr", file=str(bipyr_path))
    ET.SubElement(asset, "mesh", name="ico", file=str(ico_path))
    worldbody = ET.SubElement(mjcf, "worldbody")
    body = ET.SubElement(worldbody, "body", name="/worldbody/obj")
    ET.SubElement(body, "freejoint")
    ET.SubElement(body, "geom", type="mesh", mesh="bipyr", contype="0", conaffinity="0")
    ET.SubElement(body, "geom", type="mesh", mesh="ico", contype="0", conaffinity="0")

    file_path = str(tmp_path / "normals_mjcf.xml")
    ET.ElementTree(mjcf).write(file_path, encoding="utf-8", xml_declaration=True)
    return file_path, str(ico_path)


@pytest.fixture
def textured_mjcf():
    mjcf = ET.Element("mujoco", model="texture_mapping")
    default = ET.SubElement(mjcf, "default")
    ET.SubElement(default, "geom", contype="0", conaffinity="0")

    asset = ET.SubElement(mjcf, "asset")
    ET.SubElement(asset, "texture", name="checker", type="2d", builtin="checker", width="8", height="8")
    ET.SubElement(asset, "material", name="repeated", texture="checker", texrepeat="2 2.5")
    ET.SubElement(asset, "material", name="uniform", texture="checker", texrepeat="2 2.5", texuniform="true")

    MESH_VERTICES = "-1 -2 -3  1 -2 -3  1 2 -3  -1 2 -3  -1 -2 3  1 -2 3  1 2 3  -1 2 3"
    MESH_FACES = "0 2 1  0 3 2  4 5 6  4 6 7  0 1 5  0 5 4  3 7 6  3 6 2  0 4 7  0 7 3  1 2 6  1 6 5"
    ET.SubElement(asset, "mesh", name="plain_mesh", vertex=MESH_VERTICES, face=MESH_FACES)
    ET.SubElement(asset, "mesh", name="plain_mesh_scaled", scale="3 3 3", vertex=MESH_VERTICES, face=MESH_FACES)
    ET.SubElement(
        asset,
        "mesh",
        name="explicit_mesh",
        vertex="-1 -1 -1  1 -1 -1  0 1 -1  0 0 1",
        texcoord="0.125 0.25  0.75 0.25  0.5 0.875  0.625 0.75",
        face="0 2 1  0 1 3  1 2 3  2 0 3",
    )

    worldbody = ET.SubElement(mjcf, "worldbody")
    ET.SubElement(worldbody, "geom", name="plane_repeated", type="plane", size="3 5 0.1", material="repeated")
    ET.SubElement(worldbody, "geom", name="plane_uniform", type="plane", size="3 5 0.1", material="uniform")
    ET.SubElement(worldbody, "geom", name="plane_infinite", type="plane", size="0 0 0.1", material="uniform")
    ET.SubElement(worldbody, "geom", name="sphere_repeated", type="sphere", size="2", material="repeated")
    ET.SubElement(worldbody, "geom", name="ellipsoid_uniform", type="ellipsoid", size="2 3 4", material="uniform")
    ET.SubElement(worldbody, "geom", name="capsule_repeated", type="capsule", size="2 3", material="repeated")
    ET.SubElement(worldbody, "geom", name="cylinder_repeated", type="cylinder", size="2 3", material="repeated")
    ET.SubElement(worldbody, "geom", name="box_uniform", type="box", size="2 3 4", material="uniform")
    ET.SubElement(worldbody, "geom", name="mesh_generated", type="mesh", mesh="plain_mesh", material="repeated")
    ET.SubElement(
        worldbody,
        "geom",
        name="mesh_generated_scaled",
        type="mesh",
        mesh="plain_mesh_scaled",
        material="repeated",
    )
    ET.SubElement(worldbody, "geom", name="box_fitted_repeated", type="box", mesh="plain_mesh", material="repeated")
    ET.SubElement(
        worldbody,
        "geom",
        name="box_fitted_repeated_scaled",
        type="box",
        mesh="plain_mesh_scaled",
        material="repeated",
    )
    ET.SubElement(worldbody, "geom", name="box_fitted", type="box", mesh="plain_mesh", material="uniform")
    ET.SubElement(worldbody, "geom", name="mesh_explicit_a", type="mesh", mesh="explicit_mesh", material="uniform")
    ET.SubElement(worldbody, "geom", name="mesh_explicit_b", type="mesh", mesh="explicit_mesh", material="uniform")

    return ET.tostring(mjcf, encoding="unicode")


@pytest.fixture(scope="session")
def all_primitives_mjcf():
    """Generate an MJCF model with various geometric primitives on a plane."""
    mjcf = ET.Element("mujoco", model="primitives")

    worldbody = ET.SubElement(mjcf, "worldbody")
    floor = ET.SubElement(worldbody, "body", name="/worldbody/floor")
    ET.SubElement(floor, "geom", type="plane", pos="0. 0. 0.", size="40. 40. 40.")

    # Box
    box = ET.SubElement(worldbody, "body", name="/worldbody/box", pos="-0.6 0. 0.3")
    ET.SubElement(box, "geom", type="box", size="0.2 0.2 0.2", pos="0. 0. 0.")
    ET.SubElement(box, "joint", name="/worldbody/box_joint", type="free")

    # Cylinder
    cylinder = ET.SubElement(worldbody, "body", name="/worldbody/cylinder", pos="-0.2 0. 0.3")
    ET.SubElement(cylinder, "geom", type="cylinder", size="0.15 0.2", pos="0. 0. 0.")
    ET.SubElement(cylinder, "joint", name="/worldbody/cylinder_joint", type="free")

    # Capsule
    capsule = ET.SubElement(worldbody, "body", name="/worldbody/capsule", pos="0.2 0. 0.3")
    ET.SubElement(capsule, "geom", type="capsule", size="0.15 0.2", pos="0. 0. 0.")
    ET.SubElement(capsule, "joint", name="/worldbody/capsule_joint", type="free")

    # Sphere
    sphere = ET.SubElement(worldbody, "body", name="/worldbody/sphere", pos="0.6 0. 0.3")
    ET.SubElement(sphere, "geom", type="sphere", size="0.2", pos="0. 0. 0.")
    ET.SubElement(sphere, "joint", name="/worldbody/sphere_joint", type="free")

    return mjcf


@pytest.fixture(scope="session")
def all_primitives_usd(asset_tmp_path, all_primitives_mjcf: ET.ElementTree):
    """Generate a USD file equivalent to the MJCF all_primitives_mjcf fixture."""
    # Extract data from MJCF XML structure
    worldbody = all_primitives_mjcf.find("worldbody")

    # Floor: body contains a geom with pos and size
    floor_body = worldbody.find("body[@name='/worldbody/floor']")
    floor_geom = floor_body.find("geom[@type='plane']")
    floor_pos_str = floor_geom.get("pos", "0. 0. 0.")
    floor_pos = to_array(floor_pos_str)
    floor_size = to_array(floor_geom.get("size", "40. 40. 40."))

    # Box: body has pos, geom inside has size
    box_body = worldbody.find("body[@name='/worldbody/box']")
    box_pos_str = box_body.get("pos", "0. 0. 0.")
    box_pos = to_array(box_pos_str)
    box_geom = box_body.find("geom[@type='box']")
    box_size_str = box_geom.get("size", "0.2 0.2 0.2")
    box_size = to_array(box_size_str)

    # Cylinder: body has pos, geom has size (radius, half-height)
    cylinder_body = worldbody.find("body[@name='/worldbody/cylinder']")
    cylinder_pos_str = cylinder_body.get("pos", "0. 0. 0.")
    cylinder_pos = to_array(cylinder_pos_str)
    cylinder_geom = cylinder_body.find("geom[@type='cylinder']")
    cylinder_size_str = cylinder_geom.get("size", "0.15 0.2")
    cylinder_size = to_array(cylinder_size_str)
    cylinder_radius = cylinder_size[0]
    cylinder_half_height = cylinder_size[1]

    # Capsule: body has pos, geom has size (radius, half-height)
    capsule_body = worldbody.find("body[@name='/worldbody/capsule']")
    capsule_pos_str = capsule_body.get("pos", "0. 0. 0.")
    capsule_pos = to_array(capsule_pos_str)
    capsule_geom = capsule_body.find("geom[@type='capsule']")
    capsule_size_str = capsule_geom.get("size", "0.15 0.2")
    capsule_size = to_array(capsule_size_str)
    capsule_radius = capsule_size[0]
    capsule_half_height = capsule_size[1]

    # Sphere: body has pos, geom has size (radius)
    sphere_body = worldbody.find("body[@name='/worldbody/sphere']")
    sphere_pos_str = sphere_body.get("pos", "0. 0. 0.")
    sphere_pos = to_array(sphere_pos_str)
    sphere_geom = sphere_body.find("geom[@type='sphere']")
    sphere_size_str = sphere_geom.get("size", "0.2")
    sphere_radius = float(sphere_size_str) if isinstance(sphere_size_str, str) else sphere_size_str[0]

    # Create temporary USD file
    usd_file = str(asset_tmp_path / "all_primitives.usda")

    # Create USD stage
    stage = Usd.Stage.CreateNew(usd_file)
    UsdGeom.SetStageUpAxis(stage, "Z")
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    # Create root prim
    root_prim = stage.DefinePrim("/worldbody", "Xform")
    stage.SetDefaultPrim(root_prim)

    # Create floor plane (fixed, collision-only)
    # In MJCF: plane at floor_pos with size floor_size
    # In USD: Create a plane geometry with CollisionAPI (fixed rigid body)
    floor = UsdGeom.Plane.Define(stage, "/worldbody/floor")
    floor.GetAxisAttr().Set("Z")
    floor.AddTranslateOp().Set(Gf.Vec3d(floor_pos[0], floor_pos[1], floor_pos[2]))
    # MJCF plane size - the third value is typically ignored for plane
    # For USD Plane, we use width and length
    floor.GetWidthAttr().Set(floor_size[0] * 2)  # size[0] * 2
    floor.GetLengthAttr().Set(floor_size[1] * 2)  # size[1] * 2

    # Make it a fixed collision-only rigid body
    UsdPhysics.CollisionAPI.Apply(floor.GetPrim())
    # No RigidBodyAPI means it's kinematic/fixed

    # Create box (free rigid body)
    # In MJCF: box at box_pos with size box_size (half-extent), free joint
    box = UsdGeom.Cube.Define(stage, "/worldbody/box")
    box.AddTranslateOp().Set(Gf.Vec3d(box_pos[0], box_pos[1], box_pos[2]))
    # MJCF size is half-extent, USD size is full edge length
    # So we need to multiply by 2
    box.GetSizeAttr().Set(box_size[0] * 2.0)
    box_rigid = UsdPhysics.RigidBodyAPI.Apply(box.GetPrim())
    box_rigid.GetKinematicEnabledAttr().Set(False)

    # Create free joint for box
    free_joint_prim = UsdPhysics.Joint.Define(stage, "/worldbody/box_joint")
    free_joint_prim.CreateBody0Rel().SetTargets([root_prim.GetPath()])
    free_joint_prim.CreateBody1Rel().SetTargets([box.GetPrim().GetPath()])
    free_joint_prim.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    free_joint_prim.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))

    # Create cylinder (free rigid body)
    # In MJCF: cylinder size is (radius, half-height)
    # In USD: cylinder has radius and height (full height)
    cylinder = UsdGeom.Cylinder.Define(stage, "/worldbody/cylinder")
    cylinder.AddTranslateOp().Set(Gf.Vec3d(cylinder_pos[0], cylinder_pos[1], cylinder_pos[2]))
    cylinder.GetRadiusAttr().Set(cylinder_radius)
    cylinder.GetHeightAttr().Set(cylinder_half_height * 2.0)  # Convert half-height to full height
    cylinder.GetAxisAttr().Set("Z")
    cylinder_rigid = UsdPhysics.RigidBodyAPI.Apply(cylinder.GetPrim())
    cylinder_rigid.GetKinematicEnabledAttr().Set(False)

    # Create free joint for cylinder
    cylinder_joint_prim = UsdPhysics.Joint.Define(stage, "/worldbody/cylinder_joint")
    cylinder_joint_prim.CreateBody0Rel().SetTargets([root_prim.GetPath()])
    cylinder_joint_prim.CreateBody1Rel().SetTargets([cylinder.GetPrim().GetPath()])
    cylinder_joint_prim.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    cylinder_joint_prim.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))

    # Create capsule (free rigid body)
    # In MJCF: capsule size is (radius, half-height)
    # In USD: capsule has radius and height (full height)
    capsule = UsdGeom.Capsule.Define(stage, "/worldbody/capsule")
    capsule.AddTranslateOp().Set(Gf.Vec3d(capsule_pos[0], capsule_pos[1], capsule_pos[2]))
    capsule.GetRadiusAttr().Set(capsule_radius)
    capsule.GetHeightAttr().Set(capsule_half_height * 2.0)  # Convert half-height to full height
    capsule.GetAxisAttr().Set("Z")
    capsule_rigid = UsdPhysics.RigidBodyAPI.Apply(capsule.GetPrim())
    capsule_rigid.GetKinematicEnabledAttr().Set(False)

    # Create free joint for capsule
    capsule_joint_prim = UsdPhysics.Joint.Define(stage, "/worldbody/capsule_joint")
    capsule_joint_prim.CreateBody0Rel().SetTargets([root_prim.GetPath()])
    capsule_joint_prim.CreateBody1Rel().SetTargets([capsule.GetPrim().GetPath()])
    capsule_joint_prim.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    capsule_joint_prim.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))

    # Create sphere (free rigid body)
    # In MJCF: sphere size is radius
    # In USD: sphere has radius
    sphere = UsdGeom.Sphere.Define(stage, "/worldbody/sphere")
    sphere.AddTranslateOp().Set(Gf.Vec3d(sphere_pos[0], sphere_pos[1], sphere_pos[2]))
    sphere.GetRadiusAttr().Set(sphere_radius)
    sphere_rigid = UsdPhysics.RigidBodyAPI.Apply(sphere.GetPrim())
    sphere_rigid.GetKinematicEnabledAttr().Set(False)

    # Create free joint for sphere
    sphere_joint_prim = UsdPhysics.Joint.Define(stage, "/worldbody/sphere_joint")
    sphere_joint_prim.CreateBody0Rel().SetTargets([root_prim.GetPath()])
    sphere_joint_prim.CreateBody1Rel().SetTargets([sphere.GetPrim().GetPath()])
    sphere_joint_prim.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    sphere_joint_prim.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))

    stage.Save()

    return usd_file


@pytest.fixture(scope="session")
def all_joints_mjcf():
    """Generate an MJCF model with all joint types: prismatic, revolute, spherical, fixed, and free."""
    mjcf = ET.Element("mujoco", model="all_joints")

    worldbody = ET.SubElement(mjcf, "worldbody")
    floor = ET.SubElement(worldbody, "body", name="/worldbody/floor")
    ET.SubElement(floor, "geom", type="plane", pos="0. 0. 0.", size="40. 40. 40.")

    base = ET.SubElement(worldbody, "body", name="/worldbody/base", pos="0. 0. 0.1")
    ET.SubElement(base, "geom", type="box", size="0.1 0.1 0.1", pos="0. 0. 0.")

    # Prismatic joint branch
    prismatic_box = ET.SubElement(base, "body", name="/worldbody/base/prismatic_box", pos="-0.5 0. 0.2")
    ET.SubElement(prismatic_box, "geom", type="box", size="0.2 0.2 0.2", pos="0. 0. 0.")
    ET.SubElement(
        prismatic_box,
        "joint",
        name="/worldbody/base/prismatic_box_joint",
        type="slide",
        axis="0. 0. 1.",
        pos="0.03 -0.02 0.05",
        range="-0.1 0.4",
        stiffness="50.0",
        damping="5.0",
    )

    # Revolute joint branch
    # Add actuator for PD controller (maps to dofs_kp and dofs_kv)
    # The parser uses: dofs_kp = -gear * biasprm[1] * scale^3
    # So to get dofs_kp=120.0, we need biasprm[1] = -120.0 (with gear=1, scale=1)
    actuator = ET.SubElement(mjcf, "actuator")
    revolute_box = ET.SubElement(base, "body", name="/worldbody/base/revolute_box", pos="0. 0. 0.2")
    ET.SubElement(revolute_box, "geom", type="box", size="0.2 0.2 0.2", pos="0. 0. 0.")
    ET.SubElement(
        revolute_box,
        "joint",
        name="/worldbody/base/revolute_box_joint",
        type="hinge",
        axis="0. 0. 1.",
        pos="-0.04 0.06 -0.01",
        range="-45 45",
        stiffness="50.0",
        damping="5.0",
    )

    # Spherical joint branch
    spherical_box = ET.SubElement(base, "body", name="/worldbody/base/spherical_box", pos="0.5 0. 0.2")
    ET.SubElement(spherical_box, "geom", type="box", size="0.2 0.2 0.2", pos="0. 0. 0.")
    ET.SubElement(spherical_box, "joint", name="/worldbody/base/spherical_box_joint", type="ball")

    # Fixed joint branch (no joint element means fixed in MJCF)
    fixed_box = ET.SubElement(base, "body", name="/worldbody/base/fixed_box", pos="-0.5 0.5 0.2")
    ET.SubElement(fixed_box, "geom", type="box", size="0.2 0.2 0.2", pos="0. 0. 0.")
    # No joint element = fixed joint

    # Free joint branch (must be at top level in MJCF - directly under worldbody)
    free_box = ET.SubElement(worldbody, "body", name="/worldbody/free_box", pos="0.5 0.5 0.3")
    ET.SubElement(free_box, "geom", type="box", size="0.2 0.2 0.2", pos="0. 0. 0.")
    ET.SubElement(free_box, "joint", name="/worldbody/free_box_joint", type="free")

    # Add actuators for PD controllers (prismatic and revolute only)
    actuator = ET.SubElement(mjcf, "actuator")
    ET.SubElement(
        actuator,
        "general",
        name="/worldbody/base/prismatic_box_joint_actuator",
        joint="/worldbody/base/prismatic_box_joint",
        biastype="affine",
        gainprm="120.0 0 0",  # gainprm[0] must equal -biasprm[1] to avoid warning
        biasprm="0 -120.0 -12.0",  # biasprm format: [b0, b1, b2] where b1=kp, b2=kv (negated)
    )
    ET.SubElement(
        actuator,
        "general",
        name="/worldbody/base/revolute_box_joint_actuator",
        joint="/worldbody/base/revolute_box_joint",
        biastype="affine",
        gainprm="120.0 0 0",
        biasprm="0 -120.0 -12.0",
    )

    return mjcf


@pytest.fixture(scope="session")
def all_joints_usd(asset_tmp_path, all_joints_mjcf: ET.ElementTree, request):
    """Generate a USD file equivalent to the all joints MJCF fixture.

    Supports both with and without ArticulationRootAPI based on request.param.
    """
    # Get the use_articulation_root parameter from request.param if available
    use_articulation_root = getattr(request, "param", True)

    worldbody = all_joints_mjcf.find("worldbody")

    # Floor
    floor_body = worldbody.find("body[@name='/worldbody/floor']")
    floor_geom = floor_body.find("geom[@type='plane']")
    floor_pos_str = floor_geom.get("pos")
    floor_pos = to_array(floor_pos_str)
    floor_size = to_array(floor_geom.get("size", "40. 40. 40."))

    # Base
    base_body = worldbody.find("body[@name='/worldbody/base']")
    base_pos_str = base_body.get("pos")
    base_pos = to_array(base_pos_str)
    base_geom = base_body.find("geom[@type='box']")
    base_size_str = base_geom.get("size")
    base_size = to_array(base_size_str)

    # Prismatic box
    prismatic_box_body = base_body.find("body[@name='/worldbody/base/prismatic_box']")
    prismatic_box_pos_str = prismatic_box_body.get("pos")
    prismatic_box_pos = to_array(prismatic_box_pos_str)
    prismatic_box_geom = prismatic_box_body.find("geom[@type='box']")
    prismatic_box_size = to_array(prismatic_box_geom.get("size"))
    prismatic_joint = prismatic_box_body.find("joint[@name='/worldbody/base/prismatic_box_joint']")
    prismatic_range = to_array(prismatic_joint.get("range"))
    # MJCF anchors the joint in the child body frame; USD wants that same point in each of the two body frames.
    prismatic_anchor = to_array(prismatic_joint.get("pos"))

    # Revolute box
    revolute_box_body = base_body.find("body[@name='/worldbody/base/revolute_box']")
    revolute_box_pos_str = revolute_box_body.get("pos")
    revolute_box_pos = to_array(revolute_box_pos_str)
    revolute_box_geom = revolute_box_body.find("geom[@type='box']")
    revolute_box_size = to_array(revolute_box_geom.get("size"))
    revolute_joint = revolute_box_body.find("joint[@name='/worldbody/base/revolute_box_joint']")
    revolute_range = to_array(revolute_joint.get("range"))
    revolute_anchor = to_array(revolute_joint.get("pos"))

    # Spherical box
    spherical_box_body = base_body.find("body[@name='/worldbody/base/spherical_box']")
    spherical_box_pos_str = spherical_box_body.get("pos")
    spherical_box_pos = to_array(spherical_box_pos_str)
    spherical_box_geom = spherical_box_body.find("geom[@type='box']")
    spherical_box_size = to_array(spherical_box_geom.get("size"))

    # Fixed box (no joint in MJCF means fixed)
    fixed_box_body = base_body.find("body[@name='/worldbody/base/fixed_box']")
    fixed_box_pos_str = fixed_box_body.get("pos")
    fixed_box_pos = to_array(fixed_box_pos_str)
    fixed_box_geom = fixed_box_body.find("geom[@type='box']")
    fixed_box_size = to_array(fixed_box_geom.get("size"))

    # Free box (at top level in MJCF)
    free_box_body = worldbody.find("body[@name='/worldbody/free_box']")
    free_box_pos_str = free_box_body.get("pos")
    free_box_pos = to_array(free_box_pos_str)
    free_box_geom = free_box_body.find("geom[@type='box']")
    free_box_size = to_array(free_box_geom.get("size"))

    # Create temporary USD file with suffix based on ArticulationRootAPI usage
    suffix = "with_articulation_root" if use_articulation_root else "without_articulation_root"
    usd_file = str(asset_tmp_path / f"all_joints_{suffix}.usda")

    # Create USD stage
    stage = Usd.Stage.CreateNew(usd_file)
    UsdGeom.SetStageUpAxis(stage, "Z")
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    # Create root prim
    root_prim = stage.DefinePrim("/worldbody", "Xform")
    stage.SetDefaultPrim(root_prim)

    # Create floor plane (fixed, collision-only)
    floor = UsdGeom.Plane.Define(stage, "/worldbody/floor")
    floor.GetAxisAttr().Set("Z")
    floor.AddTranslateOp().Set(Gf.Vec3d(floor_pos[0], floor_pos[1], floor_pos[2]))
    floor.GetWidthAttr().Set(floor_size[0] * 2)
    floor.GetLengthAttr().Set(floor_size[1] * 2)
    UsdPhysics.CollisionAPI.Apply(floor.GetPrim())

    # Create base (fixed, collision-only)
    base = UsdGeom.Cube.Define(stage, "/worldbody/base")
    if use_articulation_root:
        UsdPhysics.ArticulationRootAPI.Apply(base.GetPrim())
    base.AddTranslateOp().Set(Gf.Vec3d(base_pos[0], base_pos[1], base_pos[2]))
    base.GetSizeAttr().Set(base_size[0] * 2.0)
    UsdPhysics.CollisionAPI.Apply(base.GetPrim())

    # Create prismatic box
    prismatic_box = UsdGeom.Cube.Define(stage, "/worldbody/base/prismatic_box")
    prismatic_box.AddTranslateOp().Set(Gf.Vec3d(prismatic_box_pos[0], prismatic_box_pos[1], prismatic_box_pos[2]))
    prismatic_box.GetSizeAttr().Set(prismatic_box_size[0] * 2.0)
    prismatic_box_rigid = UsdPhysics.RigidBodyAPI.Apply(prismatic_box.GetPrim())
    prismatic_box_rigid.GetKinematicEnabledAttr().Set(False)

    # Create prismatic joint
    prismatic_joint_prim = UsdPhysics.PrismaticJoint.Define(stage, "/worldbody/base/prismatic_box_joint")
    prismatic_joint_prim.CreateBody0Rel().SetTargets([base.GetPrim().GetPath()])
    prismatic_joint_prim.CreateBody1Rel().SetTargets([prismatic_box.GetPrim().GetPath()])
    prismatic_joint_prim.CreateAxisAttr().Set("Z")
    prismatic_joint_prim.CreateLowerLimitAttr().Set(prismatic_range[0])
    prismatic_joint_prim.CreateUpperLimitAttr().Set(prismatic_range[1])
    prismatic_joint_prim.CreateLocalPos0Attr().Set(Gf.Vec3f(*(prismatic_box_pos + prismatic_anchor)))
    prismatic_joint_prim.CreateLocalPos1Attr().Set(Gf.Vec3f(*prismatic_anchor))
    prismatic_joint_prim.GetPrim().CreateAttribute("linear:stiffness", Sdf.ValueTypeNames.Float).Set(50.0)
    prismatic_joint_prim.GetPrim().CreateAttribute("linear:damping", Sdf.ValueTypeNames.Float).Set(5.0)
    prismatic_drive_api = UsdPhysics.DriveAPI.Apply(prismatic_joint_prim.GetPrim(), "linear")
    prismatic_drive_api.CreateStiffnessAttr().Set(120.0)
    prismatic_drive_api.CreateDampingAttr().Set(12.0)

    # Create revolute box
    revolute_box = UsdGeom.Cube.Define(stage, "/worldbody/base/revolute_box")
    revolute_box.AddTranslateOp().Set(Gf.Vec3d(revolute_box_pos[0], revolute_box_pos[1], revolute_box_pos[2]))
    revolute_box.GetSizeAttr().Set(revolute_box_size[0] * 2.0)
    revolute_box_rigid = UsdPhysics.RigidBodyAPI.Apply(revolute_box.GetPrim())
    revolute_box_rigid.GetKinematicEnabledAttr().Set(False)

    # Create revolute joint
    revolute_joint_prim = UsdPhysics.RevoluteJoint.Define(stage, "/worldbody/base/revolute_box_joint")
    revolute_joint_prim.CreateBody0Rel().SetTargets([base.GetPrim().GetPath()])
    revolute_joint_prim.CreateBody1Rel().SetTargets([revolute_box.GetPrim().GetPath()])
    revolute_joint_prim.CreateAxisAttr().Set("Z")
    revolute_joint_prim.CreateLowerLimitAttr().Set(revolute_range[0])
    revolute_joint_prim.CreateUpperLimitAttr().Set(revolute_range[1])
    revolute_joint_prim.CreateLocalPos0Attr().Set(Gf.Vec3f(*(revolute_box_pos + revolute_anchor)))
    revolute_joint_prim.CreateLocalPos1Attr().Set(Gf.Vec3f(*revolute_anchor))
    revolute_joint_prim.GetPrim().CreateAttribute("stiffness", Sdf.ValueTypeNames.Float).Set(50.0)
    revolute_joint_prim.GetPrim().CreateAttribute("angular:damping", Sdf.ValueTypeNames.Float).Set(5.0)
    revolute_drive_api = UsdPhysics.DriveAPI.Apply(revolute_joint_prim.GetPrim(), "angular")
    revolute_drive_api.CreateStiffnessAttr().Set(120.0)
    revolute_drive_api.CreateDampingAttr().Set(12.0)

    # Create spherical box
    spherical_box = UsdGeom.Cube.Define(stage, "/worldbody/base/spherical_box")
    spherical_box.AddTranslateOp().Set(Gf.Vec3d(spherical_box_pos[0], spherical_box_pos[1], spherical_box_pos[2]))
    spherical_box.GetSizeAttr().Set(spherical_box_size[0] * 2.0)
    spherical_box_rigid = UsdPhysics.RigidBodyAPI.Apply(spherical_box.GetPrim())
    spherical_box_rigid.GetKinematicEnabledAttr().Set(False)

    # Create spherical joint
    spherical_joint_prim = UsdPhysics.SphericalJoint.Define(stage, "/worldbody/base/spherical_box_joint")
    spherical_joint_prim.CreateBody0Rel().SetTargets([base.GetPrim().GetPath()])
    spherical_joint_prim.CreateBody1Rel().SetTargets([spherical_box.GetPrim().GetPath()])
    spherical_joint_prim.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    spherical_joint_prim.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))

    # Create fixed box
    fixed_box = UsdGeom.Cube.Define(stage, "/worldbody/base/fixed_box")
    fixed_box.AddTranslateOp().Set(Gf.Vec3d(fixed_box_pos[0], fixed_box_pos[1], fixed_box_pos[2]))
    fixed_box.GetSizeAttr().Set(fixed_box_size[0] * 2.0)
    fixed_box_rigid = UsdPhysics.RigidBodyAPI.Apply(fixed_box.GetPrim())
    fixed_box_rigid.GetKinematicEnabledAttr().Set(False)

    # Create fixed joint
    fixed_joint_prim = UsdPhysics.FixedJoint.Define(stage, "/worldbody/base/fixed_box_joint")
    fixed_joint_prim.CreateBody0Rel().SetTargets([base.GetPrim().GetPath()])
    fixed_joint_prim.CreateBody1Rel().SetTargets([fixed_box.GetPrim().GetPath()])
    fixed_joint_prim.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    fixed_joint_prim.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))

    # Create free box (at top level, not under base)
    free_box = UsdGeom.Cube.Define(stage, "/worldbody/free_box")
    free_box.AddTranslateOp().Set(Gf.Vec3d(free_box_pos[0], free_box_pos[1], free_box_pos[2]))
    free_box.GetSizeAttr().Set(free_box_size[0] * 2.0)
    free_box_rigid = UsdPhysics.RigidBodyAPI.Apply(free_box.GetPrim())
    free_box_rigid.GetKinematicEnabledAttr().Set(False)

    free_joint_prim = UsdPhysics.Joint.Define(stage, "/worldbody/free_box_joint")
    free_joint_prim.CreateBody0Rel().SetTargets([root_prim.GetPath()])
    free_joint_prim.CreateBody1Rel().SetTargets([free_box.GetPrim().GetPath()])
    free_joint_prim.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    free_joint_prim.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))

    stage.Save()

    return usd_file


@pytest.fixture(scope="session")
def pure_rigid_usd(asset_tmp_path):
    """Create a minimal USD file with a single rigid body (cube) and no joints."""
    usd_file = str(asset_tmp_path / "pure_rigid.usda")
    stage = Usd.Stage.CreateNew(usd_file)
    UsdGeom.SetStageUpAxis(stage, "Z")
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    root_prim = stage.DefinePrim("/root", "Xform")
    stage.SetDefaultPrim(root_prim)

    cube = UsdGeom.Cube.Define(stage, "/root/body")
    cube.GetSizeAttr().Set(1.0)

    UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim())
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    UsdPhysics.MassAPI.Apply(cube.GetPrim())
    UsdPhysics.MassAPI(cube.GetPrim()).GetMassAttr().Set(1.0)

    stage.Save()
    return usd_file


@pytest.fixture(scope="session")
def collision_only_rigid_usd(asset_tmp_path):
    usd_file = str(asset_tmp_path / "collision_only_rigid.usda")
    stage = Usd.Stage.CreateNew(usd_file)
    UsdGeom.SetStageUpAxis(stage, "Z")
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    cube = UsdGeom.Cube.Define(stage, "/root")
    cube.GetSizeAttr().Set(1.0)
    stage.SetDefaultPrim(cube.GetPrim())
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())  # No RigidBodyAPI

    stage.Save()
    return usd_file


@pytest.fixture(scope="session")
def visual_collision_usd(asset_tmp_path):
    """Create a USD file mimicking Pan011 structure: separate Visual/Collision groups + invisible Sites.

    Structure:
        /root
        /root/Body  (RigidBodyAPI, MassAPI)
            /root/Body/Collisions  (purpose=guide)
                /root/Body/Collisions/Collider1  (Cube, CollisionAPI, purpose=guide)
                /root/Body/Collisions/Collider2  (Sphere, CollisionAPI, purpose=guide)
            /root/Body/Visuals
                /root/Body/Visuals/Visual1  (Cube)
            /root/Body/Sites  (visibility=invisible)
                /root/Body/Sites/site_marker  (Cube, purpose=guide, invisible)
    """
    usd_file = str(asset_tmp_path / "visual_collision.usda")
    stage = Usd.Stage.CreateNew(usd_file)
    UsdGeom.SetStageUpAxis(stage, "Z")
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    root_prim = stage.DefinePrim("/root", "Xform")
    stage.SetDefaultPrim(root_prim)

    # Rigid body
    body = stage.DefinePrim("/root/Body", "Xform")
    UsdPhysics.RigidBodyAPI.Apply(body)
    UsdPhysics.MassAPI.Apply(body)
    UsdPhysics.MassAPI(body).GetMassAttr().Set(1.0)

    # Collisions group (purpose=guide, like Pan011)
    collisions_xform = UsdGeom.Xform.Define(stage, "/root/Body/Collisions")
    collisions_xform.GetPrim().CreateAttribute("purpose", Sdf.ValueTypeNames.Token).Set("guide")

    col1 = UsdGeom.Cube.Define(stage, "/root/Body/Collisions/Collider1")
    col1.GetSizeAttr().Set(0.5)
    col1.GetPurposeAttr().Set("guide")
    UsdPhysics.CollisionAPI.Apply(col1.GetPrim())

    col2 = UsdGeom.Sphere.Define(stage, "/root/Body/Collisions/Collider2")
    col2.GetRadiusAttr().Set(0.3)
    col2.GetPurposeAttr().Set("guide")
    UsdPhysics.CollisionAPI.Apply(col2.GetPrim())

    # Visuals group
    UsdGeom.Xform.Define(stage, "/root/Body/Visuals")

    vis1 = UsdGeom.Cube.Define(stage, "/root/Body/Visuals/Visual1")
    vis1.GetSizeAttr().Set(1.0)

    # Sites group (invisible, like Pan011)
    sites_xform = UsdGeom.Xform.Define(stage, "/root/Body/Sites")
    UsdGeom.Imageable(sites_xform.GetPrim()).MakeInvisible()

    site_marker = UsdGeom.Cube.Define(stage, "/root/Body/Sites/site_marker")
    site_marker.GetSizeAttr().Set(0.1)
    site_marker.GetPurposeAttr().Set("guide")

    stage.Save()
    return usd_file


@pytest.fixture(scope="session")
def negative_scale_rigid_usd(asset_tmp_path):
    """Rigid body whose mesh prim uses negative xformOp:scale (reflection)."""
    usd_file = str(asset_tmp_path / "negative_scale_rigid.usda")
    stage = Usd.Stage.CreateNew(usd_file)
    UsdGeom.SetStageUpAxis(stage, "Z")
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    root_prim = stage.DefinePrim("/root", "Xform")
    stage.SetDefaultPrim(root_prim)

    mesh = UsdGeom.Cube.Define(stage, "/root/body")
    mesh.GetSizeAttr().Set(1.0)
    mesh.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.5))
    mesh.AddScaleOp().Set(Gf.Vec3d(-1.0, -1.0, -1.0))
    UsdPhysics.RigidBodyAPI.Apply(mesh.GetPrim())
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())

    stage.Save()
    return usd_file


@pytest.fixture(scope="session")
def nested_collision_joint_usd(asset_tmp_path):
    """Articulation where joints reference child collision prims under RigidBodyAPI parents."""
    usd_file = str(asset_tmp_path / "nested_collision_joint.usda")
    stage = Usd.Stage.CreateNew(usd_file)
    UsdGeom.SetStageUpAxis(stage, "Z")
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    root_prim = stage.DefinePrim("/root", "Xform")
    stage.SetDefaultPrim(root_prim)

    base = stage.DefinePrim("/root/base", "Xform")
    UsdPhysics.CollisionAPI.Apply(base)
    base_col = UsdGeom.Cube.Define(stage, "/root/base/collision")
    base_col.GetSizeAttr().Set(1.0)
    UsdPhysics.CollisionAPI.Apply(base_col.GetPrim())

    # Parent has RigidBodyAPI; joint references child collision prim (common USD authoring pattern).
    child = stage.DefinePrim("/root/child", "Xform")
    UsdPhysics.RigidBodyAPI.Apply(child)
    child_col = UsdGeom.Cube.Define(stage, "/root/child/collision")
    child_col.GetSizeAttr().Set(0.5)
    child_col.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.5))
    UsdPhysics.CollisionAPI.Apply(child_col.GetPrim())

    joint_prim = UsdPhysics.RevoluteJoint.Define(stage, "/root/child/revolute_joint")
    joint_prim.CreateBody0Rel().SetTargets([base_col.GetPrim().GetPath()])
    joint_prim.CreateBody1Rel().SetTargets([child_col.GetPrim().GetPath()])
    joint_prim.CreateAxisAttr().Set("Z")
    joint_prim.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    joint_prim.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))

    stage.Save()
    return usd_file


@pytest.fixture(scope="session")
def physics_material_usd(asset_tmp_path):
    """Three rigid bodies covering UsdPhysicsMaterialAPI parsing: full material (frictions,
    restitution, and density), explicitly authored zero dynamic friction, and MassAPI density with
    no explicit mass."""
    usd_file = str(asset_tmp_path / "physics_material.usda")
    stage = Usd.Stage.CreateNew(usd_file)
    UsdGeom.SetStageUpAxis(stage, "Z")
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    root_prim = stage.DefinePrim("/root", "Xform")
    stage.SetDefaultPrim(root_prim)

    for name in ("material_body", "frictionless_body", "density_body"):
        body = UsdGeom.Cube.Define(stage, f"/root/{name}")
        body.GetSizeAttr().Set(1.0)
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
        UsdPhysics.CollisionAPI.Apply(body.GetPrim())

    material = UsdShade.Material.Define(stage, "/root/PhysicsMaterial")
    material_api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    material_api.CreateStaticFrictionAttr(0.8)
    material_api.CreateDynamicFrictionAttr(0.6)
    material_api.CreateRestitutionAttr(0.4)
    material_api.CreateDensityAttr(300.0)
    UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath("/root/material_body")).Bind(
        material, bindingStrength=UsdShade.Tokens.weakerThanDescendants, materialPurpose="physics"
    )

    frictionless = UsdShade.Material.Define(stage, "/root/FrictionlessMaterial")
    UsdPhysics.MaterialAPI.Apply(frictionless.GetPrim()).CreateDynamicFrictionAttr(0.0)
    UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath("/root/frictionless_body")).Bind(
        frictionless, bindingStrength=UsdShade.Tokens.weakerThanDescendants, materialPurpose="physics"
    )

    UsdPhysics.MassAPI.Apply(stage.GetPrimAtPath("/root/density_body")).CreateDensityAttr(500.0)

    stage.Save()
    return usd_file


@pytest.fixture(scope="session")
def density_align_usd(asset_tmp_path):
    """Free bodies with per-geom authored densities: 'uniform_body' has two unit cubes at x=-0.5/+0.5
    with densities 100/300, 'mixed_body' authors a density on only one of its two cubes."""
    usd_file = str(asset_tmp_path / "density_align.usda")
    stage = Usd.Stage.CreateNew(usd_file)
    UsdGeom.SetStageUpAxis(stage, "Z")
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    root_prim = stage.DefinePrim("/root", "Xform")
    stage.SetDefaultPrim(root_prim)

    for body_name, densities in (("uniform_body", (100.0, 300.0)), ("mixed_body", (200.0, None))):
        body = stage.DefinePrim(f"/root/{body_name}", "Xform")
        UsdPhysics.RigidBodyAPI.Apply(body)
        for i_cube, (x, density) in enumerate(zip((-0.5, 0.5), densities)):
            cube = UsdGeom.Cube.Define(stage, f"/root/{body_name}/cube_{i_cube}")
            cube.GetSizeAttr().Set(1.0)
            cube.AddTranslateOp().Set(Gf.Vec3d(x, 0.0, 0.0))
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
            if density is not None:
                material = UsdShade.Material.Define(stage, f"/root/{body_name}/Material_{i_cube}")
                UsdPhysics.MaterialAPI.Apply(material.GetPrim()).CreateDensityAttr(density)
                UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(
                    material, bindingStrength=UsdShade.Tokens.weakerThanDescendants, materialPurpose="physics"
                )

    stage.Save()
    return usd_file


@pytest.fixture(scope="session")
def collision_approximation_usd(asset_tmp_path):
    """Mesh colliders exercising MeshCollisionAPI approximations: boundingCube and boundingSphere on
    a 2x1x1 box, convexHull and sdf on a concave (two disjoint boxes) mesh, none on a dense box,
    plus a dense box with no authored approximation at all."""
    usd_file = str(asset_tmp_path / "collision_approximation.usda")
    stage = Usd.Stage.CreateNew(usd_file)
    UsdGeom.SetStageUpAxis(stage, "Z")
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    root_prim = stage.DefinePrim("/root", "Xform")
    stage.SetDefaultPrim(root_prim)

    box = trimesh.creation.box(extents=(2.0, 1.0, 1.0))
    shifted_box = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    shifted_box.apply_translation((3.0, 0.0, 0.0))
    concave = trimesh.util.concatenate([trimesh.creation.box(extents=(1.0, 1.0, 1.0)), shifted_box])
    dense_box = trimesh.creation.box(extents=(1.0, 1.0, 1.0)).subdivide().subdivide().subdivide()
    for name, tmesh, approximation in (
        ("bounding_cube_body", box, "boundingCube"),
        ("bounding_sphere_body", box, "boundingSphere"),
        ("convex_hull_body", concave, "convexHull"),
        ("sdf_body", concave, "sdf"),
        ("raw_body", dense_box, "none"),
        ("plain_body", dense_box, None),
    ):
        mesh = UsdGeom.Mesh.Define(stage, f"/root/{name}")
        mesh.GetPointsAttr().Set([Gf.Vec3f(*map(float, v)) for v in tmesh.vertices])
        mesh.GetFaceVertexIndicesAttr().Set([int(i) for i in tmesh.faces.reshape(-1)])
        mesh.GetFaceVertexCountsAttr().Set([3] * len(tmesh.faces))
        UsdPhysics.RigidBodyAPI.Apply(mesh.GetPrim())
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        if approximation is not None:
            UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr(approximation)

    stage.Save()
    return usd_file


@pytest.fixture(scope="session")
def collision_filtering_usd(asset_tmp_path):
    """Two rigid bodies with in-model collision filtering: one via FilteredPairsAPI between its two
    collision cubes, one via CollisionGroups (A filters B, plus an ungrouped collider)."""
    usd_file = str(asset_tmp_path / "collision_filtering.usda")
    stage = Usd.Stage.CreateNew(usd_file)
    UsdGeom.SetStageUpAxis(stage, "Z")
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    root_prim = stage.DefinePrim("/root", "Xform")
    stage.SetDefaultPrim(root_prim)

    pair_body = stage.DefinePrim("/root/pair_body", "Xform")
    UsdPhysics.RigidBodyAPI.Apply(pair_body)
    for name, x in (("col_a", 0.0), ("col_b", 0.3)):
        cube = UsdGeom.Cube.Define(stage, f"/root/pair_body/{name}")
        cube.GetSizeAttr().Set(0.2)
        cube.AddTranslateOp().Set(Gf.Vec3d(x, 0.0, 0.1))
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath("/root/pair_body/col_a")).CreateFilteredPairsRel().AddTarget(
        "/root/pair_body/col_b"
    )

    group_body = stage.DefinePrim("/root/group_body", "Xform")
    UsdPhysics.RigidBodyAPI.Apply(group_body)
    UsdGeom.Xform(group_body).AddTranslateOp().Set(Gf.Vec3d(0.0, 1.0, 0.0))
    for name, x in (("col_c", 0.0), ("col_d", 0.3), ("col_e", 0.6)):
        cube = UsdGeom.Cube.Define(stage, f"/root/group_body/{name}")
        cube.GetSizeAttr().Set(0.2)
        cube.AddTranslateOp().Set(Gf.Vec3d(x, 0.0, 0.1))
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    group_a = UsdPhysics.CollisionGroup.Define(stage, "/root/collision_groups/A")
    group_b = UsdPhysics.CollisionGroup.Define(stage, "/root/collision_groups/B")
    group_a.GetCollidersCollectionAPI().CreateIncludesRel().AddTarget("/root/group_body/col_c")
    group_b.GetCollidersCollectionAPI().CreateIncludesRel().AddTarget("/root/group_body/col_d")
    group_a.CreateFilteredGroupsRel().AddTarget("/root/collision_groups/B")

    stage.Save()
    return usd_file


@pytest.fixture(scope="session")
def cross_entity_filtering_usd(asset_tmp_path):
    """Two separate rigid bodies (split into two entities by add_stage) with a FilteredPairsAPI
    relationship spanning them."""
    usd_file = str(asset_tmp_path / "cross_entity_filtering.usda")
    stage = Usd.Stage.CreateNew(usd_file)
    UsdGeom.SetStageUpAxis(stage, "Z")
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    root_prim = stage.DefinePrim("/root", "Xform")
    stage.SetDefaultPrim(root_prim)

    for name in ("body_a", "body_b"):
        cube = UsdGeom.Cube.Define(stage, f"/root/{name}")
        cube.GetSizeAttr().Set(0.2)
        UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim())
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath("/root/body_a")).CreateFilteredPairsRel().AddTarget(
        "/root/body_b"
    )

    stage.Save()
    return usd_file


@pytest.fixture(scope="session")
def oriented_capsule_usd(asset_tmp_path):
    usd_file = str(asset_tmp_path / "oriented_capsule.usda")
    stage = Usd.Stage.CreateNew(usd_file)
    UsdGeom.SetStageUpAxis(stage, "Z")
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    root_prim = stage.DefinePrim("/root", "Xform")
    stage.SetDefaultPrim(root_prim)

    body = stage.DefinePrim("/root/body", "Xform")
    UsdPhysics.RigidBodyAPI.Apply(body)
    UsdPhysics.MassAPI.Apply(body)
    UsdPhysics.MassAPI(body).GetMassAttr().Set(1.0)

    capsule = UsdGeom.Capsule.Define(stage, "/root/body/capsule")
    capsule.GetRadiusAttr().Set(0.08)
    capsule.GetHeightAttr().Set(0.4)
    capsule.GetAxisAttr().Set("X")
    UsdPhysics.CollisionAPI.Apply(capsule.GetPrim())

    stage.Save()
    return usd_file


@pytest.fixture(scope="session")
def usdz_packaged_texture_usd(asset_tmp_path):
    """Stage referencing a .usdz package whose mesh material samples a texture packed in the archive, so the
    texture resolves to a package-internal path (e.g. 'packaged_mesh.usdz[usdz_texture.png]').

    Returns the stage file path and the expected texture pixels.
    """
    texture_image = np.arange(4 * 4 * 3, dtype=np.uint8).reshape((4, 4, 3))
    Image.fromarray(texture_image).save(str(asset_tmp_path / "usdz_texture.png"))

    packaged_file = str(asset_tmp_path / "packaged_mesh.usda")
    stage = Usd.Stage.CreateNew(packaged_file)
    UsdGeom.SetStageUpAxis(stage, "Z")
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root_prim = stage.DefinePrim("/root", "Xform")
    stage.SetDefaultPrim(root_prim)

    mesh = UsdGeom.Mesh.Define(stage, "/root/mesh")
    mesh.GetPointsAttr().Set([Gf.Vec3f(0, 0, 0), Gf.Vec3f(1, 0, 0), Gf.Vec3f(0, 1, 0), Gf.Vec3f(1, 1, 0)])
    mesh.GetFaceVertexIndicesAttr().Set([0, 1, 2, 1, 3, 2])
    mesh.GetFaceVertexCountsAttr().Set([3, 3])
    uv_primvar = UsdGeom.PrimvarsAPI(mesh.GetPrim()).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
    )
    uv_primvar.Set([Gf.Vec2f(0, 0), Gf.Vec2f(1, 0), Gf.Vec2f(0, 1), Gf.Vec2f(1, 1)])
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())

    material = UsdShade.Material.Define(stage, "/root/material")
    pbr_shader = UsdShade.Shader.Define(stage, "/root/material/pbr")
    pbr_shader.CreateIdAttr("UsdPreviewSurface")
    texture_shader = UsdShade.Shader.Define(stage, "/root/material/texture")
    texture_shader.CreateIdAttr("UsdUVTexture")
    texture_shader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set("./usdz_texture.png")
    st_reader = UsdShade.Shader.Define(stage, "/root/material/st_reader")
    st_reader.CreateIdAttr("UsdPrimvarReader_float2")
    st_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    texture_shader.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st_reader.ConnectableAPI(), "result")
    pbr_shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        texture_shader.ConnectableAPI(), "rgb"
    )
    material.CreateSurfaceOutput().ConnectToSource(pbr_shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    stage.Save()

    usdz_file = str(asset_tmp_path / "packaged_mesh.usdz")
    assert UsdUtils.CreateNewUsdzPackage(packaged_file, usdz_file)

    scene_file = str(asset_tmp_path / "usdz_reference_scene.usda")
    scene_stage = Usd.Stage.CreateNew(scene_file)
    UsdGeom.SetStageUpAxis(scene_stage, "Z")
    UsdGeom.SetStageMetersPerUnit(scene_stage, 1.0)
    world_prim = scene_stage.DefinePrim("/world", "Xform")
    scene_stage.SetDefaultPrim(world_prim)
    scene_stage.DefinePrim("/world/asset", "Xform").GetReferences().AddReference("./packaged_mesh.usdz")
    scene_stage.Save()

    return scene_file, texture_image

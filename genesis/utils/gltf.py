from io import BytesIO
from urllib import request

import DracoPy
import numpy as np
import pygltflib
import trimesh
from PIL import Image

import genesis as gs

from . import mesh as mu
from . import geom as gu


ctype_to_numpy = {
    5120: np.int8,  # BYTE
    5121: np.uint8,  # UNSIGNED_BYTE
    5122: np.int16,  # SHORT
    5123: np.uint16,  # UNSIGNED_SHORT
    5124: np.int32,  # INT
    5125: np.uint32,  # UNSIGNED_INT
    5126: np.float32,  # FLOAT
}

type_to_count = {
    "SCALAR": (1, []),
    "VEC2": (2, [2]),
    "VEC3": (3, [3]),
    "VEC4": (4, [4]),
    "MAT2": (4, [2, 2]),
    "MAT3": (9, [3, 3]),
    "MAT4": (16, [4, 4]),
}

alpha_modes = {
    "OPAQUE": 0,
    "MASK": 1,
    "BLEND": 2,
}


def uri_to_PIL(data_uri):
    with request.urlopen(data_uri) as response:
        data = response.read()
    return BytesIO(data)


def get_glb_bufferview_data(glb, buffer_view):
    buffer = glb.buffers[buffer_view.buffer]
    return glb.get_data_from_buffer_uri(buffer.uri)


def get_glb_data_from_accessor(glb, accessor_index):
    accessor = glb.accessors[accessor_index]
    buffer_view = glb.bufferViews[accessor.bufferView]
    buffer_data = get_glb_bufferview_data(glb, buffer_view)

    data_type, data_ctype, count = accessor.type, accessor.componentType, accessor.count
    num_components = type_to_count[data_type][0]
    dtype = ctype_to_numpy[data_ctype]
    itemsize = np.dtype(dtype).itemsize

    # Extract data considering byteStride
    byte_offset = (buffer_view.byteOffset or 0) + (accessor.byteOffset or 0)
    byte_stride = buffer_view.byteStride

    if not byte_stride or byte_stride == num_components * itemsize:
        # Data is tightly packed
        byte_length = count * num_components * itemsize
        data = buffer_data[byte_offset : byte_offset + byte_length]
        array = np.frombuffer(data, dtype=dtype)
    else:
        # Data is interleaved
        array = np.zeros((count, num_components), dtype=dtype)
        for i in range(count):
            start = byte_offset + i * byte_stride
            end = start + num_components * itemsize
            data_slice = buffer_data[start:end]
            array[i] = np.frombuffer(data_slice, dtype=dtype, count=num_components)

    return array.reshape((count, *type_to_count[data_type][1]))


def get_glb_image(glb, image_index, image_type=None):
    if image_index is not None:
        image = Image.open(uri_to_PIL(glb.images[image_index].uri))
        if image_type is not None:
            image = image.convert(image_type)
        return mu.PIL_to_array(image)
    return None


def parse_glb_material(glb, material_index, surface):
    # parse images
    color_texture = None
    opacity_texture = None
    roughness_texture = None
    metallic_texture = None
    normal_texture = None
    emissive_texture = None

    alpha_cutoff = None
    double_sided = None
    ior = None
    uvs_used = 0

    material = glb.materials[material_index]
    double_sided = material.doubleSided

    # parse normal map
    if material.normalTexture is not None:
        texture = glb.textures[material.normalTexture.index]
        if material.normalTexture.texCoord is not None:
            uvs_used = material.normalTexture.texCoord
        normal_image = get_glb_image(glb, texture.source)
        if normal_image is not None:
            normal_texture = mu.create_texture(normal_image, None, "linear")

    # TODO: Parse occlusion
    if material.occlusionTexture is not None:
        texture = glb.textures[material.occlusionTexture.index]
        if material.occlusionTexture.texCoord is not None:
            uvs_used = material.occlusionTexture.texCoord
        occlusion_image = get_glb_image(glb, texture.source)
        if occlusion_image is not None:
            occlusion_texture = mu.create_texture(occlusion_image, None, "linear")

    # parse alpha mode
    alpha_cutoff = mu.adjust_alpha_cutoff(alpha_cutoff, alpha_modes[material.alphaMode])

    # parse pbr roughness and metallic
    if material.pbrMetallicRoughness is not None:
        pbr_texture = material.pbrMetallicRoughness

        # parse metallic and roughness
        roughness_image = None
        metallic_image = None
        if pbr_texture.metallicRoughnessTexture is not None:
            texture = glb.textures[pbr_texture.metallicRoughnessTexture.index]
            if pbr_texture.metallicRoughnessTexture.texCoord is not None:
                uvs_used = pbr_texture.metallicRoughnessTexture.texCoord

            combined_image = get_glb_image(glb, texture.source)
            if combined_image is not None:
                if combined_image.ndim == 2:
                    roughness_image = combined_image
                else:
                    roughness_image = combined_image[:, :, 1]  # G for roughness
                    metallic_image = combined_image[:, :, 2]  # B for metallic
                    # metallic_image = np.array(bands[0])     # R for metallic????

        metallic_factor = None
        if pbr_texture.metallicFactor is not None:
            metallic_factor = pbr_texture.metallicFactor

        roughness_factor = None
        if pbr_texture.roughnessFactor is not None:
            roughness_factor = pbr_texture.roughnessFactor

        metallic_texture = mu.create_texture(metallic_image, metallic_factor, "linear")
        roughness_texture = mu.create_texture(roughness_image, roughness_factor, "linear")

        # Check if material has a base color texture
        color_image = None
        if pbr_texture.baseColorTexture is not None:
            texture = glb.textures[pbr_texture.baseColorTexture.index]
            if pbr_texture.baseColorTexture.texCoord is not None:
                uvs_used = pbr_texture.baseColorTexture.texCoord
            if "KHR_texture_basisu" in texture.extensions:
                gs.logger.warning(
                    f"Mesh file `{glb.path}` uses 'KHR_texture_basisu' extension for supercompression of texture "
                    "images, which is unsupported. Ignoring texture."
                )
            color_image = get_glb_image(glb, texture.source, "RGBA")

        # parse color
        color_factor = None
        if pbr_texture.baseColorFactor is not None:
            color_factor = np.array(pbr_texture.baseColorFactor, dtype=np.float32)

        color_texture = mu.create_texture(color_image, color_factor, "srgb")

    elif "KHR_materials_pbrSpecularGlossiness" in material.extensions:
        extension_material = material.extensions["KHR_materials_pbrSpecularGlossiness"]
        color_image = None
        if "diffuseTexture" in extension_material:
            texture = extension_material["diffuseTexture"]
            if texture.get("texCoord") is not None:
                uvs_used = texture["texCoord"]
            color_image = get_glb_image(glb, texture.get("index"), "RGBA")

        color_factor = None
        if "diffuseFactor" in extension_material:
            color_factor = np.array(extension_material["diffuseFactor"], dtype=np.float32)

        color_texture = mu.create_texture(color_image, color_factor, "srgb")
        material.extensions.pop("KHR_materials_pbrSpecularGlossiness")

    if color_texture is not None:
        opacity_texture = color_texture.check_dim(3)
        if opacity_texture is not None:
            opacity_texture.apply_cutoff(alpha_cutoff)

    if "KHR_materials_unlit" in material.extensions:
        # No unlit material implemented in renderers, so surface the base color through emissive and give the base a
        # black factor. get_rgba then falls back to that emissive as the albedo, instead of the white base that
        # update_texture installs for an absent color, which would otherwise hide the unlit imagery.
        if color_texture is not None:
            emissive_texture = color_texture
            color_texture = mu.create_texture(None, (0.0, 0.0, 0.0), "srgb")
        material.extensions.pop("KHR_materials_unlit")
    else:
        # parse emissive
        emissive_image = None
        if material.emissiveTexture is not None:
            texture = glb.textures[material.emissiveTexture.index]
            # The single baked UV set follows whichever texture actually samples it. The base color owns it only
            # when it is an atlas that requires UVs and is not black; otherwise (absent, black, or a flat factor) the
            # emissive atlas owns it, so its texCoord is not silently replaced by the base's unused one.
            base_owns_uvs = color_texture is not None and not color_texture.is_black and color_texture.requires_uv
            if material.emissiveTexture.texCoord is not None and not base_owns_uvs:
                uvs_used = material.emissiveTexture.texCoord
            emissive_image = get_glb_image(glb, texture.source, "RGB")

        emissive_factor = None
        if material.emissiveFactor is not None:
            emissive_factor = np.array(material.emissiveFactor, dtype=np.float32)

        emissive_texture = mu.create_texture(emissive_image, emissive_factor, "srgb")
        if emissive_texture.is_black:  # Make sure to check emissive
            emissive_texture = None

    # TODO: Parse them!
    for extension_name, extension_material in material.extensions.items():
        if extension_name == "KHR_materials_specular":
            # https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_materials_specular/README.md
            specular_weight = extension_material.get("specularFactor", 1.0)
            specular_color = np.array(extension_material.get("specularColorFactor", [1.0, 1.0, 1.0]), dtype=np.float32)
        elif extension_name == "KHR_materials_clearcoat":
            # https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_materials_clearcoat/README.md
            clearcoat_weight = extension_material.get("clearcoatFactor", 0.0)
            clearcoat_roughness_factor = extension_material.get("clearcoatRoughnessFactor", 0.0)
        elif extension_name == "KHR_materials_volume":
            # https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_materials_volume/README.md
            attenuation_distance = extension_material.get("attenuationDistance", float("inf"))
        elif extension_name == "KHR_materials_transmission":
            # https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_materials_transmission/README.md
            specular_trans_factor = extension_material.get("transmissionFactor", 0.0)
        elif extension_name == "KHR_materials_ior":
            # https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_materials_ior/README.md
            ior = extension_material.get("ior", 1.5)

    material_surface = surface.model_copy()
    material_surface.update_texture(
        color_texture=color_texture,
        opacity_texture=opacity_texture,
        roughness_texture=roughness_texture,
        metallic_texture=metallic_texture,
        normal_texture=normal_texture,
        emissive_texture=emissive_texture,
        ior=ior,
        double_sided=double_sided,
    )

    return material_surface, uvs_used, material.name


def parse_glb_tree(glb, node_index):
    node = glb.nodes[node_index]
    non_identity = False
    if node.matrix is not None:
        transform = np.array(node.matrix, dtype=np.float32).reshape((4, 4))
        non_identity = True
    else:
        transform = np.identity(4, dtype=np.float32)
        if node.translation is not None:
            transform[:3, 3] = node.translation
            non_identity = True
        if node.rotation is not None:
            quat = np.array(node.rotation, dtype=np.float32)[[3, 0, 1, 2]]
            gu.quat_to_R(quat, out=transform[:3, :3])
            non_identity = True
        if node.scale is not None:
            transform[:3, :3] *= node.scale
            non_identity = True
        transform = transform.T  # translation at bottom

    mesh_list = []
    for sub_node_index in node.children:
        sub_mesh_list = parse_glb_tree(glb, sub_node_index)
        mesh_list += sub_mesh_list
    if non_identity:
        for _, mesh_transform in mesh_list:
            mesh_transform @= transform

    if node.mesh is not None:
        mesh_list.append([node.mesh, transform])
    return mesh_list


def parse_mesh_glb(path, group_by_material, scale, is_mesh_zup, surface):
    glb = pygltflib.GLTF2().load(path)
    assert glb is not None
    glb.convert_images(pygltflib.ImageFormat.DATAURI)
    glb.path = path

    glb_scene = glb.scene or 0
    scene = glb.scenes[glb_scene]
    mesh_list = []
    for node_index in scene.nodes:
        root_mesh_list = parse_glb_tree(glb, node_index)
        mesh_list += root_mesh_list

    mesh_infos = mu.MeshInfoGroup()
    materials = {}
    is_visual_overwritten = surface.texture is not None

    for i, (mesh_index, mesh_transform) in enumerate(mesh_list):
        mesh_glb = glb.meshes[mesh_index]
        mesh_name = mesh_glb.name

        for primitive in mesh_glb.primitives:
            if primitive.material is not None:
                material, uv_used, material_name = materials.get(primitive.material, (None, 0, ""))
                if material is None:
                    material, uv_used, material_name = materials.setdefault(
                        primitive.material, parse_glb_material(glb, primitive.material, surface)
                    )
            else:
                material, uv_used, material_name = surface.model_copy(), 0, ""

            uvs = None
            if "KHR_draco_mesh_compression" in primitive.extensions:
                KHR_index = primitive.extensions["KHR_draco_mesh_compression"]["bufferView"]
                mesh_buffer_view = glb.bufferViews[KHR_index]
                mesh_data = get_glb_bufferview_data(glb, mesh_buffer_view)
                mesh_glb = DracoPy.decode(
                    mesh_data[mesh_buffer_view.byteOffset : mesh_buffer_view.byteOffset + mesh_buffer_view.byteLength]
                )
                points = mesh_glb.points
                triangles = mesh_glb.faces
                normals = mesh_glb.normals if mesh_glb.normals is not None and len(mesh_glb.normals) > 0 else None
                uvs = mesh_glb.tex_coord if mesh_glb.tex_coord is not None and len(mesh_glb.tex_coord) > 0 else None

            else:
                # "primitive.attributes" records accessor indices in "glb.accessors", like:
                #      Attributes(POSITION=2, NORMAL=1, TANGENT=None, TEXCOORD_0=None, TEXCOORD_1=None,
                #                 COLOR_0=None, JOINTS_0=None, WEIGHTS_0=None)
                # parse vertices

                points = get_glb_data_from_accessor(glb, primitive.attributes.POSITION).astype(np.float32)

                if primitive.indices is None:
                    indices = np.arange(len(points), dtype=np.uint32)
                else:
                    indices = get_glb_data_from_accessor(glb, primitive.indices).astype(np.int32)

                mode = primitive.mode if primitive.mode is not None else 4

                if mode == 4:  # TRIANGLES
                    triangles = indices.reshape(-1, 3)
                elif mode == 5:  # TRIANGLE_STRIP
                    triangles = []
                    for i in range(len(indices) - 2):
                        if i % 2 == 0:
                            triangles.append([indices[i], indices[i + 1], indices[i + 2]])
                        else:
                            triangles.append([indices[i], indices[i + 2], indices[i + 1]])
                    triangles = np.array(triangles, dtype=np.uint32)
                elif mode == 6:  # TRIANGLE_FAN
                    triangles = []
                    for i in range(1, len(indices) - 1):
                        triangles.append([indices[0], indices[i], indices[i + 1]])
                    triangles = np.array(triangles, dtype=np.uint32)
                else:
                    gs.logger.warning(f"Primitive mode {mode} not supported.")
                    continue  # Skip unsupported modes

                # parse normals
                if primitive.attributes.NORMAL is not None:
                    normals = get_glb_data_from_accessor(glb, primitive.attributes.NORMAL).astype(np.float32)
                else:
                    normals = None

                # parse uvs
                if uv_used == 0:
                    if primitive.attributes.TEXCOORD_0 is not None:
                        uvs = get_glb_data_from_accessor(glb, primitive.attributes.TEXCOORD_0).astype(np.float32)
                elif uv_used == 1:
                    if primitive.attributes.TEXCOORD_1 is not None:
                        uvs = get_glb_data_from_accessor(glb, primitive.attributes.TEXCOORD_1).astype(np.float32)

            points, normals = mu.apply_transform(mesh_transform, points, normals)
            if normals is None:
                normals = trimesh.Trimesh(points, triangles, process=False).vertex_normals

            # A single glTF mesh may hold several primitives with distinct materials. When not grouping by material,
            # primitives must still be separated by material so each keeps its own surface and texture; otherwise
            # primitives sharing a mesh would be merged under the first primitive's material and the rest lost.
            group_idx = primitive.material if group_by_material else (i, primitive.material)
            mesh_info, first_created = mesh_infos.get(group_idx)
            if first_created:
                metadata = {"mesh_path": path, "name": material_name if group_by_material else mesh_name}
                if not group_by_material:
                    # Record the source mesh node so per-material submeshes can be regrouped into one physical body
                    # (e.g. merged into a single collision geom) while still rendering with their own textures.
                    metadata["node_index"] = i
                mesh_info.set_property(surface=material, metadata=metadata)
            mesh_info.append(points, triangles, normals, uvs)
    meshes = mesh_infos.export_meshes(scale=scale, is_mesh_zup=is_mesh_zup)
    for mesh in meshes:
        mesh.metadata["is_visual_overwritten"] = is_visual_overwritten
    return meshes

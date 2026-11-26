"""Meshes of the rasterizer, as glTF 2.0 defines them."""

import copy

import numpy as np

from OpenGL.GL import GL_POINTS
import trimesh

from .constants import ViewFlags
from .material import Material
from .primitive import Primitive


class Mesh:
    """A set of primitives drawn together.

    Parameters
    ----------
    primitives : list of Primitive
        The primitives of the mesh.
    is_marker : bool, optional
        Whether the mesh is a marker, which the debug views alone draw and which receives no shadow. Defaults to False.
    views : int, optional
        The views drawing the mesh, a bitwise or of ViewFlags: those that show the scene as it looks, the debug views,
        or both. Defaults to both.
    """

    def __init__(self, primitives, is_marker=False, views=ViewFlags.NORMAL | ViewFlags.DEBUG):
        self.primitives = primitives
        self.is_marker = is_marker
        self.views = views

    @property
    def bounds(self):
        """(2, 3) float : The axis-aligned bounds of the mesh, over the instances of its primitives."""
        if not self.primitives:
            return np.zeros((2, 3))
        # In the vast majority of scenarios there is a single primitive
        if len(self.primitives) == 1:
            return self.primitives[0].bounds
        primitives_bounds = np.stack([primitive.bounds for primitive in self.primitives])
        return np.stack((primitives_bounds[:, 0].min(axis=0), primitives_bounds[:, 1].max(axis=0)))

    @property
    def centroid(self):
        """(3,) float : The centre of the axis-aligned bounds of the mesh."""
        return np.mean(self.bounds, axis=0)

    @property
    def is_transparent(self):
        """bool : Whether some primitives of the mesh show part of what lies behind them."""
        return any(primitive.is_transparent for primitive in self.primitives)

    @staticmethod
    def from_points(
        points,
        colors=None,
        normals=None,
        poses=None,
        is_marker=False,
        views=ViewFlags.NORMAL | ViewFlags.DEBUG,
        envs=None,
    ):
        """Create a mesh drawing a set of points.

        Parameters
        ----------
        points : (n, 3) float
            The positions of the points.
        colors : (n, 3) or (n, 4) float, optional
            The RGB or RGBA color of each point. Defaults to None.
        normals : (n, 3) float, optional
            The normal of each point. Defaults to None.
        poses : (k, 4, 4) float, optional
            The pose of each instance of the points. Defaults to a single instance.
        is_marker : bool, optional
            Whether the points are a marker (see Mesh). Defaults to False.
        views : int, optional
            The views drawing the points (see Mesh). Defaults to both.
        envs : int or (n_envs,) bool, optional
            The environments the instances belong to (see Primitive). Defaults to every environment.

        Returns
        -------
        mesh : Mesh
            The mesh drawing the points.
        """
        primitive = Primitive(positions=points, normals=normals, color_0=colors, mode=GL_POINTS, poses=poses, envs=envs)
        return Mesh(primitives=[primitive], is_marker=is_marker, views=views)

    @staticmethod
    def from_trimesh(
        mesh,
        material=None,
        is_marker=False,
        views=ViewFlags.NORMAL | ViewFlags.DEBUG,
        poses=None,
        smooth=False,
        double_sided=False,
        is_floor=False,
        envs=None,
        inst_tints=None,
        metallic_factor=0.0,
        roughness_factor=1.0,
        base_color_factor=(1.0, 1.0, 1.0),
        transmission_factor=0.0,
        ior=1.5,
    ):
        """Create a mesh drawing a triangle mesh of trimesh.

        Parameters
        ----------
        mesh : trimesh.Trimesh
            The triangle mesh, its visual giving its colors, texture coordinates and material.
        material : Material, optional
            The material of the mesh, in place of the one its visual gives. Defaults to None.
        is_marker : bool, optional
            Whether the mesh is a marker (see Mesh). Defaults to False.
        views : int, optional
            The views drawing the mesh (see Mesh). Defaults to both.
        poses : (k, 4, 4) float, optional
            The pose of each instance of the mesh. Defaults to a single instance.
        smooth : bool, optional
            Whether the normals are interpolated across the faces, rather than the edges staying sharp. Defaults to
            False.
        double_sided : bool, optional
            Whether both faces of each triangle are drawn and lit (see Primitive). Defaults to False.
        is_floor : bool, optional
            Whether the mesh is the floor (see Primitive). Defaults to False.
        envs : int or (n_envs,) bool, optional
            The environments the instances belong to (see Primitive). Defaults to every environment.
        inst_tints : (k, 4) float, optional
            The RGBA tint of every instance (see Primitive). Defaults to None.
        metallic_factor : float, optional
            The metalness of the material made from the vertex colors, the face colors or the simple material of the
            mesh. A physically based rendering (PBR) material keeps its own. Defaults to 0.0.
        roughness_factor : float, optional
            The roughness of the material made from the vertex colors, the face colors or the simple material of the
            mesh. A PBR material keeps its own. Defaults to 1.0.
        base_color_factor : (3,) float, optional
            The color that the base color of the material made from the mesh is multiplied by. Defaults to white.
        transmission_factor : float, optional
            The share of the light that the material made from the mesh lets through. Defaults to 0.0.
        ior : float, optional
            The index of refraction of the material made from the mesh, 1.5 if None. Defaults to 1.5.

        Returns
        -------
        mesh : Mesh
            The mesh drawing the triangle mesh.
        """
        if smooth:
            positions, normals, indices = mesh.vertices.copy(), mesh.vertex_normals.copy(), mesh.faces.copy()
            vertex_mapping = None
        else:
            positions = mesh.vertices[mesh.faces].reshape((-1, 3))
            normals = np.repeat(mesh.face_normals, 3, axis=0)
            indices = None
            vertex_mapping = mesh.faces.reshape((-1,))
        color_0, texcoord_0, primitive_material = _get_trimesh_visual(
            mesh, smooth, metallic_factor, roughness_factor, has_material=material is not None
        )

        if material is not None:
            # A shallow copy keeps the state of the primitive (such as its wireframe) its own, while the heavy textures
            # stay shared with the source material. The primitives reusing one material then hold the same textures,
            # which the renderer uploads once.
            primitive_material = copy.copy(material)
        else:
            if primitive_material is None:
                primitive_material = Material(
                    is_blended=True,
                    base_color_factor=(0.3, 0.3, 0.3, 1.0),
                    metallic_factor=metallic_factor,
                    roughness_factor=roughness_factor,
                )
            primitive_material.base_color_factor = primitive_material.base_color_factor * (*base_color_factor, 1.0)
            # The materials of trimesh hold no transmission, so every material made here takes the given one
            primitive_material.transmission_factor = transmission_factor
            primitive_material.ior = 1.5 if ior is None else ior

        primitive = Primitive(
            positions=positions,
            normals=normals,
            texcoord_0=texcoord_0,
            color_0=color_0,
            indices=indices,
            material=primitive_material,
            poses=poses,
            vertex_mapping=vertex_mapping,
            double_sided=double_sided,
            is_floor=is_floor,
            envs=envs,
            inst_tints=inst_tints,
        )
        return Mesh(primitives=[primitive], is_marker=is_marker, views=views)


def _get_trimesh_visual(mesh, smooth, metallic_factor, roughness_factor, has_material):
    """Return the vertex colors, the texture coordinates and the material that the visual of a trimesh mesh gives.

    The vertices of a flat-shaded mesh are split per face, which the colors and the texture coordinates follow. A mesh
    given a material of its own takes the texture coordinates of its visual alone.
    """
    colors = None
    texcoords = None
    material = None
    visual = mesh.visual
    if not visual.defined:
        return colors, texcoords, material

    if visual.kind in ("vertex", "face") and not has_material:
        if visual.kind == "vertex":
            colors = visual.vertex_colors
            if not smooth and colors.ndim > 1:
                colors = colors[mesh.faces].reshape((3 * len(mesh.faces), colors.shape[1]))
        else:
            assert not smooth
            colors = np.repeat(visual.face_colors, 3, axis=0)
        material = Material(
            is_blended=colors.shape[-1] == 4 and bool((colors[..., 3] < 255).any()),
            metallic_factor=metallic_factor,
            roughness_factor=roughness_factor,
        )
    elif visual.kind == "texture":
        if isinstance(visual, trimesh.visual.texture.TextureVisuals) and visual.uv is not None:
            texcoords = visual.uv if smooth else visual.uv[mesh.faces].reshape((3 * len(mesh.faces), -1))
        trimesh_material = None if has_material else visual.material
        if isinstance(trimesh_material, trimesh.visual.texture.PBRMaterial):
            material = Material(
                normal_texture=trimesh_material.normalTexture,
                occlusion_texture=trimesh_material.occlusionTexture,
                emissive_texture=trimesh_material.emissiveTexture,
                emissive_factor=trimesh_material.emissiveFactor,
                is_blended=True,
                base_color_factor=trimesh_material.baseColorFactor,
                base_color_texture=trimesh_material.baseColorTexture,
                metallic_factor=trimesh_material.metallicFactor,
                roughness_factor=trimesh_material.roughnessFactor,
                metallic_roughness_texture=trimesh_material.metallicRoughnessTexture,
            )
        elif isinstance(trimesh_material, trimesh.visual.texture.SimpleMaterial):
            material = Material(
                is_blended=bool((np.asarray(trimesh_material.image.convert("RGBA"))[..., 3] < 255).any()),
                metallic_factor=metallic_factor,
                roughness_factor=roughness_factor,
                base_color_factor=trimesh_material.diffuse,
                base_color_texture=trimesh_material.image,
            )
    return colors, texcoords, material

from enum import IntEnum

DEFAULT_Z_NEAR = 0.05
"""The near plane of the camera of the viewer, in meters."""
DEFAULT_Z_FAR = 100.0
"""The far plane of the camera of the viewer, in meters."""
DEFAULT_SCENE_SCALE = 5.0
"""The size of the scene that the camera of the viewer frames at first, in meters."""
MAX_N_LIGHTS = 4
"""The maximum number of lights of a scene (see 'set_light' in jit_render.py)."""
TARGET_OPEN_GL_MAJOR = 4
"""The major version of the OpenGL context requested first."""
TARGET_OPEN_GL_MINOR = 2
"""The minor version of the OpenGL context requested first."""
MIN_OPEN_GL_MAJOR = 3
"""The major version of the oldest OpenGL context accepted."""
MIN_OPEN_GL_MINOR = 3
"""The minor version of the oldest OpenGL context accepted."""
SOFTWARE_RENDERERS = ("llvmpipe", "Apple Software Renderer")
"""The names of the OpenGL renderers running on the CPU, which make rendering slow."""
FLOAT_SZ = 4
"""The size of a GL float, in bytes."""
UINT_SZ = 4
"""The size of a GL unsigned integer, in bytes."""
DIRECTIONAL_SHADOW_TEX_SZ = 4096
"""The width and height of the shadow map of a directional light, in texels."""
POINT_SHADOW_TEX_SZ = 2048
"""The width and height of each face of the shadow cube map of a point light, in texels."""
SPOT_SHADOW_TEX_SZ = 2048
"""The width and height of the shadow map of a spot light, in texels."""
TRANSMITTANCE_TEX_SZ = 1024
"""The width and height of the map of the light that transparent surfaces let through, for a directional or a spot
light, in texels."""
POINT_TRANSMITTANCE_TEX_SZ = 512
"""The width and height of each face of the same cube map for a point light, in texels."""
TEXT_PADDING = 20
"""The margin between the text and the edges of the window, in pixels."""
FONT_SIZE = 26
"""The height of the text of the viewer, in pixels."""
FONT_COLOR_LIGHTMODE = (0.0, 0.0, 0.0, 0.85)
"""The RGBA color of the text over a light background."""
FONT_COLOR_DARKMODE = (1.0, 1.0, 1.0, 0.85)
"""The RGBA color of the text over a dark background."""


class RenderFlags:
    """Flags for rendering in the scene.

    Combine them with the bitwise or. For example,

    >>> flags = OFFSCREEN | SHADOWS_DIRECTIONAL | VERTEX_NORMALS

    would result in an offscreen render with directional shadows and
    vertex normals enabled.
    """

    NONE = 0
    """Normal PBR Render."""
    DEPTH_ONLY = 1
    """Only render the depth buffer."""
    OFFSCREEN = 2
    """Render offscreen and return the depth and (optionally) color buffers."""
    FLIP_WIREFRAME = 4
    """Invert the status of wireframe rendering for each mesh."""
    ALL_WIREFRAME = 8
    """Render all meshes as wireframes."""
    ALL_SOLID = 16
    """Render all meshes as solids."""
    SHADOWS_DIRECTIONAL = 32
    """Render shadows for directional lights."""
    SHADOWS_POINT = 64
    """Render shadows for point lights."""
    SHADOWS_SPOT = 128
    """Render shadows for spot lights."""
    SHADOWS_ALL = 32 | 64 | 128
    """Render shadows for all lights."""
    VERTEX_NORMALS = 256
    """Render vertex normals."""
    FACE_NORMALS = 512
    """Render face normals."""
    NORMAL = 4096
    """Render the normal of the surfaces in world space, encoded as a color."""
    SEG = 8192
    """Render segmentation."""
    RET_DEPTH = 1 << 14
    """Return depth"""
    SKIP_FLOOR = 1 << 15
    """Skip the floor"""
    REFLECTIVE_FLOOR = 1 << 16
    """Render the reflection of floor"""
    ENV_SEPARATE = 1 << 17
    """Render each environment separately."""
    SKIP_MARKERS = 1 << 18
    """Skip rendering marker nodes (debug arrows, frames, etc.)."""
    MARKER_XRAY = 1 << 19
    """Render occluded markers with darkened transparency (X-ray mode)."""


class TexChannels(IntEnum):
    """The channels that a texture holds."""

    R = 0
    GB = 1
    """The green and the blue channels of the source, those of the metalness and the roughness."""
    RGB = 2
    RGBA = 3
    DEPTH = 4

    @property
    def n_channels(self):
        """int : The number of channels."""
        return (1, 2, 3, 4, 1)[self]


class LightType:
    """The kind of a light, as the compiled passes read it."""

    DIRECTIONAL = 0
    POINT = 1
    SPOT = 2


class ViewFlags:
    """Which views draw a mesh, as a bitwise or of the flags below (see 'Mesh.views').

    The views that show the scene as it looks and the debug views render from a single scene, each drawing only the
    meshes whose flags include its kind.
    """

    NORMAL = 1
    """Drawn by the views that show the scene as it looks."""
    DEBUG = 2
    """Drawn by the debug views."""


class WireframeMode(IntEnum):
    """How the viewer draws the meshes, as solids or as wireframes."""

    MATERIAL = 0
    """Each mesh as its material says."""
    FLIP = 1
    """Each mesh the other way than its material says."""
    WIREFRAME = 2
    """Every mesh as a wireframe."""
    SOLID = 3
    """Every mesh as a solid."""


class TextAlign(IntEnum):
    """The corner of a text that its position gives."""

    TOP_LEFT = 0
    """The top-left corner."""
    BOTTOM_RIGHT = 1
    """The bottom-right corner."""


class BufFlags:
    """The vertex buffers that a primitive holds besides its positions."""

    POSITION = 0
    NORMAL = 1
    TEXCOORD_0 = 2
    COLOR_0 = 4


class TexFlags:
    NONE = 0
    NORMAL = 1
    EMISSIVE = 2
    BASE_COLOR = 4
    METALLIC_ROUGHNESS = 8
    OCCLUSION = 16


class ProgramFlags:
    NONE = 0
    USE_MATERIAL = 1
    VERTEX_NORMALS = 2
    FACE_NORMALS = 4
    POINT_SHADOW = 8
    SHADOW = 32
    TRANSMITTANCE = 64

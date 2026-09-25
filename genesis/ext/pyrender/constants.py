DEFAULT_Z_NEAR = 0.05  # Near clipping plane, in meters
DEFAULT_Z_FAR = 100.0  # Far clipping plane, in meters
DEFAULT_SCENE_SCALE = 5.0  # Default scene scale
MAX_N_LIGHTS = 4  # Maximum number of lights of each type allowed
TARGET_OPEN_GL_MAJOR = 4  # Target OpenGL Major Version
TARGET_OPEN_GL_MINOR = 2  # Target OpenGL Minor Version
MIN_OPEN_GL_MAJOR = 3  # Minimum OpenGL Major Version
MIN_OPEN_GL_MINOR = 3  # Minimum OpenGL Minor Version
FLOAT_SZ = 4  # Byte size of GL float32
UINT_SZ = 4  # Byte size of GL uint32
SHADOW_TEX_SZ = 8192  # Width and Height of Shadow Textures
TEXT_PADDING = 20  # Width of padding for rendering text (px)
FONT_SIZE = 26  # Default font size for rendering text (px)
FONT_COLOR_LIGHTMODE = (0.0, 0.0, 0.0, 0.85)  # Default font color for light background color
FONT_COLOR_DARKMODE = (1.0, 1.0, 1.0, 0.85)  # Default font color for dark background color


# Flags for render type
class RenderFlags(object):
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
    SHADOWS_ALL = 32 | 64  # | 128
    """Render shadows for all lights."""
    VERTEX_NORMALS = 256
    """Render vertex normals."""
    FACE_NORMALS = 512
    """Render face normals."""
    SKIP_CULL_FACES = 1024
    """Do not cull back faces."""
    RGBA = 2048
    """Render the color buffer with the alpha channel enabled."""
    FLAT = 4096
    """Render the color buffer flat, with no lighting computations."""
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


class TextAlign:
    """Text alignment options for captions.

    Only use one at a time.
    """

    CENTER = 0
    """Center the text by width and height."""
    CENTER_LEFT = 1
    """Center the text by height and left-align it."""
    CENTER_RIGHT = 2
    """Center the text by height and right-align it."""
    BOTTOM_LEFT = 3
    """Put the text in the bottom-left corner."""
    BOTTOM_RIGHT = 4
    """Put the text in the bottom-right corner."""
    BOTTOM_CENTER = 5
    """Center the text by width and fix it to the bottom."""
    TOP_LEFT = 6
    """Put the text in the top-left corner."""
    TOP_RIGHT = 7
    """Put the text in the top-right corner."""
    TOP_CENTER = 8
    """Center the text by width and fix it to the top."""


class GLTF(object):
    """Options for GL objects."""

    NEAREST = 9728
    """Nearest neighbor interpolation."""
    LINEAR = 9729
    """Linear interpolation."""
    NEAREST_MIPMAP_NEAREST = 9984
    """Nearest mipmapping."""
    LINEAR_MIPMAP_NEAREST = 9985
    """Linear mipmapping."""
    NEAREST_MIPMAP_LINEAR = 9986
    """Nearest mipmapping."""
    LINEAR_MIPMAP_LINEAR = 9987
    """Linear mipmapping."""
    CLAMP_TO_EDGE = 33071
    """Clamp to the edge of the texture."""
    MIRRORED_REPEAT = 33648
    """Mirror the texture."""
    REPEAT = 10497
    """Repeat the texture."""
    POINTS = 0
    """Render as points."""
    LINES = 1
    """Render as lines."""
    LINE_LOOP = 2
    """Render as a line loop."""
    LINE_STRIP = 3
    """Render as a line strip."""
    TRIANGLES = 4
    """Render as triangles."""
    TRIANGLE_STRIP = 5
    """Render as a triangle strip."""
    TRIANGLE_FAN = 6
    """Render as a triangle fan."""


class BufFlags(object):
    POSITION = 0
    NORMAL = 1
    TANGENT = 2
    TEXCOORD_0 = 4
    TEXCOORD_1 = 8
    COLOR_0 = 16
    JOINTS_0 = 32
    WEIGHTS_0 = 64


class TexFlags(object):
    NONE = 0
    NORMAL = 1
    OCCLUSION = 2
    EMISSIVE = 4
    BASE_COLOR = 8
    METALLIC_ROUGHNESS = 16
    DIFFUSE = 32
    SPECULAR_GLOSSINESS = 64


class ProgramFlags:
    NONE = 0
    USE_MATERIAL = 1
    VERTEX_NORMALS = 2
    FACE_NORMALS = 4
    POINT_SHADOW = 8
    DOUBLE_SIDED = 16


__all__ = ["RenderFlags", "TextAlign", "GLTF"]

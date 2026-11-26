import numpy as np
from PIL import Image

from .constants import TexChannels


def format_color_vector(value, length):
    """Return a color as a vector of float32 channels within [0, 1].

    Parameters
    ----------
    value : float or array_like
        The color, a scalar standing for every channel. Integer channels hold 8-bit values. Missing channels are filled
        with 1.0 and extra ones dropped.
    length : int
        The number of channels.
    """
    value = np.array(value)
    if np.issubdtype(value.dtype, np.integer):
        value = value / 255.0
    if value.ndim == 0:
        value = np.full(length, value)
    assert value.ndim == 1
    value = np.concatenate((value[:length], np.ones(max(length - len(value), 0))))
    return np.ascontiguousarray(value, dtype=np.float32)


def format_color_array(value, shape):
    """Return an array of colors as float32 channels within [0, 1].

    Parameters
    ----------
    value : array_like
        The colors, one per row, or a single color for every row. Integer channels hold 8-bit values. Missing channels
        are filled with 1.0 and extra ones dropped.
    shape : tuple of int
        The number of colors and of channels.
    """
    value = np.array(value)
    if np.issubdtype(value.dtype, np.integer):
        value = value / 255.0
    if value.ndim == 1:
        value = np.tile(value, (shape[0], 1))
    n_missing = shape[1] - value.shape[1]
    value = np.column_stack((value[:, : shape[1]], np.ones((len(value), max(n_missing, 0)))))
    return np.ascontiguousarray(value, dtype=np.float32)


def format_texture_source(texture, channels):
    """Return the image of a texture as uint8 values holding some channels.

    Parameters
    ----------
    texture : (h, w) or (h, w, c) uint8 or float or PIL.Image.Image
        The image, floats within [0, 1].
    channels : TexChannels
        The channels to hold. A gray image spans the color channels, a missing alpha is opaque, and green and blue are
        taken from the first channel of a gray image.
    """
    if isinstance(texture, Image.Image):
        # The texels of a palette image are indices into its palette
        if texture.mode == "P":
            texture = texture.convert("RGBA")
        texture = np.array(texture)
    if np.issubdtype(texture.dtype, np.floating):
        texture = (texture * 255.0).astype(np.uint8)
    else:
        assert np.issubdtype(texture.dtype, np.integer)
        texture = texture.astype(np.uint8)
    if texture.ndim == 2:
        texture = texture[:, :, np.newaxis]

    n_channels = texture.shape[2]
    if channels == TexChannels.R:
        return texture[:, :, 0]
    if channels == TexChannels.GB:
        if n_channels == 1:
            return np.repeat(texture, 2, axis=2)
        return texture if n_channels == 2 else texture[:, :, 1:3]
    # A gray image, with or without alpha, spans the three color channels
    rgb = texture[:, :, (0, 0, 0)] if n_channels < 3 else texture[:, :, :3]
    if channels == TexChannels.RGB:
        return np.ascontiguousarray(rgb)
    assert channels == TexChannels.RGBA
    alpha = texture[:, :, n_channels - 1 : n_channels] if n_channels in (2, 4) else np.full_like(rgb[:, :, :1], 255)
    return np.concatenate((rgb, alpha), axis=2)


def transform_box(box, poses):
    """Bound an axis-aligned box once moved by each of several transforms.

    The box moved by a transform is bounded by the box centred on its transformed centre, whose half-extents go through
    the absolute value of the rotation. The bounds of all the moved boxes are then merged.

    Parameters
    ----------
    box : np.ndarray
        The lower and upper corners of the box, along the last but one axis.
    poses : np.ndarray
        The transforms, along the last but two axis, the leading axes broadcasting against those of the box.

    Returns
    -------
    bounds : np.ndarray
        The lower and upper corners of the box bounding every moved box, along the last but one axis.
    """
    centre = 0.5 * (box[..., np.newaxis, 0, :] + box[..., np.newaxis, 1, :])
    half = 0.5 * (box[..., np.newaxis, 1, :] - box[..., np.newaxis, 0, :])
    rotations = poses[..., :3, :3]
    centres = np.einsum("...ij,...j->...i", rotations, centre) + poses[..., :3, 3]
    halves = np.einsum("...ij,...j->...i", np.abs(rotations), half)
    return np.stack(((centres - halves).min(axis=-2), (centres + halves).max(axis=-2)), axis=-2)

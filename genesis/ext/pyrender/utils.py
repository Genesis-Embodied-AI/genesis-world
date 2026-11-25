import numpy as np
from PIL import Image


def format_color_vector(value, length):
    """Format a color vector."""
    if isinstance(value, int):
        value = value / 255.0
    if isinstance(value, float):
        value = np.repeat(value, length)
    if isinstance(value, list) or isinstance(value, tuple):
        value = np.array(value)
    if isinstance(value, np.ndarray):
        value = value.squeeze()
        if np.issubdtype(value.dtype, np.integer):
            value = (value / 255.0).astype(np.float32)
        if value.ndim != 1:
            raise ValueError("Format vector takes only 1-D vectors")
        if length > value.shape[0]:
            value = np.hstack((value, np.ones(length - value.shape[0])))
        elif length < value.shape[0]:
            value = value[:length]
    else:
        raise ValueError("Invalid vector data type")

    return np.ascontiguousarray(value.squeeze(), dtype=np.float32)


def format_color_array(value, shape):
    """Format an array of colors."""
    # Convert uint8 to floating
    value = np.asanyarray(value)
    if np.issubdtype(value.dtype, np.integer):
        value = (value / 255.0).astype(np.float32)

    # Match up shapes
    if value.ndim == 1:
        value = np.tile(value, (shape[0], 1))
    if value.shape[1] < shape[1]:
        nc = shape[1] - value.shape[1]
        value = np.column_stack((value, np.ones((value.shape[0], nc))))
    elif value.shape[1] > shape[1]:
        value = value[:, : shape[1]]
    return np.ascontiguousarray(value, dtype=np.float32)


def format_texture_source(texture, target_channels="RGB"):
    """Format a texture as a float32 np array."""

    # Pass through None
    if texture is None:
        return None

    # Convert PIL images into numpy arrays
    if isinstance(texture, Image.Image):
        if texture.mode == "P" and target_channels in ("RGB", "RGBA"):
            texture = np.array(texture.convert(target_channels))
        else:
            texture = np.array(texture)

    # Format numpy arrays
    if isinstance(texture, np.ndarray):
        if np.issubdtype(texture.dtype, np.floating):
            texture = np.array(texture * 255.0, dtype=np.uint8)
        elif np.issubdtype(texture.dtype, np.integer):
            texture = texture.astype(np.uint8)
        else:
            raise TypeError("Invalid type {} for texture".format(type(texture)))

        # Format array by picking out correct texture channels or padding
        if texture.ndim == 2:
            texture = texture[:, :, np.newaxis]
        if target_channels == "R":
            texture = texture[:, :, 0]
            texture = texture.squeeze()
        elif target_channels == "RG":
            if texture.shape[2] == 1:
                texture = np.repeat(texture, 2, axis=2)
            else:
                texture = texture[:, :, (0, 1)]
        elif target_channels == "GB":
            if texture.shape[2] == 1:
                texture = np.repeat(texture, 2, axis=2)
            elif texture.shape[2] > 2:
                texture = texture[:, :, (1, 2)]
        elif target_channels == "RGB":
            if texture.shape[2] == 1:
                texture = np.repeat(texture, 3, axis=2)
            elif texture.shape[2] == 2:
                # Convert 2-channel texture to grayscale RGB by duplicating the first channel
                texture = texture[:, :, (0, 0, 0)]
            else:
                texture = texture[:, :, (0, 1, 2)]
        elif target_channels == "RGBA":
            if texture.shape[2] == 1:
                texture = np.repeat(texture, 4, axis=2)
                texture[:, :, 3] = 255
            elif texture.shape[2] == 2:
                # Convert 2-channel texture (luminance + alpha) to RGBA by duplicating luminance to RGB
                luminance = texture[:, :, :1]
                alpha = texture[:, :, 1]
                texture = np.empty((*texture.shape[:2], 4), dtype=texture.dtype)
                texture[:, :, :3] = luminance
                texture[:, :, 3] = alpha
            elif texture.shape[2] == 3:
                tx = np.empty((*texture.shape[:2], 4), dtype=np.uint8)
                tx[:, :, :3] = texture
                tx[:, :, 3] = 255
                texture = tx
        else:
            raise ValueError("Invalid texture channel specification: {}".format(target_channels))
    else:
        raise TypeError("Invalid type {} for texture".format(type(texture)))

    return texture


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

import numbers
from enum import Enum

import numpy as np


def pprint_oneline(data, delimiter, digits=None):
    msg_items = []
    for key, value in data.items():
        if isinstance(value, Enum):
            value = value.name
        if digits is not None and isinstance(value, (numbers.Real, np.floating)):
            value = f"{value:.{digits}f}"
        msg_item = "=".join((key, str(value)))
        msg_items.append(msg_item)
    return delimiter.join(msg_items)


def get_masks_mean_color(rgb, masks):
    """Return the mean color of an image over each of some boolean masks.

    The masks stack along the axis before their image axes, and broadcast against the image over the axes before it,
    such as the environments of a camera rendering each one apart. The result holds one color per mask.
    """
    masks_weight = masks / masks.sum(axis=(-2, -1), keepdims=True)
    return np.sum(masks_weight[..., None] * rgb[..., None, :, :, :], axis=(-3, -2))


def get_shown_color(colors, colors_black, colors_white):
    """Return the color shown by some rendered colors, each channel 1 when lit and 0 when dark.

    A channel is lit when it matches that of a white surface and dark when it matches that of a black surface, both
    rendered under the same light, which broadcast against the colors. A channel matching neither is NaN, which matches
    no color.
    """
    # The specular reflection of a surface varies with the angle it is seen from, so that surfaces of the same color
    # apart from each other render with some difference.
    COLOR_ATOL = 20.0

    is_lit, is_dark = np.abs(colors - colors_white) < COLOR_ATOL, np.abs(colors - colors_black) < COLOR_ATOL
    return np.select((is_lit, is_dark), (1.0, 0.0), np.nan)

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

    The masks broadcast against the image without its channel axis, and the result holds one color per mask.
    """
    masks_weight = masks / masks.sum(axis=(-2, -1), keepdims=True)
    return np.sum(masks_weight[..., None] * rgb, axis=(-3, -2))

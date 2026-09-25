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

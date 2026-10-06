from __future__ import annotations

import tempfile
from pathlib import Path


def cloth_grid_asset(*, resolution: int = 25, size: float = 0.24) -> Path:
    """Return the CGQ Franka-cloth grid (24x24 cells by default)."""
    if resolution < 2:
        raise ValueError("Cloth grid resolution must be at least two")
    path = Path(tempfile.gettempdir()) / f"genesis_qcloth_grid_{resolution}_{size:.6f}.obj"
    if path.exists():
        return path

    lines = []
    for row in range(resolution):
        y = (row / (resolution - 1) - 0.5) * size
        for column in range(resolution):
            x = (column / (resolution - 1) - 0.5) * size
            lines.append(f"v {x:.17g} {y:.17g} 0")
    for row in range(resolution - 1):
        for column in range(resolution - 1):
            lower_left = row * resolution + column + 1
            lower_right = lower_left + 1
            upper_left = lower_left + resolution
            upper_right = upper_left + 1
            lines.append(f"f {lower_left} {lower_right} {upper_left}")
            lines.append(f"f {lower_right} {upper_right} {upper_left}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path

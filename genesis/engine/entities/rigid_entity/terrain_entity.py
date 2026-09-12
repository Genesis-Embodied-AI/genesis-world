import numpy as np
import torch

import genesis as gs

from .description import TerrainEntityDescription
from .rigid_entity import RigidEntity


class TerrainEntity(RigidEntity):
    """A rigid entity whose single fixed link is a height field, authored as a grid of elevations.

    Links rest on the surface those elevations define, and the stored grid answers a height query at any point.
    """

    _description_cls = TerrainEntityDescription

    def _load_model(self):
        super()._load_model()
        # Collider storage covers only the first collision terrain, while height queries support every terrain
        # entity, kinematic ones included. Assigned after the model loads, so it covers a morph and a description.
        self._terrain_height_field = torch.as_tensor(self.terrain_hf, dtype=gs.tc_float, device=gs.device)

    @property
    def terrain_hf(self) -> np.ndarray:
        """Elevation of every grid cell in meters, the vertical scale of the morph applied."""
        return self._desc.terrain_hf

    @property
    def terrain_scale(self) -> np.ndarray:
        """Horizontal size of a grid cell in meters, and the factor the elevation is scaled by."""
        return self._desc.terrain_scale

    @gs.assert_built
    def get_terrain_height(self, positions, envs_idx=None):
        """
        Return terrain surface heights in meters at world-frame x-y positions.

        Heights match the piecewise-planar surface on which rigid bodies rest. Terrain translation, yaw, and
        environment-specific poses are applied to the query. Positions up to one grid cell outside the terrain are
        clamped to its edge. Terrain tilt up to 0.001 radians from world vertical is treated as numerical noise. The
        query returns not-a-number (NaN) heights for positions containing NaN or infinity, positions farther outside,
        and terrains with greater tilt.

        Parameters
        ----------
        positions : array_like
            World-frame x-y positions in meters, with shape (2,), (n_points, 2), or
            (n_selected_envs, n_points, 2). A two-dimensional array is shared across the selected environments; use a
            three-dimensional array for environment-specific positions. A leading dimension of 1 in the
            three-dimensional form is also treated as shared.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.

        Returns
        -------
        heights : torch.Tensor
            World-frame surface heights in meters. The point dimension is preserved except for an explicit `(2,)`
            input, and an environment dimension is prepended in a parallelized scene.

        Raises
        ------
        GenesisException
            If the shape of `positions` is unsupported.
        """
        return self._solver.get_terrain_height(positions, self.base_link_idx, envs_idx)

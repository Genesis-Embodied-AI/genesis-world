import torch

import genesis as gs

from .description import DroneEntityDescription
from .rigid_entity import RigidEntity


class DroneEntity(RigidEntity):
    _description_cls = DroneEntityDescription

    def _load_model(self):
        super()._load_model()

        morph = self._morph
        self._n_propellers = len(morph.propellers_link_name)

        propellers_link = gs.List([self.get_link(name) for name in morph.propellers_link_name])
        self._propellers_link_idx = torch.tensor(
            [link.idx for link in propellers_link], dtype=gs.tc_int, device=gs.device
        )
        # The visual geom turning with each propeller as drawn, None for a drone a propeller of which has none
        self._propellers_vgeom_idx: torch.Tensor | None = None
        if all(link.vgeoms for link in propellers_link):
            self._propellers_vgeom_idx = torch.tensor(
                [link.vgeoms[0].idx for link in propellers_link], dtype=gs.tc_int, device=gs.device
            )
        else:
            gs.logger.warning("No visual geometry found for propellers. Skipping propeller animation.")

        self._propellers_spin = torch.tensor(morph.propellers_spin, dtype=gs.tc_float, device=gs.device)
        self._model = morph.model

    @gs.assert_unbuilt
    def _build(self):
        super()._build()

        self._prev_prop_t = None

    def set_propellers_rpm(self, propellers_rpm):
        """
        Set the RPM (revolutions per minute) for each propeller in the drone.

        Parameters
        ----------
        propellers_rpm : array-like or torch.Tensor
            A tensor or array of shape (n_propellers,) or (n_envs, n_propellers) specifying
            the desired RPM values for each propeller. Must be non-negative.

        Raises
        ------
        RuntimeError
            If the method is called more than once per simulation step, or if the input shape
            does not match the number of propellers, or contains negative values.
        """
        if self._prev_prop_t == self._solver.sim.cur_step_global:
            gs.raise_exception("`set_propellers_rpm` can only be called once per step.")
        self._prev_prop_t = self._solver.sim.cur_step_global

        assert propellers_rpm is not None
        propellers_rpm, *_ = self._solver._sanitize_io_variables(
            propellers_rpm, self._propellers_link_idx, self._n_propellers, "propellers_link_idx"
        )
        if self._solver.n_envs == 0:
            propellers_rpm = propellers_rpm[None]

        # FIXME: This check is too expensive
        # if (propellers_rpm < 0.0).any():
        #     gs.raise_exception("`propellers_rpm` cannot be negative.")

        # The propellers turn as drawn by the revolutions they accumulate, which the render context holds
        self._solver.render_context.add_propellers_rpm(self, propellers_rpm)

        self.solver.set_drone_rpm(
            self._propellers_link_idx, self.KF, self.KM, propellers_rpm, self._propellers_spin, self._model == "RACE"
        )

    @property
    def model(self):
        """The model type of the drone."""
        return self._model

    @property
    def KF(self):
        """The drone's thrust coefficient."""
        return self._desc.kf

    @property
    def KM(self):
        """The drone's moment coefficient."""
        return self._desc.km

    @property
    def n_propellers(self):
        """The number of propellers on the drone."""
        return self._n_propellers

    @property
    def COM_link_idx(self):
        """The index of the center-of-mass (COM) link of the drone."""
        return self._COM_link_idx

    @property
    def propellers_idx(self):
        """The indices of the drone's propeller links."""
        return self._propellers_link_idx

    @property
    def propellers_vgeom_idx(self) -> torch.Tensor | None:
        """The visual geom drawn turning with each propeller, None for a drone a propeller of which has none."""
        return self._propellers_vgeom_idx

    @property
    def propellers_spin(self):
        """The spin direction for each propeller."""
        return self._propellers_spin

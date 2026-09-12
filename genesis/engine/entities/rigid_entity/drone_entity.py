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
        try:
            self._propellers_vgeom_idxs = torch.tensor(
                [link.vgeoms[0].idx for link in propellers_link], dtype=gs.tc_int, device=gs.device
            )
            self._animate_propellers = True
        except Exception:
            gs.logger.warning("No visual geometry found for propellers. Skipping propeller animation.")
            self._animate_propellers = False

        self._propellers_spin = torch.tensor(morph.propellers_spin, dtype=gs.tc_float, device=gs.device)
        self._model = morph.model

    def _build(self):
        super()._build()

        self._propellers_revs = torch.zeros((self._n_propellers, self.solver._B), dtype=gs.tc_float, device=gs.device)
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
        if self._prev_prop_t == self.sim.cur_step_global:
            gs.raise_exception("`set_propellers_rpm` can only be called once per step.")
        self._prev_prop_t = self.sim.cur_step_global

        assert propellers_rpm is not None
        propellers_rpm, *_ = self._solver._sanitize_io_variables(
            propellers_rpm, self._propellers_link_idx, self._n_propellers, "propellers_link_idx"
        )
        if self._scene.n_envs == 0:
            propellers_rpm = propellers_rpm[None]

        # FIXME: This check is too expensive
        # if (propellers_rpm < 0.0).any():
        #     gs.raise_exception("`propellers_rpm` cannot be negative.")

        self._propellers_revs = (self._propellers_revs + propellers_rpm.T) % (60 / self.solver.dt)

        self.solver.set_drone_rpm(
            self._propellers_link_idx, self.KF, self.KM, propellers_rpm, self._propellers_spin, self._model == "RACE"
        )

    def update_propeller_vgeoms(self):
        """
        Update the visual geometry of the propellers for animation based on their current rotation.

        This method is a no-op if animation is disabled due to missing visual geometry.
        """
        if self._animate_propellers:
            self.solver.update_drone_propeller_vgeoms(
                self._propellers_vgeom_idxs, self._propellers_revs, self._propellers_spin
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
    def propellers_spin(self):
        """The spin direction for each propeller."""
        return self._propellers_spin

import torch

import genesis as gs
from genesis.repr_base import RBC


class ToolEntityState:
    """
    Dynamic state queried from a genesis ToolEntity.
    """

    def __init__(self, entity, s_global):
        self.entity = entity
        self.s_global = s_global

        self.pos = torch.zeros((self.entity.sim._B, 3), dtype=gs.tc_float, device=gs.device)
        self.quat = torch.zeros((self.entity.sim._B, 4), dtype=gs.tc_float, device=gs.device)
        self.vel = torch.zeros((self.entity.sim._B, 3), dtype=gs.tc_float, device=gs.device)
        self.ang = torch.zeros((self.entity.sim._B, 3), dtype=gs.tc_float, device=gs.device)

    @property
    def grad_tensors(self):
        """The tensors of the state that carry gradients."""
        return (self.pos, self.quat, self.vel, self.ang)

    # def __repr__(self):
    #     return f'{self.__repr_name__()}\n' \
    #            f'entity : {_repr(self.entity)}\n' \
    #            f'pos    : {_repr(self.pos)}\n' \
    #            f'quat   : {_repr(self.quat)}\n' \
    #            f'vel    : {_repr(self.vel)}\n' \
    #            f'ang    : {_repr(self.ang)}'


class MPMEntityState(RBC):
    """
    Dynamic state queried from a genesis MPMEntity.
    """

    def __init__(self, entity, s_global):
        self._entity = entity
        self._s_global = s_global
        base_shape = (self.entity.sim._B, self._entity.n_particles)
        self._pos = torch.zeros(base_shape + (3,), dtype=gs.tc_float, device=gs.device)
        self._vel = torch.zeros(base_shape + (3,), dtype=gs.tc_float, device=gs.device)
        self._C = torch.zeros(base_shape + (3, 3), dtype=gs.tc_float, device=gs.device)
        self._F = torch.zeros(base_shape + (3, 3), dtype=gs.tc_float, device=gs.device)
        self._Jp = torch.zeros(base_shape, dtype=gs.tc_float, device=gs.device)
        self._active = torch.zeros(base_shape, dtype=gs.tc_int, device=gs.device)

    @property
    def grad_tensors(self):
        """The tensors of the state that carry gradients."""
        return (self._pos, self._vel, self._C, self._F, self._Jp)

    @property
    def entity(self):
        return self._entity

    @property
    def s_global(self):
        return self._s_global

    @property
    def pos(self):
        return self._pos

    @property
    def vel(self):
        return self._vel

    @property
    def C(self):
        return self._C

    @property
    def F(self):
        return self._F

    @property
    def Jp(self):
        return self._Jp

    @property
    def active(self):
        return self._active


class SPHEntityState(RBC):
    """
    Dynamic state queried from a genesis SPHEntity.
    """

    def __init__(self, entity, s_global):
        self._entity = entity
        self._s_global = s_global
        base_shape = (self.entity.sim._B, self._entity.n_particles)
        self._pos = torch.zeros(base_shape + (3,), dtype=gs.tc_float, device=gs.device)
        self._vel = torch.zeros(base_shape + (3,), dtype=gs.tc_float, device=gs.device)

    @property
    def entity(self):
        return self._entity

    @property
    def s_global(self):
        return self._s_global

    @property
    def pos(self):
        return self._pos

    @property
    def vel(self):
        return self._vel


class FEMEntityState:
    """
    Dynamic state queried from a genesis FEMEntity.
    """

    def __init__(self, entity, s_global):
        self._entity = entity
        self._s_global = s_global
        base_shape = (self.entity.sim._B, self._entity.n_vertices, 3)

        self._pos = torch.zeros(base_shape, dtype=gs.tc_float, device=gs.device)
        self._vel = torch.zeros(base_shape, dtype=gs.tc_float, device=gs.device)
        self._active = torch.zeros((self.entity.sim._B, self.entity.n_elements), dtype=gs.tc_int, device=gs.device)

    @property
    def entity(self):
        return self._entity

    @property
    def s_global(self):
        return self._s_global

    @property
    def pos(self):
        return self._pos

    @property
    def vel(self):
        return self._vel

    @property
    def active(self):
        return self._active

    @property
    def grad_tensors(self):
        """The tensors of the state that carry gradients."""
        return (self._pos, self._vel)


class RigidEntityState(RBC):
    """
    Dynamic state queried from a genesis RigidEntity.
    """

    def __init__(self, entity, s_global, pos, quat):
        self._entity = entity
        self._s_global = s_global
        # Copies, so that editing this state leaves the solver state it was read from untouched
        self._pos = pos.clone()
        self._quat = quat.clone()

    @property
    def entity(self):
        return self._entity

    @property
    def s_global(self):
        return self._s_global

    @property
    def pos(self):
        return self._pos

    @property
    def quat(self):
        return self._quat

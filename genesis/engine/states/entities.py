import genesis as gs
from genesis.repr_base import RBC


class ToolEntityState:
    """
    Dynamic state queried from a genesis ToolEntity.
    """

    def __init__(self, entity, s_global):
        self.entity = entity
        self.s_global = s_global

        args = {
            "dtype": gs.tc_float,
            "requires_grad": self.entity.scene.requires_grad,
            "scene": self.entity.scene,
        }
        self.pos = gs.zeros((self.entity.sim._B, 3), **args)
        self.quat = gs.zeros((self.entity.sim._B, 4), **args)
        self.vel = gs.zeros((self.entity.sim._B, 3), **args)
        self.ang = gs.zeros((self.entity.sim._B, 3), **args)

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
        args = {
            "dtype": gs.tc_float,
            "requires_grad": self._entity.scene.requires_grad,
            "scene": self._entity.scene,
        }
        self._pos = gs.zeros(base_shape + (3,), **args)
        self._vel = gs.zeros(base_shape + (3,), **args)
        self._C = gs.zeros(base_shape + (3, 3), **args)
        self._F = gs.zeros(base_shape + (3, 3), **args)
        self._Jp = gs.zeros(base_shape, **args)

        args["dtype"] = int
        args["requires_grad"] = False
        self._active = gs.zeros(base_shape, **args)

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
        args = {
            "dtype": gs.tc_float,
            "requires_grad": False,
            "scene": self._entity.scene,
        }

        self._pos = gs.zeros(base_shape + (3,), **args)
        self._vel = gs.zeros(base_shape + (3,), **args)

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

        args = {
            "dtype": gs.tc_float,
            "requires_grad": False,
            "scene": self.entity.scene,
        }
        self._pos = gs.zeros(base_shape, **args)
        self._vel = gs.zeros(base_shape, **args)

        args["dtype"] = int
        args["requires_grad"] = False
        self._active = gs.zeros((self.entity.sim._B, self.entity.n_elements), **args)

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


class RigidEntityState(RBC):
    """
    Dynamic state queried from a genesis RigidEntity.
    """

    def __init__(self, entity, s_global):
        self._entity = entity
        self._s_global = s_global

        num_batch = self._entity._solver._B
        requires_grad = self._entity.scene.requires_grad
        scene = self._entity.scene
        self._pos = gs.zeros((num_batch, 3), dtype=float, requires_grad=requires_grad, scene=scene)
        self._quat = gs.zeros((num_batch, 4), dtype=float, requires_grad=requires_grad, scene=scene)

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

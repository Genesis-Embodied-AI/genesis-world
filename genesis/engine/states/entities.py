from genesis.repr_base import RBC


class MPMEntityState(RBC):
    """
    Dynamic state queried from a genesis MPMEntity.
    """

    def __init__(self, entity, s_global, pos, vel, C, F, Jp, active):
        self._entity = entity
        self._s_global = s_global
        self._pos = pos
        self._vel = vel
        self._C = C
        self._F = F
        self._Jp = Jp
        self._active = active

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

    def __init__(self, entity, s_global, pos, vel):
        self._entity = entity
        self._s_global = s_global
        self._pos = pos
        self._vel = vel

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

    def __init__(self, entity, s_global, pos, vel):
        self._entity = entity
        self._s_global = s_global
        self._pos = pos
        self._vel = vel

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

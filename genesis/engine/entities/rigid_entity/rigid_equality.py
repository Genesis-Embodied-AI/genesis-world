import torch

import genesis as gs
from genesis.repr_base import RBC
from genesis.typing import FArrayType
from genesis.utils.misc import BuildableMixin

from .description import RigidEqualityDescription


class RigidEquality(BuildableMixin, RBC):
    """
    Equality class for rigid body entities.
    """

    def __init__(self, entity, idx, eq_obj1id, eq_obj2id, desc: RigidEqualityDescription):
        self.desc: RigidEqualityDescription = desc
        self._entity = entity
        self._solver = entity.solver

        self._uid = gs.UID()
        self._idx = idx

        self._eq_obj1id = eq_obj1id
        self._eq_obj2id = eq_obj2id

    @gs.assert_unbuilt
    def _build(self) -> None:
        self._is_built = True

    def _destroy(self) -> None:
        self._is_built = False

    # ------------------------------------------------------------------------------------
    # -------------------------------- real-time state -----------------------------------
    # ------------------------------------------------------------------------------------

    @gs.assert_built
    def set_sol_params(self, sol_params: "FArrayType | torch.Tensor") -> None:
        """
        Set the solver parameters of this equality constraint.
        """
        self._solver.set_sol_params(sol_params, eqs_idx=self._idx, envs_idx=None)

    @gs.assert_built
    def get_sol_params(self):
        """
        Get the solver parameters the simulation is currently using for this equality constraint.
        """
        return self._solver.get_sol_params(eqs_idx=self._idx, envs_idx=None)[..., 0, :]

    # ------------------------------------------------------------------------------------
    # ----------------------------------- properties -------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def uid(self):
        """
        Returns the unique id of the equality.
        """
        return self._uid

    @property
    def name(self):
        """
        Returns the name of the equality.
        """
        return self.desc.name

    @property
    def entity(self):
        """
        Returns the entity that the equality belongs to.
        """
        return self._entity

    @property
    def solver(self):
        """
        The RigidSolver object that the equality belongs to.
        """
        return self._solver

    @property
    def idx(self):
        """
        Returns the global index of the equality in the rigid solver.
        """
        return self._idx

    def _repr_brief(self):
        return f"{self.__repr_name__()}, idx: {self.idx}"

    @property
    def idx_local(self):
        """
        Returns the local index of the equality in the entity.
        """
        return self._idx - self._entity._equality_start

    @property
    def type(self):
        """
        Returns the type of the equality.
        """
        return self.desc.type

    @property
    def eq_obj1id(self):
        """
        Returns the index of the first object (joint for EQUALITY_TYPE.JOINT, link otherwise)
        """
        return self._eq_obj1id

    @property
    def eq_obj2id(self):
        """
        Returns the index of the second object, or -1 for a JOINT equality that fixes one joint to a constant.
        """
        return self._eq_obj2id

    @property
    def eq_data(self):
        """
        Returns the eq_data of this equality constraint.
        """
        return self.desc.data

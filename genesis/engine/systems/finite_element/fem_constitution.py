from __future__ import annotations

from abc import abstractmethod

from ..global_linear_system import GlobalLinearSystem
from ..sim_system import SimSystem
from .finite_element_method import FiniteElementMethod


class FEMConstitution(SimSystem):
    """Base contract for FEM constitutions."""

    def build(self) -> None:
        self.fem_system = self.require(FiniteElementMethod)
        self.global_linear_system_system = self.require(GlobalLinearSystem)
        self.data.extent_slot = self.global_linear_system_system.register_extent_slot()
        extent_action, assemble_action, energy_action = self.create_constitution_actions()
        self.fem_system.register_constitution_actions(
            self,
            extent_action,
            assemble_action,
            energy_action,
        )

    @abstractmethod
    def create_constitution_actions(self):
        """Create the concrete pure-kernel action protocol."""

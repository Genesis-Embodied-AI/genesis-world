from __future__ import annotations

from abc import abstractmethod

from ..sim_system import SimSystem
from .finite_element_method import FiniteElementMethod


class FEMConstitution(SimSystem):
    """Base contract for CGQ FEM constitutions."""

    def do_build(self) -> None:
        self.require(FiniteElementMethod).add_constitution(self)
        self.do_build_constitution()

    def do_build_constitution(self) -> None:
        pass

    @abstractmethod
    def report_extent(self, fem, global_linear_system):
        pass

    @abstractmethod
    def assemble(self, fem, sim_config, global_linear_system):
        pass

    @abstractmethod
    def energy(self, fem, sim_config, energy):
        pass

from __future__ import annotations

from abc import abstractmethod

from .sim_system import SimSystem


class ContactConstitution(SimSystem):
    """Base contract for graph-native contact constitutions."""

    def build(self) -> None:
        from .contact_system import ContactSystem

        self.require(ContactSystem).set_contact_constitution(self)
        self.do_build_constitution()

    def do_build_constitution(self) -> None:
        pass

    @abstractmethod
    def count_active(self, contact, surface, vertex):
        pass

    @abstractmethod
    def filter_assemble(self, contact, surface, vertex):
        pass

    @abstractmethod
    def contact_energy(self, contact, surface, vertex):
        pass

    @abstractmethod
    def friction_snapshot(self, contact, surface, vertex):
        pass

"""Particle ranges of the entities of the particle solvers."""

from dataclasses import dataclass

from . import SystemData


@dataclass(frozen=True, kw_only=True, eq=False)
class ParticleEntityData(SystemData):
    """The particle range of one entity in the aggregate record of its solver, in entity order."""

    data: SystemData
    particle_start: int
    particle_end: int

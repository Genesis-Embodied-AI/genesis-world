import numpy as np
import torch

import genesis as gs
import genesis.utils.geom as gu
import genesis.utils.particle as pu
from genesis.engine.entities.particle_entity import ParticleEntity, check_particles_inside_boundary
from genesis.repr_base import RBC


class Emitter(RBC):
    """
    A particle emitter for fluid or material simulation.

    The Emitter manages the generation of particles into the simulation domain, allowing directional or omnidirectional
    emissions with various droplet shapes. It supports resetting, shape-based emission, and spherical omni-emission.

    Parameters
    ----------
    entity : ParticleEntity
        The entity holding the particles the emitter releases.
    max_particles : int
        The maximum number of particles that this emitter can handle.
    """

    def __init__(self, entity: "ParticleEntity", max_particles: int) -> None:
        self._uid = gs.UID()
        self._entity = entity
        self._max_particles = max_particles
        self._next_particle = 0

        self._acc_droplet_len = 0.0  # accumulated droplet length to be emitted

        gs.logger.info(
            f"Creating ~<{self.__repr_name__()}>~. id: ~~~<{self._uid}>~~~, max_particles: ~<{max_particles}>~, "
            f"entity: ~<{entity._repr_briefer()}>~."
        )

    def reset(self):
        """
        Reset the emitter's internal particle index and accumulated droplet length to start emitting from the beginning.
        """
        self._next_particle = 0
        self._acc_droplet_len = 0.0

    def emit(
        self,
        droplet_shape,
        droplet_size,
        droplet_length=None,
        pos=(0.5, 0.5, 1.0),
        direction=(0, 0, -1),
        theta=0.0,
        speed=1.0,
        p_size=None,
    ):
        """
        Emit particles in a specified shape and direction from a nozzle.

        Parameters
        ----------
        droplet_shape : str
            The shape of the emitted droplet. Options: "circle", "sphere", "square", "rectangle", "omni". An "omni"
            droplet is a spherical shell of particles moving radially outward, filled between 0.4 and 1.0 times its
            radius, and is emitted at every call whatever `droplet_length`, `direction` and `theta`.
        droplet_size : float or tuple
            Size of the droplet. A single float for symmetric shapes (the diameter for "circle", "sphere" and "omni"),
            or a tuple of (width, height) for rectangles.
        droplet_length : float, optional
            Length of the droplet in the emitting direction. If None, calculated from speed and simulation timing.
        pos : tuple of float
            World position of the nozzle from which the droplet is emitted.
        direction : tuple of float
            Direction vector of the emitted droplet.
        theta : float
            Rotation angle (in radians) around the droplet axis.
        speed : float
            Emission speed of the particles.
        p_size : float, optional
            Particle size used for filling the droplet. Defaults to the solver's particle size. It only sets the number
            of particles emitted, while the particles keep the size of the solver in simulation and rendering.

        Raises
        ------
        Exception
            If the shape is unsupported, the size is invalid, or the emission would place particles outside the
            simulation boundary.
        """
        assert self._entity is not None

        if droplet_shape in ["circle", "sphere", "square", "omni"]:
            assert isinstance(droplet_size, (int, float))
        elif droplet_shape == "rectangle":
            assert isinstance(droplet_size, (tuple, list)) and len(droplet_size) == 2
        else:
            gs.raise_exception(f"Unsupported nozzle shape: {droplet_shape}.")

        direction = np.asarray(direction, dtype=gs.np_float)
        if np.linalg.norm(direction) < gs.EPS:
            gs.raise_exception("Zero-length direction.")
        else:
            direction = gu.normalize(direction)

        p_size = self._entity.particle_size if p_size is None else p_size

        if droplet_shape == "omni":
            if not 0.0 < droplet_size < float("inf"):
                gs.raise_exception(f"The size of an omni droplet must be positive and finite, got {droplet_size}.")
            positions_ = pu.shell_to_particles(
                p_size=p_size,
                outer_radius=droplet_size / 2,
                inner_radius=0.2 * droplet_size,
                sampler=self._entity.sampler,
            )
            positions = np.asarray(pos, dtype=gs.np_float) + positions_
            dists = np.linalg.norm(positions_, axis=1)
            positions[dists < gs.EPS] = gs.EPS
            vels = (speed / (dists[:, None] + gs.EPS)) * positions_
        else:
            if droplet_length is None:
                # Use the speed to determine the length of the droplet in the emitting direction
                droplet_length = speed * self._entity.solver.sim.dt + self._acc_droplet_len
                if droplet_length < p_size:  # too short, so we should not emit
                    self._acc_droplet_len = droplet_length
                    droplet_length = 0.0
                else:
                    self._acc_droplet_len = 0.0

            if droplet_length <= 0.0:
                gs.logger.debug("Droplet length is too short for current step. Skipping to next step.")
                return

            if droplet_shape == "circle":
                positions = pu.cylinder_to_particles(
                    p_size=p_size,
                    radius=droplet_size / 2,
                    height=droplet_length,
                    sampler=self._entity.sampler,
                )
            elif droplet_shape == "sphere":  # sphere droplet ignores droplet_length
                positions = pu.sphere_to_particles(
                    p_size=p_size,
                    radius=droplet_size / 2,
                    sampler=self._entity.sampler,
                )
            elif droplet_shape == "square":
                positions = pu.box_to_particles(
                    p_size=p_size,
                    size=np.array([droplet_size, droplet_size, droplet_length]),
                    sampler=self._entity.sampler,
                )
            else:
                positions = pu.box_to_particles(
                    p_size=p_size,
                    size=np.array([droplet_size[0], droplet_size[1], droplet_length]),
                    sampler=self._entity.sampler,
                )

            positions = gu.transform_by_trans_R(
                positions.astype(gs.np_float, copy=False),
                np.asarray(pos, dtype=gs.np_float),
                gu.z_up_to_R(direction) @ gu.axis_angle_to_R(np.array([0.0, 0.0, 1.0], dtype=gs.np_float), theta),
            )
            vels = speed * direction

        check_particles_inside_boundary(positions, self.solver.boundary_lower, self.solver.boundary_upper)

        n_particles = len(positions)
        if n_particles > self._entity.n_particles:
            gs.raise_exception(
                f"Number of particles to emit ({n_particles}) at the current step is larger than the maximum number "
                f"of particles ({self._entity.n_particles})."
            )

        # The particles are recycled from the first one once an emission would run past the last one
        if self._next_particle + n_particles > self._entity.n_particles:
            self._next_particle = 0
        particles_idx = torch.arange(
            self._next_particle, self._next_particle + n_particles, dtype=gs.tc_int, device=gs.device
        )
        self._entity.set_particles_pos(positions, particles_idx)
        self._entity.set_particles_vel(vels, particles_idx)
        self._entity.set_particles_active(gs.ACTIVE, particles_idx)
        self._next_particle = (self._next_particle + n_particles) % self._entity.n_particles

        gs.logger.debug(f"Emitted {n_particles} particles. Next particle index: {self._next_particle}.")

    def emit_omni(self, source_radius=0.1, pos=(0.5, 0.5, 1.0), speed=1.0, particle_size=None):
        """
        Use a sphere-shaped source to emit particles in all directions.

        Deprecated: use `emit(droplet_shape="omni", droplet_size=2 * source_radius, ...)` instead.
        """
        gs.logger.warning(
            "'emit_omni' is deprecated and will be removed in a future release. Use "
            "'emit(droplet_shape=\"omni\", droplet_size=2 * source_radius, ...)' instead."
        )
        self.emit(droplet_shape="omni", droplet_size=2 * source_radius, pos=pos, speed=speed, p_size=particle_size)

    @property
    def uid(self):
        """The unique identifier of the emitter."""
        return self._uid

    @property
    def entity(self):
        """The entity associated with the emitter."""
        return self._entity

    @property
    def max_particles(self):
        """The maximum number of particles this emitter can emit."""
        return self._max_particles

    @property
    def solver(self):
        """The solver used by the emitter's associated entity."""
        return self._entity.solver

    @property
    def next_particle(self):
        """The index of the next particle to be emitted."""
        return self._next_particle

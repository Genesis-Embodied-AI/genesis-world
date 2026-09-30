import numpy as np
import torch

import genesis as gs
import genesis.utils.geom as gu
import genesis.utils.particle as pu
from genesis.repr_base import RBC


class Emitter(RBC):
    """
    A particle emitter for fluid or material simulation.

    The Emitter manages the generation of particles into the simulation domain, allowing directional or omnidirectional
    emissions with various droplet shapes. It supports resetting, shape-based emission, and spherical omni-emission.

    Parameters
    ----------
    max_particles : int
        The maximum number of particles that this emitter can handle.
    """

    def __init__(self, max_particles):
        self._uid = gs.UID()
        self._entity = None

        self._max_particles = max_particles

        self._acc_droplet_len = 0.0  # accumulated droplet length to be emitted

        gs.logger.info(
            f"Creating ~<{self.__repr_name__()}>~. id: ~~~<{self._uid}>~~~, max_particles: ~<{max_particles}>~."
        )

    def set_entity(self, entity):
        """
        Assign an entity to the emitter and initialize relevant simulation and solver references.

        Parameters
        ----------
        entity : Entity
            The entity to associate with the emitter. This entity should contain the solver, simulation context, and particle sampler.
        """
        self._entity = entity
        self._sim = entity.sim
        self._solver = entity.solver
        self._next_particle = 0
        gs.logger.info(f"~<{self._repr_briefer()}>~ created using ~<{entity._repr_briefer()}>~.")

    def reset(self):
        """
        Reset the emitter's internal particle index to start emitting from the beginning.
        """
        self._next_particle = 0

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
            The shape of the emitted droplet. Options: "circle", "sphere", "square", "rectangle".
        droplet_size : float or tuple
            Size of the droplet. A single float for symmetric shapes, or a tuple of (width, height) for rectangles.
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
            Particle size used for filling the droplet. Defaults to the solver's particle size.

        Raises
        ------
        Exception
            If the shape is unsupported or the emission would place particles outside the simulation boundary.
        """
        assert self._entity is not None

        if droplet_shape in ["circle", "sphere", "square"]:
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

        if droplet_length is None:
            # Use the speed to determine the length of the droplet in the emitting direction
            droplet_length = speed * self._solver.substep_dt * self._sim.substeps + self._acc_droplet_len
            if droplet_length < p_size:  # too short, so we should not emit
                self._acc_droplet_len = droplet_length
                droplet_length = 0.0
            else:
                self._acc_droplet_len = 0.0

        if droplet_length > 0.0:
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
            elif droplet_shape == "rectangle":
                positions = pu.box_to_particles(
                    p_size=p_size,
                    size=np.array([droplet_size[0], droplet_size[1], droplet_length]),
                    sampler=self._entity.sampler,
                )
            else:
                gs.raise_exception(f"Unsupported droplet shape '{droplet_shape}'")

            positions = gu.transform_by_trans_R(
                positions.astype(gs.np_float, copy=False),
                np.asarray(pos, dtype=gs.np_float),
                gu.z_up_to_R(direction) @ gu.axis_angle_to_R(np.array([0.0, 0.0, 1.0], dtype=gs.np_float), theta),
            )

            if not self._solver.boundary.is_inside(positions):
                gs.raise_exception("Emitted particles are outside the boundary.")

            n_particles = len(positions)

            # Expand vels with batch dimension
            vels = speed * direction

            if n_particles > self._entity.n_particles:
                gs.raise_exception(
                    f"Number of particles to emit ({n_particles}) at the current step is larger than the maximum "
                    f"number of particles ({self._entity.n_particles})."
                )

            particles_idx = torch.arange(
                self._next_particle, self._next_particle + n_particles, dtype=gs.tc_int, device=gs.device
            )

            self._entity.set_particles_pos(positions, particles_idx)
            self._entity.set_particles_vel(vels, particles_idx)
            self._entity.set_particles_active(gs.ACTIVE, particles_idx)

            self._next_particle += n_particles

            # recycle particles
            if self._next_particle + n_particles > self._entity.n_particles:
                self._next_particle = 0

            gs.logger.debug(f"Emitted {n_particles} particles. Next particle index: {self._next_particle}.")

        else:
            gs.logger.debug("Droplet length is too short for current step. Skipping to next step.")

    def emit_omni(self, source_radius=0.1, pos=(0.5, 0.5, 1.0), speed=1.0, particle_size=None):
        """
        Use a sphere-shaped source to emit particles in all directions.

        Parameters:
        ----------
        source_radius: float, optional
            The radius of the sphere source. Particles will be emitted from a shell with inner radius using
            '0.8 * source_radius' and outer radius using source_radius.
        pos: array_like, shape=(3,)
            The center of the sphere source.
        speed: float
            The speed of the emitted particles.
        particle_size: float | None
            The size (diameter) of the emitted particles. The actual number of particles emitted is determined by the
            volume of the sphere source and the size of the particles. If None, the solver's particle size is used.
            Note that this particle size only affects computation for number of particles emitted, not the actual size
            of the particles in simulation and rendering.
        """
        assert self._entity is not None

        pos = np.asarray(pos, dtype=gs.np_float)

        if particle_size is None:
            particle_size = self._entity.particle_size

        positions_ = pu.shell_to_particles(
            p_size=particle_size,
            outer_radius=source_radius,
            inner_radius=source_radius * 0.4,
            sampler=self._entity.sampler,
        )
        positions = pos + positions_

        if not self._solver.boundary.is_inside(positions):
            gs.raise_exception("Emitted particles are outside the boundary.")

        dists = np.linalg.norm(positions_, axis=1)
        positions[dists < gs.EPS] = gs.EPS
        vels = (speed / (dists[:, None] + gs.EPS)) * positions_

        n_particles = len(positions)
        if n_particles > self._entity.n_particles:
            gs.raise_exception(
                f"Number of particles to emit ({n_particles}) at the current step is larger than the maximum number "
                f"of particles ({self._entity.n_particles})."
            )

        particles_idx = torch.arange(
            self._next_particle, self._next_particle + n_particles, dtype=gs.tc_int, device=gs.device
        )

        self._entity.set_particles_pos(positions, particles_idx)
        self._entity.set_particles_vel(vels, particles_idx)
        self._entity.set_particles_active(gs.ACTIVE, particles_idx)

        self._next_particle += n_particles

        # recycle particles
        if self._next_particle + n_particles > self._entity.n_particles:
            self._next_particle = 0

        gs.logger.debug(f"Emitted {n_particles} particles. Next particle index: {self._next_particle}.")

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
        return self._solver

    @property
    def next_particle(self):
        """The index of the next particle to be emitted."""
        return self._next_particle

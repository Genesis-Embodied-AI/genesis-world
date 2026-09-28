from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

import genesis as gs
from genesis.engine.materials.base import Material
from genesis.options.morphs import Morph
from genesis.repr_base import RBC

if TYPE_CHECKING:
    from genesis.engine.scene import Scene
    from genesis.engine.sensors.base_sensor import Sensor


@dataclass
class EntityDescription:
    """Base class of what one entity of a scene is created from, whatever the solver simulating it.

    A scene names every entity it holds as one of these, so a kind of entity described later stands there too. The
    material is what every description holds, since the simulator picks the solver from it.
    """

    material: Material


class Entity(RBC):
    """
    Base class for all types of entities.
    """

    def __init__(
        self,
        idx,
        scene,
        morph,
        solver,
        material,
        surface,
        name: str | None = None,
    ):
        # An entity is created from one morph. A kind of entity built from several passes the primary one here.
        if not isinstance(morph, Morph):
            gs.raise_exception(f"An entity is created from one morph, got {type(morph).__name__}.")
        uid = gs.UID()
        while any(entity.uid.match(uid, short_only=True) for entity in scene.entities):
            uid = gs.UID()
        self._uid = uid
        self._idx = idx
        self._scene: "Scene" = scene
        self._solver = solver
        self._material = material
        self._morph = morph
        self._surface = surface
        self._sim = scene.sim

        # Set entity name (auto-generate if not provided)
        existing_names = {entity.name for entity in scene.entities if entity.name is not None}
        if name is not None:
            if name in existing_names:
                gs.raise_exception(f"Entity name '{name}' already exists in scene.")
            self._name = name
        else:
            morph_name = self._get_morph_identifier()
            self._name = f"{morph_name}_{uid.short()}"

        gs.logger.info(
            f"Adding ~<{self.__repr_name__()}>~. idx: ~<{self._idx}>~, uid: ~~~<{self._uid}>~~~, morph: ~<{morph}>~, material: ~<{self._material}>~."
        )

    # ------------------------------------------------------------------------------------
    # ----------------------------------- properties -------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def uid(self):
        return self._uid

    @property
    def idx(self):
        return self._idx

    @property
    def scene(self):
        return self._scene

    @property
    def sim(self):
        return self._sim

    @property
    def solver(self):
        return self._solver

    @property
    def surface(self):
        return self._surface

    @property
    def morph(self):
        return self._morph

    @property
    def desc(self) -> EntityDescription | None:
        """The description this entity was created from, or None for a kind of entity no description carries.

        'Scene.export' writes every entity as its description and rejects a scene holding an entity without one. A
        kind of entity gains export support by returning its description here.
        """
        return None

    @property
    def material(self):
        return self._material

    @property
    def is_built(self):
        return self._solver._scene._is_built

    def _repr_brief(self):
        return f"{self.__repr_name__()}, idx: {self.idx}, morph: {self._repr_morph()}, material: {self.material}"

    def _repr_morph(self):
        return f"{self.morph}"

    @property
    def name(self) -> str:
        """
        The name of this entity.

        Returns
        -------
        str
            The entity's name. If a user-specified name was provided during creation,
            that name is returned. Otherwise, an auto-generated name based on the
            morph type and UID is returned.
        """
        return self._name

    @property
    def sensors(self) -> "gs.List[Sensor]":
        """List of sensors attached to this entity."""
        return self._sim._sensor_manager.get_sensors_by_entity(self._idx)

    @gs.assert_built
    def read_sensors(self, envs_idx=None) -> "dict[type[Sensor], torch.Tensor]":
        """
        Read every sensor attached to this entity as a tensor per sensor class.

        Always returns a fresh tensor independent of the internal sensor storage; the caller is free to mutate the
        result.

        Parameters
        ----------
        envs_idx : array-like | int | slice | None
            Environment selection. Defaults to all environments.

        Returns
        -------
        dict[Type[Sensor], torch.Tensor]
            For each sensor class with at least one sensor on this entity, a tensor of shape
            (B, [history,] entity_cache_size_for_class).
        """
        return self._sim._sensor_manager.read_sensors(entity_idx=self._idx, envs_idx=envs_idx)

    # ------------------------------------------------------------------------------------
    # --------------------------------- naming methods -----------------------------------
    # ------------------------------------------------------------------------------------

    def _get_morph_identifier(self) -> str:
        """
        Get the identifier string from the morph for name generation.

        Must be overridden in subclasses to provide type-specific identifiers.
        """
        raise NotImplementedError

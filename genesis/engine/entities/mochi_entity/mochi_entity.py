from typing import TYPE_CHECKING

import torch

import genesis as gs
from genesis.engine.entities.rigid_entity import RigidEntity

if TYPE_CHECKING:
    from genesis.engine.solvers.mochi import MochiSolver


def filter_entity_contacts(solver, entity, with_entity, exclude_self_contact, is_padded):
    """Contact points of the solver involving `entity` (and `with_entity`), see `MochiEntity.get_contacts`."""
    contact_data = solver.get_contacts(as_tensor=True, to_torch=True, is_padded=is_padded)
    n_contacts = contact_data["n_contacts"] if is_padded else None
    if is_padded:
        del contact_data["n_contacts"]

    logical_operation = torch.logical_xor if exclude_self_contact else torch.logical_or
    if with_entity is not None and entity.idx == with_entity.idx:
        if exclude_self_contact:
            gs.raise_exception("`with_entity` is self but `exclude_self_contact` is True.")
        logical_operation = torch.logical_and

    valid_mask = logical_operation(contact_data["entity_a"] == entity.idx, contact_data["entity_b"] == entity.idx)
    if with_entity is not None and entity.idx != with_entity.idx:
        valid_mask = torch.logical_and(
            valid_mask,
            torch.logical_or(contact_data["entity_a"] == with_entity.idx, contact_data["entity_b"] == with_entity.idx),
        )
    if n_contacts is not None:
        slots = torch.arange(valid_mask.shape[-1], device=valid_mask.device)
        if solver.n_envs == 0:
            valid_mask = torch.logical_and(valid_mask, slots < n_contacts.reshape(()))
        else:
            valid_mask = torch.logical_and(valid_mask, slots[None, :] < n_contacts[:, None])

    if solver.n_envs == 0 and not is_padded:
        contact_data = {key: value[valid_mask] for key, value in contact_data.items()}
    else:
        contact_data["valid_mask"] = valid_mask
    return contact_data


class MochiEntity(RigidEntity):
    """
    Rigid entity simulated by the MochiSolver.

    Loading, kinematics, pose/velocity access and joint control are shared with RigidEntity. Contact is the solver's
    smooth penalty model, so the contact readback exposes the sample-based contact points; the mass-matrix and
    constraint-solver specific accessors of the rigid solver are unavailable.
    """

    _solver: "MochiSolver"

    def get_contacts(self, with_entity=None, exclude_self_contact=False, is_padded=False):
        """
        Returns the contact points computed at the end of the most recent `scene.step()`, filtered to those involving
        this entity (and `with_entity` if given).

        A contact point is a sample of the collision surface of side A pressed into the distance field of side B
        (a rigid geom, or a deformable body). The returned dict has the keys 'entity_a', 'entity_b' (scene entity
        indices), 'link_a', 'link_b', 'geom_a', 'geom_b' (-1 for a deformable side), 'verts_a', 'bary_a', 'verts_b',
        'bary_b' (entity-local vertices and weights a deformable side is spread over, -1 padded), 'position', 'normal'
        (unit, pointing away from B), 'distance' (signed, negative when penetrating), 'force_a' (force on A),
        'force_b' (= -force_a), 'weight' (surface measure the sample stands for) and, for a parallelized scene,
        'valid_mask'. Shapes are (n_envs, n_contacts, ...) for a parallelized scene and (n_contacts, ...) otherwise.
        """
        return filter_entity_contacts(self._solver, self, with_entity, exclude_self_contact, is_padded)

    def get_links_net_contact_force(self, envs_idx=None):
        self._solver._record_contacts()
        return super().get_links_net_contact_force(envs_idx)

    def set_contact_params(
        self,
        *,
        penalty_coefficient=None,
        friction=None,
        penalty_smoothing_half_distance=None,
        penalty_threshold=None,
        friction_falloff_vel=None,
        viscous_friction=None,
        normal_viscous_damping=None,
        links_idx_local=None,
    ):
        """
        Update the contact parameters of the collision geoms of the given links (all links by default). See
        `gs.materials.Mochi.Rigid` for the meaning of each parameter.
        """
        links_idx = self._get_global_idx(links_idx_local, self.n_links, self._link_start, unsafe=True)
        self._solver.set_links_contact_params(
            links_idx,
            penalty_coefficient=penalty_coefficient,
            friction=friction,
            penalty_smoothing_half_distance=penalty_smoothing_half_distance,
            penalty_threshold=penalty_threshold,
            friction_falloff_vel=friction_falloff_vel,
            viscous_friction=viscous_friction,
            normal_viscous_damping=normal_viscous_damping,
        )

    def set_friction(self, friction):
        self.set_contact_params(friction=friction)

    def set_has_gravity(self, has_gravity):
        """Enable or disable gravity on every link of the entity."""
        links_idx = self._get_global_idx(None, self.n_links, self._link_start, unsafe=True)
        self._solver.set_links_has_gravity(links_idx, has_gravity)

    def _raise_unsupported(self, name):
        gs.raise_exception(f"`{name}` is not supported by entities simulated by the MochiSolver.")

    def get_dofs_force(self, dofs_idx_local=None, envs_idx=None):
        self._raise_unsupported("get_dofs_force")

    def get_mass_mat(self, envs_idx=None, decompose=False):
        self._raise_unsupported("get_mass_mat")

    def set_friction_ratio(self, friction_ratio, links_idx_local=None, envs_idx=None):
        self._raise_unsupported("set_friction_ratio")

    def set_friction_torsional(self, friction_torsional):
        self._raise_unsupported("set_friction_torsional")

    def set_friction_rolling(self, friction_rolling):
        self._raise_unsupported("set_friction_rolling")

    def set_mass_shift(self, mass_shift, links_idx_local=None, envs_idx=None):
        self._raise_unsupported("set_mass_shift")

    def set_COM_shift(self, com_shift, links_idx_local=None, envs_idx=None):
        self._raise_unsupported("set_COM_shift")

    def apply_links_external_wrench(self, *args, **kwargs):
        self._raise_unsupported("apply_links_external_wrench")

    def detect_collision(self, env_idx=0):
        self._raise_unsupported("detect_collision")

    def get_kinetic_energy(self, envs_idx=None):
        self._raise_unsupported("get_kinetic_energy")

    def get_potential_energy(self, envs_idx=None):
        self._raise_unsupported("get_potential_energy")

    def get_total_energy(self, envs_idx=None):
        self._raise_unsupported("get_total_energy")

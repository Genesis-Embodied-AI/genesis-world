"""
Data classes for IPC coupler.
"""

from enum import IntEnum
from dataclasses import dataclass
from typing import NamedTuple, TYPE_CHECKING

import numpy as np

from uipc.geometry import GeometrySlot

import genesis as gs

if TYPE_CHECKING:
    from genesis.engine.entities.rigid_entity import RigidLink


class COUPLING_TYPE(IntEnum):
    TWO_WAY_SOFT_CONSTRAINT = 0
    EXTERNAL_ARTICULATION = 1
    IPC_ONLY = 2
    NONE = 3


class ABDLinkEntry(NamedTuple):
    """Per-link, per-env ABD state retrieved from IPC after advance."""

    transform: np.ndarray  # (4, 4) IPC transform
    velocity: np.ndarray  # (4, 4) velocity matrix


@dataclass
class ArticulatedEntityData:
    """Typed container for per-entity articulation coupling data."""

    joints_child_link: list["RigidLink"]
    joints_q_idx_local: list[int]

    articulation_slots: list[GeometrySlot]

    ref_dof_prev: np.ndarray
    qpos_stored: np.ndarray
    qpos_current: np.ndarray
    qpos_new: np.ndarray
    delta_theta_tilde: np.ndarray
    delta_theta_ipc: np.ndarray

    # Previous timestep link transforms for ref_dof_prev computation {(joint, env_idx): transform_matrix_4x4}
    prev_links_transform: list[list[np.ndarray | None]]


class IPCCouplingData:
    """Pre-allocated arrays for coupling force computation."""

    def __init__(
        self,
        links: list["RigidLink"],
        abd_body_idx_by_link: dict["RigidLink", list[int]],
        n_envs: int,
    ):
        n_links = len(links)
        assert set(abd_body_idx_by_link.keys()) == set(links)

        self.links = links
        self.abd_body_idx_by_link = abd_body_idx_by_link
        self.links_idx = [link.idx for link in links]
        self.link_to_idx_local = {link: i for i, link in enumerate(links)}
        self.links_mass = np.array([link.desc.mass for link in links], dtype=gs.np_float)
        if links:
            self.links_inertia_i = np.stack([link.desc.inertia for link in links], axis=0, dtype=gs.np_float)
        else:
            self.links_inertia_i = np.empty((0, 0, 3, 3), dtype=gs.np_float)

        self.ipc_transforms = np.empty((n_envs, n_links, 4, 4), dtype=gs.np_float)
        self.aim_transforms = np.empty((n_envs, n_links, 4, 4), dtype=gs.np_float)
        self.out_forces = np.empty((n_envs, n_links, 3), dtype=gs.np_float)
        self.out_torques = np.empty((n_envs, n_links, 3), dtype=gs.np_float)

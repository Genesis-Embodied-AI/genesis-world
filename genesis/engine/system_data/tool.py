"""Pose history and signed distance fields for differentiable tools."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import quadrants as qd

import genesis as gs

from . import SystemData, SystemDataDescription

if TYPE_CHECKING:
    from genesis.engine.entities.tool_entity.tool_entity import ToolEntityDescription
    from genesis.engine.solvers.base_solver import Solver


@dataclass(frozen=True, kw_only=True, eq=False)
class ToolEntityData(SystemData):
    """The pose history of one tool entity over the local substeps, and the signed distance field of its mesh.

    The signed distance field is None for a tool whose material disables collision.
    """

    pos: qd.Field
    quat: qd.Field
    vel: qd.Field
    ang: qd.Field
    sdf_voxels: qd.Field | None
    T_mesh_to_sdf: qd.Field | None


@dataclass(frozen=True, kw_only=True, eq=False)
class ToolData(SystemData):
    """Every tool entity holds fields of its own, so the aggregate record holds the 'ToolEntityData' of each entity."""

    entities: tuple[ToolEntityData, ...]


@dataclass(frozen=True, kw_only=True)
class ToolDescription(SystemDataDescription[ToolData]):
    _B: int
    substeps_local: int
    entities_idx: tuple[int, ...]
    entities: tuple["ToolEntityDescription", ...]

    def allocate(self, owner: "Solver") -> ToolData:
        entities_data = []
        for i_e, entity in zip(self.entities_idx, self.entities, strict=True):
            pos = qd.Vector.field(n=3, dtype=gs.qd_float, needs_grad=True)
            quat = qd.Vector.field(n=4, dtype=gs.qd_float, needs_grad=True)
            vel = qd.Vector.field(n=3, dtype=gs.qd_float, needs_grad=True)
            ang = qd.Vector.field(n=3, dtype=gs.qd_float, needs_grad=True)
            qd.root.dense(qd.ij, (self.substeps_local + 1, self._B)).place(
                pos, pos.grad, quat, quat.grad, vel, vel.grad, ang, ang.grad
            )
            sdf_voxels = T_mesh_to_sdf = None
            if entity.sdf_voxels is not None:
                sdf_voxels = qd.field(dtype=gs.qd_float, shape=entity.sdf_voxels.shape)
                T_mesh_to_sdf = qd.Matrix.field(n=4, m=4, dtype=gs.qd_float, shape=())
                sdf_voxels.from_numpy(entity.sdf_voxels)
                T_mesh_to_sdf.from_numpy(entity.T_mesh_to_sdf)
            entities_data.append(
                ToolEntityData(
                    owner=owner,
                    idx=i_e,
                    pos=pos,
                    quat=quat,
                    vel=vel,
                    ang=ang,
                    sdf_voxels=sdf_voxels,
                    T_mesh_to_sdf=T_mesh_to_sdf,
                )
            )
        return ToolData(owner=owner, entities=tuple(entities_data))

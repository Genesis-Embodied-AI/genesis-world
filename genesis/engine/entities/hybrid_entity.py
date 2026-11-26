import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple, Self

import numpy as np
import torch

import trimesh

import genesis as gs
import genesis.utils.geom as gu
import genesis.utils.urdf as uu
from genesis.engine.materials.base import Material
from genesis.ext.urdfpy.urdf import URDF
from genesis.options.surfaces import Surface
from genesis.utils.hybrid import (
    compute_graph_attribute,
    gel_graph_to_nx_graph,
    graph_to_tree,
    reduce_graph,
    skeletonization,
    trimesh_to_gelmesh,
)
from genesis.utils.mesh import load_mesh

from .base_entity import Entity, EntityDescription

if TYPE_CHECKING:
    from genesis.engine.scene import Scene


class MPMHybridPart(NamedTuple):
    """The soft part of a hybrid entity, which RigidMPMHybrid drives toward the links of its rigid part."""

    groups_link_idx: np.ndarray
    groups_geom_idx: np.ndarray
    groups_trans_local_to_global: np.ndarray
    groups_quat_local_to_global: np.ndarray
    groups_damping: np.ndarray
    groups_soft_dv_coef: np.ndarray
    particles_idx: np.ndarray
    particles_init_pos: np.ndarray


@dataclass
class HybridMuscleGroupsDescription:
    """Describe the muscle groups of the soft part of a hybrid entity, each following one rigid link.

    Each group stands in its link at the given transform from the local frame of the link to the world frame.
    'particles_group' gives the group of every particle, or None when the soft part keeps its own.
    """

    particles_group: np.ndarray | None
    links_idx: np.ndarray
    geoms_idx: np.ndarray
    trans_local_to_global: np.ndarray
    quat_local_to_global: np.ndarray


@dataclass
class HybridEntityDescription(EntityDescription):
    """Describe one hybrid entity as its resolution left it: its two parts, and how the soft one follows the rigid one.

    The parts are entities of the scene in their own right, added before the hybrid entity and described on their own,
    so they are named here. The functions of the hybrid material that instantiate the parts and associate them run
    once, at resolution, and only what they returned stands here.
    """

    part_rigid_name: str
    part_soft_name: str
    muscle_groups: HybridMuscleGroupsDescription | None = None

    @classmethod
    def resolve(cls, spec: "gs.entities.Hybrid", surface: Surface, add_part: Callable[..., Entity]) -> Self:
        """Add the rigid and the soft parts of an entity spec to the scene, and associate the soft part with the links.

        'surface' is the one resolved from the spec, which the soft part takes. 'add_part' adds the entity spec of a
        part to the scene and returns the entity.
        """
        morph = spec.morph
        material = spec.material
        material_rigid = material.material_rigid
        material_soft = material.material_soft

        surface_rigid = gs.surfaces.Smooth(roughness=0.4)  # HACK hardcoded

        if isinstance(morph, gs.morphs.URDF):
            # The links of a robot description couple with the soft part through the default coupling alone
            material_rigid = material_rigid.model_copy(update={"needs_coup": material.use_default_coupling})
            part_rigid = add_part(spec=gs.entities.Rigid(morph=morph, material=material_rigid, surface=surface_rigid))

            # get rigid part in world coords
            augment_link_world_coords(part_rigid)

            # set soft parts based on rigid links
            func_instantiate_soft_from_rigid = (
                material.func_instantiate_soft_from_rigid or default_func_instantiate_soft_from_rigid
            )
            spec_soft = func_instantiate_soft_from_rigid(
                part_rigid=part_rigid,
                material_soft=material_soft,
                material_hybrid=material,
                surface=surface,
            )
            part_soft = add_part(spec=spec_soft)

        elif isinstance(morph, gs.morphs.Mesh):
            # instantiate soft part
            part_soft = add_part(spec=gs.entities.MPM(morph=morph, material=material_soft, surface=surface))

            # load mesh in the same way as the soft entity
            mesh = load_mesh(morph.file)

            # instantiate rigid part
            func_instantiate_rigid_from_soft = (
                material.func_instantiate_rigid_from_soft or default_func_instantiate_rigid_from_soft
            )
            spec_rigid = func_instantiate_rigid_from_soft(
                mesh=mesh,
                morph=morph,
                material_rigid=material_rigid,
                material_hybrid=material,
                surface=surface_rigid,
            )
            part_rigid = add_part(spec=spec_rigid)

        else:
            gs.raise_exception("`morph` in hybrid entity should be either URDF or Mesh")

        muscle_groups = None
        if not material.use_default_coupling:
            # get rigid-soft association function
            if material.func_instantiate_rigid_soft_association is not None:
                func_instantiate_rigid_soft_association = material.func_instantiate_rigid_soft_association
            elif isinstance(morph, gs.morphs.URDF):
                func_instantiate_rigid_soft_association = default_func_instantiate_rigid_soft_association_from_rigid
            else:
                func_instantiate_rigid_soft_association = default_func_instantiate_rigid_soft_association_from_soft
            muscle_group, links_idx, geoms_idx, trans_local_to_global, quat_local_to_global = (
                func_instantiate_rigid_soft_association(part_rigid=part_rigid, part_soft=part_soft)
            )
            muscle_groups = HybridMuscleGroupsDescription(
                particles_group=None if muscle_group is None else muscle_group.astype(gs.np_int, copy=False),
                links_idx=np.asarray(links_idx, dtype=gs.np_int),
                geoms_idx=np.asarray(geoms_idx, dtype=gs.np_int),
                trans_local_to_global=np.asarray(trans_local_to_global, dtype=gs.np_float),
                quat_local_to_global=np.asarray(quat_local_to_global, dtype=gs.np_float),
            )

        return cls(spec, surface, part_rigid.name, part_soft.name, muscle_groups)


class HybridEntity(Entity):
    """
    A hybrid simulation entity composed of both rigid and soft components.

    This class encapsulates logic for initializing, coupling, and simulating
    a physically hybrid object that has rigid bodies (e.g., from URDF or meshes)
    and soft materials (e.g., MPM). The coupling allows for bi-directional force
    and motion interaction between the rigid and soft parts during simulation.

    Parameters
    ----------
    idx : int
        The index of the entity within the scene.
    scene : genesis.Scene
        The simulation scene where the entity is added.
    desc : HybridEntityDescription
        The description the entity is created from, naming its parts and associating them.
    get_entity : Callable[[str], Entity]
        Returns the entity of the scene of a given name, which gives the two parts the description names.
    """

    _description_cls = HybridEntityDescription

    def __init__(
        self, idx: int, scene: "Scene", desc: HybridEntityDescription, get_entity: Callable[[str], Entity]
    ) -> None:
        super().__init__(idx, scene, None, desc)

        self._part_rigid = get_entity(desc.part_rigid_name)
        self._part_soft = get_entity(desc.part_soft_name)

        # The soft part following the rigid links of its muscle groups, held without the default coupling alone
        self._mpm_hybrid_part: MPMHybridPart | None = None
        muscle_groups = desc.muscle_groups
        if muscle_groups is not None:
            n_groups = len(muscle_groups.links_idx)
            self._mpm_hybrid_part = MPMHybridPart(
                groups_link_idx=muscle_groups.links_idx,
                groups_geom_idx=muscle_groups.geoms_idx,
                groups_trans_local_to_global=muscle_groups.trans_local_to_global,
                groups_quat_local_to_global=muscle_groups.quat_local_to_global,
                groups_damping=np.full(n_groups, self.material.damping),
                groups_soft_dv_coef=np.full(n_groups, self.material.soft_dv_coef),
                particles_idx=np.arange(self._part_soft.particle_start, self._part_soft.particle_end),
                particles_init_pos=self._part_soft.init_particles,
            )

    # ------------------------------------------------------------------------------------
    # --------------------------------- naming methods -----------------------------------
    # ------------------------------------------------------------------------------------

    def _get_morph_identifier(self) -> str:
        morph = self._morph
        if isinstance(morph, gs.morphs.URDF):
            if isinstance(morph.file, str):
                # Try to get robot name from URDF file, fall back to filename stem
                try:
                    return uu.get_robot_name(morph.file)
                except (ValueError, ET.ParseError, FileNotFoundError, OSError) as e:
                    gs.logger.warning(f"Could not extract robot name from URDF: {e}. Using filename stem instead.")
                    return Path(morph.file).stem
            return morph.file.name
        if isinstance(morph, gs.morphs.Mesh):
            return Path(morph.file).stem
        return "hybrid"

    # ------------------------------------------------------------------------------------
    # ----------------------------------- basic ops --------------------------------------
    # ------------------------------------------------------------------------------------

    def get_dofs_position(self, *args, **kwargs):
        """
        Get the current generalized coordinates (positions) of the rigid part of the hybrid entity.

        Parameters
        ----------
        *args, **kwargs
            Passed directly to the rigid entity's get_dofs_position method.

        Returns
        -------
        torch.Tensor
            A tensor containing the position values of the rigid body's degrees of freedom.
        """
        return self._part_rigid.get_dofs_position(*args, **kwargs)

    def get_dofs_velocity(self, *args, **kwargs):
        """
        Get the current generalized velocities of the rigid part of the hybrid entity.

        Parameters
        ----------
        *args, **kwargs
            Passed directly to the rigid entity's get_dofs_velocity method.

        Returns
        -------
        torch.Tensor
            A tensor containing the velocity values of the rigid body's degrees of freedom.
        """
        return self._part_rigid.get_dofs_velocity(*args, **kwargs)

    def get_dofs_force(self, *args, **kwargs):
        """
        Get the current generalized forces applied on the rigid part of the hybrid entity.

        Parameters
        ----------
        *args, **kwargs
            Passed directly to the rigid entity's get_dofs_force method.

        Returns
        -------
        torch.Tensor
            A tensor representing forces acting on the rigid body's degrees of freedom.
        """
        return self._part_rigid.get_dofs_force(*args, **kwargs)

    def get_dofs_control_force(self, *args, **kwargs):
        """
        Get the control forces currently applied to the rigid part of the hybrid entity.

        Parameters
        ----------
        *args, **kwargs
            Passed directly to the rigid entity's get_dofs_control_force method.

        Returns
        -------
        torch.Tensor
            A tensor containing the control force values for the rigid body's degrees of freedom.
        """
        return self._part_rigid.get_dofs_control_force(*args, **kwargs)

    def set_dofs_velocity(self, *args, **kwargs):
        """
        Set the generalized velocities for the rigid part of the hybrid entity.

        Parameters
        ----------
        *args, **kwargs
            Passed directly to the rigid entity's set_dofs_velocity method.
        """
        self._part_rigid.set_dofs_velocity(*args, **kwargs)

    def set_dofs_force(self, *args, **kwargs):
        """
        Set the generalized forces for the rigid part of the hybrid entity.

        Parameters
        ----------
        *args, **kwargs
            Passed directly to the rigid entity's set_dofs_force method.
        """
        self._part_rigid.set_dofs_force(*args, **kwargs)

    def control_dofs_position(self, *args, **kwargs):
        """
        Apply position control to the rigid part of the hybrid entity.

        Parameters
        ----------
        *args, **kwargs
            Passed directly to the rigid entity's control_dofs_position method.

        Returns
        -------
        torch.Tensor
            Control output for position adjustment of the rigid body's DOFs.
        """
        return self._part_rigid.control_dofs_position(*args, **kwargs)

    def control_dofs_position_velocity(self, *args, **kwargs):
        """
        Apply position control to the rigid part of the hybrid entity.

        Parameters
        ----------
        *args, **kwargs
            Passed directly to the rigid entity's control_dofs_position method.

        Returns
        -------
        torch.Tensor
            Control output for position adjustment of the rigid body's DOFs.
        """
        return self._part_rigid.control_dofs_position_velocity(*args, **kwargs)

    def control_dofs_velocity(self, *args, **kwargs):
        """
        Apply velocity control to the rigid part of the hybrid entity.

        Parameters
        ----------
        *args, **kwargs
            Passed directly to the rigid entity's control_dofs_velocity method.

        Returns
        -------
        torch.Tensor
            Control output for velocity adjustment of the rigid body's DOFs.
        """
        self._part_rigid.control_dofs_velocity(*args, **kwargs)

    def control_dofs_force(self, *args, **kwargs):
        """
        Apply force control to the rigid part of the hybrid entity.

        Parameters
        ----------
        *args, **kwargs
            Passed directly to the rigid entity's control_dofs_force method.

        Returns
        -------
        torch.Tensor
            Control output for force adjustment of the rigid body's DOFs.
        """
        self._part_rigid.control_dofs_force(*args, **kwargs)

    # ------------------------------------------------------------------------------------
    # ----------------------------------- instantiation ----------------------------------
    # ------------------------------------------------------------------------------------

    @gs.assert_unbuilt
    def _build(self):
        """
        Finalize the hybrid entity setup during simulation build.
        """
        # can only be called here (at sim build)
        muscle_groups = self._desc.muscle_groups
        if muscle_groups is not None and muscle_groups.particles_group is not None:
            self._part_soft.set_muscle(
                muscle_group=torch.tensor(muscle_groups.particles_group, dtype=gs.tc_int, device=gs.device)
                # no muscle direction as the soft body is actuated by rigid parts
            )

        self._is_built = True

    # ------------------------------------------------------------------------------------
    # ----------------------------------- properties -------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def n_dofs(self) -> int:
        """The number of degrees of freedom of the hybrid entity (inherited from the rigid part)."""
        return self._part_rigid.n_dofs

    @property
    def fixed(self) -> bool:
        """Check whether the hybrid entity is fixed in space (inherited from the rigid morph)."""
        return self._part_rigid.morph.fixed

    @property
    def part_rigid(self):
        """The rigid part of the hybrid entity."""
        return self._part_rigid

    @property
    def mpm_hybrid_part(self) -> MPMHybridPart | None:
        """The soft part that follows the rigid links of its muscle groups, None under default coupling."""
        return self._mpm_hybrid_part

    @property
    def part_soft(self):
        """The soft part of the hybrid entity."""
        return self._part_soft


# ------------------------------------------------------------------------------------
# ------------------------------------- misc -----------------------------------------
# ------------------------------------------------------------------------------------


def augment_link_world_coords(part_rigid):
    ordered_links_idx = [link.idx for link in part_rigid.links]  # now links are pre-ordered
    for i, i_l_global in enumerate(ordered_links_idx):
        i_l = i_l_global - part_rigid.link_start  # NOTE: seems like by default link idx is global
        link = part_rigid.links[i_l]
        i_p = link.parent_idx

        parent_pos = np.zeros((3,))
        parent_quat = gu.identity_quat()

        if i_p != -1:
            link_p = part_rigid.links[i_p - part_rigid.link_start]  # NOTE: seems like by default link idx is global
            parent_pos = link_p.init_x_pos
            parent_quat = link_p.init_x_quat

        if (link.joints and link.joints[0].type == gs.JOINT_TYPE.FREE) or (
            i_p == -1 and (not link.joints or link.joints[0].type == gs.JOINT_TYPE.FIXED)
        ):
            link.init_x_pos = link.desc.pos
            link.init_x_quat = link.desc.quat
        else:
            link.init_x_pos, link.init_x_quat = gu.transform_pos_quat_by_trans_quat(
                link.desc.pos, link.desc.quat, parent_pos, parent_quat
            )
            for joint in link.joints:
                link.init_x_pos, link.init_x_quat = gu.transform_pos_quat_by_trans_quat(
                    joint.pos, joint.quat, link.init_x_pos, link.init_x_quat
                )


# ------------------------------------------------------------------------------------
# ----------------------------- default instantiation --------------------------------
# ------------------------------------------------------------------------------------


def default_func_instantiate_soft_from_rigid(part_rigid, material_soft, material_hybrid, surface):
    meshes = []
    trans_local_to_global = []
    euler_local_to_global = []
    for link in part_rigid.links:
        if len(link.geoms) < 1:  # no collision geom
            continue

        geom = link.geoms[0]  # NOTE: collision geom is always the prior one based on the URDF parser
        trans, quat = gu.transform_pos_quat_by_trans_quat(
            geom.init_pos, geom.init_quat, link.init_x_pos, link.init_x_quat
        )
        euler = gu.quat_to_xyz(quat, rpy=True, degrees=True)

        # can also do link.init_verts here and it seems to have more indices than geom.init_verts (but there is no idx_offset_vert)
        verts = geom.init_verts
        inner_mesh = trimesh.Trimesh(
            vertices=verts,  # NOTE: scale is already applied here
            faces=geom.init_faces,
            vertex_normals=geom.init_normals,
        )
        outer_verts = verts + inner_mesh.vertex_normals * material_hybrid.thickness
        outer_mesh = trimesh.Trimesh(vertices=outer_verts, faces=geom.init_faces)
        # mesh = trimesh.boolean.difference([outer_mesh, inner_mesh]) # wrap around the rigid link
        mesh = outer_mesh  # FIXME: hack to avoid `ValueError: No backends available for boolean operations!`

        meshes.append(mesh)
        trans_local_to_global.append(trans)
        euler_local_to_global.append(euler)

    return gs.entities.MPM(
        morph=gs.morphs.MeshSet(
            files=meshes,
            poss=trans_local_to_global,
            eulers=euler_local_to_global,
            scale=1,  # scale is already handled in geom.init_verts
        ),
        material=material_soft,
        surface=surface,
    )


def default_func_instantiate_rigid_from_soft(
    mesh: trimesh.Trimesh,
    morph: gs.morphs.Mesh,
    material_rigid: Material,
    material_hybrid: gs.materials.Hybrid,
    surface: Surface,
) -> "gs.entities.Rigid":
    # skeletonization
    gelmesh = trimesh_to_gelmesh(mesh)
    graph_gel = skeletonization(gelmesh)

    # convert to nxgraph
    graph_nx = gel_graph_to_nx_graph(graph_gel)

    # compute graph attribute
    graph_pos = graph_gel.positions()
    compute_graph_attribute(graph_nx, graph_pos)

    # reduce nxgraph
    graph_nx_reduced = reduce_graph(graph_nx, straight_thresh=60)

    # to URDF
    G = graph_nx_reduced
    G, src_node = graph_to_tree(G)
    urdf = URDF.from_nxgraph(G)

    # add rigid entity
    mesh_center = (mesh.vertices.max(0) + mesh.vertices.min(0)) / 2.0
    offset = (graph_pos[src_node] - mesh_center) * morph.scale
    pos_rigid = morph.pos + offset.astype(gs.np_float, copy=False)
    quat_rigid = morph.quat
    scale_rigid = morph.scale
    morph_rigid = gs.morphs.URDF(file=urdf, pos=pos_rigid, quat=quat_rigid, scale=scale_rigid, fixed=morph.fixed)
    return gs.entities.Rigid(morph=morph_rigid, material=material_rigid, surface=surface)


def default_func_instantiate_rigid_soft_association_from_rigid(part_rigid, part_soft):
    muscle_group = None  # instantiate soft from rigid already set muscle group using MeshSet

    link_idcs = []
    geom_idcs = []
    trans_local_to_global = []
    quat_local_to_global = []
    for link in part_rigid.links:
        if len(link.geoms) < 1:  # no collision geom
            continue

        geom = link.geoms[0]  # NOTE: collision geom is always the prior one based on the URDF parser
        trans, quat = gu.transform_pos_quat_by_trans_quat(
            geom.init_pos, geom.init_quat, link.init_x_pos, link.init_x_quat
        )

        link_idcs.append(link.idx)
        geom_idcs.append(geom.idx)
        trans_local_to_global.append(trans)
        quat_local_to_global.append(quat)

    return muscle_group, link_idcs, geom_idcs, trans_local_to_global, quat_local_to_global


def default_func_instantiate_rigid_soft_association_from_soft(part_rigid, part_soft):
    # compute distance between particle position and line segment of each link
    augment_link_world_coords(part_rigid)
    positions = part_soft.init_particles
    dist_to_links = []
    link_idcs = []
    geom_idcs = []
    trans_local_to_global = []
    quat_local_to_global = []
    for link in part_rigid.links:
        geom = link.geoms[0]
        link_end_x_pos = (
            gu.transform_by_trans_quat(np.zeros((3,)), trans=geom.init_pos * 2, quat=geom.init_quat) + link.init_x_pos
        )  # TODO: check if this is correct

        p0 = link.init_x_pos
        p1 = link_end_x_pos

        line_vec = p1 - p0  # NOTE: assume a link is a line segment
        line_length = np.linalg.norm(line_vec)

        positions_proj_on_line_t = (positions - p0) @ line_vec / (line_length**2)

        dist_to_p0 = np.linalg.norm(positions - p0, axis=-1)
        dist_to_p1 = np.linalg.norm(positions - p1, axis=-1)
        dist_to_line = np.sqrt(dist_to_p0**2 - ((positions_proj_on_line_t[:, None] * line_vec) ** 2).sum(-1))

        is_clipped_low = positions_proj_on_line_t < 0.0
        is_clipped_high = positions_proj_on_line_t > 1.0
        is_valid = ~(is_clipped_low | is_clipped_high)
        dist_to_link = dist_to_p0 * is_clipped_low + dist_to_p1 * is_clipped_high + dist_to_line * is_valid

        trans, quat = gu.transform_pos_quat_by_trans_quat(
            geom.init_pos, geom.init_quat, link.init_x_pos, link.init_x_quat
        )

        dist_to_links.append(dist_to_link)
        link_idcs.append(link.idx)
        geom_idcs.append(geom.idx)
        trans_local_to_global.append(trans)
        quat_local_to_global.append(quat)

    # get muscle group
    dist_to_links = np.array(dist_to_links)
    muscle_group = dist_to_links.argmin(axis=0)

    return muscle_group, link_idcs, geom_idcs, trans_local_to_global, quat_local_to_global

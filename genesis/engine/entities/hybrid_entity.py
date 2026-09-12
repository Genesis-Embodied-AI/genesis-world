import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import quadrants as qd
import trimesh

import genesis as gs
import genesis.utils.array_class as array_class
import genesis.utils.geom as gu
import genesis.utils.urdf as uu
from genesis.ext.urdfpy.urdf import URDF

from genesis.utils.hybrid import (
    check_graph,
    compute_graph_attribute,
    gel_graph_to_nx_graph,
    graph_to_tree,
    reduce_graph,
    skeletonization,
    trimesh_to_gelmesh,
)
from genesis.utils.mesh import load_mesh

from .base_entity import Entity
from .mpm_entity import MPMEntity


@qd.data_oriented
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
    material : genesis.materials.Hybrid
        The hybrid material that includes both rigid and soft sub-materials,
        and defines coupling behavior.
    morph : genesis.morphs.Morph
        The shape/morphology of the entity. Must be either a `URDF` or `Mesh`.
    surface : genesis.surfaces.Surface
        The surface properties applied to the soft part of the entity.
    """

    def __init__(self, idx, scene, material, morph, surface, name: str | None = None):
        super().__init__(idx, scene, morph, None, material, surface, name=name)

        material_rigid = material.material_rigid
        material_soft = material.material_soft

        surface_rigid = gs.surfaces.Smooth(roughness=0.4)  # HACK hardcoded

        assert isinstance(material_soft, gs.materials.MPM.Base)  # TODO: need FEM and PBD

        if isinstance(morph, gs.morphs.URDF):
            # set up rigid part
            if material.use_default_coupling:
                gs.logger.info("Use default coupling in hybrid. Overwrite `needs_coup` in rigid material to True")
                material_rigid.needs_coup = True
            else:
                gs.logger.info("Use default coupling in hybrid. Overwrite `needs_coup` in rigid material to False")
                material_rigid.needs_coup = False

            part_rigid = scene.add_entity(material=material_rigid, morph=morph, surface=surface_rigid)

            # get rigid part in world coords
            augment_link_world_coords(part_rigid)

            # set soft parts based on rigid links
            func_instantiate_soft_from_rigid = (
                material.func_instantiate_soft_from_rigid or default_func_instantiate_soft_from_rigid
            )
            part_soft = func_instantiate_soft_from_rigid(
                scene=scene,
                part_rigid=part_rigid,
                material_soft=material_soft,
                material_hybrid=material,
                surface=surface,
            )

        elif isinstance(morph, gs.morphs.Mesh):
            # instantiate soft part
            part_soft = scene.add_entity(material=material_soft, morph=morph, surface=surface)

            # load mesh in the same way as the soft entity
            mesh = load_mesh(morph.file)

            # instantiate rigid part
            if material.func_instantiate_rigid_from_soft is None:
                func_instantiate_rigid_from_soft = default_func_instantiate_rigid_from_soft
            else:
                func_instantiate_rigid_from_soft = material.func_instantiate_rigid_from_soft
            part_rigid = func_instantiate_rigid_from_soft(
                scene=scene,
                mesh=mesh,
                morph=morph,
                material_rigid=material_rigid,
                material_hybrid=material,
                surface=surface_rigid,
            )

        else:
            raise ValueError("`morph` in hybrid entity should be either URDF or Mesh")

        if not material.use_default_coupling:
            # get rigid-soft association function
            if material.func_instantiate_rigid_soft_association is None:
                if isinstance(morph, gs.morphs.URDF):
                    func_instantiate_rigid_soft_association = default_func_instantiate_rigid_soft_association_from_rigid
                elif isinstance(morph, gs.morphs.Mesh):
                    func_instantiate_rigid_soft_association = default_func_instantiate_rigid_soft_association_from_soft
            else:
                func_instantiate_rigid_soft_association = material.func_instantiate_rigid_soft_association
            muscle_group, link_idcs, geom_idcs, trans_local_to_global, quat_local_to_global = (
                func_instantiate_rigid_soft_association(part_rigid=part_rigid, part_soft=part_soft)
            )
            if muscle_group is not None:
                muscle_group = muscle_group.astype(gs.np_int, copy=False)

            # set muscle group
            material_soft.n_groups = len(link_idcs)
            self._muscle_group_cache = muscle_group

            # set up info in Quadrants field
            if isinstance(material_soft, gs.materials.MPM.Base):
                part_soft_info = qd.types.struct(
                    link_idx=gs.qd_int,
                    geom_idx=gs.qd_int,
                    trans_local_to_global=gs.qd_vec3,
                    quat_local_to_global=gs.qd_vec4,
                ).field(shape=(material_soft.n_groups,), needs_grad=False, layout=qd.Layout.SOA)
                part_soft_info.link_idx.from_numpy(np.asarray(link_idcs, dtype=gs.np_int))
                part_soft_info.geom_idx.from_numpy(np.asarray(geom_idcs, dtype=gs.np_int))
                part_soft_info.trans_local_to_global.from_numpy(np.asarray(trans_local_to_global, dtype=gs.np_float))
                part_soft_info.quat_local_to_global.from_numpy(np.asarray(quat_local_to_global, dtype=gs.np_float))

                part_soft_init_positions = qd.field(dtype=gs.qd_vec3, shape=(part_soft.init_particles.shape[0],))
                part_soft_init_positions.from_torch(gs.Tensor(part_soft.init_particles))

                self._part_soft_info = part_soft_info
                self._part_soft_init_positions = part_soft_init_positions
            else:
                raise ValueError(f"Cannot handle soft material {material_soft}")

            # set coupling func
            def wrap_func(func, before=False):
                def wrapper(f):
                    if before:
                        self.update_soft_part(f)
                    func(f)
                    if not before:
                        self.update_soft_part(f)

                return wrapper

            if isinstance(material_soft, gs.materials.MPM.Base):
                # NOTE: coupling operating at particle level and here we modify post_coupling, i.e., update particle state after g2p
                self._update_soft_part_at_pre_coupling = False
                if self._update_soft_part_at_pre_coupling:
                    part_soft.solver.substep_pre_coupling = wrap_func(
                        part_soft.solver.substep_pre_coupling, before=True
                    )
                else:
                    part_soft.solver.substep_post_coupling = wrap_func(
                        part_soft.solver.substep_post_coupling, before=False
                    )
            else:
                raise ValueError(f"Cannot handle soft material {material_soft}")

        # set members
        self._material_rigid = material_rigid
        self._part_rigid = part_rigid
        self._solver_rigid = part_rigid.solver
        self._material_soft = material_soft
        self._part_soft = part_soft
        self._solver_soft = part_soft.solver

        # TODO: test with different dt
        assert self._solver_rigid.dt == self._solver_soft.dt, "Rigid and soft solver should have the same dt for now"

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
        gs.Tensor
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
        gs.Tensor
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
        gs.Tensor
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
        gs.Tensor
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
        gs.Tensor
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
        gs.Tensor
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
        gs.Tensor
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
        gs.Tensor
            Control output for force adjustment of the rigid body's DOFs.
        """
        self._part_rigid.control_dofs_force(*args, **kwargs)

    # ------------------------------------------------------------------------------------
    # ----------------------------------- instantiation ----------------------------------
    # ------------------------------------------------------------------------------------

    def build(self):
        """
        Finalize the hybrid entity setup during simulation build.
        """
        # can only be called here (at sim build)
        if not self.material.use_default_coupling and self._muscle_group_cache is not None:
            self._part_soft.set_muscle(
                muscle_group=gs.tensor(self._muscle_group_cache)
                # no muscle direction as the soft body is actuated by rigid parts
            )

    def update_soft_part(self, f):
        """
        Update the state of the soft part of the hybrid entity based on the current simulation frame.

        Parameters
        ----------
        f : int
            The current simulation frame index.
        """
        if isinstance(self._part_soft, MPMEntity):
            self._kernel_update_soft_part_mpm(
                f=f, geoms_info=self._solver_rigid.dyn_info.geoms, links_state=self._solver_rigid.dyn_state.links
            )
        else:
            raise NotImplementedError

    @qd.kernel
    def _kernel_update_soft_part_mpm(
        self, f: qd.i32, geoms_info: array_class.GeomsInfo, links_state: array_class.LinksState
    ):
        for i_p_, i_b in qd.ndrange(self._part_soft.n_particles, self._part_soft._sim._B):
            if self._solver_soft.particles_ng[f, i_p_, i_b].active:
                i_global = i_p_ + self._part_soft.particle_start
                f_ = f
                if qd.static(not self._update_soft_part_at_pre_coupling):
                    f_ = f + 1  # NOTE: this is after g2p and thus we use f + 1

                # get corresponding link
                group_idx = self._solver_soft.particles_info[i_global].muscle_group

                link_idx = self._part_soft_info.link_idx[group_idx]
                geom_idx = self._part_soft_info.geom_idx[group_idx]
                trans_local_to_global = self._part_soft_info.trans_local_to_global[group_idx]
                quat_local_to_global = self._part_soft_info.quat_local_to_global[group_idx]

                g_pos_0 = geoms_info.pos[geom_idx]
                g_quat_0 = geoms_info.quat[geom_idx]

                # compute new pos in minimal coordinate using rigid-bodied dynamics
                x_init_pos = self._part_soft_init_positions[i_p_]
                x_init_local = gu.qd_inv_transform_by_trans_quat(
                    x_init_pos, trans_local_to_global, quat_local_to_global
                )
                scaled_pos = gu.qd_transform_by_quat(gu.qd_inv_transform_by_quat(g_pos_0, g_quat_0), g_quat_0)
                tx_pos, tx_quat = gu.qd_transform_pos_quat_by_trans_quat(
                    scaled_pos, g_quat_0, links_state.pos[link_idx, i_b], links_state.quat[link_idx, i_b]
                )
                new_x_pos = gu.qd_transform_by_trans_quat(x_init_local, trans=tx_pos, quat=tx_quat)

                # compute velocity in closed-loop setting
                dt = self._solver_soft.substep_dt
                dt_scale = (
                    self._solver_soft.substep_dt / self._solver_rigid.dt
                )  # NOTE: move soft part incrementally at soft solver's substeps
                x_pos = self._solver_soft.particles.pos[f_, i_global, i_b]
                xd_vel = (new_x_pos - x_pos) / dt
                xd_vel *= dt_scale  # assume linear scaling between the timestep difference of soft/rigid solver

                vel_d = xd_vel - self._solver_soft.particles.vel[f_, i_global, i_b]
                vel_d *= qd.exp(-self._solver_soft.dt * self.material.damping)

                # soft-to-rigid coupling
                dt_for_rigid_acc = (
                    self._solver_rigid.dt
                )  # NOTE: use rigid dt here as we are sorta doing integration within soft solver substep
                mass_real = self._solver_soft.particles_info[i_global].mass / self._solver_soft._particle_volume_scale
                acc = vel_d / dt_for_rigid_acc
                frc_vel = mass_real * acc
                frc_ang = (x_pos - links_state.root_COM[link_idx, i_b]).cross(frc_vel)
                links_state.cfrc_coupling_vel[link_idx, i_b] += frc_vel
                links_state.cfrc_coupling_ang[link_idx, i_b] += frc_ang

                # rigid-to-soft coupling # NOTE: this may lead to unstable feedback loop
                self._solver_soft.particles.vel[f_, i_global, i_b] += vel_d * self.material.soft_dv_coef

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
    def part_soft(self):
        """The soft part of the hybrid entity."""
        return self._part_soft

    @property
    def solver_rigid(self):
        """The solver associated with the rigid part of the hybrid entity."""
        return self._solver_rigid

    @property
    def solver_soft(self):
        """The solver associated with the soft part of the hybrid entity."""
        return self._solver_soft


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


def _visualize_muscle_group(positions, muscle_group):
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(positions)
    colors = np.zeros((positions.shape[0], 3))
    for gii, group_id in enumerate(np.unique(muscle_group)):
        colors[muscle_group == group_id] = np.random.uniform(0, 1, (3,))
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw([pcd])


# ------------------------------------------------------------------------------------
# ----------------------------- default instantiation --------------------------------
# ------------------------------------------------------------------------------------


def default_func_instantiate_soft_from_rigid(scene, part_rigid, material_soft, material_hybrid, surface):
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
        assert hasattr(geom, "init_normals")
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

    rm_cross_link_overlap_mesh = False
    if rm_cross_link_overlap_mesh:
        for i, mesh in enumerate(meshes[:-1]):  # remove cross-link overlapping area
            meshes[i] = trimesh.boolean.difference([mesh] + meshes[i + 1 :])

    part_soft = scene.add_entity(
        material=material_soft,
        morph=gs.morphs.MeshSet(
            files=meshes,
            poss=trans_local_to_global,
            eulers=euler_local_to_global,
            scale=1,  # scale is already handled in geom.init_verts
        ),
        surface=surface,
    )

    return part_soft


def default_func_instantiate_rigid_from_soft(scene, mesh, morph, material_rigid, material_hybrid, surface):
    # skeletonization
    gelmesh = trimesh_to_gelmesh(mesh)
    graph_gel = skeletonization(gelmesh)

    # convert to nxgraph
    graph_nx = gel_graph_to_nx_graph(graph_gel)
    check_graph(graph_nx)

    # compute graph attribute
    graph_pos = graph_gel.positions()
    compute_graph_attribute(graph_nx, graph_pos)

    # reduce nxgraph
    graph_nx_reduced = reduce_graph(graph_nx, straight_thresh=60)
    check_graph(graph_nx_reduced)

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
    part_rigid = scene.add_entity(material=material_rigid, morph=morph_rigid, surface=surface)

    return part_rigid


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
    for i, link in enumerate(part_rigid.links):
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

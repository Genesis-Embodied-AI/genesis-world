import numpy as np
import pytest

try:
    import uipc
except ImportError:
    pytest.skip("IPC Coupler is not supported because 'uipc' module is not available.", allow_module_level=True)

from uipc.backend import SceneVisitor
from uipc.geometry import SimplicialComplexSlot, apply_transform, merge


def collect_ipc_geometry_entries(scene):
    visitor = SceneVisitor(scene.sim.coupler._ipc_scene)
    for geom_slot in visitor.geometries():
        if not isinstance(geom_slot, SimplicialComplexSlot):
            continue
        geom = geom_slot.geometry()
        meta_attrs = geom.meta()

        solver_type_attr = meta_attrs.find("solver_type")
        if solver_type_attr is None:
            continue
        (solver_type,) = solver_type_attr.view()
        assert solver_type in ("rigid", "fem", "cloth")

        env_idx_attr = meta_attrs.find("env_idx")
        (env_idx,) = map(int, env_idx_attr.view())

        if solver_type == "rigid":
            idx_attr = meta_attrs.find("link_idx")
        else:  # solver_type in ("fem", "cloth")
            idx_attr = meta_attrs.find("entity_idx")
        (idx,) = map(int, idx_attr.view())

        yield (solver_type, env_idx, idx, geom)


def find_ipc_geometries(scene, *, solver_type, idx=None, env_idx=None):
    geoms = []
    for solver_type_, env_idx_, idx_, geom in collect_ipc_geometry_entries(scene):
        if solver_type == solver_type_ and (idx is None or idx == idx_) and (env_idx is None or env_idx == env_idx_):
            geoms.append(geom)
    return geoms


def get_ipc_merged_geometry(scene, *, solver_type, idx, env_idx):
    (geom,) = find_ipc_geometries(scene, solver_type=solver_type, idx=idx, env_idx=env_idx)
    if geom.instances().size() >= 1:
        geom = merge(apply_transform(geom))
    return geom


def get_ipc_positions(scene, *, solver_type, idx, envs_idx):
    geoms_positions = []
    assert envs_idx
    for env_idx in envs_idx:
        merged_geom = get_ipc_merged_geometry(scene, solver_type=solver_type, idx=idx, env_idx=env_idx)
        geom_positions = merged_geom.positions().view().squeeze(axis=-1)
        geoms_positions.append(geom_positions)
    return np.stack(geoms_positions, axis=0)


def get_ipc_rigid_links_idx(scene, env_idx):
    links_idx = []
    for solver_type_, env_idx_, idx_, _geom in collect_ipc_geometry_entries(scene):
        if solver_type_ == "rigid" and env_idx_ == env_idx:
            links_idx.append(idx_)
    return links_idx

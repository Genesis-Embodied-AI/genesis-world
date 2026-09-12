import contextvars
import io
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, NamedTuple

import filelock
import numpy as np
import torch
from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

import genesis as gs
import genesis.utils.mesh as mu
from genesis.constants import USD_FORMATS

from .usd_material import parse_material_preview_surface
from .usd_utils import extract_scale

# Check for Omniverse Kit support (required for USD baking)
# Note: CI workflows should set OMNI_KIT_ACCEPT_EULA=yes in their env section
try:
    import omni.kit_app

    HAS_OMNIVERSE_KIT_SUPPORT = True
except ImportError:
    HAS_OMNIVERSE_KIT_SUPPORT = False


def decompress_usdz(usdz_path: str):
    usdz_folder = mu.get_usd_zip_path(usdz_path)

    # The first file in the package must be a native usd file.
    # See https://openusd.org/docs/Usdz-File-Format-Specification.html
    zip_files = Sdf.ZipFile.Open(usdz_path)
    zip_filelist = zip_files.GetFileNames()
    root_file = zip_filelist[0]
    if not root_file.lower().endswith(USD_FORMATS[:-1]):
        gs.raise_exception(f"Invalid usdz root file: {root_file}")
    root_path = os.path.join(usdz_folder, root_file)

    if not os.path.exists(root_path):
        for file_name in zip_filelist:
            file_data = io.BytesIO(zip_files.GetFile(file_name))
            file_path = os.path.join(usdz_folder, file_name)
            file_folder = os.path.dirname(file_path)
            os.makedirs(file_folder, exist_ok=True)
            with open(file_path, "wb") as out:
                out.write(file_data.read())
        gs.logger.warning(f"USDZ file {usdz_path} decompressed to {root_path}.")
    else:
        gs.logger.info(f"Decompressed assets detected and used: {root_path}.")
    return root_path


class PhysicsMaterial(NamedTuple):
    """Authored properties of a bound UsdPhysicsMaterialAPI material. A None field means the attribute
    is unauthored, letting callers fall back to Genesis defaults."""

    static_friction: float | None = None
    dynamic_friction: float | None = None
    restitution: float | None = None
    density: float | None = None


# The context every USD parse made inside a 'usd_context' block reads from (see 'get_current_usd_context').
_ACTIVE_CONTEXT: contextvars.ContextVar["UsdContext | None"] = contextvars.ContextVar("usd_context", default=None)


@contextmanager
def usd_context(stage_file: str) -> Iterator["UsdContext"]:
    """Open a stage once for every parse made inside the block.

    A stage split into several entities is parsed once per entity, and each parse composes the stage and walks its
    materials anew. Inside this block they all read the one context opened here (see 'get_current_usd_context').
    """
    context = UsdContext(stage_file)
    token = _ACTIVE_CONTEXT.set(context)
    try:
        yield context
    finally:
        _ACTIVE_CONTEXT.reset(token)


def get_current_usd_context() -> "UsdContext | None":
    """The context of the enclosing 'usd_context' block, or None outside of one."""
    return _ACTIVE_CONTEXT.get()


class UsdContext:
    """
    Context manager for USD stage parsing and material processing.

    This class provides a centralized context for parsing USD files, managing materials,
    computing transforms, and handling asset preprocessing. It supports USDZ decompression,
    material baking, coordinate system conversion, and asset symlink resolution.

    Parameters
    ----------
    stage_file : str
        Path to the USD stage file (.usd, .usda, .usdc) or USDZ archive (.usdz).
        If a USDZ file is provided, it will be automatically decompressed.
    usd_bake_cache : bool, optional
        If True, enables material baking and uses last time baked assets if available.
        Otherwise, will re-bake materials every time.
        Default is True.

    Notes
    -----
    - USDZ files are automatically decompressed to a temporary directory
    - The stage's up-axis and meter scale are detected and stored for transform computations
    - Material parsing is lazy (only when find_all_materials() is called)
    """

    def __init__(self, stage_file: str, use_bake_cache: bool = True):
        # decompress usdz
        if stage_file.lower().endswith(USD_FORMATS[-1]):
            stage_file = decompress_usdz(stage_file)

        # detect if baking is needed
        self._need_bake = HAS_OMNIVERSE_KIT_SUPPORT
        if HAS_OMNIVERSE_KIT_SUPPORT:
            if not torch.cuda.is_available():
                gs.logger.warning("USD baking requires CUDA GPU. USD baking will be disabled.")
                self._need_bake = False
        else:
            gs.logger.warning(
                "omniverse-kit not found. USD baking will be disabled. "
                "Please install it with `pip install --extra-index-url https://pypi.nvidia.com omniverse-kit`. "
                "See https://genesis-world.readthedocs.io/en/latest/user_guide/getting_started/usd_import.html."
            )

        # detect bake file caches
        self._bake_folder = mu.get_usd_bake_path(stage_file)
        self._bake_stage_file = os.path.join(self._bake_folder, os.path.basename(stage_file))
        if use_bake_cache:
            if os.path.exists(self._bake_stage_file):
                self._need_bake = False
                gs.logger.info(f"Baked assets detected and used: {self._bake_stage_file}")
                stage_file = self._bake_stage_file
        else:
            if os.path.exists(self._bake_stage_file):
                shutil.rmtree(self._bake_folder)

        self._stage_file = stage_file
        if not os.path.isfile(self._stage_file):
            gs.raise_exception(
                f"USD file not found: {self._stage_file}. Check that the path is correct and the file exists."
            )
        try:
            self._stage = Usd.Stage.Open(self._stage_file)
        except Exception as e:
            gs.raise_exception_from(
                f"Failed to open USD stage: {self._stage_file}. Ensure the file exists and is a valid USD file.", e
            )
        self._material_properties: dict[str, tuple[dict, str]] = {}  # material_id -> (material_dict, uv_name)
        self._physics_material_cache: dict[str, PhysicsMaterial | None] = {}  # prim_path -> physics material
        self._is_restitution_warned = False
        self._is_cross_entity_filtering_warned = False
        self._material_parsed = False
        self._bake_material_paths: dict[str, str] = {}  # material_id -> bake_material_path
        self._prim_material_bindings: dict[str, str] = {}  # prim_path -> material_path
        self._xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        self._is_yup = UsdGeom.GetStageUpAxis(self._stage) == "Y"
        self._meter_scale = UsdGeom.GetStageMetersPerUnit(self._stage)

    @property
    def stage(self) -> Usd.Stage:
        """
        Get the USD stage object.
        """
        return self._stage

    @property
    def stage_file(self) -> str:
        """
        Get the path to the USD stage file.
        """
        return self._stage_file

    def get_prim_id(self, prim: Usd.Prim) -> str:
        """
        Get a unique identifier for a prim based on its layer specification.

        The identifier is constructed from the layer file path and the prim's path
        string. This ensures uniqueness even when the same prim appears in multiple
        layers or when using baked stages.
        """
        prim_stack = prim.GetPrimStack()
        spec = next((s for s in prim_stack if s.specifier == Sdf.SpecifierOver), prim_stack[-1])
        spec_path = self._stage_file if spec.layer.identifier == self._bake_stage_file else spec.layer.identifier
        return spec_path + spec.path.pathString

    def get_binding_material(self, prim: Usd.Prim) -> UsdShade.Material | None:
        """
        Get the material bound to a geometry prim.
        """
        prim_path = str(prim.GetPath())
        if prim_path in self._prim_material_bindings:
            return UsdShade.Material(self._stage.GetPrimAtPath(self._prim_material_bindings[prim_path]))
        return None

    def get_physics_material(self, prim: Usd.Prim) -> PhysicsMaterial | None:
        """
        Resolve and parse the physics material (UsdPhysicsMaterialAPI) bound to a prim.

        USD binds physics materials on the dedicated ``"physics"`` material purpose, which may be
        authored on the prim itself or inherited from an ancestor. Returns the authored properties
        (see ``PhysicsMaterial``), or ``None`` if no physics material is bound.
        """
        prim_path = str(prim.GetPath())
        if prim_path in self._physics_material_cache:
            return self._physics_material_cache[prim_path]

        material_props = None
        material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial("physics")
        material_prim = material.GetPrim() if material else None
        if material_prim is not None and material_prim.IsValid() and material_prim.HasAPI(UsdPhysics.MaterialAPI):
            material_api = UsdPhysics.MaterialAPI(material_prim)
            values = []
            # Ordered as the PhysicsMaterial fields.
            for attr in (
                material_api.GetStaticFrictionAttr(),
                material_api.GetDynamicFrictionAttr(),
                material_api.GetRestitutionAttr(),
                material_api.GetDensityAttr(),
            ):
                value = attr.Get() if attr and attr.HasAuthoredValue() else None
                values.append(float(value) if value is not None else None)
            material_props = PhysicsMaterial(*values)

        self._physics_material_cache[prim_path] = material_props
        return material_props

    def note_unsupported_restitution(self):
        """
        Emit a one-time warning that USD physics-material restitution is not applied.

        The Genesis rigid solver has no rigid-rigid restitution coefficient; contact elasticity
        is governed by ``sol_params`` instead. Restitution authored on a UsdPhysicsMaterialAPI is
        therefore parsed but dropped.
        """
        if not self._is_restitution_warned:
            self._is_restitution_warned = True
            gs.logger.warning(
                "USD physics material 'restitution' is not supported by the rigid solver "
                "(no rigid-rigid restitution coefficient); ignoring it. Tune contact elasticity "
                "via solver 'sol_params' instead."
            )

    def note_unsupported_cross_entity_filtering(self):
        """
        Emit a one-time warning that cross-entity USD collision filtering is not applied.

        ``scene.add_stage`` splits a stage into separate entities, and collision filtering is solved
        per entity. A CollisionGroup / FilteredPairsAPI relationship that spans two prims which end up
        in different entities cannot be expressed as per-entity contype/conaffinity, so those pairs
        keep colliding. Within-entity (self-collision) filtering is unaffected.
        """
        if not self._is_cross_entity_filtering_warned:
            self._is_cross_entity_filtering_warned = True
            gs.logger.warning(
                "USD collision filtering references colliders in different entities (scene.add_stage "
                "splits the stage into separate entities); cross-entity filtering is not applied, so "
                "those geometry pairs will still collide. Within-entity self-collision filtering is applied."
            )

    def compute_transform(self, prim: Usd.Prim) -> np.ndarray:
        """
        Compute the local-to-world transformation matrix for a prim.
        """
        transform = self._xform_cache.GetLocalToWorldTransform(prim)
        # USD composes transforms in double precision, so the matrix is kept as authored and only narrowed to the
        # simulation precision once the pose reaches the entity: rounding it here costs the pose its last digits.
        T_usd = np.asarray(transform)  # translation on the bottom row
        if self._is_yup:
            T_usd @= mu.Y_UP_TRANSFORM
        T_usd[:, :3] *= self._meter_scale
        return T_usd.transpose()

    def compute_gs_transform(self, prim: Usd.Prim, ref_prim: Usd.Prim = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the Genesis transform (pose and scale) for a prim.
        """
        Q, S = extract_scale(self.compute_transform(prim))
        if ref_prim is None:
            return Q, S

        Q_ref, S_ref = self.compute_gs_transform(ref_prim)
        Q_rel = np.linalg.inv(Q_ref) @ Q
        return Q_rel, S

    def apply_surface(self, geom_prim: Usd.Prim, surface: gs.surfaces.Surface):
        """
        Apply material properties from USD to a Genesis surface object.
        """
        geom_path = str(geom_prim.GetPath())
        applied_surface = surface.model_copy()

        if geom_path in self._prim_material_bindings:
            surface_id = self._prim_material_bindings[geom_path]
            surface_dict, uv_name = self._material_properties.get(surface_id, ({}, "st"))
            # accepted keys: color_texture, opacity_texture, roughness_texture, metallic_texture, normal_texture, emissive_texture, ior
            applied_surface.update_texture(**surface_dict)
            if surface_id in self._bake_material_paths:
                bake_success = True if surface_dict else False
                if not bake_success:
                    gs.logger.warning(f"Material for '{geom_path}' could not be loaded. Using default material.")
            else:
                bake_success = None
        else:
            uv_name, surface_id = "st", None
            bake_success = None

        return applied_surface, uv_name, surface_id, bake_success

    def find_all_rigid_entities(self) -> list[Usd.Prim]:
        """
        Find all rigid body entities in the USD stage.
        """
        entity_prims = []
        stage_iter = iter(Usd.PrimRange(self._stage.GetPseudoRoot()))
        for prim in stage_iter:
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                entity_prims.append(prim)
                stage_iter.PruneChildren()
            elif prim.HasAPI(UsdPhysics.RigidBodyAPI) or prim.HasAPI(UsdPhysics.CollisionAPI):
                entity_prims.append(prim)
                stage_iter.PruneChildren()
        return entity_prims

    def find_all_materials(self):
        """
        Parse all materials in the USD stage and optionally bake complex materials.
        """
        if self._material_parsed:
            return

        # parse materials
        bound_prims = []
        for prim in self._stage.Traverse():
            if prim.IsA(UsdGeom.Gprim) or prim.IsA(UsdGeom.Subset):
                bound_prims.append(prim)
        materials = UsdShade.MaterialBindingAPI.ComputeBoundMaterials(bound_prims)[0]
        for bound_prim, material in zip(bound_prims, materials):
            geom_path = str(bound_prim.GetPath())
            material_prim = material.GetPrim()

            if material_prim.IsValid():
                # TODO: material_id is also reserved for group_by_material option.
                material_id = self.get_prim_id(material_prim)
                if material_id not in self._material_properties:
                    material_dict, uv_name = parse_material_preview_surface(material)
                    self._material_properties[material_id] = material_dict, uv_name
                    if self._need_bake and not material_dict:
                        self._bake_material_paths[material_id] = str(material_prim.GetPath())
                self._prim_material_bindings[geom_path] = material_id
            else:
                if bound_prim.IsA(UsdGeom.Gprim):
                    gprim = UsdGeom.Gprim(bound_prim)
                    display_colors = np.asarray(gprim.GetDisplayColorPrimvar().Get() or [], dtype=np.float32)
                    if display_colors.size > 0:
                        material_id = self.get_prim_id(bound_prim)
                        color_texture = gs.textures.ColorTexture(color=tuple(display_colors[0]))
                        self._material_properties[material_id] = {"color_texture": color_texture}, "st"
                        self._prim_material_bindings[geom_path] = material_id
        self._material_parsed = True

        if not self._bake_material_paths:
            return

        device = gs.utils.get_device(gs.cuda)[0] if gs.device.type == "cpu" else gs.device
        self.replace_asset_symlinks()
        os.makedirs(self._bake_folder, exist_ok=True)

        # Note that it is necessary to call 'bake_usd_material' as a subprocess to ensure proper isolation of omniverse
        # kit, otherwise the global conversion registry of some Python bindings will be conflicting with each other,
        # ultimately leading to segfault...
        # The fault handler dumps the Python stack of the child on a native crash, which is otherwise silent
        commands = [
            sys.executable,
            "-X",
            "faulthandler",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "usd_bake.py"),
            "--input_file",
            self._stage_file,
            "--output_dir",
            self._bake_folder,
            "--usd_material_paths",
            *self._bake_material_paths.values(),
            "--device",
            str(device.index if device.index is not None else 0),
            "--log_level",
            logging.getLevelName(gs.logger.level).lower(),
        ]
        gs.logger.debug(f"Execute: {' '.join(commands)}")

        env = dict(os.environ)
        env["OMNI_KIT_ALLOW_ROOT"] = "1"

        # Bootstrapping omni-kit by multiple processes concurrently causes segfaults. Use a file lock to serialize
        # bake operations across parallel processes (e.g. pytest-xdist workers).
        lock_path = os.path.join(tempfile.gettempdir(), "genesis_usd_bake.lock")
        with filelock.FileLock(lock_path, timeout=600):
            # The output is logged before the exit status is examined: Kit's own log and the crash stack live in it,
            # and raising on a bad exit would skip past them.
            try:
                result = subprocess.run(commands, capture_output=True, text=True, env=env)
            except OSError as e:
                gs.logger.warning(f"Baking process could not be started: {e}")
            else:
                if result.stdout:
                    gs.logger.debug(result.stdout)
                if result.stderr:
                    gs.logger.warning(result.stderr)
                if result.returncode != 0:
                    gs.logger.warning(
                        f"Baking process failed with exit status {result.returncode}. A few possible reasons:"
                        "\n\t1. The first launch may require accepting the Omniverse EULA. "
                        "Set `OMNI_KIT_ACCEPT_EULA=yes` to accept it automatically."
                        "\n\t2. The first launch may install additional dependencies, which can cause a timeout."
                        "\n\t3. If you have multiple Python environments (especially with different Python versions), "
                        "Omniverse Kit extensions may conflict across environments. Try to remove the shared omniverse "
                        "extension folder (e.g. `~/.local/share/ov/data/ext` in Linux) and try again."
                    )

        if os.path.exists(self._bake_stage_file):
            gs.logger.warning(f"USD materials baked to file {self._bake_stage_file}")
            self._stage = Usd.Stage.Open(self._bake_stage_file)
            for bake_material_id, bake_material_path in self._bake_material_paths.items():
                bake_material_usd = UsdShade.Material(self._stage.GetPrimAtPath(bake_material_path))
                bake_material_dict, uv_name = parse_material_preview_surface(bake_material_usd)
                self._material_properties[bake_material_id] = bake_material_dict, uv_name
            for baked_texture_obj in Path(self._bake_folder).glob("baked_textures*"):
                shutil.rmtree(baked_texture_obj)

    def replace_asset_symlinks(self):
        """
        Replace asset path symlinks with actual file copies when file extensions differ.

        Some USD assets use symlinks that point to files with different extensions
        (e.g., .png symlink pointing to .exr). This method finds such symlinks and
        replaces them with actual file copies to ensure compatibility.
        """
        asset_paths = set()

        for prim in self._stage.TraverseAll():
            for attr in prim.GetAttributes():
                value = attr.Get()
                if isinstance(value, Sdf.AssetPath):
                    asset_paths.add(value.resolvedPath)
                elif isinstance(value, list):
                    for v in value:
                        if isinstance(v, Sdf.AssetPath):
                            asset_paths.add(v.resolvedPath)

        for asset_path in map(Path, asset_paths):
            if not asset_path.is_symlink():
                continue

            real_path = asset_path.resolve()
            if asset_path.suffix.lower() == real_path.suffix.lower():
                continue

            asset_path.unlink()
            if real_path.is_file():
                gs.logger.warning(f"Replacing symlink {asset_path} with real file {real_path}.")
                shutil.copy2(real_path, asset_path)


def find_joints_in_range(prim_range: Usd.PrimRange) -> list[Usd.Prim]:
    """
    Find all joints in a prim range.

    Parameters
    ----------
    prim_range : Usd.PrimRange
        A prim range to search.

    Returns
    -------
    list[Usd.Prim]
        List of joint prims found in the range.
    """
    joints: list[Usd.Prim] = []
    for prim in prim_range:
        if prim.IsA(UsdPhysics.Joint):
            joints.append(prim)
    return joints


def find_rigid_bodies_in_range(prim_range: Usd.PrimRange) -> set[str]:
    """
    Find all rigid bodies in a prim range.

    When a rigid body is found, its children are pruned from the search since they
    are part of that rigid body and shouldn't be counted separately.

    Parameters
    ----------
    prim_range : Usd.PrimRange
        A prim range to search. Must support PruneChildren().

    Returns
    -------
    set[str]
        Set of rigid body prim paths (as strings).
    """
    rigid_bodies: set[str] = set()
    prim_iter = iter(prim_range)
    for prim in prim_iter:
        if prim.HasAPI(UsdPhysics.RigidBodyAPI) or prim.HasAPI(UsdPhysics.CollisionAPI):
            rigid_bodies.add(str(prim.GetPath()))
            prim_iter.PruneChildren()
    return rigid_bodies


def resolve_rigid_body_link_path(stage: Usd.Stage, path: str) -> str | None:
    """
    Resolve a joint body target path to the canonical rigid-body link path.

    USD assets may place RigidBodyAPI on a parent xform and CollisionAPI on a child
    mesh, with joints referencing either prim. This helper maps both conventions to the
    same link path used by `find_rigid_bodies_in_range`.
    """
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        return None

    if prim.HasAPI(UsdPhysics.RigidBodyAPI):
        return path

    if prim.HasAPI(UsdPhysics.CollisionAPI):
        canonical = path
        parent = prim.GetParent()
        while parent.IsValid():
            if parent.HasAPI(UsdPhysics.RigidBodyAPI):
                return str(parent.GetPath())
            if parent.HasAPI(UsdPhysics.CollisionAPI):
                canonical = str(parent.GetPath())
            parent = parent.GetParent()
        return canonical

    # Wrapper xform without physics API: resolve to the rigid body in its subtree.
    rigid_bodies = find_rigid_bodies_in_range(Usd.PrimRange(prim))
    if len(rigid_bodies) == 1:
        return next(iter(rigid_bodies))
    return None


def extract_links_referenced_by_joints(
    stage: Usd.Stage, joints: list[Usd.Prim], check_rigid_body: bool = True
) -> set[str]:
    """
    Extract links referenced by joints.

    Parameters
    ----------
    stage : Usd.Stage
        The USD stage.
    joints : list[Usd.Prim]
        List of joint prims to analyze.
    check_rigid_body : bool, optional
        If True, only include targets that resolve to a canonical rigid-body link. If False, also
        include the raw target path of unresolvable targets. Default is True.

    Returns
    -------
    set[str]
        Set of link prim paths (as strings) referenced by the joints.
    """
    links_referenced: set[str] = set()
    for joint_prim in joints:
        joint = UsdPhysics.Joint(joint_prim)
        for body_rel in (joint.GetBody0Rel(), joint.GetBody1Rel()):
            targets = body_rel.GetTargets()
            if not targets:
                continue
            body_path = str(targets[0])
            # A resolved path is guaranteed to carry RigidBodyAPI or CollisionAPI.
            resolved_path = resolve_rigid_body_link_path(stage, body_path)
            if resolved_path is not None:
                links_referenced.add(resolved_path)
            elif not check_rigid_body:
                links_referenced.add(body_path)

    return links_referenced

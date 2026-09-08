"""
We define all types of morphologies here: shape primitives, meshes, URDF, MJCF, and soft robot description files.

These are independent of backend solver type and are shared by different solvers, e.g. a mesh can be either loaded as a
rigid object / MPM object / FEM object.
"""

import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Annotated, Any, ClassVar, Literal
from typing_extensions import Self

import numpy as np
from pydantic import Field, StrictBool, StrictInt, model_validator

import genesis as gs
import genesis.ext.urdfpy as urdfpy
import genesis.utils.geom as gu
import genesis.utils.mjcf as mju
import genesis.utils.misc as mu
import genesis.utils.urdf as uu
from genesis.constants import (
    GLTF_FORMATS,
    MJCF_FORMAT,
    URDF_FORMAT,
    USD_FORMATS,
    XACRO_FORMAT,
    XML_ROOT_TAG_TO_FORMAT,
)
from genesis.typing import (
    FGridArrayType,
    FrozenDictType,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    PositiveVec2FType,
    StrArrayType,
    UnitVec3FType,
    UnitVec4FType,
    Vec2IType,
    Vec3FType,
)
from genesis.utils import serialization

from .misc import CoacdOptions
from .options import Options


class TetGenMixin(Options):
    """
    A mixin to introduce TetGen-related options into morph classes that support tetrahedralization using TetGen.
    """

    # FEM specific
    order: PositiveInt = 1

    # Volumetric mesh entity
    mindihedral: NonNegativeInt = 10
    minratio: PositiveFloat = 1.1
    nobisect: StrictBool = True
    quality: StrictBool = True
    maxvolume: float = -1.0
    verbose: Literal[0, 1, 2] = 0

    force_retet: StrictBool = False


@gs.assert_initialized
class Morph(Options):
    """
    This is the base class for all genesis morphs.
    A morph in genesis is a hybrid concept, encapsulating both the geometry and pose information of an entity.
    This includes shape primitives, meshes, URDF, MJCF, Terrain, and soft robot description files.

    Note
    ----
    This class should *not* be instantiated directly.

    Parameters
    ----------
    pos : tuple, shape (3,), optional
        The initial position of the entity in meters at creation time. Defaults to (0.0, 0.0, 0.0).
    euler : tuple, shape (3,), optional
        The initial euler angle of the entity in degrees at creation time. This follows scipy's extrinsic x-y-z
        rotation convention. Defaults to (0.0, 0.0, 0.0).
    quat : tuple, shape (4,), optional
        The initial quaternion (w-x-y-z convention) of the entity at creation time.
        If specified, `euler` will be ignored. Defaults to None.
    offset_pos : tuple, shape (3,), optional
        A fixed pose offset applied in the entity's own body frame on top of the `pos`/`euler` (or `pos`/`quat`) pose.
        It shifts the world pose used internally by the solver but is stripped back out by the relative getters, so
        `get_pos`/`get_quat` (which are relative by default) still report `pos`/`quat`. The morph pose and the offset
        compound exactly like a parent and a child frame: the world pose is
        `transform_pos_quat_by_trans_quat(offset_pos, offset_quat, pos, quat)`, i.e. the offset is expressed in the body
        frame defined by `pos`/`quat`. So `offset_pos` rotates together with the orientation rather than being a
        world-frame shift, and when the orientation is identity it simply adds to `pos`. Defaults to (0.0, 0.0, 0.0).
    offset_euler : tuple, shape (3,), optional
        The orientation offset `offset_quat` given as an euler angle in degrees (scipy extrinsic x-y-z convention).
        Mutually exclusive with `offset_quat`. Defaults to None.
    offset_quat : tuple, shape (4,), optional
        A fixed orientation offset (w-x-y-z convention); see `offset_pos` for how it compounds with `pos`/`quat` to
        form the world pose. Up-axis conversions (e.g. loading a Z-up asset) are stored here.
        Mutually exclusive with `offset_euler`. Defaults to None.
    visualization : bool, optional
        Whether the entity needs to be visualized. Set it to False if you need a invisible object only for collision
        purposes. Defaults to True. `visualization` and `collision` cannot both be False.
        **This is only used for RigidEntity.**
    collision : bool, optional
        Whether the entity needs to be considered for collision checking. Defaults to True.
        `visualization` and `collision` cannot both be False. **This is only used for RigidEntity.**
    requires_jac_and_IK : bool, optional
        This parameter is deprecated.
    is_free : bool, optional
        This parameter is deprecated.
    """

    # Note: pos, quat store only initial values at creation time, and are unaffected by sim
    pos: Vec3FType = (0.0, 0.0, 0.0)
    euler: Vec3FType | None = Field(default=None, exclude=True, repr=False)
    quat: UnitVec4FType | None = None
    offset_pos: Vec3FType = (0.0, 0.0, 0.0)
    offset_euler: Vec3FType | None = Field(default=None, exclude=True, repr=False)
    offset_quat: UnitVec4FType | None = None
    visualization: StrictBool = True
    collision: StrictBool = True
    enable_custom_vverts: StrictBool = False

    @model_validator(mode="before")
    @classmethod
    def _resolve_orientation(cls, data: dict) -> dict:
        is_free = data.pop("is_free", None)
        if is_free is not None:
            gs.logger.warning("'is_free' is deprecated and will be removed in the future.")
        if data.pop("requires_jac_and_IK", None) is not None:
            gs.logger.warning("'requires_jac_and_IK' is deprecated and has no effect.")
        euler = data.get("euler")
        quat = data.get("quat")
        if euler is not None and quat is not None:
            gs.raise_exception("'euler' and 'quat' cannot both be set.")
        if euler is not None:
            data["quat"] = tuple(gu.xyz_to_quat(np.array(euler), rpy=True, degrees=True))
        elif quat is None:
            data["quat"] = (1.0, 0.0, 0.0, 0.0)
        offset_euler = data.get("offset_euler")
        offset_quat = data.get("offset_quat")
        if offset_euler is not None:
            euler_quat = gu.xyz_to_quat(np.array(offset_euler), rpy=True, degrees=True)
            # A quaternion and its negation encode the same rotation, so accept either sign.
            if offset_quat is not None and not (
                np.allclose(offset_quat, euler_quat, atol=gs.EPS) or np.allclose(offset_quat, -euler_quat, atol=gs.EPS)
            ):
                gs.raise_exception("'offset_euler' and 'offset_quat' cannot both be set.")
            data["offset_quat"] = tuple(euler_quat)
        elif offset_quat is None:
            data["offset_quat"] = (1.0, 0.0, 0.0, 0.0)
        return data

    def model_post_init(self, context: Any) -> None:
        if not self.visualization and not self.collision:
            gs.raise_exception("`visualization` and `collision` cannot both be False.")

    def _identifier(self) -> str:
        # Short identifier used for entity naming and brief repr; defaults to the morph type name.
        return type(self).__name__.lower()


############################ Nowhere ############################
class Nowhere(Morph):
    """
    Reserved for emitter. Internal use only.
    """

    n_particles: StrictInt = Field(ge=1)


############################ Shape Primitives ############################


class Primitive(Morph):
    """
    This is the base class for all shape-primitive morphs.

    Note
    ----
    This class should *not* be instantiated directly.

    Parameters
    ----------
    pos : tuple, shape (3,), optional
        The position of the entity in meters. Defaults to (0.0, 0.0, 0.0).
    euler : tuple, shape (3,), optional
        The euler angle of the entity in degrees. This follows scipy's extrinsic x-y-z rotation convention.
        Defaults to (0.0, 0.0, 0.0).
    quat : tuple, shape (4,), optional
        The quaternion (w-x-y-z convention) of the entity. If specified, `euler` will be ignored. Defaults to None.
    visualization : bool, optional
        Whether the entity needs to be visualized. Set it to False if you need a invisible object only for collision
        purposes. Defaults to True. `visualization` and `collision` cannot both be False.
        **This is only used for RigidEntity.**
    collision : bool, optional
        Whether the entity needs to be considered for collision checking. Defaults to True.
        `visualization` and `collision` cannot both be False. **This is only used for RigidEntity.**
    fixed : bool, optional
        Whether the primitive should be fixed. Defaults to False. **This is only used for RigidEntity.**
    batch_fixed_verts : bool, optional
        Whether to batch fixed vertices. This will allow setting env-specific poses to fixed geometries, at the cost of
        significantly increasing memory usage. Default to true. **This is only used for RigidEntity.**
    contype : int, optional
        The 32-bit integer bitmasks used for contact filtering of contact pairs. When the contype of one geom and the
        conaffinity of the other geom share a common bit set to 1, two geoms can collide. Defaults to 0xFFFF.
    conaffinity : int, optional
        The 32-bit integer bitmasks used for contact filtering of contact pairs. When the conaffinity of one geom and
        the contype of the other geom share a common bit set to 1, two geoms can collide. Defaults to 0xFFFF.
    """

    # Rigid specific
    fixed: StrictBool = False
    batch_fixed_verts: StrictBool = True
    contype: StrictInt = Field(default=0xFFFF, ge=0, le=0xFFFFFFFF)
    conaffinity: StrictInt = Field(default=0xFFFF, ge=0, le=0xFFFFFFFF)


class Box(Primitive, TetGenMixin):
    """
    Morph defined by a box shape.

    Note
    ----
    Either [`pos` and `size`] or [`lower` and `upper`] should be specified. The latter has a higher priority.

    Parameters
    ----------
    pos : tuple, shape (3,), optional
        The position of the entity in meters. Defaults to (0.0, 0.0, 0.0).
    euler : tuple, shape (3,), optional
        The euler angle of the entity in degrees. This follows scipy's extrinsic x-y-z rotation convention.
        Defaults to (0.0, 0.0, 0.0).
    quat : tuple, shape (4,), optional
        The quaternion (w-x-y-z convention) of the entity. If specified, `euler` will be ignored. Defaults to None.
    lower : tuple, shape (3,), optional
        The lower corner of the box in meters. Defaults to None.
    upper : tuple, shape (3,), optional
        The upper corner of the box in meters. Defaults to None.
    size : tuple, shape (3,), optional
        The size of the box in meters. Defaults to None.
    visualization : bool, optional
        Whether the entity needs to be visualized. Set it to False if you need a invisible object only for collision
        purposes. Defaults to True. `visualization` and `collision` cannot both be False.
        **This is only used for RigidEntity.**
    collision : bool, optional
        Whether the entity needs to be considered for collision checking. Defaults to True.
        `visualization` and `collision` cannot both be False. **This is only used for RigidEntity.**
    fixed : bool, optional
        Whether the primitive should be fixed. Defaults to False. **This is only used for RigidEntity.**
    batch_fixed_verts : bool, optional
        Whether to batch fixed vertices. This will allow setting env-specific poses to fixed geometries, at the cost of
        significantly increasing memory usage. Default to true. **This is only used for RigidEntity.**
    contype : int, optional
        The 32-bit integer bitmasks used for contact filtering of contact pairs. When the contype of one geom and the
        conaffinity of the other geom share a common bit set to 1, two geoms can collide. Defaults to 0xFFFF.
    conaffinity : int, optional
        The 32-bit integer bitmasks used for contact filtering of contact pairs. When the conaffinity of one geom and
        the contype of the other geom share a common bit set to 1, two geoms can collide. Defaults to 0xFFFF.
    order : int, optional
        The order of the FEM mesh. Defaults to 1. **This is only used for FEMEntity.**
    mindihedral : int, optional
        The minimum dihedral angle in degrees during tetraheralization. Defaults to 10.
        **This is only used for Volumetric Entity that requires tetraheralization.**
    minratio : float, optional
        The minimum tetrahedron quality ratio during tetraheralization. Defaults to 1.1.
        **This is only used for Volumetric Entity that requires tetraheralization.**
    nobisect : bool, optional
        Whether to disable bisection during tetraheralization. Defaults to True.
        **This is only used for Volumetric Entity that requires tetraheralization.**
    quality : bool, optional
        Whether to improve quality during tetraheralization. Defaults to True.
        **This is only used for Volumetric Entity that requires tetraheralization.**
    maxvolume : float, optional
        The maximum tetrahedron volume. Defaults to -1.0 (no limit).
        **This is only used for Volumetric Entity that requires tetraheralization.**
    verbose : int, optional
        The verbosity level during tetraheralization. Defaults to 0.
        **This is only used for Volumetric Entity that requires tetraheralization.**
    force_retet : bool, optional
        Whether to force re-tetraheralization. Defaults to False.
        **This is only used for Volumetric Entity that requires tetraheralization.**
    """

    lower: Vec3FType | None = None
    upper: Vec3FType | None = None
    size: Vec3FType | None = None

    @model_validator(mode="before")
    @classmethod
    def _resolve_geometry(cls, data: dict) -> dict:
        lower, upper, size = data.get("lower"), data.get("upper"), data.get("size")

        if lower is not None and upper is not None:
            lower, upper = np.array(lower), np.array(upper)
            if not (upper >= lower).all():
                gs.raise_exception("Invalid lower and upper corner.")
            data["pos"] = tuple(((lower + upper) / 2).tolist())
            data["size"] = tuple((upper - lower).tolist())

        elif lower is None and upper is None:
            if size is None:
                gs.raise_exception("Either [`pos` and `size`] or [`lower` and `upper`] should be specified.")
            pos, size = np.array(data.get("pos", (0.0, 0.0, 0.0))), np.array(size)
            data["lower"] = tuple((pos - 0.5 * size).tolist())
            data["upper"] = tuple((pos + 0.5 * size).tolist())

        else:
            gs.raise_exception("`lower` and `upper` must be jointly specified.")

        return data


class Cylinder(Primitive, TetGenMixin):
    """
    Morph defined by a cylinder shape.

    Parameters
    ----------
    pos : tuple, shape (3,), optional
        The position of the entity in meters. Defaults to (0.0, 0.0, 0.0).
    euler : tuple, shape (3,), optional
        The euler angle of the entity in degrees. This follows scipy's extrinsic x-y-z rotation convention.
        Defaults to (0.0, 0.0, 0.0).
    quat : tuple, shape (4,), optional
        The quaternion (w-x-y-z convention) of the entity. If specified, `euler` will be ignored. Defaults to None.
    height : float, optional
        The height of the cylinder in meters. Defaults to 1.0.
    radius : float, optional
        The radius of the cylinder in meters. Defaults to 0.5.
    visualization : bool, optional
        Whether the entity needs to be visualized. Set it to False if you need a invisible object only for collision
        purposes. Defaults to True. `visualization` and `collision` cannot both be False.
        **This is only used for RigidEntity.**
    collision : bool, optional
        Whether the entity needs to be considered for collision checking. Defaults to True.
        `visualization` and `collision` cannot both be False. **This is only used for RigidEntity.**
    fixed : bool, optional
        Whether the primitive should be fixed. Defaults to False. **This is only used for RigidEntity.**
    batch_fixed_verts : bool, optional
        Whether to batch fixed vertices. This will allow setting env-specific poses to fixed geometries, at the cost of
        significantly increasing memory usage. Default to true. **This is only used for RigidEntity.**
    contype : int, optional
        The 32-bit integer bitmasks used for contact filtering of contact pairs. When the contype of one geom and the
        conaffinity of the other geom share a common bit set to 1, two geoms can collide. Defaults to 0xFFFF.
    conaffinity : int, optional
        The 32-bit integer bitmasks used for contact filtering of contact pairs. When the conaffinity of one geom and
        the contype of the other geom share a common bit set to 1, two geoms can collide. Defaults to 0xFFFF.
    order : int, optional
        The order of the FEM mesh. Defaults to 1. **This is only used for FEMEntity.**
    mindihedral : int, optional
        The minimum dihedral angle in degrees during tetraheralization. Defaults to 10.
        **This is only used for Volumetric Entity that requires tetraheralization.**
    minratio : float, optional
        The minimum tetrahedron quality ratio during tetraheralization. Defaults to 1.1.
        **This is only used for Volumetric Entity that requires tetraheralization.**
    nobisect : bool, optional
        Whether to disable bisection during tetraheralization. Defaults to True.
        **This is only used for Volumetric Entity that requires tetraheralization.**
    quality : bool, optional
        Whether to improve quality during tetraheralization. Defaults to True.
        **This is only used for Volumetric Entity that requires tetraheralization.**
    maxvolume : float, optional
        The maximum tetrahedron volume. Defaults to -1.0 (no limit).
        **This is only used for Volumetric Entity that requires tetraheralization.**
    verbose : int, optional
        The verbosity level during tetraheralization. Defaults to 0.
        **This is only used for Volumetric Entity that requires tetraheralization.**
    force_retet : bool, optional
        Whether to force re-tetraheralization. Defaults to False.
        **This is only used for Volumetric Entity that requires tetraheralization.**
    """

    height: PositiveFloat = 1.0
    radius: PositiveFloat = 0.5


class Sphere(Primitive, TetGenMixin):
    """
    Morph defined by a sphere shape.

    Parameters
    ----------
    pos : tuple, shape (3,), optional
        The position of the entity in meters. Defaults to (0.0, 0.0, 0.0).
    euler : tuple, shape (3,), optional
        The euler angle of the entity in degrees. This follows scipy's extrinsic x-y-z rotation convention.
        Defaults to (0.0, 0.0, 0.0).
    quat : tuple, shape (4,), optional
        The quaternion (w-x-y-z convention) of the entity. If specified, `euler` will be ignored. Defaults to None.
    radius : float, optional
        The radius of the sphere in meters. Defaults to 0.5.
    visualization : bool, optional
        Whether the entity needs to be visualized. Set it to False if you need a invisible object only for collision
        purposes. Defaults to True. `visualization` and `collision` cannot both be False.
        **This is only used for RigidEntity.**
    collision : bool, optional
        Whether the entity needs to be considered for collision checking. Defaults to True.
        `visualization` and `collision` cannot both be False. **This is only used for RigidEntity.**
    fixed : bool, optional
        Whether the primitive should be fixed. Defaults to False. **This is only used for RigidEntity.**
    batch_fixed_verts : bool, optional
        Whether to batch fixed vertices. This will allow setting env-specific poses to fixed geometries, at the cost of
        significantly increasing memory usage. Default to true. **This is only used for RigidEntity.**
    contype : int, optional
        The 32-bit integer bitmasks used for contact filtering of contact pairs. When the contype of one geom and the
        conaffinity of the other geom share a common bit set to 1, two geoms can collide. Defaults to 0xFFFF.
    conaffinity : int, optional
        The 32-bit integer bitmasks used for contact filtering of contact pairs. When the conaffinity of one geom and
        the contype of the other geom share a common bit set to 1, two geoms can collide. Defaults to 0xFFFF.
    order : int, optional
        The order of the FEM mesh. Defaults to 1. **This is only used for FEMEntity.**
    mindihedral : int, optional
        The minimum dihedral angle in degrees during tetraheralization. Defaults to 10.
        **This is only used for Volumetric Entity that requires tetraheralization.**
    minratio : float, optional
        The minimum tetrahedron quality ratio during tetraheralization. Defaults to 1.1.
        **This is only used for Volumetric Entity that requires tetraheralization.**
    nobisect : bool, optional
        Whether to disable bisection during tetraheralization. Defaults to True.
        **This is only used for Volumetric Entity that requires tetraheralization.**
    quality : bool, optional
        Whether to improve quality during tetraheralization. Defaults to True.
        **This is only used for Volumetric Entity that requires tetraheralization.**
    maxvolume : float, optional
        The maximum tetrahedron volume. Defaults to -1.0 (no limit).
        **This is only used for Volumetric Entity that requires tetraheralization.**
    verbose : int, optional
        The verbosity level during tetraheralization. Defaults to 0.
        **This is only used for Volumetric Entity that requires tetraheralization.**
    force_retet : bool, optional
        Whether to force re-tetraheralization. Defaults to False.
        **This is only used for Volumetric Entity that requires tetraheralization.**
    """

    radius: PositiveFloat = 0.5


class Plane(Primitive):
    """
    Morph defined by a plane shape.

    Note
    ----
    Plane is a primitive with infinite size. Note that the `pos` is the center of the plane,
    but essentially only defines a point where the plane passes through.

    Parameters
    ----------
    pos : tuple, shape (3,), optional
        The center position of the plane in meters. Defaults to (0.0, 0.0, 0.0).
    euler : tuple, shape (3,), optional
        The euler angle of the entity in degrees. This follows scipy's extrinsic x-y-z rotation convention.
        Defaults to (0.0, 0.0, 0.0).
    quat : tuple, shape (4,), optional
        The quaternion (w-x-y-z convention) of the entity. If specified, `euler` will be ignored. Defaults to None.
    normal : tuple, shape (3,), optional
        The normal normal of the plane in its local frame. Defaults to (0, 0, 1).
    visualization : bool, optional
        Whether the entity needs to be visualized. Set it to False if you need a invisible object only for collision
        purposes. Defaults to True. `visualization` and `collision` cannot both be False.
        **This is only used for RigidEntity.**
    collision : bool, optional
        Whether the entity needs to be considered for collision checking. Defaults to True.
        `visualization` and `collision` cannot both be False. **This is only used for RigidEntity.**
    fixed : bool, optional
        Whether the plane is fixed in world. The mass of a plane being ill-defined, this parameter is kept only for
        consistency but must be True, otherwise it will raise an exception.
    batch_fixed_verts : bool, optional
        Whether to batch fixed vertices. This will allow setting env-specific poses to fixed geometries, at the cost of
        significantly increasing memory usage. Default to false. **This is only used for RigidEntity.**
    contype : int, optional
        The 32-bit integer bitmasks used for contact filtering of contact pairs. When the contype of one geom and the
        conaffinity of the other geom share a common bit set to 1, two geoms can collide. Defaults to 0xFFFF.
    conaffinity : int, optional
        The 32-bit integer bitmasks used for contact filtering of contact pairs. When the conaffinity of one geom and
        the contype of the other geom share a common bit set to 1, two geoms can collide. Defaults to 0xFFFF.
    plane_size: tuple, optional
        The size of the plane in meters. Defaults to (1e3, 1e3).
    tile_size: tuple, optional
        The size of each texture tile. Defaults to (1, 1).
    """

    batch_fixed_verts: StrictBool = False
    normal: UnitVec3FType = (0.0, 0.0, 1.0)
    plane_size: PositiveVec2FType = (1e3, 1e3)
    tile_size: PositiveVec2FType = (1.0, 1.0)

    def __init__(self, *, fixed: bool = True, **data):
        if not fixed:
            gs.raise_exception("Plane `fixed` must be True.")
        super().__init__(fixed=True, **data)

        if self.enable_custom_vverts:
            gs.raise_exception("`enable_custom_vverts` must be False for `Plane`.")


############################ Mesh ############################


class FileMorph(Morph):
    """
    Morph loaded from a file.

    Parameters
    ----------
    file : str
        The path to the file.
    scale : float or tuple, optional
        The scaling factor for the size of the entity. If a float, it scales uniformly.
        If a 3-tuple, it scales along each axis. Defaults to 1.0.
        Note that 3-tuple scaling is only supported for `gs.morphs.Mesh`.
    pos : tuple, shape (3,), optional
        The position of the entity in meters. Defaults to (0.0, 0.0, 0.0).
    euler : tuple, shape (3,), optional
        The euler angle of the entity in degrees. This follows scipy's extrinsic x-y-z rotation convention.
        Defaults to (0.0, 0.0, 0.0).
    quat : tuple, shape (4,), optional
        The quaternion (w-x-y-z convention) of the entity. If specified, `euler` will be ignored. Defaults to None.
    decimate : bool, optional
        Whether to decimate (simplify) the collision mesh. Defaults to True when convexify is True and False otherwise,
        since decimation removes the surface detail a non-convex collision mesh is kept for. **This is only used for
        RigidEntity.**
    decimate_face_num : int, optional
        The number of faces to decimate to. Defaults to 500. **This is only used for RigidEntity.**
    decimate_aggressiveness : int
        How hard the decimation process will try to match the target number of faces, as a integer ranging from 0 to 8.
        0 is losseless. 2 preserves all features of the original geometry. 5 may significantly alters the original
        geometry if necessary. 8 does what needs to be done at all costs. Defaults to 2.
        **This is only used for RigidEntity.**
    watertighten : int, optional
        Aggressiveness of the watertight wrap built for a non-convex (``convexify=False``) collision mesh, as an
        integer from 0 to 8 on the same scale as ``decimate_aggressiveness``. The wrap closes an open or
        self-intersecting mesh into the single watertight surface a grid signed distance field requires, decimating
        it under a feature-preserving cost cutoff. 0 bypasses the wrap (mesh kept as-is), higher values collapse more
        of it, and 8 is the strongest decimation the cutoff still allows (every level stays watertight and preserves
        the closed shape rather than collapsing thin cross-sections). ``None`` skips watertightening altogether.
        Defaults to 5. **This is only used for RigidEntity.**
    convexify : bool, optional
        Whether to convexify the entity. When convexify is True, all the meshes in the entity will each be converted
        to a set of convex hulls. The mesh will be decomposed into multiple convex components if the convex hull is not
        sufficient to met the desired accuracy (see 'decompose_(robot|object)_error_threshold' documentation). The
        module 'coacd' is used for this decomposition process. If not given, it defaults to `True` for `RigidEntity`
        and `False` for other deformable entities.
    decompose_nonconvex : bool, optional
        This parameter is deprecated. Please use 'convexify' and 'decompose_(robot|object)_error_threshold' instead.
    decompose_object_error_threshold : float, optional:
        For basic rigid objects (mug, table...), skip convex decomposition if the relative difference between the
        volume of original mesh and its convex hull is lower than this threshold.
        0.0 to enforce decomposition, float("inf") to disable it completely. Defaults to 0.15 (15%).
    decompose_robot_error_threshold : float, optional:
        For poly-articulated robots, skip convex decomposition if the relative difference between the volume of
        original mesh and its convex hull is lower than this threshold.
        0.0 to enforce decomposition, float("inf") to disable it completely. Defaults to float("inf").
    coacd_options : CoacdOptions, optional
        Options for configuring coacd convex decomposition. Needs to be a `gs.options.CoacdOptions` object.
    recompute_inertia : bool, optional
        Force recomputing spatial inertia of links from their geometry. This option is useful to import partially
        broken assets from external providers that cannot be re-exported from source. Default to False.
    align : bool, optional
        Whether to reframe root links so that the link origin coincides with the center of mass and its axes are
        aligned with the principal axes of inertia. This makes the inertia tensor diagonal, which improves numerical
        stability. The frame is then the solver's to keep, so the center of mass and the inertia of such a link cannot
        be written once the scene is built; set it to False to keep the authored frame and write them at runtime. Only
        applies to root (floating-base) links. Uses file-specified inertia if valid (and ``recompute_inertia=False``),
        otherwise computes from geometry. Defaults to None, which resolves to True for basic rigid objects (entities
        with only a root free joint and no articulated descendants), False otherwise.
        **This is only used for RigidEntity.**
    file_meshes_are_zup : bool, optional
        Defines if the mesh files are expressed in a Z-up or Y-up coordinate system. If set to true, meshes are loaded
        as Z-up and no transforms are applied to the input data. If set to false, all meshes undergo a conversion step
        where the original coordinates are transformed as follows: (X, Y, Z) → (X, -Z, Y). Defaults to True.
    visualization : bool, optional
        Whether the entity needs to be visualized. Set it to False if you need a invisible object only for collision
        purposes. Defaults to True. `visualization` and `collision` cannot both be False.
        **This is only used for RigidEntity.**
    collision : bool, optional
        Whether the entity needs to be considered for collision checking. Defaults to True.
        `visualization` and `collision` cannot both be False. **This is only used for RigidEntity.**
    batch_fixed_verts : bool, optional
        Whether to batch fixed vertices. This will allow setting env-specific poses to fixed geometries, at the cost of
        significantly increasing memory usage. Default to true. **This is only used for RigidEntity.**
    """

    # Shown in the repr header via __repr_name__ (a bounded identifier for in-memory descriptions), so it is kept out
    # of the field listing to avoid both duplicating it and dumping a whole inline document.
    file: Any = Field(default="", repr=False)
    scale: Annotated[tuple[PositiveFloat, PositiveFloat, PositiveFloat], Field(strict=False)] | PositiveFloat = 1.0
    decimate: StrictBool | None = None
    decimate_face_num: PositiveInt = 500
    decimate_aggressiveness: StrictInt = Field(default=2, ge=0, le=8)
    watertighten: StrictInt | None = Field(default=5, ge=0, le=8)
    convexify: StrictBool | None = None
    decompose_object_error_threshold: float = Field(default=0.15, ge=0, allow_inf_nan=True)
    decompose_robot_error_threshold: float = Field(default=float("inf"), ge=0, allow_inf_nan=True)
    coacd_options: CoacdOptions | None = None
    recompute_inertia: StrictBool = False
    align: StrictBool | None = None
    file_meshes_are_zup: StrictBool | None = True
    batch_fixed_verts: StrictBool = False

    @model_validator(mode="before")
    @classmethod
    def _resolve_file_and_defaults(cls, data: dict) -> dict:
        # Clamp thresholds to avoid decomposition of convex and primitive shapes
        obj_thresh = data.get("decompose_object_error_threshold", 0.15)
        robot_thresh = data.get("decompose_robot_error_threshold", float("inf"))
        data["decompose_object_error_threshold"] = max(obj_thresh, gs.EPS)
        data["decompose_robot_error_threshold"] = max(robot_thresh, gs.EPS)

        if data.get("coacd_options") is None:
            data["coacd_options"] = CoacdOptions()

        file = data.get("file", "")
        if isinstance(file, os.PathLike):
            file = data["file"] = str(file)
        if isinstance(file, str) and file:
            # Inline XML content (a description built in-memory) parses directly and is passed through untouched to the
            # loader. A path string does not parse as XML and is resolved against the working and assets directories.
            try:
                ET.fromstring(file)
            except ET.ParseError:
                abs_file = os.path.abspath(file)
                if not os.path.exists(abs_file):
                    abs_file = os.path.join(gs.utils.get_assets_dir(), file)
                if not os.path.exists(abs_file):
                    gs.raise_exception(f"File not found in either current directory or assets directory: '{file}'.")
                data["file"] = abs_file

        return data

    def __init__(
        self,
        *,
        decompose_nonconvex: bool | None = None,
        parse_glb_with_zup: bool | None = None,
        **kwargs,
    ):
        if decompose_nonconvex is not None:
            gs.logger.warning(
                "'decompose_nonconvex' is deprecated. Use 'convexify' and "
                "'decompose_(robot|object)_error_threshold' instead."
            )
            if decompose_nonconvex:
                kwargs.setdefault("convexify", True)
                kwargs["decompose_object_error_threshold"] = 0.0
                kwargs["decompose_robot_error_threshold"] = 0.0
            else:
                kwargs["decompose_object_error_threshold"] = float("inf")
                kwargs["decompose_robot_error_threshold"] = float("inf")

        if parse_glb_with_zup is not None:
            gs.logger.warning("'parse_glb_with_zup' is deprecated. Use 'file_meshes_are_zup' instead.")
            kwargs.setdefault("file_meshes_are_zup", not parse_glb_with_zup)

        super().__init__(**kwargs)

        scale = np.atleast_1d(np.array(self.scale))
        if scale.ndim > 1 or scale.size not in (1, 3):
            gs.raise_exception("`scale` should be a scalar sequence of length 1 or 3.")

    def _identifier(self) -> str:
        file = self.file
        if not isinstance(file, str):
            return file.name
        if os.path.exists(file):
            return Path(file).stem
        # An in-memory description has no filename to fall back on; subclasses that embed a name (MJCF model,
        # URDF robot) override this, otherwise the morph type name stands in for the document.
        return super()._identifier()

    def __repr_name__(self):
        # A real file path is shown verbatim; an MJCF/URDF built in memory has no path on disk, so a bounded
        # identifier stands in for the document rather than dumping it.
        file = self.file
        if isinstance(file, str) and not os.path.exists(file):
            file = f"<inline {self._identifier()}>"
        return f"{super().__repr_name__()[:-1]}(file='{file}')>"

    def is_format(self, format):
        if not isinstance(self.file, (str, os.PathLike)):
            return False
        # Inline XML content is identified by its root tag rather than a file extension.
        try:
            root_tag = ET.fromstring(self.file).tag
        except (ET.ParseError, TypeError):
            return str(self.file).lower().endswith(format)
        return XML_ROOT_TAG_TO_FORMAT.get(root_tag) == format


class Mesh(FileMorph, TetGenMixin):
    """
    Morph loaded from a mesh file.

    Note
    ----
    In order to speed up simulation, the loaded mesh will first be decimated (simplified) to a target number of faces,
    followed by convexification (for collision mesh only).
    Such process can be disabled by setting `decimate` and `convexify` to False.

    Parameters
    ----------
    file : str
        The path to the file.
    scale : float or tuple, optional
        The scaling factor for the size of the entity. If a float, it scales uniformly.
        If a 3-tuple, it scales along each axis. Defaults to 1.0.
        Note that 3-tuple scaling is only supported for `gs.morphs.Mesh`.
    pos : tuple, shape (3,), optional
        The position of the entity in meters. Defaults to (0.0, 0.0, 0.0).
    euler : tuple, shape (3,), optional
        The euler angle of the entity in degrees. This follows scipy's extrinsic x-y-z rotation convention.
        Defaults to (0.0, 0.0, 0.0).
    quat : tuple, shape (4,), optional
        The quaternion (w-x-y-z convention) of the entity. If specified, `euler` will be ignored. Defaults to None.
    decimate : bool, optional
        Whether to decimate (simplify) the collision mesh. Defaults to True when convexify is True and False otherwise,
        since decimation removes the surface detail a non-convex collision mesh is kept for. **This is only used for
        RigidEntity.**
    decimate_face_num : int, optional
        The number of faces to decimate to. Defaults to 500. **This is only used for RigidEntity.**
    decimate_aggressiveness : int
        How hard the decimation process will try to match the target number of faces, as a integer ranging from 0 to 8.
        0 is losseless. 2 preserves all features of the original geometry. 5 may significantly alters the original
        geometry if necessary. 8 does what needs to be done at all costs. Defaults to 5.
        **This is only used for RigidEntity.**
    watertighten : int, optional
        Aggressiveness of the watertight wrap built for a non-convex (``convexify=False``) collision mesh, as an
        integer from 0 to 8 on the same scale as ``decimate_aggressiveness``. The wrap closes an open or
        self-intersecting mesh into the single watertight surface a grid signed distance field requires, decimating
        it under a feature-preserving cost cutoff. 0 bypasses the wrap (mesh kept as-is), higher values collapse more
        of it, and 8 is the strongest decimation the cutoff still allows (every level stays watertight and preserves
        the closed shape rather than collapsing thin cross-sections). ``None`` skips watertightening altogether.
        Defaults to 5. **This is only used for RigidEntity.**
    convexify : bool, optional
        Whether to convexify the entity. When convexify is True, all the meshes in the entity will each be converted
        to a set of convex hulls. The mesh with be decomposed into multiple convex components if a single one is not
        sufficient to met the desired accuracy (see 'decompose_(robot|object)_error_threshold' documentation). The
        module 'coacd' is used for this decomposition process. If not given, it defaults to `True` for `RigidEntity`
        and `False` for other deformable entities.
    decompose_nonconvex : bool, optional
        This parameter is deprecated. Please use 'convexify' and 'decompose_(robot|object)_error_threshold' instead.
    decompose_object_error_threshold : float, optional:
        For basic rigid objects (mug, table...), skip convex decomposition if the relative difference between the
        volume of original mesh and its convex hull is lower than this threshold.
        0.0 to enforce decomposition, float("inf") to disable it completely. Defaults to 0.15 (15%).
    decompose_robot_error_threshold : float, optional:
        For poly-articulated robots, skip convex decomposition if the relative difference between the volume of
        original mesh and its convex hull is lower than this threshold.
        0.0 to enforce decomposition, float("inf") to disable it completely. Defaults to float("inf").
    coacd_options : CoacdOptions, optional
        Options for configuring coacd convex decomposition. Needs to be a `gs.options.CoacdOptions` object.
    recompute_inertia : bool, optional
        Force recomputing spatial inertia of links from their geometry. This option is useful to import partially
        broken assets from external providers that cannot be re-exported from source. Default to False.
    merge_submeshes_for_collision : bool, optional
        Whether to merge submeshes for collision. Defaults to False. **This is only used for RigidEntity.**
    visualization : bool, optional
        Whether the entity needs to be visualized. Set it to False if you need a invisible object only for collision
        purposes. Defaults to True. `visualization` and `collision` cannot both be False.
        **This is only used for RigidEntity.**
    collision : bool, optional
        Whether the entity needs to be considered for collision checking. Defaults to True.
        `visualization` and `collision` cannot both be False. **This is only used for RigidEntity.**
    parse_glb_with_zup : bool, optional
        This parameter is deprecated, see file_meshes_are_zup.
    file_meshes_are_zup : bool, optional
        Defines if the mesh files are expressed in a Z-up or Y-up coordinate system. If set to true, meshes are loaded
        as Z-up and no transforms are applied to the input data. If set to false, all meshes undergo a conversion step
        where the original coordinates are transformed as follows: (X, Y, Z) → (X, -Z, Y). If None, then it will default
        to True for all mesh formats except GLTF/GLB, as they are defined as Y-up by the standard. Beware that setting
        this option to True for GLTF/GLB is not supported and will rather apply a rotation on the morph. Default to
        None.
    fixed : bool, optional
        Whether the object should be fixed. Defaults to False. **This is only used for RigidEntity.**
    batch_fixed_verts : bool, optional
        Whether to batch fixed vertices. This will allow setting env-specific poses to fixed geometries, at the cost of
        significantly increasing memory usage. Default to false. **This is only used for RigidEntity.**
    contype : int, optional
        The 32-bit integer bitmasks used for contact filtering of contact pairs. When the contype of one geom and the
        conaffinity of the other geom share a common bit set to 1, two geoms can collide. Defaults to 0xFFFF.
    conaffinity : int, optional
        The 32-bit integer bitmasks used for contact filtering of contact pairs. When the conaffinity of one geom and
        the contype of the other geom share a common bit set to 1, two geoms can collide. Defaults to 0xFFFF.
    group_by_material : bool, optional
        Whether to group submeshes by their visual material type defined in the asset file. Defaults to False.
        **This is only used for RigidEntity.**
    align : bool, optional
        Whether to reframe the mesh so that its link origin coincides with the center of mass and its axes are
        aligned with the principal axes of inertia. This makes the inertia tensor diagonal, which improves
        numerical stability. Defaults to True. **This is only used for RigidEntity.**
    order : int, optional
        The order of the FEM mesh. Defaults to 1. **This is only used for FEMEntity.**
    mindihedral : int, optional
        The minimum dihedral angle in degrees during tetraheralization. Defaults to 10.
        **This is only used for Volumetric Entity that requires tetraheralization.**
    minratio : float, optional
        The minimum tetrahedron quality ratio during tetraheralization. Defaults to 1.1.
        **This is only used for Volumetric Entity that requires tetraheralization.**
    nobisect : bool, optional
        Whether to disable bisection during tetraheralization. Defaults to True.
        **This is only used for Volumetric Entity that requires tetraheralization.**
    quality : bool, optional
        Whether to improve quality during tetraheralization. Defaults to True.
        **This is only used for Volumetric Entity that requires tetraheralization.**
    maxvolume : float, optional
        The maximum tetrahedron volume. Defaults to -1.0 (no limit).
        **This is only used for Volumetric Entity that requires tetraheralization.**
    verbose : int, optional
        The verbosity level during tetraheralization. Defaults to 0.
        **This is only used for Volumetric Entity that requires tetraheralization.**
    force_retet : bool, optional
        Whether to force re-tetraheralization. Defaults to False.
        **This is only used for Volumetric Entity that requires tetraheralization.**
    """

    # Rigid specific
    file_meshes_are_zup: StrictBool | None = None
    fixed: StrictBool = False
    contype: StrictInt = Field(default=0xFFFF, ge=0, le=0xFFFFFFFF)
    conaffinity: StrictInt = Field(default=0xFFFF, ge=0, le=0xFFFFFFFF)
    group_by_material: StrictBool = False
    merge_submeshes_for_collision: StrictBool = False

    @model_validator(mode="after")
    def _resolve_zup(self) -> Self:
        file = self.file
        is_gltf = isinstance(file, str) and str(file).lower().endswith(GLTF_FORMATS)

        if is_gltf:
            if self.file_meshes_are_zup:
                # GLTF/GLB is Y-up by standard, so a Z-up claim is honored by recording the compensating rotation in
                # 'offset_quat'. It is post-multiplied on the user orientation to form the world pose, while 'quat'
                # stays clean and is what relative getters report.
                gs.logger.info(
                    "Honoring 'file_meshes_are_zup' for a GLTF/GLB file by recording a compensating rotation in "
                    "'offset_quat'. Consider fixing your asset to use the standard Y-up convention instead."
                )
                y_up_quat = (1.0, -1.0, 0.0, 0.0)
                self.offset_quat = tuple(
                    gu.transform_quat_by_quat(
                        np.array(y_up_quat, dtype=gs.np_float), np.array(self.offset_quat, dtype=gs.np_float)
                    )
                )
                if self.scale is not None:
                    scale_arr = np.atleast_1d(np.array(self.scale))
                    if scale_arr.size == 3:
                        self.scale = (scale_arr[0], scale_arr[2], scale_arr[1])
            self.file_meshes_are_zup = False
        elif self.file_meshes_are_zup is None:
            self.file_meshes_are_zup = True

        return self


class MeshSet(Mesh):
    files: tuple[Any, ...] = Field(default=(), strict=False)
    poss: tuple[Vec3FType, ...] = Field(default=(), strict=False)
    eulers: tuple[Vec3FType, ...] = Field(default=(), strict=False)


############################ Rigid & Articulated ############################


class MJCF(FileMorph):
    """
    Morph loaded from a MJCF file. This morph only supports `RigidEntity`

    Note
    ----
    MJCF file always contains a 'world' body. Although this body is added to the kinematic tree, it is used to define
    the initial pose of the root link. If `pos`, `euler`, or `quat` is specified, it will override the root pose that
    was originally specified in the MJCF file.

    Note
    ----
    Genesis currently processes MJCF as if it describing a single entity instead of an actual scene. This means that
    there is a single gigantic kinematic chain comprising multiple physical kinematic chains connected together using
    fee joints. The definition of kinematic chain has been stretched a bit to allow us. In particular, there must be
    multiple root links instead of a single one. One other related limitation is global / world options defined in MJCF
    but must be set at the scene-level in Genesis are completely ignored at the moment, e.g. the simulation timestep,
    integrator or constraint solver. Building an actual scene hierarchy with multiple independent entities may be
    supported in the future.

    Note
    ----
    Collision filters defined in MJCF are considered "local", i.e. they only apply to collision pairs for which both
    geometries along to that specific entity. This means that there is no way to filter out collision pairs between
    primitive and MJCF entity at the moment.

    Parameters
    ----------
    file : str
        The path to the file.
    scale : float or tuple, optional
        The scaling factor for the size of the entity. If a float, it scales uniformly.
        If a 3-tuple, it scales along each axis. Defaults to 1.0.
        Note that 3-tuple scaling is only supported for `gs.morphs.Mesh`.
    pos : tuple, shape (3,), optional
        The position of the entity in meters as a translational offset. Mathematically, 'pos' and 'euler' options
        correspond respectively to the translational and rotational part of a transform that it is (left) applied on the
        original pose of all floating base links in the kinematic tree indiscriminately. Defaults to (0.0, 0.0, 0.0).
    euler : tuple, shape (3,), optional
        The euler angles of the entity in degrees as a rotational offset. This follows scipy's extrinsic x-y-z rotation
        convention. See 'pos' option documentation for details. Defaults to (0.0, 0.0, 0.0).
    quat : tuple, shape (4,), optional
        The quaternion (w-x-y-z convention) of the entity's baselink. If specified, `euler` will be ignored.
        Defaults to None.
    decimate : bool, optional
        Whether to decimate (simplify) the collision mesh. Defaults to True when convexify is True and False otherwise,
        since decimation removes the surface detail a non-convex collision mesh is kept for. **This is only used for
        RigidEntity.**
    decimate_face_num : int, optional
        The number of faces to decimate to. Defaults to 500. **This is only used for RigidEntity.**
    decimate_aggressiveness : int
        How hard the decimation process will try to match the target number of faces, as a integer ranging from 0 to 8.
        0 is losseless. 2 preserves all features of the original geometry. 5 may significantly alters the original
        geometry if necessary. 8 does what needs to be done at all costs. Defaults to 5.
        **This is only used for RigidEntity.**
    watertighten : int, optional
        Aggressiveness of the watertight wrap built for a non-convex (``convexify=False``) collision mesh, as an
        integer from 0 to 8 on the same scale as ``decimate_aggressiveness``. The wrap closes an open or
        self-intersecting mesh into the single watertight surface a grid signed distance field requires, decimating
        it under a feature-preserving cost cutoff. 0 bypasses the wrap (mesh kept as-is), higher values collapse more
        of it, and 8 is the strongest decimation the cutoff still allows (every level stays watertight and preserves
        the closed shape rather than collapsing thin cross-sections). ``None`` skips watertightening altogether.
        Defaults to 5. **This is only used for RigidEntity.**
    convexify : bool, optional
        Whether to convexify the entity. When convexify is True, all the meshes in the entity will each be converted
        to a set of convex hulls. The mesh with be decomposed into multiple convex components if a single one is not
        sufficient to met the desired accuracy (see 'decompose_(robot|object)_error_threshold' documentation). The
        module 'coacd' is used for this decomposition process. If not given, it defaults to `True` for `RigidEntity`
        and `False` for other deformable entities.
    decompose_nonconvex : bool, optional
        This parameter is deprecated. Please use 'convexify' and 'decompose_(robot|object)_error_threshold' instead.
    decompose_object_error_threshold : float, optional:
        For basic rigid objects (mug, table...), skip convex decomposition if the relative difference between the
        volume of original mesh and its convex hull is lower than this threshold.
        0.0 to enforce decomposition, float("inf") to disable it completely. Defaults to 0.15 (15%).
    decompose_robot_error_threshold : float, optional:
        For poly-articulated robots, skip convex decomposition if the relative difference between the volume of
        original mesh and its convex hull is lower than this threshold.
        0.0 to enforce decomposition, float("inf") to disable it completely. Defaults to float("inf").
    coacd_options : CoacdOptions, optional
        Options for configuring coacd convex decomposition. Needs to be a `gs.options.CoacdOptions` object.
    recompute_inertia : bool, optional
        Force recomputing spatial inertia of links from their geometry. This option is useful to import partially
        broken assets from external providers that cannot be re-exported from source. Default to False.
    parse_glb_with_zup : bool, optional
        This parameter is deprecated, see file_meshes_are_zup.
    file_meshes_are_zup : bool, optional
        Defines if the mesh files are expressed in a Z-up or Y-up coordinate system. If set to true, meshes are loaded
        as Z-up and no transforms are applied to the input data. If set to false, all meshes undergo a conversion step
        where the original coordinates are transformed as follows: (X, Y, Z) → (X, -Z, Y). Defaults to True.
    visualization : bool, optional
        Whether the entity needs to be visualized. Set it to False if you need a invisible object only for collision
        purposes. Defaults to True. `visualization` and `collision` cannot both be False.
    collision : bool, optional
        Whether the entity needs to be considered for collision checking. Defaults to True.
        `visualization` and `collision` cannot both be False.
    batch_fixed_verts : bool, optional
        Whether to batch fixed vertices. This will allow setting env-specific poses to fixed geometries, at the cost of
        significantly increasing memory usage. Default to true. **This is only used for RigidEntity.**
    align : bool, optional
        Whether to reframe root links so that the link origin coincides with the center of mass and its axes are
        aligned with the principal axes of inertia. Only applies to root (floating-base) links. Default to False.
        **This is only used for RigidEntity.**
    default_armature : float, optional
        Default rotor inertia of the actuators, **as a fraction of the joint-space sub-tree inertia each degree of
        freedom drives at the initial configuration**. It applies to every revolute or prismatic joint moving a link on
        its own whose armature the model file leaves at zero, regardless of whether it is actuated, and an authored
        armature is kept as it is. The rotor inertia regularises the constraint solve, at the cost of slowing every
        such joint down by that fraction. None to disable. Defaults to 0.1 if MuJoCo compatibility is disabled on
        the rigid solver, None otherwise.
    exclude_ground_plane : bool, optional
        Whether to exclude plane geometries authored directly under the MJCF worldbody if any. Defaults to False.
    """

    pos: Vec3FType | None = None
    quat: UnitVec4FType | None = None
    default_armature: float | None = Field(default=0.1, ge=0)
    exclude_ground_plane: StrictBool = False

    @model_validator(mode="before")
    @classmethod
    def _enforce_isotropic_scale(cls, data: dict) -> dict:
        # Anisotropic scaling is ill-defined for poly-articulated robots because link positions depend on configuration,
        # making the effect of per-axis scaling configuration-dependent. Limiting to scalar factor avoids this.
        scale = np.atleast_1d(np.array(data.get("scale", 1.0)))
        if scale.std() > gs.EPS:
            gs.raise_exception("Anisotropic scaling is not supported by MJCF morph.")
        data["scale"] = float(scale.mean())
        return data

    def model_post_init(self, context: Any) -> None:
        if not self.is_format(MJCF_FORMAT):
            gs.raise_exception(f"Expected `{MJCF_FORMAT}` extension for MJCF file: {self.file}")

    def _identifier(self) -> str:
        # A morph outlives the file it came from, because the entity builds only once. An unreadable file falls back
        # to the identifier of the morph type.
        if isinstance(self.file, str):
            try:
                if name := mju.get_model_name(self.file):
                    return name
            except (ET.ParseError, OSError):
                pass
        return super()._identifier()


class URDF(FileMorph):
    """
    Morph loaded from a URDF or XACRO file. This morph only supports `RigidEntity`.
    If you need to create a `Drone` entity, use `gs.morphs.Drone` instead.

    XACRO files (``.urdf.xacro`` or ``.xacro``) are automatically preprocessed into plain URDF using the ``xacro``
    package. All standard xacro features (macros, properties, includes, conditionals, substitution args) are supported.
    Use ``xacro_args`` to override ``xacro:arg`` declarations at load time. The only limitation is that
    ``$(find package_name)`` substitutions require ROS's ``ament_index_python``; without ROS, these will raise an error.

    Note
    ----
    As part of performance optimization, links connected via a fixed joint are merged if `merge_fixed_links` is True.
    This is turned on by default, and can help improve simulation speed without affecting any dynamics and rendering
    behaviors. However, in cases where certain links are still needed as independent links, such as virtual
    end-effector links created for being used as IK targets, these links will not be merged if their names are added
    to `links_to_keep`. You can also completely turn off link merging by setting `merge_fixed_links` to False,
    but it's recommended to use `merge_fixed_links=True` in combination with `links_to_keep` for better performance.

    Parameters
    ----------
    file : str
        The path to the file.
    scale : float or tuple, optional
        The scaling factor for the size of the entity. If a float, it scales uniformly.
        If a 3-tuple, it scales along each axis. Defaults to 1.0.
        Note that 3-tuple scaling is only supported for `gs.morphs.Mesh`.
    pos : tuple, shape (3,), optional
        The position of the entity in meters as a translational offset. Mathematically, 'pos' and 'euler' options
        correspond respectively to the translational and rotational part of a transform that it is (left) applied on the
        original pose of all floating base links in the kinematic tree indiscriminately. Defaults to (0.0, 0.0, 0.0).
    euler : tuple, shape (3,), optional
        The euler angles of the entity in degrees as a rotational offset. This follows scipy's extrinsic x-y-z rotation
        convention. See 'pos' option documentation for details. Defaults to (0.0, 0.0, 0.0).
    quat : tuple, shape (4,), optional
        The quaternion (w-x-y-z convention) of the entity. If specified, `euler` will be ignored. Defaults to None.
    decimate : bool, optional
        Whether to decimate (simplify) the collision mesh. Defaults to True when convexify is True and False otherwise,
        since decimation removes the surface detail a non-convex collision mesh is kept for. **This is only used for
        RigidEntity.**
    decimate_face_num : int, optional
        The number of faces to decimate to. Defaults to 500. **This is only used for RigidEntity.**
    decimate_aggressiveness : int
        How hard the decimation process will try to match the target number of faces, as a integer ranging from 0 to 8.
        0 is losseless. 2 preserves all features of the original geometry. 5 may significantly alters the original
        geometry if necessary. 8 does what needs to be done at all costs. Defaults to 5.
        **This is only used for RigidEntity.**
    watertighten : int, optional
        Aggressiveness of the watertight wrap built for a non-convex (``convexify=False``) collision mesh, as an
        integer from 0 to 8 on the same scale as ``decimate_aggressiveness``. The wrap closes an open or
        self-intersecting mesh into the single watertight surface a grid signed distance field requires, decimating
        it under a feature-preserving cost cutoff. 0 bypasses the wrap (mesh kept as-is), higher values collapse more
        of it, and 8 is the strongest decimation the cutoff still allows (every level stays watertight and preserves
        the closed shape rather than collapsing thin cross-sections). ``None`` skips watertightening altogether.
        Defaults to 5. **This is only used for RigidEntity.**
    convexify : bool, optional
        Whether to convexify the entity. When convexify is True, all the meshes in the entity will each be converted
        to a set of convex hulls. The mesh with be decomposed into multiple convex components if a single one is not
        sufficient to met the desired accuracy (see 'decompose_(robot|object)_error_threshold' documentation). The
        module 'coacd' is used for this decomposition process. If not given, it defaults to `True` for `RigidEntity`
        and `False` for other deformable entities.
    decompose_nonconvex : bool, optional
        This parameter is deprecated. Please use 'convexify' and 'decompose_(robot|object)_error_threshold' instead.
    decompose_object_error_threshold : float, optional:
        For basic rigid objects (mug, table...), skip convex decomposition if the relative difference between the
        volume of original mesh and its convex hull is lower than this threshold.
        0.0 to enforce decomposition, float("inf") to disable it completely. Defaults to 0.15 (15%).
    decompose_robot_error_threshold : float, optional:
        For poly-articulated robots, skip convex decomposition if the relative difference between the volume of
        original mesh and its convex hull is lower than this threshold.
        0.0 to enforce decomposition, float("inf") to disable it completely. Defaults to float("inf").
    coacd_options : CoacdOptions, optional
        Options for configuring coacd convex decomposition. Needs to be a `gs.options.CoacdOptions` object.
    recompute_inertia : bool, optional
        Force recomputing spatial inertia of links from their geometry. This option is useful to import partially
        broken assets from external providers that cannot be re-exported from source. Default to False.
    parse_glb_with_zup : bool, optional
        This parameter is deprecated, see file_meshes_are_zup.
    file_meshes_are_zup : bool, optional
        Defines if the mesh files are expressed in a Z-up or Y-up coordinate system. If set to true, meshes are loaded
        as Z-up and no transforms are applied to the input data. If set to false, all meshes undergo a conversion step
        where the original coordinates are transformed as follows: (X, Y, Z) → (X, -Z, Y). Defaults to True.
    visualization : bool, optional
        Whether the entity needs to be visualized. Set it to False if you need a invisible object only for collision
        purposes. Defaults to True. `visualization` and `collision` cannot both be False.
    collision : bool, optional
        Whether the entity needs to be considered for collision checking. Defaults to True.
        `visualization` and `collision` cannot both be False.
    fixed : bool, optional
        Whether the baselink of the entity should be fixed. Defaults to False.
    batch_fixed_verts : bool, optional
        Whether to batch fixed vertices. This will allow setting env-specific poses to fixed geometries, at the cost of
        significantly increasing memory usage. Default to true. **This is only used for RigidEntity.**
    prioritize_urdf_material : bool, optional
        Sometimes a geom in a urdf file will be assigned a color or a texture, and the geom asset file also contains its
        own visual material. This parameter controls whether to prioritize the URDF-defined material over the asset's
        own material. Defaults to False.
    merge_fixed_links : bool, optional
        Whether to merge links connected via a fixed joint. Defaults to True.
    links_to_keep : list of str, optional
        A list of link names that should not be skipped during link merging. Defaults to [].
    align : bool, optional
        Whether to reframe root links so that the link origin coincides with the center of mass and its axes are
        aligned with the principal axes of inertia. Only applies to root (floating-base) links. Default to False.
        **This is only used for RigidEntity.**
    default_armature : float, optional
        Default rotor inertia of the actuators, **as a fraction of the joint-space sub-tree inertia each degree of
        freedom drives at the initial configuration**. It applies to every revolute or prismatic joint moving a link on
        its own whose armature the model file leaves at zero, regardless of whether it is actuated, and an authored
        armature is kept as it is. The rotor inertia regularises the constraint solve, at the cost of slowing every
        such joint down by that fraction. None to disable. Defaults to 0.1 if MuJoCo compatibility is disabled on
        the rigid solver, None otherwise.
    xacro_args : dict, optional
        Key-value pairs to override ``xacro:arg`` declarations in the xacro file
        (e.g. ``{"use_sim": "true", "arm_length": "0.5"}``). Only used for ``.xacro`` files. Defaults to ``{}``.
    """

    fixed: StrictBool = False
    prioritize_urdf_material: StrictBool = False
    merge_fixed_links: StrictBool = True
    links_to_keep: StrArrayType = ()
    default_armature: float | None = Field(default=0.1, ge=0)
    xacro_args: FrozenDictType[str, str] = {}

    @model_validator(mode="before")
    @classmethod
    def _enforce_isotropic_scale(cls, data: dict) -> dict:
        # Anisotropic scaling is ill-defined for poly-articulated robots. See MJCF for details.
        scale = np.atleast_1d(np.array(data.get("scale", 1.0)))
        if scale.std() > gs.EPS:
            gs.raise_exception("Anisotropic scaling is not supported by URDF morph.")
        data["scale"] = float(scale.mean())
        return data

    def model_post_init(self, context: Any) -> None:
        if not (self.is_format(URDF_FORMAT) or self.is_format(XACRO_FORMAT)):
            gs.raise_exception(f"Expected `{URDF_FORMAT}` or `{XACRO_FORMAT}` extension for URDF file: {self.file}")

    def is_format(self, format):
        if isinstance(self.file, urdfpy.URDF):
            return format == URDF_FORMAT
        return super().is_format(format)

    def _identifier(self) -> str:
        if isinstance(self.file, str):
            try:
                return uu.get_robot_name(self.file)
            except (ValueError, ET.ParseError, FileNotFoundError, OSError):
                pass
        return super()._identifier()


class Drone(FileMorph):
    """
    Morph loaded from a URDF file for creating a `DroneEntity`.

    Note
    ----
    Visual geom in the propeller links will be used for spinning animation.

    Parameters
    ----------
    file : str
        The path to the file.
    scale : float or tuple, optional
        The scaling factor for the size of the entity. If a float, it scales uniformly. If a 3-tuple, it scales along
        each axis. Defaults to 1.0. Note that 3-tuple scaling is only supported for `gs.morphs.Mesh`.
    pos : tuple, shape (3,), optional
        The position of the entity in meters as a translational offset. Mathematically, 'pos' and 'euler' options
        correspond respectively to the translational and rotational part of a transform that it is (left) applied on the
        original pose of all floating base links in the kinematic tree indiscriminately. Defaults to (0.0, 0.0, 0.0).
    euler : tuple, shape (3,), optional
        The euler angles of the entity in degrees as a rotational offset. This follows scipy's extrinsic x-y-z rotation
        convention. See 'pos' option documentation for details. Defaults to (0.0, 0.0, 0.0).
    quat : tuple, shape (4,), optional
        The quaternion (w-x-y-z convention) of the entity. If specified, `euler` will be ignored. Defaults to None.
    decimate : bool, optional
        Whether to decimate (simplify) the collision mesh. Defaults to True when convexify is True and False otherwise,
        since decimation removes the surface detail a non-convex collision mesh is kept for. **This is only used for
        RigidEntity.**
    decimate_face_num : int, optional
        The number of faces to decimate to. Defaults to 500. **This is only used for RigidEntity.**
    decimate_aggressiveness : int
        How hard the decimation process will try to match the target number of faces, as a integer ranging from 0 to 8.
        0 is losseless. 2 preserves all features of the original geometry. 5 may significantly alters the original
        geometry if necessary. 8 does what needs to be done at all costs. Defaults to 5.
        **This is only used for RigidEntity.**
    watertighten : int, optional
        Aggressiveness of the watertight wrap built for a non-convex (``convexify=False``) collision mesh, as an
        integer from 0 to 8 on the same scale as ``decimate_aggressiveness``. The wrap closes an open or
        self-intersecting mesh into the single watertight surface a grid signed distance field requires, decimating
        it under a feature-preserving cost cutoff. 0 bypasses the wrap (mesh kept as-is), higher values collapse more
        of it, and 8 is the strongest decimation the cutoff still allows (every level stays watertight and preserves
        the closed shape rather than collapsing thin cross-sections). ``None`` skips watertightening altogether.
        Defaults to 5. **This is only used for RigidEntity.**
    convexify : bool, optional
        Whether to convexify the entity. When convexify is True, all the meshes in the entity will each be converted
        to a set of convex hulls. The mesh with be decomposed into multiple convex components if a single one is not
        sufficient to met the desired accuracy (see 'decompose_(robot|object)_error_threshold' documentation). The
        module 'coacd' is used for this decomposition process. If not given, it defaults to `True` for `RigidEntity`
        and `False` for other deformable entities.
    decompose_nonconvex : bool, optional
        This parameter is deprecated. Please use 'convexify' and 'decompose_(robot|object)_error_threshold' instead.
    decompose_object_error_threshold : float, optional:
        For basic rigid objects (mug, table...), skip convex decomposition if the relative difference between the
        volume of original mesh and its convex hull is lower than this threshold.
        0.0 to enforce decomposition, float("inf") to disable it completely. Defaults to 0.15 (15%).
    decompose_robot_error_threshold : float, optional:
        For poly-articulated robots, skip convex decomposition if the relative difference between the volume of
        original mesh and its convex hull is lower than this threshold.
        0.0 to enforce decomposition, float("inf") to disable it completely. Defaults to float("inf").
    coacd_options : CoacdOptions, optional
        Options for configuring coacd convex decomposition. Needs to be a `gs.options.CoacdOptions` object.
    recompute_inertia : bool, optional
        Force recomputing spatial inertia of links from their geometry. This option is useful to import partially
        broken assets from external providers that cannot be re-exported from source. Default to False.
    parse_glb_with_zup : bool, optional
        This parameter is deprecated, see file_meshes_are_zup.
    file_meshes_are_zup : bool, optional
        Defines if the mesh files are expressed in a Z-up or Y-up coordinate system. If set to true, meshes are loaded
        as Z-up and no transforms are applied to the input data. If set to false, all meshes undergo a conversion step
        where the original coordinates are transformed as follows: (X, Y, Z) → (X, -Z, Y). Defaults to True.
    visualization : bool, optional
        Whether the entity needs to be visualized. Set it to False if you need a invisible object only for collision
        purposes. Defaults to True. `visualization` and `collision` cannot both be False.
    collision : bool, optional
        **NB**: Drone doesn't support collision checking for now.
    prioritize_urdf_material : bool, optional
        Sometimes a geom in a urdf file will be assigned a color or a texture, and the geom asset file also contains its
        own visual material. This parameter controls whether to prioritize the URDF-defined material over the asset's
        own material. Defaults to False.
    model : str, optional
        The model of the drone. Defaults to 'CF2X'. Supported models are 'CF2X', 'CF2P', and 'RACE'.
    COM_link_name : str, optional
        This option is deprecated. The true Center of Mass (CoM) will be used instead of requesting the user to manually
        specify the name of the link that represents the center of mass.
    propellers_link_names : sequence of str, optional
        This option is deprecated and will be removed in the future. Please use 'propellers_link_name' instead.
    propellers_link_name : sequence of str, optional
        The names of the links that represent the propellers. Defaults to
        ('prop0_link', 'prop1_link', 'prop2_link', 'prop3_link').
    propellers_spin : sequence of int, optional
        The spin direction of the propellers. 1: CCW, -1: CW. Defaults to (-1, 1, -1, 1).
    merge_fixed_links : bool, optional
        Whether to merge links connected via a fixed joint. Defaults to True.
    links_to_keep : list of str, optional
        A list of link names that should not be skipped during link merging. Defaults to ().
    default_armature : float, optional
        Default rotor inertia of the actuators, **as a fraction of the joint-space sub-tree inertia each degree of
        freedom drives at the initial configuration**. It applies to every revolute or prismatic joint moving a link on
        its own whose armature the model file leaves at zero, regardless of whether it is actuated, and an authored
        armature is kept as it is. The rotor inertia regularises the constraint solve, at the cost of slowing every
        such joint down by that fraction. None to disable. Defaults to 0.1 if MuJoCo compatibility is disabled on
        the rigid solver, None otherwise.
    default_base_ang_damping_scale : float, optional
        Default angular damping applied on the floating base that will be rescaled by the total mass.
        None to disable. Default to 1e-5.
    """

    model: Literal["CF2X", "CF2P", "RACE"] = "CF2X"
    prioritize_urdf_material: StrictBool = False
    propellers_link_name: StrArrayType = ("prop0_link", "prop1_link", "prop2_link", "prop3_link")
    propellers_spin: tuple[int, ...] = Field(default=(-1, 1, -1, 1), strict=False)  # 1: CCW, -1: CW
    merge_fixed_links: StrictBool = True
    links_to_keep: StrArrayType = ()
    default_armature: float | None = Field(default=0.1, ge=0)
    default_base_ang_damping_scale: float | None = 1e-5

    def __init__(
        self,
        *,
        COM_link_name: str | None = None,
        propellers_link_names: tuple[str, ...] | None = None,
        **data,
    ):
        if COM_link_name is not None:
            gs.logger.warning("'COM_link_name' is deprecated. The true Center of Mass will be used instead.")

        if propellers_link_names is not None:
            gs.logger.warning("'propellers_link_names' is deprecated. Use 'propellers_link_name' instead.")
            if "propellers_link_name" in data:
                gs.raise_exception("'propellers_link_names' cannot be combined with 'propellers_link_name'.")
            data["propellers_link_name"] = propellers_link_names

        # Make sure that propellers links are preserved
        prop_links = data.get("propellers_link_name", self.model_fields["propellers_link_name"].default)
        links_to_keep = data.get("links_to_keep", self.model_fields["links_to_keep"].default)
        data["links_to_keep"] = tuple(set([*links_to_keep, *prop_links]))

        super().__init__(**data)

        if not self.is_format(URDF_FORMAT):
            gs.raise_exception(f"Drone only supports `{URDF_FORMAT}` extension: {self.file}")


class Terrain(Morph):
    """

    Morph for creating a rigid terrain. This can be instantiated from two choices:
    1) a grid of subterrains generated using the given configurations
    2) a terrain generated using the given height field.

    If randomize is True, subterrain type that involves randomness will have random parameters.
    Otherwise, they will use fixed random seed 0.

    Users can easily configure the subterrain types by specifying the `subterrain_types` parameter.
    If using a single string, it will be repeated for all subterrains. If it's a 2D list, it should have the same shape
    as `n_subterrains`. The supported subterrain types are:

    - 'flat_terrain': flat terrain
    - 'random_uniform_terrain': random uniform terrain
    - 'sloped_terrain': sloped terrain
    - 'pyramid_sloped_terrain': pyramid sloped terrain
    - 'discrete_obstacles_terrain': discrete obstacles terrain
    - 'wave_terrain': wave terrain
    - 'stairs_terrain': stairs terrain
    - 'pyramid_stairs_terrain': pyramid stairs terrain
    - 'stepping_stones_terrain': stepping stones terrain

    Note
    ----
    Rigid terrain will also be represented as SDF for collision checking, but its resolution is auto-computed and
    ignores the value specified in `gs.materials.Rigid()`.

    Parameters
    ----------
    file : str
        The path to the file.
    scale : float or tuple, optional
        The scaling factor for the size of the entity. If a float, it scales uniformly.
        If a 3-tuple, it scales along each axis. Defaults to 1.0.
        Note that 3-tuple scaling is only supported for `gs.morphs.Mesh`.
    pos : tuple, shape (3,), optional
        The position of the entity in meters. Defaults to (0.0, 0.0, 0.0).
    visualization : bool, optional
        Whether the entity needs to be visualized. Set it to False if you need a invisible object only for collision
        purposes. Defaults to True. `visualization` and `collision` cannot both be False.
    collision : bool, optional
        Whether the entity needs to be considered for collision checking. Defaults to True.
        `visualization` and `collision` cannot both be False.
    randomize : bool, optional
        Whether to randomize the subterrains that involve randomness. Defaults to False.
    n_subterrains : tuple of int, optional
        The number of subterrains in x and y directions. Defaults to (3, 3).
    subterrain_size : tuple of float, optional
        The size of each subterrain in meters. Defaults to (12.0, 12.0).
    horizontal_scale : float, optional
        The size of each cell in the subterrain in meters. Defaults to 0.25.
    vertical_scale : float, optional
        The height of each step in the subterrain in meters. Defaults to 0.005.
    uv_scale : float, optional
        The scale of the UV mapping for the terrain. Defaults to 1.0.
    subterrain_types : str or 2D list of str, optional
        The types of subterrains to generate. If a string, it will be repeated for all subterrains.
        If a 2D list, it should have the same shape as `n_subterrains`.
    height_field : array-like or torch.Tensor, optional
        The height field to generate the terrain. If specified, all other configurations will be ignored.
        Defaults to None.
    name : str, optional
        The name of the terrain. If specified, the terrain will only be generated once for a given set of options and
        later loaded from cache, instead of being re-generated systematically when building the scene. This holds true
        no matter if `randomize` is True.
    from_stored : str, optional
        This parameter is deprecated.
    subterrain_parameters : dictionary, optional
        Lets users pick their own subterrain parameters.
    batch_fixed_verts : bool, optional
        Whether to batch fixed vertices. This will allow setting env-specific poses to fixed geometries, at the cost of
        significantly increasing memory usage. Default to false. **This is only used for RigidEntity.**
    """

    batch_fixed_verts: StrictBool = False
    randomize: StrictBool = False
    n_subterrains: Vec2IType = (3, 3)
    subterrain_size: tuple[float, float] = (12.0, 12.0)
    horizontal_scale: PositiveFloat = 0.25
    vertical_scale: PositiveFloat = 0.005
    uv_scale: PositiveFloat = 1.0
    subterrain_types: Any = [
        ["flat_terrain", "random_uniform_terrain", "stepping_stones_terrain"],
        ["pyramid_sloped_terrain", "discrete_obstacles_terrain", "wave_terrain"],
        ["random_uniform_terrain", "pyramid_stairs_terrain", "sloped_terrain"],
    ]
    height_field: FGridArrayType | None = None
    name: str | None = None
    subterrain_parameters: dict[str, dict] | None = None

    _SUPPORTED_SUBTERRAIN_TYPES: ClassVar[tuple[str, ...]] = (
        "flat_terrain",
        "fractal_terrain",
        "random_uniform_terrain",
        "sloped_terrain",
        "pyramid_sloped_terrain",
        "discrete_obstacles_terrain",
        "wave_terrain",
        "stairs_terrain",
        "pyramid_stairs_terrain",
        "stepping_stones_terrain",
    )

    def __init__(self, *, from_stored: str | None = None, **data):
        if from_stored is not None:
            gs.logger.warning("'from_stored' is deprecated. Use 'name' instead.")
            if data.get("name") is None:
                data["name"] = from_stored
            elif from_stored != data.get("name"):
                gs.raise_exception("'from_stored' and 'name' cannot both be set to different values.")

        # Merge subterrain_parameters with defaults
        custom_params = data.get("subterrain_parameters") or {}
        terrain_types = set(self.default_params) | set(custom_params)
        overwritten_params = {}
        for terrain_type in terrain_types:
            default_value = self.default_params.get(terrain_type, {})
            custom_value = custom_params.get(terrain_type, {})
            overwritten_params[terrain_type] = default_value | custom_value
        data["subterrain_parameters"] = overwritten_params

        # Expand subterrain_types string to 2D list
        subterrain_types = data.get("subterrain_types")
        if isinstance(subterrain_types, str):
            n_subterrains = data.get("n_subterrains", self.model_fields["n_subterrains"].default)
            data["subterrain_types"] = [[subterrain_types] * n_subterrains[1] for _ in range(n_subterrains[0])]

        super().__init__(**data)

    def model_post_init(self, context: Any) -> None:
        if self.height_field is not None:
            return

        if not isinstance(self.subterrain_types, str):
            if np.array(self.subterrain_types).shape != (self.n_subterrains[0], self.n_subterrains[1]):
                gs.raise_exception(
                    "`subterrain_types` should be either a string or a 2D list of strings with the same shape as `n_subterrains`."
                )

        for row in self.subterrain_types:
            for subterrain_type in row:
                if subterrain_type not in self._SUPPORTED_SUBTERRAIN_TYPES:
                    gs.raise_exception(
                        f"Unsupported subterrain type: {subterrain_type}, should be one of {list(self._SUPPORTED_SUBTERRAIN_TYPES)}"
                    )

        if not mu.is_approx_multiple(self.subterrain_size[0], self.horizontal_scale) or not mu.is_approx_multiple(
            self.subterrain_size[1], self.horizontal_scale
        ):
            gs.raise_exception("`subterrain_size` should be divisible by `horizontal_scale`.")

    def _identifier(self) -> str:
        return self.name if self.name else super()._identifier()

    @property
    def default_params(self):
        return {
            "flat_terrain": {},
            "fractal_terrain": {
                "levels": 8,
                "scale": 5.0,
            },
            "random_uniform_terrain": {
                "min_height": -0.1,
                "max_height": 0.1,
                "step": 0.1,
                "downsampled_scale": 0.5,
            },
            "sloped_terrain": {
                "slope": -0.5,
            },
            "pyramid_sloped_terrain": {
                "slope": -0.1,
            },
            "discrete_obstacles_terrain": {
                "max_height": 0.05,
                "min_size": 1.0,
                "max_size": 5.0,
                "num_rects": 20,
            },
            "wave_terrain": {
                "num_waves": 2.0,
                "amplitude": 0.1,
            },
            "stairs_terrain": {
                "step_width": 0.75,
                "step_height": -0.1,
            },
            "pyramid_stairs_terrain": {
                "step_width": 0.75,
                "step_height": -0.1,
            },
            "stepping_stones_terrain": {
                "stone_size": 1.0,
                "stone_distance": 0.25,
                "max_height": 0.2,
                "platform_size": 0.0,
            },
        }

    @property
    def subterrain_params(self):
        return self.subterrain_parameters


class USD(FileMorph):
    """
    Configuration class for USD file loading with advanced processing options.

    This class encapsulates the file path and processing parameters for USD loading,
    allowing users to control convexification, decimation, and decomposition behavior
    when loading USD scenes via add_stage().

    Parameters
    ----------
    file : str
        The path to the USD file.
    scale : float or tuple, optional
        The scaling factor for the size of the entity. If a float, it scales uniformly.
        If a 3-tuple, it scales along each axis. Defaults to 1.0.
        Note that 3-tuple scaling is only supported for `gs.morphs.Mesh`.
    pos : tuple, shape (3,), optional
        The position of the entity in meters. Defaults to (0.0, 0.0, 0.0).
    euler : tuple, shape (3,), optional
        The euler angle of the entity in degrees. This follows scipy's extrinsic x-y-z rotation convention.
        Defaults to (0.0, 0.0, 0.0).
    quat : tuple, shape (4,), optional
        The quaternion (w-x-y-z convention) of the entity. If specified, `euler` will be ignored. Defaults to None.
    decimate : bool, optional
        Whether to decimate (simplify) the collision mesh. Defaults to True when convexify is True and False otherwise,
        since decimation removes the surface detail a non-convex collision mesh is kept for. **This is only used for
        RigidEntity.**
    decimate_face_num : int, optional
        The number of faces to decimate to. Defaults to 500. **This is only used for RigidEntity.**
    decimate_aggressiveness : int
        How hard the decimation process will try to match the target number of faces, as a integer ranging from 0 to 8.
        0 is losseless. 2 preserves all features of the original geometry. 5 may significantly alters the original
        geometry if necessary. 8 does what needs to be done at all costs. Defaults to 2.
        **This is only used for RigidEntity.**
    watertighten : int, optional
        Aggressiveness of the watertight wrap built for a non-convex (``convexify=False``) collision mesh, as an
        integer from 0 to 8 on the same scale as ``decimate_aggressiveness``. The wrap closes an open or
        self-intersecting mesh into the single watertight surface a grid signed distance field requires, decimating
        it under a feature-preserving cost cutoff. 0 bypasses the wrap (mesh kept as-is), higher values collapse more
        of it, and 8 is the strongest decimation the cutoff still allows (every level stays watertight and preserves
        the closed shape rather than collapsing thin cross-sections). ``None`` skips watertightening altogether.
        Defaults to 5. **This is only used for RigidEntity.**
    convexify : bool, optional
        Whether to convexify the entity. When convexify is True, all the meshes in the entity will each be converted
        to a set of convex hulls. The mesh will be decomposed into multiple convex components if the convex hull is not
        sufficient to met the desired accuracy (see 'decompose_(robot|object)_error_threshold' documentation). The
        module 'coacd' is used for this decomposition process. If not given, it defaults to `True` for `RigidEntity`
        and `False` for other deformable entities.
    decompose_nonconvex : bool, optional
        This parameter is deprecated. Please use 'convexify' and 'decompose_(robot|object)_error_threshold' instead.
    decompose_object_error_threshold : float, optional:
        For basic rigid objects (mug, table...), skip convex decomposition if the relative difference between the
        volume of original mesh and its convex hull is lower than this threshold.
        0.0 to enforce decomposition, float("inf") to disable it completely. Defaults to 0.15 (15%).
    decompose_robot_error_threshold : float, optional:
        For poly-articulated robots, skip convex decomposition if the relative difference between the volume of
        original mesh and its convex hull is lower than this threshold.
        0.0 to enforce decomposition, float("inf") to disable it completely. Defaults to float("inf").
    coacd_options : CoacdOptions, optional
        Options for configuring coacd convex decomposition. Needs to be a `gs.options.CoacdOptions` object.
    recompute_inertia : bool, optional
        Force recomputing spatial inertia of links from their geometry. This option is useful to import partially
        broken assets from external providers that cannot be re-exported from source. Default to False.
    align : bool, optional
        Whether to reframe root links so that the link origin coincides with the center of mass and its axes are
        aligned with the principal axes of inertia. Only applies to root (floating-base) links. Default to False.
        **This is only used for RigidEntity.**
    file_meshes_are_zup : bool, optional
        Defines if the mesh files are expressed in a Z-up or Y-up coordinate system. If set to true, meshes are loaded
        as Z-up and no transforms are applied to the input data. If set to false, all meshes undergo a conversion step
        where the original coordinates are transformed as follows: (X, Y, Z) → (X, -Z, Y). Defaults to True.
    visualization : bool, optional
        Whether the entity needs to be visualized. Set it to False if you need a invisible object only for collision
        purposes. Defaults to True. `visualization` and `collision` cannot both be False.
        **This is only used for RigidEntity.**
    collision : bool, optional
        Whether the entity needs to be considered for collision checking. Defaults to True.
        `visualization` and `collision` cannot both be False. **This is only used for RigidEntity.**
    batch_fixed_verts : bool, optional
        Whether to batch fixed vertices. This will allow setting env-specific poses to fixed geometries, at the cost of
        significantly increasing memory usage. Default to true. **This is only used for RigidEntity.**
    default_armature : float, optional
        Default rotor inertia of the actuators, **as a fraction of the joint-space sub-tree inertia each degree of
        freedom drives at the initial configuration**. It applies to every revolute or prismatic joint moving a link on
        its own whose armature the model file leaves at zero, regardless of whether it is actuated, and an authored
        armature is kept as it is. The rotor inertia regularises the constraint solve, at the cost of slowing every
        such joint down by that fraction. None to disable. Defaults to 0.1.

    Joint Dynamics Options
    ----------------------
    joint_friction_attr_candidates : List[str], optional
        List of candidate attribute names for joint friction. The parser will try these in order.
        If no matching attribute is found, Genesis default (0.0) is used.
        Defaults to ["physxJoint:jointFriction", "physics:jointFriction", "jointFriction", "friction"].
    joint_armature_attr_candidates : List[str], optional
        List of candidate attribute names for joint armature. The parser will try these in order.
        If no matching attribute is found, Genesis default (0.0) is used.
        Defaults to ["physxJoint:armature", "physics:armature", "armature"].
    revolute_joint_stiffness_attr_candidates : List[str], optional
        List of candidate attribute names for revolute joint stiffness. The parser will try these in order.
        If no matching attribute is found, Genesis default (0.0) is used.
        Defaults to ["physxLimit:angular:stiffness", "physics:stiffness", "stiffness"].
    revolute_joint_damping_attr_candidates : List[str], optional
        List of candidate attribute names for revolute joint damping. The parser will try these in order.
        If no matching attribute is found, Genesis default (0.0) is used.
        Defaults to ["physxLimit:angular:damping", "physics:angular:damping", "angular:damping"].
    prismatic_joint_stiffness_attr_candidates : List[str], optional
        List of candidate attribute names for prismatic joint stiffness. The parser will try these in order.
        If no matching attribute is found, Genesis default (0.0) is used.
        Defaults to ["physxLimit:linear:stiffness", "physxLimit:X:stiffness", "physxLimit:Y:stiffness",
        "physxLimit:Z:stiffness", "physics:linear:stiffness", "linear:stiffness"].
    prismatic_joint_damping_attr_candidates : List[str], optional
        List of candidate attribute names for prismatic joint damping. The parser will try these in order.
        If no matching attribute is found, Genesis default (0.0) is used.
        Defaults to ["physxLimit:linear:damping", "physxLimit:X:damping", "physxLimit:Y:damping",
        "physxLimit:Z:damping", "physics:linear:damping", "linear:damping"].

    Geometry Parsing Options
    -------------------------
    collision_mesh_prim_patterns : List[str], optional
        List of regex patterns to match collision mesh prim names. Patterns are tried in order
        until a match is found. The parser uses `re.match()` to check if a USD prim's name
        matches each pattern from the start of the string.

        When a prim matches a collision pattern, it is treated as collision-only geometry
        (not used for visualization). If a prim matches neither a visual nor collision pattern,
        it is treated as both visual and collision geometry by default.

        Defaults to [r"^([cC]ollision).*"].
    visual_mesh_prim_patterns : List[str], optional
        List of regex patterns to match visual mesh prim names. Patterns are tried in order
        until a match is found. The parser uses `re.match()` to check if a USD prim's name
        matches each pattern from the start of the string.

        When a prim matches a visual pattern, it is treated as visual-only geometry
        (not used for collision detection). If a prim matches neither a visual nor collision
        pattern, it is treated as both visual and collision geometry by default.

        Defaults to [r"^([vV]isual).*"].

    USD specific Options
    ----------------
    prim_path : str, optional
        The parsing target prim path. Defaults to None.
    """

    # Mesh Options
    file_meshes_are_zup: StrictBool | None = None
    fixed: StrictBool | None = None
    default_armature: float | None = Field(default=0.1, ge=0)

    # Joint Dynamics Options
    joint_friction_attr_candidates: StrArrayType = (
        "physxJoint:jointFriction",  # Isaac-Sim assets compatibility
        "physics:jointFriction",  # unoffical USD attribute, some assets may adapt to this attribute
        "jointFriction",  # unoffical USD attribute, some assets may adapt to this attribute
        "friction",  # unoffical USD attribute, some assets may adapt to this attribute
    )
    joint_armature_attr_candidates: StrArrayType = (
        "physxJoint:armature",  # Isaac-Sim assets compatibility
        "physics:armature",  # unoffical USD attribute, some assets may adapt to this attribute
        "armature",  # unoffical USD attribute, some assets may adapt to this attribute
    )
    revolute_joint_stiffness_attr_candidates: StrArrayType = (
        "physxLimit:angular:stiffness",  # Isaac-Sim assets compatibility
        "physics:stiffness",  # unoffical USD attribute, some assets may adapt to this attribute
        "stiffness",  # unoffical USD attribute, some assets may adapt to this attribute
    )
    revolute_joint_damping_attr_candidates: StrArrayType = (
        "physxLimit:angular:damping",  # Isaac-Sim assets compatibility
        "physics:angular:damping",  # unoffical USD attribute, some assets may adapt to this attribute
        "angular:damping",  # unoffical USD attribute, some assets may adapt to this attribute
    )
    prismatic_joint_stiffness_attr_candidates: StrArrayType = (
        "physxLimit:linear:stiffness",  # Isaac-Sim assets compatibility
        "physxLimit:X:stiffness",  # Isaac-Sim assets compatibility
        "physxLimit:Y:stiffness",  # Isaac-Sim assets compatibility
        "physxLimit:Z:stiffness",  # Isaac-Sim assets compatibility
        "physics:linear:stiffness",  # unoffical USD attribute, some assets may adapt to this attribute
        "linear:stiffness",  # unoffical USD attribute, some assets may adapt to this attribute
    )
    prismatic_joint_damping_attr_candidates: StrArrayType = (
        "physxLimit:linear:damping",  # Isaac-Sim assets compatibility
        "physxLimit:X:damping",  # Isaac-Sim assets compatibility
        "physxLimit:Y:damping",  # Isaac-Sim assets compatibility
        "physxLimit:Z:damping",  # Isaac-Sim assets compatibility
        "physics:linear:damping",  # unoffical USD attribute, some assets may adapt to this attribute
        "linear:damping",  # unoffical USD attribute, some assets may adapt to this attribute
    )

    # Geometry Parsing Options
    collision_mesh_prim_patterns: StrArrayType = (r"^([cC]ollision).*",)
    visual_mesh_prim_patterns: StrArrayType = (r"^([vV]isual).*",)

    # USD specific Options
    prim_path: str | None = None

    def __init__(self, **data):
        super().__init__(**data)

        if self.file_meshes_are_zup is not None:
            gs.raise_exception(
                "Specifying `file_meshes_are_zup` not supported for morph USD. "
                "USD file has independent metadata `up_axis` for up axis specification."
            )

        if not self.is_format(USD_FORMATS):
            gs.raise_exception(f"Expected `{USD_FORMATS}` extension for USD file: {self.file}")

    def _identifier(self) -> str:
        if self.prim_path:
            return self.prim_path.rstrip("/").split("/")[-1]
        return super()._identifier()

    def __repr_name__(self):
        return f"{super().__repr_name__()[:-1]}, prim_path='{self.prim_path}')>"


def _exported_urdf(robot: urdfpy.URDF, exporting: serialization.Exporting) -> str:
    """Export a URDF model built in memory as the document it stands for.

    Relative asset paths stay as written: an in-memory model resolves them against the working directory when it is
    read (see '_loaded_urdf').
    """
    return ET.tostring(robot.to_xml(), encoding="unicode")


def _loaded_urdf(raw: str, loading: serialization.Loading) -> urdfpy.URDF:
    """Recreate the URDF model from its document, as an inline document is read."""
    node = ET.fromstring(raw)
    return urdfpy.URDF._from_xml(node, node, os.getcwd())


serialization.register(urdfpy.URDF, _exported_urdf, _loaded_urdf)

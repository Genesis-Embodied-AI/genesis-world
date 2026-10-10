"""
Unit test comparing analytical capsule-capsule contact detection with GJK.

This test creates a modified version of narrowphase.py in a temporary file that forces capsule-capsule and
sphere-capsule collisions to use GJK instead of analytical methods, allowing direct comparison between the two
approaches.

# errno

We abuse errno in this test, because it is considerably easier, and needs much less code, than attempting to add a
new tensor into one of the existing structures, and have that work for both ndarray and field, via monkey-patching.

errno is NOT designed for how we use it. Nevertheless with a couple of reasonable-ish assumptions we can work with it.

Assumption 1: when code runs normally and correctly, nothing in Genesis production code (not including test code) will
ever set bit 16 of errno to any value except 0.
Assumption 2: when taking a step, nothing in Genesis production code will set bit 16 of errno to any value at all -
including 0 - when running normally.

Both of these assumptions are implicitly tested by our code, in that should Genesis code violate them, our tests will
almost certainly fail.

Note that as part of our use of errno, we take full responsibility ourselves for resetting it to 0 before each test
scenario. We do not assume - nor require - any existing Genesis code to handle this for us, for example by setting errno
to 0 in set_qpos.

Note that, for completeness, Genesis code does handle resetting errno to 0, inside set_qpos, but for simplicity, we make
resetting errno explicit in this test.
"""

import copy
import importlib.util
import xml.etree.ElementTree as ET
from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING, Callable, cast

import numpy as np
import pytest

import trimesh
from scipy.spatial import ConvexHull

import quadrants as qd

import genesis as gs
import genesis.utils.array_class as array_class
import genesis.utils.geom as gu
from genesis.engine.solvers.rigid.collider import support_field
from genesis.utils.misc import qd_to_numpy, tensor_to_array

from ..conftest import TOL_SINGLE
from ..utils.assertions import assert_allclose, assert_equal

if TYPE_CHECKING:
    from genesis.engine.entities import RigidEntity


ERRNO_CALLED_GJK_K1 = 1 << 16
ERRNO_CALLED_GJK_K2 = 1 << 17
POS_TOL = 1e-2  # otherwise tests fail

# Tolerances for checking results against hand-computed expected values.
# Analytical solutions should be near-exact; GJK needs more slack; reason unclear.
#
# Penetration tolerance: absolute error in metres.
# Normal tolerance: maximum allowed value of (1 - |dot(actual, expected)|).
#   e.g. 1e-5 means the normal must agree to within ~0.26 degrees,
#        1e-2 means within ~8 degrees.
ANALYTICAL_PEN_TOL = TOL_SINGLE
ANALYTICAL_NORMAL_TOL = TOL_SINGLE
GJK_PEN_TOL = 1e-2
GJK_NORMAL_TOL = 1e-2


def _check_expected_values(contacts, description, exp_pen, exp_normal, method_name, pen_tol, normal_tol):
    """Check that the deepest contact matches the expected penetration and/or normal, when provided.

    The other contacts of the pair are found on its perturbed copies, so that they penetrate no deeper.

    Parameters
    ----------
    pen_tol : float
        Maximum absolute penetration error (metres).
    normal_tol : float
        Maximum allowed ``1 - |dot(actual, expected)|``.
    """
    if not contacts or len(contacts["geom_a"]) == 0:
        return

    i_deepest = np.argmax(contacts["penetration"])
    if exp_pen is not None:
        pen = contacts["penetration"][i_deepest]
        assert abs(pen - exp_pen) < pen_tol, (
            f"[{method_name}] {description}: penetration {pen:.6f} != expected {exp_pen:.6f} (tol={pen_tol})"
        )

    if exp_normal is not None:
        normal = np.array(contacts["normal"][i_deepest])
        exp_n = np.array(exp_normal, dtype=gs.np_float)
        exp_n_len = np.linalg.norm(exp_n)
        assert gs.EPS is not None
        if exp_n_len > gs.EPS:
            dot_err = 1.0 - abs(np.dot(normal, exp_n / exp_n_len))
            assert dot_err < normal_tol, (
                f"[{method_name}] {description}: normal {normal} vs expected {exp_n / exp_n_len}, "
                f"1-|dot|={dot_err:.6e} >= {normal_tol}"
            )


def create_capsule_mjcf(name, pos, euler, radius, half_length):
    """Helper function to create an MJCF file with a single capsule."""
    mjcf = ET.Element("mujoco", model=name)
    ET.SubElement(mjcf, "compiler", angle="degree")
    ET.SubElement(mjcf, "option", timestep="0.01")
    worldbody = ET.SubElement(mjcf, "worldbody")
    body = ET.SubElement(
        worldbody,
        "body",
        name=name,
        pos=" ".join(map(str, pos)),
        euler=" ".join(map(str, euler)),
    )
    ET.SubElement(body, "geom", type="capsule", size=f"{radius} {half_length}")
    ET.SubElement(body, "joint", name=f"{name}_joint", type="free")
    return mjcf


def find_and_disable_condition(lines, function_name):
    """Find function call, look back for if/elif, and disable the entire multi-line condition.

    Skips occurrences whose guarding condition has already been disabled (contains 'False and').
    """
    # Find the line with the function call, skipping already-disabled occurrences
    call_line_idx = None
    for i, line in enumerate(lines):
        if function_name in line and "(" in line:
            # Look backwards for the guarding if/elif
            for j in range(i - 1, -1, -1):
                stripped = lines[j].strip()
                if stripped.startswith("if ") or stripped.startswith("elif "):
                    if "False and" in stripped:
                        break  # Already disabled, skip this occurrence
                    call_line_idx = i
                    break
                if stripped.startswith("else:"):
                    break
            if call_line_idx is not None:
                break

    if call_line_idx is None:
        raise ValueError(f"Could not find function call: {function_name}")

    # Look backwards to find the if or elif line
    condition_line_idx = None
    for i in range(call_line_idx - 1, -1, -1):
        stripped = lines[i].strip()
        if stripped.startswith("if ") or stripped.startswith("elif "):
            condition_line_idx = i
            break
        # Stop if we hit another major control structure
        if stripped.startswith("else:"):
            break

    if condition_line_idx is None:
        raise ValueError(f"Could not find if/elif for {function_name}")

    # Find the end of the condition (look for the : that ends it)
    condition_end_idx = condition_line_idx
    for i in range(condition_line_idx, call_line_idx):
        if ":" in lines[i]:
            condition_end_idx = i
            break

    # Modify the condition to wrap entire thing in False and (...)
    original_line = lines[condition_line_idx]
    indent = len(original_line) - len(original_line.lstrip())
    indent_str = original_line[:indent]

    # Extract the condition part (after if/elif and before :)
    if original_line.strip().startswith("if "):
        prefix = "if "
        rest = original_line.strip()[3:]  # Remove 'if '
    elif original_line.strip().startswith("elif "):
        prefix = "elif "
        rest = original_line.strip()[5:]  # Remove 'elif '
    else:
        raise ValueError(f"Expected if/elif but got: {original_line}")

    # If single-line condition
    if condition_end_idx == condition_line_idx:
        # Simple case: add False and
        modified_line = f"{indent_str}{prefix}False and {rest}"
        lines[condition_line_idx] = modified_line
    else:
        # Multi-line condition: wrap in False and (...)
        rest_no_colon = rest.rstrip(":").rstrip()
        lines[condition_line_idx] = f"{indent_str}{prefix}False and ({rest_no_colon}"

        # Add closing ) before the : on the last line
        last_line = lines[condition_end_idx]
        if ":" in last_line:
            # Insert ) before the :
            lines[condition_end_idx] = last_line.replace(":", "):", 1)

    return lines


def find_and_disable_all_conditions(lines, function_name):
    """Disable ALL if/elif conditions guarding calls to function_name."""
    while True:
        try:
            lines = find_and_disable_condition(lines, function_name)
        except ValueError:
            break
    return lines


def insert_errno_before_call(lines, function_call_pattern, errno_value, comment, index_var="i_b"):
    """Insert errno marker on the line before a function call."""
    call_line_idx = None
    for i, line in enumerate(lines):
        if function_call_pattern in line:
            idx = line.find(function_call_pattern)
            if idx != -1:
                if idx == 0 or not (line[idx - 1].isalnum() or line[idx - 1] == "_"):
                    stripped = line.strip()
                    if stripped.startswith("def ") or stripped.startswith("@"):
                        continue
                    call_line_idx = i
                    break
    else:
        raise ValueError(f"Could not find function call: {function_call_pattern}")

    indent_size = len(lines[call_line_idx]) - len(lines[call_line_idx].lstrip())
    errno_line = f"{' ' * indent_size}errno[{index_var}] |= {errno_value}  # {comment}"
    lines.insert(call_line_idx, errno_line)

    return lines


def insert_errno_before_all_calls(lines, function_call_pattern, errno_value, comment, index_var="i_b"):
    """Insert errno marker before ALL occurrences of a function call.

    Finds all call sites first, then inserts markers from bottom to top to preserve indices.
    """
    call_indices = []
    for i, line in enumerate(lines):
        if function_call_pattern in line:
            idx = line.find(function_call_pattern)
            if idx != -1:
                if idx == 0 or not (line[idx - 1].isalnum() or line[idx - 1] == "_"):
                    stripped = line.strip()
                    if stripped.startswith("def ") or stripped.startswith("@"):
                        continue
                    call_indices.append(i)
    if not call_indices:
        raise ValueError(f"Could not find function call: {function_call_pattern}")
    for call_line_idx in reversed(call_indices):
        indent_size = len(lines[call_line_idx]) - len(lines[call_line_idx].lstrip())
        errno_line = f"{' ' * indent_size}errno[{index_var}] |= {errno_value}  # {comment}"
        lines.insert(call_line_idx, errno_line)
    return lines


def create_modified_narrowphase_file(tmp_path: Path):
    """
    Create a modified version of narrowphase.py that forces capsule collisions to use GJK.

    Returns:
        str: Path to the temporary modified narrowphase.py file
    """
    # Find the original narrowphase.py file
    import genesis.engine.solvers.rigid.collider.narrowphase as narrowphase_module

    narrowphase_path = narrowphase_module.__file__

    with open(narrowphase_path, "r") as f:
        content = f.read()

    # remove relative imports
    content = content.replace("from . import ", "from genesis.engine.solvers.rigid.collider import ")
    content = content.replace("from .", "from genesis.engine.solvers.rigid.collider.")

    lines = content.split("\n")

    # Disable capsule-capsule analytical path in all kernels
    lines = find_and_disable_all_conditions(lines, "capsule_contact.func_capsule_capsule_contact")

    # Disable sphere-capsule analytical path in all kernels
    lines = find_and_disable_all_conditions(lines, "capsule_contact.func_sphere_capsule_contact")

    # Disable sphere-sphere analytical path in all kernels
    lines = find_and_disable_all_conditions(lines, "capsule_contact.func_sphere_sphere_contact")

    # Disable sphere-box analytical path in all kernels
    lines = find_and_disable_all_conditions(lines, "func_sphere_box_contact")

    # Insert errno marker in contact0's GJK path (before gjk.func_gjk call, uses i_b)
    lines = insert_errno_before_call(lines, "gjk.func_gjk(", ERRNO_CALLED_GJK_K1, "MODIFIED: GJK detection in contact0")

    # Insert errno before GJK calls in the monolithic kernel func_convex_convex_contact (uses i_b). This is the first
    # gjk.func_gjk_contact occurrence; the split path's call lives in _func_multicontact_run_detection, which has no
    # errno and is marked at its dispatch call site below instead.
    lines = insert_errno_before_call(
        lines, "diff_gjk.func_gjk_contact(", ERRNO_CALLED_GJK_K2, "MODIFIED: GJK called for collision detection"
    )
    lines = insert_errno_before_call(
        lines, "gjk.func_gjk_contact(", ERRNO_CALLED_GJK_K2, "MODIFIED: GJK called for collision detection"
    )

    # Split path: mark every multicontact dispatch call (in this forced-GJK scene the multicontact pass always resolves
    # contacts with GJK), indexing errno by the env of the queue entry it dispatches.
    lines = insert_errno_before_all_calls(
        lines,
        "_func_multicontact_detect(",
        ERRNO_CALLED_GJK_K2,
        "MODIFIED: GJK path in multicontact",
        "collider_state.narrowphase_work_queues.mpr_i_b[i_work]",
    )

    content = "\n".join(lines)

    # Debug: Check if errno was actually inserted
    assert content.count(f"|= {ERRNO_CALLED_GJK_K1}") >= 1, "contact0 GJK errno marker not inserted"
    assert content.count(f"|= {ERRNO_CALLED_GJK_K2}") >= 1, "multicontact GJK errno marker not inserted"

    temp_narrowphase_path = tmp_path / "narrow.py"
    with open(temp_narrowphase_path, "w") as f:
        f.write(content)

    return temp_narrowphase_path


def scene_add_sphere(tmp_path: Path, scene: gs.Scene, radius: float) -> "RigidEntity":
    sphere_mjcf = create_sphere_mjcf("sphere", (0, 0, 0), radius)
    sphere_path = tmp_path / "sphere.xml"
    ET.ElementTree(sphere_mjcf).write(sphere_path)
    entity_sphere = scene.add_entity(
        gs.morphs.MJCF(
            file=sphere_path,
            align=False,
        ),
        vis_mode="collision",
        visualize_contact=True,
    )
    return cast("RigidEntity", entity_sphere)


def scene_add_capsule(tmp_path: Path, scene: gs.Scene, half_length: float, radius: float) -> "RigidEntity":
    capsule_mjcf = create_capsule_mjcf("capsule", (0, 0, 0), (0, 0, 0), radius, half_length)
    capsule_path = tmp_path / "sphere.xml"
    ET.ElementTree(capsule_mjcf).write(capsule_path)
    entity_capsule = scene.add_entity(
        gs.morphs.MJCF(
            file=capsule_path,
            align=False,
        ),
        vis_mode="collision",
        visualize_contact=True,
    )
    return cast("RigidEntity", entity_capsule)


class AnalyticalVsGJKSceneCreator:
    def __init__(self, monkeypatch, build_scene: Callable, tmp_path: Path, show_viewer: bool) -> None:
        self.monkeypatch = monkeypatch
        self.build_scene = build_scene
        self.tmp_path = tmp_path
        self.scene_analytical: gs.Scene
        self.scene_gjk: gs.Scene
        self.entities_analytical = []
        self.entities_gjk = []
        self.show_viewer = show_viewer

    def setup_scenes(self) -> tuple[gs.Scene, gs.Scene]:
        """Build the analytical scene, then patch the narrowphase and build the GJK scene against it."""
        # Scene 1: Using ORIGINAL analytical collision detection
        self.scene_analytical = gs.Scene(
            show_viewer=self.show_viewer,
        )
        self.build_scene(
            scene=self.scene_analytical,
            entities=self.entities_analytical,
            tmp_path=self.tmp_path,
        )

        # Scene 2: Uses GJK. Building a scene compiles its substep kernels, which inline the narrowphase, so the patch
        # must precede the build. The two scenes select their collision algorithm statically, so the analytical scene
        # keeps the kernels it compiled against the original narrowphase. The tests using this creator disable the
        # kernel cache, which validates the funcs recorded when an entry was stored and would therefore serve a GJK
        # kernel compiled against the original narrowphase by another test.
        self.apply_gjk_patch()
        self.scene_gjk = gs.Scene(
            rigid_options=gs.options.RigidOptions(
                use_gjk_collision=True,
            ),
            show_viewer=self.show_viewer,
        )
        self.build_scene(scene=self.scene_gjk, tmp_path=self.tmp_path, entities=self.entities_gjk)

        return self.scene_analytical, self.scene_gjk

    def apply_gjk_patch(self) -> None:
        """Swap the narrowphase funcs called by the collider for the modified versions from a tmp file."""
        temp_narrowphase_path = create_modified_narrowphase_file(tmp_path=self.tmp_path)
        spec = importlib.util.spec_from_file_location("narrowphase_modified", temp_narrowphase_path)
        narrowphase_modified = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(narrowphase_modified)
        from genesis.engine.solvers.rigid.collider import collider

        self.monkeypatch.setattr(collider, "func_narrowphase_contact0", narrowphase_modified.func_narrowphase_contact0)
        self.monkeypatch.setattr(
            collider, "func_narrowphase_multicontact", narrowphase_modified.func_narrowphase_multicontact
        )
        self.monkeypatch.setattr(
            collider, "func_narrow_phase_convex_vs_convex", narrowphase_modified.func_narrow_phase_convex_vs_convex
        )

    def update_pos_quat_analytical(self, entity_idx: int, pos, euler) -> None:
        quat = gs.utils.geom.xyz_to_quat(xyz=np.array(euler, dtype=gs.np_float), degrees=True)
        self.entities_analytical[entity_idx].set_qpos((*pos, *quat))

    def update_pos_quat_gjk(self, entity_idx: int, pos, euler) -> None:
        quat = gs.utils.geom.xyz_to_quat(xyz=np.array(euler, dtype=gs.np_float), degrees=True)
        self.entities_gjk[entity_idx].set_qpos((*pos, *quat))

    def step_analytical(self):
        # see section '# errno' above for discussion on our abusing errno, and the assumptions which we make.
        self.scene_analytical._sim.rigid_solver._errno.fill(0)
        self.scene_analytical.step()
        errno_val = self.scene_analytical._sim.rigid_solver._errno[0]
        assert (errno_val & (ERRNO_CALLED_GJK_K1 | ERRNO_CALLED_GJK_K2)) == 0, "Analytical scene should not use GJK."

    def step_gjk(self, expect_collision: bool = True):
        # see section '# errno' above for discussion on our abusing errno, and the assumptions which we make.
        self.scene_gjk._sim.rigid_solver._errno.fill(0)
        self.scene_gjk.step()
        errno_val = self.scene_gjk._sim.rigid_solver._errno[0]
        use_split_narrowphase = self.scene_gjk._sim.rigid_solver.collider._use_split_narrowphase
        if use_split_narrowphase:
            # Kernel1 always runs GJK for collision detection (analytical paths are disabled).
            assert (errno_val & ERRNO_CALLED_GJK_K1) != 0, "GJK scene should use GJK in contact0."
        if expect_collision:
            # On GPU: multicontact is reached when contact0 detects a collision and enqueues the pair.
            # On CPU: the monolithic kernel calls gjk.func_gjk_contact directly (skipping gjk.func_gjk).
            assert (errno_val & ERRNO_CALLED_GJK_K2) != 0, "GJK scene should use GJK for contact generation."


@pytest.mark.slow("gpu")  # gpu ~400s
@pytest.mark.required
@pytest.mark.cache(False)
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
def test_capsule_capsule_vs_gjk(backend, monkeypatch, tmp_path: Path, show_viewer: bool, tol: float) -> None:
    # Compare the analytical capsule-capsule narrowphase against GJK by monkey-patching the collider. Multiple
    # configurations reuse a single scene build by moving the objects between checks.
    test_cases = [
        # (pos0, euler0, pos1, euler1, should_collide, description, exp_pen, exp_normal)
        # Segments cross at origin (distance=0), pen = sum of radii, normal is degenerate
        ((0, 0, 0), (0, 0, 0), (0.15, 0, 0), (0, 90, 0), True, "perpendicular_close", 0.2, None),
        # Parallel vertical, seg distance = 0.18, pen = 0.2 - 0.18 = 0.02
        ((0, 0, 0), (0, 0, 0), (0.18, 0, 0), (0, 0, 0), True, "parallel_light", 0.02, (-1, 0, 0)),
        ((0, 0, 0), (0, 90, 0), (0, 0.17, 0.17), (0, 90, 0), False, "horizontal_displaced", None, None),
        # Parallel vertical, seg distance = 0.15, pen = 0.2 - 0.15 = 0.05
        ((0, 0, 0), (0, 0, 0), (0.15, 0, 0), (0, 0, 0), True, "parallel_deep", 0.05, (-1, 0, 0)),
        # Segments cross at origin (distance=0), pen = sum of radii, normal is degenerate
        ((0, 0, 0), (0, 0, 0), (0, 0, 0), (90, 0, 0), True, "perpendicular_center", 0.2, None),
        # 45° capsule segment crosses the vertical segment at (0, 0, -0.15), so dist=0, pen = sum of radii
        ((0, 0, 0), (0, 0, 0), (0.15, 0, 0), (0, 45, 0), True, "diagonal_rotated", 0.2, None),
    ]

    radius = 0.1
    half_length = 0.25

    def build_scene(scene: gs.Scene, tmp_path: Path, entities: list):
        entities.append(scene_add_capsule(tmp_path, scene, half_length=half_length, radius=radius))
        entities.append(scene_add_capsule(tmp_path, scene, half_length=half_length, radius=radius))
        scene.build()

    scene_creator = AnalyticalVsGJKSceneCreator(
        monkeypatch=monkeypatch, build_scene=build_scene, tmp_path=tmp_path, show_viewer=show_viewer
    )
    scene_analytical, scene_gjk = scene_creator.setup_scenes()
    assert scene_analytical.rigid_solver.collider is not None
    assert scene_gjk.rigid_solver.collider is not None

    # Phase 1: Run all analytical scenarios (original, unpatched kernel)
    analytical_results = {}
    for pos0, euler0, pos1, euler1, should_collide, description, exp_pen, exp_normal in test_cases:
        try:
            scene_creator.update_pos_quat_analytical(entity_idx=0, pos=pos0, euler=euler0)
            scene_creator.update_pos_quat_analytical(entity_idx=1, pos=pos1, euler=euler1)
            scene_creator.step_analytical()

            contacts = scene_analytical.rigid_solver.collider.get_contacts(as_tensor=False, to_torch=False)
            has_collision = len(contacts["geom_a"]) > 0
            assert has_collision == should_collide, "Analytical collision mismatch!"
            _check_expected_values(
                contacts, description, exp_pen, exp_normal, "analytical", ANALYTICAL_PEN_TOL, ANALYTICAL_NORMAL_TOL
            )
            # Deep-copy so subsequent steps can't corrupt stored data
            analytical_results[description] = copy.deepcopy(contacts)
        except AssertionError as e:
            raise AssertionError(
                f"\nFAILED TEST SCENARIO (analytical phase): {description}\n"
                f"Capsule 0: pos={pos0}, euler={euler0}\n"
                f"Capsule 1: pos={pos1}, euler={euler1}\n"
                f"Expected collision: {should_collide}\n"
                f"Backend: {backend}\n"
                f"Radius: {radius}, Half-length: {half_length}\n"
            ) from e

    # Phase 2: Run all GJK scenarios (patched narrowphase)
    for pos0, euler0, pos1, euler1, should_collide, description, exp_pen, exp_normal in test_cases:
        try:
            scene_creator.update_pos_quat_gjk(entity_idx=0, pos=pos0, euler=euler0)
            scene_creator.update_pos_quat_gjk(entity_idx=1, pos=pos1, euler=euler1)
            scene_creator.step_gjk(should_collide)

            contacts_gjk = scene_gjk.rigid_solver.collider.get_contacts(as_tensor=False, to_torch=False)
            contacts_analytical = analytical_results[description]

            has_collision_analytical = contacts_analytical is not None and len(contacts_analytical["geom_a"]) > 0
            has_collision_gjk = contacts_gjk is not None and len(contacts_gjk["geom_a"]) > 0

            assert has_collision_analytical == has_collision_gjk, "Collision detection mismatch!"
            assert has_collision_gjk == should_collide

            _check_expected_values(contacts_gjk, description, exp_pen, exp_normal, "GJK", GJK_PEN_TOL, GJK_NORMAL_TOL)

            # Every contact lies within both capsules, no farther than the radius from their segments
            for pos, euler in ((pos0, euler0), (pos1, euler1)):
                axis = gs.utils.geom.quat_to_R(
                    gs.utils.geom.xyz_to_quat(xyz=np.array(euler, dtype=gs.np_float), degrees=True)
                )[:, 2]
                offsets = contacts_gjk["position"] - pos
                offsets_axial = np.clip(offsets @ axis, -half_length, half_length)
                dists = np.linalg.norm(offsets - offsets_axial[:, None] * axis, axis=-1)
                assert (dists <= radius + GJK_PEN_TOL).all()

            # If both detected a collision, compare the full contact manifold. Each analytical contact is matched to
            # its nearest GJK contact by position (order-independent), then position, penetration and normal are
            # compared for every contact - not just the first - so multi-contact manifolds are fully validated.
            if has_collision_analytical and has_collision_gjk:
                n_analytical = len(contacts_analytical["geom_a"])
                n_gjk = len(contacts_gjk["geom_a"])
                analytical_positions = np.array([contacts_analytical["position"][i] for i in range(n_analytical)])
                gjk_positions = np.array([contacts_gjk["position"][j] for j in range(n_gjk)])

                # Every analytical contact must have a matching GJK contact with agreeing position, penetration and
                # normal - checked for all points, not just the first. GJK may emit a few extra near-duplicate
                # manifold points, so contacts are matched by nearest position rather than requiring equal counts.
                for i in range(n_analytical):
                    j = int(np.argmin(np.linalg.norm(gjk_positions - analytical_positions[i], axis=1)))
                    assert np.linalg.norm(analytical_positions[i] - gjk_positions[j]) < POS_TOL, "Position mismatch!"
                    assert_allclose(
                        contacts_analytical["penetration"][i],
                        contacts_gjk["penetration"][j],
                        atol=POS_TOL,
                        rtol=0.1,
                        err_msg="Penetration mismatch!",
                    )
                    normal_a = np.array(contacts_analytical["normal"][i])
                    normal_g = np.array(contacts_gjk["normal"][j])
                    assert abs(np.dot(normal_a, normal_g)) > 0.95, "Normal mismatch!"

                # Parallel capsules produce a two-point manifold; verify both methods find it, and for the vertical
                # configuration that every contact lies on the line midway between the axes.
                if description in ["parallel_light", "parallel_deep"]:
                    assert n_analytical >= 2, f"Expected >=2 analytical contacts for {description}, got {n_analytical}"
                    if euler0 == (0, 0, 0) and euler1 == (0, 0, 0):
                        expected_xy = np.array([pos1[0] / 2, 0.0])  # Midpoint between capsules
                        for pos in (*analytical_positions, *gjk_positions):
                            assert_allclose(pos[:2], expected_xy, tol=POS_TOL)
                            assert -0.26 < pos[2] < 0.26
        except AssertionError as e:
            raise AssertionError(
                f"\nFAILED TEST SCENARIO (GJK phase): {description}\n"
                f"Capsule 0: pos={pos0}, euler={euler0}\n"
                f"Capsule 1: pos={pos1}, euler={euler1}\n"
                f"Expected collision: {should_collide}\n"
                f"Backend: {backend}\n"
                f"Radius: {radius}, Half-length: {half_length}\n"
            ) from e


@pytest.mark.required
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
def test_capsule_analytical_accuracy(tmp_path: Path, show_viewer: bool, tol: float):
    # Simple test case: two vertical capsules offset horizontally
    # Capsule 1: center at origin, radius=0.1, half_length=0.25
    # Capsule 2: center at (0.15, 0, 0), same size
    # Line segments are both vertical, closest points are at centers
    # Distance between segments: 0.15
    # Sum of radii: 0.2
    # Expected penetration: 0.2 - 0.15 = 0.05

    scene = gs.Scene(show_viewer=show_viewer)

    _cap1 = scene_add_capsule(tmp_path=tmp_path, scene=scene, half_length=0.25, radius=0.1)
    cap2 = scene_add_capsule(tmp_path=tmp_path, scene=scene, half_length=0.25, radius=0.1)
    scene.build()
    assert scene.rigid_solver.collider is not None

    cap2.set_pos((0.15, 0, 0))
    scene.step()

    contacts = scene.rigid_solver.collider.get_contacts(as_tensor=False, to_torch=False)
    assert len(contacts["geom_a"]) > 0

    penetration = contacts["penetration"][0]
    expected_pen = 0.05
    assert_allclose(penetration, expected_pen, tol=POS_TOL, err_msg="Analytical solution not exact!")

    assert_allclose(contacts["normal"][0], (-1.0, 0.0, 0.0), tol=tol)


def create_sphere_mjcf(name, pos, radius):
    """Helper function to create an MJCF file with a single sphere."""
    mjcf = ET.Element("mujoco", model=name)
    ET.SubElement(mjcf, "compiler", angle="degree")
    ET.SubElement(mjcf, "option", timestep="0.01")
    worldbody = ET.SubElement(mjcf, "worldbody")
    body = ET.SubElement(
        worldbody,
        "body",
        name=name,
        pos=f"{pos[0]} {pos[1]} {pos[2]}",
    )
    ET.SubElement(body, "geom", type="sphere", size=f"{radius}")
    ET.SubElement(body, "joint", name=f"{name}_joint", type="free")
    return mjcf


def create_box_mjcf(name, pos, euler, size):
    """Helper function to create an MJCF file with a single box (full-size, axis-aligned in local frame)."""
    mjcf = ET.Element("mujoco", model=name)
    ET.SubElement(mjcf, "compiler", angle="degree")
    ET.SubElement(mjcf, "option", timestep="0.01")
    worldbody = ET.SubElement(mjcf, "worldbody")
    body = ET.SubElement(
        worldbody,
        "body",
        name=name,
        pos=f"{pos[0]} {pos[1]} {pos[2]}",
        euler=f"{euler[0]} {euler[1]} {euler[2]}",
    )
    half = (0.5 * size[0], 0.5 * size[1], 0.5 * size[2])
    ET.SubElement(body, "geom", type="box", size=f"{half[0]} {half[1]} {half[2]}")
    ET.SubElement(body, "joint", name=f"{name}_joint", type="free")
    return mjcf


@pytest.mark.slow("gpu")  # gpu ~400s, dominated by recompiling the patched narrowphase kernels
@pytest.mark.required
@pytest.mark.cache(False)
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
def test_sphere_pairs_vs_gjk(backend, monkeypatch, tmp_path: Path, show_viewer: bool) -> None:
    # Compare the analytical sphere-capsule and sphere-sphere narrowphase against GJK by monkey-patching the
    # collider. One scene holds a reference sphere, a capsule and a second sphere; each configuration moves the
    # reference sphere and the case's partner while parking the third entity far away. The GJK arm of the
    # sphere-sphere cases also covers EPA robustness on smooth geometries, whose extremely small polytope faces near
    # convergence amplify the relative reprojection error and can cause false contact rejections.
    sphere_radius = 0.1
    capsule_radius = 0.1
    capsule_half_length = 0.25
    sphere_b_radius = 0.08

    test_cases = [
        # (partner_idx, sphere_pos, partner_pos, partner_euler, should_collide, description, exp_pen, exp_normal)
        # --- sphere vs capsule ---
        # Sphere above top cap: dist to segment endpoint (0,0,0.25) = 0.15, pen = 0.05
        (1, (0, 0, 0.4), (0, 0, 0), (0, 0, 0), True, "sphere_above_capsule_top", 0.05, (0, 0, 1)),
        # Sphere beside cylinder: dist to axis = 0.18, pen = 0.02
        (1, (0.18, 0, 0), (0, 0, 0), (0, 0, 0), True, "sphere_close_to_capsule", 0.02, (1, 0, 0)),
        # dist to axis = sqrt(0.17^2+0.17^2) ≈ 0.24 > 0.2, no collision
        (1, (0.17, 0.17, 0), (0, 0, 0), (0, 0, 0), False, "sphere_near_cylinder", None, None),
        (1, (0.35, 0, 0.35), (0, 0, 0), (0, 45, 0), False, "sphere_near_cap", None, None),
        # Sphere beside cylinder: dist to axis = 0.15, pen = 0.05
        (1, (0.15, 0, 0), (0, 0, 0), (0, 0, 0), True, "sphere_touching_cylinder", 0.05, (1, 0, 0)),
        # Sphere at capsule centre: dist = 0, pen = sum of radii = 0.2, normal is degenerate
        (1, (0, 0, 0), (0, 0, 0), (0, 0, 0), True, "sphere_at_capsule_center", 0.2, None),
        # Sphere near top cap: nearest segment pt = (0,0,0.25), dist = sqrt(0.15²+0.05²) ≈ 0.1581
        # pen = 0.2 - sqrt(0.025) ≈ 0.041886, normal along (3, 0, 1)
        (1, (0.15, 0, 0.3), (0, 0, 0), (0, 0, 0), True, "sphere_near_capsule_cap", 0.041886, (3, 0, 1)),
        # Horizontal capsule (axis along X after 90° Y rotation), sphere offset in Y: pen = 0.05
        (1, (0, 0.15, 0), (0, 0, 0), (0, 90, 0), True, "sphere_horizontal_capsule", 0.05, (0, 1, 0)),
        # --- sphere vs sphere ---
        # Diagonal offset: dist ≈ 0.1166, pen ≈ 0.0634
        (2, (0, 0, 0), (0.08, 0.06, 0.06), (0, 0, 0), True, "diagonal_3d", 0.0634, (0.08, 0.06, 0.06)),
        # Axis-aligned overlap: dist = 0.15, pen = 0.03
        (2, (0, 0, 0), (0.15, 0, 0), (0, 0, 0), True, "axis_aligned", 0.03, (1, 0, 0)),
        # Near-touching: dist = 0.17, pen = 0.01
        (2, (0, 0, 0), (0.17, 0, 0), (0, 0, 0), True, "near_touching", 0.01, (1, 0, 0)),
        # No collision: diagonal near-miss, dist ≈ 0.212 > 0.18, with per-axis offsets below the radii sum so the
        # bounding boxes still overlap and the pair reaches the narrowphase.
        (2, (0, 0, 0), (0.15, 0.15, 0), (0, 0, 0), False, "separated", None, None),
        # Concentric spheres: fully degenerate, just check collision is detected
        (2, (0, 0, 0), (0, 0, 0), (0, 0, 0), True, "concentric", None, None),
    ]

    def build_scene(scene: gs.Scene, tmp_path: Path, entities: list) -> None:
        entities.append(scene_add_sphere(tmp_path, scene, radius=sphere_radius))
        entities.append(scene_add_capsule(tmp_path, scene, half_length=capsule_half_length, radius=capsule_radius))
        entities.append(scene_add_sphere(tmp_path, scene, radius=sphere_b_radius))
        scene.build()

    scene_creator = AnalyticalVsGJKSceneCreator(
        monkeypatch=monkeypatch,
        build_scene=build_scene,
        tmp_path=tmp_path,
        show_viewer=show_viewer,
    )
    scene_analytical, scene_gjk = scene_creator.setup_scenes()
    assert scene_analytical.rigid_solver.collider is not None
    assert scene_gjk.rigid_solver.collider is not None

    # Phase 1: Run all analytical scenarios (original, unpatched kernel)
    analytical_results = {}
    for (
        partner_idx,
        sphere_pos,
        partner_pos,
        partner_euler,
        should_collide,
        description,
        exp_pen,
        exp_normal,
    ) in test_cases:
        try:
            scene_creator.update_pos_quat_analytical(entity_idx=0, pos=sphere_pos, euler=[0, 0, 0])
            scene_creator.update_pos_quat_analytical(entity_idx=partner_idx, pos=partner_pos, euler=partner_euler)
            scene_creator.update_pos_quat_analytical(entity_idx=3 - partner_idx, pos=(0.0, 0.0, 10.0), euler=[0, 0, 0])
            scene_creator.step_analytical()

            contacts = scene_analytical.rigid_solver.collider.get_contacts(as_tensor=False, to_torch=False)
            assert (len(contacts["geom_a"]) > 0) == should_collide, "Analytical collision mismatch"
            _check_expected_values(
                contacts, description, exp_pen, exp_normal, "analytical", ANALYTICAL_PEN_TOL, ANALYTICAL_NORMAL_TOL
            )
            # Deep-copy so subsequent steps can't corrupt stored data
            analytical_results[description] = copy.deepcopy(contacts)
        except AssertionError as e:
            raise AssertionError(f"\nFAILED TEST SCENARIO (analytical phase): {description}\n") from e

    # Phase 2: Run all GJK scenarios (patched narrowphase)
    for (
        partner_idx,
        sphere_pos,
        partner_pos,
        partner_euler,
        should_collide,
        description,
        exp_pen,
        exp_normal,
    ) in test_cases:
        try:
            scene_creator.update_pos_quat_gjk(entity_idx=0, pos=sphere_pos, euler=[0, 0, 0])
            scene_creator.update_pos_quat_gjk(entity_idx=partner_idx, pos=partner_pos, euler=partner_euler)
            scene_creator.update_pos_quat_gjk(entity_idx=3 - partner_idx, pos=(0.0, 0.0, 10.0), euler=[0, 0, 0])
            scene_creator.step_gjk(should_collide)

            contacts_gjk = scene_gjk.rigid_solver.collider.get_contacts(as_tensor=False, to_torch=False)
            contacts_analytical = analytical_results[description]
            assert (len(contacts_gjk["geom_a"]) > 0) == should_collide, "GJK collision mismatch"
            _check_expected_values(contacts_gjk, description, exp_pen, exp_normal, "GJK", GJK_PEN_TOL, GJK_NORMAL_TOL)

            # If both detected a collision, compare the contact details. The concentric sphere pair only compares
            # penetration: its contact frame is fully arbitrary and, with distinct radii, its contact position moves
            # with it. The coincident sphere-capsule case keeps a loose frame check, and its position is
            # direction-independent since the sphere radius equals half the penetration there.
            if should_collide:
                assert_allclose(
                    contacts_analytical["penetration"][0],
                    contacts_gjk["penetration"][0],
                    atol=POS_TOL,
                    rtol=0.1,
                    err_msg="Penetration mismatch!",
                )
                if description != "concentric":
                    normal_agreement = abs(
                        np.dot(np.array(contacts_analytical["normal"][0]), np.array(contacts_gjk["normal"][0]))
                    )
                    normal_tol = 0.5 if description == "sphere_at_capsule_center" else 0.95
                    assert normal_agreement > normal_tol, "Normal mismatch!"
                    assert_allclose(contacts_analytical["position"][0], contacts_gjk["position"][0], tol=POS_TOL)
        except AssertionError as e:
            raise AssertionError(f"\nFAILED TEST SCENARIO (GJK phase): {description}\n") from e


def scene_add_box(tmp_path: Path, scene: gs.Scene, size) -> "RigidEntity":
    box_mjcf = create_box_mjcf("box", (0, 0, 0), (0, 0, 0), size)
    box_path = tmp_path / "box.xml"
    ET.ElementTree(box_mjcf).write(box_path)
    entity_box = scene.add_entity(
        gs.morphs.MJCF(
            file=box_path,
            align=False,
        ),
        vis_mode="collision",
        visualize_contact=True,
    )
    return cast("RigidEntity", entity_box)


@pytest.mark.slow  # ~250s
@pytest.mark.required
@pytest.mark.cache(False)
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
def test_sphere_box_vs_gjk(backend, monkeypatch, tmp_path: Path, show_viewer: bool) -> None:
    sphere_radius = 0.1
    box_size = (0.4, 0.4, 0.2)
    half = (0.5 * box_size[0], 0.5 * box_size[1], 0.5 * box_size[2])

    test_cases = [
        # (sphere_pos, box_pos, box_euler, should_collide, description, exp_pen, exp_normal)
        # Sphere directly above top face by 0.05 -> dist 0.05, pen = 0.05, normal +z
        ((0, 0, half[2] + sphere_radius - 0.05), (0, 0, 0), (0, 0, 0), True, "top_face_center", 0.05, (0, 0, 1)),
        # Sphere off-center above top face, just touching -> shallow contact, normal must be +z
        # This is the issue #2793 regression scenario
        (
            (0.05, 0.05, half[2] + sphere_radius - 1e-4),
            (0, 0, 0),
            (0, 0, 0),
            True,
            "shallow_top_offcenter",
            1e-4,
            (0, 0, 1),
        ),
        # Sphere just outside the +x +y +z corner (AABB overlaps but no actual contact)
        # Corner = (half[0], half[1], half[2]); offset along (1,1,1)*sphere_radius*0.7 from corner
        (
            (half[0] + 0.7 * sphere_radius, half[1] + 0.7 * sphere_radius, half[2] + 0.7 * sphere_radius),
            (0, 0, 0),
            (0, 0, 0),
            False,
            "near_corner_no_contact",
            None,
            None,
        ),
        # Sphere off the +x face -> normal +x, pen = 0.04
        ((half[0] + sphere_radius - 0.04, 0, 0), (0, 0, 0), (0, 0, 0), True, "x_face", 0.04, (1, 0, 0)),
        # Sphere off the -y face -> normal -y, pen = 0.03
        ((0, -(half[1] + sphere_radius - 0.03), 0), (0, 0, 0), (0, 0, 0), True, "ny_face", 0.03, (0, -1, 0)),
        # Sphere near a +x +z edge: closest point is the edge, normal along the diagonal
        # closest = (half[0], 0, half[2]) -> diff = (0.06, 0, 0.08), dist = 0.1, pen = 0
        # offset diff to (0.06*0.5, 0, 0.08*0.5) so the sphere has pen
        ((half[0] + 0.03, 0, half[2] + 0.04), (0, 0, 0), (0, 0, 0), True, "edge_x_z", 0.05, (3, 0, 4)),
        # Box rotated 45 deg around z -- still axis-aligned in own frame; sphere above
        ((0, 0, half[2] + sphere_radius - 0.05), (0, 0, 0), (0, 0, 45), True, "top_rotated_z", 0.05, (0, 0, 1)),
        # Box rotated 90 deg around y -> original local +x face is now world +z
        # Sphere above world origin -> contact with face that was +x in local frame
        (
            (0, 0, half[0] + sphere_radius - 0.05),
            (0, 0, 0),
            (0, 90, 0),
            True,
            "top_rotated_y90",
            0.05,
            (0, 0, 1),
        ),
    ]

    def build_scene(scene: gs.Scene, tmp_path: Path, entities: list) -> None:
        entities.append(scene_add_sphere(tmp_path, scene, radius=sphere_radius))
        entities.append(scene_add_box(tmp_path, scene, size=box_size))
        scene.build()

    scene_creator = AnalyticalVsGJKSceneCreator(
        monkeypatch=monkeypatch,
        build_scene=build_scene,
        tmp_path=tmp_path,
        show_viewer=show_viewer,
    )
    scene_analytical, scene_gjk = scene_creator.setup_scenes()
    assert scene_analytical.rigid_solver.collider is not None
    assert scene_gjk.rigid_solver.collider is not None

    analytical_results = {}
    for sphere_pos, box_pos, box_euler, should_collide, description, exp_pen, exp_normal in test_cases:
        try:
            scene_creator.update_pos_quat_analytical(entity_idx=0, pos=sphere_pos, euler=[0, 0, 0])
            scene_creator.update_pos_quat_analytical(entity_idx=1, pos=box_pos, euler=box_euler)
            scene_creator.step_analytical()

            contacts = scene_analytical.rigid_solver.collider.get_contacts(as_tensor=False, to_torch=False)
            has_collision = len(contacts["geom_a"]) > 0
            assert has_collision == should_collide, "Analytical collision mismatch"
            _check_expected_values(
                contacts, description, exp_pen, exp_normal, "analytical", ANALYTICAL_PEN_TOL, ANALYTICAL_NORMAL_TOL
            )
            analytical_results[description] = copy.deepcopy(contacts)
        except AssertionError as e:
            raise AssertionError(
                f"\nFAILED TEST SCENARIO (analytical phase): {description}\n"
                f"Sphere: pos={sphere_pos}\n"
                f"Box: pos={box_pos}, euler={box_euler}\n"
                f"Expected collision: {should_collide}\n"
                f"Backend: {backend}\n"
                f"Sphere radius: {sphere_radius}\n"
                f"Box size: {box_size}\n"
            ) from e

    for sphere_pos, box_pos, box_euler, should_collide, description, exp_pen, exp_normal in test_cases:
        try:
            scene_creator.update_pos_quat_gjk(entity_idx=0, pos=sphere_pos, euler=[0, 0, 0])
            scene_creator.update_pos_quat_gjk(entity_idx=1, pos=box_pos, euler=box_euler)
            scene_creator.step_gjk(should_collide)

            contacts_gjk = scene_gjk.rigid_solver.collider.get_contacts(as_tensor=False, to_torch=False)
            contacts_analytical = analytical_results[description]

            has_collision_analytical = len(contacts_analytical["geom_a"]) > 0
            has_collision_gjk = len(contacts_gjk["geom_a"]) > 0

            assert has_collision_analytical == has_collision_gjk, "Collision detection mismatch!"
            assert has_collision_gjk == should_collide

            _check_expected_values(contacts_gjk, description, exp_pen, exp_normal, "GJK", GJK_PEN_TOL, GJK_NORMAL_TOL)

            if has_collision_analytical and has_collision_gjk:
                pen_analytical = contacts_analytical["penetration"][0]
                pen_gjk = contacts_gjk["penetration"][0]

                normal_analytical = np.array(contacts_analytical["normal"][0])
                normal_gjk = np.array(contacts_gjk["normal"][0])

                pos_analytical = np.array(contacts_analytical["position"][0])
                pos_gjk = np.array(contacts_gjk["position"][0])
                assert_allclose(pen_analytical, pen_gjk, atol=POS_TOL, rtol=0.1, err_msg="Penetration mismatch!")

                normal_agreement = abs(np.dot(normal_analytical, normal_gjk))
                assert normal_agreement > 0.95, "Normal mismatch!"

                assert_allclose(pos_analytical, pos_gjk, tol=POS_TOL)
        except AssertionError as e:
            raise AssertionError(
                f"\nFAILED TEST SCENARIO (GJK phase): {description}\n"
                f"Sphere: pos={sphere_pos}\n"
                f"Box: pos={box_pos}, euler={box_euler}\n"
                f"Expected collision: {should_collide}\n"
                f"Backend: {backend}\n"
                f"Sphere radius: {sphere_radius}\n"
                f"Box size: {box_size}\n"
            ) from e


@pytest.mark.required
# Both arms carry mode-specific narrowphase blocks: MuJoCo compatibility gates its own acceptance tolerance,
# perturbation pattern and penetration override, which must stay mirrored between the split and monolithic consumers.
@pytest.mark.parametrize("enable_mujoco_compatibility", [False, True])
@pytest.mark.parametrize("backend", [gs.gpu])
def test_split_vs_monolithic_narrowphase(
    enable_mujoco_compatibility, monkeypatch, tmp_path: Path, show_viewer: bool, tol: float
) -> None:
    radius = 0.1
    half_length = 0.25

    scene = gs.Scene(
        rigid_options=gs.options.RigidOptions(
            enable_mujoco_compatibility=enable_mujoco_compatibility,
            use_gjk_collision=True,
        ),
        show_viewer=show_viewer,
    )
    capsule_a = scene_add_capsule(tmp_path, scene, half_length=half_length, radius=radius)
    capsule_b = scene_add_capsule(tmp_path, scene, half_length=half_length, radius=radius)
    box = scene.add_entity(
        gs.morphs.Box(
            size=(0.3, 0.3, 0.3),
            pos=(10.0, 0.0, 0.0),
        ),
        vis_mode="collision",
    )
    # Batched build: the split arm packs its work queue across environments through one global counter, a structure
    # a single environment cannot exercise.
    scene.build(n_envs=2)

    collider = scene.rigid_solver.collider
    assert collider is not None
    assert collider._use_split_narrowphase, "Expected split narrowphase on GPU backend"

    test_configs = [
        # (pos_a, euler_a, pos_b, euler_b, pos_box, atol, normal_atol). Analytic capsule-capsule contacts agree
        # across arms at the generic tolerance. A GJK/EPA contact carries per-config tolerances pinned to its
        # measured cross-kernel floors, since the two arms schedule the GJK/EPA chain's instructions differently;
        # the normal is loosest because its direction divides the witness-point difference by the penetration
        # length, which amplifies witness noise (see the contact conversion in gjk.py).
        ((0, 0, 0), (0, 0, 0), (0.15, 0, 0), (0, 90, 0), (10, 0, 0), None, None),
        ((0, 0, 0), (0, 0, 0), (0.18, 0, 0), (0, 0, 0), (10, 0, 0), None, None),
        ((0, 0, 0), (0, 0, 0), (0.15, 0, 0), (0, 0, 0), (10, 0, 0), None, None),
        ((0, 0, 0), (0, 0, 0), (0, 0, 0), (90, 0, 0), (10, 0, 0), None, None),
        # Tilted capsule with its lower end sphere on a box face: the perturbed multi-contact and its acceptance
        # tolerance run in both arms, on a contact whose deepest point is unique. A capsule lying flat on the face
        # makes a line contact whose detected witness is a rounding tie-break, differing across kernels.
        ((0, 0, 0.013), (0, 87, 0), (10, 0, 0.5), (0, 0, 0), (-0.25, 0.0, -0.245), 2e-4, 2e-3),
    ]

    for pos_a, euler_a, pos_b, euler_b, pos_box, config_atol, config_normal_atol in test_configs:
        atol = tol if config_atol is None else config_atol
        normal_atol = tol if config_normal_atol is None else config_normal_atol
        quat_a = gs.utils.geom.xyz_to_quat(xyz=np.array(euler_a, dtype=gs.np_float), degrees=True)
        quat_b = gs.utils.geom.xyz_to_quat(xyz=np.array(euler_b, dtype=gs.np_float), degrees=True)

        # Run with split narrowphase (default on GPU)
        capsule_a.set_qpos((*pos_a, *quat_a))
        capsule_b.set_qpos((*pos_b, *quat_b))
        box.set_pos(pos_box)
        scene.step()
        contacts_split = collider.get_contacts(as_tensor=False, to_torch=False)

        # Run with monolithic narrowphase, which runs GJK on the per-env state the split arm leaves unallocated
        monkeypatch.setattr(collider, "_use_split_narrowphase", False)
        collider.gjk.activate()
        capsule_a.set_qpos((*pos_a, *quat_a))
        capsule_b.set_qpos((*pos_b, *quat_b))
        box.set_pos(pos_box)
        scene.step()
        contacts_mono = collider.get_contacts(as_tensor=False, to_torch=False)
        monkeypatch.undo()

        n_contacts_split = [len(geoms) for geoms in contacts_split["geom_a"]]
        n_contacts_mono = [len(geoms) for geoms in contacts_mono["geom_a"]]
        assert n_contacts_split == n_contacts_mono, (
            f"Contact count mismatch: split={n_contacts_split}, mono={n_contacts_mono}"
        )
        if any(n_contacts_split):
            for field, field_atol in (("penetration", atol), ("position", atol), ("normal", normal_atol)):
                assert_allclose(contacts_split[field], contacts_mono[field], atol=field_atol, err_msg=field)


@pytest.mark.required
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
def test_contact_patch_full_box_box_manifold(show_viewer: bool) -> None:
    scene = gs.Scene(
        rigid_options=gs.options.RigidOptions(
            box_box_detection=False,
            use_gjk_collision=True,
            enable_contact_patch=True,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.6, -0.6, 0.5),
            camera_lookat=(0.0, 0.0, 0.15),
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(
        gs.morphs.Box(
            size=(0.2, 0.2, 0.2),
            pos=(0.0, 0.0, 0.0),
            fixed=True,
        ),
        vis_mode="collision",
    )
    scene.add_entity(
        gs.morphs.Box(
            size=(0.2, 0.2, 0.2),
            pos=(0.0, 0.0, 0.199),
            euler=(0, 0, 45),
        ),
        vis_mode="collision",
    )
    scene.build(n_envs=2)
    scene.step()

    # The 45-degree overlap of the two 0.2-wide faces is a regular octagon whose corners sit at the fixed box's face
    # boundary; the contact patch must report all 8 of them, at the midpoint depth of the 1e-3 overlap.
    corner = 0.1 * (np.sqrt(2.0) - 1.0)
    corners_xy = np.array([(sx * 0.1, sy * corner) for sx in (-1, 1) for sy in (-1, 1)])
    corners_xy = np.concatenate((corners_xy, corners_xy[:, ::-1]))
    corners_xy = corners_xy[np.lexsort((corners_xy[:, 1], corners_xy[:, 0]))]
    expected = np.column_stack((corners_xy, np.full(8, 0.0995)))
    contacts = scene.rigid_solver.collider.get_contacts(as_tensor=False, to_torch=False)
    for positions, penetrations in zip(contacts["position"], contacts["penetration"]):
        assert positions.shape[0] == 8
        # Rounded sort keys: corners tied in x must order by y identically on both sides of the comparison.
        order = np.lexsort((positions[:, 1].round(4), positions[:, 0].round(4)))
        assert_allclose(positions[order], expected, atol=1e-5)
        assert_allclose(penetrations, 1e-3, atol=1e-5)


@pytest.mark.slow  # ~150s
@pytest.mark.required
@pytest.mark.parametrize("gjk_collision", [True, False])
def test_multi_contact_overlap_corners(gjk_collision, show_viewer, tol):
    # A box rests on a wider one at a random yaw and offset in each environment, so that the overlap of their faces
    # takes every shape a pair of rectangles can clip into: slivers, short edges, near-collinear corners, octagons. The
    # contacts of a pair lie within this overlap at its exact depth, and on each of its corners when it has at most 4.
    N_ENVS = 128
    PENETRATION = 1e-4
    PRUNING_TOLERANCE = 0.02
    BASE_SIZE = (1.0, 0.6, 0.1)
    BOX_SIZE = (0.5, 0.4, 0.1)

    scene = gs.Scene(
        rigid_options=gs.options.RigidOptions(
            use_gjk_collision=gjk_collision,
            contact_pruning_tolerance=PRUNING_TOLERANCE,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.0, -1.5, 1.5),
            camera_lookat=(0.0, 0.0, 0.1),
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(
        gs.morphs.Box(
            size=BASE_SIZE,
            pos=(0.0, 0.0, 0.5 * BASE_SIZE[2]),
            fixed=True,
        ),
    )
    box = scene.add_entity(
        gs.morphs.Box(
            size=BOX_SIZE,
        ),
    )
    scene.build(n_envs=N_ENVS)

    yaw = np.random.uniform(0.0, 2.0 * np.pi, N_ENVS)
    is_aligned = np.random.random(N_ENVS) < 0.3
    yaw_aligned = 0.5 * np.pi * np.random.randint(4, size=is_aligned.sum())
    yaw[is_aligned] = yaw_aligned + np.random.uniform(-1e-3, 1e-3, is_aligned.sum())
    offset = np.random.uniform(-0.6, 0.6, (N_ENVS, 2))
    box_z = BASE_SIZE[2] + 0.5 * BOX_SIZE[2] - PENETRATION
    corners_tol = scene.rigid_solver.collider._mc_tolerance * 0.5 * np.linalg.norm(BOX_SIZE)
    box.set_pos(np.concatenate((offset, np.full((N_ENVS, 1), box_z)), axis=-1))
    box.set_quat(np.stack((np.cos(0.5 * yaw), np.zeros(N_ENVS), np.zeros(N_ENVS), np.sin(0.5 * yaw)), axis=-1))
    scene.step()
    contacts = scene.rigid_solver.collider.get_contacts(as_tensor=False, to_torch=False)

    corners_sign = np.array(((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)))
    for i_b in range(N_ENVS):
        # The overlap clips the top face of the base by the half-planes of the bottom face of the box
        rot = np.array(((np.cos(yaw[i_b]), -np.sin(yaw[i_b])), (np.sin(yaw[i_b]), np.cos(yaw[i_b]))))
        clip = offset[i_b] + (0.5 * corners_sign * BOX_SIZE[:2]) @ rot.T
        clip_edges = np.roll(clip, -1, axis=0) - clip
        half_normals = np.stack((-clip_edges[:, 1], clip_edges[:, 0]), axis=-1)
        half_normals *= np.sign(((clip.mean(axis=0) - clip) * half_normals).sum(axis=-1))[:, None]
        overlap = 0.5 * corners_sign * BASE_SIZE[:2]
        for half_normal, half_offset in zip(half_normals, -(half_normals * clip).sum(axis=-1)):
            overlap_side = half_offset + overlap @ half_normal
            overlap_clipped = []
            for point, point_next, side, side_next in zip(
                overlap, np.roll(overlap, -1, axis=0), overlap_side, np.roll(overlap_side, -1)
            ):
                if side >= 0.0:
                    overlap_clipped.append(point)
                if side * side_next < 0.0:
                    overlap_clipped.append(point + (point_next - point) * side / (side - side_next))
            overlap = np.array(overlap_clipped).reshape((-1, 2))
        overlap = overlap[np.linalg.norm(overlap - np.roll(overlap, 1, axis=0), axis=-1) > gs.EPS]

        # Faces overlapping along a point or a segment at most have no contact
        contacts_pos = contacts["position"][i_b][:, :2]
        if len(overlap) < 3:
            assert len(contacts_pos) == 0
            continue

        # Every contact lies within the overlap, at the depth of the box
        overlap_edges = np.roll(overlap, -1, axis=0) - overlap
        overlap_normals = np.stack((-overlap_edges[:, 1], overlap_edges[:, 0]), axis=-1)
        overlap_normals *= np.sign(((overlap.mean(axis=0) - overlap) * overlap_normals).sum(axis=-1))[:, None]
        overlap_normals /= np.linalg.norm(overlap_normals, axis=-1, keepdims=True)
        assert ((contacts_pos[:, None] - overlap[None]) * overlap_normals[None]).sum(axis=-1).min(initial=0.0) > -tol
        assert_allclose(contacts["penetration"][i_b], PENETRATION, tol=tol)

        # An overlap of at most 4 corners has a contact on each of them, unless the pruning drops it: the triangle a
        # corner forms with its two neighbors then covers less than the pruning tolerance of the overlap area.
        # Detection merges the contacts closer than the multi-contact tolerance, which scales with the smaller box of the
        # pair, so a contact stands for every corner within this distance.
        if len(overlap) <= 4:
            overlap_prev, overlap_next = np.roll(overlap, 1, axis=0), np.roll(overlap, -1, axis=0)
            corners_edges = np.stack((overlap - overlap_prev, overlap_next - overlap), axis=-2)
            corners_area = 0.5 * np.abs(np.linalg.det(corners_edges))
            overlap_area = 0.5 * np.abs(np.linalg.det(np.stack((overlap, overlap_next), axis=-2)).sum())
            is_kept = corners_area > PRUNING_TOLERANCE * overlap_area
            corners_dist = np.linalg.norm(overlap[:, None] - contacts_pos[None], axis=-1)
            assert (corners_dist.min(axis=1, initial=np.inf)[is_kept] < corners_tol).all()


@pytest.mark.required
def test_maximum_contact_area(show_viewer, tol):
    # A pillar of three, four or six sides rests on a wider pillar of seven, both of random convex sections, at evenly
    # spread yaws and a random tilt within the reach of the multi-contact perturbation, each pair at its own scale. The
    # overlap of their faces is the bottom face of the narrow pillar. GJK keeps the first contact exact, so that only the
    # multi-contact perturbation is on trial.
    N_YAWS = 16
    N_TRIALS = 4
    TOPS_N_SIDES = (3, 4, 6, 3, 4, 6, 3, 4, 6)
    SCALES = np.geomspace(0.05, 5.0, len(TOPS_N_SIDES))
    TILT_RATIO = 0.5
    PENETRATION = 1e-4
    BASE_RADIUS = 0.5
    TOP_RADIUS = 0.15
    PILLAR_HEIGHT = 0.1
    # Multi-contact detection guarantees every corner of a patch of at most four corners whose interior angle does not
    # exceed this bound. A larger patch gets five contacts spanning close to the largest pentagon it contains.
    CORNER_ANGLE_MAX = 0.75 * np.pi
    AREA_RATIO_MIN = 0.75

    # Each convex section takes its vertices on a circle stretched along one axis by up to 3, at jittered angles. The
    # jitter bounds the gaps between them, so that the wide section always contains a disk around its center. The narrow
    # sections of at most four sides are drawn again until every interior angle complies with the guarantee.
    bases_mesh, tops_mesh, tops_corners = [], [], []
    for scale, n_sides_top in zip(SCALES, TOPS_N_SIDES):
        for n_sides, radius, aspect_max, meshes in (
            (7, BASE_RADIUS, 1.0, bases_mesh),
            (n_sides_top, TOP_RADIUS, 3.0, tops_mesh),
        ):
            while True:
                angles = 2.0 * np.pi * (np.arange(n_sides) + np.random.uniform(-0.25, 0.25, n_sides)) / n_sides
                corners = np.stack((np.random.uniform(1.0, aspect_max) * np.cos(angles), np.sin(angles)), axis=-1)
                corners *= scale * radius / np.linalg.norm(corners, axis=-1).max()
                dirs_in = corners - np.roll(corners, 1, axis=0)
                dirs_out = np.roll(corners, -1, axis=0) - corners
                corners_turn = np.arccos(
                    (dirs_in * dirs_out).sum(axis=-1)
                    / (np.linalg.norm(dirs_in, axis=-1) * np.linalg.norm(dirs_out, axis=-1))
                )
                if n_sides > 4 or (np.pi - corners_turn <= CORNER_ANGLE_MAX).all():
                    break
            heights = scale * np.repeat((0.0, PILLAR_HEIGHT), n_sides)[:, None]
            meshes.append(trimesh.Trimesh(np.concatenate((np.tile(corners, (2, 1)), heights), axis=-1)).convex_hull)
        tops_corners.append(corners)

    scene = gs.Scene(
        rigid_options=gs.options.RigidOptions(
            use_gjk_collision=True,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.0, -60.0, 60.0),
            camera_lookat=(0.0, 0.0, 0.0),
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(
        morph=[gs.morphs.MeshSet(files=(mesh,), fixed=True) for mesh in bases_mesh],
        vis_mode="collision",
    )
    top = scene.add_entity(
        morph=[gs.morphs.MeshSet(files=(mesh,)) for mesh in tops_mesh],
        visualize_contact=True,
        vis_mode="collision",
    )
    scene.build(n_envs=len(SCALES) * N_YAWS, env_spacing=(6.0, 6.0))

    # Environments come in one block per variant, sweeping the yaws of the narrow pillar
    envs_scale = np.repeat(SCALES, N_YAWS)
    yaws = np.tile(2.0 * np.pi * np.arange(N_YAWS) / N_YAWS, len(SCALES))
    quats_yaw = gu.rotvec_to_quat(yaws[:, None] * np.array((0.0, 0.0, 1.0)))
    envs_corners = [np.pad(tops_corners[i_b // N_YAWS], ((0, 0), (0, 1))) for i_b in range(scene.n_envs)]
    tilt_max = TILT_RATIO * scene.rigid_solver.collider._mc_perturbation
    for _ in range(N_TRIALS):
        # The narrow pillar lies inside the disk that the wide section contains, anywhere within it, with its deepest
        # corner pressed into the wide one by the penetration
        tilts_angle = np.random.uniform(0.0, 2.0 * np.pi, scene.n_envs)
        tilts = np.random.uniform(0.0, tilt_max, scene.n_envs)[:, None] * np.stack(
            (np.cos(tilts_angle), np.sin(tilts_angle), np.zeros(scene.n_envs)), axis=-1
        )
        quats = gu.transform_quat_by_quat(quats_yaw, gu.rotvec_to_quat(tilts))
        envs_R = gu.quat_to_R(quats)
        envs_corners_z = np.array([(corners @ R.T)[:, 2].max() for corners, R in zip(envs_corners, envs_R)])
        offsets = envs_scale[:, None] * np.random.uniform(-TOP_RADIUS, TOP_RADIUS, (scene.n_envs, 2))
        tops_z = envs_scale * (PILLAR_HEIGHT - PENETRATION) - envs_corners_z
        top.set_pos(np.concatenate((offsets, tops_z[:, None]), axis=-1))
        top.set_quat(quats)
        if show_viewer:
            scene.visualizer.update()
        scene.rigid_solver.collider.clear()
        scene.rigid_solver.collider.detection()
        contacts = scene.rigid_solver.collider.get_contacts(as_tensor=False, to_torch=False)

        tops_pos = tensor_to_array(top.get_pos())
        for i_b in range(scene.n_envs):
            corners = tops_pos[i_b] + envs_corners[i_b] @ envs_R[i_b].T
            contacts_pos = contacts["position"][i_b]
            depth_max = envs_scale[i_b] * PILLAR_HEIGHT - corners[:, 2].min()
            assert ((contacts["penetration"][i_b] > 0.0) & (contacts["penetration"][i_b] < depth_max + tol)).all()
            # Every contact lies within the patch. A contact sits midway between its witnesses, which lie its penetration
            # apart along a normal that may follow the tilted face, so it leaves the patch by half the penetration times
            # the tilt at most.
            edges = np.roll(corners[:, :2], -1, axis=0) - corners[:, :2]
            edges_normal = np.stack((edges[:, 1], -edges[:, 0]), axis=-1) / np.linalg.norm(edges, axis=-1)[:, None]
            edges_normal *= np.sign(((corners[:, :2] - corners[:, :2].mean(axis=0)) * edges_normal).sum(axis=-1))[
                :, None
            ]
            contacts_side = ((contacts_pos[:, None, :2] - corners[None, :, :2]) * edges_normal[None]).sum(axis=-1)
            contacts_side_max = 0.5 * contacts["penetration"][i_b] * np.linalg.norm(tilts[i_b]) + tol * envs_scale[i_b]
            assert (contacts_side < contacts_side_max[:, None]).all()
            if len(corners) <= 4:
                corners_dist = np.linalg.norm(corners[:, None, :2] - contacts_pos[None, :, :2], axis=-1).min(axis=1)
                assert (corners_dist < tol * envs_scale[i_b]).all()
            else:
                area_max = max(
                    ConvexHull(corners[pentagon, :2]).volume
                    for pentagon in map(list, combinations(range(len(corners)), 5))
                )
                assert ConvexHull(contacts_pos[:, :2]).volume > AREA_RATIO_MIN * area_max


@pytest.mark.required
def test_mesh_support_exactness(show_viewer, tol):
    # The support table of a mesh returns, for any direction, a vertex that supports it, wherever it falls on the grid of
    # directions the table is laid on: across its cells, on their boundaries, along the normals of the hull faces, and
    # along the arcs between the normals of adjacent hull faces, where vertices tie. Meshes range from the eight corners
    # of a box, whose supports tie along the axes, to a couple of hundred vertices, at scales from 0.05 to 5. The table
    # also counts the vertices tied for a support, every one of which the cell of the direction lists.
    N_RANDOM_DIRECTIONS = 20000
    N_ARC_SAMPLES = 8
    MESHES_N_VERTS = (40, 200)
    SCALES = (0.05, 1.0, 5.0)

    meshes = [trimesh.creation.box(extents=(SCALES[0], 2.0 * SCALES[0], 3.0 * SCALES[0]))]
    for n_verts, scale in zip(MESHES_N_VERTS, SCALES[1:]):
        verts = np.random.normal(size=(n_verts, 3))
        verts /= np.linalg.norm(verts, axis=-1, keepdims=True)
        meshes.append(trimesh.Trimesh(scale * verts * np.random.uniform(0.5, 1.5, 3)).convex_hull)

    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(4.0, -15.0, 8.0),
            camera_lookat=(4.0, 0.0, 0.0),
        ),
        show_viewer=show_viewer,
    )
    entities = [
        scene.add_entity(
            morph=gs.morphs.MeshSet(
                files=(mesh,),
                pos=(4.0 * i_e, 0.0, 0.0),
            ),
            vis_mode="collision",
        )
        for i_e, mesh in enumerate(meshes)
    ]
    scene.build()

    @qd.kernel
    def query_supports(
        i_g: int,
        directions: qd.types.ndarray(),
        supports: qd.types.ndarray(),
        supports_count: qd.types.ndarray(),
        collider_info: array_class.ColliderInfo,
    ):
        for i_d in range(directions.shape[0]):
            direction = qd.Vector([directions[i_d, 0], directions[i_d, 1], directions[i_d, 2]], dt=gs.qd_float)
            support, _ = support_field._func_support_mesh(i_g, direction, collider_info)
            for k in qd.static(range(3)):
                supports[i_d, k] = support[k]
            supports_count[i_d] = support_field._func_count_supports_mesh(i_g, direction, collider_info)

    # Directions on the boundaries of the cells of both charts, the second chart seeing a direction (x, y, z) of the
    # mesh frame as (x, -z, y)
    support_res = qd_to_numpy(scene.rigid_solver.collider.collider_info.support_field.support_res).item()
    theta = np.random.randint(support_res, size=N_RANDOM_DIRECTIONS // 4) / support_res * 2.0 * np.pi
    phi = np.random.uniform(0.25 * np.pi, 0.75 * np.pi, N_RANDOM_DIRECTIONS // 4)
    directions_meridian = np.stack((np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)), axis=-1)
    theta = np.random.uniform(0.0, 2.0 * np.pi, N_RANDOM_DIRECTIONS // 4)
    phi = np.random.randint(support_res // 4, 3 * support_res // 4 + 1, N_RANDOM_DIRECTIONS // 4) / support_res * np.pi
    directions_parallel = np.stack((np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)), axis=-1)
    directions_boundary = np.concatenate((directions_meridian, directions_parallel))
    directions_boundary = np.concatenate((directions_boundary, directions_boundary[:, (0, 2, 1)] * (1.0, 1.0, -1.0)))

    for entity in entities:
        geom = entity.geoms[0]
        verts = geom.init_verts
        hull = ConvexHull(verts)
        faces_normal = hull.equations[:, :3]
        arcs_weight = np.random.uniform(0.0, 1.0, (len(faces_normal), N_ARC_SAMPLES, 1))
        directions_arc = (
            arcs_weight * faces_normal[:, None] + (1.0 - arcs_weight) * faces_normal[hull.neighbors[:, 0], None]
        ).reshape((-1, 3))
        directions = np.concatenate(
            (
                np.random.normal(size=(N_RANDOM_DIRECTIONS, 3)),
                directions_boundary,
                faces_normal,
                directions_arc,
                np.eye(3),
                -np.eye(3),
            )
        )
        directions = (directions / np.linalg.norm(directions, axis=-1, keepdims=True)).astype(gs.np_float)
        supports = np.zeros_like(directions)
        supports_count = np.zeros(len(directions), dtype=gs.np_int)
        query_supports(geom.idx, directions, supports, supports_count, scene.rigid_solver.collider.collider_info)
        assert_allclose((supports * directions).sum(axis=-1), (directions @ verts.T).max(axis=-1), tol=tol)

    # A direction in a coordinate plane ties exactly two corners of the box, along an arc between two face normals that
    # lies on the boundaries of the equatorial cells, and an axis of the box ties the four corners of a face.
    directions_sign = np.random.choice((-1.0, 1.0), (N_RANDOM_DIRECTIONS, 3))
    directions = directions_sign * np.random.uniform(0.2, 1.0, (N_RANDOM_DIRECTIONS, 3))
    directions[np.arange(N_RANDOM_DIRECTIONS), np.random.randint(3, size=N_RANDOM_DIRECTIONS)] = 0.0
    directions = np.concatenate((directions, np.eye(3), -np.eye(3)))
    directions = (directions / np.linalg.norm(directions, axis=-1, keepdims=True)).astype(gs.np_float)
    supports = np.zeros_like(directions)
    supports_count = np.zeros(len(directions), dtype=gs.np_int)
    query_supports(
        entities[0].geoms[0].idx, directions, supports, supports_count, scene.rigid_solver.collider.collider_info
    )
    assert_equal(supports_count, np.repeat((2, 4), (N_RANDOM_DIRECTIONS, 6)))

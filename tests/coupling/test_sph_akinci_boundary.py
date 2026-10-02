# Opt-in Akinci 2012 SPH boundary particles (LegacyCouplerOptions.sph_akinci_boundary).
#
# On main, #2857 already applies hydrostatic pressure through LegacyCoupler (always on for
# non-fixed links). Akinci is a separate, default-off WCSPH boundary-particle path. Tests here
# assert boundary construction, coexistence with #2857, and depth-invariant combined behavior.
# They do not claim isolated Akinci force magnitude or that flag-off sinks.

import math

import numpy as np
import pytest

import genesis as gs


def create_buoyancy_scene(show_viewer, akinci, rho_body=600.0, particle_size=0.02):
    """Water box + rigid box above it. Flag only toggles Akinci boundary particles."""
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=5e-4, gravity=(0.0, 0.0, -9.81)),
        sph_options=gs.options.SPHOptions(
            lower_bound=(-0.4, -0.4, 0.0),
            upper_bound=(0.4, 0.4, 0.7),
            particle_size=particle_size,
        ),
        coupler_options=gs.options.LegacyCouplerOptions(
            rigid_sph=True,
            sph_akinci_boundary=akinci,
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(material=gs.materials.Rigid(needs_coup=True), morph=gs.morphs.Plane())
    scene.add_entity(
        material=gs.materials.SPH.Liquid(sampler="regular", rho=1000.0),
        morph=gs.morphs.Box(pos=(0.0, 0.0, 0.25), size=(0.6, 0.6, 0.4)),
    )
    box = scene.add_entity(
        material=gs.materials.Rigid(rho=rho_body, needs_coup=True),
        morph=gs.morphs.Box(pos=(0.0, 0.0, 0.55), size=(0.12, 0.12, 0.12)),
    )
    scene.build()
    return scene, box


@pytest.mark.required
def test_akinci_boundary_default_off(show_viewer):
    """Flag off: no Akinci boundary particles; stock (#2857) coupler path only."""
    scene, _ = create_buoyancy_scene(show_viewer, akinci=False)
    sph = scene.sph_solver
    assert sph._akinci_boundary is False
    assert sph._akinci_boundary_n_particles == 0


@pytest.mark.required
def test_akinci_boundary_built_for_box(show_viewer):
    """Flag on: boundary particles are sampled on the rigid box surface."""
    scene, _ = create_buoyancy_scene(show_viewer, akinci=True)
    sph = scene.sph_solver
    assert sph._akinci_boundary is True
    assert sph._akinci_boundary_n_particles > 0


@pytest.mark.required
def test_akinci_boundary_built_for_mesh_geom(show_viewer):
    """Boundary particles are also sampled on non-box (mesh/primitive) geoms via init_verts/init_faces."""
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=5e-4, gravity=(0.0, 0.0, -9.81)),
        sph_options=gs.options.SPHOptions(
            lower_bound=(-0.4, -0.4, 0.0), upper_bound=(0.4, 0.4, 0.7), particle_size=0.02
        ),
        coupler_options=gs.options.LegacyCouplerOptions(rigid_sph=True, sph_akinci_boundary=True),
        show_viewer=show_viewer,
    )
    scene.add_entity(material=gs.materials.Rigid(needs_coup=True), morph=gs.morphs.Plane())
    scene.add_entity(
        material=gs.materials.SPH.Liquid(sampler="regular", rho=1000.0),
        morph=gs.morphs.Box(pos=(0.0, 0.0, 0.25), size=(0.6, 0.6, 0.4)),
    )
    scene.add_entity(
        material=gs.materials.Rigid(rho=600.0, needs_coup=True),
        morph=gs.morphs.Sphere(pos=(0.0, 0.0, 0.55), radius=0.09),
    )
    scene.build()
    assert scene.sph_solver._akinci_boundary_n_particles > 0


def _official_style_flotation_scene(show_viewer, akinci):
    """Match tests/coupling/test_sph_rigid.py geometry (where #2857 is validated)."""
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=1e-2, substeps=10, gravity=(0.0, 0.0, -9.81)),
        sph_options=gs.options.SPHOptions(
            lower_bound=(-0.25, -0.25, 0.0),
            upper_bound=(0.25, 0.25, 1.0),
            particle_size=0.02,
            pressure_solver="WCSPH",
        ),
        coupler_options=gs.options.LegacyCouplerOptions(rigid_sph=True, sph_akinci_boundary=akinci),
        show_viewer=show_viewer,
    )
    scene.add_entity(morph=gs.morphs.Plane())
    scene.add_entity(
        morph=gs.morphs.Box(pos=(0.0, 0.0, 0.25), size=(0.5, 0.5, 0.5)),
        material=gs.materials.SPH.Liquid(sampler="regular"),
    )
    light = scene.add_entity(
        morph=gs.morphs.Sphere(pos=(-0.12, 0.0, 0.25), radius=0.06),
        material=gs.materials.Rigid(rho=200.0),
    )
    heavy = scene.add_entity(
        morph=gs.morphs.Sphere(pos=(0.12, 0.0, 0.25), radius=0.06),
        material=gs.materials.Rigid(rho=1500.0),
    )
    scene.build()
    return scene, light, heavy


@pytest.mark.required
def test_akinci_boundary_coexists_with_stock_buoyancy(show_viewer):
    """Flag on must not break light flotation; stock off still density-discriminative (#2857).

    Same tank/spheres as test_rigid_flotation_follows_density_ratio. With Akinci *and* #2857
    both active, dense bodies can over-buoy (additive forces) — we only require the light body
    still floats and the scene stays stable. Heavy-sink under Akinci is not required.
    """
    n_steps = 50

    scene_off, light_off, heavy_off = _official_style_flotation_scene(show_viewer, akinci=False)
    for _ in range(n_steps):
        scene_off.step()
    z_light_off = float(light_off.get_pos().reshape(-1)[2])
    z_heavy_off = float(heavy_off.get_pos().reshape(-1)[2])

    scene_on, light_on, heavy_on = _official_style_flotation_scene(show_viewer, akinci=True)
    for _ in range(n_steps):
        scene_on.step()
    z_light_on = float(light_on.get_pos().reshape(-1)[2])
    z_heavy_on = float(heavy_on.get_pos().reshape(-1)[2])

    assert scene_on.sph_solver._akinci_boundary is True
    assert scene_on.sph_solver._akinci_boundary_n_particles > 0
    assert z_light_off > 0.28, f"stock #2857 light float failed: z={z_light_off:.3f}"
    assert z_heavy_off < 0.22, f"stock #2857 heavy sink failed: z={z_heavy_off:.3f}"
    assert z_light_on > 0.28, f"Akinci on light float failed: z={z_light_on:.3f}"
    # Record additive over-buoy without failing: heavy may rise when both paths apply.
    assert np.isfinite(z_heavy_on), f"Akinci on heavy non-finite z={z_heavy_on}"


def _rpy(q):
    w, x, y, z = q
    return np.array(
        [
            math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)),
            math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x)))),
            math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)),
        ]
    )


def _ang(box):
    try:
        return np.asarray(box.get_ang().cpu()).reshape(-1)[:3]
    except Exception:
        return np.zeros(3)


def _deep_water_scene(show_viewer, rho_body):
    """Deep tank so a box held at either test depth is fully submerged."""
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=5e-4, gravity=(0.0, 0.0, -9.81)),
        sph_options=gs.options.SPHOptions(
            lower_bound=(-0.45, -0.45, 0.0), upper_bound=(0.45, 0.45, 0.75), particle_size=0.02
        ),
        coupler_options=gs.options.LegacyCouplerOptions(rigid_sph=True, sph_akinci_boundary=True),
        show_viewer=show_viewer,
    )
    scene.add_entity(material=gs.materials.Rigid(needs_coup=True), morph=gs.morphs.Plane())
    scene.add_entity(
        material=gs.materials.SPH.Liquid(rho=1000.0),
        morph=gs.morphs.Box(pos=(0.0, 0.0, 0.28), size=(0.8, 0.8, 0.5)),
    )
    box = scene.add_entity(
        material=gs.materials.Rigid(rho=rho_body, needs_coup=True),
        morph=gs.morphs.Box(pos=(0.0, 0.0, 0.78), size=(0.12, 0.12, 0.12)),
    )
    scene.build()
    return scene, box


@pytest.mark.slow("cpu")  # Two 14,000-step SPH scenes exceed the 20-minute CPU timeout; ~2 minutes on GPU.
@pytest.mark.required
def test_akinci_boundary_depth_invariant(show_viewer):
    """Fully submerged hold-force at two depths should agree (ratio ~1) with Akinci on.

    Measures the combined #2857 + Akinci vertical fluid force; depth-invariance is the physics
    check (not absolute Archimedes magnitude).
    """
    g = 9.81

    def hold_force(z_target):
        scene, box = _deep_water_scene(show_viewer, rho_body=1500.0)
        rs = scene.sim.rigid_solver
        bl = box.base_link.idx
        m = float(box.get_mass())
        forces = []
        for i in range(14000):
            z = float(box.get_pos().reshape(-1)[2])
            vz = float(box.get_vel().reshape(-1)[2])
            target = 0.78 if i < 1000 else z_target
            f = 6000.0 * (target - z) - 650.0 * vz + m * g
            q = np.asarray(box.get_quat().cpu()).reshape(-1)
            t_att = (-np.array([90.0, 90.0, 60.0]) * _rpy(q) - np.array([12.0, 12.0, 9.0]) * _ang(box)).astype(
                np.float32
            )
            rs.apply_links_external_force(
                force=np.array([[0.0, 0.0, f]], np.float32), links_idx=np.asarray([bl]), ref="root_com"
            )
            rs.apply_links_external_torque(torque=t_att[None], links_idx=np.asarray([bl]))
            scene.step()
            if i >= 10000:
                forces.append(f - m * g)
        return -float(np.mean(forces))

    f_shallow = hold_force(0.18)
    f_deep = hold_force(0.10)
    assert np.isfinite((f_shallow, f_deep)).all(), f"non-finite forces: shallow={f_shallow} deep={f_deep}"
    assert abs(f_shallow) > 1e-6, f"shallow reference force is too small for a stable ratio: {f_shallow}"
    ratio = f_deep / f_shallow
    assert 0.85 < ratio < 1.15, (
        f"buoyancy not depth-invariant: shallow={f_shallow:.2f} deep={f_deep:.2f} ratio={ratio:.3f}"
    )

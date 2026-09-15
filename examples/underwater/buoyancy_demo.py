"""Density-discriminative buoyancy demo (stock #2857 + optional Akinci boundary particles).

Two rigid boxes of different density drop into one WCSPH tank. There is no analytic Archimedes
force in this script. With the Akinci path off, a light box (rho < rho_water) must float and a
heavy box must sink through the stock #2857 coupling.

On modern Genesis main, flotation already comes from official #2857 (LegacyCoupler plane-integral
pressure). Passing ``sph_akinci_boundary=True`` *adds* Akinci 2012 boundary particles on top —
useful for visual/RL demos and research comparison, not required for basic float/sink. Because
the paths are additive, the heavy body may over-buoy when Akinci is enabled.

Run (GPU):   python examples/underwater/buoyancy_demo.py
Compare on/off:   python examples/underwater/buoyancy_demo.py --compare

See UNDERWATER.md for the fidelity envelope and double-count notes.
"""

import sys

import numpy as np
import genesis as gs


def submerged_fraction(box, water):
    """Fraction of the box's height below the SPH waterline (0 = floating on top, 1 = fully under)."""
    zs = float(np.percentile(np.asarray(water.get_particles_pos().cpu()).reshape(-1, 3)[:, 2], 92))
    ab = np.asarray(box.get_AABB().cpu()).reshape(-1)
    bz, tz = float(ab[2]), float(ab[5])
    return float(np.clip((zs - bz) / max(tz - bz, 1e-3), 0.0, 1.0)), zs


def run(akinci: bool, n_steps: int = 6000):
    scene = gs.Scene(
        show_viewer=False,
        sim_options=gs.options.SimOptions(dt=5e-4, gravity=(0.0, 0.0, -9.81)),
        sph_options=gs.options.SPHOptions(
            lower_bound=(-0.62, -0.32, 0.0), upper_bound=(0.62, 0.32, 0.85), particle_size=0.02
        ),
        coupler_options=gs.options.LegacyCouplerOptions(rigid_sph=True, sph_akinci_boundary=akinci),
    )
    scene.add_entity(material=gs.materials.Rigid(needs_coup=True), morph=gs.morphs.Plane())
    water = scene.add_entity(
        material=gs.materials.SPH.Liquid(rho=1000.0),
        morph=gs.morphs.Box(pos=(0.0, 0.0, 0.28), size=(1.15, 0.55, 0.5)),
    )
    L = 0.15
    light = scene.add_entity(  # rho 600 < 1000 -> should FLOAT
        material=gs.materials.Rigid(rho=600.0, needs_coup=True),
        morph=gs.morphs.Box(pos=(-0.30, 0.0, 0.66), size=(L, L, L)),
    )
    heavy = scene.add_entity(  # rho 1400 > 1000 -> should SINK
        material=gs.materials.Rigid(rho=1400.0, needs_coup=True),
        morph=gs.morphs.Box(pos=(0.30, 0.0, 0.66), size=(L, L, L)),
    )
    scene.build()
    print(
        f"[buoy] akinci={akinci}  boundary_particles={scene.sph_solver._akinci_boundary_n_particles}  "
        f"light=rho600  heavy=rho1400  (NO analytic buoyancy)",
        flush=True,
    )
    for i in range(n_steps):
        scene.step()
        if i % 1500 == 0:
            sl, _ = submerged_fraction(light, water)
            sh, _ = submerged_fraction(heavy, water)
            print(f"[buoy] step {i:5d}: light sub={sl:.2f}  heavy sub={sh:.2f}", flush=True)
    sl, wl = submerged_fraction(light, water)
    sh, _ = submerged_fraction(heavy, water)
    lz = float(np.asarray(light.get_pos().cpu()).reshape(-1)[2])
    hz = float(np.asarray(heavy.get_pos().cpu()).reshape(-1)[2])
    light_floats = sl < 0.9
    heavy_sinks = sh > 0.9
    print(
        f"[buoy] akinci={akinci} FINAL  light z={lz:.3f} sub={sl:.2f} "
        f"({'FLOATS' if light_floats else 'sank'})  |  heavy z={hz:.3f} sub={sh:.2f} "
        f"({'sinks' if heavy_sinks else 'floats'})  waterline={wl:.3f}",
        flush=True,
    )
    stable = np.isfinite((sl, sh, lz, hz, wl)).all()
    return light_floats, heavy_sinks, stable


def main():
    compare = "--compare" in sys.argv
    gs.init(backend=gs.cuda, precision="32", logging_level="warning")
    if compare:
        print("=== FLAG OFF (stock #2857 plane-integral pressure) ===")
        off_light_floats, off_heavy_sinks, off_stable = run(akinci=False)
        print("\n=== FLAG ON (Akinci-2012 boundary buoyancy) ===")
        on_light_floats, on_heavy_sinks, on_stable = run(akinci=True)
        ok = off_light_floats and off_heavy_sinks and off_stable and on_light_floats and on_stable
        print(
            f"\n[buoy] COMPARISON RESULT: {'PASS' if ok else 'FAIL'} — stock path remains density-discriminative; "
            f"Akinci path is stable and keeps the light body afloat. "
            f"Akinci heavy body {'sank' if on_heavy_sinks else 'over-buoyed (known additive-path limitation)'}."
        )
        sys.exit(0 if ok else 1)
    else:
        light_floats, heavy_sinks, stable = run(akinci=True)
        ok = light_floats and stable
        print(
            f"\n[buoy] RESULT: {'PASS' if ok else 'FAIL'} — Akinci path is stable and the light body floats; "
            f"heavy body {'sank' if heavy_sinks else 'over-buoyed (known additive-path limitation)'}."
        )
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

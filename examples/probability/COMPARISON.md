# Matched real-world / Genesis coin and dice comparisons

This pipeline compares tracked physical tosses with Genesis replays of their
measured release states. **No suitable measured trajectories are currently
available in this project. Each supplied 100-pair coin/die preview is synthetic on both
sides, including the orange series labeled “Genesis (placeholder)”.** The demo
uses analytic drawing fixtures, not Genesis. Its numbers are not research results.
Both charts include position/orientation error, rebound counts/heights, settling
times and final outcome frequencies, with consistent blue/orange colors and 95%
intervals. The six numbered die faces and unresolved rolls use the same denominator.

`compare_real.py` is independent of the live Probability Lab. It does not alter
the lab's sampling, contact model, interface, or existing exports. Endpoint-only
JSON exports cannot recover a trajectory, bounce history, or real-world reference.

## Where to get real data

Two verified public starting points have useful but incomplete information:

- [Visual Coin-Tracking (CTR)](https://cmp.felk.cvut.cz/coin-tracking/) provides
  17 video sequences with segmentation masks, bounding boxes and visible-side
  labels. Its published annotations do not supply calibrated metric 3D coin
  trajectories, quaternions, coin inertias, and a matched floor-contact experiment.
  It is useful for developing image tracking, not a ready-made source for these
  six physical-validation panels. Check its CC BY-NC 4.0 annotation license.
- [Bartoš et al., 350,757 flips](https://arxiv.org/abs/2310.04153) links its
  [data and materials on OSF](https://osf.io/pxu6r/). These data support outcome
  analysis. The paper describes a caught-coin protocol and says the audit video
  quality was insufficient for additional dynamical insight. These are not
  matched 3D floor-bounce trajectories. The OSF file listing could not be fetched
  during this work; no files from that repository were imported.

The earlier [Diaconis–Holmes–Montgomery study](https://people.ucsc.edu/~rmont/papers/headswithO.pdf)
also studies coins caught in the hand. Do not treat its approximately 51%
same-side prediction as a floor-bounce measurement or an orientation-error trace.
This search did not locate a ready-to-use dataset meeting all the requested
conditions; that is not a claim that none exists.

For a matched study, record your own marked coin on a measured surface:

1. Measure mass, radius, thickness and the floor plane; identify the actual
   material and finish. Use a consistent coin, patch of floor and release protocol.
   A uniform cylinder is the current inertia assumption. Embossing, mass offsets,
   edge rounding, deformation and surface texture are not modeled.
2. Calibrate camera intrinsics, distortion, metric scale and world frame. Use
   synchronized views or a validated pose estimator for a marked rigid object.
   Two-dimensional center tracking alone cannot recover arbitrary 3D motion.
   An unmarked circular silhouette does not determine rotation about the coin's
   normal. Heads/tails and an asymmetric marker must identify body +Z and body +X.
3. Choose frame rate and exposure using the observed angular and impact speeds.
   500–1,000 fps is an example acquisition target, not a guarantee of adequate
   sampling. Avoid motion blur and ambiguous between-frame rotations. Validate
   pose reconstruction against known static poses and distances first.
4. Record release through every bounce and at least 0.25 s of confirmed rest,
   or a predeclared common timeout. Record occlusion, failed tracking, exclusions,
   and operator/session IDs. Keep all physical outcomes in a separate acquisition
   log, including those whose trajectories cannot be reconstructed.
5. Export COM position, full orientation and estimated velocities. Document the
   velocity fit and smoothing, especially at impact. Estimate release velocity
   from pre-impact flight without using final outcomes. Transform all measurements
   into the convention below before running Genesis.
6. Use separate calibration tosses/drops to choose contact settings, then freeze
   those settings before evaluating held-out tosses. The script performs no
   parameter fitting and has no automatic calibration/test split. Displaying a
   favorable setting chosen on evaluation data would overstate accuracy.

The script starts with **tracked numerical data**. It does not automate camera
calibration, video tracking, marker detection, synchronization, or noise filtering.

## Input contract

One CSV row per timestamp per independent toss:

```text
trial_id,t_s,x_m,y_m,z_m,qw,qx,qy,qz,vx_m_s,vy_m_s,vz_m_s,wx_rad_s,wy_rad_s,wz_rad_s
```

- Units are seconds, metres, m/s and rad/s. `x,y,z` locate the center of mass.
- World Z points up; the horizontal floor is `z=0`; gravity is `(0,0,-9.81)` m/s².
- Quaternion order is **wxyz**, active body-to-world rotation. Body +Z is heads.
- Both recorded linear and angular velocity are **world-frame** vectors.
  The adapter converts angular velocity to body coordinates for Genesis free-joint
  DOFs and converts the recorded output back through the public world-frame getter.
- Each toss begins at its first tracked free-flight state, labeled `t_s=0`.
  No time shift is optimized to make the simulation match better.
- Require finite data, strictly increasing timestamps, >=3 frames, and quaternion
  norm within 1% of one (renormalized). Do not concatenate repeated exports as
  independent tosses. Include full time series, not just release/final rows.
- The default maximum tracking gap is 20 ms. This is an input guard, not a claim
  that 20 ms resolves your impacts. Set a stricter limit based on the experiment.

`demo` writes a complete example CSV and `experiment.json`. Replace the example
with measured values and set `data_kind` to `measured` only for actual measurements.
The manifest requires `source`, `tracking_method`, `floor_material`, `coin` or `die`, and
`contact_profiles`. A material name is metadata; matching the contact response
still requires calibration. All six candidate profiles in the generated template
are **untuned placeholders**: sliding friction 0.3/0.6 crossed with `dampratio`
0.5/1/2, time constant 0.02 s, torsional friction 0.005 m, rolling friction 0.0001 m.

## Run on this repository's existing environment

NumPy, SciPy, matplotlib and Genesis are already installed in `.venv` here.
No dependencies were added. From the repository root in PowerShell:

```powershell
$env:PYTHONUTF8='1'
$env:GS_CACHE_FILE_PATH="$PWD\.cache\genesis"
$env:QD_OFFLINE_CACHE_FILE_PATH="$PWD\.cache\quadrants"
$env:NUMBA_CACHE_DIR="$PWD\.cache\numba"
$env:MPLCONFIGDIR="$PWD\.cache\matplotlib"

# Fast illustrative charts, SVGs, per-trial metrics and input templates (100 pairs each).
.venv\Scripts\python.exe -m examples.probability.compare_real demo --kind coin
.venv\Scripts\python.exe -m examples.probability.compare_real demo --kind die

# After replacing the inputs with measured tracking and physical parameters:
.venv\Scripts\python.exe -m examples.probability.compare_real simulate `
  --reference path/to/measured-tracks.csv --manifest path/to/experiment.json `
  --out out/probability/matched-replays --backend gpu --seeds 11 22 33

# Plot a preselected setting; each setting and replay seed remains separate.
.venv\Scripts\python.exe -m examples.probability.compare_real plot `
  --reference path/to/measured-tracks.csv --manifest path/to/experiment.json `
  --run out/probability/matched-replays/mu0p6_d1-seed11.npz `
  --out out/probability/matched-replays/comparison.png
```

The coin preview is saved under `out/probability/real-comparison-demo`; the dice
preview is under `out/probability/real-comparison-die-demo`. Each directory has
`ILLUSTRATIVE-comparison.png`, its SVG/JSON companions, `reference.csv`, and
`experiment.json`. Use `--out` to choose another directory. The same `simulate`,
`plot`, and `drop-check` commands work for either object; they select geometry from
the manifest. Old coin manifests without `kind` still select the coin.

## Dice geometry and face conventions

A dice manifest sets `"kind": "die"` and supplies, for example:

```json
"die": {"mass_kg": 0.08, "edge_m": 0.08}
```

These **80 g, 80 mm values are placeholders inherited from the Probability Lab**,
not measurements of a typical gaming die. Replace them with the measured mass
and edge length. The model is a homogeneous sharp-edged cube, with principal
moments `I_x = I_y = I_z = mass * edge_length^2 / 6`. Engraved pips, bevels,
rounded edges, cavities, and unequal side lengths are absent. Such features can
affect real rolls and require a separately validated geometry/inertia model.

Body-face labels match the current lab:

| Face | Outward body normal |
| --- | --- |
| 1 | +Z |
| 2 | +X |
| 3 | +Y |
| 4 | -Y |
| 5 | -X |
| 6 | -Z |

Calibrate the tracked die frame to this mapping. Full orientation error compares
the **labeled** body, without quotienting out cube symmetries. Two geometrically
identical poses with different numbered faces up must not receive zero error.

For rotation matrix `R` and half-edge `a`, cube clearance is
`z_COM - a*(abs(R_zx) + abs(R_zy) + abs(R_zz))`. This accounts for face, edge and
corner orientations when detecting flight/contact. A die outcome requires the
most upward face normal to have dot product >=0.985 after confirmed rest.
Moving rolls, timeouts, and ambiguous edge/corner rests are `unresolved`.
Confirmed settling time can exist for an unresolved tilted pose. All seven
categories remain in the denominator; there is no imposed 1/6 probability.

The dice illustration starts with Haar-uniform synthetic orientations and uses
an artificial rotation/relaxation trajectory. Its frequencies reflect that drawing
fixture. They do not establish fairness of real dice or of Genesis.

`simulate` replays every CSV trial in an independent batched world for each
profile and seed. Six profiles × three seeds × 100 measurements means 1,800
simulated replays, with **100 independent physical tosses**, not 1,800 real tosses.
Use `--backend cpu` if GPU is unavailable. First-time compilation may take time.
There is no headless viewer or rendering dependency in the replay.

Each run stores trajectories in NPZ and configuration/provenance as JSON inside
that bundle. `runs.json` lists per-profile/seed outcomes. Each plotted PNG has a
matching vector SVG and JSON containing paired per-trial errors, bounce heights,
settling times, and outcomes. Input and script SHA-256 hashes and the Git revision
are recorded. Hash checks prevent plotting a replay against changed inputs.

## Genesis contact assumptions and restitution

This uses the local rigid solver's compliant contact model, Newton constraints,
pyramidal friction cone, approximate-implicitfast integration, 50 solver iterations,
and solver tolerance `1e-8`. Gravity and geometry are explicit. Both floor and coin
receive identical friction/contact parameters. Sliding friction is mixed by the
engine's maximum rule. The floor is fixed and flat. Air drag and aerodynamic torque
are absent. No learned model or PPO is involved.

Default integration for both objects is **1 ms with two substeps** (0.5 ms per solver substep);
trajectories are saved every 5 ms. `--dt`, `--substeps`, and `--sample-dt` are
configurable. Output spacing must be an integer multiple of the timestep. Repeat
with a halved timestep and denser recording before interpreting impact errors.

**`coup_restitution` is not a rigid-coin/rigid-floor restitution knob.** The
[Genesis maintainers describe the rigid contact parameters](https://github.com/Genesis-Embodied-AI/genesis-world/discussions/1090)
as a `solref`/`solimp` model. This implementation sets both geometries' public
`set_sol_params` to `[timeconst, dampratio, 0.9, 0.95, 0.001, 0.5, 2]`. It checks
the time-constant lower bound and records the parameters returned by Genesis.
It never labels `dampratio` as a coefficient of restitution.

To connect those controls to measurable bounce behavior, run independent
horizontal, non-spinning drops:

```powershell
.venv\Scripts\python.exe -m examples.probability.compare_real drop-check `
  --manifest path/to/experiment.json --out out/probability/rebound-check `
  --backend gpu --heights 0.1 0.2
```

The output reports a height-ratio estimate
`e_effective = sqrt(rebound_clearance / drop_clearance)` for the first detected
completed rebound. This estimates a vertical effective coefficient for that
drop protocol, ignoring drag. It is **not** a general coefficient for spinning
oblique impacts. No detected complete rebound yields `null`, not a fabricated
zero restitution. Detection thresholds and finite timestep limit the estimate.
Compare this response against independently measured real flat drops, select
candidate contact profiles, and retain the evidence. **No material parameters
have been tuned to real data in the supplied example.**

Changing the RNG seed does not change a deterministic rigid-body trajectory
when the initial state and parameters are identical. Replays explicitly request
deterministic algorithms and add no launch perturbations. Seeds are reproducibility
checks. If you later model camera/release uncertainty, specify and validate that
error model separately; these runs do not invent sensor noise. Bitwise agreement
across devices or Genesis versions is not promised.

## Metrics and uncertainty

For each paired toss at shared time `t`, position error is
`1000 * ||p_sim(t) - p_real(t)||_2` mm. Orientation error is the full SO(3) geodesic
`(180/pi) * 2*acos(min(1, abs(dot(q_sim, q_real))))` degrees for unit quaternions.
The absolute dot makes `q` and `-q` equivalent. Full orientation includes spin
about the face normal, so do not use ambiguous silhouette-only orientations.

Real positions/velocities are linearly interpolated; orientations use SLERP.
There is no extrapolation or fitted time warp. All metrics use the same saved
sampling grid and the common horizon bounded by the shortest recording. Longer
records are truncated; increasing that horizon requires tracking every trial
longer. The chart is a mean over paired tosses, not an individual flight trajectory.

For a cylinder with radius `r`, half-thickness `h/2` and world-normal Z component
`n_z`, its lowest-point clearance is
`z_COM - r*sqrt(1-n_z^2) - (h/2)*abs(n_z)`.
This geometric definition is used for both measured and simulated tracks, rather
than giving simulation privileged access to its contact-force flags.

- **Bounce count** counts complete contact → flight → contact episodes after
  the initial flight, with clearance >2 mm and flight duration >=10 ms. Contact
  tolerance is 1 mm. Open rebounds at the cutoff are recorded separately, so the
  plotted count is completed rebounds within the window, not eventual bounces.
- **Bounce heights** are the maximum **COM height above the floor** in each
  completed rebound. A spinning coin's COM apex need not decrease monotonically.
  The plot shows the first five ordinal rebounds and the sample size per bar;
  absent bounces are not assigned height zero. JSON retains every rebound and
  maximum lower-surface clearance as well.
- **Settling** starts when speed is <0.015 m/s and spin <0.15 rad/s near contact,
  sustained until the observation ends with at least 0.25 s confirmation.
  The chart shows the fraction settled by time `t`, keeping unsettled trials
  in the denominator. JSON records onset time or `null` for censoring.
- **Heads/tails** require confirmed rest and `n_z >=0.985` / `n_z <=-0.985`.
  **Edge** requires confirmed rest and `abs(n_z)<=0.1`. Moving, timed-out and
  intermediate-tilt cases are **unresolved**, not edge. All four categories use
  the full paired-trial denominator.

The **proposed** detection thresholds above are configurable in `analysis`.
They differ from the GUI's unresolved-only treatment of edge rest. They belong
to this separate comparison script and must be checked against labeled recordings
and tracking noise before the study; they are not a calibrated measurement method.

Error curves and mean bounce statistics use 2,000 percentile bootstrap draws
over toss IDs with fixed bootstrap seed 817. Fractions use marginal 95% Wilson
intervals, including zero counts. Error bands are **pointwise**, not simultaneous.
A one-toss bootstrap is omitted rather than reporting false certainty. Ordinal
bounce samples can be very small; intervals then provide little information.
These intervals assume independent representative tosses. For repeated operators
or sessions with dependence, use a justified cluster analysis before inference.
They do not include camera calibration error, velocity-fit uncertainty, contact
parameter uncertainty, measurement selection bias, or unseen solver discrepancy.

## Validation

```powershell
.venv\Scripts\python.exe -m pytest tests/core/test_probability_comparison.py `
  -o addopts='' -m 'not examples and not benchmarks' -n 1 --backend cpu `
  --basetemp=out/probability/pytest-comparison --timeout=120 -q

# Actual-Genesis checks of both geometries, release frames, contacts and seeds:
.venv\Scripts\python.exe -m pytest tests/core/test_probability_comparison_physics.py `
  -o addopts='' -m 'not examples and not benchmarks' -n 1 --backend gpu `
  --basetemp=out/probability/pytest-comparison-physics --timeout=180 -q
```

Tests cover analytic rotation errors, quaternion sign invariance, SLERP, cylinder
support geometry, invalid timestamps, finite input validation, known bounce
counts, incomplete rebounds, airborne-apex rejection, moving edges, late motion,
settled heads/tails/edge, interval boundaries and mass/inertia construction.
Additional die cases cover all six faces, cube support at face/edge/corner,
ambiguous rests, inertias, and manifest selection. These verify the implementation;
physical accuracy still requires measured data.

Local validation on 2026-11-11: **26 mathematical/input tests and four actual-GPU
tests passed**. The GPU tests checked both geometries, non-aligned release frames,
ballistic flight, mass, applied contact parameters, all two coin/six die face
landings and repeated seeds. Both 100-pair synthetic demos rendered and were
visually inspected. Earlier coin CLI runs also completed for two synthetic
releases, two contact profiles and two seeds, plus two profiles at two flat-drop
heights. Python compilation passed. Ruff was unavailable; the upstream full test
suite was not run. These checks do not validate physical fidelity or camera-data
processing.

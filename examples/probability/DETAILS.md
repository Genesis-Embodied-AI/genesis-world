# Genesis Probability Lab: full details

A local desktop experiment with real Genesis rigid-body simulation on the left,
and physical-state forecasts and observed outcome frequencies on the right.
Neither the coin nor the die uses a random choice of its final face. The engine
integrates the launch state and ground contacts; the upper face is measured only
after the body rests.

## Open and use

From the repository root, double-click **Run-Probability-Lab.cmd**. The launcher
uses this project's existing `.venv`, CUDA, and local compilation caches. The
first use of an object or batch size may take a minute to compile. The window
shows progress. Close the window, or press Alt+F4, to stop the worker and save
completed trials. No separate Genesis viewer window or screen capture is used.

1. Open **Experiment settings / reset**, select Coin or Die and press
   **Apply / reset experiment**. The previous
   experiment is exported before resetting; different configurations are never
   pooled in one frequency table.
2. Press **Flip coin** or **Roll die**. Pause/resume, step one physical timestep,
   and zoom in/out are available. A new roll abandons any unfinished visible roll;
   abandoned rolls do not enter the statistics.
3. Leave **Auto forecasts** enabled to evaluate the physical state near simulated
   times 0, 0.30 and 0.65 seconds. The animation continues while the worker
   calculates forecasts. Each result is plotted at the time its state was
   observed. **Forecast now** also works during an active roll, including while
   paused. Starting a new toss or batch clears unfinished forecasts.
   Forecasts retain their observation timestamps while the visible simulation
   continues; there is no automatic checkpoint pause or display acknowledgement.
   Experiment/roll identifiers reject late messages from older runs.
4. Run a batch (128 trials by default) to grow the observed frequency table.
   **Cancel calculation** retains completed chunks and discards the current
   incomplete chunk. Compilation itself cannot be cancelled except by closing.
5. Change launch height, spin, sliding friction, seed, or the face-up launch
   protocol, then **Apply / reset experiment** to compare a new experiment.
6. Export JSON and CSV, or close the window to save automatically. Files go to
   `out/probability/`; `launcher.log` contains diagnostics.

The graphical application uses the already installed `imgui-bundle` (including
Hello ImGui, ImPlot and ImmVision), NumPy, SciPy, Pillow, PyTorch and Genesis.
No new dependencies were introduced. It was developed against Genesis 1.4.6,
Python 3.12 and the installed CUDA PyTorch build. CPU mode is available by omitting
`-g`; its execution and rendering speed can differ substantially.

```powershell
.venv\Scripts\python.exe -m examples.probability.app -g
.venv\Scripts\python.exe -m examples.probability.app -g --kind die --batch 256
.venv\Scripts\python.exe -m examples.probability.app -g --kind coin --batch 256 --dt 0.005
```

For direct commands, use the cache environment variables in the launcher on a
restricted Windows environment. They avoid writing caches outside the project.

## Live vector layers and larger graphs

The **Live vectors / legend** checkboxes independently show the floor normals,
starting-pose ghost and axes, initial/current linear and angular velocity,
contact points, normal/friction forces, and centre-of-mass (COM) trail. Pause and
step are useful for inspecting impacts. Changing a layer or display scale also
updates a paused frame. Overlays are render-only debug geometry and cannot
participate in collisions or change inertia.

All arrows use world coordinates. Velocity arrows start at the measured COM;
initial arrows and the translucent gold ghost remain at the launch state. RGB
axes mean X/Y/Z, and the initial quaternion readout is wxyz. The floor is the
actual flat collision plane, with 0.10 m checker squares as a color texture;
the checks do not create bumps. Pale blue unit normals are drawn 0.08 m long
and follow the plane's normal and orientation.

| Layer | Color | Default display scale |
| --- | --- | --- |
| Current linear velocity | Cyan | 0.12 m per m/s |
| Current angular velocity | Magenta | 0.012 m per rad/s |
| Initial linear/angular velocity | Blue/violet | Same respective scales |
| Contact points | Yellow | Point markers |
| Contact normal force | Blue | 1 m per N |
| Tangential friction force | Green when sticking; orange when slipping | 1 m per N |
| Starting pose / COM trail | Gold | Actual pose / measured trajectory |

**Adjust vector scales** changes only arrow lengths. Arrows longer than 0.45 m
are capped, with a warning in the detailed readout; numeric magnitudes remain
uncapped. Zero vectors have no direction and are not drawn. Open **Force
magnitudes, floor and starting-pose details** for velocity components, initial
pose, friction coefficients and a per-contact table with force magnitudes in N,
positions in m, and slip speed in m/s.

For each rendered frame, public Genesis getters supply COM velocities, world
angular velocity and actual solver contact forces on the coin/die. With contact
normal `n`, the displayed decomposition is `Fn = (F dot n) n`, `Ft = F - Fn`.
Friction arrows use this measured `Ft`, not an assumed `mu * |Fn|`. Contact-point
velocity is `v_COM + omega cross (point - COM)`, relative to the floor; its
tangential component determines the color. Tangential speed above 0.005 m/s is
labelled slipping. This is a display diagnostic, not a native solver mode flag.
Unloaded contacts are gray-blue. Forces and points describe the latest solver
substep, while velocities describe the current frame state, up to one 0.005 s
substep later at the default timestep. Rolling/torsional contact torques are not
drawn by these translational force arrows.

This Genesis contact model has **one shared sliding friction coefficient**,
not independently configurable static and kinetic coefficients. Both are shown
as shared `mu`; the contact coefficient follows the engine's
`max(floor_mu, object_mu, 0.01)` rule. The existing pyramidal friction-cone
solver and physical parameters are unchanged.

The probability panel now uses 44% of the window width. Charts have thicker
lines, marked forecast samples and a minimum height of 360 pixels, increasing
with a taller window. Use **Minimum chart height** or **Enlarge graphs** for
larger plots. The enlarged window is resizable; turn off **Auto-fit axes** to
pan and zoom either plot. The forecast graph fits the available observation
times and includes unresolved futures, so probability mass remains visible
when a simulated coin stays on its edge or exceeds the time limit.

## What 50/50 and 1/6 mean

These are **symmetry-based reference hypotheses**, not laws imposed by Genesis.
Under symmetric objects and a launch distribution that treats the possible
faces symmetrically, equal face probabilities are the natural expectation.
The actual result also depends on the contact model, launch protocol, numerical
integration and the definition of a completed roll. A finite sample does not
need to be exactly balanced.

The default launch uses Haar-uniform orientations (SciPy `Rotation.random`), an
isotropic spin direction with fixed magnitude, horizontal Gaussian velocity,
and a fixed upward velocity. Uniform Euler angles would not give this orientation
distribution. Each trial has its own recorded RNG seed; changing GPU chunk size
does not change its launch state. The optional face-up protocol changes the
experiment and does not have the same symmetry justification.

### Physical model and conventions

| Quantity | Definition |
| --- | --- |
| World coordinates | metres, Z up; gravity `(0, 0, -9.81)` m/s^2 |
| Coin | Uniform cylinder, radius 45 mm, thickness 8 mm, mass 20 g |
| Die | Uniform cube, edge 80 mm, mass 80 g |
| Labels | Coin: H is local +Z, T is -Z. Die: 1 +Z, 2 +X, 3 +Y, 4 -Y, 5 -X, 6 -Z |
| Quaternion | wxyz, body frame to world frame |
| Six free-joint velocities | World linear velocity (m/s), body-local angular velocity (rad/s) |
| Default launch | Height 0.45 m, upward speed 0.8 m/s; horizontal speed SD 0.06 m/s per axis; spin 18 rad/s |
| Integration | 0.01 s timestep, two substeps; maximum roll duration 6 s |
| Contact | Sliding friction 0.6; torsional friction 0.005 m, rolling friction 0.0001 m, Genesis rigid contact solver |
| Rest | Linear speed <0.015 m/s and spin <0.15 rad/s continuously for 0.25 s; centre height <0.065 m |
| Valid face | Maximum face-normal dot product with world up >=0.985, after rest |

Mass and moments of inertia are explicitly authored. Coin letters, colors, and
die pips have zero mass and no collision geometry. The floor checkerboard is
visual only. Camera tracking does not apply physical forces.

The coin is an idealized relatively thick token; it can roll on its edge for the
full time limit. Edge rest and timeouts are reported as **unresolved**, never
randomly assigned a face. Counts and Wilson intervals condition on valid settled
faces; the unresolved count and the total number of trials remain visible.
Consequently, the displayed 50/50 comparison is among valid two-face outcomes,
not a claim that all simulated launches yield heads or tails within six seconds.
An eight-millimetre cylinder is not a calibrated model of a particular real coin.

### Live forecasts

For each observation, generate 64 perturbed physical states and run each forward
in a separate batched Genesis world with identical geometry and contact settings.
The forecast horizon is the time remaining until the visible roll's six-second
deadline. For outcome k,

`estimated P(k | observed state, error model) = number of futures ending at k / 64`.

Unresolved futures use the same denominator. The face probabilities plus the
unresolved probability sum to one. They are **not** calculated from historical
trial frequencies. The observation-error model uses independent zero-mean
Gaussian deviations: position SD 0.5 mm per axis, rotation-vector SD 0.02 rad per
axis, linear speed SD 0.03 m/s per axis and local spin SD 0.3 rad/s per axis.
Orientation perturbations compose in the body frame.

These are conditional scenario estimates under assumed sensor errors, not a
learned or calibrated Bayesian posterior. New observations can change the
forecast; simply letting time pass with unchanged information is not evidence
that the probability of a fixed final event changed. Finite Monte Carlo samples
also introduce fluctuation (near 50%, 64 samples have about 6.25 percentage points
of standard error). Late near-contact perturbations may create small overlaps
that the contact solver resolves. The shadow worlds restart contact history and
the rest-confirmation timer; warm-start/contact-memory effects are not reproduced.

### Statistics and reproducibility

The dashboard displays marginal 95% Wilson intervals and an exploratory
chi-square p-value only after the expected count is at least five per face.
Intervals are not simultaneous across faces and do not retain 95% coverage
under arbitrary repeated stopping. A large p-value is not proof of fairness;
a small one warrants checking model assumptions and numerical effects first.

JSON exports include configuration, all initial/final trial states, outcomes,
launch seeds, observation states, forecast seeds and counts, library versions,
Genesis revision and application source hashes. CSV exports summarize trials.
The launch RNG and forecast RNG are separate, so requesting a forecast does not
change later trial launches. GPU floating-point/contact calculations can still
vary across devices, versions or solver settings; bitwise replay is not promised.

## Validation

The tests exercise known face orientations, edge rejection, stable launch seeds,
zero-error forecasts, unresolved accounting, known Wilson interval values,
massless/non-colliding markings and physical drops of both coin faces and all six
die faces. The GPU contact tests use 80 steps, checking rest, height, finite
velocity, mass and face identity. They do not assert that a random finite batch
must have equal counts.
Vector tests check force decomposition on a tilted normal, rolling contact
without slip, arrow scaling without changing measurements, world-frame angular
velocity, frozen initial-state readouts and contact-force sums against the
independent net-contact getter and the object's weight.

```powershell
$env:PYTHONUTF8='1'
$env:GS_CACHE_FILE_PATH="$PWD\.cache\genesis"
$env:QD_OFFLINE_CACHE_FILE_PATH="$PWD\.cache\quadrants"
$env:NUMBA_CACHE_DIR="$PWD\.cache\numba"
.venv\Scripts\python.exe -m pytest tests/core/test_probability_lab.py -o addopts='' -m 'not examples and not benchmarks' --basetemp=out/probability/pytest-run -n 1 --backend gpu --timeout=180 -q
.venv\Scripts\python.exe -m examples.probability.app -g --kind coin --smoke
.venv\Scripts\python.exe -m examples.probability.app -g --kind die --smoke
```

`--smoke` runs the actual interface, one visible throw with automatic and manual
forecasts, pause/resume and a checked single step, a batch of 64 trials, an export,
then closes and saves the application's own framebuffer
as a dashboard PNG. It does not record the desktop. The results are experiment
artifacts, not evidence that this model reproduces real coin or die frequencies.
Experiment controls are disabled during `--smoke` so clicks cannot change the
automated sequence. It also verifies that simulation time advances while a
forecast is being calculated. Use headless `--batch` for a fixed-size experiment.

### Verified locally on 2026-11-11

- The targeted pytest suite passed: **19 cases**, including both GPU contact tests,
  world-frame vector diagnostics and rejection of stale/out-of-order frames.
- A direct worker check confirmed that simulation time advanced to 0.98 s while
  computing the forecast from 0.00 s, without a display acknowledgement.
- Both GUI smoke runs completed, exported their data, produced inspected
  framebuffer images and exited their workers cleanly (65 trials and four
  forecasts each). Additional runs exercised the enlarged graphs window for
  both objects, including the coin's unresolved-probability curve.
- A rendered sliding-coin probe showed opposing tangential force during slip
  and about 0.1962 N total upward force at rest, matching the 20 g coin's weight.
  Turning all overlay layers off changed the image without changing the
  physical position, orientation or velocity. The starting-pose ghost, contact
  arrows and floor normals were inspected in camera renders.
- A separate worker integration check passed coin -> die -> coin reconfiguration,
  a 17-trial batch (including GPU chunk padding), cancellation, face-up settings
  and shutdown with a final frame pending.
- Python compilation and whitespace checks passed. Ruff is not installed, so
  its lint check was not run. The full upstream Genesis suite was not run.

Independent headless batches used seed 2026, the default configuration and CUDA
on an RTX 5090:

| Experiment | Trials | Physical outcome counts | Unresolved |
| --- | ---: | --- | ---: |
| Coin | 256 | Heads 82, tails 76 | 98 |
| Die | 256 | 1: 48, 2: 52, 3: 40, 4: 37, 5: 33, 6: 46 | 0 |

Among the 158 valid coin outcomes, heads occurred 51.9% of the time. The
unresolved fraction was 38.3%, so this is **not** a 256-flip demonstration of
unconditional 50/50 behavior. The exploratory chi-square p-values were 0.633
(valid coin faces) and 0.299 (die). These small experiments did not establish a
departure from equal valid-face probabilities; they do not prove fairness or
real-world accuracy. The relatively thick coin and six-second horizon need
particular attention before any such claim.

The raw batch reports are
`out/probability/coin-20261111T062350535885Z.json` and
`out/probability/die-20261111T062459906447Z.json`, with matching CSV files.

Before claiming physical accuracy, measure real launch/contact parameters,
compare with physical trials, test timestep and solver sensitivity across many
seeds, and evaluate forecast calibration on held-out rolls. Those studies are
outside this initial interface implementation.

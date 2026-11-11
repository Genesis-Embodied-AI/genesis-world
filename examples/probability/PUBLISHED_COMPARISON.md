# Comparing published real experiments with Genesis

The published-reference chart uses actual previously computed Genesis outcomes
and reported real measurements. It is an **unmatched descriptive comparison**,
not a validation of the simulator's trajectory or contact physics. It does not
change the simulation or its forecast behavior.

## Reproduce the chart

From the repository root, using the existing environment:

```powershell
$env:MPLCONFIGDIR="$PWD\.cache\matplotlib"
.venv\Scripts\python.exe -m examples.probability.published_reference --export out/probability/coin-20261111T062350535885Z.json --out out/probability/published-comparison.png
```

Use any later **coin JSON export** with `--export` to regenerate it. If this
machine-local export is unavailable, run and export a coin batch in Probability
Lab first. Outputs are PNG, SVG, and a JSON containing the input hash, reported
real reference, simulation counts/configuration and source-run provenance.
NumPy, SciPy and matplotlib were already installed; no dependencies were added.

The left panel compares the real study's rounded heads estimate and published
95% Bayesian credible interval with the simulation's heads fraction conditional
on a resolved head/tail result and a marginal 95% Wilson confidence interval.
These intervals have different statistical meanings. The right panel retains
**every simulated trial** in the denominator, including unresolved tosses.
Intervals assume the relevant binomial sampling model and do not include model
error, tracking uncertainty or uncertainty about the launch distribution.

The saved 256-toss run contains 82 heads, 76 tails, and 98 unresolved outcomes.
Its conditional heads estimate is 82/158 = 51.9%, with Wilson interval
[44.2%, 59.6%]. Unresolved is 98/256 = 38.3%; it is not a measured edge rate.
No real trajectory or real running-frequency curve is reconstructed from the
published aggregate. No synthetic outcomes are used in this chart.

## Where to see real experimental graphs

1. [Bartos et al., *Fair coins tend to land on the same side they started*,
   arXiv:2310.04153v4](https://arxiv.org/html/2310.04153v4).
   Figure 2 plots estimates of heads probability by coin. The abstract reports
   50.0% heads, 95% credible interval [49.8%, 50.2%], over 350,757 accepted flips.
   The Methods specify catching coins in the hand and repeating failed catches.
   This is an outcome reference, not a floor-contact dataset. The separate
   same-side result (50.8%) is a different probability from heads.
2. [Diaconis, Holmes and Montgomery, *Dynamical Bias in the Coin Toss*,
   author-hosted manuscript](https://people.ucsc.edu/~rmont/papers/headswithO.pdf).
   Section 5 discusses physical measurements and high-speed imaging of
   orientation. Figure 10 shows ribbon-experiment observations; theoretical
   curves elsewhere in the paper should not be treated as recorded trajectories.
3. [Strzalko et al., *Dynamics of coin tossing is predictable*, Physics Reports
   469 (2008), 59-92](https://doi.org/10.1016/j.physrep.2008.08.003).
   Section 2 describes high-speed experiments at 1500 frames/s. Section 5
   presents numerical results. A numerical graph in a paper is not automatically
   a real-world measurement, and these experiments do not match our current setup.

## Why this does not yet measure simulator accuracy

The current local world uses a uniform 20 g cylinder of radius 45 mm and
thickness 8 mm on a flat plane. The saved run uses a 0.45 m launch height,
18 rad/s spin, isotropic initial orientation, sliding friction 0.6, a 0.01 s
timestep with two substeps, and a 6 s horizon. Its floor coefficients are model
parameters, not a calibration to the real study's material. These values were
not tuned by the chart script. The export records the exact run configuration;
source hashes identify its implementation.

To compare physical motion, record the same marked coin and floor, calibrated
time/length axes, release pose, and world-frame linear/angular velocity for each
real toss. Overlay real (blue) and Genesis (orange) height/orientation curves
for paired tosses. Then plot the actual errors:

- Position: `1000 * ||p_sim(t) - p_real(t)||` in mm.
- Full orientation: `2 * acos(clamp(abs(dot(q_sim, q_real)), 0, 1))` in radians,
  converted to degrees, for unit quaternions in the same coordinate convention.
- Impacts: completed rebounds, their apex heights, and time to sustained rest,
  with the same observation window and declared thresholds for both data sources.

Calibrate contact parameters on separate tosses, then evaluate on held-out
tosses. Report edge and unfinished trials separately. Replaying the identical
release with different seeds is a repeatability check, not independent toss data.
Neither visual similarity nor 50/50 outcomes alone establishes matching physics.

Matched position/orientation errors, bounce differences, and settling differences
remain **unmeasured** until suitable real tracking data is available. The OSF
materials link from the study was inaccessible during this check (HTTP 403);
no raw video or full 6-DoF tracking dataset was downloaded or validated.

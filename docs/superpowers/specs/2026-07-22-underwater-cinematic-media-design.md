---
title: Underwater Cinematic Buoyancy Media Pack
status: approved-design-pending-written-review
date: 2026-07-22
branch: release/akinci-boundary-upstream-20260721
baseline_commit: 977458ba5d752952798e528b25f2c05699ca54b1
source: api:genesis-local-repository
confidence: 0.92
---

# Underwater Cinematic Buoyancy Media Pack

## Purpose

Add high-quality, reproducible water media to the active fork branch while keeping the existing buoyancy comparison scientifically auditable. The result should look like a short scientific-documentary film, but every simulated position and reported metric must come from the real Genesis CUDA run.

The branch in `robotlearning123/genesis-world` is the primary delivery surface. Opening or updating an official-upstream pull request is outside this design. The implementation must preserve unrelated worktree changes and the repository's existing media policy.

## Non-negotiable decisions

- Render a `2560x1440`, 30 fps, 18-second MP4 master. The acceptance range is 15--20 seconds, but 18 seconds is the implementation target. Do not silently upgrade the full movie to 4K.
- Render one separate `3840x2160` hero still.
- Qualify the ray-traced master with full-resolution probes at 64 samples per pixel (spp), then 128 spp only if 64 spp is inadequate.
- Fail closed if the 128 spp probe cannot meet the visual-quality gate. A rasterized or lower-resolution recording must not be labeled movie-quality output.
- Use an evidence-preserving scientific-documentary treatment: restrained typography, deep-cyan laboratory lighting, warm rim light, and clear subject separation.
- Run the stock and Akinci simulations independently for the full 6000 steps. Never fabricate, interpolate, retime, or post-process object positions.
- Preserve the existing `*.mp4` and `outputs/` ignore policy. The tracked branch preview is animated WebP; the reproducible MP4 master remains an ignored local deliverable unless repository policy is changed separately with explicit approval.

## Deliverables

The implementation will add the following tracked files:

- `examples/underwater/render_buoyancy_media.py`: deterministic GPU renderer, compositor, encoder driver, and manifest writer.
- `examples/underwater/media/buoyancy_hero_4k.png`: 4K hero frame.
- `examples/underwater/media/buoyancy_establishing.png`: 1440p establishing frame.
- `examples/underwater/media/buoyancy_comparison.png`: 1440p synchronized comparison frame.
- `examples/underwater/media/buoyancy_results.png`: 1440p result frame with measured submerged fractions.
- `examples/underwater/media/buoyancy_compare_preview.webp`: an eight-second excerpt of the strongest comparison sequence, downsampled from the master frame stream.
- `examples/underwater/media/manifest.json`: provenance, run configuration, physics results, quality decision, and artifact hashes.
- Focused tests for capture scheduling, manifest integrity, and evidence-preserving composition.
- An updated `examples/underwater/README.md` that embeds the hero and animated preview and links the reproduction command.

The full-render command will also generate this ignored local deliverable:

- `examples/underwater/outputs/buoyancy_compare_master_1440p.mp4`

No dependency, CI, or global Git ignore changes are part of this feature.

## Architecture and boundaries

### Shared physics scene

Refactor `examples/underwater/buoyancy_demo.py` only enough to expose a small scene builder and structured result calculation. The numerical demo and media renderer must share the same constants for time step, gravity, SPH bounds, particle size, liquid density, body densities, body dimensions, and initial transforms.

The builder may accept render-only surfaces and cameras. It must not add collision geometry, modify mass or density, inject analytic buoyancy, alter solver steps, or otherwise change the physical experiment. Environment maps, ray-tracer lights, material appearance, and camera pose are visual-only inputs.

### Media renderer

`render_buoyancy_media.py` will:

1. Initialize Genesis once with CUDA and 32-bit precision.
2. Run a fail-fast environment preflight.
3. Render the 64 spp full-resolution quality probe for both stock and Akinci modes and write a temporary contact sheet and machine-readable probe report.
4. Require a human to review the contact sheet as well as the automated technical checks.
5. If either review rejects 64 spp, render and review a 128 spp probe.
6. Stop without publishing final media if the automated or human review rejects 128 spp.
7. Require the human-approved spp value to be passed explicitly to the full-render command.
8. Run the stock and Akinci scenes independently for 6000 steps with the approved renderer settings.
9. Capture the same simulation-step schedule and camera transform for both modes.
10. Compose the synchronized views, editorial slates, and measured result labels.
11. Encode the MP4 and WebP from the same accepted frame stream.
12. Write the stills and manifest atomically, then remove owned staging and consumed probe files.

The reproducible flow is two commands. First generate the quality probe:

```bash
CUDA_VISIBLE_DEVICES=0 .venv-gpu/bin/python \
  examples/underwater/render_buoyancy_media.py --probe-only --probe-spp 64
```

If 64 spp is rejected, repeat the probe at 128 spp. After inspecting and accepting a contact sheet, run the full render with the matching spp, for example:

```bash
CUDA_VISIBLE_DEVICES=0 .venv-gpu/bin/python \
  examples/underwater/render_buoyancy_media.py --approved-spp 64
```

Four narrow operational options are allowed:

- `--probe-only` runs the full-resolution quality probe without rendering the final media.
- `--probe-spp {64,128}` selects the probe quality and is valid only with `--probe-only`.
- `--approved-spp {64,128}` is mandatory for a full render and must match an accepted probe report produced from the same commit, GPU, renderer, and resolution.
- `--force` permits replacement of an existing complete output set after preflight succeeds.

The release path intentionally does not expose arbitrary step-count, frame-rate, or resolution switches. Reduced diagnostic runs must not be able to overwrite the accepted tracked media.

## Visual treatment and edit

### Look

- Deep-cyan scientific laboratory environment with controlled contrast.
- Cool ambient illumination plus a restrained warm rim light for shape separation.
- Distinct, color-safe finishes for the light and heavy bodies.
- Ray-traced water and object surfaces with denoising enabled.
- Thin-lens depth of field only in the hero and establishing shot. The evidence comparison uses a fully readable pinhole view.
- No generated concept art, stock footage, unlicensed audio, or synthetic replacement water.

### Shot structure

The target edit is exactly 18 seconds, or 540 frames at 30 fps:

1. **0--2 seconds, establishing shot:** labeled underwater tank and density pair, with a restrained camera move over an explicitly recorded held physics state.
2. **2--14 seconds, synchronized comparison:** stock #2857 on the left and `+ Akinci boundary particles` on the right, using 360 evenly scheduled real simulation states, the same camera path, and the same step indices.
3. **14--16 seconds, evidence close-ups:** matched crops of accepted comparison frames that make the light-body and heavy-body behavior readable without hiding the full-scene context.
4. **16--18 seconds, result hold:** repeated final frames with measured submerged fractions and the known additive-path heavy-body limitation.

Clean cuts and repeated result holds are allowed. Motion interpolation, generated in-between physics frames, position warping, simulation-dependent speed ramps, and unlabeled mode switching are prohibited. Editorial frames and repeated holds must be identified in the manifest.

### Labels and claims

The comparison labels are factual:

- `Stock #2857`
- `+ Akinci boundary particles`
- `light rho=600 kg/m^3`
- `heavy rho=1400 kg/m^3`
- final measured submerged fractions

The result slate must state that Akinci is additive to the stock coupling and may over-buoy the heavy body. It must not claim that Akinci is universally more realistic, more correct, or required for basic flotation.

## Resolution and encoding

### Master and stills

- Master: `2560x1440`, 30 fps, 18 seconds, H.264 High profile, `yuv420p`, constant frame rate.
- Hero: `3840x2160`, rendered at 128 spp with denoising after the master probe qualifies.
- Key stills: `2560x1440` PNG files from accepted master frames.
- Master frames: rendered at the lowest qualified setting of 64 or 128 spp.

FFmpeg must encode frames directly in the captured order. Optical flow, motion interpolation, generative upscaling, and physics-frame synthesis are prohibited.

### Tracked preview

- Animated WebP derived from the accepted master frame stream.
- Target: `1600x900`, 24 fps, exactly eight seconds (192 frames), no more than 15 MiB.
- The preview may choose a different eight-second excerpt or omit repeated holds to meet the size target. It must not use stronger temporal manipulation than the master.
- The 24 fps preview uses deterministic timestamp-preserving frame selection from the 30 fps master, with no interpolation; its source master-frame indices are recorded in the manifest.
- If the target cannot be met without obvious banding, unreadable labels, or severe motion degradation, the generator fails the preview gate instead of publishing a misleading low-quality asset.

## Quality-probe gate

The probe renders representative early, middle, and final frames for both modes at the full `2560x1440` master resolution. It stores its contact sheet and report in a unique ignored probe directory under `examples/underwater/outputs/` until human review.

1. Render at 64 spp with denoising.
2. Check technical validity and inspect the contact sheet at original resolution.
3. Accept 64 spp only if both the technical and human checks pass.
4. Otherwise discard the rejected raw probe frames and repeat the technical and human checks at 128 spp.
5. If 128 spp fails either review, exit nonzero, remove owned probe staging files, and publish no final media.
6. Start the full render with `--approved-spp` matching the accepted report. The generator refuses stale, mismatched, or technically failed probe reports.

Acceptance criteria:

- No renderer crash, CUDA error, out-of-memory error, NaN/Inf pixel data, missing geometry, or incomplete frame.
- No obvious temporal scintillation in static background regions or denoiser smearing at moving body edges.
- Water particles, waterline, both bodies, and their relative depth remain legible in every evidence shot.
- Highlights retain surface detail and shadows do not erase either body.
- Matched stock/Akinci frames have identical crop, camera transform, typography placement, and simulation-step index.
- Text is fully inside safe margins and remains readable after the WebP downsample.

The generator records the outcome of every attempted probe and records unattempted settings as `not_run`. A human visual review of the candidate contact sheet is required before the full render starts; there is no automatic lower-quality fallback. After a successful full render copies the decision into `manifest.json`, it removes the consumed probe directory. A failed probe is reported with its diagnostic contact sheet when one exists, then its owned probe directory is removed before task closeout.

## Provenance manifest

`manifest.json` will use a versioned schema and include:

- `source: "api:genesis-local-gpu"`
- `confidence: 0.95`
- UTC generation time
- repository branch, source commit, and clean/dirty state
- exact generation command
- Python, Genesis, CUDA, GPU, driver, FFmpeg, PyAV, and Pillow versions
- renderer, denoiser, master resolution, frame rate, selected spp, and probe decisions
- physics constants and the exact captured simulation-step indices
- stock and Akinci final positions, waterline, submerged fractions, and boundary-particle counts
- editorial-frame ranges and a declaration that no object positions were fabricated or altered
- artifact path, byte size, dimensions, frame count or duration, codec where applicable, `source`, `confidence`, and SHA-256 for every output

The confidence describes the lineage of the local run and artifact checks. It does not express confidence that the additive Akinci path is a universally correct physical model.

## Failure handling and temporary-file hygiene

The generator stages owned intermediate files under an ignored directory inside `examples/underwater/outputs/`, never in an untracked global scratch location. Each invocation uses a unique, explicit staging directory.

- Preflight checks CUDA availability, ray-tracer construction, required Python packages, FFmpeg/FFprobe, writable output paths, and existing destination files.
- Without `--force`, existing complete outputs cause a fail-fast error.
- Final files are moved into place only after rendering, encoding, hashing, and validation all succeed.
- Any exception removes the invocation's probe frames, raw frame sequence, encode fragments, and staging directory.
- Existing files that predate the invocation and artifacts owned by other sessions are never removed.
- A failed run leaves the previously accepted tracked media unchanged.

## Verification plan

### Focused code checks

- Unit-test the shared capture schedule to prove that stock and Akinci frame indices are identical.
- Unit-test composition with synthetic input arrays to prove fixed layout, safe labels, and stable dimensions without invoking Genesis.
- Unit-test manifest validation, per-artifact metadata, and SHA-256 recording.
- Lint only the changed Python files and run `git diff --check`.

### Real GPU checks

- Run `--probe-only` on the RTX GPU and inspect the probe contact sheet at original resolution.
- Run the full 6000-step stock/Akinci render at the explicitly approved spp.
- Re-run the focused Akinci GPU tests and the existing `buoyancy_demo.py --compare` numerical gate.
- Record the real GPU identity, peak memory where available, selected spp, final physics values, and exit codes.

### Artifact checks

- Use FFprobe to verify MP4 resolution, constant 30 fps, 18-second duration, H.264 codec, and exactly 540 frames.
- Use Pillow to verify every PNG dimension and the WebP dimensions, animation flag, frame count, and duration.
- Verify every manifest SHA-256 against the final file.
- Inspect the hero and beginning/middle/end frames for composition, water visibility, label occlusion, temporal noise, and color consistency.
- Confirm the WebP is at most 15 MiB and remains readable in the rendered README.
- Confirm `git status` includes only the intended source, documentation, tests, and tracked media before commit.

## Scope exclusions

- No official-upstream pull request work.
- No 4K full-movie render.
- No new dependency or CI configuration.
- No Git LFS introduction and no forced addition of ignored MP4 files.
- No redesign of the buoyancy model or solver.
- No fabricated physics frames, generated concept imagery, or claim that the additive Akinci result is universally correct.
- No cleanup of other sessions' worktrees, processes, logs, or temporary artifacts.

## Completion criteria

The feature is complete only when the tracked media, ignored reproducible MP4, generator, manifest, README, focused tests, full numerical comparison, focused Akinci GPU tests, artifact inspections, and clean scoped Git diff all pass. Passing a short render smoke test alone is not sufficient.

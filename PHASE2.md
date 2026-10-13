# Phase 2 validation — 13 October 2026

## Tasks 1–3 checkpoint

- Baseline: 27 tests passed, synthetic demo succeeded; actual M5 webcam at
  1280×720: 299/300 accepted pose frames and 29.7 whole-loop FPS.
- Added optional hips/nose to the existing shoulder/elbow/wrist adapter. Torso
  deformation uses shoulder orientation/scale and bounded hip lean/waist/length.
  Low-confidence observations are smoothed; missing hips transition toward
  shoulder-only geometry. Lost pose immediately restores original video.
- Added separate optional segmentation, conservative clothing-only compositing,
  head/hand landmark guards, feathered confidence edges, low-resolution optical-flow
  propagation, a scene/motion guard, and a 250-ms maximum cache age. One background
  job is in flight; no queued video or automatic recording.
- With permission, downloaded Google's Apache-2.0 256×256 SelfieMulticlass model.
  It remains ignored in `models/`; the exact MediaPipe 0.10.21 requirement and
  macOS process network restriction still apply before inference starts.
- Live M5 validation with segmentation: 300/300 accepted pose frames, 75 usable
  segmentation updates, 29.6 whole-loop FPS at 720p. Main-thread means: pose 7.9 ms,
  segmentation/cache work 1.0 ms, render 3.9 ms, protected composite 1.7 ms,
  display 15.1 ms. Virtual output is absent, therefore output FPS correctly reads 0.
- 43 automated tests pass; synthetic demo succeeds. Tested geometry, confidence
  handling, protected categories, mask freshness/motion, tracking-loss invalidation,
  BGR compositing, source immutability, and actual-output FPS accounting.
- New controls: G segmentation, D landmarks, M mask. Existing controls preserved.
  P saves a local still only on a key press. Automatic `--snapshot-frame` is disabled.

This is still a non-photorealistic 2D geometric renderer. The next increment adds
structured/RGBA garment assets and approximate fabric/lighting; it does not claim
photorealism, physical cloth simulation, or 3D reconstruction. Segmentation can
misclassify boundaries, sleeves, or rapidly moving hands; pixel-perfect protection
and end-to-end camera latency are not established by these measurements.

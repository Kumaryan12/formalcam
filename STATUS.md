# Local validation report — 13 October 2026

## Tested checkpoint

- Reviewed all project source and repository instructions.
- Project-local Python 3.11 environment on macOS 26.5 / arm64. No Homebrew or OBS installation.
- `python -m pytest -q`: 17 passed after fixes.
- `python -m formalcam.app --demo-frame`: succeeded; synthetic PNG inspected.
- Regression coverage: original-frame fallback, missing-pose smoothing reset,
  invalid/partial landmarks, camera cleanup on startup failure, timing measurements,
  screen-side light direction, degenerate light region, and interrupted model downloads.
- FPS now measures completed frame loops, including display. Optional CSV contains
  numeric timings only; optional original/preview stills stay in ignored `diagnostics/`.
- One OpenCV distribution; modern OBS camera-extension support requires pyvirtualcam >=0.14.
- Pose model downloaded only using the explicit Google model download script.

## Not yet verified

- Live camera acquisition, macOS camera permission, live pose detection, actual FPS,
  garment fit, and lighting on the user's webcam.
- OBS is not installed at `/Applications/OBS.app`; virtual-camera output is unverified.
- Tests using mocked cameras verify resource cleanup, not a physical camera or OBS.

## Visual limitations

The renderer is a non-photorealistic 2D geometric overlay. It lacks garment textures,
3D body/cloth geometry, folds, physically correct lighting, and skin/hair/hand segmentation.
Forearm restoration is an optional approximate line mask and can restore original sleeves.
Validate on-device camera and pose first; gather local stills and numeric timing logs before
optimizing alignment, lighting stability, or frame rate. Never upload webcam captures.

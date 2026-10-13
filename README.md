# FormalCam MVP 0.1 (MacBook Apple Silicon)

An **experimental local webcam suit overlay**, intended for testing with OpenAI Codex.

**Important:** This is a 2D geometric rendering that tracks shoulders using MediaPipe Pose Landmarker. It has basic illumination matching, pseudo-lapel shadows, and approximate forearm occlusion. It is **not** photorealistic, does **not** produce physically correct fabric folds or lighting, and must **not** be described as a finished virtual try-on system.

**Local runtime:** MediaPipe is pinned to **0.10.21** because newer distributed
builds include telemetry with no official opt-out ([upstream discussion](https://github.com/google-ai-edge/mediapipe/issues/6291)).
Before importing the pose runtime, FormalCam checks this exact version and uses
macOS's process sandbox to deny all networking. If either check fails, preview
does not start. This affects only the FormalCam process; it changes no system
firewall settings or camera permissions. Do not loosen the MediaPipe pin.
Installation and the explicitly invoked model downloader run separately.

## 1. Requirements

- Apple Silicon Mac (M5 is fine), macOS camera permission, terminal access
- Homebrew Python 3.11 (recommended for consistent wheels)
- Optional: OBS Studio **30+** to install the virtual camera backend
- Webcam, sufficient room lighting, upper body visible

## 2. Install on Mac

From Terminal, inside this extracted directory:

Install only with your authorization. If Python 3.11 is already installed,
skip the Homebrew install. Dependencies stay in this project's `.venv`.
Use one OpenCV wheel (`opencv-contrib-python`, also required by MediaPipe);
do not also install `opencv-python` or a headless OpenCV wheel in this environment.

```bash
brew install python@3.11
"$(brew --prefix python@3.11)/bin/python3.11" -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/download_model.py
```

`requirements-lock.txt` records the dependency versions validated on Python 3.11
and macOS arm64. Use `python -m pip install -r requirements-lock.txt` when
recreating that environment instead of resolving the version ranges afresh.
The published MediaPipe 0.10.21 universal2 wheel has an x86_64-only internal WHEEL
tag, so newer `pip check` can report it as unsupported on arm64. Its native binary
contains arm64 and has been tested on this M5; see `STATUS.md`. Do not resolve
that metadata warning by upgrading to a telemetry-bearing build.

MediaPipe model is downloaded from Google's official public storage location. It is ignored in Git.

## 3. Test without a webcam (smoke test)

```bash
python -m pytest -q
python -m formalcam.app --demo-frame
open demo_preview.png
```

This synthetic image exercises the compositing code; it does *not* validate detection or appearance on a real human.

## 4. Test webcam preview

```bash
python -m formalcam.app
```

Allow camera permissions for Terminal (or VS Code) when prompted. Put shoulders and upper chest in the frame, with good lighting. **Q** quits.

The default request is **1280 × 720 at 30 FPS**. The actual camera resolution
and reported capture FPS are printed at startup. Processing FPS is measured
across the whole loop, including capture, pose inference, rendering, and display;
it is not a measurement of camera-to-screen latency.

For a bounded validation session:

```bash
source .venv/bin/activate
python -m formalcam.app --max-frames 300 --profile diagnostics/baseline.csv --snapshot-frame 150
```

The CSV contains only numeric stage timings and pose detection counts. The
snapshot saves an original frame and desktop preview **locally** in `diagnostics/`.
Without `--snapshot-frame`, stills are saved only when you press **P**. No video
is recorded or uploaded. Do not upload webcam captures to a service for review.
Inspect them on your Mac and describe fitting or lighting problems in text.
CSV paths are overwritten on each run; use distinct names to compare sessions.

First validate shoulders with `pose: yes`, then turn sideways, raise each hand,
and briefly leave the frame. Missing pose must show the original webcam.
Compare the local original/preview stills before tuning suit geometry or lighting.
If it is too slow, use `--width 960 --height 540` for diagnosis, then return to 720p.

Keyboard controls:

| Key | Action |
|---|---|
| 1, 2, 3 | Navy, charcoal, black |
| L | Adaptive brightness on/off |
| S | Lapel shadow approximation on/off |
| A | Approximate forearm occlusion on/off |
| O | Original webcam vs suit |
| P | Save original and preview stills locally |
| I / K | Move suit up / down |
| [ / ] | Narrow / widen suit |
| R | Reset width and vertical fit |
| Q or Escape | Quit |

If the wrong camera opens, try `python -m formalcam.app --camera 1`.

If the collar sits low, begin with `--fit-offset=-0.05` and fine-tune with I/K.
Offset uses measured shoulder-width units, so -0.05 lifts the garment by 5% of
shoulder width. Width can be initialized with `--fit-width 0.95`. These adjust the
2D geometry; they do not estimate neck shape or turn the garment into a 3D model.
Light estimation still uses the original detected shoulders rather than the shifted garment.
Pose and light smoothing use elapsed frame time to keep their response consistent
when FPS varies. Losing pose clears smoothing and light history immediately; the
original webcam is displayed rather than retaining a stale garment.

To compare throughput with overlapping capture and processing:

```bash
python -m formalcam.app --latest-frame --fit-offset=-0.05 --max-frames 300 --profile diagnostics/latest-frame.csv
```

This optional mode keeps only the newest camera frame, dropping older unprocessed
frames rather than accumulating a queue. Capture owns a background reader; pose,
rendering and the macOS preview stay on the main thread. The synchronous mode is
still the default. Compare numeric CSVs under the same lighting and framing;
snapshot writes temporarily pause processing. Neither FPS nor stage timings measure
the complete sensor-to-screen latency. Low-confidence/missing pose still shows the original.

## 5. Virtual camera output (optional, after preview works)

**macOS requires a virtual camera component installed separately for this version.**

1. With your authorization, install OBS Studio 30 or later (`brew install --cask obs` if Homebrew is available). Modern macOS requires pyvirtualcam 0.14 or later, included in `requirements.txt`; see the [upstream setup guide](https://github.com/letmaik/pyvirtualcam#macos-obs).
2. Open OBS, press **Start Virtual Camera**, then **Stop Virtual Camera**. Quit OBS completely. You only need to initialize its system extension once.
3. Run:

```bash
python -m formalcam.app --virtual-camera
```

4. Open Chrome → Google Meet and choose the **OBS Virtual Camera** device in video settings. If Chrome was already running, restart it and try again. The **clean processed feed** is sent; overlay controls appear only in the desktop preview.

Important: On macOS, `pyvirtualcam` and OBS use the same single device. Don't keep OBS running while `pyvirtualcam` is sending to it. If the virtual-camera device fails to open, run the preview without `--virtual-camera` first.

## 6. Project structure

```text
formalcam/
  __init__.py
  app.py           # webcam loop, MediaPipe pose, display, virtual sender
  diagnostics.py   # local numeric performance logs and opt-in stills
  capture.py       # optional background capture with only the latest frame
  privacy.py       # version check and macOS process network restriction
  pose.py          # coordinate extraction and EMA smoothing
  render.py        # stylized blazer geometry, basic light matching
scripts/
  download_model.py
models/             # downloaded MediaPipe model (not included)
tests/
  test_renderer.py
AGENTS.md           # instructions for Codex
CODEX_TASK.md       # next tasks for Codex
requirements.txt
```

## 7. Known limitations and next improvements

- Camera permissions and virtual-camera system extension cannot be tested in a headless/cloud container; must test on your Mac.
- 2D cartoon-like geometry, not real clothing. No true mesh deformation, texture, or physically based shadows.
- Tracking fails with severe occlusion or extreme body angles.
- Shoulder tracking is smoothed, but measured webcam FPS and latency may be below target.
- Approximate forearm masking can pull original sleeves back in and look unnatural.
- No video recording, face replacement, runtime network access, analytics or remote processing. Optional stills and timing CSVs remain local.
- Improve by integrating licensed garment assets, photometric estimation, hair/neck/hand segmentation, and a rigged garment mesh.

## Licensing note

This source is an original example. Google's MediaPipe model and the dependent libraries have their own licenses; review before redistributing.

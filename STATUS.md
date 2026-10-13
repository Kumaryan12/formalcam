# Local validation — 13 October 2026

This is a **non-photorealistic 2D geometric overlay**, running on this MacBook M5
with Python 3.11, macOS 26.5, and project-local dependencies. No system software
was installed and no macOS permissions were changed by the agent.

## Working and tested

- Original webcam preview works at 1280×720; the user confirmed it was visible.
- `python -m pytest -q`: **27 passed**. `python -m formalcam.app --demo-frame`:
  succeeds; synthetic PNG inspected. All project source and instructions reviewed.
- Final runtime: MediaPipe **0.10.21**, OpenCV **4.11.0**, NumPy **1.26.4**,
  pyvirtualcam **0.15.0**. Exact environment is in `requirements-lock.txt`.
- Missing pose immediately displays the original frame; pose and lighting history reset.
- Fixed zero FPS, startup cleanup, invalid/partial landmarks, reversed directional
  lighting, one-column lighting NaN, interrupted model downloads, and forearm
  restoration changing pixels outside the garment.
- **I/K** move the suit up/down, **[/]** change width, **R** resets. User reported
  the collar sitting too low; `--fit-offset=-0.05` starts it 5% of shoulder width higher.
- Pose and lighting smoothing use elapsed time, tested for equivalent response at
  15 and 30 FPS. Optional `--latest-frame` overlaps camera capture and processing
  and keeps only the newest frame; synchronous capture remains the default.

## Privacy correction

MediaPipe 0.10.35 logged a failed Clearcut telemetry upload during an initial run.
The [maintainer confirms telemetry without an official opt-out](https://github.com/google-ai-edge/mediapipe/issues/6291).
That build was stopped and replaced with a strict 0.10.21 pin. Its native wheel
contains no Clearcut or play.googleapis.com markers; that scan alone is not proof
of all library behavior. FormalCam additionally **denies all process networking**
through the macOS sandbox before importing pose libraries, and refuses to start
if the sandbox or approved version is unavailable. A regression test verified
that even a connection to a live local TCP server is denied. Other apps and
system firewall settings are unaffected. Only the explicit model-download script
contacts Google's model URL; dependency installation runs separately.

Captures were never submitted to a remote vision tool or deliberately uploaded by
FormalCam. Initial dependency telemetry was not packet-audited, so the initial
runs cannot be described as having zero network activity. Local stills and numeric
CSV logs stay in ignored `diagnostics/`; neither they nor the model or `.venv` are in Git.

## Measurements and remaining checks

| Session | Frames | Valid pose | Whole-loop FPS |
|---|---:|---:|---:|
| Original camera | 300 | Not measured | 29.5 |
| Initial suit, before privacy correction | 300 | 299 | 19.6 |
| Fitted suit, before privacy correction | 300 | 299 | 16.8 |
| Final restricted runtime, latest-frame mode | 600 | 0 | 29.9 |

The final run verified camera/display and reader shutdown, but its FPS is **fallback
performance**, not suit-rendering performance. The final restricted detector also
recognized pose in an earlier local webcam still. With a person continuously in
view, final live fit, motion/lighting stability, and full suit throughput need review.
Changing capture waits and snapshot writes prevent a controlled FPS comparison.
A synthetic 720p rendering microbenchmark improved median time from 7.03 to 6.62 ms
before the dependency rollback; no overall preview FPS gain is claimed. These are
host loop timings, not full sensor-to-screen latency measurements.

`pip check` reports the legacy MediaPipe wheel as unsupported on arm64 because its
internal WHEEL tag says x86_64. The downloaded wheel filename is universal2 and the
native binary contains **both arm64 and x86_64**; it actually runs on this M5. No
other dependency mismatch was reported. Vendor metadata was not altered to hide this.

OBS is absent from `/Applications/OBS.app`; actual virtual-camera output remains
unverified. Install only with authorization, approve its extension permissions yourself,
then Start Virtual Camera, Stop it, quit OBS, and run with `--virtual-camera`.
Mocked tests verify clean output and graceful failure, not a real OBS connection.

The main realism blockers remain garment texture, 3D body/cloth geometry, folds,
physical lighting and skin/hair/hand segmentation. Face-adjacent light estimation
can include background; approximate forearm masks can reveal original sleeves.

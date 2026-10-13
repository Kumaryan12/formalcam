# FormalCam agent instructions

Goal: Improve a Mac M5 local webcam prototype, not a cloud service. Never upload webcam frames. No telemetry.

- Run `python -m pytest -q` and `python -m formalcam.app --demo-frame` after edits.
- Keep Python modules small and typed where practical.
- Current `render.py` is an explicitly non-photorealistic 2D geometric demo. Do not claim it is 3D or diffusion-based.
- Preserve fallback to original webcam when pose is missing.
- Preserve graceful errors if OBS/pyvirtualcam is unavailable.
- User must authorize installs and macOS camera/system extension permissions.
- Do not delete files unrelated to this repository.
- First validate the on-device webcam and pose detection; gather screenshots/performance logs before optimizing.
- Prioritize stability at 720p, lighting and occlusion correctness, and low latency before advanced generative models.
- Avoid network calls during runtime; only the explicit `scripts/download_model.py` contacts Google's public model URL.

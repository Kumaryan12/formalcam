# First task for Codex (paste into Codex)

Read README.md, AGENTS.md, and the source files first. We are developing a private, local-only macOS M5 prototype. Please do the following in order:

1. Inspect the repository and run `python -m pytest -q` and `python -m formalcam.app --demo-frame`. Fix actual failures; do not invent pass results.
2. If dependencies aren't installed, tell me the exact commands and request permission to install, rather than installing system dependencies without asking.
3. Review `pose.py`, `render.py`, and `app.py` for possible bugs with Apple Silicon, MediaPipe Tasks, cv2, and pyvirtualcam. Fix demonstrable ones and add regression tests.
4. Guide me through running `python -m formalcam.app` on my actual Mac. You cannot verify my webcam yourself unless the local environment provides it; ask me for logs or a screenshot when needed.
5. Once the basic camera preview works, improve suit fitting and anti-flicker; profile true FPS. Do not rewrite everything into a large framework.
6. When preview works, guide me through OBS virtual-camera initialization and `python -m formalcam.app --virtual-camera`.
7. Create an honest short report of what works, what isn't verified, and the biggest visual realism blockers.

Use minimal dependencies and keep user video local. Don't call a procedural overlay photorealistic. Do not modify unrelated files.

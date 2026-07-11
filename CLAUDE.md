# CLAUDE.md — OpenScrub

OpenScrub is a local, GPU-accelerated video redaction tool: it detects and
blurs PII (faces, names, SSNs, addresses, license plates — 12 categories) in
videos and screen recordings, with a human-review step before anything is
trusted. Apache-2.0. Python 3.10+. Runs on Windows/Linux; primary dev/deploy
target is Windows 10 + NVIDIA RTX 3060.

## File map

| File | Role |
|---|---|
| `openscrub.py` | The engine. CLI + all detection/render logic. Single file, ~2400 lines. |
| `openscrub_web.py` | Flask web app. The entire UI is one embedded `PAGE` string (HTML/CSS/JS). Serves via cheroot (production WSGI, TLS) with Flask-dev fallback. |
| `zones_ui.py` | Zone-editor page (`ZONES_PAGE`), served by the web app at `/zones`. |
| `openscrub_gui.py` | Legacy Tk GUI. Frozen; ships but is not actively developed. |
| `openscrub_setup.py` | `openscrub-setup` command: detects/installs Tesseract + FFmpeg (winget/apt), optional spaCy model + plate model. Ships in the wheel. |
| `install.py` | Windows-friendly installer (deps, GPU OCR, shortcut, `--with-plates`). |
| `plate_models.json` | Curated license-plate model registry (see PLATES.md). |
| `fetch_plate_models.py` | Alt path to fetch plate models via the open-image-models pip package. |
| `test_openscrub.py` | pytest suite (9 tests). Must stay green. |
| `tools/make_icons.py` | Regenerates every icon/logo asset from `assets/badge_master.png`. |
| `tools/make_wordmark.py` | Regenerates the typeset Poppins wordmarks (navy + white). |
| `assets/` | Brand assets. `badge_master.png` (canonical, mosaic+brackets style) and `badge_master_blurbox_alt.png` (alternate) are the sources; everything else is generated. |

## Engine architecture (openscrub.py)

Pipeline: `run_pipeline` → `run_scan` (OCR sampling + detectors, builds
`Detection` list) → `merge_detections` → review or render (`blur_region`
with modes blur/box/mosaic, per-category via `--mode-map`).

Key classes/functions (locate with grep, line numbers drift):
- `Detection` dataclass — has `dense: bool`. **Dense detections are per-frame
  position samples and must NEVER be merged** (see `merge_detections`), or
  boxes balloon across a moving object's path.
- `FaceDetector` — YuNet DNN (auto-downloaded ~230KB) with Haar fallback.
- `PlateDetector` — YOLO ONNX via cv2.dnn, NO torch dependency. Auto-detects
  two output formats: raw YOLOv8 `(1,5,8400)` and end2end `(1,N,6)` (what
  open-image-models YOLOv9 emits). INERT without a model file (logs and
  returns `[]`). Model resolution: `--plate-model` arg → `$OPENSCRUB_PLATE_MODEL`
  → `models/plate_yolov8.onnx` → `models/<registry-id>.onnx` (recommended
  first, adopts registry `input_size`).
- `detect_phi` — text-category detection over OCR line dicts. Word-loop
  order matters: card (Luhn-gated) before apikey before SSN.
- `PhiMemory`, `detect_phi`, PHI-worded log lines: **"PHI" here is the domain
  term (protected health information), not the old brand. Do not rename.**
- `load_plate_registry` / `download_plate_model` — TOFU hash pinning: empty
  `sha256` in the registry → first download computes and PINS the hash back
  into `plate_models.json`; later downloads must match or are rejected and
  deleted. Registry reads/writes must always pass `encoding="utf-8"`
  (Windows defaults to cp1252 and mojibakes labels).

Per-frame detection blocks (dense faces, plates) live inside the frame loop
in `run_scan`, AFTER the frame read and the detection-window check. The zone
lookups (`plate_zone_px`, `face_zone_px`) must stay AFTER `zones_px` is
defined — there was a real UnboundLocalError from ordering once.

## Web app (openscrub_web.py)

- `PAGE` is the whole UI. The header logo and favicon are **base64-embedded**
  in the HTML (constants inlined into the source) so they can never 404.
  `assets/` still ships for GitHub/social use.
- `ASSET_DIR` = `<script dir>/assets`; `SCRIPT_DIR` = script dir (LICENSE
  lives there).
- Jobs live in `openscrub_jobs/` next to the script. On startup there is a
  ONE-TIME migration renaming a legacy `phi_blur_jobs/` dir if present —
  **the literal string "phi_blur_jobs" is deliberate; do not "fix" it.**
- Plate model picker: `/api/plate_models` (+ `/download`, `/download_status`).
  A model shows a Download button only if its registry `download_url` is real.
  New models are picked up on the next job (detector instantiated per run).
- Server: cheroot with `BuiltinSSLAdapter` (self-signed or user cert), Flask
  dev server fallback if cheroot missing. `ssl_ctx` is a `(cert, key)` tuple.

## The 12-category alignment rule (easy to break!)

The category list exists in THREE places that must stay identical:
1. `openscrub.py` — argparse default `"name,dob,phone,ssn,mrn,email,address,card,apikey,ipaddr,plate,face"`
2. `openscrub_web.py` — `const CATS=[...]` in PAGE's JS
3. `zones_ui.py` — `const CATS={...}` color map in ZONES_PAGE

When adding a category, update all three + the `IMMEDIATE` set if it needs
no confirmation delay. Verify alignment:
`grep -o 'name,dob[^"]*' openscrub.py` vs the two JS lists.

## Verification workflow (do this after every change)

```
python -c "import ast; ast.parse(open('openscrub.py').read())"   # each edited .py
# extract PAGE's <script> to a file and: node --check that_file.js
python -m pytest test_openscrub.py -q                             # 9 tests, all green
python -m build          # FULL build (sdist->wheel), NEVER just `-w`:
                         # the wheel is built FROM the sdist in CI, so any
                         # file the wheel force-includes must be in the
                         # sdist include list too (v1.0.2 failed on this)
```
For engine changes, also run a real render on a small synthetic video and
check the report JSON. For web changes, boot the server and hit the routes.
After multi-part string replaces, `grep` to confirm every edit actually
landed — partial silent misses have happened repeatedly.

## Brand / assets

- Canonical badge: `assets/badge_master.png` (hexagon, bucket-hat figure,
  mosaic face + red corner brackets), cut from the full-bleed key art in
  `assets/social_preview_master.png` (navy background keyed to alpha),
  then Real-ESRGAN x4-upscaled to 1628x1848 so every icon size is a
  downscale. When new key art arrives: re-cut the badge, AI-upscale it
  ~4x, replace both masters, re-run make_icons, and re-embed the web
  header/favicon base64 (icon-32 → the `<link rel="icon">` data URI,
  icon-128 → the `<header>` img data URI in openscrub_web.py).
  Alternate blur+box style preserved.
- `social_preview.png` is `social_preview_master.png` resized to 1280x640
  by make_icons; the composited badge+wordmark layout is only a fallback
  when no master exists.
- Wordmark: typeset Poppins Bold (fonts in `assets/fonts/`, OFL). "Open"+"ub"
  sharp, "Scr" Gaussian-blurred at alpha 168, red corner brackets.
  **Never ask an image generator for wordmark text — it reliably mangles
  letterforms. Badge art = AI; wordmark = typesetting.**
- Regenerate everything: `python tools/make_icons.py` and
  `python tools/make_wordmark.py`.
- All icon sizes (including 512/1024) are now downscales of the 1628px
  AI-upscaled badge master — no soft upscaling remains in the pipeline.

## Releasing

1. Bump `VERSION` in `openscrub.py` (pyproject reads it dynamically).
2. Commit, `git tag vX.Y.Z`, push the tag.
3. GitHub → Releases → draft release on that tag → Publish.
4. `.github/workflows/publish.yml` builds and publishes to PyPI via Trusted
   Publishing (environment name `pypi` — must exist in repo settings and
   match the PyPI pending publisher exactly). First publish claims the name.

## Hard rules

- This is a privacy tool: **fail closed.** Over-blur beats under-blur;
  unverified models are rejected loudly, never run silently.
- No real patient/provider data, names, or clinic-identifying examples in
  any committed file. Test data is synthetic.
- Keep the human-review step prominent in any UX change; "best-effort
  redaction — always review output" is a product principle, not a disclaimer.
- Report JSON format is a compatibility surface (review UI + rehydration
  read it); extend, don't break.
- If `CLAUDE.local.md` exists in the working directory, read it — it holds
  private deployment context that must not be committed.

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
| `zones_ui.py` | The "Scan Setup" editor page (`ZONES_PAGE`), served at `/zones`: load a video, stack detection windows on a timeline (one lane per window, overlap allowed), give each window its own categories + zones, copy/paste zones, mute audio lanes, set clip bookends, Start scan. Replaced the old global zone editor. |
| `openscrub_gui.py` | Legacy Tk GUI. Frozen; ships but is not actively developed. |
| `openscrub_setup.py` | `openscrub-setup` command: detects/installs Tesseract + FFmpeg (winget/apt), optional spaCy model + plate model, Windows Start Menu shortcuts. Ships in the wheel. |
| `windows/` | Native Windows packaging: `openscrub.spec` (PyInstaller, two branded exes), `installer.iss` (Inno Setup → Program Files), `build_installer.bat` (runs both; attach output exe to the GitHub release). Build on Windows only. |
| `install.py` | Windows-friendly installer (deps, GPU OCR, shortcut, `--with-plates`). |
| `docker/` | `Dockerfile.opencv-cuda` builds the CUDA-OpenCV base image (rare, via opencv-cuda-base.yml); `Dockerfile.cuda` FROMs it (base published to ghcr; bump the FROM tag only when the base is rebuilt). |
| `plate_models.json` | Curated license-plate model registry (see PLATES.md). |
| `face_models.json` | Curated optional face-model registry (CenterFace/SCRFD); built-in YuNet needs no file. Ships everywhere plate_models.json does (wheel, sdist, Dockerfiles, PyInstaller spec, updater pin-carry). |
| `fetch_plate_models.py` | Alt path to fetch plate models via the open-image-models pip package. |
| `openscrub_update.py` | `openscrub-update` command + web self-update backend: PyPI version check, sha256-verified sdist download, data-preserving folder update (PRESERVE set), TOFU pin carry-forward. Ships in the wheel. |
| `openscrub_vault.py` | At-rest encryption for the job store: scrypt keystore, chunked AES-256-GCM files (`.osvault`), lock/unlock tree walkers. NO password reset by design. Ships in the wheel. |
| `test_openscrub.py` | pytest suite (42 tests). Must stay green. |
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
- `FaceDetector` — three tiers: optional ONNX model (CenterFace or SCRFD,
  auto-recognized by output-layer count: 4 → centerface, 6/9 → scrfd;
  decoders validated against reference output) → YuNet DNN
  (auto-downloaded ~230KB, zero-setup default) → Haar fallback. Model
  resolution: `--face-model` → `$OPENSCRUB_FACE_MODEL` → built-in. A model
  that fails to load falls back LOUDLY to YuNet. An optional model runs
  ALONGSIDE YuNet and detections are UNIONED (nms-merged) — a model
  upgrade can only add faces, never lose the baseline's (SCRFD through a
  squeezed 640 input detected FEWER faces than YuNet; SCRFD input is now
  letterboxed up to 1280 long-side, aspect preserved). `face_models.json`
  is the curated registry (CenterFace pinned/MIT; SCRFD
  pinned/non-commercial — never bundle it). Registry plumbing is shared with plates:
  `model_registry_path/load_model_registry/download_model(kind)` with
  plate-named wrappers kept for compatibility.
- `PlateDetector` — YOLO ONNX, NO torch dependency. **Two backends tried per
  model:** (1) cv2.dnn (fast, honours the CUDA target; handles raw YOLO heads),
  (2) **onnxruntime** fallback for "end2end" exports. cv2.dnn CANNOT build the
  baked-in ONNX `NonMaxSuppression` node — `readNetFromONNX` raises `Can't
  create layer ... NonMaxSuppression` on EVERY open-image-models YOLOv9 model
  in the registry, so before onnxruntime the plate category was silently inert
  (a fail-OPEN hole; plates never blurred). The cv2 probe is wrapped in
  `cv2.setLogLevel(0)`/restore so its expected red ERROR doesn't scare users.
  onnxruntime is a hard dep (requirements + pyproject); the CUDA image installs
  `onnxruntime-gpu --no-deps` so plates run on the GPU (CUDAExecutionProvider,
  CPU listed as runtime fallback; `OPENSCRUB_CPU_DNN=1` forces CPU). Frozen
  Windows build bundles it via `collect_all("onnxruntime")` in the spec.
  `_decode` (pure, unit-tested) auto-detects THREE output layouts by shape:
  raw YOLO head `(1,5,8400)`, 6-col end2end `(N,6)=x1,y1,x2,y2,score,class`,
  and **7-col end2end `(N,7)=batch,x1,y1,x2,y2,class,score`** — the layout the
  CURRENT open-image-models export emits (their `postprocess` reads cols
  1:5/5/6). The old code only knew 6-col and IndexError-crashed the whole scan
  on 7-col output. INERT without a model file (or if neither backend loads it).
  Model resolution: `--plate-model` arg → `$OPENSCRUB_PLATE_MODEL`
  → `models/plate_yolov8.onnx` → `models/<registry-id>.onnx` (recommended
  first, adopts registry `input_size`).
- `detect_phi` — text-category detection over OCR line dicts. Word-loop
  order matters: card (Luhn-gated) before apikey before SSN. The `mrn`
  ID-number category is BRING-YOUR-OWN-REGEX: `--mrn-regex` defaults to
  EMPTY and the category is inactive without a pattern (empty must map to
  `mrn_re=None`, never `re.compile("")` — that matches every word).
  `RE_MRN_DEFAULT` survives only as the documented example.
- `PhiMemory`, `detect_phi` and other `phi`-named INTERNAL identifiers
  keep their names (report/API compatibility — do not rename). But ALL
  user-facing text — UI labels, tooltips, log lines, CLI help, README —
  says **"PII"** or generic wording, never "PHI": the tool is no longer
  healthcare-branded. Don't reintroduce healthcare terms in UI strings.
- `load_plate_registry` / `download_plate_model` — TOFU hash pinning: empty
  `sha256` in the registry → first download computes and PINS the hash back
  into `plate_models.json`; later downloads must match or are rejected and
  deleted. Registry reads/writes must always pass `encoding="utf-8"`
  (Windows defaults to cp1252 and mojibakes labels).
- Faces are ALWAYS dense (per-frame detection) whenever the face
  category is selected — on every kind of footage. Scan-cadence face
  adds merged with the OCR hold union a moving person's positions into
  one body-sized box (the boat-video failure); scan-time face adds are
  skipped whenever dense_faces is on. Face masks render as ELLIPSES by
  default (`--face-shape`, deface-style; blur_region/_blur_yuv10 take
  shape=), and `mosaic` is a real pixelation now (region-relative tiles),
  not a silent alias of blur.
- Scroll tracking + safety bands require TEXT categories: they exist to
  cover unscanned text, and on real-world video the tracker reads camera/
  subject motion as scrolling (edge-band smears, drifting boxes). Face/
  plate-only jobs force track_on off. For text jobs, camera vs screen:
  `probe_camera_motion` (--scroll-track auto) detects continuous 2-axis
  motion and disables scroll tracking + safety bands on camera footage. `assign_dense_tracks` groups the
  per-frame dense samples into tracks (`Detection.track`, additive report
  field) so review shows ONE card per physical object with a fan-out
  toggle (web `TRKMEM`), not hundreds of frames. Dense samples keep their
  sub-frame hold through `merge_detections` — stamping them with the OCR
  hold left a trail of stale boxes along the motion path (v1.0.21 bug).
  `smooth_dense_tracks` then makes each track leak-free: interpolates the
  box across detector-flicker gaps (≤0.75s) so the blur moves WITH the
  object, template-matches each track's first sample BACKWARD through the
  file to the object's true first visible frame (closes the onset leak
  before the detector's first hit), and adds 0.25s grace pads at both
  ends. Matching happens on 3x3-smoothed half-scale frames — sub-pixel
  motion decorrelates raw fine texture. `group_persons` then clusters
  face tracks by IDENTITY (SFace embeddings, auto-downloaded ~38MB like
  YuNet + baked into Docker images; YuNet landmarks for alignment).
  Three defenses against WRONG merges (validated on real crowd footage
  where single-link at 0.40 collapsed 83% of samples into ONE card):
  temporal cannot-link (tracks co-visible >0.5s never merge — different
  people by definition), centroid-linkage (match the cluster AVERAGE,
  no single-link chaining), embeddings only from re-detected faces
  ≥32px. Cosine 0.55 (same person ≈0.9; similar-age children measure
  0.4-0.6 apart, so 0.40 merged different kids):
  `Detection.person` (additive report field, survives load_report) —
  review shows ONE card per person, best-confidence thumbnail, one
  decision for all appearances. Ungrouped tracks stay person=-1 with
  per-track cards.
- Intake normalization (`normalize_vfr`): VFR input → CFR (`probe_vfr`);
  HDR input (`probe_hdr`: PQ/HLG transfer or BT.2020 10-bit) → tone-mapped
  SDR copy (zscale/tonemap, loud NOTE if ffmpeg lacks them) that the SCAN
  runs on. With `--hdr-output match` (default) a 10-bit CFR HDR source is
  kept (`args.hdr_source`/`hdr_encoder`/`hdr_tags`) and `run_render`
  dispatches to `render_hdr`: raw yuv420p10le pipe in/out, planar blur
  (`_blur_yuv10` — no colorspace conversion ever touches unblurred
  pixels), 10-bit HEVC out (`hevc10_encoder` ladder: hevc_nvenc →
  libx265 with a loud CPU-slowness NOTE → SDR fallback), source color
  tags + `hvc1`. SDR input NEVER produces HDR output. The `--from-report`
  path resets `args.video` to provenance `original_input` before
  normalize — rendering from the scan copy silently downgraded HDR jobs
  to SDR (the web render phase does exactly this). Dolby Vision RPUs are
  dropped by design (proprietary; HLG/HDR10 base layer survives). Report
  provenance records `hdr_tonemapped` + `hdr_output`.
- Pre-scan scoping: all scoping lives in the Scan Setup editor (below);
  the timeline has WHITE clip bookends (`--clip-start/--clip-end`; dims
  outside, full height) and detection windows.
  The web sends windows/trim as FRACTIONS of duration
  (`--detect-windows-frac`, `--clip-frac`; the server resolves them
  against its OWN ffprobe duration — iPhone HEVC/VFR reported a
  different length in-browser than the server measured, desyncing
  absolute seconds). CLI keeps `--detect-windows`/`--clip-start/end`
  (seconds). Windows still override skip fields;
  windows clamp inside the bookends, and bookends pushed inward DRAG
  window edges with them, and one lane per audio track with an M button
  (`--mute-audio-tracks "1,2"|"all"` — muted tracks are REMOVED from the
  output; audio_ffmpeg_args now carries ALL source tracks, not just the
  first, and span redaction applies per kept track). Dragging any handle
  scrubs the preview live (throttled 80ms). run_scan FAST-SKIPS decode
  outside the windows (seek past head, seek across gaps >1.5s, break
  after the last) but ONLY when scroll tracking is off — sequential
  offsets otherwise. Trim is SDR-only for now (HDR logs a NOTE and renders full
  length). Category ids are a compat surface but "mrn" DISPLAYS as
  "regex" everywhere (CATDN map in PAGE, DN maps in zones page).
- Unified Scan Setup editor (`/zones`, zones_ui.py): the ONLY intake
  path — the homepage is a jobs dashboard (New scan card → /zones, Jobs
  list, review/detail, settings incl. the Learned safe words card); it
  has NO upload form. One editor: video preview (local
  file objectURL or `/api/server_video?path=` — Range-aware send_file
  behind server_path_error), STACKED detection windows each on its OWN
  timeline lane (windows may overlap: faces 1.2–19.5s AND names 5–7s),
  and each window carries its OWN categories + zones (drawn on the
  frame; whole-frame when a checked category has no zones). Copy/paste
  zones between windows, clip bookends drag window edges inward
  (`clampWins`; windows <0.2s drop; the last window resets to
  whole-clip), audio lanes with M mute buttons, iOS prime + seek-queue
  live scrubbing. Serialization: startScan POSTs the normal /api/jobs
  FormData with `options.windows` = [{t0,t1 (FRACTIONS of duration),
  cats:[ids], zones:{cat:[normrects]}}] + `options.ignore_zones`;
  build_args writes them to `windows.json` in the job dir and passes
  `--windows`. Engine (`_prep_args` → run_scan): windows_px + `_scope_at(t)`
  computes the per-category UNION of covering windows — any unzoned
  covering window ⇒ unrestricted for that category; a category in NO
  covering window is dropped silently (win_inactive counter); window
  cats union into the engine load set, and merged window coverage feeds
  the fast-skip gating. Report provenance records `windows`. Upload uses
  XHR for progress %; `out_format` and every option the old homepage
  form carried lives in the editor's Advanced accordion. PAGE keeps
  `const CATS`/`CATDN` (review rendering + the alignment rule) even
  though the homepage checkbox grid is gone.
- Targeted redaction: `track_manual_region` template-tracks a user-drawn
  box through a chosen time window (both directions from t_ref, adaptive
  template refresh gated on confidence >0.80, stops fail-closed below
  thr). The web review's box editor has a TIMELINE (canvas `beTL`):
  detection lanes, draggable orange window handles (BE.win), playhead,
  and an audio lane. "Track object" POSTs /api/jobs/<id>/track_object
  (background thread, poll /track_status) which appends dense samples on
  a fresh track id, category "manual" — one review card with fan-out.
  `--categories none` = manual-only job (no engines load; report still
  gets render_state so review works).
- Audio redaction: report-additive `audio_redactions` [{t0,t1,mode}]
  (mode mute|bleep), written by review save and PRESERVED by
  write_report (the render-end rewrite must not drop them). CLI:
  `--audio-redact "a-b,c-d"` + `--audio-redact-mode`. `audio_ffmpeg_args`
  builds the ffmpeg args: no spans or no audio stream → stream copy;
  mute → gated volume filter + aac; bleep → filter_complex mixing a
  1 kHz tone (amix normalize=0). Both render() and render_hdr take
  `audio_spans`. Empty --mrn-regex style rule applies: spans only ever
  SILENCE, never guess.
- Install-location rules: `install_is_readonly()` (site-packages OR
  `sys.frozen`) switches every write path to `user_data_dir()`
  (%LOCALAPPDATA%/OpenScrub or ~/.local/share/OpenScrub): plate-model
  downloads, the TOFU-pinned registry (per-user copy seeded from the
  packaged one; new release models merge in, pins never overwritten),
  web jobs/certs/zones. Folder deploys keep writing next to the code.

OpenCV DNN device: the stock opencv-python wheel is CPU-only, so face
detection (YuNet/SCRFD/CenterFace) + SFace grouping run on the CPU.
`cuda_dnn_available()` (gated on `cv2.cuda.getCudaEnabledDeviceCount()>0`,
overridable via `OPENSCRUB_CPU_DNN=1`) drives `_make_yunet`/`_make_sface`/
`_apply_cuda_dnn`, which push those nets to `DNN_TARGET_CUDA` when a
CUDA-built OpenCV + GPU are present (the CUDA Docker image, once it FROMs
the opencv-cuda base). CPU builds are byte-identical to before — the GPU
path is purely additive. Video frame DECODE stays on the CPU either way.

Lazy loading: `run_scan` loads ONLY what the selected categories need —
`text_cats = cats - {face, plate}`. No text cats → no OCR engine, no
PhiMemory, detector-only scan (loud log line, plus the scan-count log
says "detector scans"); no `name` → no NameDetector/spaCy (`namer` is
None; `detect_phi` and the recall path guard for it). Keep new
text-pipeline features behind these gates.

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
- Detection models panel (face + plate): `/api/models/<kind>` (+
  `/<mid>/download`, `/download_status`, `/select`). Selection persists in
  `model_select.json` (data root); build_args passes the selected file as
  `--face-model`/`--plate-model` (missing file → engine default ladder).
  Old `/api/plate_models*` routes remain as aliases. A model shows a
  Download button only if its registry `download_url` is real. New models
  are picked up on the next job (detector instantiated per run).
- Server: cheroot with `BuiltinSSLAdapter` (self-signed or user cert), Flask
  dev server fallback if cheroot missing. `ssl_ctx` is a `(cert, key)` tuple.
- Self-update: `/api/update_check` (6h-cached PyPI poll, offline-silent),
  `/api/update_run` (409 while any job is queued/running), `/api/update_status`.
  Footer shows the version via the `%%VERSION%%` placeholder replaced at
  serve time in `index()` — never hardcode a version in PAGE again (the
  v4.2.0 footer went stale once already). Updates need a server restart.

## The 12-category alignment rule (easy to break!)

The category list exists in THREE places that must stay identical:
1. `openscrub.py` — argparse default `"name,dob,phone,ssn,mrn,email,address,card,apikey,ipaddr,plate,face"`
2. `openscrub_web.py` — `const CATS=[...]` in PAGE's JS
3. `zones_ui.py` — `const CATS={...}` color map in ZONES_PAGE

When adding a category, update all three + the `IMMEDIATE` set if it needs
no confirmation delay. Verify alignment:
`grep -o 'name,dob[^"]*' openscrub.py` vs the two JS lists.

USER-DEFINED categories (custom_categories.json in the data root) are
separate from this rule: managed on the Scan Setup page (add/remove via
/api/custom_cats; adding reloads the page because colors are injected
server-side, anchored on `face:"#ec4899"` in ZONES_PAGE — keep that
literal stable; customs land between it and the `ignore` pseudo-class).
They ride every job argv as `--custom-regex id=pattern`; the engine
activates only ids present in --categories.

The zones page also has an `ignore` pseudo-class (never-blur zones,
color #334155): NOT a detection category. Ignore zones are GLOBAL (not
per-window) and win any overlap with detection zones at the engine
level: `--windows` JSON carries them in its `ignore` key →
args.ignore_regions (always applied — independent of use_zones), and
in_ignore_region also gates dense faces/plates.

## Verification workflow (do this after every change)

```
python -c "import ast; ast.parse(open('openscrub.py').read())"   # each edited .py
# JS check: extract the <script> from the EVALUATED page, not the file text —
# PAGE is a normal (non-raw) Python string, so \n in source JS becomes a real
# newline when served and can break string literals (the v1.0.6 jobs bug):
#   python -c "import openscrub_web as w, re; open('/tmp/p.js','w').write(re.search(r'<script>(.*)</script>', w.PAGE, re.S).group(1))" && node --check /tmp/p.js
python -m pytest test_openscrub.py -q                             # 42 tests, all green
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
5. `.github/workflows/windows-installer.yml` builds the native installer on
   a Windows runner (same `windows\build_installer.bat` as a local build)
   and attaches `OpenScrub-Setup-<version>.exe` to the release automatically
   — no manual exe build or upload needed.

## Hard rules

- This is a privacy tool: **fail closed.** Over-blur beats under-blur;
  unverified models are rejected loudly, never run silently.
- No real personal data, names, or client/organization-identifying
  examples in any committed file. Test data is synthetic.
- Keep the human-review step prominent in any UX change; "best-effort
  redaction — always review output" is a product principle, not a disclaimer.
- Report JSON format is a compatibility surface (review UI + rehydration
  read it); extend, don't break.
- If `CLAUDE.local.md` exists in the working directory, read it — it holds
  private deployment context that must not be committed.

# CLAUDE.md — OpenScrub

OpenScrub is a local, GPU-accelerated video redaction tool: it detects and
blurs PII (faces, names, SSNs, addresses, license plates, full-body person blur — 12 default categories) in
videos and screen recordings, with a human-review step before anything is
trusted. Apache-2.0. Python 3.10+. Runs on Windows/Linux; primary dev/deploy
target is Windows 10 + NVIDIA RTX 3060.

## File map

| File | Role |
|---|---|
| `openscrub.py` | The engine. CLI + all detection/render logic. Single file, ~2400 lines. |
| `openscrub_web.py` | Flask web app. The entire UI is one embedded `PAGE` string (HTML/CSS/JS). Serves via cheroot (production WSGI, TLS) with Flask-dev fallback. |
| `zones_ui.py` | The app SHELL (`ZONES_PAGE`): dark theme, header (gear → settings), and the Scan Setup editor — load a video, stack detection windows on a timeline (one lane per window, overlap allowed), per-window categories + zones, copy/paste zones, audio mute lanes, clip bookends, Start scan. Contains `%%MARKER%%` slots that openscrub_web.py fills at import to build the single-page app. |
| `openscrub_gui.py` | Legacy Tk GUI. Frozen; ships but is not actively developed. |
| `openscrub_setup.py` | `openscrub-setup` command: detects/installs Tesseract + FFmpeg (winget/apt), optional spaCy model + plate model, Windows Start Menu shortcuts. Ships in the wheel. |
| `windows/` | Native Windows packaging: `openscrub.spec` (PyInstaller, two branded exes), `installer.iss` (Inno Setup → Program Files), `build_installer.bat` (runs both; attach output exe to the GitHub release). Build on Windows only. |
| `install.py` | Windows-friendly installer (deps, GPU OCR, shortcut, `--with-plates`). |
| `docker/` | `Dockerfile.opencv-cuda` builds the CUDA-OpenCV base image (rare, via opencv-cuda-base.yml); `Dockerfile.cuda` FROMs it (base published to ghcr; bump the FROM tag only when the base is rebuilt). |
| `Dockerfile.intel` | `:intel` image (amd64): Debian non-free iHD media driver + libmfx/libvpl for QSV encode, `onnxruntime-openvino` replacing stock onnxruntime (build asserts the provider registered). Needs `--device /dev/dri`; everything falls back loudly to CPU without it. |
| `plate_models.json` | Curated license-plate model registry (see PLATES.md). |
| `face_models.json` | Curated optional face-model registry (CenterFace/SCRFD); built-in YuNet needs no file. Ships everywhere plate_models.json does (wheel, sdist, Dockerfiles, PyInstaller spec, updater pin-carry). |
| `person_models.json` | Curated person-model registry (YOLOv10 ONNX from onnx-community on HF, AGPL — download-only, never bundled; hashes PRE-pinned, validated at authoring). Ships everywhere plate_models.json does. |
| `fetch_plate_models.py` | Alt path to fetch plate models via the open-image-models pip package. |
| `openscrub_update.py` | `openscrub-update` command + web self-update backend: PyPI version check, sha256-verified sdist download, data-preserving folder update (PRESERVE set), TOFU pin carry-forward. Ships in the wheel. |
| `openscrub_vault.py` | At-rest encryption for the job store: scrypt keystore, chunked AES-256-GCM files (`.osvault`), lock/unlock tree walkers. NO password reset by design. Ships in the wheel. Lock-on-shutdown lives in openscrub_web: a SIGTERM handler (docker stop; locks then os._exit — sys.exit is swallowed by cheroot) + an atexit hook (Ctrl+C; uses the import-time `_HERE` constant because `__file__` is gone during interpreter teardown — both failure modes were real and verified). Encryption must finish inside the container stop grace period (`docker stop -t 120`). |
| `test_openscrub.py` | pytest suite (52 tests). Must stay green. |
| `deploy/` | App-store submission kit: winget/CasaOS/Runtipi/TrueNAS/Umbrel/Portainer/CapRover/Coolify manifests + a novice-friendly submission guide (deploy/README.md). Platforms with their own reverse proxy get `--http` in the command; direct-port platforms keep default TLS. Bump pinned versions at submission time. |
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
- `PersonDetector(PlateDetector)` — full-BODY person masking ("person"
  category): a face blur hides the face but clothing/build/gait still
  identify someone. Same dual-backend YOLO ONNX machinery as plates with
  `WANT_CLASS=0` (multi-class COCO outputs filtered to person; end2end
  6/7-col rows carry class in col 5, raw v8 heads put class-0 score at
  row[4] — the SAME channel single-class models use, so plate decode is
  byte-identical). SILHOUETTES, not boxes: a YOLO -seg model (detected by
  its second (1,32,160,160) prototype output) makes find() return a 6th
  element — body contour polygons normalized to the box
  (`_decode_seg`: sigmoid(coeffs @ protos), crop, threshold 0.5,
  approxPolyDP) — stored on `Detection.poly` (report-additive) and
  rendered by `blur_silhouette` which masks ONLY inside the contours.
  The seg mask is cropped to the EXPANDED detection region, not the
  tight box — the tight crop clipped body parts the mask knew about (a
  real dog's snout stayed unblurred at the box edge during an
  occlusion pass).
  Seg models ALWAYS run on onnxruntime: OpenCV DNN's layer fusion
  asserts AT INFERENCE TIME on multi-output graphs on some builds
  (fuseLayers, 4.10 CUDA) — the load probe can't catch it and it killed
  a whole scan once; a residual cv2 forward failure degrades LOUDLY to
  box masks instead of crashing. Silhouettes are masked in BOTH renders
  — `blur_silhouette` (SDR) and `_blur_silhouette_yuv10` (HDR: mask
  rasterized at luma res, resized NEAREST for the half-res chroma
  planes; untouched pixels never leave the native YUV domain); pad
  becomes an outward mask dilation and degenerate masks fall back to
  the full box, fail closed.
  Detection-only models still work and blur the box. ALWAYS dense
  (per-frame, like faces), feeds the same assign/smooth track pipeline —
  smoothing's `_mk` copies `poly` from the ref sample so interpolated/
  onset samples keep their silhouette — → ONE review card per tracked
  body ("person (full body)"; SFace grouping stays face-only). INERT
  without a model (exactly like plates); `--person-model` →
  `$OPENSCRUB_PERSON_MODEL` → `models/person_yolov8.onnx` → registry
  ids. `--person-threshold` default 0.5. Registry models are YOLO11n-seg
  / YOLOv8n-seg ONNX (HF mirrors, PRE-pinned sha256, validated on real
  footage at authoring) — AGPL-3.0, registry-download only, never
  bundled.
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
  toggle (web `TRKMEM`), not hundreds of frames. Association hardening
  (boat-video bug): a track absorbs at most ONE sample per timestamp
  (co-temporal detections are different objects — same cannot-link idea
  as face grouping), and person tracks associate within 0.5x box size
  (1.6x is face-calibrated; on a full-body box it spanned the frame and
  merged different people into one card). Dense samples keep their
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
  runs on. The 1:1 (non-CFR-resampling) tonemap copy is VERIFIED after
  creation (`_scan_copy_matches`: full-decode frame counts, tol 2) — the
  scan reads it by frame INDEX, and a GPU encode that drops frames
  silently shifts every later detection onto the wrong output frames
  (misplaced blur beside the subject; a real P2000/Unraid NVENC copy did
  this while the render's own timeline stayed perfect). Mismatch → the
  lying copy is DELETED (never left for the mtime cache), redone with
  libx264, re-verified; still off → RuntimeError, refuse to scan.
  Unmeasurable counts never block (guard acts only on a measured lie). With `--hdr-output match` (default) a 10-bit CFR HDR source is
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
  length). Category ids are a compat surface; legacy "mrn" detections
  DISPLAY as "regex" in review (CATDN map).
- Unified Scan Setup editor (`/zones`, zones_ui.py): the ONLY intake
  path — and the editor IS the homepage: one page with the editor on
  top, Jobs + job detail/review below it (`#appzone`), settings behind
  the header gear (`#settings` hash; incl. the Learned safe words
  card). One editor: video preview (local
  file objectURL or `/api/server_video?path=` — Range-aware send_file
  behind server_path_error), STACKED detection windows each on its OWN
  timeline lane (windows may overlap: faces 1.2–19.5s AND names 5–7s),
  and each window carries its OWN categories + zones (drawn on the
  frame; whole-frame when a checked category has no zones). Copy/paste
  zones between windows, clip bookends drag window edges inward
  (`clampWins`; windows <0.2s drop; the last window resets to
  whole-clip), audio lanes with M mute buttons — thick blue bars with a
  WAVEFORM through them (local files: in-browser WebAudio decode, 600MB size
  cap; `buildWave` retries the WHOLE construct+decode at each rate of an
  8k→48k ladder — pre-2024 iOS Safari throws CONSTRUCTING below 22050,
  newer WebKit constructs 8k fine but can fail the DECODE into it, and
  decodeAudioData detaches its buffer so each try gets a `buf.slice(0)`;
  `decodeBuf` feeds both callback and promise decodeAudioData forms
  (old-Safari error callback may pass null). iOS Safari refuses to demux
  VIDEO containers in decodeAudioData entirely (audio files only — every
  ladder rate fails on an iPhone .mov even though `<video>` plays it), so
  Plan B kicks in: `demuxMp4Aac` walks the MP4/QuickTime boxes in JS
  (moov→trak soun→stsd mp4a, esds byte-scan for the ASC, stsz/stsc/stco+
  co64/stts sample table walk — byte-exact vs ffprobe on moov-at-end AND
  faststart layouts), then `wavePlanB` tries `waveViaWebCodecs` (WebCodecs
  AudioDecoder, per-chunk peaks binned to 2000) and, because real iPhones
  ship WITHOUT AudioDecoder, falls back to `adtsFromAac`: rewrap the raw
  AAC frames as an ADTS stream (7-byte headers from the ASC) — a pure
  AUDIO payload decodeAudioData DOES accept — and run the rate ladder on
  that (ffmpeg round-trip-validated; WebKit e2e with iPhone-exact mocks).
  Non-AAC audio tracks are skipped (spatial-audio APAC first track on
  newer iPhones) and fail with a loud "unsupported audio codec (xxxx)"
  if no AAC track exists. Failures are NOT silent:
  `S.waveBusy`/`S.waveErr` render "analyzing audio…" / the error name on
  the lane (light text, vertically centered) + console.warn — a flat bar
  with no message means genuinely silent audio; browsers demux only the
  default track, extra local lanes stay flat. Server paths: per-track
  /api/waveform, ffmpeg s16le 1kHz → 2000 normalized peaks, same inline
  containment guard as server_video) — a timeline ZOOM bar
  (−/slider/+ to 40×, log scale, plus a pan slider; `tx` clamps for
  drawing while `txr` is the raw mapping used for handle hit tests so an
  off-view handle can't be grabbed at the clamped edge; ruler ticks
  adapt to the visible span down to 0.5s), iOS prime + seek-queue
  live scrubbing. Categories default ALL OFF (every new window too —
  nothing is detected until the user checks a category or draws a zone;
  the summary line under Start shows an amber "nothing will be detected"
  warning while empty, and startScan confirms manual-only). Serialization:
  startScan POSTs the normal /api/jobs
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
- Targeted redaction: `track_manual_region` tracks a user-drawn box
  through a chosen time window (both directions from t_ref). DETECTOR
  path first: when a person model is available (both callers resolve
  one — run_scan reuses `personer` or loads a dedicated det at
  thresh=0.2, the drawn box being a strong prior; the web track_object
  endpoint resolves the panel selection), `_track_person_dense` decodes
  EVERY COCO class (`det.want_any=True` — the "person" registry models
  are 80-class COCO: bird=14, sports ball=32…; `_decode`/`_decode_seg`
  gain argmax-class mode + class-aware NMS via the coordinate-offset
  trick so a ball held by a person isn't suppressed; rows grow (poly,
  cls) tails, `COCO_NAMES` maps ids), seeds on the detection best
  overlapping the drawn box (IoU + covered-fraction >= 0.15; no match →
  generic tracker) and follows THAT object with per-frame detections of
  the SAME class only: detector-tight boxes + silhouette polys (5-tuple
  samples (t,box,score,poly,cls) vs the generic 3-tuples; review text
  "tracked <classname>"; Detection.poly → blur_silhouette — validated
  frame-by-frame on real footage where it never jumped to an adjacent
  person). Association is PROPAGATE-AND-REFINE (the MaskAnyone
  principle sized to our no-torch stack: their SAM2 propagate_in_video
  carries identity by propagation, not per-frame matching; here a
  VitTrack PROPAGATION ANCHOR at half scale carries identity while
  detections REFINE with tight silhouettes — detections stay the mask
  source, the anchor never is). Identity chains to the LAST CONFIDENT
  DETECTION: a same-class detection is accepted only with IoU >= 0.10
  vs the current box AND area within 0.33-3.0x of `ref_a`, a slow-shrink
  size EMA (grow α=0.5, shrink α=0.05 — an occlusion sliver must not
  drag it down or the full-size re-appearance gets rejected; ended a
  real track mid-window). SUSPICIOUS-HANDOFF guard: weak overlap
  (IoU<0.30) AND much larger (>1.8x ref_a) needs anchor confirmation
  (vscore>=0.35 ∧ IoU(det,vbox)>=0.25) — the classic look-alike-slides-
  past handoff at a frame exit. The anchor itself is SANITY-CHECKED: a
  vbox outside the same 0.33-3.0x size band has DRIFTED and is
  discarded (on real footage it ballooned to a near-frame box over the
  second dog after the subject left and "confirmed" the wrong handoff).
  On a detection hit: emit the detector-tight box + poly EXACTLY as
  detected (blur only what is visible — an earlier predicted-union
  produced frame-spanning boxes and floating ghosts), update ref_a,
  RE-ANCHOR VitTrack on the detection (drift can never accumulate past
  one blink). On a detector miss with the anchor still AGREEING
  (vscore>=0.35 ∧ IoU(vbox,cur)>=0.20): ride the anchor — but only
  while the detector gap <= 0.8s (rides bridge BLINKS; an unbounded
  ride carried a box around the frame long after the subject left —
  the floating-blur failure). Rides paint the LAST live silhouette
  stretched over the ridden box (+6% for staleness), not a bare
  rectangle — the object is visible, the detector just blinked, and a
  block blur over a visible subject reads as "gave up". FRAGMENTS: a
  small same-class detection (<0.33x ref_a) mostly inside the held box
  (covered-fraction >= 0.5) is the object PEEKING past an occluder —
  it never touches identity/ref_a/anchor, but it IS masked: the sample
  becomes union(held box, fragment) with the fragment's silhouette
  plus the held box as mask regions (snug on the visible part, covered
  on the uncertain part), and it resets the loss timers because the
  object is visibly there. Both blind, or past a blink's length:
  FREEZE the last box CLAMPED to the frame (score 0, NO stale poly —
  full-box blur) for a 0.8s grace, then go DORMANT (not end): stop
  emitting samples (nothing visible → nothing to blur) but keep
  scanning the window for the subject's re-emergence. Re-acquisition
  needs a same-class detection in the 0.33-3.0x size band that passes
  an HSV-histogram appearance check (`_crop_hist`, HISTCMP_CORREL >=
  0.35, fingerprint refreshed on every live accept) AND is not a
  BYSTANDER — every same-class object visible at the moment ours went
  hidden is tracked forward by per-step best-IoU and can never inherit
  the track (cannot-link: co-visible ⇒ different object; this is what
  keeps the second dog out — on the real footage the dormant track
  ignored the foreground dog for 1.5s then re-acquired the returning
  seeded dog at 19.1s with boxes matching the known-good W2 track to
  0px). Frame exits go DORMANT too — the frame edge is just another
  occluder (a real subject walked out of frame and returned unblurred;
  validated on that footage: coverage resumed on her return while the
  bystanders, one walking right up to the camera, stayed unblurred).
  Detector-path tracks end only at window edges now; the GENERIC
  (no-model) tracker still ends at frame exits. NO anticipation
  or occluder modeling — waiting beats predicting (prediction caused
  two real regressions); this is SAM2's occlusion recipe sized to our
  stack: remember appearance, re-identify on re-appearance.
  test_track_dormant_reacquisition pins gap-silence + bystander
  exclusion + resumption. NO nearest-distance
  fallback and NO velocity prediction, ever (both caused real
  wrong-object/floating regressions). Two near-identical subjects: association
  stays on the SEEDED one; an unselected look-alike is never grabbed
  (validated frame-by-frame on two-street-dog footage: full-window
  coverage through a person-occlusion at ~7s, zero wrong-dog samples,
  track ends at the frame exit instead of inheriting the other dog). Seeding
  survives bad frames: detection is retried at
  t_ref±(0.33..1.0)s inside the window, and ALL tracker frame fetches go
  through `_grab_frame` (module-level), which NEVER TRUSTS A SEEK: after
  every cap.set the decoded packet's own PTS (CAP_PROP_POS_MSEC after
  read — from the bitstream, can't lie) is compared to the request; an
  early landing is repaired by decoding forward, a late one by seeking
  earlier, ultimately frame 0 + sequential — the render's access
  pattern, correct on any decodable file. TWO real seek pathologies
  demanded this: (1) deep POS_MSEC seeks FAIL outright on some builds
  (h264_nvenc HDR copies in the CUDA image returned nothing near 19.9s
  — silently failed a window's seed); (2) on another real box seeks
  SUCCEED but land at an earlier keyframe — a 20.3s seek landed ~16s
  early, the tracker followed the wrong moment of the video with full
  confidence, and the rendered blur "flew away" off the subject (report
  samples matched the clean run's boxes at t−16.3s to 0-2px — the
  smoking gun). `_seek_cap` (positions a cap so the NEXT read is frame
  N, same verification) guards run_scan's fast-skip jumps, the trimmed
  render's clip-start seek, and every other cap.set site (SFace embed
  grabs, onset walk-back `_gray`, recall `_visible`, gap-bridge
  `_frame_small`). test_grab_frame_survives_broken_random_seek +
  test_grab_frame_repairs_keyframe_snapped_seek pin both pathologies. The
  detector path samples EVERY frame (step_frames=1): masks repainted at
  a 2-frame cadence visibly STEP/flicker at 15Hz on real footage.
  Renders (SDR + HDR) apply `_dedupe_dense` per frame: dense-track
  snapshot spans carry a 1.2x grace overlap, so every other frame was
  covered by TWO snapshots of one track and got double-blurred — a
  15Hz intensity pulse (the user-visible flicker); only the
  latest-started snapshot per track may apply. LIVE-
  FRACTION GATE: if the detector saw the object in <50% of steps (a
  marginal class at 0.2 conf flickers), the whole attempt is DISCARDED
  for the generic tracker — frozen coverage must not beat live pixel
  tracking (a synthetic ball hit exactly this).
  GENERIC path (unrecognized objects, no person model): cv2.TrackerVit
  (OpenCV >= 4.9) + the opencv_zoo vittrack model
  (~0.7MB, Apache-2.0, auto-downloaded via `_fetch_model` with pinned
  VITTRACK_SHA256, baked into both Docker images) — survives scale
  change/turning/appearance drift that killed the old template matcher
  in ~1s on real footage (subject walking toward the camera; VitTrack
  held the full 26s clip). Forward direction reads frames SEQUENTIALLY
  (no per-step seeks; 33s→5s on that clip). Score gates: >=0.30 accept;
  0.15-0.30 emit the UNION of tracker box and last confident box (loose
  OR drifting — cover both); <0.15 FREEZE the box. Coverage ends early
  only when the box CENTER leaves the frame (a mid-dip box can balloon
  to touch every edge while the object is still centered — real
  footage). FALLBACK engine (old OpenCV/offline): multi-scale template
  matching (0.93/1.0/1.075 per step, cumulative scale clamped 0.4-3.0x,
  1.5x search margin, refresh gated >0.80). BOTH engines FAIL CLOSED on
  loss: the box freezes and coverage continues to the window edge with
  score 0.0 (the old stop-on-loss silently UN-blurred the subject
  mid-clip — a real user hit this); regression test
  test_track_fail_closed_hold. TWO entry points share it: PRE-SCAN — the editor's `trackobj`
  pseudo-class (yellow; own rail row like ignore; skip both in the
  renderCats loop) stores [nx1,ny1,nx2,ny2,tref_frac] per window
  (`finishRect` captures vd.currentTime as tref; select/move preserves
  the 5th element via `listFor`+slice(4)), rides `--windows` JSON as the
  window's `track` key, and run_scan's tail runs the tracker per box
  after assign/smooth (fresh track ids past n_tracks, dense "manual"
  samples, 0.25s grace pads — same recipe as the review endpoint).
  POST-SCAN — review's "Track object" (below). The web review's box editor has a TIMELINE (canvas `beTL`):
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

Encoders are a pre-flight-tested LADDER (`nvenc_available`: h264_nvenc
→ h264_qsv → libx264; `hevc10_encoder`: hevc_nvenc → hevc_qsv →
libx265; `--encoder auto|nvenc|qsv|x264`) — every GPU rung must pass a
tiny real test encode before being trusted, so a missing driver
degrades loudly to CPU. All four vargs sites (render, render_hdr,
normalize_vfr sdr copy + VFR-HDR intermediate) carry qsv branches.
onnxruntime sessions come from `_ort_session`: CUDA > OpenVINO
(AUTO:GPU,CPU device, with a plain-provider retry for option-name
drift) > CPU; `OPENSCRUB_CPU_DNN=1` forces CPU.

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

- Mobile fit rule: the page must NEVER scroll horizontally. Every CSS
  grid track is `minmax(0,1fr)` (never bare `1fr` — the implicit `auto`
  minimum lets one wide child, e.g. a `white-space:pre` select in the
  Advanced grid, blow the track past a phone viewport; this was a real
  56px overflow on iPhone), `select{max-width:100%}`, and
  `html,body{overflow-x:clip}` as the regression guard. Verified at
  320/375/390/430px across editor + advanced + settings views.
- ONE page: `PAGE` is COMPOSED at import time — `zones_ui.ZONES_PAGE` is
  the shell (Scan Setup editor on top) and openscrub_web.py fills its
  `%%MARKER%%` slots: `APP_CSS` (dark restyle of the app sections),
  `JOBS_HTML` (jobs list + `#detail`), `SETTINGS_HTML` (settings view,
  shown via `#settings` hash + header gear), `FOOT_HTML`, and `APP_JS`
  (jobs/review/settings JS as a SECOND `<script>` — both scripts share
  the global lexical scope, so top-level `const` names must be unique
  across the two; the box editor's timeline painter is `beTLDraw`
  because the editor owns `tlDraw`). `/zones` redirects to `/`. The
  header logo/favicon are **base64-embedded** (`LOGO_URI`/`WORDMARK_URI`/
  `FAVICON_URI` constants) so they can never 404. `assets/` still ships
  for GitHub/social use. The app sections are wrapped in `#appzone`/
  `#settingsview` — APP_CSS scopes its button/input/h2 overrides to
  those ids so editor styling stays untouched.
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
  `RedirectingTLS` peeks the first byte to 301 plain-HTTP → https, but the
  peek MUST be non-blocking (`select` 0.1s) and the handshake bounded
  (`settimeout(4)`): cheroot calls `ssl_adapter.wrap` on its SINGLE accept
  thread, so a blocking peek there wedges ALL new connections — normal
  Safari's speculative preconnects starved the pool and the page reload-
  looped (Private mode has no preconnect, so it worked). numthreads=64 for
  headroom past one browser's ~6 connections. HTML is served `no-store`
  (not just `no-cache`) so iOS Safari can't run stale inlined JS after an
  update — a fix that only worked in Private mode was the tell.
- Self-update: `/api/update_check` (6h-cached PyPI poll, offline-silent),
  `/api/update_run` (409 while any job is queued/running), `/api/update_status`.
  Footer shows the version via the `%%VERSION%%` placeholder replaced at
  serve time in `index()` — never hardcode a version in PAGE again (the
  v4.2.0 footer went stale once already). Updates need a server restart.

## The 12-category alignment rule (easy to break!)

The category list (12 now) exists in TWO places that must stay identical:
1. `openscrub.py` — argparse default `"name,dob,phone,ssn,email,address,card,apikey,ipaddr,plate,face,person"`
   ("mrn" is RETIRED from the defaults and from the whole UI — regex
   detection is custom-categories only now. The engine machinery stays:
   CLI `--categories ...,mrn --mrn-regex PAT` still works, review CATDN
   still displays legacy "mrn" detections as "regex", and BUILTIN_CATS
   still reserves the id so a custom category can't claim it.)
2. `zones_ui.py` — `const CATS={...}` color map in ZONES_PAGE (the app's
   only JS category list since the homepage checkbox grid was retired;
   `CATDN` in APP_JS keeps display names for review headings)

When adding a category, update both + the `IMMEDIATE` set if it needs
no confirmation delay. Verify alignment:
`grep -o 'name,dob[^"]*' openscrub.py` vs the CATS map keys. In the
CATS map `person` sits BEFORE the `face:"#ec4899"` anchor — customs still
land right after face.

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
#   PAGE now holds TWO <script> blocks (editor + app) sharing one global
#   scope — join them so duplicate top-level declarations are caught too:
#   python -c "import openscrub_web as w, re; open('/tmp/p.js','w').write('\n'.join(re.findall(r'<script>(.*?)</script>', w.PAGE, re.S)))" && node --check /tmp/p.js
python -m pytest test_openscrub.py -q                             # 51 tests, all green
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

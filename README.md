<p align="center">
  <img src="https://raw.githubusercontent.com/austinmabry/OpenScrub/main/assets/social_preview.png"
       alt="OpenScrub — the bucket-hat figure behind a fence, face pixelated, next to the OpenScrub wordmark"
       width="830">
</p>

<p align="center">
  <a href="https://pypi.org/project/OpenScrub/"><img src="https://img.shields.io/pypi/v/OpenScrub?color=1f2a44&label=PyPI" alt="PyPI version"></a>
  <a href="https://pypi.org/project/OpenScrub/"><img src="https://img.shields.io/pypi/pyversions/OpenScrub?color=1f2a44" alt="Python versions"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-e0432f" alt="License: Apache-2.0"></a>
</p>

# OpenScrub — local video & screen-recording redaction

**A local, GPU-accelerated tool that blurs faces, whole people
(silhouette-precise body masking), license plates, and any on-screen text —
names, phone numbers, SSNs, emails, dates, or anything you can express as
a regex — in videos and screen recordings, with a human review step before
anything is published.**

Runs entirely on your own machine (no cloud, no upload of sensitive
footage). OCR-driven, so it catches text anywhere on screen; face-tracked,
so a face detected once stays covered even when the detector blinks;
scroll- and motion-aware, so blur boxes follow content as it moves; and
onset-aware, so redaction starts on the exact frame a detail first appears
rather than a half-second late. Defaults are tuned for the hardest case —
dense, scrolling records screens (medical, financial, CRM, support
consoles) — but the engine is general-purpose.

> Keywords: video redaction · blur faces in video · blur a person in video ·
> body blur / silhouette masking · redact screen recording · PII redaction ·
> anonymize video · blur license plates · GDPR / CCPA / FERPA / PCI / HIPAA ·
> OCR text redaction · face blur · privacy tool

## What it does

- **Blur faces** — detected with a DNN model and visually tracked, so a
  single detection covers a face across frames where it would otherwise
  be missed. Works out of the box; optional higher-accuracy models
  (CenterFace, SCRFD) can be downloaded and selected in the web UI's
  Detection models panel — every download is hash-verified.
- **Blur whole people — silhouette-precise** — the `person` category
  detects full bodies and masks each person's **exact body outline**
  (blur, black fill, or mosaic follows the silhouette; the background
  around them stays sharp). A face blur hides a face, but clothing,
  build, and gait still identify someone — this hides the person.
  Enabled by downloading a segmentation model in the web UI's Person
  model panel (hash-verified, ~11 MB); each tracked person gets their
  own review card.
- **Blur license plates** — via an optional ONNX detector model
  (see [PLATES.md](PLATES.md)); plates re-detect every frame, so a plate
  crossing the frame stays covered.
- **Redact text by pattern** — built-in patterns for SSNs, phone
  numbers, emails, dates, addresses (including multi-line
  street/city/state/ZIP blocks), credit/debit card numbers
  (Luhn-validated), API keys/tokens, and IP addresses — plus **custom
  regex categories**: add your own (claim numbers, case IDs, employee
  IDs, account formats) in the web UI and each one becomes a first-class
  category with its own color, zones, and review section.
- **Redact names** — via named-entity recognition plus heuristics, with no
  list required (though you can supply an allowlist to *keep* specific names
  visible and a blocklist to *always* remove others).
- **Detection windows & zones** — scope the scan in time AND space
  before it runs: stack detection windows on a timeline (faces for the
  whole clip *and* names only from 5:00–7:00), draw per-window zones on
  the frame, and mark never-blur ignore zones. Clip bookends trim the
  output; audio tracks can be muted per-track.
- **Redaction styles** — blur, solid black box (irreversible), or mosaic
  pixelation, choosable per category (black-box the SSNs, blur the faces).
  Faces are masked with a tight **ellipse** by default — no smeared
  rectangle corners — and mosaic tiles scale with the face size.
- **Human review** — every detection is shown as a thumbnail you can
  keep or blur — one card per tracked face/person/plate, one decision for
  all its appearances — with an interactive box editor to resize, move,
  add, time-bound, or **template-track a manually drawn box** through a
  chosen time range, and audio mute/bleep spans for spoken PII.
- **HDR in, HDR out** — iPhone (Dolby Vision/HLG) and HDR10 footage keeps
  its 10-bit HDR signal through redaction: output is 10-bit HEVC with the
  original color primaries and transfer preserved, and the blur is applied
  directly in the native color domain (untouched pixels are never
  color-converted). Prefer compatibility instead? One toggle tone-maps the
  output to SDR properly — no washed-out colors either way. SDR sources
  always render SDR: output matches the source.
- **Audit trail** — each run produces a report with SHA-256 hashes of input
  and output for provenance.

## Use cases

The tool is built for the hardest version of the problem — dense,
scrolling, fast-changing screens full of sensitive text — and the same
engine fits many privacy workflows:

- **Content creators & tutorials** — strip inboxes, browser tabs, API keys,
  file paths, and notification pop-ups out of screen-recorded walkthroughs
  before publishing. Onset detection catches a notification that flashes for
  a fraction of a second.
- **Legal e-discovery & court exhibits** — redact names, SSNs, account
  numbers, and minors' identities from video exhibits, deposition
  recordings, and bodycam footage; keep the named party visible, blur the rest.
- **Journalism & documentary** — protect sources by blurring informants'
  faces and on-screen identifying text (badges, plates, addresses,
  shared documents); face tracking survives a source turning their head.
- **Law-enforcement & FOIA release** — redact bystander faces, minors, and
  visible PII from body-cam footage for public-records compliance, with a
  hashed audit report for chain-of-custody.
- **GDPR / CCPA compliance** — anonymize identifiable people and personal
  data in any recorded video before it's shared or published.
- **Fintech & financial support videos** — blur account numbers, balances,
  routing and card numbers in screen recordings of banking or accounting
  software (PCI-DSS-relevant), via built-in and custom regex.
- **Education & research (FERPA)** — protect student names, IDs, and grades
  in recorded lectures, gradebook screen-shares, and classroom video;
  "blur every face except the presenter" with an inverted zone.
- **Corporate training & internal demos going public** — remove real
  customer names, employee directories, internal URLs, and chat
  notifications when repurposing production-system recordings for marketing.
- **Real-estate & property walkthroughs** — blur faces, family photos,
  mail with addresses, and documents on desks captured incidentally.
- **Streaming & gaming VODs** — redact Discord DMs, donation alerts with
  real names, second-monitor leaks, and non-consenting on-cam guests.
- **Dashcam footage** — blur license plates and pedestrian faces before
  posting insurance or public clips (plates via the dedicated detector
  model — see [PLATES.md](PLATES.md)).
- **UX & usability research** — anonymize participants' faces, names, and
  on-screen account data before sharing session recordings internally.
- **Government document-on-screen redaction** — remove names, locations,
  and marked strings from recordings that walk through sensitive documents,
  with zones, regex, and an audit trail.

Several of these depend on you supplying the right regex, and on OCR
reading the target text reliably at your recording's resolution — see
[Caveats](#caveats--read-these). Face tracking works best on footage where
faces are stationary or scroll with the page rather than moving rapidly
across the frame. In all cases this is a **best-effort assistive tool:
review the output before publishing.**

## Install with Docker (recommended)

The most complete OpenScrub install: Tesseract, FFmpeg, spaCy NER, and
the face model are all preinstalled — nothing to set up, and updates are
a `docker pull` away. Every release publishes identical images to
**Docker Hub** ([`pharmhero/openscrub`](https://hub.docker.com/r/pharmhero/openscrub))
and GitHub Container Registry (`ghcr.io/austinmabry/openscrub`):

```
docker run -d -p 8384:8384 \
  -v openscrub_data:/root/.local/share/OpenScrub \
  pharmhero/openscrub:latest
```

(or `ghcr.io/austinmabry/openscrub:latest` — use whichever registry
pulls faster for you). Published tags are refreshed **weekly** with the
latest OS security patches, not just at releases.

Tesseract, FFmpeg, and the face model are baked in; jobs, certificates,
zones, and downloaded plate models live in the mounted volume, so the
container is disposable. Add `--token <secret>` after the image name
(as `openscrub-web --host 0.0.0.0 --token <secret>`) for access
control. The "process a file already on the server" box accepts any
video path the process can read; set `OPENSCRUB_MEDIA_ROOT=/path` (env
var) to confine it to one directory tree — recommended whenever the web
UI is reachable by anyone but you. To update, pull the new tag and recreate the container — the
in-app updater doesn't apply inside Docker. Both images include
spaCy NER (name detection) out of the box; the default image is
CPU-only. If you use **Encryption at rest**, stop the container with a
generous grace period (`docker stop -t 120`) so the shutdown lock has
time to encrypt large job stores.

**NVIDIA GPU build** (`:cuda` / `:<version>-cuda`) — CUDA-accelerated
PaddleOCR and NVENC hardware encoding:

```
docker run -d --gpus all -p 8384:8384 \
  -v openscrub_data:/root/.local/share/OpenScrub \
  pharmhero/openscrub:cuda
```

On **Unraid**: install the Nvidia Driver plugin, add a container with
repository `pharmhero/openscrub:cuda`, extra parameter
`--runtime=nvidia`, port 8384, and map
`/root/.local/share/OpenScrub` to `/mnt/user/appdata/openscrub`.
GPU features engage automatically (the OCR engine picks the CUDA build,
and the render's NVENC test selects hardware encoding). Note the GPU
image is several GB.

## Install from PyPI

```
pip install OpenScrub
```

This installs the Python package with all of its Python dependencies and
gives you two commands: `openscrub` (the CLI engine) and `openscrub-web`
(the web interface). The YuNet face model (~230 KB) downloads
automatically on first run.

### Fresh Windows 11 PC — complete setup (copy-paste)

```
winget install -e --id Python.Python.3.12
```
Close and reopen the terminal (so PATH updates), then:
```
pip install OpenScrub
openscrub-setup
```

`openscrub-setup` detects what's missing and installs Tesseract and FFmpeg
for you via winget (asking first; `--yes` to skip prompts, `--check` to
only report). Prefer manual control? The equivalent commands:

```
winget install -e --id UB-Mannheim.TesseractOCR
winget install -e --id Gyan.FFmpeg
```

### Fresh Linux (Debian/Ubuntu)

```
sudo apt install python3-pip
pip install OpenScrub
openscrub-setup        # offers: sudo apt install tesseract-ocr ffmpeg
```

Then run `openscrub-web` and open the URL it prints.

Two system tools are **not** pip-installable and must be present for full
functionality:

1. **Tesseract OCR** — required for every text category (names, SSNs,
   emails, …). Face and plate detection work without it; text detection
   does not.
   - Windows: `winget install -e --id UB-Mannheim.TesseractOCR`
     (or the installer from https://github.com/UB-Mannheim/tesseract/wiki)
   - Linux: `sudo apt install tesseract-ocr`
2. **ffmpeg** (ffprobe ships with it) — strongly recommended: audio
   passthrough, H.264 output, and VFR screen-recording normalization all
   depend on it.
   - Windows: `winget install -e --id Gyan.FFmpeg`
   - Linux: `sudo apt install ffmpeg`

Optional extras:

```
pip install "OpenScrub[ner]"             # spaCy name detection (recommended)
python -m spacy download en_core_web_sm
pip install paddleocr paddlepaddle       # better OCR on small UI fonts (large install)
```

spaCy is strongly recommended — it's the primary name detector. The tool
still runs without it using heuristics, but NER is more accurate.

### A proper Windows install (Program Files + Start Menu)

Prefer a normal Windows program over pip? Every release has
`OpenScrub-Setup-<version>.exe` attached on the
[Releases page](https://github.com/austinmabry/OpenScrub/releases)
(built automatically by CI). It installs branded `openscrub.exe` and
`openscrub-web.exe` into `C:\Program Files\OpenScrub` with Start Menu
shortcuts, an uninstaller, and optional one-click winget installs of
Tesseract and FFmpeg. To build it yourself instead, run
`windows\build_installer.bat` from a checkout (requires Python 3.10+
and Inno Setup 6). App data — jobs, certificates,
zones, downloaded models — lives in `%LOCALAPPDATA%\OpenScrub`, never in
Program Files. Note: the frozen build detects names with the built-in
heuristics (spaCy NER is a pip-only extra).

If you stay with pip instead, `openscrub-setup` on Windows now offers to
create Start Menu + Desktop shortcuts for OpenScrub Web, so you never
have to hunt for pip's `Scripts` folder.

## Guided installer (Windows / Linux / macOS best-effort)

Prefer a setup that installs the system tools too? Clone or download the
repository and run:

    python install.py

It probes every dependency and installs what's missing with your consent:
core pip packages, spaCy NER, Tesseract and ffmpeg (via winget / apt /
dnf / pacman / brew), and PaddleOCR — automatically offering the GPU
build when an NVIDIA card is detected — then verifies NVENC hardware
encoding and creates a launchable "OpenScrub" shortcut with the program
icon (Desktop + Start Menu on Windows, a .desktop entry on Linux, a
.command on macOS).

`--check` reports what's present without changing anything; `--yes` runs
unattended; `--cpu-only` skips GPU OCR; `--with-plates` fetches a
license-plate model (see [PLATES.md](PLATES.md)). Start the app from the
created shortcut, or `python openscrub_web.py` — the web interface is the
primary interface. (`openscrub_gui.py`, the desktop Tk interface, still
works but is legacy: new features land in the web app.)

## Updating

```
openscrub-update            # interactive: shows versions, asks, updates
openscrub-update --check    # just report whether an update exists
```

It detects how OpenScrub was installed: pip installs upgrade via
`pip install --upgrade OpenScrub`; folder deploys download the latest
release from PyPI, **verify its SHA-256** against the hash PyPI
publishes, and replace only the released files — your jobs, certificates,
zones, models, allowlist, and locally pinned plate-model hashes are
never touched, and every replaced file is backed up to
`backups/pre-update-<version>/` first. Git checkouts are left to
`git pull`.

The web interface checks for updates too: when a newer release exists,
the footer shows an update link — one click runs the same updater (only
while no job is running), then asks you to restart the server. Restart
after any update to run the new version.

## Web interface (LAN) — one page, editor to review

Run `openscrub-web` (or `python openscrub_web.py`) on an always-on
machine and open the printed URL from any device on your network —
laptop, phone, or tablet. The whole app is **one dark, video-editor-style
page**: the Scan Setup editor on top, your jobs and the review workflow
right below it, and server settings behind the gear icon in the header.

**Scan Setup editor** (the top of the page):

1. **Load a video** — pick a local file or enter a path already on the
   server. Nothing uploads until you press Start scan; the preview runs
   in your browser.
2. **Pick categories** — everything starts OFF, so nothing is detected
   until you say so (the summary line warns you while nothing is
   selected). Check what you want; drawing a zone auto-checks its
   category.
3. **Scope the scan** — detection windows live on the timeline, one lane
   each, and may overlap: blur faces for the whole clip AND names only
   from 5:00–7:00 by stacking two windows. Each window carries its own
   categories and its own zones (click a category's color square, then
   draw on the frame; Copy/Paste moves zones between windows). White
   clip bookends trim the output; audio lanes show a **waveform** so you
   can scrub straight to a loud noise or the moment someone starts
   talking, and each lane has an M button to remove that track. Need to blur something the detectors don't know —
   a tattoo, a badge, a specific person or object? Pick **track object**,
   scrub to a frame where it's clear, and draw ON it: the scan
   template-tracks it through the window and blurs it wherever it goes.
   A **zoom bar** (−/slider/+, with a pan control)
   magnifies the timeline up to 40× for sub-second placement, and
   dragging any handle scrubs the preview live — including on
   iPhone/iPad.
4. **Start scan** — the job queues instantly and its progress card opens
   right below with a live log and ETA. Jobs queue one at a time so they
   don't fight over the GPU.

**Review** (below the editor): every detection appears as a thumbnail
you keep or blur — one card per tracked face, person, or plate, one
decision for all of its appearances — with per-category all-on/all-off,
a before/after box editor to resize, move, add, or time-bound any blur,
"Track object" to template-track a manually drawn box through a time
range, and audio mute/bleep spans. Then render and download the redacted
video plus the audit report.

**Settings** (gear icon): detection model pickers for **Face**,
**License-plate**, and **Person (full-body blur)** — optional models are
downloaded on demand with SHA-256 verification and license badges —
plus optional engines, Encryption at rest, HTTPS certificates, and the
learned safe-words list.

Security: HTTPS by default (self-signed certificate — your browser warns
once; or install your own cert in the settings view). Access is open to
everyone on your network unless you start with `--token <secret>`, which
then gates every request (recommended). Either way this is LAN-grade
protection — never expose the port to the internet. The jobs folder on
the server contains PII (uploads + reports); protect it accordingly.
`--retain-days` auto-deletes finished job folders (default 7 days).

## HDR support

OpenScrub matches the output to the source:

- **SDR source → SDR output.** Nothing changes.
- **HDR source → HDR output by default.** iPhone Dolby Vision, HLG, and
  HDR10 footage is detected at intake and rendered as **10-bit HEVC** with
  the source's color primaries and transfer function (BT.2020 PQ/HLG)
  preserved. Blurs are applied directly in the 10-bit native color domain,
  so pixels outside the redacted regions never pass through a color
  conversion. Set `--hdr-output sdr` (CLI) or the **HDR output** dropdown
  (web) to tone-map the output to SDR BT.709 instead — the proper
  conversion, not a washed-out naive decode.
- Detection always runs on an internally tone-mapped SDR copy (the
  detectors are 8-bit); it shares the exact frame timeline with the HDR
  render, so blur timing is identical.

Notes:

- **Dolby Vision** clips keep their HDR10/HLG base layer — the output is
  real HDR — but the Dolby Vision *dynamic metadata* (per-scene RPUs) is
  dropped: it is a proprietary layer that cannot be re-authored with open
  tools after the frames are modified. Players simply treat the result as
  HDR10/HLG, which is how most non-Apple devices play these files anyway.
- **Hardware:** a GPU with a 10-bit HEVC encoder (NVENC on GTX 10-series
  or newer) renders HDR at full speed. Without one, OpenScrub says so in
  the job log and falls back to CPU encoding (libx265) — correct output,
  much slower. If neither encoder exists, it falls back to SDR output
  with a clear notice; it never fails silently.
- The audit report records `hdr_tonemapped` (detection copy was created)
  and `hdr_output` (output kept HDR) in its provenance block.

## ⚠️ Disclaimer — read this before using openscrub on real footage

**openscrub is a best-effort assistive tool, not a compliance guarantee
(GDPR, HIPAA, or otherwise), a de-identification certification, or a
substitute for human review.** It uses OCR, named-entity recognition, pattern matching, and
face detection — all of which can and do miss things: low-contrast text,
unusual names, stylized fonts, partially occluded faces, content visible
for only a fraction of a second, handwriting, text inside images, and
categories of identifiers it was never designed to detect.

**You remain fully responsible for reviewing every output before it is
shared, published, or distributed.** The built-in review workflow and the
final QC scrub are not optional extras — they are the compensating
control this tool is designed around. If a redacted video leaks someone's
personal information, that is your exposure, not this software's.

Specifically:

- The validation numbers in this README are measurements against a
  **synthetic corpus**. They demonstrate the pipeline works as designed;
  they are **not** a guarantee of performance on your recordings, your
  software, your fonts, or your screen resolution.
- Audit reports, job folders, and normalized/intermediate video files
  **contain PII in plaintext** unless you enable the web UI's
  **Encryption at rest** panel: set a password and job files are
  encrypted (scrypt-derived key, AES-256-GCM) whenever the vault is
  locked or the server shuts down, and decrypted while you work.
  **There is no password reset — a lost password makes encrypted files
  permanently unrecoverable.** While unlocked (and during processing)
  files are plaintext on disk, so pair the vault with OS disk
  encryption (BitLocker etc.), restricted access, and deletion when no
  longer needed.
- The web interface provides **LAN-grade access control at most**
  (HTTPS with an optional access token — set one with `--token`).
  Never expose it to the internet, and run it only on networks and
  machines already authorized to handle the footage.
- This tool addresses **on-screen visual content only**. Audio narration,
  metadata, file names, and embedded subtitles are untouched and can all
  carry PII.
- Nothing in this project constitutes legal, compliance, or regulatory
  advice. Consult your privacy officer or counsel for questions about
  GDPR, HIPAA, state privacy law, or your organization's obligations.

This software is provided "AS IS" without warranty of any kind — see the
[LICENSE](LICENSE) (Apache-2.0, §7–8) for the governing terms.

## Validation

During development the pipeline is scored against a **synthetic corpus**:
a generator plants fake PII at known locations across the hard cases —
static charts, schedule grids, scrolling notes, OCR-disrupting
highlights, embedded face photos — and a scorer checks the rendered
output against the ground truth:

    PII recall:           100.0%   (102/102 planted samples blurred)
    Benign preservation:  100.0%   (39/39 benign samples left readable)

(measured with the Tesseract fallback engine; PaddleOCR + spaCy NER, the
recommended stack, is stronger). The shipped regression suite
(`pytest test_openscrub.py`) exercises the same end-to-end pipeline on
synthetic videos — for this tool a regression is not a bug, it's a leak,
so the suite must stay green on every change.

## Feature notes

- **VFR normalization** — OBS/Game Bar variable-frame-rate recordings are
  detected (ffprobe) and normalized to CFR before processing, preventing
  blur-timing drift and audio desync; recorded in provenance.
- **OCR quality**: low-confidence words that are structurally PII-shaped
  (emails, phones, digit runs) are rescued instead of dropped; small text
  triggers an automatic 2x re-OCR; a reverse pass re-searches the whole
  timeline for near-misses of remembered PII. `--paranoid` preset maxes
  recall at the cost of false positives (clean up in review).
- **False-positive economics**: names must be caught by a primary detector
  on two separate scans before memory starts recalling them; a top-recalls
  summary prints after every scan; the web review suggests allowlisting
  strings you disabled everywhere, building a permanent allow-list.
- **Web**: before/after compare scrubber on finished jobs, ETA on the
  progress bar, `--retain-days` auto-deletes PII-bearing job folders
  (default 7 days).
- **Batch resume** — re-running `--batch` skips files already done
  (`--overwrite` to redo).
- **Review workflow** — scan and render are separate phases; between them
  you can audit every detection and correct both false positives and
  misses. CLI equivalent: run with `--report audit.json`, edit the JSON
  (set `"enabled": false`, or append boxes), then
  `openscrub.py video.mp4 --from-report audit.json` re-renders in seconds
  without re-scanning.
- **Face detection** — the `face` category blurs faces in
  photos, people on camera, and webcam bubbles, which OCR is blind to. Uses the
  YuNet DNN detector (auto-downloaded, ~230 KB) with a Haar-cascade
  fallback. Faces re-detect on every frame; boxes are expanded 15%; face
  tracks are grouped by facial identity (SFace embeddings) so review
  shows one card per person.
- **Person (full-body) detection** — the `person` category masks whole
  bodies with **silhouette precision**: a segmentation model traces each
  person's outline every frame and only the pixels inside it are
  redacted. Download a model (YOLO11n-seg recommended) in the settings
  Person panel; without one the category is inactive and says so. Tuning:
  `--person-threshold` (default 0.5). Tracks are positional — someone
  who leaves frame and returns gets a second review card.
- **Config profiles** — `--config profile.yaml` loads per-environment
  settings (engine, categories, custom regexes, ignore regions…). CLI
  flags override the file.
- **Ignore regions** — `--ignore-region X1,Y1,X2,Y2` (repeatable, or in
  config) excludes screen areas like the taskbar clock from all blurring.
- **Batch mode** — `--batch folder` processes every video, writing
  per-file outputs + audit reports and a `batch_summary.json`.
- **Provenance** — every audit report records tool version, timestamp,
  full settings, and SHA256 of input and output, making the audit trail
  independently verifiable.

## Desktop GUI (Windows, legacy)

`python openscrub_gui.py` opens a desktop app covering everything the
CLI does (legacy: it still works, but new features land in the web app):

- Source / output / audit-report file pickers
- OCR engine selection with live status (Tesseract / PaddleOCR / spaCy NER
  / NVENC detected or missing) and one-click **Install** buttons for
  PaddleOCR (CPU or GPU CUDA 12.6) and spaCy NER
- GPU/CPU toggle for OCR, NVENC/x264 toggle for encoding
- Category checkboxes, blur vs box, preview mode, memory on/off
- Allow-names and always-blur name lists (type directly or load a file)
- Sample interval / scan trigger / padding / bridge gap / regex fields
- Live preview showing each frame as it's analyzed with detection boxes
- Progress bar, log pane, and a Cancel button that cleans up partial output

Extra requirement for the preview pane: `pip install pillow`

## Usage

```
:: standard run
python openscrub.py recording.mp4

:: keep provider/staff names visible (one name per line in the file)
python openscrub.py recording.mp4 --allow-names providers.txt

:: tuning pass — draws boxes instead of blurring
:: (red = detected PII, orange = unscanned scroll safety band)
python openscrub.py recording.mp4 --preview

:: everything
python openscrub.py recording.mp4 --allow-names providers.txt ^
    --extra-names always_blur.txt --sample-interval 0.5 --scan-trigger 60 ^
    --pad 8 --mode blur --report audit.json -o recording_redacted.mp4
```

## How names are detected (no name list)

Three stacked signals, any of which triggers a blur:
1. **spaCy NER** — PERSON entities in reconstructed text lines
2. **Label heuristic** — text following "Patient:", "Name:", "Pt:",
   "Member:", "Insured:", "Guarantor:", etc., stopping at the next label
3. **Capitalized-pair heuristic** — adjacent capitalized non-UI words
   ("Maria Gonzalez", "Henderson, Robert", "Mrs. Whitfield"); auto-enabled
   as fallback when spaCy is missing, or force with --heuristic-names on

`--allow-names providers.txt` whitelists names to KEEP visible (your
physicians/PAs, e.g. Smith, Patel, Nguyen, Garcia).
`--extra-names` force-blurs specific names the detectors might miss.

## How scrolling is handled

Three mechanisms working together:
1. **Per-frame motion tracking** — global scroll offset is measured every
   frame via phase correlation against a keyframe (drift-bounded, verified
   to a few px over a 500px scroll). Every blur box is anchored in content
   coordinates and translated with the scroll, so blur rides along with
   the text on every single frame — not just at sample times.
2. **Motion-triggered scans** — in addition to the time-based interval,
   an OCR scan fires after every --scan-trigger pixels of scroll (default
   60), so newly revealed content is scanned almost immediately.
3. **Safety bands** — any strip of screen that scrolled into view since
   the last OCR scan is blurred wholesale until it has been scanned.
   Unverified content is never shown, even between scans.

Net effect: text detected once stays covered while it moves, and text
scrolling into view is covered by the safety band before it's even been
read. Verified in testing with 26/26 PII regions covered across static,
mid-scroll, and post-scroll frames.

## PII memory and gap bridging

Two reasoning layers prevent "flash of PII" from intermittent OCR misses:

1. **PII text memory** — every string confirmed as PII is remembered for
   the rest of the video. Each scan checks all OCR'd words against memory
   (fuzzy for names, near-exact for numbers), so "Henderson" identified
   once gets blurred on every later appearance anywhere on screen, even
   where NER/heuristics would fail (e.g. a bare surname mid-sentence).
   Disable with --no-memory. Memory is per-run only; nothing persists.
2. **Evidence-based gap bridging** — if the same PII is detected, missed
   for a few scans, then re-detected in the same region, the blur is held
   straight through the gap (up to --bridge-gap seconds, default 4.0) —
   UNLESS an intermediate scan positively read different text there,
   meaning the content genuinely changed. Unreadable or empty gaps fail
   closed: they stay blurred.

## Caveats — read these

- **Best-effort, not a guarantee.** OCR can miss low-contrast or tiny
  text; NER can miss unusual names (heuristics + label detection back it
  up, but nothing is 100%). Do a final QC scrub in your editor at 2x
  before anything goes public. Treat this as removing ~95% of the manual
  work.
- **All dates are blurred**, since the tool can't distinguish DOBs from
  visit dates. Usually right on record-style UIs; drop `dob` from
  --categories if too aggressive for a given recording.
- **Partial-screen scrolling** (one panel scrolls while the rest is
  static) is tracked as whichever motion dominates. If a recording is
  mostly panel-scrolling, use --preview to check coverage and consider
  --sample-interval 0.25.
- **Identifier formats are bring-your-own-pattern**: record numbers,
  claim numbers, and account numbers only get caught if you add a custom
  regex category for your format in the web UI (e.g. `\b\d{7}\b` for
  exactly 7 digits). From the CLI, the legacy `mrn` category still works
  via `--categories ...,mrn --mrn-regex PATTERN`.
- **The --report JSON contains PII in plaintext.** Handle it like any
  PII file.

## Recommended workflow

1. Record as usual
2. `--preview` pass, spot-check red boxes and orange bands
3. Real pass (optionally with --report)
4. Import `_redacted.mp4` into your editor, edit normally
5. Final QC scrub before publishing

## Tuning cheat sheet

| Symptom | Fix |
|---|---|
| Provider names blurred | add them to --allow-names |
| A name slips through | add to --extra-names; install spaCy if not present |
| Random capitalized words blurred | install spaCy so the pair heuristic turns off, or --heuristic-names off |
| Text slips through during very fast scrolling | --scan-trigger 40 and/or --sample-interval 0.25 |
| Benign numbers blurred by a custom regex category | tighten its pattern |
| A person isn't detected (small/distant) | lower --person-threshold to 0.35, or pick the larger seg model |
| Non-people masked by the person category | raise --person-threshold to 0.6 |
| Blur box clips edges of text | --pad 12 |
| Small text missed entirely | install paddleocr; record at native resolution |

## License

OpenScrub is licensed under the [Apache License 2.0](LICENSE)
(© 2026 Austin Mabry — see [NOTICE](NOTICE)). The published Docker
images additionally contain third-party components (FFmpeg, Tesseract
OCR, PaddleOCR, spaCy, OpenCV, and others) under their own licenses;
those are aggregated alongside OpenScrub, not relicensed by it.

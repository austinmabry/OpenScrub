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

**A local, GPU-accelerated tool that blurs faces and any on-screen text —
names, phone numbers, SSNs, emails, dates, ID numbers, or anything you can
express as a regex — in videos and screen recordings, with a human review
step before anything is published.**

Runs entirely on your own machine (no cloud, no upload of sensitive
footage). OCR-driven, so it catches text anywhere on screen; face-tracked,
so a face detected once stays covered even when the detector blinks;
scroll- and motion-aware, so blur boxes follow content as it moves; and
onset-aware, so redaction starts on the exact frame a detail first appears
rather than a half-second late. Defaults are tuned for the hardest case —
dense, scrolling medical-records screens — but the engine is
general-purpose.

> Keywords: video redaction · blur faces in video · redact screen recording ·
> PII redaction · anonymize video · blur license plates · GDPR / CCPA /
> FERPA / PCI / HIPAA · OCR text redaction · face blur · privacy tool

## What it does

- **Blur faces** — detected with a DNN model and visually tracked, so a
  single detection covers a face across frames where it would otherwise
  be missed.
- **Blur license plates** — via an optional ONNX detector model
  (see [PLATES.md](PLATES.md)); plates re-detect every frame, so a plate
  crossing the frame stays covered.
- **Redact text by pattern** — bring your own regex for account numbers,
  case numbers, employee IDs, order numbers; built-in
  patterns for SSNs, phone numbers, emails, dates, addresses (including
  multi-line street/city/state/ZIP blocks), credit/debit card numbers (Luhn-validated), API keys/tokens, IP addresses,
  and medical record numbers.
- **Redact names** — via named-entity recognition plus heuristics, with no
  list required (though you can supply an allowlist to *keep* specific names
  visible and a blocklist to *always* remove others).
- **Detection zones** — restrict redaction to a region, or invert it: keep a
  central subject sharp and blur everyone/everything around them, or vice versa.
- **Redaction styles** — blur, solid black box (irreversible), or mosaic
  pixelation, choosable per category (black-box the SSNs, blur the faces).
- **Human review** — every detection is shown as a thumbnail you can keep or
  blur, with an interactive box editor to resize, move, add, or time-bound
  any blur before rendering.
- **Audit trail** — each run produces a report with SHA-256 hashes of input
  and output for provenance.

## Use cases

PHI redaction for medical practices is the flagship use — it's the hardest
version of the problem (dense, scrolling, fast-changing screens) and the
tool is built to handle it. But the same engine fits many privacy workflows:

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

## Install from PyPI (quickest)

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

### Docker (Linux servers / homelab)

Every release publishes a CPU server image to GitHub Container Registry:

```
docker run -d -p 8384:8384 \
  -v openscrub_data:/root/.local/share/OpenScrub \
  ghcr.io/austinmabry/openscrub:latest
```

Tesseract, FFmpeg, and the face model are baked in; jobs, certificates,
zones, and downloaded plate models live in the mounted volume, so the
container is disposable. Add `--token <secret>` after the image name
(as `openscrub-web --host 0.0.0.0 --token <secret>`) for access
control. To update, pull the new tag and recreate the container — the
in-app updater doesn't apply inside Docker. The default image is
CPU-only and doesn't include spaCy NER.

**NVIDIA GPU build** (`:cuda` / `:<version>-cuda`) — CUDA-accelerated
PaddleOCR and NVENC hardware encoding:

```
docker run -d --gpus all -p 8384:8384 \
  -v openscrub_data:/root/.local/share/OpenScrub \
  ghcr.io/austinmabry/openscrub:cuda
```

On **Unraid**: install the Nvidia Driver plugin, add a container with
repository `ghcr.io/austinmabry/openscrub:cuda`, extra parameter
`--runtime=nvidia`, port 8384, and map
`/root/.local/share/OpenScrub` to `/mnt/user/appdata/openscrub`.
GPU features engage automatically (the OCR engine picks the CUDA build,
and the render's NVENC test selects hardware encoding). Note the GPU
image is several GB.

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

## Web interface (LAN)

Run `python openscrub_web.py` on an always-on machine and open the printed
URL from any device on your network —
laptop or phone. Workflow: upload a recording (or point at a server-side
path) → scan runs on the server with a live preview and log → **review
page**: every detection shown as a thumbnail grouped by category, uncheck
false positives, per-category all-on/all-off, draw missed regions directly
on any frame (works with touch) → render → download the redacted video and
the audit report. Jobs queue one at a time so they don't fight over the GPU.

Security: HTTPS by default (self-signed certificate — your browser warns
once; or install your own cert from the main page). Access is open to
everyone on your network unless you start with `--token <secret>`, which
then gates every request (recommended). Either way this is LAN-grade
protection — never expose the port to the internet. The jobs folder on
the server contains PHI (uploads + reports); protect it accordingly.
`--retain-days` auto-deletes finished job folders (default 7 days).

## ⚠️ Disclaimer — read this before using openscrub on real PHI

**openscrub is a best-effort assistive tool, not a HIPAA compliance
guarantee, a de-identification certification, or a substitute for human
review.** It uses OCR, named-entity recognition, pattern matching, and
face detection — all of which can and do miss things: low-contrast text,
unusual names, stylized fonts, partially occluded faces, content visible
for only a fraction of a second, handwriting, text inside images, and
categories of identifiers it was never designed to detect.

**You remain fully responsible for reviewing every output before it is
shared, published, or distributed.** The built-in review workflow and the
final QC scrub are not optional extras — they are the compensating
control this tool is designed around. If a redacted video leaks protected
health information, that is your exposure, not this software's.

Specifically:

- The validation numbers in this README are measurements against a
  **synthetic corpus**. They demonstrate the pipeline works as designed;
  they are **not** a guarantee of performance on your recordings, your
  EMR, your fonts, or your screen resolution.
- Audit reports, job folders, and normalized/intermediate video files
  **contain PHI in plaintext** unless you enable the web UI's
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
  machines already authorized to handle PHI.
- This tool addresses **on-screen visual content only**. Audio narration,
  metadata, file names, and embedded subtitles are untouched and can all
  carry PHI.
- Nothing in this project constitutes legal, compliance, or regulatory
  advice. Consult your privacy officer or counsel for questions about
  HIPAA, state privacy law, or your organization's obligations.

This software is provided "AS IS" without warranty of any kind — see the
[LICENSE](LICENSE) (Apache-2.0, §7–8) for the governing terms.

## Validation

During development the pipeline is scored against a **synthetic corpus**:
a generator plants fake PHI at known locations across the hard cases —
static charts, schedule grids, scrolling notes, OCR-disrupting
highlights, embedded face photos — and a scorer checks the rendered
output against the ground truth:

    PHI recall:           100.0%   (102/102 planted samples blurred)
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
- **OCR quality**: low-confidence words that are structurally PHI-shaped
  (emails, phones, digit runs) are rescued instead of dropped; small text
  triggers an automatic 2x re-OCR; a reverse pass re-searches the whole
  timeline for near-misses of remembered PHI. `--paranoid` preset maxes
  recall at the cost of false positives (clean up in review).
- **False-positive economics**: names must be caught by a primary detector
  on two separate scans before memory starts recalling them; a top-recalls
  summary prints after every scan; the web review suggests allowlisting
  strings you disabled everywhere, building a permanent allow-list.
- **Web**: before/after compare scrubber on finished jobs, ETA on the
  progress bar, `--retain-days` auto-deletes PHI-bearing job folders
  (default 7 days).
- **Batch resume** — re-running `--batch` skips files already done
  (`--overwrite` to redo).
- **Review workflow** — scan and render are separate phases; between them
  you can audit every detection and correct both false positives and
  misses. CLI equivalent: run with `--report audit.json`, edit the JSON
  (set `"enabled": false`, or append boxes), then
  `openscrub.py video.mp4 --from-report audit.json` re-renders in seconds
  without re-scanning.
- **Face detection** — the `face` category (on by default) blurs faces in
  clinical photos and webcam bubbles, which OCR is blind to. Uses the
  YuNet DNN detector (auto-downloaded, ~230 KB) with a Haar-cascade
  fallback. Faces re-detect on every scan; boxes are expanded 15%.
- **Config profiles** — `--config profile.yaml` loads per-environment
  settings (engine, MRN regex, categories, ignore regions…). CLI flags
  override the file.
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
- Sample interval / scan trigger / padding / bridge gap / MRN regex fields
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
:: (red = detected PHI, orange = unscanned scroll safety band)
python openscrub.py recording.mp4 --preview

:: everything
python openscrub.py recording.mp4 --allow-names providers.txt ^
    --extra-names always_blur.txt --sample-interval 0.5 --scan-trigger 60 ^
    --pad 8 --mode blur --report audit.json -o recording_redacted.mp4
```

## How names are detected (no patient list)

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
read. Verified in testing with 26/26 PHI regions covered across static,
mid-scroll, and post-scroll frames.

## PHI memory and gap bridging

Two reasoning layers prevent "flash of PHI" from intermittent OCR misses:

1. **PHI text memory** — every string confirmed as PHI is remembered for
   the rest of the video. Each scan checks all OCR'd words against memory
   (fuzzy for names, near-exact for numbers), so "Henderson" identified
   once gets blurred on every later appearance anywhere on screen, even
   where NER/heuristics would fail (e.g. a bare surname mid-sentence).
   Disable with --no-memory. Memory is per-run only; nothing persists.
2. **Evidence-based gap bridging** — if the same PHI is detected, missed
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
  visit dates. Usually right on a medical UI; drop `dob` from
  --categories if too aggressive for a given recording.
- **Partial-screen scrolling** (one panel scrolls while the rest is
  static) is tracked as whichever motion dominates. If a recording is
  mostly panel-scrolling, use --preview to check coverage and consider
  --sample-interval 0.25.
- **MRN default** is standalone 7+ digit runs, or 6+ digits near an
  MRN/chart/acct label. Tighten with --mrn-regex if benign numbers get
  caught (e.g. `\b\d{7}\b` for an exact-width MRN).
- **The --report JSON contains PHI in plaintext.** Handle it like any
  PHI file.

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
| Benign numbers blurred as MRN | tighten --mrn-regex |
| Blur box clips edges of text | --pad 12 |
| Small text missed entirely | install paddleocr; record at native resolution |

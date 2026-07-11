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
rather than a half-second late. Ships with a HIPAA preset out of the box,
but the engine is general-purpose.

> Keywords: video redaction · blur faces in video · redact screen recording ·
> PII redaction · anonymize video · blur license plates · GDPR / CCPA /
> FERPA / PCI / HIPAA · OCR text redaction · face blur · privacy tool

## What it does

- **Blur faces** — detected with a DNN model and visually tracked, so a
  single detection covers a face across frames it would otherwise be missed on.
- **Redact text by pattern** — bring your own regex for account numbers,
  license plates, case numbers, employee IDs, order numbers; built-in
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
  posting insurance or public clips (plates via regex + OCR).
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

Two system tools are **not** pip-installable and must be present for full
functionality:

1. **Tesseract OCR** — required for every text category (names, SSNs,
   emails, …). Face and plate detection work without it; text detection
   does not.
   - Windows: installer from https://github.com/UB-Mannheim/tesseract/wiki
   - Linux: `sudo apt install tesseract-ocr`
2. **ffmpeg** (ffprobe ships with it) — strongly recommended: audio
   passthrough, H.264 output, and VFR screen-recording normalization all
   depend on it.
   - Windows: `winget install ffmpeg`
   - Linux: `sudo apt install ffmpeg`

Optional extras:

```
pip install "OpenScrub[ner]"             # spaCy name detection (recommended)
python -m spacy download en_core_web_sm
pip install cheroot                      # production TLS server for the web UI
                                         # (bundled by default after v1.0.0)
```

Prefer a guided setup that installs the system tools for you? Use the
installer below.

## Windows install — easy way

Download everything into one folder and double-click **install.bat**.
It bootstraps Python if needed (via winget), then runs `installer.py`,
which probes every dependency and installs what's missing with your
consent: core pip packages, spaCy NER, Tesseract, ffmpeg (full build with
NVENC), and PaddleOCR — automatically offering the GPU build when an
NVIDIA GPU is detected, and offering an ffmpeg upgrade when NVENC is
present but broken (old build vs. new driver). Re-run any time with
`python installer.py --check` to audit, or `--yes` for unattended install.

## Windows install — manual way

1. **Python 3.10+** from python.org (check "Add to PATH" during install)
2. **Tesseract OCR**: installer from https://github.com/UB-Mannheim/tesseract/wiki
   (default location is auto-detected; no PATH edit needed)
3. **ffmpeg**: `winget install ffmpeg` in PowerShell (needed for audio + h264 output)
4. Python packages:
   ```
   pip install opencv-python rapidfuzz pytesseract spacy
   python -m spacy download en_core_web_sm
   ```
   spaCy is strongly recommended — it's the primary name detector. The tool
   still runs without it using heuristics, but NER is more accurate.

   Optional, better OCR on small UI fonts (large install):
   `pip install paddleocr paddlepaddle`

## Web interface (LAN)

Run `python openscrub_web.py` on an always-on machine and open the printed
URL (includes a required access token) from any device on your network —
laptop or phone. Workflow: upload a recording (or point at a server-side
path) → scan runs on the server with a live preview and log → **review
page**: every detection shown as a thumbnail grouped by category, uncheck
false positives, per-category all-on/all-off, draw missed regions directly
on any frame (works with touch) → render → download the redacted video and
the audit report. Jobs queue one at a time so they don't fight over the GPU.

Security: LAN-grade token auth only — never expose the port to the
internet. The jobs folder on the server contains PHI (uploads + reports);
protect it accordingly.

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
  **contain PHI in plaintext**. Protect them with the same controls as
  any other PHI: restricted access, encryption at rest where required,
  and deletion when no longer needed.
- The web interface provides **LAN-grade token access control only**.
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

## Install

One installer covers Windows and Linux (macOS best-effort). It installs the
Python dependencies, sets up GPU-accelerated OCR when an NVIDIA card is
present, installs Tesseract and ffmpeg via your system package manager,
verifies NVENC hardware encoding, and creates a launchable "OpenScrub"
shortcut with the program icon.

    Windows:      double-click install.bat        (or: python install.py)
    Linux/macOS:  ./install.sh                    (or: python3 install.py)

`--check` reports what's present without changing anything; `--yes` runs
unattended; `--cpu-only` skips GPU OCR. Start the app from the created
shortcut, or `python openscrub_web.py` — the web interface at the printed
HTTPS URL is the primary interface. (`openscrub_gui.py`, the desktop Tk
interface, still works but is legacy: new features land in the web app.)

## Validation

The corpus generator (`make_corpus.py`) plants fake PHI at known
locations across the hard cases — static charts, schedule grids,
scrolling notes, OCR-disrupting highlights, embedded face photos — and
`validate.py` scores the pipeline against the ground truth:

    PHI recall:           100.0%   (102/102 planted samples blurred)
    Benign preservation:  100.0%   (39/39 benign samples left readable)

(measured with the Tesseract fallback engine; PaddleOCR + spaCy NER, the
recommended stack, is stronger). Run it yourself:
`python make_corpus.py --out corpus && python validate.py --corpus corpus`.
CI runs the regression suite (`pytest test_openscrub.py`) and re-validates
recall on every commit — a detection regression fails the build.

## What's new in v4.1

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
- **Packaging**: `pip install .` gives `openscrub`, `openscrub-web`,
  `openscrub-gui` commands.

## What's new in v4

- **Review workflow** — scan and render are separate phases; between them
  you can audit every detection and correct both false positives and
  misses. CLI equivalent: run with `--report audit.json`, edit the JSON
  (set `"enabled": false`, or append boxes), then
  `openscrub.py video.mp4 --from-report audit.json` re-renders in seconds
  without re-scanning.
- **Face detection** — new `face` category (on by default) blurs faces in
  clinical photos and webcam bubbles, which OCR is blind to. Uses the
  YuNet DNN detector (auto-downloaded, ~230 KB) with a Haar-cascade
  fallback. Faces re-detect on every scan; boxes are expanded 15%.
- **Config profiles** — `--config ema.yaml` loads per-environment settings
  (engine, MRN regex, categories, ignore regions…). CLI flags override the
  file. See the included `ema.yaml`.
- **Ignore regions** — `--ignore-region X1,Y1,X2,Y2` (repeatable, or in
  config) excludes screen areas like the taskbar clock from all blurring.
- **Batch mode** — `--batch folder` processes every video, writing
  per-file outputs + audit reports and a `batch_summary.json`.
- **Provenance** — every audit report now records tool version, timestamp,
  full settings, and SHA256 of input and output, making the audit trail
  independently verifiable.

## GUI (Windows)

`python openscrub_gui.py` (or double-click `openscrub_gui.bat`) opens a
desktop app covering everything the CLI does:

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

## PHI memory and gap bridging (v3)

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
  up, but nothing is 100%). Do a final QC scrub in Resolve at 2x before
  anything goes public. Treat this as removing ~95% of the manual work.
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
4. Import `_redacted.mp4` into Resolve, edit normally
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

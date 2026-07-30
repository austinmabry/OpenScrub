# OpenScrub redaction benchmark

There is no industry certification for video redaction — nothing like
NIST FRVT for face recognition. So instead of quoting a number nobody
can check, everything here is **reproducible**: you run the same
commands and get your own numbers.

Two things are measured, because they answer different questions.

| Harness | Question |
|---|---|
| `corpus.py` + `score.py` | Is planted PII still **readable** in the output? |
| `reid.py` | Can a face recognizer still **identify** the person? |

## Why not detection mAP

Detection benchmarks optimise a precision/recall balance. For redaction,
precision barely matters — over-blur is acceptable by design (fail
closed) — and per-frame recall understates the problem, because a face
missed in *one frame out of three hundred* is still a leak. So the
scorer never asks "did a detection fire?". It crops the rendered output
and asks whether the sensitive string can still be **read**, and whether
the face can still be **matched**. A leak counts as a leak even when the
report claims the region was covered.

## Text PII: corpus + scorer

```
python benchmark/corpus.py --out benchmark/_corpus
python benchmark/score.py  --corpus benchmark/_corpus --engine onnx
```

The corpus is synthetic and deterministic — same seed, same layout, same
footage for everyone. Every planted value comes from a **reserved or
documentation range**: 555 phone numbers, `example.com` (RFC 2606),
`192.0.2.0/24` (RFC 5737), the public `4111…` test card. The benchmark
for a privacy tool must not itself be a privacy problem.

Eight scenarios (corpus v2), 64 PII + 43 benign samples: a static
record screen, small text (needs the 2x re-OCR pass), PII on coloured
highlight bars (the classic OCR disruptor), scrolling notes at a known
rate, dark mode (light text on dark), heavy sensor noise, a 2.5°
rotated page (photographed-document tilt), and a brutally re-compressed
copy (120 kbps — screen-share/messenger quality).

Reported: **PII recall** (samples never readable in any sampled frame),
**frame leak rate** (region-frames that were readable), and **benign
preservation** (how much readable non-PII survived — the over-blur cost).

## Faces: re-identification attack

```
python benchmark/reid.py --video yourclip.mp4
python benchmark/reid.py --video yourclip.mp4 --mode mosaic --coverage concealed
```

Finds faces in the original with a deliberately low threshold (the
oracle should over-find, never grade generously), redacts, then at each
oracle location asks whether a face is still detectable and whether its
SFace embedding still matches the identity taken from the original.

**Read the control line before quoting any result.** The harness also
compares original faces against *other original frames of the same
footage*. If the recognizer cannot link an unredacted face to another
unredacted frame of itself, then "the redacted face did not match"
proves nothing about the redaction — and the tool says so, loudly,
instead of letting you publish the number.

Supply your own footage or public-domain clips. Do not run this on real
people's faces and publish the frames.

## Measured: text PII

OpenScrub v1.0.78, CPU only (no GPU), corpus v2 (8 scenarios), 5 frames
sampled per planted value. Higher is better for recall and benign
preservation; lower is better for leak rate:

| OCR engine | PII recall | Frame leak rate | Benign preserved |
|---|---|---|---|
| ONNX PP-OCRv5 (the default on CPU/Intel/Windows) | 100.0 % | 0.00 % | 100.0 % |
| Tesseract | 100.0 % | 0.00 % | 97.7 % |

Every planted value — names, SSNs, phone numbers, addresses, cards,
emails, dates of birth, IP addresses — was unreadable in every sampled
frame of the rendered output, across all eight scenarios, on both
engines. The one benign loss (Tesseract) is a genuine over-blur, and
worth naming precisely: on the 120 kbps copy, compression noise made
Tesseract read the lowercase "in" of "Follow-up in 6 weeks" as "In", the
capitalised-pair name heuristic saw "Follow-up In", and the phrase was
blurred as a name. Wrong, but wrong in the safe direction — the
failure this tool chooses on purpose.

**No zones, no detection windows, no tuning flags.** The scorer runs
the engine on its defaults (`--engine <x> -o … --report …`) and nothing
else. Scoping features would raise the numbers by telling the engine
where the answers are — see "Why the benchmark runs unassisted" below.

A 100 % row on a fixed corpus means the corpus no longer finds leaks —
not that leaks are impossible. The honest reading is the history below.

### What building the benchmark caught (all fixed in 1.0.78)

The first runs of this harness measured the ONNX backend at **40.6 %
recall with a 57.5 % frame leak rate**. Every point of the gap traced to
a real, previously-invisible defect; each is now fixed and pinned by a
regression test:

- **Lost spaces.** PP-OCRv5's space class routinely loses the argmax to
  blank at word gaps, welding `SSN 123-45-6789` into `SSN123-45-6789` —
  every multi-token pattern silently stopped matching.
- **Full-width punctuation.** The Chinese-first vocabulary returned
  `（555）013-8842` — visually identical to a reader, matched by no regex.
- **Shredded card numbers.** Compression and noise fragment a spaced
  PAN into arbitrary digit chunks (`41 11 11 …`, `4111. 1111-1111`);
  the fixed-format matcher missed them. Fragments now join under a
  Luhn + brand-prefix gate.
- **Column-spanning over-blur.** Two-column layouts merged into one OCR
  line, so an address detection's blur swallowed unrelated text a page
  width away (benign preservation cost, found by the corpus's benign
  samples).
- **Tilted-line grouping.** On the 2.5° rotated page, line grouping by
  average y-center split a first name from its surname and the name was
  never detected at all.
- **Uncovered tails.** Under heavy compression the last scans read only
  fragments of a card number — real evidence the string was still on
  screen, but not a full match, so the blur ended ~0.7 s early.

## Measured: face re-identification

OpenScrub v1.0.78, default blur mode, SFace matcher, same-person
threshold 0.55. "Re-identified" = the redacted face still matched the
identity embedding taken from the unredacted original. All three rows
have a valid control (the recognizer reliably re-identifies the same
faces in the *unredacted* footage, so a 0 % result is meaningful):

| Footage | Faces attacked | Re-identified | Still detected as a face | Control |
|---|---|---|---|---|
| Handheld 1080p party clip, five subjects, constant motion ([Pexels 7100826](https://www.pexels.com/video/7100826/), downscaled from 4K) | 125 | **0 (0.0 %)** | 0.0 % | 0.888 |
| The same party clip at native 4K (3840×2160 — larger faces, more recoverable detail) | 128 | **0 (0.0 %)** | 0.0 % | 0.887 |
| Public-domain interview, large frontal close-up faces, 1080×1920 ([Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Interview_with_a_Marshall_Center_professor_(999984).webm)) | 133 | **0 (0.0 %)** | 0.7 % | 0.691 |
| Same party clip, `--mode mosaic` | 125 | **0 (0.0 %)** | 4.8 % | 0.888 |

Mean post-redaction similarity sat **below the cross-person chance
floor** in all three runs — a redacted face matches its own original no
better than a random stranger's face does.

This table also records a caught defect: the interview clip's close-up
faces (200–750 px) were initially re-identified at **8.3 % straight
through the old blur** — a large face keeps enough low-frequency
identity when the blur kernel is 1/3 of the region. The kernel is now
2/3 of the region (pinned by test), which also drove
"still detected as a face" from 99.2 % to 0 % on the party clip.

Two candidate clips (a street scene with small oblique faces, a
multi-person interview) produced **invalid controls** — the recognizer
could not re-identify unredacted faces in them — and the harness
refused to grade them. That refusal is the feature.

## Why the benchmark runs unassisted

OpenScrub ships scoping tools that make redaction better in practice:
**detection windows** (scan only these time ranges) and **zones**
(look for this category only inside this rectangle), plus never-blur
**ignore zones**. A benchmark run *with* them would score at least as
well as the numbers above and would remove false blurs, because a zone
that excludes a column excludes every false positive in it.

They are deliberately not used here, for one reason: on a synthetic
corpus the person drawing the zones has the answer key. Zoning the
region where you planted the SSN and then reporting that the SSN was
covered measures the renderer, not the detector — and the first
question any skeptical reader asks is "so you told it where to look?".
The number that means something is the one from the path a new user
gets by default, on footage the tool knows nothing about.

So the rule for this harness: **the engine is never told anything the
corpus generator knew.** If a future scenario defeats the unassisted
path, the fix belongs in the detector, not in a hand-drawn rectangle.

That is a statement about the *benchmark*, not about the feature.
Windows and zones are the right tool on real footage — where you, not
a generator, know the phone number only appears in the lower third, and
where excluding the rest of the frame genuinely removes false blurs and
scan time. If a zone-assisted number is ever published, it belongs in
its own clearly-labelled row ("guided mode"), never merged into the
unassisted one.

## Rules for quoting these numbers

- Always state **version, engine, corpus version, and hardware**. A
  number without them is not a claim, it is a vibe.
- Never say "HIPAA compliant" or "GDPR compliant" — tools are not
  compliant, processes are.
- Never say "guaranteed", "fully anonymised", or "undetectable" — those
  describe the tool, and no redaction tool can promise them.
- A measured 100 % is a **result**, not a promise: quote it only with
  its scope attached ("100 % of the planted values on corpus v2 were
  unreadable in the output"), never bare and never in the future tense
  ("OpenScrub blurs 100 % of PII"). The first is what happened; the
  second is a claim about footage nobody has run yet.
- Publish the failures next to the successes. A tool that says where it
  struggles is the one worth believing.

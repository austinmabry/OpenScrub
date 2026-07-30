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

Four scenarios, 32 PII + 23 benign samples: a static record screen,
small text (needs the 2x re-OCR pass), PII on coloured highlight bars
(the classic OCR disruptor), and scrolling notes at a known rate.

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

## Measured so far

OpenScrub v1.0.78, CPU only (no GPU), corpus v1, 5 frames sampled per
sample. These are **starting numbers on a deliberately hard corpus**,
not marketing figures:

| OCR engine | PII recall | Frame leak rate | Benign preserved |
|---|---|---|---|
| Tesseract | 84.4 % | 11.3 % | 87.0 % |
| ONNX PP-OCRv5 (before the fixes below) | 40.6 % | 57.5 % | 91.3 % |
| ONNX PP-OCRv5 (after) | 68.8 % | 26.9 % | 82.6 % |

Building this harness immediately found two real defects in the ONNX OCR
backend — the **default on CPU, Intel and Windows** since 1.0.69 — both
fixed in 1.0.78:

- **Lost spaces.** PP-OCRv5's space class routinely loses the argmax to
  blank at word gaps, welding `SSN 123-45-6789` into `SSN123-45-6789`,
  so every multi-token pattern (address, phone, SSN) silently stopped
  matching. The signal was recoverable: peak space probability inside a
  word gap measured ~0.15 versus ~0.000 inside a word.
- **Full-width punctuation.** Its Chinese-first vocabulary returned
  `（555）013-8842` — visually identical to a reader, matched by no
  regex.

**Known gap, stated plainly:** even after those fixes the ONNX backend
still trails Tesseract on this corpus (68.8 % vs 84.4 %), mostly on
name, phone and SSN. Line-level detection with proportional word
splitting yields coarser word boxes than Tesseract's true per-word
output. Until that closes, `--engine tesseract` is the stronger choice
for text-PII-critical work on CPU builds.

## Rules for quoting these numbers

- Always state **version, engine, corpus version, and hardware**. A
  number without them is not a claim, it is a vibe.
- Never say "HIPAA compliant" or "GDPR compliant" — tools are not
  compliant, processes are.
- Never say 100 %, "guaranteed", or "fully anonymised".
- Publish the failures next to the successes. A tool that says where it
  struggles is the one worth believing.

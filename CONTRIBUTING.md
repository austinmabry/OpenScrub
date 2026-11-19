# Contributing to OpenScrub

Thanks for helping! A few rules keep this privacy tool trustworthy.

## Hard rules

1. **Fail closed.** Over-blur beats under-blur. Unverified models or
   inputs are rejected loudly, never run silently.
2. **No real personal data anywhere** — no real names, patient data,
   provider data, or identifying screenshots in code, tests, fixtures,
   issues, or docs. All test data is synthetic.
3. **The report JSON is a compatibility surface** (review UI and
   rehydration read it): extend it, don't break it.
4. Keep the human-review step prominent in any UX change.

## Dev setup

    git clone https://github.com/austinmabry/OpenScrub
    cd OpenScrub
    pip install -e . && pip install pytest build
    # system tools: tesseract-ocr + ffmpeg (see README)

## Before you open a PR

    python -m pytest -q                        # all discovered tests must be green
    python -m build                            # full sdist->wheel build

The discovered suite includes `test_openscrub.py` and `tests/`. The HDR
cleanup regressions use synthetic media and real FFmpeg/ffprobe with libx265;
they skip when those tools are unavailable. Child-reaping observations require
POSIX, and the injected encoder-exit case also requires `waitid`/`WNOWAIT`.
Focused regressions belong in `tests/test_<topic>.py`; keep the existing root
suite. Save As behavior tests execute the real embedded JavaScript with Node.js
(18+), which must be on PATH for local and CI runs.

For web UI changes, also boot `python openscrub_web.py` and click through
what you touched — and note the PAGE string is a normal (non-raw) Python
string, so backslashes in embedded JS need doubling. For engine changes,
run a real render on a small synthetic video and sanity-check the report.

See CLAUDE.md for the full architecture map, the category-alignment rule,
and the verification workflow.

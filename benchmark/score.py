#!/usr/bin/env python3
"""Score OpenScrub against the synthetic corpus.

The test is deliberately not "did a detection fire?" but the question a
reader of the output actually cares about: **is the planted string still
readable in the rendered file?** Each planted sample's region is cropped
out of the REDACTED video at several times inside its on-screen window
and pushed through Tesseract; a fuzzy match against the known string
means the redaction failed there.

That makes the check independent of OpenScrub's own detection bookkeeping
— a leak counts as a leak even if the report claims the region was
covered.

    python benchmark/corpus.py --out benchmark/_corpus
    python benchmark/score.py  --corpus benchmark/_corpus

Metrics:
  PII recall            PII samples never readable in any sampled frame
  Leak rate             sampled (PII x frame) pairs that were readable
  Benign preservation   benign samples still readable (over-blur cost)
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import openscrub  # noqa: E402

try:
    from rapidfuzz import fuzz
except ImportError:
    fuzz = None

FRAMES_PER_SAMPLE = 5


class _Quiet(openscrub.Callbacks):
    def log(self, msg):
        pass


def _norm(s):
    return "".join(ch for ch in s.upper() if ch.isalnum())


def _readable(crop, target):
    """True if `target` can still be read out of `crop`. Upscaled 2x and
    given a generous PSM — the adversary is assumed to try at least as
    hard as a casual viewer with a zoom tool."""
    if crop.size == 0 or min(crop.shape[:2]) < 6:
        return False, ""
    import pytesseract
    big = cv2.resize(crop, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    txt = pytesseract.image_to_string(big, config="--psm 6")
    got, want = _norm(txt), _norm(target)
    if not want:
        return False, txt.strip()
    if want in got:
        return True, txt.strip()
    # A partial match only counts if a COMPARABLE amount of text came
    # back: without this length guard a surviving 3-character label
    # ("SSN") fuzzy-matches the whole redacted string "SSN 123-45-6789"
    # and a real leak is scored as a leak that isn't there — or worse,
    # a successful redaction is scored as a failure.
    if fuzz is not None and len(want) >= 6 and len(got) >= 0.7 * len(want):
        # an attacker does not need every character to re-identify someone
        return fuzz.partial_ratio(want, got) >= 85, txt.strip()
    return False, txt.strip()


def _box_at(sample, t, scroll):
    x1, y1, x2, y2 = sample["box"]
    dy = scroll * (t - sample["t0"])
    return [x1, y1 - dy, x2, y2 - dy]


def _grab(cap, t):
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
    ok, fr = cap.read()
    return fr if ok else None


def score_scenario(sc, corpus_dir, out_dir, engine, extra_args):
    src = os.path.join(corpus_dir, sc["video"])
    dst = os.path.join(out_dir, sc["name"] + "_redacted.mp4")
    report = os.path.join(out_dir, sc["name"] + "_report.json")

    parser = openscrub.build_parser()
    args = parser.parse_args([src, "--engine", engine, "-o", dst,
                              "--report", report, *extra_args])
    args = openscrub._prep_args(args, parser)
    t0 = time.time()
    openscrub.run_pipeline(args, _Quiet())
    secs = time.time() - t0

    cap = cv2.VideoCapture(dst)
    scroll = sc.get("scroll_px_per_s", 0.0)
    rows = []
    for s in sc["samples"]:
        span = max(0.01, s["t1"] - s["t0"])
        # sample inside the window, avoiding the exact edges
        times = [s["t0"] + span * f
                 for f in np.linspace(0.15, 0.85, FRAMES_PER_SAMPLE)]
        hits, seen = 0, []
        for t in times:
            fr = _grab(cap, t)
            if fr is None:
                continue
            h, w = fr.shape[:2]
            x1, y1, x2, y2 = _box_at(s, t, scroll)
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(w, int(x2)), min(h, int(y2))
            if x2 <= x1 or y2 <= y1:
                continue
            ok, got = _readable(fr[y1:y2, x1:x2], s["text"])
            seen.append(len(times))
            if ok:
                hits += 1
        rows.append({"text": s["text"], "category": s["category"],
                     "kind": s["kind"], "frames": len(seen),
                     "readable_frames": hits})
    cap.release()
    return {"scenario": sc["name"], "seconds": round(secs, 1), "rows": rows}


def summarize(results):
    pii = [r for res in results for r in res["rows"] if r["kind"] == "pii"]
    ben = [r for res in results for r in res["rows"] if r["kind"] == "benign"]
    leaked = [r for r in pii if r["readable_frames"] > 0]
    pii_frames = sum(r["frames"] for r in pii)
    leak_frames = sum(r["readable_frames"] for r in pii)
    kept = [r for r in ben if r["readable_frames"] > 0]
    by_cat = {}
    for r in pii:
        c = by_cat.setdefault(r["category"], [0, 0])
        c[1] += 1
        if r["readable_frames"] == 0:
            c[0] += 1
    return {
        "pii_samples": len(pii),
        "pii_covered": len(pii) - len(leaked),
        "pii_recall": (len(pii) - len(leaked)) / max(1, len(pii)),
        "leaked_samples": [r["text"] for r in leaked],
        "pii_frame_checks": pii_frames,
        "leaked_frame_checks": leak_frames,
        "frame_leak_rate": leak_frames / max(1, pii_frames),
        "benign_samples": len(ben),
        "benign_preserved": len(kept),
        "benign_preservation": len(kept) / max(1, len(ben)),
        "by_category": {k: {"covered": v[0], "total": v[1]}
                        for k, v in sorted(by_cat.items())},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--corpus", default=os.path.join(here, "_corpus"))
    ap.add_argument("--out", default=os.path.join(here, "_results"))
    ap.add_argument("--engine", default="tesseract",
                    help="OCR engine for the RUN (scoring always uses "
                         "Tesseract independently)")
    ap.add_argument("--json", default="", help="write raw results here")
    ap.add_argument("rest", nargs="*", help="extra flags passed to the engine")
    a = ap.parse_args()

    gt_path = os.path.join(a.corpus, "ground_truth.json")
    if not os.path.exists(gt_path):
        raise SystemExit("no corpus — run: python benchmark/corpus.py "
                         "--out %s" % a.corpus)
    with open(gt_path, encoding="utf-8") as f:
        gt = json.load(f)
    os.makedirs(a.out, exist_ok=True)

    print("OpenScrub v%s | engine=%s | corpus v%d"
          % (openscrub.VERSION, a.engine, gt.get("corpus_version", 0)))
    results = []
    for sc in gt["scenarios"]:
        print("  running %-18s ..." % sc["name"], end="", flush=True)
        res = score_scenario(sc, a.corpus, a.out, a.engine, a.rest)
        s_pii = [r for r in res["rows"] if r["kind"] == "pii"]
        leak = sum(1 for r in s_pii if r["readable_frames"] > 0)
        print(" %2d/%2d PII covered  (%.0fs)"
              % (len(s_pii) - leak, len(s_pii), res["seconds"]))
        results.append(res)

    s = summarize(results)
    print("\n" + "=" * 58)
    print(" PII recall          %6.1f%%   (%d/%d samples never readable)"
          % (100 * s["pii_recall"], s["pii_covered"], s["pii_samples"]))
    print(" Frame leak rate     %6.2f%%   (%d/%d region-frames readable)"
          % (100 * s["frame_leak_rate"], s["leaked_frame_checks"],
             s["pii_frame_checks"]))
    print(" Benign preserved    %6.1f%%   (%d/%d left readable)"
          % (100 * s["benign_preservation"], s["benign_preserved"],
             s["benign_samples"]))
    print("=" * 58)
    print(" by category:")
    for cat, v in s["by_category"].items():
        print("   %-10s %d/%d" % (cat, v["covered"], v["total"]))
    if s["leaked_samples"]:
        print(" LEAKED:")
        for t in s["leaked_samples"]:
            print("   - %s" % t)
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump({"version": openscrub.VERSION, "engine": a.engine,
                       "summary": s, "scenarios": results}, f, indent=1)
        print(" raw results -> %s" % a.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())

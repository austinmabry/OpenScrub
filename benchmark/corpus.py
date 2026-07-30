#!/usr/bin/env python3
"""Generate the synthetic redaction benchmark corpus.

Every value planted here comes from a RESERVED or documentation range —
555 phone numbers (fiction), example.com (RFC 2606), 192.0.2.0/24
(RFC 5737 TEST-NET-1), 4111... (the public Visa test card), and
constructed names. Nothing in this corpus is, or resembles, a real
person's data: the benchmark for a privacy tool must not itself be a
privacy problem.

Deterministic: a fixed seed and fixed layout mean two people running
this get byte-comparable footage, which is what makes the published
numbers checkable.

    python benchmark/corpus.py --out benchmark/_corpus

Writes <out>/<scenario>.mp4 plus <out>/ground_truth.json:

    {"scenarios": [{"name", "video", "fps", "samples": [
        {"text", "category", "box": [x1,y1,x2,y2], "t0", "t1",
         "kind": "pii"|"benign"}, ...]}]}

`box` is where the string is on screen; for scrolling scenarios the box
is the position at t0 and the scorer follows it with the known scroll
rate (`scroll_px_per_s`).
"""
import argparse
import json
import os

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX
W, H = 1280, 720
FPS = 30

# ---- planted values: reserved ranges only (see module docstring) ----------
PII = [
    ("Marguerite Vandersloot", "name"),
    ("SSN 123-45-6789", "ssn"),
    ("(555) 013-8842", "phone"),
    ("m.vandersloot@example.com", "email"),
    ("4210 Kestrel Hollow Road", "address"),
    ("DOB 04/17/1963", "dob"),
    ("4111 1111 1111 1111", "card"),
    ("192.0.2.147", "ipaddr"),
]
BENIGN = [
    "Ward 4 East",
    "Follow-up in 6 weeks",
    "Status: active",
    "Reviewed by duty desk",
    "Priority normal",
]


def _text_box(txt, org, scale, thick):
    (tw, th), base = cv2.getTextSize(txt, FONT, scale, thick)
    x, y = org
    return [x, y - th - 2, x + tw + 2, y + base + 2]


def _put(img, txt, org, scale=0.9, thick=2, colour=(20, 20, 20)):
    cv2.putText(img, txt, org, FONT, scale, colour, thick, cv2.LINE_AA)
    return _text_box(txt, org, scale, thick)


def _write(path, frames, fps=FPS):
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    for f in frames:
        out.write(f)
    out.release()


def scenario_static_record(path, seconds=4):
    """A records screen: labelled fields, PII in the values. The easy
    case — if a tool fails here it fails everywhere."""
    img = np.full((H, W), 246, np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(img, (0, 0), (W, 70), (32, 54, 92), -1)
    _put(img, "PATIENT RECORD  -  SYNTHETIC TEST DATA", (40, 45),
         0.85, 2, (255, 255, 255))
    samples = []
    y = 150
    for txt, cat in PII:
        _put(img, cat.upper() + ":", (60, y), 0.6, 1, (110, 110, 110))
        box = _put(img, txt, (330, y), 0.9, 2)
        samples.append({"text": txt, "category": cat, "box": box,
                        "t0": 0.0, "t1": seconds, "kind": "pii"})
        y += 62
    y = 150
    for txt in BENIGN:
        box = _put(img, txt, (830, y), 0.75, 2, (60, 60, 60))
        samples.append({"text": txt, "category": None, "box": box,
                        "t0": 0.0, "t1": seconds, "kind": "benign"})
        y += 62
    _write(path, [img] * int(seconds * FPS))
    return {"samples": samples, "scroll_px_per_s": 0.0}


def scenario_small_text(path, seconds=4):
    """Same fields at a small point size — the case that needs the
    automatic 2x re-OCR pass."""
    img = np.full((H, W, 3), 250, np.uint8)
    samples = []
    y = 120
    for txt, cat in PII:
        box = _put(img, txt, (70, y), 0.45, 1)
        samples.append({"text": txt, "category": cat, "box": box,
                        "t0": 0.0, "t1": seconds, "kind": "pii"})
        y += 40
    for i, txt in enumerate(BENIGN):
        box = _put(img, txt, (700, 120 + i * 40), 0.45, 1, (70, 70, 70))
        samples.append({"text": txt, "category": None, "box": box,
                        "t0": 0.0, "t1": seconds, "kind": "benign"})
    _write(path, [img] * int(seconds * FPS))
    return {"samples": samples, "scroll_px_per_s": 0.0}


def scenario_highlighted(path, seconds=4):
    """PII sitting on coloured highlight bars — the classic OCR
    disruptor (low contrast, chunky background)."""
    img = np.full((H, W, 3), 248, np.uint8)
    samples = []
    hl = [(255, 241, 118), (167, 243, 208), (196, 181, 253)]
    y = 150
    for i, (txt, cat) in enumerate(PII):
        box = _text_box(txt, (90, y), 0.9, 2)
        cv2.rectangle(img, (box[0] - 10, box[1] - 6), (box[2] + 10,
                      box[3] + 6), hl[i % len(hl)], -1)
        box = _put(img, txt, (90, y), 0.9, 2)
        samples.append({"text": txt, "category": cat, "box": box,
                        "t0": 0.0, "t1": seconds, "kind": "pii"})
        y += 66
    for i, txt in enumerate(BENIGN):
        box = _put(img, txt, (820, 150 + i * 66), 0.75, 2, (60, 60, 60))
        samples.append({"text": txt, "category": None, "box": box,
                        "t0": 0.0, "t1": seconds, "kind": "benign"})
    _write(path, [img] * int(seconds * FPS))
    return {"samples": samples, "scroll_px_per_s": 0.0}


def scenario_scrolling_notes(path, seconds=8, speed=180.0):
    """Notes scrolling upward at a known rate — exercises the scroll
    tracker and the safety bands. The scorer follows each planted box
    with `scroll_px_per_s`."""
    rows, samples = [], []
    filler = ["Shift handover note", "No changes since last review",
              "Equipment checked", "Routine observation logged"]
    # rows are laid out from the top of the screen downward past it; the
    # whole column scrolls up by speed*seconds, so every row crosses the
    # viewport at some point (all of them are scored, not just the first)
    y = 120
    for i, (txt, cat) in enumerate(PII):
        rows.append((txt, y, True, cat))
        y += 90
        rows.append((filler[i % len(filler)], y, False, None))
        y += 90
    total = int(seconds * FPS)
    frames = []
    for n in range(total):
        img = np.full((H, W, 3), 252, np.uint8)
        dy = speed * (n / FPS)
        for txt, y0, is_pii, cat in rows:
            yy = int(y0 - dy)
            if -40 < yy < H + 40:
                _put(img, txt, (80, yy), 0.85, 2,
                     (20, 20, 20) if is_pii else (90, 90, 90))
        frames.append(img)
    for txt, y0, is_pii, cat in rows:
        # time window while this row is on screen, and its box at t0
        t_enter = max(0.0, (y0 - H) / speed)
        t_exit = min(seconds, y0 / speed)
        if t_exit - t_enter < 0.4:
            continue
        box = _text_box(txt, (80, int(y0 - speed * t_enter)), 0.85, 2)
        samples.append({"text": txt, "category": cat, "box": box,
                        "t0": round(t_enter, 2), "t1": round(t_exit, 2),
                        "kind": "pii" if is_pii else "benign"})
    _write(path, frames)
    return {"samples": samples, "scroll_px_per_s": speed}


SCENARIOS = {
    "static_record": scenario_static_record,
    "small_text": scenario_small_text,
    "highlighted": scenario_highlighted,
    "scrolling_notes": scenario_scrolling_notes,
}


def build(outdir):
    os.makedirs(outdir, exist_ok=True)
    doc = {"corpus_version": 1, "fps": FPS, "size": [W, H], "scenarios": []}
    for name, fn in SCENARIOS.items():
        video = os.path.join(outdir, name + ".mp4")
        meta = fn(video)
        n_pii = sum(1 for s in meta["samples"] if s["kind"] == "pii")
        print("  %-18s %2d PII / %2d benign  -> %s"
              % (name, n_pii, len(meta["samples"]) - n_pii,
                 os.path.basename(video)))
        doc["scenarios"].append({
            "name": name, "video": os.path.basename(video), "fps": FPS,
            "scroll_px_per_s": meta["scroll_px_per_s"],
            "samples": meta["samples"]})
    gt = os.path.join(outdir, "ground_truth.json")
    with open(gt, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1)
    tot = sum(len(s["samples"]) for s in doc["scenarios"])
    print("  ground truth: %s (%d samples across %d scenarios)"
          % (gt, tot, len(doc["scenarios"])))
    return gt


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "_corpus"))
    a = ap.parse_args()
    print("building synthetic corpus (reserved/documentation values only)")
    build(a.out)

#!/usr/bin/env python3
"""Re-identification attack: the metric that actually matters.

Detection mAP answers "did a box fire?". The question a privacy officer
asks is "can this person still be identified from the output?" — so this
harness attacks the redacted file directly.

Protocol (each step exists because a naive version produced a wrong
number on real footage):

  1. CANDIDATES   scan the ORIGINAL with a low-threshold face pass.
  2. CONFIRM      re-test every candidate at the standard threshold.
                  The low-threshold pass alone "found" smiley-face
                  banner flags, the back of a head, and a birthday cake
                  — all correctly untouched by the redaction, all
                  matching themselves at 0.99 similarity, inflating the
                  "re-identified" rate with things that are not faces.
                  Confirmed faces are the attack surface; the
                  unconfirmed count is reported for transparency.
  3. TRACKS       IoU-link confirmed faces over time into per-person
                  tracks (multi-person footage!).
  4. CONTROL      same-track original-vs-original similarity. If the
                  recognizer cannot match an UNREDACTED face to another
                  frame of the same person, the clip cannot support a
                  re-id claim — the harness says so and refuses.
                  (An early version averaged across ALL faces; on a
                  five-person clip that mostly measured different-person
                  similarity and condemned a perfectly good clip.)
  5. ATTACK       for each confirmed face: is a face still detectable
                  in the redacted crop, does its SFace embedding still
                  match the original identity, and did the pixels
                  actually change (an untouched re-identified face is a
                  pipeline MISS; an altered one is weak redaction —
                  different bugs, reported separately).

    python benchmark/reid.py --video clip.mp4
    python benchmark/reid.py --video clip.mp4 --redacted out.mp4   # reuse render
    python benchmark/reid.py --video clip.mp4 --mode mosaic --coverage concealed

Use footage you own or licensed stock. Do NOT publish frames of real
people's faces from this harness.
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

SAME_PERSON = 0.55          # engine's identity-grouping threshold


class _Quiet(openscrub.Callbacks):
    def log(self, msg):
        pass


def _sface():
    p = os.path.join(openscrub._model_dir(),
                     "face_recognition_sface_2021dec.onnx")
    if not os.path.exists(p):
        openscrub._fetch_model(openscrub.SFACE_URL, p,
                               sha256=openscrub.SFACE_SHA256,
                               log_fn=lambda m: None)
    return cv2.FaceRecognizerSF_create(p, "")


def _embed(rec, crop):
    if crop.size == 0 or min(crop.shape[:2]) < 16:
        return None
    try:
        img = cv2.resize(crop, (112, 112))
        return rec.feature(img).flatten()
    except Exception:
        return None


def _cos(a, b):
    if a is None or b is None:
        return None
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return None
    return float(np.dot(a, b) / (na * nb))


def _iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def confirmed_faces(video, every=0.5, low=0.3, confirm=0.6):
    """[(t, box)] real faces + count of unconfirmed low-threshold
    candidates. Candidates come from the low pass (over-find), but only
    boxes the STANDARD-threshold detector agrees on count as attack
    surface — see the module docstring for what the low pass alone
    'found'."""
    det_low = openscrub.FaceDetector(_Quiet(), thresh=low)
    det_std = openscrub.FaceDetector(_Quiet(), thresh=confirm)
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(every * fps)))
    out, unconfirmed, n = [], 0, 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if n % step == 0:
            cand = [c[:4] for c in det_low.find(fr)]
            conf = [c[:4] for c in det_std.find(fr)]
            for b in cand:
                if any(_iou(b, c) >= 0.4 for c in conf):
                    out.append((n / fps, [int(v) for v in b]))
                else:
                    unconfirmed += 1
        n += 1
    cap.release()
    return out, unconfirmed, fps


def link_tracks(faces, max_gap=1.6, iou_min=0.25):
    """IoU-link (t, box) detections into per-person tracks."""
    tracks = []
    for t, box in sorted(faces):
        best, best_iou = None, iou_min
        for tr in tracks:
            lt, lb = tr[-1]
            if t - lt <= max_gap:
                i = _iou(box, lb)
                if i > best_iou:
                    best, best_iou = tr, i
        if best is not None:
            best.append((t, box))
        else:
            tracks.append([(t, box)])
    return tracks


def run(video, out_dir, mode, coverage, categories, extra, redacted=None):
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(video))[0]
    dst = redacted or os.path.join(out_dir, base + "_redacted.mp4")
    report = os.path.join(out_dir, base + "_report.json")

    print("  candidate+confirm pass on the original ...", end="", flush=True)
    faces, unconfirmed, fps = confirmed_faces(video)
    tracks = link_tracks(faces)
    print(" %d confirmed faces in %d track(s), %d unconfirmed "
          "low-threshold candidates excluded"
          % (len(faces), len(tracks), unconfirmed))
    if not faces:
        print("  no confirmed faces — use a clip with visible faces.")
        return None

    if redacted is None:
        parser = openscrub.build_parser()
        args = parser.parse_args([video, "-o", dst, "--report", report,
                                  "--categories", categories, "--mode", mode,
                                  "--coverage", coverage, *extra])
        args = openscrub._prep_args(args, parser)
        print("  redacting (mode=%s coverage=%s) ..." % (mode, coverage),
              end="", flush=True)
        t0 = time.time()
        openscrub.run_pipeline(args, _Quiet())
        print(" %.0fs" % (time.time() - t0))
    else:
        print("  reusing redacted file: %s" % redacted)

    rec = _sface()
    det = openscrub.FaceDetector(_Quiet(), thresh=0.5)
    cap_o, cap_r = cv2.VideoCapture(video), cv2.VideoCapture(dst)
    rows = []
    track_of = {}
    for ti, tr in enumerate(tracks):
        for t, box in tr:
            track_of[(t, tuple(box))] = ti
    embeds_by_track = {}
    for t, box in faces:
        cap_o.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        cap_r.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok_o, fo = cap_o.read()
        ok_r, fr = cap_r.read()
        if not (ok_o and ok_r):
            continue
        h, w = fo.shape[:2]
        x1, y1 = max(0, box[0]), max(0, box[1])
        x2, y2 = min(w, box[2]), min(h, box[3])
        if x2 <= x1 or y2 <= y1:
            continue
        co, cr = fo[y1:y2, x1:x2], fr[y1:y2, x1:x2]
        eo, er = _embed(rec, co), _embed(rec, cr)
        sim = _cos(eo, er)
        pixdiff = float(np.abs(co.astype(np.int16)
                               - cr.astype(np.int16)).mean())
        still = len(det.find(fr[max(0, y1 - 20):y2 + 20,
                                max(0, x1 - 20):x2 + 20])) > 0
        ti = track_of.get((t, tuple(box)))
        if eo is not None and ti is not None:
            embeds_by_track.setdefault(ti, []).append(eo)
        rows.append({"t": round(t, 2), "box": [x1, y1, x2, y2], "track": ti,
                     "similarity": None if sim is None else round(sim, 4),
                     "pix_diff": round(pixdiff, 2),
                     "still_detectable": bool(still)})
    cap_o.release()
    cap_r.release()

    # per-person positive control + cross-person chance baseline
    same, cross = [], []
    tids = sorted(embeds_by_track)
    for ti in tids:
        es = embeds_by_track[ti]
        for i in range(0, len(es) - 1, max(1, len(es) // 12)):
            c = _cos(es[i], es[i + 1])
            if c is not None:
                same.append(c)
    for ai in range(len(tids)):
        for bi in range(ai + 1, len(tids)):
            c = _cos(embeds_by_track[tids[ai]][0],
                     embeds_by_track[tids[bi]][0])
            if c is not None:
                cross.append(c)
    ctrl = float(np.mean(same)) if same else None

    sims = [r["similarity"] for r in rows if r["similarity"] is not None]
    matched = [r for r in rows
               if r["similarity"] is not None
               and r["similarity"] >= SAME_PERSON]
    untouched = [r for r in matched if r["pix_diff"] < 2.0]
    return {
        "video": os.path.basename(video), "mode": mode,
        "coverage": coverage,
        "faces": len(rows), "tracks": len(tracks),
        "unconfirmed_candidates": unconfirmed,
        "still_detectable": sum(1 for r in rows if r["still_detectable"]),
        "compared": len(sims),
        "reidentified": len(matched),
        "reid_rate": len(matched) / max(1, len(sims)),
        "reid_untouched_pixels": len(untouched),
        "sim_mean": round(float(np.mean(sims)), 4) if sims else None,
        "sim_max": round(float(max(sims)), 4) if sims else None,
        "same_person_threshold": SAME_PERSON,
        "control_same_person": round(ctrl, 4) if ctrl is not None else None,
        "chance_cross_person": (round(float(np.mean(cross)), 4)
                                if cross else None),
        "control_valid": bool(ctrl is not None and ctrl >= SAME_PERSON),
        "rows": rows,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", default=os.path.join(here, "_results"))
    ap.add_argument("--redacted", default=None,
                    help="existing redacted render to attack (skips the "
                         "pipeline run)")
    ap.add_argument("--mode", default="blur",
                    choices=["blur", "box", "mosaic", "inpaint"])
    ap.add_argument("--coverage", default="tight",
                    choices=["tight", "box", "concealed"])
    ap.add_argument("--categories", default="face,person")
    ap.add_argument("--json", default="")
    ap.add_argument("rest", nargs="*")
    a = ap.parse_args()

    print("OpenScrub v%s | re-identification attack" % openscrub.VERSION)
    res = run(a.video, a.out, a.mode, a.coverage, a.categories, a.rest,
              redacted=a.redacted)
    if res is None:
        return 1
    print("\n" + "=" * 62)
    print(" confirmed faces attacked   %d  (%d identity tracks; %d "
          "unconfirmed candidates excluded)"
          % (res["faces"], res["tracks"], res["unconfirmed_candidates"]))
    print(" still face-detectable      %d  (%.1f%%)"
          % (res["still_detectable"],
             100 * res["still_detectable"] / max(1, res["faces"])))
    print(" RE-IDENTIFIED              %d/%d  (%.1f%%)  [threshold %.2f]"
          % (res["reidentified"], res["compared"],
             100 * res["reid_rate"], SAME_PERSON))
    print("   of which untouched pixels %d  (pipeline misses, not weak "
          "redaction)" % res["reid_untouched_pixels"])
    print(" similarity mean / max      %s / %s"
          % (res["sim_mean"], res["sim_max"]))
    print(" control same-person        %s  (must be >= %.2f)"
          % (res["control_same_person"], SAME_PERSON))
    print(" chance cross-person        %s" % res["chance_cross_person"])
    print("=" * 62)
    if not res["control_valid"]:
        print(" WARNING: the recognizer could not reliably match this"
              " person to\n themselves in the UNREDACTED footage"
              " (control %s < %.2f), so the\n re-identification number"
              " above is NOT evidence the redaction works."
              % (res["control_same_person"], SAME_PERSON))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(dict(res, version=openscrub.VERSION), f, indent=1)
        print(" raw -> %s" % a.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())

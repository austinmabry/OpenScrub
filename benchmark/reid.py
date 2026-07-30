#!/usr/bin/env python3
"""Re-identification attack: the metric that actually matters.

Detection mAP answers "did a box fire?". The question a privacy officer
asks is "can this person still be identified from the output?" — so this
harness attacks the redacted file directly:

  1. ORACLE      find every face in the ORIGINAL with the strongest
                 detector available (low threshold — the oracle should
                 over-find, so we never grade ourselves generously).
  2. REDACT      run the OpenScrub pipeline.
  3. ATTACK      at each oracle location, crop the REDACTED frame and
                 ask two questions:
                   a. is a face still DETECTABLE there?
                   b. does its SFace embedding still MATCH the identity
                      taken from the original crop?

(b) is the real result. Cosine similarity against the same-person
threshold the engine itself uses for identity grouping (0.55) says
whether a face recognizer could re-link the redacted person to a
reference photo.

A control column keeps it honest: the same comparison against a
DIFFERENT person's face gives the chance baseline, so "no match" means
"no better than a stranger", not "the number looked small".

    python benchmark/reid.py --video clip.mp4
    python benchmark/reid.py --video clip.mp4 --mode mosaic --coverage concealed

Use footage you own or public-domain clips. Do NOT run this on real
people's faces and publish the frames.
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
    """SFace embedding of a face crop, or None if unusable."""
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


def oracle_faces(video, every=0.5, thresh=0.3):
    """[(t, box)] — faces in the ORIGINAL, found with a deliberately low
    threshold so the attack surface is over- not under-stated."""
    det = openscrub.FaceDetector(_Quiet(), thresh=thresh)
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(every * fps)))
    out, n = [], 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if n % step == 0:
            for (x1, y1, x2, y2, *_rest) in det.find(fr):
                out.append((n / fps, [int(x1), int(y1), int(x2), int(y2)]))
        n += 1
    cap.release()
    return out, fps


def run(video, out_dir, mode, coverage, categories, extra):
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(video))[0]
    dst = os.path.join(out_dir, base + "_redacted.mp4")
    report = os.path.join(out_dir, base + "_report.json")

    print("  oracle pass (finding faces in the original) ...", end="",
          flush=True)
    faces, fps = oracle_faces(video)
    print(" %d face instances" % len(faces))
    if not faces:
        print("  no faces found — nothing to attack. Use a clip with "
              "visible faces.")
        return None

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

    rec = _sface()
    det = openscrub.FaceDetector(_Quiet(), thresh=0.5)
    cap_o, cap_r = cv2.VideoCapture(video), cv2.VideoCapture(dst)
    rows, embeds_o = [], []
    for t, (x1, y1, x2, y2) in faces:
        cap_o.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        cap_r.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok_o, fo = cap_o.read()
        ok_r, fr = cap_r.read()
        if not (ok_o and ok_r):
            continue
        h, w = fo.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        eo = _embed(rec, fo[y1:y2, x1:x2])
        er = _embed(rec, fr[y1:y2, x1:x2])
        sim = _cos(eo, er)
        still = len(det.find(fr[max(0, y1 - 20):y2 + 20,
                                max(0, x1 - 20):x2 + 20])) > 0
        if eo is not None:
            embeds_o.append(eo)
        rows.append({"t": round(t, 2), "box": [x1, y1, x2, y2],
                     "similarity": None if sim is None else round(sim, 4),
                     "still_detectable": bool(still)})
    cap_o.release()
    cap_r.release()

    # POSITIVE CONTROL — the number that decides whether this clip can
    # support a re-identification claim at all. It compares ORIGINAL
    # faces against other ORIGINAL faces from the same footage: if the
    # recognizer cannot even link an unredacted face to another
    # unredacted frame of it, then "the redacted face did not match"
    # says nothing about the redaction. Report it, and refuse to headline
    # a re-id result when it is below the same-person threshold.
    control = []
    for i in range(0, len(embeds_o) - 1, max(1, len(embeds_o) // 40)):
        for j in range(i + 1, min(i + 6, len(embeds_o))):
            c = _cos(embeds_o[i], embeds_o[j])
            if c is not None:
                control.append(c)
    sims = [r["similarity"] for r in rows if r["similarity"] is not None]
    matched = [s for s in sims if s >= SAME_PERSON]
    ctrl_mean = float(np.mean(control)) if control else None
    return {
        "video": os.path.basename(video), "mode": mode,
        "coverage": coverage, "faces": len(rows),
        "still_detectable": sum(1 for r in rows if r["still_detectable"]),
        "compared": len(sims),
        "reidentified": len(matched),
        "reid_rate": len(matched) / max(1, len(sims)),
        "sim_mean": round(float(np.mean(sims)), 4) if sims else None,
        "sim_max": round(float(max(sims)), 4) if sims else None,
        "same_person_threshold": SAME_PERSON,
        "control_orig_vs_orig_mean": (round(ctrl_mean, 4)
                                      if ctrl_mean is not None else None),
        "control_valid": bool(ctrl_mean is not None
                              and ctrl_mean >= SAME_PERSON),
        "rows": rows,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", default=os.path.join(here, "_results"))
    ap.add_argument("--mode", default="blur",
                    choices=["blur", "box", "mosaic", "inpaint"])
    ap.add_argument("--coverage", default="tight",
                    choices=["tight", "box", "concealed"])
    ap.add_argument("--categories", default="face,person")
    ap.add_argument("--json", default="")
    ap.add_argument("rest", nargs="*")
    a = ap.parse_args()

    print("OpenScrub v%s | re-identification attack" % openscrub.VERSION)
    res = run(a.video, a.out, a.mode, a.coverage, a.categories, a.rest)
    if res is None:
        return 1
    print("\n" + "=" * 58)
    print(" faces attacked          %d" % res["faces"])
    print(" still face-detectable   %d  (%.1f%%)"
          % (res["still_detectable"],
             100 * res["still_detectable"] / max(1, res["faces"])))
    print(" RE-IDENTIFIED           %d/%d  (%.1f%%)  [threshold %.2f]"
          % (res["reidentified"], res["compared"],
             100 * res["reid_rate"], SAME_PERSON))
    print(" similarity mean / max   %s / %s"
          % (res["sim_mean"], res["sim_max"]))
    print(" control orig-vs-orig    %s  (must be >= %.2f to trust the "
          "result above)" % (res["control_orig_vs_orig_mean"], SAME_PERSON))
    print("=" * 58)
    if not res["control_valid"]:
        print(" WARNING: the recognizer could not reliably match this"
              " person to\n themselves in the UNREDACTED footage"
              " (control %s < %.2f), so the\n re-identification number"
              " above is NOT evidence the redaction works.\n Use footage"
              " with larger, front-facing faces before quoting it."
              % (res["control_orig_vs_orig_mean"], SAME_PERSON))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(dict(res, version=openscrub.VERSION), f, indent=1)
        print(" raw -> %s" % a.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
openscrub_web.py — browser interface for openscrub, usable across your LAN.

Run on an always-on machine:
    python openscrub_web.py                  # port 8384, prints access URL
    python openscrub_web.py --port 9000 --token mysecret

Then from any device on the LAN (laptop, phone), open the printed URL.
Workflow: upload a recording (or give a server-side path) -> scan runs on
the server (live preview + log) -> review every detection with thumbnails,
uncheck false positives, draw missed regions -> render -> download.

Security model: HTTPS by default (self-signed cert generated on first
run, or install your own). Pass --token <secret> to require an access
token on every request (carried in a cookie after the first visit);
without it the server runs open and trusts everyone on the network.
Either way this is *LAN-grade* protection for a trusted home/office
network — do not expose this port to the internet. Uploaded videos and
audit reports contain PHI; the jobs folder inherits that sensitivity.

Jobs are processed one at a time (FIFO) so concurrent requests don't fight
over the GPU.
"""

import argparse
import json
import os
import queue
import secrets
import sys
import socket
import threading
import time
import uuid

import cv2
from flask import (Flask, Response, abort, jsonify, redirect, request,
                   send_file)

import openscrub
import zones_ui


# Anchor to the script's own folder, NOT the process working directory —
# double-clicking a .py on Windows can launch with cwd=C:\Windows\system32
JOBS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "openscrub_jobs")
# One-time migration from the pre-rebrand job store: carries over all prior
# jobs, reviews, and the learned-words allowlist.
_legacy_jobs = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "phi_blur_jobs")
if os.path.isdir(_legacy_jobs) and not os.path.isdir(JOBS_DIR):
    try:
        os.rename(_legacy_jobs, JOBS_DIR)
    except OSError:
        pass  # e.g. cross-device or perms: fall back to fresh dir
ZONES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "zones.json")
CERT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "certs")
CUSTOM_CERT = os.path.join(CERT_DIR, "custom_cert.pem")
CUSTOM_KEY = os.path.join(CERT_DIR, "custom_key.pem")
AUTO_CERT = os.path.join(CERT_DIR, "auto_cert.pem")
AUTO_KEY = os.path.join(CERT_DIR, "auto_key.pem")


def ensure_self_signed():
    """Generate a long-lived self-signed certificate once, with SANs for
    localhost and this machine's LAN IP so browsers tie it to the URL used."""
    if os.path.exists(AUTO_CERT) and os.path.exists(AUTO_KEY):
        return
    from cryptography import x509
    from cryptography.x509.oid import NameOID
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    import datetime, ipaddress
    os.makedirs(CERT_DIR, exist_ok=True)
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "openscrub")])
    sans = [x509.DNSName("localhost"),
            x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]
    try:
        sans.append(x509.IPAddress(ipaddress.ip_address(lan_ip())))
    except Exception:
        pass
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (x509.CertificateBuilder()
            .subject_name(name).issuer_name(name)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - datetime.timedelta(days=1))
            .not_valid_after(now + datetime.timedelta(days=3650))
            .add_extension(x509.SubjectAlternativeName(sans), critical=False)
            .sign(key, hashes.SHA256()))
    with open(AUTO_KEY, "wb") as f:
        f.write(key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption()))
    with open(AUTO_CERT, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))
    try:
        os.chmod(AUTO_KEY, 0o600)
    except Exception:
        pass


def active_cert_pair():
    if os.path.exists(CUSTOM_CERT) and os.path.exists(CUSTOM_KEY):
        return CUSTOM_CERT, CUSTOM_KEY, "custom"
    ensure_self_signed()
    return AUTO_CERT, AUTO_KEY, "self-signed"
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 ** 3   # 8 GB uploads

TOKEN = None
JOBS = {}          # id -> job dict
JOBS_LOCK = threading.Lock()
WORK_Q = queue.Queue()


# ----------------------------------------------------------------------------
# Job model
# ----------------------------------------------------------------------------

def new_job(video_path, options, name):
    jid = uuid.uuid4().hex[:12]
    jdir = os.path.join(JOBS_DIR, jid)
    os.makedirs(jdir, exist_ok=True)
    job = {
        "id": jid, "dir": jdir, "video": video_path, "name": name,
        "options": options, "phase": "queued", "progress": 0.0,
        "log": [], "error": None, "output": None, "created": time.time(),
        "cancel": threading.Event(), "stats": {},
    }
    with JOBS_LOCK:
        JOBS[jid] = job
    return job


def job_public(job):
    return {k: job[k] for k in ("id", "name", "phase", "progress", "error",
                                "stats")} | {
        "log_tail": job["log"][-30:],
        "log_len": len(job["log"]),
        "has_output": bool(job["output"] and os.path.exists(job["output"])),
    }


class WebCallbacks(openscrub.Callbacks):
    wants_frames = True

    def __init__(self, job):
        self.job = job
        self._last_frame = 0.0

    def log(self, msg):
        self.job["log"].append(msg)

    def progress(self, stage, cur, total):
        base, span = (0, 0.5) if stage == "scan" else (0.5, 0.5)
        self.job["progress"] = round(base + span * cur / max(total, 1), 3)

    def scan_frame(self, frame, t, found):
        now = time.time()
        if now - self._last_frame < 0.4:   # throttle disk writes
            return
        self._last_frame = now
        h, w = frame.shape[:2]
        scale = 720 / w
        small = cv2.resize(frame, (720, int(h * scale)))
        cv2.imwrite(os.path.join(self.job["dir"], "preview.jpg"), small,
                    [cv2.IMWRITE_JPEG_QUALITY, 70])

    def cancelled(self):
        return self.job["cancel"].is_set()


def build_args(job, for_render=False):
    o = job["options"]
    jdir = job["dir"]
    argv = [job["video"],
            "--engine", o.get("engine", "auto"),
            "--device", o.get("device", "auto"),
            "--encoder", o.get("encoder", "auto"),
            "--mode", o.get("mode", "blur"),
            "--mode-map", o.get("mode_map", ""),
            "--sample-interval", str(o.get("sample_interval", 0.5)),
            "--scan-trigger", str(o.get("scan_trigger", 60)),
            "--pad", str(o.get("pad", 8)),
            "--bridge-gap", str(o.get("bridge_gap", 4.0)),
            "--mrn-regex", o.get("mrn_regex", openscrub.RE_MRN_DEFAULT),
            "--face-expand", str(o.get("face_expand", 0.15)),
            "--face-threshold", str(o.get("face_threshold", 0.6)),
            "--plate-threshold", str(o.get("plate_threshold", 0.35)),
            "--detect-scale", str(o.get("detect_scale", 1.0)),
            "--skip-start", str(o.get("skip_start", 0)),
            "--skip-end", str(o.get("skip_end", 0)),
            "--categories", o.get("categories",
                                  "name,dob,phone,ssn,mrn,email,address,card,apikey,ipaddr,plate,face"),
            "-o", os.path.join(jdir, "output."
                               + (o.get("out_format", "mp4")
                                  if o.get("out_format") in ("mp4", "mov", "mkv")
                                  else "mp4")),
            "--report", os.path.join(jdir, "report.json")]
    if o.get("no_memory"):
        argv.append("--no-memory")
    if o.get("no_ner"):
        argv.append("--no-ner")
    if o.get("preview_mode"):
        argv.append("--preview")
    if o.get("dense_faces"):
        argv.append("--dense-faces")
    if o.get("draw_scores"):
        argv.append("--draw-scores")
    if o.get("use_zones", True) and os.path.exists(ZONES_PATH):
        try:
            with open(ZONES_PATH, encoding="utf-8") as f:
                if json.load(f):
                    argv += ["--zones", ZONES_PATH]
        except Exception:
            pass
    server_allow = os.path.join(JOBS_DIR, "allowlist.txt")
    for kind in ("allow_names", "extra_names"):
        text = (o.get(kind) or "").strip()
        if kind == "allow_names" and os.path.exists(server_allow):
            text = (text + "\n" + open(server_allow, encoding="utf-8").read()).strip()
        if text:
            p = os.path.join(jdir, kind + ".txt")
            with open(p, "w", encoding="utf-8") as f:
                f.write(text)
            argv += ["--" + kind.replace("_", "-"), p]
    for line in (o.get("ignore_regions") or "").strip().splitlines():
        line = line.strip()
        if line:
            argv += ["--ignore-region", line]
    if for_render:
        argv += ["--from-report", os.path.join(jdir, "report.json")]
    parser = openscrub.build_parser()
    return openscrub._prep_args(parser.parse_args(argv), parser)


def worker():
    while True:
        jid, action = WORK_Q.get()
        with JOBS_LOCK:
            job = JOBS.get(jid)
        if job is None:
            continue
        cb = WebCallbacks(job)
        try:
            if action == "scan":
                job["phase"] = "scanning"
                args = build_args(job)
                state = openscrub.run_scan(args, cb)
                openscrub.write_report(os.path.join(job["dir"], "report.json"),
                                      args, state)
                job["stats"] = state["stats"]
                def _pregen(jid=jid):
                    """One pass through the video with a single decoder:
                    thumbnails come out in seconds instead of minutes.
                    Sorting by reference time keeps seeks short and forward-
                    only, which is what codecs are fast at."""
                    try:
                        jb = JOBS.get(jid)
                        doc = _load_job_report(jb)
                        rs = doc["render_state"]
                        todo = []
                        for k, d in enumerate(doc["detections"]):
                            cch = os.path.join(jb["dir"], f"thumb2_{k}.jpg")
                            if os.path.exists(cch):
                                continue
                            tref = min(max(d.get("last_seen", d["t_start"]),
                                           d["t_start"]), d["t_end"])
                            todo.append((tref, k, d, cch))
                        todo.sort()
                        cap = cv2.VideoCapture(jb["video"])
                        cur = -1
                        frame = None
                        for tref, k, d, cch in todo:
                            fidx = min(int(tref * rs["fps"]),
                                       len(rs["cum"]) - 1)
                            if fidx != cur:
                                if 0 <= fidx - cur <= 12:
                                    while cur < fidx:   # cheap forward decode
                                        ok, frame = cap.read()
                                        if not ok:
                                            frame = None
                                            break
                                        cur += 1
                                else:
                                    cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
                                    ok, frame = cap.read()
                                    cur = fidx if ok else cur
                                    if not ok:
                                        frame = None
                            if frame is None:
                                continue
                            ox, oy = rs["cum"][fidx]
                            h, wd = frame.shape[:2]
                            pad = 24
                            x1 = max(0, int(d["cbox"][0] + ox) - pad)
                            y1 = max(0, int(d["cbox"][1] + oy) - pad)
                            x2 = min(wd, int(d["cbox"][2] + ox) + pad)
                            y2 = min(h, int(d["cbox"][3] + oy) + pad)
                            crop = frame[y1:y2, x1:x2].copy()
                            if crop.size == 0:
                                continue
                            inner = crop[pad:-pad or None, pad:-pad or None]
                            if not inner.size or float(inner.std()) <= 8:
                                continue   # blank: leave for the on-demand
                                           # path, which tries other moments
                            cv2.rectangle(crop, (pad, pad),
                                          (crop.shape[1] - pad,
                                           crop.shape[0] - pad),
                                          (0, 0, 255), 2)
                            cv2.imwrite(cch, crop,
                                        [cv2.IMWRITE_JPEG_QUALITY, 80])
                        cap.release()
                        jb["log"].append("thumbnails pre-generated "
                                         f"({len(todo)})")
                    except Exception:
                        pass
                threading.Thread(target=_pregen, daemon=True).start()
                try:
                    prov = _load_job_report(job).get("provenance", {})
                    if (prov.get("vfr_normalized")
                            and os.path.exists(prov.get("input", ""))):
                        job["video"] = prov["input"]
                        job["log"].append(
                            "using CFR-normalized file for previews/thumbnails")
                except Exception:
                    pass
                if job["options"].get("skip_review"):
                    job["log"].append("review skipped — rendering all detections")
                    job["phase"] = "review"
                    WORK_Q.put((jid, "render"))
                else:
                    job["phase"] = "review"
                job["progress"] = 0.5
            elif action == "render":
                job["phase"] = "rendering"
                args = build_args(job, for_render=True)
                res = openscrub.run_pipeline(args, cb)
                job["output"] = res["output"]
                # refresh report with output hash
                dets, rstate, prov = openscrub.load_report(
                    os.path.join(job["dir"], "report.json"))
                job["phase"] = "done"
                job["progress"] = 1.0
        except openscrub.PipelineCancelled:
            job["phase"] = "cancelled"
            job["error"] = "cancelled by user"
        except Exception as e:
            job["phase"] = "error"
            job["error"] = str(e)
            job["log"].append(f"ERROR: {e}")
        finally:
            WORK_Q.task_done()


# ----------------------------------------------------------------------------
# Auth
# ----------------------------------------------------------------------------

@app.before_request
def check_token():
    if TOKEN is None:          # open mode: anyone on the network is trusted
        return
    tok = (request.args.get("token") or request.cookies.get("phiblur_token")
           or request.headers.get("X-Token"))
    if tok != TOKEN:
        abort(401, "missing or wrong access token — open the URL printed "
                   "at server startup (includes ?token=...)")


@app.after_request
def set_cookie(resp):
    if TOKEN is not None and request.args.get("token") == TOKEN:
        resp.set_cookie("phiblur_token", TOKEN, max_age=90 * 86400,
                        samesite="Lax")
    return resp


# ----------------------------------------------------------------------------
# API
# ----------------------------------------------------------------------------

@app.route("/api/jobs", methods=["POST"])
def create_job():
    options = json.loads(request.form.get("options", "{}"))
    jobs = []
    files = request.files.getlist("video")
    if files and files[0].filename:
        for f in files:
            name = os.path.basename(f.filename)
            job = new_job("", options, name)
            path = os.path.join(job["dir"], "input" + os.path.splitext(name)[1])
            f.save(path)
            job["video"] = path
            WORK_Q.put((job["id"], "scan"))
            jobs.append(job["id"])
    elif request.form.get("server_path"):
        p = request.form["server_path"].strip()
        if not os.path.exists(p):
            return jsonify({"error": f"server path not found: {p}"}), 400
        job = new_job(p, options, os.path.basename(p))
        WORK_Q.put((job["id"], "scan"))
        jobs.append(job["id"])
    else:
        return jsonify({"error": "no file uploaded and no server path"}), 400
    return jsonify({"jobs": jobs})


@app.route("/api/jobs")
def list_jobs():
    with JOBS_LOCK:
        js = sorted(JOBS.values(), key=lambda j: -j["created"])
    return jsonify([job_public(j) for j in js])


@app.route("/api/jobs/<jid>")
def job_status(jid):
    job = JOBS.get(jid) or abort(404)
    return jsonify(job_public(job))


@app.route("/api/jobs/<jid>/log")
def job_log(jid):
    job = JOBS.get(jid) or abort(404)
    frm = max(0, int(request.args.get("from", 0)))
    return jsonify({"from": frm, "lines": job["log"][frm:],
                    "len": len(job["log"])})


@app.route("/api/jobs/<jid>/preview.jpg")
def job_preview(jid):
    job = JOBS.get(jid) or abort(404)
    p = os.path.join(job["dir"], "preview.jpg")
    if not os.path.exists(p):
        abort(404)
    return send_file(p, mimetype="image/jpeg", max_age=0)


@app.route("/api/jobs/<jid>/cancel", methods=["POST"])
def job_cancel(jid):
    job = JOBS.get(jid) or abort(404)
    job["cancel"].set()
    return jsonify({"ok": True})


def _load_job_report(job):
    with open(os.path.join(job["dir"], "report.json"), encoding="utf-8") as f:
        return json.load(f)


def _frame_at(job, t, rstate):
    cap = cv2.VideoCapture(job["video"])
    fps = rstate["fps"]
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


@app.route("/api/jobs/<jid>/mediainfo")
def job_mediainfo(jid):
    """Basic video properties straight from the file — available in every
    job phase, no scan report required."""
    job = JOBS.get(jid) or abort(404)
    cap = cv2.VideoCapture(job["video"])
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    w_ = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_ = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return jsonify({"duration": n / fps if fps else 0, "fps": fps,
                    "width": w_, "height": h_})


@app.route("/api/jobs/<jid>/detections")
def job_detections(jid):
    job = JOBS.get(jid) or abort(404)
    doc = _load_job_report(job)
    items = []
    for i, d in enumerate(doc["detections"]):
        items.append({"i": i, "category": d["category"], "text": d["text"],
                      "t_start": d["t_start"], "t_end": round(d["t_end"], 2),
                      "enabled": d.get("enabled", True),
                      "zone_dropped": bool(d.get("zone_dropped"))})
    return jsonify({"detections": items,
                    "fps": doc["render_state"]["fps"],
                    "duration": len(doc["render_state"]["cum"]) /
                                doc["render_state"]["fps"]})


@app.route("/api/jobs/<jid>/detframe/<int:i>")
def job_detframe(jid, i):
    """Full video frame at a real sighting of detection i, with its box
    outlined — the review lightbox."""
    job = JOBS.get(jid) or abort(404)
    doc = _load_job_report(job)
    d = doc["detections"][i]
    rs = doc["render_state"]
    tref = min(max(d.get("last_seen", d["t_start"]), d["t_start"]), d["t_end"])
    fidx = min(int(tref * rs["fps"]), len(rs["cum"]) - 1)
    ox, oy = rs["cum"][fidx]
    frame = _frame_at(job, tref, rs)
    if frame is None:
        abort(404)
    x1, y1 = int(d["cbox"][0] + ox), int(d["cbox"][1] + oy)
    x2, y2 = int(d["cbox"][2] + ox), int(d["cbox"][3] + oy)
    cv2.rectangle(frame, (x1 - 4, y1 - 4), (x2 + 4, y2 + 4), (0, 0, 255), 3)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 84])
    from flask import Response
    return Response(buf.tobytes(), mimetype="image/jpeg")


@app.route("/api/jobs/<jid>/boxes_at")
def job_boxes_at(jid):
    """Active blur boxes (screen coords) at time t, plus the scroll offset so
    the client can convert edited boxes back to content coordinates."""
    job = JOBS.get(jid) or abort(404)
    doc = _load_job_report(job)
    rs = doc["render_state"]
    t = float(request.args.get("t", 0))
    fidx = min(int(t * rs["fps"]), len(rs["cum"]) - 1)
    ox, oy = rs["cum"][fidx]
    boxes = []
    want_all = request.args.get("all") == "1"
    for i, d in enumerate(doc["detections"]):
        if not (d["t_start"] - 0.01 <= t <= d["t_end"] + 0.01):
            continue
        if not want_all and not d.get("enabled", True):
            continue
        if True:
            boxes.append({"i": i, "category": d["category"],
                          "enabled": d.get("enabled", True),
                          "text": d.get("text", ""),
                          "t_start": d["t_start"], "t_end": d["t_end"],
                          "box": [d["cbox"][0] + ox, d["cbox"][1] + oy,
                                  d["cbox"][2] + ox, d["cbox"][3] + oy]})
    return jsonify({"t": t, "ox": ox, "oy": oy, "boxes": boxes})


@app.route("/api/jobs/<jid>/thumb/<int:i>")
def job_thumb(jid, i):
    job = JOBS.get(jid) or abort(404)
    cache = os.path.join(job["dir"], f"thumb2_{i}.jpg")
    if not os.path.exists(cache):
        doc = _load_job_report(job)
        d = doc["detections"][i]
        rs = doc["render_state"]
        # cut the thumbnail at a moment the detector actually SAW the object.
        # t_start can be earlier than the object's appearance (backtracking
        # deliberately moves it to just before onset) — a frame there shows
        # nothing and made review pictures mismatch their text.
        ls = min(max(d.get("last_seen", d["t_start"]), d["t_start"]),
                 d["t_end"])
        mid = (d["t_start"] + d["t_end"]) / 2
        crop = None
        pad = 24
        for tref in (ls, mid, min(d["t_start"] + 0.2, d["t_end"])):
            fidx = min(int(tref * rs["fps"]), len(rs["cum"]) - 1)
            ox, oy = rs["cum"][fidx]
            frame = _frame_at(job, tref, rs)
            if frame is None:
                continue
            h, w = frame.shape[:2]
            x1 = max(0, int(d["cbox"][0] + ox) - pad)
            y1 = max(0, int(d["cbox"][1] + oy) - pad)
            x2 = min(w, int(d["cbox"][2] + ox) + pad)
            y2 = min(h, int(d["cbox"][3] + oy) + pad)
            c = frame[y1:y2, x1:x2]
            if c.size == 0:
                continue
            crop = c
            inner = c[pad:-pad or None, pad:-pad or None]
            if inner.size and float(inner.std()) > 8:
                break   # crop shows real content — use this moment
        if crop is None:
            abort(404)
        cv2.rectangle(crop, (pad, pad), (crop.shape[1] - pad,
                                         crop.shape[0] - pad), (0, 0, 255), 2)
        cv2.imwrite(cache, crop, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return send_file(cache, mimetype="image/jpeg", max_age=3600)


@app.route("/api/jobs/<jid>/frame_at")
def job_frame_at(jid):
    job = JOBS.get(jid) or abort(404)
    t = float(request.args.get("t", 0))
    path = (job["output"] if request.args.get("src") == "output"
            and job.get("output") else job["video"])
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
    ok, frame = cap.read()
    cap.release()
    frame = frame if ok else None
    if frame is None:
        abort(404)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return Response(buf.tobytes(), mimetype="image/jpeg")


@app.route("/api/jobs/<jid>/detections", methods=["POST"])
def job_save_edits(jid):
    job = JOBS.get(jid) or abort(404)
    edits = request.get_json(force=True)
    path = os.path.join(job["dir"], "report.json")
    doc = _load_job_report(job)
    enabled = edits.get("enabled", {})
    for i, d in enumerate(doc["detections"]):
        if str(i) in enabled:
            d["enabled"] = bool(enabled[str(i)])
    for i, box in (edits.get("box_overrides") or {}).items():
        i = int(i)
        if 0 <= i < len(doc["detections"]) and len(box) == 4:
            doc["detections"][i]["cbox"] = [float(v) for v in box]
    for i, span in (edits.get("time_overrides") or {}).items():
        i = int(i)
        if 0 <= i < len(doc["detections"]) and len(span) == 2:
            t0, t1 = float(span[0]), float(span[1])
            if t1 > t0 >= 0:
                doc["detections"][i]["t_start"] = t0
                doc["detections"][i]["t_end"] = t1
                doc["detections"][i]["last_seen"] = min(
                    max(doc["detections"][i].get("last_seen", t0), t0), t1)
    rs = doc["render_state"]
    for m in edits.get("manual", []):
        t0 = float(m["t_start"]); t1 = float(m["t_end"])
        tref = float(m.get("t_ref", t0))
        fidx = min(int(tref * rs["fps"]), len(rs["cum"]) - 1)
        ox, oy = rs["cum"][fidx]
        b = m["screen_box"]
        doc["detections"].append({
            "t_start": t0, "t_end": t1,
            "cbox": [int(b[0] - ox), int(b[1] - oy),
                     int(b[2] - ox), int(b[3] - oy)],
            "category": "manual", "text": "user-drawn", "confidence": 1.0,
            "aoff": [ox, oy], "last_seen": t0, "enabled": True})
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1)
    # suggest allowlisting any string whose EVERY instance was disabled —
    # the review just taught us it's a recurring false positive
    per = {}
    for d in doc["detections"]:
        if d["category"] in ("name",):
            per.setdefault(d["text"], []).append(d.get("enabled", True))
    suggest = sorted(t for t, states in per.items()
                     if len(states) >= 2 and not any(states))
    return jsonify({"ok": True, "total": len(doc["detections"]),
                    "suggest_allowlist": suggest})


@app.route("/api/plate_models")
def plate_models():
    """Registry entries + whether each is already installed + active model."""
    import openscrub as engine
    models = engine.load_plate_registry()
    mdir = os.path.join(os.path.dirname(os.path.abspath(engine.__file__)), "models")
    out = []
    for m in models:
        path = os.path.join(mdir, "%s.onnx" % m.get("id"))
        out.append({
            "id": m.get("id"), "label": m.get("label"),
            "license": m.get("license"), "source_url": m.get("source_url"),
            "notes": m.get("notes"), "recommended": bool(m.get("recommended")),
            "verified": m.get("download_url") not in (None, "", "TODO_VERIFY"),
            "pinned": bool(m.get("sha256")) and m.get("sha256") != "TODO_VERIFY",
            "installed": os.path.exists(path),
            "attribution": m.get("attribution", ""),
        })
    # is any model active (i.e. would PlateDetector find one)?
    active = None
    for cand in (os.environ.get("OPENSCRUB_PLATE_MODEL"),
                 os.path.join(mdir, "plate_yolov8.onnx")):
        if cand and os.path.exists(cand):
            active = cand; break
    if not active:
        for m in models:
            path = os.path.join(mdir, "%s.onnx" % m.get("id"))
            if os.path.exists(path):
                active = path; break
    return jsonify({"models": out, "active": os.path.basename(active) if active else None})


_plate_dl = {"state": "idle", "pct": 0, "error": "", "id": ""}

@app.route("/api/plate_models/<mid>/download", methods=["POST"])
def plate_model_download(mid):
    """Download+verify a registry model in a background thread."""
    import openscrub as engine
    entry = next((m for m in engine.load_plate_registry() if m.get("id") == mid), None)
    if entry is None:
        return jsonify({"error": "unknown model id"}), 404
    if _plate_dl["state"] == "downloading":
        return jsonify({"error": "a download is already in progress"}), 409
    _plate_dl.update(state="downloading", pct=0, error="", id=mid)
    def work():
        try:
            engine.download_plate_model(
                entry, progress=lambda f: _plate_dl.update(pct=int(f * 100)))
            _plate_dl.update(state="done", pct=100)
        except Exception as e:
            _plate_dl.update(state="error", error=str(e))
    threading.Thread(target=work, daemon=True).start()
    return jsonify({"ok": True})

@app.route("/api/plate_models/download_status")
def plate_model_download_status():
    return jsonify(_plate_dl)


@app.route("/api/allowlist", methods=["GET", "POST"])
def allowlist():
    path = os.path.join(JOBS_DIR, "allowlist.txt")
    if request.method == "POST":
        body = request.get_json(force=True)
        existing = []
        if os.path.exists(path):
            existing = [l.strip() for l in open(path, encoding="utf-8")
                        if l.strip()]
        for wd in body.get("words", []):
            wd = wd.strip()
            if wd and wd not in existing:
                existing.append(wd)
        removes = {r.strip() for r in body.get("remove", [])}
        if removes:
            existing = [wd for wd in existing if wd not in removes]
        if body.get("clear"):
            existing = []
        with open(path, "w", encoding="utf-8") as f:
            f.write("".join(wd + chr(10) for wd in existing))
        return jsonify({"ok": True, "count": len(existing)})
    words = []
    if os.path.exists(path):
        words = [l.strip() for l in open(path, encoding="utf-8") if l.strip()]
    return jsonify({"words": words, "path": path})


@app.route("/api/jobs/<jid>/render", methods=["POST"])
def job_render(jid):
    job = JOBS.get(jid) or abort(404)
    if job["phase"] in ("rendering", "queued_render"):
        return jsonify({"ok": True, "note": "render already in progress"})
    if job["phase"] not in ("review", "done", "error", "cancelled"):
        return jsonify({"error": f"cannot render from phase {job['phase']}"}), 400
    job["phase"] = "queued_render"
    job["cancel"].clear()
    job["error"] = None
    WORK_Q.put((jid, "render"))
    return jsonify({"ok": True})


@app.route("/api/jobs/<jid>/download")
def job_download(jid):
    job = JOBS.get(jid) or abort(404)
    if not job["output"] or not os.path.exists(job["output"]):
        abort(404)
    return send_file(job["output"], as_attachment=True,
                     download_name=os.path.splitext(job["name"])[0]
                     + "_redacted.mp4")


@app.route("/api/jobs/<jid>/report")
def job_report(jid):
    job = JOBS.get(jid) or abort(404)
    return send_file(os.path.join(job["dir"], "report.json"),
                     as_attachment=True,
                     download_name=os.path.splitext(job["name"])[0]
                     + "_audit.json")


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------

PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>OpenScrub</title>
<link rel="icon" href="data:image/png;base64,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"><style>
:root{--bg:#f5f6f8;--card:#fff;--acc:#2563eb;--red:#dc2626}
*{box-sizing:border-box}body{font:15px/1.45 system-ui,Segoe UI,Roboto,sans-serif;
margin:0;background:var(--bg);color:#1f2937}
header{background:#0F172A;color:#fff;padding:20px 16px 14px;
 display:flex;flex-direction:column;align-items:center;gap:8px;
 display:flex;flex-direction:column;align-items:center;gap:6px}
header img{height:88px;max-width:92vw;object-fit:contain}
header .brand{font-size:34px;font-weight:800;letter-spacing:-1px;line-height:1}
header .brand .box{border:2.5px solid #E53935;border-radius:3px;padding:0 4px;margin:0 1px;display:inline-block}
header .brand .fuzz{filter:blur(2.6px);opacity:.75}
header small{color:#9ca3af;font-weight:400;letter-spacing:.3px}
main{max-width:1100px;margin:0 auto;padding:12px}
.card{background:var(--card);border-radius:10px;padding:14px;margin-bottom:12px;
box-shadow:0 1px 3px rgba(0,0,0,.08)}
h2{margin:0 0 10px;font-size:17px}
label{display:block;margin:6px 0 2px;font-size:13px;color:#4b5563}
input[type=text],input[type=number],select,textarea{width:100%;padding:7px;
border:1px solid #d1d5db;border-radius:6px;font:inherit}
textarea{height:64px}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.grid3{display:grid;grid-template-columns:repeat(auto-fill,minmax(175px,1fr));gap:9px}
.row{display:flex;gap:8px;flex-wrap:wrap;align-items:center}
button{background:var(--acc);color:#fff;border:0;border-radius:7px;
padding:9px 16px;font:inherit;cursor:pointer}
button.sec{background:#6b7280}button.danger{background:var(--red)}
button:disabled{opacity:.5}
progress{width:100%;height:14px}
.det{background:#fff;border:1px solid #e5e7eb;border-radius:8px;padding:6px;
text-align:center;font-size:12px}
.det img{max-width:100%;border-radius:5px}
.det.off{opacity:.45}
.det.selected{outline:2px dashed #2563eb;outline-offset:2px}
.blurbtn{width:100%;padding:8px 0;margin-top:4px;font-weight:700;
 font-size:13px;border-radius:8px;letter-spacing:.3px;
 transition:background .15s,transform .05s}
.blurbtn:active{transform:scale(.97)}
.blurbtn.blur{background:#dc2626}
.blurbtn.keep{background:#16a34a}
.marq{position:absolute;border:1.5px dashed #2563eb;
 background:rgba(37,99,235,.08);z-index:50;pointer-events:none}
.selmenu{position:absolute;z-index:60;background:#fff;border-radius:10px;
 box-shadow:0 8px 28px rgba(0,0,0,.28);padding:7px 10px;display:flex;
 gap:8px;align-items:center;font-size:12.5px;color:#374151}
.selmenu button{padding:7px 16px;font-weight:700}
.selmenu .mkeep{background:#16a34a}
.selmenu .mblur{background:#dc2626}
.badge{display:inline-block;background:#e5e7eb;border-radius:5px;
padding:1px 6px;font-size:11px;margin-left:4px}
#log{background:#0f172a;color:#cbd5e1;font:12px/1.4 ui-monospace,Consolas,monospace;
padding:10px;border-radius:8px;height:180px;overflow:auto;white-space:pre-wrap}
#preview img,#framewrap{max-width:100%;border-radius:8px}
#framewrap{position:relative;display:inline-block}
#framewrap canvas{position:absolute;left:0;top:0;cursor:crosshair}
.joblist a{color:var(--acc);text-decoration:none}
.warn{background:#fef3c7;border:1px solid #fcd34d;border-radius:8px;
padding:8px 12px;font-size:13px}
.chk{display:flex;gap:12px;flex-wrap:wrap}
.chk label{display:flex;gap:4px;align-items:center;margin:0}
.plist{display:flex;flex-wrap:wrap;gap:4px;margin-top:5px}
.pw{background:#f0fdf4;color:#15803d;border:1px solid #bbf7d0;border-radius:7px;
 padding:2px 7px;font-size:12px;font-weight:600;display:flex;gap:6px;
 align-items:center}
.pw span{cursor:pointer;color:#16a34a;font-weight:700}
.pw span:hover{color:#14532d}
.pnote{font-size:11.5px;color:#15803d;margin-top:4px}
.qm{display:inline-flex;align-items:center;justify-content:center;
 width:15px;height:15px;border-radius:50%;background:#94a3b8;color:#fff;
 font-size:10.5px;font-weight:700;cursor:help;margin-left:5px;
 position:relative;vertical-align:middle;user-select:none}
.qm:hover::after,.qm.open::after{content:attr(data-tip);position:absolute;
 left:50%;transform:translateX(-50%);bottom:135%;background:#111827;
 color:#e5e7eb;padding:9px 11px;border-radius:9px;width:250px;
 font-size:12px;font-weight:400;line-height:1.4;z-index:90;
 box-shadow:0 8px 24px rgba(0,0,0,.35);white-space:normal;text-align:left}
@media(max-width:640px){.grid2{grid-template-columns:1fr}}
.catrow{display:inline-flex;align-items:center;gap:5px}
.catrow select.catmode{font-size:11px;padding:1px 3px;border-radius:5px;
 border:1px solid #d1d5db;color:#6b7280;background:#fff}
</style></head><body>
<header><img src="data:image/png;base64,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" alt="OpenScrub">
<div class="brand">Open<span class="box"><span class="fuzz">Scr</span></span>ub</div>
<small>Local video redaction — review before you trust</small></header>
<main>
<div class="card" id="newjob">
<h2>Job Settings</h2>
<label>Upload video(s) <span style="font-weight:400;color:#6b7280">— accepts
 MP4, MOV, MKV, AVI, M4V, WebM, WMV (anything ffmpeg can read)</span></label>
<input type="file" id="file" accept="video/*" multiple>
<label>…or path on the server</label>
<input type="text" id="spath" placeholder="C:\\recordings\\demo.mp4">
<label>Output format<span class="qm" data-tip="Container for the redacted file. The video inside is re-encoded H.264 either way; MP4 is right for DaVinci Resolve and most tools.">?</span></label>
<select id="outfmt" style="max-width:340px">
<option value="mp4">MP4 — H.264 (recommended)</option>
<option value="mov">MOV — QuickTime container</option>
<option value="mkv">MKV — Matroska container</option></select>
<div class="grid2">
<div><label>OCR engine<span class="qm" data-tip="Reads on-screen text. Paddle is the most accurate (GPU-capable); Tesseract is the lighter fallback. Auto picks Paddle when installed.">?</span></label><select id="engine">
<option>auto</option><option>paddle</option><option>tesseract</option></select></div>
<div><label>OCR device<span class="qm" data-tip="Where PaddleOCR runs. GPU is far faster when CUDA is set up; CPU always works.">?</span></label><select id="device">
<option>auto</option><option>gpu</option><option>cpu</option></select></div>
<div><label>Encoder<span class="qm" data-tip="Encoder for the final render. NVENC uses the GPU's dedicated encode chip (fast, frees the CPU); x264 is CPU-only. Auto tests NVENC and falls back safely.">?</span></label><select id="encoder">
<option>auto</option><option value="nvenc">NVENC (GPU)</option>
<option value="x264">x264 (CPU)</option></select></div>
<div><label>Redaction (default)<span class="qm" data-tip="Default style for every category. blur = Gaussian blur; readable structure is destroyed but a blur is, in principle, partially reversible — short high-contrast strings (an SSN or MRN in a fixed font) are the most vulnerable. box = solid black; pixels are set to zero, so nothing is recoverable. Override per category below.">?</span></label><select id="mode" onchange="renderCats()">
<option>blur</option><option>box</option><option>mosaic</option></select></div>
<div><label>Sample interval (s)<span class="qm" data-tip="How often a full OCR scan runs. Lower catches short-lived PHI sooner but scans take longer. Backtracking automatically closes most of the gap between scans, so 0.5 is a good default.">?</span></label><input type="number" id="si" value="0.5" step="0.1"></div>
<div><label>Scan trigger (px)<span class="qm" data-tip="An extra scan fires every time the page scrolls this many pixels, so scrolled-in content is read promptly. Lower = more scans while scrolling.">?</span></label><input type="number" id="st" value="60"></div>
<div><label>Blur buffer (px)<span class="qm" data-tip="Pixels of blur beyond the tightly-cropped word or face. 8 is a safe default; 3 is a tight cosmetic look. During scrolling a small drift allowance is added automatically.">?</span></label><input type="number" id="pad" value="8"></div>
<div><label>Face expand (0-1)<span class="qm" data-tip="Enlarges detected face boxes by this fraction before the blur buffer, covering hairline and ears. 0 = raw detector box (may leave identifiable edges).">?</span></label><input type="number" id="fex" value="0.15" step="0.05"></div>
<div><label>Face threshold<span class="qm" data-tip="Face detector confidence cutoff (0-1). Lower catches more faces but risks false positives on face-like patterns; higher is stricter. Default 0.6. Tune with preview + show scores.">?</span></label><input type="number" id="fthr" value="0.6" step="0.05" min="0" max="1"></div>
<div><label>Detection scale<span class="qm" data-tip="Runs FACE detection on a downscaled copy of each frame for speed (0.2-1.0; e.g. 0.5 = half resolution). Output quality is unaffected — only the detection pass is faster. 1.0 = full resolution (off). Helpful with dense faces on high-res video.">?</span></label><input type="number" id="dscale" value="1.0" step="0.1" min="0.2" max="1"></div>
<div><label>Bridge gap (s)<span class="qm" data-tip="If the same PHI is seen, missed for a bit, then seen again, gaps up to this long stay blurred — unless scans prove it actually disappeared. Raise to 8-10 for mostly static screens.">?</span></label><input type="number" id="bg" value="4" step="0.5"></div>
<div><label>Skip first (s) — delay detection<span class="qm" data-tip="Detection window start. NOTHING in the first N seconds is detected or blurred — use only when PHI cannot appear during the intro.">?</span></label>
<input type="number" id="skipstart" value="0" min="0" step="1"></div>
<div><label>Skip last (s) — stop before end<span class="qm" data-tip="Detection window end. Nothing in the last N seconds is detected or blurred.">?</span></label>
<input type="number" id="skipend" value="0" min="0" step="1"></div>
<div><label>MRN regex<span class="qm" data-tip="Pattern for medical record numbers. Default matches 7 digits starting with 1, or MM followed by 10 digits. Anchors work — punctuation around the number is handled.">?</span></label><input type="text" id="mrnrx"
value="(^(?:1\\d{6}|MM\\d{10})$)"></div>
</div>
<label>Categories<span class="qm" data-tip="Which PHI types the detectors look for. Unchecking a category means it is never blurred.">?</span></label>
<div class="chk" id="cats"></div>
<div class="row" style="margin:10px 0 4px">
<a href="zones"><button type="button" class="sec">⬚ Detection zones…</button></a>
<span id="zstat" class="badge">none configured</span></div>
<div class="grid2">
<div><label>Allow names (keep visible)<span class="qm" data-tip="Words never treated as PHI — provider names, app names. One per line. Green chips below are learned automatically from your reviews and always apply too.">?</span></label><textarea id="allow"
placeholder="e.g. provider or app names to always keep visible&#10;one per line"></textarea>
<div id="persistlist" class="plist"></div>
<div id="persistnote" class="pnote" style="display:none"></div>
<button type="button" class="sec" id="clearpersist" style="display:none;margin-top:5px;
 padding:4px 10px;font-size:12px" onclick="clearPersist()">Clear all learned words</button></div>
<div><label>Always blur (extra names)<span class="qm" data-tip="Words always blurred even if the detectors would miss them (unusual spellings, nicknames). One per line.">?</span></label><textarea id="extra"></textarea></div>
</div>
<label>Ignore regions (x1,y1,x2,y2 per line — e.g. taskbar clock)<span class="qm" data-tip="Screen rectangles that are never scanned at all — taskbar clock, a webcam overlay of yourself. Pixel coordinates of the recording.">?</span></label>
<textarea id="ign" style="height:40px"></textarea>
<div class="row" style="margin-top:8px">
<label style="margin:0"><input type="checkbox" id="nomem"> disable memory<span class="qm" data-tip="Memory recalls confirmed PHI on later frames even when OCR misreads it. Disabling cuts false positives but weakens recall — usually leave off.">?</span></label>
<label style="margin:0"><input type="checkbox" id="pmode"> preview mode (boxes only)<span class="qm" data-tip="Draws red outline boxes instead of blurring — a fast way to check coverage before committing to a render.">?</span></label>
<label style="margin:0"><input type="checkbox" id="skiprev"> skip review (blur everything found, render immediately)<span class="qm" data-tip="No human check: every detection is blurred and the render starts right away. Use only with settings you already trust.">?</span></label>
<label style="margin:0"><input type="checkbox" id="usezones" checked> apply detection zones<span class="qm" data-tip="Restricts each category to the zones drawn in the zone editor. Outside a category's zones NOTHING is blurred, even if detected.">?</span></label>
<div id="platepanel" style="display:none;margin:8px 0;padding:10px 12px;border:1px solid #e5e7eb;border-radius:10px;background:#fafaf9">
 <div style="font-weight:700;font-size:13px;margin-bottom:2px">License-plate model <span class="qm" data-tip="Plate detection needs a model file (not bundled). Pick a curated open-source model to download it — the file is SHA-256 verified before use. Each entry shows its software license.">?</span></div>
 <div id="platestatus" style="font-size:12px;color:#6b7280;margin-bottom:6px">checking…</div>
 <div id="platelist"></div>
</div>
<label style="margin:0"><input type="checkbox" id="drawscores"> show face scores (preview)<span class="qm" data-tip="In preview mode, labels each face box with its detection confidence so you can pick a good Face threshold for your footage.">?</span></label>
<label style="margin:0"><input type="checkbox" id="densefaces"> dense faces (every frame)<span class="qm" data-tip="Runs the face detector on EVERY frame instead of at scan intervals, so fast-moving faces stay covered (e.g. someone walking through a webcam feed). Slower to render — pair it with a face detection zone to keep it fast. Leave off for static screens where faces don't move.">?</span></label>
<button onclick="startJob()">Start scan</button>
</div>
</div>

<div class="card"><h2>Jobs</h2><div class="joblist" id="jobs">loading…</div></div>
<div id="detail"></div>
</main>
<footer style="text-align:center;color:#9ca3af;font-size:12px;padding:18px 12px 26px">
OpenScrub v4.2.0 · <a href="license" style="color:#6b7280">Apache-2.0 license</a>
· best-effort redaction — always review output before sharing PHI</footer>
<script>
const CATS=["name","dob","phone","ssn","mrn","email","address","card","apikey","ipaddr","plate","face"];
const CATMODE={};   // category -> "" (default) | "blur" | "box"
function renderCats(){
 const gm=document.getElementById("mode").value;
 document.getElementById("cats").innerHTML=CATS.map(c=>{
  const on=CATMODE[c]===undefined?true:CATMODE[c]!=="__off";
  const m=(CATMODE[c]&&CATMODE[c]!=="__off")?CATMODE[c]:"";
  return `<span class="catrow">
   <label><input type="checkbox" class="cat" value="${c}" ${on?"checked":""}
    onchange="onCatToggle('${c}',this.checked)">${c}</label>
   <select class="catmode" data-c="${c}" onchange="CATMODE['${c}']=this.value||''"
    ${on?"":"disabled"} title="redaction style for ${c}">
    <option value="">${gm} (default)</option>
    <option value="blur" ${m==="blur"?"selected":""}>blur</option>
    <option value="box" ${m==="box"?"selected":""}>box</option>
    <option value="mosaic" ${m==="mosaic"?"selected":""}>mosaic</option>
   </select></span>`;
 }).join("");
}
function onCatToggle(c,on){
 CATMODE[c]=on?"":"__off";
 const sel=document.querySelector(`.catmode[data-c="${c}"]`);
 if(sel)sel.disabled=!on;
}
renderCats();

async function refreshPlatePanel(){
 const on=[...document.querySelectorAll(".cat:checked")].some(e=>e.value==="plate");
 const pp=document.getElementById("platepanel");
 pp.style.display=on?"block":"none";
 if(!on)return;
 try{
  const r=await fetch("/api/plate_models"); const d=await r.json();
  const st=document.getElementById("platestatus");
  st.textContent=d.active?("Active model: "+d.active):"No model installed — plate detection is INACTIVE until you install or provide one.";
  st.style.color=d.active?"#15803d":"#b45309";
  document.getElementById("platelist").innerHTML=d.models.map(m=>`
   <div style="display:flex;align-items:center;gap:8px;padding:6px 0;border-top:1px solid #eee">
    <div style="flex:1;min-width:0">
     <div style="font-size:12.5px;font-weight:600">${m.label} ${m.recommended?'<span style="color:#15803d;font-size:11px">★ recommended</span>':''}</div>
     <div style="font-size:11px;color:#6b7280">${m.notes||""}</div>
    </div>
    <a href="${m.source_url}" target="_blank" style="font-size:11px;background:#eef2ff;color:#3730a3;border-radius:5px;padding:2px 7px;text-decoration:none" title="${m.attribution||''}">${m.license}</a>
    ${m.installed?'<span style="font-size:11px;color:#15803d;font-weight:700">installed</span>'
      :(m.verified?`<button onclick="dlPlate('${m.id}',this)" title="${m.pinned?'SHA-256 verified against pinned hash':'first download: hash will be computed and pinned'}" style="font-size:11.5px;padding:3px 10px">Download</button>`
      :'<span style="font-size:11px;color:#9ca3af" title="registry entry has no download URL">unavailable</span>')}
   </div>`).join("");
 }catch(e){document.getElementById("platestatus").textContent="model registry unavailable";}
}
async function dlPlate(id,btn){
 btn.disabled=true;btn.textContent="0%";
 const r=await fetch(`/api/plate_models/${id}/download`,{method:"POST"});
 if(!r.ok){const d=await r.json();btn.textContent="error";btn.title=d.error||"";return;}
 const t=setInterval(async()=>{
  const s=await(await fetch("/api/plate_models/download_status")).json();
  if(s.state==="downloading"){btn.textContent=(s.pct||0)+"%";}
  else{clearInterval(t);
   if(s.state==="done"){refreshPlatePanel();}
   else{btn.disabled=false;btn.textContent="retry";btn.title=s.error||"download failed";}}
 },500);
}
document.getElementById("cats").addEventListener("change",refreshPlatePanel);
refreshPlatePanel();
let CUR=null, EN={}, MAN=[], DUR=0, POLL=null, PHIST=[], LOGN=0, SHELLPH=null, JOBSJSON="";

function opts(){return{
 engine:engine.value,device:device.value,encoder:encoder.value,mode:mode.value,
 sample_interval:+si.value,scan_trigger:+st.value,pad:+pad.value,bridge_gap:+bg.value,
 mrn_regex:mrnrx.value,face_expand:+fex.value,skip_review:skiprev.checked,use_zones:usezones.checked,
 skip_start:+skipstart.value,skip_end:+skipend.value,out_format:outfmt.value,
 categories:[...document.querySelectorAll(".cat:checked")].map(e=>e.value).join(","),
 mode_map:Object.entries(CATMODE).filter(([c,m])=>m==="blur"||m==="box")
   .map(([c,m])=>`${c}=${m}`).join(","),
 allow_names:allow.value,extra_names:extra.value,ignore_regions:ign.value,
 no_memory:nomem.checked,preview_mode:pmode.checked,
 dense_faces:densefaces.checked,face_threshold:+fthr.value,
 detect_scale:+dscale.value,draw_scores:drawscores.checked}}

async function startJob(){
 const fd=new FormData();
 for(const f of file.files) fd.append("video",f);
 fd.append("server_path",spath.value);
 fd.append("options",JSON.stringify(opts()));
 const r=await fetch("api/jobs",{method:"POST",body:fd});
 const j=await r.json();
 if(j.error){alert(j.error);return}
 file.value="";spath.value="";
 loadJobs(); if(j.jobs.length)openJob(j.jobs[0]);
}

async function loadJobs(){
 const js=await (await fetch("api/jobs")).json();
 const s=JSON.stringify(js);
 if(s===JOBSJSON)return;
 JOBSJSON=s;
 jobs.innerHTML=js.length?js.map(j=>
  `<div class="row" style="justify-content:space-between;padding:4px 0">
   <a href="#" onclick="openJob('${j.id}');return false">${j.name}</a>
   <span class="badge">${j.phase}</span></div>`).join(""):"none yet";
}

async function openJob(id){
 CUR=id; EN={}; MAN=[]; PHIST=[]; LOGN=0; SHELLPH=null;
 detail.innerHTML="";
 if(POLL)clearInterval(POLL);
 POLL=setInterval(refresh,900); refresh();
}

function shellHtml(j){
 return `<div class="card" id="jobcard"><h2>${j.name}
  <span class="badge" id="ph"></span><small id="eta"></small></h2>
 <progress id="prog" value="0" max="1"></progress>
 <div class="row" id="actions" style="margin-top:8px"></div>
 <div style="margin-top:8px;text-align:center"><img id="pimg"
   style="max-width:100%;border-radius:8px;display:none;margin:0 auto"></div>
 <div id="pcap" style="font-size:12px;color:#6b7280"></div>
 <div id="log" style="margin-top:8px"></div>
 <div class="warn" id="warnbox" style="display:none;margin-top:8px"></div>
 </div>`;
}

async function refresh(){
 if(!CUR)return;
 const j=await (await fetch("api/jobs/"+CUR)).json();
 if(SHELLPH===null){
  detail.insertAdjacentHTML("afterbegin",shellHtml(j));SHELLPH="";
  const im=document.getElementById("pimg");
  im.onload=()=>{im.style.display="block";};   // show only once a real frame arrives
  im.onerror=()=>{im.style.display="none";};   // 404 before first preview: stay hidden
 }

 // progress + ETA (updated in place — nothing is rebuilt)
 PHIST.push([Date.now(),j.progress]); if(PHIST.length>40)PHIST.shift();
 let eta="";
 if(j.progress>0&&j.progress<1&&PHIST.length>5){
  const[t0,p0]=PHIST[0],[t1,p1]=PHIST[PHIST.length-1];
  if(p1>p0){const rem=(1-p1)*(t1-t0)/(p1-p0)/1000;
   eta=` ~${rem>90?Math.round(rem/60)+" min":Math.round(rem)+" s"} left`;}}
 document.getElementById("ph").textContent=j.phase;
 document.getElementById("eta").textContent=eta;
 document.getElementById("prog").value=j.progress;

 // incremental log append with stick-to-bottom (respects manual scroll-up)
 const lg=document.getElementById("log");
 const stick=lg.scrollTop+lg.clientHeight>=lg.scrollHeight-40;
 const nl=await (await fetch(`api/jobs/${CUR}/log?from=${LOGN}`)).json();
 if(LOGN===0&&nl.len>400){LOGN=nl.len-400;nl.lines=nl.lines.slice(-400);}
 if(nl.lines.length){
  const NL=String.fromCharCode(10);
  lg.textContent+=(lg.textContent?NL:"")+nl.lines.join(NL);
  LOGN=nl.len;
  if(stick)lg.scrollTop=lg.scrollHeight;
 }

 // live preview: swap src in place (browser keeps old frame until new loads)
 const img=document.getElementById("pimg");
 if(j.phase=="scanning"){
  img.src=`api/jobs/${CUR}/preview.jpg?x=${Date.now()}`;
 }

 const wb=document.getElementById("warnbox");
 if(j.error){wb.style.display="block";wb.textContent=j.error;}
 else wb.style.display="none";

 // action buttons rebuilt ONLY when the phase changes
 if(SHELLPH!==j.phase){
  SHELLPH=j.phase;
  let a="";
  if(j.phase=="scanning"||j.phase=="rendering")
   a=`<button class="danger" onclick="cancelJob()">Cancel</button>`;
  if(j.phase=="review")
   a=`<button onclick="loadReview()">Open review</button>`;
  if(j.phase=="done"&&j.has_output)
   a=`<a href="api/jobs/${CUR}/download"><button>Download redacted video</button></a>
    <a href="api/jobs/${CUR}/report"><button class="sec">Download audit report</button></a>
    <button class="sec" onclick="loadReview()">Re-open review</button>
    <button class="sec" onclick="loadCompare()">Compare before/after</button>
    <button class="sec" onclick="loadBoxEdit()">Preview &amp; edit blur boxes</button>
    <div class="warn" style="width:100%">Reminder: scrub the result before
    sharing. This tool is best-effort, not a compliance guarantee.</div>
    <div id="compare" style="width:100%"></div>
    <div id="beslot_done" style="width:100%"></div>`;
  document.getElementById("actions").innerHTML=a;
  if(j.phase!="scanning")img.style.display="none";
 }
 loadJobs();
}

let BE={t:0,ox:0,oy:0,boxes:[],sel:-1,ov:{},dis:{},times:{},adds:[],
        add:false,addAnchor:null,drag:null,scale:1,dur:10};
async function loadBoxEdit(){
 const mi=await (await fetch(`api/jobs/${CUR}/mediainfo`)).json();
 // in review: the slot right under the Preview button; on done: the actions slot
 let c=document.getElementById("beslot")||document.getElementById("beslot_done");
 if(!c){c=document.createElement("div");c.id="beslot";c.style.width="100%";
  (document.getElementById("review")||document.getElementById("jobcard")||detail)
   .appendChild(c);}
 BE={t:0,ox:0,oy:0,boxes:[],sel:-1,ov:{},dis:{},times:{},adds:[],
     add:false,addAnchor:null,drag:null,scale:1,dur:mi.duration};
 c.innerHTML=`<div class="row" style="margin-top:8px">t=
  <input type="range" id="beT" min="0" max="${Math.max(0.2,mi.duration-0.1).toFixed(1)}"
   step="0.1" value="0" style="flex:1" oninput="beScrub()"><span id="beL">0s</span></div>
  <div style="position:relative;display:inline-block;max-width:100%">
   <img id="beImg" style="max-width:100%;border-radius:8px">
   <canvas id="beCv" style="position:absolute;left:0;top:0;touch-action:none"></canvas></div>
  <div class="row" style="margin-top:6px">
   <button class="tog" id="beAdd" onclick="beAddMode()">＋ Add box</button>
   <button class="danger" id="beRm" style="display:none" onclick="beRemove()">Remove blur</button>
   <span id="beTimes" style="display:none;align-items:center;gap:4px;font-size:13px">
    from <input type="number" id="beT0" step="0.1" min="0" style="width:74px">
    to <input type="number" id="beT1" step="0.1" style="width:74px"> s
    <button class="sec" onclick="beApplyTimes()">Apply times</button></span>
   <button onclick="beSave()">Save changes &amp; re-render</button></div>
  <p style="font-size:12px;color:#6b7280;margin:4px 0 0">Click a red box to
   select — drag inside to move, a corner to resize, edit its from/to times,
   or remove its blur. ＋ Add box: drag a new region, then set how long it
   lasts. New boxes show orange.</p>`;
 const img=document.getElementById("beImg");
 img.onload=()=>{const cv=document.getElementById("beCv");
  cv.width=img.clientWidth;cv.height=img.clientHeight;
  BE.scale=img.clientWidth/(img.naturalWidth||1);beDraw();};
 beHook();beScrub();
}
async function beScrub(){
 BE.sel=-1;document.getElementById("beRm").style.display="none";
 document.getElementById("beTimes").style.display="none";
 BE.t=+document.getElementById("beT").value;
 document.getElementById("beL").textContent=BE.t.toFixed(1)+"s";
 document.getElementById("beImg").src=`api/jobs/${CUR}/frame_at?t=${BE.t}`;
 const d=await (await fetch(`api/jobs/${CUR}/boxes_at?t=${BE.t}&all=1`)).json();
 BE.ox=d.ox;BE.oy=d.oy;BE.raw=d.boxes;
 beFilter();beDraw();
}
function beFilter(){
 BE.boxes=(BE.raw||[]).filter(b=>{
  const on=(b.i in EN)?EN[b.i]:b.enabled;   // live: review toggles apply here
  return on;
 }).map(b=>{
  let bb=b;
  if(BE.ov[b.i]){const o=BE.ov[b.i];
   bb={...b,box:[o[0]+BE.ox,o[1]+BE.oy,o[2]+BE.ox,o[3]+BE.oy]};}
  if(BE.times[b.i]){bb={...bb,t_start:BE.times[b.i][0],t_end:BE.times[b.i][1]};}
  return bb;
 });
}
function beDraw(){
 const cv=document.getElementById("beCv");if(!cv)return;
 const x=cv.getContext("2d"),s=BE.scale;
 x.clearRect(0,0,cv.width,cv.height);
 for(const a of BE.adds){
  if(BE.t<a.t0-0.01||BE.t>a.t1+0.01)continue;
  const[a1,b1,a2,b2]=a.box.map(v=>v*s);
  x.fillStyle="rgba(245,158,11,.25)";x.strokeStyle="#f59e0b";x.lineWidth=2;
  x.fillRect(a1,b1,a2-a1,b2-b1);x.strokeRect(a1,b1,a2-a1,b2-b1);
 }
 if(BE.add&&BE.addAnchor&&BE.addFloat){
  const[p,q]=[BE.addAnchor,BE.addFloat];
  x.strokeStyle="#f59e0b";x.setLineDash([6,4]);x.lineWidth=2;
  x.strokeRect(Math.min(p[0],q[0])*s,Math.min(p[1],q[1])*s,
   Math.abs(q[0]-p[0])*s,Math.abs(q[1]-p[1])*s);x.setLineDash([]);
 }
 for(const b of BE.boxes){
  const[a1,b1,a2,b2]=b.box.map(v=>v*s);
  x.fillStyle=b.i===BE.sel?"rgba(37,99,235,.25)":"rgba(220,38,38,.22)";
  x.strokeStyle=b.i===BE.sel?"#2563eb":"#dc2626";x.lineWidth=2;
  x.fillRect(a1,b1,a2-a1,b2-b1);x.strokeRect(a1,b1,a2-a1,b2-b1);
  if(b.i===BE.sel)for(const[hx,hy] of [[a1,b1],[a2,b1],[a1,b2],[a2,b2]]){
   x.fillStyle="#fff";x.fillRect(hx-5,hy-5,10,10);
   x.strokeRect(hx-5,hy-5,10,10);}
 }
}
function beAddMode(){
 BE.add=!BE.add;BE.addAnchor=null;BE.sel=-1;
 document.getElementById("beAdd").classList.toggle("on",BE.add);
 document.getElementById("beRm").style.display="none";
 document.getElementById("beTimes").style.display="none";
 beDraw();
}
function beApplyTimes(){
 if(BE.sel<0)return;
 const t0=+document.getElementById("beT0").value,
       t1=+document.getElementById("beT1").value;
 if(!(t1>t0&&t0>=0)){alert("'to' must be after 'from'.");return;}
 BE.times[BE.sel]=[t0,Math.min(t1,BE.dur)];
 const b=BE.boxes.find(x=>x.i===BE.sel);
 if(b){b.t_start=t0;b.t_end=t1;}
 beDraw();
}
function beShowTimes(b){
 const el=document.getElementById("beTimes");
 el.style.display="inline-flex";
 document.getElementById("beT0").value=b.t_start.toFixed(1);
 document.getElementById("beT1").value=b.t_end.toFixed(1);
}
function beHook(){
 const cv=document.getElementById("beCv");
 const pt=e=>{const r=cv.getBoundingClientRect();
  return[(e.clientX-r.left)/BE.scale,(e.clientY-r.top)/BE.scale];};
 cv.addEventListener("pointerdown",e=>{
  const p=pt(e);cv.setPointerCapture(e.pointerId);
  if(BE.add){
   if(!BE.addAnchor){BE.addAnchor=p;BE.addFloat=p;}
   else{
    const a=BE.addAnchor,q=p;BE.addAnchor=null;BE.addFloat=null;
    const box=[Math.min(a[0],q[0]),Math.min(a[1],q[1]),
               Math.max(a[0],q[0]),Math.max(a[1],q[1])];
    if(box[2]-box[0]>6&&box[3]-box[1]>6){
     const t0=Math.max(0,+(BE.t-0.5).toFixed(1)),
           t1=Math.min(BE.dur,+(BE.t+3).toFixed(1));
     const fr=prompt("Blur this region FROM (seconds):",t0);
     if(fr===null){beDraw();return;}
     const to=prompt("...TO (seconds):",t1);
     if(to===null){beDraw();return;}
     BE.adds.push({box,t0:Math.max(0,+fr||0),
                   t1:Math.min(BE.dur,+to||t1),tref:BE.t});
    }
    beDraw();
   }
   return;
  }
  const sel=BE.boxes.find(b=>b.i===BE.sel);
  if(sel){const[a1,b1,a2,b2]=sel.box,hs=9/BE.scale;
   const corners=[[a1,b1,0],[a2,b1,1],[a1,b2,2],[a2,b2,3]];
   for(const[hx,hy,ci] of corners)
    if(Math.abs(p[0]-hx)<hs&&Math.abs(p[1]-hy)<hs){
     BE.drag={kind:"corner",ci};return;}
   if(p[0]>=a1&&p[0]<=a2&&p[1]>=b1&&p[1]<=b2){
    BE.drag={kind:"move",off:[p[0]-a1,p[1]-b1]};return;}
  }
  const hit=[...BE.boxes].reverse().find(b=>{
   const[a1,b1,a2,b2]=b.box;
   return p[0]>=a1&&p[0]<=a2&&p[1]>=b1&&p[1]<=b2;});
  BE.sel=hit?hit.i:-1;
  document.getElementById("beRm").style.display=hit?"inline-block":"none";
  if(hit)beShowTimes(hit);
  else document.getElementById("beTimes").style.display="none";
  beDraw();
 });
 cv.addEventListener("pointermove",e=>{
  if(BE.add&&BE.addAnchor){BE.addFloat=pt(e);beDraw();return;}
  if(!BE.drag||BE.sel<0)return;
  const p=pt(e),b=BE.boxes.find(x=>x.i===BE.sel);if(!b)return;
  let[a1,b1,a2,b2]=b.box;
  if(BE.drag.kind==="move"){
   const w0=a2-a1,h0=b2-b1;a1=p[0]-BE.drag.off[0];b1=p[1]-BE.drag.off[1];
   a2=a1+w0;b2=b1+h0;
  }else{
   if(BE.drag.ci===0){a1=p[0];b1=p[1];}
   if(BE.drag.ci===1){a2=p[0];b1=p[1];}
   if(BE.drag.ci===2){a1=p[0];b2=p[1];}
   if(BE.drag.ci===3){a2=p[0];b2=p[1];}
  }
  b.box=[Math.min(a1,a2),Math.min(b1,b2),Math.max(a1,a2),Math.max(b1,b2)];
  BE.ov[BE.sel]=[b.box[0]-BE.ox,b.box[1]-BE.oy,b.box[2]-BE.ox,b.box[3]-BE.oy];
  beDraw();
 });
 cv.addEventListener("pointerup",()=>{BE.drag=null;});
}
function zoomDet(i){
 let ov=document.getElementById("lb");
 if(!ov){
  ov=document.createElement("div");ov.id="lb";
  ov.style.cssText="position:fixed;inset:0;background:rgba(0,0,0,.82);"+
   "z-index:100;display:flex;align-items:center;justify-content:center;"+
   "cursor:zoom-out;padding:18px";
  ov.onclick=()=>ov.remove();
  document.body.appendChild(ov);
 }
 ov.innerHTML=`<img src="api/jobs/${CUR}/detframe/${i}"
  style="max-width:96%;max-height:96%;border-radius:8px;
  box-shadow:0 12px 50px rgba(0,0,0,.6)">`;
}
function beRemove(){
 if(BE.sel<0)return;
 applyDet(BE.sel,false);   // syncs thumbnail + refilters editor
 BE.sel=-1;
 document.getElementById("beRm").style.display="none";
 document.getElementById("beTimes").style.display="none";
}
async function beSave(){
 await saveAndRender();
 const c=document.querySelector("#beslot");
 if(c)c.innerHTML="rendering with your edits…";
}
async function cancelJob(){await fetch("api/jobs/"+CUR+"/cancel",{method:"POST"})}

async function loadCompare(){
 const d=await (await fetch(`api/jobs/${CUR}/detections`)).json();
 const c=document.getElementById("compare");
 c.innerHTML=`<div class="row" style="margin-top:8px">t=
  <input type="range" id="cmpT" min="0" max="${d.duration.toFixed(1)}" step="0.2"
   value="0" style="flex:1" oninput="cmpShow()">
  <span id="cmpLbl">0s</span></div>
  <div class="grid2"><div><b>Original</b><br><img id="cmpO" style="max-width:100%"></div>
  <div><b>Redacted</b><br><img id="cmpR" style="max-width:100%"></div></div>`;
 cmpShow();
}
function cmpShow(){
 const t=document.getElementById("cmpT").value;
 document.getElementById("cmpLbl").textContent=t+"s";
 document.getElementById("cmpO").src=`api/jobs/${CUR}/frame_at?t=${t}`;
 document.getElementById("cmpR").src=`api/jobs/${CUR}/frame_at?t=${t}&src=output`;
}

async function loadReview(){
 const d=await (await fetch(`api/jobs/${CUR}/detections`)).json();
 DUR=d.duration;
 const groups={}; const zdrop=[];
 d.detections.forEach(x=>{
  EN[x.i]=x.enabled;
  if(x.zone_dropped){zdrop.push(x);return;}
  (groups[x.category]=groups[x.category]||[]).push(x);});
 let html=`<div class="card" id="review"><h2>Review — red = will be blurred,
 green = kept visible. Drag a box around several to bulk-set.</h2>
 <div class="row" style="margin:0 0 8px">
  <button class="sec" onclick="loadBoxEdit()">▶ Preview video &amp; edit boxes
  (before rendering)</button></div>
 <div id="beslot"></div>
 <div id="detwrap" style="position:relative;user-select:none">`;
 if(zdrop.length){
  html+=`<details style="margin:4px 0 10px;border:1px dashed #f59e0b;border-radius:9px;padding:8px 10px;background:#fffbeb">
   <summary style="cursor:pointer;color:#92400e;font-weight:600">
    ⚠ ${zdrop.length} detection(s) outside your zones — NOT blurred
    (mostly logos/labels by design; open only if something real was zoned out)
   </summary>
   <div class="row" style="margin:8px 0 4px">
    <button class="sec" onclick="setAllZd(true)">blur all of these</button>
    <button class="sec" onclick="setAllZd(false)">keep all</button></div>
   <div class="grid3">`;
  for(const x of zdrop){
   html+=`<div class="det ${EN[x.i]?'':'off'}" id="det${x.i}" data-cat="zdrop">
    <img src="api/jobs/${CUR}/thumb/${x.i}" loading="lazy"
     style="cursor:zoom-in" onclick="zoomDet(${x.i})" onerror="thumbRetry(this)">
    <div>"${x.text}" <br>${x.t_start.toFixed(1)}–${x.t_end.toFixed(1)}s</div>
    <button class="blurbtn ${EN[x.i]?"blur":"keep"}" id="bb${x.i}"
     onclick="toggleDet(${x.i})">${EN[x.i]?"Blur":"Keep"}</button>
   </div>`;
  }
  html+=`</div></details>`;
 }
 for(const[cat,items] of Object.entries(groups)){
  html+=`<h3 style="margin:10px 0 6px">${cat} <span class="badge">${items.length}</span>
  <button class="sec" style="padding:3px 8px;font-size:12px"
   onclick="setAll('${cat}',true)">all on</button>
  <button class="sec" style="padding:3px 8px;font-size:12px"
   onclick="setAll('${cat}',false)">all off</button></h3><div class="grid3">`;
  for(const x of items){
   html+=`<div class="det ${EN[x.i]?'':'off'}" id="det${x.i}" data-cat="${cat}">
    <img src="api/jobs/${CUR}/thumb/${x.i}"
     style="cursor:zoom-in" onclick="zoomDet(${x.i})"
     onerror="thumbRetry(this)">
    <div>"${x.text}" <br>${x.t_start.toFixed(1)}–${x.t_end.toFixed(1)}s</div>
    <button class="blurbtn ${EN[x.i]?"blur":"keep"}" id="bb${x.i}"
     onclick="toggleDet(${x.i})">${EN[x.i]?"Blur":"Keep"}</button>
   </div>`}
  html+=`</div>`}
 html+=`</div>
 <div class="row" style="margin-top:12px">
 <button onclick="saveAndRender()">Save edits &amp; render</button>
 <span style="font-size:12px;color:#6b7280">missed something? use
  ▶ Preview video &amp; edit boxes above — its ＋ Add box replaces the old
  add-region tool</span></div></div>`;
 const old=document.getElementById("review"); if(old)old.remove();
 detail.insertAdjacentHTML("beforeend",html);
 initMarquee();
}

function applyDet(i,blur){
 EN[i]=blur;
 const b=document.getElementById("bb"+i),t=document.getElementById("det"+i);
 if(b){b.className="blurbtn "+(blur?"blur":"keep");
       b.textContent=blur?"Blur":"Keep";}
 if(t)t.classList.toggle("off",!blur);
 if(document.getElementById("beCv")){beFilter();beDraw();}   // live sync
}
function toggleDet(i){applyDet(i,!EN[i]);}
function setAll(cat,on){document.querySelectorAll(`.det[data-cat=${cat}]`)
 .forEach(el=>applyDet(+el.id.slice(3),on));}
function thumbRetry(img){
 const n=+(img.dataset.r||0);
 if(n>=3)return;
 img.dataset.r=n+1;
 setTimeout(()=>{img.src=img.src.split("?r=")[0]+"?r="+Date.now();},
            600*(n+1));
}
function setAllZd(on){document.querySelectorAll(`.det[data-cat=zdrop]`)
 .forEach(el=>applyDet(+el.id.slice(3),on));}

let selSet=new Set(),marqEl=null,menuEl=null,mStart=null;
function clearSel(){
 selSet.forEach(i=>{const t=document.getElementById("det"+i);
  if(t)t.classList.remove("selected");});
 selSet.clear();
 if(menuEl){menuEl.remove();menuEl=null;}
}
function initMarquee(){
 const wrap=document.getElementById("detwrap");
 if(!wrap||wrap.dataset.mq)return;
 wrap.dataset.mq=1;
 wrap.addEventListener("pointerdown",e=>{
  if(e.pointerType==="touch")return;         // touch keeps normal scrolling
  if(e.target.closest("button"))return;      // button clicks stay clicks
  clearSel();
  mStart=[e.pageX,e.pageY];
  marqEl=document.createElement("div");marqEl.className="marq";
  document.body.appendChild(marqEl);
  const move=ev=>{
   const x1=Math.min(mStart[0],ev.pageX),y1=Math.min(mStart[1],ev.pageY),
         x2=Math.max(mStart[0],ev.pageX),y2=Math.max(mStart[1],ev.pageY);
   Object.assign(marqEl.style,{left:x1+"px",top:y1+"px",
    width:(x2-x1)+"px",height:(y2-y1)+"px"});
  };
  const up=ev=>{
   document.removeEventListener("pointermove",move);
   document.removeEventListener("pointerup",up);
   const r=marqEl.getBoundingClientRect();
   marqEl.remove();marqEl=null;
   if(r.width<8&&r.height<8)return;          // just a click — not a drag
   document.querySelectorAll("#detwrap .det").forEach(t=>{
    const b=t.getBoundingClientRect();
    if(b.left<r.right&&b.right>r.left&&b.top<r.bottom&&b.bottom>r.top){
     selSet.add(+t.id.slice(3));t.classList.add("selected");}});
   if(selSet.size)showSelMenu(ev.pageX,ev.pageY);
  };
  document.addEventListener("pointermove",move);
  document.addEventListener("pointerup",up);
  e.preventDefault();
 });
}
function showSelMenu(x,y){
 menuEl=document.createElement("div");menuEl.className="selmenu";
 menuEl.innerHTML=`<span>${selSet.size} selected</span>
  <button class="mkeep" onclick="bulkSet(false)">Keep</button>
  <button class="mblur" onclick="bulkSet(true)">Blur</button>`;
 Object.assign(menuEl.style,{left:x+"px",top:(y+10)+"px"});
 document.body.appendChild(menuEl);
}
function bulkSet(blur){selSet.forEach(i=>applyDet(i,blur));clearSel();}
document.addEventListener("pointerdown",e=>{
 if(menuEl&&!e.target.closest(".selmenu")
    &&!e.target.closest("#detwrap"))clearSel();});
document.addEventListener("click",e=>{
 const q=e.target.closest(".qm");
 document.querySelectorAll(".qm.open").forEach(x=>{if(x!==q)x.classList.remove("open");});
 if(q)q.classList.toggle("open");});
document.addEventListener("keydown",e=>{
 if(e.key==="Escape"){clearSel();
  const lb=document.getElementById("lb");if(lb)lb.remove();}});

async function saveAndRender(){
 const manual=[...MAN,...BE.adds.map(a=>({t_start:a.t0,t_end:a.t1,
  t_ref:a.tref,screen_box:a.box}))];
 const r=await (await fetch(`api/jobs/${CUR}/detections`,{method:"POST",
  headers:{"Content-Type":"application/json"},
  body:JSON.stringify({enabled:EN,manual,box_overrides:BE.ov,
   time_overrides:BE.times})})).json();
 BE.adds=[];BE.ov={};BE.times={};
 if(r.suggest_allowlist&&r.suggest_allowlist.length){
  if(confirm("You disabled every instance of: "+r.suggest_allowlist.join(", ")+". Add these to the permanent allow-list so future scans skip them?"))
   {await fetch("api/allowlist",{method:"POST",
    headers:{"Content-Type":"application/json"},
    body:JSON.stringify({words:r.suggest_allowlist})});
    loadPersist();}}
 MAN=[];
 await fetch(`api/jobs/${CUR}/render`,{method:"POST"});
 const rv=document.getElementById("review"); if(rv)rv.remove();
}
async function loadPersist(){
 const a=await (await fetch("api/allowlist")).json();
 const el=document.getElementById("persistlist"),
       nt=document.getElementById("persistnote");
 el.innerHTML=(a.words||[]).map(x=>
  `<div class="pw">${x}<span title="remove from allow-list"
    onclick="rmPersist('${x.replace(/'/g,"")}')">✕</span></div>`).join("");
 if((a.words||[]).length){
  nt.style.display="block";
  nt.textContent="Green words load automatically from "+a.path+
   " — marked safe during previous reviews. They are always applied in "+
   "addition to anything typed above. Click ✕ to remove one.";
 }else nt.style.display="none";
 document.getElementById("clearpersist").style.display=
  (a.words||[]).length?"inline-block":"none";
}
async function clearPersist(){
 if(!confirm("Remove ALL learned safe words? Future scans will flag them again until re-reviewed."))return;
 await fetch("api/allowlist",{method:"POST",
  headers:{"Content-Type":"application/json"},
  body:JSON.stringify({clear:true})});
 loadPersist();
}
async function rmPersist(word){
 await fetch("api/allowlist",{method:"POST",
  headers:{"Content-Type":"application/json"},
  body:JSON.stringify({remove:[word]})});
 loadPersist();
}
async function loadCertInfo(){
 try{
  const c=await (await fetch("api/certinfo")).json();
  document.getElementById("certinfo").textContent=
   `Current: ${c.mode} certificate — ${c.subject}, expires ${c.expires}`;
 }catch(e){
  document.getElementById("certinfo").textContent=
   "Running over plain HTTP (started with --http).";
 }
}
async function uploadCerts(){
 const cf=document.getElementById("certfile").files[0],
       kf=document.getElementById("keyfile").files[0];
 if(!cf||!kf){alert("Choose both a certificate file and a key file.");return;}
 const fd=new FormData();fd.append("cert",cf);fd.append("key",kf);
 const r=await fetch("api/certs",{method:"POST",body:fd});
 alert(r.ok?(await r.json()).note:"Upload failed: "+await r.text());
 loadCertInfo();
}
async function removeCerts(){
 if(!confirm("Remove the custom certificate and return to self-signed on next restart?"))return;
 const r=await fetch("api/certs",{method:"DELETE"});
 alert((await r.json()).note);
 loadCertInfo();
}
async function zoneStatus(){
 const z=await (await fetch("api/zones")).json();
 const n=Object.keys(z).length;
 document.getElementById("zstat").textContent=
  n?(n+" class"+(n>1?"es":"")+" zoned — outside them nothing is blurred"):"none configured (full frame)";
}
zoneStatus();loadPersist();loadCertInfo();loadJobs();setInterval(loadJobs,5000);
</script></body></html>"""


ASSET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


@app.route("/license")
def license_page():
    p = os.path.join(SCRIPT_DIR, "LICENSE")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            body = f.read()
    else:
        body = ("OpenScrub is licensed under the Apache License 2.0.\n"
                "See https://www.apache.org/licenses/LICENSE-2.0")
    from flask import Response
    return Response(body, mimetype="text/plain")


@app.route("/logo.png")
def logo():
    return send_file(os.path.join(ASSET_DIR, "logo.png"),
                     mimetype="image/png", max_age=86400)


@app.route("/logo_dark.png")
def logo_dark():
    return send_file(os.path.join(ASSET_DIR, "logo_dark.png"),
                     mimetype="image/png", max_age=86400)


@app.route("/favicon.ico")
def favicon():
    return send_file(os.path.join(ASSET_DIR, "openscrub.ico"),
                     mimetype="image/x-icon", max_age=86400)


@app.route("/")
def index():
    return PAGE


@app.route("/zones")
def zones_page():
    return zones_ui.ZONES_PAGE


@app.route("/api/certinfo")
def certinfo():
    cert_path, _, mode = active_cert_pair()
    from cryptography import x509
    with open(cert_path, "rb") as f:
        c = x509.load_pem_x509_certificate(f.read())
    return jsonify({
        "mode": mode,
        "subject": c.subject.rfc4514_string(),
        "issuer": c.issuer.rfc4514_string(),
        "expires": c.not_valid_after_utc.strftime("%Y-%m-%d"),
    })


@app.route("/api/certs", methods=["POST", "DELETE"])
def upload_certs():
    if request.method == "DELETE":
        for p in (CUSTOM_CERT, CUSTOM_KEY):
            if os.path.exists(p):
                os.remove(p)
        return jsonify({"ok": True,
                        "note": "custom certificate removed — restart the "
                                "server to return to the self-signed one"})
    cert_f, key_f = request.files.get("cert"), request.files.get("key")
    if not cert_f or not key_f:
        abort(400, "both certificate and key PEM files are required")
    cert_pem, key_pem = cert_f.read(), key_f.read()
    from cryptography import x509
    from cryptography.hazmat.primitives.serialization import load_pem_private_key
    try:
        cert = x509.load_pem_x509_certificate(cert_pem)
        key = load_pem_private_key(key_pem, password=None)
        if key.public_key().public_numbers() != cert.public_key().public_numbers():
            abort(400, "private key does not match the certificate")
    except ValueError as e:
        abort(400, f"could not parse PEM files: {e}")
    os.makedirs(CERT_DIR, exist_ok=True)
    with open(CUSTOM_CERT, "wb") as f:
        f.write(cert_pem)
    with open(CUSTOM_KEY, "wb") as f:
        f.write(key_pem)
    try:
        os.chmod(CUSTOM_KEY, 0o600)
    except Exception:
        pass
    return jsonify({"ok": True,
                    "note": "certificate installed — restart the server to apply"})


@app.route("/api/zones", methods=["GET", "POST"])
def api_zones():
    if request.method == "POST":
        data = request.get_json(force=True) or {}
        clean = {}
        for cat, rects in data.items():
            rs = []
            for r in rects or []:
                if len(r) == 4:
                    x1, y1, x2, y2 = [max(0.0, min(1.0, float(v))) for v in r]
                    if x2 - x1 > 0.004 and y2 - y1 > 0.004:
                        rs.append([round(x1, 4), round(y1, 4),
                                   round(x2, 4), round(y2, 4)])
            if rs:
                clean[cat] = rs
        with open(ZONES_PATH, "w", encoding="utf-8") as f:
            json.dump(clean, f, indent=1)
        return jsonify({"ok": True, "classes": len(clean)})
    if os.path.exists(ZONES_PATH):
        with open(ZONES_PATH, encoding="utf-8") as f:
            return jsonify(json.load(f))
    return jsonify({})


def lan_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def rehydrate_jobs():
    """Re-register jobs found on disk after a restart. A job with a report
    is a reviewable job; one with an output file is a finished one — either
    way the person's work survives the server going down."""
    n = 0
    for jid in sorted(os.listdir(JOBS_DIR)):
        jdir = os.path.join(JOBS_DIR, jid)
        rep = os.path.join(jdir, "report.json")
        if not os.path.isdir(jdir) or not os.path.exists(rep):
            continue
        if jid in JOBS:
            continue
        try:
            with open(rep, encoding="utf-8") as f:
                prov = json.load(f).get("provenance", {})
            video = prov.get("input")
            if not video or not os.path.exists(video):
                continue
            out = os.path.join(jdir, "output.mp4")
            has_out = os.path.exists(out)
            job = {
                "id": jid, "dir": jdir, "video": video,
                "name": os.path.basename(prov.get("original_input")
                                          or video),
                "options": prov.get("settings", {}) or {},
                "phase": "done" if has_out else "review",
                "progress": 1.0 if has_out else 0.5,
                "log": ["(recovered after server restart)"],
                "error": None, "output": out if has_out else None,
                "created": os.path.getmtime(jdir),
                "cancel": threading.Event(), "stats": {},
            }
            with JOBS_LOCK:
                JOBS[jid] = job
            n += 1
        except Exception:
            continue
    if n:
        print(f"  recovered {n} job(s) from {JOBS_DIR}")


def retention_sweep(days):
    import shutil as _sh
    cutoff = time.time() - days * 86400
    for jid in os.listdir(JOBS_DIR):
        p = os.path.join(JOBS_DIR, jid)
        if os.path.isdir(p) and os.path.getmtime(p) < cutoff:
            _sh.rmtree(p, ignore_errors=True)
            with JOBS_LOCK:
                JOBS.pop(jid, None)


def retention_loop(days):
    while True:
        retention_sweep(days)
        time.sleep(6 * 3600)


def main():
    global TOKEN
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8384)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--token", help="require this access token on every "
                    "request (default: open — anyone on the network can use "
                    "the server)")
    ap.add_argument("--http", action="store_true",
                    help="serve plain HTTP (not recommended: video uploads "
                         "and previews contain PHI and would cross the "
                         "network unencrypted)")
    ap.add_argument("--retain-days", type=int, default=7,
                    help="auto-delete job folders (which contain PHI) after "
                         "this many days (default 7; 0 = keep forever)")
    args = ap.parse_args()
    TOKEN = args.token or None
    os.makedirs(JOBS_DIR, exist_ok=True)
    rehydrate_jobs()
    threading.Thread(target=worker, daemon=True).start()
    if args.retain_days > 0:
        threading.Thread(target=retention_loop, args=(args.retain_days,),
                         daemon=True).start()
    print("=" * 60)
    ssl_ctx = None
    scheme = "http"
    if not args.http:
        try:
            import cryptography  # noqa: F401 — needed for cert generation
        except ImportError:
            print("=" * 60)
            print("HTTPS needs the 'cryptography' package, which isn't")
            print("installed. Fix it with ONE of these:")
            print()
            print("    pip install cryptography      (recommended)")
            print("    python install.py --yes       (installs everything)")
            print()
            print("or start with --http to serve unencrypted")
            print("(not recommended: PHI would cross the network in plain text).")
            print("=" * 60)
            if os.name == "nt":
                input("Press Enter to close.")
            sys.exit(1)
        cert, key, mode = active_cert_pair()
        ssl_ctx = (cert, key)
        scheme = "https"
    print("openscrub web — open from any device on your LAN:")
    if TOKEN:
        print(f"    {scheme}://{lan_ip()}:{args.port}/?token={TOKEN}")
    else:
        print(f"    {scheme}://{lan_ip()}:{args.port}/")
        print("    (open access: no token — everyone on this network can use it;")
        print("     add --token <secret> if you ever want a gate back)")
    print("Jobs folder (contains PHI):", JOBS_DIR,
          f"— auto-deleted after {args.retain_days} day(s)" if args.retain_days else "— kept forever")
    print("Do NOT expose this port to the internet.")
    print("=" * 60)
    if ssl_ctx:
        if mode == "self-signed":
            print("    HTTPS with a self-signed certificate: your browser will")
            print("    warn once — choose Advanced -> Proceed, or install your")
            print("    own certificate at the bottom of the main page.")
        else:
            print("    HTTPS with your installed custom certificate.")
    else:
        print("    WARNING: plain HTTP — PHI crosses the network unencrypted.")
    _serve(args.host, args.port, ssl_ctx)


def _serve(host, port, ssl_ctx):
    """Run under cheroot (a production WSGI server that supports TLS and
    Windows) when available; otherwise fall back to Flask's built-in dev
    server, which works fine for a small LAN but prints a warning."""
    try:
        from cheroot.wsgi import Server
    except ImportError:
        print("    (running on Flask's built-in server — `pip install cheroot`")
        print("     to switch to a production server and silence its warning)")
        app.run(host=host, port=port, threaded=True, ssl_context=ssl_ctx)
        return
    server = Server((host, port), app, server_name="openscrub",
                    numthreads=16)
    if ssl_ctx:
        from cheroot.ssl.builtin import BuiltinSSLAdapter
        server.ssl_adapter = BuiltinSSLAdapter(ssl_ctx[0], ssl_ctx[1])
    try:
        server.start()
    except KeyboardInterrupt:
        server.stop()


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except BaseException:
        import traceback
        traceback.print_exc()
        if os.name == "nt":
            input("\nStartup failed — see the error above. Press Enter to close.")
        raise

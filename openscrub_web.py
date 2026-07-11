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
def _data_root():
    """Writable data root. From a normal checkout/deploy: the script's own
    folder (unchanged behavior). When pip-installed (module lives inside
    site-packages), use a per-user data dir instead — PHI jobs and TLS keys
    must never be written into Python's install tree."""
    here = os.path.dirname(os.path.abspath(__file__))
    if "site-packages" in here or "dist-packages" in here:
        base = (os.environ.get("LOCALAPPDATA")
                or os.path.join(os.path.expanduser("~"), ".local", "share"))
        root = os.path.join(base, "OpenScrub")
        os.makedirs(root, exist_ok=True)
        return root
    return here


JOBS_DIR = os.path.join(_data_root(), "openscrub_jobs")
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
CERT_DIR = os.path.join(_data_root(), "certs")
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
    p = os.path.join(ASSET_DIR, "logo.png")
    if not os.path.exists(p):
        abort(404)
    return send_file(p,
                     mimetype="image/png", max_age=86400)


@app.route("/logo_dark.png")
def logo_dark():
    p = os.path.join(ASSET_DIR, "logo_dark.png")
    if not os.path.exists(p):
        abort(404)
    return send_file(p,
                     mimetype="image/png", max_age=86400)


@app.route("/favicon.ico")
def favicon():
    p = os.path.join(ASSET_DIR, "openscrub.ico")
    if not os.path.exists(p):
        abort(404)
    return send_file(p,
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

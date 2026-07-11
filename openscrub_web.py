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
    site-packages) or frozen into an exe (PyInstaller under Program Files),
    use a per-user data dir instead — PHI jobs and TLS keys must never be
    written into the install tree."""
    here = os.path.dirname(os.path.abspath(__file__))
    if ("site-packages" in here or "dist-packages" in here
            or getattr(sys, "frozen", False)):
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
ZONES_PATH = os.path.join(_data_root(), "zones.json")
# One-time migration: zones.json used to live next to the code, which for
# pip installs meant site-packages (lost on upgrade) and for frozen installs
# would be read-only. Carry an existing file over to the data root.
_legacy_zones = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "zones.json")
if (_legacy_zones != ZONES_PATH and os.path.exists(_legacy_zones)
        and not os.path.exists(ZONES_PATH)):
    try:
        import shutil as _sh
        _sh.copy2(_legacy_zones, ZONES_PATH)
    except OSError:
        pass
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
    # user-defined regex categories ride along on every job; the engine only
    # activates the ones whose id is in --categories
    for c in load_custom_cats():
        argv += ["--custom-regex", "%s=%s" % (c["id"], c["regex"])]
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


@app.before_request
def enforce_vault_lock():
    """While the vault is locked, every job operation is refused — the
    files are ciphertext, so this fails closed rather than half-working."""
    if request.path.startswith("/api/jobs") and vault_locked():
        abort(423, "vault is locked — unlock it (Encryption panel) to "
                   "access jobs")


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
    # models can live next to the code (folder deploys) or in the per-user
    # data dir (pip / frozen installs download there) — check both, matching
    # PlateDetector's own search.
    roots = [os.path.dirname(os.path.abspath(engine.__file__))]
    if engine.install_is_readonly():
        roots.append(engine.user_data_dir())

    def _find(fname):
        for r in roots:
            p = os.path.join(r, "models", fname)
            if os.path.exists(p):
                return p
        return None

    out = []
    for m in models:
        out.append({
            "id": m.get("id"), "label": m.get("label"),
            "license": m.get("license"), "source_url": m.get("source_url"),
            "notes": m.get("notes"), "recommended": bool(m.get("recommended")),
            "verified": m.get("download_url") not in (None, "", "TODO_VERIFY"),
            "pinned": bool(m.get("sha256")) and m.get("sha256") != "TODO_VERIFY",
            "installed": bool(_find("%s.onnx" % m.get("id"))),
            "attribution": m.get("attribution", ""),
        })
    # is any model active (i.e. would PlateDetector find one)?
    active = os.environ.get("OPENSCRUB_PLATE_MODEL")
    if not (active and os.path.exists(active)):
        active = _find("plate_yolov8.onnx")
    if not active:
        for m in models:
            p = _find("%s.onnx" % m.get("id"))
            if p:
                active = p; break
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
<div id="mainview">
<div class="card" id="newjob">
<h2>Job Settings <a href="#settings" title="App settings" style="float:right;font-size:20px;text-decoration:none" aria-label="App settings">&#9881;&#65039;</a></h2>
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
<label style="margin:0"><input type="checkbox" id="drawscores"> show face scores (preview)<span class="qm" data-tip="In preview mode, labels each face box with its detection confidence so you can pick a good Face threshold for your footage.">?</span></label>
<label style="margin:0"><input type="checkbox" id="densefaces"> dense faces (every frame)<span class="qm" data-tip="Runs the face detector on EVERY frame instead of at scan intervals, so fast-moving faces stay covered (e.g. someone walking through a webcam feed). Slower to render — pair it with a face detection zone to keep it fast. Leave off for static screens where faces don't move.">?</span></label>
<button onclick="startJob()">Start scan</button>
</div>
</div>

<div class="card"><h2>Custom regex categories</h2>
<div style="font-size:12px;color:#6b7280;margin-bottom:6px">Add your own
detection categories — claim numbers, case IDs, account formats, anything
regex-matchable. Each appears as a category checkbox above, in review and
reports, and on the <a href="zones">detection zones</a> page. No limit;
one at a time, and every category needs a name.</div>
<div id="cclist" style="font-size:13px"></div>
<div class="row" style="gap:8px;flex-wrap:wrap;font-size:13px;margin-top:6px">
 <input id="ccname" placeholder="name (required, e.g. Claim numbers)">
 <input id="ccregex" placeholder="regex, e.g. CLM-\\d{6}" style="min-width:220px">
 <button onclick="ccAdd()">Add category</button>
</div>
</div>
<div class="card"><h2>Jobs</h2><div class="joblist" id="jobs">loading…</div></div>
<div id="detail"></div>
</div>
<div id="settingsview" style="display:none">
<div class="card"><h2>App settings <a href="#" style="float:right;font-size:14px;font-weight:400" onclick="location.hash=&quot;&quot;;return false">&#8592; back to jobs</a></h2>
<div style="font-size:12px;color:#6b7280">Server-level configuration — these apply to every job.</div>
</div>
<div class="card" id="platepanel" style="display:none">
<h2>License-plate model <span class="qm" data-tip="Plate detection needs a model file (not bundled). Pick a curated open-source model to download it — the file is SHA-256 verified before use. Each entry shows its software license.">?</span></h2>
 <div id="platestatus" style="font-size:12px;color:#6b7280;margin-bottom:6px">checking…</div>
 <div id="platelist"></div>
</div>
<div class="card"><h2>Optional detection engines</h2>
<div id="extras" style="font-size:13px">loading…</div>
</div>
<div class="card"><h2>Encryption at rest</h2>
<div id="vstat" style="font-size:13px;color:#6b7280;margin-bottom:8px">loading…</div>
<div id="vsetup" style="display:none">
 <div style="background:#fef3c7;border:1px solid #f59e0b;border-radius:8px;padding:10px 12px;font-size:13px;margin-bottom:8px">
  <b>&#9888; No password reset exists.</b> If you lose this password, every
  encrypted job — uploads, audit reports, rendered output — is
  <b>permanently unrecoverable</b>. Write it down and store it safely.
 </div>
 <div class="row" style="gap:8px;flex-wrap:wrap;font-size:13px">
  <input type="password" id="vpw1" placeholder="password (8+ chars)">
  <input type="password" id="vpw2" placeholder="repeat password">
  <button onclick="vaultSetup()">Enable encryption</button>
 </div>
 <div style="font-size:12px;color:#9ca3af;margin-top:6px">Job files are
 encrypted (AES-256) whenever you lock or shut the server down, and
 decrypted while unlocked so scanning and review work normally. Pair with
 OS disk encryption (e.g. BitLocker) for full coverage.</div>
</div>
<div id="vunlock" style="display:none" class="row">
 <input type="password" id="vpw" placeholder="vault password">
 <button onclick="vaultUnlock()">Unlock</button>
</div>
<div id="vlock" style="display:none" class="row">
 <button onclick="vaultDoLock()">Lock now (encrypt all job files)</button>
</div>
</div>
<div class="card"><h2>HTTPS certificate</h2>
<div id="certinfo" style="font-size:13px;color:#6b7280;margin-bottom:8px">loading…</div>
<div class="row" style="gap:8px;flex-wrap:wrap;font-size:13px">
 <label>Certificate (PEM): <input type="file" id="certfile" accept=".pem,.crt,.cer"></label>
 <label>Private key (PEM): <input type="file" id="keyfile" accept=".pem,.key"></label>
 <button onclick="uploadCerts()">Install</button>
 <button onclick="removeCerts()">Remove custom cert</button>
</div>
<div style="font-size:12px;color:#9ca3af;margin-top:6px">Installing or removing a
certificate takes effect after a server restart.</div>
</div>
</div>
</main>
<footer style="text-align:center;color:#9ca3af;font-size:12px;padding:18px 12px 26px">
OpenScrub v%%VERSION%% <span id="upd"></span>· <a href="license" style="color:#6b7280">Apache-2.0 license</a>
· best-effort redaction — always review output before sharing PHI</footer>
<script>
const CATS=["name","dob","phone","ssn","mrn","email","address","card","apikey","ipaddr","plate","face"];
const CATMODE={};   // category -> "" (default) | "blur" | "box"
let CC=[];          // custom regex categories, loaded by ccLoad()
function renderCats(){
 const gm=document.getElementById("mode").value;
 document.getElementById("cats").innerHTML=CATS.concat(CC.map(x=>x.id)).map(c=>{
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
 const r=await fetch("api/jobs");
 if(r.status===423){jobs.innerHTML='<span style="color:#b91c1c">&#128274; locked — unlock in the Encryption panel below</span>';JOBSJSON="";return;}
 const js=await r.json();
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
async function updCheck(){
 try{
  const d=await (await fetch("/api/update_check")).json();
  if(d.available)document.getElementById("upd").innerHTML=
   '· <a href="#" style="color:#b45309" onclick="updRun();return false">v'
   +d.latest+' available — update</a> ';
 }catch(e){}
}
async function updRun(){
 if(!confirm("Update OpenScrub to the newest release now?\\n\\nJobs must be idle, and the server needs a restart afterwards to run the new version."))return;
 const r=await fetch("/api/update_run",{method:"POST"});
 if(!r.ok){alert(await r.text());return;}
 const el=document.getElementById("upd");
 el.textContent="· updating… ";
 const t=setInterval(async()=>{
  const s=await (await fetch("/api/update_status")).json();
  if(!s.running&&s.ok!==null){
   clearInterval(t);
   el.textContent=s.ok?"· updated — restart the server to finish ":"· update failed ";
   alert((s.ok?"Update installed.\\nRestart the OpenScrub server to run the new version.":"Update failed:")+"\\n\\n"+s.log.slice(-8).join("\\n"));
  }
 },2000);
}
async function vaultStatus(){
 try{
  const d=await (await fetch("/api/vault")).json();
  const st=document.getElementById("vstat");
  document.getElementById("vsetup").style.display=d.enabled?"none":"block";
  document.getElementById("vunlock").style.display=(d.enabled&&d.locked)?"flex":"none";
  document.getElementById("vlock").style.display=(d.enabled&&!d.locked)?"flex":"none";
  if(!d.enabled){st.textContent="Disabled — job files (PHI) are stored in plaintext. Set a password to enable at-rest encryption.";st.style.color="#b45309";}
  else if(d.locked){st.textContent="LOCKED — "+d.encrypted_files+" file(s) encrypted on disk. Unlock to work with jobs.";st.style.color="#b91c1c";}
  else{st.textContent="Unlocked — files decrypt while you work; they re-encrypt when you lock or shut down. Losing the password makes encrypted files unrecoverable.";st.style.color="#15803d";}
 }catch(e){}
}
async function vaultSetup(){
 const a=document.getElementById("vpw1").value,b=document.getElementById("vpw2").value;
 if(a.length<8){alert("Password must be at least 8 characters.");return;}
 if(a!==b){alert("Passwords do not match.");return;}
 if(!confirm("Enable at-rest encryption?\\n\\nTHERE IS NO PASSWORD RESET. If you lose this password, every encrypted job file is PERMANENTLY UNRECOVERABLE.\\n\\nContinue?"))return;
 const r=await fetch("/api/vault/setup",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({password:a})});
 alert(r.ok?(await r.json()).note:await r.text());
 vaultStatus();
}
async function vaultUnlock(){
 const r=await fetch("/api/vault/unlock",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({password:document.getElementById("vpw").value})});
 if(!r.ok){alert(await r.text());return;}
 document.getElementById("vpw").value="";
 vaultStatus();loadJobs();
}
async function vaultDoLock(){
 if(!confirm("Encrypt all job files now?\\n\\nYou will need the password to access them again."))return;
 const r=await fetch("/api/vault/lock",{method:"POST"});
 alert(r.ok?("Locked — "+(await r.json()).encrypted+" file(s) encrypted."):await r.text());
 vaultStatus();loadJobs();
}
async function ccLoad(){
 try{
  CC=await (await fetch("/api/custom_cats")).json();
  renderCats();
  document.getElementById("cclist").innerHTML=CC.length?CC.map(c=>
   `<div class="row" style="justify-content:space-between;padding:3px 0"><span><b>${c.label}</b> <code style="color:#6b7280">${c.regex}</code></span><button onclick="ccDel('${c.id}')">remove</button></div>`).join(""):'<span style="color:#9ca3af">none yet</span>';
 }catch(e){}
}
async function ccAdd(){
 const name=document.getElementById("ccname").value.trim(),rx=document.getElementById("ccregex").value;
 if(!name){alert("Name the new category first — the name is required and appears on the zones page.");return;}
 if(!rx){alert("A regex pattern is required.");return;}
 const r=await fetch("/api/custom_cats",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({name:name,regex:rx})});
 if(!r.ok){alert(await r.text());return;}
 document.getElementById("ccname").value="";document.getElementById("ccregex").value="";
 ccLoad();
}
async function ccDel(id){
 if(!confirm("Remove this category? Future scans stop detecting it; existing reports are unaffected."))return;
 await fetch("/api/custom_cats/"+id,{method:"DELETE"});
 ccLoad();
}
let EXTPOLL=null;
async function extrasStatus(){
 try{
  const d=await (await fetch("/api/extras")).json();
  const el=document.getElementById("extras");
  if(d.frozen){el.innerHTML='This standalone install has no pip, so optional engines cannot be added here. Use the pip install of OpenScrub (<code>pip install "OpenScrub[ner]"</code>) if you need spaCy NER or PaddleOCR.';return;}
  el.innerHTML=d.items.map(i=>{
   let right;
   if(i.installed)right='<span style="color:#15803d">installed</span>';
   else if(d.state==="installing"&&d.target===i.id)right='<span style="color:#b45309">installing… '+(d.log.length?d.log[d.log.length-1]:'')+'</span>';
   else right=`<button onclick="extraInstall('${i.id}')">Install</button>`;
   return '<div class="row" style="justify-content:space-between;gap:10px;padding:5px 0"><span><b>'+i.label+'</b><br><span style="color:#6b7280;font-size:12px">'+i.desc+'</span></span>'+right+'</div>';
  }).join("");
  if(d.state==="installing"){if(!EXTPOLL)EXTPOLL=setInterval(extrasStatus,3000);}
  else if(EXTPOLL){clearInterval(EXTPOLL);EXTPOLL=null;
   if(d.state==="error")alert("Engine install failed:\\n"+d.log.join("\\n"));}
 }catch(e){}
}
async function extraInstall(id){
 const r=await fetch("/api/extras/"+id+"/install",{method:"POST"});
 if(!r.ok){alert(await r.text());return;}
 extrasStatus();
}
zoneStatus();loadPersist();loadCertInfo();loadJobs();setInterval(loadJobs,5000);
function showView(){
 const s=location.hash==="#settings";
 document.getElementById("mainview").style.display=s?"none":"block";
 document.getElementById("settingsview").style.display=s?"block":"none";
 if(s)window.scrollTo(0,0);
}
window.addEventListener("hashchange",showView);
updCheck();vaultStatus();extrasStatus();ccLoad();showView();
</script></body></html>"""


ASSET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ----------------------------------------------------------------------------
# Self-update (openscrub_update.py does the heavy lifting). The update
# endpoints refuse to run while any job is queued or processing, and an
# installed update only takes effect after the server restarts.
# ----------------------------------------------------------------------------
try:
    import openscrub_update
except Exception:                       # module missing: endpoints degrade
    openscrub_update = None

UPD = {"running": False, "log": [], "ok": None, "cache": (0.0, None)}
UPD_LOCK = threading.Lock()


@app.route("/api/update_check")
def update_check():
    if openscrub_update is None:
        return jsonify({"current": openscrub.VERSION, "latest": None,
                        "available": False})
    now = time.time()
    with UPD_LOCK:
        ts, latest = UPD["cache"]
        if now - ts > 6 * 3600 or request.args.get("force"):
            try:
                latest = openscrub_update.get_latest(timeout=6)
            except Exception:
                latest = None       # offline / PyPI down: just no notice
            UPD["cache"] = (now, latest)
    avail = bool(latest and openscrub_update.is_newer(latest["version"],
                                                      openscrub.VERSION))
    return jsonify({"current": openscrub.VERSION,
                    "latest": latest["version"] if latest else None,
                    "available": avail})


@app.route("/api/update_run", methods=["POST"])
def update_run():
    if openscrub_update is None:
        abort(400, "updater module not available")
    with JOBS_LOCK:
        busy = any(j.get("phase") in ("queued", "scanning", "rendering",
                                      "queued_render")
                   for j in JOBS.values())
    if busy:
        abort(409, "a job is queued or running — update after it finishes")
    with UPD_LOCK:
        if UPD["running"]:
            return jsonify({"started": False, "reason": "already running"})
        UPD["running"], UPD["log"], UPD["ok"] = True, [], None

    def _go():
        def log(msg):
            with UPD_LOCK:
                UPD["log"].append(str(msg))
        try:
            openscrub_update.run_update(log=log)
            ok = True
        except Exception as e:
            log("update failed: %s" % e)
            ok = False
        with UPD_LOCK:
            UPD["ok"], UPD["running"] = ok, False

    threading.Thread(target=_go, daemon=True).start()
    return jsonify({"started": True})


@app.route("/api/update_status")
def update_status():
    with UPD_LOCK:
        return jsonify({"running": UPD["running"], "ok": UPD["ok"],
                        "log": list(UPD["log"])})


# ----------------------------------------------------------------------------
# Vault: password-based at-rest encryption of the job store (PHI!).
# LOCKED = job files encrypted on disk, job APIs refuse to run.
# UNLOCKED = decrypted in place so the pipeline works unchanged.
# NO PASSWORD RESET EXISTS — a lost password means the data is gone.
# ----------------------------------------------------------------------------
import openscrub_vault as vault

VAULT = {"key": None}          # data key while unlocked; None = locked
VAULT_LOCK = threading.Lock()


def vault_enabled():
    return vault.keystore_exists(_data_root())


def vault_locked():
    return vault_enabled() and VAULT["key"] is None


def _vault_lock_now():
    """Encrypt the job store and forget the key. Returns files encrypted."""
    with VAULT_LOCK:
        key = VAULT["key"]
        if key is None:
            return 0
        n = vault.encrypt_tree(key, JOBS_DIR)
        VAULT["key"] = None
        return n


def _vault_lock_atexit():
    if vault_enabled() and VAULT["key"] is not None:
        busy = any(j.get("phase") in ("queued", "scanning", "rendering",
                                      "queued_render") for j in JOBS.values())
        if busy:
            print("vault: NOT locking on exit — a job was still running; "
                  "job files remain in plaintext. Restart and lock.")
            return
        n = _vault_lock_now()
        print("vault: locked on shutdown (%d file(s) encrypted)" % n)


@app.route("/api/vault")
def vault_status():
    enc, plain = (0, 0)
    if vault_enabled():
        enc, plain = vault.tree_locked_state(JOBS_DIR)
    return jsonify({"enabled": vault_enabled(), "locked": vault_locked(),
                    "encrypted_files": enc, "plaintext_files": plain})


@app.route("/api/vault/setup", methods=["POST"])
def vault_setup():
    pw = (request.json or {}).get("password") or ""
    if vault_enabled():
        abort(409, "vault already set up")
    if len(pw) < 8:
        abort(400, "password must be at least 8 characters")
    with VAULT_LOCK:
        VAULT["key"] = vault.create_keystore(_data_root(), pw)
    return jsonify({"ok": True, "note":
                    "Encryption enabled. Jobs encrypt when you lock or "
                    "shut down. LOSING THIS PASSWORD MAKES THE ENCRYPTED "
                    "FILES PERMANENTLY UNRECOVERABLE."})


@app.route("/api/vault/unlock", methods=["POST"])
def vault_unlock():
    pw = (request.json or {}).get("password") or ""
    if not vault_enabled():
        abort(400, "vault is not set up")
    try:
        key = vault.open_keystore(_data_root(), pw)
    except ValueError:
        abort(403, "wrong password")
    with VAULT_LOCK:
        VAULT["key"] = key
        n = vault.decrypt_tree(key, JOBS_DIR)
    rehydrate_jobs()               # pick up jobs that were locked at startup
    return jsonify({"ok": True, "decrypted": n})


# ----------------------------------------------------------------------------
# Optional engines (spaCy NER, PaddleOCR) — installable from the web UI on
# pip/folder installs. Frozen (Program Files) builds have no pip, so the
# endpoints report that instead of pretending.
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# Custom regex categories — user-defined, stored in the data root, injected
# into the category checkboxes, job args, and the zones page.
# ----------------------------------------------------------------------------
CUSTOM_CATS_PATH = os.path.join(_data_root(), "custom_categories.json")
BUILTIN_CATS = {"name", "dob", "phone", "ssn", "mrn", "email", "address",
                "card", "apikey", "ipaddr", "plate", "face"}


def load_custom_cats():
    try:
        with open(CUSTOM_CATS_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def _cat_color(cid):
    import colorsys, hashlib
    h = int(hashlib.sha256(cid.encode()).hexdigest()[:6], 16) / 0xFFFFFF
    r, g, b = colorsys.hls_to_rgb(h, 0.45, 0.75)
    return "#%02x%02x%02x" % (int(r * 255), int(g * 255), int(b * 255))


@app.route("/api/custom_cats", methods=["GET", "POST"])
def custom_cats():
    cats = load_custom_cats()
    if request.method == "GET":
        return jsonify(cats)
    d = request.json or {}
    name = (d.get("name") or "").strip()
    pattern = (d.get("regex") or "").strip()
    if not name:
        abort(400, "the category needs a name")
    cid = "".join(ch for ch in name.lower().replace(" ", "_")
                  if ch.isalnum() or ch == "_")[:24]
    if not cid:
        abort(400, "name must contain letters or digits")
    if cid in BUILTIN_CATS or any(c["id"] == cid for c in cats):
        abort(409, "a category with that name already exists")
    try:
        import re as _re
        _re.compile(pattern)
    except Exception as e:
        abort(400, "invalid regex: %s" % e)
    if not pattern:
        abort(400, "a regex pattern is required")
    cats.append({"id": cid, "label": name, "regex": pattern,
                 "color": _cat_color(cid)})
    with open(CUSTOM_CATS_PATH, "w", encoding="utf-8") as f:
        json.dump(cats, f, indent=2)
    return jsonify({"ok": True, "id": cid})


@app.route("/api/custom_cats/<cid>", methods=["DELETE"])
def custom_cats_delete(cid):
    cats = [c for c in load_custom_cats() if c["id"] != cid]
    with open(CUSTOM_CATS_PATH, "w", encoding="utf-8") as f:
        json.dump(cats, f, indent=2)
    return jsonify({"ok": True})


def _spec(name):
    import importlib.util
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


EXTRAS = {
    "ner": {
        "label": "spaCy NER (name detection)",
        "desc": "The primary name detector. Without it, names rely on "
                "the built-in heuristics.",
        "pip": ["spacy"],
        "post": [["-m", "spacy", "download", "en_core_web_sm"]],
        "check": lambda: _spec("spacy") and _spec("en_core_web_sm"),
    },
    "paddle": {
        "label": "PaddleOCR — CPU (better OCR on small fonts)",
        "desc": "Stronger OCR engine, picked automatically once "
                "installed. Large download.",
        "pip": ["paddleocr", "paddlepaddle"],
        "post": [],
        "check": lambda: _spec("paddleocr") and _spec("paddle"),
    },
    "paddle_gpu": {
        "label": "PaddleOCR — NVIDIA GPU (CUDA 12.x)",
        "desc": "GPU-accelerated build for NVIDIA cards — much faster "
                "OCR. Very large download. Shows installed if either "
                "Paddle build is present.",
        "pip": ["paddleocr", "paddlepaddle-gpu==3.2.2", "-i",
                "https://www.paddlepaddle.org.cn/packages/stable/cu126/"],
        "post": [["-m", "pip", "install", "-U", "nvidia-cudnn-cu12"]],
        "check": lambda: _spec("paddleocr") and _spec("paddle"),
    },
}
_EXT = {"state": "idle", "target": "", "log": []}


@app.route("/api/extras")
def extras_status():
    frozen = bool(getattr(sys, "frozen", False))
    items = [{"id": k, "label": v["label"], "desc": v["desc"],
              "installed": (False if frozen else v["check"]())}
             for k, v in EXTRAS.items()]
    return jsonify({"frozen": frozen, "items": items,
                    "state": _EXT["state"], "target": _EXT["target"],
                    "log": _EXT["log"][-4:]})


@app.route("/api/extras/<eid>/install", methods=["POST"])
def extras_install(eid):
    if getattr(sys, "frozen", False):
        abort(400, "this is a standalone (Program Files) install without "
                   "pip — optional engines need the pip install of "
                   "OpenScrub")
    spec = EXTRAS.get(eid) or abort(404)
    if _EXT["state"] == "installing":
        abort(409, "another install is already running")
    _EXT.update(state="installing", target=eid, log=[])

    def work():
        import subprocess
        try:
            cmds = [[sys.executable, "-m", "pip", "install"] + spec["pip"]]
            cmds += [[sys.executable] + p for p in spec["post"]]
            for cmd in cmds:
                _EXT["log"].append("$ " + " ".join(cmd[1:]))
                r = subprocess.run(cmd, capture_output=True, text=True)
                if r.returncode != 0:
                    tail = (r.stderr or r.stdout or "").strip().splitlines()
                    _EXT["log"] += tail[-3:] or ["exit %d" % r.returncode]
                    _EXT["state"] = "error"
                    return
            _EXT["log"].append("done — takes effect on the next scan")
            _EXT["state"] = "done"
        except Exception as e:
            _EXT["log"].append(str(e))
            _EXT["state"] = "error"
    threading.Thread(target=work, daemon=True).start()
    return jsonify({"ok": True})


@app.route("/api/vault/lock", methods=["POST"])
def vault_lock_route():
    with JOBS_LOCK:
        busy = any(j.get("phase") in ("queued", "scanning", "rendering",
                                      "queued_render")
                   for j in JOBS.values())
    if busy:
        abort(409, "a job is queued or running — lock after it finishes")
    n = _vault_lock_now()
    return jsonify({"ok": True, "encrypted": n})


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
    return PAGE.replace("%%VERSION%%", openscrub.VERSION)


def _header_logo_uri():
    """The base64 logo embedded in PAGE — reused by the zones page so its
    header never 404s on installs without an assets/ folder (pip, frozen)."""
    i = PAGE.index('<header><img src="') + len('<header><img src="')
    return PAGE[i:PAGE.index('"', i)]


@app.route("/zones")
def zones_page():
    page = zones_ui.ZONES_PAGE.replace('src="logo_dark.png"',
                                       'src="%s"' % _header_logo_uri())
    # inject user-defined categories into the zone editor's color map so
    # they can be zoned exactly like built-ins
    extra = "".join(',%s:"%s"' % (c["id"], c.get("color", "#64748b"))
                    for c in load_custom_cats())
    if extra:
        page = page.replace('face:"#ec4899"}', 'face:"#ec4899"%s}' % extra)
    return page


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
    import atexit
    atexit.register(_vault_lock_atexit)
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
    if vault_enabled():
        print("Encryption: vault is LOCKED — unlock in the web UI "
              "(Encryption panel) to access jobs."
              if vault_locked() else "Encryption: vault unlocked.")
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
    # Browsers probing/rejecting the self-signed certificate abort mid-
    # handshake on every new socket, and cheroot logs each one. Harmless
    # noise (install a trusted cert to stop the aborts themselves) —
    # filter just that class of message, pass everything else through.
    _orig_log = server.error_log
    _TLS_NOISE = ("during handshake", "certificate unknown", "unknown ca",
                  "peer dropped the tls connection")

    def _quiet_log(msg="", level=20, traceback=False):
        if any(s in str(msg).lower() for s in _TLS_NOISE):
            return
        _orig_log(msg, level, traceback)

    server.error_log = _quiet_log
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

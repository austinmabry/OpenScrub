#!/usr/bin/env python3
"""Regression tests for openscrub. Run: pytest test_openscrub.py -v
For this tool a regression is not a bug, it's a PHI leak — these tests are
the gate for any change."""

import json
import os
import subprocess

import cv2
import numpy as np
import pytest

import openscrub

FONT = cv2.FONT_HERSHEY_SIMPLEX


def make_video(path, lines, seconds=1.5, fps=30):
    img = np.full((720, 1280, 3), 245, np.uint8)
    for y, txt in lines:
        cv2.putText(img, txt, (60, y), FONT, 0.9, (20, 20, 20), 2)
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (1280, 720))
    for _ in range(int(seconds * fps)):
        out.write(img)
    out.release()
    return path


def run(video, *extra):
    parser = openscrub.build_parser()
    args = parser.parse_args([video, "--engine", "tesseract",
                              "-o", video.replace(".mp4", "_red.mp4"),
                              "--report", video.replace(".mp4", "_aud.json"),
                              *extra])
    args = openscrub._prep_args(args, parser)

    class Quiet(openscrub.Callbacks):
        def log(self, m):
            pass
    res = openscrub.run_pipeline(args, Quiet())
    dets = json.load(open(video.replace(".mp4", "_aud.json")))["detections"]
    return res, dets


def sharp(path, t, box):
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    _, f = cap.read()
    cap.release()
    x1, y1, x2, y2 = [int(v) for v in box]
    roi = cv2.cvtColor(f[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(roi, cv2.CV_64F).var()


# ---------------------------------------------------------------- unit tests

def test_mrn_regex_precision():
    """The default MRN token shape is generic: standalone 6-10 digit runs
    with an optional short letter prefix. (detect_phi separately requires a
    nearby MRN/chart/acct label OR 7+ digits before flagging.) Site-specific
    formats belong in --mrn-regex, never hardcoded as the default."""
    rx = openscrub.RE_MRN_DEFAULT
    import re
    r = re.compile(rx)
    assert r.search("1234567")               # plain 7-digit run
    assert r.search("MM0123456789")          # letter prefix + digits
    assert r.search("4829173")               # any 7 digits — no magic prefix
    assert r.search("123456")                # 6 digits (label-gated upstream)
    assert not r.search("12345")             # too short
    assert not r.search("12345678901")       # too long (11+): not MRN-shaped
    assert not r.search("ABCD123456")        # prefix too long
    assert not r.search("4111111111111111")  # card-length: left to card/Luhn


def test_memory_two_sighting_gate():
    m = openscrub.PhiMemory()
    m.add("Henderson", "name", primary=True)
    assert m.recall("Henderson") is None            # 1 sighting: gated
    m.add("Henderson", "name", primary=True)
    assert m.recall("Henderson") == "name"          # 2 sightings: recalls
    m.add("5015550142", "phone", primary=True)
    assert m.recall("5015550142") == "phone"        # regex cats: immediate
    m.add("face", "face")                           # never memorized
    assert "face" not in m.items


def test_probe_vfr(tmp_path):
    cfr = make_video(str(tmp_path / "cfr.mp4"), [(150, "hello world")])
    is_vfr, avg = openscrub.probe_vfr(cfr)
    assert not is_vfr and abs(avg - 30) < 1
    vfr = str(tmp_path / "vfr.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", cfr,
                    "-vf", "select='not(mod(n\\,3))'", "-fps_mode", "vfr",
                    vfr], check=True)
    is_vfr, _ = openscrub.probe_vfr(vfr)
    assert is_vfr


def test_config_profile(tmp_path):
    cfg = tmp_path / "p.yaml"
    cfg.write_text("pad: 3\nignore_regions:\n  - [0, 600, 1280, 720]\n")
    parser = openscrub.build_parser()
    args = parser.parse_args(["x.mp4", "--config", str(cfg)])
    args = openscrub.apply_config(args, parser)
    assert args.pad == 3
    assert args.ignore_regions == [(0, 600, 1280, 720)]
    # CLI wins over config
    args2 = parser.parse_args(["x.mp4", "--config", str(cfg), "--pad", "11"])
    args2 = openscrub.apply_config(args2, parser)
    assert args2.pad == 11


# ------------------------------------------------------------ pipeline tests

@pytest.fixture(scope="module")
def chart(tmp_path_factory):
    d = tmp_path_factory.mktemp("vids")
    v = make_video(str(d / "chart.mp4"),
                   [(150, "Patient: Robert Henderson"),
                    (230, "MRN: 1234567   DOB: 03/15/1978"),
                    (310, "Assessment and Plan documented")])
    res, dets = run(v)
    return v, res, dets


def test_categories_detected(chart):
    _, _, dets = chart
    cats = {d["category"] for d in dets}
    assert {"name", "mrn", "dob"} <= cats


def test_openscrubred_benign_readable(chart):
    v, res, dets = chart
    red = v.replace(".mp4", "_red.mp4")
    name = [d for d in dets if d["text"] == "Henderson"][0]["cbox"]
    assert sharp(red, 0.7, name) < sharp(v, 0.7, name) * 0.15
    benign = [60, 285, 700, 320]   # "Assessment and Plan" line
    assert sharp(red, 0.7, benign) > sharp(v, 0.7, benign) * 0.7


def test_provenance(chart):
    v, _, _ = chart
    prov = json.load(open(v.replace(".mp4", "_aud.json")))["provenance"]
    assert prov["version"] == openscrub.VERSION
    assert len(prov["input_sha256"]) == 64
    assert len(prov["output_sha256"]) == 64


def test_from_report_respects_edits(chart, tmp_path):
    v, _, _ = chart
    doc = json.load(open(v.replace(".mp4", "_aud.json")))
    for d in doc["detections"]:
        if d["text"] == "Henderson":
            d["enabled"] = False
    doc["detections"].append({
        "t_start": 0, "t_end": 1.5, "cbox": [60, 285, 700, 320],
        "category": "manual", "text": "user", "confidence": 1.0,
        "aoff": [0, 0], "last_seen": 0, "enabled": True})
    edited = str(tmp_path / "edited.json")
    json.dump(doc, open(edited, "w"))
    parser = openscrub.build_parser()
    args = parser.parse_args([v, "--from-report", edited,
                              "-o", str(tmp_path / "rr.mp4")])
    args = openscrub._prep_args(args, parser)

    class Quiet(openscrub.Callbacks):
        def log(self, m):
            pass
    openscrub.run_pipeline(args, Quiet())
    rr = str(tmp_path / "rr.mp4")
    hend = [d for d in doc["detections"] if d.get("text") == "Henderson"][0]["cbox"]
    assert sharp(rr, 0.7, hend) > 500                       # disable honored
    assert sharp(rr, 0.7, [60, 285, 700, 320]) < 100        # manual honored


def test_ignore_region(tmp_path):
    v = make_video(str(tmp_path / "ign.mp4"),
                   [(150, "Patient: Maria Gonzalez"),
                    (700, "03/15/1978")])
    res, dets = run(v, "--ignore-region", "0,650,1280,720")
    assert not any(d["cbox"][1] > 600 for d in dets)         # corner suppressed
    assert any(d["category"] == "name" for d in dets)        # name still caught


def test_update_version_compare():
    import openscrub_update as u
    assert u.is_newer("1.0.10", "1.0.9")
    assert u.is_newer("1.1.0", "1.0.99")
    assert not u.is_newer("1.0.4", "1.0.4")
    assert not u.is_newer("0.9.9", "1.0.0")


def test_update_registry_pin_merge():
    """Locally pinned TOFU hashes must survive an update — but only when
    the download URL is unchanged; a moved URL must NOT inherit trust."""
    import openscrub_update as u
    old = [
        {"id": "a", "download_url": "https://x/a.onnx", "sha256": "aaa"},
        {"id": "b", "download_url": "https://x/b.onnx", "sha256": "bbb"},
        {"id": "c", "download_url": "https://x/c.onnx", "sha256": ""},
    ]
    new = [
        {"id": "a", "download_url": "https://x/a.onnx", "sha256": ""},
        {"id": "b", "download_url": "https://MOVED/b.onnx", "sha256": ""},
        {"id": "c", "download_url": "https://x/c.onnx", "sha256": ""},
        {"id": "d", "download_url": "https://x/d.onnx", "sha256": "ddd"},
    ]
    carried = u.merge_registry_pins(old, new)
    assert carried == 1
    assert new[0]["sha256"] == "aaa"      # same URL: pin carried
    assert new[1]["sha256"] == ""          # URL moved: trust reset
    assert new[2]["sha256"] == ""          # never pinned: stays empty
    assert new[3]["sha256"] == "ddd"       # shipped hash untouched


def test_registry_user_copy_and_pin_survival(tmp_path, monkeypatch):
    """Read-only installs (pip/frozen) must keep TOFU pins in a per-user
    registry copy: seeded once, new release models merged in, existing
    pinned hashes never overwritten."""
    monkeypatch.setattr(openscrub, "install_is_readonly", lambda: True)
    monkeypatch.setattr(openscrub, "user_data_dir", lambda: str(tmp_path))

    # first call seeds the user copy from the packaged registry
    p = openscrub.plate_registry_path()
    assert p == str(tmp_path / "plate_models.json")
    with open(p, encoding="utf-8") as f:
        reg = json.load(f)
    assert reg["models"], "seeded registry should carry the packaged models"

    # pin a hash locally, then simulate a release that adds a new model
    reg["models"][0]["sha256"] = "f" * 64
    with open(p, "w", encoding="utf-8") as f:
        json.dump(reg, f)
    packaged = os.path.join(os.path.dirname(os.path.abspath(
        openscrub.__file__)), "plate_models.json")
    with open(packaged, encoding="utf-8") as f:
        shipped = json.load(f)
    shipped["models"].append({"id": "brand-new", "sha256": "",
                              "download_url": "https://x/new.onnx"})
    fake_pkg = tmp_path / "pkg"; fake_pkg.mkdir()
    (fake_pkg / "plate_models.json").write_text(json.dumps(shipped),
                                                encoding="utf-8")
    real_dirname = os.path.dirname

    def fake_dirname(path):
        if path == os.path.abspath(openscrub.__file__):
            return str(fake_pkg)
        return real_dirname(path)
    monkeypatch.setattr(openscrub.os.path, "dirname", fake_dirname)

    p2 = openscrub.plate_registry_path()
    with open(p2, encoding="utf-8") as f:
        merged = json.load(f)
    ids = [m["id"] for m in merged["models"]]
    assert "brand-new" in ids, "new release model should merge in"
    assert merged["models"][0]["sha256"] == "f" * 64, "local pin must survive"


def test_vault_roundtrip_and_wrong_password(tmp_path):
    import openscrub_vault as v
    key = v.create_keystore(str(tmp_path), "correct horse battery")
    assert v.open_keystore(str(tmp_path), "correct horse battery") == key
    with pytest.raises(ValueError):
        v.open_keystore(str(tmp_path), "wrong password")

    big = tmp_path / "job" / "video.mp4"
    big.parent.mkdir()
    data = os.urandom(5 * 1024 * 1024)      # spans two 4MiB chunks
    big.write_bytes(data)
    (tmp_path / "job" / "report.json").write_text('{"phi": "synthetic"}')

    n = v.encrypt_tree(key, str(tmp_path / "job"))
    assert n == 2
    names = sorted(p.name for p in (tmp_path / "job").iterdir())
    assert names == ["report.json.osvault", "video.mp4.osvault"]
    enc, plain = v.tree_locked_state(str(tmp_path / "job"))
    assert (enc, plain) == (2, 0)

    n = v.decrypt_tree(key, str(tmp_path / "job"))
    assert n == 2
    assert big.read_bytes() == data
    assert (tmp_path / "job" / "report.json").read_text() == '{"phi": "synthetic"}'


def test_vault_tamper_fails_closed(tmp_path):
    import openscrub_vault as v
    key = v.create_keystore(str(tmp_path), "a strong password")
    f = tmp_path / "report.json"
    f.write_text("sensitive")
    enc = v.encrypt_file(key, str(f))
    raw = bytearray(open(enc, "rb").read())
    raw[-1] ^= 0xFF                          # flip one ciphertext bit
    open(enc, "wb").write(bytes(raw))
    with pytest.raises(Exception):
        v.decrypt_file(key, enc)
    assert not f.exists(), "tampered file must NOT decrypt to plaintext"


def test_custom_regex_category(tmp_path):
    """A user-defined regex category detects, reports, and survives the
    full pipeline like a built-in."""
    v = make_video(str(tmp_path / "c.mp4"), [(200, "Claim CLM-123456 filed")])
    _, dets = run(v, "--categories", "claim",
                  "--custom-regex", r"claim=CLM-\d+")
    hits = [d for d in dets if d["category"] == "claim"]
    assert hits, "custom category should appear in the report"


def test_deep_backtrack_finds_true_onset(tmp_path):
    """A region whose first frame lies beyond the RAM backtrack buffer must
    still get its true onset via the post-scan deep file search — and must
    not be extended earlier than it actually appeared."""
    path = str(tmp_path / "late.mp4")
    fps, seconds, appear_at = 30, 6, 3.0
    blank = np.full((720, 1280, 3), 245, np.uint8)
    with_text = blank.copy()
    cv2.putText(with_text, "SSN 123-45-6789", (60, 300), FONT, 1.1,
                (20, 20, 20), 2)
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                          (1280, 720))
    for i in range(seconds * fps):
        out.write(with_text if i / fps >= appear_at else blank)
    out.release()

    _, dets = run(path, "--categories", "ssn",
                  "--sample-interval", "2.0",
                  "--backtrack-window", "0.5")
    ssn = [d for d in dets if d["category"] == "ssn"]
    assert ssn, "SSN should be detected"
    start = min(d["t_start"] for d in ssn)
    assert start <= appear_at + 0.30, \
        f"deep backtrack should reach the true onset, got {start}"
    assert start >= appear_at - 1.0, \
        f"must not extend far before the text existed, got {start}"


def test_gap_verification_refuses_real_absence(tmp_path):
    """A gap where the content GENUINELY left the screen must not be
    bridged, no matter the verification pass — over-blur is fine but a
    6s bogus bridge across truly-absent content means merge is wrong."""
    path = str(tmp_path / "gap.mp4")
    fps = 30
    blank = np.full((720, 1280, 3), 245, np.uint8)
    wt = blank.copy()
    cv2.putText(wt, "SSN 123-45-6789", (60, 300), FONT, 1.1, (20, 20, 20), 2)
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                          (1280, 720))
    for i in range(12 * fps):
        t = i / fps
        out.write(wt if (t < 3 or t >= 9) else blank)
    out.release()
    _, dets = run(path, "--categories", "ssn")
    ssn = sorted((d for d in dets if d["category"] == "ssn"),
                 key=lambda d: d["t_start"])
    assert ssn, "SSN should be detected"
    assert not any(d["t_start"] < 4 and d["t_end"] > 8 for d in ssn), \
        "the true 6s absence must not be blurred straight through"


def test_ignore_zone_blocks_detection(tmp_path):
    """Normalized ignore regions (zone-editor rects) suppress detection
    inside them — and only inside them."""
    v1 = make_video(str(tmp_path / "a.mp4"), [(300, "SSN 123-45-6789")])
    _, dets = run(v1, "--categories", "ssn")
    assert [d for d in dets if d["category"] == "ssn"], "control run detects"
    v2 = make_video(str(tmp_path / "b.mp4"), [(300, "SSN 123-45-6789")])
    _, dets = run(v2, "--categories", "ssn",
                  "--ignore-region", "0,0,1,0.6")
    assert not [d for d in dets if d["category"] == "ssn"], \
        "text centered inside a normalized ignore region must not detect"


def test_dense_hold_not_extended_by_merge():
    """Dense per-frame samples must keep their sub-frame hold through
    merge_detections: stamping them with the multi-second OCR hold left a
    trail of stale blur boxes along a moving face's path (v1.0.21 bug)."""
    d = openscrub.Detection(5.0, 5.01, (100, 100, 160, 160), "face", "face",
                            0.9, (0, 0), dense=True)
    merged = openscrub.merge_detections([d], hold=2.3)
    assert merged[0].t_end <= 5.02, \
        "dense sample t_end must not be extended by the OCR hold"


def test_dense_track_flicker_interpolation():
    """A short detector flicker inside a dense track is filled by
    interpolating the box between the surrounding samples: the blur must
    MOVE with the face across the gap, not vanish or hang at the pre-gap
    position (the v1.0.21 sideways-trail bug, inverted)."""
    fps = 30.0
    speed = 150.0                      # face drifting right, px/s
    dets = []
    for i in range(6):                 # samples at 0.00 .. 0.17
        t = i / fps
        x = int(speed * t)
        dets.append(openscrub.Detection(t, t + 0.01, (x, 100, x + 60, 160),
                                        "face", "face", 0.9, (0, 0),
                                        dense=True))
    t2 = 0.6                           # detector missed 0.17 .. 0.60
    x2 = int(speed * t2)
    dets.append(openscrub.Detection(t2, t2 + 0.01, (x2, 100, x2 + 60, 160),
                                    "face", "face", 0.9, (0, 0), dense=True))
    openscrub.assign_dense_tracks(dets)
    assert len({d.track for d in dets}) == 1, "one physical face = one track"
    openscrub.smooth_dense_tracks(dets, fps, video=None)
    for t in np.arange(0.20, 0.58, 0.02):
        true_cx = speed * t + 30
        cover = [d for d in dets if d.t_start - 1e-6 <= t <= d.t_end + 1e-6]
        assert cover, f"flicker gap not covered at t={t:.2f}"
        assert any(abs((d.cbox[0] + d.cbox[2]) / 2 - true_cx) < 45
                   for d in cover), \
            f"box does not track the moving face at t={t:.2f}"


def test_dense_onset_walkback(tmp_path):
    """The first dense detection of a track is template-matched backward
    through the file: frames where the object was visible before the
    detector's first hit must end up covered (the v1.0.21 onset leak),
    and the walked boxes must follow the object's true positions."""
    path = str(tmp_path / "onset.mp4")
    fps = 30
    rng = np.random.default_rng(7)
    patch = rng.integers(0, 255, (64, 64, 3)).astype(np.uint8)

    def pos(t):                        # moving right at 120 px/s
        return int(60 + 120 * t), 140

    appear = 0.5
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                          (640, 360))
    for i in range(3 * fps):
        t = i / fps
        fr = np.full((360, 640, 3), 200, np.uint8)
        if t >= appear:
            x, y = pos(t)
            fr[y:y + 64, x:x + 64] = patch
        out.write(fr)
    out.release()

    t0 = 1.5                           # detector "first fires" a second late
    x0, y0 = pos(t0)
    dets = [openscrub.Detection(t0, t0 + 0.01, (x0, y0, x0 + 64, y0 + 64),
                                "face", "face", 0.9, (0, 0), dense=True)]
    openscrub.assign_dense_tracks(dets)
    openscrub.smooth_dense_tracks(dets, fps, path)
    start = min(d.t_start for d in dets)
    assert start <= appear + 0.35, \
        f"walk-back should reach near the true first frame, got {start:.2f}"
    assert start >= appear - 0.40, \
        f"must not extend far before the object existed, got {start:.2f}"
    early = [d for d in dets if appear <= d.t_start < 1.0]
    assert early, "walk-back should add pre-detection samples"
    for e in early:
        true_cx = pos(e.t_start)[0] + 32
        got_cx = (e.cbox[0] + e.cbox[2]) / 2
        assert abs(got_cx - true_cx) < 40, \
            f"walked box off the object at t={e.t_start:.2f}"


def test_hdr_probe_and_tonemap(tmp_path):
    """HDR input (PQ transfer / BT.2020 tags) is detected and tone-mapped
    to tagged SDR BT.709 at intake; SDR input is left untouched."""
    import shutil as _sh
    if not (_sh.which("ffmpeg") and _sh.which("ffprobe")):
        pytest.skip("ffmpeg not available")
    if not (openscrub._ffmpeg_has("zscale") and openscrub._ffmpeg_has("tonemap")):
        pytest.skip("ffmpeg lacks zscale/tonemap")
    hdr = str(tmp_path / "hdr.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error",
                    "-f", "lavfi", "-i", "testsrc2=duration=1:size=320x240:rate=30",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-color_primaries", "bt2020", "-color_trc", "smpte2084",
                    "-colorspace", "bt2020nc", hdr], check=True)
    is_hdr, desc = openscrub.probe_hdr(hdr)
    assert is_hdr and desc == "HDR10/PQ"

    class A:
        video = hdr
        output = str(tmp_path / "out.mp4")
        encoder = "cpu"
    args = A()
    openscrub.normalize_vfr(args, openscrub.Callbacks())
    assert args.video.endswith(".sdr.mp4") and os.path.exists(args.video)
    assert args.hdr_tonemapped and args.original_video == hdr
    again, _ = openscrub.probe_hdr(args.video)
    assert not again, "normalized file must probe as SDR"

    sdr = str(tmp_path / "sdr.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error",
                    "-f", "lavfi", "-i", "testsrc2=duration=1:size=320x240:rate=30",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", sdr], check=True)
    assert openscrub.probe_hdr(sdr) == (False, None)

    class B:
        video = sdr
        output = str(tmp_path / "out2.mp4")
        encoder = "cpu"
    args2 = B()
    openscrub.normalize_vfr(args2, openscrub.Callbacks())
    assert args2.video == sdr, "CFR SDR input must be a no-op"


def test_report_roundtrip_preserves_tracks(tmp_path):
    """dense/track must survive report write -> load: rendering rewrites
    the report through this round trip, and losing the track ids exploded
    the re-opened review into one card per frame sample (v1.0.25 bug)."""
    d = openscrub.Detection(1.0, 1.5, (10, 10, 50, 50), "face", "face",
                            0.9, (0, 0), dense=True, track=3)
    args = openscrub.build_parser().parse_args(["dummy.mp4"])
    state = {"fps": 30.0, "cum": [(0.0, 0.0)], "bands": [(0.0, 0.0)],
             "detections": [d], "input_sha256": "x"}
    rp = str(tmp_path / "r.json")
    d.person = 2
    openscrub.write_report(rp, args, state)
    dets, _, _ = openscrub.load_report(rp)
    assert dets[0].dense is True and dets[0].track == 3
    assert dets[0].person == 2, "person id must survive the round trip"


def _hdr_clip(path):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error",
                    "-f", "lavfi",
                    "-i", "testsrc2=duration=2:size=1280x720:rate=30",
                    "-vf", "drawbox=x=50:y=260:w=400:h=60:color=white:t=fill",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-color_primaries", "bt2020", "-color_trc", "smpte2084",
                    "-colorspace", "bt2020nc", path], check=True)


def _vstream(path):
    p = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                        "-show_entries",
                        "stream=codec_name,pix_fmt,color_transfer",
                        "-of", "json", path], capture_output=True, text=True)
    return json.loads(p.stdout)["streams"][0]


def test_hdr_output_preserved(tmp_path):
    """HDR source + default --hdr-output match: output must be 10-bit HEVC
    with the PQ transfer preserved, and the blur must land in it."""
    import shutil as _sh
    if not (_sh.which("ffmpeg") and _sh.which("ffprobe")):
        pytest.skip("ffmpeg not available")
    if openscrub.hevc10_encoder("x264") is None:
        pytest.skip("no 10-bit HEVC encoder in this environment")
    src = str(tmp_path / "hdrsrc.mp4")
    out = cv2.VideoWriter(src + ".tmp.mp4", cv2.VideoWriter_fourcc(*"mp4v"),
                          30, (1280, 720))
    fr = np.full((720, 1280, 3), 245, np.uint8)
    cv2.putText(fr, "SSN 123-45-6789", (60, 300), FONT, 1.1, (20, 20, 20), 2)
    for _ in range(60):
        out.write(fr)
    out.release()
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error",
                    "-i", src + ".tmp.mp4", "-c:v", "libx264",
                    "-color_primaries", "bt2020", "-color_trc", "smpte2084",
                    "-colorspace", "bt2020nc", src], check=True)
    _, dets = run(src, "--categories", "ssn", "--encoder", "x264")
    st = _vstream(src.replace(".mp4", "_red.mp4"))
    assert st["codec_name"] == "hevc", st
    assert "10" in st["pix_fmt"], st
    assert st.get("color_transfer") == "smpte2084", st
    ssn = [d for d in dets if d["category"] == "ssn"]
    assert ssn, "SSN detected on the tone-mapped scan copy"
    b = ssn[0]["cbox"]
    v = sharp(src.replace(".mp4", "_red.mp4"), 1.0,
              (b[0] + 4, b[1] + 4, b[2] - 4, b[3] - 4))
    assert v < 300, f"HDR output not blurred inside the detection ({v:.0f})"


def test_hdr_output_sdr_toggle(tmp_path):
    """--hdr-output sdr on an HDR source: output must be plain SDR H.264
    (the tone-mapped path), never HDR."""
    import shutil as _sh
    if not (_sh.which("ffmpeg") and _sh.which("ffprobe")):
        pytest.skip("ffmpeg not available")
    if not (openscrub._ffmpeg_has("zscale") and openscrub._ffmpeg_has("tonemap")):
        pytest.skip("ffmpeg lacks zscale/tonemap")
    src = str(tmp_path / "hdrsrc2.mp4")
    _hdr_clip(src)
    _, _ = run(src, "--categories", "ssn", "--encoder", "x264",
               "--hdr-output", "sdr")
    st = _vstream(src.replace(".mp4", "_red.mp4"))
    assert st["codec_name"] == "h264", st
    assert st.get("color_transfer") in (None, "bt709", "unknown"), st
    assert openscrub.probe_hdr(src.replace(".mp4", "_red.mp4")) == (False, None)


def test_codec_and_container_honored(tmp_path):
    """--codec hevc must produce an HEVC stream, and a .mkv/.mov output
    path must produce that actual container (the web download previously
    renamed everything .mp4)."""
    import shutil as _sh
    if not (_sh.which("ffmpeg") and _sh.which("ffprobe")):
        pytest.skip("ffmpeg not available")
    if openscrub.hevc10_encoder("x264") is None:
        pytest.skip("no HEVC encoder in this environment")
    src = make_video(str(tmp_path / "c.mp4"), [(300, "SSN 123-45-6789")])
    parser = openscrub.build_parser()
    out = str(tmp_path / "out.mkv")
    args = parser.parse_args([src, "--engine", "tesseract", "-o", out,
                              "--categories", "ssn", "--encoder", "x264",
                              "--codec", "hevc"])
    args = openscrub._prep_args(args, parser)

    class Quiet(openscrub.Callbacks):
        def log(self, m):
            pass
    openscrub.run_pipeline(args, Quiet())
    p = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                        "format=format_name", "-select_streams", "v:0",
                        "-show_entries", "stream=codec_name",
                        "-of", "json", out], capture_output=True, text=True)
    doc = json.loads(p.stdout)
    assert doc["streams"][0]["codec_name"] == "hevc", doc
    assert "matroska" in doc["format"]["format_name"], doc


def test_detector_only_scan_skips_ocr(tmp_path, monkeypatch):
    """A faces/plates-only job must not load the OCR engine, spaCy, or
    PHI memory — detector-only scan. Loading them anyway cost seconds of
    startup and gigabytes of RAM for detectors that read no text."""
    called = []
    monkeypatch.setattr(openscrub, "make_ocr",
                        lambda *a, **k: called.append("ocr"))
    monkeypatch.setattr(openscrub, "NameDetector",
                        lambda *a, **k: called.append("namer"))
    src = make_video(str(tmp_path / "f.mp4"), [(300, "SSN 123-45-6789")])
    parser = openscrub.build_parser()
    args = parser.parse_args([src, "--engine", "tesseract",
                              "-o", str(tmp_path / "f_red.mp4"),
                              "--categories", "face", "--encoder", "x264"])
    args = openscrub._prep_args(args, parser)
    logs = []

    class CB(openscrub.Callbacks):
        def log(self, m):
            logs.append(m)
    state = openscrub.run_scan(args, CB())
    assert called == [], f"loaded needlessly: {called}"
    assert any("detector-only scan" in l for l in logs)
    assert any("names: skipped" in l for l in logs)
    # the SSN text in the video must NOT be detected (face-only job) and
    # the run must complete cleanly
    assert not [d for d in state["detections"] if d.category == "ssn"]


def test_face_model_registry_and_fallback(tmp_path):
    """The face-model registry loads with pinned hashes, and a face model
    that can't load falls back LOUDLY to the built-in YuNet — face
    detection must never silently disappear."""
    reg = openscrub.load_model_registry("face")
    assert any(m["id"] == "centerface" for m in reg)
    assert all(m.get("sha256") for m in reg), "face models must ship pinned"
    logs = []

    class CB(openscrub.Callbacks):
        def log(self, m):
            logs.append(m)
    fd = openscrub.FaceDetector(CB(), model_path=str(tmp_path / "nope.onnx"))
    assert fd.net is None, "missing model must not leave a broken detector"
    assert any("falling back" in l for l in logs)
    assert fd.yunet is not None or fd.haar is not None


def test_mosaic_and_ellipse_blur_shapes():
    """Mosaic must actually pixelate (it silently fell back to Gaussian
    blur before), and the ellipse shape must leave region corners intact."""
    rng = np.random.default_rng(1)
    base = np.clip(rng.normal(128, 40, (200, 200, 3)), 0, 255).astype(np.uint8)
    mos = base.copy()
    openscrub.blur_region(mos, 40, 40, 160, 160, "mosaic")
    blur = base.copy()
    openscrub.blur_region(blur, 40, 40, 160, 160, "blur")
    assert np.abs(mos.astype(int) - blur.astype(int)).mean() > 3, \
        "mosaic must differ from Gaussian blur"
    # mosaic tiles: many identical adjacent pixels
    inner = mos[60:140, 60:140]
    same = (inner[:, 1:] == inner[:, :-1]).all(axis=2).mean()
    assert same > 0.5, "mosaic should produce flat tiles"

    ell = base.copy()
    openscrub.blur_region(ell, 40, 40, 160, 160, "blur", shape="ellipse")
    assert (ell[41:47, 41:47] == base[41:47, 41:47]).all(), \
        "ellipse must leave the region corner untouched"
    c = 100
    assert np.abs(ell[c-3:c+3, c-3:c+3].astype(int)
                  - blur[c-3:c+3, c-3:c+3].astype(int)).mean() < 1, \
        "ellipse center must be blurred like the rect blur"


def test_face_only_steady_camera_no_bands_no_giant_boxes(tmp_path):
    """The boat-video failure: a steady-camera video with a moving person,
    face-only job. Scan-cadence merging used to union the face's positions
    into a body-sized box, and the scroll tracker painted blur bands along
    the frame edges. Now: faces are dense on every video, and tracking/
    bands are off without text categories."""
    # synthesizing a reliably-detectable face is flaky in CI; assert the
    # MODE decisions directly on a textured video with a moving object
    rng = np.random.default_rng(3)
    bg = np.clip(rng.normal(150, 18, (360, 640, 3)), 0, 255).astype(np.uint8)
    path = str(tmp_path / "steady.mp4")
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30,
                          (640, 360))
    sq = rng.integers(0, 255, (60, 60, 3)).astype(np.uint8)
    for i in range(90):
        fr = bg.copy()
        x = 50 + i * 4
        fr[150:210, x:x+60] = sq          # moving object (not a face)
        out.write(fr)
    out.release()
    parser = openscrub.build_parser()
    args = parser.parse_args([path, "--engine", "tesseract",
                              "-o", str(tmp_path / "s_red.mp4"),
                              "--categories", "face", "--encoder", "x264"])
    args = openscrub._prep_args(args, parser)
    logs = []

    class CB(openscrub.Callbacks):
        def log(self, m):
            logs.append(m)
    state = openscrub.run_scan(args, CB())
    assert any("scroll tracking + safety bands off" in l for l in logs)
    assert any("dense faces: detecting" in l for l in logs)
    assert all(b == (0.0, 0.0) for b in state["bands"]), \
        "face-only jobs must never emit safety bands"
    assert all(c == (0.0, 0.0) for c in state["cum"]), \
        "face-only jobs must never accumulate scroll offsets"


def test_face_model_unions_with_yunet(tmp_path):
    """An optional face model must AUGMENT the built-in YuNet, never
    replace it: the union guarantees a model upgrade can only add faces.
    (SCRFD through a squeezed 640x640 input detected FEWER faces than the
    built-in — the v1.0.30 'upgrade made it worse' report.)"""
    logs = []

    class CB(openscrub.Callbacks):
        def log(self, m):
            logs.append(m)
    # a fake ONNX path fails to load -> falls back, yunet still active
    fd = openscrub.FaceDetector(CB(), model_path=str(tmp_path / "x.onnx"))
    assert fd.yunet is not None or fd.haar is not None
    # when a real model IS loaded, yunet must be loaded alongside it — the
    # union log line is the contract
    class FakeNet:
        def getUnconnectedOutLayersNames(self):
            return ("a", "b", "c", "d")
    import unittest.mock as mock
    with mock.patch.object(openscrub.cv2.dnn, "readNet",
                           return_value=FakeNet()):
        p = tmp_path / "cf.onnx"
        p.write_bytes(b"x" * 20000)
        logs.clear()
        fd2 = openscrub.FaceDetector(CB(), model_path=str(p))
    assert fd2.net is not None and fd2.arch == "centerface"
    assert fd2.yunet is not None, "built-in must run alongside the model"
    assert any("UNIONED" in l for l in logs)

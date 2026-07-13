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

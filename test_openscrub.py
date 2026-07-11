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
    rx = openscrub.RE_MRN_DEFAULT
    import re
    r = re.compile(rx)
    assert r.search("1234567")
    assert r.search("MM0123456789")
    assert not r.search("4829173")          # 7 digits, wrong prefix
    assert not r.search("123456")           # too short


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

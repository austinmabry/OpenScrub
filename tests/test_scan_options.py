"""Regression tests for the actual multipart scan admission route."""
import hashlib
import io
import json
from pathlib import Path
import queue

import cv2
import numpy as np
import pytest

import openscrub_web as web


INVALID_OPTIONS = [
    pytest.param("", id="empty"),
    pytest.param("   ", id="whitespace"),
    pytest.param("{", id="malformed-object"),
    pytest.param("not-json", id="malformed-token"),
    pytest.param("[]", id="empty-array"),
    pytest.param("[1]", id="array"),
    pytest.param("null", id="null"),
    pytest.param('"ordinary"', id="string"),
    pytest.param('""', id="empty-string"),
    pytest.param("7", id="integer"),
    pytest.param("1" * 4301, id="integer-over-runtime-limit"),
    pytest.param("0", id="zero"),
    pytest.param("1.5", id="decimal"),
    pytest.param("true", id="true"),
    pytest.param("false", id="false"),
]
SUPPORTED_OPTIONS = {
    "categories": "email", "mode": "box", "sample_interval": 0.75,
    "pad": 3, "no_ner": True, "use_zones": False,
}
UNKNOWN_OPTIONS = dict(SUPPORTED_OPTIONS, ordinary_unknown_key={"value": [7, None]})


@pytest.fixture
def intake(tmp_path, monkeypatch):
    """Keep real parsing, job persistence and queueing within synthetic storage."""
    monkeypatch.setattr(web, "_data_root", lambda: str(tmp_path))
    for name in ("JOBS_DIR", "ZONES_PATH", "MODEL_SELECT_PATH",
                 "WEB_SETTINGS_PATH", "CUSTOM_CATS_PATH"):
        monkeypatch.setattr(web, name, str(tmp_path / Path(getattr(web, name)).name))
    jobs_dir = Path(web.JOBS_DIR)
    jobs_dir.mkdir()
    monkeypatch.setattr(web, "JOBS", {})
    monkeypatch.setattr(web, "WORK_Q", queue.Queue())
    monkeypatch.setattr(web, "WORKER", {"thread": None, "current": None})
    monkeypatch.setattr(web, "TOKEN", None)
    monkeypatch.setitem(web.app.config, "TESTING", False)
    monkeypatch.setitem(web.app.config, "PROPAGATE_EXCEPTIONS", None)
    monkeypatch.setitem(web.app.config, "MAX_CONTENT_LENGTH", 1024 * 1024)
    monkeypatch.setenv("OPENSCRUB_MEDIA_ROOT", str(tmp_path))
    monkeypatch.setenv("OPENSCRUB_NO_AUTOFETCH", "1")

    def forbidden(*args, **kwargs):
        pytest.fail("scan admission must not execute a worker or model inference")

    # main() is never called: accepted work stays in the real queue.
    for name in ("start_worker", "worker", "_run_job"):
        monkeypatch.setattr(web, name, forbidden)
    for name in ("run_pipeline", "_fetch_model"):
        monkeypatch.setattr(web.openscrub, name, forbidden)
    video = tmp_path / "synthetic.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"),
                             2.0, (64, 48))
    assert writer.isOpened()
    try:
        for level in (40, 100):
            writer.write(np.full((48, 64, 3), level, dtype=np.uint8))
    finally:
        writer.release()
    source = video.read_bytes()
    client = web.app.test_client()

    def post(kind, options):
        data = {} if options is None else {"options": options}
        if kind == "upload":
            data["video"] = (io.BytesIO(source), video.name)
        else:
            data["server_path"] = str(video)
        return client.post("/api/jobs", data=data, content_type="multipart/form-data")

    yield client, post, video, source, jobs_dir
    # RED failures may leave queued work. This fixture owns the temporary
    # storage and explicitly drains the queue without starting a worker.
    while not web.WORK_Q.empty():
        web.WORK_Q.get_nowait()
        web.WORK_Q.task_done()


@pytest.mark.parametrize("kind", ["upload", "server-path"])
@pytest.mark.parametrize("options", INVALID_OPTIONS)
def test_invalid_options_rejected_before_job_side_effects(intake, kind, options):
    client, post, video, source, jobs_dir = intake
    response = post(kind, options)
    print(kind, repr(options), response.status_code, "jobs", len(web.JOBS),
          "queued", web.WORK_Q.qsize(), "job directories", len(list(jobs_dir.iterdir())))
    assert response.status_code == 400
    assert response.is_json
    assert response.get_json() == {"error": "Scan options must be a JSON object."}
    assert web.JOBS == {}
    assert client.get("/api/jobs").get_json() == []
    assert web.WORK_Q.empty()
    assert web.WORK_Q.unfinished_tasks == 0
    assert list(jobs_dir.iterdir()) == []
    assert video.read_bytes() == source


@pytest.mark.parametrize("kind", ["upload", "server-path"])
@pytest.mark.parametrize("options,expected", [
    pytest.param(None, {}, id="omitted"),
    pytest.param("{}", {}, id="empty-object"),
    pytest.param(json.dumps(SUPPORTED_OPTIONS), SUPPORTED_OPTIONS, id="supported"),
    pytest.param(json.dumps(UNKNOWN_OPTIONS), UNKNOWN_OPTIONS, id="nested-unknown"),
])
def test_object_options_preserve_input_settings_and_cleanup(intake, kind, options, expected):
    client, post, video, source, jobs_dir = intake
    response = post(kind, options)
    assert response.status_code == 200
    jid, = response.get_json()["jobs"]
    assert set(web.JOBS) == {jid}
    job = web.JOBS[jid]
    jdir = Path(job["dir"])
    assert list(jobs_dir.iterdir()) == [jdir]
    assert job["phase"] == "queued"
    assert job["options"] == expected
    assert (jdir / "options.json").read_bytes() == json.dumps(expected).encode()
    assert hashlib.sha256(Path(job["video"]).read_bytes()).digest() == hashlib.sha256(source).digest()
    assert job["size"] == len(source)
    assert Path(job["video"]) == (jdir / "input.avi" if kind == "upload" else video)
    assert {p.name for p in jdir.iterdir()} == (
        {"options.json", "input.avi"} if kind == "upload" else {"options.json"})
    assert [j["id"] for j in client.get("/api/jobs").get_json()] == [jid]
    assert web.WORK_Q.qsize() == 1
    queued = web.WORK_Q.get_nowait()
    web.WORK_Q.task_done()
    assert queued == (jid, "scan", 1)

    args = web.build_args(job)
    assert (args.engine, args.device) == ("auto", "auto")
    assert (args.mode, args.sample_interval, args.pad, args.no_ner, args.categories) == (
        ("box", 0.75, 3, True, "email") if expected else
        ("blur", 0.5, 8, False,
         "name,dob,phone,ssn,email,address,card,apikey,ipaddr,plate,face,person"))
    cancel = client.post("/api/jobs/" + jid + "/cancel")
    assert cancel.status_code == 200
    assert cancel.get_json() == {"ok": True}
    assert job["phase"] == "cancelled"
    assert job["qgen"] == 2
    deletion = client.delete("/api/jobs/" + jid)
    assert deletion.status_code == 200
    assert deletion.get_json() == {"ok": True}
    assert web.JOBS == {}
    assert list(jobs_dir.iterdir()) == []
    assert web.WORK_Q.empty() and web.WORK_Q.unfinished_tasks == 0
    assert video.read_bytes() == source

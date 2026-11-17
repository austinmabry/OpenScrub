"""Real HDR cleanup with retained owners; all media and children are test-owned.

Strong process/pipe references deliberately prevent GC from masking ownership.
They are observation effects, not evidence of a permanent application leak.
"""
import ast
from contextlib import contextmanager
import inspect
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest

import openscrub

TAGS = {"-color_primaries": "bt2020", "-color_trc": "smpte2084",
        "-colorspace": "bt2020nc"}


def _command(args):
    return subprocess.run(args, check=True, capture_output=True, timeout=30).stdout


def _decode(path):
    return _command(["ffmpeg", "-v", "error", "-i", str(path),
                     "-f", "rawvideo", "-pix_fmt", "yuv420p10le", "pipe:1"])


def _metadata(path):
    return json.loads(_command([
        "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,pix_fmt,width,height,nb_read_frames,"
        "color_transfer,color_primaries,color_space", "-of", "json", str(path)
    ]))["streams"][0]


@pytest.fixture
def hdr_source(tmp_path):
    if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
        pytest.skip("ffmpeg/ffprobe required")
    encoders = _command(["ffmpeg", "-hide_banner", "-encoders"])
    if b"libx265" not in encoders:
        pytest.skip("libx265 required for real ten-bit HDR control")
    # Three exact native 10-bit Y/U/V frames, without inference or user media.
    raw = np.concatenate([np.full(4096, 900, np.uint16),
                          np.full(1024, 800, np.uint16),
                          np.full(1024, 300, np.uint16)]).tobytes() * 3
    source_raw = tmp_path / "source.yuv"
    source_raw.write_bytes(raw)
    src = tmp_path / "source.mp4"
    colors = [a for item in TAGS.items() for a in item]
    _command(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f",
              "rawvideo", "-pix_fmt", "yuv420p10le", "-s", "64x64", "-r", "3",
              *colors, "-i", str(source_raw), "-c:v", "libx265", "-x265-params",
              "lossless=1:pools=1:colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc",
              *colors, str(src)])
    assert _decode(src) == raw
    return src, raw


@contextmanager
def _observe_children(monkeypatch):
    """Record real waits and retain owners; never replace pipe IO or commands."""
    children, error_files = [], []
    popen = subprocess.Popen
    temporary_file = openscrub.tempfile.TemporaryFile

    class ObservedPopen(popen):
        def __init__(self, *args, **kwargs):
            self.wait_returns = []
            super().__init__(*args, **kwargs)
            children.append(self)

        def wait(self, *args, **kwargs):
            rc = super().wait(*args, **kwargs)
            self.wait_returns.append(rc)
            return rc

    def observe_file(*args, **kwargs):
        file = temporary_file(*args, **kwargs)
        error_files.append(file)
        return file

    with monkeypatch.context() as patch:
        patch.setattr(openscrub.subprocess, "Popen", ObservedPopen)
        patch.setattr(openscrub.tempfile, "TemporaryFile", observe_file)
        try:
            yield children
        finally:
            # Unconditional harness containment, including when a RED assertion
            # fails. These closes do not count as production cleanup evidence.
            for child in children:
                if child.poll() is None:
                    child.terminate()
                for name in ("stdin", "stdout", "stderr"):
                    stream = getattr(child, name)
                    if stream is not None and not stream.closed:
                        stream.close()
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=5)
            for file in error_files:
                file.close()  # stderr lifetime belongs to separate lifecycle work


def _args_state(source, output):
    args = SimpleNamespace(video=str(source), output=str(output), preview=False,
                           hdr_source=str(source), hdr_encoder="libx265",
                           hdr_tags=TAGS, pad=0, mode="box", report=None)
    state = {"detections": [], "cum": [(0, 0)] * 3, "bands": [(0, 0)] * 3,
             "fps": 3.0, "stats": {"synthetic": True}}
    return args, state


def _render_children(children):
    dec, = [p for p in children if "pipe:1" in p.args]
    enc, = [p for p in children if "pipe:0" in p.args]
    return dec, enc


def _assert_clean(children):
    dec, enc = _render_children(children)
    # Check reaping independently of Popen.returncode, before harness cleanup.
    for child in (dec, enc):
        assert child.wait_returns and child.returncode is not None
        with pytest.raises(ChildProcessError):
            os.waitpid(child.pid, os.WNOHANG)
    print("HDR_CHILDREN " + json.dumps([
        {"pid": p.pid, "wait_returns": p.wait_returns,
         "returncode": p.returncode, "pipe_closed": stream.closed}
        for p, stream in ((dec, dec.stdout), (enc, enc.stdin))]))
    assert enc.stdin.closed
    assert dec.stdout.closed


@pytest.mark.parametrize("failure", ["cancel", "encoder-write"])
def test_hdr_partial_cleanup_closes_pipes_with_retained_traceback(
        failure, hdr_source, tmp_path, monkeypatch):
    """Missing decoder close fails while exception/traceback and owners live."""
    if os.name != "posix":
        pytest.skip("child-reaping observation requires POSIX waitpid")
    if failure == "encoder-write" and not all(
            hasattr(os, name) for name in ("waitid", "WNOWAIT", "WEXITED")):
        pytest.skip("non-reaping exit observation requires waitid/WNOWAIT")
    source, _ = hdr_source
    args, state = _args_state(source, tmp_path / "partial.mp4")
    held_errors = []
    checkpoints = []
    with _observe_children(monkeypatch) as children:
        class Callbacks(openscrub.Callbacks):
            def log(self, message):
                pass

            def cancelled(self):
                checkpoints.append(0)
                if failure == "encoder-write":
                    _, enc = _render_children(children)
                    # Kill only our actual encoder, then observe exit WITHOUT
                    # reaping. The real first stdin.write raises BrokenPipeError.
                    os.kill(enc.pid, signal.SIGKILL)
                    deadline = time.monotonic() + 5
                    while os.waitid(os.P_PID, enc.pid,
                                    os.WEXITED | os.WNOHANG | os.WNOWAIT) is None:
                        if time.monotonic() > deadline:
                            raise AssertionError("test encoder did not exit")
                        time.sleep(0.001)
                return failure == "cancel"

        try:
            openscrub.run_render(args, state, Callbacks())
        except Exception as error:
            held_errors.append(error)
        assert len(held_errors) == 1
        error = held_errors[0]
        assert checkpoints == [0]
        frames = []
        tb = error.__traceback__
        while tb is not None:
            frames.append(tb.tb_frame.f_code)
            tb = tb.tb_next
        assert openscrub.run_render.__code__ in frames
        assert openscrub.render_hdr.__code__ in frames
        if failure == "cancel":
            assert type(error) is openscrub.PipelineCancelled
        else:
            assert type(error) is RuntimeError
            assert "HDR encode" in str(error) and "-9" in str(error)
            assert isinstance(error.__context__, BrokenPipeError)
            _, enc = _render_children(children)
            assert enc.returncode == -signal.SIGKILL
        assert not Path(args.output).exists()
        _assert_clean(children)  # retained traceback and strong owners still live


def test_hdr_success_keeps_source_bytes_frames_and_tags(
        hdr_source, tmp_path, monkeypatch):
    """Ordinary success closes both pipes and preserves the real HDR control."""
    source, raw = hdr_source
    output = tmp_path / "success.mp4"
    args, state = _args_state(source, output)
    captured = []
    lines, start = inspect.getsourcelines(openscrub.render_hdr)
    tree = ast.parse("".join(lines))
    write_line, = [start + n.lineno - 1 for n in ast.walk(tree)
                   if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
                   and ast.unparse(n.value.func) == "enc.stdin.write"]

    def trace(frame, event, arg):
        if frame.f_code is openscrub.render_hdr.__code__:
            if event == "line" and frame.f_lineno == write_line:
                captured.append(frame.f_locals["buf"].tobytes())
            return trace
        return None

    previous_trace = sys.gettrace()
    with _observe_children(monkeypatch) as children:
        try:
            sys.settrace(trace)  # read-only pre-encoder observation, no IO wrapper
            result = openscrub.run_render(args, state, openscrub.Callbacks())
        finally:
            sys.settrace(previous_trace)
        assert result == {"synthetic": True, "output": str(output)}
        if os.name == "posix":
            _assert_clean(children)
        assert b"".join(captured) == raw
        assert len(captured) == 3
    metadata = _metadata(output)
    assert metadata == {"codec_name": "hevc", "width": 64, "height": 64,
                        "pix_fmt": "yuv420p10le", "color_space": "bt2020nc",
                        "color_transfer": "smpte2084", "color_primaries": "bt2020",
                        "nb_read_frames": "3"}
    decoded = _decode(output)
    assert len(decoded) == len(raw)  # lossy output is not source-bit-identical

#!/usr/bin/env python3
"""Docker build helper: pre-fetch the auto-downloaded detection models
into the image so first run works offline.

Standalone ON PURPOSE — stdlib only, no openscrub import. The whole point
of this script is Docker layer caching: it is COPY'd and RUN in its own
layer BEFORE the app code is copied, so a release that only changes code
does not invalidate the ~60MB model layer (previously the prefetch lived
inside the app layer and every release re-downloaded every model). The
layer re-builds only when THIS FILE changes — i.e. when a model URL/hash
actually changes.

The (url, sha256, filename) table below MUST stay identical to the
constants in openscrub.py — test_docker_prefetch_matches_engine_pins
fails the suite when they drift.

Usage:  python prefetch_models.py <key> [<key> ...]     (or no args = all)
"""
import hashlib
import os
import sys
import time
import urllib.request

DEST = "/root/.openscrub/models"

# key -> (url, pinned sha256, filename the engine looks for)
MODELS = {
    "yunet": (
        "https://media.githubusercontent.com/media/opencv/opencv_zoo/"
        "main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
        "face_detection_yunet_2023mar.onnx"),
    "sface": (
        "https://media.githubusercontent.com/media/opencv/opencv_zoo/"
        "main/models/face_recognition_sface/"
        "face_recognition_sface_2021dec.onnx",
        "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
        "face_recognition_sface_2021dec.onnx"),
    "vittrack": (
        "https://media.githubusercontent.com/media/opencv/opencv_zoo/"
        "main/models/object_tracking_vittrack/"
        "object_tracking_vittrack_2023sep.onnx",
        "2990f0b7cd44d92afa48cd97db6de7be113fc1d9594fddb74e2725c10478e91d",
        "object_tracking_vittrack_2023sep.onnx"),
    "ppdet": (
        "https://huggingface.co/PaddlePaddle/PP-OCRv5_mobile_det_onnx/"
        "resolve/main/inference.onnx",
        "a431985659dc921974177a95adcfbb90fd9e51989a5e04d70d0b75f597b6e61d",
        "text_detection_ppocrv5_mobile.onnx"),
    "pprec": (
        "https://huggingface.co/PaddlePaddle/PP-OCRv5_mobile_rec_onnx/"
        "resolve/main/inference.onnx",
        "da72dc72ca4dc220df0dfde68c1dedc31c58d3e76a25871122e5056227d50092",
        "text_recognition_ppocrv5_mobile.onnx"),
    "pprec_yml": (
        "https://huggingface.co/PaddlePaddle/PP-OCRv5_mobile_rec_onnx/"
        "resolve/main/inference.yml",
        "5dfeb2777f6d0db8177d8128a8acfcf6e6276dc4ac73ea3bf0dc06d6a5e85d8e",
        "text_recognition_ppocrv5_mobile.yml"),
}


def fetch(url, dest, sha256, tries=8, delay=15):
    """Download + sha256-verify with retries (the opencv_zoo LFS media
    host throws transient quota 404s — same policy as _fetch_model)."""
    for attempt in range(1, tries + 1):
        try:
            tmp = dest + ".part"
            with urllib.request.urlopen(url, timeout=120) as r, \
                    open(tmp, "wb") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            digest = hashlib.sha256(open(tmp, "rb").read()).hexdigest()
            if digest != sha256:
                os.remove(tmp)
                raise RuntimeError(
                    "sha256 mismatch for %s: got %s want %s"
                    % (url, digest, sha256))
            os.replace(tmp, dest)
            print("  ok  %s (%d bytes)" % (dest, os.path.getsize(dest)),
                  flush=True)
            return
        except Exception as e:
            print("  attempt %d/%d failed: %s" % (attempt, tries, e),
                  flush=True)
            if attempt == tries:
                raise
            time.sleep(delay)


def main():
    keys = sys.argv[1:] or list(MODELS)
    unknown = [k for k in keys if k not in MODELS]
    if unknown:
        raise SystemExit("unknown model key(s): %s" % ", ".join(unknown))
    os.makedirs(DEST, exist_ok=True)
    for k in keys:
        url, sha, name = MODELS[k]
        print("fetching %s -> %s" % (k, name), flush=True)
        fetch(url, os.path.join(DEST, name), sha)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""fetch_plate_models.py — populate plate_models.json with verified values.

Run this ONCE on a machine with internet (your dev box, not the clinic server):

    pip install open-image-models
    python fetch_plate_models.py

For each curated model it:
  1. has the open-image-models package download the ONNX (to its local cache),
  2. copies it into ./models/<registry-id>.onnx so OpenScrub can use it now,
  3. computes the SHA-256,
  4. prints the exact "download_url" + "sha256" values to paste into
     plate_models.json (and optionally rewrites the registry in place).

After this, the web UI's plate-model picker shows Download buttons that fetch
and verify against these hashes on any other machine.
"""

import hashlib
import json
import os
import shutil
import sys

# registry-id -> open-image-models model name
MODELS = {
    "oim-yolov9-t-640": "yolo-v9-t-640-license-plate-end2end",
    "oim-yolov9-s-608": "yolo-v9-s-608-license-plate-end2end",
    "oim-yolov9-t-256": "yolo-v9-t-256-license-plate-end2end",
}

HERE = os.path.dirname(os.path.abspath(__file__))
REGISTRY = os.path.join(HERE, "plate_models.json")
MODELS_DIR = os.path.join(HERE, "models")


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_hub_url(name):
    """Best-effort: pull the download URL from the package's model hub map."""
    try:
        from open_image_models.detection.core import hub  # noqa
        for attr in ("PLATE_DETECTION_MODELS", "MODEL_URLS", "AVAILABLE_ONNX_MODELS"):
            m = getattr(hub, attr, None)
            if isinstance(m, dict) and name in m:
                v = m[name]
                return v if isinstance(v, str) else getattr(v, "url", "")
    except Exception:
        pass
    return ""


def main():
    try:
        from open_image_models import LicensePlateDetector
    except ImportError:
        sys.exit("open-image-models is not installed.\n"
                 "    pip install open-image-models\nthen re-run.")

    os.makedirs(MODELS_DIR, exist_ok=True)
    results = {}
    for rid, name in MODELS.items():
        print(f"\n=== {rid}  ({name}) ===")
        # 1. trigger the package's own download (it caches locally)
        det = LicensePlateDetector(detection_model=name)
        # locate the cached onnx: the detector keeps the path (attr name has
        # varied across versions, so check the common ones, then fall back to
        # scanning the package cache dir)
        cached = None
        for attr in ("model_path", "onnx_path", "_model_path"):
            cand = getattr(det, attr, None)
            if cand and os.path.exists(str(cand)):
                cached = str(cand)
                break
        if not cached:
            cache_root = os.path.expanduser("~/.cache")
            for root, _dirs, files in os.walk(cache_root):
                for f in files:
                    if f.endswith(".onnx") and name in f:
                        cached = os.path.join(root, f)
                        break
                if cached:
                    break
        if not cached:
            print("  !! could not locate the cached ONNX for this model.")
            print("     Find it manually (search ~/.cache for *.onnx) and copy")
            print(f"     it to {MODELS_DIR}/{rid}.onnx, then run "
                  f"'sha256sum' on it.")
            continue
        # 2. copy into OpenScrub's models dir under the registry id
        dest = os.path.join(MODELS_DIR, f"{rid}.onnx")
        shutil.copy2(cached, dest)
        digest = sha256_of(dest)
        url = find_hub_url(name)
        results[rid] = {"sha256": digest, "download_url": url,
                        "local": dest, "size_mb": os.path.getsize(dest) / 1e6}
        print(f"  installed -> {dest}  ({results[rid]['size_mb']:.1f} MB)")
        print(f"  sha256:       {digest}")
        print(f"  download_url: {url or '(not exposed by package - see below)'}")

    if not results:
        sys.exit("\nNo models fetched.")

    # 3. offer to rewrite the registry with the verified values
    print("\n----------------------------------------------------------------")
    if os.path.exists(REGISTRY):
        reg = json.load(open(REGISTRY))
        changed = 0
        for m in reg.get("models", []):
            r = results.get(m.get("id"))
            if r:
                m["sha256"] = r["sha256"]
                if r["download_url"]:
                    m["download_url"] = r["download_url"]
                changed += 1
        ans = input(f"Write {changed} verified entr(ies) into plate_models.json? [y/N] ")
        if ans.strip().lower() == "y":
            json.dump(reg, open(REGISTRY, "w"), indent=2)
            print("plate_models.json updated.")
        else:
            print("Registry NOT modified; paste the values above manually.")
    else:
        print("plate_models.json not found next to this script; paste the "
              "values above into your registry manually.")

    print("\nNOTE: if download_url was '(not exposed by package)', the models "
          "are still installed locally and usable NOW. For the picker's "
          "Download button to work on OTHER machines, host the .onnx files "
          "somewhere you control (e.g. a GitHub release on OpenScrub) and put "
          "those URLs in the registry — you already have the correct sha256.")


if __name__ == "__main__":
    main()

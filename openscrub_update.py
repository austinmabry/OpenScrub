#!/usr/bin/env python3
"""openscrub-update — bring an existing OpenScrub install up to date.

    openscrub-update            interactive: shows versions, asks, updates
    openscrub-update --check    report current vs latest, change nothing
    openscrub-update --yes      update without prompting

Two install modes are handled automatically:

* pip install (the module lives in site-packages): updates via
  `pip install --upgrade OpenScrub==<latest>`.
* folder / source deploy (a directory of .py files): downloads the latest
  sdist from PyPI, verifies its SHA-256 against the hash PyPI publishes
  (mismatch = abort and delete — fail closed), then replaces only the
  files the release ships. Everything local survives an update:
  openscrub_jobs/ (PHI!), certs/, models/, zones.json, allowlist.txt,
  and locally pinned plate-model hashes in plate_models.json. Replaced
  files are first copied to backups/pre-update-<version>/.

A git checkout is deliberately NOT updated this way — `git pull` is the
right tool there, so we say so and stop (override with --force).

The web UI uses this module for its update notice and one-click update;
after any update the server must be restarted to run the new code.
"""

import argparse
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request

PYPI_JSON = "https://pypi.org/pypi/OpenScrub/json"
HERE = os.path.dirname(os.path.abspath(__file__))

# Local state that an update must never touch. openscrub_jobs contains
# real PHI on working deployments; certs hold private keys.
PRESERVE = {
    "openscrub_jobs", "phi_blur_jobs", "jobs", "certs", "models",
    "backups", "zones.json", "allowlist.txt", "CLAUDE.local.md",
}


def current_version():
    try:
        import openscrub
        return openscrub.VERSION
    except Exception:
        return None


def parse_ver(v):
    """'1.0.10' -> (1, 0, 10). Non-numeric parts compare as 0."""
    out = []
    for part in str(v).split("."):
        digits = "".join(ch for ch in part if ch.isdigit())
        out.append(int(digits) if digits else 0)
    return tuple(out)


def is_newer(candidate, current):
    return parse_ver(candidate) > parse_ver(current)


def get_latest(timeout=10):
    """-> {'version': str, 'sdist_url': str, 'sha256': str} from PyPI."""
    with urllib.request.urlopen(PYPI_JSON, timeout=timeout) as r:
        doc = json.load(r)
    ver = doc["info"]["version"]
    for f in doc.get("urls", []):
        if f.get("packagetype") == "sdist":
            return {"version": ver, "sdist_url": f["url"],
                    "sha256": f.get("digests", {}).get("sha256", "")}
    raise RuntimeError("PyPI lists %s but offers no sdist" % ver)


def pip_installed():
    p = HERE.replace("\\", "/")
    return "site-packages" in p or "dist-packages" in p


def merge_registry_pins(old_models, new_models):
    """Carry locally pinned TOFU sha256 hashes into a new registry.

    A pin is kept only when the model id AND download_url are unchanged —
    if the URL moved, the old hash is meaningless and trust must be
    re-established on the next download. Never overwrites a hash the new
    registry ships explicitly.
    """
    old_by_id = {m.get("id"): m for m in old_models}
    carried = 0
    for m in new_models:
        o = old_by_id.get(m.get("id"))
        if (o and not m.get("sha256") and o.get("sha256")
                and o.get("download_url") == m.get("download_url")):
            m["sha256"] = o["sha256"]
            carried += 1
    return carried


def _merge_registry_file(local_path, incoming_path, log):
    try:
        with open(local_path, encoding="utf-8") as f:
            old = json.load(f)
        with open(incoming_path, encoding="utf-8") as f:
            new = json.load(f)
        n = merge_registry_pins(old.get("models", []), new.get("models", []))
        if n:
            log("  plate_models.json: carried %d locally pinned hash(es) "
                "forward" % n)
        with open(incoming_path, "w", encoding="utf-8") as f:
            json.dump(new, f, indent=2)
            f.write("\n")
    except Exception as e:  # malformed local file: ship the new one as-is
        log("  plate_models.json: could not merge local pins (%s) — "
            "using released registry; hashes re-pin on next download" % e)


def update_pip(latest, log):
    log("updating via pip …")
    cmd = [sys.executable, "-m", "pip", "install", "--upgrade",
           "OpenScrub==%s" % latest["version"]]
    log("  $ " + " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    for line in (r.stdout or "").splitlines()[-6:]:
        log("  " + line)
    if r.returncode != 0:
        for line in (r.stderr or "").splitlines()[-8:]:
            log("  " + line)
        raise RuntimeError("pip exited with %d" % r.returncode)
    log("pip update complete.")


def update_source(latest, log):
    log("updating folder install at %s …" % HERE)
    # download the sdist and verify it against the hash PyPI publishes
    log("  downloading %s" % latest["sdist_url"])
    with urllib.request.urlopen(latest["sdist_url"], timeout=60) as r:
        blob = r.read()
    digest = hashlib.sha256(blob).hexdigest()
    if not latest["sha256"] or digest != latest["sha256"]:
        raise RuntimeError(
            "sdist SHA-256 mismatch (expected %s, got %s) — aborting, "
            "nothing was changed" % (latest["sha256"] or "<none>", digest))
    log("  sha256 verified (%s…)" % digest[:16])

    stamp = time.strftime("%Y%m%d-%H%M%S")
    bak = os.path.join(HERE, "backups",
                       "pre-update-%s-%s" % (current_version(), stamp))
    staged = 0
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
        members = [m for m in tar.getmembers() if m.isfile()]
        for m in members:
            # strip the leading 'OpenScrub-x.y.z/' directory
            rel = m.name.split("/", 1)[1] if "/" in m.name else m.name
            if not rel or rel.startswith("."):
                continue
            top = rel.split("/", 1)[0]
            if top in PRESERVE or os.path.basename(rel) == "PKG-INFO":
                continue
            data = tar.extractfile(m).read()
            dest = os.path.join(HERE, *rel.split("/"))
            tmp = dest + ".new"
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with open(tmp, "wb") as f:
                f.write(data)
            if rel in ("plate_models.json", "face_models.json") \
                    and os.path.exists(dest):
                _merge_registry_file(dest, tmp, log)
            if os.path.exists(dest):
                os.makedirs(os.path.join(bak, os.path.dirname(rel))
                            if os.path.dirname(rel) else bak, exist_ok=True)
                shutil.copy2(dest, os.path.join(bak, *rel.split("/")))
            os.replace(tmp, dest)
            staged += 1
    log("  %d file(s) updated; previous versions saved to %s" %
        (staged, os.path.relpath(bak, HERE)))
    log("folder update complete.")


def run_update(log=print, assume_yes=True):
    """Shared driver used by the CLI and the web UI. Returns new version."""
    cur = current_version() or "unknown"
    latest = get_latest()
    if not is_newer(latest["version"], cur):
        log("already up to date (v%s)." % cur)
        return cur
    log("update available: v%s -> v%s" % (cur, latest["version"]))
    if pip_installed():
        update_pip(latest, log)
    else:
        update_source(latest, log)
    log("RESTART OpenScrub (CLI or web server) to run the new version.")
    return latest["version"]


def main():
    ap = argparse.ArgumentParser(
        description="Update OpenScrub to the latest PyPI release.")
    ap.add_argument("--check", action="store_true",
                    help="report current vs latest, change nothing")
    ap.add_argument("--yes", action="store_true", help="no prompts")
    ap.add_argument("--force", action="store_true",
                    help="update a git checkout anyway (git pull is better)")
    a = ap.parse_args()

    cur = current_version() or "unknown"
    print("current version: v%s" % cur)
    try:
        latest = get_latest()
    except Exception as e:
        print("could not reach PyPI (%s) — try again later." % e)
        sys.exit(1)
    print("latest release:  v%s" % latest["version"])

    if not is_newer(latest["version"], cur):
        print("already up to date.")
        return
    if a.check:
        print("update available. run openscrub-update to install it.")
        return
    if os.path.isdir(os.path.join(HERE, ".git")) and not pip_installed() \
            and not a.force:
        print("this looks like a git checkout — use `git pull` instead "
              "(or re-run with --force).")
        sys.exit(1)
    if not a.yes:
        try:
            ok = input("update to v%s now? [y/N] " % latest["version"])
        except EOFError:
            ok = ""
        if ok.strip().lower() not in ("y", "yes"):
            print("nothing changed.")
            return
    run_update(log=print, assume_yes=True)


if __name__ == "__main__":
    main()

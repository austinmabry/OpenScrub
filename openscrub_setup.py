#!/usr/bin/env python3
"""openscrub-setup — one command to finish a fresh install.

`pip install OpenScrub` covers every Python dependency, but two system
tools live outside pip's world: Tesseract (OCR for all text categories)
and FFmpeg (audio passthrough, H.264 output, VFR normalization). This
command detects what's missing and installs it for you — with your
consent — using winget on Windows or apt on Debian/Ubuntu.

    openscrub-setup            interactive: shows status, asks before installing
    openscrub-setup --check    report only, change nothing
    openscrub-setup --yes      install anything missing without prompting
    openscrub-setup --with-plates   also download the recommended
                                    license-plate model (SHA-256 verified)

Why these aren't bundled into the pip package: they're standalone
programs, not Python libraries — bundling would exceed PyPI size limits,
freeze security updates, and (for FFmpeg's H.264 encoder) create a GPL
licensing conflict with OpenScrub's Apache-2.0 license. Installing them
at the system level keeps them patched by your package manager.
"""

import argparse
import os
import shutil
import subprocess
import sys


GREEN, YELLOW, RED, END = "\033[92m", "\033[93m", "\033[91m", "\033[0m"
if os.name == "nt":
    os.system("")  # enable ANSI colors on Windows 10+ consoles


def _ok(msg):   print(f"  {GREEN}[ok]{END}      {msg}")
def _miss(msg): print(f"  {RED}[missing]{END} {msg}")
def _note(msg): print(f"  {YELLOW}[note]{END}    {msg}")


def find_tesseract():
    """Same search the engine performs: PATH, then known Windows installs."""
    p = shutil.which("tesseract")
    if p:
        return p
    if os.name == "nt":
        try:
            from openscrub import WINDOWS_TESSERACT_PATHS
        except Exception:
            WINDOWS_TESSERACT_PATHS = [
                r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
                os.path.expandvars(
                    r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
            ]
        for c in WINDOWS_TESSERACT_PATHS:
            if os.path.exists(c):
                return c
    return None


def find_ffmpeg():
    return shutil.which("ffmpeg")


def _confirm(question, assume_yes):
    if assume_yes:
        return True
    try:
        return input(f"{question} [y/N] ").strip().lower() == "y"
    except EOFError:
        return False


def _run(cmd):
    print(f"    $ {' '.join(cmd)}")
    try:
        return subprocess.run(cmd).returncode == 0
    except FileNotFoundError:
        return False


def install_windows(pkg_id, assume_yes):
    if not shutil.which("winget"):
        _note("winget not found (it ships with Windows 10 21H2+ / 11).")
        _note(f"Install manually, or run: winget install -e --id {pkg_id}")
        return False
    if not _confirm(f"Install {pkg_id} via winget now?", assume_yes):
        _note(f"Skipped. Manual command: winget install -e --id {pkg_id}")
        return False
    return _run(["winget", "install", "-e", "--id", pkg_id,
                 "--accept-source-agreements", "--accept-package-agreements"])


def install_apt(packages, assume_yes):
    if not shutil.which("apt-get"):
        _note("No apt available — install via your distro's package manager: "
              + " ".join(packages))
        return False
    if not _confirm(f"Run 'sudo apt-get install -y {' '.join(packages)}' now?",
                    assume_yes):
        _note("Skipped. Manual command: sudo apt-get install -y "
              + " ".join(packages))
        return False
    return _run(["sudo", "apt-get", "install", "-y"] + packages)


def setup_plates(assume_yes, check_only):
    try:
        import openscrub
    except ImportError:
        _miss("openscrub package not importable — is it installed?")
        return
    models = openscrub.load_plate_registry()
    here = os.path.dirname(os.path.abspath(openscrub.__file__))
    installed = [m["id"] for m in models if os.path.exists(
        os.path.join(here, "models", f"{m['id']}.onnx"))]
    if installed:
        _ok(f"license-plate model installed: {installed[0]}")
        return
    rec = next((m for m in models if m.get("recommended")), None)
    if rec is None:
        _note("no recommended plate model in registry; see PLATES.md")
        return
    _miss(f"license-plate model (optional) — recommended: {rec['label']} "
          f"[{rec.get('license')}]")
    if check_only:
        return
    if not _confirm("Download it now (SHA-256 verified)?", assume_yes):
        _note("Skipped. The web UI's plate panel can download it later.")
        return
    try:
        path = openscrub.download_plate_model(rec)
        _ok(f"plate model saved: {path}")
    except Exception as e:
        _miss(f"plate model download failed: {e}")


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="openscrub-setup",
        description="Detect and install OpenScrub's system prerequisites "
                    "(Tesseract OCR, FFmpeg).")
    ap.add_argument("--check", action="store_true",
                    help="report status only; install nothing")
    ap.add_argument("--yes", "-y", action="store_true",
                    help="install anything missing without prompting")
    ap.add_argument("--with-plates", action="store_true",
                    help="also offer the optional license-plate model")
    args = ap.parse_args(argv)

    print("OpenScrub setup — system prerequisites")
    print("=" * 46)

    changed = False

    # Python-side sanity (pip already handled these; verify imports work)
    try:
        import cv2, flask  # noqa: F401
        _ok("python dependencies (opencv, flask, ...)")
    except ImportError as e:
        _miss(f"python dependency broken: {e} — try: pip install --force-reinstall OpenScrub")

    # Tesseract
    t = find_tesseract()
    if t:
        _ok(f"Tesseract OCR: {t}")
    else:
        _miss("Tesseract OCR — required for ALL text categories "
              "(names, SSNs, addresses, ...)")
        if not args.check:
            if os.name == "nt":
                changed |= install_windows("UB-Mannheim.TesseractOCR", args.yes)
            else:
                changed |= install_apt(["tesseract-ocr"], args.yes)

    # FFmpeg
    f = find_ffmpeg()
    if f:
        _ok(f"FFmpeg: {f}")
    else:
        _miss("FFmpeg — needed for audio passthrough, H.264 output, and "
              "screen-recording (VFR) normalization")
        if not args.check:
            if os.name == "nt":
                changed |= install_windows("Gyan.FFmpeg", args.yes)
            else:
                changed |= install_apt(["ffmpeg"], args.yes)

    # Optional: spaCy NER model
    try:
        import spacy
        try:
            spacy.load("en_core_web_sm")
            _ok("spaCy NER model (en_core_web_sm)")
        except OSError:
            _miss("spaCy is installed but its model isn't")
            if not args.check and _confirm(
                    "Download en_core_web_sm now?", args.yes):
                _run([sys.executable, "-m", "spacy", "download",
                      "en_core_web_sm"])
    except ImportError:
        _note('name detection uses heuristics; for better accuracy: '
              'pip install "OpenScrub[ner]" then re-run openscrub-setup')

    # Optional: plate model
    if args.with_plates:
        setup_plates(args.yes, args.check)

    print("=" * 46)
    if changed and os.name == "nt":
        _note("winget installs update PATH for NEW terminals only:")
        _note("close this window, open a fresh one, then run: openscrub-web")
    else:
        print("Done. Start the web app with:  openscrub-web")
    return 0


if __name__ == "__main__":
    sys.exit(main())

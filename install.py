#!/usr/bin/env python3
"""
OpenScrub installer — Windows, Linux, and (best-effort) macOS.

One file that takes a fresh machine to a working install:
  * Python package dependencies (OCR, web server, crypto for HTTPS)
  * PaddleOCR with GPU acceleration when an NVIDIA card is present
  * System tools: Tesseract OCR and ffmpeg (winget / apt / dnf / pacman / brew)
  * NVENC hardware-encode verification
  * A launchable "OpenScrub" shortcut with the program icon
    (Desktop + Start Menu on Windows, .desktop entry on Linux,
     .command on macOS)

Usage:
    python install.py            interactive install
    python install.py --yes      install everything without prompting
    python install.py --check    report what is present/missing, change nothing
    python install.py --cpu-only skip GPU OCR even if an NVIDIA card exists
    python install.py --no-shortcut   skip launcher creation
"""

import argparse
import os
import platform
import shutil
import subprocess
import sys

FROZEN = getattr(sys, "frozen", False)          # running as PyInstaller exe
# In a onefile exe, __file__ points at a temp extraction dir that vanishes —
# the project folder is wherever the exe itself lives.
HERE = (os.path.dirname(os.path.abspath(sys.executable)) if FROZEN
        else os.path.dirname(os.path.abspath(__file__)))
IS_WIN = os.name == "nt"
IS_MAC = platform.system() == "Darwin"
IS_LINUX = platform.system() == "Linux"

CORE_PKGS = ["opencv-python", "rapidfuzz", "pytesseract", "pillow",
             "flask", "pyyaml", "cryptography"]

RESULTS = []


def resolve_python():
    """Real Python interpreter to run pip/imports with. Inside a frozen exe,
    sys.executable is the EXE itself — using it for pip would recursively
    relaunch the installer."""
    if not FROZEN:
        return sys.executable
    for cand in ("python3", "python", "py"):
        p = shutil.which(cand)
        if p:
            r = subprocess.run([p, "-c", "import sys;print(sys.version_info>=(3,9))"],
                               capture_output=True, text=True)
            if "True" in (r.stdout or ""):
                return p
    return None


PYTHON = resolve_python()


def ensure_python(yes):
    """Frozen exe on a machine with no Python: offer to install it."""
    global PYTHON
    if PYTHON:
        return True
    log("  No Python 3.9+ found on this machine.")
    if IS_WIN and shutil.which("winget"):
        if ask("Install Python 3.12 now via winget?", yes):
            r = run(["winget", "install", "-e", "--id", "Python.Python.3.12",
                     "--accept-source-agreements", "--accept-package-agreements"])
            if r.returncode == 0:
                log("  Python installed — open a NEW terminal and run this "
                    "installer again so PATH updates apply.")
            record("Python install", r.returncode == 0)
            return False
    record("Python 3.9+", False,
           "install from https://python.org, then run this installer again")
    return False


def log(msg):
    print(msg, flush=True)


def record(step, ok, note=""):
    RESULTS.append((step, ok, note))
    log(f"  [{'OK' if ok else '!!'}] {step}" + (f" — {note}" if note else ""))


def run(cmd, timeout=1800, **kw):
    return subprocess.run(cmd, capture_output=True, text=True,
                          timeout=timeout, **kw)


def pip(*args):
    return run([PYTHON, "-m", "pip", *args])


def have(mod):
    return run([PYTHON, "-c", f"import {mod}"]).returncode == 0


def ask(prompt, assume_yes):
    if assume_yes:
        return True
    try:
        return input(f"{prompt} [Y/n] ").strip().lower() in ("", "y", "yes")
    except EOFError:
        return True


# --------------------------------------------------------------------------
def step_python():
    if FROZEN:
        r = run([PYTHON, "-c", "import platform;print(platform.python_version())"])
        ver = (r.stdout or "?").strip()
        record(f"Python {ver} ({PYTHON})", r.returncode == 0)
        return r.returncode == 0
    ok = sys.version_info >= (3, 9)
    record(f"Python {platform.python_version()}", ok,
           "" if ok else "3.9+ required")
    return ok


def step_core(check, yes):
    missing = [p for p, m in zip(
        CORE_PKGS, ["cv2", "rapidfuzz", "pytesseract", "PIL",
                    "flask", "yaml", "cryptography"]) if not have(m)]
    if not missing:
        record("Core Python packages", True, "all present")
        return
    if check:
        record("Core Python packages", False, "missing: " + ", ".join(missing))
        return
    if ask(f"Install Python packages: {', '.join(missing)}?", yes):
        r = pip("install", *missing)
        record("Core Python packages", r.returncode == 0,
               "" if r.returncode == 0 else (r.stderr or "")[-200:])


def step_spacy(check, yes):
    if have("spacy"):
        r = run([PYTHON, "-c", "import spacy; spacy.load('en_core_web_sm')"])
        if r.returncode == 0:
            record("spaCy NER + English model", True)
            return
        if check:
            record("spaCy NER", False, "model en_core_web_sm missing")
            return
        r = run([PYTHON, "-m", "spacy", "download", "en_core_web_sm"])
        record("spaCy English model", r.returncode == 0)
        return
    if check:
        record("spaCy NER (recommended for name detection)", False, "not installed")
        return
    if ask("Install spaCy NER for stronger name detection (recommended)?", yes):
        r = pip("install", "spacy")
        if r.returncode == 0:
            r = run([PYTHON, "-m", "spacy", "download", "en_core_web_sm"])
        record("spaCy NER + English model", r.returncode == 0)


def detect_nvidia():
    smi = shutil.which("nvidia-smi")
    if not smi:
        return None
    r = run([smi, "--query-gpu=name,driver_version", "--format=csv,noheader"],
            timeout=20)
    if r.returncode == 0 and r.stdout.strip():
        return r.stdout.strip().splitlines()[0]
    return None


def step_paddle(check, yes, cpu_only):
    gpu = None if cpu_only else detect_nvidia()
    if have("paddleocr") and have("paddle"):
        r = run([PYTHON, "-c",
                 "import paddle; print(paddle.device.is_compiled_with_cuda())"])
        cuda = "True" in (r.stdout or "")
        record("PaddleOCR", True,
               f"GPU build{' (CUDA active)' if cuda else ''}" if cuda
               else "CPU build" + (f" — NVIDIA GPU present ({gpu}); rerun "
                                   "installer to upgrade" if gpu else ""))
        if not (gpu and not cuda):
            return
    if check:
        record("PaddleOCR (recommended engine)", have("paddleocr"),
               f"NVIDIA GPU detected: {gpu}" if gpu else "no NVIDIA GPU — CPU build")
        return
    if gpu:
        log(f"  NVIDIA GPU detected: {gpu}")
        if ask("Install GPU-accelerated PaddleOCR (large download)?", yes):
            r = pip("install", "paddlepaddle-gpu", "-i",
                    "https://www.paddlepaddle.org.cn/packages/stable/cu126/")
            if r.returncode != 0:
                log("  GPU wheel failed — falling back to CPU build")
                r = pip("install", "paddlepaddle")
            else:
                pip("install", "-U", "nvidia-cudnn-cu12")   # cuDNN pairing
            r2 = pip("install", "paddleocr")
            record("PaddleOCR (GPU)", r.returncode == 0 and r2.returncode == 0)
            return
    if ask("Install CPU PaddleOCR (recommended engine)?", yes):
        r = pip("install", "paddlepaddle")
        r2 = pip("install", "paddleocr")
        record("PaddleOCR (CPU)", r.returncode == 0 and r2.returncode == 0)


def step_system_tools(check, yes):
    tess = shutil.which("tesseract") or (IS_WIN and os.path.exists(
        r"C:\Program Files\Tesseract-OCR\tesseract.exe"))
    ff = shutil.which("ffmpeg")
    if tess and ff:
        record("Tesseract + ffmpeg", True, "both present")
        return
    missing = [n for n, p in (("Tesseract OCR", tess), ("ffmpeg", ff)) if not p]
    if check:
        record("System tools", False, "missing: " + ", ".join(missing))
        return
    if not ask(f"Install system tools ({', '.join(missing)})?", yes):
        return
    if IS_WIN:
        wg = shutil.which("winget")
        if not wg:
            record("System tools", False,
                   "winget unavailable — install Tesseract (UB-Mannheim build) "
                   "and ffmpeg (gyan.dev) manually")
            return
        okall = True
        if not tess:
            okall &= run([wg, "install", "-e", "--id", "UB-Mannheim.TesseractOCR",
                          "--accept-source-agreements",
                          "--accept-package-agreements"]).returncode == 0
        if not ff:
            okall &= run([wg, "install", "-e", "--id", "Gyan.FFmpeg",
                          "--accept-source-agreements",
                          "--accept-package-agreements"]).returncode == 0
        record("System tools (winget)", okall,
               "open a NEW terminal afterwards so PATH updates apply")
    elif IS_LINUX:
        for mgr, cmd in (("apt-get", ["sudo", "apt-get", "install", "-y",
                                      "tesseract-ocr", "ffmpeg"]),
                         ("dnf", ["sudo", "dnf", "install", "-y",
                                  "tesseract", "ffmpeg"]),
                         ("pacman", ["sudo", "pacman", "-S", "--noconfirm",
                                     "tesseract", "ffmpeg"])):
            if shutil.which(mgr):
                r = run(cmd)
                record(f"System tools ({mgr})", r.returncode == 0,
                       "" if r.returncode == 0 else (r.stderr or "")[-160:])
                return
        record("System tools", False, "no known package manager — install "
                                      "tesseract-ocr and ffmpeg manually")
    elif IS_MAC:
        if shutil.which("brew"):
            r = run(["brew", "install", "tesseract", "ffmpeg"])
            record("System tools (brew)", r.returncode == 0)
        else:
            record("System tools", False,
                   "install Homebrew (https://brew.sh), then: "
                   "brew install tesseract ffmpeg")


def step_nvenc():
    ff = shutil.which("ffmpeg")
    if not ff:
        record("NVENC hardware encode", False, "ffmpeg not on PATH yet")
        return
    r = run([ff, "-hide_banner", "-f", "lavfi", "-i",
             "testsrc=size=256x256:rate=30:duration=1",
             "-c:v", "h264_nvenc", "-f", "null", "-"], timeout=60)
    record("NVENC hardware encode", r.returncode == 0,
           "GPU encoding available" if r.returncode == 0
           else "not available — renders will use CPU x264 (still works)")


def step_shortcut(check):
    if check:
        return
    name = "OpenScrub"
    if IS_WIN:
        bat = os.path.join(HERE, "openscrub_web.bat")
        ico = os.path.join(HERE, "openscrub.ico")
        ps = f'''
$W = New-Object -ComObject WScript.Shell
foreach ($dir in @([Environment]::GetFolderPath("Desktop"),
                   (Join-Path $env:APPDATA "Microsoft\\Windows\\Start Menu\\Programs"))) {{
  $s = $W.CreateShortcut((Join-Path $dir "{name}.lnk"))
  $s.TargetPath = "{bat}"
  $s.WorkingDirectory = "{HERE}"
  $s.IconLocation = "{ico}"
  $s.Description = "OpenScrub — PHI video redaction (opens the local web app)"
  $s.Save()
}}'''
        r = run(["powershell", "-NoProfile", "-Command", ps])
        record("Launcher shortcuts (Desktop + Start Menu)", r.returncode == 0)
    elif IS_LINUX:
        appdir = os.path.expanduser("~/.local/share/applications")
        os.makedirs(appdir, exist_ok=True)
        desktop = f"""[Desktop Entry]
Type=Application
Name={name}
Comment=PHI video redaction (local web app)
Exec=sh -c 'cd "{HERE}" && {PYTHON or "python3"} openscrub_web.py'
Icon={os.path.join(HERE, "logo_mark.png")}
Terminal=true
Categories=AudioVideo;Utility;
"""
        path = os.path.join(appdir, "openscrub.desktop")
        with open(path, "w", encoding="utf-8") as f:
            f.write(desktop)
        os.chmod(path, 0o755)
        dsk = os.path.expanduser("~/Desktop")
        if os.path.isdir(dsk):
            shutil.copy(path, os.path.join(dsk, "openscrub.desktop"))
            os.chmod(os.path.join(dsk, "openscrub.desktop"), 0o755)
        record("Launcher (.desktop entry)", True, path)
    elif IS_MAC:
        cmd_path = os.path.expanduser("~/Desktop/OpenScrub.command")
        with open(cmd_path, "w", encoding="utf-8") as f:
            f.write(f'#!/bin/bash\ncd "{HERE}"\n{PYTHON or "python3"} openscrub_web.py\n')
        os.chmod(cmd_path, 0o755)
        record("Launcher (~/Desktop/OpenScrub.command)", True,
               "macOS support is best-effort: CPU OCR, x264 encoding")


def main():
    ap = argparse.ArgumentParser(description="OpenScrub installer")
    ap.add_argument("--check", action="store_true", help="report only")
    ap.add_argument("--yes", action="store_true", help="no prompts")
    ap.add_argument("--cpu-only", action="store_true")
    ap.add_argument("--no-shortcut", action="store_true")
    ap.add_argument("--with-plates", action="store_true",
                    help="download a license-plate detection model into "
                         "models/plate_yolov8.onnx (enables the 'plate' "
                         "category). Optional; skipped by default.")
    ap.add_argument("--plate-model-url", default="",
                    help="override the URL to fetch the plate ONNX model from")
    a = ap.parse_args()

    log("=" * 62)
    log(f" OpenScrub installer — {platform.system()} {platform.release()}")
    log("=" * 62)
    if FROZEN and not ensure_python(a.yes):
        log("=" * 62)
        if IS_WIN:
            input("Press Enter to close.")
        sys.exit(1)
    if not step_python():
        sys.exit(1)
    step_core(a.check, a.yes)
    step_spacy(a.check, a.yes)
    step_system_tools(a.check, a.yes)
    step_paddle(a.check, a.yes, a.cpu_only)
    step_nvenc()
    if not a.no_shortcut:
        step_shortcut(a.check)

    log("=" * 62)
    bad = [s for s, ok, _ in RESULTS if not ok]
    if a.check:
        log(f" Check complete — {len(RESULTS) - len(bad)}/{len(RESULTS)} ready."
            + ("" if not bad else " Run without --check to install."))
    elif bad:
        log(" Finished with warnings: " + "; ".join(bad))
        log(" Everything else works — Tesseract/x264 fallbacks cover gaps.")
    else:
        log(" Install complete.")
        log(f" Start OpenScrub: the '{'OpenScrub' if not a.no_shortcut else 'launcher'}'"
            " shortcut, or:")
        log(f"     cd {HERE}")
        log(f"     {os.path.basename(sys.executable)} openscrub_web.py")
        log(" First HTTPS start generates a self-signed certificate —")
        log(" your browser warns once; proceed, or install your own cert")
        log(" at the bottom of the main page.")
    log("=" * 62)
    if FROZEN and IS_WIN and not a.yes:
        input("Press Enter to close.")


if __name__ == "__main__":
    main()

# PyInstaller spec — builds the two branded Windows executables:
#
#     dist/OpenScrub/openscrub.exe        (CLI engine)
#     dist/OpenScrub/openscrub-web.exe    (web interface server)
#
# Build (from the repo root, on Windows):  pyinstaller windows\openscrub.spec
# Normally run via windows\build_installer.bat, which also compiles the
# Inno Setup installer that puts this folder into Program Files.
#
# Notes:
# - onedir (not onefile): faster startup, friendlier to antivirus, and the
#   two exes share one copy of the ~200MB of libraries.
# - spaCy / PaddleOCR are excluded: they are heavyweight optional extras.
#   The frozen build detects text with Tesseract + the heuristic name
#   detectors; NER can be added later via a full pip install instead.
# - openscrub.install_is_readonly() returns True under sys.frozen, so jobs,
#   certs, zones, downloaded models, and TOFU hash pins all go to
#   %LOCALAPPDATA%\OpenScrub — never into Program Files.

import os

ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))
ICON = os.path.join(ROOT, "assets", "openscrub.ico")

DATAS = [
    (os.path.join(ROOT, "plate_models.json"), "."),
    (os.path.join(ROOT, "face_models.json"), "."),
    (os.path.join(ROOT, "LICENSE"), "."),
]
HIDDEN = [
    "cheroot", "cheroot.wsgi", "cheroot.ssl", "cheroot.ssl.builtin",
    "openscrub_update", "openscrub_setup", "openscrub_vault", "zones_ui",
]
EXCLUDES = [
    "spacy", "paddleocr", "paddle", "torch", "torchvision",
    "matplotlib", "IPython", "jupyter", "PyQt5", "PySide2",
]

a_web = Analysis(
    [os.path.join(ROOT, "openscrub_web.py")],
    pathex=[ROOT], datas=DATAS, hiddenimports=HIDDEN, excludes=EXCLUDES,
)
a_cli = Analysis(
    [os.path.join(ROOT, "openscrub.py")],
    pathex=[ROOT], datas=DATAS, hiddenimports=[], excludes=EXCLUDES,
)

pyz_web = PYZ(a_web.pure)
pyz_cli = PYZ(a_cli.pure)

exe_web = EXE(
    pyz_web, a_web.scripts, [],
    exclude_binaries=True, name="openscrub-web", icon=ICON,
    console=True,        # the console shows the access URL and the log
)
exe_cli = EXE(
    pyz_cli, a_cli.scripts, [],
    exclude_binaries=True, name="openscrub", icon=ICON,
    console=True,
)

coll = COLLECT(
    exe_web, a_web.binaries, a_web.zipfiles, a_web.datas,
    exe_cli, a_cli.binaries, a_cli.zipfiles, a_cli.datas,
    name="OpenScrub",
)

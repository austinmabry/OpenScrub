@echo off
rem ==========================================================================
rem  Build the Windows installer:
rem    1. PyInstaller -> dist\OpenScrub\{openscrub.exe, openscrub-web.exe}
rem    2. Inno Setup  -> windows\output\OpenScrub-Setup-<version>.exe
rem
rem  Run from anywhere; needs Python 3.10+ on PATH. For step 2 install
rem  Inno Setup 6 first:  winget install -e --id JRSoftware.InnoSetup
rem
rem  Attach the resulting OpenScrub-Setup-<version>.exe to the GitHub
rem  release so Windows users get a Program Files install with Start Menu
rem  shortcuts instead of hunting for pip's Scripts folder.
rem ==========================================================================
setlocal
cd /d "%~dp0.."

for /f %%v in ('python -c "import openscrub; print(openscrub.VERSION)"') do set VER=%%v
if "%VER%"=="" ( echo Could not read VERSION from openscrub.py & exit /b 1 )
echo Building OpenScrub v%VER% ...

python -m pip install --quiet --upgrade pyinstaller || exit /b 1

rem GPU acceleration: replace the stock CPU-only onnxruntime with
rem onnxruntime-directml, whose DmlExecutionProvider runs the ONNX
rem detectors (plates, person/seg, ONNX OCR, and the SCRFD face model)
rem on ANY DirectX 12 GPU — NVIDIA, AMD or Intel. Same module name
rem ("onnxruntime"), so the spec's collect_all still bundles it plus the
rem DirectML.dll; _ort_session prefers DmlExecutionProvider and falls
rem back to CPU per-node if the GPU init fails. Install LAST so its files
rem win. (The two Docker images do the CUDA/OpenVINO equivalents.)
python -m pip uninstall -y onnxruntime onnxruntime-directml >nul 2>nul
python -m pip install --quiet onnxruntime-directml || exit /b 1

python -m PyInstaller --noconfirm windows\openscrub.spec || exit /b 1
echo PyInstaller build OK: dist\OpenScrub

set ISCC=
for %%p in ("%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe" "%ProgramFiles%\Inno Setup 6\ISCC.exe") do (
  if exist %%p set ISCC=%%p
)
where ISCC >nul 2>nul && set ISCC=ISCC
if "%ISCC%"=="" (
  echo Inno Setup not found - skipping installer compile.
  echo Install it with:  winget install -e --id JRSoftware.InnoSetup
  echo Then re-run this script.
  exit /b 0
)
%ISCC% /DMyAppVersion=%VER% windows\installer.iss || exit /b 1
echo.
echo Done: windows\output\OpenScrub-Setup-%VER%.exe
endlocal

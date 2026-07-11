; Inno Setup script — packages dist\OpenScrub (built by windows\openscrub.spec)
; into a standard Windows installer that installs to Program Files with
; Start Menu entries and an uninstaller.
;
; Compile (after the PyInstaller build):
;     ISCC /DMyAppVersion=1.0.7 windows\installer.iss
; or just run windows\build_installer.bat which does both steps.
;
; Requires Inno Setup 6:  winget install -e --id JRSoftware.InnoSetup

#ifndef MyAppVersion
  #define MyAppVersion "0.0.0"
#endif

[Setup]
AppId={{7E60A2C1-58D3-4F8B-9A34-OPENSCRUB01}
AppName=OpenScrub
AppVersion={#MyAppVersion}
AppPublisher=OpenScrub project
AppPublisherURL=https://github.com/austinmabry/OpenScrub
DefaultDirName={autopf}\OpenScrub
DefaultGroupName=OpenScrub
DisableProgramGroupPage=yes
LicenseFile=..\LICENSE
OutputDir=output
OutputBaseFilename=OpenScrub-Setup-{#MyAppVersion}
SetupIconFile=..\assets\openscrub.ico
UninstallDisplayIcon={app}\openscrub-web.exe
Compression=lzma2
SolidCompression=yes
ArchitecturesInstallIn64BitMode=x64compatible
; app data (jobs/certs/zones/models) lives in %LOCALAPPDATA%\OpenScrub,
; written by the app itself — nothing under {app} is ever written to.

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; \
  GroupDescription: "Additional icons:"
Name: "tesseract"; Description: "Install Tesseract OCR via winget (needed for text detection)"; \
  GroupDescription: "System tools (skip if already installed):"; Flags: unchecked
Name: "ffmpeg"; Description: "Install FFmpeg via winget (audio + H.264 output)"; \
  GroupDescription: "System tools (skip if already installed):"; Flags: unchecked

[Files]
Source: "..\dist\OpenScrub\*"; DestDir: "{app}"; \
  Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\OpenScrub Web"; Filename: "{app}\openscrub-web.exe"; \
  Comment: "Start the OpenScrub web interface"
Name: "{group}\OpenScrub CLI"; Filename: "{cmd}"; \
  Parameters: "/k ""{app}\openscrub.exe"" --help"; \
  IconFilename: "{app}\openscrub.exe"; Comment: "OpenScrub command line"
Name: "{autodesktop}\OpenScrub Web"; Filename: "{app}\openscrub-web.exe"; \
  Tasks: desktopicon

[Run]
Filename: "winget"; Parameters: "install -e --id UB-Mannheim.TesseractOCR"; \
  Tasks: tesseract; Flags: shellexec waituntilterminated; \
  StatusMsg: "Installing Tesseract OCR (winget)…"
Filename: "winget"; Parameters: "install -e --id Gyan.FFmpeg"; \
  Tasks: ffmpeg; Flags: shellexec waituntilterminated; \
  StatusMsg: "Installing FFmpeg (winget)…"
Filename: "{app}\openscrub-web.exe"; Description: "Start OpenScrub Web now"; \
  Flags: postinstall nowait skipifsilent

#!/usr/bin/env python3
"""
openscrub_gui.py — Windows GUI for openscrub.

A tkinter front-end over openscrub.run_pipeline(). Everything the command
line can do, plus a live preview of each frame as it's analyzed, one-click
installers for the optional OCR/NER engines, and a cancel button.

Run:  python openscrub_gui.py          (or double-click openscrub_gui.bat)

Requires the same environment as openscrub.py plus Pillow for the preview
(pip install pillow). tkinter ships with the python.org Windows installer.
"""

import importlib.util
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

import openscrub

try:
    from PIL import Image, ImageTk
    HAVE_PIL = True
except ImportError:
    HAVE_PIL = False

PREVIEW_W = 560


# ----------------------------------------------------------------------------
# Environment probing
# ----------------------------------------------------------------------------

def probe_tesseract():
    if shutil.which("tesseract"):
        return True
    return os.name == "nt" and any(os.path.exists(p)
                                   for p in openscrub.WINDOWS_TESSERACT_PATHS)


def probe_paddle():
    """-> (installed: bool, gpu: bool)"""
    if importlib.util.find_spec("paddleocr") is None:
        return False, False
    gpu = False
    try:
        import paddle
        gpu = (paddle.device.is_compiled_with_cuda()
               and paddle.device.cuda.device_count() > 0)
    except Exception:
        pass
    return True, gpu


def probe_spacy():
    return (importlib.util.find_spec("spacy") is not None
            and importlib.util.find_spec("en_core_web_sm") is not None)


def probe_nvenc():
    if not shutil.which("ffmpeg"):
        return "no ffmpeg"
    try:
        listed = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"],
                                capture_output=True, text=True, timeout=15).stdout
        return "available" if "h264_nvenc" in listed else "not in build"
    except Exception:
        return "unknown"


INSTALLS = {
    "PaddleOCR (CPU)": [[sys.executable, "-m", "pip", "install", "paddleocr", "paddlepaddle"]],
    "PaddleOCR (GPU, CUDA 12.6)": [
        [sys.executable, "-m", "pip", "install", "paddleocr"],
        [sys.executable, "-m", "pip", "uninstall", "-y", "paddlepaddle"],
        [sys.executable, "-m", "pip", "install", "paddlepaddle-gpu==3.2.2",
         "-i", "https://www.paddlepaddle.org.cn/packages/stable/cu126/"]],
    "spaCy NER (recommended)": [
        [sys.executable, "-m", "pip", "install", "spacy"],
        [sys.executable, "-m", "spacy", "download", "en_core_web_sm"]],
}


# ----------------------------------------------------------------------------
# GUI <-> pipeline bridge
# ----------------------------------------------------------------------------

class GuiCallbacks(openscrub.Callbacks):
    """Thread-safe: pushes events onto a queue the UI thread drains."""
    wants_frames = True

    def __init__(self, q, cancel_event):
        self.q = q
        self.cancel_event = cancel_event

    def log(self, msg):
        self.q.put(("log", msg))

    def progress(self, stage, current, total):
        self.q.put(("progress", stage, current, total))

    def scan_frame(self, frame_bgr, t, found):
        # keep only the newest frame pending; drop stale ones
        try:
            while True:
                item = self.q.get_nowait()
                if item[0] != "frame":
                    self.q.put(item)
                    break
        except queue.Empty:
            pass
        self.q.put(("frame", frame_bgr, t, found))

    def cancelled(self):
        return self.cancel_event.is_set()


class App:
    def __init__(self, root):
        self.root = root
        root.title("OpenScrub — PHI video redaction")
        here = os.path.dirname(os.path.abspath(__file__))
        try:
            if os.name == "nt" and os.path.exists(os.path.join(here, "openscrub.ico")):
                root.iconbitmap(os.path.join(here, "openscrub.ico"))
            elif os.path.exists(os.path.join(here, "logo_mark.png")):
                root.iconphoto(True, tk.PhotoImage(
                    file=os.path.join(here, "logo_mark.png")))
        except Exception:
            pass  # branding is optional — never block startup on an icon
        root.minsize(1080, 660)
        self.q = queue.Queue()
        self.cancel_event = threading.Event()
        self.worker = None
        self._tmp_files = []
        self._preview_img = None  # keep reference or tk garbage-collects it

        self._build_ui()
        self._refresh_status()
        self.root.after(100, self._drain_queue)

    # ---------------- UI construction ----------------

    def _build_ui(self):
        outer = ttk.Frame(self.root, padding=8)
        outer.pack(fill="both", expand=True)
        left = ttk.Frame(outer)
        left.pack(side="left", fill="y", padx=(0, 8))
        right = ttk.Frame(outer)
        right.pack(side="left", fill="both", expand=True)

        # --- files ---
        files = ttk.LabelFrame(left, text="Files", padding=6)
        files.pack(fill="x", pady=(0, 6))
        self.src_var = tk.StringVar()
        self.out_var = tk.StringVar()
        self.report_var = tk.StringVar()
        self._file_row(files, 0, "Source video", self.src_var, self._browse_src)
        self._file_row(files, 1, "Output video", self.out_var, self._browse_out)
        self._file_row(files, 2, "Audit report (optional)", self.report_var,
                       self._browse_report)

        # --- engines ---
        eng = ttk.LabelFrame(left, text="OCR / detection engines", padding=6)
        eng.pack(fill="x", pady=(0, 6))
        self.engine_var = tk.StringVar(value="auto")
        row = ttk.Frame(eng); row.pack(fill="x")
        ttk.Label(row, text="OCR engine:").pack(side="left")
        for v in ("auto", "paddle", "tesseract"):
            ttk.Radiobutton(row, text=v, value=v,
                            variable=self.engine_var).pack(side="left", padx=4)
        self.status_lbl = ttk.Label(eng, text="", justify="left")
        self.status_lbl.pack(fill="x", pady=(4, 4))
        irow = ttk.Frame(eng); irow.pack(fill="x")
        ttk.Label(irow, text="Install:").pack(side="left")
        for name in INSTALLS:
            ttk.Button(irow, text=name, width=24,
                       command=lambda n=name: self._install(n)
                       ).pack(side="left", padx=2)

        # --- compute ---
        comp = ttk.LabelFrame(left, text="Compute", padding=6)
        comp.pack(fill="x", pady=(0, 6))
        self.device_var = tk.StringVar(value="auto")
        self.encoder_var = tk.StringVar(value="auto")
        row = ttk.Frame(comp); row.pack(fill="x")
        ttk.Label(row, text="OCR device:").pack(side="left")
        for v in ("auto", "gpu", "cpu"):
            ttk.Radiobutton(row, text=v, value=v,
                            variable=self.device_var).pack(side="left", padx=4)
        row = ttk.Frame(comp); row.pack(fill="x")
        ttk.Label(row, text="Video encoder:").pack(side="left")
        for v, lbl in (("auto", "auto"), ("nvenc", "NVENC (GPU)"), ("x264", "x264 (CPU)")):
            ttk.Radiobutton(row, text=lbl, value=v,
                            variable=self.encoder_var).pack(side="left", padx=4)

        # --- detection options ---
        det = ttk.LabelFrame(left, text="Detection", padding=6)
        det.pack(fill="x", pady=(0, 6))
        self.cat_vars = {}
        row = ttk.Frame(det); row.pack(fill="x")
        ttk.Label(row, text="Categories:").pack(side="left")
        for c in ("name", "dob", "phone", "ssn", "mrn", "email", "face"):
            v = tk.BooleanVar(value=True)
            self.cat_vars[c] = v
            ttk.Checkbutton(row, text=c, variable=v).pack(side="left", padx=2)

        self.preview_mode = tk.BooleanVar(value=False)
        self.paranoid = tk.BooleanVar(value=False)
        self.upscale_var = tk.StringVar(value="auto")
        self.no_memory = tk.BooleanVar(value=False)
        self.no_ner = tk.BooleanVar(value=False)
        self.mode_var = tk.StringVar(value="blur")
        row = ttk.Frame(det); row.pack(fill="x", pady=(4, 0))
        ttk.Checkbutton(row, text="Preview mode (draw boxes, don't blur)",
                        variable=self.preview_mode).pack(side="left")
        ttk.Checkbutton(row, text="Disable PHI memory",
                        variable=self.no_memory).pack(side="left", padx=8)
        row = ttk.Frame(det); row.pack(fill="x")
        ttk.Checkbutton(row, text="Disable spaCy NER",
                        variable=self.no_ner).pack(side="left")
        ttk.Checkbutton(row, text="Paranoid (max recall)",
                        variable=self.paranoid).pack(side="left", padx=8)
        ttk.Label(row, text="  Redaction:").pack(side="left")
        for v in ("blur", "box"):
            ttk.Radiobutton(row, text=v, value=v,
                            variable=self.mode_var).pack(side="left", padx=4)

        grid = ttk.Frame(det); grid.pack(fill="x", pady=(6, 0))
        self.sample_var = tk.DoubleVar(value=0.5)
        self.trigger_var = tk.DoubleVar(value=60)
        self.pad_var = tk.IntVar(value=8)
        self.bridge_var = tk.DoubleVar(value=4.0)
        self.face_expand_var = tk.DoubleVar(value=0.15)
        self.skip_start_var = tk.DoubleVar(value=0.0)
        self.skip_end_var = tk.DoubleVar(value=0.0)
        self.mrn_var = tk.StringVar(value=openscrub.RE_MRN_DEFAULT)
        for i, (lbl, var, w) in enumerate((
                ("Sample interval (s)", self.sample_var, 6),
                ("Scan trigger (px)", self.trigger_var, 6),
                ("Blur buffer (px)", self.pad_var, 6),
                ("Bridge gap (s)", self.bridge_var, 6),
                ("Face expand (0-1)", self.face_expand_var, 6),
                ("Skip first (s)", self.skip_start_var, 6),
                ("Skip last (s)", self.skip_end_var, 6))):
            ttk.Label(grid, text=lbl).grid(row=i // 2, column=(i % 2) * 2,
                                           sticky="w", padx=(0, 4), pady=2)
            ttk.Entry(grid, textvariable=var, width=w).grid(
                row=i // 2, column=(i % 2) * 2 + 1, sticky="w", pady=2)
        row = ttk.Frame(det); row.pack(fill="x", pady=(4, 0))
        ttk.Label(row, text="OCR upscale:").pack(side="left")
        for v in ("auto", "on", "off"):
            ttk.Radiobutton(row, text=v, value=v,
                            variable=self.upscale_var).pack(side="left", padx=3)
        row = ttk.Frame(det); row.pack(fill="x", pady=(4, 0))
        ttk.Label(row, text="MRN regex:").pack(side="left")
        ttk.Entry(row, textvariable=self.mrn_var, width=28).pack(
            side="left", padx=4, fill="x", expand=True)

        # --- name lists ---
        names = ttk.LabelFrame(left, text="Name lists (one name per line)",
                               padding=6)
        names.pack(fill="both", expand=True, pady=(0, 6))
        cols = ttk.Frame(names); cols.pack(fill="both", expand=True)
        lc = ttk.Frame(cols); lc.pack(side="left", fill="both", expand=True,
                                      padx=(0, 4))
        rc = ttk.Frame(cols); rc.pack(side="left", fill="both", expand=True)
        ttk.Label(lc, text="Allow (keep visible — providers/staff)").pack(anchor="w")
        self.allow_txt = tk.Text(lc, width=26, height=6)
        self.allow_txt.pack(fill="both", expand=True)
        ttk.Button(lc, text="Load from file…",
                   command=lambda: self._load_names(self.allow_txt)).pack(anchor="w")
        ttk.Label(rc, text="Always blur (extra names)").pack(anchor="w")
        self.extra_txt = tk.Text(rc, width=26, height=6)
        self.extra_txt.pack(fill="both", expand=True)
        ttk.Button(rc, text="Load from file…",
                   command=lambda: self._load_names(self.extra_txt)).pack(anchor="w")

        # --- run controls ---
        runf = ttk.Frame(left); runf.pack(fill="x")
        self.run_btn = ttk.Button(runf, text="▶  Run", command=self._start)
        self.run_btn.pack(side="left")
        self.cancel_btn = ttk.Button(runf, text="Cancel", command=self._cancel,
                                     state="disabled")
        self.cancel_btn.pack(side="left", padx=6)
        self.prog = ttk.Progressbar(runf, maximum=100)
        self.prog.pack(side="left", fill="x", expand=True, padx=6)
        self.stage_lbl = ttk.Label(runf, text="idle", width=18)
        self.stage_lbl.pack(side="left")

        # --- right side: preview + log ---
        pv = ttk.LabelFrame(right, text="Analysis preview", padding=4)
        pv.pack(fill="both", expand=True)
        self.preview_lbl = ttk.Label(
            pv, text=("(frames appear here during scanning)"
                      if HAVE_PIL else
                      "Pillow not installed — run: pip install pillow"),
            anchor="center")
        self.preview_lbl.pack(fill="both", expand=True)
        self.preview_cap = ttk.Label(pv, text="")
        self.preview_cap.pack()

        logf = ttk.LabelFrame(right, text="Log", padding=4)
        logf.pack(fill="both", expand=True, pady=(6, 0))
        self.log = scrolledtext.ScrolledText(logf, height=12, state="disabled",
                                             font=("Consolas", 9))
        self.log.pack(fill="both", expand=True)

    def _file_row(self, parent, r, label, var, cmd):
        ttk.Label(parent, text=label, width=22).grid(row=r, column=0, sticky="w")
        ttk.Entry(parent, textvariable=var, width=38).grid(row=r, column=1,
                                                           sticky="we", padx=4)
        ttk.Button(parent, text="Browse…", command=cmd).grid(row=r, column=2)
        parent.columnconfigure(1, weight=1)

    # ---------------- actions ----------------

    def _browse_src(self):
        p = filedialog.askopenfilename(
            title="Select screen recording",
            filetypes=[("Video", "*.mp4 *.mkv *.mov *.avi *.webm"), ("All", "*.*")])
        if p:
            self.src_var.set(p)
            if not self.out_var.get():
                base, _ = os.path.splitext(p)
                self.out_var.set(base + "_redacted.mp4")

    def _browse_out(self):
        p = filedialog.asksaveasfilename(defaultextension=".mp4",
                                         filetypes=[("MP4", "*.mp4")])
        if p:
            self.out_var.set(p)

    def _browse_report(self):
        p = filedialog.asksaveasfilename(defaultextension=".json",
                                         filetypes=[("JSON", "*.json")])
        if p:
            self.report_var.set(p)

    def _load_names(self, widget):
        p = filedialog.askopenfilename(filetypes=[("Text", "*.txt"), ("All", "*.*")])
        if p:
            with open(p, encoding="utf-8", errors="replace") as f:
                widget.delete("1.0", "end")
                widget.insert("1.0", f.read())

    def _refresh_status(self):
        tess = probe_tesseract()
        pad, gpu = probe_paddle()
        sp = probe_spacy()
        nv = probe_nvenc()
        self.status_lbl.config(text=(
            f"Tesseract: {'✔ found' if tess else '✘ missing'}   "
            f"PaddleOCR: {'✔ installed' + (' (GPU ready)' if gpu else ' (CPU only)') if pad else '✘ missing'}\n"
            f"spaCy NER: {'✔ ready' if sp else '✘ missing (heuristics will be used)'}   "
            f"NVENC: {nv}"))

    def _install(self, name):
        if self.worker and self.worker.is_alive():
            messagebox.showinfo("Busy", "Wait for the current job to finish.")
            return
        self._log(f"--- installing {name} (this can take several minutes) ---")
        self._set_running(True, stage="installing…")

        def work():
            ok = True
            for cmd in INSTALLS[name]:
                self.q.put(("log", "  $ " + " ".join(cmd)))
                try:
                    p = subprocess.run(cmd, capture_output=True, text=True)
                    for line in (p.stdout or "").splitlines()[-8:]:
                        self.q.put(("log", "  " + line))
                    if p.returncode != 0:
                        for line in (p.stderr or "").splitlines()[-8:]:
                            self.q.put(("log", "  " + line))
                        ok = False
                        break
                except Exception as e:
                    self.q.put(("log", f"  install error: {e}"))
                    ok = False
                    break
            self.q.put(("log", f"--- {name}: "
                        + ("installed OK" if ok else "FAILED (see above)") + " ---"))
            self.q.put(("done", None))

        self.worker = threading.Thread(target=work, daemon=True)
        self.worker.start()

    def _collect_args(self):
        src = self.src_var.get().strip()
        if not src:
            raise ValueError("Select a source video first.")
        if not os.path.exists(src):
            raise ValueError(f"Source not found: {src}")
        cats = [c for c, v in self.cat_vars.items() if v.get()]
        if not cats:
            raise ValueError("Select at least one PHI category.")

        argv = [src, "--engine", self.engine_var.get(),
                "--device", self.device_var.get(),
                "--encoder", self.encoder_var.get(),
                "--mode", self.mode_var.get(),
                "--sample-interval", str(self.sample_var.get()),
                "--scan-trigger", str(self.trigger_var.get()),
                "--pad", str(int(self.pad_var.get())),
                "--bridge-gap", str(self.bridge_var.get()),
                "--face-expand", str(self.face_expand_var.get()),
                "--skip-start", str(self.skip_start_var.get()),
                "--skip-end", str(self.skip_end_var.get()),
                "--mrn-regex", self.mrn_var.get(),
                "--categories", ",".join(cats)]
        if self.out_var.get().strip():
            argv += ["-o", self.out_var.get().strip()]
        if self.report_var.get().strip():
            argv += ["--report", self.report_var.get().strip()]
        if self.preview_mode.get():
            argv.append("--preview")
        if self.no_memory.get():
            argv.append("--no-memory")
        if self.no_ner.get():
            argv.append("--no-ner")
        if self.paranoid.get():
            argv.append("--paranoid")
        argv += ["--ocr-upscale", self.upscale_var.get()]

        for text_widget, flag in ((self.allow_txt, "--allow-names"),
                                  (self.extra_txt, "--extra-names")):
            content = text_widget.get("1.0", "end").strip()
            if content:
                fd, path = tempfile.mkstemp(suffix=".txt", text=True)
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(content)
                self._tmp_files.append(path)
                argv += [flag, path]

        parser = openscrub.build_parser()
        return openscrub._prep_args(parser.parse_args(argv), parser)

    def _start(self):
        if self.worker and self.worker.is_alive():
            return
        try:
            args = self._collect_args()
        except ValueError as e:
            messagebox.showerror("openscrub", str(e))
            return
        self.cancel_event.clear()
        self._set_running(True, stage="starting…")
        self.prog["value"] = 0
        cb = GuiCallbacks(self.q, self.cancel_event)

        def work():
            try:
                res = openscrub.run_pipeline(args, cb)
                self.q.put(("log", f"SUMMARY: {res}"))
                self.q.put(("finished", res.get("output")))
            except openscrub.PipelineCancelled:
                self.q.put(("log", "Cancelled — partial output removed."))
            except Exception as e:
                self.q.put(("log", f"ERROR: {e}"))
                self.q.put(("error", str(e)))
            finally:
                self.q.put(("done", None))

        self.worker = threading.Thread(target=work, daemon=True)
        self.worker.start()

    def _cancel(self):
        self.cancel_event.set()
        self.stage_lbl.config(text="cancelling…")

    def _set_running(self, running, stage="idle"):
        self.run_btn.config(state="disabled" if running else "normal")
        self.cancel_btn.config(state="normal" if running else "disabled")
        self.stage_lbl.config(text=stage)

    # ---------------- queue drain (UI thread) ----------------

    def _log(self, msg):
        self.log.config(state="normal")
        self.log.insert("end", msg + "\n")
        self.log.see("end")
        self.log.config(state="disabled")

    def _show_frame(self, frame_bgr, t, found):
        if not HAVE_PIL:
            return
        h, w = frame_bgr.shape[:2]
        scale = PREVIEW_W / w
        import cv2
        small = cv2.resize(frame_bgr, (PREVIEW_W, int(h * scale)))
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        img = ImageTk.PhotoImage(Image.fromarray(rgb))
        self._preview_img = img
        self.preview_lbl.config(image=img, text="")
        self.preview_cap.config(text=f"t = {t:.2f}s   |   {found} PHI region(s) this scan")

    def _drain_queue(self):
        try:
            while True:
                item = self.q.get_nowait()
                kind = item[0]
                if kind == "log":
                    self._log(item[1])
                elif kind == "progress":
                    _, stage, cur, total = item
                    pct = 100 * cur / max(total, 1)
                    self.prog["value"] = pct
                    self.stage_lbl.config(text=f"{stage} {pct:.0f}%")
                elif kind == "frame":
                    self._show_frame(item[1], item[2], item[3])
                elif kind == "finished":
                    self.stage_lbl.config(text="done")
                    messagebox.showinfo(
                        "openscrub",
                        f"Finished:\n{item[1]}\n\nReminder: review the output "
                        "before sharing — this tool is best-effort, not a "
                        "compliance guarantee.")
                elif kind == "error":
                    messagebox.showerror("openscrub", item[1])
                elif kind == "done":
                    self._set_running(False, stage="idle")
                    self._refresh_status()
                    for p in self._tmp_files:
                        try:
                            os.remove(p)
                        except OSError:
                            pass
                    self._tmp_files.clear()
        except queue.Empty:
            pass
        self.root.after(100, self._drain_queue)


def main():
    root = tk.Tk()
    try:
        ttk.Style().theme_use("vista" if os.name == "nt" else "clam")
    except tk.TclError:
        pass
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
openscrub.py — the OpenScrub engine: automatic PII/PHI redaction for
videos and screen recordings. Windows + Linux. No name list required.

Detects and blurs 12 categories — names, dates of birth, phone numbers,
SSNs, MRNs, emails, addresses, card numbers, API keys, IP addresses,
license plates, and faces — using OCR + pattern matching, spaCy NER with
heuristic fallbacks, and DNN detectors for faces and plates.

Scroll-aware: blur boxes are anchored in content coordinates and ride
along with the text on every frame; any region that scrolled into view
since the last OCR scan stays covered until it has been scanned. Name
detection needs no list — spaCy PERSON entities, a label heuristic
("Patient:", "Name:", …), and a capitalized-pair fallback stack up, with
an --allow-names file to keep chosen names visible.

Usage (see README.md):
  python openscrub.py recording.mp4
  python openscrub.py recording.mp4 --allow-names providers.txt --preview
"""

import argparse
import datetime
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, asdict

import cv2
import numpy as np

VERSION = "1.0.35"

# ----------------------------------------------------------------------------
# OCR backends
# ----------------------------------------------------------------------------

WINDOWS_TESSERACT_PATHS = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
]


class OcrBackend:
    """Returns list of (text, (x1, y1, x2, y2), confidence) for a BGR frame."""

    def read(self, frame):
        raise NotImplementedError


class TesseractBackend(OcrBackend):
    def __init__(self):
        import pytesseract
        self.pt = pytesseract
        if os.name == "nt" and not shutil.which("tesseract"):
            for p in WINDOWS_TESSERACT_PATHS:
                if os.path.exists(p):
                    pytesseract.pytesseract.tesseract_cmd = p
                    break
            else:
                sys.exit("Tesseract not found. Install from "
                         "https://github.com/UB-Mannheim/tesseract/wiki "
                         "or add it to PATH.")

    def read(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        data = self.pt.image_to_data(gray, output_type=self.pt.Output.DICT)
        out = []
        def phi_shaped(t):
            s = t.strip(".,:;()[]")
            return ("@" in t or RE_SSN.search(t) or RE_PHONE.search(t)
                    or RE_DATE.search(t)
                    or sum(ch.isdigit() for ch in s) >= 6)
        for i in range(len(data["text"])):
            txt = data["text"][i].strip()
            conf = float(data["conf"][i]) if data["conf"][i] not in ("-1", -1) else 0.0
            if not txt:
                continue
            # low-confidence words are normally dropped, but words that are
            # structurally PHI-shaped (emails, phones, SSNs, dates, long
            # digit runs) are rescued: a misread MRN is still an MRN
            if conf < 40 and not (conf >= 5 and phi_shaped(txt)):
                continue
            x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
            out.append((txt, (x, y, x + w, y + h), conf / 100.0))
        return out


class PaddleBackend(OcrBackend):
    def __init__(self, device="auto"):
        import logging
        for name in ("paddlex", "paddleocr", "ppocr", "paddle"):
            try:
                logging.getLogger(name).setLevel(logging.ERROR)
            except Exception:
                pass
        import paddleocr
        from paddleocr import PaddleOCR
        ver = getattr(paddleocr, "__version__", "3.0.0")
        try:
            self.v3 = int(str(ver).split(".")[0]) >= 3
        except ValueError:
            self.v3 = True

        # resolve device
        if device == "auto":
            try:
                import paddle
                device = ("gpu" if paddle.device.is_compiled_with_cuda()
                          and paddle.device.cuda.device_count() > 0 else "cpu")
            except Exception:
                device = "cpu"
        self.device = device
        print(f"      paddle device: {self.device}")

        if self.v3:
            # PaddleOCR >= 3.0: new pipeline API. Disable the document
            # preprocessing stages — screen recordings are already flat,
            # upright, and undistorted, so they just cost time.
            # enable_mkldnn=False works around a paddlepaddle 3.x bug on
            # Windows CPU ("ConvertPirAttribute2RuntimeAttribute not
            # support" in onednn_instruction.cc); irrelevant on GPU.
            kwargs = dict(
                lang="en",
                device=self.device,
                use_textline_orientation=False,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
            )
            if self.device == "cpu":
                kwargs["enable_mkldnn"] = False
            try:
                self.ocr = PaddleOCR(**kwargs)
            except (TypeError, ValueError):
                # builds without enable_mkldnn / device args
                kwargs.pop("enable_mkldnn", None)
                kwargs.pop("device", None)
                self.ocr = PaddleOCR(**kwargs)
        else:
            # PaddleOCR 2.x: legacy API
            self.ocr = PaddleOCR(use_angle_cls=False, lang="en",
                                 show_log=False, use_gpu=(self.device == "gpu"))

    def _lines(self, frame):
        """Yield (text, x1, y1, x2, y2, conf) line-level results, either API."""
        if self.v3:
            results = self.ocr.predict(frame)
            for res in results or []:
                texts = res.get("rec_texts") or []
                scores = res.get("rec_scores") or []
                polys = res.get("rec_polys")
                boxes = res.get("rec_boxes")
                for i, txt in enumerate(texts):
                    conf = float(scores[i]) if i < len(scores) else 1.0
                    if polys is not None and i < len(polys):
                        xs = [p[0] for p in polys[i]]
                        ys = [p[1] for p in polys[i]]
                        yield txt, min(xs), min(ys), max(xs), max(ys), conf
                    elif boxes is not None and i < len(boxes):
                        b = boxes[i]
                        yield txt, b[0], b[1], b[2], b[3], conf
        else:
            result = self.ocr.ocr(frame, cls=False)
            if not result or result[0] is None:
                return
            for line in result[0]:
                box, (txt, conf) = line
                xs = [p[0] for p in box]
                ys = [p[1] for p in box]
                yield txt, min(xs), min(ys), max(xs), max(ys), float(conf)

    def read(self, frame):
        out = []
        for txt, x1, y1, x2, y2, conf in self._lines(frame):
            words = txt.split()
            if not words:
                continue
            # Paddle returns line-level boxes; split into word boxes by
            # proportional width so per-word redaction stays tight.
            total = sum(len(w) for w in words) + (len(words) - 1)
            cursor = float(x1)
            for w in words:
                frac = (len(w) + 1) / max(total, 1)
                wx2 = cursor + (float(x2) - float(x1)) * frac
                out.append((w, (int(cursor), int(y1), int(wx2), int(y2)), conf))
                cursor = wx2
        return out


def read_adaptive(ocr, frame, mode="auto"):
    """OCR the frame; if the text is small (median word height < 15 px) or
    mode is 'on', re-OCR at 2x and keep whichever pass found more words.
    Upscaling helps small UI fonts but can hurt large text, so it's applied
    adaptively rather than blindly."""
    words = ocr.read(frame)
    if mode == "off":
        return words
    heights = sorted(b[3] - b[1] for _, b, _ in words) or [99]
    small = heights[len(heights) // 2] < 15
    if mode == "on" or small:
        # scale up for small text, but cap the result at ~4000 px on the
        # longest side — PaddleOCR resizes anything larger straight back
        # down, so exceeding it is pure waste
        scale = min(2.0, 4000.0 / max(frame.shape[:2]))
        if scale < 1.2:
            return words   # already near the cap: upscaling can't help
        big = cv2.resize(frame, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_CUBIC)
        w2 = [(t, (b[0] / scale, b[1] / scale, b[2] / scale, b[3] / scale), c)
              for t, b, c in ocr.read(big)]
        if len(w2) > len(words):
            return w2
    return words


def _ocr_selftest(backend):
    """One tiny inference at startup. GPU/cuDNN/driver failures surface at
    the first real kernel launch, not at import — so exercise a kernel HERE,
    where we can still fall back, instead of letting the first scan of a
    real job crash (seen: CUDNN error 5003 on a Paddle-GPU container)."""
    img = np.full((64, 256, 3), 255, np.uint8)
    cv2.putText(img, "TEST 123", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 0, 0), 2)
    backend.read(img)                    # must simply not raise
    return backend


def make_ocr(engine, device="auto"):
    if engine == "tesseract":
        return TesseractBackend()
    try:
        return _ocr_selftest(PaddleBackend(device=device))
    except Exception as e:
        print("      PaddleOCR failed its self-test: %s: %s"
              % (type(e).__name__, str(e)[:200]))
        if device != "cpu":
            try:
                print("      retrying PaddleOCR on CPU "
                      "(GPU/cuDNN/driver problems are the usual cause)…")
                return _ocr_selftest(PaddleBackend(device="cpu"))
            except Exception as e2:
                print("      PaddleOCR on CPU also failed: %s"
                      % str(e2)[:150])
        print("      falling back to Tesseract — the job continues on "
              "CPU OCR." + (" (--engine paddle was requested but is not "
                            "usable on this machine)" if engine == "paddle"
                            else ""))
        return TesseractBackend()


# ----------------------------------------------------------------------------
# Regex PHI detectors
# ----------------------------------------------------------------------------

RE_DATE = re.compile(
    r"""(?ix)\b(
        \d{1,2}[/\-.]\d{1,2}[/\-.](?:\d{4}|\d{2})
      | (?:19|20)\d{2}[/\-.]\d{1,2}[/\-.]\d{1,2}
      | (?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s+\d{1,2},?\s+(?:19|20)\d{2}
    )\b"""
)
RE_PHONE = re.compile(r"\(?\b\d{3}\)?[-. ]\d{3}[-. ]\d{4}\b")
RE_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
RE_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")
# API keys / tokens / secrets: common pref'd shapes + long high-entropy blobs.
RE_APIKEY = re.compile(r"""(?x)
    \b(?:
        sk-[A-Za-z0-9]{20,}                      # OpenAI-style
      | gh[pousr]_[A-Za-z0-9]{20,}               # GitHub tokens
      | xox[baprs]-[A-Za-z0-9-]{10,}             # Slack
      | AKIA[0-9A-Z]{16}                         # AWS access key id
      | AIza[0-9A-Za-z_\-]{35}                   # Google API key
      | ya29\.[0-9A-Za-z_\-]+                    # Google OAuth
      | eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}  # JWT
      | (?:api[_-]?key|secret|token|bearer)[=:\s"']{1,3}[A-Za-z0-9_\-]{16,}
    )\b""")
# generic high-entropy blob (>=32 chars) — but only if it mixes letters AND
# digits, so ordinary long words don't trip it.
RE_APIKEY_GENERIC = re.compile(r"\b(?=[A-Za-z0-9_\-]*\d)(?=[A-Za-z0-9_\-]*[A-Za-z])[A-Za-z0-9_\-]{32,}\b")
# Credit/debit card: 13-19 digits, optionally split by spaces or hyphens in
# groups. Major-brand prefixes keep it specific; the Luhn checksum below
# rejects random number strings (dates, IDs, phone runs) that happen to match.
RE_CARD = re.compile(r"""(?x)
    \b(?:
        4\d{3}                                   # Visa
      | 5[1-5]\d{2} | 2(?:2[2-9]\d|[3-6]\d\d|7[01]\d|720)  # Mastercard
      | 3[47]\d{2}                               # Amex (15 digits)
      | 6(?:011|5\d\d|4[4-9]\d)                  # Discover
      | 3(?:0[0-5]|[68]\d)\d                     # Diners
    )[ -]?\d{4}[ -]?\d{4}[ -]?\d{1,4}\b""")


def _luhn_ok(digits):
    """Luhn checksum: real card numbers pass; random digit runs almost never
    do (1-in-10 by chance, and the brand-prefix gate removes most of those)."""
    if not (13 <= len(digits) <= 19):
        return False
    total, alt = 0, False
    for ch in reversed(digits):
        d = ord(ch) - 48
        if alt:
            d *= 2
            if d > 9:
                d -= 9
        total += d
        alt = not alt
    return total % 10 == 0
# IPv4 (validated octets) and obvious IPv6.
RE_IP = re.compile(
    r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b"
    r"|\b(?:[A-Fa-f0-9]{1,4}:){2,7}[A-Fa-f0-9]{1,4}\b")
# Street address: a leading number, then words, ending in a street-type
# suffix (with or without a trailing abbreviation dot). Also matches secondary
# unit designators. Case-insensitive.
RE_STREET = re.compile(r"""(?ix)
    \b\d{1,6}\s+
    (?:[NSEW]\.?\s+|(?:north|south|east|west)\s+)?
    [A-Za-z0-9.'\-]+(?:\s+[A-Za-z0-9.'\-]+){0,4}?\s+
    (?:st|street|ave|avenue|blvd|boulevard|rd|road|dr|drive|ln|lane|ct|court|
       cir|circle|way|pl|place|ter|terrace|pkwy|parkway|hwy|highway|trl|trail|
       loop|pike|row|run|path|crossing|xing|square|sq)\.?
    (?:\s+(?:apt|apartment|suite|ste|unit|bldg|building|fl|floor|rm|room)\.?\s*
       \#?\s*\w+)?
    \b""")
# City, ST 12345  (the classic last line of a US address)
RE_CITYSTATEZIP = re.compile(
    r"(?i)\b[A-Za-z.\-]+(?:\s+[A-Za-z.\-]+)*,\s*"
    r"(?:AL|AK|AZ|AR|CA|CO|CT|DE|FL|GA|HI|ID|IL|IN|IA|KS|KY|LA|ME|MD|MA|MI|"
    r"MN|MS|MO|MT|NE|NV|NH|NJ|NM|NY|NC|ND|OH|OK|OR|PA|RI|SC|SD|TN|TX|UT|VT|"
    r"VA|WA|WV|WI|WY)\s+\d{5}(?:-\d{4})?\b")
# Secondary unit line on its own (address continuation):  "Apt 4B", "Suite 200"
RE_UNIT_LINE = re.compile(
    r"(?i)^\s*(?:apt|apartment|suite|ste|unit|bldg|building|fl|floor|rm|room|"
    r"#)\.?\s*\#?\s*\w+\s*$")
# A bare 5-digit (or ZIP+4) on its own line — a wrapped ZIP continuation.
RE_ZIP_LINE = re.compile(r"^\s*\d{5}(?:-\d{4})?\s*$")
# "City, ST" with the ZIP wrapped to the next line (continuation only).
RE_CITYSTATE_NOZIP = re.compile(
    r"(?i)^[A-Za-z.\-]+(?:\s+[A-Za-z.\-]+)*,\s*"
    r"(?:AL|AK|AZ|AR|CA|CO|CT|DE|FL|GA|HI|ID|IL|IN|IA|KS|KY|LA|ME|MD|MA|MI|"
    r"MN|MS|MO|MT|NE|NV|NH|NJ|NM|NY|NC|ND|OH|OK|OR|PA|RI|SC|SD|TN|TX|UT|VT|"
    r"VA|WA|WV|WI|WY)\s*$")
# Standalone ZIP+4 or 5-digit ZIP with a ZIP label nearby is weak on its own;
# we rely on the specific patterns above to avoid noise.

# Generic MRN token shape: a standalone 6-10 digit run, optionally with a
# short letter prefix (chart/system codes). detect_phi additionally requires
# a nearby MRN/chart/acct label OR 7+ digits before calling it an MRN, so
# this stays conservative. Sites with a known MRN format should tighten it
# via --mrn-regex (CLI) or the MRN regex field (web) for fewer false
# positives — e.g. \b\d{7}\b for an exact-width MRN.
RE_MRN_DEFAULT = r"^[A-Za-z]{0,3}\d{6,10}$"
RE_MRN_LABEL = re.compile(r"(?i)\b(mrn|med(?:ical)?\s*rec(?:ord)?|acct|account|chart)\b")
RE_NAME_LABEL = re.compile(
    r"(?i)\b(patient|name|pt|member|insured|guarantor|subscriber|responsible\s*party)\s*[:#\-]"
)

# Words that should never be treated as names (UI chrome, medical/scheduling
# vocab). Lowercase. Extend freely — over-including here only reduces
# false-positive blur, never PHI leakage, because real names still hit the
# label heuristic and NER.
STOPWORDS = {
    "patient", "patients", "name", "date", "birth", "phone", "chart", "home",
    "search", "provider", "appointment", "appointments", "visit", "visits",
    "office", "note", "notes", "history", "medications", "medication",
    "allergies", "allergy", "insurance", "today", "new", "open", "save",
    "cancel", "print", "close", "edit", "view", "help", "file", "clinic",
    "schedule", "mohs", "consult", "biopsy", "pathology", "derm",
    "dermatology", "results", "follow", "followup", "exam", "skin", "lesion",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday",
    "sunday", "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december", "morning",
    "afternoon", "refill", "request", "prior", "auth", "authorization",
    "pending", "review", "approved", "denied", "sent", "received", "inbox",
    "status", "active", "established", "est", "check", "checkout", "checkin",
    "room", "waiting", "billing", "claims", "settings", "admin", "user",
    "logout", "dashboard", "reports", "tasks", "messages", "fax", "faxes",
    "little", "rock", "clinton", "russellville", "west", "pinnacle",
    "suite", "street", "drive", "avenue", "road", "blvd",
    "am", "pm", "min", "mins", "hr", "hrs", "yes", "no",
}

HONORIFICS = {"mr", "mrs", "ms", "miss", "mx"}


# ----------------------------------------------------------------------------
# Line reconstruction (word boxes -> text lines with char->box mapping)
# ----------------------------------------------------------------------------

def group_lines(words):
    """Group OCR word boxes into lines. Returns list of dicts:
    {text, words: [(word, box, conf, char_start, char_end)]}"""
    if not words:
        return []
    items = sorted(words, key=lambda w: ((w[1][1] + w[1][3]) / 2, w[1][0]))
    lines = []
    for w in items:
        yc = (w[1][1] + w[1][3]) / 2
        h = max(w[1][3] - w[1][1], 1)
        placed = False
        for ln in lines:
            if abs(ln["yc"] - yc) < 0.7 * max(h, ln["h"]):
                ln["raw"].append(w)
                ln["yc"] = (ln["yc"] * (len(ln["raw"]) - 1) + yc) / len(ln["raw"])
                ln["h"] = max(ln["h"], h)
                placed = True
                break
        if not placed:
            lines.append({"raw": [w], "yc": yc, "h": h})
    out = []
    for ln in lines:
        ln["raw"].sort(key=lambda w: w[1][0])
        text = ""
        entries = []
        for w, box, conf in ln["raw"]:
            if text:
                text += " "
            start = len(text)
            text += w
            entries.append((w, box, conf, start, len(text)))
        out.append({"text": text, "words": entries})
    return out


# ----------------------------------------------------------------------------
# Name detection (no patient list required)
# ----------------------------------------------------------------------------

class NameDetector:
    def __init__(self, allow_names=None, extra_names=None, use_ner=True,
                 heuristic="auto"):
        self.allow = set()
        if allow_names:
            with open(allow_names, encoding="utf-8") as f:
                for line in f:
                    for tok in line.strip().replace(",", " ").split():
                        t = tok.strip(".").lower()
                        if t:
                            self.allow.add(t)
        self.extra = set()
        if extra_names:
            with open(extra_names, encoding="utf-8") as f:
                for line in f:
                    for tok in line.strip().replace(",", " ").split():
                        t = tok.lower()
                        if len(t) >= 2:
                            self.extra.add(t)
        self.nlp = None
        if use_ner:
            try:
                import spacy
                try:
                    self.nlp = spacy.load("en_core_web_sm")
                except OSError:
                    print("  WARNING: spaCy installed but model missing.\n"
                          "  Run:  python -m spacy download en_core_web_sm\n"
                          "  Falling back to heuristic name detection.")
            except ImportError:
                print("  WARNING: spaCy not installed — heuristic name "
                      "detection only.\n  For better accuracy: pip install "
                      "spacy && python -m spacy download en_core_web_sm")
        # heuristic: "auto" = on when NER unavailable; "on"/"off" force it
        self.heuristic = (heuristic == "on") or (heuristic == "auto" and self.nlp is None)

    def _allowed(self, word):
        return word.strip(".,:;()[]").lower() in self.allow

    @staticmethod
    def _namey(word):
        """Looks like a name token: alpha (plus - ' .), capitalized."""
        w = word.strip(".,:;()[]")
        if len(w) < 2 or not w[0].isupper():
            return False
        core = w.replace("-", "").replace("'", "")
        if not core.isalpha():
            return False
        return w.lower() not in STOPWORDS

    def find(self, lines):
        """Yield (box, matched_text) for name hits across reconstructed lines."""
        hits = []

        for ln in lines:
            words = ln["words"]

            # --- 1. spaCy NER ---
            if self.nlp is not None:
                doc = self.nlp(ln["text"])
                for ent in doc.ents:
                    if ent.label_ != "PERSON":
                        continue
                    for w, box, conf, s, e in words:
                        if s < ent.end_char and e > ent.start_char:
                            if self._namey(w) and not self._allowed(w):
                                hits.append((box, w))

            # --- 2. label heuristic: "Patient: John Smith DOB: ..." ---
            m = RE_NAME_LABEL.search(ln["text"])
            if m:
                started = False
                count = 0
                for w, box, conf, s, e in words:
                    if s < m.end():
                        continue
                    bare = w.strip(".,:;()[]")
                    # stop at the next label-ish token ("DOB:", "MRN:")
                    if (w.endswith(":") and started) or bare.lower() in (
                            "dob", "mrn", "phone", "sex", "gender", "age"):
                        break
                    if self._namey(w) or (bare and bare[0].isupper()):
                        if not self._allowed(w):
                            hits.append((box, w))
                        started = True
                        count += 1
                        if count >= 4:
                            break
                    elif started:
                        break

            # --- 3. extra names list (exact/substring) ---
            if self.extra:
                for w, box, conf, s, e in words:
                    if w.strip(".,:;()[]").lower() in self.extra:
                        hits.append((box, w))

            # --- 4. capitalized-pair heuristic (fallback) ---
            if self.heuristic:
                for i in range(len(words) - 1):
                    w1, b1 = words[i][0], words[i][1]
                    w2, b2 = words[i + 1][0], words[i + 1][1]
                    pair = False
                    if self._namey(w1) and self._namey(w2):
                        pair = True
                    # "Last, First"
                    elif w1.endswith(",") and self._namey(w1[:-1]) and self._namey(w2):
                        pair = True
                    # honorific + name: "Mrs. Henderson"
                    elif w1.strip(".").lower() in HONORIFICS and self._namey(w2):
                        if not self._allowed(w2):
                            hits.append((b2, w2))
                        continue
                    if pair:
                        if not self._allowed(w1):
                            hits.append((b1, w1))
                        if not self._allowed(w2):
                            hits.append((b2, w2))

        # dedupe by box
        seen = set()
        out = []
        for box, txt in hits:
            if box not in seen:
                seen.add(box)
                out.append((box, txt))
        return out


# ----------------------------------------------------------------------------
# PHI detection on one OCR'd frame
# ----------------------------------------------------------------------------

@dataclass
class Detection:
    t_start: float
    t_end: float
    cbox: tuple          # box in CONTENT coordinates (x1,y1,x2,y2)
    category: str
    text: str
    confidence: float
    aoff: tuple = (0.0, 0.0)   # cumulative offset when detected (drift anchor)
    last_seen: float = 0.0     # time of last positive sighting (t_end incl. hold)
    dense: bool = False        # per-frame dense-face detection: never merged
                               # across positions (it tracks a moving face)
    track: int = -1            # dense detections of the same physical object
                               # share a track id, so review shows ONE item
                               # per face instead of hundreds of frames
    person: int = -1           # face tracks clustered by facial IDENTITY
                               # (SFace embeddings): one review decision per
                               # PERSON, applied to all their appearances


class PhiMemory:
    """Remembers every string ever confirmed as PHI in this video. At each
    scan, all OCR'd words are checked against memory, so a name identified
    once gets blurred on every later appearance — anywhere on screen — even
    when NER/heuristics fail on that occurrence. Alpha strings match fuzzily
    (handles OCR misreads); numeric strings require same length with at most
    one differing digit (so benign numbers don't collide with MRNs)."""

    IMMEDIATE = {"dob", "phone", "ssn", "email", "mrn", "address",
                 "apikey", "ipaddr", "card", "plate"}

    def __init__(self, threshold=82, name_sightings=2):
        from rapidfuzz import fuzz
        self.fuzz = fuzz
        self.threshold = threshold
        self.name_sightings = name_sightings
        self.items = {}    # normalized text -> category
        self.counts = {}   # normalized text -> primary-detector sightings

    @staticmethod
    def norm(s):
        return s.strip(".,:;()[]").lower()

    def add(self, text, category, primary=True):
        if category in ("face", "manual"):
            return
        n = self.norm(text)
        if len(n) >= 3 and n not in STOPWORDS:
            self.items.setdefault(n, category)
            if primary:
                self.counts[n] = self.counts.get(n, 0) + 1

    def _gated(self, key, cat):
        """Names must be seen by a primary detector on name_sightings
        separate scans before memory starts recalling them — one bad
        NER hit shouldn't multiply across the whole video. Regex
        categories are high-precision and recall immediately."""
        if cat in self.IMMEDIATE:
            return cat
        return cat if self.counts.get(key, 0) >= self.name_sightings else None

    def recall(self, word):
        n = self.norm(word)
        if len(n) < 3 or n in STOPWORDS:
            return None
        if n in self.items:
            return self._gated(n, self.items[n])
        if n.isdigit():
            for k, cat in self.items.items():
                if (k.isdigit() and len(k) == len(n)
                        and sum(a != b for a, b in zip(k, n)) <= 1):
                    return self._gated(k, cat)
            return None
        if len(n) >= 4:
            for k, cat in self.items.items():
                if (not k.isdigit() and abs(len(k) - len(n)) <= 2
                        and self.fuzz.ratio(k, n) >= self.threshold):
                    return self._gated(k, cat)
        return None


def detect_phi(words, lines, t, offset, namer, mrn_re, custom_res=()):
    """offset = cumulative scroll (dx, dy) at this frame; boxes are converted
    to content coordinates by subtracting it. custom_res: sequence of
    (category_id, compiled_regex) for user-defined categories — each is
    checked independently of the built-in category chain."""
    dets = []
    ox, oy = offset

    def add(box, cat, txt, conf):
        cbox = (int(box[0] - ox), int(box[1] - oy), int(box[2] - ox), int(box[3] - oy))
        dets.append(Detection(t, t, cbox, cat, txt, round(float(conf), 3), (ox, oy)))

    for txt, box, conf in words:
        for cid, cre in custom_res:
            if cre.search(txt):
                add(box, cid, txt, conf)

    for i, (txt, box, conf) in enumerate(words):
        m_card = RE_CARD.search(txt)
        if m_card and _luhn_ok(re.sub(r"\D", "", m_card.group())):
            add(box, "card", txt, conf)
        elif RE_APIKEY.search(txt) or RE_APIKEY_GENERIC.search(txt):
            add(box, "apikey", txt, conf)
        elif RE_IP.search(txt):
            add(box, "ipaddr", txt, conf)
        elif RE_SSN.search(txt):
            add(box, "ssn", txt, conf)
        elif RE_EMAIL.search(txt):
            add(box, "email", txt, conf)
        elif RE_DATE.search(txt):
            add(box, "dob", txt, conf)
        elif RE_PHONE.search(txt):
            add(box, "phone", txt, conf)
        elif mrn_re.search(txt) or mrn_re.search(txt.strip(".,:;()[]")):
            digits = re.sub(r"\D", "", txt)
            near_label = any(
                RE_MRN_LABEL.search(w2)
                and abs((b2[1] + b2[3]) / 2 - (box[1] + box[3]) / 2) < (box[3] - box[1]) * 1.5
                for w2, b2, _ in words
            )
            if near_label or len(digits) >= 7:
                add(box, "mrn", txt, conf)

    # split-across-words card: "4111" "1111" "1111" "1111" (or 3 groups + amex)
    for i in range(len(words) - 2):
        for span in (4, 3):
            if i + span > len(words):
                continue
            grp = words[i:i + span]
            joined = "".join(re.sub(r"\D", "", g[0]) for g in grp)
            if (all(re.fullmatch(r"\d{3,6}", re.sub(r"\D", "", g[0])) for g in grp)
                    and RE_CARD.search(" ".join(g[0] for g in grp))
                    and _luhn_ok(joined)):
                bx = [g[1] for g in grp]
                add((min(b[0] for b in bx), min(b[1] for b in bx),
                     max(b[2] for b in bx), max(b[3] for b in bx)),
                    "card", joined, min(g[2] for g in grp))
                break

    # split-across-words phone: "(501)" "555-0142"
    for i in range(len(words) - 1):
        joined = words[i][0] + " " + words[i + 1][0]
        if RE_PHONE.search(joined) and not RE_PHONE.search(words[i][0]):
            b1, b2 = words[i][1], words[i + 1][1]
            add((min(b1[0], b2[0]), min(b1[1], b2[1]), max(b1[2], b2[2]), max(b1[3], b2[3])),
                "phone", joined, min(words[i][2], words[i + 1][2]))

    # split-across-words date: "Mar" "15," "1978"
    for i in range(len(words) - 2):
        trio = words[i:i + 3]
        joined = " ".join(w[0] for w in trio)
        if RE_DATE.search(joined) and not any(RE_DATE.search(w[0]) for w in trio):
            boxes = [w[1] for w in trio]
            add((min(b[0] for b in boxes), min(b[1] for b in boxes),
                 max(b[2] for b in boxes), max(b[3] for b in boxes)),
                "dob", joined, min(w[2] for w in trio))

    if namer is not None:      # None when the name category isn't selected
        for box, txt in namer.find(lines):
            add(box, "name", txt, 1.0)

    # Addresses span one to several stacked lines:
    #     111 Main St
    #     Apt 4B                (optional continuation)
    #     Little Rock, AR 72211
    # Detect the street line, then absorb the next 1-2 lines that look like
    # address continuations into ONE region, so a wrapped city/state/ZIP or a
    # unit line is covered as part of the same address. A city/state/ZIP line
    # standing alone (no street line above it) is still caught on its own.
    def _line_box(ln):
        bs = [e[1] for e in ln["words"]]
        if not bs:
            return None
        return (min(b[0] for b in bs), min(b[1] for b in bs),
                max(b[2] for b in bs), max(b[3] for b in bs))

    def _vgap_ok(a, b):
        # b is a plausible next line directly below a (allows ~1.8 line heights)
        if not a or not b:
            return False
        ah = a[3] - a[1]
        return 0 <= (b[1] - a[3]) <= 1.8 * max(ah, 1) and abs(b[0] - a[0]) < 6 * ah

    used = set()
    n = len(lines)
    for i, ln in enumerate(lines):
        if i in used:
            continue
        text = ln["text"]
        is_street = bool(RE_STREET.search(text))
        is_csz = bool(RE_CITYSTATEZIP.search(text))
        if not (is_street or is_csz):
            continue
        box = _line_box(ln)
        if box is None:
            continue
        parts_text = [text]
        used.add(i)
        if is_street:
            # absorb up to two following continuation lines
            j = i + 1
            absorbed = 0
            while j < n and absorbed < 2:
                nb = _line_box(lines[j])
                nt = lines[j]["text"]
                cont = (RE_UNIT_LINE.search(nt) or RE_ZIP_LINE.search(nt)
                        or RE_CITYSTATEZIP.search(nt)
                        or RE_CITYSTATE_NOZIP.search(nt))
                if cont and _vgap_ok(box, nb):
                    box = (min(box[0], nb[0]), min(box[1], nb[1]),
                           max(box[2], nb[2]), max(box[3], nb[3]))
                    parts_text.append(nt)
                    used.add(j)
                    absorbed += 1
                    # stop after we reach a city/state/ZIP (address is complete)
                    if RE_CITYSTATEZIP.search(nt):
                        break
                    j += 1
                else:
                    break
        add(box, "address", " / ".join(parts_text), 0.9)

    return dets


# ----------------------------------------------------------------------------
# Scroll tracking
# ----------------------------------------------------------------------------

def probe_camera_motion(path, sample_windows=14, pairs_per_window=4):
    """Screen recording or camera footage? Screen content moves along one
    axis at a time (scrolling) with long static stretches; handheld camera
    video drifts continuously on BOTH axes. Scroll tracking, content
    anchoring, and safety bands are built for the former and misfire badly
    on the latter (giant fake offsets -> edge bands and displaced boxes).

    Samples short windows spread across the WHOLE duration (not just the
    start), so tripod-then-pan footage is still recognized as camera.
    Returns (is_camera, moving_fraction, mixed_axis_fraction)."""
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    moving = mixed = pairs = 0
    win = None
    positions = ([int(total * i / sample_windows) for i in range(sample_windows)]
                 if total > sample_windows * (pairs_per_window + 1) else [0])
    for pos in positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        prev = None
        for _ in range(pairs_per_window + 1):
            ok, frame = cap.read()
            if not ok:
                break
            g = cv2.cvtColor(cv2.resize(frame, (320, max(2, int(
                frame.shape[0] * 320 / frame.shape[1])))),
                cv2.COLOR_BGR2GRAY).astype(np.float32)
            if prev is not None and prev.shape == g.shape:
                if win is None or win.shape != g.shape:
                    win = cv2.createHanningWindow(g.shape[::-1], cv2.CV_32F)
                (dx, dy), resp = cv2.phaseCorrelate(prev, g, win)
                if resp >= 0.08:
                    pairs += 1
                    if abs(dx) >= 0.4 or abs(dy) >= 0.4:
                        moving += 1
                        if abs(dx) >= 0.4 and abs(dy) >= 0.4:
                            mixed += 1
            prev = g
    cap.release()
    if pairs == 0:
        return False, 0.0, 0.0
    mov_f = moving / pairs
    mix_f = (mixed / moving) if moving else 0.0
    # camera = a solid share of sampled pairs move, and that motion is
    # 2-axis. A perfectly static camera scores like a static screen —
    # which is fine: zero motion means zero offsets and zero bands, so
    # screen mode is harmless there.
    return (mov_f > 0.35 and mix_f > 0.5), mov_f, mix_f


class ScrollTracker:
    """Estimates cumulative global (dx, dy) content motion via phase
    correlation against a KEYFRAME (the frame at the last OCR scan), not
    frame-to-frame. This bounds drift to a single sub-pixel measurement per
    scan epoch instead of accumulating error every frame. Sign convention
    verified: content moving UP on screen => dy negative.

    Call step(frame) every frame (returns cumulative offset); call anchor()
    right after each OCR scan to re-key."""

    def __init__(self, width=640):
        self.width = width
        self.key = None          # keyframe gray
        self.key_cum = (0.0, 0.0)
        self.prev = None
        self.cum = (0.0, 0.0)
        self.win = None
        self.inv_scale = 1.0

    def _prep(self, frame):
        h, w = frame.shape[:2]
        scale = self.width / w
        small = cv2.resize(frame, (self.width, max(2, int(h * scale))))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if self.win is None or self.win.shape != gray.shape:
            self.win = cv2.createHanningWindow(gray.shape[::-1], cv2.CV_32F)
        self.inv_scale = 1.0 / scale
        return gray

    def step(self, frame):
        gray = self._prep(frame)
        if self.key is None:
            self.key = gray
            self.prev = gray
            return self.cum

        h, w = gray.shape
        MAX_STEP = 250.0   # px/frame: above any smooth scroll. Bigger implied
                           # jumps are treated as content REPLACEMENT (dialog,
                           # page load) — blur boxes must NOT move for those.
        last = self.cum
        (dx, dy), resp = cv2.phaseCorrelate(self.key, gray, self.win)
        if resp >= 0.12 and abs(dy) < 0.35 * h and abs(dx) < 0.35 * w:
            ox = self.key_cum[0] + dx * self.inv_scale
            oy = self.key_cum[1] + dy * self.inv_scale
            if abs(ox - self.key_cum[0]) < 1.0:
                ox = self.key_cum[0]
            if abs(oy - self.key_cum[1]) < 1.0:
                oy = self.key_cum[1]
            if (abs(ox - last[0]) > MAX_STEP
                    or abs(oy - last[1]) > MAX_STEP):
                # implausible single-frame jump: scene change, not scroll —
                # hold the offset and re-key
                self.key = gray
                self.key_cum = self.cum
            else:
                self.cum = (ox, oy)
                # if we've moved far from the key, re-anchor so correlation
                # overlap stays healthy on long continuous scrolls
                if abs(dy) > 0.28 * h or abs(dx) > 0.28 * w:
                    self.key = gray
                    self.key_cum = self.cum
        else:
            # keyframe correlation failed (scene cut / popup). An incremental
            # frame-to-frame measurement across a visual discontinuity is
            # untrustworthy: demand HIGH confidence and a plausible motion
            # magnitude, otherwise treat as content replacement and hold the
            # offset — spurious jumps here slide every blur box off its text.
            (dx, dy), resp2 = cv2.phaseCorrelate(self.prev, gray, self.win)
            mx = dx * self.inv_scale
            my = dy * self.inv_scale
            if (resp2 >= 0.30 and abs(mx) <= MAX_STEP
                    and abs(my) <= MAX_STEP):
                if abs(mx) < 1.0:
                    mx = 0.0
                if abs(my) < 1.0:
                    my = 0.0
                self.cum = (self.cum[0] + mx, self.cum[1] + my)
            self.key = gray
            self.key_cum = self.cum
        self.prev = gray
        return self.cum

    def anchor(self):
        """Re-key on the current frame (call right after an OCR scan)."""
        if self.prev is not None:
            self.key = self.prev
            self.key_cum = self.cum


# ----------------------------------------------------------------------------
# Temporal merge (in content coordinates)
# ----------------------------------------------------------------------------

def boxes_overlap(a, b, slack=12):
    return not (a[2] + slack < b[0] or b[2] + slack < a[0]
                or a[3] + slack < b[1] or b[3] + slack < a[1])


def _box_iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def assign_dense_tracks(dets, max_gap=0.6, reach=1.6):
    """Group dense per-frame detections into tracks: consecutive samples of
    the same physical object (a face crossing the frame) get one track id.
    Purely additive metadata — rendering still uses each per-frame box; the
    review UI collapses a track into a single keep/blur decision."""
    tracks = []          # [id, category, last_t, last_box]
    next_id = 0
    for d in sorted((x for x in dets if getattr(x, "dense", False)),
                    key=lambda x: x.t_start):
        bx = d.cbox
        cx0 = (bx[0] + bx[2]) / 2
        cy0 = (bx[1] + bx[3]) / 2
        size = max(bx[2] - bx[0], bx[3] - bx[1], 1)
        best = None
        for tr in tracks:
            if tr[1] != d.category or d.t_start - tr[2] > max_gap:
                continue
            lb = tr[3]
            lcx = (lb[0] + lb[2]) / 2
            lcy = (lb[1] + lb[3]) / 2
            dist = ((cx0 - lcx) ** 2 + (cy0 - lcy) ** 2) ** 0.5
            if dist <= reach * size and (best is None or dist < best[0]):
                best = (dist, tr)
        if best is None:
            tracks.append([next_id, d.category, d.t_start, bx])
            d.track = next_id
            next_id += 1
        else:
            tr = best[1]
            tr[2], tr[3] = d.t_start, bx
            d.track = tr[0]
    return next_id


def smooth_dense_tracks(dets, fps, video, cum=None, win_start=0.0, cb=None):
    """Make each dense track leak-free from true first appearance to exit.

    Dense samples are instantaneous per-frame boxes; three gaps remain
    between them and continuous cover of a moving object:
      1. detector flicker — frames mid-track where the detector missed the
         object. The box is INTERPOLATED between the surrounding samples, so
         the blur moves with the object instead of vanishing (or hanging at
         a stale position).
      2. onset — the detector needs a few clear frames before its first hit,
         exposing the object as it enters. The first sample's pixels are
         template-matched BACKWARD through the file (the same visual match
         deep backtrack uses) and synthetic samples are added down to the
         earliest frame that still matches, plus a short unconditional
         grace pad below detection threshold.
      3. exit — mirror grace pad after the last sample.
    Every addition is a dense sample on the same track id, so review still
    shows one card per physical object. Fail closed: only ever adds cover.
    Returns (interpolated_gaps, leadin_samples, leadin_seconds)."""
    GRACE = 0.25        # s of unconditional pad at track onset/exit
    CHAIN_MAX = 0.75    # s: longest flicker gap interpolated (matches the
    #                     assign_dense_tracks max_gap, with slack)
    LEAD_MAX = 4.0      # s: farthest the onset walk seeks back
    SCALE = 0.5         # match deep backtrack's working resolution
    THR = 0.58          # TM_CCOEFF_NORMED bar (same as face backtrack)
    frame_period = 1.0 / max(fps, 1.0)

    tracks = {}
    for d in dets:
        if getattr(d, "dense", False) and getattr(d, "track", -1) >= 0:
            tracks.setdefault(d.track, []).append(d)
    if not tracks:
        return (0, 0, 0.0)

    def _off(t):
        if not cum:
            return (0.0, 0.0)
        return cum[min(int(t * fps), len(cum) - 1)]

    def _screen(d):
        return (d.cbox[0] + d.aoff[0], d.cbox[1] + d.aoff[1],
                d.cbox[2] + d.aoff[0], d.cbox[3] + d.aoff[1])

    def _mk(t0, t1, sbox, ref, tid):
        o = _off(t0)
        return Detection(t0, t1,
                         (int(sbox[0] - o[0]), int(sbox[1] - o[1]),
                          int(sbox[2] - o[0]), int(sbox[3] - o[1])),
                         ref.category, ref.text, ref.confidence, o,
                         last_seen=t0, dense=True, track=tid)

    cap = cv2.VideoCapture(video) if video else None
    if cap is not None and not cap.isOpened():
        cap = None

    def _gray(t):
        if cap is None:
            return None
        cap.set(cv2.CAP_PROP_POS_MSEC, max(t, 0.0) * 1000)
        ok, fr = cap.read()
        if not ok:
            return None
        g = cv2.resize(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), None,
                       fx=SCALE, fy=SCALE)
        # light smoothing before matching: sub-pixel motion at this scale
        # decorrelates fine texture (measured 1.0 -> 0.48 on a half-pixel
        # offset), while a smoothed match stays >0.8 present and ~0 absent
        return cv2.GaussianBlur(g, (3, 3), 0)

    # This is the longest silent stretch of a dense scan (the onset walk-back
    # re-reads the video per track), so report progress: the bar moves via
    # the "post" stage and the log shows a time-gated track counter.
    cb = cb or Callbacks()
    n_total = len(tracks)
    _last_log = time.time()
    n_gaps, n_lead, lead_s = 0, 0, 0.0
    added = []
    for n_done, (tid, samples) in enumerate(tracks.items()):
        cb.progress("post", n_done, n_total)
        if time.time() - _last_log >= 3.0:
            _last_log = time.time()
            cb.log("        …track %d/%d (matching each track's onset "
                   "backward through the video)" % (n_done + 1, n_total))
        samples.sort(key=lambda d: d.t_start)

        # 1. flicker gaps: chain, and interpolate the box across the gap
        for a, b in zip(samples, samples[1:]):
            gap = b.t_start - a.t_start
            if gap <= frame_period * 1.5:
                a.t_end = max(a.t_end, b.t_start)
                continue
            if gap > CHAIN_MAX:
                continue        # sustained absence: never bridge blind
            A, B = _screen(a), _screen(b)
            steps = min(12, max(1, int(round(gap / (2 * frame_period)))))
            ts = [a.t_start + gap * i / steps for i in range(steps + 1)]
            pos = [tuple(A[k] + (B[k] - A[k]) * i / steps for k in range(4))
                   for i in range(steps + 1)]
            a.t_end = max(a.t_end, ts[1] if steps > 1 else b.t_start)
            for i in range(1, steps):
                # union with the next step's box so movement WITHIN the
                # step stays covered
                u = tuple(min(pos[i][k], pos[i + 1][k]) if k < 2 else
                          max(pos[i][k], pos[i + 1][k]) for k in range(4))
                added.append(_mk(ts[i], ts[i + 1], u, a, tid))
            n_gaps += 1

        # 2. onset: template-match the first sample backwards to the
        # object's true first visible frame
        first = samples[0]
        g0 = _gray(first.t_start)
        walked = first
        if g0 is not None:
            sb = _screen(first)
            x1, y1 = int(sb[0] * SCALE), int(sb[1] * SCALE)
            x2, y2 = int(sb[2] * SCALE), int(sb[3] * SCALE)
            gh, gw = g0.shape[:2]
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(gw, x2), min(gh, y2)
            tmpl = g0[y1:y2, x1:x2] if (x2 - x1 >= 8 and y2 - y1 >= 8) else None
            if tmpl is not None and float(tmpl.std()) > 4:
                th_, tw_ = tmpl.shape
                box = list(sb)
                t = first.t_start
                step = 2 * frame_period
                while (first.t_start - t < LEAD_MAX
                       and t - step >= max(0.0, win_start - 0.01)):
                    t -= step
                    g = _gray(t)
                    if g is None:
                        break
                    m = int(max(10, 0.6 * max(tw_, th_)))
                    rx1 = max(0, int(box[0] * SCALE) - m)
                    ry1 = max(0, int(box[1] * SCALE) - m)
                    rx2 = min(g.shape[1], int(box[2] * SCALE) + m)
                    ry2 = min(g.shape[0], int(box[3] * SCALE) + m)
                    if rx2 - rx1 < tw_ or ry2 - ry1 < th_:
                        break   # clipped at the frame edge: object entering
                    res = cv2.matchTemplate(g[ry1:ry2, rx1:rx2], tmpl,
                                            cv2.TM_CCOEFF_NORMED)
                    _, mx, _, loc = cv2.minMaxLoc(res)
                    if mx < THR:
                        break   # genuinely not there yet
                    nx = (rx1 + loc[0]) / SCALE
                    ny = (ry1 + loc[1]) / SCALE
                    box = [nx, ny, nx + (sb[2] - sb[0]), ny + (sb[3] - sb[1])]
                    walked = _mk(t, t + step, tuple(box), first, tid)
                    added.append(walked)
                    n_lead += 1
                    lead_s += step

        # 3. grace pads: cover the sub-threshold sliver at both ends
        pre = walked.t_start
        walked.t_start = max(0.0, win_start, walked.t_start - GRACE)
        lead_s += pre - walked.t_start
        samples[-1].t_end += GRACE

    if cap is not None:
        cap.release()
    cb.progress("post", n_total, n_total)
    dets.extend(added)
    return (n_gaps, n_lead, lead_s)


def group_persons(dets, video, cb=None):
    """Cluster dense FACE tracks by facial IDENTITY so review shows one
    card per PERSON — one blur/keep decision applied to every appearance.
    Nobody blurs a face in one clip and leaves the same face visible in
    another; per-person is the decision users are actually making.

    Uses SFace embeddings (OpenCV zoo, Apache-2.0, auto-downloaded ~38 MB
    like YuNet) aligned via YuNet landmarks on each track's best frames.
    The 0.40 cosine threshold is CONSERVATIVE (same person measures ~0.9,
    different people ~0.0-0.35): a missed merge only shows an extra card,
    but a wrong merge could hide someone inside a kept person. Tracks
    where no face embeds (junk detections, extreme profiles, faces too
    small to identify) keep person=-1 and stay individual cards.

    Three defenses against WRONG merges (each validated on real crowd
    footage — a news studio with ~20 schoolchildren, where the original
    single-link union-find at 0.40 merged 83% of all face samples into
    ONE review card, hiding most of the room behind a single thumbnail):
      1. TEMPORAL CANNOT-LINK: tracks co-visible in the same frames for
         >0.5s are different people BY DEFINITION and never merge, no
         matter how similar their embeddings — the strongest signal, and
         model-free. (Embeddings measurably fail on similar-age children
         at broadcast resolution; co-visibility does not.) The 0.5s
         tolerance absorbs boundary flicker; the rare true dual
         appearance (a monitor wall showing the anchor) just costs an
         extra card — fail closed.
      2. CENTROID-linkage, not single-link: a track joins a cluster only
         if it matches the cluster's AVERAGE identity, so one noisy
         embedding can't chain strangers together.
      3. Embeddings only from re-detected faces >=32 px across — SFace on
         smaller crops is noise that links strangers.
    Cosine threshold 0.55: children of similar age measure 0.4-0.6 apart
    (adults ~0.0-0.35), so the old "conservative" 0.40 merged different
    kids; same-person tracks measure ~0.9 and still group fine. On the
    validation video: 21 persons in a room of ~22, every surviving merge
    a genuine re-appearance. Returns (embedded_tracks, n_persons)."""
    cb = cb or Callbacks()
    tracks = {}
    for d in dets:
        if getattr(d, "dense", False) and d.category == "face" \
                and getattr(d, "track", -1) >= 0:
            tracks.setdefault(d.track, []).append(d)
    if not tracks:
        return (0, 0)
    if not (hasattr(cv2, "FaceRecognizerSF_create")
            and hasattr(cv2, "FaceDetectorYN_create")):
        return (0, 0)
    mdir = _model_dir()
    sface = os.path.join(mdir, "face_recognition_sface_2021dec.onnx")
    yunet = os.path.join(mdir, "face_detection_yunet_2023mar.onnx")
    import urllib.request
    if not os.path.exists(sface) or os.path.getsize(sface) < 10000:
        cb.log("      downloading SFace identity model (~38 MB, one time)…")
        urllib.request.urlretrieve(SFACE_URL, sface)
    if not os.path.exists(yunet) or os.path.getsize(yunet) < 10000:
        urllib.request.urlretrieve(YUNET_URL, yunet)
    rec = _make_sface(sface)
    det = _make_yunet(yunet, (320, 320), 0.5)
    cap = cv2.VideoCapture(video)
    cb.log("      person grouping: matching %d face track(s) by identity…"
           % len(tracks))
    _last_log = time.time()
    embs = {}
    for n_done, (tid, samples) in enumerate(tracks.items()):
        if time.time() - _last_log >= 3.0:
            _last_log = time.time()
            cb.log("        …track %d/%d embedded"
                   % (n_done + 1, len(tracks)))
        best = sorted(samples, key=lambda d: -d.confidence)[:3]
        feats = []
        for d in best:
            t = min(max(d.last_seen, d.t_start), d.t_end)
            cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t) * 1000)
            ok, fr = cap.read()
            if not ok:
                continue
            det.setInputSize((fr.shape[1], fr.shape[0]))
            _, rows = det.detect(fr)
            if rows is None:
                continue
            sb = (d.cbox[0] + d.aoff[0], d.cbox[1] + d.aoff[1],
                  d.cbox[2] + d.aoff[0], d.cbox[3] + d.aoff[1])
            rbest, riou = None, 0.2
            for r in rows:
                rb = (r[0], r[1], r[0] + r[2], r[1] + r[3])
                iou = _box_iou(rb, sb)
                if iou > riou:
                    rbest, riou = r, iou
            if rbest is None:
                continue
            if float(rbest[2]) < 32 or float(rbest[3]) < 32:
                continue    # too small to identify: noise embedding
            f = rec.feature(rec.alignCrop(fr, rbest)).flatten()
            n = float(np.linalg.norm(f))
            if n > 0:
                feats.append(f / n)
        if feats:
            e = np.mean(feats, axis=0)
            embs[tid] = e / np.linalg.norm(e)
    cap.release()
    # centroid-linkage: big stable tracks seed clusters; each remaining
    # track joins only if it matches the cluster's AVERAGE identity. This
    # cannot chain through one noisy link the way single-link union-find
    # did (crowd footage merged most of the room into one "person").
    spans = {tid: (min(d.t_start for d in ds), max(d.t_end for d in ds))
             for tid, ds in tracks.items()}

    def _covis(a, b):
        return max(0.0, min(spans[a][1], spans[b][1])
                   - max(spans[a][0], spans[b][0]))

    order = sorted(embs, key=lambda t: (-len(tracks[t]), t))
    clusters = []            # [running_sum_vector, [track ids]]
    for t in order:
        e = embs[t]
        best, best_s = None, 0.55
        for c in clusters:
            # temporal cannot-link: on-screen at the same time for >0.5s
            # means different people, whatever the embeddings say
            if any(_covis(t, m) > 0.5 for m in c[1]):
                continue
            cen = c[0] / np.linalg.norm(c[0])
            s = float(np.dot(cen, e))
            if s >= best_s:
                best, best_s = c, s
        if best is None:
            clusters.append([e.copy(), [t]])
        else:
            best[0] = best[0] + e
            best[1].append(t)
    for pid, c in enumerate(clusters):
        for t in c[1]:
            for d in tracks[t]:
                d.person = pid
    return (len(embs), len(clusters))


def merge_detections(dets, hold, scans=None, bridge_gap=4.0, fuzz=None,
                     gap_check=None):
    """Chain detections of the same category whose content boxes overlap.

    Short gaps (within `hold`) chain unconditionally, as before. Longer gaps
    up to `bridge_gap` seconds are BRIDGED — kept blurred straight through —
    unless an intermediate scan positively saw different, readable text in
    that region (i.e. the content genuinely changed). An empty or unreadable
    region during the gap is treated as an OCR miss and stays covered:
    fail closed, never flash PHI."""
    dets = sorted(dets, key=lambda d: d.t_start)
    merged = []

    def contradicted(m, t_from, t_to):
        """True only if the gap contains STABLE different text — the same
        different string read on two or more scans. A single divergent read
        is far more likely to be the mouse cursor sitting over the word (or
        another transient occlusion) garbling OCR than genuinely new content:
        real replacement text reads consistently, cursor garble varies every
        scan. Fail closed — an unstable read keeps the region blurred."""
        if not scans:
            return False
        mx1, my1, mx2, my2 = m.cbox
        seen = {}
        for st, _cum, words in scans:
            if not (t_from + 0.01 < st < t_to - 0.01):
                continue
            for txt, (x1, y1, x2, y2), conf in words:
                if conf < 0.6:
                    continue
                cxm = (x1 + x2) / 2
                cym = (y1 + y2) / 2
                if mx1 - 6 <= cxm <= mx2 + 6 and my1 - 4 <= cym <= my2 + 4:
                    n = PhiMemory.norm(txt)
                    if fuzz:
                        mt = PhiMemory.norm(m.text)
                        # partial_ratio catches cursor-occluded reads of the
                        # SAME word ("errin" ~ "herrin"): those are evidence
                        # the word is still there, never evidence it changed
                        same = (fuzz.ratio(n, mt) >= 70
                                or fuzz.partial_ratio(n, mt) >= 85)
                    else:
                        same = False
                    if not same and len(n) >= 3:
                        seen[n] = seen.get(n, 0) + 1
                        if seen[n] >= 2:
                            return True
        return False

    for d in dets:
        d.last_seen = d.t_start
        if not getattr(d, "dense", False):
            # dense samples keep their sub-frame hold: stamping them with the
            # multi-second OCR hold leaves every PAST position blurred for
            # `hold` seconds — a trail of stale boxes marching away from a
            # moving face. Their continuity across detector flicker and the
            # onset gap is handled per-track by smooth_dense_tracks().
            d.t_end = d.t_start + hold
        for m in reversed(merged):
            if m.category != d.category or not boxes_overlap(m.cbox, d.cbox):
                continue
            if getattr(d, "dense", False) or getattr(m, "dense", False):
                # dense face boxes are per-frame position samples of a possibly
                # moving face — never merge them, or the bounding box balloons
                # to cover the whole path. Each stands alone with its short
                # hold, so the blur rides the face frame by frame.
                continue
            # different readable text in the same spot is a DIFFERENT object
            # (e.g. names sliding through one row of an inner-scrolling list)
            # — never fuse them, or the region's text and its frames diverge
            if fuzz and m.text and d.text:
                _mn = PhiMemory.norm(m.text)
                _dn = PhiMemory.norm(d.text)
                if (fuzz.ratio(_mn, _dn) < 70
                        and fuzz.partial_ratio(_mn, _dn) < 85):
                    continue
            gap = d.t_start - m.last_seen
            ok = gap <= hold
            if not ok and not contradicted(m, m.last_seen, d.t_start):
                if gap <= bridge_gap:
                    ok = True
                elif gap_check is not None:
                    # beyond the configured bridge: ask the pixels. The file
                    # is checked at points inside the gap — bridge any length
                    # of gap the content verifiably persisted through, refuse
                    # if it visibly changed. The knob stops mattering.
                    ok = gap_check(m, m.last_seen, d.t_start)
            if ok:
                m.last_seen = max(m.last_seen, d.t_start)
                m.t_end = max(m.t_end, d.t_end)
                m.cbox = (min(m.cbox[0], d.cbox[0]), min(m.cbox[1], d.cbox[1]),
                          max(m.cbox[2], d.cbox[2]), max(m.cbox[3], d.cbox[3]))
                break
        else:
            merged.append(d)
    return merged


# ----------------------------------------------------------------------------
# Render
# ----------------------------------------------------------------------------

def blur_region(frame, x1, y1, x2, y2, mode, shape="rect"):
    h, w = frame.shape[:2]
    x1 = max(0, int(x1)); y1 = max(0, int(y1))
    x2 = min(w, int(x2)); y2 = min(h, int(y2))
    if x2 <= x1 or y2 <= y1:
        return
    roi = frame[y1:y2, x1:x2]
    if mode == "box":
        filled = np.zeros_like(roi)
    elif mode == "mosaic":
        # fragment size scales with the region (~14 tiles across) so small
        # and large faces pixelate consistently — deface sizes fragments in
        # absolute pixels and its users ask for exactly this (issue #60)
        fw = max(2, (x2 - x1) // 14)
        small = cv2.resize(roi, (max(1, (x2 - x1) // fw),
                                 max(1, (y2 - y1) // fw)),
                           interpolation=cv2.INTER_LINEAR)
        filled = cv2.resize(small, (x2 - x1, y2 - y1),
                            interpolation=cv2.INTER_NEAREST)
    else:
        k = max(31, (((x2 - x1) // 3) | 1))
        filled = cv2.GaussianBlur(roi, (k, k), 0)
    if shape == "ellipse":
        # elliptical mask hugs a face: no smeared background corners, which
        # is most of why box-blurred faces read as "whole body blurred"
        rw, rh = x2 - x1, y2 - y1
        mask = np.zeros(roi.shape[:2], np.uint8)
        cv2.ellipse(mask, (rw // 2, rh // 2),
                    (max(1, rw // 2), max(1, rh // 2)),
                    0, 0, 360, 255, -1)
        # A face cut off by the frame border continues PAST that border, but
        # the inscribed ellipse above pulls AWAY from it — leaving the
        # region's border-side corners unblurred (the boat-video top-of-frame
        # face leak). For every border the region touches, union in a second
        # ellipse whose virtual box mirrors past that border: its visible
        # part is a half-ellipse that stays full-size AT the border. The
        # union only ever adds coverage — fail closed.
        vx1, vy1, vx2, vy2 = 0, 0, rw, rh
        if x1 <= 1: vx1 = -rw
        if y1 <= 1: vy1 = -rh
        if x2 >= w - 1: vx2 = 2 * rw
        if y2 >= h - 1: vy2 = 2 * rh
        if (vx1, vy1, vx2, vy2) != (0, 0, rw, rh):
            cv2.ellipse(mask, ((vx1 + vx2) // 2, (vy1 + vy2) // 2),
                        (max(1, (vx2 - vx1) // 2), max(1, (vy2 - vy1) // 2)),
                        0, 0, 360, 255, -1)
        roi[mask > 0] = filled[mask > 0]
    else:
        frame[y1:y2, x1:x2] = filled


class PipelineCancelled(Exception):
    """Raised internally when a Callbacks.cancelled() returns True."""


class Callbacks:
    """Hooks for embedding the pipeline (GUI, batch runner, tests).
    The default implementation reproduces the CLI's print behavior."""
    wants_frames = False   # set True to receive scan_frame() calls

    def log(self, msg):
        print(msg, flush=True)

    def progress(self, stage, current, total):
        pass  # stage is "scan" or "render"

    def scan_frame(self, frame_bgr, t, found):
        pass  # annotated copy of the frame just OCR'd (only if wants_frames)

    def cancelled(self):
        return False


def nvenc_available(encoder_pref, cb):
    """Pick the video encoder. Pre-flight tests NVENC with a tiny encode so
    we fail fast instead of discovering a broken NVENC after a full render."""
    if encoder_pref == "x264":
        return "libx264"
    if not shutil.which("ffmpeg"):
        return None
    try:
        listed = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=15).stdout
    except Exception:
        listed = ""
    if "h264_nvenc" not in listed:
        cb.log("      note: h264_nvenc not present in this ffmpeg build — using libx264.\n"
               "            Install a full build (e.g. `winget install Gyan.FFmpeg`) for GPU encoding.")
        return "libx264"
    try:
        # 30 frames: NVENC buffers frames internally (B-frames/lookahead),
        # so a too-short test can emit zero packets on a WORKING encoder
        test = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error",
             "-f", "lavfi", "-i", "color=black:s=256x256:r=30", "-t", "1",
             "-c:v", "h264_nvenc", "-f", "null", "-"],
            capture_output=True, text=True, timeout=60)
        if test.returncode == 0:
            return "h264_nvenc"
        err = (test.stderr or "").strip() or "(no error output)"
        cb.log("      note: h264_nvenc failed its test encode — using libx264. ffmpeg said:")
        for line in err.splitlines()[-12:]:
            cb.log(f"            {line}")
    except Exception as e:
        cb.log(f"      note: NVENC test errored ({e}) — using libx264")
    return "libx264"


_HEVC10 = {}


def hevc10_encoder(encoder="auto", cb=None):
    """-> "hevc_nvenc" | "libx265" | None. Which 10-bit HEVC encoder this
    machine can actually run (verified with a real test encode) — needed to
    PRESERVE HDR output. encoder="x264" (the CPU choice) skips the GPU."""
    cb = cb or Callbacks()
    order = ["libx265"] if encoder == "x264" else ["hevc_nvenc", "libx265"]
    key = tuple(order)
    if key in _HEVC10:
        return _HEVC10[key]
    found = None
    for enc in order:
        pixfmt = "p010le" if enc == "hevc_nvenc" else "yuv420p10le"
        try:
            t = subprocess.run(
                ["ffmpeg", "-hide_banner", "-loglevel", "error",
                 "-f", "lavfi", "-i", "color=black:s=256x256:r=30", "-t", "1",
                 "-c:v", enc, "-pix_fmt", pixfmt, "-f", "null", "-"],
                capture_output=True, text=True, timeout=60)
            if t.returncode == 0:
                found = enc
                break
        except Exception:
            continue
    _HEVC10[key] = found
    return found


def color_tags(path):
    """-> dict of the stream's color metadata (only the tags that are set).
    Used to stamp HDR output with the same primaries/transfer as the input."""
    if not shutil.which("ffprobe"):
        return {}
    try:
        p = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                            "-show_entries",
                            "stream=color_transfer,color_primaries,color_space",
                            "-of", "json", path],
                           capture_output=True, text=True, timeout=30)
        st = json.loads(p.stdout)["streams"][0]
    except Exception:
        return {}
    out = {}
    for k, flag in (("color_primaries", "-color_primaries"),
                    ("color_transfer", "-color_trc"),
                    ("color_space", "-colorspace")):
        v = st.get(k)
        if v and v != "unknown":
            out[flag] = v
    return out


def _blur_yuv10(y, u, v, x1, y1, x2, y2, mode, shape="rect"):
    """blur_region for a 10-bit planar YUV420 frame: Y full-res, U/V
    half-res. Working in the native YUV domain means untouched pixels never
    go through ANY colorspace conversion — the HDR signal passes straight
    through."""
    h, w = y.shape
    x1 = max(0, int(x1)); y1 = max(0, int(y1))
    x2 = min(w, int(x2)); y2 = min(h, int(y2))
    if x2 <= x1 or y2 <= y1:
        return
    # which frame borders the region touches, decided ONCE on the full-res
    # luma coords (the chroma planes are half-res, so per-plane checks would
    # misfire). Used by the ellipse mask below — see blur_region for why.
    edge_l, edge_t = x1 <= 1, y1 <= 1
    edge_r, edge_b = x2 >= w - 1, y2 >= h - 1

    def _fill(plane, px1, py1, px2, py2, black, kmin):
        if px2 <= px1 or py2 <= py1:
            return
        roi = plane[py1:py2, px1:px2]
        if mode == "box":
            filled = np.full_like(roi, black)
        elif mode == "mosaic":
            fw = max(2, (px2 - px1) // 14)
            small = cv2.resize(roi, (max(1, (px2 - px1) // fw),
                                     max(1, (py2 - py1) // fw)),
                               interpolation=cv2.INTER_LINEAR)
            filled = cv2.resize(small, (px2 - px1, py2 - py1),
                                interpolation=cv2.INTER_NEAREST)
        else:
            k = max(kmin, (((px2 - px1) // 3) | 1))
            filled = cv2.GaussianBlur(roi, (k, k), 0)
        if shape == "ellipse":
            rw, rh = px2 - px1, py2 - py1
            mask = np.zeros(roi.shape, np.uint8)
            cv2.ellipse(mask, (rw // 2, rh // 2),
                        (max(1, rw // 2), max(1, rh // 2)),
                        0, 0, 360, 255, -1)
            # border-touching face: union in the mirrored ellipse so the
            # visible half stays full-size at the frame border instead of
            # pinching away from it (same fix as blur_region — fail closed)
            vx1, vy1, vx2, vy2 = 0, 0, rw, rh
            if edge_l: vx1 = -rw
            if edge_t: vy1 = -rh
            if edge_r: vx2 = 2 * rw
            if edge_b: vy2 = 2 * rh
            if (vx1, vy1, vx2, vy2) != (0, 0, rw, rh):
                cv2.ellipse(mask, ((vx1 + vx2) // 2, (vy1 + vy2) // 2),
                            (max(1, (vx2 - vx1) // 2),
                             max(1, (vy2 - vy1) // 2)),
                            0, 0, 360, 255, -1)
            roi[mask > 0] = filled[mask > 0]
        else:
            plane[py1:py2, px1:px2] = filled

    _fill(y, x1, y1, x2, y2, 64, 31)          # 10-bit limited-range black
    cx1, cy1 = x1 // 2, y1 // 2
    cx2, cy2 = min(w // 2, (x2 + 1) // 2), min(h // 2, (y2 + 1) // 2)
    _fill(u, cx1, cy1, cx2, cy2, 512, 15)
    _fill(v, cx1, cy1, cx2, cy2, 512, 15)


def render_hdr(src, dst, detections, cum, bands, fps, pad, mode,
               encoder, tags, band_margin=25, progress_every=60, cb=None,
               mode_map=None, face_shape="ellipse"):
    """render(), but 10-bit end to end: decode the HDR source to raw
    yuv420p10le, blur the planes in place, encode 10-bit HEVC with the
    source's color tags (PQ/HLG + BT.2020) carried over. `hvc1` tagging
    keeps QuickTime/Apple players happy. No preview mode — previews use
    the SDR copy."""
    mode_map = mode_map or {}
    def _mode_for(cat):
        return mode_map.get(cat, mode)
    cb = cb or Callbacks()
    meta = cv2.VideoCapture(src)
    w = int(meta.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(meta.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(meta.get(cv2.CAP_PROP_FRAME_COUNT))
    meta.release()
    cb.log(f"      encoder: {encoder} 10-bit"
           + (" (GPU)" if encoder == "hevc_nvenc" else " (CPU)"))

    dec = subprocess.Popen(
        ["ffmpeg", "-loglevel", "error", "-i", src,
         "-f", "rawvideo", "-pix_fmt", "yuv420p10le", "pipe:1"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        bufsize=w * h * 6)
    vargs = (["-c:v", "hevc_nvenc", "-preset", "p4", "-cq", "19",
              "-profile:v", "main10", "-pix_fmt", "p010le"]
             if encoder == "hevc_nvenc"
             else ["-c:v", "libx265", "-crf", "18", "-preset", "fast",
                   "-pix_fmt", "yuv420p10le"])
    targs = [a for kv in (tags or {}).items() for a in kv]
    if os.path.splitext(dst)[1].lower() in (".mp4", ".mov"):
        targs += ["-tag:v", "hvc1"]          # QuickTime/Apple compatibility
    enc = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "rawvideo", "-pix_fmt", "yuv420p10le",
         "-s", f"{w}x{h}", "-r", f"{fps:.6f}", "-i", "pipe:0",
         "-i", src, "-map", "0:v:0", "-map", "1:a:0?",
         *vargs, *targs, "-c:a", "copy", dst],
        stdin=subprocess.PIPE)

    buckets = {}
    for d in detections:
        for s in range(int(d.t_start), int(d.t_end) + 2):
            buckets.setdefault(s, []).append(d)

    ysz, csz = w * h, (w // 2) * (h // 2)
    fbytes = (ysz + 2 * csz) * 2

    def cleanup_partial():
        for pr in (dec, enc):
            try:
                if pr is enc:
                    pr.stdin.close()
            except Exception:
                pass
            pr.terminate()
            pr.wait()
        if os.path.exists(dst):
            try:
                os.remove(dst)
            except OSError:
                pass

    idx = 0
    while True:
        if idx % 30 == 0 and cb.cancelled():
            cleanup_partial()
            raise PipelineCancelled()
        raw = dec.stdout.read(fbytes)
        if len(raw) < fbytes:
            break
        buf = np.frombuffer(raw, dtype=np.uint16).copy()
        y = buf[:ysz].reshape(h, w)
        u = buf[ysz:ysz + csz].reshape(h // 2, w // 2)
        v = buf[ysz + csz:].reshape(h // 2, w // 2)
        t = idx / fps
        ox, oy = cum[min(idx, len(cum) - 1)]

        for d in buckets.get(int(t), []):
            if not (d.t_start - 0.01 <= t <= d.t_end + 0.01):
                continue
            drift = min(24.0, 0.05 * (abs(ox - d.aoff[0]) + abs(oy - d.aoff[1])))
            px = pad + drift
            _blur_yuv10(y, u, v, d.cbox[0] + ox - px, d.cbox[1] + oy - px,
                        d.cbox[2] + ox + px, d.cbox[3] + oy + px,
                        _mode_for(d.category),
                        shape=("ellipse" if d.category == "face"
                               and face_shape == "ellipse" else "rect"))

        bx, by = bands[min(idx, len(bands) - 1)]
        vals = set(mode_map.values()) | {mode}
        band_mode = ("box" if "box" in vals else
                     "mosaic" if "mosaic" in vals else "blur")
        def band(x1, y1_, x2, y2_):
            _blur_yuv10(y, u, v, x1, y1_, x2, y2_, band_mode)
        if by < -2:
            band(0, h - (abs(by) + band_margin), w, h)
        elif by > 2:
            band(0, 0, w, by + band_margin)
        if bx < -2:
            band(w - (abs(bx) + band_margin), 0, w, h)
        elif bx > 2:
            band(0, 0, bx + band_margin, h)

        enc.stdin.write(buf.tobytes())
        idx += 1
        if idx % progress_every == 0:
            cb.progress("render", idx, total)
        if idx % 300 == 0:
            cb.log(f"  rendering… {idx}/{max(total, idx)}")

    dec.stdout.close()
    dec.wait()
    cb.progress("render", max(total, idx), max(total, idx))
    enc.stdin.close()
    rc = enc.wait()
    if rc != 0:
        raise RuntimeError(f"HDR encode failed (ffmpeg exit {rc})")
    cb.log(f"      HDR preserved: 10-bit HEVC, color tags "
           + (", ".join(f"{k.lstrip('-')}={v}" for k, v in (tags or {}).items())
              or "(none in source)"))


def render(src, dst, detections, cum, bands, fps, pad, mode, preview,
           encoder="auto", band_margin=25, progress_every=60, cb=None,
           mode_map=None, draw_scores=False, vcodec="h264",
           face_shape="ellipse"):
    # mode_map: {category: "blur"|"box"} overrides the global `mode` per
    # category. Lets you black-box the reversible-blur-vulnerable categories
    # (SSN, MRN, account numbers) while blurring faces, in one render.
    mode_map = mode_map or {}
    def _mode_for(cat):
        return mode_map.get(cat, mode)
    cb = cb or Callbacks()
    cap = cv2.VideoCapture(src)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # --- output sink: single-pass ffmpeg pipe (NVENC if available) ---
    proc = None
    out = None
    tmp_video = None
    codec = nvenc_available(encoder, cb)
    if vcodec == "hevc" and codec:
        henc = hevc10_encoder(encoder, cb)   # verified hevc encoder ladder
        if henc is None:
            cb.log("      note: no HEVC encoder available — falling back "
                   "to H.264")
        else:
            codec = henc
    if codec:
        cb.log(f"      encoder: {codec}"
               + (" (GPU)" if codec.endswith("_nvenc") else " (CPU)"))
        if codec == "h264_nvenc":
            vargs = ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "19"]
        elif codec == "hevc_nvenc":
            vargs = ["-c:v", "hevc_nvenc", "-preset", "p4", "-cq", "19"]
        elif codec == "libx265":
            vargs = ["-c:v", "libx265", "-crf", "20", "-preset", "fast"]
        else:
            vargs = ["-c:v", "libx264", "-crf", "18", "-preset", "fast"]
        if codec in ("hevc_nvenc", "libx265") \
                and os.path.splitext(dst)[1].lower() in (".mp4", ".mov"):
            vargs += ["-tag:v", "hvc1"]      # QuickTime/Apple compatibility
        cmd = ["ffmpeg", "-y", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "bgr24",
               "-s", f"{w}x{h}", "-r", f"{fps:.6f}", "-i", "pipe:0",
               "-i", src, "-map", "0:v:0", "-map", "1:a:0?",
               *vargs, "-pix_fmt", "yuv420p", "-c:a", "copy", dst]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    else:
        cb.log("      encoder: OpenCV mp4v (ffmpeg not found — no audio!)")
        tmp_video = dst + ".noaudio.mp4"
        out = cv2.VideoWriter(tmp_video, cv2.VideoWriter_fourcc(*"mp4v"),
                              fps, (w, h))

    buckets = {}
    for d in detections:
        for s in range(int(d.t_start), int(d.t_end) + 2):
            buckets.setdefault(s, []).append(d)

    def cleanup_partial():
        cap.release()
        if proc is not None:
            try:
                proc.stdin.close()
            except Exception:
                pass
            proc.terminate()
            proc.wait()
        if out is not None:
            out.release()
        for p in (dst, tmp_video):
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass

    idx = 0
    while True:
        if idx % 30 == 0 and cb.cancelled():
            cleanup_partial()
            raise PipelineCancelled()
        ok, frame = cap.read()
        if not ok:
            break
        t = idx / fps
        ox, oy = cum[min(idx, len(cum) - 1)]

        # 1. tracked PHI boxes, translated by scroll offset
        for d in buckets.get(int(t), []):
            if not (d.t_start - 0.01 <= t <= d.t_end + 0.01):
                continue
            # drift allowance: residual tracking error grows (slowly) with
            # distance scrolled since detection — expand the box to cover it
            drift = min(24.0, 0.05 * (abs(ox - d.aoff[0]) + abs(oy - d.aoff[1])))
            px = pad + drift
            x1 = d.cbox[0] + ox - px
            y1 = d.cbox[1] + oy - px
            x2 = d.cbox[2] + ox + px
            y2 = d.cbox[3] + oy + px
            if preview:
                cv2.rectangle(frame, (int(max(0, x1)), int(max(0, y1))),
                              (int(min(w, x2)), int(min(h, y2))), (0, 0, 255), 2)
                label = d.category
                if draw_scores and d.category == "face":
                    label = f"face {d.confidence:.2f}"
                cv2.putText(frame, label, (int(max(0, x1)), int(max(12, y1 - 4))),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            else:
                blur_region(frame, x1, y1, x2, y2, _mode_for(d.category),
                            shape=("ellipse" if d.category == "face"
                                   and face_shape == "ellipse" else "rect"))

        # 2. safety bands: unscanned content that scrolled into view
        bx, by = bands[min(idx, len(bands) - 1)]
        vals = set(mode_map.values()) | {mode}
        band_mode = ("box" if "box" in vals else
                     "mosaic" if "mosaic" in vals else "blur")  # never weaker
        def band(x1, y1, x2, y2):
            if preview:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2 - 1), int(y2 - 1)), (0, 165, 255), 2)
            else:
                blur_region(frame, x1, y1, x2, y2, band_mode)
        if by < -2:      # content moved up -> unscanned strip entering at bottom
            band(0, h - (abs(by) + band_margin), w, h)
        elif by > 2:     # content moved down -> unscanned strip at top
            band(0, 0, w, by + band_margin)
        if bx < -2:      # content moved left -> strip at right
            band(w - (abs(bx) + band_margin), 0, w, h)
        elif bx > 2:
            band(0, 0, bx + band_margin, h)

        if proc is not None:
            proc.stdin.write(frame.tobytes())
        else:
            out.write(frame)
        idx += 1
        if idx % progress_every == 0:
            cb.progress("render", idx, total)
        if idx % 300 == 0:
            cb.log(f"  rendering… {idx}/{total} ({100 * idx // max(total, 1)}%)")

    cap.release()
    cb.progress("render", total, total)
    if proc is not None:
        proc.stdin.close()
        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(f"ffmpeg encode failed (exit {rc}) — try --encoder x264")
    else:
        out.release()
        os.replace(tmp_video, dst)


# ----------------------------------------------------------------------------
# Face detection (clinical photos, webcam bubbles — OCR is blind to these)
# ----------------------------------------------------------------------------

YUNET_URL = ("https://media.githubusercontent.com/media/opencv/opencv_zoo/"
             "main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx")
SFACE_URL = ("https://media.githubusercontent.com/media/opencv/opencv_zoo/"
             "main/models/face_recognition_sface/"
             "face_recognition_sface_2021dec.onnx")


# --- OpenCV DNN device selection -------------------------------------------
# The stock opencv-python wheel is CPU-only, so face detection (YuNet/SCRFD/
# CenterFace) and SFace grouping run on the CPU. The CUDA Docker image ships
# a CUDA-built OpenCV; when a GPU is present we push those nets onto it. All
# helpers fall back to CPU cleanly, so the CPU image behaves exactly as
# before. Set OPENSCRUB_CPU_DNN=1 to force CPU even on a CUDA build.
_CUDA_DNN = None


def cuda_dnn_available():
    global _CUDA_DNN
    if _CUDA_DNN is None:
        ok = False
        if os.environ.get("OPENSCRUB_CPU_DNN", "").lower() not in (
                "1", "true", "yes"):
            try:
                ok = (hasattr(cv2, "cuda")
                      and cv2.cuda.getCudaEnabledDeviceCount() > 0
                      and hasattr(cv2.dnn, "DNN_BACKEND_CUDA"))
            except Exception:
                ok = False
        _CUDA_DNN = ok
    return _CUDA_DNN


def _apply_cuda_dnn(net):
    """Push a cv2.dnn net to the GPU when available; no-op on CPU builds."""
    if cuda_dnn_available():
        try:
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        except Exception:
            pass
    return net


def _make_yunet(model, size, thresh):
    if cuda_dnn_available():
        try:
            return cv2.FaceDetectorYN_create(
                model, "", size, thresh, 0.3, 5000,
                cv2.dnn.DNN_BACKEND_CUDA, cv2.dnn.DNN_TARGET_CUDA)
        except Exception:
            pass
    return cv2.FaceDetectorYN_create(model, "", size, thresh)


def _make_sface(model):
    if cuda_dnn_available():
        try:
            return cv2.FaceRecognizerSF_create(
                model, "", cv2.dnn.DNN_BACKEND_CUDA, cv2.dnn.DNN_TARGET_CUDA)
        except Exception:
            pass
    return cv2.FaceRecognizerSF_create(model, "")


def _model_dir():
    d = os.path.join(os.path.expanduser("~"), ".openscrub", "models")
    os.makedirs(d, exist_ok=True)
    return d


def install_is_readonly():
    """True when the code lives somewhere the user shouldn't write to:
    pip's site-packages, or a frozen (PyInstaller) install under Program
    Files. Folder/git deploys return False and keep writing next to the
    code, as always."""
    p = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
    return ("site-packages" in p or "dist-packages" in p
            or bool(getattr(sys, "frozen", False)))


def user_data_dir():
    """Per-user writable data root (mirrors openscrub_web's choice):
    %LOCALAPPDATA%/OpenScrub on Windows, ~/.local/share/OpenScrub elsewhere.

    The env-derived root is confined BEFORE any filesystem use: a hostile
    or mangled LOCALAPPDATA can't point OpenScrub's writes at a system
    directory. If it resolves outside the user's own profile (very rare:
    profile folder redirection), we fall back to ~/.local/share rather
    than honour it. NOTE the guard's exact shape — canonicalize, then a
    SINGLE startswith condition whose true branch adopts the value — is
    deliberate: it's the one form CodeQL's path-injection barrier analysis
    recognizes; compound conditions (`x != a and not x.startswith(b)`)
    defeat its dominance check and the taint (and the alert) survives."""
    home = os.path.realpath(os.path.expanduser("~"))
    base = os.path.join(home, ".local", "share")
    env = os.environ.get("LOCALAPPDATA")
    if env:
        cand = os.path.realpath(env)
        if cand.startswith(home.rstrip(os.sep) + os.sep):
            base = cand
    d = os.path.join(base, "OpenScrub")
    os.makedirs(d, exist_ok=True)
    return d


MODEL_KINDS = ("plate", "face")


def model_registry_path(kind="plate"):
    """Path of the WRITABLE model registry for `kind` ("plate" or "face" —
    TOFU pins are written back here).

    Folder deploys use <kind>_models.json next to the code. Read-only
    installs (pip / frozen) use a per-user copy seeded from the packaged
    registry; new models added by a release are merged into that copy on
    read (never overwriting an existing entry's pinned hash)."""
    # Whitelist the kind before it ever forms a filename (real defense:
    # `kind` arrives raw from the `/api/models/<kind>` route).
    if kind not in MODEL_KINDS:
        raise ValueError("unknown model kind: %r" % (kind,))
    # Then build the filename from LITERALS, branching on equality — the
    # request-tainted value never becomes part of a path expression. The
    # raise above already guarantees kind is valid, but CodeQL doesn't
    # credit membership-in-a-module-constant as a taint barrier (its
    # "uncontrolled data in path expression" alerts survived it); a literal
    # in every branch leaves nothing to flag, for this or any scanner.
    fname = "face_models.json" if kind == "face" else "plate_models.json"
    here = os.path.dirname(os.path.abspath(__file__))
    packaged = os.path.join(here, fname)
    if not install_is_readonly():
        return packaged
    user = os.path.join(user_data_dir(), fname)
    try:
        if not os.path.exists(user) and os.path.exists(packaged):
            shutil.copy2(packaged, user)
        elif os.path.exists(user) and os.path.exists(packaged):
            with open(user, encoding="utf-8") as f:
                mine = json.load(f)
            with open(packaged, encoding="utf-8") as f:
                shipped = json.load(f)
            have = {m.get("id") for m in mine.get("models", [])}
            new = [m for m in shipped.get("models", [])
                   if m.get("id") not in have]
            if new:
                mine.setdefault("models", []).extend(new)
                with open(user, "w", encoding="utf-8") as f:
                    json.dump(mine, f, indent=2)
    except Exception:
        pass                    # fall through: a readable path either way
    return user if os.path.exists(user) else packaged


def plate_registry_path():
    return model_registry_path("plate")


def load_model_registry(kind="plate"):
    """Return the curated model list for `kind`, or [] if absent."""
    try:
        with open(model_registry_path(kind), encoding="utf-8") as f:
            return json.load(f).get("models", [])
    except Exception:
        return []


def load_plate_registry():
    return load_model_registry("plate")


def download_model(entry, kind="plate", dest_dir=None, cb=None,
                   progress=None):
    """Download a registry model to models/<id>.onnx, verifying its SHA-256.

    entry: a dict from load_model_registry(kind). progress: optional
    callable (fraction_0_to_1). Returns the saved path. Raises on any
    failure (bad URL, hash mismatch) after removing a partial/incorrect
    file — a privacy tool must never silently run an unverified model.
    """
    import hashlib, urllib.request
    if kind not in MODEL_KINDS:
        raise ValueError("unknown model kind: %r" % (kind,))
    log = (cb.log if cb else print)
    url = entry.get("download_url", "")
    want = (entry.get("sha256", "") or "").lower()
    if not url or url == "TODO_VERIFY":
        raise ValueError("model '%s' has no verified download_url yet "
                         "(registry entry says TODO_VERIFY)" % entry.get("id"))
    dest_dir = dest_dir or (
        os.path.join(user_data_dir(), "models") if install_is_readonly()
        else os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "models"))
    os.makedirs(dest_dir, exist_ok=True)
    # the id comes from a JSON registry file — reduce it to a strict
    # basename so a crafted id ("../evil") can never escape dest_dir
    safe_id = re.sub(r"[^A-Za-z0-9._-]", "_", str(entry.get("id")
                                                  or (kind + "_model")))
    dest = os.path.join(dest_dir, "%s.onnx" % safe_id)
    tmp = dest + ".part"
    log("      downloading %s model: %s" % (kind, entry.get("label", entry.get("id"))))
    h = hashlib.sha256()
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        got = 0
        while True:
            chunk = r.read(65536)
            if not chunk:
                break
            f.write(chunk); h.update(chunk); got += len(chunk)
            if progress and total:
                progress(min(1.0, got / total))
    digest = h.hexdigest()
    if want and want != "todo_verify" and digest != want:
        os.remove(tmp)
        raise ValueError("SHA-256 mismatch for %s: got %s, expected %s — "
                         "file rejected." % (entry.get("id"), digest, want))
    if not want or want == "todo_verify":
        # trust-on-first-use: pin the computed hash into the registry so every
        # later download of this model must match this exact file.
        log("      first download of this model: pinning sha256=%s" % digest[:16] + "…")
        try:
            with open(model_registry_path(kind), encoding="utf-8") as f:
                reg = json.load(f)
            for m in reg.get("models", []):
                if m.get("id") == entry.get("id"):
                    m["sha256"] = digest
            with open(model_registry_path(kind), "w", encoding="utf-8") as f:
                json.dump(reg, f, indent=2)
        except Exception as e:
            log("      (could not pin hash into registry: %s)" % e)
    os.replace(tmp, dest)
    log("      saved verified model -> %s" % dest)
    return dest


def download_plate_model(entry, dest_dir=None, cb=None, progress=None):
    return download_model(entry, "plate", dest_dir=dest_dir, cb=cb,
                          progress=progress)


class PlateDetector:
    """License-plate detector using a single-class YOLO ONNX model (no PyTorch
    / ultralytics dependency at runtime).

    Two inference backends, tried in order per model:
      1. OpenCV DNN — fast, honours the app's CUDA target. Handles raw YOLO
         detect heads (1,5,8400) and any graph OpenCV can build.
      2. onnxruntime — fallback for "end2end" exports whose baked-in ONNX
         NonMaxSuppression node OpenCV's DNN CANNOT build (readNetFromONNX
         raises 'Can't create layer ... NonMaxSuppression'). EVERY
         open-image-models YOLOv9 plate model in the registry is such an
         export, so without onnxruntime the plate category would silently do
         nothing — a fail-open hole in a fail-closed privacy tool.

    Three output conventions are auto-detected by shape (see _decode): a raw
    YOLO head (1,5,8400), a 6-col end2end (N,6)=x1,y1,x2,y2,score,class, and a
    7-col end2end (N,7)=batch,x1,y1,x2,y2,class,score — the last is what the
    current open-image-models YOLOv9 models emit.

    The model file is NOT bundled — it's downloaded/placed by the optional
    installer or the model picker. If the model is absent (or neither backend
    can load it) the detector is INERT (find() returns []), so the plate
    category simply does nothing rather than erroring — mirroring how
    FaceDetector degrades, and keeping plates an opt-in capability.
    """

    INPUT = 640          # YOLOv8 square input
    MODEL_ENV = "OPENSCRUB_PLATE_MODEL"

    def __init__(self, cb=None, model_path=None, thresh=0.35, nms=0.45,
                 expand=0.08, input_size=640):
        self.log = (cb.log if cb else print)
        self.INPUT = int(input_size)
        self.thresh = float(thresh)
        self.nms = float(nms)
        self.expand = float(expand)
        self.net = None          # OpenCV DNN backend (raw-head models)
        self.ort = None          # onnxruntime backend (end2end/NMS models)
        self._ort_in = None
        self._ort_out = None
        # resolve model: explicit arg > env var > conventional locations
        candidates = []
        if model_path:
            candidates.append(model_path)
        env = os.environ.get(self.MODEL_ENV)
        if env:
            candidates.append(env)
        here = os.path.dirname(os.path.abspath(__file__))
        # read-only installs (pip / frozen) download models to the per-user
        # data dir instead of next to the code — search both.
        roots = [here]
        if install_is_readonly():
            roots.append(user_data_dir())
        for r in roots:
            candidates += [
                os.path.join(r, "models", "plate_yolov8.onnx"),
                os.path.join(r, "plate_yolov8.onnx"),
            ]
        # registry-downloaded models are saved as models/<registry-id>.onnx;
        # search those too (recommended entries first), and pick up each
        # model's declared input size from the registry.
        reg_size = {}
        try:
            reg = load_plate_registry()
            for m in sorted(reg, key=lambda x: not x.get("recommended", False)):
                for r in roots:
                    mp = os.path.join(r, "models", "%s.onnx" % m.get("id"))
                    candidates.append(mp)
                    reg_size[mp] = int(m.get("input_size", 640) or 640)
        except Exception:
            pass
        found = next((c for c in candidates if c and os.path.exists(c)), None)
        if found in reg_size:
            self.INPUT = reg_size[found]
        if not found:
            self.log("      plate detector: no model found — plate category "
                     "inactive. Place a YOLOv8 plate ONNX at models/"
                     "plate_yolov8.onnx or set $%s." % self.MODEL_ENV)
            return
        base = os.path.basename(found)
        # Backend 1 — OpenCV DNN. Handles raw YOLO detect heads (1,5,8400) and
        # any graph OpenCV can build; fast, and honours the app's CUDA target.
        # OpenCV prints a red ERROR to stderr when it can't build a node (e.g.
        # the end2end NonMaxSuppression) even though we catch it and fall back
        # to onnxruntime — silence its logger just around this probe so a
        # working fallback doesn't look like a failure to the user. (Top-level
        # cv2.setLogLevel is thread-safe and present on 4.x/5.x; the older
        # cv2.utils.logging module is absent on headless 4.x builds.)
        _prev_ll = None
        try:
            _prev_ll = cv2.getLogLevel()
            cv2.setLogLevel(0)   # 0 = SILENT
        except Exception:
            _prev_ll = None
        try:
            net = _apply_cuda_dnn(cv2.dnn.readNetFromONNX(found))
            # honour the same CPU/GPU intent the rest of the app uses
            try:
                if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            except Exception:
                pass
            self.net = net
            cv2_err = None
        except Exception as e:
            cv2_err = e
        finally:
            if _prev_ll is not None:
                try:
                    cv2.setLogLevel(_prev_ll)
                except Exception:
                    pass
        if self.net is not None:
            self.log("      plate detector: loaded %s (OpenCV DNN)" % base)
            return
        # Backend 2 — onnxruntime. REQUIRED for "end2end" exports whose baked-in
        # NonMaxSuppression node OpenCV's DNN cannot build — that includes EVERY
        # open-image-models YOLOv9 plate model in the registry. cv2.dnn raises
        # 'Can\'t create layer ... of type "NonMaxSuppression"' on those, which
        # would otherwise leave the plate category silently inactive (plates
        # never blurred) — a fail-OPEN hole in a fail-closed privacy tool.
        try:
            import onnxruntime as ort
        except Exception:
            self.log("      plate detector: OpenCV DNN can't load %s (%s) and "
                     "onnxruntime is not installed — plate category INACTIVE. "
                     "Install onnxruntime (pip install onnxruntime) to enable "
                     "the license-plate models." % (base, cv2_err))
            return
        try:
            # Run plates on the GPU when possible: prefer CUDA if this
            # onnxruntime build offers it (onnxruntime-gpu, shipped in the CUDA
            # Docker image), else CPU (the plain onnxruntime wheel). Listing
            # CPU as the second provider lets onnxruntime fall back per-node if
            # CUDA init fails at runtime, so a cuDNN mismatch degrades to CPU
            # rather than killing the job. OPENSCRUB_CPU_DNN=1 forces CPU, the
            # same escape hatch the OpenCV DNN path honours.
            avail = ort.get_available_providers()
            force_cpu = os.environ.get("OPENSCRUB_CPU_DNN") == "1"
            providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                         if ("CUDAExecutionProvider" in avail and not force_cpu)
                         else ["CPUExecutionProvider"])
            sess = ort.InferenceSession(found, providers=providers)
            self.ort = sess
            self._ort_in = sess.get_inputs()[0].name
            self._ort_out = [o.name for o in sess.get_outputs()]
            self.log("      plate detector: loaded %s (onnxruntime, %s)"
                     % (base, sess.get_providers()[0]))
        except Exception as e:
            self.log("      plate detector: failed to load model — OpenCV DNN "
                     "(%s) and onnxruntime (%s) both errored. Plate category "
                     "INACTIVE." % (cv2_err, e))
            self.net = None
            self.ort = None

    def available(self):
        return self.net is not None or self.ort is not None

    def find(self, frame, detect_scale=1.0):
        """-> [(x1,y1,x2,y2,conf)] in full-frame pixels. Empty if no model.

        detect_scale is accepted for call-site symmetry with FaceDetector but
        intentionally unused: the letterbox below already resizes every frame
        to the model's fixed input size, so an extra pre-downscale would only
        lose detail without saving time."""
        if self.net is None and self.ort is None:
            return []
        h, w = frame.shape[:2]
        # letterbox to a square INPUT (preserve aspect, pad 114 like YOLO)
        s = self.INPUT / max(h, w)
        nw, nh = int(round(w * s)), int(round(h * s))
        resized = cv2.resize(frame, (nw, nh))
        canvas = np.full((self.INPUT, self.INPUT, 3), 114, np.uint8)
        canvas[:nh, :nw] = resized
        if self.ort is not None:
            # onnxruntime: BGR->RGB, HWC->CHW, /255, batched fp32 (identical
            # preprocessing to the cv2 blob below — only the runtime differs).
            inp = np.ascontiguousarray(
                canvas[:, :, ::-1].transpose(2, 0, 1)[None]).astype(np.float32)
            inp /= 255.0
            out = self.ort.run(self._ort_out, {self._ort_in: inp})[0]
        else:
            blob = cv2.dnn.blobFromImage(canvas, 1 / 255.0,
                                         (self.INPUT, self.INPUT),
                                         swapRB=True, crop=False)
            self.net.setInput(blob)
            out = self.net.forward()
        return self._decode(np.asarray(out), s, w, h)

    def _decode(self, out, s, w, h):
        """Turn a raw model output tensor into [(x1,y1,x2,y2,conf)] full-frame
        boxes. Backend-agnostic (OpenCV DNN and onnxruntime feed the same
        arrays) and pure — unit-testable without a model. `s` is the letterbox
        scale, (w,h) the original frame size."""
        # Drop leading singleton (batch) axes WITHOUT collapsing a lone
        # detection row: np.squeeze on (1,1,C) would yield a 1-D vector and
        # lose the single box. cv2 may return (N,C) or (1,N,C) by version.
        while out.ndim > 2 and out.shape[0] == 1:
            out = out[0]
        if out.ndim != 2:
            return []
        # Three ONNX output conventions are supported, auto-detected by shape:
        #
        #  (A) raw YOLOv8 detect head: shape (5, 8400) — rows are cx,cy,w,h,
        #      score for a single class; needs decode + NMS here.
        #  (B) "end2end" export, 6 cols: (N, 6) = x1,y1,x2,y2,score,class,
        #      NMS baked into the graph. Scale back to full-frame pixels.
        #  (C) "end2end" export, 7 cols: (N, 7) = batch,x1,y1,x2,y2,class,
        #      score — the layout the CURRENT open-image-models YOLOv9 models
        #      emit (their postprocess reads cols 1:5 / 5 / 6). Earlier code
        #      only knew (B) and mis-read (C) as a raw head, IndexError-ing on
        #      row[4] and crashing the whole scan (a moving object with a
        #      plate would take the job down). Both end2end widths now parse.
        #
        # ONNX emits end2end as (batch,N,C), so after stripping batch the LAST
        # axis is the attribute axis (C in {6,7}) and rows are axis 0. Accept a
        # transposed export (C on axis 0) only when the other axis is clearly a
        # box count (larger, and not the 8400-anchor raw head).
        a, b = out.shape
        cols = None
        if b in (6, 7):
            rows, cols = out, b
        elif a in (6, 7) and b != 8400 and b > a:
            rows, cols = out.T, a

        res = []
        if cols is not None:
            for r in rows:
                if cols == 7:      # batch,x1,y1,x2,y2,class,score
                    x1, y1, x2, y2, score = (float(r[1]), float(r[2]),
                                             float(r[3]), float(r[4]),
                                             float(r[6]))
                else:              # x1,y1,x2,y2,score,class
                    x1, y1, x2, y2, score = (float(r[0]), float(r[1]),
                                             float(r[2]), float(r[3]),
                                             float(r[4]))
                if score < self.thresh:
                    continue
                # scale from letterboxed INPUT-space back to full frame
                bx1, by1, bx2, by2 = x1 / s, y1 / s, x2 / s, y2 / s
                bw, bh = bx2 - bx1, by2 - by1
                ex, ey = bw * self.expand, bh * self.expand
                res.append((max(0.0, bx1 - ex), max(0.0, by1 - ey),
                            min(float(w), bx2 + ex), min(float(h), by2 + ey),
                            round(score, 3)))
            return res

        # raw YOLOv8 head: (5, 8400) -> transpose to per-box rows
        if out.shape[0] < out.shape[1]:
            out = out.T
        boxes, scores = [], []
        for row in out:
            score = float(row[4])
            if score < self.thresh:
                continue
            cx, cy, bw, bh = row[0], row[1], row[2], row[3]
            x = (cx - bw / 2) / s
            y = (cy - bh / 2) / s
            boxes.append([int(x), int(y), int(bw / s), int(bh / s)])
            scores.append(score)
        if not boxes:
            return []
        idxs = cv2.dnn.NMSBoxes(boxes, scores, self.thresh, self.nms)
        for i in np.array(idxs).flatten():
            bx, by, bw, bh = boxes[i]
            ex, ey = int(bw * self.expand), int(bh * self.expand)
            x1 = max(0, bx - ex); y1 = max(0, by - ey)
            x2 = min(w, bx + bw + ex); y2 = min(h, by + bh + ey)
            res.append((float(x1), float(y1), float(x2), float(y2), scores[i]))
        return res


def _nms_boxes(boxes, thr=0.4):
    """Greedy IoU NMS over [x1,y1,x2,y2,score] lists."""
    if not boxes:
        return []
    b = np.array(boxes, dtype=np.float64)
    idx = b[:, 4].argsort()[::-1]
    keep = []
    while len(idx):
        i = idx[0]
        keep.append(boxes[i])
        rest = idx[1:]
        xx1 = np.maximum(b[i, 0], b[rest, 0])
        yy1 = np.maximum(b[i, 1], b[rest, 1])
        xx2 = np.minimum(b[i, 2], b[rest, 2])
        yy2 = np.minimum(b[i, 3], b[rest, 3])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        a1 = (b[i, 2] - b[i, 0]) * (b[i, 3] - b[i, 1])
        a2 = (b[rest, 2] - b[rest, 0]) * (b[rest, 3] - b[rest, 1])
        iou = inter / np.maximum(a1 + a2 - inter, 1e-6)
        idx = rest[iou < thr]
    return keep


class FaceDetector:
    """Face detector with three tiers:
    1. an optional user-installed ONNX model (--face-model / registry:
       CenterFace or SCRFD, auto-recognized by output signature) — higher
       recall on small/hard faces;
    2. YuNet DNN (auto-downloaded on first use, ~230 KB) — the zero-setup
       default;
    3. OpenCV's built-in Haar cascade as the last resort.
    Boxes are expanded 15% so hairline/chin aren't left identifiable at the
    blur edge. A model that fails to load falls back LOUDLY to YuNet —
    detection never silently disappears."""

    def __init__(self, cb=None, expand=0.15, thresh=0.6, model_path=None):
        self.expand = expand
        self.thresh = float(thresh)
        log = (cb.log if cb else print)
        self.yunet = None
        self.haar = None
        self.net = None
        self.arch = None
        path = model_path or os.environ.get("OPENSCRUB_FACE_MODEL")
        if path:
            if os.path.exists(path):
                try:
                    net = _apply_cuda_dnn(cv2.dnn.readNet(path))
                    nouts = len(net.getUnconnectedOutLayersNames())
                    if nouts == 4:
                        self.arch = "centerface"
                    elif nouts in (6, 9):
                        self.arch = "scrfd"
                    else:
                        raise ValueError("unrecognized face-model output "
                                         "signature (%d outputs)" % nouts)
                    self.net = net
                    log("      face detector: %s ONNX model (%s) + built-in "
                        "YuNet (detections are UNIONED — an optional model "
                        "can only add faces, never lose the baseline's)"
                        % (self.arch, os.path.basename(path)))
                except Exception as e:
                    log(f"      face model failed to load ({e}) — "
                        "falling back to built-in YuNet")
            else:
                log(f"      face model not found: {path} — "
                    "falling back to built-in YuNet")
        model = os.path.join(_model_dir(), "face_detection_yunet_2023mar.onnx")
        if not os.path.exists(model) or os.path.getsize(model) < 10000:
            try:
                import urllib.request
                log("      downloading YuNet face model (~230 KB, one time)…")
                urllib.request.urlretrieve(YUNET_URL, model)
            except Exception as e:
                log(f"      YuNet download failed ({e}) — using Haar cascade fallback")
        if (os.path.exists(model) and os.path.getsize(model) > 10000
                and hasattr(cv2, "FaceDetectorYN_create")):
            try:
                self.yunet = _make_yunet(model, (320, 320), self.thresh)
                self.size = None
                if self.net is None:
                    log("      face detector: YuNet (DNN)"
                        + ("  [GPU]" if cuda_dnn_available() else ""))
                return
            except Exception as e:
                log(f"      YuNet init failed ({e}) — using Haar cascade fallback")
        self.haar = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        log("      face detector: Haar cascade (install note: YuNet is more accurate)")

    def find(self, frame, detect_scale=1.0):
        """-> [(x1, y1, x2, y2, conf)] with 15% expansion. detect_scale<1.0
        runs detection on a downscaled copy for speed, mapping boxes back to
        full resolution (output quality is unaffected)."""
        h, w = frame.shape[:2]
        s = detect_scale if 0.2 <= detect_scale < 1.0 else 1.0
        dframe = (cv2.resize(frame, (max(1, int(w * s)), max(1, int(h * s))))
                  if s < 1.0 else frame)
        dh, dw = dframe.shape[:2]
        raw = []
        if self.net is not None:
            raw += (self._find_centerface(dframe) if self.arch == "centerface"
                    else self._find_scrfd(dframe))
        if self.yunet is not None:
            if self.size != (dw, dh):
                self.yunet.setInputSize((dw, dh))
                self.size = (dw, dh)
            _, faces = self.yunet.detect(dframe)
            for f in (faces if faces is not None else []):
                x, y, fw, fh, conf = f[0], f[1], f[2], f[3], float(f[-1])
                raw.append([x, y, x + fw, y + fh, conf])
        if raw or self.net is not None or self.yunet is not None:
            out = [(x1 / s, y1 / s, x2 / s, y2 / s, conf)
                   for x1, y1, x2, y2, conf in _nms_boxes(raw)]
        else:
            out = []
            gray = cv2.cvtColor(dframe, cv2.COLOR_BGR2GRAY)
            for (x, y, fw, fh) in self.haar.detectMultiScale(gray, 1.1, 5,
                                                             minSize=(36, 36)):
                if 0.8 >= self.thresh:   # Haar has no score; gate by threshold
                    out.append((x / s, y / s, (x + fw) / s,
                                (y + fh) / s, 0.8))
        expanded = []
        for x1, y1, x2, y2, conf in out:
            ex, ey = (x2 - x1) * self.expand, (y2 - y1) * self.expand
            expanded.append((max(0, x1 - ex), max(0, y1 - ey),
                             min(w, x2 + ex), min(h, y2 + ey), conf))
        return expanded

    def _find_centerface(self, frame):
        """CenterFace decode: heatmap + exp(scale)*4 + offset on a stride-4
        grid, input padded up to a multiple of 32 (validated against the
        reference implementation on a known face)."""
        h, w = frame.shape[:2]
        iw, ih = (w + 31) // 32 * 32, (h + 31) // 32 * 32
        blob = cv2.dnn.blobFromImage(frame, 1.0, (iw, ih), (0, 0, 0),
                                     swapRB=True, crop=False)
        self.net.setInput(blob)
        hm, scale, off, _lms = self.net.forward(
            self.net.getUnconnectedOutLayersNames())
        heat = hm[0, 0]
        ys, xs = np.where(heat > self.thresh)
        sx, sy = w / iw, h / ih
        boxes = []
        for y, x in zip(ys, xs):
            s0 = float(np.exp(scale[0, 0, y, x])) * 4
            s1 = float(np.exp(scale[0, 1, y, x])) * 4
            o0 = float(off[0, 0, y, x])
            o1 = float(off[0, 1, y, x])
            x1 = max(0.0, (x + o1 + 0.5) * 4 - s1 / 2)
            y1 = max(0.0, (y + o0 + 0.5) * 4 - s0 / 2)
            boxes.append([x1 * sx, y1 * sy,
                          min(iw, x1 + s1) * sx, min(ih, y1 + s0) * sy,
                          float(heat[y, x])])
        return _nms_boxes(boxes)

    def _find_scrfd(self, frame):
        """SCRFD decode: per-stride (8/16/32) score + bbox-distance heads,
        2 anchors per cell; outputs are grouped by row count so the export's
        output ordering doesn't matter (validated against det_10g).

        Input is LETTERBOXED (aspect preserved, padded to 32-multiples) up
        to 1280 on the long side — the original fixed 640x640 squeeze
        distorted faces and shrank 1080p frames 3x, making the "best" model
        detect fewer faces than the built-in YuNet."""
        h, w = frame.shape[:2]
        S = min(1280, max(640, (max(h, w) + 31) // 32 * 32))
        scale = min(S / w, S / h)
        nw, nh = int(round(w * scale)), int(round(h * scale))
        iw, ih = (nw + 31) // 32 * 32, (nh + 31) // 32 * 32
        canvas = np.zeros((ih, iw, 3), np.uint8)
        canvas[:nh, :nw] = cv2.resize(frame, (nw, nh))
        blob = cv2.dnn.blobFromImage(canvas, 1.0 / 128, (iw, ih),
                                     (127.5, 127.5, 127.5),
                                     swapRB=True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.net.getUnconnectedOutLayersNames())
        groups = {}
        for o in outs:
            o = o.reshape(o.shape[-2], o.shape[-1]) if o.ndim == 3 else o
            groups.setdefault(o.shape[0], {})[o.shape[1]] = o
        boxes = []
        for n, g in groups.items():
            if 1 not in g or 4 not in g:
                continue
            stride = int(round((2 * iw * ih / n) ** 0.5))
            cols = iw // stride
            scores = g[1][:, 0]
            bb = g[4]
            for i in np.where(scores > self.thresh)[0]:
                cell = i // 2                       # 2 anchors per cell
                cx = (cell % cols) * stride
                cy = (cell // cols) * stride
                boxes.append([(cx - bb[i, 0] * stride) / scale,
                              (cy - bb[i, 1] * stride) / scale,
                              (cx + bb[i, 2] * stride) / scale,
                              (cy + bb[i, 3] * stride) / scale,
                              float(scores[i])])
        return _nms_boxes(boxes)


# ----------------------------------------------------------------------------
# Config profiles, ignore regions, provenance
# ----------------------------------------------------------------------------

def apply_config(args, parser):
    """Overlay a YAML config profile onto parsed args. CLI flags win: a
    config value only applies where the CLI value equals the parser default."""
    if not getattr(args, "config", None):
        return args
    import yaml
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    defaults = vars(parser.parse_args([args.video or "x"]))
    for key, val in cfg.items():
        dest = key.replace("-", "_")
        if dest == "ignore_regions":
            args.ignore_regions = [tuple(map(float, r)) for r in (val or [])]
            continue
        if dest == "zones":
            args.zones_data = {c: [tuple(float(v) for v in r) for r in rs]
                               for c, rs in (val or {}).items() if rs}
            continue
        if not hasattr(args, dest):
            raise RuntimeError(f"unknown config key in {args.config}: {key}")
        if getattr(args, dest) == defaults.get(dest):
            setattr(args, dest, val)
    return args


def load_zones(path):
    """Zones file: {"name": [[x1,y1,x2,y2], ...], "dob": [...]} with
    NORMALIZED 0-1 coordinates (resolution-independent). A category with no
    zones (or absent) is unrestricted — full frame."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {cat: [tuple(float(v) for v in r) for r in rects]
            for cat, rects in data.items() if rects}


def zones_to_pixels(zones, w, h):
    return {cat: [(r[0] * w, r[1] * h, r[2] * w, r[3] * h) for r in rects]
            for cat, rects in zones.items()}


def in_any_zone(screen_box, rects):
    cx = (screen_box[0] + screen_box[2]) / 2
    cy = (screen_box[1] + screen_box[3]) / 2
    return any(r[0] <= cx <= r[2] and r[1] <= cy <= r[3] for r in rects)


def in_ignore_region(screen_box, regions):
    cx = (screen_box[0] + screen_box[2]) / 2
    cy = (screen_box[1] + screen_box[3]) / 2
    return any(r[0] <= cx <= r[2] and r[1] <= cy <= r[3] for r in regions)


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _settings_dict(args):
    skip = {"video", "output", "report", "from_report", "batch", "config"}
    out = {k: v for k, v in vars(args).items()
           if k not in skip and isinstance(v, (str, int, float, bool, list,
                                               tuple, type(None)))}
    mm = getattr(args, "mode_map", None)
    if isinstance(mm, dict) and mm:
        out["mode_map"] = ",".join(f"{k}={v}" for k, v in sorted(mm.items()))
    return out


def write_report(path, args, state, output_path=None):
    prov = {
        "tool": "openscrub", "version": VERSION,
        "timestamp": datetime.datetime.now().astimezone().isoformat(),
        "input": os.path.abspath(args.video),
        "input_sha256": state.get("input_sha256"),
        "original_input": (os.path.abspath(args.original_video)
                           if getattr(args, "original_video", None) else None),
        "vfr_normalized": bool(getattr(args, "original_video", None)),
        "hdr_tonemapped": bool(getattr(args, "hdr_tonemapped", False)),
        "hdr_output": bool(getattr(args, "hdr_source", None)),
        "zones": getattr(args, "zones_data", None),
        "settings": _settings_dict(args),
    }
    if output_path and os.path.exists(output_path):
        prov["output"] = os.path.abspath(output_path)
        prov["output_sha256"] = sha256_file(output_path)
    doc = {
        "provenance": prov,
        "render_state": {
            "fps": state["fps"],
            "cum": [[round(x, 1), round(y, 1)] for x, y in state["cum"]],
            "bands": [[round(x, 1), round(y, 1)] for x, y in state["bands"]],
        },
        "detections": ([dict(asdict(d), enabled=True)
                        for d in state["detections"]]
                       + [dict(asdict(d), enabled=False, zone_dropped=True)
                          for d in state.get("zdropped", [])]),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def load_report(path):
    """-> (detections, render_state, provenance). Accepts v3 plain-list
    reports (no render_state) and v4 dict reports; disabled detections are
    dropped."""
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    if isinstance(doc, list):
        rows, state, prov = doc, None, {}
    else:
        rows, state, prov = doc.get("detections", []), doc.get("render_state"), \
            doc.get("provenance", {})
    dets = []
    for r in rows:
        if not r.get("enabled", True):
            continue
        dets.append(Detection(
            t_start=float(r["t_start"]), t_end=float(r["t_end"]),
            cbox=tuple(r["cbox"]), category=r["category"], text=r["text"],
            confidence=float(r.get("confidence", 1.0)),
            aoff=tuple(r.get("aoff", (0.0, 0.0))),
            last_seen=float(r.get("last_seen", r["t_start"])),
            # dense/track must survive the round trip: rendering rewrites the
            # report from rehydrated detections, and losing track ids here
            # exploded the re-opened review into one card per frame sample
            dense=bool(r.get("dense", False)),
            track=int(r.get("track", -1)),
            person=int(r.get("person", -1))))
    return dets, state, prov


# ----------------------------------------------------------------------------
# Variable frame rate (VFR) handling
# ----------------------------------------------------------------------------

def probe_vfr(path):
    """-> (is_vfr, avg_fps). Screen recorders (OBS, Game Bar) often produce
    VFR video; the pipeline's frame->time mapping assumes CFR, so VFR input
    causes blur-timing drift and audio desync unless normalized first."""
    if not shutil.which("ffprobe"):
        return False, None
    rc, out, _ = 0, "", ""
    try:
        p = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                            "-show_entries",
                            "stream=r_frame_rate,avg_frame_rate",
                            "-of", "json", path],
                           capture_output=True, text=True, timeout=30)
        rc, out = p.returncode, p.stdout
    except Exception:
        return False, None
    if rc != 0 or not out:
        return False, None
    try:
        st = json.loads(out)["streams"][0]
        def frac(s):
            a, _, b = s.partition("/")
            return float(a) / float(b or 1) if float(b or 1) else 0.0
        r, avg = frac(st.get("r_frame_rate", "0/1")), frac(st.get("avg_frame_rate", "0/1"))
    except Exception:
        return False, None
    if r <= 0 or avg <= 0:
        return False, avg or None
    return abs(r - avg) / max(r, avg) > 0.005, avg


def probe_hdr(path):
    """-> (is_hdr, desc). HDR when the first video stream signals a PQ or
    HLG transfer function, or BT.2020 primaries on a 10-bit format (what
    iPhone HDR / Dolby Vision clips carry after demux)."""
    if not shutil.which("ffprobe"):
        return False, None
    try:
        p = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                            "-show_entries",
                            "stream=color_transfer,color_primaries,pix_fmt",
                            "-of", "json", path],
                           capture_output=True, text=True, timeout=30)
        if p.returncode != 0 or not p.stdout:
            return False, None
        st = json.loads(p.stdout)["streams"][0]
    except Exception:
        return False, None
    trc = (st.get("color_transfer") or "").lower()
    prim = (st.get("color_primaries") or "").lower()
    if trc == "smpte2084":
        return True, "HDR10/PQ"
    if trc == "arib-std-b67":
        return True, "HLG"
    if prim == "bt2020" and "10" in (st.get("pix_fmt") or ""):
        return True, "BT.2020 10-bit"
    return False, None


_FILTER_CACHE = {}


def _ffmpeg_has(filter_name):
    if filter_name not in _FILTER_CACHE:
        try:
            p = subprocess.run(["ffmpeg", "-hide_banner", "-filters"],
                               capture_output=True, text=True, timeout=30)
            _FILTER_CACHE[filter_name] = f" {filter_name} " in p.stdout
        except Exception:
            _FILTER_CACHE[filter_name] = False
    return _FILTER_CACHE[filter_name]


# proper PQ/HLG -> BT.709 tone mapping (linearize, gamut-map, hable curve),
# instead of the flat washed-out colors a naive 8-bit decode produces
_TONEMAP_VF = ("zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,"
               "tonemap=tonemap=hable:desat=0,"
               "zscale=t=bt709:m=bt709:r=tv,format=yuv420p")


def normalize_vfr(args, cb):
    """Intake normalization: VFR input is transcoded to CFR (frame->time
    mapping assumes CFR); HDR input additionally gets a tone-mapped SDR
    BT.709 copy for SCANNING (detectors are 8-bit, and naive decode washes
    colors out). When the output should stay HDR (--hdr-output match, the
    default), a 10-bit CFR HDR source is kept for the render and
    args.hdr_source/hdr_encoder/hdr_tags are set. Reuses NVENC when
    available. No-op for CFR SDR input."""
    if getattr(args, "no_vfr_fix", False) or getattr(args, "vfr", "auto") == "ignore":
        return
    is_vfr, avg = probe_vfr(args.video)
    is_hdr, hdesc = probe_hdr(args.video)
    if is_hdr and not (_ffmpeg_has("zscale") and _ffmpeg_has("tonemap")):
        cb.log(f"      NOTE: input looks HDR ({hdesc}) but this ffmpeg build "
               "lacks the zscale/tonemap filters — processing without tone "
               "mapping; colors may look washed out in the output.")
        is_hdr = False
    if not is_vfr and not is_hdr:
        return
    target = int(round(avg)) if avg and 10 <= avg <= 120 else 30
    out_ref = args.output or os.path.splitext(args.video)[0] + "_redacted.mp4"
    base = os.path.join(os.path.dirname(os.path.abspath(out_ref)),
                        os.path.splitext(os.path.basename(args.video))[0])
    src0 = args.video

    def _cached(path):
        return (os.path.exists(path)
                and os.path.getmtime(path) > os.path.getmtime(src0))

    def _encode(inp, outp, cfr, tonemap, vargs, extra=None):
        tmargs = (["-vf", _TONEMAP_VF, "-color_primaries", "bt709",
                   "-color_trc", "bt709", "-colorspace", "bt709"]
                  if tonemap else [])
        attempts = ([["-fps_mode", "cfr", "-r", str(target)],
                     ["-vsync", "cfr", "-r", str(target)]] if cfr else [[]])
        for cfrargs in attempts:
            p = subprocess.run(["ffmpeg", "-y", "-loglevel", "error",
                                "-i", inp, *cfrargs, *tmargs, *vargs,
                                *(extra or []), "-c:a", "copy", outp],
                               capture_output=True, text=True)
            if p.returncode == 0:
                return
        raise RuntimeError("input normalization failed: "
                           + (p.stderr or "").strip()[-300:])

    codec = nvenc_available(getattr(args, "encoder", "auto"), cb)
    sdr_vargs = ((["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "18"]
                  if codec == "h264_nvenc"
                  else ["-c:v", "libx264", "-crf", "18", "-preset", "fast"])
                 + ["-pix_fmt", "yuv420p"])
    if is_vfr:
        cb.log(f"      input is VFR (avg {avg:.2f} fps) — normalizing to CFR "
               f"{target} fps first")

    if not is_hdr:
        fixed = base + ".cfr.mp4"
        if _cached(fixed):
            cb.log(f"      reusing existing {os.path.basename(fixed)}")
        else:
            _encode(src0, fixed, cfr=True, tonemap=False, vargs=sdr_vargs)
        args.original_video = src0
        args.hdr_tonemapped = False
        args.video = fixed
        return

    # ---- HDR input ----
    want_hdr = getattr(args, "hdr_output", "match") != "sdr"
    henc = hevc10_encoder(getattr(args, "encoder", "auto"), cb) if want_hdr \
        else None
    if want_hdr and henc is None:
        cb.log("      NOTE: no 10-bit HEVC encoder available (GPU NVENC or "
               "libx265) — HDR cannot be preserved; output will be "
               "tone-mapped SDR instead.")
        want_hdr = False

    # 1. the timeline the whole job runs on: CFR, still 10-bit HDR when the
    #    output should stay HDR (VFR HDR sources need this extra pass so the
    #    scan copy and the render source share exact frame times)
    hdr_src = src0
    if is_vfr:
        if want_hdr:
            hdr_src = base + ".cfr.hdr.mp4"
            if _cached(hdr_src):
                cb.log(f"      reusing existing {os.path.basename(hdr_src)}")
            else:
                hv = (["-c:v", "hevc_nvenc", "-preset", "p4", "-cq", "18",
                       "-profile:v", "main10", "-pix_fmt", "p010le"]
                      if henc == "hevc_nvenc"
                      else ["-c:v", "libx265", "-crf", "18", "-preset",
                            "fast", "-pix_fmt", "yuv420p10le"])
                keep = [a for kv in color_tags(src0).items() for a in kv]
                _encode(src0, hdr_src, cfr=True, tonemap=False,
                        vargs=hv, extra=keep + ["-tag:v", "hvc1"])
        else:
            hdr_src = None   # no HDR render: single combined pass below

    # 2. tone-mapped SDR copy for scanning (and for the output when the
    #    user chose SDR / no 10-bit encoder exists)
    sdr = base + ".sdr.mp4"
    cb.log(f"      input is {hdesc} HDR — tone-mapping a copy to SDR "
           "(BT.709) for detection"
           + ("" if want_hdr else "; output will be SDR"))
    if _cached(sdr):
        cb.log(f"      reusing existing {os.path.basename(sdr)}")
    else:
        _encode(hdr_src or src0, sdr, cfr=(is_vfr and hdr_src is None),
                tonemap=True, vargs=sdr_vargs)

    args.original_video = src0
    args.hdr_tonemapped = True
    args.video = sdr
    if want_hdr:
        args.hdr_source = hdr_src if hdr_src else src0
        args.hdr_encoder = henc
        args.hdr_tags = color_tags(args.hdr_source)
        cb.log(f"      HDR output: preserving {hdesc} — 10-bit HEVC "
               f"via {henc}")
        if henc == "libx265":
            cb.log("      NOTE: the GPU here can't encode 10-bit HEVC — "
                   "HDR will be processed on the CPU (libx265), which is "
                   "MUCH slower. Choose SDR output for full speed.")


# ----------------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------------

def build_parser():
    ap = argparse.ArgumentParser(description="Blur PHI in screen-recording videos (scroll-aware, no patient list needed).")
    ap.add_argument("video", nargs="?", help="input video (omit only with --batch)")
    ap.add_argument("-o", "--output")
    ap.add_argument("--config", help="YAML config profile (CLI flags override it)")
    ap.add_argument("--allow-names", help="text file of provider/staff names to KEEP visible")
    ap.add_argument("--extra-names", help="text file of names to always blur")
    ap.add_argument("--engine", choices=["auto", "paddle", "tesseract"], default="auto")
    ap.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto",
                    help="PaddleOCR compute device (default: gpu if available)")
    ap.add_argument("--encoder", choices=["auto", "nvenc", "x264"], default="auto",
                    help="video encoder: auto = NVENC (GPU) if available, else libx264")
    ap.add_argument("--no-ner", action="store_true", help="disable spaCy NER")
    ap.add_argument("--heuristic-names", choices=["auto", "on", "off"], default="auto",
                    help="capitalized-pair fallback: auto = on when NER unavailable")
    ap.add_argument("--sample-interval", type=float, default=0.5,
                    help="seconds between time-based OCR samples (default 0.5)")
    ap.add_argument("--scan-trigger", type=float, default=60,
                    help="also OCR after this many pixels of scroll (default 60)")
    ap.add_argument("--pad", "--blur-buffer", type=int, default=8, dest="pad",
                    help="blur buffer: pixels of blur beyond the tightly-"
                         "cropped word/face (default 8)")
    ap.add_argument("--no-vfr-fix", action="store_true",
                    help="skip automatic CFR normalization of VFR input")
    ap.add_argument("--codec", choices=["h264", "hevc"], default="h264",
                    help="video codec for the output: h264 (default, plays "
                         "everywhere) or hevc (H.265, smaller files). HDR "
                         "output always uses 10-bit HEVC regardless.")
    ap.add_argument("--hdr-output", choices=["match", "sdr"], default="match",
                    help="for HDR input: 'match' (default) keeps the output "
                         "HDR (10-bit HEVC, PQ/HLG preserved); 'sdr' "
                         "tone-maps the output to SDR BT.709. SDR input "
                         "always renders SDR — output matches the source.")
    ap.add_argument("--dense-faces", action="store_true",
                    help="run the face detector on EVERY frame (not just at "
                         "scan intervals) so fast-moving faces stay covered. "
                         "Restricted to face detection zones when zones are "
                         "set, which keeps it fast. Higher render time.")
    ap.add_argument("--dense-face-stride", type=int, default=1,
                    help="with --dense-faces, detect every Nth frame "
                         "(1 = every frame; 2-3 trades a little coverage for "
                         "speed). Default 1.")
    ap.add_argument("--plate-model", default=None,
                    help="path to a YOLOv8 license-plate ONNX model. If omitted, "
                         "OpenScrub looks for models/plate_yolov8.onnx or the "
                         "$OPENSCRUB_PLATE_MODEL env var. Plate category is "
                         "inactive without a model.")
    ap.add_argument("--face-shape", choices=["ellipse", "rect"],
                    default="ellipse",
                    help="mask shape for face redaction: ellipse hugs the "
                         "face (no blurred background corners); rect is the "
                         "classic box. Text regions are always rectangular.")
    ap.add_argument("--face-model", default=None,
                    help="path to an optional face-detection ONNX model "
                         "(CenterFace or SCRFD, auto-recognized; see "
                         "face_models.json). Falls back to $OPENSCRUB_FACE_MODEL, "
                         "then to the built-in YuNet — face detection always "
                         "works without a model file.")
    ap.add_argument("--plate-threshold", type=float, default=0.35,
                    help="license-plate detector confidence cutoff (0-1). "
                         "Default 0.35.")
    ap.add_argument("--face-threshold", type=float, default=0.6,
                    help="face detector confidence cutoff (0-1). Lower catches "
                         "more faces but risks false positives; higher is "
                         "stricter. Default 0.6 (YuNet).")
    ap.add_argument("--draw-scores", action="store_true",
                    help="with --preview, draw each face's confidence score so "
                         "you can tune --face-threshold on your own footage")
    ap.add_argument("--detect-scale", type=float, default=1.0,
                    help="run FACE detection on a downscaled copy of each frame "
                         "for speed (0.2-1.0; e.g. 0.5 = half resolution). "
                         "Output quality is unaffected. Default 1.0 (off).")
    ap.add_argument("--face-expand", type=float, default=0.15,
                    help="expand detected face boxes by this fraction before "
                         "the blur buffer is applied (default 0.15)")
    ap.add_argument("--mode", choices=["blur", "box", "mosaic"],
                    default="blur",
                    help="default redaction style: blur (reversible-ish), "
                         "box (solid black, irreversible), or mosaic "
                         "(pixelation — censored look, not recoverable)")
    ap.add_argument("--mode-map", default="",
                    help="per-category overrides, e.g. 'ssn=box,mrn=box' — "
                         "override style per category, e.g. 'ssn=box,face=mosaic'; a "
                         "risky while blurring the rest. Categories not listed "
                         "use --mode.")
    ap.add_argument("--mrn-regex", default=RE_MRN_DEFAULT)
    ap.add_argument("--scroll-track", choices=["auto", "on", "off"],
                    default="auto",
                    help="screen-scroll tracking + safety bands. 'auto' "
                         "(default) probes the footage: camera video "
                         "(continuous 2-axis motion) disables them — "
                         "detections stay screen-anchored and no unscanned-"
                         "strip bands are drawn. 'on'/'off' force it.")
    ap.add_argument("--adaptive", choices=["on", "off"], default="on",
                    help="self-tune scan pacing: stretch the sample "
                         "interval while the screen is static, tighten it "
                         "under heavy change, and scan sooner when "
                         "scrolling fast. 'off' uses the fixed values.")
    ap.add_argument("--custom-regex", action="append", default=[],
                    metavar="ID=PATTERN",
                    help="user-defined regex category (repeatable), e.g. "
                         "--custom-regex claim=CLM-\\d+ . Matches are "
                         "detections in category ID: they appear in review, "
                         "reports, per-category modes, and detection zones "
                         "like any built-in category. Add ID to --categories "
                         "to enable it.")
    ap.add_argument("--bridge-gap", type=float, default=4.0,
                    help="max seconds to bridge blur across OCR misses when the "
                         "same PHI reappears in the same region (default 4.0)")
    ap.add_argument("--no-memory", action="store_true",
                    help="disable PHI text memory (recall of previously "
                         "confirmed strings)")
    ap.add_argument("--preview", action="store_true",
                    help="draw boxes (red=PHI, orange=unscanned band) instead of blurring")
    ap.add_argument("--report", help="write JSON audit report with provenance "
                    "(contains PHI text — protect it)")
    ap.add_argument("--from-report", help="skip scanning; re-render from an "
                    "(edited) audit report produced by --report")
    ap.add_argument("--batch", help="process every video in this folder; "
                    "files whose output already exists are skipped (resume)")
    ap.add_argument("--overwrite", action="store_true",
                    help="with --batch: reprocess even if output exists")
    ap.add_argument("--backtrack-window", type=float, default=2.5,
                    help="seconds of recent frames kept for onset "
                         "backtracking (RAM: ~1MB per frame at 1440p; "
                         "default 2.5)")
    ap.add_argument("--no-backtrack", action="store_true",
                    help="disable onset backtracking (finding the exact frame "
                         "where newly detected PHI first appeared)")
    ap.add_argument("--skip-start", type=float, default=0.0,
                    help="don't detect anything during the first N seconds")
    ap.add_argument("--skip-end", type=float, default=0.0,
                    help="stop detecting N seconds before the end of the video")
    ap.add_argument("--zones", help="JSON file of per-category detection "
                    "zones (normalized 0-1 coords). Categories with zones "
                    "are ONLY detected inside them — detections outside are "
                    "dropped and counted as a warning. Categories without "
                    "zones remain full-frame.")
    ap.add_argument("--ignore-region", action="append", default=[], metavar="X1,Y1,X2,Y2",
                    help="screen region to never blur (repeatable), e.g. taskbar clock")
    ap.add_argument("--ocr-upscale", choices=["auto", "on", "off"], default="auto",
                    help="re-OCR at 2x when text is small (default auto)")
    ap.add_argument("--paranoid", action="store_true",
                    help="maximum-recall preset: dense sampling, lenient "
                         "matching, forced upscale — more false positives, "
                         "clean them up in review")
    ap.add_argument("--vfr", choices=["auto", "ignore"], default="auto",
                    help="auto: detect variable frame rate and normalize to "
                         "CFR before processing (default); ignore: skip check")
    ap.add_argument("--categories", default="name,dob,phone,ssn,mrn,email,address,card,apikey,ipaddr,plate,face")
    return ap


def _prep_args(args, parser):
    args = apply_config(args, parser)
    regions = []
    for r in (args.ignore_region or []):
        if isinstance(r, str):
            regions.append(tuple(float(v) for v in r.split(",")))
        else:
            regions.append(tuple(r))
    args.ignore_regions = getattr(args, "ignore_regions", []) or regions
    mm = {}
    raw_mm = getattr(args, "mode_map", "") or ""
    if isinstance(raw_mm, dict):
        mm = {k: v for k, v in raw_mm.items() if v in ("blur", "box", "mosaic")}
    elif raw_mm:
        for pair in str(raw_mm).replace(";", ",").split(","):
            if "=" in pair:
                k, v = pair.split("=", 1)
                k, v = k.strip().lower(), v.strip().lower()
                if v in ("blur", "box", "mosaic"):
                    mm[k] = v
    args.mode_map = mm
    args.zones_data = None
    if getattr(args, "zones", None):
        args.zones_data = load_zones(args.zones)
    if getattr(args, "paranoid", False):
        args.sample_interval = min(args.sample_interval, 0.25)
        args.scan_trigger = min(args.scan_trigger, 30)
        args.ocr_upscale = "on"
    return args


def backtrack_onset(det, buf, cx, cy, cur_small, scale=0.5,
                    ncc_min=0.6):
    """A detection first seen at scan time t may have APPEARED any time since
    the previous scan — up to a full sample interval of exposed PHI. Walk
    backwards through the recent-frame buffer comparing the detection's
    region visually (no OCR needed: we know where it is and what it looks
    like) until it vanishes; return the earliest frame index where it is
    still present, or None if it wasn't present in any buffered frame."""
    x1 = int((det.cbox[0] + cx) * scale)
    y1 = int((det.cbox[1] + cy) * scale)
    x2 = int((det.cbox[2] + cx) * scale)
    y2 = int((det.cbox[3] + cy) * scale)
    h, w = cur_small.shape[:2]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if x2 - x1 < 6 or y2 - y1 < 4:
        return None
    tmpl = cur_small[y1:y2, x1:x2]
    tstd = float(tmpl.std())
    if tstd < 4:
        return None                      # featureless: can't match reliably
    th, tw = tmpl.shape
    M = 8                                # local search margin (tolerates small
                                         # tracking error in buffered offsets)

    def present_at(small, ox_s, oy_s):
        bx1 = int((det.cbox[0]) * scale + ox_s) - M
        by1 = int((det.cbox[1]) * scale + oy_s) - M
        bx2 = bx1 + tw + 2 * M
        by2 = by1 + th + 2 * M
        bx1, by1 = max(0, bx1), max(0, by1)
        bx2, by2 = min(w, bx2), min(h, by2)
        if bx2 - bx1 < tw or by2 - by1 < th:
            return False
        region = small[by1:by2, bx1:bx2]
        if float(region.std()) < max(3.0, 0.30 * tstd):
            return False
        return float(cv2.matchTemplate(region, tmpl,
                                       cv2.TM_CCOEFF_NORMED).max()) >= ncc_min

    onset = None
    for fidx, fcx, fcy, small in reversed(buf):
        # two coordinate hypotheses per frame: the tracker's recorded offset,
        # and screen-static (offset at detection time). Page transitions feed
        # the tracker garbage offsets — content that never moved on screen
        # would otherwise be looked for in the wrong place and the walk would
        # stop, forfeiting the whole exposure window.
        if (present_at(small, fcx * scale, fcy * scale)
                or present_at(small, cx * scale, cy * scale)):
            onset = fidx
        else:
            break
    return onset


def reverse_pass(scans, memory, cats, namer, lenient=76):
    """After the scan, re-search every OCR'd word against remembered PHI at
    a more lenient threshold. Catches near-misses (OCR misreads) of strings
    already confirmed elsewhere in the video; gating rules still apply so a
    one-off false positive can't spread."""
    from rapidfuzz import fuzz
    extra = []
    for t, cum, words in scans:
        for w, cbox, conf in words:
            n = PhiMemory.norm(w)
            if len(n) < 4 or n in STOPWORDS:
                continue
            if namer and namer._allowed(w):
                continue
            for k, cat in memory.items.items():
                if cat not in cats:
                    continue
                if memory._gated(k, cat) is None:
                    continue
                if k.isdigit() or n.isdigit():
                    hit = (k.isdigit() and n.isdigit() and len(k) == len(n)
                           and sum(a != b for a, b in zip(k, n)) <= 2)
                else:
                    hit = (abs(len(k) - len(n)) <= 2
                           and fuzz.ratio(k, n) >= lenient)
                if hit:
                    extra.append(Detection(t, t,
                                           tuple(int(v) for v in cbox),
                                           cat, w, round(float(conf), 3),
                                           tuple(cum)))
                    break
    return extra


def run_scan(args, cb=None):
    """Scan pass only: OCR + face detection + tracking. Returns a state dict
    consumed by run_render (and serialized into --report files)."""
    cb = cb or Callbacks()
    if not args.video or not os.path.exists(args.video):
        raise RuntimeError(f"input not found: {args.video}")
    # normalize attributes that may be absent when args come from an older
    # embedder (GUI/web) rather than _prep_args
    for attr, default in (("ignore_regions", []), ("config", None),
                          ("from_report", None), ("face_expand", 0.15),
                          ("no_vfr_fix", False)):
        if not hasattr(args, attr):
            setattr(args, attr, default)
    normalize_vfr(args, cb)
    cats = {c.strip() for c in args.categories.split(",")}
    # everything except the pixel detectors is text: it exists only to feed
    # OCR output into detect_phi. Load ONLY what the selected categories
    # actually need — a faces-only job must not pay for PaddleOCR (seconds
    # of startup + 2-3 GB of RAM) or spaCy.
    text_cats = cats - {"face", "plate"}

    cb.log(f"[1/4] OCR engine   (openscrub v{VERSION})")
    if text_cats:
        ocr = make_ocr(args.engine, device=args.device)
        cb.log(f"      using {type(ocr).__name__}")
    else:
        ocr = None
        cb.log("      skipped — no text categories selected "
               f"({', '.join(sorted(cats))} only): detector-only scan")

    cb.log("[2/4] Detectors")
    if cuda_dnn_available():
        cb.log("      OpenCV DNN: CUDA (GPU-accelerated face detection "
               "+ identity grouping)")
    if "name" in cats:
        namer = NameDetector(allow_names=args.allow_names,
                             extra_names=args.extra_names,
                             use_ner=not args.no_ner,
                             heuristic=args.heuristic_names)
        modes = []
        if namer.nlp is not None:
            modes.append("spaCy NER")
        modes.append("label heuristic")
        if namer.heuristic:
            modes.append("capitalized-pair heuristic")
        cb.log(f"      names: {', '.join(modes)}"
               + (f" | allowlist: {len(namer.allow)} tokens" if namer.allow
                  else ""))
    else:
        namer = None
        cb.log("      names: skipped (name category not selected)")
    facer = (FaceDetector(cb, expand=args.face_expand,
                          thresh=getattr(args, "face_threshold", 0.6),
                          model_path=getattr(args, "face_model", None))
             if "face" in cats else None)
    plater = (PlateDetector(cb, model_path=getattr(args, "plate_model", None),
                            thresh=getattr(args, "plate_threshold", 0.35))
              if "plate" in cats else None)
    detect_scale = float(getattr(args, "detect_scale", 1.0) or 1.0)
    if args.ignore_regions:
        cb.log(f"      ignore regions: {len(args.ignore_regions)}")

    mrn_re = re.compile(args.mrn_regex)
    # user-defined categories: only those enabled in --categories run, and a
    # bad pattern fails the run loudly rather than silently detecting nothing
    custom_res = []
    for spec in getattr(args, "custom_regex", []) or []:
        cid, _, pat = spec.partition("=")
        cid = cid.strip().lower()
        if not cid or not pat:
            raise ValueError("--custom-regex needs ID=PATTERN, got %r" % spec)
        if cid in cats:
            custom_res.append((cid, re.compile(pat)))
    if custom_res:
        cb.log("      custom categories: "
               + ", ".join(c for c, _ in custom_res))

    cb.log(f"[3/4] Scanning (every {args.sample_interval}s or {args.scan_trigger}px of scroll"
           + (", self-tuning to screen activity and scroll speed)"
              if getattr(args, "adaptive", "on") != "off" else ")"))
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    step = max(1, int(round(fps * args.sample_interval)))
    # ignore zones ride in the zones file under the "ignore" class:
    # normalized rects where NOTHING is ever detected or blurred. They are
    # not a detection category — pop before zone filtering. Normalized
    # --ignore-region values (all <= 1.0) scale to this video too.
    if getattr(args, "zones_data", None) and "ignore" in args.zones_data:
        args.ignore_regions = list(args.ignore_regions or []) + list(
            args.zones_data.pop("ignore"))
    if args.ignore_regions:
        args.ignore_regions = [
            (r[0] * vw, r[1] * vh, r[2] * vw, r[3] * vh)
            if max(r) <= 1.0 else tuple(r)
            for r in args.ignore_regions]
    zones_px = (zones_to_pixels(args.zones_data, vw, vh)
                if getattr(args, "zones_data", None) else None)
    duration = total / fps if fps else 0
    win_start = max(0.0, float(getattr(args, "skip_start", 0) or 0))
    win_end = duration - max(0.0, float(getattr(args, "skip_end", 0) or 0))
    if win_start > 0 or win_end < duration:
        cb.log(f"      detection window: {win_start:.1f}s to {win_end:.1f}s "
               f"(of {duration:.1f}s) — nothing outside it is detected or blurred")
    zone_dropped = {}
    zdrop_raw = []
    if zones_px:
        cb.log("      detection zones active: "
               + ", ".join(f"{c} ({len(r)})" for c, r in zones_px.items()))

    scroll_mode = getattr(args, "scroll_track", "auto")
    track_on = scroll_mode != "off"
    if track_on and not text_cats:
        # scroll tracking and safety bands exist to keep unscanned TEXT
        # covered as it scrolls into view. With no text categories they can
        # only hurt: on real-world video the tracker reads camera/subject
        # motion as scrolling — drifting boxes and smearing blur bands
        # along the frame edges (the boat-video failure).
        track_on = False
        cb.log("      scroll tracking + safety bands off (no text "
               "categories selected — they only protect text content)")
    elif scroll_mode == "auto":
        _cam, _movf, _mixf = probe_camera_motion(args.video)
        if _cam:
            track_on = False
            cb.log("      camera footage detected (continuous 2-axis "
                   "motion) — scroll tracking and safety bands off; "
                   "detections are screen-anchored. Force with "
                   "--scroll-track on if this is actually a screen "
                   "recording.")
    tracker = ScrollTracker()
    memory = None if (args.no_memory or not text_cats) else PhiMemory(
        threshold=78 if getattr(args, "paranoid", False) else 82)
    cum = []
    bands = []
    raw = []
    scans = []
    cx = cy = 0.0
    last_scan_idx = -10**9
    scan_cx = scan_cy = 0.0
    n_scans = 0
    n_recalled = 0
    recall_counts = {}
    from collections import deque
    BT_SCALE = 0.5
    bt_on = not getattr(args, "no_backtrack", False)
    bt_win = max(args.sample_interval + 0.6,
                 float(getattr(args, "backtrack_window", 2.5) or 2.5))
    bt_buf = deque(maxlen=max(3, int(round(fps * bt_win))))
    prev_keys = {}
    adapt = getattr(args, "adaptive", "on") != "off"
    scan_small = None            # gray half-res frame at the last OCR scan
    prev_fcx = prev_fcy = 0.0    # last frame's scroll offset (velocity)
    bt_count, bt_gain, bt_capped = 0, 0.0, 0
    bt_deep = []   # regions still visible at the buffer's oldest frame:
                   # their true onset is found after the scan by seeking
                   # the file itself (RAM buffer can stay small)
    face_tracks = []   # forward face tracking: detect once, hold every frame
    # faces are ALWAYS detected per-frame (dense): scan-cadence face adds
    # merged with the OCR hold union a moving person's positions into one
    # body-sized box (the boat-video failure). Per-frame re-detection +
    # track smoothing is strictly better on every kind of footage; the
    # --dense-faces flag remains for compatibility, --dense-face-stride
    # still tunes the cost.
    dense_faces = "face" in cats or bool(getattr(args, "dense_faces", False))
    args.dense_faces = dense_faces   # keep camera-mode/report paths in sync
    dense_stride = max(1, int(getattr(args, "dense_face_stride", 1) or 1))
    plate_zone_px = (zones_px.get("plate") if zones_px else None)
    plate_zone_rects = plate_zone_px if plate_zone_px else None
    face_zone_px = (zones_px.get("face") if zones_px else None)
    face_zone_rects = face_zone_px if face_zone_px else None
    if dense_faces:
        cb.log("      dense faces: detecting "
               + (f"every {dense_stride} frames" if dense_stride > 1
                  else "every frame")
               + (" inside face zone(s)" if face_zone_rects
                  else " (whole frame — set a face zone to speed this up)"))
    idx = 0
    dense_now = []
    while True:
        if idx % 30 == 0 and cb.cancelled():
            cap.release()
            raise PipelineCancelled()
        ok, frame = cap.read()
        if not ok:
            break
        cx, cy = tracker.step(frame) if track_on else (0.0, 0.0)
        cum.append((cx, cy))

        t_now = idx / fps
        if t_now < win_start or t_now > win_end:
            # outside the detection window: no scans, and no safety bands
            # (the user has declared this span PHI-free)
            scan_cx, scan_cy = cx, cy
            bands.append((0.0, 0.0))
            idx += 1
            continue
        small = (cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), None,
                            fx=BT_SCALE, fy=BT_SCALE) if bt_on else None)
        if idx % max(1, dense_stride) == 0:
            dense_now = []      # latest dense boxes, drawn on the preview
        # dense face detection: find faces on THIS frame at their true screen
        # position, independent of the OCR scan cadence — per-frame
        # re-detection (not tracking), so a face moving fast across the frame
        # stays covered because the detector re-finds it wherever it is.
        if (dense_faces and facer is not None and "face" in cats
                and idx % dense_stride == 0):
            dense_hold = 0.3 * dense_stride / fps
            for (fx1, fy1, fx2, fy2, conf) in facer.find(frame, detect_scale):
                # zone-filter by the face's screen center, exactly like every
                # other category (robust; no cropped-background detector quirks)
                if face_zone_rects is not None and not in_any_zone(
                        (fx1, fy1, fx2, fy2), face_zone_rects):
                    continue
                if in_ignore_region((fx1, fy1, fx2, fy2),
                                    args.ignore_regions):
                    continue
                raw.append(Detection(
                    t_now, t_now + dense_hold,
                    (int(fx1 - cx), int(fy1 - cy),
                     int(fx2 - cx), int(fy2 - cy)),
                    "face", "face", round(conf, 3), (cx, cy),
                    dense=True))
                dense_now.append((fx1, fy1, fx2, fy2))
        # per-frame license-plate detection: plates on dashcam/CCTV footage move
        # fast, so (like dense faces) we re-detect every frame at the true
        # position rather than tracking. Only runs if a plate model is loaded.
        if (plater is not None and plater.available() and "plate" in cats
                and idx % max(1, dense_stride) == 0):
            phold = 0.3 * max(1, dense_stride) / fps
            for (px1, py1, px2, py2, pconf) in plater.find(frame, detect_scale):
                if plate_zone_rects is not None and not in_any_zone(
                        (px1, py1, px2, py2), plate_zone_rects):
                    continue
                if in_ignore_region((px1, py1, px2, py2),
                                    args.ignore_regions):
                    continue
                raw.append(Detection(
                    t_now, t_now + phold,
                    (int(px1 - cx), int(py1 - cy),
                     int(px2 - cx), int(py2 - cy)),
                    "plate", "plate", round(pconf, 3), (cx, cy),
                    dense=True))
                dense_now.append((px1, py1, px2, py2))
        moved = abs(cx - scan_cx) + abs(cy - scan_cy)
        step_eff, trig_eff = step, args.scan_trigger
        if adapt:
            # Self-tuning pace: when the screen has barely changed since the
            # last scan, stretch the interval (2x) — nothing new to read, and
            # memory/safety bands still backstop. Under heavy change, tighten
            # it (0.5x) so new content is read sooner. Fast scrolling lowers
            # the scroll trigger so scans fire earlier in the movement.
            if scan_small is not None and scan_small.shape == small.shape:
                act = float(cv2.absdiff(small, scan_small).mean())
                if act < 1.0:
                    step_eff = step * 2
                elif act > 8.0:
                    step_eff = max(2, step // 2)
            speed = (abs(cx - prev_fcx) + abs(cy - prev_fcy)) * fps
            if speed > 300:
                trig_eff = max(20.0, args.scan_trigger * 0.5)
        prev_fcx, prev_fcy = cx, cy
        due = (idx - last_scan_idx >= step_eff) or (moved >= trig_eff)
        if due and idx - last_scan_idx >= 2:
            t = idx / fps
            scan_small = small
            if ocr is not None:
                words = read_adaptive(ocr, frame,
                                      getattr(args, "ocr_upscale", "auto"))
                lines = group_lines(words)
                found = detect_phi(words, lines, t, (cx, cy), namer, mrn_re,
                                   custom_res)
                found = [d for d in found if d.category in cats]
            else:
                words, found = [], []   # detector-only scan: no text pass

            # in dense mode the frame loop already detects faces on EVERY
            # frame — adding scan-time copies only feeds merge_detections
            # unions that balloon across a moving face's path
            if facer is not None and not dense_faces:
                for (fx1, fy1, fx2, fy2, conf) in facer.find(frame, detect_scale):
                    found.append(Detection(t, t,
                                           (int(fx1 - cx), int(fy1 - cy),
                                            int(fx2 - cx), int(fy2 - cy)),
                                           "face", "face", round(conf, 3),
                                           (cx, cy)))

            if memory is not None:
                primary_found = list(found)
                flagged = {tuple(b) for _, b, _ in
                           ((d.text, (d.cbox[0] + cx, d.cbox[1] + cy,
                                      d.cbox[2] + cx, d.cbox[3] + cy), 0)
                            for d in found)}
                for w, box, conf in words:
                    if tuple(box) in flagged:
                        continue
                    cat = memory.recall(w)
                    if cat and cat in cats and not (namer and namer._allowed(w)):
                        cbox = (int(box[0] - cx), int(box[1] - cy),
                                int(box[2] - cx), int(box[3] - cy))
                        found.append(Detection(t, t, cbox, cat, w,
                                               round(float(conf), 3), (cx, cy)))
                        n_recalled += 1
                        k = PhiMemory.norm(w)
                        recall_counts[k] = recall_counts.get(k, 0) + 1
                # only primary detections build memory (recalls must not
                # self-reinforce a false positive)
                for d in primary_found:
                    memory.add(d.text, d.category, primary=True)

            if args.ignore_regions:
                found = [d for d in found if not in_ignore_region(
                    (d.cbox[0] + cx, d.cbox[1] + cy,
                     d.cbox[2] + cx, d.cbox[3] + cy), args.ignore_regions)]

            if zones_px:
                kept = []
                for d in found:
                    rects = zones_px.get(d.category)
                    sb = (d.cbox[0] + cx, d.cbox[1] + cy,
                          d.cbox[2] + cx, d.cbox[3] + cy)
                    if getattr(d, "dense", False):
                        kept.append(d)          # dense faces pre-filtered
                    elif rects and not in_any_zone(sb, rects):
                        zone_dropped[d.category] = zone_dropped.get(d.category, 0) + 1
                        zdrop_raw.append(d)
                    else:
                        kept.append(d)
                found = kept

            if bt_on:
                # "New" must be POSITIONAL: the same patient name can already
                # be on screen in a list row when it also appears in a chart
                # banner after a click — that banner is a new appearance and
                # needs backtracking even though the text isn't new.
                def _bt_key(d):
                    return (d.category,
                            PhiMemory.norm(d.text) if d.text else "")

                def _center(d):
                    return ((d.cbox[0] + d.cbox[2]) / 2,
                            (d.cbox[1] + d.cbox[3]) / 2)
                cur_keys = {}
                for d in found:
                    cur_keys.setdefault(_bt_key(d), []).append(_center(d))
                for d in found:
                    cx0, cy0 = _center(d)
                    seen_near = any(
                        abs(cx0 - px) < 120 and abs(cy0 - py) < 120
                        for px, py in prev_keys.get(_bt_key(d), []))
                    if seen_near:
                        continue
                    onset = backtrack_onset(d, bt_buf, cx, cy, small, BT_SCALE)
                    if onset is not None:
                        if bt_buf and onset == bt_buf[0][0]:
                            bt_capped += 1   # visible beyond the buffer —
                            # queue for the post-scan deep backtrack
                            tx1 = int((d.cbox[0] + cx) * BT_SCALE)
                            ty1 = int((d.cbox[1] + cy) * BT_SCALE)
                            tx2 = int((d.cbox[2] + cx) * BT_SCALE)
                            ty2 = int((d.cbox[3] + cy) * BT_SCALE)
                            sh_, sw_ = small.shape[:2]
                            tx1, ty1 = max(0, tx1), max(0, ty1)
                            tx2, ty2 = min(sw_, tx2), min(sh_, ty2)
                            if tx2 - tx1 >= 6 and ty2 - ty1 >= 4:
                                bt_deep.append(
                                    (d, cx, cy,
                                     small[ty1:ty2, tx1:tx2].copy()))
                        new_start = max(win_start, onset / fps - 0.12)
                        if new_start < d.t_start - 0.01:
                            bt_count += 1
                            bt_gain += d.t_start - new_start
                            d.t_start = new_start
                prev_keys = cur_keys
            if bt_on:
                for d in found:
                    if d.category == "face" and dense_faces:
                        continue   # dense mode re-detects faces every frame
                    if d.category != "face" and not (
                            d.text and len(PhiMemory.norm(d.text)) >= 3):
                        continue
                    sx1 = int((d.cbox[0] + cx) * BT_SCALE)
                    sy1 = int((d.cbox[1] + cy) * BT_SCALE)
                    sx2 = int((d.cbox[2] + cx) * BT_SCALE)
                    sy2 = int((d.cbox[3] + cy) * BT_SCALE)
                    hh, ww = small.shape[:2]
                    sx1, sy1 = max(0, sx1), max(0, sy1)
                    sx2, sy2 = min(ww, sx2), min(hh, sy2)
                    if sx2 - sx1 < 8 or sy2 - sy1 < 8:
                        continue
                    tmpl = small[sy1:sy2, sx1:sx2].copy()

                    dn = PhiMemory.norm(d.text) if d.text else ""
                    dcx = (d.cbox[0] + d.cbox[2]) / 2
                    dcy = (d.cbox[1] + d.cbox[3]) / 2
                    for tr in face_tracks:
                        if tr["cat"] != d.category:
                            continue
                        if tr["norm"] == dn:
                            same_text = True
                        elif dn and tr["norm"]:
                            from rapidfuzz import fuzz as _f
                            same_text = _f.ratio(dn, tr["norm"]) >= 85
                        else:
                            same_text = (not dn and not tr["norm"])
                        near = (abs(dcx - tr["c"][0]) < 100
                                and abs(dcy - tr["c"][1]) < 100)
                        if same_text and near:
                            tr.update(tmpl=tmpl, cbox=d.cbox, last_ok=t,
                                      conf=d.confidence, c=(dcx, dcy),
                                      text=d.text)
                            break
                    else:
                        if len(face_tracks) < 500:
                            face_tracks.append({
                                "tmpl": tmpl, "cbox": d.cbox, "cat": d.category,
                                "text": d.text, "norm": dn, "c": (dcx, dcy),
                                "last_ok": t, "last_emit": t,
                                "conf": d.confidence})
            raw.extend(found)
            if ocr is not None:
                scans.append((t, (cx, cy),
                              [(w, (b[0] - cx, b[1] - cy,
                                    b[2] - cx, b[3] - cy), c)
                               for w, b, c in words]))
            if ocr is not None and n_scans == 0 and len(words) < 3:
                # First-scan sanity: a text-dense frame that OCR read nothing
                # from means a broken OCR setup — say so in minute one, not
                # after a 40-minute render.
                _edges = cv2.Canny(small, 60, 180)
                if float((_edges > 0).mean()) > 0.02:
                    cb.log("      *** OCR SANITY WARNING: the first scan read "
                           "almost no text, but the frame looks text-dense. "
                           "If this recording contains text to redact, check "
                           "the OCR setup (run openscrub-setup, or install "
                           "PaddleOCR) before trusting this run.")
            n_scans += 1
            last_scan_idx = idx
            scan_cx, scan_cy = cx, cy
            tracker.anchor()
            if found:
                cb.log(f"  t={t:7.2f}s  {len(found)} PHI region(s): "
                       + ", ".join(sorted({d.category for d in found})))
            if cb.wants_frames:
                shown = frame.copy()
                for d in found:
                    cv2.rectangle(shown,
                                  (int(d.cbox[0] + cx), int(d.cbox[1] + cy)),
                                  (int(d.cbox[2] + cx), int(d.cbox[3] + cy)),
                                  (0, 0, 255), 2)
                # dense faces/plates bypass `found`, so camera footage lost
                # its red boxes on the live preview — draw the current
                # frame's dense detections too (display only)
                for (dx1, dy1, dx2, dy2) in dense_now:
                    cv2.rectangle(shown, (int(dx1), int(dy1)),
                                  (int(dx2), int(dy2)), (0, 0, 255), 2)
                cb.scan_frame(shown, t, len(found) + len(dense_now))
            cb.progress("scan", idx, total)
        bands.append((cx - scan_cx, cy - scan_cy) if track_on
                     else (0.0, 0.0))
        if bt_on and face_tracks and not dense_faces:
            t_now2 = idx / fps
            hh, ww = small.shape[:2]
            MM = 6
            for tr in list(face_tracks):
                b = tr["cbox"]
                th_, tw_ = tr["tmpl"].shape
                bx1 = int((b[0] + cx) * BT_SCALE) - MM
                by1 = int((b[1] + cy) * BT_SCALE) - MM
                bx2 = bx1 + tw_ + 2 * MM
                by2 = by1 + th_ + 2 * MM
                bx1, by1 = max(0, bx1), max(0, by1)
                bx2, by2 = min(ww, bx2), min(hh, by2)
                ok = False
                if bx2 - bx1 >= tw_ and by2 - by1 >= th_:
                    region = small[by1:by2, bx1:bx2]
                    if float(region.std()) > 3:
                        # TM_CCOEFF_NORMED is invariant to uniform dimming:
                        # a Please-Wait overlay cannot break the track.
                        # Text needs a STRICTER bar than faces — two different
                        # short words in the same UI font correlate ~0.6, and
                        # a track that keeps matching after its word was
                        # replaced extends last_seen past the switch, pairing
                        # the region's text with frames of a different name.
                        thr = 0.58 if tr["cat"] == "face" else 0.74
                        ok = float(cv2.matchTemplate(
                            region, tr["tmpl"],
                            cv2.TM_CCOEFF_NORMED).max()) > thr
                        if not ok and tr["cat"] != "face" and tw_ >= 24:
                            # partial occlusion (a cursor parked on the word):
                            # either half still matching means the word is
                            # still there — keep covering ALL of it
                            for half in (tr["tmpl"][:, :tw_ // 2],
                                         tr["tmpl"][:, tw_ // 2:]):
                                if (float(half.std()) > 4 and float(
                                        cv2.matchTemplate(
                                            region, half,
                                            cv2.TM_CCOEFF_NORMED).max())
                                        > thr):
                                    ok = True
                                    break
                if ok:
                    tr["last_ok"] = t_now2
                    if (t_now2 - tr["last_emit"] >= 0.3
                            and win_start <= t_now2 <= win_end):
                        raw.append(Detection(t_now2, t_now2,
                                             tuple(tr["cbox"]), tr["cat"],
                                             tr["text"],
                                             round(tr["conf"], 3), (cx, cy)))
                        tr["last_emit"] = t_now2
                elif t_now2 - tr["last_ok"] > 0.8:
                    face_tracks[:] = [x for x in face_tracks if x is not tr]
        if bt_on:
            bt_buf.append((idx, cx, cy, small))
        idx += 1
    cap.release()

    if memory is not None:
        extra = reverse_pass(scans, memory, cats, namer,
                             lenient=72 if getattr(args, "paranoid", False) else 76)
        if args.ignore_regions:
            extra = [d for d in extra if not in_ignore_region(
                (d.cbox[0] + d.aoff[0], d.cbox[1] + d.aoff[1],
                 d.cbox[2] + d.aoff[0], d.cbox[3] + d.aoff[1]),
                args.ignore_regions)]
        if zones_px:
            kept = []
            for d in extra:
                rects = zones_px.get(d.category)
                sb = (d.cbox[0] + d.aoff[0], d.cbox[1] + d.aoff[1],
                      d.cbox[2] + d.aoff[0], d.cbox[3] + d.aoff[1])
                if rects and not in_any_zone(sb, rects):
                    zone_dropped[d.category] = zone_dropped.get(d.category, 0) + 1
                    zdrop_raw.append(d)
                else:
                    kept.append(d)
            extra = kept
        if extra:
            cb.log(f"      reverse pass: {len(extra)} additional near-miss "
                   "region(s) from remembered PHI")
            raw.extend(extra)
    hold = args.sample_interval + 0.3
    from rapidfuzz import fuzz as _fuzz
    if bt_on and bt_deep:
        # Deep backtrack: the RAM buffer only reaches --backtrack-window
        # seconds back, but the video file reaches all the way to frame 0.
        # For each region still visible at the buffer's edge, seek the file
        # backwards (exponential probe, then binary search) with the same
        # visual match the buffered walk uses, until its true first frame
        # is found. No knobs to raise — it goes as far back as it needs to.
        cap2 = cv2.VideoCapture(args.video)
        deep_n, deep_gain = 0, 0.0
        for d, dcx, dcy, tmpl in bt_deep:
            if float(tmpl.std()) < 4:
                continue
            th_, tw_ = tmpl.shape
            M = 10

            def _visible(t, d=d, dcx=dcx, dcy=dcy, tmpl=tmpl,
                         th_=th_, tw_=tw_):
                cap2.set(cv2.CAP_PROP_POS_MSEC, max(t, 0.0) * 1000)
                ok, fr = cap2.read()
                if not ok:
                    return False
                sm = cv2.resize(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), None,
                                fx=BT_SCALE, fy=BT_SCALE)
                sh, sw = sm.shape[:2]
                bx1 = int((d.cbox[0] + dcx) * BT_SCALE) - M
                by1 = int((d.cbox[1] + dcy) * BT_SCALE) - M
                bx2, by2 = bx1 + tw_ + 2 * M, by1 + th_ + 2 * M
                bx1, by1 = max(0, bx1), max(0, by1)
                bx2, by2 = min(sw, bx2), min(sh, by2)
                if bx2 - bx1 < tw_ or by2 - by1 < th_:
                    return False
                reg = sm[by1:by2, bx1:bx2]
                if float(reg.std()) < max(3.0, 0.30 * float(tmpl.std())):
                    return False
                return float(cv2.matchTemplate(
                    reg, tmpl, cv2.TM_CCOEFF_NORMED).max()) >= 0.6

            hi, step, lo, at_start = d.t_start, 1.0, None, False
            for _ in range(11):          # doubling covers ~34 min of video
                t = hi - step
                if t <= win_start + 0.01:
                    at_start = _visible(win_start)
                    lo = win_start
                    break
                if _visible(t):
                    hi, step = t, step * 2
                else:
                    lo = t
                    break
            if lo is None:
                lo = max(win_start, hi - step)
            if not at_start:
                for _ in range(6):       # binary search to ~0.15s
                    if hi - lo <= 0.15:
                        break
                    mid = (lo + hi) / 2
                    if _visible(mid):
                        hi = mid
                    else:
                        lo = mid
            new_start = max(win_start,
                            (win_start if at_start else hi) - 0.12)
            if new_start < d.t_start - 0.05:
                deep_n += 1
                deep_gain += d.t_start - new_start
                d.t_start = new_start
        cap2.release()
        if deep_n:
            cb.log(f"      deep backtrack: found the true onset of {deep_n} "
                   f"region(s) beyond the buffer, closing another "
                   f"{deep_gain:.2f}s of would-be exposure")

    _gap_stats = {"checked": 0, "bridged": 0}
    _gap_cap = {"cap": None}

    def _gap_check(m, t_from, t_to):
        """Visual gap verification: template-match the region at points
        inside [t_from, t_to]. True only if the content is present at ALL
        sampled points — then the dropout was an OCR miss and the blur may
        bridge it, however long the gap. Any point where it's absent or
        changed means the content genuinely went away: refuse. Dense
        detections never reach here (they never merge)."""
        if _gap_cap["cap"] is None:
            _gap_cap["cap"] = cv2.VideoCapture(args.video)
        cap3 = _gap_cap["cap"]

        def _frame_small(t):
            cap3.set(cv2.CAP_PROP_POS_MSEC, max(t, 0.0) * 1000)
            ok, fr = cap3.read()
            if not ok:
                return None
            return cv2.resize(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), None,
                              fx=BT_SCALE, fy=BT_SCALE)
        ref = _frame_small(t_from)
        if ref is None:
            return False
        ox, oy = m.aoff
        x1 = int((m.cbox[0] + ox) * BT_SCALE)
        y1 = int((m.cbox[1] + oy) * BT_SCALE)
        x2 = int((m.cbox[2] + ox) * BT_SCALE)
        y2 = int((m.cbox[3] + oy) * BT_SCALE)
        rh, rw = ref.shape[:2]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(rw, x2), min(rh, y2)
        if x2 - x1 < 6 or y2 - y1 < 4:
            return False
        tmpl = ref[y1:y2, x1:x2]
        if float(tmpl.std()) < 4:
            return False                 # featureless: cannot verify
        th_, tw_ = tmpl.shape
        M = 10
        n_pts = min(5, max(2, int(t_to - t_from)))
        _gap_stats["checked"] += 1
        for i in range(1, n_pts + 1):
            t = t_from + (t_to - t_from) * i / (n_pts + 1)
            sm = _frame_small(t)
            if sm is None:
                return False
            bx1, by1 = max(0, x1 - M), max(0, y1 - M)
            bx2, by2 = min(rw, x1 + tw_ + M), min(rh, y1 + th_ + M)
            if bx2 - bx1 < tw_ or by2 - by1 < th_:
                return False
            reg = sm[by1:by2, bx1:bx2]
            if float(cv2.matchTemplate(reg, tmpl,
                                       cv2.TM_CCOEFF_NORMED).max()) < 0.6:
                return False             # absent or changed: do not bridge
        _gap_stats["bridged"] += 1
        return True

    detections = merge_detections(raw, hold=hold, scans=scans,
                                  bridge_gap=args.bridge_gap, fuzz=_fuzz,
                                  gap_check=_gap_check)
    zdropped = merge_detections(zdrop_raw, hold=hold, scans=scans,
                                bridge_gap=args.bridge_gap, fuzz=_fuzz,
                                gap_check=_gap_check) if zdrop_raw else []
    if _gap_cap["cap"] is not None:
        _gap_cap["cap"].release()
    n_tracks = assign_dense_tracks(detections)
    if n_tracks:
        cb.log(f"      dense tracking: {sum(1 for d in detections if d.dense)}"
               f" per-frame samples grouped into {n_tracks} track(s) "
               "for review")
        cb.log(f"      dense continuity: smoothing {n_tracks} track(s) — "
               "interpolating flicker gaps and walking each onset back "
               "through the video (the slow part of a dense scan)…")
        n_gaps, n_lead, lead_s = smooth_dense_tracks(
            detections, fps, args.video, cum=cum, win_start=win_start,
            cb=cb)
        if n_gaps or n_lead or lead_s:
            cb.log(f"      dense continuity: {n_gaps} flicker gap(s) "
                   f"interpolated; onsets walked back {lead_s:.2f}s total "
                   f"({n_lead} pre-detection sample(s) matched in the file)")
        try:
            n_emb, n_people = group_persons(detections, args.video, cb)
            if n_people:
                cb.log(f"      person grouping: {n_emb} face track(s) "
                       f"identity-matched into {n_people} person(s) — "
                       "review shows one card per person")
        except Exception as e:
            cb.log(f"      person grouping unavailable ({e}) — review "
                   "shows per-track cards instead")
    if _gap_stats["checked"]:
        cb.log(f"      gap verification: {_gap_stats['checked']} long gap(s) "
               f"checked against the file, {_gap_stats['bridged']} verified "
               "and bridged")
    mem_note = (f" | {n_recalled} memory recalls, "
                f"{len(memory.items)} strings remembered" if memory else "")
    cb.log(f"      {n_scans} {'OCR' if ocr is not None else 'detector'} "
           f"scans | {len(raw)} raw hits -> "
           f"{len(detections)} merged regions{mem_note}")
    if bt_count:
        cb.log(f"      backtrack: {bt_count} region(s) start moved earlier "
               f"(avg {bt_gain / bt_count:.2f}s of would-be exposure closed)")
    if bt_capped and not bt_deep:
        cb.log(f"      note: {bt_capped} region(s) were visible beyond the "
               "backtrack buffer and could not be deep-searched "
               "(featureless region) — earlier coverage relies on "
               "gap bridging")
    if zone_dropped:
        cb.log("      *** ZONE WARNING: "
               + ", ".join(f"{c} x{n}" for c, n in sorted(zone_dropped.items()))
               + " detection(s) fell OUTSIDE their category's zones and were "
                 "NOT blurred. Verify your zones actually cover all PHI. ***")
    if recall_counts:
        top = sorted(recall_counts.items(), key=lambda kv: -kv[1])[:8]
        cb.log("      top recalled strings (check for false positives): "
               + ", ".join(f"'{k}'x{v}" for k, v in top))

    return {"fps": fps, "cum": cum, "bands": bands, "detections": detections,
            "zdropped": zdropped,
            "input_sha256": sha256_file(args.video),
            "stats": {"scans": n_scans, "raw_hits": len(raw),
                      "regions": len(detections), "recalls": n_recalled,
                      "remembered": len(memory.items) if memory else 0,
                      "zone_dropped": zone_dropped}}


def run_render(args, state, cb=None):
    cb = cb or Callbacks()
    dst = args.output or os.path.splitext(args.video)[0] + (
        "_preview.mp4" if args.preview else "_redacted.mp4")
    cb.log(f"[4/4] Rendering -> {dst}")
    if getattr(args, "hdr_source", None) and not args.preview:
        render_hdr(args.hdr_source, dst, state["detections"], state["cum"],
                   state["bands"], state["fps"], pad=args.pad, mode=args.mode,
                   encoder=args.hdr_encoder,
                   tags=getattr(args, "hdr_tags", {}),
                   face_shape=getattr(args, "face_shape", "ellipse"),
                   mode_map=getattr(args, "mode_map", None), cb=cb)
    else:
        render(args.video, dst, state["detections"], state["cum"],
               state["bands"],
               state["fps"], pad=args.pad, mode=args.mode, preview=args.preview,
               mode_map=getattr(args, "mode_map", None),
               draw_scores=bool(getattr(args, "draw_scores", False)),
               vcodec=getattr(args, "codec", "h264"),
               face_shape=getattr(args, "face_shape", "ellipse"),
               encoder=args.encoder, cb=cb)
    if args.report:
        write_report(args.report, args, state, output_path=dst)
        cb.log(f"      audit report: {args.report} (contains PHI text — protect it)")
    cb.log("done.")
    return dict(state["stats"], output=dst)


def run_pipeline(args, cb=None):
    """Scan + render (or render-only with --from-report). Returns a summary
    dict; raises PipelineCancelled if cb cancels."""
    cb = cb or Callbacks()
    if getattr(args, "from_report", None):
        cb.log(f"[1/2] Loading detections from {args.from_report}")
        dets, rstate, prov = load_report(args.from_report)
        orig = prov.get("original_input")
        if orig and os.path.exists(orig) \
                and os.path.abspath(orig) != os.path.abspath(args.video):
            # re-renders must start from the TRUE original so intake can
            # re-derive the HDR/CFR context (cached intermediates are
            # reused). Rendering straight from the tone-mapped scan copy
            # silently downgraded HDR jobs to SDR output.
            args.video = orig
        normalize_vfr(args, cb)
        if rstate is None:
            raise RuntimeError("report has no render_state — re-run a scan "
                               "with --report using openscrub v4+")
        cb.log(f"      {len(dets)} enabled detections")
        in_sha = sha256_file(args.video)
        if prov.get("input_sha256") and prov["input_sha256"] != in_sha:
            cb.log("      WARNING: input file differs from the one this report "
                   "was made from (sha256 mismatch) — blur positions may be wrong")
        state = {"fps": rstate["fps"],
                 "cum": [tuple(v) for v in rstate["cum"]],
                 "bands": [tuple(v) for v in rstate["bands"]],
                 "detections": dets, "input_sha256": in_sha,
                 "stats": {"scans": 0, "raw_hits": 0, "regions": len(dets),
                           "recalls": 0, "remembered": 0}}
        return run_render(args, state, cb)
    state = run_scan(args, cb)
    return run_render(args, state, cb)


def _batch(args, parser):
    exts = (".mp4", ".mkv", ".mov", ".avi", ".webm")
    files = sorted(f for f in os.listdir(args.batch)
                   if f.lower().endswith(exts)
                   and "_redacted" not in f and "_preview" not in f)
    if not files:
        raise RuntimeError(f"no videos found in {args.batch}")
    print(f"Batch: {len(files)} video(s) in {args.batch}")
    summary = []
    for i, name in enumerate(files, 1):
        path = os.path.join(args.batch, name)
        base = os.path.splitext(path)[0]
        print(f"\n=== [{i}/{len(files)}] {name} ===")
        a = argparse.Namespace(**vars(args))
        a.video = path
        a.output = base + "_redacted.mp4"
        a.report = base + "_audit.json"
        a.batch = None
        if os.path.exists(a.output) and not args.overwrite:
            print("skipping (output exists — use --overwrite to redo)")
            summary.append({"file": name, "ok": True, "skipped": True})
            continue
        try:
            res = run_pipeline(a)
            summary.append(dict(res, file=name, ok=True))
        except Exception as e:
            print(f"FAILED: {e}")
            summary.append({"file": name, "ok": False, "error": str(e)})
    out = os.path.join(args.batch, "batch_summary.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"tool": "openscrub", "version": VERSION,
                   "timestamp": datetime.datetime.now().astimezone().isoformat(),
                   "results": summary}, f, indent=2)
    ok = sum(1 for s in summary if s.get("ok"))
    print(f"\nBatch complete: {ok}/{len(files)} succeeded. Summary: {out}")


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        args = _prep_args(args, parser)
        if args.batch:
            _batch(args, parser)
        elif not args.video:
            parser.error("provide a video file or --batch FOLDER")
        else:
            run_pipeline(args)
    except RuntimeError as e:
        sys.exit(str(e))


if __name__ == "__main__":
    main()

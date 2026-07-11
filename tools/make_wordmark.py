#!/usr/bin/env python3
"""Regenerate the typeset OpenScrub wordmarks (navy + white).

    python tools/make_wordmark.py

"Open" and "ub" sharp Poppins Bold; "Scr" Gaussian-blurred at reduced
opacity; red corner brackets around the Scr block. Outputs
assets/wordmark_navy.png and assets/wordmark_white.png at ~2000px wide.
Tweakables are the constants below.
"""

import os
from PIL import Image, ImageDraw, ImageFont, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
A = os.path.join(HERE, "..", "assets")
FONT = os.path.join(A, "fonts", "Poppins-Bold.ttf")

CAP = 340              # cap height in px (controls overall size)
BLUR = 0.055           # Scr blur radius as a fraction of CAP
SCR_ALPHA = 168        # Scr ink opacity (0-255)
BRACKET_T = 0.05       # bracket stroke thickness (fraction of CAP)
BRACKET_L = 0.22       # bracket arm length (fraction of CAP)
RED = (229, 57, 53, 255)


def rect(d, xa, ya, xb, yb, fill):
    d.rectangle([min(xa, xb), min(ya, yb), max(xa, xb), max(ya, yb)], fill=fill)


def wordmark(ink, out_path):
    font = ImageFont.truetype(FONT, CAP)
    tmp = ImageDraw.Draw(Image.new("RGBA", (10, 10)))

    def measure(t):
        b = tmp.textbbox((0, 0), t, font=font)
        return b[2] - b[0], b[3] - b[1], b[1]

    W1, H1, off = measure("Open")
    W2, _, _ = measure("Scr")
    W3, _, _ = measure("ub")
    pad = 110
    gap = int(CAP * 0.07)
    W = pad + W1 + gap + W2 + gap + W3 + pad
    H = H1 + 2 * pad
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    y = pad - off
    d.text((pad, y), "Open", font=font, fill=ink)
    scr = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(scr).text((pad + W1 + gap, y), "Scr", font=font,
                             fill=ink + (SCR_ALPHA,))
    img.alpha_composite(scr.filter(ImageFilter.GaussianBlur(CAP * BLUR)))
    d.text((pad + W1 + gap + W2 + gap, y), "ub", font=font, fill=ink)

    t = int(CAP * BRACKET_T)
    L = int(CAP * BRACKET_L)
    x0 = pad + W1 + int(gap * 0.15)
    x1 = pad + W1 + gap + W2 + int(gap * 0.9)
    y0 = pad - int(CAP * 0.10)
    y1 = pad + H1 + int(CAP * 0.10)
    rect(d, x0, y0, x0 + L, y0 + t, RED); rect(d, x0, y0, x0 + t, y0 + L, RED)
    rect(d, x1 - L, y0, x1, y0 + t, RED); rect(d, x1 - t, y0, x1, y0 + L, RED)
    rect(d, x1 - L, y1 - t, x1, y1, RED); rect(d, x1 - t, y1 - L, x1, y1, RED)
    rect(d, x0, y1 - t, x0 + L, y1, RED); rect(d, x0, y1 - L, x0 + t, y1, RED)

    b = img.getbbox()
    img = img.crop((max(0, b[0] - 40), max(0, b[1] - 40), b[2] + 40, b[3] + 40))
    img.save(out_path)
    return img.size


if __name__ == "__main__":
    print("navy:", wordmark((15, 23, 42), os.path.join(A, "wordmark_navy.png")))
    print("white:", wordmark((255, 255, 255), os.path.join(A, "wordmark_white.png")))
    print("run tools/make_icons.py next to rebuild the lockup + social preview")

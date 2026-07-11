#!/usr/bin/env python3
"""Regenerate every icon/logo asset from assets/badge_master.png.

    python tools/make_icons.py

Outputs (all into assets/): icon-{16..1024}.png, favicon.ico,
openscrub.ico, logo.png, logo_dark.png, lockup_light.png,
social_preview.png. Wordmarks come from tools/make_wordmark.py — run that
first if you changed the wordmark.
"""

import os
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
A = os.path.join(HERE, "..", "assets")


def square(img, size):
    s = size / max(img.size)
    nw, nh = max(1, round(img.width * s)), max(1, round(img.height * s))
    r = img.resize((nw, nh), Image.LANCZOS)
    c = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    c.paste(r, ((size - nw) // 2, (size - nh) // 2), r)
    return c


def main():
    badge = Image.open(os.path.join(A, "badge_master.png"))

    for s in (16, 32, 48, 64, 128, 180, 192, 256, 512, 1024):
        square(badge, s).save(os.path.join(A, f"icon-{s}.png"))

    m = square(badge, 256)
    m.save(os.path.join(A, "favicon.ico"), format="ICO",
           sizes=[(16, 16), (32, 32), (48, 48)])
    m.save(os.path.join(A, "openscrub.ico"), format="ICO",
           sizes=[(s, s) for s in (16, 32, 48, 64, 128, 256)])
    square(badge, 256).save(os.path.join(A, "logo.png"))
    square(badge, 256).save(os.path.join(A, "logo_dark.png"))

    # social preview: prefer the hand-made full-bleed master art; fall back
    # to compositing badge + white wordmark on a navy field.
    master = os.path.join(A, "social_preview_master.png")
    if os.path.exists(master):
        Image.open(master).convert("RGB").resize(
            (1280, 640), Image.LANCZOS).save(
            os.path.join(A, "social_preview.png"))

    # lockup: badge + navy wordmark on white
    wm = Image.open(os.path.join(A, "wordmark_navy.png"))
    bh = int(wm.height * 2.1)
    bs = badge.resize((int(badge.width * bh / badge.height), bh), Image.LANCZOS)
    W = bs.width + 70 + wm.width + 90
    H = max(bs.height, wm.height) + 120
    lock = Image.new("RGB", (W, H), (255, 255, 255))
    lock.paste(bs, (45, (H - bs.height) // 2), bs)
    lock.paste(wm, (bs.width + 95, (H - wm.height) // 2), wm)
    lock.save(os.path.join(A, "lockup_light.png"))

    # social preview fallback when no master art exists: navy field,
    # badge + white wordmark, centered pair
    if not os.path.exists(master):
        wmw = Image.open(os.path.join(A, "wordmark_white.png"))
        sp = Image.new("RGBA", (1280, 640), (15, 23, 42, 255))
        bd = square(badge, 430)
        tw = 700
        wr = wmw.resize((tw, int(wmw.height * tw / wmw.width)), Image.LANCZOS)
        x0 = (1280 - (bd.width + 60 + wr.width)) // 2
        sp.paste(bd, (x0, (640 - bd.height) // 2), bd)
        sp.paste(wr, (x0 + bd.width + 60, (640 - wr.height) // 2), wr)
        sp.convert("RGB").save(os.path.join(A, "social_preview.png"))

    print("assets regenerated from badge_master.png")
    print("NOTE: the web header/favicon are base64-embedded in openscrub_web.py;")
    print("if the badge changed, re-embed (see CLAUDE.md 'Web app' notes).")


if __name__ == "__main__":
    main()

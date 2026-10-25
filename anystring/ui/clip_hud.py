"""The clip card: a standing "F8 snap a clip" hint, and the card that replaces it while a
clip is saved.

Drawn on the display image only, after the frame went to the clip buffer and the demo
recorder, so neither the hint nor the animation ever ends up in a video. Every animation is a
pure function of the capture time ``t`` and the stamps in ``ClipStatus``: a short white flash
when F8 is pressed, the card sliding in from the right edge with a progress bar, the check mark
drawing itself when the file is written, then the card sliding away and the hint fading back.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

__all__ = ["ClipStatus", "card_visibility", "draw_clip_hud"]

FONT = cv2.FONT_HERSHEY_SIMPLEX
PANEL = (35, 30, 30)
TEXT = (245, 245, 245)
MUTED = (185, 181, 173)
TRACK = (80, 76, 72)
AMBER = (0, 165, 255)
GREEN = (80, 220, 80)
RED = (60, 60, 230)
KEYCAP = (225, 225, 225)
KEYCAP_TEXT = (30, 30, 30)

CARD_W, CARD_H = 300, 92
MARGIN = 16
TOP = 236  # below the calibration banner and the song strip
SLIDE_IN_S = 0.35
SLIDE_OUT_S = 0.30
HOLD_SAVED_S = 4.0  # the card stays this long after the file is written
HOLD_FAILED_S = 6.0
FLASH_S = 0.22
FLASH_PEAK = 0.40
CHECK_S = 0.30  # the check mark draws itself over this long
RING_S = 0.50  # and a ring spreads from the icon
SPIN_PER_S = 1.2  # turns per second of the "saving" arc


@dataclass(frozen=True)
class ClipStatus:
    """Where the latest clip save is. Times are on the camera capture clock."""

    phase: Literal["saving", "saved", "failed"]
    started: float  # F8 pressed
    finished: float | None = None  # save ended (saved or failed)
    progress: float = 0.0  # 0..1 while saving
    seconds: float = 0.0  # length of the clip
    folder: str = ""  # shown as "memories/"
    name: str = ""  # saved file name
    error: str | None = None


def card_visibility(clip: ClipStatus | None, t: float) -> float:
    """0 = card hidden (hint shown), 1 = card fully in. Eases in, holds, eases out."""
    if clip is None or t < clip.started:
        return 0.0
    vis = _ease_out((t - clip.started) / SLIDE_IN_S)
    if clip.finished is None:
        return vis
    hold = HOLD_FAILED_S if clip.phase == "failed" else HOLD_SAVED_S
    leaving = (t - clip.finished - hold) / SLIDE_OUT_S
    return vis * (1.0 - _ease_in(leaving))


def draw_clip_hud(img: np.ndarray, clip: ClipStatus | None, t: float) -> None:
    """Draw the hint, flash and card onto the display image, in place."""
    h, w = img.shape[:2]
    y = min(TOP, max(8, h - CARD_H - 8))
    vis = card_visibility(clip, t)
    if vis < 1.0:
        _hint(img, w, y, 1.0 - vis)
    if clip is None:
        return
    _flash(img, t - clip.started)
    if vis > 0.0:
        x = w - MARGIN - CARD_W + round((1.0 - vis) * (CARD_W + MARGIN))
        _card(img, clip, t, x, y)


# -- pieces ---------------------------------------------------------------------------------


def _hint(img: np.ndarray, w: int, y: int, alpha: float) -> None:
    """A key cap and a label, quiet enough to ignore while playing."""
    cap_w, height, pad = 44, 34, 8
    label = "snap a clip"
    (tw, th), _ = cv2.getTextSize(label, FONT, 0.5, 1)
    width = pad + cap_w + 10 + tw + 14
    x0 = w - MARGIN - width
    x1 = w - MARGIN
    layer = img.copy()
    _rrect(layer, x0, y, x1, y + height, height // 2, PANEL)
    _rrect(layer, x0 + pad, y + 5, x0 + pad + cap_w, y + height - 5, 6, KEYCAP)
    cv2.putText(layer, "F8", (x0 + pad + 9, y + 22), FONT, 0.5, KEYCAP_TEXT, 2, cv2.LINE_AA)
    cv2.putText(
        layer, label, (x0 + pad + cap_w + 10, y + 17 + th // 2), FONT, 0.5, TEXT, 1, cv2.LINE_AA
    )
    a = 0.62 * alpha
    cv2.addWeighted(layer, a, img, 1.0 - a, 0, dst=img)


def _flash(img: np.ndarray, since: float) -> None:
    if not 0.0 <= since < FLASH_S:
        return
    a = FLASH_PEAK * (1.0 - since / FLASH_S) ** 2
    cv2.addWeighted(img, 1.0 - a, img, 0.0, 255.0 * a, dst=img)


def _card(img: np.ndarray, clip: ClipStatus, t: float, x: int, y: int) -> None:
    accent = {"saving": AMBER, "saved": GREEN, "failed": RED}[clip.phase]
    layer = img.copy()
    _rrect(layer, x, y, x + CARD_W, y + CARD_H, 12, PANEL)
    cv2.addWeighted(layer, 0.9, img, 0.1, 0, dst=img)
    cv2.rectangle(img, (x + 1, y + 14), (x + 4, y + CARD_H - 14), accent, -1)

    cx, cy = x + 36, y + CARD_H // 2
    since = 0.0 if clip.finished is None else t - clip.finished
    if clip.phase == "saving":
        turn = ((t - clip.started) * SPIN_PER_S) % 1.0 * 360.0
        cv2.circle(img, (cx, cy), 17, TRACK, 3, cv2.LINE_AA)
        cv2.ellipse(img, (cx, cy), (17, 17), 0, turn, turn + 110, AMBER, 3, cv2.LINE_AA)
    elif clip.phase == "saved":
        _saved_icon(img, cx, cy, since)
    else:
        cv2.circle(img, (cx, cy), 17, RED, -1, cv2.LINE_AA)
        for dx in (-1, 1):
            cv2.line(img, (cx - 6 * dx, cy - 6), (cx + 6 * dx, cy + 6), TEXT, 3, cv2.LINE_AA)

    tx, room = x + 66, CARD_W - 66 - 14
    title = {"saving": "Saving clip", "saved": "Clip saved", "failed": "Clip failed"}[clip.phase]
    cv2.putText(img, title, (tx, y + 34), FONT, 0.7, TEXT, 2, cv2.LINE_AA)
    where = f"{clip.folder}/" if clip.folder else ""
    if clip.phase == "saving":
        _fit(img, f"last {clip.seconds:.0f} s -> {where}", tx, y + 56, room, MUTED)
        top = y + 68
        cv2.rectangle(img, (tx, top), (tx + room, top + 5), TRACK, -1)
        fill = round(room * min(max(clip.progress, 0.0), 1.0))
        cv2.rectangle(img, (tx, top), (tx + fill, top + 5), AMBER, -1)
    elif clip.phase == "saved":
        _fit(img, f"{clip.seconds:.0f} s stored in {where}", tx, y + 56, room, MUTED)
        _fit(img, clip.name, tx, y + 76, room, TEXT, 0.5)
    else:
        _fit(img, clip.error or "could not write the file", tx, y + 56, room, MUTED)


def _saved_icon(img: np.ndarray, cx: int, cy: int, since: float) -> None:
    """A green disc, its check mark drawn stroke by stroke, and one ring spreading out."""
    cv2.circle(img, (cx, cy), 17, GREEN, -1, cv2.LINE_AA)
    pts = np.array([(-7.0, 0.0), (-2.0, 6.0), (8.0, -6.0)]) + (cx, cy)
    p = min(max(since / CHECK_S, 0.0), 1.0)
    first = min(p / 0.4, 1.0)  # short stroke down-right, then the long stroke up
    second = min(max((p - 0.4) / 0.6, 0.0), 1.0)
    mid = pts[0] + (pts[1] - pts[0]) * first
    path = [pts[0], mid]
    if second > 0.0:
        path = [pts[0], pts[1], pts[1] + (pts[2] - pts[1]) * second]
    cv2.polylines(img, [np.round(path).astype(np.int32)], False, KEYCAP_TEXT, 3, cv2.LINE_AA)
    q = since / RING_S
    if 0.0 <= q < 1.0:
        fade = tuple(int(c * (1.0 - q)) for c in GREEN)
        cv2.circle(img, (cx, cy), 17 + int(16 * _ease_out(q)), fade, 2, cv2.LINE_AA)


def _fit(
    img: np.ndarray,
    text: str,
    x: int,
    y: int,
    room: int,
    colour: tuple[int, int, int],
    scale: float = 0.45,
) -> None:
    """Draw ``text`` in ``room`` pixels: shrunk a little, then cut with an ellipsis."""
    floor = min(scale, 0.4)
    while scale > floor and cv2.getTextSize(text, FONT, scale, 1)[0][0] > room:
        scale = max(floor, scale - 0.03)
    while len(text) > 4 and cv2.getTextSize(text, FONT, scale, 1)[0][0] > room:
        text = text[:-3].rstrip() + ".."
    cv2.putText(img, text, (x, y), FONT, scale, colour, 1, cv2.LINE_AA)


def _rrect(img: np.ndarray, x0: int, y0: int, x1: int, y1: int, r: int, colour) -> None:
    """A filled rounded rectangle."""
    cv2.rectangle(img, (x0 + r, y0), (x1 - r, y1), colour, -1, cv2.LINE_AA)
    cv2.rectangle(img, (x0, y0 + r), (x1, y1 - r), colour, -1, cv2.LINE_AA)
    for cx, cy in ((x0 + r, y0 + r), (x1 - r, y0 + r), (x0 + r, y1 - r), (x1 - r, y1 - r)):
        cv2.circle(img, (cx, cy), r, colour, -1, cv2.LINE_AA)


def _ease_out(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return 1.0 - (1.0 - x) ** 3


def _ease_in(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x**3

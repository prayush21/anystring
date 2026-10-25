"""Compact song cues drawn over the camera without resizing its image."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np

from racketguitar.song.player import SongPlayer, SongState, TimedLine

FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT = (245, 245, 240)
MUTED = (185, 181, 173)
GREEN = (90, 230, 120)
AMBER = (0, 165, 255)
RED = (70, 70, 240)
HUD_WIDTH, HUD_HEIGHT = 1280, 224
LYRIC_X, LYRIC_Y, LYRIC_WIDTH, LYRIC_HEIGHT = 354, 64, 884, 126
LYRIC_FOCUS_Y = 52.0
LYRIC_ROW_HEIGHT = 34
SCROLL_SECONDS = 0.4


def _text(img, text, x, y, scale=0.55, colour=TEXT, thickness=1, width=None):
    """Fit labels to their allotted space; lyrics use wrapping instead."""
    if width is not None:
        measured = cv2.getTextSize(text, FONT, scale, thickness)[0][0]
        scale *= min(1.0, width / max(1, measured))
    cv2.putText(img, text, (x, y), FONT, scale, colour, thickness, cv2.LINE_AA)


def wrap_lyric(text: str, width: int, scale: float) -> list[tuple[int, str]]:
    """Wrap at words, retaining source columns for the chords above each word."""
    rows = []
    start = 0
    while start < len(text):
        end = start
        while end < len(text):
            size = cv2.getTextSize(text[start : end + 1], FONT, scale, 1)[0][0]
            if size > width and end > start:
                break
            end += 1
        if end < len(text):
            space = text.rfind(" ", start, end + 1)
            if space > start:
                end = space
        rows.append((start, text[start:end]))
        start = end
        while start < len(text) and text[start].isspace():
            start += 1
    return rows


@dataclass(frozen=True)
class LyricBlock:
    beat: float
    beats: float
    rows: tuple[str, ...]
    centre: float


@lru_cache(maxsize=8)
def _lyric_blocks(lines: tuple[TimedLine, ...]) -> tuple[LyricBlock, ...]:
    """Wrap once at the foreground size, so fading/scaling never reflows the words.

    Very long sheet lines use successive two-row pages spread over that line's time;
    a sheet has no word timestamps. Those pages scroll just like separate lyrics.
    """
    blocks = []
    centre = 0.0
    previous_height = 0
    for line in lines:
        rows = wrap_lyric(line.text, LYRIC_WIDTH - 8, 1.05) or [(0, "Instrumental")]
        pages = (len(rows) + 1) // 2
        for page in range(pages):
            text = tuple(text for _, text in rows[page * 2 : page * 2 + 2])
            height = len(text) * LYRIC_ROW_HEIGHT
            if blocks:
                centre += (previous_height + height) / 2 + 18
            blocks.append(
                LyricBlock(line.beat + page * line.beats / pages, line.beats / pages, text, centre)
            )
            previous_height = height
    return tuple(blocks)


def _lyric_positions(player: SongPlayer, beat: float) -> list[tuple[LyricBlock, float, float]]:
    """Block, vertical centre and focus weight, derived only from the song clock.

    A newly due lyric eases up from below while the old one recedes above. No UI
    history or wall clock: seeking, restarting and dropped display frames stay exact.
    """
    blocks = _lyric_blocks(player.timeline.lines)
    i = max(0, bisect_right([block.beat for block in blocks], beat) - 1)
    current = blocks[i]
    duration = min(SCROLL_SECONDS / player.seconds_per_beat, current.beats * 0.45)
    fraction = min(1.0, max(0.0, (beat - current.beat) / duration)) if i else 1.0
    eased = fraction * fraction * (3 - 2 * fraction)
    previous = blocks[max(0, i - 1)]
    scroll = previous.centre + (current.centre - previous.centre) * eased
    result = []
    for j in range(max(0, i - 2), min(len(blocks), i + 3)):
        block = blocks[j]
        focus = eased if j == i else 1 - eased if j == i - 1 else 0.0
        result.append((block, LYRIC_FOCUS_Y + block.centre - scroll, focus))
    return result


def _draw_scrolling_lyrics(hud: np.ndarray, player: SongPlayer, st: SongState) -> None:
    viewport = hud[LYRIC_Y : LYRIC_Y + LYRIC_HEIGHT, LYRIC_X : LYRIC_X + LYRIC_WIDTH]
    # Clip and fade at the viewport edges; lyrics never paint over title or controls.
    edge = np.minimum(np.arange(LYRIC_HEIGHT), np.arange(LYRIC_HEIGHT)[::-1])
    fade = np.clip(edge / 14.0, 0, 1)[:, None]
    for block, centre, focus in _lyric_positions(player, st.beat):
        mask = np.zeros((LYRIC_HEIGHT, LYRIC_WIDTH), np.uint8)
        scale = 0.89 + 0.16 * focus
        for row, text in enumerate(block.rows):
            y = round(centre + 10 + (row - (len(block.rows) - 1) / 2) * LYRIC_ROW_HEIGHT)
            cv2.putText(mask, text, (0, y), FONT, scale, 255, 2, cv2.LINE_AA)
        if focus < 0.99:
            mask = cv2.GaussianBlur(mask, (3, 3), 0.65 * (1 - focus))
        alpha = (mask / 255.0 * fade * (0.38 + 0.62 * focus))[..., None]
        viewport[:] = np.rint(viewport * (1 - alpha) + np.array(TEXT) * alpha).astype(np.uint8)


def draw_song_hud(
    img: np.ndarray, player: SongPlayer, st: SongState, held: str | None, recording: bool
) -> None:
    """Only modify the top strip; camera dimensions and all racket coordinates stay intact."""
    height = min(img.shape[0], round(img.shape[1] * HUD_HEIGHT / HUD_WIDTH))
    hud = cv2.resize(img[:height], (HUD_WIDTH, HUD_HEIGHT))
    shade = hud.copy()
    cv2.rectangle(shade, (16, 12), (1264, 220), (25, 25, 24), -1)
    cv2.addWeighted(shade, 0.86, hud, 0.14, 0, dst=hud)

    song = player.song
    title = song.title + (f" / {song.artist}" if song.artist else "")
    _text(hud, title, 38, 43, 0.6, width=650)
    info = f"{player.bpm:.0f} BPM  /  {player.speed:.0%}  /  capo {song.capo}"
    _text(hud, info, 880, 43, 0.46, MUTED, width=280)
    if recording:
        cv2.circle(hud, (1190, 37), 5, RED, -1, cv2.LINE_AA)
        _text(hud, "REC", 1203, 43, 0.46, RED)
    cv2.line(hud, (38, 58), (1242, 58), (75, 71, 64), 1)

    target = st.chord or (st.next_chord if st.phase != "done" else None)
    slot = st.slots.index(target) + 1 if target in st.slots else None
    _text(hud, "PLAY", 38, 83, 0.45, MUTED)
    _text(hud, target or "--", 38, 139, 2.0, AMBER, 3, width=130)
    _text(hud, f"Grip {slot}" if slot is not None else "Finished", 38, 165, 0.52, MUTED)
    _text(
        hud,
        f"You: {held or 'open'}",
        38,
        190,
        0.46,
        GREEN if target is not None and held == target else MUTED,
        width=135,
    )

    nxt = st.next_chord if st.phase == "playing" else None
    _text(hud, "NEXT", 195, 83, 0.45, MUTED)
    _text(hud, nxt or "--", 195, 130, 1.4, GREEN, 2, width=110)
    if nxt is not None and st.beats_to_next is not None:
        _text(hud, f"{max(0, st.beats_to_next):.1f} beats", 195, 165, 0.5, MUTED)
    cv2.line(hud, (324, 76), (324, 190), (75, 71, 64), 1)

    lines = player.timeline.lines
    idx = st.line if st.line is not None else 0
    line = lines[idx]
    if st.phase == "playing":
        _draw_scrolling_lyrics(hud, player, st)
    else:
        if st.phase == "ready":
            headline, subline = "Ready when you are", "SPACE to start  /  Grip the first chord"
        elif st.phase == "countin":
            headline = "GO!" if st.count == 0 else f"Starting in {st.count}"
            subline = line.text or "Instrumental"
        else:
            headline, subline = "Song complete", "SPACE to play again"
        _text(hud, headline, 354, 115, 1.3, GREEN, 2, width=884)
        _text(hud, subline, 354, 162, 0.65, MUTED, width=884)

    elapsed = min(player.timeline.total_beats, max(0.0, st.beat))
    total = player.timeline.total_beats
    seconds = elapsed * player.seconds_per_beat
    duration = total * player.seconds_per_beat
    clock = (
        f"{int(seconds) // 60}:{int(seconds) % 60:02d} / "
        f"{int(duration) // 60}:{int(duration) % 60:02d}"
    )
    section = f"{line.section}  /  " if line.section else ""
    _text(hud, section + clock, 354, 204, 0.43, MUTED, width=410)
    action = "SPACE stop/reset" if st.phase in ("playing", "countin") else "SPACE start"
    _text(hud, f"{action}    S voice    X exit song", 875, 204, 0.42, MUTED, width=367)
    if elapsed > 0:
        cv2.line(hud, (38, 218), (38 + int(1204 * elapsed / total), 218), AMBER, 2)
    img[:height] = cv2.resize(hud, (img.shape[1], height), interpolation=cv2.INTER_AREA)

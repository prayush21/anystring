"""The play screen: the mirrored camera feed with what the detectors are testing against.

Every shape is computed in the racket frame (or the un-mirrored camera frame) with the
same transform the detectors use, then mirrored for display, so what you see is what
the app tests. ``Overlay.draw`` is pure apart from the fingertip trail it remembers.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np

from racketguitar.config import Config
from racketguitar.play.engine import EngineSnapshot
from racketguitar.song.player import SongPlayer, SongState, TimedLine
from racketguitar.types import RacketPose

__all__ = ["Status", "CalibrationView", "View", "Overlay", "BGR"]

BGR = {
    "accent": (0, 165, 255),  # armed, down-strum flash, active chord
    "up": (255, 200, 40),  # up-strum flash
    "string": (175, 175, 175),
    "wound": (70, 150, 215),  # bronze: the three low (wound) strings
    "plain": (235, 235, 235),  # silver: the three high (plain) strings
    "muted": (90, 90, 90),
    "grey": (140, 140, 140),
    "ok": (80, 220, 80),
    "lost": (60, 60, 230),
    "racket": (210, 210, 210),
    "text": (245, 245, 245),
    "shadow": (20, 20, 20),
    "rec": (40, 40, 230),
    "next": (90, 230, 120),  # song mode: the section to grip next
    "panel": (35, 30, 30),
}
FONT = cv2.FONT_HERSHEY_SIMPLEX
ONSET_TICK_S = 0.3


@dataclass(frozen=True)
class Status:
    fps: float = 0.0
    vision_ms: float | None = None  # processing time of the latest frame
    latency_ms: float | None = None  # latest detect-to-audio time
    gate_mode: str = "grip"  # "grip", "mic", "grip+mic" or "free"
    mic_level: float | None = None  # band RMS, None when the mic is off
    onsets: tuple[float, ...] = ()  # recent onset times, for meter ticks
    recording: bool = False
    message: str | None = None
    grip_score: float | None = None  # strumming hand's grip score (lower = closed)
    grip_band: tuple[float, float] | None = None  # (close_below, open_above)


@dataclass(frozen=True)
class CalibrationView:
    instruction: str
    progress: float
    error: str | None = None
    warning: str | None = None


@dataclass(frozen=True, eq=False)
class View:
    """Everything one overlay frame needs, published by the vision thread."""

    image: np.ndarray  # un-mirrored BGR camera frame
    t: float
    pose: RacketPose
    snapshot: EngineSnapshot
    status: Status = field(default_factory=Status)
    calibration: CalibrationView | None = None
    track_points: np.ndarray | None = None  # features the racket tracker follows
    song: SongPlayer | None = None  # song mode: the loaded song (read only here)
    song_state: SongState | None = None  # ... and where it is this frame


class Overlay:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self._trail: deque[np.ndarray] = deque(maxlen=max(2, cfg.ui.trail_frames))
        self._width = 0

    def draw(
        self,
        view: View,
        clicks: tuple[tuple[float, float], ...] = (),
        prompt: str | None = None,
        debug: bool = False,
    ) -> np.ndarray:
        """``clicks`` (frame pixels) and ``prompt`` show a racket lock in progress;
        ``debug`` adds the tracked points and the live grip score."""
        img = cv2.flip(view.image, 1) if self.cfg.ui.mirror else view.image.copy()
        self._width = img.shape[1]
        snap = view.snapshot
        self._lock(img, view, clicks, prompt, debug)
        st = view.song_state
        next_slot = st.next_slot if st is not None and st.phase != "done" else None
        if snap.frame is not None:
            self._racket(img, snap)
            self._handle(img, snap, next_slot)
            self._strings(img, snap, view.t)
        self._fingertips(img, snap)
        self._grip_badge(img, snap)
        self._chord(img, snap)
        self._readout(img, view.status, snap)
        if debug and view.track_points is not None and len(view.track_points):
            for x, y in self.px(view.track_points):  # on top of the string band
                cv2.circle(img, (int(x), int(y)), 2, BGR["ok"], -1)
        if debug or view.calibration is not None:  # calibration shows the score live
            self._grip_meter(img, view.status)
        self._meter(img, view.status, view.t)
        if view.song is not None and st is not None:
            self._song(img, view.song, st)
        if view.calibration is not None:
            self._calibration(img, view.calibration)
        return img

    # -- geometry -----------------------------------------------------------------------

    def px(self, pts: np.ndarray) -> np.ndarray:
        """Un-mirrored pixel points (..., 2) -> display pixel ints."""
        p = np.asarray(pts, dtype=float).copy()
        if self.cfg.ui.mirror:
            p[..., 0] = self._width - 1 - p[..., 0]
        return np.round(p).astype(np.int32)

    # -- layers -------------------------------------------------------------------------

    def _lock(
        self,
        img: np.ndarray,
        view: View,
        clicks: tuple[tuple[float, float], ...],
        prompt: str | None,
        debug: bool,
    ) -> None:
        for i, c in enumerate(clicks):
            p = tuple(int(v) for v in self.px(np.array(c)))
            cv2.circle(img, p, 8, BGR["accent"], 2, cv2.LINE_AA)
            _text(img, str(i + 1), (p[0] + 10, p[1] - 10), 0.6, BGR["accent"])
        if prompt:
            cv2.rectangle(
                img, (0, img.shape[0] - 70), (img.shape[1], img.shape[0]), BGR["shadow"], -1
            )
            _text(img, f"{prompt}   (l restarts)", (24, img.shape[0] - 28), 0.8, BGR["accent"])
        elif not view.pose.visible:
            _text(
                img,
                "racket lost: press l and click it again",
                (16, img.shape[0] - 16),
                0.6,
                BGR["lost"],
            )

    def _racket(self, img: np.ndarray, snap: EngineSnapshot) -> None:
        frame = snap.frame
        assert frame is not None
        a = np.linspace(0, 2 * np.pi, 72)
        hw = self.cfg.strings.head_half_width
        head = np.c_[0.5 + 0.5 * np.cos(a), hw * np.sin(a)]
        cv2.polylines(img, [self.px(frame.to_px(head))], True, BGR["racket"], 2, cv2.LINE_AA)
        shaft = frame.to_px(np.array([[0.0, 0.0], [snap.handle.u_at(1.0), 0.0]]))
        cv2.line(img, *map(tuple, self.px(shaft)), BGR["racket"], 2, cv2.LINE_AA)

    def _handle(self, img: np.ndarray, snap: EngineSnapshot, next_slot: int | None) -> None:
        sections = snap.handle_sections_px()
        if sections is None:
            return
        frame = snap.frame
        assert frame is not None
        side = frame.v * 0.14 * frame.head_len
        for i, (seg, name) in enumerate(zip(sections, snap.chord_names, strict=True)):
            active = snap.zone == i
            upcoming = next_slot == i and not active
            colour = BGR["accent"] if active else BGR["next"] if upcoming else BGR["grey"]
            a, b = self.px(seg)
            if upcoming:  # a glow, so the next grip reads at a glance
                cv2.line(img, tuple(a), tuple(b), BGR["shadow"], 15, cv2.LINE_AA)
            cv2.line(img, tuple(a), tuple(b), colour, 9 if active or upcoming else 5, cv2.LINE_AA)
            mid = self.px(seg.mean(axis=0) + side)
            big = active or upcoming
            _text(img, name, tuple(mid - (12, -8)), 0.8 if big else 0.6, colour, 2 if big else 1)
            if upcoming:
                _text(img, "next", tuple(mid - (12, 18)), 0.5, colour)

    def _strings(self, img: np.ndarray, snap: EngineSnapshot, t: float) -> None:
        segs = snap.strings_px()
        if segs is None:
            return
        frame = snap.frame
        assert frame is not None
        # A darkened band behind the strings, so they read against a bright string bed.
        sc = self.cfg.strings
        band = np.array(
            [
                [sc.u_min - 0.03, sc.v_min - 0.05],
                [sc.u_max + 0.03, sc.v_min - 0.05],
                [sc.u_max + 0.03, sc.v_max + 0.05],
                [sc.u_min - 0.03, sc.v_max + 0.05],
            ]
        )
        shade = img.copy()
        cv2.fillPoly(shade, [self.px(frame.to_px(band))], BGR["shadow"], cv2.LINE_AA)
        cv2.addWeighted(shade, 0.45, img, 0.55, 0, dst=img)
        # Guitar-like strings: thicker low strings, bronze wound and silver plain, each
        # with a dark outline.
        for s, seg in enumerate(segs):
            a, b = (tuple(p) for p in self.px(seg))
            width = (4, 4, 3, 3, 2, 2)[s] if s < 6 else 2
            colour = BGR["wound"] if s < 3 else BGR["plain"]
            if snap.muted:
                colour = BGR["muted"]
            cv2.line(img, a, b, BGR["shadow"], width + 3, cv2.LINE_AA)
            cv2.line(img, a, b, colour, width, cv2.LINE_AA)
        flash_s = self.cfg.ui.flash_ms / 1000.0
        for rec in snap.recent:
            age = t - rec.hit.t
            if not rec.played or not 0.0 <= age < flash_s:
                continue
            strength = (1.0 - age / flash_s) * (0.4 + 0.6 * rec.velocity / 127.0)
            colour = BGR["accent"] if rec.hit.direction == "down" else BGR["up"]
            glow = tuple(int(c * strength) for c in colour)
            a, b = (tuple(p) for p in self.px(segs[rec.hit.string]))
            cv2.line(img, a, b, glow, 9, cv2.LINE_AA)
            cv2.line(img, a, b, colour, 4, cv2.LINE_AA)
        if snap.muted:
            centre = self.px(segs[:, 0].mean(axis=0))
            _text(img, "muted", tuple(centre), 0.7, BGR["grey"])

    def _fingertips(self, img: np.ndarray, snap: EngineSnapshot) -> None:
        tips = snap.fingertips_px()
        if tips is None or snap.strumming is None:
            self._trail.clear()
            return
        self._trail.append(tips[0])
        colour = BGR["accent"] if snap.grip == "closed" else BGR["grey"]
        pts = self.px(np.array(self._trail))
        for i in range(1, len(pts)):
            fade = i / len(pts)
            c = tuple(int(v * fade) for v in colour)
            cv2.line(img, tuple(pts[i - 1]), tuple(pts[i]), c, 2, cv2.LINE_AA)
        cv2.circle(img, tuple(pts[-1]), 6, colour, -1, cv2.LINE_AA)

    def _grip_badge(self, img: np.ndarray, snap: EngineSnapshot) -> None:
        """The strumming hand outlined, with a badge: ARMED (pick grip) or OFF."""
        hand = snap.strumming
        if hand is None:
            return
        armed = snap.grip == "closed"
        colour = BGR["accent"] if armed else BGR["grey"]
        hull = cv2.convexHull(self.px(hand.landmarks))
        cv2.polylines(img, [hull], True, colour, 3 if armed else 2, cv2.LINE_AA)
        label = {"closed": "ARMED", "open": "OFF", "unknown": "?"}[snap.grip]
        top = hull.reshape(-1, 2)[:, 1].min()
        cx = int(hull.reshape(-1, 2)[:, 0].mean())
        (w, h), _ = cv2.getTextSize(label, FONT, 0.9, 2)
        x0, y1 = cx - w // 2 - 10, max(h + 14, int(top) - 10)
        cv2.rectangle(img, (x0, y1 - h - 14), (x0 + w + 20, y1), colour, -1, cv2.LINE_AA)
        cv2.putText(img, label, (x0 + 10, y1 - 7), FONT, 0.9, BGR["shadow"], 2, cv2.LINE_AA)

    def _chord(self, img: np.ndarray, snap: EngineSnapshot) -> None:
        # "open" would read like the open hand, so open strings show as "no chord".
        name = snap.chord or "-"
        (w, _), _ = cv2.getTextSize(name, FONT, 2.0, 4)
        x = img.shape[1] - w - 24
        _text(img, name, (x, 70), 2.0, BGR["accent"] if snap.chord else BGR["grey"], 4)
        if snap.chord is None:
            _text(img, "no chord", (img.shape[1] - 130, 98), 0.6, BGR["grey"])

    def _readout(self, img: np.ndarray, s: Status, snap: EngineSnapshot) -> None:
        lat = "--" if s.latency_ms is None else f"{s.latency_ms:.0f}"
        vis = "--" if s.vision_ms is None else f"{s.vision_ms:.1f}"
        _text(img, f"{s.fps:4.0f} fps  vision {vis} ms  latency {lat} ms", (16, 28), 0.55)
        strum = {"closed": "ARMED", "open": "off", "unknown": "no hand"}[snap.grip]
        _text(img, f"gate: {s.gate_mode}   strum: {strum}", (16, 52), 0.55)
        if s.recording:
            cv2.circle(img, (24, 74), 7, BGR["rec"], -1, cv2.LINE_AA)
            _text(img, "REC", (38, 80), 0.55, BGR["rec"])
        if s.message:
            _text(img, s.message, (16, img.shape[0] - 44), 0.6)

    def _grip_meter(self, img: np.ndarray, s: Status) -> None:
        """Debug: the live grip score against the close/open thresholds."""
        if s.grip_band is None:
            return
        lo, hi = s.grip_band
        x0, x1, y = 16, 316, 120
        top = max(hi * 1.6, (s.grip_score or 0.0) * 1.1, 1e-6)

        def at(v: float) -> int:
            return x0 + int((x1 - x0) * min(max(v / top, 0.0), 1.0))

        cv2.rectangle(img, (x0, y - 8), (x1, y + 8), BGR["shadow"], -1)
        cv2.rectangle(img, (x0, y - 8), (at(lo), y + 8), (0, 90, 140), -1)
        cv2.line(img, (at(lo), y - 12), (at(lo), y + 12), BGR["accent"], 2)
        cv2.line(img, (at(hi), y - 12), (at(hi), y + 12), BGR["grey"], 2)
        if s.grip_score is not None:
            cv2.circle(img, (at(s.grip_score), y), 7, BGR["text"], -1, cv2.LINE_AA)
            score = f"grip {s.grip_score:.2f}  arm < {lo:.2f}  off > {hi:.2f}"
        else:
            score = "grip: no strumming hand"
        _text(img, score, (x0, y + 30), 0.5)

    def _meter(self, img: np.ndarray, s: Status, t: float) -> None:
        if s.mic_level is None:
            return
        h, x0, y0 = 120, img.shape[1] - 30, img.shape[0] - 20
        db = 20 * np.log10(max(s.mic_level, 1e-6))
        fill = int(h * np.clip((db + 80) / 80, 0, 1))
        cv2.rectangle(img, (x0, y0 - h), (x0 + 12, y0), BGR["grey"], 1)
        cv2.rectangle(img, (x0, y0 - fill), (x0 + 12, y0), BGR["ok"], -1)
        if any(0.0 <= t - o < ONSET_TICK_S for o in s.onsets):
            cv2.rectangle(img, (x0 - 8, y0 - h), (x0 - 3, y0), BGR["accent"], -1)

    # -- song mode ----------------------------------------------------------------------

    def _song(self, img: np.ndarray, player: SongPlayer, st: SongState) -> None:
        if st.phase in ("countin", "playing"):
            self._lyrics(img, player, st)
        if st.phase == "countin" and st.count is not None:
            label = "GO!" if st.count == 0 else str(st.count)
            (w, h), _ = cv2.getTextSize(label, FONT, 5.0, 10)
            org = ((img.shape[1] - w) // 2, (img.shape[0] + h) // 2)
            _text(img, label, org, 5.0, BGR["next"] if st.count == 0 else BGR["accent"], 10)
        elif st.phase in ("ready", "done"):
            self._song_card(img, player, st)

    def _song_card(self, img: np.ndarray, player: SongPlayer, st: SongState) -> None:
        song = player.song
        by = f" by {song.artist}" if song.artist else ""
        chords = "  ".join(dict.fromkeys(c.name for c in player.timeline.chords))
        speed = f"{player.bpm:.0f} BPM ({player.speed:.0%} of {song.tempo:g})"
        capo = f"   capo {song.capo}" if song.capo else ""
        if st.phase == "ready":
            rows = [
                (f"{song.title}{by} is ready", 0.9, BGR["text"]),
                (f"{speed}{capo}   starts on {st.next_chord}", 0.6, BGR["grey"]),
                (f"Chords: {chords}", 0.6, BGR["grey"]),
                (
                    "Grip the green section. Say 'start' or press SPACE when you are.",
                    0.65,
                    BGR["next"],
                ),
            ]
        else:
            rows = [
                (f"That was {song.title}{by}", 0.9, BGR["text"]),
                ("Say 'again' or press SPACE to play it again.", 0.65, BGR["next"]),
            ]
        w, h = img.shape[1], img.shape[0]
        box_w, box_h = min(w - 40, 900), 40 + 38 * len(rows)
        x0, y0 = (w - box_w) // 2, h // 2 - box_h // 2
        shade = img.copy()
        cv2.rectangle(shade, (x0, y0), (x0 + box_w, y0 + box_h), BGR["panel"], -1)
        cv2.addWeighted(shade, 0.8, img, 0.2, 0, dst=img)
        cv2.rectangle(img, (x0, y0), (x0 + box_w, y0 + box_h), BGR["next"], 2, cv2.LINE_AA)
        for i, (text, scale, colour) in enumerate(rows):
            _text(img, text, (x0 + 24, y0 + 44 + 38 * i), scale, colour, 2 if i == 0 else 1)

    def _lyrics(self, img: np.ndarray, player: SongPlayer, st: SongState) -> None:
        """The current lyric with its chords over the words, the next line below, and a
        bar of beats: what is playing and how long until the next change."""
        lines = player.timeline.lines
        i = st.line if st.line is not None else -1
        current = lines[i] if i >= 0 else None
        upcoming = lines[i + 1] if i + 1 < len(lines) else None
        w = img.shape[1]
        y0, y1 = 96, 250
        shade = img.copy()
        cv2.rectangle(shade, (0, y0), (w, y1), BGR["panel"], -1)
        cv2.addWeighted(shade, 0.7, img, 0.3, 0, dst=img)
        x = 32
        if current is not None:
            _text(img, current.section, (x, y0 + 22), 0.5, BGR["grey"])
            self._lyric_line(img, current, (x, y0 + 82), 0.9, st.chord)
            frac = (st.beat - current.beat) / current.beats if current.beats else 0.0
            cv2.rectangle(img, (x, y0 + 94), (w - x, y0 + 98), BGR["grey"], -1)
            fill = x + int((w - 2 * x) * min(max(frac, 0.0), 1.0))
            cv2.rectangle(img, (x, y0 + 94), (fill, y0 + 98), BGR["accent"], -1)
        if upcoming is not None:
            nxt_y = y0 + 140 if current is not None else y0 + 82
            self._lyric_line(img, upcoming, (x, nxt_y), 0.65, None, dim=True)
        if st.next_chord is not None and st.beats_to_next is not None:
            beats = max(0.0, st.beats_to_next)
            note = f"next {st.next_chord} in {beats:.0f}" if beats >= 0.5 else st.next_chord
            (tw, _), _ = cv2.getTextSize(note, FONT, 0.7, 2)
            _text(img, note, (w - tw - 32, y0 + 22), 0.7, BGR["next"], 2)

    def _lyric_line(
        self,
        img: np.ndarray,
        line: TimedLine,
        org: tuple[int, int],
        scale: float,
        playing: str | None,
        dim: bool = False,
    ) -> None:
        x, y = org
        thick = 1 if dim else 2
        colour = BGR["grey"] if dim else BGR["text"]
        if line.text:
            _text(img, line.text, (x, y), scale, colour, thick)
        char_w = cv2.getTextSize("a", FONT, scale, thick)[0][0]
        last_end = -1
        for col, name in line.labels:
            head = line.text[:col]
            cx = x + cv2.getTextSize(head, FONT, scale, thick)[0][0] if head else x
            cx += max(0, col - len(line.text)) * char_w
            cx = max(cx, last_end + 8)  # never let two labels overlap
            on = name == playing
            ccol = BGR["accent"] if on else BGR["next"] if not dim else BGR["grey"]
            cscale = scale * 0.8
            _text(img, name, (cx, y - int(34 * scale)), cscale, ccol, 2 if on else 1)
            last_end = cx + cv2.getTextSize(name, FONT, cscale, 2)[0][0]

    def _calibration(self, img: np.ndarray, c: CalibrationView) -> None:
        w = img.shape[1]
        cv2.rectangle(img, (0, 90), (w, 190), BGR["shadow"], -1)
        _text(img, c.instruction, (24, 130), 0.8)
        x1 = 24 + int((w - 48) * c.progress)
        cv2.rectangle(img, (24, 150), (w - 24, 162), BGR["grey"], 1)
        cv2.rectangle(img, (24, 150), (x1, 162), BGR["accent"], -1)
        note = c.error or c.warning
        if note:
            _text(img, note, (24, 184), 0.6, BGR["lost"] if c.error else BGR["accent"])


def _text(
    img: np.ndarray,
    s: str,
    org: tuple[int, int],
    scale: float = 0.6,
    colour: tuple[int, int, int] = BGR["text"],
    thickness: int = 1,
) -> None:
    x, y = int(org[0]), int(org[1])
    # Outline at the same stroke width: a thicker shadow would also be wider (Hershey
    # advances grow with thickness) and trail ghost letters past the end of the text.
    for dx, dy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
        cv2.putText(img, s, (x + dx, y + dy), FONT, scale, BGR["shadow"], thickness, cv2.LINE_AA)
    cv2.putText(img, s, (x, y), FONT, scale, colour, thickness, cv2.LINE_AA)

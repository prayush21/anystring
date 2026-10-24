"""Pick-grip classifier, mic onset window and the combined StrumGate."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from racketguitar.capture.mic import detect_onsets
from racketguitar.config import Calibration, Config, GateConfig
from racketguitar.play.gate import (
    GRIP_SCORE_VERSION,
    Gate,
    GripClassifier,
    OnsetWindow,
    calibrate_grip,
    grip_score,
    grip_thresholds,
    make_gate,
)
from racketguitar.types import Onset, StrumGate
from tests.fakes import (
    brush_strokes,
    hand_obs,
    hand_template,
    mix,
    place_hand,
    room_noise,
    speaker_bleed,
)

EXPECTED = {"open": "open", "pick": "closed", "fist": "closed"}


def hand(shape, t=0.0, *, palm_px=80.0, angle=0.0, jitter=0.0, rng=None, at=(640, 360)):
    lm = place_hand(shape, at, palm_px=palm_px, angle_deg=angle)
    return hand_obs(t, lm, jitter_px=jitter, rng=rng)


def blend(a: float) -> np.ndarray:
    """Template landmarks a fraction ``a`` of the way from the pick grip to the open hand."""
    p, o = hand_template("pick"), hand_template("open")
    return p + a * (o - p)


def blend_with_score(target: float) -> np.ndarray:
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if grip_score(blend(mid)) < target else (lo, mid)
    return blend(0.5 * (lo + hi))


def settle(clf: GripClassifier, h, n: int = 5) -> str:
    for _ in range(n):
        clf.update(h)
    return clf.state


# --- grip score and classifier ---------------------------------------------------------


def test_template_scores_are_ordered():
    pick, fist, open_ = (grip_score(hand_template(s)) for s in ("pick", "fist", "open"))
    assert pick < GateConfig.grip_closed_default < GateConfig.grip_open_default <= open_ + 0.01
    assert fist < GateConfig.grip_closed_default
    lo, hi = grip_thresholds()
    assert pick < fist < lo < hi < open_


@pytest.mark.parametrize("shape", ["open", "pick", "fist"])
@pytest.mark.parametrize("palm_px", [30.0, 80.0, 220.0])
@pytest.mark.parametrize("angle", [0.0, 37.0, 90.0, 160.0, -120.0])
def test_score_ignores_scale_and_rotation(shape, palm_px, angle):
    ref = grip_score(hand_template(shape))
    assert grip_score(hand(shape, palm_px=palm_px, angle=angle)) == pytest.approx(ref, abs=1e-9)


@pytest.mark.parametrize("shape", ["open", "pick", "fist"])
@pytest.mark.parametrize("palm_px", [40.0, 80.0, 200.0])
def test_classifies_shapes_with_rotation_and_jitter(shape, palm_px):
    rng = np.random.default_rng(int(palm_px))
    for angle in np.arange(0.0, 360.0, 15.0):
        clf = GripClassifier(hold_frames=2)
        states = [
            clf.update(hand(shape, i / 60, palm_px=palm_px, angle=angle, jitter=1.5, rng=rng))
            for i in range(8)
        ]
        # Unknown until the hold is met, then the right state on every frame.
        assert states[0] == "unknown"
        assert set(states[1:]) == {EXPECTED[shape]}, (angle, states)


def test_score_reads_raw_landmarks():
    h = hand("open")
    mixed = type(h)(t=0.0, landmarks=place_hand("pick", (640, 360)), raw=h.raw)
    assert grip_score(mixed) == pytest.approx(grip_score(h.raw))


@pytest.mark.parametrize(("good", "bad"), [("pick", "open"), ("open", "pick"), ("fist", "open")])
def test_single_frame_outliers_never_flip(good, bad):
    clf = GripClassifier(hold_frames=2)
    settle(clf, hand(good))
    for i in range(30):
        state = clf.update(hand(bad if i % 2 == 0 else good))  # every other frame wrong
        assert state == EXPECTED[good]


def test_hold_frames_before_flip():
    clf = GripClassifier(hold_frames=2)
    settle(clf, hand("pick"))
    assert clf.update(hand("open")) == "closed"
    assert clf.update(hand("open")) == "open"  # flips on the 2nd consecutive frame
    clf3 = GripClassifier(hold_frames=3)
    settle(clf3, hand("pick"))
    assert [clf3.update(hand("open")) for _ in range(3)] == ["closed", "closed", "open"]
    clf1 = GripClassifier(hold_frames=1)
    settle(clf1, hand("pick"))
    assert clf1.update(hand("open")) == "open"


def test_hysteresis_dead_band_keeps_current_state():
    lo, hi = grip_thresholds()
    middling = blend_with_score(0.5 * (lo + hi))
    assert lo < grip_score(middling) < hi
    closed = GripClassifier()
    settle(closed, hand("pick"))
    assert settle(closed, middling, n=10) == "closed"
    opened = GripClassifier()
    settle(opened, hand("open"))
    assert settle(opened, middling, n=10) == "open"
    # Just past either threshold does flip.
    assert settle(closed, blend_with_score(hi + 0.01)) == "open"
    assert settle(opened, blend_with_score(lo - 0.01)) == "closed"


def test_lost_hand_goes_unknown_after_hold():
    clf = GripClassifier(hold_frames=2)
    settle(clf, hand("pick"))
    assert clf.update(None) == "closed"  # one missing frame is tolerated
    assert clf.update(hand("pick")) == "closed"
    assert clf.update(None) == "closed"
    assert clf.update(None) == "unknown"
    assert clf.score is None
    degenerate = np.zeros((21, 2))  # palm of zero size: treated as no hand
    assert np.isnan(grip_score(degenerate))
    settle(clf, hand("pick"))
    assert settle(clf, degenerate) == "unknown"


def test_calibration_shifts_thresholds():
    assert grip_thresholds(Calibration()) == grip_thresholds()
    lo, hi = grip_thresholds(
        Calibration(grip_closed_score=0.2, grip_open_score=0.4, grip_version=GRIP_SCORE_VERSION)
    )
    h = GateConfig().grip_hysteresis
    assert lo == pytest.approx(0.2 + (0.5 - h) * 0.2) and hi == pytest.approx(0.2 + (0.5 + h) * 0.2)
    # A closed score not below the open score is ignored.
    bad = Calibration(grip_closed_score=0.5, grip_open_score=0.3, grip_version=GRIP_SCORE_VERSION)
    assert grip_thresholds(bad) == grip_thresholds()


def test_calibrated_user_with_a_half_open_hand():
    """A user whose relaxed 'open' hand is only half open: misread as closed with the
    defaults, read correctly once calibrated."""
    rng = np.random.default_rng(3)
    users_open = blend_with_score(0.42)
    closed_frames = [hand("pick", jitter=1.0, rng=rng) for _ in range(30)]
    open_frames = [hand_obs(0.0, place_hand(users_open, (640, 360)), jitter_px=1.0, rng=rng)]
    open_frames *= 30
    open_frames[3] = hand("pick")  # a frame where the thumb was misread: median ignores it
    cal = calibrate_grip(closed_frames, open_frames, base=Calibration(detect_to_audio_ms=25.0))
    assert cal.detect_to_audio_ms == 25.0
    assert cal.grip_closed_score == pytest.approx(grip_score(hand_template("pick")), abs=0.03)
    assert cal.grip_open_score == pytest.approx(0.42, abs=0.03)
    relaxed = place_hand(users_open, (640, 360))
    assert settle(GripClassifier(), relaxed) == "closed"
    assert settle(GripClassifier(calibration=cal), relaxed) == "open"
    assert settle(GripClassifier(calibration=cal), hand("pick")) == "closed"


def test_calibrate_grip_rejects_bad_recordings():
    with pytest.raises(ValueError):
        calibrate_grip([], [hand("open")])
    with pytest.raises(ValueError):
        calibrate_grip([hand("open")], [hand("pick")])
    cal = calibrate_grip([hand_template("pick")], [hand_template("open")])  # arrays work too
    assert cal.grip_closed_score < cal.grip_open_score


# --- mic onset window ------------------------------------------------------------------


def window(**kw) -> OnsetWindow:
    args = dict(lookback_ms=80.0, hold_ms=150.0, bleed_ignore_ms=25.0, latency_ms=30.0)
    return OnsetWindow(**(args | kw))


def test_lookback_boundaries_are_inclusive():
    for t, ok in [(1.0, True), (1.04, True), (1.08, True), (1.0801, False), (0.9999, False)]:
        w = window()
        w.add_onset(1.0)
        assert w.allows(t) is ok, t


def test_onset_holds_gate_open_during_a_sweep():
    w = window()
    w.add_onset(1.0)
    assert w.allows(1.03)  # first string: onset in the lookback; anchors the sweep
    assert w.allows(1.10)  # past the 80 ms lookback, inside the 150 ms hold
    assert w.allows(1.15)  # hold boundary
    assert not w.allows(1.1501)
    assert w.anchor == 1.0


def test_stray_onset_opens_only_its_lookback():
    w = window()
    w.add_onset(1.0)
    assert not w.allows(1.10)  # no crossing passed on this onset, so no hold
    assert not w.allows(1.12)


def test_hold_follows_the_newest_onset():
    w = window()
    w.add_onset(1.0)
    assert w.allows(1.05)
    w.add_onset(1.10)
    assert w.allows(1.12)
    assert w.anchor == 1.10
    assert w.allows(1.25)
    assert not w.allows(1.2501)


def test_bleed_onsets_after_emitted_notes_are_ignored():
    end = 2.0 + (GateConfig.output_latency_ms + 25.0) / 1000  # 2.055
    for t_onset, usable in [
        (1.999, True),
        (2.0, False),
        (2.03, False),
        (end, False),
        (end + 1e-4, True),
    ]:
        w = window(latency_ms=GateConfig.output_latency_ms)
        w.add_note(2.0)
        w.add_onset(t_onset)
        assert w.allows(t_onset + 0.01) is usable, t_onset


def test_bleed_filter_is_order_independent():
    w = window()
    w.add_onset(2.04)  # the bleed was drained before the engine reported its note
    w.add_note(2.0)
    assert not w.allows(2.05)


def test_onset_after_the_crossing_is_kept_for_later():
    w = window()
    w.add_onset(1.01)
    assert not w.allows(1.0)  # an onset in the crossing's future does not count ...
    assert w.allows(1.02)  # ... but it is still there for the next one


def test_old_entries_are_pruned():
    w = window()
    for i in range(2000):
        w.add_onset(i * 0.01)
        w.add_note(i * 0.01 + 0.005)
        w.allows(i * 0.01)
    assert len(w._onsets) <= 101 and len(w._notes) <= 101


def test_measured_latency_moves_the_bleed_window():
    def usable(cal):
        g = Gate(GateConfig(pick_grip=False, mic=True), cal)
        g.note_emitted(2.0)
        g.observe_onset(Onset(2.07, 10.0))
        return g.allows(2.08)

    assert usable(Calibration())  # 2.07 is past 2.0 + 30 + 25 ms
    assert not usable(Calibration(detect_to_audio_ms=60.0))  # ignored until 2.085


# --- the combined gate -----------------------------------------------------------------


def test_make_gate_defaults_and_protocol():
    g = make_gate(Config())
    assert isinstance(g, StrumGate)
    assert g.pick_grip and not g.mic
    assert g.grip == "unknown" and not g.allows(0.0)  # nothing seen yet: disarmed
    assert isinstance(make_gate(GateConfig(mic=True)), StrumGate)
    assert make_gate().cfg == GateConfig()


def drive(gate: Gate, shape: str | None, t: float, frames: int = 3) -> None:
    for i in range(frames):
        ti = t - (frames - 1 - i) / 60
        gate.observe_hand(ti, None if shape is None else hand(shape, ti))


@pytest.mark.parametrize(
    ("pick", "mic", "closed", "onset"), list(itertools.product([False, True], repeat=4))
)
def test_all_four_config_combinations(pick, mic, closed, onset):
    g = make_gate(GateConfig(pick_grip=pick, mic=mic))
    drive(g, "pick" if closed else "open", 1.0)
    assert g.grip == ("closed" if closed else "open")  # classified even when not gating
    if onset:
        g.observe_onset(Onset(0.96, 12.0))
    expected = (not pick or closed) and (not mic or onset)
    assert g.allows(1.0) is expected


def test_mic_calls_are_noops_when_mic_is_off():
    g = make_gate(GateConfig(pick_grip=False, mic=False))
    g.observe_onset(Onset(1.0, 5.0))
    g.note_emitted(1.0)
    assert g.allows(1.0) and g.allows(5.0)
    assert len(g.window._onsets) == 0 and len(g.window._notes) == 0


def test_open_hand_disarms_mid_play_and_lost_hand_disarms():
    g = make_gate(Config())
    drive(g, "pick", 1.0)
    assert g.allows(1.0)
    g.observe_hand(1.017, hand("open"))
    assert g.allows(1.017)  # one open frame: still armed
    g.observe_hand(1.033, hand("open"))
    assert not g.allows(1.033)
    drive(g, "pick", 1.1)
    drive(g, None, 1.2, frames=2)
    assert g.grip == "unknown" and not g.allows(1.2)


def test_hold_is_not_started_by_a_crossing_the_grip_blocked():
    g = make_gate(GateConfig(pick_grip=True, mic=True))
    drive(g, "open", 1.0)
    g.observe_onset(Onset(0.99, 10.0))
    assert not g.allows(1.0)
    drive(g, "pick", 1.10)
    assert not g.allows(1.10)  # onset at 0.99 is out of lookback and never anchored a sweep


def test_detected_onsets_drive_the_gate_through_bleed():
    """End to end: detector onsets from brushes + speaker bleed, fed frame by frame.

    Crossings 40 ms after a brush (camera lag) play; crossings 40 ms after a note that
    no brush accompanied do not, because its bleed onset falls in the ignore window.
    """
    rng = np.random.default_rng(31)
    brushes_t = np.arange(0.5, 6.0, 0.5)
    dur = 6.5
    b, brushes_t = brush_strokes(brushes_t, dur, peak=0.1, rng=rng)
    # The engine emits a note at t_detect; it leaves the speaker GateConfig.output_latency_ms later.
    sounds = [(t + 0.25, 196.0 * (1 + i % 4)) for i, t in enumerate(brushes_t)]
    bleed, sound_t = speaker_bleed(sounds, dur)  # band energy 1.7x the brushes
    x = mix(b, bleed, room_noise(dur, rms=0.003, rng=np.random.default_rng(32)))
    onsets = detect_onsets(x)
    assert len(onsets) > len(brushes_t)  # some bleed attacks fire in the detector
    notes = sound_t - GateConfig.output_latency_ms / 1000

    g = make_gate(GateConfig(pick_grip=False, mic=True))
    crossings = {round(t + 0.04, 4): True for t in brushes_t}
    crossings |= {round(t + 0.04, 4): False for t in sound_t}
    pending, results = list(onsets), {}
    for f in range(round(dur * 60)):
        tf = f / 60
        while pending and pending[0].t <= tf:
            g.observe_onset(pending.pop(0))
        for tn in notes[(notes > tf - 1 / 60) & (notes <= tf)]:
            g.note_emitted(float(tn))
        for tc in sorted(c for c in crossings if tf - 1 / 60 < c <= tf):
            results[tc] = g.allows(tc)
    assert results == crossings


def test_hit_times_earlier_than_the_latest_hand_frame():
    """The engine queries with interpolated crossing times up to a frame before the
    frame's observe_hand t, several hits per frame in increasing order, and reports
    note_emitted with those same past times."""
    g = make_gate(GateConfig(pick_grip=True, mic=True))
    drive(g, "pick", 1.000)  # latest observe_hand t = 1.000
    g.observe_onset(Onset(0.950, 20.0))  # contact heard before the camera saw it
    g.observe_onset(Onset(1.010, 20.0))  # drained before the frame; later than the hits
    hits = [0.968, 0.975, 0.990]  # up to ~33 ms before the frame time
    for th in hits:
        assert g.allows(th)  # grip state as of the latest frame; onset 0.950 in lookback
        g.note_emitted(th)
    assert g.window.anchor == 0.950  # the 1.010 onset is in these hits' future
    # The 1.010 onset lies in the ignore window of the note at 0.990 (to 1.045): bleed.
    # The next frame's hits still play on the sweep's hold (0.950 + 150 ms) ...
    g.observe_hand(1.033, hand("pick", 1.033))
    assert g.allows(1.020) and g.allows(1.099)
    # ... and nothing beyond it: 1.010 was discarded as bleed, so it cannot extend it.
    g.observe_hand(1.117, hand("pick", 1.117))
    assert not g.allows(1.101)
    # Pick-grip only: an open hand on the latest frame blocks hits timed before it.
    p = make_gate(Config())
    drive(p, "pick", 2.0)
    drive(p, "open", 2.033, frames=2)
    assert p.grip == "open" and not p.allows(2.010)


def test_onset_slightly_after_the_crossing_counts():
    """Onset timing and crossing interpolation each err by a few ms, so a contact
    measured up to ``onset_after_ms`` after its camera crossing still opens the gate."""
    cfg = GateConfig(pick_grip=False, mic=True)
    g = Gate(cfg)
    g.observe_onset(Onset(1.008, 10.0))
    assert g.allows(1.000)
    late = Gate(cfg)
    late.observe_onset(Onset(1.000 + (cfg.onset_after_ms + 2) / 1000, 10.0))
    assert not late.allows(1.000)


def test_calibration_from_another_score_version_is_ignored():
    old = Calibration(grip_closed_score=0.2, grip_open_score=0.4)  # no version: pre-v2
    assert grip_thresholds(old) == grip_thresholds()
    assert calibrate_grip([hand("pick")], [hand("open")]).grip_version == GRIP_SCORE_VERSION


def test_score_does_not_spike_when_the_palm_turns_edge_on():
    """Foreshorten the hand along its wrist -> knuckles axis, as a palm turning edge-on
    to the camera does. The score must stay near the face-on value instead of blowing up
    because the wrist -> middle-knuckle length collapses."""
    lm = place_hand("open", (640, 360), palm_px=90.0)
    face_on = grip_score(lm)
    axis = (lm[9] - lm[0]) / np.linalg.norm(lm[9] - lm[0])
    rel = lm - lm[0]
    along = rel @ axis
    squashed = lm[0] + rel - np.outer(along * 0.8, axis)  # 80 % shorter along the axis
    assert grip_score(squashed) < 2.0 * face_on

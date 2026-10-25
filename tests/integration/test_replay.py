"""Record -> save -> load -> replay, with recorded observations, mic audio and video."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from racketguitar.config import Calibration, Config
from racketguitar.pipeline import FrameInput, Pipeline
from racketguitar.replay.harness import compare_events, render_session, replay, stream_onsets
from racketguitar.replay.record import Recorder, TeeDetector, load_session
from tests.fakes import brush_strokes, mix, render_racket, room_noise
from tests.integration.song import T0, expected_note_count, frames

PROC_LAG = 0.03  # processing starts this long after capture


def record_song(cfg, path, *, mic_audio=None, mic_t0=None, images=None):
    """Play the song through a live-style loop and record it. Returns the live events."""
    song, _ = frames(cfg)
    tee = TeeDetector(cfg.mic) if mic_audio is not None else None
    rec = Recorder(path, cfg, Calibration(), video=images is not None, mic=tee, drop_video=False)
    pipe = Pipeline(cfg)
    fed = 0
    bs, sr = cfg.mic.block_size, cfg.mic.sample_rate
    live = []
    for i, f in enumerate(song):
        proc_t = f.t + PROC_LAG
        onsets = []
        if tee is not None:  # the mic callback has delivered every block ended by proc_t
            while fed + bs <= mic_audio.size and mic_t0 + (fed + bs) / sr <= proc_t:
                onsets += tee.process(mic_audio[fed : fed + bs], mic_t0 + fed / sr)
                fed += bs
        inp = FrameInput(f.t, proc_t, f.pose, f.hands, tuple(onsets))
        events = pipe.step(inp)
        live += events
        rec.add(inp, events, None if images is None else images[i])
    rec.stop()
    return live


def test_session_round_trip_preserves_everything(tmp_path):
    cfg = Config().with_overrides({"gate": {"mic": True}})
    mic_t0 = T0 - 0.5
    audio = room_noise(3.0, rms=0.002)
    record_song(cfg, tmp_path / "s", mic_audio=audio, mic_t0=mic_t0)
    s = load_session(tmp_path / "s")
    song, _ = frames(cfg)
    assert s.config == cfg and s.calibration == Calibration()
    assert len(s.frames) == len(song)
    for rec, f in zip(s.frames, song, strict=True):
        assert rec.inp.t == f.t and rec.inp.proc_t == f.t + PROC_LAG
        assert rec.inp.pose == f.pose
        assert len(rec.inp.hands) == len(f.hands)
        for a, b in zip(rec.inp.hands, f.hands, strict=True):
            assert a.t == b.t
            assert np.array_equal(a.landmarks, b.landmarks) and np.array_equal(a.raw, b.raw)
    fed = s.mic.size - s.mic.size % cfg.mic.block_size
    assert s.mic_t0 == pytest.approx(mic_t0)
    assert np.array_equal(s.mic, audio[: s.mic.size]) and s.mic.size == fed


def test_song_replays_identically(tmp_path, cfg):
    live = record_song(cfg, tmp_path / "s")
    on = [e for e in live if e.kind == "on"]
    assert len(on) == expected_note_count(cfg)
    assert sum(e.kind == "damp" for e in live) == 6  # the mute damps every ringing string
    for _ in range(2):
        r = replay(tmp_path / "s")
        assert r.matches, r.mismatches
        assert r.events == live


def test_replay_with_other_settings_reports_differences(tmp_path, cfg):
    record_song(cfg, tmp_path / "s")
    free = cfg.with_overrides({"gate": {"pick_grip": False}})
    r = replay(tmp_path / "s", config=free)
    # Without the grip gate, the open-hand lift-away across the strings plays too.
    assert not r.matches
    assert sum(e.kind == "on" for e in r.events) > expected_note_count(cfg)


def test_mic_session_replays_from_recording_and_from_mic(tmp_path):
    cfg = Config().with_overrides({"gate": {"mic": True}})
    song, brushes = frames(cfg)
    mic_t0 = T0 - 0.5
    dur = song[-1].t - mic_t0 + 0.5
    # Contact is heard a few ms before the camera timestamps the crossing.
    hits, _ = brush_strokes([b - mic_t0 - 0.004 for b in brushes], dur, peak=0.1)
    audio = mix(hits, room_noise(dur, rms=0.003))
    live = record_song(cfg, tmp_path / "s", mic_audio=audio, mic_t0=mic_t0)
    assert sum(e.kind == "on" for e in live) == expected_note_count(cfg)
    assert replay(tmp_path / "s").matches
    from_mic = replay(tmp_path / "s", onsets_from="mic")
    assert from_mic.matches, from_mic.mismatches


def test_mic_gate_blocks_strums_nobody_heard(tmp_path):
    cfg = Config().with_overrides({"gate": {"mic": True}})
    song, _ = frames(cfg)
    mic_t0 = T0 - 0.5
    audio = room_noise(song[-1].t - mic_t0 + 0.5, rms=0.003)
    live = record_song(cfg, tmp_path / "s", mic_audio=audio, mic_t0=mic_t0)
    assert not any(e.kind == "on" for e in live)
    assert replay(tmp_path / "s", onsets_from="mic").matches


def test_stream_onsets_reports_after_the_block_ends():
    cfg = Config()
    x, truth = brush_strokes([0.2, 0.5], 1.0, peak=0.1)
    from racketguitar.capture.mic import OnsetDetector

    found = stream_onsets(x, 10.0, OnsetDetector(cfg.mic))
    assert len(found) == 2
    for (reported, onset), t in zip(found, truth, strict=True):
        assert abs(onset.t - (10.0 + t)) < 0.005
        assert onset.t < reported <= onset.t + 0.02


def test_debug_video_is_lossless_and_flags_dropped_frames(tmp_path, cfg):
    from racketguitar.replay.harness import video_frames

    song, _ = frames(cfg)
    images = [render_racket(**f.scene).image for f in song[:30]]
    for i in range(5, len(images), 7):  # frames the writer "dropped"
        images[i] = None
    record_song(cfg, tmp_path / "s", images=images + [None] * (len(song) - len(images)))
    s = load_session(tmp_path / "s")
    assert s.video_path is not None and s.video_path.exists()
    kept = [img for img in images if img is not None]
    assert [f.in_video for f in s.frames[: len(images)]] == [img is not None for img in images]
    video = list(video_frames(s.video_path))
    assert len(video) == len(kept)
    assert all(np.array_equal(a, b) for a, b in zip(video, kept, strict=True))
    assert replay(s).matches


def test_compare_events_and_render(tmp_path, cfg):
    live = record_song(cfg, tmp_path / "s")
    assert compare_events(live, live) == []
    changed = [replace(live[0], velocity=max(1, live[0].velocity - 1)), *live[1:]]
    assert len(compare_events(live, changed)) == 1
    assert "replayed 0" in compare_events(live, [])[-1]
    audio = render_session(live, cfg)
    assert audio.dtype == np.float32 and np.abs(audio).max() > 0.05
    assert np.all(np.isfinite(audio))

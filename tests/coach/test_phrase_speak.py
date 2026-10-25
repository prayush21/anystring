"""Phrasing (Claude) and speech (ElevenLabs) with fake clients: no network."""

from __future__ import annotations

import io
import json
from types import SimpleNamespace

import anthropic
import numpy as np
import pytest

from anystring.coach.analyze import analyze
from anystring.coach.phrase import CoachError, phrase, template
from anystring.coach.speak import SpeakError, synthesize
from anystring.config import CoachConfig
from tests.coach.takes import T0, beat_strokes, frames, held_as_due, song, stroke, take


def _report(delays=None):
    info = song(["C", "G", "Am", "F", "C", "G", "Am", "F"])
    held = held_as_due(info, delays)
    log = take(beat_strokes(info, 0.0, held), frames(T0 - 1, info.chords[-1].end, held), info)
    return analyze(log, CoachConfig())


class FakeClient:
    def __init__(self, text="Nice timing.", stop_reason="end_turn", fail=False):
        self.calls: list[dict] = []
        self._reply = SimpleNamespace(
            stop_reason=stop_reason, content=[SimpleNamespace(type="text", text=text)]
        )
        self._fail = fail
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self._fail:
            raise anthropic.AnthropicError("connection reset")
        return self._reply


def test_phrase_sends_the_measured_facts_and_cleans_the_reply():
    client = FakeClient("**Great** timing! The change from G to Am was late; move early.")
    cfg = CoachConfig(max_words=40)
    text = phrase(_report({"Am": 0.3}), cfg, client=client)
    assert text == "Great timing! The change from G to A minor was late; move early."
    (call,) = client.calls
    assert call["model"] == "claude-opus-5-5"
    assert call["fallbacks"] == "default" and call["betas"] == ["server-side-fallback-2026-07-01"]
    assert call["output_config"] == {"effort": "low"}
    assert "At most 40 words" in call["system"]
    sent = json.loads(call["messages"][0]["content"])
    assert sent["chord_changes_worst_first"][0]["from"] == "G"
    assert "G to Am" in sent["issues_ranked"][0]["fact"]


@pytest.mark.parametrize(
    "client", [FakeClient(stop_reason="refusal"), FakeClient(fail=True), FakeClient(text=" ")]
)
def test_phrase_failures_raise_for_the_template_fallback(client):
    with pytest.raises(CoachError):
        phrase(_report(), CoachConfig(), client=client)


def test_template_names_strength_then_top_findings():
    text = template(_report({"Am": 0.3}))
    assert text.startswith("Nice:")
    assert "The change from G to A minor was late 2 of 2 times" in text
    assert "Try to start moving to the A minor section" in text


def test_template_asks_for_more_when_there_is_too_little():
    r = analyze(take(stroke(1.0), frames(0.5, 2.0)), CoachConfig())
    assert "play a little longer" in template(r).lower()


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_synthesize_requests_pcm_and_decodes_it():
    pcm = (np.array([0, 16384, -16384, 32767], dtype="<i2")).tobytes()
    seen = {}

    def opener(req, timeout):
        seen.update(url=req.full_url, headers=dict(req.header_items()), body=req.data, t=timeout)
        return FakeResponse(pcm + b"\x00")  # an odd trailing byte is dropped

    cfg = CoachConfig()
    out = synthesize("Nice work.", cfg, api_key="k", opener=opener)
    assert seen["url"] == (
        f"https://api.elevenlabs.io/v1/text-to-speech/{cfg.tts_voice}?output_format=pcm_24000"
    )
    assert seen["headers"]["Xi-api-key"] == "k"
    assert json.loads(seen["body"]) == {"text": "Nice work.", "model_id": cfg.tts_model}
    assert seen["t"] == cfg.tts_timeout_s
    np.testing.assert_allclose(out, [0.0, 0.5, -0.5, 32767 / 32768])
    assert out.dtype == np.float32


def test_synthesize_needs_a_key(monkeypatch):
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    with pytest.raises(SpeakError, match="ELEVENLABS_API_KEY"):
        synthesize("hi", CoachConfig(), opener=lambda r, timeout: FakeResponse(b""))


def test_synthesize_network_error_is_a_speak_error():
    def opener(req, timeout):
        raise OSError("offline")

    with pytest.raises(SpeakError, match="offline"):
        synthesize("hi", CoachConfig(), api_key="k", opener=opener)

from __future__ import annotations

import io
import json
import urllib.error
import wave

import numpy as np
import pytest

from anystring.config import SongConfig
from anystring.song.library import find_song, match_score, normalize, sheet_paths, songs_dir
from anystring.song.sheet import parse_sheet
from anystring.song.voice import (
    API_KEY_ENV,
    STT_URL,
    VoiceError,
    parse_command,
    transcribe,
    wav_bytes,
)


@pytest.fixture
def library(tmp_path):
    d = tmp_path / "songs"
    d.mkdir()
    (d / "paper-boats.txt").write_text(
        "{title: Paper Boats}\n{artist: The Place-Holders}\nG D\nla la\n"
    )
    (d / "tin-roof.cho").write_text("{title: Tin Roof}\n{artist: Rainy}\n[G]la [C]la\n")
    (d / "notes.md").write_text("not a sheet")
    (d / "broken.txt").write_text("no chords at all")
    return d


def test_normalize():
    assert normalize("Take On Me — a-ha!") == "take on me aha"
    assert normalize("Don't  Stop") == "dont stop"


def test_spoken_titles_match_with_slips(library):
    assert [p.name for p in sheet_paths(library)] == [
        "broken.txt",
        "paper-boats.txt",
        "tin-roof.cho",
    ]
    m = find_song("boats of paper by the placeholders", library)
    assert m is not None and m.song.title == "Paper Boats"
    assert find_song("tin roof", library).song.title == "Tin Roof"
    assert find_song("bohemian rhapsody", library) is None
    assert find_song(str(library / "tin-roof.cho")).score == 1.0  # a path loads directly


def test_word_order_slip_scores_high():
    song = parse_sheet("{title: Take On Me}\n{artist: a-ha}\nAm D\n")
    assert match_score("take me on by aha", song) > 0.8
    assert match_score("wonderwall", song) < 0.5


def test_songs_dir_is_under_home(tmp_path):
    assert songs_dir() == tmp_path / "home" / "songs"  # conftest's ANYSTRING_HOME


@pytest.mark.parametrize(
    ("text", "kind", "query"),
    [
        ("Let's play Take Me On by Aha.", "play", "take me on by aha"),
        ("play the song paper boats please", "play", "paper boats"),
        ("Can we play tin roof?", "play", "tin roof"),
        ("Start.", "start", ""),
        ("let's go", "start", ""),
        ("play it", "start", ""),
        ("Stop!", "stop", ""),
        ("From the top", "restart", ""),
        ("paper boats", "unknown", "paper boats"),
        ("How was that?", "feedback", ""),
        ("Give me feedback please", "feedback", ""),
        ("how did I do", "feedback", ""),
        ("What should I improve?", "feedback", ""),
        ("play feedback", "play", "feedback"),  # a song title still wins after "play"
    ],
)
def test_parse_command(text, kind, query):
    cmd = parse_command(text)
    assert (cmd.kind, cmd.query) == (kind, query) and cmd.text == text


def test_wav_bytes_is_a_valid_clip():
    data = wav_bytes(np.sin(np.linspace(0, 100, 1600)).astype(np.float32))
    with wave.open(io.BytesIO(data)) as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()) == (
            1,
            2,
            16_000,
            1600,
        )


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_transcribe_sends_a_multipart_request(monkeypatch):
    monkeypatch.setenv(API_KEY_ENV, "test-key")
    seen = {}

    def opener(req, timeout):
        seen.update(url=req.full_url, headers=dict(req.headers), body=req.data, timeout=timeout)
        return _Resp(json.dumps({"text": " let's play paper boats ", "words": []}).encode())

    cfg = SongConfig(stt_model="scribe_v2", stt_timeout_s=7.0)
    text = transcribe(b"RIFFfakewav", cfg, opener=opener)
    assert text == "let's play paper boats"
    assert seen["url"] == STT_URL and seen["timeout"] == 7.0
    assert seen["headers"]["Xi-api-key"] == "test-key"
    assert seen["headers"]["Content-type"].startswith("multipart/form-data; boundary=")
    body = seen["body"]
    assert b'name="model_id"\r\n\r\nscribe_v2' in body
    assert b'name="file"; filename="clip.wav"' in body and b"RIFFfakewav" in body


def test_transcribe_errors(monkeypatch):
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    with pytest.raises(VoiceError, match=API_KEY_ENV):
        transcribe(b"x", SongConfig(), opener=lambda r, timeout: None)

    def refuse(req, timeout):
        raise urllib.error.URLError("offline")

    with pytest.raises(VoiceError, match="offline"):
        transcribe(b"x", SongConfig(), api_key="k", opener=refuse)
    with pytest.raises(VoiceError, match="no text"):
        transcribe(b"x", SongConfig(), api_key="k", opener=lambda r, timeout: _Resp(b"{}"))

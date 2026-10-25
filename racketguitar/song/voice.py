"""Voice requests: record a short clip, turn it into text, and read the command.

Push-to-talk records ``SongConfig.listen_s`` seconds from the mic, then ElevenLabs
speech-to-text (``POST /v1/speech-to-text``, key in ``$ELEVENLABS_API_KEY``) returns
the words. None of this runs on the play path: the app does it on a worker thread and
only the resulting command touches the session.
"""

from __future__ import annotations

import io
import json
import os
import re
import ssl
import urllib.request
import uuid
import wave
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np

from racketguitar.config import SongConfig

__all__ = [
    "STT_URL",
    "API_KEY_ENV",
    "VoiceError",
    "Command",
    "parse_command",
    "record_clip",
    "wav_bytes",
    "transcribe",
]

STT_URL = "https://api.elevenlabs.io/v1/speech-to-text"
API_KEY_ENV = "ELEVENLABS_API_KEY"
CLIP_RATE = 16_000  # Hz; plenty for speech and a small upload

_PLAY = re.compile(r"^(?:(?:lets|let us|can we|can you|please)\s+)*(?:play|do|try)\s+(.+)$")
_START = {"start", "go", "begin", "ready", "lets go", "lets start", "start the song", "im ready"}
_STOP = {"stop", "pause", "end", "stop the song", "enough"}
_RESTART = {"restart", "again", "from the top", "start again", "start over", "play it again"}


class VoiceError(RuntimeError):
    pass


@dataclass(frozen=True)
class Command:
    kind: Literal["play", "start", "stop", "restart", "unknown"]
    text: str  # what was heard
    query: str = ""  # song title for "play"


def parse_command(text: str) -> Command:
    words = " ".join(re.sub(r"[^a-z0-9 ]+", " ", text.lower().replace("'", "")).split())
    words = re.sub(r"\s+(please|now)$", "", words)
    if words in _RESTART:
        return Command("restart", text)
    if words in _START:
        return Command("start", text)
    if words in _STOP:
        return Command("stop", text)
    if m := _PLAY.match(words):
        query = re.sub(r"^(the song|a song|song)\s+", "", m.group(1)).strip()
        if query in ("it", "", "the song"):
            return Command("start", text)
        return Command("play", text, query)
    return Command("unknown", text, words)


def record_clip(seconds: float, device: int | None = None) -> np.ndarray:
    """Block for ``seconds`` while recording mono float32 at ``CLIP_RATE``."""
    import sounddevice as sd

    clip = sd.rec(
        int(seconds * CLIP_RATE), samplerate=CLIP_RATE, channels=1, dtype="float32", device=device
    )
    sd.wait()
    return clip[:, 0]


def wav_bytes(samples: np.ndarray, rate: int = CLIP_RATE) -> bytes:
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def transcribe(
    wav: bytes,
    cfg: SongConfig,
    *,
    api_key: str | None = None,
    opener: Callable[..., object] | None = None,
) -> str:
    """Speech-to-text for one WAV clip. ``opener`` replaces ``urllib`` in tests."""
    key = api_key or os.environ.get(API_KEY_ENV)
    if not key:
        raise VoiceError(f"voice needs ${API_KEY_ENV}; you can also use the keys")
    boundary = uuid.uuid4().hex
    body = _multipart(boundary, {"model_id": cfg.stt_model}, ("file", "clip.wav", wav))
    req = urllib.request.Request(
        STT_URL,
        data=body,
        method="POST",
        headers={"xi-api-key": key, "Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    if opener is None:
        import certifi

        ctx = ssl.create_default_context(cafile=certifi.where())

        def opener(r, timeout):
            return urllib.request.urlopen(r, timeout=timeout, context=ctx)

    try:
        with opener(req, timeout=cfg.stt_timeout_s) as resp:
            data = json.loads(resp.read())
    except (OSError, ValueError) as exc:  # URLError and HTTPError are OSErrors
        raise VoiceError(f"speech-to-text failed: {exc}") from exc
    text = data.get("text") if isinstance(data, dict) else None
    if not isinstance(text, str):
        raise VoiceError("speech-to-text returned no text")
    return text.strip()


def _multipart(boundary: str, fields: dict[str, str], file: tuple[str, str, bytes]) -> bytes:
    out = io.BytesIO()
    for name, value in fields.items():
        out.write(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
        out.write(f"{value}\r\n".encode())
    field, filename, payload = file
    out.write(
        f'--{boundary}\r\nContent-Disposition: form-data; name="{field}"; '
        f'filename="{filename}"\r\nContent-Type: audio/wav\r\n\r\n'.encode()
    )
    out.write(payload)
    out.write(f"\r\n--{boundary}--\r\n".encode())
    return out.getvalue()

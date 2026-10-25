"""Say the feedback aloud: ElevenLabs text-to-speech, played on the default output.

``POST /v1/text-to-speech/{voice}`` with ``output_format=pcm_<rate>`` returns raw 16-bit
mono PCM, so no decoder is needed. The key is ``$ELEVENLABS_API_KEY``, the same one
song mode's voice requests use. The app runs this on a worker thread once strumming
has stopped, so the speech never overlaps playing.
"""

from __future__ import annotations

import json
import os
import ssl
import urllib.parse
import urllib.request
from collections.abc import Callable

import numpy as np

from anystring.config import CoachConfig

__all__ = ["TTS_URL", "API_KEY_ENV", "SpeakError", "synthesize", "play"]

TTS_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice}?output_format=pcm_{rate}"
API_KEY_ENV = "ELEVENLABS_API_KEY"


class SpeakError(RuntimeError):
    pass


def synthesize(
    text: str,
    cfg: CoachConfig,
    *,
    api_key: str | None = None,
    opener: Callable[..., object] | None = None,
) -> np.ndarray:
    """Mono float32 speech at ``cfg.tts_rate``. ``opener`` replaces ``urllib`` in tests."""
    key = api_key or os.environ.get(API_KEY_ENV)
    if not key:
        raise SpeakError(f"speaking feedback needs ${API_KEY_ENV}")
    url = TTS_URL.format(voice=urllib.parse.quote(cfg.tts_voice), rate=cfg.tts_rate)
    req = urllib.request.Request(
        url,
        data=json.dumps({"text": text, "model_id": cfg.tts_model}).encode(),
        method="POST",
        headers={"xi-api-key": key, "Content-Type": "application/json"},
    )
    if opener is None:
        import certifi

        ctx = ssl.create_default_context(cafile=certifi.where())

        def opener(r, timeout):
            return urllib.request.urlopen(r, timeout=timeout, context=ctx)

    try:
        with opener(req, timeout=cfg.tts_timeout_s) as resp:
            pcm = resp.read()
    except OSError as exc:  # URLError and HTTPError are OSErrors
        raise SpeakError(f"text-to-speech failed: {exc}") from exc
    if len(pcm) < 2:
        raise SpeakError("text-to-speech returned no audio")
    pcm = pcm[: len(pcm) // 2 * 2]
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0


def play(samples: np.ndarray, rate: int, device: int | None = None) -> None:
    """Block while ``samples`` play. Uses its own stream, beside the guitar's."""
    import sounddevice as sd

    sd.play(samples, samplerate=rate, device=device)
    sd.wait()

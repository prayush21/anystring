"""Put a take report into a few spoken sentences: Claude, or a fixed template.

``phrase`` sends the measured facts (``analyze.facts``) to Claude and gets back short,
plain sentences meant to be heard, not read. The model only words what was measured;
it is told not to invent anything. ``template`` says the same top findings without a
network call; the app falls back to it when the LLM is off, has no credentials or fails.
"""

from __future__ import annotations

import json
import re

from anystring.coach.analyze import TakeReport, facts
from anystring.config import CoachConfig

__all__ = ["CoachError", "SYSTEM", "phrase", "template"]

SYSTEM = """\
You are a warm, direct guitar coach. The player strums an everyday object (a racket, a \
pan, a ruler) held like a guitar; a camera tracks their hands. Strumming only sounds \
while the thumb pad rests on the side of the index finger (the "pick grip"), and chords \
come from which of four sections of the handle the fretting hand holds.

You get measurements of one take as JSON. Turn them into feedback that will be spoken \
aloud to the player right after they stop playing:
- Start with one specific thing that went well, if the data shows one.
- Then the one or two most important problems from issues_ranked, each with what to do \
about it. Name chords and numbers the way a person would say them ("G to A minor", \
"about a tenth of a second late").
- Use only what the measurements say. Do not invent problems or praise.
- If enough_to_judge is false, just ask them to play a little longer and ask again.
- Plain sentences only: no lists, markdown, emoji or stage directions. At most \
{max_words} words.\
"""

_CHORD = re.compile(r"\b([A-G])([#b]?)(m(?!aj))?(7|maj7|sus2|sus4)?\b")


class CoachError(RuntimeError):
    pass


def phrase(report: TakeReport, cfg: CoachConfig, *, client=None) -> str:
    """Claude's wording of ``report``. ``client`` replaces the Anthropic client in tests."""
    import anthropic

    try:
        if client is None:
            client = anthropic.Anthropic(timeout=cfg.llm_timeout_s, max_retries=1)
        response = client.beta.messages.create(
            model=cfg.llm_model,
            max_tokens=cfg.llm_max_tokens,
            system=SYSTEM.format(max_words=cfg.max_words),
            output_config={"effort": cfg.llm_effort},
            # On a policy decline, the API retries on a fallback model in the same call.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            messages=[{"role": "user", "content": json.dumps(facts(report))}],
        )
    except anthropic.AnthropicError as exc:
        raise CoachError(f"feedback LLM failed: {exc}") from exc
    if response.stop_reason == "refusal":
        raise CoachError("feedback LLM declined")
    text = " ".join(b.text for b in response.content if b.type == "text").strip()
    if not text:
        raise CoachError("feedback LLM returned no text")
    return _speakable(text)


def template(report: TakeReport) -> str:
    """The same feedback without an LLM: one strength, then the top two findings."""
    if not report.enough:
        return "I didn't catch enough strumming to judge yet. Play a little longer, then ask again."
    parts = []
    if report.strengths:
        parts.append(f"Nice: {report.strengths[0]}.")
    for f in report.findings[:2]:
        parts.append(f"{_sentence(f.fact)} Try to {f.tip}.")
    if not report.findings:
        parts.append("That was clean. Try it a little faster next time.")
    return _speakable(" ".join(parts))


def _sentence(text: str) -> str:
    text = text[0].upper() + text[1:]
    return text if text.endswith(".") else text + "."


def _speakable(text: str) -> str:
    """Drop markdown the model might still add, and say chord names in words."""
    text = re.sub(r"[*_#`>]+", "", text)
    text = re.sub(r"^\s*[-•]\s*", "", text, flags=re.M)
    return " ".join(_CHORD.sub(_say_chord, text).split())


def _say_chord(m: re.Match[str]) -> str:
    root, acc, minor, ext = m.groups()
    if not (acc or minor or ext):
        return m.group(0)  # a bare letter: "A" could be the word
    word = root + {"#": " sharp", "b": " flat", "": ""}[acc]
    if minor:
        word += " minor"
    if ext:
        word += (
            " " + {"7": "seven", "maj7": "major seven", "sus2": "sus two", "sus4": "sus four"}[ext]
        )
    return word

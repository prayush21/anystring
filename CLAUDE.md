# Anystring

A racket (tennis or badminton, or a pan, a ruler, anything held like a guitar) played as
a guitar using only a MacBook camera and mic, with nothing attached to it. The code and
the types still say "racket", the best-tested case. The full spec is [docs/PID.md](docs/PID.md); read
it before changing behaviour, but note its printed-marker parts are superseded (below). This file is the
short version plus the rules for working in the repo.

## Commands

```bash
uv sync                      # install (Python 3.12, deps pinned in uv.lock)
uv run pytest                # headless suite; hardware tests are deselected
uv run pytest -m hardware    # camera/mic/speaker tests, MacBook only
uv run ruff check . && uv run ruff format .
uv run anystring fetch-model   # once: MediaPipe hand model into models/
uv run anystring          # play (calibrates on first run); `calibrate`, `replay DIR`
uv run python -m tools.latency_probe analyze take.wav
uv run python -m tools.grip_sweep --closed DIR --relaxed DIR   # rank grip thresholds
```

## Architecture

Four threads pass small timestamped messages through queues, so no slow step can
delay a note:

1. **Camera** (`capture/camera.py`): 60 fps capture (30 fallback), buffer size 1, newest
   frame only, stamped with `time.monotonic()` at capture.
2. **Vision** (`vision/`, `geometry/`, `play/`): MediaPipe Hand Landmarker in VIDEO
   mode on a 640x360 copy -> `HandObs` (One Euro filtered); the markerless racket
   tracker (`vision/flow.py`) -> `RacketPose`; then strum, gate, chord and mute logic
   emit `NoteEvent`s.
3. **Mic** (`capture/mic.py`): 48 kHz, 64-sample blocks; high-band spectral-flux
   onsets -> `Onset`. Optional, off by default.
4. **Audio** (`audio/`): 48 kHz, 128-sample output callback rendering one
   Karplus-Strong voice per string. The callback allocates nothing and takes no locks.

The main thread only draws the overlay (`ui/`), because macOS needs GUI work there.

**Racket tracking is markerless.** There are no printed markers (the ArUco version was
removed; the PID's marker sections are superseded). The player clicks four points in
the window: the top of the head (tip), the throat (head meets shaft), where the grip
starts, and the handle end. `FlowTracker` then follows corner features on a ring around the
head's rim (its width fitted to the rim's edge at lock) and on a narrow strip along the
shaft and handle, with pyramidal Lucas-Kanade (forward-backward checked), dropping any
inside a hand. Each feature keeps its lock-frame anchor, and one RANSAC similarity
transform from the anchors to the features' current positions is fitted per frame and
applied to the clicked tip and throat to give the `RacketPose`. Fitting against fixed
anchors rather than chaining frame-to-frame fits is what stops slow rotation drift (a
round badminton rim barely pins rotation; the handle strip helps). New features are
only added while the last fit kept at least half of its features, so a fast swing
ends in "racket lost" rather than locking onto the background. The grip and handle-end
clicks set `chords.handle_u_near/far`, so the four chord sections cover the grip only.
Known limits: a fast swing can lose it (re-lock with `l`), and it cannot tell the
racket left the frame slowly (the static background keeps "tracking"). Its one
setting is `[tracking] hold_ms`.
No network or AI call ever sits between a strum and its sound.

Play rules (decided, do not reopen without the lead):

- **Strum gate** defaults to the **pick grip**: thumb pad on the side of the index =
  armed, thumb off it = disarmed. The grip score is the thumb gap alone (thumb tip to
  the index, in palm sizes; `GRIP_SCORE_VERSION` 3), which told pick-grip strums from
  relaxed returns best on recorded takes. Per-user calibration and a narrow hysteresis:
  arm below 28 %, disarm above 33 % of the closed-to-open span, 1 frame each (the
  strict setting from `tools/grip_sweep.py`; `grip_arm_below`, `grip_release_above`,
  `grip_hold_frames`, `grip_release_frames`; `[ ] , .` move them live). The
  **mic gate** is optional and off by
  default; with both on, a crossing needs a pick grip *and* an onset in the last 80 ms.
- **Chords** come only from which of **4 handle sections** the fretting hand grips:
  C, G, Am, F from head to handle end. There is no finger chord-shape recognition. The
  fretting hand is whichever hand is nearer the handle; no hand = open strings.
- **Strings**: 6 segments parallel to u, at u 0.15..0.85, spread across v -0.30..+0.30
  head lengths. Each string has a 40 ms refractory window, with no global cooldown.
- **Palm mute**: open strumming hand resting on the head for 150 ms damps all strings.

## Contracts (`anystring/types.py`, frozen)

| Type | Producer -> consumer | Meaning |
| --- | --- | --- |
| `Vec2(x, y)` | everywhere | pixel point or vector (a NamedTuple) |
| `RacketPose` | WP1 -> WP2 | `origin` = throat point, `u` = unit throat->tip, `v` = perpendicular, `head_len` px (throat to tip) |
| `HandObs` | WP1 -> WP2 | `(21, 2)` filtered `landmarks` and unfiltered `raw`, pixels |
| `Onset` | WP4 -> gate | mic contact onset, `strength` = ratio over noise floor |
| `StringHit` | inside WP2 | string 0 (low E)..5, `"down"`/`"up"`, speed in head lengths/s |
| `NoteEvent` | WP2 -> WP3 | `on` / `damp` / `all_off`, MIDI note, velocity 1..127 |
| `StrumGate` | WP4 -> WP2 | Protocol: `observe_hand`, `observe_onset`, `note_emitted`, `allows`, `grip` |

Conventions all packages share (the docstring in `types.py` is the source of truth):

- `t` is `time.monotonic()` seconds; for camera data it is the **capture** time.
- Pixels are in the **full-resolution, un-mirrored** camera frame. Detectors that run
  on a downscaled copy scale back. Only the overlay mirrors.
- `v = (-u.y, u.x)`. Racket coords are in head lengths:
  `px = origin + (u*U + v*V) * head_len`.
- "down" = moving from the low-E side toward high E.
- `play/engine.py` receives its `StrumGate` by injection and never imports
  `play/gate.py`.

All thresholds live in `anystring/config.py` (`Config`, TOML overrides, unknown keys
rejected); per-user measurements live in `Calibration` (`~/.anystring/`, or
`$ANYSTRING_HOME`; an older `~/.racketguitar/` is used until the new one exists). Do not hard-code a tuning number: add a `Config` key (via the
lead) and read it. `tests/test_config_wiring.py` fails if a key is never read.

## The import rule

**No package imports another package except `types.py` and `config.py`.** A work
package may import `anystring.types`, `anystring.config`, its own modules,
third-party libraries and the stdlib, and nothing else. `tests/test_import_boundaries.py`
enforces this with an ownership map:

| WP | Owns |
| --- | --- |
| WP0 (lead) | `types.py`, `config.py`, package `__init__`s, `tests/fakes/`, top-level tests |
| WP1 Vision | `capture/camera.py`, `vision/` (hands, filters, markerless `flow.py`), `tests/vision/` |
| WP2 Play logic | `geometry/`, `play/strum.py`, `play/chords.py`, `play/mute.py`, `play/engine.py`, `tests/play/` |
| WP3 Audio | `audio/`, `tests/audio/` |
| WP4 Contact gate | `capture/mic.py`, `play/gate.py`, `tests/gate/` |
| WP5 Integration | `app.py`, `pipeline.py`, `ui/`, `replay/`, `tools/latency_probe.py`, `tests/integration/` (the only layer allowed to import across packages) |
| WP7 Song mode | `song/` (sheets, voicings, player, library, voice), `tests/song/` |
| WP8 Coach | `coach/` (take log, analysis, Claude phrasing, ElevenLabs speech), `tests/coach/` |

`play/` and `capture/` are shared directories: WP2 and WP4 both live in `play/`, WP1
and WP4 in `capture/`, and they still must not import each other. A new file there
needs an owner added to the map by the lead. Changes to `types.py`, `config.py`,
`pyproject.toml` or the fakes also go through the lead; request them in your report
rather than editing.

## Testing

- Every test runs **headless**: no camera, mic, speakers or network. Anything that
  needs hardware is `@pytest.mark.hardware`, and must never be the only proof that
  something works.
- Use the fakes in `tests/fakes/` as ground truth: `make_pose`, `uv_to_px`/`px_to_uv`,
  `hand_template`/`place_hand` (`open`, `pick`, `fist`), `strum_track`,
  `fretting_hand`, `expected_slots_crossed`, `brush_strokes`, `room_noise`,
  `speaker_bleed`, `blocks`, `render_racket` (a plain textured racket on a static
  background, with exact tip and throat), `ScriptedGate`, `FakeClock`. Production code
  never imports them.
- Fixtures in `tests/conftest.py`: `rng` (seeded), `cfg` (defaults with
  `low_e_side="neg_v"`), and an autouse temp `ANYSTRING_HOME`.
- Tests must be deterministic: seed every RNG, and never sleep or read the wall clock.

## Gotchas

- The package has a module named `types.py`. Always import it as
  `anystring.types`, and never run a file inside `anystring/` as a script (use
  `python -m`), or it shadows the stdlib `types`.
- `HandObs` holds numpy arrays, so `==` on whole instances is not meaningful. Compare
  fields with `np.array_equal`.
- mediapipe brings `opencv-contrib-python`. Never add `opencv-python` too, because
  both install the same `cv2` module.
- The MediaPipe hand model (`models/hand_landmarker.task`) is downloaded, not committed.

## Phases

Work is gated: no phase starts until the lead approves the previous gate. Phase 0 is
WP0 (this skeleton). Phase 1 is WP1 to WP4 in parallel, each built against only its
own row of the PID's work-package table. Next comes WP5 integration, then on-device
acceptance on the MacBook, then the WP6 verification by a fresh agent.

## Clips (F8)

- `replay/clip.py` `ClipBuffer`: the display loop feeds it each drawn window image (kept as
  JPEG, last `[clip] seconds`) and the play step's `NoteEvent`s; F8 (or `f`) copies the ring
  and encodes it on a worker thread through `DemoRecorder`, into `memories/` (git-ignored).
  Never on the play path.
- `ui/clip_hud.py` draws the hint, flash and card from `Status.clip` (a `ClipStatus`) on the
  display image *after* the frame went to the clip buffer and demo, so videos never contain
  it. Its animation is a pure function of the capture time, like the rest of the overlay.

## Song mode (WP7)

- A chord sheet (ChordPro or chords above lyrics, in `~/.anystring/songs/`, never
  in the repo since sheets hold lyrics) becomes a timeline. The one exception is
  `examples/songs/`: chords-only demo sheets with no lyric text, copied in by hand. `SongPlayer` plans which
  chord labels each handle section over time; chords still come only from grip
  position. Relabels reach the engine only via `FrameInput.chords` ->
  `Engine.set_chords`, which keeps gate, strum and mute state, so recordings replay
  exactly and a relabel on the beat drops nothing.
- Voice (`s`): a short clip goes to ElevenLabs speech-to-text (`$ELEVENLABS_API_KEY`)
  on a worker thread; titles match song files fuzzily. No network call is on the play
  path. Tests use made-up lyrics only.

## Coach (WP8)

- Feedback on a take, never on the play path. `replay/take.py` `TakeLogger` copies each
  frame's `EngineSnapshot` (chord held, grip, mute, new string hits, the engine's
  refractory-drop count) into a rolling `coach.log.TakeLog`; `log_from_session` builds the
  same log from a recording by stepping it through a `Pipeline` (song starts are noted in
  `session.json` under `songs`).
- `coach/analyze.py` measures it: strokes (missed / grip dropped / thin, up vs down),
  timing against the beat grid in song mode (dragging, rushing, drift), stroke-rate
  steadiness outside a song, per chord-change lag, misses and early let-go
  (e.g. G -> Am), chord accuracy, likely unintended palm mutes, racket lost. Findings
  are ranked; `facts` is the compact JSON the LLM sees.
- `coach/phrase.py` words it with Claude (`[coach] llm_model`, `$ANTHROPIC_API_KEY` or an
  `ant auth` profile; server-side refusal fallback on); `template` is the no-network
  fallback. `coach/speak.py` is ElevenLabs text-to-speech (`$ELEVENLABS_API_KEY`), PCM out.
- Triggers: a song reaching its end (`[coach] after_song`), `k`, or saying "how was that?" /
  "give me feedback" (`parse_command` kind `feedback`). It speaks only after
  `quiet_s` without a string crossing. `uv run anystring coach DIR [--no-llm] [--speak]`
  judges a recording.

## Integration (WP5)

- `pipeline.Pipeline.step(FrameInput)` is the one per-frame play step: drain onsets into
  the gate, then run the engine. The live app and the replay harness both call it, so
  a replay proves the live behaviour. Never add play logic outside it.
- Replay feeds a session's recorded observations (poses, hands, drained onsets) back
  through the pipeline and must match exactly; `--from-mic` re-detects onsets from
  `mic.wav`. Poses always replay as recorded (the tracker needs the lock clicks).
  `[replay] video = true` also saves lossless FFV1 video, only for inspecting a take.
  Changing settings mid-recording (calibration, re-lock, gate toggles) updates the
  recording or splits it into `-partN`, so every recording replays exactly.
  `tests/integration/song.py` is a scripted four-chord performance used end to end.
- `audio/probe.py` (`--probe-tone`) plays a 1 kHz beep per note, so
  `tools/latency_probe.py` can measure tap-to-sound latency from a phone recording.

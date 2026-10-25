# Anystring: Project Initiation Document

Oct 24, 2026 · @PMD

> **Amendment (Oct 24, 2026): racket tracking is markerless.** The printed ArUco
> markers described below were built, then removed after on-device testing. Instead,
> the player clicks the racket four times in the window (the top of the head, the
> throat, where the grip starts, and the handle end), and the app follows it by optical flow
> on the head's rim (`anystring/vision/flow.py`). The four chord sections span the
> grip only. Wherever this document mentions markers, marker images, printing markers,
> the marker PDF tool or marker-based replay, read it as superseded. `CLAUDE.md`
> describes the current implementation.

## Executive summary

Anystring (originally "Racket Guitar") turns a tennis racket, or any guitar-shaped object, into a playable guitar using only a MacBook's camera and microphone. The user tapes two printed markers to the racket, the app overlays virtual strings on the string bed, and strumming the real strings plays guitar notes in under 50 ms.

The camera decides **where** the hand is (which string, which chord) and **whether** a stroke counts: a closed pick grip arms the strumming hand, an open hand disarms it. The microphone can optionally confirm real contact with the strings, and the physical racket gives the player real tactile feedback. That combination fixes the core weaknesses of webcam air guitars, most visibly the missing up-strum.

v1 delivers a local Python app for macOS: racket tracking, per-string strum detection, chord selection by grip position on the handle, palm muting, and low-latency synthesized guitar sound. AI services (Gemini, ElevenLabs, Jev) are a v2 layer for setup and sound design and never sit in the real-time path.

## Background and prior art

Webcam air guitars already exist, so v1 competes on feel, not novelty. The closest reference is [Maestro (TreeHacks 2026)](https://github.com/markmusic27/treehacks-26), which we reviewed line by line.

| Project | How it plays | Main limitation |
| --- | --- | --- |
| [Maestro](https://github.com/markmusic27/treehacks-26) | Strum across a line between the two wrists; iPhone app picks strings and frets; magenta tape on a pole sets the octave | Down-strums only, 80 to 150 ms lag, needs a phone app and network |
| [air-instrument.pages.dev](https://air-instrument.pages.dev) | Browser air guitar with strum-pattern scoring | Cannot tell a strum from lifting the hand away, so it splits the body into a down-only zone |
| [Simple Hand Tracking Chords](https://jasperbernaers.com/simple-hand-tracking-chords/) | Finger count picks the chord, a downward flick strums | Gesture vocabulary, not a guitar |

**Maestro shortcomings we design against:**

- **Latency:** reads the previous frame's hand result, uses fake +33 ms timestamps, heavy fixed smoothing, one serial loop.
- **Playing feel:** up-strums discarded, one neck line instead of strings, a 130 ms global cooldown, an infinite crossing line, fixed 1.5 to 2.5 s notes, velocity squeezed to 70 to 110.
- **Robustness:** strings anchored to wrists that drop out, hardcoded handedness, color tape that breaks under lighting changes, mixed 3D units.
- **Setup:** Python app plus iPhone app plus SoundFont plus FluidSynth; Mac-only audio; an unbounded timer list; no tests; no license.

**Our differentiator:** a real object with real strings. Contact makes sound the mic can hear, so up-strums work and false triggers drop; the markers sit on a rigid racket, so strings never jump; and the player feels the strings. Maestro has no license, so we reuse ideas, not code.

## Objectives and success criteria

v1 succeeds if a person who has never seen the app can strum a recognizable four-chord song on a racket within five minutes of setup, with no perceptible lag.

| Objective | Metric | Target |
| --- | --- | --- |
| Responsive | Strum-to-sound latency (finger crosses string to audio out) | p50 under 40 ms, p95 under 60 ms |
| Accurate | Strums detected vs strums performed (scripted test, 100 strums) | 95% or more |
| Clean | False triggers per minute while holding the racket but not strumming | Under 1 |
| Full rhythm | Up-strums played correctly | 90% or more of up-strums |
| Fast | Strums per second sustained without drops | 8 or more |
| Stable | Racket tracking kept while moving at playing speed | 98% of frames |
| Quick setup | Print markers to first sound | Under 5 minutes |
| Simple | Hardware needed | Racket, 2 printed markers, MacBook. No phone, no network |

Latency is measured on the MacBook by recording a string tap and the resulting sound with a second device (see Testing and acceptance); the app also logs its internal detect-to-audio time on every note.

## Scope

v1 is a local, single-player guitar on macOS. Everything else waits until the core feels right.

**In scope for v1**

- Python 3.12 app for macOS (Apple silicon), built-in or USB webcam, built-in mic
- Racket tracking with two printed ArUco markers (head tip and throat)
- 6 virtual strings across the racket head, tuned E A D G B E
- Per-string strum detection, both directions, velocity sensitive
- Pick-grip gate from the live video (closed pick grip strums, open hand never does), with optional mic onset gating
- Automatic chord selection by which section of the handle the left hand grips (4 sections, fixed preset C, G, Am, F); no finger chord-shape recognition
- Palm mute (flat strumming hand resting on the head)
- Karplus-Strong plucked-string synth, with optional SoundFont playback
- Calibration screen: marker check, string preview, mic level, latency readout
- Debug overlay and session recording (video + landmarks + audio) for tests

**Out of scope for v1**

- Browser or iOS versions
- Other instruments (harp, drums, theremin)
- Any cloud AI service in the app
- Song library, scoring, or lessons
- Multi-player

**Later phases**

- v2: AI setup layer (voice setup with Gemini Live, ElevenLabs sound packs, Jev for slow context checks)
- v3: More instruments on the same engine: harp (crossing), theremin (continuous), drums (camera for location, mic for timing)
- v4: Port the real-time core to Swift (Vision + AVAudioEngine) for a native Mac app

## Solution design

Four threads pass small timestamped messages through queues, so no slow step can delay a note. All geometry runs in the racket's own 2D coordinate frame, which cancels racket motion and fixes Maestro's mixed-unit math.

&#91;embedded content: runtime architecture · 4 threads\]

The camera path decides which string and chord; the hand's grip (and optionally the mic) decides whether a crossing counts. Both meet in the play engine.

**Threads**

1. **Camera thread:** OpenCV AVFoundation capture at 60 fps (fallback 30), buffer size 1, keeps only the newest frame with its capture time from `time.monotonic()`.
2. **Vision thread:** per frame, ArUco marker detection, then MediaPipe Hand Landmarker in VIDEO mode (`detect_for_video`, synchronous, real timestamps) on a 640x360 copy. One Euro filters on fingertips and marker corners. Runs geometry, strum, chord and mute logic and emits note events.
3. **Mic thread:** `sounddevice` input stream, 48 kHz, 64-sample blocks (1.3 ms). Detects contact onsets and timestamps them.
4. **Audio thread:** `sounddevice` output callback, 48 kHz, 128-sample blocks (2.7 ms). Pulls note events from a queue and renders Karplus-Strong voices.

The main thread only draws the overlay with `cv2.imshow` at up to 30 fps (macOS requires GUI work on the main thread).

**Key algorithms**

- **Racket frame.** Marker 0 sits at the head tip, marker 1 at the throat. Origin = throat marker centre, axis *u* = unit vector toward the tip, axis *v* = perpendicular. Every point is converted to (u, v) in units of head length, so distances do not change as the racket moves toward or away from the camera.
- **Strings.** Six segments parallel to *u* (like guitar strings along the neck), spanning u = 0.15 to 0.85 of the head, evenly spaced across v = -0.30 to +0.30 of head width. Virtual spacing is wider than the real strings on purpose.
- **Strum detection.** For the strumming index and middle fingertips, track each string's signed distance in *v*. A hit = sign change between consecutive samples, crossing point inside the segment's *u* range, and perpendicular speed above a threshold. Direction comes from the sign change. Each string has its own 40 ms refractory window, replacing the 130 ms global cooldown. A sweep crosses strings in order, so the strum cascade is free.
- **Strum gate.** Two sources, chosen in config. Pick grip (default, camera only): a crossing counts only while the strumming hand holds a pick grip, thumb pad pressed against the side of a curled index finger, other fingers curled. An open hand never strums, so you open it to lift away. The grip is classified every frame from the thumb gap (thumb tip to the side of the index, normalised by palm size), with a narrow hysteresis: it arms below 28 % of the way from your closed grip to your open hand and disarms above 33 %, one frame each. Holding a pick is the thumb on the index, and on recorded takes the thumb gap alone told pick-grip strums from relaxed returns far better than a score that also counted the other fingers (2 of 89 relaxed returns played, 9 % of strum crossings missed). A short calibration records your own closed and open hand to set the thresholds. Mic contact (optional): the mic fires an onset on a jump in high-band (2 to 8 kHz) spectral flux above an adaptive noise floor. Physical contact is heard before the camera sees the crossing (camera lag is 30 to 50 ms), so the rule is: a crossing plays only if an onset happened in the last 80 ms, and an onset keeps the gate open for 150 ms during a sweep. This adds no latency and removes lift-away false strums, which is what makes up-strums safe. With both enabled, a crossing needs a pick grip and a recent onset. Both can be turned off for free air play.
- **Chords.** The fretting hand is whichever hand is nearer the handle, so handedness needs no setting. Its middle-finger knuckle (landmark 9) is projected onto the handle axis (u below 0) to get t from 0 to 1, split into 4 zones with 0.05 hysteresis. Each section maps to a fixed chord voicing (6 MIDI notes, or none for an unplayed string), ordered C, G, Am, F from the head toward the handle end. Finger positions are never read as chord shapes: only which section the hand is in matters, so any object with a handle works. No fretting hand visible = open strings.
- **Palm mute.** Strumming hand open (all fingers extended), palm centre inside the head, speed below a threshold for 150 ms = damp all ringing strings within 30 ms.
- **Velocity.** Perpendicular fingertip speed maps to MIDI velocity 20 to 127 on a curve (gamma 1.6), driving both loudness and brightness.
- **Synth.** One Karplus-Strong voice per string; re-striking a string restarts its voice (as on a real guitar). Optional FluidSynth backend behind the same interface.
- **Record and replay.** Every session can save frames, landmarks, marker poses, onsets and audio with timestamps. A replay harness feeds recordings through the same detectors with no camera or mic, which makes all play logic testable.

**Package layout**

```
anystring/
  types.py           shared dataclasses (the contracts)
  config.py          settings + calibration file (TOML)
  capture/           camera.py, mic.py
  vision/            markers.py, hands.py, filters.py (One Euro)
  geometry/          racket.py (pose -> racket frame, strings, handle)
  play/              strum.py, gate.py, chords.py, mute.py, engine.py
  audio/             synth.py (Karplus-Strong), output.py, fluid.py
  ui/                overlay.py, calibrate.py
  replay/            record.py, harness.py
  app.py             wiring + CLI
tests/
tools/               print_markers.py, latency_probe.py
```

**What you see on screen**

The overlay is drawn on the mirrored camera feed using the same racket transform the detectors use: points are computed in racket coordinates (u, v) and projected back to pixels each frame. What you see is therefore exactly what the app is testing against.

&#91;embedded content: play screen mock-up · overlay on the mirrored camera feed\]

| Input | What the app reads | How it is drawn |
| --- | --- | --- |
| Markers on the racket | Racket position, angle and size | Racket outline and head ellipse; markers boxed green when found, red when lost |
| Racket frame | Where the 6 virtual strings sit | 6 lines across the head, thin and neutral at rest |
| Strumming fingertip | Path between frames, speed | A short fading trail; dot turns accent colour when armed |
| Strumming hand shape | Pick grip (armed) or open (disarmed) | Label beside the hand plus trail colour: accent when armed, grey when open |
| Finger crossing a string | String number, direction, velocity | That string flashes and fades over 300 ms; brightness = velocity, one colour for down, another for up |
| Fretting hand on the handle | Which of the 4 sections it is in | Handle split into F, Am, G, C bands; the active band lights up and the chord shows large in the corner |
| Open palm resting on the strings | Mute | All strings dim at once with a short "muted" label |
| Mic (optional) | Contact onsets | Small level meter that ticks on each onset |
| Pipeline health | Latency and frame rate | Corner readout: last strum-to-sound time, fps, gate state |

A calibration screen uses the same overlay before play: it checks both markers, records your closed pick grip and open hand, and shows the string lines so you can confirm they sit on the head.

## AI layer (v2)

No cloud model may sit between a strum and its sound. Network round trips alone exceed the 40 ms budget, so AI handles setup, sound design and slow context only.

| Service | Role | Cadence | Why not real-time |
| --- | --- | --- | --- |
| [Gemini Live](https://ai.google.dev/gemini-api/docs/models/gemini-3.8-live) | Voice-guided setup: "make zone 2 an A minor", "tune it like a ukulele", "switch to harp" | On request | Video input is 1 fps; speech round trips are hundreds of ms |
| Gemini (image input) | One-shot check of a calibration frame: are both markers and the head visible, is lighting OK | Once at setup | Single call, latency irrelevant |
| [ElevenLabs sound effects](https://elevenlabs.io/docs/eleven-api/quickstart) | Generate instrument texture samples from text ("warm nylon pluck", "sitar string"), cached locally, pitch-shifted per string | Once per sound pack | Generated ahead of time, played from cache |
| [Jev](https://openrouter.ai/docs/guides/community/jev) / Decisions API | Typed checks at 0.5 to 1 Hz: practising or idle, setup mode or play mode, turn a spoken request into a config change | Background | About 300 ms per call; Jev is text-only, image questions need a multimodal Decisions model |

**Rules for v2:** every AI call is async and optional; the app plays normally with no network; outputs become config or cached samples, never live note decisions. ElevenLabs output does not hit exact pitches reliably, so one sample per timbre is generated and pitch-shifted locally.

## Execution plan

v1 is built in five gated phases; only phase 3 needs your MacBook, everything before it runs in the cloud workspace.

&#91;embedded content: execution roadmap · 5 phases, 5 gates\]

No phase starts until the previous gate passes.

## Work packages for sub-agents

The lead writes the shared contracts first (WP0); four sub-agents then build in parallel against those contracts, one integrates, and a separate agent that never saw the build verifies it. Each package may import only `types.py`, `config.py` and its own modules, and must ship tests that run headless (no camera, mic or speakers).

**Shared contracts (`anystring/types.py`, frozen after WP0)**

```python
@dataclass(frozen=True)
class RacketPose:  # from WP1
    t: float  # capture time, time.monotonic() seconds
    visible: bool
    origin: Vec2  # throat marker centre, pixels
    u: Vec2  # unit vector throat -> tip, pixels
    v: Vec2  # unit perpendicular
    head_len: float  # pixels, throat -> tip


@dataclass(frozen=True)
class HandObs:  # from WP1
    t: float
    landmarks: np.ndarray  # (21, 2) pixels, filtered
    raw: np.ndarray  # (21, 2) pixels, unfiltered


@dataclass(frozen=True)
class Onset:  # from WP4
    t: float
    strength: float


@dataclass(frozen=True)
class StringHit:  # inside WP2
    t: float
    string: int  # 0 = low E ... 5 = high E
    direction: Literal["down", "up"]
    speed: float  # head lengths per second, perpendicular


@dataclass(frozen=True)
class NoteEvent:  # WP2 -> WP3
    t_detect: float
    kind: Literal["on", "damp", "all_off"]
    string: int
    midi: int | None
    velocity: int  # 1..127
```

| WP | Agent | Builds | Depends on | Done when |
| --- | --- | --- | --- | --- |
| WP0 | Lead | Repo skeleton, `pyproject.toml` (uv), `types.py`, `config.py`, test scaffolding, fake data helpers | None | Imports work; `pytest` runs green on an empty suite |
| WP1 | Vision | `capture/camera.py`, `vision/markers.py`, `vision/hands.py`, `vision/filters.py`, `tools/print_markers.py` | WP0 | Pose recovered within 2 px and 1 degree on synthetic rendered marker images; One Euro filter unit-tested; hand wrapper uses VIDEO mode with real timestamps; printable marker PDF |
| WP2 | Play logic | `geometry/racket.py`, `play/strum.py`, `play/chords.py`, `play/mute.py`, `play/engine.py` | WP0 | Synthetic fingertip tracks: every crossing detected with correct string and direction; no hit outside a segment; per-string refractory honoured; chord zones with hysteresis; mute fires; gate interface respected |
| WP3 | Audio | `audio/synth.py`, `audio/output.py`, `audio/fluid.py` | WP0 | Karplus-Strong pitch within 5 cents of target; voice restart and damping work; callback renders 128 samples in under 0.5 ms; offline render to WAV for tests |
| WP4 | Contact gate | `capture/mic.py`, `play/gate.py` | WP0 | On synthetic WAVs (brush noise, silence, room noise, speaker bleed), onsets found within 5 ms; false onsets under 1 per minute on noise; gate window logic unit-tested |
| WP5 | Integration | `app.py`, `ui/overlay.py`, `ui/calibrate.py`, `replay/`, `tools/latency_probe.py`, README | WP1 to WP4 | `uv run anystring` launches on macOS; replay harness reproduces a recorded session deterministically; end-to-end replay test passes |
| WP6 | Verifier (fresh agent) | Test report against the success criteria; code review | WP5 | Report lists each criterion as pass, fail or needs on-device check, with evidence |

WP1 to WP4 run in parallel. Each agent receives this PID, the contracts file, and only its own row.

## Risks and mitigations

The two risks most likely to sink v1 are speaker bleed into the mic and fast strums slipping between frames; both have fallbacks that keep the app playable.

| Risk | Likelihood | Impact | Mitigation |
| --- | --- | --- | --- |
| Speaker output leaks into the mic and opens the gate | High | High | Headphones recommended at setup; gate ignores onsets within 25 ms after each note it emits; high-band filter favours string brush over guitar tone; gate can be switched off |
| Racket strings are too quiet for the mic to hear | Medium | High | Calibration step measures brush level vs room noise; laptop mic sits close to the racket; optional clip-on piezo contact mic on the racket frame, which hears only the racket; fall back to vision-only mode with down-strum bias |
| Fast strums cross strings between frames | Medium | Medium | 60 fps capture; segment crossing tests the path between frames, not positions; one onset opens the gate for the whole sweep |
| MacBook webcam delivers only 30 fps | High | Medium | Design and thresholds tuned for 30 fps; 60 fps is a bonus; a USB webcam is an option |
| Hand occludes a marker mid-strum | Medium | Medium | Markers at tip and throat sit away from the strumming area; hold last pose up to 150 ms; optional third marker on the frame side |
| MediaPipe loses the strumming hand during fast motion | Medium | Medium | Lower detection confidence while tracking; ignore a hand for strums on its first frame back |
| Camera latency higher than expected on macOS | Medium | Medium | Measure early with the latency probe in phase 1; switch to a Swift capture shim if needed |
| Python thread jitter causes audio glitches | Low | High | Audio callback allocates nothing and takes no locks; events cross threads through a lock-free queue |
| Sub-agents drift from the contracts | Medium | Medium | Contracts frozen in WP0; integration tests in WP5; independent verification in WP6 |

**Pick-grip gate risk:** a curled hand facing the camera can hide the thumb and index tips, so a closed grip may read as open for a frame or two. Mitigations: per-user calibration of closed and open poses, hysteresis, and the optional mic as a second opinion. The default holds neither arming nor disarming (1 frame each), the strict setting measured with `tools/grip_sweep.py`: a held disarm let relaxed return strokes play, and with the narrow thresholds a 2-frame arming hold did too. The costs are an occasional dropped note when a closed grip misreads as open, and a single frame that misreads as a pick grip can arm the gate; `grip_hold_frames = 2` trades some of the second for more of the first.

## Testing and acceptance

Most logic is proven headless in the cloud before it touches hardware; only latency, lighting and feel need the MacBook.

**Headless (runs in CI and for sub-agents)**

- Unit tests per package, as listed in each work package's done criteria
- Synthetic scenes: rendered marker images at varied angles, distances and blur
- Synthetic fingertip tracks: slow and fast strums, both directions, near misses outside segments, jitter
- Synthetic audio: brush noise, room noise, speaker bleed of the synth's own output
- End-to-end replay: a recorded session runs through the full pipeline and must produce the same note events every time

**On the MacBook (acceptance)**

- [ ] Latency probe: tap a string beside the mic and measure tap-to-sound with a second recording device; p50 under 40 ms, p95 under 60 ms
- [ ] 100 scripted strums (50 down, 50 up) recorded: 95% detected, 90% of up-strums correct
- [ ] 2 minutes holding the racket without strumming: under 2 false notes
- [ ] Strum at 8 per second for 10 seconds without drops
- [ ] Move the racket while playing: tracking holds on 98% of frames
- [ ] New user from printed markers to first sound in under 5 minutes
- [ ] Play a four-chord song (C, G, Am, F) end to end

## Open decisions

- [ ] **Where the code lives:** a connected folder on your MacBook, a new GitHub repo, or files sent in chat
- [ ] **Default chords per handle zone:** decided: four handle sections, C, G, Am, F from head to handle end, no chord shapes; open only whether to add more presets later
- [ ] **Racket hold:** like a guitar (handle as the neck, strings along the racket) is the default; flat on a table is the fallback
- [ ] **Python first, Swift later:** Python v1 for speed of iteration, Swift port in v4 if latency demands it
- [ ] **Headphones required or recommended:** decides how hard the speaker-bleed problem must be solved in v1
- [ ] **Hackathon framing:** whether this becomes the Medal hackathon entry (best UX category), which would pull the theremin or drums forward

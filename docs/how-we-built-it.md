# How we built it

Anystring is about 8,000 lines of Python (the `anystring` package plus
`tools/`), backed by 864 headless tests. It runs on a MacBook with its built-in camera
and speakers; nothing is attached to the racket. This page walks through how a strum
travels from a camera frame to a sound, and how the pieces were built and checked.

## The pipeline at a glance

```
camera (own thread) ──► vision thread ──────────────────────────► audio callback
 newest frame only      hands → racket pose → strum / chord /     Karplus-Strong,
 stamped on capture     mute → gate → NoteEvent                   one voice per string

mic (own thread, optional) ──► contact onsets ──► gate
main thread: draws the overlay only
```

Four threads pass small timestamped messages through queues, so a slow step (drawing,
a speech request) can never delay a note. The camera thread keeps only the newest
frame, so there is no backlog. The audio callback allocates nothing and takes no locks.
Speech-to-text for voice commands runs on a separate worker, off the play path.

## 1. Seeing the hands

[`vision/hands.py`](../anystring/vision/hands.py) runs the MediaPipe Hand Landmarker
in video mode on a downscaled 640x360 copy of each frame. Landmarks come back
normalised, so they are scaled back to full-resolution pixels. Each hand gets a
One Euro filter ([`vision/filters.py`](../anystring/vision/filters.py)) to remove
jitter without adding lag. The filtered and the raw landmarks are both kept: the grip
gate uses the raw ones so that smoothing never delays a release.

## 2. Following a plain racket with no markers

We first planned printed ArUco markers, then removed them: the point is that any racket
works. Instead you click four points once (the head's tip, the throat, where the grip
starts, and the handle end), and [`vision/flow.py`](../anystring/vision/flow.py) follows
the racket from there:

- **Features:** corner features are picked on a ring around the head's rim and along a
  narrow strip down the shaft and handle. The string mesh inside is skipped, because its
  repeating grid fools optical flow. Features inside a hand are never picked or kept.
- **Tracking:** pyramidal Lucas-Kanade, checked forward and backward.
- **Pose:** every feature remembers where it was at lock. Each frame, one similarity
  transform (rotation, scale, shift) from those anchors to the features' positions is
  fitted with RANSAC, so features on a passing hand or the background are outvoted.
  That transform moves the clicked tip and throat to give the pose. Fitting against fixed
  anchors, rather than chaining frame-to-frame fits, stops small errors adding up into
  slow rotation drift.
- **Rim fit:** at lock, the head's width is fitted to the strongest edge, since a badminton
  head is rounder than a tennis head.
- **Losing the racket:** new features are only added while the last fit kept at least half
  of its features, so a fast swing ends in "racket lost" rather than locking onto the
  background.

## 3. A racket-sized coordinate system

All play logic works in the racket's own frame
([`geometry/racket.py`](../anystring/geometry/racket.py)): `u` runs from throat to tip,
`v` is perpendicular, and distances are in head lengths. Moving the racket or standing
further from the camera cancels out. Six strings run parallel to `u`, spread across the
head, and the handle is split into four chord sections. The overlay draws with the same
transform the detectors use, so what you see is what the app tests.

## 4. From fingertips to notes

- **Strum detection** ([`play/strum.py`](../anystring/play/strum.py)): a hit is a
  fingertip's signed distance to a string changing sign between two frames, with the
  crossing point inside the string and enough perpendicular speed. A fast sweep that
  crosses several strings between two frames still gives one ordered hit per string, with
  timestamps interpolated along the path. Two fingertips pass each string in one sweep, so
  duplicates are dropped, and each string has a 40 ms refractory window.
- **Chords** ([`play/chords.py`](../anystring/play/chords.py)): which of four handle
  sections the fretting hand grips picks C, G, Am or F. We never read finger shapes. No
  hand on the handle means open strings.
- **Palm mute** ([`play/mute.py`](../anystring/play/mute.py)): an open hand resting on
  the head for 150 ms damps all strings.
- **Engine** ([`play/engine.py`](../anystring/play/engine.py)): per frame, it splits the
  hands into fretting and strumming, then turns each allowed hit into a note whose
  velocity comes from the strum speed.

### The strum gate: pick grip

Plain air strumming fires on every pass of the hand, including the return stroke. The
gate ([`play/gate.py`](../anystring/play/gate.py)) only lets a crossing sound while
the strumming hand is in a **pick grip**, with the thumb pad on the side of the index
finger. The score is the thumb-to-index gap in palm sizes, so it ignores hand size and
rotation. It is calibrated per user, with a narrow hysteresis.

We chose this by measuring, not guessing. `tools/grip_sweep.py` replays recorded takes
and ranks thresholds by wrong-sounding moments. On our takes the thumb gap alone beat
any mix with finger reach: it blocked all but 2 of 89 relaxed returns while missing 9% of
strums. An optional mic gate (high-band spectral-flux onsets from the microphone) can
also be required; it is off by default.

## 5. Sound

[`audio/synth.py`](../anystring/audio/synth.py) is a Karplus-Strong plucked-string
synth: a delay-line loop per string, tuned per note with a fractional delay solved so
each note is within a fraction of a cent of equal temperament. Buffers are preallocated
and a 128-sample block renders with one vectorised matrix multiply, so the audio callback
never allocates. An optional FluidSynth backend can use a SoundFont.

## 6. Song mode

[`song/`](../anystring/song) turns a chord sheet (ChordPro, or chords above lyrics as
chord sites show them) into a timeline. The handle only has four sections, so
[`song/player.py`](../anystring/song/player.py) plans in advance which chord label
each section carries, relabelling a section as soon as its old chord is no longer needed
(furthest-next-use replacement). A chord already on the handle never moves. Chords still
come only from where you grip. A voicing table and a barre-shape fallback
([`song/voicings.py`](../anystring/song/voicings.py)) map any chord name to notes for
the six strings, with capo support.

You can say "let's play take on me": a short clip goes to ElevenLabs speech-to-text on a
worker thread, and the title is matched fuzzily against the song files. The sheets live
in `~/.anystring/songs/`, not in the repo, since they contain lyrics.

## 7. How we kept it honest

- **One play step.** `Pipeline.step` is the single per-frame play step, shared by the live
  app and the replay harness. A recorded session (poses, hands, onsets) replays through
  it and must produce exactly the same notes, so a replay proves what happened live. We
  used replays to tune the grip gate and to debug drops frame by frame.
- **Hard package boundaries.** No package imports another except for `types.py` and
  `config.py`; a test enforces it. That is what let vision, play logic, audio and the
  contact gate be built in parallel against frozen contracts, with a fake racket,
  hands and clock in `tests/fakes/` as ground truth.
- **No hard-coded tuning.** Every threshold lives in `config.py`, and a test fails if a
  key is never read.
- **Headless, deterministic tests.** 864 tests run with no camera, mic or speakers: seeded
  RNGs, no sleeping, no wall clock.

## 8. What we measured on the MacBook

Full results are in [`acceptance.md`](acceptance.md). In short:

| Check | Result |
| --- | --- |
| 100 scripted strums | **Pass**: about 98 of 100 strums and 48 to 49 of 50 up-strokes |
| App-internal delay (detect to audio) | about 49 ms median, 64 ms at the 95th percentile |
| Tap to sound, measured with a phone | **About 160 ms median, 280 ms 95th percentile.** This missed our 40 ms target. About 110 ms of it happens before the app sees the frame (camera exposure and the macOS capture stack). The camera ran at 30 fps although we request 60. |
| 8 strums per second for 10 s | **Not met**: best window had 81 strums and 3 drops. Two drops traced to the app (a 198 ms camera stall, and a one-frame hand loss that resets the detector). |
| Moving the racket while playing | **Not met**: tracked 99.8% of frames, but the pose drifted and the strings slid off the head. The plain head is a tilted ellipse, and the tracker assumes a circle. |
| Idle false triggers, new-user test, four-chord song end to end | Not yet fully run |

The honest summary: strumming and chord-by-grip work well in a calm, steady setup; fast
or large movements can still make the tracker lose or drift off the racket, and the
sound is not instant. Re-locking with `l` fixes drift. Our known next steps are fitting
the real rim ellipse, detecting drift and prompting a re-lock, and tightening the camera
delay.

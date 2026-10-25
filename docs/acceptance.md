# Phase 3: on-device acceptance

Results of the PID's on-device checks (Testing and acceptance), run on the MacBook Air
with the badminton racket and markerless click-to-lock tracking (build `2d60cc0`).
Takes were recorded in `recordings/` (not committed); the raw takes behind these results
were discarded afterwards, so the numbers below are the record. `tools/acceptance.py` judges a recorded
take; `tools/latency_probe.py` measures tap-to-sound from a phone recording.

| # | Check | Target | Status | Evidence |
| --- | --- | --- | --- | --- |
| 1 | Latency probe | p50 < 40 ms, p95 < 60 ms | **FAIL** | p50 ~160 ms, p95 ~280 ms; see below |
| 2 | 100 scripted strums | 95% detected, 90% of ups | **PASS** | ~98/100 strums, ~48-49/50 ups, by beat timing; see below |
| 3 | 2 min idle | under 2 false notes | **CHECK** | 0 false strums in 115 s, but the hand never crossed the strings; retake needed |
| 4 | 8 strums/s for 10 s | no drops | **FAIL** | best 10 s: 81 strums, 3 drops (2 caused by the app); see below |
| 5 | Move while playing | tracked 98% of frames, no drift | **FAIL** | tracked 99.8%, but the track drifted (turned ~175 deg, shrank to half) and notes stopped; see below |
| 6 | New user to first sound | under 5 min | not run | |
| 7 | Four-chord song | C, G, Am, F end to end | not run | |

## 1. Latency: FAIL

**Result: tap-to-sound p50 ~160 ms, p95 ~280 ms (min 119, max 351), over 34 taps.**
Targets are p50 under 40 ms and p95 under 60 ms.

Method: `anystring --probe-tone` (a 40 ms, 1 kHz beep per note), with about 40 taps
on one string, made with a fingernail or pick so each tap clicks, recorded by a phone
held a few centimetres from the string. Audio: `recordings/latency-3/phone.wav`. No app
session goes with this audio (the record key did not register), but the app's
detect-to-audio log for the same run was p50 48.8 ms, p95 63.3 ms over 53 notes.

Measured on the 38 real beeps (the 1 kHz bursts at full level): 34 had a clean click
before them, and click onset to beep onset gave the figures above. Most taps fell
between 120 and 180 ms. The probe's own figure (p50 127 ms, p95 271 ms) is not used,
because it counts the tail of each beep as a second beep (104 "beeps" for 38 real
ones) and pairs those as well.

Where the time goes:

| Part | Time | Source |
| --- | --- | --- |
| Crossing to the next frame capture (30 fps) | ~16.5 ms mean, 31 ms p95 | `latency-2-part2` recording |
| Capture to processing start | ~0.2 ms | `latency-2-part2` recording |
| Hand detection, flow tracking, play step, audio buffer, DAC | ~33 ms | app log minus the above |
| **Inside the app (detect-to-audio log)** | **~49 ms p50, ~64 ms p95** | app log, three runs agree |
| Before the app gets the frame (camera exposure, macOS capture stack, OpenCV) and the gap between physical contact and the 2D crossing | ~110 ms | the rest; not yet separated |

Contributing facts:

- The camera has run at 30 fps on every take, although 60 fps is requested.
- The capture thread drains frames continuously, so there is no backlog inside the
  app, but frames are stamped when OpenCV hands them over, so delay inside the macOS
  capture stack is invisible to the app's own log.

Earlier takes that did not count:

- `latency-1`: recorded on the marker build before `2d60cc0`. No marker was found, so
  nothing played.
- `latency-2`: the phone did not hear the finger taps on the badminton string. Only 5
  of 79 beeps had any sound before them, so the probe's p50 of 142 ms there came from
  stray pairs. Implication for the mic gate: finger contact on this racket is close to
  inaudible, so the optional mic gate may not be usable with it.

Next steps (not started):

1. Measure the camera's own delay (light to beep).
2. Check whether the camera offers 60 fps at the requested size.
3. If the camera stack is most of the delay, decide on the PID's fallback, a Swift
   capture shim (lead decision).
4. Fix the probe: one beep per burst, and refuse to report when most beeps have no tap
   before them.

## 2. 100 scripted strums: PASS

Take `recordings/strums-1-part2` (report: `report.json` in the same folder): 124 s after
the lock, 30 fps, 30 dropped frames (0.8%), racket tracked 100%. Fretting hand held on
C. 100 strums alternating down and up, one per metronome tick at 60 bpm (`afplay`
loop with a 4-chime count-in).

`tools.acceptance --check strums --expected 100` says 100/100 detected and 50/50 ups,
from 107 played strums (53 down, 54 up). That overstates it: the check caps extras by
direction, so the extra fragments hide misses. Lined up against the beat instead:

- 95 strums at normal 0.7 to 1.3 s spacing, plus one at 1.36 s.
- **2 beats with no strum** (gaps of about 2.2 s at 45 s and 75 s): either missed
  detections or skipped beats; the take cannot tell which. The one at 75 s is down,
  then down, so an up is missing there.
- 7 extras: short 1- or 2-string fragments, mostly in the first 4 s while settling in,
  and one up-strum split in two at 29.6 s, where the second tracked fingertip crossed
  string 5 again 70 ms later (a real double note).

**Result: about 98/100 strums (98%) and 48 to 49/50 ups (96 to 98%), against targets of
95% and 90%.** Sweep completeness: 63 of 107 strums sounded every chord string, and 90
covered 5 or more strings. The hand read as armed 98.6% of the time and no crossing was
blocked.

Tool follow-up: with a known tempo, `--check strums` should match strums to beats
instead of capping counts, so fragments cannot mask misses.

## 4. Burst, 8 strums/s for 10 s: FAIL

Take `recordings/burst-1-part2` (report: `report.json` in the same folder): 47 s
after the lock, 30 fps, racket tracked 99.3%. Fretting hand on C, down-up-down-up per
tick at 120 bpm. The player held about 8 strums/s from 17 to 30 s, then slowed to 3 to
6 per second.

`tools.acceptance --check burst`: best 10 s window has **81 strums with 3 drops**
(intervals over 1.5x the window's 122 ms median). Each drop, traced frame by frame:

| At | Gap | Cause | Whose |
| --- | --- | --- | --- |
| 18.3 s | 248 ms | one larger, slower stroke (fingertips swung to v +0.56, past the strings); nothing was missed | player |
| 21.4 s | 326 ms | **camera/vision stall of 198 ms** (6 frames missing); the hand went up and back unseen | app |
| 27.9 s | 271 ms | a 100 ms pause at the bottom, then the up-stroke crossed during **one frame where the hand landmarker lost the hand**; `StrumDetector` resets on a missing hand and `skip_first_frames_back = 1` ignores the next frame, so the stroke was lost | app (by design) |

Other measurements:

- Sweep completeness drops at speed: 17 of 139 strums hit all 6 strings, 43 hit 5, and
  79 hit 4 or fewer.
- 84 camera frames were dropped in the take (6%), mostly from 4 to 16 s, before the
  strumming started. Stalls of up to 356 ms were also seen in `latency-1`.
- At 30 fps, 8 strums/s is under 4 frames per strum, so one lost frame can hide a
  whole stroke.

Retake `recordings/burst-2-part2` (10-chime count-in) is **not a valid burst test**:
the hand moved at about 3.1 strokes/s (dominant oscillation 1.53 Hz, the same in raw and
filtered landmarks), in large full sweeps (v -0.35 to +0.5) with pauses at each end.
Best 10 s window: 33 strums, 0 drops. It does show that at that pace the app detected
79 strums for the 80 strokes the camera saw, and 71 of the 79 sounded every chord
string. One 231 ms frame gap occurred around 30 to 32 s. (A first attempt at burst-2
recorded 0 frames: the camera never delivered one and the app waited silently until
Ctrl-C. Follow-up: fail with a clear "no frames from camera" error instead.)

Possible fixes (not started):

1. Find the source of the vision stalls (time hand detection per frame; the landmarker
   is likely slower when it has to re-detect a hand).
2. Let `StrumDetector` bridge a single frame with no hand instead of resetting (WP2
   change plus a `Config` key, through the lead).
3. 60 fps capture, if the camera supports it (see latency).

## 5. Move while playing: FAIL (drift)

Take `recordings/move-1-part2`: 80.5 s after the lock, 30 fps, no video (the disk had
1.4 GB free). Script: 15 s still, 30 s slow tilts of 20 to 30 degrees and 10 to 20 cm
slides, 15 s freer movement, then hold still and re-lock.

`tools.acceptance --check move`: **tracked 99.8% of frames**, racket "moved 5.76 head
lengths, turned 181 deg". The flow tracker never reported the racket lost, but the
pose it reported is not a racket being tilted 20 to 30 degrees:

| Time | Tracked pose | Notes |
| --- | --- | --- |
| 0 to 9 s | about horizontal (-180 to -164 deg), head 223 to 384 px, jumping within seconds | 9 |
| 9 to 30 s | **rotates steadily by about 175 deg** to -5 deg; head shrinks to ~190 px | 21 |
| 30 to 60 s | -79 to -20 deg; head 108 to 227 px | 4 |
| 60 to 80 s | settles near -30 deg, ~180 px | 5 |

The tracked features most likely slid off the racket onto the arm, body or background,
taking the strings with them, which is why strums stopped sounding (22 strums in 80 s,
none with every chord string; strumming hand found in only 45% of frames). No re-lock
discontinuity appears in the recording, so drift could not be measured against fresh
clicks. There is no video, but **the player confirmed seeing the strings slide off the
racket head while moving it**, which matches the pose data.

This matches the tracker's documented limit (`vision/flow.py`): it "can drift during
fast moves and cannot tell the racket left the frame". With markers removed, nothing
anchors the pose to the real racket after the lock.

Follow-up investigation (a video take, `video-1`, replayed offline; no code changed):

- The flow tracker drifts on real footage too. By 20 s its outline sits above and to
  the left of the real head, and tilted.
- The background is a plain wall; what shows through the strings is the player's shirt
  and arm, which move independently of the racket.
- The head appears as a **tilted ellipse** (face turned partly from the camera), while
  the tracker builds a circle from the two clicks and `head_half_width = 0.5`. The
  outline it tracks therefore does not match the rim from the first frame, and the
  string overlay cannot sit on a tilted head even with perfect tracking (`RacketPose`
  has no way to express that shape; a contract change).
- A rim-and-shaft profile tracker, prototyped outside the repo, fixed drift on
  synthetic scenes but diverged on this footage, because it was seeded from the same
  circle. A next attempt would fit the actual rim ellipse at lock. Parked for now.

Possible fixes (not started, lead decision): fit and track the real rim ellipse;
detect drift (for example, head length or angle changing faster than a held racket can)
and prompt a re-lock; constrain the fit (scale and rotation rates); or re-anchor from
the hand positions on the grip.

## 3. Idle false triggers: CHECK (retake needed)

Take `recordings/idle-1-part2`: 115 s after the lock (5 s short of the 120 s the
check asks for), 30 fps, racket tracked 100%. `tools.acceptance --check idle` gives
**0 false strums, 0 notes, and 0 silent crossings**.

That passes the target but proves little:

- **The strumming fingertips never crossed a string line**, so the gate was never
  tested. They were mostly beyond the low-E side (v +0.3 to +1.2) for 0 to 60 s and
  100 to 115 s, and resting still at the throat end, between low E and A, for 60 to
  100 s.
- **The hand read as armed (pick grip) for 99.2% of strumming-hand frames**, with no
  grip flips. Any crossing in that state would have played. In `grip-test-3` the grip
  read open 9.5% of the time. So in practice the protection against idle false notes
  was keeping the hand off the strings, not the gate.

Retake plan: at least 2.5 min of holding the racket that includes non-strumming hand
movement over the head (relaxing the hand and moving it across the strings to
reposition, resting the palm on the head, lifting away), so the gate is exercised.

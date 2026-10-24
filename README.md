# Racket Guitar

Play a tennis racket like a guitar, using only a MacBook's camera (and optionally its
mic). Tape two printed markers to the racket, hold it like a guitar, and strum the real
strings: the app overlays six virtual strings on the head, picks the chord from where
your other hand grips the handle, and plays synthesized guitar with low latency.

The spec is [docs/PID.md](docs/PID.md); the architecture and repo rules are in
[CLAUDE.md](CLAUDE.md).

## Setup

You need macOS on Apple silicon, Python 3.12, [uv](https://docs.astral.sh/uv/), a
tennis racket, a printer and some tape.

```bash
uv sync
uv run racketguitar fetch-model                 # MediaPipe hand model, ~7.5 MB, into models/
uv run python -m tools.print_markers -o markers.pdf
```

Print `markers.pdf` at 100% scale (check the 100 mm ruler), cut out both markers with
their white border, and tape them flat to the racket, facing the camera:

- **TIP** (id 0) at the top of the head, on the frame or the strings near the tip
- **THROAT** (id 1) at the throat, where the head meets the shaft

## Play

```bash
uv run racketguitar
```

The first run starts calibration: show both markers, then hold your strumming hand in
a **pick grip** (thumb pad on the side of your curled index) over the strings, then
open it. The calibration is saved to `~/.racketguitar/calibration.toml`; run
`uv run racketguitar calibrate` to redo it.

- **Strum:** with a closed pick grip (or a fist), brush across the strings, down or
  up. An open hand never plays, so open it to lift away.
- **Chords:** your other hand's position on the handle picks the chord: C, G, Am, F
  from the head toward the butt. No hand on the handle plays open strings.
- **Mute:** rest your open strumming hand flat on the head.

Keys: `q` quit, `r` record, `c` calibrate, `g` toggle the pick-grip gate, `m` toggle
the mic gate, `l` re-lock (markerless), `d` debug view, `h` help.

### Without markers (experimental)

`uv run racketguitar --markerless` skips the printed markers. Click the racket four
times in the window: the top of the head, the throat (where the head meets the shaft),
where the grip starts, and the butt. The app then follows the racket by optical flow on
the head's rim (`d` shows the tracked points). The chord sections span the grip. It is less precise
than markers, can drift during fast moves, and cannot tell if you put the racket down:
press `l` and click again whenever the strings stop lining up.

Useful options: `--mic` (also require a contact sound from the mic, best with
headphones), `--record DIR`, `--no-audio`, `--config FILE`, `--probe-tone`.

Settings live in `~/.racketguitar/config.toml`; it only needs the keys you change:

```toml
[gate]
mic = true

[audio]
master_gain = 0.6
```

## Record and replay

`r` (or `--record DIR`) saves a session: poses, hands, marker detections, onsets, notes
and mic audio (a few MB a minute). Set `[replay] video = true` to also keep lossless
video for `--from-video` replays; it is about 0.8 GB a minute. Replay with no camera,
mic or speakers:

```bash
uv run racketguitar replay recordings/20261024-150000               # must match exactly
uv run racketguitar replay recordings/... --from-mic --from-video   # re-run detectors
uv run racketguitar replay recordings/... --config try.toml --wav out.wav
```

## Measuring latency

The target is strum-to-sound p50 under 40 ms and p95 under 60 ms.

1. Run `uv run racketguitar --probe-tone`. Each note becomes a short 1 kHz beep.
2. Record on a phone held next to the racket while you tap or brush a string 30 or
   more times, a beat apart.
3. Run `uv run python -m tools.latency_probe analyze take.wav`.

The app also logs its own detect-to-audio time. It prints the figures when you quit;
for a recorded session, run `uv run python -m tools.latency_probe session DIR`.

## Development

```bash
uv run pytest            # headless: no camera, mic, speakers or network
uv run pytest -m hardware
uv run ruff check . && uv run ruff format .
```

## Troubleshooting

- **"could not open camera"**: allow camera access for your terminal app in System
  Settings › Privacy & Security › Camera, then restart the terminal.
- **Markers flicker or are lost**: use more light, move the camera closer, or print
  larger markers (`--config` with `[markers] size_mm = 50`, then re-print).
- **Mic gate misses strums**: use headphones (speaker sound leaks into the mic), and
  check the calibration screen's mic warning.

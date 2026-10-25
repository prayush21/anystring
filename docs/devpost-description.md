# Devpost draft

**Tagline:** Turn any racket into a guitar with just your laptop camera.

## Inspiration

Every week, someone you know watches a reel or a TikTok and says they want to learn guitar.
Almost none of them do. The bottleneck is not talent. It is the friction between seeing
someone make music and feeling what it is like to make it yourself: a guitar to buy,
lessons to book, weeks of sore fingers before anything sounds like a song.

We wanted to shrink that to zero. Open your laptop, grab the nearest tennis or badminton
racket, turn on the camera, and play.

## What it does

Anystring turns a racket, a pan or a ruler into a playable guitar using only a MacBook's camera.
Nothing is attached to the racket and nothing is printed.

- **Strum** the racket's strings and hear a synthesized guitar.
- **Chords** come from where your other hand grips the handle (C, G, Am, F), so you
  learn chord changes by feel.
- **Learn a song:** load a chord sheet, say "let's play" and follow the scrolling
  lyrics and the highlighted next chord.
- **Mute** by resting your open hand on the strings, like on a real guitar.

It is a first step, not a replacement. Once you have the strumming in your hands, you
know what to do next: pick up a real guitar. We just get you there faster.

## How we built it

Python on macOS. The camera runs at 60 fps; MediaPipe tracks the hands, and a markerless
optical-flow tracker follows the racket after four clicks (head tip, throat, grip start,
handle end). Geometry turns the racket and hand positions into string crossings, a pick-grip
gate decides when a strum counts, and a Karplus-Strong synth renders each string in a
128-sample audio callback. Camera, vision, mic and audio run on separate threads, so
no slow step delays a note. Sessions can be recorded and replayed exactly, which let us
tune the grip thresholds on real takes. Over 860 headless tests cover it.

## Challenges

Tracking a plain racket with no markers (fast or large moves can still make the strings
drift, and you re-lock with one key), telling a real strum from a hand returning, and
latency: the app itself takes about 49 ms, but with the camera's delay we measured
about 160 ms from strum to sound, so it is playable but not yet instant. See
[docs/how-we-built-it.md](how-we-built-it.md) for the measurements.

## What's next

More chords and songs, other objects as guitars, and a hand-off to a real guitar:
showing finger positions once you outgrow the racket.

## Built with

Python, MediaPipe, OpenCV, NumPy, Karplus-Strong synthesis, ElevenLabs speech-to-text
(voice commands, off the play path), uv.

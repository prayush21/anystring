"""Feedback on a take (WP8): measure it from the play log, put it into words, speak it.

``log`` holds the plain per-frame record a take is judged from, ``analyze`` turns it into
numbers and ranked findings, ``phrase`` asks Claude to say them as a coach would (with a
fixed template when that is off or fails) and ``speak`` reads the words aloud through
ElevenLabs. The integration layer fills the log from the engine's snapshots; nothing
here runs on the play path.
"""

# Changelog

## 1.1.0 (November 2026)

Feedback from the first real-webcam session: building a sentence wasn't discoverable and
it wasn't obvious when a gesture had fired.

- Quick and sentence modes: in sentence mode, every `speak` gesture adds its words instead,
  so the default vocabulary can build sentences out of the box (OK sign switches modes)
- Live gestures panel showing what each gesture does in the current mode, with the held
  gesture's progress and the fired gesture highlighted (`b` to toggle)
- Toast for every action ("Said: Yes", "Added: water", "Sentence is empty") plus a coloured
  border flash
- Sentence bar shows words as they're added, highlights the newest, and explains how to start
  when empty
- Unbound recorded gestures now say so instead of doing nothing silently
- Keyboard shortcuts (`m`, `p`, `c`) go through the same action path and give the same feedback

## 1.0.0 (November 2026)

Ground-up rebuild of the original prototype as a finished tool.

- MediaPipe Tasks Hand Landmarker (VIDEO mode) with automatic model download
- Position-, scale-, tilt- and handedness-invariant feature vectors with an orientation term
- 10 built-in gestures that work with no setup
- Record-your-own gestures with a distance-gated k-NN (no training step)
- Hold-to-fire stabiliser with progress ring, fire-once and cooldown
- Sentence builder (add / undo / clear / speak)
- Offline text-to-speech on a background thread; typing and hotkey actions
- Pause gesture
- YAML config with clear validation errors
- CLI: `run`, `record`, `gestures`, `forget`, `init`, `doctor`
- 126 tests using synthetic hands; CI on Windows, macOS and Linux

## 0.1.0 (February 2026)

Original proof of concept: MediaPipe hand tracking mapped to a handful of hard-coded
actions. Began as an identification experiment for a defence context, then pivoted to
helping people who are mute or have throat damage communicate through gestures.

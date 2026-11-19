# Architecture

CogniSign is a pipeline of small, independently tested stages. Only `tracker.py`, `hud.py`
and the camera generator in `app.py` touch hardware or a window; everything else is pure
Python + NumPy and runs headless in CI.

## Data flow per frame

| Stage | Module | Input | Output |
|---|---|---|---|
| Capture | `app.camera_frames` | webcam | BGR frame (mirrored) |
| Track | `tracker.HandTracker` | frame, timestamp | `Hand` (21 landmarks, handedness, score) |
| Recognise | `classifier.Recogniser` | landmarks | `Prediction(label, confidence, source)` |
| Stabilise | `stabiliser.GestureStabiliser` | label per frame | `GestureEvent` or nothing |
| Act | `actions.ActionDispatcher` | `Action` | speech / keys / sentence change |
| Show | `hud.draw` | frame + `HudState` | annotated frame |

`app.GestureEngine` wires recognise → stabilise → act → HUD state and is what the end-to-end
tests drive with scripted poses.

## Feature vector (v2)

65 values:

- **63 shape values**: 21 × (x, y, z) after
  - translating the wrist to the origin,
  - dividing by palm length (wrist → middle-finger MCP),
  - mirroring left hands on the x axis,
  - rotating in the image plane so wrist → middle MCP points straight up.
- **2 orientation values**: the original wrist → middle MCP unit vector × 1.5.

Shape makes recognition invariant to distance, position and tilt. The orientation term puts
back the information that matters for gestures that differ only by direction (thumbs up vs
down), weighted so a natural 15–20° tilt doesn't move a sample far.

MediaPipe normalises x by frame width and y by height. `tracker.to_hands` multiplies x (and z)
by width/height first, otherwise a 16:9 camera would squash every hand horizontally.

## Recognition

1. **k-NN over your recordings** (k = 5). If the nearest sample is further than
   `max_distance`, the pose is rejected outright. Confidence blends the vote share with how
   close the nearest sample is; below `min_confidence` it falls through.
2. **Presets.** A finger counts as extended when its tip is ≥ 12% further from the wrist than
   its PIP joint (distance-only, so tilt-proof). The thumb is extended when its tip is further
   from the pinky knuckle than the thumb MCP is. Pinch distance and thumb direction separate
   OK / thumbs up / thumbs down.

Custom first, presets second: a person can redefine any built-in gesture for their own hand.

## Stabiliser state machine

```
idle --(label seen)--> holding --(held >= hold_ms, armed, cooldown ok)--> FIRE (disarmed)
holding --(different label / none for > grace_ms)--> holding new label (re-armed)
disarmed --(pose changes)--> re-armed
```

`grace_ms` (150 ms) absorbs single-frame tracking dropouts so the hold ring doesn't reset.

## Modes and feedback

`ActionDispatcher.effective()` translates actions for the current mode (in sentence mode a
`speak` becomes a `phrase_add`). The same translation feeds `describe()`, so the bindings
panel always shows what a gesture will actually do. Every dispatch returns a human-readable
`message` and a `kind` (said / added / info / warn) that the engine turns into a toast and a
border flash.

## Threads

Text-to-speech engines block while speaking, so `Pyttsx3Speaker` owns its engine on a
background thread fed by a queue. The camera loop never waits on audio.

## Storage

`~/.cognisign/` (or `$COGNISIGN_HOME`):

- `config.yaml`: settings and gesture → action map
- `gestures.json`: recorded feature vectors, versioned; written atomically (temp file + rename)
- `models/hand_landmarker.task`: MediaPipe model, downloaded once

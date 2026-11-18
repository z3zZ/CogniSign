"""Command line: `cognisign run | record | gestures | forget | init | doctor`."""

from __future__ import annotations

import argparse
import logging
import sys

from . import __version__
from .actions import ACTION_TYPES, ActionDispatcher, make_keyboard, make_speaker
from .classifier import GestureLibrary, LibraryError
from .config import (
    ConfigError,
    config_path,
    gestures_path,
    load_config,
    write_default_config,
)
from .presets import PRESET_GESTURES


def _tracker(cfg):
    from .tracker import HandTracker

    t = cfg.tracking
    return HandTracker(num_hands=t.num_hands,
                       min_detection_confidence=t.min_detection_confidence,
                       min_tracking_confidence=t.min_tracking_confidence)


def cmd_run(args) -> int:
    from .app import GestureEngine, camera_frames, run

    cfg = load_config(args.config)
    if args.camera is not None:
        cfg.camera.index = args.camera
    speaker = make_speaker(cfg.voice.rate, cfg.voice.voice)
    dispatcher = ActionDispatcher(speaker=speaker,
                                  keyboard=None if args.no_keyboard else make_keyboard())
    engine = GestureEngine(cfg, dispatcher)
    print(f"CogniSign {__version__}: {len(cfg.gestures)} gestures mapped, "
          f"{len(engine.recogniser.library.labels)} custom. Press q to quit.")
    try:
        with _tracker(cfg) as tracker:
            frames = camera_frames(cfg.camera.index, cfg.camera.width, cfg.camera.height,
                                   cfg.camera.mirror)
            run(cfg, engine, tracker, frames, show=not args.headless)
    finally:
        dispatcher.close()
    return 0


def cmd_record(args) -> int:
    from .app import camera_frames
    from .recorder import record

    cfg = load_config(args.config)
    lib = GestureLibrary.load(gestures_path())
    with _tracker(cfg) as tracker:
        frames = camera_frames(args.camera if args.camera is not None else cfg.camera.index,
                               cfg.camera.width, cfg.camera.height, cfg.camera.mirror)
        samples = record(args.label, tracker, frames, samples=args.samples)
    if len(samples) < max(5, args.samples // 4):
        print(f"Only captured {len(samples)} samples, not saving. Try again with your "
              "hand clearly in view.")
        return 1
    if args.replace:
        lib.remove(args.label)
    for s in samples:
        lib.add(args.label, s)
    lib.save(gestures_path())
    print(f"Saved {len(samples)} samples of '{args.label}' "
          f"({lib.counts()[args.label]} total).")
    if args.label not in load_config(args.config).gestures:
        print(f"Next: map it in {config_path()}, e.g.\n"
              f"  {args.label}: {{ action: phrase_add, value: \"{args.label}\" }}")
    return 0


def cmd_gestures(args) -> int:
    cfg = load_config(args.config)
    lib = GestureLibrary.load(gestures_path())
    print("Built-in gestures:")
    for name, desc in PRESET_GESTURES.items():
        act = cfg.gestures.get(name)
        print(f"  {name:<12} {desc:<44} -> {act.describe() if act else '(unmapped)'}")
    print("\nYour recorded gestures:")
    if not lib.labels:
        print("  none yet. Try: cognisign record water")
    for name, n in lib.counts().items():
        act = cfg.gestures.get(name)
        print(f"  {name:<12} {n:>3} samples{'':<32} -> {act.describe() if act else '(unmapped)'}")
    print("\nAction types: " + ", ".join(ACTION_TYPES))
    return 0


def cmd_forget(args) -> int:
    lib = GestureLibrary.load(gestures_path())
    if not lib.remove(args.label):
        print(f"No recorded gesture called '{args.label}'.")
        return 1
    lib.save(gestures_path())
    print(f"Forgot '{args.label}'.")
    return 0


def cmd_init(args) -> int:
    existed = config_path().exists()
    path = write_default_config(overwrite=args.force)
    if existed and not args.force:
        print(f"Config already exists at {path} (use --force to reset it).")
    else:
        print(f"Wrote config to {path}")
    return 0


def cmd_doctor(args) -> int:
    ok = True

    def check(name, fn):
        nonlocal ok
        try:
            detail = fn()
            print(f"  [ok]   {name}{': ' + detail if detail else ''}")
        except Exception as exc:  # noqa: BLE001 - doctor reports everything
            ok = False
            print(f"  [fail] {name}: {exc}")

    def _config():
        cfg = load_config(args.config)
        return f"{len(cfg.gestures)} gestures mapped"

    def _camera():
        import cv2

        cap = cv2.VideoCapture(args.camera or 0)
        good = cap.isOpened() and cap.read()[0]
        cap.release()
        if not good:
            raise RuntimeError("no frames from camera")
        return "frames OK"

    def _model():
        from .tracker import ensure_model
        return str(ensure_model())

    def _mediapipe():
        import mediapipe
        return mediapipe.__version__

    def _tts():
        import pyttsx3
        pyttsx3.init()
        return "offline voice available"

    def _library():
        return f"{len(GestureLibrary.load(gestures_path()).labels)} custom gestures"

    print("CogniSign doctor")
    for name, fn in [("config", _config), ("mediapipe", _mediapipe), ("hand model", _model),
                     ("camera", _camera), ("text-to-speech", _tts), ("gesture library", _library)]:
        check(name, fn)
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cognisign",
        description="Turn hand gestures into speech and actions with any webcam.")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    p.add_argument("--config", help="path to a config.yaml (default ~/.cognisign/config.yaml)")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command")

    r = sub.add_parser("run", help="start recognising gestures (default)")
    r.add_argument("--camera", type=int)
    r.add_argument("--headless", action="store_true", help="no preview window")
    r.add_argument("--no-keyboard", action="store_true", help="disable type/hotkey actions")
    r.set_defaults(func=cmd_run)

    rec = sub.add_parser("record", help="teach a new gesture from your own hand")
    rec.add_argument("label")
    rec.add_argument("--samples", type=int, default=40)
    rec.add_argument("--camera", type=int)
    rec.add_argument("--replace", action="store_true", help="overwrite existing samples")
    rec.set_defaults(func=cmd_record)

    g = sub.add_parser("gestures", help="list gestures and what they do")
    g.set_defaults(func=cmd_gestures)

    f = sub.add_parser("forget", help="delete a recorded gesture")
    f.add_argument("label")
    f.set_defaults(func=cmd_forget)

    i = sub.add_parser("init", help="write an editable config file")
    i.add_argument("--force", action="store_true")
    i.set_defaults(func=cmd_init)

    d = sub.add_parser("doctor", help="check camera, model and voice are working")
    d.add_argument("--camera", type=int)
    d.set_defaults(func=cmd_doctor)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    if not getattr(args, "func", None):  # bare `cognisign` means `cognisign run`
        args.func, args.camera, args.headless, args.no_keyboard = cmd_run, None, False, False
    try:
        return args.func(args)
    except (ConfigError, LibraryError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

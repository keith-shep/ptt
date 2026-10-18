#!/usr/bin/env python3
"""
Push-to-talk dictation with local Whisper.

Hold Right Option, speak, release. The transcript is pasted wherever your
cursor is (a chat box, a terminal, an editor). Everything runs on your Mac:
no audio leaves the machine.

Usage (inside your whisper-env):
    python ptt.py                    # small.en: fast, good for clear speech
    python ptt.py --model turbo      # more accurate, slower
    python ptt.py --key cmd_r        # use Right Command instead
    python ptt.py --prompt "Claude, Python, API, git, pytest"   # vocabulary hint

Quit with Ctrl+C in the terminal running it.
"""
import argparse
import os
import signal
import subprocess
import sys
import threading
import time

import numpy as np
import Quartz  # installed with pynput on macOS
import sounddevice as sd
import whisper
from pynput import keyboard

SAMPLE_RATE = 16000      # Whisper expects 16 kHz mono
MIN_SECONDS = 0.3        # ignore accidental taps
POLL_SECONDS = 0.03      # how often to check the key while recording
SILENCE_LEVEL = 0.01     # skip clips with no real sound (avoids Whisper inventing text)

parser = argparse.ArgumentParser(description="Push-to-talk dictation with local Whisper")
parser.add_argument("--model", default="small.en",
                    help="Whisper model: tiny.en, base.en, small.en, medium.en, turbo, large")
parser.add_argument("--key", default="alt_r",
                    help="Hold-to-talk key: alt_r (Right Option), cmd_r, ctrl_r, shift_r, f13...")
parser.add_argument("--prompt", default="",
                    help="Words Whisper should expect, e.g. names or technical terms")
args = parser.parse_args()

try:
    HOTKEY = getattr(keyboard.Key, args.key)
except AttributeError:
    raise SystemExit(f"Unknown key '{args.key}'. Try alt_r, cmd_r, ctrl_r, shift_r or f13.")

print(f"Loading Whisper model '{args.model}' (first time downloads it)...")
model = whisper.load_model(args.model)

kb = keyboard.Controller()
transcribe_lock = threading.Lock()
hotkey_pressed = threading.Event()
frames = []
stream = None


def audio_callback(indata, frame_count, time_info, status):
    frames.append(indata.copy())


def start_recording():
    global stream, frames
    frames = []
    # The mic is only open while the key is held.
    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                            dtype="float32", callback=audio_callback)
    stream.start()
    print("● Listening...", flush=True)


def stop_recording():
    global stream
    stream.stop()
    stream.close()
    stream = None

    if not frames:
        print("  (no audio captured: check Terminal has Microphone permission,"
              " and hold the key a little longer)")
        return
    audio = np.concatenate(frames).flatten()
    if len(audio) < SAMPLE_RATE * MIN_SECONDS:
        print("  (too short, skipped)")
        return
    if np.abs(audio).max() < SILENCE_LEVEL:
        print("  (silence, skipped)")
        return

    print(f"  transcribing {len(audio) / SAMPLE_RATE:.1f}s of audio...", flush=True)
    threading.Thread(target=transcribe_and_paste, args=(audio,), daemon=True).start()


def transcribe_and_paste(audio):
    with transcribe_lock:  # one transcription at a time, in order
        started = time.time()
        result = model.transcribe(
            audio,
            language="en",
            fp16=False,                       # CPU on Mac
            initial_prompt=args.prompt or None,
            condition_on_previous_text=False,
        )
        text = result["text"].strip()
        if not text:
            print("  (nothing recognised)")
            return
        print(f"  → {text}  [{time.time() - started:.1f}s]")
        paste(text + " ")


def paste(text):
    # Keep whatever was on the clipboard and put it back afterwards.
    previous = subprocess.run(["pbpaste"], capture_output=True).stdout
    subprocess.run(["pbcopy"], input=text.encode("utf-8"))
    time.sleep(0.05)
    with kb.pressed(keyboard.Key.cmd):
        kb.press("v")
        kb.release("v")
    time.sleep(0.3)
    subprocess.run(["pbcopy"], input=previous)


def hotkey_held():
    # Ask macOS for the physical key state rather than trusting the listener
    # to deliver the release: if that one event is lost, we would otherwise
    # sit at "Listening..." forever.
    return Quartz.CGEventSourceKeyState(
        Quartz.kCGEventSourceStateHIDSystemState, HOTKEY.value.vk)


def recorder():
    while True:
        hotkey_pressed.wait()
        try:
            start_recording()
            while hotkey_held():
                time.sleep(POLL_SECONDS)
            stop_recording()
        except Exception as e:
            print(f"  (recording failed: {e})", flush=True)
        hotkey_pressed.clear()


# Listener callbacks run inside macOS's keyboard event tap and must return
# at once. Opening or closing the mic there (slow with Bluetooth headsets, or
# while Whisper has the CPU busy) can get the tap disabled by macOS, and
# pynput never turns it back on, so the release is never seen.
def on_press(key):
    if key == HOTKEY:
        hotkey_pressed.set()


def shutdown():
    # A second Ctrl+C during cleanup quits immediately.
    signal.signal(signal.SIGINT, lambda *_: os._exit(130))
    listener.stop()
    if stream is not None:
        try:
            stream.abort()  # don't leave the mic open
            stream.close()
        except Exception:
            pass
    print("\nBye.", flush=True)
    sys.stderr.flush()
    # Skip normal interpreter teardown: PortAudio, pynput's run loop and
    # torch worker threads can block it, leaving the process hung after Ctrl+C.
    os._exit(0)


if __name__ == "__main__":
    print(f"Ready. Hold {args.key} to talk, release to paste. Ctrl+C to quit.")
    threading.Thread(target=recorder, daemon=True).start()
    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    try:
        while listener.is_alive():
            listener.join(0.5)  # short timeouts keep Ctrl+C responsive
    except KeyboardInterrupt:
        pass
    shutdown()

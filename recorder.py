"""
Meeting Audio Recorder
Records microphone + system audio (WASAPI loopback) simultaneously and saves as MP3.
Works with native Windows Teams app, Chrome browser Teams, or any audio source.

Requirements:
    pip install -r requirements.txt
    ffmpeg must be installed and on PATH (winget install ffmpeg)

Usage:
    python recorder.py
    Press ENTER to start, ENTER again to stop.
"""

import os
import sys
import threading
import time
import wave
import struct
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import noisereduce as nr
import pyaudiowpatch as pyaudio
from pydub import AudioSegment
import imageio_ffmpeg
AudioSegment.converter = imageio_ffmpeg.get_ffmpeg_exe()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SAMPLE_RATE = 44100
CHANNELS = 2
SAMPLE_WIDTH = 2          # 16-bit
CHUNK = 1024
RECORDINGS_DIR = Path(__file__).parent / "recordings"
MP3_BITRATE = "192k"


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------

def find_loopback_device(pa: pyaudio.PyAudio):
    """Return the default WASAPI loopback device (captures all system audio)."""
    try:
        wasapi_info = pa.get_host_api_info_by_type(pyaudio.paWASAPI)
    except OSError:
        print("[ERROR] WASAPI not available. This script requires Windows with WASAPI.")
        sys.exit(1)

    default_speakers_index = wasapi_info["defaultOutputDevice"]
    default_speakers = pa.get_device_info_by_index(default_speakers_index)

    # Find the loopback version of the default output device
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if info.get("isLoopbackDevice") and info["name"] == default_speakers["name"]:
            return info

    # Fallback: return any loopback device
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if info.get("isLoopbackDevice"):
            return info

    print("[ERROR] No WASAPI loopback device found.")
    print("        Make sure audio is playing or a playback device is active.")
    sys.exit(1)


def find_microphone_device(pa: pyaudio.PyAudio):
    """Return the default microphone input device."""
    try:
        default_input_index = pa.get_default_input_device_info()["index"]
        return pa.get_device_info_by_index(default_input_index)
    except OSError:
        # Fallback: find first input device
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            if info["maxInputChannels"] > 0 and not info.get("isLoopbackDevice"):
                return info
    print("[ERROR] No microphone/input device found.")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Recording threads
# ---------------------------------------------------------------------------

class AudioCapture(threading.Thread):
    """Captures audio from a single device into a buffer list."""

    def __init__(self, pa: pyaudio.PyAudio, device_info: dict, stop_event: threading.Event):
        super().__init__(daemon=True)
        self.pa = pa
        self.device_info = device_info
        self.stop_event = stop_event
        self.frames: list[bytes] = []
        self.sample_rate = int(device_info["defaultSampleRate"])
        self.channels = min(int(device_info["maxInputChannels"]), 2) or 2
        self.error: Exception | None = None

    def run(self):
        try:
            stream = self.pa.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.device_info["index"],
                frames_per_buffer=CHUNK,
            )
            while not self.stop_event.is_set():
                try:
                    data = stream.read(CHUNK, exception_on_overflow=False)
                    self.frames.append(data)
                except OSError:
                    break
            stream.stop_stream()
            stream.close()
        except Exception as exc:
            self.error = exc


# ---------------------------------------------------------------------------
# Audio mixing
# ---------------------------------------------------------------------------

def frames_to_numpy(frames: list[bytes], channels: int, sample_rate: int) -> np.ndarray:
    """Convert raw PCM bytes to a float32 numpy array (samples x channels)."""
    raw = b"".join(frames)
    arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    if channels == 1:
        arr = np.column_stack([arr, arr])   # mono -> stereo
    else:
        arr = arr.reshape(-1, channels)
        if channels > 2:
            arr = arr[:, :2]                # keep first two channels
    return arr


def resample_to(arr: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    """Simple linear resample to match target sample rate."""
    if from_rate == to_rate:
        return arr
    ratio = to_rate / from_rate
    new_length = int(len(arr) * ratio)
    indices = np.linspace(0, len(arr) - 1, new_length)
    left  = np.interp(indices, np.arange(len(arr)), arr[:, 0])
    right = np.interp(indices, np.arange(len(arr)), arr[:, 1])
    return np.column_stack([left, right])


def mix_audio(mic_capture: AudioCapture, sys_capture: AudioCapture) -> np.ndarray:
    """Mix microphone and system audio into a single stereo float32 array."""
    mic_arr = frames_to_numpy(mic_capture.frames, mic_capture.channels, mic_capture.sample_rate)
    sys_arr = frames_to_numpy(sys_capture.frames, sys_capture.channels, sys_capture.sample_rate)

    mic_arr = resample_to(mic_arr, mic_capture.sample_rate, SAMPLE_RATE)
    sys_arr = resample_to(sys_arr, sys_capture.sample_rate, SAMPLE_RATE)

    # Apply noise reduction to mic using loopback as echo reference.
    # This suppresses acoustic echo caused by laptop speakers bleeding into the mic.
    if len(mic_arr) > 0 and len(sys_arr) > 0:
        ref_len = min(len(mic_arr), len(sys_arr))
        print("  Applying echo reduction to microphone track...")
        mic_clean_L = nr.reduce_noise(
            y=mic_arr[:ref_len, 0], sr=SAMPLE_RATE,
            y_noise=sys_arr[:ref_len, 0], stationary=False, prop_decrease=0.8,
        )
        mic_clean_R = nr.reduce_noise(
            y=mic_arr[:ref_len, 1], sr=SAMPLE_RATE,
            y_noise=sys_arr[:ref_len, 1], stationary=False, prop_decrease=0.8,
        )
        mic_arr = np.column_stack([mic_clean_L, mic_clean_R]).astype(np.float32)

    # Pad shorter array to match length
    max_len = max(len(mic_arr), len(sys_arr))
    if len(mic_arr) < max_len:
        mic_arr = np.pad(mic_arr, ((0, max_len - len(mic_arr)), (0, 0)))
    if len(sys_arr) < max_len:
        sys_arr = np.pad(sys_arr, ((0, max_len - len(sys_arr)), (0, 0)))

    mixed = mic_arr + sys_arr

    # Normalize to prevent clipping
    peak = np.max(np.abs(mixed))
    if peak > 0:
        mixed = mixed / peak * 32000.0

    return mixed.astype(np.int16)


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def save_as_mp3(audio: np.ndarray, output_path: Path):
    """Write numpy int16 stereo array to an MP3 file via pydub."""
    raw_bytes = audio.tobytes()
    segment = AudioSegment(
        data=raw_bytes,
        sample_width=SAMPLE_WIDTH,
        frame_rate=SAMPLE_RATE,
        channels=CHANNELS,
    )
    segment.export(str(output_path), format="mp3", bitrate=MP3_BITRATE)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  Meeting Audio Recorder")
    print("=" * 60)

    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)

    pa = pyaudio.PyAudio()

    loopback_dev = find_loopback_device(pa)
    mic_dev      = find_microphone_device(pa)

    print(f"\n  Microphone  : {mic_dev['name']}")
    print(f"  System audio: {loopback_dev['name']} (loopback)")
    print(f"  Output dir  : {RECORDINGS_DIR.resolve()}")
    print()

    try:
        while True:
            input("  Press ENTER to start recording...")
            stop_event = threading.Event()

            mic_capture = AudioCapture(pa, mic_dev, stop_event)
            sys_capture = AudioCapture(pa, loopback_dev, stop_event)

            start_time = time.time()
            mic_capture.start()
            sys_capture.start()

            print("  [REC] Recording... Press ENTER to stop.")

            # Run a timer in background while waiting for ENTER
            stop_timer = threading.Event()

            def show_timer():
                while not stop_timer.is_set():
                    elapsed = int(time.time() - start_time)
                    mins, secs = divmod(elapsed, 60)
                    print(f"\r  [REC] {mins:02d}:{secs:02d}", end="", flush=True)
                    time.sleep(1)

            timer_thread = threading.Thread(target=show_timer, daemon=True)
            timer_thread.start()

            input()   # wait for ENTER to stop

            stop_event.set()
            stop_timer.set()
            mic_capture.join(timeout=3)
            sys_capture.join(timeout=3)

            elapsed = time.time() - start_time
            print(f"\n  Stopped. Recorded {elapsed:.1f}s")

            if mic_capture.error:
                print(f"  [WARN] Microphone error: {mic_capture.error}")
            if sys_capture.error:
                print(f"  [WARN] System audio error: {sys_capture.error}")

            if not mic_capture.frames and not sys_capture.frames:
                print("  [WARN] No audio captured — skipping save.")
                continue

            print("  Mixing and encoding to MP3...")
            audio = mix_audio(mic_capture, sys_capture)

            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            output_path = RECORDINGS_DIR / f"meeting_{timestamp}.mp3"

            try:
                save_as_mp3(audio, output_path)
                size_kb = output_path.stat().st_size // 1024
                print(f"  Saved: {output_path.name}  ({size_kb} KB)")
            except Exception as exc:
                print(f"  [ERROR] Failed to save MP3: {exc}")
                print("  Make sure imageio-ffmpeg is installed: pip install imageio-ffmpeg")

            print()
            again = input("  Record another? [y/N]: ").strip().lower()
            if again != "y":
                break

    except KeyboardInterrupt:
        print("\n  Interrupted.")
    finally:
        pa.terminate()
        print("  Done.")


if __name__ == "__main__":
    main()

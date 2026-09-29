#!/usr/bin/env python3
"""
drumkit.py — Low-latency MIDI drum sampler for DrumGizmo kits
Supports Linux/ALSA and Windows (WinMM via python-rtmidi).
"""

import sys
import os
import json
import time
import threading
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import deque

import numpy as np
#https://python-sounddevice.readthedocs.io/en/latest/installation.html
# Set environment variable before importing sounddevice. Value is not important.
os.environ["SD_ENABLE_ASIO"] = "1"

import sounddevice as sd

print(sd.query_hostapis())
import soundfile as sf
import rtmidi

# ─── Platform detection ────────────────────────────────────────────────────────

IS_WINDOWS = sys.platform == "win32"

if IS_WINDOWS:
    import msvcrt
else:
    import tty
    import termios
    import select

# ─── Configuration ────────────────────────────────────────────────────────────

CONFIG_DIR   = Path.home() / ".config" / "drumkit"
SAMPLERATE   = 48000
BLOCKSIZE    = 128          # ~2.7 ms per block at 48 kHz
CHANNELS     = 2
MAX_VOICES   = 64           # maximum simultaneous sounds
VOLUME_STEP  = 0.1


# ─── Cross-platform terminal helpers ─────────────────────────────────────────

def _save_term():
    """Save terminal state (Unix only). Returns opaque state or None."""
    if IS_WINDOWS:
        return None
    return termios.tcgetattr(sys.stdin)


def _set_cbreak():
    """Switch stdin to cbreak/raw mode for single-keypress reads (Unix only)."""
    if IS_WINDOWS:
        return  # msvcrt already reads key-by-key without echo
    tty.setcbreak(sys.stdin.fileno())


def _restore_term(old_settings):
    """Restore terminal state saved by _save_term()."""
    if IS_WINDOWS or old_settings is None:
        return
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)


def _read_key() -> str | None:
    """
    Non-blocking single-character read from the keyboard.
    Returns the character, or None if no key is currently pressed.
    """
    if IS_WINDOWS:
        if msvcrt.kbhit():
            ch = msvcrt.getwch()
            # getwch returns a 2-char sentinel for special/function keys
            if ch in ("\x00", "\xe0"):
                msvcrt.getwch()  # consume the second byte and ignore
                return None
            return ch
    else:
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1)
    return None


# ─── XML parsing ──────────────────────────────────────────────────────────────

def parse_kit(xml_path: Path) -> list[dict]:
    """
    Parse the main drumkit XML.
    Returns list of dicts: {name, file, main_channels}
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    instruments = []
    for inst in root.find("instruments").findall("instrument"):
        name = inst.get("name")
        file = inst.get("file")
        mains = [
            cm.get("in")
            for cm in inst.findall("channelmap")
            if cm.get("main") == "true"
        ]
        instruments.append({"name": name, "file": file, "main_channels": mains})
    return instruments


def parse_instrument_samples(
    inst_xml: Path, main_channels: list[str]
) -> list[tuple[float, np.ndarray, str]]:
    """
    Parse a DrumGizmo instrument XML.
    Returns a list of (power, stereo_float32_array, sample_name) sorted by power ascending.
    Only loads WAV channels needed for stereo output (main channels).
    """
    if not inst_xml.exists():
        print(f"    [warn] not found: {inst_xml}", file=sys.stderr)
        return []

    try:
        tree = ET.parse(inst_xml)
    except ET.ParseError as e:
        print(f"    [warn] XML error in {inst_xml}: {e}", file=sys.stderr)
        return []

    root = tree.getroot()
    samples_elem = root.find("samples")
    if samples_elem is None:
        return []

    base_dir = inst_xml.parent

    # Pick stereo pair from main channels
    left_ch  = main_channels[0] if len(main_channels) > 0 else None
    right_ch = main_channels[1] if len(main_channels) > 1 else left_ch

    # Cache to avoid loading the same WAV file twice
    _wav_cache: dict[tuple, np.ndarray] = {}

    def load_channel_from_wav(fpath: Path, fch_0idx: int) -> np.ndarray | None:
        key = (str(fpath), fch_0idx)
        if key in _wav_cache:
            return _wav_cache[key]
        if not fpath.exists():
            return None
        try:
            data, sr = sf.read(str(fpath), dtype="float32", always_2d=True)
            # Resample if needed (rare — most kits match the declared samplerate)
            if sr != SAMPLERATE:
                factor = SAMPLERATE / sr
                new_len = int(len(data) * factor)
                import scipy.signal
                data = scipy.signal.resample(data, new_len, axis=0).astype(np.float32)
            col = data[:, fch_0idx] if fch_0idx < data.shape[1] else data[:, 0]
            _wav_cache[key] = col
            return col
        except Exception as e:
            print(f"    [warn] cannot load {fpath}: {e}", file=sys.stderr)
            return None

    layers = []

    for sample in samples_elem.findall("sample"):
        power = float(sample.get("power", 0.5))
        sample_name = sample.get("name", "unknown")

        # Map channel name → (wav_path, 0-indexed file channel)
        ch_map: dict[str, tuple[Path, int]] = {}
        for af in sample.findall("audiofile"):
            ch = af.get("channel")
            fpath = base_dir / af.get("file")
            fch = int(af.get("filechannel", 1)) - 1  # convert to 0-indexed
            ch_map[ch] = (fpath, fch)

        if not ch_map:
            continue

        # Resolve which channel names to load
        l_ch = left_ch  if (left_ch  and left_ch  in ch_map) else next(iter(ch_map))
        r_ch = right_ch if (right_ch and right_ch in ch_map) else l_ch

        left_data  = load_channel_from_wav(*ch_map[l_ch])
        if left_data is None:
            continue

        right_data = (
            load_channel_from_wav(*ch_map[r_ch])
            if r_ch != l_ch
            else left_data
        )
        if right_data is None:
            right_data = left_data

        # Zero-pad to equal length and interleave into (N, 2) float32
        n = max(len(left_data), len(right_data))
        if len(left_data) < n:
            left_data  = np.pad(left_data,  (0, n - len(left_data)))
        if len(right_data) < n:
            right_data = np.pad(right_data, (0, n - len(right_data)))

        stereo = np.ascontiguousarray(
            np.column_stack([left_data, right_data]), dtype=np.float32
        )
        layers.append((power, stereo, sample_name))

    layers.sort(key=lambda x: x[0])
    return layers


# ─── Audio engine ─────────────────────────────────────────────────────────────

class AudioEngine:
    """
    sounddevice callback-based stereo mixer.
    New voices are enqueued via a deque (GIL-safe, lock-free hot path).
    Volume can exceed 1.0 freely.
    """

    def __init__(self, blocksize: int = BLOCKSIZE):
        self.volume: float = 1.0
        self._trigger_queue: deque = deque()   # producer: MIDI thread
        self._voices: list = []                # consumer: audio callback only
        self._blocksize = blocksize
        self._stream = sd.OutputStream(
            samplerate=SAMPLERATE,
            channels=CHANNELS,
            dtype="float32",
            blocksize=blocksize,
            callback=self._callback,
            latency="low",
            device = 'ASIO4ALL v2, ASIO',
        )

    def start(self):
        self._stream.start()

    def stop(self):
        self._stream.stop()
        self._stream.close()

    def play(self, stereo_array: np.ndarray, gain: float = 1.0):
        """Non-blocking trigger — called from MIDI thread."""
        self._trigger_queue.append((stereo_array, gain))

    def _callback(self, outdata: np.ndarray, frames: int, time_info, status):
        # Pull in any new voices from the MIDI thread
        # Each entry: [arr, position, per_voice_gain]
        while self._trigger_queue:
            try:
                arr, gain = self._trigger_queue.popleft()
                if len(self._voices) < MAX_VOICES:
                    self._voices.append([arr, 0, gain])
                elif self._voices:          # drop oldest to make room
                    self._voices[0] = [arr, 0, gain]
            except IndexError:
                break

        # Mix all active voices
        out = np.zeros((frames, CHANNELS), dtype=np.float32)
        finished = []
        for i, voice in enumerate(self._voices):
            arr, pos, gain = voice
            remaining = len(arr) - pos
            if remaining <= 0:
                finished.append(i)
                continue
            n = min(frames, remaining)
            if gain != 1.0:
                out[:n] += arr[pos : pos + n] * gain
            else:
                out[:n] += arr[pos : pos + n]
            voice[1] += n
            if voice[1] >= len(arr):
                finished.append(i)

        for i in reversed(finished):
            self._voices.pop(i)

        # Apply global volume (in-place, no extra allocation)
        if self.volume != 1.0:
            out *= self.volume

        outdata[:] = out


# ─── MIDI utilities ───────────────────────────────────────────────────────────

def list_midi_ports() -> list[str]:
    midi_in = rtmidi.MidiIn()
    ports = [midi_in.get_port_name(i) for i in range(midi_in.get_port_count())]
    del midi_in
    return ports


def select_midi_port() -> int:
    """Return the port index to use, prompting user if needed."""
    ports = list_midi_ports()
    if not ports:
        print("No MIDI input devices found.")
        if IS_WINDOWS:
            print("Make sure your device is connected and Windows sees it (Device Manager).")
        else:
            print("Make sure your device is connected and ALSA sees it (aconnect -i).")
        sys.exit(1)
    if len(ports) == 1:
        print(f"MIDI: {ports[0]}")
        return 0
    print("\nAvailable MIDI input devices:")
    for i, p in enumerate(ports):
        print(f"  [{i}] {p}")
    while True:
        try:
            raw = input("Select device number: ").strip()
            idx = int(raw)
            if 0 <= idx < len(ports):
                print(f"Using: {ports[idx]}")
                return idx
        except (ValueError, EOFError):
            pass
        print("  Invalid selection, try again.")


# ─── Mapping persistence ──────────────────────────────────────────────────────

def _safe_filename(kit_xml: Path) -> str:
    """
    Convert an absolute path to a safe flat filename component.
    Works on both Unix (/home/user/...) and Windows (C:/Users/...).
    """
    # as_posix() normalises backslashes to forward slashes on all platforms.
    posix = kit_xml.resolve().as_posix()
    # Strip drive-letter colons ("C:") then replace separators with "_".
    safe = posix.replace(":", "").replace("/", "_").lstrip("_")
    return safe


def get_mapping_path(kit_xml: Path) -> Path:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    return CONFIG_DIR / f"{_safe_filename(kit_xml)}.json"


def load_mapping(kit_xml: Path) -> dict[int, str] | None:
    path = get_mapping_path(kit_xml)
    if not path.exists():
        return None
    with open(path) as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def save_mapping(kit_xml: Path, mapping: dict[int, str]):
    path = get_mapping_path(kit_xml)
    with open(path, "w") as f:
        json.dump({str(k): v for k, v in mapping.items()}, f, indent=2)
    print(f"\nMapping saved → {path}")


# ─── Velocity calibration persistence ────────────────────────────────────────

def get_calibration_path(kit_xml: Path) -> Path:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    return CONFIG_DIR / f"{_safe_filename(kit_xml)}_calibration.json"


def load_calibration(kit_xml: Path) -> dict[int, dict]:
    """Returns {note: {min: int, max: int}}. Empty dict if no file yet."""
    path = get_calibration_path(kit_xml)
    if not path.exists():
        return {}
    with open(path) as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def save_calibration(kit_xml: Path, calibration: dict[int, dict]):
    path = get_calibration_path(kit_xml)
    with open(path, "w") as f:
        json.dump({str(k): v for k, v in calibration.items()}, f, indent=2)


def calibrated_vel_norm(velocity: int, note: int, calibration: dict[int, dict]) -> float:
    """
    Normalise a raw MIDI velocity (0-127) to 0.05-1.0.
    If calibration data exists for this note, stretch the observed
    min-max range to fill the full 0-1 window.
    Falls back to velocity/127 when no data is available yet.
    """
    floor = 0.05
    cal = calibration.get(note)
    if cal and cal["max"] > cal["min"]:
        # Mapea el rango [min, max] al rango [floor, 1.0]
        norm = (velocity - cal["min"]) / (cal["max"] - cal["min"])
        return max(floor, min(1.0, norm))
    return max(floor, velocity / 127.0)


def update_calibration(velocity: int, note: int, calibration: dict[int, dict]):
    """Expand the min/max window for this note in-place."""
    if note not in calibration:
        calibration[note] = {"min": velocity, "max": velocity}
    else:
        if velocity < calibration[note]["min"]:
            calibration[note]["min"] = velocity
        if velocity > calibration[note]["max"]:
            calibration[note]["max"] = velocity


# ─── Interactive mapping ───────────────────────────────────────────────────────

def do_mapping(
    instruments: list[dict], midi_port_idx: int, kit_xml: Path
) -> dict[int, str]:
    """
    Walk through each instrument, wait for a MIDI note-on or Enter (skip).
    Returns {midi_note: instrument_name}.
    """
    print("\n╔══════════════════════════════════════╗")
    print("║         MIDI MAPPING MODE            ║")
    print("║  Hit the pad → mapped                ║")
    print("║  Press Enter → skip                  ║")
    print("╚══════════════════════════════════════╝\n")

    midi_in = rtmidi.MidiIn()
    midi_in.open_port(midi_port_idx)
    midi_in.ignore_types(sysex=True, timing=True, active_sense=True)

    mapping: dict[int, str] = {}
    old_settings = _save_term()

    try:
        _set_cbreak()
        kit_dir = kit_xml.parent
        for inst in instruments:
            name = inst["name"]
            if not (kit_dir / inst["file"]).exists():
                continue
            print(f"  {name:<22} → ", end="", flush=True)

       

            # Flush stale MIDI messages
            while midi_in.get_message():
                pass

            note = None
            while True:
                # Check MIDI
                msg = midi_in.get_message()
                if msg:
                    data, _ = msg
                    if (
                        len(data) >= 3
                        and (data[0] & 0xF0) == 0x90  # note-on
                        and data[2] > 0                # velocity > 0
                    ):
                        note = data[1]
                        break

                # Check keyboard (Enter = skip, non-blocking)
                ch = _read_key()
                if ch is not None:
                    if ch in ("\n", "\r"):
                        break
                    # Any other key also skips
                    break

                time.sleep(0.005)

            if note is not None:
                mapping[note] = name
                print(f"note {note:3d}  ok")
            else:
                print("(skipped)")

    finally:
        _restore_term(old_settings)
        midi_in.close_port()
        del midi_in

    save_mapping(kit_xml, mapping)
    return mapping


# ─── Sample selection ─────────────────────────────────────────────────────────

def pick_velocity_layer(
    inst_name: str,
    layers: list[tuple[float, np.ndarray, str]],
    vel_norm: float,                        # 0.0–1.0, already calibrated
) -> tuple[np.ndarray, str]:
    """
    Select a velocity layer based on normalised velocity.
    Layers are sorted by power. The index is chosen proportionally to vel_norm.
    Returns (stereo_array, sample_name).
    """
    if not layers:
        raise ValueError("No layers available")
    sorted_layers = sorted(layers, key=lambda x: x[0])
    n = len(sorted_layers)
    idx = int(vel_norm * (n - 1))
    idx = max(0, min(n - 1, idx))
    power, stereo, name = sorted_layers[idx]
    return stereo, name


# ─── Main play loop ───────────────────────────────────────────────────────────

def run_player(
    samples: dict[str, list],
    mapping: dict[int, str],
    midi_port_idx: int,
    engine: AudioEngine,
    midi_channel: int | None = None,
    velocity_volume: bool = False,
    velocity_calibrate: bool = True,
    kit_xml: Path | None = None,
):
    midi_in = rtmidi.MidiIn()
    midi_in.open_port(midi_port_idx)
    midi_in.ignore_types(sysex=True, timing=True, active_sense=True)

    calibration: dict[int, dict] = {}
    if velocity_calibrate and kit_xml:
        calibration = load_calibration(kit_xml)

    print("\nActive mapping:")
    for note in sorted(mapping):
        inst  = mapping[note]
        n_lay = len(samples.get(inst, []))
        cal   = calibration.get(note)
        cal_str = f"  cal [{cal['min']}–{cal['max']}]" if cal else ""
        print(f"  note {note:3d}  →  {inst}  ({n_lay} layers){cal_str}")

    ch_label = f"channel {midi_channel}" if midi_channel is not None else "all channels"
    print(f"\nListening on {ch_label}")
    print(f"Velocity→volume: {'ON' if velocity_volume else 'OFF'}   "
          f"Velocity calibration: {'ON' if velocity_calibrate else 'OFF'}")

    vol_pct = lambda: f"{engine.volume * 100:.0f}%"
    print(f"Volume: {vol_pct()}")
    print("  [+] vol up   [-] vol down   [d] debug MIDI   [q] quit\n")

    debug   = False
    running = True
    old_settings = _save_term()

    try:
        _set_cbreak()

        while running:
            msg = midi_in.get_message()
            if msg:
                data, delta_t = msg

                if debug:
                    hex_bytes = " ".join(f"{b:02X}" for b in data)
                    dec_bytes = " ".join(f"{b:3d}" for b in data)
                    status_nibble = f"{data[0] & 0xF0:02X}" if data else "??"
                    ch_nibble     = (data[0] & 0x0F) + 1   if data else 0
                    print(
                        f"\r  MIDI  hex:[{hex_bytes}]  dec:[{dec_bytes}]"
                        f"  status:0x{status_nibble}  ch:{ch_nibble}"
                        f"  Δt:{delta_t:.4f}s"
                    )

                if len(data) >= 3:
                    status   = data[0] & 0xF0
                    ch       = data[0] & 0x0F
                    note     = data[1]
                    velocity = data[2]

                    if midi_channel is not None and ch != (midi_channel - 1):
                        pass
                    elif status == 0x90 and velocity > 0 and note in mapping:

                        if velocity_calibrate:
                            update_calibration(velocity, note, calibration)

                        if velocity_calibrate:
                            vel_norm = calibrated_vel_norm(velocity, note, calibration)
                        else:
                            vel_norm = velocity / 127.0

                        inst_name = mapping[note]
                        layers    = samples.get(inst_name)
                        if layers:
                            arr, sample_name = pick_velocity_layer(inst_name, layers, vel_norm)
                            if debug:
                                print(f"     sample: {sample_name}  vel_raw:{velocity:3d}  vel_norm:{vel_norm:.3f}")
                            gain = vel_norm if velocity_volume else 1.0
                            engine.play(arr, gain)

            ch = _read_key()
            if ch is not None:
                if ch == "+":
                    engine.volume = round(engine.volume + VOLUME_STEP, 3)
                    print(f"\r  Volume: {vol_pct()}    ", end="", flush=True)
                elif ch == "-":
                    engine.volume = max(0.0, round(engine.volume - VOLUME_STEP, 3))
                    print(f"\r  Volume: {vol_pct()}    ", end="", flush=True)
                elif ch in ("d", "D"):
                    debug = not debug
                    state = "ON  — raw MIDI bytes below" if debug else "OFF"
                    print(f"\r  Debug MIDI: {state}          ")
                elif ch in ("q", "Q"):
                    running = False

            time.sleep(0.001)

    except KeyboardInterrupt:
        pass
    finally:
        _restore_term(old_settings)
        midi_in.close_port()
        del midi_in
        if velocity_calibrate and kit_xml and calibration:
            save_calibration(kit_xml, calibration)


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Low-latency MIDI drum sampler for DrumGizmo kits",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "kit_xml",
        nargs="?",
        help="Path to the drumkit XML file (e.g. CrocellKit_default.xml)",
    )
    parser.add_argument("--remap", action="store_true",
        help="Discard saved mapping and re-map MIDI notes")
    parser.add_argument("--blocksize", type=int, default=BLOCKSIZE, metavar="N",
        help=f"Audio block size in frames (default {BLOCKSIZE} ≈ {1000*BLOCKSIZE/SAMPLERATE:.1f} ms)")
    parser.add_argument("--channel", type=int, default=None, metavar="N",
        help="Only respond to MIDI channel N (1-16). Default: all channels")
    parser.add_argument("--velocity-volume", choices=["on", "off"], default="off",
        help="Apply velocity as volume multiplier (default: off)")
    parser.add_argument("--velocity-calibrate", choices=["on", "off"], default="on",
        help="Dynamically calibrate velocity range per note (default: on)")
    parser.add_argument("--volume", type=float, default=1.0, metavar="F",
        help="Initial volume multiplier (default 1.0 = 100%%, values >1 allowed)")
    args = parser.parse_args()

    velocity_calibrate_enabled = args.velocity_calibrate == "on"

    if not args.kit_xml:
        ports = list_midi_ports()
        if not ports:
            print("No MIDI input devices detected.")
        else:
            print(f"Found {len(ports)} MIDI input device(s):")
            for i, p in enumerate(ports):
                print(f"  [{i}] {p}")
        sys.exit(0)

    kit_xml = Path(args.kit_xml)
    if not kit_xml.exists():
        print(f"Error: file not found: {kit_xml}", file=sys.stderr)
        sys.exit(1)

    print(f"Kit: {kit_xml.name}")
    try:
        instruments = parse_kit(kit_xml)
    except Exception as e:
        print(f"Error parsing kit XML: {e}", file=sys.stderr)
        sys.exit(1)

    inst_names = [i["name"] for i in instruments]
    print(f"Instruments ({len(inst_names)}): {', '.join(inst_names)}")

    midi_port_idx = select_midi_port()

    mapping: dict[int, str] | None = None
    if not args.remap:
        mapping = load_mapping(kit_xml)
        if mapping:
            print(f"Loaded mapping from disk ({len(mapping)} notes mapped)")

    if mapping is None:
        mapping = do_mapping(instruments, midi_port_idx, kit_xml)
        if velocity_calibrate_enabled and mapping:
            init_cal = {note: {"min": 127, "max": 0} for note in mapping}
            save_calibration(kit_xml, init_cal)
            print("Calibration file initialised with min=127, max=0 for all mapped notes.")

    if not mapping:
        print("No notes mapped. Exiting.")
        sys.exit(0)

    print("\nLoading samples…")
    kit_dir      = kit_xml.parent
    mapped_insts = set(mapping.values())
    samples: dict[str, list] = {}

    for inst in instruments:
        name = inst["name"]
        if name not in mapped_insts:
            continue
        inst_xml = kit_dir / inst["file"]
        print(f"  {name:<22} ", end="", flush=True)
        layers = parse_instrument_samples(inst_xml, inst["main_channels"])
        if layers:
            _ = layers[0][1].sum()
            samples[name] = layers
            total_s = sum(len(arr) for _, arr, _ in layers) / SAMPLERATE
            print(f"{len(layers):2d} layers  {total_s:.1f}s")
        else:
            print("no samples found")

    if not samples:
        print("No samples could be loaded. Check WAV paths.", file=sys.stderr)
        sys.exit(1)

    engine = AudioEngine(blocksize=args.blocksize)
    engine.volume = args.volume
    try:
        engine.start()
    except sd.PortAudioError as e:
        print(f"Audio error: {e}", file=sys.stderr)
        if IS_WINDOWS:
            print("Try reinstalling PortAudio or check your audio device settings.", file=sys.stderr)
        else:
            print("Try: apt install libportaudio2  or install PipeWire/JACK", file=sys.stderr)
        sys.exit(1)

    try:
        run_player(
            samples, mapping, midi_port_idx, engine,
            midi_channel=args.channel,
            velocity_volume=(args.velocity_volume == "on"),
            velocity_calibrate=velocity_calibrate_enabled,
            kit_xml=kit_xml,
        )
    finally:
        engine.stop()
        print("\nBye!")


if __name__ == "__main__":
    main()

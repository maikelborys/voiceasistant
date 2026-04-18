"""Quick mic level probe.

Opens each candidate input device for 3 seconds, prints RMS level.
If every device shows near-zero RMS while you're speaking, mic is muted/routed wrong.
"""

import argparse
import audioop
import time

import pyaudio

CHUNK = 1024
RATE = 16000
DURATION_SEC = 3.0


def probe(index: int, name: str) -> None:
    pa = pyaudio.PyAudio()
    try:
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=RATE,
            input=True,
            input_device_index=index,
            frames_per_buffer=CHUNK,
        )
    except Exception as e:
        print(f"  [{index}] {name}  -> FAILED to open: {e}")
        pa.terminate()
        return

    peak = 0
    samples = 0
    end = time.time() + DURATION_SEC
    try:
        while time.time() < end:
            data = stream.read(CHUNK, exception_on_overflow=False)
            rms = audioop.rms(data, 2)
            peak = max(peak, rms)
            samples += 1
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()

    bar = "#" * min(40, peak // 50)
    print(f"  [{index}] peak_rms={peak:>5}  {bar:<40}  {name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, default=None, help="Only probe this device index")
    args = parser.parse_args()

    pa = pyaudio.PyAudio()
    try:
        devs = [
            (i, pa.get_device_info_by_index(i))
            for i in range(pa.get_device_count())
        ]
    finally:
        pa.terminate()

    inputs = [(i, d) for (i, d) in devs if d["maxInputChannels"] > 0]
    if args.index is not None:
        inputs = [(i, d) for (i, d) in inputs if i == args.index]

    print(f"Speak continuously — each device probed for {DURATION_SEC}s.")
    for i, d in inputs:
        print(f"-> probing [{i}] {d['name']} ...")
        probe(i, d["name"])


if __name__ == "__main__":
    main()

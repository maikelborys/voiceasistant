"""Shared PyAudio device enumeration.

Used by both the local-audio transport (to log device state) and mic_probe.py
(to find input devices to RMS-probe). Keeps the PyAudio lifecycle managed in
one place so callers don't leak PortAudio instances.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

import pyaudio


@dataclass(frozen=True)
class AudioDevice:
    index: int
    name: str
    max_input_channels: int
    max_output_channels: int

    @property
    def is_input(self) -> bool:
        return self.max_input_channels > 0

    @property
    def is_output(self) -> bool:
        return self.max_output_channels > 0


@contextmanager
def pyaudio_session() -> Iterator[pyaudio.PyAudio]:
    pa = pyaudio.PyAudio()
    try:
        yield pa
    finally:
        pa.terminate()


def _device_from_info(info: dict) -> AudioDevice:
    return AudioDevice(
        index=int(info["index"]),
        name=str(info["name"]),
        max_input_channels=int(info["maxInputChannels"]),
        max_output_channels=int(info["maxOutputChannels"]),
    )


def list_devices() -> list[AudioDevice]:
    with pyaudio_session() as pa:
        return [
            _device_from_info(pa.get_device_info_by_index(i))
            for i in range(pa.get_device_count())
        ]


def default_input_device() -> AudioDevice:
    with pyaudio_session() as pa:
        return _device_from_info(pa.get_default_input_device_info())


def default_output_device() -> AudioDevice:
    with pyaudio_session() as pa:
        return _device_from_info(pa.get_default_output_device_info())

# SPDX-FileCopyrightText: 2017 Scott Shawcroft, written for Adafruit Industries
# SPDX-FileCopyrightText: Copyright (c) 2024 Cooper Dalrymple
#
# SPDX-License-Identifier: MIT
"""
`synthwaveform`
================================================================================

Helper library to generate waveforms.


* Author(s): Cooper Dalrymple

Implementation Notes
--------------------

**Hardware:**

.. todo:: Add links to any specific hardware product page(s), or category page(s).
  Use unordered list & hyperlink rST inline format: "* `Link Text <url>`_"

**Software and Dependencies:**

* Adafruit CircuitPython firmware for the supported boards:
  https://circuitpython.org/downloads

.. todo:: Uncomment or remove the Bus Device and/or the Register library dependencies
  based on the library's use of either.

# * Adafruit's Bus Device library: https://github.com/adafruit/Adafruit_CircuitPython_BusDevice
# * Adafruit's Register library: https://github.com/adafruit/Adafruit_CircuitPython_Register
"""

# imports

__version__ = "0.0.0+auto.0"
__repo__ = "https://github.com/dcooperdalrymple/CircuitPython_SynthWaveform.git"


import random

import adafruit_wave
import ulab.numpy as np
from micropython import const

_DEFAULT_SIZE = const(256)
_DEFAULT_DTYPE = np.int16

# Tools


def _minmax(dtype: np._DType, amplitude: float = 1.0) -> tuple[int, int]:
    # Determine range by data type
    if dtype is np.int8:
        minmax = -128, 127
    elif dtype is np.int16:
        minmax = -32768, 32767
    elif dtype is np.uint8:
        minmax = 0, 255
    elif dtype is np.uint16:
        minmax = 0, 65535
    elif dtype is np.float:
        minmax = -1.0, 1.0
    else:
        # NOTE: np.bool not supported
        raise ValueError("Invalid ulab.numpy._DType")

    # Scale range
    amplitude = min(max(amplitude, 0.0), 1.0)
    if amplitude < 1.0:
        mid = np.sum(minmax) / 2
        range = (minmax[1] - minmax[0]) / 2 * amplitude
        minmax = mid - range, mid + range
        if dtype is not np.float:
            minmax = round(minmax[0]), round(minmax[1])

    return minmax


def _phase(data: np.ndarray, amount: float = 0.0):
    # NOTE: If roll distance is 0, function returns an array of all 0s
    if (roll := round(data.size * -amount)) != 0:
        data = np.roll(data, roll)
    return data


def sine(
    amplitude: float = 1.0,
    phase: float = 0.0,
    scale: float = 1.0,
    size: int = _DEFAULT_SIZE,
    dtype: np._DType = _DEFAULT_DTYPE,
) -> np.ndarray:
    minmax = _minmax(dtype, amplitude)
    return np.array(
        np.sin(np.linspace(phase * np.pi, (phase + 2 * scale) * np.pi, size, endpoint=False))
        * minmax[1],
        dtype=dtype,
    )


def square(
    amplitude: float = 1.0,
    phase: float = 0.0,
    duty_cycle: float = 0.5,
    size: int = _DEFAULT_SIZE,
    dtype: np._DType = _DEFAULT_DTYPE,
) -> np.ndarray:
    if size < 2:
        raise ValueError("Array size must be greater than or equal to 2")
    minmax = _minmax(dtype, amplitude)
    duty_cycle = min(max(round(size * duty_cycle), 1), size - 1)
    return _phase(
        np.concatenate(
            (
                np.ones(duty_cycle, dtype=dtype) * minmax[1],
                np.ones(size - duty_cycle, dtype=dtype) * minmax[0],
            )
        ),
        phase,
    )


def triangle(
    amplitude: float = 1.0,
    phase: float = 0.0,
    shape: float = 0.5,
    size: int = _DEFAULT_SIZE,
    dtype: np._DType = _DEFAULT_DTYPE,
) -> np.ndarray:
    # NOTE: Numpy requires linspace arrays to at least have 2 elements
    if size < 4:
        raise ValueError("Array size must be greater than or equal to 4")
    minmax = _minmax(dtype, amplitude)
    shape = min(max(round(size * min(max(shape, 0.0), 1.0)), 2), size - 2)
    return _phase(
        np.concatenate(
            (
                np.linspace(minmax[0], minmax[1], num=shape, dtype=dtype),
                np.linspace(minmax[1], minmax[0], num=size - shape, dtype=dtype),
            )
        ),
        phase + 0.25,
    )


def saw(
    amplitude: float = 1.0,
    phase: float = 0.0,
    reverse: bool = False,
    size: int = _DEFAULT_SIZE,
    dtype: np._DType = _DEFAULT_DTYPE,
) -> np.ndarray:
    minmax = _minmax(dtype, amplitude)
    return _phase(np.linspace(minmax[not reverse], minmax[reverse], num=size, dtype=dtype), phase)


def noise(
    amplitude: float = 1.0, size: int = _DEFAULT_SIZE, dtype: np._DType = _DEFAULT_DTYPE
) -> np.ndarray:
    minmax = _minmax(dtype, amplitude)
    return np.array(
        [random.random() * (minmax[1] - minmax[0]) + minmax[0] for i in range(size)], dtype=dtype
    )


def mix(*waveforms: np.ndarray | tuple[np.ndarray, float]):
    # Check that all data types are the same
    if len(waveforms) > 1:
        for i in range(len(waveforms) - 1):
            if (
                waveforms[i] if type(waveforms[i]) is np.ndarray else waveforms[i][0]
            ).dtype is not (
                waveforms[i + 1] if type(waveforms[i + 1]) is np.ndarray else waveforms[i + 1][0]
            ).dtype:
                raise ValueError("Arrays must share the same data type")

    dtype = (waveforms[0] if type(waveforms[0]) is np.ndarray else waveforms[0][0]).dtype
    size = np.min(
        [(waveform if type(waveform) is np.ndarray else waveform[0]).size for waveform in waveforms]
    )
    mid = np.sum(_minmax(dtype)) / 2

    data = np.empty(size, dtype=dtype) + mid
    for waveform in waveforms:
        data += ((waveform if type(waveform) is np.ndarray else waveform[0])[:size] - mid) * (
            1.0 if type(waveform) is np.ndarray else min(max(waveform[1], 0.0), 1.0)
        )
    return data


def from_wav(path: str, max_size: int = None, channel: int = 0) -> tuple[np.ndarray, int]:
    # TODO: Support other WAV file data types
    # TODO: Add dtype argument and handle conversion

    try:
        with adafruit_wave.open(path, "rb") as wave:
            if wave.getsampwidth() != 2:
                # NOTE: There is likely a better exception type
                raise NotImplementedError("Only 16-bit WAV files are supported")

            # Read sample and convert to numpy
            nframes = wave.getnframes() if max_size is None else min(wave.getnframes(), max_size)
            data = list(memoryview(wave.readframes(nframes)).cast("h"))

            # Select only requested channel (waveform must be mono)
            if (nchannels := wave.getnchannels()) > 1:
                data = [data[i] for i in range(channel % nchannels, len(data), nchannels)]

            return np.array(data, dtype=np.int16), wave.getframerate()
    except OSError:
        pass

    return None

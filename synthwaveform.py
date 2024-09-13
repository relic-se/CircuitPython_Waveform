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
from micropython import const

try:
    import ulab.numpy as np
    from ulab.numpy import float as npfloat

    try:
        from ulab.numpy import _DType as DType
    except ImportError:
        pass
except ImportError:
    import numpy as np
    from numpy import dtype as DType
    from numpy import float64 as npfloat

_DEFAULT_SIZE = const(256)
_DEFAULT_DTYPE = np.int16


def _minmax(dtype: DType, amplitude: float = 1.0) -> tuple[int, int]:
    # Determine range by data type
    if dtype == np.int8:
        minmax = -128, 127
    elif dtype == np.int16:
        minmax = -32768, 32767
    elif dtype == np.uint8:
        minmax = 0, 255
    elif dtype == np.uint16:
        minmax = 0, 65535
    elif dtype == npfloat:
        minmax = -1.0, 1.0
    else:
        # NOTE: np.bool not supported
        raise ValueError("Invalid DType")

    # Scale range
    amplitude = min(max(amplitude, 0.0), 1.0)
    if amplitude < 1.0:
        mid = np.sum(minmax) / 2
        range = (minmax[1] - minmax[0]) / 2 * amplitude
        minmax = mid - range, mid + range
        if dtype != npfloat:
            minmax = round(minmax[0]), round(minmax[1])

    return minmax


def _prepare(data: np.ndarray | list, amplitude: float = 1.0, dtype: DType = None) -> np.ndarray:
    # Determine data type and minimum/maximum range
    if dtype is None:
        dtype = data.dtype if type(data) is np.ndarray else _DEFAULT_DTYPE
    minmax = _minmax(dtype)

    # Convert list of floats to ndarray
    if type(data) is list:
        data = np.array(data, dtype=npfloat)

    # Amplify and clip
    if data.dtype == npfloat:
        data = np.array(
            [min(max(i, -1.0), 1.0) for i in data * amplitude],
            dtype=npfloat,
        )

    if dtype != npfloat:
        # Scale to desired range
        data = (data + 1) / 2 * (minmax[1] - minmax[0]) + minmax[0]
        # Convert to integers
        data = [int(i) for i in data]

    # Convert to ndarray of desired type
    return np.array(data, dtype=dtype)


def sine(
    amplitude: float = 1.0,
    phase: float = 0.0,
    frequency: float = 1.0,
    size: int = _DEFAULT_SIZE,
    dtype: DType = _DEFAULT_DTYPE,
) -> np.ndarray:
    return _prepare(
        np.sin(np.linspace(phase * np.pi, (phase + 2 * frequency) * np.pi, size, endpoint=False)),
        amplitude,
        dtype,
    )


def square(  # noqa: PLR0913
    amplitude: float = 1.0,
    phase: float = 0.0,
    frequency: float = 1.0,
    duty_cycle: float = 0.5,
    size: int = _DEFAULT_SIZE,
    dtype: DType = _DEFAULT_DTYPE,
) -> np.ndarray:
    if size < 2:
        raise ValueError("Array size must be greater than or equal to 2")
    full_cycle = size / frequency
    duty_cycle = min(max(round(full_cycle * duty_cycle), 1), full_cycle - 1)
    phase *= full_cycle
    return _prepare(
        [1.0 if (i + phase) % full_cycle < duty_cycle else -1.0 for i in range(size)],
        amplitude,
        dtype,
    )


def triangle(
    amplitude: float = 1.0,
    phase: float = 0.0,
    frequency: float = 1.0,
    size: int = _DEFAULT_SIZE,
    dtype: DType = _DEFAULT_DTYPE,
) -> np.ndarray:
    phase *= size / frequency
    return _prepare(
        [abs(((i + phase) / size * frequency * 2 - 0.5) % 2 - 1) * 2 - 1 for i in range(size)],
        amplitude,
        dtype,
    )


def saw(  # noqa: PLR0913
    amplitude: float = 1.0,
    phase: float = 0.0,
    frequency: float = 1.0,
    reverse: bool = False,
    size: int = _DEFAULT_SIZE,
    dtype: DType = _DEFAULT_DTYPE,
) -> np.ndarray:
    phase *= size / frequency
    return _prepare(
        [
            ((((i + phase) / size * frequency) % 1) * 2 - 1) * (1 if reverse else -1)
            for i in range(size)
        ],
        amplitude,
        dtype,
    )


def noise(
    amplitude: float = 1.0, size: int = _DEFAULT_SIZE, dtype: DType = _DEFAULT_DTYPE
) -> np.ndarray:
    return _prepare(
        [random.random() * 2 - 1 for i in range(size)],
        amplitude,
        dtype,
    )


def mix(*waveforms: tuple) -> np.ndarray:
    # Check that all data types are the same
    if len(waveforms) > 1:
        for i in range(len(waveforms) - 1):
            if (waveforms[i] if type(waveforms[i]) is np.ndarray else waveforms[i][0]).dtype != (
                waveforms[i + 1] if type(waveforms[i + 1]) is np.ndarray else waveforms[i + 1][0]
            ).dtype:
                raise ValueError("Arrays must share the same data type")

    # Get properties of ndarray
    dtype = (waveforms[0] if type(waveforms[0]) is np.ndarray else waveforms[0][0]).dtype
    size = np.min(
        [(waveform if type(waveform) is np.ndarray else waveform[0]).size for waveform in waveforms]
    )
    minmax = _minmax(dtype)

    # Convert to float and mix
    data = np.empty(size, dtype=npfloat)
    for waveform in waveforms:
        data += (
            (
                np.array(
                    (waveform if type(waveform) is np.ndarray else waveform[0])[:size],
                    dtype=npfloat,
                )
                - minmax[0]
            )
            / (minmax[1] - minmax[0])
            * 2
            - 1
        ) * (1.0 if type(waveform) is np.ndarray else waveform[1])

    # Clip and convert to original data type
    return _prepare(data, dtype=dtype)


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

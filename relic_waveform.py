# SPDX-FileCopyrightText: 2017 Scott Shawcroft, written for Adafruit Industries
# SPDX-FileCopyrightText: Copyright (c) 2024 Cooper Dalrymple
#
# SPDX-License-Identifier: MIT
"""
`relic_waveform`
================================================================================

Helper library to generate waveforms.

* Author(s): Cooper Dalrymple

Implementation Notes
--------------------

**Software and Dependencies:**

* Adafruit CircuitPython firmware for the supported boards:
  https://circuitpython.org/downloads

* Adafruit's Wave library: https://github.com/adafruit/Adafruit_CircuitPython_Wave

Shared Parameters
-----------------

Each waveform type (besides :func:`relic_waveform.noise`) has 5 shared function parameters:
amplitude, phase, frequency, size, and dtype (data type).

Amplitude
^^^^^^^^^

By default, the waveform will reach the maximum value allowed by the range of the data type (or -1.0
and 1.0 in the case of :class:`ulab.numpy.float`). By modifying the amplitude parameter (which is
1.0 by default), you can increase or decrease the strength of the waveform or even invert the shape
by using a negative value. If the value of amplitude is greater than 1.0 (or less than -1.0), this
will introduce clipping to the waveform data at the minimum and maximum values.

.. |amplitude-1| image:: _static/amplitude-1.jpg
    :alt: sinusoidal wave with an amplitude of 1.0

.. |amplitude-2| image:: _static/amplitude-2.jpg
    :alt: sinusoidal wave with an amplitude of 0.5

.. |amplitude-3| image:: _static/amplitude-3.jpg
    :alt: sinusoidal wave with an amplitude of 2.0

.. |amplitude-4| image:: _static/amplitude-4.jpg
    :alt: sinusoidal wave with an amplitude of -1.0

.. table::
    :widths: 1 1 1 1

    +---------------+---------------+---------------+----------------+
    | |amplitude-1| | |amplitude-2| | |amplitude-3| | |amplitude-4|  |
    +---------------+---------------+---------------+----------------+
    | amplitude=1.0 | amplitude=0.5 | amplitude=2.0 | amplitude=-1.0 |
    +---------------+---------------+---------------+----------------+

Phase
^^^^^

Phase specifies the location of the wave cycle to begin the resulting waveform data. The phase
parameter value is relative to a single cycle, so modifying the frequency parameter will affect the
scale of the phase in relation to the size of the data.

.. |phase-1| image:: _static/phase-1.jpg
    :alt: sinusoidal wave with an phase of 0.0

.. |phase-2| image:: _static/phase-2.jpg
    :alt: sinusoidal wave with an phase of 0.5

.. |phase-3| image:: _static/phase-3.jpg
    :alt: sinusoidal wave with an phase of 0.5 and a frequency of 2.0

.. table::
    :widths: 1 1 1

    +-----------+-----------+--------------------------+
    | |phase-1| | |phase-2| | |phase-3|                |
    +-----------+-----------+--------------------------+
    | phase=0.0 | phase=0.5 | phase=0.5, frequency=2.0 |
    +-----------+-----------+--------------------------+

Frequency
^^^^^^^^^

By default, all waveforms will include a single wave cycle in the resulting data. The frequency
parameter defines the number of cycles of the waveform. By using a value of 0.0, the waveform will
stay at a constant value (not recommended). By using a negative value, the waveform will reverse
direction.

.. |frequency-1| image:: _static/frequency-1.jpg
    :alt: sinusoidal wave with an frequency of 1.0

.. |frequency-2| image:: _static/frequency-2.jpg
    :alt: sinusoidal wave with an frequency of 1.5

.. |frequency-3| image:: _static/frequency-3.jpg
    :alt: sinusoidal wave with an frequency of 4.0

.. |frequency-4| image:: _static/frequency-4.jpg
    :alt: sinusoidal wave with an frequency of -1.0

.. table::
    :widths: 1 1 1 1

    +---------------+---------------+---------------+----------------+
    | |frequency-1| | |frequency-2| | |frequency-3| | |frequency-4|  |
    +---------------+---------------+---------------+----------------+
    | frequency=1.0 | frequency=1.5 | frequency=4.0 | frequency=-1.0 |
    +---------------+---------------+---------------+----------------+

Size
^^^^

The size of the waveform data is 256 by default. For standard applications such as
:class:`synthio.Note`, this should be adequate. If a different array length is desired, change this
parameter to the desired positive integer.

Data Type
^^^^^^^^^

The data type and range of the resulting waveform data can be specified using :code:`dtype`. By
default, the data type is :class:`ulab.numpy.int16` which is commonly used for :mod:`synthio`.
The waveform data is scaled to the minimum and maximum values allowed by the selected data type. In
the case of :class:`ulab.numpy.int16`, the minimum and maximum is -32768 and 32767 respectively.

See :mod:`ulab.numpy` for the available data type options (excluding :class:`ulab.numpy.bool`).

"""

# imports

__version__ = "0.0.0+auto.0"
__repo__ = "https://github.com/dcooperdalrymple/CircuitPython_Waveform.git"


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


def _prepare(data, amplitude: float = 1.0, dtype: DType = None) -> np.ndarray:
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
    """Generate a sinusoidal waveform.

    .. figure:: _static/sine-1.jpg
        :width: 200
        :alt: sinusoidal wave

    :param amplitude: The level of the waveform
    :param phase: The cycle offset
    :param frequency: The number of cycles
    :param size: The number of frames of the resulting array
    :param dtype: Data type code to use for the resulting array
    """
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
    """Generate a square waveform.

    .. figure:: _static/square-1.jpg
        :width: 200
        :alt: square wave

    :param amplitude: The level of the waveform
    :param phase: The cycle offset
    :param frequency: The number of cycles
    :param duty_cycle: The ratio of a cycle that is "high" from 0.0 to 1.0

        .. |square-2| image:: _static/square-2.jpg
            :width: 200
            :alt: square wave with a 25% duty cycle

        .. |square-3| image:: _static/square-3.jpg
            :width: 200
            :alt: square wave with a 75% duty cycle

        .. table::
            :widths: 1 1

            +-----------------+-----------------+
            | |square-2|      | |square-3|      |
            +-----------------+-----------------+
            | duty_cycle=0.25 | duty_cycle=0.75 |
            +-----------------+-----------------+

    :param size: The number of frames of the resulting array
    :param dtype: Data type code to use for the resulting array
    """
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
    """Generate a triangle waveform.

    .. figure:: _static/triangle-1.jpg
        :width: 200
        :alt: triangle wave

    :param amplitude: The level of the waveform
    :param phase: The cycle offset
    :param frequency: The number of cycles
    :param size: The number of frames of the resulting array
    :param dtype: Data type code to use for the resulting array
    """
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
    """Generate a sawtooth waveform.

    .. figure:: _static/saw-1.jpg
        :width: 200
        :alt: sawtooth wave

    :param amplitude: The level of the waveform
    :param phase: The cycle offset
    :param frequency: The number of cycles
    :param reverse: The direction of the waveform, either decrementing (`False`) or incrementing
        (`True`)

        .. figure:: _static/saw-2.jpg
            :width: 200
            :alt: reversed sawtooth wave

            reverse=True

    :param size: The number of frames of the resulting array
    :param dtype: Data type code to use for the resulting array
    """
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
    """Generate random "white" noise

    .. figure:: _static/noise-1.jpg
        :width: 200
        :alt: noise

    :param amplitude: The level of the waveform
    :param size: The number of frames of the resulting array
    :param dtype: Data type code to use for the resulting array
    """
    return _prepare(
        [random.random() * 2 - 1 for i in range(size)],
        amplitude,
        dtype,
    )


def mix(*waveforms: tuple) -> np.ndarray:
    """Combine multiple waveforms together to a single waveform. All :class:`ulab.numpy.ndarray`
    objects must have the same data type. If the sizes of the arrays are inconsistent, the output
    will be sized to the shortest array.

    .. |mix-1| image:: _static/mix-1.jpg
        :width: 200
        :alt: example of waveform mixing

    .. table::

        +--------------------------------------------------+---------+
        | .. code-block:: python                           | |mix-1| |
        |                                                  |         |
        |     import relic_waveform                         |         |
        |     waveform = relic_waveform.mix(                |         |
        |         (relic_waveform.triangle(), 0.7),         |         |
        |         (relic_waveform.saw(frequency=2.0), 0.1), |         |
        |         (relic_waveform.saw(frequency=3.0), 0.1), |         |
        |         (relic_waveform.saw(frequency=4.0), 0.1), |         |
        |     )                                            |         |
        +--------------------------------------------------+---------+

    :param waveforms: The arrays to be mixed together. In order to specify the level for each
        waveform, each waveform can be provided as a tuple with the first element being the waveform
        data and the second being the level.
    :type waveform: np.ndarray | tuple[np.ndarray, float]
    """
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
    data = np.zeros(size, dtype=npfloat)
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
        ) * (1.0 if type(waveform) is np.ndarray or type(waveform[1]) is not float else waveform[1])

    # Clip and convert to original data type
    return _prepare(data, dtype=dtype)


def from_wav(path: str, max_size: int = None, channel: int = 0) -> tuple[np.ndarray, int]:
    """Read an single channel from a 16-bit audio wave file (".wav").

    :param path: The path to the ".wav" file
    :param max_size: The maximum limit of which to load samples from the audio file. If set as
        `None`, there is no limit in buffer size. Use to avoid memory overflow with large audio
        files. Defaults to `None`.
    :param channel: The channel to extract mono audio data from. Defaults to 0.
    :return: A tuple of the waveform data and the sample rate of the source wav file
    """
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

# SPDX-FileCopyrightText: 2017 Scott Shawcroft, written for Adafruit Industries
# SPDX-FileCopyrightText: Copyright (c) 2024 Cooper Dalrymple
#
# SPDX-License-Identifier: Unlicense

import ulab.numpy as np

import relic_waveform

SIZE = 8
TYPE = np.float

print(relic_waveform.sine(size=SIZE, dtype=TYPE))
print(relic_waveform.triangle(size=SIZE, dtype=TYPE))
print(relic_waveform.saw(size=SIZE, dtype=TYPE))
print(relic_waveform.square(size=SIZE, dtype=TYPE))
print(relic_waveform.noise(size=SIZE, dtype=TYPE))
print(
    relic_waveform.mix(
        relic_waveform.sine(size=SIZE, dtype=TYPE),
        (relic_waveform.noise(size=SIZE, dtype=TYPE), 0.5),
    )
)
print(relic_waveform.from_wav("test.wav", SIZE))

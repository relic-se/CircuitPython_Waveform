# SPDX-FileCopyrightText: 2017 Scott Shawcroft, written for Adafruit Industries
# SPDX-FileCopyrightText: Copyright (c) 2024 Cooper Dalrymple
#
# SPDX-License-Identifier: Unlicense

import audiopwmio
import board
import synthio

import relic_waveform

audio = audiopwmio.PWMAudioOut(board.A0)
synth = synthio.Synthesizer()
audio.play(synth)
synth.press(synthio.Note(440, waveform=relic_waveform.sine()))

# SPDX-FileCopyrightText: 2017 Scott Shawcroft, written for Adafruit Industries
# SPDX-FileCopyrightText: Copyright (c) 2024 Cooper Dalrymple
#
# SPDX-License-Identifier: Unlicense

import adafruit_debouncer
import adafruit_ssd1306
import board
import busio
import digitalio
import displayio
import rotaryio
import ulab.numpy as np

import relic_waveform

displayio.release_displays()

WIDTH = 128
HEIGHT = 64
INCREMENT = 0.1

i2c = busio.I2C(board.GP1, board.GP0)
display = adafruit_ssd1306.SSD1306_I2C(WIDTH, HEIGHT, i2c)

data = None

current_type = 0
current_index = 0
values = [1.0, 0.0, 1.0, 0.5]  # amplitude, phase, frequency, duty cycle/reverse


def update():
    if current_type == 0:
        data = relic_waveform.sine(
            amplitude=values[0],
            phase=values[1],
            frequency=values[2],
            size=WIDTH,
            dtype=np.float,
        )
    elif current_type == 1:
        data = relic_waveform.triangle(
            amplitude=values[0],
            phase=values[1],
            frequency=values[2],
            size=WIDTH,
            dtype=np.float,
        )
    elif current_type == 2:
        data = relic_waveform.saw(
            amplitude=values[0],
            phase=values[1],
            frequency=values[2],
            reverse=values[3] > 0.5,
            size=WIDTH,
            dtype=np.float,
        )
    elif current_type == 3:
        data = relic_waveform.square(
            amplitude=values[0],
            phase=values[1],
            frequency=values[2],
            duty_cycle=values[3],
            size=WIDTH,
            dtype=np.float,
        )
    elif current_type == 4:
        data = relic_waveform.noise(
            amplitude=values[0],
            size=WIDTH,
            dtype=np.float,
        )

    # Draw waveform on screen
    display.fill(0)
    for i in range(data.size):
        y1 = min(int((HEIGHT - data[i] * HEIGHT) / 2), HEIGHT - 1)
        y2 = min(int((HEIGHT - data[(i + 1) % WIDTH] * HEIGHT) / 2), HEIGHT - 1)
        if y1 == y2:
            display.pixel(i, y1, 1)
        else:
            display.vline(i, min(y1, y2), abs(y1 - y2), 1)
    display.show()


switch_pin = digitalio.DigitalInOut(board.GP4)
switch_pin.direction = digitalio.Direction.INPUT
switch_pin.pull = digitalio.Pull.UP
switch = adafruit_debouncer.Button(switch_pin)

encoder = rotaryio.IncrementalEncoder(board.GP2, board.GP3)
last_position = encoder.position

update()

while True:
    switch.update()
    if switch.short_count == 2:
        current_type = (current_type + 1) % 5
        update()
    elif switch.short_count == 1:
        current_index = (current_index + 1) % len(values)

    position = encoder.position
    if position != last_position:
        values[current_index] += (position - last_position) * INCREMENT
        last_position = position
        update()

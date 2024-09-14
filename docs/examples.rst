Simple test
-----------

Test each waveform type provided by this library. Provide a 16-bit WAV file in the root directory
called "test.wav" to test the :func:`synthwaveform.from_wav` function.

.. literalinclude:: ../examples/synthwaveform_simpletest.py
    :caption: examples/synthwaveform_simpletest.py
    :linenos:

synthio
-------

Generate a waveform to be used with a `synthio.Note` object.

.. literalinclude:: ../examples/synthwaveform_synthio.py
    :caption: examples/synthwaveform_synthio.py
    :linenos:

Oscilloscope
------------

View and manipulate waveform options with a rotary encoder and SSD1306 display.

.. literalinclude:: ../examples/synthwaveform_oscilloscope.py
    :caption: examples/synthwaveform_oscilloscope.py
    :linenos:

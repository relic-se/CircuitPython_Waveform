Simple test
-----------

Test each waveform type provided by this library. Provide a 16-bit WAV file in the root directory
called "test.wav" to test the :func:`relic_waveform.from_wav` function.

.. literalinclude:: ../examples/waveform_simpletest.py
    :caption: examples/waveform_simpletest.py
    :linenos:

synthio
-------

Generate a waveform to be used with a `synthio.Note` object.

.. literalinclude:: ../examples/waveform_synthio.py
    :caption: examples/waveform_synthio.py
    :linenos:

Oscilloscope
------------

View and manipulate waveform options with a rotary encoder and SSD1306 display.

.. literalinclude:: ../examples/waveform_oscilloscope.py
    :caption: examples/waveform_oscilloscope.py
    :linenos:

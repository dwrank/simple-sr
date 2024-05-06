#!/usr/bin/env python

import os
import sys
import pyaudio
import numpy as np
import wave
import subprocess


CHUNK = 1024
FORMAT = pyaudio.paInt16
SAMPLE_WIDTH = 2
CHANNELS = 1
RATE = 16000

RECORD_SECONDS = 5
CAPTURES = 12
WAVFILE = 'output.wav'

AMPLITUDE_TRIGGER = 1500


def open_wavfile(name):
    f = wave.open('output.wav', 'wb')
    f.setnchannels(CHANNELS)
    f.setsampwidth(SAMPLE_WIDTH)
    f.setframerate(RATE)
    return f

if __name__ == '__main__':
    record_seconds = RECORD_SECONDS

    if len(sys.argv) > 1:
        try:
            record_seconds = int(sys.argv[1])
        except Exception as e:
            print('Invalid argument: %s' % repr(e))

    print('Capturing for %d seconds.' % record_seconds)

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, #format=p.get_format_from_width(2),
                    channels=CHANNELS,
                    rate=RATE,
                    input=True)

    i = 0
    prev0 = np.zeros(CHUNK, dtype=np.int16)
    prev1 = prev0
    prev2 = prev0
    prev3 = prev0
    zeros = np.zeros(RATE - CAPTURES * CHUNK, dtype=np.int16)

    for _ in range(RATE // CHUNK * record_seconds):
        data = stream.read(CHUNK, exception_on_overflow=False)
        data = np.frombuffer(data, dtype=np.int16)

        if i == 0:
            n = data.max()
            if n > AMPLITUDE_TRIGGER:
                # capture the preceeding chunks
                f = open_wavfile(WAVFILE)
                f.writeframes(prev0)
                f.writeframes(prev1)
                f.writeframes(prev2)
                f.writeframes(prev3)
                i = 4

        if i > 0:
            print('^', end='', flush=True)
            f.writeframes(data)
            i += 1

            if i >= CAPTURES:
                # pad the end with zeros to make a 1 second sample
                f.writeframes(zeros)
                f.close()
                text = subprocess.check_output(['simple-sr', WAVFILE]).strip().decode('UTF-8')
                print(' %s' % text)

                # reset
                i = 0
        else:
            print('.', end='', flush=True)

        prev0 = prev1
        prev1 = prev2
        prev2 = prev3
        prev3 = data

    print('Stopping')
    stream.close()
    p.terminate()
    f.close()


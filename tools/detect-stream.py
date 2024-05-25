#!/usr/bin/env python3

import os
import sys
import pyaudio
#import numpy as np
import wave
import subprocess
import ctypes
import array


CHUNK = 1024
FORMAT = pyaudio.paInt16
SAMPLE_WIDTH = 2
CHANNELS = 1
RATE = 16000

RECORD_SECONDS = 5
CAPTURES = 12
WAVFILE = 'output.wav'

AMPLITUDE_TRIGGER = 1500

LABELS = [
    "eight", "five", "four",
    "nine", "no",
    "one", "seven", "six",
    "three", "two", 
    "yes", "zero"
]

def open_wavfile(name):
    f = wave.open('output.wav', 'wb')
    f.setnchannels(CHANNELS)
    f.setsampwidth(SAMPLE_WIDTH)
    f.setframerate(RATE)
    return f


def exceeds_val(data, amp):
    for i in range(0, len(data), 2):
        n = int.from_bytes(data[i:i+2], byteorder='little', signed=True)
        if n >= amp:
            return True
    return False


if __name__ == '__main__':
    record_seconds = RECORD_SECONDS

    if len(sys.argv) > 1:
        try:
            record_seconds = int(sys.argv[1])
        except Exception as e:
            print('Invalid argument: %s' % repr(e))

    print('Capturing for %d seconds.' % record_seconds)

    lib = ctypes.CDLL('libsimplesr.dylib')

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, #format=p.get_format_from_width(2),
                    channels=CHANNELS,
                    rate=RATE,
                    input=True)

    i = 0
    wav = []
    prev0 = b'\x00\x00' * CHUNK
    prev1 = prev0
    prev2 = prev0
    prev3 = prev0
    zeros = b'\x00\x00' * (RATE - CAPTURES * CHUNK)

    for _ in range(RATE // CHUNK * record_seconds):
        data = stream.read(CHUNK, exception_on_overflow=False)

        if i == 0:
            if exceeds_val(data, AMPLITUDE_TRIGGER):
                # capture the preceeding chunks
                wav = prev0 + prev1 + prev2 + prev3
                i = 4

        if i > 0:
            print('^', end='', flush=True)
            wav += data
            i += 1

            if i >= CAPTURES:
                # pad the end with zeros to make a 1 second sample
                wav += zeros

                float_ptr_t = ctypes.POINTER(ctypes.c_float * 12)
                pred = (ctypes.c_float * 12)()
                frames = ctypes.c_int(len(wav)//2)
                lib.predict_wav(wav, frames, pred)

                res = zip(LABELS, list(pred))
                res = sorted(list(res), key=lambda x: x[1], reverse=True)
                print(' %s' % res[0][0])

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


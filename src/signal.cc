#include <stdio.h>
#include <cmath>

#include <sndfile.hh>
#include <fftw3.h>

#include "signal.h"

#define PI 3.141592654

// T[] for unique_ptr is supported since C++11.
// make_unique for arrays is available since C++14.
// T[] for shared_ptr is supported since C++17.

Tensor1s read_wavfile(const char *fname, int &frames)
{
    SndfileHandle file(fname);

    Tensor1s tensor;
    frames = (int)file.frames();

    //printf("wav file: %s, sample rate: %d, channels: %d, format: %d, frames: %d\n",
    //        fname, file.samplerate(), file.channels(),
    //        file.format(), frames);

    if (!(file.format() & SF_FORMAT_PCM_16)) {
        printf ("Must be in PCM 16 format.\n") ;
        return false;
    }
    else {
        tensor.alloc(frames);
        file.read(tensor.data, frames);
    }

    return tensor;
}

void hanning(int window_length, float *buffer)
{
    // from numpy.hanning
    for (int i = 0; i < window_length; i++) {
        buffer[i] = 0.5 - 0.5 * cos(2.0 * PI * i / (window_length - 1.0));
    }
}

Tensor1f normalize_signal(short *data, int length)
{
    Tensor1f tensor(length);

    // from tf.audio.decode_wav 
    // The -32768 to 32767 signed 16-bit values will be scaled to -1.0 to 1.0 in float.
    double norm = 32768.0;

    for (int i = 0; i < length; i++) {
        tensor.data[i] = (float)data[i] / norm;
    }

    return tensor;
}

// Computes a magnitude spectrogram for a given vector of samples at a given
// frame length (in samples) and frame step (in samples).
Tensor2f get_spectrogram(const float *signal, int signal_len, int frame_len, int frame_step)
{
    int rfft_len = frame_len / 2 + 1;  // +1 for the nyquist frequency
    int chunk_end_pos = signal_len - frame_len + 1;
    int chunks = chunk_end_pos / frame_step + 1;

    float window[frame_len];
    hanning(frame_len, window);

    auto spec = Tensor2f(chunks, rfft_len);

    double *in = (double*)malloc(sizeof(double) * frame_len);
    fftw_complex *out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * frame_len);

    fftw_plan plan = fftw_plan_dft_r2c_1d(frame_len, in, out, FFTW_ESTIMATE);

    int chunk = 0;
    int chunk_pos = 0;
    float *data;
    while (chunk_pos < chunk_end_pos) {
        for (int i = 0; i < frame_len; i++) {
            int j = chunk_pos + i;

            if (j < signal_len) {
                in[i] = signal[j] * window[i];
            }
            else {
                // zero pad the remainder
                in[i] = 0.0;
            }
        }

        fftw_execute(plan);

        // store the rfft values
        for (int i = 0; i < rfft_len; i++) {
            // get the absolute value of the complex number
            spec(chunk, i) = std::sqrt(std::pow(out[i][0], 2) + std::pow(out[i][1], 2));
            if (chunk <= 1) {
                //printf("%d 0x%x: %f %f\n", chunk*spec->rows()+i, (long)data, *data, std::sqrt(std::pow(out[i][0], 2) + std::pow(out[i][1], 2)));
            }
        }
        if (chunk <= 1) {
            //printf("%f\n", *spec->data(0, spec->cols()-1));
        }

        chunk++;
        chunk_pos += frame_step;
    }

    fftw_destroy_plan(plan);
    free(in);
    fftw_free(out);

    return spec;
}


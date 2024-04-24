#ifndef __SIGNAL_H__
#define __SIGNAL_H__

#include <memory>
#include "tensor.h"

Tensor1s read_wavfile(const char *fname, int &frames);

void hanning(int window_length, float *buffer);

// normalize 16 bit signed values
Tensor1f normalize_signal(short *data, int length);

Tensor2f get_spectrogram(const float *signal, int signal_len, int frame_len, int frame_step);

#endif //__SIGNAL_H__

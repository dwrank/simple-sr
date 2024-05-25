#ifndef __MODEL_H__
#define __MODEL_H__

#include "tensor.h"
#include "signal.h"
#include "preprocess.h"
#include "weights.h"
#include "activation.h"
#include "conv2d.h"
#include "max_pooling.h"
#include "dense.h"

Tensor4f predict_wavfile(const char *wavfile);

extern "C" int predict_wav(short *wav, int frames, float *res);

Tensor4f predict_wav(Tensor1s &wav, int frames);

#endif //__MODEL_H__

#ifndef __UTILS_H__
#define __UTILS_H__

#include <string>
#include <vector>
#include "tensor.h"

template <typename T>
struct AudioSample
{
    int label;
    T *sample;
};

using AudioSampleS = AudioSample<short>;
using AudioSampleF = AudioSample<float>;

std::vector<std::string> get_audio_labels(const char *data_dir);

std::vector<Tensor1s> get_audio_samples(const char *data_dir);
std::vector<AudioSampleS> get_audio_samples_top(const char *data_dir);

Tensor2f preprocess_wav(const char *wav_file, int frame_len, int frame_step,
                        double mean, double std);
std::vector<Tensor2f> get_pre_audio_samples(const char *data_dir, int frame_len, int frame_step,
                                            double mean, double std);
std::vector<AudioSampleF> get_pre_audio_samples_top(const char *data_dir, int frame_len, int frame_step,
                                                    double mean, double std, int &n_labels);

void split_audio_samples(std::vector<AudioSampleF> &samples,
                         int n_train, int n_test,
                         float **x_train, float **y_train,
                         float **x_test, float **y_test);

#endif //__UTILS_H__

#include <dirent.h>  // Sorry Windows, POSIX only!
#include <sstream>
#include <algorithm>
#include <random>
#include <string.h>

#include "samples.h"
#include "signal.h"
#include "preprocess.h"

static bool is_dir(const dirent *dir)
{
    return dir->d_type == DT_DIR &&
           strcmp(dir->d_name, ".") != 0 &&
           strcmp(dir->d_name, "..") != 0;
}

std::vector<std::string> get_audio_labels(const char *data_dir)
{
    DIR *d = opendir(data_dir);
    struct dirent *dir;
    std::vector<std::string> labels;

    if (d) {
        while ((dir = readdir(d)) != NULL) {
            if (is_dir(dir)) {
                labels.push_back(std::string(dir->d_name));
            }
        }
    }

    closedir(d);

    return labels;
}

std::vector<Tensor1s> get_audio_samples(const char *data_dir)
{
    DIR *d = opendir(data_dir);
    struct dirent *dir;

    std::vector<Tensor1s> samples;

    std::stringstream ss;

    if (d) {
        while ((dir = readdir(d)) != NULL) {
            if (dir->d_type == DT_REG) {
                // get the filename
                ss.str("");
                ss.clear();
                ss << data_dir << "/" << dir->d_name;

                // read the wav file
                int frames;
                auto wav = read_wavfile(ss.str().c_str(), frames);
                if (!wav.is_empty()) {
                    samples.push_back(wav);
                }
            }
        }
    }

    closedir(d);

    return samples;
}

std::vector<AudioSampleS> get_audio_samples_top(const char *data_dir)
{
    printf("\nGet Audio Samples:\n");

    std::vector<AudioSampleS> samples;
    std::stringstream ss;

    auto labels = get_audio_labels(data_dir);

    int i = 1;
    for (const auto &label : labels) {
        printf("%-3d%-14s", i, label.c_str());

        ss.str("");
        ss.clear();
        ss << data_dir << "/" << label;
        auto new_samples = get_audio_samples(ss.str().c_str());
        for (auto &sample : new_samples) {
            samples.push_back(AudioSampleS {i, sample.data});
        }

        printf("%d files.\n", (int)new_samples.size());
        i++;
    }

    return samples;
}

Tensor2f preprocess_wav(const char *wav_file, int frame_len, int frame_step,
                        double mean, double std)
{
    int frames;

    auto wav_tensor = read_wavfile(wav_file, frames);

    if (!wav_tensor.is_empty()) {
        auto norm_tensor = normalize_signal(wav_tensor.data, frames);
        wav_tensor.free();

        auto spec = get_spectrogram(norm_tensor.data, frames, frame_len, frame_step);
        norm_tensor.free();

        auto resized_spec = resize_tensor2(spec, 32, 32);
        spec.free();

        auto norm_spec = normalize_tensor2(resized_spec, mean, std);
        resized_spec.free();

        return norm_spec;
    }

    return Tensor2f();
}

std::vector<Tensor2f> get_pre_audio_samples(const char *data_dir, int frame_len, int frame_step,
                                            double mean, double std)
{
    DIR *d = opendir(data_dir);
    struct dirent *dir;

    std::vector<Tensor2f> samples;
    std::stringstream ss;

    if (d) {
        while ((dir = readdir(d)) != NULL) {
            if (dir->d_type == DT_REG) {
                // get the filename
                ss.str("");
                ss.clear();
                ss << data_dir << "/" << dir->d_name;

                // get the preprocessed wav
                auto tensor = preprocess_wav(ss.str().c_str(), frame_len, frame_step, mean, std);
                if (!tensor.is_empty()) {
                    samples.push_back(tensor);
                }
            }
        }
    }

    closedir(d);

    return samples;
}

std::vector<AudioSampleF> get_pre_audio_samples_top(const char *data_dir, int frame_len, int frame_step,
                                                    double mean, double std, int &n_labels)
{
    printf("\nGet Audio Samples:\n");

    std::vector<AudioSampleF> samples;
    std::stringstream ss;

    auto labels = get_audio_labels(data_dir);
    n_labels = labels.size();

    int i = 0;
    for (const auto &label : labels) {
        printf("%-3d%-14s", i+1, label.c_str());

        ss.str("");
        ss.clear();
        ss << data_dir << "/" << label;
        auto new_samples = get_pre_audio_samples(ss.str().c_str(), frame_len, frame_step, mean, std);
        for (auto &sample : new_samples) {
            samples.push_back(AudioSampleF {i, sample.data});
        }

        printf("%d files.\n", (int)new_samples.size());
        i++;
    }

    return samples;
}

void split_audio_samples(std::vector<AudioSampleF> &samples,
                         int n_train, int n_test,
                         float **x_train, float **y_train,
                         float **x_test, float **y_test)
{
    printf("\nSplit Audio Samples\n");

    // shuffle the audio samples
    // auto rd = std::random_device {};  // for different outcomes
    // auto rng = std::default_random_engine { rd() };
    auto rng = std::default_random_engine {};
    std::shuffle(std::begin(samples), std::end(samples), rng);

    for (int i = 0; i < n_train; i++) {
        x_train[i] = samples[i].sample;
        y_train[i][samples[i].label] = 1.0;
    }

    int total = n_train + n_test;
    for (int i = 0; i < n_test; i++) {
        int si = n_train + i;
        x_test[i] = samples[si].sample;
        y_test[i][samples[si].label] = 1.0;
        i++;
    } 
}

void test(int a, int b, int c, int d)
{
    //int (*data)[a][b][c];
    //data = malloc(d*sizeof(*data));
}

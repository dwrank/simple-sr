#include <stdio.h>
#include <unistd.h>
#include <cmath>

#include "utils.h"
#include "tensor.h"
#include "signal.h"
#include "preprocess.h"
#include "weights.h"
#include "conv2d.h"

#include "signal_test.h"

#define FRAME_LEN 256
#define FRAME_STEP 128

int main(int argc, char **argv)
{
    char *fn_in = 0, *fn_out = 0;
    int c, seed = 131, max_epoch = 20, n_threads = 1, mini_size=64;

    // parse args
    while ((c = getopt(argc, argv, "i:o:m:h:f:d:s:t:v:")) >= 0) {
        if (c == 'i') fn_in = optarg;
        else if (c == 'o') fn_out = optarg;
        else if (c == 'm') max_epoch = atoi(optarg);
        //else if (c == 'h') n_h_fc = atoi(optarg);
        //else if (c == 'f') n_h_flt = atoi(optarg);
        //else if (c == 'd') dropout = atof(optarg);
        else if (c == 's') seed = atoi(optarg);
        else if (c == 't') n_threads = atoi(optarg);
        //else if (c == 'v') frac_val = atof(optarg);
    }

    if (argc - optind == 0 || (argc - optind == 1 && fn_in == 0)) {
        FILE *fp = stdout;
        fprintf(fp, "Usage: simple-sr [-i model] [-o model] [-t nThreads] <x.knd> [y.knd]\n");
        //return 1;
    }

    // Spectrogram test - add an option for this
    const char *wav_file = "../data/h_yes.wav";
    //const char *wav_file = "/Users/drank/dev/ml/python/audio/data/mini_speech_commands/right/988e2f9a_nohash_0.wav";

    //spectrogram_test(wav_file);

    int frames;
    auto t_wav = read_wavfile(wav_file, frames);

    if (!t_wav.is_empty()) {
        auto t_norm = normalize_signal(t_wav.data, frames);
        t_wav.free();

        auto t_spec = get_spectrogram(t_norm.data, frames, FRAME_LEN, FRAME_STEP);
        t_norm.free();

        print_data("Spec", t_spec);

        double mean = weights_mean;
        double std = std::sqrt(weights_variance);

        auto resized_spec = resize_tensor2(t_spec, 32, 32);
        t_spec.free();
        print_data("Resize Spec", resized_spec);

        auto t_norm_spec = normalize_tensor2(resized_spec, mean, std);
        resized_spec.free();
        print_data("Norm Spec", t_norm_spec);

        // Conv2D
        Tensor<float> t_conv_kernel;
        Tensor<float> t_conv_bias;

        t_conv_kernel.data = weights_conv2d_kernel;
        t_conv_kernel.set_dims(weights_conv2d_kernel_d0, weights_conv2d_kernel_d1,
                               weights_conv2d_kernel_d2, weights_conv2d_kernel_d3);

        t_conv_bias.data = weights_conv2d_bias;
        t_conv_bias.set_dims(weights_conv2d_bias_d0, weights_conv2d_bias_d1,
                             weights_conv2d_bias_d2, weights_conv2d_bias_d3);

        // t_norm_spec is a (1, 1, 32, 32) matrix
        // conv2d input needs it to be a (1, 32, 32, 1) matrix
        t_norm_spec.set_dims(1, 32, 32, 1);

        auto t_conv = conv2d(t_norm_spec, 32, 3, t_conv_kernel, t_conv_bias);
        t_norm_spec.free();
        print_data("Conv2D", t_conv);

        // Conv2D 1
        t_conv_kernel.data = weights_conv2d_1_kernel;
        t_conv_kernel.set_dims(weights_conv2d_1_kernel_d0, weights_conv2d_1_kernel_d1,
                               weights_conv2d_1_kernel_d2, weights_conv2d_1_kernel_d3);

        t_conv_bias.data = weights_conv2d_1_bias;
        t_conv_bias.set_dims(weights_conv2d_1_bias_d0, weights_conv2d_1_bias_d1,
                             weights_conv2d_1_bias_d2, weights_conv2d_1_bias_d3);

        auto t_conv_1 = conv2d(t_conv, 64, 3, t_conv_kernel, t_conv_bias);
        t_conv.free();
        print_data("Conv2D 1", t_conv_1);

        t_conv_1.free();
    }
#if 0
    // Get training data
    const char *data_dir = "/Users/drank/dev/ml/python/audio/data/mini_speech_commands";
    double mean = 0.12540944;
    double std = std::sqrt(0.58403146);

    int n_labels;
    auto samples = get_pre_audio_samples_top(data_dir, FRAME_LEN, FRAME_STEP, mean, std, n_labels);

    float val_split = 0.10;
    int n_train = 100;//samples.size() - static_cast<int>(samples.size() * val_split);
    int n_test = 10;//samples.size() - n_train;

#endif
    return 0;
}

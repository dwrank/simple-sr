#include <stdio.h>
#include <unistd.h>
#include <cmath>

#include "utils.h"
#include "tensor.h"
#include "signal.h"
#include "preprocess.h"
#include "weights.h"
#include "activation.h"
#include "conv2d.h"
#include "max_pooling.h"
#include "dense.h"
#include "labels.h"

#define FRAME_LEN 256
#define FRAME_STEP 128

int main(int argc, char **argv)
{
    int c;
    bool help = false;

    // parse args
    while ((c = getopt(argc, argv, "dh")) >= 0) {
        if (c == 'd') Debug = true;
        else if (c == 'h') help = true;
    }

    if (argc - optind == 0) {
        help = true;
    }

    if (help) {
        FILE *fp = stdout;
        fprintf(fp, "Usage: simple-sr [-h] [-d] <wav file>\n");
        fprintf(fp, "   -h: help\n");
        fprintf(fp, "   -d: debug\n");
        return 0;
    }

    std::string wav_file = argv[optind];

    int frames;
    auto t_wav = read_wavfile(wav_file.c_str(), frames);

    if (!t_wav.is_empty()) {
        auto t_norm = normalize_signal(t_wav.data, frames);
        t_wav.free();

        auto t_spec = get_spectrogram(t_norm.data, frames, FRAME_LEN, FRAME_STEP);
        t_norm.free();
        print_data("Spec", t_spec);

        double mean = weights_mean;
        double std = std::sqrt(weights_variance);

        auto resized_spec = resize_tensor2(t_spec, 64, 64);
        t_spec.free();
        print_data("Resize Spec", resized_spec);

        auto t_norm_spec = normalize_tensor2(resized_spec, mean, std);
        resized_spec.free();
        print_data("Norm Spec", t_norm_spec);

        // Conv2D
        Tensor<float> t_weights_kernel;
        Tensor<float> t_weights_bias;

        // set the weights
        t_weights_kernel.data = weights_conv2d_kernel;
        t_weights_kernel.set_dims(weights_conv2d_kernel_d0, weights_conv2d_kernel_d1,
                                  weights_conv2d_kernel_d2, weights_conv2d_kernel_d3);

        t_weights_bias.data = weights_conv2d_bias;
        t_weights_bias.set_dims(weights_conv2d_bias_d0, weights_conv2d_bias_d1,
                                weights_conv2d_bias_d2, weights_conv2d_bias_d3);

        // t_norm_spec is a (1, 1, 64, 64) matrix
        // conv2d input needs it to be a (1, 64, 64, 1) matrix
        t_norm_spec.set_dims(1, 64, 64, 1);

        auto t_conv = conv2d(t_norm_spec, t_weights_kernel, t_weights_bias, relu<float>);
        t_norm_spec.free();
        print_data("Conv2D", t_conv);

        auto t_max_pool = max_pooling_2d(t_conv);
        t_conv.free();
        print_data("MaxPooling2D", t_max_pool);

        // Conv2D V1
        t_weights_kernel.data = weights_conv2d_1_kernel;
        t_weights_kernel.set_dims(weights_conv2d_1_kernel_d0, weights_conv2d_1_kernel_d1,
                                  weights_conv2d_1_kernel_d2, weights_conv2d_1_kernel_d3);

        t_weights_bias.data = weights_conv2d_1_bias;
        t_weights_bias.set_dims(weights_conv2d_1_bias_d0, weights_conv2d_1_bias_d1,
                                weights_conv2d_1_bias_d2, weights_conv2d_1_bias_d3);

        auto t_conv_1 = conv2d(t_max_pool, t_weights_kernel, t_weights_bias, relu<float>);
        t_max_pool.free();
        print_data("Conv2D 1", t_conv_1);

        auto t_max_pool_1 = max_pooling_2d(t_conv_1);
        t_conv_1.free();
        print_data("MaxPooling2D", t_max_pool_1);

        // Conv2D V2
        t_weights_kernel.data = weights_conv2d_2_kernel;
        t_weights_kernel.set_dims(weights_conv2d_2_kernel_d0, weights_conv2d_2_kernel_d1,
                                  weights_conv2d_2_kernel_d2, weights_conv2d_2_kernel_d3);

        t_weights_bias.data = weights_conv2d_2_bias;
        t_weights_bias.set_dims(weights_conv2d_2_bias_d0, weights_conv2d_2_bias_d1,
                                weights_conv2d_2_bias_d2, weights_conv2d_2_bias_d3);

        auto t_conv_2 = conv2d(t_max_pool_1, t_weights_kernel, t_weights_bias, relu<float>);
        t_max_pool_1.free();
        print_data("Conv2D 2", t_conv_2);

        auto t_max_pool_2 = max_pooling_2d(t_conv_2);
        t_conv_2.free();
        print_data("MaxPooling2D", t_max_pool_2);

        // Dense 128
        // flatten the input to dense
        t_max_pool.set_dims(1, 1, 1, t_max_pool_2.length());

        // set the weights
        t_weights_kernel.data = weights_dense_kernel;
        t_weights_kernel.set_dims(weights_dense_kernel_d0, weights_dense_kernel_d1,
                                  weights_dense_kernel_d2, weights_dense_kernel_d3);

        t_weights_bias.data = weights_dense_bias;
        t_weights_bias.set_dims(weights_dense_bias_d0, weights_dense_bias_d1,
                                weights_dense_bias_d2, weights_dense_bias_d3);

        auto t_dense = dense(t_max_pool_2, t_weights_kernel, t_weights_bias, relu<float>);
        t_max_pool_2.free();
        print_data("Dense 128", t_dense);

        t_max_pool.free();

        // Dense 8
        // set the weights
        t_weights_kernel.data = weights_dense_1_kernel;
        t_weights_kernel.set_dims(weights_dense_1_kernel_d0, weights_dense_1_kernel_d1,
                               weights_dense_1_kernel_d2, weights_dense_1_kernel_d3);

        t_weights_bias.data = weights_dense_1_bias;
        t_weights_bias.set_dims(weights_dense_1_bias_d0, weights_dense_1_bias_d1,
                             weights_dense_1_bias_d2, weights_dense_1_bias_d3);

        auto t_labels = dense(t_dense, t_weights_kernel, t_weights_bias);
        t_dense.free();
        print_data("Dense 8", t_labels);

        print_label(t_labels);
        t_labels.free();
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

#include <stdio.h>
#include <unistd.h>
#include <cmath>

#include "utils.h"
#include "tensor.h"
#include "signal.h"
#include "preprocess.h"

#include "signal_test.h"

#include "kann_extra/kann_data.h"
#include "kann.h"

#define FRAME_LEN 256
#define FRAME_STEP 128

static void print_data(float *data, int len)
{
    printf("\nTrain");
    for (int i = 0; i < len; i++) {
        if (i % 16 == 0) {
            printf("\n");
        }
        printf("%f ", data[i]);
    }
    putchar('\n');
}

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
    //const char *wav_file = "../data/h_yes.wav";
    const char *wav_file = "/Users/drank/dev/ml/python/audio/data/mini_speech_commands/right/988e2f9a_nohash_0.wav";
    spectrogram_test(wav_file);

#if 1
    // Get training data
    const char *data_dir = "/Users/drank/dev/ml/python/audio/data/mini_speech_commands";
    double mean = 0.12540944;
    double std = std::sqrt(0.58403146);

    int n_labels;
    auto samples = get_pre_audio_samples_top(data_dir, FRAME_LEN, FRAME_STEP, mean, std, n_labels);

    float val_split = 0.10;
    int n_train = 100;//samples.size() - static_cast<int>(samples.size() * val_split);
    int n_test = 10;//samples.size() - n_train;

    // allocate memory for the x and y matrices
    float **x_train = (float**)calloc(n_train, sizeof(float*));
    float **y_train = (float**)calloc(n_train, sizeof(float*));
    float **x_test = (float**)calloc(n_test, sizeof(float*));
    float **y_test = (float**)calloc(n_test, sizeof(float*));
    for (int i = 0; i < n_train; i++) {
        y_train[i] = (float*)calloc(n_labels, sizeof(float));
    }
    for (int i = 0; i < n_test; i++) {
        y_test[i] = (float*)calloc(n_labels, sizeof(float));
    }

    split_audio_samples(samples,
                        n_train, n_test,
                        x_train, y_train,
                        x_test, y_test);
    print_data(x_train[0], 32);
    print_data(y_train[0], 8);

    // CNN
    kann_t *ann = nullptr;
    float **x, **y;
    int n_samples = 0;
    int n_out = n_labels;
    //kann_data_t *x, *y;

    kad_trap_fe();
    kann_srand(seed);

    if (fn_in) {
        ann = kann_load(fn_in);
    }
    else {
        // based on tensorflow simple audio speech recognition tutorial:
        // https://www.tensorflow.org/tutorials/audio/simple_audio

        kad_node_t *t;
        // 4-D input: (mini batch size, # channels, height, width)
        t = kad_feed(4, 1, 1, 32, 32), t->ext_flag |= KANN_F_IN;
        // conv2d: 32 filters, 3x3 kernel, 1x1 stride, 0x0 padding
        t = kad_relu(kann_layer_conv2d(t, 32, 3, 3, 1, 1, 0, 0));
        // conv2d: 64 filters, ...
        t = kad_relu(kann_layer_conv2d(t, 64, 3, 3, 1, 1, 0, 0));
        // max pooling 2-D: 2x2 kernel, 2x2 stride, 0x0 padding
        t = kad_max2d(t, 2, 2, 2, 2, 0, 0);
        // dropout
        t = kann_layer_dropout(t, 0.25);
        // flatten: the dense layer implicitly flattens
        // dense: 128 hidden units
        t = kann_layer_dense(t, 128);
        // dropout
        t = kann_layer_dropout(t, 0.5);
        // dense: n_out=8 classification labels
        t = kann_layer_dense(t, n_out);
        // multi-class cross entropy cost
        ann = kann_new(kann_layer_cost(t, n_out, KANN_C_CEM), 0);
    }

    if (fn_out) {  //train
        //assert(y->n_col == n_out);  // expected output dimensions

        if (n_threads > 1) {
            kann_mt(ann, n_threads, mini_size);
        }

        // ???
        int max_drop_streak = 10;
        float lr = 0.001f, frac_val = 0.1f;
        kann_train_fnn1(ann, lr, mini_size, max_epoch, max_drop_streak, frac_val,
                        n_train, x_train, y_train);

        kann_save(fn_out, ann);
        // free y
    }
    else {  // apply
        kann_switch(ann, 0);  // 0 is no train
        int n_out_model = kann_dim_out(ann);
        assert(n_out_model == n_out);  // expected output dimensions

        for (int i = 0; i < n_samples; i++) {
            const float *y = kann_apply1(ann, x[i]);

            for (int j = 0; j < n_out; j++) {
                if (j) putchar('\t');
                printf("%.3g", y[j] + 1.0f - 1.0f);
            }
            putchar('\n');
        }
    }

    // free x and y
    for (int i = 0; i < n_train; i++) {
        free(x_train[i]);
        free(y_train[i]);
    }
    free(x_train);
    free(y_train);

    for (int i = 0; i < n_test; i++) {
        free(x_test[i]);
        free(y_test[i]);
    }
    free(x_test);
    free(y_test);

    kann_delete(ann);

#endif
    return 0;
}

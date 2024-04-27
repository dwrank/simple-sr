#include <stdio.h>
#include <unistd.h>
#include <cmath>

#include "utils.h"
#include "tensor.h"
#include "signal.h"
#include "preprocess.h"

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

    int frames;
    auto wav = read_wavfile(wav_file, frames);
    if (!wav.is_empty()) {
        print_data(wav);
    }
    wav.free();
    spectrogram_test(wav_file);

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

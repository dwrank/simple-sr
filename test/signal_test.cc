
#include <cmath>

#include "signal_test.h"
#include "signal.h"
#include "preprocess.h"
//#include "utils.h"

#define FRAME_LEN 256
#define FRAME_STEP 128

static void print_data(const char *name, Tensor2f &t)
{
    printf("\n%s", name);
    for (int i = 0; i < t.cols(); i++) {
        if (i % 16 == 0) {
            printf("\n");
        }
        printf("%f ", t(0, i));
    }
    putchar('\n');
}

void spectrogram_test(const char *wav_file)
{
    int frames;

    auto wav_tensor = read_wavfile(wav_file, frames);

    if (!wav_tensor.is_empty()) {
        auto norm_tensor = normalize_signal(wav_tensor.data, frames);
        wav_tensor.free();

        auto spec = get_spectrogram(norm_tensor.data, frames, FRAME_LEN, FRAME_STEP);
        norm_tensor.free();

        print_data("Spec", spec);

        double mean = 0.12540944;
        double std = std::sqrt(0.58403146);

        auto resized_spec = resize_tensor2(spec, 32, 32);
        spec.free();
        print_data("Resize Spec", resized_spec);

        auto norm_spec = normalize_tensor2(resized_spec, mean, std);
        resized_spec.free();
        print_data("Norm Spec", norm_spec);

        norm_spec.free();
    }
}

int main(int argc, char **argv)
{
    const char *wavfile = "../data/yes.wav";

    spectrogram_test(wavfile);

    return 0;
}

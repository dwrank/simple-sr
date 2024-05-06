#include <unistd.h>

#include "model.h"
#include "labels.h"

int main(int argc, char **argv)
{
    int c;
    bool help = false;

    // parse args
    while ((c = getopt(argc, argv, "dhv")) >= 0) {
        if (c == 'd') Debug = true;
        else if (c == 'h') help = true;
        if (c == 'v') Verbose = true;
    }

    if (argc - optind == 0) {
        help = true;
    }

    if (help) {
        FILE *fp = stdout;
        fprintf(fp, "Usage: simple-sr [-h] [-d] <wav file>\n");
        fprintf(fp, "   -h: help\n");
        fprintf(fp, "   -d: debug\n");
        fprintf(fp, "   -v: verbose\n");
        return 0;
    }

    std::string wavfile = argv[optind];

    auto pred = predict_wavfile(wavfile.c_str());

    if (!pred.is_empty()) {
        print_label(pred);
        pred.free();
    }

    return 0;
}

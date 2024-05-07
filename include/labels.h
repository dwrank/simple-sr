#ifndef __LABELS_H__
#define __LABELS_H__

#include <vector>
#include "utils.h"

template<typename T>
struct LabelValue
{
    std::string label;
    T value;
};

enum Labels {
    eight, five, four,
    nine, no,
    one, seven, six,
    three, two,
    yes, zero
};

const std::string labels[] = {
    "down", "eight", "five", "four",
    "go", "left", "nine", "no",
    "one", "right", "seven", "six",
    "stop", "three", "two", "up",
    "yes", "zero"
};

template<typename T>
void print_label(const Tensor<T> &t)
{
    std::vector<LabelValue<T>> lvs;

    for (int i = 0; i < t.length(); i++) {
        lvs.push_back(LabelValue<T> { labels[i], t.data[i] });
    }

    std::sort(lvs.begin(), lvs.end(),
            [](const LabelValue<T> &lv1, const LabelValue<T> &lv2) {
                return lv1.value > lv2.value;
            });

    if (Verbose) {
        printf("\n[Prediction]\n");

        for (const auto &lv : lvs) {
            printf("%-12s", lv.label.c_str());
        }
        putchar('\n');

        for (const auto &lv : lvs) {
            printf("%-12f", static_cast<float>(lv.value));
        }
        putchar('\n');

        printf("\n==========>   %s   <==========\n\n", lvs[0].label.c_str());
    }
    else {
        printf("%s\n", lvs[0].label.c_str());
    }
}

#endif //__LABELS_H__

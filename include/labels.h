#ifndef __LABELS_H__
#define __LABELS_H__

enum Labels {
    down, go, left, no, right, stop, up, yes
};

const std::string labels[] = {
    "down", "go", "left", "no", "right", "stop", "up", "yes"
};

template<typename T>
void print_label(const Tensor<T> &t)
{
    T max = t.data[0];
    int max_i = 0;

    printf("\n[Prediction]\n");

    for (int i = 0; i < t.length(); i++) {
        printf("%-12s", labels[i].c_str());
    }
    putchar('\n');

    for (int i = 0; i < t.length(); i++) {
        printf("%-12f", static_cast<float>(t.data[i]));
        if (t.data[i] > max) {
            max = t.data[i];
            max_i = i;
        }
    }
    putchar('\n');

    // The prediction for 'no' and 'down' is close to 'go',
    // but the prediction for 'go' is not so close to 'no' and 'down',
    // so select 'no' or 'down' if either is close to 'go'.
    if (max_i == Labels::go) {
        if (std::abs(t.data[Labels::go] - t.data[Labels::no]) < 2) {
            max_i = Labels::no;
        }
        else if (std::abs(t.data[Labels::go] - t.data[Labels::down]) < 2) {
            max_i = Labels::down;
        }
    }

    printf("\n==========>   %s   <==========\n\n", labels[max_i].c_str());
}

#endif //__LABELS_H__

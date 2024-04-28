#ifndef __MAX_POOLING_H__
#define __MAX_POOLING_H__

#include "utils.h"

// expects a 4 dimensional input (groups, rows, cols, channels)
template<typename T>
Tensor<T> max_pooling_2d(const Tensor<T> &in_t)
{
    int groups = in_t.d0();
    int in_rows = in_t.d1();
    int in_cols = in_t.d2();
    int channels = in_t.d3();

    int out_rows = in_rows / 2;
    int out_cols = in_cols / 2;

    Tensor<T> out_t(groups, out_rows, out_cols, channels); 

    if (Debug) {
        printf("\n[MaxPooling2D]\n");
        in_t.print_dims("in_t");
        out_t.print_dims("out_t");
    }

    for (int g = 0; g < groups; g++) {
        for (int out_r = 0; out_r < out_rows; out_r++) {

            int in_r = out_r * 2;

            for (int out_c = 0; out_c < out_cols; out_c++) {

                int in_c = out_c * 2;
                int out_c_pos = out_t.pos(g, out_r, out_c, 0);

                for (int ch = 0; ch < channels; ch++) {

                    T max_val = in_t(g, in_r, in_c, ch);

                    for (int r = 0; r < 2; r++) {
                        for (int c = 0; c < 2; c++) {
                            T in_val = in_t(g, in_r + r, in_c + c, ch);
                            if (in_val > max_val) {
                                max_val = in_val;
                            }
                        }
                    }

                    out_t[out_c_pos + ch] = max_val;
                }
            }
        }
    }

    return out_t;
}

#endif //__MAX_POOLING_H__

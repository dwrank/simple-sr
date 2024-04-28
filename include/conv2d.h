#ifndef __CONV2D_H__
#define __CONV2D_H__

#include "tensor.h"
#include "activation.h"

// filters is a 4-D matrix (kernel_rows, kernel_cols, in_channels, out_channels)
// Fill the kernel matrix with the out channel weight from each
// kernel row, kernel col, and input channel.
template<typename T>
static void fill_kernel(T *kernel, const Tensor<T> &filters, int out_ch)
{
    int i = 0;
    for (int r = 0; r < filters.d0(); r++) {
        for (int c = 0; c < filters.d1(); c++) {
            for (int in_ch = 0; in_ch < filters.d2(); in_ch++) {
                kernel[i++] = filters(r, c, in_ch, out_ch);
            }
        }
    }
}

// expects a 4 dimensional input (groups, in_rows, in_cols, in_channels)
// and filters (kernel_rows, kernel_cols, in_channels, out_channels)
// biases is a 1 dimensional vector matching the number of channels or is empty
template<typename T>
Tensor<T> conv2d(const Tensor<T> &in_t, int out_size, int kernel_size, const Tensor<T> &filters, const Tensor<T> &biases)
{
    // the edges are not padded, so only points that the filters can cover are included
    int window_row_size = filters.d0();
    int window_col_size = filters.d1();
    int window_row_margin = static_cast<int>(window_row_size / 2);
    int window_col_margin = static_cast<int>(window_col_size / 2);
    
    int groups = in_t.d0();
    int in_rows = in_t.d1();
    int in_cols = in_t.d2();
    int in_channels = in_t.d3();

    int out_rows = in_rows - window_row_margin * 2;
    int out_cols = in_cols - window_col_margin * 2;

    // number of output channels
    int out_channels = filters.d3();


    Tensor<T> out_t(groups, out_rows, out_cols, out_channels); 

#if 1
    printf("\n[Conv2D]\n");
    in_t.print_dims("in_t");
    out_t.print_dims("out_t");
    filters.print_dims("filters");
    biases.print_dims("biases");
#endif

    bool debug = true;
    T kernel[window_row_size * window_col_size * in_t.d3()];

    for (int g = 0; g < groups; g++) {
        for (int r = 0; r < out_rows; r++) {  // skip unpadded edges
            for (int c = 0; c < out_cols; c++) {  // skip unpadded edges

                int out_pos = out_t.pos(g, r, c, 0);

                for (int out_ch = 0; out_ch < out_channels; out_ch++) {

                    // bias
                    T bias = 0.0;
                    if (!biases.is_empty()) {
                        bias = biases[out_ch];
                    }

                    // kernel
                    fill_kernel(kernel, filters, out_ch);

                    // sum over the product of the input window and kernel
                    int i = 0;
                    T sum = 0;

                    for (int kr = r; kr < r + window_row_size; kr++) {
                        for (int kc = c; kc < c + window_col_size; kc++) {

                            int in_pos = in_t.pos(g, kr, kc, 0);

                            for (int in_ch = 0; in_ch < in_channels; in_ch++) {
                                //sum += in_t(g, kr, kc, in_ch) * kernel[i++];
                                sum += in_t.data[in_pos + in_ch] * kernel[i++];
                            }
                        }
                    }
                    
                    /*if (debug) {
                        printf('in shape: %s, filter: %s' % (input_window.shape, filter.shape));
                        debug = false;
                    }*/

                    // add the bias and apply the activation function (relu)
                    // to get the output channel value
                    //out_t(g, r, c, out_ch) = relu(sum + bias);
                    out_t.data[out_pos + out_ch] = relu(sum + bias);
                }
	    }
	}
    }

    return out_t;
}

#endif //__CONV2D_H__

#ifndef __DENSE_H__
#define __DENSE_H__

#include "utils.h"
#include "tensor.h"
#include "activation.h"

// expects a 1 dimensional input (1, 1, 1, values)
// weights is a 2 dimensional matrix (1, 1, inputs, outputs)
// biases is a 1 dimensional matrix (1, 1, 1, outputs)
template<typename T>
Tensor<T> dense(const Tensor<T> &in_t,
                const Tensor<T> &weights, const Tensor<T> &biases,
                std::function<float(float)> activation=nullptr)
{
    int inputs = in_t.length();
    int outputs = weights.d3();

    // assumes the input min is 0 (relu)
    Tensor<T> out_t(1, 1, 1, outputs); 

    if (Debug) {
        printf("\n[Dense]\n");
        in_t.print_dims("in_t");
        out_t.print_dims("out_t");
    }

    for (int out_i = 0; out_i < outputs; out_i++) {

        T sum = 0;

        for (int in_i = 0; in_i < inputs; in_i++) {
            sum += in_t.data[in_i] * weights(in_i, out_i);
        }

        sum += biases.data[out_i];
        if (activation) {
            out_t[out_i] = activation(sum);
        }
        else {
            out_t[out_i] = sum;
        }
    }

    return out_t;
}

#endif //__DENSE_H__

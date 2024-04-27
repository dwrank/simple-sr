#ifndef __CONV2D_H__
#define __CONV2D_H__

#include "tensor.h"
#include "activation.h"

template<typename T>
Tensor4f conv2d(Tensor4f &in_t, int out_size, int kernel_size, const Tensor4f &filters, const Tensor1f &biases);

#endif //__CONV2D_H__

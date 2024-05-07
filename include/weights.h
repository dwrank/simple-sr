#ifndef __WEIGHTS_H__
#define __WEIGHTS_H__
// dimensions: (1,)
extern float weights_mean;

// dimensions: (1,)
extern float weights_variance;

// dimensions: ()
extern int weights_count;

// dimensions: (3, 3, 1, 32)
const int weights_conv2d_kernel_d0 = 3;
const int weights_conv2d_kernel_d1 = 3;
const int weights_conv2d_kernel_d2 = 1;
const int weights_conv2d_kernel_d3 = 32;
const int weights_conv2d_kernel_len = 288;
extern float weights_conv2d_kernel[weights_conv2d_kernel_len];

// dimensions: (32,)
const int weights_conv2d_bias_d0 = 1;
const int weights_conv2d_bias_d1 = 1;
const int weights_conv2d_bias_d2 = 1;
const int weights_conv2d_bias_d3 = 32;
const int weights_conv2d_bias_len = 32;
extern float weights_conv2d_bias[weights_conv2d_bias_len];

// dimensions: (3, 3, 32, 64)
const int weights_conv2d_1_kernel_d0 = 3;
const int weights_conv2d_1_kernel_d1 = 3;
const int weights_conv2d_1_kernel_d2 = 32;
const int weights_conv2d_1_kernel_d3 = 64;
const int weights_conv2d_1_kernel_len = 18432;
extern float weights_conv2d_1_kernel[weights_conv2d_1_kernel_len];

// dimensions: (64,)
const int weights_conv2d_1_bias_d0 = 1;
const int weights_conv2d_1_bias_d1 = 1;
const int weights_conv2d_1_bias_d2 = 1;
const int weights_conv2d_1_bias_d3 = 64;
const int weights_conv2d_1_bias_len = 64;
extern float weights_conv2d_1_bias[weights_conv2d_1_bias_len];

// dimensions: (2304, 128)
const int weights_dense_kernel_d0 = 1;
const int weights_dense_kernel_d1 = 1;
const int weights_dense_kernel_d2 = 2304;
const int weights_dense_kernel_d3 = 128;
const int weights_dense_kernel_len = 294912;
extern float weights_dense_kernel[weights_dense_kernel_len];

// dimensions: (128,)
const int weights_dense_bias_d0 = 1;
const int weights_dense_bias_d1 = 1;
const int weights_dense_bias_d2 = 1;
const int weights_dense_bias_d3 = 128;
const int weights_dense_bias_len = 128;
extern float weights_dense_bias[weights_dense_bias_len];

// dimensions: (128, 12)
const int weights_dense_1_kernel_d0 = 1;
const int weights_dense_1_kernel_d1 = 1;
const int weights_dense_1_kernel_d2 = 128;
const int weights_dense_1_kernel_d3 = 12;
const int weights_dense_1_kernel_len = 1536;
extern float weights_dense_1_kernel[weights_dense_1_kernel_len];

// dimensions: (12,)
const int weights_dense_1_bias_d0 = 1;
const int weights_dense_1_bias_d1 = 1;
const int weights_dense_1_bias_d2 = 1;
const int weights_dense_1_bias_d3 = 12;
const int weights_dense_1_bias_len = 12;
extern float weights_dense_1_bias[weights_dense_1_bias_len];

#endif  // __WEIGHTS_H__

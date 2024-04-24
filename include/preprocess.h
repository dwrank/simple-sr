#ifndef __PREPROCESS_H__
#define __PREPROCESS_H__

#include <cmath>
#include <algorithm>

#include "tensor.h"

// from tensorflow/lite/kernels/internal/reference/resize_bilinear.h ComputeInterpolationValues
inline void compute_interpolation_values(
        const float value, const float scale,
        const bool half_pixel_centers,
        int32_t input_size, float* scaled_value,
        int32_t* lower_bound,
        int32_t* upper_bound)
{
    if (half_pixel_centers) {
        *scaled_value = (value + 0.5f) * scale - 0.5f;
    } else {
        *scaled_value = value * scale;
    }

    float scaled_value_floor = std::floor(*scaled_value);

    *lower_bound = std::max(static_cast<int32_t>(scaled_value_floor),
                            static_cast<int32_t>(0));
    *upper_bound =
        std::min(static_cast<int32_t>(std::ceil(*scaled_value)), input_size - 1);
}

// from tensorflow/lite/kernels/internal/reference/resize_bilinear.h ResizeBilinear
template <typename T>
inline void resize_bilinear(Tensor2<T> &in_t,
                            Tensor2<T> &out_t,
                            bool align_corners=false)
{
    int input_height = in_t.rows;
    int input_width = in_t.cols;
    int output_height = out_t.rows;
    int output_width = out_t.cols;

    //printf("shape: %d %d, %d %d\n", input_height, input_width, output_height, output_width);
    // If half_pixel_centers is True, align_corners must be False.
    bool half_pixel_centers = !align_corners;

    float height_scale = static_cast<float>(input_height) / output_height;
    float width_scale = static_cast<float>(input_width) / output_width;

    if (align_corners && output_height > 1) {
        height_scale = static_cast<float>(input_height - 1) / (output_height - 1);
    }

    if (align_corners && output_width > 1) {
        width_scale = static_cast<float>(input_width - 1) / (output_width - 1);
    }

    const float rounding_offset = std::numeric_limits<T>::is_integer ? .5f : .0f;

    for (int y = 0; y < output_height; ++y) {
        float input_y;
        int32_t y0, y1;
        compute_interpolation_values(y, height_scale, half_pixel_centers,
                                     input_height, &input_y, &y0, &y1);

        for (int x = 0; x < output_width; ++x) {
            float input_x;
            int32_t x0, x1;
            compute_interpolation_values(x, width_scale, half_pixel_centers,
                                       input_width, &input_x, &x0, &x1);

            T interpolation = static_cast<T>(
                    *in_t.Data(y0, x0) * (1 - (input_y - y0)) * (1 - (input_x - x0)) +
                    *in_t.Data(y1, x0) * (input_y - y0) * (1 - (input_x - x0)) +
                    *in_t.Data(y0, x1) * (1 - (input_y - y0)) * (input_x - x0) +
                    *in_t.Data(y1, x1) * (input_y - y0) * (input_x - x0) +
                    rounding_offset);

            //if (y == 0 && x < 10) printf("Interp: %f %d %d, %f %d %d\n", input_y, y0, y1, input_x, x0, x1);
            *out_t.Data(y, x) = interpolation;
      }
    }
}

template<typename T>
inline Tensor2<T> resize_tensor2(Tensor2<T> &in_t, int height, int width)
{
    auto out_t = Tensor2f(height, width);

    resize_bilinear(in_t, out_t);

    return out_t;
}

template<typename T>
inline Tensor2f normalize_tensor2(Tensor2<T> &in_t, double mean, double std)
{
    auto out_t = Tensor2f(in_t.rows, in_t.cols);

    for (int i = 0; i < in_t.length; i++) {
        out_t.data[i] = (static_cast<double>(in_t.data[i]) - mean) / std;
    }

    return out_t;
}

#endif //__PREPROCESS_H__

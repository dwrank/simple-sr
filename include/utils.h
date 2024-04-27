#ifndef __UTILS_H__
#define __UTILS_H__

#include <iostream>

#include "tensor.h"

template<typename T>
void print_data(Tensor<T> &tensor)
{
    int len = std::min(tensor.length(), 32);

    printf("\nTrain");
    for (int i = 0; i < len; i++) {
        if (i % 16 == 0) {
            std::cout << std::endl;
        }
        std::cout << tensor[i] << " ";
    }
    std::cout << std::endl;
}

#endif //__UTILS_H__

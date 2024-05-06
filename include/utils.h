#ifndef __UTILS_H__
#define __UTILS_H__

#include <iostream>

#include "tensor.h"

extern bool Debug;
extern bool Verbose;

template<typename T>
void print_data(const char *name, Tensor<T> &tensor)
{
    if (Debug) {
        int len = std::min(tensor.length(), 129);

        printf("\n%s", name);
        for (int i = 0; i < len; i++) {
            if (i % 16 == 0) {
                std::cout << std::endl;
            }
            std::cout << tensor[i] << " ";
        }
        std::cout << std::endl;
    }
}

#endif //__UTILS_H__

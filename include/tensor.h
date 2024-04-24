#ifndef __TENSOR_H__
#define __TENSOR_H__

#include <stdio.h>

// 2-D tensor
template<typename T>
class Tensor2
{
public:
    Tensor2() :
        data(nullptr)
    {
        this->free();
    }

    Tensor2(int r, int c)
    {
        alloc(r, c);
    }

    virtual ~Tensor2() { }//free(); }

    void free()
    {
        rows = 0;
        cols = 0;
        length = 0;

        if (data != nullptr) {
            delete[] data;
            data = nullptr;
        }
    }

    inline void alloc(int r, int c)
    {
        rows = r;
        cols = c;
        length = rows * cols;
        data = new T[length]();
    }

    inline void alloc(int length)
    {
        alloc(1, length);
    }

    inline void realloc(int r, int c)
    {
        this->free();
        alloc(r, c);
    }

    inline void realloc(int length)
    {
        this->free();
        alloc(length);
    }

    inline T* Data(int row, int col)
    {
        int i = row * cols + col;
        if (i >= length) {
            fprintf(stderr, "Tensor: Index out of bounds: %d. Length is %d.\n", i, length);
            i = length - 1;
        }
        return data + i;
    }

    inline bool is_empty() { return data == nullptr; }

    int rows;
    int cols;
    int length;
    T *data;
};

template<typename T>
class Tensor1 : public Tensor2<T>
{
public:
    Tensor1() : Tensor2<T>() {}

    Tensor1(int length) :
        Tensor2<T>(1, length)
    {}

    virtual ~Tensor1() {}
};

using Tensor1s = Tensor1<short>;
using Tensor1f = Tensor1<float>;
using Tensor2f = Tensor2<float>;

#endif //__TENSOR_H__

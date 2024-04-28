#ifndef __TENSOR_H__
#define __TENSOR_H__

#include <stdio.h>
#include <array>
#include <vector>
#include <memory>
#include <sstream>
#include <stdexcept>

// 4-D tensor
template<typename T>
class Tensor
{
public:
    Tensor() :
        data(nullptr)
    {
        this->free();
    }

    Tensor(int d0, int d1, int d2, int d3)
    {
        alloc(d0, d1, d2, d3);
    }

    Tensor(int d1, int d2, int d3) : Tensor(1, d1, d2, d3) {}
    Tensor(int rows, int cols) : Tensor(1, rows, cols) {}
    Tensor(int length) : Tensor(1, 1, length) {}

    virtual ~Tensor() {}

    inline void free()
    {
        std::fill(dims.begin(), dims.end(), 0);
        n_dims = 0;
        len = 0;

        if (data != nullptr) {
            delete[] data;
            data = nullptr;
        }
    }

    inline void alloc(int d0, int d1, int d2, int d3)
    {
        set_dims(d0, d1, d2, d3);
        data = new T[len]();
    }

    inline void alloc(int d1, int d2, int d3)
    {
        alloc(1, d1, d2, d3);
    }

    inline void alloc(int rows, int cols)
    {
        alloc(1, 1, rows, cols);
    }

    inline void alloc(int length)
    {
        alloc(1, 1, 1, length);
    }

    inline void realloc(int d0, int d1, int d2, int d3)
    {
        this->free();
        alloc(d0, d1, d2, d3);
    }

    inline void realloc(int d1, int d2, int d3)
    {
        realloc(1, d1, d2, d3);
    }

    inline void realloc(int rows, int cols)
    {
        realloc(1, 1, rows, cols);
    }

    inline void realloc(int length)
    {
        realloc(1, 1, 1, length);
    }

    inline int pos(int i0, int i1, int i2, int i3) const
    {
        return dims[3] * (dims[2] * (dims[1] * i0 + i1) + i2) + i3;
    }

    inline const T& operator()(int i0, int i1, int i2, int i3) const
    {
        // group_size = dims[1] * dims[2] * dims[3];
        // channel_size = dims[2] * dims[3];
        // row_size = dims[3];
        // col_size = 1;
        // pos = group_size * i0 + channel_size * i1 + row_size * i2 + col_size * i3;
        // pos = (dims[1] * dims[2] * dims[3]) * i0 + (dims[2] * dims[3]) * i1 +
        //       (dims[3]) * i2 + (1) * i3;
        // pos = dims[3] * (dims[2] * (dims[1] * i0 + i1) + i2) + i3;
        
        //int pos = pos(i0, i1, i2, i3);
        int pos = dims[3] * (dims[2] * (dims[1] * i0 + i1) + i2) + i3;

        if (pos >= len) {
            std::stringstream ss;
            ss << "Tensor: Index is out of bounds: " << pos << ". Length is " << len << ".";
            throw std::runtime_error(ss.str());
        }
        return data[pos];
    }

    inline T& operator()(int i0, int i1, int i2, int i3)
    {
        //int pos = pos(i0, i1, i2, i3);
        int pos = dims[3] * (dims[2] * (dims[1] * i0 + i1) + i2) + i3;

        if (pos >= len) {
            std::stringstream ss;
            ss << "Tensor: Index is out of bounds: " << pos << ". Length is " << len << ".";
            throw std::runtime_error(ss.str());
        }
        return data[pos];
    }

    inline const T& operator()(int i1, int i2, int i3) const
    {
        return (*this)(0, i1, i2, i3);
    }

    inline T& operator()(int i1, int i2, int i3)
    {
        return (*this)(0, i1, i2, i3);
    }

    inline const T& operator()(int row, int col) const
    {
        return (*this)(0, 0, row, col);
    }

    inline T& operator()(int row, int col)
    {
        return (*this)(0, 0, row, col);
    }

    inline const T& operator()(int i) const
    {
        return (*this)(0, 0, 0, i);
    }

    inline T& operator()(int i)
    {
        return (*this)(0, 0, 0, i);
    }

    inline const T& operator[](int i) const
    {
        return (*this)(0, 0, 0, i);
    }

    inline T& operator[](int i)
    {
        return (*this)(0, 0, 0, i);
    }

    inline bool is_empty() const { return data == nullptr; }

    inline int d0() const { return dims[0]; }
    inline int d1() const { return dims[1]; }
    inline int d2() const { return dims[2]; }
    inline int d3() const { return dims[3]; }

    inline int groups() const { return d0(); }
    inline int channels() const { return d1(); }
    inline int rows() const { return d2(); }
    inline int cols() const { return d3(); }

    inline int length() const { return len; }

    inline void set_dims(int d0, int d1, int d2, int d3)
    {
        dims[0] = d0;
        dims[1] = d1;
        dims[2] = d2;
        dims[3] = d3;
        len = d0 * d1 * d2 *d3;

        n_dims = 0;
        if (d3) { n_dims++; }
        if (d2 > 1) { n_dims++; }
        if (d1 > 1) { n_dims++; }
        if (d0 > 1) { n_dims++; }
    }

    inline void print_dims(const char *name="") const
    {
        printf("[Tensor] %s dims: (%d, %d, %d, %d) = %d\n", name, d0(), d1(), d2(), d3(), len);
    }

    T *data;

private:
    int len;
    int n_dims;
    std::array<int,4> dims;
};

template<typename T>
inline void free_vec_tensor(std::vector<Tensor<T>> &vec)
{
    for (auto &t : vec) {
        t.free();
    }
}

using Tensor1s = Tensor<short>;
using Tensor1f = Tensor<float>;
using Tensor2f = Tensor<float>;
using Tensor3f = Tensor<float>;
using Tensor4f = Tensor<float>;

#endif //__TENSOR_H__

#ifndef __ACTIVATION_H__
#define __ACTIVATION_H__


template<typename T>
inline T relu(T val)
{
    if (val < 0) { return 0; }
    return val;
}

#endif //__ACTIVATION_H__

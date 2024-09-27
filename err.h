#pragma once

#define CUDA_CALL(f)                                                                     \
    {                                                                                    \
        cudaError_t err = f;                                                             \
        if(err != cudaSuccess){                                                           \
            fprintf(stderr, "%s\n at %s:%d\n", cudaGetErrorName(err), __FILE__, __LINE__);\
            exit(1);\
        }\
    }\

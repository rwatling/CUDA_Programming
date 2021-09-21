#ifndef SHFL_MATCH_H
#define SHFL_MATCH_H 1
#define WARP_SIZE 32

ifndef ARRAY_SIZE
#define ARRAY_SIZE 8
#endif

#include "cuda_includes.h"
__global__ void shfl_match(int* all_arrays, int num_threads);
#endif

#ifndef SHFL_UNROLL2_MATCH_H
#define SHFL_UNROLL2_MATCH_H 1
#define WARP_SIZE 32
#include "cuda_includes.h"
#include "unroll2_match.h"
__global__ void shfl_unroll2_match(int* global_arrays, int num_threads);
#endif

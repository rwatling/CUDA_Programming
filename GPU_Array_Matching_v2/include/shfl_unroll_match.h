#ifndef SHFL_UNROLL_MATCH_H
#define SHFL_UNROLL_MATCH_H 1
#define WARP_SIZE 32
#include "cuda_includes.h"
#include "unroll_match.h"
__global__ void shfl_unroll_match(int* global_arrays, int num_threads);
#endif

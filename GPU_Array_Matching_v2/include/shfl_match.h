#ifndef SHFL_MATCH_H
#define SHFL_MATCH_H 1
#define WARP_SIZE 32
#include "cuda_includes.h"
#include "match.h"
__global__ void shfl_match(int* global_arrays, int num_threads);
#endif

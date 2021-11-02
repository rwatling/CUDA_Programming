#ifndef SHFL_BS_MATCH_H
#define SHFL_BS_MATCH_H 1
#define WARP_SIZE 32
#include "cuda_includes.h"
#include "bs_match.h"
__global__ void shfl_bs_match(int* global_arrays, int num_threads);
#endif

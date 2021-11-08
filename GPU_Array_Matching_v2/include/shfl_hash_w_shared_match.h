#ifndef SHFL_HASH_W_SHARED_MATCH_H
#define SHFL_HASH_W_SHARED_MATCH_H 1
#define WARP_SIZE 32
#include "cuda_includes.h"
__global__ void shfl_hash_w_shared_match(int* global_arrays, int num_threads);
#endif

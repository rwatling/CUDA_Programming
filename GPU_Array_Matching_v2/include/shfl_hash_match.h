#ifndef SHFL_HASH_MATCH_H
#define SHFL_HASH_MATCH_H 1
#define WARP_SIZE 32
#include "cuda_includes.h"
#include "hash_match.h"
__global__ void shfl_hash_match(int* global_arrays, int num_threads);
#endif

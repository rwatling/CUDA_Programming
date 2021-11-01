#ifndef SHM_UNROLL_MATCH_H
#define SHM_UNROLL_MATCH_H 1
#include "cuda_includes.h"
#include "unroll_match.h"
__global__ void shm_unroll_match(int* global_arrays, int num_threads);
#endif

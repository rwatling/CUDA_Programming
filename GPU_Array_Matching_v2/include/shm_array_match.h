#ifndef SHM_ARRAY_MATCH_H
#define SHM_ARRAY_MATCH_H 1
#include "cuda_includes.h"
#include "match.h"
__global__ void shm_array_match(int* global_arrays, int num_threads);
#endif

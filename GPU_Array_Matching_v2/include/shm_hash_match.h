#ifndef SHM_HASH_MATCH_H
#define SHM_HASH_MATCH_H 1
#include "cuda_includes.h"
#include "hash_match.h"
__global__ void shm_hash_match(int* global_arrays, int num_threads);
#endif

#ifndef SHM_ARRAY_MATCH_H
#define SHM_ARRAY_MATCH_H 1
#define MAX_SHM 48000
#define MAX_INTS (MAX_SHM / sizeof(int))
#define ARRAY_SIZE 8
#include "cuda_includes.h"
#include <time.h>
__global__ void shm_array_match(int* global_arrays, int num_arrays);
#endif

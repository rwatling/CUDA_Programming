#ifndef SHM_ARRAY_MATCH_H
#define SHM_ARRAY_MATCH_H 1
#define MAX_SHM 48000
#define MAX_INTS (MAX_SHM / sizeof(int))
#include "cuda_includes.h"
#include <time.h>
__global__ void shm_array_match(int* all_arrays, int* match_array, int* prev_end, int* current_start, int num_arrays, int size);
#endif

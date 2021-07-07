#ifndef SHFL_MATCH_H
#define SHFL_MATCH_H 1
#define WARP_SIZE 32
#define NUM_REGS 4
#include "cuda_includes.h"
#include <time.h>
__global__ void shfl_match(int* all_arrays, int* match_array, int num_arrays,  int size, unsigned long long* elapsed);
#endif

#ifndef ARRAY_MATCH_H
#define ARRAY_MATCH_H 1
#include "cuda_includes.h"
#include <time.h>
__global__ void array_match(int* all_arrays, int* match_array, int num_arrays,  int size, clock_t* elapsed);
#endif

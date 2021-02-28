#ifndef RAND_INIT_H
#define RAND_INIT_H 1
#include "cuda_includes.h"
#include <ctime>
__global__ void rand_init(int* all_arrays, int num_arrays, int size);
#endif

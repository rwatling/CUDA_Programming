#ifndef WARP_MATCH_H
#define WARP_MATCH_H 1

#ifndef WARP_SIZE
  #define WARP_SIZE 32
#endif

#include "cuda_includes.h"
#include "match.h"
__device__ void warp_match(int thread_id, int* current_arr1, int* current_arr2);
#endif

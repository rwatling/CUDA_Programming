#include "warp_match.h"

__device__ void warp_match(int thread_id, int* current_arr1, int* current_arr2) {
  int next_arr1[ARRAY_SIZE];
  int next_arr2[ARRAY_SIZE];
  int size = ARRAY_SIZE;
  unsigned int mask = 0xffffffff;

  for (int delta = 1; delta < WARP_SIZE; delta *= 2) {
    for (int i = 0; i < size; i++) {
      next_arr1[i] = __shfl_down_sync(mask, current_arr1[i], delta, WARP_SIZE);
      next_arr2[i] = __shfl_down_sync(mask, current_arr2[i], delta, WARP_SIZE);
    }

    if ((thread_id % (delta * 2)) == 0) {
      match(current_arr2, next_arr1, next_arr2);
    }

    __syncthreads();
  }
}

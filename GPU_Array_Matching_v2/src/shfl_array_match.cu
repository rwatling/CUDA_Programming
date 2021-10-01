#include "shfl_array_match.h"

__global__ void shfl_array_match(int* global_arrays, int num_threads) {

  int thread_id = (blockIdx.x * blockDim.x) + threadIdx.x;
	extern __shared__ int shared_arrays[];
	int current_arr1[ARRAY_SIZE];
	int current_arr2[ARRAY_SIZE];
	int next_arr1[ARRAY_SIZE];
	int next_arr2[ARRAY_SIZE];
	int size = ARRAY_SIZE;
  int arr1_index = 0;
  int arr2_index = 0;
  unsigned int mask = 0xffffffff;

  //Stage 0: Retrieve from global memory
  for (int i = 0; i < size; i++) {
    arr1_index = (thread_id * 2 * size) + i;
    current_arr1[i] = global_arrays[arr1_index];

    arr2_index = (thread_id * 2 * size) + size + i;
    current_arr2[i] = global_arrays[arr2_index];
  }

  __syncthreads();

  //Stage 1: Match by shuffle arrays
  for (int delta = 1; delta < 32; delta *= 2) {

    for (int i = 0; i < size; i++) {
      next_arr1[i] = __shfl_down_sync(mask, current_arr1[i], delta, WARP_SIZE);
      next_arr2[i] = __shfl_down_sync(mask, current_arr2[i], delta, WARP_SIZE);
    }

    __syncthreads();

    if ((thread_id % (delta * 2)) == 0) {
      match(current_arr2, next_arr1, next_arr2);
    }

    __syncthreads();
  }

  //Stage 2: Write to shared memory
  if ((thread_id % WARP_SIZE) == 0) {
    for(int i = 0; i < size; i++) {
      arr1_index = ((thread_id / WARP_SIZE) * 2 * size) + i;
      shared_arrays[arr1_index] = current_arr1[i];

      arr2_index = ((thread_id / WARP_SIZE) * 2 * size) + size + i;
      shared_arrays[arr2_index] = current_arr2[i];
    }
  }

  __syncthreads();

  //Stage 3: Reload memory
  if (thread_id < WARP_SIZE) {
    for(int i = 0; i < size; i++) {
      arr1_index = (thread_id * 2 * size) + i;
      arr2_index = (thread_id * 2 * size) + size + i;

      current_arr1[i] = shared_arrays[arr1_index];
      current_arr2[i] = shared_arrays[arr2_index];
    }
  }

  __syncthreads();

  //Step 4: Shuffle again
  for (int delta = 1; delta < 32; delta *= 2) {

    if (thread_id < WARP_SIZE) {
      for (int i = 0; i < size; i++) {
        next_arr1[i] = __shfl_down_sync(mask, current_arr1[i], delta, WARP_SIZE);
        next_arr2[i] = __shfl_down_sync(mask, current_arr2[i], delta, WARP_SIZE);
      }
    }

    __syncthreads();

    if (thread_id < WARP_SIZE) {
      if ((thread_id % (delta * 2)) == 0) {
        match(current_arr2, next_arr1, next_arr2);
      }
    }

    __syncthreads();
  }

  //Stage 5: Write back to global memory
  if (thread_id == 0) {
		for (int i = 0; i < size; i++) {
			arr1_index = (thread_id * 2 * size) + i;
			global_arrays[arr1_index] = current_arr1[i];

			arr2_index = (thread_id * 2 * size) + size + i;
			global_arrays[arr2_index] = current_arr2[i];
		}
	}
}

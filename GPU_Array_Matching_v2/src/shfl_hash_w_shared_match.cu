#include "shfl_hash_w_shared_match.h"

__global__ void shfl_hash_w_shared_match(int* global_arrays, int num_threads) {
  int thread_id = (blockIdx.x * blockDim.x) + threadIdx.x;
  __shared__ int shared_arrays[ARRAY_SIZE * WARP_SIZE];
  __shared__ int h_table1[HASH_SIZE][MAX_COLLISIONS];
  __shared__ int h_table2[HASH_SIZE][MAX_COLLISIONS];
	int current_arr1[ARRAY_SIZE];
	int current_arr2[ARRAY_SIZE];
  int next_arr1[ARRAY_SIZE];
  int next_arr2[ARRAY_SIZE];
	int size = ARRAY_SIZE;
  int arr1_index = 0;
  int arr2_index = 0;
  int key = 0;
  int hashed_key = 0;
  unsigned int mask = 0xffffffff;

  //Stage 0: Retrieve from global memory
  for (int i = 0; i < size; i++) {
    arr1_index = (thread_id * 2 * size) + i;
    current_arr1[i] = global_arrays[arr1_index];

    arr2_index = (thread_id * 2 * size) + size + i;
    current_arr2[i] = global_arrays[arr2_index];
  }

  //Stage 1: Match by shuffle arrays with tree like reduction
  for (int delta = 1; delta < WARP_SIZE; delta = delta << 1) {

    //Retrieve value from register from thread_id + delta
    for (int i = 0; i < size; i++) {
      next_arr1[i] = __shfl_down_sync(mask, current_arr1[i], delta, WARP_SIZE);
      next_arr2[i] = __shfl_down_sync(mask, current_arr2[i], delta, WARP_SIZE);
    }

    if ((thread_id % (delta * 2)) == 0) {
      hash_match(current_arr2, next_arr1, next_arr2);
    }
  }

  if (num_threads > WARP_SIZE) {

    //Stage 2: Warp thread 0 write warp shuffle result to shared memory
    if ((thread_id % WARP_SIZE) == 0) {

      for(int i = 0; i < size; i++) {
        arr1_index = ((thread_id / WARP_SIZE) * 2 * size) + i;
        shared_arrays[arr1_index] = current_arr1[i];
      }

      for(int i = 0; i < size; i++) {
        arr2_index = ((thread_id / WARP_SIZE) * 2 * size) + size + i;
        shared_arrays[arr2_index] = current_arr2[i];
      }
    }

    __syncthreads();

    //Stage 3: Read all warps 0 thread from shared memory
    if (thread_id < WARP_SIZE) {
      for(int i = 0; i < size; i++) {
        arr1_index = (thread_id * 2 * size) + i;
        current_arr1[i] = shared_arrays[arr1_index];
      }

      for(int i = 0; i < size; i++) {
        arr2_index = (thread_id * 2 * size) + size + i;
        current_arr2[i] = shared_arrays[arr2_index];
      }
    }

    __syncthreads();

    //Stage 4: Shuffle again
    if (thread_id < WARP_SIZE) {

      // Tree like reduction, notice for loop condition
      for (int delta = 1; delta < (num_threads / WARP_SIZE); delta = delta << 1) {

        //Retrieve value from register from thread_id + delta
        for (int i = 0; i < size; i++) {
          next_arr1[i] = __shfl_down_sync(mask, current_arr1[i], delta, WARP_SIZE);
          next_arr2[i] = __shfl_down_sync(mask, current_arr2[i], delta, WARP_SIZE);
        }

        if ((thread_id % (delta * 2)) == 0) {
          //Hash match
          //Hash tables are not garunteed to be 0
          for (int i = 0; i < HASH_SIZE; i++) {
            for (int j = 0; j < MAX_COLLISIONS; j++) {
              h_table1[i][j] = 0;
            }
          }

          //Hash "next" arrays
          for (int i = 0; i < ARRAY_SIZE; i++) {
            key = next_arr1[i];
            hashed_key = hash(key);

            if (h_table1[hashed_key][0] == 0) {
              h_table1[hashed_key][0] = next_arr1[i];
              h_table2[hashed_key][0] = next_arr2[i];
            } else if (h_table1[hashed_key][1] == 0) {
              h_table1[hashed_key][1] = next_arr1[i];
              h_table2[hashed_key][1] = next_arr2[i];
            }
          }

          //Find values
          for (int i = 0; i < ARRAY_SIZE; i++) {
            key = current_arr2[i];
            hashed_key = hash(key);

            //array2[i] = h_table2[hashed_key][0];

            if (key == h_table1[hashed_key][0]) {
              current_arr2[i] = h_table2[hashed_key][0];
            } else if (key == h_table1[hashed_key][1]) {
              current_arr2[i] = h_table2[hashed_key][1];
            }
          }
        }
      }
    }
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

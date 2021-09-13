#include "shm_array_match.h"

__global__ void shm_array_match(int* global_arrays, int num_arrays) {

	//Essential variables
	int thread_id = (blockIdx.x * blockDim.x) + threadIdx.x;
	extern __shared__ int shared_arrays[];
	int local_reg_arr1[ARRAY_SIZE];
	int local_reg_arr2[ARRAY_SIZE];
	int next_reg_arr1[ARRAY_SIZE];
	int next_reg_arr2[ARRAY_SIZE];
	int size = ARRAY_SIZE;

	//Retrieve global values from major operation
	//Assign the global values to registers for array_1 and array_2
	//Assign the initial global values to shared memory
	for (int i = 0; i < size; i++) {
		int arr1_index = (thread_id * size) + i;
		int arr2_index = (thread_id * size) + size + i;

		local_reg_arr1[i] = global_arrays[arr1_index];
		local_reg_arr2[i] = global_arrays[arr2_index];
		shared_arrays[arr1_index] = local_reg_arr1[i];
		shared_arrays[arr2_index] = local_reg_arr2[i];
	}

	__syncthreads();

	//Tree like grouping for matching operations
	for (int group_size = 2; group_size < num_arrays; group_size *= 2) {

		//Load "next" arrays from shared memory
		for (int i = 0; i < size; i++) {
			int shm_next_arr1_index = (thread_id + group_size) * size + i;
			int shm_next_arr2_index = (thread_id + group_size) * size + size + i;

			next_reg_arr1[i] = shared_arrays[shm_next_arr1_index];
			next_reg_arr2[i] = shared_arrays[shm_next_arr2_index];
		}

		//Matching operation
		for (int i = 0; i < size; i++) {
			int match = 0;

			for (int j = 0; j < size; j++) {
				if (local_reg_arr2[i] == next_reg_arr1[j]) {
					local_reg_arr2[i] = next_reg_arr2[j];
					match = 1;
				}
			}

			if (!match) {
				local_reg_arr2[i] *= -1;
			}
		}

		__syncthreads();
	}
}

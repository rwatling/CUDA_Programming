#include "shm_array_match.h"

__global__ void shm_array_match(int* all_arrays, int* match_array, int* global_arr1, int* global_arr2, int num_arrays, int size) {

	//Essential variables
	int thread_id = (blockIdx.x * blockDim.x) + threadIdx.x;
	int log_num_arrays = (int) log2((float) num_arrays);
	int match;
	int next_reg_arr1[ARRAY_SIZE];
	int local_reg_arr1[ARRAY_SIZE];
	int local_reg_arr2[ARRAY_SIZE];
	extern __shared__ int shared_arrays[];

	for (int group_size = 2; group_size < log_num_arrays; group_size = group_size * 2) {
		for (int i = 0; i < size; i++) {
			local_reg_arr1[i] = global_arr1[i];
		}

		if (group_size == 2) {
			for (int i = 0; i < size; i++) {
				local_reg_arr2[i] = global_arr2[i];
			}
		}

		if ((thread_id % group_size) == (group_size - 1)) {
			for (int i = 0; i < size; i++) {
				shared_arrays[(thread_id * size) + i] = local_reg_arr1[i];
			}

			for (int i = 0; i < size; i++) {
				shared_arrays[((thread_id + size) * size) + i] = local_reg_arr2[i];
			}
		}

		__syncthreads();

		if ((thread_id % group_size) == 0) {
			for (int i = 0; i < size; i++) {
				next_reg_arr1[i] = shared_arrays[(thread_id + group_size) * size + i];
			}

			for (int i = 0; i < size; i++) {
				for (int j = 0; j < size; j++) {
					if (local_reg_arr2[i] == next_reg_arr1[j]) {
						local_reg_arr2[i] = next_reg_arr1[j];
						match = 1;
					}
				}
			}

			if (match == 0) {
				//then stop?
			}

			__syncthreads();
		}
	}
}

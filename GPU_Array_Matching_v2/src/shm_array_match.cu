#include "shm_array_match.h"

__global__ void shm_array_match(int* all_arrays, int* match_array, int* prev_end, int* current_start, int num_arrays, int size) {

	//Essential variables
	int thread_id = (blockIdx.x * blockDim.x) + threadIdx.x;
	int check;
	int log_num_arrays = (int) log2((float) num_arrays);
	extern __shared__ int shared_arrays[];

	for (int i = 0; i < num_arrays * 2; i++) {
		for (int j = 0; j < num_arrays * 2; j++) {
			shared_arrays[(i * size) + j] = all_arrays[(i * size) + j];
		}
	}

	__syncthreads();

	for (int group_size = 1; group_size < log_num_arrays; group_size = group_size << 1) {
		int index = 0;

		for (int i = 0; i < size; i++) {
			index = ((thread_id - 1) / group_size) + (((thread_id - 1) % group_size) * size);
			prev_end[i] = shared_arrays[(index * size) + size + i];
		}

		for (int i = 0; i < size; i++) {
			check = current_start[i];

			for (int j = 0; j < size; j++) {
				if (prev_end[j] == check) {
					match_array[thread_id] = 1;
				}
			}
		}
		__syncthreads();
	}
}

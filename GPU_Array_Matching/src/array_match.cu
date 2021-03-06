#include "array_match.h"

__global__ void array_match(int* all_arrays, int* match_array, int num_arrays, int size, unsigned long long* elapsed) {

	// Essential variables
	int thread_id = (blockIdx.x * blockDim.x) + threadIdx.x;
	unsigned long long start;
	unsigned long long stop;

	//For random number generation
	int maxRand = 100;
	curandState state;
	unsigned long long seed = clock();

	curand_init(seed + thread_id, 0, 0, &state);

	if (thread_id >= num_arrays) { return; }

	int* current_array = all_arrays + (thread_id * size); //Pointer arithmetic
	int* prev_array = all_arrays + ((thread_id - 1) * size); //Pointer arithmetic

	for (int i = 0; i < size; i++) {
			//At runtime moment, generate random number
			current_array[i] = (int) (curand_uniform(&state) * maxRand);
	}

	__syncthreads();

	if (thread_id == 0) {
		start = clock();
	}

	if (thread_id > 0) {
		int match = 0;

		for (int i = 0; i < size; i++) {
			for (int j = 0; j < size; j++) {
				if (current_array[i] == prev_array[j]) {
					match = 1;
				}
			}
		}

		match_array[thread_id] = match;
	}
	__syncthreads();

	if (thread_id == 0) {
		stop = clock();
		*elapsed = stop - start;
	}
}

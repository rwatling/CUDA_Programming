#include "shm_array_match.h"

__global__ void shm_array_match(int* all_arrays, int* match_array, int num_arrays,  int size) {
	int thread_id = (blockIdx.x * blockDim.x) + threadIdx.x;

	//Declare current and previous arrays
	extern __shared__ int current_arr[];
	int* prev_arr = all_arrays + ((thread_id - 1) * size);
	int* g_current_arr = all_arrays + (thread_id * size);

	//For random number generation
	int maxRand = 100;
	curandState state;
	unsigned long long seed;

	//Initialize random numbers
	seed = clock();
	curand_init(seed + thread_id, 0, 0, &state);

	if (thread_id >= num_arrays) { return; }

	int step = thread_id * size;

	//Get random array element
	for (int i = 0; i < size; i++) {
		int rand_num = (int) (curand_uniform(&state) * maxRand);
		current_arr[step+i] = rand_num;
		g_current_arr[i] = rand_num;
	}

	__syncthreads();

	match_array[thread_id] = 0;

	//find matches using shared current and global previous
	if (thread_id > 0) {
		int match = 0;

		for (int i = 0; i < size; i++) {
			for (int j = 0; j < size; j++) {
				if (current_arr[step+i] == prev_arr[j]) {
					match_array[thread_id] = 1;
					match = 1;
					break;
				}
			}

			if (match) {
				break;
			}
		}
	}
}

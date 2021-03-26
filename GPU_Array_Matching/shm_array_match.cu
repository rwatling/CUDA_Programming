#include "shm_array_match.h"

__global__ void shm_array_match(int* all_arrays, int* match_array, int num_arrays, int size) {
	int thread_id = (blockIdx.x * blockDim.x) + threadIdx.x;

	//Declare current and previous arrays
	extern __shared__ int shared_arrays[];
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

	// MAX_SHM = 48 Kb of memory is the max shared memory size
	// MAX_INT = ~12000 integers assuming sizeof(int) = 4
	int my_pass = ((thread_id + 1) * size) / MAX_INTS;
	int total_pass = (num_arrays * size) / MAX_INTS;
	int step = thread_id * size;

	for (int pass = 0; pass <= total_pass; pass++) {

		__syncthreads();

		if (pass == my_pass) {

			for (int i = 0; i < size; i++) {
				int rand_num = (int) (curand_uniform(&state) * maxRand);

				shared_arrays[(step+i) % MAX_INTS] = rand_num;
				g_current_arr[i] = rand_num;
			}
		}

		//Wait for threads in my block to write to global
		__syncthreads();

		if (pass == my_pass) {

			//find matches using shared current and global previous
			if (thread_id > 0) {
				int match = 0;

				for (int i = 0; i < size; i++) {
					for (int j = 0; j < size; j++) {

						if (shared_arrays[(step+i) % MAX_INTS] == prev_arr[j]) {
							match = 1;
						}

					}
				}

				match_array[thread_id] = match;
			}
		}
	}
}

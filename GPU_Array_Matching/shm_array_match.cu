#include "shm_array_match.h"

__global__ void shm_array_match(int* all_arrays, int* match_array, int num_arrays,  int size) {
	int thread_id = (blockIdx.x * blockDim.x) + threadIdx.x;

	//Declare current and previous arrays
	extern __shared__ int current_arr[];
	int* prev_arr = all_arrays + ((thread_id - 1) * size);
	int* g_current_arr = all_arrays + (thread_id * size);
	int match;


	//For random number generation
	int maxRand = 100;
	curandState state;
	unsigned long long seed;

	//Initialize random numbers
	seed = clock();
	curand_init(seed + thread_id, 0, 0, &state);

	if (thread_id > num_arrays) { return; }

	if (thread_id > 0) {

		//Get random array element
		for (int i = 0; i < size; i++) {
			for (int j = 0; j < size; j++) {
				int rand_num = (int) (curand_uniform(&state) * maxRand);
				current_arr[i] = rand_num;
				g_current_arr[i] = rand_num; //Not certain this is necessary
			}
		}

	} else {

		//Get random array element
		for (int i = 0; i < size; i++) {
			int rand_num = (int) (curand_uniform(&state) * maxRand);
			current_arr[i] = rand_num;
			g_current_arr[i] = rand_num;
		}
	}

	__syncthreads();

	//find matches using shared current and global previous
	//update match if found
	match = 0;

	if (thread_id > 0) {
		for (int i = 0; i < size; i++) {
			for (int j = 0; j < size; j++) {
				if (current_arr[i] == prev_arr[j]) {
					match = 1;
					break;
				}
			}

			if (match) {
				match_array[thread_id] = match;
				break;
			}
		}
	}

	__syncthreads();
}

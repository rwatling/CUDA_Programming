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

	//48 Kb of memory is the max shared memory size
	// ~12000 integers assuming sizeof(int) = 4
	int total_shm_space = MAX_SHM / sizeof(int);

	//Step -> ints before start of thread array
	//(Step+size)-> ints at end of start array
	int step = thread_id * size;
	int turn = ((step + size) / total_shm_space); //Number of times we will overwrite shared memory
	int loop = 0;

	//For small arrays, pass = 0
	//For larger arrays we will overwrite previously used shared memory
	while (loop <= turn) {

		if (loop == turn) {

			for (int i = 0; i < size; i++) {
				int rand_num = (int) (curand_uniform(&state) * maxRand);

				current_arr[(step+i) % total_shm_space] = rand_num;
				g_current_arr[i] = rand_num;
			}

			__syncthreads();

			match_array[thread_id] = 0;

			//find matches using shared current and global previous
			if (thread_id > 0) {
				int match = 0;

				for (int i = 0; i < size; i++) {
					for (int j = 0; j < size; j++) {

						if (current_arr[(step+i) % total_shm_space] == prev_arr[j]) {
							match_array[thread_id] = 1;
							match = 1;
							break;
						}

					}

					if (match) { break; }
				}
			}

		}

		loop++;

		//Wait for others to finish writing to shared memory before current threads turn
		__syncthreads();
	}

}

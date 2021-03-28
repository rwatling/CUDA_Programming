#include "shm_array_match.h"

__global__ void shm_array_match(int* all_arrays, int* match_array, int num_arrays, int size, int debug, clock_t* elapsed) {

	//Essential variables
	int thread_id = (blockIdx.x * blockDim.x) + threadIdx.x;
	int match;
	clock_t start;

	//Declare current and previous arrays
	extern __shared__ int shared_arrays[];
	extern __shared__ int boundary_prev[];
	int* g_current_arr = all_arrays + (thread_id * size);

	//Row of 2D array represented as 1D array
	int my_step = thread_id * size;
	int prev_step = (thread_id - 1) * size;

	// MAX_SHM = 48 Kb of memory is the max shared memory size
	// MAX_INT = ~12000 integers assuming sizeof(int) = 4
	// Defined in shm_array_match.h
	int my_pass = ((thread_id) * size) / MAX_INTS;
	int next_thread_id_pass = ((thread_id + 1) * size) / MAX_INTS;
	int total_pass = (num_arrays * size) / MAX_INTS;

	//If I am a thread that is right next to boundary
	//This allows the next pass to read my information for matching
	bool boundary = (my_pass == next_thread_id_pass - 1);

	//For random number generation
	int maxRand = 100;
	curandState state;
	unsigned long long seed;

	//Initialize random numbers
	seed = clock();
	curand_init(seed + thread_id, 0, 0, &state);

	if (thread_id >= num_arrays) { return; }

	//If debug don't use global memory for arrays and report time
	if (!debug) {
		for (int pass = 0; pass <= total_pass; pass++) {

			__syncthreads();

			if ((pass == my_pass) && boundary) {
				for (int i = 0; i < size; i++) {
					int rand_num = (int) (curand_uniform(&state) * maxRand);

					shared_arrays[(my_step+i) % MAX_INTS] = rand_num;
					boundary_prev[i] = rand_num;
				}
			} if (pass == my_pass) {

				for (int i = 0; i < size; i++) {
					int rand_num = (int) (curand_uniform(&state) * maxRand);
					shared_arrays[(my_step+i) % MAX_INTS] = rand_num;
				}
			}

			//Wait for threads to write
			__syncthreads();

			if (thread_id == 0) { start = clock(); }

			//Find match
			match = 0;

			if ((pass == my_pass) && boundary) {

				for (int i = 0; i < size; i++) {
					for (int j = 0; j < size; j++) {

						if (shared_arrays[(my_step+i) % MAX_INTS] == boundary_prev[j]) { match = 1;}

					}
				}

				match_array[thread_id] = match;

			} else if ((pass == my_pass) && (thread_id > 0)) {

				for (int i = 0; i < size; i++) {
					for (int j = 0; j < size; j++) {

						if (shared_arrays[(my_step+i) % MAX_INTS] == shared_arrays[(prev_step+j) % MAX_INTS]) { match = 1;}

					}
				}

				match_array[thread_id] = match;

			} else if (thread_id == 0) {
				*elapsed += (clock() - start);
			}
		}

	//If debug use global
	} else if (debug) {
		for (int pass = 0; pass <= total_pass; pass++) {

			__syncthreads();

			if ((pass == my_pass) && boundary) {
				for (int i = 0; i < size; i++) {
					int rand_num = (int) (curand_uniform(&state) * maxRand);

					shared_arrays[(my_step+i) % MAX_INTS] = rand_num;
					boundary_prev[i] = rand_num;
					g_current_arr[i] = rand_num;
				}
			} if (pass == my_pass) {

				for (int i = 0; i < size; i++) {
					int rand_num = (int) (curand_uniform(&state) * maxRand);
					shared_arrays[(my_step+i) % MAX_INTS] = rand_num;
					g_current_arr[i] = rand_num;
				}
			}

			//Wait for threads to write
			__syncthreads();

			if (thread_id == 0) { start = clock(); }

			//Find match
			match = 0;

			if ((pass == my_pass) && boundary) {

				for (int i = 0; i < size; i++) {
					for (int j = 0; j < size; j++) {

						if (shared_arrays[(my_step+i) % MAX_INTS] == boundary_prev[j]) { match = 1;}

					}
				}

				match_array[thread_id] = match;

			} else if ((pass == my_pass) && (thread_id > 0)) {

				for (int i = 0; i < size; i++) {
					for (int j = 0; j < size; j++) {

						if (shared_arrays[(my_step+i) % MAX_INTS] == shared_arrays[(prev_step+j) % MAX_INTS]) { match = 1;}

					}
				}

				match_array[thread_id] = match;

			} else if (thread_id == 0) {
				*elapsed += (clock() - start);
			}
		}

	}
}

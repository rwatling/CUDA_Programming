#include "shfl_match.h"

__global__ void shfl_match(int* all_arrays, int* match_array, int num_arrays, int size, unsigned long long* elapsed) {

	// Essential variables
	int thread_id = (blockIdx.x * blockDim.x) + threadIdx.x;
	int lane_id = threadIdx.x % WARP_SIZE;
	int match = 0;
	int prev_num = 0;
	int current_num = 0;
	int mask = 0xfffffff;

	int* g_current_arr = all_arrays + (thread_id * size); //Pointer arithmetic
	int* g_prev_arr = all_arrays + ((thread_id - 1) * size); //Pointer arithmetic

	int local_arr[64];

	// Timing information
	//unsigned long long start;
	//unsigned long long stop;

	//For random number generation
	int maxRand = 100;
	curandState state;
	unsigned long long seed = clock();

	curand_init(seed + thread_id, 0, 0, &state);

	if (thread_id >= num_arrays) { return; }

	/*if (thread_id == 0) {
		start = clock();
	}*/


	/* Needs:
	-zero boundary
	-global memory write
	-compilation
	*/

	// Current just psuedo code
	for (int i = 0; i < 2; i ++) {
		if (thread_id % 2 == 0) {
			if (i == 0) {
				//Comparison
				for (int i = 0; i < size; i++) {
					for (int j = 0; j < size; j++) {
						current_num = (int) (curand_uniform(&state) * maxRand);
						local_arr[j] = current_num;

						prev_num = __shfl_sync(mask, current_num, lane_id - 1);

						//ignore result
					}
				}
			} else if (i == 1) {
				//Comparison
				for (int i = 0; i < size; i++) {
					current_num = local_arr[i];

					for (int j = 0; j < size; j++) {
						prev_num = __shfl_sync(mask, current_num, lane_id - 1);

						if (current_num == prev_num) {
							match_array[thread_id] = 1;
						}
					}
				}
			}
		} else if (thread_id % 2 == 1) {
			if (i == 0) {

				//Comparison
				for (int i = 0; i < size; i++) {
					local_arr[i] = (int) (curand_uniform(&state) * maxRand);;
					current_num = local_arr[i];

					for (int j = 0; j < size; j++) {
						prev_num = __shfl_sync(mask, current_num, lane_id - 1);

						if (current_num == prev_num) {
							match_array[thread_id] = 1;
						}
					}
				}
			} else if (i == 1) {
				//Comparison
				for (int i = 0; i < size; i++) {
					for (int j = 0; j < size; j++) {
						current_num = local_arr[j];

						prev_num = __shfl_sync(mask, current_num, lane_id - 1);

						//ignore result
					}
				}
			}
		}
	}

	//Old code
	/*match_array[thread_id] = 0;
	for (int pass = 0; pass <= total_pass; pass++) {

		__syncthreads();

		//Generate num_regs random numbers
		for (int i = 0; i < NUM_REGS; i++) {
			current_num = (int) (curand_uniform(&state) * maxRand);
			g_current_arr[total_count++] = current_num;
		}

		// Compare num_regs at a time?
		for (int i = 0; i < NUM_REGS; i++) {
			for (int j = 0; j < NUM_REGS; j++) {
				prev_num = __shfl_sync(0xfffffff, current_num, lane_id + 1);
				match_array[thread_id] = current_num;
			}
		}
	}*/


	__syncthreads();



	if (thread_id == 0) {
		*elapsed = 0;
	}
}

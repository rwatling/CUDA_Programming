#include "shfl_match.h"

__global__ void shfl_match(int* all_arrays, int* match_array, int num_arrays, int size, unsigned long long* elapsed) {

	// Essential variables
	int thread_id = (blockIdx.x * blockDim.x) + threadIdx.x;
	int lane_id = threadIdx.x % WARP_SIZE;
	//int match = 0;
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
	-zero boundary warp_size - 1 thread and 0
	-Larger problem sizes -> registers
	-timing information
	*/

	// Variation of even odd sort
	for (int i = 0; i < 2; i++) {

		// Even pass
		if (i == 0) {

			/* Even thread generates its random numbers in local array
			*  It participates in the shuffle for synchronization but ignores the result
			*/
			if (thread_id % 2 == 0) {

				//Comparison
				for (int i = 0; i < size; i++) {
					for (int j = 0; j < size; j++) {

						// First pass of outer loop
						if (i == 0) {
							current_num = (int) (curand_uniform(&state) * maxRand);
							g_current_arr[j] = current_num;
							local_arr[j] = current_num;

						// All other passes
						} else {
							current_num = local_arr[j];
						}

						prev_num = __shfl_sync(mask, current_num, lane_id - 1);

						//ignore result
					}
				}

			/*  Odd thread generates its random numbers in local array
			 *  It participates in the shuffle, comparing its result
			 */
			} else if (thread_id % 2 == 1) {

				//Comparison
				for (int i = 0; i < size; i++) {

					current_num = (int) (curand_uniform(&state) * maxRand);
					g_current_arr[i] = current_num;
					local_arr[i] = current_num;

					for (int j = 0; j < size; j++) {

						prev_num = __shfl_sync(mask, current_num, lane_id - 1);

						if (current_num == prev_num) {
							match_array[thread_id] = 1;
						}

					}
				}

			}

		// Odd pass
		} else if (i == 1) {

			// Even thread
			if (thread_id % 2 == 0) {

				// Compares each previously generated even element to
				// Previously generated odd element
				for (int i = 0; i < size; i++) {

					current_num = local_arr[i];

					for (int j = 0; j < size; j++) {

						prev_num = __shfl_sync(mask, current_num, lane_id - 1);

						if ((current_num == prev_num) && (lane_id != 0)) {
							match_array[thread_id] = 1;
						}

					}
				}

			// Odd thread
			} else if (thread_id % 2 == 1) {

				// Compared to each previously generated even element
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

	__syncthreads();



	if (thread_id == 0) {
		*elapsed = 0;
	}
}

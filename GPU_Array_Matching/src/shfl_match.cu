#include "shfl_match.h"

__global__ void shfl_match(int* match_array, int num_arrays, int size, unsigned long long* elapsed) {

	// Essential variables
	int thread_id = (blockIdx.x * blockDim.x) + threadIdx.x;
	int lane_id = threadIdx.x % WARP_SIZE;
	int current_num;
	int prev_num;
	unsigned long long start;
	unsigned long long stop;

	//For lane0 to get lane31 data
	//__shared__ int shared_arrays[WARP_SIZE];

	//For random number generation
	int maxRand = 100;
	curandState state;
	unsigned long long seed = clock();

	curand_init(seed + thread_id, 0, 0, &state);

	if (thread_id >= num_arrays) { return; }

	if (thread_id == 0) {
		start = clock();
	}

		int match = 0;

	for (int i = 0; i < size; i++) {
		__syncthreads();

		current_num = (int) (curand_uniform(&state) * maxRand);
		prev_num = __shfl_sync(0xffffffff, current_num, lane_id - 1);

		//if (lane_id == WARP_SIZE - 1) {
			//shared_arrays[thread_id / WARP_SIZE] = currentNum;
		//}

		if (current_num == prev_num) {
			match_array[thread_id] = 1;
			match = 1;
		}

		match_array[thread_id] = match;
	}

	__syncthreads();

	if (thread_id == 0) {
		stop = clock();
		*elapsed = stop - start;
	}
}

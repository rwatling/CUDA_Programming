#include "rand_init.h"

__global__ void rand_init(int* all_arrays, int num_arrays, int size) {
	int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
	int maxRand = 10;
	curandState state;
	unsigned long long seed = clock();

	curand_init(seed + thread_id, 0, 0, &state);

	all_arrays[thread_id] = (int) (curand_uniform(&state) * maxRand);
}

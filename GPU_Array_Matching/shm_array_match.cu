#include "shm_array_match.h"

__global__ void shm_array_match(int* all_arrays, int* match_array, int num_arrays,  int size) {
	int thread_id = (blockIdx.x * blockDim.x) + threadIdx.x;

	//Declare shared arrays
	extern __shared__ int shared_arrays[];

	//For random number generation
	int maxRand = 100;
	curandState state;
	unsigned long long seed = clock();

	if (thread_id > num_arrays) { return; }

	//Copy all_arrays segment current and previous segment to shared_arrays
	int* temp_all = all_arrays + ((thread_id - 1) * size);

	for (int i = 0; i < 2; i++) {
		for (int j = 0; j < size; j++) {
			//shared_arrays[j + (i * size)]  = temp_all[j + (i * size)];
			shared_arrays[j + (i * size)] = 3;
		}
	}

	__syncthreads();

	for (int i = 0; i < size; i++) {
			all_arrays[i+ (thread_id * size)] = shared_arrays[i + size];	
	}

	//Initialize random numbers
	/*curand_init(seed + thread_id, 0, 0, &state);

	//Shared memory locations for current and previous
	int* current_array = shared_arrays + size ; //Pointer arithmetic
	int* prev_array = shared_arrays; //Pointer arithmetic

	int match = 0;

	if (thread_id > 0) {

		for (int i = 0; i < size; i++) {
			for (int j = 0; j < size; j++) {

				//At runtime moment, generate random number
				int rand_num = (int) (curand_uniform(&state) * maxRand);;
				current_array[i] = rand_num;

				//Check if previous match
				if (rand_num == prev_array[j]) {
					match = 1;
				}
			}
		}

		match_array[thread_id] = match;

	} else if (thread_id == 0) {
		
		for (int i = 0; i < size; i++) {
			
			//At runtime moment, generate random number
			current_array[i] = (int) (curand_uniform(&state) * maxRand);
		}
	}

	__syncthreads();

	//Copy just the current array back to global memory
	temp_all = all_arrays + (thread_id * size);

	for (int i = 0; i < size; i++) {
			temp_all[i] = shared_arrays[i + size];	
	}*/
}

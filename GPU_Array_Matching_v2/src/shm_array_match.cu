#include "shm_array_match.h"

__global__ void shm_array_match(int* global_arrays, int num_threads) {

	int thread_id = (blockIdx.x * blockDim.x) + threadIdx.x;
	extern __shared__ int shared_arrays[];
	int current_arr1[ARRAY_SIZE];
	int current_arr2[ARRAY_SIZE];
	int next_arr1[ARRAY_SIZE];
	int next_arr2[ARRAY_SIZE];
	int size = ARRAY_SIZE;
	int arr1_index = 0;
	int arr2_index = 0;

	//Retrieve global values from major operation
	//Assign the global values to registers for array_1 and array_2
	//Assign the initial global values to shared memory
	for (int i = 0; i < size; i++) {
		int arr1_index = (thread_id * 2 * size) + i;
		current_arr1[i] = global_arrays[arr1_index];
		shared_arrays[arr1_index] = current_arr1[i];

		int arr2_index = (thread_id * 2 * size) + size + i;
		current_arr2[i] = global_arrays[arr2_index];
		shared_arrays[arr2_index] = current_arr2[i];
	}

	__syncthreads();

	//Re implement
	//1 and 3 write to shared_array
	//Then 0 and 2 read from shared

	//Tree style grouping for threads
	for (int k = 1; k < num_threads; k *= 2) {

		//Condition: thread_id % 2^k+1
		if ((thread_id % (k * 2)) == 0) {

			//Step 1: Read next_arr1 and next_arr2 from shared memory
			for (int i = 0; i < size; i++) {

				//index = (thread_id + 2^k) * 16 + i if array_size = 8
				int arr1_index = (thread_id + k) * (2 * size) + i;
				next_arr1[i] = shared_arrays[arr1_index];

				//index = (thread_id + 2^k) * 16 + 8 + i if array_size = 8
				arr2_index = (thread_id + k) * (2 * size) + size + i;
				next_arr2[i] = shared_arrays[arr2_index];
			}

			//Step 2: Find the match
			match(current_arr2, next_arr1, next_arr2);
		}

		__syncthreads();

		//Step 3: Write back to shared memory
		if ((thread_id % (k * 2)) == 0) {
			for (int i = 0; i < size; i++) {
				arr2_index = (thread_id * 2 * size) + size + i;
				shared_arrays[arr2_index] = current_arr2[i];
			}
		}

		__syncthreads();
	}

	//Write shared memory to global memory for verification
	if (thread_id == 0) {
		for (int i = 0; i < 2 * size; i++) {
			arr1_index = (thread_id * 2 * size) + i;
			global_arrays[arr1_index] = shared_arrays[arr1_index];
		}
	}

}

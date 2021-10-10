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

		int arr2_index = (thread_id * 2 * size) + size + i;
		current_arr2[i] = global_arrays[arr2_index];
	}

	__syncthreads();

	// Tree like match reduction using shared memory
	for (int k = 1; k < num_threads; k = k << 2) {

		// If thread is a writer
		if ((thread_id % (k * 2)) == k) {

			//Write my first array to shared memory for communication
			for (int i = 0; i < size; i++) {
				arr1_index = (thread_id / (k * 2)) * 2 * size + i;
				shared_arrays[arr1_index] = current_arr1[i];
			}

			//Write my second array to shared memory for communication
			for (int i = 0; i < size; i++) {
				arr2_index = (thread_id / (k * 2)) * 2 * size + size + i;
				shared_arrays[arr2_index] = current_arr2[i];
			}
		}

		__syncthreads();

		// If thread is a reader
		if ((thread_id % (k * 2) == 0)) {

			//Read my writers first array
			for (int i = 0; i < size; i++) {
				arr1_index = (thread_id / (k * 2)) * 2 * size + i;
				next_arr1[i] = shared_arrays[arr1_index];
			}

			//Read my writers second array
			for (int i = 0; i < size; i++) {
				arr2_index = (thread_id / (k * 2)) * 2 * size + size + i;
				next_arr2[i] = shared_arrays[arr2_index];
			}

			match(current_arr2, next_arr1, next_arr2);
		}

		__syncthreads();
	}

	//Write shared memory to global memory for verification
	if (thread_id == 0) {
		for (int i = 0; i < size; i++) {
			arr1_index = (thread_id * 2 * size) + i;
			global_arrays[arr1_index] = current_arr1[i];

			arr2_index = (thread_id * 2 * size) + size + i;
			global_arrays[arr2_index] = current_arr2[i];
		}
	}

}

/* This program searches "states" for matches in their arrays using CUDA
*  
* Author: Robbie Watling
*/

# include <cuda.h>
# include <cuda_runtime_api.h>
# include <device_launch_parameters.h>
# include <iostream>
# include <vector>
# include <utility>
# include <iostream>

__global__ void array_match(int* all_arrays, int* match_array, int size) {
	int thread_id = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (thread_id > 0 && thread_id < size) {
		int* current_array = &all_arrays[thread_id];
		int* prev_array = &all_arrays[thread_id];
		int match = 0;

		for (int i = 0; i < size; i++) {
			for (int j = 0; j < size; j++) {
				if (current_array[i] == prev_array[j]) {
					match = 1;
				}
			}
		}

		match_array[thread_id] = match;
	}
}

using namespace std;

int main() {

	/***Variable Declarations***/
	int** host_arrays;
	int** device_arrays;
	int* host_match;
	int* device_match;
	int* device_array_size;
	int array_size;
	int match_size;
	int num_arrays;
	int NUM_THREADS;
	int NUM_BLOCKS;
	size_t one_t;
	size_t array_set_bytes;
	size_t match_bytes;
	cudaError cuda_err;

	/***Initialization***/
	array_size = 1 << 3;
	num_arrays = 1 << 3;
	match_size = num_arrays;
	NUM_THREADS = num_arrays;
	NUM_BLOCKS = 1;

	// Host allocation
	one_t = (size_t) 1;
	array_set_bytes = (size_t) num_arrays * array_size * sizeof(int);
	match_bytes = (size_t) match_size * sizeof(int);

	host_arrays = (int**) calloc(one_t, array_set_bytes);
	host_match = (int*) calloc(one_t, match_bytes);

	if (host_arrays == NULL) {
		cerr << "Host arrays calloc failed\n" << endl;
		return -1;
	}

	if (host_match == NULL) {
		cerr << "Host match calloc failed\n" << endl;
		return -1;
	}

	//Device Allocation
	cuda_err = cudaMalloc((void**)&device_arrays, array_set_bytes);

	if (cuda_err != cudaSuccess) {
		cerr << "Device allocation for array set failed" << endl;
		return -1;
	}

	cuda_err = cudaMalloc((void**)&device_match, match_bytes);

	if (cuda_err != cudaSuccess) {
		cerr << "Device allcoation for match array failed" << endl;
		return -1;
	}

	cuda_err = cudaMalloc((void**)&device_array_size, sizeof(int)*array_size);

	if (cuda_err != cudaSuccess) {
		cerr << "Device allocation for device array size failed" << endl;
	}

	/*** Copy arrays to device ***/
	cudaMemcpy((void*)device_arrays, (void*)host_arrays, array_set_bytes, cudaMemcpyHostToDevice);
	cudaMemcpy((void*)device_match, (void*)host_match, match_bytes, cudaMemcpyHostToDevice);
	cudaMemcpy((void*)device_array_size, (void*) &array_size, array_size * sizeof(int), cudaMemcpyHostToDevice);

	/*** Search arrays and copy result back to host ***/
	//Memcopy works as a synchronization layer
	/*array_match <<<NUM_BLOCKS, NUM_THREADS >>> (device_arrays, device_match, array_size);
	
	//Copy match back to host
	cudaMemcpy(&host_match, device_match, match_bytes, cudaMemcpyDeviceToHost);

	//Print match array
	cout << "Match array: [";
	for (int i = 0; i < match_size; i++) {
		cout << host_match[i];
	}
	cout << "]" << endl;

	/***Free variables***/
	cudaFree(device_arrays);
	cudaFree(device_match);
	cudaFree(device_array_size);
	free(host_arrays);
	free(host_match);

	return 0;
}

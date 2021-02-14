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
		//if member can be found in thread_id-1 update match
	}
}

using namespace std;

int main() {

	/***Variable Declarations***/
	int* host_arrays;
	int* device_arrays;
	int* host_match;
	int* device_match;
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
	array_size = 1 << 4;
	num_arrays = 1 << 6;
	match_size = num_arrays;
	NUM_THREADS = num_arrays;
	NUM_BLOCKS = 1;

	// Host allocation
	one_t = (size_t) 1;
	array_set_bytes = (size_t) num_arrays * array_size * sizeof(int);
	match_bytes = (size_t) match_size * sizeof(int);

	host_arrays = (int*) calloc(one_t, array_set_bytes);
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
		cerr << "Device allocation for array set failed\n" << endl;
		return -1;
	}

	cuda_err = cudaMalloc((void**)&device_match, match_bytes);

	if (cuda_err != cudaSuccess) {
		cerr << "Device allcoation for match array failed\n" << endl;
		return -1;
	}


	/*** Copy arrays to device ***/
	cudaMemcpy((void*)device_arrays, (void*)host_arrays, array_set_bytes, cudaMemcpyHostToDevice);
	cudaMemcpy((void*)device_match, (void*)host_match, match_bytes, cudaMemcpyHostToDevice);

	//Search arrays and copy result back to host
	//Memcopy works as a synchronization layer
	//array_match <<<NUM_BLOCKS, NUM_THREADS >>> (dev_a, dev_b, dev_match, N);
	
	//Copy match back to host
	//cudaMemcpy(&host_match, dev_match, match_bytes, cudaMemcpyDeviceToHost);

	/***Free variables***/
	cudaFree(device_arrays);
	cudaFree(device_match);
	free(host_arrays);
	free(host_match);

	return 0;
}
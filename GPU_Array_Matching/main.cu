/* This program searches "states" for matches in their arrays using CUDA
*  
* Author: Robbie Watling
*/

#include "cuda_includes.h"
#include "rand_init.h"
#include <iostream>
#include <vector>
#include <utility>
#include <iostream>

__global__ void array_match(int* all_arrays, int* match_array, int num_arrays,  int size) {
	int thread_id = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (thread_id > 0 && thread_id < num_arrays) {
		int* current_array = all_arrays + (thread_id * size); //Pointer arithmetic
		int* prev_array = all_arrays + ((thread_id - 1) * size); //Pointer arithmetic
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
	//curandStatus_t rand_err;
	//curandGenerator_t gen;

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
		cerr << "Device allocation for array set failed" << endl;
		return -1;
	}

	cuda_err = cudaMalloc((void**)&device_match, match_bytes);

	if (cuda_err != cudaSuccess) {
		cerr << "Device allcoation for match array failed" << endl;
		return -1;
	}

	/*
	for (int i = 0; i < num_arrays; i++) {
		int step = i * array_size;
		int value = i / 2;

		for (int j = 0; j < array_size; j++) {
			host_arrays[step + j] = value;
		}
	}*/

	rand_init <<<NUM_BLOCKS, NUM_THREADS >>> (device_arrays, num_arrays, array_size);

	/*** Copy arrays to device ***/
	//cudaMemcpy((void*)device_arrays, (void*)host_arrays, array_set_bytes, cudaMemcpyHostToDevice);
	//cudaMemcpy((void*)device_match, (void*)host_match, match_bytes, cudaMemcpyHostToDevice);
	
	/*** Search arrays and copy result back to host ***/
	//Memcopy works as a synchronization layer
	array_match <<<NUM_BLOCKS, NUM_THREADS >>> (device_arrays, device_match, num_arrays, array_size);
	
	//Copy match back to host
	cudaMemcpy(host_match, device_match, match_bytes, cudaMemcpyDeviceToHost);

	//Copy gpu arrays to host for verification
	cudaMemcpy(host_arrays, device_arrays, array_set_bytes, cudaMemcpyDeviceToHost);

	//Print arrays
	cout << "Original arrays:" << endl;
	for (int i = 0; i < num_arrays; i++) {
		int step = i * array_size;
		cout << "[";

		for (int j = 0; j < array_size; j++) {
			cout << host_arrays[step + j] << " " ;
		}

		cout << "]" << endl;
	}

	//Print match array
	cout << "Match array: [";
	for (int i = 0; i < match_size; i++) {
		cout << host_match[i] << " ";
	}
	cout << "]" << endl;

	/***Free variables***/
	cudaFree(device_arrays);
	cudaFree(device_match);
	//free(host_arrays);
	free(host_match);

	return 0;
}

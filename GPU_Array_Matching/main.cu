/* This program searches "states" for matches in their arrays using CUDA
*
* Author: Robbie Watling
*/

#include "cuda_includes.h"
#include "array_match.h"
#include "shm_array_match.h"
#include <iostream>
#include <vector>
#include <utility>
#include <iostream>
#include <sys/time.h>

using namespace std;

int main(int argc, char** argv) {

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
	int shared;

	size_t one_t;
	size_t array_set_bytes;
	size_t match_bytes;

	cudaError cuda_err;

	struct timeval startShm;
	struct timeval startG;
	struct timeval stopShm;
	struct timeval stopG;

	double elapsed;

	if (argc < 4) {
		cerr << "./main array_size num_arrays shared(y/n)" << endl;
		return -1;
	}

	/***Initialization***/
	array_size = atoi(argv[1]);
	num_arrays = atoi(argv[2]);
	shared = atoi(argv[3]);
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

	cudaMemset(device_arrays, 0, array_size);
	cudaMemset(device_match, 0, match_size);

	//If shared is specified
	if (shared) {
		
		//Start timer shm
		gettimeofday(&startShm, 0);

		/*** Search arrays and copy result back to host using shared memory***/
		shm_array_match <<<NUM_BLOCKS, NUM_THREADS >>> (device_arrays, device_match, num_arrays, array_size);

		gettimeofday(&stopShm, 0);

		//Copy match back to host
		cudaMemcpy(host_match, device_match, match_bytes, cudaMemcpyDeviceToHost);

		//Copy gpu arrays to host for verification
		cudaMemcpy(host_arrays, device_arrays, array_set_bytes, cudaMemcpyDeviceToHost);

		//Print arrays
		/*cout << "Original arrays:" << endl;
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
		cout << "]" << endl;*/
		

		long shm_sec = stopShm.tv_sec - startShm.tv_sec;
		long shm_ms = stopShm.tv_usec - startShm.tv_usec;
		elapsed = shm_sec + shm_ms*1e-6;
	}

	//If not shared is specified
	if (!shared) {

		gettimeofday(&startG, 0);

		/*** Search arrays and copy back to host using global memory ***/
		array_match <<<NUM_BLOCKS, NUM_THREADS >>> (device_arrays, device_match, num_arrays, array_size);

		//Copy match back to host
		cudaMemcpy(host_match, device_match, match_bytes, cudaMemcpyDeviceToHost);

		gettimeofday(&stopG, 0);

		//Copy gpu arrays to host for verification
		cudaMemcpy(host_arrays, device_arrays, array_set_bytes, cudaMemcpyDeviceToHost);

		//Print arrays
		/*cout << "Original arrays:" << endl;
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
		cout << "]" << endl;*/

		long g_sec = stopG.tv_sec - startG.tv_sec;
		long g_ms = stopG.tv_usec - startG.tv_usec;
		elapsed = g_sec + g_ms*1e-6;
	}

	cout << shared << "," << num_arrays << "," << array_size << "," << elapsed << endl;

	/***Free variables***/
	cudaFree(device_arrays);
	cudaFree(device_match);
	free(host_arrays);
	free(host_match);

	return 0;
}

/* This program searches "states" for matches in their arrays using CUDA
*
* Author: Robbie Watling
*/

#include "cuda_includes.h"
#include "shm_array_match.h"
#include "shfl_array_match.h"
#include <iostream>
#include <vector>
#include <utility>
#include <iostream>
#include <iomanip>
#include <time.h>
#include <unistd.h>

using namespace std;

// For shuffling host arrays
void shuffle(int *array, size_t n)
{
    usleep(100); //Clock is too fast on server. This is to seed rand
    srand(time(NULL));

    if (n > 1) {
        size_t i;
        for (i = 0; i < n - 1; i++){
          size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
          int t = array[j];
          array[j] = array[i];
          array[i] = t;
        }
    }
}

int main(int argc, char** argv) {

	/***Variable Declarations***/
	int* host_arrays;
  int* experiment1_arrays;
  int* experiment2_arrays;
	int* device_arrays;
	int array_size;
  int num_arrays;
	int NUM_THREADS;
	int NUM_BLOCKS;
	int SHARE_SIZE;
  float milliseconds;
  cudaEvent_t start, stop;
  cudaEvent_t start1, stop1;
	cudaError_t cuda_err;
	size_t one_t;
	size_t array_set_bytes;

	/*** Read args ***/
	if (argc < 2) {
		cerr << "./gpu_match num_operating_threads" << endl;
		return -1;
	}

	/***Initialization***/
	array_size = ARRAY_SIZE; //Ignoring array size right now
	num_arrays = atoi(argv[1]);
  NUM_THREADS = num_arrays;
	NUM_BLOCKS = 1;
  SHARE_SIZE = 65536; //64 kibibytes, subject to change based on GPU requirements. See set attribute below


	//Host allocation
	one_t = (size_t) 1;
	array_set_bytes = (size_t) NUM_THREADS * array_size * 2 * sizeof(int);
  host_arrays = (int*) calloc(one_t, array_set_bytes);

	if (host_arrays == NULL) {
		cerr << "Host arrays calloc failed\n" << endl;
		return -1;
	}


  experiment1_arrays = (int*) calloc(one_t, array_set_bytes);

  if (experiment1_arrays == NULL) {
		cerr << "experiment1 arrays calloc failed\n" << endl;
		return -1;
	}

  experiment2_arrays = (int*) calloc(one_t, array_set_bytes);

  if (experiment2_arrays == NULL) {
		cerr << "experiment2 arrays calloc failed\n" << endl;
		return -1;
	}

	//Device Allocation
	cuda_err = cudaMalloc((void**)&device_arrays, array_set_bytes);

	if (cuda_err != cudaSuccess) {
		cerr << "Device allocation for array set failed" << endl;
		return -1;
	}

  //Fill in host arrays to emulate major operation
  for(int i = 0; i < NUM_THREADS; i++) {

    //Start array
		for(int j = 0; j < array_size; j++) {
      if (i != 0) {
        host_arrays[(i * array_size * 2) + j] = j;
      }
		}

    if (i != 0) {
      shuffle(host_arrays + (i * array_size * 2), array_size);
    }

    //End array
    for(int j = array_size; j < array_size * 2; j++) {
      host_arrays[(i * array_size * 2) + j] = j % 8;
		}

    shuffle(host_arrays + (i * array_size * 2) + array_size, array_size);
	}

  //Print arrays before matching
  for(int i = 0; i < NUM_THREADS; i++) {

    cout << "Arrays " << i << ": [";

		for(int j = 0; j < array_size * 2; j++) {
			cout << host_arrays[(i * array_size * 2) + j] << " ";

      if (j == array_size - 1) {
        cout << "]\t[";
      }
		}

    cout << "]" << endl;
	}

  /************************Experiment 1***************************************/
  //Set max dynamic shared memory size to either 96 kibibytes or 64 kibibytes
  cuda_err = cudaFuncSetAttribute(shm_array_match, cudaFuncAttributeMaxDynamicSharedMemorySize, SHARE_SIZE);

  if (cuda_err != cudaSuccess) {

    cerr << endl << "Dynamic shared memory size of 96kb for array set failed, trying 64kb" << endl << endl;
    SHARE_SIZE = 65536;

    cuda_err = cudaFuncSetAttribute(shm_array_match, cudaFuncAttributeMaxDynamicSharedMemorySize, SHARE_SIZE);

    if (cuda_err != cudaSuccess) {
      cerr << "Dynamic shared memory size of 64000 for array set failed. Exiting program..." << endl;

      return -1;
    }
	}

  cout << "***Experiment1***" << endl;

  //Copy host arrays to device
  cudaMemcpy(device_arrays, host_arrays, array_set_bytes, cudaMemcpyHostToDevice);


  cout << "--------------------KERNEL CALL--------------------" << endl;

  //Timing
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  //Kernel call
  shm_array_match <<<NUM_BLOCKS, NUM_THREADS, SHARE_SIZE>>> (device_arrays, NUM_THREADS);

  //Timing
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  //Copy device arrays back to host
  cudaMemcpy(experiment1_arrays, device_arrays, array_set_bytes, cudaMemcpyDeviceToHost);

  //Print arrays after matching
  for(int i = 0; i < 1; i++) {

    cout << "Arrays " << i << ": [";

    for(int j = 0; j < array_size * 2; j++) {
      cout << experiment1_arrays[(i * array_size * 2) + j] << " ";

      if (j == array_size - 1) {
        cout << "]\t[";
      }
    }

    cout << "]" << endl;
  }

  cout << milliseconds << "ms" << endl << endl;

  /************************Experiment 2***************************************/
  //Set max dynamic shared memory size to either 96 kibibytes or 64 kibibytes
  cuda_err = cudaFuncSetAttribute(shfl_array_match, cudaFuncAttributeMaxDynamicSharedMemorySize, SHARE_SIZE);

  if (cuda_err != cudaSuccess) {

    cerr << endl << "Dynamic shared memory size of 96kb for array set failed, trying 64kb" << endl << endl;
    SHARE_SIZE = 65536;

    cuda_err = cudaFuncSetAttribute(shm_array_match, cudaFuncAttributeMaxDynamicSharedMemorySize, SHARE_SIZE);

    if (cuda_err != cudaSuccess) {
      cerr << "Dynamic shared memory size of 64000 for array set failed. Exiting program..." << endl;

      return -1;
    }
	}

  //Copy host arrays to device
  cudaMemcpy(device_arrays, host_arrays, array_set_bytes, cudaMemcpyHostToDevice);


  cout << "***Experiment2***" << endl;

  cout << "--------------------KERNEL CALL--------------------" << endl;

  //Timing
  cudaEventCreate(&start1);
  cudaEventCreate(&stop1);
  cudaEventRecord(start1, 0);

  //Kernel call
  shfl_array_match <<<NUM_BLOCKS, NUM_THREADS, SHARE_SIZE>>> (device_arrays, NUM_THREADS);

  //Timing
  cudaEventRecord(stop1, 0);
  cudaEventSynchronize(stop1);
  cudaEventElapsedTime(&milliseconds, start1, stop1);
  cudaEventDestroy(start1);
  cudaEventDestroy(stop1);

  //Copy device arrays back to host
  cudaMemcpy(experiment2_arrays, device_arrays, array_set_bytes, cudaMemcpyDeviceToHost);

  //Print arrays after matching
  for(int i = 0; i < 1; i++) {

    cout << "Arrays " << i << ": [";

    for(int j = 0; j < array_size * 2; j++) {
      cout << experiment2_arrays[(i * array_size * 2) + j] << " ";

      if (j == array_size - 1) {
        cout << "]\t[";
      }
    }

    cout << "]" << endl;
  }

  cout << milliseconds << "ms" << endl;

	/***Free variables***/
	cudaFree(device_arrays);
	free(host_arrays);
  free(experiment1_arrays);
  free(experiment2_arrays);

	return 0;
}

/* This program searches "states" for matches in their arrays using CUDA
*
* Author: Robbie Watling
*/

#include "cuda_includes.h"
#include "shm_array_match.h"
#include "shfl_array_match.h"
#include "shfl_hash_match.h"
#include "shm_hash_match.h"
#include "cpu_array_match.h"
#include <iostream>
#include <sys/time.h>

#define SHM_96_KB 98304
#define SHM_64_KB 65536

using namespace std;

// For shuffling host arrays
void shuffle(int *array, size_t n)
{
  struct timeval tp;
  gettimeofday(&tp, NULL);
  long int mytime = tp.tv_sec * 1000 + tp.tv_usec;
  srand(mytime);

  if (n > 1) {
      int i;
      for (i = 0; i < n - 1; i++){
        int j = i + rand() / (RAND_MAX / (n - i) + 1);
        int t = array[j];
        array[j] = array[i];
        array[i] = t;
      }
  }
}

int main(int argc, char** argv) {

	/***Variable Declarations***/
  float milliseconds;

  int* host_arrays;
  int* experiment1_arrays;
  int* experiment2_arrays;
  int* experiment3_arrays;
	int* device_arrays;

	int array_size;
  int num_arrays;
	int num_threads;
	int num_blocks;
  int share_size;
  int debug;

	size_t one_t;
	size_t array_set_bytes;

  cudaEvent_t start1, stop1;
  cudaEvent_t start2, stop2;
  cudaEvent_t start3, stop3;
  cudaError_t cuda_err;

	/*** Read args ***/
	if (argc < 3) {
		cerr << "./gpu_match num_operating_threads debug(1 or 0)" << endl;
		return -1;
	}

	/***Initialization***/
	array_size = ARRAY_SIZE;
	num_arrays = atoi(argv[1]);
  debug = (atoi(argv[2]));
  num_threads = num_arrays;
	num_blocks = 1;
  share_size = SHM_64_KB;


	//Host allocation
	one_t = (size_t) 1;
	array_set_bytes = (size_t) num_threads * array_size * 2 * sizeof(int);
  host_arrays = (int*) calloc(one_t, array_set_bytes);

	if (host_arrays == NULL) {
		cerr << "Host arrays calloc failed\n" << endl;
		return -1;
	}

  //Experiment arrays allocation
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

  experiment3_arrays = (int*) calloc(one_t, array_set_bytes);

  if (experiment3_arrays == NULL) {
		cerr << "experiment3 arrays calloc failed\n" << endl;
		return -1;
	}

	//Device Allocation
	cuda_err = cudaMalloc((void**)&device_arrays, array_set_bytes);

	if (cuda_err != cudaSuccess) {
		cerr << "Device allocation for array set failed" << endl;
		return -1;
	}

  //Fill in host arrays to emulate major operation
  for(int i = 0; i < num_threads; i++) {

    //Start array
		for(int j = 0; j < array_size; j++) {
      if (i != 0) {
        host_arrays[(i * array_size * 2) + j] = j;
      }
		}

    if (i != 0) { shuffle(host_arrays + (i * array_size * 2), array_size); }

    //End array
    for(int j = array_size; j < array_size * 2; j++) {
      host_arrays[(i * array_size * 2) + j] = j % array_size;
		}

    shuffle(host_arrays + (i * array_size * 2) + array_size, array_size);
	}

  //Print arrays before matching
  if (debug) {
    for(int i = 0; i < num_threads; i++) {

      cout << "Arrays " << i << ": [";

  		for(int j = 0; j < array_size * 2; j++) {
  			cout << host_arrays[(i * array_size * 2) + j] << " ";

        if (j == array_size - 1) { cout << "]\t["; }
  		}

      cout << "]" << endl;
  	}
  }

  /************************Experiment 1***************************************/

  //Set max dynamic shared memory size to either 96 kibibytes or 64 kibibytes
  share_size = SHM_96_KB;
  cuda_err = cudaFuncSetAttribute(shm_array_match, cudaFuncAttributeMaxDynamicSharedMemorySize, share_size);

  if (cuda_err != cudaSuccess) {

    if (debug) { cerr << endl << "Dynamic shared memory size of 96kb for array set failed, trying 64kb" << endl; }
    share_size = SHM_64_KB;

    cuda_err = cudaFuncSetAttribute(shm_array_match, cudaFuncAttributeMaxDynamicSharedMemorySize, share_size);

    if (cuda_err != cudaSuccess) {

      if (debug) { cerr << "Dynamic shared memory size of 64000 for array set failed. Exiting program..." << endl; }

      return -1;
    }
	}

  //Copy host arrays to device
  cudaMemcpy(device_arrays, host_arrays, array_set_bytes, cudaMemcpyHostToDevice);

  if (debug) {
    cout << endl << "***Experiment 1 Shared Mem***" << endl;

    cout << "--------------------KERNEL CALL--------------------" << endl;
  }

  //Timing
  cudaEventCreate(&start1);
  cudaEventCreate(&stop1);
  cudaEventRecord(start1, 0);

  //Kernel call
  shm_array_match <<<num_blocks, num_threads, share_size>>> (device_arrays, num_threads);

  //Timing
  cudaEventRecord(stop1, 0);
  cudaEventSynchronize(stop1);
  cudaEventElapsedTime(&milliseconds, start1, stop1);
  cudaEventDestroy(start1);
  cudaEventDestroy(stop1);

  //Copy device arrays back to host
  cudaMemcpy(experiment1_arrays, device_arrays, array_set_bytes, cudaMemcpyDeviceToHost);

  if (debug) {
    //Print arrays after matching
    for(int i = 0; i < 1; i++) {

      cout << "Arrays " << i << ": [";

      for(int j = 0; j < array_size * 2; j++) {
        cout << experiment1_arrays[(i * array_size * 2) + j] << " ";

        if (j == array_size - 1) { cout << "]\t["; }
      }

      cout << "]" << endl;
    }
  }

  cout << 0 << "," << num_threads << "," << array_size << "," << milliseconds << endl;

  /************************Experiment 2***************************************/
  //Set max dynamic shared memory size to either 96 kibibytes or 64 kibibytes
  cuda_err = cudaFuncSetAttribute(shfl_array_match, cudaFuncAttributeMaxDynamicSharedMemorySize, share_size);

  if (cuda_err != cudaSuccess) {
    if (debug) { cerr << endl << "Second attempt of defining dynamic shared memory size of 96kb for array set failed" << endl << endl; }
    return -1;
	}

  //Copy host arrays to device
  cudaMemcpy(device_arrays, host_arrays, array_set_bytes, cudaMemcpyHostToDevice);

  if (debug) {
    cout << endl << "***Experiment 2 Shuffle***" << endl;

    cout << "--------------------KERNEL CALL--------------------" << endl;
  }

  //Timing
  cudaEventCreate(&start2);
  cudaEventCreate(&stop2);
  cudaEventRecord(start2, 0);

  //Kernel call
  shfl_array_match <<<num_blocks, num_threads, share_size>>> (device_arrays, num_threads);

  //Timing
  cudaEventRecord(stop2, 0);
  cudaEventSynchronize(stop2);
  cudaEventElapsedTime(&milliseconds, start2, stop2);
  cudaEventDestroy(start2);
  cudaEventDestroy(stop2);

  cout << 1 << "," << num_threads << "," << array_size << "," << milliseconds << endl;

  //Copy device arrays back to host
  cudaMemcpy(experiment2_arrays, device_arrays, array_set_bytes, cudaMemcpyDeviceToHost);

  if (debug) {
    //Print arrays after matching
    for(int i = 0; i < 1; i++) {

      cout << "Arrays " << i << ": [";

      for(int j = 0; j < array_size * 2; j++) {
        cout << experiment2_arrays[(i * array_size * 2) + j] << " ";

        if (j == array_size - 1) { cout << "]\t["; }
      }

      cout << "]" << endl;
    }

    /************************Experiment 3***************************************/
    //Set max dynamic shared memory size to either 96 kibibytes or 64 kibibytes
    cuda_err = cudaFuncSetAttribute(shm_hash_match, cudaFuncAttributeMaxDynamicSharedMemorySize, share_size);

    if (cuda_err != cudaSuccess) {
      if (debug) { cerr << endl << "Third attempt of defining dynamic shared memory size of 96kb for array set failed" << endl << endl; }
      return -1;
  	}

    //Copy host arrays to device
    cudaMemcpy(device_arrays, host_arrays, array_set_bytes, cudaMemcpyHostToDevice);

    if (debug) {
      cout << endl << "***Experiment 3 Shared with Hash***" << endl;

      cout << "--------------------KERNEL CALL--------------------" << endl;
    }

    //Timing
    cudaEventCreate(&start3);
    cudaEventCreate(&stop3);
    cudaEventRecord(start3, 0);

    //Kernel call
    shm_hash_match<<<num_blocks, num_threads, share_size>>>(device_arrays, num_threads);

    //Timing
    cudaEventRecord(stop3, 0);
    cudaEventSynchronize(stop3);
    cudaEventElapsedTime(&milliseconds, start3, stop3);
    cudaEventDestroy(start3);
    cudaEventDestroy(stop3);

    cout << 2 << "," << num_threads << "," << array_size << "," << milliseconds << endl;

    //Copy device arrays back to host
    cudaMemcpy(experiment3_arrays, device_arrays, array_set_bytes, cudaMemcpyDeviceToHost);

    if (debug) {
      //Print arrays after matching
      for(int i = 0; i < 1; i++) {

        cout << "Arrays " << i << ": [";

        for(int j = 0; j < array_size * 2; j++) {
          cout << experiment3_arrays[(i * array_size * 2) + j] << " ";

          if (j == array_size - 1) { cout << "]\t["; }
        }

        cout << "]" << endl;
      }
    }

    /************************Experiment 4***************************************/
    //Set max dynamic shared memory size to either 96 kibibytes or 64 kibibytes
    /*cuda_err = cudaFuncSetAttribute(shfl_array_match, cudaFuncAttributeMaxDynamicSharedMemorySize, share_size);

    if (cuda_err != cudaSuccess) {
      if (debug) { cerr << endl << "Third attempt of defining dynamic shared memory size of 96kb for array set failed" << endl << endl; }
      return -1;
    }

    //Copy host arrays to device
    cudaMemcpy(device_arrays, host_arrays, array_set_bytes, cudaMemcpyHostToDevice);

    if (debug) {
      cout << endl << "***Experiment 3 Shared with Hash***" << endl;

      cout << "--------------------KERNEL CALL--------------------" << endl;
    }

    //Timing
    cudaEventCreate(&start4);
    cudaEventCreate(&stop4);
    cudaEventRecord(start4, 0);

    //Kernel call
    shfl_array_match <<<num_blocks, num_threads, share_size>>> (device_arrays, num_threads);

    //Timing
    cudaEventRecord(stop4, 0);
    cudaEventSynchronize(stop4);
    cudaEventElapsedTime(&milliseconds, start3, stop3);
    cudaEventDestroy(start4);
    cudaEventDestroy(stop4);

    cout << 1 << "," << num_threads << "," << array_size << "," << milliseconds << endl;

    //Copy device arrays back to host
    cudaMemcpy(experiment4_arrays, device_arrays, array_set_bytes, cudaMemcpyDeviceToHost);

    if (debug) {
      //Print arrays after matching
      for(int i = 0; i < 1; i++) {

        cout << "Arrays " << i << ": [";

        for(int j = 0; j < array_size * 2; j++) {
          cout << experiment3_arrays[(i * array_size * 2) + j] << " ";

          if (j == array_size - 1) { cout << "]\t["; }
        }

        cout << "]" << endl;
      }
    }*/


    /************************CPU Verification***************************************/
    cout << endl << "***Host Arrays***" << endl;

    cpu_array_match(host_arrays, num_threads, array_size);

    for(int i = 0; i < 1; i++) {

      cout << "Arrays " << i << ": [";

      for(int j = 0; j < array_size * 2; j++) {
        cout << host_arrays[(i * array_size * 2) + j] << " ";

        if (j == array_size - 1) { cout << "]\t["; }
      }

      cout << "]" << endl;
    }
  }

	/***Free variables***/
	cudaFree(device_arrays);
	free(host_arrays);
  free(experiment1_arrays);
  free(experiment2_arrays);
  free(experiment3_arrays);

	return 0;
}

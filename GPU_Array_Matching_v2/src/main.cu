/* This program searches "states" for matches in their arrays using CUDA
*
* Author: Robbie Watling
*/

#include "cuda_includes.h"
#include "shm_array_match.h"
#include <iostream>
#include <vector>
#include <utility>
#include <iostream>
#include <iomanip>
#include <time.h>

using namespace std;

// For shuffling host arrays
void shuffle(int *array, size_t n)
{
    srand(clock());
    if (n > 1)
    {
        size_t i;
        for (i = 0; i < n - 1; i++)
        {
          size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
          int t = array[j];
          array[j] = array[i];
          array[i] = t;
        }
    }
}

int main(int argc, char** argv) {

	/***Variable Declarations***/
	//Host and device variables
	int* host_arrays;
	int* device_arrays;

	//Host and device detail variables
	int array_size;
  int num_arrays;
	int NUM_THREADS;
	int NUM_BLOCKS;
	int SHARE_SIZE;
	cudaError_t cuda_err;

	//Byte size variables
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
  SHARE_SIZE = MAX_SHM;

	//Host allocation
	one_t = (size_t) 1;
	array_set_bytes = (size_t) NUM_THREADS * array_size * 2 * sizeof(int);
  host_arrays = (int*) calloc(one_t, array_set_bytes);

	if (host_arrays == NULL) {
		cerr << "Host arrays calloc failed\n" << endl;
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

  //Print arrays
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

  //cudaFuncGetAttributes(&attributes, shm_array_match);
  //cudaFuncSetAttribue(shm_array_match, maxDynamicSharedSizeBytes)

  cudaMemcpy(device_arrays, host_arrays, array_set_bytes, cudaMemcpyHostToDevice);

  shm_array_match <<<NUM_BLOCKS, NUM_THREADS, SHARE_SIZE>>> (device_arrays, NUM_THREADS);

  cudaMemcpy(host_arrays, device_arrays, array_set_bytes, cudaMemcpyDeviceToHost);

  cout << "----------------KERNEL CALL----------------" << endl;

  //Print arrays
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

	/***Free variables***/
	cudaFree(device_arrays);
	free(host_arrays);

	return 0;
}

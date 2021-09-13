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
	int num_operating_threads;
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
	num_operating_threads = atoi(argv[1]);
	NUM_THREADS = num_operating_threads;
	NUM_BLOCKS = 1;

	//Host allocation
	one_t = (size_t) 1;
	array_set_bytes = (size_t) num_operating_threads * array_size * 2 * sizeof(int);
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
  for(int i = 0; i < num_operating_threads; i++) {

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
  for(int i = 0; i < num_operating_threads; i++) {

    cout << "Arrays " << i << ": [";

		for(int j = 0; j < array_size * 2; j++) {
			cout << host_arrays[(i * array_size * 2) + j] << " ";
		}

    cout << "]" << endl;
	}

  //cuda memcopy
  //cuda shared memory
  //cudaFuncGetAttributes(&attributes, shm_array_match);
  //cudaFuncSetAttribue(shm_array_match, maxDynamicSharedSizeBytes)

  /*Questions:
  1) number of threads not power of 2?
  2) Shared memory max size
  3) Confirming matches
    -> 2d array?
  */

	/***Free variables***/
	cudaFree(device_arrays);
	free(host_arrays);

	return 0;
}

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
	int* host_match;
	int* device_match;
  int* prev_end;
  int* current_start;

	//Host and device detail variables
	int array_size;
	int match_size;
	int num_operating_threads;
	int NUM_THREADS;
	int NUM_BLOCKS;
	int SHARE_SIZE;
	cudaError_t cuda_err;

	// Byte size variables
	size_t one_t;
	size_t array_set_bytes;
	size_t match_bytes;

	/*** Read args ***/
	if (argc < 3) {
		cerr << "./gpu_match array_size num_operating_threads" << endl;
		return -1;
	}

	/***Initialization***/
	array_size = 8; //Ignoring array size right now
	num_operating_threads = atoi(argv[2]); //One start array and one end array
	match_size = num_operating_threads;
	NUM_THREADS = num_operating_threads;
	NUM_BLOCKS = 1;

	//Host allocation
	one_t = (size_t) 1;
	array_set_bytes = (size_t) num_operating_threads * array_size * 2 * sizeof(int);
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

  cuda_err = cudaMalloc((void**)&prev_end, array_size * sizeof(int));

  if (cuda_err != cudaSuccess) {
		cerr << "Device allcoation for prev_end failed" << endl;
		return -1;
	}

  cuda_err = cudaMalloc((void**) &current_start, array_size * sizeof(int));

  if (cuda_err != cudaSuccess) {
		cerr << "Device allcoation for current_start failed" << endl;
		return -1;
	}

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

  /*Questions:
  prev_end and current_start?
  how to denote match?

	/***Free variables***/
	cudaFree(device_arrays);
	cudaFree(device_match);
  cudaFree(prev_end);
  cudaFree(current_start);
	free(host_arrays);
	free(host_match);

	return 0;
}

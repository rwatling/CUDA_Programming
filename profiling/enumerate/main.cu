/* This program searches "states" for matches in their arrays using CUDA.
* This version for profiling memory type usages.
* Author: Robbie Watling
*/

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <curand.h>
#include <curand_kernel.h>
#include <iostream>
#include <sys/time.h>

#define SHM_96_KB 98304
#define SHM_64_KB 65536
#define ARRAY_SIZE 8
#define WARP_SIZE 32

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

__device__ void match(int* array2, int* next_arr1, int* next_arr2) {

  //TODO: Having an issue with no-matches
  for (int i = 0; i < ARRAY_SIZE; i++) {
    for (int j = 0; j < ARRAY_SIZE; j++) {
      if (array2[i] == next_arr1[j]) {
        array2[i] = next_arr2[j];
        break;
      }
    }
  }
}

__global__ void shm_array_match(int* global_arrays, int num_threads) {

	int thread_id = (blockIdx.x * blockDim.x) + threadIdx.x;
	extern __shared__ int shared_arrays[];
	//int current_arr1[ARRAY_SIZE];
	//int current_arr2[ARRAY_SIZE];
	//int next_arr1[ARRAY_SIZE];
	//int next_arr2[ARRAY_SIZE];

  int size = ARRAY_SIZE; //STILL NEED TO UPDATE THIS FOR ENUMERATED VERSION

  //int arr1_index = 0;
	//int arr2_index = 0;

	//Retrieve global values from major operation
	//Assign the global values to registers for array_1 and array_2
	//Assign the initial global values to shared memory
	/*for (int i = 0; i < size; i++) {
		int arr1_index = (thread_id * 2 * size) + i;
		current_arr1[i] = global_arrays[arr1_index];

		int arr2_index = (thread_id * 2 * size) + size + i;
		current_arr2[i] = global_arrays[arr2_index];
	}*/

  //Enumerated version
  //Current array 1
  int current_a1_e0 = global_arrays[0];
  int current_a1_e1 = global_arrays[1];
  int current_a1_e2 = global_arrays[2];
  int current_a1_e3 = global_arrays[3];

  int current_a1_e4 = global_arrays[4];
  int current_a1_e5 = global_arrays[5];
  int current_a1_e6 = global_arrays[6];
  int current_a1_e7 = global_arrays[7];

  //Current array 2
  int current_a2_e0 = global_arrays[size + 0];
  int current_a2_e1 = global_arrays[size + 1];
  int current_a2_e2 = global_arrays[size + 2];
  int current_a2_e3 = global_arrays[size + 3];

  int current_a2_e4 = global_arrays[size + 4];
  int current_a2_e5 = global_arrays[size + 5];
  int current_a2_e6 = global_arrays[size + 6];
  int current_a2_e7 = global_arrays[size + 7];

  //Init next array 1
  int next_a1_e0 = 0;
  int next_a1_e1 = 0;
  int next_a1_e2 = 0;
  int next_a1_e3 = 0;

  int next_a1_e4 = 0;
  int next_a1_e5 = 0;
  int next_a1_e6 = 0;
  int next_a1_e7 = 0;

  //Init next array 2
  int next_a2_e0 = 0;
  int next_a2_e1 = 0;
  int next_a2_e2 = 0;
  int next_a2_e3 = 0;

  int next_a2_e4 = 0;
  int next_a2_e5 = 0;
  int next_a2_e6 = 0;
  int next_a2_e7 = 0;

	__syncthreads();

	// Tree like match reduction using shared memory
	for (int k = 1; k < num_threads; k = k << 1) {

		// If thread is a writer
		if ((thread_id % (k * 2)) == k) {

			//Write my first array to shared memory for communication
			/*for (int i = 0; i < size; i++) {
				arr1_index = (thread_id / (k * 2)) * 2 * size + i;
				shared_arrays[arr1_index] = current_arr1[i];
			}

			//Write my second array to shared memory for communication
			for (int i = 0; i < size; i++) {
				arr2_index = (thread_id / (k * 2)) * 2 * size + size + i;
				shared_arrays[arr2_index] = current_arr2[i];
			}*/

      //Write for current array 1
      shared_arrays[(thread_id / (k * 2)) * 2 * size + 0] = current_a1_e0;
      shared_arrays[(thread_id / (k * 2)) * 2 * size + 1] = current_a1_e1;
      shared_arrays[(thread_id / (k * 2)) * 2 * size + 2] = current_a1_e2;
      shared_arrays[(thread_id / (k * 2)) * 2 * size + 3] = current_a1_e3;

      shared_arrays[(thread_id / (k * 2)) * 2 * size + 4] = current_a1_e4;
      shared_arrays[(thread_id / (k * 2)) * 2 * size + 5] = current_a1_e5;
      shared_arrays[(thread_id / (k * 2)) * 2 * size + 6] = current_a1_e6;
      shared_arrays[(thread_id / (k * 2)) * 2 * size + 7] = current_a1_e7;

      //Write for current array 2
      shared_arrays[(thread_id / (k * 2)) * 2 * size + size + 0] = current_a2_e0;
      shared_arrays[(thread_id / (k * 2)) * 2 * size + size + 1] = current_a2_e1;
      shared_arrays[(thread_id / (k * 2)) * 2 * size + size + 2] = current_a2_e2;
      shared_arrays[(thread_id / (k * 2)) * 2 * size + size + 3] = current_a2_e3;

      shared_arrays[(thread_id / (k * 2)) * 2 * size + size + 4] = current_a2_e4;
      shared_arrays[(thread_id / (k * 2)) * 2 * size + size + 5] = current_a2_e5;
      shared_arrays[(thread_id / (k * 2)) * 2 * size + size + 6] = current_a2_e6;
      shared_arrays[(thread_id / (k * 2)) * 2 * size + size + 7] = current_a2_e7;
		}

		__syncthreads();

		// If thread is a reader
		if ((thread_id % (k * 2) == 0)) {

			//Read my writers first array
			/*for (int i = 0; i < size; i++) {
				arr1_index = (thread_id / (k * 2)) * 2 * size + i;
				next_arr1[i] = shared_arrays[arr1_index];
			}

			//Read my writers second array
			for (int i = 0; i < size; i++) {
				arr2_index = (thread_id / (k * 2)) * 2 * size + size + i;
				next_arr2[i] = shared_arrays[arr2_index];
			}*/

      //Read next array 1
      next_a1_e0 = shared_arrays[(thread_id / (k*2)) * 2 * size + 0];
      next_a1_e1 = shared_arrays[(thread_id / (k*2)) * 2 * size + 1];
      next_a1_e2 = shared_arrays[(thread_id / (k*2)) * 2 * size + 2];
      next_a1_e3 = shared_arrays[(thread_id / (k*2)) * 2 * size + 3];

      next_a1_e4 = shared_arrays[(thread_id / (k*2)) * 2 * size + 4];
      next_a1_e5 = shared_arrays[(thread_id / (k*2)) * 2 * size + 5];
      next_a1_e6 = shared_arrays[(thread_id / (k*2)) * 2 * size + 6];
      next_a1_e7 = shared_arrays[(thread_id / (k*2)) * 2 * size + 7];

      //Read next array 2
      next_a2_e0 = shared_arrays[(thread_id / (k*2)) * 2 * size + size + 0];
      next_a2_e1 = shared_arrays[(thread_id / (k*2)) * 2 * size + size + 1];
      next_a2_e2 = shared_arrays[(thread_id / (k*2)) * 2 * size + size + 2];
      next_a2_e3 = shared_arrays[(thread_id / (k*2)) * 2 * size + size + 3];

      next_a2_e4 = shared_arrays[(thread_id / (k*2)) * 2 * size + size + 4];
      next_a2_e5 = shared_arrays[(thread_id / (k*2)) * 2 * size + size + 5];
      next_a2_e6 = shared_arrays[(thread_id / (k*2)) * 2 * size + size + 6];
      next_a2_e7 = shared_arrays[(thread_id / (k*2)) * 2 * size + size + 7];

			//match(current_arr2, next_arr1, next_arr2);

      //current_a2_e0
      if (current_a2_e0 == next_a1_e0) {
        current_a2_e0 = next_a2_e0;
      } else if (current_a2_e0 == next_a1_e1) {
        current_a2_e0 = next_a2_e1;
      } else if (current_a2_e0 == next_a1_e2) {
        current_a2_e0 = next_a2_e2;
      } else if (current_a2_e0 == next_a1_e3) {
        current_a2_e0 = next_a2_e3;
      } else if (current_a2_e0 == next_a1_e4) {
        current_a2_e0 = next_a2_e4;
      } else if (current_a2_e0 == next_a1_e5) {
        current_a2_e0 = next_a2_e5;
      } else if (current_a2_e0 == next_a1_e6) {
        current_a2_e0 = next_a2_e6;
      } else if (current_a2_e0 == next_a1_e7) {
        current_a2_e0 = next_a2_e7;
      }

      //current_a2_e1
      if (current_a2_e1 == next_a1_e0) {
        current_a2_e1 = next_a2_e0;
      } else if (current_a2_e1 == next_a1_e1) {
        current_a2_e1 = next_a2_e1;
      } else if (current_a2_e1 == next_a1_e2) {
        current_a2_e1 = next_a2_e2;
      } else if (current_a2_e1 == next_a1_e3) {
        current_a2_e1 = next_a2_e3;
      } else if (current_a2_e1 == next_a1_e4) {
        current_a2_e1 = next_a2_e4;
      } else if (current_a2_e1 == next_a1_e5) {
        current_a2_e1 = next_a2_e5;
      } else if (current_a2_e1 == next_a1_e6) {
        current_a2_e1 = next_a2_e6;
      } else if (current_a2_e1 == next_a1_e7) {
        current_a2_e1 = next_a2_e7;
      }

      //current_a2_e2
      if (current_a2_e2 == next_a1_e0) {
        current_a2_e2 = next_a2_e0;
      } else if (current_a2_e2 == next_a1_e1) {
        current_a2_e2 = next_a2_e1;
      } else if (current_a2_e2 == next_a1_e2) {
        current_a2_e2 = next_a2_e2;
      } else if (current_a2_e2 == next_a1_e3) {
        current_a2_e2 = next_a2_e3;
      } else if (current_a2_e2 == next_a1_e4) {
        current_a2_e2 = next_a2_e4;
      } else if (current_a2_e2 == next_a1_e5) {
        current_a2_e2 = next_a2_e5;
      } else if (current_a2_e2 == next_a1_e6) {
        current_a2_e2 = next_a2_e6;
      } else if (current_a2_e2 == next_a1_e7) {
        current_a2_e2 = next_a2_e7;
      }

      //current_a2_e3
      if (current_a2_e3 == next_a1_e0) {
        current_a2_e3 = next_a2_e0;
      } else if (current_a2_e3 == next_a1_e1) {
        current_a2_e3 = next_a2_e1;
      } else if (current_a2_e3 == next_a1_e2) {
        current_a2_e3 = next_a2_e2;
      } else if (current_a2_e3 == next_a1_e3) {
        current_a2_e3 = next_a2_e3;
      } else if (current_a2_e3 == next_a1_e4) {
        current_a2_e3 = next_a2_e4;
      } else if (current_a2_e3 == next_a1_e5) {
        current_a2_e3 = next_a2_e5;
      } else if (current_a2_e3 == next_a1_e6) {
        current_a2_e3 = next_a2_e6;
      } else if (current_a2_e3 == next_a1_e7) {
        current_a2_e3 = next_a2_e7;
      }

      //current_a2_e4
      /*if (current_a2_e4 == next_a1_e0) {
        current_a2_e4 = next_a2_e0;
      } else if (current_a2_e4 == next_a1_e1) {
        current_a2_e4 = next_a2_e1;
      } else if (current_a2_e4 == next_a1_e2) {
        current_a2_e4 = next_a2_e2;
      } else if (current_a2_e4 == next_a1_e3) {
        current_a2_e4 = next_a2_e3;
      } else if (current_a2_e4 == next_a1_e4) {
        current_a2_e4 = next_a2_e4;
      } else if (current_a2_e4 == next_a1_e5) {
        current_a2_e4 = next_a2_e5;
      } else if (current_a2_e4 == next_a1_e6) {
        current_a2_e4 = next_a2_e6;
      } else if (current_a2_e4 == next_a1_e7) {
        current_a2_e4 = next_a2_e7;
      }

      //current_a2_e5
      if (current_a2_e5 == next_a1_e0) {
        current_a2_e5 = next_a2_e0;
      } else if (current_a2_e5 == next_a1_e1) {
        current_a2_e5 = next_a2_e1;
      } else if (current_a2_e5 == next_a1_e2) {
        current_a2_e5 = next_a2_e2;
      } else if (current_a2_e5 == next_a1_e3) {
        current_a2_e5 = next_a2_e3;
      } else if (current_a2_e5 == next_a1_e4) {
        current_a2_e5 = next_a2_e4;
      } else if (current_a2_e5 == next_a1_e5) {
        current_a2_e5 = next_a2_e5;
      } else if (current_a2_e5 == next_a1_e6) {
        current_a2_e5 = next_a2_e6;
      } else if (current_a2_e5 == next_a1_e7) {
        current_a2_e5 = next_a2_e7;
      }

      //current_a2_e6
      if (current_a2_e6 == next_a1_e0) {
        current_a2_e6 = next_a2_e0;
      } else if (current_a2_e6 == next_a1_e1) {
        current_a2_e6 = next_a2_e1;
      } else if (current_a2_e6 == next_a1_e2) {
        current_a2_e6 = next_a2_e2;
      } else if (current_a2_e6 == next_a1_e3) {
        current_a2_e6 = next_a2_e3;
      } else if (current_a2_e6 == next_a1_e4) {
        current_a2_e6 = next_a2_e4;
      } else if (current_a2_e6 == next_a1_e5) {
        current_a2_e6 = next_a2_e5;
      } else if (current_a2_e6 == next_a1_e6) {
        current_a2_e6 = next_a2_e6;
      } else if (current_a2_e6 == next_a1_e7) {
        current_a2_e6 = next_a2_e7;
      }

      //current_a2_e7
      if (current_a2_e7 == next_a1_e0) {
        current_a2_e7 = next_a2_e0;
      } else if (current_a2_e7 == next_a1_e1) {
        current_a2_e7 = next_a2_e1;
      } else if (current_a2_e7 == next_a1_e2) {
        current_a2_e7 = next_a2_e2;
      } else if (current_a2_e7 == next_a1_e3) {
        current_a2_e7 = next_a2_e3;
      } else if (current_a2_e7 == next_a1_e4) {
        current_a2_e7 = next_a2_e4;
      } else if (current_a2_e7 == next_a1_e5) {
        current_a2_e7 = next_a2_e5;
      } else if (current_a2_e7 == next_a1_e6) {
        current_a2_e7 = next_a2_e6;
      } else if (current_a2_e7 == next_a1_e7) {
        current_a2_e7 = next_a2_e7;
      }*/
    }

    __syncthreads();
  }

	//Write shared memory to global memory for verification
	/*if (thread_id == 0) {
		for (int i = 0; i < size; i++) {
			arr1_index = (thread_id * 2 * size) + i;
			global_arrays[arr1_index] = current_arr1[i];

			arr2_index = (thread_id * 2 * size) + size + i;
			global_arrays[arr2_index] = current_arr2[i];
		}
	}*/

  if (thread_id == 0) {
    global_arrays[0] = current_a1_e0;
    global_arrays[1] = current_a1_e1;
    global_arrays[2] = current_a1_e2;
    global_arrays[3] = current_a1_e3;

    global_arrays[4] = current_a1_e4;
    global_arrays[5] = current_a1_e5;
    global_arrays[6] = current_a1_e6;
    global_arrays[7] = current_a1_e7;

    global_arrays[size + 0] = current_a2_e0;
    global_arrays[size + 1] = current_a2_e1;
    global_arrays[size + 2] = current_a2_e2;
    global_arrays[size + 3] = current_a2_e3;

    global_arrays[12] = current_a2_e4;
    global_arrays[13] = current_a2_e5;
    global_arrays[14] = current_a2_e6;
    global_arrays[15] = current_a2_e7;
  }
}

__global__ void shfl_array_match(int* global_arrays, int num_threads) {
  int thread_id = (blockIdx.x * blockDim.x) + threadIdx.x;
	extern __shared__ int shared_arrays[];
	//int current_arr1[ARRAY_SIZE];
	//int current_arr2[ARRAY_SIZE];
  //int next_arr1[ARRAY_SIZE];
  //int next_arr2[ARRAY_SIZE];
	int size = ARRAY_SIZE;
  //int arr1_index = 0;
  //int arr2_index = 0;
  unsigned int mask = 0xffffffff;

  //Stage 0: Retrieve from global memory
  /*for (int i = 0; i < size; i++) {
    arr1_index = (thread_id * 2 * size) + i;
    current_arr1[i] = global_arrays[arr1_index];

    arr2_index = (thread_id * 2 * size) + size + i;
    current_arr2[i] = global_arrays[arr2_index];
  }*/

  //Enumerated version
  //Current array 1
  int current_a1_e0 = global_arrays[0];
  int current_a1_e1 = global_arrays[1];
  int current_a1_e2 = global_arrays[2];
  int current_a1_e3 = global_arrays[3];

  int current_a1_e4 = global_arrays[4];
  int current_a1_e5 = global_arrays[5];
  int current_a1_e6 = global_arrays[6];
  int current_a1_e7 = global_arrays[7];

  //Current array 2
  int current_a2_e0 = global_arrays[size + 0];
  int current_a2_e1 = global_arrays[size + 1];
  int current_a2_e2 = global_arrays[size + 2];
  int current_a2_e3 = global_arrays[size + 3];

  int current_a2_e4 = global_arrays[size + 4];
  int current_a2_e5 = global_arrays[size + 5];
  int current_a2_e6 = global_arrays[size + 6];
  int current_a2_e7 = global_arrays[size + 7];

  //Init next array 1
  int next_a1_e0 = 0;
  int next_a1_e1 = 0;
  int next_a1_e2 = 0;
  int next_a1_e3 = 0;

  int next_a1_e4 = 0;
  int next_a1_e5 = 0;
  int next_a1_e6 = 0;
  int next_a1_e7 = 0;

  //Init next array 2
  int next_a2_e0 = 0;
  int next_a2_e1 = 0;
  int next_a2_e2 = 0;
  int next_a2_e3 = 0;

  int next_a2_e4 = 0;
  int next_a2_e5 = 0;
  int next_a2_e6 = 0;
  int next_a2_e7 = 0;

  //Stage 1: Match by shuffle arrays with tree like reduction
  for (int delta = 1; delta < WARP_SIZE; delta = delta << 1) {

    //Retrieve value from register from thread_id + delta
    /*for (int i = 0; i < size; i++) {
      next_arr1[i] = __shfl_down_sync(mask, current_arr1[i], delta, WARP_SIZE);
      next_arr2[i] = __shfl_down_sync(mask, current_arr2[i], delta, WARP_SIZE);
    }*/

    next_a1_e0 = __shfl_down_sync(mask, current_a1_e0, delta, WARP_SIZE);
    next_a2_e0 = __shfl_down_sync(mask, current_a2_e0, delta, WARP_SIZE);

    next_a1_e1 = __shfl_down_sync(mask, current_a1_e1, delta, WARP_SIZE);
    next_a2_e1 = __shfl_down_sync(mask, current_a2_e1, delta, WARP_SIZE);

    next_a1_e2 = __shfl_down_sync(mask, current_a1_e2, delta, WARP_SIZE);
    next_a2_e2 = __shfl_down_sync(mask, current_a2_e2, delta, WARP_SIZE);

    next_a1_e3 = __shfl_down_sync(mask, current_a1_e3, delta, WARP_SIZE);
    next_a2_e3 = __shfl_down_sync(mask, current_a2_e3, delta, WARP_SIZE);

    next_a1_e4 = __shfl_down_sync(mask, current_a1_e4, delta, WARP_SIZE);
    next_a2_e4 = __shfl_down_sync(mask, current_a2_e4, delta, WARP_SIZE);

    next_a1_e5 = __shfl_down_sync(mask, current_a1_e5, delta, WARP_SIZE);
    next_a2_e5 = __shfl_down_sync(mask, current_a2_e5, delta, WARP_SIZE);

    next_a1_e6 = __shfl_down_sync(mask, current_a1_e6, delta, WARP_SIZE);
    next_a2_e6 = __shfl_down_sync(mask, current_a2_e6, delta, WARP_SIZE);

    next_a1_e7 = __shfl_down_sync(mask, current_a1_e7, delta, WARP_SIZE);
    next_a2_e7 = __shfl_down_sync(mask, current_a2_e7, delta, WARP_SIZE);

    if ((thread_id % (delta * 2)) == 0) {
      //current_a2_e0
      if (current_a2_e0 == next_a1_e0) {
        current_a2_e0 = next_a2_e0;
      } else if (current_a2_e0 == next_a1_e1) {
        current_a2_e0 = next_a2_e1;
      } else if (current_a2_e0 == next_a1_e2) {
        current_a2_e0 = next_a2_e2;
      } else if (current_a2_e0 == next_a1_e3) {
        current_a2_e0 = next_a2_e3;
      } else if (current_a2_e0 == next_a1_e4) {
        current_a2_e0 = next_a2_e4;
      } else if (current_a2_e0 == next_a1_e5) {
        current_a2_e0 = next_a2_e5;
      } else if (current_a2_e0 == next_a1_e6) {
        current_a2_e0 = next_a2_e6;
      } else if (current_a2_e0 == next_a1_e7) {
        current_a2_e0 = next_a2_e7;
      }

      //current_a2_e1
      if (current_a2_e1 == next_a1_e0) {
        current_a2_e1 = next_a2_e0;
      } else if (current_a2_e1 == next_a1_e1) {
        current_a2_e1 = next_a2_e1;
      } else if (current_a2_e1 == next_a1_e2) {
        current_a2_e1 = next_a2_e2;
      } else if (current_a2_e1 == next_a1_e3) {
        current_a2_e1 = next_a2_e3;
      } else if (current_a2_e1 == next_a1_e4) {
        current_a2_e1 = next_a2_e4;
      } else if (current_a2_e1 == next_a1_e5) {
        current_a2_e1 = next_a2_e5;
      } else if (current_a2_e1 == next_a1_e6) {
        current_a2_e1 = next_a2_e6;
      } else if (current_a2_e1 == next_a1_e7) {
        current_a2_e1 = next_a2_e7;
      }

      //current_a2_e2
      if (current_a2_e2 == next_a1_e0) {
        current_a2_e2 = next_a2_e0;
      } else if (current_a2_e2 == next_a1_e1) {
        current_a2_e2 = next_a2_e1;
      } else if (current_a2_e2 == next_a1_e2) {
        current_a2_e2 = next_a2_e2;
      } else if (current_a2_e2 == next_a1_e3) {
        current_a2_e2 = next_a2_e3;
      } else if (current_a2_e2 == next_a1_e4) {
        current_a2_e2 = next_a2_e4;
      } else if (current_a2_e2 == next_a1_e5) {
        current_a2_e2 = next_a2_e5;
      } else if (current_a2_e2 == next_a1_e6) {
        current_a2_e2 = next_a2_e6;
      } else if (current_a2_e2 == next_a1_e7) {
        current_a2_e2 = next_a2_e7;
      }

      //current_a2_e3
      if (current_a2_e3 == next_a1_e0) {
        current_a2_e3 = next_a2_e0;
      } else if (current_a2_e3 == next_a1_e1) {
        current_a2_e3 = next_a2_e1;
      } else if (current_a2_e3 == next_a1_e2) {
        current_a2_e3 = next_a2_e2;
      } else if (current_a2_e3 == next_a1_e3) {
        current_a2_e3 = next_a2_e3;
      } else if (current_a2_e3 == next_a1_e4) {
        current_a2_e3 = next_a2_e4;
      } else if (current_a2_e3 == next_a1_e5) {
        current_a2_e3 = next_a2_e5;
      } else if (current_a2_e3 == next_a1_e6) {
        current_a2_e3 = next_a2_e6;
      } else if (current_a2_e3 == next_a1_e7) {
        current_a2_e3 = next_a2_e7;
      }

      //current_a2_e4
      if (current_a2_e4 == next_a1_e0) {
        current_a2_e4 = next_a2_e0;
      } else if (current_a2_e4 == next_a1_e1) {
        current_a2_e4 = next_a2_e1;
      } else if (current_a2_e4 == next_a1_e2) {
        current_a2_e4 = next_a2_e2;
      } else if (current_a2_e4 == next_a1_e3) {
        current_a2_e4 = next_a2_e3;
      } else if (current_a2_e4 == next_a1_e4) {
        current_a2_e4 = next_a2_e4;
      } else if (current_a2_e4 == next_a1_e5) {
        current_a2_e4 = next_a2_e5;
      } else if (current_a2_e4 == next_a1_e6) {
        current_a2_e4 = next_a2_e6;
      } else if (current_a2_e4 == next_a1_e7) {
        current_a2_e4 = next_a2_e7;
      }

      //current_a2_e5
      if (current_a2_e5 == next_a1_e0) {
        current_a2_e5 = next_a2_e0;
      } else if (current_a2_e5 == next_a1_e1) {
        current_a2_e5 = next_a2_e1;
      } else if (current_a2_e5 == next_a1_e2) {
        current_a2_e5 = next_a2_e2;
      } else if (current_a2_e5 == next_a1_e3) {
        current_a2_e5 = next_a2_e3;
      } else if (current_a2_e5 == next_a1_e4) {
        current_a2_e5 = next_a2_e4;
      } else if (current_a2_e5 == next_a1_e5) {
        current_a2_e5 = next_a2_e5;
      } else if (current_a2_e5 == next_a1_e6) {
        current_a2_e5 = next_a2_e6;
      } else if (current_a2_e5 == next_a1_e7) {
        current_a2_e5 = next_a2_e7;
      }

      //current_a2_e6
      if (current_a2_e6 == next_a1_e0) {
        current_a2_e6 = next_a2_e0;
      } else if (current_a2_e6 == next_a1_e1) {
        current_a2_e6 = next_a2_e1;
      } else if (current_a2_e6 == next_a1_e2) {
        current_a2_e6 = next_a2_e2;
      } else if (current_a2_e6 == next_a1_e3) {
        current_a2_e6 = next_a2_e3;
      } else if (current_a2_e6 == next_a1_e4) {
        current_a2_e6 = next_a2_e4;
      } else if (current_a2_e6 == next_a1_e5) {
        current_a2_e6 = next_a2_e5;
      } else if (current_a2_e6 == next_a1_e6) {
        current_a2_e6 = next_a2_e6;
      } else if (current_a2_e6 == next_a1_e7) {
        current_a2_e6 = next_a2_e7;
      }

      //current_a2_e7
      if (current_a2_e7 == next_a1_e0) {
        current_a2_e7 = next_a2_e0;
      } else if (current_a2_e7 == next_a1_e1) {
        current_a2_e7 = next_a2_e1;
      } else if (current_a2_e7 == next_a1_e2) {
        current_a2_e7 = next_a2_e2;
      } else if (current_a2_e7 == next_a1_e3) {
        current_a2_e7 = next_a2_e3;
      } else if (current_a2_e7 == next_a1_e4) {
        current_a2_e7 = next_a2_e4;
      } else if (current_a2_e7 == next_a1_e5) {
        current_a2_e7 = next_a2_e5;
      } else if (current_a2_e7 == next_a1_e6) {
        current_a2_e7 = next_a2_e6;
      } else if (current_a2_e7 == next_a1_e7) {
        current_a2_e7 = next_a2_e7;
      }
    }
  }

  if (num_threads > WARP_SIZE) {

    //Stage 2: Warp thread 0 write warp shuffle result to shared memory
    if ((thread_id % WARP_SIZE) == 0) {

      /*for(int i = 0; i < size; i++) {
        arr1_index = ((thread_id / WARP_SIZE) * 2 * size) + i;
        shared_arrays[arr1_index] = current_arr1[i];
      }

      for(int i = 0; i < size; i++) {
        arr2_index = ((thread_id / WARP_SIZE) * 2 * size) + size + i;
        shared_arrays[arr2_index] = current_arr2[i];
      }*/

      shared_arrays[((thread_id / WARP_SIZE) * 2 * size) + 0] = current_a1_e0;
      shared_arrays[((thread_id / WARP_SIZE) * 2 * size) + size + 0] = current_a2_e0;

      shared_arrays[((thread_id / WARP_SIZE) * 2 * size) + 1] = current_a1_e1;
      shared_arrays[((thread_id / WARP_SIZE) * 2 * size) + size + 1] = current_a2_e1;

      shared_arrays[((thread_id / WARP_SIZE) * 2 * size) + 2] = current_a1_e2;
      shared_arrays[((thread_id / WARP_SIZE) * 2 * size) + size + 2] = current_a2_e2;

      shared_arrays[((thread_id / WARP_SIZE) * 2 * size) + 3] = current_a1_e3;
      shared_arrays[((thread_id / WARP_SIZE) * 2 * size) + size + 3] = current_a2_e3;

      shared_arrays[((thread_id / WARP_SIZE) * 2 * size) + 4] = current_a1_e4;
      shared_arrays[((thread_id / WARP_SIZE) * 2 * size) + size + 4] = current_a2_e4;

      shared_arrays[((thread_id / WARP_SIZE) * 2 * size) + 5] = current_a1_e5;
      shared_arrays[((thread_id / WARP_SIZE) * 2 * size) + size + 5] = current_a2_e5;

      shared_arrays[((thread_id / WARP_SIZE) * 2 * size) + 6] = current_a1_e6;
      shared_arrays[((thread_id / WARP_SIZE) * 2 * size) + size + 6] = current_a2_e6;

      shared_arrays[((thread_id / WARP_SIZE) * 2 * size) + 7] = current_a1_e7;
      shared_arrays[((thread_id / WARP_SIZE) * 2 * size) + size + 7] = current_a2_e7;
    }

    __syncthreads();

    //Stage 3: Read all warps 0 thread from shared memory
    if (thread_id < WARP_SIZE) {
      current_a1_e0 = shared_arrays[(thread_id * 2 * size) + 0];
      current_a2_e0 = shared_arrays[(thread_id * 2 * size) + size + 0];

      current_a1_e1 = shared_arrays[(thread_id * 2 * size) + 1];
      current_a2_e1 = shared_arrays[(thread_id * 2 * size) + size + 1];

      current_a1_e2 = shared_arrays[(thread_id * 2 * size) + 2];
      current_a2_e2 = shared_arrays[(thread_id * 2 * size) + size + 2];

      current_a1_e3 = shared_arrays[(thread_id * 2 * size) + 3];
      current_a2_e3 = shared_arrays[(thread_id * 2 * size) + size + 3];

      current_a1_e4 = shared_arrays[(thread_id * 2 * size) + 4];
      current_a2_e4 = shared_arrays[(thread_id * 2 * size) + size + 4];

      current_a1_e5 = shared_arrays[(thread_id * 2 * size) + 5];
      current_a2_e5 = shared_arrays[(thread_id * 2 * size) + size + 5];

      current_a1_e6 = shared_arrays[(thread_id * 2 * size) + 6];
      current_a2_e6 = shared_arrays[(thread_id * 2 * size) + size + 6];

      current_a1_e7 = shared_arrays[(thread_id * 2 * size) + 7];
      current_a2_e7 = shared_arrays[(thread_id * 2 * size) + size + 7];
    }

    __syncthreads();

    //Stage 4: Shuffle again
    if (thread_id < WARP_SIZE) {

      // Tree like reduction, notice for loop condition
      for (int delta = 1; delta < (num_threads / WARP_SIZE); delta = delta << 1) {

        next_a1_e0 = __shfl_down_sync(mask, current_a1_e0, delta, WARP_SIZE);
        next_a2_e0 = __shfl_down_sync(mask, current_a2_e0, delta, WARP_SIZE);

        next_a1_e1 = __shfl_down_sync(mask, current_a1_e1, delta, WARP_SIZE);
        next_a2_e1 = __shfl_down_sync(mask, current_a2_e1, delta, WARP_SIZE);

        next_a1_e2 = __shfl_down_sync(mask, current_a1_e2, delta, WARP_SIZE);
        next_a2_e2 = __shfl_down_sync(mask, current_a2_e2, delta, WARP_SIZE);

        next_a1_e3 = __shfl_down_sync(mask, current_a1_e3, delta, WARP_SIZE);
        next_a2_e3 = __shfl_down_sync(mask, current_a2_e3, delta, WARP_SIZE);

        next_a1_e4 = __shfl_down_sync(mask, current_a1_e4, delta, WARP_SIZE);
        next_a2_e4 = __shfl_down_sync(mask, current_a2_e4, delta, WARP_SIZE);

        next_a1_e5 = __shfl_down_sync(mask, current_a1_e5, delta, WARP_SIZE);
        next_a2_e5 = __shfl_down_sync(mask, current_a2_e5, delta, WARP_SIZE);

        next_a1_e6 = __shfl_down_sync(mask, current_a1_e6, delta, WARP_SIZE);
        next_a2_e6 = __shfl_down_sync(mask, current_a2_e6, delta, WARP_SIZE);

        next_a1_e7 = __shfl_down_sync(mask, current_a1_e7, delta, WARP_SIZE);
        next_a2_e7 = __shfl_down_sync(mask, current_a2_e7, delta, WARP_SIZE);

        if ((thread_id % (delta * 2)) == 0) {
          //current_a2_e0
          if (current_a2_e0 == next_a1_e0) {
            current_a2_e0 = next_a2_e0;
          } else if (current_a2_e0 == next_a1_e1) {
            current_a2_e0 = next_a2_e1;
          } else if (current_a2_e0 == next_a1_e2) {
            current_a2_e0 = next_a2_e2;
          } else if (current_a2_e0 == next_a1_e3) {
            current_a2_e0 = next_a2_e3;
          } else if (current_a2_e0 == next_a1_e4) {
            current_a2_e0 = next_a2_e4;
          } else if (current_a2_e0 == next_a1_e5) {
            current_a2_e0 = next_a2_e5;
          } else if (current_a2_e0 == next_a1_e6) {
            current_a2_e0 = next_a2_e6;
          } else if (current_a2_e0 == next_a1_e7) {
            current_a2_e0 = next_a2_e7;
          }

          //current_a2_e1
          if (current_a2_e1 == next_a1_e0) {
            current_a2_e1 = next_a2_e0;
          } else if (current_a2_e1 == next_a1_e1) {
            current_a2_e1 = next_a2_e1;
          } else if (current_a2_e1 == next_a1_e2) {
            current_a2_e1 = next_a2_e2;
          } else if (current_a2_e1 == next_a1_e3) {
            current_a2_e1 = next_a2_e3;
          } else if (current_a2_e1 == next_a1_e4) {
            current_a2_e1 = next_a2_e4;
          } else if (current_a2_e1 == next_a1_e5) {
            current_a2_e1 = next_a2_e5;
          } else if (current_a2_e1 == next_a1_e6) {
            current_a2_e1 = next_a2_e6;
          } else if (current_a2_e1 == next_a1_e7) {
            current_a2_e1 = next_a2_e7;
          }

          //current_a2_e2
          if (current_a2_e2 == next_a1_e0) {
            current_a2_e2 = next_a2_e0;
          } else if (current_a2_e2 == next_a1_e1) {
            current_a2_e2 = next_a2_e1;
          } else if (current_a2_e2 == next_a1_e2) {
            current_a2_e2 = next_a2_e2;
          } else if (current_a2_e2 == next_a1_e3) {
            current_a2_e2 = next_a2_e3;
          } else if (current_a2_e2 == next_a1_e4) {
            current_a2_e2 = next_a2_e4;
          } else if (current_a2_e2 == next_a1_e5) {
            current_a2_e2 = next_a2_e5;
          } else if (current_a2_e2 == next_a1_e6) {
            current_a2_e2 = next_a2_e6;
          } else if (current_a2_e2 == next_a1_e7) {
            current_a2_e2 = next_a2_e7;
          }

          //current_a2_e3
          if (current_a2_e3 == next_a1_e0) {
            current_a2_e3 = next_a2_e0;
          } else if (current_a2_e3 == next_a1_e1) {
            current_a2_e3 = next_a2_e1;
          } else if (current_a2_e3 == next_a1_e2) {
            current_a2_e3 = next_a2_e2;
          } else if (current_a2_e3 == next_a1_e3) {
            current_a2_e3 = next_a2_e3;
          } else if (current_a2_e3 == next_a1_e4) {
            current_a2_e3 = next_a2_e4;
          } else if (current_a2_e3 == next_a1_e5) {
            current_a2_e3 = next_a2_e5;
          } else if (current_a2_e3 == next_a1_e6) {
            current_a2_e3 = next_a2_e6;
          } else if (current_a2_e3 == next_a1_e7) {
            current_a2_e3 = next_a2_e7;
          }

          //current_a2_e4
          if (current_a2_e4 == next_a1_e0) {
            current_a2_e4 = next_a2_e0;
          } else if (current_a2_e4 == next_a1_e1) {
            current_a2_e4 = next_a2_e1;
          } else if (current_a2_e4 == next_a1_e2) {
            current_a2_e4 = next_a2_e2;
          } else if (current_a2_e4 == next_a1_e3) {
            current_a2_e4 = next_a2_e3;
          } else if (current_a2_e4 == next_a1_e4) {
            current_a2_e4 = next_a2_e4;
          } else if (current_a2_e4 == next_a1_e5) {
            current_a2_e4 = next_a2_e5;
          } else if (current_a2_e4 == next_a1_e6) {
            current_a2_e4 = next_a2_e6;
          } else if (current_a2_e4 == next_a1_e7) {
            current_a2_e4 = next_a2_e7;
          }

          //current_a2_e5
          if (current_a2_e5 == next_a1_e0) {
            current_a2_e5 = next_a2_e0;
          } else if (current_a2_e5 == next_a1_e1) {
            current_a2_e5 = next_a2_e1;
          } else if (current_a2_e5 == next_a1_e2) {
            current_a2_e5 = next_a2_e2;
          } else if (current_a2_e5 == next_a1_e3) {
            current_a2_e5 = next_a2_e3;
          } else if (current_a2_e5 == next_a1_e4) {
            current_a2_e5 = next_a2_e4;
          } else if (current_a2_e5 == next_a1_e5) {
            current_a2_e5 = next_a2_e5;
          } else if (current_a2_e5 == next_a1_e6) {
            current_a2_e5 = next_a2_e6;
          } else if (current_a2_e5 == next_a1_e7) {
            current_a2_e5 = next_a2_e7;
          }

          //current_a2_e6
          if (current_a2_e6 == next_a1_e0) {
            current_a2_e6 = next_a2_e0;
          } else if (current_a2_e6 == next_a1_e1) {
            current_a2_e6 = next_a2_e1;
          } else if (current_a2_e6 == next_a1_e2) {
            current_a2_e6 = next_a2_e2;
          } else if (current_a2_e6 == next_a1_e3) {
            current_a2_e6 = next_a2_e3;
          } else if (current_a2_e6 == next_a1_e4) {
            current_a2_e6 = next_a2_e4;
          } else if (current_a2_e6 == next_a1_e5) {
            current_a2_e6 = next_a2_e5;
          } else if (current_a2_e6 == next_a1_e6) {
            current_a2_e6 = next_a2_e6;
          } else if (current_a2_e6 == next_a1_e7) {
            current_a2_e6 = next_a2_e7;
          }

          //current_a2_e7
          if (current_a2_e7 == next_a1_e0) {
            current_a2_e7 = next_a2_e0;
          } else if (current_a2_e7 == next_a1_e1) {
            current_a2_e7 = next_a2_e1;
          } else if (current_a2_e7 == next_a1_e2) {
            current_a2_e7 = next_a2_e2;
          } else if (current_a2_e7 == next_a1_e3) {
            current_a2_e7 = next_a2_e3;
          } else if (current_a2_e7 == next_a1_e4) {
            current_a2_e7 = next_a2_e4;
          } else if (current_a2_e7 == next_a1_e5) {
            current_a2_e7 = next_a2_e5;
          } else if (current_a2_e7 == next_a1_e6) {
            current_a2_e7 = next_a2_e6;
          } else if (current_a2_e7 == next_a1_e7) {
            current_a2_e7 = next_a2_e7;
          }
      }
    }
  }
}

  //Stage 5: Write back to global memory
  if (thread_id == 0) {
		/*for (int i = 0; i < size; i++) {
			arr1_index = (thread_id * 2 * size) + i;
			global_arrays[arr1_index] = current_arr1[i];

			arr2_index = (thread_id * 2 * size) + size + i;
			global_arrays[arr2_index] = current_arr2[i];
		}*/

    global_arrays[0] = current_a1_e0;
    global_arrays[1] = current_a1_e1;
    global_arrays[2] = current_a1_e2;
    global_arrays[3] = current_a1_e3;

    global_arrays[4] = current_a1_e4;
    global_arrays[5] = current_a1_e5;
    global_arrays[6] = current_a1_e6;
    global_arrays[7] = current_a1_e7;

    global_arrays[size + 0] = current_a2_e0;
    global_arrays[size + 1] = current_a2_e1;
    global_arrays[size + 2] = current_a2_e2;
    global_arrays[size + 3] = current_a2_e3;

    global_arrays[12] = current_a2_e4;
    global_arrays[13] = current_a2_e5;
    global_arrays[14] = current_a2_e6;
    global_arrays[15] = current_a2_e7;
	}
}

void cpu_array_match(int* arrays, int num_threads, int array_size) {
  for (int i = 1; i < num_threads; i++) {
    int* next_arr1 = arrays + (i * 2 * array_size);
    int* next_arr2 = arrays + (i * 2 * array_size) + array_size;

    for (int j = 0; j < array_size; j++) {

      for (int k = 0; k < array_size; k++) {
        if (arrays[array_size + j] == next_arr1[k]) {
          arrays[array_size + j] = next_arr2[k];
          break;
        }
      }

    }
  }

}

int main(int argc, char** argv) {

	/***Variable Declarations***/
  float milliseconds;

  int* host_arrays;
  int* experiment1_arrays;
	int* device_arrays;

	int array_size;
  int num_arrays;
	int num_threads;
	int num_blocks;
  int share_size;
  int debug;

	size_t one_t;
	size_t array_set_bytes;

  cudaEvent_t start, stop;
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

  //Copy host arrays to device
  cudaMemcpy(device_arrays, host_arrays, array_set_bytes, cudaMemcpyHostToDevice);

  //Set max dynamic shared memory size to either 96 kibibytes or 64 kibibytes
  share_size = SHM_96_KB;
  cuda_err = cudaFuncSetAttribute(shfl_array_match, cudaFuncAttributeMaxDynamicSharedMemorySize, share_size);

  if (cuda_err != cudaSuccess) {

    if (debug) { cerr << endl << "Dynamic shared memory size of 96kb for array set failed, trying 64kb" << endl; }
    share_size = SHM_64_KB;

    cuda_err = cudaFuncSetAttribute(shfl_array_match, cudaFuncAttributeMaxDynamicSharedMemorySize, share_size);

    if (cuda_err != cudaSuccess) {

      if (debug) { cerr << "Dynamic shared memory size of 64000 for array set failed. Exiting program..." << endl; }

      return -1;
    }
  }

  if (debug) {
    cout << endl << "***Experiment1***" << endl;
    cout << "--------------------KERNEL CALL--------------------" << endl;
  }

  //Timing
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  //Kernel call
  shm_array_match <<<num_blocks, num_threads, share_size>>> (device_arrays, num_threads);

  //Timing
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  //Copy device arrays back to host
  cudaMemcpy(experiment1_arrays, device_arrays, array_set_bytes, cudaMemcpyDeviceToHost);

  cout << 1 << "," << num_threads << "," << array_size << "," << milliseconds << endl;


  /************************Experiment 1***************************************/

  //Set max dynamic shared memory size to either 96 kibibytes or 64 kibibytes
  share_size = SHM_96_KB;
  cuda_err = cudaFuncSetAttribute(shfl_array_match, cudaFuncAttributeMaxDynamicSharedMemorySize, share_size);

  if (cuda_err != cudaSuccess) {

    if (debug) { cerr << endl << "Dynamic shared memory size of 96kb for array set failed, trying 64kb" << endl; }
    share_size = SHM_64_KB;

    cuda_err = cudaFuncSetAttribute(shfl_array_match, cudaFuncAttributeMaxDynamicSharedMemorySize, share_size);

    if (cuda_err != cudaSuccess) {

      if (debug) { cerr << "Dynamic shared memory size of 64000 for array set failed. Exiting program..." << endl; }

      return -1;
    }
	}

  //Copy host arrays to device
  cudaMemcpy(device_arrays, host_arrays, array_set_bytes, cudaMemcpyHostToDevice);

  if (debug) {
    cout << endl << "***Experiment2***" << endl;
    cout << "--------------------KERNEL CALL--------------------" << endl;
  }

  //Timing
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  //Kernel call
  shfl_array_match <<<num_blocks, num_threads, share_size>>> (device_arrays, num_threads);

  //Timing
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  //Copy device arrays back to host
  cudaMemcpy(experiment1_arrays, device_arrays, array_set_bytes, cudaMemcpyDeviceToHost);

  cout << 1 << "," << num_threads << "," << array_size << "," << milliseconds << endl;

	/***Free variables***/
	cudaFree(device_arrays);
	free(host_arrays);
  free(experiment1_arrays);

	return 0;
}

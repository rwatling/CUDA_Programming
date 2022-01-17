/* This program searches "states" for matches in their arrays using CUDA
*
* Author: Robbie Watling
*/

#include "cuda_includes.h"
#include "shm_array_match.h"
#include "shfl_array_match.h"
#include "shfl_hash_match.h"
#include "cpu_array_match.h"
#include "shfl_unroll_match.h"
#include "shfl_unroll2_match.h"
#include "shfl_bs_match.h"
#include "shfl_hash_w_shared_match.h"
#include "nvmlClass.h"
#include "shuffle.h"

#include <iostream>
#include <unistd.h>

#define SHM_96_KB 98304
#define SHM_64_KB 65536

using namespace std;

int main(int argc, char** argv) {

	/***Variable Declarations***/
  float milliseconds;

  int* host_arrays;
  int* experiment_arrays;
	int* device_arrays;

	int array_size;
	int num_threads;
	int num_blocks;
  int share_size;

	size_t one_t;
	size_t array_set_bytes;

  //NVML
  string base_nvml_filename = "./analysis/data/hardware_stats";
  string nvml_filename;
  string type;
  vector<thread> cpu_threads;

  //CUDA Error Checking
  cudaEvent_t start, stop;
  cudaError_t cuda_err;

	/*** Read args ***/
	if (argc < 2) {
		cerr << "./gpu_match num_operating_threads" << endl;
		return -1;
	}

	/***Initialization***/
	array_size = ARRAY_SIZE;
	num_threads = atoi(argv[1]);
	num_blocks = 1;
  share_size = SHM_64_KB;

  unsigned int temp_num_threads = num_threads;
  bool pow_2 = temp_num_threads && !(temp_num_threads & (temp_num_threads - 1));
  if (!pow_2) {
    cerr << "Number of threads not a power of 2" << endl;
    return -1;
  }

  /* Defined by compiler flags:
    ARRAY_SIZE
    DEBUG
  */

	//Host allocation
	one_t = (size_t) 1;
	array_set_bytes = (size_t) num_threads * array_size * 2 * sizeof(int);
  host_arrays = (int*) calloc(one_t, array_set_bytes);

	if (host_arrays == NULL) {
		cerr << "Host arrays calloc failed\n" << endl;
		return -1;
	}

  //Experiment arrays allocation
  experiment_arrays = (int*) calloc(one_t, array_set_bytes);

  if (experiment_arrays == NULL) {
		cerr << "experiment arrays calloc failed\n" << endl;
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
  if (DEBUG) {
    for(int i = 0; i < num_threads; i++) {

      cout << "Arrays " << i << ": [";

  		for(int j = 0; j < array_size * 2; j++) {
  			cout << host_arrays[(i * array_size * 2) + j] << " ";

        if (j == array_size - 1) { cout << "]\t["; }
  		}

      cout << "]" << endl;
  	}
  }

  /************************NVML get device********************************/
  int dev {};
  cudaGetDevice( &dev );
  cuda_err = cudaSetDevice( dev );

  if (cuda_err != cudaSuccess) {
		cerr << "cudaSetDevice failed for nvml\n" << endl;
		return -1;
	}

  /************************Experiment 1***************************************/

  //Set max dynamic shared memory size to either 96 kibibytes or 64 kibibytes
  share_size = SHM_96_KB;
  cuda_err = cudaFuncSetAttribute(shm_array_match, cudaFuncAttributeMaxDynamicSharedMemorySize, share_size);

  if (cuda_err != cudaSuccess) {

    if (DEBUG) { cerr << endl << "Dynamic shared memory size of 96kb for array set failed, trying 64kb" << endl; }
    share_size = SHM_64_KB;

    cuda_err = cudaFuncSetAttribute(shm_array_match, cudaFuncAttributeMaxDynamicSharedMemorySize, share_size);

    if (cuda_err != cudaSuccess) {

      if (DEBUG) { cerr << "Dynamic shared memory size of 64000 for array set failed. Exiting program..." << endl; }

      return -1;
    }
	}

  //Copy host arrays to device
  cudaMemcpy(device_arrays, host_arrays, array_set_bytes, cudaMemcpyHostToDevice);

  if (DEBUG) {
    cout << endl << "***Experiment 1 Shared Mem***" << endl;

    cout << "--------------------KERNEL CALL--------------------" << endl;
  }


  //Create nvml class
  nvml_filename.append(base_nvml_filename);
  nvml_filename.append("_shm_nested.csv");
  type.append("shm_nested");
  nvmlClass nvml( dev, nvml_filename, type);

  cpu_threads.emplace_back(thread(&nvmlClass::getStats, &nvml));

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

  // NVML
  // Create thread to kill GPU stats
  // Join both threads to main
  cpu_threads.emplace_back(thread( &nvmlClass::killThread, &nvml));

  for (auto& th : cpu_threads) {
    th.join();
    th.~thread();
  }

  cpu_threads.clear();
  nvml_filename.clear();
  type.clear();

  //Copy device arrays back to host
  cudaMemcpy(experiment_arrays, device_arrays, array_set_bytes, cudaMemcpyDeviceToHost);

  if (DEBUG) {
    //Print arrays after matching
    for(int i = 0; i < 1; i++) {

      cout << "Arrays " << i << ": [";

      for(int j = 0; j < array_size * 2; j++) {
        cout << experiment_arrays[(i * array_size * 2) + j] << " ";

        if (j == array_size - 1) { cout << "]\t["; }
      }

      cout << "]" << endl;
    }
  }

  cout << "Nested Shm" << "," << num_threads << "," << array_size << "," << milliseconds << endl;

  /************************Experiment 2***************************************/
  //Set max dynamic shared memory size to either 96 kibibytes or 64 kibibytes
  cuda_err = cudaFuncSetAttribute(shfl_array_match, cudaFuncAttributeMaxDynamicSharedMemorySize, share_size);

  if (cuda_err != cudaSuccess) {
    if (DEBUG) { cerr << endl << "Second attempt of defining dynamic shared memory size of 96kb for array set failed" << endl << endl; }
    return -1;
	}

  //Copy host arrays to device
  cudaMemcpy(device_arrays, host_arrays, array_set_bytes, cudaMemcpyHostToDevice);

  if (DEBUG) {
    cout << endl << "***Experiment 2 Shuffle***" << endl;

    cout << "--------------------KERNEL CALL--------------------" << endl;
  }

  //Create new nvml file
  nvml_filename.append(base_nvml_filename);
  nvml_filename.append("_shfl_nested.csv");
  type.append("shfl_nested");
  nvml.new_experiment(nvml_filename, type);

  cpu_threads.emplace_back(thread(&nvmlClass::getStats, &nvml));

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

  // NVML
  // Create thread to kill GPU stats
  // Join both threads to main
  cpu_threads.emplace_back(thread( &nvmlClass::killThread, &nvml));

  for (auto& th : cpu_threads) {
    th.join();
    th.~thread();
  }

  cpu_threads.clear();
  nvml_filename.clear();
  type.clear();

  cout << "Nested Shfl" << "," << num_threads << "," << array_size << "," << milliseconds << endl;

  //Copy device arrays back to host
  cudaMemcpy(experiment_arrays, device_arrays, array_set_bytes, cudaMemcpyDeviceToHost);

  if (DEBUG) {
    //Print arrays after matching
    for(int i = 0; i < 1; i++) {

      cout << "Arrays " << i << ": [";

      for(int j = 0; j < array_size * 2; j++) {
        cout << experiment_arrays[(i * array_size * 2) + j] << " ";

        if (j == array_size - 1) { cout << "]\t["; }
      }

      cout << "]" << endl;
    }
  }

  /************************Experiment 3***************************************/
  //Set max dynamic shared memory size to either 96 kibibytes or 64 kibibytes
  cuda_err = cudaFuncSetAttribute(shfl_hash_match, cudaFuncAttributeMaxDynamicSharedMemorySize, share_size);

  if (cuda_err != cudaSuccess) {
    if (DEBUG) { cerr << endl << "Third attempt of defining dynamic shared memory size of 96kb for array set failed" << endl << endl; }
    return -1;
	}

  if (DEBUG) {
    cout << endl << "***Experiment 3 Shuffle with Hash***" << endl;

    cout << "--------------------KERNEL CALL--------------------" << endl;
  }

  //Copy host arrays to device
  cudaMemcpy(device_arrays, host_arrays, array_set_bytes, cudaMemcpyHostToDevice);

  //Create new nvml file
  nvml_filename.append(base_nvml_filename);
  nvml_filename.append("_shfl_hash.csv");
  type.append("shfl_hash");
  nvml.new_experiment(nvml_filename, type);

  cpu_threads.emplace_back(thread(&nvmlClass::getStats, &nvml));

  //Timing
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  //Kernel call
  shfl_hash_match<<<num_blocks, num_threads, share_size>>>(device_arrays, num_threads);

  //Timing
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  // NVML
  // Create thread to kill GPU stats
  // Join both threads to main
  cpu_threads.emplace_back(thread( &nvmlClass::killThread, &nvml));

  for (auto& th : cpu_threads) {
    th.join();
    th.~thread();
  }

  cpu_threads.clear();
  nvml_filename.clear();
  type.clear();

  cout << "Shfl Hash" << "," << num_threads << "," << array_size << "," << milliseconds << endl;

  //Copy device arrays back to host
  cudaMemcpy(experiment_arrays, device_arrays, array_set_bytes, cudaMemcpyDeviceToHost);

  if (DEBUG) {
    //Print arrays after matching
    for(int i = 0; i < 1; i++) {

      cout << "Arrays " << i << ": [";

      for(int j = 0; j < array_size * 2; j++) {
        cout << experiment_arrays[(i * array_size * 2) + j] << " ";

        if (j == array_size - 1) { cout << "]\t["; }
      }

      cout << "]" << endl;
    }
  }

  /************************Experiment 4***************************************/
  //Set max dynamic shared memory size to either 96 kibibytes or 64 kibibytes
  cuda_err = cudaFuncSetAttribute(shfl_unroll2_match, cudaFuncAttributeMaxDynamicSharedMemorySize, share_size);

  if (cuda_err != cudaSuccess) {
    if (DEBUG) { cerr << endl << "Fourth attempt of defining dynamic shared memory size of 96kb for array set failed" << endl << endl; }
    return -1;
  }

  if (DEBUG) {
    cout << endl << "***Experiment 4 Shfl Unroll 2***" << endl;

    cout << "--------------------KERNEL CALL--------------------" << endl;
  }

  //Copy host arrays to device
  cudaMemcpy(device_arrays, host_arrays, array_set_bytes, cudaMemcpyHostToDevice);

  //Create new nvml file
  nvml_filename.append(base_nvml_filename);
  nvml_filename.append("_shfl_unroll2.csv");
  type.append("shfl_unroll2");
  nvml.new_experiment(nvml_filename, type);

  cpu_threads.emplace_back(thread(&nvmlClass::getStats, &nvml));

  //Timing
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  //Kernel call
  shfl_unroll2_match<<<num_blocks, num_threads, share_size>>>(device_arrays, num_threads);

  //Timing
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  // NVML
  // Create thread to kill GPU stats
  // Join both threads to main
  cpu_threads.emplace_back(thread( &nvmlClass::killThread, &nvml));

  for (auto& th : cpu_threads) {
    th.join();
    th.~thread();
  }

  cpu_threads.clear();
  nvml_filename.clear();
  type.clear();

  cout << "Shfl Unroll 2" << "," << num_threads << "," << array_size << "," << milliseconds << endl;

  //Copy device arrays back to host
  cudaMemcpy(experiment_arrays, device_arrays, array_set_bytes, cudaMemcpyDeviceToHost);

  if (DEBUG) {
    //Print arrays after matching
    for(int i = 0; i < 1; i++) {

      cout << "Arrays " << i << ": [";

      for(int j = 0; j < array_size * 2; j++) {
        cout << experiment_arrays[(i * array_size * 2) + j] << " ";

        if (j == array_size - 1) { cout << "]\t["; }
      }

      cout << "]" << endl;
    }
  }

  /************************Experiment 5***************************************/
  //Set max dynamic shared memory size to either 96 kibibytes or 64 kibibytes
  cuda_err = cudaFuncSetAttribute(shfl_unroll_match, cudaFuncAttributeMaxDynamicSharedMemorySize, share_size);

  if (cuda_err != cudaSuccess) {
    if (DEBUG) { cerr << endl << "Fifth attempt of defining dynamic shared memory size of 96kb for array set failed" << endl << endl; }
    return -1;
  }

  if (DEBUG) {
    cout << endl << "***Experiment 5 Shfl with Unroll***" << endl;

    cout << "--------------------KERNEL CALL--------------------" << endl;
  }

  //Copy host arrays to device
  cudaMemcpy(device_arrays, host_arrays, array_set_bytes, cudaMemcpyHostToDevice);

  //Create new nvml file
  nvml_filename.append(base_nvml_filename);
  nvml_filename.append("_shfl_unroll.csv");
  type.append("shfl_unroll");
  nvml.new_experiment(nvml_filename, type);

  cpu_threads.emplace_back(thread(&nvmlClass::getStats, &nvml));

  //Timing
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  //Kernel call
  shfl_unroll_match<<<num_blocks, num_threads, share_size>>>(device_arrays, num_threads);

  //Timing
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  // NVML
  // Create thread to kill GPU stats
  // Join both threads to main
  cpu_threads.emplace_back(thread( &nvmlClass::killThread, &nvml));

  for (auto& th : cpu_threads) {
    th.join();
    th.~thread();
  }

  cpu_threads.clear();
  nvml_filename.clear();
  type.clear();

  cout << "Shfl Unroll" << "," << num_threads << "," << array_size << "," << milliseconds << endl;

  //Copy device arrays back to host
  cudaMemcpy(experiment_arrays, device_arrays, array_set_bytes, cudaMemcpyDeviceToHost);

  if (DEBUG) {
    //Print arrays after matching
    for(int i = 0; i < 1; i++) {

      cout << "Arrays " << i << ": [";

      for(int j = 0; j < array_size * 2; j++) {
        cout << experiment_arrays[(i * array_size * 2) + j] << " ";

        if (j == array_size - 1) { cout << "]\t["; }
      }

      cout << "]" << endl;
    }
  }

  /************************Experiment 6***************************************/
  //Set max dynamic shared memory size to either 96 kibibytes or 64 kibibytes
  cuda_err = cudaFuncSetAttribute(shfl_bs_match, cudaFuncAttributeMaxDynamicSharedMemorySize, share_size);

  if (cuda_err != cudaSuccess) {
    if (DEBUG) { cerr << endl << "Sixth attempt of defining dynamic shared memory size of 96kb for array set failed" << endl << endl; }
    return -1;
  }

  if (DEBUG) {
    cout << endl << "***Experiment 6 Shfl with Binary Search***" << endl;

    cout << "--------------------KERNEL CALL--------------------" << endl;
  }

  //Copy host arrays to device
  cudaMemcpy(device_arrays, host_arrays, array_set_bytes, cudaMemcpyHostToDevice);

  //Create new nvml file
  nvml_filename.append(base_nvml_filename);
  nvml_filename.append("_shfl_bs.csv");
  type.append("shfl_bs");
  nvml.new_experiment(nvml_filename, type);

  cpu_threads.emplace_back(thread(&nvmlClass::getStats, &nvml));

  //Timing
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  //Kernel call
  shfl_bs_match<<<num_blocks, num_threads, share_size>>>(device_arrays, num_threads);

  //Timing
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  // NVML
  // Create thread to kill GPU stats
  // Join both threads to main
  cpu_threads.emplace_back(thread( &nvmlClass::killThread, &nvml));

  for (auto& th : cpu_threads) {
    th.join();
    th.~thread();
  }

  cpu_threads.clear();
  nvml_filename.clear();
  type.clear();

  cout << "Shfl Sort Search" << "," << num_threads << "," << array_size << "," << milliseconds << endl;

  //Copy device arrays back to host
  cudaMemcpy(experiment_arrays, device_arrays, array_set_bytes, cudaMemcpyDeviceToHost);

  if (DEBUG) {
    //Print arrays after matching
    for(int i = 0; i < 1; i++) {

      cout << "Arrays " << i << ": [";

      for(int j = 0; j < array_size * 2; j++) {
        cout << experiment_arrays[(i * array_size * 2) + j] << " ";

        if (j == array_size - 1) { cout << "]\t["; }
      }

      cout << "]" << endl;
    }
  }

  /************************CPU Verification***************************************/
  if (DEBUG) {
    cout << endl << "***CPU Verification***" << endl;

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
  free(experiment_arrays);

	return 0;
}

/* Copyright (c) 1993-2015, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#include "nvmlClass.h"

#include <stdio.h>
#include <assert.h>

// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
inline
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
  return result;
}

template <typename T>
__global__ void offset(T* a, int s)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x + s;
  a[i] = a[i] + 1;
}

template <typename T>
__global__ void stride(T* a, int s)
{
  int i = (blockDim.x * blockIdx.x + threadIdx.x) * s;
  a[i] = a[i] + 1;
}

template <typename T>
void runTest(int deviceId, int nMB)
{
  //int blockSize = 256;
  //float ms;

  T *d_a;
  //cudaEvent_t startEvent, stopEvent;


  //Default, nMB = 4
  //Sp 4 * 1024 * 1024 / sizeof(double) = 4 * 1024 * 1024 / 8 = 524288
  int n = nMB*1024*1024/sizeof(T);

  // NB:  d_a(33*nMB) for stride case
  checkCuda( cudaMalloc(&d_a, n * 33 * sizeof(T)) );

  /*checkCuda( cudaEventCreate(&startEvent) );
  checkCuda( cudaEventCreate(&stopEvent) );*/

  /*printf("Offset, Bandwidth (GB/s):\n");

  offset<<<n/blockSize, blockSize>>>(d_a, 0); // warm up

  for (int i = 0; i <= 32; i++) {
    checkCuda( cudaMemset(d_a, 0, n * sizeof(T)) );

    checkCuda( cudaEventRecord(startEvent,0) );
    offset<<<n/blockSize, blockSize>>>(d_a, i);
    checkCuda( cudaEventRecord(stopEvent,0) );
    checkCuda( cudaEventSynchronize(stopEvent) );

    checkCuda( cudaEventElapsedTime(&ms, startEvent, stopEvent) );
    printf("%d, %f\n", i, 2*nMB/ms);
  }

  printf("\n");
  printf("Stride, Bandwidth (GB/s):\n");*/

  /************************NVML get device********************************/
  int nvml_dev {};
  cudaError_t cuda_err;
  cudaGetDevice( &nvml_dev );
  cuda_err = cudaSetDevice( nvml_dev );

  /*************************CUDA Timing***********************************/
  cudaEvent_t start, stop;
  float milliseconds;
  int iterations = 10000;

  // Original
  // n/blockSize = 4096 blocks
  // blockSize = 256 threads

  //Same ratio
  // blocks 2048, threads 512
  // blocks 1024, threads 1024

  // Change Blocks
  // blocks 2048 threads 256
  // blocks 1024 threads 256
  // blocks 512 threads 256

  // Change Threads
  // blocks 4096 threads 512
  // block 4096 threads 128
  // block 4096 threads 64

  if (cuda_err != cudaSuccess) {
    std::cerr << "cudaSetDevice failed for nvml\n" << std::endl;
  }

  std::string nvml_filename = "./coalescing_default.csv";
  std::vector<std::thread> cpu_threads;
  std::string type;

  type.append("coalescing_memory");
  nvmlClass nvml( nvml_dev, nvml_filename, type);

  cpu_threads.emplace_back(std::thread(&nvmlClass::getStats, &nvml));

  nvml.log_start();

  //Timing
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  for (int i = 0; i < iterations; i++) {
    //stride<<<n/blockSize, blockSize>>>(d_a, 1); // warm up
    stride<<<4096, 256>>>(d_a, 1);
  }

  //Timing
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  nvml.log_stop();

  // NVML
  // Create thread to kill GPU stats
  // Join both threads to main
  cpu_threads.emplace_back(std::thread( &nvmlClass::killThread, &nvml));

  for (auto& th : cpu_threads) {
    th.join();
    th.~thread();
  }

  cpu_threads.clear();
  nvml_filename.clear();
  type.clear();

  std::cout << "Kernel elapsed time: " << milliseconds << " (ms)" << std::endl << std::endl;

  /*for (int i = 1; i <= 32; i++) {
    checkCuda( cudaMemset(d_a, 0, n * sizeof(T)) );

    checkCuda( cudaEventRecord(startEvent,0) );
    stride<<<n/blockSize, blockSize>>>(d_a, i);
    checkCuda( cudaEventRecord(stopEvent,0) );
    checkCuda( cudaEventSynchronize(stopEvent) );

    checkCuda( cudaEventElapsedTime(&ms, startEvent, stopEvent) );
    printf("%d, %f\n", i, 2*nMB/ms);
  }

  checkCuda( cudaEventDestroy(startEvent) );
  checkCuda( cudaEventDestroy(stopEvent) );*/
  cudaFree(d_a);
}

int main(int argc, char **argv)
{
  int nMB = 4;
  int deviceId = 0;
  bool bFp64 = false;

  for (int i = 1; i < argc; i++) {
    if (!strncmp(argv[i], "dev=", 4))
      deviceId = atoi((char*)(&argv[i][4]));
    else if (!strcmp(argv[i], "fp64"))
      bFp64 = true;
  }

  cudaDeviceProp prop;

  checkCuda( cudaSetDevice(deviceId) )
  ;
  checkCuda( cudaGetDeviceProperties(&prop, deviceId) );
  printf("Device: %s\n", prop.name);
  printf("Transfer size (MB): %d\n", nMB);

  printf("%s Precision\n", bFp64 ? "Double" : "Single");

  if (bFp64) runTest<double>(deviceId, nMB);
  else       runTest<float>(deviceId, nMB);
}

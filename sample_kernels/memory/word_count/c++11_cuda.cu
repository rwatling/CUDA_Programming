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

#include <thrust/device_ptr.h>
#include <thrust/count.h>
#include <thrust/execution_policy.h>

#include <iostream>

/////////////////////////////////////////////////////////////////
// Some utility code to define grid_stride_range
// Normally this would be in a header but it's here
// for didactic purposes. Uses
#include "range.hpp"
using namespace util::lang;

// type alias to simplify typing...
template<typename T>
using step_range = typename range_proxy<T>::step_range_proxy;

template <typename T>
__device__
step_range<T> grid_stride_range(T begin, T end) {
    begin += blockDim.x * blockIdx.x + threadIdx.x;
    return range(begin, end).step(gridDim.x * blockDim.x);
}
/////////////////////////////////////////////////////////////////

template <typename T, typename Predicate>
__device__
void count_if(int *count, T *data, int n, Predicate p)
{
  for (auto i : grid_stride_range(0, n)) {
    if (p(data[i])) atomicAdd(count, 1);
  }
}

// Use count_if with a lambda function that searches for x, y, z or w
// Note the use of range-based for loop and initializer_list inside the functor
// We use auto so we don't have to know the type of the functor or array
__global__
void xyzw_frequency(int *count, char *text, int n, int workThreads)
{
  int threadId = (blockIdx.x * blockDim.x) + threadIdx.x;
  const char letters[] { 'x','y','z','w' };

  if (threadId <= workThreads) {
    count_if(count, text, n, [&](char c) {
      for (const auto x : letters)
        if (c == x) return true;
      return false;
    });
  }
}

// A bug in CUDA 7.0 causes errors when this is called
// Comment out by default, but will work in CUDA 7.5
#if 0
__global__
void xyzw_frequency_thrust_device(int *count, char *text, int n)
{
  const char letters[] { 'x','y','z','w' };

  *count = thrust::count_if(thrust::device, text, text+n, [=](char c) {
    for (const auto x : letters)
      if (c == x) return true;
    return false;
  });
}
#endif

// a bug in Thrust 1.8 causes warnings when this is uncommented
// so commented out by default -- fixed in Thrust master branch
#if 0
void xyzw_frequency_thrust_host(int *count, char *text, int n)
{

  const char letters[] { 'x','y','z','w' };

  *count = thrust::count_if(thrust::host, text, text+n, [&](char c) {
    for (const auto x : letters)
      if (c == x) return true;
    return false;
  });
}
#endif

int main(int argc, char** argv)
{

  /************************NVML get device********************************/
  int nvml_dev {};
  cudaError_t cuda_err;
  cudaGetDevice( &nvml_dev );
  cuda_err = cudaSetDevice( nvml_dev );

  /*************************CUDA Timing***********************************/
  cudaEvent_t start, stop;
  float milliseconds;
  int iterations = 15500;
  int numThreads = 256;
  int numIdle = 512;
  int numBlocks = 8;

  if (cuda_err != cudaSuccess) {
    std::cerr << "cudaSetDevice failed for nvml\n" << std::endl;
  }

  std::string nvml_filename = "./wordcount_idle512.csv";
  std::vector<std::thread> cpu_threads;
  std::string type;

  type.append("idle512_wordcount_memory");
  nvmlClass nvml( nvml_dev, nvml_filename, type);

  cpu_threads.emplace_back(std::thread(&nvmlClass::getStats, &nvml));

  nvml.log_start();

  const char *filename = "warandpeace.txt";

  int numBytes = 16*1048576;
  char *h_text = (char*)malloc(numBytes);

  char *d_text;
  cudaMalloc((void**)&d_text, numBytes);

  FILE *fp = fopen(filename, "r");
  int len = fread(h_text, sizeof(char), numBytes, fp);
  fclose(fp);
  std::cout << "Read " << len << " byte corpus from " << filename << std::endl;

  cudaMemcpy(d_text, h_text, len, cudaMemcpyHostToDevice);

  int count = 0;
  int *d_count;
  cudaMalloc(&d_count, sizeof(int));
  cudaMemset(d_count, 0, sizeof(int));

  // threads and blocks configurations
  // Original: 8, 256

  // Keep ratio the same
  // Test1: 16, 128
  // Test2: 4, 512
  // Test3: 32, 64

  // Change blocks
  // Test4: 1, 256
  // Test5: 4, 256
  // Test6: 16, 256
  // Test7: 32, 256

  //Change threads
  // Test8: 8, 512
  // Test9: 8, 128
  // Test10: 8, 1024

  nvml.log_point();

  //Timing
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  for (int i = 0; i < iterations; i++) {
    xyzw_frequency<<<numBlocks, numThreads + numIdle>>>(d_count, d_text, len, numThreads * numBlocks);
  }

  //Timing
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  nvml.log_point();

  std::cout << "Kernel elapsed time: " << milliseconds << " (ms)" << std::endl << std::endl;

  //xyzw_frequency_thrust_device<<<1, 1>>>(d_count, d_text, len);
  cudaMemcpy(&count, d_count, sizeof(int), cudaMemcpyDeviceToHost);

  //xyzw_frequency_thrust_host(&count, h_text, len);

  //std::cout << "counted " << count << " instances of 'x', 'y', 'z', or 'w' in \""
  //<< filename << "\"" << std::endl;

  cudaFree(d_count);
  cudaFree(d_text);

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

  return 0;
}

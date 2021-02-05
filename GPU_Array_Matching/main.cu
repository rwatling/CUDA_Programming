/* This program searches "states" for matches in their arrays using CUDA
*  
* Author: Robbie Watling
*/

# include <cuda.h>
# include <cuda_runtime_api.h>
# include <device_launch_parameters.h>
# include <iostream>
# include <vector>
# include <utility>
# include <iostream>

__global__ void array_search(int* a, int* b, int* match, int n) {
	
	int thread_id = (blockIdx.x * blockDim.x) + threadIdx.x;
	int val;

	if (thread_id < n) {
		
		val = a[thread_id];

		for (int i = 0; i < n; i++) {
			if (b[i] == val) {
				*match = 1;
				break;
			}
		}
	}
}

using namespace std;

int main() {

	/***Variable Declarations***/
	int N;
	int host_match;
	int* dev_match;

	//Host int arrays
	vector<int> host_a;
	vector<int> host_b;
	vector<int> host_c;

	//Device int arrays
	int* dev_a;
	int* dev_b;
	int* dev_c;

	//Memory variables
	size_t arr_bytes;
	size_t match_bytes;
	int NUM_THREADS;
	int NUM_BLOCKS;

	/***Initialize data structures***/
	N = 1 << 5;

	// "Match" int allocation
	host_match = 1;
	match_bytes = (size_t) 1 * sizeof(int);
	cudaMalloc((void**) &dev_match, match_bytes);

	//Vector a
	for (int i = 0; i < N; i++) {
		host_a.push_back(i);
	}

	//Vector b: Will have matches with even elements of a
	for (int i = 0; i < N; i++) {
		host_b.push_back(i);
	}

	//Vector c: Will have no matches with a or b
	for (int i = 0; i < N; i++) {
		host_c.push_back(-1);
	}

	//Int arrays of size N for device
	arr_bytes = N * sizeof(int);
	cudaMalloc((void**) &dev_a, arr_bytes);
	cudaMalloc((void**) &dev_b, arr_bytes);
	cudaMalloc((void**)&dev_c, arr_bytes);

	/*** Copy arrays to device ***/
	cudaMemcpy(dev_a, host_a.data(), arr_bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, host_b.data(), arr_bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_c, host_c.data(), arr_bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_match, &host_match, match_bytes, cudaMemcpyHostToDevice);

	/*** Thread setup ***/
	NUM_THREADS = 1 << 3;
	NUM_BLOCKS = (N + NUM_THREADS - 1) / NUM_THREADS;

	//Search arrays and copy result back to host
	//Memcopy works as a synchronization layer
	array_search <<<NUM_BLOCKS, NUM_THREADS >>> (dev_a, dev_b, dev_match, N);
	cudaMemcpy(&host_match, dev_match, match_bytes, cudaMemcpyDeviceToHost);

	cout << "Search array A and B result: " << host_match << endl;

	//Zero out match
	cudaMemset((void*) dev_match, 0, match_bytes);

	//Search arrays ad copy result back to host
	array_search << <NUM_BLOCKS, NUM_THREADS >> > (dev_b, dev_c, dev_match, N);
	cudaMemcpy(&host_match, dev_match, match_bytes, cudaMemcpyDeviceToHost);

	cout << "Search array B and C result: " << host_match << endl;

	/***Free variables***/
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
	cudaFree(dev_match);

	return 0;
}
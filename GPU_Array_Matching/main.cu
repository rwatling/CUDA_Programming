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

typedef struct {
	int* arr1;
	int* arr2;
	int match;
	int size;
} Array_Comp_Struct;

__global__ void array_match(int* a, int* b, int* match, int n) {
	
	int thread_id = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (thread_id < n) {
		*match = ((int) a[thread_id] == b[thread_id]);
	}
}

using namespace std;

int main() {

	/***Variable Declarations***/

	// Supervising variables
	int N;
	int NUM_THREADS;
	int NUM_BLOCKS;
	size_t byte;
	cudaError cuda_err;

	//Host variables
	//vector<int> host_a;
	//vector<int> host_b;
	//vector<int> host_c;
	//int host_match;
	Array_Comp_Struct* host_data;
	size_t host_struct_size;

	//Device variables
	//int* dev_a;
	//int* dev_b;
	//int* dev_c;
	Array_Comp_Struct* device_data;
	size_t device_struct_size;

	/***Initialization***/
	N = 1 << 5;

	// Struct for host information. 
	// Members: Two arrays of size N, integer N, and integer match
	byte = (size_t) 1;
	host_struct_size = (size_t) (N * 2 * sizeof(int)) + sizeof(int) + sizeof(int);
	host_data = (Array_Comp_Struct*) calloc(byte, host_struct_size);

	if (host_data == NULL) {
		cerr << "HOST DATA IS NULL" << endl;
	}

	// Struct for GPU information. 
	// Members: Two arrays of size N, integer N, and integer match
	device_struct_size = (size_t) (N * 2 * sizeof(int)) + sizeof(int) + sizeof(int);
	cuda_err = cudaMalloc((void**) &device_data, device_struct_size);

	if (cuda_err != cudaSuccess) {
		cerr << "CUDA MALLOC FAILED" << endl;
	}


	//Vector a

	//Vector b: Will have matches with even elements of a

	//Vector c: Will have no matches with a or b


	/*** Copy arrays to device ***/
	//cudaMemcpy(dev_a, host_a.data(), arr_bytes, cudaMemcpyHostToDevice);
	//cudaMemcpy(dev_b, host_b.data(), arr_bytes, cudaMemcpyHostToDevice);
	//cudaMemcpy(dev_c, host_c.data(), arr_bytes, cudaMemcpyHostToDevice);
	//cudaMemcpy(dev_match, &host_match, match_bytes, cudaMemcpyHostToDevice);

	/*** Thread setup ***/
	NUM_THREADS = N;
	NUM_BLOCKS = (N + NUM_THREADS - 1) / NUM_THREADS;

	//Search arrays and copy result back to host
	//Memcopy works as a synchronization layer
	//array_match <<<NUM_BLOCKS, NUM_THREADS >>> (dev_a, dev_b, dev_match, N);
	//cudaMemcpy(&host_match, dev_match, match_bytes, cudaMemcpyDeviceToHost);

	//cout << "Search array A and B result: " << host_match << endl;

	//Zero out match
	//cudaMemset((void*) dev_match, 0, match_bytes);

	//Search arrays ad copy result back to host
	//array_match << <NUM_BLOCKS, NUM_THREADS >> > (dev_b, dev_c, dev_match, N);
	//cudaMemcpy(&host_match, dev_match, match_bytes, cudaMemcpyDeviceToHost);

	//cout << "Search array B and C result: " << host_match << endl;

	/***Free variables***/
	cudaFree(device_data);
	free(host_data);

	return 0;
}
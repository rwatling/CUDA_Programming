/* This program searches "states" for matches in their arrays using CUDA
*  
* Author: Robbie Watling
*/

# include "system_includes.h"

__global__ void array_search(int* a, int* b, int* match) {
	//a[thread_id] = b[thread_id] return match
}


using namespace std;

int main() {

	/***Variable Declarations***/
	//Array size
	int N;
	int num_states;

	//Host int arrays
	vector<int> host_a;
	vector<int> host_b;
	vector<int> host_c;

	//Device int arrays
	int* dev_a;
	int* dev_b;
	int* dev_c;

	//Memory variables
	size_t bytes;

	/***Initialize data structures***/
	num_states = 3;

	//Initialize host arrays
	N = 1 << 4;

	//Vector a
	for (int i = 0; i < N; i++) {
		host_a.push_back(i);
	}

	//Vector b: Will have matches with even elements of a
	for (int i = 0; i < N; i++) {
		if (i % 2 == 0) {
			host_b.push_back(i);
		}
		else {
			host_b.push_back(0);
		}
	}

	//Vector c: Will have no matches with a or b
	for (int i = 0; i < N; i++) {
		host_c.push_back((i + 1) * 2);
	}

	//Int arrays of size N for device
	N = 1 << 4;
	bytes = N * sizeof(int);
	cudaMalloc((void**) &dev_a, bytes);
	cudaMalloc((void**) &dev_b, bytes);
	cudaMalloc((void**) &dev_c, bytes);

	/***Free variables***/
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);

	return 0;
}
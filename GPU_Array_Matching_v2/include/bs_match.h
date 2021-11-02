#ifndef BS_MATCH_H
#define BS_MATCH_H 1

#include "cuda_includes.h"

__device__ void bs_match(int* array2, int* next_arr1, int* next_arr2);
__device__ int partition(int* arr1, int* arr2, int low, int high);
__device__ void quicksort(int* arr1, int* arr2, int low, int high);
__device__ int binary_search(int* arr, int l, int h, int target);

#endif

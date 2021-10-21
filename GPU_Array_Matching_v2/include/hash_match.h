#ifndef HASH_MATCH_H
#define HASH_MATCH_H 1
#define HASH_SIZE 37
#define MAX_COLLISIONS 4
#include "match.h"
#include "cuda_includes.h"
__device__ void hash_match(int* array2, int* next_arr1, int* next_arr2);
__device__ int hash(int key);
#endif

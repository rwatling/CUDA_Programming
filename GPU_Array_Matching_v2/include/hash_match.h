#ifndef HASH_MATCH_H
#define HASH_MATCH_H 1

#include "match.h"
#include "cuda_includes.h"

#if ARRAY_SIZE == 4
  #define HASH_SIZE 5
  #define MAX_COLLISIONS 2
  #define SCALE 3
#elif ARRAY_SIZE == 8
  #define HASH_SIZE 11
  #define MAX_COLLISIONS 2
  #define SCALE 7
#elif ARRAY_SIZE == 12
  #define HASH_SIZE 13
  #define MAX_COLLISIONS 2
  #define SCALE 11
#elif ARRAY_SIZE == 16
  #define HASH_SIZE 19
  #define MAX_COLLISIONS 2
  #define SCALE 13
#elif ARRAY_SIZE == 24
  #define HASH_SIZE 31
  #define MAX_COLLISIONS 2
  #define SCALE 23
#endif

__device__ void hash_match(int* array2, int* next_arr1, int* next_arr2);
__device__ int hash(int key);
#endif

#include "match.h"

__device__ void match(int* array2, int* next_arr1, int* next_arr2) {

  //TODO: Having an issue with no-matches
  #pragma nounroll
  for (int i = 0; i < ARRAY_SIZE; i++) {
    for (int j = 0; j < ARRAY_SIZE; j++) {
      if (array2[i] == next_arr1[j]) {
        array2[i] = next_arr2[j];
        break;
      }
    }
  }
}

#include "unroll_match.h"

__device__ void unroll_match(int* array2, int* next_arr1, int* next_arr2) {

  //TODO: Having an issue with no-matches
  for (int i = 0; i < ARRAY_SIZE; i++) {

    //Note: >> 2 is / 4 and << 2 is * 4
    for (int j = 0; j < ARRAY_SIZE; j += 4) {
      //int hoist_mult = j << 2;

      if (array2[i] == next_arr1[j + 0]) {
        array2[i] = next_arr2[j + 0];
        break;
      } else if (array2[i] == next_arr1[j + 1]) {
        array2[i] = next_arr2[j + 1];
        break;
      } else if (array2[i] == next_arr1[j + 2]) {
        array2[i] = next_arr2[j + 2];
        break;
      } else if (array2[i] == next_arr1[j + 3]) {
        array2[i] = next_arr2[j + 3];
        break;
      }
    }
  }
}

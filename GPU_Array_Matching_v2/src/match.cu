#include "match.h"

__device__ void match(int* array2, int* next_arr1, int* next_arr2) {
  for (int i = 0; i < ARRAY_SIZE; i++) {
    int match = 0;

    for (int j = 0; j < ARRAY_SIZE; j++) {

      if (array2[i] == next_arr1[j]) {
        array2[i] = next_arr2[j];
        match = 1;
        break;
      }
    }

    if ((!match) && (array2[i] > 0)) {
      array2[i] = array2[i] * -1;
    }
  }
}

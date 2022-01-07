#include "cpu_array_match.h"

void cpu_array_match(int* arrays, int num_threads, int array_size) {
  for (int i = 1; i < num_threads; i++) {
    int* next_arr1 = arrays + (i * 2 * array_size);
    int* next_arr2 = arrays + (i * 2 * array_size) + array_size;

    for (int j = 0; j < array_size; j++) {

      for (int k = 0; k < array_size; k++) {
        if (arrays[array_size + j] == next_arr1[k]) {
          arrays[array_size + j] = next_arr2[k];
          break;
        }
      }
    }
  }
}

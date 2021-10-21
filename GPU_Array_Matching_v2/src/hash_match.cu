#include "hash_match.h"

__device__ void hash_match(int* array2, int* next_arr1, int* next_arr2) {
  //Hash table1 for next arr1 value
  //Hash table2 for next arr2 value at table1 index
  int h_table1[HASH_SIZE][MAX_COLLISIONS];
  int h_table2[HASH_SIZE][MAX_COLLISIONS];
  int key = 0;
  int hashed_key = 0;

  //Hash next arrays
  for (int i = 0; i < ARRAY_SIZE; i++) {
    key = next_arr1[i];
    hashed_key = hash(key);

    for (int j = 0; j < MAX_COLLISIONS; j++) {
      if (h_table1[hashed_key][j] == 0) {
        h_table1[hashed_key][j] = next_arr1[i];
        h_table2[hashed_key][j] = next_arr2[i];
        break;
      } else if ((h_table1[hashed_key][j] != 0) && (j == MAX_COLLISIONS -1)) {
        for (int k = 0; k < size; k++) {
          array2[k] = INT_MIN;
          return;
        }
      }
    }
  }

  for (int i = 0; i < ARRAY_SIZE; i++) {
    key = array2[i];
    hashed_key = hash(key);

    for (int j = 0; j < MAX_COLLISIONS; j++) {
      if (h_table1[hashed_key][j] == key) {
        array2[i] = h_table2[hashed_key][j];
        break;
      }
    }
  }
}

__device__ int hash(int key) {
  float c = .618;
  int pow_2 = 16;

  int temp = (int) (key * c);
  temp = temp * pow_2;

  return temp % HASH_SIZE;
}

#include "hash_match.h"

__device__ void hash_match(int* array2, int* next_arr1, int* next_arr2) {

  //Hash table1 for next arr1 value
  //Hash table2 for next arr2 value at table1 index
  int h_table1[HASH_SIZE][MAX_COLLISIONS];
  int h_table2[HASH_SIZE][MAX_COLLISIONS];
  int key = 0;
  int hashed_key = 0;

  //Hash tables are not garunteed to be 0
  for (int i = 0; i < HASH_SIZE; i++) {
    for (int j = 0; j < MAX_COLLISIONS; j++) {
      h_table1[i][j] = 0;
    }
  }

  //Hash "next" arrays
  for (int i = 0; i < ARRAY_SIZE; i++) {
    key = next_arr1[i];
    hashed_key = hash(key);

    if (h_table1[hashed_key][0] == 0) {
      h_table1[hashed_key][0] = next_arr1[i];
      h_table2[hashed_key][0] = next_arr2[i];
    } else if (h_table1[hashed_key][1] == 0) {
      h_table1[hashed_key][1] = next_arr1[i];
      h_table2[hashed_key][1] = next_arr2[i];
    }/* else if (h_table1[hashed_key][2] == 0) {
      h_table1[hashed_key][2] = next_arr1[i];
      h_table2[hashed_key][2] = next_arr2[i];
    } else if (h_table1[hashed_key][3] == 0) {
      h_table1[hashed_key][3] = next_arr1[i];
      h_table2[hashed_key][3] = next_arr2[i];
    }*/
  }

  //Find values
  for (int i = 0; i < ARRAY_SIZE; i++) {
    key = array2[i];
    hashed_key = hash(key);

    //array2[i] = h_table2[hashed_key][0];

    if (key == h_table1[hashed_key][0]) {
      array2[i] = h_table2[hashed_key][0];
    } else if (key == h_table1[hashed_key][1]) {
      array2[i] = h_table2[hashed_key][1];
    }/* else if (key == h_table1[hashed_key][2]) {
      array2[i] = h_table2[hashed_key][2];
    } else if (key == h_table1[hashed_key][3]) {
      array2[i] = h_table2[hashed_key][3];
    }*/
  }
}

//Simpler but effective hashing function
__device__ int hash(int key) {
  return key * (key + 3) % HASH_SIZE;
}

#include "bs_match.h"

__device__ void bs_match(int* array2, int* next_arr1, int* next_arr2) {

  //Quick sort next_arr1 and next_arr2
  //quicksort(next_arr1, next_arr2, 0, ARRAY_SIZE - 1);

  //Binary search on array1
  for (int i = 0; i < ARRAY_SIZE; i++) {
    int rslt = binary_search(next_arr1, 0, ARRAY_SIZE - 1, array2[i]);

    //TO DO: No match
    if (rslt != -1) {
      array2[i] = next_arr2[rslt];
    }

  }
}
__device__ int partition(int* arr1, int* arr2, int low, int high) {
  int pivot = arr1[low];
  int i = low - 1;
  int j = high + 1;

  while (1) {
    do {
        i++;
    } while (arr1[i] < pivot);

    do {
        j--;
    } while (arr1[j] > pivot);

    if (i >= j) {
        return j;
    }

    //Swap arr1[i] and arr1[j]
    int temp = arr1[i];
    arr1[i] = arr1[j];
    arr1[j] = temp;

    //Swap arr2[i] and arr2[j]
    temp = arr2[i];
    arr2[i] = arr2[j];
    arr2[j] = temp;
  }
}

//Implementation: https://www.techiedelight.com/quick-sort-using-hoares-partitioning-scheme/
__device__ void quicksort(int* arr1, int* arr2, int low, int high) {
  // base condition
  if (low >= high) {
      return;
  }

  // rearrange elements across pivot
  int pivot = partition(arr1, arr2, low, high);

  // recur on subarray containing elements that are less than the pivot
  quicksort(arr1, arr2, low, pivot);

  // recur on subarray containing elements that are more than the pivot
  quicksort(arr1, arr2, pivot + 1, high);
}

//Implementation: https://www.geeksforgeeks.org/binary-search/
__device__ int binary_search(int* arr, int l, int r, int target) {
  if (r >= l) {
    int mid = l + (r - l) / 2;

    // If the element is present at the middle
    // itself
    if (arr[mid] == target)
        return mid;

    // If element is smaller than mid, then
    // it can only be present in left subarray
    if (arr[mid] > target)
        return binary_search(arr, l, mid - 1, target);

    // Else the element can only be present
    // in right subarray
    return binary_search(arr, mid + 1, r, target);
  }

  // We reach here when element is not
  // present in array
  return -1;
}

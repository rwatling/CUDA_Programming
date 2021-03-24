# GPU Array Matching

Robbie Watling
Independent Study with Dr. Junqiao Qiu Spring '21

## Description

## Requirements
* NVIDIA Capable GPU
* Unix Enviroment (bash, make)
* CUDA library (nvcc)
* (Optional) R for analysis script

## Build instructions
This project relies on the 'nvcc' compiler and 'make' commands.

To make the executable 'main':
'make'

To clean the directory of object files:
'make clean'

To run a set of performance runs:
'source performance.sh'
which will output a series to the 'data' folder

To run a single run of the program:
'./main arg1 arg2 arg3 arg4'

Where the arugments are specified as follows:
1) Argument 1: The size of each array (up to 1024)
2) Argument 2: The number of arrays of size arg1
3) Argument 3: Memory type. 1 for GPU shared memory or 0 for global memory.
4) Argument 4: Debug option. 0 for no debugging, 1 for simple debug check, 2 or more for verbose check.

## Edit instructions
* 'main.cu': main function that coordinates calls to the GPU
* 'array_match.cu': function that conducts the array matching via global memory

* 'shm_array_match.cu': function that conducts the array matching via shared memory

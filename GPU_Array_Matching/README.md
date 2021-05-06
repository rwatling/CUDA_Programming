# GPU Array Matching

Author: Robbie Watling

Independent Study with Dr. Junqiao Qiu Spring '21

## Description
This program performs an array matching experiment on a NVIDIA capable GPU. A specified number of arrays are allocated at the beginning of the program of a specified size. These arrays are then copied to a NVIDIA device where each array is operated on by a GPU thread. Note that the current implementation uses only one thread block and up to 1024 individual threads. The thread then generates a random number for its array and stores it in either global memory or shared memory on the device. The type of memory is specified as a program argument. This current array is then compared to the array of the thread that is one less than itself. The previous array is stored in global memory everytime. In the comparison, each thread conducts a linear search with the `target` being any of the elements in the current array. If a `target` is found, the search stops and it is denoted that a match has been found by writing a `1` at index of the thread's `id` in an array called `match`. The `match` array and all the global arrays are copied back to the host. Timing information is then printed in the form of a csv. See `data/empty_results.csv` to disambiguate the output. In debug mode, the experiment is performed sequentially by the CPU and compared against GPU run experiment. If there is an error it will print information to the standard error file stream. If in verbose debug mode all of the arrays will be printed for manual comparison.

## Limitations
Currently this only runs on one thread block with 1024 arrays.
The total number of arrays must be less than 1024.
The size of the arrays must be less than 1024.

## Requirements
* NVIDIA Capable GPU
* CMake version >= 3.8
* Unix Enviroment (bash, make)
* CUDA library (nvcc)
* (Optional) R for analysis script

## Build instructions
This project relies on the `nvcc` compiler and `cmake` commands.

* To make the executable `gpu_matching`:
`cmake .' in the base directory
'make'

* To run a set of performance runs:
(Needs to be reconfigured)
`source performance.sh`
which will output performance information to the `data` folder

* To run a single run of the program:
`./main arg1 arg2 arg3 arg4`

* Program arguments are specified as follows:
1) Argument 1: The size of each array (up to 1024)
2) Argument 2: The number of arrays of size arg1
3) Argument 3: Memory type. 1 for GPU shared memory or 0 for global memory.
4) Argument 4: Debug option. 0 for no debugging, 1 for simple debug check, 2 or more for verbose check.

* To run `analysis.R` it is recommended to install and use RStudio to run the script

## Edit instructions
Source files are in the 'src/' folder
* `main.cu`: main function that coordinates calls to the GPU
* `array_match.cu`: function that conducts the array matching via global memory

* `shm_array_match.cu`: function that conducts the array matching via shared memory

* `anaylsis.R`: Analysis R script to create graphs of performance data.

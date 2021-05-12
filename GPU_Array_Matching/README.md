# GPU Array Matching

Author: Robbie Watling

## Description
This program performs an array matching experiment on a NVIDIA capable GPU. A specified number of arrays are allocated at the beginning of the program. These arrays are then copied to the NVIDIA device where each array is operated on by a GPU thread. The thread then generates a random number for its array and stores it in either global memory or shared memory on the device. The type of memory is specified as a program argument. This current array is then compared to the array of the thread that is one less than itself. In the comparison, each thread conducts a linear search with the `target` being any of the elements in the current array. If a `target` is found, the search stops and it is denoted that a match has been found by writing a `1` at index of the thread's `id` in an array called `match`. The `match` array and all the global arrays are copied back to the host. Timing information is then printed in a 'csv' compatable form. See `data/empty_results.csv` to disambiguate the output. In debug mode (1), the experiment is performed sequentially by the CPU and compared against GPU run experiment. If the debug more is verbose (2) debug mode all of the arrays will be printed for manual comparison.

## Limitations
* Currently this only runs on one thread block with 1024 arrays.
* The total number of arrays must be less than 1024.
* The size of the arrays must be less than 1024.

## Requirements
* NVIDIA Capable GPU
* CMake version >= 3.8
* Make
* CUDA library (nvcc)
* (Optional) bash
* (Optional) R for analysis script

## Build instructions
This project relies on the `nvcc` compiler and `cmake` commands.

* Make the executable `gpu_matching`:<br>
`cmake .` in the base directory<br>
`make` <br>

* Run a set of performance runs:<br>
'cd /analysis/'<br>
`source performance.sh` which will output performance information to the `analysis/data` folder

* Run a single run of the program:<br>
`./gpu_matching arg1 arg2 arg3 arg4` in the base directory

* Program arguments are specified as follows:<br>
`arg1`: The size of each array (up to 1024)<br>
`arg2`: The number of arrays of size arg1<br>
`arg3`: Memory type. 1 for GPU shared memory or 0 for global memory.<br>
`arg4`: Debug option. 0 for no debugging, 1 for simple debug check, 2 or more for verbose check.<br>

* To run `analysis.R` it is recommended to install and use RStudio to run the script

## Edit instructions
Source files are in the `src/` folder

* `main.cu`: main function that coordinates calls to the GPU
* `array_match.cu`: function that conducts the array matching via global memory
* `shm_array_match.cu`: function that conducts the array matching via shared memory

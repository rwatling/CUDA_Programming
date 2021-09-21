# GPU Array Matching

Author: Robbie Watling

## Description
This is the second version of the GPU match experiment. It is currently under development.

## Requirements
* NVIDIA Capable GPU
* GPU needs to have a maximum shared memory size greater than or equal to 64 kibibytes
* CMake version >= 3.8
* Make
* CUDA library (nvcc)

## Build instructions
This project relies on the `nvcc` compiler and `cmake` commands.

* Make the executable `gpu_matching`:<br>
`cmake .` in the base directory<br>
`make` <br>

* Run a single run of the program:<br>
`./gpu_matching num_threads` in the base directory

## Edit instructions
Source files are in the `src/` folder

* `main.cu`: main function that coordinates calls to the GPU
* `shm_array_match.cu`: function that conducts the array matching via shared memory

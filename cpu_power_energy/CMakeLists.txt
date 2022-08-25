cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 11)

# Project name
project(hello_papi LANGUAGES CXX)

set(CMAKE_VERBOSE_MAKEFILE OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")

file(GLOB main_file "${CMAKE_SOURCE_DIR}/src/hello_papi.cpp")

# Required PAPI environment variables
execute_process(COMMAND bash -c "export PAPI_DIR=/home/rwatling/CUDA_Programming/papi-install")
execute_process(COMMAND bash -c "export PATH=${PAPI_DIR}/bin:$PATH")
execute_process(COMMAND bash -c "export LD_LIBRARY_PATH=${PAPI_DIR}/lib:$LD_LIBRARY_PATH")

# Include directories
include_directories(${CMAKE_PROJECT_NAME} PUBLIC /$ENV{PAPI_DIR}/include)
include_directories(${CMAKE_PROJECT_NAME} PUBLIC /$ENV{PAPI_DIR}/share/papi/testlib)

# Link Directories
link_directories(/$ENV{PAPI_DIR}/lib)

# Link libraries
link_libraries(papi)

# Create executable
add_executable(${PROJECT_NAME} ${main_file})
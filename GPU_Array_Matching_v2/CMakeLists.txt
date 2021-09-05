cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(gpu_match LANGUAGES CXX CUDA)
set(GPU_LIB "_gpu_lib")
set(CMAKE_VERBOSE_MAKEFILE OFF)

file(GLOB gpu_source_files "${CMAKE_SOURCE_DIR}/src/shm_array_match.cu")
file(GLOB main_file "${CMAKE_SOURCE_DIR}/src/main.cu")

include_directories(${CMAKE_PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${GPU_LIB} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_library(${GPU_LIB} STATIC ${gpu_source_files})

# Request that ${GPU_LIB} be built with -std=c++11
# As this is a public compile feature anything that links to
# will also build with -std=c++11
target_compile_features(${GPU_LIB} PUBLIC cxx_std_11)

# We need to explicitly state that we need all CUDA files in ${GPU_LIB}
# to be built with -dc
set_target_properties(${GPU_LIB} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(${PROJECT_NAME} ${main_file})

set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(${PROJECT_NAME} PRIVATE ${GPU_LIB})

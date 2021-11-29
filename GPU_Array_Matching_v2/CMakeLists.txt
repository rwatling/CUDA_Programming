cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

# Project name
project(gpu_match LANGUAGES CXX CUDA)

# Default definitions for user defined variables
if(NOT DEFINED ARRAY_SIZE)
    message("\n*** AUTOMATICALLY SETTING ARRAY SIZE TO 4 ***")
    set(ARRAY_SIZE 4)
endif()

if(NOT DEFINED DEBUG)
    message("\n*** AUTOMATICALLY SETTING DEBUG TO 0 ***")
    set(DEBUG 0)
endif()

if(NOT DEFINED LG_REL_HASH)
    message("\n*** AUTOMATICALLY SETTING LG_REL_HASH TO 0 ***")
    set(LG_REL_HASH 0)
endif()

if(NOT DEFINED MEM_OPT)
  message("\n*** AUTOMATICALLY SETTING MEM_OPT TO 0 ***")
  set(MEM_OPT 0)
endif()

if ((NOT ARRAY_SIZE EQUAL 4)
AND (NOT ARRAY_SIZE EQUAL 8)
AND (NOT ARRAY_SIZE EQUAL 12)
AND (NOT ARRAY_SIZE EQUAL 16)
AND (NOT ARRAY_SIZE EQUAL 24))
    message(FATAL_ERROR "ARRAY_SIZE not valid (4, 8, 12, 16, or 24)")
endif()

# Add compile definitions for specified directory
add_compile_definitions(ARRAY_SIZE=${ARRAY_SIZE})
add_compile_definitions(DEBUG=${DEBUG})
add_compile_definitions(LG_REL_HASH=${LG_REL_HASH})

# CMake variable definitions
set(GPU_LIB "_gpu_lib")
set(CPU_LIB "_cpu_lib")
set(CMAKE_VERBOSE_MAKEFILE OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")

file(GLOB gpu_source_files
  "${CMAKE_SOURCE_DIR}/src/shm_array_match.cu"
  "${CMAKE_SOURCE_DIR}/src/match.cu"
  "${CMAKE_SOURCE_DIR}/src/shfl_array_match.cu"
  "${CMAKE_SOURCE_DIR}/src/shfl_hash_match.cu"
  "${CMAKE_SOURCE_DIR}/src/shm_hash_match.cu"
  "${CMAKE_SOURCE_DIR}/src/hash_match.cu"
  "${CMAKE_SOURCE_DIR}/src/unroll_match.cu"
  "${CMAKE_SOURCE_DIR}/src/shfl_unroll_match.cu"
  "${CMAKE_SOURCE_DIR}/src/shm_unroll_match.cu"
  "${CMAKE_SOURCE_DIR}/src/shfl_bs_match.cu"
  "${CMAKE_SOURCE_DIR}/src/bs_match.cu"
  "${CMAKE_SOURCE_DIR}/src/shfl_hash_w_shared_match.cu"
  "${CMAKE_SOURCE_DIR}/src/shfl_unroll2_match.cu"
  "${CMAKE_SOURCE_DIR}/src/unroll2_match.cu")


file(GLOB cpp_source_files
  "${CMAKE_SOURCE_DIR}/src/cpu_array_match.cpp")

file(GLOB main_file "${CMAKE_SOURCE_DIR}/src/main.cu")

# Include directories
include_directories(${CMAKE_PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${GPU_LIB} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CPU_LIB} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR/include})

# GPU Library
add_library(${GPU_LIB} STATIC ${gpu_source_files})

# CPU Library
add_library(${CPU_LIB} STATIC ${cpp_source_files})

# Request that ${GPU_LIB} be built with -std=c++11
# As this is a public compile feature anything that links to
# will also build with -std=c++11
target_compile_features(${GPU_LIB} PUBLIC cxx_std_11)

# We need to explicitly state that we need all CUDA files in ${GPU_LIB}
# to be built with -dc
set_target_properties(${GPU_LIB} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

if (NOT MEM_OPT EQUAL 0)
  target_compile_options(${GPU_LIB} PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>)
endif()

# Create executable
add_executable(${PROJECT_NAME} ${main_file})

# Cuda seperable compilation
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Link libraries
target_link_libraries(${PROJECT_NAME} PRIVATE ${GPU_LIB})
target_link_libraries(${PROJECT_NAME} PRIVATE ${CPU_LIB})


# Print user variable definitions
message("\n*** ALL USER DEFINED VARIABLES ***")
message("ARRAY_SIZE=${ARRAY_SIZE}")
message("DEBUG=${DEBUG}")
message("LG_REL_HASH=${LG_REL_HASH}")
message("MEM_OPT=${MEM_OPT}")
message("******************************\n")

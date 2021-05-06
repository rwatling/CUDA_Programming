cmake_minimum_required(VERSION 3.8)
message(STATUS "CMake version: ${CMAKE_VERSION}")

project(gpu_match)

option(USE_CUDA "Use CUDA" ON)

file(GLOB cpu_source_files "${CMAKE_SOURCE_DIR}/src/*.cpp")
file(GLOB gpu_source_files "${CMAKE_SOURCE_DIR}/src/*.cu")

include_directories(${CMAKE_PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

if(USE_CUDA)
    enable_language("CUDA")
    add_executable(${CMAKE_PROJECT_NAME} ${cpu_source_files} ${gpu_source_files})
else(USE_CUDA)
    add_executable(${CMAKE_PROJECT_NAME} ${cpu_source_files})
endif()

message(STATUS "USE_CUDA: ${USE_CUDA}")
if(USE_CUDA)
    add_definitions(-DUSE_CUDA)
endif()

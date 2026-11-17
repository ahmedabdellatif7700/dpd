# The dpd stage in C++ with hand-written CUDA kernels (docs/testbed/jetson/README.md).
# Build conventions from the user's C++ template (CMake + Ninja, warnings, Debug/Release flags,
# binaries in bin/) and its Jetson CUDA template (CMAKE_CUDA_ARCHITECTURES=87 for the Orin).
#
#   make            (cpp/Makefile, on the host)     or     the image's build (docker/Dockerfile)
cmake_minimum_required(VERSION 3.24)

project(arof_dpd LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 87: the Jetson Orin's GPU, native code (no PTX JIT at start-up)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 87)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)

set(WARNINGS -Wall -Wextra -Wpedantic)
add_compile_options(
    "$<$<COMPILE_LANGUAGE:CXX>:${WARNINGS}>"
    "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>"
    "$<$<CONFIG:Debug>:-g>"
    "$<$<CONFIG:Debug>:-O0>"
    "$<$<CONFIG:Release>:-O3>"
)

# the image's self-test: the GPU from inside the container
add_executable(gpu_check src/gpu_check.cu)
target_include_directories(gpu_check PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

message(STATUS "")
message(STATUS "arof_dpd configuration")
message(STATUS "CMake            : ${CMAKE_VERSION}")
message(STATUS "Generator        : ${CMAKE_GENERATOR}")
message(STATUS "System           : ${CMAKE_SYSTEM_NAME} ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "Build type       : ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ compiler     : ${CMAKE_CXX_COMPILER} (C++${CMAKE_CXX_STANDARD})")
message(STATUS "CUDA compiler    : ${CMAKE_CUDA_COMPILER} ${CMAKE_CUDA_COMPILER_VERSION}")
message(STATUS "CUDA archs       : ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Binaries         : ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
message(STATUS "")

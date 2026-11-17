# The C++/CUDA twin of the reference DPD, and the dpd stage (docs/testbed/jetson/README.md).
# Build conventions from the user's C++ template (CMake + Ninja, warnings, Debug/Release flags,
# binaries in bin/) and its Jetson CUDA template (CMAKE_CUDA_ARCHITECTURES=87 for the Orin).
#
#   make            (cpp/Makefile, on the host)     or     the image's build (docker/Dockerfile)
#   make test       the checks against the reference (ctest)
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

# the model: the DPDT reader, the weights, the plain C++ forward pass (the twin's own reference)
add_library(dpd_core STATIC
    src/tensor_file.cpp
    src/weights.cpp
    src/cpu_model.cpp
)
target_include_directories(dpd_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# the C++ model against the PyTorch reference's golden files
add_executable(golden_check tools/golden_check.cpp)
target_link_libraries(golden_check PRIVATE dpd_core)

# the image's self-test: the GPU from inside the container
add_executable(gpu_check tools/gpu_check.cu)

# the checks, with the models exported by the reference (results/models/, scripts/export_model.py)
enable_testing()
set(MODELS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../results/models" CACHE PATH "the exported models")
foreach(MODEL transformer_nominal transformer_nominal_memory)
    add_test(NAME golden_${MODEL}
        COMMAND golden_check ${MODELS_DIR}/${MODEL}.dpdt ${MODELS_DIR}/golden/${MODEL}.dpdt)
endforeach()

message(STATUS "")
message(STATUS "arof_dpd configuration")
message(STATUS "CMake            : ${CMAKE_VERSION}")
message(STATUS "Generator        : ${CMAKE_GENERATOR}")
message(STATUS "System           : ${CMAKE_SYSTEM_NAME} ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "Build type       : ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ compiler     : ${CMAKE_CXX_COMPILER} (C++${CMAKE_CXX_STANDARD})")
message(STATUS "CUDA compiler    : ${CMAKE_CUDA_COMPILER} ${CMAKE_CUDA_COMPILER_VERSION}")
message(STATUS "CUDA archs       : ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Models           : ${MODELS_DIR}")
message(STATUS "Binaries         : ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
message(STATUS "")

cmake_minimum_required(VERSION 3.20)

# Host compiler: when the default g++ is newer than nvcc supports, prefer
# g++-15; otherwise fall back to the override flag with -std=c++20 (newer
# libstdc++ headers do not parse under nvcc's c++17 mode). Runs before
# project(), where CMAKE_CXX_COMPILER* are not yet known.
# (Block copied from gpu-bundle-adjustment/CMakeLists.txt.)
find_program(_cxx_compiler NAMES $ENV{CXX} c++ g++)
if(_cxx_compiler)
    execute_process(COMMAND ${_cxx_compiler} -dumpversion
                    OUTPUT_VARIABLE _gcc_version OUTPUT_STRIP_TRAILING_WHITESPACE
                    ERROR_QUIET)
endif()
if(_gcc_version VERSION_GREATER_EQUAL 16)
    find_program(_gcc15 NAMES g++-15)
    if(_gcc15)
        set(CMAKE_CUDA_HOST_COMPILER "${_gcc15}")
    else()
        set(CMAKE_CUDA_FLAGS_INIT "-allow-unsupported-compiler -std=c++20")
    endif()
endif()

project(nanosheet-sp LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
if(CMAKE_CUDA_HOST_COMPILER)
    set(CMAKE_CUDA_STANDARD 17)
else()
    set(CMAKE_CUDA_STANDARD 20)
endif()
set(CMAKE_CUDA_ARCHITECTURES native)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(CUDAToolkit REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(sp_reference src/sp_reference.cpp)
add_executable(sp_batched src/sp_batched.cu)
add_executable(test_solver tests/test_solver.cpp)

enable_testing()
add_test(NAME solver COMMAND test_solver)

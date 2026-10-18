#pragma once

// hip_shim.h — HIP-API portability shim for nanosheet-sp.
//
// Validation used CUDA sm_120 (RTX 5060 Laptop, CUDA 13.2). The AMD/Instinct
// path is compile-checked with ROCm hipcc; AMD runtime validation is not recorded.
//
// Under a real HIP/ROCm compile (__HIP_PLATFORM_AMD__ or __HIP__) this just
// pulls in <hip/hip_runtime.h>. Otherwise it maps the hip* runtime API names
// used by this project onto their cuda* equivalents. Kernel launches use
// plain <<<>>> syntax, which both nvcc and hipcc accept.

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP__)

#include <hip/hip_runtime.h>

#else

#include <cuda_runtime.h>

using hipError_t = cudaError_t;
#define hipSuccess cudaSuccess

#define hipMalloc cudaMalloc
#define hipFree cudaFree
#define hipMemcpy cudaMemcpy
#define hipMemset cudaMemset
#define hipMemcpyHostToDevice cudaMemcpyHostToDevice
#define hipMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define hipDeviceSynchronize cudaDeviceSynchronize
#define hipGetLastError cudaGetLastError
#define hipGetErrorString cudaGetErrorString

#endif

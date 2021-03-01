#include "monolithic.h"

// simple tensor core gemm implementation
__global__ void wmma_gpu_gemm() {

}


// GPU GEMM kernel that doesn't use tensor cores
// (use elliot's or cublas implementation)
__global__ void no_wmma_gpu_gemm() {

}

#include "cuda.h"
#include "cuda_runtime.h"
#include "mma.h"
#include "cuda_pipeline.h"
#include <stdio.h>
#include <assert.h>
using namespace nvcuda;

// gemm prototypes
__global__ void no_wmma_gpu_gemm();
__global__ void wmma_gpu_gemm();

void cpu_ref_sgemm(
    float *A, // GEMM A. m x k
    float *B, // GEMM B. k x n
    float *C, // GEMM C. m x n
    float *D, // GEMM D. m x n
    int m,
    int n,
    int k,
    float alpha,
    float beta
);

// system util prototypes
void init();
void cu_error_check(CUresult error);

// matrix util prototypes
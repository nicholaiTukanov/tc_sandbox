#include "monolithic.h"

void test_kernel()
{
    int m = 80*32,
        n = 32,
        k = 16*512; // 512 iterations

    float *C_host, *C_dev;

    C_host = (float *) malloc (sizeof(float) * m * n);

    cudaMalloc((void **)&C_dev, sizeof(float) * m * n);

    cudaMemcpy(C_dev, C_host, sizeof(float) * m * n, cudaMemcpyHostToDevice);


    // 4 warps per thread block
    // 80 blocks
    

    gpu_tc_gemm_ <<< 80,4*32 >>> (
    NULL,
    NULL,
    C_dev,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    m,
    n,
    k
    );

    cudaDeviceSynchronize();

    cudaMemcpy(C_host, C_dev, sizeof(float) * m * n, cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    print_matrix(C_host, "C_host", 64, n, n, 1);

}
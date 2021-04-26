#include "monolithic.h"

void test_packing()
{
    int m = 16,
        k = 32;

    half *A_host, *A_pack_host,
         *A_dev, *A_pack_dev;

    A_host = ( half *) malloc (sizeof( half) * m * k);
    A_pack_host = ( half *) malloc (sizeof( half) * m * k);
    init_matrix<half>(A_host, m, k, k, 1);

    cudaMalloc((void **)&A_dev, sizeof( half) * m * k);
    cudaMalloc((void **)&A_pack_dev, sizeof( half) * m * k);

    cudaMemcpy(A_dev, A_host, sizeof( half) * m * k, cudaMemcpyHostToDevice);

    gpu_pack_test <<<1,32,m*k>>> (A_dev, m, k, A_pack_dev);

    cudaMemcpy(A_pack_host, A_pack_dev, sizeof(half) * m * k, cudaMemcpyDeviceToHost);

    print_matrix(A_host, "A_host", m, k, k, 1);
    print_matrix(A_pack_host, "A_pack_host", k, m, m, 1);
}
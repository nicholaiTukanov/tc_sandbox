// #include "monolithic.h"

void cpu_ref_sgemm
(
    float *A, // GEMM A. m x k
    float *B, // GEMM B. k x n
    float *C, // GEMM C. m x n
    float *D, // GEMM D. m x n
    int m,
    int n,
    int k,
    float alpha,
    float beta 
) 
{
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float d = 0.0f;
            for (int p = 0; p < k; p++) {
                // d += A[i*k + p] * B[p*n + j];
                d += A[p*m + i] * B[j*k + p];
            }
            D[i*n + j] = alpha * d + beta * C[i*n + j];;
        }
    }
}
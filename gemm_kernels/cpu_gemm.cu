#include "monolithic.h"


// assumes row major order for A, B, C
template <typename mat_type_in, typename mat_type_out>
void cpu_ref_sgemm
(
    mat_type_in *A, // A is m x k
    mat_type_in *B, // B is k x n
    mat_type_out *C, // C is m x n
    int m, int n, int k,
    int rs_a, int cs_a,
    int rs_b, int cs_b,
    int rs_c, int cs_c,
    mat_type_out alpha,
    mat_type_out beta 
) 
{
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float d = 0.0f;
            
            for (int p = 0; p < k; p++)
                d += (mat_type_out) A[i*rs_a + p*cs_a] * (mat_type_out) B[p*rs_b + j*cs_b];

            C[i*rs_c + j*cs_c] = alpha * d + beta * C[i*rs_c + j*cs_c];;
        }
    }
}

template void cpu_ref_sgemm<half, float>
(
    half *A, // A is m x k
    half *B, // B is k x n
    float *C, // C is m x n
    int m,
    int n,
    int k,
    int rs_a, int cs_a,
    int rs_b, int cs_b,
    int rs_c, int cs_c,
    float alpha,
    float beta 
);
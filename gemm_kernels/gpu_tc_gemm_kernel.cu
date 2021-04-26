#include "monolithic.h"

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

/* 
    function details: pack a matrix into shared memory

    facts:
        * 1 block will get 48 KB of shared memory
        * total memory taken by matrix = m*n*S_data
        
    assume:
        * incoming matrix is in row major order
        * we want to pack into m x k_mma panels
        
    cases to consider: 
        * matrix size > shared memory
        * how to tell threads to pack?
    
*/
// __device__ void pack_half_matrix_a(
//                half *A,
//                half *A_sh,
//                int ld_a,
//                int k,
//                half *A_pack
// )
// {

//     /*
//         number of threads = warps * 32
//         warps = (m*n) / (WMMA_M * WMMA_N)

//         threads are configured in the following way (all elems enclosed in brackets are in a warp)

//                         warp_x 0               warp_x 1                   warp_x x
//         warp_y 0  [t(0,0) ... t(0,31)] [t(0,32) ... t(0,63)] ... [t(0,32*x) ... t(0,32*x+31)]

//         warp_y 1  [t(1,0) ... t(1,31)] [t(1,32) ... t(1,63)] ... [t(1,32*x) ... t(1,32*x+31)]
//                                                 .
//                                                 .
//                                                 .
//         warp_y y  [t(y,0) ... t(y,31)] [t(y,32) ... t(y,63)] ... [t(y,32*x) ... t(y,32*x+31)]

//         if 1 warp = 1 block, then a warp needs to pack the entire shared memory allocated to a block
//     */
//     int i = (blockIdx.x * blockDim.x + threadIdx.x),
//         j = (blockIdx.y * blockDim.y + threadIdx.y);


//     // 1 warp will pack 32 elements (2 cols of A)

//     // lets assume we have access to only 1 thread
//     if(i < 32 && j == 0)
//     {
//         int k_idx = 0;
//         for(int p=0; p < (k / 32); p+=32)
//         {
//             // col 1
//             if (i < 16)
//             {
//                 A_sh [p + i] = A_pack [p + i] = A [k_idx + i*ld_a];
                
//             }
                
//             // col 2
//             else
//                 A_sh [p + i] = A_pack [p + i] = A [(k_idx+1) + (i-16)*ld_a];
//             k_idx += 2;
//         }
//     }
//     __syncthreads();
// }


/*
    function details: computes mxnxk shgemm using wmma

    facts:
        * 1 warp will compute a 16x16x16 mat-mul
        * each warp holds its own fragements
        * each thread in a warp must get the same starting address for the matrix
        
    assume: 
        * matrices are multiples of WMMA_M x WMMA_N x WMMA_K
        * matrices are not transposed
    
    consider:
        * packing into shared (how to choose tile size?)
        * number of threads to use

*/
__global__ void  gpu_tc_gemm(
    half *A,
    half *B,
    float *C,
    float alpha,
    float beta,
    int ld_a,
    int ld_b,
    int ld_c,
    int m,
    int n,
    int k
)
{

    // Declare the fragments
    // each warp will have its own fragments

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a,    WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::row_major> a_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b,    WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::row_major> b_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag; 


    // __shared__ half A_sh[WMMA_M];
    // pack_half_matrix_a(A, A_sh, ld_a, k, NULL);

    // pack ideas:
    // pack a row/col major matrix into each warp
    // in other words, each warp will have access to a 16x16 matrix of A or B
    // reason: the data needed to do 1 WMMA will be right next to each other

    // assumes that each block will take up 1 warp
    int i = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize,
        j = (blockIdx.y * blockDim.y + threadIdx.y);

    // Initialize the output to zero
    nvcuda::wmma::fill_fragment(acc_frag, 0.0f);

    int row_a = i * WMMA_M,
        col_b = j * WMMA_N;

    for (int i = 0; i < k; i += WMMA_K)
    {
        int col_a = i,
            row_b = i;

        if (row_a < m && col_a < k && 
            row_b < k && col_b < n) 
        {
            // load a 16x16 matrix into the matrix fragements
            nvcuda::wmma::load_matrix_sync(a_frag, A + col_a + row_a * ld_a, WMMA_M);
            nvcuda::wmma::load_matrix_sync(b_frag, B + col_b + row_b * ld_b, ld_b);

            nvcuda::wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
        
    }

    int row_c = i * WMMA_M;
    int col_c = j * WMMA_N;
    
    // bounds checking
    if (row_c < m && col_c < n) {

        // load in C into the accumulator fragment
        nvcuda::wmma::load_matrix_sync(c_frag, C + col_c + row_c * ld_c, ld_c, nvcuda::wmma::mem_row_major);

        // scale result and C and sum them together
        for (int i = 0; i < c_frag.num_elements; i++) {
            c_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
        }

        // store result
        nvcuda::wmma::store_matrix_sync(C + col_c + row_c * ld_c, c_frag, ld_c, nvcuda::wmma::mem_row_major);
    }

}



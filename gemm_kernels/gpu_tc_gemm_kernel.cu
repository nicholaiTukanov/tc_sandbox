#include "monolithic.h"


#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16


/*

    general question:

    can i use the same packing for SIMD CPU GEMM kernels for SIMT GPU GEMM kernels?



80 Volta MPs
2048 threads per MP
64 warps per MP
96 KB shared memory per MP
48 KB shared memory per thread blocks

L2 Cache size = 4608 KB


Number of times a TC will be used in a wmma instruction is WMMA_
- number comes from doing size of 

equation to determine number of bytes per wmma

    A takes (mr)(k_wmma)(S_data_in)
    B takes (nr)(k_wmma)(S_data_in)
    C takes (mr)(nr)(S_data_out)

total = (mr)(k_wmma)(S_data_in) + (nr)(k_wmma)(S_data_in) + (mr)(nr)(S_data_out)
      = (k_wmma) (S_data_in) (mr+nr) + (mr)(nr)(S_data_out)

thus in shgemm we get

    A and B both take (WMMA_)(WMMA_)(2) = 512
    C takes (WMMA_)(WMMA_)(4) = 1024

    total = 2*512 + 1024 = 2048 bytes per wmma in shgemm


my approach:

each thread in a warp will get the same address of the result matrix

therefore we need equations to calculate the address

assumptions:

split shared memory by half
1 warp = 1 block
32 threads = 1 warp

48 KB per


*/


/* 
    function details: pack a matrix into shared memory
    facts:
        * 1 block will get 48 KB of shared memory
        * total memory taken by matrix = m*n*S_data
    cases to consider: 
        * matrix size > shared memory
        * how to tell threads to pack?
    assume:
        * incoming matrix is in row major order
        * we want to pack into m x k_mma panels
*/
__device__ void pack_half_matrix(
               half *matrix,
    __shared__ half *sh_matrix
)
{
    
}


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

    // __shared__ half *A_sh[6144];
    // __shared__ half *B_sh[18432];

    // pack ideas:
    // pack a row/col major matrix into each warp
    // in other words, each warp will have access to a 16x16 matrix of A or B
    // reason: the data needed to do 1 WMMA will be right next to each other

    // assumes that each block will take up 1 warp
    int i = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize,
    j = (blockIdx.y * blockDim.y + threadIdx.y);


    // printf("thread(%d, %d)\n", i, j);

    // pack m_mma x k_mma tiles of A and k_mma x n_mma tiles of B into shared memory
    // store the tiles in row major order (order of the block matrix)
    // pack more tiles of into shared since we can get more reuse out of it
    // after shared memory buffers have been created
    // run through shared memory buffers and compute into accumulator


    

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
            nvcuda::wmma::load_matrix_sync(a_frag, A + col_a + row_a * ld_a, ld_a);
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



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

    C = C + blockIdx.x * 32 * 32;
    int block_num = threadIdx.x / 32;
    int row = block_num / 2, 
        col = block_num % 2;

    // Declare the fragments
    // each warp will have its own fragments
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a,    WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::col_major> a_frag_0;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a,    WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::row_major> a_frag_1;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b,    WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::row_major> b_frag_0;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b,    WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::row_major> b_frag_1;

    // nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag_00; 
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag_01;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag_10;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag_11;

    // Initialize the output to zero
    nvcuda::wmma::fill_fragment(acc_frag_00, 0.0f);
    nvcuda::wmma::fill_fragment(acc_frag_01, 0.0f);
    nvcuda::wmma::fill_fragment(acc_frag_10, 0.0f);
    nvcuda::wmma::fill_fragment(acc_frag_11, 0.0f);


    __shared__ half shared_memory[2*32*32];
    half *A_sh = &shared_memory[0];
    half *B_sh = &shared_memory[32*32];

    // pack_half_matrix_a(A, A_sh, ld_a, k, NULL);

    // pack ideas:
    // pack a row/col major matrix into each warp
    // in other words, each warp will have access to a 16x16 matrix of A or B
    // reason: the data needed to do 1 WMMA will be right next to each other

    // assumes that each block will take up 1 warp

    // int i = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize, 
    //     j = (blockIdx.y * blockDim.y + threadIdx.y);

    // int i_warp = i / warpSize;


    for (int ii=0; ii<32; ii++)
    {
        for (int jj=0; jj<32; jj++)
        {
            A_sh[ii*16 + jj] = (half)(1.0*ii);
            B_sh[ii*32 + jj] = (half)(1.0*ii);
        }
    }

    __syncthreads();

    nvcuda::wmma::load_matrix_sync(a_frag_0, A_sh, 16);
    nvcuda::wmma::load_matrix_sync(a_frag_1, A_sh + 16*16, 16);

    nvcuda::wmma::load_matrix_sync(b_frag_0, B_sh, 16);
    nvcuda::wmma::load_matrix_sync(b_frag_1, B_sh + 16, 32);

    // 16x16x16 shgemm
    nvcuda::wmma::mma_sync(acc_frag_00, a_frag_0, b_frag_0, acc_frag_00);
    // nvcuda::wmma::mma_sync(acc_frag_01, a_frag_0, b_frag_1, acc_frag_01);
    // nvcuda::wmma::mma_sync(acc_frag_10, a_frag_1, b_frag_0, acc_frag_10);
    // nvcuda::wmma::mma_sync(acc_frag_11, a_frag_1, b_frag_1, acc_frag_11);

    // store to c
    nvcuda::wmma::store_matrix_sync(C + row*16 + col*16*32, acc_frag_00, 32, nvcuda::wmma::mem_row_major);
    nvcuda::wmma::store_matrix_sync(C + row*16 + col*0*32, acc_frag_01, 32, nvcuda::wmma::mem_row_major);
    nvcuda::wmma::store_matrix_sync(C + row* 0 + col*16*32, acc_frag_10, 32, nvcuda::wmma::mem_row_major);
    nvcuda::wmma::store_matrix_sync(C + row*16 + col*16*32, acc_frag_11, 32, nvcuda::wmma::mem_row_major);

    __syncthreads();



















    
    // if(i == 0  && j == 0){
    //     for (int ii=0; ii<16; ii++)
    //     {
    //         for (int jj=0; jj<16; jj++)
    //         {
    //             printf("%f ", C[ii*16 + jj]);
    //         }
    //         printf("\n");
    //     }
    // }


    

    // int row_a = i * WMMA_M,
    //     col_b = j * WMMA_N;

    // for (int i = 0; i < k; i += WMMA_K)
    // {
    //     int col_a = i,
    //         row_b = i;

    //     if (row_a < m && col_a < k && 
    //         row_b < k && col_b < n) 
    //     {
    //         // load a 16x16 matrix into the matrix fragements
    //         nvcuda::wmma::load_matrix_sync(a_frag, A_sh + col_a + row_a * WMMA_M, WMMA_M);
    //         nvcuda::wmma::load_matrix_sync(a_frag, B_sh + col_b + row_b * WMMA_N, WMMA_N);

    //         // nvcuda::wmma::load_matrix_sync(a_frag, A + col_a + row_a * ld_a, ld_a);
    //         // nvcuda::wmma::load_matrix_sync(b_frag, B + col_b + row_b * ld_b, ld_b);

    //         // wmma instruction computes 16x16x16
    //         // each TC does 4x4x4
    //         // 16 cycles to do 1 wmma
    //         nvcuda::wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);

    //         nvcuda::wmma::store_matrix_sync(C + col_b + row_a * ld_c, acc_frag, ld_c, nvcuda::wmma::mem_row_major);
    //     }
        
    // }


    // int row_c = i * WMMA_M;
    // int col_c = j * WMMA_N;
    
    // // bounds checking
    // if (row_c < m && col_c < n) {

    //     // load in C into the accumulator fragment
    //     // nvcuda::wmma::load_matrix_sync(c_frag, C + col_c + row_c * ld_c, ld_c, nvcuda::wmma::mem_row_major);

    //     // scale result and C and sum them together
    //     // for (int i = 0; i < c_frag.num_elements; i++) {
    //     //     c_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
    //     // }

    //     // store result
        
    // }

}


// __device__ void pack_sh_mem(
//     half *A, 
//     int ld_a, 
//     int k, 
//     half *A_sh
// )
// {
//     nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::col_major> a_frag;

//     nvcuda::wmma::load_matrix_sync(a_frag, A, ld_a);

//     nvcuda::wmma::store_matrix_sync(A_sh, a_frag, WMMA_M)
// }

__global__ void  gpu_tc_gemm_(
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






    // 48 KB of shared memory per block
    __shared__ char shared_memory[48 * 1024]; 

    // let A and B get half of the shared memory
    half *A_sh = (half*) &shared_memory[0];
    half *B_sh = (half*) &shared_memory[24 * 1024];

    // pack_sh_mem(A, m, k, A_sh)

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::col_major> a_frag_0;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::row_major> b_frag_0;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag_00; 
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag_00; 
   
    // zero out accumulator fragements
    nvcuda::wmma::fill_fragment(acc_frag_00, 0.0f);

    for(int i=0; i<k; i+=WMMA_K)
    {
        nvcuda::wmma::load_matrix_sync(a_frag_0, A_sh, 16);
        nvcuda::wmma::load_matrix_sync(b_frag_0, B_sh, 16);
        // nvcuda::wmma::load_matrix_sync(a_frag_1, A_sh, 16);
        // nvcuda::wmma::load_matrix_sync(b_frag_1, B_sh, 16);

        nvcuda::wmma::mma_sync(acc_frag_00, a_frag_0, b_frag_0, acc_frag_00);
        // nvcuda::wmma::mma_sync(acc_frag_01, a_frag_0, b_frag_1, acc_frag_01);
        // nvcuda::wmma::mma_sync(acc_frag_10, a_frag_1, b_frag_0, acc_frag_10);
        // nvcuda::wmma::mma_sync(acc_frag_11, a_frag_1, b_frag_1, acc_frag_11);
    }

    // scale C
    nvcuda::wmma::load_matrix_sync(c_frag_00, C, 32, nvcuda::wmma::mem_row_major);
    // nvcuda::wmma::load_matrix_sync(c_frag_01, C + row*32*16 + col*16, 32, nvcuda::wmma::mem_row_major);
    // nvcuda::wmma::load_matrix_sync(c_frag_10, C + row*32*16 + col*16, 32, nvcuda::wmma::mem_row_major);
    // nvcuda::wmma::load_matrix_sync(c_frag_11, C + row*32*16 + col*16, 32, nvcuda::wmma::mem_row_major);
    
    for (int i = 0; i < c_frag_00.num_elements; i++) {
      c_frag_00.x[i] = alpha * acc_frag_00.x[i] + beta * c_frag_00.x[i];
    //   c_frag_01.x[i] = alpha * acc_frag_01.x[i] + beta * c_frag_01.x[i];
    //   c_frag_10.x[i] = alpha * acc_frag_10.x[i] + beta * c_frag_10.x[i];
    //   c_frag_11.x[i] = alpha * acc_frag_11.x[i] + beta * c_frag_11.x[i];
    }

    nvcuda::wmma::store_matrix_sync(C + row*32*16 + col*16, c_frag_00, 32, nvcuda::wmma::mem_row_major);
    // nvcuda::wmma::store_matrix_sync(C + row*32*16 + col*16, c_frag_01, 32, nvcuda::wmma::mem_row_major);
    // nvcuda::wmma::store_matrix_sync(C + row*32*16 + col*16, c_frag_10, 32, nvcuda::wmma::mem_row_major);
    // nvcuda::wmma::store_matrix_sync(C + row*32*16 + col*16, c_frag_11, 32, nvcuda::wmma::mem_row_major);

}



/*

    TODO:
    large C - ensure that it is being written out disjointly

    compare - 1 frag vs multiple (whats the difference?)

    try out different combos of warps and frags
    - 1 warp  | 4 frags (current code)
    - 4 warps | 2 frags
    - 8 warps | 1 frag
    ... 

    what we know
    - increasing k -> increases TC util

*/
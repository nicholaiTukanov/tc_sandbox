#include "monolithic.h"

__device__ void pack_half_matrix_a(
               half *A,
               half *A_sh,
               int ld_a,
               int k,
               half *A_pack
)
{

    /*
        number of threads = warps * 32
        warps = (m*n) / (WMMA_M * WMMA_N)

        threads are configured in the following way (all elems enclosed in brackets are in a warp)

                        warp_x 0               warp_x 1                   warp_x x
        warp_y 0  [t(0,0) ... t(0,31)] [t(0,32) ... t(0,63)] ... [t(0,32*x) ... t(0,32*x+31)]

        warp_y 1  [t(1,0) ... t(1,31)] [t(1,32) ... t(1,63)] ... [t(1,32*x) ... t(1,32*x+31)]
                                                .
                                                .
                                                .
        warp_y y  [t(y,0) ... t(y,31)] [t(y,32) ... t(y,63)] ... [t(y,32*x) ... t(y,32*x+31)]

        if 1 warp = 1 block, then a warp needs to pack the entire shared memory allocated to a block
    */
    int i = (blockIdx.x * blockDim.x + threadIdx.x),
        j = (blockIdx.y * blockDim.y + threadIdx.y);

    // 1 warp will pack 32 elements (2 cols of A)

    // lets assume we have access to only 1 thread
    if(i < 32 && j == 0)
    {
        int k_idx = 0;
        for(int p=0; p < (k / 32); p+=32)
        {
            // col 1
            if (i < 16)
            {
                A_pack [p + i] = A [k_idx + i*k];
                // A_sh [p + i] = A [k_idx + i*k];
                
            }
                
            // col 2
            else
            {
                A_pack [p + i] = A [(k_idx+1) + (i-16)*k];
                // A_sh [p + i] =  A [(k_idx+1) + (i-16)*k];
            }
                
            k_idx += 2;
        }
    }
    __syncthreads();
}

__global__ void gpu_pack_test(
        half *A,
        int m,
        int k,
        half *A_pack
)
{

    extern __shared__ half A_sh[];

    // half * = shared_mem;
    
    int i = (blockIdx.x * blockDim.x + threadIdx.x);

    // 1 warp will pack 32 elements (2 cols of A)

    // lets assume we have access to only 1 warp

    int p = 0;
    for(int k_idx=0; k_idx < k; k_idx+=2)
    {
        // col 1
        if (i < 16)
        {
            A_pack [p + i] = A [k_idx + i*k];
            // A_sh [p + i] = A [k_idx + i*k];
            
        }
            
        // col 2
        else
        {
            A_pack [p + i] = A [(k_idx+1) + (i-16)*k];
            // A_sh [p + i] =  A [(k_idx+1) + (i-16)*k];
        }
            
        p += 32;
    }

    __syncthreads();


    // if (i == 0)
    // {
    //     for (int ii = 0; ii < (m*k); ii++)
    //         A_pack[ii] = A_sh [ii];
    // }

}
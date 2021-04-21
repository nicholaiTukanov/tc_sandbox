#include "monolithic.h"

__global__ void kernel_call()
{

}


typedef half in_type;
typedef float out_type;


__host__ void fp16_gemm_driver()
{

    // matrix size
    int m, n, k;

    int p_start,
        p_end,
        p_inc;

    float milli, best_time;

    int nrep = 5;

    cudaEvent_t start, stop;
    create_cuda_event(start);
    create_cuda_event(stop);


    // input matrix buffers
    in_type  *A_host, *B_host;
    out_type *C_host;
    in_type  *A_dev, *B_dev;
    out_type *C_dev;


    for (int p = p_start; p < p_end; p += p_inc)
    {
        best_time = 1e9;

        m = n = k = p;

        A_host = ( in_type *) malloc (sizeof( in_type) * m * k); 
        B_host = ( in_type *) malloc (sizeof( in_type) * k * n);
        C_host = (out_type *) malloc (sizeof(out_type) * m * n);
    

        for (int irep=0; irep < nrep; irep++) 
        {

            cudaEventRecord(start);
            kernel_call <<< BLOCKS, THREADS >>> ();

            milli = 0;
            cudaEventElapsedTime(&milli, start, stop);
            best_time = min(best_time, milli);
        }
    



        free(A_host); free(B_host); free(C_host);
        cudaFree(A_dev); cudaFree(B_dev); cudaFree(C_dev);
    }

}
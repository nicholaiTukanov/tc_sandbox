#include "monolithic.h"

#define BLOCKS 1
#define THREADS 1

#define ROW_MAJOR 1

#if ROW_MAJOR
#define LAYOUT nvcuda::wmma::row_major
#else
#define LAYOUT nvcuda::wmma::col_major 
#endif

typedef half in_type;
typedef float out_type;


__global__ void kernel_call(
    in_type *A,
    in_type *B,
    out_type *C,
    out_type alpha,
    out_type beta,
    int ld_a,
    int ld_b,
    int ld_c,
    int m,
    int n,
    int k
)
{

}


__host__ void fp16_gemm_driver()
{

    // matrix size
    int m, n, k;

    // strides
    int rs_a, cs_a, 
        rs_b, cs_b, 
        rs_c, cs_c; 
    int ld_a,
        ld_b,
        ld_c;

    int p_start = 16,
        p_end   = 16,
        p_inc   = 16;

    // perf and correctness
    float milli, best_time;
    double diff, tflops;

    int nrep = 5;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // input matrix buffers
    in_type  *A_host, *B_host;
    out_type *C_host, *C_ref_host;

    in_type  *A_dev, *B_dev;
    out_type *C_dev;

    // scalars
    out_type alpha = 1.0, beta = 1.0;


    printf("m, n, k, tflops, diff\n");
    // loop over different problem sizes
    for (int p = p_start; p < p_end; p += p_inc)
    {
        best_time = 1e9;

        m = n = k = p;

        #if ROW_MAJOR
        rs_a = ld_a = k, cs_a = 1, 
        rs_b = ld_b = n, cs_b = 1, 
        rs_c = ld_c = n, cs_c = 1;
        #else
        rs_a = 1, cs_a = ld_a = m, 
        rs_b = 1, cs_b = ld_b = k,
        rs_c = 1, cs_c = ld_c = m;
        #endif    

        A_host = ( in_type *) malloc (sizeof( in_type) * m * k); 
        B_host = ( in_type *) malloc (sizeof( in_type) * k * n);
        C_host = (out_type *) malloc (sizeof(out_type) * m * n);
        C_ref_host = (out_type *) malloc (sizeof(out_type) * m * n);

        init_matrix<in_type>(A_host, m, k, rs_a, cs_a);
        init_matrix<in_type>(B_host, k, n, rs_b, cs_b);
        init_matrix<out_type>(C_host, m, n, rs_c, cs_c);


    
        cudaMalloc((void **)&A_dev, sizeof( in_type) * m * k);
        cudaMalloc((void **)&B_dev, sizeof( in_type) * k * n);
        cudaMalloc((void **)&C_dev, sizeof(out_type) * m * n);

        cudaMemcpy(A_dev, A_host, sizeof( in_type) * m * k, cudaMemcpyHostToDevice);
        cudaMemcpy(B_dev, B_host, sizeof( in_type) * k * n, cudaMemcpyHostToDevice);
        cudaMemcpy(C_dev, C_host, sizeof( in_type) * m * n, cudaMemcpyHostToDevice);

        for (int irep=0; irep < nrep; irep++) 
        {

            cudaEventRecord(start);

            kernel_call <<< BLOCKS, THREADS >>> 
                    (   
                        A_dev, 
                        B_dev, 
                        C_dev,
                        alpha,
                        beta,
                        ld_a,
                        ld_b,
                        ld_c,
                        m,
                        n,
                        k
                    );
        

            milli = 0;
            cudaEventElapsedTime(&milli, start, stop);
            best_time = min(best_time, milli);
        }

        memcpy(C_ref_host, C_host, sizeof(float) * m * n);

        cpu_ref_sgemm<half, float>
        (
            A_host, B_host, C_ref_host, 
            m, n, k,
            rs_a, cs_a,
            rs_b, cs_b,
            rs_c, cs_c,
            alpha, beta
        );
        
        max_abs_diff<float>(C_ref_host, C_dev, m, n, &diff, 1e-3);

        tflops = (((double)m * n * k * 2) / (best_time / 1000.)) / 1e12;
        printf("%d, %d, %d, %7.2f, %8.4le\n",
                  m, n, k, tflops, diff);

        free(A_host); free(B_host); free(C_host);
        cudaFree(A_dev); cudaFree(B_dev); cudaFree(C_dev);
    }

}
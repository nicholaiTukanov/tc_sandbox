#include "monolithic.h"

#define ROW_MAJOR 1

#if ROW_MAJOR
#define LAYOUT nvcuda::wmma::row_major
#else
#define LAYOUT nvcuda::wmma::col_major 
#endif

__host__ void fp16_gemm_driver(
    vector<int> inputs
)
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

    // perf and correctness
    float milli, best_time;
    double diff, tflops;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // input matrix buffers
    half  *A_host, *B_host;
    float *C_host, *C_ref_host;

    half  *A_dev, *B_dev;
    float *C_dev;

    // scalars
    float alpha = 1.0, 
          beta = 0.0;

    int p_start  = inputs[P_START],
        p_end    = inputs[P_END],
        p_inc    = inputs[P_INC],
        nrepeats = inputs[NREPEATS];

    printf("m, n, k, tflops, diff\n");
    // loop over different problem sizes
    for (int p = p_start; p <= p_end; p += p_inc)
    {
        // printf("starting to run over set of problem sizes\n");

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

        // allocate matrix buffers mem on heap
        A_host = ( half *) malloc (sizeof( half) * m * k); 
        B_host = ( half *) malloc (sizeof( half) * k * n);
        C_host = (float *) malloc (sizeof(float) * m * n);
        C_ref_host = (float *) malloc (sizeof(float) * m * n);

        init_matrix<half>(A_host, m, k, rs_a, cs_a);
        init_matrix<half>(B_host, k, n, rs_b, cs_b);
        init_matrix<float>(C_host, m, n, rs_c, cs_c);
        // printf("matrices have been initialized\n");
    
        cudaMalloc((void **)&A_dev, sizeof( half) * m * k);
        cudaMalloc((void **)&B_dev, sizeof( half) * k * n);
        cudaMalloc((void **)&C_dev, sizeof(float) * m * n);

        cudaMemcpy(A_dev, A_host, sizeof( half) * m * k, cudaMemcpyHostToDevice);
        cudaMemcpy(B_dev, B_host, sizeof( half) * k * n, cudaMemcpyHostToDevice);
        // printf("data has been copied to global memory\n");

        int x = m / 16;
        int y = n / 16;

        // printf("number of warps = %d\n", x * y);
        // printf("x = %d\n", x);
        // printf("y = %d\n", y);

        dim3 BLOCKS( x, y );
        dim3 THREADS(32);

        for (int irep=0; irep < nrepeats; irep++) 
        {
            // original data
            cudaMemcpy(C_dev, C_host, sizeof(float) * m * n, cudaMemcpyHostToDevice);

            cudaEventRecord(start);

            // call kernel

            // 1 block = 1 warp
            // 1 warp will compute a 16x16x16 shgemm
            // number of warps = number of 16x16 tiles of c
            gpu_tc_gemm <<< BLOCKS, THREADS >>> 
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

            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
        

            milli = 0;
            cudaEventElapsedTime(&milli, start, stop);
            best_time = min(best_time, milli);
        }

        // printf("kernel has finished running\n");

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
        
        cudaMemcpy(C_host, C_dev, sizeof(float) * m * n, cudaMemcpyDeviceToHost);

        // print_matrix(C_ref_host, "C_cpu", m, n, rs_c, cs_c);
        // print_matrix(C_host, "C_gpu", m, n, rs_c, cs_c);

        max_abs_diff<float>(C_ref_host, C_host, m, n, &diff, 1e-3);

        tflops = (((double)m * n * k * 2) / (best_time / 1000.)) / 1e12;
        printf("%d, %d, %d, %.2f, %8.4le\n",
                  m, n, k, tflops, diff);

        free(A_host); free(B_host); free(C_host); free(C_ref_host);
        cudaFree(A_dev); cudaFree(B_dev); cudaFree(C_dev);
    }

}
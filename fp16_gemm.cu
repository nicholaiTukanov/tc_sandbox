#include "monolithic.h"

#define ROW_MAJOR 1

#if ROW_MAJOR
#define C_LAYOUT nvcuda::wmma::mem_row_major
#else
#define C_LAYOUT nvcuda::wmma::mem_col_major
#endif

#define NUM_REPEATS 3

#define P_START 16
#define P_END 16
#define P_INC 16

// each warp computes a 16x16x16 mat-computation
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

#define M_TILES 1
#define N_TILES 1
#define K_TILES 1

// Must be multiples of 16 for wmma code to work
//
#define REAL_M M_TILES * WMMA_M
#define REAL_N N_TILES * WMMA_N
#define REAL_K K_TILES * WMMA_K

#define WARP_SIZE 32
#define WARPS_PER_BLOCK 8
#define THREADS_PER_BLOCK (WARP_SIZE * WARPS_PER_BLOCK)

// Performs an MxNxK GEMM (D=alpha*A*B + beta*C) assuming:
//  1) Matrices are packed in memory.
//  2) M, N and K are multiples of 16.
//  3) Neither A nor B are transposed.
__global__ void wmma_fp16_gemm(half *A, half *B, float *C, float *D, 
                            int m_ld, int n_ld, int k_ld, 
                            float alpha, float beta) 
{
    // Leading dimensions. Packed with no transpositions.

    #if ROW_MAJOR
    int lda = k_ld;
    int ldb = n_ld;
    int ldc = n_ld;
    #else
    int lda = m_ld;
    int ldb = k_ld;
    int ldc = m_ld;
    #endif

    // Tile using a 2D grid
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);

    #if ROW_MAJOR
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::row_major> a_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::row_major> b_frag;
    #else 
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::col_major> a_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::col_major> b_frag;
    #endif

    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    nvcuda::wmma::fill_fragment(acc_frag, 0.0f);

    // perform 16x16x16 mat-mul
    for (int i = 0; i < k_ld; i += WMMA_K) {

        int aCol = i;
        int aRow = warpM * WMMA_M;
        int bCol = warpN * WMMA_N;
        int bRow = i;

        // bounds checking
        if (aRow < m_ld && aCol < k_ld && bRow < k_ld && bCol < n_ld) {

            // load inputs
            #if ROW_MAJOR
            nvcuda::wmma::load_matrix_sync(a_frag, A + aCol + aRow * lda, lda);
            nvcuda::wmma::load_matrix_sync(b_frag, B + bCol + bRow * ldb, ldb);
            #else
            nvcuda::wmma::load_matrix_sync(a_frag, A + aRow + aCol * lda, lda);
            nvcuda::wmma::load_matrix_sync(b_frag, B + bRow + bCol * ldb, ldb);
            #endif


            // do matrix-multiplication
            nvcuda::wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
    }

    int cCol = warpN * WMMA_N;
    int cRow = warpM * WMMA_M;

    // bounds checking
    if (cRow < m_ld && cCol < n_ld) {

        // load in c (give it a access pattern)
        nvcuda::wmma::load_matrix_sync(c_frag, C + cCol + cRow * ldc, ldc, C_LAYOUT);

        // scale result and C and sum them together
        for (int i = 0; i < c_frag.num_elements; i++) {
            c_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
        }

        // store result
        nvcuda::wmma::store_matrix_sync(D + cCol + cRow * ldc, c_frag, ldc, C_LAYOUT);
    }
}

// do i get expected throughput
// large k
// parameterize it approt
__host__ void fp16_gemm_driver(int MP_count) {

    cout << "Entering " << __func__ << endl;

    // vars with _h attacted are on the host
    // vars with nothing attached are on the device
    half *A_h, *B_h,
         *A, *B;

    float *C_h_old, *C_h, *D_h, 
          *C, *D;

    // vars for data collection
    float milli, best_time = 1e9;
    double diff, tflops;
    int wrong_count = 0;
    int correct;

    // num times kernel is run
    int nrepeats = NUM_REPEATS;

    float alpha = 1;
    float beta = 0;

    // strides
    int rs_a, cs_a,
        rs_b, cs_b,
        rs_c, cs_c;

    // // 2d tile
    // dim3 gridDim;
    // dim3 blockDim;

    int p_start = P_START,
        p_end   = P_END,
        p_inc   = P_INC;


    cout << endl;

    // loop over k
    for (int p = p_start; p <= p_end; p += p_inc) {

        int m = 16, 
            n = 32,
            k = 16;

        // init the strides
        #if ROW_MAJOR
            rs_a = k; cs_a = 1;
            rs_b = n; cs_b = 1;
            rs_c = n; cs_c = 1;
        #else
            rs_a = 1; cs_a = m;
            rs_b = 1; cs_b = k;
            rs_c = 1; cs_c = m;
        #endif

        // host matrices
        A_h     = (half  *) malloc(sizeof(half ) * m * k);
        B_h     = (half  *) malloc(sizeof(half ) * k * n);
        C_h     = (float *) malloc(sizeof(float) * m * n);
        C_h_old = (float *) malloc(sizeof(float) * m * n);
        D_h     = (float *) malloc(sizeof(float) * m * n);

        // init matrices
        init_matrix<half>(A_h, m, k, rs_a, cs_a);
        init_matrix<half>(B_h, k, n, rs_b, cs_b);
        init_matrix<float>(C_h_old, m, n, rs_c, cs_c);

        print_matrix<half>(A_h, "A_h", m, k, rs_a, cs_a);
        print_matrix<half>(B_h, "B_h", k, n, rs_b, cs_b);

        
        /////////////////////////////////////////////////////////////////////////////////////////////
        // START GPU SECTION

        // device matrices
        cu_error_check((CUresult) cudaMalloc((void **)&A, sizeof(half ) * m * k));
        cu_error_check((CUresult) cudaMalloc((void **)&B, sizeof(half ) * n * k));
        cu_error_check((CUresult) cudaMalloc((void **)&C, sizeof(float) * m * n));
        cu_error_check((CUresult) cudaMalloc((void **)&D, sizeof(float) * m * n));

        // assert that the matrix buffers are 128 B aligned
        assert(((unsigned long long)A) % 128 == 0);
        assert(((unsigned long long)B) % 128 == 0);
        assert(((unsigned long long)C) % 128 == 0);
        assert(((unsigned long long)D) % 128 == 0);

        // copy host matrices to device's matrices (resides in global memory)
        cu_error_check((CUresult) cudaMemcpy(A,     A_h, sizeof(half ) * m * k, cudaMemcpyHostToDevice));
        cu_error_check((CUresult) cudaMemcpy(B,     B_h, sizeof(half ) * n * k, cudaMemcpyHostToDevice));
        cu_error_check((CUresult) cudaMemcpy(C, C_h_old, sizeof(float) * m * n, cudaMemcpyHostToDevice));

        // init D to all zeros
        cu_error_check((CUresult) cudaMemset(D,   0, sizeof(float) * m * n));

        cudaEvent_t start, stop;
        cu_error_check((CUresult) cudaEventCreate(&start));
        cu_error_check((CUresult) cudaEventCreate(&stop ));

        for (int irep = 0; irep < nrepeats; irep++) {
            cu_error_check((CUresult) cudaEventRecord(start));
            wmma_fp16_gemm <<< MP_count, THREADS_PER_BLOCK >>> \
                        (A, B, C, D, 
                         m, n, k, 
                         alpha, beta);
            cu_error_check((CUresult) cudaEventRecord(stop));
            cu_error_check((CUresult) cudaEventSynchronize(stop));
            
            milli = 0;
            cu_error_check((CUresult) cudaEventElapsedTime(&milli, start, stop));
            best_time = min(best_time, milli);
        }

        cu_error_check((CUresult) cudaMemcpy(D_h, D, sizeof(float) * m * n, cudaMemcpyDeviceToHost));

        // END GPU SECTION
        /////////////////////////////////////////////////////////////////////////////////////////////

        // START CPU CORRECTNESS CHECK
        memcpy(C_h, C_h_old, sizeof(float) * m * n);

        // compute cpu ref for correctness checks
        cpu_ref_sgemm<half, float>(A_h, B_h, C_h, 
                                m, n, k,
                                rs_a, cs_a,
                                rs_b, cs_b,
                                rs_c, cs_c,
                                alpha, beta);

        // return holds count of values that are off by greater than the allowed error
        wrong_count += max_abs_diff<float>(C_h, D_h, m, n, &diff, 1e-3);
        // correct = (diff > 1e-3 ? 0 : 1);  
       

        // print data
        tflops = (((double)m * n * k * 2) / (best_time / 1000.)) / 1e12;

        print_matrix<float>(C_h, "C_h", m, n, rs_c, cs_c);
        print_matrix<float>(D_h, "D_h", m, n, rs_c, cs_c);


        printf( "data_gpu_simple_gemm" );
        printf( "( %2lu, 1:4 ) = [ %4lu %4lu %4lu ",
                ( unsigned long )(p - p_start + 1)/p_inc + 1,
                ( unsigned long )m,
                ( unsigned long )n,
                ( unsigned long )k );
        printf( " %7.2f     %8.4le ];\n", tflops, diff );


        // free host matrices
        free(A_h); free(B_h); free(C_h); free(C_h_old); free(D_h);

        // free device matrices
        cu_error_check((CUresult) cudaFree((void *) A)); 
        cu_error_check((CUresult) cudaFree((void *) B)); 
        cu_error_check((CUresult) cudaFree((void *) C)); 
        cu_error_check((CUresult) cudaFree((void *) D));
    }

    cout << endl;

    cout << "Leaving " << __func__ << endl;
}
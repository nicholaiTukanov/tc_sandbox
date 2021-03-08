#include "cuda.h"
#include "cuda_fp16.h"
#include "cuda_runtime.h"
#include "mma.h"
#include "cuda_pipeline.h"
using namespace nvcuda::wmma;

#include <iostream>
#include <stdio.h>
#include <assert.h>
#include <type_traits>
using namespace std;

// misc defines
#define DID_WE_MAKE_IT printf("we made it.\n");
#define max( x, y ) ( ( x ) > ( y ) ? x : y )
#define dabs( x ) ( (x) < 0 ? -(x) : x )

template <typename mat_type_in, typename mat_type_out>
void cpu_ref_sgemm
(
    mat_type_in *A, // GEMM A. m x k
    mat_type_in *B, // GEMM B. k x n
    mat_type_out *C, // GEMM C. m x n
    int m,
    int n,
    int k,
    int rs_a, int cs_a,
    int rs_b, int cs_b,
    int rs_c, int cs_c,
    mat_type_out alpha,
    mat_type_out beta 
);

__host__ void fp16_gemm_driver(int MP_Count);

// system util prototypes
int init();
void cu_error_check(CUresult error);

// matrix util prototypes

template <typename half> 
void init_matrix(half *mat, int m, int n, int rs, int cs);

template <typename mat_type>
void print_matrix(mat_type *mat, const char *name, int m, int n, int rs, int cs);

template <typename mat_type>
int max_abs_diff(mat_type *ref, mat_type *result, int m, int n, double *diff, double max_err);
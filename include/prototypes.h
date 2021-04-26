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
);

// test suite for gpu shgemm 
__host__ void fp16_gemm_driver(
    vector<int> inputs
);

// system util prototypes
int init(bool force_create);
void cu_error_check(CUresult error);

// matrix util prototypes

template <typename half> 
void init_matrix(half *mat, int m, int n, int rs, int cs);

template <typename mat_type>
void print_matrix(mat_type *mat, const char *name, int m, int n, int rs, int cs);

template <typename mat_type>
int max_abs_diff(mat_type *ref, mat_type *result, int m, int n, double *diff, double max_err);
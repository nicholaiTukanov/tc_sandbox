#include "monolithic.h"
#include "cuda_fp16.h"

template <typename mat_type>
void init_matrix(mat_type *mat, int m, int n, int rs, int cs)
{
    for (int i = 0; i<m; i++)
        for (int j = 0; j<n; j++)
            mat[i*rs + j*cs] = (mat_type) (rand() % 3);
}

template <typename mat_type>
void print_matrix(mat_type *mat, const char *name, int m, int n, int rs, int cs)
{

    cout << name << " = [ \n";
    for (int i = 0; i<m; i++) {
        for (int j = 0; j<n; j++)
        {
            if (std::is_same<mat_type, half>::value)
                std::cout << (float) __half2float(mat[i*rs + j*cs]) << " ";
            else
                std::cout << (float) mat[i*rs + j*cs] << " ";
        }
        cout << endl;
    }
    cout << " ]; " << endl;
}

template <typename mat_type>
int max_abs_diff(mat_type *ref, mat_type *result, int m, int n, double *diff, double max_err)
{
    int wrong_count = 0;

    for(int i = 0; i < m; i++)
        for(int j = 0; j < n; j++) {
            *diff = max(*diff, dabs(ref[i*m + j] - result[i*m + j]));
            if (*diff > max_err)
                wrong_count++;
        }
            
    return wrong_count;
} 


template void init_matrix<half> (half *mat, int m, int n, int rs, int cs);
template void init_matrix<float> (float *mat, int m, int n, int rs, int cs);
template void print_matrix<half> (half *mat, const char *name, int m, int n, int rs, int cs);
template void print_matrix<float> (float *mat, const char *name, int m, int n, int rs, int cs);


template int max_abs_diff<float>(float *ref, float *result, int m, int n, double *diff, double max_err);
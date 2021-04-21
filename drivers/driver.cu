#include "monolithic.h"

int main(int argc, char *argv[]) 
{
    // init cuda for tensor cores
    int MP_count = init();
    
    fp16_gemm_driver();
}


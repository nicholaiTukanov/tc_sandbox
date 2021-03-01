#include "monolithic.h"


int main(int argc, char *argv[])
{
    // init cuda for tensor cores
    init();

    fp16_gemm_driver();
}


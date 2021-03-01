#include "monolithic.h"

// prints error number and exits if error is detected
void cu_error_check(CUresult error) {

    if (error) {
        printf("Error #%d occured. Exiting. \n", error);
        exit(error);
    }

}

// init cuda and check for possible errors/compatiblility issues
void init() {

    printf("Initializing CUDA...\n");
    cuInit(0);

    int dev_count;
    cu_error_check(cuDeviceGetCount(&dev_count));
    
    if (dev_count == 0) {
        printf("There are no devices that support CUDA.\n");
        exit (0);
    }

    // get handle for device 0
    CUdevice dev;
    cu_error_check(cuDeviceGet(&dev, 0));

    // get dev 0 properties
    cudaDeviceProp dev_prop;
    cu_error_check((CUresult) cudaGetDeviceProperties(&dev_prop, dev));
    
    // ensure device arch is volta or higher
    if (dev_prop.major < 7) {
        printf("cudaTensorCoreGemm requires SM 7.0 or higher to use Tensor Cores.  Exiting...\n");
        exit(-1);
    }

    printf("Everything looks good.\n");

}
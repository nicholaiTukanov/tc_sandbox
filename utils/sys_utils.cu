#include "monolithic.h"

long int SH_MEM_SZ;

// prints error number and exits if error is detected
void cu_error_check(CUresult error) {

    if (error) {
        printf("Error #%d occured. Exiting. \n", error);
        exit(error);
    }

}

void print_dev_prop(cudaDeviceProp dev_prop) {
    printf("\n");
    printf("Device Name                     = %s\n", dev_prop.name);
    printf("MP count                        = %d\n", dev_prop.multiProcessorCount);
    printf("Max blocks per MP               = %d\n", dev_prop.maxBlocksPerMultiProcessor);
    printf("Max threads per MP              = %d\n", dev_prop.maxThreadsPerMultiProcessor);
    printf("Max threads per block           = %d\n", dev_prop.maxThreadsPerBlock);
    printf("Max thread dims per block       x = %d | y = %d | z = %d\n", dev_prop.maxThreadsDim[0], dev_prop.maxThreadsDim[1], dev_prop.maxThreadsDim[2]);
    printf("Max Grid Size:                  x = %d | y = %d | z = %d\n", dev_prop.maxGridSize[0], dev_prop.maxGridSize[1], dev_prop.maxGridSize[2]);
    printf("Memory Clock Rate               =  %d MHz\n", dev_prop.memoryClockRate / 1000);
    printf("Warp Size                       = %d threads\n", dev_prop.warpSize);
    printf("Shared Memory per MP            = %ld KB\n", dev_prop.sharedMemPerMultiprocessor / 1024UL);
    printf("Shared Memory per Block         = %ld KB\n", dev_prop.sharedMemPerBlock / 1024UL);
    printf("L2 Cache Size                   = %ld KB\n", dev_prop.l2CacheSize / 1024UL);
    printf("Global Memory Size              = %ld GB\n", (long int) (dev_prop.totalGlobalMem / 1e9));
    printf("\n");
}

// init cuda and check for possible errors/compatiblility issues
int init() {

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

    SH_MEM_SZ = dev_prop.sharedMemPerMultiprocessor;

    #if PRINT_PROP
    print_dev_prop(dev_prop);
    #endif

    printf("Initialization is complete.\n\n\n");

    return dev_prop.multiProcessorCount;
}
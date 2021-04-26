#include "monolithic.h"

long int SH_MEM_SZ;

// prints error number and exits if error is detected
void cu_error_check(CUresult error) {

    if (error) {
        printf("Error #%d occured. Exiting. \n", error);
        exit(error);
    }

}

// stringfy the dim3 value
string str_dim3(int *dim_3) {
    string first  = to_string(dim_3[0]);
    string second = to_string(dim_3[1]);
    string third  = to_string(dim_3[2]);
    return "( " + first + ", " + second + ", " + third + " )"; 
}

void save_dev_prop(cudaDeviceProp dev_prop, bool force_create) {

    string file_name = dev_prop.name + (string)"_properites";
    string path_to_file = (string)"./device_properties/" + file_name;

    if ( !file_exists(path_to_file) || force_create)
    {
        ofstream dev_prop_file;
        dev_prop_file.open(path_to_file);
        dev_prop_file << "Device Name                    = " << dev_prop.name << endl;
        dev_prop_file << "MP count                       = " << dev_prop.multiProcessorCount << endl;
        dev_prop_file << "Max blocks per MP              = " << dev_prop.maxBlocksPerMultiProcessor << endl;
        dev_prop_file << "Max threads per MP             = " << dev_prop.maxThreadsPerMultiProcessor << endl;
        dev_prop_file << "Max threads per block          = " << dev_prop.maxThreadsPerBlock << endl;
        dev_prop_file << "Max thread dims per block      " << str_dim3(dev_prop.maxThreadsDim) << endl; 
        dev_prop_file << "Max Grid Size:                 " << str_dim3(dev_prop.maxGridSize) << endl;
        dev_prop_file << "Memory Clock Rate (GHz)        = " << dev_prop.memoryClockRate / 1000 << endl;
        dev_prop_file << "Warp Size                      = " << dev_prop.warpSize << endl;
        dev_prop_file << "Shared Memory per MP (KB)      = " << dev_prop.sharedMemPerMultiprocessor / 1024UL << endl;
        dev_prop_file << "Shared Memory per Block (KB)   = " << dev_prop.sharedMemPerBlock / 1024UL << endl;
        dev_prop_file << "L2 Cache Size (KB)             = " << dev_prop.l2CacheSize / 1024UL << endl;
        dev_prop_file << "Global Memory Size (GB)        = " << (long int) (dev_prop.totalGlobalMem / 1e9) << endl;
        dev_prop_file.close();
    }
    
}

// init cuda and check for possible errors/compatiblility issues
int init(bool force_create) {

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

    save_dev_prop(dev_prop, force_create);

    printf("Initialization is complete.\n\n\n");

    return dev_prop.multiProcessorCount;
}
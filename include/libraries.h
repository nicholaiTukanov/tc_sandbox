
// c libs
#include <stdio.h>
#include <assert.h>
#include <sys/stat.h>
#include <unistd.h>

// c++ libs
#include <iostream>
#include <vector>
#include <string>
#include <type_traits>
using namespace std;
#include <fstream>
#include <cstdarg>
#include <sstream>
#include <iterator>

// cuda libs
#include "cuda.h"
#include "cuda_fp16.h"
#include "cuda_runtime.h"
#include "mma.h"
#include "cuda_pipeline.h"
using namespace nvcuda::wmma;
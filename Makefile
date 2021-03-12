# A simple Makefile for compiling and linking tensor core GEMM kernel

.PHONY  = all \
		  driver

# functions
change_file_types = $(filter-out , $(patsubst %.cu, %.o, $(wildcard $(1)/*.cu)))

# include/lib paths
SRC_PATH       := .
CUDA_PATH      := /usr/local/cuda-11.2# ensure this points to installed cuda dir
CUDA_INCLUDE   := $(CUDA_PATH)/include# ensure this points to installed cuda's include dir
INCLUDE        := $(SRC_PATH)/include# sandbox include

# compiler/linker
CC             := $(CUDA_PATH)/bin/nvcc # target nvidia compiler
LINKER         := $(CC)

# compile time macros
PRINT_PROP     := 1
MACROS         := -DPRINT_PROP=1$(PRINT_PROP)

# flags to be passed into CC
CFLAGS         := -I$(CUDA_INCLUDE) -I$(INCLUDE) $(MACROS) -arch=sm_70 -std=c++11

# load flags
CUDA_LD        := -lcuda 

# object files
DRIVER_OBJS    := $(call change_file_types, $(SRC_PATH)/drivers, cu, o)
KERNEL_OBJS    := $(call change_file_types, $(SRC_PATH)/gemm_kernels, cu, o)
UTIL_OBJS      := $(call change_file_types, $(SRC_PATH)/utils, cu, o)

# list of all object files
OBJS          := $(sort $(DRIVER_OBJS) $(KERNEL_OBJS) $(UTIL_OBJS))

all: driver

# compilation rule
$(OBJS): %.o: %.cu
	$(CC) $(CFLAGS) -c $< -o $@

# linker rule
driver: $(OBJS)
	$(LINKER) $(OBJS) -o ./driver.x $(CUDA_LD) 	

clean_driver:
	rm -f ./drivers/*.o

clean_kernels:
	rm -f ./gemm_kernels/*.o

clean_utils:
	rm -f ./utils/*.o

clean: clean_driver clean_kernels clean_utils 
	rm -f *.x

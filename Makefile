# a simple Makefile for cuda compiling

# ensure this points to installed cuda dir
CUDA_PATH    := /usr/local/cuda-11.2

CUDA_INCLUDE := $(CUDA_PATH)/include # ensure this points to installed cuda include dir
# CUDA_LIB     := $(CUDA_PATH)/lib64/libcudart.so
CUDA_LD      := -lcuda # load cuda flags for nvcc

CC           := $(CUDA_PATH)/bin/nvcc # target nvidia compiler
LINKER       := $(CC)

CFLAGS       := -I$(CUDA_INCLUDE) -arch=sm_70 -std=c++11

# grab object files
TEST_DIR     := .
TEST_OBJS    := $(patsubst %.cu,%.o,$(wildcard $(TEST_DIR)/*.cu))

# linker rule
test: $(TEST_OBJS) $(UTIL_OBJS)
	$(LINKER) $(TEST_OBJS) $(CUDA_LIB) -o ./driver.x $(CUDA_LD) 	

# compile rule
$(TEST_OBJS): %.o: %.cu
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf *.o *.x
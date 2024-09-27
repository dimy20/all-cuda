CC=nvcc

INCLUDE_DIR=/usr/local/cuda-12.5/include
LIBS_DIR=/usr/local/cuda-12.5/lib64
BUILD_DIR=./build

CFLAGS = -I$(INCLUDE_DIR) -L$(LIBS_DIR) -arch=sm_86

vec_add: $(BUILD_DIR)/vec_add
indexing: $(BUILD_DIR)/indexing
matmul: $(BUILD_DIR)/matmul

$(BUILD_DIR)/indexing: indexing.cu
	$(CC) $(CFLAGS) -o $@ $^

$(BUILD_DIR)/vec_add: vec_add.cu
	$(CC) $(CFLAGS) -o $@ $^

$(BUILD_DIR)/matmul: matmul.cu
	$(CC) $(CFLAGS) -o $@ $^

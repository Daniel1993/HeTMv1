CUDA_PATH     ?= /usr/local/cuda-11.0
NVCC          ?= $(CUDA_PATH)/bin/nvcc --default-stream per-thread -arch sm_60

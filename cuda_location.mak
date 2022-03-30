# TODO: in rsync.sh change this
# CUDA_PATH     ?= /usr/local/cuda-11.0 
CUDA_PATH     ?= /usr/local/cuda
NVCC          ?= $(CUDA_PATH)/bin/nvcc --default-stream per-thread -arch sm_60
# NVCC := nvcc 

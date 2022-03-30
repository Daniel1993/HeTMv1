### CONFIG
CUDA_UTIL_DIR ?= ../cuda-utils

include ../../cuda_location.mak
# CUDA_DIR      ?= /usr/local/cuda-11.0
# CUDA_DIR      ?= /usr/local/cuda
# NVCC     := $(CUDA_DIR)/bin/nvcc

TESTS_DIR     ?= ./tests
FVS_LIB_DIR   ?= ../../sAnnealing/BB
FVS_LIB       ?= bb

INCLUDES := -I ./include -I $(CUDA_UTIL_DIR)/include 
LIBS     :=
DEFINES  :=
INCLUDES += -I $(FVS_LIB_DIR)
LIBS     += -L $(FVS_LIB_DIR) -l$(FVS_LIB) -lbsd

CC       := g++
CFLAGS   := -c -std=gnu99
CXXFLAGS := -c -std=c++11
NVFLAGS  := -dc -std=c++11
# --default-stream per-thread -arch sm_60 
LDFLAGS  :=
# NVCC     := nvcc
AR       := ar rcs

ifeq ($(CFG),debug)
CFLAGS   += -g 
CXXFLAGS += -g 
NVFLAGS  += -g -G 
endif

SRCS     := ./src/zipf_dist.cpp
#$(shell ls -rt -d -1 ./src/*.c ./src/*.cpp ./src/*.cu 2>/dev/null)
OBJS     := $(addsuffix .o, $(basename $(SRCS)))

PRSTM     := ./pr-stm-obj.o
BANK_OBJ  := ./bank.o
BANK      := ./bank
MEMCD_OBJ := ./memcd.o
MEMCD     := ./memcd

CFLAGS   += $(INCLUDES) $(DEFINES)
CXXFLAGS += $(INCLUDES) $(DEFINES)
NVFLAGS  += $(INCLUDES) $(DEFINES)
LDFLAGS  += $(LIBS)
#--default-stream per-thread -arch sm_60 

### RULES

all: $(PRSTM) $(BANK) $(MEMCD)
	$(MAKE) $(TESTS_DIR)

%.o: %.cu
	$(NVCC) $(NVFLAGS) -o $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^

%.o: %.c
	$(CC) $(CFLAGS) -o $@ $^

$(BANK): $(OBJS) $(BANK_OBJ) $(PRSTM)
	$(NVCC) -o $@ $^ $(LDFLAGS)

$(MEMCD): $(OBJS) $(MEMCD_OBJ) $(PRSTM)
	$(NVCC) -o $@ $^ $(LDFLAGS)

$(PRSTM): ./src/aux.cu
	$(NVCC) $(NVFLAGS) $^ -o $@ $(INCLUDES)

clean:
	rm -f $(BANK) $(BANK_OBJ) $(PRSTM) $(MEMCD) $(MEMCD_OBJ) $(OBJS)

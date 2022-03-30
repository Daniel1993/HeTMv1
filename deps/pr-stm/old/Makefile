### CONFIG
CUDA_DIR := /usr/local/cuda
# CUDA_DIR := /extra/cuda-8.0

INCLUDES :=
DEFINES  :=
LIBS     := -L$(CUDA_DIR)/lib64

CFLAGS   := -g -O0
CXXFLAGS := -g -O0
LDFLAGS  := -lcuda -lcudart

SRCS_C   := $(shell ls -rt -d -1 *.c *.cpp)
SRCS_CU  := $(shell ls -rt -d -1 *.cu)

APP      := ./bank
OBJS_C   := $(addsuffix .o, $(basename $(SRCS_C)))
OBJS_CU  := $(addsuffix .o, $(basename $(SRCS_CU)))

CFLAGS   += $(INCLUDES) $(DEFINES)
CXXFLAGS += $(INCLUDES) $(DEFINES)
LDFLAGS  += $(LIBS)

NVCC     := nvcc
CC       := gcc
CXX      := g++

### RULES
all: $(APP)

$(OBJS_C): $(SRCS_C)

$(OBJS_CU): $(SRCS_CU)
	$(NVCC) -c $@ $^ $(CXXFLAGS)

$(APP): $(OBJS_C) $(OBJS_CU)
	$(CXX) -o $@ $^ $(LDFLAGS)

clean:
	rm -f $(APP) $(OBJS_C) $(OBJS_CU)

# CUDA_PATH     ?= /usr/local/cuda-11.0
# CUDA_PATH     ?= /usr/local/cuda
# NVCC = $(CUDA_PATH)/bin/nvcc
# NVCC:= nvcc
include ../../cuda_location.mak

CFG           ?= prod

INCLUDES := -I ./include \
	-I $(CUDA_PATH)/include
#
DEFINES  :=
#
LIBS     := -L $(CUDA_PATH)/lib64 -lcudart -lcuda -lnuma
#
LIB  = libcuda-util.a
CC   = gcc
CXX  = g++
AR   = ar rcs

### Add this if debug needed (GPU run much slower)
DEBUG_FLAGS  := -g -lineinfo
NV_DEB_FLAGS := -G -g -lineinfo

ifeq ($(BMAP_ENC_1BIT),1)
DEFINES += -DBMAP_ENC_1BIT
endif

CFLAGS     := -c $(DEFINES) $(INCLUDES) -Wall 
CXXFLAGS   := -c $(DEFINES) $(INCLUDES) -Wall -std=c++11
NVFLAGS    := -c $(DEFINES) $(INCLUDES) -std=c++11
# --default-stream per-thread -arch sm_60
LDFLAGS    := $(LIBS)

ifeq ($(CFG),debug)
CFLAGS   += $(DEBUG_FLAGS)
CXXFLAGS += $(DEBUG_FLAGS)
NVFLAGS  += $(NV_DEB_FLAGS)
endif

ifeq ($(CFG),prod)
# not working with -O2
CFLAGS   += -O1
CXXFLAGS += -O1
NVFLAGS  += -O1
endif

SRC        := ./src
DEPS_CUDA_UTIL_SRC_DIR := $(SRC)
include Makefile.sources

BANK_OBJS  := \
	$(DEPS_CUDA_UTIL_OBJS)

.PHONY:	all clean

all: $(LIB)
	# done

test_bandwidth: test_bandwidth.o
	$(NVCC) $(DEFINES) $(INCLUDES) -std=c++11 -o $@ $<

$(LIB): $(BANK_OBJS)
	$(AR) $@ $(BANK_OBJS)

%.o:	%.c
	@echo ""
	$(CC) $(CFLAGS) -o $@ $<

%.o:	%.cpp
	@echo ""
	$(CXX) $(CXXFLAGS) -o $@ $<

%.o:	%.cu
	@echo ""
	$(NVCC) $(NVFLAGS) -o $@ $<

clean:
	rm -f $(LIB) *.o $(SRC)/*.o

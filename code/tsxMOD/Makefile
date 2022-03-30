CURR_DIR      ?= ~/projs/HeTM_V1/code/tsxMOD
HETM_ROOT     ?= $(CURR_DIR)/../hetm

CUDA_UTIL_DIR ?= $(CURR_DIR)/../../deps/cuda-utils
HTM_ARCH_DEPS ?= $(CURR_DIR)/../../deps/arch_dep
HTM_SGL       ?= $(CURR_DIR)/../../deps/htm_alg

CFG           ?= prod

INCLUDES := -I $(CURR_DIR)/include \
	-I $(HETM_ROOT)/include
#
DEFINES  :=
#
# LIBS     := -L $(HETM_ROOT) -lhetm -lpthread
LIBS     := -lpthread
#
LIB  = libtsxMOD.a
NVCC = nvcc
CC   = gcc
CXX  = g++
AR   = ar rcs

### Add this if debug needed (GPU run much slower)
DEBUG_FLAGS  := -g -lineinfo -DHETM_DEB=1
NV_DEB_FLAGS := -G -g -lineinfo -DHETM_DEB=1

INCLUDES  += 	-I $(HTM_ARCH_DEPS)/include \
	-I $(HTM_SGL)/include \
	-I $(CUDA_UTIL_DIR)/include
LIBS      += -L $(HTM_SGL)/bin -lhtm_sgl \
	-L $(CUDA_UTIL_DIR) -lcuda-util

LOG_TYPE ?= BMAP
ifeq ($(INST_CPU),1)
  DEFINES += -DHETM_INSTRUMENT_CPU
endif
############
### LOG_TYPE
ifeq ($(LOG_TYPE),VERS)
DEFINES += -DHETM_LOG_TYPE=1
endif
ifeq ($(LOG_TYPE),ADDR)
DEFINES += -DHETM_LOG_TYPE=2
endif
ifeq ($(LOG_TYPE),BMAP)
DEFINES += -DHETM_LOG_TYPE=3
endif
ifeq ($(LOG_TYPE),VERS2)
DEFINES += -DHETM_LOG_TYPE=4
endif
############

CFLAGS     := -c $(DEFINES) $(INCLUDES)
CXXFLAGS   := -c $(DEFINES) $(INCLUDES) -std=c++11
LDFLAGS    := $(LIBS)

TSX_FLAGS += -DUSE_TSX_IMPL
CFLAGS    += $(TSX_FLAGS) -mrtm
CXXFLAGS  += $(TSX_FLAGS) -mrtm
NVFLAGS   += $(TSX_FLAGS) --compiler-options='-mrtm'

ifeq ($(CFG),debug)
CFLAGS   += $(DEBUG_FLAGS)
CXXFLAGS += $(DEBUG_FLAGS)
NVFLAGS  += $(NV_DEB_FLAGS)
endif

SRC      := $(CURR_DIR)/src
OBJS     := \
	$(SRC)/tsx_impl.o

.PHONY:	all clean

all: $(LIB)
	# Done

$(LIB): $(OBJS)
	@echo " ------ FLAGS ------ \n$(DEFINES)"
	@echo " ------------------- \nLinking..."
	$(AR) $@ $(OBJS)

%.o:	%.c
	@echo "   >>> Building $@"
	@$(CC) $(CFLAGS) -o $@ $<

%.o:	%.cpp
	@echo "   >>> Building $@"
	@$(CXX) $(CXXFLAGS) -o $@ $<

run:
	make all

clean:
	rm -f $(LIB) *.o $(SRC)/*.o

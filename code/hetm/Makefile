CURR_DIR      ?= ~/projs/HeTM_V1/code/hetm

# CUDA_PATH     ?= /usr/local/cuda-11.0
# CUDA_PATH     ?= /usr/local/cuda
# NVCC := $(CUDA_PATH)/bin/nvcc
# NVCC := nvcc
include ../../cuda_location.mak

TINY_ROOT     ?= $(CURR_DIR)/../tinyMOD
TSX_ROOT      ?= $(CURR_DIR)/../tsxMOD
INPUT_HANDLER ?= $(CURR_DIR)/../shared

CUDA_UTIL_DIR ?= $(CURR_DIR)/../../deps/cuda-utils
HTM_ARCH_DEPS ?= $(CURR_DIR)/../../deps/arch_dep
HTM_SGL       ?= $(CURR_DIR)/../../deps/htm_alg
PR_STM_DIR    ?= $(CURR_DIR)/../../deps/pr-stm
FVS_LIB_DIR   ?= $(CURR_DIR)/../../sAnnealing/BB

CFG           ?= prod

INCLUDES := \
	-I $(CUDA_UTIL_DIR)/include \
	-I $(PR_STM_DIR)/include \
	-I $(TINY_ROOT)/include \
	-I $(TINY_ROOT)/src \
	-I $(TSX_ROOT)/include \
	-I $(CUDA_PATH)/include \
	-I $(CURR_DIR)/include \
	-I $(INPUT_HANDLER)/include \
#
DEFINES  :=
#
LIBS     := \
	-L $(INPUT_HANDLER) -linput_handler \
	-L $(CUDA_PATH)/lib64 -lcudart -lcuda -lcuda-util -lnuma \
	-L $(TINY_ROOT)/lib -lstm -lpthread \
#
LIB  := libhetm.a
CC   := gcc
CXX  := g++
AR   := ar rcs

# Parameters for HeTM
include $(CURR_DIR)/Makefile.defines

DEPS_PR_STM_DIR := $(PR_STM_DIR)
include $(PR_STM_DIR)/Makefile.sources

### Add this if debug needed (GPU run much slower)
DEBUG_FLAGS  := -g
NV_DEB_FLAGS := -G -g

DEBUG_PRINT_FLAGS  := -DHETM_DEB=1
NV_DEB_PRINT_FLAGS := -DHETM_DEB=1

# this is needed for the barrier implementation
INCLUDES  += -I $(HTM_ARCH_DEPS)/include -I $(FVS_LIB_DIR)
ifeq ($(USE_TSX_IMPL),1)
INCLUDES  += -I $(HTM_SGL)/include
LIBS      += -L $(HTM_SGL)/bin -lhtm_sgl -L $(TSX_ROOT) -ltsxMOD
endif

CFLAGS     := -c $(DEFINES) $(INCLUDES)
CXXFLAGS   := -c $(DEFINES) $(INCLUDES) -std=c++11
NVFLAGS    := -dc $(DEFINES) $(INCLUDES) -std=c++11
LDFLAGS    := $(LIBS)

ifeq ($(USE_TSX_IMPL),1)
TSX_FLAGS += -DUSE_TSX_IMPL
endif
CFLAGS    += $(TSX_FLAGS) -march=native -mrtm
CXXFLAGS  += $(TSX_FLAGS) -march=native -mrtm
NVFLAGS   += $(TSX_FLAGS) --compiler-options='-march=native -mrtm'

ifeq ($(CFG),debug)
CFLAGS   += $(DEBUG_FLAGS)
CXXFLAGS += $(DEBUG_FLAGS)
NVFLAGS  += $(NV_DEB_FLAGS)
endif
ifneq ($(CFG),debug)
CFLAGS   += -g -DNDEBUG
CXXFLAGS += -g -DNDEBUG
NVFLAGS  += -g -DNDEBUG
endif
ifeq ($(CFG),deb_cpu_only)
CFLAGS   += $(DEBUG_FLAGS)
CXXFLAGS += $(DEBUG_FLAGS)
NVFLAGS  += $(DEBUG_FLAGS)
endif
ifeq ($(PRINT_DEB),1)
CFLAGS   += $(DEBUG_PRINT_FLAGS)
CXXFLAGS += $(DEBUG_PRINT_FLAGS)
NVFLAGS  += $(NV_DEB_PRINT_FLAGS)
endif

SRC      := $(CURR_DIR)/src
CODE_HTM_SRC_DIR:= $(SRC)
include Makefile.sources
SRCS     := $(CODE_HETM_SRCS)
OBJS     := $(CODE_HETM_OBJS)

.PHONY:	all clean

all: $(LIB) hetm-inst-cpu.o hetm-inst-gpu.o
	# Done

# $(FVS_LIB_DIR)/libgsort.a: $(FVS_LIB_DIR)/gsort_lib.c $(FVS_LIB_DIR)/../graph.c
# 	@echo "Making FVS lib"
# 	$(MAKE) --directory $(FVS_LIB_DIR) libgsort.a

$(FVS_LIB_DIR)/libbb.a: $(FVS_LIB_DIR)/bb_lib.c $(FVS_LIB_DIR)/../graph.c
	@echo "Making FVS lib"
	$(MAKE) --directory $(FVS_LIB_DIR) libbb.a

$(LIB): $(OBJS) ./hetm-inst-cpu.o ./hetm-inst-gpu.o
	@echo " ------ FLAGS ------ \n$(DEFINES)"
	@echo " ------------------- \nLinking..."
	@# $(NVCC) $(NVFLAGS) -O2 -o $(SRC)/hetm-threading-gpu.o $(SRC)/hetm-threading-gpu.cu
	@### TODO: any optimization level blocks for VERS --> TODO: do optimization for BMAP
	@# $(NVCC) $(NVFLAGS) -o $(SRC)/hetm-threading-cpu.o $(SRC)/hetm-threading-cpu.cu
	@# $(CXX) $(CXXFLAGS) -O2 -o $(SRC)/hetm-producer-consumer.o $(SRC)/hetm-producer-consumer.cpp
	@# $(CXX) $(CXXFLAGS) -O2 -o $(SRC)/hetm-log.o $(SRC)/hetm-log.cpp
	@$(AR) $@ $(OBJS)

%.o:	%.c
	@echo "   >>> Building $@"
	$(CC) $(CFLAGS) -o $@ $<

%.o:	%.cpp
	@echo "   >>> Building $@"
	$(CXX) $(CXXFLAGS) -o $@ $<

%.o:	%.cu
	@echo "   >>> Building $@"
	$(NVCC) $(NVFLAGS) -o $@ $<

hetm-inst-cpu.o: ./src/hetm-inst-cpu.cpp
	@echo "   >>> Building $@"
	@# "instrument CPU object (instrument and then compile CPU TM with it)"
	$(NVCC) $(NVFLAGS) -o $@ $<

hetm-inst-gpu.o: ./src/hetm-inst-gpu.cu $(DEPS_PR_STM_OBJS)
	@echo "   >>> Building $@"
	@# "instrument GPU object (instrument and then compile GPU TM with it)"
	$(NVCC) $(NVFLAGS) -o $@ $<

$(DEPS_PR_STM_OBJS):
	$(MAKE) --directory $(PR_STM_DIR)

run:
	make all

clean:
	rm -f $(LIB) *.o $(SRC)/*.o

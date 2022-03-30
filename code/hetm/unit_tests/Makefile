### CONFIG
CURR_DIR      ?= ~/projs/HeTM_V1/code/hetm/unit_tests


TINY_ROOT     ?= ../../tinyMOD
TSX_ROOT      ?= ../../tsxMOD
HETM_ROOT     ?= ../../hetm
INPUT_HANDLER ?= ../../shared

FVS_LIB_DIR   ?= ../../../sAnnealing/BB
FVS_LIB       ?= bb

CUDA_UTIL_DIR ?= ../../../deps/cuda-utils
HTM_ARCH_DEPS ?= ../../../deps/arch_dep
HTM_SGL       ?= ../../../deps/htm_alg
PR_STM_DIR    ?= ../../../deps/pr-stm

# CUDA_PATH     ?= /usr/local/cuda-11.0
# CUDA_PATH     ?= /usr/local/cuda
# NVCC     := $(CUDA_PATH)/bin/nvcc --default-stream per-thread -arch sm_60
# NVCC     := nvcc --default-stream per-thread -arch sm_60
include ../../../cuda_location.mak

CFG           ?= prod

include Makefile.defines

INCLUDES := \
	-I ../include -I $(CUDA_UTIL_DIR)/include \
	-I $(HETM_ROOT)/include \
	-I $(TINY_ROOT)/include \
	-I $(TSX_ROOT)/include \
	-I $(CUDA_PATH)/include \
	-I $(FVS_LIB_DIR) \
	-I $(PR_STM_DIR)/include
DEFINES  :=
LIBS     := \
	-L $(HETM_ROOT) -lhetm \
	-L $(CUDA_UTIL_DIR) -lcuda-util \
	-L $(TINY_ROOT)/lib -lstm \
	-L $(FVS_LIB_DIR) -l$(FVS_LIB) -lbsd \
	-L $(HTM_SGL)/bin -lhtm_sgl -L $(TSX_ROOT) -ltsxMOD \
	-L $(CUDA_PATH)/lib64 -lcudart -lcuda -lnvToolsExt

CFLAGS   := -g -c -O0 -std=gnu99
CXXFLAGS := -g -c -O0 -std=c++11
NVFLAGS  := -g -G -dc -std=c++11 
LDFLAGS  := -lcppunit


TEST_SRC := $(shell ls -rt -d -1 ./Test*.c ./Test*.cpp ./Test*.cu 2>/dev/null)
RUNNER   := ./runner.cpp
RUNNER_B := ./Runner
TEST_SRC += $(RUNNER)
OBJS     := $(addsuffix .o, $(basename $(TEST_SRC)))
OBJS     += \
	$(PR_STM_DIR)/pr-stm-obj.o \
	$(HETM_ROOT)/hetm-inst-cpu.o \
	$(HETM_ROOT)/hetm-inst-gpu.o \
#
ifeq ($(PROFILE),1)
DEFINES += -DUSE_NVTX
endif

CODE_HTM_SRC_DIR:=$(HETM_ROOT)/src
include $(HETM_ROOT)/Makefile.sources

SRC        := $(CURR_DIR)/src
### TODO: to use the version that enqueues requests in a different thread use bank_w_enqueue.o instead of bank.o

DEPS_PR_STM_DIR := $(PR_STM_DIR)
include $(PR_STM_DIR)/Makefile.sources

DEPS_CUDA_UTIL_SRC_DIR := $(CUDA_UTIL_DIR)/src
include $(CUDA_UTIL_DIR)/Makefile.sources

HTM_ALG_DIR := $(HTM_SGL)
include $(HTM_ALG_DIR)/Makefile.sources

TSX_MOD_DIR := $(TSX_ROOT)
include $(TSX_MOD_DIR)/Makefile.sources

TINY_MOD_DIR := $(TINY_ROOT)
include $(TINY_MOD_DIR)/Makefile.sources

CFLAGS   += $(INCLUDES) $(DEFINES)
CXXFLAGS += $(INCLUDES) $(DEFINES)
NVFLAGS  += $(INCLUDES) $(DEFINES)
LDFLAGS  += $(LIBS)

### RULES

all: $(RUNNER_B)


alldeps:
	@echo " ============================================ "
	@echo " === Building CUDA utils lib "
	@echo " ============================================ "
	$(MAKE) --directory $(CUDA_UTIL_DIR) clean
	$(MAKE) -j14 --directory $(CUDA_UTIL_DIR) \
		CFG=$(CFG) 
	@echo " ============================================ "
	@echo " === Building PR-STM lib "
	@echo " ============================================ "
	$(MAKE) --directory $(PR_STM_DIR) clean
	$(MAKE) -j14 --directory $(PR_STM_DIR) \
		CFG=$(CFG) 
	@echo " ============================================ "
	@echo " === Building HeTM lib "
	@echo " ============================================ "
	$(MAKE) --directory $(HETM_ROOT) clean
	$(MAKE) -j14 --directory $(HETM_ROOT) \
		USE_TSX_IMPL=$(USE_TSX_IMPL) \
		CMP_TYPE=$(CMP_TYPE) \
		LOG_TYPE=$(LOG_TYPE) \
		PR_MAX_RWSET_SIZE=$(PR_MAX_RWSET_SIZE) \
		CFG=$(CFG) \
		PROFILE=$(PROFILE) \
		INST_CPU=$(INST_CPU) \
		USE_UNIF_MEM=$(USE_UNIF_MEM) \
		PRINT_DEB=$(PRINT_DEB) \
		DISABLE_EARLY_VALIDATION=$(DISABLE_EARLY_VALIDATION) \
		HETM_NB_DEVICES=$(HETM_NB_DEVICES) \
		REDUCED_RS=$(REDUCED_RS) \
		DISABLE_RS=$(DISABLE_RS) \
		DISABLE_WS=$(DISABLE_WS) libhetm.a hetm-inst-cpu.o hetm-inst-gpu.o 
	@echo " ============================================ "
	@echo " === Building TinyMOD lib "
	@echo " ============================================ "
	$(MAKE) --directory $(TINY_ROOT) clean
	$(MAKE) -j14 --directory $(TINY_ROOT) \
		USE_TSX_IMPL=$(USE_TSX_IMPL) \
		CMP_TYPE=$(CMP_TYPE) \
		LOG_TYPE=$(LOG_TYPE) \
		PR_MAX_RWSET_SIZE=$(PR_MAX_RWSET_SIZE) \
		CFG=$(CFG) \
		INST_CPU=$(INST_CPU) \
		PRINT_DEB=$(PRINT_DEB)
	@echo " ============================================ "
	@echo " === Building TSXMOD lib "
	@echo " ============================================ "
	$(MAKE) --directory $(TSX_ROOT) clean
	$(MAKE) -j14 --directory $(TSX_ROOT) \
		USE_TSX_IMPL=$(USE_TSX_IMPL) \
		CMP_TYPE=$(CMP_TYPE) \
		LOG_TYPE=$(LOG_TYPE) \
		PR_MAX_RWSET_SIZE=$(PR_MAX_RWSET_SIZE) \
		CFG=$(CFG) \
		INST_CPU=$(INST_CPU) \
		PRINT_DEB=$(PRINT_DEB)
	@echo " ============================================ "
	@echo " === Building HeTM lib (again) "
	@echo " ============================================ "
	$(MAKE) --directory $(HETM_ROOT) clean
	$(MAKE) -j14 --directory $(HETM_ROOT) \
		USE_TSX_IMPL=$(USE_TSX_IMPL) \
		CMP_TYPE=$(CMP_TYPE) \
		LOG_TYPE=$(LOG_TYPE) \
		PR_MAX_RWSET_SIZE=$(PR_MAX_RWSET_SIZE) \
		CFG=$(CFG) \
		PROFILE=$(PROFILE) \
		INST_CPU=$(INST_CPU) \
		USE_UNIF_MEM=$(USE_UNIF_MEM) \
		PRINT_DEB=$(PRINT_DEB) \
		LOG_SIZE=$(LOG_SIZE) \
		DISABLE_EARLY_VALIDATION=$(DISABLE_EARLY_VALIDATION) \
		STM_LOG_BUFFER_SIZE=$(STM_LOG_BUFFER_SIZE) \
		HETM_NB_DEVICES=$(HETM_NB_DEVICES) \
		REDUCED_RS=$(REDUCED_RS) \
		DISABLE_RS=$(DISABLE_RS) \
		DISABLE_WS=$(DISABLE_WS) libhetm.a hetm-inst-cpu.o hetm-inst-gpu.o 


$(HETM_ROOT)/hetm-inst-cpu.o $(HETM_ROOT)/hetm-inst-gpu.o: alldeps

$(PR_STM_DIR)/pr-stm-obj.o: 
	$(MAKE) --directory $(PR_STM_DIR)

%.o: %.cu
	$(NVCC) $(NVFLAGS) -o $@ $^

%.o: %.c
	$(CC) $(CFLAGS) -o $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^

$(RUNNER_B): $(OBJS)
	$(NVCC) $(LDFLAGS)  -o $@ $^ 

clean:
	rm -f $(RUNNER_B) $(OBJS)

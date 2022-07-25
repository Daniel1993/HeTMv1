### TODO: create include files for pr-stm, cuda-utils, tsx and tiny

CURR_DIR      ?= ~/projs/HeTM_V1/benches/memcd
include ../../cuda_location.mak

TINY_ROOT     ?= ../../code/tinyMOD
TSX_ROOT      ?= ../../code/tsxMOD
HETM_ROOT     ?= ../../code/hetm
SHARED        ?= ../shared_bench
INPUT_HANDLER ?= ../../code/shared

CUDA_UTIL_DIR ?= ../../deps/cuda-utils
HTM_ARCH_DEPS ?= ../../deps/arch_dep
HTM_SGL       ?= ../../deps/htm_alg
PR_STM_DIR    ?= ../../deps/pr-stm

FVS_LIB_DIR   ?= ../../sAnnealing/BB
FVS_LIB       ?= bb

CFG           ?= prod

INCLUDES := -I $(CUDA_UTIL_DIR)/include \
	-I $(PR_STM_DIR)/include \
	-I $(CUDA_PATH)/include \
	-I $(CURR_DIR)/include \
	-I $(TINY_ROOT)/include \
	-I $(TSX_ROOT)/include \
	-I $(HETM_ROOT)/include \
	-I $(INPUT_HANDLER)/include \
	-I $(SHARED)/include \
	-I $(TINY_ROOT)/src
#
DEFINES  :=
#
LIBS     := -L $(CUDA_PATH)/lib64 -L $(CUDA_PATH)/compat -lcudart -lcuda \
	-L $(INPUT_HANDLER) -linput_handler \
	-L $(TINY_ROOT)/lib -lstm -lpthread \
	-L $(HETM_ROOT) -lhetm -L$(FVS_LIB_DIR) -l$(FVS_LIB) -lbsd \
	-L $(CUDA_UTIL_DIR) -lcuda-util
#
APP  = memcd
BENCH= memcd
# NVCC = nvcc
CC   = gcc
CXX  = g++

include $(CURR_DIR)/Makefile.defines

# the log only works with TM_STATISTICS3 on
DEFINES    += -DTM_STATISTICS3

### Add this if debug needed (GPU run much slower)
DEBUG_FLAGS  := -g -DHETM_DEB=1
NV_DEB_FLAGS := -G -g -DHETM_DEB=1

INCLUDES  += 	-I $(HTM_ARCH_DEPS)/include
ifeq ($(USE_TSX_IMPL),1)
INCLUDES  += -I $(HTM_SGL)/include
LIBS      += -L $(HTM_SGL)/bin -lhtm_sgl -L $(TSX_ROOT) -ltsxMOD
endif
ifneq ($(CFG),debug)
CFLAGS   += -DNDEBUG
CXXFLAGS += -DNDEBUG
NVFLAGS  += -DNDEBUG
endif
CFLAGS     := -c $(DEFINES) $(INCLUDES) -std=c99
CXXFLAGS   := -c $(DEFINES) $(INCLUDES) -std=c++11
NVFLAGS    := -c -dc $(DEFINES) $(INCLUDES) -std=c++11 
LDFLAGS    := $(LIBS)  -l tbb -l tbbmalloc

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

BENCH_BANK_SRC_DIR := $(SRC)
include Makefile.sources

ifeq ($(USE_TSX_IMPL),1)
TSX_FLAGS += -DUSE_TSX_IMPL
CFLAGS    += $(TSX_FLAGS) -mrtm
CXXFLAGS  += $(TSX_FLAGS) -mrtm
NVFLAGS   += $(TSX_FLAGS) --compiler-options='-mrtm'
endif

ifeq ($(CFG),debug)
CFLAGS   += $(DEBUG_FLAGS)
CXXFLAGS += $(DEBUG_FLAGS)
NVFLAGS  += $(NV_DEB_FLAGS)
endif

SRC        := $(CURR_DIR)/src

# $(SRC)/cmp_kernels.o
MEMCD_OBJS  := \
	$(SRC)/kernel.o \
	$(SRC)/cmp_kernels.o \
	$(SRC)/memcd2.o \
	$(SRC)/bankKernel.o \
	$(SHARED)/src/shared.o \
	$(SRC)/setupKernels.o \
	$(SRC)/memcdKernel.o \
#

OBJS := $(MEMCD_OBJS)

.PHONY:	all clean

all: $(APP)
	# Done


simple: $(OBJS) $(HETM_ROOT)/hetm-inst-cpu.o \
		$(HETM_ROOT)/hetm-inst-gpu.o \
		$(PR_STM_DIR)/pr-stm-obj.o
	# does not recompile everything
	# $(CXX) $(CXXFLAGS) -o $(SRC)/workloads/cpu_txs.o $(SRC)/workloads/cpu_txs.cpp
	#$(CXX) $(CXXFLAGS) -O1 -o $(SRC)/memcd2.o $(SRC)/memcd2.cpp
	$(NVCC) $(LDFLAGS) -o $(APP) 	$(HETM_ROOT)/hetm-inst-gpu.o $(HETM_ROOT)/hetm-inst-cpu.o $(OBJS)

# TODO: propagate PR_MAX_RWSET_SIZE and HETM_CMP_TYPE
$(APP): $(OBJS) \
$(CUDA_UTIL_DIR)/libcuda-util.a \
$(HETM_ROOT)/libhetm.a \
$(TINY_ROOT)/lib/libstm.a \
$(HTM_SGL)/bin/libhtm_sgl.a \
$(TSX_ROOT)/libtsxMOD.a \
$(CODE_HETM_OBJS) \
$(DEPS_PR_STM_OBJS) \
$(PR_STM_DIR)/pr-stm-obj.o \
$(HETM_ROOT)/hetm-inst-cpu.o \
$(HETM_ROOT)/hetm-inst-gpu.o
	@echo " ------ FLAGS ------ \n$(DEFINES)"
	@echo " ------------------- \nLinking..."
	### TODO: optimization breaks performance
	# $(CXX) $(CXXFLAGS) -o $(SRC)/workloads/cpu_txs.o $(SRC)/workloads/cpu_txs.cpp
	$(NVCC) $(LDFLAGS) -o $@ $(OBJS) \
		$(HETM_ROOT)/hetm-inst-cpu.o \
		$(HETM_ROOT)/hetm-inst-gpu.o \
		$(PR_STM_DIR)/pr-stm-obj.o
	@echo "Linking..."
	### TODO: optimization breaks performance
	#$(CXX) $(CXXFLAGS) -O1 -o $(SRC)/memcd2.o $(SRC)/memcd2.cpp
	#$(NVCC) $(LDFLAGS) -o $@ $(OBJS)


$(TINY_ROOT)/lib/libstm.a: $(TINY_MOD_SRCS)
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

$(TSX_ROOT)/libtsxMOD.a: $(TSX_MOD_SRCS)
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

$(HTM_SGL)/bin/libhtm_sgl.a: $(HTM_SGL_SRCS)
	@echo " ============================================ "
	@echo " === Building HTM alg lib "
	@echo " ============================================ "
	@bash $(HTM_SGL)/compile.sh


$(CODE_HETM_OBJS) \
$(HETM_ROOT)/libhetm.a \
$(HETM_ROOT)/hetm-inst-cpu.o \
$(HETM_ROOT)/hetm-inst-gpu.o: $(DEPS_PR_STM_OBJS) $(CODE_HTM_SRCS)
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
		BMAP_ENC_1BIT=$(BMAP_ENC_1BIT) \
		USE_UNIF_MEM=$(USE_UNIF_MEM) \
		PRINT_DEB=$(PRINT_DEB) \
		DISABLE_EARLY_VALIDATION=$(DISABLE_EARLY_VALIDATION) \
		HETM_NB_DEVICES=$(HETM_NB_DEVICES) \
		REDUCED_RS=$(REDUCED_RS) \
		DISABLE_RS=$(DISABLE_RS) \
		DISABLE_WS=$(DISABLE_WS)

$(DEPS_CUDA_UTIL_OBJS) $(CUDA_UTIL_DIR)/libcuda-util.a: $(DEPS_CUDA_UTIL_SRCS)
	@echo " ============================================ "
	@echo " === Building CUDA utils lib "
	@echo " ============================================ "
	$(MAKE) --directory $(CUDA_UTIL_DIR) clean
	$(MAKE) -j14 --directory $(CUDA_UTIL_DIR) \
		BMAP_ENC_1BIT=$(BMAP_ENC_1BIT) \
		CFG=$(CFG) 

$(DEPS_PR_STM_OBJS): $(DEPS_PR_STM_SRCS)
	@echo " ============================================ "
	@echo " === Building PR-STM lib "
	@echo " ============================================ "
	$(MAKE) --directory $(PR_STM_DIR) clean
	$(MAKE) -j14 --directory $(PR_STM_DIR) \
		CFG=$(CFG) 

alldeps:
	@echo " ============================================ "
	@echo " === Building CUDA utils lib "
	@echo " ============================================ "
	$(MAKE) --directory $(CUDA_UTIL_DIR) clean
	$(MAKE) -j14 --directory $(CUDA_UTIL_DIR) \
		BMAP_GRAN_BITS=$(BMAP_GRAN_BITS) \
		BMAP_ENC_1BIT=$(BMAP_ENC_1BIT) \
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
		BMAP_GRAN_BITS=$(BMAP_GRAN_BITS) \
		USE_UNIF_MEM=$(USE_UNIF_MEM) \
		PRINT_DEB=$(PRINT_DEB) \
		DISABLE_EARLY_VALIDATION=$(DISABLE_EARLY_VALIDATION) \
		BMAP_ENC_1BIT=$(BMAP_ENC_1BIT) \
		HETM_NB_DEVICES=$(HETM_NB_DEVICES) \
		REDUCED_RS=$(REDUCED_RS) \
		DISABLE_RS=$(DISABLE_RS) \
		DISABLE_WS=$(DISABLE_WS)
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
		BMAP_GRAN_BITS=$(BMAP_GRAN_BITS) \
		PRINT_DEB=$(PRINT_DEB) \
		LOG_SIZE=$(LOG_SIZE) \
		BMAP_ENC_1BIT=$(BMAP_ENC_1BIT) \
		DISABLE_EARLY_VALIDATION=$(DISABLE_EARLY_VALIDATION) \
		STM_LOG_BUFFER_SIZE=$(STM_LOG_BUFFER_SIZE) \
		HETM_NB_DEVICES=$(HETM_NB_DEVICES) \
		REDUCED_RS=$(REDUCED_RS) \
		DISABLE_RS=$(DISABLE_RS) \
		DISABLE_WS=$(DISABLE_WS)

%.o:	%.c
	@echo ""
	$(CC) $(CFLAGS) -o $@ $<

%.o:	%.cpp
	@echo ""
	$(CXX) $(CXXFLAGS) -o $@ $<

%.o:	%.cu
	@echo ""
	$(NVCC) $(NVFLAGS) -o $@ $<

zipf:
	g++ -I ./include ./src/test_zipf.cpp -o gen_zip -std=c++14

# ptx:
# 	$(NVCC) $(NVFLAGS) $(NV_DEB_FLAGS) --ptx cmp_kernels.cu

clean:
	$(MAKE) --directory $(HETM_ROOT) clean
	$(MAKE) --directory $(HETM_ROOT) clean
	rm -f $(APP) *.o $(SRC)/*.o

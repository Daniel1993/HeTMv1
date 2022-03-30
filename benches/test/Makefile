### TODO: create include files for pr-stm, cuda-utils, tsx and tiny

CURR_DIR      ?= ~/projs/HeTM_V1/test
TINY_ROOT     ?= $(CURR_DIR)/../tinyMOD
TSX_ROOT      ?= $(CURR_DIR)/../tsxMOD
HETM_ROOT     ?= $(CURR_DIR)/../hetm
HTM_SGL       ?= ~/projs/htm_alg
PR_STM_DIR    ?= ~/projs/pr-stm
CUDA_PATH     ?= /usr/local/cuda

CUDA_UTIL_DIR ?= ~/projs/cuda-utils
HTM_ARCH_DEPS ?= ~/projs/arch_dep

CFG           ?= prod

INCLUDES := -I $(CUDA_UTIL_DIR)/include \
	-I $(PR_STM_DIR)/include \
	-I $(CUDA_PATH)/include \
	-I $(CURR_DIR)/include \
	-I $(TINY_ROOT)/include \
	-I $(TSX_ROOT)/include \
	-I $(HETM_ROOT)/include \
	-I $(TINY_ROOT)/src
#
DEFINES  :=
#
LIBS     := -L $(CUDA_PATH)/lib64 -lcudart -lcuda \
	-L $(TINY_ROOT)/lib -lstm -lpthread \
	-L $(HETM_ROOT) -lhetm \
	-L $(CUDA_UTIL_DIR) -lcuda-util
#
APP  = test
BENCH= test
NVCC = nvcc
CC   = gcc
CXX  = g++

include $(CURR_DIR)/Makefile.defines

# the log only works with TM_STATISTICS3 on
DEFINES    += -DTM_STATISTICS3

### Add this if debug needed (GPU run much slower)
DEBUG_FLAGS  := -g -lineinfo -DHETM_DEB=1
NV_DEB_FLAGS := -G -g -lineinfo -DHETM_DEB=1

INCLUDES  += 	-I $(HTM_ARCH_DEPS)/include
ifeq ($(USE_TSX_IMPL),1)
INCLUDES  += -I $(HTM_SGL)/include
LIBS      += -L $(HTM_SGL)/bin -lhtm_sgl -L $(TSX_ROOT) -ltsxMOD
endif

CFLAGS     := -c $(DEFINES) $(INCLUDES) -std=c99
CXXFLAGS   := -c $(DEFINES) $(INCLUDES) -std=c++11
NVFLAGS    := -c $(DEFINES) $(INCLUDES) -std=c++11 \
	--default-stream per-thread \
	-arch sm_50
LDFLAGS    := $(LIBS) \
--default-stream per-thread \
-arch sm_50

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

OBJS  := \
	$(SRC)/testKernel.o \
	$(SRC)/test.o

.PHONY:	all clean

all: $(APP)
	# Done

# TODO: propagate PR_MAX_RWSET_SIZE and HETM_CMP_TYPE
$(APP): $(OBJS)
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
		DISABLE_NON_BLOCKING=$(DISABLE_NON_BLOCKING) \
		PRINT_DEB=$(PRINT_DEB)
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
		PRINT_DEB=$(PRINT_DEB)
	@echo "Linking..."
	### TODO: optimization breaks performance
	# $(CXX) $(CXXFLAGS) -O2 -o $(SRC)/memcd2.o $(SRC)/memcd2.cpp
	$(NVCC) $(LDFLAGS) -o $@ $(OBJS)

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

ptx:
	$(NVCC) $(NVFLAGS) $(NV_DEB_FLAGS) --ptx cmp_kernels.cu

clean:
	$(MAKE) --directory $(HETM_ROOT) clean
	$(MAKE) --directory $(HETM_ROOT) clean
	rm -f $(APP) *.o $(SRC)/*.o

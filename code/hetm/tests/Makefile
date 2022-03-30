### CONFIG
CURR_DIR      ?= ~/projs/HeTM_V1/code/hetm/tests
HETM_ROOT     ?= $(CURR_DIR)/..
CUDA_UTIL_DIR ?= $(HETM_ROOT)/../../deps/cuda-utils
PR_STM_DIR    ?= $(HETM_ROOT)/../../deps/pr-stm
TINY_ROOT     ?= $(HETM_ROOT)/../tinyMOD
TSX_ROOT      ?= $(HETM_ROOT)/../tsxMOD
HTM_SGL       ?= $(HETM_ROOT)/../../deps/htm_alg
CUDA_PATH     ?= /usr/local/cuda

PROFILE ?= 0

INCLUDES := \
	-I ../include -I $(CUDA_UTIL_DIR)/include \
	-I $(HETM_ROOT)/include \
	-I $(TINY_ROOT)/include \
	-I $(TSX_ROOT)/include \
	-I $(CUDA_PATH)/include \
	-I $(PR_STM_DIR)/include
DEFINES  :=
LIBS     := \
	-L $(HETM_ROOT) -lhetm \
	-L $(CUDA_UTIL_DIR) -lcuda-util \
	-L $(TINY_ROOT)/lib -lstm \
	-L $(HTM_SGL)/bin -lhtm_sgl -L $(TSX_ROOT) -ltsxMOD \
	-L $(CUDA_PATH)/lib64 -lcudart -lcuda

CFLAGS   := -g -c -O0 -std=gnu99
CXXFLAGS := -g -c -O0 -std=c++11
NVFLAGS  := -g -G -c -std=c++11
LDFLAGS  := -lcppunit $(LIBS)
NVCC     := nvcc

OBJS     := pr-stm-impl.o # needed

OBJS_test_cpu_gpu := test_cpu_gpu.o

ifeq ($(PROFILE),1)
DEFINES += -DUSE_NVTX
endif

CFLAGS   += $(INCLUDES) $(DEFINES)
CXXFLAGS += $(INCLUDES) $(DEFINES)
NVFLAGS  += $(INCLUDES) $(DEFINES)
LDFLAGS  += $(LIBS)

### RULES

all: $(RUNNER_B)

%.o: %.cu
	$(NVCC) $(NVFLAGS) -o $@ $^

%.o: %.c
	$(CC) $(CFLAGS) -o $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^

$(RUNNER_B): $(OBJS)
	$(NVCC) -o $@ $^ $(LDFLAGS)

clean:
	rm -f $(RUNNER_B) $(OBJS)

TINYSTM := ../tinystm
HTM_SGL       ?= ~/projs/htm_alg
HTM_ARCH_DEPS ?= ~/projs/arch_dep
TBB := ../tbb-2019_U8

HeTM_DIR      ?= ~/projs/HeTM_V1
HeTM_ROOT     ?= $(HeTM_DIR)/hetm
TINY_ROOT     ?= $(HeTM_DIR)/tinyMOD
TSX_ROOT      ?= $(HeTM_DIR)/tsxMOD
PR_STM_DIR    ?= ~/projs/pr-stm
CUDA_PATH     ?= /usr/local/cuda

CUDA_UTIL_DIR ?= ~/projs/cuda-utils

BINS = bank

TMLIB := $(TINYSTM)/lib/libstm.a

INCS := -I $(TINYSTM)/include -I $(TBB)/include/ -I $(CUDA_UTIL_DIR)/include \
	-I $(PR_STM_DIR)/include \
	-I $(CUDA_PATH)/include \
	-I $(HeTM_ROOT)/include \
	-I $(TINY_ROOT)/include \
	-I $(TSX_ROOT)/include
LIBS := -L $(TBB)/build/lib -l tbb -l tbbmalloc \
	-L $(CUDA_PATH)/lib64 -lcudart -lcuda -lnuma \
	-L $(TINY_ROOT)/lib -lstm \
	-L $(HeTM_ROOT) -lhetm \
	-L $(CUDA_UTIL_DIR) -lcuda-util \
	-l pthread

DEFINES := $(INCS) -DCPU_FREQ=$(CPU_FREQ) -DUSE_WORKLOAD=$(USE_WORKLOAD) -O2
LDFLAGS := $(LIBS)

ifeq ($(USE_HTM),1)
LDFLAGS += -L $(TSX_ROOT) -l tsxMOD -L $(HTM_SGL)/bin -l htm_sgl
DEFINES += -I $(HTM_ARCH_DEPS)/include -I $(HTM_SGL)/include -DUSE_HTM -mrtm
endif

.PHONY:	all clean

all:	$(BINS)

%.o:	%.c
	$(CC) $(CFLAGS) $(DEFINES) -c -o $@ $<

%.o:	%.cpp
	$(CXX) $(CPPFLAGS) $(DEFINES) -c -o $@ $<

$(BINS):	%:	%.o $(TMLIB)
	$(CXX) -o $@ $< $(LDFLAGS)

clean:
	rm -f $(BINS) *.o

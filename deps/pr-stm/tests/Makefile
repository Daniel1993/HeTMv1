### CONFIG
CUDA_UTIL_DIR ?= ~/projs/cuda-utils

INCLUDES := -I ../include -I $(CUDA_UTIL_DIR)/include
DEFINES  :=
LIBS     :=

CFLAGS   := -g -c -O0 -std=gnu99
CXXFLAGS := -g -c -O0 -std=c++11
NVFLAGS  := -G -g -c -std=c++11
LDFLAGS  := -lcppunit
NVCC     := nvcc

SRCS     := $(shell ls -rt -d -1 ../src/*.c ../src/*.cpp ../src/*.cu 2>/dev/null)

TEST_SRC := $(shell ls -rt -d -1 ./Test*.c ./Test*.cpp ./Test*.cu 2>/dev/null)
RUNNER   := ./runner.cpp
RUNNER_B := ./Runner
SRCS     += $(RUNNER)
OBJS     := $(addsuffix .o, $(basename $(SRCS)))
OBJS     += $(addsuffix .o, $(basename $(TEST_SRC)))

CFLAGS   += $(INCLUDES) $(DEFINES)
CXXFLAGS += $(INCLUDES) $(DEFINES)
NVFLAGS  += $(INCLUDES) $(DEFINES)
LDFLAGS  += $(LIBS)

### RULES

all: $(RUNNER_B)

%.o: %.cu
	$(NVCC) $(NVFLAGS) -o $@ $^

$(RUNNER_B): $(OBJS)
	$(NVCC) -o $@ $^ $(LDFLAGS)

clean:
	rm -f $(RUNNER_B) $(OBJS)

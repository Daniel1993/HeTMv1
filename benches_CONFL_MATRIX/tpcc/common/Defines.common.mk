# ==============================================================================
#
# Defines.common.mk
#
# ==============================================================================


CC       := gcc
CFLAGS   += -g -pthread
CFLAGS   += #-O2 
CFLAGS   += -I$(LIB)
CPP      := g++
CPPFLAGS += $(CFLAGS) -std=c++11
LD       := g++
LIBS     += -lpthread

SRC_DIR := ./src
CODE_DIR := ./code

SRCS += \
	$(CODE_DIR)/memory.cc \
	$(CODE_DIR)/pair.cc \
	$(CODE_DIR)/list.cc \
	$(CODE_DIR)/hashtable.cc \
	$(CODE_DIR)/tpcc.cc \
	$(CODE_DIR)/tpccclient.cc \
	$(CODE_DIR)/tpccgenerator.cc \
	$(CODE_DIR)/tpcctables.cc \
	$(CODE_DIR)/tpccdb.cc \
	$(CODE_DIR)/clock.cc \
	$(CODE_DIR)/randomgenerator.cc \
	$(CODE_DIR)/stupidunit.cc \
	$(SRC_DIR)/mt19937ar.c \
	$(SRC_DIR)/random.c\
	$(SRC_DIR)/thread.c
#
OBJS     := ${SRCS:.cc=.o}
OBJS     := ${OBJS:.c=.o}
OBJS     := ${OBJS:.cpp=.o}

# Remove these files when doing clean
OUTPUT +=

# LIB := ../lib
STM := ../backend/tinySTM
# STM := ../backend/graphTM



# ==============================================================================
#
# End of Defines.common.mk
#
# ==============================================================================

INCLUDES := -I .
#
DEFINES  :=
#
LIBS     := -lpthread -lm
#
CC       := gcc
CXX      := g++
AR       := ar rcs

INCLUDES += -I $(BB_LIB) -I $(SA_LIB)

### Add this if debug needed (GPU run much slower)
DEBUG_FLAGS  := -g

# this is needed for the barrier implementation
ifeq ($(DEBUG),1)
DEFINES += $(DEBUG_FLAGS)
endif

CFLAGS     := -c $(DEFINES) $(INCLUDES) -std=gnu99
CXXFLAGS   := -c $(DEFINES) $(INCLUDES) -std=c++11
NVFLAGS    := -c $(DEFINES) $(INCLUDES) -std=c++11
LDFLAGS    := $(LIBS)

LIB        := prodCons

OBJS       := \
	prod-cons.o \
	threading.o

.PHONY:	all clean

all: lib$(LIB).a main
	# Done

main: lib$(LIB).a main.o $(OBJS)
	$(CC) -o main main.o $(OBJS) $(LIBS)

lib$(LIB).a: $(OBJS)
	@echo "Linking..."
	$(AR) $@ $(OBJS)

%.o:	%.c
	@echo ""
	$(CC) $(CFLAGS) -o $@ $<

%.o:	%.cpp
	@echo ""
	$(CXX) $(CXXFLAGS) -o $@ $<

run:
	make all

clean:
	rm -f lib$(LIB).a *.o

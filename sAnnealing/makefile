#
# This is a multipurpose makefile
#
# Procedures:
# To run Valgrind, start GUD in emacs and then press F9. The yes file will put the GUD in the first problem
# IMPORTANT: splint main.c
# To pack the project use the pack-F2 dir, with symlinks and the archiver, for both procedures there are nautilus scripts.
# Profiling: gprof project
# kprof is graphical gprof: kprof
# Anotated code : gprof -l -A -x project
# Better anotated code : make coverage (then just less the .c.cov)
# WARNING just use kcachegrind it is good:
#                    valgrind --tool=callgrind -v ./project && kcachegrind callgrind.out.4330
# Memory alignment stuff
# pahole project
# valgrind --tool=cachegrind -v ./project
DEBUG ?= 0
######### IMPORTANT DO NOT COMMENT THESE FLAGS

# The compiler to use
CC        = gcc
AR        = ar rcs
# The name of the executable to produce
EXEC      = project
# Ansi flags, make the compiler more informative
ANSI      = -ansi -pedantic -Wall -Wextra -std=c99
# Debug flags for gdb/GUD
DEBUG_F   = -O0 -gdwarf-2 -g3 -ggdb
#DEBUG     = -O0 -gdwarf-2
# Profile flags for gprof
PROFILE   = -pg -fprofile-arcs -ftest-coverage
# Optimization flags for production
### TODO: NDEBUG is really needed there is a lot of checking going on with NDEBUG off
OPTIMIZE  = -O9 #-D NDEBUG
# Static for static StandAlones
STATIC = -static
# PThreads are usefull
THREADS = -pthread

#########  COMMENT THESE

# Uncomment the flags that you need in a particular compile
# Always use ANSI
CFLAGS   = $(ANSI)
# CFLAGS  += $(DEBUG_F)
CFLAGS  += -D_DEFAULT_SOURCE
# CFLAGS  += -D NDEBUG
# Use this flag to generate an animation on the log file
#
# watch with
# cat log - | gs -dNOPROMPT -g680x680 -dGraphicsAlphaBits=2 -r144 -q
# CFLAGS  += -D POSTSCRIPT
# CFLAGS  += $(PROFILE)
# CFLAGS  += $(OPTIMIZE)
#CFLAGS  += $(STATIC)
CFLAGS  += $(THREADS)

ifeq ($(CHECK_ONLY),1)
CFLAGS += -DCHECK_ONLY
endif

ifeq ($(DEBUG),1)
CFLAGS  += $(DEBUG_F)
else
CFLAGS  += $(OPTIMIZE)
CFLAGS  += -D NDEBUG
endif

##  Base Makefile
##  Diferent classes of files
CS    = $(wildcard *.c)
HS    = $(wildcard *.h)
IS    = $(addsuffix .i, $(basename $(CS)))
SS    = $(addsuffix .s, $(basename $(IS)))
OS    = $(addsuffix .o, $(basename $(SS)))
HOLES = $(addsuffix .hls, $(basename $(SS)))
COVS  = $(addsuffix .c.gcov, $(basename $(CS)))
STATS = $(addsuffix .gcda, $(basename $(CS))) $(addsuffix .gcno, $(basename $(CS)))
CUR_SOL = $(shell basename $(CURDIR))
OBJ_LIB ?= sa_lib.o
OBJ_LIB_ROLLBACK ?= sa_lib_ROLLBACK.o
OBJ_STATE ?= sa_state.o
OBJ_STATE_NOINIT ?= sa_state_NO_INIT.o

ifeq ($(CUR_SOL),SA)
OBJ_LIB = sa_lib.o
OBJ_LIB_ROLLBACK = sa_lib_ROLLBACK.o
OBJ_STATE = sa_state.o
OBJ_STATE_NOINIT = sa_state_NO_INIT.o
endif

ifeq ($(CUR_SOL),BB)
OBJ_LIB = bb_lib.o
OBJ_LIB_ROLLBACK = bb_lib_ROLLBACK.o
OBJ_STATE = bb_state.o
OBJ_STATE_NOINIT = bb_state_NO_INIT.o
endif

ifeq ($(CUR_SOL),hybrid)
OBJ_LIB = sa_lib.o bb_lib.o hy_lib.o
OBJ_LIB_ROLLBACK = sa_lib_ROLLBACK.o bb_lib_ROLLBACK.o hy_lib_ROLLBACK.o
OBJ_STATE = bb_state.o sa_state.o
OBJ_STATE_NOINIT = bb_state_NO_INIT.o sa_state_NO_INIT.o
endif

SA_STRUCTS_OS = \
graph.o \
darray.o \
queue.o \
reorganize.o \
splayTree.o

SA_OS = \
$(OBJ_STATE) \
$(OBJ_LIB) \
simulatedAnnealing.o

GREEDY_OS = \
$(OBJ_STATE) \
$(OBJ_LIB) \
simulatedAnnealing_GREEDY.o

ROLLBACK_OS = \
$(OBJ_STATE) \
$(OBJ_LIB_ROLLBACK) \
simulatedAnnealing.o

NO_INIT_OS = \
$(OBJ_STATE_NOINIT) \
$(OBJ_LIB) \
simulatedAnnealing.o

GREEDY_NO_INIT_OS = \
$(OBJ_STATE_NOINIT) \
$(OBJ_LIB) \
simulatedAnnealing_GREEDY.o

# Phony targets
.PHONY: all coverage clean dotclean tags depend dox check

# Default Compile
all: $(SS) $(EXEC) $(HOLES) project_GREEDY project_NO_INIT project_GREEDY_NO_INIT project_ROLLBACK

libbb: libbb.a

libsa: libsa.a

coverage: $(COVS)

## Linking rule
$(EXEC): $(OS)
	@echo Linking: $@
	$(CC) $(CFLAGS) $(OS) -o $@ -lm -lrt -lbsd

## Linking rule
libbb.a: bb_lib.o bb_state.o graph.o
	@echo Linking: $@
	$(AR) $@ $^ 

libgsort.a: $(OS)
	@echo Linking: $@
	$(AR) $@ $^ 
	cp libgsort.a libfvsgsort.a

## Linking rule
libsa.a: $(SA_STRUCTS_OS) $(SA_OS)
	@echo Linking: $@
	$(AR) $@ $^ 

project_GREEDY: main.o  $(SA_STRUCTS_OS) $(GREEDY_OS)
	$(CC) $(CFLAGS) -o $@ $^ -lm -lrt -lbsd

project_ROLLBACK: main.o $(SA_STRUCTS_OS) $(ROLLBACK_OS)
	$(CC) $(CFLAGS) -o $@ $^ -lm -lrt -lbsd

project_NO_INIT: main.o $(SA_STRUCTS_OS) $(NO_INIT_OS)
	$(CC) $(CFLAGS) -o $@ $^ -lm -lrt -lbsd

project_GREEDY_NO_INIT: main.o $(SA_STRUCTS_OS) $(GREEDY_NO_INIT_OS)
	$(CC) $(CFLAGS) -o $@ $^ -lm -lrt -lbsd

project_CHECK_ONLY: main.o bb_lib_CHECK_ONLY.o bb_state.o graph.o
	$(CC) $(CFLAGS) -o $@ $^ -lm -lrt -lbsd

## Run Preprocessor
%.i: %.c
	@echo PreProcess: $<
	$(CC) $(CFLAGS) -E -o $@ $<

%_CHECK_ONLY.i: %.c
	@echo PreProcess: $<
	$(CC) $(THREADS) $(DEBUG_F) -DCHECK_ONLY -E -o $@ $<

%_ROLLBACK.i: %.c
	@echo PreProcess: $<
	$(CC) $(CFLAGS) -DUSE_ROLLBACK -E -o $@ $<

simulatedAnnealing_GREEDY.i: simulatedAnnealing.c
	@echo PreProcess: $<
	$(CC) $(CFLAGS) -DUSE_GREEDY -E -o $@ $<

sa_state_NO_INIT.i: sa_state.c
	@echo PreProcess: $<
	$(CC) $(CFLAGS) -DNUSE_INIT_APPROX -E -o $@ $<

bb_state_NO_INIT.i: bb_state.c
	@echo PreProcess: $<
	$(CC) $(CFLAGS) -DNUSE_INIT_APPROX -E -o $@ $<

hy_lib.o : sa_lib.o bb_lib.o
hy_lib_ROLLBACK.o : sa_lib_ROLLBACK.o bb_lib_ROLLBACK.o

## Generate Assembly
%.s: %.i
	@echo Assemble: $<
	$(CC) $(CFLAGS) -S -o $@ $<

## Build Object
%.o: %.s
	@echo Build Object from: $<
	$(CC) $(CFLAGS) -c -o $@ $<

## Find Holes
%.hls: %.o
	@echo Holes in: $<
	touch $@
	# pahole -I -H 1 $< | sed 's/.* \(.\+:[0-9]\+\) .*/\1:1:/'

## Pack using directory links
%.tgz: $(CS) $(HS) Makefile
	@echo Pack Using directory
	tar -chvzf $@ $(basename $@ .tgz)

## Coverage statistics
%.c.gcov: %.gcda %.gcno
	@echo Anotated coverage source $<
	gcov -a -b -f $<
	dos2unix $@

## Make dependencies
depend: depend.mak

## Make dependencies forcing creation of all files
depend.mak: $(CS) $(HS)
	@echo Making dependencies ...
	$(CC) -MM $(CS) | sed 's/\.o/\.i/g' > depend.mak

-include depend.mak

## Clean up
clean:
	@echo Cleaning Up
	rm -f libbb.a $(EXEC) $(STATS) $(COVS) $(IS) $(SS) $(OS) $(HOLES) $(OBJ_LIB) $(OBJ_LIB_ROLLBACK) $(OBJ_STATE) $(OBJ_STATE_NOINIT) *_GREEDY.o *_NO_INIT.o  *_ROLLBACK.o *_GREEDY.i *_NO_INIT.i *_ROLLBACK.i gmon.out depend.mak *~ callgrind.out.* cachegrind.out.*

dotclean:
	@echo Cleaning Up
	rm *.dot *.pdf

## TAGS
tags: TAGS
TAGS: $(CS) $(HS)
	@echo Making TAGS
	etags $(CS) $(HS)

## doc, make documentation with doxymacs
dox: $(CS) $(HS) Doxyfile
	doxygen

## run splint for checking code
check: $(CS) $(HS)
	splint $<

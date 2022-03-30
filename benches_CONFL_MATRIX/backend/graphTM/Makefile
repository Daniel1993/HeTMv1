CC       := gcc -c
AR       := ar rcs
CFLAGS   := -std=gnu11 -g -Wall -pthread # -DNDEBUG=1 -mcpu=power8 -mtune=power8 
CFLAGS   += #-O3 # TODO
#CFLAGS   += -I $(LIB)
CPP      := g++
CPPFLAGS := $(CFLAGS)
LD       := g++
LIBS     += -lpthread

# Remove these files when doing clean
OUTPUT +=

LIB := ../lib

all: libstm.a test

libstm.a: stm.o #../../bench/lib/thread.o
	$(AR) $@ $^

test: libstm.a test.o
	$(LD) test.o -L . -l stm -l pthread -o $@

%.o: %.c
	$(CC) $(CFLAGS) $^ -o $@

clean:
	rm -f libstm.a stm.o

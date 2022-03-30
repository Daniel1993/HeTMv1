ROOT = ../..
HTM_SGL       ?= ~/projs/htm_alg
HTM_ARCH_DEPS ?= ~/projs/arch_dep

include $(ROOT)/Makefile.common

USE_TSX_IMPL ?= 0
ifeq ($(USE_TSX_IMPL),1)
DEFINES += -DUSE_TSX_IMPL
CFLAGS  += -mrtm -I $(HTM_ARCH_DEPS)/include -I $(HTM_SGL)/include
LDFLAGS += -L $(HTM_SGL)/bin -lhtm_sgl
endif

BINS = bank

.PHONY:	all clean

all:	$(BINS)

%.o:	%.c
	$(CC) $(CPPFLAGS) $(CFLAGS) $(DEFINES) -c -o $@ $<

$(BINS):	%:	%.o $(TMLIB)
	$(CXX) -o $@ $< $(LDFLAGS)

clean:
	rm -f $(BINS) *.o

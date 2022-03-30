LIB      := libinput_handler.a
OBJS     := src/input_handler.o

CXX      := g++
CXXFLAGS := -I include -std=c++11
CFLAGS   := -I include
AR       := ar rcs

all: $(LIB)

$(LIB): $(OBJS)
	$(AR) $(LIB) $(OBJS)

test: test.o
	$(CXX) $(CXXFLAGS) -o test test.o -L . -linput_handler
	./test ARG1=VAL1 ARG2=VAL2 ARG3=VAL3

clean:
	rm -f $(OBJS) $(LIB)

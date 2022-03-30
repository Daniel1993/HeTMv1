DEFINES :=
INCLUDES := -I ../include

CFLAGS     := -c $(DEFINES) $(INCLUDES) -std=c99
CXXFLAGS   := -c $(DEFINES) $(INCLUDES) -std=c++11

OBJS  := \
	main.o \
	../src/zipf_dist.o

all: $(OBJS)
	javac Main.java
	$(CXX) $(CCFLAGS) -o main_c $(OBJS)

%.o:	%.c
	@echo ""
	$(CC) $(CFLAGS) -o $@ $<

%.o:	%.cpp
	@echo ""
	$(CXX) $(CXXFLAGS) -o $@ $<

clean:
	rm -f *.class

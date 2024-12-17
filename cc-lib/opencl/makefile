default : clutil_test.exe

UNAME := $(shell uname -o)
ifeq ($(UNAME),Cygwin)
  # for 64 bits on windows (mingw).
  CXX=x86_64-w64-mingw32-g++
  CC=x86_64-w64-mingw32-g++

  # Note, this builds against AMD's distribution of OpenCL, since that's
  # the card I had when I started. It also works fine with an NVIDIA GPU,
  # and nvidia libs are a bit hard to find since they love CUDA.
  AMD_ARCH=x86_64
  # AMDSDK="/c/Program Files (x86)/AMD APP SDK"
  # mingw wants /c/, but TDM likes c:/
  # AMDSDK=c:/Program Files (x86)/AMD APP SDK/3.0
  AMDSDK=c:/code/AMD_SDK/
  # XXX necessary?
  PLATFORM_CXXFLAGS=-D__MINGW32__
  PLATFORM_LFLAGS=-lpsapi -Wl,--subsystem,console -static
  CL_CFLAGS="-I$(AMDSDK)/include"
  CL_LFLAGS='-L${AMDSDK}/lib/${AMD_ARCH}' -lOpenCL
else ifeq ($(UNAME),Darwin)

  CXX=g++
  CC=gcc

  # Check if g++ is actually clang
  ifeq ($(shell $(CXX) -v 2>&1 | grep "clang version" | wc -l | tr -cd '0-9'), 1)
    CLANG=true
  endif

  # XXX: Not tested
  CL_LFLAGS-lOpenCL

else
# Linux
  CXX=g++
  CC=gcc

  # Check if g++ is actually clang
  ifeq ($(shell $(CXX) -v 2>&1 | grep "clang version" | wc -l), 1)
    CLANG=true
  endif

  CL_LFLAGS=-lOpenCL
endif

CC_LIB=../

OPT=-O3

# ASAN=-g -fsanitize=address
# OPT=-O

CXXFLAGS=$(ASAN) $(OPT) $(PLATFORM_CFLAGS) -march=native -m64 -Wall -Wno-format -Wno-unused-function -Wno-deprecated -Wno-sign-compare -I. -I$(CC_LIB) --std=c++23
LFLAGS=$(ASAN) $(PLATFORM_CFLAGS) $(PLATFORM_LDFLAGS) $(CL_LFLAGS)

CC_LIB_OBJECTS=$(CC_LIB)/util.o $(CC_LIB)/base/stringprintf.o $(CC_LIB)/base/logging.o $(CC_LIB)/ansi.o

OBJECTS=$(CC_LIB_OBJECTS) clutil.o

$(CC_LIB)/%.o : $(CC_LIB)/%.cc makefile
	@$(CXX) $(CXXFLAGS) -c $< -o $@
	@echo -n "."

%.o : %.cc *.h makefile
	@$(CXX) $(CXXFLAGS) -c $< -o $@
	@echo -n "."

clutil_test.exe : clutil_test.o $(OBJECTS)
	@$(CXX) $^ -o $@ $(LFLAGS)
	@echo -n "!"


clean :
	rm -f *.o *.exe $(OBJECTS)

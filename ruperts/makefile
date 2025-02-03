default : ruperts.exe tosvg.exe gpuperts.exe polyhedra_test.exe scoreboard.exe noboard.exe query.exe visualize.exe tostl.exe noperts.exe imperts.exe validate.exe ratperts.exe big-polyhedra_test.exe

# This assumes OpenCL is installed as a system library, which
# is now the case for msys2 clang64. If it's not, you could
# see how I used to do this for the mingw toolchain in
# ../cc-lib/opencl/makefile.

CL_LFLAGS=-lOpenCL

# This is optional; if you want to use plain C++ code, just
# comment it out. But GMP is much faster.
GMP_LFLAGS=-lgmp
GMP_CFLAGS=-DBIG_USE_GMP

CC_LIB=../cc-lib
CODEC=../codec
SQLITE=../sqlite

OPT=-O3 -flto=thin

# ASAN=-g -fsanitize=address -fsanitize=undefined
# OPT=-O

CXXFLAGS=$(ASAN) $(OPT) $(GMP_CFLAGS) -march=native -Wall -Wno-psabi -Wno-format -Wno-unused-function -Wno-deprecated -Wno-sign-compare -I. -I$(CC_LIB) -I$(CODEC) -I$(SQLITE) --std=c++23
LFLAGS=$(ASAN) $(GMP_LFLAGS) -flto -lz

CC_LIB_OBJECTS=$(CC_LIB)/util.o $(CC_LIB)/arcfour.o $(CC_LIB)/base/stringprintf.o $(CC_LIB)/base/logging.o $(CC_LIB)/stb_image.o $(CC_LIB)/stb_image_write.o $(CC_LIB)/stb_truetype.o $(CC_LIB)/color-util.o $(CC_LIB)/image.o $(CC_LIB)/ansi.o  $(CC_LIB)/atomic-util.o $(CC_LIB)/status-bar.o $(CC_LIB)/zip.o $(CC_LIB)/miniz.o $(CC_LIB)/ansi-image.o $(CC_LIB)/image-resize.o $(CC_LIB)/opt/opt.o $(CC_LIB)/bounds.o $(CC_LIB)/interval-cover.o $(CC_LIB)/interval-cover-util.o $(CC_LIB)/bignum/big.o $(CC_LIB)/bignum/bigz.o $(CC_LIB)/bignum/bign.o $(CC_LIB)/bignum/bigq.o $(CC_LIB)/bignum/big-numbers.o $(CC_LIB)/bignum/polynomial.o

# This is only for generating animations of polyhedra; it should be pretty
# easy to excise the code if you're having trouble with it.
CODEC_OBJECTS=$(CODEC)/mov.o $(CODEC)/mov-recorder.o
CL_OBJECTS=$(CC_LIB)/opencl/clutil.o
SQLITE_OBJECTS=$(SQLITE)/database.o $(SQLITE)/sqlite3.o

OBJECTS=$(CC_LIB_OBJECTS) $(CODEC_OBJECTS) $(SQLITE_OBJECTS) polyhedra.o rendering.o solutions.o hull.o hull3d.o big-polyhedra.o symmetry-groups.o

$(CC_LIB)/%.o : $(CC_LIB)/%.cc makefile
	@$(CXX) $(CXXFLAGS) -c $< -o $@
	@echo -n "."

$(CODEC)/%.o : $(CODEC)/%.cc makefile
	@$(CXX) $(CXXFLAGS) -c $< -o $@
	@echo -n "."

%.o : %.cc *.h makefile
	@$(CXX) $(CXXFLAGS) -c $< -o $@
	@echo -n "."

ruperts.exe : ruperts.o $(OBJECTS)
	@$(CXX) $^ -o $@ $(LFLAGS)
	@echo -n "!"

gpuperts.exe : gpuperts.o $(OBJECTS) $(CL_OBJECTS)
	@$(CXX) $^ -o $@ $(LFLAGS) $(CL_LFLAGS)
	@echo -n "!"

topng.exe : topng.o $(OBJECTS)
	@$(CXX) $^ -o $@ $(LFLAGS)
	@echo -n "!"

tostl.exe : tostl.o $(OBJECTS)
	@$(CXX) $^ -o $@ $(LFLAGS)
	@echo -n "!"

polyhedra_test.exe : polyhedra_test.o $(OBJECTS)
	@$(CXX) $^ -o $@ $(LFLAGS)
	@echo -n "!"

polyhedra_bench.exe : polyhedra_bench.o $(OBJECTS)
	@$(CXX) $^ -o $@ $(LFLAGS)
	@echo -n "!"

scoreboard.exe : scoreboard.o $(OBJECTS)
	@$(CXX) $^ -o $@ $(LFLAGS)
	@echo -n "!"

noboard.exe : noboard.o $(OBJECTS)
	@$(CXX) $^ -o $@ $(LFLAGS)
	@echo -n "!"

visualize.exe : visualize.o $(OBJECTS)
	@$(CXX) $^ -o $@ $(LFLAGS)
	@echo -n "!"

query.exe : query.o $(OBJECTS)
	@$(CXX) $^ -o $@ $(LFLAGS)
	@echo -n "!"

noperts.exe : noperts.o $(OBJECTS)
	@$(CXX) $^ -o $@ $(LFLAGS)
	@echo -n "!"

imperts.exe : imperts.o $(OBJECTS)
	@$(CXX) $^ -o $@ $(LFLAGS)
	@echo -n "!"

hull_test.exe : hull_test.o hull.o hull3d.o $(OBJECTS)
	@$(CXX) $^ -o $@ $(LFLAGS)
	@echo -n "!"

validate.exe : validate.o $(OBJECTS)
	@$(CXX) $^ -o $@ $(LFLAGS)
	@echo -n "!"

ratperts.exe : ratperts.o $(OBJECTS)
	@$(CXX) $^ -o $@ $(LFLAGS)
	@echo -n "!"

big-polyhedra_test.exe : big-polyhedra_test.o $(OBJECTS)
	@$(CXX) $^ -o $@ $(LFLAGS)
	@echo -n "!"

zuperts.exe : zuperts.o $(OBJECTS)
	@$(CXX) $^ -o $@ $(LFLAGS)
	@echo -n "!"

symmetry-groups_test.exe : symmetry-groups_test.o $(OBJECTS)
	@$(CXX) $^ -o $@ $(LFLAGS)
	@echo -n "!"

clean :
	rm -f *.o *.exe $(OBJECTS)

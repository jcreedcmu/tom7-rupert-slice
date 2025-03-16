
default : ruperts.pdf table.exe

SQLITE=../../SQLITE
CC_LIB=../../cc-lib

# requires building bovex.exe
BOVEX=../../rephrase

CXXFLAGS=$(ASAN) $(OPT) $(GMP_CFLAGS) -march=native -Wall -Wno-psabi -Wno-format -Wno-unused-function -Wno-deprecated -Wno-sign-compare -I. -I.. -I$(CC_LIB) -I$(SQLITE) --std=c++23
LFLAGS=$(ASAN) $(GMP_LFLAGS) -flto -lz

CC_LIB_OBJECTS=$(CC_LIB)/util.o $(CC_LIB)/arcfour.o $(CC_LIB)/base/stringprintf.o $(CC_LIB)/base/logging.o $(CC_LIB)/stb_image.o $(CC_LIB)/stb_image_write.o $(CC_LIB)/stb_truetype.o $(CC_LIB)/color-util.o $(CC_LIB)/image.o $(CC_LIB)/ansi.o  $(CC_LIB)/atomic-util.o $(CC_LIB)/status-bar.o $(CC_LIB)/zip.o $(CC_LIB)/miniz.o $(CC_LIB)/ansi-image.o $(CC_LIB)/image-resize.o $(CC_LIB)/opt/opt.o $(CC_LIB)/bounds.o $(CC_LIB)/interval-cover.o $(CC_LIB)/interval-cover-util.o $(CC_LIB)/bignum/big.o $(CC_LIB)/bignum/bigz.o $(CC_LIB)/bignum/bign.o $(CC_LIB)/bignum/bigq.o $(CC_LIB)/bignum/big-numbers.o $(CC_LIB)/bignum/polynomial.o $(CC_LIB)/textsvg.o $(CC_LIB)/hexdump.o

SQLITE_OBJECTS=$(SQLITE)/sqlite3.o $(SQLITE)/database.o
OBJECTS=$(CC_LIB_OBJECTS) $(SQLITE_OBJECTS) ../solutions.o ../mesh.o

$(CC_LIB)/%.o : $(CC_LIB)/%.cc makefile
	@$(CXX) $(CXXFLAGS) -c $< -o $@
	@echo -n "."

$(SQLITE)/%.o : $(SQLITE)/%.cc makefile
	@$(CXX) $(CXXFLAGS) -c $< -o $@
	@echo -n "."

%.o : %.cc *.h makefile
	@$(CXX) $(CXXFLAGS) -c $< -o $@
	@echo -n "."

%.svg : %.stl
	../makesvg.exe $< $@

BOVEX_LIB=$(BOVEX)/stdlib.bovex $(BOVEX)/cite.bovex $(BOVEX)/layout.bovex $(BOVEX)/palatino.bovex $(BOVEX)/paper.bib $(BOVEX)/hyph-en-us.tex

solution-table.bovex : table.exe
	./table.exe > $@

SOLID_SVG=platonic-cube.svg platonic-icosahedron.svg platonic-tetrahedron.svg platonic-dodecahedron.svg platonic-octahedron.svg

RESIDUE_SVG=cube-residue.svg cuboctahedron-residue.svg deltoidalicositetrahedron-residue.svg disdyakisdodecahedron-residue.svg disdyakistriacontahedron-residue.svg dodecahedron-residue.svg icosahedron-residue.svg icosidodecahedron-residue.svg octahedron-residue.svg pentagonalicositetrahedron-residue.svg pentakisdodecahedron-residue.svg rhombicdodecahedron-residue.svg rhombictriacontahedron-residue.svg rhombicuboctahedron-residue.svg tetrahedron-residue.svg tetrakishexahedron-residue.svg triakisicosahedron-residue.svg triakisoctahedron-residue.svg triakistetrahedron-residue.svg truncatedcube-residue.svg truncatedcuboctahedron-residue.svg truncateddodecahedron-residue.svg truncatedicosahedron-residue.svg truncatedicosidodecahedron-residue.svg truncatedoctahedron-residue.svg truncatedtetrahedron-residue.svg

default : $(RESIDUE_SVG) $(SOLID_SVG)


ruperts.pdf : ruperts.bovex solution-table.bovex ruperts.bib $(BOVEX_LIB)
	$(BOVEX)/bovex.exe -I . -v ruperts.bovex -o $@

table.exe : table.o $(OBJECTS)
	@$(CXX) $^ -o $@ $(LFLAGS)
	@echo -n "!"

makesvg.exe : makesvg.o $(OBJECTS)
	@$(CXX) $^ -o $@ $(LFLAGS)
	@echo -n "!"

clean :
	rm -f *.o $(OBJECTS) ruperts.pdf $(RESIDUE_SVG) $(SOLID_SVG)

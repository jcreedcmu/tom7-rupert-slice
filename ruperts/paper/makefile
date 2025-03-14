
default : paper.pdf

%.svg : %.stl
	../makesvg.exe $< $@

# requires building bovex.exe
BOVEX=../../rephrase
BOVEX_LIB=$(BOVEX)/stdlib.bovex $(BOVEX)/cite.bovex $(BOVEX)/layout.bovex $(BOVEX)/palatino.bovex $(BOVEX)/paper.bib $(BOVEX)/hyph-en-us.tex

SOLID_SVG=platonic-cube.svg platonic-icosahedron.svg platonic-tetrahedron.svg platonic-dodecahedron.svg platonic-octahedron.svg

RESIDUE_SVG=cube-residue.svg cuboctahedron-residue.svg deltoidalicositetrahedron-residue.svg disdyakisdodecahedron-residue.svg disdyakistriacontahedron-residue.svg dodecahedron-residue.svg icosahedron-residue.svg icosidodecahedron-residue.svg octahedron-residue.svg pentagonalicositetrahedron-residue.svg pentakisdodecahedron-residue.svg rhombicdodecahedron-residue.svg rhombictriacontahedron-residue.svg rhombicuboctahedron-residue.svg tetrahedron-residue.svg tetrakishexahedron-residue.svg triakisicosahedron-residue.svg triakisoctahedron-residue.svg triakistetrahedron-residue.svg truncatedcube-residue.svg truncatedcuboctahedron-residue.svg truncateddodecahedron-residue.svg truncatedicosahedron-residue.svg truncatedicosidodecahedron-residue.svg truncatedoctahedron-residue.svg truncatedtetrahedron-residue.svg

default : $(RESIDUE_SVG) $(SOLID_SVG)


paper.pdf : paper.bovex ruperts.bib $(BOVEX_LIB)
	$(BOVEX)/bovex.exe -I . -v paper.bovex -o paper.pdf

clean :
	rm -f paper.pdf $(RESIDUE_SVG) $(SOLID_SVG)


default : math-hole.mov math-hole-sit.mov

REPHRASE=../../rephrase
BOVEX=$(REPHRASE)/bovex.exe

math-hole.mov : math-hole.bovex
	$(BOVEX) math-hole.bovex -o $@

math-hole-sit.mov : math-hole-sit.bovex
	$(BOVEX) math-hole-sit.bovex -o $@

clean :


default : math-hole.mov math-hole-sit.mov rupert-portrait.mov rupert-pub.mov

REPHRASE=../../rephrase
BOVEX=$(REPHRASE)/bovex.exe

math-hole.mov : math-hole.bovex
	$(BOVEX) math-hole.bovex -o $@

math-hole-sit.mov : math-hole-sit.bovex
	$(BOVEX) math-hole-sit.bovex -o $@

rupert-portrait.mov : rupert-portrait.bovex
	$(BOVEX) rupert-portrait.bovex -o $@

rupert-pub.mov : rupert-pub.bovex
	$(BOVEX) rupert-pub.bovex -o $@

clean :

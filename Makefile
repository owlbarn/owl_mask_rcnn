PROGRAMS =  eval.ml
OCAMLPACKS = owl graphics camlimages.all_formats camlimages.graphics #hdf5_caml

SOURCES = $(addprefix src/, $(PROGRAMS))

OCAMLBUILD = ocamlbuild -use-ocamlfind $(OCAMLFLAGS)

default: all

.PHONY: all clean
all:	$(SOURCES)
	$(OCAMLBUILD) -r -pkg "$(OCAMLPACKS)" $(SOURCES:.ml=.native)
$(SOURCES:.ml=.native): %.native: %.ml
	$(OCAMLBUILD) -r -pkg "$(OCAMLPACKS)" $@

clean::
	ocamlbuild -clean

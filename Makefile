PROGRAMS = weights/load_weights.ml eval.ml
OCAMLPACKS = owl camlimages.all_formats hdf5_caml

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

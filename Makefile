PROGRAMS = eval.ml
OCAMLPACKS = owl camlimages.all_formats

SOURCES = $(addprefix src/, $(PROGRAMS))

CURRENT_DIR = $(notdir $(shell pwd))
OCAMLBUILD  = ocamlbuild -use-ocamlfind $(OCAMLFLAGS)

default: all

.PHONY: all clean
all:	$(SOURCES)
	$(OCAMLBUILD) -r -pkg "$(OCAMLPACKS)" $(SOURCES:.ml=.native)
$(SOURCES:.ml=.native): %.native: %.ml
	$(OCAMLBUILD) -pkg "$(OCAMLPACKS)" $@

clean::
	ocamlbuild -clean
	-$(RM) -r $(wildcard *~ *.tar.gz) *.docdir
	$(RM) $(wildcard $(addprefix doc/, *.html *.css *.aux *.log))

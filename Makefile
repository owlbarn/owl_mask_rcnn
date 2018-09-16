
default: all

.PHONY: all run clean
all:
	dune build examples/evalImage.ml

run:
	dune exec ./examples/evalImage.exe

clean:
	dune clean

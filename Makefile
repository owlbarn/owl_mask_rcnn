
default: build

.PHONY: build run clean
build:
	dune build examples/evalImage.ml

run:
	dune exec ./examples/evalImage.exe

clean:
	dune clean

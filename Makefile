
default: build

.PHONY: build run video clean
build:
	dune build examples/evalImage.ml

run:
	dune exec ./examples/evalImage.exe

video:
	dune exec ./examples/evalVideo.exe

clean:
	dune clean

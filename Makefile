
default: build

.PHONY: build run video clean
build:
	dune build examples/evalImage.exe

run:
	dune exec ./examples/evalImage.exe

video:
	dune exec ./examples/evalVideo.exe

clean:
	dune clean

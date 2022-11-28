all: test

drx.o: drx.cpp drx.hpp
	g++ -c -o drx.o drx.cpp
vdif.o: vdif.cpp vdif.hpp
	g++ -c -o vdif.o vdif.cpp
test.o: test.cpp
	g++ -c -o test.o test.cpp
test: test.o drx.o vdif.o
	g++ -o test test.o drx.o vdif.o

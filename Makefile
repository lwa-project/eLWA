all: convertDRX

drx.o: drx.cpp drx.hpp
	g++ -c -o drx.o drx.cpp
vdif.o: vdif.cpp vdif.hpp
	g++ -c -o vdif.o vdif.cpp
convertDRX.o: convertDRX.cpp
	g++ -c -o convertDRX.o convertDRX.cpp
convertDRX: convertDRX.o drx.o vdif.o
	g++ -o convertDRX convertDRX.o drx.o vdif.o

clean:
	rm -f convertDRX *.o

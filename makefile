benchmark: benchmark.c FFT.o
	gcc benchmark.c FFT.o -o benchmark -lm -O3

FFT.o: FFT.c
	gcc -c FFT.c -o FFT.o -lm -mavx -mavx2 -mfma -O3

clean:
	rm -f benchmark FFT.o test

test: test.c FFT.o
	gcc test.c FFT.o -o test -O3 -lm

FFT_debug.o: FFT.c
	gcc -c FFT.c -o FFT.o -lm -mavx -mavx2 -mfma -O0 -g

bencmark_debug: benchmark.c FFT_debug.o
	gcc benchmark.c FFT.o -o benchmark -lm -O0 -g

test_debug: test.c FFT_debug.o
	gcc test.c FFT.o -o test -O0 -g -lm

spectrum.o: spectrum.c
	gcc $(shell pkg-config --cflags --libs gtk+-3.0)  -c spectrum.c -o spectrum.o

spectrum: spectrum.o FFT.o
	gcc spectrum.o FFT.o $(shell pkg-config --cflags --libs gtk+-3.0)  -lm -lpthread -o spectrum

default: benchmark
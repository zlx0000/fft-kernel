#include <complex.h>
#include <math.h>
#include <immintrin.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>


float complex *FFT_SIMD(float complex *a, size_t n);
float complex *FFT(float complex *a, size_t n);
float complex *IFFT_SIMD(float complex *a, size_t n);
float complex *IFFT(float complex *a, size_t n);
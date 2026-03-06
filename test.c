#include <stdio.h>
#include <stdlib.h>
#include "FFT.h"

int main()
{
    size_t n = 1 << 20;
    float complex *A = (float complex *)_mm_malloc(n * sizeof(float complex), 32);

    float x = -1.0;
    for (size_t i = 0; i < n; i++) {
        if (i % 16384 == 0) x *= -1.0;
        A[i] = x * sqrt(67108864 - ((i%16384)-8192)*((i%16384)-8192)) + 0.0 * I;
    }

    float complex *A_fft = FFT_SIMD(A, n);
    _mm_free(A);
    A = IFFT_SIMD(A_fft, n);
    for (size_t i = 0; i < n; i++) {
        printf("%f + %fi\n", crealf(A[i]), cimagf(A[i]));
    }
    _mm_free(A);
    _mm_free(A_fft);
    return 0;
}
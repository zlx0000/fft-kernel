#include <stdio.h>
#include <stdlib.h>
#include "FFT.h"

int main()
{
    size_t n = 1 << 20;
    float complex *A = (float complex *)_mm_malloc(n * sizeof(float complex), 32);

    for (size_t i = 0; i < n; i++) {
        if (i % 65536 < 32768) {
            A[i] = 1.0 + 0.0 * I;
        } else {
            A[i] = 0.0 + 0.0 * I;
        }
    }
    float complex *B = FFT_SIMD(A, n);
    _mm_free(A);
    float complex *C = IFFT_SIMD(B, n);
    _mm_free(B);
    for (size_t i = 0; i < n; i++) {
        printf("%f + %fi\n", crealf(C[i]), cimagf(C[i]));
    }
    _mm_free(C);
    return 0;
}
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
    A = IFFT_SIMD(FFT_SIMD(A, n), n);
    for (size_t i = 0; i < n; i++) {
        printf("%f + %fi\n", crealf(A[i]), cimagf(A[i]));
    }
    _mm_free(A);
    return 0;
}
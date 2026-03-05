#include "FFT.h"
#include <stdio.h>
#include <time.h>
#include <string.h>
#include <linux/time.h>


static double now()
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

void benchmark_fft(size_t n, int runs)
{
    printf("FFT size: %zu\n", n);
    printf("Runs: %d\n\n", runs);

    float complex *input =
        (float complex *)_mm_malloc(n * sizeof(float complex), 32);

    float complex *input_copy =
        (float complex *)_mm_malloc(n * sizeof(float complex), 32);

    /* generate deterministic input */
    for (size_t i = 0; i < n; i++)
        input[i] = cos(i * 0.1f) + I * sin(i * 0.1f);

    /* ---------------- scalar ---------------- */

    double best_scalar = 1e30;

    for (int r = 0; r < runs; r++) {

        memcpy(input_copy, input, n * sizeof(float complex));

        double t0 = now();
        float complex *A = FFT(input_copy, n);
        double t1 = now();

        double dt = t1 - t0;

        if (dt < best_scalar)
            best_scalar = dt;

        _mm_free(A);
    }

    /* ---------------- SIMD ---------------- */

    double best_simd = 1e30;

    for (int r = 0; r < runs; r++) {

        memcpy(input_copy, input, n * sizeof(float complex));

        double t0 = now();
        float complex *A = FFT_SIMD(input_copy, n);
        double t1 = now();

        double dt = t1 - t0;

        if (dt < best_simd)
            best_simd = dt;

        _mm_free(A);
    }

    printf("Scalar time : %.6f s\n", best_scalar);
    printf("SIMD time   : %.6f s\n", best_simd);
    printf("SIMD Speedup     : %.2fx\n", best_scalar / best_simd);

    _mm_free(input);
    _mm_free(input_copy);
}

int main(int argc, char *argv[])
{
    size_t n = 1 << 20;

    benchmark_fft(n, 5);

    return 0;
}
#include "FFT.h"


static uint32_t rev(uint32_t x, uint32_t n)
{
    uint32_t res = 0;
    uint32_t mask = 0b01 << (n - 1);
    for (int i = 0; i < n; i++) {
        if (x & mask) {
            res += 1 << i;
        }
        mask >>= 1;
    }
    return res;
}

static float complex *bit_reverse_copy(float complex *a, size_t n)
{
    float complex *res = (float complex *)_mm_malloc(n * sizeof(float complex), 32);
    for (uint32_t i = 0; i <= n-1; i++) {
        res[rev(i, __builtin_ctz(n))] = a[i];
    }
    return res;
}

float complex *FFT_SIMD(float complex *a, size_t n)
{
    if (! (n > 0 || (n & (n - 1)) != 0)) {
        return NULL;
    }

    float complex *A = bit_reverse_copy(a, n);
    __m256 minus = _mm256_set1_ps(-1.0);
    __m256 two = _mm256_set1_ps(2.0);
    __m256i fix = _mm256_setr_epi32(0,1,4,5,2,3,6,7);
    for (uint32_t i = 1; i <= __builtin_ctz(n); i++) {
        uint32_t m = 1 << i;
        float complex w_m = cexp(-2.0 * I * M_PI / m);
        float complex w_m_8 = w_m * w_m * w_m * w_m
                            * w_m * w_m * w_m * w_m;
        __m256 w_m_re_vec_8 = _mm256_set1_ps(creal(w_m_8));
        __m256 w_m_im_vec_8 = _mm256_set1_ps(cimag(w_m_8));
        for (uint32_t j = 0; j <= n-1; j+=m) {
            float complex w = 1.0;
            __m256 w_re_vec = _mm256_setr_ps(creal(w), creal(w * w_m), creal(w * w_m * w_m),                      \
            creal(w * w_m * w_m * w_m), creal(w * w_m * w_m * w_m * w_m), creal(w * w_m * w_m * w_m * w_m * w_m),\
            creal(w * w_m * w_m * w_m * w_m * w_m * w_m), creal(w * w_m * w_m * w_m * w_m * w_m * w_m * w_m));
            __m256 w_im_vec = _mm256_setr_ps(cimag(w), cimag(w * w_m), cimag(w * w_m * w_m),                      \
            cimag(w * w_m * w_m * w_m), cimag(w * w_m * w_m * w_m * w_m), cimag(w * w_m * w_m * w_m * w_m * w_m),\
            cimag(w * w_m * w_m * w_m * w_m * w_m * w_m), cimag(w * w_m * w_m * w_m * w_m * w_m * w_m * w_m));
            for (uint32_t k = 0; k <= m/2-1; k++) {
                if ((m/2-1) - k >= 8) {
                    __m256 A_vec_a = _mm256_loadu_ps((float *)&A[j + m/2 + k]);
                    __m256 A_vec_b = _mm256_loadu_ps((float *)&A[j + m/2 + k] + 8);
                    __m256 A_re_vec = _mm256_shuffle_ps(A_vec_a, A_vec_b, 0x88);
                    __m256 A_im_vec = _mm256_shuffle_ps(A_vec_a, A_vec_b, 0xDD);
                    A_re_vec = _mm256_permutevar8x32_ps(A_re_vec, fix);
                    A_im_vec = _mm256_permutevar8x32_ps(A_im_vec, fix);
                    __m256 t_re_vec = _mm256_fmadd_ps(w_re_vec, A_re_vec, _mm256_mul_ps(minus, _mm256_mul_ps(w_im_vec, A_im_vec)));
                    __m256 t_im_vec = _mm256_fmadd_ps(w_re_vec, A_im_vec, _mm256_mul_ps(w_im_vec, A_re_vec));
                    __m256 u_vec_a = _mm256_loadu_ps((float *)&A[j + k]);
                    __m256 u_vec_b = _mm256_loadu_ps((float *)&A[j + k] + 8);
                    __m256 u_re_vec = _mm256_shuffle_ps(u_vec_a, u_vec_b, 0x88);
                    __m256 u_im_vec = _mm256_shuffle_ps(u_vec_a, u_vec_b, 0xDD);
                    u_re_vec = _mm256_permutevar8x32_ps(u_re_vec, fix);
                    u_im_vec = _mm256_permutevar8x32_ps(u_im_vec, fix);
                    u_re_vec = _mm256_add_ps(u_re_vec, t_re_vec);
                    u_im_vec = _mm256_add_ps(u_im_vec, t_im_vec);
                    A_re_vec = _mm256_sub_ps(u_re_vec, _mm256_mul_ps(t_re_vec, two));
                    A_im_vec = _mm256_sub_ps(u_im_vec, _mm256_mul_ps(t_im_vec, two));
                    __m256 lo_A = _mm256_unpacklo_ps(A_re_vec, A_im_vec);
                    __m256 hi_A = _mm256_unpackhi_ps(A_re_vec, A_im_vec);
                    __m256 out0_A = _mm256_permute2f128_ps(lo_A, hi_A, 0x20);
                    __m256 out1_A = _mm256_permute2f128_ps(lo_A, hi_A, 0x31);
                    __m256 lo_u = _mm256_unpacklo_ps(u_re_vec, u_im_vec);
                    __m256 hi_u = _mm256_unpackhi_ps(u_re_vec, u_im_vec);
                    __m256 out0_u = _mm256_permute2f128_ps(lo_u, hi_u, 0x20);
                    __m256 out1_u = _mm256_permute2f128_ps(lo_u, hi_u, 0x31);
                    _mm256_storeu_ps((float *)&A[j + k], out0_u);
                    _mm256_storeu_ps((float *)&A[j + k] + 8, out1_u);
                    _mm256_storeu_ps((float *)&A[j + m/2 + k], out0_A);
                    _mm256_storeu_ps((float *)&A[j + m/2 + k] + 8, out1_A);

                    w *= w_m_8;
                    __m256 w_re_old = w_re_vec;
                    w_re_vec = _mm256_fmadd_ps(w_re_vec, w_m_re_vec_8, _mm256_mul_ps(minus, _mm256_mul_ps(w_im_vec, w_m_im_vec_8)));
                    w_im_vec = _mm256_fmadd_ps(w_re_old, w_m_im_vec_8, _mm256_mul_ps(w_im_vec, w_m_re_vec_8));
                    k += 7;
                    continue;
                }
                float complex t = w * A[j + k + m/2];
                float complex u = A[j + k];
                A[j + k ] = u + t;
                A[j + k + m/2] = u - t;
                w *= w_m;
            }
        }
    }
    return A;
}

float complex *IFFT_SIMD(float complex *a, size_t n)
{
    if (! (n > 0 || (n & (n - 1)) != 0)) {
        return NULL;
    }

    float complex *A = bit_reverse_copy(a, n);
    __m256 minus = _mm256_set1_ps(-1.0);
    __m256 half = _mm256_set1_ps(0.5);
    __m256i fix = _mm256_setr_epi32(0,1,4,5,2,3,6,7);
    for (uint32_t i = 1; i <= __builtin_ctz(n); i++) {
        uint32_t m = 1 << i;
        float complex w_m = cexp(2.0 * I * M_PI / m);
        float complex w_m_8 = w_m * w_m * w_m * w_m
                            * w_m * w_m * w_m * w_m;
        __m256 w_m_re_vec_8 = _mm256_set1_ps(creal(w_m_8));
        __m256 w_m_im_vec_8 = _mm256_set1_ps(cimag(w_m_8));
        for (uint32_t j = 0; j <= n-1; j+=m) {
            float complex w = 1.0;
            __m256 w_re_vec = _mm256_setr_ps(creal(w), creal(w * w_m), creal(w * w_m * w_m),                      \
            creal(w * w_m * w_m * w_m), creal(w * w_m * w_m * w_m * w_m), creal(w * w_m * w_m * w_m * w_m * w_m),\
            creal(w * w_m * w_m * w_m * w_m * w_m * w_m), creal(w * w_m * w_m * w_m * w_m * w_m * w_m * w_m));
            __m256 w_im_vec = _mm256_setr_ps(cimag(w), cimag(w * w_m), cimag(w * w_m * w_m),                      \
            cimag(w * w_m * w_m * w_m), cimag(w * w_m * w_m * w_m * w_m), cimag(w * w_m * w_m * w_m * w_m * w_m),\
            cimag(w * w_m * w_m * w_m * w_m * w_m * w_m), cimag(w * w_m * w_m * w_m * w_m * w_m * w_m * w_m));
            for (uint32_t k = 0; k <= m/2-1; k++) {
                if ((m/2-1) - k >= 8) {
                    __m256 A_vec_a = _mm256_loadu_ps((float *)&A[j + m/2 + k]);
                    __m256 A_vec_b = _mm256_loadu_ps((float *)&A[j + m/2 + k] + 8);
                    __m256 A_re_vec = _mm256_shuffle_ps(A_vec_a, A_vec_b, 0x88);
                    __m256 A_im_vec = _mm256_shuffle_ps(A_vec_a, A_vec_b, 0xDD);
                    A_re_vec = _mm256_permutevar8x32_ps(A_re_vec, fix);
                    A_im_vec = _mm256_permutevar8x32_ps(A_im_vec, fix);
                    __m256 t_re_vec = _mm256_fmadd_ps(w_re_vec, A_re_vec, _mm256_mul_ps(minus, _mm256_mul_ps(w_im_vec, A_im_vec)));
                    __m256 t_im_vec = _mm256_fmadd_ps(w_re_vec, A_im_vec, _mm256_mul_ps(w_im_vec, A_re_vec));
                    __m256 u_vec_a = _mm256_loadu_ps((float *)&A[j + k]);
                    __m256 u_vec_b = _mm256_loadu_ps((float *)&A[j + k] + 8);
                    __m256 u_re_vec = _mm256_shuffle_ps(u_vec_a, u_vec_b, 0x88);
                    __m256 u_im_vec = _mm256_shuffle_ps(u_vec_a, u_vec_b, 0xDD);
                    u_re_vec = _mm256_permutevar8x32_ps(u_re_vec, fix);
                    u_im_vec = _mm256_permutevar8x32_ps(u_im_vec, fix);
                    u_re_vec = _mm256_mul_ps(half, _mm256_add_ps(u_re_vec, t_re_vec));
                    u_im_vec = _mm256_mul_ps(half, _mm256_add_ps(u_im_vec, t_im_vec));
                    A_re_vec = _mm256_sub_ps(u_re_vec, t_re_vec);
                    A_im_vec = _mm256_sub_ps(u_im_vec, t_im_vec);
                    __m256 lo_A = _mm256_unpacklo_ps(A_re_vec, A_im_vec);
                    __m256 hi_A = _mm256_unpackhi_ps(A_re_vec, A_im_vec);
                    __m256 out0_A = _mm256_permute2f128_ps(lo_A, hi_A, 0x20);
                    __m256 out1_A = _mm256_permute2f128_ps(lo_A, hi_A, 0x31);
                    __m256 lo_u = _mm256_unpacklo_ps(u_re_vec, u_im_vec);
                    __m256 hi_u = _mm256_unpackhi_ps(u_re_vec, u_im_vec);
                    __m256 out0_u = _mm256_permute2f128_ps(lo_u, hi_u, 0x20);
                    __m256 out1_u = _mm256_permute2f128_ps(lo_u, hi_u, 0x31);
                    _mm256_storeu_ps((float *)&A[j + k], out0_u);
                    _mm256_storeu_ps((float *)&A[j + k] + 8, out1_u);
                    _mm256_storeu_ps((float *)&A[j + m/2 + k], out0_A);
                    _mm256_storeu_ps((float *)&A[j + m/2 + k] + 8, out1_A);

                    w *= w_m_8;
                    __m256 w_re_old = w_re_vec;
                    w_re_vec = _mm256_fmadd_ps(w_re_vec, w_m_re_vec_8, _mm256_mul_ps(minus, _mm256_mul_ps(w_im_vec, w_m_im_vec_8)));
                    w_im_vec = _mm256_fmadd_ps(w_re_old, w_m_im_vec_8, _mm256_mul_ps(w_im_vec, w_m_re_vec_8));
                    k += 7;
                    continue;
                }
                float complex t = w * A[j + k + m/2];
                float complex u = A[j + k];
                A[j + k ] = (u + t) * 0.5;
                A[j + k + m/2] = (u - t) * 0.5;
                w *= w_m;
            }
        }
    }
    return A;
}

float complex *FFT(float complex *a, size_t n)
{
    if (n == 0 || (n & (n - 1)) != 0) {
        return NULL;
    }

    float complex *A = bit_reverse_copy(a, n);
    for (uint32_t i = 1; i <= __builtin_ctz(n); i++) {
        uint32_t m = 1 << i;
        float complex w_m = cexp(-2.0 * I * M_PI / m);
        for (uint32_t j = 0; j <= n-1; j+=m) {
            float complex w = 1.0;
            for (uint32_t k = 0; k <= m/2-1; k++) {
                float complex t = w * A[j + k + m/2];
                float complex u = A[j + k];
                A[j + k ] = u + t;
                A[j + k + m/2] = u - t;
                w *= w_m;
            }
        }
    }
    return A;
}

float complex *IFFT(float complex *a, size_t n)
{
    if (n == 0 || (n & (n - 1)) != 0) {
        return NULL;
    }
    
    float complex *A = bit_reverse_copy(a, n);
    for (uint32_t i = 1; i <= __builtin_ctz(n); i++) {
        uint32_t m = 1 << i;
        float complex w_m = cexp(2.0 * I * M_PI / m);
        for (uint32_t j = 0; j <= n-1; j+=m) {
            float complex w = 1.0;
            for (uint32_t k = 0; k <= m/2-1; k++) {
                float complex t = w * A[j + k + m/2];
                float complex u = A[j + k];
                A[j + k ] = (u + t) * 0.5;
                A[j + k + m/2] = (u - t) * 0.5;
                w *= w_m;
            }
        }
    }

    return A;
}
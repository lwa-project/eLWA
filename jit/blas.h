#ifndef JIT_BLAS_H_INCLUDE_GUARD_
#define JIT_BLAS_H_INCLUDE_GUARD_

#include <complex.h>

/*
  Easy BLAS for C
*/

// cblas_[sd]scal replacements
inline void blas_sscal(const int N, 
                       const float alpha, 
                       float *X, 
                       const int incX) {
    for(int i=0; i<N; i++) {
        *X *= alpha;
        X += incX;
    }
}

inline void blas_dscal(const int N, 
                       const double alpha, 
                       double *X, 
                       const int incX) {
    for(int i=0; i<N; i++) {
        *X *= alpha;
        X += incX;
    }
}

// cblas_[cz]dotc_sub replacements
inline void blas_cdotc_sub(const int N, 
                           const float complex* X, const int incX, 
                           const float complex* Y, const int incY,
                           float complex* dotc) {
    float complex accum = 0.0;
    for(int i=0; i<N; i++) {
        accum += conj(*X) * *Y;
        X += incX;
        Y += incY;
    }
    *dotc = accum;
}

inline void blas_zdotc_sub(const int N, 
                           const double complex* X, const int incX, 
                           const double complex* Y, const int incY,
                           double complex* dotc) {
    double complex accum = 0.0;
    for(int i=0; i<N; i++) {
        accum += conj(*X) * *Y;
        X += incX;
        Y += incY;
    }
    *dotc = accum;
}
#endif // JIT_BLAS_H_INCLUDE_GUARD_

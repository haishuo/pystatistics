#ifndef DVEB_DENSE_RT_H
#define DVEB_DENSE_RT_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace dveb::dense {

enum Status : int {
    Ok = 0,
    Invalid = 2,
    NonFinite = 3,
    Shape = 4,
    Alias = 5,
    Factorization = 6,
    Allocation = 7,
    Schedule = 8,
};

enum Schedule : int {
    Auto = 0,
    Serial = 1,
    WorkItemParallel = 2,
};

inline bool checked_product(std::size_t a, std::size_t b, std::size_t &out) {
    if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a) return false;
    out = a * b;
    return true;
}

inline bool overlaps(const void *a, std::size_t a_bytes,
                     const void *b, std::size_t b_bytes) {
    if (!a || !b || a_bytes == 0 || b_bytes == 0) return false;
    const auto ab = reinterpret_cast<std::uintptr_t>(a);
    const auto bb = reinterpret_cast<std::uintptr_t>(b);
    return ab < bb + b_bytes && bb < ab + a_bytes;
}

inline std::size_t max_segment(const std::int64_t *offsets, std::size_t count) {
    if (!offsets || count < 1 || offsets[0] != 0) {
        return std::numeric_limits<std::size_t>::max();
    }
    std::size_t result = 0;
    for (std::size_t i = 1; i < count; ++i) {
        if (offsets[i] < offsets[i - 1]) {
            return std::numeric_limits<std::size_t>::max();
        }
        const auto delta = static_cast<std::uint64_t>(offsets[i] - offsets[i - 1]);
        if (delta > std::numeric_limits<std::size_t>::max()) {
            return std::numeric_limits<std::size_t>::max();
        }
        result = std::max(result, static_cast<std::size_t>(delta));
    }
    return result;
}

inline std::size_t all_in_range(const std::int64_t *values, std::size_t count,
                                std::size_t upper) {
    if (!values) return count == 0 ? 1 : 0;
    for (std::size_t i = 0; i < count; ++i) {
        if (values[i] < 0 || static_cast<std::uint64_t>(values[i]) >= upper) return 0;
    }
    return 1;
}

inline void fill_vector(double *dst, std::size_t count, double value) {
    std::fill(dst, dst + count, value);
}

inline void fill_matrix(double *dst, std::size_t rows, std::size_t cols,
                        std::size_t stride, double value) {
    for (std::size_t i = 0; i < rows; ++i) {
        std::fill(dst + i * stride, dst + i * stride + cols, value);
    }
}

inline void matmul(const double *a, std::size_t as,
                   const double *b, std::size_t bs,
                   double *c, std::size_t cs,
                   std::size_t m, std::size_t n, std::size_t k) {
    fill_matrix(c, m, n, cs, 0.0);
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t z = 0; z < k; ++z) {
            const double av = a[i * as + z];
#pragma omp simd
            for (std::size_t j = 0; j < n; ++j) {
                c[i * cs + j] += av * b[z * bs + j];
            }
        }
    }
}

inline void matmul_nt(const double *a, std::size_t as,
                      const double *b, std::size_t bs,
                      double *c, std::size_t cs,
                      std::size_t rows_a, std::size_t rows_b,
                      std::size_t common) {
    for (std::size_t i = 0; i < rows_a; ++i) {
        for (std::size_t j = 0; j < rows_b; ++j) {
            double sum = 0.0;
#pragma omp simd reduction(+:sum)
            for (std::size_t k = 0; k < common; ++k) {
                sum += a[i * as + k] * b[j * bs + k];
            }
            c[i * cs + j] = sum;
        }
    }
}

inline void matmul_sym_left(const double *g, std::size_t gs,
                            const double *l, std::size_t ls,
                            double *out, std::size_t os, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            double sum = 0.0;
#pragma omp simd reduction(+:sum)
            for (std::size_t k = 0; k < n; ++k) {
                sum += (g[i * gs + k] + g[k * gs + i]) * l[k * ls + j];
            }
            out[i * os + j] = sum;
        }
    }
}

inline bool cholesky(const double *a, std::size_t as,
                     double *l, std::size_t ls, std::size_t n) {
    fill_matrix(l, n, n, ls, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j <= i; ++j) {
            double sum = a[i * as + j];
            for (std::size_t k = 0; k < j; ++k) {
                sum -= l[i * ls + k] * l[j * ls + k];
            }
            if (i == j) {
                if (!(sum > 0.0) || !std::isfinite(sum)) return false;
                l[i * ls + j] = std::sqrt(sum);
            } else {
                l[i * ls + j] = sum / l[j * ls + j];
            }
        }
    }
    return true;
}

inline void inverse_from_cholesky(const double *l, std::size_t ls,
                                  double *inverse, std::size_t is,
                                  double *work, std::size_t n) {
    fill_matrix(inverse, n, n, is, 0.0);
    for (std::size_t col = 0; col < n; ++col) {
        for (std::size_t i = 0; i < n; ++i) {
            double sum = i == col ? 1.0 : 0.0;
            for (std::size_t k = 0; k < i; ++k) {
                sum -= l[i * ls + k] * work[k];
            }
            work[i] = sum / l[i * ls + i];
        }
        for (std::size_t ii = n; ii-- > 0;) {
            double sum = work[ii];
            for (std::size_t k = ii + 1; k < n; ++k) {
                sum -= l[k * ls + ii] * inverse[k * is + col];
            }
            inverse[ii * is + col] = sum / l[ii * ls + ii];
        }
    }
}

inline bool finite_vector(const double *value, std::size_t count) {
    for (std::size_t i = 0; i < count; ++i) {
        if (!std::isfinite(value[i])) return false;
    }
    return true;
}

}  // namespace dveb::dense

#endif

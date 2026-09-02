#include "arma_native.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <new>
#include <vector>

#include <omp.h>

namespace {

constexpr double kKappa = 1.0e6;
constexpr double kPenalty = 1.0e18;
constexpr double kStationaryRtol = 1.0e-13;
constexpr int kMaxDoublings = 60;

struct Workspace {
    explicit Workspace(int max_r)
        : max_r(max_r), matrices(static_cast<std::size_t>(7) * max_r * max_r),
          vectors(static_cast<std::size_t>(4) * max_r) {}

    int max_r;
    std::vector<double> matrices;
    std::vector<double> vectors;
};

bool stationary(
    const double* phi,
    const double* loading,
    int r,
    Workspace& workspace,
    double* p
) {
    const std::size_t rr = static_cast<std::size_t>(r) * r;
    double* s = workspace.matrices.data();
    double* a = s + rr;
    double* as = a + rr;
    double* u = as + rr;
    double* anew = u + rr;

    std::fill(a, a + rr, 0.0);
    for (int i = 0; i < r; ++i) {
        a[static_cast<std::size_t>(i) * r] = phi[i];
    }
    for (int i = 0; i + 1 < r; ++i) {
        a[static_cast<std::size_t>(i) * r + i + 1] = 1.0;
    }
    for (int i = 0; i < r; ++i) {
        for (int j = 0; j < r; ++j) {
            s[static_cast<std::size_t>(i) * r + j] = loading[i] * loading[j];
        }
    }

    for (int iteration = 0; iteration < kMaxDoublings; ++iteration) {
        std::fill(as, as + rr, 0.0);
        for (int i = 0; i < r; ++i) {
            for (int inner = 0; inner < r; ++inner) {
                const double left = a[static_cast<std::size_t>(i) * r + inner];
                for (int j = 0; j < r; ++j) {
                    as[static_cast<std::size_t>(i) * r + j] +=
                        left * s[static_cast<std::size_t>(inner) * r + j];
                }
            }
        }

        double u_max = 0.0;
        for (int i = 0; i < r; ++i) {
            for (int j = 0; j < r; ++j) {
                double value = 0.0;
                for (int inner = 0; inner < r; ++inner) {
                    value += as[static_cast<std::size_t>(i) * r + inner] *
                             a[static_cast<std::size_t>(j) * r + inner];
                }
                if (!std::isfinite(value)) {
                    return false;
                }
                u[static_cast<std::size_t>(i) * r + j] = value;
                u_max = std::max(u_max, std::abs(value));
            }
        }

        double scale = 0.0;
        for (std::size_t index = 0; index < rr; ++index) {
            s[index] += u[index];
            scale = std::max(scale, std::abs(s[index]));
        }
        if (!std::isfinite(scale)) {
            return false;
        }
        if (u_max <= kStationaryRtol * std::max(1.0, scale)) {
            for (int i = 0; i < r; ++i) {
                for (int j = 0; j < r; ++j) {
                    p[static_cast<std::size_t>(i) * r + j] =
                        0.5 * (s[static_cast<std::size_t>(i) * r + j] +
                               s[static_cast<std::size_t>(j) * r + i]);
                }
            }
            return true;
        }

        std::fill(anew, anew + rr, 0.0);
        for (int i = 0; i < r; ++i) {
            for (int inner = 0; inner < r; ++inner) {
                const double left = a[static_cast<std::size_t>(i) * r + inner];
                for (int j = 0; j < r; ++j) {
                    anew[static_cast<std::size_t>(i) * r + j] +=
                        left * a[static_cast<std::size_t>(inner) * r + j];
                }
            }
        }
        for (std::size_t index = 0; index < rr; ++index) {
            if (!std::isfinite(anew[index])) {
                return false;
            }
            a[index] = anew[index];
        }
    }
    return false;
}

bool evaluate_one(
    const double* z,
    const double* phi,
    const double* loading,
    int64_t n,
    int r,
    Workspace& workspace,
    double& nll,
    double& sigma2
) {
    const std::size_t rr = static_cast<std::size_t>(r) * r;
    double* p = workspace.matrices.data() + static_cast<std::size_t>(5) * rr;
    double* filtered_p = p + rr;
    double* state = workspace.vectors.data();
    double* filtered_state = state + r;
    double* gain = filtered_state + r;
    double* row_zero = gain + r;

    if (!stationary(phi, loading, r, workspace, p)) {
        std::fill(p, p + rr, 0.0);
        for (int i = 0; i < r; ++i) {
            p[static_cast<std::size_t>(i) * r + i] = kKappa;
        }
    }
    std::fill(state, state + r, 0.0);

    double sse = 0.0;
    double sum_log_f = 0.0;
    for (int64_t t = 0; t < n; ++t) {
        const double innovation = z[t] - state[0];
        const double f = p[0];
        if (!std::isfinite(f) || f <= 0.0) {
            return false;
        }
        const double inverse_f = 1.0 / f;
        for (int i = 0; i < r; ++i) {
            gain[i] = p[static_cast<std::size_t>(i) * r] * inverse_f;
        }
        for (int i = 0; i < r; ++i) {
            filtered_state[i] = state[i] + gain[i] * innovation;
            for (int j = 0; j < r; ++j) {
                filtered_p[static_cast<std::size_t>(i) * r + j] =
                    p[static_cast<std::size_t>(i) * r + j] -
                    gain[i] * p[j];
            }
        }
        for (int i = 0; i < r; ++i) {
            double value = phi[i] * filtered_state[0];
            if (i + 1 < r) {
                value += filtered_state[i + 1];
            }
            state[i] = value;
        }
        for (int j = 0; j < r; ++j) {
            row_zero[j] = phi[0] * filtered_p[j];
            if (r > 1) {
                row_zero[j] += filtered_p[r + j];
            }
        }
        for (int i = 0; i < r; ++i) {
            double first_column;
            if (i == 0) {
                first_column = row_zero[0];
            } else {
                first_column = phi[i] * filtered_p[0];
                if (i + 1 < r) {
                    first_column += filtered_p[static_cast<std::size_t>(i + 1) * r];
                }
            }
            for (int j = 0; j < r; ++j) {
                double value = first_column * phi[j];
                if (j + 1 < r) {
                    if (i == 0) {
                        value += row_zero[j + 1];
                    } else {
                        double shifted = phi[i] * filtered_p[j + 1];
                        if (i + 1 < r) {
                            shifted += filtered_p[
                                static_cast<std::size_t>(i + 1) * r + j + 1
                            ];
                        }
                        value += shifted;
                    }
                }
                value += loading[i] * loading[j];
                if (!std::isfinite(value)) {
                    return false;
                }
                p[static_cast<std::size_t>(i) * r + j] = value;
            }
        }
        sse += innovation * innovation / f;
        sum_log_f += std::log(f);
    }

    if (!std::isfinite(sse) || sse <= 0.0) {
        return false;
    }
    sigma2 = sse / static_cast<double>(n);
    nll = 0.5 * static_cast<double>(n) * std::log(2.0 * std::acos(-1.0) * sigma2) +
          0.5 * sum_log_f + 0.5 * static_cast<double>(n);
    return std::isfinite(nll) && std::isfinite(sigma2);
}

}  // namespace

struct arma_cpu_context {
    int max_r;
    int max_threads;
    std::vector<Workspace> workspaces;

    arma_cpu_context(int requested_r, int requested_threads)
        : max_r(requested_r), max_threads(requested_threads) {
        workspaces.reserve(static_cast<std::size_t>(max_threads));
        for (int index = 0; index < max_threads; ++index) {
            workspaces.emplace_back(max_r);
        }
    }
};

extern "C" const char* arma_native_version(void) {
    return "pystatistics.dveb-arma-phase0b.native.v1";
}

extern "C" arma_cpu_context* arma_cpu_create(int max_r, int max_threads) {
    if (max_r < 1 || max_r > 25 || max_threads < 1 || max_threads > 256) {
        return nullptr;
    }
    try {
        return new arma_cpu_context(max_r, max_threads);
    } catch (const std::bad_alloc&) {
        return nullptr;
    }
}

extern "C" void arma_cpu_destroy(arma_cpu_context* context) {
    delete context;
}

extern "C" int arma_cpu_evaluate(
    arma_cpu_context* context,
    const double* z,
    const double* phi,
    const double* loading,
    int64_t k,
    int64_t n,
    int r,
    int threads,
    double* nll,
    double* sigma2,
    uint8_t* status
) {
    if (context == nullptr || z == nullptr || phi == nullptr || loading == nullptr ||
        nll == nullptr || sigma2 == nullptr || status == nullptr || k < 1 || n < 1 ||
        r < 1 || r > context->max_r || threads < 1 || threads > context->max_threads) {
        return ARMA_NATIVE_INVALID_ARGUMENT;
    }

    omp_set_dynamic(0);
    omp_set_num_threads(threads);
#pragma omp parallel for schedule(static)
    for (int64_t row = 0; row < k; ++row) {
        Workspace& workspace = context->workspaces[static_cast<std::size_t>(omp_get_thread_num())];
        double row_nll = kPenalty;
        double row_sigma2 = 1.0;
        const bool ok = evaluate_one(
            z + row * n,
            phi + row * r,
            loading + row * r,
            n,
            r,
            workspace,
            row_nll,
            row_sigma2
        );
        nll[row] = ok ? row_nll : kPenalty;
        sigma2[row] = ok ? row_sigma2 : 1.0;
        status[row] = ok ? 1 : 0;
    }
    return ARMA_NATIVE_OK;
}

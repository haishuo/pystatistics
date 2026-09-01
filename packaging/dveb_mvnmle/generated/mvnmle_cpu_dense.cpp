// dveb-generated CPU dense implementation — do not edit.
// Source: examples/mvnmle/mvnmle_cpu.dveb
#include "mvnmle_cpu_abi_v1.h"
#include "dense/dense_rt.h"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <new>
#include <memory>
#include <vector>
#include <omp.h>

struct DenseWorkerScratch {
    std::vector<double> s_a;
    std::vector<double> s_chol;
    std::vector<double> s_inverse;
    std::vector<double> s_moment;
    std::vector<double> s_temporary;
    std::vector<double> s_delta;
    double r_value = 0;
    std::vector<double> r_total_gmu;
    std::vector<double> r_total_gsigma;
};

struct dveb_dense_context {
    static constexpr std::size_t max_local = 64;
    size_t p = 0;
    size_t patterns = 0;
    std::vector<int64_t> offsets;
    std::vector<int64_t> centered_offsets;
    std::vector<int64_t> observed_index;
    std::vector<double> n_k;
    std::vector<double> ybar;
    std::vector<double> centered;
    std::vector<double> jitter_scale;
    double epsilon = 0;
    std::vector<double> s_lower;
    std::vector<double> s_sigma;
    std::vector<double> s_total_gmu;
    std::vector<double> s_total_gsigma;
    std::vector<double> s_dlower;
    std::vector<DenseWorkerScratch> workers;
    std::size_t max_threads = 0;
    std::size_t scratch_bytes = 0;
};

extern "C" uint32_t dveb_dense_abi_version(void) { return DVEB_DENSE_ABI_V1; }
extern "C" const char *dveb_dense_status_string(int status) {
    switch (status) {
      case DVEB_DENSE_OK: return "ok";
      case DVEB_DENSE_INVALID: return "invalid argument";
      case DVEB_DENSE_NONFINITE: return "non-finite input or output";
      case DVEB_DENSE_SHAPE: return "shape contract failed";
      case DVEB_DENSE_ALIAS: return "illegal alias";
      case DVEB_DENSE_FACTORIZATION: return "factorization failed";
      case DVEB_DENSE_ALLOCATION: return "context allocation failed";
      case DVEB_DENSE_SCHEDULE: return "illegal schedule";
      default: return "unknown status";
    }
}

extern "C" int dveb_dense_context_create(
    size_t p,
    size_t patterns,
    const int64_t *offsets,
    size_t offsets_count,
    const int64_t *centered_offsets,
    size_t centered_offsets_count,
    const int64_t *observed_index,
    size_t observed_index_count,
    const double *n_k,
    size_t n_k_count,
    const double *ybar,
    size_t ybar_count,
    const double *centered,
    size_t centered_count,
    const double *jitter_scale,
    size_t jitter_scale_count,
    double epsilon,
    size_t max_threads,
    dveb_dense_context **out_context) {
    if (!out_context || max_threads < 1) return DVEB_DENSE_INVALID;
    *out_context = nullptr;
    if (offsets_count && !offsets) return DVEB_DENSE_INVALID;
    if (centered_offsets_count && !centered_offsets) return DVEB_DENSE_INVALID;
    if (observed_index_count && !observed_index) return DVEB_DENSE_INVALID;
    if (n_k_count && !n_k) return DVEB_DENSE_INVALID;
    if (ybar_count && !ybar) return DVEB_DENSE_INVALID;
    if (centered_count && !centered) return DVEB_DENSE_INVALID;
    if (jitter_scale_count && !jitter_scale) return DVEB_DENSE_INVALID;
    try {
        auto owned = std::make_unique<dveb_dense_context>();
        auto *ctx = owned.get();
        ctx->max_threads = max_threads;
        ctx->p = p;
        ctx->patterns = patterns;
        if (offsets_count) ctx->offsets.assign(offsets, offsets + offsets_count);
        if (centered_offsets_count) ctx->centered_offsets.assign(centered_offsets, centered_offsets + centered_offsets_count);
        if (observed_index_count) ctx->observed_index.assign(observed_index, observed_index + observed_index_count);
        if (n_k_count) ctx->n_k.assign(n_k, n_k + n_k_count);
        if (!dveb::dense::finite_vector(ctx->n_k.data(), ctx->n_k.size())) return DVEB_DENSE_NONFINITE;
        if (ybar_count) ctx->ybar.assign(ybar, ybar + ybar_count);
        if (!dveb::dense::finite_vector(ctx->ybar.data(), ctx->ybar.size())) return DVEB_DENSE_NONFINITE;
        if (centered_count) ctx->centered.assign(centered, centered + centered_count);
        if (!dveb::dense::finite_vector(ctx->centered.data(), ctx->centered.size())) return DVEB_DENSE_NONFINITE;
        if (jitter_scale_count) ctx->jitter_scale.assign(jitter_scale, jitter_scale + jitter_scale_count);
        if (!dveb::dense::finite_vector(ctx->jitter_scale.data(), ctx->jitter_scale.size())) return DVEB_DENSE_NONFINITE;
        ctx->epsilon = epsilon;
        if (!std::isfinite(ctx->epsilon)) return DVEB_DENSE_NONFINITE;
        if (!((ctx->offsets.size() == (ctx->patterns + 1)))) return DVEB_DENSE_SHAPE;
        if (!((ctx->centered_offsets.size() == (ctx->patterns + 1)))) return DVEB_DENSE_SHAPE;
        if (!((ctx->n_k.size() == ctx->patterns))) return DVEB_DENSE_SHAPE;
        if (!((ctx->observed_index.size() == static_cast<std::size_t>(ctx->offsets.data()[ctx->patterns])))) return DVEB_DENSE_SHAPE;
        if (!((ctx->ybar.size() == ctx->observed_index.size()))) return DVEB_DENSE_SHAPE;
        if (!((ctx->jitter_scale.size() == ctx->observed_index.size()))) return DVEB_DENSE_SHAPE;
        if (!((ctx->centered.size() == static_cast<std::size_t>(ctx->centered_offsets.data()[ctx->patterns])))) return DVEB_DENSE_SHAPE;
        if (!((dveb::dense::max_segment(ctx->offsets.data(), ctx->offsets.size()) <= ctx->max_local))) return DVEB_DENSE_SHAPE;
        if (!((dveb::dense::all_in_range(ctx->observed_index.data(), ctx->observed_index.size(), ctx->p) == 1))) return DVEB_DENSE_SHAPE;
        if (!((ctx->epsilon > 0.0))) return DVEB_DENSE_SHAPE;
        std::size_t dense_size_lower = 0;
        if (!dveb::dense::checked_product(ctx->p, ctx->p, dense_size_lower)) return DVEB_DENSE_SHAPE;
        ctx->s_lower.resize(dense_size_lower);
        ctx->scratch_bytes += dense_size_lower * sizeof(double);
        std::size_t dense_size_sigma = 0;
        if (!dveb::dense::checked_product(ctx->p, ctx->p, dense_size_sigma)) return DVEB_DENSE_SHAPE;
        ctx->s_sigma.resize(dense_size_sigma);
        ctx->scratch_bytes += dense_size_sigma * sizeof(double);
        const std::size_t dense_size_total_gmu = ctx->p;
        ctx->s_total_gmu.resize(dense_size_total_gmu);
        ctx->scratch_bytes += dense_size_total_gmu * sizeof(double);
        std::size_t dense_size_total_gsigma = 0;
        if (!dveb::dense::checked_product(ctx->p, ctx->p, dense_size_total_gsigma)) return DVEB_DENSE_SHAPE;
        ctx->s_total_gsigma.resize(dense_size_total_gsigma);
        ctx->scratch_bytes += dense_size_total_gsigma * sizeof(double);
        std::size_t dense_size_dlower = 0;
        if (!dveb::dense::checked_product(ctx->p, ctx->p, dense_size_dlower)) return DVEB_DENSE_SHAPE;
        ctx->s_dlower.resize(dense_size_dlower);
        ctx->scratch_bytes += dense_size_dlower * sizeof(double);
        std::size_t dense_size_a = 0;
        if (!dveb::dense::checked_product(ctx->max_local, ctx->max_local, dense_size_a)) return DVEB_DENSE_SHAPE;
        std::size_t dense_size_chol = 0;
        if (!dveb::dense::checked_product(ctx->max_local, ctx->max_local, dense_size_chol)) return DVEB_DENSE_SHAPE;
        std::size_t dense_size_inverse = 0;
        if (!dveb::dense::checked_product(ctx->max_local, ctx->max_local, dense_size_inverse)) return DVEB_DENSE_SHAPE;
        std::size_t dense_size_moment = 0;
        if (!dveb::dense::checked_product(ctx->max_local, ctx->max_local, dense_size_moment)) return DVEB_DENSE_SHAPE;
        std::size_t dense_size_temporary = 0;
        if (!dveb::dense::checked_product(ctx->max_local, ctx->max_local, dense_size_temporary)) return DVEB_DENSE_SHAPE;
        const std::size_t dense_size_delta = ctx->max_local;
        const std::size_t dense_size_reduction_total_gmu = ctx->p;
        std::size_t dense_size_reduction_total_gsigma = 0;
        if (!dveb::dense::checked_product(ctx->p, ctx->p, dense_size_reduction_total_gsigma)) return DVEB_DENSE_SHAPE;
        ctx->workers.resize(max_threads);
        for (auto &ws : ctx->workers) {
            ws.s_a.resize(dense_size_a);
            ctx->scratch_bytes += dense_size_a * sizeof(double);
            ws.s_chol.resize(dense_size_chol);
            ctx->scratch_bytes += dense_size_chol * sizeof(double);
            ws.s_inverse.resize(dense_size_inverse);
            ctx->scratch_bytes += dense_size_inverse * sizeof(double);
            ws.s_moment.resize(dense_size_moment);
            ctx->scratch_bytes += dense_size_moment * sizeof(double);
            ws.s_temporary.resize(dense_size_temporary);
            ctx->scratch_bytes += dense_size_temporary * sizeof(double);
            ws.s_delta.resize(dense_size_delta);
            ctx->scratch_bytes += dense_size_delta * sizeof(double);
            ws.r_total_gmu.resize(dense_size_reduction_total_gmu);
            ctx->scratch_bytes += dense_size_reduction_total_gmu * sizeof(double);
            ws.r_total_gsigma.resize(dense_size_reduction_total_gsigma);
            ctx->scratch_bytes += dense_size_reduction_total_gsigma * sizeof(double);
        }
        *out_context = owned.release();
        return DVEB_DENSE_OK;
    } catch (const std::bad_alloc &) {
        return DVEB_DENSE_ALLOCATION;
    }
}

extern "C" int dveb_dense_value_gradient(
    dveb_dense_context *ctx, const double *theta, std::size_t theta_count,
    double *gradient, std::size_t gradient_count,
    std::size_t threads, int schedule_override,
    double *value_out, int *selected_schedule_out) {
    if (!ctx || !theta || !gradient || !value_out ||
        threads < 1 || threads > ctx->max_threads) return DVEB_DENSE_INVALID;
    if (dveb::dense::overlaps(theta, theta_count * sizeof(double),
                              gradient, gradient_count * sizeof(double))) return DVEB_DENSE_ALIAS;
    if (!dveb::dense::finite_vector(theta, theta_count)) return DVEB_DENSE_NONFINITE;
    if (!((theta_count == (ctx->p + ((ctx->p * (ctx->p + 1)) / 2))))) return DVEB_DENSE_SHAPE;
    if (!((gradient_count == theta_count))) return DVEB_DENSE_SHAPE;
    if (!((ctx->offsets.size() == (ctx->patterns + 1)))) return DVEB_DENSE_SHAPE;
    if (!((ctx->centered_offsets.size() == (ctx->patterns + 1)))) return DVEB_DENSE_SHAPE;
    if (!((ctx->n_k.size() == ctx->patterns))) return DVEB_DENSE_SHAPE;
    if (!((ctx->observed_index.size() == static_cast<std::size_t>(ctx->offsets.data()[ctx->patterns])))) return DVEB_DENSE_SHAPE;
    if (!((ctx->ybar.size() == ctx->observed_index.size()))) return DVEB_DENSE_SHAPE;
    if (!((ctx->jitter_scale.size() == ctx->observed_index.size()))) return DVEB_DENSE_SHAPE;
    if (!((ctx->centered.size() == static_cast<std::size_t>(ctx->centered_offsets.data()[ctx->patterns])))) return DVEB_DENSE_SHAPE;
    if (!((dveb::dense::max_segment(ctx->offsets.data(), ctx->offsets.size()) <= ctx->max_local))) return DVEB_DENSE_SHAPE;
    if (!((dveb::dense::all_in_range(ctx->observed_index.data(), ctx->observed_index.size(), ctx->p) == 1))) return DVEB_DENSE_SHAPE;
    if (!((ctx->epsilon > 0.0))) return DVEB_DENSE_SHAPE;
    int selected = schedule_override;
    if (selected == DVEB_DENSE_SCHEDULE_AUTO) {
        selected = (threads == 1 || (ctx->patterns) < 2 * threads)
            ? DVEB_DENSE_SCHEDULE_SERIAL
            : DVEB_DENSE_SCHEDULE_WORK_ITEM_PARALLEL;
    }
    if (selected != DVEB_DENSE_SCHEDULE_SERIAL &&
        selected != DVEB_DENSE_SCHEDULE_WORK_ITEM_PARALLEL) return DVEB_DENSE_SCHEDULE;
    if (selected_schedule_out) *selected_schedule_out = selected;
    std::atomic<int> dense_status{DVEB_DENSE_OK};
    dveb::dense::fill_matrix(ctx->s_lower.data(), ctx->p, ctx->p, ctx->p, 0.0);
    for (std::size_t i = 0; i < ctx->p; ++i) {
        ctx->s_lower.data()[(i) * (ctx->p) + (i)] = std::exp(theta[(ctx->p + i)]);
    }
    size_t q = (2 * ctx->p);
    for (std::size_t i = 1; i < ctx->p; ++i) {
        for (std::size_t j = 0; j < i; ++j) {
            ctx->s_lower.data()[(i) * (ctx->p) + (j)] = theta[q];
            q = (q + 1);
        }
    }
    dveb::dense::matmul_nt(ctx->s_lower.data(), ctx->p, ctx->s_lower.data(), ctx->p, ctx->s_sigma.data(), ctx->p, ctx->p, ctx->p, ctx->p);
    dveb::dense::fill_vector(ctx->s_total_gmu.data(), ctx->p, 0.0);
    dveb::dense::fill_matrix(ctx->s_total_gsigma.data(), ctx->p, ctx->p, ctx->p, 0.0);
    double value = 0.0;
    for (std::size_t t_ = 0; t_ < threads; ++t_) {
        auto &ws = ctx->workers[t_];
        ws.r_value = 0;
        std::fill(ws.r_total_gmu.begin(), ws.r_total_gmu.end(), 0.0);
        std::fill(ws.r_total_gsigma.begin(), ws.r_total_gsigma.end(), 0.0);
    }
    if (selected == DVEB_DENSE_SCHEDULE_WORK_ITEM_PARALLEL) {
        omp_set_dynamic(0);
        omp_set_num_threads(static_cast<int>(threads));
#pragma omp parallel for schedule(static) num_threads(threads)
        for (std::int64_t dense_k_ = static_cast<std::int64_t>(0);
             dense_k_ < static_cast<std::int64_t>(ctx->patterns); ++dense_k_) {
            const std::size_t k = static_cast<std::size_t>(dense_k_);
            auto &ws = ctx->workers[static_cast<std::size_t>(omp_get_thread_num())];
            if (dense_status.load(std::memory_order_relaxed) != DVEB_DENSE_OK) continue;
            const size_t begin = static_cast<std::size_t>(ctx->offsets.data()[k]);
            const size_t v = static_cast<std::size_t>((ctx->offsets.data()[(k + 1)] - ctx->offsets.data()[k]));
            const size_t cb = static_cast<std::size_t>(ctx->centered_offsets.data()[k]);
            dveb::dense::fill_matrix(ws.s_a.data(), v, v, ctx->max_local, 0.0);
            dveb::dense::fill_matrix(ws.s_moment.data(), v, v, ctx->max_local, 0.0);
            for (std::size_t i = 0; i < v; ++i) {
                const size_t gi = static_cast<std::size_t>(ctx->observed_index.data()[(begin + i)]);
                ws.s_delta.data()[i] = (ctx->ybar.data()[(begin + i)] - theta[gi]);
            }
            for (std::size_t i = 0; i < v; ++i) {
                const size_t gi_matrix = static_cast<std::size_t>(ctx->observed_index.data()[(begin + i)]);
                for (std::size_t j = 0; j < v; ++j) {
                    const size_t gj = static_cast<std::size_t>(ctx->observed_index.data()[(begin + j)]);
                    ws.s_a.data()[(i) * (ctx->max_local) + (j)] = ctx->s_sigma.data()[(gi_matrix) * (ctx->p) + (gj)];
                    ws.s_moment.data()[(i) * (ctx->max_local) + (j)] = (ctx->centered.data()[((cb + (i * v)) + j)] + ((ctx->n_k.data()[k] * ws.s_delta.data()[i]) * ws.s_delta.data()[j]));
                }
                ws.s_a.data()[(i) * (ctx->max_local) + (i)] = (ws.s_a.data()[(i) * (ctx->max_local) + (i)] + (ctx->epsilon * ctx->jitter_scale.data()[(begin + i)]));
            }
            if (!dveb::dense::cholesky(ws.s_a.data(), ctx->max_local, ws.s_chol.data(), ctx->max_local, v)) { dense_status.store(DVEB_DENSE_FACTORIZATION, std::memory_order_relaxed); continue; }
            dveb::dense::inverse_from_cholesky(ws.s_chol.data(), ctx->max_local, ws.s_inverse.data(), ctx->max_local, ws.s_temporary.data(), v);
            for (std::size_t i = 0; i < v; ++i) {
                ws.r_value = (ws.r_value + ((2.0 * ctx->n_k.data()[k]) * std::log(ws.s_chol.data()[(i) * (ctx->max_local) + (i)])));
                for (std::size_t j = 0; j < v; ++j) {
                    ws.r_value = (ws.r_value + (ws.s_inverse.data()[(i) * (ctx->max_local) + (j)] * ws.s_moment.data()[(i) * (ctx->max_local) + (j)]));
                }
            }
            dveb::dense::matmul(ws.s_inverse.data(), ctx->max_local, ws.s_moment.data(), ctx->max_local, ws.s_temporary.data(), ctx->max_local, v, v, v);
            dveb::dense::matmul(ws.s_temporary.data(), ctx->max_local, ws.s_inverse.data(), ctx->max_local, ws.s_a.data(), ctx->max_local, v, v, v);
            for (std::size_t i = 0; i < v; ++i) {
                const size_t gi_grad = static_cast<std::size_t>(ctx->observed_index.data()[(begin + i)]);
                double x = 0.0;
                for (std::size_t j = 0; j < v; ++j) {
                    x = (x + (ws.s_inverse.data()[(i) * (ctx->max_local) + (j)] * ws.s_delta.data()[j]));
                    const size_t gj_grad = static_cast<std::size_t>(ctx->observed_index.data()[(begin + j)]);
                    ws.r_total_gsigma.data()[(gi_grad) * (ctx->p) + (gj_grad)] = ((ws.r_total_gsigma.data()[(gi_grad) * (ctx->p) + (gj_grad)] + (ctx->n_k.data()[k] * ws.s_inverse.data()[(i) * (ctx->max_local) + (j)])) - ws.s_a.data()[(i) * (ctx->max_local) + (j)]);
                }
                ws.r_total_gmu.data()[gi_grad] = (ws.r_total_gmu.data()[gi_grad] - ((2.0 * ctx->n_k.data()[k]) * x));
            }
        }
    } else {
        auto &ws = ctx->workers[0];
        for (std::size_t k = 0; k < ctx->patterns; ++k) {
            const size_t begin = static_cast<std::size_t>(ctx->offsets.data()[k]);
            const size_t v = static_cast<std::size_t>((ctx->offsets.data()[(k + 1)] - ctx->offsets.data()[k]));
            const size_t cb = static_cast<std::size_t>(ctx->centered_offsets.data()[k]);
            dveb::dense::fill_matrix(ws.s_a.data(), v, v, ctx->max_local, 0.0);
            dveb::dense::fill_matrix(ws.s_moment.data(), v, v, ctx->max_local, 0.0);
            for (std::size_t i = 0; i < v; ++i) {
                const size_t gi = static_cast<std::size_t>(ctx->observed_index.data()[(begin + i)]);
                ws.s_delta.data()[i] = (ctx->ybar.data()[(begin + i)] - theta[gi]);
            }
            for (std::size_t i = 0; i < v; ++i) {
                const size_t gi_matrix = static_cast<std::size_t>(ctx->observed_index.data()[(begin + i)]);
                for (std::size_t j = 0; j < v; ++j) {
                    const size_t gj = static_cast<std::size_t>(ctx->observed_index.data()[(begin + j)]);
                    ws.s_a.data()[(i) * (ctx->max_local) + (j)] = ctx->s_sigma.data()[(gi_matrix) * (ctx->p) + (gj)];
                    ws.s_moment.data()[(i) * (ctx->max_local) + (j)] = (ctx->centered.data()[((cb + (i * v)) + j)] + ((ctx->n_k.data()[k] * ws.s_delta.data()[i]) * ws.s_delta.data()[j]));
                }
                ws.s_a.data()[(i) * (ctx->max_local) + (i)] = (ws.s_a.data()[(i) * (ctx->max_local) + (i)] + (ctx->epsilon * ctx->jitter_scale.data()[(begin + i)]));
            }
            if (!dveb::dense::cholesky(ws.s_a.data(), ctx->max_local, ws.s_chol.data(), ctx->max_local, v)) { dense_status.store(DVEB_DENSE_FACTORIZATION, std::memory_order_relaxed); continue; }
            dveb::dense::inverse_from_cholesky(ws.s_chol.data(), ctx->max_local, ws.s_inverse.data(), ctx->max_local, ws.s_temporary.data(), v);
            for (std::size_t i = 0; i < v; ++i) {
                ws.r_value = (ws.r_value + ((2.0 * ctx->n_k.data()[k]) * std::log(ws.s_chol.data()[(i) * (ctx->max_local) + (i)])));
                for (std::size_t j = 0; j < v; ++j) {
                    ws.r_value = (ws.r_value + (ws.s_inverse.data()[(i) * (ctx->max_local) + (j)] * ws.s_moment.data()[(i) * (ctx->max_local) + (j)]));
                }
            }
            dveb::dense::matmul(ws.s_inverse.data(), ctx->max_local, ws.s_moment.data(), ctx->max_local, ws.s_temporary.data(), ctx->max_local, v, v, v);
            dveb::dense::matmul(ws.s_temporary.data(), ctx->max_local, ws.s_inverse.data(), ctx->max_local, ws.s_a.data(), ctx->max_local, v, v, v);
            for (std::size_t i = 0; i < v; ++i) {
                const size_t gi_grad = static_cast<std::size_t>(ctx->observed_index.data()[(begin + i)]);
                double x = 0.0;
                for (std::size_t j = 0; j < v; ++j) {
                    x = (x + (ws.s_inverse.data()[(i) * (ctx->max_local) + (j)] * ws.s_delta.data()[j]));
                    const size_t gj_grad = static_cast<std::size_t>(ctx->observed_index.data()[(begin + j)]);
                    ws.r_total_gsigma.data()[(gi_grad) * (ctx->p) + (gj_grad)] = ((ws.r_total_gsigma.data()[(gi_grad) * (ctx->p) + (gj_grad)] + (ctx->n_k.data()[k] * ws.s_inverse.data()[(i) * (ctx->max_local) + (j)])) - ws.s_a.data()[(i) * (ctx->max_local) + (j)]);
                }
                ws.r_total_gmu.data()[gi_grad] = (ws.r_total_gmu.data()[gi_grad] - ((2.0 * ctx->n_k.data()[k]) * x));
            }
        }
    }
    if (dense_status.load(std::memory_order_relaxed) != DVEB_DENSE_OK) return dense_status.load(std::memory_order_relaxed);
    const std::size_t dense_active_workers = selected == DVEB_DENSE_SCHEDULE_WORK_ITEM_PARALLEL ? threads : 1;
    for (std::size_t t_ = 0; t_ < dense_active_workers; ++t_) {
        const auto &ws = ctx->workers[t_];
        value += ws.r_value;
        for (std::size_t r_ = 0; r_ < ctx->p; ++r_)
            ctx->s_total_gmu.data()[r_] += ws.r_total_gmu[r_];
        for (std::size_t r_ = 0; r_ < ((ctx->p) * (ctx->p)); ++r_)
            ctx->s_total_gsigma.data()[r_] += ws.r_total_gsigma[r_];
    }
    dveb::dense::matmul_sym_left(ctx->s_total_gsigma.data(), ctx->p, ctx->s_lower.data(), ctx->p, ctx->s_dlower.data(), ctx->p, ctx->p);
    for (std::size_t i = 0; i < ctx->p; ++i) {
        gradient[i] = ctx->s_total_gmu.data()[i];
        gradient[(ctx->p + i)] = (ctx->s_dlower.data()[(i) * (ctx->p) + (i)] * ctx->s_lower.data()[(i) * (ctx->p) + (i)]);
    }
    q = (2 * ctx->p);
    for (std::size_t i = 1; i < ctx->p; ++i) {
        for (std::size_t j = 0; j < i; ++j) {
            gradient[q] = ctx->s_dlower.data()[(i) * (ctx->p) + (j)];
            q = (q + 1);
        }
    }
    const double result_value = value;
    if (!std::isfinite(result_value) || !dveb::dense::finite_vector(gradient, gradient_count)) return DVEB_DENSE_NONFINITE;
    *value_out = result_value;
    return DVEB_DENSE_OK;
}

extern "C" void dveb_dense_context_destroy(dveb_dense_context *ctx) { delete ctx; }
extern "C" size_t dveb_dense_context_scratch_bytes(const dveb_dense_context *ctx) {
    return ctx ? ctx->scratch_bytes : 0;
}

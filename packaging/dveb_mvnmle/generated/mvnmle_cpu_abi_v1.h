/* dveb-generated CPU dense ABI v1 — do not edit. */
#ifndef DVEB_DENSE_MVNMLE_CPU_ABI_V1_H
#define DVEB_DENSE_MVNMLE_CPU_ABI_V1_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

#define DVEB_DENSE_ABI_V1 1u
enum {
    DVEB_DENSE_OK = 0,
    DVEB_DENSE_INVALID = 2,
    DVEB_DENSE_NONFINITE = 3,
    DVEB_DENSE_SHAPE = 4,
    DVEB_DENSE_ALIAS = 5,
    DVEB_DENSE_FACTORIZATION = 6,
    DVEB_DENSE_ALLOCATION = 7,
    DVEB_DENSE_SCHEDULE = 8
};
enum {
    DVEB_DENSE_SCHEDULE_AUTO = 0,
    DVEB_DENSE_SCHEDULE_SERIAL = 1,
    DVEB_DENSE_SCHEDULE_WORK_ITEM_PARALLEL = 2
};

typedef struct dveb_dense_context dveb_dense_context;
uint32_t dveb_dense_abi_version(void);
const char *dveb_dense_status_string(int status);
int dveb_dense_context_create(
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
    dveb_dense_context **out_context);
int dveb_dense_value_gradient(
    dveb_dense_context *context,
    const double *theta, size_t theta_count,
    double *gradient, size_t gradient_count,
    size_t threads, int schedule_override,
    double *value_out, int *selected_schedule_out);
void dveb_dense_context_destroy(dveb_dense_context *context);
size_t dveb_dense_context_scratch_bytes(const dveb_dense_context *context);

#ifdef __cplusplus
}
#endif
#endif

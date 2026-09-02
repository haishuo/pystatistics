/* dveb-generated stateful recurrence ABI v1 — do not edit. */
#ifndef DVEB_RECURRENCE_EXACT_ARMA_ABI_V1_H
#define DVEB_RECURRENCE_EXACT_ARMA_ABI_V1_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

#define DVEB_RECURRENCE_ABI_V1 1u
enum {
    DVEB_RECURRENCE_OK = 0,
    DVEB_RECURRENCE_INVALID = 2,
    DVEB_RECURRENCE_SHAPE = 3,
    DVEB_RECURRENCE_ALIAS = 4,
    DVEB_RECURRENCE_ALLOCATION = 5,
    DVEB_RECURRENCE_SCHEDULE = 6,
    DVEB_RECURRENCE_CUDA = 7
};
enum {
    DVEB_RECURRENCE_CPU_AUTO = 0,
    DVEB_RECURRENCE_CPU_SERIAL = 1,
    DVEB_RECURRENCE_CPU_ITEM_PARALLEL = 2
};

typedef struct dveb_recurrence_cpu_context dveb_recurrence_cpu_context;
typedef struct dveb_recurrence_cuda_context dveb_recurrence_cuda_context;

uint32_t dveb_recurrence_abi_version(void);
const char *dveb_recurrence_status_string(int status);
int dveb_recurrence_cpu_context_create(
    size_t max_threads, dveb_recurrence_cpu_context **out_context);
void dveb_recurrence_cpu_context_destroy(dveb_recurrence_cpu_context *context);
size_t dveb_recurrence_cpu_scratch_bytes(const dveb_recurrence_cpu_context *context);
int dveb_recurrence_cpu_run(
    dveb_recurrence_cpu_context *context,
    const double *z,
    size_t z_rows,
    size_t z_cols,
    const double *phi,
    size_t phi_rows,
    size_t phi_cols,
    const double *loading,
    size_t loading_rows,
    size_t loading_cols,
    double *nll,
    size_t nll_count,
    double *sigma2,
    size_t sigma2_count,
    int64_t *status,
    size_t status_count,
    size_t threads, int schedule_override, int *selected_schedule_out);

int dveb_recurrence_cuda_context_create(
    int device, size_t max_f64_elements, size_t max_i64_elements,
    const int *block_by_state, size_t block_by_state_count,
    dveb_recurrence_cuda_context **out_context);
void dveb_recurrence_cuda_context_destroy(dveb_recurrence_cuda_context *context);
size_t dveb_recurrence_cuda_payload_bytes(const dveb_recurrence_cuda_context *context);
int dveb_recurrence_cuda_run_device(
    dveb_recurrence_cuda_context *context,
    const double *z,
    size_t z_rows,
    size_t z_cols,
    const double *phi,
    size_t phi_rows,
    size_t phi_cols,
    const double *loading,
    size_t loading_rows,
    size_t loading_cols,
    double *nll,
    size_t nll_count,
    double *sigma2,
    size_t sigma2_count,
    int64_t *status,
    size_t status_count,
    int block_override, void *stream, int *selected_block_out);
/* run_device launches asynchronously and never allocates, transfers, or synchronizes.
   All array arguments must be naturally aligned pointers on context->device. */
int dveb_recurrence_cuda_run_host(
    dveb_recurrence_cuda_context *context,
    const double *z,
    size_t z_rows,
    size_t z_cols,
    const double *phi,
    size_t phi_rows,
    size_t phi_cols,
    const double *loading,
    size_t loading_rows,
    size_t loading_cols,
    double *nll,
    size_t nll_count,
    double *sigma2,
    size_t sigma2_count,
    int64_t *status,
    size_t status_count,
    int block_override, int *selected_block_out);

#ifdef __cplusplus
}
#endif
#endif

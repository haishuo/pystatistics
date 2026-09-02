#ifndef PYSTATISTICS_DVEB_ARMA_NATIVE_H
#define PYSTATISTICS_DVEB_ARMA_NATIVE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum arma_native_error {
    ARMA_NATIVE_OK = 0,
    ARMA_NATIVE_INVALID_ARGUMENT = 2,
    ARMA_NATIVE_ALLOCATION_FAILED = 3,
    ARMA_NATIVE_CUDA_ERROR = 4,
};

const char* arma_native_version(void);

typedef struct arma_cpu_context arma_cpu_context;

arma_cpu_context* arma_cpu_create(int max_r, int max_threads);
void arma_cpu_destroy(arma_cpu_context* context);
int arma_cpu_evaluate(
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
);

typedef struct arma_cuda_context arma_cuda_context;

arma_cuda_context* arma_cuda_create(int64_t k, int64_t n, int r, int proposals);
void arma_cuda_destroy(arma_cuda_context* context);
int arma_cuda_upload_base(
    arma_cuda_context* context,
    const double* z,
    const double* phi,
    const double* loading
);
int arma_cuda_upload_trace(
    arma_cuda_context* context,
    const double* phi_trace,
    const double* loading_trace
);
int arma_cuda_launch(arma_cuda_context* context, int proposal, int block_size);
int arma_cuda_synchronize(arma_cuda_context* context);
int arma_cuda_download(
    arma_cuda_context* context,
    int proposal,
    double* nll,
    double* sigma2,
    uint8_t* status
);
int arma_cuda_device_info(int* compute_major, int* compute_minor, int* sm_count);
const char* arma_cuda_last_error(void);

#ifdef __cplusplus
}
#endif

#endif

#include "arma_native.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <new>

namespace {

constexpr double kKappa = 1.0e6;
constexpr double kPenalty = 1.0e18;
constexpr double kStationaryRtol = 1.0e-13;
constexpr int kMaxDoublings = 60;

thread_local char last_error[512] = "no CUDA error";

int record(cudaError_t error, const char* operation) {
    if (error == cudaSuccess) {
        return ARMA_NATIVE_OK;
    }
    std::snprintf(last_error, sizeof(last_error), "%s: %s", operation, cudaGetErrorString(error));
    return ARMA_NATIVE_CUDA_ERROR;
}

__global__ void arma_kernel(
    const double* z,
    const double* phi_global,
    const double* loading_global,
    int64_t k,
    int64_t n,
    int r,
    double* nll,
    double* sigma2,
    uint8_t* status
) {
    const int64_t row = static_cast<int64_t>(blockIdx.x);
    if (row >= k) {
        return;
    }
    const int lane = threadIdx.x;
    const int rr = r * r;
    extern __shared__ double storage[];
    double* s = storage;
    double* a_matrix = s + rr;
    double* product = a_matrix + rr;
    double* update = product + rr;
    double* next_a = update + rr;
    double* phi = next_a + rr;
    double* loading = phi + r;
    double* state = loading + r;
    double* filtered_state = state + r;
    double* gain = filtered_state + r;
    double* scratch_vector = gain + r;

    __shared__ int live;
    __shared__ int converged;
    __shared__ double innovation;
    __shared__ double observation_variance;
    __shared__ double inverse_variance;
    __shared__ double accumulated_sse;
    __shared__ double accumulated_log_f;

    for (int index = lane; index < r; index += blockDim.x) {
        phi[index] = phi_global[row * r + index];
        loading[index] = loading_global[row * r + index];
        state[index] = 0.0;
    }
    for (int index = lane; index < rr; index += blockDim.x) {
        const int i = index / r;
        const int j = index - i * r;
        s[index] = loading_global[row * r + i] * loading_global[row * r + j];
        a_matrix[index] = 0.0;
    }
    __syncthreads();
    for (int index = lane; index < r; index += blockDim.x) {
        a_matrix[index * r] = phi[index];
    }
    for (int index = lane; index + 1 < r; index += blockDim.x) {
        a_matrix[index * r + index + 1] = 1.0;
    }
    if (lane == 0) {
        live = 1;
        converged = 0;
    }
    __syncthreads();

    for (int iteration = 0; iteration < kMaxDoublings; ++iteration) {
        for (int index = lane; index < rr; index += blockDim.x) {
            const int i = index / r;
            const int j = index - i * r;
            double value = 0.0;
            for (int inner = 0; inner < r; ++inner) {
                value += a_matrix[i * r + inner] * s[inner * r + j];
            }
            product[index] = value;
        }
        __syncthreads();
        for (int index = lane; index < rr; index += blockDim.x) {
            const int i = index / r;
            const int j = index - i * r;
            double value = 0.0;
            for (int inner = 0; inner < r; ++inner) {
                value += product[i * r + inner] * a_matrix[j * r + inner];
            }
            update[index] = value;
            if (!isfinite(value)) {
                atomicExch(&live, 0);
            }
        }
        __syncthreads();
        if (!live) {
            break;
        }
        for (int index = lane; index < rr; index += blockDim.x) {
            s[index] += update[index];
        }
        __syncthreads();
        if (lane == 0) {
            double update_max = 0.0;
            double scale = 0.0;
            for (int index = 0; index < rr; ++index) {
                update_max = fmax(update_max, fabs(update[index]));
                scale = fmax(scale, fabs(s[index]));
            }
            if (!isfinite(scale)) {
                live = 0;
            } else if (update_max <= kStationaryRtol * fmax(1.0, scale)) {
                converged = 1;
            }
        }
        __syncthreads();
        if (!live || converged) {
            break;
        }
        for (int index = lane; index < rr; index += blockDim.x) {
            const int i = index / r;
            const int j = index - i * r;
            double value = 0.0;
            for (int inner = 0; inner < r; ++inner) {
                value += a_matrix[i * r + inner] * a_matrix[inner * r + j];
            }
            next_a[index] = value;
            if (!isfinite(value)) {
                atomicExch(&live, 0);
            }
        }
        __syncthreads();
        if (!live) {
            break;
        }
        for (int index = lane; index < rr; index += blockDim.x) {
            a_matrix[index] = next_a[index];
        }
        __syncthreads();
    }

    if (converged && live) {
        for (int index = lane; index < rr; index += blockDim.x) {
            const int i = index / r;
            const int j = index - i * r;
            product[index] = 0.5 * (s[index] + s[j * r + i]);
        }
    } else {
        for (int index = lane; index < rr; index += blockDim.x) {
            const int i = index / r;
            const int j = index - i * r;
            product[index] = i == j ? kKappa : 0.0;
        }
    }
    if (lane == 0) {
        live = 1;
        accumulated_sse = 0.0;
        accumulated_log_f = 0.0;
    }
    __syncthreads();

    const double* row_z = z + row * n;
    for (int64_t t = 0; t < n; ++t) {
        if (lane == 0) {
            innovation = row_z[t] - state[0];
            observation_variance = product[0];
            if (!isfinite(observation_variance) || observation_variance <= 0.0) {
                live = 0;
            } else {
                inverse_variance = 1.0 / observation_variance;
            }
        }
        __syncthreads();
        if (!live) {
            break;
        }
        for (int i = lane; i < r; i += blockDim.x) {
            gain[i] = product[i * r] * inverse_variance;
            filtered_state[i] = state[i] + gain[i] * innovation;
        }
        __syncthreads();
        for (int index = lane; index < rr; index += blockDim.x) {
            const int i = index / r;
            const int j = index - i * r;
            a_matrix[index] = product[index] - gain[i] * product[j];
        }
        __syncthreads();
        for (int i = lane; i < r; i += blockDim.x) {
            double value = phi[i] * filtered_state[0];
            if (i + 1 < r) {
                value += filtered_state[i + 1];
            }
            state[i] = value;
        }
        for (int j = lane; j < r; j += blockDim.x) {
            scratch_vector[j] = phi[0] * a_matrix[j];
            if (r > 1) {
                scratch_vector[j] += a_matrix[r + j];
            }
        }
        __syncthreads();
        for (int index = lane; index < rr; index += blockDim.x) {
            const int i = index / r;
            const int j = index - i * r;
            double first_column;
            if (i == 0) {
                first_column = scratch_vector[0];
            } else {
                first_column = phi[i] * a_matrix[0];
                if (i + 1 < r) {
                    first_column += a_matrix[(i + 1) * r];
                }
            }
            double value = first_column * phi[j];
            if (j + 1 < r) {
                if (i == 0) {
                    value += scratch_vector[j + 1];
                } else {
                    double shifted = phi[i] * a_matrix[j + 1];
                    if (i + 1 < r) {
                        shifted += a_matrix[(i + 1) * r + j + 1];
                    }
                    value += shifted;
                }
            }
            value += loading[i] * loading[j];
            product[index] = value;
            if (!isfinite(value)) {
                atomicExch(&live, 0);
            }
        }
        __syncthreads();
        if (!live) {
            break;
        }
        if (lane == 0) {
            accumulated_sse +=
                innovation * innovation / observation_variance;
            accumulated_log_f += log(observation_variance);
        }
        __syncthreads();
    }

    if (lane == 0) {
        const double row_sigma2 = accumulated_sse / static_cast<double>(n);
        const double row_nll =
            0.5 * static_cast<double>(n) * log(2.0 * acos(-1.0) * row_sigma2) +
            0.5 * accumulated_log_f + 0.5 * static_cast<double>(n);
        const bool ok = live && isfinite(accumulated_sse) && accumulated_sse > 0.0 &&
                        isfinite(row_sigma2) && isfinite(row_nll);
        nll[row] = ok ? row_nll : kPenalty;
        sigma2[row] = ok ? row_sigma2 : 1.0;
        status[row] = ok ? 1 : 0;
    }
}

std::size_t shared_bytes(int r) {
    return static_cast<std::size_t>(5 * r * r + 6 * r) * sizeof(double);
}

}  // namespace

struct arma_cuda_context {
    int64_t k;
    int64_t n;
    int r;
    int proposals;
    double* z;
    double* phi;
    double* loading;
    double* phi_trace;
    double* loading_trace;
    double* nll;
    double* sigma2;
    uint8_t* status;
};

extern "C" arma_cuda_context* arma_cuda_create(int64_t k, int64_t n, int r, int proposals) {
    if (k < 1 || n < 1 || r < 1 || r > 25 || proposals < 0 || proposals > 100) {
        return nullptr;
    }
    arma_cuda_context* context = new (std::nothrow) arma_cuda_context{};
    if (context == nullptr) {
        return nullptr;
    }
    context->k = k;
    context->n = n;
    context->r = r;
    context->proposals = proposals;
    const std::size_t z_bytes = static_cast<std::size_t>(k) * n * sizeof(double);
    const std::size_t parameter_bytes = static_cast<std::size_t>(k) * r * sizeof(double);
    const std::size_t trace_bytes = static_cast<std::size_t>(proposals) * parameter_bytes;
    const std::size_t output_count = static_cast<std::size_t>(proposals + 1) * k;
    cudaError_t error = cudaMalloc(&context->z, z_bytes);
    if (error == cudaSuccess) error = cudaMalloc(&context->phi, parameter_bytes);
    if (error == cudaSuccess) error = cudaMalloc(&context->loading, parameter_bytes);
    if (error == cudaSuccess && proposals > 0) error = cudaMalloc(&context->phi_trace, trace_bytes);
    if (error == cudaSuccess && proposals > 0) error = cudaMalloc(&context->loading_trace, trace_bytes);
    if (error == cudaSuccess) error = cudaMalloc(&context->nll, output_count * sizeof(double));
    if (error == cudaSuccess) error = cudaMalloc(&context->sigma2, output_count * sizeof(double));
    if (error == cudaSuccess) error = cudaMalloc(&context->status, output_count * sizeof(uint8_t));
    if (error != cudaSuccess) {
        record(error, "arma_cuda_create");
        arma_cuda_destroy(context);
        return nullptr;
    }
    return context;
}

extern "C" void arma_cuda_destroy(arma_cuda_context* context) {
    if (context == nullptr) return;
    cudaFree(context->z);
    cudaFree(context->phi);
    cudaFree(context->loading);
    cudaFree(context->phi_trace);
    cudaFree(context->loading_trace);
    cudaFree(context->nll);
    cudaFree(context->sigma2);
    cudaFree(context->status);
    delete context;
}

extern "C" int arma_cuda_upload_base(
    arma_cuda_context* context,
    const double* z,
    const double* phi,
    const double* loading
) {
    if (context == nullptr || z == nullptr || phi == nullptr || loading == nullptr) {
        return ARMA_NATIVE_INVALID_ARGUMENT;
    }
    const std::size_t z_bytes = static_cast<std::size_t>(context->k) * context->n * sizeof(double);
    const std::size_t parameter_bytes =
        static_cast<std::size_t>(context->k) * context->r * sizeof(double);
    int code = record(cudaMemcpy(context->z, z, z_bytes, cudaMemcpyHostToDevice), "upload z");
    if (code == ARMA_NATIVE_OK) {
        code = record(cudaMemcpy(context->phi, phi, parameter_bytes, cudaMemcpyHostToDevice), "upload phi");
    }
    if (code == ARMA_NATIVE_OK) {
        code = record(
            cudaMemcpy(context->loading, loading, parameter_bytes, cudaMemcpyHostToDevice),
            "upload loading"
        );
    }
    return code;
}

extern "C" int arma_cuda_upload_trace(
    arma_cuda_context* context,
    const double* phi_trace,
    const double* loading_trace
) {
    if (context == nullptr || context->proposals < 1 || phi_trace == nullptr ||
        loading_trace == nullptr) {
        return ARMA_NATIVE_INVALID_ARGUMENT;
    }
    const std::size_t bytes = static_cast<std::size_t>(context->proposals) * context->k *
                              context->r * sizeof(double);
    int code = record(
        cudaMemcpy(context->phi_trace, phi_trace, bytes, cudaMemcpyHostToDevice),
        "upload phi trace"
    );
    if (code == ARMA_NATIVE_OK) {
        code = record(
            cudaMemcpy(context->loading_trace, loading_trace, bytes, cudaMemcpyHostToDevice),
            "upload loading trace"
        );
    }
    return code;
}

extern "C" int arma_cuda_launch(arma_cuda_context* context, int proposal, int block_size) {
    if (context == nullptr || (block_size != 32 && block_size != 64 && block_size != 128 &&
                               block_size != 256) ||
        proposal < -1 || proposal >= context->proposals) {
        return ARMA_NATIVE_INVALID_ARGUMENT;
    }
    const std::size_t parameter_stride = static_cast<std::size_t>(context->k) * context->r;
    const double* phi = proposal < 0 ? context->phi : context->phi_trace + proposal * parameter_stride;
    const double* loading =
        proposal < 0 ? context->loading : context->loading_trace + proposal * parameter_stride;
    const std::size_t output_offset = static_cast<std::size_t>(proposal + 1) * context->k;
    arma_kernel<<<static_cast<unsigned>(context->k), block_size, shared_bytes(context->r)>>>(
        context->z,
        phi,
        loading,
        context->k,
        context->n,
        context->r,
        context->nll + output_offset,
        context->sigma2 + output_offset,
        context->status + output_offset
    );
    return record(cudaGetLastError(), "arma kernel launch");
}

extern "C" int arma_cuda_synchronize(arma_cuda_context* context) {
    if (context == nullptr) return ARMA_NATIVE_INVALID_ARGUMENT;
    return record(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
}

extern "C" int arma_cuda_download(
    arma_cuda_context* context,
    int proposal,
    double* nll,
    double* sigma2,
    uint8_t* status
) {
    if (context == nullptr || nll == nullptr || sigma2 == nullptr || status == nullptr ||
        proposal < -1 || proposal >= context->proposals) {
        return ARMA_NATIVE_INVALID_ARGUMENT;
    }
    const std::size_t offset = static_cast<std::size_t>(proposal + 1) * context->k;
    const std::size_t double_bytes = static_cast<std::size_t>(context->k) * sizeof(double);
    const std::size_t status_bytes = static_cast<std::size_t>(context->k) * sizeof(uint8_t);
    int code = record(
        cudaMemcpy(nll, context->nll + offset, double_bytes, cudaMemcpyDeviceToHost),
        "download nll"
    );
    if (code == ARMA_NATIVE_OK) {
        code = record(
            cudaMemcpy(sigma2, context->sigma2 + offset, double_bytes, cudaMemcpyDeviceToHost),
            "download sigma2"
        );
    }
    if (code == ARMA_NATIVE_OK) {
        code = record(
            cudaMemcpy(status, context->status + offset, status_bytes, cudaMemcpyDeviceToHost),
            "download status"
        );
    }
    return code;
}

extern "C" int arma_cuda_device_info(int* compute_major, int* compute_minor, int* sm_count) {
    if (compute_major == nullptr || compute_minor == nullptr || sm_count == nullptr) {
        return ARMA_NATIVE_INVALID_ARGUMENT;
    }
    int device = 0;
    int code = record(cudaGetDevice(&device), "cudaGetDevice");
    cudaDeviceProp properties{};
    if (code == ARMA_NATIVE_OK) {
        code = record(cudaGetDeviceProperties(&properties, device), "cudaGetDeviceProperties");
    }
    if (code == ARMA_NATIVE_OK) {
        *compute_major = properties.major;
        *compute_minor = properties.minor;
        *sm_count = properties.multiProcessorCount;
    }
    return code;
}

extern "C" const char* arma_cuda_last_error(void) {
    return last_error;
}

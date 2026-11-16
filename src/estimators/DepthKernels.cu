#include "estimators/DepthKernels.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace DepthKernels {

namespace {

constexpr int THREADS = 256;

__global__ void collectValidKernel(
    const float* input,
    float* output,
    unsigned int* validCount,
    std::size_t count
) {
    const std::size_t idx =
        blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= count) {
        return;
    }

    const float value = input[idx];

    if (isfinite(value) && value > 0.0f) {
        const unsigned int position =
            atomicAdd(validCount, 1u);

        output[position] = value;
    }
}

}

float calculateNearestPercentile(
    const float* depth,
    std::size_t count,
    float percentile
) {
    if (!depth || count == 0) {
        return -1.0f;
    }

    percentile = std::clamp(
        percentile,
        0.0f,
        1.0f
    );

    float* deviceInput = nullptr;
    float* deviceOutput = nullptr;
    unsigned int* deviceCount = nullptr;

    const std::size_t bytes =
        count * sizeof(float);

    cudaError_t status;

    status = cudaMalloc(
        &deviceInput,
        bytes
    );

    if (status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(status));
    }

    status = cudaMalloc(
        &deviceOutput,
        bytes
    );

    if (status != cudaSuccess) {
        cudaFree(deviceInput);
        throw std::runtime_error(cudaGetErrorString(status));
    }

    status = cudaMalloc(
        &deviceCount,
        sizeof(unsigned int)
    );

    if (status != cudaSuccess) {
        cudaFree(deviceInput);
        cudaFree(deviceOutput);
        throw std::runtime_error(cudaGetErrorString(status));
    }

    unsigned int zero = 0;

    try {
        status = cudaMemcpy(
            deviceInput,
            depth,
            bytes,
            cudaMemcpyHostToDevice
        );

        if (status != cudaSuccess) {
            throw std::runtime_error(cudaGetErrorString(status));
        }

        status = cudaMemcpy(
            deviceCount,
            &zero,
            sizeof(unsigned int),
            cudaMemcpyHostToDevice
        );

        if (status != cudaSuccess) {
            throw std::runtime_error(cudaGetErrorString(status));
        }

        const int blocks =
            static_cast<int>(
                (count + THREADS - 1) / THREADS
            );

        collectValidKernel<<<blocks, THREADS>>>(
            deviceInput,
            deviceOutput,
            deviceCount,
            count
        );

        status = cudaGetLastError();

        if (status != cudaSuccess) {
            throw std::runtime_error(cudaGetErrorString(status));
        }

        unsigned int validCount = 0;

        status = cudaMemcpy(
            &validCount,
            deviceCount,
            sizeof(unsigned int),
            cudaMemcpyDeviceToHost
        );

        if (status != cudaSuccess) {
            throw std::runtime_error(cudaGetErrorString(status));
        }

        if (validCount == 0) {
            cudaFree(deviceInput);
            cudaFree(deviceOutput);
            cudaFree(deviceCount);
            return -1.0f;
        }

        std::vector<float> values(validCount);

        status = cudaMemcpy(
            values.data(),
            deviceOutput,
            validCount * sizeof(float),
            cudaMemcpyDeviceToHost
        );

        if (status != cudaSuccess) {
            throw std::runtime_error(cudaGetErrorString(status));
        }

        const std::size_t index =
            static_cast<std::size_t>(
                percentile * (validCount - 1)
            );

        std::nth_element(
            values.begin(),
            values.begin() + index,
            values.end()
        );

        const float result = values[index];

        cudaFree(deviceInput);
        cudaFree(deviceOutput);
        cudaFree(deviceCount);

        return result;
    }
    catch (...) {
        cudaFree(deviceInput);
        cudaFree(deviceOutput);
        cudaFree(deviceCount);
        throw;
    }
}

}

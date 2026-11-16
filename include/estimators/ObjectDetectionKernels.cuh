#pragma once

#include <cuda_runtime.h>
#include <cstdint>

void launchDetectionPreprocess(
    const unsigned char* source,
    int sourceWidth,
    int sourceHeight,
    std::size_t sourcePitch,

    float* destination,

    int resizedWidth,
    int resizedHeight,
    int padLeft,
    int padTop,

    cudaStream_t stream = nullptr
);

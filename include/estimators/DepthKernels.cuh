#pragma once

#include <cstddef>

namespace DepthKernels {

float calculateNearestPercentile(
    const float* depth,
    std::size_t count,
    float percentile
);

}

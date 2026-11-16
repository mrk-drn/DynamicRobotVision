#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cstddef>
#include <cstdint>


__global__ void motionReductionKernel(
    const unsigned char* previousPoints,
    size_t previousStep,

    const unsigned char* currentPoints,
    size_t currentStep,

    const unsigned char* status,
    size_t statusStep,

    int rows,
    int cols,

    float threshold,

    int* result
)
{
    constexpr int BLOCK_SIZE = 256;

    __shared__ float displacementSum[BLOCK_SIZE];
    __shared__ int validCount[BLOCK_SIZE];

    const int tid = threadIdx.x;

    const int totalPoints = rows * cols;

    float localSum = 0.0f;
    int localCount = 0;

    // Each thread processes multiple points.

    for (
        int i = blockIdx.x * blockDim.x + tid;
        i < totalPoints;
        i += blockDim.x * gridDim.x
    )
    {
        const int row = i / cols;
        const int col = i % cols;

        const unsigned char* statusRow =
            status + row * statusStep;

        if (statusRow[col] == 0)
            continue;

        const float2* previousRow =
            reinterpret_cast<const float2*>(
                previousPoints + row * previousStep
            );

        const float2* currentRow =
            reinterpret_cast<const float2*>(
                currentPoints + row * currentStep
            );

        const float2 p1 = previousRow[col];
        const float2 p2 = currentRow[col];

        const float dx = p2.x - p1.x;
        const float dy = p2.y - p1.y;

        localSum += sqrtf(dx * dx + dy * dy);

        ++localCount;
    }

    displacementSum[tid] = localSum;
    validCount[tid] = localCount;

    __syncthreads();

    // Parallel reduction.

    for (int stride = BLOCK_SIZE / 2;
         stride > 0;
         stride >>= 1)
    {
        if (tid < stride)
        {
            displacementSum[tid] +=
                displacementSum[tid + stride];

            validCount[tid] +=
                validCount[tid + stride];
        }

        __syncthreads();
    }

    // Only one thread writes the result.

    if (tid == 0)
    {
        const int count = validCount[0];

        if (count == 0)
        {
            *result = 0;
        }
        else
        {
            const float average =
                displacementSum[0] / count;

            *result = average > threshold ? 1 : 0;
        }
    }
}


// Host launcher

extern "C"
int launchMotionReduction(
    const void* previousPoints,
    size_t previousStep,

    const void* currentPoints,
    size_t currentStep,

    const void* status,
    size_t statusStep,

    int rows,
    int cols,

    float threshold,

    int* result
)
{
    constexpr int THREADS = 256;

    constexpr int BLOCKS = 1;

    motionReductionKernel<<<BLOCKS, THREADS>>>(
        static_cast<const unsigned char*>(previousPoints),
        previousStep,

        static_cast<const unsigned char*>(currentPoints),
        currentStep,

        static_cast<const unsigned char*>(status),
        statusStep,

        rows,
        cols,

        threshold,

        result
    );

    return static_cast<int>(
        cudaGetLastError()
    );
}
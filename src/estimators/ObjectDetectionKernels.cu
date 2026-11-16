#include "estimators/ObjectDetectionKernels.cuh"

#include <cuda_runtime.h>
#include <algorithm>

namespace {

constexpr int INPUT_WIDTH = 768;
constexpr int INPUT_HEIGHT = 768;

constexpr int BLOCK_X = 32;
constexpr int BLOCK_Y = 8;

__device__ __forceinline__
float clampFloat(float value, float low, float high)
{
    return fminf(fmaxf(value, low), high);
}

__device__ __forceinline__
float sampleChannel(
    const unsigned char* source,
    std::size_t pitch,
    int width,
    int height,
    float x,
    float y,
    int channel
) {
    x = clampFloat(x, 0.0f, static_cast<float>(width - 1));
    y = clampFloat(y, 0.0f, static_cast<float>(height - 1));

    const int x0 = static_cast<int>(floorf(x));
    const int y0 = static_cast<int>(floorf(y));

    const int x1 = min(x0 + 1, width - 1);
    const int y1 = min(y0 + 1, height - 1);

    const float dx = x - x0;
    const float dy = y - y0;

    const unsigned char* row0 =
        source + static_cast<std::size_t>(y0) * pitch;

    const unsigned char* row1 =
        source + static_cast<std::size_t>(y1) * pitch;

    const float p00 = row0[x0 * 3 + channel];
    const float p01 = row0[x1 * 3 + channel];
    const float p10 = row1[x0 * 3 + channel];
    const float p11 = row1[x1 * 3 + channel];

    const float top = p00 + dx * (p01 - p00);
    const float bottom = p10 + dx * (p11 - p10);

    return top + dy * (bottom - top);
}

__global__
void detectionPreprocessKernel(
    const unsigned char* source,
    int sourceWidth,
    int sourceHeight,
    std::size_t sourcePitch,

    float* destination,

    int resizedWidth,
    int resizedHeight,
    int padLeft,
    int padTop
) {
    const int x =
        blockIdx.x * blockDim.x + threadIdx.x;

    const int y =
        blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= INPUT_WIDTH || y >= INPUT_HEIGHT) {
        return;
    }

    const int index = y * INPUT_WIDTH + x;

    const int planeSize = INPUT_WIDTH * INPUT_HEIGHT;

    if (
        x < padLeft ||
        x >= padLeft + resizedWidth ||
        y < padTop ||
        y >= padTop + resizedHeight
    ) {
        const float padding = 114.0f / 255.0f;

        destination[index] = padding;
        destination[planeSize + index] = padding;
        destination[2 * planeSize + index] = padding;

        return;
    }

    // Half-pixel coordinates for bilinear resizing.

    const float sourceX =
        (static_cast<float>(x - padLeft) + 0.5f) *
        sourceWidth / resizedWidth - 0.5f;

    const float sourceY =
        (static_cast<float>(y - padTop) + 0.5f) *
        sourceHeight / resizedHeight - 0.5f;

    // Source image is BGR.
    // Model input is RGB, planar and normalized.

    const float blue = sampleChannel(
        source,
        sourcePitch,
        sourceWidth,
        sourceHeight,
        sourceX,
        sourceY,
        0
    );

    const float green = sampleChannel(
        source,
        sourcePitch,
        sourceWidth,
        sourceHeight,
        sourceX,
        sourceY,
        1
    );

    const float red = sampleChannel(
        source,
        sourcePitch,
        sourceWidth,
        sourceHeight,
        sourceX,
        sourceY,
        2
    );

    destination[index] = red * (1.0f / 255.0f);

    destination[planeSize + index] =
        green * (1.0f / 255.0f);

    destination[2 * planeSize + index] =
        blue * (1.0f / 255.0f);
}

}

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

    cudaStream_t stream
) {
    const dim3 block(
        BLOCK_X,
        BLOCK_Y
    );

    const dim3 grid(
        (INPUT_WIDTH + BLOCK_X - 1) / BLOCK_X,
        (INPUT_HEIGHT + BLOCK_Y - 1) / BLOCK_Y
    );

    detectionPreprocessKernel<<<grid, block, 0, stream>>>(
        source,
        sourceWidth,
        sourceHeight,
        sourcePitch,

        destination,

        resizedWidth,
        resizedHeight,
        padLeft,
        padTop
    );
}

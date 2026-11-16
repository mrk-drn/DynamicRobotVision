
#include "visualization/PointCloudKernels.cuh"
#include "visualization/PointCloudConfig.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace
{

using namespace PointCloudConfig;

constexpr unsigned long long EMPTY_PIXEL =
    0xFFFFFFFFFFFFFFFFULL;

void checkCuda(
    cudaError_t error,
    const char* operation
)
{
    if (error != cudaSuccess)
    {
        throw std::runtime_error(
            std::string(operation) +
            ": " +
            cudaGetErrorString(error)
        );
    }
}

__device__ __forceinline__
unsigned int orderedFloat(float value)
{
    unsigned int bits = __float_as_uint(value);

    return bits ^
        ((bits & 0x80000000u)
            ? 0xFFFFFFFFu
            : 0x80000000u);
}

__global__ void clearBufferKernel(
    unsigned long long* buffer,
    int total
)
{
    const int index =
        blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= total)
        return;

    buffer[index] = EMPTY_PIXEL;
}

__global__ void renderPointsKernel(
    const float* depth,
    const unsigned char* pointColors,
    unsigned long long* framebuffer,

    int width,
    int height,

    int step,
    int sampledWidth,
    int sampledHeight,

    float fx,
    float fy,
    float cx,
    float cy,

    float cosYaw,
    float sinYaw,
    float cosPitch,
    float sinPitch
)
{
    const int index =
        blockIdx.x * blockDim.x + threadIdx.x;

    const int total =
        sampledWidth * sampledHeight;

    if (index >= total)
        return;

    const int u =
        (index % sampledWidth) * step;

    const int v =
        (index / sampledWidth) * step;

    const int pixel =
        v * width + u;

    const int colorIndex = pixel * 3;

    const unsigned int blue =
        pointColors[colorIndex];

    const unsigned int green =
        pointColors[colorIndex + 1];

    const unsigned int red =
        pointColors[colorIndex + 2];

    // Black means background.
    if (
        blue == 0 &&
        green == 0 &&
        red == 0
    )
    {
        return;
    }

    const float rawDepth = depth[pixel];

    if (!isfinite(rawDepth) || rawDepth <= 0.0f)
        return;

    const float distance = rawDepth * 1000.0f;

    const float x =
        (static_cast<float>(u) - cx) *
        distance / fx;

    const float y = distance;

    const float z =
        -(static_cast<float>(v) - cy) *
        distance / fy;

    if (
        !isfinite(x) ||
        !isfinite(y) ||
        !isfinite(z)
    )
        return;

    const float rotatedX =
        x * cosYaw - y * sinYaw;

    const float rotatedY =
        x * sinYaw + y * cosYaw;

    const float rotatedZ =
        rotatedY * sinPitch + z * cosPitch;

    const float cameraDepth =
        rotatedY * cosPitch - z * sinPitch;

    const float denominator =
        CAMERA_FOCAL_LENGTH +
        (cameraDepth + CAMERA_DISTANCE) *
        PROJECTION_SCALE;

    if (!isfinite(denominator) || denominator <= 1.0f)
        return;

    const float perspective =
        CAMERA_FOCAL_LENGTH / denominator;

    const float screenX =
        RENDER_WIDTH * 0.5f +
        rotatedX * PROJECTION_SCALE * perspective;

    const float screenY =
        RENDER_HEIGHT * 0.5f -
        rotatedZ * PROJECTION_SCALE * perspective;

    if (!isfinite(screenX) || !isfinite(screenY))
        return;

    const int px = __float2int_rz(screenX);
    const int py = __float2int_rz(screenY);

    if (
        px < 0 || px >= RENDER_WIDTH ||
        py < 0 || py >= RENDER_HEIGHT
    )
        return;

    const unsigned int color =
        blue | (green << 8) | (red << 16);

    const unsigned int depthKey =
        orderedFloat(cameraDepth);

    const unsigned long long packed =
        (static_cast<unsigned long long>(depthKey) << 32) |
        color;

    #pragma unroll
    for (int dy = -POINT_RADIUS; dy <= POINT_RADIUS; ++dy)
    {
        #pragma unroll
        for (int dx = -POINT_RADIUS; dx <= POINT_RADIUS; ++dx)
        {
            const int sx = px + dx;
            const int sy = py + dy;

            if (
                sx < 0 || sx >= RENDER_WIDTH ||
                sy < 0 || sy >= RENDER_HEIGHT
            )
                continue;

            const int screenIndex =
                sy * RENDER_WIDTH + sx;

            atomicMin(
                &framebuffer[screenIndex],
                packed
            );
        }
    }
}

__global__ void resolveKernel(
    const unsigned long long* framebuffer,
    uchar3* output,
    int total
)
{
    const int index =
        blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= total)
        return;

    const unsigned long long packed =
        framebuffer[index];

    if (packed == EMPTY_PIXEL)
    {
        output[index] = make_uchar3(
            BACKGROUND_BLUE,
            BACKGROUND_GREEN,
            BACKGROUND_RED
        );

        return;
    }

    const unsigned int color =
        static_cast<unsigned int>(packed);

    output[index] = make_uchar3(
        static_cast<unsigned char>(color & 0xFF),
        static_cast<unsigned char>((color >> 8) & 0xFF),
        static_cast<unsigned char>((color >> 16) & 0xFF)
    );
}

}

struct CudaPointCloudBackend::Impl
{
    float* d_depth = nullptr;

    unsigned char* d_pointColors = nullptr;

    unsigned long long* d_framebuffer = nullptr;

    uchar3* d_output = nullptr;

    cudaStream_t stream = nullptr;

    size_t depthCapacity = 0;
    size_t colorCapacity = 0;

    Impl()
    {
        checkCuda(
            cudaStreamCreateWithFlags(
                &stream,
                cudaStreamNonBlocking
            ),
            "Create CUDA stream"
        );

        constexpr size_t pixels =
            RENDER_WIDTH * RENDER_HEIGHT;

        checkCuda(
            cudaMalloc(
                &d_framebuffer,
                pixels * sizeof(unsigned long long)
            ),
            "Allocate framebuffer"
        );

        checkCuda(
            cudaMalloc(
                &d_output,
                pixels * sizeof(uchar3)
            ),
            "Allocate output"
        );
    }

    void ensureDepthCapacity(size_t count)
    {
        if (count <= depthCapacity)
            return;

        if (d_depth)
            checkCuda(cudaFree(d_depth), "Free depth");

        checkCuda(
            cudaMalloc(
                &d_depth,
                count * sizeof(float)
            ),
            "Allocate depth"
        );

        depthCapacity = count;
    }

    void ensureColorCapacity(size_t count)
    {
        if (count <= colorCapacity)
            return;

        if (d_pointColors)
        {
            checkCuda(
                cudaFree(d_pointColors),
                "Free point colours"
            );
        }

        checkCuda(
            cudaMalloc(
                &d_pointColors,
                count
            ),
            "Allocate point colours"
        );

        colorCapacity = count;
    }

    ~Impl()
    {
        if (stream)
            cudaStreamSynchronize(stream);

        if (d_depth)
            cudaFree(d_depth);

        if (d_pointColors)
            cudaFree(d_pointColors);

        if (d_framebuffer)
            cudaFree(d_framebuffer);

        if (d_output)
            cudaFree(d_output);

        if (stream)
            cudaStreamDestroy(stream);
    }
};

CudaPointCloudBackend::CudaPointCloudBackend()
    : impl_(std::make_unique<Impl>())
{
}

CudaPointCloudBackend::~CudaPointCloudBackend() = default;

void CudaPointCloudBackend::render(
    const cv::Mat& depth,
    const cv::Mat& pointColors,
    int samplingStep,
    float fx,
    float fy,
    float cx,
    float cy,
    cv::Mat& output
)
{
    if (depth.empty() || depth.type() != CV_32FC1)
        throw std::invalid_argument(
            "Depth must be CV_32FC1"
        );

    if (
        pointColors.empty() ||
        pointColors.type() != CV_8UC3 ||
        pointColors.size() != depth.size()
    )
    {
        throw std::invalid_argument(
            "Point colours must be matching CV_8UC3"
        );
    }

    if (samplingStep < 1)
        throw std::invalid_argument(
            "Sampling step must be at least 1"
        );

    if (
        !std::isfinite(fx) ||
        !std::isfinite(fy) ||
        fx <= 0.0f ||
        fy <= 0.0f
    )
        throw std::invalid_argument(
            "Invalid focal lengths"
        );

    const int width = depth.cols;
    const int height = depth.rows;

    const int sampledWidth =
        (width + samplingStep - 1) / samplingStep;

    const int sampledHeight =
        (height + samplingStep - 1) / samplingStep;

    const size_t total =
        static_cast<size_t>(sampledWidth) *
        sampledHeight;

    const size_t pixelCount =
        static_cast<size_t>(width) * height;

    const size_t colorBytes =
        pixelCount * 3;

    impl_->ensureDepthCapacity(pixelCount);
    impl_->ensureColorCapacity(colorBytes);

    checkCuda(
        cudaMemcpy2DAsync(
            impl_->d_depth,
            width * sizeof(float),
            depth.ptr<float>(),
            depth.step,
            width * sizeof(float),
            height,
            cudaMemcpyHostToDevice,
            impl_->stream
        ),
        "Upload depth"
    );

    checkCuda(
        cudaMemcpy2DAsync(
            impl_->d_pointColors,
            width * 3,
            pointColors.data,
            pointColors.step,
            width * 3,
            height,
            cudaMemcpyHostToDevice,
            impl_->stream
        ),
        "Upload point colours"
    );

    constexpr int THREADS = 256;

    const int pixelCountOutput =
        RENDER_WIDTH * RENDER_HEIGHT;

    const int clearBlocks =
        (pixelCountOutput + THREADS - 1) / THREADS;

    const int pointBlocks =
        static_cast<int>(
            (total + THREADS - 1) / THREADS
        );

    clearBufferKernel<<<
        clearBlocks,
        THREADS,
        0,
        impl_->stream
    >>>(
        impl_->d_framebuffer,
        pixelCountOutput
    );

    checkCuda(
        cudaGetLastError(),
        "Clear framebuffer"
    );

    renderPointsKernel<<<
        pointBlocks,
        THREADS,
        0,
        impl_->stream
    >>>(
        impl_->d_depth,
        impl_->d_pointColors,
        impl_->d_framebuffer,
        width,
        height,
        samplingStep,
        sampledWidth,
        sampledHeight,
        fx,
        fy,
        cx,
        cy,
        std::cos(CAMERA_YAW),
        std::sin(CAMERA_YAW),
        std::cos(CAMERA_PITCH),
        std::sin(CAMERA_PITCH)
    );

    checkCuda(
        cudaGetLastError(),
        "Render points"
    );

    const int resolveBlocks =
        (pixelCountOutput + THREADS - 1) / THREADS;

    resolveKernel<<<
        resolveBlocks,
        THREADS,
        0,
        impl_->stream
    >>>(
        impl_->d_framebuffer,
        impl_->d_output,
        pixelCountOutput
    );

    checkCuda(
        cudaGetLastError(),
        "Resolve framebuffer"
    );

    output.create(
        RENDER_HEIGHT,
        RENDER_WIDTH,
        CV_8UC3
    );

    checkCuda(
        cudaMemcpy2DAsync(
            output.data,
            output.step,
            impl_->d_output,
            RENDER_WIDTH * sizeof(uchar3),
            RENDER_WIDTH * sizeof(uchar3),
            RENDER_HEIGHT,
            cudaMemcpyDeviceToHost,
            impl_->stream
        ),
        "Download visualization"
    );

    checkCuda(
        cudaStreamSynchronize(impl_->stream),
        "Synchronize rendering"
    );
}
